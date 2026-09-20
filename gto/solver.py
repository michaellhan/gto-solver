"""Heads-up postflop solver: range-vs-range discounted CFR over a fixed bet tree.

Units are big blinds (floats). Player 0 = OOP, player 1 = IP.

Speed tricks (pure Python + numpy):
* vectors are compressed to the combos that are actually in either range
* a showdown is a single matvec with a precomputed antisymmetric matrix
* the subtrees below a chance node (one per turn or river card) are structurally
  identical, so they are stored once and traversed once with a batch axis: every
  regret/strategy array is (B, actions, combos). That makes enumerating every
  river from the turn cheap.
* streets beyond `depth` are replaced by equity leaves that average the
  showdown matrices over the remaining runouts
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
np.seterr(divide="ignore", over="ignore", invalid="ignore")
from .cards import ALL_COMBOS, COMBO_INDEX
from .evaluator import eval_board

N_FULL = 1326
MEMBER_FULL = np.zeros((52, N_FULL))
for _i, (_a, _b) in enumerate(ALL_COMBOS):
    MEMBER_FULL[_a, _i] = 1.0
    MEMBER_FULL[_b, _i] = 1.0
CA_FULL = np.array([a for a, b in ALL_COMBOS])
CB_FULL = np.array([b for a, b in ALL_COMBOS])


@dataclass
class Config:
    bet_sizes: Dict[str, List[float]] = field(default_factory=lambda: {
        "flop": [0.33, 0.75], "turn": [0.5, 1.0], "river": [0.5, 1.0]})
    raise_sizes: Dict[str, List[float]] = field(default_factory=lambda: {
        "flop": [0.5, 1.0], "turn": [0.5, 1.0], "river": [0.5, 1.0]})   # fractions of pot-after-call
    max_raises: int = 2               # bet, raise, re-raise; then call/fold/all-in
    allin_spr: float = 2.5            # offer all-in when stack/pot <= this
    depth: int = 2                    # streets with full decisions from the root street
    n_turns: int = 6                  # sampled turns for full-decision turn subtrees (from flop)
    n_rivers: Optional[int] = 12      # rivers for full-decision river subtrees from the turn (None = all)
    n_rivers_from_flop: int = 8       # rivers per turn for full-decision river subtrees when the root is the flop
    leaf_turns: int = 6               # sampled turns for equity leaves reached from the flop
    leaf_rivers: Optional[int] = None # rivers for equity leaves when solving from the turn (None = all)
    leaf_rivers_flop: int = 16        # rivers per turn card for equity leaves when solving from the flop
    seed: int = 7

    def copy(self) -> "Config":
        return Config(bet_sizes={k: list(v) for k, v in self.bet_sizes.items()},
                      raise_sizes={k: list(v) for k, v in self.raise_sizes.items()},
                      max_raises=self.max_raises, allin_spr=self.allin_spr, depth=self.depth,
                      n_turns=self.n_turns, n_rivers=self.n_rivers, n_rivers_from_flop=self.n_rivers_from_flop,
                      leaf_turns=self.leaf_turns, leaf_rivers=self.leaf_rivers, leaf_rivers_flop=self.leaf_rivers_flop,
                      seed=self.seed)

    def sizes_key(self) -> tuple:
        return (tuple((k, tuple(v)) for k, v in sorted(self.bet_sizes.items())),
                tuple((k, tuple(v)) for k, v in sorted(self.raise_sizes.items())), self.depth)


def street_of(board_len: int) -> str:
    return {3: "flop", 4: "turn", 5: "river"}[board_len]


class Ctx:
    """Per-solve compressed combo space and cached showdown matrices."""

    def __init__(self, active: np.ndarray, cfg: Config):
        self.active = active
        self.n = len(active)
        self.member = MEMBER_FULL[:, active]
        self.ca = CA_FULL[active]
        self.cb = CB_FULL[active]
        self.compat = 1.0 - np.clip(self.member.T @ self.member, 0, 1)
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._M: Dict[tuple, np.ndarray] = {}
        self._leaf: Dict[tuple, np.ndarray] = {}
        self._stacks: Dict[tuple, np.ndarray] = {}
        self._turn_picks: Dict[tuple, list] = {}
        self.active_list = [int(i) for i in active]
        self.root_len = 5

    def strength(self, board: tuple) -> np.ndarray:
        return eval_board(tuple(sorted(board)), self.active_list)

    def showdown_matrix(self, board: tuple) -> np.ndarray:
        """M[i, j] = +1 if opp combo j is weaker and compatible with i, -1 if stronger."""
        key = tuple(sorted(board))
        M = self._M.get(key)
        if M is None:
            s = self.strength(key).astype(np.float64)
            M = np.sign(s[:, None] - s[None, :]) * self.compat
            blocked = s < 0
            M[blocked, :] = 0.0
            M[:, blocked] = 0.0
            M = M.astype(np.float32)
            self._M[key] = M
        return M

    def stacked(self, kind: str, boards, dead: set) -> np.ndarray:
        """(B, N, N) matrices for a list of boards, shared by every terminal with the same boards."""
        key = (kind, tuple(boards))
        M = self._stacks.get(key)
        if M is None:
            fn = self.showdown_matrix if kind == "showdown" else (lambda b: self.leaf_matrix(b, dead))
            M = np.stack([fn(b) for b in boards])
            self._stacks[key] = M
        return M

    def turn_picks(self, flop: tuple, dead: set, k: int) -> list:
        key = tuple(sorted(flop))
        picks = self._turn_picks.get(key)
        avail = [c for c in range(52) if c not in dead and c not in flop]
        if picks is None:
            picks = avail if k >= len(avail) else sorted(self.rng.sample(avail, k))
            self._turn_picks[key] = picks
        elif len(picks) < k:
            extra = [c for c in avail if c not in picks]
            picks = sorted(picks + self.rng.sample(extra, min(k - len(picks), len(extra))))
            self._turn_picks[key] = picks
        return picks[:k] if len(picks) > k else picks

    def leaf_matrix(self, board: tuple, dead: set) -> np.ndarray:
        """Average showdown matrix over the remaining runout(s) from `board`."""
        key = tuple(sorted(board))
        L = self._leaf.get(key)
        if L is not None:
            return L
        if len(board) == 5:
            L = self.showdown_matrix(board)
        else:
            avail = [c for c in range(52) if c not in dead and c not in board]
            if len(board) == 3:
                picks = self.turn_picks(board, dead, self.cfg.leaf_turns)
            else:
                lr = self.cfg.leaf_rivers_flop if self.root_len == 3 else self.cfg.leaf_rivers
                k = len(avail) if lr is None else min(lr, len(avail))
                picks = avail if k >= len(avail) else sorted(self.rng.sample(avail, k))
            L = np.zeros((self.n, self.n), dtype=np.float32)
            for c in picks:
                L += self.leaf_matrix(board + (c,), dead)
            L /= len(picks)
        self._leaf[key] = L
        return L


# ----------------------------------------------------------------- tree nodes
class Terminal:
    __slots__ = ("boards", "kind", "folder", "inv", "M")

    def __init__(self, boards, kind, inv, folder=None):
        self.boards, self.kind, self.inv, self.folder = boards, kind, inv, folder
        self.M = None


class Chance:
    """Parent batch B, K cards per parent element; the child has batch B*K."""
    __slots__ = ("boards", "K", "masks", "child")

    def __init__(self, boards, K, masks, child):
        self.boards, self.K, self.masks, self.child = boards, K, masks, child


class Decision:
    __slots__ = ("boards", "player", "names", "children", "regret", "ssum", "pot_now", "bets", "B")

    def __init__(self, boards, player, names, children, pot_now, bets, n):
        self.boards, self.player, self.names, self.children = boards, player, names, children
        self.pot_now, self.bets = pot_now, bets
        self.B = len(boards)
        A = len(names)
        self.regret = np.zeros((self.B, A, n))
        self.ssum = np.zeros((self.B, A, n))

    @property
    def board(self):
        return self.boards[0]

    def strategy(self) -> np.ndarray:
        pos = np.maximum(self.regret, 0.0)
        tot = pos.sum(axis=1, keepdims=True)
        return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / len(self.names))

    def avg_strategy(self) -> np.ndarray:
        """(B, A, N) average strategy."""
        tot = self.ssum.sum(axis=1, keepdims=True)
        return np.where(tot > 0, self.ssum / np.where(tot > 0, tot, 1.0), 1.0 / len(self.names))

    def avg_strategy0(self) -> np.ndarray:
        """(A, N) average strategy of the first (or only) batch element."""
        return self.avg_strategy()[0]


# ----------------------------------------------------------------- builder
class TreeBuilder:
    def __init__(self, ctx: Ctx, dead: Sequence[int]):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.rng = random.Random(self.cfg.seed + 1)
        self.dead = set(dead)
        self.n_nodes = 0
        self.root_street_len = 0

    def build(self, board, pot, stack, bets=(0.0, 0.0), to_act=0, n_raises=0, inv=None):
        board = tuple(board)
        self.root_street_len = len(board)
        if inv is None:
            inv = (pot / 2 + bets[0], pot / 2 + bets[1])
        checked = (to_act == 1 and bets[0] == bets[1] and bets[0] == 0 and n_raises == 0)
        return self._decision([board], pot, stack, bets, to_act, n_raises, inv, checked=checked)

    def _street_over(self, boards, pot, stack, bets, inv, allin):
        new_pot = pot + bets[0] + bets[1]
        new_stack = stack - max(bets)
        if len(boards[0]) == 5:
            return Terminal(boards, "showdown", inv)
        if allin or new_stack <= 1e-9:
            return Terminal(boards, "leaf", inv)
        streets_used = len(boards[0]) - self.root_street_len + 1
        if streets_used >= self.cfg.depth:
            return Terminal(boards, "leaf", inv)
        return self._chance(boards, new_pot, new_stack, inv)

    def _next_cards(self, board, k: Optional[int]):
        if len(board) == 3:
            return self.ctx.turn_picks(board, self.dead, self.cfg.n_turns)
        avail = [c for c in range(52) if c not in self.dead and c not in board]
        if k is None or k >= len(avail):
            return avail
        return sorted(self.rng.sample(avail, k))

    def _chance(self, boards, pot, stack, inv):
        cfg = self.cfg
        if len(boards[0]) == 3:
            k = cfg.n_turns
        else:
            k = cfg.n_rivers_from_flop if self.root_street_len == 3 else cfg.n_rivers
        child_boards, cards = [], []
        for b in boards:
            cs = self._next_cards(b, k)
            if len(cs) > k if k is not None else False:
                cs = cs[:k]
            cards.append(cs)
        K = min(len(cs) for cs in cards)
        masks = np.ones((len(boards) * K, self.ctx.n))
        for bi, b in enumerate(boards):
            for ki in range(K):
                c = cards[bi][ki]
                child_boards.append(b + (c,))
                masks[bi * K + ki] = 1.0 - self.ctx.member[c]
        child = self._decision(child_boards, pot, stack, (0.0, 0.0), 0, 0, inv, False)
        self.n_nodes += 1
        return Chance(boards, K, masks, child)

    def _decision(self, boards, pot, stack, bets, p, n_raises, inv, checked):
        cfg = self.cfg
        street = street_of(len(boards[0]))
        o = 1 - p
        b_me, b_op = bets[p], bets[o]
        pot_total = pot + bets[0] + bets[1]
        remaining = stack - b_me
        names, kids = [], []

        def with_bet(new_me):
            nb = list(bets); nb[p] = new_me
            ninv = list(inv); ninv[p] += new_me - b_me
            return tuple(nb), tuple(ninv)

        if b_op <= b_me + 1e-9:  # nothing to call
            names.append("check")
            if checked or p == 1:
                kids.append(self._street_over(boards, pot, stack, bets, inv, allin=False))
            else:
                kids.append(self._decision(boards, pot, stack, bets, o, n_raises, inv, checked=True))
            offered_allin = False
            for frac in cfg.bet_sizes[street]:
                amt = max(round(frac * pot_total * 2) / 2, 1.0)
                if amt >= remaining - 1e-9:
                    if not offered_allin and remaining > 0:
                        offered_allin = True
                        nb, ninv = with_bet(b_me + remaining)
                        names.append("all-in")
                        kids.append(self._decision(boards, pot, stack, nb, o, n_raises + 1, ninv, False))
                    continue
                name = f"bet {int(round(frac * 100))}%"
                if name in names:
                    continue
                nb, ninv = with_bet(b_me + amt)
                names.append(name)
                kids.append(self._decision(boards, pot, stack, nb, o, n_raises + 1, ninv, False))
            if not offered_allin and remaining > 0 and remaining <= cfg.allin_spr * pot_total:
                nb, ninv = with_bet(b_me + remaining)
                names.append("all-in")
                kids.append(self._decision(boards, pot, stack, nb, o, n_raises + 1, ninv, False))
        else:  # facing a bet
            names.append("fold")
            kids.append(Terminal(boards, "fold", inv, folder=p))
            call_amt = min(b_op - b_me, remaining)
            nb, ninv = with_bet(b_me + call_amt)
            names.append("call")
            allin = b_op >= stack - 1e-9
            kids.append(self._street_over(boards, pot, stack, nb, ninv, allin=allin))
            if remaining > call_amt + 1e-9:
                pot_after_call = pot_total + call_amt
                offered = set()
                capped = False
                if n_raises < cfg.max_raises:
                    for frac in cfg.raise_sizes[street]:
                        raise_to = b_op + round(frac * pot_after_call * 2) / 2
                        raise_to = max(raise_to, 2 * b_op - b_me)
                        if raise_to >= stack - 1e-9:
                            capped = True
                            continue
                        if raise_to in offered:
                            continue
                        offered.add(raise_to)
                        nb, ninv = with_bet(raise_to)
                        names.append(f"raise to {raise_to:g}")
                        kids.append(self._decision(boards, pot, stack, nb, o, n_raises + 1, ninv, False))
                if capped or not offered or remaining <= cfg.allin_spr * pot_total:
                    nb, ninv = with_bet(stack)
                    names.append("all-in")
                    kids.append(self._decision(boards, pot, stack, nb, o, n_raises + 1, ninv, False))
        self.n_nodes += 1
        return Decision(boards, p, names, kids, pot_total, bets, self.ctx.n)


# ----------------------------------------------------------------- solver
class Solver:
    def __init__(self, ctx: Ctx, root: Decision, dead: set):
        self.ctx, self.root, self.dead = ctx, root, dead
        self.iters = 0
        self.alpha, self.beta, self.gamma = 1.5, 0.0, 2.0   # DCFR parameters
        self.prune_eps = 0.0          # skip subtrees the opponent reaches with negligible mass
        self._nodes: List[Decision] = []
        self._collect(root)

    def _collect(self, node):
        if isinstance(node, Decision):
            self._nodes.append(node)
            for c in node.children:
                self._collect(c)
        elif isinstance(node, Chance):
            self._collect(node.child)

    def compat_sum(self, ro):
        """ro: (B, N) -> (B, N) mass of opponent combos sharing no card with each combo."""
        cs = ro @ self.ctx.member.T            # (B, 52)
        return ro.sum(axis=1, keepdims=True) - cs[:, self.ctx.ca] - cs[:, self.ctx.cb] + ro

    def _terminal_M(self, node: Terminal):
        if node.M is None:
            node.M = self.ctx.stacked(node.kind, node.boards, self.dead)
        return node.M

    def _cfr(self, node, p, r0, r1, update, use_avg):
        """r0, r1: (B, N) reaches. Returns (B, N) utility for player p."""
        ro = r1 if p == 0 else r0
        if update and ro.sum() < self.prune_eps:
            return np.zeros_like(r0)
        if isinstance(node, Terminal):
            if node.kind == "fold":
                if node.folder == p:
                    return -node.inv[p] * self.compat_sum(ro)
                return node.inv[1 - p] * self.compat_sum(ro)
            M = self._terminal_M(node)
            return node.inv[p] * np.matmul(M, ro[:, :, None].astype(np.float32))[:, :, 0].astype(np.float64)
        if isinstance(node, Chance):
            B, K, n = r0.shape[0], node.K, self.ctx.n
            m = node.masks
            r0c = (r0[:, None, :] * m.reshape(B, K, n)).reshape(B * K, n)
            r1c = (r1[:, None, :] * m.reshape(B, K, n)).reshape(B * K, n)
            u = self._cfr(node.child, p, r0c, r1c, update, use_avg)
            return (u * m).reshape(B, K, n).mean(axis=1)
        strat = node.avg_strategy() if use_avg else node.strategy()     # (B, A, N)
        A = len(node.names)
        if node.player == p:
            utils = np.empty_like(strat)
            for a, child in enumerate(node.children):
                utils[:, a, :] = self._cfr(child, p, r0, r1, update, use_avg)
            node_util = (strat * utils).sum(axis=1)                     # (B, N)
            if update:
                rp = r0 if p == 0 else r1
                node.regret += utils - node_util[:, None, :]
                node.ssum += rp[:, None, :] * strat
            return node_util
        util = np.zeros_like(r0)
        for a, child in enumerate(node.children):
            if p == 0:
                util += self._cfr(child, p, r0, r1 * strat[:, a, :], update, use_avg)
            else:
                util += self._cfr(child, p, r0 * strat[:, a, :], r1, update, use_avg)
        return util

    def _discount(self):
        t = float(self.iters)
        pos_f = (t ** self.alpha) / (t ** self.alpha + 1)
        neg_f = 0.5 if self.beta == 0 else (t ** self.beta) / (t ** self.beta + 1)
        s_f = (t / (t + 1)) ** self.gamma
        for n in self._nodes:
            r = n.regret
            r *= np.where(r > 0, pos_f, neg_f)
            n.ssum *= s_f

    def run(self, r0, r1, time_budget=3.0, max_iters=1000, min_iters=10):
        """r0, r1: (N,) root reaches. Adds iterations until the budget is spent."""
        r0b, r1b = r0[None, :], r1[None, :]
        self.prune_eps = 1e-6 * min(r0.sum(), r1.sum())
        t0 = time.time()
        start = self.iters
        while self.iters < max_iters:
            for p in (0, 1):
                self._cfr(self.root, p, r0b, r1b, update=True, use_avg=False)
            self.iters += 1
            self._discount()
            if self.iters - start >= min_iters and time.time() - t0 > time_budget:
                break
        return self.iters, time.time() - t0

    def node_action_evs(self, node: Decision, r0, r1):
        """EV per action for the node's player, chips relative to folding now, both players
        on their average strategies. r0, r1: (N,) reaches at the node."""
        p = node.player
        ro = (r1 if p == 0 else r0)[None, :]
        cs = self.compat_sum(ro)[0]
        inv_p = _inv_of(node, p)
        evs = np.zeros((len(node.names), self.ctx.n))
        for a, child in enumerate(node.children):
            u = self._cfr(child, p, r0[None, :], r1[None, :], update=False, use_avg=True)[0]
            evs[a] = np.where(cs > 1e-12, u / np.where(cs > 1e-12, cs, 1.0), 0.0) + inv_p
        return evs


def _inv_of(node, p: int) -> float:
    n = node
    while not isinstance(n, Terminal):
        n = n.child if isinstance(n, Chance) else n.children[0]
    return n.inv[p]


@dataclass
class Solution:
    root: Decision
    names: List[str]
    strategy: np.ndarray    # (A, 1326) average strategy of the root player
    evs: np.ndarray         # (A, 1326) EV per action in bb relative to folding now
    iters: int
    seconds: float
    n_nodes: int
    ctx: Ctx
    r0: np.ndarray          # compressed reaches at root
    r1: np.ndarray
    solver: Solver

    def for_combo(self, cb: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
        i = COMBO_INDEX[(min(cb), max(cb))]
        return {n: (float(self.strategy[a, i]), float(self.evs[a, i])) for a, n in enumerate(self.names)}

    def range_strategy(self, r_full: np.ndarray) -> Dict[str, float]:
        w = r_full / max(r_full.sum(), 1e-12)
        return {n: float((self.strategy[a] * w).sum()) for a, n in enumerate(self.names)}

    def extend(self, time_budget: float, max_iters: int = 1000):
        """Keep solving from the current regrets for another `time_budget` seconds."""
        if self.iters >= max_iters:
            return self
        _, secs = self.solver.run(self.r0, self.r1, time_budget=time_budget, max_iters=max_iters, min_iters=1)
        self.iters = self.solver.iters
        self.seconds += secs
        self._refresh()
        return self

    def _refresh(self):
        A = len(self.names)
        active = self.ctx.active
        evs = self.solver.node_action_evs(self.root, self.r0, self.r1)
        strat = self.root.avg_strategy0()
        self.strategy = np.zeros((A, N_FULL)); self.evs = np.zeros((A, N_FULL))
        self.strategy[:, active] = strat; self.evs[:, active] = evs


def expand(ctx: Ctx, v: np.ndarray) -> np.ndarray:
    out = np.zeros(N_FULL)
    out[ctx.active] = v
    return out


def solve(board: Sequence[int], r_oop: np.ndarray, r_ip: np.ndarray, pot: float, stack: float,
          bets=(0.0, 0.0), to_act: int = 0, n_raises: int = 0, cfg: Optional[Config] = None,
          time_budget: float = 3.0, max_iters: int = 1000, dead: Sequence[int] = ()) -> Solution:
    cfg = cfg or Config()
    board = tuple(board)
    bm = np.ones(N_FULL)
    for c in list(board) + list(dead):
        bm *= 1.0 - MEMBER_FULL[c]
    r0f, r1f = r_oop * bm, r_ip * bm
    active = np.nonzero((r0f > 0) | (r1f > 0))[0]
    ctx = Ctx(active, cfg)
    ctx.root_len = len(board)
    dead_set = set(dead) | set(board)
    tb = TreeBuilder(ctx, dead=list(dead_set))
    root = tb.build(board, pot, stack, bets, to_act, n_raises)
    r0, r1 = r0f[active], r1f[active]
    s = Solver(ctx, root, dead_set)
    iters, secs = s.run(r0, r1, time_budget=time_budget, max_iters=max_iters)
    sol = Solution(root, root.names, None, None, iters, secs, tb.n_nodes, ctx, r0, r1, s)
    sol._refresh()
    return sol


def match_action(names: List[str], wanted: str) -> int:
    """Map a real action ('bet 12', 'raise to 30', 'check', 'call', 'fold', 'all-in', 'bet 50%')
    to the nearest tree action index."""
    w = wanted.strip().lower()
    if w in names:
        return names.index(w)
    kind = w.split()[0]
    if kind in ("check", "call", "fold") and kind in names:
        return names.index(kind)
    if kind in ("all-in", "allin", "shove", "jam"):
        if "all-in" in names:
            return names.index("all-in")
        cands = [i for i, n in enumerate(names) if n.startswith(("bet", "raise"))]
        return cands[-1] if cands else len(names) - 1
    try:
        val = float(w.replace("%", "").split()[-1])
    except ValueError:
        val = None
    cands = [i for i, n in enumerate(names) if n.startswith(kind)]
    if not cands:
        cands = [i for i, n in enumerate(names) if n.startswith(("bet", "raise", "all-in"))]
    if not cands:
        return 0
    if val is None or len(cands) == 1:
        return cands[0]

    def num(n):
        try:
            return float(n.replace("%", "").split()[-1])
        except ValueError:
            return 1e9
    return min(cands, key=lambda i: abs(num(names[i]) - val))
