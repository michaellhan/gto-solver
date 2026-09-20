"""Heads-up postflop solver: range-vs-range discounted CFR over a fixed bet tree.

Units are big blinds (floats). Player 0 = OOP, player 1 = IP.

Speed tricks (pure Python + numpy):
* vectors are compressed to the combos that are actually in either range
* a showdown is a single matvec with a precomputed antisymmetric matrix
* streets beyond `depth` are replaced by equity leaves that average the
  showdown matrices over every remaining runout (exact rivers from the turn;
  sampled turns from the flop)
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
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
    raise_sizes: List[float] = field(default_factory=lambda: [0.5, 1.0])   # fractions of pot-after-call
    max_raises: int = 2               # bet, raise, re-raise; then call/fold/all-in
    allin_spr: float = 2.5            # offer all-in when stack/pot <= this
    depth: int = 2                    # streets with full decisions from the root street
    n_turns: int = 4                  # sampled turns for full-decision turn subtrees (from flop)
    n_rivers: int = 6                 # sampled rivers for full-decision river subtrees (from turn)
    leaf_turns: int = 6               # sampled turns for equity leaves reached from the flop
    leaf_rivers: Optional[int] = None # rivers for equity leaves when solving from the turn (None = all)
    leaf_rivers_flop: int = 16        # rivers per turn card for equity leaves when solving from the flop
    seed: int = 7


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
        # compat[i, j] = 1 if combos share no card
        self.compat = 1.0 - np.clip(self.member.T @ self.member, 0, 1)
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._M: Dict[tuple, np.ndarray] = {}
        self._leaf: Dict[tuple, np.ndarray] = {}
        self._turn_picks: Dict[tuple, list] = {}
        self.active_list = [int(i) for i in active]
        self.root_len = 5

    def strength(self, board: tuple) -> np.ndarray:
        return eval_board(tuple(sorted(board)), self.active_list)

    def turn_picks(self, flop: tuple, dead: set, k: int) -> list:
        """Sampled turn cards for this flop, shared by decision subtrees and equity leaves."""
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
            self._M[key] = M
        return M

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
            L = np.zeros((self.n, self.n))
            for c in picks:
                L += self.leaf_matrix(board + (c,), dead)
            L /= len(picks)
        self._leaf[key] = L
        return L


# ----------------------------------------------------------------- tree nodes
class Terminal:
    __slots__ = ("board", "kind", "folder", "inv", "M")

    def __init__(self, board, kind, inv, folder=None):
        self.board, self.kind, self.inv, self.folder = board, kind, inv, folder
        self.M = None


class Chance:
    __slots__ = ("board", "cards", "children")

    def __init__(self, board, cards, children):
        self.board, self.cards, self.children = board, cards, children


class Decision:
    __slots__ = ("board", "player", "names", "children", "regret", "ssum", "pot_now", "bets")

    def __init__(self, board, player, names, children, pot_now, bets, n):
        self.board, self.player, self.names, self.children = board, player, names, children
        self.pot_now, self.bets = pot_now, bets
        A = len(names)
        self.regret = np.zeros((A, n))
        self.ssum = np.zeros((A, n))

    def strategy(self) -> np.ndarray:
        pos = np.maximum(self.regret, 0.0)
        tot = pos.sum(axis=0)
        A = len(self.names)
        return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / A)

    def avg_strategy(self) -> np.ndarray:
        tot = self.ssum.sum(axis=0)
        A = len(self.names)
        return np.where(tot > 0, self.ssum / np.where(tot > 0, tot, 1.0), 1.0 / A)


# ----------------------------------------------------------------- builder
class TreeBuilder:
    def __init__(self, ctx: Ctx, dead: Sequence[int]):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.rng = random.Random(self.cfg.seed + 1)
        self.dead = set(dead)
        self.n_nodes = 0
        self.root_street_len = 0

    def _next_cards(self, board, k: Optional[int]):
        avail = [c for c in range(52) if c not in self.dead and c not in board]
        if k is None or k >= len(avail):
            return avail
        return sorted(self.rng.sample(avail, k))

    def build(self, board, pot, stack, bets=(0.0, 0.0), to_act=0, n_raises=0, inv=None):
        """pot: chips in the middle before this street's bets. stack: effective stack at
        the start of this street. bets: chips committed this street by (oop, ip)."""
        board = tuple(board)
        self.root_street_len = len(board)
        if inv is None:
            inv = (pot / 2 + bets[0], pot / 2 + bets[1])
        checked = (to_act == 1 and bets[0] == bets[1] and bets[0] == 0 and n_raises == 0)
        return self._decision(board, pot, stack, bets, to_act, n_raises, inv, checked=checked)

    def _street_over(self, board, pot, stack, bets, inv, allin):
        new_pot = pot + bets[0] + bets[1]
        new_stack = stack - max(bets)
        if len(board) == 5:
            return Terminal(board, "showdown", inv)
        if allin or new_stack <= 1e-9:
            return Terminal(board, "leaf", inv)
        streets_used = len(board) - self.root_street_len + 1
        if streets_used >= self.cfg.depth:
            return Terminal(board, "leaf", inv)
        return self._chance(board, new_pot, new_stack, inv)

    def _chance(self, board, pot, stack, inv):
        if len(board) == 3:
            cards = self.ctx.turn_picks(board, self.dead, self.cfg.n_turns)
        else:
            cards = self._next_cards(board, self.cfg.n_rivers)
        kids = [self._decision(board + (c,), pot, stack, (0.0, 0.0), 0, 0, inv, False) for c in cards]
        self.n_nodes += 1
        return Chance(board, cards, kids)

    def _decision(self, board, pot, stack, bets, p, n_raises, inv, checked):
        cfg = self.cfg
        street = street_of(len(board))
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
                kids.append(self._street_over(board, pot, stack, bets, inv, allin=False))
            else:
                kids.append(self._decision(board, pot, stack, bets, o, n_raises, inv, checked=True))
            offered_allin = False
            for frac in cfg.bet_sizes[street]:
                amt = max(round(frac * pot_total * 2) / 2, 1.0)
                if amt >= remaining - 1e-9:
                    if not offered_allin and remaining > 0:
                        offered_allin = True
                        nb, ninv = with_bet(b_me + remaining)
                        names.append("all-in")
                        kids.append(self._decision(board, pot, stack, nb, o, n_raises + 1, ninv, False))
                    continue
                nb, ninv = with_bet(b_me + amt)
                names.append(f"bet {int(round(frac * 100))}%")
                kids.append(self._decision(board, pot, stack, nb, o, n_raises + 1, ninv, False))
            if not offered_allin and remaining > 0 and remaining <= cfg.allin_spr * pot_total:
                nb, ninv = with_bet(b_me + remaining)
                names.append("all-in")
                kids.append(self._decision(board, pot, stack, nb, o, n_raises + 1, ninv, False))
        else:  # facing a bet
            names.append("fold")
            kids.append(Terminal(board, "fold", inv, folder=p))
            call_amt = min(b_op - b_me, remaining)
            nb, ninv = with_bet(b_me + call_amt)
            names.append("call")
            allin = b_op >= stack - 1e-9
            kids.append(self._street_over(board, pot, stack, nb, ninv, allin=allin))
            if remaining > call_amt + 1e-9:
                pot_after_call = pot_total + call_amt
                offered = set()
                capped = False
                if n_raises < cfg.max_raises:
                    for frac in cfg.raise_sizes:
                        raise_to = b_op + round(frac * pot_after_call * 2) / 2
                        raise_to = max(raise_to, 2 * b_op - b_me)      # at least a min-raise
                        if raise_to >= stack - 1e-9:
                            capped = True
                            continue
                        if raise_to in offered:
                            continue
                        offered.add(raise_to)
                        nb, ninv = with_bet(raise_to)
                        names.append(f"raise to {raise_to:g}")
                        kids.append(self._decision(board, pot, stack, nb, o, n_raises + 1, ninv, False))
                if capped or not offered or remaining <= cfg.allin_spr * pot_total:
                    nb, ninv = with_bet(stack)
                    names.append("all-in")
                    kids.append(self._decision(board, pot, stack, nb, o, n_raises + 1, ninv, False))
        self.n_nodes += 1
        return Decision(board, p, names, kids, pot_total, bets, self.ctx.n)


# ----------------------------------------------------------------- solver
class Solver:
    def __init__(self, ctx: Ctx, root: Decision, dead: set):
        self.ctx, self.root, self.dead = ctx, root, dead
        self.iters = 0
        self.alpha, self.beta, self.gamma = 1.5, 0.0, 2.0   # DCFR parameters
        self._nodes: List[Decision] = []
        self._collect(root)

    def _collect(self, node):
        if isinstance(node, Decision):
            self._nodes.append(node)
            for c in node.children:
                self._collect(c)
        elif isinstance(node, Chance):
            for c in node.children:
                self._collect(c)

    def compat_sum(self, ro):
        cs = self.ctx.member @ ro
        return ro.sum() - cs[self.ctx.ca] - cs[self.ctx.cb] + ro

    def _cfr(self, node, p, r0, r1, update, use_avg):
        ro = r1 if p == 0 else r0
        if isinstance(node, Terminal):
            if node.kind == "fold":
                if node.folder == p:
                    return -node.inv[p] * self.compat_sum(ro)
                return node.inv[1 - p] * self.compat_sum(ro)
            if node.M is None:
                node.M = (self.ctx.showdown_matrix(node.board) if node.kind == "showdown"
                          else self.ctx.leaf_matrix(node.board, self.dead))
            return node.inv[p] * (node.M @ ro)
        if isinstance(node, Chance):
            util = np.zeros(self.ctx.n)
            w = 1.0 / len(node.children)
            for c, child in zip(node.cards, node.children):
                mask = 1.0 - self.ctx.member[c]
                util += w * mask * self._cfr(child, p, r0 * mask, r1 * mask, update, use_avg)
            return util
        strat = node.avg_strategy() if use_avg else node.strategy()
        A = len(node.names)
        if node.player == p:
            utils = np.empty((A, self.ctx.n))
            for a, child in enumerate(node.children):
                utils[a] = self._cfr(child, p, r0, r1, update, use_avg)
            node_util = (strat * utils).sum(axis=0)
            if update:
                rp = r0 if p == 0 else r1
                node.regret += utils - node_util
                node.ssum += rp[None, :] * strat
            return node_util
        util = np.zeros(self.ctx.n)
        for a, child in enumerate(node.children):
            if p == 0:
                util += self._cfr(child, p, r0, r1 * strat[a], update, use_avg)
            else:
                util += self._cfr(child, p, r0 * strat[a], r1, update, use_avg)
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
        t0 = time.time()
        while self.iters < max_iters:
            for p in (0, 1):
                self._cfr(self.root, p, r0, r1, update=True, use_avg=False)
            self.iters += 1
            self._discount()
            if self.iters >= min_iters and time.time() - t0 > time_budget:
                break
        return self.iters, time.time() - t0

    def root_action_evs(self, r0, r1):
        root = self.root
        p = root.player
        ro = r1 if p == 0 else r0
        cs = self.compat_sum(ro)
        inv_p = _root_inv(root, p)
        evs = np.zeros((len(root.names), self.ctx.n))
        for a, child in enumerate(root.children):
            u = self._cfr(child, p, r0, r1, update=False, use_avg=True)
            evs[a] = np.where(cs > 1e-12, u / np.where(cs > 1e-12, cs, 1.0), 0.0) + inv_p
        return evs


def _root_inv(root: Decision, p: int) -> float:
    n = root
    while not isinstance(n, Terminal):
        n = n.children[0]
    if n.kind == "fold" and n.folder != p:
        # child 0 of a "facing bet" node is our fold; inv unchanged. For check paths inv unchanged too.
        return n.inv[p]
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

    def for_combo(self, cb: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
        i = COMBO_INDEX[(min(cb), max(cb))]
        return {n: (float(self.strategy[a, i]), float(self.evs[a, i])) for a, n in enumerate(self.names)}

    def range_strategy(self, r_full: np.ndarray) -> Dict[str, float]:
        """Overall frequency of each root action across the root player's range."""
        w = r_full / max(r_full.sum(), 1e-12)
        return {n: float((self.strategy[a] * w).sum()) for a, n in enumerate(self.names)}


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
    evs = s.root_action_evs(r0, r1)
    strat = root.avg_strategy()
    A = len(root.names)
    strat_full = np.zeros((A, N_FULL)); evs_full = np.zeros((A, N_FULL))
    strat_full[:, active] = strat; evs_full[:, active] = evs
    return Solution(root, root.names, strat_full, evs_full, iters, secs, tb.n_nodes, ctx, r0, r1)


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


def apply_actions(sol: Solution, actions: List[str]):
    """Walk the solved tree along real actions, multiplying each actor's range by the
    average frequency of that action. Returns (node, r0_full, r1_full, sizes_bb).
    Stops at the first chance/terminal node."""
    node = sol.root
    r0, r1 = sol.r0.copy(), sol.r1.copy()
    for act in actions:
        if not isinstance(node, Decision):
            break
        a = match_action(node.names, act)
        s = node.avg_strategy()[a]
        if node.player == 0:
            r0 = r0 * s
        else:
            r1 = r1 * s
        node = node.children[a]
    return node, expand(sol.ctx, r0), expand(sol.ctx, r1)
