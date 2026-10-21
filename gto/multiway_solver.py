"""Three-player (and two-player) postflop solver: range-vs-range CFR over a small bet tree.

Why and how (docs/multiway_research.md has the sources):
* With three players there is no Nash guarantee for CFR (it can cycle; equilibria aren't unique or
  interchangeable). In practice self-play CFR (Pluribus, Gibson's 3-player ACPC agents, the commercial
  3-way solvers) gives strategies whose best-response gains shrink towards zero, and that is what we
  measure: `nash_conv` is each player's best-response gain against the others' average strategies,
  summed, in % of the pot.
* Vectorised full-width CFR: every node holds (boards, actions, combos) regrets; one pass computes the
  counterfactual values of all three players at once (simultaneous updates) with DCFR discounting.
* Card removal for three ranges, exactly. Player p's value for hand h at a terminal is a sum over the
  other two players' hands (q, r) that share no card with h or with each other:
      sum_{q, r} x_q(q) A(h, q) x_r(r) B(h, r)
  for "kernels" A, B (all hands, weaker hands, tied hands). Dropping the q-r overlap from the product
  of the two one-sided sums is inclusion-exclusion over the 52 cards: a pair overlapping in one card is
  counted once in sum_c P_q(h, c) P_r(h, c) (P_q(h, c): q's hands holding c, compatible with h, kernel
  A), a pair holding the same combo twice, so
      S = (A x_q)(h) (B x_r)(h) - sum_c P_q(h, c) P_r(h, c) + sum_{q = r} A B x_q x_r.
  Every term comes from running sums over the hands sorted by strength (per card for P), O(52 n) per
  terminal instead of the O(n^3) triple loop. `exact=False` drops the last two terms (q and r
  independent given h: the "pairwise" approximation); see pairwise_error() for what that costs.
* Showdown shares with ties: hero's share against two opponents is W_q W_r + E_q E_r / 12 with
  W = weaker + tied / 2 and E = tied (beat both: 1, tie one: 1/2, tie both: 1/3).
* Streets: the river is solved per combo on its board; from the turn every river card is a batch row
  under a chance node (or a subset, `rivers`); from the flop the turns are sampled (`turns`) and the
  river is an equity leaf (checked down over `leaf_rivers` rivers per turn).

Units: big blinds. Players 0..n-1 in postflop acting order. Values are chips relative to the
contributions at the root street's start (u = share x (pot - rake) - chips put in); action EVs are
reported like the heads-up solver's, relative to folding now.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cards import ALL_COMBOS, COMBO_INDEX
from .evaluator import eval_boards
np.seterr(divide="ignore", over="ignore", invalid="ignore")   # (spurious matmul warnings on macOS Accelerate)

N_FULL = 1326
_MEM = np.zeros((N_FULL, 52))
for _i, (_a, _b) in enumerate(ALL_COMBOS):
    _MEM[_i, _a] = _MEM[_i, _b] = 1.0
_CA = np.array([a for a, b in ALL_COMBOS]); _CB = np.array([b for a, b in ALL_COMBOS])
_PAIR = np.full((52, 52), -1, dtype=np.int64)          # full combo index of {c, d}
for _i, (_a, _b) in enumerate(ALL_COMBOS):
    _PAIR[_a, _b] = _PAIR[_b, _a] = _i


@dataclass
class MWConfig:
    bet: Dict[str, float] = field(default_factory=lambda: {"flop": 0.33, "turn": 0.5, "river": 0.5})
    raise_frac: float = 0.75     # raise to: the bet plus this fraction of the pot after calling
    max_raises: int = 2          # a bet and one raise per street; then fold or call (all-in below)
    allin_spr: float = 1.5       # all-in is offered too when the stack behind is at most this many pots
    depth: int = 1               # streets bet on from the root; after them the cards are dealt out (equity leaves)
    exact: bool = True           # joint card removal for three ranges on the river root's board (the rows of a
                                 # turn or flop solve's runouts are pairwise, see module doc); False: pairwise there too
    rivers: Optional[int] = None   # rivers under a turn solve (None: every one)
    turns: Optional[int] = None  # turns under a flop solve (None: every one)
    leaf_rivers: int = 1         # rivers per turn under a flop solve (sampled)
    rake_rate: float = 0.0
    rake_cap: Optional[float] = None
    alternate: bool = True       # alternating updates (one player per pass) instead of simultaneous
    alpha: float = 1.5           # DCFR: positive regrets x t^a / (t^a + 1), negative x 1/2,
    gamma: float = 2.0           # average strategy x (t / (t + 1))^g
    seed: int = 3

    def rake(self, pot: float) -> float:
        if self.rake_rate <= 0:
            return 0.0
        r = self.rake_rate * pot
        return r if self.rake_cap is None else min(r, self.rake_cap)


# ------------------------------------------------------------------ hands on a set of boards
class Hands:
    """The combos in play (the union of the ranges) and their card structure."""

    def __init__(self, active: np.ndarray):
        self.active = active
        n = self.n = len(active)
        self.ca, self.cb = _CA[active], _CB[active]
        self.mem = _MEM[active]                                       # (n, 52)
        f2a = np.full(N_FULL + 1, n, dtype=np.int64)
        f2a[active] = np.arange(n)
        # pidx[k][h, c]: the active index of the combo {c, card k of h} (n = not in play / no such combo)
        cards = np.arange(52)
        self.pidx = [f2a[_PAIR[cards[None, :], cc[:, None]]] for cc in (self.ca, self.cb)]
        for k in (0, 1):
            self.pidx[k][self.pidx[k] < 0] = n
        self.notmine = 1.0 - self.mem                                 # (n, 52)


class Boards:
    """A batch of B boards (complete five-card boards for showdowns): strengths and sort orders."""

    def __init__(self, hands: Hands, boards: List[tuple]):
        self.boards = boards
        self.h = hands
        n, B = hands.n, len(boards)
        self.B = B
        if len(boards[0]) == 5:
            s = eval_boards(boards, hands.active).astype(np.float64)    # (B, n), -1 = holds a board card
        else:
            s = np.zeros((B, n))
            for i, b in enumerate(boards):
                s[i, hands.mem[:, list(b)].sum(axis=1) > 0] = -1
        self.s = s
        if len(boards[0]) < 5:
            return
        self.order = np.argsort(s, axis=1, kind="stable")
        ss = np.take_along_axis(s, self.order, axis=1)
        self.lo = np.stack([np.searchsorted(ss[i], s[i], "left") for i in range(B)])
        self.hi = np.stack([np.searchsorted(ss[i], s[i], "right") for i in range(B)])
        self.r = np.arange(B)[:, None]
        rows = self.r
        # (card, combo) pairs, two per combo, sorted by card then strength (as the heads-up solver's Ranking):
        # one running sum over them gives, for each combo h and each of its cards, the reach of the combos
        # holding that card that are weaker (or weaker or tied) than h
        pc = np.concatenate([hands.ca, hands.cb])
        pi = np.concatenate([np.arange(n), np.arange(n)])
        key = pc[None, :] * 1e4 + s[:, pi]
        po = np.argsort(key, axis=1, kind="stable")
        skey = np.take_along_axis(key, po, axis=1)
        counts = np.bincount(pc, minlength=52)
        cstart = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.fsort = (rows * n + self.order).ravel()
        self.fpair = (rows * n + pi[po]).ravel()
        self.flo = (rows * (n + 1) + self.lo).ravel()
        self.fhi = (rows * (n + 1) + self.hi).ravel()
        self.fp = {}
        for k, cc in ((0, hands.ca), (1, hands.cb)):
            q = cc[None, :] * 1e4 + s
            plo = np.stack([np.searchsorted(skey[i], q[i], "left") for i in range(B)])
            phi = np.stack([np.searchsorted(skey[i], q[i], "right") for i in range(B)])
            self.fp[k] = tuple((rows * (2 * n + 1) + a).ravel() for a in (plo, phi, np.broadcast_to(cstart[cc], (B, n))))
        self._exact_ready = False

    def _exact_prep(self):
        """The (B, n, 52) tables for the exact overlap term (built on first use)."""
        if self._exact_ready:
            return
        hands, s, B = self.h, self.s, self.B
        self.mem_s = hands.mem.astype(np.float32)[self.order]         # (B, n, 52) in strength order
        self.mem32 = hands.mem.astype(np.float32)
        self.notmine32 = hands.notmine.astype(np.float32)
        self.sp = np.concatenate([s, np.full((B, 1), -9.0)], axis=1)  # (pad for pidx = n)
        # strength comparisons of the combos {c, card of h} with h, for the weaker / tied kernels
        self.cond = {}
        for kind in ("L", "E"):
            out = []
            for k in (0, 1):
                other = self.sp[:, hands.pidx[k]]                    # (B, n, 52)
                out.append((other < s[:, :, None]) if kind == "L" else (other == s[:, :, None]))
            self.cond[kind] = out
        self.cond["W"] = [(self.cond["L"][k] + 0.5 * self.cond["E"][k]).astype(np.float32) for k in (0, 1)]
        self.cond["E"] = [c.astype(np.float32) for c in self.cond["E"]]
        self._exact_ready = True

    # -- O(n) kernels
    def k_le(self, x):
        """(K_L, K_E): per combo h, the reach x over the combos compatible with h that are weaker than h / tied."""
        B, n = self.B, self.h.n
        X = np.zeros((B, n + 1)); np.cumsum(x.ravel()[self.fsort].reshape(B, n), axis=1, out=X[:, 1:])
        cs = np.zeros((B, 2 * n + 1)); np.cumsum(x.ravel()[self.fpair].reshape(B, 2 * n), axis=1, out=cs[:, 1:])
        X = X.ravel(); cs = cs.ravel()
        (lo0, hi0, st0), (lo1, hi1, st1) = self.fp[0], self.fp[1]
        b0, b1 = cs[st0], cs[st1]
        kl = X[self.flo] - (cs[lo0] - b0) - (cs[lo1] - b1)
        kle = X[self.fhi] - (cs[hi0] - b0) - (cs[hi1] - b1)
        kl = kl.reshape(B, n)
        return kl, kle.reshape(B, n) + x - kl

    def k_u(self, x):
        """(K_U, C, T): the reach over the combos compatible with h; per-card totals; the total."""
        h = self.h
        C = x @ h.mem
        T = x.sum(axis=1, keepdims=True)
        return T - C[:, h.ca] - C[:, h.cb] + x, C, T

    def pair_matrix(self, x):
        """(B, 52, 52): x of the combo {c, d} at [c, d] and [d, c]."""
        h = self.h
        M = np.zeros((self.B, 52 * 52))
        M[:, h.ca * 52 + h.cb] = x
        M[:, h.cb * 52 + h.ca] = x
        return M.reshape(self.B, 52, 52)

    def p_tables(self, x, kinds):
        """{kind: P} for kinds among U, W, E: P[b, h, c] = x over the combos holding card c that are compatible
        with h and counted by the kernel (U all, W weaker + tied / 2, E tied); 0 for h's own cards. float32."""
        self._exact_prep()
        h, B, n = self.h, self.B, self.h.n
        x32 = x.astype(np.float32)
        xp = np.concatenate([x32, np.zeros((B, 1), np.float32)], axis=1)
        y0, y1 = xp[:, h.pidx[0]], xp[:, h.pidx[1]]                  # (B, n, 52): x of {c, h's card}
        out = {}
        if "U" in kinds:
            out["U"] = ((x32 @ self.mem32)[:, None, :] - y0 - y1) * self.notmine32
        if "W" in kinds or "E" in kinds:
            xs = x32.ravel()[self.fsort].reshape(B, n)
            Y = np.zeros((B, n + 1, 52), np.float32)
            np.cumsum(xs[:, :, None] * self.mem_s, axis=1, out=Y[:, 1:])
            YL, YH = Y[self.r, self.lo], Y[self.r, self.hi]
            for kind in ("W", "E"):
                if kind in kinds:
                    c0, c1 = self.cond[kind]
                    C = 0.5 * (YL + YH) if kind == "W" else YH - YL
                    C -= y0 * c0; C -= y1 * c1; C *= self.notmine32
                    out[kind] = C
        return out


# ------------------------------------------------------------------ tree
class Terminal:
    __slots__ = ("bd", "live", "pot", "c", "rake", "kind")

    def __init__(self, bd, live, pot, c, rake, kind):
        self.bd, self.live, self.pot, self.c, self.rake, self.kind = bd, live, pot, c, rake, kind


class Chance:
    __slots__ = ("bd", "child", "K", "masks", "w")

    def __init__(self, bd, child, K, masks, w):
        self.bd, self.child, self.K, self.masks, self.w = bd, child, K, masks, w


class Decision:
    __slots__ = ("bd", "player", "names", "children", "regret", "ssum", "pot_now", "c", "street", "amounts")

    def __init__(self, bd, player, names, children, pot_now, c, street):
        self.bd, self.player, self.names, self.children = bd, player, names, children
        self.pot_now, self.c, self.street = pot_now, c, street
        A, B, n = len(children), bd.B, bd.h.n
        self.regret = np.zeros((B, A, n), np.float32)
        self.ssum = np.zeros((B, A, n), np.float32)

    def strategy(self):
        r = np.maximum(self.regret, 0.0)
        s = r.sum(axis=1, keepdims=True)
        return np.where(s > 0, r / np.where(s > 0, s, 1.0), 1.0 / len(self.children))

    def avg_strategy(self):
        s = self.ssum.sum(axis=1, keepdims=True)
        return np.where(s > 0, self.ssum / np.where(s > 0, s, 1.0), 1.0 / len(self.children))


STREET = {3: "flop", 4: "turn", 5: "river"}


class Builder:
    def __init__(self, hands: Hands, cfg: MWConfig, n_players: int, dead: set):
        self.h, self.cfg, self.np, self.dead = hands, cfg, n_players, set(dead)
        self.rng = random.Random(cfg.seed)
        self.decisions: List[Decision] = []
        self.n_nodes = 0
        self._bd: Dict[tuple, Boards] = {}
        self.root_len = 5

    def boards(self, boards):
        key = tuple(boards)
        if key not in self._bd:
            self._bd[key] = Boards(self.h, list(boards))
        return self._bd[key]

    def build(self, board, pot_start, stack, bets, to_act, n_raises, pending):
        """The tree from a node in the middle of a street: `bets` this street per player, `stack` each player's
        chips behind at the street's start (the same for all), `pending` the players still to act."""
        c = list(bets)
        return self._decision([tuple(board)], pot_start, stack, list(bets), c, to_act, n_raises,
                              set(pending), frozenset())

    def _next(self, p, pending):
        for k in range(1, self.np + 1):
            q = (p + k) % self.np
            if q in pending:
                return q
        return None

    def _close(self, boards, pot_start, stack, bets, c, folded):
        live = [i for i in range(self.np) if i not in folded]
        street_len = len(boards[0])
        if len(live) == 1:
            # uncalled chips go back to the winner
            w = live[0]
            back = bets[w] - max(bets[i] for i in range(self.np) if i != w)
            c2 = list(c); c2[w] -= max(back, 0.0)
            pot = pot_start + sum(bets) - max(back, 0.0)
            return self._terminal(boards, tuple(live), pot, c2, "fold")
        pot = pot_start + sum(bets)
        behind = stack - max(bets)
        if street_len == 5:
            return self._terminal(boards, tuple(live), pot, list(c), "show")
        if behind <= 1e-9 or street_len - self.root_len + 1 >= self.cfg.depth:
            # all-in, or the last street the tree bets on: the rest is dealt out (an equity leaf)
            return self._runout(boards, tuple(live), pot, list(c))
        return self._chance(boards, pot, behind, list(c), folded)

    def _terminal(self, boards, live, pot, c, kind):
        self.n_nodes += 1
        return Terminal(self.boards(boards), live, pot, np.array(c, dtype=np.float64), self.cfg.rake(pot), kind)

    def _cards_after(self, b):
        return [x for x in range(52) if x not in self.dead and x not in b]

    def _children(self, boards, k):
        """The next card for every board: all of them, or k sampled per board."""
        out = []
        for b in boards:
            cs = self._cards_after(b)
            if k is not None and k < len(cs):
                cs = sorted(self.rng.sample(cs, k))
            out.append(cs)
        K = min(len(cs) for cs in out)
        return [cs[:K] for cs in out], K

    def _chance_node(self, boards, child_fn, k):
        cards, K = self._children(boards, k)
        child_boards = [b + (x,) for b, cs in zip(boards, cards) for x in cs]
        masks = np.stack([self.h.notmine[:, x] for cs in cards for x in cs]).reshape(len(boards), K, self.h.n)
        unseen = 52 - len(self.dead | set(boards[0])) - 2 * self.np    # cards none of the hands holds
        avail = len(self._cards_after(boards[0]))
        # each of the K cards stands for avail / K of them; a card is dealt with probability 1 / unseen
        w = avail / K / unseen
        self.n_nodes += 1
        return Chance(self.boards(boards), child_fn(child_boards), K, masks, w)

    def _chance(self, boards, pot, behind, c, folded):
        k = self.cfg.turns if len(boards[0]) == 3 else self._k_rivers()
        pending = {i for i in range(self.np) if i not in folded}
        first = min(pending)
        return self._chance_node(boards, lambda cb: self._decision(cb, pot, behind, [0.0] * self.np, c, first, 0,
                                                                   set(pending), folded), k)

    def _runout(self, boards, live, pot, c):
        """Everyone left is all-in: the remaining cards are dealt (an equity leaf)."""
        if len(boards[0]) == 4:
            return self._chance_node(boards, lambda cb: self._terminal(cb, live, pot, c, "show"), self._k_rivers())
        return self._chance_node(boards, lambda cb: self._runout(cb, live, pot, c), self.cfg.turns)

    def _k_rivers(self):
        return self.cfg.rivers if self.root_len == 4 else self.cfg.leaf_rivers

    def _decision(self, boards, pot_start, stack, bets, c, p, n_raises, pending, folded):
        cfg = self.cfg
        street = STREET[len(boards[0])]
        pot_now = pot_start + sum(bets)
        top = max(bets)
        rem = stack - bets[p]
        names, kids = [], []

        def act(new_bet, raised, fold=False):
            nb = list(bets); nc = list(c)
            nf = folded | {p} if fold else folded
            if not fold:
                nc[p] += new_bet - bets[p]; nb[p] = new_bet
            npend = set(pending) - {p}
            if raised:
                npend = {i for i in range(self.np) if i != p and i not in nf and nb[i] < stack - 1e-9}
            live = [i for i in range(self.np) if i not in nf]
            if len(live) == 1 or not npend:
                return self._close(boards, pot_start, stack, nb, nc, nf)
            # a lone player with chips facing nothing more to call: the street is over
            return self._decision(boards, pot_start, stack, nb, nc, self._next(p, npend), n_raises + raised,
                                  npend, nf)

        others_can = any(i != p and i not in folded and bets[i] < stack - 1e-9 for i in range(self.np))
        amts = []      # per action: the player's chips in this street after it (to call: the call's size)

        def add(name, kid, amt):
            names.append(name); kids.append(kid); amts.append(amt)
        if top <= bets[p] + 1e-9:
            add("check", act(bets[p], False), None)
            if others_can:
                amt = max(round(cfg.bet[street] * pot_now * 2) / 2, 1.0)
                if amt < rem - 1e-9:
                    add(f"bet {int(round(cfg.bet[street] * 100))}%", act(bets[p] + amt, True), bets[p] + amt)
                if amt >= rem - 1e-9 or rem <= cfg.allin_spr * pot_now:
                    add("all-in", act(stack, True), stack)
        else:
            add("fold", act(bets[p], False, fold=True), None)
            add("call", act(min(top, stack), False), min(top, stack) - bets[p])
            if rem > top - bets[p] + 1e-9 and others_can:
                allin = rem <= cfg.allin_spr * pot_now
                if n_raises < cfg.max_raises:
                    to = top + round(cfg.raise_frac * (pot_now + top - bets[p]) * 2) / 2
                    if to < stack - 1e-9:
                        add(f"raise to {to:g}", act(to, True), to)
                    else:
                        allin = True
                if allin:
                    add("all-in", act(stack, True), stack)
        self.n_nodes += 1
        d = Decision(self.boards(boards), p, names, kids, pot_now, list(c), street)
        d.amounts = amts
        self.decisions.append(d)
        return d


# ------------------------------------------------------------------ solver
class MultiwaySolver:
    def __init__(self, board, ranges: Sequence[np.ndarray], pot: float, stack: float, bets=None, to_act=0,
                 n_raises=0, pending=None, cfg: Optional[MWConfig] = None, dead: Sequence[int] = (),
                 include: Sequence[int] = ()):
        """`ranges`: one (1326,) weight vector per player, in postflop acting order. `pot`: the chips in the
        middle at the start of this street; `stack`: each player's chips behind at the start of this street
        (one number: the solver plays equal stacks); `bets`: chips each put in this street so far; `to_act`:
        who acts at the root; `pending`: who still has to act on this street (default everyone from to_act on
        when nobody has bet, else everyone who hasn't matched the top bet). `include`: combos (indices into the 1326)
        to solve even with no weight in any range (a hand's strategy is then the regret-matching one: see
        hand_strategy)."""
        self.cfg = cfg or MWConfig()
        self.board = tuple(board)
        self.np = len(ranges)
        assert self.np in (2, 3)
        dead_all = set(board) | set(dead)
        bm = np.ones(N_FULL)
        for x in dead_all:
            bm *= 1.0 - _MEM[:, x]
        rs = [np.asarray(r, dtype=np.float64) * bm for r in ranges]
        live = np.any([r > 0 for r in rs], axis=0)
        for i in include:
            live[i] = live[i] or bm[i] > 0
        active = np.nonzero(live)[0]
        self.hands = Hands(active)
        self.reach0 = np.stack([r[active] for r in rs])[:, None, :]       # (np, 1, n)
        bets = list(bets) if bets is not None else [0.0] * self.np
        if pending is None:
            top = max(bets)
            if top <= 0:
                pending = {(to_act + k) % self.np for k in range(self.np)}
            else:
                pending = {i for i in range(self.np) if bets[i] < top - 1e-9}
                pending.add(to_act)
        self.pot_start, self.stack = pot, stack
        b = Builder(self.hands, self.cfg, self.np, dead_all - set(board))
        b.root_len = len(self.board)
        self.builder = b
        self.root = b.build(self.board, pot, stack, bets, to_act, n_raises, pending)
        self.decisions = b.decisions
        self.iters = 0
        self.seconds = 0.0

    # -- terminal values
    def _term(self, node: Terminal, R, need=None, rho=None):
        """(np, B, n) values: u_p = share x (pot - rake) - chips put in, summed over the others' compatible hands
        (only player `need`'s row when given)."""
        bd = node.bd
        npl = self.np
        exact = self.cfg.exact and bd.B == 1
        out = np.zeros((npl, bd.B, self.hands.n))
        net = node.pot - node.rake
        players = range(npl) if need is None else [need]
        cache = {}

        def get(key, fn):
            if key not in cache:
                cache[key] = fn()
            return cache[key]

        KU = lambda j: get(("U", j), lambda: bd.k_u(R[j]))
        KLE = lambda j: get(("LE", j), lambda: bd.k_le(R[j]))
        KW = lambda j: get(("W", j), lambda: KLE(j)[0] + 0.5 * KLE(j)[1])
        live3 = len(node.live) == 3
        P = lambda j, kind: get(("P", j), lambda: bd.p_tables(R[j], ("W", "E") if live3 else
                                                              ("W",) if j in node.live else ("U",)))[kind]
        y = lambda q, r: get(("y", min(q, r), max(q, r)), lambda: R[q] * R[r])
        ykle = lambda q, r: get(("yLE", min(q, r), max(q, r)), lambda: bd.k_le(y(q, r)))

        def mass(q, r):
            if rho is not None:
                return get(("m", min(q, r), max(q, r)), lambda: rho[(min(q, r), max(q, r))] * KU(q)[0] * KU(r)[0])
            return get(("m", min(q, r), max(q, r)), lambda: self._mass(bd, R, q, r, KU(q), KU(r)))

        dot = lambda a, b: np.einsum("bnc,bnc->bn", a, b)

        def win_of(p):
            """p's pot share summed over the others' compatible hands, and the joint mass."""
            others = [j for j in range(npl) if j != p]
            if npl == 2:
                q = others[0]
                m = KU(q)[0]
                if p not in node.live:
                    return 0.0, m
                return (m if len(node.live) == 1 else KW(q)), m
            q, r = others
            m = mass(q, r)
            if p not in node.live:
                return 0.0, m
            if len(node.live) == 1:
                return m, m
            if len(node.live) == 2:
                o = q if q in node.live else r
                f = r if o == q else q
                if exact:
                    kl, ke = ykle(o, f)
                    return KW(o) * KU(f)[0] - dot(P(o, "W"), P(f, "U")) + kl + 0.5 * ke, m
                return KW(o) * KU(f)[0] * self._ratio(m, KU(o)[0], KU(f)[0]), m
            KEq, KEr = KLE(q)[1], KLE(r)[1]
            win = KW(q) * KW(r) + KEq * KEr / 12.0
            if exact:
                kl, ke = ykle(q, r)
                return win - dot(P(q, "W"), P(r, "W")) - dot(P(q, "E"), P(r, "E")) / 12.0 + kl + ke / 3.0, m
            return win * self._ratio(m, KU(q)[0], KU(r)[0]), m

        if not exact and npl == 3 and len(node.live) >= 2:
            # pairwise: q and r independent given h (the joint mass kept exact). The live players' shares then
            # needn't add up to every deal's pot; scaled per board so they do, the game stays constant-sum.
            wm = {p: win_of(p) for p in node.live}
            tot = sum((R[p] * wm[p][0]).sum(axis=1, keepdims=True) for p in node.live)
            p0 = node.live[0]
            M = (R[p0] * wm[p0][1]).sum(axis=1, keepdims=True)
            k = np.where(tot > 1e-300, M / np.where(tot > 1e-300, tot, 1.0), 1.0)
            for p in players:
                win, m = (wm[p][0] * k, wm[p][1]) if p in wm else win_of(p)
                out[p] = net * win - node.c[p] * m
            return out
        for p in players:
            win, m = win_of(p)
            out[p] = net * win - node.c[p] * m
        return out

    @staticmethod
    def _ratio(m, a, b):
        d = a * b
        return np.where(d > 1e-300, m / np.where(d > 1e-300, d, 1.0), 0.0)

    @staticmethod
    def _mass(bd, R, q, r, kuq=None, kur=None):
        """Exact joint mass: per combo h of the third player, the sum of x_q(h_q) x_r(h_r) over the pairs of
        combos that share no card with h or with each other, in O(52^3 + n) from the 52 x 52 card-pair matrices:
        mass = K_U(q) K_U(r) - sum_{c not in h} P_q(h, c) P_r(h, c) + sum_{h' compatible} x_q x_r(h'), with
        P_q(h, c) = C_q[c] - X_q[c, e] - X_q[c, f] (h = {e, f}) expanded into card-level sums."""
        h = bd.h
        xq, xr = R[q], R[r]
        (Kq, Cq, Tq) = kuq if kuq is not None else bd.k_u(xq)
        (Kr, Cr, Tr) = kur if kur is not None else bd.k_u(xr)
        Xq, Xr = bd.pair_matrix(xq), bd.pair_matrix(xr)
        e, f = h.ca, h.cb
        D = (Cq * Cr).sum(axis=1, keepdims=True)
        Fqr = np.einsum("bcd,bd->bc", Xr, Cq)       # sum_d X_r[c, d] C_q[d]
        Frq = np.einsum("bcd,bd->bc", Xq, Cr)
        G = np.matmul(Xq, Xr).reshape(bd.B, 52 * 52)
        full = (D - Fqr[:, e] - Fqr[:, f] - Frq[:, e] - Frq[:, f]
                + G[:, e * 52 + e] + G[:, e * 52 + f] + G[:, f * 52 + e] + G[:, f * 52 + f])
        own = (Cq[:, e] - xq) * (Cr[:, e] - xr) + (Cq[:, f] - xq) * (Cr[:, f] - xr)
        y = xq * xr
        Ky = bd.k_u(y)[0]
        return Kq * Kr - (full - own) + Ky

    def _mass_exact(self, bd, R, q, r):
        return self._mass(bd, R, q, r)

    # -- traversal
    def _walk(self, node, R, mode, p=None, rho=None):
        """mode "cfr": update regrets and average strategies, return all players' values; "avg": values under
        the average strategies; "br": player p best-responds to the others' average strategies (p's values only);
        "alt": the traverser p's regrets and average strategy only (alternating updates), p's values.
        `rho`: an equity leaf's overlap ratios, from the board above its runouts (see _leaf_rho)."""
        if isinstance(node, Terminal):
            v = self._term(node, R, None if mode not in ("br", "alt") else p, rho)
            return v[p] if mode in ("br", "alt") else v
        if isinstance(node, Chance):
            B, K, n = node.bd.B, node.K, self.hands.n
            m = node.masks                                                   # (B, K, n)
            Rc = (R[:, :, None, :] * m[None]).reshape(self.np, B * K, n)
            if rho is None and self.np == 3 and self._is_leaf(node):
                rho = self._leaf_rho(node.bd, R)
            if rho is not None:
                rho = {k: np.repeat(v, K, axis=0) for k, v in rho.items()}
            v = self._walk(node.child, Rc, mode, p, rho)
            if mode in ("br", "alt"):
                return (v.reshape(B, K, n) * m).sum(axis=1) * node.w
            return (v.reshape(self.np, B, K, n) * m[None]).sum(axis=2) * node.w
        q = node.player
        if mode == "alt":
            # alternating updates: only the traverser p's regrets and average strategy
            sig = node.strategy()
            vals = []
            for a, ch in enumerate(node.children):
                Ra = R.copy(); Ra[q] = R[q] * sig[:, a, :]
                vals.append(self._walk(ch, Ra, mode, p))
            V = np.stack(vals)                                               # (A, B, n)
            if q != p:
                return V.sum(axis=0)
            vp = (V * np.moveaxis(sig, 1, 0)).sum(axis=0)
            node.regret += np.moveaxis(V - vp[None], 0, 1)
            node.ssum += R[p][:, None, :] * sig
            return vp
        if mode == "cfr":
            sig = node.strategy()
        else:
            sig = node.avg_strategy()
        vals = []
        for a, ch in enumerate(node.children):
            # (best-responding, p's own reach still follows its average strategy: p's values don't depend on it,
            # except through the pairwise leaves' per-board scaling, which should see the range p actually has)
            Ra = R.copy(); Ra[q] = R[q] * sig[:, a, :]
            vals.append(self._walk(ch, Ra, mode, p))
        if mode == "br":
            if q == p:
                return np.max(vals, axis=0)
            return np.sum(vals, axis=0)
        V = np.stack(vals)                                                   # (A, np, B, n)
        v = V.sum(axis=0)
        vq = (V[:, q] * np.moveaxis(sig, 1, 0)).sum(axis=0)
        v[q] = vq
        if mode == "cfr":
            node.regret += np.moveaxis(V[:, q] - vq[None], 0, 1)
            node.ssum += R[q][:, None, :] * sig
        return v

    @staticmethod
    def _is_leaf(node):
        """A chance node whose runouts end in showdowns with no more betting (an equity leaf)."""
        while isinstance(node, Chance):
            node = node.child
        return isinstance(node, Terminal)

    def _leaf_rho(self, bd, R):
        """For an equity leaf: per pair of players (q, r) and combo h of the third, the share of the q x r hand pairs
        compatible with h that are also compatible with each other (exact mass / product of one-sided masses), on
        the board above the runouts. The cards dealt below change it very little, so every runout reuses it: its
        showdowns then cost O(n) per board (pairwise shares, renormalized) instead of the O(52^3) exact mass."""
        out = {}
        for q, r in ((0, 1), (0, 2), (1, 2)):
            kq, kr = bd.k_u(R[q]), bd.k_u(R[r])
            out[(q, r)] = self._ratio(self._mass(bd, R, q, r, kq, kr), kq[0], kr[0])
        return out

    def _discount(self):
        t = float(self.iters)
        pos = t ** self.cfg.alpha / (t ** self.cfg.alpha + 1)
        sf = (t / (t + 1)) ** self.cfg.gamma
        for d in self.decisions:
            d.regret *= np.where(d.regret > 0, np.float32(pos), np.float32(0.5))
            d.ssum *= np.float32(sf)

    def run(self, iters=1000, time_budget=None, min_iters=5, should_stop=None):
        t0 = time.time()
        start = self.iters
        while self.iters - start < iters:
            if should_stop is not None and should_stop():
                break
            if self.cfg.alternate:
                for p in range(self.np):
                    self._walk(self.root, self.reach0, "alt", p)
            else:
                self._walk(self.root, self.reach0, "cfr")
            self.iters += 1
            self._discount()
            if time_budget is not None and self.iters - start >= min_iters and time.time() - t0 > time_budget:
                break
        self.seconds += time.time() - t0
        return self

    # -- diagnostics and results
    def total_mass(self):
        """Total weight of the deals (sum over compatible hand tuples of the reach products)."""
        R = self.reach0
        bd = self.root.bd
        if self.np == 2:
            return float((R[0] * bd.k_u(R[1])[0]).sum())
        return float((R[0] * self._mass_exact(bd, R, 1, 2)).sum())

    def values(self):
        """Each player's expected value (bb, relative to the root street's start) under the average strategies."""
        v = self._walk(self.root, self.reach0, "avg")
        m = self.total_mass()
        return np.array([(v[p][0] * self.reach0[p][0]).sum() for p in range(self.np)]) / m

    def br_gains(self):
        """What each player gains by best-responding to the others' average strategies (bb per deal)."""
        v = self.values()
        m = self.total_mass()
        out = []
        for p in range(self.np):
            br = self._walk(self.root, self.reach0, "br", p)
            out.append(float((br[0] * self.reach0[p][0]).sum()) / m - v[p])
        return np.array(out)

    def nash_conv(self):
        """Sum of the players' best-response gains, % of the pot at the root (0 at an equilibrium)."""
        return float(self.br_gains().sum()) / self.root_pot() * 100

    def root_pot(self):
        return self.pot_start + sum(self.root.c) if isinstance(self.root, Decision) else self.pot_start

    def action_evs(self, node: Optional[Decision] = None, R=None):
        """(actions, n) EV per action for the node's player, chips relative to folding now, everyone on their
        average strategies (the root by default)."""
        node = node or self.root
        R = self.reach0 if R is None else R
        p = node.player
        others = [j for j in range(self.np) if j != p]
        bd = node.bd
        if self.np == 2:
            mass = bd.k_u(R[others[0]])[0]
        else:
            mass = self._mass_exact(bd, R, *others)
        evs = []
        for ch in node.children:
            v = self._walk(ch, R, "avg")[p]
            evs.append(np.where(mass > 1e-12, v / np.where(mass > 1e-12, mass, 1.0), 0.0) + node.c[p])
        return np.stack(evs)[:, 0, :]

    def hand_strategy(self, ci: int, p: int, node: Optional[Decision] = None) -> np.ndarray:
        """Player p's mix at the node (the root) for combo ci: the average strategy, or where p's range gives the hand
        no weight (the average is undefined there) the current regret-matching one: its regrets still learn from
        the counterfactual values, which don't depend on its own weight."""
        node = node or self.root
        if self.reach0[p, 0, ci] > 1e-12:
            return node.avg_strategy()[0][:, ci]
        return node.strategy()[0][:, ci]

    def showdown_share(self, ci: int) -> float:
        """Combo ci's share of the pot against the other two ranges at showdown on the root's board (river)."""
        bd = self.root.bd
        t = Terminal(bd, tuple(range(self.np)), 1.0, np.zeros(self.np), 0.0, "show")
        p = self.root.player
        R = self.reach0
        win = self._term(t, R, p)[p][0]
        others = [j for j in range(self.np) if j != p]
        m = bd.k_u(R[others[0]])[0][0] if self.np == 2 else self._mass(bd, R, *others)[0]
        return float(win[ci] / m[ci]) if m[ci] > 1e-12 else 0.0

    def combo_index(self, hole) -> Optional[int]:
        i = COMBO_INDEX[(min(hole), max(hole))]
        pos = np.nonzero(self.hands.active == i)[0]
        return int(pos[0]) if len(pos) else None

    def range_freqs(self, node: Optional[Decision] = None, p_reach=None):
        """The node player's whole-range frequency of each action (average strategy)."""
        node = node or self.root
        r = self.reach0[node.player][0] if p_reach is None else p_reach
        s = node.avg_strategy()[0]
        return {nm: float((s[a] * r).sum() / max(r.sum(), 1e-12)) for a, nm in enumerate(node.names)}

    def expand(self, v):
        out = np.zeros(N_FULL)
        out[self.hands.active] = v
        return out


def solve(board, ranges, pot, stack, bets=None, to_act=0, n_raises=0, pending=None, cfg=None,
          iters=1000, time_budget=None, should_stop=None, dead=()) -> MultiwaySolver:
    s = MultiwaySolver(board, ranges, pot, stack, bets, to_act, n_raises, pending, cfg, dead)
    return s.run(iters=iters, time_budget=time_budget, should_stop=should_stop)
