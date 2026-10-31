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
* Streets: the root street is solved per combo. With later="buckets" the streets after it are bet on by hand
  buckets (see the section "later streets by hand buckets"); with later="deal" (`depth`) they are dealt out: from
  the turn every river card is a batch row under a chance node (or a subset, `rivers`), from the flop the turns
  are sampled (`turns`) and the river is an equity leaf (checked down over `leaf_rivers` rivers per turn).

Units: big blinds. Players 0..n-1 in postflop acting order. Values are chips relative to the
contributions at the root street's start (u = share x (pot - rake) - chips put in); action EVs are
reported like the heads-up solver's, relative to folding now.
"""
from __future__ import annotations

import itertools
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cards import ALL_COMBOS, COMBO_INDEX
from .evaluator import eval_boards
np.seterr(divide="ignore", over="ignore", invalid="ignore")   # (spurious matmul warnings on macOS Accelerate)

N_FULL = 1326
MAX_PLAYERS = 9   # four or more (or a 3-way pot with two all-in): pairwise card removal and equity leaves (_term_pw)
_MEM = np.zeros((N_FULL, 52))
for _i, (_a, _b) in enumerate(ALL_COMBOS):
    _MEM[_i, _a] = _MEM[_i, _b] = 1.0
_CA = np.array([a for a, b in ALL_COMBOS]); _CB = np.array([b for a, b in ALL_COMBOS])
_PAIR = np.full((52, 52), -1, dtype=np.int64)          # full combo index of {c, d}
for _i, (_a, _b) in enumerate(ALL_COMBOS):
    _PAIR[_a, _b] = _PAIR[_b, _a] = _i


class BuildTooSlow(Exception):
    """The tree and bucket tables were still building at the solve's deadline (MultiwaySolver's `deadline`): a quick
    answer's 3-way flop with a wide range or a 4-way limped flop spent its whole budget building and ran no
    iteration, so the betting rules answer at once instead."""
    def __init__(self, seconds: float):
        super().__init__(f"the tables were still building after {seconds:.2f} s")
        self.seconds = seconds


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
    # later="buckets": the streets after the root's are bet on too, by hand buckets (see BucketTables): from a flop
    # root the turn, and the river played on the turn's boards without its card (its showdowns are the turn's
    # equity over every river); from a turn root the river. "deal": they are dealt out (equity leaves, `depth`).
    later: str = "deal"
    buckets: int = 16            # hand buckets per player and card below the root street
    classes: bool = True         # the next card as one per class (a rank; a suit with two or more board cards apart)
    bucket_river_step: int = 1   # from a flop: the turn's showdown tensors average every step-th river card
    # allin: the players all-in at the root (their indices; three players: at most one, more is pairwise): they never
    # act. A player all-in for less, at the root or below it (stacks differ: MultiwaySolver's `stack` per player),
    # contests the pot up to its stake: a showdown pays each layer of the pot (MultiwaySolver._layers: the main pot,
    # everyone live, and the side pot, the players with more in; a fold gives up both) to the best hand among the live
    # players whose chips reach it, from every seat's chips in (inv0: each player's before the root street, dead: the
    # chips of seats folded before it; none given, everyone live had the same in before the root street)
    allin: tuple = ()
    inv0: tuple = ()
    dead: tuple = ()
    # icm: a tournament's payoffs (gto/icm.py MWPayoffs): every terminal pays each player the prize equity of its
    # outcomes in chip equivalents, as weights on the deals' weight and the player's share of each pot layer
    # (Terminal.W, BTerminal.W; MultiwaySolver._icm_setup); a player who folds below the root street is paid the pot
    # as it stood at its fold going to one of the players left. None: chips.
    icm: Optional[object] = None
    # root_raise: the root's player may raise once more than max_raises allows (four or more betting play max_raises 1,
    # no raises, which keeps the tree small enough to settle: 4-way 65 nodes where one raise made it 1053; the player
    # advised still gets to raise)
    root_raise: bool = False
    # root_jam: the root's player may go all-in however deep (a jam request: the page grading your all-in), the rest
    # of the tree as it is (all-in everywhere made the 4-way tree 401 nodes, 4% of the pot from settled)
    root_jam: bool = False
    alpha: float = 1.5           # DCFR: positive regrets x t^a / (t^a + 1), negative x 1/2,
    gamma: float = 3.0           # average strategy x (t / (t + 1))^g (3: 20-30% less exploitable than 2 at equal time, 3-way river and turns)
    seed: int = 3

    def rake(self, pot: float) -> float:
        if self.rake_rate <= 0:
            return 0.0
        r = self.rake_rate * pot
        return r if self.rake_cap is None else min(r, self.rake_cap)


def _pw_scale(want, have):
    """The pairwise terminals' per-board correction want / have, a factor near 1 (the pairs' overlap: ~0.9-1.1), held to
    [0.5, 2]: on a line its players almost never take, both are products of five or more near-zero reaches, and their
    ratio (denormal floats) overflowed the regrets to inf within ten iterations."""
    ok = have > 1e-280
    return np.clip(np.where(ok, want / np.where(ok, have, 1.0), 1.0), 0.5, 2.0)


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


def _tie_bounds(srt: np.ndarray, perm: np.ndarray):
    """For rows srt sorted ascending, srt = np.take_along_axis(x, perm, axis=1): np.searchsorted(srt[i], x[i], "left")
    and "right" for every row at once, by one scan for the runs of equal values (each element's own value is in its
    row, so the searches are those runs' ends) rather than a binary search per element."""
    B, m = srt.shape
    idx = np.broadcast_to(np.arange(m), (B, m))
    new = np.ones((B, m), dtype=bool)
    new[:, 1:] = srt[:, 1:] != srt[:, :-1]
    last = np.ones((B, m), dtype=bool)
    last[:, :-1] = new[:, 1:]
    lo = np.maximum.accumulate(np.where(new, idx, 0), axis=1)
    hi = np.minimum.accumulate(np.where(last, idx + 1, m)[:, ::-1], axis=1)[:, ::-1]
    out = []
    for v in (lo, hi):
        o = np.empty((B, m), dtype=np.intp)
        np.put_along_axis(o, perm, v, axis=1)
        out.append(o)
    return out


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
        self.lo, self.hi = _tie_bounds(ss, self.order)
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
        # a combo's own (card, combo) pairs are among the keys: its bounds are those pairs' runs of equal keys
        klo, khi = _tie_bounds(skey, po)
        for k, cc in ((0, hands.ca), (1, hands.cb)):
            plo, phi = klo[:, k * n:(k + 1) * n], khi[:, k * n:(k + 1) * n]
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

    def k_le_multi(self, x):
        """k_le for K reach vectors at once: x (B, n, K) -> (K_L, K_E), each (B, n, K)."""
        B, n, K = x.shape
        xf = x.reshape(B * n, K)
        X = np.zeros((B, n + 1, K)); np.cumsum(xf[self.fsort].reshape(B, n, K), axis=1, out=X[:, 1:])
        cs = np.zeros((B, 2 * n + 1, K)); np.cumsum(xf[self.fpair].reshape(B, 2 * n, K), axis=1, out=cs[:, 1:])
        X = X.reshape(-1, K); cs = cs.reshape(-1, K)
        (lo0, hi0, st0), (lo1, hi1, st1) = self.fp[0], self.fp[1]
        b0, b1 = cs[st0], cs[st1]
        kl = (X[self.flo] - (cs[lo0] - b0) - (cs[lo1] - b1)).reshape(B, n, K)
        kle = (X[self.fhi] - (cs[hi0] - b0) - (cs[hi1] - b1)).reshape(B, n, K)
        return kl, kle + x - kl

    def k_u_multi(self, x):
        """K_U for K reach vectors at once: x (B, n, K) -> (B, n, K)."""
        h = self.h
        C = np.matmul(h.mem.T, x)                     # (B, 52, K) per-card totals (einsum: 4x slower here)
        return x.sum(axis=1, keepdims=True) - C[:, h.ca] - C[:, h.cb] + x

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
    __slots__ = ("bd", "live", "pot", "c", "rake", "kind", "lay", "W")

    def __init__(self, bd, live, pot, c, rake, kind):
        self.bd, self.live, self.pot, self.c, self.rake, self.kind = bd, live, pot, c, rake, kind
        self.lay = None
        self.W = None       # a tournament's weights (np, 1 + layers): see MWConfig.icm


class Chance:
    __slots__ = ("bd", "child", "K", "masks", "w")

    def __init__(self, bd, child, K, masks, w):
        self.bd, self.child, self.K, self.masks, self.w = bd, child, K, masks, w


class Decision:
    __slots__ = ("bd", "player", "names", "children", "regret", "ssum", "pot_now", "c", "street", "amounts", "folded",
                 "fold_v")

    def __init__(self, bd, player, names, children, pot_now, c, street):
        self.bd, self.player, self.names, self.children = bd, player, names, children
        self.pot_now, self.c, self.street = pot_now, c, street
        self.folded, self.fold_v = frozenset(), None    # (the players folded before it; a tournament's fold value)
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
NEXT_STREET = {"flop": "turn", "turn": "river"}


def bet_sizes(cfg, street) -> list:
    """The street's bet sizes (fractions of the pot): MWConfig.bet holds one number or a list."""
    b = cfg.bet[street]
    return sorted(b) if isinstance(b, (list, tuple)) else [b]


def _per(stack, npl):
    """Each player's chips behind (a number: everyone's the same)."""
    return tuple(stack) if isinstance(stack, (tuple, list)) else (stack,) * npl


def _actions(cfg, npl, street, pot_start, stack, bets, c, p, n_raises, pending, folded):
    """Player p's actions at a node: [(name, amount (chips this street after it; a call: its size; else None),
    bets, contributions, folded, pending after it, raised)]. The street's bet sizes (MWConfig.bet: one or a list), one
    raise (to the bet plus raise_frac of the pot after calling), all-in at allin_spr pots behind or less, at most
    max_raises bets and raises a street. `stack`: each player's chips behind at the street's start (or one number for
    everyone); p puts in at most what another player still in can match (its all-in), and a player whose bet is its
    stack is all-in: it no longer acts."""
    stk = _per(stack, npl)
    pot_now = pot_start + sum(bets)
    top = max(bets)
    cap = min(stk[p], max((stk[i] for i in range(npl) if i != p and i not in folded), default=stk[p]))
    rem = cap - bets[p]
    out = []

    def act(name, amt, new_bet, raised, fold=False):
        nb = list(bets); nc = list(c)
        nf = folded | {p} if fold else folded
        if not fold:
            nc[p] += new_bet - bets[p]; nb[p] = new_bet
        npend = set(pending) - {p}
        if raised:
            npend = {i for i in range(npl) if i != p and i not in nf and nb[i] < stk[i] - 1e-9 and i not in cfg.allin}
        out.append((name, amt, nb, nc, nf, npend, int(raised)))

    others_can = any(i != p and i not in folded and i not in cfg.allin and bets[i] < stk[i] - 1e-9 for i in range(npl))
    if top <= bets[p] + 1e-9:
        act("check", None, bets[p], False)
        if others_can and rem > 1e-9:
            allin, seen = rem <= cfg.allin_spr * pot_now, set()
            for f in bet_sizes(cfg, street):
                amt = max(round(f * pot_now * 2) / 2, 1.0)
                if amt >= rem - 1e-9:
                    allin = True
                elif amt not in seen:
                    seen.add(amt)
                    act(f"bet {int(round(f * 100))}%", bets[p] + amt, bets[p] + amt, True)
            if allin:
                act("all-in", cap, cap, True)
    else:
        call = min(top, stk[p])
        act("fold", None, bets[p], False, fold=True)
        act("call", call - bets[p], call, False)
        if rem > top - bets[p] + 1e-9 and others_can:
            allin = rem <= cfg.allin_spr * pot_now
            if n_raises < cfg.max_raises:
                to = top + round(cfg.raise_frac * (pot_now + top - bets[p]) * 2) / 2
                if to < cap - 1e-9:
                    act(f"raise to {to:g}", to, to, True)
                else:
                    allin = True
            if allin:
                act("all-in", cap, cap, True)
    return out


def _next_player(npl, p, pending):
    for k in range(1, npl + 1):
        q = (p + k) % npl
        if q in pending:
            return q
    return None


def _street_end(npl, pot_start, stack, bets, c, folded):
    """(live, pot, contributions, each player's chips behind) when a street closes: the top bet's chips that nobody
    matched (a lone live player's, or a bet over a call all-in for less) go back to its bettor."""
    stk = _per(stack, npl)
    live = tuple(i for i in range(npl) if i not in folded)
    c2, nb = list(c), list(bets)
    w = max(range(npl), key=lambda i: bets[i])
    back = max(bets[w] - max(bets[i] for i in range(npl) if i != w), 0.0) if npl > 1 else 0.0
    if back > 1e-12:
        c2[w] -= back; nb[w] -= back
    else:
        back = 0.0
    return live, pot_start + sum(bets) - back, c2, tuple(stk[i] - nb[i] for i in range(npl))


def _acting(cfg, live, behind):
    """The live players with chips behind to bet (not all-in)."""
    return tuple(i for i in live if i not in cfg.allin and behind[i] > 1e-9)


class Exit:
    """The end of the root street where the rest of the hand is played by hand buckets (MWConfig.later): its reach
    going in (X, per combo) and its values coming back (V) are filled by MultiwaySolver._deep_pass."""
    __slots__ = ("live", "folded", "pot", "c", "behind", "X", "V", "F")

    def __init__(self, live, folded, pot, c, behind):
        self.live, self.folded, self.pot, self.c, self.behind = live, frozenset(folded), pot, list(c), behind
        self.X = self.V = None
        self.F = {}         # a tournament's: each folded player's value (MWConfig.icm)


class Builder:
    def __init__(self, hands: Hands, cfg: MWConfig, n_players: int, dead: set):
        self.h, self.cfg, self.np, self.dead = hands, cfg, n_players, set(dead)
        self.rng = random.Random(cfg.seed)
        self.decisions: List[Decision] = []
        self.exits: List[Exit] = []
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
        chips behind at the street's start (one number: the same for all), `pending` the players still to act. A
        player all-in at the root (MWConfig.allin) has its bet behind it; nobody has more than another can match."""
        stk = list(_per(stack, self.np))
        for a in self.cfg.allin:
            stk[a] = bets[a]
        stk = tuple(min(stk[p], max(stk[q] for q in range(self.np) if q != p)) for p in range(self.np))
        self.stacks = stk
        c = list(bets)
        self._at_root = True
        return self._decision([tuple(board)], pot_start, stk, list(bets), c, to_act, n_raises,
                              set(pending), frozenset())

    def _next(self, p, pending):
        return _next_player(self.np, p, pending)

    def _close(self, boards, pot_start, stack, bets, c, folded):
        street_len = len(boards[0])
        # (uncalled chips go back to their bettor: a lone winner's, or a bet over a call all-in for less)
        live, pot, c2, behind = _street_end(self.np, pot_start, stack, bets, c, folded)
        # (fewer than two live players with chips behind (the rest all-in) have nobody to bet against: the cards are
        # dealt out)
        alone = len(_acting(self.cfg, live, behind)) < 2
        if self.cfg.later == "buckets" and street_len < 5 and len(live) > 1:
            if alone:
                behind = (0.0,) * self.np
            e = Exit(live, folded, pot, c2, behind)
            self.exits.append(e)
            self.n_nodes += 1
            return e
        if len(live) == 1:
            return self._terminal(boards, live, pot, c2, "fold")
        if street_len == 5:
            return self._terminal(boards, live, pot, c2, "show")
        if alone or street_len - self.root_len + 1 >= self.cfg.depth:
            # all-in, or the last street the tree bets on: the rest is dealt out (an equity leaf)
            return self._runout(boards, live, pot, c2)
        return self._chance(boards, pot, behind, c2, folded)

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
        pending = set(_acting(self.cfg, [i for i in range(self.np) if i not in folded], behind))
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
        street = STREET[len(boards[0])]
        names, kids, amts = [], [], []
        # (root_raise: the root's player gets one raise more than max_raises; the nodes below keep the count)
        at_root = getattr(self, "_at_root", False)
        extra = 1 if at_root and self.cfg.root_raise else 0
        self._at_root = False
        cfg = self.cfg
        if at_root and cfg.root_jam:
            import dataclasses
            cfg = dataclasses.replace(cfg, allin_spr=1e9)
        for name, amt, nb, nc, nf, npend, raised in _actions(cfg, self.np, street, pot_start, stack, bets, c, p,
                                                              n_raises - extra, pending, folded):
            raised = raised + extra if raised else 0
            live = [i for i in range(self.np) if i not in nf]
            if len(live) == 1 or not npend:
                # (a lone player with chips facing nothing more to call: the street is over)
                kid = self._close(boards, pot_start, stack, nb, nc, nf)
            else:
                kid = self._decision(boards, pot_start, stack, nb, nc, self._next(p, npend), n_raises + raised, npend, nf)
            names.append(name); kids.append(kid); amts.append(amt)
        self.n_nodes += 1
        d = Decision(self.boards(boards), p, names, kids, pot_start + sum(bets), list(c), street)
        d.amounts = amts
        d.folded = frozenset(folded)
        self.decisions.append(d)
        return d


# ------------------------------------------------------------------ later streets by hand buckets
# MWConfig.later = "buckets" (the heads-up solver's flop model, solver.Config river_buckets / turn_buckets, for three):
# the root street is played per combo; each line that ends it with two or more players live (an Exit) deals the next
# card as one card per class (card_classes), and from there each player's range on that card is K hand buckets
# (equal shares of its root range, ordered by equity against the other two) that bet the rest of the hand: from a
# flop root the turn, then the river on the turn's boards without its card; from a turn root the river. Showdowns
# are tensors over the bucket triples (BucketTables). The later streets' subtrees after different root lines with the
# same shape are merged into one batch (_Group: a row per line and card class), so a flop solve walks a few hundred
# nodes a pass instead of ~11,000.

def card_classes(board, dead) -> list:
    """The next card's classes: [(the card played, the class's cards)], as solver.Ctx.turn_classes: one rank's cards,
    except that a suit with two or more cards on the board has its own class per rank; the rest of the rank is played
    as one of its suits, rotating over the board's suits rank by rank."""
    avail = [c for c in range(52) if c not in dead and c not in board]
    fs = [c % 4 for c in board]
    flushy = {x for x in range(4) if fs.count(x) >= 2}
    order = [x for x in sorted(set(fs), key=lambda x: -fs.count(x)) + [x for x in range(4) if x not in fs]
             if x not in flushy]
    out = []
    for r in range(13):
        cs = [c for c in avail if c // 4 == r]
        out += [(c, (c,)) for c in cs if c % 4 in flushy]
        rest = [c for c in cs if c % 4 not in flushy]
        if rest:
            pref = order[r % len(order):] + order[:r % len(order)] if order else list(range(4))
            out.append((next(c for x in pref + list(range(4)) for c in rest if c % 4 == x), tuple(rest)))
    return sorted(out)


class _TD:            # a later street's decision, one line's numbers (templates are zipped into batches)
    __slots__ = ("player", "names", "children", "live", "pot", "c")

    def __init__(self, player, names, children, live=(), pot=0.0, c=()):
        self.player, self.names, self.children = player, names, children
        self.live, self.pot, self.c = live, pot, c        # (who is live, the pot and the chips in at it)


class _TT:            # ... terminal
    __slots__ = ("live", "pot", "c", "kind")

    def __init__(self, live, pot, c, kind):
        self.live, self.pot, self.c, self.kind = live, pot, c, kind


class _TX:            # ... the street's end with another street to come
    __slots__ = ("live", "folded", "pot", "c", "behind")

    def __init__(self, live, folded, pot, c, behind):
        self.live, self.folded, self.pot, self.c, self.behind = live, folded, pot, c, behind


def _template(cfg, npl, street, pot, c, behind, folded):
    """One line's later street from its start, as plain numbers: everyone live with chips acts from the first, each
    with its `behind` chips; a street that closes with two or more live ends in a showdown on the river (or with
    fewer than two left to bet: all-in), else in a _TX to the next street."""
    behind = _per(behind, npl)
    live = tuple(i for i in range(npl) if i not in folded)
    acting = _acting(cfg, live, behind)
    if len(acting) < 2:
        return _TT(live, pot, list(c), "show")

    def dec(bets, cc, p, n_raises, pending, fl):
        names, kids = [], []
        for name, amt, nb, nc, nf, npend, raised in _actions(cfg, npl, street, pot, behind, bets, cc, p, n_raises,
                                                              pending, fl):
            lv = [i for i in range(npl) if i not in nf]
            if len(lv) == 1 or not npend:
                lv2, pot2, c2, b2 = _street_end(npl, pot, behind, nb, nc, nf)
                if len(lv2) == 1:
                    kid = _TT(lv2, pot2, c2, "fold")
                elif street == "river" or len(_acting(cfg, lv2, b2)) < 2:
                    kid = _TT(lv2, pot2, c2, "show")
                else:
                    kid = _TX(lv2, frozenset(nf), pot2, c2, b2)
            else:
                kid = dec(nb, nc, _next_player(npl, p, npend), n_raises + raised, npend, nf)
            names.append(name); kids.append(kid)
        return _TD(p, names, kids, tuple(i for i in range(npl) if i not in fl), pot + sum(bets), list(cc))
    return dec([0.0] * npl, list(c), acting[0], 0, set(acting), frozenset(folded))


def _tlive(t):
    """The players live at a template's start: its root's and every node's below (none folds before it starts)."""
    if isinstance(t, (_TT, _TX)):
        return t.live
    return tuple(sorted({t.player} | set(_tlive_all(t))))


def _tlive_all(t):
    if isinstance(t, _TD):
        out = {t.player}
        for k in t.children:
            out |= _tlive_all(k)
        return out
    return set(t.live)


def _sig(t):
    if isinstance(t, _TD):
        return ("d", t.player, tuple(nm.split()[0] for nm in t.names), tuple(_sig(k) for k in t.children))
    if isinstance(t, _TT):
        return ("t", t.kind, t.live)
    return ("x", t.live)


class BDecision(Decision):
    """A later street's decision by hand buckets: regrets per (row, action, bucket)."""

    def __init__(self, player, names, children, rows, K):
        self.player, self.names, self.children = player, names, children
        self.bd = None
        self.street = None
        self.regret = np.zeros((rows, len(children), K), np.float32)
        self.ssum = np.zeros((rows, len(children), K), np.float32)
        # its player's fold: the action's index, and a leaf standing for the subtree below it in that player's own
        # passes (a folded player's value is minus its chips in times the deals' weight, whatever follows)
        self.fold = names.index("fold") if "fold" in names else None
        self.leaf = None
        if self.fold is not None:
            t = _first_term(children[self.fold])
            self.leaf = BTerminal(tuple(j for j in t.live if j != player), None, t.c, None, "fold")


class BTerminal:
    __slots__ = ("live", "pot", "c", "net", "kind", "X", "V", "fctx", "fm", "side", "W")

    def __init__(self, live, pot, c, net, kind):
        self.live, self.pot, self.c, self.net, self.kind = live, pot, c, net, kind
        self.X = self.V = self.fctx = None
        self.fm, self.side = None, live      # the main pot's share per row and the side pot's players (_split)
        self.W = None       # a tournament's weights (np, 3, rows): the deals' weight's, the main pot's, the side pot's


def _bterminals(node):
    """The later-street subtree's terminals (not its exits' streets)."""
    if isinstance(node, BTerminal):
        return [node]
    if isinstance(node, BDecision):
        return [t for ch in node.children for t in _bterminals(ch)]
    return []


def _first_term(node):
    """A terminal below the node (depth first)."""
    if isinstance(node, (BTerminal, BExit)):
        return node
    for ch in (node.children if isinstance(node, BDecision) else ()):
        t = _first_term(ch)
        if t is not None:
            return t
    return None


class BExit:
    """A later street's end with another to come: the members' numbers (_TX), its reach (X) and values (V)."""
    __slots__ = ("live", "members", "c", "X", "V", "F")

    def __init__(self, live, members, C):
        self.live, self.members = live, members
        self.c = np.repeat(np.array([t.c for t in members], dtype=np.float64).T, C, axis=1)   # (np, rows)
        self.X = self.V = None
        self.F = None       # a tournament's: the folded players' values (np, rows)


class _Group:
    """Later-street subtrees of one shape, batched: blocks [(exit, member)] of C rows each (one per card class)."""

    def __init__(self, tree, blocks, live):
        self.tree, self.blocks, self.live = tree, blocks, live
        self.X = self.V = None
        self.F = None       # a tournament's: the folded players' values per row (MWConfig.icm)
        self.skip = False
        self.c = _first_term(tree).c if not isinstance(tree, BExit) else None   # (a folded player's chips in, per row)


def _zip(ts, C, K, cfg, sink):
    t0 = ts[0]
    if isinstance(t0, _TD):
        kids = [_zip([t.children[a] for t in ts], C, K, cfg, sink) for a in range(len(t0.children))]
        d = BDecision(t0.player, list(t0.names), kids, len(ts) * C, K)
        if cfg.icm is not None:     # (the numbers at the decision, for a fold's value)
            d.live = t0.live
            d.pot_at = np.repeat(np.array([t.pot for t in ts], dtype=np.float64), C)
            d.c_at = np.repeat(np.array([t.c for t in ts], dtype=np.float64).T, C, axis=1)
        sink.append(d)
        return d
    if isinstance(t0, _TX):
        return BExit(t0.live, list(ts), C)
    pot = np.repeat(np.array([t.pot for t in ts], dtype=np.float64), C)
    c = np.repeat(np.array([t.c for t in ts], dtype=np.float64).T, C, axis=1)       # (np, rows)
    net = pot - np.array([cfg.rake(float(x)) for x in pot])
    return BTerminal(t0.live, pot, c, net, t0.kind)


class BucketTables:
    """Hand buckets and showdown tensors below the root street, per class of the next card (card_classes).

    For class t (its card played as `rep`, standing for `size` cards), each player's root range on the board plus
    rep (weights w_i: the root reach times the share of the class's cards the combo leaves live) is split into K
    equal shares ordered by its equity against the other two ranges (over every river from a flop root). The tensors
    are over bucket triples (a0, a1, a2): J, the weight of the compatible deals (w w w summed over the triples of
    combos in the buckets, card removal pairwise with the third pair's overlap share taken exactly per combo, as the
    equity leaves do); and each live set's pot shares (a player's share summed the same way, over every river card
    from a flop root, the river's own showdown from a turn root), scaled per cell so the live players' shares add up
    to J: the bucket game is constant-sum. Divided by the buckets' weights, a terminal's value for player p's bucket
    a is sum over (b, c) of the others' bucket reaches times net x share - chips put in x J, which is the per-combo
    value with each bucket's reach spread inside it like its root range. A combo's value is its bucket's."""

    def __init__(self, s: "MultiwaySolver", K: int, classes: bool, step: int):
        h, npl = s.hands, s.np
        n = h.n
        dead = set(s.board) | s.builder.dead
        avail = [c for c in range(52) if c not in dead]
        cls = card_classes(s.board, dead) if classes else [(c, (c,)) for c in avail]
        C = len(cls)
        unseen = 52 - len(dead) - 2 * npl
        self.C, self.K, self.np, self.cls = C, K, npl, cls
        mem = h.mem
        masks = np.zeros((C, n))
        w = np.zeros(C)
        # a combo holding the card played sees the class's other cards: it plays as its image with that card
        # swapped for another of the class (the image's bucket); with no such combo in play, it sits the class out
        f2a = np.full(N_FULL, -1, dtype=np.intp)
        f2a[h.active] = np.arange(n)
        image = np.tile(np.arange(n), (C, 1))
        for t, (rep, cs) in enumerate(cls):
            masks[t] = (len(cs) - mem[:, list(cs)].sum(axis=1)) / len(cs)
            w[t] = len(cs) / unseen
            for i in np.flatnonzero(mem[:, rep] > 0):
                y = h.cb[i] if h.ca[i] == rep else h.ca[i]
                alt = [c for c in cs if c != rep and c != y]
                j = f2a[_PAIR[alt[0], y]] if alt else -1
                if j >= 0:
                    image[t, i] = j
                else:
                    masks[t, i] = 0.0
        self.masks = masks
        self.wmask = masks * w[:, None]
        R0 = s.reach0[:, 0, :]
        self.ids = np.zeros((npl, C, n), dtype=np.intp)
        keys = [("J",)] + [("S", lv, p) for lv in _live_sets(npl) for p in lv]
        T = {k: np.zeros((C,) + (K,) * npl) for k in keys}
        t_cls = time.time()
        for t, (rep, _) in enumerate(cls):
            s.check_deadline(t / C, t_cls)
            base = tuple(s.board) + (rep,)
            if len(base) == 4:
                rv = [c for c in range(52) if c not in dead and c != rep]
                rv = rv[base[-1] % step::step] if step > 1 else rv
                sd = [base + (r,) for r in rv]
            else:
                sd = [base]
            wt = R0 * masks[t] * (1.0 - mem[:, rep])
            ids, tens = (self._class if npl <= 3 else self._class_n)(s, base, sd, wt, K)
            self.ids[:, t] = ids[:, image[t]]
            for k in keys:
                T[k][t] = tens[k]
        # per unit of bucket reach (the tensors' weights: the combos live on the card played)
        Wb = np.zeros((npl, C, K))
        for i in range(npl):
            for t, (rep, _) in enumerate(cls):
                Wb[i, t] = np.bincount(self.ids[i, t], weights=R0[i] * masks[t] * (1.0 - mem[:, rep]), minlength=K)[:K]
        den = Wb[0]
        for i in range(1, npl):
            den = den[..., None] * Wb[i].reshape((C,) + (1,) * i + (K,))
        inv = np.where(den > 0, 1.0 / np.where(den > 0, den, 1.0), 0.0)
        self.OH = [np.eye(K)[self.ids[i]] for i in range(npl)]                 # (C, n, K) one-hot
        # the deals' total weight below the root scaled to the root's own (exact) mass: pairwise removal and the
        # classes move it a few percent
        X0 = self.to_buckets(s.reach0)
        Jt = T[("J",)] * inv
        ax = "abcdefgh"[:npl]
        M_b = float((np.einsum(f"t{ax}," + ",".join(f"t{x}" for x in ax) + "->t", Jt, *X0[:npl]) * w).sum())
        kappa = s.total_mass() / M_b if M_b > 0 else 1.0
        self.kappa = kappa
        inv *= kappa
        self.perm = {}
        self.tp = {}     # the compiled core's (gto/mwcore.py): per player p, (C, K_o0, K_o1, K_p), p's buckets innermost
        for k in keys:
            Tk = T[k] * inv
            for p in (range(npl) if k[0] == "J" else [k[2]]):
                o = [j for j in range(npl) if j != p]
                self.tp[(k[:2], p)] = np.ascontiguousarray(np.transpose(Tk, (0,) + tuple(j + 1 for j in o) + (p + 1,)),
                                                           dtype=np.float32)
                if npl > 3:      # (contract_n: tp only)
                    continue
                if npl == 3:     # (t, a_r, a_p a_q): the buckets of r (either other player) first (contract, reduce)
                    for r in o:
                        q = o[0] if r == o[1] else o[1]
                        Tp = np.transpose(Tk, (0, r + 1, p + 1, q + 1)).reshape(C, K, K * K)
                        self.perm[(k[:2], p, r)] = np.ascontiguousarray(Tp, dtype=np.float32)
                else:            # (t, a_q, a_p)
                    self.perm[(k[:2], p, o[0])] = np.ascontiguousarray(np.transpose(Tk, (0, o[0] + 1, p + 1)),
                                                                        dtype=np.float32)

    @staticmethod
    def _buckets(eq, wt, K):
        """Bucket ids: K equal shares of the weight wt ordered by eq; combos with no weight by their equity."""
        n = len(eq)
        ids = np.zeros(n, dtype=np.intp)
        idx = np.flatnonzero(wt > 0)
        if not len(idx):
            return ids
        order = idx[np.argsort(eq[idx], kind="stable")]
        ws = wt[order]
        cum = np.cumsum(ws)
        b = np.minimum(((cum - ws / 2) / cum[-1] * K).astype(np.intp), K - 1)
        ids[order] = b
        top = np.full(K, -np.inf)
        np.maximum.at(top, b, eq[order])
        top = np.maximum.accumulate(top)
        rest = np.flatnonzero(wt <= 0)
        ids[rest] = np.minimum(np.searchsorted(top, eq[rest], "left"), K - 1)
        return ids

    def _class(self, s, base, sd, wt, K):
        h, npl, n = s.hands, s.np, s.hands.n
        bd = Boards(h, sd)
        bb = Boards(h, [base])
        rm = 1.0 - h.mem[:, [b[4] for b in sd]].T if len(base) == 4 else np.ones((1, n))   # (R, n): the river's live
        x = [wt[i][None, :] * rm for i in range(npl)]
        kle = [bd.k_le(xi) for xi in x]
        ku = [bd.k_u(xi)[0] for xi in x]
        ids = np.zeros((npl, n), dtype=np.intp)
        for i in range(npl):
            o = [j for j in range(npl) if j != i]
            if npl == 3:
                (lq, eq_), (lr, er) = kle[o[0]], kle[o[1]]
                num = (((lq + 0.5 * eq_) * (lr + 0.5 * er) + eq_ * er / 12.0) * rm).sum(axis=0)
                den = (ku[o[0]] * ku[o[1]] * rm).sum(axis=0)
            else:
                lq, eq_ = kle[o[0]]
                num = ((lq + 0.5 * eq_) * rm).sum(axis=0)
                den = (ku[o[0]] * rm).sum(axis=0)
            ids[i] = self._buckets(np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0), wt[i], K)
        OH = [np.eye(K)[ids[i]] for i in range(npl)]                           # (n, K)
        Wk, Ek, Uk, Ub = [], [], [], []
        for i in range(npl):
            xb = x[i][:, :, None] * OH[i][None]                                  # (R, n, K)
            kl, ke = bd.k_le_multi(xb)
            Wk.append(kl + 0.5 * ke); Ek.append(ke); Uk.append(bd.k_u_multi(xb))
            Ub.append(bb.k_u_multi((wt[i][:, None] * OH[i])[None])[0])          # (n, K) on the base board
        Rb = wt[:, None, :]
        rho = {}
        if npl == 3:
            for q, r in ((0, 1), (0, 2), (1, 2)):
                kq, kr = bb.k_u(Rb[q]), bb.k_u(Rb[r])
                rho[(q, r)] = MultiwaySolver._ratio(MultiwaySolver._mass(bb, Rb, q, r, kq, kr), kq[0], kr[0])[0]
        persp = {}
        for p in range(npl):
            o = [j for j in range(npl) if j != p]
            rh = rho[tuple(o)] if npl == 3 else np.ones(n)
            hwB = wt[p] * rh
            hwR = hwB[None, :] * rm
            if npl == 3:
                q, r = o
                J = np.zeros((K, K, K)); S3 = np.zeros((K, K, K)); Sq = np.zeros((K, K, K)); Sr = np.zeros((K, K, K))
            else:
                (q,) = o
                J = np.zeros((K, K)); S2 = np.zeros((K, K))
            for a in range(K):
                idx = np.flatnonzero((ids[p] == a) & (wt[p] > 0))
                if not len(idx):
                    continue
                f = lambda A: A[:, idx, :].reshape(-1, K)
                hB, hR = hwB[idx, None], hwR[:, idx, None]
                if npl == 3:
                    J[a] = (Ub[q][idx] * hB).T @ Ub[r][idx]
                    Wq_ = (Wk[q][:, idx, :] * hR).reshape(-1, K)
                    Wr_ = (Wk[r][:, idx, :] * hR).reshape(-1, K)
                    S3[a] = Wq_.T @ f(Wk[r]) + ((Ek[q][:, idx, :] * hR).reshape(-1, K)).T @ f(Ek[r]) / 12.0
                    Sq[a] = Wq_.T @ f(Uk[r])         # p's share with q live, r folded
                    Sr[a] = (f(Uk[q]).T @ Wr_)       # p's share with r live, q folded (axes q, r)
                else:
                    J[a] = (Ub[q][idx] * hB).sum(axis=0)
                    S2[a] = (Wk[q][:, idx, :] * hR).sum(axis=(0, 1))
            if npl == 3:
                ax = [p, q, r]
                canon = lambda X: np.transpose(X, [ax.index(k) for k in range(3)])
                persp[p] = {"J": canon(J), "S3": canon(S3), ("S2", q): canon(Sq), ("S2", r): canon(Sr)}
            else:
                canon = (lambda X: X) if p == 0 else (lambda X: X.T)
                persp[p] = {"J": canon(J), ("S2", q): canon(S2)}
        Jc = sum(persp[p]["J"] for p in range(npl)) / npl
        out = {("J",): Jc}

        def norm(parts):
            tot = sum(parts)
            ok = tot > 0
            return [np.where(ok, X * Jc / np.where(ok, tot, 1.0), Jc / len(parts)) for X in parts]
        if npl == 3:
            for p, X in zip(range(3), norm([persp[p]["S3"] for p in range(3)])):
                out[("S", (0, 1, 2), p)] = X
        for i, j in [(i, j) for i in range(npl) for j in range(i + 1, npl)]:
            si, sj = norm([persp[i][("S2", j)], persp[j][("S2", i)]])
            out[("S", (i, j), i)] = si
            out[("S", (i, j), j)] = sj
        return ids, out

    def _class_n(self, s, base, sd, wt, K):
        """_class for four or more players: a hand's share of a pot against the others' buckets is the product of its
        heads-up shares against each (weaker + tied / 2: the others' hands independent given its own, each pair of them
        overlap-corrected per combo on the class's board, as _term_pw), a folded player's hand only blocking cards;
        bucketed by that share against everyone's whole range (over every river from a flop root)."""
        h, npl, n = s.hands, s.np, s.hands.n
        bd = Boards(h, sd)
        bb = Boards(h, [base])
        rm = 1.0 - h.mem[:, [b[4] for b in sd]].T if len(base) == 4 else np.ones((1, n))   # (R, n): the river's live
        x = [wt[i][None, :] * rm for i in range(npl)]
        kle = [bd.k_le(xi) for xi in x]
        ku = [bd.k_u(xi)[0] for xi in x]
        Rb = wt[:, None, :]
        kub = [bb.k_u(Rb[i]) for i in range(npl)]
        rho = {(q, r): MultiwaySolver._ratio(MultiwaySolver._mass(bb, Rb, q, r, kub[q], kub[r]), kub[q][0], kub[r][0])[0]
               for q in range(npl) for r in range(q + 1, npl)}
        corr = []
        for p in range(npl):
            c = np.ones(n)
            for (q, r), v in rho.items():
                if p not in (q, r):
                    c = c * v
            corr.append(c)
        ids = np.zeros((npl, n), dtype=np.intp)
        for i in range(npl):
            num, den = rm.copy(), rm.copy()
            for o in range(npl):
                if o != i:
                    num = num * (kle[o][0] + 0.5 * kle[o][1])
                    den = den * ku[o]
            num, den = num.sum(axis=0), den.sum(axis=0)
            ids[i] = self._buckets(np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0), wt[i], K)
        OH = [np.eye(K)[ids[i]] for i in range(npl)]                           # (n, K)
        Wk, Uk, Ub = [], [], []
        for i in range(npl):
            xb = x[i][:, :, None] * OH[i][None]                                  # (R, n, K)
            kl, ke = bd.k_le_multi(xb)
            Wk.append(kl + 0.5 * ke); Uk.append(bd.k_u_multi(xb))
            Ub.append(bb.k_u_multi((wt[i][:, None] * OH[i])[None])[0])          # (n, K) on the base board
        sets = _live_sets(npl)
        persp = {}
        for p in range(npl):
            o = [j for j in range(npl) if j != p]
            hwB = wt[p] * corr[p]
            hwR = hwB[None, :] * rm
            mine = [lv for lv in sets if p in lv]
            J = np.zeros((K,) * npl)
            S = {lv: np.zeros((K,) * npl) for lv in mine}
            for a in range(K):
                idx = np.flatnonzero((ids[p] == a) & (wt[p] > 0))
                if not len(idx):
                    continue
                J[a] = _outer_sum([Ub[o[0]][idx] * hwB[idx, None]] + [Ub[j][idx] for j in o[1:]])
                hR = hwR[:, idx].reshape(-1, 1)
                for lv in mine:
                    fs = [(Wk[j] if j in lv else Uk[j])[:, idx, :].reshape(-1, K) for j in o]
                    fs[0] = fs[0] * hR
                    S[lv][a] = _outer_sum(fs)
            axes = [p] + o
            perm = [axes.index(k) for k in range(npl)]
            persp[p] = {"J": np.transpose(J, perm), **{lv: np.transpose(S[lv], perm) for lv in mine}}
        Jc = sum(persp[p]["J"] for p in range(npl)) / npl
        out = {("J",): Jc}
        for lv in sets:
            parts = [persp[p][lv] for p in lv]
            tot = sum(parts)
            ok = tot > 0
            for p, X in zip(lv, parts):
                out[("S", lv, p)] = np.where(ok, X * Jc / np.where(ok, tot, 1.0), Jc / len(lv))
        return ids, out

    # -- the root's per-combo reach into buckets and the buckets' values back
    def to_buckets(self, R):
        """(np, 1, n) per combo -> (np, C, K) per class and bucket."""
        return np.stack([np.matmul((R[i, 0][None, :] * self.masks)[:, None, :], self.OH[i])[:, 0]
                         for i in range(self.np)])

    def from_buckets(self, V, p=None):
        """(C, K) for player p (or (np, C, K)) -> (1, n) (or (np, 1, n)): each class's value of the combo's bucket,
        weighted by the class's cards the combo leaves live."""
        if p is not None:
            return (np.take_along_axis(V, self.ids[p], axis=1) * self.wmask).sum(axis=0)[None, :]
        return np.stack([self.from_buckets(V[i], i) for i in range(self.np)])

    def contract(self, key, p, X):
        """Player p's (rows, K): the tensor summed over the other players' bucket reaches X (np, rows, K)."""
        C, K = self.C, self.K
        rows = X.shape[1]
        nb = rows // C
        o = [j for j in range(self.np) if j != p]
        if self.np > 3:
            # (C, K_o0, .., K_p): summed over each other player's reach in turn
            T = self.tp[(key, p)]
            X = X.astype(np.float32)
            Xo = [X[j].reshape(nb, C, K).transpose(1, 0, 2) for j in o]       # (C, nb, K)
            Y = np.matmul(Xo[0], T.reshape(C, K, -1))                         # (C, nb, K^(np-1))
            for Xj in Xo[1:]:
                Y = np.matmul(Xj[:, :, None, :], Y.reshape(C, nb, K, -1))[:, :, 0]
            return Y.transpose(1, 0, 2).reshape(rows, K).astype(np.float64)
        T = self.perm[(key, p, o[-1])]
        X = X.astype(np.float32)        # (the tensors are float32: half the arithmetic of a float64 product)
        Xq = X[o[0]].reshape(nb, C, K).transpose(1, 0, 2)                      # (C, nb, K)
        if self.np == 2:
            out = np.matmul(Xq, T)                                              # (C, nb, K)
        else:
            Xr = X[o[1]].reshape(nb, C, K).transpose(1, 0, 2)
            Y = np.matmul(Xr, T).reshape(C, nb, K, K)                           # summed over r: (t, n, a_p, a_q)
            out = np.matmul(Y, Xq[..., None])[..., 0]
        return out.transpose(1, 0, 2).reshape(rows, K).astype(np.float64)

    def reduce(self, key, p, f, Xf):
        """Three players, f's reach fixed (folded): the tensor summed over f's buckets, (C, nb, K_p, K_q) for the
        remaining pair (p, q): a terminal's contraction is then a K x K matrix times q's reach (apply)."""
        C, K = self.C, self.K
        nb = Xf.shape[0] // C
        Y = np.matmul(Xf.astype(np.float32).reshape(nb, C, K).transpose(1, 0, 2), self.perm[(key, p, f)])
        return Y.reshape(C, nb, K, K)

    def apply(self, M, Xq):
        """M (C, nb, K, K) from reduce, q's reach (rows, K) over a whole number of M's rows -> (rows, K)."""
        C, nb, K = M.shape[0], M.shape[1], self.K
        rows = Xq.shape[0]
        m = rows // (nb * C)
        Xq = Xq.astype(np.float32).reshape(m, nb, C, K).transpose(2, 0, 1, 3)  # (C, m, nb, K)
        out = np.matmul(M[:, None], Xq[..., None])[..., 0]                    # (C, m, nb, K)
        return out.transpose(1, 2, 0, 3).reshape(rows, K).astype(np.float64)


def _bexits(node):
    if isinstance(node, BExit):
        return [node]
    if isinstance(node, BDecision):
        return [x for ch in node.children for x in _bexits(ch)]
    return []


def _live_sets(npl):
    """The sets of two or more players a pot can be contested by (largest first)."""
    if npl == 3:
        return [(0, 1, 2), (0, 1), (0, 2), (1, 2)]
    return [lv for k in range(npl, 1, -1) for lv in itertools.combinations(range(npl), k)]


def _outer_sum(fs):
    """sum over rows of the outer product of the rows of fs ((rows, K) each): shape (K,) * len(fs)."""
    K = fs[0].shape[1]
    P = fs[0]
    for f in fs[1:-1]:
        P = (P[:, :, None] * f[:, None, :]).reshape(P.shape[0], -1)
    return (P.T @ fs[-1]).reshape((K,) * len(fs))


# ------------------------------------------------------------------ solver
class MultiwaySolver:
    def __init__(self, board, ranges: Sequence[np.ndarray], pot: float, stack: float, bets=None, to_act=0,
                 n_raises=0, pending=None, cfg: Optional[MWConfig] = None, dead: Sequence[int] = (),
                 include: Sequence[int] = (), deadline: Optional[float] = None):
        """`ranges`: one (1326,) weight vector per player, in postflop acting order. `pot`: the chips in the
        middle at the start of this street; `stack`: each player's chips behind at the start of this street
        (one number: the solver plays equal stacks); `bets`: chips each put in this street so far; `to_act`:
        who acts at the root; `pending`: who still has to act on this street (default everyone from to_act on
        when nobody has bet, else everyone who hasn't matched the top bet). `include`: combos (indices into the 1326)
        to solve even with no weight in any range (a hand's strategy is then the regret-matching one: see
        hand_strategy). `deadline` (a time.time()): the bucket tables still building then raise BuildTooSlow."""
        t_init = time.time()
        self.deadline, self._t_init = deadline, t_init
        self.cfg = cfg or MWConfig()
        self.board = tuple(board)
        self.np = len(ranges)
        assert 2 <= self.np <= MAX_PLAYERS
        # (four or more players: the streets after the root's dealt out, equity leaves; no hand buckets)
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
        pending = set(pending) - set(self.cfg.allin)
        # pairwise terminals (_term_pw): four or more players, or three with two all-in (layered pots); else exact
        self._pw = self.np > 3 or len(self.cfg.allin) > 1
        assert not self.cfg.allin or self.np >= 3, "allin: three players or more"
        # (4+ players by buckets: at most one all-in, a main pot and a side pot, as the bucket terminals play them)
        assert not self._pw or self.cfg.later == "deal" or (self.np > 3 and len(self.cfg.allin) <= 1), \
            "pairwise terminals by buckets: four or more players, at most one all-in"
        self.pot_start, self.stack = pot, stack
        b = Builder(self.hands, self.cfg, self.np, dead_all - set(board))
        b.root_len = len(self.board)
        self.builder = b
        self.root = b.build(self.board, pot, stack, bets, to_act, n_raises, pending)
        self.stacks = b.stacks           # (each player's chips behind at the root, as the tree plays them)
        self.root_bd = b.boards([self.board])
        self.decisions = b.decisions
        self.iters = 0
        # alternating updates: a player's reach at a node is its own strategies' product, so it only changes in the
        # pass that updates that player. The showdown tables built from one opponent's reach (and the strategies) are
        # kept by player and version (how many times its regrets were updated) for the next pass that needs them
        # unchanged: of each player's two opponent passes an iteration, player 0's (passes 1 and 2) and the last
        # player's (0 and 1) share one version. Pair terms (two opponents' reaches) change every pass.
        self._ver = [0] * self.np
        self._tmemo: Dict[tuple, tuple] = {}
        self._smemo: Dict[int, tuple] = {}
        self.seconds = 0.0
        self.out_of_time = False      # a run ended at its time budget (the advisor then doesn't run it again)
        self.levels: List[List[_Group]] = []
        self.tabs: Optional[BucketTables] = None
        if b.exits:
            self._build_later(pot)
        if self.cfg.icm is not None:
            self._icm_setup()
        self.build_seconds = time.time() - t_init     # the tree and the bucket tables (part of a solve's budget)

    def check_deadline(self, done: float = 0.0, since: float = 0.0):
        """BuildTooSlow past the deadline, or once `done` (a share of a step begun at `since`, a quarter at least)
        says the step ends past it."""
        if self.deadline is None:
            return
        now = time.time()
        if now > self.deadline or done >= 0.25 and now + (now - since) * (1.0 - done) / done > self.deadline:
            raise BuildTooSlow(now - self._t_init)

    # -- later streets by hand buckets
    def _build_later(self, pot):
        cfg = self.cfg
        K = cfg.buckets
        self.tabs = tb = BucketTables(self, K, cfg.classes, cfg.bucket_river_step)
        C = tb.C
        street = NEXT_STREET[STREET[len(self.board)]]
        members = [(e, 0, _template(cfg, self.np, street, e.pot, e.c, e.behind, e.folded)) for e in self.builder.exits]
        while members:
            self.check_deadline()
            groups: Dict[tuple, list] = {}
            for m in members:
                groups.setdefault(_sig(m[2]), []).append(m)
            lvl, sink = [], []
            for mem in groups.values():
                lvl.append(_Group(_zip([t for _, _, t in mem], C, K, cfg, sink), [(e, j) for e, j, _ in mem],
                                  _tlive(mem[0][2])))
                for t in _bterminals(lvl[-1].tree):
                    self._split(t)
            self.levels.append(lvl)
            self.decisions.extend(sink)
            self.builder.n_nodes += len(sink)
            if street == "river":
                break
            street = NEXT_STREET[street]
            members = []
            for g in lvl:
                for bx in _bexits(g.tree):
                    members += [(bx, j, _template(cfg, self.np, street, tx.pot, tx.c, tx.behind, tx.folded))
                                for j, tx in enumerate(bx.members)]

    # -- a tournament's payoffs
    def _lay(self, live, c, pot):
        """The pot's layers and who contests each (_layers), one layer when one player is live."""
        return self._layers(live, c, pot) if len(live) > 1 and pot > 1e-12 else [(pot, tuple(live))]

    def _icm_setup(self):
        """MWConfig.icm: every terminal's weights (Terminal.W, BTerminal.W: icm.MWPayoffs.weights), and the values of
        folding: at a root-street decision its player's (Decision.fold_v, action_evs' baseline), and for each player
        folded above a later street's node the pot as it stood at its fold (Exit.F, _Group.F, BExit.F, a BDecision's
        leaf): what the passes pay a folded player whatever follows (its own passes skip the subtree).
        Each player's values count from its folding at the root street's start (the pot as it stood going to one of
        the others; the root's player from its fold at the root), as chips count from the street's start: a constant a
        player, which moves no strategy, but the
        later streets' hand buckets keep a shift only on average over a bucket's combos (a combo's deals' weight is
        its bucket's), so values near 0 where a player folds keep its fold's value per combo near exact (from the
        start of the hand, 1.6 bb on a bubble, the fold's EV read 0.3 bb off for some combos). In a winner-take-all
        the weights are the chips' exactly (icm_base: the origins, (np,))."""
        raw, npl = self.cfg.icm, self.np
        terms, decs, exits = [], [], []
        zero = [0.0] * npl
        start = {q: (tuple(j for j in range(npl) if j != q), zero, self.pot_start) for q in range(npl)}
        if isinstance(self.root, Decision):     # (the root's player: from its fold there, which its EVs count from)
            q = self.root.player
            start[q] = (start[q][0], list(self.root.c), float(self.root.pot_now))
        base = np.array([raw.weights([(o, c, self._lay(o, c, pot))])[0][q, 0] for q, (o, c, pot) in start.items()])
        self.icm_base = base

        class _Pay:     # (the weights from each player's origin)
            @staticmethod
            def weights(states):
                out = []
                for W in raw.weights(states):
                    W = W.copy()
                    W[:, 0] -= base
                    out.append(W)
                return out
        pay = _Pay

        def walk(node, fst):
            if isinstance(node, Decision):
                live = tuple(i for i in range(npl) if i not in node.folded)
                st = (tuple(i for i in live if i != node.player), list(node.c), float(node.pot_now))
                decs.append((node, st))
                for a, ch in enumerate(node.children):
                    walk(ch, {**fst, node.player: st} if node.names[a] == "fold" else fst)
            elif isinstance(node, Chance):
                walk(node.child, fst)
            elif isinstance(node, Terminal):
                if node.lay is None:
                    node.lay = self._lay(node.live, node.c, node.pot)
                terms.append(node)
            elif isinstance(node, Exit):
                exits.append((node, fst))
        walk(self.root, {})
        fold_state = lambda st: (st[0], st[1], self._lay(st[0], st[1], st[2]))
        Ws = pay.weights([(t.live, t.c, t.lay) for t in terms])
        for t, W in zip(terms, Ws):
            t.W = W
        Ws = pay.weights([fold_state(st) for _, st in decs])
        for (d, _), W in zip(decs, Ws):
            d.fold_v = float(W[d.player, 0])
        for e, fst in exits:
            Ws = pay.weights([fold_state(fst[q]) for q in sorted(fst)])
            e.F = {q: float(W[q, 0]) for q, W in zip(sorted(fst), Ws)}
        if not self.levels:
            return
        C = self.tabs.C

        rows_memo = {}

        def row_states(node, live, c, pot):
            """A node's rows (c (np, rows), pot (rows)) as its distinct states for `live` contesting the pot, and each
            row's state (kept by node: the walk below asks again)."""
            if id(node) not in rows_memo:
                cols = np.round(np.concatenate([c, pot[None, :]], axis=0), 9)
                B = cols.shape[1] // C
                blk = cols.reshape(npl + 1, B, C) if B * C == cols.shape[1] else None
                if blk is not None and (blk == blk[:, :, :1]).all():   # (a root-street line's rows share its chips)
                    u, inv = np.unique(blk[:, :, 0], axis=1, return_inverse=True)
                    inv = np.repeat(np.asarray(inv).reshape(-1), C)
                else:
                    u, inv = np.unique(cols, axis=1, return_inverse=True)
                    inv = np.asarray(inv).reshape(-1)
                st = [(live, u[:npl, j], self._lay(live, u[:npl, j], float(u[npl, j]))) for j in range(u.shape[1])]
                rows_memo[id(node)] = (st, inv)
            return rows_memo[id(node)]

        w_memo = {}

        def row_weights(node, live, c, pot):
            """Per row: the weights for `live` contesting the row's pot (by distinct state) and each row's state."""
            st, inv = row_states(node, live, c, pot)
            if id(node) not in w_memo:
                w_memo[id(node)] = pay.weights(st)
            return w_memo[id(node)], inv

        def folder_live(node):
            return tuple(i for i in node.live if i != node.player)

        def collect(node, out):
            if isinstance(node, BTerminal):
                out.append((node, row_states(node, node.live, node.c, node.pot)[0]))
            elif isinstance(node, BDecision):
                if node.fold is not None:
                    out.append((node, row_states(node, folder_live(node), node.c_at, node.pot_at)[0]))
                for ch in node.children:
                    collect(ch, out)

        # every bucketed state priced in one batch (one ICM call over the table, not one a node)
        every = []
        for lvl in self.levels:
            for g in lvl:
                collect(g.tree, every)
        Ws = pay.weights([x for _, st in every for x in st])
        at = 0
        for node, st in every:
            w_memo[id(node)] = Ws[at:at + len(st)]
            at += len(st)

        def bwalk(node, F):
            if isinstance(node, BTerminal):
                Ws, inv = row_weights(node, node.live, node.c, node.pot)
                rows = len(node.pot)
                W = np.zeros((npl, 3, rows))
                for j, Wj in enumerate(Ws):
                    r = inv == j
                    L = Wj.shape[1] - 1
                    if L > 2:
                        raise ValueError("a bucketed terminal with three pots")
                    W[:, :L + 1, r] = Wj[:, :, None]
                for q in range(npl):
                    if q not in node.live:
                        W[q, 0], W[q, 1:] = F[q], 0.0
                if not np.isfinite(W).all():
                    raise ValueError("a tournament's terminal without a folded player's value")
                node.W = W
                return
            if isinstance(node, BExit):
                node.F = F
                return
            Ff = F
            if node.fold is not None:
                q = node.player
                Ws, inv = row_weights(node, folder_live(node), node.c_at, node.pot_at)
                Ff = F.copy()
                Ff[q] = np.array([W[q, 0] for W in Ws])[inv]
                lw = np.zeros((npl, 3, len(inv)))
                lw[q, 0] = Ff[q]
                node.leaf.W = lw
            for a, ch in enumerate(node.children):
                bwalk(ch, Ff if a == node.fold else F)

        for li, lvl in enumerate(self.levels):
            for g in lvl:
                F = np.full((npl, len(g.blocks) * C), np.nan)    # (a live player's: none)
                for i, (e, j) in enumerate(g.blocks):
                    if li == 0:
                        for q, v in e.F.items():
                            F[q, i * C:(i + 1) * C] = v
                    else:
                        F[:, i * C:(i + 1) * C] = e.F[:, j * C:(j + 1) * C]
                g.F = F
                bwalk(g.tree, F)

    def _fwd(self, node, R, mode):
        """Reaches down the root street to its Exits (the strategies the walk that follows uses)."""
        if isinstance(node, Exit):
            node.X = R
            return
        if not isinstance(node, Decision):
            return
        sig = node.strategy() if mode in ("alt", "cfr") else node.avg_strategy()
        q = node.player
        for a, ch in enumerate(node.children):
            Ra = R.copy(); Ra[q] = R[q] * sig[:, a, :]
            self._fwd(ch, Ra, mode)

    def _bfwd(self, node, X, mode, p, terms, fctx=None):
        """Reaches down a later-street subtree: kept at its nodes (with the strategy used), terminals into `terms`. In
        a pass for one player p (modes alt, br) the subtree below p's own fold is its leaf, and below another player's
        fold in a 3-way subtree the terminals carry fctx (the folder, its reach there, a key): see _bterms."""
        if isinstance(node, BTerminal):
            node.X, node.fctx = X, fctx
            terms.append(node)
            return
        if isinstance(node, BExit):
            node.X = X
            node.V = np.zeros(X.shape[1:] if p is not None else X.shape)
            return
        sig = node.strategy() if mode in ("alt", "cfr") else node.avg_strategy()
        node.X, node.sig = X, sig
        q = node.player
        for a, ch in enumerate(node.children):
            Xa = X.copy(); Xa[q] = X[q] * sig[:, a, :]
            if a == node.fold and p is not None:
                if q == p:
                    node.leaf.X, node.leaf.fctx = Xa, None
                    terms.append(node.leaf)
                    continue
                if fctx is None and self.np == 3:
                    self._fc += 1
                    self._bfwd(ch, Xa, mode, p, terms, (q, Xa[q], self._fc))
                    continue
            self._bfwd(ch, Xa, mode, p, terms, fctx)

    def _later(self, node, R, mode, p):
        """Fill the root street's Exits with the later streets' values (by buckets) for a walk from `node`."""
        self._fwd(node, R, mode)
        tb, C = self.tabs, self.tabs.C
        one = mode in ("alt", "br")
        pp = p if one else None
        self._fc = 0
        for li, lvl in enumerate(self.levels):
            for g in lvl:
                g.skip = one and p not in g.live
                if g.skip and li > 0:
                    continue        # (p folded above it: its parent's leaf or group stands for it)
                if li == 0:
                    g.X = np.concatenate([tb.to_buckets(e.X if e.X is not None else np.zeros_like(self.reach0))
                                          for e, _ in g.blocks], axis=1)
                else:
                    g.X = np.concatenate([e.X[:, j * C:(j + 1) * C] for e, j in g.blocks], axis=1)
                if g.skip:
                    # p folded on the root street: its value is minus its chips in times the deals' weight
                    g.V = (g.F[p] if g.F is not None else -g.c[p])[:, None] * tb.contract(("J",), p, g.X)
                    continue
                g.V = None
                terms = []
                fold = [j for j in range(self.np) if j not in g.live]
                self._bfwd(g.tree, g.X, mode, pp, terms,
                           (fold[0], g.X[fold[0]], -1) if self.np == 3 and len(fold) == 1 else None)
                self._bterms(terms, pp)
        for li in reversed(range(len(self.levels))):
            for g in self.levels[li]:
                if g.skip and li > 0:
                    continue
                V = g.V if g.V is not None else self._bwalk(g.tree, mode, p)
                for i, (e, j) in enumerate(g.blocks):
                    blk = V[..., i * C:(i + 1) * C, :]
                    if li == 0:
                        e.V = tb.from_buckets(blk, p) if one else tb.from_buckets(blk)
                    elif e.V is not None:
                        e.V[..., j * C:(j + 1) * C, :] = blk

    def _bwalk(self, node, mode, p):
        """_walk below the root street, by buckets, on the reaches and strategies _bfwd kept: (rows, K) values of
        player p (modes alt, br) or (np, rows, K)."""
        if isinstance(node, BTerminal):
            return node.V
        if isinstance(node, BExit):
            return node.V
        q, sig, X = node.player, node.sig, node.X
        vals = [node.leaf.V if a == node.fold and q == p and mode in ("alt", "br") else self._bwalk(ch, mode, p)
                for a, ch in enumerate(node.children)]
        if mode == "alt":
            V = np.stack(vals)                                               # (A, rows, K)
            if q != p:
                return V.sum(axis=0)
            vp = (V * np.moveaxis(sig, 1, 0)).sum(axis=0)
            node.regret += np.moveaxis(V - vp[None], 0, 1)
            node.ssum += X[p][:, None, :] * sig
            return vp
        if mode == "br":
            return np.max(vals, axis=0) if q == p else np.sum(vals, axis=0)
        V = np.stack(vals)                                                   # (A, np, rows, K)
        v = V.sum(axis=0)
        vq = (V[:, q] * np.moveaxis(sig, 1, 0)).sum(axis=0)
        v[q] = vq
        if mode == "cfr":
            node.regret += np.moveaxis(V[:, q] - vq[None], 0, 1)
            node.ssum += X[q][:, None, :] * sig
        return v

    def _bterms(self, terms, need=None):
        """The terminals' values (node.V) from their reaches (node.X), all of a subtree's at once: one tensor contraction
        per kind (the deals' weight, and each live set's shares) over their rows stacked. For one player p (`need`),
        terminals below another player f's fold (fctx: f, its reach there, a key) sum the tensors over f's buckets once
        per fold (reduce) and cost K x K a row instead of K^3 (apply)."""
        if not terms:
            return
        tb = self.tabs
        if need is not None:
            p = need
            byf: Dict[int, list] = {}
            for t in terms:
                if t.fctx is not None:
                    byf.setdefault(t.fctx[2], []).append(t)
            # (a fold with only a terminal or two below: their own contractions cost less than the two reductions)
            full = [t for t in terms if t.fctx is None]
            for k in list(byf):
                ts = byf[k]
                two = any(len(t.live) > 1 for t in ts)
                if sum(1 + (len(t.live) > 1) for t in ts) <= 1 + two:
                    full += byf.pop(k)
            if full:
                self._bterms_full(full, p)
            for ts in byf.values():
                f, Xf, _ = ts[0].fctx
                q = 3 - p - f
                MJ = tb.reduce(("J",), p, f, Xf)
                MS = tb.reduce(("S", tuple(sorted((p, q)))), p, f, Xf) if any(len(t.live) > 1 for t in ts) else None
                for t in ts:
                    m = tb.apply(MJ, t.X[q])
                    W = t.W
                    if W is not None:     # (a tournament's weights: the side pot, if any, p's alone or not p's)
                        if p in t.live:
                            win = tb.apply(MS, t.X[q]) if len(t.live) > 1 else m
                            t.V = W[p, 0][:, None] * m + W[p, 1][:, None] * win \
                                + (W[p, 2][:, None] * m if p in t.side else 0.0)
                        else:
                            t.V = W[p, 0][:, None] * m
                        continue
                    if p in t.live:
                        win = tb.apply(MS, t.X[q]) if len(t.live) > 1 else m
                        # (with a player all-in for less: the side pot is p's alone here or not p's, the third
                        # player having folded)
                        fm = t.fm
                        if fm is not None:
                            win = fm[:, None] * win + (1.0 - fm[:, None]) * (m if p in t.side else 0.0)
                        t.V = t.net[:, None] * win - t.c[p][:, None] * m
                    else:
                        t.V = -t.c[p][:, None] * m
            return
        out = [self._bterms_full(terms, p, keep=False) for p in range(self.np)]
        for i, t in enumerate(terms):
            t.V = np.stack([o[i] for o in out])

    def _bterms_full(self, terms, p, keep=True):
        """Player p's values at the terminals by full contractions, stacked (node.V, or returned when not keep)."""
        tb = self.tabs
        rows = [t.X.shape[1] for t in terms]
        cut = np.cumsum([0] + rows)
        Xall = np.concatenate([t.X for t in terms], axis=1)
        mJ = tb.contract(("J",), p, Xall)
        # each terminal's pots by who contests them: the whole pot by its live players, or with a player all-in for
        # less the main pot by everyone live and the side pot by the players with more in (_split)
        side = lambda t: t.side
        wins = {}
        for key, sel in (("all", lambda t: t.live), ("side", side)):
            if key == "side" and all(t.fm is None for t in terms):
                break
            for lv in {sel(t) for t in terms if len(sel(t)) > 1 and p in sel(t)}:
                ix = [i for i, t in enumerate(terms) if sel(t) == lv]
                sub = np.concatenate([Xall[:, cut[i]:cut[i + 1]] for i in ix], axis=1)
                w = tb.contract(("S", lv), p, sub)
                o = np.cumsum([0] + [rows[i] for i in ix])
                for k, i in enumerate(ix):
                    wins[(key, i)] = w[o[k]:o[k + 1]]
        res = []
        for i, t in enumerate(terms):
            m = mJ[cut[i]:cut[i + 1]]
            W = t.W
            if W is not None:     # (a tournament's weights)
                v = W[p, 0][:, None] * m
                if p in t.live:
                    v = v + W[p, 1][:, None] * (wins[("all", i)] if len(t.live) > 1 else m)
                    sv = side(t)
                    if t.fm is not None and p in sv:
                        v = v + W[p, 2][:, None] * (wins[("side", i)] if len(sv) > 1 else m)
                if keep:
                    t.V = v
                res.append(v)
                continue
            if p in t.live:
                win = wins[("all", i)] if len(t.live) > 1 else m
                fm = t.fm
                if fm is not None:
                    sv = side(t)
                    ws = (wins[("side", i)] if len(sv) > 1 else m) if p in sv else 0.0
                    win = fm[:, None] * win + (1.0 - fm[:, None]) * ws
                v = t.net[:, None] * win - t.c[p][:, None] * m
            else:
                v = -t.c[p][:, None] * m
            if keep:
                t.V = v
            res.append(v)
        return res

    def _split(self, t):
        """A bucketed terminal's pots (t.fm, t.side): with a player all-in for less on some row, the main pot's share
        of each row's pot (1 where nobody is) and the players who contest the side pot; else (None, its live)."""
        if len(t.live) < 2 or t.pot is None or not self.cfg.allin and len(set(self.stacks)) <= 1:
            return
        fm, side, memo = np.ones(len(t.pot)), None, {}
        for r in range(len(t.pot)):
            key = (tuple(t.c[:, r]), float(t.pot[r]))
            lay = memo.get(key)
            if lay is None:
                lay = memo[key] = self._layers(t.live, t.c[:, r], float(t.pot[r]))
            if len(lay) == 1:
                continue
            if len(lay) > 2 or (side is not None and lay[1][1] != side):
                raise ValueError(f"a bucketed terminal with three pots or two side pots: {lay}")
            side = lay[1][1]
            fm[r] = lay[0][0] / float(t.pot[r])
        if side is not None:
            t.fm, t.side = fm, side

    def _pass(self, R, mode, p=None, node=None):
        """A walk from `node` (the root), the later streets (if any) evaluated first."""
        node = self.root if node is None else node
        if self.levels:
            self._later(node, R, mode, p)
        return self._walk(node, R, mode, p)

    # -- terminal values
    def _term(self, node: Terminal, R, need=None, rho=None, memo=False):
        """(np, B, n) values: u_p = share x (pot - rake) - chips put in, summed over the others' compatible hands
        (only player `need`'s row when given). `memo`: an alternating pass's, whose tables are kept (see __init__)."""
        bd = node.bd
        npl = self.np
        if self._pw:
            return self._term_pw(node, R, need)
        exact = self.cfg.exact and bd.B == 1
        out = np.zeros((npl, bd.B, self.hands.n))
        net = node.pot - node.rake
        players = range(npl) if need is None else [need]
        cache = {}

        def get(key, fn):
            if key not in cache:
                if memo:
                    who = key[1:]
                    stamp = tuple(self._ver[j] for j in who)
                    mk = (id(node),) + key
                    hit = self._tmemo.get(mk)
                    if hit is not None and hit[1] == stamp:
                        cache[key] = hit[2]
                    else:
                        cache[key] = fn()
                        self._tmemo[mk] = (who, stamp, cache[key])
                else:
                    cache[key] = fn()
            return cache[key]

        KU = lambda j: get(("U", j), lambda: bd.k_u(R[j]))
        KLE = lambda j: get(("LE", j), lambda: bd.k_le(R[j]))
        KW = lambda j: get(("W", j), lambda: KLE(j)[0] + 0.5 * KLE(j)[1])

        def P(j, kind, live):
            # the card tables for live set `live` (the terminal's own, or a side pot's: a key of its own)
            kinds = ("W", "E") if len(live) == 3 else ("W",) if j in live else ("U",)
            return get(("P" if live == node.live else "PS", j), lambda: bd.p_tables(R[j], kinds))[kind]
        y = lambda q, r: get(("y", min(q, r), max(q, r)), lambda: R[q] * R[r])
        ykle = lambda q, r: get(("yLE", min(q, r), max(q, r)), lambda: bd.k_le(y(q, r)))

        def mass(q, r):
            if rho is not None:
                return get(("m", min(q, r), max(q, r)), lambda: rho[(min(q, r), max(q, r))] * KU(q)[0] * KU(r)[0])
            return get(("m", min(q, r), max(q, r)), lambda: self._mass(bd, R, q, r, KU(q), KU(r)))

        dot = lambda a, b: np.einsum("bnc,bnc->bn", a, b)

        def win_of(p, live):
            """p's share of a pot the players `live` contest, summed over the others' compatible hands, and the joint
            mass."""
            others = [j for j in range(npl) if j != p]
            if npl == 2:
                q = others[0]
                m = KU(q)[0]
                if p not in live:
                    return 0.0, m
                return (m if len(live) == 1 else KW(q)), m
            q, r = others
            m = mass(q, r)
            if p not in live:
                return 0.0, m
            if len(live) == 1:
                return m, m
            if len(live) == 2:
                o = q if q in live else r
                f = r if o == q else q
                if exact:
                    kl, ke = ykle(o, f)
                    return KW(o) * KU(f)[0] - dot(P(o, "W", live), P(f, "U", live)) + kl + 0.5 * ke, m
                return KW(o) * KU(f)[0] * self._ratio(m, KU(o)[0], KU(f)[0]), m
            KEq, KEr = KLE(q)[1], KLE(r)[1]
            win = KW(q) * KW(r) + KEq * KEr / 12.0
            if exact:
                kl, ke = ykle(q, r)
                return win - dot(P(q, "W", live), P(r, "W", live)) - dot(P(q, "E", live), P(r, "E", live)) / 12.0 \
                    + kl + ke / 3.0, m
            return win * self._ratio(m, KU(q)[0], KU(r)[0]), m

        # the pots and who contests each: the whole pot by the live players, or with a player all-in for less the
        # main pot by everyone live and the side pot by the players with more in (_layers)
        if node.lay is None:
            node.lay = self._layers(node.live, node.c, node.pot) if len(node.live) > 1 and node.pot > 1e-12 \
                else [(node.pot, node.live)]
        pots = [(net, node.live)] if len(node.lay) == 1 else [(net * amt / node.pot, lv) for amt, lv in node.lay]
        wins = [{} for _ in pots]
        for k, (_, live) in enumerate(pots):
            if not exact and npl == 3 and len(live) >= 2:
                # pairwise: q and r independent given h (the joint mass kept exact). The live players' shares then
                # needn't add up to every deal's pot; scaled per board so they do, the game stays constant-sum.
                wm = {p: win_of(p, live) for p in live}
                tot = sum((R[p] * wm[p][0]).sum(axis=1, keepdims=True) for p in live)
                p0 = live[0]
                M = (R[p0] * wm[p0][1]).sum(axis=1, keepdims=True)
                sc = np.where(tot > 1e-300, M / np.where(tot > 1e-300, tot, 1.0), 1.0)
                wins[k] = {p: (w * sc, m) for p, (w, m) in wm.items()}
        W = node.W          # (a tournament's weights: the deals' weight's, then each pot's)
        for p in players:
            m = None
            v = 0.0
            for k, (net_k, live) in enumerate(pots):
                win, m = wins[k][p] if p in wins[k] else win_of(p, live)
                v = v + (W[p, 1 + k] if W is not None else net_k) * win
            out[p] = v + (W[p, 0] if W is not None else -node.c[p]) * m
        return out

    # -- four or more players: pairwise terminals and layered pots
    def _layers(self, live, c, pot):
        """[(chips, players who contest them)]: the pot layered by the live players' stakes, each seat's chips in
        (MWConfig.inv0, each player's before the root street (none given: the same for everyone), plus `c`, its chips
        from the root street's start, and the dead seats' MWConfig.dead): each layer goes to the best hand among the
        live players whose chips reach it. One layer, the whole pot, when every live player has the same in (or nobody
        can be all-in for less: nobody all-in at the root and the stacks the same)."""
        cfg, npl = self.cfg, self.np
        if not cfg.allin and len(set(getattr(self, "stacks", ()))) <= 1:
            return [(pot, tuple(live))]
        inv0 = cfg.inv0 or (0.0,) * npl
        tot = [float(inv0[i]) + float(c[i]) for i in range(npl)]
        levels = []
        for x in sorted(tot[i] for i in live):
            if not levels or x > levels[-1] + 1e-9:
                levels.append(x)
        if len(levels) == 1:
            return [(pot, tuple(live))]
        chips = tot + [float(x) for x in cfg.dead]
        out, prev = [], 0.0
        for lev in levels:
            out.append([sum(min(t, lev) - min(t, prev) for t in chips), tuple(i for i in live if tot[i] >= lev - 1e-9)])
            prev = lev
        out[0][0] += pot - sum(a for a, _ in out)    # (the chips not counted per seat: before the root street, antes)
        return [tuple(x) for x in out if x[0] > 1e-12] or [(pot, tuple(live))]

    def _pw_rho(self):
        """Per pair of players (j, k), per combo h of any other: the share of the j x k hand pairs compatible with h that
        are also compatible with each other (exact mass / product of the one-sided ones), on the root's board from the
        root's ranges (as _leaf_rho: the strategies and the cards below move it very little). (1, n)."""
        if getattr(self, "_rho_pw", None) is None:
            bd, R = self.root_bd, self.reach0
            out = {}
            for j in range(self.np):
                for k in range(j + 1, self.np):
                    kj, kk = bd.k_u(R[j]), bd.k_u(R[k])
                    out[(j, k)] = self._ratio(self._mass(bd, R, j, k, kj, kk), kj[0], kk[0])
            self._rho_pw = out
        return self._rho_pw

    def _pw_corr(self, p):
        """Player p's pairwise overlap correction: the product of _pw_rho over the pairs of the others."""
        cache = self.__dict__.setdefault("_corr_pw", {})
        if p not in cache:
            rho = self._pw_rho()
            c = np.ones((1, self.hands.n))
            for (j, k), v in rho.items():
                if p not in (j, k):
                    c = c * v
            cache[p] = c
        return cache[p]

    def _pw_masses(self, bd, R, ku=None):
        """Four or more players: per player p, the deals' weight per combo h of p (the others' hands compatible with
        h) as the product of the one-sided masses (pairwise: the others' hands independent given h) times the pairs'
        overlap correction (_pw_corr), scaled per board so every player's total is the first's (the deals counted the
        same from every seat); and that total."""
        ku = ku or [bd.k_u(R[j])[0] for j in range(self.np)]
        ms = []
        for p in range(self.np):
            m = self._pw_corr(p) * np.ones_like(ku[0])
            for j in range(self.np):
                if j != p:
                    m = m * ku[j]
            ms.append(m)
        tot = [(R[p] * ms[p]).sum(axis=1, keepdims=True) for p in range(self.np)]
        return [ms[p] * _pw_scale(tot[0], tot[p]) for p in range(self.np)], tot[0]

    def _term_pw(self, node: Terminal, R, need=None):
        """_term for four or more players, pairwise: p's share of a pot against the players contesting it is the
        product of its share against each (weaker + tied / 2, as heads-up) times the others' masses, scaled per board
        so the contesting players' shares add up to every deal's pot (constant-sum, as the 3-way pairwise terminal).
        Pots are layered by the all-in players' stakes (_layers)."""
        bd, npl = node.bd, self.np
        ku = [bd.k_u(R[j])[0] for j in range(npl)]
        ms, M = self._pw_masses(bd, R, ku)
        out = np.zeros((npl, bd.B, self.hands.n))
        live = node.live
        kw = {}
        share = (node.pot - node.rake) / node.pot if node.pot > 1e-12 else 0.0
        val = {p: 0.0 for p in range(npl)}
        W = node.W          # (a tournament's weights: the deals' weight's, then each layer's)
        lay = node.lay if node.lay is not None else self._layers(live, node.c, node.pot)
        for li, (amt, elig) in enumerate(lay):
            net = amt * share
            coef = (lambda p: W[p, 1 + li]) if W is not None else (lambda p: net)
            if len(elig) == 1:
                val[elig[0]] = val[elig[0]] + coef(elig[0]) * ms[elig[0]]
                continue
            for j in elig:
                if j not in kw:
                    kl, ke = bd.k_le(R[j])
                    kw[j] = kl + 0.5 * ke
            w = {}
            for p in elig:
                x = self._pw_corr(p) * np.ones_like(ku[0])
                for j in range(npl):
                    if j != p:
                        x = x * (kw[j] if j in elig else ku[j])
                w[p] = x
            tot = sum((R[p] * w[p]).sum(axis=1, keepdims=True) for p in elig)
            k = _pw_scale(M, tot)
            for p in elig:
                val[p] = val[p] + coef(p) * w[p] * k
        for p in (range(npl) if need is None else [need]):
            out[p] = val[p] + (W[p, 0] if W is not None else -node.c[p]) * ms[p]
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
            v = self._term(node, R, None if mode not in ("br", "alt") else p, rho, memo=mode == "alt")
            return v[p] if mode in ("br", "alt") else v
        if isinstance(node, Exit):
            return node.V
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
            # (q's strategy is the same until q's own pass updates its regrets)
            hit = self._smemo.get(id(node))
            if hit is not None and hit[0] == self._ver[q]:
                sig = hit[1]
            else:
                sig = node.strategy()
                self._smemo[id(node)] = (self._ver[q], sig)
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
        self._ver = [v + 1 for v in self._ver]      # (every player's regrets changed)

    def _core(self):
        """The compiled core's program for this solve (gto/mwcore.py: a bucketed solve, or a 3-way river), or None."""
        if not self.levels and len(self.board) != 5:
            return None
        from . import mwcore
        return mwcore.program(self)

    def _core_eval(self, node, mode, p, R, kids=False):
        """A pass at a root-street node in the compiled core (mwcore.Program.eval), or None without the core. When the
        core fails (mwcore.Failed) it's started again (mwcore.restart: the program laid out again from the solver's
        arrays, which the failed pass didn't touch) and the pass run again in it once; else None: numpy's pass."""
        from . import mwcore
        for again in (False, True):
            prog = self._core()
            if prog is None:
                return None
            try:
                return prog.eval(node, mode, p, R, kids=kids)
            except mwcore.Failed:
                if again:
                    mwcore.off()
                if again or not mwcore.restart():
                    return None

    def run(self, iters=1000, time_budget=None, min_iters=5, should_stop=None, reserve=0.0):
        """`time_budget` (seconds): no iteration starts that would end past it (by the mean iteration so far), with
        `reserve` more iterations' time kept back for what follows (br_gains and action_evs cost about two)."""
        prog = self._core()
        if prog is not None:
            from . import mwcore
            for again in (False, True):
                t0 = time.time()
                try:
                    return self._run_core(prog, iters, time_budget, min_iters, should_stop, reserve)
                except mwcore.Failed:
                    # (the core failed: the run's iterations are dropped, the solver's arrays as they were before it.
                    # Started again (mwcore.restart) the run starts over in it from them, once; else numpy runs the
                    # solve from there. Either in what's left of the budget)
                    if time_budget is not None:
                        time_budget = max(time_budget - (time.time() - t0), 0.0)
                    if again:
                        mwcore.off()
                    prog = self._core() if not again and mwcore.restart() else None
                    if prog is None:
                        break
        t0 = time.time()
        start = self.iters
        self._tmemo.clear(); self._smemo.clear()     # (kept within a run: its reaches are this run's root reaches')
        while self.iters - start < iters:
            if should_stop is not None and should_stop():
                break
            done = self.iters - start
            if time_budget is not None and done >= min_iters and \
                    time.time() - t0 + (time.time() - t0) / max(done, 1) * (1.0 + reserve) > time_budget:
                self.out_of_time = True
                break
            if self.cfg.alternate:
                for p in range(self.np):
                    if p not in self.cfg.allin:   # (no decisions to learn)
                        self._pass(self.reach0, "alt", p)
                    self._ver[p] += 1
                    # (memory) what no later pass can reuse: the pair terms, and the next traverser's tables (stale
                    # by its next opponent pass)
                    nxt = (p + 1) % self.np
                    for k in [k for k, (who, _, _) in self._tmemo.items() if len(who) == 2 or who[0] == nxt]:
                        del self._tmemo[k]
            else:
                self._pass(self.reach0, "cfr")
            self.iters += 1
            self._discount()
            if time_budget is not None and self.iters - start >= min_iters and time.time() - t0 > time_budget:
                self.out_of_time = True
                break
        self.seconds += time.time() - t0
        return self

    def _run_core(self, prog, iters, time_budget, min_iters, should_stop, reserve):
        """run() in the compiled core: iterations in chunks of about mwcore.CHUNK_S seconds, the budget and should_stop
        checked between them (the same rule as run's: no iteration starts that would end past the budget), the regrets
        and sums back in the solver's arrays after."""
        from . import mwcore
        t0 = time.time()
        start = self.iters
        per = None
        ok = False
        try:
            while self.iters - start < iters:
                if should_stop is not None and should_stop():
                    break
                done = self.iters - start
                el = time.time() - t0
                if time_budget is not None and done >= min_iters and \
                        el + (el / max(done, 1)) * (1.0 + reserve) > time_budget:
                    self.out_of_time = True
                    break
                k = 1 if per is None else max(1, min(mwcore.MAX_CHUNK, int(mwcore.CHUNK_S / max(per, 1e-6))))
                k = min(k, iters - done)
                if time_budget is not None and per is not None and done >= min_iters:
                    k = max(1, min(k, int((time_budget - el) / per - 1.0 - reserve)))
                c0 = time.time()
                prog.run(k, self.iters)
                self.iters += k
                dt = max((time.time() - c0) / k, 1e-7)   # (the browser's clock is whole milliseconds: core.run's note)
                per = dt if per is None else 0.5 * per + 0.5 * dt
                if time_budget is not None and self.iters - start >= min_iters and time.time() - t0 > time_budget:
                    self.out_of_time = True
                    break
            ok = True
        finally:
            if ok or not isinstance(sys.exc_info()[1], mwcore.Failed):
                prog.pull()
            else:   # (a failed run's arena isn't read: its iterations are dropped, the solver's arrays as they were)
                self.iters = start
                self.seconds += time.time() - t0
        self._ver = [v + self.iters - start for v in self._ver]
        self._gains_memo = None
        self.seconds += time.time() - t0
        return self

    # -- diagnostics and results
    def total_mass(self):
        """Total weight of the deals (sum over compatible hand tuples of the reach products)."""
        R = self.reach0
        bd = self.root.bd
        if self.np == 2:
            return float((R[0] * bd.k_u(R[1])[0]).sum())
        if self._pw:
            return float(self._pw_masses(bd, R)[1].sum())
        return float((R[0] * self._mass_exact(bd, R, 1, 2)).sum())

    def values(self):
        """Each player's expected value (bb, relative to the root street's start) under the average strategies."""
        m = self.total_mass()
        if self._core() is not None:
            from .mwcore import EV
            R = self.reach0[:, 0, :]
            out = []
            for p in range(self.np):
                v = self._core_eval(self.root, EV, p, R)
                if v is None:   # (the core failed: numpy's pass)
                    break
                out.append((v * R[p]).sum())
            else:
                return np.array(out) / m
        v = self._pass(self.reach0, "avg")
        return np.array([(v[p][0] * self.reach0[p][0]).sum() for p in range(self.np)]) / m

    def br_gains(self):
        """What each player gains by best-responding to the others' average strategies (bb per deal). Kept until
        the next iteration (the advisor reads it at checkpoints and again for the answer)."""
        memo = getattr(self, "_gains_memo", None)
        if memo is not None and memo[0] == self.iters:
            return memo[1].copy()
        g = self._br_gains()
        self._gains_memo = (self.iters, g)
        return g.copy()

    def _br_gains(self):
        v = self.values()
        m = self.total_mass()
        core = self._core() is not None
        out = []
        for p in range(self.np):
            if p in self.cfg.allin:   # (no decisions: nothing to gain)
                out.append(0.0)
                continue
            br = None
            if core:
                from .mwcore import BR
                br = self._core_eval(self.root, BR, p, self.reach0[:, 0, :])
            if br is None:   # (no core, or it failed: numpy's pass)
                br = self._pass(self.reach0, "br", p)
            else:
                br = br[None, :]
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
        elif self._pw:
            mass = self._pw_masses(bd, R)[0][p]
        else:
            mass = self._mass_exact(bd, R, *others)
        evs = []
        if self._core() is not None and node is not None and R.shape[1] == 1:
            from .mwcore import EV, Unsupported
            try:
                got = self._core_eval(node, EV, p, R[:, 0, :], kids=True)
            except Unsupported:   # (a node the program doesn't hold: numpy's pass)
                got = None
            if got is not None:   # (else the core failed: numpy's pass)
                kids = got[1]
                return np.where(mass > 1e-12, kids / np.where(mass > 1e-12, mass, 1.0), 0.0) + self._fold_base(node)
        if self.levels:
            self._later(node, R, "avg", None)
        for ch in node.children:
            v = self._walk(ch, R, "avg")[p]
            evs.append(np.where(mass > 1e-12, v / np.where(mass > 1e-12, mass, 1.0), 0.0) + self._fold_base(node))
        return np.stack(evs)[:, 0, :]

    def _fold_base(self, node: Decision) -> float:
        """What the node's player gives up by folding now: its chips in, or in a tournament minus the value of
        folding (Decision.fold_v)."""
        if self.cfg.icm is not None and node.fold_v is not None:
            return -node.fold_v
        return float(node.c[node.player])

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
        s = self.showdown_shares()[ci]
        return 0.0 if np.isnan(s) else float(s)

    def showdown_shares(self, p: Optional[int] = None) -> np.ndarray:
        """Each of player p's combos' share of the pot against the others' ranges at showdown on the root's board
        (river; p the root's player by default); nan where the others' ranges leave a combo no deal."""
        bd = self.root.bd
        t = Terminal(bd, tuple(range(self.np)), 1.0, np.zeros(self.np), 0.0, "show")
        p = self.root.player if p is None else p
        R = self.reach0
        win = self._term(t, R, p)[p][0]
        others = [j for j in range(self.np) if j != p]
        m = bd.k_u(R[others[0]])[0][0] if self.np == 2 else \
            self._pw_masses(bd, R)[0][p][0] if self._pw else self._mass(bd, R, *others)[0]
        return np.where(m > 1e-12, win / np.where(m > 1e-12, m, 1.0), np.nan)

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


# showdown_equities: boards a batch (the exact count's (boards, combos, 52) card tables: ~3 MB each at 16 boards x 1000
# combos), and the exact count up to this many runouts (a turn's 46; a flop's 1176 cost ~1.6 s natively that way,
# the pairwise count over all of them 0.2 s)
EQ_BATCH = {True: 16, False: 64}
EQ_EXACT_RUNOUTS = 64


def showdown_equities(board, ranges: Sequence[np.ndarray], exact: Optional[bool] = None) -> List[np.ndarray]:
    """Each player's equity per combo (1326 arrays; NaN outside its range or where the others' ranges leave it no
    deal): its share of the pot at showdown against the other ranges (the hands as dealt, nobody folding), ties split,
    averaged over every runout of the rest of the board with each runout and deal by its weight (the others' hands that
    share no card with it or with each other, by their reaches), as range_equities heads-up (two players: the same
    numbers). Three players on the turn and river (EQ_EXACT_RUNOUTS) count the card removal between the three hands
    exactly (the solver's kernels: the same numbers as a triple loop over the hands); on the flop the two others' hands
    are taken as independent given the combo (pairwise, their joint deal weight exact) and each runout's shares scaled
    so they add up to its pot: over all 1176 runouts of Kd7c2h and Jh8h4c (3-way charts' ranges) 0.2-0.3 points of
    equity from the exact count on average, 1.5 at most, where an even sample of 168 runouts counted exactly is 1.6-1.9
    points off on average (7-9 at most). Two or three players only (NaN for more: four players counted pairwise were
    2.5-4 points low on their strongest hands against a Monte Carlo count)."""
    board = [int(c) for c in board]
    N = len(ranges)
    R = [np.asarray(r, dtype=np.float64).reshape(-1) for r in ranges]
    out = [np.full(N_FULL, np.nan) for _ in range(N)]
    active = np.flatnonzero(np.any([r > 0 for r in R], axis=0))
    if not 2 <= N <= 3 or not len(active):
        return out
    rest = [c for c in range(52) if c not in set(board)]
    runs = list(itertools.combinations(rest, 5 - len(board)))
    exact = N == 3 and (len(runs) <= EQ_EXACT_RUNOUTS if exact is None else exact)
    h = Hands(active)
    x = [r[active] for r in R]
    num, den = np.zeros((N, h.n)), np.zeros((N, h.n))
    dot = lambda a, b: np.einsum("bnc,bnc->bn", a, b)
    step = EQ_BATCH[exact]
    for b0 in range(0, len(runs), step):
        bd = Boards(h, [tuple(board) + r for r in runs[b0:b0 + step]])
        ok = bd.s >= 0                                   # (a combo holding a runout card: no deal there)
        X = [np.where(ok, xi[None, :], 0.0) for xi in x]
        KU = [bd.k_u(Xi) for Xi in X]
        KLE = [bd.k_le(Xi) for Xi in X]
        KW = [kl + 0.5 * ke for kl, ke in KLE]
        PT = [bd.p_tables(Xi, ("W", "E")) for Xi in X] if exact else None
        wins, mass = [], []
        for p in range(N):
            o = [j for j in range(N) if j != p]
            if N == 2:
                m, w = KU[o[0]][0], KW[o[0]]
            else:
                q, r = o
                m = MultiwaySolver._mass(bd, X, q, r, KU[q], KU[r])
                w = KW[q] * KW[r] + KLE[q][1] * KLE[r][1] / 12.0       # (beat both 1, tie one 1/2, tie both 1/3)
                if exact:
                    kl, ke = bd.k_le(X[q] * X[r])
                    w = w - dot(PT[q]["W"], PT[r]["W"]) - dot(PT[q]["E"], PT[r]["E"]) / 12.0 + kl + ke / 3.0
                else:
                    w = w * MultiwaySolver._ratio(m, KU[q][0], KU[r][0])
            wins.append(np.where(ok, w, 0.0))
            mass.append(np.where(ok, m, 0.0))
        sc = 1.0
        if N == 3 and not exact:
            # (each runout's shares scaled so the players' add up to its deals' weight: constant-sum, as _term's)
            tot = sum((X[p] * wins[p]).sum(axis=1, keepdims=True) for p in range(N))
            M = (X[0] * mass[0]).sum(axis=1, keepdims=True)
            sc = np.where(tot > 1e-300, M / np.where(tot > 1e-300, tot, 1.0), 1.0)
        for p in range(N):
            num[p] += (wins[p] * sc).sum(axis=0)
            den[p] += mass[p].sum(axis=0)
    for p in range(N):
        good = (den[p] > 1e-12) & (x[p] > 0)
        out[p][active[good]] = np.clip(num[p][good] / den[p][good], 0.0, 1.0)
    return out


def solve(board, ranges, pot, stack, bets=None, to_act=0, n_raises=0, pending=None, cfg=None,
          iters=1000, time_budget=None, should_stop=None, dead=()) -> MultiwaySolver:
    s = MultiwaySolver(board, ranges, pot, stack, bets, to_act, n_raises, pending, cfg, dead)
    return s.run(iters=iters, time_budget=time_budget, should_stop=should_stop)
