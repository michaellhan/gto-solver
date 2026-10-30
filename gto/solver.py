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
  showdown matrices over the remaining runouts; from the flop, the river beyond
  `depth` is still bet on (river_play), and the turns are one per rank class
  (turn_classes)
"""
from __future__ import annotations
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .cards import ALL_COMBOS, COMBO_INDEX
from .evaluator import eval_board, eval_boards
from . import core
np.seterr(divide="ignore", over="ignore", invalid="ignore")
# The browser's Pyodide numpy has no BLAS and its matmul is a plain loop, so a browser flop solve is
# ~95% equity-leaf matvecs (arithmetic, not call overhead); einsum's dot-product loops run them and the
# compat sums ~3x faster there. Natively BLAS matmul is far faster than einsum and is kept.
NO_BLAS = sys.platform == "emscripten"


def _matvec(M, x):
    """M: (B, R, C) matrices, x: (..., B, C) vectors -> (..., B, R), each vector times its row's matrix."""
    if NO_BLAS:
        return np.einsum("bij,...bj->...bi", M, x)
    return np.matmul(M, x[..., None])[..., 0]


TINY = np.float32(2.0 ** -126)   # float32's smallest normal: regret matching takes a sum under it for none (its
                                 # reciprocal can overflow, and a zero regret times inf is NaN), as the core does
N_FULL = 1326
FLOP_LEAF_RIVERS = 4  # a flop all-in's equity: at least this many rivers under every turn card (Ctx._flop_leaf)
MEMBER_FULL = np.zeros((52, N_FULL))
for _i, (_a, _b) in enumerate(ALL_COMBOS):
    MEMBER_FULL[_a, _i] = 1.0
    MEMBER_FULL[_b, _i] = 1.0
CA_FULL = np.array([a for a, b in ALL_COMBOS])
COMBO_TABLE = np.full((52, 52), -1, dtype=np.intp)     # full combo index of a card pair, either order
for _i, (_a, _b) in enumerate(ALL_COMBOS):
    COMBO_TABLE[_a, _b] = COMBO_TABLE[_b, _a] = _i
CB_FULL = np.array([b for a, b in ALL_COMBOS])


def eq_key(eq: np.ndarray) -> np.ndarray:
    """Equities as hand buckets sort them (Ctx.bucket_table, river_table): to 2^-20, so that ties (every hand that loses
    to the whole range is at -1) go in combo order however the sums were rounded (numpy's, or the compiled core's:
    core/gtocore.c t_bucket). A binary grid: a tie at a fraction of a few hundred combos (-123/128) is on it, where a
    decimal one put it halfway between two steps for the rounding to pick."""
    return np.rint(eq * 1048576.0)


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
    n_rivers: Optional[int] = None    # rivers in the tree for full-decision river subtrees from the turn (None = all)
    river_sample: Optional[int] = None  # rivers traversed per iteration from the turn, redrawn every iteration
                                      # (public chance sampling; None = traverse all of them)
    n_rivers_from_flop: int = 8       # rivers per turn for full-decision river subtrees when the root is the flop
    leaf_turns: int = 6               # sampled turns for equity leaves reached from the flop
    leaf_rivers: Optional[int] = None # rivers for equity leaves when solving from the turn (None = all)
    leaf_rivers_flop: int = 16        # rivers per turn card for equity leaves when solving from the flop
    # From the flop (see README, Accuracy vs speed):
    # river_play: the river beyond `depth` keeps its betting, played without seeing the river card (its showdowns
    # are the turn's equity leaves), instead of being checked down. A checked-down river gave position and
    # betting no value there: BTN c-bet a third pot on QQ6 under 60% where every turn with river betting has 88%.
    river_play: bool = True
    # turn_classes: one turn card per rank (a flush suit's cards apart), weighted by the turns it stands for,
    # instead of n_turns random turns. A few fixed turns let the flop's ranges fit the cards that come (BB's calls
    # the hands that pair them): with river betting 8 random turns had K72's c-bet at 68-94% by seed, every turn 97%.
    turn_classes: bool = True
    # later_sizes: bet and raise sizes for the turn and river inside a flop solve ({"bet": {...}, "raise": {...}};
    # None = bet_sizes and raise_sizes). One size there moved the flop's c-bet by about a point on K72 and 765.
    later_sizes: Optional[dict] = field(default_factory=lambda: {
        "bet": {"turn": [0.75], "river": [0.75]}, "raise": {"turn": [0.75], "river": [0.75]}})
    # Hand buckets inside a flop solve (README, Accuracy vs speed; Ctx.bucket_table, BucketEntry). river_buckets: with
    # river_play, the river beyond `depth` is played by this many hand buckets per player and turn card instead of per
    # combo: its strategies, reaches and payoffs are K wide and its showdowns a K x K bucket-pair matrix instead of the
    # N x N turn leaf (None = per combo). turn_buckets: the turn too (its all-ins included), from the chance node
    # down; the flop stays per combo. bucket_river_step: the bucket tables average every step-th river card, not all.
    river_buckets: Optional[int] = None
    turn_buckets: bool = False
    bucket_river_step: int = 1
    # river_strength: with river_play, the river beyond `depth` is played by this many buckets of river strength
    # (RiverEntry, Ctx.river_table): on each river card each player's range splits into equal shares by equity there
    # against the other's root range, and a hand coming from the turn (a combo, or a turn bucket) enters the river
    # in each bucket with the share of the river cards that put it there. So a draw is a strong hand on the rivers it
    # hits and air on the others, which the turn's equity leaf (a draw as a middling hand) can't play. The players
    # don't see the river card itself, only where it puts them (bucket-pair showdowns averaged over the rivers).
    # bucket_river_step: every step-th river card in the tables. None = the river on the turn's boards as above.
    river_strength: Optional[int] = None
    # ... on flops with at least this stack-to-pot ratio (the root's stack over its pot). Below it (3bet pots and up)
    # the river stays on the turn's boards: against the river with its card in 3bet pots (3 fixed turns, every river,
    # K72, A83, 765) river strength had the 3-bettor bet three quarters pot with 15-31% of its range where the
    # reference bets it 0-9% (the turn's boards 0%), its c-bet as far off on average (README, Accuracy vs speed).
    # postflop-solver's full game (every turn and river) bets it 40-55% on AJ3 and Q55, as river strength does, but
    # there river strength's flop gives up as much as the turn's boards' (0.26 / 0.46% of the pot OOP / IP against
    # 0.40 / 0.29; scripts/bench_flop_abstraction.py): the library's 3bet pots deal the river (river_dealt).
    river_strength_spr: float = 8.0
    # river_dealt: with river_play, the river beyond `depth` inside a flop solve is dealt, every card under each turn,
    # and played on each card by this many buckets of strength there (BucketEntry under the river's chance node,
    # Ctx.river_card_table: each player's root range split into equal shares by equity on that card against the
    # other's), instead of on the turn's boards: a draw knows when it got there, and its opponent that the card came.
    # The library's entries (flop_library.RIVER_DEALT): 64 buckets in a 3bet pot are 7-17x the blind river's work, the
    # river per combo (depth 3, every river) 50-90x for 0.01-0.02% of the pot less given up; in a single-raised pot
    # 6-7x, for 0.1-0.25% of the pot less (README, Precomputed flop library).
    river_dealt: Optional[int] = None
    # river_classes: from a turn root, the river as one card per class (Ctx.turn_classes on the turn's board: a rank's
    # cards, a suit with two or more board cards apart), each class's subtree standing for its every card through the
    # card swap that maps one onto the other (RiverClasses). With no flush possible in the swapped suits the swap keeps
    # every hand's strength, so the class is exact when the ranges are symmetric in those suits: 13 rivers on a rainbow
    # turn instead of 48, 24 on a two-tone one. The game stays zero-sum.
    river_classes: bool = False
    seed: int = 7
    # cfr: the regret minimizer, a key of CFR_SCHEMES (docs/cfr_research.md; README, Accuracy vs speed)
    cfr: str = "dcfr-ramp"
    # rake: the winner of a postflop pot (showdown or fold) pays rake_rate of the final pot (without an
    # uncalled bet, which goes back), at most
    # rake_cap bb (None = no cap). 0 = no rake. Preflop pots aren't raked ("no flop, no drop").
    rake_rate: float = 0.0
    rake_cap: Optional[float] = None
    # root_allin: every decision on the root street offers all-in, however deep (not just at allin_spr pots or less),
    # so a jam there is priced and can be graded (a 15x-pot flop jam had no action in the tree to grade against)
    root_allin: bool = False
    # keep_sizes: the root street's sizes the hand's actions took ({"bet": [...], "raise": [...], "aggr": raises made}),
    # kept when a tree over its cap is cut (Advisor._trim_cfg); it doesn't change the tree itself
    keep_sizes: Optional[dict] = None
    # icm: a tournament's payoffs (gto/icm.py HUPayoffs): every terminal pays prize equity (in chip equivalents) instead
    # of chips. The game is then not zero-sum (the players outside the pot gain when either busts). None = chips.
    icm: Optional[object] = None

    def rake(self, pot: float) -> float:
        """Rake in bb taken from a final pot of `pot` bb."""
        if self.rake_rate <= 0:
            return 0.0
        r = self.rake_rate * pot
        return r if self.rake_cap is None else min(r, self.rake_cap)

    def copy(self) -> "Config":
        return Config(bet_sizes={k: list(v) for k, v in self.bet_sizes.items()},
                      raise_sizes={k: list(v) for k, v in self.raise_sizes.items()},
                      max_raises=self.max_raises, allin_spr=self.allin_spr, depth=self.depth,
                      n_turns=self.n_turns, n_rivers=self.n_rivers, river_sample=self.river_sample,
                      n_rivers_from_flop=self.n_rivers_from_flop,
                      leaf_turns=self.leaf_turns, leaf_rivers=self.leaf_rivers, leaf_rivers_flop=self.leaf_rivers_flop,
                      river_play=self.river_play, turn_classes=self.turn_classes, river_buckets=self.river_buckets,
                      turn_buckets=self.turn_buckets, bucket_river_step=self.bucket_river_step,
                      river_strength=self.river_strength, river_strength_spr=self.river_strength_spr,
                      river_dealt=self.river_dealt, river_classes=self.river_classes,
                      later_sizes=None if self.later_sizes is None else
                      {k: {st: list(v) for st, v in d.items()} for k, d in self.later_sizes.items()},
                      seed=self.seed, cfr=self.cfr, rake_rate=self.rake_rate, rake_cap=self.rake_cap,
                      root_allin=self.root_allin, icm=self.icm,
                      keep_sizes=None if self.keep_sizes is None else {k: (list(v) if isinstance(v, list) else v)
                                                                       for k, v in self.keep_sizes.items()})

    def sizes_key(self) -> tuple:
        return (tuple((k, tuple(v)) for k, v in sorted(self.bet_sizes.items())),
                tuple((k, tuple(v)) for k, v in sorted(self.raise_sizes.items())), self.depth, self.max_raises,
                None if self.later_sizes is None else
                tuple((k, tuple((st, tuple(v)) for st, v in sorted(d.items()))) for k, d in sorted(self.later_sizes.items()))) + \
            ((("rake", self.rake_rate, self.rake_cap),) if self.rake_rate > 0 else ()) + \
            (("root_allin",) if self.root_allin else ()) + ((("icm", self.icm.key),) if self.icm is not None else ())


def street_of(board_len: int) -> str:
    return {3: "flop", 4: "turn", 5: "river"}[board_len]


class Ctx:
    """Per-solve compressed combo space and cached showdown matrices."""

    def __init__(self, active: np.ndarray, cfg: Config):
        self.active = active
        self.n = len(active)
        self.member = MEMBER_FULL[:, active]
        self.member32 = self.member.astype(np.float32)
        self.ca = CA_FULL[active]
        self.cb = CB_FULL[active]
        self.cab = np.concatenate([self.ca, self.cb])     # both cards' columns, gathered in one call
        # the (card, combo) pairs grouped by card, for per-card sums: combos in pair order, where each held
        # card's group starts, and each combo's two cards as indices into the held cards
        held, counts = np.unique(self.cab, return_counts=True)
        self.card_perm = np.concatenate([np.arange(self.n), np.arange(self.n)])[np.argsort(self.cab, kind="stable")]
        self.card_starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.intp)
        self.cab_held = np.searchsorted(held, self.cab).astype(np.intp)
        self._compat = None
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._M: Dict[tuple, np.ndarray] = {}
        self._leaf: Dict[tuple, np.ndarray] = {}
        self._stacks: Dict[tuple, np.ndarray] = {}
        self._turn_picks: Dict[tuple, tuple] = {}   # flop -> (picks in draw order, order random)
        self.active_list = [int(i) for i in active]
        self.root_len = 5
        self.root_reach = None     # (r0, r1) at the root, for river buckets (set by solve)
        self.tables = None         # the compiled core's table builds while it lays a program out (gto/core.py Tables)

    @property
    def compat(self) -> np.ndarray:
        """(n, n): 1 where two combos share no card. Made on first use: an n x n product over the 52 cards, ~0.1-0.2 s
        for a limped pot's 1,126 combos in Pyodide (no BLAS), which a combo space only counted (tree_cells, solve()'s
        river-class closure) never uses."""
        if self._compat is None:
            self._compat = 1.0 - np.clip(self.member.T @ self.member, 0, 1)
        return self._compat

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

    def ranking(self, boards) -> "Ranking":
        """Sorted-strength index for a list of river boards (shared by every terminal on them)."""
        key = ("rank", tuple(boards))
        R = self._stacks.get(key)
        if R is None:
            R = Ranking(self, [self.strength(b) for b in boards])
            self._stacks[key] = R
        return R

    def turn_picks(self, flop: tuple, dead: set, k: int) -> list:
        """k turn cards for `flop`, sorted, the same for every caller: picks are cached per flop in
        the order they were drawn, so a call for fewer than are cached gets the first k drawn (a
        random subset), not the lowest cards."""
        key = tuple(sorted(flop))
        cached = self._turn_picks.get(key)
        avail = [c for c in range(52) if c not in dead and c not in flop]
        if cached is None:
            # every card when k covers them (no draw: the order is not random yet)
            cached = (list(avail), False) if k >= len(avail) else (self.rng.sample(avail, k), True)
        elif len(cached[0]) < k:
            extra = [c for c in avail if c not in cached[0]]
            cached = (cached[0] + self.rng.sample(extra, min(k - len(cached[0]), len(extra))), True)
        elif len(cached[0]) > k and not cached[1]:
            cached = (self.rng.sample(cached[0], len(cached[0])), True)
        self._turn_picks[key] = cached
        return sorted(cached[0][:k])

    def turn_classes(self, flop: tuple, dead: set) -> list:
        """Every turn card in a class: [(the card played, the class's cards)], sorted. A class is one rank's cards,
        except that a suit with two or more cards on the flop has its own class per rank (its flush draws); the
        rest of the rank is played as one of its suits, rotating over the flop's suits rank by rank."""
        avail = [c for c in range(52) if c not in dead and c not in flop]
        fs = [c % 4 for c in flop]
        flushy = {x for x in range(4) if fs.count(x) >= 2}
        order = [x for x in sorted(set(fs), key=lambda x: -fs.count(x)) + [x for x in range(4) if x not in fs]
                 if x not in flushy]
        out = []
        for r in range(13):
            cs = [c for c in avail if c // 4 == r]
            out += [(c, (c,)) for c in cs if c % 4 in flushy]
            rest = [c for c in cs if c % 4 not in flushy]
            if rest:
                pref = order[r % len(order):] + order[:r % len(order)]
                out.append((next(c for x in pref for c in rest if c % 4 == x), tuple(rest)))
        return sorted(out)

    def class_perm(self, classes) -> tuple:
        """For river classes (Config.river_classes): (G, flat, starts, sizes). Row e of G (one per card c of each class,
        the classes in order) maps every combo to its image under the swap of the class's played card and c (identity
        for c itself); flat is G offset by its class's index times n (one gather over (K * n)); starts: each class's
        first row; sizes: the classes' sizes. The combo space holds every active combo's images (solve() adds them); an
        image outside it (of a combo added that way, so with no reach of its own) is n in G, a zero reach column, and
        the combo itself in flat (its utility is never used)."""
        n = self.n
        pos = np.full(N_FULL, -1, dtype=np.intp)
        pos[self.active] = np.arange(n)
        G, cls_of, starts, sizes = [], [], [], []
        for k, (played, cs) in enumerate(classes):
            starts.append(len(G)); sizes.append(len(cs))
            for c in cs:
                a = np.where(self.ca == played, c, np.where(self.ca == c, played, self.ca))
                b = np.where(self.cb == played, c, np.where(self.cb == c, played, self.cb))
                G.append(pos[COMBO_TABLE[a, b]]); cls_of.append(k)
        G = np.array(G, dtype=np.intp).reshape(len(G), n)
        out = G < 0
        G[out] = n
        flat = np.array(cls_of, dtype=np.intp)[:, None] * n + np.where(out, np.arange(n)[None, :], G)
        return G, flat, np.array(starts, dtype=np.intp), np.array(sizes, dtype=np.float32)

    def leaf_matrix(self, board: tuple, dead: set) -> np.ndarray:
        """Average showdown matrix over the remaining runout(s) from `board`."""
        key = tuple(sorted(board))
        L = self._leaf.get(key)
        if L is not None:
            return L
        if len(board) == 5:
            L = self.showdown_matrix(board)
        elif len(board) == 3:
            L = self._flop_leaf(board, dead)
        elif self.root_len == 3:
            # a turn leaf inside a flop solve (an all-in or a check-down on the turn): every river, uncached.
            # leaf_rivers_flop random rivers (4 in the browser's Fast tree) moved a flush draw's turn equity by
            # tens of points, and a 4bet pot's flop with it: JsTs jammed Ks7s2d at +28.1 over a third-pot bet's
            # +26.4, against +27.2 and +31.4 with every river
            avail = [c for c in range(52) if c not in dead and c not in board]
            L = self._avg_showdown([board + (r,) for r in avail])
        else:
            avail = [c for c in range(52) if c not in dead and c not in board]
            if len(board) == 3:
                picks = self.turn_picks(board, dead, self.cfg.leaf_turns)
            else:
                lr = self.cfg.leaf_rivers_flop if self.root_len == 3 else self.cfg.leaf_rivers
                k = len(avail) if lr is None else min(lr, len(avail))
                picks = avail if k >= len(avail) else sorted(self.rng.sample(avail, k))
                if k >= len(avail):
                    # every river: the same numbers by _avg_showdown (a count of wins a pair, in the core where it
                    # lays a program out) without a float showdown matrix a river, each kept: 48 of 5 MB for a
                    # limped pot's 1,126 combos, 0.6 s natively of a turn solve's setup and ~1.3 s in Pyodide
                    L = self._avg_showdown([board + (c,) for c in picks])
                    self._leaf[key] = L
                    return L
            L = np.zeros((self.n, self.n), dtype=np.float32)
            for c in picks:
                L += self.leaf_matrix(board + (c,), dead)
            # average per pair over the picked cards neither hand holds (the sub-matrices
            # are zero for the rest), not over all picks
            h = self.member[picks].sum(axis=0)
            L /= np.maximum(len(picks) - h[:, None] - h[None, :], 1.0).astype(np.float32)
        self._leaf[key] = L
        return L

    def bucket_table(self, board: tuple, dead: set, K: int) -> tuple:
        """River buckets for a turn board (Config.river_buckets): (ids, E, kappa). ids[q]: each combo's bucket for
        player q (K: not in q's range on this board). Buckets split q's root range (self.root_reach) into K equal
        shares ordered by its equity against the opponent's root range here (the turn leaf matrix, every river).
        E[q] (K x K): q's bucket against the opponent's, the root-reach-weighted mean over their compatible pairs of
        the turn leaf matrix times their compatible share, so that E[q] @ (the opponent's bucket masses) is the leaf
        matrix's product with a reach spread inside each bucket like the root range; kappa[q]: the compatible share
        (folds). E[1] = -E[0].T and kappa[1] = kappa[0].T, so a bucket game is zero-sum."""
        key = ("bucket", tuple(sorted(board)), K)
        T = self._stacks.get(key)
        if T is not None:
            return T
        n = self.n
        step = self.cfg.bucket_river_step
        avail = [c for c in range(52) if c not in dead and c not in board]
        live = self.member[list(board)].sum(axis=0) == 0
        w = [np.where(live, self.root_reach[q], 0.0) for q in (0, 1)]
        # every step-th river card (step coprime to 4 keeps the suits and ranks even); every one is leaf_matrix's
        # inside a flop solve
        rivers = avail[board[-1] % step::step] if step > 1 else avail if self.root_len == 3 else None
        if self.tables is not None and rivers is not None:
            T = self.tables.bucket_table(self, self._strengths([tuple(board) + (r,) for r in rivers]), w, K)
            self._stacks[key] = T
            return T
        if step > 1:
            L = self._avg_showdown([tuple(board) + (r,) for r in rivers])
        else:
            L = self.leaf_matrix(board, dead)
        ids = np.full((2, n), K, dtype=np.intp)
        orders, ends = [], []
        for q in (0, 1):
            eq = np.einsum("ij,j->i", L, w[1 - q]) / np.maximum(np.einsum("ij,j->i", self.compat, w[1 - q]), 1e-12)
            idx = np.flatnonzero(w[q] > 0)
            order = idx[np.argsort(eq_key(eq[idx]), kind="stable")]
            ws = w[q][order]
            cum = np.cumsum(ws)
            # (equal shares of the range; bucket edges evenly spaced in equity, or half and half, were 0.5-0.8 points
            # further from a solve over every turn in c-bet frequency)
            b = np.minimum(((cum - ws / 2) / max(cum[-1], 1e-12) * K).astype(np.intp), K - 1) if len(order) else order
            ids[q, order] = b
            orders.append(order)
            ends.append(np.searchsorted(b, np.arange(K + 1), "left"))     # bucket k's combos: order[ends[k]:ends[k+1]]

        def group_sums(A, q, axis):
            """Sums of A's entries along `axis` over q's buckets, weighted by q's root reach."""
            o = orders[q]
            if not len(o):   # (q's range is empty on this turn: a one-combo range the turn card blocks)
                shape = list(A.shape); shape[axis] = K
                return np.zeros(shape)
            X = np.take(A, o, axis=axis) * (w[q][o][:, None] if axis == 0 else w[q][o][None, :])
            cs = np.cumsum(X, axis=axis)
            cs = np.concatenate([np.zeros_like(np.take(cs, [0], axis=axis)), cs], axis=axis)
            e = ends[q]
            return np.take(cs, e[1:], axis=axis) - np.take(cs, e[:-1], axis=axis)

        W = [np.array([w[q][orders[q][ends[q][k]:ends[q][k + 1]]].sum() for k in range(K)]) for q in (0, 1)]
        den = np.outer(W[0], W[1])
        den = np.where(den > 0, den, np.inf)
        num = group_sums(group_sums(L.astype(np.float64), 1, 1), 0, 0) / den
        pm = group_sums(group_sums(self.compat, 1, 1), 0, 0) / den
        T = (ids, np.stack([num, -num.T]).astype(np.float32), np.stack([pm, pm.T]).astype(np.float32))
        self._stacks[key] = T
        return T

    def _compat_rows(self, ro: np.ndarray) -> np.ndarray:
        """ro: (B, N) -> (B, N) mass of the combos sharing no card with each combo (Solver.compat_sum)."""
        cs = (ro @ self.member.T)[:, self.cab]
        n = self.n
        return ro.sum(axis=1, keepdims=True) - cs[:, :n] - cs[:, n:] + ro

    def _compat_cols(self, X: np.ndarray) -> np.ndarray:
        """X: (B, N, K) -> (B, N, K) each column's mass of the combos sharing no card with each combo."""
        n = self.n
        cs = np.add.reduceat(X[:, self.card_perm, :], self.card_starts, axis=1)[:, self.cab_held, :]
        return X.sum(axis=1, keepdims=True) - cs[:, :n] - cs[:, n:] + X

    def river_table(self, board: tuple, dead: set, K: int, outer: Optional[int]) -> tuple:
        """River-strength buckets for a turn board (Config.river_strength): (X, E, kappa). On each river card (every
        bucket_river_step-th) each player q's root range (self.root_reach) splits into K equal shares by equity there
        against the opponent's root range. X[q] (N x K, or K_t x K from `outer` turn buckets: Ctx.bucket_table's) is
        the share of the rivers that put each combo (each turn bucket, its combos weighted by the root range) in each
        river bucket: a reach r enters the river as r @ X[q], and the river's bucket utilities u come back as X[q] @ u.
        kappa[q] (K x K): the compatible share of the opponent's bucket mass, as it is spread by X at root reach (folds);
        E[q]: kappa times the mean showdown (+1 / -1) of the bucket pair's compatible pairs over the rivers, per pair
        of hands on the same river card. E[1] = -E[0].T and kappa[1] = kappa[0].T: the game stays zero-sum."""
        key = ("river", tuple(sorted(board)), K, outer)
        T = self._stacks.get(key)
        if T is not None:
            return T
        n = self.n
        step = self.cfg.bucket_river_step
        avail = [c for c in range(52) if c not in dead and c not in board]
        rivers = avail[board[-1] % step::step] if step > 1 else avail
        boards = [tuple(board) + (r,) for r in rivers]
        if self.tables is not None:
            r = [np.where(self.member[list(board)].sum(axis=0) == 0, self.root_reach[q], 0.0) for q in (0, 1)]
            tids = self.bucket_table(board, dead, outer)[0] if outer else None
            T = self.tables.river_table(self, self._strengths(boards), r, K, tids, outer)
            self._stacks[key] = T
            return T
        rk = Ranking(self, list(self._strengths(boards)))
        R = len(boards)
        live = (~rk.blocked).astype(np.float64)                         # (R, n): holds no board card
        w = [live * self.root_reach[q][None, :] for q in (0, 1)]
        ids = []
        for q in (0, 1):
            eq = rk.utility(w[1 - q]) / np.maximum(self._compat_rows(w[1 - q]), 1e-12)
            on = w[q] > 0
            order = np.argsort(np.where(on, eq_key(eq), np.inf), axis=1, kind="stable")
            ws = np.take_along_axis(w[q], order, axis=1)
            cum = np.cumsum(ws, axis=1)
            b = np.minimum(((cum - ws / 2) / np.maximum(cum[:, -1:], 1e-12) * K).astype(np.intp), K - 1)
            b[ws <= 0] = K
            bid = np.empty_like(b)
            np.put_along_axis(bid, order, b, axis=1)
            ids.append(bid)
        rows = np.arange(n)[None, :] * (K + 1)
        P = []
        for q in (0, 1):
            cnt = np.bincount((rows + ids[q]).ravel(), minlength=n * (K + 1)).reshape(n, K + 1)[:, :K].astype(np.float64)
            P.append(cnt / np.maximum(cnt.sum(axis=1, keepdims=True), 1.0))
        # the bucket pairs' showdown sums and compatible masses, on the same river card
        # (K opponent buckets as columns of one pass, a few rivers at a time: (rivers, n, K) arrays)
        num = np.zeros((K + 1, K)); den = np.zeros((K + 1, K))
        cr = max(1, 4096 // max(n, 1) * 16 // K)
        for lo in range(0, R, cr):
            sel = np.arange(lo, min(R, lo + cr))
            b = len(sel)
            ro = np.zeros((b, n, K + 1), dtype=np.float32)
            np.put_along_axis(ro, ids[1][sel][:, :, None], w[1][sel][:, :, None].astype(np.float32), axis=2)
            ro = ro[:, :, :K]
            u = rk._sub(sel).utility_cols(ro)
            c = self._compat_cols(ro)
            order = np.argsort(ids[0][sel].ravel(), kind="stable")
            bk = ids[0][sel].ravel()[order]
            starts = np.searchsorted(bk, np.arange(K + 1))
            w0 = w[0][sel].reshape(-1, 1)[order].astype(np.float32)
            for acc, X in ((num, u), (den, c)):
                Y = np.concatenate([X.reshape(-1, K)[order] * w0, np.zeros((1, K), dtype=np.float32)])
                part = np.add.reduceat(Y, np.minimum(starts, len(Y) - 1), axis=0)
                part[starts >= len(Y) - 1] = 0
                part[np.append(starts[1:] == starts[:-1], False)] = 0
                acc += part
        num, den = num[:K], den[:K]
        sd = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        # kappa at the reaches X spreads: A_q = root reach x X[q] (n, K); compat @ A1 by per-card sums
        r = [np.where(self.member[list(board)].sum(axis=0) == 0, self.root_reach[q], 0.0) for q in (0, 1)]
        A = [r[q][:, None] * P[q] for q in (0, 1)]
        cA1 = (self._compat_rows(A[1].T)).T                            # (n, K)
        M = [A[q].sum(axis=0) for q in (0, 1)]
        mm = np.outer(M[0], M[1])
        kap = np.where(mm > 0, (A[0].T @ cA1) / np.where(mm > 0, mm, 1.0), 0.0)
        E = sd * kap
        if outer:
            tids = self.bucket_table(board, dead, outer)[0]
            X = []
            for q in (0, 1):
                Q = np.zeros((outer + 1, K))
                np.add.at(Q, tids[q], A[q])
                wt = np.bincount(tids[q], weights=r[q], minlength=outer + 1)
                X.append((Q / np.maximum(wt, 1e-300)[:, None])[:outer])
        else:
            X = P
        T = (np.stack(X).astype(np.float32), np.stack([E, -E.T]).astype(np.float32),
             np.stack([kap, kap.T]).astype(np.float32))
        self._stacks[key] = T
        return T

    def river_card_table(self, board: tuple, dead: set, K: int) -> tuple:
        """Buckets for a river board (Config.river_dealt), as bucket_table's (ids, E, kappa): each player q's root
        range (self.root_reach) on this board split into K equal shares by equity here against the opponent's; E[q]
        the bucket pairs' mean showdown (+1 / -1) over their compatible pairs times their compatible share, kappa[q]
        the share. Computed for every river under the board's turn at once."""
        key = ("rcard", tuple(board), K)
        T = self._stacks.get(key)
        if T is not None:
            return T
        n = self.n
        turn = tuple(board[:4])
        boards = [turn + (r,) for r in range(52) if r not in dead and r not in turn]
        rk = Ranking(self, list(self._strengths(boards)))
        R = len(boards)
        live = (~rk.blocked).astype(np.float64)                         # (R, n): holds no board card
        w = [live * self.root_reach[q][None, :] for q in (0, 1)]
        ids = []
        for q in (0, 1):
            eq = rk.utility(w[1 - q]) / np.maximum(self._compat_rows(w[1 - q]), 1e-12)
            order = np.argsort(np.where(w[q] > 0, eq, np.inf), axis=1, kind="stable")
            ws = np.take_along_axis(w[q], order, axis=1)
            cum = np.cumsum(ws, axis=1)
            b = np.minimum(((cum - ws / 2) / np.maximum(cum[:, -1:], 1e-12) * K).astype(np.intp), K - 1)
            b[ws <= 0] = K
            bid = np.empty_like(b)
            np.put_along_axis(bid, order, b, axis=1)
            ids.append(bid)
        # per river: the bucket pairs' showdown sums and compatible masses (opponent buckets as columns), in chunks
        num, den = np.zeros((R, K, K)), np.zeros((R, K, K))
        cr = max(1, 4096 // max(n, 1) * 16 // K)
        for lo in range(0, R, cr):
            sel = np.arange(lo, min(R, lo + cr))
            oh = []
            for q in (0, 1):
                x = np.zeros((len(sel), n, K + 1), dtype=np.float32)
                np.put_along_axis(x, ids[q][sel][:, :, None], w[q][sel][:, :, None].astype(np.float32), axis=2)
                oh.append(x[:, :, :K])
            ohT = oh[0].transpose(0, 2, 1)
            num[sel] = np.matmul(ohT, rk._sub(sel).utility_cols(oh[1]))
            den[sel] = np.matmul(ohT, self._compat_cols(oh[1]))
        M = [np.stack([np.bincount(ids[q][i], weights=w[q][i], minlength=K + 1)[:K] for i in range(R)]) for q in (0, 1)]
        mm = M[0][:, :, None] * M[1][:, None, :]
        ok = mm > 0
        E = np.where(ok, num / np.where(ok, mm, 1.0), 0.0)
        kap = np.where(ok, den / np.where(ok, mm, 1.0), 0.0)
        for i, bd in enumerate(boards):
            self._stacks[("rcard", bd, K)] = (np.stack([ids[0][i], ids[1][i]]),
                                              np.stack([E[i], -E[i].T]).astype(np.float32),
                                              np.stack([kap[i], kap[i].T]).astype(np.float32))
        return self._stacks[key]

    def _flop_leaf(self, board: tuple, dead: set) -> np.ndarray:
        """The flop's equity matrix (a flop all-in): every turn card, each with FLOP_LEAF_RIVERS or more
        rivers (leaf_turns x leaf_rivers_flop runouts in all, spread over the turns). Sampled turns gave a
        draw a share of its outs set by which few turns came: with 4 turns x 4 rivers, JsTs called a flop jam
        on Ks7s2d at +19.5 bb where the default tree's 6 x 16 runouts had it at -18.6. The river boards'
        matrices aren't kept (a few hundred of them)."""
        avail = [c for c in range(52) if c not in dead and c not in board]
        per = max(FLOP_LEAF_RIVERS, -(-self.cfg.leaf_turns * self.cfg.leaf_rivers_flop // max(len(avail), 1)))
        m = len(avail)
        step = max(1, (m - 1) // per)
        boards = []
        for ti, t in enumerate(avail):
            # rivers spread evenly so every card is a river about equally often over all the turns (random
            # picks left a flush draw's river outs over- or under-drawn: a 5-6% equity error at 2 per turn)
            rivers = [avail[(ti + 1 + j * step) % m] for j in range(min(per, m - 1))]
            boards += [board + (t, r) for r in dict.fromkeys(rivers)]
        return self._avg_showdown(boards)

    def _strengths(self, boards) -> np.ndarray:
        """(boards, n) hand strengths on 5-card boards (-1: holds a board card). Those of a turn's rivers are kept,
        for the bucket tables and the river-strength tables both (river_strength)."""
        key = ("S", tuple(boards))
        S = self._stacks.get(key)
        if S is None:
            S = np.concatenate([eval_boards(boards[i:i + 64], self.active_list) for i in range(0, len(boards), 64)]) \
                if boards else np.zeros((0, self.n), dtype=np.int32)
            if len(boards) <= 52:
                self._stacks[key] = S
        return S

    def _avg_showdown(self, boards) -> np.ndarray:
        """The showdown matrix averaged over 5-card boards, per pair over the boards neither hand blocks;
        the boards' own matrices aren't kept."""
        if self.tables is not None:
            return self.tables.avg_showdown(self, self._strengths(boards))
        # sum over boards of sign(s_i - s_j) = G - G.T, G counting the boards where i beats j: one comparison and
        # one small-integer add per board (a float sign matrix and its masking were ~5x the work). A combo holding
        # a board card has strength NaN there, so it neither beats nor loses to anything.
        G = np.zeros((self.n, self.n), dtype=np.uint8 if len(boards) < 256 else np.uint16)
        gt = np.empty((self.n, self.n), dtype=bool)
        gv = gt.view(np.uint8)
        S = self._strengths(boards).astype(np.float32)
        bl = S < 0                           # combos holding a card of the board
        S[bl] = np.nan
        for s in S:
            np.greater(s[:, None], s[None, :], out=gt)
            np.add(G, gv, out=G, casting="unsafe")
        ok = (~bl).astype(np.float32)        # (boards, n): the boards each pair is counted on
        cnt = np.einsum("bi,bj->ij", ok, ok) if NO_BLAS else ok.T @ ok
        L = G.astype(np.float32)
        L = L - L.T
        return (L / np.maximum(cnt, 1.0) * self.compat).astype(np.float32)


def quarter_cumsum(x: np.ndarray, out: np.ndarray) -> None:
    """out[:, 0] = 0 and out[:, 1:] = x's running sums along axis 1, summed as the compiled core's prefix() sums them
    (core/gtocore.c: one running sum per quarter, at once, then each quarter's offset added), so the two round alike."""
    m = x.shape[1]
    q = m // 4
    out[:, 0] = 0
    if q == 0:
        np.cumsum(x, axis=1, out=out[:, 1:])
        return
    b = (0, q, 2 * q, 3 * q, m)
    for s, e in zip(b, b[1:]):
        np.cumsum(x[:, s:e], axis=1, out=out[:, s + 1:e + 1])
    o1 = out[:, q].copy()
    o2 = o1 + out[:, 2 * q]
    o3 = o2 + out[:, 3 * q]
    out[:, q + 1:2 * q + 1] += o1[:, None]
    out[:, 2 * q + 1:3 * q + 1] += o2[:, None]
    out[:, 3 * q + 1:] += o3[:, None]


class Ranking:
    """Showdown utilities in O(N) per board instead of an N x N matrix product.

    For combo i, wins = reach of all weaker opponent combos minus the weaker ones that share
    a card with i, and likewise for losses. The card-sharing part comes from the (card, combo)
    pairs sorted by card, then strength: one running sum over them gives every card's combos
    weaker or stronger than i. Only i itself holds both of i's cards and it is neither weaker
    nor stronger than itself, so no inclusion-exclusion term is needed."""

    def __init__(self, ctx: "Ctx", strengths):
        n = ctx.n
        B = len(strengths)
        S = np.stack(strengths).astype(np.float64)            # (B, n); -1 = holds a board card
        rows = np.arange(B)[:, None]
        # every board's rows at once: each board's keys are offset by the row's index times a span
        # wider than any key, so one searchsorted serves them all
        self.order = np.argsort(S, axis=1, kind="stable")
        ss = np.take_along_axis(S, self.order, axis=1)
        flat = (ss + rows * 1e4).ravel()
        q = (S + rows * 1e4).ravel()
        self.lo = np.searchsorted(flat, q, "left").reshape(B, n) - rows * n      # weaker combos
        self.hi = np.searchsorted(flat, q, "right").reshape(B, n) - rows * n     # weaker or tied
        # (card, combo) pairs, two per combo, sorted by card then strength; strengths are under 1e4
        pc = np.concatenate([ctx.ca, ctx.cb])
        pi = np.concatenate([np.arange(n), np.arange(n)])
        key = pc[None, :] * 1e4 + S[:, pi]                    # (B, 2n)
        po = np.argsort(key, axis=1, kind="stable")
        self.pcombo = pi[po]
        skey = (np.take_along_axis(key, po, axis=1) + rows * 1e6).ravel()
        self.plo = np.empty((B, 2, n), dtype=np.intp); self.phi = np.empty((B, 2, n), dtype=np.intp)
        for slot, cards in enumerate((ctx.ca, ctx.cb)):
            q = (cards[None, :] * 1e4 + S + rows * 1e6).ravel()
            self.plo[:, slot] = np.searchsorted(skey, q, "left").reshape(B, n) - rows * 2 * n
            self.phi[:, slot] = np.searchsorted(skey, q, "right").reshape(B, n) - rows * 2 * n
        counts = np.bincount(pc, minlength=52)
        self.cstart = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.cend = self.cstart + counts
        self.ca, self.cb = ctx.ca, ctx.cb
        self.blocked = S < 0
        self.n = n
        self._flat()

    def _flat(self):
        """Flat indices into raveled (B, ...) arrays for every lookup in utility()."""
        B, n = self.order.shape
        r = np.arange(B)[:, None]
        self.B = B
        self.fsort = (r * n + self.order).ravel()
        self.fwin = (r * (n + 1) + self.lo).ravel()
        self.flose = (r * (n + 1) + self.hi).ravel()
        self.pgather = (r * n + self.pcombo).ravel()
        self.f4 = np.concatenate([(r * (2 * n + 1) + a[:, s]).ravel() for s in (0, 1) for a in (self.plo, self.phi)])
        self.fK = np.concatenate([(r * 52 + c[None, :]).ravel() for c in (self.ca, self.cb)])
        self.live = (~self.blocked).astype(np.float32)

    def utility(self, ro: np.ndarray, sel=None) -> np.ndarray:
        """ro: (B, n) opponent reach -> (B, n) sum over compatible opponent combos of
        +reach (weaker) / -reach (stronger). With `sel`, ro holds only those boards' rows."""
        if sel is not None:
            return self._sub(sel).utility(ro)
        B, n = self.B, self.n
        # (fancy indexing with intp indices: np.take, and int64 indices on 32-bit Pyodide, are several times slower)
        cum = np.empty((B, n + 1), dtype=ro.dtype)
        ro = ro.ravel()
        quarter_cumsum(ro[self.fsort].reshape(B, n), cum)
        cs = np.empty((B, 2 * n + 1), dtype=ro.dtype)
        quarter_cumsum(ro[self.pgather].reshape(B, 2 * n), cs)
        cum, total = cum.ravel(), cum[:, n:]
        # wins - losses: weaker reach + (weaker or tied) reach - all of it
        u = cum[self.fwin]
        u += cum[self.flose]
        # card-sharing corrections, per card c of i: -weaker + stronger among c's combos,
        # = (c's total + 2 * the sum before c's pairs) - sum to i's weaker - sum to i's weaker or tied
        k = (cs[:, self.cend] + cs[:, self.cstart]).ravel()[self.fK]        # (2 B n,)
        u += k[:B * n]; u += k[B * n:]
        q = cs.ravel()[self.f4]
        for j in range(4):
            u -= q[j * B * n:(j + 1) * B * n]
        u = u.reshape(B, n)
        u -= total
        u *= self.live
        return u

    def utility_cols(self, ro: np.ndarray) -> np.ndarray:
        """utility() for K reach vectors at once: ro (B, n, K) -> (B, n, K)."""
        B, n = self.B, self.n
        K = ro.shape[2]
        cum = np.zeros((B, n + 1, K), dtype=ro.dtype)
        r2 = ro.reshape(B * n, K)
        quarter_cumsum(r2[self.fsort].reshape(B, n, K), cum)
        cs = np.zeros((B, 2 * n + 1, K), dtype=ro.dtype)
        quarter_cumsum(r2[self.pgather].reshape(B, 2 * n, K), cs)
        total = cum[:, n:, :]
        c2 = cum.reshape(-1, K)
        u = c2[self.fwin] + c2[self.flose]
        k = (cs[:, self.cend] + cs[:, self.cstart]).reshape(-1, K)[self.fK]
        u += k[:B * n]; u += k[B * n:]
        q = cs.reshape(-1, K)[self.f4]
        for j in range(4):
            u -= q[j * B * n:(j + 1) * B * n]
        u = u.reshape(B, n, K)
        u -= total
        u *= self.live[:, :, None]
        return u

    def _sub(self, sel) -> "Ranking":
        """The same index restricted to boards `sel` (flat offsets rebuilt for fewer rows)."""
        r = Ranking.__new__(Ranking)
        for f in ("order", "lo", "hi", "pcombo", "plo", "phi", "blocked"):
            setattr(r, f, getattr(self, f)[sel])
        r.cstart, r.cend, r.ca, r.cb, r.n = self.cstart, self.cend, self.ca, self.cb, self.n
        r._flat()
        return r


# ----------------------------------------------------------------- tree nodes
class Terminal:
    """`inv`: each player's chips in, floats, or (B, 1) float32 columns for a merged terminal (one
    value per batch row). `tile`: set on a merged equity leaf whose rows repeat the same `tile` boards.
    `rake`: what the winner pays out of the pot (a float, or a (B, 1) column when merged); None = no rake.
    `pay`: a tournament's payoffs (Config.icm), in chip equivalents: at a fold (v0, v1), each player's value; at a
    showdown or equity leaf (A0, B0, A1, B1), player p's value being A_p + B_p (P(win) - P(lose)). None = chips."""
    __slots__ = ("boards", "kind", "folder", "inv", "M", "tile", "rake", "pay")

    def __init__(self, boards, kind, inv, folder=None, rake=None, pay=None):
        self.boards, self.kind, self.inv, self.folder = boards, kind, inv, folder
        self.M = None
        self.tile = None
        self.rake = rake if rake is None or np.any(rake > 0) else None
        self.pay = pay

    def fold_pay(self, p):
        """p's chips won (lost when negative) per opponent combo at a fold."""
        if self.pay is not None:
            return self.pay[p]
        if self.folder == p:
            return -self.inv[p]
        return self.inv[1 - p] if self.rake is None else self.inv[1 - p] - self.rake


class Chance:
    """Parent batch B, K cards per parent element; the child has batch B*K.

    A combo's summed child utility is divided by the number of traversed cards it doesn't
    hold, times the share of those the opponent's hand doesn't hold either
    ((avail - 4) / (avail - 2)). With every card enumerated that is exactly the avail - 4
    runouts possible for any pair of hands.

    With `sample` set, each traversal visits a fresh random subset of `sample` of the K
    cards (public chance sampling). Every card keeps its own strategy, so the solver still
    solves the real game rather than one where only a few fixed cards can come."""
    __slots__ = ("boards", "K", "masks", "child", "norm", "avail", "sample", "umasks", "perm")

    def __init__(self, boards, K, masks, child, avail, sample=None, umasks=None, perm=None):
        self.boards, self.K, self.masks, self.child = boards, K, masks, child
        # river classes (Config.river_classes): child k stands for every card of class k through card swaps
        # (Ctx.class_perm, spread, gather); masks are then the played card's live combos
        self.perm = perm
        self.avail = avail
        self.sample = sample if sample is not None and sample < K else None
        # a child standing for several cards (turn_classes): masks are the share of its cards a combo leaves live
        # (its reach there), umasks how many (its utility's weight); a pair's weight is then symmetric, and with
        # every combo's utility over the same divisor (as with every card enumerated) the game stays zero-sum
        self.umasks = umasks
        B = len(boards)
        if perm is not None:
            self.norm = np.full((B, masks.shape[1]), float(avail - 4), dtype=np.float32)
        elif umasks is None:
            self.norm = self.norm_for(masks.reshape(B, K, -1))
        else:
            self.norm = np.full((B, masks.shape[1]), float(avail - 4), dtype=np.float32)

    def norm_for(self, m3: np.ndarray) -> np.ndarray:
        """m3: (B, k, N) masks of the traversed cards -> (B, N) divisor."""
        return np.maximum(m3.sum(axis=1) * ((self.avail - 4) / (self.avail - 2)), 1e-9)

    def spread(self, r, rows) -> np.ndarray:
        """River classes: (B, N) reaches -> (B * K, N) the class subtrees' reaches: a combo's reach in class k is the
        mean over the class's cards c of its image's reach under the swap (played card <-> c), if the played card
        leaves it live."""
        G, _, starts, sizes = self.perm
        B, n = r.shape
        rp = np.zeros((B, n + 1), dtype=r.dtype)
        rp[:, :n] = r
        R = np.add.reduceat(rp[:, G], starts, axis=1)
        R *= (self.masks.reshape(-1, self.K, n)[rows] / sizes[None, :, None]).astype(R.dtype, copy=False)
        return R.reshape(B * self.K, n)

    def gather(self, u, rows) -> np.ndarray:
        """River classes: (B * K, N) class subtree utilities -> (B, N): a combo's utility sums, over every river card
        c, its image's utility in c's class (the image under the swap), divided by the avail - 4 runouts of a pair."""
        n = u.shape[1]
        B = len(rows)
        u = u.reshape(B, self.K * n) * self.masks.reshape(-1, self.K * n)[rows]
        return u[:, self.perm[1]].sum(axis=1) / self.norm[rows]

    def weights(self, rows=None) -> np.ndarray:
        """(B, K, N) weights of each child's utility, for batch rows `rows` (None = all)."""
        m = (self.masks if self.umasks is None else self.umasks).reshape(-1, self.K, self.masks.shape[1])
        return m if rows is None else m[rows]


class BucketEntry:
    """The river beyond `depth` played by hand buckets (Config.river_buckets): the reaches coming in are summed
    into each player's buckets for the row's turn board, the subtree below (Decisions K wide, terminals "bfold" and
    "bleaf") is solved on those bucket masses, and its bucket utilities go back to every combo of the bucket.
    Each row's boards repeat with period `tile` (the turn cards under one flop), so the bucket tables are per board
    and broadcast over the repeats."""
    __slots__ = ("boards", "child", "tile")

    def __init__(self, boards, child):
        self.boards, self.child = boards, child
        self.tile = len(boards)


class RiverEntry(BucketEntry):
    """The river beyond `depth` played by river-strength buckets (Config.river_strength): the reaches coming in (per
    combo, or per turn bucket: `outer`) are spread over each player's river buckets by Ctx.river_table's X, the
    subtree below (Decisions K wide, terminals "rfold" and "rleaf") is solved on those masses, and its bucket
    utilities go back through X."""
    __slots__ = ("outer",)

    def __init__(self, boards, child, outer=None):
        super().__init__(boards, child)
        self.outer = outer


class Decision:
    __slots__ = ("boards", "player", "names", "children", "regret", "ssum", "pred", "pot_now", "bets", "B", "blind",
                 "bucketed")

    def __init__(self, boards, player, names, children, pot_now, bets, n, blind=False, bucketed=False):
        self.boards, self.player, self.names, self.children = boards, player, names, children
        self.blind = blind        # a river decision played on the turn's boards (river_play)
        self.bucketed = bucketed  # its last axis is hand buckets, not combos (river_buckets)
        self.pot_now, self.bets = pot_now, bets
        self.B = len(boards)
        A = len(names)
        self.regret = np.zeros((self.B, A, n), dtype=np.float32)
        self.ssum = np.zeros((self.B, A, n), dtype=np.float32)
        self.pred = None          # the last instantaneous regrets, a predictive solver's prediction (Config.cfr)

    @property
    def board(self):
        return self.boards[0]

    def strategy(self, sel=None) -> np.ndarray:
        # regret matching in as few numpy calls as possible (a native solve is call-overhead bound): one
        # reciprocal a combo and an in-place multiply (as the compiled core does it), then uniform wherever no regret
        # is positive (rare once a solve is under way) or the positive ones sum under TINY
        if self.pred is None:
            pos = np.maximum(self.regret if sel is None else self.regret[sel], 0.0)
        else:
            # predictive regret matching: the regrets plus the prediction of the next iteration's (the last one's)
            pos = self.regret + self.pred if sel is None else self.regret[sel] + self.pred[sel]
            np.maximum(pos, 0.0, out=pos)
        tot = pos.sum(axis=1, keepdims=True)
        np.multiply(pos, 1 / np.maximum(tot, TINY), out=pos)
        zero = tot < TINY
        if zero.any():
            np.copyto(pos, np.float32(1.0 / len(self.names)), where=zero)
        return pos

    def avg_strategy(self, sel=None) -> np.ndarray:
        """(B, A, N) average strategy (rows `sel` only if given)."""
        ss = self.ssum if sel is None else self.ssum[sel]
        tot = ss.sum(axis=1, keepdims=True)
        return np.where(tot >= TINY, ss * (1 / np.maximum(tot, TINY)), 1.0 / len(self.names))

    def avg_strategy0(self) -> np.ndarray:
        """(A, N) average strategy of the first (or only) batch element."""
        return self.avg_strategy()[0]


# ----------------------------------------------------------------- builder
class TreeTooBig(Exception):
    """A tree past solve()'s max_cells: its decisions' boards x actions x width (combos or buckets), each a cell of
    the regrets, the strategy sums and the core's copies of them, so its memory and an iteration's time."""

    def __init__(self, cells: int, cap: int):
        super().__init__(f"the game tree has over {cells / 1e6:.1f}M cells, past {cap / 1e6:.1f}M")
        self.cells, self.cap = cells, cap


class TreeBuilder:
    def __init__(self, ctx: Ctx, dead: Sequence[int], max_cells: Optional[int] = None, dry: bool = False):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.rng = random.Random(self.cfg.seed + 1)
        self.dead = set(dead)
        self.n_nodes = 0
        self.cells = 0             # decisions' boards x actions x width so far (TreeTooBig past max_cells)
        self.max_cells = max_cells
        self.dry = dry             # count the cells only (tree_cells): no arrays, masks or payoffs, the nodes None
        self.root_street_len = 0
        self.root_spr = 0.0
        self._blind = False       # building a river played without its card (river_play)
        self._width = None        # ... by river_buckets hand buckets (its nodes' width) instead of per combo
        self._kind = "b"          # bucket terminals' kind prefix: "r" below a RiverEntry (river_strength)

    def build(self, board, pot, stack, bets=(0.0, 0.0), to_act=0, n_raises=0, inv=None):
        board = tuple(board)
        self.root_street_len = len(board)
        self.root_spr = stack / max(pot + bets[0] + bets[1], 1e-9)
        if inv is None:
            inv = (pot / 2 + bets[0], pot / 2 + bets[1])
        checked = (to_act == 1 and bets[0] == bets[1] and bets[0] == 0 and n_raises == 0)
        return self._decision([board], pot, stack, bets, to_act, n_raises, inv, checked=checked)

    def _street_over(self, boards, pot, stack, bets, inv, allin):
        new_pot = pot + bets[0] + bets[1]
        new_stack = stack - max(bets)
        if len(boards[0]) == 5:
            return Terminal(boards, self._kind + "leaf" if self._width else "showdown", inv, rake=self._rake(inv),
                            pay=self._pay(inv))
        if allin or new_stack <= 1e-9 or self._blind:
            return Terminal(boards, self._kind + "leaf" if self._width else "leaf", inv, rake=self._rake(inv),
                            pay=self._pay(inv))
        streets_used = len(boards[0]) - self.root_street_len + 1
        if streets_used >= self.cfg.depth:
            if self.cfg.river_play and self.cfg.river_dealt and self.root_street_len == 3 and len(boards[0]) == 4 \
                    and not self._width:
                return self._river_dealt(boards, new_pot, new_stack, inv)
            if self.cfg.river_play and self.root_street_len == 3 and len(boards[0]) == 4:
                # the river's betting on the turn's boards; its showdowns are the turn's equity leaves (every river)
                outer = self._width     # (already by buckets when the turn is: turn_buckets)
                rs = self.cfg.river_strength if self.root_spr >= self.cfg.river_strength_spr else None
                self._blind = True
                self._width = rs or self.cfg.river_buckets
                if rs:
                    self._kind = "r"
                try:
                    d = self._decision(boards, new_pot, new_stack, (0.0, 0.0), 0, 0, inv, False)
                finally:
                    self._blind = False
                    self._width = outer
                    self._kind = "b"
                if rs:
                    return RiverEntry(boards, d, outer)
                return BucketEntry(boards, d) if self.cfg.river_buckets and not outer else d
            # an equity leaf: raked on the pot as it stands. The leaf pays equity x this pot (later bets
            # aren't modelled, neither the chips they'd add nor their rake), so raking this pot keeps the
            # rake's share of what the leaf pays right uncapped; with a cap, the bigger final pot's share
            # would be lower, so it's slightly high there (in absolute bb, slightly low)
            return Terminal(boards, "leaf", inv, rake=self._rake(inv), pay=self._pay(inv))
        return self._chance(boards, new_pot, new_stack, inv)

    def _pay(self, inv, folder=None):
        """A tournament's payoffs at a terminal where the players have inv in (Terminal.pay), or None (chips)."""
        icm = self.cfg.icm
        if icm is None or self.dry:
            return None
        return icm.fold(inv, folder) if folder is not None else icm.showdown(inv)

    def _rake(self, inv):
        # an uncalled bet goes back to the bettor before the pot is raked: the pot is twice the smaller stake
        r = self.cfg.rake(2 * min(inv[0], inv[1]))
        return r if r > 0 else None

    def _next_cards(self, board, k: Optional[int]):
        if len(board) == 3:
            return self.ctx.turn_picks(board, self.dead, self.cfg.n_turns)
        avail = [c for c in range(52) if c not in self.dead and c not in board]
        if k is None or k >= len(avail):
            return avail
        return sorted(self.rng.sample(avail, k))

    def _chance(self, boards, pot, stack, inv):
        cfg = self.cfg
        if len(boards[0]) == 3 and cfg.turn_classes:
            return self._turn_classes(boards, pot, stack, inv)
        if len(boards[0]) == 4 and self.root_street_len == 4 and cfg.river_classes:
            return self._river_classes(boards, pot, stack, inv)
        sample = None
        if len(boards[0]) == 3:
            k = cfg.n_turns
        elif self.root_street_len == 3:
            k = cfg.n_rivers_from_flop
        else:
            k, sample = cfg.n_rivers, cfg.river_sample
        child_boards, cards = [], []
        for b in boards:
            cs = self._next_cards(b, k)
            if len(cs) > k if k is not None else False:
                cs = cs[:k]
            cards.append(cs)
        K = min(len(cs) for cs in cards)
        masks = None if self.dry else np.ones((len(boards) * K, self.ctx.n), dtype=np.float32)
        for bi, b in enumerate(boards):
            for ki in range(K):
                c = cards[bi][ki]
                child_boards.append(b + (c,))
                if masks is not None:
                    masks[bi * K + ki] = 1.0 - self.ctx.member[c]
        child = self._bucketed_turn(child_boards, pot, stack, inv) if len(boards[0]) == 3 else \
            self._decision(child_boards, pot, stack, (0.0, 0.0), 0, 0, inv, False)
        self.n_nodes += 1
        if self.dry:
            return None
        avail = sum(1 for c in range(52) if c not in self.dead and c not in boards[0])
        return Chance(boards, K, masks, child, avail, sample)

    def _turn_classes(self, boards, pot, stack, inv):
        """The turn as one card per class (Ctx.turn_classes). A pair of combos plays each class with the weight of
        its cards both leave live, approximated as size x (share live for one) x (share live for the other), and
        not at all when either holds the card played (a few percent of the turns for those combos)."""
        ctx = self.ctx
        cls = [ctx.turn_classes(b, self.dead) for b in boards]
        K = len(cls[0])
        child_boards = []
        masks = None if self.dry else np.ones((len(boards) * K, ctx.n), dtype=np.float32)
        umasks = None if self.dry else np.ones_like(masks)
        for bi, b in enumerate(boards):
            for ki, (c, cards_) in enumerate(cls[bi]):
                child_boards.append(b + (c,))
                if self.dry:
                    continue
                live = (len(cards_) - ctx.member[list(cards_)].sum(axis=0)) * (1.0 - ctx.member[c])
                masks[bi * K + ki] = live / len(cards_)
                umasks[bi * K + ki] = live
        child = self._bucketed_turn(child_boards, pot, stack, inv)
        self.n_nodes += 1
        if self.dry:
            return None
        avail = sum(1 for c in range(52) if c not in self.dead and c not in boards[0])
        return Chance(boards, K, masks, child, avail, umasks=umasks)

    def _river_classes(self, boards, pot, stack, inv):
        """The river as one card per class under a turn root (Config.river_classes; Chance.spread and gather)."""
        ctx = self.ctx
        if any(b != boards[0] for b in boards):
            raise ValueError("river classes need one turn board")
        cls = ctx.turn_classes(boards[0], self.dead)
        K = len(cls)
        masks = None if self.dry else np.ones((len(boards) * K, ctx.n), dtype=np.float32)
        child_boards = []
        for bi, b in enumerate(boards):
            for ki, (c, _) in enumerate(cls):
                child_boards.append(b + (c,))
                if masks is not None:
                    masks[bi * K + ki] = 1.0 - ctx.member[c]
        child = self._decision(child_boards, pot, stack, (0.0, 0.0), 0, 0, inv, False)
        self.n_nodes += 1
        if self.dry:
            return None
        avail = sum(1 for c in range(52) if c not in self.dead and c not in boards[0])
        return Chance(boards, K, masks, child, avail, perm=ctx.class_perm(cls))

    def _river_dealt(self, boards, pot, stack, inv):
        """The river beyond `depth` inside a flop solve, every card dealt, each card's subtree by buckets of strength
        there (Config.river_dealt)."""
        ctx = self.ctx
        cards = [[c for c in range(52) if c not in self.dead and c not in b] for b in boards]
        child_boards = [b + (c,) for b, cs in zip(boards, cards) for c in cs]
        masks = None if self.dry else (1.0 - ctx.member[[c for cs in cards for c in cs]]).astype(np.float32)
        self._width = self.cfg.river_dealt
        try:
            child = BucketEntry(child_boards, self._decision(child_boards, pot, stack, (0.0, 0.0), 0, 0, inv, False))
        finally:
            self._width = None
        self.n_nodes += 1
        if self.dry:
            return None
        return Chance(boards, len(cards[0]), masks, child, len(cards[0]))

    def _bucketed_turn(self, child_boards, pot, stack, inv):
        """The turn's subtree under a flop solve's chance node: by river_buckets hand buckets with turn_buckets."""
        if not (self.cfg.turn_buckets and self.cfg.river_buckets):
            return self._decision(child_boards, pot, stack, (0.0, 0.0), 0, 0, inv, False)
        self._width = self.cfg.river_buckets
        try:
            return BucketEntry(child_boards, self._decision(child_boards, pot, stack, (0.0, 0.0), 0, 0, inv, False))
        finally:
            self._width = None

    def _sizes(self, kind: str, street: str) -> list:
        """Bet ("bet") or raise ("raise") sizes on `street`; the turn and river inside a flop solve take later_sizes."""
        cfg = self.cfg
        if self.root_street_len == 3 and street != "flop" and cfg.later_sizes and street in cfg.later_sizes.get(kind, {}):
            return cfg.later_sizes[kind][street]
        return (cfg.bet_sizes if kind == "bet" else cfg.raise_sizes)[street]

    def _decision(self, boards, pot, stack, bets, p, n_raises, inv, checked):
        cfg = self.cfg
        street = "river" if self._blind else street_of(len(boards[0]))
        o = 1 - p
        b_me, b_op = bets[p], bets[o]
        pot_total = pot + bets[0] + bets[1]
        remaining = stack - b_me
        names, kids = [], []
        # (root_allin: all-in offered at any depth on the root street)
        jam_ok = cfg.root_allin and not self._blind and not self._width and len(boards[0]) == self.root_street_len

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
            for frac in self._sizes("bet", street):
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
            if not offered_allin and remaining > 0 and (remaining <= cfg.allin_spr * pot_total or jam_ok):
                nb, ninv = with_bet(b_me + remaining)
                names.append("all-in")
                kids.append(self._decision(boards, pot, stack, nb, o, n_raises + 1, ninv, False))
        else:  # facing a bet
            names.append("fold")
            kids.append(Terminal(boards, self._kind + "fold" if self._width else "fold", inv, folder=p, rake=self._rake(inv),
                                 pay=self._pay(inv, p)))
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
                    for frac in self._sizes("raise", street):
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
                if capped or not offered or remaining <= cfg.allin_spr * pot_total or jam_ok:
                    nb, ninv = with_bet(stack)
                    names.append("all-in")
                    kids.append(self._decision(boards, pot, stack, nb, o, n_raises + 1, ninv, False))
        self.n_nodes += 1
        self.cells += len(boards) * len(names) * (self._width or self.ctx.n)
        if self.max_cells is not None and self.cells > self.max_cells:
            raise TreeTooBig(self.cells, self.max_cells)   # (built bottom up: what's allocated so far is under the cap)
        if self.dry:
            return None
        return Decision(boards, p, names, kids, pot_total, bets, self._width or self.ctx.n, self._blind, bool(self._width))


# ----------------------------------------------------------------- merged subtrees
def _shape(node) -> tuple:
    """A subtree's structure without pots, bet amounts or cards: subtrees with the same shape can
    be traversed as one batch."""
    if isinstance(node, Decision):
        return ("D", node.player, len(node.boards), tuple(_shape(c) for c in node.children))
    if isinstance(node, Chance):
        return ("C", node.K, node.sample, node.avail, len(node.boards), node.umasks is None, node.perm is None,
                _shape(node.child))
    if isinstance(node, RiverEntry):
        return ("R", node.outer, len(node.boards), _shape(node.child))
    if isinstance(node, BucketEntry):
        return ("E", len(node.boards), _shape(node.child))
    return ("T", node.kind, node.folder, len(node.boards))


def _merge(nodes, pairs: list):
    """One node standing for same-shaped subtrees `nodes`, their batch rows stacked in order.
    Appends (merged decision, originals) to `pairs` so the originals' arrays can become views."""
    first = nodes[0]
    boards = [b for n in nodes for b in n.boards]
    if isinstance(first, Decision):
        m = Decision.__new__(Decision)
        m.boards, m.player, m.names, m.pot_now, m.bets, m.B = boards, first.player, first.names, None, None, len(boards)
        m.blind, m.bucketed = first.blind, first.bucketed
        m.children = [_merge([n.children[a] for n in nodes], pairs) for a in range(len(first.children))]
        m.regret = np.concatenate([n.regret for n in nodes])
        m.ssum = np.concatenate([n.ssum for n in nodes])
        m.pred = None
        pairs.append((m, nodes))
        return m
    if isinstance(first, BucketEntry):
        m = BucketEntry(boards, _merge([n.child for n in nodes], pairs)) if not isinstance(first, RiverEntry) else \
            RiverEntry(boards, _merge([n.child for n in nodes], pairs), first.outer)
        m.tile = first.tile if all(n.boards == first.boards for n in nodes) else len(boards)
        return m
    if isinstance(first, Chance):
        if first.perm is not None and any(b != boards[0] for b in boards):
            raise ValueError("river classes merge only on one turn board")
        return Chance(boards, first.K, np.concatenate([n.masks for n in nodes]), _merge([n.child for n in nodes], pairs),
                      first.avail, first.sample, None if first.umasks is None else np.concatenate([n.umasks for n in nodes]),
                      first.perm)
    # chips in differ between the merged lines (different pots): one float32 per row, which is what a
    # Python float becomes when it multiplies a float32 array, so the results don't change
    inv = tuple(np.concatenate([np.full((len(n.boards), 1), n.inv[q], dtype=np.float32) for n in nodes]) for q in (0, 1))
    rake = None
    if any(n.rake is not None for n in nodes):
        # (a node merged before, as a river bucket subtree inside a merged turn, already has a (B, 1) column)
        rake = np.concatenate([np.full((len(n.boards), 1), 0.0 if n.rake is None else n.rake, dtype=np.float32)
                               for n in nodes])
    pay = None
    if first.pay is not None:
        pay = tuple(np.concatenate([np.full((len(n.boards), 1), n.pay[q], dtype=np.float32) for n in nodes])
                    for q in range(len(first.pay)))
    t = Terminal(boards, first.kind, inv, first.folder, rake, pay)
    if first.kind == "leaf" and all(n.boards == first.boards for n in nodes):
        t.tile = len(first.boards)
    return t


def _has_sampling(node) -> bool:
    if isinstance(node, Chance):
        return node.sample is not None or _has_sampling(node.child)
    return isinstance(node, Decision) and any(_has_sampling(c) for c in node.children)


# ----------------------------------------------------------------- solver
# Regret minimizers (Config.cfr; docs/cfr_research.md). After iteration t the cumulative regrets are scaled by
# t^alpha / (t^alpha + 1) where positive (alpha None: not discounted) and t^beta / (t^beta + 1) where negative (beta
# None: set to 0, CFR+'s floor), and the strategy sums by (t / (t + 1))^gamma, which weights iteration s in the
# average by about s^gamma. A pair (start, end) is a hyperparameter schedule, linear in t over the solve's horizon
# (Solver.horizon). predictive: regret matching on the regrets plus the last iteration's instantaneous regrets
# (the prediction of the next ones; Farina et al. 2021). reset4: the strategy sums start over at every power of 4.
# Updates alternate between the players in every scheme.
def _gamma_ramp(t: int) -> float:
    """dcfr-ramp's averaging exponent after iteration t: 3 at the start, 8 from the 50th iteration on. A higher
    gamma averages fewer, later iterations (s^gamma: the last ~1/(gamma+1) of them). gamma 8 throughout was 10-25%
    less exploitable than 4 from 50 iterations on, but up to 40% more before 20 (the first iterates are poor);
    the ramp is as good as 4 early and as 8 late (docs/cfr_research.md)."""
    return min(8.0, 3.0 + t / 10)


def _beta_step(t: int) -> float:
    """dcfr-ramp's negative-regret exponent after iteration t: 0 (negatives halved every iteration, as in
    postflop-solver) for the first 100 iterations, then 0.5 (x sqrt(t) / (sqrt(t) + 1), the DCFR paper's other
    setting). Beta 0 converges fastest early, but a line one player has all but stopped taking gets almost no
    regret, the halving wipes what it has, and its hands drift back toward their first strategy: the other player's
    bets into it turn profitable again and the iterates swing. Two of 15 Accurate turns (~180 iterations) jumped
    from 0.15-0.33% of the pot to 2-2.8% (the average, weighted to the last ~30 iterations, takes the swing in) and
    took 100+ iterations to recover; with 0.5 from the 100th iteration neither did, and Fast turns (~100
    iterations) are unchanged (docs/cfr_research.md)."""
    return 0.0 if t < 100 else 0.5


CFR_SCHEMES = {
    # the default: DCFR (Brown & Sandholm 2019, alpha 1.5) with beta 0 rising to 0.5 at the 100th iteration,
    # averaging with gamma 3 -> 8
    "dcfr-ramp": dict(alpha=1.5, beta=_beta_step, gamma=_gamma_ramp),
    "dcfr-ramp-b0": dict(alpha=1.5, beta=0.0, gamma=_gamma_ramp),      # beta 0 throughout, the default before
    # DCFR with gamma 4 (the paper: 2), the default before: the same exploitability as gamma 2 in 25-35% fewer
    # iterations on flop, turn and 3bet spots (alpha 1-1.25 spiky, 2+ slower)
    "dcfr": dict(alpha=1.5, beta=0.0, gamma=4.0),
    "dcfr-reset4": dict(alpha=1.5, beta=0.0, gamma=3.0, reset4=True),   # postflop-solver's (b-inary)
    "dcfr+": dict(alpha=1.5, beta=None, gamma=4.0),           # DCFR with CFR+'s floor (Xu et al. 2024)
    "lcfr": dict(alpha=1.0, beta=1.0, gamma=1.0),             # Linear CFR
    "cfr+": dict(alpha=None, beta=None, gamma=2.0),           # Tammelin 2014, quadratic averaging
    "pcfr+": dict(alpha=None, beta=None, gamma=2.0, predictive=True),         # Farina, Kroer & Sandholm 2021
    "pdcfr+": dict(alpha=2.3, beta=None, gamma=5.0, predictive=True),         # Xu et al. 2024 (IJCAI)
    "hs-pcfr+": dict(alpha=None, beta=None, gamma=(30.0, 25.0), predictive=True),   # Zhang, McAleer & Sandholm
    "hs-dcfr": dict(alpha=(1.0, 4.0), beta=(-1.0, -3.0), gamma=(30.0, 25.0)),         # (AAAI 2026), HS(30)
    "hs-dcfr15": dict(alpha=(1.0, 4.0), beta=(-1.0, -3.0), gamma=(15.0, 10.0)),       # HS(15)
}


class Solver:
    """Speed (natively the solve is bound by Python and numpy per-call overhead, besides the leaf matvecs):
    * the subtrees below the root street's chance nodes mostly share one shape (pots differ, actions
      don't), so same-shaped ones are merged into one batch at setup and traversed once: a flop solve
      walks ~30 decision nodes per pass instead of ~90. The root street is walked down to collect the
      reaches arriving at each chance node, the merged subtrees run, and the root street is walked back up.
    * every node's regrets and strategy sums are views into two flat buffers, discounted in one call
    * a player's strategies computed in the other player's pass are reused in their own (unchanged regrets),
      below the root street too; there the NO_BLAS kernels also trim the leaf matrices to each range
    * with BLAS, a pass first walks down collecting the opponent's reach at every equity leaf, and each leaf
      matrix stack multiplies all of its reaches at once (one matrix-matrix product, see _leaf_batch)
    All of it gives the same numbers as walking the tree node by node (the batched leaves up to float32 rounding). The original nodes' arrays are views of the merged ones, so reading and EV code sees current values."""

    def __init__(self, ctx: Ctx, root: Decision, dead: set):
        self.ctx, self.root, self.dead = ctx, root, dead
        self.iters = 0
        # the regret minimizer (CFR_SCHEMES, docs/cfr_research.md)
        sch = CFR_SCHEMES[ctx.cfg.cfr]
        self.alpha, self.beta, self.gamma = sch["alpha"], sch["beta"], sch["gamma"]
        self.predictive = sch.get("predictive", False)
        self.reset4 = sch.get("reset4", False)
        self.horizon = None           # a schedule's length in iterations (None: the first run's max_iters)
        self.rng = random.Random(ctx.cfg.seed + 2)
        self.sampling = True          # chance sampling while training; off for EVs and reports
        self._nodes: List[Decision] = []
        self._collect(root)
        self._groups: List[Tuple[Chance, List[Chance]]] = []
        self._bgroups: List[Tuple[BucketEntry, List[BucketEntry]]] = []
        self._merge_subtrees()
        # training on a numpy without BLAS (see NO_BLAS) skips the leaf-matrix rows and columns of combos
        # outside each player's range (always 0 reach): ~1/3 less arithmetic on a typical flop
        self.trim_leaves = NO_BLAS
        # with BLAS, each pass's equity leaves are computed together up front (see _leaf_batch)
        self.batch_leaves = not NO_BLAS
        self._pre: Dict[int, np.ndarray] = {}
        self._live = None
        self._trimmed: Dict[tuple, np.ndarray] = {}
        self._btabs: Dict[int, tuple] = {}
        # a training pass's reaches into each chance node's (and bucket and river entry's) children, by (node, player):
        # the opponent's computed once a pass (_down's, for _cfr's walk of the same pass), the traverser's once a run:
        # a pass never multiplies the traverser's reach by its own strategies, so it is the root reach's every pass
        self._cin: Dict[tuple, object] = {}
        self._own: Dict[tuple, object] = {}
        self._p = None              # the training pass's player
        self.use_core = True        # run() in the compiled core when it covers the tree (gto/core.py)
        self._core_ok = True        # False: the tree has something the core doesn't run

    def _child_reach(self, node, q, r, fn):
        """fn(r), player q's reach into node's children, through the memo (training passes, every row only)."""
        if self._p is None:         # (a pass outside run(), e.g. a test's or a report's: no memo)
            return fn(r)
        key = (id(node), q)
        memo = self._own if q == self._p else self._cin
        v = memo.get(key)
        if v is None:
            v = memo[key] = fn(r)
        return v

    def _collect(self, node):
        if isinstance(node, Decision):
            self._nodes.append(node)
            for c in node.children:
                self._collect(c)
        elif isinstance(node, (Chance, BucketEntry)):
            self._collect(node.child)

    def _merge_subtrees(self):
        # the root street's chance nodes, minus sampled ones (they draw their cards in traversal order)
        points, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if isinstance(n, Chance):
                if not _has_sampling(n):
                    points.append(n)
            elif isinstance(n, Decision):
                stack.extend(n.children)
        by_shape: Dict[tuple, List[Chance]] = {}
        for c in points:
            by_shape.setdefault(_shape(c), []).append(c)
        pairs: list = []
        for members in by_shape.values():
            self._groups.append((_merge(members, pairs), members))
        # then the river bucket subtrees (BucketEntry) inside those, merged again across every line that reaches
        # them: a bucket subtree's arrays are small, so walking each line's one by one was mostly call overhead
        entries, stack = [], [c for c, _ in self._groups]
        while stack:
            n = stack.pop()
            if isinstance(n, BucketEntry):
                entries.append(n)
            elif isinstance(n, Decision):
                stack.extend(n.children)
            elif isinstance(n, Chance):
                stack.append(n.child)
        by_shape = {}
        for e in entries:
            by_shape.setdefault(_shape(e), []).append(e)
        pairs2: list = []
        for members in by_shape.values():
            self._bgroups.append((_merge(members, pairs2), members))
        replaced = {id(n) for _, nodes in pairs + pairs2 for n in nodes}
        owners = [n for n in self._nodes if id(n) not in replaced] + [m for m, _ in pairs + pairs2 if id(m) not in replaced]
        for f in ("regret", "ssum") + (("pred",) if self.predictive else ()):
            flat = np.zeros(sum(n.regret.size for n in owners), dtype=np.float32)
            off = 0
            for n in owners:
                a = getattr(n, f)
                v = flat[off:off + n.regret.size].reshape(n.regret.shape)
                if a is not None:
                    v[...] = a
                setattr(n, f, v)
                off += v.size
            setattr(self, "_" + f, flat)
        for m, nodes in pairs2 + pairs:     # (the second merge's first: the first merge's nodes are its views)
            off = 0
            for n in nodes:
                n.regret, n.ssum = m.regret[off:off + n.B], m.ssum[off:off + n.B]
                if self.predictive:
                    n.pred = m.pred[off:off + n.B]
                off += n.B

    def compat_sum(self, ro):
        """ro: (B, N) -> (B, N) mass of opponent combos sharing no card with each combo."""
        ctx = self.ctx
        if NO_BLAS and ctx.n:
            # each card's reach as a sum over its combos (2N adds), not a (B, N) x (N, 52) product: ~4x faster
            # in Pyodide, where the product has no BLAS
            cs = np.add.reduceat(ro[:, ctx.card_perm], ctx.card_starts, axis=1)[:, ctx.cab_held]
        else:
            member = ctx.member32 if ro.dtype == np.float32 else ctx.member
            cs = (ro @ member.T)[:, ctx.cab]   # (B, 2N)
        n = ro.shape[1]
        return ro.sum(axis=1, keepdims=True) - cs[:, :n] - cs[:, n:] + ro

    def _terminal_M(self, node: Terminal):
        if node.M is None:
            node.M = self.ctx.stacked(node.kind, node.boards if node.tile is None else node.boards[:node.tile], self.dead)
        return node.M

    def _trim(self, M, p):
        """M's rows for p's live combos and columns for the opponent's (built once per matrix). The reach
        side is gathered with np.take: a fancy-indexed array comes out strided, which einsum runs ~2x slower."""
        key = (id(M), p)
        T = self._trimmed.get(key)
        if T is None:
            T = self._trimmed[key] = np.ascontiguousarray(M[:, self._live[p]][:, :, self._live[1 - p]])
        return T

    def _terminal(self, node: Terminal, p, ro, sel=None, train=False):
        """(B, N) utility for p at a terminal, ro: the opponent's reach. `train`: a training pass, where
        combos outside p's range may get 0 (their regrets never matter: their reach is 0)."""
        if node.kind == "fold":
            return node.fold_pay(p) * self.compat_sum(ro)
        if node.kind[0] in "br":
            return self._bucket_terminal(node, p, ro, sel)
        u, f32 = self._showdown(node, p, ro, sel, train)
        if node.pay is not None:
            A, B = node.pay[2 * p], node.pay[2 * p + 1]
            if sel is not None and not np.isscalar(A):
                A, B = A[sel], B[sel]
            return B * u + A * self.compat_sum(ro)
        inv = np.float32(node.inv[p]) if f32 else node.inv[p]
        if node.rake is None:
            return inv * u
        # with rake R, p wins the opponent's chips minus R, loses its own, and a tie costs each R/2. With
        # u = sum over opponent combos of reach * (P(win) - P(lose)) and C their compatible reach, that
        # is (inv - R/2) * u - R/2 * C, since P(win) + P(tie)/2 = (1 + P(win) - P(lose)) / 2
        R = node.rake
        if sel is not None and not np.isscalar(R):
            R = R[sel]
        half = R / 2
        return (inv - half) * u - half * self.compat_sum(ro)

    def _btab(self, node):
        """Bucket tables for a BucketEntry or a bucket terminal's rows: (period T of its boards, ids (T, 2, N),
        E (2, T, K, K), kappa (2, T, K, K)), see Ctx.bucket_table."""
        t = self._btabs.get(id(node))
        if t is None:
            boards, B = node.boards, len(node.boards)
            T = next(k for k in range(1, B + 1) if B % k == 0 and all(boards[i] == boards[i % k] for i in range(B)))
            key = ("tab", tuple(boards[:T]))
            t = self._btabs.get(key)
            if t is None:
                cfg = self.ctx.cfg
                tabs = [self.ctx.river_card_table(b, self.dead, cfg.river_dealt) if len(b) == 5 else
                        self.ctx.bucket_table(b, self.dead, cfg.river_buckets) for b in boards[:T]]
                t = self._btabs[key] = (T, np.stack([x[0] for x in tabs]), np.stack([x[1] for x in tabs], axis=1),
                                        np.stack([x[2] for x in tabs], axis=1))
            self._btabs[id(node)] = t
        return t

    def _rtab(self, node):
        """River-strength tables for a RiverEntry or an "r" terminal's rows: (period T of its boards, X (2, T, W, K),
        E (2, T, K, K), kappa (2, T, K, K)), see Ctx.river_table (W: N per combo, or the turn's buckets)."""
        t = self._btabs.get(id(node))
        if t is None:
            boards, B = node.boards, len(node.boards)
            T = next(k for k in range(1, B + 1) if B % k == 0 and all(boards[i] == boards[i % k] for i in range(B)))
            cfg = self.ctx.cfg
            outer = cfg.river_buckets if cfg.turn_buckets and cfg.river_buckets else None
            key = ("rtab", tuple(boards[:T]))
            t = self._btabs.get(key)
            if t is None:
                tabs = [self.ctx.river_table(b, self.dead, cfg.river_strength, outer) for b in boards[:T]]
                t = self._btabs[key] = (T, np.stack([x[0] for x in tabs], axis=1), np.stack([x[1] for x in tabs], axis=1),
                                        np.stack([x[2] for x in tabs], axis=1))
            self._btabs[id(node)] = t
        return t

    @staticmethod
    def _per_board(X, x, T, sel, transpose=False):
        """x: (B, W) rows of boards with period T, X: (T, W, K) -> (B, K), each row times its board's matrix (X's
        transpose, (T, K, W) -> (B, W) with `transpose`)."""
        Xm = X.transpose(0, 2, 1) if transpose else X
        if sel is None:
            B, W = x.shape
            return np.matmul(x.reshape(B // T, T, 1, W), Xm)[:, :, 0, :].reshape(B, -1)
        rows = np.asarray(sel) % T
        out = np.empty((x.shape[0], Xm.shape[2]), dtype=np.float32)
        for t in np.unique(rows):
            m = rows == t
            out[m] = x[m] @ Xm[t]
        return out

    def _river_in(self, node: RiverEntry, rs, sel):
        """Reaches (B, W) each -> river bucket masses (B, K) each."""
        T, X, _, _ = self._rtab(node)
        return [self._per_board(X[q], rs[q].astype(np.float32, copy=False), T, sel) for q in (0, 1)]

    def _river_in1(self, node: RiverEntry, q, r):
        """Player q's reach (B, W) -> its river bucket masses (B, K), every row."""
        T, X, _, _ = self._rtab(node)
        return self._per_board(X[q], r.astype(np.float32, copy=False), T, None)

    def _river_out(self, node: RiverEntry, u, p, sel):
        """p's river bucket utilities (B, K) -> (B, W)."""
        T, X, _, _ = self._rtab(node)
        return self._per_board(X[p], u.astype(np.float32, copy=False), T, sel, transpose=True)

    def _bucket_matvec(self, node, M, ro, sel):
        """M: (T, K, K) per board of the node's rows' period, ro: (B, K) -> (B, K)."""
        # (matmul broadcasting the per-board matrices over the repeats: ~4x faster than einsum natively, as fast in Pyodide)
        T = M.shape[0]
        if sel is None:
            B, K = ro.shape
            return np.matmul(M, ro.reshape(B // T, T, K, 1))[..., 0].reshape(B, -1)
        return np.matmul(M[np.asarray(sel) % T], ro[..., None])[..., 0]

    def _bucket_terminal(self, node: Terminal, p, ro, sel=None):
        """p's bucket utilities at a terminal of a bucket subtree; ro: the opponent's bucket masses."""
        _, _, E, kap = self._btab(node) if node.kind[0] == "b" else self._rtab(node)
        if node.kind[1:] == "fold":
            return node.fold_pay(p) * self._bucket_matvec(node, kap[p], ro, sel)
        u = self._bucket_matvec(node, E[p], ro, sel)
        if node.pay is not None:
            A, B = node.pay[2 * p], node.pay[2 * p + 1]
            if sel is not None and not np.isscalar(A):
                A, B = A[sel], B[sel]
            return B * u + A * self._bucket_matvec(node, kap[p], ro, sel)
        inv = node.inv[p]
        if node.rake is None:
            return inv * u
        R = node.rake
        if sel is not None and not np.isscalar(R):
            R = R[sel]
        half = R / 2
        return (inv - half) * u - half * self._bucket_matvec(node, kap[p], ro, sel)

    def _bucket_in(self, node: BucketEntry, rs, sel):
        """Per-combo reaches (B, N) each -> bucket masses (B, K) each, and the flat bucket index of every combo
        (for _bucket_out)."""
        K = self._bk(node)
        B, n = rs[0].shape
        key = ("flat", id(node))
        idx = self._btabs.get(key) if sel is None else None
        if idx is None:
            # per player: each combo's flat bucket index (for _bucket_out), the rows' combos sorted by bucket (flat
            # gather indices), where each bucket starts in that order, and which buckets are empty: a bucket's sum
            # is then one np.add.reduceat (np.bincount, with its float64 weights, took twice as long)
            T, ids, _, _ = self._btab(node)
            rows = np.arange(B) if sel is None else np.asarray(sel)
            r = np.arange(B)[:, None]
            idx = []
            for q in (0, 1):
                bid = ids[rows % T, q]                                          # (B, n); K: outside the range
                order = np.argsort(bid, axis=1, kind="stable")
                flat = (r * (K + 1) + bid).ravel()
                e = np.zeros((B, K + 2), dtype=np.intp)
                np.cumsum(np.bincount(flat, minlength=B * (K + 1)).reshape(B, K + 1), axis=1, out=e[:, 1:])
                e = e[:, :K + 1]
                idx.append((flat, (r * n + order).ravel(), (r * n + e).ravel(), e[:, 1:] == e[:, :-1]))
            if sel is None:
                self._btabs[key] = idx
        ms = [self._bucket_mass(idx[q], rs[q], B, K) for q in (0, 1)]
        return ms, [x[0] for x in idx]

    def _bucket_in1(self, node: BucketEntry, q, r):
        """_bucket_in for player q's reach alone, every row: (masses, flat bucket index)."""
        idx = self._btabs.get(("flat", id(node)))
        if idx is None:
            self._bucket_in(node, (r, r), None)
            idx = self._btabs[("flat", id(node))]
        return self._bucket_mass(idx[q], r, r.shape[0], self._bk(node)), idx[q][0]

    def _bk(self, node) -> int:
        """A bucket entry's width: river_dealt's on river boards, river_buckets' on turn boards."""
        cfg = self.ctx.cfg
        return cfg.river_dealt if len(node.boards[0]) == 5 else cfg.river_buckets

    @staticmethod
    def _bucket_mass(ix, r, B, K):
        _, fsort, starts, empty = ix
        x = np.empty(r.size + 1, dtype=np.float32)
        x[-1] = 0      # (an empty last bucket starts there)
        x[:-1] = r.ravel()[fsort]
        m = np.add.reduceat(x, starts).reshape(B, K + 1)[:, :K]
        m[empty] = 0
        return m

    @staticmethod
    def _bucket_out(u, flat):
        """(B, K) bucket utilities -> (B, N): every combo gets its bucket's (0 outside the range)."""
        B, K = u.shape
        up = np.zeros((B, K + 1), dtype=u.dtype)
        up[:, :K] = u
        return up.ravel()[flat].reshape(B, -1)

    def _showdown(self, node: Terminal, p, ro, sel=None, train=False):
        """(sum over opponent combos of reach * (P(win) - P(lose)), whether p's chips in multiply it as
        float32) at a showdown or equity leaf."""
        if len(node.boards[0]) == 5:
            if node.M is None:
                node.M = self.ctx.ranking(node.boards)
            return node.M.utility(ro, sel), False
        pre = self._pre.pop(id(node), None) if self._pre else None
        if pre is not None:
            return pre, node.tile is None
        if node.tile is not None:
            # rows repeat the same boards: matmul broadcasts the one (tile, N, N) stack over the repeats
            # (the same matvecs as unmerged, in one call; merged terminals only see full batches, never `sel`)
            M = self._terminal_M(node)
            n = ro.shape[1]
            ro = ro.astype(np.float32, copy=False).reshape(-1, node.tile, n)
            if train and self._live is not None:
                u = np.zeros(ro.shape, dtype=np.float32)
                u[:, :, self._live[p]] = _matvec(self._trim(M, p), np.take(ro, self._live[1 - p], axis=2))
            else:
                u = _matvec(M, ro)
            return u.reshape(-1, n), False
        M = self._terminal_M(node)
        ro = ro.astype(np.float32)
        if train and self._live is not None:
            T = self._trim(M, p)
            u = np.zeros(ro.shape, dtype=np.float32)
            u[:, self._live[p]] = _matvec(T if sel is None else T[sel], np.take(ro, self._live[1 - p], axis=1))
            return u, True
        if sel is not None:
            M = M[sel]
        return _matvec(M, ro), True

    def _cfr(self, node, p, r0, r1, update, use_avg, sel=None, strats=None, done=None):
        """r0, r1: (B, N) reaches. Returns (B, N) utility for player p.
        `sel`: which of the node's batch rows r0/r1 are for (None = all of them).
        `strats`, `done`: training passes only (see _traverse): each player's current strategies by node,
        filled as they are computed and reused while that player's regrets are unchanged, and the
        merged chance nodes' utilities for the root street's walk back up."""
        # no reach pruning: a subtree the opponent almost never enters must still be trained,
        # because the real hand can go there (a donk bet, an overbet) and we read its strategy
        if isinstance(node, Terminal):
            return self._terminal(node, p, r1 if p == 0 else r0, sel, strats is not None)
        if isinstance(node, RiverEntry):
            if done is not None and id(node) in done:
                return done[id(node)]
            if strats is not None and sel is None:
                m0, m1 = (self._child_reach(node, q, r, lambda x, q=q: self._river_in1(node, q, x)) for q, r in ((0, r0), (1, r1)))
            else:
                m0, m1 = self._river_in(node, (r0, r1), sel)
            return self._river_out(node, self._cfr(node.child, p, m0, m1, update, use_avg, sel, strats), p, sel)
        if isinstance(node, BucketEntry):
            if done is not None and id(node) in done:
                return done[id(node)]
            if strats is not None and sel is None:
                (m0, f0), (m1, f1) = (self._child_reach(node, q, r, lambda x, q=q: self._bucket_in1(node, q, x))
                                      for q, r in ((0, r0), (1, r1)))
                flat = (f0, f1)
            else:
                (m0, m1), flat = self._bucket_in(node, (r0, r1), sel)
            return self._bucket_out(self._cfr(node.child, p, m0, m1, update, use_avg, sel, strats), flat[p])
        if isinstance(node, Chance):
            if done is not None and id(node) in done:
                return done[id(node)]
            B, K, n = r0.shape[0], node.K, self.ctx.n
            rows = np.arange(B) if sel is None else np.asarray(sel)
            if node.perm is not None:
                csel = None if sel is None else (rows[:, None] * K + np.arange(K)).ravel()
                if strats is not None and sel is None:
                    c0, c1 = (self._child_reach(node, q, r, lambda x: node.spread(x, rows)) for q, r in ((0, r0), (1, r1)))
                else:
                    c0, c1 = node.spread(r0, rows), node.spread(r1, rows)
                u = self._cfr(node.child, p, c0, c1, update, use_avg, csel, strats, done if sel is None else None)
                return node.gather(u, rows)
            if node.sample is not None and self.sampling:
                ks = np.array(sorted(self.rng.sample(range(K), node.sample)))
            else:
                ks = np.arange(K)
            k = len(ks)
            csel = (rows[:, None] * K + ks[None, :]).ravel()
            full = sel is None and k == K
            m3 = node.masks.reshape(B, k, n) if full else node.masks[csel].reshape(B, k, n)
            if full and strats is not None:
                r0c, r1c = (self._child_reach(node, q, r, lambda x: (x[:, None, :] * m3).reshape(B * k, n))
                            for q, r in ((0, r0), (1, r1)))
            else:
                r0c = (r0[:, None, :] * m3).reshape(B * k, n)
                r1c = (r1[:, None, :] * m3).reshape(B * k, n)
            u = self._cfr(node.child, p, r0c, r1c, update, use_avg, None if full else csel, strats, done if full else None)
            norm = node.norm_for(m3) if k < K else node.norm[rows]
            w = m3 if node.umasks is None else node.weights(None if sel is None else rows)
            return (u.reshape(B, k, n) * w).sum(axis=1) / norm
        if strats is None or sel is not None:
            strat = node.avg_strategy(sel) if use_avg else node.strategy(sel)     # (B, A, N)
        else:
            cache = strats[node.player]
            strat = cache.get(id(node))
            if strat is None:
                strat = cache[id(node)] = node.strategy()
        if node.player == p:
            ro = r1 if p == 0 else r0
            utils = np.empty_like(strat)
            cs = None      # the opponent's reach is the same under each of p's actions: one compat_sum for all folds
            for a, child in enumerate(node.children):
                if isinstance(child, Terminal):
                    if child.kind == "fold":
                        if cs is None:
                            cs = self.compat_sum(ro)
                        utils[:, a, :] = child.fold_pay(p) * cs
                    else:
                        utils[:, a, :] = self._terminal(child, p, ro, sel, strats is not None)
                else:
                    utils[:, a, :] = self._cfr(child, p, r0, r1, update, use_avg, sel, strats, done)
            node_util = (strat * utils).sum(axis=1)                     # (B, N)
            if update:
                rp = r0 if p == 0 else r1
                utils -= node_util[:, None, :]          # the instantaneous regrets
                if sel is None:
                    node.regret += utils
                    node.ssum += rp[:, None, :] * strat
                    if node.pred is not None:
                        node.pred[...] = utils
                else:
                    node.regret[sel] += utils
                    node.ssum[sel] += rp[:, None, :] * strat
                    if node.pred is not None:
                        node.pred[sel] = utils
            return node_util
        # the opponent acts: every child's reach in one product
        reach = (r1 if p == 0 else r0)[:, None, :] * strat
        util = None
        for a, child in enumerate(node.children):
            if isinstance(child, Terminal):
                u = self._terminal(child, p, reach[:, a, :], sel, strats is not None)
            elif p == 0:
                u = self._cfr(child, p, r0, reach[:, a, :], update, use_avg, sel, strats, done)
            else:
                u = self._cfr(child, p, reach[:, a, :], r1, update, use_avg, sel, strats, done)
            util = u if util is None else util + u
        return util

    def _down(self, node, p, r0, r1, strats, arrive, want=None, into=None):
        """A training pass's walk down: strategies, and the reaches arriving at each chance node in `arrive`
        (the root street's: the walk stops there), or with `into`, through chance nodes too (into the merged
        subtrees; sampled ones are left to _cfr). With `want`, the opponent's reach at every equity leaf
        is appended to it, for _leaf_batch. The reaches are the ones _cfr computes."""
        if isinstance(node, BucketEntry):
            if into:
                arrive[id(node)] = (r0, r1)     # its subtree is run merged with the others (see _traverse)
            return
        if isinstance(node, Terminal):
            if want is not None and node.kind != "fold" and len(node.boards[0]) < 5:
                want.append((node, r1 if p == 0 else r0))
        elif isinstance(node, Chance):
            if not into:
                arrive[id(node)] = (r0, r1)
            elif node.perm is not None:
                rows = np.arange(r0.shape[0])
                c0, c1 = (self._child_reach(node, q, r, lambda x: node.spread(x, rows)) for q, r in ((0, r0), (1, r1)))
                self._down(node.child, p, c0, c1, strats, arrive, want, into)
            elif node.sample is None:
                B, K, n = r0.shape[0], node.K, self.ctx.n
                m3 = node.masks.reshape(B, K, n)
                c0, c1 = (self._child_reach(node, q, r, lambda x: (x[:, None, :] * m3).reshape(B * K, n))
                          for q, r in ((0, r0), (1, r1)))
                self._down(node.child, p, c0, c1, strats, arrive, want, into)
        else:
            cache = strats[node.player]
            strat = cache.get(id(node))
            if strat is None:
                strat = cache[id(node)] = node.strategy()
            for a, child in enumerate(node.children):
                if isinstance(child, Terminal) and want is None:
                    continue
                if node.player == p:
                    self._down(child, p, r0, r1, strats, arrive, want, into)
                elif p == 0:
                    self._down(child, p, r0, r1 * strat[:, a, :], strats, arrive, want, into)
                else:
                    self._down(child, p, r0 * strat[:, a, :], r1, strats, arrive, want, into)

    def _leaf_batch(self, want):
        """Every equity leaf's matrix products for a pass, as one matrix-matrix product per matrix stack
        (natively the leaves are memory-bound: one product per leaf read each stack again, ~3/4 of a flop
        solve). Results wait in self._pre for _showdown."""
        n = self.ctx.n
        by_stack: Dict[int, tuple] = {}
        for node, ro in want:
            M = self._terminal_M(node)
            x = ro.astype(np.float32, copy=False).reshape(-1, M.shape[0], n)    # (repeats, matrices, N)
            by_stack.setdefault(id(M), (M, []))[1].append((node, x))
        for M, items in by_stack.values():
            X = items[0][1] if len(items) == 1 else np.concatenate([x for _, x in items])
            U = np.matmul(X.transpose(1, 0, 2), M.transpose(0, 2, 1)).transpose(1, 0, 2)
            off = 0
            for node, x in items:
                self._pre[id(node)] = U[off:off + len(x)].reshape(-1, n)
                off += len(x)

    def _traverse(self, p, r0, r1, strats):
        """One training pass for p. Same result as _cfr on the root; the merged subtrees run in one batch.
        strats: (player 0's, player 1's) current strategies by node; p's are stale once this pass updates them."""
        if not self._groups:
            return self._cfr(self.root, p, r0, r1, update=True, use_avg=False, strats=strats)
        arrive, done = {}, {}
        want = [] if self.batch_leaves else None
        self._down(self.root, p, r0, r1, strats, arrive, want)
        reach = [(np.concatenate([arrive[id(c)][0] for c in members]), np.concatenate([arrive[id(c)][1] for c in members]))
                 for _, members in self._groups]
        if want is not None or self._bgroups:
            for (chance, _), (g0, g1) in zip(self._groups, reach):
                self._down(chance, p, g0, g1, strats, arrive, want, into=True)
            if want is not None:
                self._leaf_batch(want)
        for entry, members in self._bgroups:
            u = self._cfr(entry, p, np.concatenate([arrive[id(e)][0] for e in members]),
                          np.concatenate([arrive[id(e)][1] for e in members]), update=True, use_avg=False, strats=strats)
            off = 0
            for e in members:
                done[id(e)] = u[off:off + len(e.boards)]
                off += len(e.boards)
        for (chance, members), (g0, g1) in zip(self._groups, reach):
            u = self._cfr(chance, p, g0, g1, update=True, use_avg=False, strats=strats, done=done)
            off = 0
            for c in members:
                done[id(c)] = u[off:off + len(c.boards)]
                off += len(c.boards)
        u = self._cfr(self.root, p, r0, r1, update=True, use_avg=False, strats=strats, done=done)
        self._pre.clear()
        return u

    def _param(self, v):
        """A scheme's parameter now: a pair is a schedule, linear in the iteration over the horizon; a function
        of the iteration count its value."""
        if callable(v):
            return v(self.iters)
        if not isinstance(v, tuple):
            return v
        h = max(self.horizon or 1, 1)
        return v[0] + (v[1] - v[0]) * min(self.iters / h, 1.0)

    def _factors(self) -> tuple:
        """The discount after iteration self.iters: (floor, the positive regrets' factor, the negative ones', the
        strategy sums'); floor: the regrets are floored at 0 first (beta None, CFR+'s)."""
        t = float(self.iters)
        alpha, beta, gamma = self._param(self.alpha), self._param(self.beta), self._param(self.gamma)
        pos_f = 1.0 if alpha is None else (t ** alpha) / (t ** alpha + 1)
        neg_f = 0.0 if beta is None else (t ** beta) / (t ** beta + 1)
        if self.reset4:
            # postflop-solver's averaging: the strategy sums start over at every power of 4
            t -= 4 ** int(np.log(t) / np.log(4) + 1e-9)
        return beta is None, pos_f, neg_f, (t / (t + 1)) ** gamma

    def _discount(self):
        floor, pos_f, neg_f, ss_f = self._factors()
        r = self._regret
        if floor:
            np.maximum(r, 0.0, out=r)
            if pos_f != 1.0:
                r *= np.float32(pos_f)
        else:
            r *= np.where(r > 0, np.float32(pos_f), np.float32(neg_f))     # (a float64 factor array: twice the traffic)
        self._ssum *= np.float32(ss_f)

    def run(self, r0, r1, time_budget=3.0, max_iters=1000, min_iters=10, should_stop=None, reserve=None,
            max_per=None):
        """r0, r1: (N,) root reaches. Adds iterations until the budget is spent.
        Raises Cancelled as soon as should_stop() is true, and TooSlow when the first iteration takes over max_per s.
        `reserve` (iterations): when given, no iteration starts that would end past the budget (by the mean
        iteration so far) with this many iterations' time kept back for what follows (an exploitability costs
        about one); else iterations run until one ends past the budget (up to one iteration over)."""
        r0b, r1b = r0[None, :].astype(np.float32), r1[None, :].astype(np.float32)
        if self.trim_leaves:
            live = (np.flatnonzero(r0 > 0), np.flatnonzero(r1 > 0))
            if self._live is None or any(not np.array_equal(a, b) for a, b in zip(live, self._live)):
                self._live, self._trimmed = live, {}
        if self.horizon is None and max_iters > self.iters:
            self.horizon = max_iters
        t0 = time.time()
        start = self.iters
        # the compiled core (gto/core.py) runs the iterations of the trees it covers, numpy the rest
        if self.use_core and core.run(self, r0b[0], r1b[0], t0, time_budget, max_iters, min_iters, should_stop, reserve,
                                      max_per):
            return self.iters, time.time() - t0
        self._own.clear()       # (the traverser's reaches are this run's root reaches')
        while self.iters < max_iters:
            if should_stop is not None and should_stop():
                raise Cancelled()
            done = self.iters - start
            if reserve is not None and done >= min_iters and \
                    time.time() - t0 + (time.time() - t0) / max(done, 1) * (1.0 + reserve) > time_budget:
                break
            # the opponent's strategies computed in player 0's pass are still current in player 1's
            strats = ({}, {})
            try:
                for p in (0, 1):
                    self._p = p
                    self._traverse(p, r0b, r1b, strats)
                    self._cin.clear()
                    strats[p].clear()
            finally:
                self._p = None
                self._cin.clear()
            self.iters += 1
            self._discount()
            if max_per is not None and self.iters - start == 1 and time.time() - t0 > max_per:
                raise TooSlow(time.time() - t0)
            if self.iters - start >= min_iters and time.time() - t0 > time_budget:
                break
        return self.iters, time.time() - t0

    def _best_response(self, node, p, r0, r1, lock=None):
        """Counterfactual utility for p playing a best response to the opponent's average
        strategy, every chance card enumerated. `lock(node, strat, reach)` turns the opponent's
        strategy into the one they actually play (exploit mode's player pool)."""
        if isinstance(node, Terminal):
            return self._cfr(node, p, r0, r1, update=False, use_avg=True)
        if isinstance(node, RiverEntry):
            m0, m1 = self._river_in(node, (r0, r1), None)
            return self._river_out(node, self._best_response(node.child, p, m0, m1, lock), p, None)
        if isinstance(node, BucketEntry):
            # (p best-responds by bucket, the bucket game's best response)
            (m0, m1), flat = self._bucket_in(node, (r0, r1), None)
            return self._bucket_out(self._best_response(node.child, p, m0, m1, lock), flat[p])
        if isinstance(node, Chance):
            B, K, n = r0.shape[0], node.K, self.ctx.n
            if node.perm is not None:
                rows = np.arange(B)
                return node.gather(self._best_response(node.child, p, node.spread(r0, rows), node.spread(r1, rows), lock), rows)
            m3 = node.masks.reshape(B, K, n)
            u = self._best_response(node.child, p, (r0[:, None, :] * m3).reshape(B * K, n),
                                    (r1[:, None, :] * m3).reshape(B * K, n), lock)
            return (u.reshape(B, K, n) * node.weights()).sum(axis=1) / node.norm
        if node.player == p:
            return np.max([self._best_response(c, p, r0, r1, lock) for c in node.children], axis=0)
        strat = node.avg_strategy()
        if lock is not None:
            strat = lock(node, strat, r1 if p == 0 else r0)
        u = np.zeros(r0.shape)
        for a, c in enumerate(node.children):
            if p == 0:
                u += self._best_response(c, p, r0, r1 * strat[:, a, :], lock)
            else:
                u += self._best_response(c, p, r0 * strat[:, a, :], r1, lock)
        return u

    def exploitability(self, r0, r1) -> float:
        """Average of what each player's best response wins against the other's average
        strategy, in percent of the pot at the root (0 at an exact equilibrium of this tree)."""
        a, b = r0[None, :].astype(np.float32), r1[None, :].astype(np.float32)
        raked = self.ctx.cfg.rake_rate > 0 or self.ctx.cfg.icm is not None
        u = core.evaluate(self, [(self.root, p, a, b, core.BR) for p in (0, 1)] +
                          [(self.root, p, a, b, core.AVG) for p in ((0, 1) if raked else ())])
        if u is None:
            u = [self._best_response(self.root, p, a, b) for p in (0, 1)]
        v0 = float((u[0][0] * r0).sum())
        v1 = float((u[1][0] * r1).sum())
        pairs = float((self.compat_sum(r1[None, :])[0] * r0).sum())
        if raked:
            # raked (or a tournament's payoffs), the game isn't zero-sum: each best response gains over what its player gets now,
            # and the two now sum to minus the rake the average strategies pay (0 without rake)
            if len(u) == 2:
                self.sampling = False
                try:
                    u += [self._cfr(self.root, p, a, b, update=False, use_avg=True) for p in (0, 1)]
                finally:
                    self.sampling = True
            for p in (0, 1):
                v0 -= float((u[2 + p][0] * (r0 if p == 0 else r1)).sum())
        return (v0 + v1) / 2 / max(pairs, 1e-12) / self.root.pot_now * 100

    def node_action_evs(self, node: Decision, r0, r1):
        """EV per action for the node's player, chips relative to folding now, both players
        on their average strategies. r0, r1: (N,) reaches at the node."""
        p = node.player
        ro = (r1 if p == 0 else r0)[None, :]
        cs = self.compat_sum(ro)[0]
        inv_p = _inv_of(node, p, self.ctx.cfg.icm)
        evs = np.zeros((len(node.names), self.ctx.n))
        got = core.evaluate(self, [(c, p, r0, r1, core.AVG) for c in node.children])
        self.sampling = False           # exact expectation over every card in the tree
        try:
            for a, child in enumerate(node.children):
                u = got[a][0] if got else self._cfr(child, p, r0[None, :], r1[None, :], update=False, use_avg=True)[0]
                evs[a] = np.where(cs > 1e-12, u / np.where(cs > 1e-12, cs, 1.0), 0.0) + inv_p
        finally:
            self.sampling = True
        return evs

    def node_action_br_evs(self, node: Decision, r0, r1, lock=None):
        """Like node_action_evs, but the node's player best-responds after each action (to the
        opponent's average strategy, or the strategy `lock` makes of it)."""
        p = node.player
        ro = (r1 if p == 0 else r0)[None, :]
        cs = self.compat_sum(ro)[0]
        inv_p = _inv_of(node, p, self.ctx.cfg.icm)
        evs = np.zeros((len(node.names), self.ctx.n))
        got = core.evaluate(self, [(c, p, r0, r1, core.BR) for c in node.children]) if lock is None else None
        for a, child in enumerate(node.children):
            u = got[a][0] if got else \
                self._best_response(child, p, r0[None, :].astype(np.float32), r1[None, :].astype(np.float32), lock)[0]
            evs[a] = np.where(cs > 1e-12, u / np.where(cs > 1e-12, cs, 1.0), 0.0) + inv_p
        return evs


def _inv_of(node, p: int, icm=None) -> float:
    """What p gives up by folding now (its chips in; with a tournament's payoffs, minus the prize equity of folding)."""
    n = node
    while not isinstance(n, Terminal):
        n = n.child if isinstance(n, (Chance, BucketEntry)) else n.children[0]
    if icm is not None:
        return -float(np.ravel(icm.fold(tuple(float(np.ravel(x)[0]) for x in n.inv), p)[p])[0])
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

    def exploitability(self) -> float:
        """Percent of the pot a best response wins on average (see Solver.exploitability)."""
        return self.solver.exploitability(self.r0, self.r1)

    def extend(self, time_budget: float, max_iters: int = 1000, should_stop=None, reserve=None):
        """Keep solving from the current regrets for another `time_budget` seconds (`reserve`: Solver.run)."""
        if self.iters >= max_iters:
            return self
        _, secs = self.solver.run(self.r0, self.r1, time_budget=time_budget, max_iters=max_iters, min_iters=1,
                                  should_stop=should_stop, reserve=reserve)
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


def warm_start(dst: Decision, dst_ctx: Ctx, src: Decision, src_ctx: Ctx):
    """Copy regrets and average-strategy sums from another solve of the same spot into this
    tree wherever the two line up: same action sequence, same board, same combo. Lets a
    bigger tree (more turn/river cards) pick up where a smaller one stopped."""
    pos = np.full(N_FULL, -1)
    pos[src_ctx.active] = np.arange(src_ctx.n)
    s_cols = pos[dst_ctx.active]
    d_cols = np.nonzero(s_cols >= 0)[0]
    s_cols = s_cols[d_cols]

    def walk(d, s):
        if isinstance(d, Chance) and isinstance(s, Chance):
            return walk(d.child, s.child)
        if not (isinstance(d, Decision) and isinstance(s, Decision)) or d.names != s.names:
            return
        rows = {tuple(sorted(b)): i for i, b in enumerate(s.boards)}
        pairs = [(i, rows[k]) for i, k in enumerate(tuple(sorted(b)) for b in d.boards) if k in rows]
        if pairs:
            d_rows, s_rows = (np.array(x) for x in zip(*pairs))
            for f in ("regret", "ssum"):
                getattr(d, f)[d_rows[:, None, None], np.arange(len(d.names))[None, :, None], d_cols[None, None, :]] = \
                    getattr(s, f)[s_rows[:, None, None], np.arange(len(s.names))[None, :, None], s_cols[None, None, :]]
        for dc, sc in zip(d.children, s.children):
            walk(dc, sc)

    walk(dst, src)


def lift_classes(dst: "Solution", src: "Solution"):
    """Copy a river-classes turn solve's average strategy (src, Config.river_classes) into the same spot's tree with
    every river card (dst, built without iterating: max_iters=0): river card c's nodes take its class's, each combo
    the strategy of its image under the swap. dst.exploitability() is then src's strategy measured in the real game."""
    pos = np.full(N_FULL, -1, dtype=np.intp)
    pos[src.ctx.active] = np.arange(src.ctx.n)
    d_full = dst.ctx.active
    cols = pos[d_full]

    def walk(d, s, rows, colmap):
        if isinstance(d, Chance):
            classes = {c: (k, played) for k, (played, cs) in enumerate(src.ctx.turn_classes(src.root.board, src.solver.dead))
                       for c in cs}
            rmap, cmaps = [], []
            for b in d.child.boards:
                k, played = classes[b[-1]]
                rmap.append(k)
                x, y = played, b[-1]
                a, bb = CA_FULL[d_full], CB_FULL[d_full]
                img = COMBO_TABLE[np.where(a == x, y, np.where(a == y, x, a)), np.where(bb == x, y, np.where(bb == y, x, bb))]
                cmaps.append(pos[img])
            return walk(d.child, s.child, np.array(rmap), np.array(cmaps))
        if not isinstance(d, Decision):
            return
        cm = np.broadcast_to(colmap, (d.B, len(d_full)))
        ok = cm >= 0
        src_ss = s.ssum[rows]                                       # (B, A, n_src)
        vals = np.take_along_axis(src_ss, np.where(ok, cm, 0)[:, None, :].repeat(len(d.names), axis=1), axis=2)
        d.ssum[...] = np.where(ok[:, None, :], vals, 0.0)
        for dc, sc in zip(d.children, s.children):
            walk(dc, sc, rows, colmap)

    walk(dst.root, src.root, np.zeros(1, dtype=np.intp), cols[None, :])
    dst._refresh()


class Cancelled(Exception):
    """A solve was stopped early through its should_stop callback."""


class TooSlow(Exception):
    """Solver.run's first iteration took longer than its max_per (seconds): the tree is too big for the budget.
    solve() adds the tree's cells (TreeBuilder.cells)."""

    def __init__(self, per: float):
        super().__init__(f"an iteration takes {per:.2f} s")
        self.per, self.cells = per, 0


def expand(ctx: Ctx, v: np.ndarray) -> np.ndarray:
    out = np.zeros(N_FULL)
    out[ctx.active] = v
    return out


def _class_closure(active: np.ndarray, classes) -> np.ndarray:
    """`active` (full combo indices) with every combo's images under the river classes' card swaps added (with zero
    reach): a class subtree's combo h stands for its images on the class's other cards, so it is needed when any of
    them is in a range (5c4s on a 5c river plays 5s4s's hand on the 5s)."""
    have = np.zeros(N_FULL, dtype=bool)
    have[active] = True
    a, b = CA_FULL[active], CB_FULL[active]
    for x, cs in classes:
        for y in cs:
            if y != x:
                have[COMBO_TABLE[np.where(a == x, y, np.where(a == y, x, a)), np.where(b == x, y, np.where(b == y, x, b))]] = True
    return np.flatnonzero(have)


def _spot(board: tuple, r_oop: np.ndarray, r_ip: np.ndarray, cfg: Config, dead: Sequence[int]):
    """A solve's combos (Ctx), its active combos, the reaches with the board's cards out, and the dead cards."""
    bm = np.ones(N_FULL)
    for c in list(board) + list(dead):
        bm *= 1.0 - MEMBER_FULL[c]
    r0f, r1f = r_oop * bm, r_ip * bm
    active = np.nonzero((r0f > 0) | (r1f > 0))[0]
    if cfg.river_classes and len(board) == 4:
        active = _class_closure(active, Ctx(active, cfg).turn_classes(board, set(dead) | set(board)))
    ctx = Ctx(active, cfg)
    ctx.root_len = len(board)
    return ctx, active, r0f, r1f, set(dead) | set(board)


def tree_cells(board: Sequence[int], r_oop: np.ndarray, r_ip: np.ndarray, pot: float, stack: float,
               bets=(0.0, 0.0), to_act: int = 0, n_raises: int = 0, cfg: Optional[Config] = None,
               dead: Sequence[int] = (), max_cells: Optional[int] = None) -> int:
    """The cells (TreeBuilder.cells) of the tree solve() would build, counted without building it (no arrays: a
    few ms where the build takes ~0.1 s a million cells in Pyodide); TreeTooBig past max_cells."""
    board = tuple(board)
    ctx, _, _, _, dead_set = _spot(board, r_oop, r_ip, cfg or Config(), dead)
    tb = TreeBuilder(ctx, dead=list(dead_set), max_cells=max_cells, dry=True)
    tb.build(board, pot, stack, bets, to_act, n_raises)
    return tb.cells


def solve(board: Sequence[int], r_oop: np.ndarray, r_ip: np.ndarray, pot: float, stack: float,
          bets=(0.0, 0.0), to_act: int = 0, n_raises: int = 0, cfg: Optional[Config] = None,
          time_budget: float = 3.0, max_iters: int = 1000, dead: Sequence[int] = (), should_stop=None,
          warm: Optional[Solution] = None, reserve: Optional[float] = None, horizon: Optional[int] = None,
          max_cells: Optional[int] = None, max_per: Optional[float] = None, seed=None) -> Solution:
    """`warm`: an earlier solve of the same spot (possibly with a different card sampling) to start from.
    `seed(root, solver)`: else sets starting regrets before the first iteration (Advisor._store_seed: a library
    flop's stored turn).
    `max_cells`: TreeTooBig, before the tree is all built, when its cells (TreeBuilder.cells) pass this.
    `max_per`: TooSlow (with the tree's cells) when the first iteration takes longer than this (seconds).
    `horizon`: the schedule's length (Solver.horizon) when the solve is run in pieces (max_iters the first piece).
    `reserve` (iterations, see Solver.run): when given, the budget also covers building the tree (the Accurate
    flop's takes ~1-1.5 s in Pyodide) and that many iterations' time after the solve."""
    t_start = time.time()
    cfg = cfg or Config()
    board = tuple(board)
    ctx, active, r0f, r1f, dead_set = _spot(board, r_oop, r_ip, cfg, dead)
    tb = TreeBuilder(ctx, dead=list(dead_set), max_cells=max_cells)
    root = tb.build(board, pot, stack, bets, to_act, n_raises)
    r0, r1 = r0f[active], r1f[active]
    ctx.root_reach = (r0, r1)
    s = Solver(ctx, root, dead_set)
    s.horizon = horizon
    if warm is not None:
        # DCFR's iteration count starts over: carrying it on would weight the copied average strategy
        # (from a different tree, and slightly different ranges) like hundreds of iterations of this one
        warm_start(root, ctx, warm.root, warm.ctx)
    elif seed is not None:
        seed(root, s)
    if reserve is not None:
        time_budget = max(time_budget - (time.time() - t_start), 0.0)
    t_run = time.time()
    try:
        iters, secs = s.run(r0, r1, time_budget=time_budget, max_iters=max_iters, should_stop=should_stop,
                            reserve=reserve, max_per=max_per)
    except TooSlow as e:
        e.cells = tb.cells
        raise
    except Cancelled as e:
        # (stopped part way: the solve so far rides on the exception, its iterations in the solver, for the caller to
        # keep and go on with; its root mix and EVs aren't read until then)
        e.sol = Solution(root, root.names, None, None, s.iters, time.time() - t_run, tb.n_nodes, ctx, r0, r1, s)
        e.sol.cells = tb.cells
        raise
    sol = Solution(root, root.names, None, None, iters, secs, tb.n_nodes, ctx, r0, r1, s)
    sol.cells = tb.cells
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
