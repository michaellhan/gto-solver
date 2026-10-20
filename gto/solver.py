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
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .cards import ALL_COMBOS, COMBO_INDEX
from .evaluator import eval_board
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
    n_rivers: Optional[int] = None    # rivers in the tree for full-decision river subtrees from the turn (None = all)
    river_sample: Optional[int] = None  # rivers traversed per iteration from the turn, redrawn every iteration
                                      # (public chance sampling; None = traverse all of them)
    n_rivers_from_flop: int = 8       # rivers per turn for full-decision river subtrees when the root is the flop
    leaf_turns: int = 6               # sampled turns for equity leaves reached from the flop
    leaf_rivers: Optional[int] = None # rivers for equity leaves when solving from the turn (None = all)
    leaf_rivers_flop: int = 16        # rivers per turn card for equity leaves when solving from the flop
    seed: int = 7
    # rake: the winner of a postflop pot (showdown or fold) pays rake_rate of the final pot (without an
    # uncalled bet, which goes back), at most
    # rake_cap bb (None = no cap). 0 = no rake. Preflop pots aren't raked ("no flop, no drop").
    rake_rate: float = 0.0
    rake_cap: Optional[float] = None

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
                      seed=self.seed, rake_rate=self.rake_rate, rake_cap=self.rake_cap)

    def sizes_key(self) -> tuple:
        return (tuple((k, tuple(v)) for k, v in sorted(self.bet_sizes.items())),
                tuple((k, tuple(v)) for k, v in sorted(self.raise_sizes.items())), self.depth, self.max_raises) + \
            ((("rake", self.rake_rate, self.rake_cap),) if self.rake_rate > 0 else ())


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
        self.compat = 1.0 - np.clip(self.member.T @ self.member, 0, 1)
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._M: Dict[tuple, np.ndarray] = {}
        self._leaf: Dict[tuple, np.ndarray] = {}
        self._stacks: Dict[tuple, np.ndarray] = {}
        self._turn_picks: Dict[tuple, tuple] = {}   # flop -> (picks in draw order, order random)
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

    def ranking(self, boards) -> "Ranking":
        """Sorted-strength index for a list of river boards (shared by every terminal on them)."""
        key = ("rank", tuple(boards))
        R = self._stacks.get(key)
        if R is None:
            R = Ranking(self, [self.strength(b) for b in boards])
            self._stacks[key] = R
        return R

    def card_rows(self):
        """(52, m) combo indices holding each card, padded with n (a dummy combo)."""
        if not hasattr(self, "_card_rows"):
            rows = [np.nonzero(self.member[c])[0] for c in range(52)]
            m = max(1, max(len(r) for r in rows))
            G = np.full((52, m), self.n, dtype=np.int64)
            for c, r in enumerate(rows):
                G[c, :len(r)] = r
            self._card_rows = G
        return self._card_rows

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
            # average per pair over the picked cards neither hand holds (the sub-matrices
            # are zero for the rest), not over all picks
            h = self.member[picks].sum(axis=0)
            L /= np.maximum(len(picks) - h[:, None] - h[None, :], 1.0).astype(np.float32)
        self._leaf[key] = L
        return L


class Ranking:
    """Showdown utilities in O(N) per board instead of an N x N matrix product.

    For combo i, wins = reach of all weaker opponent combos minus the weaker ones that share
    a card with i, and likewise for losses. The card-sharing part is computed per card from
    each card's combos in strength order; only i itself holds both of i's cards and it is
    neither weaker nor stronger than itself, so no inclusion-exclusion term is needed."""

    def __init__(self, ctx: "Ctx", strengths):
        n = ctx.n
        G = ctx.card_rows()                                   # (52, m)
        B, m = len(strengths), G.shape[1]
        S = np.stack(strengths).astype(np.float64)            # (B, n); -1 = holds a board card
        self.order = np.argsort(S, axis=1, kind="stable")
        ss = np.take_along_axis(S, self.order, axis=1)
        self.lo = np.empty((B, n), dtype=np.int64); self.hi = np.empty((B, n), dtype=np.int64)
        Sp = np.concatenate([S, np.full((B, 1), np.inf)], axis=1)
        self.G = np.empty((B, 52, m), dtype=np.int64)
        self.klo = np.zeros((B, 2, n), dtype=np.int64); self.khi = np.zeros((B, 2, n), dtype=np.int64)
        for b in range(B):
            self.lo[b] = np.searchsorted(ss[b], S[b], "left")
            self.hi[b] = np.searchsorted(ss[b], S[b], "right")
            sg = Sp[b][G]                                     # (52, m)
            o = np.argsort(sg, axis=1, kind="stable")
            self.G[b] = np.take_along_axis(G, o, axis=1)
            sg = np.take_along_axis(sg, o, axis=1)
            for slot, cards in enumerate((ctx.ca, ctx.cb)):
                for c in np.unique(cards):
                    idx = np.nonzero(cards == c)[0]
                    self.klo[b, slot, idx] = np.searchsorted(sg[c], S[b, idx], "left")
                    self.khi[b, slot, idx] = np.searchsorted(sg[c], S[b, idx], "right")
        self.blocked = S < 0
        # every lookup below is a flat index into a raveled (B, ...) array
        rowc = np.arange(B)[:, None] * (52 * (m + 1))
        cw = [rowc + cards[None, :] * (m + 1) for cards in (ctx.ca, ctx.cb)]
        self.flo = np.concatenate([(cw[k] + self.klo[:, k]).ravel() for k in range(2)])
        self.fhi = np.concatenate([(cw[k] + self.khi[:, k]).ravel() for k in range(2)])
        self.fend = np.concatenate([(cw[k] + m).ravel() for k in range(2)])
        rown = np.arange(B)[:, None] * (n + 1)
        self.fsort = (np.arange(B)[:, None] * n + self.order).ravel()
        self.fwin = (rown + self.lo).ravel()
        self.flose = (rown + self.hi).ravel()
        self.ftop = np.broadcast_to(rown + n, (B, n)).ravel()
        self.Gflat = (np.arange(B)[:, None, None] * (n + 1) + self.G).ravel()
        self.n, self.m, self.B = n, m, B

    def utility(self, ro: np.ndarray, sel=None) -> np.ndarray:
        """ro: (B, n) opponent reach -> (B, n) sum over compatible opponent combos of
        +reach (weaker) / -reach (stronger). With `sel`, ro holds only those boards' rows."""
        if sel is not None:
            return self._sub(sel).utility(ro)
        B, n, m = self.B, self.n, self.m
        cum = np.zeros((B, n + 1), dtype=ro.dtype); np.cumsum(ro.ravel()[self.fsort].reshape(B, n), axis=1, out=cum[:, 1:])
        cum = cum.ravel()
        u = cum[self.fwin] - (cum[self.ftop] - cum[self.flose])             # wins - losses
        rp = np.concatenate([ro, np.zeros((B, 1), dtype=ro.dtype)], axis=1).ravel()
        cg = np.zeros((B, 52, m + 1), dtype=ro.dtype); np.cumsum(rp[self.Gflat].reshape(B, 52, m), axis=2, out=cg[:, :, 1:])
        cg = cg.ravel()
        # card-sharing corrections: -weaker + stronger, for each of the two cards
        corr = cg[self.fend] - cg[self.fhi] - cg[self.flo]
        u += corr[:B * n] + corr[B * n:]
        u = u.reshape(B, n)
        u[self.blocked] = 0.0
        return u

    def _sub(self, sel) -> "Ranking":
        """The same index restricted to boards `sel` (flat offsets rebuilt for fewer rows)."""
        B, n, m = len(sel), self.n, self.m
        r = Ranking.__new__(Ranking)
        r.B, r.n, r.m = B, n, m
        r.blocked = self.blocked[sel]
        rows = np.arange(B)[:, None]
        r.fsort = (rows * n + self.order[sel]).ravel()
        r.fwin = (rows * (n + 1) + self.lo[sel]).ravel()
        r.flose = (rows * (n + 1) + self.hi[sel]).ravel()
        r.ftop = np.broadcast_to(rows * (n + 1) + n, (B, n)).ravel()
        r.Gflat = (rows[:, :, None] * (n + 1) + self.G[sel]).ravel()
        shift = (np.asarray(sel)[:, None] - rows) * (52 * (m + 1))
        full = self.B * n
        for name in ("flo", "fhi", "fend"):
            flat = getattr(self, name)
            setattr(r, name, np.concatenate([(flat[k * full:(k + 1) * full].reshape(self.B, n)[sel] - shift).ravel()
                                             for k in range(2)]))
        return r


# ----------------------------------------------------------------- tree nodes
class Terminal:
    """`inv`: each player's chips in, floats, or (B, 1) float32 columns for a merged terminal (one
    value per batch row). `tile`: set on a merged equity leaf whose rows repeat the same `tile` boards.
    `rake`: what the winner pays out of the pot (a float, or a (B, 1) column when merged); None = no rake."""
    __slots__ = ("boards", "kind", "folder", "inv", "M", "tile", "rake")

    def __init__(self, boards, kind, inv, folder=None, rake=None):
        self.boards, self.kind, self.inv, self.folder = boards, kind, inv, folder
        self.M = None
        self.tile = None
        self.rake = rake if rake is None or np.any(rake > 0) else None

    def fold_pay(self, p):
        """p's chips won (lost when negative) per opponent combo at a fold."""
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
    __slots__ = ("boards", "K", "masks", "child", "norm", "avail", "sample")

    def __init__(self, boards, K, masks, child, avail, sample=None):
        self.boards, self.K, self.masks, self.child = boards, K, masks, child
        self.avail = avail
        self.sample = sample if sample is not None and sample < K else None
        B = len(boards)
        self.norm = self.norm_for(masks.reshape(B, K, -1))

    def norm_for(self, m3: np.ndarray) -> np.ndarray:
        """m3: (B, k, N) masks of the traversed cards -> (B, N) divisor."""
        return np.maximum(m3.sum(axis=1) * ((self.avail - 4) / (self.avail - 2)), 1e-9)


class Decision:
    __slots__ = ("boards", "player", "names", "children", "regret", "ssum", "pot_now", "bets", "B")

    def __init__(self, boards, player, names, children, pot_now, bets, n):
        self.boards, self.player, self.names, self.children = boards, player, names, children
        self.pot_now, self.bets = pot_now, bets
        self.B = len(boards)
        A = len(names)
        self.regret = np.zeros((self.B, A, n), dtype=np.float32)
        self.ssum = np.zeros((self.B, A, n), dtype=np.float32)

    @property
    def board(self):
        return self.boards[0]

    def strategy(self, sel=None) -> np.ndarray:
        # regret matching in as few numpy calls as possible (a native solve is call-overhead bound): one
        # in-place divide, then uniform wherever no regret is positive (rare once a solve is under way)
        pos = np.maximum(self.regret if sel is None else self.regret[sel], 0.0)
        tot = pos.sum(axis=1, keepdims=True)
        np.divide(pos, tot, out=pos)
        zero = tot == 0
        if zero.any():
            np.copyto(pos, np.float32(1.0 / len(self.names)), where=zero)
        return pos

    def avg_strategy(self, sel=None) -> np.ndarray:
        """(B, A, N) average strategy (rows `sel` only if given)."""
        ss = self.ssum if sel is None else self.ssum[sel]
        tot = ss.sum(axis=1, keepdims=True)
        return np.where(tot > 0, ss / np.where(tot > 0, tot, 1.0), 1.0 / len(self.names))

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
            return Terminal(boards, "showdown", inv, rake=self._rake(inv))
        if allin or new_stack <= 1e-9:
            return Terminal(boards, "leaf", inv, rake=self._rake(inv))
        streets_used = len(boards[0]) - self.root_street_len + 1
        if streets_used >= self.cfg.depth:
            # an equity leaf: raked on the pot as it stands. The leaf pays equity x this pot (later bets
            # aren't modelled, neither the chips they'd add nor their rake), so raking this pot keeps the
            # rake's share of what the leaf pays right uncapped; with a cap, the bigger final pot's share
            # would be lower, so it's slightly high there (in absolute bb, slightly low)
            return Terminal(boards, "leaf", inv, rake=self._rake(inv))
        return self._chance(boards, new_pot, new_stack, inv)

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
        masks = np.ones((len(boards) * K, self.ctx.n), dtype=np.float32)
        for bi, b in enumerate(boards):
            for ki in range(K):
                c = cards[bi][ki]
                child_boards.append(b + (c,))
                masks[bi * K + ki] = 1.0 - self.ctx.member[c]
        child = self._decision(child_boards, pot, stack, (0.0, 0.0), 0, 0, inv, False)
        self.n_nodes += 1
        avail = sum(1 for c in range(52) if c not in self.dead and c not in boards[0])
        return Chance(boards, K, masks, child, avail, sample)

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
            kids.append(Terminal(boards, "fold", inv, folder=p, rake=self._rake(inv)))
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


# ----------------------------------------------------------------- merged subtrees
def _shape(node) -> tuple:
    """A subtree's structure without pots, bet amounts or cards: subtrees with the same shape can
    be traversed as one batch."""
    if isinstance(node, Decision):
        return ("D", node.player, len(node.boards), tuple(_shape(c) for c in node.children))
    if isinstance(node, Chance):
        return ("C", node.K, node.sample, node.avail, len(node.boards), _shape(node.child))
    return ("T", node.kind, node.folder, len(node.boards))


def _merge(nodes, pairs: list):
    """One node standing for same-shaped subtrees `nodes`, their batch rows stacked in order.
    Appends (merged decision, originals) to `pairs` so the originals' arrays can become views."""
    first = nodes[0]
    boards = [b for n in nodes for b in n.boards]
    if isinstance(first, Decision):
        m = Decision.__new__(Decision)
        m.boards, m.player, m.names, m.pot_now, m.bets, m.B = boards, first.player, first.names, None, None, len(boards)
        m.children = [_merge([n.children[a] for n in nodes], pairs) for a in range(len(first.children))]
        m.regret = np.concatenate([n.regret for n in nodes])
        m.ssum = np.concatenate([n.ssum for n in nodes])
        pairs.append((m, nodes))
        return m
    if isinstance(first, Chance):
        return Chance(boards, first.K, np.concatenate([n.masks for n in nodes]), _merge([n.child for n in nodes], pairs),
                      first.avail, first.sample)
    # chips in differ between the merged lines (different pots): one float32 per row, which is what a
    # Python float becomes when it multiplies a float32 array, so the results don't change
    inv = tuple(np.concatenate([np.full((len(n.boards), 1), n.inv[q], dtype=np.float32) for n in nodes]) for q in (0, 1))
    rake = None
    if any(n.rake is not None for n in nodes):
        rake = np.concatenate([np.full((len(n.boards), 1), n.rake or 0.0, dtype=np.float32) for n in nodes])
    t = Terminal(boards, first.kind, inv, first.folder, rake)
    if first.kind == "leaf" and all(n.boards == first.boards for n in nodes):
        t.tile = len(first.boards)
    return t


def _has_sampling(node) -> bool:
    if isinstance(node, Chance):
        return node.sample is not None or _has_sampling(node.child)
    return isinstance(node, Decision) and any(_has_sampling(c) for c in node.children)


# ----------------------------------------------------------------- solver
class Solver:
    """Speed (natively the solve is bound by Python and numpy per-call overhead, besides the leaf matvecs):
    * the subtrees below the root street's chance nodes mostly share one shape (pots differ, actions
      don't), so same-shaped ones are merged into one batch at setup and traversed once: a flop solve
      walks ~30 decision nodes per pass instead of ~90. The root street is walked down to collect the
      reaches arriving at each chance node, the merged subtrees run, and the root street is walked back up.
    * every node's regrets and strategy sums are views into two flat buffers, discounted in one call
    * a player's strategies computed in the other player's pass are reused in their own (unchanged regrets),
      below the root street too; there the NO_BLAS kernels also trim the leaf matrices to each range
    All of it gives the same numbers as walking the tree node by node. The original nodes' arrays are views of the merged ones, so reading and EV code sees current values."""

    def __init__(self, ctx: Ctx, root: Decision, dead: set):
        self.ctx, self.root, self.dead = ctx, root, dead
        self.iters = 0
        self.alpha, self.beta, self.gamma = 1.5, 0.0, 2.0   # DCFR parameters
        self.rng = random.Random(ctx.cfg.seed + 2)
        self.sampling = True          # chance sampling while training; off for EVs and reports
        self._nodes: List[Decision] = []
        self._collect(root)
        self._groups: List[Tuple[Chance, List[Chance]]] = []
        self._merge_subtrees()
        # training on a numpy without BLAS (see NO_BLAS) skips the leaf-matrix rows and columns of combos
        # outside each player's range (always 0 reach): ~1/3 less arithmetic on a typical flop
        self.trim_leaves = NO_BLAS
        self._live = None
        self._trimmed: Dict[tuple, np.ndarray] = {}

    def _collect(self, node):
        if isinstance(node, Decision):
            self._nodes.append(node)
            for c in node.children:
                self._collect(c)
        elif isinstance(node, Chance):
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
        replaced = {id(n) for _, nodes in pairs for n in nodes}
        owners = [n for n in self._nodes if id(n) not in replaced] + [m for m, _ in pairs]
        for f in ("regret", "ssum"):
            flat = np.empty(sum(getattr(n, f).size for n in owners), dtype=np.float32)
            off = 0
            for n in owners:
                a = getattr(n, f)
                v = flat[off:off + a.size].reshape(a.shape)
                v[...] = a
                setattr(n, f, v)
                off += a.size
            setattr(self, "_" + f, flat)
        for m, nodes in pairs:
            off = 0
            for n in nodes:
                n.regret, n.ssum = m.regret[off:off + n.B], m.ssum[off:off + n.B]
                off += n.B

    def compat_sum(self, ro):
        """ro: (B, N) -> (B, N) mass of opponent combos sharing no card with each combo."""
        member = self.ctx.member32 if ro.dtype == np.float32 else self.ctx.member
        cs = (np.einsum("bn,cn->bc", ro, member) if NO_BLAS else ro @ member.T)[:, self.ctx.cab]   # (B, 2N)
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
        u, f32 = self._showdown(node, p, ro, sel, train)
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

    def _showdown(self, node: Terminal, p, ro, sel=None, train=False):
        """(sum over opponent combos of reach * (P(win) - P(lose)), whether p's chips in multiply it as
        float32) at a showdown or equity leaf."""
        if len(node.boards[0]) == 5:
            if node.M is None:
                node.M = self.ctx.ranking(node.boards)
            return node.M.utility(ro, sel), False
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
        if isinstance(node, Chance):
            if done is not None and id(node) in done:
                return done[id(node)]
            B, K, n = r0.shape[0], node.K, self.ctx.n
            rows = np.arange(B) if sel is None else np.asarray(sel)
            if node.sample is not None and self.sampling:
                ks = np.array(sorted(self.rng.sample(range(K), node.sample)))
            else:
                ks = np.arange(K)
            k = len(ks)
            csel = (rows[:, None] * K + ks[None, :]).ravel()
            m3 = node.masks[csel].reshape(B, k, n)
            r0c = (r0[:, None, :] * m3).reshape(B * k, n)
            r1c = (r1[:, None, :] * m3).reshape(B * k, n)
            full = sel is None and k == K
            u = self._cfr(node.child, p, r0c, r1c, update, use_avg, None if full else csel, strats)
            norm = node.norm_for(m3) if k < K else node.norm[rows]
            return (u.reshape(B, k, n) * m3).sum(axis=1) / norm
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
                if sel is None:
                    node.regret += utils - node_util[:, None, :]
                    node.ssum += rp[:, None, :] * strat
                else:
                    node.regret[sel] += utils - node_util[:, None, :]
                    node.ssum[sel] += rp[:, None, :] * strat
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

    def _down(self, node, p, r0, r1, strats, arrive):
        """The root street's walk down: strategies, and the reaches arriving at each chance node."""
        if isinstance(node, Chance):
            arrive[id(node)] = (r0, r1)
        elif isinstance(node, Decision):
            cache = strats[node.player]
            strat = cache.get(id(node))
            if strat is None:
                strat = cache[id(node)] = node.strategy()
            for a, child in enumerate(node.children):
                if isinstance(child, Terminal):
                    continue
                if node.player == p:
                    self._down(child, p, r0, r1, strats, arrive)
                elif p == 0:
                    self._down(child, p, r0, r1 * strat[:, a, :], strats, arrive)
                else:
                    self._down(child, p, r0 * strat[:, a, :], r1, strats, arrive)

    def _traverse(self, p, r0, r1, strats):
        """One training pass for p. Same result as _cfr on the root; the merged subtrees run in one batch.
        strats: (player 0's, player 1's) current strategies by node; p's are stale once this pass updates them."""
        if not self._groups:
            return self._cfr(self.root, p, r0, r1, update=True, use_avg=False, strats=strats)
        arrive, done = {}, {}
        self._down(self.root, p, r0, r1, strats, arrive)
        for chance, members in self._groups:
            u = self._cfr(chance, p, np.concatenate([arrive[id(c)][0] for c in members]),
                          np.concatenate([arrive[id(c)][1] for c in members]), update=True, use_avg=False, strats=strats)
            off = 0
            for c in members:
                done[id(c)] = u[off:off + len(c.boards)]
                off += len(c.boards)
        return self._cfr(self.root, p, r0, r1, update=True, use_avg=False, strats=strats, done=done)

    def _discount(self):
        t = float(self.iters)
        pos_f = (t ** self.alpha) / (t ** self.alpha + 1)
        neg_f = 0.5 if self.beta == 0 else (t ** self.beta) / (t ** self.beta + 1)
        s_f = (t / (t + 1)) ** self.gamma
        r = self._regret
        r *= np.where(r > 0, pos_f, neg_f)
        self._ssum *= s_f

    def run(self, r0, r1, time_budget=3.0, max_iters=1000, min_iters=10, should_stop=None):
        """r0, r1: (N,) root reaches. Adds iterations until the budget is spent.
        Raises Cancelled as soon as should_stop() is true."""
        r0b, r1b = r0[None, :].astype(np.float32), r1[None, :].astype(np.float32)
        if self.trim_leaves:
            live = (np.flatnonzero(r0 > 0), np.flatnonzero(r1 > 0))
            if self._live is None or any(not np.array_equal(a, b) for a, b in zip(live, self._live)):
                self._live, self._trimmed = live, {}
        t0 = time.time()
        start = self.iters
        while self.iters < max_iters:
            if should_stop is not None and should_stop():
                raise Cancelled()
            # the opponent's strategies computed in player 0's pass are still current in player 1's
            strats = ({}, {})
            for p in (0, 1):
                self._traverse(p, r0b, r1b, strats)
                strats[p].clear()
            self.iters += 1
            self._discount()
            if self.iters - start >= min_iters and time.time() - t0 > time_budget:
                break
        return self.iters, time.time() - t0

    def _best_response(self, node, p, r0, r1, lock=None):
        """Counterfactual utility for p playing a best response to the opponent's average
        strategy, every chance card enumerated. `lock(node, strat, reach)` turns the opponent's
        strategy into the one they actually play (exploit mode's player pool)."""
        if isinstance(node, Terminal):
            return self._cfr(node, p, r0, r1, update=False, use_avg=True)
        if isinstance(node, Chance):
            B, K, n = r0.shape[0], node.K, self.ctx.n
            m3 = node.masks.reshape(B, K, n)
            u = self._best_response(node.child, p, (r0[:, None, :] * m3).reshape(B * K, n),
                                    (r1[:, None, :] * m3).reshape(B * K, n), lock)
            return (u.reshape(B, K, n) * m3).sum(axis=1) / node.norm
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
        v0 = float((self._best_response(self.root, 0, a, b)[0] * r0).sum())
        v1 = float((self._best_response(self.root, 1, a, b)[0] * r1).sum())
        pairs = float((self.compat_sum(r1[None, :])[0] * r0).sum())
        if self.ctx.cfg.rake_rate > 0:
            # raked, the game isn't zero-sum: each best response gains over what its player gets now,
            # and the two now sum to minus the rake the average strategies pay (0 without rake)
            self.sampling = False
            try:
                for p in (0, 1):
                    u = self._cfr(self.root, p, a, b, update=False, use_avg=True)[0]
                    v0 -= float((u * (r0 if p == 0 else r1)).sum())
            finally:
                self.sampling = True
        return (v0 + v1) / 2 / max(pairs, 1e-12) / self.root.pot_now * 100

    def node_action_evs(self, node: Decision, r0, r1):
        """EV per action for the node's player, chips relative to folding now, both players
        on their average strategies. r0, r1: (N,) reaches at the node."""
        p = node.player
        ro = (r1 if p == 0 else r0)[None, :]
        cs = self.compat_sum(ro)[0]
        inv_p = _inv_of(node, p)
        evs = np.zeros((len(node.names), self.ctx.n))
        self.sampling = False           # exact expectation over every card in the tree
        try:
            for a, child in enumerate(node.children):
                u = self._cfr(child, p, r0[None, :], r1[None, :], update=False, use_avg=True)[0]
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
        inv_p = _inv_of(node, p)
        evs = np.zeros((len(node.names), self.ctx.n))
        for a, child in enumerate(node.children):
            u = self._best_response(child, p, r0[None, :].astype(np.float32), r1[None, :].astype(np.float32), lock)[0]
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

    def exploitability(self) -> float:
        """Percent of the pot a best response wins on average (see Solver.exploitability)."""
        return self.solver.exploitability(self.r0, self.r1)

    def extend(self, time_budget: float, max_iters: int = 1000, should_stop=None):
        """Keep solving from the current regrets for another `time_budget` seconds."""
        if self.iters >= max_iters:
            return self
        _, secs = self.solver.run(self.r0, self.r1, time_budget=time_budget, max_iters=max_iters, min_iters=1,
                                  should_stop=should_stop)
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


class Cancelled(Exception):
    """A solve was stopped early through its should_stop callback."""


def expand(ctx: Ctx, v: np.ndarray) -> np.ndarray:
    out = np.zeros(N_FULL)
    out[ctx.active] = v
    return out


def solve(board: Sequence[int], r_oop: np.ndarray, r_ip: np.ndarray, pot: float, stack: float,
          bets=(0.0, 0.0), to_act: int = 0, n_raises: int = 0, cfg: Optional[Config] = None,
          time_budget: float = 3.0, max_iters: int = 1000, dead: Sequence[int] = (), should_stop=None,
          warm: Optional[Solution] = None) -> Solution:
    """`warm`: an earlier solve of the same spot (possibly with a different card sampling) to start from."""
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
    if warm is not None:
        # DCFR's iteration count starts over: carrying it on would weight the copied average strategy
        # (from a different tree, and slightly different ranges) like hundreds of iterations of this one
        warm_start(root, ctx, warm.root, warm.ctx)
    iters, secs = s.run(r0, r1, time_budget=time_budget, max_iters=max_iters, should_stop=should_stop)
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
