"""The full preflop game (2 to 6 players at 100bb by default) and its equilibrium over the 169 hand classes.

Each player is dealt a hand class (with its share of the 1326 combos); card removal between players is
class level (pf_classes.PAIR_COMPAT), each opponent's against hero's class independently. The tree: an
unopened pot is folded or raised (no limps, as in the charts); facing a raise a player folds, calls or
raises once more, to a 5bet all-in. One size a raise (Sizes: 2.5bb opens, 3bb from the small blind; 3bets
3x in position, 4.8x from a blind, 10bb in the big blind over a small-blind open, one more open per cold
caller; 4bets 2.3x, plus one 3bet per caller; then all-in). At most `max_players` players see a flop: a
player who would be one more than that can only fold or raise.

Terminal values (chips won, net of what each put in):
* everyone else folds: the pot;
* all-in preflop: the pot by all-in equity (heads-up the class equity matrix; three-way see _multi);
* a flop heads-up: the pot by each player's realized pot share, from a realization model (EQRModel) by pot
  type (single-raised, 3bet, 4bet), whether the aggressor (the last raiser) has position, and the two hands;
* a flop three-way: all-in equity three ways (MW_CORR, see _multi) by a realization factor per postflop
  seat and where the opener sits (MW_REALIZE, measured).
With `rake` (rate, cap) a pot that sees a flop is short the expected rake (advisor.RAKE_GROWTH times the pot
at the flop, to the cap), as the advice prices calls.

solve() runs discounted CFR (all players updated together each iteration, regrets discounted DCFR-style,
strategies averaged with linear weights) and returns each decision node's average strategy per class.
Multiplayer CFR has no convergence guarantee; exploitability() measures how much each player gains by
best-responding, in bb per hand, which is the check.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from .pf_classes import CLASSES, N_IN, PAIR_COMPAT, equity_matrix

K = len(CLASSES)
PRIOR = N_IN / N_IN.sum()
SIX = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
POST_ORDER = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]     # postflop action order (first = out of position)
BLINDS = ("SB", "BB")


def post_rank(positions: Sequence[str]) -> List[int]:
    """Each seat's place in the postflop order (0 acts first): the blinds, then the rest in seating order; heads-up
    the small blind (the button) acts last."""
    pos = list(positions)
    if len(pos) == 2:
        return [1, 0]
    rest = [p for p in pos if p not in BLINDS]
    return [0 if p == "SB" else 1 if p == "BB" else 2 + rest.index(p) for p in pos]


@dataclass
class Sizes:
    open: float = 2.5          # raise first in, bb (the small blind: sb_open)
    sb_open: float = 3.0
    ip_3bet: float = 3.0       # x the open, from a seat that isn't a blind
    blind_3bet: float = 4.8    # x the open, from a blind over a seat's open (2.5 -> 12)
    bvb_3bet: float = 10.0 / 3.0   # x the open, the big blind over a small-blind open (3 -> 10)
    four_x: float = 2.3        # 4bet, x the 3bet
    four_x_ip: Optional[float] = None    # in position against the 3bettor (None: four_x)
    four_x_oop: Optional[float] = None   # out of position against it (None: four_x)
    jam_frac: float = 0.4      # a raise putting in this share of the stack or more goes all-in
    iso: float = 3.5           # the big blind's raise over a small-blind limp (sb_limp), bb
    limp_3bet: float = 3.0     # the small blind's re-raise over that, x the raise


@dataclass
class Node:
    kind: str                              # "dec", "fold" (one player left) or "show" (a flop or an all-in)
    player: int = -1                       # seat index (dec)
    names: List[str] = field(default_factory=list)
    children: List["Node"] = field(default_factory=list)
    inv: Optional[np.ndarray] = None       # chips each seat put in (terminals)
    active: Tuple[int, ...] = ()           # seats still in (terminals)
    n_raises: int = 0
    aggressor: int = -1                    # last raiser (show)
    allin: bool = False
    line: Tuple[Tuple[str, str, float], ...] = ()   # (position, action, to)
    regret: Optional[np.ndarray] = None
    ssum: Optional[np.ndarray] = None


def build(positions: Sequence[str] = SIX, stack: float = 100.0, sizes: Optional[Sizes] = None,
          max_players: int = 3, max_players_raised: int = 2, sb_limp: bool = False, ante: float = 0.0,
          straddle: bool = False) -> Node:
    """The preflop tree for `positions` in seating order (the last two the blinds; heads-up SB, BB). `sb_limp`:
    the small blind first in may also complete (limp); the big blind then checks or raises (Sizes.iso), and the
    small blind facing that folds, calls or re-raises (Sizes.limp_3bet). `ante`: each player's ante, bb (dead
    money: in every terminal's chips, out of the stack before the betting). `straddle`: the first seat posts 2bb
    and acts last preflop; opens are then Sizes.open times the straddle."""
    sizes = sizes or Sizes()
    n = len(positions)
    pos = list(positions)
    sb, bb = (n - 2, n - 1) if n > 2 else (0, 1)    # heads-up the button is the small blind
    straddle = bool(straddle) and n >= 3
    stack = stack - ante                           # (the betting's: what is left after the ante)
    rank = post_rank(pos)

    def raise_to(p, cur, n_raises, last, callers, bets):
        if n_raises == 0:
            if p == bb and not straddle:         # over a small-blind limp
                return sizes.iso
            return (sizes.sb_open if p == sb else sizes.open) * cur
        if n_raises == 1:
            if p == sb and last == bb and abs(bets[sb] - 1.0) < 1e-9:
                return cur * sizes.limp_3bet
            if p == bb and last == sb:
                x = sizes.bvb_3bet
            elif p in (sb, bb):
                x = sizes.blind_3bet
            else:
                x = sizes.ip_3bet
            return cur * x + callers * cur
        if n_raises == 2:
            ip = rank[p] > rank[last]
            x = (sizes.four_x_ip if ip else sizes.four_x_oop) or sizes.four_x
            return cur * x + callers * cur
        return stack

    def rec(bets, folded, acted, n_raises, last, p, line):
        live = [q for q in range(n) if q not in folded]
        if len(live) == 1:
            return Node("fold", inv=np.array(bets) + ante, active=tuple(live), n_raises=n_raises, line=line)
        cur = max(bets)
        # who acts next: from p on, a live player not all-in who hasn't acted since the last raise or is behind
        order = [(p + k) % n for k in range(n)]
        nxt = next((q for q in order if q not in folded and bets[q] < stack - 1e-9
                    and (q not in acted or bets[q] < cur - 1e-9)), None)
        if nxt is None:
            return Node("show", inv=np.array(bets) + ante, active=tuple(live), n_raises=n_raises, aggressor=last,
                        allin=cur >= stack - 1e-9, line=line)
        q = nxt
        names, kids = [], []
        # fold (not when checking is free: the big blind behind a limp; a straddle's big blind faces 2bb)
        if bets[q] < cur - 1e-9:
            names.append("fold")
            kids.append(rec(bets, folded | {q}, acted | {q}, n_raises, last, q + 1, line + ((pos[q], "fold", 0.0),)))
        if n_raises == 0 and sb_limp and not straddle and q in (sb, bb) \
                and all(x in folded for x in range(n) if x not in (sb, bb)) and bets[q] < cur + 1e-9:
            # the small blind completes, or the big blind checks behind a limp (named "call": a flop at 1bb each)
            if q == sb or sb in acted:
                b2 = list(bets); b2[q] = cur
                names.append("call")
                kids.append(rec(b2, folded, acted | {q}, 0, last, q + 1, line + ((pos[q], "call", cur),)))
        if n_raises > 0:
            in_level = sum(1 for r in live if bets[r] >= cur - 1e-9)
            cap = max_players if n_raises == 1 else max_players_raised
            if cur >= stack - 1e-9:
                # calling an all-in: a showdown, not a pot to play, so up to three (else a 3bettor facing a
                # cold jam and a call could only fold: at 40bb with antes the solve 3bet nothing over a button
                # open, its 3bet answered by the big blind jamming any two and the button calling)
                cap = max(cap, 3)
            if in_level < cap:
                b2 = list(bets); b2[q] = min(cur, stack)
                names.append("call")
                kids.append(rec(b2, folded, acted | {q}, n_raises, last, q + 1, line + ((pos[q], "call", b2[q]),)))
        if cur < stack - 1e-9 and n_raises < 4:
            callers = sum(1 for r in live if r != last and r != q and bets[r] >= cur - 1e-9) if n_raises > 0 else 0
            to = raise_to(q, cur, n_raises, last, callers, bets)
            if to >= sizes.jam_frac * stack or n_raises == 3:
                to = stack
            b2 = list(bets); b2[q] = to
            names.append("raise" if to < stack else "all-in")
            kids.append(rec(b2, folded, {q}, n_raises + 1, q, q + 1, line + ((pos[q], names[-1], to),)))
        node = Node("dec", player=q, names=names, children=kids, n_raises=n_raises, line=line)
        return node

    bets = [0.0] * n
    bets[sb] += 0.5
    bets[bb] += 1.0
    first = 0 if n > 2 else sb
    if straddle:
        bets[0] += 2.0
        first = 1
    return rec(bets, frozenset(), frozenset(), 0, -1, first, ())


def walk(node: Node):
    yield node
    for c in node.children:
        yield from walk(c)


# ----------------------------------------------------------------------------- terminal values
class EQRModel:
    """Heads-up postflop pot shares: share(key)[i, j] is the aggressor's (class i) share of the pot against
    the caller's class j after a call that sees a flop; key = (pot type 1/2/3 = single-raised/3bet/4bet,
    whether the aggressor has position). Built from per-role realization functions: `excess(key, role)`
    returns a (K, K) matrix of how far the role's pot share against each opposing class runs above its
    equity; the two sides combine as share = eq + (x_aggr - x_caller^T) / 2, which keeps each pair's shares
    summing to the pot."""

    def __init__(self, excess: Optional[Callable[[tuple, str], np.ndarray]] = None, share_fn=None):
        self.excess = excess
        self.share_fn = share_fn
        self._cache: Dict[tuple, np.ndarray] = {}

    from_tables = staticmethod(lambda ex, w: _from_tables(ex, w))

    by_spot = False   # whether share() uses the seats of a detailed key (pot type, aggressor ip, its seat, caller's)

    def share(self, key: tuple) -> np.ndarray:
        if not self.by_spot:
            key = key[:2]
        s = self._cache.get(key)
        if s is None:
            if self.share_fn is not None:
                s = self.share_fn(key)
            else:
                eq = equity_matrix()
                xa = self.excess(key, "aggr")            # aggressor class i vs caller class j
                xc = self.excess(key, "caller")          # caller class j vs aggressor class i
                s = eq + (xa - xc.T) / 2
            s = self._cache[key] = np.clip(s, -3.0, 4.0)
        return s


class SpotModel(EQRModel):
    """Pot shares by spot: `models` maps (pot type, aggressor in position, aggressor's seat, caller's seat) to the
    model measured there; a flop between other seats uses the measured spot of the same pot type and position
    with the same caller (else the same aggressor), else `fallback` (pooled over the spots)."""
    by_spot = True

    def __init__(self, models: Dict[tuple, EQRModel], fallback: EQRModel):
        super().__init__(share_fn=self._pick)
        self.models, self.fallback = models, fallback

    def match(self, key: tuple):
        if len(key) < 4:
            return None
        if key in self.models:
            return key
        for k in self.models:
            if k[:2] == key[:2] and k[3] == key[3]:
                return k
        for k in self.models:
            if k[:2] == key[:2] and k[2] == key[2]:
                return k
        return None

    def _pick(self, key):
        k = self.match(key)
        return (self.models[k] if k else self.fallback).share(key[:2])


def _from_tables(ex: Dict[tuple, np.ndarray], w: Dict[tuple, np.ndarray]) -> "EQRModel":
    """A model from measured excess tables: ex[(pot type, aggressor in position, role)] is (K, buckets), the
    role's excess share against the opponent's hands in each strength bucket (pf_solve.bucket_of); w[...] the
    class weights of the role's range where it was measured. A pair's excess is the two-way additive
    decomposition e_ij = a(i, b_j) - c(j, b_i) - M(b_i, b_j), whose averages over the opponent's bucket give
    back each side's measurement (M the block mean, from the two sides' tables and range weights)."""
    from .pf_solve import bucket_of
    b = bucket_of()
    nb = int(b.max()) + 1
    onehot = np.eye(nb)[b]                                    # (K, buckets)

    def share(key):
        with np.errstate(all="ignore"):     # macOS Accelerate's matmul raises spurious warnings
            return _share(key)

    def _share(key):
        xa, xc = ex[key + ("aggr",)], ex[key + ("caller",)]
        wa, wc = w[key + ("aggr",)], w[key + ("caller",)]
        # block means: aggressor bucket x caller bucket, from each side
        ma = (onehot * wa[:, None]).T @ xa / np.maximum((onehot * wa[:, None]).sum(0), 1e-12)[:, None]
        mc = (onehot * wc[:, None]).T @ xc / np.maximum((onehot * wc[:, None]).sum(0), 1e-12)[:, None]
        M = (ma - mc.T) / 2                                   # (aggr bucket, caller bucket)
        e = xa[:, b] - xc[:, b].T - M[b][:, b]
        return equity_matrix() + e
    return EQRModel(share_fn=share)


EQR_FEATURES = ("1", "x", "x|x|", "pair", "suited", "connected", "gap", "high", "pair*x", "suited*x", "conn*x")


@lru_cache(maxsize=1)
def class_shape():
    """Per class: pair, suited, connected (gap 0 or 1, not pairs), gap (0..1, six or more ranks apart = 1) and
    high card (0..1)."""
    rk = "23456789TJQKA"
    pair = np.array([len(c) == 2 for c in CLASSES], float)
    suited = np.array([len(c) == 3 and c[2] == "s" for c in CLASSES], float)
    gap = np.array([0 if len(c) == 2 else abs(rk.index(c[0]) - rk.index(c[1])) - 1 for c in CLASSES], float)
    conn = (1 - pair) * (gap <= 1)
    high = np.array([rk.index(c[0]) / 12.0 for c in CLASSES])
    return pair, suited, conn, np.minimum(gap, 6) / 6 * (1 - pair), high


def eqr_features(e: np.ndarray) -> np.ndarray:
    """(K, buckets, features) of the realization model, for equity e (K, buckets) against each bucket."""
    pair, suited, conn, gap, high = (v[:, None] for v in class_shape())
    x = e - 0.5
    one = np.ones_like(x)
    return np.stack([one, x, x * np.abs(x), pair * one, suited * one, conn * one, gap * one, high * one,
                     pair * x, suited * x, conn * x], -1)


def excess_from_fit(coef, resid) -> np.ndarray:
    """(K, buckets) excess share: the smooth model on each class's equity against each bucket plus its residual."""
    from .pf_solve import _bucket_equity
    X = eqr_features(_bucket_equity())
    return np.clip(X @ np.asarray(coef) + np.asarray(resid)[:, None], -1.5, 2.5)


@lru_cache(maxsize=1)
def measured_model() -> Optional[EQRModel]:
    """The realization model measured with the postflop solver (gto/pf_eqr.py), or None."""
    try:
        from .pf_eqr import COEF, RESID, W
    except ImportError:
        return None
    ex = {k: excess_from_fit(COEF[k], RESID[k]) for k in COEF}
    return EQRModel.from_tables(ex, {k: np.asarray(v) for k, v in W.items()})


def spot_key_of(line: str, seats: Sequence[str]) -> tuple:
    """(pot type, aggressor in position, aggressor's seat, caller's seat) of a flop library spot: its preflop line
    as the CLI writes it ('UTG:f,HJ:r2.5,...') and its seats [oop, ip]."""
    acts = [a.split(":") for a in line.split(",") if a]
    raisers = [p for p, a in acts if a.startswith("r")]
    aggr = raisers[-1]
    caller = seats[0] if seats[1] == aggr else seats[1]
    return (min(len(raisers), 3), aggr == seats[1], aggr, caller)


def _edge_offset(x_lib: np.ndarray, x_mod: np.ndarray, rng: np.ndarray, eq: np.ndarray, share: float = 0.3) -> float:
    """The mean gap between the library's and the model's excess over the range's weakest classes (the bottom
    `share` of its combos by equity against the opponent): where the range ends and the model takes over."""
    idx = np.nonzero(rng > 0)[0]
    if len(idx) == 0:
        return 0.0
    order = idx[np.argsort(eq[idx])]
    cum = np.cumsum(rng[order])
    sel = order[: max(1, int(np.searchsorted(cum, share * cum[-1])) + 1)]
    w = rng[sel]
    return float((w * (x_lib[sel] - x_mod[sel])).sum() / w.sum())


class LibraryModel(EQRModel):
    """Pot shares measured by the flop library (gto/pf_libvals.py, scripts/pf_library_values.py): per stored spot each
    hand class's mean share of the pot over the spot's flops, against the range it met there, for both players.
    Per spot the shares become pairs S[i, j] = eq[i, j] + a[i] - c[j] (the aggressor's class i against the caller's
    class j: all-in equity plus each side's excess; a constant-sum pair), a and c solved so each class's share against
    the other player's range is the library's. A class outside the spot's range has no library value: its excess
    is the measured model's (`fallback`, measured_model(): every class measured at a tiny weight, so it has
    counterfactual values), shifted by the gap between the two models over the range's weakest classes (where it
    leaves off). `river_fix` (per class, realization) adds advisor._river_fix's correction times the equity, scaled
    per pot type by `river_scale`.
    A flop between seats with no spot of its own takes the stored spot of the same pot type and position with the
    same caller, else the same aggressor, else any; a pot type without any (4bet pots) the fallback model.
    `scale` {pot type: k} plays another stack depth: every share's excess over equity times k (depth_scale: the
    excess measured at that depth against 100bb's on the same flops, scripts/pf_depth_values.py)."""
    by_spot = True

    def __init__(self, spots: Optional[dict] = None, fallback: Optional[EQRModel] = None,
                 river_fix: Optional[np.ndarray] = None, river_scale: Optional[Dict[int, float]] = None,
                 scale: Optional[Dict[int, float]] = None):
        super().__init__(share_fn=self._pick)
        self.scale = scale or {}
        if spots is None:
            from .pf_libvals import SPOTS as spots
        self.fallback = fallback or measured_model() or flat_model()
        self.river_fix, self.river_scale = river_fix, river_scale or {1: 1.0, 2: 0.0}
        self.spots = {}
        for sid, v in spots.items():
            self.spots[spot_key_of(v["line"], v["seats"])] = (sid, v)
        self._S: Dict[tuple, np.ndarray] = {}
        self.fit_info: Dict[str, dict] = {}

    def match(self, key: tuple):
        if key in self.spots:
            return key
        for test in (lambda k: k[3] == key[3], lambda k: k[2] == key[2], lambda k: True):
            for k in self.spots:
                if k[:2] == key[:2] and test(k):
                    return k
        return None

    def _pick(self, key):
        k = self.match(key) if len(key) >= 4 else None
        if k is None:
            S = self.fallback.share(key[:2])
        else:
            if k not in self._S:
                self._S[k] = self._fit(k)
            S = self._S[k]
        x = self.scale.get(key[0], 1.0)
        if x != 1.0:
            eq = equity_matrix()
            S = np.clip(eq + x * (S - eq), -3.0, 4.0)
        return S

    def _fit(self, k: tuple) -> np.ndarray:
        sid, v = self.spots[k]
        eq = equity_matrix()
        aggr_side = "ip" if k[1] else "oop"
        call_side = "oop" if k[1] else "ip"
        ma, mc = np.asarray(v[aggr_side]["share"]), np.asarray(v[call_side]["share"])
        ra, rc = np.asarray(v[aggr_side]["range"], float), np.asarray(v[call_side]["range"], float)
        ina, inc = np.asarray(v[aggr_side]["w"]) > 0, np.asarray(v[call_side]["w"]) > 0
        P = PAIR_COMPAT * rc[None, :]                      # aggressor class i against the caller's range
        Q = PAIR_COMPAT * ra[:, None]                      # caller class j against the aggressor's range
        Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
        Qn = Q / np.maximum(Q.sum(0, keepdims=True), 1e-12)
        ea = (Pn * eq).sum(1)                              # aggressor's equity against the caller's range
        ec = (Qn * (1 - eq)).sum(0)                        # caller's equity against the aggressor's range
        if self.river_fix is not None:
            sc = self.river_scale.get(k[0], 0.0)
            ma = ma + sc * self.river_fix * ea
            mc = mc + sc * self.river_fix * ec
        # the fallback model's excess against the same ranges (every class)
        Sm = self.fallback.share(k[:2])
        xa_m = (Pn * (Sm - eq)).sum(1)
        xc_m = (Qn * (eq - Sm)).sum(0)
        a = np.where(ina, ma - ea, xa_m)
        c = np.zeros(K)
        da = dc = 0.0
        with np.errstate(all="ignore"):     # macOS Accelerate's matmul raises spurious warnings
            a, c, da, dc = self._solve_ac(a, c, ma, mc, ea, ec, xa_m, xc_m, Pn, Qn, ra, rc, ina, inc)
        S = np.clip(eq + a[:, None] - c[None, :], -3.0, 4.0)
        fa, fc = (Pn * S).sum(1), (Qn * (1 - S)).sum(0)
        self.fit_info[sid] = {"key": k, "resid_aggr": float(np.abs(fa - ma)[ina].max()),
                              "resid_caller": float(np.abs(fc - mc)[inc].max()),
                              "edge_aggr": da, "edge_caller": dc,
                              "realize_aggr": float((ra * ina * fa).sum() / max((ra * ina * ea).sum(), 1e-12)),
                              "realize_caller": float((rc * inc * fc).sum() / max((rc * inc * ec).sum(), 1e-12))}
        return S

    @staticmethod
    def _solve_ac(a, c, ma, mc, ea, ec, xa_m, xc_m, Pn, Qn, ra, rc, ina, inc):
        """Gauss-Seidel on the two sides' class excesses (see LibraryModel)."""
        da = dc = 0.0
        for _ in range(500):
            cbar = Pn @ c
            a_in = ma - ea + cbar
            abar = Qn.T @ np.where(ina, a_in, a)
            c_in = mc - ec + abar
            # outside the ranges: the model's excess (relative to the other side's mean), joined at the range's edge
            da = _edge_offset(a_in - cbar, xa_m, ra * ina, ea)
            dc = _edge_offset(c_in - abar, xc_m, rc * inc, ec)
            a_new = np.where(ina, a_in, xa_m + da + cbar)
            c_new = np.where(inc, c_in, xc_m + dc + abar)
            g = float((rc * c_new).sum() / max(rc.sum(), 1e-12))   # (a + g, c + g is the same pair: fix g)
            a_new, c_new = a_new - g, c_new - g
            done = max(np.abs(a_new - a).max(), np.abs(c_new - c).max()) < 1e-10
            a, c = a_new, c_new
            if done:
                break
        return a, c, da, dc


def library_model(**kw) -> Optional[EQRModel]:
    """The flop library's model (LibraryModel), or None without gto/pf_libvals.py."""
    try:
        return LibraryModel(**kw)
    except ImportError:
        return None


def realization(model: EQRModel, n_raises: int, aggr_ip: bool, role: str, opp: np.ndarray) -> np.ndarray:
    """Per class: pot share over all-in equity after a call that sees a flop, against the opponent's class
    weights `opp` (role "caller": the hand called the last raise; "aggr": its raise was called)."""
    S = model.share((min(n_raises, 3), aggr_ip))
    E = equity_matrix()
    with np.errstate(all="ignore"):
        if role == "aggr":
            W = PAIR_COMPAT * opp[None, :]
            return (S * W).sum(1) / np.maximum((E * W).sum(1), 1e-12)
        W = PAIR_COMPAT * opp[:, None]
        return ((1 - S) * W).sum(0) / np.maximum(((1 - E) * W).sum(0), 1e-12)


def flat_model() -> EQRModel:
    """Everyone realizes its equity (no postflop edge): a baseline."""
    return EQRModel(lambda key, role: np.zeros((K, K)))


# Three-way pots. MW_CORR[i] scales the product of hero's heads-up equities against the two opponents into
# its three-way equity (hands that beat one opponent tend to beat the other); see three_way_corr().
# MW_REALIZE: pot share over equity by postflop seat among the three (first, middle, last to act), by where the
# aggressor (the opener) sits among them. Measured with the 3-player solver (scripts/pf_three_way.py: the charts'
# ranges, 6 stratified flops a spot, flop and turn bet, the river dealt out; shares and equities each scaled to add
# to 1): BTN opens and both blinds call 0.95 / 0.91 / 1.11; CO opens, BTN and BB call 0.88 / 1.05 / 1.05; UTG
# opens, CO and BB call 0.87 / 0.97 / 1.14 (the two averaged for "middle"). It was a guess by seat alone, 0.88 /
# 0.97 / 1.12, which the measurement mostly keeps.
MW_REALIZE = {"last": (0.95, 0.91, 1.11), "middle": (0.875, 1.01, 1.095)}
LIMP_KEY = (1, True)        # the realization model's key a limped heads-up pot is played with


@lru_cache(maxsize=1)
def three_way_corr() -> np.ndarray:
    """Per class: its three-way all-in equity against two opponents holding the top 30% of hands, over the
    product of its heads-up equities against each (from the cached table in pf_game_data, or 1)."""
    try:
        from .pf_game_data import MW_CORR
        return np.array(MW_CORR)
    except ImportError:
        return np.full(K, 4.0 / 3.0)


class Game:
    """The tree flattened into arrays (nodes in breadth-first order) and solved level by level: every
    iteration is a walk down (reaches) and one back up (values, regrets), each a few numpy calls a depth
    rather than a Python call a node."""

    def __init__(self, root: Node, positions: Sequence[str], model: EQRModel, stack: float = 100.0,
                 rake: Optional[Tuple[float, Optional[float]]] = None, root_reach: Optional[np.ndarray] = None):
        """`root` may be a node inside a full tree (a subgame): `root_reach` (n, K) then gives each seat's class
        weights arriving there (the class prior times its actions so far), e.g. from the charts."""
        self.root, self.positions, self.model, self.stack, self.rake = root, list(positions), model, stack, rake
        self.root_reach = np.tile(PRIOR, (len(self.positions), 1)) if root_reach is None else np.asarray(root_reach, float)
        n = self.n = len(self.positions)
        self.post_rank = post_rank(self.positions)               # (heads-up the SB has position)
        # breadth-first order
        nodes, parent, act, depth = [root], [-1], [-1], [0]
        i = 0
        while i < len(nodes):
            for a, c in enumerate(nodes[i].children):
                nodes.append(c); parent.append(i); act.append(a); depth.append(depth[i] + 1)
            i += 1
        self.nodes_all = nodes
        for k, nd in enumerate(nodes):
            nd.index = k
        N = self.N = len(nodes)
        self.parent, self.act, self.depth = np.array(parent), np.array(act), np.array(depth)
        self.dec = np.array([k for k, nd in enumerate(nodes) if nd.kind == "dec"])
        self.dec_of = -np.ones(N, dtype=int)
        self.dec_of[self.dec] = np.arange(len(self.dec))
        D = len(self.dec)
        self.A = 3
        self.player = np.array([nodes[k].player for k in self.dec])
        self.ch = np.full((D, self.A), N, dtype=int)            # N: a dummy row of zeros
        for d, k in enumerate(self.dec):
            for a, c in enumerate(nodes[k].children):
                self.ch[d, a] = c.index
        self.valid = self.ch < N
        self.regret = np.zeros((D, self.A, K))
        self.ssum = np.zeros((D, self.A, K))
        self.maxd = int(self.depth.max())
        self.by_depth = [np.nonzero(self.depth == d)[0] for d in range(self.maxd + 1)]
        self.dec_by_depth = [self.dec_of[x][self.dec_of[x] >= 0] for x in self.by_depth]
        self._prep_terminals()

    # -------------------------------------------------------------- terminals
    def _pot_eff(self, pot: float, behind: float) -> float:
        if not self.rake:
            return pot
        from .advisor import RAKE_GROWTH
        rate, cap = self.rake
        r = rate * min(RAKE_GROWTH * pot, pot + 2 * behind)
        return pot - (r if cap is None else min(r, cap))

    def _prep_terminals(self):
        nodes, n = self.nodes_all, self.n
        terms = [k for k, nd in enumerate(nodes) if nd.kind != "dec"]
        self.terms = np.array(terms)
        self.inv = np.array([nodes[k].inv for k in terms])                   # (T, n)
        pot = self.inv.sum(1)
        self.t_active = np.array([[q in nodes[k].active for q in range(n)] for k in terms])
        is_fold = np.array([nodes[k].kind == "fold" for k in terms])
        pay = -self.inv.copy()
        for r, k in enumerate(terms):
            if nodes[k].kind == "fold":
                pay[r, nodes[k].active[0]] += pot[r]
        self.pay = pay
        self.f_rows = np.nonzero(is_fold)[0]
        eq = equity_matrix()
        groups: Dict[tuple, list] = {}
        three = []
        for r, k in enumerate(terms):
            nd = nodes[k]
            if nd.kind == "fold":
                continue
            act = list(nd.active)
            behind = self.stack - float(self.inv[r].max())
            pe = self._pot_eff(float(pot[r]), behind)
            if len(act) == 2:
                if nd.allin:
                    key, (a, c) = ("allin",), act
                elif nd.n_raises == 0:
                    # a limped pot: played as a single-raised one with the player in position as the aggressor
                    # (LIMP_KEY; nothing measured for it)
                    a = max(act, key=lambda q: self.post_rank[q])
                    c = act[0] if act[1] == a else act[1]
                    key = LIMP_KEY + (self.positions[a], self.positions[c])
                else:
                    a = nd.aggressor
                    c = act[0] if act[1] == a else act[1]
                    key = (min(nd.n_raises, 3), self.post_rank[a] > self.post_rank[c],
                           self.positions[a], self.positions[c])
                groups.setdefault(key, []).append((r, a, c, pe))
            else:
                ranks = sorted(act, key=lambda q: self.post_rank[q])
                place = "last" if nd.aggressor == ranks[-1] else "middle"
                fac = [1.0 if nd.allin else MW_REALIZE[place][0 if j == 0 else 2 if j == len(act) - 1 else 1]
                       for j in range(len(act))]
                three.append((r, ranks[:3], fac[:3], pe))
        self.hu = []
        for key, rows in groups.items():
            S = PAIR_COMPAT * (eq if key == ("allin",) else self.model.share(key))
            R = np.array(rows)
            self.hu.append((key, S, R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2].astype(int), R[:, 3]))
        self.three = None
        if three:
            self.three = (np.array([x[0] for x in three]), np.array([x[1] for x in three]),
                          np.array([x[2] for x in three]), np.array([x[3] for x in three]))
        self.s_rows = np.nonzero(~is_fold)[0]

    def set_model(self, model: EQRModel):
        self.model = model
        self._prep_terminals()

    def _term_values(self, reach: np.ndarray, V: np.ndarray):
        """Fill V's terminal rows: each seat's counterfactual value (payoff summed over the other seats'
        classes weighted by their reaches; card removal by class pairs against each opponent)."""
        T = self.terms
        RT = reach[T]                                            # (T, n, K)
        m = RT @ PAIR_COMPAT                                     # PAIR_COMPAT is symmetric
        n = self.n
        ones = np.ones((len(T), 1, K))
        pre = np.cumprod(np.concatenate([ones, m[:, :-1]], 1), 1)
        suf = np.cumprod(np.concatenate([ones, m[:, ::-1][:, :-1]], 1), 1)[:, ::-1]
        others = pre * suf                                       # product over the other seats
        out = self.pay[:, :, None] * others                      # folds; folded seats at a showdown
        fold_f = np.prod(np.where(self.t_active[:, :, None], 1.0, m), axis=1)   # (T, K) folded seats' mass
        for key, S, rows, a, c, pe in self.hu:
            ra, rc = RT[rows, a], RT[rows, c]
            ma, mc = m[rows, a], m[rows, c]
            inv_a, inv_c = self.inv[rows, a][:, None], self.inv[rows, c][:, None]
            ff = fold_f[rows]
            out[rows, a] = (pe[:, None] * (rc @ S.T) - inv_a * mc) * ff
            out[rows, c] = (pe[:, None] * (ma - ra @ S) - inv_c * ma) * ff
        if self.three is not None:
            rows, act, fac, pe = self.three
            E = PAIR_COMPAT * equity_matrix()
            corr = three_way_corr()
            er = np.stack([RT[rows, act[:, k]] @ E.T for k in range(3)], 1)      # (T3, 3, K)
            ms = np.stack([m[rows, act[:, k]] for k in range(3)], 1)
            ff = fold_f[rows]
            shares, masses = [], []
            for k in range(3):
                o = [j for j in range(3) if j != k]
                shares.append(corr * er[:, o[0]] * er[:, o[1]])
                masses.append(ms[:, o[0]] * ms[:, o[1]])
            # the three shares of the pot, normalised to add to 1 over the three ranges (the product form
            # with MW_CORR is only near it), then each by its seat's realization
            own = [RT[rows, act[:, k]] for k in range(3)]
            tot = sum((own[k] * shares[k]).sum(1) for k in range(3))
            mass = sum((own[k] * masses[k]).sum(1) for k in range(3)) / 3
            norm = np.where(tot > 1e-300, mass / np.maximum(tot, 1e-300), 1.0)[:, None]
            for k in range(3):
                inv = self.inv[rows, act[:, k]][:, None]
                share = shares[k] * norm * fac[:, k][:, None]
                out[rows, act[:, k]] = (pe[:, None] * share - inv * masses[k]) * ff
        V[T] = out

    # -------------------------------------------------------------- CFR
    def _strat(self, regret: np.ndarray) -> np.ndarray:
        pos = np.maximum(regret, 0.0) * self.valid[:, :, None]
        tot = pos.sum(1, keepdims=True)
        uni = self.valid[:, :, None] / self.valid.sum(1)[:, None, None]
        return np.where(tot > 1e-12, pos / np.maximum(tot, 1e-12), uni)

    def current(self) -> np.ndarray:
        return self._strat(self.regret)

    def average(self) -> np.ndarray:
        tot = self.ssum.sum(1, keepdims=True)
        uni = self.valid[:, :, None] / self.valid.sum(1)[:, None, None]
        return np.where(tot > 1e-12, self.ssum / np.maximum(tot, 1e-12), uni)

    def _down(self, strat: np.ndarray, skip: int = -1) -> np.ndarray:
        """Reaches (N, n, K): the class prior times each seat's action probabilities (seat `skip`'s own
        left out: a best response's)."""
        reach = np.empty((self.N, self.n, K))
        reach[0] = self.root_reach
        for d in range(1, self.maxd + 1):
            idx = self.by_depth[d]
            par = self.parent[idx]
            reach[idx] = reach[par]
            pd = self.dec_of[par]
            pl = self.player[pd]
            f = strat[pd, self.act[idx]]
            if skip >= 0:
                f = np.where((pl == skip)[:, None], 1.0, f)
            reach[idx, pl] *= f
        return reach

    def _up(self, reach, strat, t=0, update=False, br=-1) -> np.ndarray:
        V = np.zeros((self.N + 1, self.n, K))
        with np.errstate(all="ignore"):     # macOS Accelerate's matmul raises spurious warnings
            self._term_values(reach, V)
        for d in range(self.maxd, -1, -1):
            g = self.dec_by_depth[d]
            if len(g) == 0:
                continue
            nodes = self.dec[g]
            Vc = V[self.ch[g]]                                   # (G, A, n, K)
            p = self.player[g]
            G = np.arange(len(g))
            vp = Vc[G, :, p]                                     # (G, A, K)
            out = Vc.sum(1)
            if br >= 0:
                mine = p == br
                best = np.where(self.valid[g][:, :, None], vp, -np.inf).max(1)
                out[G, p] = np.where(mine[:, None], best, (strat[g] * vp).sum(1))
            else:
                out[G, p] = (strat[g] * vp).sum(1)
            if update:
                self.regret[g] += (vp - out[G, p][:, None, :]) * self.valid[g][:, :, None]
                self.ssum[g] += t * reach[nodes, p][:, None, :] * strat[g]
            V[nodes] = out
        return V

    def iterate(self, t: int):
        s = self.current()
        reach = self._down(s)
        self._up(reach, s, t, update=True)
        # discounted CFR: positive regrets by t^1.5 / (t^1.5 + 1), negative halved
        pa = t ** 1.5 / (t ** 1.5 + 1)
        self.regret *= np.where(self.regret > 0, pa, 0.5)

    def solve(self, iters: int = 1000, verbose: bool = False, every: int = 100, start: int = 1):
        for t in range(start, start + iters):
            self.iterate(t)
            if verbose and t % every == 0:
                print(f"iter {t}: exploitability {self.exploitability():.4f} bb/hand", flush=True)
        return self

    def values(self, strat=None) -> np.ndarray:
        """Each seat's expected value, bb per hand, under `strat` (default the average strategy)."""
        s = self.average() if strat is None else strat
        V = self._up(self._down(s), s)
        return (V[0] * self.root_reach).sum(1)

    def exploitability(self, per_seat: bool = False):
        """What each seat gains by best-responding to the others' average strategies, bb per hand (summed). Both
        values on the best response's reaches (the seat's own left out): a three-way flop's shares are normalised
        over all three players' reaches (_term_values), so values() differs from them by up to ~0.05 bb a seat."""
        s = self.average()
        gains = []
        for p in range(self.n):
            reach = self._down(s, skip=p)
            V0, V = self._up(reach, s), self._up(reach, s, br=p)
            gains.append(float(((V[0, p] - V0[0, p]) * self.root_reach[p]).sum()))
        return gains if per_seat else sum(gains)

    # -------------------------------------------------------------- reading the result
    def find(self, line: Sequence[Tuple[str, str]]) -> Node:
        """The node after `line`, [(position, action)], everyone skipped in turn folding."""
        node = self.root
        for posn, act in line:
            while node.kind == "dec" and self.positions[node.player] != posn:
                node = node.children[node.names.index("fold")]
            node = node.children[node.names.index(act)]
        return node

    def strategy_at(self, node: Node) -> Dict[str, np.ndarray]:
        """The average strategy (per class) of the player to act at `node`, by action name."""
        s = self.average()[self.dec_of[node.index]]
        return {nm: s[a] for a, nm in enumerate(node.names)}

    def reach(self) -> np.ndarray:
        return self._down(self.average())

    def save(self, path: str):
        np.savez_compressed(path, regret=self.regret, ssum=self.ssum)

    def load(self, path: str):
        z = np.load(path)
        self.regret, self.ssum = z["regret"], z["ssum"]
