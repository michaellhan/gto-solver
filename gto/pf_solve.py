"""Preflop subgame solver: the 3bet pot heads-up, from the opener facing the 3bet down to the 5bet jam.

    opener (O) facing the 3bet:      fold | call (see a flop) | 4bet
      3bettor (T) facing the 4bet:   fold | call (see a flop) | jam
        opener facing the jam:       fold | call

Hands are the 169 classes, with class-level card removal (pf_classes.PAIR_COMPAT). An all-in pays the
hand's equity; a call that sees a flop pays its equity scaled by how much of it each player realizes
(in position more than out of position, more at a low stack-to-pot), normalised so the two shares add
to 1. That keeps the game constant-sum, so CFR+ converges to an equilibrium of this model.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple
import numpy as np
from .pf_classes import CLASSES, CLASS_OF, PAIR_COMPAT, equity_matrix, to_classes

K = len(CLASSES)
# Share of its equity a hand realizes after a call sees a flop: a base by position and pot type, more
# when it is ahead (strong hands get paid, dominated ones pay off or fold out), more when suited or
# connected (they flop draws and play well), less offsuit and unconnected.
REALIZE = {"3bet": {"ip": 1.0, "oop": 0.80}, "4bet": {"ip": 1.0, "oop": 0.90}}
# per unit of equity below 50% (a dominated hand pays off or folds out) and above it (a strong hand gets
# paid); little above in a 4bet pot, where at a stack-to-pot near 2 the stacks mostly go in and a bigger
# bonus would let a call beat a jam and trap
REALIZE_SLOPE = {"3bet": (1.2, 0.3), "4bet": (3.5, 0.0)}
PLAY_SUITED = 0.08
PLAY_CONNECTED = 0.04    # gap of 0 or 1 between the ranks
PLAY_OFFSUIT = -0.06     # offsuit non-pairs


def _playability() -> np.ndarray:
    rk = "23456789TJQKA"
    out = np.zeros(K)
    for i, c in enumerate(CLASSES):
        if len(c) == 2:
            continue
        gap = abs(rk.index(c[0]) - rk.index(c[1])) - 1
        out[i] = (PLAY_SUITED if c[2] == "s" else PLAY_OFFSUIT) + (PLAY_CONNECTED if gap <= 1 else 0.0)
    return out


def _realize(eq: np.ndarray, base: float, slope: Tuple[float, float], play: float = 1.0) -> np.ndarray:
    """(K, K) realization of the row hand against the column hand."""
    d = eq - 0.5
    return np.clip(base + slope[0] * np.minimum(d, 0.0) + slope[1] * np.maximum(d, 0.0)
                   + play * _playability()[:, None], 0.2, 1.6)


@dataclass
class Spot:
    pot: float          # chips in before the opener acts on the 3bet (blinds, open, 3bet, dead money)
    open_in: float      # the opener's chips in (its open)
    three: float        # the 3bet, to
    four: float         # the 4bet, to
    stack: float        # effective stack
    opener_ip: bool     # the opener has position postflop

    def key(self) -> tuple:
        return tuple(round(x, 2) for x in (self.pot, self.open_in, self.three, self.four, self.stack)) + (self.opener_ip,)


# How a call that sees a flop is paid:
#   "param": the parametric model above (fitted to the charts' continue rates)
#   "table": the excess pot shares measured with the postflop solver (gto/pf_realize.py, from
#            scripts/build_realization.py) per class against each opponent strength bucket, by pot
#            type, role (caller, or the aggressor whose bet was called) and position. Too noisy
#            alone for close calls: the tables from the two halves of the flops put the opener's calls
#            against a 3bet 10-22 points apart in four of six spots (tests TestRealizationTable).
#   "model": a smooth model of those measurements (gto/pf_realize_model.py, from
#            scripts/fit_realization.py): a few coefficients per pot type, role and position on the hand's
#            equity against the bucket and its shape (pair, suited, connected), which averages the noise.
MODE = "param"
TABLE_OVERRIDE = None   # a table to use instead of gto/pf_realize.py's (tests: the two halves of the flops)
MODEL_OVERRIDE = None   # coefficients to use instead of gto/pf_realize_model.py's


TABLE_PRIOR = 0.5  # the parametric model counts as this much of a typical entry's measurement
# Opponent strength buckets for the table: hands ranked by all-in equity against a random hand, cut at
# these shares of all combos (finer at the top, where 3bet and 4bet ranges live: AA, KK, QQ-JJ, ...)
BUCKET_EDGES = (0.005, 0.01, 0.02, 0.035, 0.06, 0.10, 0.18, 0.35)


@lru_cache(maxsize=1)
def bucket_of() -> np.ndarray:
    """The strength bucket (0 = strongest) of each of the 169 classes."""
    from .pf_classes import N_IN, PAIR_COMPAT
    strength = (equity_matrix() * PAIR_COMPAT * N_IN[None, :]).sum(1) / (PAIR_COMPAT * N_IN[None, :]).sum(1)
    order = np.argsort(-strength)
    cum = np.cumsum(N_IN[order]) / N_IN.sum()
    out = np.empty(K, dtype=int)
    out[order] = np.searchsorted(np.array(BUCKET_EDGES), cum - N_IN[order] / N_IN.sum() / 2)   # by each class's midpoint
    return out


def _table(pot_type: str, pos: str):
    """Per class and opponent strength bucket, the measured excess pot share and the weight it rests on
    (0 where unmeasured), or None. `pos`: role and position, e.g. "caller_ip"."""
    table = TABLE_OVERRIDE
    if table is None:
        try:
            from .pf_realize import TABLE as table
        except ImportError:
            return None
    rows = table[pot_type][pos]
    d = np.array([[0.0 if v is None else v[0] for v in row] for row in rows])
    w = np.array([[0.0 if v is None else v[1] for v in row] for row in rows])
    return d, w


@lru_cache(maxsize=1)
def _bucket_equity() -> np.ndarray:
    """(K, buckets): each class's all-in equity against all the combos of each strength bucket."""
    from .pf_classes import N_IN
    b = bucket_of()
    W = PAIR_COMPAT * N_IN[None, :]
    return np.stack([(equity_matrix() * W * (b == k)).sum(1) / (W * (b == k)).sum(1)
                     for k in range(int(b.max()) + 1)], 1)


@lru_cache(maxsize=1)
def _shape() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per class: pair, suited, connected (ranks at most two apart; not pairs)."""
    rk = "23456789TJQKA"
    pair = np.array([len(c) == 2 for c in CLASSES], float)
    suited = np.array([len(c) == 3 and c[2] == "s" for c in CLASSES], float)
    conn = np.array([len(c) == 3 and abs(rk.index(c[0]) - rk.index(c[1])) <= 2 for c in CLASSES], float)
    return pair, suited, conn


MODEL_FEATURES = ("1", "x", "x|x|", "pair", "suited", "connected", "pair*x", "suited*x")


def model_features(e: np.ndarray) -> np.ndarray:
    """(K, buckets, features) for the excess-share model, from equity e (K, buckets) against each bucket;
    x = e - 0.5."""
    pair, suited, conn = (v[:, None] for v in _shape())
    x = e - 0.5
    one = np.ones_like(x)
    return np.stack([one, x, x * np.abs(x), pair * one, suited * one, conn * one, pair * x, suited * x], -1)


# Added to the model's realization (pot share over equity) per pot type, role and position: the excess
# gains shift x equity. Public EQR figures (solver reports: the in-position aggressor of a single-raised
# pot realizes 1.14-1.17, the out-of-position caller 0.78-0.87; one 3bet-pot flop, BTN 3bet vs CO, 1.17
# / 0.83) match what the postflop solver measures for those roles (3bet pots: aggressor in position
# 1.15, caller out of position 0.80). None is published for an in-position caller of a 3bet; the
# measured 0.97 sits below the 1.05-1.15 rules of thumb for in-position flats, and shifting it to 1.15
# brings the in-position openers' calls to the charts (published solves): BTN vs BB 41% against 34%,
# CO vs BB 31% against 41%, BTN vs SB 38% against 36% (unshifted 28%, 25%, 26%).
MODEL_SHIFT = {"3bet": {"caller_ip": 0.18}}


def _model(pot_type: str, pos: str):
    """(K, buckets): the smooth model's excess pot share per class against each bucket, or None."""
    coef = MODEL_OVERRIDE
    if coef is None:
        try:
            from .pf_realize_model import COEF as coef
        except ImportError:
            return None
    e = _bucket_equity()
    shift = MODEL_SHIFT.get(pot_type, {}).get(pos, 0.0)
    return np.clip(model_features(e) @ np.array(coef[pot_type][pos]) + shift * e, -2.0, 2.0)


def _param_shares(eq: np.ndarray, r_o: float, r_t: float, pot_type: str) -> np.ndarray:
    """The parametric model: the opener's share from all-in equity and each side's realization,
    normalised so the two shares add to 1."""
    slope, play = REALIZE_SLOPE[pot_type], 1.0 if pot_type == "3bet" else 0.5
    a = eq * _realize(eq, r_o, slope, play)
    b = (1.0 - eq) * _realize(1.0 - eq.T, r_t, slope, play).T
    return a / np.maximum(a + b, 1e-12)


def _shares(eq: np.ndarray, r_o: float, r_t: float, pot_type: str, opener_ip: bool) -> np.ndarray:
    """The opener's pot share after a call that sees a flop (EV over the pot, later bets included), by
    MODE. With the table or the model: each side's excess share over its equity against the other's
    strength bucket, the two combined so the shares add to 1 (the table's thin entries blended with the
    parametric model by the weight behind them)."""
    param = _param_shares(eq, r_o, r_t, pot_type)
    # the opener called the 3bet, or had its 4bet called; the 3bettor the other way round
    ro_, rt_ = ("caller", "aggr") if pot_type == "3bet" else ("aggr", "caller")
    po, pt = (f"{ro_}_ip", f"{rt_}_oop") if opener_ip else (f"{ro_}_oop", f"{rt_}_ip")
    b = bucket_of()
    if MODE == "model":
        mo, mt = _model(pot_type, po), _model(pot_type, pt)
        if mo is None or mt is None:
            return param
        return eq + (mo[:, b] - mt[:, b].T) / 2
    to, tt = (_table(pot_type, po), _table(pot_type, pt)) if MODE == "table" else (None, None)
    if to is None or tt is None:
        return param
    pd = param - eq                                   # the parametric model's excess, opener's view
    d_o, w_o = to[0][:, b], to[1][:, b]               # opener class i against 3bettor j's bucket
    d_t, w_t = tt[0][:, b].T, tt[1][:, b].T           # 3bettor class j against opener i's bucket
    d_o = (w_o * d_o + TABLE_PRIOR * pd) / (w_o + TABLE_PRIOR)
    d_t = (w_t * d_t - TABLE_PRIOR * pd) / (w_t + TABLE_PRIOR)
    return eq + (d_o - d_t) / 2


def _payoffs(spot: Spot) -> Dict[str, np.ndarray]:
    """The opener's payoff (K, K: opener class x 3bettor class) at each terminal, in chips relative to
    folding to the 3bet. The 3bettor's is the pot at the root minus it."""
    eq = equity_matrix()
    s = spot
    ro3, rt3 = (REALIZE["3bet"]["ip"], REALIZE["3bet"]["oop"]) if s.opener_ip else (REALIZE["3bet"]["oop"], REALIZE["3bet"]["ip"])
    ro4, rt4 = (REALIZE["4bet"]["ip"], REALIZE["4bet"]["oop"]) if s.opener_ip else (REALIZE["4bet"]["oop"], REALIZE["4bet"]["ip"])
    call3, add4 = s.three - s.open_in, s.four - s.open_in
    return {
        "call3": _shares(eq, ro3, rt3, "3bet", s.opener_ip) * (s.pot + call3) - call3,
        "t_fold": np.full((K, K), s.pot),
        "t_call": _shares(eq, ro4, rt4, "4bet", s.opener_ip) * (s.pot + add4 + s.four - s.three) - add4,
        "o_fold_jam": np.full((K, K), -add4),
        "o_call_jam": eq * (s.pot + (s.stack - s.open_in) + (s.stack - s.three)) - (s.stack - s.open_in),
    }


def _values(U, w_o, w_t, pot, so0, st1, so2):
    """Counterfactual values per class: the opener's at the root (v0) and facing the jam (v2), the
    3bettor's facing the 4bet (v1), for the current strategies."""
    M = PAIR_COMPAT
    ro4 = w_o * so0[2]
    rt_jam = w_t * st1[2]
    v2 = np.stack([(M * U["o_fold_jam"]) @ rt_jam, (M * U["o_call_jam"]) @ rt_jam])
    u_jam = U["o_fold_jam"] * so2[0][:, None] + U["o_call_jam"] * so2[1][:, None]
    tot = M.T @ ro4
    v1 = np.stack([pot * tot - (M * U["t_fold"]).T @ ro4, pot * tot - (M * U["t_call"]).T @ ro4,
                   pot * tot - (M * u_jam).T @ ro4])
    u_4 = U["t_fold"] * st1[0][None, :] + U["t_call"] * st1[1][None, :] + u_jam * st1[2][None, :]
    v0 = np.stack([np.zeros(K), (M * U["call3"]) @ w_t, (M * u_4) @ w_t])
    return v0, v1, v2


def _strat(reg):
    pos = np.maximum(reg, 0.0)
    tot = pos.sum(0)
    return np.where(tot > 1e-12, pos / np.maximum(tot, 1e-12), 1.0 / reg.shape[0])


def _onehot(v):
    out = np.zeros_like(v)
    out[np.argmax(v, axis=0), np.arange(v.shape[1])] = 1.0
    return out


def solve(w_o: np.ndarray, w_t: np.ndarray, spot: Spot, iters: int = 1500) -> Dict[str, np.ndarray]:
    """Equilibrium of the subgame for class weights w_o (opener's range facing the 3bet) and w_t (the
    3bettor's 3bet range), by CFR+ with linear averaging. Returns per-class strategies o_fold/o_call/
    o_4bet, t_fold/t_call/t_jam (facing the 4bet), o_vs_jam (call frequency facing the jam), and
    `exploitability`: how much the two players gain on average by best-responding, % of the pot."""
    with np.errstate(all="ignore"):   # macOS Accelerate's matmul raises spurious warnings
        return _solve(w_o, w_t, spot, iters)


def _solve(w_o, w_t, spot, iters):
    U = _payoffs(spot)
    reg = [np.zeros((3, K)), np.zeros((3, K)), np.zeros((2, K))]
    avg = [np.zeros((3, K)), np.zeros((3, K)), np.zeros((2, K))]
    for t in range(1, iters + 1):
        so0, st1, so2 = (_strat(r) for r in reg)
        v0, v1, v2 = _values(U, w_o, w_t, spot.pot, so0, st1, so2)
        for k, (v, sk) in enumerate(((v0, so0), (v1, st1), (v2, so2))):
            reg[k] = np.maximum(reg[k] + v - (sk * v).sum(0), 0.0)
        avg[0] += t * w_o * so0
        avg[1] += t * w_t * st1
        avg[2] += t * w_o * so0[2] * so2
    norm = lambda a: np.where(a.sum(0) > 1e-12, a / np.maximum(a.sum(0), 1e-12), 1.0 / a.shape[0])
    o0, t1, o2 = (norm(a) for a in avg)
    return {"o_fold": o0[0], "o_call": o0[1], "o_4bet": o0[2],
            "t_fold": t1[0], "t_call": t1[1], "t_jam": t1[2], "o_vs_jam": o2[1],
            "exploitability": exploitability(U, w_o, w_t, spot.pot, o0, t1, o2)}


def exploitability(U, w_o, w_t, pot, o0, t1, o2) -> float:
    """Average of what each player gains by best-responding to the other, % of the pot at the root."""
    pairs = float(w_o @ PAIR_COMPAT @ w_t)
    ev = lambda so0, st1, so2: float(w_o @ (so0 * _values(U, w_o, w_t, pot, so0, st1, so2)[0]).sum(0)) / pairs
    base = ev(o0, t1, o2)
    # the opener best-responds facing the jam, then at the root
    v0, v1, v2 = _values(U, w_o, w_t, pot, o0, t1, o2)
    br2 = _onehot(v2)
    v0, _, _ = _values(U, w_o, w_t, pot, o0, t1, br2)
    br_o = ev(_onehot(v0), t1, br2)
    # the 3bettor best-responds facing the 4bet (minimising the opener's payoff)
    br_t = ev(o0, _onehot(v1), o2)
    return (br_o - br_t) / 2 / pot * 100
