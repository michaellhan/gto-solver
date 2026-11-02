"""Answers facing a bet or raise of a size the stored tree doesn't have, read off the stored decisions facing the
sizes it does (a library flop's entry: 33% and 75% bets, a 50% raise).

A decision facing an aggressive action is described by its price: C, what the player must put in to call, and T, the
pot once the action is made (before the call). x = C / (T - C) is the action's size as a share of the pot it went into
(a bet of 50% pot: 0.5; a raise, its share of the pot after calling, as the tree's raise sizes are given), and the
minimum defence frequency 1 / (1 + x) = 1 - C / T is how often the player must continue so that a bluff of that size
doesn't win at once.

Methods (each combo's mix over fold / call / raise / all-in, from the stored decisions' mixes and EVs):
  near    the stored decision at the nearest size (what the walk reads: solver.match_action);
  interp  between the two stored sizes either side of x, each combo's mix linear in log x; outside them the nearest;
  mdf     interp, then the range's continuing share moved to k(x) / (1 + x), k (the share over the minimum defence)
          from the stored sizes, linear in log x between them and as the nearest outside; the share removed (or added)
          by each combo's realized share of the pot when it continues, q = (EV + C) / (T + C) from the stored EVs
          (linear in log x), the lowest continuing hands folding first (the highest folding hands continuing first);
  prop    interp, then every combo's continuing share scaled to the same target;
  nearmdf the nearest's mix (and q), then mdf's move to the same target (k still linear in log x): between two sizes
          the nearest size's hands keep their own mix, and only the range's continuing share follows the price; outside
          them the same as mdf.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

KINDS = ("fold", "call", "raise", "allin")


def kind(name: str) -> str:
    n = name.split()[0]
    return {"fold": "fold", "call": "call", "check": "call", "all-in": "allin"}.get(n, "raise")


def by_kind(names: Sequence[str], strat: np.ndarray) -> Dict[str, np.ndarray]:
    """{kind: (N,)} the mix summed by action kind (a node's two raise sizes add up)."""
    out = {k: np.zeros(strat.shape[1]) for k in KINDS}
    for a, nm in enumerate(names):
        out[kind(nm)] = out[kind(nm)] + strat[a]
    return out


def onto(names: Sequence[str], mix: Dict[str, np.ndarray]) -> np.ndarray:
    """(A, N): a mix by kind laid onto a decision's actions: a kind it lacks goes to the nearest it has (all-in to its
    raise and back, a raise or all-in it can't make to calling), its raises split evenly."""
    ks = [kind(n) for n in names]
    n = len(next(iter(mix.values())))
    out = np.zeros((len(names), n))
    for k in KINDS:
        m = mix.get(k)
        if m is None or not np.any(m):
            continue
        tgt = [i for i, kk in enumerate(ks) if kk == k]
        if not tgt:
            alt = {"allin": ("raise", "call"), "raise": ("allin", "call"), "call": ("fold",), "fold": ("call",)}[k]
            for a in alt:
                tgt = [i for i, kk in enumerate(ks) if kk == a]
                if tgt:
                    break
        if not tgt:
            continue
        for i in tgt:
            out[i] += m / len(tgt)
    s = out.sum(axis=0, keepdims=True)
    return np.where(s > 1e-12, out / np.maximum(s, 1e-12), 1.0 / len(names))


def size_of(C: float, T: float) -> float:
    """x = C / (T - C): the action's size as a share of the pot it went into."""
    return C / max(T - C, 1e-9)


def mdf(x: float) -> float:
    return 1.0 / (1.0 + x)


def _weights(xs: Sequence[float], x: float):
    """[(index, weight)] of the stored sizes `xs` that x reads: the two either side, linear in log x; outside, the
    nearest alone."""
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    lx = np.log(max(x, 1e-6))
    if lx <= np.log(xs[o[0]]):
        return [(o[0], 1.0)]
    if lx >= np.log(xs[o[-1]]):
        return [(o[-1], 1.0)]
    for i, j in zip(o, o[1:]):
        a, b = np.log(xs[i]), np.log(xs[j])
        if a <= lx <= b:
            w = (lx - a) / max(b - a, 1e-12)
            return [(i, 1.0 - w), (j, w)]
    return [(o[-1], 1.0)]


def nearest(xs: Sequence[float], x: float) -> int:
    """The stored size the tree's walk reads x as (solver.match_action: nearest in pot share)."""
    return min(range(len(xs)), key=lambda i: abs(xs[i] - x))


def readable(xs: Sequence[float], taken: Sequence[bool], x: float) -> List[int]:
    """The sizes (indices into xs) whose decisions a read of x takes its mix from: of those `taken` (the bettor bets or
    raises them often enough for the decisions facing them to be trained), the one just below x and all above it; else
    every size. (A smaller size's play facing a bet more than one size bigger doesn't carry over: a river's 33%
    decision read for a 100% bet or lead gave up 4-10% of the pot.)"""
    below = [v for v in xs if v <= x]
    floor = max(below) if below else 0.0
    return [i for i, v in enumerate(xs) if taken[i] and v >= floor] or list(range(len(xs)))


def answer(sources: List[dict], C: float, T: float, method: str = "mdf", reach: Optional[np.ndarray] = None,
           mix_from: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Each combo's mix by kind ({kind: (N,)}) facing an action of price C into T, read off `sources`: the stored
    decisions facing other sizes of the same action, each {"names", "strat" (A, N), "evs" (A, N) chips relative to
    folding now, or None, "C", "T", "reach" (N,) the player's reach there}, over the same N combos (NaN columns: not
    held). `reach`: the player's reach facing this action (default the sources' first), for the range's continuing
    share. `mix_from`: the indices of the sources whose mixes (and q) are read, the nearest of them, or the two of them
    either side (default all); the range's continuing share still follows every source's (k)."""
    xs = [size_of(s["C"], s["T"]) for s in sources]
    x = size_of(C, T)
    mixes = [by_kind(s["names"], np.nan_to_num(s["strat"], nan=0.0)) for s in sources]
    held = np.all([~np.isnan(s["strat"][0]) for s in sources], axis=0)
    mi = list(mix_from) if mix_from else list(range(len(sources)))
    near = mi[nearest([xs[i] for i in mi], x)]
    if method == "near":
        return _masked(mixes[near], ~np.isnan(sources[near]["strat"][0]))
    ws = _weights(xs, x)
    # (the mix's and q's weights)
    wm = [(near, 1.0)] if method == "nearmdf" else [(mi[i], w) for i, w in _weights([xs[i] for i in mi], x)]
    mix = {k: sum(w * mixes[i][k] for i, w in wm) for k in KINDS}
    if method == "interp":
        return _masked(mix, held)
    r = np.nan_to_num(reach if reach is not None else sources[ws[0][0]]["reach"])
    r = np.where(held, r, 0.0)
    tot = max(float(r.sum()), 1e-12)
    cont = lambda m: float(((m["call"] + m["raise"] + m["allin"]) * r).sum() / tot)
    ks = [cont(_masked(m, held, 0.0)) / mdf(xs[i]) for i, m in enumerate(mixes)]
    k = sum(w * ks[i] for i, w in ws)
    target = min(1.0, k * mdf(x))
    now = cont(_masked(mix, held, 0.0))
    if now <= 1e-9:
        return _masked(mix, held)
    if method == "prop":
        return _masked(_scale(mix, target / now), held)
    if method not in ("mdf", "nearmdf"):
        raise ValueError(method)
    # each combo's realized share of the pot when it continues (its best continuing action's), linear in log x
    qs = []
    for s in sources:
        if s.get("evs") is None:
            return _masked(_scale(mix, target / now), held)
        ev = np.nan_to_num(s["evs"])
        ks_ = [kind(n) for n in s["names"]]
        best = np.max([ev[a] for a, kk in enumerate(ks_) if kk != "fold"], axis=0)
        qs.append((best + s["C"]) / (s["T"] + s["C"]))
    q = sum(w * qs[i] for i, w in wm)
    return _masked(_by_rank(mix, q, r, target - now), held)


def _scale(mix: Dict[str, np.ndarray], f: float) -> Dict[str, np.ndarray]:
    """Every combo's continuing share times f (folds take up the rest; over 1, folds shrink by the same share)."""
    c = mix["call"] + mix["raise"] + mix["allin"]
    if f <= 1.0:
        return {"fold": mix["fold"] + (1 - f) * c, "call": f * mix["call"], "raise": f * mix["raise"],
                "allin": f * mix["allin"]}
    # more continuing: each fold continues at the share that adds the range's (f - 1) x continuing, capped at all
    add = np.minimum(1.0, (f - 1.0) * np.where(mix["fold"] > 1e-12, c / np.maximum(mix["fold"], 1e-12), 0.0)) * mix["fold"]
    cs = np.maximum(c, 1e-12)
    share = {k: np.where(c > 1e-12, mix[k] / cs, 1.0 if k == "call" else 0.0) for k in ("call", "raise", "allin")}
    return {"fold": mix["fold"] - add, **{k: mix[k] + add * share[k] for k in share}}


def _by_rank(mix: Dict[str, np.ndarray], q: np.ndarray, r: np.ndarray, delta: float) -> Dict[str, np.ndarray]:
    """Moves `delta` of the range (reach r) between folding and continuing: negative, the continuing combos with the
    lowest q fold first (each its whole continuing share, the last in part); positive, the folding combos with the
    highest q continue first (calling)."""
    tot = max(float(r.sum()), 1e-12)
    out = {k: v.copy() for k, v in mix.items()}
    c = mix["call"] + mix["raise"] + mix["allin"]
    if delta < 0:
        need = -delta * tot
        for i in np.argsort(q):
            if need <= 1e-12:
                break
            m = c[i] * r[i]
            if m <= 0:
                continue
            f = min(1.0, need / m)
            for k in ("call", "raise", "allin"):
                out["fold"][i] += f * out[k][i]
                out[k][i] *= 1 - f
            need -= f * m
    else:
        need = delta * tot
        for i in np.argsort(-q):
            if need <= 1e-12:
                break
            m = mix["fold"][i] * r[i]
            if m <= 0:
                continue
            f = min(1.0, need / m)
            out["call"][i] += f * out["fold"][i]
            out["fold"][i] *= 1 - f
            need -= f * m
    return out


def _masked(mix: Dict[str, np.ndarray], held: np.ndarray, fill=np.nan) -> Dict[str, np.ndarray]:
    return {k: np.where(held, v, fill) for k, v in mix.items()}
