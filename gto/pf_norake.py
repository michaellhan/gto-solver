"""No-rake charts: hero's preflop chart at a table without rake, moved from the published raked solves it was fit to.

Most charts are fit to published raked solves (5% capped at 4bb, advisor.CHART_RAKE); the big blind's defence of an open
and the opener's in-position calls of a blind's 3bet are written without rake already (advisor.Advisor._priced), so
they're left as they are, and so is every chart at a table with rake (the advice re-prices its calls for the rake).

Without rake the chart keeps its own hands and is moved only by how much more (or less) Lizard's preflop solver
continues and raises at the same spot without rake than at 5% capped at 3bb (gto/pf_norake_data.py, built by
scripts/build_pf_norake.py; the Preflop Trainer's method, preflop/scripts/build-norake.py):
- the share of hero's range that continues is scaled by the no-rake / raked ratio at the node, or, for a range that
  plays over half its hands, the share it folds; the share that raises by the square root of its ratio (the model's
  raise / call split is the less certain part); ratios held to 0.67-1.5, shares under 1.5% left alone;
- the hands that take up or give back the difference are picked by the chart's own mix (mixed hands fill up or empty
  out before pure ones flip) plus the no-rake solve's EV: its best hands to continue go first, its worst first out.
Opens barely move; the blinds and the callers defend wider (the small blind against a button open 18 -> 25% of hands).

The spot is the chart's own (pf.chart_key's seat, 6-max or heads-up): opens, facing an open (callers of it aside),
the opener facing a 3bet, a cold 3bet, the 3bettor facing a 4bet, and heads-up the limp lines. Other spots (limped
pots at 6-max, a player facing a raise after calling, 5bets) are left as they are.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from .pf_classes import CLASS_OF, N_IN

CLIP = (0.67, 1.5)
MIN_SHARE = 0.015
NOTE = ("no rake: the chart (fit to raked solves) moved by how much more a preflop solver continues and raises here "
        "without rake")


@lru_cache(maxsize=None)
def _spot(key: str) -> Optional[dict]:
    from .pf_norake_data import DATA
    d = DATA.get(key)
    if d is None:
        return None
    dec = lambda s, t: np.frombuffer(base64.b64decode(s), dtype=t).astype(float)
    return {"s": d["s"], "best": dec(d["best"], "i1") / 100, "edge": dec(d["edge"], "i1") / 100,
            "w": dec(d["w"], "u1") / 255}


def spot_key(hero: str, actions: List[dict], positions) -> Optional[str]:
    """The chart spot's key in pf_norake_data ('6|BTN|CO', '2|SB|L,BB'...), or None where the chart is written without
    rake or the spot isn't one the data covers."""
    from . import preflop as pf
    seats = list(positions or pf.POSITIONS)
    role = lambda p: pf._norm(pf.chart_key(p, positions or pf.POSITIONS))
    hu = len(seats) == 2
    line, acted = [], False
    for a in actions:
        if a["action"] in ("raise", "all-in"):
            line.append(role(a["pos"]))
        elif a["action"] == "call" and not line:
            if not hu or a["pos"] != "SB":
                return None          # limped pots: heads-up only (the small blind's limp)
            line.append("L")
        if a["pos"] == hero and a["action"] != "fold":
            acted = True
    h = role(hero)
    raisers = [r for r in line if r != "L"]
    mine = [("SB" if r == "L" else r) for r in line[:-1]]    # (heads-up the limp is the small blind's)
    if acted and (not line or line[-1] == h or h not in mine):
        return None                  # hero called earlier (or raised last): not a chart spot covered
    # written without rake: the big blind facing one raise, the opener's calls in position of a blind's 3bet
    if len(raisers) == 1 and line == raisers and h == "BB":
        return None
    if len(raisers) == 2 and raisers[0] == h and h in ("UTG", "HJ", "CO", "BTN") and raisers[1] in ("SB", "BB"):
        return None
    if len(raisers) > 3:
        return None
    return f"{2 if hu else 6}|{h}|{','.join(line)}"


def shift(acts: Dict[str, np.ndarray], key: str) -> Dict[str, np.ndarray]:
    """The chart `acts` (action -> weight per combo; fold and check implicit) moved to no rake at spot `key`."""
    d = _spot(key)
    if d is None:
        return acts
    c0, c1, r0, r1 = d["s"]
    rk = [k for k in acts if k in ("raise", "all-in")]
    ck = [k for k in acts if k == "call"]
    if len(rk) > 1 or not (rk or ck) or set(acts) - {"raise", "all-in", "call", "check", "fold"}:
        return acts
    # per class (the charts are class-level): [raise, call]
    cls = lambda v: np.bincount(CLASS_OF, weights=np.broadcast_to(np.asarray(v, dtype=float), CLASS_OF.shape),
                                minlength=169) / N_IN
    a = np.stack([cls(acts[rk[0]]) if rk else np.zeros(169), cls(acts["call"]) if ck else np.zeros(169)], axis=1)
    a0 = a.copy()
    wgt = N_IN * d["w"]
    best, edge = d["best"], d["edge"]
    ratio = lambda x, y: float(np.clip(y / x, *CLIP)) if x > MIN_SHARE else 1.0
    score = a.sum(1) + 0.5 * np.clip(best, -1, 1)
    cur, tot = float((a.sum(1) * wgt).sum()), float(wgt.sum())
    if tot <= 0:
        return acts
    target = cur * ratio(c0, c1) if cur / tot <= 0.5 else tot - (tot - cur) * ratio(1 - c0, 1 - c1)
    if target > cur:
        for i in np.argsort(-score, kind="stable"):
            if cur >= target - 1e-9:
                break
            room = 1 - a[i].sum()
            if room <= 1e-9 or wgt[i] == 0 or best[i] <= -1.26:
                continue
            add = min(room, (target - cur) / wgt[i])
            a[i, 1 if ck and (not rk or edge[i] < 0) else 0] += add
            cur += add * wgt[i]
    else:
        for i in np.argsort(score, kind="stable"):
            if cur <= target + 1e-9:
                break
            have = a[i].sum()
            if have <= 0 or wgt[i] == 0:
                continue
            cut = min(have, (cur - target) / wgt[i])
            a[i] *= (have - cut) / have
            cur -= cut * wgt[i]
    if rk and ck:   # the raise share within it (call <-> raise)
        t = a.sum(1)
        e = np.where(t > 0, a[:, 0] / np.maximum(t, 1e-9), 0.0) + np.clip(edge, -1, 1)
        cur = float((a[:, 0] * wgt).sum())
        target = cur * ratio(r0, r1) ** 0.5
        for i in (np.argsort(-e, kind="stable") if target > cur else np.argsort(e, kind="stable")):
            if abs(cur - target) < 1e-9:
                break
            if wgt[i] == 0:
                continue
            if target > cur and a[i, 1] > 0:
                mv = min(a[i, 1], (target - cur) / wgt[i])
                a[i] += [mv, -mv]
                cur += mv * wgt[i]
            elif target < cur and a[i, 0] > 0:
                mv = min(a[i, 0], (cur - target) / wgt[i])
                a[i] += [-mv, mv]
                cur -= mv * wgt[i]
    a = np.clip(a, 0.0, 1.0)
    out = dict(acts)
    # per combo: a class's combos move together (a combo the chart weights apart from its class keeps that ratio)
    for j, k in ((0, rk[0] if rk else None), (1, "call" if ck else None)):
        if k is None:
            continue
        old = np.broadcast_to(np.asarray(acts[k], dtype=float), CLASS_OF.shape)
        new_c, old_c = a[CLASS_OF, j], a0[CLASS_OF, j]
        out[k] = np.clip(np.where(old_c > 1e-12, old * new_c / np.maximum(old_c, 1e-12), new_c), 0.0, 1.0)
    for k in ("check", "fold"):   # the residual action, when the chart carries it
        if k in out:
            out[k] = np.clip(1.0 - sum(v for j, v in out.items() if j not in ("check", "fold")), 0.0, 1.0)
    return out


def chart(acts: Dict[str, np.ndarray], hero: str, actions: List[dict], positions) -> Dict[str, np.ndarray]:
    """The chart at a table without rake (shift at the spot's key); with rake, or off the covered spots, as it is."""
    if getattr(positions, "rake", None) is not None or getattr(positions, "icm", None) is not None:
        return acts
    key = spot_key(hero, actions, positions)
    return shift(acts, key) if key else acts
