"""Range parsing into a 1326-length numpy weight vector.

Syntax (comma separated): AA, KK+, AKs, AKo, AK, A5s+, T9s-65s, 22-77, 'AhKh',
optional weight suffix ':0.5' e.g. 'A5s:0.5'. 'random'/'100%' = all combos.
"""
from __future__ import annotations
import re
import numpy as np
from typing import Dict, Iterable, Tuple
from .cards import RANKS, ALL_COMBOS, COMBO_INDEX, rank, suit, card, hand_class

N_COMBOS = 1326


_CLASS = re.compile(r"([2-9TJQKA])\1|(?!(.)\2)[2-9TJQKA]{2}[so]?")


def _class_combos(cls: str):
    """Combos for a class like 'AKs', 'AKo', 'QQ'."""
    if not _CLASS.fullmatch(cls):   # 'AKS' / 'AKx' read as offsuit and 'AAs' as AA
        raise ValueError(f"bad hand class {cls!r}")
    r1, r2 = RANKS.index(cls[0]), RANKS.index(cls[1])
    out = []
    for cb in ALL_COMBOS:
        a, b = cb
        ra, rb = rank(a), rank(b)
        if {ra, rb} != {r1, r2}:
            continue
        if r1 == r2:
            out.append(cb)
        elif len(cls) == 3:
            same = suit(a) == suit(b)
            if (cls[2] == "s") == same:
                out.append(cb)
        else:
            out.append(cb)
    return out


def _expand_token(tok: str):
    tok = tok.strip()
    if not tok:
        return []
    if tok.lower() in ("random", "100%", "any"):
        return list(ALL_COMBOS)
    # ranks in capitals, the suitedness letter small, whatever the case typed ('aks', 'AKS')
    tok = re.sub(r"([2-9TJQKA])([2-9TJQKA])([SO]?)", lambda m: m[1] + m[2] + m[3].lower(), tok.upper())
    # specific combo like AhKh
    if len(tok) == 4 and tok[1].lower() in "cdhs" and tok[3].lower() in "cdhs":
        a, b = card(tok[:2]), card(tok[2:])
        return [(min(a, b), max(a, b))]
    if "-" in tok:
        lo, hi = tok.split("-")
        lo, hi = lo.strip(), hi.strip()
        if len(lo) == 2 and lo[0] == lo[1]:  # pair range 22-77
            a, b = RANKS.index(lo[0]), RANKS.index(hi[0])
            return [cb for r in range(min(a, b), max(a, b) + 1) for cb in _class_combos(RANKS[r] * 2)]
        if lo[0] == hi[0]:  # ATo-A5o : same high card, kicker range
            hi_r = RANKS.index(lo[0])
            k1, k2 = RANKS.index(lo[1]), RANKS.index(hi[1])
            suff = lo[2:] if len(lo) > 2 else ""
            return [cb for k in range(min(k1, k2), max(k1, k2) + 1) for cb in _class_combos(RANKS[hi_r] + RANKS[k] + suff)]
        # T9s-65s : same gap, kicker range
        top = RANKS.index(lo[0]); top2 = RANKS.index(hi[0])
        gap = RANKS.index(lo[0]) - RANKS.index(lo[1])
        suff = lo[2:] if len(lo) > 2 else ""
        out = []
        for t in range(min(top, top2), max(top, top2) + 1):
            out += _class_combos(RANKS[t] + RANKS[t - gap] + suff)
        return out
    plus = tok.endswith("+")
    if plus:
        tok = tok[:-1]
    if len(tok) == 2 and tok[0] == tok[1]:
        r = RANKS.index(tok[0])
        rs = range(r, 13) if plus else [r]
        return [cb for rr in rs for cb in _class_combos(RANKS[rr] * 2)]
    hi_r, lo_r = RANKS.index(tok[0]), RANKS.index(tok[1])
    if hi_r < lo_r:
        hi_r, lo_r = lo_r, hi_r
        tok = tok[1] + tok[0] + tok[2:]
    suff = tok[2:] if len(tok) > 2 else ""
    if plus:
        out = []
        for k in range(lo_r, hi_r):
            out += _class_combos(RANKS[hi_r] + RANKS[k] + suff)
        return out
    return _class_combos(tok)


def parse_range(text: str) -> np.ndarray:
    w = np.zeros(N_COMBOS)
    # commas, or just spaces / new lines, between hands; spaces around a weight's ':' are fine
    for part in re.split(r"[,\s]+", re.sub(r"\s*:\s*", ":", text)):
        part = part.strip()
        if not part:
            continue
        weight = 1.0
        if ":" in part:
            part, ws = part.split(":")
            weight = float(ws)
        for cb in _expand_token(part):
            w[COMBO_INDEX[cb]] = max(w[COMBO_INDEX[cb]], weight)
    return w


def range_from_classes(weights: Dict[str, float]) -> np.ndarray:
    w = np.zeros(N_COMBOS)
    for cls, wt in weights.items():
        for cb in _class_combos(cls):
            w[COMBO_INDEX[cb]] = wt
    return w


def remove_blockers(w: np.ndarray, dead: Iterable[int]) -> np.ndarray:
    dead = set(dead)
    out = w.copy()
    for i, (a, b) in enumerate(ALL_COMBOS):
        if a in dead or b in dead:
            out[i] = 0.0
    return out


def range_to_text(w: np.ndarray, thresh: float = 0.0) -> str:
    """Summarise a weight vector by hand class with average weight."""
    acc: Dict[str, list] = {}
    for i, cb in enumerate(ALL_COMBOS):
        if w[i] > thresh:
            acc.setdefault(hand_class(cb), []).append(w[i])
    items = sorted(acc.items(), key=lambda kv: -np.mean(kv[1]))
    return ", ".join(f"{k}" if abs(np.mean(v) - 1) < 1e-9 else f"{k}:{np.mean(v):.2f}" for k, v in items)
