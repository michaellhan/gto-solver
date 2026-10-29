"""Preflop charts of other games: stack depths, antes, a straddle, 7-9 handed tables and heads-up at other depths.

The hand-written charts (gto/preflop.py) are 6-max (and heads-up) at 100bb without antes. Each other game is solved
as the 100bb game is (gto/pf_game.py: the 169 classes, discounted CFR, flops paid by the flop library's realization,
LibraryModel; another depth's realization scaled by what live solves measure there, scripts/pf_depth_values.py),
and its chart at a decision is the 100bb chart plus how far the solve moves each class from the 100bb solve at the
same decision (scripts/build_pf_formats.py writes those moves to gto/pf_format_data.py):

    chart(game, decision)[class] = chart_100(decision)[class] + solve_game(decision)[class] - solve_100(decision)[class]

The solve's own errors at 100bb (docs/preflop_research.md: its opens are tight, it calls 3bets too little) are
mostly the same in both solves and cancel; what the game changes (a 3bet that commits the stack at 40bb, dead money
in the pot, more players behind an open) is the solve's. A class's move counts only as far as both solves bring
it to the decision (build_pf_formats.REACH), and a moved chart is kept in the dominance order by pooling (apply).
Each dimension is its own set of moves, added together:
  * depth: 6-max games at DEPTHS bb against 100bb (no ante), linear between neighbouring depths; heads-up its own
    (HU_DEPTHS) against heads-up 100bb;
  * ante: 6-max games with an ANTE bb ante against the same depth without, at ANTE_DEPTHS, scaled by the dead money
    (ante x players over ANTE x 6, at most 2x);
  * players: 7- and 8-handed 100bb games against the 6-max seats whose charts they play (preflop.chart_key);
    9-handed plays the 8-handed moves of the seat with as many players behind it (at most 7);
  * straddle: the straddled 6-max 100bb game against the seats whose charts it plays.
The players and straddle moves are added to the 6-max chart of those seats (the 100bb game's chart at the decision
the table's seats play), not to the table's own chart, whose rules for those games (the 7+ handed EP chart) the
moves replace; depth and ante moves to the table's own 100bb chart.
A decision is one of the charts' heads-up lines (raise first in; facing an open; the opener facing a 3bet; the
3bettor facing a 4bet; the opener facing a 5bet), keyed by hero and the raises before it. Depth and ante moves are
looked up at the 6-max (or heads-up) decision whose charts the table plays; players and straddle moves at the
table's own seats. Lines with callers, limps and other games' missing pieces are left to the charts' rules
(Advisor._priced re-prices them from the game the moves cover)."""
from __future__ import annotations
import base64
import zlib
from typing import Dict, List, Optional, Tuple
import numpy as np

DEPTHS = (20.0, 40.0, 60.0, 100.0, 150.0, 200.0)   # 6-max depths solved (100: the charts' own)
HU_DEPTHS = (20.0, 40.0, 100.0, 200.0)
ANTE = 0.125                                        # the ante solved (bb a player: 12.5% of a big blind)
ANTE_DEPTHS = (20.0, 40.0, 100.0)
MIN_DEPTH = 20.0                                    # shallower is push/fold (Advisor._push)

_DATA = None


def _data() -> dict:
    global _DATA
    if _DATA is None:
        from .pf_format_data import DELTAS
        _DATA = {dim: {sub: decode(*packed) for sub, packed in sets.items()} for dim, sets in DELTAS.items()}
    return _DATA


STEP = 100                                          # moves stored in 1% steps


def encode(nodes: Dict[str, Dict[str, np.ndarray]]) -> Tuple[str, str]:
    """One game's moves {line key: {action: per-class move (-1..1)}} as (index, data): the index 'key=rc ...'
    (each decision's actions, r raise, c call), the data every move in that order in 1% steps, int8, zlib, base64."""
    index, vecs = [], []
    for key in sorted(nodes):
        acts = [a for a in ("raise", "call") if a in nodes[key]]
        index.append(key + "=" + "".join(a[0] for a in acts))
        vecs += [np.asarray(nodes[key][a], float) for a in acts]
    q = np.clip(np.round(np.concatenate(vecs) * STEP), -127, 127).astype(np.int8) if vecs else np.zeros(0, np.int8)
    return " ".join(index), base64.b64encode(zlib.compress(q.tobytes(), 9)).decode()


def decode(index: str, data: str) -> Dict[str, Dict[str, np.ndarray]]:
    from .pf_classes import CLASSES
    q = np.frombuffer(zlib.decompress(base64.b64decode(data)), dtype=np.int8).astype(np.float64) / STEP
    out, i, k = {}, 0, len(CLASSES)
    for entry in index.split():
        key, acts = entry.rsplit("=", 1)
        out[key] = {}
        for c in acts:
            out[key]["raise" if c == "r" else "call"] = q[i:i + k]
            i += k
    assert i == len(q), "pf_format_data: index and data disagree"
    return out


def line_key(hero: str, actions: List[dict]) -> Optional[str]:
    """'hero|seat:r,seat:r' of a heads-up chart line, raises only: raise first in, facing an open, the opener facing
    the 3bet, the 3bettor facing the 4bet, the opener facing the 5bet; else None (a call or limp in the line)."""
    raisers = []
    for a in actions:
        if a["action"] in ("raise", "all-in", "bet"):
            raisers.append(a["pos"])
        elif a["action"] != "fold":
            return None
    n = len(raisers)
    if n > 4 or hero in raisers[:1] and n == 1 or (n >= 2 and raisers[0] == raisers[1]):
        return None
    if n >= 2:
        o, t = raisers[0], raisers[1]
        if raisers != [o, t, o, t][:n] or hero != (o if n % 2 == 0 else t):
            return None
    return hero + "|" + ",".join(f"{p}:r" for p in raisers)


def std_actions(hero: str, actions: List[dict]) -> Optional[List[dict]]:
    """The line with other players' calls of a raise as folds (the chart's own heads-up line; a squeeze plays the
    chart against the open), or None where a limp or hero's own call is in it."""
    out, raised = [], False
    for a in actions:
        if a["action"] in ("raise", "all-in", "bet"):
            raised = True
        elif a["action"] in ("call", "check"):
            if not raised or a["pos"] == hero:
                return None
            a = {"pos": a["pos"], "action": "fold"}
        out.append(a)
    return out


def node_key(hero: str, line) -> Optional[str]:
    """line_key of a pf_game node's line ((seat, action, to), ...)."""
    return line_key(hero, [{"pos": p, "action": a} for p, a, _ in line if a != "fold"])


def analog(hero: str, actions: List[dict], positions) -> Optional[str]:
    """The key of the 6-max (or heads-up) decision whose charts `hero` plays at this table (preflop.chart_key)."""
    from .preflop import chart_key, _norm
    if len(positions) == 2:
        return line_key(hero, actions)
    m = lambda p: _norm(chart_key(p, positions))
    acts = [dict(a, pos=m(a["pos"])) for a in actions if a["action"] != "fold"]
    return line_key(m(hero), acts)


def _interp(table: Dict[float, Dict[str, dict]], grid, x: float, key: str) -> Optional[Dict[str, np.ndarray]]:
    """The move at `key`, linear in x between the grid's neighbours (clamped to its ends); 100 is no move."""
    x = min(max(x, grid[0]), grid[-1])
    lo = max(g for g in grid if g <= x)
    hi = min(g for g in grid if g >= x)

    def at(g):
        if g == 100.0 and g not in table:
            return {}
        return table.get(g, {}).get(key)
    a, b = at(lo), at(hi)
    if a is None or b is None:
        return None
    t = 0.0 if hi == lo else (x - lo) / (hi - lo)
    out = {}
    for k in set(a) | set(b):
        out[k] = (1 - t) * a.get(k, 0.0) + t * b.get(k, 0.0)
    return out


def _seat8(hero: str, positions) -> str:
    """A 9-handed seat as the 8-handed seat with as many players behind it (at most 7)."""
    from .preflop import SEATS
    behind = min(len(positions) - 1 - list(positions).index(hero), 7)
    return SEATS[8][7 - behind]


def moves(hero: str, actions: List[dict], positions) -> Tuple[Optional[Dict[str, np.ndarray]], dict]:
    """The game's per-class moves for this decision ({action: (169,)}) and what they cover ({"depth": bb the chart is
    at, "ante": bb, "base": "6max" where the moves are from the 6-max chart of the seats the table plays, the players
    and straddle moves: the table's own chart has rules of its own for those games}), or (None, {}) where none apply
    (the default game, a line off the charts, too shallow)."""
    n = len(positions)
    depth = float(getattr(positions, "depth", 100.0))
    ante = float(getattr(positions, "ante", 0.0) or 0.0)
    straddle = bool(getattr(positions, "straddle", False))
    if (n in (2, 6) and abs(depth - 100.0) < 1e-9 and ante == 0.0 and not straddle) or depth < MIN_DEPTH:
        return None, {}
    data = _data()
    if not data:
        return None, {}
    key = line_key(hero, actions)
    if key is None:
        return None, {}
    total: Dict[str, np.ndarray] = {}

    def add(m):
        for a, v in m.items():
            total[a] = total.get(a, 0.0) + v
    cover = {"depth": 100.0, "ante": 0.0}
    if n in (7, 8, 9) and not straddle:
        pl = data.get("players", {})
        if n == 9:
            acts8 = [dict(a, pos=_seat8(a["pos"], positions)) for a in actions if a["action"] != "fold"]
            m = pl.get(8.0, {}).get(line_key(_seat8(hero, positions), acts8) or "")
        else:
            m = pl.get(float(n), {}).get(key)
        if m is not None:
            add(m)
            cover["base"] = "6max"
    if straddle and n == 6:
        m = data.get("straddle", {}).get(100.0, {}).get(key)
        if m is not None:
            add(m)
            cover["base"] = "6max"
    an = analog(hero, actions, positions)
    if an is not None and abs(depth - 100.0) > 1e-9:
        grid = HU_DEPTHS if n == 2 else DEPTHS
        m = _interp(data.get("hu_depth" if n == 2 else "depth", {}), grid, depth, an)
        if m is not None:
            add(m)
            cover["depth"] = min(max(depth, grid[0]), grid[-1])
    if an is not None and ante > 0 and n > 2:
        m = _interp(data.get("ante", {}), ANTE_DEPTHS, depth, an)
        if m is not None:
            x = min(2.0, ante * n / (ANTE * 6))
            add({a: x * v for a, v in m.items()})
            cover["ante"] = ante
    if not total:
        return None, {}
    return total, cover


def label(positions, cover: dict) -> str:
    """The game a decision's charts are for, in words ('40bb, 0.125bb ante, 8-handed'), from moves()'s cover."""
    n = len(positions)
    parts = [f"{cover.get('depth', 100.0):g}bb"]
    if cover.get("ante"):
        parts.append(f"{cover['ante']:g}bb ante")
    if n == 2:
        parts.append("heads-up")
    elif n != 6:
        parts.append(f"{n}-handed")
    if getattr(positions, "straddle", False):
        parts.append("straddled")
    return ", ".join(parts)


def apply(base: Dict[str, np.ndarray], mv: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The chart (per combo: {action: (1326,)}, fold implicit) moved per class, each action kept in [0, 1] and the
    continuing total at most 1, then made monotone without adding to it (isotonic)."""
    from .pf_classes import CLASS_OF
    out = {}
    for a in set(base) | {k for k in mv if k != "fold"}:
        v = np.asarray(base.get(a, 0.0), float) + (mv[a][CLASS_OF] if a in mv else 0.0)
        out[a] = np.clip(v, 0.0, 1.0)
    tot = sum(out.values())
    scale = np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)
    return isotonic({a: v * scale for a, v in out.items()})


def isotonic(acts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """A strategy (per combo, fold implicit) whose continuing frequency breaks the dominance order
    (preflop.monotone's chains: a hand continuing less than one it dominates) pooled where it does, along each
    chain (pool adjacent violators, by combos), keeping the continuing combos: a class the chart's move took out
    below a weaker one the chart keeps (the 8-handed UTG folding 33 and 44 and opening 22) meets it halfway, where
    preflop.monotone would add the stronger back. A pooled class plays its block's mix."""
    from .pf_classes import CLASS_INDEX, CLASS_OF, N_IN
    from .preflop import _dominance_chains
    names = sorted(acts)
    A = np.array([np.bincount(CLASS_OF, weights=acts[a], minlength=len(N_IN)) / N_IN for a in names])
    A0 = A.copy()
    n = np.asarray(N_IN, float)
    chains = [[CLASS_INDEX[h] for h in c] for c in _dominance_chains()]
    for _ in range(3):
        moved = False
        for chain in chains:
            cont = A[:, chain].sum(axis=0)
            if np.all(np.diff(cont) >= -1e-9):
                continue
            blocks = []                       # [start, end): pooled while a stronger block continues less
            for i in range(len(chain)):
                blocks.append([i, i + 1])
                while len(blocks) > 1:
                    (s0, e0), (s1, e1) = blocks[-2], blocks[-1]
                    w0, w1 = n[chain[s0:e0]], n[chain[s1:e1]]
                    if (cont[s1:e1] * w1).sum() / w1.sum() >= (cont[s0:e0] * w0).sum() / w0.sum() - 1e-12:
                        break
                    blocks[-2:] = [[s0, e1]]
            for s0, e0 in blocks:
                if e0 - s0 > 1:
                    ix = chain[s0:e0]
                    A[:, ix] = ((A[:, ix] * n[ix]).sum(axis=1) / n[ix].sum())[:, None]
                    moved = True
        if not moved:
            break
    changed = np.abs(A - A0).max(axis=0) > 1e-12
    if not changed.any():
        return acts
    out = {}
    for k, a in enumerate(names):
        v = np.asarray(acts[a], float).copy()
        m = changed[CLASS_OF]
        v[m] = A[k][CLASS_OF][m]
        out[a] = v
    return out