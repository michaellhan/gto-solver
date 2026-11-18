"""Builds gto/pf_norake_data.py: what the no-rake charts (gto/pf_norake.py) shift by, read off Lizard's preflop solver.

    .venv/bin/python scripts/build_pf_norake.py <lizard repo>      (seconds: it reads stored solves, solves nothing)

Lizard (michaellhan/lizard, engine/pf_solve.py: a calibrated class-level preflop model) stores its 6-max and heads-up
100bb games solved with no rake and with 5% capped at 3bb (engine/pf/<tag>_6max_100bb.bin, ..._r5c3.bin; the charts'
rake is 5% capped at 4bb, advisor.CHART_RAKE, which preflop differs from a 3bb cap only in the biggest pots). For every
chart spot the charts play (opens, facing an open, the opener facing a 3bet, a cold 3bet, the 3bettor facing a 4bet;
heads-up also the limp lines) it records, at the matching node of each solve: the share of hero's range there that
continues (raises or calls) and that raises, and from the no-rake solve each hand class's best EV against folding, its
raise-minus-call EV and its weight in hero's range there. The chart keeps its own hands; pf_norake.shift moves it by
those shares (the Preflop Trainer's method, preflop/scripts/build-norake.py).
"""
from __future__ import annotations

import base64
import os
import sys

import numpy as np

LIZ = sys.argv[1]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(LIZ, "solver"), LIZ]
import gto.native  # noqa: E402,F401   (Lizard's snapshot of this engine)
from gto.pf_classes import CLASSES, N_IN  # noqa: E402
from engine import pf_solve as PS  # noqa: E402

POS = {6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"], 2: ["SB", "BB"]}


def load(n, rake):
    spec = PS.Spec(players=n, rake=rake)
    return PS.Solve(spec, open(os.path.join(LIZ, "engine", "pf", spec.key + ".bin"), "rb").read())


SOLVES = {n: (load(n, (0.05, 3.0)), load(n, None)) for n in (6, 2)}


def line(n, events, hero):
    """The line to hero's decision: `events` [(seat, action)] in order, every other seat folding when its turn comes."""
    out, ev, folded = [], list(events), set()
    for seat in POS[n] * 4:
        if not ev and seat == hero:
            return out
        if seat in folded:
            continue
        if ev and seat == ev[0][0]:
            out.append({"pos": seat, "action": ev.pop(0)[1]})
        elif not any(e["pos"] == seat for e in out):
            out.append({"pos": seat, "action": "fold"})
            folded.add(seat)
    raise ValueError((events, hero))


def node(solve, ln):
    """Per class: range weight, mix [raise, call] and EVs [raise, call] (bb against folding; -inf where absent). A check
    is the residual action (as in the charts), so it counts as neither."""
    r = solve.node(ln)
    if "grid" not in r:
        raise RuntimeError((ln, r.get("error")))
    names = r["grid"]["actions"]
    w, f, ev = np.zeros(169), np.zeros((169, 2)), np.full((169, 2), -np.inf)
    for ci, c in enumerate(CLASSES):
        cell = r["grid"]["cells"].get(c)
        if cell is None:
            continue
        w[ci] = cell["w"]
        for nm, fr, e in zip(names, cell["f"], cell["ev"]):
            k = 0 if nm in ("raise", "all-in") else 1 if nm == "call" else None
            if k is not None:
                f[ci, k] += fr
                ev[ci, k] = max(ev[ci, k], e)
    return w, f, ev


def b64(a, dtype):
    return base64.b64encode(np.asarray(a).astype(dtype).tobytes()).decode()


def spots():
    """(key, game, events, hero): pf_norake.spot_key's keys, '<game>|<hero>|<raisers in order>' ('L' a limp)."""
    P = POS[6]
    for h in P[:-1]:
        yield f"6|{h}|", 6, [], h
    for i, o in enumerate(P):
        for h in P[i + 1:]:
            yield f"6|{h}|{o}", 6, [(o, "raise")], h
        for t in P[i + 1:]:
            yield f"6|{o}|{o},{t}", 6, [(o, "raise"), (t, "raise")], o
            yield f"6|{t}|{o},{t},{o}", 6, [(o, "raise"), (t, "raise"), (o, "raise")], t
            for h in P[P.index(t) + 1:]:
                yield f"6|{h}|{o},{t}", 6, [(o, "raise"), (t, "raise")], h
    yield "2|SB|", 2, [], "SB"
    yield "2|BB|SB", 2, [("SB", "raise")], "BB"
    yield "2|SB|SB,BB", 2, [("SB", "raise"), ("BB", "raise")], "SB"
    yield "2|BB|SB,BB,SB", 2, [("SB", "raise"), ("BB", "raise"), ("SB", "raise")], "BB"
    yield "2|BB|L", 2, [("SB", "call")], "BB"
    yield "2|SB|L,BB", 2, [("SB", "call"), ("BB", "raise")], "SB"


out, rows = {}, []
for key, n, events, hero in spots():
    ln = line(n, events, hero)
    try:
        w0, f0, _ = node(SOLVES[n][0], ln)
        w1, f1, ev = node(SOLVES[n][1], ln)
    except RuntimeError as e:
        rows.append(f"{key:20} skipped ({e})")
        continue
    sh = lambda w, f, k: float((f[:, k] * w * N_IN).sum() / max((w * N_IN).sum(), 1e-12))
    c0, c1 = sh(w0, f0, 0) + sh(w0, f0, 1), sh(w1, f1, 0) + sh(w1, f1, 1)
    r0, r1 = sh(w0, f0, 0), sh(w1, f1, 0)
    best = np.max(np.where(np.isfinite(ev), ev, -9.0), axis=1)
    edge = np.where(np.isfinite(ev[:, 0]) & np.isfinite(ev[:, 1]), ev[:, 0] - ev[:, 1], 0.0)
    out[key] = {"s": [round(c0, 4), round(c1, 4), round(r0, 4), round(r1, 4)],
                "best": b64(np.clip(np.round(best * 100), -127, 127), "i1"),       # 0.01bb, held to +-1.27
                "edge": b64(np.clip(np.round(edge * 100), -127, 127), "i1"),
                "w": b64(np.round(np.clip(w1, 0, 1) * 255), "u1")}
    rows.append(f"{key:20} continue {100 * c0:5.1f} -> {100 * c1:5.1f}   raise {100 * r0:5.1f} -> {100 * r1:5.1f}")

labels = {n: [s.spec.label() for s in SOLVES[n]] for n in SOLVES}
with open(os.path.join(HERE, "gto", "pf_norake_data.py"), "w") as f:
    f.write('"""Generated by scripts/build_pf_norake.py (don\'t edit): per chart spot, Lizard\'s preflop solver at 5% rake\n'
            "capped at 3bb and at no rake. s: [continue raked, continue no rake, raise raked, raise no rake] as shares of\n"
            "hero's range at the node; best / edge: per hand class (gto.pf_classes.CLASSES order) the no-rake solve's best\n"
            "EV against folding and raise-minus-call EV, int8 in 0.01bb; w: the class's weight in hero's range there, uint8\n"
            '/255."""\n')
    f.write(f"SOURCE = {labels!r}\n")
    f.write(f"MODEL = {PS.MODEL_TAG!r}\n")
    f.write("DATA = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in out.items()) + "}\n")
print("\n".join(rows))
print(len(out), "spots;", labels)
