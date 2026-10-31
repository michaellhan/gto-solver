"""How quickly Study's preflop solves (gto/pf_study.py) converge: a game solved from scratch with the average strategy
weighing iteration t by t^gamma, its exploitability (mbb/hand a seat) at checkpoints, and with --ev how far its EVs
are at each checkpoint from the same solve run on to --ref iterations (at six nodes: the first decisions UTG and BTN,
BB and SB facing a BTN open, BTN facing BB's 3bet, CO facing BTN's 3bet; per node the mean over the classes' played
actions (>= 5% of the class, weighted by the class's share) of |EV - reference EV|, the worst class, the range's EV
and the actions' frequencies).

    python scripts/pf_study_convergence.py [--game 6max|r5c3|r5c4|s40|4max|4r5c3|a50] [--gamma 3,8] [--beta none|0.5] [--ev --ref 2000]

One thread; a 6-max run ~0.13 s an iteration natively (~2 minutes to 1,000, ~5 to --ref 2000). Prints one JSON line a
run. README (Study, Preflop: Charts or Solver) and pf_study's docstring quote its numbers.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gto.native  # noqa: F401,E402
from gto import pf_study as PS  # noqa: E402

GAMES = {"6max": PS.Spec(), "r5c3": PS.Spec(rake=(0.05, 3.0)), "r5c4": PS.Spec(rake=(0.05, 4.0)), "s40": PS.Spec(stack=40.0),
         "4max": PS.Spec(players=4), "4r5c3": PS.Spec(players=4, rake=(0.05, 3.0)),
         "a50": PS.Spec(stack=50.0, ante=0.12)}       # (a50 levels off at ~3 mbb: pf_study.STALL)
FOLDS = [("UTG", "fold"), ("HJ", "fold"), ("CO", "fold")]
NODES = {
    "UTG open": [],
    "BTN open": FOLDS,
    "BB vs BTN open": FOLDS + [("BTN", "raise", 2.5), ("SB", "fold")],
    "SB vs BTN open": FOLDS + [("BTN", "raise", 2.5)],
    "BTN vs BB 3bet": FOLDS + [("BTN", "raise", 2.5), ("SB", "fold"), ("BB", "raise", 12.0)],
    "CO vs BTN 3bet": FOLDS[:2] + [("CO", "raise", 2.5), ("BTN", "raise", 7.5), ("SB", "fold"), ("BB", "fold")],
}


def line(x):
    return [{"pos": a[0], "action": a[1], **({"to": a[2]} if len(a) > 2 else {})} for a in x]


def mbb(g) -> float:
    e = g.exploitability(per_seat=True)
    return 1000.0 * sum(e) / len(e)


def answers(sv: PS.Solve) -> dict:
    sv._vals = None
    return {k: sv.node(line(x)) for k, x in NODES.items()} if sv.spec.players == 6 else {}


def distance(a: dict, b: dict) -> dict:
    num = den = worst = 0.0
    for cl, cb in b["grid"]["cells"].items():
        ca = a["grid"]["cells"].get(cl)
        if not ca or not cb["w"]:
            continue
        for j, f in enumerate(cb["f"]):
            if f >= 0.05:
                d = abs(ca["ev"][j] - cb["ev"][j])
                num += cb["w"] * f * d
                den += cb["w"] * f
                worst = max(worst, d)
    return {"ev_mean": round(num / max(den, 1e-12), 4), "ev_worst": round(worst, 3),
            "range_ev": round(abs(a["range_ev_bb"] - b["range_ev_bb"]), 4),
            "freq": round(max(abs(x["freq"] - y["freq"]) for x, y in zip(a["actions"], b["actions"])), 4)}


def run(game: str, gamma: float, beta, checks, ev: bool, ref: int) -> dict:
    sv = PS.Solve(GAMES[game])
    g = sv.game
    g.avg_power = gamma
    out = {"game": game, "gamma": gamma, "beta": beta, "mbb": {}}
    snaps = {}
    t0 = time.time()
    last = max(checks + ([ref] if ev else []))
    for t in range(1, last + 1):
        if beta is None:
            g.iterate(t)
        else:   # (DCFR's discount of negative regrets, t^beta / (t^beta + 1), in place of halving them)
            s = g.current()
            g._up(g._down(s), s, float(t) ** gamma, update=True)
            g.regret *= np.where(g.regret > 0, t ** 1.5 / (t ** 1.5 + 1), t ** beta / (t ** beta + 1))
        sv.iters = t
        if t in checks:
            out["mbb"][t] = round(mbb(g), 2)
            if ev:
                snaps[t] = answers(sv)
    if ev:
        final = answers(sv)
        out["ref"] = {"iterations": ref, "mbb": round(mbb(g), 2)}
        out["ev"] = {t: {k: distance(a[k], final[k]) for k in final} for t, a in snaps.items()}
    out["seconds"] = round(time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="6max", choices=sorted(GAMES))
    ap.add_argument("--gamma", default="3,8")
    ap.add_argument("--beta", default="none")
    ap.add_argument("--checks", default="100,250,500,750,1000")
    ap.add_argument("--ev", action="store_true")
    ap.add_argument("--ref", type=int, default=2000)
    a = ap.parse_args()
    checks = [int(x) for x in a.checks.split(",")]
    beta = None if a.beta == "none" else float(a.beta)
    for gamma in a.gamma.split(","):
        print(json.dumps(run(a.game, float(gamma), beta, checks, a.ev, a.ref)), flush=True)


if __name__ == "__main__":
    main()
