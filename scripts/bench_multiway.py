"""Speed and convergence of the 3-way solver (gto/multiway_solver.py), by iterations.

CO opens, BTN and BB call (6-max, 100bb); the preflop ranges go to the board as they are. Prints CPU seconds
per iteration, each player's best-response gain (% of the pot) and the values (which sum to the pot).

  python scripts/bench_multiway.py --board Kh7d2c9s4h --iters 10,30,100
  python scripts/bench_multiway.py --board Kh7d2c9s --iters 10,30          # a turn: every river
  python scripts/bench_multiway.py --pairwise                               # without the joint card removal
  node scripts/pyodide_run.mjs scripts/bench_multiway.py --iters 5,10       # in Pyodide (see that file)
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gto.advisor as A
from gto import preflop as pf
from gto.cards import cards
from gto.multiway_solver import MWConfig, MultiwaySolver

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
       {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="Kh7d2c9s4h")
    ap.add_argument("--iters", default="10,30,100")
    ap.add_argument("--pairwise", action="store_true")
    ap.add_argument("--rivers", type=int, default=None)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--leaf-rivers", type=int, default=4)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--simultaneous", action="store_true")
    a = ap.parse_args()
    R = A.preflop_ranges(PRE, pf.Table(pf.SEATS[6]), 100.0)
    rs = [R["BB"], R["CO"], R["BTN"]]
    cfg = MWConfig(exact=not a.pairwise, depth=a.depth, alternate=not a.simultaneous, rivers=a.rivers, turns=a.turns, leaf_rivers=a.leaf_rivers)
    t0 = time.time()
    s = MultiwaySolver(cards(a.board), rs, 8.0, 97.5, cfg=cfg)
    print(f"{a.board}: build {time.time() - t0:.2f} s, {s.hands.n} combos, {len(s.decisions)} decisions, "
          f"{s.builder.n_nodes} nodes", flush=True)
    for k in [int(x) for x in a.iters.split(",")]:
        t0, w0 = time.process_time(), time.time()
        n0 = s.iters
        s.run(iters=k - n0)
        dt = (time.process_time() - t0) / max(k - n0, 1)
        dw = (time.time() - w0) / max(k - n0, 1)
        g = s.br_gains() / s.root_pot() * 100
        v = s.values()
        print(f"{s.iters:5d} iters  {dt:.3f} CPU s/iter ({dw:.3f} wall)  best-response gains % pot {np.round(g, 2)}  "
              f"values {np.round(v, 2)} (sum {v.sum():.3f}, pot {s.root_pot():.1f})  root {s.root.names} "
              f"{[round(x, 3) for x in s.range_freqs().values()]}", flush=True)


if __name__ == "__main__":
    main()
