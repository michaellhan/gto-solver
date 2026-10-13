"""Postflop solver speed benchmark: BTN vs BB single-raised pot on three flops with the browser's Fast
config, a fixed 150 iterations. Prints iterations/second and the exploitability reached.

    python scripts/bench_solver.py [--save FILE] [--compare FILE]

--save writes each board's root average strategy to an .npz; --compare checks a run against one
(speedups must leave the results unchanged).
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from gto import preflop as pf
from gto.advisor import preflop_ranges
from gto.cards import cards
from gto.solver import Config, solve

FAST = dict(n_turns=3, n_rivers=8, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4,
            bet_sizes={"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]},
            raise_sizes={"flop": [0.5], "turn": [0.75], "river": [0.75]})
BOARDS = ["Qs8d3c", "Kd7c2h", "9h8h6c"]
ITERS = 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save"); ap.add_argument("--compare")
    args = ap.parse_args()
    T = pf.Table(pf.SEATS[6])
    line = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
            {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
    rng = preflop_ranges(line, T, 100)
    ref = np.load(args.compare) if args.compare else None
    saved = {}
    for b in BOARDS:
        t0 = time.time()
        sol = solve(cards(b), rng["BB"], rng["BTN"], 5.5, 97.5, cfg=Config(**FAST), time_budget=1e9, max_iters=ITERS)
        total = time.time() - t0
        expl = sol.exploitability()
        strat = sol.root.avg_strategy0()
        saved[b] = strat
        msg = f"{b}: {sol.iters / sol.seconds:6.1f} it/s ({sol.seconds:.2f}s solving, {total:.2f}s total)  exploitability {expl:.3f}%"
        if ref is not None:
            msg += f"  max strategy diff {np.abs(strat - ref[b]).max():.2e}"
        print(msg, flush=True)
    if args.save:
        np.savez(args.save, **saved)


if __name__ == "__main__":
    main()
