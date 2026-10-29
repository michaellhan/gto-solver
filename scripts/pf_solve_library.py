"""The preflop game (gto/pf_game.py) solved with the flop library's values (pf_game.LibraryModel), against the
published spots (tests/preflop_refs.py) and the charts.

    python scripts/pf_solve_library.py --iters 1500 --save run.npz       # solve, then the report
    python scripts/pf_solve_library.py --load run.npz --compare          # a saved solve: report, charts in the game

Options: --rake RATE,CAP (default 5%,4bb: the references' rake), --river-fix (advisor._river_fix's correction on
the library's single-raised pots), --model eqr (the measured realization model instead, for comparison).
"""
from __future__ import annotations
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import gto.native  # noqa: F401,E402
import numpy as np  # noqa: E402


def make_model(kind: str, river_fix: float = 0.0, river3: float = 0.0):
    from gto import pf_game as G
    if kind == "eqr":
        return G.measured_model()
    fix = None
    if river_fix or river3:
        from gto.advisor import _river_fix
        fix = _river_fix()
    return G.LibraryModel(river_fix=fix, river_scale={1: river_fix, 2: river3})


def make_game(model, rake, positions=None, stack=100.0, **build):
    from gto import pf_game as G
    pos = positions or G.SIX
    return G.Game(G.build(pos, stack=stack, **build), pos, model, stack=stack, rake=rake)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--rake", default="0.05,4")
    ap.add_argument("--model", default="library", choices=("library", "eqr"))
    ap.add_argument("--river-fix", type=float, default=0.0, help="scale of advisor._river_fix in single-raised pots")
    ap.add_argument("--river-fix3", type=float, default=0.0, help="...and in 3bet pots")
    ap.add_argument("--save", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--compare", action="store_true", help="the charts against the solve, spot by spot (pf_compare)")
    a = ap.parse_args()
    rake = tuple(float(x) for x in a.rake.split(",")) if a.rake else None
    g = make_game(make_model(a.model, a.river_fix, a.river_fix3), rake)
    if a.load:
        g.load(a.load)
    else:
        t0 = time.time()
        g.solve(a.iters, verbose=True, every=500)
        print(f"solved {a.iters} iterations in {time.time() - t0:.0f}s", flush=True)
        if a.save:
            g.save(a.save)
    import pf_equilibrium as P
    P.print_report(g)
    if a.compare:
        import pf_compare
        pf_compare.compare(g)


if __name__ == "__main__":
    main()
