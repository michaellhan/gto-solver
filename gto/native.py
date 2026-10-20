"""Solve settings for the native server and CLI (the browser's are in static/worker.js).

Import this before numpy: it holds BLAS to one thread. The solver's matrices are small, and with other
work on the machine multithreaded Accelerate ran ~10 iterations/s where one thread ran ~150 (unloaded the
two are the same speed).
"""
from __future__ import annotations
import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from typing import Dict, Optional  # noqa: E402
from .advisor import Advisor        # noqa: E402
from .solver import Config          # noqa: E402

# Each speed is a tree and its budgets (seconds for the street advised on, and for each earlier street
# used to narrow ranges). Narrowing walks the earlier streets' solves, so they must settle too: Config()'s
# full menu (two bet and two raise sizes a street) ran 10 turn iterations in 3 s, 60-70% of the pot from
# equilibrium, and narrowing on it left BB 7 combos checking a K72T5 river where a long solve keeps 81.
# Measured on BTN vs BB single-raised pots (Ks7d2c, Ks7d2cTh, Qs8d3c4h, Ks7d2cTh5s):
# - fast: the browser's sizes (one turn and river bet and raise size) with every river under each turn:
#   the turn at 2.5 s is 1.5-2% of the pot from equilibrium (40-45 iterations), the flop 0-0.2%, and BB's
#   river check range 79 combos. With 1.2 s for an earlier turn it was 8-10% (58-64 combos); 8 fixed
#   rivers settle sooner but narrow to 63 even when converged.
# - accurate: two turn and river bet sizes: the turn at 30 s is 0.3-0.7%; an earlier turn at 12 s is
#   1.7-2% and leaves 89 combos (at 6 s, 9-10%). Its sizes differ from fast's, so it can't warm-start
#   from a fast solve of the spot.
_FLOP = {"bet": [0.33, 0.75], "raise": [0.5]}
SPEEDS = {
    "fast": (dict(bet_sizes={"flop": _FLOP["bet"], "turn": [0.75], "river": [0.75]},
                  raise_sizes={"flop": _FLOP["raise"], "turn": [0.75], "river": [0.75]}), 2.5, 2.5),
    "accurate": (dict(bet_sizes={"flop": _FLOP["bet"], "turn": [0.5, 1.0], "river": [0.5, 1.0]},
                      raise_sizes={"flop": _FLOP["raise"], "turn": [0.75], "river": [0.75]}), 30.0, 12.0),
}


def advisor(speed: str = "fast", depth: int = 2, budget: Optional[float] = None,
            prior_budget: Optional[float] = None) -> Advisor:
    sizes, b, pb = SPEEDS[speed]
    cfg = Config(bet_sizes={k: list(v) for k, v in sizes["bet_sizes"].items()},
                 raise_sizes={k: list(v) for k, v in sizes["raise_sizes"].items()}, depth=depth)
    return Advisor(cfg, time_budget=b if budget is None else budget,
                   prior_budget=pb if prior_budget is None else prior_budget)


def advisors(depth: int = 2, budget: Optional[float] = None, prior_budget: Optional[float] = None) -> Dict[str, Advisor]:
    """One advisor per speed (each keeps its own solve cache); budget/prior_budget override fast's."""
    return {"fast": advisor("fast", depth, budget, prior_budget), "accurate": advisor("accurate", depth)}
