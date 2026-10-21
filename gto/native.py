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
from .flop_library import default_library  # noqa: E402

# Each speed is a tree and its budgets (seconds for the street advised on, and for each earlier street
# used to narrow ranges). Narrowing walks the earlier streets' solves, so they must settle too: Config()'s
# full menu (two bet and two raise sizes a street) ran 10 turn iterations in 3 s, 60-70% of the pot from
# equilibrium, and narrowing on it left BB 7 combos checking a K72T5 river where a long solve keeps 81.
# Measured on BTN vs BB single-raised pots (Ks7d2c, Ks7d2cTh, Qs8d3c4h, Ks7d2cTh5s), before river classes and the
# turn menus below:
# - fast: the browser's sizes (one turn and river bet and raise size) with every river under each turn:
#   the turn at 2.5 s is 1.5-2% of the pot from equilibrium (40-45 iterations), the flop 0-0.2%, and BB's
#   river check range 79 combos (before the charts followed published raked solves; long solves now keep
#   87.7 with the bucketed flop below, and a loaded machine's short earlier solves are followed only as far
#   as they've settled: advisor.PRIOR_TRUST). With 1.2 s for an earlier turn it was 8-10% (58-64 combos); 8 fixed
#   rivers settle sooner but narrow to 63 even when converged.
# - accurate: two turn and river bet sizes: the turn at 30 s is 0.3-0.7%; an earlier turn at 12 s is
#   1.7-2% and leaves 89 combos (at 6 s, 9-10%). Its sizes differ from fast's, so it can't warm-start
#   from a fast solve of the spot.
# From the flop, accurate plays the river with its betting and one turn per rank class (Config's river_play
# and turn_classes, with one turn and river size inside the flop solve): ~15 iterations a second on a rainbow
# flop, half that on a two-tone or monotone one, and from ~100 iterations its c-bet frequencies are within a
# few points of a solve over every turn (README, Accuracy vs speed). Fast plays the same flop model with the turn
# and river by 32 hand buckets per player and turn card (river_buckets, turn_buckets): an iteration costs about
# what the old fast tree's did (6 random turns, the river checked down: 12-18 ms), so 2.5 s gives 140-210, and
# against every turn played its c-bet is 3.5 points off on average (was 12), its size split 7 (33), BB's check 2 (6).
#
# From the turn (scripts/turn_river_reference.py: six turn and four river spots against every-river solves run to
# convergence with a 33/75/125% menu; README Accuracy vs speed), both play the river as one card per class
# (river_classes: 13 cards on a rainbow turn, 22-24 with a flush suit, each standing for its cards through card swaps;
# 0.3-0.5x every river's cost an iteration, about as close to equilibrium at the same iterations). The menu was the
# biggest error: those solves overbet 29-36% of the range on blank and overcard turns, so one 75% size, converged,
# was 0.6-3.8% of the pot off per combo. Accurate plays 33/75/125% (about its old two sizes' cost with every river):
# 0.11-0.32% of the pot off after 30 s, where it was 0.7-2.9. Fast plays the browser Fast's turn menu (33/125%, the
# river inside at 75%) and its turn also gets the 2.5 s the narrowing flop didn't spend (Advisor.spare_prior: a flop
# from the library or from the hand's flop advice is free, so a turn request never takes longer than a cold one's
# 5 s): at 5 s (80-150 iterations) 0.5-1.8% of the pot off, 1.0 on average, where 75/125% at 2.5 s was 0.6-2.6, 1.65
# (75/125% at 5 s: 1.5, its missing third-pot lead 3.0 on K72-7). Cold (the flop solved in the same request) the turn
# keeps 2.5 s: 1.9 on average, 0.7-4.3.
# A river root bets 33/75/125/250% pot (Advisor.root_sizes): polar rivers overbet half the range, and 75% alone
# was 1.2-7.2% of the pot off per combo where these are 0.1-0.6; a river iteration takes milliseconds.
_FLOP = {"bet": [0.33, 0.75], "raise": [0.5]}
ROOT_SIZES = {"river": {"bet": [0.33, 0.75, 1.25, 2.5]}}
SPEEDS = {
    "fast": (dict(bet_sizes={"flop": _FLOP["bet"], "turn": [0.33, 1.25], "river": [0.75]},
                  raise_sizes={"flop": _FLOP["raise"], "turn": [0.75], "river": [0.75]},
                  model=dict(river_buckets=32, turn_buckets=True, bucket_river_step=3, river_classes=True)), 2.5, 2.5),
    "accurate": (dict(bet_sizes={"flop": _FLOP["bet"], "turn": [0.33, 0.75, 1.25], "river": [0.33, 0.75, 1.25]},
                      raise_sizes={"flop": _FLOP["raise"], "turn": [0.75], "river": [0.75]},
                      model=dict(river_classes=True)), 30.0, 12.0),
}


def config(speed: str = "fast", depth: int = 2) -> Config:
    """The speed's tree (its model's flags included)."""
    sizes = SPEEDS[speed][0]
    kw = dict(sizes.get("model", {}))
    return Config(bet_sizes={k: list(v) for k, v in sizes["bet_sizes"].items()},
                  raise_sizes={k: list(v) for k, v in sizes["raise_sizes"].items()}, depth=depth, **kw)


# 3-way pots solved by the 3-player solver (Advisor.mw_solve_streets; the rest use the betting rules). A river iteration
# costs 30-90 ms here (by machine load), so fast's 2.5 s runs 25-80 (0.1-1% of the pot from equilibrium). Turns and flops
# bet the later streets by 16 hand buckets a player (Advisor.mw_street_config, README 3-way pots): a turn iteration costs
# 0.1-0.2 s, a flop one 0.15-0.35 s after 1-2 s of tables, so both wait for accurate's 30 s (turn 150-300 iterations, flop
# 90-200, 0.5-1.5% of the pot from equilibrium checked to, 2-6% for the first to act's bigger tree).
MW_STREETS = {"fast": ("river",), "accurate": ("flop", "turn", "river")}


def advisor(speed: str = "fast", depth: int = 2, budget: Optional[float] = None,
            prior_budget: Optional[float] = None, library: bool = True) -> Advisor:
    """`library`: answer flop spots the precomputed library holds from it (gto/flop_library.py), at either speed."""
    b, pb = SPEEDS[speed][1:]
    adv = Advisor(config(speed, depth), time_budget=b if budget is None else budget,
                  prior_budget=pb if prior_budget is None else prior_budget)
    adv.root_sizes = {st: {k: list(v) for k, v in d.items()} for st, d in ROOT_SIZES.items()}
    adv.mw_solve_streets = MW_STREETS[speed]
    adv.library = default_library() if library else None
    adv.spare_prior = speed == "fast"
    return adv


def advisors(depth: int = 2, budget: Optional[float] = None, prior_budget: Optional[float] = None) -> Dict[str, Advisor]:
    """One advisor per speed (each keeps its own solve cache); budget/prior_budget override fast's."""
    return {"fast": advisor("fast", depth, budget, prior_budget), "accurate": advisor("accurate", depth)}
