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
from .advisor import Advisor, MW_FAST_STREETS, MW4_FAST_STREETS  # noqa: E402
from .solver import Config          # noqa: E402
from .flop_library import default_library  # noqa: E402
from . import mw_library  # noqa: E402
from . import core  # noqa: E402

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
# The compiled core runs it 2.1-4.7x faster than numpy (gto/core.py; README, Compiled core), and then building its
# bucket and river tables in numpy takes much of the 2.5 s on a loaded machine.
#
# From the turn (scripts/turn_river_reference.py: six turn and four river spots against every-river solves run to
# convergence with a 33/75/125% menu; README Accuracy vs speed), both play the river as one card per class
# (river_classes: 13 cards on a rainbow turn, 22-24 with a flush suit, each standing for its cards through card swaps;
# 0.3-0.5x every river's cost an iteration, about as close to equilibrium at the same iterations). The menu was the
# biggest error: those solves overbet 29-36% of the range on blank and overcard turns, so one 75% size, converged,
# was 0.6-3.8% of the pot off per combo. Accurate plays 33/75/125% (about its old two sizes' cost with every river):
# 0.11-0.32% of the pot off after 30 s, where it was 0.7-2.9. Fast plays the browser Fast's turn menu (33/75/125%, the
# river inside at 33/125%) and its turn also gets the 2.5 s the narrowing flop didn't spend (Advisor.spare_prior: a flop
# from the library or from the hand's flop advice is free, so a turn request never takes longer than a cold one's
# 5 s). The compiled core (gto/core.py) runs 2.8-3.6x numpy's turn iterations, so Fast's menu grew from 33/125% with
# the river at 75%: that one, converged in the core (~410 iterations in 5 s, 0.05% of the pot from equilibrium), was
# 0.58-1.48% of the pot off per combo on the five single-raised turns, 1.02 on average, frequencies 3.2 points; this
# one costs 2.6x an iteration (~150 in 5 s, 0.3%; ~105 at the browser's 8 s on a loaded machine): 0.34-0.55, 0.47 on
# average, 1.8 points (33/125% on the turn and 33/75/125% on the river: 0.48 and 2.9; 33/125% both: 0.62 and 2.5).
# Cold (the flop solved in the same request) the turn keeps 2.5 s (~75 iterations, 1.2% of the pot from equilibrium).
# A river root bets 33/75/125/250% pot (Advisor.root_sizes): polar rivers overbet half the range, and 75% alone
# was 1.2-7.2% of the pot off per combo where these are 0.1-0.6; a river iteration takes milliseconds.
# A shallow turn (at most 4.5 x the pot behind: a 3bet pot's turn has 3.2, a single-raised pot's 6.8-18) plays Fast at
# 33/75% instead of 33/125% (Advisor.spr_sizes): a 3bet pot barrels three quarters pot where a single-raised one
# overbets. Four CO vs BTN 3bet turns against the rich reference (scripts/turn_river_reference.py): per-combo EVs 1.95%
# of the pot off on average -> 0.81 at 5 s (0.55-4.8 -> 0.42-1.5), frequencies 6.8 points -> 2.0 (33/75/125%: 1.07 and
# 2.4, fewer iterations); cold at 2.5 s 1.82 -> 1.38. In single-raised pots 33/75% was 1.6-3.7% off (no overbet).
# Its river inside stays at 75%: in the core at the same budget 33/75% there gave 0.61% of the pot off on average,
# 33/125% on the river 0.57, 33/75/125% on the turn 0.62, both of the single-raised menu's 0.68 (one board 1.1-1.8).
FAST_SPR_SIZES = {"turn": (4.5, {"bet": [0.33, 0.75], "river": [0.75]})}
# With the core on THREADED_FAST threads or more (core.threads: gto/server.py --threads, worker.js's helpers) Fast's turn
# plays Accurate's tree, the river inside at 33/75/125% and shallow turns too: threads give 2.5-3x the iterations
# (README, Threads), and there the river's 75% is worth more than the iterations it costs. Per-combo EVs against the rich
# reference (% of the pot, the five single-raised turns' mean; scripts/turn_river_reference.py's spots, in the core) at
# 150 / 200 / 300 / 450 iterations: Accurate's tree 0.31 / 0.21 / 0.12 / 0.07, Fast's 0.38 / 0.35 / 0.34 / 0.34 (it
# settles there); the four 3bet turns 0.82 / 0.45 / 0.16 / 0.13 against the shallow menu's 0.69 at 200, 0.59 at 450.
# Accurate's tree costs 1.4x Fast's an iteration in single-raised pots and 2.2x in 3bet pots', so it pays from ~200 of
# Fast's iterations; on one thread Fast gets 105-240 in the browser's 8 s and ~150 natively in 5 s, and keeps its own.
THREADED_FAST = 4
# From the flop the river is played by 32 buckets of river strength (Config.river_strength; README, Accuracy vs
# speed): a draw enters the river strong on the cards that hit it and weak on the rest, where the river on the turn's
# boards made it a middling hand and the flop c-bet too little.
RIVER_STRENGTH = 32
_FLOP = {"bet": [0.33, 0.75], "raise": [0.5]}
ROOT_SIZES = {"river": {"bet": [0.33, 0.75, 1.25, 2.5]}}
SPEEDS = {
    "fast": (dict(bet_sizes={"flop": _FLOP["bet"], "turn": [0.33, 0.75, 1.25], "river": [0.33, 1.25]},
                  raise_sizes={"flop": _FLOP["raise"], "turn": [0.75], "river": [0.75]},
                  model=dict(river_buckets=32, turn_buckets=True, bucket_river_step=3, river_classes=True,
                             river_strength=RIVER_STRENGTH)), 2.5, 2.5),
    "accurate": (dict(bet_sizes={"flop": _FLOP["bet"], "turn": [0.33, 0.75, 1.25], "river": [0.33, 0.75, 1.25]},
                      raise_sizes={"flop": _FLOP["raise"], "turn": [0.75], "river": [0.75]},
                      model=dict(river_classes=True, river_strength=RIVER_STRENGTH)), 30.0, 12.0),
}


# A bet or raise leaving under 15% of the pot behind once called is the all-in (Config.force_allin; the browser's too).
FORCE_ALLIN = 0.15


def config(speed: str = "fast", depth: int = 2) -> Config:
    """The speed's tree (its model's flags included)."""
    sizes = SPEEDS[speed][0]
    kw = dict(sizes.get("model", {}))
    return Config(bet_sizes={k: list(v) for k, v in sizes["bet_sizes"].items()},
                  raise_sizes={k: list(v) for k, v in sizes["raise_sizes"].items()}, depth=depth,
                  force_allin=FORCE_ALLIN, **kw)


# 3-way pots solved by the 3-player solver (Advisor.mw_solve_streets; the rest use the betting rules), and 4-way flops
# and turns by the 4-player solve. Flops and turns bet the later streets by hand buckets (Advisor.mw_street_config,
# README 3-way pots), their iterations in the compiled multiway core (gto/mwcore.py). Measured on a quiet machine
# (CO opens, BTN and BB call; Kh8d3c): Accurate's 30 s settle a flop first to act at 400 iterations, 0.05% of the pot
# from equilibrium, in 21-25 s (its 16 buckets, the flop 33/75%), a turn in 7 s. An earlier 3-way flop is narrowed
# through your advice's solve of it; a solve of it from its start (Advisor.mw_prior_solve) stays off.
# Fast solves them too, in a smaller flop game (advisor.MW_FAST_STREETS) and 5 s (MW_FAST_BUDGET; 2.5 s ran a flop
# first to act 22 iterations in numpy, 17% of the pot from equilibrium): a flop first to act 111-125 iterations,
# 0.7-1.05%; checked to twice 400, 0.03%; a turn 248, 0.1%; a 4-way flop (Kh7d2c, HJ opens, three call) 200, 0.04%.
MW_STREETS = {"fast": ("flop", "turn", "river"), "accurate": ("flop", "turn", "river")}
MW_FAST_BUDGET = 5.0


def advisor(speed: str = "fast", depth: int = 2, budget: Optional[float] = None,
            prior_budget: Optional[float] = None, library: bool = True) -> Advisor:
    """`library`: answer flop spots the precomputed libraries hold from them (gto/flop_library.py, heads-up; and
    gto/mw_library.py, a 3-way flop's first decisions), at either speed."""
    b, pb = SPEEDS[speed][1:]
    adv = Advisor(config(speed, depth), time_budget=b if budget is None else budget,
                  prior_budget=pb if prior_budget is None else prior_budget)
    adv.root_sizes = {st: {k: list(v) for k, v in d.items()} for st, d in ROOT_SIZES.items()}
    adv.mw_solve_streets = MW_STREETS[speed]
    if speed == "fast":
        adv.mw_street_config = {st: dict(c) for st, c in MW_FAST_STREETS.items()}
        adv.mw4_street_config = {st: dict(c) for st, c in MW4_FAST_STREETS.items()}
        adv.mw_budget = MW_FAST_BUDGET
    adv.library = default_library() if library else None
    adv.mw_library = mw_library.default_library() if library else None
    adv.spare_prior = speed == "fast"
    if speed == "fast" and core.threads() >= THREADED_FAST:
        adv.cfg.bet_sizes["river"] = list(SPEEDS["accurate"][0]["bet_sizes"]["river"])
    elif speed == "fast":
        adv.spr_sizes = {st: (m, {k: list(v) for k, v in d.items()}) for st, (m, d) in FAST_SPR_SIZES.items()}
    return adv


def advisors(depth: int = 2, budget: Optional[float] = None, prior_budget: Optional[float] = None) -> Dict[str, Advisor]:
    """One advisor per speed (each keeps its own solves, under one memory budget: Advisor.cache_mb); budget/prior_budget
    override fast's."""
    advs = {"fast": advisor("fast", depth, budget, prior_budget), "accurate": advisor("accurate", depth)}
    advs["accurate"]._held = advs["fast"]._held
    return advs
