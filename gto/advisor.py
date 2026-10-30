"""Turn a game state into a recommendation.

Game state (all money in big blinds unless `bb` is given, then chips / bb):
{
  "players": 6,                                 # 2..9 (seat names from preflop.SEATS)
  "hero": "BTN", "hole": "AhKd", "board": "Ks7d2c",
  "stack": 100,                                 # everyone's stack
  "stacks": {"BTN": 40, "BB": 250},             # optional: per-seat stacks (seats left out have "stack")
  "preflop": [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "call"}, ...],
  "flop": [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 2.0}],
  "turn": [], "river": [],
  "sizes": {"flop": [33, 75], "raise": [60]},   # optional: postflop bet sizes per street, % of the pot
  "ranges": {"BB": "22+,A2s+,KTo+"},            # optional: a player's range entering the flop
  "optimize_sizes": true                        # optional: pick this street's bet sizes from a wide menu
  "one_size": "all"                             # optional: the one-size view instead (Advisor._one_size_view)
}
Actions: fold, check, call, bet (amount), raise (to), all-in.
"""
from __future__ import annotations
import dataclasses
import functools
import hashlib
import importlib
import random
import re
import time
import types
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from .cards import cards as parse_cards, COMBO_INDEX, hand_class, ALL_COMBOS, cards_str, card_str
from .ranges import N_COMBOS, range_to_text, parse_range
from . import preflop as pf
from .pf_classes import push_fold
# imported where used (exploit, breakdown, like multiway_solver and pf_game): the browser's worker imports this module
# before its first answer, and a preflop answer or a library flop needs none of them (tests/test_bundle.py)
from .solver import solve, Config, Solution, Decision, Terminal, Chance, BucketEntry, RiverEntry, match_action, expand, \
    MEMBER_FULL, _inv_of, Cancelled, TreeTooBig, TooSlow, tree_cells

DEFAULT_OPEN = {"SB": 3.0}
BVB_3BET = 10.0 / 3.0   # the big blind's default 3bet over a small-blind open, x the open (_default_size)
FOLD_SHADE = 0.75   # multiway: opponents fold this share of what a balanced defence would
# The next four are fitted to the heads-up solver by scripts/calibrate_multiway.py (RAISE_EQ barely
# mattered there, so it stays a judgment call).
NARROW_SOFT = 0.3   # multiway narrowing: weight kept by hands the rules say would have played differently
OFF_RULE_MAX = 0.4   # ...but those hands are at most this share of the narrowed range (a big bet's rule range is small)
# multiway: the small bet (share of the pot), beside 66%, for us and the opponents on every street.
# Solver data backs a third of the pot: GTO Wizard's 3-way flop study ("Playing In Position Against Two
# Callers", 100bb 8-max cEV, LJ opens, SB and BB call, 20 weighted flops, sizes 33/66/100%) has the opener
# checking 34.1%, betting 33% 56.2%, 66% 8.4% and pot 1.3% (heads-up vs the BB: 23.0/38.1/20.9/18.0), so
# 85% of its 3-way bets are the small size, where heads-up it was 49%. Larger multiway bets show up
# where the bettor has the nut advantage (GTO Wizard's 3-way launch post: BTN vs SB + BB on QsTs7h bets
# 75% when checked to). The field's betting rules (_bet_ranges) are fitted to that mix: the 6-max UTG
# opener into SB + BB checks 43%, bets 33% 51% and 66% 6% over 200 random flops (heads-up vs the BB
# 24/36/40); the old rules (the big bet the top of the range, the small one a thin layer next to it)
# checked 67% and bet 66% 26%, 33% 7% (heads-up 47/8/45). Hero's own size is picked by EV.
SMALL_BET = 0.33
MW_LATER = {"later": "buckets", "buckets": 16}   # 3-way flop and turn solves: the later streets by hand buckets
# 3-way flop solves bet a third or three quarters of the pot (the heads-up flop menu; the turn and river inside one size)
MW_FLOP_BETS = {"flop": [0.33, 0.75], "turn": 0.5, "river": 0.5}
# 3-way turn solves bet the heads-up turn menu, a third, three quarters or 125% of the pot (the river inside half): with
# the compiled core (gto/mwcore.py) a turn iteration costs 8 ms one size, 25-48 ms three, in the browser. Checked to
# on Kh8d3c9s (CO opens, BTN and BB call) the button bets 125% 30% of its range where it can (its bet 50%: 37%; with
# 33/75 75% 36%), and its value is 1.3% of the pot more (natively 300 iterations: 0.03-0.2% from equilibrium)
MW_TURN_BETS = {"flop": 0.33, "turn": [0.33, 0.75, 1.25], "river": 0.5}
MW_QUICK_BUDGET = 1.8   # Practice's quick answers: a 3-way pot's least budget (a river's tables take ~0.7 s in the
                        # browser: at 1.2 s it ran 4 iterations, 19% of the pot from equilibrium; at 1.5, 7-19, 2-8.5%)
# a quick answer's 3-way turn: 8 buckets (in the browser's 1.8 s, the compiled core: 87 iterations, 0.58% of the pot
# from equilibrium; 16 buckets in numpy ran 25, 7.1%)
MW_QUICK_TURN = {"later": "buckets", "buckets": 8, "bet": MW_TURN_BETS}
# a quick answer's 3-way flop the library doesn't hold (and natively a 4-way flop, Advisor.mw4_quick_flop): the smallest
# game, 6 hand buckets, the turn's tables over every sixth river, the flop betting half the pot (a pot with a player
# all-in keeps the street's own game: two bettors, small enough). In the browser's 1.8 s a flop first to act (Kh8d3c, CO
# opens, BTN and BB call) ran 114 iterations, 0.43% of the pot from equilibrium; checked to twice 200, 0.07%. Natively
# 88-128, 0.31-0.79%; a 4-way flop (Kh7d2c) 41, 1.07% (in the browser its tables took 1.6 s: 15 iterations, 9.4%)
MW_QUICK_FLOP = {"later": "buckets", "buckets": 6, "bucket_river_step": 6, "bet": {"flop": [0.5], "turn": 0.5, "river": 0.5}}
MW4_QUICK_FLOP = {"later": "buckets", "buckets": 6, "bucket_river_step": 6}
MW_RULES_MIN_S = 0.4    # the betting rules' trials take at least this long (400 trials: ~0.3 s natively)
MW_QUICK_BUILD = 0.75   # a quick answer's multiway tree and tables done by this share of its budget, or the rules answer
# Study's 3-way flop (one the 3-way library doesn't hold) gets longer than the advice's: browsing waits for an answer,
# and Fast's 5 s rarely settled one (natively on a loaded machine 12-23 iterations, 17-40% of the pot from equilibrium,
# over mw_max_expl: no answer); in 15 s 51-77, 1.1-1.9% (Qs8h3d, Ah9s9d, Tc9c8d, BTN open, SB and BB call)
MW_STUDY_FLOP_BUDGET = 15.0
# Fast's 3-way flop and turn solves: the flop in a smaller game than Accurate's (12 hand buckets, the turn's tables over
# every third river, the flop betting a third or three quarters of the pot: against the 32-bucket game played out to
# 400 iterations, its values 2.2-3.1% of the pot apart where 16 buckets on every river are 1.8-2.2 and 6 on every sixth
# 5.3-10.9; scripts/bench_mw_abstraction.py), the turn Accurate's. Measured (scripts/bench_browser.mjs; native 5 s,
# gto/native.py): a flop first to act (Kh8d3c, CO opens, BTN and BB call) in the browser's 8 s 203 iterations, 0.25% of
# the pot from equilibrium, natively 111-125, 0.7-1.05%; checked to twice 400, 0.03%; a turn 248-400, 0.04-0.1%
MW_FAST_STREETS = {"flop": {"later": "buckets", "buckets": 12, "bucket_river_step": 3, "bet": MW_FLOP_BETS},
                   "turn": dict(MW_LATER, bet=MW_TURN_BETS)}
# 4-way flops and turns (nobody all-in): the 4-player solve, the later streets by hand buckets (the root street per combo,
# card removal pair by pair: MultiwaySolver._term_pw), one bet a street and no raise but the player advised's
# (root_raise), as the N-player river solve
MW4_BETS = {"flop": 0.33, "turn": 0.5, "river": 0.5}
MW4_STREETS = {"flop": {"later": "buckets", "buckets": 8, "bucket_river_step": 3},
               "turn": {"later": "buckets", "buckets": 16}}
MW4_FAST_STREETS = {"flop": {"later": "buckets", "buckets": 6, "bucket_river_step": 4},
                    "turn": {"later": "buckets", "buckets": 8}}
HU_RESERVE = 1.0   # heads-up solves likewise keep one iteration's time back (the exploitability read after: ~1 iteration)
OVER_TARGET = 10   # an answer more than this many times its advisor's expl_target from equilibrium is approximate
HU_CHECKS = (50, 100, 150, 200, 300, 400, 600, 800)   # a heads-up street's solve, with Advisor.hu_settled set, stops at
                   # the first of these iterations where it is that close to equilibrium (an exploitability read each,
                   # about one iteration's time)...
HU_SETTLED_FROM = {"flop": 300, "turn": 300, "shallow": 300, "river": 300}   # ...from this one on (Advisor.hu_settled_from):
                   # per-combo EVs settle after the exploitability does. Against the rich reference
                   # (scripts/turn_river_reference.py, % of the pot off per combo): a river at 100 iterations, 0.15% of
                   # the pot from equilibrium, 0.99 where 300 and 1000 iterations are 0.61 and 0.60; a 3bet pot's turn
                   # (shallow: Advisor.spr_sizes) at 150, 0.27%, 1.60 against 0.69 and 0.68. "turn" is a deeper one: the
                   # browser's one-thread Fast plays its own menu there, 0.35 off at 200 and 0.34 from 300 on (worker.js)
# A heads-up tree's size in cells (solver.TreeBuilder.cells: its decisions' boards x actions x width) sets a solve's
# memory, 20-27 bytes a cell at the peak (a limped turn's 13.75M cells: 472 MB of heap in Pyodide, 527 MB natively;
# before the turn's equity leaf was built from the rivers' strengths, ~47 with 48 showdown matrices kept: a limped 200bb
# turn with three raise sizes peaked at 2.1 GB), and part of an iteration's time: 5-8 ns a cell on a big tree (that
# turn: 72 ms an iteration in Pyodide, 116 natively), 12-30 on a 2.7M-cell one with an iteration's other costs (34-84 ms
# in Pyodide, 13-47 natively). Quick's 1.5 s turn ran Solver.run's 10-iteration floor in 3-14 s at 0.7-2.1 GB in limped
# pots and a wide single-raised one, 42-85% of the pot from equilibrium. A new solve's tree is counted first
# (solver.tree_cells, no arrays) and cut (Advisor._trim_cfg, TRIMS) to the first level under Advisor.max_cells (memory:
# ~0.6 GB of tree; Accurate's limped turns are ~20M, its single-raised ones 7.6M) and, on the turn and river, under what
# runs CAP_ITERS iterations in the budget at Advisor.cell_s seconds a cell (CELL_S, lowered by the big trees solved
# here); a tree whose first iteration says the floor would take over SLOW x the budget (solver.TooSlow) sets cell_s
# from it and is cut to the budget left.
MAX_CELLS = 24_000_000
CELL_S = 8e-9
CAP_ITERS = 30     # (at 10, the floor, a limped turn's uncut tree ran 19 iterations in Quick's budget: 38% of the pot
                   # from equilibrium; cut to fit 30, 41 iterations, 8.0% in its own game)
SLOW = 2.0
TRIMS = ("one raise size", "one raise size, and one bet size on the later streets",
         "one bet size and one raise size a street", "one bet size and one raise size a street, and one raise")
MW_SETTLED = 0.1   # a 3-way solve stops once its players' mean best-response gain is under this (% of the pot)...
MW_CHECKS = (25, 50, 100, 200, 400, 700)   # ...checked at these iterations (about two iterations' time each). The
                   # browser's Accurate ran a 3-way river 60 s to 386 iterations, 0.01% of the pot, and a turn 330, 0.03%
MW_RESERVE = 2.5   # 3-way solves stop this many iterations' time short of the budget: best-response gains and EVs after
XR_Z = 2.0          # multiway: a check-raise beats calling/folding by this many standard errors (see chk_plan)
XR_MIN = 30         # ...over at least this many trials of that situation
LATER_CACHE = 40000  # multiway: most later-street decisions stored per call (memory bound, ~25 KB each at most)
# Solves kept for reuse (Advisor.cache_mb): a heads-up turn's holds ~200 MB (its rivers' equity tables), an Accurate
# one's 240-330, a 3-way turn's ~10-100; kept without bound, the browser's worker passed its 2 GB heap after a few
# hands (12 answers: Fast and Accurate turns and rivers) and every later solve failed for want of memory.
CACHE_MB = 1000.0
CACHE_KEEP = 2      # ...the latest this many stay whatever they hold (the solve being answered from and the one before)
HIT_TOP = 0.15      # multiway bluffs: a hand "gets there" on a runout ending in this top share of combos (judgment)
SHOWDOWN_Q = 0.6    # ...and has showdown value (bluffs last) when it usually ends above this share (judgment)
VALUE_SHARE = 0.35  # multiway: a bet's value range is this share of what would call it
BEATS_NOTE_MAX = 0.05  # river, facing a bet: say so when our hand beats at most this share of the bettor's range
SMALL_SHARE = 0.7   # multiway: the small bet's value range is this share of what one player would call it with,
                    # compounding with more players (1 - 0.3^n of what the field calls); fitted, see _bet_ranges
SMALL_VALUE_MAX = 0.55  # ...but at most this share of the bettor's range: behind two checks (capped ranges) the
                        # compounding made nearly all of it value and the button bet 93-100%, so AA checked to
                        # check-raise it (check +18.8 bb over a bet's +15.3; now 13.5 / 13.4, the button betting 66%);
                        # the fitted UTG-into-both-blinds mix over 200 flops moves from 42 / 44 / 14 to 45 / 41 / 14
NUT_TOP = 0.1       # multiway: the nuts, for the big bet: this top share of the players' ranges pooled
ADV0, ADV_SPAN = 0.4, 0.2  # ...the big bet starts at this share of the nuts and takes all the value over this much more
RAISE_SHARE = 0.2   # multiway: facing a bet, this share of what continues (the strongest) raises
CHECKED_RAISE_SHARE = 0.05  # ...and of a player who checked to our bet (a capped range: the check kept few raises), at the
                   # rules' root in 4+ way pots (3-way ones keep RAISE_SHARE). Scored against 4-way turn solves
                   # (scripts/score_multiway4.py: six boards, 40 hands a line; only the advised player may raise there):
                   # checked to last, the rules bet 36% of hands for the solve's 53% and gave up 0.77% of the pot to its
                   # mix; at 0.05 they bet 53%, giving up 0.35% (0.1: 48%, 0.65%), the other lines unchanged
CHECK_SOFT = 0.7   # multiway narrowing: a flop check keeps this much of the hands the rules have betting (NARROW_SOFT
                   # elsewhere; capped by OFF_RULE_MAX as it is). 3-way solves from the SB's action (UTG opens, both blinds
                   # call; ten flops) check through with 73-91% of the BB's range where the rules kept 68-80%: the rules'
                   # check ranges were 0.21 (SB) and 0.14 (BB) off the solve's in total variation, now 0.15 and 0.05, and
                   # UTG's solve in front of them bet 72.5% (five flops) where the full street's bets 49%, now 51.5%
OVERCALL = 0.25    # multiway narrowing: a call behind another caller continues with this share of what the first would. In
                   # 3-way solves (UTG bets a third into both blinds, the SB calls 53-78%) the BB overcalls 3.5-20% of its
                   # range (13% on average, eight flops), where the first caller's share kept 57% (compare_multiway.py
                   # --narrowing: the turn's ranges 0.52 off -> 0.44, and 0.52 -> 0.46 on four other flops)
SLOWPLAY = 0.5      # multiway narrowing: weight a call keeps on the hands RAISE_SHARE has raising (judgment)
SLOWPLAY_AGAIN = 0.85  # ...and on a later street's call, for a hand that already slowplayed one (judgment)
LEAD_EQ = 0.5       # multiway, later streets: we lead when our equity beats this vs every calling range
RAISE_EQ = 0.55     # multiway, later streets: we raise their bet when our equity beats this vs what continues
ROOT_RAISE_EQ = 0.65  # multiway, 4+ players facing a bet and a call: the trials' raise is picked only when our equity
                   # beats this against what each player continues with. Their later streets pay a raise off too often
                   # (opponents call our river value bets by a balanced defence): against the 4-way references (see
                   # CHECKED_RAISE_SHARE) the rules raised AT-AQ on AcKd5c5s for +3.9bb over calling where the solve
                   # (which lets nobody re-raise, so if anything overvalues raising) calls for +2.8bb. Facing a bet and
                   # a call 0.65 gives up 1.49% of the pot to the solve's mix where none gave up 1.89% (0.55: 2.08,
                   # 0.6: 1.65, 0.7: 1.49); facing a lone bet any bar did worse (0.57 -> 0.64-0.70), so it isn't used there
RARE_LINE = 0.10    # heads-up: widen the villain's range when the solve takes their line less often than this
UNSETTLED = 3.0     # heads-up: a solve this far from equilibrium (% pot) hasn't settled (the page's "converged" cutoff)
# Heads-up, earlier streets: narrowing follows a solve fully this close (% pot) to equilibrium, not at all this far,
# linearly between; the rest of each action's weights are the betting rules' range for it (softened, as multiway),
# or for a check the whole range (no evidence: the big blind checks nearly every flop to the raiser).
# An unsettled solve's lines are far from its equilibrium's: BTN vs BB K72 c-bet called, T turn bet called, BB's
# river check range is 69.5 combos after long solves, and a flop at 20 iterations (9% of the pot, BB checking 61%
# and calling the c-bet 21% where it settles at 95% and 47%) left 24, 20 iterations and a 26-iteration turn (6%)
# 18 (a loaded machine's 2.5 s; 4% of its preflop range). Followed by how settled they are (checks then narrowed
# by the rules too): 89 there, 55-65 with 40-iteration flops (2.6%, but its BB still checks 75%, so the flop's ramp
# starts low; were 41-43), 63-72 at 100-300. Fast's bucketed flop (river_buckets) is further off at a given count (20 iterations 27%, 40 7.7% with
# BB checking 87% and calling 33% for 99% and 50%, 80 1.8%, 150 0.5%); long solves keep 87.7 there, short ones
# kept 29 (20 flop, 26 turn iterations), 50 (40/26), 52 (40/45), 70-82 (80-150/45) and now keep 112, 112, 87,
# 78-81. A settled solve (the accurate speed's, or fast's unloaded: flop 140-210 iterations, turn 1.5-2%) is
# followed fully. Short solves err wide, not thin.
PRIOR_TRUST = {"flop": (0.5, 5.0), "turn": (2.0, 10.0), "river": (2.0, 10.0)}
SETTLE_EXPL = 10.0  # heads-up: the street being advised on gets one more time budget when this far (% pot) from settled
JAM_BLEND = 0.04    # preflop vs a jam: call frequency ramps from 0 to 1 over this much equity around the pot odds
PREFLOP_BOARDS = 400  # boards sampled for preflop all-in equity
MULTI_BOARDS = 400    # ...of which preflop equity against several ranges uses this many
PRICE_BLEND = 0.02  # preflop, off-chart price: continue frequency ramps over this much (scaled) equity around the pot odds
RAISE_BLEND = 0.03  # preflop, more players in: value-raise frequency ramps over this much equity when called
COMMIT = 0.4       # preflop: a raise that would put in this share of the stack goes all-in instead
JAM_SPR = 1.5      # ...as does one leaving less than this many pots behind once called
CHECK_REALIZE = 0.8  # preflop, short stacks: the share of its equity the big blind's free check over limpers realizes
JAM_SPR_DEEP = 1.25 # ...this many at the charts' own depth (within DEPTH_TOL of CHART_STACK), where a 4bet to 2.3x
                    # leaves ~1.45 pots behind and is called, as in 100bb solves
ICM_COMMIT = 1.0   # tournament: a call's risk premium is the all-in's times min(1, this / (1 + SPR after it))
OPEN_WIDEN = 1.5   # bounties: a discount on an open widens it to at most this many times the chart's raises
PUSH_FOLD_STACK = 12.0  # preflop: at or below this many bb, raising first in is all-in (push/fold)
CHART_STACK = 100.0  # the depth the preflop charts are written for
IMPLIED = {"pair": 0.06, "suited": 0.03}  # preflop calls: share of the stack behind (beyond the chart's) small pairs / suited hands expect to win later
DEPTH_TOL = 20.0   # ...a spot this many bb deeper or shallower than the charts re-prices calls
PRICE_TOL = 0.02    # ...and a spot within this much of the chart's pot odds, against the raisers only, is the chart's own
CUT_FLOOR = 0.05    # ...a worse price holds the chart's continuing hands to the equity of its weakest this share of them
LOCK_BLUFF_MAX = 1.5  # preflop, your locks: hero's bluff raises grow at most this much against a player who folds more (as the pool's)
EV_GAP = 1.0        # heads-up, unsettled solve: recommend the highest EV when it beats the top of the mix by this % of pot
BR_TIE_MIN = 0.005  # exploit mode: best-response EVs this close (share of pot; more when the solve is unsettled) count as tied
COMMIT_BET = 0.75   # exploit mode: a postflop bet this big (share of the pot) marks its maker committed: no extra folds
                    # (a 96%-pot donk on AKQ left 4-high calling for its river bluffs, +1.4 bb)
BR_TIE_MAX = 0.05   # ...and at most this: a 10-iteration solve 41% of the pot from settled tied every action, so
                    # its unsettled mix folded 55 to a turn check-raise where calling was +6.7 bb
MAX_SIZES = 3       # custom/optimized sizes: at most this many per street (each multiplies the tree; the browser's is small)
SIZE_PCT = (5.0, 500.0)  # custom sizes are held to this range, % of the pot
OPT_SIZES = [0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0]  # optimize_sizes: the wide menu the first solve picks from
OPT_KEEP = 0.03     # ...a size stays when either player's first bet uses it with this share of their range
OPT_SHARE = 0.35    # ...that solve's share of the time budget (the advice's solve gets the rest: same total)
# The one-size view (state "one_size"; Advisor._one_size_view): at a heads-up decision where the player can bet, the
# street solved from there with one bet size at that decision, for each of these, and with all of them to mix freely
# (the full menu); the rest of the tree is the advice's. A quarter pot to 1.5 pots: the sizes single-size strategies
# use (a range bet small, a polar range big), closer together where a few points of pot move the EV most; the river
# twice the pot too (a polar river's one size is the biggest: the advice's own river bets up to 250%). Plus all-in
# where the tree offers it (the stack at most allin_spr pots: a 3bet pot's turn and river). A size reaching the stack
# is the all-in.
ONE_SIZES = (0.25, 0.33, 0.5, 0.75, 1.0, 1.25, 1.5)
ONE_SIZES_RIVER = ONE_SIZES + (2.0,)
ONE_SIZE_ALLIN = 1000.0   # (the all-in candidate: a bet of this many pots reaches any stack)
# ...each solve stops at the first of HU_CHECKS where it is this close to equilibrium (% of the pot; expl_target when
# lower), or at its budget (the request's): the EV given up is the difference of two such solves' values, so sizes
# closer than this are a tie. Half of Fast's 0.2 (at 0.2 most of a turn's sizes tied): a page's turn 1.3-16 s for
# every size, with threads (scripts/bench_browser.mjs --one-size)
ONE_SIZE_SETTLED = {"flop": 0.1, "turn": 0.1, "river": 0.05}
ONE_SIZE_CACHE = 256    # rows kept per advisor (a few KB each: a grid and the numbers), least recently used out first
ONE_SIZE_DEF = ("best = the size whose one-size strategy keeps the most EV for the player betting: each game is solved "
                "from this decision with the same ranges, the opponent adapting; EV given up = the range's EV with "
                "every candidate size to mix freely minus with that size alone, in % of the pot")
STREET_OF = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
STREETS = ["preflop", "flop", "turn", "river"]


def _custom_sizes(raw) -> Dict[str, List[float]]:
    """state["sizes"]: {"flop"|"turn"|"river"|"raise": [% of pot, ...]} -> fractions, sorted, held to
    SIZE_PCT. Raises ValueError with a message for the user."""
    if not isinstance(raw, dict):
        raise ValueError('sizes must be an object like {"flop": [33, 75], "raise": [60]} (percent of the pot)')
    out = {}
    for k, v in raw.items():
        if k not in ("flop", "turn", "river", "raise"):
            raise ValueError(f"sizes: unknown street {k!r} (use flop, turn, river or raise)")
        try:
            vals = [float(x) for x in (v if isinstance(v, list) else [v])]
        except (TypeError, ValueError):
            vals = [-1.0]
        if any(not x > 0 for x in vals):   # (not x > 0 also catches nan)
            raise ValueError(f"sizes: {k} needs positive numbers (percent of the pot), got {v!r}")
        if len(set(vals)) > MAX_SIZES:
            raise ValueError(f"sizes: at most {MAX_SIZES} {k} sizes (each one multiplies the game tree), got {len(set(vals))}")
        if vals:
            out[k] = sorted({round(min(max(x, SIZE_PCT[0]), SIZE_PCT[1]) / 100, 3) for x in vals})
    return out


def _custom_ranges(raw, positions) -> Dict[str, np.ndarray]:
    """state["ranges"]: {seat: PIO-style range} -> weight vectors. Raises ValueError for the user."""
    if not isinstance(raw, dict):
        raise ValueError('ranges must be an object like {"BB": "22+,A2s+,KTo+"}')
    out = {}
    for p, text in raw.items():
        if not str(text or "").strip():
            continue   # an empty box: the chart's range
        if str(p).upper() not in positions:
            raise ValueError(f"ranges: {p} is not a seat at this table ({', '.join(positions)})")
        try:
            r = parse_range(str(text))
        except (ValueError, IndexError, KeyError, TypeError, AttributeError):
            r = None
        if r is None or not (r > 0).any() or (r < 0).any():
            # (the text isn't echoed: the page shows errors as HTML, and a share link carries ranges)
            raise ValueError(f"ranges: can't read {p}'s range (PIO style: AA, KK+, AKs, A5s+, T9s-65s, 22-77, AhKh, QJo:0.5)")
        out[str(p).upper()] = np.minimum(r, 1.0)
    return out


def _state_stacks(stack_raw, raw, positions, bb: float) -> Tuple[float, Optional[Dict[str, float]]]:
    """(the table's stack, per-seat stacks or None when everyone's is the same), in bb, from the state's "stack"
    and the optional state["stacks"] ({seat: chips}; seats left out have "stack", or without one the
    smallest stack given, as when "stacks" set one effective stack). Raises ValueError for the user."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError('stacks must be an object like {"BTN": 40, "BB": 250}')
    given = {}
    for p, v in raw.items():
        p = str(p).upper()
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = float("nan")
        if p not in positions:
            raise ValueError(f"stacks: {p} is not a seat at this table ({', '.join(positions)})")
        if not x > 0:
            raise ValueError(f"stacks: {p}'s stack must be a positive number")
        given[p] = x / bb
    if stack_raw is not None:
        stack = float(stack_raw) / bb
    else:
        stack = min(given.values()) if given else 100.0
    if not stack > 0:
        raise ValueError("stack must be a positive number")
    stacks = {p: given.get(p, stack) for p in positions}
    if all(abs(v - stack) < 1e-9 for v in stacks.values()):
        return stack, None
    return stack, stacks


def _num_list(raw, what: str) -> List[float]:
    """A list of numbers from a list or a comma-separated string. Raises ValueError for the user."""
    if isinstance(raw, (int, float)):
        raw = [raw]
    if isinstance(raw, str):
        raw = [x for x in raw.replace(";", ",").replace(" ", ",").split(",") if x.strip()]
    try:
        out = [float(x) for x in raw]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be numbers like 50, 30, 20")
    if any(not np.isfinite(x) or x < 0 for x in out):
        raise ValueError(f"{what} must be numbers of 0 or more")
    return out


_BOUNTY_ALIASES = {"knockout": "ko", "progressive": "pko", "progressive knockout": "pko", "mb": "mystery",
                   "mystery bounty": "mystery"}


def _bounties(state: dict, positions):
    """A bounty tournament's bounties (gto/icm.py Bounties) from the state's "bounties", in the payouts' units:
    {"format": "ko" | "pko", "seats": {seat: bounty} (or one number for every seat, or a list in seat order),
    "field": the other tables' average bounty} or {"format": "mystery", "envelope": the average envelope left} (or
    "pool" and "left": the money in envelopes and how many). None without, or with nothing at stake. Raises ValueError
    for the user."""
    spec = state.get("bounties")
    if not spec:
        return None
    if not isinstance(spec, dict):
        raise ValueError('bounties must be {"format": "ko", "pko" or "mystery", "seats": {seat: bounty}} '
                         '(mystery: "envelope", or "pool" and "left")')
    from . import icm
    fmt = str(spec.get("format") or "ko").strip().lower()
    fmt = _BOUNTY_ALIASES.get(fmt, fmt)
    if fmt not in icm.BOUNTY_FORMATS:
        raise ValueError(f"bounties: the format is ko, pko or mystery, not {spec.get('format')!r}")

    def num(k):
        v = spec.get(k)
        if v in (None, ""):
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"bounties: {k} must be a number")
        if not np.isfinite(x) or x < 0:
            raise ValueError(f"bounties: {k} must be 0 or more")
        return x
    if fmt == "mystery":
        env, pool, left = num("envelope"), num("pool"), num("left")
        if env is None and not (pool and left):
            raise ValueError("mystery bounties: give the average envelope left, or the pool and the bounties left")
        b = icm.Bounties("mystery", envelope=env, pool=pool or None, left=left)
        return b if b else None
    seats = spec.get("seats")
    if isinstance(seats, dict):
        vals = {}
        for p, v in seats.items():
            p = str(p).upper()
            if p not in positions:
                raise ValueError(f"bounties: {p} is not one of {', '.join(positions)}")
            vals[p] = _num_list(v, "bounties")[0] if not isinstance(v, (int, float)) else float(v)
            if not np.isfinite(vals[p]) or vals[p] < 0:
                raise ValueError("bounties must be numbers of 0 or more")
    else:
        xs = _num_list(seats if seats is not None else [], "bounties")
        if len(xs) == 1:
            xs = xs * len(positions)
        if len(xs) != len(positions):
            raise ValueError(f"bounties: one for each of the {len(positions)} seats ({', '.join(positions)}), "
                             f"or one for all")
        vals = dict(zip(positions, xs))
    b = icm.Bounties(fmt, vals, num("field"))
    return b if b else None


def _tournament(state: dict, positions, stack: float, stacks, bb: float):
    """The tournament's prize model (gto/icm.py) from the state's "payouts" (the prizes still to be paid, first place
    first, in any unit) and optional "field": the other tables' stacks in chips, as a list or {"players": n, "chips":
    total} (an even split). None when there are no payouts, or one (winner-take-all: chips are prizes). Stacks in bb.
    With bounties (_bounties) a model all the same: one prize is winner-take-all, and none takes the prizes left as
    equal to the bounties in play (a buy-in split evenly between them), chips in chip EV."""
    bounty = _bounties(state, positions)
    if not state.get("payouts") and bounty is None:
        return None
    pay = [x for x in _num_list(state["payouts"], "payouts") if x > 0] if state.get("payouts") else []
    field = state.get("field") or []
    if isinstance(field, dict):
        try:
            n_f, chips = int(field.get("players") or 0), float(field.get("chips") or 0.0)
        except (TypeError, ValueError):
            raise ValueError('field must be the other tables\' stacks, or {"players": n, "chips": total}')
        if n_f < 0 or chips < 0 or (n_f and chips <= 0):
            raise ValueError("field: the players at other tables need chips")
        others = [chips / n_f / bb] * n_f if n_f else []
    else:
        others = [x / bb for x in _num_list(field, "field") if x > 0]
    if len(pay) <= 1 and bounty is None:
        return None
    from . import icm
    seats = {p: float(stacks.get(p, stack)) if stacks else float(stack) for p in positions}
    if len(pay) > len(seats) + len(others):
        pay = pay[:len(seats) + len(others)]
    if not pay:
        # no payouts: chip EV, the prizes left taken as the bounties in play (the pool the model works out)
        pay = [icm.Model(seats, [1.0], others, bounty).bounty_pool]
        model = icm.Model(seats, pay, others, bounty)
        model.assumed_prizes = True
        return model
    return icm.Model(seats, pay, others, bounty)


def _tournament_info(model, hero: str, positions, street, out=()) -> dict:
    """What the answer says about the tournament: hero's prize equity (share of the prizes left) and the bubble
    factor against each opponent (what a chip lost costs over what one won gains, all in against them); in a bounty
    tournament the bounties' part (_bounty_info; `out`: the seats that have folded)."""
    eq = model.base[hero] / model.real_pool
    bf = {}
    for p in positions:
        if p != hero:
            b = model.bubble_factor(hero, p)
            bf[p] = round(b, 2) if np.isfinite(b) else None
    top = max((v for v in bf.values() if v), default=1.0)
    approx = f" ({model.players} players modelled by {len(model.seats) + len(model.field)})" if model.approx else ""
    if getattr(model, "assumed_prizes", False):
        note = ("tournament: no payouts given, so chip EV: the prizes left are taken as equal to the bounties in play "
                "(a buy-in split evenly between the prize pool and the bounties) to price chips against bounties")
    elif len(model.payouts) == 1 and not model.approx:
        note = f"tournament: winner-take-all (prize equity is chips), your equity {eq:.1%} of the prize pool left"
    else:
        note = (f"tournament: ICM{approx}, your prize equity {eq:.1%} of what's left to pay; bubble factor up to {top:.2f} "
                f"(a chip lost costs {top:.2f}x what one won gains)")
    res = {"equity_share": round(eq, 4), "bubble_factors": bf, "note": note}
    if model.bounty is not None:
        res["bounties"] = _bounty_info(model, hero, positions, out)
    return res


def _bounty_info(model, hero: str, positions, out=()) -> dict:
    """A bounty tournament's part of the answer: the format and model, chips per unit of money (bounty_k: bb per unit
    of the payouts), the pool in play, each seat's bounty, what busting each opponent hero covers is worth to hero
    (you_bust: in bb and money, the cash it pays less hero's chip share of it after the hand; in_hand: still in, so
    what hero's call or shove here can win), and what hero's own bounty is worth to each opponent who covers hero
    (bust_you, bb)."""
    from .icm import BOUNTY_FORMATS
    b = model.bounty
    you_bust, bust_you = {}, {}
    for p in positions:
        if p == hero:
            continue
        v = model.bust_value(hero, p)
        if v is not None:
            you_bust[p] = {"bb": round(v[0], 2), "value": round(v[1], 4), "cash": round(model.cash[p], 4),
                           "in_hand": p not in out}
        v = model.bust_value(p, hero)
        if v is not None:
            bust_you[p] = round(v[0], 2)
    k = model.bounty_k
    what = {"ko": "the whole bounty to whoever busts a player",
            "pko": "half the bounty at once to whoever busts a player, half added to their own bounty"}.get(b.fmt) \
        or f"an envelope from the pool for each player busted, {b.envelope or 0:g} on average"
    live = {p: d for p, d in you_bust.items() if d["in_hand"] and d["bb"] > 0.005}
    if live:
        gain = "; ".join(f"busting {p} (you cover) pays {d['cash']:g}, worth {d['bb']:.1f}bb to you"
                         for p, d in sorted(live.items(), key=lambda x: -x[1]["bb"]))
    else:
        gain = "you cover nobody still in the hand who carries a bounty"
    risk = [p for p in bust_you if p not in out]
    note = (f"bounties: {BOUNTY_FORMATS[b.fmt]} ({what}), valued by the proportional model (each player owns their "
            f"chip share of the {model.bounty_pool:g} in bounties in play, so 1 is worth {k:.3g}bb: GTO Wizard's bounty "
            f"power{'; the half added to your head counts at your chip share of it' if b.fmt == 'pko' else ''}): {gain}"
            + (f"; {', '.join(risk)} cover{'s' * (len(risk) == 1)} you (your bounty {model.bounties[hero]:g})"
               if risk and model.bounties.get(hero) else ""))
    return {"format": b.fmt, "model": "proportional", "bb_per_unit": round(k, 4), "pool": round(model.bounty_pool, 4),
            "bounties": {p: round(model.bounties[p], 4) for p in model.seats}, "you_bust": you_bust,
            "bust_you": bust_you, "note": note}


def _norm_actions(acts: List[dict], bb: float) -> List[dict]:
    out = []
    for a in acts or []:
        d = dict(a)
        d["pos"] = str(d["pos"]).upper()
        act = str(d["action"]).lower().replace("allin", "all-in").replace("shove", "all-in").replace("jam", "all-in")
        act = {"x": "check", "f": "fold", "c": "call", "r": "raise", "b": "bet", "ai": "all-in"}.get(act, act)
        d["action"] = act
        for k in ("to", "amount"):
            if k in d and d[k] is not None:
                d[k] = float(d[k]) / bb
        out.append(d)
    return out


# ----------------------------------------------------------------------------- hand replay
@dataclass
class StreetState:
    name: str
    order: List[str]                 # acting order this street
    players_start: List[str]         # not folded at the start of the street, in acting order
    pot_start: float                 # chips in the middle at the start of the street
    inv_start: Dict[str, float]      # total invested per player at the start
    bets: Dict[str, float] = field(default_factory=dict)   # committed this street
    cur_bet: float = 0.0
    n_raises: int = 0
    acted: set = field(default_factory=set)
    active: set = field(default_factory=set)               # not folded after the actions so far
    allin: set = field(default_factory=set)
    to_act: Optional[str] = None
    closed: bool = False
    actions: List[dict] = field(default_factory=list)
    posted: Dict[str, float] = field(default_factory=dict)   # blinds (and straddle) posted this street
    # the last bet or raise: the pot just before it, the chips its maker added, and how many players
    # (live and not all-in, the maker aside) faced it
    bet_pot: float = 0.0
    bet_risk: float = 0.0
    bet_faced: int = 0
    # no-limit raise rules: the last full raise's increment (a raise has to add at least this much), and the
    # players who may not raise: they acted since the last full raise (an all-in raise for less than a full
    # one doesn't reopen the betting to them)
    min_raise: float = 1.0
    locked: set = field(default_factory=set)
    returned: Dict[str, float] = field(default_factory=dict)   # uncalled chips given back when the street closed


def _table_stacks(positions, stack: float) -> Dict[str, float]:
    """Each seat's stack in bb: the table's per-seat stacks (Table.stacks), else everyone has `stack`."""
    s = getattr(positions, "stacks", None)
    return {p: float(s.get(p, stack)) if s else float(stack) for p in positions}


def _eff_stack(positions, p: str, actions: List[dict], stack: float) -> float:
    """p's effective stack preflop after `actions`: its own, capped by the deepest opponent who hasn't
    folded (the depth p's charts, push/fold and implied odds play at). `stack` without per-seat stacks."""
    s = getattr(positions, "stacks", None)
    if not s:
        return float(stack)
    folded = {a["pos"] for a in actions if a["action"] == "fold"}
    others = [float(s.get(q, stack)) for q in positions if q != p and q not in folded]
    own = float(s.get(p, stack))
    return min(own, max(others)) if others else own


def can_raise(st: "StreetState", p: str) -> bool:
    """Whether p may raise (or bet) now: not locked by an incomplete raise, and someone left with chips to
    face it."""
    return p not in st.locked and any(q != p and q not in st.allin for q in st.active)


def _showdown_share(st, idx, hands) -> float:
    """Hero's (combo idx) share of the pot at showdown against these opponent combos, strengths st."""
    best = max(st[j] for j in hands)
    if st[idx] != best:
        return float(st[idx] > best)
    return 1.0 / (1 + sum(st[j] == best for j in hands))


def replay_hand(positions: List[str], stack: float, streets: Dict[str, List[dict]], n_streets: int) -> List[StreetState]:
    """Replay preflop..current street. Sizes: bet=amount added this street, raise=to. Each player's chips
    are their stack (Table.stacks, else `stack`): a call or all-in for less puts in what they have, a bet
    nobody can match in full is given back down to the most anyone else put in once the street closes,
    and an all-in raise for less than a full raise doesn't reopen the betting (StreetState.locked)."""
    stk = _table_stacks(positions, stack)
    inv = {p: min(getattr(positions, "ante", 0.0), stk[p]) for p in positions}   # antes: dead money in the pot from the start
    active = set(positions)
    allin: set = set()
    out: List[StreetState] = []
    post = pf.postflop_order(positions)
    for si in range(n_streets):
        name = STREETS[si]
        pre = name == "preflop"
        order = (positions.preflop_order() if hasattr(positions, "preflop_order") else list(positions)) if pre else post
        st = StreetState(name, order, [p for p in order if p in active], sum(inv.values()), dict(inv))
        st.bets = {p: 0.0 for p in positions}
        if pre:
            st.posted = positions.posted() if hasattr(positions, "posted") else (
                {"SB": 0.5, "BB": 1.0} if "SB" in positions else {"BB": 1.0})
            for p, b in st.posted.items():
                st.bets[p] = min(b, stk[p] - inv[p])
            st.cur_bet = max(st.posted.values())
            st.min_raise = st.cur_bet
        st.active, st.allin = set(active), set(allin)
        # (a seat whose ante or blind took all its chips is all-in: it has nothing left to act with)
        st.allin |= {p for p in st.active if st.bets[p] + inv[p] >= stk[p] - 1e-9}
        last = None
        for a in streets.get(name, []):
            p, act = a["pos"], a["action"]
            if p not in st.active or p in st.allin:
                continue
            remaining = stk[p] - inv[p] - st.bets[p]
            if act == "fold":
                st.active.discard(p)
            elif act in ("check", "call"):
                st.bets[p] = min(st.cur_bet, st.bets[p] + remaining)
                if st.bets[p] + inv[p] >= stk[p] - 1e-9:
                    st.allin.add(p)
                st.acted.add(p)
                st.locked.add(p)
            elif act in ("bet", "raise", "all-in"):
                if act == "all-in":
                    to = st.bets[p] + remaining
                elif a.get("to") is not None:
                    to = a["to"]
                elif a.get("amount") is not None:
                    to = a["amount"]
                else:
                    to = _default_size(st, p, pre)
                to = min(to, st.bets[p] + remaining)
                if to > st.cur_bet + 1e-9:
                    st.bet_pot = st.pot_start + sum(st.bets.values())
                    st.bet_risk = to - st.bets[p]
                    st.bet_faced = sum(1 for q in st.active if q != p and q not in st.allin)
                st.bets[p] = to
                if to > st.cur_bet + 1e-9:
                    if to - st.cur_bet >= st.min_raise - 1e-9:   # a full raise reopens the betting
                        st.min_raise = to - st.cur_bet
                        st.locked = {p}
                    else:
                        st.locked.add(p)
                    st.cur_bet = to
                    st.n_raises += 1
                    st.acted = {p}
                else:
                    st.acted.add(p)
                    st.locked.add(p)
                if st.bets[p] + inv[p] >= stk[p] - 1e-9:
                    st.allin.add(p)
            st.actions.append(a)
            last = p
        actors = [p for p in st.active if p not in st.allin]
        # (one player left with chips who has matched the bet has nobody to bet against)
        st.closed = (len(st.active) <= 1 or all(p in st.acted for p in actors)
                     or (len(actors) <= 1 and all(st.bets[p] >= st.cur_bet - 1e-9 for p in actors)))
        if st.closed and len(st.active) > 1:
            # a bet nobody matched in full goes back down to the most anyone else put in
            top = max(st.bets, key=st.bets.get)
            second = max((b for q, b in st.bets.items() if q != top), default=0.0)
            if st.bets[top] > second + 1e-9:
                st.returned = {top: st.bets[top] - second}
                st.bets[top] = second
                st.cur_bet = second
                if inv[top] + second < stk[top] - 1e-9:
                    st.allin.discard(top)
        if not st.closed:
            start = (order.index(last) + 1) % len(order) if last else 0
            for k in range(len(order)):
                q = order[(start + k) % len(order)]
                if q in st.active and q not in st.allin and q not in st.acted:
                    st.to_act = q
                    break
        out.append(st)
        for p in positions:
            inv[p] += st.bets[p]
        active, allin = set(st.active), set(st.allin)
        if not st.closed:
            break
    return out


def _study_next(positions, stack: float, streets: Dict[str, List[dict]], n_board: int):
    """Study mode (Advisor.study): the seat to act after the hand's actions, else what the hand needs next:
    {"study": "deal", "street"} once a street's betting closes ("allin": nobody can bet any more), {"study": "over"}
    when one player is left or the river's betting closed."""
    n = {0: 1, 3: 2, 4: 3, 5: 4}[n_board]
    rep = replay_hand(positions, stack, streets, n)
    if len(rep) < n:
        return {"error": "an earlier street's betting is not finished"}
    cur = rep[-1]
    if len(cur.active) <= 1:
        return {"study": "over", "street": cur.name, "winner": next(iter(cur.active), None)}
    if cur.closed:
        if n == 4:
            return {"study": "over", "street": "river", "showdown": True}
        return {"study": "deal", "street": STREETS[n], "allin": sum(p not in cur.allin for p in cur.active) <= 1}
    return cur.to_act


# ----------------------------------------------------------------------------- multiway range rules
# Not a solve: simple rules for how players with ranges act once 3+ are in the pot. Used by
# Advisor._multiway and to narrow ranges through multiway streets (see _narrow_street).

def _top(r: np.ndarray, frac: float, rank: np.ndarray) -> np.ndarray:
    """Mask of the strongest combos (in `rank` order) holding `frac` of range r's weight."""
    m = np.zeros(N_COMBOS, dtype=bool)
    if frac > 1e-9 and r.sum() > 0:
        cum = np.cumsum(r[rank])
        m[rank[:int(np.searchsorted(cum, frac * cum[-1] - 1e-9)) + 1]] = True
    return m


def _aggro(r, v, b, p, rank, bluff=None, n=1) -> np.ndarray:
    """A bet or raise of b into p: value with the top v of r plus balanced bluffs,
    bluffs = value * b / (p + b), taken in `bluff` order (see _bluff_order: draws first) from the
    hands outside the value range; without one, from the bottom. Into n players, a bluff has to get past
    all of them: 1/n of those bluffs (a check-raise jam into two players was half bluffs, AQs and KQs on
    J72 beside JJ, and QQ called it with AA and KK still behind)."""
    val = _top(r, v, rank)
    bf = min(v * b / (p + b) / max(n, 1), 1.0 - v)
    if bluff is None:
        return val | _top(r, bf, rank[::-1])
    rest = r * ~val
    left, tot = float(rest.sum()), float(r.sum())
    return val | (_top(rest, bf * tot / left, bluff) if left > 1e-12 else val)


def _value_share(r, n, b, p, rank, others=None, joint=None, share=None) -> float:
    """Share of range r that bets b into p for value, with n players facing it: the hands above the
    top VALUE_SHARE of what would call (balanced, unshaded, by the joint rule of _joint_top). With the
    others' ranges, what they would call with; without, as if each held r (VALUE_SHARE of what r keeps,
    the same number when they do). A strong range betting into capped ones values thinner (top pair
    from the opener into two flatters), a capped one into a strong one tighter. joint: _Joint(others,
    rank), when the caller has it (the same for both bet sizes). share: in place of VALUE_SHARE."""
    share = VALUE_SHARE if share is None else share
    if joint is None:
        others = [o for o in others if o.sum() > 0] if others else []
        if not others:
            return _keep(b, p, n, 1.0) * share
        joint = _Joint(others, rank)
    if not joint.n:
        return _keep(b, p, n, 1.0) * share
    fold = (1.0 - _keep(b, p, joint.n, 1.0)) ** joint.n
    return _share(r, joint.value_mask(fold, share))


class _Joint:
    """The joint fold rule's pieces for a fixed set of ranges in one rank order (see _joint_top),
    computed once and used for any fold share: below[m] is the chance every hand is outside the top
    m combos; norm the ranges' sum, each normalised (what they'd call with, combined)."""
    __slots__ = ("n", "rank", "below", "cum_norm")

    def __init__(self, rs, rank):
        rs = [r for r in rs if r.sum() > 0]
        self.n, self.rank = len(rs), rank
        self.below = _joint_below(rs, rank)
        self.cum_norm = np.cumsum(sum(r / r.sum() for r in rs)[rank]) if rs else None

    def cut(self, fold) -> int:
        """How many of the strongest combos continue when they fold together with probability fold."""
        return int(np.argmax(self.below <= fold + 1e-9)) if fold < 1.0 else 0

    def value_mask(self, fold, share) -> np.ndarray:
        """_top(what they continue with, combined, share): the strongest combos holding `share` of it."""
        m = np.zeros(N_COMBOS, dtype=bool)
        k = self.cut(fold)
        if k > 0 and share > 1e-9 and self.cum_norm[k - 1] > 0:
            cum = self.cum_norm[:k]
            m[self.rank[:int(np.searchsorted(cum, share * cum[-1] - 1e-9)) + 1]] = True
        return m


def _nut_weight(r, others, rank) -> float:
    """How much of its value range r bets big against the ranges others (0..1): by r's share of the
    nuts, the top NUT_TOP of every player's range pooled (each normalised), against the field's: r's
    part of the pooled nuts above ADV0 over ADV_SPAN. Equal ranges give 1 / players, so heads-up a
    range that holds more of the nuts bets big and into two players only one that dominates them."""
    rs = [x / x.sum() for x in [r] + list(others) if x.sum() > 0]
    if len(rs) < 2:
        return 0.0
    nuts = _top(sum(rs), NUT_TOP, rank)
    held = [float(x[nuts].sum()) for x in rs]
    if sum(held) <= 0:
        return 0.0
    return float(np.clip((held[0] / sum(held) - ADV0) / ADV_SPAN, 0.0, 1.0))


def _with_ties(m, r, sc) -> np.ndarray:
    """Mask m (the strongest hands of range r) and every hand of r as strong as its weakest: a cut by rank
    order splits hands of equal strength (a straight's combos) arbitrarily."""
    live = m & (r > 0)
    if sc is None or not live.any():
        return m
    return m | ((sc >= sc[live].min() - 1e-12) & (r > 0))


def _bet_ranges(r, n, p, big, small, rank, bluff=None, others=None, drop=False, sc=None):
    """(big, small) masks: what range r bets at each size into p with n players facing it (others:
    their ranges, see _value_share; without them, as if each held r). The small bet is the default:
    value from the top SMALL_SHARE of what would call it (a merged range: medium hands bet for
    protection) plus balanced bluffs. The big bet takes the top of that value, the hands above the
    top VALUE_SHARE of what would call it, by r's nut advantage (_nut_weight), with its own balanced
    bluffs first in bluff order. GTO Wizard's 3-way flop study (100bb, LJ opens, both blinds call) has
    the opener check 34%, bet 33% 56% and bigger 10%; heads-up against the big blind 23%, 38% and 39%.
    The old rule (big bet the top, small a thin layer below it) checked 67% 3-way and bet small 7%.
    small None: no small bet (an empty mask) and the big bet takes the value share whole.
    drop: also return what a check rules out: the big bet, the small bet's bluffs and its value above
    the top VALUE_SHARE of what would call it; the merged middle (thin value, protection) checks too,
    as in the solves, whose checking ranges keep medium hands and some traps (ruling out every hand that
    would bet small left the checkers so weak that top pair raised a river lead for +47 bb). sc (strength,
    rank's order): the value a check rules out takes every hand as strong as its weakest (_with_ties), so
    a range of mostly one nut hand doesn't check the part of it the cut happened to leave out."""
    others = [o for o in others if o.sum() > 0] if others else []
    joint = _Joint(others, rank) if others else None
    empty = np.zeros(N_COMBOS, dtype=bool)
    v_big = _value_share(r, n, big, p, rank, others, joint)
    if small is None:
        bg = _aggro(r, v_big, big, p, rank, bluff)
        return (bg, empty, bg) if drop else (bg, empty)
    f = _nut_weight(r, others if others else [r] * max(n, 1), rank)
    bg = _aggro(r, v_big * f, big, p, rank, bluff) if f > 0 else empty
    s_share = 1.0 - (1.0 - SMALL_SHARE) ** (len(others) if others else max(n, 1))
    val = _top(r, min(_value_share(r, n, small, p, rank, others, joint, s_share), SMALL_VALUE_MAX), rank) & ~bg
    rest = r * ~(bg | val)
    left, tot = float(rest.sum()), float(r.sum())
    bf = _share(r, val) * small / (p + small)
    order = rank[::-1] if bluff is None else bluff
    sm = val | (_top(rest, min(bf * tot / left, 1.0), order) if left > 1e-12 and bf > 0 else val)
    if not drop:
        return bg, sm & ~bg
    strong = _top(r, _value_share(r, n, small, p, rank, others, joint), rank)
    if strong.any() and sc is not None:
        strong = _with_ties(strong, r, sc)
        return bg, sm & ~bg, bg | (sm & ~val) | strong
    return bg, sm & ~bg, bg | (sm & ~val) | (val & strong)


def _bluff_order(sts, rank, sc) -> np.ndarray:
    """Combos in the order a range picks its bluffs, over the runouts sts (strength arrays, -1 =
    blocked): first the hands without showdown value (in under half the runouts do they end above
    SHOWDOWN_Q of the live combos), the ones that most often end in the top HIT_TOP first (draws,
    overcards), weakest first among equals; then the rest, weakest first. With one runout (the
    river) it is simply weakest first. Betting ranges bluffed from the bottom before, so on the flop
    their bluffs were air and a draw out-continued top pair against them."""
    if len(sts) <= 1:
        return rank[::-1]
    hit = np.zeros(N_COMBOS); sd = np.zeros(N_COMBOS); cnt = np.zeros(N_COMBOS)
    for s in sts:
        live = s >= 0
        if not live.any():
            continue
        hi, mid = np.quantile(s[live], [1.0 - HIT_TOP, SHOWDOWN_Q])
        hit += live & (s >= hi); sd += live & (s >= mid); cnt += live
    cnt = np.maximum(cnt, 1)
    return np.lexsort((sc, -hit / cnt, sd / cnt >= 0.5))


def _raise_to(bet, calls) -> float:
    """A raise over a bet (to-level `bet`) that `calls` chips have called: 3x the bet plus the calls."""
    return 3.0 * bet + calls


def _keep(b, p, n, shade=None) -> float:
    """Share of their range each of n players continues with against a bet of b into p: together
    they fold shade * b / (p + b) of the time (shade 1 = a balanced defence)."""
    shade = FOLD_SHADE if shade is None else shade
    return 1.0 - (shade * b / (p + b)) ** (1.0 / max(n, 1))


def _joint_below(rs, rank) -> np.ndarray:
    """below[m]: the chance that every one of the ranges rs holds a hand outside the top m combos (in
    `rank` order), m = 0..N_COMBOS (empty ranges left out)."""
    rs = [r for r in rs if r.sum() > 0]
    below = np.ones(N_COMBOS + 1)
    if rs:
        c = np.cumsum(np.stack(rs)[:, rank], axis=1)
        below[1:] = np.prod(1.0 - c / c[:, -1:], axis=0)
    return below


def _joint_top(rs: List[np.ndarray], fold: float, rank: np.ndarray) -> List[np.ndarray]:
    """What each of the ranges rs continues with against a bet they fold to together with
    probability `fold`: everyone keeps the hands above one strength bar (in `rank` order), set so
    the chance that every player's hand is below it is `fold`. A range of weak hands gives up more
    of itself than a strong one: after a lead and a call, the caller (who called with what beats
    the lead) keeps most of their range and the leader's bluffs fold. One range: _top(r, 1 - fold)."""
    if not len(rs):
        return []
    m = int(np.argmax(_joint_below(rs, rank) <= fold + 1e-9)) if fold < 1.0 else 0
    mask = np.zeros(N_COMBOS, dtype=bool)
    mask[rank[:m]] = True
    return [mask] + [mask.copy() for _ in rs[1:]]


def _share(r, mask) -> float:
    """Share of range r's weight on mask."""
    t = r.sum()
    return float((r * mask).sum() / t) if t > 0 else 0.0


def _fold_together(costs, p, shade=None) -> float:
    """Chance the players paying `costs` to continue into a pot of p fold together (see _keep)."""
    return float(np.prod([1.0 - _keep(c, p, len(costs), shade) for c in costs]))


def _street_rank(board: tuple):
    """(strength averaged over runouts, combos strongest first) on this board; -1 = blocked."""
    return _street_info(board)[:2]


@lru_cache(maxsize=64)
def _street_info(board: tuple):
    """(strength averaged over runouts, combos strongest first, bluff order) on this board (see
    _bluff_order); -1 = blocked."""
    from .evaluator import eval_board
    rng = random.Random(sum(c * 53 ** i for i, c in enumerate(board)))
    avail = [c for c in range(52) if c not in board]
    tot = np.zeros(N_COMBOS); cnt = np.zeros(N_COMBOS)
    sts = []
    # Every turn card twice from the flop (a random river each), every river from the turn; strength as a
    # percentile of the live combos on each runout, averaged. 24 random runouts of raw strength ranked
    # K5s and K9o above AA on K72 (two-pair runouts weigh more in raw strength than an overpair holding).
    if len(board) == 5:
        runs = [()]
    elif len(board) == 4:
        runs = [(c,) for c in avail]
    else:
        runs = [(t, rng.choice([c for c in avail if c != t])) for t in avail for _ in range(2)]
    for run in runs:
        s = eval_board(list(board) + list(run))
        ok = s >= 0
        pct = np.zeros(N_COMBOS)
        srt = np.sort(s[ok])
        mid = (np.searchsorted(srt, s[ok], "left") + np.searchsorted(srt, s[ok], "right") - 1) / 2.0   # ties share a rank
        pct[ok] = mid / max(ok.sum() - 1, 1)
        tot[ok] += pct[ok]; cnt[ok] += 1
        sts.append(s)
    sc = np.where(cnt > 0, tot / np.maximum(cnt, 1), -1.0)
    rank = np.argsort(-sc)
    return sc, rank, _bluff_order(sts, rank, sc)


# The preflop boards and their lookups (below) built so far, in order: warm_step() builds them a slice at a time while
# the browser's worker is idle, where the first multiway preflop spot priced (a big blind facing an open and a call,
# a 3-way flop's ranges) took 0.5 s for them in Pyodide; whatever is left is built when first needed, the same
# boards either way (one random sequence, drawn in order).
_TABLES: dict = {"rng": None, "boards": [], "index": []}


def _preflop_boards_upto(k: int) -> None:
    from .evaluator import eval_board
    t = _TABLES
    if t["rng"] is None:
        t["rng"] = random.Random(1)
    while len(t["boards"]) < min(k, PREFLOP_BOARDS):
        t["boards"].append(eval_board(t["rng"].sample(range(52), 5)))


@lru_cache(maxsize=1)
def _preflop_boards():
    """Fixed random boards and every combo's strength on each (-1 = blocked), for preflop equity."""
    _preflop_boards_upto(PREFLOP_BOARDS)
    return np.array(_TABLES["boards"])


def _implied_w(h: str) -> float:
    """Implied-odds weight of a hand class: the ones that win big pots by hitting (pairs 22-99, suited
    hands that aren't two big cards); premiums and offsuit hands don't depend on it."""
    if len(h) == 2:
        return IMPLIED["pair"] if h[0] in "23456789" else 0.0
    if h.endswith("s") and not (h[0] in "AKQJT" and h[1] in "AKQJT"):
        return IMPLIED["suited"]
    return 0.0


_IMPLIED_W = np.array([_implied_w(hand_class(c)) for c in ALL_COMBOS])
_CARD_MASK = np.array([(1 << a) | (1 << b) for a, b in ALL_COMBOS], dtype=np.int64)


def _preflop_equity(villain: np.ndarray) -> np.ndarray:
    """All-in preflop equity of every combo against the weighted range `villain` (pf_classes.equity_vs)."""
    from .pf_classes import equity_vs
    return equity_vs(villain)


_CA = np.array([a for a, b in ALL_COMBOS])
_CB = np.array([b for a, b in ALL_COMBOS])


def _board_index_upto(k: int) -> None:
    _preflop_boards_upto(k)
    out = _TABLES["index"]
    for s in _TABLES["boards"][len(out):min(k, PREFLOP_BOARDS)]:
        o = np.argsort(s, kind="stable")
        ss = s[o]
        keys = np.concatenate([_CA, _CB]) * 8192 + np.concatenate([s, s])
        o2 = np.argsort(keys, kind="stable")
        k2 = keys[o2]
        q = np.maximum(s, -1)
        ix = [np.searchsorted(ss, s, "left"), np.searchsorted(ss, s, "right")]
        for c in (_CA, _CB):
            ix += [np.searchsorted(k2, c * 8192, "left"), np.searchsorted(k2, c * 8192 + q, "left"),
                   np.searchsorted(k2, c * 8192 + q, "right"), np.searchsorted(k2, c * 8192 + 8191, "right")]
        out.append((o.astype(np.int16), o2.astype(np.int16), [i.astype(np.int16) for i in ix]))   # (< 2653: 14 MB)


@lru_cache(maxsize=1)
def _board_index():
    """Per preflop board, the orders and lookups _beat_share needs that don't depend on the range: every
    combo by strength, every (card, strength) entry of a combo holding that card, and where each combo's own
    strength falls in both."""
    _board_index_upto(PREFLOP_BOARDS)
    return list(_TABLES["index"])


# the modules an answer imports on first use (the worker's setup imports only what a preflop or library answer needs):
# 3-way pots, preflop pricing, exploits, the breakdown, tournaments (~0.1 s in Pyodide, compiled on each visit)
WARM_IMPORTS = ("multiway_solver", "pf_formats", "pf_equity", "pf_game", "pf_eqr", "pf_realize", "pf_libvals",
                "pf_solve", "breakdown", "exploit", "icm")


def warm_step(budget: float = 0.05) -> bool:
    """Work for idle time (the browser's worker between requests): the preflop equity tables (_preflop_boards,
    _board_index, _board_index_all) built for about `budget` seconds at a time, so the first multiway preflop spot
    doesn't wait for them (0.5 s in Pyodide), then WARM_IMPORTS one at a time. True once there is nothing left to
    do."""
    t0 = time.time()
    while time.time() - t0 < budget:
        t = _TABLES
        if len(t["boards"]) < PREFLOP_BOARDS:
            _preflop_boards_upto(len(t["boards"]) + 10)
        elif len(t["index"]) < PREFLOP_BOARDS:
            _board_index_upto(len(t["index"]) + 5)
        elif _board_index_all.cache_info().currsize == 0:
            _board_index_all()
        else:
            i = t.setdefault("imported", 0)
            if i >= len(WARM_IMPORTS):
                return True
            t["imported"] = i + 1
            try:
                importlib.import_module(f"{__package__}.{WARM_IMPORTS[i]}")
            except Exception:   # (imported when an answer needs it, as before)
                pass
    return False


def _beat_share(bi: int, w: np.ndarray) -> np.ndarray:
    """On preflop board bi, every combo's chance of beating a hand drawn from the range w (1326 weights,
    zero on combos the board blocks), ties half, counting only villain combos sharing no card with it:
    sorted cumulative weights, and per card the same for the villain combos holding it (inclusion-exclusion;
    the only combo holding both of ours is ours). Also returns that compatible weight, per combo."""
    o, o2, ix = _board_index()[bi]
    cw = np.concatenate([[0.0], np.cumsum(w[o])])
    c2 = np.concatenate([[0.0], np.cumsum(np.concatenate([w, w])[o2])])
    lt, le = cw[ix[0]], cw[ix[1]]
    lt_ab = le_ab = tot_ab = 0.0
    for j in (2, 6):
        base = c2[ix[j]]
        lt_ab = lt_ab + c2[ix[j + 1]] - base
        le_ab = le_ab + c2[ix[j + 2]] - base
        tot_ab = tot_ab + c2[ix[j + 3]] - base
    tot = w.sum() - tot_ab + w
    win = (lt - lt_ab) + 0.5 * ((le - le_ab + w) - (lt - lt_ab))
    return np.where(tot > 1e-12, win / np.maximum(tot, 1e-12), 0.0), tot


@lru_cache(maxsize=1)
def _board_index_all():
    """_board_index for the first MULTI_BOARDS boards stacked (B, ...) with each board's offsets into the flattened
    cumulative sums, so _beat_share_all reads every board's shares in one pass of numpy calls rather than one per
    board (preflop equity against several ranges: 400 boards x 30 calls each was most of a big multiway preflop
    line's time, more so in the browser)."""
    bi = _board_index()[:MULTI_BOARDS]
    B = len(bi)
    rows1 = (np.arange(B) * (N_COMBOS + 1))[:, None]
    rows2 = (np.arange(B) * (2 * N_COMBOS + 1))[:, None]
    o = np.stack([x[0] for x in bi]).astype(np.int64)
    o2 = np.stack([x[1] for x in bi]).astype(np.int64)
    ix = [np.stack([x[2][j] for x in bi]).astype(np.int64) + (rows1 if j < 2 else rows2) for j in range(10)]
    live = np.stack([s >= 0 for s in _preflop_boards()[:MULTI_BOARDS]])
    return o, o2, ix, live


def _beat_share_all(w: np.ndarray):
    """_beat_share on every one of the first MULTI_BOARDS boards at once: w (B, 1326), each board's range (zero on
    the combos it blocks) -> (share, compatible weight), each (B, 1326)."""
    o, o2, ix, _ = _board_index_all()
    B = w.shape[0]
    cw = np.zeros((B, N_COMBOS + 1)); np.cumsum(np.take_along_axis(w, o, axis=1), axis=1, out=cw[:, 1:])
    c2 = np.zeros((B, 2 * N_COMBOS + 1))
    np.cumsum(np.take_along_axis(np.concatenate([w, w], axis=1), o2, axis=1), axis=1, out=c2[:, 1:])
    cw, c2 = cw.ravel(), c2.ravel()
    lt, le = cw[ix[0]], cw[ix[1]]
    lt_ab = le_ab = tot_ab = 0.0
    for j in (2, 6):
        base = c2[ix[j]]
        lt_ab = lt_ab + c2[ix[j + 1]] - base
        le_ab = le_ab + c2[ix[j + 2]] - base
        tot_ab = tot_ab + c2[ix[j + 3]] - base
    tot = w.sum(axis=1, keepdims=True) - tot_ab + w
    win = (lt - lt_ab) + 0.5 * ((le - le_ab + w) - (lt - lt_ab))
    return np.where(tot > 1e-12, win / np.maximum(tot, 1e-12), 0.0), tot


BEAT_CACHE = 4   # ranges whose shares on every board are kept (8.5 MB each): a line's pricing meets the same few
_BEAT: Dict[str, tuple] = {}


def _beat_share_cached(r: np.ndarray, live: np.ndarray):
    key = _range_key(r)
    hit = _BEAT.pop(key, None)
    if hit is None:
        hit = _beat_share_all(np.where(live, r[None, :], 0.0))
    _BEAT[key] = hit   # (most recent last)
    while len(_BEAT) > BEAT_CACHE:
        _BEAT.pop(next(iter(_BEAT)))
    return hit


def _preflop_equity_multi(villains: List[np.ndarray]) -> np.ndarray:
    """Preflop equity of every combo against several weighted ranges at once: on each board, the product
    of the chances of beating each (ties count half; card removal between the villains is ignored), over
    the first MULTI_BOARDS boards (each class pools its combos below)."""
    if len(villains) == 1:
        return _preflop_equity(villains[0])
    # (a preflop line's pricing asks the same ranges again and again: an 8-way limped pot's 57 calls were 22 of them)
    key = tuple(_range_key(v) for v in villains)
    hit = _EQ_MULTI.get(key)
    if hit is None:
        hit = _EQ_MULTI[key] = _preflop_equity_multi_calc(villains)
        while len(_EQ_MULTI) > EQ_MULTI_CACHE:
            _EQ_MULTI.pop(next(iter(_EQ_MULTI)))
    return hit.copy()


EQ_MULTI_CACHE = 256   # preflop equities against several ranges kept (10 KB each)
_EQ_MULTI: Dict[tuple, np.ndarray] = {}


def _preflop_equity_multi_calc(villains: List[np.ndarray]) -> np.ndarray:
    live = _board_index_all()[3]
    p = live.astype(float)
    bw = live.astype(float)
    for r in villains:
        t = max(float(r.sum()), 1e-12)
        sh, tot = _beat_share_cached(r, live)
        p = p * sh
        # boards are as likely as the players' hands leave them: a board of the cards their ranges hold
        # comes less often (high cards, against ranges of broadways), which the uniform boards missed: 76s
        # and 22 read 2-3 points under a dealt-out simulation against five players, Q9o 2 over
        bw = bw * tot / t
    num, den = (p * bw).sum(axis=0), bw.sum(axis=0)
    # Every combo of a hand class is the same hand up to suits, and the (chart) ranges are suit-symmetric,
    # so the class's combos share one estimate: each combo alone saw only its 60-75 boards, and 76s came out at
    # 5.1% against five players where 65s had 9.5% and Q9o 14.7%, so BB folded 76s, 65s and 54s 70-77% of the
    # time getting 10:1. Pooled, each class sees 4-12x the boards.
    from .pf_classes import CLASS_OF
    k = int(CLASS_OF.max()) + 1
    num_c = np.bincount(CLASS_OF, weights=num, minlength=k)
    den_c = np.bincount(CLASS_OF, weights=den, minlength=k)
    return np.where(den_c[CLASS_OF] > 0, num_c[CLASS_OF] / np.maximum(den_c[CLASS_OF], 1), np.nan)


def _narrow_street(ranges: Dict[str, np.ndarray], board, st: StreetState, stack, hero=None, hole=(),
                   slow: Optional[Dict[str, np.ndarray]] = None, check_whole: bool = False) -> None:
    """Narrow each player's range (in place) by what they did on a street that started multiway,
    with the multiway rules: a check drops most of the hands that would have bet, a bet or raise
    keeps its value + bluff range, a call keeps what continues minus what would have raised.
    Hands a rule rules out keep some weight (players don't follow our rules), see _soften. `slow`
    (carried from street to street): per player, the combos a call has already slowplayed. `stack`: everyone's
    stack in bb, or {seat: stack}. `check_whole`: a check keeps the range whole, as a heads-up street's rules do
    (Advisor._walk at trust 0: heads-up the rules' check took the value out of ranges that check nearly all of it)."""
    sc, rank, bluff = _street_info(tuple(board))
    dead = [MEMBER_FULL[c] for c in hole]
    bets = {p: 0.0 for p in st.inv_start}
    cur = 0.0
    active = set(st.players_start)
    faced = None    # the last bet or raise: (pot before it, chips its maker added, players facing it)
    called = 0      # calls of it so far
    allin = False   # the bet to call is an all-in: nobody can raise it
    start = {p: ranges[p] for p in ranges}   # the ranges the street began with
    checked = set()
    for a in st.actions:
        p, act = a["pos"], a["action"]
        if p not in active or p not in ranges:
            continue
        if act == "fold":
            active.discard(p)
            continue
        # A check-raise (or a bet after checking) is read from the range the street began with: the check
        # dropped what would have bet, sets and two pair among it (88, 44 and K8s at 0.3 in a check-raise
        # on K84 where T8o and A8o were whole).
        src = start[p] if p in checked and act in ("bet", "raise", "all-in") else ranges[p]
        r = src * (sc >= 0)
        for d in dead if p != hero else ():   # opponents can't hold our cards
            r = r * (1.0 - d)
        if r.sum() <= 0:
            continue
        n = len(active) - 1
        pot = st.pot_start + sum(bets.values())
        remaining = (stack[p] if isinstance(stack, dict) else stack) - st.inv_start[p] - bets[p]
        to_call = max(0.0, min(cur - bets[p], remaining))
        if act == "all-in":
            to = bets[p] + remaining
        elif act in ("bet", "raise"):
            to = a.get("to") if a.get("to") is not None else a.get("amount")
            to = min(float(to) if to is not None else _default_size_post(pot, cur, bets[p]), bets[p] + remaining)
        else:
            to = cur
        # A call continues against the bet as everyone it was made into would together (pricing the bet
        # into the pot it had grown to left overcallers wider than the first caller); an overcaller,
        # who needs to beat the caller too, with OVERCALL of that.
        call_k = _keep(faced[1], faced[0], faced[2]) if faced and to_call > 1e-9 and to <= cur + 1e-9 else None
        if call_k is not None and called:
            call_k *= OVERCALL
        others = [ranges[q] * (sc >= 0) for q in active if q != p and q in ranges]
        # calling an all-in (or calling all-in) keeps the hands that would have raised: there's no raise
        raise_ok = not (allin or to_call >= remaining - 1e-9)
        keep_m = _rule_mask(r, to, cur, to_call, pot, n, rank, call_k, bluff, others, raise_ok=raise_ok, sc=sc)
        if to > cur + 1e-9:
            faced = (pot, min(to, bets[p] + remaining) - bets[p], max(n, 1))
            called = 0
            allin = to >= bets[p] + remaining - 1e-9
        if act == "check":
            checked.add(p)
        if act == "call" and to_call > 1e-9:
            called += 1
        cur = max(cur, to)
        bets[p] = max(bets[p], min(to, bets[p] + remaining))
        if act == "check" and check_whole:
            continue
        ranges[p] = r * (_soften(r, keep_m, CHECK_SOFT) if act == "check" and len(board) == 3 else _soften(r, keep_m))
        if act == "call" and to_call > 1e-9 and raise_ok:
            k = _keep(to_call, pot, n) if call_k is None else call_k
            ranges[p] = r * _slowplayed(r, ranges[p] / np.where(r > 0, r, 1.0), k, rank, slow, p)


def _slowplayed(r, w, k, rank, slow=None, p=None) -> np.ndarray:
    """Weights w for a call of range r that continues with its top k (rank order), with the hands the rules
    have raising instead (the top k * RAISE_SHARE) kept at SLOWPLAY: cut whole, an overcaller's range after
    three streets of calls was underpairs (TT-55 on K72-5-9, no Kx left) and KQ overcalled a pot-sized river
    bet; heads-up, BB's flop and turn calls on Q32ss-J kept 18% of its sets and 27% of its flushes where it
    kept half its air, and A4 jammed the river into the rest (converged solves keep the flushes).
    `slow` (carried from street to street, keyed by p): the combos a call has already slowplayed; such a hand
    mostly keeps on (SLOWPLAY_AGAIN: halved again each street, an overcaller after three calls was TT-55 with
    KJs at 0.07). Updated in place."""
    band = _top(r, k * RAISE_SHARE, rank)
    again = slow.get(p, np.zeros(N_COMBOS, dtype=bool)) if slow is not None else np.zeros(N_COMBOS, dtype=bool)
    if slow is not None:
        slow[p] = again | band
    return np.where(band, np.maximum(w, np.where(again, SLOWPLAY_AGAIN, SLOWPLAY)), w)


def trust_of(expl: float, street: str) -> float:
    """How far an earlier street's narrowing follows its solve, by the solve's exploitability (% pot); see
    PRIOR_TRUST."""
    lo, hi = PRIOR_TRUST[street]
    return float(min(1.0, max(0.0, (hi - expl) / (hi - lo))))


def _soften(r, keep_m, soft=None) -> np.ndarray:
    """Per-combo weights for range r after a line the rules play with keep_m: 1 on the rule's hands,
    NARROW_SOFT off them, lowered so the off-rule hands make up at most OFF_RULE_MAX of the result."""
    on, off = float(r[keep_m].sum()), float(r[~keep_m].sum())
    if on <= 1e-9:
        # the rules play none of the range this way (a 3bet-pot flop overbet's value range, JJ+ and sets,
        # checking the turn): nothing to narrow by, so the range stays as it was rather than emptying (0 combos
        # left, and the solve against it put every hand of ours at exactly half the pot)
        return np.ones_like(r)
    soft = NARROW_SOFT if soft is None else soft
    if off * soft > 0 and OFF_RULE_MAX < 1.0:
        soft = min(soft, OFF_RULE_MAX / (1.0 - OFF_RULE_MAX) * on / off)
    return np.where(keep_m, 1.0, soft)


def _rule_mask(r, to, cur, to_call, pot, n, rank, call_k=None, bluff=None, others=None, raise_ok=True,
               sc=None) -> np.ndarray:
    """The part of range r the multiway rules play this way, with n opponents and `pot` in the
    middle: a bet or raise to `to` over the current bet `cur`, else a call of to_call (or a check).
    call_k: for a call, the share of the range that continues (default: facing to_call into pot).
    bluff: the order bluffs are picked in (see _bluff_order; default the weakest hands). others: the
    other players' ranges, for the betting ranges (see _bet_ranges). A big bet's value share stays the
    fitted one (as if the others held r), not _value_share's against their ranges."""
    if to > cur + 1e-9:                       # bet or raise: value + balanced bluffs
        if to_call <= 1e-9:
            if to - cur < 0.5 * pot:          # a small bet: the small betting range (merged), see _bet_ranges
                big, small = _bet_ranges(r, n, pot, max(0.66 * pot, to - cur), to - cur, rank, bluff, others)
                if r[small].sum() > 0:
                    return small | big       # (strong hands bet small too: players mix)
            return _aggro(r, _value_share(r, n, to - cur, pot, rank), to - cur, pot, rank, bluff, n)
        return _aggro(r, _keep(to_call, pot, n) * RAISE_SHARE, to - cur, pot + to_call, rank, bluff, n)
    if to_call <= 1e-9:                       # check: not what would have bet at either size
        return ~_bet_ranges(r, n, pot, 0.66 * pot, SMALL_BET * pot, rank, bluff, others, drop=True, sc=sc)[2]
    k = _keep(to_call, pot, n) if call_k is None else call_k   # call: continues, but wouldn't raise
    return _top(r, k, rank) & ~_top(r, k * RAISE_SHARE if raise_ok else 0.0, rank)


def _default_size_post(pot, cur, mine) -> float:
    return cur + 0.75 * (pot + cur - mine) if cur > mine else 0.66 * pot


def _default_size(st: StreetState, p: str, pre: bool) -> float:
    if pre:
        # chips the other live players (not the last raiser) put in beyond their blinds: a caller of
        # the current bet adds one bet to the size (a squeeze), a limper 1bb (an iso-raise)
        last = next((a["pos"] for a in reversed(st.actions) if a["action"] in ("raise", "all-in", "bet")), None)
        extra = sum(max(0.0, st.bets[q] - st.posted.get(q, 0.0)) for q in st.active if q not in (p, last))
        if st.n_raises == 0:
            if extra > 0:   # over limpers: 3x (4x from the blinds) plus one bet per limper
                return st.cur_bet * (4.0 if p in st.posted else 3.0) + extra
            return DEFAULT_OPEN.get(p, 2.5) * st.cur_bet
        if st.n_raises == 1:
            if p == "BB" and last == "SB" and extra == 0:
                # blind vs blind the big blind has position: 10bb over a 3bb open, as gtopreflop.com's solve
                # (pokalab and GTO Ranges+ 9bb), which the small blind's chart against it is matched to
                return st.cur_bet * BVB_3BET
            return st.cur_bet * (4.0 if p in st.posted else 3.0) + extra
        return st.cur_bet * 2.3 + extra
    pot = st.pot_start + sum(st.bets.values())
    if st.cur_bet > st.bets[p]:
        return st.cur_bet + 0.75 * (pot + st.cur_bet - st.bets[p])
    return 0.66 * pot


def preflop_investments(actions: List[dict], stacks: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """Kept for compatibility: chips each player put in preflop and who is still in (per-seat stacks)."""
    stack = min(stacks.values()) if stacks else 100.0
    table = pf.Table(pf.POSITIONS, stacks={p: stacks.get(p, stack) for p in pf.POSITIONS} if stacks else None)
    st = replay_hand(table, stack, {"preflop": actions}, 1)[0]
    inv = {p: st.inv_start[p] + st.bets[p] for p in pf.POSITIONS}
    return inv, [p for p in pf.POSITIONS if p in st.active]


# ----------------------------------------------------------------------------- adjusted preflop play
# Every player's preflop decisions get the same off-chart adjustments as hero's (Advisor._priced: price,
# players in, depth, antes), so the ranges replayed into later decisions match the advice. Cached per
# action line: each decision is worked out once.
_PF_CACHE: Dict[tuple, object] = {}


def _pf_key(kind, p, actions, positions, stack) -> tuple:
    acts = tuple((a["pos"], a["action"], a.get("to"), a.get("amount")) for a in actions)
    icm = getattr(positions, "icm", None)
    return (kind, p, acts, tuple(positions), getattr(positions, "ante", 0.0), getattr(positions, "straddle", False), stack,
            getattr(positions, "rake", None), pf.stacks_key(positions), icm.key if icm is not None else None)


def _rake(positions, pot: float) -> float:
    """Rake in bb on a pot of `pot` bb that sees the flop (the table's rate, to its cap; 0 without rake)."""
    rk = getattr(positions, "rake", None)
    if not rk:
        return 0.0
    return rk[0] * pot if rk[1] is None else min(rk[0] * pot, rk[1])


# Rake on preflop calls (see Advisor._priced): a call that sees the flop pays rake on the pot it goes on
# to win, which later streets grow. Its expected rake is taken on RAKE_GROWTH times the pot the call
# takes to the flop, to the cap. Charts fit to published raked solves (CHART_RAKE: gtopreflop.com's 5%
# capped at 4bb) are only charged the rake beyond that structure's. 2.5 is fit to that solve's calls at
# 5%/4bb (CO vs a BB 3bet 23.8% of its opens against 22.4%, BTN vs a BB 3bet 20.3% against 20.0%, the
# BB vs a CO open 22.9% of hands against 23.8%; the flop pot alone, 1.0, left the BB calling 48% vs BTN).
RAKE_GROWTH = 2.5
CHART_RAKE = (0.05, 4.0)


def _call_rake(positions, pot: float, behind: float = float("inf"), raked_chart: bool = False) -> float:
    """Expected rake in bb a preflop call pays on a pot of `pot` bb at the flop: the table's rate on
    RAKE_GROWTH x pot (no more than the pot with `behind` bb called by each of two players), to its cap
    (0 without rake). For a chart already fit to a raked solve (`raked_chart`), only what this takes
    beyond CHART_RAKE's."""
    rk = getattr(positions, "rake", None)
    if not rk:
        return 0.0
    final = min(RAKE_GROWTH * pot, pot + 2.0 * max(behind, 0.0))
    cost = lambda rate, cap: rate * final if cap is None else min(rate * final, cap)
    r = cost(*rk)
    return max(0.0, r - cost(*CHART_RAKE)) if raked_chart else r


def _call_realization(hero: str, raiser: str, n_raises: int, ranges: Dict[str, np.ndarray], positions) -> np.ndarray:
    """Per combo: the share of its all-in equity hero realizes after calling `raiser`'s raise (the
    n_raises-th) heads-up, from the measured realization model (pf_game.measured_model; ones without it)."""
    from . import pf_game
    from .pf_classes import CLASS_OF, to_classes
    model = pf_game.measured_model()
    opp = ranges.get(raiser)
    if model is None or opp is None or opp.sum() <= 0 or n_raises < 1:
        return np.ones(N_COMBOS)
    order = pf.postflop_order(list(positions))
    aggr_ip = order.index(raiser) > order.index(hero)
    r = pf_game.realization(model, n_raises, aggr_ip, "caller", to_classes(opp))
    return (r + RIVER_FIX.get(n_raises, 0.0) * _river_fix())[CLASS_OF]


# The realization model is measured with the fast flop model (turn and river by hand buckets, the river
# played without its card). Re-solved with the river dealt and played (scripts/pf_fidelity.py, docs/
# preflop_research.md section 7), callers in single-raised pots realize more with connectors and gappers
# (65s +0.23, T8o and 54s +0.15, T9s +0.11, J8s +0.20) and small pairs (22 +0.08 to +0.20), and less with
# offsuit aces (AJo -0.17); about half that in 3bet pots. Suited high cards gain too (KTs +0.14, A5s
# +0.12: the flush draw is paid on the river the fast model plays blind). _river_fix is that correction
# smoothed on hand shape: connectors and gappers up to J-high +0.15, 22-66 +0.10, other suited hands
# +RIVER_FIX_SUITED, other offsuit hands RIVER_FIX_OFFSUIT (the aces' shrunk from 8 flops' -0.17 to -0.05:
# at -0.10 the big blind at 5%/4bb folded A2o-A8o but for 3bets), scaled by RIVER_FIX per number of raises
# called. Used only to rank re-priced calls (_rerank), whose number the price sets. Until 2026-10 only
# the offsuit aces had theirs and suited non-connectors none: at 5%/4bb the big blind then kept Q9o 100%
# (gtopreflop.com 27%) and cut K2s to 60% (K2s-K9s 95%, published 100%); the suited +0.10 and every
# offsuit non-connector at -0.05 give Q9o 33%, K2s-K9s 100%, K9o/KTo 73/69% and K7o/K8o folds (no
# published figure for those), the 22 spots 1.03 -> 1.02 points (docs/preflop_research.md section 8).
RIVER_FIX = {1: 1.0, 2: 0.5}
RIVER_FIX_SUITED = 0.10
RIVER_FIX_OFFSUIT = -0.05
_RIVER_FIX = None


def _beats_note(hole, board, bettor: str, r: np.ndarray) -> Optional[str]:
    """On the river, facing a bet: a note when our hand beats at most BEATS_NOTE_MAX of the bettor's range r
    (combos blocked by the board or our cards left out), naming the range's weakest hand classes, so a fold
    to a tiny bet reads as what it is (A8s on 79QKJ ties every Ax and beats none of a range of pairs and Ax)."""
    if len(board) != 5:
        return None
    from .evaluator import eval_board
    s = eval_board(list(board))
    hi = COMBO_INDEX[(min(hole), max(hole))]
    r = r * (s >= 0)
    for c in hole:
        r = r * (1.0 - MEMBER_FULL[c])
    tot = float(r.sum())
    if tot <= 1e-9:
        return None
    beat = float(r[s < s[hi]].sum()) / tot
    if beat > BEATS_NOTE_MAX:
        return None
    tie = float(r[s == s[hi]].sum()) / tot
    weak: List[str] = []
    for i in np.argsort(s, kind="stable"):
        if r[i] > 1e-3 and hand_class(ALL_COMBOS[i]) not in weak:
            weak.append(hand_class(ALL_COMBOS[i]))
            if len(weak) == 5:
                break
    return (f"{hand_class(hole)} beats {beat:.0%} of {bettor}'s range"
            + (f" and ties {tie:.0%}" if tie >= 0.005 else "") + f" (weakest: {', '.join(weak)})")


def _mw_applies(remaining, cur, stk) -> bool:
    """Whether the 3-way solver can take this node (Advisor._multiway_solve), whatever streets it's set to solve:
    three players live, at most one of them all-in (the other two bet for a side pot: MWConfig.allin), the two who bet
    with chips behind (each plays its own stack: a call all-in for less plays for the main pot)."""
    if len(remaining) != 3 or len(set(remaining) & cur.allin) > 1:
        return False
    return min(stk[p] - cur.inv_start[p] for p in remaining if p not in cur.allin) > 1e-9


def _mw4_applies(remaining, cur, stk, hero) -> bool:
    """Whether the 4-player bucketed solve can take this node: four players live on the flop or turn, at most one of
    them all-in (the other three bet for a side pot), and no bet bigger than the shortest stack behind of those who
    bet (they play that stack; an all-in's bigger only when that stack is yours: _mw_fits)."""
    if len(remaining) != 4 or len(set(remaining) & cur.allin) > 1 or cur.name not in ("flop", "turn"):
        return False
    acting = [p for p in remaining if p not in cur.allin]
    S = min(stk[p] - cur.inv_start[p] for p in acting)
    return S > 1e-9 and max(cur.bets[p] for p in acting) <= S + 1e-9 and _mw_fits(remaining, cur, stk, hero)


def _mw_fits(remaining, cur, stk, hero) -> bool:
    """Whether your price fits a game where everyone with chips plays the shortest stack behind among them (the 4-way
    and N-player solves, which play every bet at most that stack): no bet, an all-in's included, is bigger than it,
    or that stack is yours (you can only call it all-in: the rest is a side pot you can't win). A deeper hero facing
    a bigger one would be priced at the short stack's call (a call of 71.5 played as 5)."""
    acting = [p for p in remaining if p not in cur.allin]
    S = min(stk[p] - cur.inv_start[p] for p in acting)
    return S > 1e-9 and (max(cur.bets[p] for p in remaining) <= S + 1e-9 or stk[hero] - cur.inv_start[hero] <= S + 1e-9)


def _nway_applies(remaining, cur, stk, hero, max_players: int) -> bool:
    """Whether the N-player solve takes the node (Advisor._multiway_solve, where _mw_applies doesn't: four or more in,
    or three with two all-in): at most `max_players` in, hero among those with chips, and on the river (no equity
    leaves: dealt-out later streets did worse than the rules on 3-way turns) or with hero the only one left with chips
    (then nothing is bet later: dealt out is exact)."""
    if len(remaining) < 3 or len(remaining) > max_players or hero in cur.allin:
        return False
    acting = [p for p in remaining if p not in cur.allin]
    if cur.name != "river" and len(acting) > 1:
        return False
    return _mw_fits(remaining, cur, stk, hero)


def _river_fix() -> np.ndarray:
    """Per class: the realization the river-dealt flop model adds over the fast one (see RIVER_FIX)."""
    global _RIVER_FIX
    if _RIVER_FIX is None:
        from .pf_classes import CLASSES
        rk = "23456789TJQKA"
        out = np.zeros(len(CLASSES))
        for i, c in enumerate(CLASSES):
            hi, lo = rk.index(c[0]), rk.index(c[1])
            if hi == lo:
                out[i] = 0.10 if hi <= rk.index("6") else 0.0
            elif hi <= rk.index("J") and hi - lo <= 2:
                out[i] = 0.15
            else:
                out[i] = RIVER_FIX_SUITED if c[2] == "s" else RIVER_FIX_OFFSUIT
        _RIVER_FIX = out
    return _RIVER_FIX


RERANK_BLEND = 0.03   # realized equity over which a re-ranked call ramps in
RERANK_SPR = 0.6      # re-rank only while the stack-to-pot ratio after calling is at least this share of the chart's


def _rerank(new: np.ndarray, base: np.ndarray, raise_n: np.ndarray, eq_r: np.ndarray, w: np.ndarray,
            better) -> np.ndarray:
    """The re-priced calls (`new`, from all-in equity against the price) moved onto realized equity, keeping
    how many hands they add or remove: a worse price (rake, a bigger raise) cuts the chart's calls with the
    least realized equity rather than the least all-in equity (raked solves fold offsuit kings and keep
    suited connectors, which all-in equity ranks the other way round), a better one adds the folds with the
    most. Raises are never cut."""
    better = np.broadcast_to(np.asarray(better), new.shape)
    out = new.copy()
    for side in (False, True):
        sel = better == side
        if not sel.any():
            continue
        target = float((w * (new - base))[sel].sum())
        if abs(target) < 1e-9:
            continue

        def moved(t):
            if side:     # adds: folds with realized equity above t
                return np.maximum(base, np.clip(0.5 + (eq_r - t) / RERANK_BLEND, 0.0, 1.0))
            return np.minimum(base, np.maximum(np.clip(0.5 + (eq_r - t) / RERANK_BLEND, 0.0, 1.0), raise_n))

        lo, hi = float(eq_r.min()) - 1.0, float(eq_r.max()) + 1.0
        for _ in range(50):           # the mass moved (adds positive, cuts negative) falls as t rises
            mid = (lo + hi) / 2
            got = float((w * (moved(mid) - base))[sel].sum())
            if got > target:
                lo = mid
            else:
                hi = mid
        out[sel] = moved((lo + hi) / 2)[sel]
    return out


def _pf_cached(key, fn):
    if key not in _PF_CACHE:
        if len(_PF_CACHE) > 512:
            _PF_CACHE.clear()
        _PF_CACHE[key] = fn()
    return _PF_CACHE[key]


def preflop_strategy(p: str, before: List[dict], positions, stack: float) -> Dict[str, np.ndarray]:
    """p's preflop strategy per combo after the actions `before`: the chart, adjusted like hero's."""
    positions = pf.at_depth(positions, stack)

    def work():
        full = pf.strategy_for(p, before, positions)
        if "call" in full or getattr(positions, "icm", None) is not None:
            st = replay_hand(positions, stack, {"preflop": before}, 1)[0]
            if not st.closed and st.to_act == p:
                full = Advisor._priced(p, before, positions, stack, st, full, []) if "call" in full else \
                    Advisor._icm_open(p, before, positions, stack, st, full, [])
        return full
    return _pf_cached(_pf_key("s", p, before, positions, stack), work)


def preflop_ranges(actions: List[dict], positions, stack: float, pool_except: Optional[str] = None,
                   locks=None) -> Dict[str, np.ndarray]:
    """Each still-active player's range after the preflop actions (pf.replay_ranges with the adjusted
    strategies). With `pool_except` (hero's seat), everyone else plays like a typical player pool
    (exploit.pool_preflop). With `locks` (your reads, {seat: {key: value}} or _lock_spec's tuple), a locked
    seat's decisions follow its lock (lock_strategy), on top of the pool's with pool_except."""
    positions = pf.at_depth(positions, stack)
    locks = _lock_spec(locks)
    locked = {p: dict(kv) for p, kv in locks}

    def work():
        ranges = {p: np.ones(N_COMBOS) for p in positions}
        active = set(positions)
        rebuilt: List[str] = []
        for i, a in enumerate(actions):
            pos, act = a["pos"], a["action"]
            if pos not in ranges:
                continue
            if act == "fold":
                active.discard(pos)
                continue
            # (each player's decisions at their own effective stack: per-seat stacks differ)
            depth = _eff_stack(positions, pos, actions[:i], stack)
            strat = preflop_strategy(pos, actions[:i], positions, depth)
            limp = act == "call" and "call" not in strat and not any(h["action"] in ("raise", "all-in") for h in actions[:i])
            if pool_except is not None and pos != pool_except:
                strat = pool_strategy(strat, pos, actions[:i], ranges)
            if pos in locked:
                strat = lock_strategy(strat, pos, actions[:i], ranges, locked[pos], positions)
            key = "raise" if act == "all-in" else ("call" if act == "check" and "check" not in strat else act)
            jam = next((h for h in reversed(actions[:i]) if h["action"] in ("raise", "all-in")), None)
            if limp:
                # An open-limp off the chart reads as the limp range, the pool's too: its own open-limps are the
                # band of hands just below its opens (CO: K7o, K8o, Q9o, K5s, Q8s, 28 combos; the small blind
                # 6-max no aces or broadways, heads-up T3o and T4o), which sets the pool's limp rate but as a
                # range is face-up (a CO limp that overbet 8x a Q43 flop was Q8/Q9 or K-high, so bottom pair
                # overcalled it). Real limpers are mostly recreational players limping a wide range.
                ranges[pos] = ranges[pos] * pf.LIMP
            elif act == "call" and jam is None and "call" in strat and pos != "BB":
                # An over-limp (or the small blind completing): at least the limp range less what would
                # raise. The chart's over-limps are what its iso-raises leave (BTN behind one limper 64s and
                # A2o-A4o, 15 combos), a face-up range where an over-limper is mostly a recreational player.
                base = preflop_strategy(pos, actions[:i], positions, depth)
                ranges[pos] = ranges[pos] * np.maximum(strat["call"], pf.LIMP * (1.0 - base.get("raise", 0.0)))
            elif act == "call" and jam is not None and jam["action"] == "all-in" and jam["pos"] != pos:
                # Calling an all-in: by equity against the jam (and anyone who called it first) versus the
                # price, as our own call is (_vs_jam). The charts' calls there are flats (the big blind
                # called a 100bb open-jam with Q2s and 43s) and lose the hands the chart raises (calling a
                # 3bet jam without QQ+ or AK).
                ji = max(j for j in range(i) if actions[j] is jam)
                jr, _ = _jam_range(actions, ji, positions, stack, pool_except, locks)
                st = replay_hand(positions, stack, {"preflop": actions[:i]}, 1)[0]
                callers = [h["pos"] for h in actions[ji + 1:i] if h["action"] in ("call", "all-in") and h["pos"] in ranges]
                eq = _preflop_equity_multi([jr] + [ranges[c] for c in callers])
                rem = depth - st.inv_start[pos] - st.bets[pos]
                to_call = min(st.cur_bet - st.bets[pos], rem)
                pot = st.pot_start + sum(st.bets.values())
                win = pot - max(0.0, st.bets[jam["pos"]] - (st.bets[pos] + to_call))
                need = to_call / max(win + to_call - _rake(positions, win + to_call), 1e-9)
                need = _icm_need(positions, st, pos, jam["pos"], to_call, need)
                call = np.nan_to_num(np.clip(0.5 + (eq - need) / JAM_BLEND, 0.0, 1.0), nan=0.0)
                r, mostly_filler = pf.thin_line(ranges[pos] * call, call, ranges[pos])
                if mostly_filler:
                    rebuilt.append(pos)
                ranges[pos] = r
            elif act == "all-in" and sum(h["action"] in ("raise", "all-in") for h in actions[:i]) <= 1:
                # an open- or 3bet-jam: the push/fold range, as our call of it reads it (_jam_range), not the
                # chart's open or 3bet (a 100bb open-jam read as CO's whole 27% open)
                ranges[pos] = _jam_range(actions, i, positions, stack, pool_except, locks)[0]
            elif key in strat:
                # an off-chart or very narrow line: filled out with the hands that take this action, so
                # the solve doesn't face a face-up hand (pf.thin_line)
                r, mostly_filler = pf.thin_line(ranges[pos] * strat[key], strat[key], ranges[pos],
                                                passive=key in ("call", "check"))
                if mostly_filler:
                    rebuilt.append(pos)
                ranges[pos] = r
        out = PreflopRanges({p: ranges[p] for p in active})
        out.rebuilt = [p for p in dict.fromkeys(rebuilt) if p in active]
        return out
    return _pf_cached(_pf_key("r", (pool_except, locks) if locks else pool_except, actions, positions, stack), work)


def _flat_ev(full: Dict[str, np.ndarray], w: np.ndarray, raiser: np.ndarray, pot: float, to_call: float,
             behind: float = 0.0) -> np.ndarray:
    """Per combo, the EV (over folding) of flatting a raise instead of jamming over it: its equity against the
    raiser's range, realized at the rate that makes the chart's weakest flats (CUT_FLOOR of them) break even,
    over the pot after the call, less the call, plus the implied odds small pairs and suited hands have on the
    stack `behind` (IMPLIED, as the priced calls); 0 for hands the chart doesn't continue with. A short-stack
    rejam competes with this, not just with folding (see push_fold's `alt`)."""
    from .pf_classes import equity_vs
    eq = np.nan_to_num(equity_vs(raiser), nan=0.0)
    cont = full.get("call", 0.0) + full.get("raise", 0.0)
    c = w * full["call"]
    if c.sum() < 1e-9:
        return np.zeros(N_COMBOS)
    order = np.argsort(eq)                              # weakest first
    cum = np.cumsum(c[order])
    marg = float(eq[order][min(int(np.searchsorted(cum, CUT_FLOOR * cum[-1])), len(cum) - 1)])
    price = to_call / max(pot + to_call, 1e-9)
    realize = float(np.clip(price / max(marg, 1e-6), 0.5, 1.2))
    ev = realize * eq * (pot + to_call) - to_call + _IMPLIED_W * max(behind, 0.0)
    return np.where(cont > 1e-9, np.maximum(ev, 0.0), 0.0)


def _put(st: "StreetState") -> Dict[str, float]:
    """Every seat's chips in the pot now (antes and blinds included)."""
    return {p: st.inv_start[p] + st.bets[p] for p in st.inv_start}


def _icm_push_vals(model, st: "StreetState", jammer: str, jam: float, callers: List[str], taker: Optional[str],
                   ranges: Dict[str, np.ndarray]):
    """A tournament's push/fold payoffs (pf_classes.push_fold's `icm`) for jammer's all-in to `jam` (a street total):
    each outcome's prize equity in chip equivalents against folding, for each player who might call (in the order they
    act, each up to their stack, from their range). Folding leaves the pot to `taker` (the last raiser, or the big
    blind)."""
    put = _put(st)
    F = model.value(put, {taker or callers[0]: sum(put.values())})
    pj = dict(put)
    pj[jammer] = st.inv_start[jammer] + jam
    S = model.value(pj, {jammer: sum(pj.values())})
    vals = []
    for c in callers:
        pc = dict(pj)
        pc[c] = min(model.stacks[c], st.inv_start[c] + max(jam, st.bets[c]))
        wj, wc = model.showdown(pc, jammer, c)
        vals.append((wj[jammer] - F[jammer], wc[jammer] - F[jammer], wc[c] - S[c], wj[c] - S[c]))
    return S[jammer] - F[jammer], vals, [ranges.get(c, np.ones(N_COMBOS)) for c in callers]


def _icm_need(positions, st: "StreetState", hero: str, jammer: str, to_call: float, need: float) -> float:
    """The equity hero's call of an all-in needs: in a tournament, the prize equity's (folding leaves the pot to the
    jammer; anyone else in is dead money), else `need` (the chips')."""
    model = getattr(positions, "icm", None)
    if model is None:
        return need
    return model.call_need(_put(st), hero, jammer, to_call)[0]


def _icm_premium(positions, st: "StreetState", hero: str, villains: List[str], call: float, pot: float,
                 behind: float) -> Tuple[float, float]:
    """(premium, all-in premium): the risk premium (gto/icm.py Model.premium: the extra equity ICM asks of getting
    the stacks in, averaged over `villains`) scaled by how much of the stack a pot this deep tends to take,
    min(1, ICM_COMMIT / (1 + SPR)) after putting in `call` (all of it once the stacks are in; 0.2 at 4 pots behind,
    0.05 at 20). (0, 0) without a tournament."""
    model = getattr(positions, "icm", None)
    vs = [v for v in villains if v != hero]
    if model is None or not vs:
        return 0.0, 0.0
    put = _put(st)
    full = float(np.mean([model.premium(put, hero, v) for v in vs]))
    spr = max(behind, 0.0) / max(pot + call, 1e-9)
    return full * min(1.0, ICM_COMMIT / (1.0 + spr)), full


def _jam_range(actions: List[dict], ji: int, positions, stack: float, pool_except: Optional[str] = None, locks=None):
    """(the range of the player whose all-in is actions[ji], notes). An open- or 3bet-jam: push/fold
    equilibrium against the players left to act (the opener, if any, is the one who calls); later, the
    replayed (or the pool's, or your lock's) raising range through the jam. `locks`: as preflop_ranges'."""
    jammer, before = actions[ji]["pos"], actions[:ji]
    prev = replay_hand(positions, stack, {"preflop": before}, 1)[0]
    arr = preflop_ranges(before, positions, stack, pool_except, locks)
    rng = arr.get(jammer, np.ones(N_COMBOS))
    notes = []
    if prev.n_raises <= 1:
        jam = replay_hand(positions, stack, {"preflop": actions[:ji + 1]}, 1)[0].bets[jammer]
        opp = [p for p in prev.order if p in prev.active and p != jammer]
        caller = next((p for p in opp if prev.bets[p] >= prev.cur_bet - 1e-9 and prev.n_raises), None)
        call_rng = arr.get(caller, np.ones(N_COMBOS)) if caller else np.ones(N_COMBOS)
        caller_in = prev.bets[caller] if caller else float(np.mean([prev.bets[p] for p in opp]))
        pot_before = prev.pot_start + sum(prev.bets.values())
        # a jam over a raise competes with flatting it, as our own (Advisor._push)
        alt = None
        if prev.n_raises == 1 and caller:
            full_j = pf.strategy_for(jammer, before, positions)
            to_call = min(prev.cur_bet - prev.bets[jammer], jam - prev.bets[jammer])
            if "call" in full_j and to_call > 1e-9:
                alt = _flat_ev(full_j, rng, call_rng, pot_before, to_call,
                               _eff_stack(positions, jammer, before, stack) - prev.inv_start[jammer]
                               - prev.bets[jammer] - to_call)
        model = getattr(positions, "icm", None)
        vals = _icm_push_vals(model, prev, jammer, jam, [caller] if caller else opp,
                              caller or ("BB" if "BB" in opp else opp[-1] if opp else None), arr) \
            if model is not None and opp else None
        push = push_fold(rng, call_rng, jam, prev.bets[jammer], caller_in, pot_before, len(opp),
                         rake=getattr(positions, "rake", None), alt=alt, icm=vals)
        share = float((rng * push).sum() / max(rng.sum(), 1e-12))
        rng = rng * push
        notes.append(f"{jammer}'s {jam:g}bb jam range: push/fold equilibrium against {len(opp)} "
                     f"player{'s' * (len(opp) > 1)} left ({share:.0%} of their hands)")
    else:
        # the replayed range through the jam: the chart's (or the pool's) raising range, rebuilt
        # from it alone when the jammer's line is off-chart (a flat, then a jam over the 4bet)
        jr = preflop_ranges(actions[:ji + 1], positions, stack, pool_except, locks)
        rng = jr.get(jammer, rng)
        if any(p == jammer for p, _ in _lock_spec(locks)):
            notes.append(f"locked: {jammer}'s jam is their raising range as you locked it")
        elif pool_except is not None:
            notes.append(f"exploit mode: {jammer}'s jam is the player pool's raising range")
        if jammer in getattr(jr, "rebuilt", []):
            notes.append(f"{jammer}'s line isn't in the chart, so their jam range is the hands that jam "
                         "there from scratch")
    return rng, notes


def _next_lock_notes(node, nxt, lock, vr, villain: str, adj: dict) -> List[str]:
    """For each of hero's actions leading to a decision of the villain's that a next_* lock sets: what share of
    their range takes the locked actions there, locked and at equilibrium."""
    out = []
    tot = max(float(vr.sum()), 1e-12)
    for a, c in enumerate(node.children):
        if not any(c is x for x in nxt):
            continue
        facing = c.names[0] == "fold"
        keys = [k for k in (("next_fold", "next_raise") if facing else ("next_bet",)) if k in adj]
        if not keys:
            continue
        g = c.avg_strategy()
        lk = lock(c, g, vr[None, :])
        parts = []
        for key in keys:
            acts = [0] if key == "next_fold" else list(range(2 if facing else 1, len(c.names)))
            if not acts:
                continue
            word = {"next_fold": "folds", "next_raise": "raises", "next_bet": "bets"}[key]
            share = lambda s: float((vr * s[0, acts, :].sum(axis=0)).sum()) / tot * 100
            parts.append(f"{word} {share(lk):.0f}% (GTO {share(g):.0f}%)")
        if parts:
            out.append(f"locked: after your {node.names[a]}, {villain} {' and '.join(parts)}")
    return out


def _committed(p: str, streets) -> bool:
    """Whether p raised a bet, or bet COMMIT_BET of the pot or more, on these postflop streets (StreetStates)."""
    for st in streets:
        bets = dict.fromkeys(st.inv_start, 0.0)
        cur = 0.0
        for a in st.actions:
            q, act = a["pos"], a["action"]
            if q not in bets:
                continue
            amt = a.get("to") if a.get("to") is not None else a.get("amount")
            if act in ("bet", "raise", "all-in"):
                to = float(amt) if amt is not None and act != "all-in" else float("inf")
                pot = st.pot_start + sum(bets.values())
                if q == p and (cur > 1e-9 or to - bets[q] >= COMMIT_BET * pot - 1e-9):
                    return True
                if to != float("inf"):
                    cur = max(cur, to); bets[q] = to
            elif act == "call":
                bets[q] = cur
    return False


class PreflopRanges(dict):
    """preflop_ranges' result; `rebuilt`: seats whose line left their range nearly empty, so it is mostly
    the hands that take their last such action (pf.thin_line)."""
    rebuilt: List[str] = []


def _post_floor(before: np.ndarray, after: np.ndarray, board, pos: str, street: str, notes: List[str]) -> np.ndarray:
    """A player's range after an earlier street's actions, filled out when the line leaves fewer than
    preflop.MIN_LINE_COMBOS live combos (counting the cards dealt since): the next street's solve would
    otherwise play against a face-up hand (a 4bet pot's ~12-combo range checking back the flop left 2 for
    the turn). The filler is the rest of the range the player started the street with, weighted as
    preflop (pf.thin_line: 1 - combos / MIN_LINE_COMBOS); the hands that took the line are kept."""
    live = np.ones(N_COMBOS)
    for c in board:
        live = live * (1.0 - MEMBER_FULL[c])
    r, _ = pf.thin_line(after * live, before * live)
    if r.sum() > (after * live).sum() + 1e-9:
        notes.append(f"{street}: {pos}'s line left {float((after * live).sum()):.1f} combos, so their range "
                     f"is filled out with the rest of their {street} range")
    return r


def pool_strategy(strat: Dict[str, np.ndarray], p: str, before: List[dict], ranges: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """p's preflop strategy `strat` (after the actions `before`) as a player-pool player would play it:
    exploit.pool_preflop, with p's equity against the last raiser to pick the hands that move (facing an
    open: its extra 3bets and flats; the opener facing a 3bet: its extra calls). `ranges`: everyone's
    range at that point."""
    raisers = [h["pos"] for h in before if h["action"] in ("raise", "all-in")]
    # (not an iso-raiser facing the limper's re-raise: that's QQ+/AK, and the pool's extra 3bet defence
    # had the button calling it with 87 combos, 22-QQ, where the chart calls 22)
    limped = len(raisers) == 2 and any(h["pos"] == raisers[1] and h["action"] == "call"
                                       for h in before[:next(i for i, h in enumerate(before) if h["action"] in ("raise", "all-in"))])
    opener = len(raisers) == 2 and raisers[0] == p and not limped
    first_in = not raisers and not any(h["action"] == "call" for h in before)
    eq = None
    if (len(raisers) == 1 or opener) and raisers[-1] in ranges:
        eq = _preflop_equity(ranges[raisers[-1]])
    elif first_in:
        eq = _equity_vs_any()
    from .exploit import pool_preflop
    return pool_preflop(strat, len(raisers), p, ranges.get(p, np.ones(N_COMBOS)), eq, opener, first_in,
                        heads_up=len(ranges) == 2)


def _lock_spec(locks) -> tuple:
    """Your locks (exploit.parse_locks' {seat: {key: value}}, or this tuple already) as far as they act preflop
    (exploit.LOCK_PREFLOP_KEYS), frozen for the preflop caches: ((seat, ((key, value), ...)), ...); () for none."""
    if not locks:
        return ()
    if isinstance(locks, tuple):
        return locks
    from .exploit import LOCK_PREFLOP_KEYS
    out = []
    for p, adj in sorted(locks.items()):
        kv = tuple((k, float(adj[k])) for k in LOCK_PREFLOP_KEYS if adj.get(k))
        if kv:
            out.append((p, kv))
    return tuple(out)


def _quantile(eq: np.ndarray, w: np.ndarray, share: float) -> float:
    """The equity of the hand at `share` of the range `w` counted from the top by `eq` (blocked combos left out)."""
    ok = ~np.isnan(eq) & (w > 0)
    order = np.argsort(-eq[ok])
    cum = np.cumsum(w[ok][order])
    return float(eq[ok][order][min(np.searchsorted(cum, share * cum[-1]), len(cum) - 1)])


def lock_strategy(strat: Dict[str, np.ndarray], p: str, before: List[dict], ranges: Dict[str, np.ndarray],
                  adj: Dict[str, float], positions) -> Dict[str, np.ndarray]:
    """p's preflop strategy `strat` (after the actions `before`) with your lock `adj` on them (exploit.lock_preflop):
    calls and folds move by p's equity against the last raiser's range (against any hand when nobody has raised),
    raises by p's equity against what the last raiser continues with facing p's raise (their chart there)."""
    raisers = [h["pos"] for h in before if h["action"] in ("raise", "all-in")]
    eq = eq_raise = None
    if not raisers:
        eq = _equity_vs_any()
    elif raisers[-1] in ranges:
        last = raisers[-1]
        eq = eq_raise = _preflop_equity(ranges[last])
        if "raise" in strat:
            s_last = pf.strategy_for(last, before + [{"pos": p, "action": "raise"}], positions)
            cont = ranges[last] * (1.0 - s_last.get("fold", 0.0))
            if cont.sum() >= 1.0:
                eq_raise = _preflop_equity(cont)
    from .exploit import lock_preflop
    return lock_preflop(strat, adj, len(raisers), ranges.get(p, np.ones(N_COMBOS)), eq, eq_raise)


_EQ_ANY = None


def _equity_vs_any() -> np.ndarray:
    """Each combo's preflop equity against a random hand (how a pool player ranks hands to limp)."""
    global _EQ_ANY
    if _EQ_ANY is None:
        _EQ_ANY = _preflop_equity(np.ones(N_COMBOS))
    return _EQ_ANY


def _held_bytes(obj) -> int:
    """Bytes of the numpy arrays reachable from obj, each counted once: about what dropping a cached solve frees
    (arrays it shares with module-level tables count too). Functions, methods and classes aren't followed."""
    seen, todo, total = set(), [obj], 0
    skip = (str, bytes, int, float, bool, type(None), type, types.ModuleType, types.FunctionType,
            types.BuiltinFunctionType, types.MethodType, functools.partial)
    while todo:
        o = todo.pop()
        if id(o) in seen or isinstance(o, skip):
            continue
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            if isinstance(o.base, np.ndarray):
                todo.append(o.base)
            else:
                total += o.nbytes
        elif isinstance(o, dict):
            todo.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            todo.extend(o)
        else:
            todo.extend(getattr(o, "__dict__", {}).values())
            for t in type(o).__mro__:
                sl = t.__dict__.get("__slots__", ())
                todo.extend(getattr(o, a) for a in ((sl,) if isinstance(sl, str) else sl) if hasattr(o, a))
    return total


def _range_key(v: np.ndarray) -> str:
    return hashlib.md5(np.round(v, 6).tobytes()).hexdigest()[:12]


def _mw_pick(node, a: dict) -> Optional[int]:
    """The 3-way tree's action at `node` for the real action `a`: by kind (bets and raises, all-in for either), the
    size nearest the one made (the solve's all-in for a deeper player is a bet to the shortest stack, played as that
    bet: _mw_names), the biggest for an all-in; None when the tree has no such kind."""
    kind = {"bet": ("bet", "all-in"), "raise": ("raise", "all-in"), "all-in": ("all-in", "raise", "bet")}.get(
        a["action"], (a["action"],))
    cand = [i for i, nm in enumerate(node.names) if nm.split()[0] in kind]
    if not cand:
        return None
    to = 1e9 if a["action"] == "all-in" else a.get("to") if a.get("to") is not None else a.get("amount")
    return cand[0] if to is None else min(cand, key=lambda i: (abs((node.amounts[i] or 0.0) - float(to)),
                                                               kind.index(node.names[i].split()[0])))


def _mw_line(board, hero, hole, stk, streets, street, n) -> tuple:
    """A 3-way solve's place in the hand: who it was solved for, the cards, the stacks, and every action before its
    root (the earlier streets', and this street's first n)."""
    norm = lambda acts: tuple((a.get("pos"), a.get("action"), round(float(a.get("amount") or a.get("to") or 0.0), 2))
                              for a in acts or [])
    before = tuple(norm(streets.get(st)) for st in STREETS[:STREETS.index(street)])
    return (tuple(board), hero, tuple(sorted(hole)), tuple(sorted((k, round(v, 2)) for k, v in stk.items())),
            before, norm((streets.get(street) or [])[:n]))


def _unusable(res) -> Optional[str]:
    """Why an answer has nothing usable in it (None when it has, or it's an error about the request, a skip or a
    cancel): no actions, a recommendation that isn't one of them, a number that isn't finite, or an error the solve
    itself gave (_SOLVE_ERRORS). Advisor.advise's catch-all answers these by the betting rules."""
    if not isinstance(res, dict):
        return "no answer"
    if "error" in res:
        return f"the solve gave no answer ({res['error']})" if str(res["error"]).startswith(_SOLVE_ERRORS) else None
    if any(k in res for k in ("skipped", "cancelled", "one_size")) or res.get("study") in ("deal", "over", "runouts"):
        return None
    acts = res.get("actions")
    if not acts or not isinstance(acts, list):
        return "an answer with no actions"
    if res.get("study") == "node":
        return None   # (Study's range answer: no one hand's pick)
    if res.get("recommended") not in [a.get("action") for a in acts]:
        return f"the recommended action ({res.get('recommended')}) isn't one of the answer's actions"
    for a in acts:
        for k in ("freq", "ev_bb", "amount_bb"):
            v = a.get(k)
            if v is not None and not np.isfinite(v):
                return f"{a.get('action')}'s {k} is {v}"
    if not sum(a.get("freq") or 0.0 for a in acts) > 0:
        return "every action's frequency is 0"
    return None


# errors an answer can carry that are the solve's, not the request's: Advisor.advise answers by the rules instead
_SOLVE_ERRORS = ("no decision pending",)


def _one_size_failed(why) -> dict:
    """A one-size request (state "one_size") whose answer failed: an error (the betting rules' advice doesn't answer
    it)."""
    reason = why if isinstance(why, str) else f"{type(why).__name__}: {why}"
    return {"error": f"the one-size view couldn't be worked out here ({reason})"[:500]}


def _legal_to(st: "StreetState", p: str, to: float, remaining: float) -> float:
    """A bet or raise-to as the page takes it (index.html act): to the nearest half big blind (half to even, as the
    rules always sized: the page keeps a half big blind as it is), at least the smallest full raise (the bet plus the
    last full raise; 1bb to bet) rounded up to a half big blind, at most all-in (an all-in for less always goes).
    Rounding the size alone could leave it under the smallest raise (1.25 to 1.0) or over the stack (an all-in to 0.85
    to 1.0)."""
    lo = np.ceil((st.cur_bet + st.min_raise) * 2 - 1e-9) / 2
    return float(min(max(round(float(to) * 2) / 2, lo), st.bets[p] + remaining))


def _mw_names(names, amounts, behind: float, S: float, pot_now: float) -> List[str]:
    """A multiway solve's action names as the player plays them. The solve plays the player's stack as far as another
    can match it (four or more: everyone at the shortest stack behind), S, so its all-in for a player with more is a
    bet (or raise) to S, not their whole stack: recorded as all-in on the page, a deep player's all-in put 97.5 bb in
    where the solve bet 9.5 (a short stack's cap)."""
    out = list(names)
    facing = bool(names) and names[0] == "fold"   # (fold is offered only facing a bet; else nothing is in this street)
    for i, (nm, amt) in enumerate(zip(names, amounts)):
        if nm == "all-in" and amt is not None and behind > S + 1e-6:
            alt = f"raise to {amt:g}" if facing else f"bet {int(round(100 * amt / max(pot_now, 1e-9)))}%"
            out[i] = alt if alt not in out else f"{'raise to' if facing else 'bet'} {amt:g}bb"
    return out


# ----------------------------------------------------------------------------- instant answers
class _NoInstant(Exception):
    """Advisor.instant: the answer needs a solve, or is quick anyway (nothing to answer ahead of it)."""


class _NotHeld(Exception):
    """Study's report from what's in hand ("runouts": "held"): a solve it needs isn't cached (nothing is solved)."""


@dataclass
class InstantSpot:
    """A heads-up street the hero acts on, for Advisor.instant's sources: the cards, the street's real actions so far
    (`actions`, the page's), the pot at its start and the effective stack (chips), hero's seat (0 = OOP), `own`: the
    street's own solve when one is cached (one stopped part way, or short of its budget), `prev`: (solution, the node
    the previous street's line ends at, the tree's names of the actions taken there) when that street was walked
    through a live solve or the flop library this request, else None."""
    street: str
    board: tuple
    hole: tuple
    hero_p: int
    oop: str
    ip: str
    actions: list
    pot: float
    eff: float
    behind: float
    own: Optional[Solution] = None
    prev: Optional[tuple] = None
    own_earlier: bool = False   # `own` is the street's solve from an earlier decision, its tree without a size since taken
    near: Optional[Callable] = None   # a library spot's flop not stored: () -> flop_approx.ApproxSolution or None


def _swap_combo(hole, a: int, b: int) -> int:
    """The index of `hole`'s image under the swap of cards a and b (a class's played card and the card dealt)."""
    x, y = [b if c == a else a if c == b else c for c in hole]
    return COMBO_INDEX[(min(x, y), max(x, y))]


def _read_line(node, row: int, col: int, spot: InstantSpot):
    """Walks the street's real actions (spot.actions) down a solve's tree from `node` (the street's first decision;
    batch row `row`, hero's column `col`: their combo or bucket) by the tree's nearest actions; at hero's decision
    reached: (that decision, hero's average mix there, the real bets then), else None."""
    bets = [0.0, 0.0]
    for a in spot.actions:
        if not isinstance(node, Decision):
            return None
        p = 0 if a["pos"] == spot.oop else 1
        if p != node.player:
            return None
        name = Advisor._tree_action_name(node, a, p, bets, spot.pot, spot.eff)
        k = match_action(node.names, name)
        act = a["action"]
        amt = a.get("to") if a.get("to") is not None else a.get("amount")
        if act in ("bet", "raise") and amt is not None:
            bets[p] = float(amt)
        elif act == "all-in":
            bets[p] = spot.eff
        elif act == "call":
            bets[p] = min(bets[1 - p], spot.eff)
        node = node.children[k]
    if not isinstance(node, Decision) or node.player != spot.hero_p:
        return None
    s = node.avg_strategy()[row][:, col]
    return node, np.asarray(s, dtype=np.float64), bets


def _real_actions(node: Decision, strat, bets, spot: InstantSpot) -> List[dict]:
    """Hero's actions at a tree decision `node` read for the hand's real pot and stack: a bet as its share of the pot,
    a raise scaled by the pots' ratio (to the real bets), each amount at most hero's stack."""
    p = spot.hero_p
    pt = spot.pot + bets[0] + bets[1]
    cap = bets[p] + spot.behind
    scale = pt / max(node.pot_now, 1e-9)
    out = []
    for a, name in enumerate(node.names):
        amt = None
        if name == "call":
            amt = round(min(bets[1 - p], cap) - bets[p], 2)
        elif name == "all-in":
            amt = round(cap, 2)
        elif name.startswith("bet"):
            amt = round(min(bets[p] + float(name.split()[-1].rstrip("%")) / 100.0 * pt, cap), 2)
        elif name.startswith("raise"):
            amt = round(min(max(bets) + (float(name.split()[-1]) - max(node.bets)) * scale, cap), 2)
            name = f"raise to {amt:g}"
        out.append({"action": name, "freq": round(float(strat[a]), 3), "amount_bb": amt})
    return out


def instant_from_own(adv: "Advisor", spot: InstantSpot) -> Optional[dict]:
    """The street's own solve, cached short of its budget (a request stopped part way, a quick answer's), at hero's
    decision: its tree and mix as they stand (from HU_CHECKS[0] iterations on)."""
    sol = spot.own
    if sol is None or sol.solver.iters < HU_CHECKS[0] or not hole_col(sol, spot.hole):
        return None
    got = _read_line(sol.root, 0, hole_col(sol, spot.hole)[0], spot)
    if got is None:
        return None
    node, strat, bets = got
    label = (f"this street's solve from your earlier decision ({sol.solver.iters} iterations; the sizes since read at "
             f"its nearest)" if spot.own_earlier else f"this street's solve so far ({sol.solver.iters} iterations)")
    return {"actions": _real_actions(node, strat, bets, spot), "source": "solve", "label": label,
            "iterations": int(sol.solver.iters)}


def hole_col(sol: Solution, hole, combo: Optional[int] = None):
    """[hero's column in sol's combos] (`combo`: that combo's instead), or [] when the solve doesn't hold it."""
    i = COMBO_INDEX[(min(hole), max(hole))] if combo is None else combo
    return [int(x) for x in np.nonzero(sol.ctx.active == i)[0][:1]]


def instant_from_prev(adv: "Advisor", spot: InstantSpot) -> Optional[dict]:
    """The previous street's live solve (the flop's, or the turn's), which plays this street under it: a flop solve the
    turn by card class (Ctx.turn_classes: a rank's cards alike, a flush suit's apart) and hand buckets, one bet size
    (Config.later_sizes); a turn solve the river by card class, per combo, at its river sizes. The card dealt is read
    as its class's played card, hero's combo through the swap of the two (Config.river_classes, solver.lift_classes).
    Not the flop library's solves (their entries hold the flop only)."""
    if spot.prev is None:
        return None
    sol, ch, _ = spot.prev
    if not isinstance(sol, Solution) or hasattr(sol, "stored") or not isinstance(ch, Chance) or len(ch.boards) != 1:
        return None
    dec, card = ch.child, spot.board[-1]
    rows = dec.boards
    if len(rows) != ch.K or isinstance(dec, RiverEntry):
        return None
    if ch.umasks is not None or ch.perm is not None:
        classes = sol.ctx.turn_classes(tuple(ch.boards[0]), sol.solver.dead)
        if [c for c, _ in classes] != [b[-1] for b in rows]:
            return None
        k = next((i for i, (_, cs) in enumerate(classes) if card in cs), None)
    else:
        k = next((i for i, b in enumerate(rows) if b[-1] == card), None)
    if k is None:
        return None
    col = hole_col(sol, spot.hole, _swap_combo(spot.hole, rows[k][-1], card))
    if not col:
        return None
    col = col[0]
    if isinstance(dec, BucketEntry):
        T, ids = sol.solver._btab(dec)[:2]
        dec = dec.child
        col = int(ids[k % T, spot.hero_p][col])
        if col >= dec.avg_strategy().shape[2]:
            return None   # (dead on the class's played card)
    got = _read_line(dec, k, col, spot)
    if got is None:
        return None
    node, strat, bets = got
    nxt = spot.street
    how = ("by card class and hand buckets, one bet size" if node.bucketed else "by card class, its river sizes")
    return {"actions": _real_actions(node, strat, bets, spot), "source": "prev",
            "label": f"the {STREETS[STREETS.index(nxt) - 1]} solve's view of the {nxt} ({how})",
            "iterations": int(sol.iters)}


# A library spot's flop the library doesn't store is read off its stored flops most like it (instant_from_near) until
# the flop's own solve in hand has run this many iterations, from when instant_from_own reads it: EV given up at the
# stored decisions (scripts/bench_flop_seed.py: 104 of the flatted opens' and 3bet pots' stored flops solved live in
# the Fast tree, priced by their entries' EVs, % of the pot) the stored flops like it 0.41-0.42 (mean), the solve
# 0.63-0.70 at 20 iterations, 0.32-0.35 at 30, 0.21 at 40, 0.12-0.13 at 60 (better on 40-44%, 50-59%, 68-81% of the
# flops at 30, 40, 60)
NEAR_UNTIL = HU_CHECKS[0]


def instant_from_near(adv: "Advisor", spot: InstantSpot) -> Optional[dict]:
    """A flop of a library spot that the library doesn't store (the flatted opens' and the 3bet pots' other 1203): its
    decisions read off the spot's stored flops most like it (gto/flop_approx.py: the same pairing and suit pattern, the
    ranks nearest, each carried over by card substitution and their mixes averaged), on the line the hand took (sizes
    at the tree's nearest); not once the flop's own solve in hand has run NEAR_UNTIL iterations (instant_from_own reads
    that). None where hero's decision isn't one an entry stores or no stored flop holds an image of hero's combo."""
    if spot.street != "flop" or spot.near is None:
        return None
    if spot.own is not None and spot.own.solver.iters >= NEAR_UNTIL:
        return None
    sol = spot.near()
    if sol is None:
        return None
    col = hole_col(sol, spot.hole)
    if not col:
        return None
    got = _read_line(sol.root, 0, col[0], spot)
    if got is None:
        return None
    node, strat, bets = got
    if id(node) not in sol.stored or not sol.known[id(node)][col[0]]:
        return None
    flops = ", ".join(nm for nm, _ in sol.near)
    return {"actions": _real_actions(node, strat, bets, spot), "source": "near",
            "label": f"the precomputed solves of the most similar flops ({flops}), by card substitution",
            "iterations": None}


# the stored turn answers before the turn's own solve in hand until that has run this many iterations: EV given up
# (each combo's mix priced by the turn's own Fast solve run to 600 iterations; 29 turns of four BTN vs BB flops, % of
# the pot) the stored turn 0.04 first to act / 0.08 in position after a check, the solve at 50 iterations 0.09 / 0.06,
# at 100 0.02 / 0.02 (README, Accuracy vs speed: "The turn after a library flop")
STORE_UNTIL = 100
# A live turn solve after a library flop starts from the flop solve's stored turn where the entry's turn file holds the
# line the hand took (Advisor._store_seed): at the turn's first decision and in position after a check each stored
# combo's regrets are STORE_SEED x the pot x the opponent's reach compatible with it x its stored mix (each stored
# action at the live tree's nearest), so the first iteration plays the stored mix; nothing else is seeded. On the
# threaded browser's tree (30 turns of four BTN vs BB flops, natively) the settled stop (0.2% of the pot from the
# 300th) fires at 300 on 28 of 30 turns where a cold start did on 13 (mean 357 -> 307 iterations): there 0.142 ->
# 0.160% of the pot from equilibrium, each combo's EVs over the actions it plays 0.269 -> 0.175% of the pot off the
# 1000th's, B3's given-up 0.0017 / 0.0015 -> 0.0012 / 0.0009 (README, "The turn solve starts from the stored turn")
STORE_SEED = 1.0


def _turn_file(lib, sol, flop) -> Optional[tuple]:
    """(the decoded turn file, the hand's flop's suit map) beside the library entry `sol` (a LibrarySolution) was read
    from, or None: not a library flop, the index lists no turn file for it, or it isn't the entry's."""
    head = getattr(sol, "head", None)
    if lib is None or not hasattr(lib, "turns") or not hasattr(sol, "stored") or not isinstance(head, dict):
        return None
    from . import flop_library as fl
    canon, smap = fl.canonical(flop)
    name = cards_str(canon)
    sp = next((s for s in lib.index().get("spots", ()) if s.get("id") == head.get("spot")), None)
    if name != head.get("board") or sp is None:
        return None
    t = lib.turns(sp, name)
    return None if t is None else (t, smap)


def instant_from_store(adv: "Advisor", spot: InstantSpot) -> Optional[dict]:
    """A library flop's turn: the precomputed flop solve's own turn strategies, stored beside its entry (the entry's
    turn file: flop_library.encode_turns, FlopLibrary.turns, turn_node), on the flop lines most hands take: the
    turn's first decision, and in position after a check, at the flop solve's turn sizes (a 75% bet), the card dealt
    read as its class's played card (each combo the same combo's; one holding the played card its image's holding
    the card dealt). On the line the flop's walk took (its sizes at their nearest) and on the entry's canonical flop;
    not once the turn's own solve in hand has run STORE_UNTIL iterations (instant_from_own reads that). Against a live
    Accurate turn with the opponent free to adapt (BTN vs BB on Kc7d2h) it gives up 0.50 / 0.67% of the pot out of /
    in position on the played cards, 0.81 / 0.63 on the others; priced by the turn's own solve's EVs 0.04 / 0.08."""
    lib = adv.library
    if spot.street != "turn" or spot.prev is None or lib is None or not hasattr(lib, "turns"):
        return None
    if spot.own is not None and spot.own.solver.iters >= STORE_UNTIL:
        return None   # (the turn's own solve, a presolve's or one stopped part way, is the closer read by now)
    sol, _, taken = spot.prev
    if not spot.actions:
        at = ()
    elif len(spot.actions) == 1 and spot.actions[0]["action"] == "check" and spot.actions[0]["pos"] == spot.oop:
        at = ("check",)
    else:
        return None
    got = _turn_file(lib, sol, spot.board[:3])
    if got is None:
        return None
    from . import flop_library as fl
    t, smap = got
    mp = lambda c: c // 4 * 4 + smap[c % 4]
    nd = fl.turn_node(t, taken, at, mp(spot.board[3]))
    if nd is None or nd["player"] != spot.hero_p:
        return None
    x, y = sorted(mp(c) for c in spot.hole)
    i = COMBO_INDEX[(x, y)]
    j = int(np.searchsorted(nd["combos"], i))
    if j >= len(nd["combos"]) or nd["combos"][j] != i:
        return None   # (a combo the flop solve hardly brings here: not stored)
    node = types.SimpleNamespace(names=nd["names"], pot_now=0.0, bets=(0.0, 0.0))
    bets = [n for n in nd["names"] if n.startswith("bet")]
    size = f"one bet size, {bets[0].split()[-1]}" if len(bets) == 1 else "its turn sizes"
    return {"actions": _real_actions(node, nd["strat"][:, j], [0.0, 0.0], spot), "source": "store",
            "label": f"the precomputed flop solve's view of the turn (by card class, {size})",
            "iterations": int(sol.iters)}


# ----------------------------------------------------------------------------- advisor
class Advisor:
    def __init__(self, cfg: Optional[Config] = None, time_budget: float = 4.0, prior_budget: float = 2.0,
                 prior_smoothing: float = 0.15):
        self.cfg = cfg or Config()
        self.time_budget = time_budget
        self.prior_budget = prior_budget
        self.prior_smoothing = prior_smoothing
        self._cache: Dict[tuple, Solution] = {}
        # the solves in _cache and _mw_cache by (advisor, store, key) -> bytes of their arrays, least recently used first:
        # once they hold more than cache_mb the oldest go (_keep). A later street then solves an earlier one again to
        # narrow through it, as with no advice on it. Advisors can share one (native.advisors and the browser's worker:
        # Fast's and Accurate's), so switching speed drops the other's old hands first.
        self.cache_mb = CACHE_MB
        self._held: Dict[tuple, int] = {}
        self.should_stop = None   # callable; when it returns true the running solve raises solver.Cancelled
        self._presolving = False   # (Advisor.presolve running)
        self._presolved = set()    # cache keys of solves a presolve made that no request of the hand has asked for yet
        # a heads-up solve (the street advised on, or an earlier one narrowing its ranges) stops once its
        # exploitability is under this, % of the pot (read at HU_CHECKS), instead of running its whole budget; None
        # runs the budget
        self.hu_settled: Optional[float] = None
        self.hu_settled_from: Dict[str, int] = dict(HU_SETTLED_FROM)   # (by street; "shallow": spr_sizes' turn)
        # what this speed's heads-up answers after the flop reach, % of the pot from equilibrium: one over OVER_TARGET
        # times it says so ("approximate" and a note: _flag_short); None flags nothing. Its solve ran short: a busy or
        # slow device, a big tree, or the compiled core failed and numpy answered (the browser's 5-7% on a Fast turn,
        # against its 0.2)
        self.expl_target: Optional[float] = None
        # a new heads-up solve's tree is cut to fit (MAX_CELLS): at most this many cells, and on the turn and river
        # what CAP_ITERS iterations at cell_s seconds a cell (updated from the solves run here) fit in its budget
        self.max_cells: Optional[int] = MAX_CELLS
        self.cell_s = CELL_S
        self.mw_rake_rules = True   # multiway: our in-trial calls, leads and raises count the rake (off: payouts only)
        self.mw_trials: Optional[int] = None   # multiway: play at least this many trials (whole passes), no time caps or early stop (tests)
        # 3-way pots: the streets solved by the 3-player solver (gto/multiway_solver.py; others, and 4+ players off the river, use the
        # betting rules), MWConfig overrides, and the most iterations a solve runs (it stops at time_budget before)
        self.mw_solve_streets: Tuple[str, ...] = ("turn", "river")   # (native and browser Accurate add the flop)
        self.mw_config: Optional[dict] = None
        # per street MWConfig overrides: from the flop and turn the later streets are bet on by hand buckets (dealt out
        # as equity leaves, a flop solve bet half as often as published solves: README, 3-way pots)
        self.mw_street_config: Dict[str, dict] = {"flop": dict(MW_LATER, bet=MW_FLOP_BETS),
                                                  "turn": dict(MW_LATER, bet=MW_TURN_BETS)}
        self.mw_max_iters: int = 1000
        self.mw_budget: Optional[float] = None   # a 3-way solve's budget when longer than time_budget (native Fast's 2.5 s)
        self.mw_max_expl: float = 10.0   # a solve further than this from equilibrium (% pot) hands over to the rules
        # the N-player solve (four or more in, or two all-in): at most this many players (the browser's numpy takes 5-way
        # rivers past its budget: gto/static/worker.js lowers it)
        self.mw_nway_max: int = 6
        # 4-way flops and turns (_mw4_applies): the 4-player solve's game per street (MW4_STREETS; Fast MW4_FAST_STREETS),
        # on the streets in mw_solve_streets
        self.mw4_street_config: Dict[str, dict] = {s: dict(c) for s, c in MW4_STREETS.items()}
        # ...and a quick answer's 4-way flop (None: the betting rules; the browser's, where its tables alone take the
        # budget: gto/static/worker.js)
        self.mw4_quick_flop: Optional[dict] = dict(MW4_QUICK_FLOP)
        # an earlier 3-way street with no solve from your advice on it: solve it from its start (prior_budget) to narrow
        # the ranges through (_mw_walk); off: the rules narrow it (native Accurate on: its 12 s settle a flop enough)
        self.mw_prior_solve = False
        self._mw_cache: Dict[tuple, object] = {}
        self._mw_lines: Dict[tuple, tuple] = {}   # the hand's line up to a 3-way solve's root -> (solve, players)
        self.warm_from: Optional["Advisor"] = None   # another advisor (e.g. a faster config) whose solves seed ours
        # Config fields for the earlier streets' solves that narrow ranges (prior_budget each), e.g. the browser's
        # Accurate keeps their flop's river checked down: 12 s there is ~10 iterations with river_play
        self.prior_model: Optional[dict] = None
        # a street's bet and raise sizes when a solve starts on it ({"river": {"bet": [...], "raise": [...]}}), in place
        # of cfg's, which also shape that street inside an earlier street's solve: a river root solves in milliseconds
        # an iteration, so it can afford the overbets a polar river uses that a turn solve's rivers can't (README,
        # Accuracy vs speed). Sizes the request sets for that street (state["sizes"]) win.
        self.root_sizes: Dict[str, Dict[str, List[float]]] = {}
        # a street's sizes in a shallow pot: {"turn": (max_spr, {"bet": [...]})} replaces cfg's (and root_sizes') where
        # the street starts with at most max_spr x its pot behind (a 3bet pot's turn), for the street's own solve and
        # an earlier street's solve of it alike. Sizes the request sets for that street win.
        self.spr_sizes: Dict[str, Tuple[float, Dict[str, List[float]]]] = {}
        self._by_spot: Dict[tuple, Solution] = {}    # latest solve per spot, whatever the ranges
        # precomputed flop solves (gto/flop_library.py: FlopLibrary), used where they hold the spot; None = off
        self.library = None
        # precomputed 3-way flop solves (gto/mw_library.py: MWLibrary): a 3-way flop's first decisions, at every speed
        self.mw_library = None
        # the street advised on (heads-up from its start) also gets the earlier streets' prior_budget its narrowing
        # didn't spend (native Fast: README, Accuracy vs speed); off: time_budget alone
        self.spare_prior = False
        self.rng = random.Random()
        self._breakdown = False   # this request asked for the breakdown (state "breakdown": true)
        self._lib_only = False    # state "library_only": a flop the library holds, else {"skipped": …}, no solve
        self._study = False       # state "study": Study mode's answer (Advisor.study)
        self._runout_close = None # Study's runout report: the action that closed the flop or turn (Advisor._runouts)
        self._runout_held = False # ...from what's in hand only: the library's stored turns, cached solves (_NotHeld)
        self._runout_lines = None # ...the stored turn's flop lines, where the file is there but not for this line
        self._req_sizes: Dict[str, List[float]] = {}   # the request's custom sizes (state "sizes"), as fractions
        self._one_size = None     # state "one_size": the one-size view's request (Advisor._one_size_view), else None
        # ...its solves stop this close to equilibrium (% of the pot, by street; the lower expl_target where set)
        self.one_size_settled: Dict[str, float] = dict(ONE_SIZE_SETTLED)
        self._one_rows: Dict[tuple, dict] = {}   # its rows solved here, by spot and size (ONE_SIZE_CACHE, oldest first)
        # Advisor.instant's sources, tried in order: each (advisor, InstantSpot) -> {"actions", "source", "label"} or None
        self.instant_sources = [instant_from_store, instant_from_near, instant_from_own, instant_from_prev]
        self._instant = False     # an Advisor.instant request: nothing is solved (_NoInstant where it would be)

    # ------------------------------------------------------------ public
    _icm = None   # the tournament's prize model while a request is answered (_advise), else None

    def advise(self, state: dict) -> dict:
        if state.get("study"):
            return self.study(state)
        # the catch-all: a spot whose solve fails (an exception other than a cancel, or an answer with nothing usable
        # in it) is answered by the betting rules instead (Advisor._rules), not left with an error
        try:
            with np.errstate(all="ignore"):
                res = self._advise(state)
            why = _unusable(res)
        except Cancelled:
            raise
        except Exception as e:   # noqa: BLE001
            why = e
        if why and state.get("one_size") is not None:
            return _one_size_failed(why)   # (the view or nothing: not the betting rules' advice)
        if why:
            return self._fallback(state, why)
        self._flag_short(state, res)
        return res

    def _flag_short(self, state: dict, res: dict):
        """A heads-up answer after the flop over OVER_TARGET times expl_target from equilibrium (a solve's, not Quick's:
        its smaller game and 1.5 s are rough by design): "approximate" (solver.approximate, the target in
        solver.target_pct) and a note. (Not a multiway solve's: a 3-way flop at Accurate 0.7-1.9% and one over 10% left
        to the betting rules.)"""
        sv = res.get("solver") if isinstance(res, dict) else None
        x = sv.get("exploitability_pct") if isinstance(sv, dict) else None
        t = self.expl_target
        if t is None or x is None or state.get("quick") or res.get("multiway") or \
                res.get("street") not in ("flop", "turn", "river") or x <= OVER_TARGET * t:
            return
        sv["approximate"], sv["target_pct"] = True, t
        res.setdefault("notes", []).append(
            f"Approximate: this solve ran short, {x:.1f}% of the pot from equilibrium where answers at this speed are "
            f"{t}% or less (a busy or slow device, a big tree, or the compiled solver failed and the slower one "
            f"answered), so take its mix and EVs as a rough guide.")

    def _fallback(self, state: dict, why) -> dict:
        """The catch-all's answer for a spot advise or study couldn't answer (`why`: the exception, or what was wrong
        with the answer): the same request through Advisor._rules, marked "fallback" (what answered: "rules",
        "chart" or "equity") with "fallback_reason" (and "fallback_where", the line that raised). A state it can't
        read either gets an error, as do Study's spots after the flop (it shows a solve or nothing)."""
        reason = why if isinstance(why, str) else f"{type(why).__name__}: {why}"[:300]
        if state.get("library_only"):
            return {"skipped": f"no library answer ({reason})"}
        try:
            with np.errstate(all="ignore"):
                res = self._advise(state, rules=reason)
        except Cancelled:
            raise
        except Exception as e:   # noqa: BLE001
            res = {"error": f"couldn't answer this spot ({reason}; then {type(e).__name__}: {e})"[:500]}
        res["fallback_reason"] = reason
        if isinstance(why, BaseException) and why.__traceback__ is not None:
            import traceback
            f = traceback.extract_tb(why.__traceback__)[-1]
            res["fallback_where"] = f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}"
        return res

    def instant(self, state: dict) -> Optional[dict]:
        """An approximate answer at once, for the browser to show while the request itself solves (the worker posts it
        first, the full answer replaces it): a heads-up street whose own solve would take seconds, read off what is in
        hand already by instant_sources (a library flop's stored turn, until the street's own solve has run
        STORE_UNTIL iterations; the street's own solve stopped part way; the previous street's live solve, which plays
        this street under it), with "approximate": {source, label} and no EVs. Nothing is solved: None where the
        answer needs a solve before anything can be read (an earlier street not solved yet), is quick anyway (preflop,
        a library flop, a street solved already), isn't heads-up, or asks for more than the equilibrium (exploit mode,
        locks, custom ranges or sizes, a tournament, Study, a jam, the breakdown)."""
        if len(str(state.get("board") or "")) < 6 or state.get("opponents") == "pool" or any(
                state.get(k) for k in ("study", "runouts", "jam", "breakdown", "optimize_sizes", "locks", "library_only",
                                       "payouts", "bounties", "ranges", "sizes")) or state.get("one_size") is not None:
            return None
        self._instant = True
        try:
            with np.errstate(all="ignore"):
                res = self._advise(state)
        except _NoInstant:
            return None
        finally:
            self._instant = False
        return res if isinstance(res, dict) and res.get("approximate") else None

    def presolve(self, state: dict, seat: str) -> Optional[dict]:
        """The solve your next decision on this street will read, started while `seat` (the opponent) is to act: the
        browser's worker runs it while you enter their move, the page stops it when you do (or leave the tab), and
        the advice goes on from where it stopped (a solve stopped part way stays cached), so the time you take counts
        toward the answer's. The street's solve with your cards as your advice makes it (Advisor.play's actor path),
        within the request's budget, its answer for them never read; your advice that follows takes what's left of
        that budget. Only where an instant answer is offered (heads-up from the flop, at equilibrium: not exploit
        mode, locks, custom ranges or sizes, a tournament, Study, a jam, the breakdown) and the street is solved live
        (not a flop the library holds). {"presolved": street, "iterations"}, else None."""
        if len(str(state.get("board") or "")) < 6 or state.get("opponents") == "pool" or any(
                state.get(k) for k in ("study", "runouts", "jam", "breakdown", "optimize_sizes", "locks", "library_only",
                                       "payouts", "bounties", "ranges", "sizes")) or state.get("one_size") is not None:
            return None
        seat = str(seat or "").upper()
        if not seat or seat == str(state.get("hero") or "").upper():
            return None
        self._presolved &= set(self._cache)
        self._presolving = True
        try:
            with np.errstate(all="ignore"):
                res = self._advise(state, actor=(seat, None))
        except Cancelled:
            raise
        except Exception:   # noqa: BLE001 - (nothing to show: the advice itself says what's wrong)
            return None
        finally:
            self._presolving = False
        return res if isinstance(res, dict) and res.get("presolved") else None

    def prefetch(self, state: dict) -> None:
        """The flop's library lookups for a hand as soon as its flop is known, without reading any entry: a dry run
        for the browser's worker, whose fetch then records the files asked for and requests them ahead
        (gto/static/worker.js), so the answer at the flop finds them in hand. The flop before anyone acts, asked as
        the player first to act ("library_only": nothing is solved), and the turn files of the flop's entries the index
        lists. The files it asked for and didn't get stay unread (they would otherwise be kept as missing)."""
        board = str(state.get("board") or "")[:6]
        libs = [x for x in (self.library, self.mw_library) if x is not None and hasattr(x, "_entries")]
        if len(board) < 6 or not libs:
            return
        st = {k: v for k, v in state.items() if k not in ("study", "runouts", "jam", "breakdown", "optimize_sizes",
                                                           "one_size")}
        st.update(board=board, flop=[], turn=[], river=[], library_only=True)
        before = [set(x._entries) for x in libs]
        try:
            res = self.advise(st)
            m = re.match(r"it is (\w+)'s turn", str(res.get("error", "")))
            if m:
                self.advise(dict(st, hero=m.group(1)))
            # this flop's entries' turn files too, where the index lists them (the turn's approximate answer at once:
            # instant_from_store)
            lib = self.library
            if lib is not None and lib in libs and hasattr(lib, "turns"):
                from .flop_library import flop_name
                name = flop_name(parse_cards(board))
                spots = {s.get("id"): s for s in lib.index().get("spots", ())}
                for k in [k for k in lib._entries if len(k) == 2 and k[1] == name and k[0] in spots]:
                    lib.turns(spots[k[0]], name)
        except Exception:
            pass
        finally:
            for x, b in zip(libs, before):
                for k in [k for k, e in x._entries.items() if e is None and k not in b]:
                    del x._entries[k]

    def study(self, state: dict) -> dict:
        """Study mode (the page's solution browser): the hand as recorded, answered for whoever acts next with every
        hand of their range instead of one: per hand class and combo the mix, each action's EV and the equity
        against the other range (heads-up and 3-way after the flop; preflop the charts' mix), the range's totals per
        action.
        Where nobody acts: {"study": "deal", "street"} (the next card is due) or {"study": "over"}. At equilibrium
        (no pool, locks, tournament payouts or jam requests) and without hero's cards (no card removal for them)."""
        st = {k: v for k, v in state.items() if k not in ("hero", "hole", "opponents", "locks", "jam", "library_only",
                                                           "payouts", "bounties", "optimize_sizes")}
        st["study"] = True
        runouts = st.pop("runouts", None)
        try:
            if runouts:
                return self._runouts(st, held=runouts == "held")
            with np.errstate(all="ignore"):
                res = self._advise(st)
            why = _unusable(res)
        except Cancelled:
            raise
        except Exception as e:   # noqa: BLE001
            if runouts:
                return {"error": f"the runout report failed ({type(e).__name__})",
                        "fallback_reason": f"{type(e).__name__}: {e}"[:300]}
            why = e
        if why and st.get("one_size") is not None:
            return _one_size_failed(why)
        return self._fallback(st, why) if why else res   # (the catch-all, as advise's)

    def _runouts(self, st: dict, held: bool = False) -> dict:
        """Study mode's runout report (state "runouts": true, once a heads-up flop's or turn's betting closed): the
        next street's first decision on every card, the first player's mix and the second's after a check, read off
        a live solve of the street that closed (the library's entries hold the flop only), whose tree plays the next
        street under each card or card class: the decision that closed the street is walked to as Study would answer
        it, and the street below its closing action read (Advisor._runout_report). The flop solve's view of the
        turn: one bet size there (Config.later_sizes) and each turn class's cards alike; the turn solve's view of
        the river: its sizes there. A card dealt and solved on its own can mix in other sizes.
        A library flop whose entry has a turn file (flop_library's stored turns) answers the turn report from it
        (Advisor._store_report), on the flop lines it holds. held (state "runouts": "held"): from what's in hand only,
        the stored turns or a solve cached already (Study's own answer at that node), else {"skipped", "lines": the
        flop lines a stored turn holds, where there is one} and nothing is solved."""
        street = {3: "flop", 4: "turn"}.get(len(parse_cards(str(st.get("board") or ""))))
        line = list(st.get(street) or []) if street else []
        if not line or st.get("river"):
            return {"error": "the runout report needs a flop or turn whose betting is done (and the next card not dealt)"}
        self._runout_close = str(line[-1].get("action"))
        self._runout_held, self._runout_lines = held, None
        try:
            with np.errstate(all="ignore"):
                return self._advise(dict(st, **{street: line[:-1]}))
        except _NotHeld:
            out = {"study": "runouts", "street": "turn" if street == "flop" else "river",
                   "skipped": "not solved yet"}
            if self._runout_lines:
                out["lines"] = self._runout_lines
            return out
        finally:
            self._runout_close, self._runout_held, self._runout_lines = None, False, None

    def _runout_report(self, sol, node, n_r0, n_r1, hero, villain, hero_p, board, expl, notes, t0):
        """The runout report (Advisor._runouts) from the flop's or turn's solve at the decision that closed the
        street: per card of the next street (per class where the tree plays one for several: Ctx.turn_classes, a
        rank's cards, a flush suit's apart) the first decision's mix over its player's range there (f), and the second
        player's after a check; avg: over every card. A turn played by hand buckets (Fast's flop model, BucketEntry)
        is read by the buckets' masses. cards: per card off the board its chance, each player's range equity and EV
        there (Advisor._runout_cards)."""
        a = match_action(node.names, self._runout_close)
        ch = node.children[a]
        nxt = "turn" if len(board) == 3 else "river"
        if not isinstance(ch, Chance) or not isinstance(ch.child, (Decision, BucketEntry)):
            return {"error": f"no {nxt} to report on: the betting isn't closed, or nobody can bet any more"}
        seat = lambda p: hero if p == hero_p else villain
        r = [np.asarray(n_r0, dtype=np.float64).reshape(-1), np.asarray(n_r1, dtype=np.float64).reshape(-1)]
        r[node.player] = r[node.player] * node.avg_strategy()[0, a]
        K = ch.K
        if ch.perm is not None:              # river classes: a class's reach through its cards' swaps (Chance.spread)
            m = [ch.spread(x[None, :], np.zeros(1, dtype=np.intp)) for x in r]
        else:
            masks = ch.masks.reshape(K, -1)
            m = [r[0] * masks, r[1] * masks]     # (K, n): each player's reach on each card or class
        evs = self._row_evs(sol, ch.child, m)   # (2, K)
        dec, rows = ch.child, ch.child.boards
        if isinstance(dec, BucketEntry):
            T, ids = sol.solver._btab(dec)[:2]
            dec = dec.child
            W = dec.avg_strategy().shape[2]

            def to_buckets(p, x):             # a combo's reach into its bucket on that card (W: dead there)
                out = np.zeros((K, W + 1))
                for k in range(K):
                    np.add.at(out[k], ids[k % T, p], x[k])
                return out[:, :W]
            m = [to_buckets(0, m[0]), to_buckets(1, m[1])]
        i = len(board)                       # (the dealt card's place in a row's board)
        if ch.umasks is not None or ch.perm is not None:     # a child per class of cards
            classes = dict(sol.ctx.turn_classes(tuple(board), set(board)))
            members = [classes.get(b[i], (b[i],)) for b in rows]
        else:
            members = [(b[i],) for b in rows]
        n_cards = np.array([len(c) for c in members], dtype=np.float64)

        def part(d, mm):
            s = d.avg_strategy()
            w = mm[d.player]
            tot = w.sum(axis=1)
            f = (s * w[:, None, :]).sum(axis=2) / np.maximum(tot, 1e-12)[:, None]
            # over every turn card: each class by its cards and its player's reach there
            wk = n_cards * tot
            avg = (f * wk[:, None]).sum(axis=0) / max(float(wk.sum()), 1e-12)
            return s, {"actor": seat(d.player), "actions": list(d.names), "avg": [round(float(x), 4) for x in avg],
                       "cards": [{"card": card_str(rows[k][i]), "cards": [card_str(c) for c in members[k]],
                                  "f": [round(float(x), 4) for x in f[k]], "w": round(float(tot[k]), 3)}
                                 for k in range(K)]}
        s1, first = part(dec, m)
        after = None
        if "check" in dec.names:
            c = dec.names.index("check")
            then = dec.children[c]
            if isinstance(then, Decision):
                m2 = list(m)
                m2[dec.player] = m[dec.player] * s1[:, c, :]
                after = part(then, m2)[1]
        notes = list(notes) + [
            "the flop solve's view of the turn: it plays one bet size there and each turn class's cards alike (a "
            "rank's cards, a flush suit's apart); a turn dealt is solved on its own, with more sizes" if nxt == "turn"
            else "the turn solve's view of the river: its river sizes (" + ", ".join(dec.names[1:]) + "); a river "
            "dealt is solved on its own, with its own menu of sizes"]
        ev_of = {c: evs[:, k] for k in range(K) for c in members[k]}
        seats = [seat(0), seat(1)]
        return {"study": "runouts", "street": nxt, "first": first, "after_check": after, "seats": seats,
                "pot_bb": round(float(dec.pot_now), 2),
                "cards": self._runout_cards(board, expand(sol.ctx, r[0]), expand(sol.ctx, r[1]), ev_of),
                "source": "solve",
                "solver": {"iterations": int(sol.iters), "exploitability_pct": round(float(expl), 3),
                           "by": "hand buckets" if isinstance(ch.child, BucketEntry) else "every hand"},
                "notes": notes, "seconds": round(time.time() - t0, 2)}

    @staticmethod
    def _row_evs(sol, child, m) -> np.ndarray:
        """(2, K): each player's range EV on each row of a chance node's child (m: the two players' reaches there,
        (K, n) each), in chips relative to folding now, both players on their average strategies, every later card
        dealt (no sampling); range against range, each pair of hands that share no card by its two reaches (as the
        range equities: the two EVs add up to the pot, the equities to one); NaN where a player has no hand there."""
        s = sol.solver
        R = [np.ascontiguousarray(x, dtype=np.float32) for x in m]
        out = np.full((2, R[0].shape[0]), np.nan)
        was = getattr(s, "sampling", True)
        s.sampling = False
        try:
            for p in (0, 1):
                u = np.asarray(s._cfr(child, p, R[0], R[1], update=False, use_avg=True), dtype=np.float64)
                cs = np.asarray(s.compat_sum(R[1 - p]), dtype=np.float64)
                num, den = (R[p] * u).sum(axis=1), (R[p] * cs).sum(axis=1)
                ok = den > 1e-9
                out[p, ok] = num[ok] / den[ok] + _inv_of(child, p, sol.ctx.cfg.icm)
        finally:
            s.sampling = was
        return out

    @staticmethod
    def _runout_cards(board, w0, w1, ev_of=None) -> List[dict]:
        """The report's cards: per card off the board {"card", "p": its chance given the two ranges, "eq": each
        player's range equity on it (the rest of the board dealt), "ev": each player's EV there (ev_of, by card;
        None where the report has none)}. w0, w1: the two ranges entering the next street (1326 each)."""
        from .runouts import next_card_equities
        cards, eq, p = next_card_equities(list(board), np.asarray(w0, dtype=np.float64), np.asarray(w1, dtype=np.float64))
        out = []
        for c, e, q in zip(cards, eq, p):
            ev = None if ev_of is None or c not in ev_of else ev_of[c]
            out.append({"card": card_str(c), "p": round(float(q), 5),
                        "eq": None if not np.isfinite(e) else [round(float(e), 4), round(1.0 - float(e), 4)],
                        "ev": None if ev is None or not np.all(np.isfinite(ev)) else [round(float(x), 2) for x in ev]})
        return out

    def _store_report(self, sol, node, n_r0, n_r1, taken, board, seats, notes, t0) -> Optional[dict]:
        """The turn report (Advisor._runouts) from a library flop's stored turns (its entry's turn file,
        runouts.store_parts): per turn card the first decision's range mix and in position's after a check, as the
        library's flop solve plays them (its turn sizes; a turn class's cards alike), with each card's chance and
        range equities; no EVs (the file holds strategies only). None where the entry has no turn file, or it doesn't
        hold the hand's flop line (the lines it holds go to self._runout_lines)."""
        got = _turn_file(self.library, sol, board[:3])
        if got is None or not isinstance(node, Decision) or id(node) not in getattr(sol, "stored", ()):
            return None
        t, smap = got
        a = match_action(node.names, self._runout_close)
        path = list(taken) + [node.names[a]]
        lines = self._store_lines(sol, t, seats)
        at = next((k for k, ln in enumerate(lines) if ln["path"] == path), None)
        if at is None:
            self._runout_lines = lines
            return None
        from .runouts import store_parts
        r = [expand(sol.ctx, np.asarray(n_r0, dtype=np.float64).reshape(-1)),
             expand(sol.ctx, np.asarray(n_r1, dtype=np.float64).reshape(-1))]
        r[node.player] = r[node.player] * expand(sol.ctx, np.asarray(node.avg_strategy()[0, a], dtype=np.float64))
        first, after = store_parts(t, smap, path, r, board, seats)
        if first is None:
            self._runout_lines = lines
            return None
        ln = next(x for x in t["lines"] if x["path"] == path)
        notes = list(notes) + [
            "the flop library's stored turns: the precomputed flop solve's turn strategies (its turn sizes, each turn "
            "class's cards alike: a rank's cards, a flush suit's apart; mixes in 5% steps), strategies only, so no EVs; "
            "a turn dealt is solved on its own, with more sizes"]
        return {"study": "runouts", "street": "turn", "first": first, "after_check": after, "seats": list(seats),
                "pot_bb": round(float(ln["pot"]), 2), "cards": self._runout_cards(board, r[0], r[1]),
                "source": "store", "lines": lines, "line": at,
                "solver": {"iterations": int(sol.iters), "exploitability_pct": round(float(sol.exploitability()), 3),
                           "by": "the flop library"},
                "notes": notes, "seconds": round(time.time() - t0, 2)}

    def _store_lines(self, sol, t, seats) -> List[dict]:
        """A turn file's flop lines as Study's flop actions ({"flop": [{"pos", "action", "to"}], "share": how often
        the flop solve takes it, "path": its tree names}), walked down the library entry's tree `sol` (the amounts
        its sizes come to); a line it can't walk is left out."""
        out = []
        for ln in t.get("lines", ()):
            node, acts = sol.root, []
            for name in ln["path"]:
                if not isinstance(node, Decision) or name not in node.names:
                    acts = None
                    break
                p = node.player
                kind = "all-in" if name == "all-in" else name.split()[0]
                act = {"pos": seats[p], "action": kind}
                if kind in ("bet", "raise"):
                    act["to"] = self._amount(name, node, p)
                acts.append(act)
                node = node.children[node.names.index(name)]
            if acts:
                out.append({"flop": acts, "share": round(float(ln.get("share", 0.0)), 4), "path": list(ln["path"])})
        return out

    def _advise(self, state: dict, actor=None, rules: Optional[str] = None) -> dict:
        t0 = time.time()
        bb = float(state.get("bb", 1.0))
        n = int(state.get("players", 6))
        if n not in pf.SEATS:
            return {"error": f"players must be 2..9, got {n}"}
        ante = float(state.get("ante") or 0.0) / bb
        straddle = bool(state.get("straddle")) and n >= 3
        # rake: a share of the pot (0.05 = 5%) up to a cap (in chips, like the other amounts), taken from
        # pots that see a flop
        rate = max(0.0, min(0.25, float(state.get("rake") or 0.0)))
        cap = state.get("rake_cap")
        cap = float(cap) / bb if cap not in (None, "") and float(cap) > 0 else None
        positions = pf.Table(pf.SEATS[n], ante, straddle, (rate, cap) if rate > 0 else None)
        study = bool(state.get("study"))   # (Advisor.study: hero is whoever acts next)
        hero = str(state.get("hero") or "").upper()
        if hero not in positions and not study:
            return {"error": f"hero seat {hero} is not one of {positions}"}
        try:
            sizes = _custom_sizes(state["sizes"]) if state.get("sizes") else {}
            custom = _custom_ranges(state["ranges"], positions) if state.get("ranges") else {}
            if state.get("locks"):
                from .exploit import parse_locks
                locks = parse_locks(state["locks"], positions, hero)
            else:
                locks = {}
        except ValueError as e:
            return {"error": str(e)}
        hole = () if study else tuple(parse_cards(str(state["hole"])))
        board = parse_cards(str(state.get("board") or ""))
        if (len(hole) != 2 and not study) or len(board) not in (0, 3, 4, 5):
            return {"error": "need two hole cards and 0, 3, 4 or 5 board cards"}
        if len(set(hole) | set(board)) != len(hole) + len(board):
            return {"error": "the same card appears twice in your hand and the board"}
        try:
            stack, stacks = _state_stacks(state.get("stack"), state.get("stacks"), positions, bb)
        except ValueError as e:
            return {"error": str(e)}
        streets = {s: _norm_actions(state.get(s), bb) for s in STREETS}
        try:
            model = _tournament(state, positions, stack, stacks, bb)
        except ValueError as e:
            return {"error": str(e)}
        self._icm = model
        if model is not None and positions.rake:
            positions = pf.Table(list(positions), positions.ante, positions.straddle, None)
        if study:
            tab = pf.Table(list(positions), positions.ante, positions.straddle, positions.rake, stack, stacks) \
                if stacks else positions
            nxt = _study_next(tab, stack, streets, len(board))
            if isinstance(nxt, dict):
                return nxt
            hero = nxt
        if stacks or model is not None:
            # each seat's own stack; the depth hero plays at is the effective stack against the deepest
            # opponent still in (preflop: who hasn't folded)
            positions = pf.Table(list(positions), positions.ante, positions.straddle, positions.rake, stack, stacks, model)
            stack = _eff_stack(positions, hero, streets["preflop"], stack)
        if rules is not None:   # (the catch-all: Advisor._fallback)
            return self._rules(state, t0, hero, hole, board, positions, stack, streets, custom, model, rules)
        positions = pf.at_depth(positions, stack)
        res = self._advise_at(state, actor, t0, hero, hole, board, positions, stack, streets, sizes, custom, locks)
        if model is not None and "actions" in res:
            out = {a["pos"] for s in STREETS for a in streets[s] if a["action"] == "fold"}
            res["tournament"] = _tournament_info(model, hero, positions, res.get("street"), out)
            res.setdefault("notes", []).insert(0, res["tournament"]["note"])   # (a practice move's answer has none)
            if "bounties" in res["tournament"]:
                res["notes"].insert(1, res["tournament"]["bounties"]["note"])
        return res

    def _rules(self, state, t0, hero, hole, board, positions, stack, streets, custom, model, reason) -> dict:
        """The catch-all's answer (Advisor._fallback), for a spot whose solve failed: preflop the chart's mix for the
        hand ("chart"), after the flop the betting rules' Monte Carlo against everyone's ranges (Advisor._multiway, as
        for the pots no solver takes; heads-up too: "rules"), and where either of those fails too, the hand's equity
        against random hands priced against the pot ("equity"). Each step uses what still works: the charts' ranges
        narrowed by the rules, else any two cards. In chips (a tournament's prizes aside), at equilibrium (no pool or
        locks), the default sizes."""
        try:
            positions = pf.at_depth(positions, stack)
        except Exception:   # noqa: BLE001
            pass
        study = bool(state.get("study"))
        n_streets = {0: 1, 3: 2, 4: 3, 5: 4}[len(board)]
        rep = replay_hand(positions, stack, streets, n_streets)
        cur = rep[-1]
        if len(rep) < n_streets or hero not in cur.active or cur.closed or cur.to_act != hero:
            return {"error": f"no decision of {hero}'s here"}
        stk = _table_stacks(positions, stack)
        if study and board:
            return {"error": f"Study couldn't solve this spot ({reason}); it shows a solve or nothing after the flop"}
        res, used, failed = None, None, []
        if not board:
            try:
                res, used = self._chart_rules(hero, hole or (0, 1), streets["preflop"], positions, stack, cur, t0), "chart"
            except Cancelled:
                raise
            except Exception as e:   # noqa: BLE001
                failed.append(f"the chart, {type(e).__name__}")
        else:
            notes: List[str] = []
            try:
                try:
                    ranges = dict(preflop_ranges(streets["preflop"], positions, stack))
                except Exception as e:   # noqa: BLE001
                    ranges = {}
                    failed.append(f"the preflop ranges, {type(e).__name__}")
                for p, r in (custom or {}).items():
                    if p in rep[0].active:
                        ranges[p] = r
                for st in rep[1:-1] + [cur]:
                    try:
                        _narrow_street(ranges, board[:{"flop": 3, "turn": 4, "river": 5}[st.name]], st, stk, hero, hole, {})
                    except Exception as e:   # noqa: BLE001
                        failed.append(f"narrowing the {st.name}, {type(e).__name__}")
                        break
                remaining = [p for p in cur.order if p in cur.active]
                res, used = self._multiway(hero, hole, board, ranges, remaining, cur, stk, t0, notes), "rules"
                # (its "not a solve here" note is for the pots no solver takes: the first note says why here)
                res["notes"] = [n for n in res.get("notes", []) if "not a solve here" not in n]
            except Cancelled:
                raise
            except Exception as e:   # noqa: BLE001
                failed.append(f"the betting rules, {type(e).__name__}")
        if res is None or _unusable(res):
            res, used = self._equity_rules(hero, hole or (0, 1), board, positions, cur, stk, t0), "equity"
        if study:   # (preflop: the charts' mix for the whole range)
            if used != "chart":
                return {"error": f"Study couldn't answer this spot ({reason})"}
            res = self._study_preflop(hero, streets["preflop"], positions, stack, t0, res)
        res["fallback"] = used
        what = {"chart": "the preflop chart's mix for your hand",
                "rules": "the betting rules (a Monte Carlo against everyone's ranges, as for the pots no solver takes)",
                "equity": "your equity against random hands, priced against the pot (a rough last resort)"}[used]
        res.setdefault("notes", []).insert(0, f"the solver couldn't answer this spot ({reason}), so this answer is "
                                              f"{what}" + (f" (also failed: {', '.join(failed)})" if failed else ""))
        if model is not None and "actions" in res:
            try:
                out = {a["pos"] for s in STREETS for a in streets[s] if a["action"] == "fold"}
                res["tournament"] = _tournament_info(model, hero, positions, res.get("street"), out)
                res["notes"].insert(1, res["tournament"]["note"] + " (the fallback plays chips)")
            except Exception:   # noqa: BLE001
                pass
        return res

    def _chart_rules(self, hero, hole, preflop, positions, stack, st: "StreetState", t0) -> dict:
        """The catch-all's preflop answer (Advisor._rules): the chart's mix for the hand as the chart has it (no
        re-pricing for the stacks, rake, jams or locks), its raises at the default size held to a legal one."""
        full = {k: np.asarray(v, dtype=np.float64) for k, v in pf.strategy_for(hero, preflop, positions).items()}
        remaining = stack - st.inv_start[hero] - st.bets[hero]
        to_call = max(0.0, min(st.cur_bet - st.bets[hero], remaining))
        passive = "call" if to_call > 1e-9 else "check"
        for k in [k for k in full if k in ("call", "check") and k != passive] + \
                 (["raise"] if "raise" in full and (not can_raise(st, hero) or remaining <= to_call + 1e-9) else []) + \
                 (["fold"] if "fold" in full and to_call <= 1e-9 else []):
            full[passive] = np.minimum(full.get(passive, 0.0) + full.pop(k), 1.0)   # (what can't be done here: the passive move)
        idx = COMBO_INDEX[(min(hole), max(hole))]
        size = _legal_to(st, hero, _default_size(st, hero, True), remaining)
        acts = []
        for k, v in full.items():
            a = {"action": k, "freq": round(float(v[idx]), 3)}
            if k == "raise":
                a["action"] = "all-in" if size >= st.bets[hero] + remaining - 1e-9 else "raise"
                a["amount_bb"] = round(float(size), 2)
            elif k == "call":
                a["amount_bb"] = round(float(to_call), 2)
            acts.append(a)
        if sum(a["freq"] for a in acts) <= 0:
            acts[0]["freq"] = 1.0
        names = list(full)
        try:
            arriving = preflop_ranges(preflop, positions, stack).get(hero, np.ones(N_COMBOS))
        except Exception:   # noqa: BLE001
            arriving = np.ones(N_COMBOS)
        return {"street": "preflop", "hero": hero, "hand": hand_class(hole), "players": len(positions),
                "actions": acts, "recommended": max(acts, key=lambda a: a["freq"])["action"], "sampled": self._sample(acts),
                "pot_bb": round(st.pot_start + sum(st.bets.values()), 2), "to_call_bb": round(to_call, 2),
                "grid": strategy_grid([full[k] for k in names], names, arriving), "notes": [],
                "seconds": round(time.time() - t0, 3)}

    def _equity_rules(self, hero, hole, board, positions, st: "StreetState", stk, t0) -> dict:
        """The catch-all's last resort (Advisor._rules): the hand's equity against a random hand for each opponent
        still in (Monte Carlo), priced against the pot. Facing a bet: call when the equity pays for it, else fold;
        otherwise bet 66% of the pot with more than half the pot's equity, else check. EVs as if every bet is
        called; frequencies 0/1, as the betting rules'."""
        from .evaluator import eval_boards
        rng = np.random.default_rng(1)
        opps = [p for p in st.order if p in st.active and p != hero]
        dead = set(board) | set(hole)
        deck = np.array([c for c in range(52) if c not in dead])
        need = 5 - len(board)
        n = 300
        wins = 0.0
        for _ in range(n):
            draw = rng.permutation(deck)[:need + 2 * len(opps)]
            full = list(board) + [int(c) for c in draw[:need]]
            s = eval_boards([full], [COMBO_INDEX[(min(hole), max(hole))]] +
                            [COMBO_INDEX[(int(min(draw[need + 2 * i], draw[need + 2 * i + 1])),
                                          int(max(draw[need + 2 * i], draw[need + 2 * i + 1])))] for i in range(len(opps))])[0]
            best = s.max()
            if s[0] >= best:
                wins += 1.0 / float((s == best).sum())
        eq = wins / n
        remaining = max(0.0, stk[hero] - st.inv_start[hero] - st.bets[hero])
        pot = st.pot_start + sum(st.bets.values())
        to_call = max(0.0, min(st.cur_bet - st.bets[hero], remaining))
        acts = []
        if to_call > 1e-9:
            acts.append({"action": "fold", "freq": 0.0, "ev_bb": 0.0})
            acts.append({"action": "call", "freq": 0.0, "ev_bb": round(eq * (pot + to_call) - to_call, 2),
                         "amount_bb": round(to_call, 2)})
        else:
            acts.append({"action": "check", "freq": 0.0, "ev_bb": round(eq * pot, 2)})
            if can_raise(st, hero) and remaining > 1e-9:
                to = _legal_to(st, hero, _default_size(st, hero, not board), remaining)
                add = to - st.bets[hero]
                acts.append({"action": "all-in" if to >= st.bets[hero] + remaining - 1e-9 else
                             f"bet {add / max(pot, 1e-9) * 100:.0f}%" if board else "raise",
                             "freq": 0.0, "ev_bb": round(eq * (pot + 2 * add) - add, 2),
                             "amount_bb": round(float(to), 2)})
        best = max(acts, key=lambda a: a["ev_bb"])
        best["freq"] = 1.0
        return {"street": st.name, "hero": hero, "hand": hand_class(hole), "board": cards_str(board),
                "players": len(opps) + 1, "multiway": True, "opponents": opps, "equity_vs_field": round(eq, 3),
                "pot_bb": round(pot, 2), "to_call_bb": round(to_call, 2), "actions": acts,
                "recommended": best["action"], "sampled": best["action"],
                "notes": [f"equity {eq * 100:.0f}% against {len(opps)} random hand{'s' if len(opps) > 1 else ''}"],
                "seconds": round(time.time() - t0, 2)}

    def _advise_at(self, state, actor, t0, hero, hole, board, positions, stack, streets, sizes, custom, locks):
        if not board and state.get("one_size") is not None and actor is None:
            return {"error": "the one-size view is for bets after the flop: preflop is the charts' raise sizes"}
        if not board and state.get("study"):
            return self._study_preflop(hero, streets["preflop"], positions, stack, t0)
        if not board:
            return self._preflop(hero, hole, streets["preflop"], positions, stack, t0,
                                 pool=state.get("opponents") == "pool", locks=locks)
        saved = self.time_budget, self.prior_budget, self.cfg, self.root_sizes, self.spr_sizes
        if sizes:
            self.root_sizes = {st: {k: v for k, v in d.items() if k != "raise" or "raise" not in sizes}
                               for st, d in self.root_sizes.items() if st not in sizes}
            self.spr_sizes = {st: (m, {k: v for k, v in d.items() if k not in sizes})
                              for st, (m, d) in self.spr_sizes.items() if st not in sizes}
        if positions.rake or sizes:
            self.cfg = self.cfg.copy()
        if positions.rake:
            self.cfg.rake_rate, self.cfg.rake_cap = positions.rake
        for k, v in sizes.items():   # (_cfg_with_real_sizes maps the sizes actually taken onto these)
            if k == "raise":
                self.cfg.raise_sizes = {st: list(v) for st in ("flop", "turn", "river")}
            else:
                self.cfg.bet_sizes[k] = list(v)
        if state.get("budget"):
            self.time_budget = max(0.5, min(120.0, float(state["budget"])))
        if state.get("prior_budget"):
            self.prior_budget = max(0.3, min(30.0, float(state["prior_budget"])))
        # Practice's quick answers (~2 s): the page's budget as sent (a 3-way river MW_QUICK_BUDGET, not mw_budget; no second
        # budget for an unsettled street, SETTLE_EXPL), 3-way solves on the river only (a flop's or turn's tables alone
        # take ~1 s in the browser: the betting rules answer)
        self._quick = bool(state.get("quick"))
        # "jam": the street advised on offers all-in at every decision however deep (Config.root_allin), so a jam the
        # usual tree leaves out (15x the pot on a flop) is solved and can be graded; the page asks it again with this
        # once you jam there. Not the earlier streets' narrowing solves, nor the flop library (no all-in stored)
        self._jam = bool(state.get("jam"))
        if self._jam:
            self.cfg = self.cfg.copy()
            self.cfg.root_allin = True
        # the Breakdown view (gto/breakdown.py) on the advice's own spot; not for a practice opponent's move
        self._breakdown = bool(state.get("breakdown")) and actor is None
        self._req_sizes = sizes   # (optimize_sizes puts a street's custom sizes in its wide menu)
        # the one-size view (Advisor._one_size_view) in place of the advice: "all", "full", "list" or a size ("50%",
        # 50, "all-in"); not for a practice opponent's move or a presolve
        self._one_size = state.get("one_size") if actor is None else None
        # the page's session analyzer in its "flop-library spots only" scope: a flop the library holds, never a solve
        self._lib_only = bool(state.get("library_only")) and actor is None
        if self._lib_only and len(board) != 3:
            return {"skipped": "only flop spots are answered from the library"}
        self._study = bool(state.get("study")) and actor is None
        try:
            res = self._postflop(hero, hole, board, stack, streets, positions, t0,
                                 pool=state.get("opponents") == "pool", actor=actor, custom=custom,
                                 optimize=bool(state.get("optimize_sizes")), locks=locks)
            if self._breakdown and "error" not in res and "breakdown" not in res:
                res["breakdown"] = None   # the betting rules answered (4+ players): no solve to break down
            return res
        finally:
            self.time_budget, self.prior_budget, self.cfg, self.root_sizes, self.spr_sizes = saved
            self._breakdown = self._lib_only = self._jam = self._study = False
            self._one_size = None

    # ------------------------------------------------------------ preflop
    def _preflop(self, hero, hole, preflop, positions, stack, t0, pool=False, locks=None):
        # your locks on the players still in (exploit.lock_preflop; the keys that act preflop): the advice is
        # re-drawn against them (_locked, _vs_jam), with the chart's mix alongside as GTO %
        live, later = (), []
        if locks:
            st = replay_hand(positions, stack, {"preflop": preflop}, 1)[0]
            live = tuple((p, kv) for p, kv in _lock_spec(locks) if p in st.active and p != hero)
            later = [p for p in locks if p in st.active and p != hero and p not in dict(live)]
        res = self._preflop_play(hero, hole, preflop, positions, stack, t0, pool, live)
        if (pool or live) and "actions" in res:
            # Against the pool a hand takes its best price, so a chart mix (44 flatting a CO open 30%) plays
            # 100% one way: show the chart's share beside it, as postflop shows the equilibrium's.
            g = {a["action"]: a["freq"] for a in self._preflop_play(hero, hole, preflop, positions, stack, t0)["actions"]}
            for a in res["actions"]:
                a["gto_freq"] = g.get(a["action"], 0.0)
            if pool and any(abs(a["freq"] - a["gto_freq"]) > 0.05 for a in res["actions"]):
                res["notes"].append("exploit mode: against the pool each hand takes its best-paying action, so the "
                                    "chart's mix (GTO %) plays one way")
        if "actions" in res:
            if live:
                res["locked"] = {p: dict(locks[p]) for p, _ in live}
            for p in later:
                res["notes"].append(f"locked: {p}'s lock ({', '.join(locks[p])}) acts after the flop only")
        return res

    def _preflop_play(self, hero, hole, preflop, positions, stack, t0, pool=False, locks=()):
        st = replay_hand(positions, stack, {"preflop": preflop}, 1)[0]
        if st.closed:
            return {"error": "preflop betting is already closed (deal the flop)"}
        if st.to_act != hero:
            return {"error": f"it is {st.to_act}'s turn, not {hero}'s"}
        aggr = [a["pos"] for a in preflop if a["action"] in ("raise", "all-in", "bet")]
        # (facing a jam: the last all-in is the bet to call. One for less than the bet already in, a 1.875bb stack's
        # under a straddle, is a short call, not a jam)
        if aggr and aggr[-1] in st.allin and aggr[-1] != hero and st.bets[aggr[-1]] >= st.cur_bet - 1e-9 \
                and st.cur_bet > st.bets[hero] + 1e-9:
            return self._vs_jam(hero, hole, preflop, positions, stack, st, t0, pool, locks)
        full = pf.strategy_for(hero, preflop, positions)
        notes = ["preflop: chart approximation"]
        cover = pf.format_chart(hero, preflop, positions)[1] if isinstance(positions, pf.Table) else {}
        if cover:
            from .pf_formats import label
            notes = [f"preflop: the {label(positions, cover)} chart (the 100bb chart moved by solves of this game)"]
        if "call" in full:
            full = self._priced(hero, preflop, positions, stack, st, full, notes, pool)
        elif getattr(positions, "icm", None) is not None:
            full = self._icm_open(hero, preflop, positions, stack, st, full, notes)
        elif pool and not locks:
            notes.append("exploit mode: opens follow the chart (the pool 3bets more and flats more, and "
                         "folds less to 3bets, so there's no clear edge in opening wider or tighter)")
        if locks:
            full = self._locked(hero, preflop, positions, stack, st, full, notes, pool, locks)
        if "raise" in full and not can_raise(st, hero):
            # an all-in for less than a full raise doesn't reopen the betting (or nobody behind has chips
            # left to face a raise): the chart's raises call
            full = dict(full)
            r = full.pop("raise")
            k = "call" if "call" in full or st.cur_bet > st.bets[hero] + 1e-9 else "check"
            full[k] = np.minimum(full.get(k, 0.0) + r, 1.0)
            notes.append("you can't raise here (an all-in for less than a full raise doesn't reopen the betting, "
                         "or nobody left has chips to call one): the chart's raises call")
        # the chart's passive move as the spot has it: with nothing to call (a big blind's or straddle's option, an
        # all-in for less than the blind before it) its calls and folds check; facing a bet its checks call
        facing = st.cur_bet > st.bets[hero] + 1e-9
        moved = [k for k in (("check",) if facing else ("call", "fold")) if k in full]
        if moved:
            full = dict(full)
            k = "call" if facing else "check"
            full[k] = np.minimum(full.get(k, 0.0) + sum(full.pop(m) for m in moved), 1.0)
        idx = COMBO_INDEX[(min(hole), max(hole))]
        strat = {k: float(v[idx]) for k, v in full.items()}
        remaining = stack - st.inv_start[hero] - st.bets[hero]
        to_call = min(st.cur_bet - st.bets[hero], remaining)
        size = _legal_to(st, hero, _default_size(st, hero, True), remaining)
        if "raise" in full and remaining > to_call + 1e-9:
            # all-in instead: the raise is most of the stack, or leaves under JAM_SPR pots behind once called
            pot = st.pot_start + sum(st.bets.values())
            pot_called = pot + 2 * size - st.bets[hero] - st.cur_bet
            short_open = st.n_raises == 0 and stack <= PUSH_FOLD_STACK
            # (at the charts' depth their lines play as written: a 100bb 4bet to 25 over an 11bb 3bet leaves
            # 1.45 pots behind, and under JAM_SPR it went all-in with the chart's bluffs, A5s and K5s jamming
            # 100bb, while the 3bettor's calls of it all became jams; solves 4bet small and call 4bets there)
            spr_bar = JAM_SPR_DEEP if stack >= CHART_STACK - DEPTH_TOL else JAM_SPR
            if (size >= st.bets[hero] + remaining - 1e-9 or st.inv_start[hero] + size >= COMMIT * stack
                    or (stack - st.inv_start[hero] - size) < spr_bar * pot_called or short_open):
                size = st.bets[hero] + remaining
                if short_open or stack < CHART_STACK - DEPTH_TOL:
                    full = self._push(hero, preflop, positions, stack, st, size, full, notes, pool, locks)
                else:
                    notes.append("the raise would commit the stack, so it goes all-in")
                if "call" in full and st.n_raises >= 2 and remaining - to_call < spr_bar * (pot + to_call):
                    # a call that leaves under JAM_SPR pots behind is committed too: it jams instead, for the
                    # fold equity (QQ, AK and JJ called 9 of 18bb behind a 3bet and a cold 4bet at 20bb)
                    full = dict(full, **{"raise": np.minimum(full["raise"] + full["call"], 1.0), "call": np.zeros(N_COMBOS)})
                    notes.append("a call would commit the stack too, so the hands that continue go all-in")
                strat = {k: float(v[idx]) for k, v in full.items()}
        acts = []
        for k, v in strat.items():
            a = {"action": k, "freq": round(v, 3)}
            if k == "raise":
                if size >= st.bets[hero] + remaining - 1e-9:
                    a["action"] = "all-in"
                a["amount_bb"] = round(size, 2)
            elif k == "call":
                a["amount_bb"] = round(to_call, 2)
            acts.append(a)
        best = max(acts, key=lambda a: a["freq"])["action"]
        names = list(full.keys())
        arriving = preflop_ranges(preflop, positions, stack).get(hero, np.ones(N_COMBOS))
        return {
            "street": "preflop", "hero": hero, "hand": hand_class(hole), "players": len(positions),
            "actions": acts, "recommended": best, "sampled": self._sample(acts),
            "pot_bb": round(st.pot_start + sum(st.bets.values()), 2), "to_call_bb": round(to_call, 2),
            "grid": strategy_grid([full[k] for k in names], names, arriving),
            "notes": notes, "seconds": round(time.time() - t0, 3),
        }

    def _study_preflop(self, hero, preflop, positions, stack, t0, res=None):
        """Study mode preflop: the charts' mix for hero's whole range (no EVs or equities: the charts aren't solves).
        `res`: the advice to read it from (the catch-all's chart answer), else Advisor._preflop's."""
        res = res or self._preflop(hero, (0, 1), preflop, positions, stack, t0)   # (any hand: the grid is the range's)
        if "actions" not in res:
            return res
        g = res["grid"]
        n = {k: c["w"] * (6 if len(k) == 2 else 4 if k.endswith("s") else 12) for k, c in g["cells"].items()}
        tot = max(sum(n.values()), 1e-12)
        acts = []
        for a, act in enumerate(res["actions"]):   # (in the grid's order)
            share = sum(n[k] * c["f"][a] for k, c in g["cells"].items())
            acts.append({"action": act["action"], "amount_bb": act.get("amount_bb"), "freq": round(share / tot, 3),
                         "combos": round(share, 1), "ev_bb": None})
        return {"study": "node", "street": "preflop", "actor": hero, "players": res["players"], "board": "",
                "pot_bb": res["pot_bb"], "to_call_bb": res["to_call_bb"], "actions": acts,
                "grid": {"actions": g["actions"], "cells": {k: c for k, c in g["cells"].items() if c["w"] > 0}},
                "range_combos": round(tot, 1),
                "notes": [x for x in res["notes"] if not x.startswith(("equity ", "preflop: chart approximation"))],
                "seconds": round(time.time() - t0, 3)}

    @staticmethod
    def _push(hero, preflop, positions, stack, st, jam, full, notes, pool=False, locks=()):
        """A raise that would commit much of the stack is an all-in: hero's range for it is the push/fold
        equilibrium against the players still in (the last raiser's range calls, if there is one); hands
        the chart continued with that don't shove call. Facing a 3bet or more, the chart's raise is a value
        range already and simply goes all-in: a push/fold equilibrium there (both players left given the
        4bettor's premiums, and no option to call) settled on KK jamming 25% at 20bb."""
        if st.n_raises >= 2:
            notes.append(f"a raise here commits too much of a {jam + st.inv_start[hero]:g}bb stack: the chart's raises go all-in")
            return full
        arr = preflop_ranges(preflop, positions, stack, hero if pool else None, locks)
        last = next((a["pos"] for a in reversed(preflop) if a["action"] in ("raise", "all-in")), None)
        opp = [p for p in st.order if p in st.active and p != hero and p not in st.allin]
        if last:
            call_rng = arr.get(last, np.ones(N_COMBOS))
        else:
            # first in or over limpers: whoever calls is one of the players left, each from their own range
            # (a limper's, not any two cards: the big blind jammed over two limpers with 55% of hands at 12bb)
            rs = [arr.get(p, np.ones(N_COMBOS)) for p in opp] or [np.ones(N_COMBOS)]
            call_rng = sum(r / max(r.sum(), 1e-12) for r in rs) / len(rs) * N_COMBOS
        caller_in = st.bets[last] if last else float(np.mean([st.bets[p] for p in opp]))
        pot = st.pot_start + sum(st.bets.values())
        to_call = min(st.cur_bet - st.bets[hero], jam - st.bets[hero])
        alt = None
        if "call" in full and to_call > 1e-9:
            alt = _flat_ev(full, arr.get(hero, np.ones(N_COMBOS)), call_rng, pot, to_call,
                           stack - st.inv_start[hero] - st.bets[hero] - to_call)
        elif "check" in full:
            # the big blind's free check over limpers: its share of the pot, realized at CHECK_REALIZE
            limpers = [p for p in opp if st.bets[p] > 0] or opp
            eq = _preflop_equity_multi([arr.get(p, np.ones(N_COMBOS)) for p in limpers]) if limpers else None
            if eq is not None:
                alt = CHECK_REALIZE * np.nan_to_num(eq, nan=0.0) * pot
        model = getattr(positions, "icm", None)
        vals = _icm_push_vals(model, st, hero, jam, [last] if last else opp,
                              last or ("BB" if "BB" in opp else opp[-1] if opp else None), arr) \
            if model is not None and opp else None
        push = push_fold(arr.get(hero, np.ones(N_COMBOS)), call_rng, jam, st.bets[hero], caller_in, pot,
                         max(1, len(opp)), rake=getattr(positions, "rake", None), alt=alt, icm=vals)
        out = dict(full)
        out["raise"] = push
        if "call" in full:   # what the chart continued with and doesn't shove calls
            out["call"] = np.minimum(full["call"] + full.get("raise", 0.0), 1.0 - push)
        if "check" in full:
            out["check"] = 1.0 - push
        else:
            out["fold"] = 1.0 - push - out.get("call", 0.0)
            out = pf.monotone(out)
        notes.append(f"a raise here commits too much of a {jam + st.inv_start[hero]:g}bb stack: all-in, "
                     f"range from the push/fold equilibrium against {len(opp)} player{'s' * (len(opp) > 1)}"
                     + (" in prize equity (ICM: both the jams and the calls)" if model is not None else "")
                     + (", bounties included" if getattr(model, "bounty", None) is not None else ""))
        return out

    @staticmethod
    def _priced(hero, preflop, positions, stack, st, full, notes, pool=False):
        """The charts assume a standard-sized raise, nobody else in the pot, 100bb and no antes. When the
        spot differs, re-draw hero's line against the chart's own standard line (same raisers at default
        sizes, everyone else folding, chart depth, no antes):
        - continuing: the chart's continuing share of hero's range there marks an equity bar (its
          weakest continuing hand's equity against the raisers, relative to that line's pot odds); hero
          continues where equity against everyone in, held to the same bar, beats the actual pot odds
          (with implied odds for small pairs and suited hands by how much deeper or shallower than the
          charts). A better price only adds calls, a worse one only removes them.
        - raising, when more players are in: each player's chart facing hero's raise gives their continuing
          range and how often they continue. The chart's raises split into value (the top of hero's range
          by equity when called, as many hands as it raises) and bluffs. A value raise stays only while its
          equity when called (against each continuing range, weighted by how often it continues) clears
          break-even by as much as the chart's weakest value raise heads-up does (more dead money,
          thinner value); bluffs shrink by how much less often everyone folds, against how often a
          bluff of that size into that pot needs them to.
        With `pool` (exploit mode) the other players' ranges and their responses to hero's raise are the
        player pool's (exploit.pool_preflop), so this runs even in the chart's own spot: calls follow
        equity against what the pool actually plays, and bluff raises grow (up to 1.5x) when the pool
        folds more often."""
        def priced(s, stk, table=None, raked_chart=False):
            opp = [p for p in s.order if p in s.active and p != hero and
                   (p in s.acted or s.bets[p] > s.posted.get(p, 0.0) + 1e-9)]
            call = min(s.cur_bet - s.bets[hero], stk - s.inv_start[hero] - s.bets[hero])
            pot = s.pot_start + sum(s.bets.values())
            # with rake, the pot hero wins is short the rake it goes on to pay (_call_rake)
            behind = stk - s.inv_start[hero] - s.bets[hero] - call
            rk = _call_rake(table, pot + call, behind, raked_chart)
            return opp, call / (pot + call - rk), call, pot, behind

        def facing_raise(line, opp, arr, positions, as_pool=False):
            # each player facing hero's raise, by their chart (or as the pool plays it): the chance
            # everyone folds, and hero's equity when called (against each continuing range, weighted by
            # how often it continues)
            line = line + [{"pos": hero, "action": "raise"}]
            if as_pool:   # hero's raising range, for the pool's extra flats against an open
                arr = dict(arr, **{hero: arr.get(hero, np.ones(N_COMBOS)) * full.get("raise", 1.0)})
            f, num, den = 1.0, np.zeros(N_COMBOS), 0.0
            for p in opp:
                w = arr[p]
                s_p = pf.strategy_for(p, line, positions)
                if as_pool:
                    s_p = pool_strategy(s_p, p, line, arr)
                cont = w * (1.0 - s_p.get("fold", 0.0))
                c = float(cont.sum() / max(w.sum(), 1e-12))
                f *= 1.0 - c
                if c > 1e-6:
                    num = num + c * _preflop_equity(cont)
                    den += c
            return f, num / max(den, 1e-12)

        def quantile(eq, w, share):
            ok = ~np.isnan(eq) & (w > 0)
            order = np.argsort(-eq[ok])
            cum = np.cumsum(w[ok][order])
            return float(eq[ok][order][min(np.searchsorted(cum, share * cum[-1]), len(cum) - 1)])

        std_line = [{"pos": a["pos"], "action": "fold"} if a["action"] == "call" and a["pos"] != hero
                    else {"pos": a["pos"], "action": a["action"]} for a in preflop]
        # the chart's own spot: its depth (100bb, or a short-stack chart's: pf.spot_depth), no antes; the game's own
        # where the table's game has its charts (pf.format_chart: their depth and ante)
        cs = pf.spot_depth(hero, preflop, positions)
        ante_std = 0.0
        if isinstance(positions, pf.Table):
            cover = pf.format_chart(hero, preflop, positions)[1]
            if cover:
                cs, ante_std = cover["depth"], cover["ante"]
        table = pf.Table(positions, ante_std, getattr(positions, "straddle", False), depth=cs)
        std = replay_hand(table, cs, {"preflop": std_line}, 1)[0]
        if std.closed or std.to_act != hero:
            return full
        # Rake re-prices every call that sees the flop. The big blind's defence of an open and the opener's
        # calls of a blind's 3bet in position are charts written without rake and pay all of it; the rest
        # were fit to published raked solves and pay only rake beyond theirs (_call_rake)
        rk_on = getattr(positions, "rake", None) is not None
        key_h = pf._norm(pf.chart_key(hero, positions))
        raisers = [a["pos"] for a in preflop if a["action"] in ("raise", "all-in")]
        unraked = ((st.n_raises == 1 and key_h == "BB") or
                   (st.n_raises == 2 and raisers[0] == hero and key_h in ("UTG", "HJ", "CO", "BTN")
                    and pf._norm(pf.chart_key(raisers[1], positions)) in ("SB", "BB")))
        opp, need, call, pot, behind = priced(st, stack, positions if rk_on else None, not unraked)
        raked = rk_on and need > call / max(pot + call, 1e-9) + 1e-12
        # a tournament: the call's price plus the risk premium against the raiser (the players in, limped), and the
        # raises re-drawn with it below
        icm_on = getattr(positions, "icm", None) is not None
        prem = _icm_premium(positions, st, hero, raisers[-1:] or opp, call, pot, behind)[0] if icm_on else 0.0
        need += prem
        opp_std, need_std, _, _, behind_std = priced(std, cs)
        # implied odds: pairs and suited hands win more when they hit, by the stack behind. Deeper or
        # shallower than the charts moves their price by how their odds with those winnings change at
        # this price, so 20bb shallower costs a small pair a little of the implied odds it had at chart
        # depth, not all of them.
        # More players in pay off more often too, by the square root of how many (judgment: a hit is paid by
        # one or two of them, not all), so a suited connector or small pair closing a pot five ways expects
        # 2.2x the chart's heads-up winnings.
        depth = behind - behind_std
        implied = lambda b, n=1: call / np.maximum(pot + call + _IMPLIED_W * max(b, 0.0) * np.sqrt(n), call + 1e-9)
        need_v = need + implied(behind, max(len(opp), 1)) - implied(behind_std, max(len(opp_std), 1))
        more = sorted(opp) != sorted(opp_std)
        reprice = (abs(need - need_std) >= PRICE_TOL or abs(depth) >= DEPTH_TOL or pool or raked
                   or (more and st.n_raises >= 1 and bool(opp_std)))
        if not (more or reprice or icm_on):
            return full
        raise_c = full.get("raise", np.zeros(N_COMBOS))
        call_c = full["call"]
        arr_std = pf.replay_ranges(std_line, table)
        w = arr_std[hero]
        tot = max(w.sum(), 1e-12)
        eq_std = _preflop_equity_multi([arr_std[p] for p in opp_std])
        arr = preflop_ranges(preflop, positions, stack, hero if pool else None)
        eq = _preflop_equity_multi([arr[p] for p in opp])
        # realized equity: all-in equity times the share of it the hand realizes after calling the last raise
        # (gto/pf_eqr.py, measured with the postflop solver); see the re-ranking below
        # (measured heads-up at the standard sizes' stack-to-pot ratios: not with more players in, nor after a
        # raise that leaves much less behind)
        eq_r = None
        spr = lambda b, p, c: b / max(p + c, 1e-9)
        if (raisers and st.n_raises < 4 and not pool and len(opp) == 1
                and spr(behind, pot, call) >= RERANK_SPR * spr(behind_std, std.pot_start + sum(std.bets.values()),
                                                               std.cur_bet - std.bets[hero])):
            eq_r = np.nan_to_num(eq * _call_realization(hero, raisers[-1], st.n_raises, arr, positions), nan=0.0)
        out = dict(full)
        msg = []
        value_n = bluff_n = raise_c
        value_c = raise_c
        r = float((w * raise_c).sum() / tot)
        if (more or pool or icm_on) and r > 0.005:
            f, eq_c = facing_raise(preflop, opp, arr, positions, pool)
            f_std, eq_c_std = facing_raise(std_line, opp_std, arr_std, table)
            v_bar = quantile(eq_c_std, w, r)
            value_c = np.where(eq_c_std >= v_bar, raise_c, 0.0)
            bluff_c = raise_c - value_c
            # held to the bar over break-even (equity when called that pays for the raise), since the
            # callers' dead money in a squeeze makes thinner value raises pay
            def breakeven(s, stk):
                to = min(_default_size(s, hero, True), stk - s.inv_start[hero])
                add = to - s.bets[hero]
                return add / max(s.pot_start + sum(s.bets.values()) + add + to - s.cur_bet, 1e-9)
            # (a tournament: the raise's risk premium against the players left, at the SPR of a called raise, on top)
            pr = 0.0
            if icm_on:
                to_r = min(_default_size(st, hero, True), stack - st.inv_start[hero])
                pr = _icm_premium(positions, st, hero, opp, 2 * to_r - st.bets[hero] - st.cur_bet, pot,
                                  stack - st.inv_start[hero] - to_r)[0]
            keep = np.clip(0.5 + (eq_c - breakeven(st, stack) - pr - v_bar + breakeven(std, cs)) / RAISE_BLEND,
                           0.0, 1.0)
            keep[np.isnan(keep)] = 0.0
            # only ever tighter than the chart: the charts' continuing ranges don't see the bigger
            # squeeze size, and capped callers would otherwise make value raises look wider
            value_n = np.minimum(value_c, keep)
            # bluffs: by how often everyone folds over how often they must for a pure bluff to break even
            # (the raise over the pot before it). The callers' dead money lowers that bar, so a squeeze
            # bluff needs fewer folds than its heads-up 3bet (BB over CO + BTN: 64% against 69%). Fold
            # rates alone had the big blind's squeezes at 0.87-0.90x its heads-up 3bets; published solves
            # have 0.95x (GTO Wizard NL500 40bb, CO open + BTN call: 9.8% against 10.3%,
            # https://blog.gtowizard.com/how-to-construct-a-squeezing-range/) to 1.10x (Poker Academy
            # NL200 100bb, BTN open + SB call: 12.7% against 11.5%). Now 0.88-0.94x.
            def fold_need(s, stk):
                to = min(_default_size(s, hero, True), stk - s.inv_start[hero])
                add = to - s.bets[hero]
                return add / max(s.pot_start + sum(s.bets.values()) + add, 1e-9)
            # (a seat with no chips behind risks nothing: no bar, the chart's bluffs as they are)
            fn_now, fn_std = fold_need(st, stack) + pr, fold_need(std, cs)
            scale = min(1.5 if pool else 1.0, (f / fn_now) / max(f_std / fn_std, 1e-9)) \
                if fn_now > 1e-9 and fn_std > 1e-9 else 1.0
            bluff_n = bluff_c * scale
            out["raise"] = np.minimum(1.0, value_n + bluff_n)
            who = 'the player pool' if pool and not more else (f'{len(opp)} players in' if more else 'the tournament')
            msg.append(f"raises re-drawn for {who}: value raises keep the equity over break-even "
                       f"against what continues that the chart's weakest value raise has heads-up ({v_bar:.0%})"
                       + (f", plus a {pr:.1%} risk premium (ICM)" if pr > 0.0005 else "") + f"; bluffs x{scale:.2f} "
                       f"(everyone folds {f:.0%} vs {f_std:.0%}, needing {fn_now:.0%} vs "
                       f"{fn_std:.0%})")
        raise_n = out.get("raise", raise_c)
        # continuing before the price: the chart's calls and value raises, kept bluffs, new value raises
        base = np.minimum(1.0, np.maximum(call_c + value_c + (raise_n - value_n if more or pool or icm_on else raise_c - value_c), raise_n))
        new = base
        if reprice:
            q = float((w * (call_c + raise_c)).sum() / tot)
            if 0.02 < q < 0.98:
                # a better price adds hands the chart folds that clear its bar (the equity at its continuing share)
                realize = need_std / max(quantile(eq_std, w, q), 1e-6)
                add = np.clip(0.5 + (eq * realize - need_v) / PRICE_BLEND, 0.0, 1.0)
                add[np.isnan(add)] = 0.0
                # a worse price removes calls, never raises. The chart keeps playable hands (small pairs,
                # suited connectors) below that raw-equity bar, so its continuing hands are held to its
                # weakest ones instead: a slightly worse price trims the bottom, not every such hand.
                floor = quantile(eq_std, w * (call_c + raise_c), 1.0 - CUT_FLOOR)
                cut = np.clip(1.0 + (eq * need_std / max(floor, 1e-6) - need_v) / PRICE_BLEND, 0.0, 1.0)
                cut[np.isnan(cut)] = 0.0
                # a hand gains where the price is better, or (exploit mode) where it does better against the
                # pool's range than against the chart's
                better = need_v < need_std
                if pool:
                    better = better | (np.nan_to_num(eq, nan=0.0) > np.nan_to_num(eq_std, nan=0.0) + 1e-6)
                new = np.where(better, np.maximum(base, add), np.minimum(base, np.maximum(cut, raise_n)))
                if more and not pool and st.n_raises >= 1 and opp_std:
                    # more players in than the chart's: every call is re-drawn by its equity against all of
                    # them at the chart's own bar (the realization that makes its weakest continuing hand break
                    # even), not only added to or cut from the chart's calls. Equity shared three ways is
                    # bunched, and the chart's calls (built on heads-up equity) aren't a floor: Poker Academy's
                    # NL200 solve has the big blind calling 16.1% over a button open and a small-blind call at a
                    # 20% price, against 29.0% heads-up at 27.3%; adding to the chart's calls had it at 44.5%
                    # (5% capped at 4bb), the re-draw 15.0%.
                    redraw = np.clip(0.5 + (eq * realize - need_v) / PRICE_BLEND, 0.0, 1.0)
                    redraw[np.isnan(redraw)] = 0.0
                    new = np.maximum(redraw, raise_n)
                if eq_r is not None:
                    new = _rerank(new, base, raise_n, eq_r, w, better)
                deep =(f", {abs(depth):g}bb {'deeper' if depth > 0 else 'shallower'} than the charts "
                        f"(implied odds for pairs and suited hands)" if abs(depth) >= DEPTH_TOL else "")
                rk = " after rake" if raked else ""
                if prem > 0.0005:
                    rk += f" with a {prem:.1%} risk premium (ICM)"
                elif prem < -0.0005:
                    rk += f" with a {-prem:.1%} bounty discount (busting a player you cover pays their bounty)"
                msg.insert(0, f"price {need:.0%} of the pot{rk} against {', '.join(opp)} (the chart assumes "
                              f"{need_std:.0%} against {', '.join(opp_std)}{deep}): calls re-drawn by equity "
                              f"against the pot odds" + (", against the player pool's ranges" if pool else ""))
        if "raise" in full:
            out["raise"] = np.minimum(raise_n, new)
        out["call"] = new - out.get("raise", 0.0)
        out["fold"] = 1.0 - new
        notes.extend(msg)
        return pf.monotone(out)

    @staticmethod
    def _icm_open(hero, preflop, positions, stack, st, full, notes):
        """A tournament, raising first in (or over limpers): the chart's raises less the weakest of them, by equity
        against what the players left continue with, as the risk premium raises the steal's price. In chips the raise
        needs everyone to fold add / (pot + add) of the time; in prize equity that plus the premium against them
        (averaged, at the SPR of a called raise), so the share raised scales by the chips' price over the prizes'
        (a 2.5bb open into 2.5bb needing 50% folds, with a 10% premium: 5/6 of the chart's opens). Bounties on players
        hero covers can make it a discount (Model.premium): the share then grows the same way (up to OPEN_WIDEN times
        the chart's), the best by equity against what continues among the hands the chart folds added first."""
        if "raise" not in full:
            return full
        opp = [p for p in st.order if p in st.active and p != hero and p not in st.allin]
        if not opp:
            return full
        to = min(_default_size(st, hero, True), stack - st.inv_start[hero])
        add = to - st.bets[hero]
        pot = st.pot_start + sum(st.bets.values())
        prem = _icm_premium(positions, st, hero, opp, add + max(to - st.cur_bet, 0.0), pot,
                            stack - st.inv_start[hero] - to)[0]
        fn = add / max(pot + add, 1e-9)
        widen = prem < -0.0005 and fn > 0      # (a bounty discount)
        if (prem < 0.0005 and not widen) or fn <= 0:
            return full
        r = full["raise"]
        share = float(r.mean())
        target = share * fn / (fn + prem) if not widen else min(1.0, share * min(OPEN_WIDEN, fn / max(fn + prem, 1e-9)))
        # by equity against what continues behind (each player's chart facing the raise)
        arr = preflop_ranges(preflop, positions, stack)
        line = preflop + [{"pos": hero, "action": "raise"}]
        cont = np.zeros(N_COMBOS)
        for p in opp:
            s_p = pf.strategy_for(p, line, positions)
            c = arr.get(p, np.ones(N_COMBOS)) * (1.0 - s_p.get("fold", 0.0))
            cont += c / max(c.sum(), 1e-12)
        eq = np.nan_to_num(_preflop_equity(cont), nan=0.0)
        order = np.argsort(-eq)
        rest = "check" if "check" in full else "fold"
        out = dict(full)
        if widen:
            # the hands the chart folds (or checks), best first, raised until the share is the target
            avail = np.asarray(full.get(rest, np.zeros(N_COMBOS)), dtype=np.float64)
            cum = np.cumsum(avail[order])
            take_o = np.clip(((target - share) * N_COMBOS - (cum - avail[order])) / np.maximum(avail[order], 1e-12),
                             0.0, 1.0)
            take = np.zeros(N_COMBOS)
            take[order] = take_o
            out["raise"] = r + avail * take
            out[rest] = avail * (1.0 - take)
            notes.append(f"tournament: raises {float(out['raise'].mean()):.0%} of hands instead of the chart's {share:.0%}, "
                         f"the best against what continues added (a {-prem:.1%} bounty discount on a steal needing "
                         f"{fn:.0%} folds in chips: busting a player you cover pays their bounty)")
            return out
        cum = np.cumsum(r[order])
        keep_o = np.clip((target * N_COMBOS - (cum - r[order])) / np.maximum(r[order], 1e-12), 0.0, 1.0)
        keep = np.zeros(N_COMBOS)
        keep[order] = keep_o
        new_r = r * keep
        out["raise"] = new_r
        out[rest] = full.get(rest, 0.0) + (r - new_r)
        notes.append(f"tournament: raises {target:.0%} of hands instead of the chart's {share:.0%}, the weakest against "
                     f"what continues dropped (a {prem:.1%} risk premium on a steal needing {fn:.0%} folds in chips)")
        return out

    @staticmethod
    def _locked(hero, preflop, positions, stack, st, full, notes, pool=False, locks=()):
        """Your locks on players still in (_lock_spec; exploit.lock_preflop sets their strategies), against `full`, the
        advice without them (the chart re-priced for this spot by _priced, against the pool with `pool`). Hero's line
        is re-drawn against the locked ranges and answers at the bars `full` itself sets, so only what the lock
        changes moves (a spot the charts don't cover keeps its own adjustments):
        - calls, with a locked player in the pot: each hand's equity against everyone in moves by what it gains or loses
          against the locked ranges instead of the unlocked ones, priced at the realization that makes the hand at
          full's continuing share (by that equity) break even, as _priced's re-pricing. A continuing hand keeps
          continuing while what it had over the price covers the loss (one below that bar, a hand continuing for its
          implied odds or playability, has no such margin: it goes as soon as it loses equity); a folding hand starts
          once the gain covers its shortfall. (All-in equity, not realized: the measured realizations are against the
          charts' ranges, and KQs realizes far less against a nit's QQ+/AK than against them.)
        - raises, with a locked player to answer them (in the pot or behind: an open with a locked blind, a 3bet of a
          locked opener): full's raises split into value (the top of hero's range by equity when called, as many hands
          as it raises) and bluffs, as _priced does. A value raise stays while its equity against what the locked players
          continue with clears the weakest value raise's unlocked; one that no longer does calls where the lock costs
          the raise more than the call, else raises on (first in, with no call, it is a steal like the bluffs); a hand
          under that bar unlocked that now clears it raises (from its calls, then its folds). Bluffs scale by how often
          everyone folds, locked over unlocked (at most LOCK_BLUFF_MAX times), the extra first from the hands already
          mixing raises with folds, then the strongest folds by equity when called. Equity when called lumps re-raises
          in with calls, as _priced does."""
        spec = tuple((p, kv) for p, kv in _lock_spec(locks) if p in st.active and p != hero and p not in st.allin)
        if not spec:
            return full
        from .exploit import describe_lock, _move
        lk = {p: dict(kv) for p, kv in spec}
        pe = hero if pool else None
        arr0 = preflop_ranges(preflop, positions, stack, pe)
        arrL = preflop_ranges(preflop, positions, stack, pe, spec)
        w = arr0.get(hero, np.ones(N_COMBOS))
        tot = max(float(w.sum()), 1e-12)
        share = lambda v: float((w * v).sum() / tot)
        opp = [p for p in st.order if p in st.active and p != hero and
               (p in st.acted or st.bets[p] > st.posted.get(p, 0.0) + 1e-9)]
        behind = [p for p in st.order if p in st.active and p != hero and p not in opp and p not in st.allin]
        out = {k: np.broadcast_to(np.asarray(v, dtype=np.float64), (N_COMBOS,)).copy() for k, v in full.items()}
        on = "the pool" if pool else "GTO"
        msg = []
        for p in lk:
            where = (f"their range here is {arrL[p].sum():.0f} combos ({arr0[p].sum():.0f} unlocked)" if p in opp
                     else "to act behind you")
            msg.append(f"locked: {p} {describe_lock(lk[p], preflop=True)} (on {on}); preflop {where}")
        # calls, against the locked ranges of the players in the pot: each hand's margin over the price unlocked (v0)
        # and what the lock adds or takes (delta)
        lk_in = [p for p in opp if p in lk]
        to_call = min(st.cur_bet - st.bets[hero], stack - st.inv_start[hero] - st.bets[hero])
        cont0 = np.broadcast_to(full.get("call", 0.0) + full.get("raise", 0.0), (N_COMBOS,))
        q0 = share(cont0)
        priced = "call" in out and "fold" in out and lk_in and to_call > 1e-9 and 0.02 < q0 < 0.98
        if priced:
            eq0 = np.nan_to_num(_preflop_equity_multi([arr0[p] for p in opp]), nan=0.0)
            eqL = np.nan_to_num(_preflop_equity_multi([arrL[p] for p in opp]), nan=0.0)
            pot = st.pot_start + sum(st.bets.values())
            left = stack - st.inv_start[hero] - st.bets[hero] - to_call
            need = to_call / max(pot + to_call - _call_rake(positions, pot + to_call, left), 1e-9)
            k = need / max(_quantile(eq0, w, q0), 1e-6)
            v0 = k * eq0 - need
            delta = k * (eqL - eq0)
        # raises: against what the players facing it do, locked and not
        face = opp + [p for p in behind if p in lk]
        raise0 = out.get("raise")
        if raise0 is not None and share(raise0) > 0.005 and any(p in lk for p in face) and can_raise(st, hero):
            line = preflop + [{"pos": hero, "action": "raise"}]

            def facing(arr, locked):
                # the chance everyone facing the raise folds, and hero's equity when called (against each continuing
                # range, weighted by how often it continues), as _priced's facing_raise
                arr = dict(arr, **{hero: w * raise0})
                f, num, den = 1.0, np.zeros(N_COMBOS), 0.0
                for p in face:
                    wp = arr.get(p, np.ones(N_COMBOS))
                    s_p = pf.strategy_for(p, line, positions)
                    if pool:
                        s_p = pool_strategy(s_p, p, line, arr)
                    if locked and p in lk:
                        s_p = lock_strategy(s_p, p, line, arr, lk[p], positions)
                    cont = wp * (1.0 - s_p.get("fold", 0.0))
                    c = float(cont.sum() / max(wp.sum(), 1e-12))
                    f *= 1.0 - c
                    if c > 1e-6:
                        num = num + c * np.nan_to_num(_preflop_equity(cont), nan=0.0)
                        den += c
                return f, num / max(den, 1e-12)

            f0, e0 = facing(arr0, False)
            fL, eL = facing(arrL, True)
            v_bar = _quantile(e0, w, share(raise0))
            # each hand's raise is value as far as its equity when called clears the bar (ramped), the rest bluff
            ramp = lambda e: np.clip(0.5 + (e - v_bar) / RAISE_BLEND, 0.0, 1.0)
            k0, kL = ramp(e0), ramp(eL)
            # (the hands that play for implied odds, small pairs and suited non-broadways, never turn into new value
            # raises: all-in equity overrates them, and a lock that has opponents calling more pays their flats more,
            # not their raises; ranked by it a station's wide calls had BTN 3betting 44 and 55 33-46% for value over
            # a CO open, and the monotone pass then had 44-66 calling 70% where the chart folds half)
            kL = np.where(_IMPLIED_W > 0, np.minimum(kL, k0), kL)
            kept = raise0 * np.minimum(k0, kL)               # value that still clears it
            lost = raise0 * np.maximum(k0 - kL, 0.0)         # value that no longer does
            now = raise0 * np.maximum(kL - k0, 0.0)          # bluffs that now do: value, not scaled
            bluff = raise0 * (1.0 - np.maximum(k0, kL))      # still bluffs
            gain = np.maximum(kL - k0, 0.0) * (1.0 - raise0)   # new value raises, from calls, then folds
            passive = "check" if "check" in out else (
                "call" if "call" in out and st.cur_bet > st.bets[hero] + 1e-9 else None)
            sink = "fold" if "fold" in out else passive
            got = np.zeros(N_COMBOS)
            for src in dict.fromkeys(k for k in (passive, sink) if k):
                take = np.minimum(gain - got, out[src])
                out[src] = out[src] - take
                got = got + take
            # a lost value raise calls (checks) instead where the lock costs the raise more than the call: the raise's
            # EV (everyone folds: the pot; called: its equity of the pot then, less what it puts in) against the call's
            # (its equity against the locked ranges, at the price); first in, with no call, it is a bluff like the rest
            # (a nit's 3bet: QQ calls rather than 4bets; an overfolder opening: KQs 3bets on, for the folds)
            pot = st.pot_start + sum(st.bets.values())
            to = min(_default_size(st, hero, True), stack - st.inv_start[hero])
            add = to - st.bets[hero]
            pot_c = pot + add + max(to - st.cur_bet, 0.0)
            d_raise = (fL - f0) * pot + (1.0 - fL) * (eL * pot_c - add) - (1.0 - f0) * (e0 * pot_c - add)
            d_call = k * (eqL - eq0) * (pot + to_call) if priced else np.zeros(N_COMBOS)
            calls = np.zeros(N_COMBOS) if passive is None else (d_call > d_raise).astype(float)
            out[passive or sink] = out[passive or sink] + lost * calls
            kept = kept + lost * (1.0 - calls) * (passive is not None)
            bluff = bluff + lost * (1.0 - calls) * (passive is None)
            s = float(np.clip(fL / f0, 0.0, LOCK_BLUFF_MAX)) if f0 > 1e-6 else 1.0
            out[sink] = out[sink] + bluff * (1.0 - min(s, 1.0))   # bluffs dropped fold
            raise_n = kept + now + got + bluff * min(s, 1.0)
            if s > 1.0:
                # more bluffs: first the hands already mixing raises with folds (the baseline's own bluffs), then the
                # strongest folds by equity when called against the locked continuers
                mixed = (raise_n >= 0.02) & (out[sink] >= 0.02)
                sm = np.stack([out[sink], raise_n])[None]
                _move(sm, 0, 1, w[None], np.lexsort((-eL, ~mixed))[None], (s - 1.0) * share(bluff))
                out[sink], raise_n = sm[0, 0], sm[0, 1]
            out["raise"] = np.minimum(raise_n, 1.0)
            who = ", ".join(p for p in face if p in lk)
            msg.append(f"raises re-drawn against {who} as locked: everyone folds {fL:.0%} ({f0:.0%} unlocked), so bluffs "
                       f"x{s:.2f}; value raises keep the equity when called the weakest one has unlocked ({v_bar:.0%}): "
                       f"you raise {share(out['raise']):.0%} of your range ({share(raise0):.0%} unlocked)")
        if priced:
            # a continuing hand keeps continuing while its margin covers what it loses; a folding one starts once the
            # gain covers its shortfall (raises are never cut here)
            rs = out.get("raise", np.zeros(N_COMBOS))
            cont = rs + out["call"]
            drop = np.maximum(cont * np.clip(1.0 + (np.maximum(v0, 0.0) + delta) / PRICE_BLEND, 0.0, 1.0), rs)
            add = cont + (1.0 - cont) * np.clip((delta - np.maximum(-v0, 0.0)) / PRICE_BLEND, 0.0, 1.0)
            new = np.where(delta < 0, drop, add)
            out["call"] = new - rs
            out["fold"] = 1.0 - new
            msg.append(f"calls re-drawn by equity against {', '.join(lk_in)}'s locked range{'s' * (len(lk_in) > 1)}: "
                       f"you continue with {share(new):.0%} of your range ({q0:.0%} unlocked)")
        notes.extend(msg)
        return pf.monotone(out)

    def _vs_jam(self, hero, hole, preflop, positions, stack, st, t0, pool=False, locks=()):
        """Facing a preflop all-in: call when equity against the jammer's range beats the pot odds.
        With `pool`, a jam over a raise comes from the player pool's raising range (premiums, mostly); with
        `locks` (_lock_spec), a locked jammer's from their raising range as locked."""
        ji = max(i for i, a in enumerate(preflop) if a["action"] in ("raise", "all-in", "bet"))
        jammer = preflop[ji]["pos"]
        pool_except = hero if pool else None
        notes = ["preflop: facing an all-in, call when equity beats the pot odds"]
        rng, more = _jam_range(preflop, ji, positions, stack, pool_except, locks)
        notes += more
        remaining = stack - st.inv_start[hero] - st.bets[hero]
        to_call = min(st.cur_bet - st.bets[hero], remaining)
        pot = st.pot_start + sum(st.bets.values())
        # hero's call only wins what the jammer can match
        win = pot - max(0.0, st.bets[jammer] - (st.bets[hero] + to_call))
        # a called all-in sees the flop, so the pot is raked
        need = to_call / (win + to_call - _rake(positions, win + to_call))
        if getattr(positions, "icm", None) is not None:
            chips = need
            need = _icm_need(positions, st, hero, jammer, to_call, need)
            bty = getattr(positions.icm, "bounty", None) is not None
            notes.append(f"tournament: calling needs {need:.1%} equity in prize equity{' with the bounties' if bty else ''} "
                         f"against {chips:.1%} in chips ("
                         + (f"a risk premium of {need - chips:+.1%}" if need >= chips or not bty else
                            f"a bounty discount of {chips - need:.1%}") + ")")
        # everyone who already called the jam is in the pot too: equity against all of them
        after = preflop_ranges(preflop, positions, stack, pool_except, locks)
        callers = [a["pos"] for a in preflop[ji + 1:] if a["action"] in ("call", "all-in") and a["pos"] in st.active]
        eq = _preflop_equity_multi([rng] + [after[p] for p in callers])
        idx = COMBO_INDEX[(min(hole), max(hole))]
        call = np.clip(0.5 + (eq - need) / JAM_BLEND, 0.0, 1.0)
        call[np.isnan(call)] = 0.0
        if callers:
            notes.append(f"{', '.join(callers)} already called: equity is against {jammer} and them")
        behind = [p for p in st.active if p not in (hero, jammer) and p not in st.allin and p not in callers]
        if behind:
            notes.append(f"{', '.join(behind)} still to act behind (not counted)")
        notes.append(f"equity {eq[idx]:.1%} vs {need:.1%} needed")
        acts = [{"action": "call", "freq": round(float(call[idx]), 3), "amount_bb": round(to_call, 2)},
                {"action": "fold", "freq": round(1 - float(call[idx]), 3)}]
        arriving = after.get(hero, np.ones(N_COMBOS))
        return {
            "street": "preflop", "hero": hero, "hand": hand_class(hole), "players": len(positions),
            "actions": acts, "recommended": "call" if call[idx] >= 0.5 else "fold", "sampled": self._sample(acts),
            "pot_bb": round(pot, 2), "to_call_bb": round(to_call, 2), "equity": round(float(eq[idx]), 4),
            "grid": strategy_grid([call, 1 - call], ["call", "fold"], arriving),
            "notes": notes, "seconds": round(time.time() - t0, 3),
        }

    # ------------------------------------------------------------ postflop
    def _postflop(self, hero, hole, board, stack, streets, positions, t0, pool=False, actor=None, custom=None,
                  optimize=False, locks=None):
        """`actor` (seat, hole): the opponent to act instead, heads-up, read off hero's solve (see play).
        `locks` (exploit.parse_locks): the opponents' strategies as you read them (heads-up; the advice best-responds)."""
        notes: List[str] = []
        n_streets = {3: 2, 4: 3, 5: 4}[len(board)]
        rep = replay_hand(positions, stack, streets, n_streets)
        if len(rep) < n_streets:
            return {"error": "an earlier street's betting is not finished", "notes": notes}
        cur = rep[-1]
        stk = _table_stacks(positions, stack)   # each seat's stack
        if hero not in cur.active:
            return {"error": "hero has folded"}
        if cur.closed:
            return {"error": "this street's betting is already closed (deal the next card)"}
        if cur.to_act != (actor[0] if actor else hero):
            return {"error": f"it is {cur.to_act}'s turn, not {actor[0] if actor else hero}'s"}
        # a copy: narrowing below replaces entries, and the cached dict is shared with later requests
        # (with your locks: their preflop decisions as locked, so a nit's 3bet pot starts from their tighter range)
        pf_r = preflop_ranges(streets["preflop"], positions, stack, hero if pool else None, locks)
        ranges = dict(pf_r)
        bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        for p, r in (custom or {}).items():
            if p not in rep[0].active:
                continue   # folded preflop: nothing to override
            if (r * bm).sum() <= 0:
                # (the board blocks every hand of it: the chart's range answers, with a note, rather than no answer)
                notes.append(f"{'your' if p == hero else p + chr(39) + 's'} custom range has no combos left on this "
                             f"board, so the preflop chart's range is used")
                continue
            ranges[p] = r
            notes.append(f"{'your' if p == hero else p + chr(39) + 's'} range entering the flop is the custom one "
                         f"({(r * bm).sum():.0f} combos), not the preflop chart's")
        for p in [q for q in getattr(pf_r, "rebuilt", []) if q not in (custom or {})]:
            notes.append(f"{'your' if p == hero else p + chr(39) + 's'} preflop line isn't in the chart, so "
                         f"{'your' if p == hero else 'their'} range is the hands that take its last action")
        if self._runout_close is not None and len(cur.active) > 2:
            return {"error": f"the runout report reads heads-up pots: {len(cur.active)} players are in"}
        if self._instant and (actor or pool or optimize or locks or custom or self._runout_close is not None
                              or any(len(st.players_start) > 2 for st in rep[1:])):
            raise _NoInstant()   # (heads-up from the flop at equilibrium only)
        if actor and actor[1] is None and any(len(st.players_start) > 2 for st in rep[1:]):
            return {"skipped": "multiway"}   # (a presolve: heads-up from the flop only, as an instant answer)
        # a 3-way flop's first decision the 3-way library holds: its precomputed solve answers (no narrowing needed)
        if len(cur.players_start) == 3 and actor is None and not pool and not optimize and not self._jam \
                and getattr(self, "_icm", None) is None and self._one_size is None:
            res = self._mw_from_library(hero, hole, board, ranges, cur, stk, streets, t0, notes, locks, positions)
            if res is not None:
                return res
        # Streets that started with 3+ players can't be solved; narrow ranges there by rule.
        if self._lib_only and len(cur.players_start) > 2:
            return {"skipped": "multiway: not a flop-library spot"}
        mw_streets = [st for st in rep[1:] if len(st.players_start) > 2]
        big = [st for st in mw_streets if len(st.players_start) > 4]
        if self._study and big:
            return {"error": f"Study mode browses heads-up, 3-way and 4-way pots after the flop: "
                             f"{len(big[0].players_start)} players saw the {big[0].name}"}
        slow: Dict[str, np.ndarray] = {}
        by_rules = []
        for st in mw_streets:
            before = dict(ranges)
            _narrow_street(ranges, board[:{"flop": 3, "turn": 4, "river": 5}[st.name]], st, stk, hero, hole, slow)
            # 3-way: through the street's 3-way solve where there is one (the one your advice on it came from, or
            # one from the street's start), as far as it has settled; the rest by the rules
            # (a street nobody has acted on yet, the current one at its first decision, has nothing to narrow)
            if not (len(st.players_start) == 3 and self._mw_walk(ranges, before, board, st, stk, hero, hole, streets,
                                                                  notes, current=st is cur)) and st.actions:
                by_rules.append(st.name)
            # (a line the rules leave nearly face-up is filled out as heads-up streets are: a 3bet pot's turn
            # pot-sized bet into two kept 1.9 combos, and heads-up on the river the check left 0.5, where
            # checking and jamming tied for every hand)
            for p in st.players_start:
                if p in ranges and p in before and p != hero:
                    ranges[p] = _post_floor(before[p], ranges[p], board, p, st.name, notes)
        if by_rules:
            notes.append(f"{', '.join(by_rules)}: multiway, ranges narrowed by "
                         "simple betting rules rather than a solve")
        remaining = [p for p in cur.order if p in cur.active]
        pot_now = cur.pot_start + sum(cur.bets.values())
        to_call = cur.cur_bet - cur.bets[hero]
        hero_remaining = stk[hero] - cur.inv_start[hero] - cur.bets[hero]

        live_opps = [p for p in remaining if p != hero]
        if len(live_opps) != 1 or cur.allin & set(remaining) - {hero} and len(remaining) > 2:
            if self._one_size is not None and not actor:
                return {"error": f"the one-size view is for heads-up pots: {len(remaining)} players are in"}
            if self._lib_only:
                return {"skipped": "multiway: not a flop-library spot"}
            if actor:
                move = self._mw_actor_move(board, hero, hole, stk, streets, cur, actor) if len(remaining) >= 3 else None
                return move or {"error": "multiway"}
            if optimize:
                notes.append("bet sizes are optimized only heads-up (multiway is a heuristic, not a solve)")
            if locks and any(p in locks for p in live_opps):
                notes.append(f"locks ({', '.join(p for p in live_opps if p in locks)}) apply heads-up only after the flop: "
                             f"this multiway pot is {'solved' if len(remaining) == 3 else 'played by the betting rules'} without them "
                             f"(they shape the ranges preflop)")
            if pool:
                notes.append("exploit mode applies heads-up after the flop: this multiway pot is played at equilibrium "
                             "(the pool's leaks only shape the ranges preflop)")
            n_notes = len(notes)
            res = self._multiway_solve(hero, hole, board, ranges, remaining, cur, stk, t0, notes,
                                       line=_mw_line(board, hero, hole, stk, streets, cur.name, len(streets.get(cur.name) or [])),
                                       positions=positions)
            if res is not None:
                return res
            if getattr(self, "_icm", None) is not None:
                notes.append("tournament: the betting rules play this multiway pot in chips (the 3- and 4-way solves "
                             "pay prize equity)")
            if self._study:   # (Study shows a solve or nothing: not the betting rules)
                why = notes[n_notes] if len(notes) > n_notes else (
                    f"the {len(remaining)}-way solver plays {len(remaining)} players with chips and no bet bigger than the "
                    f"shortest stack can call" if len(remaining) <= 4 else "it solves at most 4-way pots before the river")
                return {"error": f"Study can't solve this {len(remaining)}-way spot: {why}"}
            return self._multiway(hero, hole, board, ranges, remaining, cur, stk, t0, notes)
        villain = live_opps[0]
        post = cur.order
        oop, ip = sorted([hero, villain], key=post.index)
        hero_p = 0 if hero == oop else 1
        r0, r1 = ranges[oop].copy(), ranges[ip].copy()
        full = (ranges[oop], ranges[ip])   # the ranges entering the flop: they pick a precomputed spot (flop_library)
        # (the library's solves are in chips: a tournament solves live; the flop's runout report reads the turn under a
        # live flop solve, as the library's entries hold the flop only, unless the entry's turn file holds the line
        # (Advisor._store_report), while the river's keeps the flop Study answered)
        lib = self.library if self.library is not None and not pool and not optimize and not (locks or {}).get(villain) \
            and not getattr(self, "_jam", False) and getattr(self, "_icm", None) is None else None
        lib_sol = None
        for c in hole:  # villain can't hold hero's cards
            if hero_p == 0:
                r1 = r1 * (1.0 - MEMBER_FULL[c])
            else:
                r0 = r0 * (1.0 - MEMBER_FULL[c])

        hu_from_start = cur.players_start == remaining
        opt = {}
        if optimize and not (hu_from_start and to_call <= 1e-9):
            notes.append("bet sizes are optimized only where you can bet (nothing to call) on a street that began heads-up")
        committed = _committed(villain, rep[1:])
        adj = (locks or {}).get(villain)

        def mk_lock(sol, next_nodes=()):
            # the opponent as hero plays against them: the pool (exploit mode) and/or your locks, else None (GTO)
            from .exploit import PoolLock, TendencyLock
            base = PoolLock(sol.ctx, 1 - hero_p, committed) if pool else None
            return TendencyLock(sol.ctx, 1 - hero_p, adj, base, next_nodes) if adj else base
        widened: List[str] = []
        rare_lines: List[str] = []
        spare, t_prior = 0.0, time.time()   # earlier streets' budgets (prior_budget each) left unspent
        slow_hu: Dict[int, np.ndarray] = {}   # per player, the combos a call by the rules slowplayed (_slowplayed)
        slow_rules: Dict[str, np.ndarray] = {}   # ...the same by seat, for a street the rules narrow alone (quick)
        walked: Dict[str, tuple] = {}   # street -> (solve, node its line ends at, tree names taken): InstantSpot.prev
        if hu_from_start:
            # narrow through earlier streets that were also heads-up between these two
            for st in rep[1:-1]:
                sub_board = board[:{"flop": 3, "turn": 4, "river": 5}[st.name]]
                # the next street's board: its ranges leave out the cards dealt by then, not later ones, so they are
                # the ranges that street's own advice solved with, and its solve is found in the cache (the river's
                # turn was solved again for its narrowing, its ranges short of the river card's combos: 3 s of the
                # browser's 3.4 s river)
                nxt_board = board[:len(sub_board) + 1]
                if st.players_start != remaining:
                    continue   # multiway: narrowed above
                spare += self.prior_budget
                eff = min(stk[oop] - st.inv_start[oop], stk[ip] - st.inv_start[ip])
                cfg = self._cfg_with_real_sizes(st, oop, ip, eff)
                taken: List[str] = []
                hit = self._from_library(lib, full, sub_board, r0, r1, st, eff, cfg, oop, ip, notes,
                                         smoothing=self.prior_smoothing, board=sub_board, rules=(0, 1), slow=slow_hu,
                                         taken=taken) \
                    if lib is not None and st.name == "flop" else None
                if hit is not None:
                    sol, node, c0, c1 = hit
                    notes.append(sol.note())
                else:
                    if self.prior_model:
                        cfg = dataclasses.replace(cfg, **self.prior_model)
                    if cfg.root_allin:
                        cfg = dataclasses.replace(cfg, root_allin=False)   # (a jam request: the street advised on only)
                    # (your hand out of your range here: the ranges with it in, as the street's own advice solved, so
                    # that solve is the one read, not a second without it: Advisor._as_advised)
                    s0, s1 = self._as_advised(r0, r1, hero_p, hole)
                    # (the hand's own solve of this street, made before a later action took a size off its tree)
                    cfg = self._hand_cfg(sub_board, s0, s1, st, oop, ip, eff, cfg, adjust=lambda c: dataclasses.replace(
                        c, **(self.prior_model or {}), **({"root_allin": False} if c.root_allin else {})))
                    if getattr(self, "_quick", False) and \
                            self._cache.get(self._solve_key(sub_board, s0, s1, st.pot_start, eff, cfg=cfg)) is None:
                        # A quick answer has no time to solve an earlier street (0.3 s: the 10 iterations a solve
                        # runs at least took 2-4 s natively on a limped flop, more in the browser, and ended 20-90% of
                        # the pot from equilibrium, which the narrowing then trusts 0%: 88% of flop narrowing solves at
                        # half a second). The betting rules narrow it, as they would have; one solved already (your
                        # advice on that street) is still walked.
                        rs = {oop: r0, ip: r1}
                        _narrow_street(rs, sub_board, st, stk, hero, hole, slow_rules, check_whole=True)
                        notes.append(f"{st.name}: ranges narrowed by simple betting rules (a quick answer: the street "
                                     f"isn't solved)")
                        r0 = _post_floor(r0, rs[oop], nxt_board, oop, st.name, notes)
                        r1 = _post_floor(r1, rs[ip], nxt_board, ip, st.name, notes)
                        continue
                    sol = self._solve_cached(sub_board, s0, s1, st.pot_start, eff, self.prior_budget, cfg=cfg,
                                             seed=self._store_seed(walked.get("flop"), sub_board)
                                             if st.name == "turn" else None)
                    expl = self._expl(sol)
                    trust = trust_of(expl, st.name)
                    if trust < 1.0:
                        notes.append(f"{st.name}: the solve that narrows ranges is {expl:.1f}% of the pot from "
                                     f"equilibrium ({sol.iters} iterations), so ranges follow it {trust * 100:.0f}% and "
                                     "simple betting rules the rest")
                    taken = []
                    node, c0, c1 = self._walk(sol, st.actions, oop, ip, st.pot_start, eff, notes, st.name, self.prior_smoothing,
                                              board=sub_board, rules=(0, 1),
                                              lock=mk_lock(sol), trust=trust, slow=slow_hu, taken=taken)
                    if s0 is not r0 or s1 is not r1:   # (and out again for the next street: _with_hero puts it back)
                        (c0 if hero_p == 0 else c1)[sol.ctx.active == COMBO_INDEX[(min(hole), max(hole))]] = 0.0
                walked[st.name] = (sol, node, taken)
                r0 = _post_floor(r0, expand(sol.ctx, c0), nxt_board, oop, st.name, notes)
                r1 = _post_floor(r1, expand(sol.ctx, c1), nxt_board, ip, st.name, notes)
            eff = min(stk[oop] - cur.inv_start[oop], stk[ip] - cur.inv_start[ip])
            cfg = self._cfg_with_real_sizes(cur, oop, ip, eff)
            r0, r1 = self._with_hero(r0, r1, hero_p, hole, notes)
            if optimize and to_call <= 1e-9:
                cfg, opt = self._optimize_sizes(board, r0, r1, cur, eff, cfg, hero_p, notes)
            if lib is not None and cur.name == "flop":
                # a precomputed solve answers when it holds the spot and the line, the villain's line isn't rare in
                # it and the node is one it stores (else the live solve below)
                w2, rl2, n2, tk = [], [], [], []
                hit = self._from_library(lib, full, board, r0, r1, cur, eff, cfg, oop, ip, n2,
                                         smoothing=self.prior_smoothing, only=1 - hero_p, widened=w2, board=board,
                                         rare_lines=rl2, taken=tk)
                if hit is not None and self._presolving:
                    return {"skipped": "a flop-library spot"}   # (its answers are read off the entry, not solved)
                if self._runout_close is not None:
                    # the turn report: the entry's stored turns where they hold the line, else a live flop solve
                    rep = None if hit is None else \
                        self._store_report(hit[0], hit[1], hit[2], hit[3], tk, board, (oop, ip), notes + n2, t0)
                    if rep is not None:
                        return rep
                    hit = None
                if hit is not None and not w2 and not rl2 and id(hit[1]) in hit[0].stored:
                    lib_sol, node, n_r0, n_r1 = hit
                    notes.extend(n2)
                    notes.append(lib_sol.note())
        if self._instant:
            if lib_sol is not None or not hu_from_start:
                raise _NoInstant()   # (a library flop is instant anyway; a pot heads-up from this street on solves first)
            near = None
            if lib is not None and cur.name == "flop":
                from .flop_approx import approx_solution
                near = lambda: approx_solution(lib, full[0], full[1], board, r0, r1, cur.pot_start, eff, cfg)
            return self._instant_answer(cur, board, hole, hero, villain, hero_p, oop, ip, eff, cfg, r0, r1, walked,
                                        hero_remaining, to_call, len(positions), t0, near)
        if lib_sol is None and self._lib_only:
            if lib is not None and cur.name == "flop" and hu_from_start:
                # (a library spot's flop not stored: the stored flops its approximate answer reads, named for the
                # browser's prefetch: instant_from_near)
                from .flop_approx import approx_solution
                approx_solution(lib, full[0], full[1], board, r0, r1, cur.pot_start, eff, cfg, fetch_only=True)
            return {"skipped": "not a flop-library spot"}
        if lib_sol is not None:
            sol = lib_sol
        elif hu_from_start:
            # spare_prior: the street advised on also gets what the narrowing solves didn't spend (a flop from the
            # library or from this hand's flop advice costs nothing), so a request never takes longer than cold
            spare = max(0.0, spare - (time.time() - t_prior)) if self.spare_prior else 0.0
            sol = self._solve_cached(board, r0, r1, cur.pot_start, eff, self.time_budget + spare, cfg=cfg,
                                     seed=self._store_seed(walked.get("flop"), board) if cur.name == "turn" else None)
            if actor and actor[1] is None:   # (a presolve: the street's solve is what it came for)
                return {"presolved": cur.name, "iterations": int(sol.iters)}
            self._trim_note(sol, cur.name, notes)
            # The villain's actions this street are exact unless the solve rarely takes them: then
            # smooth as on earlier streets, so we aren't advised against a sliver of a range.
            walk = lambda: self._walk(sol, cur.actions, oop, ip, cur.pot_start, eff, list(notes), cur.name,
                                      self.prior_smoothing, only=1 - hero_p, widened=[], board=board,
                                      lock=mk_lock(sol), rare_lines=[], slow=dict(slow_hu))
            node, n_r0, n_r1 = self._walk(sol, cur.actions, oop, ip, cur.pot_start, eff, notes, cur.name,
                                          self.prior_smoothing, only=1 - hero_p, widened=widened, board=board,
                                          lock=mk_lock(sol), rare_lines=rare_lines, slow=dict(slow_hu))
            if not rare_lines and self._settle(sol):
                # (far from settled: one more budget, then walk the line again; when the line is rare the
                # solve from the node below gets it instead, so a spot doesn't pay for both)
                node, n_r0, n_r1 = walk()
            if rare_lines and isinstance(node, Decision) and node.player == hero_p:
                # Their line was rare in the solve, so everything below it barely converged: answered from
                # there, a min-reraise on KQT9 read by the rules had AQ jamming for +31.8 bb. Solve again
                # from here, against the range they showed.
                bets_now = (min(cur.bets[oop], eff), min(cur.bets[ip], eff))
                sol = self._solve_cached(board, expand(sol.ctx, n_r0), expand(sol.ctx, n_r1), cur.pot_start, eff,
                                         self.time_budget, bets=bets_now, to_act=hero_p,
                                         n_raises=cur.n_raises, settle=True, cfg=self._icm_cfg(self.cfg, cur, oop, ip, bets_now))
                self._trim_note(sol, cur.name, notes)
                node, n_r0, n_r1 = sol.root, sol.r0.copy(), sol.r1.copy()
                notes.append(f"{cur.name}: {', '.join(rare_lines)}'s line is rare in the solve, so solved again from here"
                             + (" against their rule-based range" if widened else ""))
        elif actor:
            return {"error": "the pot became heads-up this street"}   # a solve from the actor's own node
        else:
            # pot became heads-up during this street: solve from the current node
            eff = min(stk[oop] - cur.inv_start[oop], stk[ip] - cur.inv_start[ip])
            r0, r1 = self._with_hero(r0, r1, hero_p, hole, notes)
            bets_now = (min(cur.bets[oop], eff), min(cur.bets[ip], eff))
            sol = self._solve_cached(board, r0, r1, cur.pot_start, eff, self.time_budget,
                                     bets=bets_now, to_act=hero_p, n_raises=cur.n_raises,
                                     settle=True, cfg=self._icm_cfg(self.cfg, cur, oop, ip, bets_now))
            self._trim_note(sol, cur.name, notes)
            node, n_r0, n_r1 = sol.root, sol.r0.copy(), sol.r1.copy()
            notes.append(f"{cur.name}: pot became heads-up this street; solved from here")
        if not isinstance(node, Decision):
            return {"error": "no decision pending at this point in the tree", "notes": notes}
        if actor:
            return self._actor_move(sol, node, n_r0, n_r1, actor[1], 1 - hero_p, cur.name)
        if self._one_size is not None:
            return self._one_size_view(sol, node, n_r0, n_r1, cur, board, hole, hero, villain, hero_p, oop, ip, eff,
                                       pool or bool(adj), notes, t0)
        if node.player != hero_p:
            notes.append("warning: tree expected the other player to act; reporting that node")
        expl = max(0.0, self._expl(sol))   # how far this street's solve is from equilibrium, % of its pot
        if self._runout_close is not None:
            return self._runout_report(sol, node, n_r0, n_r1, hero, villain, hero_p, board, expl, notes, t0)
        exploit = pool or bool(adj)
        if exploit:
            # the best response to the pool (their equilibrium with its leaks) or to your locks, on later streets
            # too: each hand takes its highest-EV action
            nxt = [c for c in node.children if isinstance(c, Decision) and c.player != hero_p]
            lock = mk_lock(sol, nxt)
            evs = sol.solver.node_action_br_evs(node, n_r0, n_r1, lock)
            # each hand takes its highest-EV action, but actions within the solve's noise of it (its
            # distance from equilibrium, at least BR_TIE_MIN of the pot) are tied, and a tied hand plays the
            # equilibrium mix among them: a short solve otherwise picks one at 100% on noise (AQo checked
            # at 6.45 over bets at 6.40 in a solve 3.1% of the pot from settled)
            tol = min(max(BR_TIE_MIN, expl / 100.0), BR_TIE_MAX) * node.pot_now
            tied = evs >= evs.max(axis=0, keepdims=True) - tol
            g = node.avg_strategy0() * tied
            strat = np.where(g.sum(axis=0) > 1e-9, g / np.maximum(g.sum(axis=0), 1e-12), tied / tied.sum(axis=0))
            if pool:
                notes.append(f"exploit mode: {villain} plays like a typical player pool (folds too often to bets of every size, "
                             "most of all small ones, and raises and bluffs less); EVs assume you keep best-responding on later streets")
            if adj:
                from .exploit import describe_lock
                notes.append(f"locked: {villain} {describe_lock(adj)} (on {'the pool' if pool else 'GTO'}); the advice is "
                             "your best response to that, and EVs assume you keep best-responding on later streets")
                notes.extend(_next_lock_notes(node, nxt, lock, n_r1 if hero_p == 0 else n_r0, villain, adj))
        elif node is sol.root and isinstance(sol, Solution) and sol.fresh() and np.array_equal(n_r0, sol.r0) and \
                np.array_equal(n_r1, sol.r1):
            evs = sol.evs[:, sol.ctx.active]    # (the root's, as the solve stands: worked out once, _refresh)
            strat = node.avg_strategy0()
        else:
            evs = sol.solver.node_action_evs(node, n_r0, n_r1)
            strat = node.avg_strategy0()
        if self._study:
            return self._study_node(sol, node, n_r0, n_r1, evs, strat, cur, hero, villain, hero_p, board,
                                    min(to_call, hero_remaining), min(eff - max(node.bets), hero_remaining), expl,
                                    lib_sol, positions, notes, t0)
        idx = COMBO_INDEX[(min(hole), max(hole))]
        pos_in_active = np.nonzero(sol.ctx.active == idx)[0]
        ci = int(pos_in_active[0]) if len(pos_in_active) else None
        acts_out = []
        for a, name in enumerate(node.names):
            f = float(strat[a, ci]) if ci is not None else 1.0 / len(node.names)
            ev = float(evs[a, ci]) if ci is not None else 0.0
            acts_out.append({"action": name, "freq": round(f, 3), "ev_bb": round(ev, 2),
                             "amount_bb": self._amount(name, node, hero_p)})
            if exploit and ci is not None:
                acts_out[-1]["gto_freq"] = round(float(node.avg_strategy0()[a, ci]), 3)
        if ci is None:
            notes.append("hero's hand was not solved; frequencies are uniform")
        best = max(acts_out, key=lambda a: a["freq"])["action"]
        sampled = self._sample(acts_out)
        if widened and ci is not None and not exploit and node is not sol.root:
            # (only when the spot wasn't solved again: our decision came before the rare line) The solve's
            # mix assumes the villain almost never plays this way; against the range they showed, go with the EVs.
            ev_best = max(acts_out, key=lambda a: a["ev_bb"])["action"]
            if ev_best != best:
                notes.append(f"recommending {ev_best}, the highest EV against {', '.join(widened)}'s rule-based range; "
                             f"the solve's mix ({best}) assumes they almost never play this way")
            best = sampled = ev_best
        if exploit and expl >= UNSETTLED:
            notes.append(f"the solve is {expl:.1f}% of the pot from equilibrium, so the {'pool' if pool else 'locked'} model built on it "
                         "is rough: Accurate mode gives steadier exploits")
        if not exploit and not widened and ci is not None and expl >= UNSETTLED:
            # Mixed actions have equal EVs only at equilibrium; short of it the mix lags behind, so
            # a clear EV gap says where the solve is heading.
            top = max(acts_out, key=lambda a: a["ev_bb"])
            mix_ev = next(a["ev_bb"] for a in acts_out if a["action"] == best)
            if top["action"] != best and top["ev_bb"] - mix_ev > EV_GAP / 100 * node.pot_now:
                notes.append(f"recommending {top['action']}, the highest EV: the solve is {expl:.1f}% of the pot "
                             f"from equilibrium, so its mix ({best} most often) hasn't caught up")
                best = sampled = top["action"]
        hero_r = expand(sol.ctx, n_r0 if hero_p == 0 else n_r1)
        vill_r = expand(sol.ctx, n_r1 if hero_p == 0 else n_r0)
        if to_call > 1e-9 and (bn := _beats_note(hole, board, villain, vill_r)):
            notes.append(bn)
        w = hero_r / max(hero_r.sum(), 1e-12)
        sf = np.zeros((len(node.names), N_COMBOS)); sf[:, sol.ctx.active] = strat
        range_freq = {name: round(float((sf[a] * w).sum()), 3) for a, name in enumerate(node.names)}
        if self._breakdown:   # the page's Breakdown view (gto/breakdown.py): asked for, on a cached solve
            from . import breakdown as BD
            ev_full = np.zeros(N_COMBOS); ev_full[sol.ctx.active] = (strat * evs).sum(axis=0)
            opt = {**opt, "breakdown": BD.breakdown(board, node.names, hero, hero_r, sf, ev_full, {villain: vill_r},
                                                    node.pot_now, villain_ev=not self.cfg.rake_rate and self._icm is None, hole=hole)}
        return {
            "street": cur.name, "hero": hero, "villain": villain, "position": "OOP" if hero_p == 0 else "IP",
            "players": len(positions), "hand": hand_class(hole), "board": cards_str(board),
            "pot_bb": round(node.pot_now, 2), "to_call_bb": round(min(to_call, hero_remaining), 2),
            "stack_bb": round(min(eff - max(node.bets), hero_remaining), 2),
            "actions": acts_out, "recommended": best, "sampled": sampled,
            "grid": strategy_grid([sf[a] for a in range(len(node.names))], list(node.names), hero_r),
            "range_action_freq": range_freq,
            "hero_range_combos": round(float(hero_r.sum()), 1),
            "villain_range_combos": round(float(vill_r.sum()), 1),
            "villain_range": range_to_text(vill_r, 0.05)[:400],
            # the same range by hand class for the page's grid (the text above is cut short)
            "villain_grid": strategy_grid([np.ones(N_COMBOS)], ["in range"], vill_r),
            **opt,
            "solver": {"iterations": sol.iters, "solve_seconds": round(sol.seconds, 2), "nodes": sol.n_nodes,
                       "active_combos": sol.ctx.n, "tree_actions": node.names,
                       # how far this street's solve is from equilibrium, % of the street's starting pot
                       "exploitability_pct": round(expl, 2),
                       # (the tree's size: solver.TreeBuilder.cells, MAX_CELLS)
                       **({"cells": sol.cells} if getattr(sol, "cells", None) else {})},
            "widened": widened,
            **({"precomputed": lib_sol.info()} if lib_sol is not None else {}),
            **({"locked": {villain: adj}} if adj else {}),
            "notes": notes, "seconds": round(time.time() - t0, 2),
        }

    def _study_node(self, sol, node, n_r0, n_r1, evs, strat, cur, hero, villain, hero_p, board, to_call, behind, expl,
                    lib_sol, positions, notes, t0):
        """Study mode's answer at a heads-up node (Advisor.study): the actor's whole range, per hand class (the
        grid) and combo: its reach (w), the mix (f), each action's EV in bb relative to folding now (ev; evm the
        mix's) and its equity against the opponent's range at the node (eq: all-in, the rest of the board dealt), and
        its made hand and draw (indices into "categories": the Breakdown's, for the page's filter)."""
        from .breakdown import range_equities
        me = expand(sol.ctx, n_r0 if hero_p == 0 else n_r1)
        opp = expand(sol.ctx, n_r1 if hero_p == 0 else n_r0)
        names = list(node.names)
        A = len(names)
        sf = np.zeros((A, N_COMBOS)); sf[:, sol.ctx.active] = strat
        ev = np.zeros((A, N_COMBOS)); ev[:, sol.ctx.active] = evs
        eq, eq_opp = range_equities(board, me, opp)
        part, (og,) = _study_range(board, names, [self._amount(n, node, hero_p) for n in names], me, sf, ev, eq,
                                   [(villain, opp, eq_opp)])
        res = {
            "study": "node", "street": cur.name, "actor": hero, "opponent": villain,
            "position": "OOP" if hero_p == 0 else "IP", "players": len(positions), "board": cards_str(board),
            "pot_bb": round(node.pot_now, 2), "to_call_bb": round(to_call, 2), "stack_bb": round(behind, 2),
            **part, "opponent_combos": og["combos"], "opponent_eq": og["eq"], "opponent_grid": og["grid"],
            "solver": {"iterations": sol.iters, "solve_seconds": round(sol.seconds, 2), "nodes": sol.n_nodes,
                       "active_combos": sol.ctx.n, "exploitability_pct": round(expl, 2)},
            **({"precomputed": lib_sol.info()} if lib_sol is not None else {}),
            # (the page plays the tree's own sizes, rounded: their mapping back onto it isn't news)
            "notes": [n for n in notes if "to tree action" not in n], "seconds": round(time.time() - t0, 2),
        }
        if self._breakdown:
            from . import breakdown as BD
            res["breakdown"] = BD.breakdown(board, names, hero, me, sf, (sf * ev).sum(axis=0), {villain: opp},
                                            node.pot_now, villain_ev=not self.cfg.rake_rate and self._icm is None)
        return res

    def _study_mw_node(self, sol, node, R, players, hp, shown, strat, evs, to_call, behind, board, cur, positions,
                       solver, notes, t0, precomputed=None):
        """Study mode's answer at a 3-way node (Advisor.study), as _study_node's, from the 3-player solve of the street
        from this node (Advisor._multiway_solve): the actor's range per class and combo, its mix and each action's EV
        (bb relative to folding now), and each opponent's range. Equity against the other two ranges on the river only
        (an earlier street's solve deals the cards to come into equity leaves or buckets: no showdown to read it off).
        `node`: the solve's root, or a node the 3-way library stores (R: the players' reach there)."""
        names = list(shown)
        A = len(names)
        rng = [sol.expand(R[i][0]) for i in range(len(players))]
        sf = np.zeros((A, N_COMBOS)); sf[:, sol.hands.active] = strat
        ev = np.zeros((A, N_COMBOS)); ev[:, sol.hands.active] = evs

        def eq_of(i):
            e = np.full(N_COMBOS, np.nan)
            if len(board) == 5:
                e[sol.hands.active] = sol.showdown_shares(i)
            return e
        amounts = [None if x is None else round(float(x), 2) for x in node.amounts]
        opps = [i for i in range(len(players)) if i != hp]
        part, og = _study_range(board, names, amounts, rng[hp], sf, ev, eq_of(hp),
                                [(players[i], rng[i], eq_of(i)) for i in opps])
        hero = players[hp]
        res = {
            "study": "node", "street": cur.name, "actor": hero, "multiway": True,
            "opponent": " and ".join(players[i] for i in opps), "opponents": og,
            "position": f"{hp + 1} of {len(players)} to act", "players": len(positions), "board": cards_str(board),
            "pot_bb": round(node.pot_now, 2), "to_call_bb": round(to_call, 2), "stack_bb": round(behind, 2),
            **part, "solver": solver, "precomputed": precomputed, "notes": notes, "seconds": round(time.time() - t0, 2),
        }
        if self._breakdown:   # (no equity view 3-way, as the advice's)
            from . import breakdown as BD
            res["breakdown"] = BD.breakdown(board, names, hero, rng[hp], sf, (sf * ev).sum(axis=0),
                                            {players[i]: rng[i] for i in opps}, node.pot_now, equity=False)
        return res

    # ------------------------------------------------------------ multiway
    def _multiway_solve(self, hero, hole, board, ranges, remaining, cur: StreetState, stk, t0, notes, line=None,
                        positions=None):
        """Three players in: a 3-player CFR solve of this street from the current node (gto/multiway_solver.py),
        the later streets bet on by hand buckets (mw_street_config). None when it doesn't
        apply (4+ players, someone all-in, the street not in mw_solve_streets, or a bet bigger than the shortest
        stack can call) or the solve is still over mw_max_expl from equilibrium after its budget (a note says so):
        the betting rules (_multiway) answer instead."""
        from . import multiway_solver as MW
        fallback_notes, notes = notes, list(notes)
        exact3 = _mw_applies(remaining, cur, stk)
        four = not exact3 and self.mw4_street_config is not None and _mw4_applies(remaining, cur, stk, hero)
        nway = not exact3 and not four and _nway_applies(remaining, cur, stk, hero, self.mw_nway_max)
        # (the N-player solve off the river is the hand of the only player with chips: nothing to solve but its call)
        if not self.mw_solve_streets or not ((exact3 or four) and cur.name in self.mw_solve_streets
                                             or nway and (cur.name != "river" or "river" in self.mw_solve_streets)):
            return None
        # (quick: rivers and turns (MW_QUICK_TURN), and a pot with a player all-in on any street: two bettors, solved in
        # the browser in 1.4-1.6 s on a flop or turn, 100 iterations, 0.05% of the pot from equilibrium)
        quick = getattr(self, "_quick", False)
        if four and quick and cur.name == "flop" and self.mw4_quick_flop is None:
            fallback_notes.append("a quick answer leaves a 4-way flop to the betting rules (Fast and Accurate solve it)")
            return None
        players = [p for p in cur.order if p in remaining]          # postflop acting order
        behind = {p: stk[p] - cur.inv_start[p] for p in players}
        # a player all-in (at most one, _mw_applies) doesn't act: the others bet for the side pot; the main pot
        # (everyone's chips up to the all-in's) goes to the best of the three
        shoved = [p for p in players if p in cur.allin]
        S = min(v for p, v in behind.items() if p not in shoved)
        # 3-way: each player bets its own stack (the deepest only what another can match); a player all-in for less
        # contests the main pot, the others the side pot (MWConfig.inv0). Four or more: everyone the shortest stack
        uneven = max(v for p, v in behind.items() if p not in shoved) - S > 1e-6
        stacks = tuple(float(behind[p]) for p in players) if exact3 and uneven else S
        # your raise as the solve plays it goes to at most `cap` (everyone the shortest stack, four or more; three, what
        # another can match). Under the smallest raise you can make, with chips for it, that's no move the page takes (a
        # raise to 77.5 where the smallest is to 144, an all-in to 77.5 of a 5bb stack's cap): the betting rules, which
        # play each seat's own stack, answer
        cap = min(behind[hero], max(behind[q] for q in players if q != hero)) if isinstance(stacks, tuple) else S
        smallest = cur.cur_bet + cur.min_raise
        if hero not in shoved and behind[hero] > cap + 1e-9 and cur.cur_bet + 1e-9 < cap \
                and np.floor(cap * 2 + 0.5) / 2 < smallest - 1e-9 and can_raise(cur, hero):
            fallback_notes.append(f"the smallest raise you can make (to {smallest:g}) is more than the {len(players)}-way "
                                  f"solve lets you put in ({cap:g}: " + ("what another player can match" if exact3 else
                                                                        "everyone plays the shortest stack behind")
                                  + "), so the betting rules answer")
            return None
        # (an all-in bet the others can't match in full is played at what they can: the rest comes back; N players at
        # the shortest stack)
        bets = [min(cur.bets[p], S if nway or four else max(behind[q] for q in players if q != p)) for p in players]
        dead_bets = sum(b for q, b in cur.bets.items() if q not in remaining)
        pot0 = cur.pot_start + dead_bets
        hp = players.index(hero)
        hi = COMBO_INDEX[(min(hole), max(hole))] if hole else None     # (Study: no cards of the actor's own)
        rs = [ranges.get(p, np.ones(N_COMBOS)) for p in players]
        if hi is not None and rs[hp][hi] <= 1e-9:
            notes.append("your hand isn't in your range for this line: its mix is the solve's current (regret-matching) "
                         "strategy for it, against the ranges as they are")
        pending = {i for i, p in enumerate(players) if p not in cur.acted}
        pending.add(hp)
        n_act = len(players) - len(shoved)
        if four:
            # 4-way flops and turns: the later streets by hand buckets (MW4_STREETS), one bet a street, no raises but
            # yours; card removal pairwise on this street, as the N-player solve's
            mcfg = MW.MWConfig(rake_rate=self.cfg.rake_rate, rake_cap=self.cfg.rake_cap, bet=MW4_BETS, max_raises=1,
                               root_raise=True, **(self.mw4_quick_flop if quick and cur.name == "flop" else
                                                   self.mw4_street_config.get(cur.name, {})))
        elif nway:
            # Four or more in (or three, two of them all-in): pairwise card removal (the others' hands independent
            # given yours, their overlap corrected), the pots layered by the all-in players' stakes, a bet of half the
            # pot; with four or more betting one bet a street and no raises but yours (MWConfig.root_raise: a raise
            # there made the 4-way tree 1053 nodes, unsettled; without, 65 and 0.02% of the pot in 150 iterations)
            mcfg = MW.MWConfig(rake_rate=self.cfg.rake_rate, rake_cap=self.cfg.rake_cap, later="deal", depth=1,
                               bet={"flop": 0.33, "turn": 0.5, "river": 0.5}, max_raises=2 if n_act <= 3 else 1,
                               root_raise=n_act >= 4, allin=tuple(players.index(a) for a in shoved),
                               inv0=tuple(float(cur.inv_start[p]) for p in players),
                               dead=tuple(float(cur.inv_start[q] + cur.bets[q]) for q in cur.inv_start if q not in remaining))
            for a in shoved:
                pending.discard(players.index(a))
        else:
            street_cfg = MW_QUICK_TURN if quick and cur.name == "turn" else MW_QUICK_FLOP if quick and cur.name == "flop" \
                and not shoved else self.mw_street_config.get(cur.name, {})
            mcfg = MW.MWConfig(rake_rate=self.cfg.rake_rate, rake_cap=self.cfg.rake_cap,
                               **{**street_cfg, **(self.mw_config or {})})
        if getattr(self, "_jam", False):
            mcfg.root_jam = True   # (a jam request: all-in offered however deep, at your decision: the solve's root)
        model = getattr(self, "_icm", None)
        if model is not None:
            # a tournament: every outcome paid in prize equity (icm.MWPayoffs), each seat's chips in at the root
            # (the folded seats' dead, this street's bets with them) plus the solve's
            from .icm import MWPayoffs
            contrib = {q: float(cur.inv_start.get(q, 0.0)) + (0.0 if q in remaining else float(cur.bets.get(q, 0.0)))
                       for q in model.seats}
            mcfg.icm = MWPayoffs(model, players, contrib)
            mcfg.rake_rate = 0.0
        if (exact3 or four) and (shoved or isinstance(stacks, tuple)):
            # (the pots layered by each seat's chips in: before this street, inv0, and the folded seats', dead)
            mcfg.inv0 = tuple(float(cur.inv_start[p]) for p in players)
            mcfg.dead = tuple(float(cur.inv_start[q] + cur.bets[q]) for q in cur.inv_start if q not in remaining)
        if shoved and not nway:
            a = shoved[0]
            cap = cur.inv_start[a] + cur.bets[a]
            mcfg.allin = (players.index(a),)
            # (the main pot once the others match the all-in)
            main = cap * len(remaining) + sum(min(cur.inv_start[q] + cur.bets[q], cap) for q in cur.inv_start
                                              if q not in remaining)
            pending.discard(players.index(a))
            rest = [q for q in players if q not in (hero, a)]
            notes.append(f"{a} is all-in: the main pot ({main:.1f} bb) goes to the best of the {['', '', 'two', 'three', 'four'][len(players)]} hands, "
                         f"the side pot to the " + (f"better of yours and {rest[0]}'s" if len(rest) == 1 else
                                                   f"best of yours, {', '.join(rest[:-1])}'s and {rest[-1]}'s"))
        inc = () if hi is None or any(r[hi] > 0 for r in rs) else (hi,)    # (a hand nobody's range holds still gets solved)
        key = (tuple(board), tuple(_range_key(r) for r in rs), round(pot0, 2),
               tuple(round(x, 2) for x in stacks) if isinstance(stacks, tuple) else round(S, 2), tuple(round(b, 2) for b in bets),
               hp, cur.n_raises, tuple(sorted(pending)), repr(mcfg), inc)
        # the budget covers the whole answer: the tree and bucket tables (a flop's take 1-2 s natively, a few in the
        # browser), the iterations, and the gains and EVs read off after them (MW_RESERVE iterations' time)
        # (a solve its budget ran out on isn't run again: a later request for the spot gets the same answer)
        budget = max(self.time_budget, MW_QUICK_BUDGET if getattr(self, "_quick", False) else self.mw_budget or 0.0)
        if self._study and cur.name == "flop":
            budget = max(budget, MW_STUDY_FLOP_BUDGET)
        sol = self._mw_cache.get(key)
        if sol is None:
            # a quick answer's tables still building at MW_QUICK_BUILD of its budget (or on course to) leave too little
            # time to iterate: the betting rules answer then (a 3-way flop with a player all-in and a random range took
            # 8.8 s in the browser to run no iteration, a 4-way limped flop 7.8 s natively to run 7, 22% of the pot
            # from equilibrium)
            try:
                sol = MW.MultiwaySolver(board, rs, pot0, stacks, bets=bets, to_act=hp, n_raises=cur.n_raises,
                                        pending=pending, cfg=mcfg, include=inc,
                                        deadline=t0 + MW_QUICK_BUILD * budget if quick and not self._study else None)
            except MW.BuildTooSlow:
                fallback_notes.append(f"the {len(players)}-way solve's tables take longer than a quick answer's "
                                      f"{budget:.1f} s here, so the betting rules answer (Fast and Accurate solve it)")
                return None
            self._mw_cache[key] = sol
        spent = sol.build_seconds + sol.seconds
        left = min(budget - spent, budget - (time.time() - t0))
        self._mw_run(sol, left)
        self._keep("_mw_cache", key, sol)
        root = sol.root
        if not isinstance(root, MW.Decision) or root.player != hp:
            return None
        S_h = sol.stacks[hp]          # (your chips behind as the solve plays them: at most what another can match)
        if getattr(self, "_jam", False) and "all-in" not in _mw_names(root.names, root.amounts, behind[hero], S_h,
                                                                      root.pot_now):
            # a jam request the solve can't price (four or more play everyone at the shortest stack: a 0.6bb one left
            # no jam against the deep player): the betting rules, which play each seat's own stack, price it
            fallback_notes.append(f"the {len(players)}-way solve plays everyone at the shortest stack, so the betting "
                                  f"rules price your all-in")
            return None
        gains = sol.br_gains() / sol.root_pot() * 100        # each player's best-response gain, % of the pot
        # (the mean over the players who bet: an all-in player has no decisions, nothing to gain)
        expl = float(max(np.mean([g for i, g in enumerate(gains) if i not in mcfg.allin]), 0.0))
        if expl > self.mw_max_expl:
            fallback_notes.append(f"the {len(players)}-way solve was still {expl:.0f}% of the pot from equilibrium after "
                                  f"{sol.iters} iterations" + ("" if self._study else ", so the betting rules answer instead")
                                  # (Fast's 3-way budget: mw_budget; Accurate solves for its whole time_budget)
                                  + (" (Accurate gives it longer)" if self.mw_budget else ""))
            return None
        if line is not None:
            self._mw_lines[line] = (sol, players, expl)   # practice opponents and later streets read it (_mw_walk)
        strat = root.avg_strategy()[0]
        evs = sol.action_evs()
        shown = _mw_names(root.names, root.amounts, behind[hero], S_h, root.pot_now)
        if not self._study:   # (Study: no cards of the actor's own; the whole range's answer, _study_mw_node)
            ci = sol.combo_index(hole)
            mine = sol.hand_strategy(ci, hp)
            acts = []
            for a, name in enumerate(shown):
                amt = root.amounts[a]
                acts.append({"action": name, "freq": round(float(mine[a]), 3),
                             "ev_bb": round(float(evs[a, ci]), 2),
                             "amount_bb": None if amt is None else round(float(amt), 2)})
            best = max(acts, key=lambda x: x["freq"])["action"]
            sampled = self._sample(acts)
            if expl >= UNSETTLED:
                top = max(acts, key=lambda x: x["ev_bb"])
                mix_ev = next(x["ev_bb"] for x in acts if x["action"] == best)
                if top["action"] != best and top["ev_bb"] - mix_ev > EV_GAP / 100 * root.pot_now:
                    notes.append(f"recommending {top['action']}, the highest EV: the solve is {expl:.1f}% of the pot "
                                 f"from equilibrium, so its mix ({best} most often) hasn't caught up")
                    best = sampled = top["action"]
        hero_r = sol.expand(sol.reach0[hp][0])
        w = hero_r / max(hero_r.sum(), 1e-12)
        sf = np.zeros((len(root.names), N_COMBOS)); sf[:, sol.hands.active] = strat
        opps = [p for p in players if p != hero]
        if self._breakdown and not self._study:   # hero's categories and the opponents' (no equity view 3-way)
            from . import breakdown as BD
            ev_full = np.zeros(N_COMBOS); ev_full[sol.hands.active] = (strat * evs).sum(axis=0)
            bd = BD.breakdown(board, shown, hero, hero_r, sf, ev_full,
                              {p: sol.expand(sol.reach0[i][0]) for i, p in enumerate(players) if p != hero},
                              root.pot_now, equity=False, hole=hole)
        to_call = max(bets) - bets[hp]
        bettor = next((a["pos"] for a in reversed(cur.actions) if a["action"] in ("bet", "raise", "all-in")), None)
        if hole and to_call > 1e-9 and bettor in players and (bn := _beats_note(hole, board, bettor,
                                                                        sol.expand(sol.reach0[players.index(bettor)][0]))):
            notes.append(bn)
        street_note = {"river": "the river's showdowns with every card removal between the three hands counted exactly",
                       "turn": "the river dealt out after the turn's betting (an equity leaf: no river betting), every river card",
                       "flop": "the turn and river dealt out after the flop's betting (equity leaves: no later betting), every turn"}
        if mcfg.later == "buckets":
            street_note["turn"] = (f"the river bet on too, by {mcfg.buckets} hand buckets a player on each river card "
                                   f"(one card per rank class)")
            street_note["flop"] = (f"the turn and river bet on too, by {mcfg.buckets} hand buckets a player on one turn card "
                                   f"per rank class, the river played on the turn's boards without its card")
        sizes = MW.bet_sizes(mcfg, cur.name)
        if four:
            street_note[cur.name] += ("; card removal between the hands taken pair by pair (each opponent's hand "
                                      "independent of the others' given yours, their overlap corrected)")
        if nway:
            street_note = {s_: "the cards to come dealt out: nobody else has chips to bet" for s_ in ("flop", "turn")}
            street_note["river"] = ("card removal between the hands taken pair by pair (each opponent's hand independent "
                                    "of the others' given yours, their overlap corrected): within ~1% of the pot of the "
                                    "exact count, 3-way")
            if shoved:
                # (the pots as they stand once you match the bet)
                c_call = list(sol.root.c); c_call[hp] = max(min(max(c_call), S), c_call[hp])
                lv = sol._layers(tuple(range(len(players))), c_call, sol.root_pot() + c_call[hp] - sol.root.c[hp])
                notes.append(f"{', '.join(shoved)} all-in: " + "; ".join(
                    f"{'the main pot' if i == 0 else 'a side pot'} ({amt:.1f} bb) to the best of "
                    + ", ".join(players[j] for j in el) for i, (amt, el) in enumerate(lv)))
        raises = ("no raises but yours" if mcfg.root_raise else "one raise") if mcfg.max_raises <= 1 or mcfg.root_raise \
            else "one raise"
        notes.append(f"{len(players)}-way pot solved by {len(players)}-player CFR ({sol.iters} iterations): "
                     + (f"one bet size ({sizes[0] * 100:.0f}% pot)" if len(sizes) == 1 else
                        f"bets of {', '.join(f'{x * 100:.0f}' for x in sizes)}% pot")
                     + f", {raises} and all-in; {street_note[cur.name]}")
        if (cur.acted or cur.n_raises) and not any(n.startswith(f"{cur.name}: 3-way, ranges narrowed") for n in notes):
            notes.append("this street's earlier actions narrow the ranges by the betting rules; the solve starts from here")
        if isinstance(stacks, tuple):
            notes.append("stacks differ: each player bets their own stack (" + ", ".join(
                f"{p} {x:.1f}" for p, x in zip(players, sol.stacks)) + " bb behind, as far as another can match it); "
                "a player all-in for less plays for the main pot, the others for the side pot too")
        elif max(behind.values()) - S > 0.05 * S:
            notes.append(f"stacks differ: everyone plays the shortest stack behind ({S:.1f} bb)")
        notes.append(f"best-response gains (% pot): " + ", ".join(f"{p} {g:.2f}" for p, g in zip(players, gains)))
        if mcfg.icm is not None:
            notes.append(f"tournament: the {len(players)}-way solve pays every outcome in prize equity (ICM)"
                         + (" and bounties (each bust's to whoever wins the pot that takes the player's last chips)"
                            if getattr(model, "bounty", None) is not None else "")
                         + ", EVs in chip equivalents; who wins when you lose is taken as even among the others")
        if self._study:
            return self._study_mw_node(sol, root, sol.reach0, players, hp, shown, strat, evs, to_call, S_h - bets[hp], board,
                                       cur, positions or [], {
                                           "iterations": sol.iters, "solve_seconds": round(sol.seconds, 2),
                                           "nodes": sol.builder.n_nodes, "active_combos": sol.hands.n,
                                           "exploitability_pct": round(expl, 2),
                                           "best_response_gains_pct": {p: round(float(g), 2) for p, g in zip(players, gains)}},
                                       notes, t0)
        out = {"street": cur.name, "hero": hero, "hand": hand_class(hole), "board": cards_str(board),
               "players": len(players), "multiway": True, "solved": True, "opponents": opps,
               "position": f"{hp + 1} of {len(players)} to act",
               "pot_bb": round(root.pot_now, 2), "to_call_bb": round(to_call, 2), "stack_bb": round(S_h - bets[hp], 2),
               "actions": acts, "recommended": best, "sampled": sampled,
               "grid": strategy_grid([sf[a] for a in range(len(root.names))], shown, hero_r),
               "range_action_freq": {name: round(float((sf[a] * w).sum()), 3) for a, name in enumerate(shown)},
               "hero_range_combos": round(float(hero_r.sum()), 1),
               "opponent_range_combos": {p: round(float(sol.reach0[i][0].sum()), 1) for i, p in enumerate(players) if p != hero},
               "solver": {"iterations": sol.iters, "solve_seconds": round(sol.seconds, 2), "nodes": sol.builder.n_nodes,
                          "active_combos": sol.hands.n, "tree_actions": shown,
                          # the mean of the three players' best-response gains, % of the pot (heads-up: the same mean of two)
                          "exploitability_pct": round(expl, 2),
                          "best_response_gains_pct": {p: round(float(g), 2) for p, g in zip(players, gains)}},
               "notes": notes, "seconds": round(time.time() - t0, 2)}
        if len(board) == 5:
            out["equity_vs_field"] = round(sol.showdown_share(ci), 3)
        if self._breakdown:
            out["breakdown"] = bd
        return out

    def _multiway(self, hero, hole, board, ranges, remaining, cur: StreetState, stack, t0, notes):
        """3+ players still in: Monte Carlo vs every live range, not a solve. Frequencies are 0/1 (the
        highest EV action); EVs are relative to folding now.

        Opponents follow simple range rules, with hands ranked by average strength over runouts:
        - Facing a bet or raise of size b into pot p, the n players who can continue fold together
          with probability FOLD_SHADE * b / (p + b): a quarter less often than a balanced defence
          (which would let a pure bluff break even), since real players call too much. They all
          keep the hands above one strength bar, so a range of weak hands (a lead into the field
          after checking twice) folds more of itself than a strong one (a call of that lead), and
          the strongest part of what each continues with raises instead. Players still to act behind
          a bet we call faced that bet with us, so they fold as often as each would with everyone it
          was made into facing it together (not as if it were a fresh bet into the pot our call grew).
        - Betting (_bet_ranges): mostly SMALL_BET (33%), a merged range: value from the top SMALL_SHARE
          of what the players it's made into would call with (compounding with more players; their
          actual ranges, balanced and unshaded, see _value_share) plus bluffs at the balanced ratio,
          bluffs = value * b / (p + b). 66% takes the top of that value (above the top VALUE_SHARE of
          what would call it) in proportion to the bettor's nut advantage (_nut_weight), with its own
          balanced bluffs. Fitted to GTO Wizard's 3-way flop study (the opener checks 34%, bets a
          third 56%, bigger 10%; heads-up 23/38/39); the old rule (66% the top, 33% a thin layer)
          checked 67% 3-way. Bluffs are the hands without showdown value that most often end up strong
          (draws and overcards first; see _bluff_order), on the river the weakest hands. A check rules
          out the big bet, the small bet's bluffs and its strong value; the merged middle checks too.
          A player who checked behind us before the bettor holds that check range.
        - Raises go to 3x the bet plus the calls in front (_raise_to); our raise option facing a bet
          is the page's default size, the bet plus 75% of the pot after calling.
        - We choose between checking and betting SMALL_BET (33%) or 66% of the pot (facing a bet:
          folding, calling, or raising) by EV. Each size has its own continuing ranges by the rule
          above, so a small bet keeps more of the field in (3-way, each keeps 57% against 33% and
          45% against 66%) and gets raised more often.
        - After we check, the first opponent behind us whose betting range at either size holds their
          hand bets it. The others face that bet like players behind a call: one strength bar, each
          folding as often as with everyone it was made into facing it together, the strongest part
          raising. Players between the bettor and us act first, and we fold, call or check-raise
          knowing what they did (for each bettor, size and what the players between did, whichever
          pays over the trials, and a call our equity against the bettor and callers pays for by
          itself, see priced); players after us act once we've called, and we call their raise by
          equity against it. Our check-raise is faced by everyone still in (the bettor, the callers
          between, the players behind) with the same joint fold rule; the strongest part of what
          continues re-raises and we call that by equity.
        - Once someone raises on this street, the players not involved (the first bettor and earlier
          callers too) fold, except against our check-raise.
        Every line plays on to the river by the same rules (later_street): players act in seat order,
        an opponent bets 66% or 33% by the betting ranges above, we lead 66% when we're ahead of every
        range that would call it, else 33% when ahead of what calls that (LEAD_EQ), and everyone
        still in faces a bet: opponents call, raise or fold by the joint rule (against our bet only
        call or fold), we raise (RAISE_EQ,
        against each range that would continue), call or fold by our equity against the bettor and
        the callers on the board as it stands (no peeking at later cards; on the turn, counting the
        river bet that may follow). One raise per later street; everyone still in then calls or
        folds it.
        With rake, every pot hero wins is short min(rate x pot, cap) (the pot without an uncalled bet; a
        split pot costs hero its share), as at the solver's terminals, and hero's calls, leads and raises
        within a trial count it (a call pays when equity x the raked pot beats its price; the lead and
        raise equity bars rise by what the rake takes of the chips they add); the opponents' rules stay
        unraked. Our check-raise after a bet has to beat calling or folding by XR_Z standard errors over
        that situation's trials (see chk_plan)."""
        from .evaluator import eval_board
        cfg = self.cfg
        def won(pot_):
            """What the winner collects of a pot of pot_ bb that saw the flop (pot_ itself without rake)."""
            return pot_ - cfg.rake(pot_)
        # what our in-trial rules (calls, leads, raises) count a pot as paying: raked, unless mw_rake_rules is
        # off (the old rules, which left rake to the payouts alone)
        dwon = won if self.mw_rake_rules else (lambda pot_: pot_)

        def rake_bar(add, pot_):
            """How much higher an equity bar set for an unraked pot has to be when each of two players adds
            `add` to a pot of pot_ (the winner pays rake on the chips added, up to the cap): the extra the
            winner collects is short of 2 x add. 1 without rake (the bars then stay exactly as fitted)."""
            if cfg.rake_rate <= 0 or not self.mw_rake_rules:
                return 1.0
            gain = won(pot_ + 2 * add) - won(pot_)
            return 2 * add / gain if gain > 1e-12 else float("inf")
        rng = random.Random(1)
        opps = [p for p in remaining if p != hero]
        n_opp = len(opps)
        dead = set(board) | set(hole)
        avail = [c for c in range(52) if c not in dead]
        idx = COMBO_INDEX[(min(hole), max(hole))]
        opp_ranges = []
        for p in opps:
            r = ranges.get(p, np.ones(N_COMBOS)).copy()
            for c in dead:
                r *= 1.0 - MEMBER_FULL[c]
            opp_ranges.append(r if r.sum() > 0 else np.ones(N_COMBOS))
        # our range as the opponents see it (what their bets are made into)
        hero_range = ranges.get(hero, np.ones(N_COMBOS)).copy()
        for c in board:
            hero_range *= 1.0 - MEMBER_FULL[c]
        # Stacks: each seat's own (`stack` is {seat: bb}, or one number for everyone). We can put in no more
        # than the deepest opponent still in with chips can match, and chips an opponent has beyond ours
        # never reach us: each opponent is capped at our stack (a bet of theirs past it is cut back to it),
        # so the trials play the pot we can win. An opponent already all-in for less than us only wins the
        # chips up to their stake from each player: see showdown().
        stk = stack if isinstance(stack, dict) else {p: float(stack) for p in cur.inv_start}
        deepest = max(stk[p] for p in opps)
        hero_rem = max(0.0, min(stk[hero], deepest) - cur.inv_start[hero] - cur.bets[hero])
        h_max = cur.inv_start[hero] + cur.bets[hero] + hero_rem          # the most we can have in, in all
        over = {p: max(0.0, cur.inv_start[p] + cur.bets[p] - h_max) for p in opps}
        if any(v > 1e-9 for v in over.values()):
            bets_c = {p: b - over.get(p, 0.0) for p, b in cur.bets.items()}
            cur = dataclasses.replace(cur, bets=bets_c, cur_bet=min(cur.cur_bet, max(bets_c.values())))
        pot = cur.pot_start + sum(cur.bets.values())
        to_call = cur.cur_bet - cur.bets[hero]
        call_amt = min(to_call, hero_rem)
        seq = cur.order[cur.order.index(hero) + 1:] + cur.order[:cur.order.index(hero)]
        after = sorted(range(n_opp), key=lambda i: seq.index(opps[i]))   # opponents in acting order after us
        cap = [min(stk[p], h_max) - cur.inv_start[p] for p in opps]      # most each opponent can have in this street
        # side pots: an opponent all-in for less than we can put in only contests the chips up to their stake
        h0 = cur.inv_start[hero] + cur.bets[hero]
        fixed = {i: cur.inv_start[opps[i]] + cur.bets[opps[i]] for i in range(n_opp) if opps[i] in cur.allin}
        tot_cap = [cur.inv_start[opps[i]] + cap[i] for i in range(n_opp)]
        dead = [cur.inv_start[q] + cur.bets[q] for q in cur.inv_start if q not in cur.active]
        short = [opps[i] for i, c in fixed.items() if c < h0 + hero_rem - 1e-9]
        if short:
            notes.append(f"{', '.join(short)} {'is' if len(short) == 1 else 'are'} all-in for less: "
                         f"side pots, each player wins at most their stake from everyone")
        seat = {i: cur.order.index(opps[i]) for i in range(n_opp)}       # acting order on later streets
        seat["hero"] = cur.order.index(hero)
        accurate = self.time_budget >= 20
        # (the trials take what's left of the request's budget, at least MW_RULES_MIN_S: after a multiway solve that
        # didn't settle, a quick answer's 1.5 s of trials made it 2-4x its estimate)
        rules_left = max(MW_RULES_MIN_S, self.time_budget - (time.time() - t0))

        # Runouts, shared by the strength rankings and the trials: from the flop every turn with a few
        # rivers each, from the turn every river.
        runs = []
        if len(board) == 5:
            runs = [(tuple(board), eval_board(board))]
        elif len(board) == 4:
            runs = [(tuple(board) + (c,), eval_board(list(board) + [c])) for c in avail]
        else:
            for t in avail:
                for c in rng.sample([c for c in avail if c != t], 8 if accurate else 4):
                    runs.append((tuple(board) + (t, c), eval_board(list(board) + [t, c])))

        def strength(sts):
            """(average strength, combos strongest first, the samples, bluff order, (win, live)) over these
            runouts; win and live are (runouts, combos) matrices: our share of the pot against each combo
            (1 win, 0.5 tie, 0 blocked) and whether it is live, so equities are matrix products."""
            tot = np.zeros(N_COMBOS); cnt = np.zeros(N_COMBOS)
            for s in sts:
                ok = s >= 0
                tot[ok] += s[ok]; cnt[ok] += 1
            sc = np.where(cnt > 0, tot / np.maximum(cnt, 1), -1.0)
            rank = np.argsort(-sc)
            return sc, rank, sts, _bluff_order(sts, rank, sc), mats(sts)

        def mats(sts):
            S = np.array(sts)
            h = S[:, idx:idx + 1]
            live = (S >= 0).astype(float)
            return ((S < h) + 0.5 * (S == h)) * live, live

        now = strength([s for _, s in runs])
        turns = {}
        if len(board) == 3:
            by_turn = {}
            for run, s in runs:
                by_turn.setdefault(run[3], []).append(s)
            turns = {t: strength(sts) for t, sts in by_turn.items()}
        rivers = {}
        def info(run, n, st):
            if n == len(board):
                return now
            if n == 4:
                return turns[run[3]]
            if run not in rivers:
                rk = np.argsort(-st)
                rivers[run] = (st, rk, [st], rk[::-1], mats([st]))
            return rivers[run]

        aggro, keep = _aggro, _keep

        def equity(w, mt):
            """Our equity against range w over the runouts of mt (strength()'s matrices), pooled."""
            win, live = mt
            num, den = float((win @ w).sum()), float((live @ w).sum())
            return num / den if den > 0 else 0.5

        def equity_all(ws, mt):
            """Our chance of beating every one of the ranges ws (the product per runout)."""
            if len(ws) == 1:
                return equity(ws[0], mt)
            win, live = mt
            W = np.array(ws).T
            num, den = win @ W, live @ W
            e = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.5)
            return float(e.prod(axis=1).mean()) if len(e) else 0.5

        def calls(w, mt, cost, p):
            """Do we call `cost` more into a pot of p (their chips included) against range w (or a list
            of ranges, all of which we have to beat)? Raked: we win what the final pot pays its winner."""
            return (equity_all(w, mt) if isinstance(w, list) else equity(w, mt)) * dwon(p + cost) > cost

        ranks = {}
        def calls_turn(ws, sts, mt, b, p, left):
            """Turn version of calls(): average over the river samples of calling b, then facing
            the bettor's (ws[0]) river bet (value + balanced bluffs from what they bet the turn with)
            and calling or folding it, so a call that a river bet will push us off isn't counted at
            full equity. Other ranges in ws (callers) only have to be beaten at showdown."""
            P = p + b
            b2 = min(round(0.66 * P * 2) / 2, left - b)
            won_P, won_2 = dwon(P), dwon(P + 2 * b2)
            win_m, live_m = mt
            tot = 0.0
            for k, s in enumerate(sts):
                win, live = win_m[k], live_m[k]
                w = ws[0] * live
                w_sum = w.sum()
                if w_sum <= 0:
                    continue
                f = 1.0
                for o in ws[1:]:
                    d = o @ live
                    f *= (o @ win) / d if d > 0 else 0.5
                w_win = w @ win
                if b2 <= 1e-9:
                    tot += w_win / w_sum * f * won_P
                    continue
                if id(s) not in ranks:
                    ranks[id(s)] = np.argsort(-s)
                m = aggro(w, keep(b2, P, 1, 1.0) * VALUE_SHARE, b2, P, ranks[id(s)])
                wm = w * m
                wm_win = wm @ win
                ev_bet = wm_win * f * won_2 - wm.sum() * b2
                tot += (max(ev_bet, 0.0) + (w_win - wm_win) * f * won_P) / w_sum
            return tot / max(len(sts), 1) > b

        cache = {}
        # what the later streets did over the trials (bets by size, bets that drew more than one call,
        # raises and how many came over a call, so their size counted the calls)
        ls_stats = dict.fromkeys(("bets", "small", "multi_called", "raises", "raises_over_calls", "ours", "our_folds"), 0)

        def later_street(n, run, st, hands, alive, rs, left, p, put):
            """One later street by the rules (see the docstring). Returns (result, pot, our chips in,
            opponents still in, their ranges); result is None unless the hand ended here (we folded:
            -put; everyone else folded: the pot less our chips)."""
            sc, rank, sts, bluff, mt = info(run, n, st)
            # players all-in (no chips left) stay in for the showdown but don't act
            chips = [i for i in alive if left[i] > 1e-9]
            eff = min([left["hero"]] + [left[i] for i in chips])
            big = min(round(0.66 * p * 2) / 2, eff)
            if big <= 1e-9 or not chips:
                return None, p, put, alive, rs
            small = min(round(SMALL_BET * p * 2) / 2, eff)
            small = small if 1e-9 < small < big - 1e-9 else None
            liv = sc >= 0
            # Decisions are cached (key: the cards dealt since and the ranges going in by id, plus the actions
            # so far), so a runout revisited with the same action replays without recomputing the ranges. The
            # "in" entry holds the ranges going in, so no id in a stored key can be reused by a new array; once
            # the cache holds LATER_CACHE entries, new decisions are computed without being stored.
            key0 = (run[len(board):n], tuple(alive), tuple(id(rs[i]) for i in alive), round(p, 2),
                    tuple(round(left[x], 2) for x in ["hero"] + alive))

            def C(path, fn):
                k = key0 + path
                v = cache.get(k)
                if v is None:
                    v = fn()
                    if len(cache) < LATER_CACHE:
                        cache[k] = v
                return v

            # drop combos holding a card dealt since (the ranges going in kept alongside, see above)
            rs = dict(C(("in",), lambda: (tuple((i, rs[i] * liv) for i in alive), tuple(rs[i] for i in alive)))[0])
            alive = list(alive)
            stuck = [i for i in alive if i not in chips]
            actors = sorted(chips + ["hero"], key=seat.get)
            inv = dict.fromkeys(actors, 0.0)

            def put_in(a, to):
                d = max(0.0, min(to - inv[a], left[a]))
                inv[a] += d; left[a] -= d
                return d

            # Checked to the first bet: an opponent bets the size whose range holds their hand; we lead
            # 66% when ahead of every range that would call it, else 33% when ahead of what calls that.
            path, bet = (), None
            for a in actors:
                if a == "hero":
                    def lead_fn(rs=rs, al=tuple(alive)):
                        for b in (big, small):
                            if b is None:
                                continue
                            cs = _joint_top([rs[i] for i in al], _fold_together([b] * len(al), p), rank)
                            bar = LEAD_EQ * rake_bar(b, p)
                            if all(equity(rs[i] * c, mt) > bar for i, c in zip(al, cs)):
                                return (b,)
                        return (None,)
                    (b,) = C(path + ("lead",), lead_fn)
                    if b is not None:
                        bet = ("hero", b)
                        break
                    path += ("x",)
                else:
                    def bet_fn(r=rs[a]):
                        into = [rs[o] for o in alive if o != a] + [hero_range * liv]
                        bg, sm, dr = _bet_ranges(r, len(alive), p, big, small, rank, bluff, into, drop=True, sc=sc)
                        # a check keeps the hands that would have bet at NARROW_SOFT, as narrowing does (see
                        # _soften): cut to nothing, the checker was a capped range paying our value bets off
                        return bg, sm, r * bg, r * sm, r * _soften(r, ~dr)
                    bg, sm, r_bg, r_sm, r_x = C(path + (("bm", a),), bet_fn)
                    if bg[hands[a]] or sm[hands[a]]:
                        bet = (a, big) if bg[hands[a]] else (a, small)
                        rs[a] = r_bg if bg[hands[a]] else r_sm
                        break
                    rs[a] = r_x
                    path += (("x", a),)
            if bet is None:
                return None, p, put, alive, rs

            # The bet: everyone else faces it once, in order after the bettor.
            X, b = bet
            ls_stats["bets"] += 1
            ls_stats["small"] += b == small
            ls_stats["ours"] += X == "hero"
            p0 = p
            d = put_in(X, b); p += d
            if X == "hero":
                put += d
            path += (("b", X, b),)
            k = actors.index(X)
            order = actors[k + 1:] + actors[:k]
            facing = [a for a in order if a != "hero"]

            def face_fn(fa=tuple(facing), rr=tuple(rs[a] for a in facing)):
                fold_each = 1.0 - keep(b, p0, len(chips))     # it was made into everyone else
                out = {}
                for a, r, c in zip(fa, rr, _joint_top(list(rr), fold_each ** len(fa), rank)):
                    # against our bet they call or fold: the strongest share of a range capped by checks and
                    # calls isn't a raise against a bet we only make when ahead (it stacked off 77 against
                    # top pair); another opponent's bet they raise with the strongest part of what continues
                    rm = (np.zeros(N_COMBOS, dtype=bool) if X == "hero"
                          else aggro(r, _share(r, c) * RAISE_SHARE, 2 * b, p0 + b, rank, bluff))
                    out[a] = (c, rm, r * (c & ~rm), r * rm)
                return (out,)
            (fc,) = C(path + ("face",), face_fn)
            called, raiser, R = [], None, 0.0
            for j, a in enumerate(order):
                if a == "hero":
                    yet = tuple(o for o in order[j + 1:])

                    def hero_fn(cl=tuple(called), yet=yet, pc=p):
                        ws = [rs[X]] + [rs[c] for c in cl]
                        to = min(round(_raise_to(b, sum(inv[c] for c in cl)) * 2) / 2, left["hero"])
                        if to > b + 1e-9:
                            others = [X] + list(cl) + list(yet)
                            rr = [rs[o] for o in others]
                            fold = (1.0 - keep(to, pc, len(others))) ** len(others)
                            bar = RAISE_EQ * rake_bar(to - b, pc + b)
                            if all(equity(r * c, mt) > bar for r, c in zip(rr, _joint_top(rr, fold, rank))):
                                return ("raise", to)
                        ok = calls_turn(ws, sts, mt, b, pc, left["hero"]) if n == 4 else calls(ws, mt, b, pc)
                        return ("call" if ok else "fold", b)
                    dec, to = C(path + ("hero",), hero_fn)
                    if dec == "fold":
                        ls_stats["our_folds"] += 1
                        return -put, p, put, alive, rs
                    if dec == "raise":
                        raiser, R = "hero", to
                        break
                    d = put_in("hero", b); p += d; put += d
                    called.append("hero"); path += ("c",)
                    continue
                c, rm, r_call, r_raise = fc[a]
                if rm[hands[a]]:
                    to = min(round(_raise_to(b, sum(inv[o] for o in called)) * 2) / 2, inv[a] + left[a])
                    if to > b + 1e-9:
                        raiser, R = a, to
                        rs[a] = r_raise
                        break
                if c[hands[a]]:
                    p += put_in(a, b); rs[a] = r_call; called.append(a)
                    path += (("c", a),)
                else:
                    alive.remove(a)
                    path += (("f", a),)
            def folded_to():
                """Everyone else folded to our bet or raise: the pot less its rake (our uncalled chips come back).
                With an opponent all-in from before, the pot goes to showdown with them instead: (None, pot, put)."""
                unc = inv["hero"] - max(inv[a] for a in actors if a != "hero")
                if stuck:
                    left["hero"] += unc
                    return None, p - unc, put - unc
                return won(p - unc) + unc - put, p, put
            ls_stats["multi_called"] += len(called) >= 2
            if raiser is not None:
                ls_stats["raises"] += 1
                ls_stats["raises_over_calls"] += bool(called)
            if raiser is None:
                if X == "hero" and not called:
                    res, p, put = folded_to()
                    return res, p, put, alive, rs
                return None, p, put, alive, rs

            # A raise: everyone still in calls or folds it, in order after the raiser (one strength bar).
            path += (("r", raiser),)
            p1, add = p, R - inv[raiser]
            d = put_in(raiser, R); p += d
            if raiser == "hero":
                put += d
            k = actors.index(raiser)
            resp = [a for a in actors[k + 1:] + actors[:k] if a == "hero" or a in alive]
            ro = [a for a in resp if a != "hero"]

            def rface_fn(ro=tuple(ro), rr=tuple(rs[a] for a in ro)):
                fold_each = 1.0 - keep(add, p1, len(resp))
                return tuple((a, c, r * c) for a, r, c in zip(ro, rr, _joint_top(list(rr), fold_each ** len(ro), rank)))
            rf = {a: (c, r) for a, c, r in C(path + ("rface",), rface_fn)}
            inr = [raiser]
            for a in resp:
                if a == "hero":
                    def call_fn(inr=tuple(inr), pc=p):
                        return (calls([rs[o] for o in inr], mt, R - inv["hero"], pc),)
                    if not C(path + ("hr", tuple(inr)), call_fn)[0]:
                        return -put, p, put, alive, rs
                    d = put_in("hero", R); p += d; put += d
                    continue
                c, r_c = rf[a]
                if c[hands[a]]:
                    p += put_in(a, R); rs[a] = r_c; inr.append(a)
                else:
                    alive.remove(a)
            if raiser == "hero" and len(inr) == 1:
                res, p, put = folded_to()
                return res, p, put, alive, rs
            return None, p, put, alive, rs

        def showdown(st, hands, alive, p, put):
            """Our result at showdown against the opponents `alive`, the pot p, having added `put`: our share
            of the pot (raked), or with an opponent all-in for less than us, of each side pot we're in (the
            chips up to each all-in player's stake; the dead money of players who folded on the way goes in
            by what they put in, and what folded inside the trial by width)."""
            H = h0 + put
            stake = {i: min(fixed.get(i, tot_cap[i]), H) for i in alive}
            if all(c >= H - 1e-9 for c in stake.values()):
                return _showdown_share(st, idx, [hands[i] for i in alive]) * won(p) - put
            levels = sorted(set(stake.values()) | {H})
            known = H + sum(stake.values()) + sum(dead)
            resid = max(0.0, p - known)
            got, prev = 0.0, 0.0
            for lev in levels:
                w = lev - prev
                if w <= 1e-12:
                    continue
                amt = w + sum(min(c, lev) - min(c, prev) for c in stake.values())
                amt += sum(min(d, lev) - min(d, prev) for d in dead) + resid * w / max(H, 1e-9)
                elig = [hands[i] for i in alive if stake[i] >= lev - 1e-9]
                got += (_showdown_share(st, idx, elig) if elig else 1.0) * amt
                prev = lev
            return got * won(p) / max(p, 1e-9) - put

        def play_out(run, st, hands, live, p, put, matched):
            """Our result (relative to folding now) once this street's betting is over. live:
            [(opponent, their range now)]; p the pot; put what we've added this street; matched the
            bet everyone left has matched (an opponent all-in for less has what their stake left)."""
            alive = [i for i, _ in live]
            rs = dict(live)
            left = {"hero": hero_rem - put, **{i: max(0.0, cap[i] - matched) for i in alive}}
            for n in range(len(board) + 1, 6):
                res, p, put, alive, rs = later_street(n, run, st, hands, alive, rs, left, p, put)
                if res is not None:
                    return res
            return showdown(st, hands, alive, p, put)

        sc0, rank0, sts0, bluff0, mt0 = now
        trials = 0
        share_all = 0.0

        # Calling: players yet to act behind us fold, call or raise.
        if to_call > 1e-9:
            p_call = pot + call_amt
            behind = [i for i in after if cur.bets[opps[i]] < cur.cur_bet - 1e-9 and opps[i] not in cur.allin]
            matched = [i for i in range(n_opp) if i not in behind]
            # chips that called the bet before them (ours included; the bettor's own aside)
            calls_in = max(0.0, sum(min(cur.bets[opps[i]], cur.cur_bet) for i in matched) - cur.cur_bet) + cur.bets[hero] + call_amt
            beh = {}
            costs = [min(cur.cur_bet, cap[i]) - cur.bets[opps[i]] for i in behind]
            # They face the bet we call, which has to get past everyone it was made into: each folds as
            # often as when all of them face it together (see _keep), not as if it were a fresh bet into
            # the pot it has already grown with our call (that kept 80-90% of their range, junk included).
            fold_each = 1.0 - _keep(cur.bet_risk, cur.bet_pot, max(cur.bet_faced, 1))
            beh_conts = _joint_top([opp_ranges[i] for i in behind], fold_each ** len(behind), rank0)
            for i, cost, cont in zip(behind, costs, beh_conts):
                k = _share(opp_ranges[i], cont)
                rto = min(round(_raise_to(cur.cur_bet, calls_in) * 2) / 2, cap[i])
                hero_to = min(rto, cur.bets[hero] + hero_rem)
                extra = hero_to - cur.cur_bet
                raise_m = aggro(opp_ranges[i], k * RAISE_SHARE, rto - cur.cur_bet, p_call + cost, rank0, bluff0) if extra > 1e-9 else None
                p_r = p_call + rto - cur.bets[opps[i]]
                beh[i] = (cost, cont, raise_m, rto, hero_to, extra, p_r,
                          raise_m is not None and calls(opp_ranges[i] * raise_m, mt0, extra, p_r))
            cont_r = {i: opp_ranges[i] * beh[i][1] for i in behind}
            raise_r = {i: opp_ranges[i] * beh[i][2] for i in behind if beh[i][2] is not None}

        # Betting or raising: to 75% of the pot after calling, or 33% / 66% pot, to the nearest 0.5bb. Each
        # size gets its own continuing ranges (a smaller bet keeps more of everyone's range) and raises.
        def agg_line(agg_to):
            hero_add = agg_to - cur.bets[hero] if agg_to > cur.cur_bet + 1e-9 else 0.0   # no raise once a call is all-in
            if hero_add <= 1e-9:
                return None
            opp_add = [max(0.0, min(agg_to, cap[i]) - cur.bets[opps[i]]) for i in range(n_opp)]
            # Players already all-in can't fold; the rest share one strength bar (see _joint_top).
            can_fold = [i for i in range(n_opp) if opp_add[i] > 1e-9]
            conts = [np.ones(N_COMBOS, dtype=bool) for _ in range(n_opp)]
            if can_fold:
                # each prices what they must add against the pot our bet makes (the same as before for a first
                # bet): priced on our whole raise, CO's 85bb turn bet folded to a 12.5bb jam and air jammed for +2.9
                n_f = len(can_fold)
                fold = float(np.prod([1.0 - _keep(opp_add[i], pot + hero_add - opp_add[i], n_f) for i in can_fold]))
                for i, m in zip(can_fold, _joint_top([opp_ranges[i] for i in can_fold], fold, rank0)):
                    conts[i] = m
            k_agg = [_share(opp_ranges[i], conts[i]) for i in range(n_opp)]
            rr = {}
            for i in range(n_opp):
                rto = min(round(_raise_to(agg_to, 0.0) * 2) / 2, cap[i])
                hero_to = min(rto, cur.bets[hero] + hero_rem)
                extra = hero_to - agg_to
                if extra > 1e-9:
                    rs_i = CHECKED_RAISE_SHARE if n_opp >= 3 and cur.cur_bet <= 1e-9 and opps[i] in cur.acted else RAISE_SHARE
                    m = aggro(opp_ranges[i], k_agg[i] * rs_i, rto - agg_to, pot + hero_add + opp_add[i], rank0, bluff0)
                    p_r = pot + hero_add + rto - cur.bets[opps[i]]
                    rr[i] = (m, opp_ranges[i] * m, hero_to, extra, p_r,
                             calls(opp_ranges[i] * m, mt0, extra, p_r))
            return {"to": agg_to, "add": hero_add, "opp_add": opp_add, "conts": conts, "k": k_agg,
                    "cont_r": [opp_ranges[i] * conts[i] for i in range(n_opp)], "rr": rr}

        # (a size held to what the deepest opponent can match, under the smallest bet or raise while you have more, is
        # shown as that smallest one: the same move, nobody can put in more; a raise to 55.88 is a raise to 90. One
        # that's your whole stack is all-in: the page takes a bet or raise in half big blinds, so a raise to 1.55 was
        # a raise to 1.5, not above the bet)
        real_rem = max(0.0, stk[hero] - cur.inv_start[hero] - cur.bets[hero])
        shown_to = {}

        def shown_as(name, to):
            v = _legal_to(cur, hero, to, real_rem)
            name = "all-in" if v >= cur.bets[hero] + real_rem - 1e-9 else name.format(v)
            shown_to[name] = v
            return name

        if to_call > 1e-9:
            # our raise option keeps the page's default size (the bet plus 75% of the pot after calling,
            # callers' chips included); raises inside the trials go 3x plus the calls. None when an all-in
            # for less than a full raise left the betting closed to us, or nobody has chips to face one.
            agg_to = _legal_to(cur, hero, cur.cur_bet + 0.75 * (pot + call_amt), hero_rem)
            sizes = [(shown_as("raise to {:g}", agg_to), agg_to)] if can_raise(cur, hero) else []
        else:
            big, small = (_legal_to(cur, hero, f * pot, hero_rem) for f in (0.66, SMALL_BET))
            sizes = ([(shown_as("bet 33%", small), small)] if 1e-9 < small < big - 1e-9 else []) + \
                [(shown_as("bet 66%", big), big)]
        if getattr(self, "_jam", False) and (to_call <= 1e-9 or can_raise(cur, hero)):
            # a jam request (the page grading your all-in): the rules price all-in too, as its own line or as the size
            # that already puts the stack in
            jam_to = cur.bets[hero] + hero_rem
            sizes = [(n, t) for n, t in sizes if t < jam_to - 1e-9] + [("all-in", jam_to)]
        aggs = {}
        for name, to in sizes:
            a = agg_line(to)
            if a is not None:
                aggs[name] = a

        # Checking: an opponent behind may bet (the first whose betting range at either size holds their
        # hand). The others face that bet like players behind a call: one strength bar, each folding as
        # often as with everyone it was made into facing it together, the strongest part of what
        # continues raising. Those between the bettor and us act first, so we fold, call or check-raise
        # knowing what they did (per bettor, size and what they did, whichever pays over the trials);
        # those after us act once we've called.
        bets, bet_masks, chk_r = {}, {}, {}
        if to_call <= 1e-9:
            pos_of = {i: seq.index(opps[i]) for i in range(n_opp)}
            # A player who checks behind us before the bettor holds the check range (what a check rules out
            # keeps NARROW_SOFT, as narrowing does): with the full range, its folds to our check-raise came
            # out a third of the time where its actual checking hands folded twice as often.
            for j in after:
                if opps[j] in cur.acted or opps[j] in cur.allin:
                    continue
                room = min(cap[j] - cur.bets[opps[j]], hero_rem)
                b_big = min(round(0.66 * pot * 2) / 2, room)
                if b_big <= 1e-9:
                    continue
                b_small = min(round(SMALL_BET * pot * 2) / 2, room)
                b_small = b_small if 1e-9 < b_small < b_big - 1e-9 else None
                into = [opp_ranges[i] for i in range(n_opp) if i != j] + [hero_range]
                bg_j, sm_j, drop_j = _bet_ranges(opp_ranges[j], n_opp, pot, b_big, b_small, rank0, bluff0, into, drop=True,
                                                 sc=sc0)
                bet_masks[j] = (b_big, b_small, bg_j, sm_j)
                chk_r[j] = opp_ranges[j] * _soften(opp_ranges[j], ~drop_j)
                others = [i for i in range(n_opp) if i != j and opps[i] not in cur.allin
                          and cap[i] - cur.bets[opps[i]] > 1e-9]
                for b, m in ((b_big, bet_masks[j][2]), (b_small, bet_masks[j][3])):
                    if b is None:
                        continue
                    fold_each = 1.0 - keep(b, pot, n_opp)
                    # their ranges facing the bet: the check range for those who checked before the bettor
                    base = {i: chk_r[i] if i in chk_r and pos_of[i] < pos_of[j] else opp_ranges[i] for i in others}
                    conts = _joint_top([base[i] for i in others], fold_each ** len(others), rank0) if others else []
                    face = {}
                    for i, cont in zip(others, conts):
                        ri = base[i]
                        cost = min(b, cap[i] - cur.bets[opps[i]])
                        before_us = pos_of[i] > pos_of[j]
                        p_i = pot + b + (0.0 if before_us else b)   # the pot they face (callers before them aside)
                        rto = min(round(_raise_to(b, 0.0 if before_us else b) * 2) / 2, cap[i])
                        hero_to = min(rto, hero_rem)
                        extra = hero_to - (0.0 if before_us else b)
                        raise_m = (aggro(ri, _share(ri, cont) * RAISE_SHARE, rto - b, p_i + cost, rank0, bluff0)
                                   if rto > b + 1e-9 and extra > 1e-9 else None)
                        face[i] = {"first": before_us, "cost": cost, "cont": cont, "cont_r": ri * cont, "range": ri,
                                   "raise_m": raise_m, "raise_r": ri * raise_m if raise_m is not None else None,
                                   "rto": rto, "hero_to": hero_to, "extra": extra,
                                   # after our call we face their raise: call by equity against it (as behind a call)
                                   "we_call": (raise_m is not None and not before_us
                                               and calls(ri * raise_m, mt0, extra, p_i + rto))}
                    bets[(j, b)] = {"b": b, "m": m, "r": opp_ranges[j] * m, "face": face,
                                    "before": sorted((i for i in others if face[i]["first"]), key=pos_of.get),
                                    "after": sorted((i for i in others if not face[i]["first"]), key=pos_of.get)}
            stuck = [(i, opp_ranges[i]) for i in range(n_opp) if opps[i] in cur.allin]   # all-in: can't act, still in

        xr_cache = {}

        def check_raise(key):
            """Our check-raise after the bet in `key` (bettor, size, players between who called), to 3x
            the bet plus their calls: everyone still in faces it (the bettor, the callers, the players
            behind us), folding together by the joint rule; the strongest part of what continues
            re-raises 3x and we call that by equity against it. None when we can't raise."""
            if key in xr_cache:
                return xr_cache[key]
            j, b, seen = key
            bt = bets[(j, b)]
            face = bt["face"]
            calls_in = sum(face[i]["cost"] for i in seen)
            R = min(round(_raise_to(b, calls_in) * 2) / 2, hero_rem)
            out = None
            if R > b + 1e-9:
                facing = sorted([j] + list(seen) + list(bt["after"]), key=pos_of.get)
                rng_of = {j: bt["r"], **{i: face[i]["cont_r"] for i in seen}, **{i: face[i]["range"] for i in bt["after"]}}
                in_of = {j: b, **{i: face[i]["cost"] for i in seen}}
                p0 = pot + b + calls_in
                fold = (1.0 - keep(R, p0, len(facing))) ** len(facing)
                resp = {}
                for i, c in zip(facing, _joint_top([rng_of[i] for i in facing], fold, rank0)):
                    r, have = rng_of[i], in_of.get(i, 0.0)
                    cost = max(0.0, min(R, cap[i] - cur.bets[opps[i]]) - have)
                    rr_to = min(round(_raise_to(R, 0.0) * 2) / 2, cap[i] - cur.bets[opps[i]])
                    hero_to = min(rr_to, hero_rem)
                    rm = (aggro(r, _share(r, c) * RAISE_SHARE, rr_to - R, p0 + R + cost, rank0, bluff0)
                          if hero_to > R + 1e-9 else None)
                    p_rr = p0 + R + rr_to - have      # the pot we face their re-raise in (other callers aside)
                    resp[i] = (c, r * c, rm, r * rm if rm is not None else None, cost, rr_to - have, hero_to,
                               rm is not None and calls(r * rm, mt0, hero_to - R, p_rr))
                out = {"to": R, "p0": p0, "facing": facing, "resp": resp}
            xr_cache[key] = out
            return out

        cums = [np.cumsum(r) for r in opp_ranges]
        # Trials reuse the runouts, each pass with fresh opponent hands, and every action is scored on
        # the same trials. Stop once the gap between the two best actions is clear of the noise.
        min_trials, max_seconds = (1200, 5.0) if accurate else (400, 1.5)
        max_seconds = min(max_seconds, rules_left)
        everyone = [(i, chk_r.get(i, opp_ranges[i])) for i in range(n_opp)]   # nobody bet: the check ranges
        # per trial; "chk" holds (what we face after checking, value if we call, value if we check-raise):
        # None (nobody bets), (bettor, size, players between who called) or (bettor, size, "raise", the
        # raiser between; no check-raise then)
        vals = {"call": [], "chk": [], **{name: [] for name in aggs}}
        t_mc = time.time()

        def chk_plan():
            """{check-line situation: "call" / "raise" / "fold"}, whichever pays over the trials. A
            check-raise has to beat the better of calling and folding by XR_Z standard errors of the paired
            difference (the same trials scored both ways), over at least XR_MIN trials: the situations
            split the trials finely and the raise's edge is often small, so on the bare means middle pair
            check-raised a third of the bets and a gutshot most of them, whichever way the noise fell."""
            tot = {}
            for k, v, x in vals["chk"]:
                if k is not None:
                    tot.setdefault(k, []).append((v, x))
            plan = {}
            for k, vx in tot.items():
                v = np.array([a for a, _ in vx])
                alt = "call" if v.sum() > 0 or priced(k) else "fold"
                if vx[0][1] is not None and len(vx) >= XR_MIN:
                    d = np.array([b for _, b in vx]) - (v if alt == "call" else 0.0)
                    if d.mean() > XR_Z * d.std() / np.sqrt(len(d)):
                        alt = "raise"
                plan[k] = alt
            return plan

        priced_cache = {}

        def priced(key):
            """Does our equity against the bettor and the callers between (or the raiser between) pay the
            price of calling by itself, as our calls on later streets are decided? Such a call is made
            even where the trials, split by bettor, size and what the players between did, came out
            slightly negative: their noise (and the later streets' rules) folded top pair against a bet
            that a draw with less equity called."""
            if key not in priced_cache:
                bt = bets[key[:2]]
                face = bt["face"]
                if key[2] == "raise":
                    f = face[key[3]]
                    ws, cost = [f["raise_r"]], f["hero_to"]
                    p_ = pot + bt["b"] + f["rto"]
                else:
                    ws, cost = [bt["r"]] + [face[i]["cont_r"] for i in key[2]], bt["b"]
                    p_ = pot + bt["b"] + sum(face[i]["cost"] for i in key[2])
                priced_cache[key] = calls(ws, mt0, cost, p_)
            return priced_cache[key]

        def summary():
            """{action: per-trial values} with each check-line bet called, check-raised or folded,
            whichever pays."""
            out = {}
            if to_call > 1e-9:
                out["fold"] = np.zeros(len(vals["call"])); out["call"] = np.array(vals["call"])
            else:
                plan = chk_plan()
                out["check"] = np.array([v if k is None else v if plan[k] == "call" else x if plan[k] == "raise" else 0.0
                                         for k, v, x in vals["chk"]])
            for name in aggs:
                out[name] = np.array(vals[name])
            return out

        def gap():
            """(gap between the two best mean EVs, its standard error from paired trials)."""
            out = summary()
            if len(out) < 2 or trials < 2:
                return 0.0, 0.0
            a, b = sorted(out.values(), key=lambda x: -x.mean())[:2]
            d = a - b
            return float(d.mean()), float(d.std() / np.sqrt(len(d)))

        while True:
            if self.should_stop is not None and self.should_stop():
                break
            for run, st in runs:
                used = set(run) | set(hole)
                hands = []
                for cum in cums:
                    for _ in range(50):
                        j = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
                        j = min(j, N_COMBOS - 1)
                        if not (set(ALL_COMBOS[j]) & used):
                            break
                    else:
                        break
                    used |= set(ALL_COMBOS[j]); hands.append(j)
                if len(hands) < n_opp:
                    continue
                trials += 1
                share_all += _showdown_share(st, idx, hands)
                if to_call > 1e-9:
                    p, live, res = p_call, [(i, opp_ranges[i]) for i in matched], None
                    for i in behind:
                        cost, cont, raise_m, rto, hero_to, extra, p_r, we_call = beh[i]
                        if raise_m is not None and raise_m[hands[i]]:
                            res = (play_out(run, st, hands, [(i, raise_r[i])], p_r + extra, call_amt + extra, hero_to)
                                   if we_call else -call_amt)
                            break
                        if cont[hands[i]]:
                            p += cost; live.append((i, cont_r[i]))
                    vals["call"].append(res if res is not None else play_out(run, st, hands, live, p, call_amt, cur.cur_bet))
                else:
                    first = None
                    for j in after:
                        if j in bet_masks:
                            b_big, b_small, bg, sm = bet_masks[j]
                            if bg[hands[j]] or sm[hands[j]]:
                                first = (j, b_big if bg[hands[j]] else b_small)
                                break
                    if first is None:
                        vals["chk"].append((None, play_out(run, st, hands, everyone, pot, 0.0, cur.cur_bet), None))
                    else:
                        bt = bets[first]
                        b, face = bt["b"], bt["face"]
                        p, live, seen, key, res, res_x = pot + b, [(first[0], bt["r"])] + stuck, [], None, None, None
                        for i in bt["before"]:          # between the bettor and us
                            f = face[i]
                            if f["raise_m"] is not None and f["raise_m"][hands[i]]:
                                key = first + ("raise", i)
                                res = play_out(run, st, hands, [(i, f["raise_r"])] + stuck, p + f["rto"] + f["hero_to"],
                                               f["hero_to"], f["hero_to"])
                                break
                            if f["cont"][hands[i]]:
                                p += f["cost"]; live.append((i, f["cont_r"])); seen.append(i)
                        if key is None:
                            key = first + (tuple(seen),)
                            xr = check_raise(key)
                            if xr is not None:          # we check-raise
                                R = xr["to"]
                                px, livex = xr["p0"] + R, list(stuck)
                                for i in xr["facing"]:
                                    c, r_c, rm, r_rm, cost, rr_add, hero_to, we_call = xr["resp"][i]
                                    if rm is not None and rm[hands[i]]:
                                        res_x = (play_out(run, st, hands, [(i, r_rm)] + stuck, px + rr_add + hero_to - R,
                                                          hero_to, hero_to) if we_call else -R)
                                        break
                                    if c[hands[i]]:
                                        px += cost; livex.append((i, r_c))
                                if res_x is None:
                                    # everyone folded to it: the pot with their bet and our call of it, raked
                                    res_x = (won(xr["p0"] + b) - b if len(livex) == len(stuck) == 0
                                             else play_out(run, st, hands, livex, px, R, R))
                            p += b
                            for i in bt["after"]:       # after us, once we've called
                                f = face[i]
                                if f["raise_m"] is not None and f["raise_m"][hands[i]]:
                                    res = (play_out(run, st, hands, [(i, f["raise_r"])] + stuck, p + f["rto"] + f["extra"],
                                                    b + f["extra"], f["hero_to"]) if f["we_call"] else -b)
                                    break
                                if f["cont"][hands[i]]:
                                    p += f["cost"]; live.append((i, f["cont_r"]))
                            if res is None:
                                res = play_out(run, st, hands, live, p, b, b)
                        vals["chk"].append((key, res, res_x))
                for name, a in aggs.items():
                    res = None
                    for i in after:
                        if i in a["rr"] and a["rr"][i][0][hands[i]]:
                            _, r_rr, hero_to, extra, p_r, we_call = a["rr"][i]
                            res = (play_out(run, st, hands, [(i, r_rr)], p_r + extra, a["add"] + extra, hero_to)
                                   if we_call else -a["add"])
                            break
                    if res is None:
                        callers = [i for i in range(n_opp) if a["conts"][i][hands[i]]]
                        if not callers:
                            res = won(pot)
                        else:
                            final = pot + a["add"] + sum(a["opp_add"][i] for i in callers)
                            res = play_out(run, st, hands, [(i, a["cont_r"][i]) for i in callers], final, a["add"], a["to"])
                    vals[name].append(res)
            if self.mw_trials is not None:
                if trials >= self.mw_trials:
                    break
            elif trials >= min_trials:
                d, se = gap()
                # (or within 2% of the pot: two lines that close are a tie for advice, and chasing the noise
                # between them ran into the time cap every time, so the same spot gave different numbers)
                if se < max(0.25, abs(d) / 3, 0.02 * pot) or time.time() - t_mc > max_seconds or trials >= 20000:
                    break
            elif time.time() - t_mc > 3 * max_seconds:
                break
        trials = max(trials, 1)
        eq = float(share_all / trials)
        ev = {k: float(v.mean()) if len(v) else 0.0 for k, v in summary().items()}
        d, se = gap()
        acts = []
        check_line = None
        if to_call > 1e-9:
            need = call_amt / (pot + call_amt)
            acts.append({"action": "fold", "freq": 0.0, "ev_bb": 0.0})
            acts.append({"action": "call", "freq": 0.0, "ev_bb": round(ev["call"], 2), "amount_bb": round(call_amt, 2)})
            notes.append(f"pot odds need {need*100:.0f}% equity, you have {eq*100:.0f}% against everyone's whole range")
            if behind:
                notes.append(f"after a call, {', '.join(opps[i] for i in behind)} still to act: "
                             + ", ".join(f"{opps[i]} continues with {_share(opp_ranges[i], beh[i][1]) * 100:.0f}%"
                                         for i in behind) + " of their range")
        else:
            acts.append({"action": "check", "freq": 0.0, "ev_bb": round(ev["check"], 2)})
            faced = [k for k, _, _ in vals["chk"] if k is not None]
            if faced:
                plan = chk_plan()
                n_raised = sum(1 for k in faced if k[2] == "raise")
                n_called = sum(1 for k in faced if k[2] != "raise" and k[2])
                n_small = sum(1 for k in faced if k[1] < 0.5 * pot)
                n_call = sum(1 for k in faced if plan[k] == "call")
                n_xr = sum(1 for k in faced if plan[k] == "raise")
                plain = [k for k in faced if k[2] != "raise"]
                check_line = {"bet": len(faced) / trials, "small": n_small / len(faced),
                              "call": n_call / len(faced), "raise": n_xr / len(faced),
                              "fold": 1.0 - (n_call + n_xr) / len(faced),
                              # continuing against a bet nobody between raised
                              "continue_unraised": sum(1 for k in plain if plan[k] != "fold") / max(len(plain), 1),
                              # ...and against the small ones (the big bet is the nut-advantage range: see _bet_ranges)
                              "continue_small": (sum(1 for k in plain if k[1] < 0.5 * pot and plan[k] != "fold")
                                                 / max(sum(1 for k in plain if k[1] < 0.5 * pot), 1))}
                between = (f"; a player between the bettor and you calls it {n_called / len(faced) * 100:.0f}% of the time"
                           f" and raises {n_raised / len(faced) * 100:.0f}%") if n_raised + n_called else ""
                notes.append(f"if you check, someone bets {len(faced) / trials * 100:.0f}% of the time"
                             f" (a third of the pot in {n_small / len(faced) * 100:.0f}% of those){between}"
                             f"; you'd call {n_call / len(faced) * 100:.0f}% of those"
                             f" and check-raise {n_xr / len(faced) * 100:.0f}%")
        for name, a in aggs.items():
            acts.append({"action": name, "freq": 0.0, "ev_bb": round(ev[name], 2),
                         "amount_bb": round(shown_to.get(name, a["to"]), 2)})
            notes.append(f"vs {name}: " + ", ".join(f"{opps[i]} continues with {a['k'][i]*100:.0f}%"
                                                    for i in range(n_opp)) + " of their range")
        thin = set()
        callers_in = sum(1 for i in range(n_opp) if cur.bets[opps[i]] >= cur.cur_bet - 1e-9) - 1   # (the bettor aside)
        if to_call > 1e-9 and n_opp >= 3 and callers_in >= 1:
            for name, a in aggs.items():
                eqs = [equity(a["cont_r"][i], mt0) for i in range(n_opp) if a["opp_add"][i] > 1e-9 and a["cont_r"][i].sum() > 0]
                if eqs and min(eqs) <= ROOT_RAISE_EQ:
                    thin.add(name)
        pick = max((i for i in range(len(acts)) if acts[i]["action"] not in thin), key=lambda i: acts[i]["ev_bb"])
        if any(acts[i]["ev_bb"] > acts[pick]["ev_bb"] for i in range(len(acts))):
            notes.append(f"not raising: the raise isn't for value ({ROOT_RAISE_EQ * 100:.0f}%+ equity against what each "
                         f"player continues with), and the trials' later streets pay a thin raise off too often")
        acts[pick]["freq"] = 1.0
        notes.append(f"{trials} trials; the best action leads the next by {d:.2f} ± {se:.2f} bb")
        out = {"street": cur.name, "hero": hero, "hand": hand_class(hole), "board": cards_str(board),
               "players": len(remaining), "multiway": True, "opponents": opps,
               "equity_vs_field": round(eq, 3), "pot_bb": round(pot, 2), "to_call_bb": round(call_amt, 2),
               "actions": acts, "recommended": acts[pick]["action"], "sampled": acts[pick]["action"],
               "notes": notes + [f"{len(remaining)} players in the pot: not a solve here (the solvers take heads-up and 3-way pots and 4+ way rivers), this is a Monte Carlo heuristic"],
               "seconds": round(time.time() - t0, 2)}
        if ls_stats["bets"]:
            n_b = ls_stats["bets"]
            out["later_streets"] = {"bets": n_b, "small": round(ls_stats["small"] / n_b, 3),
                                    "multi_called": round(ls_stats["multi_called"] / n_b, 3),
                                    "raised": round(ls_stats["raises"] / n_b, 3),
                                    "raises_over_calls": ls_stats["raises_over_calls"],
                                    # our bets and our folds to a bet (counts), for checking our in-trial rules
                                    "ours": ls_stats["ours"], "our_folds": ls_stats["our_folds"]}
        if check_line is not None:
            out["check_line"] = {k: round(v, 3) for k, v in check_line.items()}   # after a check: how often someone bets, and our plan
        return out

    # ------------------------------------------------------------ practice opponents
    def play(self, state: dict, seat: str, hole, seed: Optional[int] = None) -> dict:
        """Practice opponents: a move for `seat` holding `hole`, drawn from the GTO mix for that hand (preflop
        the chart's, after the flop the solve's). `state` is the hand as the page sends it, from the player
        practising (its "hero" and "hole"; GTO opponents whatever its "opponents" says). Heads-up after the
        flop the move is read off that player's own solve of the street, so one solve serves the villain's
        move and the hero's advice (the villain's range there leaves out the hero's cards, as in the hero's
        advice: an edge the size of a blocker). Otherwise (preflop, multiway, a pot that became heads-up this
        street, a line of the hero's the solve rarely takes, a hand it gives no weight) it's the advice for
        `seat` holding `hole`. `seed` makes the draw repeatable.
        Returns the move in the page's format: {"pos", "action", "to" (bets and raises, chips), "name" (the
        advisor's), "freq", "street", "actions" (the mix it was drawn from)}, or {"error"}."""
        rng = random.Random(seed) if seed is not None else self.rng
        st = {k: v for k, v in state.items() if k != "opponents"}
        seat = str(seat).upper()
        hole = tuple(parse_cards(hole)) if isinstance(hole, str) else tuple(hole)
        res = None
        with np.errstate(all="ignore"):
            if st.get("board") and st.get("hole") and str(st.get("hero", "")).upper() != seat:
                try:
                    res = self._advise(st, actor=(seat, hole))
                except Cancelled:
                    raise
                except Exception:   # noqa: BLE001 - (the seat's own advice below, with advise's catch-all)
                    res = None
        if res is None or "error" in res or not res.get("actions"):
            # from the seat's own view: a GTO player, so not best-responding to the locks the player practising
            # set on their opponents (which can name this seat: "you can't lock your own seat" stalled practice)
            res = self.advise({**{k: v for k, v in st.items() if k not in ("locks", "study")}, "hero": seat,
                               "hole": cards_str(hole)})
        if "error" in res:
            return {"error": res["error"]}
        acts = [a for a in res["actions"] if a["freq"] > 0] or res["actions"]
        r = rng.random() * sum(a["freq"] for a in acts)
        pick = acts[-1]
        for a in acts:
            r -= a["freq"]
            if r <= 0:
                pick = a
                break
        name = pick["action"]
        out = {"pos": seat, "action": name, "name": name, "freq": pick["freq"], "street": res.get("street"),
               "actions": res["actions"]}
        if name.startswith(("bet", "raise")):
            out["action"] = name.split()[0]
            out["to"] = round(float(pick["amount_bb"]) * float(state.get("bb", 1.0)), 2)
        return out

    def _mw_walk(self, ranges, before, board, st: StreetState, stk, hero, hole, streets, notes, current=False) -> bool:
        """Narrow the ranges (in place, already narrowed by the rules) through a 3-way street's solve: the latest one
        your advice on the street came from (its root's ranges are the rules' up to there), else, for an earlier
        street the speed solves (mw_solve_streets) when mw_prior_solve is on, one from the street's start
        (prior_budget). Each action since the solve's root multiplies its player's range by the solve's average
        strategy for it (smoothed by prior_smoothing, as heads-up), and the result counts by how settled the solve
        is (trust_of; the rest stays the rules'). The smoothing keeps a line the solve rarely takes from leaving a
        sliver (the heads-up walk's RARE_LINE fallback to the rules made the turn's ranges worse here: scripts/
        compare_multiway.py --narrowing). False when no solve was followed or its tree lacks a line's action."""
        from . import multiway_solver as MW
        acts = list(streets.get(st.name) or [])
        bk = board[:{"flop": 3, "turn": 4, "river": 5}[st.name]]
        hit, k = None, 0
        # (Study: the solve of an earlier node was stored under that node's actor)
        heroes = [hero] + ([p for p in st.players_start if p != hero] if self._study else [])
        for k in range(len(acts), -1, -1):
            hit = next((h for h in (self._mw_lines.get(_mw_line(bk, x, hole, stk, streets, st.name, k)) for x in heroes)
                        if h is not None), None)
            if hit is not None:
                break
        src = "the solve Study answered from" if self._study else "the solve your advice came from"
        if hit is None and st.name == "flop" and len(st.players_start) == 3:
            # a flop the 3-way library holds (before: the ranges entering it), through its stored lines
            lib = self._mw_library_solution(before, bk, st, stk)
            if lib is not None:
                hit, k, src = (lib, [p for p in st.order if p in st.players_start], lib.exploitability()), 0, \
                    "the precomputed 3-way solve"
        if hit is None:
            if current or not self.mw_prior_solve or st.name not in self.mw_solve_streets:
                return False
            hit, k = self._mw_street_solve(bk, before, st, stk), 0
            src = "a solve from the street's start"
            if hit is None:
                return False
        sol, players, expl = hit
        trust = trust_of(expl, st.name)
        if trust <= 0.0 or k >= len(acts):
            return False
        R = sol.reach0[:, 0, :].copy()
        node = sol.root
        for a in acts[k:]:
            if getattr(node, "unstored", False):
                return False   # (a library line it doesn't hold)
            if not isinstance(node, MW.Decision):
                break
            if a.get("pos") not in players or players.index(a["pos"]) != node.player:
                return False
            p = node.player
            i = _mw_pick(node, a)
            if i is None:
                return False
            sg = node.avg_strategy()[0][i].astype(np.float64)
            R[p] = R[p] * ((1.0 - self.prior_smoothing) * sg + self.prior_smoothing / len(node.names))
            node = node.children[i]
        for p, pos in enumerate(players):
            if pos in ranges:
                ranges[pos] = trust * sol.expand(R[p]) + (1.0 - trust) * ranges[pos]
        notes.append(f"{st.name}: 3-way, ranges narrowed through {src} ({sol.iters} iterations, {expl:.1f}% of the pot "
                     f"from equilibrium: followed {trust * 100:.0f}%, the rest by the betting rules)")
        return True

    def _mw_library_solution(self, entering, flop, st: StreetState, stk):
        """The 3-way library's solve of this flop (mw_library.LibraryMWSolution) when it holds the spot: the three
        players' ranges entering the flop (`entering`), the pot and the stack behind, no rake and no custom sizes."""
        if self.mw_library is None or len(flop) != 3 or len(st.players_start) != 3 or self.cfg.rake_rate > 0 \
                or self._req_sizes or getattr(self, "_icm", None) is not None:   # (solved in chips)
            return None
        players = [p for p in st.order if p in st.players_start]
        if any(p not in entering for p in players):
            return None
        S = min(stk[p] - st.inv_start[p] for p in players)
        try:
            return self.mw_library.solution([entering[p] for p in players], tuple(flop[:3]), st.pot_start, S)
        except Exception:   # a library that can't be read is no library
            return None

    def _mw_from_library(self, hero, hole, board, entering, cur: StreetState, stk, streets, t0, notes, locks,
                         positions=None):
        """A 3-way flop's first decision (first to act, or checked to) answered from the 3-way library's solve
        (gto/mw_library.py), shaped like _multiway_solve's answer; None where it doesn't hold the spot or the node."""
        if len(board) != 3 or cur.name != "flop" or cur.n_raises or any(a["action"] != "check" for a in cur.actions):
            return None
        players = [p for p in cur.order if p in cur.players_start]
        if set(players) != set(cur.active) or hero not in players or any(p in (locks or {}) for p in players):
            return None
        sol = self._mw_library_solution(entering, board, cur, stk)
        if sol is None:
            return None
        path = tuple("check" for _ in cur.actions)
        node = sol.nodes.get(path)
        hp = players.index(hero)
        ci = sol.combo_index(hole) if hole else None    # (Study: no cards of the actor's own)
        if node is None or node.player != hp or ci is None and not self._study:
            return None   # (a node it doesn't store: solved live)
        if ci is not None and sol.reach0[hp, 0, ci] <= 1e-12:
            notes.append("your hand isn't in your range for this line: its mix is the solve's current (regret-matching) "
                         "strategy for it, against the ranges as they are")
        R = sol.reach_at(path)
        expl = sol.exploitability()
        gains = sol.br_gains() / sol.root_pot() * 100
        S = min(stk[p] - cur.inv_start[p] for p in players)
        self._mw_lines[_mw_line(board, hero, hole, stk, streets, cur.name, 0)] = (sol, players, expl)
        strat = node.avg_strategy()[0]
        evs = sol.action_evs(node)
        shown = _mw_names(node.names, node.amounts, stk[hero] - cur.inv_start[hero], S, node.pot_now)
        lib_note = (f"precomputed 3-way solve ({sol.label}, {sol.iters} iterations, {expl:.2f}% of the pot from "
                    f"equilibrium): three-player CFR from the flop's start, bets of 33 and 75% pot, one raise and "
                    f"all-in; the turn and river bet on too, by 16 hand buckets a player")
        solver = {"iterations": sol.iters, "solve_seconds": 0.0, "nodes": len(sol.nodes), "active_combos": sol.hands.n,
                  "tree_actions": shown, "exploitability_pct": round(expl, 2),
                  "best_response_gains_pct": {p: round(float(g), 2) for p, g in zip(players, gains)}}
        if self._study:
            return self._study_mw_node(sol, node, R, players, hp, shown, strat, evs, 0.0, S, board, cur, positions or [],
                                       solver, notes + [lib_note, "best-response gains (% pot): " + ", ".join(
                                           f"{p} {g:.2f}" for p, g in zip(players, gains))], t0, precomputed=sol.info())
        mine = strat[:, ci]
        acts = [{"action": name, "freq": round(float(mine[a]), 3), "ev_bb": round(float(evs[a, ci]), 2),
                 "amount_bb": None if node.amounts[a] is None else round(float(node.amounts[a]), 2)}
                for a, name in enumerate(shown)]
        best = max(acts, key=lambda x: x["freq"])["action"]
        hero_r = sol.expand(R[hp, 0])
        w = hero_r / max(hero_r.sum(), 1e-12)
        sf = np.zeros((len(node.names), N_COMBOS)); sf[:, sol.hands.active] = strat
        opps = [p for p in players if p != hero]
        notes.append(lib_note)
        notes.append(f"best-response gains (% pot): " + ", ".join(f"{p} {g:.2f}" for p, g in zip(players, gains)))
        out = {"street": cur.name, "hero": hero, "hand": hand_class(hole), "board": cards_str(board),
               "players": 3, "multiway": True, "solved": True, "opponents": opps, "position": f"{hp + 1} of 3 to act",
               "pot_bb": round(node.pot_now, 2), "to_call_bb": 0.0, "stack_bb": round(S, 2),
               "actions": acts, "recommended": best, "sampled": self._sample(acts),
               "grid": strategy_grid([sf[a] for a in range(len(node.names))], shown, hero_r),
               "range_action_freq": {name: round(float((sf[a] * w).sum()), 3) for a, name in enumerate(shown)},
               "hero_range_combos": round(float(hero_r.sum()), 1),
               "opponent_range_combos": {p: round(float(R[i, 0].sum()), 1) for i, p in enumerate(players) if p != hero},
               "solver": solver, "precomputed": sol.info(), "notes": notes, "seconds": round(time.time() - t0, 2)}
        if self._breakdown:
            from . import breakdown as BD
            ev_full = np.zeros(N_COMBOS); ev_full[sol.hands.active] = (strat * evs).sum(axis=0)
            out["breakdown"] = BD.breakdown(board, shown, hero, hero_r, sf, ev_full,
                                            {p: sol.expand(R[i, 0]) for i, p in enumerate(players) if p != hero},
                                            node.pot_now, equity=False, hole=hole)
        return out

    def _mw_run(self, sol, left: float):
        """Runs a 3-way solve for at most `left` seconds (MW_RESERVE iterations' time kept back for what is read
        off after), stopping early once settled (MW_SETTLED, checked at MW_CHECKS). A solve its budget ran out on,
        or a settled one, isn't run again: a later request for the spot gets the same answer."""
        while not sol.out_of_time and not getattr(sol, "settled", False) and sol.iters < self.mw_max_iters:
            if left <= 0:
                sol.out_of_time = True
                break
            if self.should_stop is not None and self.should_stop():
                break
            t = time.time()
            nxt = min([c for c in MW_CHECKS if c > sol.iters] + [self.mw_max_iters])
            sol.run(iters=nxt - sol.iters, time_budget=left, should_stop=self.should_stop, reserve=MW_RESERVE, min_iters=1)
            if sol.out_of_time or sol.iters < nxt:
                break
            if sol.br_gains().mean() / sol.root_pot() * 100 < MW_SETTLED:
                sol.settled = True
            left -= time.time() - t

    def _mw_street_solve(self, board, ranges, st: StreetState, stk):
        """(solve, players, exploitability % pot) of a 3-way street from its start, for narrowing (prior_budget), or
        None (someone all-in at its start)."""
        from . import multiway_solver as MW
        players = [p for p in st.order if p in st.players_start]
        stacks = tuple(round(float(stk[p] - st.inv_start[p]), 2) for p in players)    # (each its own: side pots)
        if min(stacks) <= 1e-9:
            return None
        rs = [ranges.get(p, np.ones(N_COMBOS)) for p in players]
        mcfg = MW.MWConfig(rake_rate=self.cfg.rake_rate, rake_cap=self.cfg.rake_cap,
                           **{**self.mw_street_config.get(st.name, {}), **(self.mw_config or {})})
        model = getattr(self, "_icm", None)
        if model is not None:     # (a tournament's street narrows by its prize payoffs too: see _multiway_solve)
            from .icm import MWPayoffs
            mcfg.icm = MWPayoffs(model, players, {q: float(st.inv_start.get(q, 0.0)) for q in model.seats})
            mcfg.rake_rate = 0.0
        key = (tuple(board), tuple(_range_key(r) for r in rs), round(st.pot_start, 2), stacks, "start", repr(mcfg))
        sol = self._mw_cache.get(key)
        if sol is None:
            sol = self._mw_cache[key] = MW.MultiwaySolver(board, rs, st.pot_start,
                                                          stacks if len(set(stacks)) > 1 else stacks[0], cfg=mcfg)
        self._mw_run(sol, self.prior_budget - sol.build_seconds - sol.seconds)
        self._keep("_mw_cache", key, sol)
        gains = sol.br_gains() / sol.root_pot() * 100
        return sol, players, float(max(gains.mean(), 0.0))

    def _mw_actor_move(self, board, hero, hole, stk, streets, cur: StreetState, actor) -> Optional[dict]:
        """A practice opponent's mix in a 3-way pot, read off the 3-way solve the player practising was advised from
        earlier this street (its tree walked through the actions since its root: the root's own player acting where
        the hero acted, as solved), so one solve serves the hero's advice and the opponents' moves after it. None
        (play then asks from the opponent's seat) with no such solve, an action the tree doesn't have, a line the
        solve takes with under RARE_LINE of the acting player's range (its subtree barely converged), or a hand not
        in the solve."""
        from . import multiway_solver as MW
        acts = list(streets.get(cur.name) or [])
        for k in range(len(acts), -1, -1):
            hit = self._mw_lines.get(_mw_line(board, hero, hole, stk, streets, cur.name, k))
            if hit is not None:
                break
        else:
            return None
        sol, players = hit[:2]
        node, R = sol.root, sol.reach0.copy()
        for a in acts[k:]:
            if not isinstance(node, MW.Decision) or a.get("pos") not in players:
                return None
            p = players.index(a["pos"])
            if p != node.player:
                return None
            idx = _mw_pick(node, a)
            if idx is None:
                return None
            sg = node.avg_strategy()[0][idx]
            if (R[p, 0] * sg).sum() < RARE_LINE * max(R[p, 0].sum(), 1e-12):
                return None
            R = R.copy(); R[p] = R[p] * sg
            node = node.children[idx]
        seat, ahole = actor
        if not isinstance(node, MW.Decision) or seat not in players or node.player != players.index(seat):
            return None
        q = node.player
        ci = sol.combo_index(ahole)
        if ci is None:
            return None
        # the average strategy where the hand reaches the node; else (it would have played differently earlier, or
        # its range gives it no weight) the regret-matching one, which still learns from its counterfactual values
        mix = node.avg_strategy()[0][:, ci] if R[q, 0, ci] > 1e-12 and node.ssum[0, :, ci].sum() > 0 \
            else node.strategy()[0][:, ci]
        # (the seat's chips behind as solved: at most what another can match)
        names = _mw_names(node.names, node.amounts, stk[seat] - cur.inv_start[seat], sol.stacks[q], node.pot_now)
        return {"street": cur.name, "solved": True,
                "actions": [{"action": nm, "freq": round(float(mix[a]), 3),
                             "amount_bb": None if node.amounts[a] is None else round(float(node.amounts[a]), 2)}
                            for a, nm in enumerate(names)]}

    def _instant_answer(self, cur, board, hole, hero, villain, hero_p, oop, ip, eff, cfg, r0, r1, walked, behind, to_call,
                        n_players, t0, near=None) -> dict:
        """Advisor.instant's answer at the street advised on (heads-up from its start, its ranges narrowed through the
        earlier streets' solves as they stand): the first of instant_sources that reads one, else _NoInstant; and
        _NoInstant when the street's own solve is cached and its answer quick (settled, or run its budget)."""
        own = self._cache.get(self._solve_key(board, r0, r1, cur.pot_start, eff, cfg=cfg))
        if own is not None and not getattr(own, "stopped", False) and (
                getattr(own, "settled_at", None) is not None or own.iters >= 1000 or own.seconds >= 0.8 * self.time_budget):
            raise _NoInstant()
        earlier = False
        if own is None:   # (facing a size off the tree your advice on this street solved: that solve, at its nearest)
            c = self._hand_cfg(board, r0, r1, cur, oop, ip, eff, cfg)
            k = self._solve_key(board, r0, r1, cur.pot_start, eff, cfg=c)
            if c is not cfg and k not in self._presolved:   # (a presolve's, the move off its tree: never shown)
                own, earlier = self._cache.get(k), True
        spot = InstantSpot(cur.name, tuple(board), tuple(hole), hero_p, oop, ip, list(cur.actions), cur.pot_start, eff,
                           behind, own, walked.get(STREETS[STREETS.index(cur.name) - 1]), earlier, near)
        got = next((g for g in (src(self, spot) for src in self.instant_sources) if g), None)
        if got is None:
            raise _NoInstant()
        acts = got["actions"]
        best = max(acts, key=lambda a: a["freq"])["action"]
        bets = max(cur.bets[oop], cur.bets[ip])
        notes = [f"approximate, at once: {got['label']}; this {cur.name}'s own solve replaces it"]
        if got["source"] == "solve":   # (read off a solve cut to fit Quick: its game has fewer sizes, said plainly)
            self._trim_note(own, cur.name, notes)
        elif got["source"] == "prev":
            self._trim_note(spot.prev[0], STREETS[STREETS.index(cur.name) - 1], notes)
        return {"street": cur.name, "hero": hero, "villain": villain, "position": "OOP" if hero_p == 0 else "IP",
                "players": n_players, "hand": hand_class(hole), "board": cards_str(board),
                "pot_bb": round(cur.pot_start + cur.bets[oop] + cur.bets[ip], 2),
                "to_call_bb": round(min(to_call, behind), 2), "stack_bb": round(min(eff - bets, behind), 2),
                "actions": acts, "recommended": best,
                "approximate": {"source": got["source"], "label": got["label"], "iterations": got.get("iterations")},
                "notes": notes, "seconds": round(time.time() - t0, 3)}

    def _actor_move(self, sol: Solution, node: Decision, n_r0, n_r1, hole, p: int, street: str) -> dict:
        """The mix of player p (to act at `node`) for `hole`, from hero's solve; an error (play then asks
        from p's own seat) when hero's line this street is rare in the solve (its subtree barely converged:
        hero's advice solves again from there) or the hand has no weight."""
        if node.player != p:
            return {"error": "the tree expected the other player to act"}
        h0, h1 = (sol.r0, n_r0) if p == 1 else (sol.r1, n_r1)
        if h1.sum() < RARE_LINE * max(h0.sum(), 1e-12):
            return {"error": "the hero's line is rare in the solve"}
        idx = np.nonzero(sol.ctx.active == COMBO_INDEX[(min(hole), max(hole))])[0]
        reach = n_r0 if p == 0 else n_r1
        if not len(idx) or reach[idx[0]] <= 1e-9:
            return {"error": "the hand has no weight in the solve"}
        strat = node.avg_strategy0()[:, int(idx[0])]
        return {"street": street, "actions": [{"action": name, "freq": round(float(strat[a]), 3),
                                               "amount_bb": self._amount(name, node, p)}
                                              for a, name in enumerate(node.names)]}

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _with_hero(r0, r1, hero_p, hole, notes):
        """Our hand gets a negligible weight when the ranges say we'd never be here, so the solve
        still gives it a strategy and EVs (against the opponent's range) without changing the rest."""
        if not hole:   # (Study mode: no hand of ours)
            return r0, r1
        i = COMBO_INDEX[(min(hole), max(hole))]
        r = r0 if hero_p == 0 else r1
        if r[i] > 1e-9:
            return r0, r1
        r = r.copy(); r[i] = 1e-3
        notes.append("your hand isn't in your range for this line, so it was solved as a near-zero-weight combo")
        return (r, r1) if hero_p == 0 else (r0, r)

    def _as_advised(self, r0, r1, hero_p, hole):
        """The ranges an earlier street's advice solved with, for its narrowing: your hand in at a near-zero weight
        where its range there doesn't hold it (_with_hero), so the next street reads that solve (cached) instead of
        solving the street again without it (3.1 s of a limped river's 3.25 at Fast in the browser). Without it the solve
        is the same but for that one combo's weight (1e-3 of a combo's), and the walk takes it out again after."""
        return self._with_hero(r0, r1, hero_p, hole, [])

    def _from_library(self, lib, full, flop, r0, r1, st: StreetState, eff, cfg: Config, oop, ip, notes, **walk_kw):
        """(solution, node, r0, r1) walked from the library's solve of this flop (flop_library.LibrarySolution)
        when it holds the spot (the ranges entering the flop, pot, stack and tree) and every node of the line
        taken; else None. The walk's notes go to `notes` only when it is used."""
        try:
            sol = lib.solution(full[0], full[1], flop, r0, r1, st.pot_start, eff, cfg)
        except Exception:   # a library that can't be read is no library
            sol = None
        if sol is None:
            return None
        visited, n2 = [], []
        node, c0, c1 = self._walk(sol, st.actions, oop, ip, st.pot_start, eff, n2, st.name, visited=visited, **walk_kw)
        if not sol.covers(visited):
            return None
        notes.extend(n2)
        return sol, node, c0, c1

    def _solve_key(self, board, r0, r1, pot, stack, bets=(0.0, 0.0), to_act=0, n_raises=0, cfg=None):
        """(cache key, spot) of a heads-up solve: its ranges and the spot (_by_spot's key)."""
        cfg = cfg or self.cfg
        spot = (tuple(board), round(pot, 2), round(stack, 2), tuple(round(b, 2) for b in bets), to_act, n_raises,
                cfg.sizes_key()) + (((cfg.river_play, cfg.turn_classes, cfg.river_buckets, cfg.turn_buckets, cfg.river_strength),)
                                    if len(board) == 3 else ())
        # (a flop solve's river_play and turn_classes are part of the spot: a Fast flop solve, river checked down,
        # seeded an Accurate one at its lower c-bet frequencies and did worse in the same time than a cold start)
        return (_range_key(r0), _range_key(r1)) + spot

    def _store_seed(self, flop_walk, board) -> Optional[Callable]:
        """solve()'s `seed` for the turn's own solve after a library flop (STORE_SEED): the entry's stored turn on the
        line the flop's walk took (`flop_walk`: walked["flop"], (sol, node, taken)), or None where there is none (not
        a library flop, no turn file, a line it doesn't hold). The seed sets the regrets at the turn's first decision
        and in position after a check, each stored combo's; a node the file doesn't hold, a tree with no check there
        or one played on several boards is left as it is."""
        if flop_walk is None or len(board) != 4:
            return None
        sol, _, taken = flop_walk
        got = _turn_file(self.library, sol, board[:3])
        if got is None:
            return None
        t, smap = got
        if not any(list(ln.get("path", ())) == list(taken) for ln in t.get("lines", ())):
            return None
        from . import flop_library as fl

        def seed(root, s):
            if not isinstance(root, Decision) or root.player != 0 or "check" not in root.names \
                    or root.regret.shape[0] != 1:
                return
            act = s.ctx.active
            r0, r1 = s.ctx.root_reach
            k = root.names.index("check")
            share = None
            for at, nd in (((), root), (("check",), root.children[k])):
                if not isinstance(nd, Decision) or nd.regret.shape[0] != 1:
                    return
                st = fl.turn_strategies(t, taken, at, board[3], smap)
                if st is None or st["player"] != nd.player:
                    return
                mix = st["strat"][:, act]
                ok = ~np.isnan(mix[0])
                live = np.zeros((len(nd.names), len(act)))
                for a, name in enumerate(st["names"]):
                    live[match_action(nd.names, name)] += np.nan_to_num(mix[a])
                if share is None:
                    opp = r1
                    share = np.where(ok, live[k], 1.0 / len(nd.names))   # (OOP's check share: IP's opponent's reach)
                else:
                    opp = r0 * share
                mass = s.compat_sum(np.asarray(opp, dtype=np.float32)[None])[0]
                nd.regret[0][:, ok] = (STORE_SEED * root.pot_now * mass[ok] * live[:, ok]).astype(np.float32)
        return seed

    def _solve_cached(self, board, r0, r1, pot, stack, budget, bets=(0.0, 0.0), to_act=0, n_raises=0, cfg=None,
                      settle=False, seed=None):
        """A heads-up solve, from the cache when it holds this spot and ranges (run on while it has had under 80% of
        `budget`), else a new one, warm from warm_from's solve of the spot (else from `seed`: _store_seed). With
        hu_settled set it stops at the first of HU_CHECKS where it is that close to equilibrium; one settled already
        isn't run on. A solve stopped part way (should_stop: Cancelled) stays cached, and the spot asked again goes
        on from where it stopped."""
        cfg = cfg or self.cfg
        key = self._solve_key(board, r0, r1, pot, stack, bets, to_act, n_raises, cfg)
        spot = key[2:]
        sol = self._cache.get(key)
        target = self.hu_settled
        n = len(board)
        street = ("flop" if n == 3 else "river" if n == 5 else
                  "shallow" if stack <= self.spr_sizes.get("turn", (0.0, {}))[0] * pot else "turn")
        first = max(self.hu_settled_from.get(street, HU_CHECKS[0]), HU_CHECKS[0])   # (the first check it may stop at)
        t0 = time.time()
        if self._instant:   # (as it stands: an instant answer solves nothing)
            if sol is None:
                raise _NoInstant()
            return sol
        if self._runout_held:   # (a runout report from what's in hand: a solve cached already, as it stands)
            if sol is None:
                raise _NotHeld()
            return sol
        if sol is None:
            # Same spot solved by warm_from (ranges can differ a little: they come from its own earlier
            # streets): start from it and count its time against the budget, as extend() does below.
            warm = self.warm_from._by_spot.get(spot) if self.warm_from is not None else None
            # (the budget covers the tree's build and the exploitability read after the iterations: the browser's
            # Accurate flop answered 4.5 s past its 60 s, a two-tone one 9 s)
            left = max(budget - warm.seconds, 0.2 * budget) if warm else budget
            cap = self._cells_cap(n, left)
            level, full, use, slow = 0, None, cfg, False
            while True:   # (a tree past the cap is cut, TRIMS, until it fits)
                rest = max(left - (time.time() - t0), 0.0)
                try:
                    if cap is not None:   # (counted first: a tree past the cap costs a walk, not its arrays)
                        tree_cells(board, r0, r1, pot, stack, bets=bets, to_act=to_act, n_raises=n_raises, cfg=use,
                                   max_cells=cap)
                    sol = solve(board, r0, r1, pot, stack, bets=bets, to_act=to_act, n_raises=n_raises, cfg=use,
                                time_budget=rest, should_stop=self.should_stop, warm=warm, reserve=HU_RESERVE,
                                max_cells=cap, max_iters=min(c for c in HU_CHECKS if c >= first) if target is not None
                                else 1000, horizon=1000, seed=seed,
                                max_per=SLOW * left / 10 if n >= 4 and not (warm or slow) and level < len(TRIMS)
                                else None)
                    break
                except Cancelled as e:
                    # (stopped before its first check: kept all the same, its tree built and its iterations run, so
                    # the spot asked again goes on from there, as one stopped later does; a presolve's mostly are)
                    part = getattr(e, "sol", None)
                    if part is not None:
                        if level:
                            part.trimmed = (level, full, cap)
                        if warm is not None:
                            part.seconds += warm.seconds
                        part.stopped = True
                        self._cache[key] = part
                        self._by_spot[spot] = part
                        self._keep("_cache", key, part)
                        if self._presolving:
                            self._presolved.add(key)
                    raise
                except (TreeTooBig, TooSlow) as e:
                    full = full or e.cells
                    if isinstance(e, TooSlow):
                        # (this device is slower than cell_s: its first iteration says by how much)
                        self.cell_s, slow = e.per / max(e.cells, 1), True
                        cap = min(cap or e.cells, self._cells_cap(n, max(left - (time.time() - t0), 0.5 * left)))
                    level += 1
                    if level > len(TRIMS):
                        raise
                    if level == len(TRIMS):
                        cap = self.max_cells   # (the smallest tree runs whatever its time: within memory)
                    use = self._trim_cfg(cfg, ("flop", "turn", "river")[n - 3], level)
            if level:
                sol.trimmed = (level, full, cap)
            per = getattr(sol.solver, "per_iter", None)
            if n >= 4 and per and getattr(sol, "cells", 0) > 1e6:
                # (this device's speed, as it runs now; a small tree's iteration is mostly its other costs, so a cell
                # looks dearer there than CELL_S: it doesn't raise cell_s, which only TooSlow does)
                self.cell_s = 0.5 * self.cell_s + 0.5 * min(per / sol.cells, CELL_S)
            if warm is not None:
                sol.seconds += warm.seconds
            self._cache[key] = sol
            self._by_spot[spot] = sol
            self._keep("_cache", key, sol)
            if self._presolving:
                self._presolved.add(key)
            if target is not None:
                self._hu_run(sol, left - (time.time() - t0), target, first)
        else:
            self._keep("_cache", key, None)
            if not self._presolving:
                self._presolved.discard(key)   # (asked for by a request of the hand's own: its solve now)
            if getattr(sol, "stopped", False):   # (stopped part way: its iterations and root mix as they are now)
                sol.stopped = False
                sol.iters = sol.solver.iters
                sol._refresh()
            if getattr(sol, "settled_at", None) is not None and (target is None or sol.settled_at <= target):
                pass        # (settled already)
            elif sol.seconds < budget * 0.8 and sol.iters < 1000:
                if target is not None:
                    self._hu_run(sol, budget - sol.seconds, target, first)
                else:       # keep going from the stored regrets instead of restarting
                    try:
                        sol.extend(budget - sol.seconds, should_stop=self.should_stop, reserve=HU_RESERVE)
                    except Cancelled:
                        sol.stopped = True
                        raise
        if settle:
            self._settle(sol, budget)
        return sol

    @staticmethod
    def _expl(sol: Solution) -> float:
        """sol.exploitability(), read once per iteration count."""
        if getattr(sol, "expl_at", (None,))[0] != sol.iters:
            sol.expl_at = (sol.iters, sol.exploitability())
        return sol.expl_at[1]

    def _hu_run(self, sol: Solution, left: float, target: float, first: int = HU_CHECKS[0]) -> None:
        """Runs a heads-up solve on for at most `left` seconds, stopping at the first of HU_CHECKS from `first` on
        where its exploitability is under `target` (% of the pot): sol.settled_at = target then. (Its pieces run the
        solver directly: Solution.extend's EV pass after each would cost about an iteration a piece.)"""
        t0 = time.time()
        checks = [c for c in HU_CHECKS if c >= first]
        while sol.iters < 1000:
            if sol.iters in checks and self._expl(sol) < target:
                sol.settled_at = target
                break
            rem = left - (time.time() - t0)
            nxt = min([c for c in checks if c > sol.iters] + [1000])
            if rem <= 0:
                break
            t1 = time.time()
            try:
                _, secs = sol.solver.run(sol.r0, sol.r1, time_budget=rem, max_iters=nxt, min_iters=1,
                                         should_stop=self.should_stop, reserve=HU_RESERVE)
            except Cancelled:
                sol.stopped = True
                sol.seconds += time.time() - t1   # (its iterations ran: the budget asked again counts them)
                raise
            sol.iters = sol.solver.iters
            sol.seconds += secs
            if sol.iters < nxt:
                break       # (out of time)
        if not sol.fresh():     # (settled where it stood, as solve() left it: its root's EVs already worked out)
            sol._refresh()

    def _keep(self, store: str, key, sol) -> None:
        """A solve in self.<store> (_cache or _mw_cache) just used: the most recent now, its arrays counted when `sol`
        is given (a new or grown one); then the least recently used go while they hold more than cache_mb."""
        k = (self, store, key)
        held = self._held.pop(k, None)
        self._held[k] = _held_bytes(sol) if sol is not None else (held or 0)
        while len(self._held) > CACHE_KEEP and sum(self._held.values()) > self.cache_mb * 2 ** 20:
            adv, st, old = next(iter(self._held))
            del self._held[(adv, st, old)]
            gone = getattr(adv, st).pop(old, None)
            # (the spot and line indexes hold the same solve: drop theirs too, or it stays in memory)
            for d, of in ((adv._by_spot, lambda v: v), (adv._mw_lines, lambda v: v[0])):
                for s in [s for s, v in d.items() if of(v) is gone]:
                    del d[s]

    def _settle(self, sol: Solution, budget: Optional[float] = None) -> bool:
        """The street being advised on, far from settled after its budget (SETTLE_EXPL): one more budget,
        once. Limped pots, turn probes and short-stack check-raises stopped at 10 iterations 27-55% of the
        pot from settled, showing confident mixes (an all-in 75% at +15 beside a call at +22). Whether it ran."""
        if getattr(self, "_quick", False) or self._runout_held or getattr(sol, "settled_once", False) \
                or sol.iters >= 1000 or self._expl(sol) < SETTLE_EXPL:
            return False
        sol.settled_once = True
        try:
            sol.extend(self.time_budget if budget is None else budget, should_stop=self.should_stop, reserve=HU_RESERVE)
        except Cancelled:
            sol.stopped = True
            raise
        return True

    def _one_size_view(self, sol, node, n_r0, n_r1, cur: StreetState, board, hole, hero, villain, hero_p, oop, ip, eff,
                       exploit, notes, t0) -> dict:
        """The one-size view (state "one_size"), in place of the advice at a heads-up decision where the player to act
        (hero; Study's actor) can bet: the street solved again from this decision, with the ranges the advice's walk
        reached it with (Study's and the advice's own), once with one bet size there for each candidate
        (one_size_candidates: ONE_SIZES, the river's ONE_SIZES_RIVER, all-in where the tree offers it) and once with
        every candidate to mix freely (the full menu, "full"); the rest of the tree (the opponent's bets, raises, the
        later streets) is the advice's (_cfg_with_real_sizes), cut alike for every size when the full menu's tree is
        past the advice's cap. Each solve stops once settled (one_size_settled) or at the request's budget. A row per
        solve: the root's actions and the range's frequency of each, the range's EV (bb relative to folding now,
        weighted by hand pairs: the game's value), the grid, hero's hand (with cards), its iterations and
        exploitability. "one_size": "all" solves the full menu and every size and adds each one's EV given up against
        the full menu (loss_pct, % of the pot) and the best (one_size_summary); "full" or a size ("50%", 50, "all-in";
        one that isn't a candidate is solved too, 5-500%) solves that one; "list" none. Rows solved here are kept
        (ONE_SIZE_CACHE), so asking again answers at once, and a request answers with every row it has kept for the
        spot ("rows"). At equilibrium only: not against the pool or locks, nor with tournament payouts."""
        want = self._one_size
        street = cur.name
        if exploit:
            return {"error": "the one-size view is at equilibrium: it isn't worked out against the pool or your locks"}
        if getattr(self, "_icm", None) is not None:
            return {"error": "the one-size view is in chips: it isn't worked out with tournament payouts or bounties"}
        if node.player != hero_p:
            return {"error": "no decision of yours here"}
        if "check" not in node.names:
            return {"error": "the one-size view is for a decision where you can bet: here you face a bet (and raise "
                             "sizes aren't part of it)"}
        bets = tuple(float(b) for b in node.bets)
        behind = eff - bets[hero_p]
        if behind <= 1e-9:
            return {"error": "no chips behind to bet"}
        pot = float(node.pot_now)
        base = self._cfg_with_real_sizes(cur, oop, ip, eff)
        base.keep_sizes, base.root_allin, base.root_bets = None, False, None
        cands = one_size_candidates(pot, behind, base.allin_spr, river=street == "river")
        frac = dict(cands)
        if want not in ("all", "full", "list"):
            try:
                pct = float(str(want).strip().rstrip("%")) if str(want).strip() != "all-in" else None
            except ValueError:
                return {"error": f"one_size: {want!r} is not a size (\"50%\", 50, \"all-in\", \"full\", \"all\" or \"list\")"}
            if pct is None:
                want = "all-in"
                frac.setdefault("all-in", ONE_SIZE_ALLIN)
            else:
                if not SIZE_PCT[0] <= pct <= SIZE_PCT[1]:
                    return {"error": f"one_size: sizes are {SIZE_PCT[0]:g}-{SIZE_PCT[1]:g}% of the pot"}
                f = pct / 100.0
                amt = max(round(f * pot * 2) / 2, 1.0)
                want = "all-in" if amt >= behind - 1e-9 else f"{int(round(pct))}%"
                frac.setdefault(want, f)
        todo = (["full"] + [c for c, _ in cands]) if want == "all" else [] if want == "list" else [want]
        R0, R1 = expand(sol.ctx, n_r0), expand(sol.ctx, n_r1)
        target = self.one_size_settled[street] if self.expl_target is None else \
            min(self.one_size_settled[street], self.expl_target)
        spot = (self._solve_key(board, R0, R1, cur.pot_start, eff, bets, hero_p, cur.n_raises, base), target)
        full_menu = [frac[c] for c, _ in cands]
        level = self._one_size_trim(board, R0, R1, cur, eff, bets, hero_p, base, full_menu)
        cfg0 = base if not level else self._trim_cfg(base, street, level)
        for label in todo:
            key = (spot, label)
            if key in self._one_rows:
                self._one_rows[key] = self._one_rows.pop(key)   # (the most recently used last)
                continue
            cfg = cfg0.copy()
            cfg.root_bets = full_menu if label == "full" else [frac[label]]
            t1 = time.time()
            s1 = self._one_size_solve(board, R0, R1, cur, eff, bets, hero_p, cfg, target)
            row = self._one_size_row(s1, label, hero_p, hole, time.time() - t1)
            del s1
            self._one_rows[key] = row
            while len(self._one_rows) > ONE_SIZE_CACHE:
                del self._one_rows[next(iter(self._one_rows))]
        known = {lab: r for (sp, lab), r in self._one_rows.items() if sp == spot}
        labels = [c for c, _ in cands]
        rows = [known[c] for c in labels if c in known] + \
            [r for lab, r in known.items() if lab not in labels and lab != "full"]   # (sizes asked that aren't candidates)
        full = known.get("full")
        out = {"street": street, "actor": hero, "opponent": villain, "position": "OOP" if hero_p == 0 else "IP",
               "board": cards_str(board), "pot_bb": round(pot, 2), "stack_bb": round(behind, 2),
               "candidates": [c for c, _ in cands], "asked": want, "definition": ONE_SIZE_DEF, "target_pct": target,
               "full": full, **one_size_summary(full, rows, pot)}
        if hole:
            out["hand"] = hand_class(hole)
            out["hole"] = cards_str(hole)
        more = []
        if level:
            more.append(f"{street}: every size's tree is cut alike to fit this speed's time and memory: {TRIMS[level - 1]}")
        rest = base.bet_sizes.get(street, [])
        more.append(f"{street}: one size at {'your' if hole else hero + chr(39) + 's'} decision only; the rest of the tree "
                    f"is the advice's ({villain}'s bets {', '.join(f'{int(round(x * 100))}%' for x in rest)}"
                    f"{', and the later streets' if street != 'river' else ''})")
        return {"one_size": out, "street": street, "hero": hero, "villain": villain,
                "notes": [n for n in notes if "to tree action" not in n] + more, "seconds": round(time.time() - t0, 2)}

    def _one_size_trim(self, board, R0, R1, cur: StreetState, eff, bets, to_act, base: Config, full_menu) -> int:
        """The TRIMS level the one-size view's trees are cut to: the full menu's tree (the biggest) at the cap a new
        solve of the street has here (_cells_cap: memory, and on the turn and river the request's time), else 0."""
        cap = self._cells_cap(len(board), self.time_budget)
        if cap is None:
            return 0
        for level in range(len(TRIMS) + 1):
            cfg = base if not level else self._trim_cfg(base, cur.name, level)
            cfg = cfg.copy()
            cfg.root_bets = list(full_menu)
            try:
                tree_cells(board, R0, R1, cur.pot_start, eff, bets=bets, to_act=to_act, n_raises=cur.n_raises,
                           cfg=cfg, max_cells=cap if level < len(TRIMS) else self.max_cells)
                return level
            except TreeTooBig:
                continue
        return len(TRIMS)

    def _one_size_solve(self, board, R0, R1, cur: StreetState, eff, bets, to_act, cfg: Config, target: float):
        """A one-size view's solve from the decision: settled at `target` (% of the pot, read at HU_CHECKS) or the
        request's budget, whichever comes first. Not kept in the advice's cache (_cache): the view keeps its rows."""
        t0 = time.time()
        sol = solve(board, R0, R1, cur.pot_start, eff, bets=bets, to_act=to_act, n_raises=cur.n_raises, cfg=cfg,
                    time_budget=self.time_budget, max_iters=HU_CHECKS[0], horizon=1000, should_stop=self.should_stop,
                    reserve=HU_RESERVE)
        self._hu_run(sol, self.time_budget - (time.time() - t0), target, HU_CHECKS[0])
        return sol

    def _one_size_row(self, sol: Solution, label: str, p: int, hole, secs: float) -> dict:
        """A one-size solve's row (_one_size_view) for the player p acting at its root."""
        root, ctx = sol.root, sol.ctx
        names, act = list(root.names), ctx.active
        A = len(names)
        w = np.asarray(sol.r0 if p == 0 else sol.r1, dtype=np.float64)
        o = sol.r1 if p == 0 else sol.r0
        cs = np.asarray(sol.solver.compat_sum(np.asarray(o, dtype=np.float32)[None, :])[0], dtype=np.float64)
        if not sol.fresh():
            sol._refresh()
        strat, evs = sol.strategy[:, act], sol.evs[:, act]
        pw = w * cs
        ev = float((pw * (strat * evs).sum(axis=0)).sum() / max(float(pw.sum()), 1e-12))
        freq = (strat * w[None, :]).sum(axis=1) / max(float(w.sum()), 1e-12)
        sf = np.zeros((A, N_COMBOS)); sf[:, act] = strat
        row = {"size": label, "actions": names, "freq": [round(float(f), 4) for f in freq],
               "amounts": [self._amount(n, root, p) for n in names], "ev_bb": round(ev, 4),
               "grid": strategy_grid([sf[a] for a in range(A)], names, expand(ctx, w)),
               "iterations": int(sol.iters), "exploitability_pct": round(max(0.0, float(self._expl(sol))), 3),
               "seconds": round(secs, 2)}
        if hole:
            ci = np.nonzero(act == COMBO_INDEX[(min(hole), max(hole))])[0]
            if len(ci):
                row["hand"] = {"freq": [round(float(x), 3) for x in strat[:, ci[0]]],
                               "ev": [round(float(x), 2) for x in evs[:, ci[0]]]}
        return row

    def _optimize_sizes(self, board, r0, r1, st: StreetState, eff, cfg: Config, hero_p, notes):
        """optimize_sizes: solve this street once with OPT_SIZES (one street deep and one raise: the size is
        chosen here, and the smaller tree buys iterations), keep the sizes either player's first bet uses
        with OPT_KEEP of their range (the MAX_SIZES most used), and return the advice's config with them and
        hero's mix there by size. The two solves split the time budget, so the total stays the same. Sizes the
        request set for the street (state "sizes") join the wide menu, so they compete with it rather than being
        silently replaced: kept if the first solve uses them."""
        mine_sizes = list(self._req_sizes.get(st.name) or [])
        menu = {f"bet {int(round(f * 100))}%": f for f in sorted(set(OPT_SIZES) | set(mine_sizes))}
        wide = cfg.copy()
        wide.bet_sizes[st.name] = sorted(menu.values())
        wide.depth, wide.max_raises = 1, 1
        sol = self._solve_cached(board, r0, r1, st.pot_start, eff, OPT_SHARE * self.time_budget, cfg=wide)
        self.time_budget *= 1 - OPT_SHARE   # (restored by _advise)
        root = sol.root
        chk = root.children[root.names.index("check")]
        firsts = [(root, sol.r0)] + ([(chk, sol.r1)] if isinstance(chk, Decision) else [])
        mixes = [dict(zip(n.names, n.avg_strategy0() @ (r / max(r.sum(), 1e-12)))) for n, r in firsts]
        frac = dict(menu)
        frac["all-in"] = round(eff / max(st.pot_start, 1e-9) + 0.01, 3)   # a bet that reaches the stack
        use: Dict[str, float] = {}
        for m in mixes:
            for n, f in m.items():
                if n != "check":
                    use[n] = max(use.get(n, 0.0), float(f))
        top = sorted(use, key=use.get, reverse=True)[:MAX_SIZES]
        keep = sorted([n for n in top if use[n] >= OPT_KEEP] or top[:1], key=frac.get)
        out = cfg.copy()
        out.bet_sizes[st.name] = sorted(frac[n] for n in keep)
        label = lambda n: n[4:] if n.startswith("bet ") else n   # "bet 33%" -> "33%"
        mine = mixes[min(hero_p, len(mixes) - 1)]
        custom = [f"{int(round(f * 100))}%" for f in sorted(mine_sizes)]
        notes.append(f"{st.name}: bet sizes chosen by a first solve over {', '.join(label(n) for n in root.names[1:])}"
                     + (f" (your {st.name} sizes {', '.join(custom)} among them)" if custom else "")
                     + f": kept {', '.join(label(n) for n in keep)} (used by {OPT_KEEP:.0%}+ of a range)")
        return out, {"size_usage": {label(n): round(float(f), 3) for n, f in mine.items()},
                     "sizes_kept": [label(n) for n in keep], **({"sizes_custom": custom} if custom else {})}

    def _hand_cfg(self, board, r0, r1, st: StreetState, oop: str, ip: str, stack: float, cfg: Config, adjust=None):
        """The config a solve of this street the hand made already was built with, when `cfg` (the street's real sizes,
        _cfg_with_real_sizes) has none cached: the hand's advice on the street before a later action took a size off
        that tree (a turn bet of 50% where its advice solved 33/75/125%) is the same street, its ranges, and settled, and
        its walk reads the action at the nearest size; a second solve of it with the size in (the river's narrowing:
        3 s of a limped river's 3.2, 38 iterations, where the advice's was settled at 200) would be shorter and worse.
        The longest earlier line's first; `adjust` (Config -> Config) as the caller's cfg was; else `cfg`."""
        adjust = adjust or (lambda c: c)
        if self._cache.get(self._solve_key(board, r0, r1, st.pot_start, stack, cfg=cfg)) is not None:
            return cfg
        for k in range(len(st.actions) - 1, -1, -1):
            c = adjust(self._cfg_with_real_sizes(dataclasses.replace(st, actions=st.actions[:k]), oop, ip, stack))
            if c.sizes_key() != cfg.sizes_key() and \
                    self._cache.get(self._solve_key(board, r0, r1, st.pot_start, stack, cfg=c)) is not None:
                return c
        return cfg

    def _cfg_with_real_sizes(self, st: StreetState, oop: str, ip: str, stack: float) -> Config:
        """Tree config for this street where the bet/raise sizes actually taken replace the
        nearest abstract size (only when they differ by more than 15%), so range narrowing
        uses the real action instead of snapping it to the abstraction. A size already taken
        this street is never swapped out by a later one, an all-in deeper than the tree's
        all-in SPR gets a size that reaches the stack, and the raise cap covers the raises made."""
        cfg = self.cfg.copy()
        for kind, v in self.root_sizes.get(st.name, {}).items():
            (cfg.bet_sizes if kind == "bet" else cfg.raise_sizes)[st.name] = list(v)
        max_spr, shallow = self.spr_sizes.get(st.name, (0.0, {}))
        if stack <= max_spr * st.pot_start:
            for kind, v in shallow.items():   # (a street's name: its bets inside this street's solve)
                (cfg.raise_sizes if kind == "raise" else cfg.bet_sizes)[st.name if kind in ("bet", "raise") else kind] = list(v)
        bets = [0.0, 0.0]
        pot = st.pot_start
        fixed = {"bet": set(), "raise": set()}
        n_aggr = 0
        for a in st.actions:
            if a["pos"] not in (oop, ip):
                continue
            p = 0 if a["pos"] == oop else 1
            act = a["action"]
            pot_total = pot + bets[0] + bets[1]
            amt = a.get("amount") or a.get("to") if act == "bet" else a.get("to") or a.get("amount")
            if act in ("bet", "raise") and amt and (bets[p] + amt if act == "bet" else amt) >= stack - 1e-9:
                act = "all-in"   # a bet or raise of the whole stack
            if act == "bet" and amt:
                _swap_nearest(cfg.bet_sizes[st.name], amt / max(pot_total, 1e-9), fixed=fixed["bet"])
                bets[p] = amt
            elif act == "raise" and amt:
                call_amt = bets[1 - p] - bets[p]
                _swap_nearest(cfg.raise_sizes[st.name], (amt - bets[1 - p]) / max(pot_total + call_amt, 1e-9),
                              fixed=fixed["raise"])
                bets[p] = amt
            elif act == "call":
                bets[p] = min(bets[1 - p], stack)
            elif act == "all-in":
                # the tree offers all-in only at low SPR; a deeper jam would map to the biggest bet,
                # so swap in a size that reaches the stack (half a bb over, past the tree's rounding)
                if bets[1 - p] <= bets[p]:
                    _swap_nearest(cfg.bet_sizes[st.name], (stack - bets[p] + 0.5) / max(pot_total, 1e-9),
                                  fixed=fixed["bet"])
                else:
                    call_amt = bets[1 - p] - bets[p]
                    _swap_nearest(cfg.raise_sizes[st.name],
                                  (stack - bets[1 - p] + 0.5) / max(pot_total + call_amt, 1e-9), fixed=fixed["raise"])
                bets[p] = stack
            if act in ("bet", "raise", "all-in"):
                n_aggr += 1
        # past the cap the tree offers only all-in, so a 3bet that isn't one would be read as a jam
        cfg.max_raises = max(cfg.max_raises, n_aggr)
        cfg.keep_sizes = {"bet": [cfg.bet_sizes[st.name][i] for i in sorted(fixed["bet"])],
                          "raise": [cfg.raise_sizes[st.name][i] for i in sorted(fixed["raise"])], "aggr": n_aggr}
        return self._icm_cfg(cfg, st, oop, ip)

    @staticmethod
    def _trim_note(sol: Solution, street: str, notes: List[str]) -> None:
        """The note for a solve whose tree was cut to fit (_solve_cached: sol.trimmed)."""
        if getattr(sol, "trimmed", None):
            level, full, _ = sol.trimmed
            notes.append(f"{street}: solved in a smaller game than the full one, to fit this speed's time and memory: "
                         f"{TRIMS[level - 1]} (the sizes your hand has used are kept; the full game is {full / 1e6:.1f}M+ "
                         f"cells), so the sizes offered are fewer")

    @staticmethod
    def _trim_cfg(cfg: Config, street: str, level: int) -> Config:
        """cfg cut for a tree past its cap (TRIMS[level - 1]): 1, every street's raises at one size; 2, also the later
        streets' bets; 3, also this street's bets; 4, also one raise a street. The size kept is the one nearest 75%
        pot, or the ones the hand's actions took this street (Config.keep_sizes)."""
        c = cfg.copy()
        keep = cfg.keep_sizes or {}
        order = ("flop", "turn", "river")

        def one(sizes, kept):
            return sorted(set(kept)) if kept else [min(sizes, key=lambda x: abs(x - 0.75))] if sizes else sizes
        if level >= 1:
            for st in c.raise_sizes:
                c.raise_sizes[st] = one(c.raise_sizes[st], keep.get("raise") if st == street else None)
        if level >= 2:
            for st in c.bet_sizes:
                if st in order and order.index(st) > order.index(street):
                    c.bet_sizes[st] = one(c.bet_sizes[st], None)
        if level >= 3 and street in c.bet_sizes:
            c.bet_sizes[street] = one(c.bet_sizes[street], keep.get("bet"))
        if level >= 4:
            c.max_raises = max(1, keep.get("aggr", 0))
        return c

    def _cells_cap(self, n_board: int, budget: float) -> Optional[int]:
        """A new heads-up solve's most cells (MAX_CELLS): max_cells, and on the turn and river the tree whose CAP_ITERS
        iterations at cell_s fit in `budget`."""
        caps = [c for c in (self.max_cells,) if c]
        if n_board >= 4 and self.cell_s > 0:
            caps.append(int(budget / (CAP_ITERS * self.cell_s)))
        return min(caps) if caps else None

    def _icm_cfg(self, cfg: Config, st: StreetState, oop: str, ip: str, bets=(0.0, 0.0)) -> Config:
        """cfg with a tournament's payoffs (Config.icm: gto/icm.py HUPayoffs) for a heads-up solve of oop and ip from
        street st's start (bets: theirs at the solve's root, when it starts mid-street; everyone's bets this street
        are then in the pot), or cfg itself without a tournament."""
        model = getattr(self, "_icm", None)
        if model is None:
            return cfg
        from .icm import HUPayoffs
        mid = any(b > 1e-9 for b in bets)
        contrib = {p: st.inv_start[p] + (st.bets[p] if mid else 0.0) for p in st.inv_start}
        inv0 = (st.pot_start / 2 + bets[0], st.pot_start / 2 + bets[1])
        cfg = cfg.copy()
        cfg.icm = HUPayoffs(model, (oop, ip), contrib, inv0)
        cfg.rake_rate = 0.0
        return cfg

    def _walk(self, sol: Solution, acts: List[dict], oop, ip, pot, stack, notes, street, smoothing=0.0, only=None,
              widened=None, board=None, rules=None, lock=None, rare_lines=None, visited=None, trust=1.0, slow=None,
              taken=None):
        """Follow real actions through the solved tree, narrowing ranges. Smoothing mixes each
        strategy with uniform so a line the solve rarely takes doesn't shrink a range to nothing.
        With `only` (0 = OOP, 1 = IP) and `board`, that player's actions are exact unless the solve
        takes one with less than RARE_LINE of their range: then their range for it comes from the
        multiway betting rules instead (see _rule_mask), and their seat is appended to `widened`; a raise
        over a bet keeps the solve's range (its strongest hands). Every such rare line of `only`'s is
        appended to `rare_lines`.
        `rules` (players, with `board`) gives the same fallback on an earlier street, where the other
        actions stay smoothed: a short prior solve can be far from equilibrium, and a line it rarely
        takes (a turn call it folds two pair to) would leave the range a sliver.
        `lock` (exploit mode) narrows its player's range by how the pool plays instead of the solve.
        `visited` (a list) gets every node an action was taken at; `taken` the tree's names of the actions taken.
        `trust` (with `rules` and `board`): the weight the solve's strategy gets on `rules`' other actions,
        the rest going to the rules' range for the action (softened, as multiway), for a solve that hasn't
        settled (trust_of). A call by the rules keeps the hands they have raising at SLOWPLAY when the player
        could have raised (_slowplayed; `slow`, {player: mask} carried from street to street, as multiway)."""
        rules = (only,) if rules is None and only is not None else rules or ()
        node = sol.root
        r0, r1 = sol.r0.copy(), sol.r1.copy()
        bets = [0.0, 0.0]
        for a in acts:
            if not isinstance(node, Decision):
                break
            if a["pos"] not in (oop, ip):
                continue
            p = 0 if a["pos"] == oop else 1
            if p != node.player:
                notes.append(f"{street}: action order mismatch at {a['pos']} {a['action']}; ignoring")
                continue
            if visited is not None:
                visited.append(node)
            name = self._tree_action_name(node, a, p, bets, pot, stack)
            k = match_action(node.names, name)
            rp = r0 if p == 0 else r1
            if lock is not None and p == lock.player:
                s = lock(node, node.avg_strategy(), rp[None, :])[0][k]
            else:
                s = node.avg_strategy0()[k]
            rare = (rp * s).sum() < RARE_LINE * max(rp.sum(), 1e-12)
            # A raise over a bet that the solve rarely makes is its strongest hands: the rules' raising range
            # (bluffs, and off-rule hands up to OFF_RULE_MAX) read a turn min-reraise on KQT9 as 99, 66 and
            # air where the solve had straights, sets and two pair, and AQ called it. Kept as solved.
            # So is a bet of the pot or more: the rules balance it with bluffs at b / (p + b), about one
            # per value hand for a jam, and a 99bb river jam into 2bb read as 53s, J4s, 43s was called by
            # a hand playing the board.
            amt0 = a.get("to") if a.get("to") is not None else a.get("amount")
            add0 = (stack - bets[p] if a["action"] == "all-in" else float(amt0) - bets[p] if amt0 is not None else 0.0)
            reraise = a["action"] in ("raise", "all-in", "bet") and (
                max(bets) > bets[p] + 1e-9 or add0 >= pot + bets[0] + bets[1] - 1e-9)
            if rare and rare_lines is not None and p == only and a["action"] != "fold":
                rare_lines.append(a["pos"])
            # (a rare raise or big bet keeps the solve's hands and adds the rules' value: kept alone, a sliver
            # from a short solve decided the answer, 0.7 combos of 65s for a river overbet, and a flop jam's
            # range just above a combo threshold had AK calling it at +48 bb one run in four)
            # facing a bet with chips behind the call (as multiway's raise_ok: by the real bets, not the tree's, where an
            # all-in can map to a bet with raises under it); facing an all-in the rules' call keeps its strongest hands
            can_raise = bets[p] + 1e-9 < max(bets) < stack - 1e-9
            def rule_mask():
                # the part of p's range the betting rules play this way (see _rule_mask), and that range
                sc, rank, bluff = _street_info(tuple(board))
                cur_bet, pot_total = max(bets), pot + bets[0] + bets[1]
                act_ = a["action"]
                amt = a.get("to") if a.get("to") is not None else a.get("amount")
                to = (stack if act_ == "all-in" else float(amt) if act_ in ("bet", "raise") and amt is not None
                      else node.children[k].bets[p] if act_ in ("bet", "raise") and isinstance(node.children[k], Decision)
                      else cur_bet)
                full = expand(sol.ctx, rp) * (sc >= 0)
                other = expand(sol.ctx, r1 if p == 0 else r0) * (sc >= 0)
                return full, rank, _rule_mask(full, to, cur_bet, max(0.0, cur_bet - bets[p]), pot_total, 1, rank,
                                              bluff=bluff, others=[other], raise_ok=can_raise, sc=sc)
            def rule_weights(full, rank, m):
                # the rules' weights for the action (_soften); a call that could have raised keeps the hands
                # they have raising at SLOWPLAY, as multiway (heads-up they were cut to NARROW_SOFT)
                w = _soften(full, m)
                if a["action"] == "call" and can_raise:
                    w = _slowplayed(full, w, _keep(max(bets) - bets[p], pot + bets[0] + bets[1], 1), rank, slow, p)
                return w
            def as_locked(t):
                # weights t for this action as the locked player (exploit mode's pool, your locks) plays them: the
                # node's other actions share the rest as the solve has them, and the lock moves hands between them
                # (the rules' range for a rare river bet, balanced with bluffs, had the pool bluffing as often as
                # equilibrium, and the best response called it as often)
                if lock is None or p != lock.player:
                    return t
                base = node.avg_strategy0().astype(np.float64)
                rest = base.sum(axis=0) - base[k]
                n_act = len(node.names)
                strat = np.where(rest > 1e-12, base / np.maximum(rest, 1e-12) * (1.0 - t), (1.0 - t) / max(n_act - 1, 1))
                strat[k] = t
                return lock(node, strat[None].astype(np.float32), rp[None, :])[0][k]
            if p in rules and rare and board is not None and a["action"] != "fold":
                if widened is not None and p == only:
                    widened.append(a["pos"])
                notes.append(f"{street}: the solve rarely has {a['pos']} {a['action']} here "
                             f"({(rp * s).sum() / max(rp.sum(), 1e-12) * 100:.0f}% of their range), so their range "
                             + ("comes from simple betting rules instead" if not reraise else
                                "is the solve's with the rules' value hands added"))
                full, rank, m = rule_mask()
                if reraise:
                    # no bluffs: as many combos as the rule's range, from the top (a 99bb river jam into 2bb kept
                    # 0 combos as solved, and as the rules' about half bluffs a hand playing the board called it)
                    # The solve's own hands are those it still plays this way now (its current strategy) as well
                    # as on average: an average from a short solve keeps the uniform start of its first
                    # iterations, which for a line it rarely takes is most of it (14 iterations in, K8o and
                    # KQo jammed a K72 flop 18% on average and 0% currently, so AK called the jam at +6 to +19 bb
                    # on short budgets and folded at -67 on long ones).
                    if lock is None or p != lock.player:
                        s = np.minimum(s, node.strategy()[0][k])
                    s = np.maximum(s, _top(full, _share(full, m), rank)[sol.ctx.active])   # with the solve's own
                else:
                    s = as_locked(rule_weights(full, rank, m)[sol.ctx.active])
            else:
                if trust < 1.0 and p in rules and board is not None and a["action"] != "fold":
                    # an unsettled earlier-street solve: its strategy counts by how settled it is, the rest is
                    # the betting rules' range for the line (see trust_of). A check keeps the rest whole: heads-up
                    # the big blind checks 95-99% of flops to the raiser, where the rules' check (everything but
                    # what would bet) took its value out, and after a 10-iteration flop and turn (88% and 51% of
                    # the pot from equilibrium, trust 0) a K8372 river checked through had the button betting
                    # 7c6c, third pair, for value at 100% where it checks
                    if a["action"] == "check":
                        s = trust * s + (1.0 - trust) * as_locked(np.ones_like(s))
                    else:
                        full, rank, m = rule_mask()
                        s = trust * s + (1.0 - trust) * as_locked(rule_weights(full, rank, m)[sol.ctx.active])
                if smoothing > 0 and only is None:
                    s = (1.0 - smoothing) * s + smoothing / len(node.names)
            if p == 0:
                r0 = r0 * s
            else:
                r1 = r1 * s
            if node.names[k] != name:
                notes.append(f"{street}: mapped '{name}' to tree action '{node.names[k]}'")
            if taken is not None:
                taken.append(node.names[k])
            act = a["action"]
            if act in ("bet", "raise"):
                amt = a.get("to") if a.get("to") is not None else a.get("amount")
                if amt is None:
                    amt = node.children[k].bets[p] if isinstance(node.children[k], Decision) else bets[1 - p]
                bets[p] = float(amt)
            elif act == "all-in":
                bets[p] = stack
            elif act == "call":
                bets[p] = min(bets[1 - p], stack)
            node = node.children[k]
        return node, r0, r1

    @staticmethod
    def _tree_action_name(node: Decision, a: dict, p: int, bets, pot, stack: float = float("inf")) -> str:
        act = a["action"]
        if act in ("check", "fold", "call", "all-in"):
            return act
        pot_total = pot + bets[0] + bets[1]
        if act == "bet":
            amt = a.get("amount") or a.get("to") or 0.0
            if bets[p] + amt >= stack - 1e-9:
                return "all-in"
            return f"bet {amt / max(pot_total, 1e-9) * 100:.0f}%"
        to = a.get("to") or a.get("amount") or 0.0
        return "all-in" if to >= stack - 1e-9 else f"raise to {to:g}"

    @staticmethod
    def _amount(name: str, node: Decision, p: int) -> Optional[float]:
        child = dict(zip(node.names, node.children)).get(name)
        if name.startswith(("bet", "raise")) or name == "all-in":
            return round(child.bets[p], 2) if isinstance(child, Decision) else round(node.bets[p] + _stack_left(node, p), 2)
        if name == "call":
            return round(node.bets[1 - p] - node.bets[p], 2)
        return None

    def _sample(self, acts) -> str:
        r = self.rng.random(); acc = 0.0
        for a in acts:
            acc += a["freq"]
            if r <= acc:
                return a["action"]
        return acts[-1]["action"]


def _stack_left(node: Decision, p: int) -> float:
    for n, c in zip(node.names, node.children):
        if n == "all-in":
            return _inv_of(c, p) - _inv_of(node, p)
    return 0.0


def _swap_nearest(sizes: List[float], frac: float, tol: float = 0.15, fixed: Optional[set] = None) -> None:
    """Replace the entry of `sizes` closest to `frac` with `frac` if it is more than `tol`
    (relative) away; otherwise keep the abstraction (keeps solves cache-friendly). Entries in
    `fixed` (indices of sizes taken earlier this street) are never replaced: with all of them
    taken, `frac` is added. The entry used is added to `fixed`."""
    fixed = set() if fixed is None else fixed
    if not sizes or frac <= 0:
        return
    near = min(range(len(sizes)), key=lambda k: abs(sizes[k] - frac))
    if abs(sizes[near] - frac) / max(sizes[near], 1e-9) <= tol or frac in sizes:
        fixed.add(sizes.index(frac) if frac in sizes else near)
        return
    free = [k for k in range(len(sizes)) if k not in fixed]
    if free:
        i = min(free, key=lambda k: abs(sizes[k] - frac))
        sizes[i] = round(frac, 3)
    else:
        sizes.append(round(frac, 3))
        i = len(sizes) - 1
    fixed.add(i)


RANKS = "AKQJT98765432"


def _study_range(board, names, amounts, me, sf, ev, eq, opps):
    """Study's answer for the whole range at a node (Advisor._study_node, _study_mw_node): per action the range's
    frequency, combos and the EV of the hands that take it; per hand class (the grid) and combo the reach (w), the
    mix (f), each action's EV in bb (ev; evm the mix's), the equity (eq; nan: none) and the made hand and draw (indices
    into "categories"); and per opponent (seat, range, its equity) its grid, combos and equity."""
    from .breakdown import categorize, MADE, DRAWS
    A = len(names)
    evm = (sf * ev).sum(axis=0)
    tot = max(float(me.sum()), 1e-12)

    def mean(x, w):
        ok = ~np.isnan(x) & (w > 0)
        return round(float((x[ok] * w[ok]).sum() / w[ok].sum()), 3) if ok.any() else None
    acts = []
    for a, name in enumerate(names):
        wa = me * sf[a]
        acts.append({"action": name, "amount_bb": amounts[a],
                     "freq": round(float(wa.sum()) / tot, 3), "combos": round(float(wa.sum()), 1),
                     # (the EV of the hands that take it; none for an action under a twentieth of a combo)
                     "ev_bb": round(float((wa * ev[a]).sum() / wa.sum()), 2) if wa.sum() >= 0.05 else None})
    cells, combos = {}, {}
    held = np.flatnonzero(me >= 5e-4)
    mi, di = {k: i for i, (k, _) in enumerate(MADE)}, {k: i for i, (k, _) in enumerate(DRAWS)}
    cat = {int(i): (mi[m], di[d]) for i, m, d in zip(held, *categorize(board, held))}
    opp_cells = [{} for _ in opps]
    for cls, idx in _class_index().items():
        w = me[idx]
        sw = float(w.sum())
        if sw > 1e-9:
            cells[cls] = {"w": round(sw / len(idx), 3),
                          "f": [round(float((sf[a, idx] * w).sum()) / sw, 3) for a in range(A)],
                          "ev": [round(float((ev[a, idx] * w).sum()) / sw, 2) for a in range(A)],
                          "evm": round(float((evm[idx] * w).sum()) / sw, 2), "eq": mean(eq[idx], w)}
            combos[cls] = [[cards_str(sorted(ALL_COMBOS[i], reverse=True)), round(float(me[i]), 3),
                            [round(float(sf[a, i]), 3) for a in range(A)], [round(float(ev[a, i]), 2) for a in range(A)],
                            None if np.isnan(eq[i]) else round(float(eq[i]), 3), *cat[int(i)]]
                           for i in idx if me[i] >= 5e-4]
        for oc, (_, opp, eq_opp) in zip(opp_cells, opps):
            wo = opp[idx]
            if wo.sum() > 1e-9:
                oc[cls] = {"w": round(float(wo.sum()) / len(idx), 3), "f": [1.0], "eq": mean(eq_opp[idx], wo)}
    part = {"actions": acts, "grid": {"actions": list(names), "cells": cells}, "combos": combos,
            "categories": {"made": [list(x) for x in MADE], "draws": [list(x) for x in DRAWS] if len(board) < 5 else []},
            "range_combos": round(float(me.sum()), 1), "range_ev_bb": round(float((me * evm).sum()) / tot, 2),
            "range_eq": mean(eq, me)}
    return part, [{"seat": seat, "combos": round(float(opp.sum()), 1), "eq": mean(eq_opp, opp),
                   "grid": {"actions": ["in range"], "cells": oc}} for oc, (seat, opp, eq_opp) in zip(opp_cells, opps)]


@lru_cache(maxsize=1)
def _class_index() -> Dict[str, np.ndarray]:
    """Each hand class's combos (indices into ALL_COMBOS)."""
    cls_idx: Dict[str, List[int]] = {}
    for i, cb in enumerate(ALL_COMBOS):
        cls_idx.setdefault(hand_class(cb), []).append(i)
    return {k: np.array(v) for k, v in cls_idx.items()}


def strategy_grid(strats: List[np.ndarray], names: List[str], weights: np.ndarray) -> dict:
    """13x13 hand-class grid: for each class, the range weight and per-action frequency."""
    out = {}
    for cls, idx in _class_index().items():
        w = weights[idx]
        tot = float(w.sum())
        if tot <= 1e-12:
            out[cls] = {"w": 0.0, "f": [0.0] * len(names)}
            continue
        out[cls] = {"w": round(tot / len(idx), 3),
                    "f": [round(float((s[idx] * w).sum() / tot), 3) for s in strats]}
    return {"actions": names, "cells": out}


def state_board(board) -> str:
    return cards_str(board)


def one_size_candidates(pot: float, behind: float, allin_spr: float, river: bool = False) -> List[Tuple[str, float]]:
    """The one-size view's sizes at a decision with `pot` in the middle and `behind` to bet: (label, fraction of the
    pot) for each of ONE_SIZES (ONE_SIZES_RIVER on the river), and all-in where the tree offers it (behind at most
    allin_spr pots), as the tree builds them (solver.TreeBuilder._decision): a bet rounds to half a big blind (at least
    1), one that reaches the stack is the all-in, and sizes that come out the same bet are one."""
    out, seen = [], set()
    for f in (ONE_SIZES_RIVER if river else ONE_SIZES) + ((ONE_SIZE_ALLIN,) if behind <= allin_spr * pot else ()):
        amt = max(round(f * pot * 2) / 2, 1.0)
        label = "all-in" if amt >= behind - 1e-9 else f"{int(round(f * 100))}%"
        key = label if label == "all-in" else amt
        if key in seen:
            continue
        seen.add(key)
        out.append((label, f))
    return out


def one_size_summary(full: Optional[dict], rows: List[dict], pot: float) -> dict:
    """The one-size view's verdict over the rows solved so far (each {"size", "ev_bb", "exploitability_pct", ...}) and
    the full menu's row: each row's EV given up against the full menu (loss_pct, % of the pot; None without it), the
    best size (the highest EV: the least given up) and the sizes tied with it (within the precision of the two solves,
    the larger of their exploitabilities)."""
    rows = [dict(r) for r in rows]
    for r in rows:
        r["loss_pct"] = None if full is None else round((full["ev_bb"] - r["ev_bb"]) / max(pot, 1e-9) * 100, 3)
    if not rows:
        return {"rows": rows, "best": None, "tied": []}
    top = max(rows, key=lambda r: r["ev_bb"])
    tol = lambda r: max(r.get("exploitability_pct") or 0.0, top.get("exploitability_pct") or 0.0) / 100 * pot
    tied = [r["size"] for r in rows if r is not top and top["ev_bb"] - r["ev_bb"] <= tol(r)]
    return {"rows": rows, "best": top["size"], "tied": tied}
