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
# The session analyzer's Quick triage (Advisor.triage, "triage": true): a decision nothing in hand answers at once is
# solved as Quick solves it (index.html QUICK_BUDGETS: [the street, each earlier street], "quick": the earlier streets
# the hand hasn't solved narrowed by the betting rules), and such an answer this close to equilibrium (% of the pot)
# is graded with medium confidence, further with low
TRIAGE_BUDGETS = {"flop": (1.5, 0.3), "turn": (1.5, 0.3), "river": (0.8, 0.3)}
TRIAGE_SETTLED = 1.0
LOOK_MIN_S = 1.0   # Advisor.progress looks at a solve only with at least this many seconds of it left (at its pace so
                   # far) to where it next may stop (a settled check, or its end): a quicker one answers in full instead
                   # (at 0.5 a 3bet pot's turn showed its look 0.5 s before its answer, one thread)
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
MW_SETTLED = 0.02  # a 3-way solve stops once its players' mean best-response gain is under this (% of the pot; at
                   # 0.1 a turn stopped at 50 iterations with a thin spot's shove 4.9 bb from its settled EV)...
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
# Heads-up, an earlier street: the play after a rare line (the line itself read by the rules, RARE_LINE) is that street
# solved again from the node, against the range the rules give the line (Advisor._rare_resolver, prior_budget), not read
# off the street's own solve, which barely reaches the node and plays it against the sliver of a range it takes there.
# Against exact solves rooted at the node (b-inary's postflop-solver) the next street's first two decisions gave up
# 0.114% of the pot on average over 17 lines whose answer is common at equilibrium, where the solve's node gave up 0.159
# and the rules' range for the answers too 0.333 (README, Earlier streets). A library flop that stores the node is read.
RARE_RESOLVE = True
RARE_RESOLVE_SETTLED = 0.1   # ...stopped at the first of HU_CHECKS (HU_SETTLED_FROM on) where it is this close (% pot):
                             # 300 iterations natively, as close to the exact answers as 1000 (0.114 / 0.116)
MW_OFF_RANGE = 1e-10   # multiway: the weight of your hand where your range holds none of it (under the 1e-9 that says so)
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
# heads-up: an action hero's range takes less often than this (the solve's average strategy over the range at the node)
# gets "rough_ev": the opponent's answer to a branch the range almost never takes is barely trained (its regrets move
# only while the solve's current strategy still takes it), so the EV is rough and usually lower than exact. Against
# postflop-solver with every turn decision locked to ours and the river solved exactly (docs/cfr_research.md, EV gaps
# on live turns): single-raised turns' actions the range takes 0.01-1% 6.0% of the pot off on average at 100 iterations,
# 2.4 at 300, 1.6 at Accurate's 400, 0.6 at 1000 (ours lower: -4.5, -1.5, -0.8, -0.3); 1-5%: 1.4, 0.6, 0.6, 0.15;
# 5% or more 0.50, 0.16, 0.11, 0.06. The page shows such an EV with a "≈" (Practice's EV given up too); grading is
# unchanged.
RARE_BRANCH = 0.01
BR_TIE_MIN = 0.005  # exploit mode: best-response EVs this close (share of pot; more when the solve is unsettled) count as tied
COMMIT_BET = 0.75   # exploit mode: a postflop bet this big (share of the pot) marks its maker committed: no extra folds
                    # (a 96%-pot donk on AKQ left 4-high calling for its river bluffs, +1.4 bb)
BR_TIE_MAX = 0.05   # ...and at most this: a 10-iteration solve 41% of the pot from settled tied every action, so
                    # its unsettled mix folded 55 to a turn check-raise where calling was +6.7 bb
MAX_SIZES = 3       # custom/optimized sizes: at most this many per street (each multiplies the tree; the browser's is small)
STUDY_MAX_SIZES = 5  # ...and in Study (its size menus: "More sizes" bets 25/33/50/75/125% on the flop, study.js SIZE_PRESETS)
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
# ...each solve stops at the first of its checks (below) where it is this close to equilibrium (% of the pot; expl_target when
# lower), or at its budget (the request's): the EV given up is the difference of two such solves' values, so sizes
# closer than this are a tie. Half of Fast's 0.2 (at 0.2 most of a turn's sizes tied): a page's turn 1.3-16 s for
# every size, with threads (scripts/bench_browser.mjs --one-size)
ONE_SIZE_SETTLED = {"flop": 0.1, "turn": 0.1, "river": 0.05}
# ...read at HU_CHECKS, and the full menu's at each of these (an exploitability read each, about one iteration's
# time: HU_CHECKS' gaps past 300 cost a solve settled at 410 iterations 190 more). Not the sizes': a size's
# exploitability is its tie's tolerance (one_size_summary), and read this finely its solve stops nearer the target,
# which tied one more size on two spots of scripts/bench_one_size.py's twelve (0.041 and 0.018% of the pot apart, the
# targets 0.05 and 0.02); the full menu's sets only the EV given up
ONE_SIZE_CHECKS = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900)
# ...and a size whose solve shows it can't be the best nor tied with it stops there (ONE_SIZE_CUT x the target under
# the best's lower bound: Advisor._one_size_run), its EV given up as precise as it is then
ONE_SIZE_CUT = 2.0
# Facing a bet, the view's raise sizes: a raise to these multiples of the bet faced (the opponent's street total; a
# check-raise of a third-pot bet to 3x is 40% of the pot after calling, of a pot-sized bet 67%), and all-in where the
# tree offers it (at most allin_spr pots behind after calling) or a multiple reaches the stack
ONE_RAISES = (2.5, 3.0, 4.0, 5.0)
ONE_RAISE_X = (2.0, 10.0)   # ...a raise asked for by its multiple ("3.5x") is held to this range
ONE_SIZE_CACHE = 256    # rows kept per advisor (a few KB each: a grid and the numbers), least recently used out first
ONE_SIZE_DEF = ("best = the size whose one-size strategy keeps the most EV for the player betting: each game is solved "
                "from this decision with the same ranges, the opponent adapting; EV given up = the range's EV with "
                "every candidate size to mix freely minus with that size alone, in % of the pot")
ONE_RAISE_DEF = ("best = the raise size whose one-size strategy keeps the most EV for the player facing the bet: each "
                 "game is solved from this decision with the same ranges, the opponent adapting; EV given up = the "
                 "range's EV with every candidate raise size to mix freely minus with that size alone, in % of the pot")
ONE_SIZE_ICM = ("; in a tournament every solve pays prize equity (ICM), so the EVs are chips worth as much prize "
                "equity")
STREET_OF = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
STREETS = ["preflop", "flop", "turn", "river"]


def _custom_sizes(raw, limit: int = MAX_SIZES) -> Dict[str, List[float]]:
    """state["sizes"]: {"flop"|"turn"|"river"|"raise": [% of pot, ...]} -> fractions, sorted, held to
    SIZE_PCT, at most `limit` a street. Raises ValueError with a message for the user."""
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
        if len(set(vals)) > limit:
            raise ValueError(f"sizes: at most {limit} {k} sizes (each one multiplies the game tree), got {len(set(vals))}")
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


def _tournament_info(model, hero: str, positions, street, out=(), named=False) -> dict:
    """What the answer says about the tournament: hero's prize equity (share of the prizes left) and the bubble
    factor against each opponent (what a chip lost costs over what one won gains, all in against them); in a bounty
    tournament the bounties' part (_bounty_info; `out`: the seats that have folded). named: the notes name hero's
    seat instead of saying "you" (Study: hero is whoever acts)."""
    your = f"{hero}'s" if named else "your"
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
        note = f"tournament: winner-take-all (prize equity is chips), {your} equity {eq:.1%} of the prize pool left"
    else:
        note = (f"tournament: ICM{approx}, {your} prize equity {eq:.1%} of what's left to pay; bubble factor up to {top:.2f} "
                f"(a chip lost costs {top:.2f}x what one won gains)")
    res = {"equity_share": round(eq, 4), "bubble_factors": bf, "note": note}
    if model.bounty is not None:
        res["bounties"] = _bounty_info(model, hero, positions, out, named)
    return res


def _bounty_info(model, hero: str, positions, out=(), named=False) -> dict:
    """A bounty tournament's part of the answer: the format and model, chips per unit of money (bounty_k: bb per unit
    of the payouts), the pool in play, each seat's bounty, what busting each opponent hero covers is worth to hero
    (you_bust: in bb and money, the cash it pays less hero's chip share of it after the hand; in_hand: still in, so
    what hero's call or shove here can win), and what hero's own bounty is worth to each opponent who covers hero
    (bust_you, bb). named: the note names hero's seat instead of saying "you" (Study)."""
    from .icm import BOUNTY_FORMATS
    you, your = (hero, f"{hero}'s") if named else ("you", "your")
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
        gain = "; ".join(f"busting {p} ({you} cover{'s' * named}) pays {d['cash']:g}, worth {d['bb']:.1f}bb to {you}"
                         for p, d in sorted(live.items(), key=lambda x: -x[1]["bb"]))
    else:
        gain = f"{you} cover{'s' * named} nobody still in the hand who carries a bounty"
    risk = [p for p in bust_you if p not in out]
    note = (f"bounties: {BOUNTY_FORMATS[b.fmt]} ({what}), valued by the proportional model (each player owns their "
            f"chip share of the {model.bounty_pool:g} in bounties in play, so 1 is worth {k:.3g}bb: GTO Wizard's bounty "
            f"power{f'; the half added to {your} head counts at {your} chip share of it' if b.fmt == 'pko' else ''}): {gain}"
            + (f"; {', '.join(risk)} cover{'s' * (len(risk) == 1)} {you} ({your} bounty {model.bounties[hero]:g})"
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
    from .evaluator import eval_boards
    rng = random.Random(sum(c * 53 ** i for i, c in enumerate(board)))
    avail = [c for c in range(52) if c not in board]
    tot = np.zeros(N_COMBOS); cnt = np.zeros(N_COMBOS)
    # Every turn card twice from the flop (a random river each), every river from the turn; strength as a
    # percentile of the live combos on each runout, averaged. 24 random runouts of raw strength ranked
    # K5s and K9o above AA on K72 (two-pair runouts weigh more in raw strength than an overpair holding).
    if len(board) == 5:
        runs = [()]
    elif len(board) == 4:
        runs = [(c,) for c in avail]
    else:
        runs = [(t, rng.choice([c for c in avail if c != t])) for t in avail for _ in range(2)]
    # (every runout's strengths at once, and each live combo's rank among its runout's from a count of each strength:
    # the live combos below it and up to it, a sorted array's searchsorted left and right, ties sharing a rank; the
    # same numbers as a runout at a time, in a few numpy calls where that took ~10 a runout: 20 ms natively a turn)
    S = eval_boards([list(board) + list(run) for run in runs])          # (runs, combos): 1..7462, -1 blocked
    ok = S >= 0
    nr, top = len(runs), 7463
    at = np.arange(nr)[:, None] * top + S
    upto = np.cumsum(np.bincount(at[ok], minlength=nr * top).reshape(nr, top), axis=1).ravel()
    mid = (upto[np.where(ok, at - 1, 0)] + upto[np.where(ok, at, 0)] - 1) / 2.0
    live = ok.sum(axis=1)
    for j in range(nr):
        o = ok[j]
        tot[o] += mid[j][o] / max(live[j] - 1, 1); cnt[o] += 1
    sts = list(S)
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


def _narrow_street(ranges: Dict[str, np.ndarray], board, st: StreetState, stack,
                   slow: Optional[Dict[str, np.ndarray]] = None, check_whole: bool = False) -> None:
    """Narrow each player's range (in place) by what they did on a street that started multiway,
    with the multiway rules: a check drops most of the hands that would have bet, a bet or raise
    keeps its value + bluff range, a call keeps what continues minus what would have raised.
    Hands a rule rules out keep some weight (players don't follow our rules), see _soften. `slow`
    (carried from street to street): per player, the combos a call has already slowplayed. `stack`: everyone's
    stack in bb, or {seat: stack}. `check_whole`: a check keeps the range whole, as a heads-up street's rules do
    (Advisor._walk at trust 0: heads-up the rules' check took the value out of ranges that check nearly all of it).
    Your own cards play no part: a player's line doesn't depend on what you hold, so it is read off their whole range
    (what you block comes out only when the hand is counted against those ranges: the solve's card removal)."""
    sc, rank, bluff = _street_info(tuple(board))
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


def _post_floor(before: np.ndarray, after: np.ndarray, board, pos: str, street: str, notes: List[str],
                off: Optional[List[dict]] = None) -> np.ndarray:
    """A player's range after an earlier street's actions, filled out when the line leaves fewer than
    preflop.MIN_LINE_COMBOS live combos (counting the cards dealt since): the next street's solve would
    otherwise play against a face-up hand (a 4bet pot's ~12-combo range checking back the flop left 2 for
    the turn). The filler is the rest of the range the player started the street with, weighted as
    preflop (pf.thin_line: 1 - combos / MIN_LINE_COMBOS); the hands that took the line are kept. `off` (the
    answer's off_solution) gets the fill."""
    live = np.ones(N_COMBOS)
    for c in board:
        live = live * (1.0 - MEMBER_FULL[c])
    r, _ = pf.thin_line(after * live, before * live)
    if r.sum() > (after * live).sum() + 1e-9:
        notes.append(f"{street}: {pos}'s line left {float((after * live).sum()):.1f} combos, so their range "
                     f"is filled out with the rest of their {street} range")
        if off is not None:
            off.append({"street": street, "pos": pos, "why": "filled", "combos": round(float((after * live).sum()), 1)})
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


_HELD_SKIP = (str, bytes, int, float, bool, type(None), type, types.ModuleType, types.FunctionType,
              types.BuiltinFunctionType, types.MethodType, functools.partial)
_HELD_ATOMS = frozenset((str, bytes, int, float, bool, type(None)))
_HELD_SLOTS: dict = {}


def _held_slots(t) -> tuple:
    """The __slots__ of class t and its bases (once a class)."""
    sl = _HELD_SLOTS.get(t)
    if sl is None:
        sl = []
        for k in t.__mro__:
            s = k.__dict__.get("__slots__", ())
            sl.extend((s,) if isinstance(s, str) else s)
        sl = _HELD_SLOTS[t] = tuple(sl)
    return sl


def _held_bytes(obj) -> int:
    """Bytes of the numpy arrays reachable from obj, each counted once: about what dropping a cached solve frees
    (arrays it shares with module-level tables count too). Functions, methods and classes aren't followed. (Exact
    types first and the slots once a class: a limped flop's solve, ~16,000 objects, 36 -> 27 ms in Pyodide.)"""
    seen, todo, total = set(), [obj], 0
    pop, extend, add, nd = todo.pop, todo.extend, seen.add, np.ndarray
    while todo:
        o = pop()
        t = type(o)
        if t in _HELD_ATOMS:
            continue
        i = id(o)
        if i in seen or isinstance(o, _HELD_SKIP):
            continue
        add(i)
        if t is nd or isinstance(o, nd):
            b = o.base
            if isinstance(b, nd):
                todo.append(b)
            else:
                total += o.nbytes
        elif t is dict or isinstance(o, dict):
            extend(o.values())
        elif t is list or t is tuple or isinstance(o, (list, tuple, set, frozenset)):
            extend(o)
        else:
            d = getattr(o, "__dict__", None)
            if d:
                extend(d.values())
            sl = _held_slots(t)
            if sl:
                extend(getattr(o, a) for a in sl if hasattr(o, a))
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


def _mw_follow(node, R, a: dict, players, rare: float = RARE_LINE):
    """One real action `a` down a multiway solve's root street from `node` (its tree's nearest: _mw_pick): its
    player's reach R[p] (per combo, in place) times the solve's average mix for it; the child. _NoInstant where the
    tree doesn't hold it (another player to act, a line the 3-way library doesn't store) or the range keeps under
    `rare` of itself (RARE_LINE: the solve's play there barely converged)."""
    from . import multiway_solver as MW
    if not isinstance(node, MW.Decision) or getattr(node, "unstored", False) or a.get("pos") not in players \
            or players.index(a["pos"]) != node.player:
        raise _NoInstant()
    i = _mw_pick(node, a)
    if i is None:
        raise _NoInstant()
    p = node.player
    before = R[p].sum()
    R[p] *= node.avg_strategy()[0][i]
    if R[p].sum() < rare * max(before, 1e-12):
        raise _NoInstant()
    return node.children[i]


def _td_amounts(td, c0) -> list:
    """A later street's decision (multiway_solver._template's _TD, the street begun with chips in c0): its actions'
    amounts as a root decision's (multiway_solver._actions): a call its size, a bet, raise or all-in the street's
    chips after it, else None."""
    p = td.player
    out = []
    for nm, kid in zip(td.names, td.children):
        k = nm.split()[0]
        out.append(None if k in ("check", "fold") else kid.c[p] - td.c[p] if k == "call" else kid.c[p] - c0[p])
    return out


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
    if any(k in res for k in ("skipped", "cancelled", "one_size", "showdown")) or res.get("study") in ("deal", "over", "runouts"):
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


class _FirstLook(Cancelled):
    """Advisor.progress: the request's own street's solve reached its next look (Advisor.first_look), cached as it
    stands. A Cancelled, so the catch-alls let it through as they do a stop."""


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
    off: Optional[Callable] = None    # a library flop facing a size off its tree: () -> instant_from_off_tree's answer
    unraked: Optional[Callable] = None   # a raked table's library flop: () -> its unraked entry (instant_from_unraked)


def _pf_aggressor(preflop: List[dict], oop: str, ip: str) -> Optional[int]:
    """Who raised last preflop of a heads-up pot's two players (0 out of position, 1 in position), None: nobody (a
    limped pot) or someone since folded (flop_approx.near_spot reads another pot's spot only for the same raiser)."""
    last = None
    for a in preflop or ():
        if a.get("action") in ("raise", "all-in", "bet"):
            last = str(a.get("pos", "")).upper()
    return 0 if last == oop else 1 if last == ip else None


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
    if spot.own_earlier and adv._off_later(spot.street):
        got = adv._off_tree_own(sol, spot)
        if got is not None:
            return got
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
# the flop's own solve in hand has run this many iterations, from when instant_from_own reads it.
# scripts/bench_flop_seed.py (80 stored flops of the flatted opens and the 3bet pots, none the learned correction was
# fit on, solved live in the Fast tree; EV given up at the stored decisions, over the combos the approximation
# answers, % of the pot, mean), against the flop solved live in the Accurate tree (--truth accurate:600): the
# approximation as answered (hand classes, facing a bet corrected) 0.20-0.24, the solve 0.31-0.33 at 40 iterations,
# 0.21-0.24 at 50 (the approximation better on 18-22 of 40 flops), 0.07-0.10 at 100 (0-5 of 40), 0.03-0.05 from 300
# on. Priced by the library's own entry instead the approximation (0.06) beats even the converged solve (0.08-0.10:
# its tree's distance from the library's), but that reference is the approximation's own source. The crossover is the
# 50th iteration (the approximation better on 13-14 of 40 at 60). Reads off another spot or at another depth hand over
# there too.
NEAR_UNTIL = HU_CHECKS[0]
OFF_TREE_TOL = 0.15   # a real size more than this share off the tree's nearest is swapped in (_swap_nearest)


def instant_from_near(adv: "Advisor", spot: InstantSpot) -> Optional[dict]:
    """A flop of a library spot that the library doesn't store (the flatted opens' and the 3bet pots' other 1203): its
    decisions read off the spot's stored flops most like it (gto/flop_approx.py: the same pairing and suit pattern, the
    ranks nearest, each carried over by card substitution and their mixes averaged over the images of hero's hand
    class), on the line the hand took (sizes at the tree's nearest); a heads-up pot the library holds no spot for (the
    HJ's and UTG's opens 3bet by the blinds or flatted by the SB, the BTN's 3bet of the UTG) or at another stack depth
    off the spot whose ranges are nearest, where one covers it (flop_approx.near_spot: its entry of the flop where it
    stores it, else those most like it); not once the flop's own solve in hand has run NEAR_UNTIL iterations
    (instant_from_own reads that). None where hero's decision isn't one an entry stores or no stored flop holds an
    image of hero's combo."""
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
    label = _near_label(sol)
    f = getattr(sol, "filled", {}).get(id(node))
    if f is not None and f[col[0]]:
        label += "; your hand isn't in that spot's range: the average mix there of its hand class (made hand and draw)"
    return {"actions": _real_actions(node, strat, bets, spot), "source": "near", "label": label, "iterations": None}


# A raked table's flop of a library spot is read off the spot's unraked entry (instant_from_unraked) until the flop's own
# solve, with the rake, has run this many iterations, from when instant_from_own reads it (flop_library.RAKE_READ).
RAKE_UNTIL = HU_CHECKS[0]


def instant_from_unraked(adv: "Advisor", spot: InstantSpot) -> Optional[dict]:
    """A raked table's flop that the library stores for the same preflop line without rake (every entry is solved
    without; with rake the advisor re-prices the preflop calls, so the callers' ranges are narrower and the flop is
    solved live, Advisor._unraked_entry): hero's mix read off the entry, for the raked ranges' combos, on the line the
    hand took (sizes at the tree's nearest), at a decision the entry stores; not once the flop's own raked solve has
    run RAKE_UNTIL iterations (instant_from_own reads that). No EVs (the entry's leave out the rake and the ranges'
    change). What it gives up: flop_library.RAKE_READ."""
    if spot.street != "flop" or spot.unraked is None:
        return None
    if spot.own is not None and spot.own.solver.iters >= RAKE_UNTIL:
        return None
    sol = spot.unraked()
    if sol is None:
        return None
    col = hole_col(sol, spot.hole)
    if not col:
        return None
    got = _read_line(sol.root, 0, col[0], spot)
    if got is None or id(got[0]) not in sol.stored:
        return None
    node, strat, bets = got
    return {"actions": _real_actions(node, strat, bets, spot), "source": "unraked", "label": sol.unraked_text(),
            "iterations": sol.iters}


# A library flop facing a bet or raise of a size its tree doesn't have is read off the stored decisions facing the sizes
# it does (instant_from_off_tree) until the flop's own solve, with the real size in its tree, has run this many
# iterations, from when instant_from_own reads it. scripts/bench_off_tree.py (16 BTN vs BB flops in texture order;
# EV given up at the decision facing the size, hero's whole range priced by the flop solved in the Accurate tree with
# the size in place of the nearest, 1000 iterations, natively; % of the pot, mean over the flops where the bettor takes
# the line 2%+): facing a 25 / 50 / 100 / 150% bet the stored nearest size's answer gives up 1.00 / 0.60 / 0.59 / 1.69,
# read by the price ("mdf") 0.39 / 0.42 / 0.31 / 0.40 (the entry's own answers facing its 33% and 75% bets: 0.26 /
# 0.30), the Fast solve with the size 0.50 / 0.38 / 0.50 / 0.44 at 50 iterations and 0.21 / 0.16 / 0.22 / 0.17 at 100;
# a 33% bet raised 30 / 85 / 130% of the pot after the call: nearest 1.04 / 2.16 / 6.06, by the price 0.50 / 0.81 /
# 1.10, Fast 0.43 / 0.99 / 2.18 at 50 iterations. The crossover is about the 50th iteration.
OFF_UNTIL = HU_CHECKS[0]
# off_tree.answer's method for those reads (the flop's, and the turn's and river's off your earlier decision's solve).
# "nearmdf": the nearest size's mix, the range's continuing share moved by the price; outside the sizes the same as
# "mdf", which between two takes each combo's mix between them. A 50% bet (12 flops a spot, the size in place of the
# nearest, the Accurate tree at 1000 iterations; % of the pot), nearest / "mdf" / "nearmdf": BTN vs BB 0.59 / 0.37 /
# 0.35, CO vs BB 0.60 / 0.45 / 0.30, SB vs BB (the big blind's stab) 0.51 / 0.65 / 0.43, BB vs BTN 3bet pot 0.56 / 0.42
# / 0.33; with 50% the bettor's only size 0.55 / 0.34 / 0.32, 0.58 / 0.42 / 0.33, 0.54 / 0.55 / 0.40, 0.55 / 0.41 /
# 0.33.
OFF_METHOD = "nearmdf"
# A turn or river facing a bet or raise off the tree of the street's solve from your earlier decision (instant_from_own,
# Advisor._off_tree_own), by street: "outside", read by its price where it is smaller or bigger than every size that
# solve has; "above", only bigger; "always", between two of them too; None, never; else at the nearest size.
# scripts/bench_off_tree.py --street turn / river (BTN vs BB and BB vs BTN 3bet pot turns after the library flop's 33%
# bet called, a random card, and rivers after the turn's 75% bet called; the source the street's Fast solve without
# the size, 300 iterations; hero's whole range at the decision facing it priced by the street solved in the Accurate
# tree with the size in place of the nearest, 600 / 1000 iterations; % of the pot, mean where the line is taken 2%+).
# The turn, bigger or smaller than every size, read by the price: a 200% bet 0.10 where the nearest (125%) gave up
# 0.25, in a 3bet pot (the bettor's only size) 1.61 against 2.28; the 75% bet check-raised 40% / 120% of the pot after
# the call 0.62 / 0.75 against 1.29 / 1.51 (30%: 1.06 against 3.23), a 20% bet (the only size) 0.46 against 0.92, a
# 3bet pot's 0.08 against 0.14; a 3bet pot's turn as Fast plays it on one thread (33/75%: FAST_SPR_SIZES;
# --shallow), checked to and bet 100 / 125 / 150 / 200% (the only size) 0.22 / 0.42 / 0.77 / 1.39 against 0.35 / 0.94
# / 2.12 / 5.18, led 0.20 / 0.38 / 0.40 / 1.53 against 0.45 / 1.23 / 2.30 / 5.64. The river, bigger: the 75% bet
# raised 120% 0.14 against 0.35 (16 rivers), a 300% bet (sizes 33/75/125/250%) 0.03 against 0.04, the bettor's only
# size 2.76 against 4.03; smaller, the nearest's: raised 30% / 50% 0.34 / 0.14 against 3.06 / 0.51 (8 rivers each), a
# 20% lead 0.37 against 0.63 (20 rivers; a 3bet pot's 0.04 against 0.15), a 20% bet 0.13 against 0.17 (5 rivers; the
# only size 0.80 against 0.73). Between two sizes (OFF_METHOD: the nearest's mix, the continuing share by the price),
# the nearest / each combo's mix between the two ("mdf") / OFF_METHOD: the turn (sizes 33/75/125%) checked to and bet
# 50% / 100%, 3bet pots (12 / 6 turns) 0.15 / 0.19, 0.21 / 0.49, 0.07 / 0.11, single-raised (5 turns, 100%) 0.09,
# 0.12, 0.05; led 50% / 100% in 3bet pots (9 / 7) 0.22 / 0.26, 0.24 / 0.55, 0.09 / 0.10; the river, a tie: a 100% bet
# (11 rivers) 0.11, 0.16, 0.12, a 50% lead (4) 0.12, 0.31, 0.12.
OFF_LATER = {"turn": "always", "river": "above"}
# Those reads take hero's mix only from the decisions facing a size the villain bets or raises with this share of
# their range or more (the flop library's STORE_LINE: an entry doesn't store the rest; off_tree.readable: of those, the
# one just below the real size and all above), the continuing share still by every size's price, and the river reads
# by the price between two sizes too where the villain doesn't take the nearest. Hero's play facing a size the solve
# never bets is its noise: single-raised turns (the BTN betting 75/125%, 33% almost never) read a 50% bet off the 33%
# decision, raising 6-15% of the range where the reference raised 0-11%. scripts/bench_off_tree.py --rescore
# every,taken (every size read / these; % of the pot, mean / median / 90th; against the bettor's only size, and the
# advice's menu): single-raised turns (28, bets of 40/50/60/100% and leads) 0.17 / 0.10 / 0.38 -> 0.11 / 0.05 / 0.31,
# 0.045 / 0.026 / 0.125 -> 0.036 / 0.026 / 0.076; 3bet pot turns (28) 0.15 / 0.09 / 0.34 -> 0.13 / 0.09 / 0.32,
# 0.059 / 0.048 / 0.131 -> 0.066 / 0.050 / 0.143; single-raised rivers (48, bets of 50/100/175% and leads) 1.07 / 0.68
# / 2.13 -> 0.89 / 0.36 / 1.73, 0.27 / 0.18 / 0.67 -> 0.26 / 0.16 / 0.67 (Fast at 50 iterations on the advice's
# menu: 0.125, 0.075, 0.50). 0.01-0.08 measured the same.
OFF_SOURCE_LINE = 0.05


def instant_from_off_tree(adv: "Advisor", spot: InstantSpot) -> Optional[dict]:
    """A library flop facing a bet or raise more than 15% off every size its tree has (the flop's own solve swaps the
    real size in, so the entry doesn't answer it): hero's mix read off the entry's stored decisions facing the sizes
    of that action it does store (gto/off_tree.py, OFF_METHOD: each combo's mix at the nearest stored size, then the
    range's continuing share moved to what the stored sizes defend relative to the minimum defence, the hands that
    realize the least of the pot folding first), the line before it walked as the library's answers are. Not once
    the flop's own solve has run OFF_UNTIL iterations (instant_from_own reads that)."""
    if spot.street != "flop" or spot.off is None:
        return None
    if spot.own is not None and spot.own.solver.iters >= OFF_UNTIL:
        return None
    return spot.off()


def _off_tree_price(acts: List[dict], oop: str, ip: str, hero_p: int, pot: float, eff: float):
    """(the villain's bet to, C hero's price to call, T the pot with it, the real bets before it) where the street's last
    action is the villain's bet or raise (not a jam), else None. The bets as _cfg_with_real_sizes reads them."""
    if not acts or acts[-1]["pos"] not in (oop, ip) or acts[-1]["action"] not in ("bet", "raise"):
        return None
    vp = 1 - hero_p
    if (0 if acts[-1]["pos"] == oop else 1) != vp:
        return None
    amount = lambda a: (a.get("amount") or a.get("to") if a["action"] == "bet" else a.get("to") or a.get("amount"))
    bets = [0.0, 0.0]
    for a in acts[:-1]:
        if a["pos"] not in (oop, ip):
            continue
        p = 0 if a["pos"] == oop else 1
        if a["action"] in ("bet", "raise") and amount(a):
            bets[p] = float(amount(a))
        elif a["action"] == "call":
            bets[p] = min(bets[1 - p], eff)
        elif a["action"] == "all-in":
            return None
    to = float(amount(acts[-1]) or 0.0)
    if to >= eff - 1e-9 or to <= bets[hero_p] + 1e-9:
        return None   # (a jam, or not a bet)
    C = to - bets[hero_p]
    return to, C, pot + bets[0] + bets[1] + (to - bets[vp]), bets


def _off_tree_read(node, hero_p: int, c0, c1, C: float, T: float, evs: Callable, stored=None, tol=OFF_TREE_TOL,
                   between: bool = True, below: bool = True):
    """Hero facing the villain's bet or raise of price (C, T) at the villain's decision `node` of a solve (c0, c1: the
    ranges there, over its combos): (off_tree.answer, OFF_METHOD, over the decisions facing the node's bets or
    raises, by kind; the decision facing the size nearest; their sizes x) where the real size is more than `tol` off
    every one of them, else None; None too where the villain's bets or raises there are rare (RARE_LINE of their
    range), no decision facing them is held (`stored`: the ids of the nodes held, None every one), or (not
    `between`) the real size is between two of them, or (not `below`) smaller than all, unless the villain takes the
    nearest with under OFF_SOURCE_LINE of their range. Off a live solve (`stored` None) the mix (and the decision
    facing the size nearest) from the sizes they take that often (off_tree.readable: the one just below the real size
    and those above; else all).
    evs(child, k): its EVs per action, chips relative to folding (k: the villain's action leading to it)."""
    from . import off_tree
    vp = 1 - hero_p
    if not isinstance(node, Decision) or node.player != vp:
        return None
    sv = node.avg_strategy0()
    rv = np.asarray(c1 if vp == 1 else c0, dtype=np.float64)
    me = np.asarray(c0 if hero_p == 0 else c1, dtype=np.float64)
    cand, tot = [], 0.0
    for k, (nm, ch) in enumerate(zip(node.names, node.children)):
        if not nm.startswith(("bet", "raise")):
            continue
        tot += float((rv * sv[k]).sum())
        if isinstance(ch, Decision) and ch.player == hero_p and (stored is None or id(ch) in stored):
            cand.append((k, ch))
    if not cand or tot < RARE_LINE * max(float(rv.sum()), 1e-12):
        return None
    xs = [off_tree.size_of(float(max(ch.bets) - ch.bets[hero_p]), float(ch.pot_now)) for _, ch in cand]
    x = off_tree.size_of(C, T)
    i = off_tree.nearest(xs, x)
    if abs(xs[i] - x) <= tol * xs[i]:
        return None   # (a size the tree has: read as it is)
    # off a live solve (every decision held) the mix from the sizes the villain takes OFF_SOURCE_LINE of the time or
    # more (hero's mix facing one they don't is the solve's noise; the flop library doesn't store those: STORE_LINE),
    # the continuing share by every size's price
    mi = off_tree.readable(xs, [stored is not None or float((rv * sv[k]).sum()) >= OFF_SOURCE_LINE * float(rv.sum())
                                for k, _ in cand], x)
    if i in mi:   # (else the nearest's answer is that noise: read by the price)
        if not between and min(xs) < x < max(xs):
            return None   # (between two of its sizes: the nearest's answer, OFF_LATER)
        if not below and x < min(xs):
            return None   # (smaller than all: the nearest's answer, OFF_LATER's "above")
    i = mi[off_tree.nearest([xs[j] for j in mi], x)]
    sources = [{"names": list(ch.names), "strat": np.asarray(ch.avg_strategy0(), dtype=np.float64),
                "evs": np.asarray(evs(ch, k), dtype=np.float64), "C": float(max(ch.bets) - ch.bets[hero_p]),
                "T": float(ch.pot_now), "reach": me} for k, ch in cand]
    return off_tree.answer(sources, C, T, OFF_METHOD, reach=me, mix_from=mi), cand[i][1], xs


def _off_tree_actions(near: Decision, mix: Dict[str, float], to: float, C: float, T: float, bets, hero_p: int,
                      behind: float, raise_fracs=None) -> List[dict]:
    """Hero's actions facing the real bet (to; price C into T) as the decision facing the nearest size has them, with
    `mix` (by kind) laid on (off_tree.onto): a call of the real bet, a raise as the tree's share of the pot after the
    call (`raise_fracs`, the tree's raise sizes in order, else read off `near`) laid on it, each at most hero's stack."""
    from . import off_tree
    strat = off_tree.onto(near.names, {k: np.atleast_1d(v) for k, v in mix.items()})[:, 0]
    cap = bets[hero_p] + behind
    rs = [nm for nm in near.names if nm.startswith("raise")]
    cn = float(max(near.bets) - near.bets[hero_p])
    out = []
    for a, nm in enumerate(near.names):
        amt = None
        if nm == "call":
            amt = round(min(to, cap) - bets[hero_p], 2)
        elif nm == "all-in":
            amt = round(cap, 2)
        elif nm.startswith("raise"):
            f = raise_fracs[rs.index(nm)] if raise_fracs and len(raise_fracs) == len(rs) else \
                (float(nm.split()[-1]) - max(near.bets)) / max(near.pot_now + cn, 1e-9)
            amt = round(min(to + f * (T + C), cap), 2)
            nm = f"raise to {amt:g}"
        out.append({"action": nm, "freq": round(float(strat[a]), 3), "amount_bb": amt})
    return out


def _near_label(sol) -> str:
    """What a flop's approximate answer is read off (instant_from_near, Advisor._study_near): the stored flops most
    like it, or for a pot the library holds no spot for or another stack depth, the spot borrowed and how."""
    flops = ", ".join(nm for nm, _ in sol.near)
    if not sol.other:
        return f"the precomputed solves of the most similar flops ({flops}), by card substitution"
    # (a pot the library holds no spot for: the nearest spot's)
    what = (f"the precomputed solve of this flop in the library's {sol.label}" if sol.near[0][1] == 0 else
            f"the library's {sol.label} on the most similar flops ({flops}), by card substitution")
    if sol.spr is None:
        return f"{what}, the spot whose ranges are nearest this pot's"
    if getattr(sol, "pots", None):
        # (another game's pot: antes, a straddle, heads-up blinds; the same seat raised last preflop)
        return (f"{what}, another game's pot ({sol.pots[0]:g} bb, a stack-to-pot ratio of {sol.spr[0]:.1f}; this "
                f"one {sol.pots[1]:g} bb, {sol.spr[1]:.1f}): the spot whose ranges are nearest this pot's with the "
                "same preflop raiser")
    # (another stack depth)
    label = (f"{what}, solved {'deeper' if sol.spr[0] > sol.spr[1] else 'shallower'} (a stack-to-pot ratio "
             f"of {sol.spr[0]:.1f}, this pot's {sol.spr[1]:.1f})")
    return label if sol.same_seats else label + ", the spot whose ranges are nearest this pot's"


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
# The same where the spot stores no turn files from the near spot's file that holds the line (NEAR_SEED,
# _near_turn_files), and after a flop solved live from that solve's own turn (PREV_SEED, _prev_turn_strategies). On
# the threaded tree (26 turns, natively, against each solved cold to 1000 iterations): 12 CO, HJ and UTG vs BB turns
# off BTN vs BB's file stop at 300 iterations where 3 took 400 (mean 325 -> 300), each combo's EVs over the actions it
# plays 0.264 -> 0.166% of the pot off, EV given up 0.0045 -> 0.0044; 8 limped turns 450 -> 413, 0.213 -> 0.196,
# 0.0159 -> 0.0133; 6 at 40 and 60bb 300 either way, 0.148 -> 0.140 (README, "The turn solve starts from the near
# spot's stored turn, or from the flop's own solve")
NEAR_SEED = True
PREV_SEED = True

# A library flop whose spot stores no turn files (the spots solved before them, and the pots of 8- and 9-max read as
# theirs: flop_approx.near_exact) reads its turn off the nearest spot that stores them (_near_turn_spots): the same
# preflop raiser in the same seat, as many raises preflop (NEAR_TURN_RAISES), the pot and the stack-to-pot ratio
# within NEAR_TURN_POT of its own and the two ranges within NEAR_TURN_DIST (summed flop_approx.range_dist), until the
# turn's own solve has run NEAR_TURN_UNTIL iterations (instant_from_own reads it from there). scripts/
# bench_turn_near.py (six flops of each pot, its two likeliest flop lines, a turn each; each combo's mix priced by the
# turn's live Accurate solve's EVs per action, % of the pot, first to act / in position after a check): the single-
# raised pots off BTN vs BB's turn files (CO, HJ and UTG vs BB, 1.04-1.89 apart; BTN and CO vs SB, 1.66 and 2.37;
# 8-max CO vs BB, 9-max UTG vs BB) 0.04-0.23 / 0.08-0.21, the live Fast solve at 50 iterations 0.08-0.13 / 0.07-0.11
# (4-15% / 13-21% of the reach off the near file: left to the live solve); the 3bet pots off another 3bet pot's
# 0.03-0.33 / up to 0.64 where that solve gives up 0.04-0.06 / 0.05-0.13 (the blinds' 3bets of the BTN off BB vs CO's
# file 5-13 times as much), so they're left out.
NEAR_TURN_RAISES = (1,)
NEAR_TURN_POT = (0.8, 1.25)
NEAR_TURN_DIST = 2.4
NEAR_TURN_UNTIL = HU_CHECKS[0]
NEAR_TURN_HELD = 0.9   # (Study's whole range: the share of the actor's the near spot's turn file must store)


def _near_turn_spots(lib, sid) -> List[dict]:
    """The library spots (index rows) whose turn files a spot `sid` that stores none of its own reads its turns off
    (instant_from_store), nearest ranges first; [] for a spot that stores turn files, or with none near enough."""
    memo = lib.__dict__.setdefault("_near_turns", {})   # (by spot, kept with the library and its index)
    if sid not in memo:
        from . import flop_approx as fa
        spots = list(lib.index().get("spots", ()))
        sp = next((s for s in spots if s.get("id") == sid), None)
        out = []
        raises = str((sp or {}).get("line") or "").count(":r")
        if sp is not None and not sp.get("turns") and raises in NEAR_TURN_RAISES:
            mine, aggr = fa._spot_ranges(sp), fa.spot_aggressor(sp)
            for s in spots:
                spr = (float(s["stack"]) / float(s["pot"])) / (float(sp["stack"]) / float(sp["pot"]))
                if s is sp or not s.get("turns") or mine is None or aggr is None or fa.spot_aggressor(s) != aggr or \
                        str(s.get("line") or "").count(":r") != raises or not NEAR_TURN_POT[0] <= spr <= NEAR_TURN_POT[1] \
                        or not NEAR_TURN_POT[0] <= float(s["pot"]) / float(sp["pot"]) <= NEAR_TURN_POT[1]:
                    continue
                st = fa._spot_ranges(s)
                if st is None:
                    continue
                d = fa.range_dist(mine["full0"], st["full0"]) + fa.range_dist(mine["full1"], st["full1"])
                if d <= NEAR_TURN_DIST:
                    out.append((d, s))
        memo[sid] = [s for _, s in sorted(out, key=lambda x: x[0])]
    return memo[sid]


def _near_turn_files(lib, sol, flop):
    """(a decoded turn file, the hand's flop's suit map, the spot it's from) for each spot near the one the library
    entry `sol` (a LibrarySolution) was read from (its own, or the one a borrowed pot reads as its own) that stores
    this flop's turns, nearest first, where that spot stores none (_near_turn_spots); each fetched as it's reached."""
    head = getattr(sol, "head", None)
    if lib is None or not hasattr(lib, "turns") or not hasattr(sol, "stored") or not isinstance(head, dict):
        return
    from . import flop_library as fl
    canon, smap = fl.canonical(flop)
    name = cards_str(canon)
    if name != head.get("board"):
        return
    for sp in _near_turn_spots(lib, head.get("spot")):
        t = lib.turns(sp, name)
        if t is not None:
            yield t, smap, sp


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


def _prev_turn_strategies(sol, ch, card: int) -> Optional[Callable]:
    """The turn as a live flop solve `sol` plays it under `ch`, the turn's deal where the flop line ended (instant_from_
    prev's read, for every combo at once): the card dealt read as its class's played card, each combo as its image
    under the swap of the two, by its hand bucket where the solve plays the turn by buckets. at -> {"names", "player",
    "strat" (A, 1326): each combo's mix, NaN where the solve doesn't hold it} at the turn's actions `at` from its first
    decision (flop_library.turn_strategies' shape), or None where it has no such node; None where `sol` isn't a live
    flop solve or the line doesn't end in the turn's deal."""
    if not isinstance(sol, Solution) or hasattr(sol, "stored") or not isinstance(ch, Chance) or len(ch.boards) != 1:
        return None
    dec = ch.child
    rows = dec.boards
    if len(rows) != ch.K or isinstance(dec, RiverEntry) or len(rows[0]) != 4:
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
    played = rows[k][-1]
    ids = None
    if isinstance(dec, BucketEntry):
        T, ids = sol.solver._btab(dec)[:2]
        ids = np.asarray(ids[k % T])
        dec = dec.child
    pos = np.full(N_COMBOS, -1, dtype=np.int64)
    pos[np.asarray(sol.ctx.active)] = np.arange(len(sol.ctx.active))
    sw = lambda c: card if c == played else played if c == card else c
    img = np.array([COMBO_INDEX[(min(sw(x), sw(y)), max(sw(x), sw(y)))] for x, y in ALL_COMBOS], dtype=np.int64)
    j = pos[img]   # (each combo's column in the flop solve, -1 where it doesn't hold its image)

    def strategies(at):
        node = dec
        for name in at:
            if not isinstance(node, Decision) or name not in node.names:
                return None
            node = node.children[node.names.index(name)]
        if not isinstance(node, Decision):
            return None
        s = np.asarray(node.avg_strategy()[k], dtype=np.float64)   # (actions, the solve's combos or buckets)
        col = j if ids is None else np.where(j >= 0, ids[node.player][np.maximum(j, 0)], -1)
        ok = (col >= 0) & (col < s.shape[1])   # (a bucket dead on the class's played card: none)
        out = np.full((len(node.names), N_COMBOS), np.nan)
        out[:, ok] = s[:, col[ok]]
        return {"names": list(node.names), "player": node.player, "strat": out}
    return strategies


def instant_from_store(adv: "Advisor", spot: InstantSpot) -> Optional[dict]:
    """A library flop's turn: the precomputed flop solve's own turn strategies, stored beside its entry (the entry's
    turn file: flop_library.encode_turns, FlopLibrary.turns, turn_node), on the flop lines most hands take: the
    turn's first decision, and in position after a check, at the flop solve's turn sizes (a 75% bet), the card dealt
    read as its class's played card (each combo the same combo's; one holding the played card its image's holding
    the card dealt). On the line the flop's walk took (its sizes at their nearest) and on the entry's canonical flop;
    not once the turn's own solve in hand has run STORE_UNTIL iterations (instant_from_own reads that). Against a live
    Accurate turn with the opponent free to adapt (BTN vs BB on Kc7d2h) it gives up 0.50 / 0.67% of the pot out of /
    in position on the played cards, 0.81 / 0.63 on the others; priced by the turn's own solve's EVs 0.04 / 0.08.
    A format-2 turn file (flop_library.turn_evs) adds the flop solve's EVs there, its turn class's, never your hand's
    own: each action's "range_ev_bb" (that of the hands in your range that take it; None where none does) and at the
    turn's first decision "range_ev" (your range's and theirs), in big blinds relative to folding now, scaled from the
    stored line's pot to the hand's. (Against the library's own game solved live, river dealt: the ranges' 0.15-0.36%
    of the pot on average, at most 0.73; the actions' 0.16-0.34, at most 1.0. A turn report read off a flop solved
    live, the river by strength buckets, differs by more: 0.6-0.9% of the pot in single-raised pots, 1.6-4.5 in 3bet
    pots.)"""
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
    if got is not None:
        files = [got + (None,)]
    elif spot.own is not None and spot.own.solver.iters >= NEAR_TURN_UNTIL:
        return None
    else:   # (a spot with no turn files of its own: the nearest that stores this flop's, _near_turn_spots)
        files = _near_turn_files(lib, sol, spot.board[:3])
    from . import flop_library as fl
    for t, smap, src in files:
        mp = lambda c: c // 4 * 4 + smap[c % 4]
        nd = fl.turn_node(t, taken, at, mp(spot.board[3]))
        if nd is None or nd["player"] != spot.hero_p:
            continue
        x, y = sorted(mp(c) for c in spot.hole)
        i = COMBO_INDEX[(x, y)]
        j = int(np.searchsorted(nd["combos"], i))
        if j < len(nd["combos"]) and nd["combos"][j] == i:
            break
        # (a combo the flop solve hardly brings here: not stored)
    else:
        return None
    node = types.SimpleNamespace(names=nd["names"], pot_now=0.0, bets=(0.0, 0.0))
    acts = _real_actions(node, nd["strat"][:, j], [0.0, 0.0], spot)
    out = {"actions": acts, "source": "store", "label": _store_label(nd["names"], src), "iterations": int(sol.iters)}
    if src is not None:   # (another spot's: no EVs, they're its pot's and ranges', not this one's)
        out["near_spot"] = src.get("id")
        return out
    evs, rng = _store_evs(t, taken, at, mp(spot.board[3]), nd, spot.pot, spot.hero_p)
    if evs is not None:   # (format 2: the class's range EVs, at the hand's pot; never your hand's own)
        for a, e in zip(acts, evs):
            a["range_ev_bb"] = e
        if rng is not None:
            out["range_ev"] = rng
    return out


def _store_label(names, src: Optional[dict] = None) -> str:
    """What a stored turn's answer is read off (instant_from_store, Advisor._study_store): src, the spot whose turn
    file it is where that isn't the hand's own (_near_turn_spots)."""
    bets = [n for n in names if n.startswith("bet")]
    size = f"one bet size, {bets[0].split()[-1]}" if len(bets) == 1 else "its turn sizes"
    if src is not None:
        return (f"the precomputed flop solve's view of the turn in the library's {src.get('label') or src.get('id')} "
                f"(by card class, {size}), the nearest spot that stores its turns")
    return f"the precomputed flop solve's view of the turn (by card class, {size})"


def _store_evs(t: dict, path, at, card: int, nd: dict, pot: float, p: int):
    """A format-2 turn file's EVs at its stored node `nd` (flop_library.turn_node of t on the flop line `path`, the
    turn actions `at`, `card` on the entry's flop): the flop solve's, its turn class's, never a hand's own, in bb
    relative to folding now, scaled from the stored line's pot to the hand's `pot`: (each action's, that of the hands
    of the acting player's range that take it, None where none does; at the turn's first decision the ranges', player
    p's and the other's, else None). (None, None) for a format-1 file: strategies only."""
    if nd.get("evs") is None:
        return None, None
    from . import flop_library as fl
    ln = next(x for x in t["lines"] if x["path"] == list(path))
    k = pot / max(float(ln["pot"]), 1e-9)
    acts = [round(float(e) * k, 2) if np.isfinite(e) else None for e in nd["evs"]]
    r = fl.turn_range_evs(t, path, card) if not at else None
    rng = (round(float(r[p]) * k, 2), round(float(r[1 - p]) * k, 2)) if r is not None and np.all(np.isfinite(r)) else None
    return acts, rng


# what Study asks without (Advisor.study): no cards of anyone's own, at equilibrium, every hand of the range
STUDY_DROPS = ("hero", "hole", "opponents", "locks", "jam", "library_only", "optimize_sizes")


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
        # and a spot of the same pot and stack whose ranges are a few combos from this pot's (another table size's:
        # flop_approx.near_exact), read as its own, marked borrowed; False: the library's own spots alone
        self.library_borrow = True
        # a raked table's flop of a library spot: an answer at once off the spot's unraked entry while the flop's
        # own solve runs (instant_from_unraked); False: none
        self.rake_library = True
        # precomputed 3-way flop solves (gto/mw_library.py: MWLibrary): a 3-way flop's first decisions, at every speed
        self.mw_library = None
        # the street advised on (heads-up from its start) also gets the earlier streets' prior_budget its narrowing
        # didn't spend (native Fast: README, Accuracy vs speed); off: time_budget alone
        self.spare_prior = False
        self.rng = random.Random()
        self._breakdown = False   # this request asked for the breakdown (state "breakdown": true)
        self._lib_only = False    # state "library_only": a flop the library holds, else {"skipped": …}, no solve
        self._study = False       # state "study": Study mode's answer (Advisor.study)
        self._seen_from = None    # play's fallback: (seat, hand) of the player practising, whose ranges it solves with
        self._runout_close = None # Study's runout report: the action that closed the flop or turn (Advisor._runouts)
        self._runout_held = False # ...from what's in hand only: the library's stored turns, cached solves (_NotHeld)
        self._runout_lines = None # ...the stored turn's flop lines, where the file is there but not for this line
        self._runout_state = None # ...the request at the decision that closed the street (a 3-way report solves it)
        self._mw_nolib = False    # a 3-way flop decision solved live though the 3-way library holds it (the report)
        self._mw_borrow = None    # the table's size, where a 3-way pot of another one may read a library spot (no
        # ranges of your own: MWLibrary.borrow), else None
        self._showdown = False    # state "equity": hero's equity against the ranges once betting is closed (_equity_answer)
        self._req_sizes: Dict[str, List[float]] = {}   # the request's custom sizes (state "sizes"), as fractions
        self._one_size = None     # state "one_size": the one-size view's request (Advisor._one_size_view), else None
        # ...its solves stop this close to equilibrium (% of the pot, by street; the lower expl_target where set)
        self.one_size_settled: Dict[str, float] = dict(ONE_SIZE_SETTLED)
        self._one_rows: Dict[tuple, dict] = {}   # its rows solved here, by spot and size (ONE_SIZE_CACHE, oldest first)
        self.one_size_cut = True    # (a size clearly behind the best stops early: ONE_SIZE_CUT)
        # Advisor.instant's sources, tried in order: each (advisor, InstantSpot) -> {"actions", "source", "label"} or None
        self.instant_sources = [instant_from_store, instant_from_off_tree, instant_from_near, instant_from_unraked,
                                instant_from_own, instant_from_prev]
        # instant_from_own facing a bet off the tree of your earlier decision's solve: its answers facing the sizes it
        # has read by the bet's price (_off_tree_own) where the bet is smaller or bigger than all of them ("outside";
        # "above": bigger only; "always": between two of them too; None: never), else the nearest size's; by street
        # ({"turn": ..., "river": ...}) or one for both (OFF_LATER)
        self.off_tree_later = dict(OFF_LATER)
        self._instant = False     # an Advisor.instant request: nothing is solved (_NoInstant where it would be)
        # Advisor.progress: by street ("flop", "turn"), the iterations at which a live street's solve stops to be read
        # (its answer so far, as instant_from_own reads it) before it goes on; none by default (the browser's worker
        # sets them). A look comes only where the solve has at least LOOK_MIN_S left to run to its first check.
        self.first_look: Dict[str, Tuple[int, ...]] = {}
        self._look = None         # (Advisor.progress running: the request's street's board length)
        self._looking = False     # (Advisor._study_look: Study's answer at a look, no equities worked out)

    # ------------------------------------------------------------ public
    _icm = None   # the tournament's prize model while a request is answered (_advise), else None

    def advise(self, state: dict) -> dict:
        if state.get("study"):
            return self.study(state)
        if state.get("triage"):
            return self.triage(state)
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
        this street under it), with "approximate": {source, label} and no EVs of your hand's (a library flop's turn
        file in format 2 adds its range EVs: each action's "range_ev_bb", the hands in your range that take it, and
        "range_ev_bb" / "villain_range_ev_bb", the ranges' at the turn's first decision). Nothing is solved: None
        where the answer needs a solve before anything can be read (an earlier street not solved yet), is quick anyway
        (preflop, a library flop, a street solved already), has 4 or more players from the flop, or asks for more than
        the equilibrium (exploit mode, locks, custom ranges or sizes, a tournament, a jam, the breakdown; Study's size
        menus answer at once on a heads-up turn or river only). In a 3-way pot your hand's mix where Study's reads below
        hold it (_mw_instant's `hole`), shaped as the heads-up answer with "multiway" and "villain" naming both. Study
        ("study", asked as
        Advisor.study asks: no cards of anyone's own) answers at once for the whole range on a library flop's turn from
        its stored turn (Advisor._study_store), on a flop the library doesn't store from the stored flops most like
        it (Advisor._study_near), and on a turn or river after the street before was solved live from that solve
        (Advisor._study_prev), on a 3-way flop the 3-way library doesn't store from its stored flops most like it
        (Advisor._study_mw_near), at a later 3-way node before the river from the hand's earlier solve of its street,
        or a turn's from the flop's (Advisor._study_mw_prev), and nowhere else."""
        if state.get("study"):
            state = dict({k: v for k, v in state.items() if k not in STUDY_DROPS}, study=True)
        if len(str(state.get("board") or "")) < 6 or state.get("opponents") == "pool" or any(
                state.get(k) for k in ("runouts", "jam", "breakdown", "optimize_sizes", "locks", "library_only",
                                       "payouts", "bounties", "ranges")) or state.get("one_size") is not None \
                or state.get("sizes") and not state.get("study"):
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

    def progress(self, state: dict) -> Optional[dict]:
        """A live street's answer in steps, for the browser to show while it solves (the worker posts each, the full
        answer replaces them): the request's own street's solve runs to its next look (first_look: by street, the
        iterations at which it stops) and stops there, cached, and its answer so far is read off it as Advisor.instant
        reads a solve stopped part way (instant_from_own: hero's mix as it stands, "approximate", no EVs); the
        request asked again goes on from it. Study ("study") the same, for the actor's whole range
        (Advisor._study_look). {"partial": that answer}; {"result": the answer} where it answered in full instead (a
        library flop, a solve cached or settled already, one with under LOOK_MIN_S left to its first check, a 3-way
        pot); None where nothing is looked at: as Advisor.instant (preflop, exploit mode, locks, custom ranges or
        sizes, a tournament, ...), a river (quick anyway), a runout report or a street with no looks."""
        if state.get("study"):
            state = dict({k: v for k, v in state.items() if k not in STUDY_DROPS}, study=True)
        board = str(state.get("board") or "")
        street = {6: "flop", 8: "turn"}.get(len(board))
        if street is None or not self.first_look.get(street) \
                or state.get("opponents") == "pool" or any(
                state.get(k) for k in ("runouts", "jam", "breakdown", "optimize_sizes", "locks", "library_only",
                                       "payouts", "bounties", "ranges", "sizes")) or state.get("one_size") is not None:
            return None
        self._look = len(board) // 2
        try:
            res = self.advise(state)
        except _FirstLook:
            res = None
        finally:
            self._look = None
        if res is not None:
            return {"result": res}
        got = self._study_look(state) if state.get("study") else self.instant(state)
        return {"partial": got} if got else None

    def triage(self, state: dict) -> dict:
        """The session analyzer's Quick triage ("triage": true): your decision answered fast enough to grade a whole
        session, from what is in hand first. Preflop the charts (as always); a flop the library holds (heads-up, or a
        3-way pot's first decisions) from it, as "library_only" asks; else an answer at once off what is in hand
        (Advisor.instant: the spot's stored flops most like this one, a library flop's stored turn, or the previous
        street's live solve, which plays this street under it: your hand's mix, no EVs of its own); else a Quick
        solve (TRIAGE_BUDGETS, "quick": an earlier street the hand hasn't solved narrowed by the betting rules). The
        answer carries "triage": {"source": "chart" | "library" | "approx" | "quick" | "rules", "confidence": "high"
        | "medium" | "low", "label"}: high where the analyzer's deep pass (the decision at Fast) answers the same (the
        charts; the library's own spot for the pot), medium for a spot borrowed for this pot, a library flop's stored
        turn, or a Quick solve within TRIAGE_SETTLED of equilibrium, low for the other reads (the nearest stored
        flops, the library's answers facing another bet size, a live solve's view of a later street or of its own
        street so far), a rougher Quick solve, or the betting rules (a multiway spot no solve answers). Measured on
        a 1,000-hand session against the same decisions at Fast: the stored turn 13 of 13 the same grade, a Quick
        solve within TRIAGE_SETTLED 88%, the nearest flops and an earlier street's solve 62%. A cancel or an error
        comes back as it is."""
        st = {k: v for k, v in state.items() if k != "triage"}
        street = {6: "flop", 8: "turn", 10: "river"}.get(len(str(st.get("board") or "")))

        def mark(res, source: str, conf: str, label: str) -> dict:
            if isinstance(res, dict) and not any(k in res for k in ("error", "cancelled", "skipped")):
                res["triage"] = {"source": source, "confidence": conf, "label": label}
            return res

        def lib(res) -> dict:
            pc = res.get("precomputed") or {}
            name = pc.get("label") or "precomputed"
            return mark(res, "library", "medium", f"{name}, read as this pot") if pc.get("borrowed") \
                else mark(res, "library", "high", name)

        if street is None:
            return mark(self.advise(st), "chart", "high", "the preflop charts")
        if street == "flop":
            res = self.advise(dict(st, library_only=True))
            if not isinstance(res, dict) or res.get("cancelled"):
                return res
            if not any(k in res for k in ("skipped", "error")) and res.get("precomputed"):
                return lib(res)
        try:
            got = self.instant(st)
        except Cancelled:
            raise
        except Exception:   # noqa: BLE001  (nothing read at once: solved below)
            got = None
        if got:
            ap = got.get("approximate") if isinstance(got.get("approximate"), dict) else {}
            return mark(got, "approx", "medium" if ap.get("source") == "store" else "low",
                        ap.get("label") or "approximate")
        budget, prior = TRIAGE_BUDGETS[street]
        res = self.advise(dict(st, quick=True, budget=budget, prior_budget=prior))
        if not isinstance(res, dict) or any(k in res for k in ("error", "cancelled", "skipped")):
            return res
        if res.get("precomputed"):
            return lib(res)
        if res.get("multiway") and not res.get("solved") or res.get("fallback"):
            return mark(res, "rules", "low", "the betting rules")
        sv = res.get("solver") or {}
        x = sv.get("exploitability_pct")
        settled = x is not None and x <= TRIAGE_SETTLED and not sv.get("approximate")
        return mark(res, "quick", "medium" if settled else "low",
                    f"a Quick solve ({sv.get('iterations')} iterations" + (f", {x}% of the pot from equilibrium)"
                                                                          if x is not None else ")"))

    def _study_look(self, state: dict) -> Optional[dict]:
        """Advisor.progress's look for Study: Study's answer off the solves in hand as they stand, the street's own
        stopped at its look (nothing is solved: a runout report's held answer, _NotHeld where one is missing), the mix
        alone (no EVs or equities until the answer: _no_values; the equities aren't worked out), marked approximate
        ("this street's solve so far"); None where it doesn't come out as a heads-up node."""
        self._runout_held = self._looking = True
        try:
            with np.errstate(all="ignore"):
                res = self._advise(state)
        except _NotHeld:
            return None
        finally:
            self._runout_held = self._looking = False
        if not isinstance(res, dict) or res.get("study") != "node" or res.get("multiway") or "grid" not in res \
                or _unusable(res):
            return None
        _no_values(res)
        for c in (res.get("opponent_grid") or {}).get("cells", {}).values():
            c["eq"] = None
        s = res.get("solver") or {}
        it, ex = int(s.get("iterations") or 0), s.get("exploitability_pct")
        label = f"this street's solve so far ({it} iterations" + (
            f", {ex}% of the pot from equilibrium)" if ex is not None else ")")
        res.pop("breakdown", None)
        res.update(range_ev_bb=None, range_eq=None, opponent_eq=None,
                   approximate={"source": "solve", "label": label, "iterations": it},
                   # (study.js's line names the label, its stats in it, as a 3-way look's off an earlier decision)
                   solver=dict(s, iterations=None, exploitability_pct=None),
                   notes=[f"approximate: {label}; no EVs or equities until its answer"] + list(res.get("notes") or []))
        return res

    def _look_at(self, n: int, sol) -> Optional[int]:
        """Advisor.progress: the iterations the request's own street's solve (`sol`, None before it starts; `n` its
        board's cards) stops at for its next look, or None (not progress's request, not its street, or no look left).
        A shallow turn's (a 3bet pot's) are the turn's."""
        if self._look != n or self._instant:
            return None
        done = 0 if sol is None else sol.solver.iters
        return next((c for c in self.first_look.get("flop" if n == 3 else "turn", ()) if c > done), None)

    def _looked(self, sol, look: Optional[int], first: int) -> bool:
        """Whether a solve run to its look (`look`) stops there for progress to read: not where it fell short of it
        (out of time), settled, or with under LOOK_MIN_S left at its pace so far to where it next may stop (the
        next of HU_CHECKS from `first`, its first check, on; `first` 1000: no target, on to its 1000th)."""
        if look is None or sol.solver.iters < look or getattr(sol, "settled_at", None) is not None:
            return False
        per = getattr(sol.solver, "per_iter", None) or sol.seconds / max(sol.solver.iters, 1)   # (the core's, its setup out)
        it = sol.solver.iters
        nxt = 1000 if first >= 1000 else next((c for c in HU_CHECKS if c >= first and c > it), 1000)
        return per * max(nxt - it, 0) >= LOOK_MIN_S

    def _go_on(self, sol, n: int, left: Callable[[], float], target: Optional[float], first: int,
               look: int) -> None:
        """_solve_cached: a solve run to a look of progress's (`look`) stops there to be read (_FirstLook), else goes
        on to the next look, and so on, then to the answer: a look passed by (too near where the solve next may stop)
        doesn't take the later ones with it. `left()`: the seconds of its budget left."""
        while True:
            if self._looked(sol, look, first if target is not None else 1000):
                raise _FirstLook()
            if sol.iters < look or sol.iters >= 1000 or getattr(sol, "settled_at", None) is not None:
                return   # (out of time, or settled at the look: the answer)
            look = self._look_at(n, sol)
            if target is not None:
                self._hu_run(sol, left(), target, first, stop_at=look)
            else:
                try:
                    sol.extend(left(), max_iters=look or 1000, should_stop=self.should_stop, reserve=HU_RESERVE)
                except Cancelled:
                    sol.stopped = True
                    raise
            if look is None:
                return

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
        the player first to act ("library_only": nothing is solved; Study's request, with no seat or cards, as a seat
        still in the pot), and the turn files of the flop's entries the index lists (or on a flop the library doesn't
        store, the stored flops its approximate answer reads). The files it asked for and didn't get stay unread (they
        would otherwise be kept as missing)."""
        board = str(state.get("board") or "")[:6]
        libs = [x for x in (self.library, self.mw_library) if x is not None and hasattr(x, "_entries")]
        if len(board) < 6 or not libs:
            return
        st = {k: v for k, v in state.items() if k not in ("study", "runouts", "jam", "breakdown", "optimize_sizes",
                                                           "one_size")}
        st.update(board=board, flop=[], turn=[], river=[], library_only=True)
        if not st.get("hero"):   # (Study's: no seat or cards of its own; a seat still in the pot, then the one to act,
            # and two cards off the board: the dry run reads no hand's answer)
            folded = {str(a.get("pos", "")).upper() for a in st.get("preflop") or () if a.get("action") == "fold"}
            st["hero"] = next((x for x in pf.SEATS.get(int(st.get("players", 6)), ()) if x not in folded), "")
            st["hole"] = "".join([c for c in ("2c", "2d", "3c", "3d", "4c") if c not in board][:2])
        before = [set(x._entries) for x in libs]
        try:
            res = self.advise(st)
            m = re.match(r"it is (\w+)'s turn", str(res.get("error", "")))
            if m:
                self.advise(dict(st, hero=m.group(1)))
            # this flop's entries' turn files too, where the index lists them (the turn's approximate answer at once:
            # instant_from_store), and for a spot that stores none, the nearest spot's that does (_near_turn_spots)
            lib = self.library
            if lib is not None and lib in libs and hasattr(lib, "turns"):
                from .flop_library import flop_name
                name = flop_name(parse_cards(board))
                spots = {s.get("id"): s for s in lib.index().get("spots", ())}
                for k in [k for k in lib._entries if len(k) == 2 and k[1] == name and k[0] in spots]:
                    if lib.turns(spots[k[0]], name) is None:
                        for sp in _near_turn_spots(lib, k[0]):
                            if lib.turns(sp, name) is not None or (sp["id"], name, "turns") in lib._entries:
                                break
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
        (no pool, locks or jam requests) and without hero's cards (no card removal for them). A tournament's payouts
        and bounties stay: the answers are in prize equity as the advice's are (preflop the charts' mix as the advice
        re-prices it, postflop the solves' payoffs), with the tournament's notes for the player to act."""
        st = {k: v for k, v in state.items() if k not in STUDY_DROPS}
        st["study"] = True
        runouts = st.pop("runouts", None)
        pfs = st.pop("preflop_solve", None)
        if pfs and not str(st.get("board") or ""):
            return self._study_pf(st, pfs)
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

    def _study_pf(self, st: dict, pfs) -> dict:
        """Study's Solver source preflop (state "preflop_solve"; gto/pf_study.py): the node from the solve of the page's
        game (pf_game's equilibrium: every class's mix and EVs), where one is in hand (solved here, or stored: the
        standard games); else {"study": "pf_unsolved", "game", "key", "estimate_s"} (the page offers to solve it) or
        {"study": "pf_none", "reason"} (a game the solve doesn't take). "preflop_solve": {"run": seconds} runs the
        game's solve that much more (from where it was) and answers {"study": "pf_progress", "pf_solve": its status}.
        {"export": true} packs a finished solve run here for the page to keep ({"study": "pf_export", "key",
        "iterations", "mbb", "blob"}; "blob" null if there's none), and {"load": blob, "iterations": n} takes it back
        (where the worker no longer holds it) before answering the node."""
        from . import pf_study
        t0 = time.time()
        spec, why = pf_study.spec_of(st)
        if spec is None:
            return {"study": "pf_none", "street": "preflop", "reason": why}
        if isinstance(pfs, dict) and pfs.get("run"):
            sec = max(0.5, min(30.0, float(pfs["run"])))
            s = pf_study.solve(spec, sec, self.should_stop)
            return {"study": "pf_progress", "street": "preflop", "pf_solve": s.status(),
                    "seconds": round(time.time() - t0, 2)}
        if isinstance(pfs, dict) and pfs.get("export"):
            out = pf_study.export(spec) or {"key": spec.key, "blob": None}
            return {"study": "pf_export", "street": "preflop", **out}
        if isinstance(pfs, dict) and pfs.get("load") and pf_study.held(spec) is None:
            try:
                pf_study.load(spec, str(pfs["load"]), int(pfs.get("iterations") or 0))
            except (ValueError, TypeError):   # (another tree's, or not a solve: the game offered as unsolved)
                pass
        s = pf_study.held(spec)
        if s is None:
            return {"study": "pf_unsolved", "street": "preflop", "game": spec.label(), "key": spec.key,
                    "estimate_s": round(pf_study.estimate_seconds(spec)), "seconds": round(time.time() - t0, 2)}
        res = s.node(_norm_actions(st.get("preflop"), float(st.get("bb", 1.0))))
        res["seconds"] = round(time.time() - t0, 3)
        return res

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
        flop lines a stored turn holds, where there is one} and nothing is solved. A 3-way pot's report reads the
        hand's 3-way solve of the street (Advisor._mw_runouts)."""
        street = {3: "flop", 4: "turn"}.get(len(parse_cards(str(st.get("board") or ""))))
        line = list(st.get(street) or []) if street else []
        if not line or st.get("river"):
            return {"error": "the runout report needs a flop or turn whose betting is done (and the next card not dealt)"}
        self._runout_close = str(line[-1].get("action"))
        self._runout_held, self._runout_lines = held, None
        self._runout_state = dict(st, **{street: line[:-1]})
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
            self._runout_close, self._runout_held, self._runout_lines, self._runout_state = None, False, None, None

    def _mw_runouts(self, hero, hole, board, rep, stk, streets, notes, t0) -> dict:
        """The runout report (Advisor._runouts) in a 3-way pot: the hand's latest live 3-way solve of the street that
        closed whose tree holds the line (Study's own answer at the closing decision, or an earlier one's), followed
        to the street's end (however rarely taken), and its next street read as it plays it: per card class (a
        rank's cards, a flush suit's apart; MultiwaySolver.tabs) by hand buckets at one bet size: the first
        decision's mix over its player's range there (its bucket mixes weighted by its range's mass in each bucket
        on the class's card), a river's the second player's after a check too. Its range on each card weights the
        averages; each card's chance is given the three ranges (their card overlaps taken as independent). No EVs or
        equities, and on the turn no later decision: a flop solve's were far from each turn's own solve (betting
        16-19 points less: Advisor._study_mw_prev), where a turn solve's river after a check was 4 points from each
        river's. With nothing in hand: held, _NotHeld; else the closing decision solved live as Study answers it (a
        flop the 3-way library holds too: its entries hold the flop only)."""
        from . import multiway_solver as MW
        cur = rep[-1]
        nxt = "turn" if cur.name == "flop" else "river"
        acts = list(streets.get(cur.name) or []) + [{"pos": hero, "action": self._runout_close}]

        def read():
            for k, (sol, pl, expl) in self._mw_hits(board, cur, stk, hole, streets):
                if not isinstance(sol, MW.MultiwaySolver) or getattr(sol, "tabs", None) is None or not sol.levels \
                        or len(pl) != 3:
                    continue
                R = np.asarray(sol.reach0[:, 0, :], dtype=np.float64).copy()
                node = sol.root
                try:
                    for a in acts[k:]:
                        node = _mw_follow(node, R, a, pl, rare=0.0)
                except _NoInstant:
                    continue
                if isinstance(node, MW.Runout):
                    return node
                loc = next(((g, b) for g in sol.levels[0] for b, (e, _) in enumerate(g.blocks) if e is node), None)
                if isinstance(node, MW.Exit) and loc is not None:
                    return sol, pl, expl, R, node, loc
            return None

        got = read()
        if got is None and not self._runout_held and self._runout_state is not None:
            close, self._runout_close, self._mw_nolib = self._runout_close, None, True
            try:     # (Study's own answer at the closing decision: its solve, kept in _mw_lines)
                self._advise(dict(self._runout_state))
            finally:
                self._runout_close, self._mw_nolib = close, False
            got = read()
        if got is None:
            if self._runout_held:
                raise _NotHeld()
            return {"error": f"no {nxt} to report on: the 3-way solve doesn't play the {nxt} on this line"}
        if isinstance(got, MW.Runout):
            return {"error": f"no {nxt} to report on: nobody can bet any more"}
        sol, pl, expl, R, node, (g, b) = got
        tb = sol.tabs
        if not isinstance(g.tree, MW.BDecision):
            return {"error": f"no {nxt} to report on: nobody can bet any more"}
        C = tb.C
        X = tb.to_buckets(R[:, None, :].astype(np.float32))      # (np, C, K): each player's reach by class and bucket
        n_cards = np.array([len(cs) for _, cs in tb.cls], dtype=np.float64)

        def part(d):
            S = d.avg_strategy()
            m = np.asarray(X[d.player], dtype=np.float64)          # (C, K)
            tot = m.sum(axis=1)
            f = np.stack([(S[b * C + t] * m[t][None, :]).sum(axis=1) for t in range(C)]) \
                / np.maximum(tot, 1e-12)[:, None]
            wk = n_cards * tot
            avg = (f * wk[:, None]).sum(axis=0) / max(float(wk.sum()), 1e-12)
            return {"actor": pl[d.player], "actions": list(d.names), "avg": [round(float(x), 4) for x in avg],
                    "cards": [{"card": card_str(tb.cls[t][0]), "cards": [card_str(c) for c in tb.cls[t][1]],
                               "f": [round(float(x), 4) for x in f[t]], "w": round(float(tot[t]), 3)}
                              for t in range(C) if tot[t] > 1e-9]}

        d1 = g.tree
        d2 = None    # (the river's second player after a check; not the turn's)
        if nxt == "river" and "check" in d1.names and isinstance(d1.children[d1.names.index("check")], MW.BDecision):
            d2 = d1.children[d1.names.index("check")]
        # each card's chance: the share of the three ranges' hands that leave it live (overlaps independent)
        dead = set(board)
        w = [sol.expand(R[i]) for i in range(len(pl))]
        cards = []
        live = np.ones(52)
        for i in range(len(pl)):
            tot = max(float(w[i].sum()), 1e-12)
            hold = np.zeros(52)
            for ci, cb in enumerate(ALL_COMBOS):
                if w[i][ci] > 0:
                    hold[cb[0]] += w[i][ci]
                    hold[cb[1]] += w[i][ci]
            live *= np.clip(1.0 - hold / tot, 0.0, 1.0)
        off = [c for c in range(52) if c not in dead]
        z = max(float(sum(live[c] for c in off)), 1e-12)
        cards = [{"card": card_str(c), "p": round(float(live[c] / z), 5), "eq": None, "ev": None} for c in off]
        K = d1.regret.shape[2]
        notes = list(notes) + [
            f"the {cur.name} solve's view of the {nxt}: one bet size there, each {nxt} class's cards alike (a rank's "
            f"cards, a flush suit's apart) and hands by {K} buckets a player; strategies only (no EVs or equities "
            f"3-way{'' if d2 is not None else ', nor the decisions after the first'}); a {nxt} dealt is solved on its "
            f"own, with more sizes"]
        return {"study": "runouts", "street": nxt, "multiway": True, "first": part(d1),
                "after_check": part(d2) if d2 is not None else None, "seats": list(pl),
                "pot_bb": round(float(node.pot), 2), "cards": cards, "source": "solve",
                "solver": {"iterations": int(sol.iters), "exploitability_pct": round(float(expl), 3),
                           "by": "hand buckets"},
                "notes": notes, "seconds": round(time.time() - t0, 2)}

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
        range equities, and each player's EV there where the file holds the flop solve's EVs (format 2: its turn
        class's, as a live report's; format 1 holds strategies only, so no EVs). None where the entry has no turn file,
        or it doesn't hold the hand's flop line (the lines it holds go to self._runout_lines)."""
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
        ev_of = None
        if ln.get("evs") is not None:        # (format 2: each card its class's range EVs, on the entry's suits)
            from .flop_library import turn_range_evs
            ev_of = {c: turn_range_evs(t, path, c // 4 * 4 + smap[c % 4]) for c in range(52) if c not in set(board)}
            ev_of = {c: e for c, e in ev_of.items() if e is not None}
        notes = list(notes) + [
            "the flop library's stored turns: the precomputed flop solve's turn strategies (its turn sizes, each turn "
            "class's cards alike: a rank's cards, a flush suit's apart; mixes in 5% steps), "
            + ("and its EVs there (each turn class's)" if ev_of else "strategies only, so no EVs")
            + "; a turn dealt is solved on its own, with more sizes"]
        return {"study": "runouts", "street": "turn", "first": first, "after_check": after, "seats": list(seats),
                "pot_bb": round(float(ln["pot"]), 2), "cards": self._runout_cards(board, r[0], r[1], ev_of),
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
        # pots that see a flop; a tournament has none, whatever the rake given (its payouts or bounties, or
        # "tournament": an imported tournament hand's, hh.js)
        tourney = bool(state.get("payouts") or state.get("bounties") or state.get("tournament"))
        rate = 0.0 if tourney else max(0.0, min(0.25, float(state.get("rake") or 0.0)))
        cap = state.get("rake_cap")
        cap = float(cap) / bb if cap not in (None, "") and float(cap) > 0 else None
        positions = pf.Table(pf.SEATS[n], ante, straddle, (rate, cap) if rate > 0 else None)
        study = bool(state.get("study"))   # (Advisor.study: hero is whoever acts next)
        hero = str(state.get("hero") or "").upper()
        if hero not in positions and not study:
            return {"error": f"hero seat {hero} is not one of {positions}"}
        try:
            sizes = _custom_sizes(state["sizes"], STUDY_MAX_SIZES if study else MAX_SIZES) if state.get("sizes") else {}
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
        if model is not None and ("actions" in res or "one_size" in res or res.get("study") == "runouts"):
            out = {a["pos"] for s in STREETS for a in streets[s] if a["action"] == "fold"}
            res["tournament"] = _tournament_info(model, hero, positions, res.get("street"), out, named=study)
            res.setdefault("notes", []).insert(0, res["tournament"]["note"])   # (a practice move's answer has none)
            if "bounties" in res["tournament"]:
                res["notes"].insert(1, res["tournament"]["bounties"]["note"])
            if "one_size" in res:   # (the one-size panel shows them: simple.js)
                res["one_size"]["tournament"] = res["notes"][:1 + ("bounties" in res["tournament"])]
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
                        _narrow_street(ranges, board[:{"flop": 3, "turn": 4, "river": 5}[st.name]], st, stk, {})
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
                res["tournament"] = _tournament_info(model, hero, positions, res.get("street"), out, named=study)
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
        try:
            arriving = preflop_ranges(preflop, positions, stack).get(hero, np.ones(N_COMBOS))
        except Exception:   # noqa: BLE001
            arriving = np.ones(N_COMBOS)
        return {"street": "preflop", "hero": hero, "hand": hand_class(hole), "players": len(positions),
                "actions": acts, "recommended": max(acts, key=lambda a: a["freq"])["action"], "sampled": self._sample(acts),
                "pot_bb": round(st.pot_start + sum(st.bets.values()), 2), "to_call_bb": round(to_call, 2),
                "grid": strategy_grid(list(full.values()), [a["action"] for a in acts], arriving), "notes": [],
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
        if not board and state.get("equity") and actor is None:
            return self._showdown_preflop(hero, hole, streets["preflop"], positions, stack, t0,
                                          pool=state.get("opponents") == "pool", locks=locks)
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
        self._mw_borrow = None if custom else len(positions)
        # "equity": once a street's betting is closed (checked through, called, or everyone all-in), the ranges narrowed
        # through it as the next street's advice would narrow them, and your equity against them (_equity_answer)
        self._showdown = bool(state.get("equity")) and actor is None and not self._study
        try:
            res = self._postflop(hero, hole, board, stack, streets, positions, t0,
                                 pool=state.get("opponents") == "pool", actor=actor, custom=custom,
                                 optimize=bool(state.get("optimize_sizes")), locks=locks)
            if self._breakdown and "error" not in res and "breakdown" not in res:
                res["breakdown"] = None   # the betting rules answered (4+ players): no solve to break down
            return res
        finally:
            self.time_budget, self.prior_budget, self.cfg, self.root_sizes, self.spr_sizes = saved
            self._breakdown = self._lib_only = self._jam = self._study = self._showdown = False
            self._one_size = self._mw_borrow = None

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

    def _showdown_preflop(self, hero, hole, preflop, positions, stack, t0, pool=False, locks=None):
        """State "equity" preflop: once the preflop betting is closed, your equity against the ranges that took the
        line (the charts', with the pool's or your locks' preflop where set), over the board to come."""
        cur = replay_hand(positions, stack, {"preflop": preflop}, 1)[0]
        if hero not in cur.active:
            return {"error": "hero has folded"}
        if not cur.closed:
            return {"error": "the preflop betting isn't over: equity is answered once nobody can bet any more"}
        ranges = preflop_ranges(preflop, positions, stack, hero if pool else None, locks)
        notes = [f"{p}'s preflop line isn't in the chart, so their range is the hands that take its last action"
                 for p in getattr(ranges, "rebuilt", []) if p != hero and p in cur.active]
        return self._equity_answer(hero, hole, [], {p: ranges[p] for p in pf.postflop_order(positions)
                                                    if p in cur.active and p != hero}, cur, 1, notes, t0)

    def _equity_answer(self, hero, hole, board, opps: Dict[str, np.ndarray], cur: "StreetState", n_streets: int,
                       notes: List[str], t0) -> dict:
        """Hero's share of the pot at showdown against the opponents' ranges `opps` (narrowed through the closed
        street), over the cards to come (breakdown.hand_equity), and each opponent's range."""
        from .breakdown import hand_equity
        if not opps:
            return {"error": "the hand is over: nobody is left to show down against"}
        dead = np.prod([1.0 - MEMBER_FULL[c] for c in list(hole) + list(board)], axis=0)
        e = hand_equity(hole, board, list(opps.values()))
        pot = cur.pot_start + sum(cur.bets.values())
        rnd = lambda x: None if x is None else round(x, 4)
        villains = []
        for (p, w), eq in zip(opps.items(), e["vs"]):
            w = w * dead
            villains.append({"seat": p, "equity": rnd(eq), "combos": round(float(w.sum()), 1),
                             "range": range_to_text(w, 0.05)[:400],
                             "grid": strategy_grid([np.ones(N_COMBOS)], ["in range"], w)})
        if not e["exact"]:
            notes.append(f"equity against several ranges preflop is over {e['runouts']} sampled boards (about ±1%)")
        if len(opps) > 1:
            notes.append("against several ranges their hands are taken as independent (one's cards don't block "
                         "another's); a tie splits the pot among the tied")
        return {
            "showdown": True, "street": cur.name, "hero": hero, "hand": hand_class(hole), "board": cards_str(board),
            # the next street to deal (None: the river's betting closed, it's the showdown), and whether anyone can
            # still bet (False: everyone but at most one player is all-in, so the cards just run out)
            "next": STREETS[n_streets] if n_streets < len(STREETS) else None,
            "allin": sum(p not in cur.allin for p in cur.active) <= 1,
            "pot_bb": round(pot, 2), "equity": rnd(e["equity"]), "win": rnd(e["win"]), "tie": rnd(e["tie"]),
            # heads-up the whole pot is contested (an uncalled bet went back): your equity's share of it
            **({"share_bb": round(e["equity"] * pot, 2)} if len(opps) == 1 and e["equity"] is not None else {}),
            "villains": villains, "runouts": e["runouts"], "exact": e["exact"],
            "notes": notes, "seconds": round(time.time() - t0, 2),
        }

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
        names = [a["action"] for a in acts]   # (the grid's too: a jam is "all-in" there as in the actions)
        arriving = preflop_ranges(preflop, positions, stack).get(hero, np.ones(N_COMBOS))
        return {
            "street": "preflop", "hero": hero, "hand": hand_class(hole), "players": len(positions),
            "actions": acts, "recommended": best, "sampled": self._sample(acts),
            "pot_bb": round(st.pot_start + sum(st.bets.values()), 2), "to_call_bb": round(to_call, 2),
            "grid": strategy_grid(list(full.values()), names, arriving),
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
        if cur.closed and not self._showdown:
            return {"error": "this street's betting is already closed (deal the next card)"}
        if self._showdown and not cur.closed:
            return {"error": "this street's betting isn't over: equity is answered once nobody can bet any more"}
        if not self._showdown and cur.to_act != (actor[0] if actor else hero):
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
        # where the ranges this answer is solved against left the solves (the answer's "off_solution": a line the
        # solve rarely took, read by the betting rules; a range filled out; an earlier street narrowed by the rules,
        # or partly, by an unsettled solve). Its mix is the best answer to those ranges, not the whole hand's
        # equilibrium, and the page says so instead of a plain "converged".
        off_solution: List[dict] = []
        for p in [q for q in getattr(pf_r, "rebuilt", []) if q not in (custom or {})]:
            off_solution.append({"street": "preflop", "pos": p, "why": "filled"})
            notes.append(f"{'your' if p == hero else p + chr(39) + 's'} preflop line isn't in the chart, so "
                         f"{'your' if p == hero else 'their'} range is the hands that take its last action")
        if self._runout_close is not None and len(cur.active) > 2:
            if len(cur.active) == 3 and all(len(s.players_start) == 3 for s in rep[1:]):
                return self._mw_runouts(hero, hole, board, rep, stk, streets, notes, t0)
            big = max(rep[1:], key=lambda s: len(s.players_start))
            return {"error": f"the runout report reads heads-up and 3-way pots: {len(big.players_start)} players saw "
                             f"the {big.name}"}
        # (3-way and 4-way nodes before the river, three or four from the flop: a 3-way flop the library doesn't
        # store read off its stored flops most like it, Advisor._study_mw_near; a later node off the hand's earlier
        # solve, Advisor._study_mw_prev; Study's for the actor's whole range, the advice's for your hand alone)
        mw_inst = self._instant and cur.name != "river" and all(len(st.players_start) in (3, 4) for st in rep[1:])
        if self._instant and (actor or pool or optimize or locks or custom or self._runout_close is not None
                              or any(len(st.players_start) > 2 for st in rep[1:]) and not mw_inst
                              or self._req_sizes and (mw_inst or cur.name == "flop")):
            # (heads-up from the flop, or 3-way and 4-way as above, at equilibrium only; Study's size menus a turn or
            # river heads-up, off the street before's solve or a library flop's stored turn: a flop's reads are the
            # library's sizes)
            raise _NoInstant()
        if actor and actor[1] is None and any(len(st.players_start) > 2 for st in rep[1:]):
            return {"skipped": "multiway"}   # (a presolve: heads-up from the flop only, as an instant answer)
        # a 3-way flop's first decision the 3-way library holds: its precomputed solve answers (no narrowing needed)
        if len(cur.players_start) == 3 and actor is None and not pool and not optimize and not self._jam \
                and getattr(self, "_icm", None) is None and self._one_size is None and not self._mw_nolib \
                and not self._showdown:
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
            if self._showdown and not st.actions:
                continue   # (everyone all-in: nothing to narrow by)
            before = dict(ranges)
            _narrow_street(ranges, board[:{"flop": 3, "turn": 4, "river": 5}[st.name]], st, stk, slow)
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
        if mw_inst:   # (the ranges narrowed as the node's own solve will start from them)
            res = self._study_mw_prev(hero, hole, board, rep, stk, streets, ranges, positions or [], notes, t0)
            if res is None:
                raise _NoInstant()
            return res
        remaining = [p for p in cur.order if p in cur.active]
        pot_now = cur.pot_start + sum(cur.bets.values())
        to_call = cur.cur_bet - cur.bets[hero]
        hero_remaining = stk[hero] - cur.inv_start[hero] - cur.bets[hero]

        live_opps = [p for p in remaining if p != hero]
        if self._showdown and (len(live_opps) != 1 or cur.players_start != remaining):
            # (a street that began multiway is narrowed above, through its whole line)
            return self._equity_answer(hero, hole, board, {p: ranges[p] for p in live_opps}, cur, n_streets, notes, t0)
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
        # (No one's cards come out of the ranges: both play the whole range they got here with, and the solve's card
        # removal keeps your row off what you block. Out of the villain's range they'd make a different game, where
        # every other hand of yours plays as if it held your blockers too: TT called a 4bet pot's turn bet 100% where it
        # folds 79%, KK read that call 6 bb short on another board.)
        # (play's fallback, an opponent's move from its own seat: `seen` is the hand of the player practising, the
        # villain here, in at a near-zero weight where its range doesn't hold it, so the earlier streets' solves are the
        # ones that player's advice cached, and this street's too where this seat's hand has weight in it)
        seen = getattr(self, "_seen_from", None)
        seen = tuple(seen[1]) if seen is not None and seen[0] == villain and seen[1] and \
            cur.players_start == remaining else None   # (a pot heads-up from this street's start: solves to reuse)
        adv_p, adv_hole = (hero_p, hole) if seen is None else (1 - hero_p, seen)   # (the hand _as_advised puts in)

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
            # narrow through earlier streets that were also heads-up between these two (equity: the closed street too)
            for st in rep[1:] if self._showdown else rep[1:-1]:
                sub_board = board[:{"flop": 3, "turn": 4, "river": 5}[st.name]]
                # the next street's board: its ranges leave out the cards dealt by then, not later ones, so they are
                # the ranges that street's own advice solved with, and its solve is found in the cache (the river's
                # turn was solved again for its narrowing, its ranges short of the river card's combos: 3 s of the
                # browser's 3.4 s river)
                nxt_board = board[:len(sub_board) + 1]
                if st.players_start != remaining or (self._showdown and not st.actions):
                    continue   # multiway: narrowed above (or everyone all-in: nothing to narrow by)
                spare += self.prior_budget
                eff = min(stk[oop] - st.inv_start[oop], stk[ip] - st.inv_start[ip])
                cfg = self._cfg_with_real_sizes(st, oop, ip, eff)
                taken: List[str] = []
                resolved: List[tuple] = []   # (_walk's: a solve from the node after a rare line, where taken was cut)
                lib_off: List[dict] = []
                hit = self._from_library(lib, full, sub_board, r0, r1, st, eff, cfg, oop, ip, notes,
                                         holes=(hole, adv_hole), borrow=not custom, players=len(positions),
                                         smoothing=self.prior_smoothing,
                                         board=sub_board, rules=(0, 1), slow=slow_hu, taken=taken, off=lib_off,
                                         resolve=None if pool or adj else self._rare_resolver(
                                             sub_board, st, eff, dataclasses.replace(
                                                 cfg, **(self.prior_model or {}),
                                                 **({"root_allin": False} if cfg.root_allin else {}))),
                                         resolved=resolved) \
                    if lib is not None and st.name == "flop" else None
                if hit is not None:
                    sol, node, c0, c1 = hit
                    notes.append(sol.note())
                    off_solution.extend(lib_off)
                    if resolved:   # (the walk went on in a solve from the node after a rare line: its tree from there)
                        sol, cut = resolved[-1]
                        taken = taken[cut:]
                else:
                    resolved.clear()   # (a library walk that went on in a new solve, then wasn't covered)
                    if self.prior_model:
                        cfg = dataclasses.replace(cfg, **self.prior_model)
                    if cfg.root_allin:
                        cfg = dataclasses.replace(cfg, root_allin=False)   # (a jam request: the street advised on only)
                    # (your hand out of your range here: the ranges with it in, as the street's own advice solved, so
                    # that solve is the one read, not a second without it: Advisor._as_advised)
                    s0, s1 = self._as_advised(r0, r1, adv_p, adv_hole)
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
                        _narrow_street(rs, sub_board, st, stk, slow_rules, check_whole=True)
                        notes.append(f"{st.name}: ranges narrowed by simple betting rules (a quick answer: the street "
                                     f"isn't solved)")
                        off_solution.append({"street": st.name, "why": "rules"})
                        r0 = _post_floor(r0, rs[oop], nxt_board, oop, st.name, notes, off_solution)
                        r1 = _post_floor(r1, rs[ip], nxt_board, ip, st.name, notes, off_solution)
                        continue
                    sol = self._solve_cached(sub_board, s0, s1, st.pot_start, eff, self.prior_budget, cfg=cfg,
                                             seed=self._store_seed(walked.get("flop"), sub_board)
                                             if st.name == "turn" else None)
                    expl = self._expl(sol)
                    trust = trust_of(expl, st.name)
                    if trust < 1.0:
                        off_solution.append({"street": st.name, "why": "unsettled", "trust": round(trust * 100)})
                        notes.append(f"{st.name}: the solve that narrows ranges is {expl:.1f}% of the pot from "
                                     f"equilibrium ({sol.iters} iterations), so ranges follow it {trust * 100:.0f}% and "
                                     "simple betting rules the rest")
                    taken = []
                    node, c0, c1 = self._walk(sol, st.actions, oop, ip, st.pot_start, eff, notes, st.name, self.prior_smoothing,
                                              board=sub_board, rules=(0, 1),
                                              lock=mk_lock(sol), trust=trust, slow=slow_hu, taken=taken, off=off_solution,
                                              resolve=self._rare_resolver(sub_board, st, eff, cfg), resolved=resolved)
                    if resolved:
                        sol, cut = resolved[-1]
                        taken = taken[cut:]
                    if s0 is not r0 or s1 is not r1:   # (and out again for the next street: _with_hero puts it back)
                        (c0 if adv_p == 0 else c1)[sol.ctx.active == COMBO_INDEX[(min(adv_hole), max(adv_hole))]] = 0.0
                walked[st.name] = (sol, node, taken)
                r0 = _post_floor(r0, expand(sol.ctx, c0), nxt_board, oop, st.name, notes, off_solution)
                r1 = _post_floor(r1, expand(sol.ctx, c1), nxt_board, ip, st.name, notes, off_solution)
            if self._showdown:
                return self._equity_answer(hero, hole, board, {villain: r1 if hero_p == 0 else r0}, cur, n_streets,
                                           notes, t0)
            eff = min(stk[oop] - cur.inv_start[oop], stk[ip] - cur.inv_start[ip])
            cfg = self._cfg_with_real_sizes(cur, oop, ip, eff)
            if seen is not None:
                r0, r1 = self._with_hero(r0, r1, adv_p, adv_hole, [])
            r0, r1 = self._with_hero(r0, r1, hero_p, hole, notes)
            if optimize and to_call <= 1e-9:
                cfg, opt = self._optimize_sizes(board, r0, r1, cur, eff, cfg, hero_p, notes)
            if lib is not None and cur.name == "flop":
                # a precomputed solve answers when it holds the spot and the line, the villain's line isn't rare in
                # it and the node is one it stores (else the live solve below)
                w2, rl2, n2, tk = [], [], [], []
                hit = self._from_library(lib, full, board, r0, r1, cur, eff, cfg, oop, ip, n2, holes=(hole,),
                                         borrow=not custom, players=len(positions), smoothing=self.prior_smoothing,
                                         only=1 - hero_p,
                                         widened=w2, board=board, rare_lines=rl2, taken=tk)
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
            near = off = None
            if lib is not None and cur.name == "flop":
                from .flop_approx import approx_solution
                near = lambda: approx_solution(lib, full[0], full[1], board, r0, r1, cur.pot_start, eff, cfg,
                                               seats=(oop, ip), hole=hole or None,
                                               aggr=_pf_aggressor(streets["preflop"], oop, ip),
                                               players=len(positions))
                if not custom and hole:
                    off = lambda: self._off_tree_answer(lib, full, board, r0, r1, cur, eff, oop, ip, hero_p, hole,
                                                        hero_remaining, len(positions))
            unr = None
            if lib is not None and cur.name == "flop" and self.cfg.rake_rate and self.rake_library and not custom:
                unr = lambda: self._unraked_entry(lib, streets["preflop"], positions, stack, board, r0, r1, cur, eff,
                                                  cfg, oop, ip, hole)
            return self._instant_answer(cur, board, hole, hero, villain, hero_p, oop, ip, eff, cfg, r0, r1, walked,
                                        hero_remaining, to_call, len(positions), t0, near, off, unr)
        if lib_sol is None and self._lib_only:
            if lib is not None and cur.name == "flop" and hu_from_start:
                # (a library spot's flop not stored: the stored flops its approximate answer reads, named for the
                # browser's prefetch: instant_from_near)
                from .flop_approx import approx_solution
                approx_solution(lib, full[0], full[1], board, r0, r1, cur.pot_start, eff, cfg, fetch_only=True,
                                aggr=_pf_aggressor(streets["preflop"], oop, ip), players=len(positions))
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
                                          lock=mk_lock(sol), rare_lines=rare_lines, slow=dict(slow_hu),
                                          off=off_solution)
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
        n_r = n_r0 if hero_p == 0 else n_r1     # (the range's equilibrium frequency of each action: RARE_BRANCH)
        rare = (node.avg_strategy0() * n_r).sum(axis=1) < RARE_BRANCH * max(float(n_r.sum()), 1e-12)
        for a, name in enumerate(node.names):
            f = float(strat[a, ci]) if ci is not None else 1.0 / len(node.names)
            ev = float(evs[a, ci]) if ci is not None else 0.0
            acts_out.append({"action": name, "freq": round(f, 3), "ev_bb": round(ev, 2),
                             "amount_bb": self._amount(name, node, hero_p)})
            if exploit and ci is not None:
                acts_out[-1]["gto_freq"] = round(float(node.avg_strategy0()[a, ci]), 3)
            if rare[a]:
                acts_out[-1]["rough_ev"] = True
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
        # (what they can hold given your cards: the solve's ranges are whole, card removal in its rows)
        vill_r = expand(sol.ctx, n_r1 if hero_p == 0 else n_r0) * np.prod([1.0 - MEMBER_FULL[c] for c in hole], axis=0)
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
            **({"off_solution": off_solution} if off_solution else {}),
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
        eq, eq_opp = (np.full(N_COMBOS, np.nan),) * 2 if self._looking else range_equities(board, me, opp)
        part, (og,) = _study_range(board, names, [self._amount(n, node, hero_p) for n in names], me, sf, ev, eq,
                                   [(villain, opp, eq_opp)])
        n_r = n_r0 if hero_p == 0 else n_r1   # (the advice's rough_ev: an action the range takes under RARE_BRANCH)
        rare = (node.avg_strategy0() * n_r).sum(axis=1) < RARE_BRANCH * max(float(n_r.sum()), 1e-12)
        for a, r in zip(part["actions"], rare):
            if r:
                a["rough_ev"] = True
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
        (bb relative to folding now), and each opponent's range. Equity: each combo's share of the pot at showdown
        against the other two ranges, the rest of the board dealt (multiway_solver.showdown_equities; the river's from
        the solve's own showdown); 4-way on the river only. An action the range takes under RARE_BRANCH is marked
        "rough_ev", as heads-up. `node`: the solve's root, or a node the 3-way library stores (R: the players' reach
        there)."""
        from .multiway_solver import showdown_equities
        names = list(shown)
        A = len(names)
        rng = [sol.expand(R[i][0]) for i in range(len(players))]
        sf = np.zeros((A, N_COMBOS)); sf[:, sol.hands.active] = strat
        ev = np.zeros((A, N_COMBOS)); ev[:, sol.hands.active] = evs
        eqs = showdown_equities(board, rng) if len(board) < 5 and len(players) == 3 else None

        def eq_of(i):
            if eqs is not None:
                return eqs[i]
            e = np.full(N_COMBOS, np.nan)
            if len(board) == 5:
                e[sol.hands.active] = sol.showdown_shares(i)
            return e
        amounts = [None if x is None else round(float(x), 2) for x in node.amounts]
        opps = [i for i in range(len(players)) if i != hp]
        part, og = _study_range(board, names, amounts, rng[hp], sf, ev, eq_of(hp),
                                [(players[i], rng[i], eq_of(i)) for i in opps])
        n_r = np.asarray(R[hp][0], dtype=np.float64)   # (the advice's rough_ev: an action the range takes under RARE_BRANCH)
        rare = (np.asarray(strat, dtype=np.float64) * n_r).sum(axis=1) < RARE_BRANCH * max(float(n_r.sum()), 1e-12)
        for a, r in zip(part["actions"], rare):
            if r:
                a["rough_ev"] = True
        cfg = getattr(sol, "cfg", None)
        if cfg is None:   # (the 3-way library's: its entries are read only where solved on this game)
            from .mw_library import library_config
            cfg = library_config()
        later, K = cfg.later, cfg.buckets
        if len(board) < 5 and later == "buckets" and K:
            notes = list(notes) + [
                f"per hand: each hand's mix and EVs on the {cur.name} are its own, and its equity; on the streets after "
                f"it the solve plays hands by bucket ({K} a player on each card, by equity against the other ranges), "
                f"so the EVs count a hand's later play as its bucket's"]
        hero = players[hp]
        res = {
            "study": "node", "street": cur.name, "actor": hero, "multiway": True,
            "opponent": " and ".join(players[i] for i in opps), "opponents": og,
            "position": f"{hp + 1} of {len(players)} to act", "players": len(positions), "board": cards_str(board),
            "pot_bb": round(node.pot_now, 2), "to_call_bb": round(to_call, 2), "stack_bb": round(behind, 2),
            **part, "solver": solver, "precomputed": precomputed, "notes": notes, "seconds": round(time.time() - t0, 2),
        }
        if self._breakdown:   # (each player's equity against the two others where Study has it: 3-way)
            from . import breakdown as BD
            res["breakdown"] = BD.breakdown(board, names, hero, rng[hp], sf, (sf * ev).sum(axis=0),
                                            {players[i]: rng[i] for i in opps}, node.pot_now, equity=False,
                                            equities={players[i]: eq_of(i) for i in range(len(players))}
                                            if len(players) == 3 else None)
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
        off = hi if hi is not None and rs[hp][hi] <= 1e-9 else None
        if off is not None:
            # (near-zero, not out: a hand with no weight has no average strategy at any later decision, so its EVs read
            # uniform play there: calling and folding to a bet alike. It changes nobody's play)
            rs[hp] = rs[hp].copy(); rs[hp][off] = MW_OFF_RANGE
            notes.append("your hand isn't in your range for this line, so it was solved as a near-zero-weight combo")
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
               hp, cur.n_raises, tuple(sorted(pending)), repr(mcfg), inc, off)
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
        # (what they can hold given your cards: the solve's ranges are the whole ranges, card removal in its rows)
        unblocked = np.prod([1.0 - MEMBER_FULL[c] for c in hole], axis=0)
        if self._breakdown and not self._study:   # hero's categories and the opponents' (no equity view 3-way)
            from . import breakdown as BD
            ev_full = np.zeros(N_COMBOS); ev_full[sol.hands.active] = (strat * evs).sum(axis=0)
            bd = BD.breakdown(board, shown, hero, hero_r, sf, ev_full,
                              {p: sol.expand(sol.reach0[i][0]) * unblocked for i, p in enumerate(players) if p != hero},
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
               "opponent_range_combos": {p: round(float((sol.expand(sol.reach0[i][0]) * unblocked).sum()), 1)
                                         for i, p in enumerate(players) if p != hero},
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
        move and the hero's advice. Otherwise (preflop, multiway, a pot that became heads-up this
        street, a line of the hero's the solve rarely takes, a hand it gives no weight) it's the advice for
        `seat` holding `hole`, heads-up with the hero's advice's ranges (its earlier streets' solves are the
        cached ones). `seed` makes the draw repeatable.
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
            # set on their opponents (which can name this seat: "you can't lock your own seat" stalled practice).
            # Heads-up its ranges are the ones the practising player's advice solved with (_seen_from: their cards
            # out of this seat's range, their hand in where off its range), so the earlier streets' solves are cached
            # ones: from its own view's ranges it solved them again (3.5 s of a limped river's 3.6).
            view = st.get("hero"), st.get("hole")
            try:
                self._seen_from = (str(view[0]).upper(), tuple(parse_cards(view[1])) if isinstance(view[1], str)
                                   else tuple(view[1])) if view[0] and view[1] and str(view[0]).upper() != seat else None
            except Exception:   # noqa: BLE001 - (an unreadable hand: the seat's own view, as before)
                self._seen_from = None
            try:
                res = self.advise({**{k: v for k, v in st.items() if k not in ("locks", "study")}, "hero": seat,
                                   "hole": cards_str(hole)})
            finally:
                self._seen_from = None
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
        players' ranges entering the flop (`entering`), the pot and the stack behind, no rake and no custom sizes; at
        another table size, the spot it reads the pot as (MWLibrary.borrow; not with ranges of your own)."""
        if self.mw_library is None or len(flop) != 3 or len(st.players_start) != 3 or self.cfg.rake_rate > 0 \
                or self._req_sizes or getattr(self, "_icm", None) is not None:   # (solved in chips)
            return None
        players = [p for p in st.order if p in st.players_start]
        if any(p not in entering for p in players):
            return None
        S = min(stk[p] - st.inv_start[p] for p in players)
        try:
            return self.mw_library.solution([entering[p] for p in players], tuple(flop[:3]), st.pot_start, S,
                                            players=self._mw_borrow)
        except Exception:   # a library that can't be read is no library
            return None

    def _mw_library_near(self, entering, flop, st: StreetState, stk):
        """A flop of a 3-way library spot the library doesn't store, read off its stored flops most like it
        (mw_library.MWLibrary.near), as _mw_library_solution finds the spot; None where it doesn't apply."""
        if self.mw_library is None or not hasattr(self.mw_library, "near") or len(flop) != 3 \
                or len(st.players_start) != 3 or self.cfg.rake_rate > 0 or self._req_sizes \
                or getattr(self, "_icm", None) is not None:
            return None
        players = [p for p in st.order if p in st.players_start]
        if any(p not in entering for p in players):
            return None
        S = min(stk[p] - st.inv_start[p] for p in players)
        try:
            return self.mw_library.near([entering[p] for p in players], tuple(flop[:3]), st.pot_start, S,
                                        players=self._mw_borrow)
        except Exception:   # a library that can't be read is no library
            return None

    def _study_mw_near(self, sol, node, path, players, hp, behind, S, board, positions, notes, t0, hole=None) -> dict:
        """Study's answer at once on a 3-way flop the 3-way library doesn't store (Advisor.instant with "study"), while
        the flop's own solve runs: the library's stored flops most like it read for this one (MWLibrary.near: card
        substitution, each combo's mix its images' of the same hand class averaged), at a decision the library stores
        (first to act, checked to once or twice): the actor's whole range per hand class and combo its mix (combos no
        stored flop holds an image of left out), per action the range's frequency and combos, each opponent's range
        (through the read mixes before), marked "approximate" with the flops read. No EVs or equities: the flop's own
        solve, which replaces it, has them. With `hole` (the advice's, Advisor.instant without "study"): that combo's
        mix alone (_mw_instant)."""
        R = sol.reach_at(path)
        rng = [sol.expand(R[i][0]) for i in range(len(players))]
        me = rng[hp] * sol.expand(np.asarray(sol.approx["known"].get(path, np.ones(sol.hands.n, bool)), dtype=np.float64))
        if me.sum() < 1e-6:
            raise _NoInstant()
        shown = _mw_names(node.names, node.amounts, behind, S, node.pot_now)
        sf = np.zeros((len(shown), N_COMBOS))
        sf[:, sol.hands.active] = node.avg_strategy()[0]
        flops = ", ".join(nm for nm, _ in sol.approx["near"])
        bw = getattr(sol, "borrowed", None)
        lent = "" if not bw else f" in the library's {sol.label}, that spot read " + \
            ("for this pot" if "pot" in bw else "as this one") + f" ({sol.borrowed_text()})"
        if sol.approx.get("stored"):    # (another pot's stand-in on a flop the spot stores: MWLibrary.stand_in)
            label = f"the precomputed 3-way solve of this flop{lent}"
            note = (f"read off the 3-way library's {sol.label} on this flop, that spot read for this pot: no EVs or "
                    f"equities until the flop's own solve")
        else:
            label = f"the precomputed 3-way solves of the most similar flops ({flops}), by card substitution{lent}"
            note = (f"read off the 3-way library's {sol.label} on stored flops by card substitution: no EVs or "
                    f"equities until the flop's own solve")
        return self._mw_instant(
            "flop", players, hp, [i for i in range(len(players)) if i != hp], rng, me, sf, shown, node.amounts,
            node.pot_now, 0.0, S, board, positions,
            {"source": "near", "label": label, "iterations": None},
            {"iterations": None, "exploitability_pct": None, "by": "the 3-way flop library"},
            list(notes) + [note], t0, hole)

    @staticmethod
    def _mw_instant(street, players, hp, opps, rng, me, sf, shown, amounts, pot_now, to_call, stack, board, positions,
                    approximate, solver, notes, t0, hole=None) -> dict:
        """A multiway Study answer at once (_study_mw_near, _study_mw_prev): the actor's range `me` and mix `sf` (A x
        N_COMBOS) as _study_mw_node's answer without EVs or equities, each opponent's range (rng, players' order),
        marked approximate. With `hole` (the advice's early panel, Advisor.instant without "study"), the answer for
        that combo alone, shaped as a heads-up instant answer (_instant_answer): its mix, the recommended action, no
        EVs; _NoInstant where the read doesn't hold it (a combo off the range, or no image of it on the flops read)."""
        none = np.full(N_COMBOS, np.nan)
        amounts = [None if x is None else round(float(x), 2) for x in amounts]
        if hole is not None:
            ci = COMBO_INDEX[tuple(sorted(hole))]
            mix = sf[:, ci]
            if me[ci] <= 1e-9 or mix.sum() < 1e-6:
                raise _NoInstant()
            mix = mix / mix.sum()
            acts = [{"action": nm, "freq": round(float(f), 3), "amount_bb": amt}
                    for nm, f, amt in zip(shown, mix, amounts)]
            return {"street": street, "hero": players[hp], "villain": " and ".join(players[i] for i in opps),
                    "multiway": True, "position": f"{hp + 1} of {len(players)} to act", "players": len(positions),
                    "hand": hand_class(hole), "board": cards_str(board), "pot_bb": round(pot_now, 2),
                    "to_call_bb": round(to_call, 2), "stack_bb": round(stack, 2), "actions": acts,
                    "recommended": shown[int(np.argmax(mix))], "approximate": approximate,
                    "notes": [f"approximate, at once: {approximate['label']}; this {street}'s own solve replaces it"]
                    + list(notes), "seconds": round(time.time() - t0, 3)}
        part, og = _study_range(board, list(shown), amounts, me, sf, np.zeros_like(sf), none,
                                [(players[i], rng[i], none) for i in opps])
        _no_values(part)
        return {"study": "node", "street": street, "actor": players[hp], "multiway": True,
                "opponent": " and ".join(players[i] for i in opps), "opponents": og,
                "position": f"{hp + 1} of {len(players)} to act", "players": len(positions), "board": cards_str(board),
                "pot_bb": round(pot_now, 2), "to_call_bb": round(to_call, 2), "stack_bb": round(stack, 2),
                **part, "range_ev_bb": None, "range_eq": None, "approximate": approximate, "solver": solver,
                "precomputed": None, "notes": notes, "seconds": round(time.time() - t0, 3)}

    def _mw_hits(self, board, st: StreetState, stk, hole, streets) -> List[tuple]:
        """The hand's multiway solves of street st that Study or the advice answered from (_mw_lines), latest first:
        [(k, (solve, players, exploitability))], k the solve's root's place in the street's actions, under any of the
        street's players (Study stores a node's solve under its actor)."""
        acts = streets.get(st.name) or []
        bk = board[:{"flop": 3, "turn": 4, "river": 5}[st.name]]
        out = []
        for k in range(len(acts), -1, -1):
            h = next((h for h in (self._mw_lines.get(_mw_line(bk, x, hole, stk, streets, st.name, k))
                                  for x in st.order if x in st.players_start) if h is not None), None)
            if h is not None:
                out.append((k, h))
        return out

    def _study_mw_prev(self, hero, hole, board, rep, stk, streets, ranges, positions, notes, t0) -> Optional[dict]:
        """Study's answer at once at a 3-way or 4-way node before the river while its own solve runs (Advisor.instant
        with "study"): each hand's mix read off the hand's latest live solve of an earlier decision on this street (its
        tree from there on the line taken, the tree's nearest actions: each combo's own mix at its sizes), else at a
        3-way turn's first decision a 3-way flop's, which plays the turn by card class and hand buckets at one bet size
        (MWConfig.later: the card dealt read as its class's played card, each combo through the swap of the two, as
        heads-up Advisor._study_prev; the turn's later decisions wait on a turn solve: read off the flop's buckets
        they were far from their own answers, after a check 1.1% of the pot given up and the range's frequencies
        19 points apart, after two checks 1.2% and 17 points; a 4-way flop's view of the turn was 0.6% of the
        pot from the turn's own answer,
        its range frequencies 23 points apart: scripts/bench_mw_prev.py --lines hj4; facing a bet 4-way, where
        the 4-way tree lets only its first decision's player raise, _NoInstant). The players' ranges are `ranges`, as the node's own solve will start from them (the
        streets' actions narrowed through the solves Study answered from, Advisor._mw_walk). The actor's whole range
        per hand class and combo its mix, per action the range's frequency and combos, each opponent's range, marked
        "approximate"; no EVs or equities. None where no such solve is in hand (the 3-way library's flops hold the
        flop's checked-to decisions only: none after a bet or on the turn); _NoInstant where the node's own solve is
        in hand (its answer is quick) or the solve rarely takes the line (an action keeping a range under RARE_LINE of
        itself in the solve: its play there barely converged)."""
        from . import multiway_solver as MW
        cur = rep[-1]
        remaining = [p for p in cur.order if p in cur.active]
        acts = list(streets.get(cur.name) or [])
        hits = self._mw_hits(board, cur, stk, hole, streets)
        if len(remaining) < 3:
            return None
        if hits and hits[0][0] == len(acts):
            raise _NoInstant()   # (the node's own solve: its answer is quick)
        live = lambda h: isinstance(h[1][0], MW.MultiwaySolver) and set(remaining) <= set(h[1][1]) \
            and hero in h[1][1]
        own = next((h for h in hits if live(h)), None)
        behind = stk[hero] - cur.inv_start[hero]
        S = min(stk[p] - cur.inv_start[p] for p in cur.players_start)
        to_call = cur.cur_bet - cur.bets[hero]
        stack = min(behind, max(stk[p] - cur.inv_start[p] for p in remaining if p != hero)) - cur.bets[hero]
        pot_now = cur.pot_start + sum(cur.bets.values())
        if own is not None:   # (this street's solve from an earlier decision: its tree per combo)
            k, (sol, pl, expl) = own
            R = np.asarray(sol.reach0[:, 0, :], dtype=np.float64).copy()
            node = sol.root
            for a in acts[k:]:
                node = _mw_follow(node, R, a, pl)
            hp = pl.index(hero)
            if not isinstance(node, MW.Decision) or isinstance(node, MW.BDecision) or node.player != hp:
                raise _NoInstant()
            sf = np.zeros((len(node.names), N_COMBOS))
            sf[:, sol.hands.active] = node.avg_strategy()[0]
            cover = sol.hands.active
            names, amounts, pot_at = node.names, node.amounts, node.pot_now
            whose = pl[sol.root.player] if isinstance(sol.root, MW.Decision) else "an earlier"
            approximate = {"source": "solve", "label": f"this street's {len(pl)}-way solve from {whose}'s decision "
                                                       f"({sol.iters} iterations, {expl:.2f}% of the pot from "
                                                       f"equilibrium), on the line since", "iterations": None}
            solver = {"iterations": None, "exploitability_pct": None, "by": f"the {cur.name}'s earlier solve"}
            note = (f"read off this street's solve from {whose}'s decision: each hand's mix there, no EVs or equities "
                    f"until this decision's own solve")
            if len(pl) > 3 and to_call > 1e-9 and not any(n.startswith("raise") for n in names):
                # (the 4-way tree lets only its first decision's player raise, where this decision's own solve raises:
                # 4c4sQs, HJ opens, CO, BTN and BB call, BTN's bet after three checks: BB raises 19% there, the read
                # 0, giving up 1.1% of the pot)
                raise _NoInstant()
        else:
            if cur.name != "turn" or len(rep[-2].players_start) != 3 or len(cur.players_start) != 3:
                return None   # (a 4-way flop's view of the turn: too far from the turn's own answer)
            fl = rep[-2]
            fh = next((h for h in self._mw_hits(board, fl, stk, hole, streets)
                       if live(h) and getattr(h[1][0], "tabs", None) is not None and h[1][0].levels), None)
            if fh is None:
                return None
            k, (sol, pl, expl) = fh
            R = np.asarray(sol.reach0[:, 0, :], dtype=np.float64).copy()
            node = sol.root
            for a in list(streets.get("flop") or [])[k:]:
                node = _mw_follow(node, R, a, pl)
            tb, card = sol.tabs, board[3]
            loc = next(((g, b) for g in sol.levels[0] for b, (e, _) in enumerate(g.blocks) if e is node), None)
            t = next((i for i, (_, cs) in enumerate(tb.cls) if card in cs), None)
            if not isinstance(node, MW.Exit) or loc is None or t is None:
                raise _NoInstant()
            (g, b), played = loc, tb.cls[t][0]
            row = b * tb.C + t
            act = sol.hands.active
            pos = np.full(N_COMBOS, -1)
            pos[act] = np.arange(len(act))
            img = np.full(len(act), -1)   # (each combo live on the turn: its image's column on the class's played card)
            for i, f in enumerate(act):
                if card not in ALL_COMBOS[f]:
                    img[i] = pos[_swap_combo(ALL_COMBOS[f], played, card)] if played != card else i
            m = img >= 0
            cols = np.full((len(pl), len(act)), -1)
            cols[:, m] = tb.ids[:, t][:, img[m]]
            R[:, ~m] = 0.0
            td = MW._template(sol.cfg, sol.np, "turn", node.pot, node.c, node.behind, node.folded)
            c0, bn = node.c, g.tree
            if cur.actions:
                raise _NoInstant()   # (the turn's first decision only: its later ones were far from their own)
            hp = pl.index(hero)
            if not isinstance(bn, MW.BDecision) or not isinstance(td, MW._TD) or bn.player != hp:
                raise _NoInstant()
            sf = np.zeros((len(td.names), N_COMBOS))
            sf[:, act[m]] = bn.avg_strategy()[row][:, cols[hp, m]]
            cover = act[m]
            names, amounts, pot_at = td.names, _td_amounts(td, c0), td.pot
            K = bn.regret.shape[2]
            approximate = {"source": "prev", "label": f"the flop solve's view of the turn (by card class and {K} hand "
                                                      f"buckets a player, one bet size)", "iterations": int(sol.iters)}
            solver = {"iterations": int(sol.iters), "exploitability_pct": round(float(expl), 2), "by": "the flop's solve"}
            note = (f"read off the flop's solve, which plays the turn at one bet size, a turn class's cards alike (a "
                    f"rank's cards, a flush suit's apart) and hands by {K} buckets a player: no EVs or equities until "
                    f"the turn's own solve")
        bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        rng = [np.asarray(ranges[p], dtype=np.float64) * bm for p in pl]
        me = np.zeros(N_COMBOS)
        me[cover] = rng[hp][cover]   # (the combos the solve plays: its mix for each)
        if me.sum() < 1e-6:
            raise _NoInstant()
        shown = _mw_names(names, amounts, behind, S, pot_at)
        return self._mw_instant(cur.name, pl, hp, [i for i, p in enumerate(pl) if i != hp and p in remaining], rng, me,
                                sf, shown, amounts, pot_now, to_call, stack, board, positions, approximate, solver,
                                list(notes) + [note], t0, None if self._study else hole)

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
        if self._instant:   # (the answer at once, Study's or the advice's: a flop it holds is at once anyway; one it
            # doesn't, read off)
            if sol is not None:
                raise _NoInstant()
            from .multiway_solver import MultiwaySolver
            if any(isinstance(h[0], MultiwaySolver) for _, h in self._mw_hits(board, cur, stk, hole, streets)):
                return None   # (this flop's own solve of an earlier decision reads it better: Advisor._study_mw_prev)
            sol = self._mw_library_near(entering, board, cur, stk)
        if sol is None:
            return None
        path = tuple("check" for _ in cur.actions)
        node = sol.nodes.get(path)
        hp = players.index(hero)
        ci = sol.combo_index(hole) if hole else None    # (Study: no cards of the actor's own)
        if node is None or node.player != hp or ci is None and not self._study:
            return None   # (a node it doesn't store: solved live)
        if sol.approx is not None:
            return self._study_mw_near(sol, node, path, players, hp, stk[hero] - cur.inv_start[hero],
                                       min(stk[p] - cur.inv_start[p] for p in players), board, positions or [],
                                       notes, t0, None if self._study else hole)
        # (a hand off its range: the stored EVs play it uniformly at its later decisions, so they aren't read. A live
        # solve answers it, _multiway_solve; a quick answer, which has no time for one, gives its mix alone)
        off = ci is not None and sol.reach0[hp, 0, ci] <= 1e-12
        if off:
            if not getattr(self, "_quick", False):
                return None
            notes.append("your hand isn't in your range for this line: its mix is the solve's current (regret-matching) "
                         "strategy for it, against the ranges as they are, with no EVs")
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
                    f"all-in; the turn and river bet on too, by 16 hand buckets a player"
                    + (f"; approximate: that spot read as this one ({sol.borrowed_text()})"
                       if getattr(sol, "borrowed", None) else ""))
        solver = {"iterations": sol.iters, "solve_seconds": 0.0, "nodes": len(sol.nodes), "active_combos": sol.hands.n,
                  "tree_actions": shown, "exploitability_pct": round(expl, 2),
                  "best_response_gains_pct": {p: round(float(g), 2) for p, g in zip(players, gains)}}
        if self._study:
            return self._study_mw_node(sol, node, R, players, hp, shown, strat, evs, 0.0, S, board, cur, positions or [],
                                       solver, notes + [lib_note, "best-response gains (% pot): " + ", ".join(
                                           f"{p} {g:.2f}" for p, g in zip(players, gains))], t0, precomputed=sol.info())
        mine = strat[:, ci]
        acts = [{"action": name, "freq": round(float(mine[a]), 3), **({} if off else {"ev_bb": round(float(evs[a, ci]), 2)}),
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
                        n_players, t0, near=None, off=None, unraked=None) -> dict:
        """Advisor.instant's answer at the street advised on (heads-up from its start, its ranges narrowed through the
        earlier streets' solves as they stand): the first of instant_sources that reads one, else _NoInstant; and
        _NoInstant when the street's own solve is cached and its answer quick (settled, or run its budget)."""
        own = self._cache.get(self._solve_key(board, r0, r1, cur.pot_start, eff, cfg=cfg))
        if own is not None and not getattr(own, "stopped", False) and (
                getattr(own, "settled_at", None) is not None or own.iters >= 1000 or own.seconds >= 0.8 * self.time_budget):
            raise _NoInstant()
        if self._study:   # (Study: the actor's whole range, off a library flop's stored turn or its near flops, or
            # the previous street's live solve)
            if cur.name == "flop":
                return self._study_near(cur, board, hero, villain, hero_p, oop, ip, eff, to_call, behind, n_players,
                                        t0, near)
            try:
                return self._study_store(cur, board, hero, villain, hero_p, oop, ip, eff, r0, r1, walked, behind,
                                         to_call, n_players, t0)
            except _NoInstant:
                return self._study_prev(cur, board, hero, villain, hero_p, oop, ip, eff, r0, r1, walked, behind,
                                        to_call, n_players, t0)
        earlier = False
        if own is None:   # (facing a size off the tree your advice on this street solved: that solve, at its nearest)
            c = self._hand_cfg(board, r0, r1, cur, oop, ip, eff, cfg)
            k = self._solve_key(board, r0, r1, cur.pot_start, eff, cfg=c)
            if c is not cfg and k not in self._presolved:   # (a presolve's, the move off its tree: never shown)
                own, earlier = self._cache.get(k), True
        spot = InstantSpot(cur.name, tuple(board), tuple(hole), hero_p, oop, ip, list(cur.actions), cur.pot_start, eff,
                           behind, own, walked.get(STREETS[STREETS.index(cur.name) - 1]), earlier, near, off, unraked)
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
                "notes": notes, "seconds": round(time.time() - t0, 3),
                **({"range_ev_bb": got["range_ev"][0], "villain_range_ev_bb": got["range_ev"][1]} if "range_ev" in got
                   else {})}

    def _study_store(self, cur, board, hero, villain, hero_p, oop, ip, eff, r0, r1, walked, behind, to_call,
                     n_players, t0) -> dict:
        """Study's answer at once on a library flop's turn (Advisor.instant with "study"), while the turn's own solve
        runs: the precomputed flop solve's turn strategies stored beside its entry (instant_from_store's turn file),
        for the actor's whole range at the turn's first decision and in position after a check: per hand class (the
        grid) and combo the mix, per action the range's frequency and combos, the opponent's range (after a check:
        the hands the stored first decision checks), marked "approximate". A format-2 file adds its EVs (_store_evs:
        each action's, the hands that take it, and the range's), as the early panel shows them; no hand's own EVs or
        equities (they come with the turn's own solve, which replaces it). Combos the file doesn't store (under
        TURN_REACH of the most reached) are left out of the range. _NoInstant where it doesn't hold the node."""
        prev, lib = walked.get("flop"), self.library
        if cur.name != "turn" or prev is None or lib is None or not hasattr(lib, "turns"):
            raise _NoInstant()
        sol, _, taken = prev
        acts = list(cur.actions)
        if not acts:
            at = ()
        elif len(acts) == 1 and acts[0]["action"] == "check" and acts[0]["pos"] == oop:
            at = ("check",)
        else:
            raise _NoInstant()
        got = _turn_file(lib, sol, board[:3])
        # (a spot with no turn files of its own: the nearest that stores this flop's, _near_turn_spots, where it holds
        # NEAR_TURN_HELD of the actor's range)
        files = [got + (None,)] if got is not None else _near_turn_files(lib, sol, board[:3])
        from . import flop_library as fl
        live = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        for t, smap, src in files:
            st = fl.turn_strategies(t, taken, at, board[3], smap)
            nd = fl.turn_node(t, taken, at, board[3] // 4 * 4 + smap[board[3] % 4])
            if st is None or nd is None or st["player"] != hero_p:
                continue
            r = [np.asarray(r0, dtype=np.float64) * live, np.asarray(r1, dtype=np.float64) * live]
            if at:   # (in position after a check: the hands the stored first decision checks; unstored ones at an
                # even mix)
                first = fl.turn_strategies(t, taken, (), board[3], smap)
                if first is None or "check" not in first["names"]:
                    continue
                x = first["strat"][first["names"].index("check")]
                r[0] = r[0] * np.where(np.isnan(x), 1.0 / len(first["names"]), x)
            held = ~np.isnan(st["strat"][0])
            me, opp = r[hero_p] * held, r[1 - hero_p]
            if me.sum() >= 1e-6 and (src is None or me.sum() >= NEAR_TURN_HELD * r[hero_p].sum()):
                break
        else:
            raise _NoInstant()
        sf = np.nan_to_num(st["strat"])
        spot = InstantSpot("turn", tuple(board), (), hero_p, oop, ip, acts, cur.pot_start, eff, behind)
        real = _real_actions(types.SimpleNamespace(names=st["names"], pot_now=0.0, bets=(0.0, 0.0)),
                             np.zeros(len(st["names"])), [0.0, 0.0], spot)
        part, og = _study_mix(board, [a["action"] for a in real], [a["amount_bb"] for a in real], me, sf, villain, opp)
        evs, rng = (None, None) if src is not None else \
            _store_evs(t, taken, at, board[3] // 4 * 4 + smap[board[3] % 4], nd, cur.pot_start, hero_p)
        for a, act in enumerate(part["actions"]):
            act["ev_bb"] = evs[a] if evs is not None else None
        if rng is not None:
            range_ev = rng[0]
        elif evs is not None:   # (after a check: the actions' EVs by how often the range takes each)
            w = [(x["freq"], e) for x, e in zip(part["actions"], evs) if e is not None and x["freq"] > 0]
            range_ev = round(sum(f * e for f, e in w) / sum(f for f, _ in w), 2) if w else None
        else:
            range_ev = None
        label = _store_label(st["names"], src)   # (the panel says what it is and that the turn's own solve replaces it)
        notes = ["the stored turn's EVs are the flop solve's for the turn card's class (the range's and each action's), "
                 "no hand's own" if evs is not None else "the stored turn holds strategies only, so no EVs until the "
                 "turn's own solve" if src is None else "another spot's stored turn: its strategies only, so no EVs "
                 "until the turn's own solve"]
        return {"study": "node", "street": "turn", "actor": hero, "opponent": villain,
                "position": "OOP" if hero_p == 0 else "IP", "players": n_players, "board": cards_str(board),
                "pot_bb": round(cur.pot_start, 2), "to_call_bb": round(min(to_call, behind), 2),
                "stack_bb": round(min(eff, behind), 2),
                **part, "range_ev_bb": range_ev, "range_eq": None,
                "opponent_combos": og["combos"], "opponent_eq": None, "opponent_grid": og["grid"],
                "approximate": {"source": "store", "label": label, "iterations": int(sol.iters)},
                "solver": {"iterations": int(sol.iters), "exploitability_pct": round(float(sol.exploitability()), 3),
                           "by": "the flop library"},
                "notes": notes, "seconds": round(time.time() - t0, 3)}

    def _study_near(self, cur, board, hero, villain, hero_p, oop, ip, eff, to_call, behind, n_players, t0,
                    near) -> dict:
        """Study's answer at once on a flop the library doesn't store (Advisor.instant with "study"), while the flop's
        own solve runs: instant_from_near's approximation (gto/flop_approx.py) for every combo, the flop's decisions
        read off the stored flops most like it (a library spot's other flops; a pot the library holds no spot for, or
        another stack depth, off the spot nearest it), walked down the line as the library's answers are (its sizes at
        the tree's nearest, the ranges narrowed by its mixes); at a decision an entry would store, the actor's whole
        range: per hand class (the grid) and combo the mix (combos no stored flop holds an image of left out, and the
        hands the spot's ranges don't hold where it reads them alone: another game's or a bigger table's pot, another
        depth; a note says how much of the range that leaves out), per action the range's frequency and combos, the
        opponent's range, marked "approximate" with what it's read off (the advice's early panel's label: the spot
        borrowed, at what depth). No EVs or equities: the flop's own solve, which replaces it, has them. _NoInstant
        where it doesn't hold the node."""
        sol = near() if near is not None and cur.name == "flop" else None
        if sol is None:
            raise _NoInstant()
        visited, n2, w2, rl2 = [], [], [], []
        node, c0, c1 = self._walk(sol, cur.actions, oop, ip, cur.pot_start, eff, n2, "flop", visited=visited,
                                  smoothing=self.prior_smoothing, only=1 - hero_p, widened=w2, board=board,
                                  rare_lines=rl2)
        if not sol.covers(visited) or w2 or rl2 or not isinstance(node, Decision) or node.player != hero_p or \
                id(node) not in sol.stored:
            raise _NoInstant()
        act = sol.ctx.active
        me, opp = np.zeros(N_COMBOS), np.zeros(N_COMBOS)
        whole = np.asarray(c0 if hero_p == 0 else c1, dtype=np.float64)
        me[act] = whole * sol.known[id(node)]
        opp[act] = np.asarray(c1 if hero_p == 0 else c0, dtype=np.float64)
        if me.sum() < 1e-6:
            raise _NoInstant()
        out = 1.0 - me.sum() / max(float(whole.sum()), 1e-12)   # (the range's share no read holds)
        names = list(node.names)
        sf = np.zeros((len(names), N_COMBOS))
        sf[:, act] = node.avg_strategy0()
        part, og = _study_mix(board, names, [self._amount(n, node, hero_p) for n in names], me, sf, villain, opp)
        return {"study": "node", "street": "flop", "actor": hero, "opponent": villain,
                "position": "OOP" if hero_p == 0 else "IP", "players": n_players, "board": cards_str(board),
                "pot_bb": round(node.pot_now, 2), "to_call_bb": round(min(to_call, behind), 2),
                "stack_bb": round(min(eff - max(node.bets), behind), 2),
                **part, "range_ev_bb": None, "range_eq": None,
                "opponent_combos": og["combos"], "opponent_eq": None, "opponent_grid": og["grid"],
                "approximate": {"source": "near", "label": _near_label(sol), "iterations": None},
                "solver": {"iterations": None, "exploitability_pct": None, "by": "the flop library"},
                "notes": ["read off stored flops by card substitution: no EVs or equities until the flop's own solve"]
                + (["the hands that spot's range doesn't hold: the average mix there of their hand class (made hand "
                    "and draw)"] if getattr(sol, "filled", {}).get(id(node), np.zeros(1, bool)).any() else [])
                + ([f"{out:.0%} of this range (by weight) isn't shown: hands the stored flops' ranges don't hold, "
                    f"which the flop's own solve answers"] if out >= 0.005 else []),
                "seconds": round(time.time() - t0, 3)}

    def _study_prev(self, cur, board, hero, villain, hero_p, oop, ip, eff, r0, r1, walked, behind, to_call,
                    n_players, t0) -> dict:
        """Study's answer at once on a turn after a flop solved live, or a river after a turn solved live
        (Advisor.instant with "study"), while the street's own solve runs: the street before's solve, which plays this
        street under it (instant_from_prev's read: a flop solve's turn by card class and hand buckets at one bet size,
        a turn solve's river by card class at its river sizes; the card dealt read as its class's played card, each
        combo through the swap of the two), for the actor's whole range on the line the hand took (the tree's nearest
        actions, each player's range narrowed by that solve's mix): per hand class (the grid) and combo the mix (the
        mix the advice's early panel gives that hand), per action the range's frequency and combos, the opponent's
        range, marked "approximate" with the solve's stats. No EVs or equities: the street's own solve, which replaces
        it, has them. _NoInstant where the solve doesn't play the line (an action off its tree, a line it takes under
        RARE_LINE of the range, on its own street or this one) or isn't a live one (the library's flops hold the flop
        only)."""
        i = STREETS.index(cur.name)
        prev = walked.get(STREETS[i - 1]) if i >= 2 else None
        if prev is None:
            raise _NoInstant()
        sol, ch, taken = prev
        if not isinstance(sol, Solution) or hasattr(sol, "stored") or not isinstance(ch, Chance) or len(ch.boards) != 1:
            raise _NoInstant()
        node, r = sol.root, [sol.r0.copy(), sol.r1.copy()]
        for name in taken:   # (a line it rarely took there: the walk gave that range to the betting rules, _walk)
            if not isinstance(node, Decision) or name not in node.names:
                raise _NoInstant()
            j, p = node.names.index(name), node.player
            before = r[p].sum()
            r[p] = r[p] * node.avg_strategy0()[j]
            if r[p].sum() < RARE_LINE * max(before, 1e-12):
                raise _NoInstant()
            node = node.children[j]
        dec, card = ch.child, board[-1]
        rows = dec.boards
        if len(rows) != ch.K or isinstance(dec, RiverEntry):
            raise _NoInstant()
        if ch.umasks is not None or ch.perm is not None:
            classes = sol.ctx.turn_classes(tuple(ch.boards[0]), sol.solver.dead)
            if [c for c, _ in classes] != [b[-1] for b in rows]:
                raise _NoInstant()
            k = next((j for j, (_, cs) in enumerate(classes) if card in cs), None)
        else:
            k = next((j for j, b in enumerate(rows) if b[-1] == card), None)
        if k is None:
            raise _NoInstant()
        played = rows[k][-1]
        live = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0) > 0
        pos = np.full(N_COMBOS, -1)
        pos[sol.ctx.active] = np.arange(len(sol.ctx.active))
        col = np.full(N_COMBOS, -1)   # (each live combo's column in the solve: its image under the swap)
        for c in np.nonzero(live)[0]:
            col[c] = pos[_swap_combo(ALL_COMBOS[c], played, card) if played != card else c]
        cols = [col, col]
        if isinstance(dec, BucketEntry):   # (a flop solve's turn: each player's hand buckets)
            T, ids = sol.solver._btab(dec)[:2]
            dec = dec.child
            m = col >= 0
            cols = [col.copy(), col.copy()]
            for p in (0, 1):
                cols[p][m] = ids[k % T, p][col[m]]
        if not isinstance(dec, Decision):
            raise _NoInstant()
        n_cols = dec.avg_strategy().shape[2]   # (a bucket past it: dead on the class's played card)
        r = [np.where((cols[p] >= 0) & (cols[p] < n_cols), np.asarray(rp, dtype=np.float64), 0.0)
             for p, rp in enumerate((r0, r1))]
        node, bets = dec, [0.0, 0.0]
        for a in cur.actions:   # (_read_line's walk, each player's range narrowed by their mix)
            if not isinstance(node, Decision):
                raise _NoInstant()
            p = 0 if a["pos"] == oop else 1
            if p != node.player:
                raise _NoInstant()
            j = match_action(node.names, Advisor._tree_action_name(node, a, p, bets, cur.pot_start, eff))
            before = r[p].sum()
            m = r[p] > 0
            r[p][m] *= node.avg_strategy()[k][j][cols[p][m]]
            if r[p].sum() < RARE_LINE * before:
                raise _NoInstant()   # (a line the solve rarely takes: its play there barely converged)
            act = a["action"]
            amt = a.get("to") if a.get("to") is not None else a.get("amount")
            if act in ("bet", "raise") and amt is not None:
                bets[p] = float(amt)
            elif act == "all-in":
                bets[p] = eff
            elif act == "call":
                bets[p] = min(bets[1 - p], eff)
            node = node.children[j]
        if not isinstance(node, Decision) or node.player != hero_p:
            raise _NoInstant()
        me, opp = r[hero_p], r[1 - hero_p]
        if me.sum() < 1e-6:
            raise _NoInstant()
        m = me > 0
        sf = np.zeros((len(node.names), N_COMBOS))
        sf[:, m] = node.avg_strategy()[k][:, cols[hero_p][m]]
        spot = InstantSpot(cur.name, tuple(board), (), hero_p, oop, ip, list(cur.actions), cur.pot_start, eff, behind)
        real = _real_actions(node, np.zeros(len(node.names)), bets, spot)
        part, og = _study_mix(board, [x["action"] for x in real], [x["amount_bb"] for x in real], me, sf, villain, opp)
        last = STREETS[i - 1]
        how = "by card class and hand buckets, one bet size" if node.bucketed else "by card class, its river sizes"
        notes = [f"read off the flop's solve, which plays the turn at one bet size and a turn class's cards alike (a "
                 f"rank's cards, a flush suit's apart): no EVs or equities until the turn's own solve" if last == "flop"
                 else f"read off the turn's solve, which plays the river at its own sizes and a river class's cards "
                      f"alike: no EVs or equities until the river's own solve"]
        self._trim_note(sol, last, notes)
        top = max(cur.bets[oop], cur.bets[ip])
        return {"study": "node", "street": cur.name, "actor": hero, "opponent": villain,
                "position": "OOP" if hero_p == 0 else "IP", "players": n_players, "board": cards_str(board),
                "pot_bb": round(cur.pot_start + cur.bets[oop] + cur.bets[ip], 2),
                "to_call_bb": round(min(to_call, behind), 2), "stack_bb": round(min(eff - top, behind), 2),
                **part, "range_ev_bb": None, "range_eq": None,
                "opponent_combos": og["combos"], "opponent_eq": None, "opponent_grid": og["grid"],
                "approximate": {"source": "prev", "label": f"the {last} solve's view of the {cur.name} ({how})",
                                "iterations": int(sol.iters)},
                "solver": {"iterations": int(sol.iters), "by": f"the {last}'s solve",   # (measured by the walk)
                           "exploitability_pct": round(max(0.0, float(self._expl(sol))), 3)},
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

    def _from_library(self, lib, full, flop, r0, r1, st: StreetState, eff, cfg: Config, oop, ip, notes, holes=(),
                      borrow=True, players=6, **walk_kw):
        """(solution, node, r0, r1) walked from the library's solve of this flop (flop_library.LibrarySolution)
        when it holds the spot (the ranges entering the flop, pot, stack and tree) and every node of the line
        taken; else None. Where it doesn't, a spot whose ranges are that close (library_borrow: flop_approx.near_exact)
        is read as this pot's, the combos off its ranges left out but for the hands in `holes` (yours: None where one
        is); not with `borrow` False (ranges of your own: their answer is a solve); `players`: the table's size (only
        another's spots are borrowed). The walk's notes go to `notes` only when it is used."""
        try:
            sol = lib.solution(full[0], full[1], flop, r0, r1, st.pot_start, eff, cfg)
            if sol is None and borrow and self.library_borrow and len(flop) == 3:
                from .flop_approx import near_exact
                ne = near_exact(lib, full[0], full[1], st.pot_start, eff, players)
                if ne is not None:
                    need = [COMBO_INDEX[(min(h), max(h))] for h in holes if h]
                    sol = lib.solution(full[0], full[1], flop, r0, r1, st.pot_start, eff, cfg, borrow=ne, need=need)
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

    def _unraked_entry(self, lib, preflop, positions, stack, board, r0, r1, st: StreetState, eff, cfg: Config, oop,
                       ip, hole=()):
        """instant_from_unraked's solution: the library's entry for this raked table's flop, its spot found by the
        ranges the preflop line brings to the flop without rake (with it the callers' are narrower: _call_rake), read
        for the raked ranges r0, r1, every combo of which it must hold (FlopLibrary.solution's `unraked`); at another
        table size the spot those ranges borrow (library_borrow: flop_approx.near_exact, as without rake), the combos
        off its ranges left out but for your hand (`hole`), which it must hold; else None."""
        flat = pf.Table(list(positions), positions.ante, positions.straddle, None, positions.depth, positions.stacks,
                        positions.icm)
        try:
            u = preflop_ranges(preflop, flat, stack)
            sol = lib.solution(u[oop], u[ip], board, r0, r1, st.pot_start, eff, cfg, unraked=True)
            if sol is None and self.library_borrow:
                from .flop_approx import near_exact
                ne = near_exact(lib, u[oop], u[ip], st.pot_start, eff, len(positions))
                if ne is not None:
                    need = [COMBO_INDEX[(min(hole), max(hole))]] if hole else []
                    sol = lib.solution(u[oop], u[ip], board, r0, r1, st.pot_start, eff, cfg, borrow=ne, need=need,
                                       unraked=True)
            return sol
        except Exception:   # (a library that can't be read is no library)
            return None

    def _off_later(self, street: str) -> Optional[str]:
        """off_tree_later's mode on `street`."""
        m = self.off_tree_later
        return m.get(street) if isinstance(m, dict) else m

    def _off_tree_own(self, sol: Solution, spot: InstantSpot) -> Optional[dict]:
        """instant_from_own's answer facing the villain's bet or raise just made off the tree of this street's solve
        from your earlier decision (spot.own_earlier): off_tree.answer (OFF_METHOD) over that solve's decisions facing
        the sizes it has (_off_tree_read: their EVs worked out, a pass over each), the line before it walked through
        the solve. None where the last action isn't such a bet or raise."""
        from . import off_tree
        price = _off_tree_price(spot.actions, spot.oop, spot.ip, spot.hero_p, spot.pot, spot.eff)
        col = hole_col(sol, spot.hole)
        if price is None or not col:
            return None
        to, C, T, bets = price
        try:
            node, c0, c1 = self._walk(sol, spot.actions[:-1], spot.oop, spot.ip, spot.pot, spot.eff, [], spot.street)
        except (ValueError, IndexError, KeyError):
            return None
        if not isinstance(node, Decision) or node.player != 1 - spot.hero_p:
            return None
        sv = node.avg_strategy0()
        vp = 1 - spot.hero_p
        evs = lambda ch, k: sol.solver.node_action_evs(ch, c0 * sv[k] if vp == 0 else c0, c1 * sv[k] if vp == 1 else c1)
        mode = self._off_later(spot.street)
        got = _off_tree_read(node, spot.hero_p, c0, c1, C, T, evs, between=mode == "always",
                             below=mode in ("always", "outside"))
        if got is None:
            return None
        mix, near, xs = got
        what = "raise" if spot.actions[-1]["action"] == "raise" else "bet"
        pc = [f"{v * 100:.0f}%" for v in sorted(xs)]
        stored = ", ".join(pc[:-1]) + " and " + pc[-1] if len(pc) > 1 else pc[0]
        return {"actions": _off_tree_actions(near, {k: v[col[0]] for k, v in mix.items()}, to, C, T, bets, spot.hero_p,
                                             spot.behind),
                "source": "solve", "iterations": int(sol.solver.iters),
                "label": f"this street's solve from your earlier decision ({sol.solver.iters} iterations): its answers "
                         f"facing its {stored} {what}{'s' if len(xs) > 1 else ''}, read for this "
                         f"{off_tree.size_of(C, T) * 100:.0f}% {what} by its price"}

    def _off_tree_answer(self, lib, full, board, r0, r1, cur: StreetState, eff, oop, ip, hero_p, hole, behind,
                         players) -> Optional[dict]:
        """instant_from_off_tree's answer: hero facing the villain's bet or raise just made, of a size more than 15%
        off every one the library's tree has for it (OFF_TREE_TOL, as _swap_nearest), on a flop the library stores
        with the line before it (each earlier size within 15% of one in the tree): off_tree.answer (OFF_METHOD) over the
        entry's stored decisions facing that action's sizes, hero's whole range for the range's continuing share
        (_off_tree_read). None where it isn't such a spot, the villain's line is rare in the entry (RARE_LINE: the live
        solve answers it) or the entry doesn't store a decision facing it."""
        from . import off_tree
        acts = list(cur.actions)
        price = _off_tree_price(acts, oop, ip, hero_p, cur.pot_start, eff)
        if price is None:
            return None
        to, C, T, bets = price
        prev = dataclasses.replace(cur, actions=acts[:-1])
        base = self._cfg_with_real_sizes(prev, oop, ip, eff)
        real = self._cfg_with_real_sizes(cur, oop, ip, eff)
        if real.bet_sizes["flop"] == base.bet_sizes["flop"] and real.raise_sizes["flop"] == base.raise_sizes["flop"]:
            return None   # (within 15% of a size the tree has: the library answers it)
        w2, rl2 = [], []
        hit = self._from_library(lib, full, board, r0, r1, prev, eff, base, oop, ip, [], holes=(hole,),
                                 players=players, smoothing=self.prior_smoothing, only=1 - hero_p, widened=w2,
                                 board=board, rare_lines=rl2)
        if hit is None or w2 or rl2:
            return None
        sol, node, c0, c1 = hit
        col = hole_col(sol, hole)
        if not col:
            return None
        got = _off_tree_read(node, hero_p, c0, c1, C, T, lambda ch, k: sol.solver.node_action_evs(ch),
                             stored=sol.stored, tol=0.0)
        if got is None:
            return None
        mix, near, xs = got
        what = "raise" if acts[-1]["action"] == "raise" else "bet"
        tree = sorted(base.raise_sizes["flop"] if what == "raise" else base.bet_sizes["flop"])
        stored = " and ".join(f"{v * 100:.0f}%" for v in (tree if len(tree) == len(xs) else sorted(xs)))
        out = _off_tree_actions(near, {k: v[col[0]] for k, v in mix.items()}, to, C, T, bets, hero_p, behind,
                                sorted(base.raise_sizes["flop"]))
        return {"actions": out, "source": "offtree", "iterations": int(sol.iters),
                "label": f"the flop library's answers facing its {stored} {what}{'s' if len(xs) > 1 else ''}, read for "
                         f"this {off_tree.size_of(C, T) * 100:.0f}% {what} by its price (defending as often, relative to "
                         f"the minimum defence, the hands that do worst folding first)"}

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
        """solve()'s `seed` for the turn's own solve (STORE_SEED): after a library flop the entry's stored turn on the
        line the flop's walk took (`flop_walk`: walked["flop"], (sol, node, taken)), where the entry has no turn file
        the near spot's that holds the line (NEAR_SEED, _near_turn_files), after a flop solved live that solve's own
        turn (PREV_SEED, _prev_turn_strategies); else None. The seed sets the regrets at the turn's first decision and
        in position after a check, each combo's the source holds; a node it doesn't hold, a tree with no check there
        or one played on several boards is left as it is."""
        if flop_walk is None or len(board) != 4:
            return None
        sol, node, taken = flop_walk
        from . import flop_library as fl
        strategies = None
        got = _turn_file(self.library, sol, board[:3])
        files = [got] if got is not None else (f[:2] for f in _near_turn_files(self.library, sol, board[:3])) \
            if NEAR_SEED else ()
        for t, smap in files:
            if any(list(ln.get("path", ())) == list(taken) for ln in t.get("lines", ())):
                strategies = functools.partial(fl.turn_strategies, t, taken, card=board[3], smap=smap)
                break
        if strategies is None and PREV_SEED:
            strategies = _prev_turn_strategies(sol, node, board[3])
        if strategies is None:
            return None

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
                st = strategies(at)
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
                      settle=False, seed=None, settled=None):
        """A heads-up solve, from the cache when it holds this spot and ranges (run on while it has had under 80% of
        `budget`), else a new one, warm from warm_from's solve of the spot (else from `seed`: _store_seed). With
        hu_settled set it stops at the first of HU_CHECKS where it is that close to equilibrium; one settled already
        isn't run on. A solve stopped part way (should_stop: Cancelled) stays cached, and the spot asked again goes
        on from where it stopped. `settled`: hu_settled's target for this solve instead."""
        cfg = cfg or self.cfg
        key = self._solve_key(board, r0, r1, pot, stack, bets, to_act, n_raises, cfg)
        spot = key[2:]
        sol = self._cache.get(key)
        target = self.hu_settled if settled is None else settled
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
            look = None if warm else self._look_at(n, None)   # (Advisor.progress: stopped there to be read)
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
                                max_cells=cap, max_iters=min([c for c in HU_CHECKS if c >= first] if target is not None
                                                             else [1000], default=1000) if look is None else look,
                                horizon=1000, seed=seed,
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
            if look is not None:   # (Advisor.progress: read at its look, or on to the next)
                self._go_on(sol, n, lambda: left - (time.time() - t0), target, first, look)
            elif target is not None:
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
                look = self._look_at(n, sol)   # (Advisor.progress: stopped there to be read)
                if target is not None:
                    self._hu_run(sol, budget - sol.seconds, target, first, stop_at=look)
                else:       # keep going from the stored regrets instead of restarting
                    try:
                        sol.extend(budget - sol.seconds, max_iters=look or 1000, should_stop=self.should_stop,
                                   reserve=HU_RESERVE)
                    except Cancelled:
                        sol.stopped = True
                        raise
                if look is not None:
                    self._go_on(sol, n, lambda: budget - sol.seconds, target, first, look)
        if settle:
            self._settle(sol, budget)
        return sol

    @staticmethod
    def _expl(sol: Solution) -> float:
        """sol.exploitability(), read once per iteration count."""
        if getattr(sol, "expl_at", (None,))[0] != sol.iters:
            sol.expl_at = (sol.iters, sol.exploitability())
        return sol.expl_at[1]

    def _hu_run(self, sol: Solution, left: float, target: float, first: int = HU_CHECKS[0],
                stop_at: Optional[int] = None) -> None:
        """Runs a heads-up solve on for at most `left` seconds, stopping at the first of HU_CHECKS from `first` on
        where its exploitability is under `target` (% of the pot): sol.settled_at = target then; or at `stop_at`
        iterations (Advisor.progress's look). (Its pieces run the solver directly: Solution.extend's EV pass after each
        would cost about an iteration a piece.)"""
        t0 = time.time()
        checks = [c for c in HU_CHECKS if c >= first]
        while sol.iters < 1000:
            if sol.iters in checks and self._expl(sol) < target:
                sol.settled_at = target
                break
            if stop_at is not None and sol.iters >= stop_at:
                break
            rem = left - (time.time() - t0)
            nxt = min([c for c in checks if c > sol.iters] + [1000] + ([stop_at] if stop_at is not None else []))
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
        past the advice's cap. Each solve stops once settled (one_size_settled), once clearly behind the best size
        solved before it ("cut": _one_size_run; "all" solves the sizes in one_size_order, the full menu's favourites
        first) or at the request's budget. A row per
        solve: the root's actions and the range's frequency of each, the range's EV (bb relative to folding now,
        weighted by hand pairs: the game's value), the grid, hero's hand (with cards), its iterations and
        exploitability. "one_size": "all" solves the full menu and every size and adds each one's EV given up against
        the full menu (loss_pct, % of the pot) and the best (one_size_summary); "full" or a size ("50%", 50, "all-in";
        one that isn't a candidate is solved too, 5-500%) solves that one; "list" none. Rows solved here are kept
        (ONE_SIZE_CACHE), so asking again answers at once, and a request answers with every row it has kept for the
        spot ("rows"). At equilibrium only: not against the pool or locks. A tournament's solves pay prize equity
        (Config.icm, as the advice's: EVs in chips of equal worth), and stop early only when settled: the bounds the
        early stop reads are a zero-sum game's."""
        want = self._one_size
        street = cur.name
        if exploit:
            return {"error": "the one-size view is at equilibrium: it isn't worked out against the pool or your locks"}
        if node.player != hero_p:
            return {"error": "no decision of yours here"}
        bets = tuple(float(b) for b in node.bets)
        behind = eff - bets[hero_p]
        facing = "check" not in node.names
        if behind <= 1e-9:
            return {"error": "no chips behind to bet"}
        if facing and (eff - bets[1 - hero_p] <= 1e-9 or not any(n.startswith("raise") or n == "all-in"
                                                                   for n in node.names)):
            return {"error": "no raise here: calling puts the stacks in"}
        pot = float(node.pot_now)
        base = self._cfg_with_real_sizes(cur, oop, ip, eff)
        base.keep_sizes, base.root_allin, base.root_bets, base.root_raises = None, False, None, None
        icm = getattr(self, "_icm", None) is not None
        if icm:   # (a tournament: the payoffs from this decision, the bets already in counted: _icm_cfg)
            base = self._icm_cfg(base, cur, oop, ip, bets)
        if facing:
            # (raises: by the multiple of the bet faced, each the tree's raise to as a fraction of the pot after calling)
            b_op, b_me = bets[1 - hero_p], bets[hero_p]
            raises = one_size_raises(b_op, b_me, pot, eff, base.allin_spr)
            cands = [(lab, f) for lab, f, _ in raises]
            raise_to = {lab: t for lab, _, t in raises}
        else:
            cands = one_size_candidates(pot, behind, base.allin_spr, river=street == "river")
        frac = dict(cands)
        if facing and want not in ("all", "full", "list"):
            w = str(want).strip().lower()
            try:
                x = None if w == "all-in" else float(w.rstrip("x"))
            except ValueError:
                return {"error": f"one_size: {want!r} is not a raise size (\"3x\", 3, \"all-in\", \"full\", \"all\" "
                                 f"or \"list\": facing a bet the sizes are raises, by the multiple of the bet)"}
            if x is not None and not ONE_RAISE_X[0] <= x <= ONE_RAISE_X[1]:
                return {"error": f"one_size: raises are {ONE_RAISE_X[0]:g}-{ONE_RAISE_X[1]:g}x the bet"}
            lab, f, t = one_size_raises(b_op, b_me, pot, eff, base.allin_spr, (x,) if x is not None else (),
                                        allin=None if x is not None else True)[0]
            want = lab
            frac.setdefault(lab, f)
            raise_to.setdefault(lab, t)
        elif want not in ("all", "full", "list"):
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
        # (one_size_order's: each raise's action in the full menu's tree and its multiple of the bet)
        order_by = ({lab: "all-in" if lab == "all-in" else f"raise to {raise_to[lab]:g}" for lab in raise_to},
                    {lab: raise_to[lab] / max(b_op, 1e-9) for lab in raise_to}) if facing else ()
        R0, R1 = expand(sol.ctx, n_r0), expand(sol.ctx, n_r1)
        target = self.one_size_settled[street] if self.expl_target is None else \
            min(self.one_size_settled[street], self.expl_target)
        # (the rows hold your hand's line, and every hand of a node has the same ranges: by hand too)
        spot = (self._solve_key(board, R0, R1, cur.pot_start, eff, bets, hero_p, cur.n_raises, base), target, tuple(hole))
        full_menu = [frac[c] for c, _ in cands]
        level = self._one_size_trim(board, R0, R1, cur, eff, bets, hero_p, base, full_menu, facing)
        cfg0 = base if not level else self._trim_cfg(base, street, level)
        i = 0
        while i < len(todo):
            label = todo[i]
            i += 1
            key = (spot, label)
            if key in self._one_rows:
                self._one_rows[key] = self._one_rows.pop(key)   # (the most recently used last)
            else:
                cfg = cfg0.copy()
                setattr(cfg, "root_raises" if facing else "root_bets", full_menu if label == "full" else [frac[label]])
                cut = self._one_size_bar(spot, target, pot) if label != "full" and self.one_size_cut else None
                t1 = time.time()
                s1 = self._one_size_solve(board, R0, R1, cur, eff, bets, hero_p, cfg, target, cut=cut,
                                          checks=ONE_SIZE_CHECKS if label == "full" else HU_CHECKS)
                row = self._one_size_row(s1, label, hero_p, hole, time.time() - t1)
                del s1
                self._one_rows[key] = row
                while len(self._one_rows) > ONE_SIZE_CACHE:
                    del self._one_rows[next(iter(self._one_rows))]
            if label == "full" and want == "all":
                # (the sizes in the order the full menu suggests: the likely best first, so the rest can stop early)
                todo = todo[:i] + one_size_order([c for c, _ in cands], self._one_rows[key], *order_by)
        known = {lab: r for (sp, lab), r in self._one_rows.items() if sp == spot}
        labels = [c for c, _ in cands]
        rows = [known[c] for c in labels if c in known] + \
            [r for lab, r in known.items() if lab not in labels and lab != "full"]   # (sizes asked that aren't candidates)
        full = known.get("full")
        out = {"street": street, "actor": hero, "opponent": villain, "position": "OOP" if hero_p == 0 else "IP",
               "board": cards_str(board), "pot_bb": round(pot, 2), "stack_bb": round(behind, 2),
               "candidates": [c for c, _ in cands], "asked": want,
               "definition": (ONE_RAISE_DEF if facing else ONE_SIZE_DEF) + (ONE_SIZE_ICM if icm else ""),
               "target_pct": target,
               "full": None if full is None else {k: v for k, v in full.items() if not k.startswith("_")},
               **one_size_summary(full, rows, pot)}
        if facing:
            # (each raise: what it raises to, bb, and that as % of the pot after calling)
            for lab in [r["size"] for r in rows if r["size"] not in raise_to]:   # (asked before, not a candidate)
                raise_to[lab] = one_size_raises(b_op, b_me, pot, eff, base.allin_spr, (float(lab[:-1]),))[0][2]
            out.update(kind="raise", to_call_bb=round(b_op - b_me, 2),
                       raises={lab: {"to": round(raise_to[lab], 2),
                                     "pct": round((raise_to[lab] - b_op) / (pot + b_op - b_me) * 100)}
                               for lab in labels + [r["size"] for r in rows if r["size"] not in labels]})
        if full is not None:
            out["order"] = one_size_order(out["candidates"], full, *order_by)
        if hole:
            out["hand"] = hand_class(hole)
            out["hole"] = cards_str(hole)
        more = []
        if level:
            more.append(f"{street}: every size's tree is cut alike to fit this speed's time and memory: {TRIMS[level - 1]}")
        rest = base.raise_sizes.get(street, []) if facing else base.bet_sizes.get(street, [])
        more.append(f"{street}: one {'raise ' if facing else ''}size at {'your' if hole else hero + chr(39) + 's'} "
                    f"decision only; the rest of the tree is the advice's ({villain}'s "
                    f"{'re-raises' if facing else 'bets'} {', '.join(f'{int(round(x * 100))}%' for x in rest)}"
                    f"{' of the pot after calling' if facing else ''}{', and the later streets' if street != 'river' else ''})")
        return {"one_size": out, "street": street, "hero": hero, "villain": villain,
                "notes": [n for n in notes if "to tree action" not in n] + more, "seconds": round(time.time() - t0, 2)}

    def _one_size_trim(self, board, R0, R1, cur: StreetState, eff, bets, to_act, base: Config, full_menu,
                       facing: bool = False) -> int:
        """The TRIMS level the one-size view's trees are cut to: the full menu's tree (the biggest) at the cap a new
        solve of the street has here (_cells_cap: memory, and on the turn and river the request's time), else 0."""
        cap = self._cells_cap(len(board), self.time_budget)
        if cap is None:
            return 0
        for level in range(len(TRIMS) + 1):
            cfg = base if not level else self._trim_cfg(base, cur.name, level)
            cfg = cfg.copy()
            setattr(cfg, "root_raises" if facing else "root_bets", list(full_menu))
            try:
                tree_cells(board, R0, R1, cur.pot_start, eff, bets=bets, to_act=to_act, n_raises=cur.n_raises,
                           cfg=cfg, max_cells=cap if level < len(TRIMS) else self.max_cells)
                return level
            except TreeTooBig:
                continue
        return len(TRIMS)

    def _one_size_solve(self, board, R0, R1, cur: StreetState, eff, bets, to_act, cfg: Config, target: float,
                        cut=None, checks=HU_CHECKS):
        """A one-size view's solve from the decision: settled at `target` (% of the pot, read at `checks`: the full
        menu's ONE_SIZE_CHECKS), out of contention (`cut`: _one_size_run) or at the request's budget, whichever comes
        first. Not kept in the advice's cache (_cache): the view keeps its rows. (Starting a size's solve from the full menu's regrets, or the full
        menu's from the advice's, saved nothing: a turn's 150% solve 0.092% of the pot from equilibrium at 300
        iterations cold, 0.102% seeded; its full menu 0.19 cold, 0.17 seeded.)"""
        t0 = time.time()
        sol = solve(board, R0, R1, cur.pot_start, eff, bets=bets, to_act=to_act, n_raises=cur.n_raises, cfg=cfg,
                    time_budget=self.time_budget, max_iters=checks[0], horizon=1000,
                    should_stop=self.should_stop, reserve=HU_RESERVE)
        self._one_size_run(sol, self.time_budget - (time.time() - t0), target, to_act, cut, checks)
        return sol

    def _one_size_run(self, sol: Solution, left: float, target: float, p: int, cut=None, checks=HU_CHECKS) -> None:
        """Runs a one-size solve on for at most `left` seconds, stopping at the first of `checks` where it is
        within `target` (% of the pot) of equilibrium (sol.settled_at), or, given `cut` (bb: _one_size_bar), where
        what the player p betting can get at best (its best response's value against the opponent's average strategy,
        an upper bound on the game's value) is under it: the size can't be the best nor tied with it (sol.cut_at, its
        iterations). Each read leaves sol.bounds: (the lower, the upper bound on p's value, bb relative to folding
        now), from the two best responses (a zero-sum game's; None with rake or a tournament's payoffs)."""
        t0 = time.time()
        inv = _inv_of(sol.root, p, sol.ctx.cfg.icm)
        while sol.iters < 1000:
            if sol.iters in checks:
                e = self._expl(sol)
                br = getattr(sol.solver, "br_values", None)
                sol.bounds = None if br is None or br[0] != sol.iters or br[3] else \
                    (-br[2 - p] + inv, br[1 + p] + inv)
                if e < target:
                    sol.settled_at = target
                    break
                if cut is not None and sol.bounds is not None and sol.bounds[1] < cut(e):
                    sol.cut_at = sol.iters
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
                sol.seconds += time.time() - t1
                raise
            sol.iters = sol.solver.iters
            sol.seconds += secs
            if sol.iters < nxt:
                break       # (out of time)
        if not sol.fresh():
            sol._refresh()

    def _one_size_bar(self, spot, target: float, pot: float):
        """The one-size view's cut for a size's solve (_one_size_run), from the best of the spot's sizes solved so far
        (the highest EV; None before any): a function of the solve's exploitability e (% of the pot) giving the bar
        (bb) its upper bound must be under to stop: ONE_SIZE_CUT x the target under the best's lower bound, so its game
        is worth less than the best's even once both are settled, and more than e under the best's EV, so the verdict
        (one_size_summary: tied within the larger exploitability) doesn't count it tied either."""
        rows = [r for (sp, lab), r in self._one_rows.items() if sp == spot and lab != "full"]
        if not rows:
            return None
        top = max(rows, key=lambda r: r["ev_bb"])
        lo = top.get("_lo")
        if lo is None:
            lo = top["ev_bb"] - 2 * (top.get("exploitability_pct") or 0.0) / 100 * pot
        bar = lo - ONE_SIZE_CUT * target / 100 * pot
        return lambda e: min(bar, top["ev_bb"] - max(e, top.get("exploitability_pct") or 0.0) / 100 * pot)

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
        if getattr(sol, "cut_at", None) is not None:
            row["cut"] = True   # (stopped once clearly behind the best: _one_size_run)
        if getattr(sol, "bounds", None) is not None and sol.iters == sol.solver.br_values[0]:
            row["_lo"] = round(float(sol.bounds[0]), 4)   # (its game's value is at least this: the next sizes' bar)
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
              taken=None, off=None, resolve=None, resolved=None):
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
        could have raised (_slowplayed; `slow`, {player: mask} carried from street to street, as multiway).
        `off` (the answer's off_solution) gets every line read by the rules: street, seat, the tree's action and the
        share of their range the solve takes it with (%), and "resolved" where the play after it was solved again.
        `resolve` (_rare_resolver; not with `lock`): after a line read by the rules, the walk goes on in the street
        solved again from the node with the ranges it reached it with (RARE_RESOLVE), but where a library flop stores
        that node; `resolved` (a list) gets (that solve, the number of `taken` before it)."""
        rules = (only,) if rules is None and only is not None else rules or ()
        node = sol.root
        r0, r1 = sol.r0.copy(), sol.r1.copy()
        bets = [0.0, 0.0]
        n_bets = 0          # bets and raises so far this street (the tree's n_raises)
        after_rare = None   # with `resolve`: off's entry (or {}) for the rare line the rules read just before
        for a in acts:
            if not isinstance(node, Decision):
                break
            if a["pos"] not in (oop, ip):
                continue
            p = 0 if a["pos"] == oop else 1
            if p != node.player:
                notes.append(f"{street}: action order mismatch at {a['pos']} {a['action']}; ignoring")
                continue
            # (a library flop: where it stores the node it's read; where it doesn't store the line up to it, it isn't used)
            lib_sol = hasattr(sol, "covers")
            if after_rare is not None and not (lib_sol and (sol.covers([node]) or not sol.covers(visited or []))):
                got = resolve(expand(sol.ctx, r0), expand(sol.ctx, r1), (bets[0], bets[1]), p, n_bets)
                if got is not None:
                    sol, trust = got
                    node = sol.root
                    r0, r1 = sol.r0.copy(), sol.r1.copy()
                    visited = None   # (the nodes from here are the new solve's)
                    if resolved is not None:
                        resolved.append((sol, len(taken) if taken is not None else 0))
                    after_rare["resolved"] = True
                    notes.append(f"{street}: the play after it solved again from there ({sol.iters} iterations)")
            after_rare = None
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
                share = float((rp * s).sum() / max(rp.sum(), 1e-12) * 100)
                entry = {"street": street, "pos": a["pos"], "why": "rare", "action": node.names[k],
                         "share": round(share, 1)}
                if off is not None:
                    off.append(entry)
                if resolve is not None and lock is None:
                    after_rare = entry
                notes.append(f"{street}: the solve rarely has {a['pos']} {a['action']} here "
                             f"({share:.0f}% of their range), so their range "
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
            top = max(bets)
            if act in ("bet", "raise"):
                amt = a.get("to") if a.get("to") is not None else a.get("amount")
                if amt is None:
                    amt = node.children[k].bets[p] if isinstance(node.children[k], Decision) else bets[1 - p]
                bets[p] = float(amt)
            elif act == "all-in":
                bets[p] = stack
            elif act == "call":
                bets[p] = min(bets[1 - p], stack)
            if bets[p] > top + 1e-9:
                n_bets += 1
            node = node.children[k]
        return node, r0, r1

    def _rare_resolver(self, board, st, eff, cfg) -> Optional[Callable]:
        """_walk's `resolve` on an earlier street (RARE_RESOLVE): (R0, R1, bets, to_act, n_raises) -> (the street
        solved again from the node after a rare line, with the ranges the walk reached it with, at prior_budget, and its
        trust_of), or None where it isn't solved (an answer that is instant or quick: the street's solve is read on)."""
        if not RARE_RESOLVE or getattr(self, "_quick", False):
            return None

        def resolve(R0, R1, bets, to_act, n_raises):
            b = (min(bets[0], eff), min(bets[1], eff))
            try:
                sol = self._solve_cached(board, R0, R1, st.pot_start, eff, self.prior_budget, bets=b, to_act=to_act,
                                         n_raises=n_raises, cfg=cfg, settled=RARE_RESOLVE_SETTLED)
            except (_NoInstant, _NotHeld):
                return None
            return sol, trust_of(self._expl(sol), st.name)
        return resolve

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


def _study_mix(board, names, amounts, me, sf, villain, opp):
    """_study_range for Study's answer at once (Advisor._study_store, _study_near): the mix alone, every EV and
    equity None (the grid's cells', the combos' and the actions'); (part, the opponent's)."""
    none = np.full(N_COMBOS, np.nan)
    part, (og,) = _study_range(board, names, amounts, me, sf, np.zeros_like(sf), none, [(villain, opp, none)])
    _no_values(part)
    return part, og


def _no_values(part: dict) -> None:
    """A _study_range answer's EVs and equities set to None (the grid's cells', the combos' and the actions')."""
    for c in part["grid"]["cells"].values():
        c.update(ev=None, evm=None, eq=None)
    for rows in part["combos"].values():
        for row in rows:
            row[3], row[4] = None, None
    for a in part["actions"]:
        a["ev_bb"] = None


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


def one_size_raises(b_op: float, b_me: float, pot: float, stack: float, allin_spr: float, mults=ONE_RAISES,
                    allin: Optional[bool] = None) -> List[Tuple[str, float, float]]:
    """The one-size view's raise sizes facing a bet (b_op the opponent's street total, b_me ours, `pot` in the middle
    with both, `stack` the street's effective stack): (label, fraction of the pot after calling: Config.root_raises,
    raise to) for a raise to each of `mults` times the bet ("3x"), as the tree builds it (solver.TreeBuilder._decision:
    the raise rounds to half a big blind, at least a min-raise; one reaching the stack is the all-in), sizes that come
    out the same raise once, and all-in where the tree offers it (at most allin_spr pots behind once called) or a
    multiple reaches the stack (`allin` True or False: always or never)."""
    pac = pot + b_op - b_me
    out, seen = [], set()
    for k in mults:
        to = max(b_op + round((k * b_op - b_op) / pac * pac * 2) / 2, 2 * b_op - b_me)
        if to >= stack - 1e-9:
            allin = True if allin is None else allin
            continue
        if to in seen:
            continue
        seen.add(to)
        out.append((f"{k:g}x", (to - b_op) / pac, to))
    if allin or (allin is None and stack - b_op <= allin_spr * pac):
        out.append(("all-in", ONE_SIZE_ALLIN, stack))
    return out


def one_size_order(labels: List[str], full: dict, names: Optional[dict] = None, sizes: Optional[dict] = None) -> List[str]:
    """The order the one-size view solves its sizes in (the answer's "order"; the page asks them so): the sizes the
    full menu's range bets (or raises) most first, then the rest nearest the sizes it uses (in log size, weighted by
    how much), all-in as the biggest, so the likely best is in early and the sizes clearly behind it can stop early
    (_one_size_run). Which size is best doesn't depend on it. names: each label's action in the full menu's tree
    (else "bet 50%" for "50%"), sizes: each label's size (else the label's %; all-in 10 pots)."""
    names, sizes = names or {}, sizes or {}
    act = lambda c: names.get(c) or (c if c == "all-in" else "bet " + c)
    size = lambda c: sizes.get(c) or (10.0 if c == "all-in" else float(c.rstrip("%")) / 100)
    share = {n: float(f) for n, f in zip(full.get("actions", ()), full.get("freq", ()))}
    used = {c: share.get(act(c), 0.0) for c in labels}
    tot = sum(used.values())

    def away(c):
        if tot <= 0:
            return 0.0
        return sum(f * abs(np.log(size(c) / size(k))) for k, f in used.items()) / tot
    return sorted(labels, key=lambda c: (-round(used[c], 3), away(c), size(c)))


def one_size_summary(full: Optional[dict], rows: List[dict], pot: float) -> dict:
    """The one-size view's verdict over the rows solved so far (each {"size", "ev_bb", "exploitability_pct", ...}) and
    the full menu's row: each row's EV given up against the full menu (loss_pct, % of the pot; None without it), the
    best size (the highest EV: the least given up) and the sizes tied with it (within the precision of the two solves,
    the larger of their exploitabilities)."""
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    for r in rows:
        r["loss_pct"] = None if full is None else round((full["ev_bb"] - r["ev_bb"]) / max(pot, 1e-9) * 100, 3)
    if not rows:
        return {"rows": rows, "best": None, "tied": []}
    top = max(rows, key=lambda r: r["ev_bb"])
    tol = lambda r: max(r.get("exploitability_pct") or 0.0, top.get("exploitability_pct") or 0.0) / 100 * pot
    tied = [r["size"] for r in rows if r is not top and top["ev_bb"] - r["ev_bb"] <= tol(r)]
    return {"rows": rows, "best": top["size"], "tied": tied}
