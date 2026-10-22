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
}
Actions: fold, check, call, bet (amount), raise (to), all-in.
"""
from __future__ import annotations
import dataclasses
import hashlib
import random
import time
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .cards import cards as parse_cards, COMBO_INDEX, hand_class, ALL_COMBOS, cards_str
from .ranges import N_COMBOS, range_to_text, parse_range
from . import preflop as pf
from .pf_classes import push_fold
# imported where used (exploit, breakdown, like multiway_solver and pf_game): the browser's worker imports this module
# before its first answer, and a preflop answer or a library flop needs none of them (tests/test_bundle.py)
from .solver import solve, Config, Solution, Decision, Terminal, Chance, match_action, expand, MEMBER_FULL, _inv_of

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
HU_RESERVE = 1.0   # heads-up solves likewise keep one iteration's time back (the exploitability read after: ~1 iteration)
MW_SETTLED = 0.1   # a 3-way solve stops once its players' mean best-response gain is under this (% of the pot)...
MW_CHECKS = (25, 50, 100, 200, 400, 700)   # ...checked at these iterations (about two iterations' time each). The
                   # browser's Accurate ran a 3-way river 60 s to 386 iterations, 0.01% of the pot, and a turn 330, 0.03%
MW_RESERVE = 2.5   # 3-way solves stop this many iterations' time short of the budget: best-response gains and EVs after
XR_Z = 2.0          # multiway: a check-raise beats calling/folding by this many standard errors (see chk_plan)
XR_MIN = 30         # ...over at least this many trials of that situation
LATER_CACHE = 40000  # multiway: most later-street decisions stored per call (memory bound, ~25 KB each at most)
HIT_TOP = 0.15      # multiway bluffs: a hand "gets there" on a runout ending in this top share of combos (judgment)
SHOWDOWN_Q = 0.6    # ...and has showdown value (bluffs last) when it usually ends above this share (judgment)
VALUE_SHARE = 0.35  # multiway: a bet's value range is this share of what would call it
SMALL_SHARE = 0.7   # multiway: the small bet's value range is this share of what one player would call it with,
                    # compounding with more players (1 - 0.3^n of what the field calls); fitted, see _bet_ranges
SMALL_VALUE_MAX = 0.55  # ...but at most this share of the bettor's range: behind two checks (capped ranges) the
                        # compounding made nearly all of it value and the button bet 93-100%, so AA checked to
                        # check-raise it (check +18.8 bb over a bet's +15.3; now 13.5 / 13.4, the button betting 66%);
                        # the fitted UTG-into-both-blinds mix over 200 flops moves from 42 / 44 / 14 to 45 / 41 / 14
NUT_TOP = 0.1       # multiway: the nuts, for the big bet: this top share of the players' ranges pooled
ADV0, ADV_SPAN = 0.4, 0.2  # ...the big bet starts at this share of the nuts and takes all the value over this much more
RAISE_SHARE = 0.2   # multiway: facing a bet, this share of what continues (the strongest) raises
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
PUSH_FOLD_STACK = 12.0  # preflop: at or below this many bb, raising first in is all-in (push/fold)
CHART_STACK = 100.0  # the depth the preflop charts are written for
IMPLIED = {"pair": 0.06, "suited": 0.03}  # preflop calls: share of the stack behind (beyond the chart's) small pairs / suited hands expect to win later
DEPTH_TOL = 20.0   # ...a spot this many bb deeper or shallower than the charts re-prices calls
PRICE_TOL = 0.02    # ...and a spot within this much of the chart's pot odds, against the raisers only, is the chart's own
CUT_FLOOR = 0.05    # ...a worse price holds the chart's continuing hands to the equity of its weakest this share of them
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


@lru_cache(maxsize=1)
def _preflop_boards():
    """Fixed random boards and every combo's strength on each (-1 = blocked), for preflop equity."""
    from .evaluator import eval_board
    rng = random.Random(1)
    return np.array([eval_board(rng.sample(range(52), 5)) for _ in range(PREFLOP_BOARDS)])


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


@lru_cache(maxsize=1)
def _board_index():
    """Per preflop board, the orders and lookups _beat_share needs that don't depend on the range: every
    combo by strength, every (card, strength) entry of a combo holding that card, and where each combo's own
    strength falls in both."""
    out = []
    for s in _preflop_boards():
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
    return out


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


def _preflop_equity_multi(villains: List[np.ndarray]) -> np.ndarray:
    """Preflop equity of every combo against several weighted ranges at once: on each board, the product
    of the chances of beating each (ties count half; card removal between the villains is ignored), over
    the first MULTI_BOARDS boards (each class pools its combos below)."""
    if len(villains) == 1:
        return _preflop_equity(villains[0])
    sts = _preflop_boards()
    num = np.zeros(N_COMBOS); den = np.zeros(N_COMBOS)
    tw = [max(float(r.sum()), 1e-12) for r in villains]
    for bi, s in enumerate(sts[:MULTI_BOARDS]):
        live = s >= 0
        p = live.astype(float)
        bw = live.astype(float)
        for r, t in zip(villains, tw):
            sh, tot = _beat_share(bi, np.where(live, r, 0.0))
            p = p * sh
            # boards are as likely as the players' hands leave them: a board of the cards their ranges hold
            # comes less often (high cards, against ranges of broadways), which the uniform boards missed: 76s
            # and 22 read 2-3 points under a dealt-out simulation against five players, Q9o 2 over
            bw = bw * tot / t
        num += p * bw; den += bw
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
                   slow: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Narrow each player's range (in place) by what they did on a street that started multiway,
    with the multiway rules: a check drops most of the hands that would have bet, a bet or raise
    keeps its value + bluff range, a call keeps what continues minus what would have raised.
    Hands a rule rules out keep some weight (players don't follow our rules), see _soften. `slow`
    (carried from street to street): per player, the combos a call has already slowplayed. `stack`: everyone's
    stack in bb, or {seat: stack}."""
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
        ranges[p] = r * (_soften(r, keep_m, CHECK_SOFT) if act == "check" and len(board) == 3 else _soften(r, keep_m))
        if act == "call" and to_call > 1e-9 and raise_ok:
            # the hands the rules have raising instead are slowplayed some of the time: cut whole, an
            # overcaller's range after three streets of calls was underpairs (TT-55 on K72-5-9, no Kx left)
            # and KQ overcalled a pot-sized river bet
            k = _keep(to_call, pot, n) if call_k is None else call_k
            band = _top(r, k * RAISE_SHARE, rank)
            # a hand that already slowplayed an earlier call mostly keeps on (halved again each street, an
            # overcaller after three calls was TT-55 with KJs at 0.07)
            again = slow.get(p, np.zeros(N_COMBOS, dtype=bool)) if slow is not None else np.zeros(N_COMBOS, dtype=bool)
            w = np.where(again, SLOWPLAY_AGAIN, SLOWPLAY)
            ranges[p] = np.where(band, np.maximum(ranges[p], w * r), ranges[p])
            if slow is not None:
                slow[p] = again | band


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
    return (kind, p, acts, tuple(positions), getattr(positions, "ante", 0.0), getattr(positions, "straddle", False), stack,
            getattr(positions, "rake", None), pf.stacks_key(positions))


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
        if "call" in full:
            st = replay_hand(positions, stack, {"preflop": before}, 1)[0]
            if not st.closed and st.to_act == p:
                full = Advisor._priced(p, before, positions, stack, st, full, [])
        return full
    return _pf_cached(_pf_key("s", p, before, positions, stack), work)


def preflop_ranges(actions: List[dict], positions, stack: float, pool_except: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Each still-active player's range after the preflop actions (pf.replay_ranges with the adjusted
    strategies). With `pool_except` (hero's seat), everyone else plays like a typical player pool
    (exploit.pool_preflop)."""
    positions = pf.at_depth(positions, stack)

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
                jr, _ = _jam_range(actions, ji, positions, stack, pool_except)
                st = replay_hand(positions, stack, {"preflop": actions[:i]}, 1)[0]
                callers = [h["pos"] for h in actions[ji + 1:i] if h["action"] in ("call", "all-in") and h["pos"] in ranges]
                eq = _preflop_equity_multi([jr] + [ranges[c] for c in callers])
                rem = depth - st.inv_start[pos] - st.bets[pos]
                to_call = min(st.cur_bet - st.bets[pos], rem)
                pot = st.pot_start + sum(st.bets.values())
                win = pot - max(0.0, st.bets[jam["pos"]] - (st.bets[pos] + to_call))
                need = to_call / max(win + to_call - _rake(positions, win + to_call), 1e-9)
                call = np.nan_to_num(np.clip(0.5 + (eq - need) / JAM_BLEND, 0.0, 1.0), nan=0.0)
                r, mostly_filler = pf.thin_line(ranges[pos] * call, call, ranges[pos])
                if mostly_filler:
                    rebuilt.append(pos)
                ranges[pos] = r
            elif act == "all-in" and sum(h["action"] in ("raise", "all-in") for h in actions[:i]) <= 1:
                # an open- or 3bet-jam: the push/fold range, as our call of it reads it (_jam_range), not the
                # chart's open or 3bet (a 100bb open-jam read as CO's whole 27% open)
                ranges[pos] = _jam_range(actions, i, positions, stack, pool_except)[0]
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
    return _pf_cached(_pf_key("r", pool_except, actions, positions, stack), work)


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


def _jam_range(actions: List[dict], ji: int, positions, stack: float, pool_except: Optional[str] = None):
    """(the range of the player whose all-in is actions[ji], notes). An open- or 3bet-jam: push/fold
    equilibrium against the players left to act (the opener, if any, is the one who calls); later, the
    replayed (or the pool's) raising range through the jam."""
    jammer, before = actions[ji]["pos"], actions[:ji]
    prev = replay_hand(positions, stack, {"preflop": before}, 1)[0]
    arr = preflop_ranges(before, positions, stack, pool_except)
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
        push = push_fold(rng, call_rng, jam, prev.bets[jammer], caller_in, pot_before, len(opp),
                         rake=getattr(positions, "rake", None), alt=alt)
        share = float((rng * push).sum() / max(rng.sum(), 1e-12))
        rng = rng * push
        notes.append(f"{jammer}'s {jam:g}bb jam range: push/fold equilibrium against {len(opp)} "
                     f"player{'s' * (len(opp) > 1)} left ({share:.0%} of their hands)")
    else:
        # the replayed range through the jam: the chart's (or the pool's) raising range, rebuilt
        # from it alone when the jammer's line is off-chart (a flat, then a jam over the 4bet)
        jr = preflop_ranges(actions[:ji + 1], positions, stack, pool_except)
        rng = jr.get(jammer, rng)
        if pool_except is not None:
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


_EQ_ANY = None


def _equity_vs_any() -> np.ndarray:
    """Each combo's preflop equity against a random hand (how a pool player ranks hands to limp)."""
    global _EQ_ANY
    if _EQ_ANY is None:
        _EQ_ANY = _preflop_equity(np.ones(N_COMBOS))
    return _EQ_ANY


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


def _mw_names(names, amounts, behind: float, S: float, pot_now: float) -> List[str]:
    """A 3-way solve's action names as the player plays them. The solve plays everyone at the shortest stack behind
    (S), so its all-in for a player with more is a bet (or raise) to S, not their whole stack: recorded as all-in on
    the page, a deep player's all-in put 97.5 bb in where the solve bet 9.5 (a short stack's cap)."""
    out = list(names)
    facing = bool(names) and names[0] == "fold"   # (fold is offered only facing a bet; else nothing is in this street)
    for i, (nm, amt) in enumerate(zip(names, amounts)):
        if nm == "all-in" and amt is not None and behind > S + 1e-6:
            alt = f"raise to {amt:g}" if facing else f"bet {int(round(100 * amt / max(pot_now, 1e-9)))}%"
            out[i] = alt if alt not in out else f"{'raise to' if facing else 'bet'} {amt:g}bb"
    return out


# ----------------------------------------------------------------------------- advisor
class Advisor:
    def __init__(self, cfg: Optional[Config] = None, time_budget: float = 4.0, prior_budget: float = 2.0,
                 prior_smoothing: float = 0.15):
        self.cfg = cfg or Config()
        self.time_budget = time_budget
        self.prior_budget = prior_budget
        self.prior_smoothing = prior_smoothing
        self._cache: Dict[tuple, Solution] = {}
        self.should_stop = None   # callable; when it returns true the running solve raises solver.Cancelled
        self.mw_rake_rules = True   # multiway: our in-trial calls, leads and raises count the rake (off: payouts only)
        self.mw_trials: Optional[int] = None   # multiway: play at least this many trials (whole passes), no time caps or early stop (tests)
        # 3-way pots: the streets solved by the 3-player solver (gto/multiway_solver.py; others, and 4+ players, use the
        # betting rules), MWConfig overrides, and the most iterations a solve runs (it stops at time_budget before)
        self.mw_solve_streets: Tuple[str, ...] = ("turn", "river")   # (native and browser Accurate add the flop)
        self.mw_config: Optional[dict] = None
        # per street MWConfig overrides: from the flop and turn the later streets are bet on by hand buckets (dealt out
        # as equity leaves, a flop solve bet half as often as published solves: README, 3-way pots)
        self.mw_street_config: Dict[str, dict] = {"flop": dict(MW_LATER, bet=MW_FLOP_BETS), "turn": dict(MW_LATER)}
        self.mw_max_iters: int = 1000
        self.mw_max_expl: float = 10.0   # a solve further than this from equilibrium (% pot) hands over to the rules
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
        # the street advised on (heads-up from its start) also gets the earlier streets' prior_budget its narrowing
        # didn't spend (native Fast: README, Accuracy vs speed); off: time_budget alone
        self.spare_prior = False
        self.rng = random.Random()
        self._breakdown = False   # this request asked for the breakdown (state "breakdown": true)
        self._lib_only = False    # state "library_only": a flop the library holds, else {"skipped": …}, no solve
        self._req_sizes: Dict[str, List[float]] = {}   # the request's custom sizes (state "sizes"), as fractions

    # ------------------------------------------------------------ public
    def advise(self, state: dict) -> dict:
        with np.errstate(all="ignore"):
            return self._advise(state)

    def _advise(self, state: dict, actor=None) -> dict:
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
        hero = str(state["hero"]).upper()
        if hero not in positions:
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
        hole = tuple(parse_cards(str(state["hole"])))
        board = parse_cards(str(state.get("board") or ""))
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
            return {"error": "need two hole cards and 0, 3, 4 or 5 board cards"}
        if len(set(hole) | set(board)) != len(hole) + len(board):
            return {"error": "the same card appears twice in your hand and the board"}
        try:
            stack, stacks = _state_stacks(state.get("stack"), state.get("stacks"), positions, bb)
        except ValueError as e:
            return {"error": str(e)}
        streets = {s: _norm_actions(state.get(s), bb) for s in STREETS}
        if stacks:
            # each seat's own stack; the depth hero plays at is the effective stack against the deepest
            # opponent still in (preflop: who hasn't folded)
            positions = pf.Table(list(positions), positions.ante, positions.straddle, positions.rake, stack, stacks)
            stack = _eff_stack(positions, hero, streets["preflop"], stack)
        positions = pf.at_depth(positions, stack)
        if not board:
            res = self._preflop(hero, hole, streets["preflop"], positions, stack, t0,
                                pool=state.get("opponents") == "pool")
            if locks and "notes" in res:
                res["notes"].append("locks apply after the flop (preflop follows the charts)")
            return res
        saved = self.time_budget, self.prior_budget, self.cfg, self.root_sizes, self.spr_sizes
        if sizes:
            self.root_sizes = {st: {k: v for k, v in d.items() if k != "raise" or "raise" not in sizes}
                               for st, d in self.root_sizes.items() if st not in sizes}
            self.spr_sizes = {st: (m, {k: v for k, v in d.items() if k != "raise" or "raise" not in sizes})
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
        # the Breakdown view (gto/breakdown.py) on the advice's own spot; not for a practice opponent's move
        self._breakdown = bool(state.get("breakdown")) and actor is None
        self._req_sizes = sizes   # (optimize_sizes puts a street's custom sizes in its wide menu)
        # the page's session analyzer in its "flop-library spots only" scope: a flop the library holds, never a solve
        self._lib_only = bool(state.get("library_only")) and actor is None
        if self._lib_only and len(board) != 3:
            return {"skipped": "only flop spots are answered from the library"}
        try:
            res = self._postflop(hero, hole, board, stack, streets, positions, t0,
                                 pool=state.get("opponents") == "pool", actor=actor, custom=custom,
                                 optimize=bool(state.get("optimize_sizes")), locks=locks)
            if self._breakdown and "error" not in res and "breakdown" not in res:
                res["breakdown"] = None   # the betting rules answered (4+ players): no solve to break down
            return res
        finally:
            self.time_budget, self.prior_budget, self.cfg, self.root_sizes, self.spr_sizes = saved
            self._breakdown = self._lib_only = False

    # ------------------------------------------------------------ preflop
    def _preflop(self, hero, hole, preflop, positions, stack, t0, pool=False):
        res = self._preflop_play(hero, hole, preflop, positions, stack, t0, pool)
        if pool and "actions" in res:
            # Against the pool a hand takes its best price, so a chart mix (44 flatting a CO open 30%) plays
            # 100% one way: show the chart's share beside it, as postflop shows the equilibrium's.
            g = {a["action"]: a["freq"] for a in self._preflop_play(hero, hole, preflop, positions, stack, t0)["actions"]}
            for a in res["actions"]:
                a["gto_freq"] = g.get(a["action"], 0.0)
            if any(abs(a["freq"] - a["gto_freq"]) > 0.05 for a in res["actions"]):
                res["notes"].append("exploit mode: against the pool each hand takes its best-paying action, so the "
                                    "chart's mix (GTO %) plays one way")
        return res

    def _preflop_play(self, hero, hole, preflop, positions, stack, t0, pool=False):
        st = replay_hand(positions, stack, {"preflop": preflop}, 1)[0]
        if st.closed:
            return {"error": "preflop betting is already closed (deal the flop)"}
        if st.to_act != hero:
            return {"error": f"it is {st.to_act}'s turn, not {hero}'s"}
        aggr = [a["pos"] for a in preflop if a["action"] in ("raise", "all-in", "bet")]
        if aggr and aggr[-1] in st.allin and aggr[-1] != hero:
            return self._vs_jam(hero, hole, preflop, positions, stack, st, t0, pool)
        full = pf.strategy_for(hero, preflop, positions)
        notes = ["preflop: chart approximation"]
        if "call" in full:
            full = self._priced(hero, preflop, positions, stack, st, full, notes, pool)
        elif pool:
            notes.append("exploit mode: opens follow the chart (the pool 3bets more and flats more, and "
                         "folds less to 3bets, so there's no clear edge in opening wider or tighter)")
        if "raise" in full and not can_raise(st, hero):
            # an all-in for less than a full raise doesn't reopen the betting (or nobody behind has chips
            # left to face a raise): the chart's raises call
            full = dict(full)
            r = full.pop("raise")
            k = "call" if "call" in full or st.cur_bet > st.bets[hero] + 1e-9 else "check"
            full[k] = np.minimum(full.get(k, 0.0) + r, 1.0)
            notes.append("you can't raise here (an all-in for less than a full raise doesn't reopen the betting, "
                         "or nobody left has chips to call one): the chart's raises call")
        idx = COMBO_INDEX[(min(hole), max(hole))]
        strat = {k: float(v[idx]) for k, v in full.items()}
        remaining = stack - st.inv_start[hero] - st.bets[hero]
        to_call = min(st.cur_bet - st.bets[hero], remaining)
        size = min(max(_default_size(st, hero, True), st.cur_bet + st.min_raise), st.bets[hero] + remaining)
        size = round(size * 2) / 2
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
                    full = self._push(hero, preflop, positions, stack, st, size, full, notes, pool)
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

    @staticmethod
    def _push(hero, preflop, positions, stack, st, jam, full, notes, pool=False):
        """A raise that would commit much of the stack is an all-in: hero's range for it is the push/fold
        equilibrium against the players still in (the last raiser's range calls, if there is one); hands
        the chart continued with that don't shove call. Facing a 3bet or more, the chart's raise is a value
        range already and simply goes all-in: a push/fold equilibrium there (both players left given the
        4bettor's premiums, and no option to call) settled on KK jamming 25% at 20bb."""
        if st.n_raises >= 2:
            notes.append(f"a raise here commits too much of a {jam + st.inv_start[hero]:g}bb stack: the chart's raises go all-in")
            return full
        arr = preflop_ranges(preflop, positions, stack, hero if pool else None)
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
        push = push_fold(arr.get(hero, np.ones(N_COMBOS)), call_rng, jam, st.bets[hero], caller_in, pot,
                         max(1, len(opp)), rake=getattr(positions, "rake", None), alt=alt)
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
                     f"range from the push/fold equilibrium against {len(opp)} player{'s' * (len(opp) > 1)}")
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
        # the chart's own spot: its depth (100bb, or a short-stack chart's: pf.spot_depth), no antes
        cs = pf.spot_depth(hero, preflop, positions)
        table = pf.Table(positions, 0.0, getattr(positions, "straddle", False), depth=cs)
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
        if not (more or reprice):
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
        if (more or pool) and r > 0.005:
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
            keep = np.clip(0.5 + (eq_c - breakeven(st, stack) - v_bar + breakeven(std, cs)) / RAISE_BLEND,
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
            scale = min(1.5 if pool else 1.0,
                        (f / fold_need(st, stack)) / max(f_std / fold_need(std, cs), 1e-9))
            bluff_n = bluff_c * scale
            out["raise"] = np.minimum(1.0, value_n + bluff_n)
            msg.append(f"raises re-drawn for {'the player pool' if pool and not more else f'{len(opp)} players in'}: value raises keep the equity over break-even "
                       f"against what continues that the chart's weakest value raise has heads-up ({v_bar:.0%}); bluffs x{scale:.2f} "
                       f"(everyone folds {f:.0%} vs {f_std:.0%}, needing {fold_need(st, stack):.0%} vs "
                       f"{fold_need(std, cs):.0%})")
        raise_n = out.get("raise", raise_c)
        # continuing before the price: the chart's calls and value raises, kept bluffs, new value raises
        base = np.minimum(1.0, np.maximum(call_c + value_c + (raise_n - value_n if more or pool else raise_c - value_c), raise_n))
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
                msg.insert(0, f"price {need:.0%} of the pot{rk} against {', '.join(opp)} (the chart assumes "
                              f"{need_std:.0%} against {', '.join(opp_std)}{deep}): calls re-drawn by equity "
                              f"against the pot odds" + (", against the player pool's ranges" if pool else ""))
        if "raise" in full:
            out["raise"] = np.minimum(raise_n, new)
        out["call"] = new - out.get("raise", 0.0)
        out["fold"] = 1.0 - new
        notes.extend(msg)
        return pf.monotone(out)

    def _vs_jam(self, hero, hole, preflop, positions, stack, st, t0, pool=False):
        """Facing a preflop all-in: call when equity against the jammer's range beats the pot odds.
        With `pool`, a jam over a raise comes from the player pool's raising range (premiums, mostly)."""
        ji = max(i for i, a in enumerate(preflop) if a["action"] in ("raise", "all-in", "bet"))
        jammer = preflop[ji]["pos"]
        pool_except = hero if pool else None
        notes = ["preflop: facing an all-in, call when equity beats the pot odds"]
        rng, more = _jam_range(preflop, ji, positions, stack, pool_except)
        notes += more
        remaining = stack - st.inv_start[hero] - st.bets[hero]
        to_call = min(st.cur_bet - st.bets[hero], remaining)
        pot = st.pot_start + sum(st.bets.values())
        # hero's call only wins what the jammer can match
        win = pot - max(0.0, st.bets[jammer] - (st.bets[hero] + to_call))
        # a called all-in sees the flop, so the pot is raked
        need = to_call / (win + to_call - _rake(positions, win + to_call))
        # everyone who already called the jam is in the pot too: equity against all of them
        after = preflop_ranges(preflop, positions, stack, pool_except)
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
        pf_r = preflop_ranges(streets["preflop"], positions, stack, hero if pool else None)
        ranges = dict(pf_r)
        bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        for p, r in (custom or {}).items():
            if p not in rep[0].active:
                continue   # folded preflop: nothing to override
            if (r * bm).sum() <= 0:
                return {"error": f"ranges: {p}'s range has no combos left on this board"}
            ranges[p] = r
            notes.append(f"{'your' if p == hero else p + chr(39) + 's'} range entering the flop is the custom one "
                         f"({(r * bm).sum():.0f} combos), not the preflop chart's")
        for p in [q for q in getattr(pf_r, "rebuilt", []) if q not in (custom or {})]:
            notes.append(f"{'your' if p == hero else p + chr(39) + 's'} preflop line isn't in the chart, so "
                         f"{'your' if p == hero else 'their'} range is the hands that take its last action")
        # Streets that started with 3+ players can't be solved; narrow ranges there by rule.
        if self._lib_only and len(cur.players_start) > 2:
            return {"skipped": "multiway: not a flop-library spot"}
        mw_streets = [st for st in rep[1:] if len(st.players_start) > 2]
        slow: Dict[str, np.ndarray] = {}
        by_rules = []
        for st in mw_streets:
            before = dict(ranges)
            _narrow_street(ranges, board[:{"flop": 3, "turn": 4, "river": 5}[st.name]], st, stk, hero, hole, slow)
            # 3-way: through the street's 3-way solve where there is one (the one your advice on it came from, or
            # one from the street's start), as far as it has settled; the rest by the rules
            if not (len(st.players_start) == 3 and self._mw_walk(ranges, before, board, st, stk, hero, hole, streets,
                                                                  notes, current=st is cur)):
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
            if self._lib_only:
                return {"skipped": "multiway: not a flop-library spot"}
            if actor:
                move = self._mw_actor_move(board, hero, hole, stk, streets, cur, actor) if len(remaining) == 3 else None
                return move or {"error": "multiway"}
            if optimize:
                notes.append("bet sizes are optimized only heads-up (multiway is a heuristic, not a solve)")
            if locks and any(p in locks for p in live_opps):
                notes.append(f"locks ({', '.join(p for p in live_opps if p in locks)}) apply heads-up only: "
                             f"this multiway pot is {'solved' if len(remaining) == 3 else 'played by the betting rules'} without them")
            if pool:
                notes.append("exploit mode applies heads-up after the flop: this multiway pot is played at equilibrium "
                             "(the pool's leaks only shape the ranges preflop)")
            res = self._multiway_solve(hero, hole, board, ranges, remaining, cur, stk, t0, notes,
                                       line=_mw_line(board, hero, hole, stk, streets, cur.name, len(streets.get(cur.name) or [])))
            if res is not None:
                return res
            return self._multiway(hero, hole, board, ranges, remaining, cur, stk, t0, notes)
        villain = live_opps[0]
        post = cur.order
        oop, ip = sorted([hero, villain], key=post.index)
        hero_p = 0 if hero == oop else 1
        r0, r1 = ranges[oop].copy(), ranges[ip].copy()
        full = (ranges[oop], ranges[ip])   # the ranges entering the flop: they pick a precomputed spot (flop_library)
        lib = self.library if self.library is not None and not pool and not optimize and not (locks or {}).get(villain) else None
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
        if hu_from_start:
            # narrow through earlier streets that were also heads-up between these two
            for st in rep[1:-1]:
                sub_board = board[:{"flop": 3, "turn": 4, "river": 5}[st.name]]
                if st.players_start != remaining:
                    continue   # multiway: narrowed above
                spare += self.prior_budget
                eff = min(stk[oop] - st.inv_start[oop], stk[ip] - st.inv_start[ip])
                cfg = self._cfg_with_real_sizes(st, oop, ip, eff)
                hit = self._from_library(lib, full, sub_board, r0, r1, st, eff, cfg, oop, ip, notes,
                                         smoothing=self.prior_smoothing, board=sub_board, rules=(0, 1)) \
                    if lib is not None and st.name == "flop" else None
                if hit is not None:
                    sol, node, c0, c1 = hit
                    notes.append(sol.note())
                else:
                    if self.prior_model:
                        cfg = dataclasses.replace(cfg, **self.prior_model)
                    sol = self._solve_cached(sub_board, r0, r1, st.pot_start, eff, self.prior_budget, cfg=cfg)
                    if getattr(sol, "expl_at", (None,))[0] != sol.iters:
                        sol.expl_at = (sol.iters, sol.exploitability())
                    expl = sol.expl_at[1]
                    trust = trust_of(expl, st.name)
                    if trust < 1.0:
                        notes.append(f"{st.name}: the solve that narrows ranges is {expl:.1f}% of the pot from "
                                     f"equilibrium ({sol.iters} iterations), so ranges follow it {trust * 100:.0f}% and "
                                     "simple betting rules the rest")
                    node, c0, c1 = self._walk(sol, st.actions, oop, ip, st.pot_start, eff, notes, st.name, self.prior_smoothing,
                                              board=sub_board, rules=(0, 1),
                                              lock=mk_lock(sol), trust=trust)
                r0 = _post_floor(r0, expand(sol.ctx, c0), board, oop, st.name, notes)
                r1 = _post_floor(r1, expand(sol.ctx, c1), board, ip, st.name, notes)
            eff = min(stk[oop] - cur.inv_start[oop], stk[ip] - cur.inv_start[ip])
            cfg = self._cfg_with_real_sizes(cur, oop, ip, eff)
            r0, r1 = self._with_hero(r0, r1, hero_p, hole, notes)
            if optimize and to_call <= 1e-9:
                cfg, opt = self._optimize_sizes(board, r0, r1, cur, eff, cfg, hero_p, notes)
            if lib is not None and cur.name == "flop":
                # a precomputed solve answers when it holds the spot and the line, the villain's line isn't rare in
                # it and the node is one it stores (else the live solve below)
                w2, rl2, n2 = [], [], []
                hit = self._from_library(lib, full, board, r0, r1, cur, eff, cfg, oop, ip, n2,
                                         smoothing=self.prior_smoothing, only=1 - hero_p, widened=w2, board=board,
                                         rare_lines=rl2)
                if hit is not None and not w2 and not rl2 and id(hit[1]) in hit[0].stored:
                    lib_sol, node, n_r0, n_r1 = hit
                    notes.extend(n2)
                    notes.append(lib_sol.note())
        if lib_sol is None and self._lib_only:
            return {"skipped": "not a flop-library spot"}
        if lib_sol is not None:
            sol = lib_sol
        elif hu_from_start:
            # spare_prior: the street advised on also gets what the narrowing solves didn't spend (a flop from the
            # library or from this hand's flop advice costs nothing), so a request never takes longer than cold
            spare = max(0.0, spare - (time.time() - t_prior)) if self.spare_prior else 0.0
            sol = self._solve_cached(board, r0, r1, cur.pot_start, eff, self.time_budget + spare, cfg=cfg)
            # The villain's actions this street are exact unless the solve rarely takes them: then
            # smooth as on earlier streets, so we aren't advised against a sliver of a range.
            walk = lambda: self._walk(sol, cur.actions, oop, ip, cur.pot_start, eff, list(notes), cur.name,
                                      self.prior_smoothing, only=1 - hero_p, widened=[], board=board,
                                      lock=mk_lock(sol), rare_lines=[])
            node, n_r0, n_r1 = self._walk(sol, cur.actions, oop, ip, cur.pot_start, eff, notes, cur.name,
                                          self.prior_smoothing, only=1 - hero_p, widened=widened, board=board,
                                          lock=mk_lock(sol), rare_lines=rare_lines)
            if not rare_lines and self._settle(sol):
                # (far from settled: one more budget, then walk the line again; when the line is rare the
                # solve from the node below gets it instead, so a spot doesn't pay for both)
                node, n_r0, n_r1 = walk()
            if rare_lines and isinstance(node, Decision) and node.player == hero_p:
                # Their line was rare in the solve, so everything below it barely converged: answered from
                # there, a min-reraise on KQT9 read by the rules had AQ jamming for +31.8 bb. Solve again
                # from here, against the range they showed.
                sol = self._solve_cached(board, expand(sol.ctx, n_r0), expand(sol.ctx, n_r1), cur.pot_start, eff,
                                         self.time_budget, bets=(min(cur.bets[oop], eff), min(cur.bets[ip], eff)), to_act=hero_p,
                                         n_raises=cur.n_raises, settle=True)
                node, n_r0, n_r1 = sol.root, sol.r0.copy(), sol.r1.copy()
                notes.append(f"{cur.name}: {', '.join(rare_lines)}'s line is rare in the solve, so solved again from here"
                             + (" against their rule-based range" if widened else ""))
        elif actor:
            return {"error": "the pot became heads-up this street"}   # a solve from the actor's own node
        else:
            # pot became heads-up during this street: solve from the current node
            eff = min(stk[oop] - cur.inv_start[oop], stk[ip] - cur.inv_start[ip])
            r0, r1 = self._with_hero(r0, r1, hero_p, hole, notes)
            sol = self._solve_cached(board, r0, r1, cur.pot_start, eff, self.time_budget,
                                     bets=(min(cur.bets[oop], eff), min(cur.bets[ip], eff)), to_act=hero_p, n_raises=cur.n_raises,
                                     settle=True)
            node, n_r0, n_r1 = sol.root, sol.r0.copy(), sol.r1.copy()
            notes.append(f"{cur.name}: pot became heads-up this street; solved from here")
        if not isinstance(node, Decision):
            return {"error": "no decision pending at this point in the tree", "notes": notes}
        if actor:
            return self._actor_move(sol, node, n_r0, n_r1, actor[1], 1 - hero_p, cur.name)
        if node.player != hero_p:
            notes.append("warning: tree expected the other player to act; reporting that node")
        expl = max(0.0, sol.exploitability())   # how far this street's solve is from equilibrium, % of its pot
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
        else:
            evs = sol.solver.node_action_evs(node, n_r0, n_r1)
            strat = node.avg_strategy0()
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
        w = hero_r / max(hero_r.sum(), 1e-12)
        sf = np.zeros((len(node.names), N_COMBOS)); sf[:, sol.ctx.active] = strat
        range_freq = {name: round(float((sf[a] * w).sum()), 3) for a, name in enumerate(node.names)}
        if self._breakdown:   # the page's Breakdown view (gto/breakdown.py): asked for, on a cached solve
            from . import breakdown as BD
            ev_full = np.zeros(N_COMBOS); ev_full[sol.ctx.active] = (strat * evs).sum(axis=0)
            opt = {**opt, "breakdown": BD.breakdown(board, node.names, hero, hero_r, sf, ev_full, {villain: vill_r},
                                                    node.pot_now, villain_ev=not self.cfg.rake_rate, hole=hole)}
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
                       "exploitability_pct": round(expl, 2)},
            "widened": widened,
            **({"precomputed": lib_sol.info()} if lib_sol is not None else {}),
            **({"locked": {villain: adj}} if adj else {}),
            "notes": notes, "seconds": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------ multiway
    def _multiway_solve(self, hero, hole, board, ranges, remaining, cur: StreetState, stk, t0, notes, line=None):
        """Three players in: a 3-player CFR solve of this street from the current node (gto/multiway_solver.py),
        the later streets bet on by hand buckets (mw_street_config). None when it doesn't
        apply (4+ players, someone all-in, the street not in mw_solve_streets, or a bet bigger than the shortest
        stack can call) or the solve is still over mw_max_expl from equilibrium after its budget (a note says so):
        the betting rules (_multiway) answer instead."""
        from . import multiway_solver as MW
        fallback_notes, notes = notes, list(notes)
        if not self.mw_solve_streets or cur.name not in self.mw_solve_streets or len(remaining) != 3:
            return None
        if set(remaining) & cur.allin:
            return None
        players = [p for p in cur.order if p in remaining]          # postflop acting order
        behind = {p: stk[p] - cur.inv_start[p] for p in players}
        S = min(behind.values())
        bets = [cur.bets[p] for p in players]
        if max(bets) > S + 1e-9 or S <= 1e-9:
            return None
        dead_bets = sum(b for q, b in cur.bets.items() if q not in remaining)
        pot0 = cur.pot_start + dead_bets
        hp = players.index(hero)
        hi = COMBO_INDEX[(min(hole), max(hole))]
        rs = [ranges.get(p, np.ones(N_COMBOS)) for p in players]
        if rs[hp][hi] <= 1e-9:
            notes.append("your hand isn't in your range for this line: its mix is the solve's current (regret-matching) "
                         "strategy for it, against the ranges as they are")
        pending = {i for i, p in enumerate(players) if p not in cur.acted}
        pending.add(hp)
        mcfg = MW.MWConfig(rake_rate=self.cfg.rake_rate, rake_cap=self.cfg.rake_cap,
                           **{**self.mw_street_config.get(cur.name, {}), **(self.mw_config or {})})
        inc = () if any(r[hi] > 0 for r in rs) else (hi,)    # (a hand nobody's range holds still gets solved)
        key = (tuple(board), tuple(_range_key(r) for r in rs), round(pot0, 2), round(S, 2), tuple(round(b, 2) for b in bets),
               hp, cur.n_raises, tuple(sorted(pending)), repr(mcfg), inc)
        sol = self._mw_cache.get(key)
        if sol is None:
            sol = MW.MultiwaySolver(board, rs, pot0, S, bets=bets, to_act=hp, n_raises=cur.n_raises, pending=pending,
                                    cfg=mcfg, include=inc)
            self._mw_cache[key] = sol
        # the budget covers the whole answer: the tree and bucket tables (a flop's take 1-2 s natively, a few in the
        # browser), the iterations, and the gains and EVs read off after them (MW_RESERVE iterations' time)
        # (a solve its budget ran out on isn't run again: a later request for the spot gets the same answer)
        spent = sol.build_seconds + sol.seconds
        left = min(self.time_budget - spent, self.time_budget - (time.time() - t0))
        self._mw_run(sol, left)
        root = sol.root
        if not isinstance(root, MW.Decision) or root.player != hp:
            return None
        gains = sol.br_gains() / sol.root_pot() * 100        # each player's best-response gain, % of the pot
        expl = float(max(gains.mean(), 0.0))
        if expl > self.mw_max_expl:
            fallback_notes.append(f"the 3-way solve was still {expl:.0f}% of the pot from equilibrium after {sol.iters} "
                                  f"iterations, so the betting rules answer instead (Accurate gives it longer)")
            return None
        if line is not None:
            self._mw_lines[line] = (sol, players, expl)   # practice opponents and later streets read it (_mw_walk)
        ci = sol.combo_index(hole)
        strat = root.avg_strategy()[0]
        mine = sol.hand_strategy(ci, hp)
        evs = sol.action_evs()
        acts = []
        shown = _mw_names(root.names, root.amounts, behind[hero], S, root.pot_now)
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
        if self._breakdown:   # hero's categories and the opponents' (no equity view 3-way)
            from . import breakdown as BD
            ev_full = np.zeros(N_COMBOS); ev_full[sol.hands.active] = (strat * evs).sum(axis=0)
            bd = BD.breakdown(board, shown, hero, hero_r, sf, ev_full,
                              {p: sol.expand(sol.reach0[i][0]) for i, p in enumerate(players) if p != hero},
                              root.pot_now, equity=False, hole=hole)
        to_call = max(bets) - bets[hp]
        street_note = {"river": "the river's showdowns with every card removal between the three hands counted exactly",
                       "turn": "the river dealt out after the turn's betting (an equity leaf: no river betting), every river card",
                       "flop": "the turn and river dealt out after the flop's betting (equity leaves: no later betting), every turn"}
        if mcfg.later == "buckets":
            street_note["turn"] = (f"the river bet on too, by {mcfg.buckets} hand buckets a player on each river card "
                                   f"(one card per rank class)")
            street_note["flop"] = (f"the turn and river bet on too, by {mcfg.buckets} hand buckets a player on one turn card "
                                   f"per rank class, the river played on the turn's boards without its card")
        sizes = MW.bet_sizes(mcfg, cur.name)
        notes.append(f"3-way pot solved by three-player CFR ({sol.iters} iterations): "
                     + (f"one bet size ({sizes[0] * 100:.0f}% pot)" if len(sizes) == 1 else
                        f"bets of {', '.join(f'{x * 100:.0f}' for x in sizes)}% pot")
                     + f", one raise and all-in; {street_note[cur.name]}")
        if (cur.acted or cur.n_raises) and not any(n.startswith(f"{cur.name}: 3-way, ranges narrowed") for n in notes):
            notes.append("this street's earlier actions narrow the ranges by the betting rules; the solve starts from here")
        if max(behind.values()) - S > 0.05 * S:
            notes.append(f"stacks differ: everyone plays the shortest stack behind ({S:.1f} bb)")
        notes.append(f"best-response gains (% pot): " + ", ".join(f"{p} {g:.2f}" for p, g in zip(players, gains)))
        out = {"street": cur.name, "hero": hero, "hand": hand_class(hole), "board": cards_str(board),
               "players": len(players), "multiway": True, "solved": True, "opponents": opps,
               "position": f"{hp + 1} of 3 to act",
               "pot_bb": round(root.pot_now, 2), "to_call_bb": round(to_call, 2), "stack_bb": round(S - bets[hp], 2),
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
                    m = aggro(opp_ranges[i], k_agg[i] * RAISE_SHARE, rto - agg_to, pot + hero_add + opp_add[i], rank0, bluff0)
                    p_r = pot + hero_add + rto - cur.bets[opps[i]]
                    rr[i] = (m, opp_ranges[i] * m, hero_to, extra, p_r,
                             calls(opp_ranges[i] * m, mt0, extra, p_r))
            return {"to": agg_to, "add": hero_add, "opp_add": opp_add, "conts": conts, "k": k_agg,
                    "cont_r": [opp_ranges[i] * conts[i] for i in range(n_opp)], "rr": rr}

        if to_call > 1e-9:
            # our raise option keeps the page's default size (the bet plus 75% of the pot after calling,
            # callers' chips included); raises inside the trials go 3x plus the calls. None when an all-in
            # for less than a full raise left the betting closed to us, or nobody has chips to face one.
            agg_to = min(max(round((cur.cur_bet + 0.75 * (pot + call_amt)) * 2) / 2, cur.cur_bet + cur.min_raise),
                         cur.bets[hero] + hero_rem)
            sizes = [(f"raise to {agg_to:g}", agg_to)] if can_raise(cur, hero) else []
        else:
            big, small = (min(round(f * pot * 2) / 2, hero_rem) for f in (0.66, SMALL_BET))
            sizes = ([("bet 33%", small)] if 1e-9 < small < big - 1e-9 else []) + [("bet 66%", big)]
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
            acts.append({"action": name, "freq": 0.0, "ev_bb": round(ev[name], 2), "amount_bb": round(a["to"], 2)})
            notes.append(f"vs {name}: " + ", ".join(f"{opps[i]} continues with {a['k'][i]*100:.0f}%"
                                                    for i in range(n_opp)) + " of their range")
        pick = max(range(len(acts)), key=lambda i: acts[i]["ev_bb"])
        acts[pick]["freq"] = 1.0
        notes.append(f"{trials} trials; the best action leads the next by {d:.2f} ± {se:.2f} bb")
        out = {"street": cur.name, "hero": hero, "hand": hand_class(hole), "board": cards_str(board),
               "players": len(remaining), "multiway": True, "opponents": opps,
               "equity_vs_field": round(eq, 3), "pot_bb": round(pot, 2), "to_call_bb": round(call_amt, 2),
               "actions": acts, "recommended": acts[pick]["action"], "sampled": acts[pick]["action"],
               "notes": notes + [f"{len(remaining)} players in the pot: the solver is heads-up only, this is a Monte Carlo heuristic"],
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
                res = self._advise(st, actor=(seat, hole))
            if res is None or "error" in res:
                # from the seat's own view: a GTO player, so not best-responding to the locks the player practising
                # set on their opponents (which can name this seat: "you can't lock your own seat" stalled practice)
                res = self._advise({**{k: v for k, v in st.items() if k != "locks"}, "hero": seat,
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
        for k in range(len(acts), -1, -1):
            hit = self._mw_lines.get(_mw_line(bk, hero, hole, stk, streets, st.name, k))
            if hit is not None:
                break
        src = "the solve your advice came from"
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
        S = min(stk[p] - st.inv_start[p] for p in players)
        if S <= 1e-9:
            return None
        rs = [ranges.get(p, np.ones(N_COMBOS)) for p in players]
        mcfg = MW.MWConfig(rake_rate=self.cfg.rake_rate, rake_cap=self.cfg.rake_cap,
                           **{**self.mw_street_config.get(st.name, {}), **(self.mw_config or {})})
        key = (tuple(board), tuple(_range_key(r) for r in rs), round(st.pot_start, 2), round(S, 2), "start", repr(mcfg))
        sol = self._mw_cache.get(key)
        if sol is None:
            sol = self._mw_cache[key] = MW.MultiwaySolver(board, rs, st.pot_start, S, cfg=mcfg)
        self._mw_run(sol, self.prior_budget - sol.build_seconds - sol.seconds)
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
        S = min(stk[p] - cur.inv_start[p] for p in players)
        names = _mw_names(node.names, node.amounts, stk[seat] - cur.inv_start[seat], S, node.pot_now)
        return {"street": cur.name, "solved": True,
                "actions": [{"action": nm, "freq": round(float(mix[a]), 3),
                             "amount_bb": None if node.amounts[a] is None else round(float(node.amounts[a]), 2)}
                            for a, nm in enumerate(names)]}

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
        i = COMBO_INDEX[(min(hole), max(hole))]
        r = r0 if hero_p == 0 else r1
        if r[i] > 1e-9:
            return r0, r1
        r = r.copy(); r[i] = 1e-3
        notes.append("your hand isn't in your range for this line, so it was solved as a near-zero-weight combo")
        return (r, r1) if hero_p == 0 else (r0, r)

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

    def _solve_cached(self, board, r0, r1, pot, stack, budget, bets=(0.0, 0.0), to_act=0, n_raises=0, cfg=None,
                      settle=False):
        cfg = cfg or self.cfg
        spot = (tuple(board), round(pot, 2), round(stack, 2), tuple(round(b, 2) for b in bets), to_act, n_raises,
                cfg.sizes_key()) + (((cfg.river_play, cfg.turn_classes, cfg.river_buckets, cfg.turn_buckets),)
                                    if len(board) == 3 else ())
        # (a flop solve's river_play and turn_classes are part of the spot: a Fast flop solve, river checked down,
        # seeded an Accurate one at its lower c-bet frequencies and did worse in the same time than a cold start)
        key = (_range_key(r0), _range_key(r1)) + spot
        sol = self._cache.get(key)
        if sol is None:
            # Same spot solved by warm_from (ranges can differ a little: they come from its own earlier
            # streets): start from it and count its time against the budget, as extend() does below.
            warm = self.warm_from._by_spot.get(spot) if self.warm_from is not None else None
            # (the budget covers the tree's build and the exploitability read after the iterations: the browser's
            # Accurate flop answered 4.5 s past its 60 s, a two-tone one 9 s)
            sol = solve(board, r0, r1, pot, stack, bets=bets, to_act=to_act, n_raises=n_raises, cfg=cfg,
                        time_budget=max(budget - warm.seconds, 0.2 * budget) if warm else budget,
                        should_stop=self.should_stop, warm=warm, reserve=HU_RESERVE)
            if warm is not None:
                sol.seconds += warm.seconds
            self._cache[key] = sol
            self._by_spot[spot] = sol
        elif sol.seconds < budget * 0.8 and sol.iters < 1000:
            sol.extend(budget - sol.seconds, should_stop=self.should_stop, reserve=HU_RESERVE)        # keep going from the stored regrets instead of restarting
        if settle:
            self._settle(sol, budget)
        return sol

    def _settle(self, sol: Solution, budget: Optional[float] = None) -> bool:
        """The street being advised on, far from settled after its budget (SETTLE_EXPL): one more budget,
        once. Limped pots, turn probes and short-stack check-raises stopped at 10 iterations 27-55% of the
        pot from settled, showing confident mixes (an all-in 75% at +15 beside a call at +22). Whether it ran."""
        if getattr(sol, "settled_once", False) or sol.iters >= 1000 or sol.exploitability() < SETTLE_EXPL:
            return False
        sol.settled_once = True
        sol.extend(self.time_budget if budget is None else budget, should_stop=self.should_stop, reserve=HU_RESERVE)
        return True

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
            for kind, v in shallow.items():
                (cfg.bet_sizes if kind == "bet" else cfg.raise_sizes)[st.name] = list(v)
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
        return cfg

    def _walk(self, sol: Solution, acts: List[dict], oop, ip, pot, stack, notes, street, smoothing=0.0, only=None,
              widened=None, board=None, rules=None, lock=None, rare_lines=None, visited=None, trust=1.0):
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
        `visited` (a list) gets every node an action was taken at.
        `trust` (with `rules` and `board`): the weight the solve's strategy gets on `rules`' other actions,
        the rest going to the rules' range for the action (softened, as multiway), for a solve that hasn't
        settled (trust_of)."""
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
                                              bluff=bluff, others=[other], sc=sc)
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
                    s = as_locked(_soften(full, m)[sol.ctx.active])
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
                        full, _, m = rule_mask()
                        s = trust * s + (1.0 - trust) * as_locked(_soften(full, m)[sol.ctx.active])
                if smoothing > 0 and only is None:
                    s = (1.0 - smoothing) * s + smoothing / len(node.names)
            if p == 0:
                r0 = r0 * s
            else:
                r1 = r1 * s
            if node.names[k] != name:
                notes.append(f"{street}: mapped '{name}' to tree action '{node.names[k]}'")
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

    def _tree_action_name(self, node: Decision, a: dict, p: int, bets, pot, stack: float = float("inf")) -> str:
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


def strategy_grid(strats: List[np.ndarray], names: List[str], weights: np.ndarray) -> dict:
    """13x13 hand-class grid: for each class, the range weight and per-action frequency."""
    cls_idx: Dict[str, List[int]] = {}
    for i, cb in enumerate(ALL_COMBOS):
        cls_idx.setdefault(hand_class(cb), []).append(i)
    out = {}
    for cls, idxs in cls_idx.items():
        idx = np.array(idxs)
        w = weights[idx]
        tot = float(w.sum())
        if tot <= 1e-12:
            out[cls] = {"w": 0.0, "f": [0.0] * len(names)}
            continue
        out[cls] = {"w": round(tot / len(idxs), 3),
                    "f": [round(float((s[idx] * w).sum() / tot), 3) for s in strats]}
    return {"actions": names, "cells": out}


def state_board(board) -> str:
    return cards_str(board)
