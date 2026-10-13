"""Chart-based preflop strategy for 6-max, 100bb, no limps.

These are hand-written approximations of solver outputs. Every chart is a
weight vector over 1326 combos; fold = 1 - sum(other actions).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from .ranges import parse_range, N_COMBOS
from .cards import COMBO_INDEX

POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]          # 6-max default
POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}
POSTFLOP_ORDER = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]

# seat names in preflop acting order for each table size
SEATS = {
    2: ["SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    9: ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
}


class Table(list):
    """Seats in seating order (SEATS), plus the game's ante (bb per player, dead money) and whether the
    first seat straddles 2bb (it then acts last preflop and plays the big blind's charts). `rake`: the
    game's (rate, cap in bb or None), taken from pots that see a flop; None = no rake."""
    def __init__(self, seats, ante: float = 0.0, straddle: bool = False, rake=None):
        super().__init__(seats)
        self.ante = float(ante or 0.0)
        self.straddle = bool(straddle) and len(seats) >= 3
        self.rake = tuple(rake) if rake and rake[0] > 0 else None

    def preflop_order(self) -> List[str]:
        return list(self[1:]) + [self[0]] if self.straddle else list(self)

    def posted(self) -> Dict[str, float]:
        out = {"SB": 0.5, "BB": 1.0} if "SB" in self else {"BB": 1.0}
        if self.straddle:
            out[self[0]] = 2.0
        return out


def postflop_order(positions: List[str]) -> List[str]:
    """First to act first. Heads-up the button (SB) acts last postflop."""
    if len(positions) == 2:
        return ["BB", "SB"]
    return ["SB", "BB"] + [p for p in positions if p not in ("SB", "BB")]


def chart_key(pos: str, positions: List[str]) -> str:
    """Map a seat to the chart it plays. Seats with the same number of players
    behind them play the same chart; 7+ handed early seats use the tighter EP chart."""
    n = len(positions)
    if n == 2:
        return {"SB": "HU_SB", "BB": "HU_BB"}[pos]
    if getattr(positions, "straddle", False):
        # straddled: the straddler plays the big blind, the big blind the small blind, the small blind a
        # middle seat; the rest by how many act behind them
        if pos == positions[0]:
            return "BB"
        if pos == "BB":
            return "SB"
        if pos == "SB":
            return "CO"
        behind = len(positions) - 1 - positions.preflop_order().index(pos)
        return {2: "BTN", 3: "CO", 4: "HJ", 5: "UTG"}.get(behind, "EP")
    if pos in ("UTG", "UTG1") and n >= 7:
        return "EP"
    if pos in ("UTG2", "LJ"):
        return "UTG"
    return pos

R = parse_range

# ---------- raise first in ----------
RFI: Dict[str, np.ndarray] = {
    # 6-max UTG (~17%, published solver charts put it at 15-18%)
    "UTG": R("55+, 44:0.5, 33:0.3, 22:0.3, A2s+, ATo+, KTs+, K9s:0.5, KQo, KJo:0.5, QTs+, QJo:0.3, JTs, J9s:0.3, T9s, 98s:0.5, 87s:0.3, 76s:0.3, 65s:0.3"),
    "HJ": R("22+, A2s+, ATo+, A9o:0.3, KTs+, K9s:0.5, KJo+, KTo:0.3, QTs+, QJo, JTs, J9s:0.5, T9s, 98s, 87s:0.5, 76s:0.5"),
    "CO": R("22+, A2s+, A8o+, A7o:0.5, A5o:0.3, K8s+, K7s:0.5, K6s:0.5, K5s:0.5, KTo+, K9o:0.3, Q9s+, Q8s:0.5, QTo+, J9s+, J8s:0.5, JTo, T9s, T8s, 98s, 97s:0.5, 87s, 76s, 65s, 54s:0.5"),
    "BTN": R("22+, A2s+, A5o+, A4o:0.5, A3o:0.5, A2o:0.5, K2s+, K7o+, K6o:0.5, K5o:0.5, Q4s+, Q3s:0.5, Q2s:0.5, Q9o+, Q8o:0.5, J6s+, J9o+, J8o:0.5, T6s+, T9o, T8o:0.5, 96s+, 98o, 86s+, 87o:0.5, 75s+, 64s+, 54s, 53s:0.5, 43s:0.5"),
    "SB": R("22+, A2s+, A2o+, K2s+, K5o+, Q2s+, Q8o+, J4s+, J8o+, T6s+, T8o+, 96s+, 98o, 85s+, 75s+, 64s+, 54s, 53s, 43s"),
    # 7+ handed early position (~13%)
    "EP": R("77+, 66:0.5, 55:0.3, ATs+, A5s:0.3, A4s:0.3, KTs+, QTs+, JTs, T9s:0.5, AJo+, ATo:0.3, KQo, KJo:0.3"),
    # heads-up button/SB open (~85%)
    "HU_SB": R("22+, A2s+, A2o+, K2s+, K2o+, Q2s+, Q2o+, J2s+, J4o+, J3o:0.5, J2o:0.5, T2s+, T5o+, T4o:0.5, 92s+, 96o+, 95o:0.5, 82s+, 86o+, 85o:0.5, 72s+, 76o, 75o:0.5, 62s+, 65o:0.5, 52s+, 42s+, 32s:0.5"),
}
# heads-up big blind facing the button open
HU_BB_VS_SB = {
    "3bet": R("77+, 66:0.5, A8s+, A5s-A2s:0.5, ATo+, A9o:0.5, K9s+, KJo+, QTs+, JTs, T9s:0.5"),
    "call": R("22-66, A7s-A2s, A9o-A2o, K8s-K2s, KTo-K3o, Q5s+, Q4s:0.5, QTo-Q6o, J6s+, J5s:0.5, JTo-J7o, T6s+, T9o-T7o, 96s+, 98o, 97o, 85s+, 87o, 86o:0.5, 74s+, 76o, 75o:0.5, 64s+, 65o, 53s+, 43s, 42s:0.5"),
}

# ---------- facing a single raise ----------
# key: (hero_pos, opener_pos) -> {"3bet": vec, "call": vec}
def _vs_rfi_ip(early: bool) -> Dict[str, np.ndarray]:
    if early:  # HJ/CO facing UTG/HJ
        # Mostly 3bet or fold, matched to published solver frequencies (100bb cash, 2.5bb open): HJ
        # against a UTG ("LJ") open folds 91.1%, calls 1.5% and 3bets 7.4% of hands, AQo 3betting 55%
        # and folding the rest (Poker Academy, NL200 GG); CO against UTG calls 1.4-3.2% and 3bets 8-9%
        # (GTO Wizard NL50, by rake structure, via thepokeragent.com). Here 90/2.5/7.2. The earlier
        # chart called 8.3% and 3bet 3.4% (88/8/3).
        return {
            "3bet": R("QQ+, JJ:0.7, TT:0.4, 99:0.4, 88:0.2, AKs, AKo, AQs, AJs:0.7, ATs:0.6, AQo:0.55, AJo:0.4, A5s, A4s, A3s:0.7, A2s:0.3, KQs, KJs:0.4, KTs:0.3, K9s:0.2, KQo:0.4, QJs:0.4, JTs:0.3, T9s:0.3, 98s:0.3, 87s:0.2, 76s:0.2"),
            "call": R("JJ:0.3, TT:0.6, 99:0.6, 88:0.6, 77:0.5, 66:0.4, 55:0.3, 44:0.2, 33:0.2, 22:0.2, AJs:0.3, ATs:0.4, KJs:0.4, QJs:0.4, JTs:0.4, T9s:0.3"),
        }
    # BTN facing UTG/HJ, CO facing HJ: 85.7/9.1/5.2 fold/call/3bet. Published: BTN vs UTG 84.3/8.2/7.5
    # (GTO Wizard, NL500 6-max 100bb, "Playing calls from the button"; open size not stated); BeyondGTO
    # (NL500, 2.5bb open, unnamed solver) CO vs HJ ~87/5/8, BTN vs HJ ~84/8/8, BTN vs LJ 14% defend.
    # All within 5 points: left.
    return {  # BTN facing UTG/HJ/CO, or CO facing HJ-ish
        "3bet": R("JJ+, TT:0.4, AKs, AKo, AQs, AQo:0.4, AJs:0.5, A5s, A4s, A3s:0.5, KQs:0.5, KJs:0.3, 76s:0.3, 65s:0.3"),
        "call": R("22-99, TT:0.6, AJs:0.5, ATs, A9s:0.5, AQo:0.6, AJo:0.5, KQs:0.5, KJs:0.7, KTs, KQo:0.5, QJs, QTs, JTs, J9s:0.5, T9s, 98s, 87s, 76s:0.7, 65s:0.7, 54s:0.5"),
    }

VS_RFI: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
for hero in ["HJ", "CO", "BTN"]:
    for opener in POSITIONS[:POS_INDEX[hero]]:
        early = hero in ("HJ", "CO") and opener in ("UTG", "HJ") and not (hero == "CO" and opener == "HJ")
        VS_RFI[(hero, opener)] = _vs_rfi_ip(early)
# the button against a cutoff open: 3bet or fold, matched to published solver frequencies (2.5bb
# open, 3bet to 7.5bb, 100bb cash): fold 84.5%, call 3.0%, 3bet 12.5% of hands, with AJo 95%, KQo 84%,
# KJs 86%, T9s 87%, 65s 85% and A8s-A3s 77-99% 3bets, TT 3betting 68% and calling 32%, ATs calling 74%
# and AJs 54% (Poker Academy, NL200 GG); GTO Wizard has BTN against a 2.3bb CO open calling about 5%
# and 3betting about 12% (NL500). Here 84/3/13. The earlier chart called 9.4% and 3bet 8.5% (82/9/9).
VS_RFI[("BTN", "CO")] = {
    "3bet": R("QQ+, JJ:0.9, TT:0.7, 99:0.5, 88:0.3, 77:0.2, AKs, AQs, AJs:0.45, ATs:0.25, A9s:0.7, A8s-A3s:0.9, A2s:0.7, AKo, AQo, AJo:0.95, ATo:0.4, KQs, KJs:0.85, KTs:0.6, K9s:0.4, KQo:0.85, KJo:0.4, QJs:0.7, QTs:0.5, JTs:0.6, T9s:0.85, 98s:0.6, 87s:0.6, 76s:0.7, 65s:0.85, 54s:0.6"),
    "call": R("JJ:0.1, TT:0.3, 99:0.5, 88:0.6, 77:0.6, 66:0.5, 55:0.4, 44:0.3, 33:0.2, 22:0.2, AJs:0.55, ATs:0.75, A9s:0.3, KJs:0.15, KTs:0.3, QJs:0.3, QTs:0.4, JTs:0.4, T9s:0.15, 98s:0.3, 87s:0.3"),
}
# SB: mostly 3bet or fold, small calling range. Against an early open the flat is low frequency
# (~2-3% of hands) but not only middling hands: published solver summaries have the SB "sometimes"
# calling JJ and TT against early-position opens and its 3bets there mostly value (Preflop Wizard,
# poker-small-blind-strategy), and a flat without its traps is face-up once the big blind squeezes.
# The old chart 3bet every TT+, AJs+ and KQs, so the flat was 66-99 plus JTs-type hands and the
# continue against a BB squeeze (the playability top-up, see CALLER_CONTINUE) was 13 combos of
# 77-99: on an A-K-x flop the squeezer's QQ won the pot every time. Now JJ/TT/AQs/AJs/KQs flat
# a share, and that continue holds them too.
# Shares of hands (fold/call/3bet): vs UTG/HJ 91.7/2.7/5.5, vs CO/BTN 80.4/6.4/13.2. Published, all
# within 5 points: SB vs BTN 82/5/13 (BeyondGTO, NL500, 2.5bb open); SB vs CO 86/5/9 (BeyondGTO, an
# unnamed solver whose BTN vs CO 81/9/10 is 6 points off Poker Academy's and GTO Wizard's calls),
# 3bet 10-13% (Preflop Wizard), blinds vs late opens 12-16% (GTO Gecko); SB vs HJ 13% and vs LJ 11%
# defend (BeyondGTO). Left.
for opener in ["UTG", "HJ", "CO", "BTN"]:
    if opener in ("UTG", "HJ"):
        VS_RFI[("SB", opener)] = {
            "3bet": R("QQ+, JJ:0.75, TT:0.5, 99:0.4, AKs, AQs:0.7, AJs:0.6, ATs:0.5, A5s:0.7, A4s:0.5, AQo+, KQs:0.6, KJs:0.5, QJs:0.3"),
            "call": R("JJ:0.25, TT:0.5, 99:0.6, 77-88, 66:0.5, AQs:0.3, AJs:0.4, ATs:0.5, KQs:0.4, KJs:0.5, QJs:0.5, JTs:0.7"),
        }
    else:
        VS_RFI[("SB", opener)] = {
            "3bet": R("88+, 77:0.5, ATs+, A9s:0.5, A5s-A2s, AJo+, ATo:0.5, KTs+, K9s:0.5, KQo, KJo:0.5, QTs+, JTs, T9s, 98s:0.5, 87s:0.5, 76s:0.5"),
            "call": R("77:0.5, 22-66, A9s:0.5, A8s-A6s, ATo:0.5, K9s:0.5, K8s, Q9s, J9s, T8s, 98s:0.5, 87s:0.5, 76s:0.5, 65s, 54s"),
        }
# BB
VS_RFI[("BB", "UTG")] = {
    "3bet": R("QQ+, JJ:0.6, AKs, AKo, AQs:0.7, AQo:0.25, AJs:0.3, A5s, A4s, A3s:0.4, A2s:0.3, KQs:0.5, KJs:0.2, QJs:0.2, T9s:0.2, 76s:0.3, 65s:0.4, 54s:0.4"),
    "call": R("22-TT, JJ:0.6, AQs:0.5, AJs-A6s, A3s, A2s, AQo, AJo, ATo, A9o:0.5, K8s+, K7s-K2s:0.7, KQo, KJo, KTo:0.6, Q8s+, Q7s-Q6s:0.7, QJo, QTo:0.5, J8s+, J7s:0.7, JTo:0.5, T7s+, T6s:0.5, 96s+, 95s:0.5, 85s+, 84s:0.5, 74s+, 73s:0.3, 63s+, 53s+, 52s:0.3, 43s:0.6, 42s:0.3, A8o-A7o:0.5, K9o:0.5, Q9o:0.4, J9o:0.4, T9o:0.5, 98o:0.3, Q5s-Q2s:0.5, J6s-J5s:0.5"),
}
VS_RFI[("BB", "CO")] = {
    "3bet": R("JJ+, TT:0.7, 99:0.2, AKs, AKo, AQs, AQo:0.7, AJs:0.7, ATs:0.4, A9s:0.2, A5s, A4s, A3s:0.8, A2s:0.6, KQs:0.8, KJs:0.5, KTs:0.3, K9s:0.3, QJs:0.4, QTs:0.2, JTs:0.4, J9s:0.2, T9s:0.4, 98s:0.3, 87s:0.3, 76s:0.5, 65s:0.5, 54s:0.5, AJo:0.2, KQo:0.4"),
    "call": R("22-99, TT:0.5, AJs:0.5, ATs-A6s, A5s:0, A2s, AQo:0.5, AJo, ATo, A9o, A8o:0.7, A7o-A2o:0.5, K2s+, KQo:0.7, KJo, KTo, K9o, K8o:0.5, Q2s+, QJo, QTo, Q9o, Q8o:0.5, J5s+, JTo, J9o, J8o:0.5, T6s+, T9o, T8o, 95s+, 98o, 97o:0.3, 84s+, 87o:0.6, 74s+, 63s+, 53s+, 43s, 42s:0.5"),
}
# the big blind against a HJ open (it used the UTG chart): 3bets between the UTG and CO charts', and
# calls halfway between theirs
VS_RFI[("BB", "HJ")] = {
    "3bet": R("QQ+, JJ:0.7, TT:0.3, AKs, AKo, AQs:0.8, AQo:0.4, AJs:0.5, ATs:0.3, A5s, A4s, A3s:0.6, A2s:0.4, KQs:0.6, KJs:0.4, KTs:0.2, K9s:0.2, QJs:0.3, JTs:0.2, T9s:0.3, 98s:0.2, 87s:0.3, 76s:0.4, 65s:0.5, 54s:0.5, KQo:0.15, AJo:0.1"),
    "call": 0.5 * (VS_RFI[("BB", "UTG")]["call"] + VS_RFI[("BB", "CO")]["call"]),
}
VS_RFI[("BB", "BTN")] = {
    "3bet": R("TT+, 99:0.8, AJs+, ATs:0.8, A9s:0.6, A8s:0.3, AQo+, AJo:0.8, ATo:0.2, A5s-A2s, KJs+, KTs:0.8, K9s:0.5, K5s-K2s:0.2, KQo:0.8, KJo:0.2, QJs:0.8, QTs:0.3, Q9s:0.2, JTs:0.6, T9s:0.6, 98s:0.3, 87s:0.3, 76s:0.5, 65s:0.5, 54s:0.5"),
    "call": R("22-88, 99:0.5, ATs:0.5, A9s-A6s, AJo:0.5, ATo-A2o, K2s+, KQo:0.5, KJo-K7o, K6o:0.5, Q2s+, QJo-Q8o, Q7o:0.5, J2s+, JTo-J8o, J7o:0.5, T3s+, T9o-T7o, 95s+, 98o, 97o, 84s+, 87o, 86o:0.5, 74s+, 76o:0.5, 63s+, 65o:0.5, 53s+, 43s, 42s:0.5, 32s:0.5, K6o-K4o, Q7o, Q6o-Q5o:0.5, J7o, J6o:0.5, T6o:0.5, 96o:0.5, 86o, 85o:0.3, 75o:0.5, 64o:0.3, 54o:0.3, 52s, 42s, 32s"),
}
VS_RFI[("BB", "SB")] = {
    "3bet": R("88+, 77:0.3, A9s+, A8s:0.6, A7s-A6s:0.3, ATo+, A9o:0.2, A5s-A2s, K9s+, K8s:0.6, KQo, KJo, KTo:0.3, QTs+, Q9s:0.7, JTs, J9s:0.7, T9s, T8s:0.4, 98s:0.8, 87s:0.8, 76s:0.6, 65s:0.6, 54s:0.4"),
    "call": R("22-77, 88:0.5, A9s:0.5, A8s-A6s, ATo:0.5, A9o-A2o, K9s:0.5, K2s+, KJo:0.5, KTo-K5o, Q9s+, Q8s-Q2s, QJo-Q7o, J9s+, J8s-J2s, JTo-J7o, T2s+, T9o-T6o, 92s+, 98o-96o, 82s+, 87o-85o, 72s+, 76o-74o, 62s+, 65o, 64o, 52s+, 54o, 53o, 42s+, 43o, 32s"),
}

# The big blind's 3bets against the button and the small blind, matched to published solver
# frequencies (100bb cash): against a 2.5bb BTN open it 3bets 11.5% of hands to 11bb (Poker Academy,
# NL200 GG; 10.6% at NL50, and 11.5% to 12bb in gtopreflop.com's raked solve), against a 3bb SB open
# 16.2-16.4% (gtopreflop.com to 10bb, Poker Academy to 7.9bb). The charts 3bet 8.5% and 11.7%; now
# 11.5% and 16.4%, the new 3bets taken from calls so the defence totals stay (65% and 72%). Those
# raked solves defend less in all (BTN 40-41%, SB 43-52%; vs UTG 21% where the chart defends 33%), but
# the big blind's flats are where rake bites hardest (no flop, no drop) and the solver here plays
# without rake, so the totals aren't matched (a solver-based guide puts BB vs BTN at 52-58% with
# NL50-NL200 rake, 3bet 11-13%, and 3-5 points wider at higher stakes: vip-grinders.com).
# Against UTG, HJ and CO the big blind 3bet 3.6%, 3.6% (HJ played the UTG chart) and 5.6%, under every
# published number: gtopreflop.com's raked solve (8-max 100bb, 2.5bb opens, 5% rake capped at 4bb; it
# says 6-max is within a point; https://gtopreflop.com/posts/big-blind-defense) 3bets 4.5% against
# the LJ (the 6-max UTG seat), 5.9% against the HJ and 7.7% against the CO; Poker Academy NL200 4.5%
# against the LJ; pokalab.com (NL50 rake) 5.1% LJ and 6.7% HJ; BeyondGTO (NL500) 6%, 8%, 10%. Now
# 4.8%, 6.2% (a chart of its own, calls halfway between the UTG and CO charts') and 8.0%, the new 3bets
# (more JJ-TT, AQ, suited broadways and wheel aces, a few suited connectors) taken from calls so the
# defence totals stay (33%, 39%, 45%).
# Squeezes (the advisor re-draws the chart for the callers, see Advisor._priced): BB over a BTN open
# and an SB call 30/60/10 against Poker Academy's 71.2/16.1/12.7 (NL200, to 12.2bb; the calls are
# rake); over CO + BTN 7.3% (was 4.9%) and UTG + a caller 4.5% (was 3.4%). The only cash squeeze solve
# found, GTO Wizard's NL500 at 40bb (CO min-open, BTN call: 9.8% against 10.3% heads-up,
# https://blog.gtowizard.com/how-to-construct-a-squeezing-range/), puts the squeeze at 0.95x the
# heads-up 3bet, which the charts now give (7.3% against 8.0%). GTO Wizard's 13% and 7.9% over CO/UTG
# + BTN + SB ("Responding to BB squeezes") are MTT solves with 2.3bb opens and two callers: not matched.
# SB and BTN squeezes follow their 3bet charts the same way (SB over CO + BTN 12.2%, over UTG + HJ
# 5.2%; BTN over UTG + HJ 4.9%); no cash numbers for those were found.
for _o in ("UTG", "HJ", "CO", "BTN", "SB"):
    VS_RFI[("BB", _o)]["call"] = np.minimum(VS_RFI[("BB", _o)]["call"], 1.0 - VS_RFI[("BB", _o)]["3bet"])


def _norm(key: str) -> str:
    return {"EP": "UTG", "HU_SB": "SB", "HU_BB": "BB"}.get(key, key)


def vs_rfi_chart(hero_key: str, opener_key: str) -> Dict[str, np.ndarray]:
    if hero_key == "HU_BB":
        return HU_BB_VS_SB
    if opener_key == "EP":
        return {"BB": BB_VS_EP, "SB": SB_VS_EP}.get(hero_key, VS_EP)
    h, o = _norm(hero_key), _norm(opener_key)
    chart = VS_RFI.get((h, o))
    if chart is not None:
        return chart
    if h == "SB":
        return VS_RFI[("SB", "UTG")]
    if h == "BB":
        return VS_RFI[("BB", "UTG")]
    return _vs_rfi_ip(True)   # early seat facing an even earlier open


# ---------- opener facing a 3bet ----------
def vs_3bet(opener: str, threebettor: str) -> Dict[str, np.ndarray]:
    """The opener facing a 3bet. Out of position postflop (a later seat's 3bet, or the small blind
    against the big blind's) it plays VS_3BET_OOP; against a blind's 3bet it has position and plays
    the _vs_3bet base with calls widened."""
    if opener == "HU_SB":
        return HU_SB_VS_3BET
    o = _norm(opener)
    if o in ("UTG", "HJ", "CO", "BTN") and _norm(threebettor) in ("SB", "BB"):
        if o == "BTN":
            return dict(BTN_VS_BB_3BET if _norm(threebettor) == "BB" else BTN_VS_SB_3BET)
        return _ip_vs_blind_3bet(o)
    if o in VS_3BET_OOP:
        return dict(VS_3BET_OOP[o])
    return _vs_3bet(o)


# The opener out of position facing a 3bet, matched to published solver frequencies: CO against a
# BTN 3bet (2.5bb open, 7.5bb 3bet, 100bb cash) folds 63.7%, calls 17.2% and 4bets 19.1% of its opening
# range (Poker Academy); openers respond in similar proportions from every seat (GTO Wizard), UTG
# 4betting about half its continues against a BTN 3bet (888poker); the small blind against the big
# blind's 3bet folds about half and 4bets a third of its continues (888poker). Here: UTG 61/18/21, HJ
# 63/17/19, CO 63/18/19, SB 53/30/16 (fold/call/4bet). The earlier charts called about twice that and
# 4bet half as often (CO 58/33/9).
_UTG_VS_3BET_OOP = {
    "4bet": R("AA, KK, QQ:0.8, AKs, AKo, JJ:0.2, AQs:0.3, A5s, A4s:0.6, A3s:0.2, KQs:0.2, KJs:0.2, AJo:0.1, KQo:0.15"),
    "call": R("QQ:0.2, JJ:0.8, TT, 99:0.9, 88:0.5, 77:0.25, AQs:0.7, AJs:0.7, ATs:0.3, KQs:0.6, KJs:0.4, QJs:0.3, JTs:0.3, AQo:0.35"),
}
VS_3BET_OOP = {
    "UTG": _UTG_VS_3BET_OOP,
    "HJ": {   # opens wider than UTG: a few more calls and 4bet bluffs
        "4bet": R("AA, KK, QQ:0.8, AKs, AKo, JJ:0.2, AQs:0.3, A5s, A4s:0.7, A3s:0.3, KQs:0.2, KJs:0.3, AJo:0.15, KQo:0.2"),
        "call": R("QQ:0.2, JJ:0.8, TT, 99:0.9, 88:0.6, 77:0.4, 66:0.2, AQs:0.7, AJs:0.8, ATs:0.4, KQs:0.7, KJs:0.4, QJs:0.4, JTs:0.4, T9s:0.2, AQo:0.35"),
    },
    "CO": {
        "4bet": R("AA, KK, QQ, AKs, AKo, JJ:0.3, AQs:0.4, A5s, A4s, A3s, A2s:0.5, KJs:0.4, KTs:0.4, K9s:0.4, QJs:0.2, AJo:0.3, KQo:0.4, ATo:0.2"),
        "call": R("JJ:0.7, TT, 99, 88:0.9, 77:0.7, 66:0.5, 55:0.3, AQs:0.6, AJs, ATs:0.7, A9s:0.3, KQs, KJs:0.6, QJs:0.8, JTs:0.8, T9s:0.6, 98s:0.4, 87s:0.3, AQo:0.5"),
    },
    "SB": {
        "4bet": R("JJ+, TT:0.5, AKs, AKo, AQs:0.6, AQo:0.5, A5s, A4s, A3s, A2s, KQs:0.4, KJs:0.4, KTs:0.4, K9s:0.5, K8s:0.5, K7s:0.5, K6s:0.4, Q9s:0.4, Q8s:0.3, J9s:0.3, AJo:0.4, ATo:0.4, KQo:0.5, KJo:0.4, QJo:0.3"),
        "call": R("TT:0.5, 99, 88, 77, 66, 55, 44, 33:0.8, 22:0.8, AQs:0.4, AJs, ATs, A9s, A8s, A7s, A6s, AQo:0.5, AJo:0.6, ATo:0.6, A9o:0.5, A8o:0.3, KQs:0.6, KJs:0.6, KTs:0.6, K9s:0.5, K8s:0.5, K7s:0.3, KQo:0.5, KJo:0.6, KTo:0.4, QJs, QTs, Q9s:0.6, Q8s:0.4, QJo:0.5, QTo:0.3, JTs, J9s:0.7, J8s:0.4, JTo:0.4, T9s, T8s:0.7, 98s, 97s:0.5, 87s, 86s:0.4, 76s, 75s:0.3, 65s, 54s, 43s:0.3"),
    },
}


IP_VS_3BET_EXTRA = R("88:0.3, 77:0.4, 66:0.4, 55:0.3, A9s:0.4, KTs:0.4, QTs:0.4, J9s:0.3, 98s:0.4, 87s:0.3")


def _vs_3bet(opener: str) -> Dict[str, np.ndarray]:
    if opener in ("UTG", "HJ"):
        return {
            "4bet": R("KK+, QQ:0.7, AKs, AKo:0.6, A5s:0.4, A4s:0.3, JJ:0.2"),
            "call": R("QQ:0.3, JJ:0.8, TT, 99, 88:0.7, 77:0.4, AKo:0.4, AQs, AJs, ATs:0.6, KQs, KJs:0.6, QJs:0.6, JTs:0.6, T9s:0.4, AQo:0.4"),
        }
    if opener == "CO":
        return {
            "4bet": R("KK+, QQ:0.6, AKs, AKo:0.7, A5s:0.5, A4s:0.4, A3s:0.2, JJ:0.2, KQs:0.2"),
            "call": R("QQ:0.4, JJ:0.8, TT, 99, 88, 77:0.6, 66:0.4, 55:0.3, AKo:0.3, AQs, AJs, ATs, A9s:0.4, AQo:0.6, AJo:0.3, KQs, KJs, KTs:0.6, QJs, QTs:0.5, JTs, T9s, 98s:0.6, 87s:0.5, 76s:0.6, 65s:0.5, 54s:0.3, 66:0.6, 55:0.5, 44:0.3, A8s:0.4, AJo:0.5, KQo:0.5, KJo:0.3, K9s:0.4, Q9s:0.3, J9s:0.5, T8s:0.4, 97s:0.3"),
        }
    if opener == "BTN":
        return {
            "4bet": R("KK+, QQ:0.5, AKs, AKo:0.7, A5s:0.6, A4s:0.5, A3s:0.3, JJ:0.2, KQs:0.2, AQo:0.2"),
            "call": R("QQ:0.5, JJ:0.8, TT, 99, 88, 77, 66:0.6, 55:0.5, 44:0.4, 33:0.3, 22:0.3, AKo:0.3, AQs, AJs, ATs, A9s:0.6, A8s:0.4, AQo:0.6, AJo:0.5, ATo:0.3, KQs, KJs, KTs, K9s:0.5, KQo:0.5, QJs, QTs, Q9s:0.8, JTs, J9s:0.6, T9s, T8s:0.5, 98s, 87s, 76s, 65s, 54s:0.7, A7s-A6s:0.5, ATo:0.6, A9o:0.3, KJo:0.5, KTo:0.3, QJo:0.4, J8s:0.4, 97s:0.6, 86s:0.5, 75s:0.5, 64s:0.4, 53s:0.3, 44:0.7, 33:0.6, 22:0.6"),
        }
    # SB opener facing BB 3bet
    return {
        "4bet": R("JJ+, TT:0.4, AKs, AKo, AQs:0.5, A5s, A4s, A3s:0.5, KQs:0.3, AJo:0.3"),
        "call": R("TT:0.6, 99, 88, 77, 66, 55:0.7, 44:0.5, 33:0.4, 22:0.4, AQs:0.5, AJs, ATs, A9s, A8s:0.7, A7s:0.5, A6s:0.5, A2s:0.5, AQo, AJo:0.7, ATo:0.5, KQs:0.7, KJs, KTs, K9s, K8s:0.5, KQo, KJo:0.5, QJs, QTs, Q9s, JTs, J9s, T9s, T8s, 98s, 97s:0.5, 87s, 76s, 65s, 54s, K7s:0.5, Q8s:0.5, J8s:0.5, A9o:0.4, KTo:0.5, QJo:0.5, QTo:0.3, JTo:0.4, 86s:0.5, 75s:0.5, 64s:0.4, 53s:0.3"),
    }


# The opener in position facing a blind's 3bet (to 11bb), as shares of its opening range
# (fold/call/4bet): UTG 51/36/14, HJ 57/31/12, CO 50/41/9. Preflop Wizard's per-seat table (UTG
# 74-82/10-15/8-11, CO 59-68/22-28/10-13) isn't split by who 3bets; gtopreflop.com's raked 8-max solve
# (5% rake capped at 4bb, 12bb blind 3bets, https://gtopreflop.com/posts/4-bet-ranges-guide) has UTG
# vs SB 71.4/14.1/14.5 and CO vs BB 68.0/22.4/9.6: the 4bets match, the calls are about 20 points
# lower there (rake, as with the big blind's defence). Left derived. The button has its own charts
# below.
def _ip_vs_blind_3bet(o: str) -> Dict[str, np.ndarray]:
    c = _vs_3bet(o)
    return {"4bet": c["4bet"], "call": np.minimum(1.0, c["call"] * 1.25 + IP_VS_3BET_EXTRA)}


# The button facing the big blind's 3bet (2.5bb open, 11bb 3bet): fold ~54.6%, call ~33.9%, 4bet
# ~11.5% of its opening range (888poker, "button vs blind 3bets defence", stack depth not stated; an
# older 888poker article has 4bets at 23.6% of the continuing range, and a rough 6-max 100bb guide
# BTN at 50-55/30-35/12-15 fold/call/4bet, Preflop Wizard). The derived chart folded 58.7%, called
# 35.3% and 4bet 6.0%: a 4bet range of nearly all value. Here 54/34/12: more value (QQ, AKo, JJ, AQs)
# from the calls and 4bet bluffs from the folds (the rest of A5s-A2s, K8s-K5s, Q8s, A8o/A7o/A5o/A4o,
# K9o).
_extra_value = R("QQ:0.3, AKo:0.3, JJ:0.2, AQs:0.3")
_extra_bluff = R("A5s:0.4, A4s:0.5, A3s:0.5, A2s:0.6, K8s:0.4, K7s:0.4, K6s:0.3, K5s:0.3, Q8s:0.3, A8o:0.2, A7o:0.2, A5o:0.3, A4o:0.2, K9o:0.2")
BTN_VS_BB_3BET = {"4bet": np.minimum(1.0, _ip_vs_blind_3bet("BTN")["4bet"] + _extra_value + _extra_bluff),
                  "call": np.maximum(0.0, _ip_vs_blind_3bet("BTN")["call"] - _extra_value)}

# The button facing the small blind's 3bet (2.5bb open, 10-12bb 3bet). gtopreflop.com's solve (8-max
# 100bb, 5% rake capped at 4bb, 12bb 3bets from either blind; it says 6-max lands within a point,
# https://gtopreflop.com/posts/4-bet-ranges-guide) has the button 4betting as often against either
# blind (10.4% of its opens against the SB, 10.3% against the BB) and calling less against the SB
# (13.9% against 20.0%, folding 75.7% against 69.7%). Its hand mix against the SB: QQ 89.5% 4bet,
# AKo 80%, A9s 77.5%, KTs 75.8%, AKs 59.4% (calls 41%), A4s 42.8%, K9s 33.9%, KQo 32.6%, JJ 23.6%
# (calls 76%), AJo 23.4%, A5s 21.9% (calls 72%), TT 19.5% (calls 81%); AQs and KQs call 100%; the
# offsuit hands that don't 4bet (AJo, KQo) and K9s fold rather than call. 888poker's remark that
# 4betting is more attractive against SB 3bets is about the regular's tendencies (wider SB 3bets than
# theory, tighter BB 3bets), not the equilibrium. The derived chart (the BB one with calls widened)
# folded 58.4%, called 35.6% and 4bet 6.0%; here the 4bets match the BB chart's 11.6% and the calls
# are its 34% less gtopreflop's 6-point gap: about 59/29/11.6.
_sb_4bet = R("KK+, QQ:0.9, AKo:0.8, AKs:0.6, A9s:0.75, KTs:0.75, A4s:0.45, K9s:0.35, KQo:0.35, JJ:0.25, "
             "AJo:0.35, A5s:0.2, TT:0.2, A3s:0.4, A2s:0.4, K8s:0.3, K7s:0.3, K6s:0.3, K5s:0.3, Q9s:0.3, Q8s:0.2, "
             "J9s:0.2, KJo:0.2, A8o:0.15, A5o:0.2, K9o:0.15")
BTN_VS_SB_3BET = {
    "4bet": _sb_4bet,
    "call": np.clip(np.minimum(_ip_vs_blind_3bet("BTN")["call"], 1.0 - _sb_4bet)
                    - R("AJo, KQo, K9s, AQo:0.3, ATo:0.4, KJo:0.4") + R("AKs:0.4, A5s:0.72"), 0.0, 1.0 - _sb_4bet),
}


# ---------- 3bettor facing a 4bet ----------
VS_4BET = {
    "jam": R("KK+, QQ:0.6, AKs, AKo:0.5, A5s:0.3"),
    "call": R("QQ:0.4, JJ, TT:0.6, 99:0.3, AKo:0.5, AQs, AJs:0.5, KQs:0.5, A5s:0.3, A4s:0.3, AQo:0.3"),
}
# cold-facing a 3bet (hero neither opener nor 3bettor)
COLD_VS_3BET = {   # the 3bet came over an early open (a tight 3bet)
    "4bet": R("KK+, AKs, QQ:0.3, AKo:0.3"),
    "call": R("QQ:0.7, JJ, TT:0.5, AKo:0.5, AQs, AJs:0.5, KQs:0.5"),
}
# ...over a late open (CO/BTN/SB: the 3bet is wider, so more continues)
COLD_VS_LATE_3BET = {
    "4bet": R("QQ+, AKs, AKo:0.7, JJ:0.3, A5s:0.4, A4s:0.3"),
    "call": R("JJ:0.7, TT, 99:0.5, AKo:0.3, AQs, AJs:0.6, ATs:0.3, KQs:0.7, KJs:0.3, AQo:0.3"),
}
COLD_OOP_CALL = 0.5   # out of position against the 3bettor postflop, cold calls are this share (4bet or fold)


def cold_vs_3bet(hero: str, opener_key: str, threebettor: str, positions) -> Dict[str, np.ndarray]:
    """Facing a 3bet without having raised: by how wide the 3bet is (the opener's seat) and whether
    hero has position on the 3bettor postflop."""
    c = COLD_VS_LATE_3BET if _norm(opener_key) in ("CO", "BTN", "SB") else COLD_VS_3BET
    order = postflop_order(list(positions))
    if order.index(hero) < order.index(threebettor):
        # out of position: half the cold calls go, premiums into 4bets, the rest folds
        moved = c["call"] * (1.0 - COLD_OOP_CALL)
        c = {"4bet": np.minimum(1.0, c["4bet"] + moved * COLD_PREMIUM), "call": c["call"] * COLD_OOP_CALL}
    return c


COLD_PREMIUM = R("QQ+, AKs, AKo")
# facing a 5bet: call or fold when it is all-in, and deep enough to have chips behind, jam the top
VS_5BET = {"call": R("KK+, QQ:0.7, AKs, AKo:0.7, JJ:0.3")}
VS_5BET_DEEP = {"raise": R("AA, KK:0.6, AKs:0.4"), "call": R("KK:0.4, QQ:0.8, AKs:0.6, AKo:0.6, JJ:0.3")}

# heads-up button (SB, in position postflop) facing the big blind's 3bet
HU_SB_VS_3BET = {
    "4bet": R("QQ+, JJ:0.5, AKs, AKo, AQs:0.5, A5s-A2s:0.4, K9s:0.2, Q9s:0.2, J9s:0.2"),
    "call": R("22-TT, JJ:0.5, A2s-AJs, AQs:0.5, A8o-AQo, A7o-A5o, A4o-A2o:0.5, K2s+, K9o+, K8o:0.5, Q5s+, QTo+, Q9o:0.6, J6s+, JTo, J9o:0.6, T6s+, T9o, T8o:0.4, 96s+, 98o:0.4, 85s+, 87o:0.3, 74s+, 64s+, 63s:0.3, 53s+, 43s:0.5"),
}

# heads-up big blind facing the button's 4bet (its 3bets are wide, so it continues wider than 6-max)
HU_BB_VS_4BET = {
    "jam": R("QQ+, JJ:0.5, AKs, AKo, AQs:0.3, A5s:0.3, A4s:0.2"),
    "call": R("JJ:0.5, TT, 99:0.5, AQs:0.7, AQo:0.6, AJs, AJo:0.3, KQs, KJs:0.6, KQo:0.3, QJs:0.5, JTs:0.5"),
}
# an iso-raiser facing the limper's re-raise (a trapped range: mostly QQ+ and AK)
VS_LIMP_RERAISE = {"4bet": R("KK+, AKs:0.5"), "call": R("QQ, JJ:0.6, TT:0.3, AKs:0.5, AKo:0.6, AQs:0.3")}
# a seat in position facing a 7+ handed early-position open (~13%: tighter than the 6-max UTG chart)
# the blinds facing a 7+ handed early-position open (tighter than vs a 6-max UTG open)
BB_VS_EP = {
    "3bet": R("QQ+, JJ:0.3, AKs, AKo:0.8, AQs:0.3, A5s:0.5, A4s:0.3"),
    "call": R("22-TT, JJ:0.7, AQs:0.7, AJs-A6s, A5s:0.5, A4s-A2s:0.7, AQo, AJo:0.7, ATo:0.3, KTs+, K9s:0.6, KQo, KJo:0.3, QTs+, Q9s:0.5, J9s+, T8s+, 97s+, 86s+, 75s+, 64s:0.6, 54s:0.5"),
}
SB_VS_EP = {
    "3bet": R("QQ+, JJ:0.4, AKs, AKo:0.8, AQs:0.5, A5s:0.5"),
    "call": R("TT-88:0.5, JJ:0.3, AQs:0.3, AJs:0.3, KQs:0.3"),
}
VS_EP = {
    "3bet": R("QQ+, AKs, AKo:0.6, A5s:0.3, JJ:0.3"),
    "call": R("22-TT, JJ:0.7, AQs, AJs, ATs:0.5, AQo:0.5, KQs, KJs:0.5, QJs:0.5, JTs:0.5, T9s:0.3, 98s:0.2"),
}

# ---------- limped pots ----------
# open-limp range (players who limp first in: rarely the premiums, which raise)
LIMP = R("22-99, TT+:0.2, A2s-AJs, AQs+:0.2, A2o-ATo, AJo+:0.3, K2s+, K9o+, Q6s+, Q9o+, J7s+, J9o+, T7s+, T9o, 96s+, 86s+, 75s+, 64s+, 54s")
# Raising over limpers (iso-raising). Against one limper the range is a little tighter than the seat's
# open (the limper is in the pot, and a limping range calls a raise a lot: LIMP_VS_RAISE) and mostly
# linear: every hand that opens strongly raises, with suited wheel aces and suited connectors as the
# lighter raises in position. Published widths against one limper (6-max, 100bb): BTN ~30-40% of hands,
# CO ~22-28%, HJ ~15-20%, SB ~18-25%, BB ~15% (Preflop Wizard, preflopwizard.app/blog/
# poker-iso-raise-strategy); BTN 35-45%, CO 28-35%, HJ and earlier 18-22%, SB 20-25% (Grindlab,
# grindlab.gg/en/blog/iso-raise-vs-limper-poker); a common default is the open range of the seat
# before (BTN raises the CO's opens). Here UTG (7+ handed, a seat behind an early limp) 14%, HJ 18%,
# CO 24%, BTN 36%, SB 19% (out of position, the big blind behind), against opens of 17/19/27/46/50%.
# The old charts raised 10/10/13/16/10%: well under half the button's opens.
ISO = {
    "UTG": R("55+, A8s+, A5s-A4s, KTs+, QTs+, JTs, T9s:0.5, ATo+, KJo+"),
    "HJ": R("44+, 33:0.5, A2s+, K9s+, QTs+, Q9s:0.5, JTs, J9s:0.5, T9s, 98s:0.5, ATo+, KJo+, QJo:0.5"),
    "CO": R("22+, A2s+, K8s+, Q9s+, J9s+, T8s+, 98s, 87s, 76s:0.5, 65s:0.5, A9o+, A8o:0.5, KTo+, QTo+, JTo:0.5"),
    "BTN": R("22+, A2s+, K5s+, Q7s+, J7s+, T7s+, 96s+, 86s+, 75s+, 65s, 54s, A5o+, A4o-A2o:0.5, K9o+, K8o:0.5, QTo+, Q9o:0.5, JTo, J9o:0.5, T9o:0.5"),
    "SB": R("55+, 44:0.5, A2s+, K9s+, QTs+, Q9s:0.5, JTs, J9s:0.5, T9s, 98s:0.5, 87s:0.5, ATo+, A9o:0.5, A8o:0.5, KJo+, KTo:0.5, QJo"),
}
# Against two or more limpers: value-heavy, the weak suited connectors and offsuit hands dropped
# (Preflop Wizard: "tighten each range by roughly one tier"; two limpers drop weak suited connectors
# and offsuit bluffs, three are value-heavy), about 0.6x the one-limper range: UTG 9%, HJ 11%, CO 14%,
# BTN 19%, SB 11%. The pairs and suited hands this drops over-limp instead (in OVERLIMP_MULTI).
ISO_MULTI = {
    "UTG": R("88+, 77:0.5, ATs+, A5s:0.5, KJs+, QJs:0.5, AJo+, KQo:0.5"),
    "HJ": R("77+, 66:0.5, A9s+, A5s-A4s:0.5, KTs+, QJs, JTs:0.5, AJo+, KQo"),
    "CO": R("66+, 55:0.5, A7s+, A5s-A2s:0.5, KTs+, QTs+, JTs, T9s:0.5, ATo+, KJo+"),
    "BTN": R("44+, 33:0.5, 22:0.5, A2s+, K9s+, QTs+, Q9s:0.5, JTs, J9s:0.5, T9s, 98s:0.5, ATo+, A9o:0.5, KJo+, KTo:0.5, QJo:0.5"),
    "SB": R("77+, 66:0.5, A9s+, A5s-A4s:0.5, KTs+, QJs, JTs:0.5, AJo+, KQo"),
}
OVERLIMP = {
    "UTG": R("22-66, A2s-A9s:0.5, KTs:0.5, QTs:0.5, JTs:0.5, T9s:0.5, 98s:0.5, 87s:0.3, 76s:0.3"),
    "HJ": R("22-66, A2s-A9s:0.5, KTs:0.5, QTs:0.5, JTs:0.5, T9s:0.5, 98s:0.5, 87s:0.3, 76s:0.3"),
    "CO": R("22-55, A2s-A7s:0.5, K8s-K9s:0.5, Q9s, J9s, T9s, 98s, 87s, 76s, 65s:0.5"),
    "BTN": R("22-44, A2s-A6s:0.3, K5s-K8s, Q8s-Q9s, J8s-J9s, T8s-T9s, 97s+, 86s+, 75s+, 64s+, 54s, A9o-A2o:0.3, KTo:0.5, QTo+, JTo"),
    # small blind completing
    "SB": R("22-66, A2s-A8s, A2o-ATo:0.5, K2s-K9s, KJo-KTo, Q5s+, Q9o+, J7s+, J9o+, T7s+, T9o, 96s+, 86s+, 75s+, 64s+, 54s, 53s, 43s"),
}
# a limper facing a raise: the trapped premiums re-raise, the rest of a limping range calls a lot
# (it already has chips in and the range is capped) and folds its weakest hands
LIMP_VS_RAISE = {
    "raise": R("QQ+, AKs, AKo:0.7"),
    "call": R("22-JJ, A2s-AQs, ATo-AQo, A9o:0.5, K9s+, KJo+, KTo:0.5, Q9s+, QJo, J9s+, T8s+, 97s+, 86s+, 75s+, 64s+, 54s"),
}
# an over-limp never takes a hand the iso chart raises; with two or more limpers the pairs and suited
# hands that the one-limper chart raises and ISO_MULTI doesn't over-limp (set-mining and suited hands
# play well multiway), the rest of the dropped hands fold
_PAIRS_SUITED = R("22+, A2s+, K2s+, Q2s+, J2s+, T2s+, 92s+, 82s+, 72s+, 62s+, 52s+, 42s+, 32s")
OVERLIMP_MULTI = {k: np.minimum(1.0 - ISO_MULTI[k], OVERLIMP[k] + np.maximum(ISO[k] - ISO_MULTI[k], 0.0) * _PAIRS_SUITED)
                  for k in ISO}
OVERLIMP = {k: np.minimum(OVERLIMP[k], 1.0 - ISO[k]) for k in ISO}
# Big blind raising over limps (the rest checks): only the small blind limped (24%) / one other limper
# (15%: Preflop Wizard has the big blind raising ~15% to isolate; the old chart raised 8% over any
# limpers) / several limpers (9%, value only)
BB_VS_SB_LIMP = R("55+, A2s+, A7o+, K7s+, KTo+, Q9s+, QJo, J9s+, T9s, 98s:0.5, 87s:0.5, 65s:0.5, 54s:0.5, K6o-K2o:0.3, Q5s-Q2s:0.3")
BB_VS_LIMP = R("66+, 55:0.5, A8s+, A5s-A2s:0.6, KTs+, K9s:0.5, QTs+, JTs, T9s:0.5, ATo+, KJo+, QJo:0.5")
BB_VS_LIMPS = R("88+, 77:0.5, ATs+, A5s:0.5, KJs+, QJs:0.5, AJo+, KQo")

# antes: the dead money makes opens wider, played as the chart this many seats later (UTG -> HJ ...)
RFI_LADDER = ["EP", "UTG", "HJ", "CO", "BTN"]


def ante_shift(positions) -> int:
    dead = getattr(positions, "ante", 0.0) * len(positions)
    return min(2, int(round(dead / 0.8)))


def _actions_at(hero: str, actions: List[dict], positions: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Decide the chart scenario from the preflop action list.

    actions: list of {"pos": str, "action": "fold"|"call"|"raise"} in order.
    Returns dict of aggressive/passive action names -> weight vectors.
    Fold is implicit.
    """
    positions = positions or POSITIONS
    key = lambda p: chart_key(p, positions)
    raisers = [a["pos"] for a in actions if a["action"] in ("raise", "all-in")]
    n_raises = len(raisers)
    if n_raises == 0:
        limpers = [a["pos"] for a in actions if a["action"] == "call"]
        role = _norm(key(hero))
        if role == "BB":   # the big blind (or straddler) checks or raises the limpers
            if limpers == ["SB"] or (len(positions) == 2 and limpers):
                r = BB_VS_SB_LIMP
            else:
                r = BB_VS_LIMP if len(limpers) == 1 else BB_VS_LIMPS
            return {"raise": r, "check": 1.0 - r}
        if limpers:
            k = role if role in ISO else "UTG"
            if len(limpers) >= 2:
                return {"raise": ISO_MULTI[k], "call": OVERLIMP_MULTI[k]}
            return {"raise": ISO[k], "call": OVERLIMP[k]}
        k = key(hero)
        if k in RFI_LADDER:
            k = RFI_LADDER[min(RFI_LADDER.index(k) + ante_shift(positions), len(RFI_LADDER) - 1)]
        return {"raise": RFI[k if k in RFI else _norm(k)]}
    if n_raises == 1:
        first = next(i for i, a in enumerate(actions) if a["action"] in ("raise", "all-in"))
        if any(a["pos"] == hero and a["action"] == "call" for a in actions[:first]):
            return dict(LIMP_VS_RAISE)      # hero limped, then someone raised
        chart = vs_rfi_chart(key(hero), key(raisers[0]))
        return {"raise": chart["3bet"], "call": chart["call"]}
    if n_raises == 2:
        opener, threebettor = raisers
        first = next(i for i, a in enumerate(actions) if a["action"] in ("raise", "all-in"))
        if hero == opener and any(a["pos"] == threebettor and a["action"] == "call" for a in actions[:first]):
            c = VS_LIMP_RERAISE        # the 3bettor limped first: a limp-reraise
            return {"raise": c["4bet"], "call": c["call"]}
        if hero == opener:
            c = vs_3bet(key(opener), key(threebettor))
            return {"raise": c["4bet"], "call": c["call"]}
        c = cold_vs_3bet(hero, key(opener), threebettor, positions)
        return {"raise": c["4bet"], "call": c["call"]}
    if n_raises == 3:
        if hero == raisers[1]:
            c = HU_BB_VS_4BET if len(positions) == 2 else VS_4BET
            return {"raise": c["jam"], "call": c["call"]}
        c = cold_vs_3bet(hero, key(raisers[0]), raisers[1], positions)
        return {"raise": c["4bet"] * 0.5, "call": c["call"] * 0.5}
    if [a["action"] for a in actions if a["action"] in ("raise", "all-in")][-1] == "raise":
        return dict(VS_5BET_DEEP)       # a 5bet that leaves chips behind (an all-in goes to the jam logic)
    return {"call": VS_5BET["call"]}


def strategy_for(hero: str, actions: List[dict], positions: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Full mixed strategy (including fold) per combo for hero at this point."""
    key = (hero, tuple((a["pos"], a["action"], a.get("to"), a.get("amount")) for a in actions),
           tuple(positions or POSITIONS), getattr(positions, "ante", 0.0), getattr(positions, "straddle", False))
    if key not in _STRAT_CACHE:
        if len(_STRAT_CACHE) > 4096:
            _STRAT_CACHE.clear()
        acts = _actions_at(hero, actions, positions)
        total = sum(acts.values())
        scale = np.where(total > 1, 1.0 / np.maximum(total, 1e-12), 1.0)
        acts = {k: v * scale for k, v in acts.items()}
        total = sum(acts.values())
        if "check" not in acts:
            acts["fold"] = np.clip(1.0 - total, 0.0, 1.0)
            acts = _caller_continues(hero, actions, positions, acts)
            acts = _vs_4bet_range(hero, actions, positions, acts)
            acts = _vs_5bet_jam(hero, actions, positions, acts)
        _STRAT_CACHE[key] = monotone(acts)
    return dict(_STRAT_CACHE[key])


_STRAT_CACHE: Dict[tuple, Dict[str, np.ndarray]] = {}
MIN_LINE_COMBOS = 8.0   # a line leaving fewer combos than this is filled out (see thin_line)


def thin_line(r: np.ndarray, scratch: np.ndarray, prior: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    """A player's range after an action (`r`, their range so far times the action's frequencies), filled
    out with the hands that take the action from scratch (`scratch`, the action's frequencies alone)
    when the line leaves fewer than MIN_LINE_COMBOS: a near-empty range has the postflop solve play
    against a face-up hand (a flat that then 4bets left none; a cold call of a 3bet that then calls a
    4bet a handful). The filler weighs 1 - combos / MIN_LINE_COMBOS: all of it for an empty line,
    none at the threshold. The hands that did reach the spot are kept. When even that falls short
    (the action's own chart is that narrow: an opener calling a cold 4bet calls 6.8 combos from
    scratch) and `prior` (the range before the action) is given, the strongest hands of it not yet in
    top the range up to MIN_LINE_COMBOS, by all-in equity against a random hand. Returns the range and
    whether it is mostly filler."""
    tot, add = float(r.sum()), float(scratch.sum())
    if tot >= MIN_LINE_COMBOS:
        return r, False
    out = r
    if add > tot:
        lam = 1.0 - tot / MIN_LINE_COMBOS
        out = np.minimum(r + lam * scratch, 1.0)
    need = MIN_LINE_COMBOS - float(out.sum())
    if prior is not None and need > 1e-9:
        room = np.clip(prior - out, 0.0, None)
        order = np.argsort(-_strength())
        cum = np.cumsum(room[order])
        k = int(np.searchsorted(cum, need))
        add_ = np.zeros_like(out)
        add_[order[:k]] = room[order[:k]]
        if k < len(order):
            add_[order[k]] = need - (cum[k - 1] if k > 0 else 0.0)
        out = np.minimum(out + np.clip(add_, 0.0, None), 1.0)
    return out, float(out.sum()) - tot > tot


_STRENGTH = None


def _strength() -> np.ndarray:
    """Per combo, all-in equity against a random hand (the order thin_line tops a range up in)."""
    global _STRENGTH
    if _STRENGTH is None:
        from .pf_classes import equity_vs
        _STRENGTH = np.nan_to_num(equity_vs(np.ones(N_COMBOS)), nan=0.0)
    return _STRENGTH
# A player who called a raise and now faces the next one continues with at least this share of the
# range they called with, by the raises in now: a flat facing a squeeze, a cold call of a 3bet facing
# the 4bet, a call of a 4bet facing the jam. The charts for those spots are written for a player who
# hasn't put chips in (continuing with QQ+/AK, which a flat or cold call rarely holds), so on their own
# they leave a caller's range nearly empty. For scale, openers continue ~40% against a 3bet and
# 3bettors ~50% against a 4bet.
CALLER_CONTINUE = {2: 0.30, 3: 0.35, 4: 0.55}


VS_4BET_BLEND = 0.04   # equity over which a 3bettor's continue frequency ramps against an unusual 4bet range
VS_4BET_REALIZE = 0.8  # ...and the share of its equity a call of a 4bet realizes (sets the value bar)


def _bets(actions: List[dict]) -> Dict[str, float]:
    """Each seat's chips in after the preflop actions (blinds posted; default sizes where none given)."""
    bets = {"SB": 0.5, "BB": 1.0}
    cur = 1.0
    for a in actions:
        if a["action"] == "call":
            bets[a["pos"]] = cur
        elif a["action"] in ("raise", "all-in"):
            cur = 100.0 if a["action"] == "all-in" else float(a.get("to") or a.get("amount") or cur * 3)
            bets[a["pos"]] = cur
        else:
            bets.setdefault(a["pos"], 0.0)
    return bets


def _vs_4bet_range(hero: str, actions: List[dict], positions, acts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The 3bettor facing a 4bet: the vs-4bet charts are written against the opener's 4bet, which
    carries bluffs. A cold 4bet, or one from a player who flatted first, is nearly all value, so each
    hand is re-drawn by its equity against the actual 4bet range versus against the opener's: a hand
    that does worse continues less (a value hand once it drops below the price of calling over the
    equity a call realizes, a bluff or a playable hand by how far it drops), and one that does better
    calls more. Bluff jams, and the value jams the chart mixes with calls, also scale by how often the
    4bettor folds to a jam compared with the opener (the rest of those value jams call).
    Against the opener's own 4bet nothing changes."""
    from .pf_classes import equity_vs
    is_raise = lambda a: a["action"] in ("raise", "all-in")
    raisers = [a["pos"] for a in actions if is_raise(a)]
    if len(raisers) != 3 or raisers[1] != hero or len(positions or POSITIONS) == 2 or "fold" not in acts:
        return acts
    opener, four = raisers[0], raisers[2]
    i3 = max(i for i, a in enumerate(actions) if a["pos"] == hero and is_raise(a))
    std = [a if a["pos"] in (opener, hero) else {"pos": a["pos"], "action": "fold"} for a in actions[:i3 + 1]]
    std.append({"pos": opener, "action": "raise"})
    act_r, std_r = replay_ranges(actions, positions), replay_ranges(std, positions)
    if four not in act_r or opener not in std_r or hero not in std_r:
        return acts
    eq, eq_std = equity_vs(act_r[four]), equity_vs(std_r[opener])
    if np.allclose(np.nan_to_num(eq), np.nan_to_num(eq_std), atol=1e-6):
        return acts
    # the bar a value hand is held to: the price of calling the 4bet, over the share of its equity a
    # call realizes (the same in every spot, however wide the 3bet range the chart is applied to)
    bets = _bets(actions)
    owe = bets[four] - bets[hero]
    bar = owe / max(sum(bets.values()) + owe, 1e-9) / VS_4BET_REALIZE
    eq, eq_std = np.nan_to_num(eq, nan=0.0), np.nan_to_num(eq_std, nan=0.0)
    keep = np.clip(1.0 + (eq - np.minimum(eq_std, bar)) / VS_4BET_BLEND, 0.0, 1.0)
    add = np.clip(0.5 + (eq - bar) / VS_4BET_BLEND, 0.0, 1.0) * acts["fold"]
    better = eq > eq_std
    out = {k: (v if k == "fold" else np.where(better, v, v * keep)) for k, v in acts.items()}
    out["call"] = out["call"] + np.where(better, add, 0.0)
    if "raise" in out:
        # how often the 4bettor folds to a jam, here and for the opener (a value-only 4bet hardly folds):
        # calling by equity against the chart's jams from hero's range there (not the scaled jams below,
        # which depend on this)
        def folds(line, p4, ranges):
            jams = line + [{"pos": hero, "action": "all-in"}]
            call = _jam_calls(ranges[hero] * acts["raise"], jams, p4)
            return float((ranges[p4] * (1.0 - call)).sum() / max(ranges[p4].sum(), 1e-12))
        f_s = folds(std, opener, std_r)
        scale = min(1.0, folds(actions, four, act_r) / max(f_s, 1e-9)) if f_s > 1e-6 else 1.0
        # jams below the bar are bluffs: they live on the 4bettor folding, so they scale by it. Value
        # hands the chart mixes between jam and call (QQ, AK) are about even against the opener: a jam's
        # edge over a call is the equity it denies the 4bettor's bluffs (against its value both get the
        # stacks in), while a call keeps the option of folding bad flops. So their jams scale by it too,
        # the rest becoming calls; hands that always jam (KK+, AKs) keep jamming.
        bluff = eq_std < bar
        mixed = ~bluff & (acts["raise"] > 1e-9) & (acts["call"] > 1e-9)
        cut = out["raise"] * (1.0 - scale)
        out["raise"] = np.where(bluff | mixed, out["raise"] - cut, out["raise"])
        out["call"] = out["call"] + np.where(mixed, cut, 0.0)
    out["fold"] = np.clip(1.0 - sum(v for k, v in out.items() if k != "fold"), 0.0, 1.0)
    return out


JAM_CALL_BLEND = 0.04   # facing a 5bet jam: call frequency ramps over this much equity around the pot odds


def _vs_5bet_jam(hero: str, actions: List[dict], positions, acts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Facing a 5bet (or later) all-in, call where equity against the jammer's replayed range beats the
    pot odds, as hero's own advice does (Advisor._vs_jam). The fixed VS_5BET chart called QQ 70% and AK
    70% whoever jammed, looser than the charts' own value-heavy jam ranges allow (QQ has 36% against a
    standard one, needing 41%), and blind to a jam from a tighter line or with dead money in."""
    is_raise = lambda a: a["action"] in ("raise", "all-in")
    aggr = [a for a in actions if is_raise(a)]
    if len(aggr) < 4 or aggr[-1]["action"] != "all-in" or aggr[-1]["pos"] == hero or "call" not in acts:
        return acts
    r = replay_ranges(actions, positions).get(aggr[-1]["pos"])
    if r is None or r.sum() < 1e-9:
        return acts
    call = _jam_calls(r, actions, hero)
    return {"call": call, "fold": 1.0 - call}


def _jam_calls(jam_range: np.ndarray, actions: List[dict], caller: str) -> np.ndarray:
    """Per combo, how often `caller` calls the all-in that ends `actions` from `jam_range`: by equity
    against it versus the pot odds."""
    from .pf_classes import equity_vs
    if jam_range.sum() < 1e-9:
        return np.zeros(N_COMBOS)
    bets = _bets(actions)
    owe = bets[actions[-1]["pos"]] - bets.get(caller, 0.0)
    need = owe / max(sum(bets.values()) + owe, 1e-9)
    return np.clip(0.5 + (np.nan_to_num(equity_vs(jam_range), nan=0.0) - need) / JAM_CALL_BLEND, 0.0, 1.0)


def _caller_continues(hero: str, actions: List[dict], positions, acts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Top up a caller's continuing range (see CALLER_CONTINUE) with calls by its strongest folds:
    ranked by how readily the button continues against a 3bet (the charts' sense of what plays well
    in a reraised pot: TT-77 and suited broadways over 22-55), then equity against the raiser's
    range. Only for the raise right after hero's call; once another raise is in too (hero
    sandwiched), the charts' near-total folding stands."""
    from .pf_classes import equity_vs
    mine = [i for i, a in enumerate(actions) if a["pos"] == hero and a["action"] not in ("fold", "check")]
    if not mine or actions[mine[-1]]["action"] != "call" or "call" not in acts:
        return acts
    is_raise = lambda a: a["action"] in ("raise", "all-in")
    n_then = sum(map(is_raise, actions[:mine[-1]]))
    n_now = sum(map(is_raise, actions))
    if n_then == 0 or n_now - n_then != 1:
        return acts        # a limp (LIMP_VS_RAISE is written for it), or sandwiched
    target = CALLER_CONTINUE[min(n_now, max(CALLER_CONTINUE))]
    ranges = replay_ranges(actions, positions)
    w = ranges.get(hero)
    raiser = next(a["pos"] for a in reversed(actions) if is_raise(a))
    if w is None or w.sum() < 1e-9 or raiser not in ranges:
        return acts
    fold = acts["fold"]
    need = (target - float((w * (1.0 - fold)).sum() / w.sum())) * w.sum()
    if need <= 1e-9:
        return acts
    ref = _vs_3bet("BTN")
    eq = np.minimum(ref["4bet"] + ref["call"], 1.0) + 0.01 * np.nan_to_num(equity_vs(ranges[raiser]), nan=-1.0)
    # strongest folds first: whole classes above the bar, a share of the class at it
    levels = np.unique(eq)[::-1]
    add = np.zeros(N_COMBOS)
    for e in levels:
        at = eq == e
        avail = float((w * fold * at).sum())
        if avail <= 1e-12:
            continue
        frac = min(1.0, need / avail)
        add[at] = fold[at] * frac
        need -= avail * frac
        if need <= 1e-9:
            break
    # hands outside the range above the bar continue too, so a hand hero holds off-range plays the same
    bar = e
    add = np.where(eq > bar, fold, add)
    out = dict(acts)
    out["call"] = acts["call"] + add
    out["fold"] = fold - add
    return out


def _dominance_chains() -> List[List[str]]:
    """Hand classes in order weakest -> strongest where the stronger never continues less: the pairs,
    each high card's broadway kickers (suited and offsuit), and each suited hand over its offsuit twin."""
    rk = "AKQJT98765432"
    chains = [[r + r for r in reversed(rk)]]
    for hi in "AKQJ":
        ks = [k for k in rk[rk.index(hi) + 1:] if k in "KQJT"]
        for s in "so":
            chains.append([hi + k + s for k in reversed(ks)])
    for i in range(13):
        for j in range(i + 1, 13):
            chains.append([rk[i] + rk[j] + "o", rk[i] + rk[j] + "s"])
    return [c for c in chains if len(c) > 1]


_CHAINS = None


def monotone(acts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Keep the strategy consistent: a hand never continues (doesn't fold) less often than a hand it
    dominates. Where it would, the shortfall moves from its fold to its call (or raise, when calling
    isn't an option). Fixes chart slips and equity noise alike."""
    global _CHAINS
    if "fold" not in acts:
        return acts
    from .pf_classes import CLASS_OF, CLASS_INDEX, N_IN
    if _CHAINS is None:
        _CHAINS = [[CLASS_INDEX[h] for h in c] for c in _dominance_chains()]
    cont = np.bincount(CLASS_OF, weights=1.0 - acts["fold"], minlength=len(N_IN)) / N_IN
    new = cont.copy()
    for _ in range(3):
        for chain in _CHAINS:
            new[chain] = np.maximum.accumulate(new[chain])
    gap = (new - cont)[CLASS_OF]
    if gap.max() < 1e-9:
        return acts
    out = dict(acts)
    k = "call" if "call" in out else "raise"
    add = np.minimum(gap, out["fold"])
    out[k] = out[k] + add
    out["fold"] = out["fold"] - add
    return out


def hand_strategy(hero: str, hole: Tuple[int, int], actions: List[dict], positions: Optional[List[str]] = None) -> Dict[str, float]:
    idx = COMBO_INDEX[(min(hole), max(hole))]
    return {k: float(v[idx]) for k, v in strategy_for(hero, actions, positions).items()}


def replay_ranges(actions: List[dict], positions: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Run the preflop action sequence through the charts to get each still-active
    player's range (product of chart frequencies for the actions they took)."""
    positions = positions or POSITIONS
    ranges: Dict[str, np.ndarray] = {p: np.ones(N_COMBOS) for p in positions}
    active = set(positions)
    history: List[dict] = []
    for a in actions:
        pos, act = a["pos"], a["action"]
        if pos not in ranges:
            continue
        strat = strategy_for(pos, history, positions)
        if act == "fold":
            active.discard(pos)
        else:
            key = "raise" if act == "all-in" else ("call" if act == "check" and "check" not in strat else act)
            if key in strat:
                ranges[pos] = thin_line(ranges[pos] * strat[key], strat[key], ranges[pos])[0]
            elif act == "call" and not any(h["action"] in ("raise", "all-in") for h in history):
                ranges[pos] = ranges[pos] * LIMP     # open-limp
            # other off-chart actions leave the range unchanged
        history.append(a)
    return {p: ranges[p] for p in active}


def is_preflop_closed(actions: List[dict]) -> bool:
    """True when the betting round is complete."""
    active = list(POSITIONS)
    last_raiser: Optional[str] = None
    acted_since_raise = set()
    for a in actions:
        pos, act = a["pos"], a["action"]
        if act == "fold":
            active.remove(pos)
        elif act == "raise":
            last_raiser = pos
            acted_since_raise = {pos}
        else:
            acted_since_raise.add(pos)
    if len(active) <= 1:
        return True
    if last_raiser is None:
        return "BB" in acted_since_raise and all(p in acted_since_raise for p in active)
    return all(p in acted_since_raise for p in active)


def next_to_act(actions: List[dict]) -> Optional[str]:
    active = list(POSITIONS)
    order = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    acted_since_raise: set = set()
    last_raiser = None
    for a in actions:
        pos, act = a["pos"], a["action"]
        if act == "fold":
            active.remove(pos)
        elif act == "raise":
            last_raiser = pos
            acted_since_raise = {pos}
        else:
            acted_since_raise.add(pos)
    if is_preflop_closed(actions):
        return None
    # find the next position after the last actor in order
    start = 0 if not actions else (order.index(actions[-1]["pos"]) + 1) % 6
    for k in range(6):
        p = order[(start + k) % 6]
        if p in active and p not in acted_since_raise:
            return p
    return None
