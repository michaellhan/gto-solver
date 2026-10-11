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
    first seat straddles 2bb (it then acts last preflop and plays the big blind's charts)."""
    def __init__(self, seats, ante: float = 0.0, straddle: bool = False):
        super().__init__(seats)
        self.ante = float(ante or 0.0)
        self.straddle = bool(straddle) and len(seats) >= 3

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
        return {
            "3bet": R("QQ+, JJ:0.5, AKs, AKo:0.7, AQs, A5s, A4s:0.5, KQs:0.3"),
            "call": R("22-TT, JJ:0.5, AKo:0.3, AQo, AJs, ATs, KQs:0.7, KJs, KTs:0.5, QJs, QTs:0.5, JTs, T9s, 98s, 87s:0.5"),
        }
    return {  # BTN facing UTG/HJ/CO, or CO facing HJ-ish
        "3bet": R("JJ+, TT:0.4, AKs, AKo, AQs, AQo:0.4, AJs:0.5, A5s, A4s, A3s:0.5, KQs:0.5, KJs:0.3, 76s:0.3, 65s:0.3"),
        "call": R("22-99, TT:0.6, AJs:0.5, ATs, A9s:0.5, AQo:0.6, AJo:0.5, KQs:0.5, KJs:0.7, KTs, KQo:0.5, QJs, QTs, JTs, J9s:0.5, T9s, 98s, 87s, 76s:0.7, 65s:0.7, 54s:0.5"),
    }

VS_RFI: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
for hero in ["HJ", "CO", "BTN"]:
    for opener in POSITIONS[:POS_INDEX[hero]]:
        early = hero in ("HJ", "CO") and opener in ("UTG", "HJ") and not (hero == "CO" and opener == "HJ")
        VS_RFI[(hero, opener)] = _vs_rfi_ip(early)
# the button against a cutoff open (a wide open: more 3bets and calls than vs an early seat)
VS_RFI[("BTN", "CO")] = {
    "3bet": R("TT+, 99:0.4, AJs+, ATs:0.6, A5s-A2s:0.8, AQo+, AJo:0.6, KQs, KJs:0.6, KTs:0.3, QJs:0.4, JTs:0.3, T9s:0.3, 98s:0.3, 76s:0.4, 65s:0.4, 54s:0.3, KQo:0.4"),
    "call": R("22-88, 99:0.6, ATs:0.4, A9s-A6s:0.7, AJo:0.4, ATo:0.5, KJs:0.4, KTs:0.7, K9s:0.6, KQo:0.6, KJo:0.4, QJs:0.6, QTs, Q9s:0.6, JTs:0.7, J9s:0.8, T9s:0.7, T8s:0.6, 98s:0.7, 97s:0.4, 87s, 86s:0.4, 76s:0.6, 65s:0.6, 54s:0.5"),
}
# SB: 3bet or fold, small calling range vs late opens
for opener in ["UTG", "HJ", "CO", "BTN"]:
    if opener in ("UTG", "HJ"):
        VS_RFI[("SB", opener)] = {
            "3bet": R("TT+, 99:0.5, AJs+, ATs:0.5, A5s:0.7, A4s:0.5, AQo+, KQs, KJs:0.5, QJs:0.3"),
            "call": R("99:0.5, 77-88, 66:0.5, ATs:0.5, KJs:0.5, QJs:0.5, JTs:0.7"),
        }
    else:
        VS_RFI[("SB", opener)] = {
            "3bet": R("88+, 77:0.5, ATs+, A9s:0.5, A5s-A2s, AJo+, ATo:0.5, KTs+, K9s:0.5, KQo, KJo:0.5, QTs+, JTs, T9s, 98s:0.5, 87s:0.5, 76s:0.5"),
            "call": R("77:0.5, 22-66, A9s:0.5, A8s-A6s, ATo:0.5, K9s:0.5, K8s, Q9s, J9s, T8s, 98s:0.5, 87s:0.5, 76s:0.5, 65s, 54s"),
        }
# BB
VS_RFI[("BB", "UTG")] = {
    "3bet": R("QQ+, JJ:0.4, AKs, AKo, AQs:0.5, A5s, A4s:0.7, KQs:0.3, 65s:0.3, 54s:0.3"),
    "call": R("22-TT, JJ:0.6, AQs:0.5, AJs-A6s, A3s, A2s, AQo, AJo, ATo, A9o:0.5, K8s+, K7s-K2s:0.7, KQo, KJo, KTo:0.6, Q8s+, Q7s-Q6s:0.7, QJo, QTo:0.5, J8s+, J7s:0.7, JTo:0.5, T7s+, T6s:0.5, 96s+, 95s:0.5, 85s+, 84s:0.5, 74s+, 73s:0.3, 63s+, 53s+, 52s:0.3, 43s:0.6, 42s:0.3, A8o-A7o:0.5, K9o:0.5, Q9o:0.4, J9o:0.4, T9o:0.5, 98o:0.3, Q5s-Q2s:0.5, J6s-J5s:0.5"),
}
VS_RFI[("BB", "HJ")] = VS_RFI[("BB", "UTG")]
VS_RFI[("BB", "CO")] = {
    "3bet": R("JJ+, TT:0.5, AKs, AKo, AQs, AQo:0.5, AJs:0.5, A5s, A4s, A3s:0.5, KQs:0.5, KJs:0.3, 76s:0.3, 65s:0.3, 54s:0.3, KQo:0.3"),
    "call": R("22-99, TT:0.5, AJs:0.5, ATs-A6s, A5s:0, A2s, AQo:0.5, AJo, ATo, A9o, A8o:0.7, A7o-A2o:0.5, K2s+, KQo:0.7, KJo, KTo, K9o, K8o:0.5, Q2s+, QJo, QTo, Q9o, Q8o:0.5, J5s+, JTo, J9o, J8o:0.5, T6s+, T9o, T8o, 95s+, 98o, 97o:0.3, 84s+, 87o:0.6, 74s+, 63s+, 53s+, 43s, 42s:0.5"),
}
VS_RFI[("BB", "BTN")] = {
    "3bet": R("TT+, 99:0.5, AJs+, ATs:0.5, AQo+, AJo:0.5, A5s-A2s, KJs+, KTs:0.5, KQo:0.5, QJs:0.5, JTs:0.3, T9s:0.3, 76s:0.3, 65s:0.3, 54s:0.3, K9s:0.3, A9s:0.3"),
    "call": R("22-88, 99:0.5, ATs:0.5, A9s-A6s, AJo:0.5, ATo-A2o, K2s+, KQo:0.5, KJo-K7o, K6o:0.5, Q2s+, QJo-Q8o, Q7o:0.5, J2s+, JTo-J8o, J7o:0.5, T3s+, T9o-T7o, 95s+, 98o, 97o, 84s+, 87o, 86o:0.5, 74s+, 76o:0.5, 63s+, 65o:0.5, 53s+, 43s, 42s:0.5, 32s:0.5, K6o-K4o, Q7o, Q6o-Q5o:0.5, J7o, J6o:0.5, T6o:0.5, 96o:0.5, 86o, 85o:0.3, 75o:0.5, 64o:0.3, 54o:0.3, 52s, 42s, 32s"),
}
VS_RFI[("BB", "SB")] = {
    "3bet": R("99+, 88:0.5, ATs+, A9s:0.5, AJo+, ATo:0.5, A5s-A2s, KTs+, K9s:0.5, KQo, KJo:0.5, QTs+, JTs, T9s, 98s:0.5, 87s:0.5, K8s:0.3, Q9s:0.3, J9s:0.3, 76s:0.3, 65s:0.3"),
    "call": R("22-77, 88:0.5, A9s:0.5, A8s-A6s, ATo:0.5, A9o-A2o, K9s:0.5, K2s+, KJo:0.5, KTo-K5o, Q9s+, Q8s-Q2s, QJo-Q7o, J9s+, J8s-J2s, JTo-J7o, T2s+, T9o-T6o, 92s+, 98o-96o, 82s+, 87o-85o, 72s+, 76o-74o, 62s+, 65o, 64o, 52s+, 54o, 53o, 42s+, 43o, 32s"),
}

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
    """The UTG/HJ/CO charts are for a 3bet from a later seat (opener out of position postflop); against
    a blind's 3bet the opener has position and calls wider."""
    if opener == "HU_SB":
        return HU_SB_VS_3BET
    c = _vs_3bet(_norm(opener))
    if _norm(opener) in ("UTG", "HJ", "CO", "BTN") and _norm(threebettor) in ("SB", "BB"):
        c = {"4bet": c["4bet"], "call": np.minimum(1.0, c["call"] * 1.25 + IP_VS_3BET_EXTRA)}
    return c


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
# raising over limpers (tighter than an open: the limpers are in the pot) and over-limping
ISO = {
    "UTG": R("77+, ATs+, A5s:0.5, KTs+, QJs, AJo+, KQo"),
    "HJ": R("77+, ATs+, A5s:0.5, KTs+, QJs, AJo+, KQo"),
    "CO": R("66+, A8s+, A5s-A4s:0.5, KTs+, QTs+, JTs, ATo+, KJo+"),
    "BTN": R("55+, A7s+, A5s-A2s:0.7, K9s+, QTs+, JTs, T9s:0.5, ATo+, KJo+, QJo:0.5"),
    "SB": R("77+, A9s+, A5s:0.5, KTs+, QJs, AJo+, KQo"),
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
# big blind raising over limps (the rest checks): only the small blind limped / several limpers
BB_VS_SB_LIMP = R("55+, A2s+, A7o+, K7s+, KTo+, Q9s+, QJo, J9s+, T9s, 98s:0.5, 87s:0.5, 65s:0.5, 54s:0.5, K6o-K2o:0.3, Q5s-Q2s:0.3")
BB_VS_LIMPS = R("88+, ATs+, A5s:0.5, KJs+, AJo+, KQo:0.5")

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
            r = BB_VS_SB_LIMP if limpers == ["SB"] or (len(positions) == 2 and limpers) else BB_VS_LIMPS
            return {"raise": r, "check": 1.0 - r}
        if limpers:
            k = role if role in ISO else "UTG"
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
MIN_LINE_COMBOS = 3.0   # a line leaving fewer combos than this is off-chart; the range is rebuilt
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
                r = ranges[pos] * strat[key]
                # an off-chart line (a flat, then a 4bet): the hands that take this action alone
                ranges[pos] = r if r.sum() >= MIN_LINE_COMBOS or strat[key].sum() <= r.sum() else strat[key].copy()
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
