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
    if pos in ("UTG", "UTG1") and n >= 7:
        return "EP"
    if pos in ("UTG2", "LJ"):
        return "UTG"
    return pos

R = parse_range

# ---------- raise first in ----------
RFI: Dict[str, np.ndarray] = {
    "UTG": R("66+, 22-55:0.5, A9s+, A5s:0.5, A4s:0.5, A3s:0.3, A2s:0.3, AJo+, ATo:0.5, KJs+, KTs:0.5, KQo, QJs, QTs:0.5, JTs, T9s:0.5"),
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
            "call": R("22-TT, JJ:0.5, AQo, AJs, ATs, KQs:0.7, KJs, KTs:0.5, QJs, QTs:0.5, JTs, T9s, 98s, 87s:0.5"),
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
    "call": R("22-TT, JJ:0.6, AQs:0.5, AJs-A6s, A3s, A2s, AQo, AJo, ATo:0.5, K9s+, K8s:0.5, KQo, KJo:0.5, Q9s+, QJo:0.5, J9s+, T8s+, 97s+, 86s+, 75s+, 64s+, 54s:0.7, 43s:0.3"),
}
VS_RFI[("BB", "HJ")] = VS_RFI[("BB", "UTG")]
VS_RFI[("BB", "CO")] = {
    "3bet": R("JJ+, TT:0.5, AKs, AKo, AQs, AQo:0.5, AJs:0.5, A5s, A4s, A3s:0.5, KQs:0.5, KJs:0.3, 76s:0.3, 65s:0.3, 54s:0.3, KQo:0.3"),
    "call": R("22-99, TT:0.5, AJs:0.5, ATs-A6s, A5s:0, A2s, AQo:0.5, AJo, ATo, A9o:0.7, A8o:0.5, K5s+, K4s:0.5, KQo:0.7, KJo, KTo, K9o:0.5, Q6s+, QJo, QTo, Q9o:0.5, J7s+, JTo, J9o:0.5, T7s+, T9o, T8o:0.5, 96s+, 98o:0.5, 85s+, 87o:0.3, 75s+, 64s+, 54s, 53s, 43s"),
}
VS_RFI[("BB", "BTN")] = {
    "3bet": R("TT+, 99:0.5, AJs+, ATs:0.5, AQo+, AJo:0.5, A5s-A2s, KJs+, KTs:0.5, KQo:0.5, QJs:0.5, JTs:0.3, T9s:0.3, 76s:0.3, 65s:0.3, 54s:0.3, K9s:0.3, A9s:0.3"),
    "call": R("22-88, 99:0.5, ATs:0.5, A9s-A6s, AJo:0.5, ATo-A2o, K2s+, KQo:0.5, KJo-K7o, K6o:0.5, Q2s+, QJo-Q8o, Q7o:0.5, J2s+, JTo-J8o, J7o:0.5, T3s+, T9o-T7o, 95s+, 98o, 97o, 84s+, 87o, 86o:0.5, 74s+, 76o:0.5, 63s+, 65o:0.5, 53s+, 43s, 42s:0.5, 32s:0.5"),
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
    opener = _norm(opener)
    if opener in ("UTG", "HJ"):
        return {
            "4bet": R("KK+, QQ:0.7, AKs, AKo:0.6, A5s:0.4, A4s:0.3, JJ:0.2"),
            "call": R("QQ:0.3, JJ:0.8, TT, 99, 88:0.7, 77:0.4, AKo:0.4, AQs, AJs, ATs:0.6, KQs, KJs:0.6, QJs:0.6, JTs:0.6, T9s:0.4, AQo:0.4"),
        }
    if opener == "CO":
        return {
            "4bet": R("KK+, QQ:0.6, AKs, AKo:0.7, A5s:0.5, A4s:0.4, A3s:0.2, JJ:0.2, KQs:0.2"),
            "call": R("QQ:0.4, JJ:0.8, TT, 99, 88, 77:0.6, 66:0.4, 55:0.3, AKo:0.3, AQs, AJs, ATs, A9s:0.4, AQo:0.6, AJo:0.3, KQs, KJs, KTs:0.6, QJs, QTs:0.5, JTs, T9s, 98s:0.6, 87s:0.5, 76s:0.4, 65s:0.3"),
        }
    if opener == "BTN":
        return {
            "4bet": R("KK+, QQ:0.5, AKs, AKo:0.7, A5s:0.6, A4s:0.5, A3s:0.3, JJ:0.2, KQs:0.2, AQo:0.2"),
            "call": R("QQ:0.5, JJ:0.8, TT, 99, 88, 77, 66:0.6, 55:0.5, 44:0.4, 33:0.3, 22:0.3, AKo:0.3, AQs, AJs, ATs, A9s:0.6, A8s:0.4, AQo:0.6, AJo:0.5, ATo:0.3, KQs, KJs, KTs, K9s:0.5, KQo:0.5, QJs, QTs, Q9s:0.5, JTs, J9s:0.6, T9s, T8s:0.5, 98s, 87s, 76s, 65s, 54s:0.7"),
        }
    # SB opener facing BB 3bet
    return {
        "4bet": R("JJ+, TT:0.4, AKs, AKo, AQs:0.5, A5s, A4s, A3s:0.5, KQs:0.3, AJo:0.3"),
        "call": R("TT:0.6, 99, 88, 77, 66, 55:0.7, 44:0.5, 33:0.4, 22:0.4, AQs:0.5, AJs, ATs, A9s, A8s:0.7, A7s:0.5, A6s:0.5, A2s:0.5, AQo, AJo:0.7, ATo:0.5, KQs:0.7, KJs, KTs, K9s, K8s:0.5, KQo, KJo:0.5, QJs, QTs, Q9s, JTs, J9s, T9s, T8s, 98s, 97s:0.5, 87s, 76s, 65s, 54s"),
    }

# ---------- 3bettor facing a 4bet ----------
VS_4BET = {
    "jam": R("KK+, QQ:0.6, AKs, AKo:0.5, A5s:0.3"),
    "call": R("QQ:0.4, JJ, TT:0.6, 99:0.3, AKo:0.5, AQs, AJs:0.5, KQs:0.5, A5s:0.3, A4s:0.3, AQo:0.3"),
}
# cold-facing a 3bet (hero neither opener nor 3bettor)
COLD_VS_3BET = {
    "4bet": R("KK+, AKs, QQ:0.3, AKo:0.3"),
    "call": R("QQ:0.7, JJ, TT:0.5, AKo:0.5, AQs, AJs:0.5, KQs:0.5"),
}
# opener facing a 5bet jam (call or fold)
VS_5BET = {"call": R("KK+, QQ:0.7, AKs, AKo:0.7, JJ:0.3")}


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
        if hero == "BB":  # everyone folded / limped: treat as check
            return {"check": np.ones(N_COMBOS)}
        return {"raise": RFI[_norm(key(hero)) if key(hero) not in RFI else key(hero)]}
    if n_raises == 1:
        chart = vs_rfi_chart(key(hero), key(raisers[0]))
        return {"raise": chart["3bet"], "call": chart["call"]}
    if n_raises == 2:
        opener, threebettor = raisers
        if hero == opener:
            c = vs_3bet(key(opener), key(threebettor))
            return {"raise": c["4bet"], "call": c["call"]}
        return {"raise": COLD_VS_3BET["4bet"], "call": COLD_VS_3BET["call"]}
    if n_raises == 3:
        if hero == raisers[1]:
            return {"raise": VS_4BET["jam"], "call": VS_4BET["call"]}
        return {"raise": COLD_VS_3BET["4bet"] * 0.5, "call": COLD_VS_3BET["call"] * 0.5}
    return {"call": VS_5BET["call"]}


def strategy_for(hero: str, actions: List[dict], positions: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Full mixed strategy (including fold) per combo for hero at this point."""
    acts = _actions_at(hero, actions, positions)
    total = sum(acts.values())
    scale = np.where(total > 1, 1.0 / np.maximum(total, 1e-12), 1.0)
    acts = {k: v * scale for k, v in acts.items()}
    total = sum(acts.values())
    if "check" not in acts:
        acts["fold"] = np.clip(1.0 - total, 0.0, 1.0)
    return acts


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
            key = "raise" if act == "all-in" else ("call" if act == "check" else act)
            if key in strat:
                ranges[pos] = ranges[pos] * strat[key]
            # unknown action (e.g. limp) leaves range unchanged
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
