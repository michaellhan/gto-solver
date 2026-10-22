"""Published 6-max 100bb cash preflop solver frequencies, and how the charts measure against them.

Each reference is a spot (a preflop line, everyone not named folding in turn, default sizes unless
given), the hero to act, the rake the solve was run with (the table then plays with it: the advice
re-prices calls by the rake they go on to pay, see advisor._call_rake), whether the shares are of all hands ("hands") or of the range hero arrives
with ("range"), the published fold/call/raise shares, a tolerance (points, on each share) and the
source. `uncertain` marks a reference resting on one source whose conditions (rake, sizes, solver)
differ from the charts' or aren't stated; the test reports those but doesn't fail on them.

    python scripts/preflop_reference.py      # the table, chart against published
"""
from __future__ import annotations
import numpy as np

# Sources (accessed 2026-10):
GTOPRE = "gtopreflop.com (8-max 100bb cash, 5% rake capped at 4bb, 2.5bb opens, SB 3bb; 6-max within a point)"
REFS: list = []   # filled below


def ref(name, hero, line, want, tol, source, of="hands", table=6, rake=None, uncertain=False, note=""):
    REFS.append(dict(name=name, hero=hero, line=line, want=want, tol=tol, source=source, of=of,
                     table=table, rake=rake, uncertain=uncertain, note=note))


def _line(s: str, T):
    from gto.advisor import replay_hand
    acts = []
    for a in (s.split(", ") if s else []):
        w = a.split()
        d = {"pos": w[0], "action": w[1], **({"to": float(w[2])} if len(w) > 2 else {})}
        while (st := replay_hand(T, 100, {"preflop": acts}, 1)[0]).to_act != d["pos"]:
            acts.append({"pos": st.to_act, "action": "fold"})
        acts.append(d)
    return acts


def measure(r: dict) -> dict:
    """The charts' fold/call/raise shares (the advice, off-chart adjustments included) in reference r."""
    from gto import preflop as pf
    from gto.advisor import preflop_strategy, preflop_ranges, replay_hand
    T = pf.Table(pf.SEATS[r["table"]], rake=r["rake"])
    pre = _line(r["line"], T)
    while (st := replay_hand(T, 100, {"preflop": pre}, 1)[0]).to_act != r["hero"]:
        pre.append({"pos": st.to_act, "action": "fold"})
    s = preflop_strategy(r["hero"], pre, T, 100)
    w = preflop_ranges(pre, T, 100)[r["hero"]] if r["of"] == "range" else np.ones(pf.N_COMBOS)
    out = {k: float((w * v).sum() / w.sum()) for k, v in s.items()}
    if "check" in out:
        out["call"] = out.pop("check")
    out.setdefault("fold", 0.0)
    return out


def misses(r: dict, got: dict) -> dict:
    """Shares outside the tolerance: action -> (got, want)."""
    return {k: (got.get(k, 0.0), v) for k, v in r["want"].items() if abs(got.get(k, 0.0) - v) > r["tol"] + 1e-9}


# ---- gtopreflop.com: raked solve (5% capped at 4bb), so the table plays with that rake ----
# https://gtopreflop.com/posts/big-blind-defense, /posts/blind-vs-blind-strategy, /posts/4-bet-ranges-guide
# (8-max: its LJ is the 6-max UTG seat; it says shared 6-max nodes land within a point)
RK = (0.05, 4.0)
ref("BTN open", "BTN", "", {"raise": 0.437}, 0.03, GTOPRE, rake=RK)
ref("SB first in (raise + limp)", "SB", "", {"raise": 0.457}, 0.03, GTOPRE, rake=RK,
    note="raise 43.4%, limp 2.3%; the charts don't limp")
for _o, (_f, _c, _r) in {"UTG": (0.786, 0.168, 0.045), "HJ": (0.744, 0.196, 0.059), "CO": (0.686, 0.238, 0.077),
                         "BTN": (0.587, 0.298, 0.115)}.items():
    ref(f"BB vs {_o} open", "BB", f"{_o} raise 2.5", {"fold": _f, "call": _c, "raise": _r}, 0.03, GTOPRE, rake=RK,
        note="UTG = its LJ" if _o == "UTG" else "")
ref("BB vs SB open", "BB", "SB raise 3", {"fold": 0.50, "call": 0.338, "raise": 0.162}, 0.03, GTOPRE, rake=RK)
ref("SB vs BTN open", "SB", "BTN raise 2.5", {"fold": 0.837, "call": 0.047, "raise": 0.115}, 0.025, GTOPRE, rake=RK)
ref("BTN vs CO open (3bet)", "BTN", "CO raise 2.5", {"raise": 0.112}, 0.02, GTOPRE, rake=RK)
ref("SB vs CO open (3bet)", "SB", "CO raise 2.5", {"raise": 0.082}, 0.02, GTOPRE, rake=RK)
for _o, _t, _three, _want in (("UTG", "BTN", 7.5, (0.570, 0.244, 0.187)), ("HJ", "CO", 7.5, (0.636, 0.183, 0.181)),
                              ("CO", "BTN", 7.5, (0.640, 0.169, 0.192)), ("CO", "BB", 12, (0.680, 0.224, 0.096)),
                              ("BTN", "SB", 12, (0.757, 0.139, 0.104)), ("BTN", "BB", 12, (0.697, 0.200, 0.103))):
    ref(f"{_o} open vs {_t} 3bet", _o, f"{_o} raise 2.5, {_t} raise {_three}",
        dict(zip(("fold", "call", "raise"), _want)), 0.04, GTOPRE, of="range", rake=RK,
        note="UTG = its LJ" if _o == "UTG" else "")
ref("SB open vs BB 3bet", "SB", "SB raise 3, BB raise 10", {"fold": 0.636, "call": 0.214, "raise": 0.150}, 0.04,
    GTOPRE, of="range", rake=RK)

# cold calls of a UTG (its LJ) open by the seats with players behind: "between 2.1% and 4.6% of hands",
# and they "3-bet more often than they call" (https://gtopreflop.com/posts/multiway-preflop-ranges)
for _h, _c in (("HJ", 0.021), ("CO", 0.029), ("BTN", 0.046), ("SB", 0.041)):
    ref(f"{_h} cold-calls UTG open", _h, "UTG raise 2.5", {"call": _c}, 0.015, GTOPRE, rake=RK, note="UTG = its LJ")

# ---- opening ranges: raked 2.5bb solves (SB 3bb), consistent across sources ----
OPENS = ("pokalab.com/tools/preflop-charts (GG NL50 rake, raise or fold) 17.6/22.3/29.1/43.6/43.5; "
         "Poker Academy GG NL200 18.5/-/-/46.5 (poker.academy/blog/post/what-is-gto-in-poker); "
         "thinkgto.com (GTO Ranges+) 18.7/22.9/29.4/43.3 and SB 33.1 raise + 20.5 limp; gtopreflop.com BTN 43.7")
for _p, _v, _t in (("UTG", 0.183, 0.015), ("HJ", 0.226, 0.015), ("CO", 0.293, 0.015), ("BTN", 0.445, 0.02),
                   ("SB", 0.445, 0.02)):
    ref(f"{_p} open (consensus)", _p, "", {"raise": _v}, _t, OPENS,
        note="SB: raise-only solves 43.5, raise + limp 45.0-45.7" if _p == "SB" else "")

# ---- other solves, conditions not matched exactly (reported, not enforced) ----
PA = "Poker Academy (GG NL200, 2.5bb opens, no ante; what-is-a-3-bet / what-is-gto posts)"
ref("BB vs BTN open (PA NL200)", "BB", "BTN raise 2.5", {"fold": 0.595, "call": 0.290, "raise": 0.115}, 0.03, PA,
    rake=RK, uncertain=True, note="GG NL200 rake, played at 5%/4bb")
ref("BTN vs CO open (PA NL200)", "BTN", "CO raise 2.5", {"fold": 0.845, "call": 0.030, "raise": 0.125}, 0.02, PA,
    uncertain=True)
ref("CO open vs BTN 3bet (PA NL200)", "CO", "CO raise 2.5, BTN raise 7.5",
    {"fold": 0.637, "call": 0.172, "raise": 0.191}, 0.04, PA, of="range", uncertain=True)
ref("BB vs UTG open (pokalab NL50)", "BB", "UTG raise 2.5", {"fold": 0.783, "call": 0.166, "raise": 0.051}, 0.03,
    "pokalab.com (GG NL50 rake, 3bet to 13bb)", rake=RK, uncertain=True)
ref("BB vs SB open (thinkgto)", "BB", "SB raise 3", {"fold": 0.445, "call": 0.387, "raise": 0.169}, 0.04,
    "thinkgto.com blind-vs-blind (GTO Ranges+, 3bet to 9bb, rake not stated)", uncertain=True)
ref("BB squeeze BTN open + SB call (PA NL200)", "BB", "BTN raise 2.5, SB call",
    {"fold": 0.712, "call": 0.161, "raise": 0.127}, 0.04, PA, rake=RK, uncertain=True,
    note="squeeze to 12.2bb; calls 16.1% at a 20% price against 29.0% heads-up at 27.3%")
ref("BTN cold vs HJ open + CO 3bet (GTOW NL500)", "BTN", "HJ raise 2.5, CO raise 7.5", {"fold": 0.935}, 0.035,
    "GTO Wizard NL500 100bb, blog.gtowizard.com/should-you-ever-cold-call-a-3-bet: 'more than 90%' to "
    "'just under 97%' folds", uncertain=True)
# the 3bettor facing a 4bet: one library export, BB's 3bet range (146.7 combos) against BTN's 24bb 4bet,
# with hero's As5s removed (so its AA/AK share is low): 46.9% fold, 29.3% call, 23.8% jam
# (gtogecko.com/blog/wheel-ace-fourbet-study, "catalogued rake 5%, capped at 4bb", solver not named,
# "not validated optimal responses")
ref("BB 3bet vs BTN 4bet to 24 (GTO Gecko)", "BB", "BTN raise 2.5, BB raise 12, BTN raise 24",
    {"fold": 0.469, "call": 0.293, "raise": 0.238}, 0.06, "GTO Gecko library export (5% capped at 4bb)",
    of="range", rake=RK, uncertain=True, note="card-removed with As5s; the only published 4bet response found")


# ---- hand-level facts: (name, hero, line, hands, action, low, high, source), on the chart's own line ----
HANDS = [
    ("UTG opens 44", "UTG", "", "44", "raise", 0.95, 1.0, "thinkgto.com (GTO Ranges+): 44+ always"),
    ("UTG opens 22 mostly", "UTG", "", "22", "raise", 0.6, 0.95, "thinkgto.com: 79%"),
    ("UTG opens AJo, KQo", "UTG", "", "AJo, KQo", "raise", 0.99, 1.0, "thinkgto.com"),
    ("UTG rarely opens KTo", "UTG", "", "KTo", "raise", 0.0, 0.25, "thinkgto.com: 11%"),
    ("BTN never opens A2o", "BTN", "", "A2o", "raise", 0.0, 0.05, "thinkgto.com"),
    ("BTN opens A4o", "BTN", "", "A4o", "raise", 0.6, 1.0, "thinkgto.com"),
    ("BB 3bets JJ+, AKo, AQs vs BTN", "BB", "BTN raise 2.5", "JJ+, AKo, AQs", "raise", 0.99, 1.0,
     "Poker Academy NL200, gtopreflop.com: always"),
    ("BB 3bets TT mostly vs BTN", "BB", "BTN raise 2.5", "TT", "raise", 0.55, 0.9, "gtopreflop.com: 67%"),
    ("BB calls A2s vs BTN", "BB", "BTN raise 2.5", "A2s", "call", 0.6, 1.0, "Poker Academy, gtopreflop.com"),
    ("BB 3bets A5o often vs BTN", "BB", "BTN raise 2.5", "A5o", "raise", 0.4, 0.9, "gtopreflop.com: 87%"),
    ("UTG flats QQ vs BTN 3bet", "UTG", "UTG raise 2.5, BTN raise 7.5", "QQ", "call", 0.7, 0.95, "gtopreflop.com: 85%"),
    ("CO 4bets A5s vs BTN 3bet", "CO", "CO raise 2.5, BTN raise 7.5", "A5s", "raise", 0.75, 1.0, "gtopreflop.com: 85%"),
    ("CO 4bets A4s about half vs BTN 3bet", "CO", "CO raise 2.5, BTN raise 7.5", "A4s", "raise", 0.35, 0.7,
     "gtopreflop.com: 55.9% (GTO Gecko's library 44%)"),
    ("BTN calls AQs vs SB 3bet", "BTN", "BTN raise 2.5, SB raise 12", "AQs", "call", 0.95, 1.0, "gtopreflop.com: 100%"),
    ("BTN 4bets A9s vs SB 3bet", "BTN", "BTN raise 2.5, SB raise 12", "A9s", "raise", 0.6, 0.9, "gtopreflop.com: 77.5%"),
]


def measure_hand(h) -> float:
    """The chart's frequency of `action` for the hand classes h[3] on the line h[2] (the chart itself)."""
    from gto import preflop as pf
    from gto.ranges import parse_range
    from gto.advisor import replay_hand
    _, hero, line, hand, action = h[:5]
    T = pf.Table(pf.SEATS[6])
    pre = _line(line, T)
    while (st := replay_hand(T, 100, {"preflop": pre}, 1)[0]).to_act != hero:
        pre.append({"pos": st.to_act, "action": "fold"})
    s = pf.strategy_for(hero, pre, T)
    m = parse_range(hand) > 0
    return float(s.get(action, np.zeros(pf.N_COMBOS))[m].mean())


# ---- hand-level facts of raked solves, played at their rake through the advice (re-pricing included):
# (name, hero, line, hands, action, low, high, source, rake); "continue" is call + raise ----
GTOPRE_BB = "gtopreflop.com/posts/big-blind-defense (5% capped at 4bb)"
GTOPRE_4B = "gtopreflop.com/posts/4-bet-ranges-guide (5% capped at 4bb)"
RAKED_HANDS = [
    ("BB folds A2o vs BTN (raked)", "BB", "BTN raise 2.5", "A2o", "continue", 0.0, 0.1, GTOPRE_BB + ": pure fold", RK),
    ("BB defends Q9o rarely vs BTN (raked)", "BB", "BTN raise 2.5", "Q9o", "continue", 0.1, 0.5,
     GTOPRE_BB + ": 27%, EV about 0", RK),
    ("BB defends K2s-K9s vs BTN (raked)", "BB", "BTN raise 2.5", "K2s-K9s", "continue", 0.95, 1.0,
     GTOPRE_BB + ": 100%", RK),
    ("BB calls A2s-A4s vs BTN (raked)", "BB", "BTN raise 2.5", "A2s-A4s", "call", 0.9, 1.0,
     GTOPRE_BB + ": call essentially always", RK),
    ("BB 3bets A5o mostly vs BTN (raked)", "BB", "BTN raise 2.5", "A5o", "raise", 0.4, 1.0, GTOPRE_BB + ": 87%", RK),
] + [(f"BTN {_w} vs SB 3bet (raked)", "BTN", "BTN raise 2.5, SB raise 12", _h, _a, _lo, _hi, GTOPRE_4B + _src, RK)
     for _w, _h, _a, _lo, _hi, _src in (
         ("4bets QQ", "QQ", "raise", 0.8, 1.0, ": 89.5%"),
         ("calls JJ, TT", "JJ, TT", "call", 0.65, 0.9, ": 76%, 81%"),
         ("calls AQs, KQs", "AQs, KQs", "call", 0.95, 1.0, ": 100%"),
         ("folds AJo, KQo mostly", "AJo, KQo", "fold", 0.55, 0.85, ": 77%, 67%"),
         ("4bets KTs mostly", "KTs", "raise", 0.6, 0.9, ": 75.8%"),
         ("folds K9s mostly", "K9s", "fold", 0.5, 0.8, ": 66%"),
     )]


def measure_raked_hand(h) -> float:
    """The advice's frequency of `action` for the hand classes h[3] on the line h[2] at the rake h[8]."""
    from gto import preflop as pf
    from gto.ranges import parse_range
    from gto.advisor import preflop_strategy, replay_hand
    _, hero, line, hand, action = h[:5]
    T = pf.Table(pf.SEATS[6], rake=h[8])
    pre = _line(line, T)
    while (st := replay_hand(T, 100, {"preflop": pre}, 1)[0]).to_act != hero:
        pre.append({"pos": st.to_act, "action": "fold"})
    s = preflop_strategy(hero, pre, T, 100)
    m = parse_range(hand) > 0
    v = 1.0 - s["fold"] if action == "continue" else s.get(action, np.zeros(pf.N_COMBOS))
    return float(v[m].mean())
