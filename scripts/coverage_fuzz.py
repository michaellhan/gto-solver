"""Coverage fuzz: random hands played the way the page plays them, over everything the page lets you set up, with
the advisor answering each decision as the browser's worker (or the local server) asks for it. It flags any answer
that isn't a usable, legal one, so a spot no other check reaches shows up here rather than at the table.

Setups (drawn per hand): 2-9 players; the table's stack 5-300bb (the common depths most often), each seat's own
stack (1-300bb, some very short) in a third of the hands; antes, a straddle, rake with and without a cap,
tournament payouts (ICM; some with a field at other tables); and the Advanced settings: exploit mode (opponents
"pool"), locks on opponents, custom bet sizes, custom opponent ranges, size optimization; now and then the page's
jam and breakdown requests.

Lines: each seat plays the hand in a style drawn for it ("advice": the advisor's own sampled move, most of the
time; "passive": limps and calls, so pots go 4-9 way; "aggro": raises of every size, min-raises to overbets and
all-ins, so 3bets to 5bets and cold 4bets; "nit"; "random"), entered as the page enters moves (index.html act:
amounts in half big blinds, held to the smallest raise as a practice opponent's are). Every decision of every seat
is asked (that seat as the hero, holding its own cards), so hero sits at every seat on every street. Some
opponents' moves go through Advisor.play (practice opponents) and some spots through Advisor.study.

Budgets and advisors (a "profile" per hand): the browser's worker (its Fast advisor as gto/static/worker.js builds
it, at Quick's budgets: index.html QUICK_BUDGETS), the local server's (gto/native.py advisors) at Quick's and at
Fast's 2.5 s. The browser's Fast (8 s) is left out: one hand there with three custom raise sizes passed 3 GB, and the
fuzz keeps under about 1 GB.

--harsh draws the odd setups far more often (mixed and very short stacks, antes, a straddle, 7-9 players, ICM,
locks, custom sizes and ranges, exploit mode). --standard SHARE plays that share of the hands in the flop library's
game on one of its stored preflop lines (gto/static/flop_library/index.json), mostly by the advice, so library_miss
has cases to check.

Flags (per decision):
  exception        advise, study or play raised
  error            an error answer: the fuzz sends only states the page can, so none is expected (Study's
                   refusals of 5+ way pots aside, which are counted as study_refused)
  no_rec           no actions, or a recommendation that isn't one of them
  nan              a number that isn't finite
  illegal          the recommended action can't be played as given (index.html act): a check facing a bet, a call
                   with nothing to call, a bet or raise below the smallest raise or not above the bet, a raise when
                   raising is closed (an all-in for less didn't reopen it, or nobody left has chips), an amount over
                   the stack, a call for the wrong amount
  illegal_option   the same for another action the answer gives a frequency
  free_fold        (odd) fold recommended with nothing to call
  fallback         (count) the catch-all answered (gto/advisor.py Advisor._rules): the heuristic's share
  slow             more than SLOW_X times the request's budget (index.html's estimate) plus SLOW_S seconds
  premium_fold     preflop: AA folds; KK to at most a 4bet; QQ or AKs to an open (not in tournaments)
  strong_fold      postflop: a set or better on an unpaired board folds to a bet (as scripts/sanity_sweep.py)
  play_bad         a practice opponent's move (Advisor.play) that's an error or can't be played
  library_miss     a heads-up flop decision the flop library stores that wasn't answered from it (library_expected:
                   the library's game, 6-max 100bb with nothing else set; a stored spot's preflop line, flop and node;
                   bets within 15% of its 33/75% sizes; the hero's hand in its stored range)
  library_slow     (count) a library answer that took over LIB_SLOW seconds (not counting the Breakdown view's
                   requests: its range equities take ~1 s on top)

Resumable and seeded: hand i plays from Random(f"{seed}:{i}"), its records are written when it ends (one line per
decision, then a "done" line), and a run skips hands already done in --out. Run it in chunks:

  python scripts/coverage_fuzz.py --hands 400 --minutes 25            # play (resumes), at most 25 minutes
  python scripts/coverage_fuzz.py --report                            # counts per flag and category, examples
  python scripts/coverage_fuzz.py --repro 17:3                        # hand 17's record 3 asked again

Results: .work/coverage/fuzz_<seed>.jsonl (--out).
"""
import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

SLOW_X, SLOW_S = 2.5, 4.0
QUICK_BUDGETS = {"flop": (1.5, 0.3), "turn": (1.5, 0.3), "river": (0.8, 0.3)}   # index.html
BUDGETS = {"server": {"quick": (1.5, 0.3), "fast": (2.5, 2.5)}, "wasm": {"quick": (1.5, 0.3), "fast": (8.0, 3.0)}}
# (the browser's Fast, 8 s on its Accurate-sized turn tree, is left out: with three custom raise sizes a hand's turn
# took 2.9 GB natively, over what a fuzz process may use here; its tree is Quick's, so Quick covers its paths)
PROFILES = [("wasm", "quick", 0.6), ("server", "quick", 0.25), ("server", "fast", 0.15)]
STYLES = [("advice", 0.45), ("passive", 0.2), ("aggro", 0.15), ("nit", 0.08), ("random", 0.12)]
RANGES = ["AA", "QQ+,AKs", "22+,A2s+,KTo+", "TT+,AQ+", "random", "77-22,A5s-A2s,KQo", "AsKs,AhKh", "JJ:0.5,AKo",
          "22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32+"]
PAYOUTS = [[50, 30, 20], [65, 35], [40, 25, 15, 10, 10], [100, 50], [30, 20, 15, 10, 8, 7, 5, 5], [3, 2, 1]]
LOCK_PRESETS = [{"fold": -20, "bluff": -50, "raise": -50}, {"fold": 15, "bluff": -60, "bet": -25, "raise": -30},
                {"fold": -10, "bluff": 50, "bet": 30, "raise": 60}, {"fold": 15}]
STUDY_OK = ("Study mode browses", "Study can't solve")
OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".work", "coverage", "fuzz_{seed}.jsonl")


# ------------------------------------------------------------------ advisors
def worker_advisors():
    """The browser's advisors as gto/static/worker.js builds them (without its core's helper threads), natively."""
    from gto.advisor import Advisor, MW_FAST_STREETS, MW4_FAST_STREETS
    from gto.solver import Config
    from gto import native, mw_library
    raises = {"flop": [0.5], "turn": [0.75], "river": [0.75]}
    fast = Advisor(Config(n_turns=3, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4, river_buckets=32,
                          turn_buckets=True, bucket_river_step=3, river_classes=True, river_strength=32,
                          bet_sizes={"flop": [0.33, 0.75], "turn": [0.33, 0.75, 1.25], "river": [0.33, 1.25]},
                          raise_sizes=dict(raises)))
    acc = Advisor(Config(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8, river_classes=True,
                         river_buckets=64, turn_buckets=True, bucket_river_step=1, river_strength=32,
                         bet_sizes={"flop": [0.33, 0.75], "turn": [0.33, 0.75, 1.25], "river": [0.33, 0.75, 1.25]},
                         raise_sizes=dict(raises)))
    for a in (fast, acc):
        a.root_sizes = {"river": {"bet": [0.33, 0.75, 1.25, 2.5]}}
        a.library = native.default_library()
        a.mw_library = mw_library.default_library()
        a.mw_solve_streets = ("flop", "turn", "river")
    acc.warm_from = fast
    fast.spare_prior = True
    fast.spr_sizes = {"turn": (4.5, {"bet": [0.33, 0.75], "river": [0.75]})}
    fast.mw_street_config = {s: dict(c) for s, c in MW_FAST_STREETS.items()}
    fast.mw4_street_config = {s: dict(c) for s, c in MW4_FAST_STREETS.items()}
    fast.mw4_quick_flop = None
    fast.mw_nway_max, acc.mw_nway_max = 5, 6
    acc._held = fast._held
    return {"fast": fast, "accurate": acc}


_ADVS = {}


def advisors(mode):
    if mode not in _ADVS:
        if mode == "wasm":
            _ADVS[mode] = worker_advisors()
        else:
            from gto import native
            _ADVS[mode] = native.advisors()
        for a in _ADVS[mode].values():
            a.cache_mb = 250      # (the fuzz keeps each process well under a GB)
    return _ADVS[mode]


# ------------------------------------------------------------------ the hand
def pick(rng, weighted):
    r = rng.random() * sum(w for *_, w in weighted)
    for item in weighted:
        r -= item[-1]
        if r <= 0:
            return item[:-1] if len(item) > 2 else item[0]
    return weighted[-1][:-1] if len(weighted[-1]) > 2 else weighted[-1][0]


HARSH = False   # (--harsh)
_LINE_RANGES = {}   # (--standard: a library line's two ranges entering the flop, by line)
STANDARD = 0.0  # (--standard: the share of hands on a flop library line)


def gen_setup(rng):
    from gto import preflop as pf
    if STANDARD and rng.random() < STANDARD:
        # the flop library's game on one of its stored lines (the opens the big blind calls, stored on every flop,
        # most often), played mostly by the advice
        _, by_line = library_spots()
        line = pick(rng, [(ln, 3 if sp.get("flops", 0) and len(sp["flops"]) >= 1755 else 1) for ln, sp in by_line.items()])
        mode, speed = pick(rng, PROFILES)
        return {"seats": list(pf.SEATS[6]), "table": {"players": 6, "stack": 100.0}, "adv": {}, "mode": mode,
                "speed": speed, "jam": False, "breakdown": rng.random() < 0.05, "line": line}
    hx = 2.5 if HARSH else 1.0   # (--harsh: the odd setups' shares, multiplied)
    n = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7] if HARSH else [2, 2, 3, 4, 5, 6, 6, 6, 6, 7, 8, 9, 9])
    seats = list(pf.SEATS[n])
    stack = float(rng.choice([100] * 6 + [20, 25, 40, 50, 60, 200, 300, 150, 250, 10, 15, 5]
                             + [round(rng.uniform(5, 300))] * 3))
    s = {"players": n, "stack": stack}
    if rng.random() < 0.33 * hx:
        st = {}
        for p in seats:
            st[p] = float(pick(rng, [(stack, 3), (round(rng.uniform(1, 300) * 2) / 2, 3),
                                     (rng.choice([1, 1.5, 2, 3, 4.5, 6, 8, 12]), 2), (round(rng.uniform(1, 30), 1), 1)]))
        if any(abs(v - stack) > 1e-9 for v in st.values()):
            s["stacks"] = st
    if rng.random() < 0.25 * hx:
        s["ante"] = rng.choice([0.1, 0.12, 0.125, 0.15, 0.2, 0.25, 0.5, 1.0])
    if n >= 3 and rng.random() < 0.15 * hx:
        s["straddle"] = True
    if rng.random() < 0.3:
        s["rake"] = rng.choice([2.5, 3, 4, 5, 5, 6, 7.5, 10]) / 100
        cap = rng.choice([0, 0, 0.5, 1, 2, 3, 4, 10])
        if cap:
            s["rake_cap"] = cap
    if rng.random() < 0.15 * hx:
        s["payouts"] = list(rng.choice(PAYOUTS))
        if rng.random() < 0.4:
            k = rng.randint(1, 120)
            s["field"] = {"players": k, "chips": k * rng.choice([20, 40, 60, 100])}
    adv = {}
    if rng.random() < 0.12 * hx:
        adv["opponents"] = "pool"
    if rng.random() < 0.12 * hx:
        adv["locks"] = {p: dict(rng.choice(LOCK_PRESETS)) if rng.random() < 0.7 else
                        {k: rng.randint(-60, 60) for k in rng.sample(["fold", "bluff", "bet", "raise", "fold_turn"], 2)}
                        for p in rng.sample(seats, rng.randint(1, min(3, n)))}
    if rng.random() < 0.10 * hx:
        adv["sizes"] = {k: sorted(rng.sample([10, 20, 25, 33, 50, 66, 75, 100, 125, 150, 200, 300], rng.randint(1, 3)))
                        for k in rng.sample(["flop", "turn", "river", "raise"], rng.randint(1, 4))}
    if rng.random() < 0.08 * hx:
        adv["ranges"] = {p: rng.choice(RANGES) for p in rng.sample(seats, rng.randint(1, min(2, n)))}
    if rng.random() < 0.04 * hx:
        adv["optimize_sizes"] = True
    mode, speed = pick(rng, PROFILES)
    return {"seats": seats, "table": s, "adv": adv, "mode": mode, "speed": speed,
            "jam": rng.random() < 0.05 * hx, "breakdown": rng.random() < 0.05 * hx}


def page_table(setup):
    """The setup as gto.preflop.Table and each seat's stack (as Advisor._advise reads the state)."""
    from gto import preflop as pf
    t = setup["table"]
    seats = setup["seats"]
    stacks = {p: float(t.get("stacks", {}).get(p, t["stack"])) for p in seats}
    T = pf.Table(seats, float(t.get("ante", 0.0)), bool(t.get("straddle")), None, t["stack"],
                 stacks if "stacks" in t else None)
    return T, stacks


def conv(acts):
    """index.html payload(): the page's recorded moves as it sends them."""
    out = []
    for a in acts:
        if a["action"] == "bet":
            out.append({"pos": a["pos"], "action": "bet", "amount": a["to"]})
        elif a["action"] == "raise":
            out.append({"pos": a["pos"], "action": "raise", "to": a["to"]})
        else:
            out.append({"pos": a["pos"], "action": a["action"]})
    return out


def page_state(setup, seat, hole, board, streets):
    """index.html payload() for `seat` as the hero."""
    t = setup["table"]
    st = {"players": t["players"], "hero": seat, "hole": hole, "board": board, "stack": t["stack"]}
    if "stacks" in t:
        st["stacks"] = {p: t["stacks"].get(p, t["stack"]) for p in setup["seats"]}
    for k in ("ante", "straddle", "rake", "rake_cap", "payouts", "field"):
        if k in t:
            st[k] = t[k]
    for s in ("preflop", "flop", "turn", "river"):
        st[s] = conv(streets[s])
    mode, speed = setup["mode"], setup["speed"]
    street = {0: None, 6: "flop", 8: "turn", 10: "river"}[len(board)]
    if street and speed == "quick":
        b, pb = QUICK_BUDGETS[street]
        st.update(budget=b, prior_budget=pb, speed="quick", quick=True)
    else:
        b, pb = BUDGETS[mode][speed]
        st.update(budget=b, prior_budget=pb, speed=speed)
    adv = setup["adv"]
    if adv.get("opponents"):
        st["opponents"] = adv["opponents"]
    if board:
        if adv.get("sizes"):
            st["sizes"] = adv["sizes"]
        r = {p: v for p, v in (adv.get("ranges") or {}).items() if p != seat}
        if r:
            st["ranges"] = r
        if adv.get("optimize_sizes"):
            st["optimize_sizes"] = True
        if setup.get("jam"):
            st["jam"] = True
        if setup.get("breakdown"):
            st["breakdown"] = True
    lk = {p: v for p, v in (adv.get("locks") or {}).items() if p != seat}
    if lk:
        st["locks"] = lk
    return st


def info_of(cur, p, stk):
    to_call = cur.cur_bet - cur.bets[p]
    remaining = stk[p] - cur.inv_start[p] - cur.bets[p]
    from gto.advisor import can_raise
    return {"to_call": to_call, "remaining": remaining, "facing": to_call > 1e-9, "raise_ok": can_raise(cur, p),
            "all_in_to": cur.bets[p] + remaining, "min_to": cur.cur_bet + cur.min_raise, "cur_bet": cur.cur_bet,
            "pre": cur.name == "preflop"}


def page_act(a, cur, p, stk, clamp=True):
    """index.html act(): the move as the page records it ({action, to}), or (None, why) where the page refuses
    it. `a`: {"action": fold|check|call|all-in|bet|raise, "to"}."""
    i = info_of(cur, p, stk)
    act = a["action"]
    if act == "fold":
        return {"action": "fold"}, None
    if act == "check":
        return ({"action": "check"}, None) if not i["facing"] else (None, "check facing a bet")
    if act == "call":
        return {"action": "call" if i["facing"] else "check"}, None
    if act == "all-in":
        if i["raise_ok"] or i["remaining"] <= i["to_call"] + 1e-9:
            return {"action": "all-in"}, None
        return {"action": "call" if i["facing"] else "check"}, None
    to = math.floor(float(a["to"]) * 2 + 0.5) / 2
    if not to > 0:
        return None, "no amount"
    if to <= i["cur_bet"] + 1e-9 and to < i["all_in_to"] - 1e-9:
        return None, "not above the bet"
    if not i["raise_ok"]:
        if not clamp:
            return None, "raising is closed"
        return {"action": "call" if i["facing"] else "check"}, None
    if to < i["min_to"] - 1e-9 and to < i["all_in_to"] - 1e-9:
        if not clamp:
            return None, "below the smallest raise"
        to = i["min_to"]
    if to >= i["all_in_to"] - 1e-9:
        return {"action": "all-in"}, None
    return {"action": "raise" if i["facing"] or i["cur_bet"] > 0 else "bet", "to": to}, None


def check_action(a, cur, p, stk):
    """Why the page can't play advice action `a` as given (None when it can)."""
    i = info_of(cur, p, stk)
    name, amt = a.get("action"), a.get("amount_bb")
    if name == "fold":
        return None
    if name == "check":
        return "check facing a bet" if i["facing"] else None
    if name == "call":
        if not i["facing"]:
            return "call with nothing to call"
        if amt is not None and abs(amt - min(i["to_call"], i["remaining"])) > 0.011:
            return f"call amount {amt} (to call {min(i['to_call'], i['remaining']):.2f})"
        return None
    if name == "all-in":
        if not i["raise_ok"] and i["remaining"] > i["to_call"] + 1e-9:
            return "all-in when raising is closed"
        if amt is not None and amt > i["all_in_to"] + 0.011:
            return f"all-in amount {amt} over the stack ({i['all_in_to']:.2f})"
        return None
    if not str(name).startswith(("bet", "raise")):
        return f"unknown action {name}"
    if amt is None:
        return "no amount"
    if str(name).startswith("bet") and i["facing"]:
        return "bet facing a bet"
    if amt > i["all_in_to"] + 0.011:
        return f"amount {amt} over the stack ({i['all_in_to']:.2f})"
    _, why = page_act({"action": "raise", "to": amt}, cur, p, stk, clamp=False)
    return why


def choose(rng, style, cur, p, stk, res, n_raises_cap=6):
    """A seat's move in its style, as {action, to}, before the page's act()."""
    i = info_of(cur, p, stk)
    if style == "advice" and res and res.get("actions") and rng.random() < 0.85:
        name = res.get("sampled") or res.get("recommended")
        a = next((x for x in res["actions"] if x["action"] == name), None)
        if a is not None:
            if name in ("fold", "check", "call", "all-in"):
                return {"action": name}
            if a.get("amount_bb") is not None:
                return {"action": "raise", "to": a["amount_bb"]}
    w = {"advice": (0.35, 0.45, 0.2), "passive": (0.12, 0.83, 0.05), "aggro": (0.1, 0.3, 0.6),
         "nit": (0.6, 0.3, 0.1), "random": (0.33, 0.34, 0.33)}.get(style, (0.33, 0.34, 0.33))
    fold_w, call_w, raise_w = w
    if not i["facing"]:
        fold_w = 0.01 if style == "random" else 0.0    # (the page lets you fold for free: now and then someone does)
    if not i["raise_ok"] or i["remaining"] <= i["to_call"] + 1e-9 or cur.n_raises >= n_raises_cap:
        raise_w = 0.02 if i["raise_ok"] and i["remaining"] > i["to_call"] + 1e-9 else 0.0
    kind = pick(rng, [("fold", fold_w), ("call", call_w), ("raise", raise_w)])
    if kind == "fold":
        return {"action": "fold"}
    if kind == "call":
        return {"action": "call" if i["facing"] else "check"}
    pot = cur.pot_start + sum(cur.bets.values())
    size = pick(rng, [("min", 2), ("std", 4), ("pot", 3), ("over", 1.5), ("allin", 1.5), ("odd", 1)])
    if size == "allin":
        return {"action": "all-in"}
    if size == "min":
        to = i["min_to"]
    elif size == "std":
        to = (cur.cur_bet * rng.choice([2.0, 2.2, 2.5, 3.0, 4.0]) if i["pre"] else
              cur.cur_bet + rng.choice([0.25, 0.33, 0.5, 0.66, 0.75]) * (pot + i["to_call"]))
    elif size == "pot":
        to = cur.cur_bet + rng.choice([0.75, 1.0]) * (pot + i["to_call"])
    elif size == "over":
        to = cur.cur_bet + rng.choice([1.5, 2.0, 3.0, 5.0]) * (pot + i["to_call"])
    else:
        to = i["min_to"] + rng.uniform(0, max(1.0, 0.5 * pot))
    return {"action": "raise", "to": max(to, 0.5)}


def tags_of(setup, cur, p, rep, n_board, streets):
    t, adv = setup["table"], setup["adv"]
    pre = rep[0]
    n_raises_pre = pre.n_raises
    limped = any(a["action"] == "call" for a in streets["preflop"]) and \
        not any(a["action"] in ("raise", "all-in") for a in streets["preflop"][:next(
            (k for k, a in enumerate(streets["preflop"]) if a["action"] == "call"), 0)])
    allin_short = [q for q in cur.active if q in cur.allin]
    return {"players": t["players"], "street": cur.name, "in_pot": len(cur.active),
            "start": len(cur.players_start), "stack": t["stack"], "mixed": "stacks" in t,
            "short": min(t.get("stacks", {}).values(), default=t["stack"]) < 12 or t["stack"] < 12,
            "ante": "ante" in t, "straddle": bool(t.get("straddle")), "rake": "rake" in t, "icm": "payouts" in t,
            "pool": bool(adv.get("opponents")), "locks": bool(adv.get("locks")), "sizes": bool(adv.get("sizes")),
            "ranges": bool(adv.get("ranges")), "optimize": bool(adv.get("optimize_sizes")),
            "mode": setup["mode"], "speed": setup["speed"], "pre_raises": min(n_raises_pre, 5),
            "limped": limped, "allin_in": len(allin_short), "facing": cur.cur_bet > cur.bets[p] + 1e-9,
            "locked_out": p in cur.locked and cur.cur_bet > cur.bets[p] + 1e-9}


def hand_rank(hole, board):
    from treys import Card, Evaluator
    ev = Evaluator()
    h = [Card.new(hole[:2]), Card.new(hole[2:])]
    bd = [Card.new(board[i:i + 2]) for i in range(0, len(board), 2)]
    ranks = [board[i] for i in range(0, len(board), 2)]
    return ev.get_rank_class(ev.evaluate(bd, h)), len(set(ranks)) < len(ranks)


def finite(res):
    for a in res.get("actions") or []:
        for k in ("freq", "ev_bb", "amount_bb", "gto_freq"):
            v = a.get(k)
            if isinstance(v, (int, float)) and not math.isfinite(v):
                return False
    for k in ("pot_bb", "to_call_bb", "equity_vs_field"):
        v = res.get(k)
        if isinstance(v, (int, float)) and not math.isfinite(v):
            return False
    return True


def flags_of(res, cur, p, stk, state, est, secs):
    """The flags of one advise answer."""
    from gto.cards import cards, hand_class
    fl = []
    if "error" in res:
        return ["error"]
    if "skipped" in res or "cancelled" in res:
        return ["error"]
    acts = res.get("actions") or []
    names = [a.get("action") for a in acts]
    if not acts or res.get("recommended") not in names:
        fl.append("no_rec")
    if not finite(res):
        fl.append("nan")
    rec = next((a for a in acts if a.get("action") == res.get("recommended")), None)
    if rec is not None:
        why = check_action(rec, cur, p, stk)
        if why:
            fl.append("illegal")
        if rec["action"] == "fold" and not cur.cur_bet > cur.bets[p] + 1e-9:
            fl.append("free_fold")
    for a in acts:
        if a is not rec and (a.get("freq") or 0) > 0 and check_action(a, cur, p, stk):
            fl.append("illegal_option")
            break
    if res.get("fallback"):
        fl.append("fallback")
    if secs > SLOW_X * est + SLOW_S:
        fl.append("slow")
    if rec is not None and rec["action"] == "fold":
        hole = state["hole"]
        if not state["board"]:
            hc = hand_class(tuple(cards(hole)))
            nr = cur.n_raises
            if "payouts" not in state and (hc == "AA" or (hc == "KK" and nr <= 3) or (hc in ("QQ", "AKs") and nr <= 1)):
                fl.append("premium_fold")
        elif cur.cur_bet > cur.bets[p] + 1e-9:
            cls, paired = hand_rank(hole, state["board"])
            if cls <= 6 and not paired:
                fl.append("strong_fold")
    return fl


LIB_SLOW = 0.2   # library_slow: seconds
_LIB = []
# a request with any of these set isn't in the library's game (or the advisor solves it live on purpose: exploit mode,
# a lock, custom sizes, size optimization, a jam request, a tournament)
NOT_LIBRARY = ("stacks", "ante", "straddle", "rake", "rake_cap", "payouts", "field", "opponents", "locks", "sizes",
               "ranges", "optimize_sizes", "jam")


def library_spots():
    """(the flop library, its spots by preflop line), read once."""
    if not _LIB:
        from gto import flop_library as fl
        lib = fl.default_library()
        idx = lib.index() if lib is not None else {}
        _LIB.extend([lib, {sp["line"]: sp for sp in idx.get("spots", [])}])
    return _LIB


def library_expected(state):
    """Where the flop library should answer this request from ("<spot> <flop> at <node path>"), or None: a heads-up
    flop decision in the library's game (6-max, 100bb, nothing in NOT_LIBRARY set) on a stored spot's preflop line
    and a stored flop, at a node the entry stores (checks, and bets within 15% of its 33/75% sizes), with the hero's
    hand in the entry's range. Read from the index and the entry's file directly, not through the advisor's lookup."""
    from gto import flop_library as fl
    from gto.cards import cards, cards_str, COMBO_INDEX
    if len(state.get("board") or "") != 6 or int(state.get("players", 6)) != 6 or float(state["stack"]) != 100.0 \
            or any(state.get(k) for k in NOT_LIBRARY):
        return None
    lib, by_line = library_spots()
    code = {"fold": "f", "call": "c", "check": "x"}
    line = ",".join(f"{a['pos']}:{code[a['action']]}" if a["action"] in code else
                    f"{a['pos']}:r{a.get('to', a.get('amount')):g}" if a["action"] == "raise" else f"{a['pos']}:?"
                    for a in state.get("preflop") or [])
    sp = by_line.get(line)
    if sp is None:
        return None
    board = cards(state["board"])
    canon, smap = fl.canonical(board)
    name = cards_str(canon)
    if name not in sp.get("flops", {}):
        return None
    e = lib.entry(sp, name)
    if e is None:
        return f"{sp['id']} {name}: stored, but its entry can't be read"
    pot, path = float(sp["pot"]), []
    for a in state.get("flop") or []:
        if a["action"] == "check":
            path.append("check")
        elif a["action"] == "bet":
            frac = float(a.get("amount", a.get("to"))) / pot
            near = min((0.33, 0.75), key=lambda x: abs(x - frac))
            if abs(near - frac) / near > 0.15:
                return None
            path.append(f"bet {int(round(near * 100))}%")
        else:
            return None   # (raises: not checked here)
    nd = next((x for x in e["nodes"] if list(x["path"]) == path), None)
    seats = e["head"]["seats"]
    if nd is None or state["hero"] not in seats or seats[nd["player"]] != state["hero"]:
        return None
    hole = cards(state["hole"])
    if not e["masks"][nd["player"]][fl.combo_map(smap)[COMBO_INDEX[(min(hole), max(hole))]]]:
        return None   # (the hand is off its range here: the advisor solves it live, as a near-zero-weight combo)
    return f"{sp['id']} {name} at {'/'.join(path) or 'the root'}"


def spot_of(state):
    """(street state at the decision, each seat's starting stack) of a page request (page_state's shape)."""
    from gto import preflop as pf
    from gto.advisor import replay_hand, _table_stacks
    stack = float(state["stack"])
    T = pf.Table(pf.SEATS[int(state.get("players", 6))], float(state.get("ante", 0.0)), bool(state.get("straddle")),
                 None, stack, state.get("stacks"))
    streets = {s: [dict(a, **({"to": a["amount"]} if "amount" in a else {})) for a in state.get(s) or []]
               for s in ("preflop", "flop", "turn", "river")}
    return replay_hand(T, stack, streets, {0: 1, 6: 2, 8: 3, 10: 4}[len(state.get("board") or "")])[-1], \
        _table_stacks(T, stack)


def state_flags(res, state, secs=0.0, est=None):
    """flags_of for a page request on its own (scripts/sanity_sweep.py's exotic spots, tests/test_coverage.py):
    the hard flags only, plus "slow" when `est` is given."""
    cur, stk = spot_of(state)
    fl = flags_of(res, cur, state["hero"], stk, state, est if est is not None else float("inf"), secs)
    return [f for f in fl if f not in SOFT or (f == "slow" and est is not None)]


def illegal_detail(res, cur, p, stk):
    out = {}
    for a in res.get("actions") or []:
        why = check_action(a, cur, p, stk)
        if why:
            out[a["action"]] = why
    return out


def estimate(state, n_in):
    """index.html analyze()'s estimate of the request's time (budget + prior budget per earlier postflop street)."""
    st = {0: 0, 6: 1, 8: 2, 10: 3}[len(state["board"])]
    if st == 0:
        return 0.5
    est = state["budget"] + state["prior_budget"] * max(0, st - 1)
    if state.get("speed") == "fast" and n_in >= 3:
        est = max(est, 5.0)
    if state.get("optimize_sizes"):
        est *= 1.5
    return est


def peak_mb():
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)   # (bytes on macOS)


def drop_caches():
    """The advisors' kept solves, between hands (a hand's solves serve only its own later streets)."""
    for advs in _ADVS.values():
        for a in advs.values():
            a._cache.clear(); a._mw_cache.clear(); a._by_spot.clear(); a._mw_lines.clear(); a._held.clear()


def call(fn, *a):
    t = time.time()
    try:
        return fn(*a), None, time.time() - t
    except Exception as e:   # noqa: BLE001 - the fuzz reports everything
        return None, {"exc": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}, time.time() - t


def play_hand(seed, h):
    """Plays hand h; returns its records (one per request, then a done record)."""
    from gto.advisor import replay_hand, _table_stacks
    from gto.cards import card_str
    rng = random.Random(f"{seed}:{h}")
    setup = gen_setup(rng)
    advs = advisors(setup["mode"])
    seats = setup["seats"]
    T, _ = page_table(setup)
    stack = setup["table"]["stack"]
    stk = _table_stacks(T, stack)
    deck = list(range(52))
    rng.shuffle(deck)
    if setup.get("line"):
        # a library line's two players hold hands from their ranges entering the flop (as the charts play the line),
        # so their flop decisions are ones the library stores; the rest of the deck as dealt
        from gto import flop_library as fl
        from gto.cards import ALL_COMBOS
        if setup["line"] not in _LINE_RANGES:
            sp = fl.spot_setup(setup["line"])
            _LINE_RANGES[setup["line"]] = {sp["oop"]: sp["full0"], sp["ip"]: sp["full1"]}
        held = {}
        for p_, w in _LINE_RANGES[setup["line"]].items():
            used = {c for cb in held.values() for c in cb}
            w = np.array([0.0 if used & set(cb) else x for cb, x in zip(ALL_COMBOS, w)])
            held[p_] = ALL_COMBOS[int(np.searchsorted(np.cumsum(w), rng.random() * w.sum()))]
        taken = {c for cb in held.values() for c in cb}
        deck = list(held[p_][i] for p_ in held for i in (0, 1)) + [c for c in deck if c not in taken]
        order = list(held) + [p_ for p_ in seats if p_ not in held]
        holes = {p_: card_str(deck[2 * k]) + card_str(deck[2 * k + 1]) for k, p_ in enumerate(order)}
    else:
        holes = {p: card_str(deck[2 * k]) + card_str(deck[2 * k + 1]) for k, p in enumerate(seats)}
    board_all = "".join(card_str(c) for c in deck[2 * len(seats):2 * len(seats) + 5])
    styles = {p: "advice" if setup.get("line") and rng.random() < 0.8 else pick(rng, STYLES) for p in seats}
    script = []   # (a library line's preflop moves, played as they come)
    if setup.get("line"):
        for x in setup["line"].split(","):
            pos, c = x.split(":")
            script.append({"pos": pos, **({"action": "raise", "to": float(c[1:])} if c.startswith("r") else
                                          {"action": {"f": "fold", "c": "call", "x": "check"}[c]})})
    streets = {"preflop": [], "flop": [], "turn": [], "river": []}
    n_board, k, recs = 0, 0, []
    t_hand = time.time()
    study_at = rng.random() < 0.25
    for _ in range(80):
        rep = replay_hand(T, stack, streets, {0: 1, 3: 2, 4: 3, 5: 4}[n_board])
        cur = rep[-1]
        if len(cur.active) <= 1:
            break
        if cur.closed:
            if n_board == 5:
                break
            n_board = 3 if n_board == 0 else n_board + 1
            continue
        p = cur.to_act
        board = board_all[:2 * n_board]
        state = page_state(setup, p, holes[p], board, streets)
        rec = {"hand": h, "k": k, "kind": "advise", "tags": tags_of(setup, cur, p, rep, n_board, streets),
               "style": styles[p]}
        adv = advs.get(state.get("speed"), advs["fast"])
        res, exc, secs = call(adv.advise, state)
        est = estimate(state, len(cur.active))
        rec.update(seconds=round(secs, 3), est=est, rss_mb=peak_mb())
        if exc:
            rec.update(flags=["exception"], **exc)
        else:
            rec["flags"] = flags_of(res, cur, p, stk, state, est, secs)
            rec["rec"] = res.get("recommended")
            rec["actions"] = [(a.get("action"), a.get("freq"), a.get("ev_bb"), a.get("amount_bb")) for a in res.get("actions") or []]
            for key in ("error", "fallback", "fallback_reason", "fallback_where", "skipped"):
                if key in res:
                    rec[key] = res[key]
            if "illegal" in rec["flags"] or "illegal_option" in rec["flags"]:
                rec["illegal"] = illegal_detail(res, cur, p, stk)
            rec["by"] = ("fallback-" + res["fallback"] if res.get("fallback") else "library" if res.get("precomputed")
                         else "chart" if cur.name == "preflop" else "rules" if res.get("multiway") and not res.get("solved")
                         else "solve")
            lib_at = library_expected(state)
            if lib_at:
                rec["lib_expected"] = lib_at
                if rec["by"] != "library":
                    rec["flags"].append("library_miss")
            if rec["by"] == "library" and secs > LIB_SLOW and not state.get("breakdown"):
                rec["flags"].append("library_slow")
        rec["state"] = state
        recs.append(rec)
        k += 1
        # Study, now and then, on the hand as it stands (index.html's payload without hero, hole cards and the advice's
        # own settings: study.js fromPayload; no budget, as the page sends none)
        if study_at and rng.random() < 0.3:
            sst = {x: y for x, y in state.items() if x in ("players", "stack", "stacks", "ante", "straddle", "rake",
                                                            "rake_cap", "board", "preflop", "flop", "turn", "river")}
            sst.update(study=True, speed="fast", budget=state["budget"], prior_budget=state["prior_budget"])
            sres, sexc, ssecs = call(adv.study, sst)
            srec = {"hand": h, "k": k, "kind": "study", "tags": rec["tags"], "seconds": round(ssecs, 3), "state": sst}
            if sexc:
                srec.update(flags=["exception"], **sexc)
            elif "error" in sres:
                srec.update(flags=["study_refused" if str(sres["error"]).startswith(STUDY_OK) else "error"],
                            error=sres["error"], **{x: sres[x] for x in ("fallback_reason", "fallback_where") if x in sres})
            elif sres.get("study") not in ("node", "deal", "over"):
                srec.update(flags=["no_rec"], got=str(sres)[:300])
            else:
                srec["flags"] = (["nan"] if not finite(sres) else []) + (["fallback"] if sres.get("fallback") else [])
                srec["study"] = sres.get("study")
            recs.append(srec)
            k += 1
            study_at = False
        # the move: a practice opponent's through Advisor.play now and then, else the seat's style
        move = None
        if rng.random() < 0.12 and not exc:
            pres, pexc, psecs = call(adv.play, {x: y for x, y in state.items()
                                               if x not in ("hero", "hole")} | {"hero": seats[(seats.index(p) + 1) % len(seats)],
                                                                                "hole": holes[seats[(seats.index(p) + 1) % len(seats)]]},
                                     p, holes[p], h * 100 + k)
            prec = {"hand": h, "k": k, "kind": "play", "tags": rec["tags"], "seconds": round(psecs, 3), "flags": []}
            if pexc:
                prec.update(flags=["exception"], **pexc)
            elif "error" in pres or pres.get("action") not in ("fold", "check", "call", "bet", "raise", "all-in"):
                prec.update(flags=["play_bad"], got=str(pres)[:300])
            else:
                move = {"action": pres["action"], **({"to": pres["to"]} if "to" in pres else {})}
                if move["action"] in ("bet", "raise"):
                    move["action"] = "raise"
                _, why = page_act(move, cur, p, stk, clamp=True)
                if why:
                    prec.update(flags=["play_bad"], got=str(pres)[:300], why=why)
                    move = None
            if prec["flags"]:
                prec["state"] = state
                recs.append(prec)
                k += 1
        if script and cur.name == "preflop" and script[0]["pos"] == p:
            move = {k_: v_ for k_, v_ in script.pop(0).items() if k_ != "pos"}
        if move is None:
            move = choose(rng, styles[p], cur, p, stk, res)
        a, why = page_act(move, cur, p, stk, clamp=True)
        if a is None:
            a = {"action": "call" if cur.cur_bet > cur.bets[p] + 1e-9 else "check"}
        streets[cur.name].append({"pos": p, **a})
    recs.append({"hand": h, "done": True, "setup": {x: y for x, y in setup.items() if x != "seats"},
                 "seconds": round(time.time() - t_hand, 1), "decisions": sum(1 for r in recs if r.get("kind") == "advise"),
                 "board": board_all[:2 * n_board], "lines": {s: conv(v) for s, v in streets.items()}})
    return recs


# ------------------------------------------------------------------ runs and the report
def load(path):
    recs, done = [], set()
    if not os.path.exists(path):
        return recs, done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            recs.append(r)
            if r.get("done"):
                done.add(r["hand"])
    return [r for r in recs if r["hand"] in done], done


def run(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    _, done = load(args.out)
    t0 = time.time()
    n_new = 0
    for h in range(args.start, args.start + args.hands):
        if h in done:
            continue
        if args.minutes and time.time() - t0 > args.minutes * 60:
            print(f"time's up after {n_new} hands", flush=True)
            break
        recs = play_hand(args.seed, h)
        drop_caches()
        with open(args.out, "a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        n_new += 1
        bad = [f"{r['k']}:{','.join(x for x in r['flags'] if x not in ('fallback',))}" for r in recs
               if r.get("flags") and set(r["flags"]) - {"fallback", "slow", "study_refused", "free_fold"}]
        print(f"hand {h}: {recs[-1]['decisions']} decisions, {recs[-1]['seconds']}s"
              + (f"  FLAGS {' '.join(bad)}" if bad else ""), flush=True)
    print(f"{n_new} hands in {time.time() - t0:.0f}s", flush=True)


SOFT = {"fallback", "slow", "study_refused", "free_fold", "strong_fold", "library_slow"}


def report(path, out=None, examples=3):
    recs, done = load(path)
    dec = [r for r in recs if not r.get("done")]
    hands = [r for r in recs if r.get("done")]
    by_kind = Counter(r["kind"] for r in dec)
    flags = Counter(f for r in dec for f in r.get("flags", []))
    lines = [f"# Coverage fuzz: {os.path.basename(path)}", "",
             f"{len(hands)} hands, {len(dec)} requests ({', '.join(f'{k} {v}' for k, v in by_kind.items())}).", ""]
    adv = [r for r in dec if r["kind"] == "advise"]
    by = Counter(r.get("by", "-") for r in adv)
    lines += ["Answered by: " + ", ".join(f"{k} {v} ({v / max(len(adv), 1):.1%})" for k, v in by.most_common()), ""]
    exp = [r for r in adv if r.get("lib_expected")]
    lib_secs = sorted(r["seconds"] for r in adv if r.get("by") == "library" and not r["state"].get("breakdown"))
    lines += [f"Flop library: {len(exp)} decisions it stores (library_expected), {sum(r.get('by') == 'library' for r in exp)} "
              f"answered from it; {len(lib_secs)} library answers in all (the Breakdown view's left out)"
              + (f", seconds median {lib_secs[len(lib_secs) // 2]:.3f} / 95th {lib_secs[int(len(lib_secs) * 0.95)]:.3f} / "
                 f"max {lib_secs[-1]:.3f}, {sum(x > LIB_SLOW for x in lib_secs)} over {LIB_SLOW} s" if lib_secs else "")
              + ".", ""]
    lines += ["| flag | count | share of advise |", "|---|---|---|"]
    for f, c in flags.most_common():
        lines.append(f"| {f}{' (soft)' if f in SOFT else ''} | {c} | {c / max(len(adv), 1):.2%} |")
    # coverage per category
    cats = ["players", "street", "in_pot", "pre_raises", "mixed", "short", "ante", "straddle", "rake", "icm", "pool",
            "locks", "sizes", "ranges", "optimize", "limped", "allin_in", "facing", "locked_out", "mode", "speed"]
    lines += ["", "## Coverage (advise requests; flagged = any flag but the soft ones)", ""]
    for c in cats:
        cnt, bad = Counter(), Counter()
        for r in adv:
            v = r["tags"].get(c)
            cnt[v] += 1
            if set(r.get("flags", [])) - SOFT:
                bad[v] += 1
        lines.append(f"- {c}: " + ", ".join(f"{k}={cnt[k]}" + (f" ({bad[k]} flagged)" if bad[k] else "")
                                            for k in sorted(cnt, key=lambda x: (str(type(x)), x))))
    secs = defaultdict(list)
    for r in adv:
        secs[(r["tags"]["street"], r["tags"]["speed"])].append(r["seconds"])
    lines += ["", "## Seconds per answer (median / 95th / max)", ""]
    for key in sorted(secs):
        v = sorted(secs[key])
        lines.append(f"- {key[0]} {key[1]}: {v[len(v) // 2]:.2f} / {v[int(len(v) * 0.95)]:.2f} / {v[-1]:.2f} ({len(v)})")
    reasons = Counter((r.get("fallback_reason", "")[:90], r.get("fallback_where", "")) for r in dec if "fallback" in r.get("flags", []))
    if reasons:
        lines += ["", "## Fallback reasons", ""] + [f"- {c} x {a} @ {b}" for (a, b), c in reasons.most_common(30)]
    for f, _ in flags.most_common():
        ex = [r for r in dec if f in r.get("flags", [])][:examples]
        lines += ["", f"## {f}", ""]
        for r in ex:
            s = r.get("state", {})
            head = (f"- hand {r['hand']}:{r['k']} {r['kind']} {r['tags']['players']}p {r['tags']['street']} "
                    f"{r['tags']['in_pot']}-way {s.get('hero', '')} {s.get('hole', '')} {s.get('board', '')}")
            rest = {x: r[x] for x in ("rec", "by", "lib_expected", "error", "exc", "fallback_reason", "fallback_where",
                                      "illegal", "got", "why", "seconds", "est") if x in r}
            lines.append(head + "  \n  `" + json.dumps(rest, default=str)[:600] + "`")
            if r.get("tb"):
                lines.append("  ```\n  " + r["tb"].strip().replace("\n", "\n  ")[-900:] + "\n  ```")
            lines.append("  state: `" + json.dumps(s, default=str)[:1200] + "`")
    text = "\n".join(lines) + "\n"
    if out:
        with open(out, "w") as f:
            f.write(text)
    return text, flags, len(adv)


def repro(path, h, k):
    """The request of hand h's k-th record in `path` again, its answer printed."""
    with open(path) as f:
        r = next((x for x in map(json.loads, f) if x.get("hand") == h and x.get("k") == k and "state" in x), None)
    if r is None:
        sys.exit(f"no record {h}:{k} with a state in {path}")
    st = r["state"]
    advs = advisors("wasm" if "wasm" in json.dumps(r["tags"].get("mode")) else "server")
    adv = advs.get(st.get("speed"), advs["fast"])
    print(json.dumps(st))
    t = time.time()
    res = adv.study(st) if r["kind"] == "study" else adv.advise(st)
    print(json.dumps(res, indent=1, default=str)[:8000])
    print(f"{time.time() - t:.2f}s; recorded flags {r.get('flags')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hands", type=int, default=300, help="hand indices start..start+hands-1 (done ones skipped)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (resume later)")
    ap.add_argument("--out", default=None, help="jsonl (default .work/coverage/fuzz_<seed>.jsonl)")
    ap.add_argument("--report", nargs="?", const="-", default=None, help="summarize --out (to this file, or stdout)")
    ap.add_argument("--harsh", action="store_true", help="the odd setups far more often")
    ap.add_argument("--standard", type=float, default=0.0, help="this share of hands on the flop library's lines")
    ap.add_argument("--repro", default=None, help="HAND:K: ask hand HAND's record K again (its stored state), verbose")
    args = ap.parse_args()
    args.out = args.out or OUT_DEFAULT.format(seed=args.seed)
    global HARSH, STANDARD
    HARSH, STANDARD = args.harsh, args.standard
    if args.report:
        text, _, _ = report(args.out, None if args.report == "-" else args.report)
        print(text if args.report == "-" else text.split("\n## Coverage")[0])
        return
    if args.repro:
        h, k = (int(x) for x in args.repro.split(":"))
        repro(args.out, h, k)
        return
    run(args)


if __name__ == "__main__":
    main()
