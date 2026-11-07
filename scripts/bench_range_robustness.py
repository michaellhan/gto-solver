#!/usr/bin/env python3
"""Hands off their range: how often does the answer change with the ranges assumed, and what does the change cost?

Deals spots from a spread of preflop lines at a seeded random, keeps the ones whose answer says "off_range", and answers
each as the page does: the main answer (the pool, as a cash hand), then the chart and the wider range in turn; the wider
range against the pool too, for comparison. One line of JSON per spot, resumable: spots already in OUT are skipped.

    python scripts/bench_range_robustness.py OUT.jsonl [--spots 48] [--seed 7] [--threads 1] [--in-range]

--in-range is the control: spots whose hand is in its range, the main answer and the chart's alone, for what the two
ranges' difference is worth where no hand is off anything. Then --report OUT.jsonl prints the table. Each spot's regret under an assumption is what the main answer's action gives
up there: the best action's EV less its EV under that assumption, in bb and % of the pot.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RANKS, SUITS = "23456789TJQKA", "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]


def C(p): return {"pos": p, "action": "call"}
def F(*ps): return [{"pos": p, "action": "fold"} for p in ps]
def R(p, to): return {"pos": p, "action": "raise", "to": to}


# name, players, preflop line, who's still in (postflop order), hero's seats
SEVEN = [C("UTG"), R("LJ", 4)] + [C("HJ")] + F("CO", "BTN", "SB") + [C("BB"), C("UTG")]
LINES = [
    ("BTN v BB", 6, F("UTG", "HJ", "CO") + [R("BTN", 2.5)] + F("SB") + [C("BB")], ["BB", "BTN"], ["BB", "BTN"]),
    ("CO v BTN", 6, F("UTG", "HJ") + [R("CO", 2.5), C("BTN")] + F("SB", "BB"), ["CO", "BTN"], ["BTN"]),
    ("UTG v BB", 6, [R("UTG", 2.5)] + F("HJ", "CO", "BTN", "SB") + [C("BB")], ["BB", "UTG"], ["BB"]),
    ("SB limp", 6, F("UTG", "HJ", "CO", "BTN") + [C("SB"), {"pos": "BB", "action": "check"}], ["SB", "BB"], ["BB", "SB"]),
    ("3bet", 6, F("UTG", "HJ", "CO") + [R("BTN", 2.5), R("SB", 11)] + F("BB") + [C("BTN")], ["SB", "BTN"], ["BTN"]),
    ("3-way", 6, F("UTG", "HJ") + [R("CO", 2.5), C("BTN")] + F("SB") + [C("BB")], ["BB", "CO", "BTN"], ["BB", "BTN"]),
    ("limp raise", 7, SEVEN, ["BB", "UTG", "LJ", "HJ"], ["BB"]),
]
THE_HAND = {"name": "the hand", "players": 7, "hero": "BB", "hole": "As4s", "board": "2s4d6c6h", "stack": 100,
            "preflop": SEVEN, "turn": [], "river": [],
            "flop": [{"pos": "BB", "action": "bet", "amount": 6.5}] + F("UTG") + [C("LJ")] + F("HJ")}


def street_line(rng, order, hero, pot, last):
    """A street's actions among `order`: closed when not `last`, else up to hero's decision (None: can't reach it)."""
    frac = rng.choice([0.33, 0.75])
    if not last:
        kind = rng.choice(["checks", "checks", "bet"])
        if kind == "checks":
            return [{"pos": p, "action": "check"} for p in order]
        b = order[rng.randrange(len(order))]
        i = order.index(b)
        amt = round(pot * frac, 1)
        return [{"pos": p, "action": "check"} for p in order[:i]] + [{"pos": b, "action": "bet", "amount": amt}] + \
            [C(p) for p in order[i + 1:] + order[:i]]
    i = order.index(hero)
    if rng.random() < 0.5 or i == 0:          # checked to hero (or hero first to act)
        if i == 0 and rng.random() < 0.5 and len(order) > 1:   # hero checks, the next bets, back to hero
            return [{"pos": hero, "action": "check"}, {"pos": order[1], "action": "bet", "amount": round(pot * frac, 1)}] + \
                [C(p) for p in order[2:]]
        return [{"pos": p, "action": "check"} for p in order[:i]]
    return [{"pos": p, "action": "check"} for p in order[:i - 1]] + \
        [{"pos": order[i - 1], "action": "bet", "amount": round(pot * frac, 1)}]


def playable(hole) -> bool:
    """The stronger half of hands: a hand someone plays off the chart, not any two cards."""
    from gto import preflop as pf
    from gto.cards import COMBO_INDEX, card
    a, b = sorted((card(hole[:2]), card(hole[2:])))
    return pf._strength()[COMBO_INDEX[(a, b)]] >= 0.50


def deal(rng, n):
    name, players, pre, order, heroes = LINES[n % len(LINES)]
    hero = rng.choice(heroes)
    while True:
        cards = rng.sample(DECK, 7)
        if playable(cards[0] + cards[1]):
            break
    hole, board = cards[0] + cards[1], cards[2:7]
    street = rng.choice(["flop", "turn", "river"])
    n = {"flop": 3, "turn": 4, "river": 5}[street]
    st = {"name": name, "players": players, "hero": hero, "hole": hole, "board": "".join(board[:n]), "stack": 100,
          "preflop": pre, "flop": [], "turn": [], "river": []}
    if name == "limp raise":   # (4-way flop: everyone in, the bet folds two out, as the hand did)
        order = ["BB", "UTG", "LJ", "HJ"]
    from gto import advisor as A, preflop as pf
    pos = list(pf.SEATS[players])
    for s in ["flop", "turn", "river"][:["flop", "turn", "river"].index(street) + 1]:
        k = 2 + ["flop", "turn", "river"].index(s)
        pot = A.replay_hand(pos, 100.0, {q: st[q] for q in ("preflop", "flop", "turn", "river")}, k)[-1].pot_start
        last = s == street
        st[s] = street_line(rng, order, hero, pot, last)
    return st


def outcome(adv, st):
    t = time.time()
    r = adv.advise(st)
    sec = time.time() - t
    if "actions" not in r:
        return {"error": str(r.get("error") or r)[:200], "seconds": round(sec, 2)}
    return {"rec": r["recommended"], "seconds": round(sec, 2), "pot": r.get("pot_bb"),
            "evs": {a["action"]: a.get("ev_bb") for a in r["actions"]}, "mix": {a["action"]: a["freq"] for a in r["actions"]},
            "off": r.get("off_range"), "hero_combos": r.get("hero_range_combos"), "multiway": bool(r.get("multiway")),
            "solved": bool(r.get("solver") or r.get("precomputed")), "street": r.get("street")}


def run(a):
    from gto.native import advisors
    from gto import core
    core.threads(a.threads)
    adv = advisors(2, None, None)["fast"]
    done = []
    if os.path.exists(a.out):
        done = [json.loads(x) for x in open(a.out)]
    seen = {d["key"] for d in done}
    rng = random.Random(a.seed)
    out = open(a.out, "a")
    kept, tried = len(done), 0
    first = True
    while kept < a.spots:
        st = dict(THE_HAND) if first else deal(rng, tried)
        first = False
        tried += 1
        key = json.dumps({k: st[k] for k in ("hero", "hole", "board", "preflop", "flop", "turn", "river")}, sort_keys=True)
        if key in seen:
            continue
        name = st.pop("name")
        main = outcome(adv, {**st, "opponents": "pool"})
        if "rec" not in main or bool(main.get("off")) == a.in_range:
            continue
        row = {"key": key, "name": name, "main": main, "chart": outcome(adv, st)}
        if not a.in_range:
            row["wider"] = outcome(adv, {**st, "wider": True})
            row["wider_pool"] = outcome(adv, {**st, "wider": True, "opponents": "pool"})
        out.write(json.dumps(row) + "\n"); out.flush()
        kept += 1
        print(kept, tried, name, st["hole"], st["board"], *[row[k].get("rec") for k in row if k not in ("key", "name")],
              [row[k]["seconds"] for k in row if k not in ("key", "name")], flush=True)


def family(a):
    return "aggressive" if a and a.split()[0] in ("bet", "raise", "all-in") else a


def regret(main, other):
    e = {k: v for k, v in other.get("evs", {}).items() if v is not None}
    if not e or main.get("rec") not in e:
        return None
    return max(e.values()) - e[main["rec"]]


def control(rows):
    """--in-range's: the answer against the pool and at the charts' ranges, for a hand in its range."""
    ok = [r for r in rows if "rec" in r["chart"]]
    for title, sel in (("all", ok), ("heads-up", [r for r in ok if not r["main"]["multiway"]]),
                       ("multiway", [r for r in ok if r["main"]["multiway"]])):
        n = len(sel)
        if n:
            lost = [(regret(r["main"], r["chart"]) or 0) / r["main"]["pot"] * 100 for r in sel]
            print(f"in range, {title}: n={n}  differ {sum(r['main']['rec'] != r['chart']['rec'] for r in sel)}, "
                  f"costs >=1% pot {sum(x >= 1 for x in lost)}, >=3% {sum(x >= 3 for x in lost)}")


def report(path):
    rows = [json.loads(x) for x in open(path)]
    if rows and "wider" not in rows[0]:
        return control(rows)
    ok = [r for r in rows if all("rec" in r[k] for k in ("main", "chart", "wider"))]
    print(len(rows), "spots,", len(ok), "answered under all three")
    for title, sel in (("all", ok), ("heads-up", [r for r in ok if not r["main"]["multiway"]]),
                       ("multiway", [r for r in ok if r["main"]["multiway"]]),
                       ("hand left its range this street", [r for r in ok if (r["main"]["off"].get("at_street_start") or 0) >= 0.005]),
                       ("off at the street's start", [r for r in ok if (r["main"]["off"].get("at_street_start") or 0) < 0.005])):
        n = len(sel)
        if not n:
            continue
        exact = sum(len({r[k]["rec"] for k in ("main", "chart", "wider")}) > 1 for r in sel)
        fam = sum(len({family(r[k]["rec"]) for k in ("main", "chart", "wider")}) > 1 for r in sel)
        widerfam = sum(family(r["wider"]["rec"]) != family(r["chart"]["rec"]) for r in sel)
        poolfam = sum(family(r["main"]["rec"]) != family(r["chart"]["rec"]) for r in sel)
        rc = [(regret(r["main"], r["chart"]) or 0, r["main"]["pot"]) for r in sel]
        rw = [(regret(r["main"], r["wider"]) or 0, r["main"]["pot"]) for r in sel]
        rworst = [max(a[0] / a[1], b[0] / b[1]) * 100 for a, b in zip(rc, rw)]
        big = sum(x >= 1.0 for x in rworst)
        huge = sum(x >= 3.0 for x in rworst)
        print(f"{title}: n={n}  differ exactly {exact} ({exact/n:.0%}), differ bet v passive {fam} ({fam/n:.0%}); "
              f"wider v chart {widerfam}, pool v chart {poolfam}; main costs >=1% pot under chart or wider {big} ({big/n:.0%}), "
              f">=3% {huge} ({huge/n:.0%}); median worst {sorted(rworst)[n // 2]:.2f}% pot")
    for k in ("main", "chart", "wider", "wider_pool"):
        s = sorted(r[k]["seconds"] for r in rows if "seconds" in r[k])
        print(f"{k}: median {s[len(s) // 2]:.1f} s, p90 {s[int(len(s) * .9)]:.1f}, max {s[-1]:.1f}")
    tot = sorted(r["chart"]["seconds"] + r["wider"]["seconds"] for r in rows)
    print(f"the two follow-ups together: median {tot[len(tot) // 2]:.1f} s, p90 {tot[int(len(tot) * .9)]:.1f}, max {tot[-1]:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--spots", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--in-range", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a.out) if a.report else run(a)
