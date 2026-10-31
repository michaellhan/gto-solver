"""Study's answers at once at later 3-way and 4-way nodes (Advisor._study_mw_prev), measured against the answers that
replace them: Study walked node by node as the page does (each node's own live solve, Fast, natively), at each node
first the answer at once (Advisor.instant) read off the hand's earlier solve, then the node's own. On a flop the
library doesn't store: checked to once, twice (and 4-way three times; the first decision's solve, or the latest
one's), facing the last player's bet (its solve); on a turn after the flop checked through, first to act (the
flop's solve, which plays the turn by hand buckets at one bet size; 3-way only) and checked to (the turn's first
decision's solve); on the turn after the last player's flop bet was called by everyone, first to act (the flop's
solve). Each combo's mix at once is priced by the replacing answer's EVs per action (a size its tree doesn't have by
interpolation between the two nearest, by amount): EV given up = the replacing mix's EV - the mix at once's,
weighted by the actor's range there, % of the pot; the range's frequencies by kind (check, bet, fold, call, raise)
apart in total variation. Lines: btn (BTN opens, SB and BB call), co (CO opens, BTN and BB call), hj4 (HJ opens, CO,
BTN and BB call: 4-way).

    python scripts/bench_mw_prev.py [--flops N] [--lines btn,co,hj4] [--seed 7] [--flop-budget S] [--json OUT.json]
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gto import advisor as A  # noqa: E402
from gto.native import advisor  # noqa: E402

FOLDS = {"UTG": {"pos": "UTG", "action": "fold"}, "HJ": {"pos": "HJ", "action": "fold"},
         "CO": {"pos": "CO", "action": "fold"}, "BTN": {"pos": "BTN", "action": "fold"}}
LINES = {   # preflop: an open called by two
    "btn": [FOLDS["UTG"], FOLDS["HJ"], FOLDS["CO"], {"pos": "BTN", "action": "raise", "to": 2.5},
            {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}],
    "co": [FOLDS["UTG"], FOLDS["HJ"], {"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "call"},
           {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}],
    "hj4": [FOLDS["UTG"], {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "call"},
            {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}],
}
RANKS, SUITS = "23456789TJQKA", "cdhs"
KIND = lambda nm: nm.split()[0]


def price(inst: dict, live: dict):
    """(EV given up % of the pot, frequencies by kind apart) of the answer at once against the live one, or None."""
    names = [a["action"] for a in live["actions"]]
    amt = [a["amount_bb"] for a in live["actions"]]
    cols = []
    for a in inst["actions"]:
        nm, k = a["action"], KIND(a["action"])
        if nm in names:
            cols.append([(names.index(nm), 1.0)])
            continue
        same = [i for i, n in enumerate(names) if KIND(n) == k or {k, KIND(n)} <= {"bet", "raise", "all-in"}]
        if not same or a["amount_bb"] is None:
            return None
        x = float(a["amount_bb"])
        lo = [i for i in same if amt[i] is not None and amt[i] <= x]
        hi = [i for i in same if amt[i] is not None and amt[i] >= x]
        if lo and hi:
            i, j = max(lo, key=lambda i: amt[i]), min(hi, key=lambda i: amt[i])
            f = 0.0 if amt[j] == amt[i] else (x - amt[i]) / (amt[j] - amt[i])
            cols.append([(i, 1.0 - f), (j, f)])
        else:
            cols.append([(min(same, key=lambda i: abs((amt[i] or 0.0) - x)), 1.0)])
    got = {r[0]: r[2] for rows in inst["combos"].values() for r in rows}
    num = den = 0.0
    for rows in live["combos"].values():
        for r in rows:
            c, w, s, ev = r[0], r[1], r[2], r[3]
            if c not in got or w <= 0:
                continue
            e_live = sum(si * ei for si, ei in zip(s, ev))
            e_inst = sum(pi * sum(f * ev[i] for i, f in cols[a]) for a, pi in enumerate(got[c]))
            num += w * (e_live - e_inst)
            den += w
    if den <= 0:
        return None
    fk = lambda r: {k: sum(a["freq"] for a in r["actions"] if KIND(a["action"]) == k)
                    for k in {KIND(a["action"]) for a in inst["actions"] + live["actions"]}}
    fi, fl = fk(inst), fk(live)
    tv = sum(abs(fi.get(k, 0.0) - fl.get(k, 0.0)) for k in set(fi) | set(fl)) / 2
    return num / den / live["pot_bb"] * 100, tv


def measure(adv, ln, fl, tag, s, rows) -> dict:
    """A node's answer at once, then its own (returned), priced; a row in `rows`."""
    t0 = time.time()
    inst = adv.instant(s)
    ti = time.time() - t0
    live = adv.advise(s)
    rec = {"line": ln, "flop": fl, "board": s["board"], "node": tag, "instant_s": round(ti, 3),
           "source": (inst or {}).get("approximate", {}).get("source"),
           "iters": (live.get("solver") or {}).get("iterations")}
    if inst is not None and live.get("study") == "node":
        got = price(inst, live)
        if got is not None:
            rec.update(loss=round(got[0], 4), tv=round(got[1], 4))
    rows.append(rec)
    print(json.dumps(rec), flush=True)
    return live


def boards(seed):
    """Random flops (any suits), each with a turn card."""
    rnd = random.Random(seed)
    deck = [r + x for r in RANKS for x in SUITS]
    while True:
        b = rnd.sample(deck, 4)
        yield "".join(b[:3]), b[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flops", type=int, default=4)
    ap.add_argument("--lines", default="btn,co")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--flop-budget", type=float, default=None, help="Study's 3-way flop budget, s (default "
                    "advisor.MW_STUDY_FLOP_BUDGET)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.flop_budget:
        A.MW_STUDY_FLOP_BUDGET = a.flop_budget
    rows = []
    for ln in a.lines.split(","):
        pre = LINES[ln]
        adv = advisor("fast")
        done, gen = 0, boards(a.seed)
        while done < a.flops:
            fl, turn = next(gen)
            st = {"players": 6, "stack": 100, "preflop": pre, "board": fl, "study": True}
            root = adv.advise(st)
            if root.get("study") != "node" or not root.get("multiway") or root.get("precomputed"):
                continue   # (a flop the 3-way library stores: its checked-to decisions answer at once anyway)
            done += 1
            order = [root["actor"]] + [o["seat"] for o in root["opponents"]]
            X = lambda p: {"pos": p, "action": "check"}
            n = len(order)
            nodes = [("flop " + "x" * i, dict(st, flop=[X(p) for p in order[:i]])) for i in range(1, n)]
            t = dict(st, board=fl + turn, flop=[X(p) for p in order])
            nodes += [("turn", t), ("turn x", dict(t, turn=[X(order[0])]))]
            for tag, s in nodes:
                live = measure(adv, ln, fl, tag, s, rows)
                if tag == "flop " + "x" * (n - 1) and live.get("study") == "node":
                    b = next((x for x in live["actions"] if KIND(x["action"]) == "bet"), None)
                    if b is not None and b["freq"] >= 0.15:   # facing the last player's bet: the first player's call
                        bet = {"pos": order[-1], "action": "bet", "to": b["amount_bb"]}
                        s2 = dict(s, flop=s["flop"] + [bet])
                        live2 = measure(adv, ln, fl, tag + "b", s2, rows)
                        c = next((x for x in live2["actions"] if x["action"] == "call"), None) \
                            if live2.get("study") == "node" else None
                        if c is not None and c["freq"] >= 0.15:   # bet, called by everyone: the turn first to act
                            tb = dict(t, flop=s2["flop"] + [{"pos": p, "action": "call"} for p in order[:-1]])
                            measure(adv, ln, fl, "turn b", tb, rows)
    ok = [r for r in rows if "loss" in r]
    print(f"{len(ok)} of {len(rows)} nodes answered at once")
    for tag in ("flop x", "flop xx", "flop xxb", "flop xxx", "flop xxxb", "turn", "turn x", "turn b"):
        L = [r["loss"] for r in ok if r["node"] == tag]
        T = [r["tv"] for r in ok if r["node"] == tag]
        if L:
            print(f"{tag:9s} n={len(L):2d}  EV given up {np.mean(L):.3f}% of the pot (median {np.median(L):.3f}, max "
                  f"{max(L):.3f})  frequencies {np.mean(T) * 100:.1f} points apart (max {max(T) * 100:.1f})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
