"""The charts (gto/preflop.py) against the derived preflop equilibrium (scripts/pf_equilibrium.py) spot by spot.

For each spot: the solve's and the chart's fold/call/raise shares (of hands, or of the range arriving), and
what the chart's strategy there costs inside the solved game: hero plays the chart at that one node and
the solve everywhere else (every other player, and hero's later decisions, as solved), against the solve's
own strategy there. In the solve's model its strategy is a best response, so the cost is >= 0; a spot where
the chart gives up a lot is one the solve says the chart gets wrong.

    python scripts/pf_compare.py --round 2 [--cache DIR]
"""
from __future__ import annotations
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import gto.native  # noqa: F401
import numpy as np

# (name, line in pf_game actions up to hero's decision, hero, shares of "hands" or "range")
SPOTS = [
    # references exist (tests/preflop_refs.py)
    ("BB vs BTN open", [("BTN", "raise")], "BB", "hands"),
    ("CO open vs BTN 3bet", [("CO", "raise"), ("BTN", "raise")], "CO", "range"),
    # no published reference
    ("HJ vs UTG open", [("UTG", "raise")], "HJ", "hands"),
    ("CO vs UTG open", [("UTG", "raise")], "CO", "hands"),
    ("BTN vs UTG open", [("UTG", "raise")], "BTN", "hands"),
    ("BTN vs HJ open", [("HJ", "raise")], "BTN", "hands"),
    ("SB vs UTG open", [("UTG", "raise")], "SB", "hands"),
    ("BB squeeze, CO open BTN call", [("CO", "raise"), ("BTN", "call")], "BB", "hands"),
    ("BB squeeze, UTG open BTN call", [("UTG", "raise"), ("BTN", "call")], "BB", "hands"),
    ("SB squeeze, CO open BTN call", [("CO", "raise"), ("BTN", "call")], "SB", "hands"),
    ("BB squeeze, BTN open SB call", [("BTN", "raise"), ("SB", "call")], "BB", "hands"),
    ("BTN caller vs BB squeeze", [("CO", "raise"), ("BTN", "call"), ("BB", "raise"), ("CO", "fold")], "BTN", "range"),
    ("CO opener vs BB squeeze", [("CO", "raise"), ("BTN", "call"), ("BB", "raise")], "CO", "range"),
    ("BTN cold vs UTG open + HJ 3bet", [("UTG", "raise"), ("HJ", "raise")], "BTN", "hands"),
    ("BB cold vs CO open + BTN 3bet", [("CO", "raise"), ("BTN", "raise")], "BB", "hands"),
    ("BTN 3bettor vs CO 4bet", [("CO", "raise"), ("BTN", "raise"), ("CO", "raise")], "BTN", "range"),
    ("BB 3bettor vs BTN 4bet", [("BTN", "raise"), ("BB", "raise"), ("BTN", "raise")], "BB", "range"),
    ("CO 3bettor vs UTG 4bet", [("UTG", "raise"), ("CO", "raise"), ("UTG", "raise")], "CO", "range"),
    ("BB 3bettor vs SB 4bet", [("SB", "raise"), ("BB", "raise"), ("SB", "raise")], "BB", "range"),
    ("CO 4bettor vs BTN jam", [("CO", "raise"), ("BTN", "raise"), ("CO", "raise"), ("BTN", "all-in")], "CO", "range"),
    ("BTN 4bettor vs BB jam", [("BTN", "raise"), ("BB", "raise"), ("BTN", "raise"), ("BB", "all-in")], "BTN", "range"),
]


def node_after(g, line, hero):
    node = g.root
    for posn, act in line + [(hero, None)]:
        while node.kind == "dec" and g.positions[node.player] != posn:
            node = node.children[node.names.index("fold")]
        if act is not None:
            node = node.children[node.names.index(act)]
    return node


def chart_strategy(g, node, hero):
    """The chart's strategy (per class, by the node's action names) on the node's line."""
    from gto import preflop as pf
    from gto.pf_classes import CLASS_OF, N_IN
    T = pf.Table(pf.SEATS[6])
    acts = [{"pos": p, "action": "raise" if a == "all-in" else a, **({"to": to} if to else {})}
            for p, a, to in node.line]
    s = pf.strategy_for(hero, acts, T)
    cls = lambda v: np.bincount(CLASS_OF, weights=v, minlength=len(N_IN)) / N_IN
    out = np.zeros((g.A, len(N_IN)))
    for a, nm in enumerate(node.names):
        key = {"fold": "fold", "call": "call" if "call" in s else "check", "raise": "raise", "all-in": "raise"}[nm]
        out[a] = cls(s.get(key, np.zeros(len(CLASS_OF))))
    tot = out.sum(0)
    out[0] += np.maximum(0.0, 1.0 - tot)          # anything the chart leaves out folds
    return out / np.maximum(out.sum(0), 1e-12)


def hero_value(g, strat, node):
    """Hero's expected value from `node` on, bb per hand dealt (its reach times its counterfactual value)."""
    reach = g._down(strat)
    V = g._up(reach, strat)
    p = node.player
    return float((reach[node.index, p] * V[node.index, p]).sum()), float(reach[node.index, p].sum())


def compare(g, verbose=True):
    from gto.pf_game import PRIOR
    avg = g.average()
    reach = g._down(avg)
    rows = []
    for name, line, hero, of in SPOTS:
        node = node_after(g, line, hero)
        d = g.dec_of[node.index]
        w = PRIOR if of == "hands" else reach[node.index, node.player]
        w = w / w.sum()
        ch = chart_strategy(g, node, hero)
        sol = {nm: float((w * avg[d, a]).sum()) for a, nm in enumerate(node.names)}
        cht = {nm: float((w * ch[a, :]).sum()) for a, nm in enumerate(node.names)}
        alt = avg.copy()
        alt[d, :len(node.names)] = ch[:len(node.names)]
        v_sol, m = hero_value(g, avg, node)
        v_ch, _ = hero_value(g, alt, node)
        # per arrival: the reach mass at the node (hero's range there times the others' reaching it) is m
        # times the others' share; report the cost per 100 hands dealt and per time hero is there
        others = float(np.prod([reach[node.index, q].sum() / PRIOR.sum() for q in range(g.n) if q != hero and q != node.player]))
        cost = (v_sol - v_ch) * 100
        arrive = float(np.prod([reach[node.index, q].sum() for q in range(g.n)]))
        per = (v_sol - v_ch) / max(arrive, 1e-12)
        rows.append((name, of, sol, cht, cost, per, arrive))
        if verbose:
            fmt = lambda s: " ".join(f"{k} {v * 100:5.1f}" for k, v in s.items())
            print(f"{name:34s} {of:5s} solve [{fmt(sol)}]  chart [{fmt(cht)}]  chart costs {cost:6.3f} bb/100 "
                  f"({per:5.2f} bb a time; there {arrive * 100:.2f}% of hands)")
    return rows


def main():
    import pf_equilibrium as P
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rake", default="0.05,4")
    ap.add_argument("--cache", default=P.DEFAULT_CACHE)
    args = ap.parse_args()
    rake = tuple(float(x) for x in args.rake.split(",")) if args.rake else None
    g = P.game(rake=rake)
    g.set_model(P.load_model(args.cache, args.round))
    g.load(os.path.join(args.cache, f"solve{args.round}{args.tag}.npz"))
    compare(g)


if __name__ == "__main__":
    main()
