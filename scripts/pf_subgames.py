"""Subgames of the preflop game entered with the charts' ranges: each reference spot of tests/preflop_refs.py
(and the spots without one in pf_compare.SPOTS) solved from hero's decision on, every player arriving with
the range the charts (gto/preflop.py) give it there and everything after (hero's choice, the others'
responses, later raises) played to equilibrium with the measured realization model (gto/pf_eqr.py).

This tests the realization model where the published solves can check it, without the rest of the full
solve's ranges in the way: the full solve's opens and 3bets differ from the published ones, which moves
every later spot; here the entry ranges are the charts' (matched to the published solves).

    python scripts/pf_subgames.py [--round N --cache DIR]   # model from a round of pf_equilibrium (default gto/pf_eqr.py)
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


def chart_line(line_text, hero, T):
    """The chart action list up to hero's turn ("CO raise 2.5, BTN raise 7.5"), everyone skipped folding."""
    from tests.preflop_refs import _line
    from gto.advisor import replay_hand
    pre = _line(line_text, T)
    while (st := replay_hand(T, 100, {"preflop": pre}, 1)[0]).to_act != hero:
        pre.append({"pos": st.to_act, "action": "fold"})
    return pre


def subgame(model, pre, hero, rake):
    """The pf_game subgame at hero's decision after the chart line `pre`, entered with the charts' ranges."""
    from gto import pf_game as G
    from gto import preflop as pf
    from gto.pf_classes import to_classes
    root = G.build()
    node = root
    for a in pre:
        while node.kind == "dec" and G.SIX[node.player] != a["pos"]:
            node = node.children[node.names.index("fold")]
        act = {"raise": "raise", "all-in": "all-in", "call": "call", "fold": "fold"}[a["action"]]
        if act == "raise" and "raise" not in node.names:
            act = "all-in"
        node = node.children[node.names.index(act)]
    while node.kind == "dec" and G.SIX[node.player] != hero:
        node = node.children[node.names.index("fold")]
    T = pf.Table(pf.SEATS[6])
    R = pf.replay_ranges(pre, T)
    reach = np.tile(G.PRIOR, (6, 1))
    for i, p in enumerate(G.SIX):
        if p in R:
            reach[i] = to_classes(R[p]) / 1326.0
    return G.Game(node, G.SIX, model, rake=rake, root_reach=reach), node


def run_refs(model, iters=600, verbose=True):
    from tests.preflop_refs import REFS, HANDS
    from gto import preflop as pf
    from gto.pf_game import PRIOR
    from gto.pf_classes import CLASS_OF
    from gto.ranges import parse_range
    errs, n_miss, n_refs, cache = [], 0, 0, {}
    for r in REFS:
        if not r["line"]:
            continue                                   # opens: the full game's
        T = pf.Table(pf.SEATS[6], rake=r["rake"])
        pre = chart_line(r["line"], r["hero"], T)
        g, node = subgame(model, pre, r["hero"], r["rake"])
        g.solve(iters)
        s = g.strategy_at(node)
        w = PRIOR if r["of"] == "hands" else g.root_reach[node.player]
        w = w / w.sum()
        got = {("raise" if k == "all-in" else k): float((w * v).sum()) for k, v in s.items()}
        bad = {k: (got.get(k, 0), v) for k, v in r["want"].items() if abs(got.get(k, 0) - v) > r["tol"] + 1e-9}
        cache[(r["line"], r["hero"])] = s
        if not r["uncertain"]:
            errs += [abs(got.get(k, 0) - v) for k, v in r["want"].items()]
            n_miss += bool(bad)
            n_refs += 1
        if verbose:
            cell = lambda k: f"{got.get(k, 0) * 100:5.1f}/{r['want'][k] * 100:5.1f}" if k in r["want"] else " " * 11
            print(f"{r['name']:34s} {r['of']:5s} fold {cell('fold')}  call {cell('call')}  raise {cell('raise')}  "
                  f"{'MISS' if bad else 'ok'}{'?' if r['uncertain'] else ''}", flush=True)
    mae = float(np.mean(errs))
    print(f"subgames: certain references outside tolerance {n_miss} of {n_refs} (the opens are the full game's), "
          f"mean abs error {mae * 100:.2f} points")
    nb = 0
    for h in HANDS:
        name, hero, line, hand, action, lo, hi = h[:7]
        if not line:
            continue
        T = pf.Table(pf.SEATS[6], rake=(0.05, 4.0))
        key = next((k for k in cache if k[0] == line and k[1] == hero), None)
        if key is None:
            pre = chart_line(line, hero, T)
            g, node = subgame(model, pre, hero, (0.05, 4.0))
            g.solve(iters)
            cache[(line, hero)] = g.strategy_at(node)
        s = cache[(line, hero)]
        k = action if action in s else "all-in"
        v = float(np.mean(s.get(k, np.zeros(169))[CLASS_OF][parse_range(hand) > 0]))
        ok = lo - 1e-9 <= v <= hi + 1e-9
        nb += not ok
        print(f"  {name:36s} {action} {v * 100:5.1f}% (want {lo * 100:.0f}-{hi * 100:.0f}) {'ok' if ok else 'MISS'}")
    print(f"hand facts (not opens) outside range {nb}")
    return mae, n_miss, nb


def main():
    import pf_equilibrium as P
    from gto import pf_game as G
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int)
    ap.add_argument("--cache", default=P.DEFAULT_CACHE)
    ap.add_argument("--iters", type=int, default=600)
    args = ap.parse_args()
    model = P.load_model(args.cache, args.round) if args.round is not None else G.measured_model()
    run_refs(model, args.iters)


if __name__ == "__main__":
    main()
