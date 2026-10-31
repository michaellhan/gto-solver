"""3-way flops the 3-way library doesn't store, read off its stored flops most like them (MWLibrary.near), measured
against stored solves: each stored flop left out of its spot in turn and read off the others, as Study's answer at
once reads a flop the library doesn't hold. Each combo's mix at the STORED decisions is priced by the left-out
flop's own EVs per action: EV given up = the stored mix's EV - the read mix's, weighted by the acting player's range
at the node (each player acts there for the first time: its range entering the flop), % of the pot. Per node, by how
many checks came before it, and per spot.

    python scripts/bench_mw_approx.py [--step 1] [--spots mw_co_btn_bb,...] [--json OUT.json]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gto import mw_library as ML  # noqa: E402
from gto.cards import cards as parse_cards  # noqa: E402
from gto.solver import MEMBER_FULL  # noqa: E402


def leave_one_out(lib, spot, name, ranges):
    """[(path, EV given up % pot, range frequencies' total variation)] for one stored flop read off the others, or
    None where nothing near enough is stored."""
    canon = parse_cards(name)
    e = lib.entry(spot, name)
    h = spot["flops"].pop(name)
    try:
        sol = lib.near(ranges, canon, spot["pot"], spot["stack"])
    finally:
        spot["flops"][name] = h
    if sol is None or e is None:
        return None
    bm = np.prod([1.0 - MEMBER_FULL[c] for c in canon], axis=0)
    act = sol.hands.active
    out = []
    for nd in e["nodes"]:
        path, p = tuple(nd["path"]), nd["player"]
        d = sol.nodes.get(path)
        if d is None or list(d.names) != list(nd["names"]):
            out.append((path, None, None))
            continue
        cols = np.flatnonzero(e["masks"][p])
        pos = np.full(len(MEMBER_FULL[0]), -1)
        pos[act] = np.arange(len(act))
        P = d.avg_strategy()[0][:, pos[cols]].astype(np.float64)
        S, E = nd["strat"], nd["evs"]
        w = (ranges[p] * bm)[cols]
        loss = float((w * ((S - P) * E).sum(axis=0)).sum() / w.sum() / nd["pot_now"] * 100)
        tv = float(np.abs((S * w).sum(axis=1) - (P * w).sum(axis=1)).sum() / w.sum() / 2)
        out.append((path, loss, tv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1, help="every step-th stored flop of each spot")
    ap.add_argument("--spots", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    lib = ML.MWLibrary.from_dir()
    rows, t0 = [], time.time()
    for spot in lib.index()["spots"]:
        if a.spots and spot["id"] not in a.spots.split(","):
            continue
        ranges = [np.asarray(r, dtype=np.float64) for r in ML.spot_setup(spot["line"])["ranges"]]
        for name in sorted(spot["flops"])[::a.step]:
            got = leave_one_out(lib, spot, name, ranges)
            if got is None:
                rows.append({"spot": spot["id"], "flop": name, "path": None, "loss": None})
                continue
            rows += [{"spot": spot["id"], "flop": name, "path": list(path), "loss": loss, "tv": tv}
                     for path, loss, tv in got]
    ok = [r for r in rows if r["loss"] is not None]
    L = np.array([r["loss"] for r in ok])
    T = np.array([r["tv"] for r in ok])
    print(f"{len({(r['spot'], r['flop']) for r in rows})} flops, {len(ok)} decisions read "
          f"({sum(r['path'] is None for r in rows)} flops with nothing near), {time.time() - t0:.0f} s")
    print(f"EV given up, % of the pot: mean {L.mean():.3f}, median {np.median(L):.3f}, 90th percentile "
          f"{np.percentile(L, 90):.3f}, max {L.max():.2f}; range frequencies off by {T.mean():.3f} (total variation)")
    for k in range(3):
        m = np.array([len(r["path"]) == k for r in ok])
        if m.any():
            print(f"  after {k} checks: mean {L[m].mean():.3f}, 90th percentile {np.percentile(L[m], 90):.3f}, "
                  f"frequencies {T[m].mean():.3f}")
    for sid in sorted({r["spot"] for r in ok}):
        m = np.array([r["spot"] == sid for r in ok])
        print(f"  {sid}: mean {L[m].mean():.3f}, 90th percentile {np.percentile(L[m], 90):.3f}")
    worst = sorted(ok, key=lambda r: -r["loss"])[:5]
    print("worst:", ", ".join(f"{r['spot']} {r['flop']} after {len(r['path'])} checks {r['loss']:.2f}" for r in worst))
    if a.json:
        json.dump(rows, open(a.json, "w"))


if __name__ == "__main__":
    main()
