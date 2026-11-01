"""3-way pots no library spot holds read off one (gto/mw_library.py: its 6-max 100bb spots): at a bigger table (8- and
9-max) as a spot's own where the ranges entering the flop are a few combos apart (MWLibrary.borrow), elsewhere (6-max's
other opens and flats, antes: another pot) at once only, off the nearest spot (MWLibrary.stand_in). Measured against
each pot's own solve: on stored flops of the spot, the pot's three ranges solved from the flop on the library's game
until settled (as scripts/build_mw_library.py), and each combo's mix at the STORED decisions priced by that solve's EVs
per action: EV given up = the solve's mix's EV - the read mix's, weighted by the acting player's range entering the
flop (each acts there for the first time), % of the pot. Read three ways: the spot's stored entry of the flop (the
borrowed precomputed answer, or the stand-in's mixes), the spot's stored flops most like it with this one left out
(MWLibrary.near: the answer at once on a flop the library doesn't store), and with --live-iters the live Fast solve of
the pot's own ranges (what replaces a read at once); the floor, each spot's own pot (its ranges solved the same way
against its entry). Per case and node; one solve per (case, flop), single thread.

    python scripts/bench_mw_formats.py [--flops 8] [--cases btn_sb_bb8,...] [--live-iters 200] [--json OUT.jsonl]
    python scripts/bench_mw_formats.py --report OUT.jsonl
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
import gto.native  # noqa: F401,E402  (holds BLAS to one thread before numpy loads)
from gto import flop_approx as FA  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import mw_library as ML  # noqa: E402
from gto.cards import cards as parse_cards  # noqa: E402
from gto.solver import MEMBER_FULL  # noqa: E402

# case: (players, preflop line, the library spot read)
CASES = {
    "utg_utg1_bb8": (8, "UTG:r2.5,UTG1:c,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:c", "mw_utg_btn_bb"),
    "hj_btn_bb8": (8, "UTG:f,UTG1:f,LJ:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:c", "mw_hj_ip_bb"),
    "btn_sb_bb8": (8, "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:c", "mw_btn_sb_bb"),
    "co_btn_bb8": (8, "UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:c", "mw_co_btn_bb"),
    "lj_btn_bb8": (8, "UTG:f,UTG1:f,LJ:r2.5,HJ:f,CO:f,BTN:c,SB:f,BB:c", "mw_utg_btn_bb"),
    "utg_btn_bb8": (8, "UTG:r2.5,UTG1:f,LJ:f,HJ:f,CO:f,BTN:c,SB:f,BB:c", "mw_utg_btn_bb"),
    "hj_co_bb8": (8, "UTG:f,UTG1:f,LJ:f,HJ:r2.5,CO:c,BTN:f,SB:f,BB:c", "mw_hj_ip_bb"),
    "utg_hj_bb8": (8, "UTG:r2.5,UTG1:f,LJ:f,HJ:c,CO:f,BTN:f,SB:f,BB:c", "mw_utg_btn_bb"),
    "co_sb_bb8": (8, "UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:c", "mw_btn_sb_bb"),
    # 6-max pots the library doesn't hold, read off the nearest spot of the same pot (MWLibrary.stand_in's read; the
    # bench reaches it through borrow, its gates lifted: see main)
    "utg_hj_bb6": (6, "UTG:r2.5,HJ:c,CO:f,BTN:f,SB:f,BB:c", "mw_utg_btn_bb"),
    "co_sb_bb6": (6, "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:c", "mw_btn_sb_bb"),
    "hj_sb_bb6": (6, "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:c,BB:c", "mw_btn_sb_bb"),
    # antes (0.125bb each): another pot, read off the nearest spot (MWLibrary.stand_in's read)
    "utg_utg1_bb8a": (8, "UTG:r2.5,UTG1:c,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:c", "mw_utg_btn_bb", 0.125),
    "btn_sb_bb8a": (8, "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:c", "mw_btn_sb_bb", 0.125),
    "btn_sb_bb6a": (6, "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:c", "mw_btn_sb_bb", 0.125),
    "hj_btn_bb6a": (6, "UTG:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:c", "mw_hj_ip_bb", 0.125),
    "co_sb_bb6a": (6, "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:c", "mw_btn_sb_bb", 0.125),
    "hj_sb_bb6a": (6, "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:c,BB:c", "mw_btn_sb_bb", 0.125),
    # other depths (each player's stack, the depth's charts): another stack-to-pot ratio (MWLibrary.stand_in's read)
    "btn_sb_bb6d120": (6, "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:c", "mw_btn_sb_bb", 0.0, 120.0),
    "co_btn_bb6d85": (6, "UTG:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:c", "mw_co_btn_bb", 0.0, 85.0),
    # the floor: each spot's own pot
    "own_utg_btn_bb": (6, "UTG:r2.5,HJ:f,CO:f,BTN:c,SB:f,BB:c", "mw_utg_btn_bb"),
    "own_hj_ip_bb": (6, "UTG:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:c", "mw_hj_ip_bb"),
    "own_btn_sb_bb": (6, "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:c", "mw_btn_sb_bb"),
    "own_co_btn_bb": (6, "UTG:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:c", "mw_co_btn_bb"),
}


def truth(canon, s, a):
    """The pot's own solve of the canonical flop, as an entry (decoded): run until settled."""
    sol = ML.solve_all(canon, s["ranges"], s["pot"], s["stack"])
    sol.run(iters=a.min_iters)
    expl = sol.br_gains().mean() / sol.root_pot() * 100
    while expl > a.target and sol.iters < a.max_iters:
        sol.run(iters=min(100, a.max_iters - sol.iters))
        expl = sol.br_gains().mean() / sol.root_pot() * 100
    meta = {"spot": "bench", "fp": [fl.range_fp(r) for r in s["ranges"]], "pot": s["pot"], "stack": s["stack"],
            "tree": ML.tree_key(ML.library_config()), "seats": s["seats"], "board": fl.flop_name(canon),
            "all_hands": True}
    return ML.decode(ML.encode(sol, meta)), int(sol.iters), float(expl)


def price(e, read, ranges, canon):
    """{path: EV given up % pot} of the read's mix (read: {path: (names, strat (A, 1326 canonical combos))}) by the
    entry e's EVs, weighted by the acting player's range entering the flop."""
    bm = np.prod([1.0 - MEMBER_FULL[c] for c in canon], axis=0)
    out = {}
    for nd in e["nodes"]:
        path, p = tuple(nd["path"]), nd["player"]
        got = read.get(path)
        if got is None or list(got[0]) != list(nd["names"]):
            continue
        cols = np.flatnonzero(e["masks"][p])
        P = got[1][:, cols].astype(np.float64)
        ok = ~np.isnan(P[0])
        S, E = nd["strat"][:, ok], nd["evs"][:, ok]
        w = (np.asarray(ranges[p]) * bm)[cols][ok]
        if w.sum() <= 0:
            continue
        out[path] = float((w * ((S - P[:, ok]) * E).sum(axis=0)).sum() / w.sum() / nd["pot_now"] * 100)
    return out


def live_read(canon, s, iters):
    """The live Fast 3-way flop solve (advisor.MW_FAST_STREETS' flop game: 12 hand buckets, the turn's tables over every
    third river) of the pot's ranges at `iters` iterations (the browser's Fast reaches ~200 in its 8 s on a first
    decision, natively ~110-125 in 5 s), read at the stored decisions as entry_read; (read, its exploitability % pot)."""
    from gto.advisor import MW_FAST_STREETS
    from gto.multiway_solver import Decision, MultiwaySolver, MWConfig
    sol = MultiwaySolver(canon, s["ranges"], s["pot"], s["stack"], cfg=MWConfig(**dict(MW_FAST_STREETS["flop"])))
    sol.run(iters=iters)
    out = {}
    for path in ML.STORED:
        node = sol.root
        try:
            for nm in path:
                node = node.children[node.names.index(nm)]
        except (ValueError, AttributeError):
            continue
        if not isinstance(node, Decision):
            continue
        st = np.full((len(node.names), len(MEMBER_FULL[0])), np.nan)
        st[:, sol.hands.active] = node.avg_strategy()[0]
        out[path] = (node.names, st)
    return out, float(sol.br_gains().mean() / sol.root_pot() * 100)


def entry_read(e):
    """{path: (names, strat over the 1326 canonical combos)} of a stored entry (nan where it holds no column)."""
    out = {}
    for nd in e["nodes"]:
        p = nd["player"]
        st = np.full((len(nd["names"]), len(MEMBER_FULL[0])), np.nan)
        st[:, np.flatnonzero(e["masks"][p])] = nd["strat"]
        out[tuple(nd["path"])] = (nd["names"], st)
    return out


def near_read(lib, spot, name, ranges, players):
    """The spot's stored flops most like this one (this one left out), read for the canonical flop: as entry_read,
    nan where no source holds an image (MWLibrary.near's 'known'); another table size's pot read off the spot
    (MWLibrary.borrow, its thresholds lifted)."""
    canon = parse_cards(name)
    h = spot["flops"].pop(name)
    pick = lib._pick
    if isinstance(players, tuple):       # (another pot's: the spot named, read as for its own pot and stack)
        lib._pick = lambda *_: (spot, None)
        players = None
    try:
        sol = lib.near(ranges, canon, spot["pot"], spot["stack"], players=players)
    finally:
        spot["flops"][name] = h
        lib._pick = pick
    if sol is None:
        return None
    out = {}
    act = sol.hands.active
    for path, d in sol.nodes.items():
        st = np.full((len(d.names), len(MEMBER_FULL[0])), np.nan)
        k = sol.approx["known"].get(path, np.ones(len(act), bool))
        st[:, act[k]] = d.avg_strategy()[0][:, k]
        out[path] = (d.names, st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flops", type=int, default=8)
    ap.add_argument("--cases", default="")
    ap.add_argument("--min-iters", type=int, default=200)
    ap.add_argument("--max-iters", type=int, default=800)
    ap.add_argument("--target", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--live-iters", type=int, default=0, help="also price the live Fast solve at this many iterations")
    ap.add_argument("--json", default="")
    ap.add_argument("--report", default="", help="only summarize this jsonl")
    a = ap.parse_args()
    if a.report:
        return report([json.loads(x) for x in open(a.report)])
    lib = ML.MWLibrary.from_dir()
    ML.MW_BORROW_DIST, ML.MW_BORROW_COVER = 99.0, 0.0     # (each case's nearest spot, whatever its distance)
    spots = {s["id"]: s for s in lib.index()["spots"]}
    rows = []
    out = open(a.json, "a") if a.json else None
    cases = []
    for case, (players, line, sid, *extra) in CASES.items():
        if a.cases and case not in a.cases.split(","):
            continue
        ante, depth = (list(extra) + [0.0, 100.0][len(extra):])[:2]
        s = ML.spot_setup(line, players=players, ante=ante, stack=depth)
        spot = spots[sid]
        own = ML.spot_setup(spot["line"])
        other = bool(ante) or depth != 100.0      # (another pot or stack: no spot has it)
        assert other or abs(s["pot"] - spot["pot"]) < 0.01 and abs(s["stack"] - spot["stack"]) < 0.01, case
        table = players if players != 6 or case.startswith("own_") else 7   # (borrow reads only another table's)
        if other:
            table = ("pot", sid)     # (near_read reads the spot for the other pot: see there)
        else:
            b = lib.borrow(s["ranges"], s["pot"], s["stack"], table) if table != 6 else None
            assert table == 6 or (b and b["spot"]["id"] == sid), (case, b and b["spot"]["id"])
        dist = sum(FA.range_dist(x, y) for x, y in zip(s["ranges"], own["ranges"]))
        cover = [FA.range_cover(x, y) for x, y in zip(s["ranges"], own["ranges"])]
        names = sorted(spot["flops"])
        random.Random(a.seed).shuffle(names)
        cases.append((case, table, sid, s, spot, dist, cover, names[:a.flops]))
    done = set()
    if a.json and os.path.exists(a.json):
        done = {(r["case"], r["flop"]) for r in map(json.loads, open(a.json))}
    for i in range(a.flops):          # (round robin: every case a flop at a time)
        for case, players, sid, s, spot, dist, cover, names in cases:
            if i >= len(names) or (case, names[i]) in done:
                continue
            name = names[i]
            t0 = time.time()
            canon = parse_cards(name)
            e, iters, expl = truth(canon, s, a)
            stored = lib.entry(spot, name)
            near = near_read(lib, spot, name, s["ranges"], players)
            row = {"case": case, "spot": sid, "players": players, "flop": name, "dist": round(dist, 3),
                   "cover": [round(c, 3) for c in cover], "iters": iters, "expl": round(expl, 4),
                   "stored": {"|".join(k): v for k, v in price(e, entry_read(stored), s["ranges"], canon).items()},
                   "near": None if near is None else
                   {"|".join(k): v for k, v in price(e, near, s["ranges"], canon).items()},
                   "secs": round(time.time() - t0, 1)}
            if a.live_iters:
                lr, lx = live_read(canon, s, a.live_iters)
                row["live"] = {"|".join(k): v for k, v in price(e, lr, s["ranges"], canon).items()}
                row["live_expl"] = round(lx, 4)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if out:
                out.write(json.dumps(row) + "\n")
                out.flush()
    report(rows)


def report(rows):
    print("| case | distance | cover | flops | stored entry: first / checked once / twice, mean | near read: mean | "
          "live Fast: mean |")
    print("|---|---|---|---|---|---|---|")
    for case in dict.fromkeys(r["case"] for r in rows):
        rs = [r for r in rows if r["case"] == case]
        by = {k: [r["stored"].get(k) for r in rs if r["stored"].get(k) is not None] for k in ("", "check", "check|check")}
        allst = [v for r in rs for v in r["stored"].values()]
        alln = [v for r in rs if r["near"] for v in r["near"].values()]
        alll = [v for r in rs if r.get("live") for v in r["live"].values()]
        print(f"| {case} | {rs[0]['dist']:.2f} | {', '.join(f'{c:.2f}' for c in rs[0]['cover'])} | {len(rs)} | "
              + " / ".join(f"{np.mean(v):.3f}" if v else "-" for v in by.values())
              + f", {np.mean(allst):.3f} (max {np.max(allst):.2f}) | "
              + (f"{np.mean(alln):.3f} (max {np.max(alln):.2f})" if alln else "-") + " | "
              + (f"{np.mean(alll):.3f} (max {np.max(alll):.2f}; {len(alll) // 3} flops)" if alll else "-") + " |")


if __name__ == "__main__":
    main()
