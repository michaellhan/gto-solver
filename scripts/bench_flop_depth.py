"""A library spot's flop at another stack depth, answered by its 100bb solve: what that gives up.

The flop library holds 100bb spots only; the same preflop line at 60 or 150bb has nearly the same ranges and pot and
a different stack behind. Here a spot's flop is solved at each depth (the browser's Fast or Accurate tree, natively,
one thread) and at 100bb, each solve made an entry (flop_library.encode: the decisions an entry stores, with each
action's EV). The 100bb solve's mixes ("solve100": the depth's own cost), the 100bb library entry's ("lib100": what
an answer read off the library gives, its tree's distance from this one included) and, with --knn, the spot's three
stored flops most like it other than itself ("knn100": a flop the spot doesn't store, read as flop_approx.predict
reads it) are each priced by the depth's entry (bench_flop_approx.given_up: each combo's mix priced by the EVs per
action, the acting player's reach, % of the pot; per flop over the stored decisions by how often each comes). At
100bb itself lib100 is the measure's floor.
One JSON line a (spot, flop, depth) to OUT; resumable.

    python scripts/bench_flop_depth.py OUT.jsonl --spots srp_btn_bb,3bp_bb_btn --depths 60,80,150 --n 6
    python scripts/bench_flop_depth.py OUT.jsonl --report
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import random
import sys
import time

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("GTO_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gto.native  # noqa: E402,F401
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import flop_approx as fa  # noqa: E402
from gto import solver as S  # noqa: E402
from gto.cards import cards as parse_cards  # noqa: E402
import bench_flop_approx as BA  # noqa: E402
import bench_flop_seed as SEED  # noqa: E402


def setup(sid, depth):
    line = dict((s[0], s[2]) for s in fl.SPOTS)[sid]
    return fl.spot_setup(fl.format_line(line, fl.Format(stack=float(depth))), 6, float(depth))


def entry(board, s, cfg, iters, sid):
    """The flop solved at s's ranges, pot and stack, as a decoded library entry; (entry, exploitability, seconds)."""
    t0 = time.time()
    sol = S.solve(board, s["full0"], s["full1"], s["pot"], s["stack"], cfg=cfg, time_budget=1e9, max_iters=iters)
    meta = {"spot": sid, "fp": [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])], "pot": s["pot"],
            "stack": s["stack"], "tree": fl.tree_key(cfg), "board": fl.cards_str(board)}
    e = fl.decode(fl.encode(sol, meta))
    return e, float(sol.exploitability()), time.time() - t0


def as_pred(e):
    """An entry's own mixes as predict()'s nodes (identity card map)."""
    return fa.predict([(e, np.arange(52), 1.0)])


def price(truth, s, pred, pot):
    """(per-flop given up, {path: (share, loss, unseen)}) of pred's mixes at the truth entry's stored nodes."""
    reaches = BA.node_reaches(truth, s["full0"], s["full1"])
    num = den = 0.0
    per = {}
    for nd in truth["nodes"]:
        path = tuple(nd["path"])
        pr = pred.get(path)
        r, share = reaches[path]
        if pr is None or pr["names"] != list(nd["names"]):
            per["|".join(path)] = {"share": share, "missing": True}
            continue
        loss, _, unseen = BA.given_up(nd, truth["masks"][nd["player"]], r, pr["strat"], pot)
        per["|".join(path)] = {"share": share, "loss": round(loss, 4), "unseen": round(unseen, 4)}
        if share:
            num += share * loss
            den += share
    return (num / den if den else None), per


def run(a):
    lib = fl.FlopLibrary.from_dir()
    spots = {x["id"]: x for x in lib.index()["spots"]}
    cfg = SEED.config(a.config)
    depths = [float(x) for x in a.depths.split(",")]
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            r = json.loads(x)
            done.add((r["spot"], r["flop"], r["depth"]))
    for sid in a.spots.split(","):
        spot = spots[sid]
        names = sorted(spot["flops"])
        random.Random(a.seed).shuffle(names)
        s100 = setup(sid, 100)
        for name in names[:a.n]:
            todo = [d for d in depths + [100.0] if (sid, name, d) not in done]
            if not todo:
                continue
            board = parse_cards(name)
            e100, x100, t100 = entry(board, s100, cfg, a.iters, sid)
            lib100 = lib.entry(spot, name)
            for d in todo:
                s = s100 if d == 100.0 else setup(sid, d)
                ed, xd, td = (e100, x100, t100) if d == 100.0 else entry(board, s, cfg, a.iters, sid)
                rec = {"spot": sid, "flop": name, "depth": d, "pot": s["pot"], "stack": s["stack"],
                       "spr": round(s["stack"] / s["pot"], 2), "expl": round(xd, 4), "secs": round(td, 1),
                       "cover": [round(fa.range_cover(s["full0"], s100["full0"]), 3),
                                 round(fa.range_cover(s["full1"], s100["full1"]), 3)]}
                preds = [("solve100", as_pred(e100)), ("lib100", as_pred(lib100))]
                if a.knn:
                    src = [(lib.entry(spot, nm), m, 1.0 / (1.0 + dd))
                           for nm, m, dd in fa.neighbours([n for n in names if n != name], board, fa.K)
                           if dd <= fa.MAX_DIST]
                    preds.append(("knn100", fa.predict(src, board=list(board)) if src else {}))
                for key, pred in preds:
                    g, per = price(ed, s, pred, s["pot"])
                    rec[key] = None if g is None else round(g, 4)
                    rec[key + "_nodes"] = per
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(sid, name, d, rec["spr"], rec["solve100"], rec["lib100"], rec["secs"], flush=True)


def report(path):
    rows = [json.loads(x) for x in open(path)]
    g = collections.defaultdict(list)
    first = collections.defaultdict(list)
    for r in rows:
        k = (r["spot"][:3], r["depth"])
        g[k].append((r["solve100"], r["lib100"], r["spr"]))
        for key in ("solve100", "lib100"):
            for p, v in r[key + "_nodes"].items():
                if p in ("", "check") and not v.get("missing"):
                    first[k + (key,)].append(v["loss"])
    print("kind depth  n  spr    solve100 per flop mean / median / 90th (first)     "
          "lib100 mean / median / 90th (first)")
    for k in sorted(g):
        a = np.array([[x if x is not None else np.nan for x in t] for t in g[k]], dtype=float)
        cols = []
        for j, key in enumerate(("solve100", "lib100")):
            v = a[:, j][~np.isnan(a[:, j])]
            f = first[k + (key,)]
            cols.append(f"{v.mean():.3f} / {np.median(v):.3f} / {np.percentile(v, 90):.3f} ({np.mean(f):.3f})"
                        if len(v) else "-")
        print(f"{k[0]}  {k[1]:5g} {len(a):3d} {np.mean(a[:, 2]):5.1f}   {cols[0]:40s}   {cols[1]}")
    # the depths the advisor reads off the 100bb spot (flop_approx.near_spot: NEAR_MIN_SPR, NEAR_SPR, NEAR_COVER)
    spr100 = {r["spot"]: r["spr"] for r in rows if r["depth"] == 100.0}
    print("as the advisor reads them (another depth inside NEAR_MIN_SPR, NEAR_SPR and NEAR_COVER):")
    for kind in sorted({r["spot"][:3] for r in rows}):
        sel = [r for r in rows if r["spot"][:3] == kind and r["depth"] != 100.0 and r["spot"] in spr100
               and r["spr"] >= fa.NEAR_MIN_SPR and fa.NEAR_SPR[0] <= r["spr"] / spr100[r["spot"]] <= fa.NEAR_SPR[1]
               and min(r["cover"]) >= fa.NEAR_COVER and r["lib100"] is not None]
        if not sel:
            continue
        floor = [x["lib100"] for x in rows if x["depth"] == 100.0 and x["spot"][:3] == kind and x["lib100"] is not None]
        cols = []
        for key in ("solve100", "lib100", "knn100"):
            v = np.array([r[key] for r in sel if r.get(key) is not None])
            if not len(v):
                continue
            cols.append(f"{key} {v.mean():.3f} / {np.median(v):.3f} / {np.percentile(v, 90):.3f}")
        print(f"{kind} n={len(sel)} depths {sorted({r['depth'] for r in sel})}: {'; '.join(cols)} "
              f"(lib100 at 100bb {np.mean(floor):.3f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--spots", default="srp_btn_bb,3bp_bb_btn")
    ap.add_argument("--depths", default="60,80,150")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--config", default="accurate")
    ap.add_argument("--knn", action="store_true", help="also knn100 (the depth's own entry is priced again)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.out)
    run(a)
    report(a.out)


if __name__ == "__main__":
    main()
