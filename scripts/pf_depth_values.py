"""Per-class flop values at other stack depths, from live solves on a few flops, against the library's at 100bb
on the same flops: how the postflop realization the preflop game uses (pf_game.LibraryModel) moves with depth.

For each spot and depth the spot is set up as the advisor plays it there (flop_library.spot_setup at that stack:
the ranges entering the flop, pot, stack behind), each flop of a systematic subset of the 184 representative flops
is solved on the library's tree for that pot and stack (flop_library.library_config: the river dealt) and read as the library is (pf_library_values.entry_values: each combo's pot share, weighted by
how often it meets the flop), and summed per class. The same flops of the 100bb library give the baseline.

    python scripts/pf_depth_values.py --spots srp_btn_bb,srp_sb_bb,3bp_bb_btn --depths 20,40,200 --flops 23 \\
        --lib gto/static/flop_library --lib ../.work/cov-lib --cache DIR --out depthvals.py

Entries are cached in --cache (DIR/<spot>_<depth>bb/<flop>.bin), so a stopped run resumes. The output module holds
SPOTS = {"<spot>@<depth>": {...as gto/pf_libvals.py...}} for every depth and 100 (the library on the same flops).

    python scripts/pf_depth_values.py --fit depthvals.py      # the depth scales (build_pf_formats.DEPTH_SCALE)

--fit reads such a module: per spot and depth, each class's excess pot share over its all-in equity against the
other player's range (both players), and k, the least-squares factor taking 100bb's excess to that depth's over
the classes both ranges hold (weighted by how often they meet the flop); per pot type the mean over its spots.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import gto.native  # noqa: F401,E402
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto.cards import cards as parse_cards  # noqa: E402
from gto.pf_classes import CLASS_OF, to_classes  # noqa: E402
from gto.solver import solve  # noqa: E402
import pf_library_values as LV  # noqa: E402

K = LV.K


def subset(n: int) -> list:
    """n flops of the 184 representative ones, systematically (every 184/n-th from the middle of the first step):
    each stands for the same share of all flops."""
    rep = [fl.cards_str(f) for f in fl.representative_flops(184)]
    step = len(rep) / n
    return [rep[int((i + 0.5) * step)] for i in range(n)]


def accumulate(read, names, full0, full1, pot):
    acc = [np.zeros(K), np.zeros(K)]
    wacc = [np.zeros(K), np.zeros(K)]
    gaps = []
    for nm in names:
        e = fl.decode(read(nm))
        per, gap = LV.entry_values(e, full0, full1, pot)
        gaps.append(gap)
        for p, (ix, share, dw) in enumerate(per):
            d = dw * (full0, full1)[p][ix]
            acc[p] += np.bincount(CLASS_OF[ix], weights=d * share, minlength=K)
            wacc[p] += np.bincount(CLASS_OF[ix], weights=d, minlength=K)
    out = {}
    for p, side in ((0, "oop"), (1, "ip")):
        out[side] = {"share": np.where(wacc[p] > 0, acc[p] / np.maximum(wacc[p], 1e-300), 0.0),
                     "w": wacc[p] / max(wacc[p].sum(), 1e-300),
                     "range": to_classes(full0 if p == 0 else full1)}
    return out, float(np.abs(gaps).mean()) if gaps else 0.0


def excess(v: dict, side: str) -> tuple:
    """(class weights meeting the flop, each class's share minus its equity against the other side's range)."""
    from gto.pf_classes import PAIR_COMPAT, equity_matrix
    other = "ip" if side == "oop" else "oop"
    eq = equity_matrix()
    ro = np.asarray(v[other]["range"], float)
    P = PAIR_COMPAT * ro[None, :]
    e = (P * eq).sum(1) / np.maximum(P.sum(1), 1e-12)
    return np.asarray(v[side]["w"], float), np.asarray(v[side]["share"], float) - e


def fit(path: str) -> dict:
    """{depth: {pot type: k}} from a module this script wrote (see --fit), printing each spot's."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("depthvals", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vals = mod.SPOTS
    by: dict = {}
    for key in sorted(vals, key=lambda k: (k.split("@")[0], float(k.split("@")[1]))):
        sid, d = key.split("@")
        if d == "100" or f"{sid}@100" not in vals:
            continue
        v, b = vals[key], vals[f"{sid}@100"]
        num = den = 0.0
        parts = []
        for side in ("oop", "ip"):
            w, x = excess(v, side)
            wb, xb = excess(b, side)
            both = (w > 0) & (wb > 0)
            num += float((w * x * xb)[both].sum())
            den += float((w * xb * xb)[both].sum())
            parts.append(f"{side} edge {float((w * x).sum()):+.4f} (100bb {float((wb * xb).sum()):+.4f}), "
                         f"spread {float((w * np.abs(x)).sum()):.4f} ({float((wb * np.abs(xb)).sum()):.4f})")
        k = num / max(den, 1e-12)
        pt = min(str(v["line"]).count(":r"), 3)
        by.setdefault(float(d), {}).setdefault(pt, []).append(k)
        print(f"{key}: k {k:.3f}, pot {v['pot']:g} stack {v['stack']:g}; " + "; ".join(parts))
    out = {d: {pt: float(np.mean(ks)) for pt, ks in sorted(x.items())} for d, x in sorted(by.items())}
    for d, x in out.items():
        print(f"{d:g}bb: " + ", ".join(f"pot type {pt} k {k:.3f}" for pt, k in x.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="", help="a module this script wrote: print the depth scales")
    ap.add_argument("--spots", default="srp_btn_bb,srp_sb_bb,3bp_bb_btn")
    ap.add_argument("--depths", default="20,40,200")
    ap.add_argument("--flops", type=int, default=23)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lib", action="append", default=None)
    ap.add_argument("--values-only", action="store_true", help="no solving: read what the cache holds")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.fit:
        fit(a.fit)
        return
    if not (a.cache and a.out):
        ap.error("--cache and --out are required")
    lines = {s[0]: s[2] for s in fl.SPOTS}
    names = subset(a.flops)
    vals = {}
    for sid in [x for x in a.spots.split(",") if x]:
        # the 100bb baseline: the library's entries for these flops (the first --lib holding all of them)
        for lib in a.lib or [fl.DIR]:
            try:
                idx = json.load(open(os.path.join(lib, "index.json")))
            except (OSError, ValueError):
                continue
            sp = next((s for s in idx["spots"] if s["id"] == sid), None)
            if sp is None or not all(nm in sp["flops"] for nm in names):
                continue
            rg = json.load(open(os.path.join(lib, sid, "ranges.json")))
            f0, f1 = (np.asarray(rg["w"][p], float) / 10000.0 for p in (0, 1))
            out, gap = accumulate(lambda nm: open(os.path.join(lib, sid, nm + ".bin"), "rb").read(), names, f0, f1,
                                  float(sp["pot"]))
            vals[f"{sid}@100"] = dict(out, seats=sp["seats"], pot=float(sp["pot"]), stack=float(sp["stack"]),
                                      flops=len(names), line=sp["line"])
            print(f"{sid}@100: library {lib}, gap {gap:.4f}", flush=True)
            break
        for depth in [float(x) for x in a.depths.split(",") if x]:
            s = fl.spot_setup(lines[sid], stack=depth)
            cfg = fl.library_config(s["pot"], s["stack"])
            meta = {"spot": sid, "fp": [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])], "pot": s["pot"],
                    "stack": s["stack"], "tree": fl.tree_key(cfg), "seats": [s["oop"], s["ip"]]}
            d = os.path.join(a.cache, f"{sid}_{depth:g}bb")
            os.makedirs(d, exist_ok=True)
            for nm in names:
                path = os.path.join(d, nm + ".bin")
                if os.path.exists(path) or a.values_only:
                    continue
                t0 = time.time()
                sol = solve(parse_cards(nm), s["full0"], s["full1"], s["pot"], s["stack"], cfg=cfg, time_budget=1e9,
                            max_iters=a.iters)
                ex = sol.exploitability()
                with open(path + ".tmp", "wb") as f:
                    f.write(fl.encode(sol, dict(meta, board=nm)))
                os.replace(path + ".tmp", path)
                print(f"{sid}@{depth:g} {nm}: {sol.iters} iters, {ex:.3f}% of the pot, {time.time() - t0:.0f}s", flush=True)
            have = [nm for nm in names if os.path.exists(os.path.join(d, nm + ".bin"))]
            if not have:
                continue
            out, gap = accumulate(lambda nm: open(os.path.join(d, nm + ".bin"), "rb").read(), have, s["full0"],
                                  s["full1"], s["pot"])
            vals[f"{sid}@{depth:g}"] = dict(out, seats=[s["oop"], s["ip"]], pot=s["pot"], stack=s["stack"],
                                            flops=len(have), line=lines[sid])
            print(f"{sid}@{depth:g}: {len(have)} flops, pot {s['pot']}, stack {s['stack']}, gap {gap:.4f}", flush=True)
    LV.write_module(vals, a.out, f"Live solves at other depths on {len(names)} flops ({','.join(names)}), "
                                 f"{a.iters} iterations; @100 the library's entries for the same flops.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
