"""What reading the unraked flop library for a raked game gives up, against the flop solved live with the rake.

The flop library is solved without rake, so with rake set the advisor used to solve every flop live. Here each flop a
spot stores is solved live with the rake (the browser's Accurate tree, natively, one thread: a reference neither the
library nor the read uses), and the library's unraked entry is priced by it at every decision the entry stores: each
combo's mix priced by the raked solve's EVs per action (bench_flop_approx.given_up: the reference mix's EV less the
read's, weighted by the acting player's reach, % of the pot entering the flop; per flop over its stored decisions
weighted by how often each comes, the reference's own line frequencies). For comparison, on the same flops:

  lib0   the library's entry priced by the flop solved the same way without rake (the trees' own distance)
  pure   that unraked solve's mix priced by the raked one (what ignoring the rake alone gives up, in one tree)
  fast   the browser's Fast tree solved live with the rake, at each checkpoint, priced by the raked reference
         (what the advisor answered with rake set: the Fast flop runs ~90-200 iterations in the browser's 8 s)
  tv     strategy distance: half the L1 distance between two mixes per combo, reach-weighted, % (tv_lib: the
         library's entry against the raked solve; tv_pure: the unraked solve against the raked one)

Rakes as RATE-CAP (percent, cap in bb; 0 = no cap), "no flop, no drop" as the solver plays it.

--ev: the EVs instead, on the same flops: what the entry says each action is worth (bb, the advisor's ev_bb) against
the raked solve's, per combo weighted by the reference's mix, reach-weighted, % of the pot entering the flop (ev_abs
the mean gap, ev_bias its signed mean: above 0, the unraked entry's EVs run high), and the same against the flop solved
without rake (ev0_abs, ev0_bias: the library's own EV error, rake aside).

    python scripts/bench_rake_read.py OUT.jsonl --spots srp_btn_bb,3bp_bb_btn --n 4 --rakes 10-3,5-1
    python scripts/bench_rake_read.py OUT.jsonl --report
--raked-ranges: the ranges a raked table brings to the flop. With rake on, the advisor re-prices the preflop calls by
the rake they go on to pay (advisor._call_rake), so the caller's range entering the flop is narrower than the library
spot's (the big blind's defence against the button at 10% capped at 3bb: 218 combos, against 576) and the library's
range fingerprints no longer find the spot. Here each flop is solved with those ranges and the rake (the reference),
and the entry's mix for those combos (every one of them in the entry's ranges) is priced by it, as the advisor would
answer were it to read the spot's entry for the raked table: "lib" its loss, "tv_lib" its distance, "fast" the Fast
raked solve of those ranges, "ev_abs"/"ev_bias" the entry's EVs against the reference's, "dist" the two ranges'
distance from the spot's (flop_approx.range_dist, 0-2).

    python scripts/bench_rake_read.py RR.jsonl --raked-ranges --rakes 10-3,5-3,10-1 [--report]
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
from gto import solver as S  # noqa: E402
from gto.cards import cards as parse_cards  # noqa: E402
from gto.flop_approx import N_COMBOS  # noqa: E402
import bench_flop_approx as BA  # noqa: E402
import bench_flop_seed as BS  # noqa: E402


def rake_of(s: str):
    """'10-3' -> (0.10, 3.0); '5-0' -> (0.05, None); '0' -> None."""
    if s in ("0", "", "off"):
        return None
    r, _, c = s.partition("-")
    return float(r) / 100.0, (float(c) if c and float(c) > 0 else None)


def cfg_for(name: str, rake) -> S.Config:
    cfg = BS.config(name)
    if rake:
        cfg.rake_rate, cfg.rake_cap = rake
    return cfg


def solve(spot, s, board, cfg, iters):
    t0 = time.time()
    sol = S.solve(board, s["full0"], s["full1"], float(spot["pot"]), float(spot["stack"]), cfg=cfg, time_budget=1e9,
                  max_iters=iters)
    return sol, time.time() - t0


def as_entry(sol, spot, s, board, cfg):
    meta = {"spot": spot["id"], "fp": [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])], "pot": float(spot["pot"]),
            "stack": float(spot["stack"]), "tree": fl.tree_key(cfg), "board": fl.cards_str(board)}
    return fl.decode(fl.encode(sol, meta))


def full_mix(nd, mask):
    """An entry node's mix over all 1326 combos (NaN off the player's range)."""
    st = np.full((len(nd["names"]), len(mask)), np.nan)
    st[:, np.nonzero(mask)[0]] = nd["strat"]
    return st


def tv(nd_ref, mask_ref, reach, mix):
    """Half the L1 distance between the reference's mix and `mix` (A, 1326), reach-weighted over the reference's
    combos that `mix` answers, % (0: the same mix, 100: no action in common)."""
    idx = np.nonzero(mask_ref)[0]
    P = mix[:, idx]
    ok = ~np.isnan(P[0])
    w = reach[idx] * ok
    d = 0.5 * np.abs(nd_ref["strat"] - np.nan_to_num(P)).sum(axis=0)
    return float((w * d).sum() / max(float(w.sum()), 1e-12) * 100)


def price(ref, reaches, pot, mixes):
    """(loss, tv, {path: loss}) of `mixes` ({path: (names, (A, 1326))}) at the reference's stored nodes, per flop
    weighted by each node's share of the hands."""
    num = den = tnum = 0.0
    per = {}
    for nd in ref["nodes"]:
        path = tuple(nd["path"])
        m = mixes.get(path)
        r, share = reaches[path]
        if m is None or m[0] != list(nd["names"]) or not share:
            continue
        mask = ref["masks"][nd["player"]]
        loss, _, _ = BA.given_up(nd, mask, r, m[1], pot, seen=True)
        d = tv(nd, mask, r, m[1])
        per["|".join(path) or "root"] = [round(loss, 4), round(d, 2), round(share, 4)]
        num += share * loss
        tnum += share * d
        den += share
    return (num / den if den else None), (tnum / den if den else None), per


def entry_mixes(e):
    return {tuple(nd["path"]): (list(nd["names"]), full_mix(nd, e["masks"][nd["player"]])) for nd in e["nodes"]}


def sol_mixes(sol, ref):
    """The live solve's average mixes at the reference's stored paths."""
    act = sol.ctx.active
    out = {}
    for nd in ref["nodes"]:
        path = tuple(nd["path"])
        try:
            n = fl.node_at(sol.root, path)
        except (ValueError, IndexError, AttributeError):
            continue
        if not isinstance(n, S.Decision):
            continue
        st = np.full((len(n.names), N_COMBOS), np.nan)
        st[:, act] = n.avg_strategy()[0]
        out[path] = (list(n.names), st)
    return out


def root_freqs(e):
    nd = next(n for n in e["nodes"] if not n["path"])
    tot = max(sum(nd["freqs"].values()), 1e-12)
    return {k: round(v / tot * 100, 1) for k, v in nd["freqs"].items()}


def run_flop(lib, spot, s, name, rakes, ref_iters, cks, done, out):
    pot = float(spot["pot"])
    board = parse_cards(name)
    e = lib.entry(spot, name)
    if e is None:
        print(spot["id"], name, "no entry", flush=True)
        return
    lib_mix = entry_mixes(e)
    key0 = (spot["id"], name, "0")
    ref0 = None
    if key0 not in done or any((spot["id"], name, r) not in done for r in rakes):
        sol0, t0 = solve(spot, s, board, cfg_for("accurate", None), ref_iters)
        ref0 = as_entry(sol0, spot, s, board, cfg_for("accurate", None))
        x0 = sol0.exploitability()
        del sol0
    if key0 not in done:
        reaches0 = BA.node_reaches(ref0, s["full0"], s["full1"])
        loss0, tv0, per0 = price(ref0, reaches0, pot, lib_mix)
        row = {"spot": spot["id"], "flop": name, "rake": "0", "ref_iters": ref_iters, "ref_expl": round(x0, 4),
               "ref_s": round(t0, 1), "lib0": round(loss0, 4), "tv_lib0": round(tv0, 2), "nodes_lib0": per0,
               "freqs": {"lib": root_freqs(e), "ref": root_freqs(ref0)}}
        _write(out, row)
        print(spot["id"], name, "rake 0", f"lib0 {loss0:.3f} tv {tv0:.1f}", f"ref {x0:.3f}% {t0:.0f}s", flush=True)
    for rk in rakes:
        if (spot["id"], name, rk) in done:
            continue
        rake = rake_of(rk)
        cfg = cfg_for("accurate", rake)
        sol, t1 = solve(spot, s, board, cfg, ref_iters)
        ref = as_entry(sol, spot, s, board, cfg)
        x = sol.exploitability()
        del sol
        reaches = BA.node_reaches(ref, s["full0"], s["full1"])
        loss, tvl, per = price(ref, reaches, pot, lib_mix)
        pure, tvp, perp = price(ref, reaches, pot, entry_mixes(ref0))
        lib0, _, _ = price(ref0, BA.node_reaches(ref0, s["full0"], s["full1"]), pot, lib_mix)
        fast = {}
        t2 = time.time()
        fsol = S.solve(board, s["full0"], s["full1"], pot, float(spot["stack"]), cfg=cfg_for("fast", rake),
                       time_budget=1e9, max_iters=0)
        for ck in cks:
            fsol.solver.run(fsol.r0, fsol.r1, time_budget=1e9, max_iters=ck, min_iters=1)
            fsol.iters = fsol.solver.iters
            g, d, _ = price(ref, reaches, pot, sol_mixes(fsol, ref))
            fast[str(ck)] = [round(g, 4), round(d, 2), round(fsol.exploitability(), 4)]
        t2 = time.time() - t2
        del fsol
        row = {"spot": spot["id"], "flop": name, "rake": rk, "ref_iters": ref_iters, "ref_expl": round(x, 4),
               "ref_s": round(t1, 1), "fast_s": round(t2, 1), "lib": round(loss, 4), "lib0": round(lib0, 4),
               "pure": round(pure, 4), "tv_lib": round(tvl, 2), "tv_pure": round(tvp, 2), "fast": fast,
               "nodes_lib": per, "nodes_pure": perp,
               "freqs": {"lib": root_freqs(e), "ref0": root_freqs(ref0), "ref": root_freqs(ref)}}
        _write(out, row)
        print(spot["id"], name, rk, f"lib {loss:.3f} (lib0 {lib0:.3f}, pure {pure:.3f}) tv {tvl:.1f}/{tvp:.1f}",
              "fast", {k: v[0] for k, v in fast.items()}, f"ref {x:.3f}% {t1:.0f}s", flush=True)


def ev_gap(ref, e, reaches, pot):
    """(mean |gap|, mean gap) of entry e's EVs per action against the reference's at its stored nodes: each combo's
    gaps weighted by the reference's mix there, then by its reach, each node by its share of the hands; % of pot."""
    lib = {tuple(nd["path"]): nd for nd in e["nodes"]}
    num = nab = den = 0.0
    for nd in ref["nodes"]:
        path = tuple(nd["path"])
        ln = lib.get(path)
        r, share = reaches[path]
        if ln is None or list(ln["names"]) != list(nd["names"]) or not share:
            continue
        p = nd["player"]
        mr, ml = ref["masks"][p], e["masks"][p]
        idx = np.nonzero(mr & ml)[0]
        ir, il = (np.cumsum(mr) - 1)[idx], (np.cumsum(ml) - 1)[idx]
        d = ln["evs"][:, il] - nd["evs"][:, ir]
        mix = nd["strat"][:, ir]
        w = r[idx]
        if not w.sum():
            continue
        num += share * float((w * (d * mix).sum(axis=0)).sum() / w.sum())
        nab += share * float((w * (np.abs(d) * mix).sum(axis=0)).sum() / w.sum())
        den += share
    return (nab / den / pot * 100, num / den / pot * 100) if den else (None, None)


def run_flop_ev(lib, spot, s, name, rakes, ref_iters, done, out):
    pot = float(spot["pot"])
    board = parse_cards(name)
    e = lib.entry(spot, name)
    if e is None or all((spot["id"], name, r) in done for r in rakes):
        return
    sol0, _ = solve(spot, s, board, cfg_for("accurate", None), ref_iters)
    ref0 = as_entry(sol0, spot, s, board, cfg_for("accurate", None))
    del sol0
    a0, b0 = ev_gap(ref0, e, BA.node_reaches(ref0, s["full0"], s["full1"]), pot)
    for rk in rakes:
        if (spot["id"], name, rk) in done:
            continue
        cfg = cfg_for("accurate", rake_of(rk))
        sol, t1 = solve(spot, s, board, cfg, ref_iters)
        ref = as_entry(sol, spot, s, board, cfg)
        del sol
        a, b = ev_gap(ref, e, BA.node_reaches(ref, s["full0"], s["full1"]), pot)
        row = {"spot": spot["id"], "flop": name, "rake": rk, "ev_abs": round(a, 4), "ev_bias": round(b, 4),
               "ev0_abs": round(a0, 4), "ev0_bias": round(b0, 4)}
        _write(out, row)
        print(spot["id"], name, rk, f"EV gap {a:.3f} (bias {b:+.3f}); unraked {a0:.3f} ({b0:+.3f})", f"{t1:.0f}s", flush=True)


def report_ev(path):
    rows = [json.loads(x) for x in open(path)]
    by = collections.defaultdict(list)
    for r in rows:
        kind = "3bet" if r["spot"].startswith("3bp") else "single-raised"
        by[(kind, r["rake"])].append(r)
        by[("all", r["rake"])].append(r)
    print("Entry EVs against the solve's, % of the pot, per flop: mean / median / 90th pct / worst")
    for (kind, rk), rs in sorted(by.items()):
        print(f"\n{kind}, rake {rk} ({len(rs)} flops)")
        print("  |gap|, raked solve   ", _stats([r["ev_abs"] for r in rs]))
        print("  bias, raked solve    ", _stats([r["ev_bias"] for r in rs]))
        print("  |gap|, unraked solve ", _stats([r["ev0_abs"] for r in rs]))
        print("  bias, unraked solve  ", _stats([r["ev0_bias"] for r in rs]))


def _write(out, row):
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")


def _stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "-"
    n = len(xs)
    return f"{sum(xs) / n:.3f} / {xs[n // 2]:.3f} / {xs[min(n - 1, int(0.9 * n))]:.3f} / {xs[-1]:.3f}"


def report(path):
    rows = [json.loads(x) for x in open(path)]
    by = collections.defaultdict(list)
    for r in rows:
        if r["rake"] != "0":
            kind = "3bet" if r["spot"].startswith("3bp") else "single-raised"
            by[(kind, r["rake"])].append(r)
            by[("all", r["rake"])].append(r)
    print("EV given up, % of the pot, per flop: mean / median / 90th pct / worst")
    for (kind, rk), rs in sorted(by.items()):
        cks = sorted({int(k) for r in rs for k in r["fast"]})
        print(f"\n{kind}, rake {rk} ({len(rs)} flops)")
        print("  library read (unraked entry)  ", _stats([r["lib"] for r in rs]))
        print("  same, priced without rake      ", _stats([r["lib0"] for r in rs]))
        print("  unraked solve, same tree       ", _stats([r["pure"] for r in rs]))
        for ck in cks:
            print(f"  Fast raked solve @{ck:<4d}       ", _stats([r["fast"][str(ck)][0] for r in rs if str(ck) in r["fast"]]))
        print("  strategy distance lib / pure, %", _stats([r["tv_lib"] for r in rs]), "|", _stats([r["tv_pure"] for r in rs]))
        better = {ck: sum(r["lib"] <= r["fast"][str(ck)][0] for r in rs if str(ck) in r["fast"]) for ck in cks}
        print("  library read better than Fast on", {k: f"{v}/{len(rs)}" for k, v in better.items()})
        print("  reference exploitability       ", _stats([r["ref_expl"] for r in rs]), "secs", _stats([r["ref_s"] for r in rs]))


class _Got(BaseException):
    pass


_RR = {}


def raked_ranges(spot_id: str, line: str, oop: str, rake):
    """The two ranges entering the flop at a 6-max 100bb table with `rake` (the advisor's: its preflop calls re-priced
    by the rake), caught at its library lookup."""
    key = (spot_id, rake)
    if key not in _RR:
        from gto import native
        from gto.cli import parse_actions
        adv = native.advisor("accurate", budget=0.1, prior_budget=0.1)
        adv.rake_library = False
        orig = fl.FlopLibrary.solution

        def spy(self, full0, full1, *a, **k):
            raise _Got((full0.copy(), full1.copy()))
        fl.FlopLibrary.solution = spy
        try:
            adv.advise({"players": 6, "stack": 100, "preflop": parse_actions(line), "board": "Kh7s2d", "hero": oop,
                        "hole": "AsAc", "rake": rake[0], "rake_cap": rake[1]})
            got = None
        except _Got as g:
            got = g.args[0]
        finally:
            fl.FlopLibrary.solution = orig
        _RR[key] = got
    return _RR[key]


def run_flop_rr(lib, spot, s, name, rakes, ref_iters, cks, done, out):
    from gto.flop_approx import range_dist
    pot = float(spot["pot"])
    board = parse_cards(name)
    e = lib.entry(spot, name)
    if e is None or all((spot["id"], name, r) in done for r in rakes):
        return
    lib_mix = entry_mixes(e)
    line = dict((x[0], x[2]) for x in fl.SPOTS)[spot["id"]]
    for rk in rakes:
        if (spot["id"], name, rk) in done:
            continue
        rake = rake_of(rk)
        got = raked_ranges(spot["id"], line, s["oop"], rake)
        if got is None:
            print(spot["id"], rk, "no flop ranges", flush=True)
            continue
        sr = dict(s, full0=got[0], full1=got[1])
        dist = [round(range_dist(got[0], s["full0"]), 3), round(range_dist(got[1], s["full1"]), 3)]
        cover = [round(float((got[p] * (s["full" + str(p)] > 0)).sum() / max(got[p].sum(), 1e-12)), 4) for p in (0, 1)]
        cfg = cfg_for("accurate", rake)
        sol, t1 = solve(spot, sr, board, cfg, ref_iters)
        ref = as_entry(sol, spot, sr, board, cfg)
        x = sol.exploitability()
        del sol
        reaches = BA.node_reaches(ref, sr["full0"], sr["full1"])
        loss, tvl, per = price(ref, reaches, pot, lib_mix)
        ea, eb = ev_gap(ref, e, reaches, pot)
        fast = {}
        t2 = time.time()
        fsol = S.solve(board, sr["full0"], sr["full1"], pot, float(spot["stack"]), cfg=cfg_for("fast", rake),
                       time_budget=1e9, max_iters=0)
        for ck in cks:
            fsol.solver.run(fsol.r0, fsol.r1, time_budget=1e9, max_iters=ck, min_iters=1)
            fsol.iters = fsol.solver.iters
            g, d, _ = price(ref, reaches, pot, sol_mixes(fsol, ref))
            fast[str(ck)] = [round(g, 4), round(d, 2), round(fsol.exploitability(), 4)]
        t2 = time.time() - t2
        del fsol
        row = {"spot": spot["id"], "flop": name, "rake": rk, "mode": "raked_ranges", "ref_iters": ref_iters,
               "ref_expl": round(x, 4), "ref_s": round(t1, 1), "fast_s": round(t2, 1), "lib": round(loss, 4),
               "tv_lib": round(tvl, 2), "fast": fast, "ev_abs": round(ea, 4) if ea is not None else None,
               "ev_bias": round(eb, 4) if eb is not None else None, "dist": dist, "cover": cover,
               "combos": [round(float(got[0].sum()), 1), round(float(got[1].sum()), 1)],
               "combos_lib": [round(float(s["full0"].sum()), 1), round(float(s["full1"].sum()), 1)],
               "nodes_lib": per, "freqs": {"lib": root_freqs(e), "ref": root_freqs(ref)}}
        _write(out, row)
        print(spot["id"], name, rk, f"lib {loss:.3f} tv {tvl:.1f} ev {ea:.2f} ({eb:+.2f}) dist {dist}",
              "fast", {k: v[0] for k, v in fast.items()}, f"ref {x:.3f}% {t1:.0f}s", flush=True)


def report_rr(path):
    rows = [json.loads(x) for x in open(path)]
    by = collections.defaultdict(list)
    for r in rows:
        kind = "3bet" if r["spot"].startswith("3bp") else "single-raised"
        by[(kind, r["rake"])].append(r)
        by[("all", r["rake"])].append(r)
    print("The raked table's ranges: the entry's mix priced by their raked solve, % of the pot: mean / median / 90th / worst")
    for (kind, rk), rs in sorted(by.items()):
        cks = sorted({int(k) for r in rs for k in r["fast"]})
        print(f"\n{kind}, rake {rk} ({len(rs)} flops)")
        print("  library entry read           ", _stats([r["lib"] for r in rs]))
        for ck in cks:
            print(f"  Fast raked solve @{ck:<4d}     ", _stats([r["fast"][str(ck)][0] for r in rs if str(ck) in r["fast"]]))
        print("  strategy distance, %         ", _stats([r["tv_lib"] for r in rs]))
        print("  EV |gap| / bias              ", _stats([r["ev_abs"] for r in rs]), "|", _stats([r["ev_bias"] for r in rs]))
        print("  range distance (caller's)    ", _stats([max(r["dist"]) for r in rs]))
        better = {ck: sum(r["lib"] <= r["fast"][str(ck)][0] for r in rs if str(ck) in r["fast"]) for ck in cks}
        print("  entry read better than Fast on", {k: f"{v}/{len(rs)}" for k, v in better.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--spots", default="srp_btn_bb,srp_co_bb,srp_sb_bb,3bp_bb_btn,3bp_sb_btn,3bp_btn_co")
    ap.add_argument("--n", type=int, default=4, help="flops a spot (from its 552 representative flops)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rakes", default="10-3,5-1")
    ap.add_argument("--ref-iters", type=int, default=600)
    ap.add_argument("--checkpoints", default="50,100,200")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--ev", action="store_true", help="the entries' EVs against the raked solve's (see above)")
    ap.add_argument("--raked-ranges", action="store_true", help="the raked table's own flop ranges (see above)")
    a = ap.parse_args(argv)
    if a.report:
        return report_rr(a.out) if a.raked_ranges else report_ev(a.out) if a.ev else report(a.out)
    lib = fl.FlopLibrary.from_dir()
    spots = {x["id"]: x for x in lib.index()["spots"]}
    rakes = a.rakes.split(",")
    cks = [int(x) for x in a.checkpoints.split(",")]
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            d = json.loads(x)
            done.add((d["spot"], d["flop"], d["rake"]))
    rep = set(fl.cards_str(f) for f in fl.representative_flops(552))
    for sid in a.spots.split(","):
        spot = spots[sid]
        s = BA.setup(sid)
        if [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])] != spot["fp"]:
            print(sid, "ranges differ from the index's: skipped", flush=True)
            continue
        names = sorted(n for n in spot["flops"] if n in rep)
        random.Random(a.seed).shuffle(names)
        for name in names[:a.n]:
            if a.raked_ranges:
                run_flop_rr(lib, spot, s, name, rakes, a.ref_iters, cks, done, a.out)
            elif a.ev:
                run_flop_ev(lib, spot, s, name, rakes, a.ref_iters, done, a.out)
            else:
                run_flop(lib, spot, s, name, rakes, a.ref_iters, cks, done, a.out)
    report_rr(a.out) if a.raked_ranges else report_ev(a.out) if a.ev else report(a.out)


if __name__ == "__main__":
    main()
