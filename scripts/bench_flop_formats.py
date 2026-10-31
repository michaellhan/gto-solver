"""Other games' flops (heads-up, 8- and 9-handed, antes, a straddle) read off the 6-max 100bb library: what that gives up.

The library's spots are 6-max at 100bb; another game's pot has its own ranges (that game's charts, gto/pf_formats.py),
pot and stack. Here a case's flop (a preflop line in a game) is solved natively in the browser's Accurate tree, one
thread (--iters; the reference, a solve the answers don't read), made an entry (flop_library.encode: the decisions an
entry stores, each action's EV, as bench_flop_depth.py does), and each candidate answer's mixes are priced by it
(bench_flop_approx.given_up: each combo's mix priced by the reference's EVs per action, the acting player's reach, %
of the pot; per flop over the stored decisions by how often each comes):

* lib: the source spot's library entry of the flop, over the combos it holds (as the advisor answers: a combo the
  source's range doesn't hold has no answer; "unseen" is the reach-weighted share left so);
* lib_all: the same with the unanswered combos at the node's range average;
* libcl: lib with flop_approx's learned correction;
* fill: lib with each unanswered combo playing its hand class's average mix among the combos the entry holds
  (flop_approx.class_fill), over every combo;
* fast50 (--fast): the case's own flop solved live in the browser's Fast tree to 50 iterations (the approximate
  answer's hand-off, advisor.NEAR_UNTIL).

The reference entries are kept (TRUTH_DIR/<case>/<flop>.bin) so other cases' can price each other (--cross A:B: case
A's reference read as an answer for case B, e.g. the 6-max spot's own solve for an 8-handed pot: the ranges' cost
alone). One JSON line a (case, flop) to OUT; resumable.

    python scripts/bench_flop_formats.py OUT.jsonl --cases 8max_btn_bb,ante_btn_bb --n 6
    python scripts/bench_flop_formats.py OUT.jsonl --report
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
import bench_flop_seed as SEED  # noqa: E402
from bench_flop_depth import as_pred, price  # noqa: E402

# case -> (Format tag, preflop line in that game, the library spot read)
CASES = {
    "6max_btn_bb": ("", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", "srp_btn_bb"),
    "6max_co_bb": ("", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c", "srp_co_bb"),
    "6max_sb_bb": ("", "UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:c", "srp_sb_bb"),
    "6max_utg_bb": ("", "UTG:r2.5,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_utg_bb"),
    "8max_btn_bb": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", "srp_btn_bb"),
    "8max_co_bb": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c", "srp_co_bb"),
    "8max_sb_bb": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:f,SB:r3,BB:c", "srp_sb_bb"),
    "8max_lj_bb": ("8max", "UTG:f,UTG1:f,LJ:r2.5,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_utg_bb"),
    "9max_utg2_bb": ("9max", "UTG:f,UTG1:f,UTG2:r2.5,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_utg_bb"),
    "9max_utg_bb": ("9max", "UTG:r2.5,UTG1:f,UTG2:f,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_utg_bb"),
    "ante_btn_bb": ("a0.125", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", "srp_btn_bb"),
    "ante_co_bb": ("a0.125", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c", "srp_co_bb"),
    "ante_sb_bb": ("a0.125", "UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:c", "srp_sb_bb"),
    "ante_hj_bb": ("a0.125", "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:f,BB:c", "srp_hj_bb"),
    "ante_3bp_bb_btn": ("a0.125", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c", "3bp_bb_btn"),
    "straddle_bb_utg": ("straddle", "HJ:f,CO:f,BTN:f,SB:f,BB:r5,UTG:c", "srp_sb_bb"),
    "straddle_btn_utg": ("straddle", "HJ:f,CO:f,BTN:r5,SB:f,BB:f,UTG:c", "srp_co_bb"),
    "straddle_co_utg": ("straddle", "HJ:f,CO:r5,BTN:f,SB:f,BB:f,UTG:c", "srp_hj_bb"),
    "ante_utg_bb": ("a0.125", "UTG:r2.5,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_utg_bb"),
    "ante_3bp_sb_btn": ("a0.125", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:r10,BB:f,BTN:c", "3bp_sb_btn"),
    "ante_btn_sb": ("a0.125", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:f", "srp_btn_sb"),
    "straddle_hj_utg": ("straddle", "HJ:r5,CO:f,BTN:f,SB:f,BB:f,UTG:c", "srp_utg_bb"),
    "straddle_sb_utg": ("straddle", "HJ:f,CO:f,BTN:f,SB:r6,BB:f,UTG:c", "srp_sb_bb"),
    "straddle_bb_co": ("straddle", "HJ:f,CO:r5,BTN:f,SB:f,BB:c,UTG:f", "srp_co_sb"),
    "hu_sb_bb": ("2max", "SB:r3,BB:c", "srp_btn_bb"),
    "hu_3bp_bb_sb": ("2max", "SB:r3,BB:r10,SB:c", "3bp_bb_btn"),
    "6max_sqz_sb_bb": ("", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:r10,BB:c,BTN:f", "3bp_sb_btn"),
    "6max_btn_sb": ("", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:f", "srp_btn_sb"),
    "6max_hj_bb": ("", "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:f,BB:c", "srp_hj_bb"),
    "6max_3bp_bb_btn": ("", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c", "3bp_bb_btn"),
    "6max_3bp_sb_btn": ("", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:r10,BB:f,BTN:c", "3bp_sb_btn"),
    "6max_3bp_sb_co": ("", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:r10,BB:f,CO:c", "3bp_sb_co"),
    "8max_hj_bb": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:r2.5,CO:f,BTN:f,SB:f,BB:c", "srp_hj_bb"),
    "8max_btn_sb": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:f", "srp_btn_sb"),
    "8max_3bp_bb_btn": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c", "3bp_bb_btn"),
    "8max_3bp_sb_btn": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:r10,BB:f,BTN:c", "3bp_sb_btn"),
    "8max_3bp_sb_co": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:f,SB:r10,BB:f,CO:c", "3bp_sb_co"),
    "8max_utg1_bb": ("8max", "UTG:f,UTG1:r2.5,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_utg_bb"),
    "8max_utg_utg1": ("8max", "UTG:r2.5,UTG1:c,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:f", "srp_utg_btn"),
    "9max_btn_bb": ("9max", "UTG:f,UTG1:f,UTG2:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", "srp_btn_bb"),
    "6max_co_sb": ("", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:f", "srp_co_sb"),
    "8max_co_sb": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:f", "srp_co_sb"),
    "6max_3bp_bb_co": ("", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:r10,CO:c", "3bp_bb_co"),
    "8max_3bp_bb_co": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:r10,CO:c", "3bp_bb_co"),
    "6max_3bp_btn_co": ("", "UTG:f,HJ:f,CO:r2.5,BTN:r7.5,SB:f,BB:f,CO:c", "3bp_btn_co"),
    "8max_3bp_btn_co": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:r7.5,SB:f,BB:f,CO:c", "3bp_btn_co"),
    "6max_hj_co": ("", "UTG:f,HJ:r2.5,CO:c,BTN:f,SB:f,BB:f", "srp_hj_co"),
    "8max_hj_co": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:r2.5,CO:c,BTN:f,SB:f,BB:f", "srp_hj_co"),
    "6max_utg_btn": ("", "UTG:r2.5,HJ:f,CO:f,BTN:c,SB:f,BB:f", "srp_utg_btn"),
    "8max_lj_btn": ("8max", "UTG:f,UTG1:f,LJ:r2.5,HJ:f,CO:f,BTN:c,SB:f,BB:f", "srp_utg_btn"),
    "6max_utg_hj": ("", "UTG:r2.5,HJ:c,CO:f,BTN:f,SB:f,BB:f", "srp_utg_hj"),
    "8max_lj_co": ("8max", "UTG:f,UTG1:f,LJ:r2.5,HJ:f,CO:c,BTN:f,SB:f,BB:f", "srp_utg_hj"),
    "6max_3bp_hj_utg": ("", "UTG:r2.5,HJ:r7.5,CO:f,BTN:f,SB:f,BB:f,UTG:c", "3bp_hj_utg"),
    "8max_3bp_hj_lj": ("8max", "UTG:f,UTG1:f,LJ:r2.5,HJ:r7.5,CO:f,BTN:f,SB:f,BB:f,LJ:c", "3bp_hj_utg"),
    "8max_utg_bb": ("8max", "UTG:r2.5,UTG1:f,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_utg_bb"),
    "6max_cc3b_bb_btn": ("", "UTG:f,HJ:f,CO:r2.5,BTN:r7.5,SB:f,BB:c,CO:f", "3bp_btn_co"),
    "6max_sqz_sb_bb_co": ("", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:r10,BB:c,CO:f", "3bp_sb_co"),
    "ante_hj_btn": ("a0.125", "UTG:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:f", "srp_hj_co"),
    "ante_utg_btn": ("a0.125", "UTG:r2.5,HJ:f,CO:f,BTN:c,SB:f,BB:f", "srp_utg_btn"),
    "ante_3bp_btn_co": ("a0.125", "UTG:f,HJ:f,CO:r2.5,BTN:r7.5,SB:f,BB:f,CO:c", "3bp_btn_co"),
    "ante_3bp_bb_sb": ("a0.125", "UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:r10,SB:c", "3bp_bb_sb"),
    "straddle_bb_btn": ("straddle", "HJ:f,CO:f,BTN:r5,SB:f,BB:c,UTG:f", "srp_co_sb"),
    "straddle_sb_hj": ("straddle", "HJ:r5,CO:f,BTN:f,SB:c,BB:f,UTG:f", "3bp_btn_co"),
    "6max_3bp_bb_sb": ("", "UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:r10,SB:c", "3bp_bb_sb"),
    "8max_3bp_bb_sb": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:f,SB:r3,BB:r10,SB:c", "3bp_bb_sb"),
    "6max_3bp_btn_utg": ("", "UTG:r2.5,HJ:f,CO:f,BTN:r7.5,SB:f,BB:f,UTG:c", "3bp_co_hj"),
    "6max_3bp_co_hj": ("", "UTG:f,HJ:r2.5,CO:r7.5,BTN:f,SB:f,BB:f,HJ:c", "3bp_co_hj"),
    "straddle_btn_sb": ("straddle", "HJ:f,CO:f,BTN:r5,SB:c,BB:f,UTG:f", "srp_co_sb"),
    "straddle_bb_hj": ("straddle", "HJ:r5,CO:f,BTN:f,SB:f,BB:c,UTG:f", "srp_co_sb"),
    "straddle_btn_co": ("straddle", "HJ:f,CO:r5,BTN:c,SB:f,BB:f,UTG:f", "srp_hj_co"),
    "ante_3bp_sb_co": ("a0.125", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:r10,BB:f,CO:c", "3bp_sb_co"),
    "ante_hj_co": ("a0.125", "UTG:f,HJ:r2.5,CO:c,BTN:f,SB:f,BB:f", "srp_hj_co"),
    "ante_utg_hj": ("a0.125", "UTG:r2.5,HJ:c,CO:f,BTN:f,SB:f,BB:f", "srp_utg_hj"),
    "ante_3bp_hj_utg": ("a0.125", "UTG:r2.5,HJ:r7.5,CO:f,BTN:f,SB:f,BB:f,UTG:c", "3bp_hj_utg"),
    "ante_3bp_btn_hj": ("a0.125", "UTG:f,HJ:r2.5,CO:f,BTN:r7.5,SB:f,BB:f,HJ:c", "3bp_co_hj"),
}
TRUTH_DIR = os.environ.get("GTO_BENCH_FORMATS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports",
                                                               "bench_formats"))


def case_of(name):
    """(base case, its game's tag, line, the library spot read): NAME or NAME@SPOT (another spot read for it)."""
    base, _, src = name.partition("@")
    tag, line, sid = CASES[base]
    return base, tag, line, src or sid


def setup(case):
    _, tag, line, _ = case_of(case)
    f = fl.Format.parse(tag)
    return fl.spot_setup(line, f.players, f.stack, f.ante, f.straddle)


def solve_entry(board, s, cfg, iters, sid):
    """The flop solved at s's ranges, pot and stack: (encoded entry, exploitability, seconds)."""
    t0 = time.time()
    sol = S.solve(board, s["full0"], s["full1"], s["pot"], s["stack"], cfg=cfg, time_budget=1e9, max_iters=iters)
    meta = {"spot": sid, "fp": [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])], "pot": s["pot"],
            "stack": s["stack"], "tree": fl.tree_key(cfg), "board": fl.cards_str(board)}
    return fl.encode(sol, meta), float(sol.exploitability()), time.time() - t0


def truth(case, name, s, cfg, iters, root=TRUTH_DIR):
    """The case's reference entry of the flop (decoded), solved once and kept; (entry, expl, seconds)."""
    path = os.path.join(root, case, f"{name}.bin")
    meta_path = path + ".json"
    if os.path.exists(path) and os.path.exists(meta_path):
        with open(path, "rb") as f, open(meta_path) as g:
            m = json.load(g)
            return fl.decode(f.read()), m["expl"], m["secs"]
    raw, x, secs = solve_entry(parse_cards(name), s, cfg, iters, case)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(raw)
    with open(meta_path, "w") as g:
        json.dump({"expl": x, "secs": secs, "iters": iters}, g)
    return fl.decode(raw), x, secs


def run(a):
    lib = fl.FlopLibrary.from_dir()
    spots = {x["id"]: x for x in lib.index()["spots"]}
    cfg = SEED.config(a.config)
    fcfg = SEED.config("fast")
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            r = json.loads(x)
            done.add((r["case"], r["flop"]))
    for case in a.cases.split(","):
        base, tag, line, sid = case_of(case)
        s = setup(case)
        ag = fa.spot_aggressor({"seats": [s["oop"], s["ip"]], "line": line})
        ns = fa.near_spot(lib, s["full0"], s["full1"], s["pot"], s["stack"], aggr=ag)
        if ns is not None and "@" not in case:
            sid = ns["id"]                  # (the spot the advisor reads; CASES' where it reads none)
        spot = spots[sid]
        names = sorted(spots[CASES[base][2]]["flops"])   # (the same flops whatever spot is read)
        random.Random(a.seed).shuffle(names)
        names = [n for n in names if n in spot["flops"]]
        src = fa._spot_ranges(spot)
        for name in names[:a.n]:
            if (case, name) in done:
                continue
            board = parse_cards(name)
            e, x, secs = truth(base, name, s, cfg, a.iters)
            le = lib.entry(spot, name)
            rec = {"case": case, "flop": name, "source": sid, "near_spot": ns and ns["id"], "pot": s["pot"],
                   "stack": s["stack"], "spr": round(s["stack"] / s["pot"], 2),
                   "spr_src": round(float(spot["stack"]) / float(spot["pot"]), 2), "expl": round(x, 4),
                   "secs": round(secs, 1),
                   "cover": [round(fa.range_cover(s["full0"], src["full0"]), 3),
                             round(fa.range_cover(s["full1"], src["full1"]), 3)],
                   "dist": round(fa.range_dist(s["full0"], src["full0"]) + fa.range_dist(s["full1"], src["full1"]), 3)}
            ident = [(le, np.arange(52), 1.0)]
            pl = as_pred(le)
            fill = fa.class_fill(as_pred(le), board, (s["full0"], s["full1"]))
            only = {k: dict(o, strat=np.where(np.isnan(pl[k]["strat"][:1]), o["strat"], np.nan)) for k, o in fill.items()
                    if k in pl}
            preds = [("lib", pl, True), ("lib_all", pl, False),
                     ("libcl", fa.predict(ident, board=list(board), correct=True), True),
                     ("fill", fill, False), ("fill_only", only, True)]
            if a.fast:
                t0 = time.time()
                fe = fl.decode(solve_entry(board, s, fcfg, 50, case)[0])
                rec["fast50_secs"] = round(time.time() - t0, 2)
                preds.append(("fast50", as_pred(fe), False))
            for key, pred, seen in preds:
                pred = renamed(pred, e)
                g, per = price(e, s, pred, s["pot"], seen=seen)
                rec[key] = None if g is None else round(g, 4)
                rec[key + "_unseen"] = round(_unseen(e, s, pred), 4)
                rec[key + "_nodes"] = per
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(case, name, rec["lib"], rec["lib_all"], rec["fill"], rec.get("fast50"), rec["lib_unseen"],
                  rec["secs"], flush=True)


def renamed(pred, truth_e):
    """pred's nodes under the reference's action names (a raise named by its size in bb: "raise to 11" in a 5.5 bb
    pot is "raise to 12.5" in a 6.25 bb one; flop_approx.norm_name)."""
    by = {tuple(fa.norm_name(x) for x in nd["path"]): nd for nd in truth_e["nodes"]}
    out = {}
    for path, o in pred.items():
        nd = by.get(tuple(fa.norm_name(x) for x in path))
        if nd is None or [fa.norm_name(x) for x in o["names"]] != [fa.norm_name(x) for x in nd["names"]]:
            out[path] = o
            continue
        out[tuple(nd["path"])] = dict(o, names=list(nd["names"]))
    return out


def _unseen(truth_e, s, pred):
    """The reach-weighted share of the reference's stored decisions (by how often each comes) whose combo the
    answer has no mix for."""
    import bench_flop_approx as BA
    reaches = BA.node_reaches(truth_e, s["full0"], s["full1"])
    num = den = 0.0
    for nd in truth_e["nodes"]:
        path = tuple(nd["path"])
        r, share = reaches[path]
        pr = pred.get(path)
        if not share:
            continue
        idx = np.nonzero(truth_e["masks"][nd["player"]])[0]
        w = r[idx]
        if pr is None or pr["names"] != list(nd["names"]):
            bad = np.ones(len(idx), bool)
        else:
            bad = np.isnan(pr["strat"][0, idx])
        num += share * float(w[bad].sum()) / max(float(w.sum()), 1e-12)
        den += share
    return num / den if den else 0.0


def cross(a):
    """--cross A:B: case A's reference entries read as answers for case B on the flops both hold."""
    for pair in a.cross.split(","):
        ca, cb = pair.split(":")
        sb = setup(cb)
        da, db = os.path.join(TRUTH_DIR, ca), os.path.join(TRUTH_DIR, cb)
        vals = []
        for fn in sorted(os.listdir(db)) if os.path.isdir(db) else []:
            if not fn.endswith(".bin") or not os.path.exists(os.path.join(da, fn)):
                continue
            with open(os.path.join(da, fn), "rb") as f:
                ea = fl.decode(f.read())
            with open(os.path.join(db, fn), "rb") as f:
                eb = fl.decode(f.read())
            g, _ = price(eb, sb, renamed(as_pred(ea), eb), sb["pot"], seen=True)
            vals.append(g)
            print(ca, "->", cb, fn[:-4], None if g is None else round(g, 4))
        v = np.array([x for x in vals if x is not None])
        if len(v):
            print(f"{ca} read for {cb}: n={len(v)} mean {v.mean():.3f} median {np.median(v):.3f} max {v.max():.3f}")


def report(path):
    rows = [json.loads(x) for x in open(path)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    keys = ("lib", "libcl", "fill", "fill_only", "fast50")
    print(f"{'case':18} {'n':>2} {'source':12} {'cover':>11} {'dist':>5} {'spr':>11}  " +
          "  ".join(f"{k:>18}" for k in keys) + "  unseen(lib)")
    for case, rs in by.items():
        cols = []
        for k in keys:
            v = np.array([r[k] for r in rs if r.get(k) is not None])
            cols.append(f"{v.mean():.3f}/{np.median(v):.3f}/{v.max():.2f}" if len(v) else "-")
        r0 = rs[0]
        print(f"{case:18} {len(rs):2d} {r0['source']:12} {r0['cover'][0]:.2f}/{r0['cover'][1]:.2f} {r0['dist']:5.2f} "
              f"{r0['spr']:5.1f}/{r0['spr_src']:4.1f}  " + "  ".join(f"{c:>18}" for c in cols) +
              f"  {np.mean([r['lib_unseen'] for r in rs]):.3f}")
    # first decisions (the root and in position after a check) apart
    print("first decisions (root, after a check), mean:")
    for case, rs in by.items():
        out = []
        for k in keys:
            v = [nv["loss"] for r in rs for p, nv in (r.get(k + "_nodes") or {}).items()
                 if p in ("", "check") and "loss" in nv]
            out.append(f"{k} {np.mean(v):.3f}" if v else f"{k} -")
        print(f"  {case:18} " + "; ".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--cases", default="6max_btn_bb,8max_btn_bb")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--config", default="accurate")
    ap.add_argument("--fast", action="store_true", help="also the case's live Fast solve at 50 iterations")
    ap.add_argument("--cross", default="", help="A:B[,C:D]: case A's references priced as answers for case B")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--gates", default="", help="COVER,DIST[,CLOSE_COVER,CLOSE_DIST]: flop_approx's NEAR_POT_COVER and "
                    "NEAR_POT_DIST (and NEAR_POT_CLOSE_COVER, NEAR_POT_CLOSE_DIST) for the run")
    a = ap.parse_args(argv)
    if a.gates:
        g = [float(x) for x in a.gates.split(",")]
        fa.NEAR_POT_COVER, fa.NEAR_POT_DIST = g[:2]
        if len(g) > 2:
            fa.NEAR_POT_CLOSE_COVER, fa.NEAR_POT_CLOSE_DIST = g[2:4]
    if a.cross:
        return cross(a)
    if a.report:
        return report(a.out)
    run(a)
    report(a.out)


if __name__ == "__main__":
    main()
