"""Approximate answers for flops the library doesn't store (gto/flop_approx.py), measured against stored solves.

For a library spot's flop that isn't stored, its decisions are read off the K stored flops most like it (card
substitution, averaged). Here each method predicts flops that *are* stored, from the others, and each combo's
predicted mix is priced by the true entry's EVs per action (the approximate answer's measure in README, Accuracy vs
speed: B3): EV given up = the true mix's EV - the predicted mix's, per combo, weighted by the acting player's reach at
the node, % of the pot entering the flop. Per node, and per flop over its stored decisions weighted by how often each
is reached (the range frequencies along its line).

Protocols: rep (the spots on every flop: neighbours only from the 552 representative flops, the set the flatted opens
and 3bet pots store, tested on the other flops: the case the advisor meets); loo (any spot: a stored flop from the
spot's other stored flops); spot (a whole spot held out: its flops read off the spot flop_approx.near_spot picks with
it left out of the index, or the one named as TARGET<SOURCE; methods near, the source's entry of the same flop where
it stores it, else its K most like it, as the advisor reads it, and nearx, its K most like it other than the flop:
one it doesn't store).

    python scripts/bench_flop_approx.py OUT.jsonl --protocol rep --spots srp_btn_bb,srp_co_bb --n 150 --methods knn1,knn3
    python scripts/bench_flop_approx.py OUT.jsonl --protocol loo --spots 3bp_bb_btn --n 150
    python scripts/bench_flop_approx.py OUT.jsonl --protocol spot --spots srp_co_bb,srp_hj_bb --n 60 --methods near,nearx
    python scripts/bench_flop_approx.py OUT.jsonl --report
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import flop_approx as fa  # noqa: E402
from gto.cards import cards as parse_cards, cards_str  # noqa: E402
from gto.solver import MEMBER_FULL  # noqa: E402

LINES = {s[0]: s[2] for s in fl.SPOTS}
_SETUP = {}


def setup(sid):
    if sid not in _SETUP:
        _SETUP[sid] = fl.spot_setup(LINES[sid])
    return _SETUP[sid]


def node_reaches(e, full0, full1):
    """{path: (acting player's reach (1326) at the node, the node's share of the hands: range frequencies along the
    line)} from an entry's own strategies."""
    board = parse_cards(e["head"]["board"])
    bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
    full = [full0 * bm, full1 * bm]
    nodes = {tuple(n["path"]): n for n in e["nodes"]}
    out = {}
    for path, nd in nodes.items():
        p = nd["player"]
        r = full[p].copy()
        share = 1.0
        for i in range(len(path)):
            anc = nodes.get(path[:i])
            if anc is None:
                share = None
                break
            k = anc["names"].index(path[i])
            share *= anc["freqs"][path[i]] / max(sum(anc["freqs"].values()), 1e-12)
            if anc["player"] == p:
                m = anc_mask = e["masks"][p]
                rr = np.zeros(len(r))
                rr[np.nonzero(anc_mask)[0]] = anc["strat"][k]
                r = r * rr
        out[path] = (r, share)
    return out


def given_up(nd, mask, reach, pred, pot):
    """(EV given up, % of the pot, reach-weighted; the same measured to the best action) of the predicted mix (A, 1326;
    NaN rows where none) at a true node."""
    idx = np.nonzero(mask)[0]
    S, V = nd["strat"], nd["evs"]
    P = pred[:, idx]
    w = reach[idx]
    bad = np.isnan(P[0])
    if bad.any():   # (no image anywhere: the predicted node's range average)
        fill = np.nansum(P * w[None, :], axis=1) / max(float(w[~bad].sum()), 1e-12)
        if not np.isfinite(fill).all() or fill.sum() <= 0:
            fill = np.full(len(P), 1.0 / len(P))
        P = np.where(bad[None, :], (fill / fill.sum())[:, None], P)
    P = P / np.maximum(P.sum(axis=0, keepdims=True), 1e-12)
    true_ev = (S * V).sum(axis=0)
    pred_ev = (P * V).sum(axis=0)
    W = max(float(w.sum()), 1e-12)
    loss = float((w * (true_ev - pred_ev)).sum() / W / pot * 100)
    regret = float((w * (V.max(axis=0) - pred_ev)).sum() / W / pot * 100)
    return loss, regret, float(w[bad].sum() / W)


def method_pred(method, lib, spot, names, board, target_name):
    """{path: predicted node} for a flop, by method: knnK (K nearest stored flops, weights 1 / (1 + distance)),
    nnK (K nearest, equal weights), avgK (every combo the K nearest flops' range frequencies); a trailing c (knn3c):
    each combo averages the images of its own hand class (flop_approx.predict's board)."""
    cls = method.endswith("c")
    method = method[:-1] if cls else method
    kind = method.rstrip("0123456789")
    k = int(method[len(kind):] or 1)
    nb = [x for x in fa.neighbours(names, board, k + 1) if x[0] != target_name][:k]
    srcs = []
    for nm, m, d in nb:
        e = lib.entry(spot, nm)
        if e is None:
            continue
        w = 1.0 / (1.0 + d) if kind == "knn" else 1.0
        srcs.append((e, m, w))
    pred = fa.predict(srcs, board=list(board) if cls else None)
    if kind == "avg":
        for path, o in pred.items():
            # each source's node frequency, the same for every combo
            fr = np.zeros(len(o["names"]))
            tw = 0.0
            for e, m, w in srcs:
                nd = next((x for x in e["nodes"] if tuple(x["path"]) == path), None)
                if nd is None:
                    continue
                f = np.array([nd["freqs"][n] for n in o["names"]])
                fr += w * f / max(f.sum(), 1e-12)
                tw += w
            o["strat"] = np.repeat((fr / max(tw, 1e-12))[:, None], fa.N_COMBOS, axis=1)
    return pred, [(nm, d) for nm, m, d in nb]


def run(a):
    lib = fl.FlopLibrary.from_dir(a.library)
    spots = {s["id"]: s for s in lib.index()["spots"]}
    rep = set(cards_str(f) for f in fl.representative_flops(552))
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            d = json.loads(x)
            done.add((d["spot"], d["flop"], d["method"], d["protocol"]))
    methods = a.methods.split(",")
    for item in a.spots.split(","):
        sid, src_id = item.split("<") if "<" in item else (item, None)
        spot = spots[sid]
        stored = sorted(spot["flops"])
        source = None
        if a.protocol == "spot":
            if src_id is None:
                s0 = setup(sid)
                got = fa.near_spot(lib, s0["full0"], s0["full1"], float(spot["pot"]), float(spot["stack"]), exclude=sid)
                if got is None:
                    print(f"{sid}: no spot near enough with it left out")
                    continue
                src_id = got["id"]
            source = spots[src_id]
            train, test = sorted(source["flops"]), list(stored)
        elif a.protocol == "rep":
            train = [n for n in stored if n in rep]
            test = [n for n in stored if n not in rep]
        else:
            train = stored
            test = list(stored)
        random.Random(a.seed).shuffle(test)
        test = test[:a.n]
        s = setup(sid)
        pot = float(spot["pot"])
        t0 = time.time()
        for tn in test:
            todo = [m for m in methods if (sid, tn, m, a.protocol) not in done]
            if not todo:
                continue
            e = lib.entry(spot, tn)
            if e is None:
                continue
            board = parse_cards(tn)
            reaches = node_reaches(e, s["full0"], s["full1"])
            for meth in todo:
                if source is not None:   # (protocol spot)
                    canon = list(fl.canonical(board)[0])
                    src = fa.sources(lib, source, canon, same=True) if meth.startswith("near") and \
                        not meth.startswith("nearx") else \
                        [(lib.entry(source, nm), m, 1.0 / (1.0 + d), nm, d)
                         for nm, m, d in fa.neighbours([n for n in train if n != tn], canon, fa.K) if d <= fa.MAX_DIST]
                    pred = fa.predict([x[:3] for x in src], board=canon if meth.endswith("c") else None)
                    nb = [(x[3], x[4]) for x in src]
                else:
                    pred, nb = method_pred(meth, lib, spot, train, board, tn)
                rec = {"spot": sid, "flop": tn, "method": meth, "protocol": a.protocol, "nb": nb, "nodes": {}}
                if source is not None:
                    rec["source"] = source["id"]
                for nd in e["nodes"]:
                    path = tuple(nd["path"])
                    r, share = reaches[path]
                    pr = pred.get(path)
                    if pr is None:
                        rec["nodes"]["|".join(path)] = {"share": share, "missing": True}
                        continue
                    loss, regret, unseen = given_up(nd, e["masks"][nd["player"]], r, pr["strat"], pot)
                    rec["nodes"]["|".join(path)] = {"share": share, "loss": round(loss, 4), "regret": round(regret, 4),
                                                    "unseen": round(unseen, 4)}
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        print(f"{sid}: {len(test)} flops, {time.time() - t0:.0f} s", flush=True)


def report(path, by_node=True):
    rows = [json.loads(x) for x in open(path)]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["protocol"], r["spot"] + ("<" + r["source"] if r.get("source") else ""), r["method"])].append(r)
    print("EV given up (% of the pot entering the flop): per flop, its decisions weighted by how often each comes; "
          "mean / median / 90th percentile / worst; then per node, mean (90th)")
    for (proto, sid, meth), rs in sorted(groups.items()):
        agg, per = [], collections.defaultdict(list)
        miss = 0
        for r in rs:
            num = den = 0.0
            for k, v in r["nodes"].items():
                if v.get("missing"):
                    miss += 1
                    continue
                if v["share"] is None:
                    continue
                num += v["share"] * v["loss"]
                den += v["share"]
                per[k].append(v["loss"])
            if den > 0:
                agg.append(num / den)
        agg = np.array(agg)
        line = (f"{proto:4s} {sid:12s} {meth:6s} n={len(agg):4d}  {agg.mean():6.3f} {np.median(agg):6.3f} "
                f"{np.percentile(agg, 90):6.3f} {agg.max():6.3f}")
        if by_node:
            line += "   " + "  ".join(f"{k or 'root'}: {np.mean(v):.3f} ({np.percentile(v, 90):.3f})"
                                     for k, v in sorted(per.items(), key=lambda x: (len(x[0]), x[0]))
                                     if len(v) >= 0.2 * len(rs))
        if miss:
            line += f"  [{miss} nodes unpredicted]"
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--protocol", default="rep", choices=("rep", "loo", "spot"))
    ap.add_argument("--spots", default="srp_btn_bb")
    ap.add_argument("--methods", default="avg3,knn1,knn2,knn3,knn4")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--library", default=fl.DIR)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.out)
    run(a)
    report(a.out)


if __name__ == "__main__":
    main()
