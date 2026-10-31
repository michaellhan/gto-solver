"""A live flop solve of a library spot's flop that isn't stored, against the approximate answer read off the spot's
stored flops most like it (gto/flop_approx.py), and started from it (a seed).

Each flop here *is* stored (its entry the reference), and is solved live as the advisor solves a flop it doesn't
store (the browser's Fast or Accurate tree, natively, one thread), the approximation read off the spot's other
stored flops. At iteration checkpoints: the solve's exploitability (% of the pot) and the EV its average mix gives up
at the entry's stored nodes, priced by the entry's EVs per action (bench_flop_approx.py's measure, % of the pot,
per flop over its decisions weighted by how often each comes), with the approximation's own for comparison.

Modes: cold; regret:S (each predicted node's regrets S x the pot x the opponent's compatible reach x the predicted
mix, as Advisor._store_seed does a stored turn, so the first iteration plays it); avg:W (the strategy sums W x the
player's reach x the predicted mix: only the average starts there, the iterates as cold; DCFR discounts it away);
both:S:W.

    python scripts/bench_flop_seed.py OUT.jsonl --spots srp_hj_co,3bp_bb_btn --n 6 --config fast --modes cold,regret:1
    python scripts/bench_flop_seed.py OUT.jsonl --spots srp_hj_co,3bp_bb_btn --n 8 --modes cold --seed 11 --skip 80 \
        --checkpoints 20,30,40,50,60,70,100
    python scripts/bench_flop_seed.py OUT.jsonl --report
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

RAISES = {"flop": [0.5], "turn": [0.75], "river": [0.75]}


def config(name: str) -> S.Config:
    """The browser's trees (gto/static/worker.js; Fast's one-thread river sizes)."""
    if name == "fast":
        return S.Config(n_turns=3, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4, river_buckets=32,
                        turn_buckets=True, bucket_river_step=3, river_classes=True, river_strength=32,
                        bet_sizes={"flop": [0.33, 0.75], "turn": [0.33, 0.75, 1.25], "river": [0.33, 1.25]},
                        raise_sizes={k: list(v) for k, v in RAISES.items()})
    if name == "accurate":
        return S.Config(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8, river_classes=True,
                        river_buckets=64, turn_buckets=True, bucket_river_step=1, river_strength=32,
                        bet_sizes={"flop": [0.33, 0.75], "turn": [0.33, 0.75, 1.25], "river": [0.33, 0.75, 1.25]},
                        raise_sizes={k: list(v) for k, v in RAISES.items()})
    return gto.native.config(name)


def seeder(pred, mode: str, pot: float):
    """solve()'s seed: the predicted mixes (flop_approx.predict, on the canonical flop) set as the mode says."""
    parts = mode.split(":")
    kind = parts[0]
    rs = float(parts[1]) if kind in ("regret", "both") else 0.0
    ws = float(parts[-1]) if kind in ("avg", "both") else 0.0

    def seed(root, s):
        act = s.ctx.active
        r0, r1 = s.ctx.root_reach
        reach = {}
        for path in sorted(pred, key=len):
            o = pred[path]
            try:
                n = fl.node_at(root, path)
            except (ValueError, IndexError, AttributeError):
                continue
            if not isinstance(n, S.Decision) or list(n.names) != o["names"] or n.regret.shape[0] != 1:
                continue
            mix = o["strat"][:, act]
            ok = ~np.isnan(mix[0])
            mix = np.nan_to_num(mix)
            # the reaches at the node: each player's own predicted mixes along the line
            a0, a1 = r0.astype(np.float64), r1.astype(np.float64)
            for i in range(len(path)):
                anc = pred.get(path[:i])
                if anc is None:
                    continue
                am = np.nan_to_num(anc["strat"][anc["names"].index(path[i])][act], nan=1.0 / len(anc["names"]))
                if anc["player"] == 0:
                    a0 = a0 * am
                else:
                    a1 = a1 * am
            own, opp = (a0, a1) if n.player == 0 else (a1, a0)
            if rs:
                mass = s.compat_sum(np.asarray(opp, dtype=np.float32)[None])[0]
                n.regret[0][:, ok] = (rs * pot * mass[ok] * mix[:, ok]).astype(np.float32)
            if ws:
                n.ssum[0][:, ok] = (ws * own[ok] * mix[:, ok]).astype(np.float32)
    return seed


def measure(sol, e, reaches, pot, only=None):
    """The solve's average mix at the entry's stored nodes, priced by the entry's EVs: (per-flop given-up, {path: loss}).
    only: {path: predicted node}, priced over the combos the prediction answers there (bench_flop_approx's seen)."""
    act = sol.ctx.active
    num = den = 0.0
    per = {}
    for nd in e["nodes"]:
        path = tuple(nd["path"])
        try:
            n = fl.node_at(sol.root, path)
        except (ValueError, IndexError, AttributeError):
            continue
        if not isinstance(n, S.Decision) or list(n.names) != list(nd["names"]):
            continue
        st = np.full((len(n.names), fa.N_COMBOS), np.nan)
        st[:, act] = n.avg_strategy()[0]
        r, share = reaches[path]
        if only is not None:
            if path not in only:
                continue
            st[:, np.isnan(only[path]["strat"][0])] = np.nan
        loss, _, _ = BA.given_up(nd, e["masks"][nd["player"]], r, st, pot, seen=only is not None)
        per["|".join(path)] = round(loss, 4)
        if share:
            num += share * loss
            den += share
    return (num / den if den else None), per


def approx_loss(pred, e, reaches, pot, seen=False):
    num = den = 0.0
    for nd in e["nodes"]:
        path = tuple(nd["path"])
        pr = pred.get(path)
        r, share = reaches[path]
        if pr is None or not share or pr["names"] != list(nd["names"]):
            continue
        loss, _, _ = BA.given_up(nd, e["masks"][nd["player"]], r, pr["strat"], pot, seen=seen)
        num += share * loss
        den += share
    return num / den if den else None


def truth_entry(spot, s, board, cfg_name, iters):
    """The flop solved live (cfg_name's tree, iters iterations), as a decoded library entry: a reference of its own."""
    cfg = config(cfg_name)
    sol = S.solve(board, s["full0"], s["full1"], float(spot["pot"]), float(spot["stack"]), cfg=cfg, time_budget=1e9,
                  max_iters=iters)
    meta = {"spot": spot["id"], "fp": [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])], "pot": float(spot["pot"]),
            "stack": float(spot["stack"]), "tree": fl.tree_key(cfg), "board": fl.cards_str(board)}
    return fl.decode(fl.encode(sol, meta))


def run_one(lib, spot, s, name, cfg_name, mode, cks, truth=None, source=None):
    pot, stack = float(spot["pot"]), float(spot["stack"])
    board = parse_cards(name)
    e = lib.entry(spot, name) if not truth else truth_entry(spot, s, board, truth.split(":")[0],
                                                             int(truth.split(":")[1]))
    reaches = BA.node_reaches(e, s["full0"], s["full1"])
    others = [n for n in sorted(spot["flops"]) if n != name]
    src = []
    if source is not None:   # (another spot's, as the advisor reads it: its entry of the flop, else those most like it)
        src = [x[:3] for x in fa.sources(lib, source, list(board), same=True)]
    for nm, m, d in fa.neighbours(others, board, fa.K) if source is None else ():
        if d <= fa.MAX_DIST:
            src.append((lib.entry(spot, nm), m, 1.0 / (1.0 + d)))
    pred = fa.predict(src)
    # the approximation as the advisor answers now: each combo off its own hand class's images, facing a bet corrected,
    # over the combos some source holds (the solve's checkpoints priced over the same combos too: loss_seen)
    pcl = fa.predict(src, board=list(board), correct=True) if src else {}
    pc = fa.predict(src, board=list(board)) if src else {}
    seed = None if mode == "cold" else seeder(pred, mode, pot)
    t0 = time.time()
    sol = S.solve(board, s["full0"], s["full1"], pot, stack, cfg=config(cfg_name), time_budget=1e9, max_iters=0,
                  seed=seed)
    build = time.time() - t0
    pts, secs = [], 0.0
    for ck in cks:
        t0 = time.time()
        sol.solver.run(sol.r0, sol.r1, time_budget=1e9, max_iters=ck, min_iters=1)
        secs += time.time() - t0
        sol.iters = sol.solver.iters
        g, per = measure(sol, e, reaches, pot)
        gs, _ = measure(sol, e, reaches, pot, only=pcl)
        pts.append({"it": ck, "s": round(secs, 2), "expl": round(sol.exploitability(), 4),
                    "loss": None if g is None else round(g, 4), "loss_seen": None if gs is None else round(gs, 4),
                    "nodes": per})
    return {"spot": spot["id"], "flop": name, "cfg": cfg_name, "mode": mode, "truth": truth or "library",
            "source": source["id"] if source is not None else None,
            "build_s": round(build, 2),
            "approx": round(approx_loss(pred, e, reaches, pot) or 0.0, 4),
            "approx_c": round(approx_loss(pc, e, reaches, pot, seen=True) or 0.0, 4),
            "approx_cl": round(approx_loss(pcl, e, reaches, pot, seen=True) or 0.0, 4), "pts": pts}


def report(path):
    rows = [json.loads(x) for x in open(path)]
    g = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        g[(r["cfg"], r["spot"][:3])][(r["spot"], r["flop"])][r["mode"]] = r
    for (cfg, kind), flops in sorted(g.items()):
        modes = sorted({m for d in flops.values() for m in d}, key=lambda m: (m != "cold", m))
        full = [d for d in flops.values() if all(m in d for m in modes)]
        if not full:
            continue
        cks = [p["it"] for p in full[0][modes[0]]["pts"]]
        ax = np.array([d[modes[0]]["approx"] for d in full])
        print(f"{cfg} {kind}: {len(full)} flops; the approximation gives up {ax.mean():.3f}% of the pot (mean, "
              f"median {np.median(ax):.3f})")
        print(f"  iterations      " + " ".join(f"{c:>13d}" for c in cks))
        for m in modes:
            ex = np.array([[p["expl"] for p in d[m]["pts"]] for d in full])
            ls = np.array([[p["loss"] for p in d[m]["pts"]] for d in full], dtype=float)
            sec = np.array([[p["s"] for p in d[m]["pts"]] for d in full])
            gm = np.exp(np.log(np.maximum(ex, 1e-4)).mean(axis=0))
            print(f"  {m:14s}  " + " ".join(f"{a:6.3f}/{b:6.3f}" for a, b in zip(gm, ls.mean(axis=0)))
                  + f"   ({sec[:, -1].mean():.1f} s)")
            if m != "cold":
                c = np.array([[p["expl"] for p in d["cold"]["pts"]] for d in full])
                print(f"  {'(worse than cold)':14s}  " + " ".join(f"{int(x):13d}" for x in (ex > c * 1.02).sum(axis=0)))
        print("  (exploitability geometric mean / given up at the stored nodes, mean; % of the pot)")
        if all("approx_cl" in d["cold"] for d in full if "cold" in d):
            cl = np.array([d["cold"]["approx_cl"] for d in full if "cold" in d])
            c0 = np.array([d["cold"].get("approx_c", np.nan) for d in full if "cold" in d], dtype=float)
            ls = np.array([[p.get("loss_seen") for p in d["cold"]["pts"]] for d in full if "cold" in d], dtype=float)
            print(f"  as answered now (hand classes, the correction; the combos it answers): {cl.mean():.3f} (median "
                  f"{np.median(cl):.3f}; uncorrected {np.nanmean(c0):.3f}, median {np.nanmedian(c0):.3f}; the "
                  f"correction lower by over 0.01 on {int((cl < c0 - 0.01).sum())}, higher on "
                  f"{int((cl > c0 + 0.01).sum())})")
            print("  cold over the same combos "
                  + " ".join(f"{c}: {v:.3f}" for c, v in zip(cks, np.nanmean(ls, axis=0)))
                  + "; the approximation better on " + " ".join(f"{c}: {int((cl < ls[:, j]).sum())}/{len(cl)}"
                                                                 for j, c in enumerate(cks)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--spots", default="srp_hj_co,3bp_bb_btn")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--skip", type=int, default=0, help="leave out the first SKIP flops of the shuffle (the learned "
                    "correction's training flops on a spot of 552: --seed 11 --skip 80)")
    ap.add_argument("--config", default="fast")
    ap.add_argument("--modes", default="cold,regret:1")
    ap.add_argument("--checkpoints", default="5,10,20,40,70,100,150,200,300")
    ap.add_argument("--truth", default="", help="CFG:ITERS: price against the flop solved live in that tree (e.g. "
                    "accurate:600) instead of its library entry")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.out)
    lib = fl.FlopLibrary.from_dir()
    spots = {x["id"]: x for x in lib.index()["spots"]}
    cks = [int(x) for x in a.checkpoints.split(",")]
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            d = json.loads(x)
            done.add((d["spot"], d["flop"], d["cfg"], d["mode"]))
    for item in a.spots.split(","):
        sid, src_id = item.split("<") if "<" in item else (item, None)   # (TARGET<SOURCE: read off another spot)
        spot = spots[sid]
        s = BA.setup(sid)
        names = sorted(spot["flops"])
        random.Random(a.seed).shuffle(names)
        for name in names[a.skip:a.skip + a.n]:
            for mode in a.modes.split(","):
                if (sid, name, a.config, mode) in done:
                    continue
                r = run_one(lib, spot, s, name, a.config, mode, cks, a.truth, src_id and spots[src_id])
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
                last = r["pts"][-1]
                print(sid, name, mode, f"approx {r['approx']:.3f}", "->", [(p["it"], p["expl"], p["loss"]) for p in r["pts"][::3]],
                      last["s"], flush=True)
    report(a.out)


if __name__ == "__main__":
    main()
