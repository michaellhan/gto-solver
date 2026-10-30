"""What the approximate answer for a flop the library doesn't store (gto/flop_approx.py) gives up against an opponent
free to adapt: the flop solved (numpy, GTO_CORE=0, so nodes lock; --config: the browser's Fast or Accurate tree, or
the library's own, which takes 1.5-1.9 GB), then per player q solved again with q's flop decisions at the nodes an
entry stores locked to the approximation (read off the spot's other stored flops; everything else free, so the
opponent and q's later streets adapt): q's EV loss, % of the pot entering the flop. The lock "ref" locks the entry's
own stored strategies instead: the measure's floor (the stored mix in 1% steps, solved in the library's game, played
in this one). One JSON line a (flop, lock) to OUT; resumable.

    python scripts/bench_flop_approx_lock.py OUT.jsonl --spots srp_hj_co,3bp_bb_btn --n 3 --iters 300 --config fast
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_menu_loss as M          # (sets GTO_CORE=0; Decision.strategy reads M.LOCK)
import numpy as np
from gto import flop_library as fl
from gto import flop_approx as fa
from gto import solver as S
from gto.cards import cards as parse_cards
import bench_flop_approx as BA
import bench_flop_seed as SEED


def locks(root, act, nodes, q):
    """{id(node): (1, A, n)} for q's nodes: nodes {path: (names, (A, 1326) mix, NaN where none)}."""
    out = {}
    for path, (names, st) in nodes.items():
        try:
            n = fl.node_at(root, path)
        except (ValueError, IndexError, AttributeError):
            continue
        if not isinstance(n, S.Decision) or n.player != q or list(n.names) != list(names):
            continue
        L = st[:, act].astype(np.float64)
        bad = np.isnan(L[0])
        L[:, bad] = 1.0 / len(names)
        L /= np.maximum(L.sum(axis=0, keepdims=True), 1e-12)
        out[id(n)] = L[None].astype(np.float32)
    return out


def solve(board, s, cfg, iters, lk=None):
    M.wait_memory()
    M.LOCK.clear()
    sol = S.solve(board, s["full0"], s["full1"], s["pot"], s["stack"], cfg=cfg, time_budget=1e9, max_iters=1)
    for n in M.decisions(sol.solver.root, []):
        n.regret[...] = 0
        n.ssum[...] = 0
        if n.pred is not None:
            n.pred[...] = 0
    k = 0
    if lk is not None:
        L = lk(sol.solver.root, sol.ctx.active)
        M.LOCK.update(L)
        k = len(L)
    sol.solver.iters = 0
    sol.extend(1e9, max_iters=iters)
    M.check_rss()
    return sol, k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--spots", default="srp_hj_co")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--locks", default="approx,ref")
    ap.add_argument("--max-gb", type=float, default=1.4)
    ap.add_argument("--config", default="fast", help="the game solved: fast or accurate (the browser's trees: "
                    "bench_flop_seed.config) or library (the entry's own; ~1.5-1.9 GB in numpy)")
    a = ap.parse_args()
    M.MAX_GB = a.max_gb
    lib = fl.FlopLibrary.from_dir()
    spots = {x["id"]: x for x in lib.index()["spots"]}
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            r = json.loads(x)
            done.add((r["spot"], r["flop"], r["lock"], r["iters"], r.get("config", "library")))
    for sid in a.spots.split(","):
        spot = spots[sid]
        s = BA.setup(sid)
        names = sorted(spot["flops"])
        random.Random(a.seed).shuffle(names)
        for name in names[:a.n]:
            todo = [x for x in a.locks.split(",") if (sid, name, x, a.iters, a.config) not in done]
            if not todo:
                continue
            board = parse_cards(name)
            if a.config == "library":
                tree = lib.entry_tree(spot, name)
                cfg = fl.library_config(spot["pot"], spot["stack"]) if (tree.get("model") or {}).get("river_dealt") \
                    else fl.library_config()
            else:
                cfg = SEED.config(a.config)
            e = lib.entry(spot, name)
            others = [n for n in names if n != name]
            src = [(lib.entry(spot, nm), m, 1.0 / (1.0 + d)) for nm, m, d in fa.neighbours(others, board, fa.K)
                   if d <= fa.MAX_DIST]
            pred = fa.predict(src)
            stored_paths = {tuple(nd["path"]) for nd in e["nodes"]}
            approx = {p: (o["names"], o["strat"]) for p, o in pred.items() if p in stored_paths}
            ref = {}
            for nd in e["nodes"]:
                st = np.full((len(nd["names"]), fa.N_COMBOS), np.nan)
                st[:, np.nonzero(e["masks"][nd["player"]])[0]] = nd["strat"]
                ref[tuple(nd["path"])] = (nd["names"], st)
            t0 = time.time()
            base, _ = solve(board, s, cfg, a.iters)
            eb = [M.ev_pct(base, p) for p in (0, 1)]
            rec0 = dict(spot=sid, flop=name, config=a.config, iters=a.iters, expl=round(base.exploitability(), 4),
                        ev=[round(x, 4) for x in eb], secs=round(time.time() - t0))
            del base
            for lock in todo:
                nodes = approx if lock == "approx" else ref
                rec = dict(rec0, lock=lock, loss=[], nodes=[])
                for q in (0, 1):
                    sl, k = solve(board, s, cfg, a.iters, lambda root, act: locks(root, act, nodes, q))
                    rec["loss"].append(round(eb[q] - M.ev_pct(sl, q), 4))
                    rec["nodes"].append(k)
                    del sl
                rec["maxrss_gb"] = round(M.rss_gb(), 2)
                print(json.dumps(rec), flush=True)
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
