"""What more iterations buy a threaded Fast turn (Accurate's tree: turn and river 33/75/125%, river classes): the
check behind worker.js's hu_settled_from (turn and shallow 600 with threads).

Each turn spot of scripts/turn_river_reference.py solved once, read at checkpoints, against the same tree at FINAL
iterations (its own convergence) and against the stored rich every-river reference (menu + classes + convergence).
Per-combo EVs (% of the pot, range-weighted, both players) and the root aggregates (points). The stored references
(400-500 iterations, 0.03-0.13% of the pot from equilibrium) carry the same unconverged error as a solve near their
iterations, so past ~400 the distance from the same tree's FINAL is the measure of convergence.

    heavy python scripts/turn_convergence.py [spots]     # ~15 min on one core; CHECKS=..., FINAL=...
"""
import os, sys, json, time
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import turn_river_reference as T
from gto import solver as S
from gto import core

CHECKS = [int(x) for x in os.environ.get("CHECKS", "200,300,400,600,800,1000").split(",")]
FINAL = int(os.environ.get("FINAL", "2000"))
KW = dict(T.RICH, river_classes=True, force_allin=0.15)
OUT = os.environ.get("OUT", os.path.join(ROOT, ".work", "turn_convergence.json"))   # (resumes: spots in it are skipped)


def snap(sol):
    if not sol.fresh():
        sol._refresh()
    v = T.root_values(sol)
    return dict(iters=sol.solver.iters, expl=sol.exploitability(), vals=v, stats=T.stats(sol))


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else [n for n, s in T.SPOTS.items() if s[5] is None]
    refs = T.load_refs()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print("core:", bool(core.lib()), "threads", core.threads(), flush=True)
    for name in names:
        if name in res:
            continue
        t0 = time.time()
        board, r0, r1, pot, stack = T.load_spot(name)
        sol = S.solve(board, r0, r1, pot, stack, cfg=S.Config(**KW), time_budget=1e9, max_iters=CHECKS[0],
                      horizon=1000)   # (as Advisor runs it in pieces)
        snaps = [snap(sol)]
        for c in CHECKS[1:] + [FINAL]:
            sol.solver.run(sol.r0, sol.r1, time_budget=1e9, max_iters=c, min_iters=1)
            sol.iters = sol.solver.iters
            snaps.append(snap(sol))
        fin = snaps[-1]
        ref = refs[name]["rich"]
        rv = [np.asarray(ref["values"][p], dtype=np.float64) for p in (0, 1)]
        rows = []
        for s in snaps:
            mine = [S.expand(sol.ctx, s["vals"][p]) for p in (0, 1)]
            finv = [S.expand(sol.ctx, fin["vals"][p]) for p in (0, 1)]
            rf = [S.expand(sol.ctx, r) for r in (sol.r0, sol.r1)]
            d_fin = float(np.mean([(np.abs(mine[p] - finv[p]) * rf[p]).sum() / rf[p].sum() for p in (0, 1)]))
            d_ref = float(np.mean([(np.abs(mine[p] - rv[p]) * rf[p]).sum() / rf[p].sum() for p in (0, 1)]))
            keys = T.compared(fin["stats"])
            f_fin = float(np.mean([abs(s["stats"].get(k, 0) - fin["stats"].get(k, 0)) for k in keys]))
            f_ref = T.freq_error(s["stats"], ref["stats"])[0]
            rows.append(dict(iters=s["iters"], expl=s["expl"], ev_final=d_fin, ev_ref=d_ref, freq_final=f_fin,
                             freq_ref=f_ref))
            print(f"{name:12s} {s['iters']:5d} expl {s['expl']:.3f}%  EV off final {d_fin:.3f} ref {d_ref:.3f}  "
                  f"freq off final {f_fin:.2f} ref {f_ref:.2f}", flush=True)
        res[name] = dict(rows=rows, seconds=time.time() - t0)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"{name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
