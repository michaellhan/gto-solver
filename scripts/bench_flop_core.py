"""Live flop solves of flops the precomputed library doesn't store, numpy against the compiled core (README, Compiled
core): Fast's flop model (gto/native.py SPEEDS["fast"], the browser's too), on flops of a spot the library doesn't
store every flop of (seeded draw: rainbow, two-tone, monotone and paired), each flop run in numpy and then in the
core, back to back.

    python scripts/bench_flop_core.py OUT.jsonl [--budget 8] [--checkpoints 50,100,200,400] [--noblas]
    node scripts/pyodide_run.mjs scripts/bench_flop_core.py OUT.jsonl ...      (the browser's numpy and core)
    python scripts/bench_flop_core.py OUT.jsonl --report

A budget run solves as the Advisor does (solve(reserve=advisor.HU_RESERVE): the budget covers building the tree and
keeps one iteration's time back) and records the iterations, the seconds and the exploitability (% of the pot);
--checkpoints records the exploitability at those iterations instead, and the root's and the c-bet's frequencies at the
last (the same answers at the same iterations). --noblas runs the browser's numpy kernels natively (solver.NO_BLAS).
Each line of OUT is one (flop, mode, run); a run skips lines already there.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gto.native as gn  # noqa: E402  (holds BLAS to one thread before numpy loads)
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import solver as S  # noqa: E402
from gto.advisor import HU_RESERVE  # noqa: E402
from gto.cards import cards as parse_cards, cards_str  # noqa: E402

LINES = {s[0]: s[2] for s in fl.SPOTS}


def pick_flops(sid: str, want: dict, seed: int = 5) -> list:
    stored = set(next(s for s in fl.FlopLibrary.from_dir(fl.DIR).index()["spots"] if s["id"] == sid)["flops"])
    cand = [cards_str(f) for f in fl.all_flops() if cards_str(f) not in stored]
    random.Random(seed).shuffle(cand)
    want, out = dict(want), []
    for f in cand:
        b = parse_cards(f)
        k = "paired" if len({c // 4 for c in b}) < 3 else \
            {1: "rainbow", 2: "twotone", 3: "mono"}[max(sum(1 for c in b if c % 4 == s) for s in range(4))]
        if want.get(k, 0) > 0:
            out.append(f)
            want[k] -= 1
    return out


def run(sid: str, flop: str, mode: str, budget: float, cks) -> dict:
    s = fl.spot_setup(LINES[sid])
    cfg = gn.config("fast")
    t0 = time.time()
    if cks:
        sol = S.solve(parse_cards(flop), s["full0"], s["full1"], s["pot"], s["stack"], cfg=cfg, time_budget=1e9,
                      max_iters=0)
        sol.solver.use_core = mode == "core"
        pts, secs = [], 0.0
        for ck in cks:
            t1 = time.time()
            sol.solver.run(sol.r0, sol.r1, time_budget=1e9, max_iters=ck, min_iters=1)
            secs += time.time() - t1
            sol.iters = sol.solver.iters
            pts.append((ck, round(secs, 2), round(sol.exploitability(), 4)))
        # the frequencies at the last checkpoint (% of the acting player's range) at the root and after its first
        # action (a check: the c-bet)
        freqs = []
        for node in (sol.root, sol.root.children[0]):
            r = (sol.r0, sol.r1)[node.player]
            freqs.append([round(float(x), 2) for x in 100 * (node.avg_strategy()[0] * r).sum(axis=1) / r.sum()])
        return dict(sid=sid, flop=flop, mode=mode, run="checkpoints", pts=pts, freqs=freqs, core=sol.solver._core_ok)
    with_core = mode == "core"
    from unittest import mock
    from gto import core
    with mock.patch.object(core, "lib", return_value=core.lib() if with_core else None):
        sol = S.solve(parse_cards(flop), s["full0"], s["full1"], s["pot"], s["stack"], cfg=cfg, time_budget=budget,
                      max_iters=100000, reserve=HU_RESERVE)
        t1 = time.time()
        ex = sol.exploitability()
    return dict(sid=sid, flop=flop, mode=mode, run=f"budget {budget:g}", iters=sol.iters, solve_s=round(sol.seconds, 2),
                total_s=round(time.time() - t0, 2), expl_s=round(time.time() - t1, 2), expl=round(ex, 4),
                core=sol.solver._core_ok and with_core)


def report(path: str) -> None:
    rows = [json.loads(x) for x in open(path)]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["run"], r["flop"])][r["mode"]] = r
    for run_ in sorted({r["run"] for r in rows}):
        print(run_)
        ratios = []
        for (rn, flop), d in sorted(by.items()):
            if rn != run_:
                continue
            if run_ == "checkpoints":
                line = "  ".join(f"{m}: " + " ".join(f"{p[0]}:{p[2]:.3f}" for p in d[m]["pts"]) +
                                 f" ({d[m]['pts'][-1][1]} s)" + (f" root, c-bet {d[m]['freqs']}" if "freqs" in d[m] else "")
                                 for m in ("numpy", "core") if m in d)
                if "numpy" in d and "core" in d:
                    ratios.append(d["numpy"]["pts"][-1][1] / max(d["core"]["pts"][-1][1], 1e-9))
            else:
                line = "  ".join(f"{m}: {d[m]['iters']:5d} it {d[m]['expl']:.3f}% ({d[m]['total_s']} s, expl "
                                 f"{d[m]['expl_s']} s)" for m in ("numpy", "core") if m in d)
                if "numpy" in d and "core" in d:
                    ratios.append(d["core"]["iters"] / max(d["numpy"]["iters"], 1))
            print(f"  {flop:8s} {line}")
        if ratios:
            what = "numpy / core seconds" if run_ == "checkpoints" else "core / numpy iterations"
            print(f"  {what}: {min(ratios):.1f}-{max(ratios):.1f}x, "
                  f"geometric mean {float(np.exp(np.mean(np.log(ratios)))):.1f}x")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--spot", default="srp_hj_bb")
    ap.add_argument("--mix", default="rainbow:3,twotone:3,mono:1,paired:1")
    ap.add_argument("--flops", default=None, help="comma-separated flops instead of the seeded draw")
    ap.add_argument("--modes", default="numpy,core")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--checkpoints", default=None)
    ap.add_argument("--noblas", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.out)
    S.NO_BLAS = S.NO_BLAS or a.noblas
    cks = [int(x) for x in a.checkpoints.split(",")] if a.checkpoints else None
    flops = a.flops.split(",") if a.flops else \
        pick_flops(a.spot, {k: int(v) for k, v in (x.split(":") for x in a.mix.split(","))})
    run_ = "checkpoints" if cks else f"budget {a.budget:g}"
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            d = json.loads(x)
            done.add((d["sid"], d["flop"], d["mode"], d["run"]))
    for f in flops:
        for m in a.modes.split(","):
            if (a.spot, f, m, run_) in done:
                continue
            r = run(a.spot, f, m, a.budget, cks)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
    report(a.out)


if __name__ == "__main__":
    main()
