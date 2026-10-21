"""How the flop model the preflop realization is measured with (pf_equilibrium.post_config: the fast speed's,
turn and river by hand buckets, the river played without seeing its card) compares with richer ones, per
hand class, on a few flops of one of pf_equilibrium.SPOTS (the charts' ranges, every class of the pre-action
range at weight EPS as in the measurement).

Models: fast (the measurement's), combo (turn per combo, river still blind), d3 (depth 3: turn per combo and
the river dealt, 8 rivers a turn, bet and played; ~100x fast's cost, 15-60 minutes a flop).

  python scripts/pf_fidelity.py --spot UTG-BB --model d3 --flops 8 --procs 3     # solve (resumable)
  python scripts/pf_fidelity.py --spot UTG-BB --compare fast d3                  # per class realization
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import gto.native  # noqa: F401
import numpy as np
import pf_equilibrium as P

DEFAULT_DIR = os.path.join(ROOT, ".pf_cache", "fidelity")
MODEL = os.environ.get("PF_FIDELITY_MODEL", "fast")
HANDS = ["AA", "KK", "TT", "77", "55", "22", "AKs", "AJo", "A9o", "A5s", "A2o", "KQo", "KTo", "K7o", "QJo",
         "KTs", "QJs", "T9s", "87s", "65s", "54s", "J8s", "T8o", "75o"]


def config(model):
    from gto.solver import Config
    from gto.native import SPEEDS
    if model == "fast":
        return P.post_config()
    sizes = SPEEDS["fast"][0]
    kw = dict(bet_sizes={k: list(v) for k, v in sizes["bet_sizes"].items()},
              raise_sizes={k: list(v) for k, v in sizes["raise_sizes"].items()})
    if model == "combo":
        return Config(**kw)
    if model == "d3":
        return Config(depth=3, **kw)
    raise ValueError(model)


# the worker processes re-import this module: the model to measure with travels in the environment
P.post_config = (lambda c: (lambda: c))(config(MODEL))


def flops(n):
    import build_realization as B
    return B.flop_sample(68)[:: max(1, 68 // n)][:n]


def run(args):
    from multiprocessing import Pool
    g = P.game()
    line = {n: l for n, l, k in P.SPOTS}[args.spot]
    oop, ip, a, rng, pre, pot, stack = P.chart_ranges(g, line)
    r_oop, r_ip = rng[oop] + P.EPS * pre[oop], rng[ip] + P.EPS * pre[ip]
    os.makedirs(args.dir, exist_ok=True)
    jobs = [(os.path.join(args.dir, f"{args.spot}_{MODEL}_{args.budget}_{k}.npz"), f, r_oop, r_ip, pot, stack,
             args.budget) for k, (f, *_) in enumerate(flops(args.flops))]
    t0 = time.time()
    with Pool(args.procs) as pool:
        for i, _ in enumerate(pool.imap_unordered(P.measure_job, jobs)):
            print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)


def load(d, spot, model, budget):
    fs = {os.path.basename(f).rsplit("_", 1)[1]: f for f in glob.glob(os.path.join(d, f"{spot}_{model}_{budget}_*.npz"))}
    return fs


def compare(args):
    """Per class realization (pot share over equity against the whole range, flops pooled) per model, and the
    range realization on the flop ranges."""
    from gto.pf_classes import CLASSES, N_IN, to_classes
    models = args.compare
    sets = [load(args.dir, args.spot, m, args.budget) for m in models]
    common = sorted(set.intersection(*[set(s) for s in sets]))
    print(f"{args.spot}: {len(common)} flops in common")
    acc = [{side: sum(np.load(s[k])[side] for k in common) for side in ("oop", "ip")} for s in sets]
    g = P.game()
    line = {n: l for n, l, k in P.SPOTS}[args.spot]
    oop, ip, a, rng, *_ = P.chart_ranges(g, line)
    for side, q in (("oop", oop), ("ip", ip)):
        w = to_classes(rng[q]) / N_IN
        rr = [(w * x[side][0].sum(1)).sum() / (w * x[side][1].sum(1)).sum() for x in acc]
        print(f"{side} ({'aggressor' if q == a else 'caller'}): range " + "  ".join(f"{m} {r:.3f}" for m, r in zip(models, rr)))
        for h in HANDS:
            i = CLASSES.index(h)
            cells = [x[side][0, i].sum() / max(x[side][1, i].sum(), 1e-12) for x in acc]
            print(f"  {h:4s}" + "".join(f"  {c:6.3f}" for c in cells) + f"   {cells[-1] - cells[0]:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", default="UTG-BB")
    ap.add_argument("--model", default="fast", choices=("fast", "combo", "d3"))
    ap.add_argument("--flops", type=int, default=8)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()
    if args.compare:
        compare(args)
        return
    if args.model != MODEL:          # re-exec with the model in the environment (the workers read it)
        os.environ["PF_FIDELITY_MODEL"] = args.model
        os.execv(sys.executable, [sys.executable] + sys.argv)
    run(args)


if __name__ == "__main__":
    main()
