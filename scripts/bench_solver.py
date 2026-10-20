"""Postflop solver speed benchmarks.

Default (regression check): BTN vs BB single-raised pot on three flops with the browser's Fast config, a
fixed 150 iterations. Prints iterations/second and the exploitability reached.

    python scripts/bench_solver.py [--save FILE] [--compare FILE]

--save writes each board's root average strategy to an .npz; --compare checks a run against one
(speedups must leave the results unchanged).

Curves (--curve): exploitability (% of the pot) against solving time for a set of spots (single-raised and
3bet pots; dry, wet, monotone and paired flops, a turn, a river) with the browser's Fast and Accurate
configs and the native default. Exploitability is measured at fixed iteration counts (deterministic) and
mapped to time by the setup time and the fastest per-iteration rate seen, in CPU seconds, so other work on
the machine moves the numbers little. numpy's BLAS is held to one thread (the browser has one) unless
BENCH_THREADS is set. --noblas runs the browser's kernels (einsum, trimmed leaves); to run it in Pyodide
itself (Node, `npm install pyodide@0.26.4`), mount the repo and run this file with runpy: NO_BLAS is on
there and times are wall-clock.

    python scripts/bench_solver.py --curve [--noblas] [--spots srp-dry,turn] [--configs fast] [--budget 8]
"""
import os, sys, time, argparse, json
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("BENCH_THREADS", "1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from gto import preflop as pf
from gto import solver as S
from gto.advisor import preflop_ranges
from gto.cards import cards
from gto.solver import Config, solve

LS = dict(bet_sizes={"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]},
          raise_sizes={"flop": [0.5], "turn": [0.75], "river": [0.75]})
FAST = dict(n_turns=3, n_rivers=8, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4, **LS)
ACCURATE = dict(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8, **LS)
CONFIGS = {"fast": FAST, "accurate": ACCURATE, "native": {}}
BOARDS = ["Qs8d3c", "Kd7c2h", "9h8h6c"]
ITERS = 150

SRP = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
THREEBET = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
            {"pos": "BTN", "action": "raise", "to": 7.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "fold"},
            {"pos": "CO", "action": "call"}]
# name: (preflop line, oop, ip, board, pot, stack)
SPOTS = {
    "srp-dry": (SRP, "BB", "BTN", "Qs8d3c", 5.5, 97.5),
    "srp-wet": (SRP, "BB", "BTN", "9h8h6c", 5.5, 97.5),
    "srp-mono": (SRP, "BB", "BTN", "Kh7h2h", 5.5, 97.5),
    "srp-paired": (SRP, "BB", "BTN", "8s8d3c", 5.5, 97.5),
    "3bet-dry": (THREEBET, "CO", "BTN", "Kd7c2h", 16.5, 92.5),
    "turn": (SRP, "BB", "BTN", "Ks7d2cTh", 11.5, 94.5),
    "river": (SRP, "BB", "BTN", "Ks7d2cTh5s", 23.5, 88.5),
}
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64)
# CPU time natively; Pyodide has no process clock
clock = time.perf_counter if sys.platform == "emscripten" else time.process_time


def ranges(line):
    return preflop_ranges(line, pf.Table(pf.SEATS[6]), 100)


def curve(spot, cfg_kw, budget):
    """[(seconds, iterations, exploitability %)] for one spot. Exploitability is measured at fixed iteration
    counts (deterministic); seconds are the setup (tree, leaf matrices, first iteration; the faster of two
    builds) plus the iterations at the fastest per-iteration rate seen over the run's chunks, so a busy
    machine moves the numbers little."""
    line, oop, ip, board, pot, stack = SPOTS[spot]
    rng = ranges(line)
    setup = 1e9
    for _ in range(2):
        t0 = clock()
        sol = solve(cards(board), rng[oop], rng[ip], pot, stack, cfg=Config(**cfg_kw), time_budget=1e9, max_iters=1)
        setup = min(setup, clock() - t0)
    s = sol.solver
    per_it, pts, k = 1e9, [(1, s.exploitability(sol.r0, sol.r1))], 1
    while setup + (k - 1) * min(per_it, 10.0) < budget and k < 1000:
        k = min(1000, max(k + 1, int(k * 1.4)))
        n = k - s.iters
        t0 = clock()
        s.run(sol.r0, sol.r1, time_budget=1e9, max_iters=k, min_iters=n)
        per_it = min(per_it, (clock() - t0) / n)
        pts.append((k, s.exploitability(sol.r0, sol.r1)))
    return [(setup + (i - 1) * per_it, i, e) for i, e in pts]


def at(points, t):
    """Exploitability at the last measurement within t seconds (None before the first)."""
    v = None
    for sec, _, e in points:
        if sec <= t * 1.05:
            v = e
    return v


def run_curves(args):
    if args.noblas:
        S.NO_BLAS = True
    spots = args.spots.split(",") if args.spots else list(SPOTS)
    cfgs = args.configs.split(",")
    rows = {}
    hdr = f"{'spot':12s} {'config':9s} {'setup':>6s} {'it/s':>6s} " + " ".join(f"{str(c) + 's':>7s}" for c in CHECKPOINTS if c <= args.budget)
    print(("NO_BLAS " if S.NO_BLAS else "BLAS ") + sys.platform + ": exploitability (% pot) after N seconds of solving")
    print(hdr, flush=True)
    for spot in spots:
        for c in cfgs:
            pts = curve(spot, CONFIGS[c], args.budget)
            rows[f"{spot}/{c}"] = pts
            ips = (pts[-1][1] - 1) / max(pts[-1][0] - pts[0][0], 1e-9)
            cells = []
            for cp in CHECKPOINTS:
                if cp > args.budget:
                    continue
                v = at(pts, cp)
                cells.append(f"{v:7.2f}" if v is not None else "      -")
            print(f"{spot:12s} {c:9s} {pts[0][0]:6.2f} {ips:6.1f} " + " ".join(cells), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save"); ap.add_argument("--compare")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--noblas", action="store_true")
    ap.add_argument("--spots"); ap.add_argument("--configs", default="fast,accurate")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.curve:
        return run_curves(args)
    if args.noblas:
        S.NO_BLAS = True
    rng = ranges(SRP)
    ref = np.load(args.compare) if args.compare else None
    saved = {}
    for b in BOARDS:
        t0 = time.time()
        sol = solve(cards(b), rng["BB"], rng["BTN"], 5.5, 97.5, cfg=Config(**FAST), time_budget=1e9, max_iters=ITERS)
        total = time.time() - t0
        expl = sol.exploitability()
        strat = sol.root.avg_strategy0()
        saved[b] = strat
        msg = f"{b}: {sol.iters / sol.seconds:6.1f} it/s ({sol.seconds:.2f}s solving, {total:.2f}s total)  exploitability {expl:.3f}%"
        if ref is not None:
            msg += f"  max strategy diff {np.abs(strat - ref[b]).max():.2e}"
        print(msg, flush=True)
    if args.save:
        np.savez(args.save, **saved)


if __name__ == "__main__":
    main()
