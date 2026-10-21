"""Postflop solver speed benchmarks.

Default (regression check): BTN vs BB single-raised pot on three flops with the browser's Fast config, a
fixed 150 iterations. Prints iterations/second and the exploitability reached.

    python scripts/bench_solver.py [--save FILE] [--compare FILE]

--save writes each board's root average strategy to an .npz; --compare checks a run against one
(speedups must leave the results unchanged).

Flop strategies (--flop): c-bet frequency and sizes, the defender's check and its response to each bet, per board
and config, solved for --budget seconds (wall clock).

    python scripts/bench_solver.py --flop [--spots K72,QQ6] [--configs fast,accurate,native] [--budget 8] [--noblas]

Curves (--curve): exploitability (% of the pot) against solving time for a set of spots (single-raised and
3bet pots; dry, wet, monotone and paired flops, a turn, a river) with the browser's Fast and Accurate
configs and the native default. Exploitability is measured at fixed iteration counts (deterministic) and
mapped to time by the setup time and the fastest per-iteration rate seen, in CPU seconds, so other work on
the machine moves the numbers little. numpy's BLAS is held to one thread (the browser has one) unless
BENCH_THREADS is set. --noblas runs the browser's kernels (einsum, trimmed leaves) natively; for Pyodide
itself use scripts/pyodide_run.mjs (NO_BLAS is on there and times are wall-clock).

    python scripts/bench_solver.py --curve [--noblas] [--spots srp-dry,turn] [--configs fast] [--budget 8]

Regret minimizers (--algos): exploitability (% of the pot) at fixed iteration counts for each Config.cfr scheme
(gto.solver.CFR_SCHEMES), per spot and config, and the per-iteration cost (the fastest chunk, CPU ms). Iterations,
not seconds, so the comparison is exact on a loaded machine; the cost column says what an iteration takes.

    python scripts/bench_solver.py --algos dcfr,pcfr+,pdcfr+ [--spots river,turn] [--configs fast] [--iters 200]
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
BUCKETS = dict(river_buckets=32, turn_buckets=True, bucket_river_step=3)    # Fast's flop model (native and browser)
# (from a turn root: river classes, Fast 33/125% and Accurate 33/75/125% on the turn and river; a flop solve plays
# later_sizes' one size there)
FAST = dict(n_turns=3, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4, river_classes=True, **BUCKETS,
            **dict(LS, bet_sizes={"flop": [0.33, 0.75], "turn": [0.33, 1.25], "river": [0.75]}))
ACCURATE = dict(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8, river_classes=True,
                **dict(LS, bet_sizes={"flop": [0.33, 0.75], "turn": [0.33, 0.75, 1.25], "river": [0.33, 0.75, 1.25]}))
# the browser's configs (static/worker.js) and Config()'s defaults; "every-turn" plays every turn card from the
# flop (the reference the turn classes are measured against: slow), "accurate-old" is Accurate before river_play
# and turn_classes (4 random turns, the river checked down), "fast-old" Fast before the bucketed turn and river (3
# random turns, the river checked down), "native-fast" and "native-fast-old" gto/native.py's fast tree now and before
CONFIGS = {"fast": FAST, "accurate": ACCURATE, "native": {}, "every-turn": dict(ACCURATE, turn_classes=False, n_turns=49),
           "accurate-old": dict(ACCURATE, river_play=False, turn_classes=False),
           "fast-old": dict(FAST, river_play=False, turn_classes=False, river_buckets=None, turn_buckets=False),
           "native-fast": dict(BUCKETS, **LS), "native-fast-old": dict(river_play=False, turn_classes=False, **LS)}
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


def run_algos(args):
    if args.noblas:
        S.NO_BLAS = True
    spots = args.spots.split(",") if args.spots else list(SPOTS)
    algos = args.algos.split(",")
    marks = [m for m in (5, 10, 20, 50, 100, 200, 500, 1000, 2000) if m <= args.iters]
    if args.iters not in marks:
        marks.append(args.iters)
    print(("NO_BLAS " if S.NO_BLAS else "BLAS ") + "exploitability (% pot) after N iterations; ms: CPU per iteration")
    print(f"{'spot':12s} {'config':9s} {'cfr':9s} {'ms':>6s} " + " ".join(f"{m:>7d}" for m in marks), flush=True)
    rows = {}
    for spot in spots:
        line, oop, ip, board, pot, stack = SPOTS[spot]
        rng = ranges(line)
        for c in args.configs.split(","):
            for algo in algos:
                sol = solve(cards(board), rng[oop], rng[ip], pot, stack, cfg=Config(**CONFIGS[c], cfr=algo),
                            time_budget=1e9, max_iters=0)
                s = sol.solver
                s.horizon = args.iters
                per_it, cells, pts = 1e9, [], []
                for m in marks:
                    n = m - s.iters
                    t0 = clock()
                    s.run(sol.r0, sol.r1, time_budget=1e9, max_iters=m, min_iters=n)
                    per_it = min(per_it, (clock() - t0) / n)
                    e = s.exploitability(sol.r0, sol.r1)
                    pts.append((m, e))
                    cells.append(f"{e:7.3f}")
                rows[f"{spot}/{c}/{algo}"] = {"ms": per_it * 1000, "points": pts}
                print(f"{spot:12s} {c:9s} {algo:9s} {per_it * 1000:6.1f} " + " ".join(cells), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f)


# --flop: BTN vs BB single-raised pots and CO vs BTN 3bet pots on a spread of textures (see tests TestFlopStrategy)
FLOP_BOARDS = {"K72": "Kd7c2h", "QQ6": "QsQd6c", "A74": "As7h4d", "883": "8s8d3c", "765": "7h6d5c", "J95": "Jc9d5h",
               "Kh7h2h": "Kh7h2h", "986": "9h8h6c", "T98": "Th9c8d", "3b-K72": "Kd7c2h", "3b-A83": "Ah8c3d",
               "3b-765": "7h6d5c"}


def flop_stats(sol) -> dict:
    """The flop root's aggregate frequencies: the out-of-position player's check, the in-position player's
    actions after it, and the out-of-position player's response to each of those bets."""
    root, r0, r1 = sol.root, sol.r0, sol.r1
    st0 = root.avg_strategy0()
    out = {"oop check": float((st0[0] * r0).sum() / r0.sum())}
    ipn = root.children[0]
    st1 = ipn.avg_strategy0()
    for a, name in enumerate(ipn.names):
        out[name] = float((st1[a] * r1).sum() / r1.sum())
        if name != "check":
            reach = r0 * st0[0]
            s = ipn.children[a].avg_strategy0()
            resp = {}
            for b, m in enumerate(ipn.children[a].names):
                k = "raise" if m.startswith("raise") or m == "all-in" else m
                resp[k] = resp.get(k, 0.0) + float((s[b] * reach).sum() / reach.sum())
            out["vs " + name] = resp
    return out


def run_flop(args):
    """Flop strategies per board and config under a wall-clock budget (Pyodide has no process clock)."""
    if args.noblas:
        S.NO_BLAS = True
    rng = {"srp": ranges(SRP), "3b": ranges(THREEBET)}
    rows = {}
    for name in (args.spots.split(",") if args.spots else list(FLOP_BOARDS)):
        board = FLOP_BOARDS[name]
        if name.startswith("3b-"):
            r0, r1, pot, stack = rng["3b"]["CO"], rng["3b"]["BTN"], 16.5, 92.5
        else:
            r0, r1, pot, stack = rng["srp"]["BB"], rng["srp"]["BTN"], 5.5, 97.5
        for c in args.configs.split(","):
            t0 = time.time()
            sol = solve(cards(board), r0, r1, pot, stack, cfg=Config(**CONFIGS[c]), time_budget=args.budget, max_iters=10000)
            st = flop_stats(sol)
            st["iters"], st["seconds"] = sol.iters, time.time() - t0
            rows[f"{name}/{c}"] = st
            bets = {k: v for k, v in st.items() if k.startswith(("bet", "all-in"))}
            resp = " ".join(f"{k[3:]}: " + "/".join(f"{a[0]}{v * 100:.0f}" for a, v in r.items())
                            for k, r in st.items() if k.startswith("vs "))
            print(f"{name:7s} {c:9s} check {st['oop check'] * 100:3.0f} | c-bet {sum(bets.values()) * 100:5.1f} = "
                  + " ".join(f"{k}: {v * 100:.0f}" for k, v in bets.items()) + f" | {resp} | {sol.iters} it {st['seconds']:.1f}s",
                  flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save"); ap.add_argument("--compare")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--flop", action="store_true")
    ap.add_argument("--noblas", action="store_true")
    ap.add_argument("--algos"); ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--spots"); ap.add_argument("--configs", default="fast,accurate")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.algos:
        return run_algos(args)
    if args.curve:
        return run_curves(args)
    if args.flop:
        return run_flop(args)
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
