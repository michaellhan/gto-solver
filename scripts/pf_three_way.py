"""Three-way realization for the preflop game (gto/pf_game.py, MW_REALIZE): single-raised pots that three players
see, solved with the 3-player solver (gto/multiway_solver.py) on the charts' ranges (gto/preflop.py), and each
player's pot share (EV over the pot, later bets included) against its three-way all-in equity, per postflop seat.

The tree: flop and turn bet (a third pot, two thirds on the turn; one raise), the river dealt out (an equity
leaf, 4 rivers a turn, 6 turns); 40 iterations. Shares and equities are reported as solved and both scaled to add
to the pot. Stratified flops (build_realization.flop_sample), every 11th of 68.

  python scripts/pf_three_way.py --spot CO-BTN-BB --flops 6 --procs 2     # solve (resumable)
  python scripts/pf_three_way.py --report
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

DEFAULT_DIR = os.path.join(ROOT, ".pf_cache", "three_way")
# name: (the chart actions to the flop, the three in postflop order)
SPOTS = {
    "BTN-SB-BB": ([("UTG", "fold"), ("HJ", "fold"), ("CO", "fold"), ("BTN", "raise", 2.5), ("SB", "call"),
                   ("BB", "call")], ["SB", "BB", "BTN"]),
    "CO-BTN-BB": ([("UTG", "fold"), ("HJ", "fold"), ("CO", "raise", 2.5), ("BTN", "call"), ("SB", "fold"),
                   ("BB", "call")], ["BB", "CO", "BTN"]),
    "UTG-CO-BB": ([("UTG", "raise", 2.5), ("HJ", "fold"), ("CO", "call"), ("BTN", "fold"), ("SB", "fold"),
                   ("BB", "call")], ["BB", "UTG", "CO"]),
}


def ranges(spot):
    from gto import preflop as pf
    line, order = SPOTS[spot]
    acts = [{"pos": a[0], "action": a[1], **({"to": a[2]} if len(a) > 2 else {})} for a in line]
    R = pf.replay_ranges(acts, pf.SEATS[6])
    return [R[p] for p in order]


def job(args):
    path, board, rs, pot, stack, iters = args
    if os.path.exists(path):
        return path
    from gto.multiway_solver import MultiwaySolver, MWConfig
    cfg = MWConfig(depth=2, turns=6, leaf_rivers=4, bet={"flop": 0.33, "turn": 0.66, "river": 0.75})
    s = MultiwaySolver(board, rs, pot, stack, cfg=cfg)
    s.run(iters=iters)
    m = s.total_mass()
    v = s._walk(s.root, s.reach0, "avg")
    # the same deal with no betting: all-in equity
    e = s._walk(s.builder._runout([tuple(board)], (0, 1, 2), pot, [0.0, 0.0, 0.0]), s.reach0, "avg")
    out = {}
    for p in range(3):
        r = s.reach0[p][0]
        out[f"share{p}"] = float((v[p][0] * r).sum() / m / pot)
        out[f"eq{p}"] = float((e[p][0] * r).sum() / m / pot)
    np.savez(path + ".tmp.npz", **out)
    os.replace(path + ".tmp.npz", path)
    return path


def run(args):
    import build_realization as B
    from multiprocessing import Pool
    rs = ranges(args.spot)
    pot = 7.5 + (0.0 if "SB" in SPOTS[args.spot][1] else 0.5)
    flops = B.flop_sample(68)[:: max(1, 68 // args.flops)][:args.flops]
    os.makedirs(args.dir, exist_ok=True)
    jobs = [(os.path.join(args.dir, f"{args.spot}_{k}.npz"), f, rs, pot, 97.5, args.iters)
            for k, (f, *_) in enumerate(flops)]
    t0 = time.time()
    with Pool(args.procs) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, jobs)):
            print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)


def report(args):
    for spot, (line, order) in SPOTS.items():
        fs = sorted(glob.glob(os.path.join(args.dir, f"{spot}_*.npz")))
        if not fs:
            continue
        sh = np.mean([[np.load(f)[f"share{p}"] for p in range(3)] for f in fs], 0)
        eq = np.mean([[np.load(f)[f"eq{p}"] for p in range(3)] for f in fs], 0)
        r = sh / eq
        print(f"{spot:10s} {len(fs)} flops  {' / '.join(order)}: equity {np.round(eq, 3)}  share {np.round(sh, 3)}"
              f"  realization {np.round(r, 3)}, both scaled to add to 1: {np.round(r * eq.sum() / sh.sum(), 3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", choices=sorted(SPOTS))
    ap.add_argument("--flops", type=int, default=6)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
