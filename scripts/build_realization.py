"""Build gto/pf_realize.py: per hand class and opponent strength bucket, how far a hand's pot share after a
preflop call that sees a flop runs above its all-in equity, measured with the postflop solver, for the
preflop subgame solver (gto/pf_solve.py).

Measured once, on the charts' ranges (gto/preflop.py), for six standard 3bet pots (the spots in
scripts/calibrate_pf_solve.py: openers out of position against a later seat's 3bet and in position
against a blind's): in the 3bet pot, the opener's chart calls against the whole 3bet range; in the 4bet
pot, the 3bettor's chart calls against the opener's chart 4bets. Each is solved on a fixed sample of
flops; a combo's pot share is its EV over the pot (later bets included, so it can fall below 0 or rise
above 1) and its excess that share minus its preflop equity against the opponent's combos in that
bucket. Pair-weighted sums by class and bucket, for each pot type and position, make the table.

An earlier version re-measured on the ranges of the subgame's own solution, round after round, to a
fixed point: it didn't settle (many hands sit near indifference, so a small change in the table flipped
strategies and so the ranges measured next), so the reference ranges are now fixed.

  python scripts/build_realization.py                        # 96 flops, 5s per solve
  python scripts/build_realization.py --flops 8 --budget 1   # quick look
  python scripts/build_realization.py --split DIR            # also write tables from each half of the flops
                                                              # (DIR/pf_realize_a.py, _b.py) to check stability
"""
import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "gto", "pf_realize.py")


def flop_sample(n, seed=11):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        f = tuple(sorted(rng.sample(range(52), 3)))
        if f not in out:
            out.append(f)
    return out


def measure(job):
    """One flop solve: per class and opponent strength bucket, the pair-weighted pot share (EV over the
    pot, which goes below 0 or above 1 when later bets are lost or won), the pair-weighted preflop
    equity and the pair weight, for the out-of-position (0) and in-position (1) players."""
    board, r_oop, r_ip, pot, stack, budget = job
    from gto.solver import solve
    from gto.pf_classes import CLASS_OF, equity_vs, to_classes
    from gto.pf_solve import bucket_of, K
    sol = solve(board, r_oop, r_ip, pot, stack, time_budget=budget)
    s, r0, r1, act = sol.solver, sol.r0, sol.r1, sol.ctx.active
    bfull = bucket_of()[CLASS_OF]
    nb = int(bfull.max()) + 1
    out = []
    s.sampling = False                  # exact expectation over every card in the tree
    try:
        for p, (rc, r_opp_c, r_opp_full) in enumerate(((r0, r1, r_ip), (r1, r0, r_oop))):
            acc = np.zeros((3, K, nb))       # pair-weighted share, pair-weighted equity, pair weight
            for b in range(nb):
                mb = (bfull[act] == b).astype(float)
                ro = r_opp_c * mb
                if ro.sum() < 1e-9:
                    continue
                # utility is zero-sum around half the pot each: + pot/2 is the player's pot share in chips
                rr = (r0, ro) if p == 0 else (ro, r1)
                u = s._cfr(sol.root, p, rr[0][None, :], rr[1][None, :], update=False, use_avg=True)[0]
                cs = s.compat_sum(ro[None, :])[0]
                share = np.where(cs > 1e-12, u / np.maximum(cs, 1e-12) + pot / 2, 0.0) / pot
                w = np.zeros(len(r_oop)); sh = np.zeros(len(r_oop))
                w[act] = rc * cs                 # pairs against this bucket on this board
                sh[act] = share
                eq = np.nan_to_num(equity_vs(r_opp_full * (bfull == b)), nan=0.0)
                acc[0, :, b] = to_classes(w * sh)
                acc[1, :, b] = to_classes(w * eq)
                acc[2, :, b] = to_classes(w)
            out.append(acc)
    finally:
        s.sampling = True
    return out


def chart_spots(o, t, line_text, four):
    """The two flop spots of a case on the charts' ranges: (pot type, oop range, ip range, pot, stack behind)."""
    import calibrate_pf_solve as C
    from gto import preflop as pf
    a = C.line(line_text)
    st = C.replay_hand(C.T, 100, {"preflop": a}, 1)[0]
    while st.to_act != o:               # everyone else folds before the opener acts on the 3bet
        a.append({"pos": st.to_act, "action": "fold"})
        st = C.replay_hand(C.T, 100, {"preflop": a}, 1)[0]
    R = pf.replay_ranges(a, C.T)
    ch = pf.strategy_for(o, a, C.T)
    a4 = a + [{"pos": o, "action": "raise", "to": four}]
    cht = pf.strategy_for(t, a4, C.T)
    o_call, o_4bet = R[o] * ch["call"], R[o] * ch["raise"]
    t_call = pf.replay_ranges(a4, C.T)[t] * cht["call"]
    pot0 = st.pot_start + sum(st.bets.values())
    three, open_in = st.bets[t], st.bets[o]
    pot3 = pot0 + (three - open_in)
    pot4 = pot0 + (four - open_in) + (four - three)
    post = pf.postflop_order(list(C.T))
    if post.index(o) > post.index(t):   # opener in position
        return [("3bet", R[t], o_call, pot3, 100 - three), ("4bet", t_call, o_4bet, pot4, 100 - four)]
    return [("3bet", o_call, R[t], pot3, 100 - three), ("4bet", o_4bet, t_call, pot4, 100 - four)]


def write_table(acc, n_cases, flops, budget):
    from gto.pf_classes import CLASSES
    from gto.pf_solve import BUCKET_EDGES
    lines = ['"""Generated by scripts/build_realization.py: per hand class and opponent strength bucket (gto/pf_solve.py',
             'bucket_of, BUCKET_EDGES), how far its pot share after a call that sees a flop (its EV over the pot,',
             'later bets included) runs above its preflop all-in equity, measured with the postflop solver on the',
             f"charts' ranges of {n_cases} standard 3bet pots ({flops} flops, {budget}s per solve; buckets",
             f'{BUCKET_EDGES}). Entries are [excess share, weight relative to a typical entry], None where',
             'unmeasured; gto/pf_solve.py blends thin measurements with its parametric model."""',
             "TABLE = {"]
    for pt in ("3bet", "4bet"):
        lines.append(f'    "{pt}": {{')
        for pos in ("oop", "ip"):
            sh, eq, w = acc[(pt, pos)]
            med = float(np.median(w[w > 0])) if (w > 0).any() else 1.0
            rows = [[[round(float((sh[i, b] - eq[i, b]) / w[i, b]), 4), round(float(w[i, b] / med), 3)]
                     if w[i, b] > 0 else None for b in range(w.shape[1])] for i in range(len(CLASSES))]
            lines.append(f'        "{pos}": {rows!r},')
        lines.append("    },")
    lines.append("}")
    lines.append(f"CLASSES = {list(CLASSES)!r}")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flops", type=int, default=96)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--split", help="directory to also write the tables from each half of the flops")
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    import calibrate_pf_solve as C
    t0 = time.time()
    flops = flop_sample(args.flops)
    jobs, keys = [], []
    for o, t, s, four in C.CASES:
        for pt, r_oop, r_ip, pot, stack in chart_spots(o, t, s, four):
            for k, f in enumerate(flops):
                jobs.append((f, r_oop, r_ip, pot, stack, args.budget))
                keys.append((pt, k % 2))
    with Pool(args.procs) as pool:
        res = pool.map(measure, jobs)
    halves = [{(pt, pos): 0.0 for pt in ("3bet", "4bet") for pos in ("oop", "ip")} for _ in range(2)]
    for (pt, h), r in zip(keys, res):
        for pos, a in zip(("oop", "ip"), r):
            halves[h][(pt, pos)] = halves[h][(pt, pos)] + a
    total = {k: halves[0][k] + halves[1][k] for k in halves[0]}
    write_table(total, len(C.CASES), args.flops, args.budget)
    if args.split:
        global OUT
        out = OUT
        for h, name in ((0, "a"), (1, "b")):
            OUT = os.path.join(args.split, f"pf_realize_{name}.py")
            write_table(halves[h], len(C.CASES), args.flops // 2, args.budget)
        OUT = out
    print(f"{len(jobs)} solves in {time.time() - t0:.0f}s")
    for pt in ("3bet", "4bet"):
        for pos in ("oop", "ip"):
            sh, eq, w = total[(pt, pos)]
            print(f"  {pt} {pos}: pot share {float(sh.sum() / w.sum()):.3f} against equity {float(eq.sum() / w.sum()):.3f}")


if __name__ == "__main__":
    main()
