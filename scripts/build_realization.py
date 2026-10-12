"""Build gto/pf_realize.py: per hand class and opponent strength bucket, how far a hand's pot share after a
preflop call that sees a flop runs above its all-in equity, measured with the postflop solver, for the
preflop subgame solver (gto/pf_solve.py).

Measured once, on the charts' ranges (gto/preflop.py), for six standard 3bet pots (the spots in
scripts/calibrate_pf_solve.py: openers out of position against a later seat's 3bet and in position
against a blind's): in the 3bet pot, the opener's chart calls against the whole 3bet range; in the 4bet
pot, the 3bettor's chart calls against the opener's chart 4bets. Each is solved on the same flops,
stratified by texture (suit pattern, pairing, high card, connectedness) and weighted by the share of all
flops each stands for, which removes most of the noise of picking flops at random; a combo's pot share is its EV over the pot (later bets included, so it can fall below 0 or rise
above 1) and its excess that share minus its preflop equity against the opponent's combos in that
bucket. Pair-weighted sums by class and bucket, for each pot type, role (the caller, or the aggressor
whose bet was called) and position, make the table: position alone mixes roles that realize very
differently (in a 3bet pot an in-position 3bettor runs ~6 points of pot share above its equity, an
in-position caller about level with it).

An earlier version re-measured on the ranges of the subgame's own solution, round after round, to a
fixed point: it didn't settle (many hands sit near indifference, so a small change in the table flipped
strategies and so the ranges measured next), so the reference ranges are now fixed.

  python scripts/build_realization.py                        # 128 flops, 5s per solve
  python scripts/build_realization.py --flops 64 --budget 1  # quick look
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


def iso_flops():
    """Every flop up to suit symmetry, with how many of the 22100 flops each stands for."""
    from itertools import combinations, permutations
    counts = {}
    for f in combinations(range(52), 3):
        key = min(tuple(sorted(4 * (c // 4) + p[c % 4] for c in f)) for p in permutations(range(4)))
        counts[key] = counts.get(key, 0) + 1
    return counts


def texture(f):
    """Stratum of a flop: suit pattern, pairing, high card, and (unpaired) whether it is connected; the
    rare textures coarser (trips one stratum, monotone by high card only)."""
    ranks = sorted((c // 4 for c in f), reverse=True)
    suits = len({c % 4 for c in f})
    pairing = 3 - len(set(ranks))
    hi = ranks[0]
    if pairing == 2:
        return ("trips",)
    if suits == 1:
        return ("monotone", 0 if hi == 12 else 1 if hi >= 10 else 2)
    hi_b = 0 if hi == 12 else 1 if hi == 11 else 2 if hi >= 9 else 3 if hi >= 6 else 4
    conn = pairing == 0 and (ranks[0] - ranks[2] <= 4 or (hi == 12 and ranks[1] <= 3))
    return (suits, pairing, hi_b, conn)


def flop_sample(n, seed=11, min_per=2):
    """n flops stratified by texture: each stratum gets at least min_per (so each half has it) and the
    rest in proportion to the flops it holds (largest remainder), picked by systematic sampling over its
    suit-isomorphic flops by weight. Returns [(flop, weight, half, weight within its half)]: a flop's
    weight is its stratum's share of all flops over the number picked there (scaled to sum to n), the
    halves alternate within each stratum, and each half is weighted on its own."""
    rng = random.Random(seed)
    counts = iso_flops()
    strata = {}
    for f, c in counts.items():
        strata.setdefault(texture(f), []).append((f, c))
    total = sum(counts.values())
    keys = sorted(strata, key=str)
    share = {k: sum(c for _, c in strata[k]) / total for k in keys}
    rest = n - min_per * len(keys)
    if rest < 0:
        raise ValueError(f"need at least {min_per * len(keys)} flops for {len(keys)} strata")
    alloc = {k: min_per + int(rest * share[k]) for k in keys}
    for k in sorted(keys, key=lambda k: rest * share[k] - int(rest * share[k]), reverse=True)[:n - sum(alloc.values())]:
        alloc[k] += 1
    out = []
    for k in keys:
        m = alloc[k]
        items = strata[k][:]
        rng.shuffle(items)
        w_tot = sum(c for _, c in items)
        step, start = w_tot / m, rng.random() * w_tot / m
        cum, j, picks = 0.0, 0, []
        for f, c in items:
            while j < m and start + j * step < cum + c:
                picks.append(f); j += 1
            cum += c
        m_half = [sum(1 for i in range(m) if i % 2 == h) for h in (0, 1)]
        for idx, f in enumerate(picks):
            h = idx % 2
            out.append((f, share[k] * n / m, h, share[k] * (n / 2) / m_half[h]))
    return out


def measure(job):
    """One flop solve: per class and opponent strength bucket, the pair-weighted pot share (EV over the
    pot, which goes below 0 or above 1 when later bets are lost or won), the pair-weighted preflop
    equity and the pair weight, for the out-of-position (0) and in-position (1) players."""
    board, r_oop, r_ip, pot, stack, budget = job[:6]
    depth = job[6] if len(job) > 6 else None      # streets with full decisions (None: the solver's default)
    sizes = job[7] if len(job) > 7 else None      # bet sizes per street (None: the solver's default)
    from gto.solver import solve, Config
    from gto.pf_classes import CLASS_OF, equity_vs, to_classes
    from gto.pf_solve import bucket_of, K
    cfg = Config()
    if depth is not None:
        cfg.depth = depth
    if sizes is not None:
        cfg.bet_sizes = {k: list(v) for k, v in sizes.items()}
    sol = solve(board, r_oop, r_ip, pot, stack, cfg=cfg, time_budget=budget)
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
    """The two flop spots of a case on the charts' ranges."""
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
    # (pot type, oop range, ip range, pot, stack behind, oop's role, ip's role): in the 3bet pot the
    # opener called, in the 4bet pot the 3bettor did; the other is the aggressor
    if post.index(o) > post.index(t):   # opener in position
        return [("3bet", R[t], o_call, pot3, 100 - three, "aggr", "caller"),
                ("4bet", t_call, o_4bet, pot4, 100 - four, "caller", "aggr")]
    return [("3bet", o_call, R[t], pot3, 100 - three, "caller", "aggr"),
            ("4bet", o_4bet, t_call, pot4, 100 - four, "aggr", "caller")]


def write_table(acc, n_cases, flops, budget):
    from gto.pf_classes import CLASSES
    from gto.pf_solve import BUCKET_EDGES
    lines = ['"""Generated by scripts/build_realization.py: per hand class and opponent strength bucket (gto/pf_solve.py',
             'bucket_of, BUCKET_EDGES), per pot type, role (caller / aggr, the aggressor whose bet was called) and',
             'position, how far its pot share after a call that sees a flop (its EV over the pot,',
             'later bets included) runs above its preflop all-in equity, measured with the postflop solver on the',
             f"charts' ranges of {n_cases} standard 3bet pots ({flops} flops, {budget}s per solve; buckets",
             f'{BUCKET_EDGES}). Entries are [excess share, weight relative to a typical entry], None where',
             'unmeasured; gto/pf_solve.py blends thin measurements with its parametric model."""',
             "TABLE = {"]
    for pt in ("3bet", "4bet"):
        lines.append(f'    "{pt}": {{')
        for pos in sorted(k for p, k in acc if p == pt):
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
    ap.add_argument("--flops", type=int, default=128)
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
        for pt, r_oop, r_ip, pot, stack, role_oop, role_ip in chart_spots(o, t, s, four):
            for f, fw, half, fw_half in flops:
                jobs.append((f, r_oop, r_ip, pot, stack, args.budget))
                keys.append((pt, (f"{role_oop}_oop", f"{role_ip}_ip"), half, fw, fw_half))
    with Pool(args.procs) as pool:
        res = pool.map(measure, jobs)
    total, halves = {}, [{}, {}]
    for (pt, sides, h, fw, fw_half), r in zip(keys, res):
        for side, a in zip(sides, r):            # each flop by its stratum's weight
            total[(pt, side)] = total.get((pt, side), 0.0) + fw * a
            halves[h][(pt, side)] = halves[h].get((pt, side), 0.0) + fw_half * a
    write_table(total, len(C.CASES), args.flops, args.budget)
    if args.split:
        global OUT
        out = OUT
        for h, name in ((0, "a"), (1, "b")):
            OUT = os.path.join(args.split, f"pf_realize_{name}.py")
            write_table(halves[h], len(C.CASES), args.flops // 2, args.budget)
        OUT = out
    print(f"{len(jobs)} solves in {time.time() - t0:.0f}s")
    for pt, side in sorted(total):
        sh, eq, w = total[(pt, side)]
        print(f"  {pt} {side}: pot share {float(sh.sum() / w.sum()):.3f} against equity {float(eq.sum() / w.sum()):.3f}"
              f" (realization {float(sh.sum() / eq.sum()):.3f})")


if __name__ == "__main__":
    main()
