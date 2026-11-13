"""Calibrate exploit mode's preflop pool (gto/exploit.py) to population HUD stats.

Walks the 6-max 100bb preflop tree (every line the charts take, sizes as scripts/sanity_sweep.py, lines
rarer than --prune dropped), weighting each line by how often it happens (each player's share of their
range taking their action), and reports the HUD stats of the charts and of the pool:

  VPIP          share of hands a seat puts chips in voluntarily (BB checks and blinds don't count)
  PFR           share of hands a seat raises
  3bet          raises when facing exactly one raise, of those opportunities
  4bet          the opener's raises facing a 3bet, of those opportunities
  fold to 3bet  the opener's folds facing a 3bet
  flat vs open  calls facing one raise, the big blind excepted (limpers facing a raise included)
  limp          open-limps, of the chances to enter first (the big blind excepted)
  limps, flats  open-limps and non-BB calls facing one raise, as a share of all hands a seat is dealt
                (where the extra VPIP goes)

The targets (exploit.py) are Hand2Note mass data of GGPoker 6-max, all players, NL50-NL500 (Bluffaces
MDA): VPIP 28.75, PFR 18.7, 3bet 8.9, fold to 3bet 48.25, 4bet 12.75. POOL_3BET_SCALE sets the 3bet,
POOL_EXTRA_ENTER the VPIP, POOL_3BET_DEFEND the fold to 3bet, POOL_EXTRA_OPEN (the share of the extra
first-in hands that open-raise rather than limp) the PFR; the 4bet is left at the chart's.

  python scripts/calibrate_pool.py                          # charts and the current pool
  python scripts/calibrate_pool.py 1.1,0.1,0.03 1.1,0.15,0.03  # pool at scale,enter,defend (parallel)
  python scripts/calibrate_pool.py 1.1,0.1,0.03,0.4         # ... and open
  python scripts/calibrate_pool.py --no-charts 1.1,0.1,0.03
  python scripts/calibrate_pool.py --live 0.75,0.48,0.22,0.15,0.15   # the live player (9-handed): ...,open,openlimp

Each tree takes a few minutes (the pool's longer: it computes equities against the raiser).
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

N = 1326


NAMES = ("POOL_3BET_SCALE", "POOL_EXTRA_ENTER", "POOL_3BET_DEFEND", "POOL_EXTRA_OPEN", "POOL_OPEN_LIMP")


def hud(pool, params=None, prune=1e-5, live=False, seats=6):
    """The tree's HUD stats (percent), the charts' or (pool) the player pool's with params
    (POOL_3BET_SCALE, POOL_EXTRA_ENTER, POOL_3BET_DEFEND[, POOL_EXTRA_OPEN]; None:
    exploit.py's; parameters left out keep exploit.py's values)."""
    from gto import preflop as pf, exploit as ex
    from gto import advisor as adv
    from gto.advisor import replay_hand, preflop_strategy, pool_strategy
    from sanity_sweep import _size
    if live:
        ex.set_profile("live")
    if params is not None:
        for name, v in zip(NAMES, params):
            if live:
                ex.LIVE[name] = v
            else:
                setattr(ex, name, v)
    adv._PF_CACHE.clear()
    T = pf.Table(pf.SEATS[seats], 0.0, False)
    stack = 100.0
    vpip = {p: 0.0 for p in T}
    pfr = {p: 0.0 for p in T}
    tb, fb, f3, flat, limp = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]   # [hits, opportunities]

    def dfs(acts, ranges, reach, vol, rai):
        st = replay_hand(T, stack, {"preflop": acts}, 1)[0]
        if st.closed or st.to_act is None:
            for p in vol:
                vpip[p] += reach
            for p in rai:
                pfr[p] += reach
            return
        p, nr = st.to_act, st.n_raises
        strat = preflop_strategy(p, acts, T, stack)
        if pool:
            strat = pool_strategy(strat, p, acts, {q: ranges.get(q, np.ones(N)) for q in T})
        w = ranges.get(p, np.ones(N))
        tot = max(w.sum(), 1e-12)
        sh = {k: float((w * v).sum() / tot) for k, v in strat.items()}
        raised = sh.get("raise", 0.0) + sh.get("all-in", 0.0)
        opener = next((h["pos"] for h in acts if h["action"] in ("raise", "all-in")), None)
        if nr == 1 and p not in rai:
            tb[1] += reach; tb[0] += reach * raised
            if p != "BB":
                flat[1] += reach; flat[0] += reach * sh.get("call", 0.0)
        if nr == 0 and p != "BB" and not any(h["action"] == "call" for h in acts):
            limp[1] += reach; limp[0] += reach * sh.get("call", 0.0)
        if nr == 2 and p == opener:
            fb[1] += reach; fb[0] += reach * raised
            f3[1] += reach; f3[0] += reach * sh.get("fold", 0.0)
        for key, v in strat.items():
            s = sh[key]
            if s < 1e-9 or reach * s < prune:
                continue
            if key == "fold":
                a, v2, r2 = {"pos": p, "action": "fold"}, vol, rai
            elif key in ("call", "check"):
                a = {"pos": p, "action": key}
                v2 = vol | ({p} if not (key == "check" or (p == "BB" and nr == 0)) else set())
                r2 = rai
            else:
                to = _size(st, p, nr, T, stack)
                if nr == 0:   # an iso-raise: a big blind more per limper
                    to += st.cur_bet * sum(h["action"] == "call" for h in acts)
                a = ({"pos": p, "action": "all-in"} if to >= stack - st.inv_start[p] - 1e-9 or nr >= 3
                     else {"pos": p, "action": "raise", "to": to})
                v2, r2 = vol | {p}, rai | {p}
            dfs(acts + [a], dict(ranges, **{p: w * v}), reach * s, v2, r2)

    dfs([], {}, 1.0, frozenset(), frozenset())
    n = len(T)
    return {"VPIP": 100 * sum(vpip.values()) / n, "PFR": 100 * sum(pfr.values()) / n,
            "3bet": 100 * tb[0] / tb[1], "4bet": 100 * fb[0] / fb[1], "fold to 3bet": 100 * f3[0] / f3[1],
            "flat vs open": 100 * flat[0] / flat[1], "limp": 100 * limp[0] / limp[1],
            "limps": 100 * limp[0] / n, "flats": 100 * flat[0] / n}


def _job(job):
    pool, params, live, seats = job
    t = time.time()
    out = hud(pool, params, live=live, seats=seats)
    if not pool:
        label = "charts"
    else:
        from gto import exploit as ex
        v = dict(zip(NAMES, ex._params(*NAMES)))
        label = (f"{'live' if live else 'pool'} scale={v['POOL_3BET_SCALE']:g} enter={v['POOL_EXTRA_ENTER']:g} "
                 f"defend={v['POOL_3BET_DEFEND']:g} open={v['POOL_EXTRA_OPEN']:g} openlimp={v['POOL_OPEN_LIMP']:g}")
    return f"{label}: " + " ".join(f"{k} {v:.1f}" for k, v in out.items()) + f"  ({time.time() - t:.0f}s)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("params", nargs="*", help="pool parameters to try, as scale,enter,defend[,open[,openlimp]]")
    ap.add_argument("--no-charts", action="store_true", help="skip the charts' own stats")
    ap.add_argument("--live", action="store_true", help="the typical live player (exploit.LIVE), 9-handed by default")
    ap.add_argument("--seats", type=int, default=None, help="players at the table (default 6, --live 9)")
    ap.add_argument("--jobs", type=int, default=2, help="processes at once (default 2)")
    args = ap.parse_args()
    seats = args.seats or (9 if args.live else 6)
    jobs = [] if args.no_charts else [(False, None, False, seats)]
    jobs += [(True, tuple(map(float, s.split(","))), args.live, seats) for s in args.params] or [(True, None, args.live, seats)]
    if args.live:
        print("targets (live $1/$2-$5/$10, exploit.LIVE): VPIP 38 PFR 10-12 3bet 3 fold to 3bet 35")
    else:
        print("targets (GGPoker 6-max NL50-NL500, all players): VPIP 28.75 PFR 18.7 3bet 8.9 4bet 12.75 fold to 3bet 48.25")
    with Pool(max(1, min(len(jobs), args.jobs))) as p:
        for line in p.imap(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
