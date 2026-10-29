"""3-way flop abstractions against a rich one: a 3-way flop (CO opens, BTN and BB call; 6-max 100bb) solved in several
bucket games for the same iterations and each compared with the first game, the reference: the root's range
frequencies, the first to act's per-combo value at the root and its check EV (% of the pot, weighted by its range),
the mix's total variation, each game's own exploitability and CPU (the README's 3-way pots: Fast's flop game).

  python scripts/bench_mw_abstraction.py Kh8d3c 0 400 32:1:33+75:50,16:1:33+75:50,12:3:33+75:50,6:6:50:50
  python scripts/bench_mw_abstraction.py Kh8d3c 2 400 ...          # checked to twice (the BTN)

A game is K:step:flop sizes:turn sizes (buckets a player below the flop, the turn's tables over every step-th river,
the sizes in % of the pot joined by +).
"""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import gto.advisor as A
from gto import preflop as pf
from gto.cards import cards
from gto import multiway_solver as MW
PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
       {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
R = A.preflop_ranges(PRE, pf.Table(pf.SEATS[6]), 100.0)
rs = [R["BB"], R["CO"], R["BTN"]]
board = sys.argv[1]; checked = int(sys.argv[2]); iters = int(sys.argv[3])
games = [g.split(":") for g in sys.argv[4].split(",")]    # K:step:flopsizes(+ separated):turnsizes
def solve(K, step, flop, turn):
    bet = {"flop": [float(x) / 100 for x in flop.split("+")], "turn": [float(x) / 100 for x in turn.split("+")] if "+" in turn else float(turn) / 100,
           "river": 0.5}
    cfg = MW.MWConfig(later="buckets", buckets=int(K), bucket_river_step=int(step), bet=bet)
    kw = dict(to_act=checked, pending=set(range(checked, 3))) if checked else {}
    t0 = time.process_time()
    s = MW.MultiwaySolver(cards(board), rs, 8.0, 97.5, cfg=cfg, **kw)
    b = time.process_time() - t0
    s.run(iters=iters)
    g = s.br_gains() / s.root_pot() * 100
    root = s.root
    p = root.player
    sig = root.avg_strategy()[0]
    ev = s.action_evs()
    val = (sig * ev).sum(axis=0)
    w = s.reach0[p, 0] / s.reach0[p, 0].sum()
    freqs = dict(zip(root.names, (sig * w).sum(axis=1).round(3)))
    return dict(val=val, w=w, freqs=freqs, sig=dict(zip(root.names, sig)), ev=dict(zip(root.names, ev)),
                build=b, cpu=time.process_time() - t0 - b, expl=g.mean(), pot=s.root_pot())
ref = solve(*games[0])
print(f"{board} checked={checked} iters={iters} ref {games[0]}: expl {ref['expl']:.3f}% freqs {ref['freqs']} "
      f"build {ref['build']:.1f} cpu {ref['cpu']:.1f}", flush=True)
for g in games[1:]:
    o = solve(*g)
    dv = (np.abs(o["val"] - ref["val"]) * ref["w"]).sum() / ref["pot"] * 100
    bias = ((o["val"] - ref["val"]) * ref["w"]).sum() / ref["pot"] * 100
    common = [a for a in o["sig"] if a in ref["sig"]]
    tv = 0.5 * sum((np.abs(o["sig"][a] - ref["sig"][a]) * ref["w"]).sum() for a in common) if set(common) == set(ref["sig"]) else float("nan")
    dch = (np.abs(o["ev"]["check"] - ref["ev"]["check"]) * ref["w"]).sum() / ref["pot"] * 100 if "check" in common else float("nan")
    print(f"  {':'.join(g):18s} expl {o['expl']:.3f}%  value |d| {dv:.3f}% (bias {bias:+.3f})  check-EV |d| {dch:.3f}%  "
          f"mix TV {tv:.3f}  freqs {o['freqs']}  build {o['build']:.1f} cpu {o['cpu']:.1f}", flush=True)
