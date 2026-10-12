"""Calibrate the preflop subgame solver's realization constants against the charts.

gto/pf_solve.py solves the 3bet pot heads-up (opener facing the 3bet: fold/call/4bet; 3bettor facing
the 4bet: fold/call/jam; opener facing the jam: fold/call). Calls that see a flop pay realized equity,
set by REALIZE (base by position and pot type) and REALIZE_SLOPE (below / above 50% equity). The
charts in gto/preflop.py are hand-written approximations of solver output, so their continue rates
are the target: for six standard spots, the opener's call and 4bet shares and the 3bettor's call
and jam shares against the 4bet. Position stays ordered (in position realizes at least as much).

Score: sum of squared differences in those shares.

  python scripts/calibrate_pf_solve.py           # score the current constants, with ranges
  python scripts/calibrate_pf_solve.py --tune    # grid over the slopes and the 4bet-pot OOP base
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import gto.pf_solve as P
from gto import preflop as pf
from gto.advisor import replay_hand
from gto.pf_classes import CLASSES, to_classes

T = pf.Table(pf.SEATS[6])
# opener out of position (a later seat 3bets) and in position (a blind 3bets)
CASES = [("CO", "BTN", "CO raise 2.5, BTN raise 7.5", 18.0),
         ("BTN", "BB", "BTN raise 2.5, BB raise 10", 23.0),
         ("UTG", "CO", "UTG raise 2.5, CO raise 7.5", 18.0),
         ("HJ", "BTN", "HJ raise 2.5, BTN raise 7.5", 18.0),
         ("CO", "BB", "CO raise 2.5, BB raise 10", 23.0),
         ("BTN", "SB", "BTN raise 2.5, SB raise 11", 25.0)]


def line(s):
    """'CO raise 2.5, BTN raise 7.5' -> the full action list, everyone skipped in turn folding."""
    acts = []
    for a in s.split(", "):
        w = a.split()
        d = {"pos": w[0], "action": w[1], **({"to": float(w[2])} if len(w) > 2 else {})}
        while (st := replay_hand(T, 100, {"preflop": acts}, 1)[0]).to_act != d["pos"]:
            acts.append({"pos": st.to_act, "action": "fold"})
        acts.append(d)
    return acts


def prepare():
    out = []
    for o, t, s, four in CASES:
        a = line(s)
        st = replay_hand(T, 100, {"preflop": a}, 1)[0]
        # everyone else folds before the opener acts on the 3bet
        while st.to_act != o:
            a.append({"pos": st.to_act, "action": "fold"})
            st = replay_hand(T, 100, {"preflop": a}, 1)[0]
        R = pf.replay_ranges(a, T)
        ch = pf.strategy_for(o, a, T)
        a4 = a + [{"pos": o, "action": "raise", "to": four}]
        cht, wt = pf.strategy_for(t, a4, T), pf.replay_ranges(a4, T)[t]
        sh = lambda w, f: float((w * f).sum() / w.sum())
        target = [sh(R[o], ch["call"]), sh(R[o], ch["raise"]), sh(wt, cht["call"]), sh(wt, cht["raise"])]
        post = pf.postflop_order(list(T))
        spot = P.Spot(st.pot_start + sum(st.bets.values()), st.bets[o], st.bets[t], four, 100.0,
                      post.index(o) > post.index(t))
        out.append((f"{o} vs {t} 3bet", to_classes(R[o]), to_classes(R[t]), spot, target))
    return out


def score(cases, iters=600, verbose=False):
    err = 0.0
    for name, wo, wt, spot, target in cases:
        S = P.solve(wo, wt, spot, iters=iters)
        sh = lambda w, f: float((w * f).sum() / w.sum())
        got = [sh(wo, S["o_call"]), sh(wo, S["o_4bet"]), sh(wt, S["t_call"]), sh(wt, S["t_jam"])]
        err += sum((x - y) ** 2 for x, y in zip(got, target))
        if verbose:
            print(f"{name}: opener call {got[0]:.2f} 4bet {got[1]:.2f} (chart {target[0]:.2f} {target[1]:.2f}); "
                  f"3bettor vs 4bet call {got[2]:.2f} jam {got[3]:.2f} (chart {target[2]:.2f} {target[3]:.2f}); "
                  f"exploitability {S['exploitability']:.3f}% pot")
            for key, w in (("o_4bet", wo), ("t_jam", wt)):
                top = np.argsort(-(w * S[key]))[:12]
                print(f"  {key}: " + ", ".join(f"{CLASSES[i]}:{S[key][i]:.2f}" for i in top if w[i] * S[key][i] > 0.05))
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    cases = prepare()
    if not args.tune:
        print(f"score {score(cases, 1500, verbose=True):.3f}")
        return
    res = []
    for lo3, hi3, lo4, hi4, r4o in itertools.product([0.8, 1.2, 2.0], [0.0, 0.3, 0.8], [1.5, 2.5, 3.5],
                                                     [0.0, 0.3], [0.8, 0.9, 1.0]):
        P.REALIZE_SLOPE = {"3bet": (lo3, hi3), "4bet": (lo4, hi4)}
        P.REALIZE["4bet"]["oop"] = r4o
        res.append((score(cases), (lo3, hi3, lo4, hi4, r4o)))
    res.sort()
    for e, p in res[:8]:
        print(f"{e:.3f}  3bet slope {p[0]}/{p[1]}  4bet slope {p[2]}/{p[3]}  4bet oop {p[4]}")


if __name__ == "__main__":
    main()
