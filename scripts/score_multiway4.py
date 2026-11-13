"""4-way turns: the betting rules (Advisor._multiway, the Monte Carlo heuristic 4+ way flops and turns use) scored
against reference solves (gto/multiway_solver.py, pairwise N-player, the river played: depth 2), on a 4-way single-raised
pot (HJ opens, CO, BTN and BB call; 6-max, 100bb; the flop checked through) and four lines: first to act, checked to
last, facing a bet, facing a bet and a call. The references let only the advised player raise (root_raise; raises for
everyone made the tree too big to settle), so they overvalue hero's raises.

  python scripts/score_multiway4.py --build Kh7d2c9s          # a board's references -> reports/multiway4/ref4_<board>.json
  python scripts/score_multiway4.py --score                   # per line: the rules' best-action share and EV given up
  python scripts/score_multiway4.py --score --set ROOT_RAISE_EQ=0.6 --set CHECKED_RAISE_SHARE=0.1   # a constant changed
  python scripts/score_multiway4.py --build Kh7d2c9s --pres limp5 --rivers 8   # 5-way (limped) -> ref5_<board>.json
  python scripts/score_multiway4.py --score --glob 'ref5_*' --solver fast      # the advisor's bucket solve, not the rules

Scoring draws 40 hands from hero's range by weight per spot (a fixed draw) and plays the rules on each at 400 trials:
"mix loss" is the reference's EV of its own mix less that of the rules' pick, in % of the pot. The boards used for
the README: Kh7d2c9s, AcKd5c5s, Jh9h4c8s, Qd6c5sAh, Ts9s6d2c, 8s8d3c5h (~10-25 min each to build).
"""
import sys, os, json, time, argparse, glob
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import compare_multiway as CM
import gto.advisor as A
from gto.cards import cards
from gto.multiway_solver import MWConfig, MultiwaySolver

HERE = os.path.join(ROOT, "reports", "multiway4")
PRES = {
    "srp4": [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "call"},
             {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}],
    "limp5": [{"pos": "UTG", "action": "call"}, {"pos": "HJ", "action": "call"}, {"pos": "CO", "action": "fold"},
              {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "check"}],
}
ORDER = {"srp4": ["BB", "HJ", "CO", "BTN"], "limp5": ["SB", "BB", "UTG", "HJ", "BTN"]}
X = lambda ps: [{"pos": p, "action": "check"} for p in ps]
def lines_for(pre):
    o = ORDER[pre]
    bet = 3.5 if pre == "srp4" else 2.5
    return {"first": (o[0], []),
            "last checked to": (o[-1], X(o[:-1])),
            "vs bet": (o[2], X(o[:1]) + [{"pos": o[1], "action": "bet", "amount": bet}]),
            "vs bet+call": (o[3], X(o[:1]) + [{"pos": o[1], "action": "bet", "amount": bet}, {"pos": o[2], "action": "call"}])}
def mk_state(pre):
    def st(board, hero, hole, acts):
        s = {"players": 6, "hero": hero, "hole": hole, "board": board, "stack": 100, "preflop": PRES[pre]}
        names = ["flop", "turn", "river"][: len(board) // 2 - 2]
        for n in names[:-1]:
            s[n] = X(ORDER[pre])
        s[names[-1]] = acts
        return s
    return st
BET = {"flop": 0.33, "turn": 0.5, "river": 0.5}
def cfg(**kw):
    return MWConfig(bet=BET, max_raises=1, root_raise=True, **kw)

def solve_adv(speed):
    """The 4-way (srp4) or 5-way (limp5) bucket solve the advisor runs on a turn, at a speed's game, capped by iterations
    (not time, so the score doesn't move with the machine's load)."""
    r = A.Advisor(time_budget=1e9); r.mw_solve_streets = ("flop", "turn"); r.mw_max_iters = 200
    fast = speed == "fast"
    r.mw4_street_config = {k: dict(v) for k, v in (A.MW4_FAST_STREETS if fast else A.MW4_STREETS).items()}
    r.mw5_street_config = {k: dict(v) for k, v in (A.MW5_FAST_STREETS if fast else A.MW5_STREETS).items()}
    return r

def rules_adv(trials):
    r = A.Advisor(time_budget=1e9); r.mw_solve_streets = (); r.mw_nway_max = 0; r.mw_trials = trials
    return r

def build(a):
    os.makedirs(HERE, exist_ok=True)
    for pre in a.pres.split(","):
        fn = os.path.join(HERE, f"ref{5 if pre == 'limp5' else 4}_{a.build}.json")
        out = json.load(open(fn)) if os.path.exists(fn) else []
        mk = mk_state(pre); L = lines_for(pre)
        for ln in a.lines.split(","):
            hero, acts = L[ln]
            t0 = time.time()
            _, _, b, ranges, remaining, cur, stk = CM._spot_args(rules_adv(10), mk(a.build, hero, "2s2h", acts))
            players = [p for p in cur.order if p in remaining]
            hp = players.index(hero)
            rs = [ranges[p] for p in players]
            S = min(stk[p] - cur.inv_start[p] for p in players)
            pending = {i for i, p in enumerate(players) if p not in cur.acted} | {hp}
            depth = 2 if len(a.build) == 8 else 1
            kw = dict(bets=[cur.bets[p] for p in players], to_act=hp, n_raises=cur.n_raises, pending=pending)
            ref = MultiwaySolver(b, rs, cur.pot_start, S, cfg=cfg(depth=depth, rivers=a.rivers or None), **kw).run(iters=a.ref_iters)
            dealt = MultiwaySolver(b, rs, cur.pot_start, S, cfg=cfg(), **kw).run(iters=a.ref_iters)
            ev, evd = ref.action_evs(), dealt.action_evs()
            names = list(ref.root.names)
            hands = {}
            for h in CM.HANDS:
                hole = cards(h)
                if set(hole) & set(b):
                    continue
                ci = ref.combo_index(hole)
                if ci is None or ref.reach0[hp, 0, ci] <= 0:
                    continue
                hands[h] = {"ref": [float(ev[i, ci]) for i in range(len(names))],
                            "ref_mix": [float(x) for x in ref.hand_strategy(ci, hp)],
                            "dealt": [float(evd[i, ci]) for i in range(len(names))],
                            "dealt_mix": [float(x) for x in dealt.hand_strategy(ci, hp)]}
            # every combo in hero's range: its EVs and mixes, and its weight (scoring samples them by weight)
            from gto.cards import ALL_COMBOS, cards_str
            allc = {}
            for ci in np.nonzero(ref.reach0[hp, 0] > 0)[0]:
                gi = int(ref.hands.active[ci])
                allc[cards_str(ALL_COMBOS[gi])] = {"w": float(ref.reach0[hp, 0, ci]),
                    "ref": [float(ev[i, ci]) for i in range(len(names))], "ref_mix": [float(x) for x in ref.hand_strategy(ci, hp)],
                    "dealt": [float(evd[i, ci]) for i in range(len(names))], "dealt_mix": [float(x) for x in dealt.hand_strategy(ci, hp)]}
            g = ref.br_gains() / ref.root_pot() * 100
            rec = {"pre": pre, "board": a.build, "line": ln, "hero": hero, "acts": acts, "names": names,
                   "amounts": [x or 0 for x in ref.root.amounts], "pot": float(ref.root_pot()), "players": players,
                   "ref_gains": [float(x) for x in g], "ref_bet": 1 - ref.range_freqs().get("check", 1 - 0),
                   "dealt_bet": 1 - dealt.range_freqs().get("check", 1), "hands": hands, "all": allc, "secs": time.time() - t0}
            out = [o for o in out if (o.get("pre", "srp4"), o["line"]) != (pre, ln)] + [rec]
            print(f"{pre} {a.build} {ln}: {len(hands)} hands, gains {np.round(g, 2)}, {time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(fn, "w"))

def kind(n): return CM.kind(n)

def score_spot(args):
    rec, sets, trials, sample, solver = args
    for k, v in sets.items():
        setattr(A, k, v)
    rules = solve_adv(solver) if solver else rules_adv(trials)
    mk = mk_state(rec["pre"])
    names = rec["names"]
    res = []
    hands = rec["hands"]
    if sample and rec.get("all"):
        # hands drawn from hero's range by weight (a fixed draw per spot): the mean is the range's
        ks = sorted(rec["all"])
        w = np.array([rec["all"][k]["w"] for k in ks])
        rs = np.random.default_rng(__import__("zlib").crc32((rec["board"] + rec["line"]).encode()))
        pick_ = rs.choice(len(ks), size=min(sample, len(ks)), replace=False, p=w / w.sum())
        hands = {ks[i]: rec["all"][ks[i]] for i in sorted(pick_)}
    for h, d in hands.items():
        r = rules.advise(mk(rec["board"], rec["hero"], h, rec["acts"]))
        if solver and not r.get("solved"):
            raise RuntimeError(f"{rec['board']} {rec['line']} {h}: not solved: {r['notes']}")
        er = dict(zip(names, d["ref"]))
        best = max(er, key=er.get)
        top = max(r["actions"], key=lambda x: x["freq"])
        same = [i for i, n in enumerate(names) if kind(n) == kind(top["action"])]
        pick = names[min(same, key=lambda i: abs(rec["amounts"][i] - (top.get("amount_bb") or 0)))] if same else best
        dpick = names[int(np.argmax(d["dealt_mix"]))]
        # the rules' mix played against the reference's EVs (by kind)
        mix_ev = 0.0
        for x in r["actions"]:
            s = [i for i, n in enumerate(names) if kind(n) == kind(x["action"])]
            if s:
                i = min(s, key=lambda i: abs(rec["amounts"][i] - (x.get("amount_bb") or 0)))
                mix_ev += x["freq"] * d["ref"][i]
        ref_mix_ev = sum(f * e for f, e in zip(d["ref_mix"], d["ref"]))
        res.append({"h": h, "best": best, "pick": pick, "loss": er[best] - er[pick], "dpick": dpick, "dloss": er[best] - er[dpick],
                    "mixloss": ref_mix_ev - mix_ev, "pot": rec["pot"],
                    "rules_bet": sum(x["freq"] for x in r["actions"] if kind(x["action"]) in ("bet", "raise")),
                    "ref_bet": sum(f for f, n in zip(d["ref_mix"], names) if kind(n) in ("bet", "raise"))})
    return rec["pre"], rec["board"], rec["line"], res

def score(a):
    from multiprocessing import Pool
    recs = [r for f in sorted(glob.glob(os.path.join(HERE, a.glob))) for r in json.load(open(f))]
    sets = {}
    for s in a.set or []:
        k, v = s.split("="); sets[k] = float(v)
    with Pool(a.procs) as pool:
        outs = pool.map(score_spot, [(r, sets, a.trials, a.sample, a.solver) for r in recs])
    allr = []
    by = {}
    for pre, b, ln, res in outs:
        allr += res
        by.setdefault(ln, []).extend(res)
        if a.verbose:
            print(f"{pre} {b} {ln}: rules best {np.mean([x['pick'] == x['best'] for x in res])*100:.0f}% loss {np.mean([x['loss']/x['pot'] for x in res])*100:.2f}% pot "
                  f"(mix {np.mean([x['mixloss']/x['pot'] for x in res])*100:.2f}), dealt {np.mean([x['dloss']/x['pot'] for x in res])*100:.2f}%")
    def line(res, label):
        return (f"{label:16s} n={len(res):3d}  rules: best {np.mean([x['pick'] == x['best'] for x in res])*100:3.0f}%, "
                f"pick loss {np.mean([x['loss']/x['pot'] for x in res])*100:5.2f}% pot, mix loss {np.mean([x['mixloss']/x['pot'] for x in res])*100:5.2f}%, "
                f"bets {np.mean([x['rules_bet'] for x in res])*100:4.0f}% (ref {np.mean([x['ref_bet'] for x in res])*100:4.0f}%) | "
                f"dealt solve: best {np.mean([x['dpick'] == x['best'] for x in res])*100:3.0f}%, loss {np.mean([x['dloss']/x['pot'] for x in res])*100:5.2f}%")
    for ln, res in by.items():
        print(line(res, ln))
    print(line(allr, "ALL " + " ".join(a.set or [])))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build"); ap.add_argument("--pres", default="srp4")
    ap.add_argument("--lines", default="first,last checked to,vs bet,vs bet+call")
    ap.add_argument("--ref-iters", type=int, default=120); ap.add_argument("--rivers", type=int, default=0)
    ap.add_argument("--score", action="store_true"); ap.add_argument("--set", action="append")
    ap.add_argument("--glob", default="ref4_*.json"); ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--solver", choices=["fast", "accurate"], help="score the bucket solve at this speed's game, not the rules")
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--sample", type=int, default=40); ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    build(a) if a.build else score(a)
