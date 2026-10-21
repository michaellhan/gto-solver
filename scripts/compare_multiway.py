"""The 3-way solver (gto/multiway_solver.py) against the betting rules (Advisor._multiway) and against facts about
multiway play, on 3-way pots: CO opens, BTN and BB call (6-max, 100bb), earlier streets checked through.

  python scripts/compare_multiway.py --spots          # per hand: the rules' pick and EVs against the solve's
  python scripts/compare_multiway.py --facts          # range-level facts (3-way vs heads-up, bluffs, shared defence)
  python scripts/compare_multiway.py --flop-study     # the opener's flop c-bet into two callers (GTO Wizard's 3-way study)
  python scripts/compare_multiway.py --turn-reference # turn: the solve (river dealt out) and the rules against a solve betting the river

Every solve runs a fixed number of iterations (--iters), so the numbers don't depend on machine load.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gto.advisor as A
from gto import preflop as pf
from gto.cards import cards
from gto.evaluator import eval_board
from gto.multiway_solver import MultiwaySolver

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
       {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
CHECKS = [{"pos": p, "action": "check"} for p in ("BB", "CO", "BTN")]
RIVERS = ["Kh7d2c9s4h", "Jh9h4c8s2d", "8s8d3c5h5d", "Qd6c5sAh2h"]
TURNS = ["Kh7d2c9s", "Jh9h4c8s"]
LINES = {   # name: (hero, this street's actions so far); bets are half the pot (the solver's size)
    "BB first": ("BB", []),
    "BTN checked to": ("BTN", [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "check"}]),
    "BTN vs bet": ("BTN", [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 4.0}]),
    "BB vs bet and call": ("BB", [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 4.0},
                                  {"pos": "BTN", "action": "call"}]),
}
HANDS = ["AsAd", "KsKd", "QsQd", "JsJd", "TsTd", "9c9d", "7s7c", "5c5s", "AsKs", "AdQd", "AcJc", "KsQs", "KdJd",
         "QsJs", "JsTs", "Ts9s", "9d8d", "8c7c", "7d6d", "6s5s", "As5s", "KcTd", "Qc8c", "Jc7c", "AdTc", "3c3d"]


def kind(name):
    for k in ("bet", "raise", "all-in", "check", "call", "fold"):
        if name.startswith(k):
            return "raise" if k == "all-in" else k
    return name


def state(board, hero, hole, acts):
    st = {"players": 6, "hero": hero, "hole": hole, "board": board, "stack": 100, "preflop": PRE}
    names = ["flop", "turn", "river"][: len(board) // 2 - 2]
    for s in names[:-1]:
        st[s] = CHECKS
    st[names[-1]] = acts
    return st


def spots(args):
    sol = A.Advisor(time_budget=1e9)
    sol.mw_max_iters = args.iters
    rules = A.Advisor(time_budget=1e9)
    rules.mw_solve_streets = ()
    rules.mw_trials = args.trials
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    rows = []
    for board in RIVERS + TURNS:
        bset = set(cards(board))
        for line, (hero, acts) in LINES.items():
            for h in HANDS:
                if set(cards(h)) & bset:
                    continue
                key = f"{board}|{line}|{h}|{args.iters}"
                if key not in cache:
                    st = state(board, hero, h, acts)
                    a, b = sol.advise(st), rules.advise(st)
                    if "error" in a or "error" in b or not a.get("solved") \
                            or any("near-zero-weight" in n for n in a["notes"]):
                        cache[key] = None
                    else:
                        cache[key] = {"solver": a["actions"], "rules": b["actions"], "pot": a["pot_bb"],
                                      "expl": a["solver"]["exploitability_pct"]}
                    json.dump(cache, open(args.cache, "w"))
                r = cache[key]
                if r:
                    rows.append((board, line, h, r))
    # per spot: does the rules' pick match the solve's most frequent action; what the pick gives up by the
    # solve's EVs; how far the two EVs are apart for the same kind of action
    by = {}
    for board, line, h, r in rows:
        s = {kind(a["action"]): a for a in r["solver"]}
        s_best = max(r["solver"], key=lambda a: a["freq"])
        s_top = max(a["ev_bb"] for a in r["solver"])
        pick = kind(max(r["rules"], key=lambda a: a["freq"])["action"])
        loss = s_top - s[pick]["ev_bb"] if pick in s else None
        diffs = [abs(a["ev_bb"] - s[kind(a["action"])]["ev_bb"]) for a in r["rules"] if kind(a["action"]) in s
                 and kind(a["action"]) in ("check", "call", "fold")]
        st = "river" if len(board) == 10 else "turn"
        d = by.setdefault((st, line), {"n": 0, "agree": 0, "loss": [], "evd": [], "pot": [], "aggr_s": 0.0, "aggr_r": 0,
                                       "expl": []})
        d["n"] += 1
        d["agree"] += kind(s_best["action"]) == pick
        if loss is not None:
            d["loss"].append(loss)
        d["evd"] += diffs
        d["pot"].append(r["pot"])
        d["aggr_s"] += sum(a["freq"] for a in r["solver"] if kind(a["action"]) in ("bet", "raise"))
        d["aggr_r"] += pick in ("bet", "raise")
        d["expl"].append(r["expl"])
    print(f"{'street':6s} {'line':22s} {'hands':>5s} {'same pick':>9s} {'rules give up':>13s} {'EV diff':>8s} "
          f"{'aggr solve':>10s} {'aggr rules':>10s} {'expl':>5s}")
    for (st, line), d in sorted(by.items()):
        pot = np.mean(d["pot"])
        print(f"{st:6s} {line:22s} {d['n']:5d} {d['agree'] / d['n'] * 100:8.0f}% "
              f"{np.mean(d['loss']):7.2f} bb {np.mean(d['loss']) / pot * 100:3.0f}% "
              f"{np.mean(d['evd']) if d['evd'] else float('nan'):7.2f} {d['aggr_s'] / d['n'] * 100:9.0f}% "
              f"{d['aggr_r'] / d['n'] * 100:9.0f}% {np.mean(d['expl']):5.2f}")
    tot = sum(d["n"] for d in by.values())
    print(f"all: {tot} hands, same pick {sum(d['agree'] for d in by.values()) / tot * 100:.0f}%, the rules' pick gives up "
          f"{np.mean([x for d in by.values() for x in d['loss']]):.2f} bb by the solve's EVs")


def facts(args):
    R = A.preflop_ranges(PRE, pf.Table(pf.SEATS[6]), 100.0)
    rs = [R["BB"], R["CO"], R["BTN"]]
    for board in RIVERS:
        b = cards(board)
        st = eval_board(b)
        s3 = MultiwaySolver(b, rs, 8.0, 97.5).run(iters=args.iters)
        s2 = MultiwaySolver(b, [R["BB"], R["CO"]], 5.5, 97.5).run(iters=args.iters)
        # checked to the button, 3-way, and the CO checked to heads-up (both last to act)
        b3 = MultiwaySolver(b, rs, 8.0, 97.5, to_act=2, pending={2}).run(iters=args.iters)
        b2 = MultiwaySolver(b, [R["BB"], R["CO"]], 5.5, 97.5, to_act=1, pending={1}).run(iters=args.iters)

        def bets(s, p):
            f = s.reach0[p, 0]
            sg = s.root.avg_strategy()[0]
            agg = sum(sg[a] for a, n in enumerate(s.root.names) if n != "check")
            sc = st[s.hands.active]
            live = f > 0
            weak = live & (sc <= np.quantile(sc[live], 0.4))
            top = live & (sc >= np.quantile(sc[live], 0.9))
            tot = float((agg * f).sum() / f.sum())
            return tot, float((agg * f)[weak].sum() / max((agg * f).sum(), 1e-12)), float((agg * f)[top].sum() / f[top].sum())
        # facing the CO's bet after the BB checked: each defender's continue share, and how often someone continues
        node = b3.root
        co = MultiwaySolver(b, rs, 8.0, 97.5, to_act=1, pending={1, 2}).run(iters=args.iters)
        bet_i = co.root.names.index("bet 50%")
        after = co.root.children[bet_i]
        btn = after
        f_btn = co.reach0[2, 0]
        sg = btn.avg_strategy()[0]
        btn_fold = float((sg[0] * f_btn).sum() / f_btn.sum())
        bbn = btn.children[0]          # the button folds, the big blind faces the bet alone
        f_bb = co.reach0[0, 0]
        bb_fold = float((bbn.avg_strategy()[0][0] * f_bb).sum() / f_bb.sum())
        out = {"board": board,
               "BB lead 3-way / heads-up": [round(bets(s3, 0)[0], 3), round(bets(s2, 0)[0], 3)],
               "last to act bets 3-way / heads-up": [round(bets(b3, 2)[0], 3), round(bets(b2, 1)[0], 3)],
               "bluff share of those bets (weakest 40% of range) 3-way / heads-up": [round(bets(b3, 2)[1], 3), round(bets(b2, 1)[1], 3)],
               "top 10% bet 3-way / heads-up": [round(bets(b3, 2)[2], 3), round(bets(b2, 1)[2], 3)],
               "vs CO half-pot bet: BTN folds, BB folds after, both fold (MDF 0.333)":
                   [round(btn_fold, 3), round(bb_fold, 3), round(btn_fold * bb_fold, 3)],
               "nash conv % pot (3-way first, checked to, bet node)": [round(s3.nash_conv(), 2), round(b3.nash_conv(), 2), round(co.nash_conv(), 2)]}
        print(json.dumps(out), flush=True)


def flop_study(args):
    """GTO Wizard's 3-way flop study (README: LJ opens, both blinds call, checked to the opener: checks 34%, bets a third
    56%, bigger 9%; heads-up against the big blind 23/38/39): our one-size (33%) tree, 6-max UTG opening."""
    pre = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
           {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}]
    R = A.preflop_ranges(pre, pf.Table(pf.SEATS[6]), 100.0)
    pre2 = [dict(a, action="fold") if a["pos"] == "SB" else a for a in pre]
    R2 = A.preflop_ranges(pre2, pf.Table(pf.SEATS[6]), 100.0)
    rng = np.random.default_rng(args.seed)
    flops = []
    while len(flops) < args.flops:
        f = tuple(int(x) for x in rng.choice(52, 3, replace=False))
        flops.append(f)
    tot3, tot2 = [], []
    for f in flops:
        s3 = MultiwaySolver(f, [R["SB"], R["BB"], R["UTG"]], 7.5, 97.5, to_act=2, pending={2}).run(iters=args.iters)
        s2 = MultiwaySolver(f, [R2["BB"], R2["UTG"]], 5.5, 97.5, to_act=1, pending={1}).run(iters=args.iters)
        b3 = 1 - s3.range_freqs()["check"]
        b2 = 1 - s2.range_freqs()["check"]
        tot3.append(b3); tot2.append(b2)
        from gto.cards import cards_str
        print(f"{cards_str(f)}: opener bets {b3 * 100:.0f}% into two (heads-up {b2 * 100:.0f}%), nash conv "
              f"{s3.nash_conv():.2f}% / {s2.nash_conv():.2f}%", flush=True)
    print(f"mean over {len(flops)} flops: 3-way bets {np.mean(tot3) * 100:.0f}%, heads-up {np.mean(tot2) * 100:.0f}% "
          f"(GTO Wizard's study: 66% and 77%, several sizes)")


def _spot_args(adv, st):
    """What the advisor hands its multiway code for this state: (hero, hole, board, ranges, remaining, cur, stacks)."""
    got = {}

    def grab(self, hero, hole, board, ranges, remaining, cur, stk, t0, notes):
        got["a"] = (hero, hole, board, dict(ranges), list(remaining), cur, stk)
        return {"stub": True}
    saved = A.Advisor._multiway_solve
    A.Advisor._multiway_solve = grab
    try:
        adv.advise(st)
    finally:
        A.Advisor._multiway_solve = saved
    return got["a"]


def turn_reference(args):
    """On the turn the solve deals the river out (no river betting). Against a deeper solve that bets the river too
    (depth 2, --ref-rivers sampled rivers), who is closer per hand: the depth-1 solve or the betting rules?"""
    from gto.cards import COMBO_INDEX
    from gto.multiway_solver import MWConfig
    adv = A.Advisor(time_budget=1e9)
    adv.mw_max_iters = args.iters
    rules = A.Advisor(time_budget=1e9)
    rules.mw_solve_streets = ()
    rules.mw_trials = args.trials
    tot = {"d1": [], "rules": [], "d1_pick": 0, "rules_pick": 0, "n": 0, "evd_d1": [], "evd_rules": []}
    for board in TURNS:
        for line in ("BB first", "BTN checked to"):
            hero, acts = LINES[line]
            hero, _, b, ranges, remaining, cur, stk = _spot_args(adv, state(board, hero, "2s2h", acts))
            players = [p for p in cur.order if p in remaining]
            hp = players.index(hero)
            rs = [ranges[p] for p in players]
            S = min(stk[p] - cur.inv_start[p] for p in players)
            pending = {i for i, p in enumerate(players) if p not in cur.acted} | {hp}
            ref = MultiwaySolver(b, rs, cur.pot_start, S, to_act=hp, pending=pending,
                                 cfg=MWConfig(depth=2, rivers=args.ref_rivers)).run(iters=args.ref_iters)
            d1 = MultiwaySolver(b, rs, cur.pot_start, S, to_act=hp, pending=pending).run(iters=args.iters)
            ev_ref, ev_d1 = ref.action_evs(), d1.action_evs()
            names = ref.root.names
            print(f"{board} {line}: reference nash conv {ref.nash_conv():.2f}% pot, depth-1 {d1.nash_conv():.2f}%; "
                  f"range bets {1 - ref.range_freqs()['check']:.3f} (reference) {1 - d1.range_freqs()['check']:.3f} (depth 1)",
                  flush=True)
            bset = set(b)
            for h in HANDS:
                hole = cards(h)
                if set(hole) & bset:
                    continue
                ci = ref.combo_index(hole)
                if ci is None or ref.reach0[hp, 0, ci] <= 0:
                    continue
                st = state(board, hero, h, acts)
                r = rules.advise(st)
                rk = {kind(a["action"]): a for a in r["actions"]}
                er = {kind(n): float(ev_ref[a, ci]) for a, n in enumerate(names)}
                e1 = {kind(n): float(ev_d1[a, ci]) for a, n in enumerate(names)}
                best = max(er, key=er.get)
                p1 = kind(names[int(np.argmax(d1.hand_strategy(ci, hp)))])
                pr = kind(max(r["actions"], key=lambda a: a["freq"])["action"])
                tot["n"] += 1
                tot["d1_pick"] += p1 == best
                tot["rules_pick"] += pr == best
                tot["d1"].append(er[best] - er.get(p1, er[best]))
                tot["rules"].append(er[best] - er.get(pr, er[best]))
                for k in er:
                    tot["evd_d1"].append(abs(e1[k] - er[k]))
                    if k in rk:
                        tot["evd_rules"].append(abs(rk[k]["ev_bb"] - er[k]))
    n = tot["n"]
    print(f"{n} hands: best action by the reference picked {tot['d1_pick'] / n * 100:.0f}% by the depth-1 solve, "
          f"{tot['rules_pick'] / n * 100:.0f}% by the rules; reference EV given up {np.mean(tot['d1']):.2f} bb against "
          f"{np.mean(tot['rules']):.2f} bb; EVs off by {np.mean(tot['evd_d1']):.2f} bb against {np.mean(tot['evd_rules']):.2f} bb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", action="store_true")
    ap.add_argument("--turn-reference", action="store_true")
    ap.add_argument("--ref-rivers", type=int, default=12)
    ap.add_argument("--ref-iters", type=int, default=60)
    ap.add_argument("--facts", action="store_true")
    ap.add_argument("--flop-study", action="store_true")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--flops", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "compare_multiway.json"))
    a = ap.parse_args()
    if a.spots:
        spots(a)
    if a.facts:
        facts(a)
    if a.flop_study:
        flop_study(a)
    if a.turn_reference:
        turn_reference(a)


if __name__ == "__main__":
    main()
