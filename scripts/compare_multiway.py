"""The 3-way solver (gto/multiway_solver.py) against the betting rules (Advisor._multiway) and against facts about
multiway play, on 3-way pots: CO opens, BTN and BB call (6-max, 100bb), earlier streets checked through.

  python scripts/compare_multiway.py --spots          # per hand: the rules' pick and EVs against the solve's
  python scripts/compare_multiway.py --facts          # range-level facts (3-way vs heads-up, bluffs, shared defence)
  python scripts/compare_multiway.py --flop-study     # the opener's flop c-bet into two callers (GTO Wizard's 3-way study)
      --rules-checks   # ... the blinds' checks narrowed by the betting rules first (what the advice plays)
      --full-street    # ... solved from the SB's action, read where both blinds checked (and the rules' check ranges against it)
      --raise-frac 33  # ... raises of a third of the pot (the study's)
  python scripts/compare_multiway.py --turn-reference # turn: the solve (river dealt out) and the rules against a solve betting the river
  python scripts/compare_multiway.py --narrowing      # the turn's ranges after a 3-way flop: rules vs through the flop solve

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


FLOP_STUDY = {"3-way": 0.659, "heads-up": 0.770}   # GTO Wizard's 3-way flop study: the opener's bets when checked to


def STUDY_RANGES():
    """Ranges shaped like the study's (its text: the SB calls with low to middle pairs, suited connectors and suited
    Ax / Kx; with the SB in, the BB 3bets TT-JJ and suited broadways and calls a capped range; an 8-max LJ open):
    (3-way {SB, BB, UTG}, heads-up {BB, UTG}), the opener under UTG's name."""
    from gto.ranges import parse_range
    lj = parse_range("22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,65s,ATo+,KJo+,QJo")
    sb = parse_range("22-99,A2s-A9s,K9s-KTs,T9s,98s,87s,76s,65s,54s")
    bb3 = parse_range("22-99,A2s-A9s,K2s-K9s,Q6s-Q9s,J7s-J9s,T7s-T9s,96s+,85s+,75s+,64s+,53s+,43s,A9o-AJo,KTo-KQo,QTo+,JTo")
    bb2 = bb3 + parse_range("TT-JJ,ATs-AJs,KTs-KQs,QTs-QJs,JTs")
    return {"UTG": lj, "SB": sb, "BB": bb3}, {"UTG": lj, "BB": bb2}


def _cfg(args):
    from gto.multiway_solver import MWConfig
    bet = {"flop": [float(x) / 100 for x in args.flop_sizes.split(",")], "turn": 0.5, "river": 0.5}
    kw = {} if args.raise_frac is None else {"raise_frac": args.raise_frac / 100}
    if args.later == "deal":
        return MWConfig(bet=bet, **kw)
    return MWConfig(later="buckets", buckets=args.buckets, bucket_river_step=args.step, bet=bet, **kw)


def flop_study(args):
    """GTO Wizard's 3-way flop study ("Playing In Position Against Two Callers", 100bb 8-max, LJ opens, SB and BB call,
    sizes 33/66/100%, checked to the opener: checks 34.1%, bets 65.9%; heads-up against the BB checks 23.0%, bets
    77.0%): our one-size (33%) tree, 6-max UTG opening, on --flops random flops. --later deal: the turn and river dealt
    out (equity leaves); buckets: bet by --buckets hand buckets."""
    from gto.cards import cards_str
    pre = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
           {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}]
    R = A.preflop_ranges(pre, pf.Table(pf.SEATS[6]), 100.0)
    pre2 = [dict(a, action="fold") if a["pos"] == "SB" else a for a in pre]
    R2 = A.preflop_ranges(pre2, pf.Table(pf.SEATS[6]), 100.0)
    if args.study_ranges:
        R, R2 = STUDY_RANGES()
    R0, R20 = R, R2
    rng = np.random.default_rng(args.seed)
    flops = []
    while len(flops) < args.flops:
        f = tuple(int(x) for x in rng.choice(52, 3, replace=False))
        flops.append(f)
    cfg = _cfg(args)
    tot3, tot2, sz3, sz2 = [], [], [], []
    def reach_at(s, target, i):
        """Player i's strategy product down to `target` along checks (the full-street solve's root to the opener)."""
        node, out = s.root, np.ones(s.hands.n)
        while node is not target:
            k = node.names.index("check")
            if node.player == i:
                out = out * node.avg_strategy()[0][k]
            node = node.children[k]
        return out

    def checked_to(s, n):
        """(--full-street) The opener's node after the blinds' checks in a solve from the first blind's action, and the
        share of the time the blinds check through to it."""
        node, ch = s.root, 1.0
        for _ in range(n - 1):
            i = node.names.index("check")
            r = s.reach0[node.player, 0]
            ch *= float(node.avg_strategy()[0][i] @ r / r.sum())
            node = node.children[i]
        return node, ch

    def rules_checks(f, rs, pre_, who):
        """The ranges after the blinds' checks narrowed by the betting rules, as the advisor narrows them in front of the
        opener's solve when it has no solve of the blinds' actions."""
        out = dict(rs)
        st = A.replay_hand(pf.Table(pf.SEATS[6]), 100.0, {"preflop": pre_, "flop": [{"pos": p, "action": "check"} for p in who]},
                           2)[1]
        A._narrow_street(out, f, st, 100.0)
        return out

    def tv(a, b):
        return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())

    tvs = []
    for f in flops:
        if args.rules_checks:
            R = rules_checks(f, R0, pre, ("SB", "BB"))
            R2 = rules_checks(f, R20, pre2, ("BB",))
        if args.full_street:
            s3 = MultiwaySolver(f, [R["SB"], R["BB"], R["UTG"]], 7.5, 97.5, cfg=cfg).run(iters=args.iters)
            node3, chk3 = checked_to(s3, 3)
            fr3 = s3.range_freqs(node3, s3.reach0[2, 0])
        else:
            s3 = MultiwaySolver(f, [R["SB"], R["BB"], R["UTG"]], 7.5, 97.5, to_act=2, pending={2}, cfg=cfg).run(iters=args.iters)
            fr3 = s3.range_freqs()
        b3 = 1 - fr3["check"]
        tot3.append(b3)
        sz3.append(fr3)
        line = f"{cards_str(f)}: opener bets {b3 * 100:.1f}% into two"
        if args.full_street:
            line += f" (the blinds check {chk3 * 100:.0f}%)"
            # the blinds' ranges where they checked to the opener, against what the betting rules leave them
            ru = rules_checks(f, R0, pre, ("SB", "BB"))
            d = [tv(s3.expand(s3.reach0[i, 0] * reach_at(s3, node3, i)), ru[p]) for i, p in ((0, "SB"), (1, "BB"))]
            tvs.append(d)
            line += f" [the rules' check ranges off by SB {d[0]:.2f}, BB {d[1]:.2f}]"
        if not args.three_only:
            if args.full_street:
                s2 = MultiwaySolver(f, [R2["BB"], R2["UTG"]], 5.5, 97.5, cfg=cfg).run(iters=args.iters)
                node2, _ = checked_to(s2, 2)
                fr2 = s2.range_freqs(node2, s2.reach0[1, 0])
            else:
                s2 = MultiwaySolver(f, [R2["BB"], R2["UTG"]], 5.5, 97.5, to_act=1, pending={1}, cfg=cfg).run(iters=args.iters)
                fr2 = s2.range_freqs()
            b2 = 1 - fr2["check"]
            tot2.append(b2)
            sz2.append(fr2)
            line += f" (heads-up {b2 * 100:.1f}%)"
        print(line + f", best-response gains {np.round(s3.br_gains() / s3.root_pot() * 100, 2)} % pot", flush=True)
    print(f"mean over {len(flops)} flops: 3-way bets {np.mean(tot3) * 100:.1f}%"
          + (f", heads-up {np.mean(tot2) * 100:.1f}%" if tot2 else "")
          + f" (GTO Wizard's study: {FLOP_STUDY['3-way'] * 100:.0f}% and {FLOP_STUDY['heads-up'] * 100:.0f}%, several sizes)")
    if tvs:
        print(f"  the betting rules' check ranges against the solve's (total variation): SB {np.mean([d[0] for d in tvs]):.2f}, "
              f"BB {np.mean([d[1] for d in tvs]):.2f}")
    for nm, sz in (("3-way", sz3), ("heads-up", sz2)):
        if sz:
            print(f"  {nm} by action: " + ", ".join(f"{k} {np.mean([d[k] for d in sz]) * 100:.1f}%" for k in sz[0]))
    return np.mean(tot3), (np.mean(tot2) if tot2 else None)


def hu_check(args):
    """Two players: the 3-way solver's flop model (--later) played two-handed against the heads-up solver's (river bet
    on, one turn per rank class, per combo; README Accuracy vs speed) on the same one-size tree (flop 33%, turn and
    river 50%, raises 75%), UTG open called by the BB, from the BB's first action: the BB's bets, and UTG's when checked
    to, over --flops random flops."""
    from gto.cards import cards_str
    from gto.solver import Config, solve
    pre = [{"pos": "UTG", "action": "raise", "to": 2.5}] + [{"pos": p, "action": "fold"} for p in ("HJ", "CO", "BTN", "SB")] \
        + [{"pos": "BB", "action": "call"}]
    R = A.preflop_ranges(pre, pf.Table(pf.SEATS[6]), 100.0)
    sizes = {"flop": [0.33], "turn": [0.5], "river": [0.5]}
    hcfg = Config(bet_sizes=sizes, raise_sizes={k: [0.75] for k in sizes}, max_raises=2, allin_spr=1.5,
                  later_sizes=None)
    rng = np.random.default_rng(args.seed)
    d_oop, d_ip = [], []
    for _ in range(args.flops):
        f = tuple(int(x) for x in rng.choice(52, 3, replace=False))
        hu = solve(f, R["BB"], R["UTG"], 5.5, 97.5, cfg=hcfg, time_budget=1e9, max_iters=args.ref_iters)
        mw = MultiwaySolver(f, [R["BB"], R["UTG"]], 5.5, 97.5, cfg=_cfg(args)).run(iters=args.iters)
        r0 = np.zeros(1326); r0[mw.hands.active] = mw.reach0[0, 0]
        h_oop = 1 - float(hu.root.avg_strategy0()[0] @ hu.r0 / hu.r0.sum())
        m_oop = 1 - mw.range_freqs()["check"]
        hc, mc = hu.root.children[0], mw.root.children[0]
        # (the IP player's range after the check: its whole range, the check doesn't narrow it)
        h_ip = 1 - float(hc.avg_strategy0()[0] @ hu.r1 / hu.r1.sum())
        m_ip = 1 - mw.range_freqs(mc, mw.reach0[1, 0])["check"]
        d_oop.append(m_oop - h_oop); d_ip.append(m_ip - h_ip)
        print(f"{cards_str(f)}: BB bets {m_oop * 100:.1f}% (heads-up solver {h_oop * 100:.1f}%), UTG checked to bets "
              f"{m_ip * 100:.1f}% ({h_ip * 100:.1f}%); exploitability {hu.exploitability():.2f}% / "
              f"{mw.nash_conv() / 2:.2f}% pot", flush=True)
    print(f"mean difference (3-way solver - heads-up solver): BB {np.mean(d_oop) * 100:+.1f} points (mean abs "
          f"{np.mean(np.abs(d_oop)) * 100:.1f}), UTG {np.mean(d_ip) * 100:+.1f} (mean abs {np.mean(np.abs(d_ip)) * 100:.1f})")


def _spot_args(adv, st):
    """What the advisor hands its multiway code for this state: (hero, hole, board, ranges, remaining, cur, stacks)."""
    got = {}

    def grab(self, hero, hole, board, ranges, remaining, cur, stk, t0, notes, line=None):
        got["a"] = (hero, hole, board, dict(ranges), list(remaining), cur, stk)
        return {"stub": True}
    saved = A.Advisor._multiway_solve
    A.Advisor._multiway_solve = grab
    try:
        adv.advise(st)
    finally:
        A.Advisor._multiway_solve = saved
    return got["a"]


def _against(spots, ref_cfg, ref_iters, cands, rules):
    """Each spot (board, hero, this street's actions, state function) solved by a reference (ref_cfg, ref_iters) and by
    each candidate {name: (MWConfig, iterations)}; the betting rules too. Per hand: is the reference's best action the
    candidate's most frequent one, what that pick gives up by the reference's EVs, how far each action's EV is."""
    tot = {k: {"pick": 0, "loss": [], "evd": [], "freq": []} for k in list(cands) + ["rules"]}
    n = 0
    for board, hero, acts, mk in spots:
        hero, _, b, ranges, remaining, cur, stk = _spot_args(rules, mk(board, hero, "2s2h", acts))
        players = [p for p in cur.order if p in remaining]
        hp = players.index(hero)
        rs = [ranges[p] for p in players]
        S = min(stk[p] - cur.inv_start[p] for p in players)
        pending = {i for i, p in enumerate(players) if p not in cur.acted} | {hp}
        ref = MultiwaySolver(b, rs, cur.pot_start, S, to_act=hp, pending=pending, cfg=ref_cfg).run(iters=ref_iters)
        sols = {k: MultiwaySolver(b, rs, cur.pot_start, S, to_act=hp, pending=pending, cfg=c).run(iters=it)
                for k, (c, it) in cands.items()}
        ev_ref = ref.action_evs()
        evs = {k: s_.action_evs() for k, s_ in sols.items()}
        names = ref.root.names
        fr = 1 - ref.range_freqs()["check"]
        out = f"{board} {hero}: range bets {fr * 100:.1f}% (reference, best-response gains " \
              f"{np.round(ref.br_gains() / ref.root_pot() * 100, 2)} % pot)"
        for k, s_ in sols.items():
            f = 1 - s_.range_freqs()["check"]
            tot[k]["freq"].append(abs(f - fr))
            out += f", {k} {f * 100:.1f}%"
        print(out, flush=True)
        bset = set(b)
        for h in HANDS:
            hole = cards(h)
            if set(hole) & bset:
                continue
            ci = ref.combo_index(hole)
            if ci is None or ref.reach0[hp, 0, ci] <= 0:
                continue
            r = rules.advise(mk(board, hero, h, acts))
            er = {nm: float(ev_ref[a, ci]) for a, nm in enumerate(names)}
            best = max(er, key=er.get)
            n += 1

            def to_ref(x):
                # the rules' action as the reference tree's: its kind, the nearest size among several bets
                same = [a for a, nm in enumerate(names) if kind(nm) == kind(x["action"])]
                if not same:
                    return None
                amt = x.get("amount_bb")
                return names[min(same, key=lambda a: abs((ref.root.amounts[a] or 0) - (amt or 0)))]
            picks = {k: names[int(np.argmax(s_.hand_strategy(ci, hp)))] for k, s_ in sols.items()}
            picks["rules"] = to_ref(max(r["actions"], key=lambda x: x["freq"]))
            for k, pk in picks.items():
                tot[k]["pick"] += pk == best
                tot[k]["loss"].append(er[best] - er.get(pk, er[best]))
            for k in sols:
                tot[k]["evd"] += [abs(float(evs[k][a, ci]) - er[nm]) for a, nm in enumerate(names)]
            tot["rules"]["evd"] += [abs(x["ev_bb"] - er[to_ref(x)]) for x in r["actions"] if to_ref(x) is not None]
    print(f"{n} hands against the reference:")
    for k, d in tot.items():
        print(f"  {k:10s}: picks its best action {d['pick'] / n * 100:.0f}%, gives up {np.mean(d['loss']):.3f} bb, "
              f"EVs off {np.mean(d['evd']):.2f} bb" + (f", range bets off {np.mean(d['freq']) * 100:.1f} points"
                                                       if d["freq"] else ""), flush=True)


def _rules(args):
    rules = A.Advisor(time_budget=1e9)
    rules.mw_solve_streets = ()
    rules.mw_trials = args.trials
    return rules


def turn_reference(args):
    """Turn solves against a reference betting the river per combo (depth 2, every river with --ref-rivers 0, else that
    many sampled): the river dealt out (equity leaves), bet by --buckets hand buckets, and the betting rules."""
    from gto.multiway_solver import MWConfig
    spots = [(b, LINES[ln][0], LINES[ln][1], state) for b in TURNS for ln in ("BB first", "BTN checked to")]
    _against(spots, MWConfig(depth=2, rivers=args.ref_rivers or None), args.ref_iters,
             {"dealt": (MWConfig(), args.iters), "buckets": (MWConfig(later="buckets", buckets=args.buckets), args.iters)},
             _rules(args))


FLOP_PRE = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
            {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}]
FLOPS = ["Kh7d2c", "8h7sJs", "QdAd3s", "Tc9c7c"]


def flop_state(board, hero, hole, acts):
    return {"players": 6, "hero": hero, "hole": hole, "board": board, "stack": 100, "preflop": FLOP_PRE, "flop": acts}


def flop_reference(args):
    """3-way flops (UTG opens, SB and BB call; UTG checked to, and the SB first) against a deeper solve of the same
    model (--ref-buckets buckets, every turn card, every river, --ref-iters): --buckets buckets with one turn per rank
    class, 12 buckets, the turn and river dealt out, and the betting rules."""
    from gto.multiway_solver import MWConfig
    chk = [{"pos": "SB", "action": "check"}, {"pos": "BB", "action": "check"}]
    spots = [(b, h, a, flop_state) for b in FLOPS[:args.flops] for h, a in (("UTG", chk), ("SB", []))]
    bet = _cfg(args).bet
    later = lambda k, **kw: MWConfig(later="buckets", buckets=k, bet=bet, **kw)
    _against(spots, later(args.ref_buckets, classes=False), args.ref_iters,
             {f"buckets {args.buckets}": (later(args.buckets), args.iters), "buckets 12": (later(12), args.iters),
              "dealt": (MWConfig(bet=bet), args.iters)}, _rules(args))


def narrowing(args):
    """The turn's ranges after a 3-way flop (UTG opens, SB and BB call; the blinds check, UTG bets a third, both call),
    narrowed by the betting rules alone and through UTG's flop advice solve (--iters, and 60) as the advisor does
    (trust by settledness, smoothing), against the walk through a long solve (--ref-iters, followed fully, unsmoothed):
    per player the total variation distance between the normalized ranges, and the combos."""
    from gto.cards import cards_str
    chk = [{"pos": "SB", "action": "check"}, {"pos": "BB", "action": "check"}]
    line = chk + [{"pos": "UTG", "action": "bet", "amount": 2.5}, {"pos": "SB", "action": "call"},
                  {"pos": "BB", "action": "call"}]
    narrow = A._narrow_street
    # (the turn's own checks left out: the ranges compared are the flop's)
    A._narrow_street = lambda r, b, st_, *a, **k: None if st_.name == "turn" else narrow(r, b, st_, *a, **k)
    trust = A.trust_of
    tot = {}
    for f in FLOPS[:args.flops]:
        turn = next(c for c in ("9s", "2h", "5d") if c not in f)
        st = dict(flop_state(f + turn, "UTG", "AsKs", line), turn=chk)

        def turn_ranges(iters, full=False):
            adv = A.Advisor(time_budget=1e9)
            adv.mw_solve_streets = ("flop",) if iters else ()
            adv.mw_max_iters = iters or 1
            adv.mw_street_config["flop"]["buckets"] = args.buckets
            if iters:
                adv.advise(flop_state(f, "UTG", "AsKs", chk))
                expl = next(iter(adv._mw_lines.values()))[2]
            if full:
                adv.prior_smoothing = 0.0
                A.trust_of = lambda e, s_: 1.0
            try:
                rs = _spot_args(adv, st)[3]
            finally:
                A.trust_of = trust
            return rs, (expl if iters else None)
        ref, ref_e = turn_ranges(args.ref_iters, full=True)
        out = f"{f}{turn}: reference {args.ref_iters} iterations ({ref_e:.2f}% pot)"
        for name, it in (("rules", 0), ("advice 60", 60), (f"advice {args.iters}", args.iters)):
            rs, e = turn_ranges(it)
            d = {}
            for p in ("SB", "BB", "UTG"):
                a_, b_ = rs[p] / rs[p].sum(), ref[p] / ref[p].sum()
                d[p] = 0.5 * np.abs(a_ - b_).sum()
                tot.setdefault(name, []).append(d[p])
            out += f"; {name}" + (f" ({e:.2f}%, trust {trust(e, 'flop'):.2f})" if e is not None else "") + ": " + \
                ", ".join(f"{p} {d[p]:.3f} ({rs[p].sum():.0f} vs {ref[p].sum():.0f})" for p in d)
        print(out, flush=True)
    A._narrow_street = narrow
    print("mean distance from the reference's turn ranges: " + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in tot.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", action="store_true")
    ap.add_argument("--turn-reference", action="store_true")
    ap.add_argument("--flop-reference", action="store_true")
    ap.add_argument("--narrowing", action="store_true")
    ap.add_argument("--ref-buckets", type=int, default=24)
    ap.add_argument("--ref-rivers", type=int, default=12)
    ap.add_argument("--ref-iters", type=int, default=60)
    ap.add_argument("--facts", action="store_true")
    ap.add_argument("--flop-study", action="store_true")
    ap.add_argument("--hu-check", action="store_true")
    ap.add_argument("--three-only", action="store_true")
    ap.add_argument("--rules-checks", action="store_true",
                    help="--flop-study: the blinds' checks narrow their ranges by the betting rules first (as the advisor)")
    ap.add_argument("--full-street", action="store_true",
                    help="--flop-study: solve from the first blind's action and read the opener where both checked")
    ap.add_argument("--study-ranges", action="store_true")
    ap.add_argument("--later", default="buckets", choices=("deal", "buckets"))
    ap.add_argument("--buckets", type=int, default=16)
    ap.add_argument("--flop-sizes", default="33", help="the flop's bet sizes, %% pot (comma separated)")
    ap.add_argument("--raise-frac", type=float, default=None, help="raises to the bet plus this %% of the pot after calling")
    ap.add_argument("--step", type=int, default=1)
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
    if a.hu_check:
        hu_check(a)
    if a.turn_reference:
        turn_reference(a)
    if a.flop_reference:
        flop_reference(a)
    if a.narrowing:
        narrowing(a)


if __name__ == "__main__":
    main()
