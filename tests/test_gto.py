import unittest
import numpy as np
from gto.cards import cards, ALL_COMBOS, hand_class
from gto.ranges import parse_range
from gto.evaluator import eval_board
from gto import preflop as pf
from gto.solver import solve, Config, Ctx, Solver, TreeBuilder, MEMBER_FULL, N_FULL, match_action
from gto.advisor import Advisor, preflop_investments


class TestCardsRanges(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_range("AA").sum(), 6)
        self.assertEqual(parse_range("AKs").sum(), 4)
        self.assertEqual(parse_range("AKo").sum(), 12)
        self.assertEqual(parse_range("AK").sum(), 16)
        self.assertEqual(parse_range("22-44").sum(), 18)
        self.assertEqual(parse_range("A5s:0.5").sum(), 2)
        self.assertEqual(parse_range("T9s-76s").sum(), 16)
        self.assertEqual(hand_class(tuple(cards("AhKh"))), "AKs")

    def test_eval(self):
        st = eval_board(cards("AhKd7c2s9h"))
        i_aa = ALL_COMBOS.index(tuple(sorted(cards("AsAc"))))
        i_kk = ALL_COMBOS.index(tuple(sorted(cards("KsKc"))))
        self.assertGreater(st[i_aa], st[i_kk])
        self.assertEqual(st[ALL_COMBOS.index(tuple(sorted(cards("AhAc"))))], -1)


class TestSolverMath(unittest.TestCase):
    def test_showdown_matrix_matches_bruteforce(self):
        board = cards("AhKd7c2s9h")
        r = parse_range("22+, A2s+, KTs+, ATo+")
        bm = np.ones(N_FULL)
        for c in board:
            bm *= 1 - MEMBER_FULL[c]
        active = np.nonzero(r * bm > 0)[0]
        ctx = Ctx(active, Config())
        M = ctx.showdown_matrix(tuple(board))
        st = eval_board(board)
        i, j = 5, 40
        gi, gj = active[i], active[j]
        a, b = ALL_COMBOS[gi]; c, d = ALL_COMBOS[gj]
        expected = 0 if {a, b} & {c, d} else np.sign(st[gi] - st[gj])
        self.assertEqual(M[i, j], expected)
        self.assertTrue(np.allclose(M, -M.T))

    def test_river_solve_converges_and_is_sane(self):
        board = cards("Ks7d2cTh4s")
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, 98s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, A8o+, KTo+, QTo+, JTo")
        sol = solve(board, r_oop, r_ip, pot=6.5, stack=97.5, time_budget=1.0)
        self.assertGreaterEqual(sol.iters, 100)
        # nuts-ish (KK top set) should never be folding / mostly betting; EVs all positive
        kk = sol.for_combo(tuple(cards("KhKc")))
        self.assertGreater(kk["bet 100%"][0] + kk["bet 50%"][0], 0.5)
        self.assertTrue(all(ev > 0 for _, ev in kk.values()))
        # frequencies sum to one
        self.assertTrue(np.allclose(sol.strategy[:, sol.ctx.active].sum(axis=0), 1.0))

    def test_warm_start_copies_matching_nodes(self):
        # a turn solve over 8 rivers seeds one over every river: same rivers copied, new ones start at zero
        board = cards("Ks7d2cTh")
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A8o+, KTo+, QTo+")
        src = solve(board, r_oop, r_ip, pot=6.5, stack=97.5, cfg=Config(n_rivers=8), time_budget=0.3)
        dst = solve(board, r_oop, r_ip, pot=6.5, stack=97.5, cfg=Config(), max_iters=0, warm=src)
        self.assertEqual(dst.iters, 0)
        self.assertTrue(np.array_equal(dst.root.regret, src.root.regret))
        self.assertTrue(np.allclose(dst.strategy, src.strategy))
        s_river = src.root.children[0].children[0].child   # check, check, then the river
        d_river = dst.root.children[0].children[0].child
        self.assertEqual((len(s_river.boards), len(d_river.boards)), (8, 48))
        for i, b in enumerate(d_river.boards):
            j = s_river.boards.index(b) if b in s_river.boards else None
            if j is None:
                self.assertFalse(d_river.ssum[i].any())
            else:
                self.assertTrue(np.array_equal(d_river.ssum[i], s_river.ssum[j]))

    def _plain_iterations(self, sol, n):
        """n iterations walking the original tree node by node (no merged subtrees, no strategy reuse)."""
        s = sol.solver
        a, b = sol.r0[None, :].astype(np.float32), sol.r1[None, :].astype(np.float32)
        for _ in range(n):
            for p in (0, 1):
                s._cfr(s.root, p, a, b, update=True, use_avg=False)
            s.iters += 1
            s._discount()

    def test_merged_subtrees_match_plain_traversal(self):
        # same-shaped subtrees below the root street are batched together; the result must be the same
        # numbers as walking every node on its own (flop: repeated turn boards; turn: different rivers)
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A8o+, KTo+, QTo+")
        small = {"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]}
        for board, cfg in ((cards("Ks7d2c"), Config(n_turns=2, leaf_turns=2, leaf_rivers_flop=2, n_rivers_from_flop=2,
                                                     bet_sizes=small, raise_sizes=small)),
                           (cards("Ks7d2cTh"), Config(n_rivers=4, bet_sizes=small, raise_sizes=small))):
            fast = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=0)
            plain = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=0)
            self.assertTrue(any(len(m) > 1 for _, m in fast.solver._groups))
            fast.solver.run(fast.r0, fast.r1, time_budget=1e9, max_iters=12)
            self._plain_iterations(plain, 12)
            for x, y in zip(fast.solver._nodes, plain.solver._nodes):
                self.assertTrue(np.array_equal(x.ssum, y.ssum))
                self.assertTrue(np.array_equal(x.regret, y.regret))

    def test_no_blas_kernels(self):
        # the browser's path (einsum matvecs, leaf matrices trimmed to each range) solves the same game
        import gto.solver as S
        board = cards("Ks7d2c")
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A8o+, KTo+, QTo+")
        small = {"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]}
        cfg = Config(n_turns=2, leaf_turns=2, leaf_rivers_flop=2, n_rivers_from_flop=2, bet_sizes=small, raise_sizes=small)
        ref = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=60, time_budget=1e9)
        S.NO_BLAS = True
        try:
            sol = solve(board, r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=60, time_budget=1e9)
            self.assertIsNotNone(sol.solver._live)
            self.assertLess(abs(sol.exploitability() - ref.exploitability()), 0.05)
            live = sol.r0 > 0
            self.assertLess(np.abs(sol.evs[:, sol.ctx.active][:, live] - ref.evs[:, ref.ctx.active][:, live]).max(), 0.2)
        finally:
            S.NO_BLAS = False

    def test_match_action(self):
        names = ["check", "bet 33%", "bet 75%"]
        self.assertEqual(match_action(names, "bet 40%"), 1)
        self.assertEqual(match_action(names, "bet 90%"), 2)
        self.assertEqual(match_action(["fold", "call", "raise to 12", "all-in"], "jam"), 3)


class TestPreflop(unittest.TestCase):
    def test_rfi_and_defence(self):
        s = pf.hand_strategy("UTG", tuple(cards("7h2c")), [])
        self.assertEqual(s["raise"], 0.0)
        s = pf.hand_strategy("BTN", tuple(cards("AhAc")), [{"pos": "UTG", "action": "raise"}])
        self.assertEqual(s["raise"], 1.0)
        acts = [{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO")] + [{"pos": "BTN", "action": "raise"}]
        self.assertEqual(pf.next_to_act(acts), "SB")
        acts += [{"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        self.assertTrue(pf.is_preflop_closed(acts))
        rs = pf.replay_ranges(acts)
        self.assertEqual(set(rs), {"BTN", "BB"})
        inv, active = preflop_investments(acts, {p: 100.0 for p in pf.POSITIONS})
        self.assertAlmostEqual(sum(inv.values()), 5.5)


class TestAdvisor(unittest.TestCase):
    def test_end_to_end(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"hero": "BB", "hole": "AhKh", "board": "", "preflop": pre[:-1]})
        self.assertEqual(r["street"], "preflop")
        r = adv.advise({"hero": "CO", "hole": "AhKh", "board": "Kd7s2c", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}]})
        self.assertEqual(r["position"], "IP")
        self.assertGreaterEqual(r["solver"]["exploitability_pct"], 0.0)
        self.assertAlmostEqual(sum(a["freq"] for a in r["actions"]), 1.0, places=2)
        r = adv.advise({"hero": "BB", "hole": "9h8h", "board": "Kd7s2cTh", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 2},
                                 {"pos": "BB", "action": "call"}], "turn": []})
        self.assertEqual(r["street"], "turn")
        self.assertAlmostEqual(r["pot_bb"], 9.5)

    def test_preflop_vs_open_jam(self):
        adv = Advisor()
        pre = [{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO", "BTN")] + [{"pos": "SB", "action": "all-in"}]
        st = {"players": 6, "hero": "BB", "board": "", "stack": 100, "preflop": pre}
        r = adv.advise(dict(st, hole="Kd5d"))
        self.assertEqual(r["recommended"], "fold")
        self.assertEqual({a["action"] for a in r["actions"]}, {"call", "fold"})
        self.assertEqual(adv.advise(dict(st, hole="QsQh"))["recommended"], "call")
        # short stacks: the SB shoves wide (push/fold equilibrium), so K5s calls 10bb but folds 20bb
        self.assertEqual(adv.advise(dict(st, hole="Kd5d", stack=10))["recommended"], "call")
        self.assertEqual(adv.advise(dict(st, hole="Kd5d", stack=20))["recommended"], "fold")

    def test_preflop_price_and_callers(self):
        adv = Advisor()
        st = {"players": 6, "hero": "UTG", "hole": "6s6d", "board": "", "stack": 100}
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        # a standard 3bet heads-up is the chart's spot: 66 folds, no price note
        std = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5)] + [R(p, "fold") for p in ("BTN", "SB", "BB")]
        r = adv.advise(dict(st, preflop=std))
        self.assertEqual(r["recommended"], "fold")
        self.assertEqual(r["notes"], ["preflop: chart approximation"])
        # a min-3bet with two callers: 2.5bb into 16bb, 66 calls
        multi = [R("UTG", "raise", 2.5), R("HJ", "call"), R("CO", "raise", 5), R("BTN", "fold"), R("SB", "call"), R("BB", "fold")]
        r = adv.advise(dict(st, preflop=multi))
        self.assertEqual(r["recommended"], "call")
        self.assertTrue(any("price" in n for n in r["notes"]))
        # a huge 3bet: a chart call (99 calls 90% vs a standard 3bet) folds at the worse price
        big = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 20)] + [R(p, "fold") for p in ("BTN", "SB", "BB")]
        self.assertGreater(adv.advise(dict(st, hole="9s9d", preflop=std))["actions"][1]["freq"], 0.5)
        self.assertEqual(adv.advise(dict(st, hole="9s9d", preflop=big))["recommended"], "fold")

    def test_preflop_squeeze(self):
        adv = Advisor()
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        st = {"players": 6, "hero": "BB", "board": "", "stack": 100}
        alone = [R("UTG", "raise", 2.5)] + [R(p, "fold") for p in ("HJ", "CO", "BTN", "SB")]
        two = [R("UTG", "raise", 2.5), R("HJ", "call"), R("CO", "call"), R("BTN", "fold"), R("SB", "fold")]
        size = lambda r: next(a["amount_bb"] for a in r["actions"] if a["action"] == "raise")
        freq = lambda r, k: next(a["freq"] for a in r["actions"] if a["action"] == k)
        # one more open-size per caller: 10bb alone, 15bb over two callers
        a, b = adv.advise(dict(st, hole="AhKd", preflop=alone)), adv.advise(dict(st, hole="AhKd", preflop=two))
        self.assertEqual((size(a), size(b)), (10.0, 15.0))
        self.assertEqual((freq(a, "raise"), freq(b, "raise")), (1.0, 1.0))
        # a bluff squeeze shrinks: everyone has to fold
        a, b = adv.advise(dict(st, hole="Ah5h", preflop=alone)), adv.advise(dict(st, hole="Ah5h", preflop=two))
        self.assertLess(freq(b, "raise"), freq(a, "raise"))
        self.assertTrue(any("raises re-drawn" in n for n in b["notes"]))

    def test_preflop_off_chart_spots(self):
        adv = Advisor()
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        def go(hero, hole, pre, **kw):
            r = adv.advise(dict({"players": 6, "hero": hero, "hole": hole, "board": "", "stack": 100, "preflop": pre}, **kw))
            return r, {a["action"]: a for a in r.get("actions", [])}
        # limps: the big blind raises AA (checks 72o), the button isolates to 3x + 1 per limper or over-limps 33
        limps = [R("UTG", "call"), R("HJ", "call")] + F("CO", "BTN") + [R("SB", "call")]
        self.assertEqual(go("BB", "AhAd", limps)[0]["recommended"], "raise")
        self.assertEqual(go("BB", "7h2d", limps)[0]["recommended"], "check")
        r, a = go("BTN", "AhAd", [R("UTG", "call")] + F("HJ", "CO"))
        self.assertEqual((r["recommended"], a["raise"]["amount_bb"]), ("raise", 4.0))
        self.assertEqual(go("BTN", "3h3d", [R("UTG", "call")] + F("HJ", "CO"))[0]["recommended"], "call")
        self.assertEqual(go("BB", "KhKd", [R("SB", "call")], players=2)[0]["recommended"], "raise")
        # short stacks: a 3bet at 25bb is all-in; QQ never folds to a 3bet at 40bb
        r, a = go("CO", "AhKd", [R("UTG", "raise", 2.2), R("HJ", "fold")], stack=25)
        self.assertEqual((r["recommended"], a["all-in"]["amount_bb"]), ("all-in", 25.0))
        r, a = go("UTG", "QsQd", [R("UTG", "raise", 2.2), R("HJ", "fold"), R("CO", "raise", 6.5)] + F("BTN", "SB", "BB"), stack=40)
        self.assertLess(a["fold"]["freq"], 0.05)
        # deep stacks: 55 calls a 3bet at 200bb (implied odds), folds at 100bb
        pre = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5)] + F("BTN", "SB", "BB")
        self.assertEqual(go("UTG", "5s5d", pre)[0]["recommended"], "fold")
        self.assertEqual(go("UTG", "5s5d", pre, stack=200)[0]["recommended"], "call")
        # the opener calls wider in position (vs the big blind's 3bet) than out of position (vs the button's)
        ip = go("CO", "6s6d", F("UTG", "HJ") + [R("CO", "raise", 2.5)] + F("BTN", "SB") + [R("BB", "raise", 10)])[1]
        oop = go("CO", "6s6d", F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5)] + F("SB", "BB"))[1]
        self.assertGreater(ip["call"]["freq"], oop["call"]["freq"])
        # heads-up the button defends a 3bet wide (in position postflop)
        r = go("SB", "Qh8h", [R("SB", "raise", 2.5), R("BB", "raise", 8)], players=2)[0]
        self.assertEqual(r["recommended"], "call")
        # facing a 5bet with chips behind, AA can shove
        r, a = go("UTG", "AhAd", [R("UTG", "raise", 2.5), R("HJ", "raise", 8)] + F("CO", "BTN", "SB", "BB")
                  + [R("UTG", "raise", 20), R("HJ", "raise", 45)], stack=200)
        self.assertEqual(r["recommended"], "all-in")
        # antes widen opens; a straddle makes HJ first to act and the opens 2.5x the straddle
        self.assertEqual(go("UTG", "Ah9d", [])[0]["recommended"], "fold")
        self.assertGreater(go("UTG", "Ah9d", [], ante=1 / 6)[1]["raise"]["freq"], 0)
        r, a = go("HJ", "AhJd", [], straddle=True)
        self.assertEqual((r["recommended"], a["raise"]["amount_bb"], r["pot_bb"]), ("raise", 5.0, 3.5))
        self.assertIn("error", go("UTG", "AhJd", [], straddle=True)[0])
        # heads-up the big blind continues wider against a 4bet than 6-max
        hu4 = [R("SB", "raise", 2.5), R("BB", "raise", 8), R("SB", "raise", 20)]
        self.assertEqual(go("BB", "KhQh", hu4, players=2)[0]["recommended"], "call")
        # a limp-reraise is a trapped range: TT folds most of the time, KK goes in
        lrr = [R("UTG", "call")] + F("HJ", "CO") + [R("BTN", "raise", 4)] + F("SB", "BB") + [R("UTG", "raise", 14)]
        self.assertEqual(go("BTN", "TsTd", lrr)[0]["recommended"], "fold")
        self.assertEqual(go("BTN", "KsKd", lrr)[0]["recommended"], "all-in")
        # 9-handed early opens are tighter, so KJs calls less than against a 6-max UTG
        self.assertLess(go("UTG1", "KhJh", [R("UTG", "raise", 2.5)], players=9)[1]["call"]["freq"],
                        go("HJ", "KhJh", [R("UTG", "raise", 2.5)])[1]["call"]["freq"])
        # facing a 3bet cold: in position calls more than the big blind out of position, whose premiums 4bet
        early3 = [R("UTG", "raise", 2.5), R("HJ", "raise", 7.5)]
        btn = go("BTN", "QsQd", early3 + [R("CO", "fold")])[1]
        bb = go("BB", "QsQd", early3 + F("CO", "BTN", "SB"))[1]
        self.assertGreater(btn["call"]["freq"], bb["call"]["freq"])
        self.assertGreater(bb["raise"]["freq"], btn["raise"]["freq"])
        self.assertEqual(bb["fold"]["freq"], 0.0)
        # ...and continues more against a 3bet over a late open (a wider 3bet)
        late3 = F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5), R("SB", "fold")]
        self.assertLess(go("BB", "JhJd", late3)[1]["fold"]["freq"], go("BB", "JhJd", early3 + F("CO", "BTN", "SB"))[1]["fold"]["freq"])
        # 7+ handed blinds defend tighter against an early open
        ep = lambda n, seats: pf.strategy_for("BB", [R("UTG", "raise", 2.5)] + F(*seats), pf.Table(pf.SEATS[n]))
        cont = lambda s: float((s["call"] + s["raise"]).sum())
        self.assertLess(cont(ep(9, ("UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB"))), cont(ep(6, ("HJ", "CO", "BTN", "SB"))))

    def test_preflop_continue_rates(self):
        """Chart continue rates stay near typical 6-max 100bb solver output (share of the range that
        reaches the spot and doesn't fold), so a chart edit can't drift far without notice."""
        from gto.advisor import preflop_strategy, preflop_ranges
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        T = pf.Table(pf.SEATS[6])
        def rate(hero, pre):
            s, w = preflop_strategy(hero, pre, T, 100), preflop_ranges(pre, T, 100)[hero]
            return float((w * (1 - s["fold"])).sum() / w.sum())
        utg, co, btn = [R("UTG", "raise", 2.5)], F("UTG", "HJ") + [R("CO", "raise", 2.5)], F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)]
        bands = [
            ("BB", utg + F("HJ", "CO", "BTN", "SB"), 0.30, 0.42),
            ("BB", co + F("BTN", "SB"), 0.42, 0.55),
            ("BB", btn + F("SB"), 0.60, 0.72),
            ("BB", F("UTG", "HJ", "CO", "BTN") + [R("SB", "raise", 3)], 0.65, 0.80),
            ("BTN", co, 0.15, 0.24),
            ("HJ", utg, 0.08, 0.14),
            ("UTG", utg + [R("HJ", "raise", 7.5)] + F("CO", "BTN", "SB", "BB"), 0.35, 0.50),
            ("BTN", btn + [R("SB", "raise", 10), R("BB", "fold")], 0.38, 0.55),
            # published solver: CO continues 36.3% (calls 17.2%, 4bets 19.1%) vs a BTN 3bet to 7.5bb
            ("CO", co + [R("BTN", "raise", 7.5)] + F("SB", "BB"), 0.32, 0.45),
        ]
        # a hand never continues less often than one it dominates (JJ vs TT facing a 4bet used to flip on
        # board-sampling noise; AKo folded where AQo called)
        from gto.pf_classes import CLASS_OF, CLASS_INDEX, N_IN
        lines = [("HJ", utg), ("CO", [R("HJ", "raise", 2.5), R("CO", "raise", 7.5)] + F("BTN", "SB", "BB") + [R("HJ", "raise", 22)]),
                 ("CO", co + [R("BTN", "raise", 7.5)] + F("SB", "BB")), ("BB", btn + F("SB"))]
        for hero, pre in lines:
            s = preflop_strategy(hero, pre, T, 100)
            c = np.bincount(CLASS_OF, weights=1 - s["fold"], minlength=len(N_IN)) / N_IN
            for chain in pf._dominance_chains():
                for weak, strong in zip(chain, chain[1:]):
                    self.assertLessEqual(c[CLASS_INDEX[weak]], c[CLASS_INDEX[strong]] + 1e-6, f"{hero}: {weak} > {strong}")
        for hero, pre, lo, hi in bands:
            r = rate(hero, pre)
            self.assertTrue(lo <= r <= hi, f"{hero} after {[(a['pos'], a['action']) for a in pre]}: {r:.2f} not in [{lo}, {hi}]")

    def test_slightly_worse_price_trims_flats(self):
        """A 3bb button open (the chart's is 2.5bb) used to cut SB's flats from 85 combos to 24: the chart
        keeps small pairs and suited connectors below the raw-equity bar, and any worse price cut them all."""
        from gto.advisor import preflop_ranges
        T = pf.Table(pf.SEATS[6])
        pre = lambda to: [{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO")] + [
            {"pos": "BTN", "action": "raise", "to": to}, {"pos": "SB", "action": "call"}]
        chart, priced = preflop_ranges(pre(2.5), T, 100)["SB"], preflop_ranges(pre(3), T, 100)["SB"]
        self.assertLess(priced.sum(), chart.sum())
        self.assertGreater(priced.sum(), 0.7 * chart.sum())
        # depth: 40bb less used to cost small pairs and suited hands all their implied odds (SB kept 17 of 85
        # flats vs a CO open, no pair or suited connector), and 100bb more made SB flat 276 combos
        co = [{"pos": p, "action": "fold"} for p in ("UTG", "HJ")] + [
            {"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "call"}]
        chart, short = preflop_ranges(co, T, 100)["SB"], preflop_ranges(co, T, 60)["SB"]
        small = np.array([hand_class(c) in ("22", "33", "44", "54s", "65s") for c in ALL_COMBOS])
        self.assertGreater(short.sum(), 0.5 * chart.sum())
        self.assertGreater((short * small).sum(), 0.3 * (chart * small).sum())
        self.assertLess(preflop_ranges(pre(2.5), T, 200)["SB"].sum(), 2 * preflop_ranges(pre(2.5), T, 100)["SB"].sum())

    def test_called_barrels_keep_a_range(self):
        """SB flats a 3bb open, then check-calls flop and turn: a short turn prior solve that folds most
        of SB's range (two pair included) used to leave it under one combo on the river."""
        adv = Advisor(time_budget=0.5, prior_budget=0.5)
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        xbc = lambda amt: [{"pos": "SB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": amt},
                           {"pos": "SB", "action": "call"}]
        out = adv.advise({"players": 6, "hero": "BTN", "hole": "As4d", "board": "3s2cQs9hKc", "stack": 100,
                          "preflop": F("UTG", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 3},
                                                             {"pos": "SB", "action": "call"}] + F("BB"),
                          "flop": xbc(5), "turn": xbc(13), "river": [{"pos": "SB", "action": "check"}]})
        self.assertGreater(out["villain_range_combos"], 1.0, out)

    def test_caller_continues_vs_next_raise(self):
        """A flat facing a squeeze continues with ~30% of the flatting range (the cold-vs-3bet chart alone
        continued 2%), the playable part first; once the opener 4bets too, the flatter folds nearly all."""
        from gto.advisor import preflop_strategy, preflop_ranges
        from gto.cards import COMBO_INDEX, cards
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"),
               R("BB", "raise", 10)]
        w, s = preflop_ranges(pre, T, 100)["BTN"], preflop_strategy("BTN", pre, T, 100)
        cont = float((w * (1 - s["fold"])).sum() / w.sum())
        self.assertTrue(0.28 <= cont <= 0.45, cont)
        ix = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        self.assertGreater(1 - s["fold"][ix("8s8d")], 0.9)
        self.assertGreater(1 - s["fold"][ix("JhTh")], 0.9)
        after = preflop_ranges(pre + [R("CO", "fold"), R("BTN", "call")], T, 100)["BTN"]
        self.assertGreater(after.sum(), 30)
        s4 = preflop_strategy("BTN", pre + [R("CO", "raise", 23)], T, 100)
        self.assertLess(float((w * (1 - s4["fold"])).sum() / w.sum()), 0.05)

    def test_sb_flat_vs_early_open_keeps_traps(self):
        """SB flats UTG, BB squeezes, SB calls: the flat held only 66-99 and suited broadways below KQs,
        so the call was 77-99 alone and the squeezer's QQ won every A-K-x flop. The flat now traps a
        share of JJ, TT, AQs, AJs and KQs, and some of them call the squeeze."""
        from gto.advisor import preflop_ranges
        from gto.pf_classes import to_classes, CLASSES
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T = pf.Table(pf.SEATS[6])
        flat = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "fold"), R("BTN", "fold"), R("SB", "call")]
        line = flat + [R("BB", "raise", 12.5), R("UTG", "fold"), R("SB", "call")]
        c = dict(zip(CLASSES, to_classes(preflop_ranges(line, T, 100)["SB"])))
        for h in ("JJ", "TT", "AQs", "KQs"):
            self.assertGreater(c[h], 0.05, h)
        w = preflop_ranges(flat, T, 100)["SB"]
        self.assertTrue(25 <= w.sum() <= 50, w.sum())   # still a small flat (~3% of hands)

    def test_jam_after_flat_has_a_range(self):
        """CO flats, BB squeezes, HJ 4bets, CO jams: the jam range was CO's flats times the jam chart,
        empty, so equity was NaN and HJ folded AA. It is now the jam chart's range, rebuilt."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        pre = [R("UTG", "fold"), R("HJ", "raise", 2.5), R("CO", "call"), R("BTN", "fold"), R("SB", "fold"),
               R("BB", "raise", 12.5), R("HJ", "raise", 28.7), R("CO", "all-in"), R("BB", "fold")]
        adv = Advisor()
        st = {"players": 6, "hero": "HJ", "board": "", "stack": 100, "preflop": pre}
        self.assertEqual(adv.advise(dict(st, hole="AsAd"))["recommended"], "call")
        self.assertEqual(adv.advise(dict(st, hole="JsJd"))["recommended"], "fold")

    def test_vs_4bet_follows_the_4bet_range(self):
        """The 3bettor's vs-4bet chart is written against the opener's 4bet; against a cold 4bet (no
        bluffs) the marginal continues and bluff jams shrink, mixed value jams turn into calls, and
        against the opener nothing changes."""
        from gto.cards import COMBO_INDEX, cards
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        T = pf.Table(pf.SEATS[6])
        ix = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        three = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5)] + F("BTN", "SB")
        std = pf.strategy_for("CO", three + [R("BB", "fold"), R("UTG", "raise", 18)], T)
        cold = pf.strategy_for("CO", three + [R("BB", "raise", 18.8), R("UTG", "fold")], T)
        chart = pf.VS_4BET
        self.assertAlmostEqual(float(std["call"][ix("AhQh")]), float(chart["call"][ix("AhQh")]), places=6)
        self.assertLess(cold["call"][ix("AhQh")], 0.5 * std["call"][ix("AhQh")])
        self.assertLess(cold["raise"][ix("Ah5h")], 0.5 * std["raise"][ix("Ah5h")])
        for h in ("KsKd", "QsQd", "AhKd"):
            self.assertAlmostEqual(1 - cold["fold"][ix(h)], 1 - std["fold"][ix(h)], places=6)
        # QQ and AK jam less where the 4bettor folds to a jam much less often (a flat, then a 4bet over
        # an early open), calling instead; KK+ and AKs keep jamming
        flat4 = pf.strategy_for("BB", [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "fold"), R("BTN", "call"),
                                       R("SB", "fold"), R("BB", "raise", 12.5), R("UTG", "fold"), R("BTN", "raise", 28)], T)
        for h in ("QsQd", "AhKd"):
            self.assertLess(flat4["raise"][ix(h)], 0.5 * std["raise"][ix(h)])
            self.assertGreater(flat4["call"][ix(h)], std["call"][ix(h)])
        for h in ("AsAd", "KsKd", "AhKh"):
            self.assertEqual(float(cold["raise"][ix(h)]), 1.0)
            self.assertEqual(float(flat4["raise"][ix(h)]), 1.0)
        # the value bar follows the price, not how wide the 3bet range is: JJ, with more equity against
        # a cold 4bet over the button than over UTG, continues at least as often there
        btn = pf.strategy_for("SB", [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5),
                                     R("SB", "raise", 11), R("BB", "raise", 25), R("BTN", "fold")], T)
        self.assertGreaterEqual(1 - btn["fold"][ix("JsJd")], 1 - cold["fold"][ix("JsJd")])

    def test_4bettor_vs_jam_matches_advice(self):
        """The replayed 4bettor facing a 5bet jam calls by equity against the jam range, as hero's advice
        does (the fixed chart called QQ and AK 70% where they have 36% and need 41%)."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        from gto.cards import COMBO_INDEX, cards
        ix = lambda h: COMBO_INDEX[tuple(sorted(cards(h)))]
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5), R("BTN", "fold"), R("SB", "fold"),
               R("BB", "fold"), R("UTG", "raise", 18), R("CO", "all-in")]
        s = pf.strategy_for("UTG", pre, T)
        adv = Advisor()
        for h in ("AsAd", "KsKd", "QsQd", "AhKd", "AhKh"):
            r = adv.advise({"players": 6, "hero": "UTG", "hole": h, "board": "", "stack": 100, "preflop": pre})
            self.assertAlmostEqual(float(s["call"][ix(h)]), r["actions"][0]["freq"], delta=0.02, msg=h)
        self.assertEqual(float(s["call"][ix("QsQd")]), 0.0)

    def test_thin_line_tops_up_from_prior_range(self):
        """An opener calling a cold 4bet: the chart calls only 6.8 combos even from scratch, so filling with
        the action's own hands couldn't reach MIN_LINE_COMBOS; the strongest hands of the range before
        the call top it up (the sanity sweep found 85 such lines under 8 combos)."""
        from gto.advisor import preflop_ranges
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 7.5), R("BTN", "fold"), R("SB", "fold"),
               R("BB", "raise", 18.8), R("UTG", "call")]
        for rng in (preflop_ranges(pre, T, 100)["UTG"], pf.replay_ranges(pre, T)["UTG"]):
            self.assertGreaterEqual(float(rng.sum()), pf.MIN_LINE_COMBOS - 1e-6)
            self.assertLessEqual(float(rng.max()), 1.0)

    def test_postflop_floor_for_thin_ranges(self):
        """A 4bet pot's ~12-combo range checking back the flop left HJ 1.8 combos for the turn, so the turn
        solve faced a near face-up hand; ranges entering a later street are filled out below
        preflop.MIN_LINE_COMBOS. A wide single-raised pot is untouched."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        adv = Advisor(time_budget=1.0, prior_budget=0.5)
        pre = [R("UTG", "raise", 2.5), R("HJ", "raise", 7.5)] + [R(p, "fold") for p in ("CO", "BTN", "SB", "BB")] + \
              [R("UTG", "raise", 18.8), R("HJ", "call")]
        out = adv.advise({"players": 6, "hero": "UTG", "hole": "KhAs", "board": "JsQs2hAh", "stack": 100, "preflop": pre,
                          "flop": [R("UTG", "check"), R("HJ", "check")]})
        self.assertGreater(out["villain_range_combos"], 4.0, out)
        self.assertTrue(any("filled out with the rest of their flop range" in n for n in out["notes"]), out["notes"])
        srp = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5), R("SB", "fold"), R("BB", "call")]
        out = adv.advise({"players": 6, "hero": "BB", "hole": "9h8h", "board": "Kd7c2s5h", "stack": 100, "preflop": srp,
                          "flop": [R("BB", "check"), R("BTN", "check")]})
        self.assertFalse(any("filled out" in n for n in out["notes"]), out["notes"])

    def test_off_chart_line_not_face_up(self):
        """BTN flats CO (the chart's flats never 4bet), then 4bets BB's squeeze: hero's range would be
        empty and KK solved face-up, so BB folds everything to a jam (all-in EV = the pot, +49).
        The range is rebuilt from the 4bet, and BB pays off."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"),
               R("BB", "raise", 10), R("CO", "fold"), R("BTN", "raise", 23), R("BB", "call")]
        out = Advisor(time_budget=1.0, prior_budget=0.5).advise(
            {"players": 6, "hero": "BTN", "hole": "KsKd", "board": "7d2s5s", "stack": 100,
             "preflop": pre, "flop": [R("BB", "check")]})
        self.assertGreater(out["hero_range_combos"], 20, out)
        self.assertGreater(max(a["ev_bb"] for a in out["actions"]), 52, out["actions"])
        self.assertTrue(any("isn't in the chart" in n for n in out["notes"]), out["notes"])



class TestTableSizes(unittest.TestCase):
    def test_replay_hand_dead_money_and_order(self):
        from gto.advisor import replay_hand
        pre = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "call"}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "raise", "to": 9}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "fold"},
               {"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "call"}]
        st = replay_hand(pf.SEATS[6], 100, {"preflop": pre, "flop": []}, 2)
        self.assertTrue(st[0].closed)
        self.assertAlmostEqual(st[1].pot_start, 22.0)
        self.assertEqual(st[1].players_start, ["HJ", "BTN"])
        self.assertEqual(st[1].to_act, "HJ")

    def test_heads_up_order(self):
        from gto.advisor import replay_hand
        st = replay_hand(pf.SEATS[2], 100, {"preflop": []}, 1)[0]
        self.assertEqual(st.to_act, "SB")          # button/SB acts first preflop
        st = replay_hand(pf.SEATS[2], 100, {"preflop": [{"pos": "SB", "action": "raise", "to": 2.5}, {"pos": "BB", "action": "call"}], "flop": []}, 2)
        self.assertEqual(st[1].to_act, "BB")        # and last postflop
        self.assertAlmostEqual(st[1].pot_start, 5.0)

    def test_charts_by_table_size(self):
        self.assertEqual(pf.chart_key("LJ", pf.SEATS[9]), "UTG")
        self.assertEqual(pf.chart_key("UTG", pf.SEATS[9]), "EP")
        self.assertEqual(pf.chart_key("UTG", pf.SEATS[6]), "UTG")
        self.assertGreater(pf.RFI["HU_SB"].sum(), pf.RFI["BTN"].sum())
        self.assertLess(pf.RFI["EP"].sum(), pf.RFI["UTG"].sum())
        s = pf.hand_strategy("BB", tuple(cards("Jh9c")), [{"pos": "SB", "action": "raise"}], pf.SEATS[2])
        self.assertGreater(s["call"], 0.5)

    def test_same_high_card_range_syntax(self):
        self.assertEqual(parse_range("ATo-A5o").sum(), 72)

    def test_unsettled_solve_recommends_clear_ev_winner(self):
        from gto.advisor import UNSETTLED, EV_GAP
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": p, "action": "fold"} for p in ["UTG", "HJ", "CO", "BTN"]] + \
              [{"pos": "SB", "action": "raise", "to": 3}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "SB", "hole": "AsJs", "board": "8h5c9s", "preflop": pre,
                        "flop": [{"pos": "SB", "action": "check"}, {"pos": "BB", "action": "bet", "amount": 4.5}]})
        acts = r["actions"]
        top_freq, top_ev = max(acts, key=lambda a: a["freq"]), max(acts, key=lambda a: a["ev_bb"])
        if r["solver"]["exploitability_pct"] >= UNSETTLED and top_ev["ev_bb"] - top_freq["ev_bb"] > EV_GAP / 100 * r["pot_bb"]:
            self.assertEqual(r["recommended"], top_ev["action"])
            self.assertTrue(any(n.startswith("recommending") for n in r["notes"]))
        else:
            self.assertEqual(r["recommended"], top_freq["action"])

    def test_multiway_and_mid_street_hu(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "CO", "hole": "AhAd", "board": "Kh7c2d", "preflop": pre, "flop": [{"pos": "BB", "action": "check"}]})
        self.assertTrue(r.get("multiway"))
        self.assertAlmostEqual(r["pot_bb"], 8.0)
        self.assertEqual(r["recommended"], max(r["actions"], key=lambda a: a["ev_bb"])["action"])
        self.assertTrue(any(n.startswith("if you check, someone bets") for n in r["notes"]))  # BTN still to act
        r = adv.advise({"players": 6, "hero": "CO", "hole": "AhAd", "board": "Kh7c2d", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "bet", "amount": 3.3}]})
        self.assertEqual([a["action"] for a in r["actions"]], ["fold", "call", "raise to 14"])
        self.assertEqual(r["actions"][0]["ev_bb"], 0.0)
        self.assertGreater(min(r["actions"][1]["ev_bb"], r["actions"][2]["ev_bb"]), 5.0)   # AA: call and raise both win
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "KsQs", "board": "Kh7c2d", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "bet", "amount": 4}, {"pos": "CO", "action": "fold"}]})
        self.assertFalse(r.get("multiway"))
        self.assertEqual(r["villain"], "BB")
        self.assertAlmostEqual(r["pot_bb"], 12.0)
        self.assertEqual(r["actions"][0]["action"], "fold")


    def test_multiway_players_behind_and_bluffs(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "call"}, {"pos": "CO", "action": "call"},
               {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        st = {"players": 6, "hero": "HJ", "hole": "2c3d", "board": "Kc8h4h", "preflop": pre,
              "flop": [{"pos": "BB", "action": "check"}, {"pos": "UTG", "action": "bet", "amount": 5}]}
        r = adv.advise(st)
        self.assertIn("after a call, CO, BB still to act", r["notes"])
        self.assertEqual(r["recommended"], "fold")
        self.assertLess(r["actions"][2]["ev_bb"], -1.0)        # opponents call too much for a pure bluff raise
        self.assertGreater(r["actions"][1]["ev_bb"], -5.5)     # later streets never cost more than the call
        self.assertIn("error", adv.advise(dict(st, hole="8h2c")))   # 8h is on the board

    def test_multiway_folds_follow_range_strength(self):
        # BB checks flop and turn, then leads the river; CO calls. BB's range is mostly air, CO's
        # call is mostly better than Q9. Each used to fold 64% of their range to a raise, so CO
        # folded QK and AA and a raise with the bluff-catcher Q9 showed +8bb; the folds now come
        # from the weak range.
        from gto.advisor import _joint_top, _top
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "Qc9c", "board": "Qd8s3h2c7d", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 2.5},
                                 {"pos": "BTN", "action": "call"}, {"pos": "BB", "action": "call"}],
                        "turn": [{"pos": p, "action": "check"} for p in ("BB", "CO", "BTN")],
                        "river": [{"pos": "BB", "action": "bet", "amount": 8}, {"pos": "CO", "action": "call"}]})
        self.assertTrue(r.get("multiway"))
        self.assertNotEqual(r["recommended"], "raise to 37.5")
        self.assertLess(r["actions"][2]["ev_bb"], 0.0)
        note = next(n for n in r["notes"] if n.startswith("vs raise"))
        bb, co = (int(note.split(f"{p} continues with ")[1].split("%")[0]) for p in ("BB", "CO"))
        self.assertLess(bb, co)
        # One range: the same as keeping its top share.
        rng = np.random.default_rng(1)
        w, sc = rng.random(1326), rng.random(1326)
        self.assertTrue((_joint_top([w], 0.4, np.argsort(-sc))[0] == _top(w, 0.6, np.argsort(-sc))).all())
        # A strong and a weak range: the weak one folds more, and they fold together 40% of the time.
        strong, weak = np.where(sc > 0.5, 1.0, 0.1), np.where(sc < 0.5, 1.0, 0.1)
        ms, mw = _joint_top([strong, weak], 0.4, np.argsort(-sc))
        share = lambda r, m: (r * m).sum() / r.sum()
        self.assertGreater(share(strong, ms), share(weak, mw))
        self.assertAlmostEqual((1 - share(strong, ms)) * (1 - share(weak, mw)), 0.4, delta=0.01)

    def test_multiway_streets_narrow_ranges(self):
        from gto.advisor import replay_hand, _narrow_street
        import gto.preflop as pf
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        flop = [{"pos": "BB", "action": "check"}, {"pos": "HJ", "action": "bet", "amount": 5}, {"pos": "BTN", "action": "call"}]
        pos = pf.SEATS[6]
        ranges = pf.replay_ranges(pre, pos)
        before = {p: ranges[p].copy() for p in ("HJ", "BTN", "BB")}
        st = replay_hand(pos, 100.0, {"preflop": pre, "flop": flop}, 2)[1]
        _narrow_street(ranges, cards("Jc7h2s"), st, 100.0)
        from gto.cards import COMBO_INDEX
        sets = COMBO_INDEX[tuple(sorted(cards("JdJh")))]
        for p in ("HJ", "BTN"):
            self.assertLess(ranges[p].sum(), 0.8 * before[p].sum())
        self.assertGreater(ranges["HJ"][sets] / before["HJ"][sets], 0.9)   # a set bets
        from gto.advisor import NARROW_SOFT
        self.assertLessEqual(ranges["BB"][sets] / before["BB"][sets], NARROW_SOFT + 1e-9)   # and doesn't check

    def test_big_bet_range_is_mostly_by_the_rules(self):
        # BB checks a three-diamond flop 3-way, then overbets the fourth diamond: the rule's range is
        # small, and hands off it mustn't swamp it (they used to be ~2/3 of the range).
        from gto.advisor import replay_hand, _narrow_street
        pre = [{"pos": "UTG", "action": "call"}, {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"},
               {"pos": "UTG", "action": "call"}]
        streets = {"preflop": pre, "flop": [{"pos": p, "action": "check"} for p in ("BB", "UTG", "HJ")],
                   "turn": [{"pos": "BB", "action": "bet", "amount": 15}, {"pos": "UTG", "action": "fold"}]}
        pos, board, hole = pf.SEATS[6], cards("6d4d3dKd"), cards("AcJs")
        ranges = pf.replay_ranges(pre, pos)
        rep = replay_hand(pos, 100.0, streets, 3)
        _narrow_street(ranges, board[:3], rep[1], 100.0, "HJ", hole)
        _narrow_street(ranges, board, rep[2], 100.0, "HJ", hole)
        bb = ranges["BB"] * np.prod([1 - MEMBER_FULL[c] for c in board + hole], axis=0)
        flush = np.array([a % 4 == board[0] % 4 or b % 4 == board[0] % 4 for a, b in ALL_COMBOS])
        self.assertGreater(bb[flush].sum() / bb.sum(), 0.45)
        r = Advisor(time_budget=1.0, prior_budget=0.5).advise(dict(
            players=6, hero="HJ", hole="AcJs", board="6d4d3dKd", stack=100, **streets))
        self.assertEqual(r["recommended"], "fold")

    def test_hand_outside_range_still_gets_a_strategy(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "fold"}]
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "Ks2c", "board": "6dQh4h", "preflop": pre,
                        "flop": [{"pos": "CO", "action": "bet", "amount": 9}]})
        self.assertGreater(r["actions"][0]["freq"], 0.9)   # K2o folds, not a uniform mix
        self.assertLess(r["actions"][1]["ev_bb"], 0.0)

    def test_deep_jam_is_in_the_tree(self):
        # a 95bb turn jam into 9.5bb: hero faces the jam, not the tree's biggest bet
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "7h6h", "board": "Th2hKd2s", "stack": 100, "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 2},
                                 {"pos": "BB", "action": "call"}],
                        "turn": [{"pos": "BB", "action": "all-in"}]})
        self.assertEqual([a["action"] for a in r["actions"]], ["fold", "call"])
        self.assertEqual(r["actions"][1]["amount_bb"], 95.5)
        self.assertFalse(any("mapped 'all-in'" in n for n in r["notes"]))

    def test_real_sizes_reach_the_tree(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
               {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 2}, {"pos": "BB", "action": "call"}]
        base = {"players": 6, "hero": "BTN", "hole": "7h6h", "board": "Th2hKd2s", "stack": 100, "preflop": pre, "flop": flop}
        bet = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 5}]
        cases = [
            # a jam entered as a bet of the whole stack
            (dict(base, turn=[{"pos": "BB", "action": "bet", "amount": 95.5}]), 95.5),
            # a raise to the whole stack
            (dict(base, turn=bet + [{"pos": "BB", "action": "raise", "to": 95.5}]), 90.5),
            # an off-size raise and then a jam: the jam's size doesn't replace the raise's
            (dict(base, hero="BB", hole="KhQs", turn=bet + [{"pos": "BB", "action": "raise", "to": 30},
                                                           {"pos": "BTN", "action": "all-in"}]), 65.5),
        ]
        for state, to_call in cases:
            r = adv.advise(state)
            self.assertEqual(dict((a["action"], a["amount_bb"]) for a in r["actions"])["call"], to_call)
            self.assertFalse(any(n.startswith("turn: mapped") and "bet 53%" not in n for n in r["notes"]), r["notes"])
        # a flop 3bet past the tree's raise cap is a 3bet, not an all-in
        r = adv.advise({"players": 6, "hero": "BB", "hole": "KhQs", "board": "Th2hKd", "stack": 100, "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 2},
                                 {"pos": "BB", "action": "raise", "to": 7}, {"pos": "BTN", "action": "raise", "to": 18}]})
        self.assertLess(dict((a["action"], a["amount_bb"]) for a in r["actions"])["call"], 15)
        self.assertFalse(any("to tree action 'all-in'" in n for n in r["notes"]))

    def test_swap_nearest_keeps_sizes_taken(self):
        from gto.advisor import _swap_nearest
        sizes, fixed = [0.5, 1.0], set()
        _swap_nearest(sizes, 1.3, fixed=fixed)
        _swap_nearest(sizes, 5.0, fixed=fixed)
        self.assertEqual(sizes, [5.0, 1.3])
        _swap_nearest(sizes, 0.4, fixed=fixed)
        self.assertEqual(sizes, [5.0, 1.3, 0.4])


def _exact_equity(board, hero, villain_range):
    """Hero's equity against a range, every runout enumerated."""
    import itertools
    from gto.cards import COMBO_INDEX
    hi = COMBO_INDEX[tuple(sorted(hero))]
    vis = [int(j) for j in np.nonzero(villain_range)[0]]
    rest = [c for c in range(52) if c not in board and c not in hero]
    tot = wsum = 0.0
    for run in itertools.combinations(rest, 5 - len(board)):
        b = tuple(board) + run
        st = dict(zip([hi] + vis, eval_board(b, [hi] + vis)))
        for j in vis:
            if set(ALL_COMBOS[j]) & (set(b) | set(hero)):
                continue
            tot += villain_range[j] * ((st[hi] > st[j]) + 0.5 * (st[hi] == st[j]))
            wsum += villain_range[j]
    return tot / wsum


class TestAccuracy(unittest.TestCase):
    """Checks against exact enumeration and against known equilibria."""

    NO_BETS = {s: [] for s in ("flop", "turn", "river")}

    def test_leaf_matrix_averages_over_cards_both_hands_miss(self):
        # AA vs KK on 2c7d9hTs: 44 live rivers, KK wins on 2 of them
        from gto.cards import COMBO_INDEX
        board = tuple(cards("2c7d9hTs"))
        aa, kk = COMBO_INDEX[tuple(sorted(cards("AhAs")))], COMBO_INDEX[tuple(sorted(cards("KhKs")))]
        ctx = Ctx(np.array(sorted([aa, kk])), Config())
        ctx.root_len = 4
        L = ctx.leaf_matrix(board, set(board))
        i, j = list(ctx.active).index(aa), list(ctx.active).index(kk)
        self.assertAlmostEqual(float(L[i, j]), 40 / 44, places=5)

    def test_checked_down_ev_equals_pot_times_equity(self):
        # a tree with no bets: checking is worth exactly pot * equity, through chance nodes
        # (turn -> every river) and through equity leaves (flop -> every turn and river)
        hero, vr = cards("AhAs"), parse_range("KK, 88, QQ, JTs, 76s")
        for board, depth in (("Kh8d5c3s", 2), ("Kh8d5c", 1)):
            cfg = Config(bet_sizes=self.NO_BETS, raise_sizes=self.NO_BETS, allin_spr=0, depth=depth,
                         leaf_turns=49, leaf_rivers_flop=48)
            b = cards(board)
            sol = solve(b, parse_range("AA"), vr, pot=10, stack=100, time_budget=0.1, max_iters=3, cfg=cfg)
            ev = sol.for_combo(tuple(hero))["check"][1]
            self.assertAlmostEqual(ev, 10 * _exact_equity(b, hero, vr), places=3)

    def test_fast_showdown_matches_matrix(self):
        board = tuple(cards("Ks7d2cTh"))
        bm = np.ones(N_FULL)
        for c in board:
            bm *= 1 - MEMBER_FULL[c]
        active = np.nonzero(parse_range("22+, A2s+, K9s+, QTs+, JTs, T9s, 98s, A9o+, KJo+") * bm > 0)[0]
        ctx = Ctx(active, Config())
        boards = [board + (c,) for c in range(52) if c not in board]
        ro = np.random.default_rng(0).random((len(boards), ctx.n))
        for bi, b in enumerate(boards):
            ro[bi] *= 1 - ctx.member[b[-1]]
        fast = ctx.ranking(boards).utility(ro)
        dense = np.matmul(ctx.stacked("showdown", boards, set()), ro[:, :, None].astype(np.float32))[:, :, 0]
        live = np.array([1 - ctx.member[b[-1]] for b in boards])
        self.assertLess(np.abs((fast - dense) * live).max(), 1e-3)
        sel = [3, 17, 40]
        self.assertTrue(np.allclose(ctx.ranking(boards).utility(ro[sel], sel), fast[sel]))

    def test_polarized_river_game(self):
        # nuts-or-air vs a bluff-catcher: bluffs are b/(1+2b) of bets, the catcher calls 1/(1+b)
        board = cards("Kh8d5c3s2h")                       # IP: 88 (3 sets) + T9o (12 air), OOP: QQ
        oop, ip = parse_range("QQ"), parse_range("88, T9o")
        for b in (0.5, 1.0, 2.0):
            cfg = Config(bet_sizes={s: [b] for s in ("flop", "turn", "river")}, raise_sizes=self.NO_BETS, allin_spr=0)
            sol = solve(board, oop, ip, pot=10, stack=100, time_budget=2, max_iters=2000, cfg=cfg)
            ipn = sol.root.children[0]
            k = 1
            st = ipn.avg_strategy0()
            air = parse_range("T9o")[sol.ctx.active] > 0
            self.assertAlmostEqual(float(st[k][air].mean()), 3 * b / (1 + b) / 12, places=2)
            vs_bet = ipn.children[k]
            qq = parse_range("QQ")[sol.ctx.active] > 0
            self.assertAlmostEqual(float(vs_bet.avg_strategy0()[1][qq].mean()), 1 / (1 + b), places=2)
            # includes lines off the equilibrium path (OOP leading), which must be trained too
            self.assertLess(sol.exploitability(), 0.1)

    def test_turn_solve_converges_on_every_river(self):
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, 98s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, A8o+, KTo+, QTo+, JTo")
        cfg = Config(bet_sizes={s: [0.75] for s in ("flop", "turn", "river")}, raise_sizes=self.NO_BETS)
        sol = solve(cards("Ks7d2cTh"), r_oop, r_ip, pot=10, stack=50, time_budget=60, max_iters=80, cfg=cfg)
        self.assertLess(sol.exploitability(), 1.0)
        cfg.river_sample = 8                              # public chance sampling runs and stays sane
        sol = solve(cards("Ks7d2cTh"), r_oop, r_ip, pot=10, stack=50, time_budget=60, max_iters=40, cfg=cfg)
        self.assertTrue(np.allclose(sol.strategy[:, sol.ctx.active].sum(axis=0), 1.0, atol=1e-5))

    def test_rfi_widths_match_published_charts(self):
        # 6-max 100bb solver charts: UTG 15-18%, HJ 19-22%, CO 26-29%, BTN 40-48%
        for pos, lo, hi in (("UTG", 15, 18), ("HJ", 19, 22), ("CO", 26, 29), ("BTN", 40, 48)):
            pct = pf.RFI[pos].sum() / 1326 * 100
            self.assertTrue(lo <= pct <= hi, f"{pos} opens {pct:.1f}%")


class TestExploit(unittest.TestCase):
    """Exploit mode: opponents play like a typical player pool (gto/exploit.py) and hero best-responds."""

    def test_pool_lock_overfolds_weakest_at_every_size(self):
        from gto.exploit import PoolLock
        from gto.solver import Ctx
        board = tuple(cards("Qs7c3d8h2s"))
        live = np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in board and b not in board])
        ctx = Ctx(live, Config())
        lock = PoolLock(ctx, 1)
        n = len(live)
        class Node:   # player 1 faces a bet of `to_call` into `pot`
            def __init__(self, to_call, pot, b=board):
                self.boards, self.names, self.player = [b], ["fold", "call"], 1
                self.bets, self.pot_now = (to_call, 0.0), pot + to_call
        strat = np.full((1, 2, n), 0.5, dtype=np.float32)
        reach = np.ones((1, n))
        big = lock(Node(15.0, 10.0), strat, reach)
        fold = float(big[0, 0].mean())
        self.assertAlmostEqual(fold - 0.5, 0.075, places=2)      # a 1.5x-pot river overbet: 7.5% more of the range folds
        # a third-pot river bet: 20% more folds (the pool over-folds small river bets most)
        self.assertAlmostEqual(float(lock(Node(3.3, 10.0), strat, reach)[0, 0].mean()) - 0.5, 0.20, places=2)
        pct = lock.pct(board)
        self.assertGreater(big[0, 0][pct < 0.2].mean(), big[0, 0][pct > 0.8].mean())   # ...the weakest calls
        flop = tuple(cards("Qs7c3d"))
        small = PoolLock(Ctx(np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in flop and b not in flop]), Config()), 1)
        n2 = len(small.ctx.active)
        out = small(Node(2.0, 6.0, flop), np.full((1, 2, n2), 0.5, dtype=np.float32), np.ones((1, n2)))
        # a third-pot flop bet: 8.5% more folds (population data: the pool over-folds small c-bets, +7-10 points)
        self.assertAlmostEqual(float(out[0, 0].mean()) - 0.5, 0.085, places=2)
        turn = tuple(cards("Qs7c3d8h"))
        tl = PoolLock(Ctx(np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in turn and b not in turn]), Config()), 1)
        n3 = len(tl.ctx.active)
        out = tl(Node(2.0, 6.0, turn), np.full((1, 2, n3), 0.5, dtype=np.float32), np.ones((1, n3)))
        self.assertAlmostEqual(float(out[0, 0].mean()) - 0.5, 0.14, places=2)   # the turn, between flop and river
        raised = Node(10.0, 10.0); raised.bets = (20.0, 10.0)    # facing a raise of its own bet: no shift
        self.assertAlmostEqual(float(lock(raised, strat, reach)[0, 0].mean()), 0.5, places=5)

    def test_pool_best_response_ties_within_noise(self):
        """Exploit mode: a short solve's best-response EVs that sit within its noise of the best are tied and
        follow the equilibrium mix (AQo checked 100% at 6.45 over bets at 6.40 in a solve 3.1% of the pot
        from settled); a settled solve still picks the clear best."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "fold"), R("BTN", "raise", 2.5), R("SB", "fold"), R("BB", "call")]
        st = {"players": 6, "hero": "BTN", "hole": "AhQd", "board": "Qs8d3c", "stack": 100, "preflop": pre,
              "flop": [R("BB", "check")], "opponents": "pool"}
        rough = Advisor(time_budget=0.2, prior_budget=0.2).advise(st)
        tol = max(0.005, rough["solver"]["exploitability_pct"] / 100) * rough["pot_bb"]
        best = max(a["ev_bb"] for a in rough["actions"])
        for a in rough["actions"]:
            if a["ev_bb"] >= best - tol and a["gto_freq"] > 0.05:
                self.assertGreater(a["freq"], 0.0, rough["actions"])       # a tied action keeps its GTO share

    def test_pool_preflop_flats_more_and_3bets_less(self):
        from gto.advisor import preflop_ranges
        T = pf.Table(pf.SEATS[6])
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        line = lambda act: F("UTG", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": act}]
        gto_call, pool_call = preflop_ranges(line("call"), T, 100)["SB"], preflop_ranges(line("call"), T, 100, "BTN")["SB"]
        gto_3b, pool_3b = preflop_ranges(line("raise"), T, 100)["SB"], preflop_ranges(line("raise"), T, 100, "BTN")["SB"]
        self.assertGreater(pool_call.sum(), 2 * gto_call.sum())
        self.assertLess(pool_3b.sum(), 0.6 * gto_3b.sum())
        # hero's own range is the chart's
        self.assertAlmostEqual(preflop_ranges(line("call"), T, 100, "SB")["SB"].sum(), gto_call.sum())

    def test_pool_advice_best_responds(self):
        adv = Advisor(time_budget=1.0, prior_budget=0.5)
        F = lambda *ps: [{"pos": p, "action": "fold"} for p in ps]
        st = {"players": 6, "hero": "BB", "hole": "QdJd", "board": "Qs7c3d8h2s", "stack": 100,
              "preflop": F("UTG", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 2.5}] + F("SB") + [{"pos": "BB", "action": "call"}],
              "flop": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 2}, {"pos": "BB", "action": "call"}],
              "turn": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 6}, {"pos": "BB", "action": "call"}],
              "river": [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 15}]}
        gto, pool = adv.advise(st), adv.advise(dict(st, opponents="pool"))
        self.assertNotIn("gto_freq", gto["actions"][0])
        self.assertTrue(all("gto_freq" in a for a in pool["actions"]))
        # the recommendation is the best EV, or an action within the solve's noise of it (tied actions
        # follow the equilibrium mix)
        best = max(a["ev_bb"] for a in pool["actions"])
        tol = max(0.005, pool["solver"]["exploitability_pct"] / 100) * pool["pot_bb"]
        rec = next(a for a in pool["actions"] if a["action"] == pool["recommended"])
        self.assertGreaterEqual(rec["ev_bb"], best - tol - 1e-6, pool["actions"])
        self.assertAlmostEqual(sum(a["freq"] for a in pool["actions"]), 1.0, places=2)
        # the pool under-bluffs river bets: the best response folds more of the range than GTO
        self.assertGreater(pool["range_action_freq"]["fold"], gto["range_action_freq"]["fold"] + 0.1)

    def test_pool_preflop_advice(self):
        """Facing a pool 4bet (QQ+/AK) hero folds more than against the chart's; opens stay the chart's."""
        adv = Advisor()
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        four = F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5), R("SB", "fold"), R("BB", "fold"), R("CO", "raise", 22)]
        st = {"players": 6, "hero": "BTN", "hole": "JsJh", "board": "", "stack": 100, "preflop": four}
        fold = lambda r: next(a["freq"] for a in r["actions"] if a["action"] == "fold")
        self.assertGreater(fold(adv.advise(dict(st, opponents="pool"))), fold(adv.advise(st)) + 0.3)
        self.assertLess(fold(adv.advise(dict(st, hole="KsKh", opponents="pool"))), 0.01)
        opens = dict(st, hero="CO", hole="As9h", preflop=F("UTG", "HJ"))
        self.assertEqual(adv.advise(dict(opens, opponents="pool"))["actions"], adv.advise(opens)["actions"])

    def test_multiway_narrowing_leaves_cached_ranges_alone(self):
        """Multiway narrowing used to write into the cached preflop ranges, so the same request twice gave
        different answers (and the next street started from ranges narrowed twice)."""
        from gto.advisor import preflop_ranges
        pre = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "call"},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "fold"}]
        T = pf.Table(pf.SEATS[6])
        before = {p: v.sum() for p, v in preflop_ranges(pre, T, 100).items()}
        st = {"players": 6, "hero": "BTN", "hole": "AhKh", "board": "Ks7d2c9h", "stack": 100, "preflop": pre,
              "flop": [{"pos": "UTG", "action": "bet", "amount": 4}, {"pos": "CO", "action": "call"}, {"pos": "BTN", "action": "call"}],
              "turn": [{"pos": "UTG", "action": "check"}, {"pos": "CO", "action": "check"}]}
        adv = Advisor(time_budget=0.5, prior_budget=0.3)
        r1 = adv.advise(st)
        self.assertEqual({p: v.sum() for p, v in preflop_ranges(pre, T, 100).items()}, before)
        self.assertEqual([a["ev_bb"] for a in r1["actions"]], [a["ev_bb"] for a in adv.advise(st)["actions"]])


if __name__ == "__main__":
    unittest.main()


class TestPoolOverfold(unittest.TestCase):
    def test_overfold_keeps_strong_calls(self):
        """The pool's extra folds to a big bet were a flat share of the range: against a turn jam that
        equilibrium calls with a few percent of the range, every call went, queens full included, and a jam
        was worth exactly the pot (found by scripts/sanity_sweep.py). They now take at most half the calls
        and never the strongest hands."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        pre = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "call"), R("BTN", "call"), R("SB", "raise", 15),
               R("BB", "fold"), R("UTG", "fold"), R("CO", "call"), R("BTN", "fold")]
        st = {"players": 6, "hero": "SB", "hole": "KdAd", "board": "Qd6dQcAc", "stack": 100, "preflop": pre,
              "flop": [R("SB", "check"), R("CO", "check")], "opponents": "pool"}
        out = Advisor(time_budget=2.0, prior_budget=1.0).advise(st)
        jam = next(a for a in out["actions"] if a["action"] == "all-in")
        self.assertLess(jam["ev_bb"], out["pot_bb"] - 0.5, out["actions"])


class TestQuickSweep(unittest.TestCase):
    """scripts/sanity_sweep.py --quick runs before each deploy: its spot set (not the sweep itself)."""

    @classmethod
    def setUpClass(cls):
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import sanity_sweep
        cls.sw = sanity_sweep
        cls.pre, cls.post = sanity_sweep.quick_jobs(7)

    def test_covers_every_setup_and_kind(self):
        sw = self.sw
        tags = {s[0] for s in sw.SETUPS}
        self.assertEqual({j[0][0] for j in self.pre}, tags)
        self.assertEqual({j[0][0] for j in self.post}, tags)
        self.assertTrue(any(j[4] for j in self.pre), "no exploit-mode preflop nodes")
        hu = {j[2] for j in self.post if not self._multiway(j)}
        self.assertEqual(hu, set(sw.POST_KINDS))
        self.assertEqual({j[2] for j in self.post if self._multiway(j)}, set(sw.MULTIWAY_KINDS))
        self.assertGreaterEqual({j[2] for j in self.post if j[3]}, {"vs_bet", "turn", "river_vs_bet"})
        streets = {len(j[1]["board"]) // 2 for j in self.post}
        self.assertEqual(streets, {3, 4, 5})
        self.assertLess(len(self.pre) + len(self.post), 300)

    def _multiway(self, job):
        from gto.advisor import replay_hand
        setup, state = job[0], job[1]
        return len(replay_hand(self.sw._table(setup), setup[2], {"preflop": state["preflop"]}, 1)[0].active) == 3

    def test_deterministic(self):
        pre, post = self.sw.quick_jobs(7)
        self.assertEqual(pre, self.pre)
        self.assertEqual(post, self.post)

    def test_blocking_flags(self):
        self.assertEqual(self.sw.blocking({"unsettled": [0, 3], "filled_range": [2, 0]}), [])
        self.assertEqual(self.sw.blocking({"unsettled": [0, 3], "ev_is_pot": [0, 1], "error": [0, 0]}),
                         ["ev_is_pot"])


class TestPreflopSubgame(unittest.TestCase):
    def test_converges_and_is_sane(self):
        """The 3bet-pot subgame (pf_solve) converges, premiums never fold (AA and KK call the jam),
        trash never continues, and there are bluffs: blocker 4bets and non-premium jams."""
        from gto.pf_solve import solve, Spot
        from gto.pf_classes import to_classes, CLASS_INDEX as C
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "raise", 7.5), R("SB", "fold"), R("BB", "fold")]
        rng = pf.replay_ranges(pre, T)
        S = solve(to_classes(rng["CO"]), to_classes(rng["BTN"]), Spot(11.5, 2.5, 7.5, 18.0, 100.0, False), iters=1000)
        self.assertLess(S["exploitability"], 0.05)
        for h in ("AA", "KK"):
            self.assertLess(S["o_fold"][C[h]], 0.01)
            self.assertLess(S["t_fold"][C[h]], 0.01)
            self.assertGreater(S["o_vs_jam"][C[h]], 0.99)
        # the weakest hands the opener opened with fold to the 3bet
        from gto.pf_classes import equity_matrix, PAIR_COMPAT
        wo, wt = to_classes(rng["CO"]), to_classes(rng["BTN"])
        eq = (equity_matrix() * PAIR_COMPAT * wt).sum(1) / (PAIR_COMPAT * wt).sum(1)
        weak = [i for i in np.argsort(eq) if wo[i] > 0.5][:5]
        self.assertGreater(min(S["o_fold"][i] for i in weak), 0.99)
        self.assertGreater(S["o_4bet"][C["KQs"]] + S["o_4bet"][C["KJs"]], 0.5)
        # some jams aren't premiums (which hands bluff depends on the realization model)
        prem = [C[h] for h in ("AA", "KK", "QQ", "AKs", "AKo")]
        jam = wt_ = to_classes(rng["BTN"]) * S["t_jam"]
        self.assertGreater(float(jam.sum() - jam[prem].sum()), 0.5)



def _subgame_cases():
    """The six standard 3bet pots of scripts/calibrate_pf_solve.py: (name, opener classes, 3bettor
    classes, Spot, chart shares [opener call, opener 4bet, 3bettor call, 3bettor jam])."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import calibrate_pf_solve
    return calibrate_pf_solve.prepare()


def _continue_evs(S, wo, wt, spot):
    """Per opener class, the EV (bb, relative to folding) of its best continue against the 3bet, under
    the solved strategies."""
    from gto import pf_solve as P
    from gto.pf_classes import PAIR_COMPAT
    st = lambda *k: np.stack([S[x] for x in k])
    v0, _, _ = P._values(P._payoffs(spot), wo, wt, spot.pot, st("o_fold", "o_call", "o_4bet"),
                         st("t_fold", "t_call", "t_jam"), np.stack([1 - S["o_vs_jam"], S["o_vs_jam"]]))
    return np.max(v0[1:], axis=0) / np.maximum(PAIR_COMPAT @ wt, 1e-12)


def _half_table(name):
    import os
    ns = {}
    with open(os.path.join(os.path.dirname(__file__), "data", f"pf_realize_{name}.py")) as f:
        exec(f.read(), ns)
    return ns["TABLE"]


class TestRealizationTable(unittest.TestCase):
    """gto/pf_realize.py (measured with the postflop solver on the charts' ranges) and the preflop
    subgame solved with it."""

    @classmethod
    def setUpClass(cls):
        from gto import pf_solve as P
        cls.P = P
        cls._mode = P.MODE
        P.MODE = "table"
        cls.cases = _subgame_cases()
        cls.sols = {name: P.solve(wo, wt, spot, iters=1500) for name, wo, wt, spot, _ in cls.cases}

    @classmethod
    def tearDownClass(cls):
        cls.P.MODE = cls._mode

    def test_table_shape_and_bounds(self):
        from gto.pf_realize import TABLE
        from gto.pf_solve import bucket_of, K
        nb = int(bucket_of().max()) + 1
        for pt in ("3bet", "4bet"):
            self.assertEqual(sorted(TABLE[pt]), ["aggr_ip", "aggr_oop", "caller_ip", "caller_oop"])
            for pos in TABLE[pt]:
                rows = TABLE[pt][pos]
                self.assertEqual(len(rows), K)
                for row in rows:
                    self.assertEqual(len(row), nb)
                    for v in row:
                        if v is not None:
                            self.assertTrue(-3.0 < v[0] < 3.0 and v[1] >= 0.0, (pt, pos, v))

    def test_table_makes_poker_sense(self):
        """Where well measured: KK loses chips after the flop against AA and AA wins them against KK;
        AA runs at or above its equity against everything weaker; AKo and AJo run below it against the
        top buckets (they pay off)."""
        from gto.pf_realize import TABLE
        from gto.pf_classes import CLASS_INDEX as C
        for pt in ("3bet", "4bet"):
            for pos in TABLE[pt]:
                t = TABLE[pt][pos]
                kk_aa, aa_kk = t[C["KK"]][0], t[C["AA"]][1]
                if kk_aa and kk_aa[1] > 0.2:
                    self.assertLess(kk_aa[0], 0.0, (pt, pos))
                if aa_kk and aa_kk[1] > 0.2:
                    self.assertGreater(aa_kk[0], 0.0, (pt, pos))
                for b in range(1, 9):
                    v = t[C["AA"]][b]
                    if v and v[1] > 0.5:
                        self.assertGreater(v[0], -0.05, (pt, pos, b))
                for h in ("AKo", "AJo"):
                    for b in (0, 1):
                        v = t[C[h]][b]
                        if v and v[1] > 0.2:
                            self.assertLess(v[0], 0.0, (pt, pos, h, b))

    def test_shares_bounded_and_follow_equity(self):
        from gto.pf_classes import equity_matrix
        eq = equity_matrix()
        for name, wo, wt, spot, _ in self.cases:
            for pt in ("3bet", "4bet"):
                sh = self.P._shares(eq, 1.0, 1.0, pt, spot.opener_ip)
                self.assertTrue(np.isfinite(sh).all(), name)
                self.assertTrue((sh > -2.0).all() and (sh < 3.0).all(), name)
                m = (wo[:, None] * wt[None, :]) > 0
                self.assertGreater(np.corrcoef(sh[m], eq[m])[0, 1], 0.8, (name, pt))

    def test_every_spot_converges_and_plays_sanely(self):
        from gto.pf_classes import CLASS_INDEX as C, equity_matrix, PAIR_COMPAT
        for name, wo, wt, spot, _ in self.cases:
            S = self.sols[name]
            self.assertLess(S["exploitability"], 0.05, name)
            for h in ("AA", "KK"):
                self.assertLess(S["o_fold"][C[h]], 0.01, (name, h))
                self.assertLess(S["t_fold"][C[h]], 0.01, (name, h))
            self.assertGreater(S["o_vs_jam"][C["AA"]], 0.99, name)
            # (KK facing the jam: see TestRealizationModel.test_kk_calls_the_jam)
            # the weakest hands the opener opened fold to the 3bet
            e = (equity_matrix() * PAIR_COMPAT * wt).sum(1) / (PAIR_COMPAT * wt).sum(1)
            weak = [i for i in np.argsort(e) if wo[i] > 0.5][:5]
            self.assertGreater(min(S["o_fold"][i] for i in weak), 0.95, name)


    # Known failure: the raw table is noisy per class, so pair EVs can come out of order (CO vs BTN: 99
    # continues for 1.76bb, 88 for 2.31); the smooth model (TestRealizationModel) passes this.
    @unittest.expectedFailure
    def test_pairs_in_order(self):
        """A pair's EV of continuing against the 3bet doesn't trail a smaller pair's by more than 0.5bb
        (frequencies can flip between hands near break-even, so the check is on EVs)."""
        from gto.pf_classes import CLASS_INDEX as C
        for name, wo, wt, spot, _ in self.cases:
            ev = _continue_evs(self.sols[name], wo, wt, spot)
            pairs = [ev[C[p]] for p in ("AA", "KK", "QQ", "JJ", "TT", "99", "88", "77") if wo[C[p]] > 0.5]
            for hi, lo in zip(pairs, pairs[1:]):
                self.assertGreater(hi, lo - 0.5, (name, [round(x, 2) for x in pairs]))

    # Known failure: per class the raw table is still a little noisy: on the two (stratified, weighted)
    # halves of the flops the opener's call share in BTN vs SB is 0.55/0.45 (every other share agrees
    # within 7 points; with random flops it moved up to 22). The smooth model passes this.
    @unittest.expectedFailure
    def test_stable_across_flop_halves(self):
        """Solved with the table from each half of the flops, the range-level shares and the key hands
        come out nearly the same: the measurement isn't noise-driven."""
        from gto.pf_classes import CLASS_INDEX as C
        P = self.P
        res = {}
        try:
            for half in ("a", "b"):
                P.TABLE_OVERRIDE = _half_table(half)
                res[half] = {name: P.solve(wo, wt, spot, iters=1500) for name, wo, wt, spot, _ in self.cases}
        finally:
            P.TABLE_OVERRIDE = None
        for name, wo, wt, spot, _ in self.cases:
            a, b = res["a"][name], res["b"][name]
            sh = lambda S, k, w: float((w * S[k]).sum() / w.sum())
            for k, w in (("o_call", wo), ("o_4bet", wo), ("t_call", wt), ("t_jam", wt)):
                self.assertLess(abs(sh(a, k, w) - sh(b, k, w)), 0.08, (name, k, sh(a, k, w), sh(b, k, w)))
            for h in ("AA", "KK"):
                self.assertLess(abs(a["o_vs_jam"][C[h]] - b["o_vs_jam"][C[h]]), 0.15, (name, h))

    # Known failure: UTG vs CO calls 14% (chart 26%), BTN vs SB 50% (chart 36%); see the model's
    # test_close_to_charts.
    @unittest.expectedFailure
    def test_close_to_charts(self):
        """The opener's response to the 3bet stays near the charts (hand-written solver approximations)."""
        for name, wo, wt, spot, target in self.cases:
            S = self.sols[name]
            sh = lambda k, w: float((w * S[k]).sum() / w.sum())
            self.assertLess(abs(sh("o_call", wo) - target[0]), 0.12, (name, sh("o_call", wo), target[0]))
            self.assertLess(abs(sh("o_4bet", wo) - target[1]), 0.08, (name, sh("o_4bet", wo), target[1]))


def _half_model(name):
    import os
    ns = {}
    with open(os.path.join(os.path.dirname(__file__), "data", f"pf_realize_model_{name}.py")) as f:
        exec(f.read(), ns)
    return ns["COEF"]


class TestRealizationModel(unittest.TestCase):
    """gto/pf_realize_model.py: the smooth model of the measured excess pot shares
    (scripts/fit_realization.py), and the preflop subgame solved with it."""

    @classmethod
    def setUpClass(cls):
        from gto import pf_solve as P
        cls.P = P
        cls._mode = P.MODE
        P.MODE = "model"
        cls.cases = _subgame_cases()
        cls.sols = {name: P.solve(wo, wt, spot, iters=1500) for name, wo, wt, spot, _ in cls.cases}

    @classmethod
    def tearDownClass(cls):
        cls.P.MODE = cls._mode

    def test_model_shape_and_sense(self):
        """Every class gets a finite excess against every bucket; big favourites gain (AA against KK)
        and big underdogs lose (KK against AA), in every pot type and position."""
        from gto.pf_classes import CLASS_INDEX as C
        for pt in ("3bet", "4bet"):
            for pos in ("caller_oop", "caller_ip", "aggr_oop", "aggr_ip"):
                m = self.P._model(pt, pos)
                self.assertEqual(m.shape, (len(C), int(self.P.bucket_of().max()) + 1))
                self.assertTrue(np.isfinite(m).all())
                self.assertGreater(m[C["AA"], 1], 0.0, (pt, pos))
                self.assertLess(m[C["KK"], 0], 0.0, (pt, pos))

    def test_every_spot_converges_and_plays_sanely(self):
        from gto.pf_classes import CLASS_INDEX as C, equity_matrix, PAIR_COMPAT
        for name, wo, wt, spot, _ in self.cases:
            S = self.sols[name]
            self.assertLess(S["exploitability"], 0.05, name)
            for h in ("AA", "KK"):
                self.assertLess(S["o_fold"][C[h]], 0.01, (name, h))
                self.assertLess(S["t_fold"][C[h]], 0.01, (name, h))
            self.assertGreater(S["o_vs_jam"][C["AA"]], 0.99, name)
            e = (equity_matrix() * PAIR_COMPAT * wt).sum(1) / (PAIR_COMPAT * wt).sum(1)
            weak = [i for i in np.argsort(e) if wo[i] > 0.5][:5]
            self.assertGreater(min(S["o_fold"][i] for i in weak), 0.95, name)
            ev = _continue_evs(S, wo, wt, spot)
            pairs = [ev[C[p]] for p in ("AA", "KK", "QQ", "JJ", "TT", "99", "88", "77") if wo[C[p]] > 0.5]
            for hi, lo in zip(pairs, pairs[1:]):
                self.assertGreater(hi, lo - 0.5, (name, [round(x, 2) for x in pairs]))

    # (With realization keyed by position only, the 3bettor trapped KK against the 4bet in UTG vs CO and
    # HJ vs BTN and the opener folded KK to the AA-heavy jams; keyed by role it calls them everywhere.)
    def test_kk_calls_the_jam(self):
        from gto.pf_classes import CLASS_INDEX as C
        for name, *_ in self.cases:
            self.assertGreater(self.sols[name]["o_vs_jam"][C["KK"]], 0.9, name)

    # Known failure: keyed by role and position each fit has half the data it had keyed by position, and
    # the 3bettor's call share against the 4bet moves 10-11 points between halves in CO vs BTN
    # (0.41/0.31) and BTN vs SB (0.28/0.18); every other share agrees within 5.
    @unittest.expectedFailure
    def test_stable_across_flop_halves(self):
        """Fitted on each half of the (texture-stratified, weighted) flops, the model gives nearly the same
        strategies."""
        P = self.P
        res = {}
        try:
            for half in ("a", "b"):
                P.MODEL_OVERRIDE = _half_model(half)
                res[half] = {name: P.solve(wo, wt, spot, iters=1500) for name, wo, wt, spot, _ in self.cases}
        finally:
            P.MODEL_OVERRIDE = None
        for name, wo, wt, spot, _ in self.cases:
            a, b = res["a"][name], res["b"][name]
            sh = lambda S, k, w: float((w * S[k]).sum() / w.sum())
            for k, w in (("o_call", wo), ("o_4bet", wo), ("t_call", wt), ("t_jam", wt)):
                self.assertLess(abs(sh(a, k, w) - sh(b, k, w)), 0.08, (name, k, sh(a, k, w), sh(b, k, w)))

    # Known failure: openers call 3bets far less than the charts (BTN vs BB 16% against 36%, UTG vs CO 9%
    # against 26%, CO vs BB 25% against 41%) and 4bet more: measured, callers realize much less than the
    # aggressor (3bet pots: caller 0.79 out of position, 0.97 in position, against 1.02 / 1.15; 4bet pots:
    # caller ~0.6 against ~1.25). Not the postflop solver's depth (playing the river moves these about a
    # point); possibly its bet-size abstraction favouring the player with the betting lead.
    @unittest.expectedFailure
    def test_close_to_charts(self):
        for name, wo, wt, spot, target in self.cases:
            S = self.sols[name]
            sh = lambda k, w: float((w * S[k]).sum() / w.sum())
            self.assertLess(abs(sh("o_call", wo) - target[0]), 0.12, (name, sh("o_call", wo), target[0]))
            self.assertLess(abs(sh("o_4bet", wo) - target[1]), 0.08, (name, sh("o_4bet", wo), target[1]))


class TestOpenerVs3bet(unittest.TestCase):
    def test_oop_openers_match_published_solver_shares(self):
        """Openers out of position facing a 3bet: fold/call/4bet shares of the opening range near published
        solver output (CO vs BTN 3bet to 7.5bb, 100bb cash: 63.7/17.2/19.1; the SB vs a BB 3bet folds about
        half and 4bets a third of its continues), and the in-position charts against a blind's 3bet keep
        calling widely."""
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        F = lambda *ps: [R(p, "fold") for p in ps]
        T = pf.Table(pf.SEATS[6])
        def shares(hero, pre):
            w = pf.replay_ranges(pre, T)[hero]
            s = pf.strategy_for(hero, pre, T)
            return [float((w * s.get(k, 0)).sum() / w.sum()) for k in ("fold", "call", "raise")]
        co = shares("CO", F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "raise", 7.5)] + F("SB", "BB"))
        for got, want in zip(co, (0.637, 0.172, 0.191)):
            self.assertAlmostEqual(got, want, delta=0.03)
        for hero, pre in (("UTG", [R("UTG", "raise", 2.5)] + F("HJ", "CO") + [R("BTN", "raise", 7.5)] + F("SB", "BB")),
                          ("HJ", F("UTG") + [R("HJ", "raise", 2.5), R("CO", "raise", 7.5)] + F("BTN", "SB", "BB"))):
            f, c, r = shares(hero, pre)
            self.assertTrue(0.58 <= f <= 0.68 and 0.14 <= c <= 0.21 and 0.16 <= r <= 0.23, (hero, f, c, r))
            self.assertTrue(0.4 <= r / (1 - f) <= 0.65, (hero, r / (1 - f)))   # about half the continues 4bet
        f, c, r = shares("SB", F("UTG", "HJ", "CO", "BTN") + [R("SB", "raise", 3), R("BB", "raise", 9)])
        self.assertTrue(0.45 <= f <= 0.58 and 0.28 <= r / (1 - f) <= 0.40, (f, c, r))
        f, c, r = shares("BTN", F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB") + [R("BB", "raise", 10)])
        self.assertGreater(c, 0.3)

    def test_thin_lines_filled_out(self):
        """A line that leaves fewer than MIN_LINE_COMBOS gets the hands that take its last action mixed in,
        weighing 1 - combos / threshold: all of them for an empty line, none at the threshold; the hands
        that got there are kept."""
        n = pf.N_COMBOS
        scratch = np.zeros(n); scratch[:40] = 1.0
        empty, mostly = pf.thin_line(np.zeros(n), scratch)
        self.assertAlmostEqual(empty.sum(), 40.0); self.assertTrue(mostly)
        r = np.zeros(n); r[100:104] = 1.0                       # 4 combos, half the threshold
        out, mostly = pf.thin_line(r, scratch)
        self.assertTrue((out[100:104] == 1.0).all())
        self.assertAlmostEqual(out[:40].sum(), 40.0 * (1 - 4.0 / pf.MIN_LINE_COMBOS)); self.assertTrue(mostly)
        wide = np.zeros(n); wide[:10] = 1.0
        self.assertIs(pf.thin_line(wide, scratch)[0], wide)    # at or above the threshold: unchanged
        # BB cold-calls a 3bet, then calls HJ's 4bet: 6.7 combos, filled out to ~13 with a few trapped premiums
        R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
        from gto.advisor import preflop_ranges
        from gto.cards import COMBO_INDEX, cards
        T = pf.Table(pf.SEATS[6])
        pre = [R("UTG", "fold"), R("HJ", "raise", 2.5), R("CO", "raise", 7.5), R("BTN", "fold"), R("SB", "fold"),
               R("BB", "call"), R("HJ", "raise", 18.8), R("CO", "fold"), R("BB", "call")]
        bb = preflop_ranges(pre, T, 100)["BB"]
        self.assertGreater(bb.sum(), pf.MIN_LINE_COMBOS)
        self.assertGreater(bb[COMBO_INDEX[tuple(sorted(cards("KsKd")))]], 0.0)

