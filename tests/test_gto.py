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
        # a huge 3bet: a chart call (88 mixes vs a standard 3bet) folds at the worse price
        big = [R("UTG", "raise", 2.5), R("HJ", "fold"), R("CO", "raise", 20)] + [R(p, "fold") for p in ("BTN", "SB", "BB")]
        self.assertGreater(adv.advise(dict(st, hole="8s8d", preflop=std))["actions"][1]["freq"], 0.5)
        self.assertEqual(adv.advise(dict(st, hole="8s8d", preflop=big))["recommended"], "fold")

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
            ("CO", co + [R("BTN", "raise", 7.5)] + F("SB", "BB"), 0.38, 0.52),
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

    def test_pool_lock_folds_weakest_to_big_bets_and_calls_small_ones(self):
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
        self.assertAlmostEqual(fold - 0.5, 0.15, places=2)       # a 1.5x-pot river overbet: 15% more of the range folds
        pct = lock.pct(board)
        self.assertGreater(big[0, 0][pct < 0.2].mean(), big[0, 0][pct > 0.8].mean())   # ...the weakest calls
        flop = tuple(cards("Qs7c3d"))
        small = PoolLock(Ctx(np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in flop and b not in flop]), Config()), 1)
        n2 = len(small.ctx.active)
        out = small(Node(2.0, 6.0, flop), np.full((1, 2, n2), 0.5, dtype=np.float32), np.ones((1, n2)))
        self.assertLess(float(out[0, 0].mean()), 0.5)             # a third-pot flop bet: fewer folds
        raised = Node(10.0, 10.0); raised.bets = (20.0, 10.0)    # facing a raise of its own bet: no shift
        self.assertAlmostEqual(float(lock(raised, strat, reach)[0, 0].mean()), 0.5, places=5)

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
        best = max(pool["actions"], key=lambda a: a["ev_bb"])
        self.assertEqual(pool["recommended"], best["action"])
        self.assertEqual(sorted(a["freq"] for a in pool["actions"])[-1], 1.0)
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
