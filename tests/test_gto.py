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
        self.assertAlmostEqual(sum(a["freq"] for a in r["actions"]), 1.0, places=2)
        r = adv.advise({"hero": "BB", "hole": "9h8h", "board": "Kd7s2cTh", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 2},
                                 {"pos": "BB", "action": "call"}], "turn": []})
        self.assertEqual(r["street"], "turn")
        self.assertAlmostEqual(r["pot_bb"], 9.5)


if __name__ == "__main__":
    unittest.main()


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

    def test_multiway_and_mid_street_hu(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "CO", "hole": "AhAd", "board": "Kh7c2d", "preflop": pre, "flop": [{"pos": "BB", "action": "check"}]})
        self.assertTrue(r.get("multiway"))
        self.assertAlmostEqual(r["pot_bb"], 8.0)
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "KsQs", "board": "Kh7c2d", "preflop": pre,
                        "flop": [{"pos": "BB", "action": "bet", "amount": 4}, {"pos": "CO", "action": "fold"}]})
        self.assertFalse(r.get("multiway"))
        self.assertEqual(r["villain"], "BB")
        self.assertAlmostEqual(r["pot_bb"], 12.0)
        self.assertEqual(r["actions"][0]["action"], "fold")
