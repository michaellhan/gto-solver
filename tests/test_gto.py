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
        self.assertLess(ranges["BB"][sets] / before["BB"][sets], 0.2)      # and doesn't check

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
