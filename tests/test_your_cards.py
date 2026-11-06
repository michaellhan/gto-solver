"""Your cards aren't public: nobody's range or line depends on them. They come out only where your hand is counted
against the ranges, in the solve's row for it (card removal) and in what the answer reports."""
import unittest
from unittest import mock

import numpy as np

from gto import multiway_solver as MW
from gto import advisor as A
from gto.advisor import MEMBER_FULL
from gto.cards import COMBO_INDEX, cards
from gto.native import advisor

FOLDS = lambda *seats: [{"pos": s, "action": "fold"} for s in seats]
# the hand that found it: 7-handed, a 3bet pot three-way, where TT is a fifth of CO's line
PRE7 = FOLDS("UTG", "LJ", "HJ") + [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "raise", "to": 7.5},
                                   {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"},
                                   {"pos": "CO", "action": "call"}]
FLOP7 = [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 7.5}, {"pos": "BTN", "action": "call"},
         {"pos": "BB", "action": "call"}]
TURN7 = [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 34.0}]
SEVEN = dict(players=7, hero="BTN", board="Ks7s2h2c", stack=100, preflop=PRE7, flop=FLOP7, turn=TURN7)
# heads-up: a 4bet pot's turn bet, BTN to act
PRE_4B = FOLDS("UTG", "LJ", "HJ") + [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "raise", "to": 7.5}] \
    + FOLDS("SB", "BB") + [{"pos": "CO", "action": "raise", "to": 19}, {"pos": "BTN", "action": "call"}]
FOUR_BET = dict(players=6, hero="BTN", board="Ks7s2h2c", stack=100, preflop=PRE_4B,
                flop=[{"pos": "CO", "action": "bet", "amount": 10.0}, {"pos": "BTN", "action": "call"}],
                turn=[{"pos": "CO", "action": "bet", "amount": 25.0}])


def spy_multiway(state):
    """The ranges a fresh Fast advice hands the multiway solver, by seat order, and its answer."""
    seen = []
    real = MW.MultiwaySolver.__init__

    def init(self, board, ranges, *a, **k):
        seen.append([np.array(r, dtype=float) for r in ranges])
        return real(self, board, ranges, *a, **k)
    with mock.patch.object(MW.MultiwaySolver, "__init__", init):
        res = advisor("fast").advise(dict(state))
    return seen[-1], res


class TestLinesDontDependOnYourCards(unittest.TestCase):
    def test_multiway_solve_gets_the_same_ranges_whatever_you_hold(self):
        # with your cards taken out before the rules and the floor read the line, CO's turn range was 11.3 combos
        # holding TT where it's 9.7 less what TT blocks: more filler, less value, a call that isn't there
        tt, res = spy_multiway(dict(SEVEN, hole="ThTs"))
        ak, _ = spy_multiway(dict(SEVEN, hole="AhKd"))
        self.assertIn("multiway", res)
        for i in (0, 1):    # BB and CO; BTN's own is the same anyway
            np.testing.assert_array_equal(tt[i], ak[i])

    def test_the_floor_reads_the_whole_line(self):
        # the combos a line left, and so how much filler it gets, are the player's, not what's left of them after you
        notes = lambda h: [n for n in advisor("fast").advise(dict(SEVEN, hole=h))["notes"] if "filled out" in n]
        self.assertEqual(notes("ThTs"), notes("AhKd"))
        self.assertTrue(notes("ThTs"))

    def test_heads_up_solve_gets_the_same_ranges_whatever_you_hold(self):
        adv = advisor("fast")
        seen = {}
        real = adv._solve_cached

        def solve(board, r0, r1, *a, **k):
            seen.setdefault(len(board), []).append((np.array(r0, dtype=float), np.array(r1, dtype=float)))
            return real(board, r0, r1, *a, **k)
        adv._solve_cached = solve
        adv.advise(dict(FOUR_BET, hole="TdTc"))
        adv.advise(dict(FOUR_BET, hole="QdQc"))
        (a0, a1), (b0, b1) = seen[4][0], seen[4][-1]
        np.testing.assert_array_equal(a0, b0)      # CO's
        np.testing.assert_array_equal(a1, b1)      # and BTN's


    def test_one_size_rows_are_each_hands_own(self):
        # one solve serves every hand of a node: the view's rows, which hold a hand's line, can't
        river = dict(players=6, hero="BB", board="Ks7d2cTh3s", stack=100, turn=[{"pos": "BB", "action": "check"},
                                                                                 {"pos": "SB", "action": "check"}],
                     preflop=FOLDS("UTG", "LJ", "HJ", "CO", "BTN") + [{"pos": "SB", "action": "call"},
                                                                      {"pos": "BB", "action": "check"}],
                     flop=[{"pos": "SB", "action": "check"}, {"pos": "BB", "action": "check"}],
                     river=[{"pos": "SB", "action": "check"}])
        adv = advisor("fast")
        a = adv.advise(dict(river, hole="Kh9h", one_size="50%"))["one_size"]["rows"][0]["hand"]
        b = adv.advise(dict(river, hole="9c8c", one_size="50%"))["one_size"]["rows"][0]["hand"]
        self.assertNotEqual(a["ev"], b["ev"])


class TestYourCardsStillBlock(unittest.TestCase):
    def test_multiway_report_leaves_out_what_you_block(self):
        rs, res = spy_multiway(dict(SEVEN, hole="ThTs"))
        board = cards("Ks7s2h2c")
        live = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        off = np.prod([1.0 - MEMBER_FULL[c] for c in cards("ThTs")], axis=0)
        self.assertAlmostEqual(res["opponent_range_combos"]["CO"], float((rs[1] * live * off).sum()), places=1)
        self.assertLess(res["opponent_range_combos"]["CO"], float((rs[1] * live).sum()) - 1.0)

    def test_heads_up_report_leaves_out_what_you_block(self):
        with mock.patch("gto.advisor.range_to_text", wraps=A.range_to_text) as shown:
            res = advisor("fast").advise(dict(FOUR_BET, hole="TdTc"))
        r = shown.call_args_list[-1].args[0]
        for c in cards("TdTc"):
            self.assertEqual(float(r[MEMBER_FULL[c] > 0].sum()), 0.0)
        self.assertAlmostEqual(res["villain_range_combos"], float(r.sum()), places=1)
        self.assertGreater(res["villain_range_combos"], 0.0)

    def test_a_range_you_block_whole(self):
        # nothing to count your hand against: the hand still answers
        pre = FOLDS("UTG", "LJ", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 2.5}] + FOLDS("SB") \
            + [{"pos": "BB", "action": "call"}]
        res = advisor("fast").advise(dict(players=6, hero="BTN", hole="ThTs", board="Ks7s2h", stack=100, preflop=pre,
                                          flop=[{"pos": "BB", "action": "check"}],
                                          ranges={"BB": "ThTc,ThTd,TsTc,TsTd,ThTs"}))
        self.assertNotIn("error", res)
        self.assertEqual(res["villain_range_combos"], 0.0)


class TestTheHand(unittest.TestCase):
    def test_tens_dont_read_a_game_where_co_cant_hold_tens(self):
        # With the opponents' ranges conditioned on TT, BTN's whole range played as if CO held none: its shove 21 combos
        # of 94 where it's 31 (AKo 2.7 of 6.4), so CO's JJ called 30% of the time, not 71%. With the lines read after
        # your cards, call +1.4 (100%) and all-in -0.8; in the game itself about 0 and -14.
        res = advisor("fast").advise(dict(SEVEN, hole="ThTs"))
        ev = {a["action"]: a["ev_bb"] for a in res["actions"]}
        self.assertLess(ev["call"], 0.5)
        self.assertLess(ev["all-in"], -8.0)
        self.assertEqual(res["recommended"], "fold")


class TestHandOutOfYourRange(unittest.TestCase):
    """The hand that found it: 7-handed, A4s calling a raise the charts fold, 4-way. With no weight its later decisions
    had no average strategy, so its EVs read uniform play there: check 1.82, bet 3.75, the bet "0.00 bb lost" at 0%."""
    PRE = [{"pos": "UTG", "action": "call"}, {"pos": "LJ", "action": "raise", "to": 4}, {"pos": "HJ", "action": "call"}] \
        + FOLDS("CO", "BTN", "SB") + [{"pos": "BB", "action": "call"}, {"pos": "UTG", "action": "call"}]
    ST = {"players": 7, "hero": "BB", "board": "2s4d6c", "stack": 100, "preflop": PRE, "flop": [], "turn": [], "river": []}

    def test_its_later_decisions_are_played(self):
        adv = advisor("fast")
        adv.mw_budget = 10
        res = adv.advise(dict(self.ST, hole="As4s"))
        self.assertTrue(res.get("solved"), res.get("notes"))
        self.assertTrue(any("near-zero-weight" in n for n in res["notes"]), res["notes"])
        sol = list(adv._mw_cache.values())[-1]
        hp, ci = sol.root.player, sol.combo_index(cards("As4s"))
        self.assertAlmostEqual(float(sol.reach0[hp, 0, ci]), A.MW_OFF_RANGE, delta=1e-11)
        def decisions(node):
            if isinstance(node, MW.Decision):
                yield node
                for ch in node.children:
                    yield from decisions(ch)
        later = [n for n in decisions(sol.root) if n is not sol.root and n.player == hp and n.bd is sol.root.bd]
        self.assertTrue(later)
        for n in later:   # (uniform is what a hand with no strategy sums reads)
            self.assertGreater(float(n.ssum[0, :, ci].sum()), 0.0)

    def test_two_such_hands_dont_share_a_solve(self):
        adv = advisor("fast")
        adv.mw_budget = 1
        for hole in ("As4s", "Qh3d"):
            adv.advise(dict(self.ST, hole=hole))
        self.assertEqual(len(adv._mw_cache), 2)


class TestNoHandIsPricedAsRandomPlay(unittest.TestCase):
    """The rule under TestHandOutOfYourRange: a hand with no strategy sums at a decision plays its current strategy,
    in the solver's own pass and the compiled core's, so nothing that reads an average or an EV gets uniform play."""
    def solve(self):
        # BB's range hasn't got AsKs; another's has, so the solve carries the hand with no weight for BB
        board = cards("2s4d6c9hTd")
        rs = [np.ones(1326) for _ in range(3)]
        i = COMBO_INDEX[tuple(sorted(cards("AsKs")))]
        rs[0][i] = 0.0
        sol = MW.MultiwaySolver(board, rs, 10.0, 90.0)
        sol.run(iters=60)
        return sol, 0, sol.combo_index(cards("AsKs"))

    def test_a_hand_with_no_sums_plays_its_current_strategy(self):
        sol, p, ci = self.solve()
        seen = 0
        def walk(n):
            nonlocal seen
            if not isinstance(n, MW.Decision):
                return
            if n.player == p and float(n.ssum[0, :, ci].sum()) == 0.0:
                seen += 1
                np.testing.assert_allclose(n.avg_strategy()[0][:, ci], n.strategy()[0][:, ci])
            for ch in n.children:
                walk(ch)
        walk(sol.root)
        self.assertGreater(seen, 0)

    def test_the_core_and_numpy_price_it_the_same(self):
        sol, p, ci = self.solve()
        core = sol.action_evs()[:, ci]
        with mock.patch.object(type(sol), "_core", lambda self: None):
            np.testing.assert_allclose(sol.action_evs()[:, ci], core, atol=0.02)


class TestHeadsUpHandOutOfItsRange(unittest.TestCase):
    def test_it_is_in_the_solve_and_every_later_decision_of_it_has_a_strategy(self):
        adv = advisor("fast", library=False)
        st = {"players": 6, "hero": "BB", "hole": "7h2d", "board": "Ks8d3c", "stack": 100,
              "preflop": FOLDS("UTG", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 2.5}]
              + FOLDS("SB") + [{"pos": "BB", "action": "call"}], "flop": [], "turn": [], "river": []}
        res = adv.advise(st)
        self.assertTrue(any("near-zero-weight" in n for n in res["notes"]), res["notes"])
        evs = [a["ev_bb"] for a in res["actions"]]
        self.assertTrue(all(e is not None and abs(e) < 50 for e in evs), res["actions"])


if __name__ == "__main__":
    unittest.main()
