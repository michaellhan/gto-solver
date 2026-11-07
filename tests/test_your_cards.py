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


class TestAHandOffItsRangeMovesNoRange(unittest.TestCase):
    """A hand the line's range doesn't hold is the case the tests above leave out, and the one a repair would want to
    touch: "you played A4s, so your range holds it" is the rule that hands your cards to the opponents. Your hand goes
    into its own row at a near-zero weight and nothing else moves. scripts/bench_off_range.py, docs/off_range_research.md."""
    BTN_BB = dict(players=6, hero="BB", board="Ks8d3c2h", stack=100,
                  preflop=FOLDS("UTG", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 2.5}] + FOLDS("SB")
                  + [{"pos": "BB", "action": "call"}],
                  flop=[{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 2.0},
                        {"pos": "BB", "action": "call"}])

    def test_multiway_ranges_are_the_same_in_range_or_out(self):
        off, res = spy_multiway(dict(SEVEN, hole="7h2d"))       # (BTN 3bet: not this)
        self.assertTrue(any("near-zero-weight" in n for n in res["notes"]), res["notes"])
        ak, _ = spy_multiway(dict(SEVEN, hole="AhKd"))
        for i in (0, 1):    # BB and CO
            np.testing.assert_array_equal(off[i], ak[i])
        d = np.nonzero(off[2] != ak[2])[0]
        self.assertEqual(d.tolist(), [COMBO_INDEX[tuple(sorted(cards("7h2d")))]])   # (BTN's: that one row)

    def test_heads_up_it_is_a_row_in_the_solve_not_a_range(self):
        # The street before was solved with the hand's row in (Advisor._as_advised), and any added row runs that solve a
        # little differently: 4-5 combos of the BTN's 242 at a weight of 1e-3, 1e-6 or 2e-9 alike, for AA and 72o alike. So
        # it isn't the hand's weight or strength reaching the range. It isn't nothing, so it is held here: under 3%
        adv = advisor("fast", budget=20, prior_budget=20, library=False)
        adv.hu_settled = 1.0     # (the same iterations each: a solve's budget would measure the load)
        seen = {}
        real = adv._solve_cached

        def solve(board, r0, r1, *a, **k):
            seen.setdefault(len(board), []).append((np.array(r0, dtype=float), np.array(r1, dtype=float)))
            return real(board, r0, r1, *a, **k)
        adv._solve_cached = solve
        res = adv.advise(dict(self.BTN_BB, hole="AhAd"))         # (BB 3bets aces: not a call)
        self.assertTrue(any("near-zero-weight" in n for n in res["notes"]), res["notes"])
        adv.advise(dict(self.BTN_BB, hole="QhJh"))
        (o0, o1), (i0, i1) = seen[4][0], seen[4][-1]
        hand = COMBO_INDEX[tuple(sorted(cards("AhAd")))]
        self.assertAlmostEqual(float(o0[hand]), 1e-3)            # (BB's own row, as it stands)
        self.assertEqual(float(i0[hand]), 0.0)
        o0 = o0.copy(); o0[hand] = 0.0
        self.assertLess(float(np.abs(o0 - i0).sum()), 0.03 * float(i0.sum()))
        self.assertLess(float(np.abs(o1 - i1).sum()), 0.03 * float(i1.sum()))


class TestANarrowFlopKeepsABucket(unittest.TestCase):
    """Fast's flop plays the turn and river by hand buckets, equal shares of a range's weight, so a hand with no weight has
    no value of its own: above a 3bet or 4bet pot's range it played as the hands around it. There it has a bucket kept for
    it (Config.near_bucket), which a flop takes by the ranges' sizes alone (Advisor._flop_model), so your cards can't choose
    the model, and the range's own buckets are the same with the hand or without."""
    PRE = FOLDS("UTG", "HJ", "CO") + [{"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"},
                                      {"pos": "BB", "action": "raise", "to": 10.0}, {"pos": "BTN", "action": "raise", "to": 23.0},
                                      {"pos": "BB", "action": "call"}]

    def test_by_the_ranges_sizes(self):
        cfg = advisor("fast", library=False).cfg
        self.assertTrue(cfg.turn_buckets)
        self.assertFalse(cfg.near_bucket)
        flop = cards("8s8d3c")
        wide, narrow = np.ones(1326) * 0.5, np.zeros(1326)
        narrow[:int(A.NARROW_FLOP) - 1] = 1.0
        for pair in ((wide, narrow), (narrow, wide), (narrow, narrow)):
            self.assertTrue(A.Advisor._flop_model(flop, *pair, cfg).near_bucket)
        self.assertIs(A.Advisor._flop_model(flop, wide, wide, cfg), cfg)
        self.assertIs(A.Advisor._flop_model(cards("8s8d3c2h"), narrow, narrow, cfg), cfg)    # (the flop's model only)
        acc = advisor("accurate", library=False).cfg
        self.assertIs(A.Advisor._flop_model(flop, narrow, narrow, acc), acc)                 # (no buckets)

    def test_a_near_zero_row_counts_for_nothing(self):
        cfg = advisor("fast", library=False).cfg
        flop, wide, r = cards("8s8d3c"), np.ones(1326), np.zeros(1326)
        r[:int(A.NARROW_FLOP)] = 1.0                 # (just not narrow)
        self.assertIs(A.Advisor._flop_model(flop, r, wide, cfg), cfg)
        r[0] = 0.0                                   # (narrow by one combo; a hand off the range at 1e-3 doesn't lift it)
        r[1000] = 1e-3
        self.assertTrue(A.Advisor._flop_model(flop, r, wide, cfg).near_bucket)

    def tables(self, hole=None):
        """A 4bet pot's flop, as the Advisor would solve it, and its bucket table on a turn."""
        adv = advisor("fast", budget=2, prior_budget=2, library=False)
        seen = {}
        real = A.solve

        def spy(board, r0, r1, *a, **k):
            seen["sol"] = real(board, r0, r1, *a, **k)
            return seen["sol"]
        with mock.patch.object(A, "solve", spy):
            adv.advise(dict(players=6, hero="BB", board="8s8d3c", stack=100, preflop=self.PRE, flop=[], hole=hole))
        return seen["sol"]

    def test_the_ranges_own_buckets_dont_move(self):
        out, on = self.tables("Ad8h"), self.tables("QhQd")           # (trips: off the range; queens: in it)
        self.assertTrue(out.ctx.cfg.near_bucket and on.ctx.cfg.near_bucket)
        i = COMBO_INDEX[tuple(sorted(cards("Ad8h")))]
        ca, cb = list(out.ctx.active), list(on.ctx.active)
        turn = cards("8s8d3c2h")
        K = out.ctx.cfg.river_buckets
        ia, Ea, ka = out.ctx.bucket_table(tuple(turn), set(), K)
        ib, Eb, kb = on.ctx.bucket_table(tuple(turn), set(), K)
        self.assertEqual(ia[0, ca.index(i)], K - 1)
        self.assertLess(int(ib[0, ib[0] < K].max()), K - 1)          # (empty without such a hand)
        kept = ia[:, np.array([ca.index(c) for c in cb])].copy()
        kept[0][np.array(cb) == i] = K                           # (BB's own row: the one difference)
        ib = ib.copy()
        ib[0][np.array(cb) == i] = K
        self.assertTrue(np.array_equal(kept, ib))
        np.testing.assert_array_equal(Ea[:, :K - 1, :K - 1], Eb[:, :K - 1, :K - 1])
        self.assertGreater(float(Ea[0, K - 1, :K - 1].max()), 0.5)   # (trips against the BTN's range: its own value)

    def test_the_kept_bucket_is_the_same_without_the_core(self):
        sol = self.tables("Ad8h")
        ctx, turn, K = sol.ctx, tuple(cards("8s8d3c2h")), sol.ctx.cfg.river_buckets
        ids, E, kap = ctx.bucket_table(turn, set(), K)
        held = dict(ctx._stacks)
        with mock.patch.object(ctx, "tables", None), mock.patch.object(ctx, "_stacks", {}):
            i2, E2, k2 = ctx.bucket_table(turn, set(), K)
        self.assertEqual(held.keys(), ctx._stacks.keys())
        self.assertTrue(np.array_equal(ids, i2))
        np.testing.assert_allclose(E, E2, atol=1e-5)
        np.testing.assert_allclose(kap, k2, atol=1e-5)

    def test_trips_off_a_4bet_pots_range_dont_shove(self):
        adv = advisor("fast", budget=5, prior_budget=2, library=False)
        res = adv.advise(dict(players=6, hero="BB", board="8s8d3c", stack=100, preflop=self.PRE, flop=[], hole="Ad8h"))
        self.assertTrue(any("near-zero-weight" in n for n in res["notes"]), res["notes"])
        ev = {a["action"]: a["ev_bb"] for a in res["actions"]}
        self.assertNotEqual(max(ev, key=ev.get), "all-in", res["actions"])
        self.assertGreater(ev["check"], 75.0)        # (65.7 by the range's buckets; 83.0 per combo)


class TestYourCardsStillBlock(unittest.TestCase):
    def test_multiway_report_leaves_out_what_you_block(self):
        rs, res = spy_multiway(dict(SEVEN, hole="ThTs"))
        board = cards("Ks7s2h2c")
        live = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        off = np.prod([1.0 - MEMBER_FULL[c] for c in cards("ThTs")], axis=0)
        self.assertAlmostEqual(res["opponent_range_combos"]["CO"], float((rs[1] * live * off).sum()), places=1)
        self.assertLess(res["opponent_range_combos"]["CO"], float((rs[1] * live).sum()) - 0.5)

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
