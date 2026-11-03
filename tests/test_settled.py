"""Heads-up solves that stop once settled (Advisor.hu_settled: the first of HU_CHECKS where the solve is that close to
equilibrium) and solves stopped part way (Advisor.should_stop: the browser's stop flag, the local server's cancel)."""
import time
import unittest

from gto import native
from gto.advisor import HU_CHECKS, HU_SETTLED_FROM
from gto.solver import Cancelled

F = lambda p: {"pos": p, "action": "fold"}
C = lambda p: {"pos": p, "action": "call"}
X = lambda p: {"pos": p, "action": "check"}
R = lambda p, to: {"pos": p, "action": "raise", "to": to}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}
SRP = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), C("BB")]
TURN = {"players": 6, "stack": 100, "preflop": SRP, "flop": [X("BB"), B("BTN", 1.8), C("BB")], "turn": [], "river": [],
        "hero": "BB", "hole": "QhJh", "board": "Ks7d2cTh"}
RIVER = dict(TURN, turn=[X("BB"), B("BTN", 6), C("BB")], board="Ks7d2cTh5s")
LIMP = [F("UTG"), F("HJ"), F("CO"), F("BTN"), C("SB"), X("BB")]


def cached(adv, n_board):
    """The street's one solve from its start (no chips in, the first player to act)."""
    sols = [s for k, s in adv._cache.items() if len(k[2]) == n_board and tuple(k[5]) == (0.0, 0.0) and k[6] == 0]
    assert len(sols) == 1, len(sols)
    return sols[0]


def resolved(adv, res, street, n_board):
    """The street's solves from a node (chips in, or the second player to act): each the play after a rare line
    solved again (advisor.RARE_RESOLVE), as many as the answer's off_solution says were."""
    sols = [s for k, s in adv._cache.items() if len(k[2]) == n_board and (tuple(k[5]) != (0.0, 0.0) or k[6] != 0)]
    n = sum(1 for o in res.get("off_solution", []) if o.get("street") == street and o.get("resolved"))
    assert len(sols) == n, (len(sols), n)
    return sols


class TestSettled(unittest.TestCase):
    def test_stops_at_the_first_check_under_the_target(self):
        """A river stops at the first of HU_CHECKS where its exploitability is under hu_settled (read there only),
        well before its budget; asked again it isn't run on."""
        adv = native.advisor("fast", budget=10, prior_budget=0.5)
        adv.hu_settled = 0.1
        adv.hu_settled_from = {}   # (from the first check: the floors have their own test)
        reads = []
        read = adv._expl
        adv._expl = lambda sol: reads.append((sol.ctx.root_len, sol.iters, read(sol))) or reads[-1][2]
        t = time.time()
        res = adv.advise(RIVER)
        self.assertLess(time.time() - t, 8)
        river = sorted({(it, e) for n, it, e in reads if n == 5})   # (each check read once: the answer reads it again)
        it = res["solver"]["iterations"]
        self.assertIn(it, HU_CHECKS)
        self.assertEqual(river[-1][0], it)
        self.assertLess(river[-1][1], 0.1)
        self.assertTrue(all(i in HU_CHECKS and e >= 0.1 for i, e in river[:-1]), river)
        self.assertEqual([i for i, _ in river[:-1]], [c for c in HU_CHECKS if c < it])
        sol = cached(adv, 5)
        self.assertEqual(sol.settled_at, 0.1)
        again = adv.advise(RIVER)
        self.assertEqual(again["solver"]["iterations"], it)
        self.assertIs(cached(adv, 5), sol)

    def test_a_settled_street_narrowing_the_next_isnt_run_on(self):
        """The turn advised on, settled, then narrows the river's ranges as it is: not run on to the earlier
        street's budget (prior_budget), where an unsettled one would be."""
        adv = native.advisor("fast", budget=10, prior_budget=10)
        adv.hu_settled = 50.0   # (settled at the first check, 50 iterations)
        adv.hu_settled_from = {}
        adv.advise(TURN)
        turn = cached(adv, 4)
        self.assertEqual((turn.iters, turn.settled_at), (HU_CHECKS[0], 50.0))
        adv.advise(RIVER)
        self.assertIs(cached(adv, 4), turn)
        self.assertEqual(turn.iters, HU_CHECKS[0])

    def test_none_stops_before_its_streets_first_check(self):
        """However settled, a river doesn't stop before HU_SETTLED_FROM's check (its per-combo EVs settle later than
        its exploitability), nor a turn before its own; a deep turn is "turn", a shallow one (spr_sizes) "shallow"."""
        self.assertEqual(HU_SETTLED_FROM["river"], 300)
        adv = native.advisor("fast", budget=10, prior_budget=0.5)
        adv.hu_settled = 50.0
        adv.advise(RIVER)
        self.assertEqual((cached(adv, 5).iters, cached(adv, 5).settled_at), (300, 50.0))
        adv = native.advisor("fast", budget=10, prior_budget=0.5)
        adv.hu_settled = 50.0
        adv.hu_settled_from = dict(HU_SETTLED_FROM, river=150, turn=100)
        adv.advise(RIVER)
        self.assertEqual(cached(adv, 5).iters, 150)
        adv.advise(TURN)
        self.assertEqual(cached(adv, 4).iters, 100)   # (Ks7d2cTh in a single-raised pot: 10 pots behind)

    def test_the_rivers_narrowing_reads_the_hands_turn_solve(self):
        """The river's ranges come through the turn solve the turn's advice made (found in the cache: its ranges
        left out the turn's cards, not the river's), not a second solve of the turn."""
        adv = native.advisor("fast", budget=1, prior_budget=1)
        adv.advise(TURN)
        turn = cached(adv, 4)
        n = turn.iters
        t = time.time()
        res = adv.advise(RIVER)
        self.assertIs(cached(adv, 4), turn)
        # (BTN's 6 bb bet is under RARE_LINE of its range in this 1 s solve, ~5%: the rest of the turn is solved
        # again from the bet, a solve of its own, not the turn's again)
        resolved(adv, res, "turn", 4)
        self.assertGreaterEqual(turn.iters, n)
        self.assertNotIn("error", res)
        self.assertLess(time.time() - t, 2.5)

    def test_through_a_size_off_the_turn_solves_tree(self):
        """Their turn bet off the tree the turn's advice solved (49% where it has 33/75/125%): the river's narrowing walks
        that solve at its nearest size (Advisor._hand_cfg) instead of solving the turn again with the size in."""
        adv = native.advisor("fast", budget=1, prior_budget=1)
        adv.advise(TURN)
        turn = cached(adv, 4)
        res = adv.advise(dict(RIVER, turn=[X("BB"), B("BTN", 4.5), C("BB")]))
        self.assertIs(cached(adv, 4), turn)
        resolved(adv, res, "turn", 4)   # (its 33% is ~1% of BTN's range here: solved again from the bet)
        self.assertNotIn("error", res)
        self.assertTrue(any(n.startswith("turn: mapped 'bet 49%'") for n in res["notes"]), res["notes"])

    def test_your_hand_out_of_your_range(self):
        """Your hand out of your range for the line (the big blind checking KQs behind a limp, which it raises): each
        street's advice solves with it in at a near-zero weight (Advisor._with_hero), and the next street's narrowing
        walks that solve (Advisor._as_advised) instead of solving the street again without it. One solve a street;
        the hand is still out of its range (the note); the turn's approximate answer is read off the flop's solve,
        which holds the hand (advisor.instant_from_prev)."""
        adv = native.advisor("fast", budget=1, prior_budget=1)
        flop = {"players": 6, "stack": 100, "preflop": LIMP, "flop": [X("SB")], "turn": [], "river": [], "hero": "BB",
                "hole": "KhQh", "board": "Ks7d2c"}
        turn = dict(flop, flop=[X("SB"), B("BB", 1.5), C("SB")], turn=[X("SB")], board="Ks7d2cTh")
        river = dict(turn, turn=[X("SB"), B("BB", 3.75), C("SB")], river=[X("SB")], board="Ks7d2cTh5s")
        out = lambda r: any("isn't in your range" in n for n in r["notes"])
        self.assertTrue(out(adv.advise(flop)))
        f = cached(adv, 3)
        self.assertEqual(adv.instant(turn)["approximate"]["source"], "prev")
        self.assertTrue(out(adv.advise(turn)))
        t = cached(adv, 4)
        self.assertIs(cached(adv, 3), f)
        res = adv.advise(river)
        self.assertNotIn("error", res)
        self.assertTrue(out(res), res["notes"])
        self.assertIs(cached(adv, 3), f)
        self.assertIs(cached(adv, 4), t)

    def test_stopped_part_way_and_asked_again(self):
        """A solve stopped part way raises Cancelled and stays cached; asked again, the spot goes on from where it
        stopped (its iterations and root mix brought up to date) and answers."""
        adv = native.advisor("fast", budget=10, prior_budget=0.5)
        adv.hu_settled = 1e-9   # (never settled: run in pieces, a check between them)
        stop = [False]
        adv.should_stop = lambda: stop[0]
        read = adv._expl

        def expl(sol):   # the river's first check: stop it there (the core sees the flag at its next chunk)
            if sol.ctx.root_len == 5:
                stop[0] = True
            return read(sol)
        adv._expl = expl
        with self.assertRaises(Cancelled):
            adv.advise(RIVER)
        sol = cached(adv, 5)
        self.assertTrue(sol.stopped)
        self.assertGreaterEqual(sol.solver.iters, HU_CHECKS[0])
        stop[0] = False
        del adv._expl
        adv.hu_settled = None
        res = adv.advise(RIVER)
        self.assertIs(cached(adv, 5), sol)
        self.assertFalse(sol.stopped)
        self.assertEqual(sol.iters, sol.solver.iters)
        self.assertGreater(res["solver"]["iterations"], HU_CHECKS[0])
        self.assertIn(res["recommended"], [a["action"] for a in res["actions"]])

    def test_the_roots_evs_worked_out_once(self):
        """An answer at its solve's root reads the root's EVs as the solve left them (Solution.fresh: worked out at
        its last iterations) instead of working them out again (Solver.node_action_evs: 20-30 ms on a Fast turn),
        and they are the numbers a fresh evaluation gives, bit for bit. Iterations on make them stale."""
        import numpy as np
        from unittest import mock
        from gto.cards import COMBO_INDEX, cards
        from gto.solver import Solver
        adv = native.advisor("fast", budget=10, prior_budget=0.5)
        real, calls = Solver.node_action_evs, []

        def nae(self, *a):
            calls.append((self, self.iters))
            return real(self, *a)
        with mock.patch.object(Solver, "node_action_evs", nae):
            res = adv.advise(RIVER)
        sol = cached(adv, 5)
        self.assertEqual([it for s, it in calls if s is sol.solver], [sol.iters])
        self.assertTrue(sol.fresh())
        fresh = sol.solver.node_action_evs(sol.root, sol.r0, sol.r1)
        np.testing.assert_array_equal(sol.evs[:, sol.ctx.active], fresh)
        h = cards(RIVER["hole"])
        ci = list(sol.ctx.active).index(COMBO_INDEX[(min(h), max(h))])
        self.assertEqual([a["ev_bb"] for a in res["actions"]], [round(float(x), 2) for x in fresh[:, ci]])
        sol.solver.run(sol.r0, sol.r1, time_budget=1e9, max_iters=sol.solver.iters + 1, min_iters=1)
        self.assertFalse(sol.fresh())
        sol._refresh()
        self.assertTrue(sol.fresh())


if __name__ == "__main__":
    unittest.main()


class TestApproximate(unittest.TestCase):
    def test_an_answer_far_over_its_target_says_so(self):
        """A heads-up turn or river answer more than OVER_TARGET times its speed's expl_target from equilibrium (its
        solve ran short: a busy device, or numpy after the compiled core failed, 5-7% on the browser's Fast turn) is
        marked approximate, with the target and a note; within it, or with no target, it is a normal answer. Quick's
        answers and multiway solves aren't marked."""
        from gto.advisor import OVER_TARGET
        adv = native.advisor("fast", budget=2, prior_budget=0.5)
        res = adv.advise(RIVER)
        x = res["solver"]["exploitability_pct"]
        self.assertGreater(x, 0)
        self.assertNotIn("approximate", res["solver"])
        adv.expl_target = x / OVER_TARGET   # (exactly at it: within)
        res = adv.advise(RIVER)
        self.assertEqual(res["solver"]["exploitability_pct"], x)   # (the cached solve)
        self.assertNotIn("approximate", res["solver"])
        adv.expl_target = x / OVER_TARGET / 2
        res = adv.advise(RIVER)
        self.assertIs(res["solver"]["approximate"], True)
        self.assertEqual(res["solver"]["target_pct"], adv.expl_target)
        self.assertEqual(sum(n.startswith("Approximate:") for n in res["notes"]), 1)
        for name, state, got in (("quick", dict(RIVER, quick=True), dict(street="river")),
                                 ("preflop", RIVER, dict(street="preflop")),
                                 ("multiway", RIVER, dict(street="river", multiway=True))):
            with self.subTest(name):
                got = dict(got, solver={"exploitability_pct": 100.0}, notes=[])
                adv._flag_short(state, got)
                self.assertNotIn("approximate", got["solver"])
                self.assertEqual(got["notes"], [])
