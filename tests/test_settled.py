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


def cached(adv, n_board):
    sols = [s for k, s in adv._cache.items() if len(k[2]) == n_board]
    assert len(sols) == 1, len(sols)
    return sols[0]


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
        self.assertGreaterEqual(turn.iters, n)
        self.assertNotIn("error", res)
        self.assertLess(time.time() - t, 2.5)

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


if __name__ == "__main__":
    unittest.main()
