"""The native server's and CLI's solve settings (gto/native.py) settle within their budgets."""
import unittest
from unittest import mock

from gto.native import advisor, advisors, SPEEDS

SRP = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
FLOP = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 1.8}, {"pos": "BB", "action": "call"}]
UNLOADED_TURN_ITERS = 45   # fast's turn in its 2.5 s on an unloaded machine (40-45)
TURN = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 6.0}, {"pos": "BB", "action": "call"}]


def fixed_iters(iters):
    """Patch the advisor's solves to run a fixed number of iterations by street ({3: flop, 4: turn, 5: river})
    whatever the machine's load, as time budgets give them on an unloaded or a loaded machine."""
    import gto.advisor as A
    from gto.solver import Solution
    orig = A.solve

    def solve(board, *a, **k):
        k["time_budget"], k["max_iters"] = 1e9, iters[len(board)]
        return orig(board, *a, **k)
    return mock.patch.multiple(A, solve=solve), mock.patch.object(Solution, "extend", lambda self, *a, **k: self)


class TestNativeSpeeds(unittest.TestCase):
    def river_check_range(self, iters):
        p1, p2 = fixed_iters(iters)
        with p1, p2:
            r = advisor("fast").advise(dict(hero="BTN", hole="AhQd", stack=100, preflop=SRP, flop=FLOP, board="Ks7d2cTh5s",
                                            turn=TURN, river=[{"pos": "BB", "action": "check"}]))
        return r["villain_range_combos"], r

    def test_fast_turn_settles_and_river_range_holds(self):
        """BTN vs BB, K72 c-bet called, turn T. Config()'s full menu ran 10 turn iterations in the server's
        2.5 s, 60-70% of the pot from equilibrium, and BB's river check range narrowed on it to 7 combos
        (long solves keep 87.7). Fast settles the turn (~2%, 40-45 iterations unloaded). Judged by iterations,
        not time: a loaded machine gets fewer in the budget (26 turn iterations at load 20), so a turn short
        of the unloaded count is run on to it."""
        adv = advisor("fast")
        self.assertEqual((adv.time_budget, adv.prior_budget), SPEEDS["fast"][1:])
        base = dict(hero="BTN", hole="AhQd", stack=100, preflop=SRP, flop=FLOP)
        r = adv.advise(dict(base, board="Ks7d2cTh", turn=[{"pos": "BB", "action": "check"}]))
        sol = next(s for k, s in adv._cache.items() if len(k[2]) == 4)   # the turn's (key: two ranges, then the board)
        if sol.iters < UNLOADED_TURN_ITERS:
            sol.extend(1e9, max_iters=UNLOADED_TURN_ITERS)
        self.assertLess(sol.exploitability(), 5.0, (sol.iters, r["solver"]))
        # the river range, at the iterations an unloaded machine's budgets give (bucketed flop 140-210, turn ~45): 80.7
        combos, r = self.river_check_range({3: 150, 4: UNLOADED_TURN_ITERS, 5: 100})
        self.assertGreater(combos, 65, r["villain_range"])

    def test_unsettled_earlier_streets_keep_the_range(self):
        """A loaded machine's (or the browser's) earlier streets: a flop of 20 iterations (27% of the pot from
        equilibrium with Fast's buckets) and a turn of 26 (6%) left BB 29 combos checking the river where long solves
        keep 87.7 (with the old fast flop, 9%: 18 for 69.5). Narrowing follows a solve by how settled it is
        (advisor.PRIOR_TRUST), the rest by the betting rules, so the range errs wide, not thin (89)."""
        combos, r = self.river_check_range({3: 20, 4: 26, 5: 100})
        self.assertGreater(combos, 45, r["villain_range"])
        self.assertTrue(any("ranges follow it" in n for n in r["notes"]), r["notes"])

    def test_server_picks_the_advisor_by_speed(self):
        advs = advisors(budget=1.0)
        self.assertEqual(advs["fast"].time_budget, 1.0)
        self.assertEqual(advs["accurate"].time_budget, SPEEDS["accurate"][1])
        self.assertNotEqual(advs["fast"].cfg.sizes_key(), advs["accurate"].cfg.sizes_key())


if __name__ == "__main__":
    unittest.main()
