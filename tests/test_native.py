"""The native server's and CLI's solve settings (gto/native.py) settle within their budgets."""
import unittest

from gto.native import advisor, advisors, SPEEDS

SRP = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
FLOP = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 1.8}, {"pos": "BB", "action": "call"}]
TURN = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 6.0}, {"pos": "BB", "action": "call"}]


class TestNativeSpeeds(unittest.TestCase):
    def test_fast_turn_settles_and_river_range_holds(self):
        """BTN vs BB, K72 c-bet called, turn T. Config()'s full menu ran 10 turn iterations in the server's
        2.5 s, 60-70% of the pot from equilibrium, and BB's river check range narrowed on it to 7 combos
        (a long solve keeps 81). Fast settles the turn (~2%) and keeps ~79 (time-budgeted: a loaded
        machine gets fewer iterations, hence the slack)."""
        adv = advisor("fast")
        self.assertEqual((adv.time_budget, adv.prior_budget), SPEEDS["fast"][1:])
        base = dict(hero="BTN", hole="AhQd", stack=100, preflop=SRP, flop=FLOP)
        r = adv.advise(dict(base, board="Ks7d2cTh", turn=[{"pos": "BB", "action": "check"}]))
        self.assertLess(r["solver"]["exploitability_pct"], 5.0, r["solver"])
        r = adv.advise(dict(base, board="Ks7d2cTh5s", turn=TURN, river=[{"pos": "BB", "action": "check"}]))
        self.assertGreater(r["villain_range_combos"], 50, r["villain_range"])

    def test_server_picks_the_advisor_by_speed(self):
        advs = advisors(budget=1.0)
        self.assertEqual(advs["fast"].time_budget, 1.0)
        self.assertEqual(advs["accurate"].time_budget, SPEEDS["accurate"][1])
        self.assertNotEqual(advs["fast"].cfg.sizes_key(), advs["accurate"].cfg.sizes_key())


if __name__ == "__main__":
    unittest.main()
