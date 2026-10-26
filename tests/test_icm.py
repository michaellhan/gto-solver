"""Tournaments: ICM prize equity (gto/icm.py), its payoffs in the heads-up solver (Config.icm) and push/fold
(pf_classes.push_fold's icm), and the advisor's tournament mode (state "payouts" / "field")."""
import itertools
import unittest
import numpy as np
from gto import icm
from gto import preflop as pf
from gto.advisor import Advisor, preflop_strategy, preflop_ranges, _eff_stack
from gto.cards import cards as parse_cards
from gto.pf_classes import push_fold, _fictitious_play, _fictitious_play_seq, to_classes, CLASSES
from gto.ranges import parse_range
from gto.solver import solve, Config


def brute(stacks, pay):
    """Malmuth-Harville by every finishing order."""
    n = len(stacks)
    eq = np.zeros(n)
    for order in itertools.permutations(range(n)):
        p, left = 1.0, float(sum(stacks))
        for k, i in enumerate(order):
            p *= stacks[i] / left
            left -= stacks[i]
        for k, i in enumerate(order[:len(pay)]):
            eq[i] += p * pay[k]
    return eq


def R(pos, action, to=None):
    return {"pos": pos, "action": action, **({"to": to} if to is not None else {})}


S4 = {"CO": 40, "BTN": 25, "SB": 8, "BB": 25}     # 4 left, 3 paid: the bubble


def freq(stacks, pay, line, p):
    """p's range-wide frequencies after `line`, with or without the tournament."""
    t0 = pf.Table(list(stacks), 0.0, False, None, 100, stacks, icm.Model(stacks, pay) if pay else None)
    depth = _eff_stack(t0, p, line, 100)
    t = pf.at_depth(t0, depth)
    s = preflop_strategy(p, line, t, depth)
    w = preflop_ranges(line, t, depth).get(p, np.ones(1326))
    return {k: float((v * w).sum() / w.sum()) for k, v in s.items()}


class TestModel(unittest.TestCase):
    def test_equities_match_every_finishing_order(self):
        for stacks, pay in (([10, 20, 30, 40], [50, 30, 20]), ([5, 5, 90], [70, 30]), ([1, 2, 3, 4, 5], [40, 25, 15, 12, 8])):
            np.testing.assert_allclose(icm.equities(stacks, pay), brute(stacks, pay), atol=1e-9)
            self.assertAlmostEqual(float(icm.equities(stacks, pay).sum()), sum(pay[:len(stacks)]))

    def test_busted_players_share_the_places_below(self):
        e = icm.equities([10, 0, 30], [50, 30, 20])
        self.assertAlmostEqual(e[1], 20.0)
        self.assertAlmostEqual(e.sum(), 100.0)

    def test_bubble_factor(self):
        m = icm.Model({"BTN": 20, "SB": 15, "BB": 5}, [50, 30, 20])
        self.assertGreater(m.bubble_factor("SB", "BTN"), 1.2)
        wta = icm.Model({"BTN": 20, "SB": 15, "BB": 5}, [100, 1e-12])
        self.assertAlmostEqual(wta.bubble_factor("SB", "BTN"), 1.0, places=6)

    def test_big_field_keeps_each_players_share(self):
        """27 left, 12 paid, 6 at the table: modelled by 12 players, an even field's equity stays 1/27 of the prizes."""
        pay = [30, 20, 15, 10, 8, 6, 4, 3, 2, 1, 0.5, 0.5]
        m = icm.Model({p: 20 for p in pf.SEATS[6]}, pay, [20] * 21)
        self.assertTrue(m.approx)
        self.assertEqual(len(m.seats) + len(m.field), icm.MAX_PLAYERS)
        self.assertAlmostEqual(m.base["BTN"], 100 / 27, places=6)
        # the money still starts 12/27 of the way down: the model pays 12 * 12/27 = 5.3 places
        self.assertEqual(len(m.payouts), 6)

    def test_call_need(self):
        put = {"CO": 0.0, "BTN": 20.0, "SB": 0.5, "BB": 1.0}
        bubble = icm.Model(dict(CO=30, **{"BTN": 20, "SB": 15, "BB": 5}), [50, 30, 20])
        need, chips = bubble.call_need(put, "SB", "BTN", 14.5)
        self.assertAlmostEqual(chips, 14.5 / 31)          # the BTN's 5 over the SB's stack goes back
        self.assertGreater(need, chips + 0.15)
        wta = icm.Model(dict(CO=30, **{"BTN": 20, "SB": 15, "BB": 5}), [100, 1e-12])
        need, chips = wta.call_need(put, "SB", "BTN", 14.5)
        self.assertAlmostEqual(need, chips, places=6)


class TestSolver(unittest.TestCase):
    def test_winner_take_all_is_chips(self):
        """Prize equity in a winner-take-all is chips, so the solve's EVs are the chips solve's."""
        board = parse_cards("Ks7d2c9h4s")
        r0 = parse_range("22+,A2s+,KTs+,QJs,ATo+,KJo+")
        r1 = parse_range("33+,A5s+,KQs,AJo+")
        stacks = {"CO": 40, "BTN": 25, "SB": 8, "BB": 25}
        contrib = {"CO": 0.0, "BTN": 3.0, "SB": 0.5, "BB": 3.0}
        out = []
        for c in (None, icm.HUPayoffs(icm.Model(stacks, [100, 1e-12]), ("BB", "BTN"), contrib, (3.25, 3.25))):
            cfg = Config(bet_sizes={"river": [0.5, 1.0]}, raise_sizes={"river": [1.0]})
            cfg.icm = c
            sol = solve(board, r0, r1, 6.5, 22.0, cfg=cfg, time_budget=100, max_iters=40)
            out.append(sol.solver.node_action_evs(sol.root, sol.r0, sol.r1))
        np.testing.assert_allclose(out[0], out[1], atol=2e-3)

    def test_bubble_changes_the_bets(self):
        """On the bubble the river's big bets lose value: the bettor risks more prize equity than it wins."""
        board = parse_cards("Ks7d2c9h4s")
        r0 = parse_range("22+,A2s+,KTs+,QJs,ATo+,KJo+")
        r1 = parse_range("33+,A5s+,KQs,AJo+")
        stacks = {"CO": 40, "BTN": 25, "SB": 8, "BB": 25}
        contrib = {"CO": 0.0, "BTN": 3.0, "SB": 0.5, "BB": 3.0}
        allin = []
        for pay in ([100, 1e-12], [50, 30, 20]):
            cfg = Config(bet_sizes={"river": [0.5, 3.0]}, raise_sizes={"river": [1.0]})
            cfg.icm = icm.HUPayoffs(icm.Model(stacks, pay), ("BB", "BTN"), contrib, (3.25, 3.25))
            sol = solve(board, r0, r1, 6.5, 22.0, cfg=cfg, time_budget=100, max_iters=150)
            st = sol.root.avg_strategy0()
            names = list(sol.root.names)
            big = max((n for n in names if n.startswith("bet") or n == "all-in"), key=lambda n: sol.root.children[names.index(n)].bets[0])
            allin.append(float((st[names.index(big)] * sol.r0).sum() / sol.r0.sum()))
        self.assertLess(allin[1], allin[0])
        self.assertGreaterEqual(sol.exploitability(), 0.0)


class TestPushFold(unittest.TestCase):
    def test_sequential_callers_alike_match_the_symmetric_play(self):
        wj = to_classes(np.ones(1326))
        wc = to_classes(parse_range("22+,A2s+,K9s+,ATo+,KJo+"))
        alt = np.zeros(len(CLASSES))
        vals = (1.5, 12.0, -10.0, 11.0, -9.0)
        a = _fictitious_play(wj, wc, alt, vals, 2, 200)
        b = _fictitious_play_seq(wj, [wc, wc], alt, vals[0], [(vals[1], vals[2], vals[3], vals[4])] * 2, 200)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_icm_payoffs(self):
        """A bubble's payoffs: the callers' losses cost more, so they call less and the jammer jams more."""
        r = np.ones(1326)
        chips = push_fold(r, r, 15.0, 0.5, 1.0, 1.5)
        wide = push_fold(r, r, 15.0, 0.5, 1.0, 1.5, icm=(1.5, [(15.0, -14.5, 3.0, -14.0)], [r]))
        self.assertGreater(wide.mean(), chips.mean() + 0.1)


class TestAdvisor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = Advisor(time_budget=1.0, prior_budget=0.5)

    def state(self, **kw):
        return dict({"players": 4, "bb": 1, "stacks": S4}, **kw)

    def test_no_tournament_without_two_prizes(self):
        jam = [R("CO", "fold"), R("BTN", "all-in")]
        for pay in (None, [100], "100"):
            st = self.state(hero="SB", hole="AhJd", preflop=jam, **({"payouts": pay} if pay else {}))
            r = self.A.advise(st)
            self.assertNotIn("tournament", r)
        r = self.A.advise(self.state(hero="SB", hole="AhJd", preflop=jam, payouts="50, x"))
        self.assertIn("error", r)

    def test_bubble_call_of_a_jam(self):
        """SB 15bb facing BTN's 20bb jam, 4 left 3 paid with a 5bb big blind: AJo and 99 call in chips, fold in prizes."""
        stacks = {"CO": 30, "BTN": 20, "SB": 15, "BB": 5}
        jam = [R("CO", "fold"), R("BTN", "all-in")]
        for hole in ("AhJd", "9s9c"):
            chips = self.A.advise(dict(players=4, bb=1, stacks=stacks, hero="SB", hole=hole, preflop=jam))
            prize = self.A.advise(dict(players=4, bb=1, stacks=stacks, hero="SB", hole=hole, preflop=jam, payouts=[50, 30, 20]))
            self.assertEqual(chips["recommended"], "call")
            self.assertEqual(prize["recommended"], "fold")
            self.assertIn("tournament", prize)
            self.assertGreater(prize["tournament"]["bubble_factors"]["BTN"], 1.5)
        aa = self.A.advise(dict(players=4, bb=1, stacks=stacks, hero="SB", hole="AhAd", preflop=jam, payouts=[50, 30, 20]))
        self.assertEqual(aa["recommended"], "call")

    def test_bubble_tightens_defence_and_3bets(self):
        bb = [R("CO", "fold"), R("BTN", "raise", 2), R("SB", "fold")]
        chips, prize = freq(S4, None, bb, "BB"), freq(S4, [50, 30, 20], bb, "BB")
        self.assertLess(1 - prize["fold"], 1 - chips["fold"] - 0.15)
        three = [R("CO", "raise", 2.2)]
        chips, prize = freq(S4, None, three, "BTN"), freq(S4, [50, 30, 20], three, "BTN")
        self.assertLess(prize["raise"], chips["raise"])
        opens_c, opens_i = freq(S4, None, [R("CO", "fold")], "BTN"), freq(S4, [50, 30, 20], [R("CO", "fold")], "BTN")
        self.assertLessEqual(opens_i["raise"], opens_c["raise"])
        self.assertGreater(opens_i["raise"], 0.8 * opens_c["raise"])   # opens move least

    def test_heads_up_postflop_solves_in_prizes(self):
        st = self.state(hero="BB", hole="7s6s", board="Kh7d2c",
                        preflop=[R("CO", "fold"), R("BTN", "raise", 2), R("SB", "fold"), R("BB", "call")],
                        flop=[{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "amount": 3}],
                        payouts=[50, 30, 20])
        r = self.A.advise(st)
        self.assertNotIn("error", r)
        self.assertIn("tournament", r)
        self.assertNotIn("precomputed", r)
        self.assertAlmostEqual(sum(a["freq"] for a in r["actions"]), 1.0, places=2)
        # every solve this answer ran paid prize equity
        keys = [k for k in self.A._cache if k[2] == tuple(sorted(parse_cards("Kh7d2c"))) or k[2] == tuple(parse_cards("Kh7d2c"))]
        self.assertTrue(keys)
        self.assertTrue(all(any(isinstance(x, tuple) and x[:1] == ("icm",) for x in k[8]) for k in keys))

    def test_practice_opponent_after_the_flop(self):
        """A practice opponent's move after the flop in a tournament (its answer carries no notes: it raised)."""
        st = {"players": 6, "bb": 1, "stack": 150, "payouts": [75.8, 75.1, 38.9], "board": "Ac9hQh", "hero": "BB",
              "hole": "Qs6s", "preflop": [R("UTG", "raise", 2.5)] + [R(p, "fold") for p in ("HJ", "CO", "BTN", "SB")]
              + [R("BB", "call")], "flop": [{"pos": "BB", "action": "check"}]}
        m = self.A.play(st, "UTG", "3sAh", 1)
        self.assertNotIn("error", m)
        self.assertIn(m["action"], ("check", "bet", "all-in"))

    def test_field(self):
        pay9 = [1000, 700, 500, 400, 300, 250, 200, 150, 120, 100, 90, 80, 70, 60, 50]
        st = {"players": 9, "bb": 1, "hero": "CO", "hole": "Ah9c", "payouts": pay9, "field": {"players": 30, "chips": 900},
              "stacks": {"UTG": 30, "UTG1": 12, "UTG2": 45, "LJ": 20, "HJ": 8, "CO": 60, "BTN": 25, "SB": 15, "BB": 35},
              "preflop": [R(p, "fold") for p in ("UTG", "UTG1", "UTG2", "LJ", "HJ")]}
        r = self.A.advise(st)
        self.assertNotIn("error", r)
        self.assertIn("39 players modelled by 12", r["tournament"]["note"])
        self.assertLess(r["tournament"]["equity_share"], 0.1)
        bad = dict(st, field={"players": 3, "chips": 0})
        self.assertIn("error", self.A.advise(bad))


if __name__ == "__main__":
    unittest.main()
