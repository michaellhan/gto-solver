"""The 3-player solver (gto/multiway_solver.py): exact card removal for three ranges, constant-sum payoffs,
convergence, agreement with the heads-up solver when played two-handed, and the advisor's use of it."""
import unittest

import numpy as np

import gto.advisor as A
from gto import preflop as pf
from gto.cards import ALL_COMBOS, cards
from gto.evaluator import eval_board
from gto.multiway_solver import MWConfig, MultiwaySolver, Terminal
from gto.solver import Config, solve

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
       {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
CHECKS = [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "check"}, {"pos": "BTN", "action": "check"}]


def _ranges():
    R = A.preflop_ranges(PRE, pf.Table(pf.SEATS[6]), 100.0)
    return [R["BB"], R["CO"], R["BTN"]]


def _brute(s: MultiwaySolver, term: Terminal, board):
    """Every player's terminal values by the triple loop over compatible hands."""
    st = eval_board(board)
    act = s.hands.active
    R = s.reach0[:, 0, :]
    out = np.zeros((3, len(act)))
    for p in range(3):
        q, r = [j for j in range(3) if j != p]
        for ih, fh in enumerate(act):
            ch = set(ALL_COMBOS[fh])
            tot = 0.0
            for iq in np.nonzero(R[q])[0]:
                cq = set(ALL_COMBOS[act[iq]])
                if cq & ch:
                    continue
                for ir in np.nonzero(R[r])[0]:
                    cr = set(ALL_COMBOS[act[ir]])
                    if cr & ch or cr & cq:
                        continue
                    hs = {p: st[fh], q: st[act[iq]], r: st[act[ir]]}
                    if p not in term.live:
                        share = 0.0
                    else:
                        best = max(hs[j] for j in term.live)
                        share = 0.0 if hs[p] < best else 1.0 / sum(1 for j in term.live if hs[j] == best)
                    tot += R[q, iq] * R[r, ir] * (share * (term.pot - term.rake) - term.c[p])
            out[p, ih] = tot
    return out


class TestCardRemoval(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.board = tuple(cards("Kh7d2c9s9h"))      # paired: ties between kickers and the board
        ok = [i for i, (a, b) in enumerate(ALL_COMBOS) if a not in self.board and b not in self.board]
        rs = []
        for _ in range(3):
            r = np.zeros(1326)
            pick = rng.choice(ok, 40, replace=False)
            r[pick] = rng.random(40)
            rs.append(r)
        rs[2][np.nonzero(rs[1])[0][:10]] = 0.5       # the same combos in two ranges
        self.s = MultiwaySolver(self.board, rs, 10.0, 50.0)

    def test_exact_terminals_match_the_triple_loop(self):
        s, bd = self.s, self.s.root.bd
        for live in [(0, 1, 2), (0, 2), (1, 2), (1,)]:
            t = Terminal(bd, live, 30.0, np.array([5.0, 7.0, 9.0]), 0.6, "show")
            with self.subTest(live=live):
                got = s._term(t, s.reach0)[:, 0]
                want = _brute(s, t, self.board)
                self.assertLess(np.abs(got - want).max(), 1e-3 * np.abs(want).max())

    def test_pairwise_is_an_approximation(self):
        s, bd = self.s, self.s.root.bd
        t = Terminal(bd, (0, 1, 2), 30.0, np.array([5.0, 7.0, 9.0]), 0.0, "show")
        want = _brute(s, t, self.board)
        s.cfg.exact = False
        got = s._term(t, s.reach0)[:, 0]
        s.cfg.exact = True
        err = np.abs(got - want).max() / np.abs(want).max()
        self.assertGreater(err, 1e-3)
        self.assertLess(err, 0.05)
        # ... whose shares are scaled to keep the game constant-sum: the players' totals add up to the pot
        tot = sum(float((s.reach0[p, 0] * got[p]).sum()) for p in range(3))
        m = s.total_mass()
        self.assertAlmostEqual(tot / m, 30.0 - 21.0, places=6)


class TestSolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = _ranges()
        cls.board = cards("Kh7d2c9s4h")
        cls.s = MultiwaySolver(cls.board, cls.rs, 8.0, 97.5)
        cls.s.run(iters=8)
        cls.g8 = cls.s.br_gains()
        cls.s.run(iters=52)
        cls.g60 = cls.s.br_gains()

    def test_constant_sum(self):
        v = self.s.values()
        self.assertAlmostEqual(float(v.sum()), 8.0, places=4)    # what the three expect adds up to the pot

    def test_best_response_gains_shrink(self):
        # no Nash guarantee with three players, but each player's best-response gain falls
        pot = self.s.root_pot()
        self.assertTrue(np.all(self.g60 < self.g8))
        self.assertLess(self.g60.max() / pot * 100, 1.0)
        self.assertGreaterEqual(self.g60.min(), -1e-6)

    def test_nuts_bet_and_air_rarely_bluffs(self):
        st = eval_board(self.board)

        def bet_share(s, p, lo, hi):
            f = s.reach0[p, 0]
            sc = st[s.hands.active]
            live = f > 0
            sel = live & (sc >= np.quantile(sc[live], lo)) & (sc <= np.quantile(sc[live], hi))
            return float((s.root.avg_strategy()[0][1] * f)[sel].sum() / f[sel].sum())
        # first to act into two players: the weaker half of the BB's range almost never bets
        self.assertLess(bet_share(self.s, 0, 0.0, 0.5), 0.1)
        # checked to the button: its strongest hands bet nearly always, its weaker half seldom
        btn = MultiwaySolver(self.board, self.rs, 8.0, 97.5, to_act=2, pending={2}).run(iters=60)
        self.assertGreater(bet_share(btn, 2, 0.97, 1.0), 0.8)
        self.assertLess(bet_share(btn, 2, 0.0, 0.5), 0.25)

    def test_three_way_bets_less_than_heads_up(self):
        # BB first to act on the river: against CO and BTN it bets less of its range than against CO alone
        hu = MultiwaySolver(self.board, self.rs[:2], 5.5, 97.5).run(iters=60)
        self.assertLess(self.s.range_freqs()["bet 50%"], hu.range_freqs()["bet 50%"])

    def test_two_player_mode_matches_the_heads_up_solver(self):
        R = A.preflop_ranges([dict(a, action="fold") if a["pos"] == "BTN" else a for a in PRE], pf.Table(pf.SEATS[6]), 100.0)
        cfg = Config(bet_sizes={s: [0.5] for s in ("flop", "turn", "river")},
                     raise_sizes={s: [0.75] for s in ("flop", "turn", "river")}, max_raises=2, allin_spr=1.5)
        hu = solve(self.board, R["BB"], R["CO"], 5.5, 97.5, cfg=cfg, time_budget=1e9, max_iters=150)
        mw = MultiwaySolver(self.board, [R["BB"], R["CO"]], 5.5, 97.5, cfg=MWConfig(allin_spr=1.5)).run(iters=150)
        self.assertEqual(hu.root.names, mw.root.names)
        r0 = np.zeros(1326); r0[mw.hands.active] = mw.reach0[0, 0]
        sh = np.zeros((2, 1326)); sh[:, hu.ctx.active] = hu.root.avg_strategy0()
        sm = np.zeros((2, 1326)); sm[:, mw.hands.active] = mw.root.avg_strategy()[0]
        self.assertLess(abs(float((sh[1] - sm[1]) @ r0) / r0.sum()), 0.02)
        eh = np.zeros((2, 1326)); eh[:, hu.ctx.active] = hu.solver.node_action_evs(hu.root, hu.r0, hu.r1)
        em = np.zeros((2, 1326)); em[:, mw.hands.active] = mw.action_evs()
        self.assertLess(float((np.abs(eh - em) * r0).sum(axis=1).max() / r0.sum()), 0.15)   # bb, range-weighted

    def test_later_streets_build_and_stay_constant_sum(self):
        cfg = MWConfig(turns=4, rivers=6)
        for board in ("Kh7d2c9s", "Kh7d2c"):
            with self.subTest(board=board):
                s = MultiwaySolver(cards(board), self.rs, 8.0, 97.5, cfg=cfg).run(iters=3)
                self.assertAlmostEqual(float(s.values().sum()), 8.0, delta=0.1)


class TestAdvisor(unittest.TestCase):
    def _state(self, board, hero, hole, street_actions):
        st = {"players": 6, "hero": hero, "hole": hole, "board": board, "stack": 100, "preflop": PRE}
        names = ["flop", "turn", "river"][: len(board) // 2 - 2]
        for s in names[:-1]:
            st[s] = CHECKS
        st[names[-1]] = street_actions
        return st

    def test_three_way_river_is_solved(self):
        adv = A.Advisor(time_budget=1e9)
        adv.mw_max_iters = 25
        r = adv.advise(self._state("Kh7d2c9s4h", "BB", "Kd9d", []))
        self.assertTrue(r["multiway"] and r["solved"])
        self.assertEqual(r["solver"]["iterations"], 25)
        self.assertEqual([a["action"] for a in r["actions"]], ["check", "bet 50%"])
        self.assertAlmostEqual(sum(a["freq"] for a in r["actions"]), 1.0, places=2)
        self.assertGreater(r["equity_vs_field"], 0.7)            # two pair against two ranges that checked twice
        self.assertGreater(r["range_action_freq"]["check"], 0.3)

    def test_facing_a_bet_and_a_call(self):
        adv = A.Advisor(time_budget=1e9)
        adv.mw_max_iters = 20
        acts = [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 4.0},
                {"pos": "BTN", "action": "call"}]
        r = adv.advise(self._state("Kh7d2c9s4h", "BB", "7c7s", acts))
        self.assertTrue(r.get("solved"))
        names = [a["action"] for a in r["actions"]]
        self.assertEqual(names[:2], ["fold", "call"])
        self.assertEqual(r["to_call_bb"], 4.0)
        self.assertNotEqual(r["recommended"], "fold")             # a set, with three of a kind of nothing left to fear

    def test_flop_uses_the_betting_rules_by_default(self):
        # (its equity leaves bet half as often as published solves: README, Known limits)
        adv = A.Advisor(time_budget=1e9)
        adv.mw_trials = 100
        r = adv.advise(self._state("Kh7d2c", "BB", "Kd9d", []))
        self.assertTrue(r["multiway"])
        self.assertFalse(r.get("solved", False))

    def test_four_way_uses_the_betting_rules(self):
        adv = A.Advisor(time_budget=1e9)
        adv.mw_trials = 200
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5},
               {"pos": "CO", "action": "call"}, {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"},
               {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "BB", "hole": "Kd9d", "board": "Kh7d2c9s4h", "stack": 100, "preflop": pre,
                        "flop": [], "turn": [], "river": []} | {
            "flop": [{"pos": p, "action": "check"} for p in ("BB", "HJ", "CO", "BTN")],
            "turn": [{"pos": p, "action": "check"} for p in ("BB", "HJ", "CO", "BTN")]})
        self.assertTrue(r["multiway"])
        self.assertFalse(r.get("solved", False))


if __name__ == "__main__":
    unittest.main()
