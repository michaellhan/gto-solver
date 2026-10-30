"""The 3-player solver (gto/multiway_solver.py): exact card removal for three ranges, constant-sum payoffs,
convergence, agreement with the heads-up solver when played two-handed, and the advisor's use of it."""
import unittest

import numpy as np

import gto.advisor as A
from gto import preflop as pf
from gto.cards import ALL_COMBOS, cards
from gto.evaluator import eval_board
import gto.multiway_solver as MW
from gto.multiway_solver import MWConfig, MultiwaySolver, Terminal
from gto.solver import Config, solve

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
       {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
CHECKS = [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "check"}, {"pos": "BTN", "action": "check"}]


def _ranges():
    R = A.preflop_ranges(PRE, pf.Table(pf.SEATS[6]), 100.0)
    return [R["BB"], R["CO"], R["BTN"]]


PRE4 = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "call"},
        {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
X4 = [{"pos": p, "action": "check"} for p in ("BB", "HJ", "CO", "BTN")]


def _ranges4():
    R = A.preflop_ranges(PRE4, pf.Table(pf.SEATS[6]), 100.0)
    return [R["BB"], R["HJ"], R["CO"], R["BTN"]]


def _cfg4(**kw):
    """The advisor's 4-way game (advisor.MW4_BETS: one bet a street, no raise but the root player's)."""
    return MWConfig(later="buckets", max_raises=1, root_raise=True, bet=dict(A.MW4_BETS), **kw)


def _brute(s: MultiwaySolver, term: Terminal, board, pots=None):
    """Every player's terminal values by the triple loop over compatible hands (`pots`: [(chips, who contests them)],
    else the whole pot among everyone live)."""
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

                    def share(live):
                        if p not in live:
                            return 0.0
                        best = max(hs[j] for j in live)
                        return 0.0 if hs[p] < best else 1.0 / sum(1 for j in live if hs[j] == best)
                    net = term.pot - term.rake
                    won = net * sum(amt / term.pot * share(lv) for amt, lv in (pots or [(term.pot, term.live)]))
                    tot += R[q, iq] * R[r, ir] * (won - term.c[p])
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

    def test_side_pot_terminals_match_the_triple_loop(self):
        # player 2 all-in for 4 over a pot of 18, the others in for 9: a main pot of 30 that everyone live contests
        # and a side pot of 10 between the others live (the third player's alone when the second folded)
        s, bd = self.s, self.s.root.bd
        s.cfg.allin = (2,)
        try:
            for live, pots in [((0, 1, 2), [(30.0, (0, 1, 2)), (10.0, (0, 1))]), ((0, 2), [(30.0, (0, 2)), (10.0, (0,))]),
                               ((1, 2), [(30.0, (1, 2)), (10.0, (1,))])]:
                t = Terminal(bd, live, 40.0, np.array([9.0, 9.0, 4.0]), 0.6, "show")
                with self.subTest(live=live):
                    self.assertEqual(s._layers(live, t.c, t.pot), pots)
                    got = s._term(t, s.reach0)[:, 0]
                    want = _brute(s, t, self.board, pots)
                    self.assertLess(np.abs(got - want).max(), 1e-3 * np.abs(want).max())
        finally:
            s.cfg.allin = ()

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


class TestLaterStreetsByBuckets(unittest.TestCase):
    """MWConfig.later = "buckets": the streets after the root's bet on by hand buckets."""

    @classmethod
    def setUpClass(cls):
        cls.rs = _ranges()
        cls.turn = MultiwaySolver(cards("Kh7d2c9s"), cls.rs, 8.0, 97.5, cfg=MWConfig(later="buckets", buckets=8))
        cls.turn.run(iters=6)
        cls.g6 = cls.turn.br_gains()
        cls.turn.run(iters=94)
        cls.g100 = cls.turn.br_gains()

    def test_tensors_are_constant_sum(self):
        # every live set's shares add up to the deals' weight in every bucket triple and class, so any reaches pay
        # out exactly the pot
        tb = self.turn.tabs
        rng = np.random.default_rng(1)
        X = rng.random((3, tb.C * 2, tb.K))
        for live in [(0, 1, 2), (0, 1), (0, 2), (1, 2)]:
            with self.subTest(live=live):
                tot = sum((X[p] * tb.contract(("S", live), p, X)).sum() for p in live)
                mass = [(X[p] * tb.contract(("J",), p, X)).sum() for p in range(3)]
                self.assertAlmostEqual(mass[0] / mass[1], 1.0, places=5)
                self.assertAlmostEqual(mass[0] / mass[2], 1.0, places=5)
                self.assertAlmostEqual(tot / mass[0], 1.0, places=5)

    def test_a_build_past_its_deadline_stops(self):
        # (a quick answer's: advisor.MW_QUICK_BUILD)
        import time
        with self.assertRaises(MW.BuildTooSlow):
            MultiwaySolver(cards("Kh7d2c9s"), self.rs, 8.0, 97.5, cfg=MWConfig(later="buckets", buckets=8),
                           deadline=time.time())

    def test_values_add_up_to_the_pot(self):
        # (exactly at the root's reaches, which the deals' weight below is scaled to; within 1% after play)
        self.assertAlmostEqual(float(self.turn.values().sum()), 8.0, delta=0.08)

    def test_best_response_gains_shrink(self):
        pot = self.turn.root_pot()
        self.assertTrue(np.all(self.g100 < self.g6))
        self.assertLess(self.g100.max() / pot * 100, 1.0)

    def test_folded_player_reduction_matches_the_full_contraction(self):
        tb = self.turn.tabs
        rng = np.random.default_rng(2)
        X = rng.random((3, tb.C * 3, tb.K))
        for p, f in [(0, 2), (1, 0), (2, 1)]:
            q = 3 - p - f
            for key in [("J",), ("S", tuple(sorted((p, q))))]:
                full = tb.contract(key, p, X)
                red = tb.apply(tb.reduce(key, p, f, X[f]), X[q])
                self.assertLess(np.abs(full - red).max(), 1e-4 * np.abs(full).max())

    def test_flop_builds_later_streets_and_stays_constant_sum(self):
        s = MultiwaySolver(cards("Kh7d2c"), self.rs, 8.0, 97.5, to_act=2, pending={2},
                           cfg=MWConfig(later="buckets", buckets=6, bucket_river_step=4))
        self.assertEqual([len(lv) > 0 for lv in s.levels], [True, True])     # the turn, then the river
        s.run(iters=4)
        self.assertAlmostEqual(float(s.values().sum()), 8.0, delta=0.1)

    def test_several_flop_bet_sizes(self):
        # (the flop root bets a third and three quarters of the pot; the turn and river below keep one size)
        s = MultiwaySolver(cards("Kh7d2c"), self.rs, 8.0, 97.5, to_act=2, pending={2},
                           cfg=MWConfig(later="buckets", buckets=6, bucket_river_step=4,
                                        bet={"flop": [0.75, 0.33], "turn": 0.5, "river": 0.5}))
        self.assertEqual(s.root.names, ["check", "bet 33%", "bet 75%"])
        self.assertEqual(s.root.amounts[1:], [2.5, 6.0])
        self.assertEqual({nm for lv in s.levels for g in lv for nm in getattr(g.tree, "names", [])
                          if nm.startswith("bet")}, {"bet 50%"})
        self.assertEqual(s.root.children[2].names, ["fold", "call", "raise to 21"])   # (one raise size)
        s.run(iters=4)
        self.assertAlmostEqual(float(s.values().sum()), 8.0, delta=0.1)
        self.assertEqual(sorted(s.range_freqs()), ["bet 33%", "bet 75%", "check"])

    def test_run_stops_short_of_its_budget(self):
        # no iteration starts that would end past the budget, `reserve` iterations' time kept back
        s = MultiwaySolver(cards("Kh7d2c9s"), self.rs, 8.0, 97.5, cfg=MWConfig(later="buckets", buckets=8))
        s.run(iters=5)
        per = s.seconds / 5
        s.run(iters=1000, time_budget=12 * per, min_iters=1, reserve=3.0)
        self.assertLess(s.seconds - 5 * per, 12 * per * 1.15)
        # (iterations can run faster than the first 5 timed on a loaded machine, so the count is loose; time is the bound)
        self.assertLess(s.iters, 5 + 12 * 2)
        self.assertGreater(s.build_seconds, 0.0)


class _MWFlaky:
    """The native multiway core as the browser's module fails: its `at`th call of `what` ("run" or "eval") raises
    mwcore.Failed (a trap there), and reload() gives another (worker.js: a new instance), whose `then`th call fails."""

    def __init__(self, L, what, at, then=None):
        self.L, self.what, self.at, self.then, self.calls = L, what, at, then, 0

    def __getattr__(self, k):
        return getattr(self.L, k)

    def _call(self, name, *a):
        if name == self.what:
            self.calls += 1
            if self.calls == self.at:
                from gto import mwcore
                raise mwcore.Failed("the multiway core failed: unreachable")
        return getattr(self.L, name)(*a)

    def run(self, *a):
        return self._call("run", *a)

    def eval(self, *a):
        return self._call("eval", *a)

    def reload(self):
        return _MWFlaky(self.L, self.what, self.then)


def _core_on():
    from gto import mwcore
    return mwcore.lib() is not None


@unittest.skipUnless(_core_on(), "no C compiler for the multiway core")
class TestCore(unittest.TestCase):
    """gto/mwcore.py: a bucketed solve's iterations and its evaluation passes in C, the numpy solver's arithmetic in
    float32. An iteration from the same regrets and sums agrees to rounding; over several, regret matching's kinks (a
    regret near 0 on a hand the range hardly holds, positive in one and not the other) carry rounding into different,
    as good, strategies, so they are compared an iteration at a time."""

    @classmethod
    def setUpClass(cls):
        cls.rs = _ranges()

    def _pair(self, board, iters, rs=None, pot=8.0, stack=97.5, **kw):
        """A numpy solve `iters` iterations in, and the core's from its arrays; then one more iteration each."""
        a = MultiwaySolver(cards(board), rs or self.rs, pot, stack, **kw)
        a._mwc_off = True
        a.run(iters=iters)
        b = MultiwaySolver(cards(board), rs or self.rs, pot, stack, **kw)
        for da, db in zip(a.decisions, b.decisions):
            db.regret[...] = da.regret
            db.ssum[...] = da.ssum
        b.iters = a.iters
        a.run(iters=1)
        b.run(iters=1)
        self.assertIsNone(getattr(a, "_mwc", None))
        self.assertIsNotNone(getattr(b, "_mwc", None))
        return a, b

    def _same(self, a, b, tol=2e-3):
        for da, db in zip(a.decisions, b.decisions):
            for x, y in ((da.regret, db.regret), (da.ssum, db.ssum)):
                self.assertLess(np.abs(x - y).max(), tol * max(np.abs(x).max(), 1e-6))
        pot = a.root_pot()
        np.testing.assert_allclose(a.br_gains() / pot, b.br_gains() / pot, atol=2e-4)
        np.testing.assert_allclose(a.values() / pot, b.values() / pot, atol=2e-4)
        ea, eb = a.action_evs(), b.action_evs()
        self.assertLess(np.abs(ea - eb).max() / pot, 2e-4)

    def test_turn_and_flop_agree_with_numpy(self):
        for board, kw in (("Kh7d2c9s", dict(cfg=MWConfig(later="buckets", buckets=8))),
                          ("Kh7d2c", dict(cfg=MWConfig(later="buckets", buckets=6, bucket_river_step=4,
                                                       bet={"flop": [0.33, 0.75], "turn": 0.5, "river": 0.5}))),
                          # checked to the last player; and facing a bet and a call
                          ("Ts9s4h", dict(to_act=2, pending={2}, cfg=MWConfig(later="buckets", buckets=6,
                                                                              bucket_river_step=4)))):
            with self.subTest(board):
                self._same(*self._pair(board, 3, **kw))

    def test_all_in_player_and_two_players_agree(self):
        # a player all-in for less (its main pot and the others' side pot), and a heads-up bucketed turn
        a, b = self._pair("Kh7d2c9s", 3, pot=40.0, stack=77.5,
                          cfg=MWConfig(allin=(2,), later="buckets", buckets=8))
        self._same(a, b)
        a, b = self._pair("Kh7d2c9s", 3, rs=self.rs[:2], cfg=MWConfig(later="buckets", buckets=8))
        self._same(a, b)
        # stacks that differ: a player all-in for less below the root (its main pot, a side pot per row)
        a, b = self._pair("Kh7d2c9s", 3, pot=20.0, stack=(8.0, 60.0, 60.0), cfg=MWConfig(later="buckets", buckets=8))
        self.assertTrue(any(t.fm is not None for g in a.levels[0] for t in MW._bterminals(g.tree)))
        self._same(a, b)

    def test_four_players_agree(self):
        # 4-way flop and turn: the tensors over three others' buckets, the root street's deals pairwise
        for board, kw in (("Kh7d2c9s", dict(buckets=8)), ("Kh7d2c", dict(buckets=6, bucket_river_step=4))):
            with self.subTest(board):
                self._same(*self._pair(board, 3, rs=_ranges4(), pot=10.5, cfg=_cfg4(**kw)))
        # a player all-in: the three others bet for a side pot (the kernel's third tensor)
        self._same(*self._pair("Kh7d2c9s", 3, rs=_ranges4(), pot=30.0, stack=85.0, cfg=_cfg4(buckets=8, allin=(3,))))

    def test_river_agrees_with_numpy(self):
        # a 3-way river root (no later streets): its showdowns per combo in the core (RSHOW, exact card removal by
        # inclusion-exclusion over the cards two of the others' hands could share), the numpy solver's arithmetic: a
        # player all-in for less (a main pot and a side pot), stacks that differ, rake, the last player to act, and
        # boards where many hands tie (the board plays: one tie group; a full house on the board)
        for board, kw in (("Kh7d2c9s4h", {}),
                          ("Kh7d2c9s4h", dict(pot=40.0, stack=77.5, cfg=MWConfig(allin=(2,)))),
                          ("Kh7d2c9s4h", dict(pot=20.0, stack=(8.0, 60.0, 60.0))),
                          ("Ts9s4h4c2d", dict(cfg=MWConfig(rake_rate=0.05, rake_cap=3.0))),
                          ("Kh7d2c9s4h", dict(to_act=2, pending={2})),
                          ("AhKhQhJhTh", {}), ("7c7d7h2s2d", {})):
            with self.subTest(board, **{k: str(v) for k, v in kw.items()}):
                a, b = self._pair(board, 3, **kw)
                self.assertTrue(b._mwc.river)
                self._same(a, b)

    def test_another_solve_replaces_the_program(self):
        # one program lives in the core at a time: a solver whose program was replaced lays its tree out again from
        # its arrays and carries on where it was
        cfg = MWConfig(later="buckets", buckets=6)
        a = MultiwaySolver(cards("Kh7d2c9s"), self.rs, 8.0, 97.5, cfg=cfg).run(iters=6)
        ref = MultiwaySolver(cards("Kh7d2c9s"), self.rs, 8.0, 97.5, cfg=cfg).run(iters=3)
        b = MultiwaySolver(cards("Qs8h3d2c"), self.rs, 8.0, 97.5, cfg=cfg).run(iters=2)
        ref.run(iters=3)
        self.assertFalse(ref._mwc.valid(ref._mwc.L, ref) and b._mwc.valid(b._mwc.L, b))
        for da, db in zip(a.decisions, ref.decisions):
            np.testing.assert_array_equal(da.regret, db.regret)
            np.testing.assert_array_equal(da.ssum, db.ssum)
        np.testing.assert_array_equal(a.br_gains(), ref.br_gains())

    def test_converges_as_numpy_does(self):
        s = MultiwaySolver(cards("Kh7d2c9s"), self.rs, 8.0, 97.5, cfg=MWConfig(later="buckets", buckets=8))
        g6 = s.run(iters=6).br_gains()
        g100 = s.run(iters=94).br_gains()
        self.assertTrue(np.all(g100 < g6))
        self.assertLess(g100.max() / s.root_pot() * 100, 1.0)
        self.assertAlmostEqual(float(s.values().sum()), 8.0, delta=0.08)

    def test_threads_change_nothing(self):
        # helper threads run a pass's root-street leaves (each Exit's bucket subtree, each fold's deal masses, each
        # river showdown), each writing records of its own: regrets, sums, best-response gains, values and action EVs
        # the same, bit for bit, on 1 thread and on 4 (3-way flop and turn, a player all-in, 4-way, a 3-way river with
        # side pots)
        from gto import mwcore
        L = mwcore.lib()
        cases = (("Kh7d2c9s", self.rs, 8.0, 97.5, MWConfig(later="buckets", buckets=8)),
                 ("Kh7d2c", self.rs, 8.0, 97.5, MWConfig(later="buckets", buckets=6, bucket_river_step=4,
                                                         bet={"flop": [0.33, 0.75], "turn": 0.5, "river": 0.5})),
                 ("Kh7d2c9s", self.rs, 40.0, 77.5, MWConfig(allin=(2,), later="buckets", buckets=8)),
                 ("Kh7d2c9s", _ranges4(), 10.5, 97.5, _cfg4(buckets=8)),
                 ("Kh7d2c9s4h", self.rs, 20.0, (8.0, 60.0, 60.0), MWConfig()))
        was = mwcore.threads()
        try:
            for board, rs, pot, stack, cfg in cases:
                with self.subTest(board, players=len(rs), allin=cfg.allin):
                    got = []
                    for t in (1, 4):
                        self.assertEqual(mwcore.threads(t), t)
                        s = MultiwaySolver(cards(board), rs, pot, stack, cfg=cfg).run(iters=4)
                        self.assertIsNotNone(s._mwc)
                        got.append([d.regret.copy() for d in s.decisions] + [d.ssum.copy() for d in s.decisions]
                                   + [s.br_gains(), s.values(), s.action_evs()])
                    for x, y in zip(*got):
                        np.testing.assert_array_equal(x, y)
            # the helpers ran some of the leaves (on a busy machine a pass can be over before one is scheduled: again
            # until one has)
            for _ in range(50):
                if L.L.mw_helped() > 0:
                    break
                MultiwaySolver(cards("Kh7d2c9s"), self.rs, 8.0, 97.5, cfg=cases[0][4]).run(iters=4)
            self.assertGreater(L.L.mw_helped(), 0)
        finally:
            mwcore.threads(was)

    def test_a_failed_run_starts_the_core_again(self):
        # As the heads-up core's (tests/test_core.py): in the browser a trap in a run or a pass starts the core again
        # (mwcore.restart: worker.js reload, a new instance on a fresh memory with new helpers), and the run starts
        # over in it from the solver's arrays as they were before it, or the pass runs again: the same floats as a run
        # that didn't fail. Failing again in the new instance, numpy runs the solve and the core is off.
        from gto import mwcore
        L = mwcore.lib()
        cfg = MWConfig(later="buckets", buckets=8)
        go = lambda: MultiwaySolver(cards("Kh7d2c9s"), self.rs, 8.0, 97.5, cfg=cfg).run(iters=12)
        want = go()
        flat = lambda s: [a.copy() for d in s.decisions for a in (d.regret, d.ssum)]
        try:
            for at in (1, 2):
                with self.subTest(failing_chunk=at):
                    first = mwcore._lib = _MWFlaky(L, "run", at)
                    mwcore._restarts = 0
                    s = go()
                    self.assertEqual((mwcore._restarts, first.calls, s.iters), (1, at, 12))
                    self.assertIsInstance(mwcore.lib(), _MWFlaky)
                    self.assertIsNot(mwcore.lib(), first)
                    for x, y in zip(flat(s), flat(want)):
                        np.testing.assert_array_equal(x, y)
                    np.testing.assert_array_equal(s.br_gains(), want.br_gains())
            with self.subTest("a pass"):
                s = go()
                mwcore._lib, mwcore._restarts = _MWFlaky(L, "eval", 2), 0
                s._gains_memo = None
                np.testing.assert_array_equal(s.br_gains(), want.br_gains())
                np.testing.assert_array_equal(s.action_evs(), want.action_evs())
                self.assertEqual(mwcore._restarts, 1)
                self.assertGreater(mwcore.lib().calls, 0)   # (the passes ran on in the new instance)
            with self.subTest("again in the new instance"):
                a = MultiwaySolver(cards("Kh7d2c9s"), self.rs, 8.0, 97.5, cfg=cfg)
                a._mwc_off = True
                a.run(iters=12)
                mwcore._lib, mwcore._restarts = _MWFlaky(L, "run", 2, then=1), 0
                s = go()
                self.assertIsNone(mwcore.lib())
                self.assertEqual(s.iters, 12)
                for x, y in zip(flat(s), flat(a)):
                    np.testing.assert_array_equal(x, y)   # (numpy ran the solve from the start)
        finally:
            mwcore._lib, mwcore._restarts = L, 0

    def test_wasm_built_from_the_current_source(self):
        # (python scripts/build_core.py after a change to core/mwcore.c)
        from gto import mwcore
        for path in (mwcore.WASM, mwcore.WASM_MT):
            with open(path, "rb") as f:
                self.assertTrue(mwcore.source_hash().encode() in f.read(), f"rebuild {path}")


class TestFourWay(unittest.TestCase):
    """4+ players with the later streets by hand buckets (BucketTables._class_n): the root street per combo with pairwise
    card removal (_term_pw), the tensors over every player's buckets."""

    def test_turn_and_flop_settle_and_stay_constant_sum(self):
        rs = _ranges4()
        for board, kw, iters, bound in (("Kh7d2c9s", dict(buckets=8), 60, 0.5),
                                        ("Kh7d2c", dict(buckets=6, bucket_river_step=4), 60, 1.5)):
            with self.subTest(board):
                s = MultiwaySolver(cards(board), rs, 10.5, 97.5, cfg=_cfg4(**kw))
                self.assertEqual(s.tabs.np, 4)
                g5 = s.run(iters=5).br_gains()
                g = s.run(iters=iters - 5).br_gains()
                self.assertTrue(np.all(g < g5))
                self.assertLess(g.max() / s.root_pot() * 100, bound)
                # (the root street's pairwise deals against the tables' scaled to them: within half a percent)
                self.assertAlmostEqual(float(s.values().sum()), 10.5, delta=0.05)
                # the deals' weight below the root is the root's (the tables scaled to it), every class's showdown
                # shares adding up to it
                tb = s.tabs
                J = tb.tp[(("J",), 0)]
                for lv in ((0, 1, 2, 3), (0, 2, 3), (1, 3)):
                    tot = sum(_canon4(tb.tp[(("S", lv), p)], p) for p in lv)
                    np.testing.assert_allclose(tot, _canon4(J, 0), rtol=1e-4, atol=1e-9 * J.max())


    def test_a_player_all_in(self):
        # the BTN all-in preflop and called: the other three bet the turn and river for a side pot, the main pot (the
        # pot at the root) to the best of the four
        s = MultiwaySolver(cards("Kh7d2c9s"), _ranges4(), 30.0, 85.0, cfg=_cfg4(buckets=8, allin=(3,))).run(iters=100)
        self.assertEqual(sorted({d.player for d in s.decisions}), [0, 1, 2])
        sides = {t.side for lvl in s.levels for g in lvl for t in MW._bterminals(g.tree) if t.fm is not None}
        self.assertIn((0, 1, 2), sides)
        self.assertAlmostEqual(float(s.values().sum()), 30.0, delta=0.3)
        g = s.br_gains() / s.root_pot() * 100
        self.assertEqual(float(g[3]), 0.0)
        self.assertLess(float(g.mean()), 0.5)


def _canon4(T, p):
    """A player's tensor (C, K_o0, K_o1, K_o2, K_p) back to (C, K_0, K_1, K_2, K_3)."""
    o = [j for j in range(4) if j != p] + [p]
    return np.transpose(T, [0] + [1 + o.index(k) for k in range(4)])


class TestFlopAgainstPublished(unittest.TestCase):
    """GTO Wizard's 3-way flop study ("Playing In Position Against Two Callers", 100bb 8-max, LJ opens, SB and BB call,
    33/66/100% sizes, 20 weighted flops): checked to, the opener bets 65.9% into two and 77.0% heads-up against the BB
    (docs/multiway_research.md). Ranges shaped like the study's (scripts/compare_multiway.py STUDY_RANGES), our one
    third-pot size, two flops, a small fast tree (8 buckets, every 4th river, 40 iterations): with the turn and river bet
    on, the opener bets 56% into two and 64% heads-up here (16 buckets, 150 iterations, ten flops: 47% and 61%), where
    with them dealt out it bet 39% and 49% (ten flops: 39% and 48%)."""

    def test_opener_checked_to(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from compare_multiway import STUDY_RANGES
        R3, R2 = STUDY_RANGES()
        cfg = MWConfig(later="buckets", buckets=8, bucket_river_step=4)
        b3, b2 = [], []
        for f in ("QdAd3s", "4h7cKh"):
            s3 = MultiwaySolver(cards(f), [R3["SB"], R3["BB"], R3["UTG"]], 7.5, 97.5, to_act=2, pending={2}, cfg=cfg)
            s2 = MultiwaySolver(cards(f), [R2["BB"], R2["UTG"]], 5.5, 97.5, to_act=1, pending={1}, cfg=cfg)
            b3.append(1 - s3.run(iters=40).range_freqs()["check"])
            b2.append(1 - s2.run(iters=40).range_freqs()["check"])
        self.assertGreater(np.mean(b3), 0.45)                    # published 0.66 (several sizes); dealt out 0.39
        self.assertLess(np.mean(b3), 0.80)
        self.assertGreater(np.mean(b2), 0.55)                    # published 0.77; dealt out 0.49
        self.assertLess(np.mean(b2), 0.90)
        self.assertGreater(np.mean(b2), np.mean(b3) + 0.04)      # bets less into two


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

    def test_practice_opponents_sample_the_cached_solve(self):
        # the BB is advised (one 3-way solve), checks; the CO's move is drawn from that solve's mix for its hand at the
        # node after the check, without another solve
        adv = A.Advisor(time_budget=1e9)
        adv.mw_max_iters = 30
        st = self._state("Kh7d2c9s4h", "BB", "Kd9d", [])
        r = adv.advise(st)
        self.assertTrue(r.get("solved"))
        n_solves = len(adv._mw_cache)
        st2 = self._state("Kh7d2c9s4h", "BB", "Kd9d", [{"pos": "BB", "action": "check"}])
        sol = next(iter(adv._mw_cache.values()))
        node = sol.root.children[0]                                  # the BB checks: the CO to act
        for hole in ("AsKs", "QcJc", "8c8s"):
            with self.subTest(hole=hole):
                mix = node.avg_strategy()[0][:, sol.combo_index(cards(hole))]
                moves = [adv.play(st2, "CO", hole, seed=i) for i in range(300)]
                self.assertEqual(len(adv._mw_cache), n_solves)        # no new solve
                self.assertEqual([a["action"] for a in moves[0]["actions"]], node.names)
                for a, nm in enumerate(node.names):
                    self.assertAlmostEqual(moves[0]["actions"][a]["freq"], float(mix[a]), places=3)
                    got = np.mean([m["name"] == nm for m in moves])
                    self.assertLess(abs(got - mix[a]), 0.08)

    def test_flop_is_solved_with_later_streets_by_buckets(self):
        # (Accurate: native.MW_STREETS, the worker) checked to after two checks the opener bets AK on K72 nearly always
        adv = A.Advisor(time_budget=1e9)
        adv.mw_solve_streets = ("flop", "turn", "river")
        adv.mw_street_config = {"flop": {"later": "buckets", "buckets": 8, "bucket_river_step": 4}}
        adv.mw_max_iters = 60
        r = adv.advise(self._state("Kh7d2c", "BTN", "AsKs", [{"pos": "BB", "action": "check"},
                                                             {"pos": "CO", "action": "check"}]))
        self.assertTrue(r["multiway"] and r["solved"])
        self.assertEqual([a["action"] for a in r["actions"]], ["check", "bet 33%"])
        self.assertIn("hand buckets", " ".join(r["notes"]))
        self.assertGreater(r["actions"][1]["freq"], 0.6)

    def test_settled_solve_stops_early(self):
        # a 3-way solve stops at the first checkpoint (MW_CHECKS) where it is settled (MW_SETTLED) and isn't run again:
        # the browser's Accurate ran a 3-way river its whole 60 s to 0.01% of the pot (now 100 iterations, 15 s)
        from unittest import mock
        adv = A.Advisor(time_budget=1e9)
        with mock.patch.object(A, "MW_SETTLED", 20.0):
            st = self._state("Kh7d2c9s4h", "BB", "Kd9d", [])
            r = adv.advise(st)
            self.assertTrue(r.get("solved"))
            self.assertEqual(r["solver"]["iterations"], A.MW_CHECKS[0])
            self.assertLess(r["solver"]["exploitability_pct"], 20.0)
            self.assertEqual(adv.advise(st)["solver"]["iterations"], A.MW_CHECKS[0])

    def test_flop_answer_keeps_to_its_budget(self):
        # the tree and bucket tables, the iterations and the gains and EVs after them all fit in the budget (the tables
        # used to come on top: the browser's Accurate flops ran ~10 s past 60 s)
        adv = A.Advisor(time_budget=4.0)
        adv.mw_solve_streets = ("flop",)
        adv.mw_street_config = {"flop": {"later": "buckets", "buckets": 8, "bucket_river_step": 4,
                                         "bet": {"flop": [0.33, 0.75], "turn": 0.5, "river": 0.5}}}
        adv.mw_max_expl = 1e9
        r = adv.advise(self._state("Kh7d2c", "BTN", "AsKs", [{"pos": "BB", "action": "check"},
                                                             {"pos": "CO", "action": "check"}]))
        self.assertTrue(r.get("solved"))
        self.assertEqual([a["action"] for a in r["actions"]], ["check", "bet 33%", "bet 75%"])
        self.assertLess(r["seconds"], 4.0 * 1.15)
        self.assertGreaterEqual(r["solver"]["iterations"], 5)

    def test_later_street_narrows_through_the_flop_solve(self):
        # the BTN is advised on the flop (checked to), bets a third; the BB and CO call. On the turn their ranges come
        # from that solve's strategies (as far as it has settled), not the betting rules alone
        adv = A.Advisor(time_budget=1e9)
        adv.mw_solve_streets = ("flop",)
        adv.mw_street_config = {"flop": {"later": "buckets", "buckets": 8, "bucket_river_step": 4}}
        adv.mw_max_iters = 150
        adv.mw_trials = 50
        chk = [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "check"}]
        r = adv.advise(self._state("Kh7d2c", "BTN", "AsKs", chk))
        self.assertTrue(r.get("solved"))
        sol, players, expl = next(iter(adv._mw_lines.values()))
        trust = A.trust_of(expl, "flop")
        self.assertGreater(trust, 0.5)
        flop = chk + [{"pos": "BTN", "action": "bet", "amount": 2.5}, {"pos": "BB", "action": "call"},
                      {"pos": "CO", "action": "call"}]
        st = {"players": 6, "hero": "BTN", "hole": "AsKs", "board": "Kh7d2c9s", "stack": 100, "preflop": PRE,
              "flop": flop, "turn": chk}
        got = {}

        def grab(self_, hero, hole, board, ranges, remaining, cur, stk, t0, notes):
            got["ranges"], got["notes"] = dict(ranges), list(notes)
            # (an answer the catch-all takes as one: Advisor.advise would answer a stub with no actions again)
            return {"stub": True, "actions": [{"action": "check", "freq": 1.0}], "recommended": "check"}
        saved, narrow = A.Advisor._multiway, A._narrow_street
        A.Advisor._multiway = grab
        # (the turn's checks left out: the ranges compared are the flop's)
        A._narrow_street = lambda ranges, board, st_, *a, **k: None if st_.name == "turn" else narrow(ranges, board, st_, *a, **k)
        try:
            adv.advise(st)
            solved = got.pop("ranges"), got.pop("notes")
            adv._mw_lines.clear()                                    # no solve to follow: the rules alone
            adv.advise(st)
            rules = got["ranges"]
        finally:
            A.Advisor._multiway, A._narrow_street = saved, narrow
        self.assertTrue(any(n.startswith("flop: 3-way, ranges narrowed through the solve your advice came from")
                            for n in solved[1]))
        # the BB's call: its range at the solve's root times its call, smoothed, mixed with the rules' by trust
        node = sol.root.children[1]
        bb = players.index("BB")
        s = node.avg_strategy()[0][1].astype(float)
        want = sol.reach0[bb, 0] * ((1 - adv.prior_smoothing) * s + adv.prior_smoothing / len(node.names))
        want = trust * sol.expand(want) + (1 - trust) * rules["BB"]
        live = (sol.expand(np.ones(sol.hands.n)) > 0) & np.array([cards("9s")[0] not in c for c in ALL_COMBOS])
        self.assertLess(np.abs(solved[0]["BB"] - want)[live].max(), 1e-6)
        self.assertGreater(np.abs(solved[0]["BB"] - rules["BB"])[live].max(), 0.05)

    def test_flop_uses_the_betting_rules_where_not_enabled(self):
        adv = A.Advisor(time_budget=1e9)
        adv.mw_trials = 100
        r = adv.advise(self._state("Kh7d2c", "BB", "Kd9d", []))
        self.assertTrue(r["multiway"])
        self.assertFalse(r.get("solved", False))

    def test_four_way_river_turn_and_flop_are_solved(self):
        adv = A.Advisor(time_budget=2.0)
        adv.mw_trials = 200
        adv.mw4_street_config = {"flop": {"later": "buckets", "buckets": 6, "bucket_river_step": 4},
                                 "turn": {"later": "buckets", "buckets": 8}}
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5},
               {"pos": "CO", "action": "call"}, {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"},
               {"pos": "BB", "action": "call"}]
        X4 = [{"pos": p, "action": "check"} for p in ("BB", "HJ", "CO", "BTN")]
        st = {"players": 6, "hero": "BB", "hole": "Kd9d", "board": "Kh7d2c9s4h", "stack": 100, "preflop": pre,
              "flop": X4, "turn": X4, "river": []}
        r = adv.advise(st)
        self.assertTrue(r["multiway"] and r["solved"], r.get("notes"))
        self.assertEqual(r["position"], "1 of 4 to act")
        self.assertLess(r["solver"]["exploitability_pct"], 1.0)
        self.assertTrue(any("4-player CFR" in n for n in r["notes"]))
        # the turn and (where the speed solves it) the flop: the 4-player solve, the later streets by hand buckets
        r = adv.advise(dict(st, board="Kh7d2c9s", turn=[]))
        self.assertTrue(r["multiway"] and r["solved"], r.get("notes"))
        self.assertTrue(any("4-player CFR" in n and "8 hand buckets" in n for n in r["notes"]), r["notes"])
        self.assertLess(r["solver"]["exploitability_pct"], 1.0)
        self.assertEqual([a["action"] for a in r["actions"]], ["check", "bet 50%"])
        flop = dict(st, board="Kh7d2c", flop=[])
        self.assertFalse(adv.advise(flop).get("solved", False))     # (the default solves turns and rivers)
        adv.mw_solve_streets = ("flop", "turn", "river")
        adv.time_budget = 6.0
        r = adv.advise(flop)
        self.assertTrue(r["solved"], r.get("notes"))
        self.assertLess(r["solver"]["exploitability_pct"], 3.0)
        adv.time_budget = 2.0
        # facing a bet with players behind: fold, call, and the only raise in the tree, hero's
        r = adv.advise(dict(st, hero="CO", hole="Ks8s", river=[{"pos": "BB", "action": "bet", "amount": 5},
                                                               {"pos": "HJ", "action": "call"}]))
        self.assertTrue(r["solved"])
        self.assertEqual([a["action"].split()[0] for a in r["actions"]], ["fold", "call", "raise"])

    def test_four_way_with_a_player_all_in_is_solved(self):
        # the 15bb CO shoves over the HJ's open and three call: on the turn the other three bet for the side pot (the
        # 4-player solve with a player all-in), where the betting rules answered before
        adv = A.Advisor(time_budget=2.0)
        adv.mw4_street_config = {"flop": {"later": "buckets", "buckets": 6, "bucket_river_step": 4},
                                 "turn": {"later": "buckets", "buckets": 8}}
        F = lambda p: {"pos": p, "action": "fold"}
        X = lambda p: {"pos": p, "action": "check"}
        pre = [F("UTG"), {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "all-in"},
               {"pos": "BTN", "action": "call"}, F("SB"), {"pos": "BB", "action": "call"}, {"pos": "HJ", "action": "call"}]
        r = adv.advise({"players": 6, "stack": 100, "stacks": {"CO": 15}, "hero": "HJ", "hole": "7s7c",
                        "board": "Kh7d2c9s", "preflop": pre, "flop": [X("BB"), X("HJ"), X("BTN")], "turn": [X("BB")]})
        self.assertTrue(r.get("solved"), r.get("notes"))
        self.assertTrue(any("4-player CFR" in n for n in r["notes"]), r["notes"])
        self.assertTrue(any(n.startswith("CO is all-in: the main pot (60.5 bb) goes to the best of the four hands")
                            for n in r["notes"]), r["notes"])
        self.assertLess(r["solver"]["exploitability_pct"], 1.0)

    def test_jam_request_offers_all_in_at_the_root_only(self):
        # (the page grading your all-in asks again with "jam": all-in at your decision however deep, the tree below as
        # it was: all-in everywhere made a 4-way river 401 nodes, 4% of the pot from settled)
        adv = A.Advisor(time_budget=2.0)
        pre = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5},
               {"pos": "CO", "action": "call"}, {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"},
               {"pos": "BB", "action": "call"}]
        X4 = [{"pos": p, "action": "check"} for p in ("BB", "HJ", "CO", "BTN")]
        st = {"players": 6, "hero": "BB", "hole": "Kd9d", "board": "Kh7d2c9s4h", "stack": 100, "preflop": pre,
              "flop": X4, "turn": X4, "river": [], "jam": True}
        r = adv.advise(st)
        acts = {a["action"]: a for a in r["actions"]}
        self.assertIn("all-in", acts)
        self.assertLess(r["solver"]["nodes"], 120)
        self.assertLess(r["solver"]["exploitability_pct"], 1.0)
        self.assertLess(acts["all-in"]["ev_bb"], max(a["ev_bb"] for a in r["actions"]) - 1.0)

    def test_a_bet_over_the_short_stack_is_solved_with_side_pots(self):
        # the 25bb BB checks the turn and the CO bets 30, more than the BB has behind: each player bets its own stack
        # (the BB's call all-in for less plays for the main pot), where everyone played the shortest stack before and
        # a bet over it was left to the betting rules
        adv = A.Advisor(time_budget=3.0)
        F = lambda p: {"pos": p, "action": "fold"}
        X = lambda p: {"pos": p, "action": "check"}
        pre = [F("UTG"), F("HJ"), {"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "call"}, F("SB"),
               {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "stack": 100, "stacks": {"BB": 25}, "hero": "BTN", "hole": "KsQs",
                        "board": "Kh7d2c3s", "preflop": pre, "flop": [X("BB"), X("CO"), X("BTN")],
                        "turn": [X("BB"), {"pos": "CO", "action": "bet", "amount": 30}]})
        self.assertTrue(r.get("solved"), r.get("notes"))
        self.assertEqual(r["stack_bb"], 97.5)
        self.assertTrue(any(n.startswith("stacks differ: each player bets their own stack (BB 22.5, CO 97.5, BTN 97.5")
                            for n in r["notes"]), r["notes"])
        self.assertLess(r["solver"]["exploitability_pct"], 1.0)

    def test_two_all_in_and_one_left_with_chips(self):
        # two betting with two all-in on the river (layered pots), and a player alone with chips facing two all-ins on
        # the flop (the cards to come dealt out: exact) are solved, not left to the rules
        adv = A.Advisor(time_budget=2.0)
        F = lambda p: {"pos": p, "action": "fold"}
        X = lambda p: {"pos": p, "action": "check"}
        pre = [F("UTG"), {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "all-in"},
               {"pos": "BTN", "action": "all-in"}, F("SB"), {"pos": "BB", "action": "call"}, {"pos": "HJ", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "HJ", "hole": "AhKd", "board": "Kh7d2c9s4h", "stack": 100,
                        "stacks": {"CO": 12, "BTN": 25}, "preflop": pre, "flop": [X("BB"), X("HJ")],
                        "turn": [X("BB"), X("HJ")], "river": [X("BB")]})
        self.assertTrue(r["solved"], r.get("notes"))
        self.assertTrue(any("main pot (48.5 bb)" in n and "side pot (39.0 bb)" in n for n in r["notes"]), r["notes"])
        pre2 = [F("UTG"), {"pos": "HJ", "action": "raise", "to": 2.5}, F("CO"), {"pos": "BTN", "action": "call"}, F("SB"),
                {"pos": "BB", "action": "call"}]
        r = adv.advise({"players": 6, "hero": "BTN", "hole": "QhQd", "board": "Kh7d2c", "stack": 100,
                        "stacks": {"BTN": 30, "BB": 20}, "preflop": pre2,
                        "flop": [{"pos": "BB", "action": "all-in"}, {"pos": "HJ", "action": "all-in"}]})
        self.assertTrue(r["solved"], r.get("notes"))
        self.assertEqual([a["action"] for a in r["actions"]], ["fold", "call"])


if __name__ == "__main__":
    unittest.main()


class TestAllIn(unittest.TestCase):
    """MWConfig.allin: a player all-in at the root never acts; the other two bet for the side pot."""

    def test_river_and_turn_with_a_player_all_in(self):
        R = _ranges()
        # BTN (2) all-in for 20 over a 8bb pot: the main pot 8 + 3 x 20 = 68 at the flop... here the river's root:
        # main 40, the pot 40, BB and CO 77.5 behind to bet for the side pot
        for board, cfg in (("Kh7d2c9s4h", MWConfig(allin=(2,))),
                           ("Kh7d2c9s", MWConfig(allin=(2,), later="buckets", buckets=8))):
            with self.subTest(board=board):
                s = MultiwaySolver(cards(board), R, 40.0, 77.5, cfg=cfg).run(iters=60)
                self.assertNotIn(2, [d.player for d in s.decisions])     # the all-in player never acts
                self.assertAlmostEqual(float(s.values().sum()), 40.0, delta=0.4)   # constant-sum: the pot
                g = s.br_gains() / s.root_pot() * 100
                self.assertAlmostEqual(float(g[2]), 0.0, places=6)
                self.assertLess(float(g.mean()), 3.0)
                # the root offers the bets to its first player, and a fold leaves the main pot to the others
                self.assertEqual(s.root.player, 0)
                self.assertIn("check", s.root.names)


class TestSidePots(unittest.TestCase):
    """Stacks that differ (MultiwaySolver's `stack` per player): a player all-in for less, at the root or below it,
    contests the main pot; the players with more in the side pot too (MultiwaySolver._layers)."""

    @staticmethod
    def _at(s, path):
        node = s.root
        for nm in path:
            node = node.children[node.names.index(nm)]
        return node

    def test_a_short_stack_all_in_below_the_root(self):
        R = _ranges()
        s = MultiwaySolver(cards("Kh7d2c9s4h"), R, 20.0, (8.0, 60.0, 60.0))
        self.assertEqual(s.root.names, ["check", "all-in"])
        # the short stack's all-in raised and re-raised: a main pot of 20 + 3 x 8 between all three, the rest between
        # the two deep players
        t = self._at(s, ["all-in", "raise to 35", "all-in", "call"])
        self.assertEqual(s._layers(t.live, t.c, t.pot), [(44.0, (0, 1, 2)), (104.0, (1, 2))])
        # the raise nobody matched comes back: a pot of 36 between the short stack and the raiser
        t = self._at(s, ["all-in", "raise to 35", "fold"])
        self.assertEqual((t.live, t.pot, list(t.c)), ((0, 1), 36.0, [8.0, 8.0, 0.0]))
        # a call all-in for less, then the third player folds the raise behind it: the main pot to the best of the two
        # live, the folded player's chips over the short stack's (and the raiser's matching them) the raiser's alone
        t = self._at(s, ["check", "bet 50%", "raise to 40", "call", "fold"])
        self.assertEqual(s._layers(t.live, t.c, t.pot), [(44.0, (0, 2)), (4.0, (2,))])
        for d in s.decisions:     # (all-in, the short stack never acts again)
            if d.player == 0:
                self.assertEqual(d.c[0], 0.0)
        s.run(iters=40)
        self.assertAlmostEqual(float(s.values().sum()), 20.0, places=4)    # constant-sum
        self.assertLess(float((s.br_gains() / s.root_pot() * 100).mean()), 1.0)
        # the deepest player bets only what another can match
        s = MultiwaySolver(cards("Kh7d2c9s4h"), R, 20.0, (60.0, 30.0, 8.0))
        self.assertEqual(s.stacks, (30.0, 30.0, 8.0))

    def test_an_all_in_this_street_shares_the_main_pot_with_its_callers(self):
        # player 2 all-in for 20 into 20 on the river: the calls of it are the main pot's (a main pot fixed at the
        # root's chips gave a call and a fold behind it to the side pot, the caller's alone)
        s = MultiwaySolver(cards("Kh7d2c9s4h"), _ranges(), 20.0, 60.0, bets=[0.0, 0.0, 20.0], to_act=0,
                           pending={0, 1}, cfg=MWConfig(allin=(2,)))
        t = self._at(s, ["call", "fold"])
        self.assertEqual(s._layers(t.live, t.c, t.pot), [(60.0, (0, 2))])
        t = self._at(s, ["all-in", "call"])
        self.assertEqual(s._layers(t.live, t.c, t.pot), [(80.0, (0, 1, 2)), (80.0, (0, 1))])
        s.run(iters=30)
        self.assertAlmostEqual(float(s.values().sum()), 20.0, places=4)     # (chips in count from the root street)

    def test_turn_by_buckets_with_a_short_stack(self):
        s = MultiwaySolver(cards("Kh7d2c9s"), _ranges(), 20.0, (8.0, 60.0, 60.0),
                           cfg=MWConfig(later="buckets", buckets=8)).run(iters=60)
        sides = {t.side for g in s.levels[0] for t in MW._bterminals(g.tree) if t.fm is not None}
        self.assertEqual(sides, {(1, 2), (1,), (2,)})
        self.assertAlmostEqual(float(s.values().sum()), 20.0, delta=0.4)
        self.assertLess(float((s.br_gains() / s.root_pot() * 100).mean()), 1.0)


class TestNWay(unittest.TestCase):
    """Four or more players (or two all-in): pairwise terminals, layered pots, one bet a street with four or more."""

    @classmethod
    def setUpClass(cls):
        R6 = A.preflop_ranges([dict(a, action="call") if a["pos"] in ("SB",) else a for a in
                               [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5},
                                {"pos": "CO", "action": "call"}, {"pos": "BTN", "action": "call"},
                                {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}]], pf.Table(pf.SEATS[6]), 100.0)
        cls.R6 = R6
        cls.board = tuple(cards("Kh7d2c9s4h"))

    def test_pairwise_values_per_deal_near_exact_three_way(self):
        # the N-player terminal on three players against the exact 3-way kernels: the EV per deal within ~1% of the pot
        rs = _ranges()
        s = MultiwaySolver(self.board, rs, 20.0, 80.0)
        R, bd = s.reach0, s.root.bd
        m_ex = [s._mass(bd, R, *[j for j in range(3) if j != p])[0] for p in range(3)]
        m_pw = [x[0] for x in s._pw_masses(bd, R)[0]]
        for live, c in (((0, 1, 2), (6.0, 6.0, 0.0)), ((0, 2), (6.0, 0.0, 0.0))):
            t = Terminal(bd, live, 20.0 + sum(c), np.array(c), 0.0, "show")
            ex, pw = s._term(t, R)[:, 0], s._term_pw(t, R)[:, 0]
            for p in range(3):
                ok = m_ex[p] > 1e-9
                d = np.abs(pw[p][ok] / m_pw[p][ok] - ex[p][ok] / m_ex[p][ok]) / t.pot * 100
                w = R[p, 0][ok] * m_ex[p][ok]
                with self.subTest(live=live, p=p):
                    self.assertLess(float((w * d).sum() / w.sum()), 1.2)
                    self.assertLess(float(d.max()), 4.0)

    def test_layered_pots(self):
        rs = [self.R6[p] for p in ("SB", "BB", "CO", "BTN")]
        cfg = MWConfig(later="deal", allin=(0, 3), inv0=(10.0, 25.0, 25.0, 25.0), dead=(2.5,))
        s = MultiwaySolver(self.board, rs, 87.5, 75.0, cfg=cfg, to_act=1)
        self.assertEqual(s._layers((0, 1, 2, 3), [0.0] * 4, 87.5), [(42.5, (0, 1, 2, 3)), (45.0, (1, 2, 3))])
        # the two who bet: 10 more each makes a pot of theirs alone
        self.assertEqual(s._layers((0, 1, 2, 3), [0.0, 10.0, 10.0, 0.0], 107.5)[-1], (20.0, (1, 2)))
        self.assertEqual({d.player for d in s.decisions}, {1, 2})   # the all-in players never act
        s.run(iters=60)
        g = s.br_gains() / s.root_pot() * 100
        self.assertLess(float(g.max()), 1.0)
        self.assertAlmostEqual(float(s.values().sum()), 87.5, delta=1.5)

    def test_four_and_five_way_rivers_settle(self):
        cfg = MWConfig(later="deal", max_raises=1, root_raise=True)
        for seats, kw in ((("BB", "HJ", "CO", "BTN"), {}),
                          (("SB", "BB", "HJ", "CO", "BTN"), {"bets": [6.0, 6.0, 0, 0, 0], "to_act": 2, "n_raises": 1})):
            with self.subTest(players=len(seats), **{k: str(v) for k, v in kw.items()}):
                s = MultiwaySolver(self.board, [self.R6[p] for p in seats], 12.5, 97.5, cfg=cfg, **kw).run(iters=60)
                for d in s.decisions:   # (a vanishing line's ratio of near-zero reaches overflowed the regrets before)
                    self.assertTrue(np.isfinite(d.regret).all())
                g = s.br_gains() / s.root_pot() * 100
                self.assertLess(float(g.mean()), 0.5)
                self.assertAlmostEqual(float(s.values().sum()), 12.5, delta=0.4)   # (constant-sum: the root's pot)
                # the root's player may raise; nobody else does
                raisers = {d.player for d in s.decisions if any(nm.startswith("raise") for nm in d.names)}
                self.assertLessEqual(raisers, {s.root.player})


class TestTournament(unittest.TestCase):
    """A tournament's payoffs (MWConfig.icm, gto/icm.py MWPayoffs): every terminal pays each player the prize equity
    of its outcomes in chip equivalents, linear in the solver's shares (the outcomes averaged over the orders of the
    live hands), so the passes (numpy's and the compiled core's) stay contractions."""
    SEATS = ("BB", "CO", "BTN")

    @staticmethod
    def _pay(seats, behind, contrib_each, payouts, short=8.0):
        """The solve's seats with contrib_each in and `behind` chips behind, the small blind (`short`) out of the pot."""
        from gto import icm
        contrib = {s: contrib_each for s in seats}
        stacks = {s: contrib_each + b for s, b in zip(seats, behind)}
        stacks["SB"], contrib["SB"] = short, 0.5
        return icm.MWPayoffs(icm.Model(stacks, payouts), seats, contrib)

    def test_batched_equities_are_icm(self):
        from gto import icm
        rng = np.random.default_rng(1)
        for n, pay in ((3, [50, 30, 20]), (6, [50, 30, 20]), (9, [30, 20, 14, 10, 8, 6, 5, 4, 3]), (5, [60, 40])):
            S = rng.uniform(0, 30, (60, n))
            S[rng.random((60, n)) < 0.2] = 0.0      # (busted players: the places below everyone left, shared)
            np.testing.assert_allclose(icm.equities_batch(S, pay), [icm.equities(x, pay) for x in S], atol=1e-9)

    def test_weights(self):
        # winner-take-all: chips (from the start of the hand): a pot, a main and a side pot, a lone winner; a folded
        # player's chips in
        P = self._pay(self.SEATS, (22.5,) * 3, 2.5, [100.0])
        W = P.weights([((0, 1, 2), (5, 5, 5), [(23.0, (0, 1, 2))]),
                       ((0, 1, 2), (3, 8, 8), [(17.0, (0, 1, 2)), (10.0, (1, 2))]),
                       ((1,), (2, 4, 1), [(15.0, (1,))])])
        np.testing.assert_allclose(W[0], [[-7.5, 23.0]] * 3, atol=1e-9)
        np.testing.assert_allclose(W[1], [[-5.5, 17, 0], [-10.5, 17, 10], [-10.5, 17, 10]], atol=1e-9)
        np.testing.assert_allclose(W[2], [[-4.5, 0], [0, 8.5], [-3.5, 0]], atol=1e-9)
        # on a bubble, averaged over the orders of the live hands the weights pay each player its exact prize value
        import itertools
        P = self._pay(self.SEATS, (17.5, 17.5, 17.5), 7.5, [50, 30, 20])
        for live, c, layers in (((0, 1, 2), (6, 6, 6), [(41.0, (0, 1, 2))]),
                                ((0, 2), (6, 0, 6), [(35.0, (0, 2))]),
                                ((0, 1, 2), (3, 10, 10), [(32.0, (0, 1, 2)), (14.0, (1, 2))])):
            W = P.weights([(live, c, layers)])[0]
            put = P._put0.copy(); put[P._ix] += c
            got, want = np.zeros(3), np.zeros(3)
            perms = list(itertools.permutations(live))
            for perm in perms:
                won = np.zeros(len(put))
                wins = np.zeros((3, len(layers)))
                for k, (amt, elig) in enumerate(layers):
                    w = next(x for x in perm if x in elig)
                    won[P._ix[w]] += amt
                    wins[w, k] = 1.0
                want += P.model.value_batch(put[None], won[None])[0, P._ix]
                got += W[:, 0] + (W[:, 1:] * wins).sum(axis=1)
            np.testing.assert_allclose(got / len(perms), want / len(perms), atol=1e-9)
            for q in range(3):
                if q not in live:      # (a folded player: the mean over the orders)
                    self.assertAlmostEqual(float(W[q, 0]), float(want[q] / len(perms)), places=9)

    def _solves(self, board, rs, pot, stack, cfgkw, seats, iters, core, payouts, **kw):
        npl = len(rs)
        behind = stack if isinstance(stack, tuple) else (stack,) * npl
        out = []
        for P in (None, self._pay(seats, behind, (pot - 0.5) / npl, payouts)):
            s = MultiwaySolver(cards(board), rs, pot, stack, cfg=MWConfig(icm=P, **cfgkw), **kw)
            if not core:
                s._mwc_off = True
            out.append(s.run(iters=iters))
        return out

    def test_winner_take_all_solves_are_chips(self):
        """Prize equity in a winner-take-all is chips: the same strategies, best-response gains and EVs (a river per
        combo, turns by buckets in numpy and the core, a side pot, four players)."""
        R3, R4 = _ranges(), _ranges4()
        cases = [("river", "Kh7d2c9s4h", R3, 8.0, 97.5, {}, self.SEATS, False),
                 ("turn numpy", "Kh7d2c9s", R3, 8.0, 97.5, dict(later="buckets", buckets=8), self.SEATS, False),
                 ("side pot numpy", "Kh7d2c9s", R3, 20.0, (8.0, 60.0, 60.0), dict(later="buckets", buckets=8), self.SEATS,
                  False)]
        if _core_on():
            cases += [("river core", "Kh7d2c9s4h", R3, 8.0, 97.5, {}, self.SEATS, True),
                      ("turn core", "Kh7d2c9s", R3, 8.0, 97.5, dict(later="buckets", buckets=8), self.SEATS, True),
                      ("four core", "Kh7d2c9s", R4, 10.5, 95.0, dict(later="buckets", max_raises=1, root_raise=True,
                                                                     bet=dict(A.MW4_BETS), buckets=8),
                       ("BB", "HJ", "CO", "BTN"), True)]
        for name, board, rs, pot, stack, cfgkw, seats, core in cases:
            with self.subTest(name):
                a, b = self._solves(board, rs, pot, stack, cfgkw, seats, 8, core, [100.0])
                for da, db in zip(a.decisions, b.decisions):
                    np.testing.assert_allclose(da.ssum, db.ssum, atol=1e-3 * max(float(da.ssum.max()), 1e-6))
                pot_r = a.root_pot()
                np.testing.assert_allclose(a.br_gains() / pot_r, b.br_gains() / pot_r, atol=1e-6)
                np.testing.assert_allclose(a.action_evs() / pot_r, b.action_evs() / pot_r, atol=1e-6)
                np.testing.assert_allclose(a.values() / pot_r, b.values() / pot_r, atol=1e-6)

    @unittest.skipUnless(_core_on(), "no C compiler for the multiway core")
    def test_core_agrees_with_numpy_in_prizes(self):
        R3, R4 = _ranges(), _ranges4()
        for name, board, rs, pot, stack, cfgkw, seats, kw in (
                ("turn", "Kh7d2c9s", R3, 8.0, 22.5, dict(later="buckets", buckets=8), self.SEATS, {}),
                ("facing a bet and a call", "Kh7d2c9s", R3, 8.0, 22.5, dict(later="buckets", buckets=8), self.SEATS,
                 dict(bets=[0.0, 4.0, 4.0], to_act=0, pending={0})),
                ("side pots", "Kh7d2c9s", R3, 20.0, (8.0, 30.0, 30.0), dict(later="buckets", buckets=8), self.SEATS, {}),
                ("river", "Kh7d2c9s4h", R3, 8.0, 22.5, {}, self.SEATS, {}),
                ("river side pots", "Kh7d2c9s4h", R3, 20.0, (8.0, 30.0, 30.0), {}, self.SEATS, {}),
                ("flop", "Kh7d2c", R3, 8.0, 22.5, dict(later="buckets", buckets=6, bucket_river_step=4), self.SEATS, {}),
                ("four", "Kh7d2c9s", R4, 10.5, 20.0, dict(later="buckets", max_raises=1, root_raise=True,
                                                         bet=dict(A.MW4_BETS), buckets=8), ("BB", "HJ", "CO", "BTN"), {})):
            with self.subTest(name):
                npl = len(rs)
                behind = stack if isinstance(stack, tuple) else (stack,) * npl
                P = self._pay(seats, behind, (pot - 0.5) / npl, [50, 30, 20])
                mk = lambda: MultiwaySolver(cards(board), rs, pot, stack, cfg=MWConfig(icm=P, **cfgkw), **kw)
                a = mk()
                a._mwc_off = True
                a.run(iters=3)
                b = mk()
                for da, db in zip(a.decisions, b.decisions):
                    db.regret[...] = da.regret
                    db.ssum[...] = da.ssum
                b.iters = a.iters
                a.run(iters=1)
                b.run(iters=1)
                self.assertIsNotNone(getattr(b, "_mwc", None))
                for da, db in zip(a.decisions, b.decisions):
                    self.assertLess(np.abs(da.regret - db.regret).max(), 2e-3 * max(np.abs(da.regret).max(), 1e-6))
                pot_r = a.root_pot()
                np.testing.assert_allclose(a.br_gains() / pot_r, b.br_gains() / pot_r, atol=2e-4)
                np.testing.assert_allclose(a.values() / pot_r, b.values() / pot_r, atol=2e-4)
                ea = a.action_evs()
                self.assertLess(np.abs(ea - b.action_evs()).max() / pot_r, 2e-4)
                if kw:      # (EVs count from folding now: the fold's is 0)
                    self.assertLess(float(np.abs(ea[0]).max()), 1e-6)

    def test_prizes_against_the_exact_game(self):
        """On a bubble 3-way river (thinned ranges, every deal summed: scripts/bench_mw_icm.py) the prize solve gives up
        almost nothing in exact prize equity where the chip solve gives up percents of the pot, and plays the all-in
        less."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import bench_mw_icm as B
        from gto import icm
        from gto.evaluator import eval_board
        board = cards("Kh7d2c9s4h")
        Rp = A.preflop_ranges(B.PRE, pf.Table(pf.SEATS[6]), 100.0)
        st = eval_board(board)
        rs = [B.thin(np.asarray(Rp[p]), st, 25) for p in B.SEATS]
        contrib = {s: 7.5 for s in B.SEATS}
        contrib["SB"] = 0.5
        P = icm.MWPayoffs(icm.Model({"BB": 25.0, "CO": 25.0, "BTN": 25.0, "SB": 8.0}, [50, 30, 20]), B.SEATS, contrib)
        chips = MultiwaySolver(board, rs, 23.0, 17.5, cfg=MWConfig()).run(iters=200)
        prizes = MultiwaySolver(board, rs, 23.0, 17.5, cfg=MWConfig(icm=P)).run(iters=200)
        _, gc = B.Exact(chips, P, board, prizes.icm_base).report()
        v, gp = B.Exact(prizes, P, board, prizes.icm_base).report()
        self.assertGreater(float(gc.mean()) / 23.0 * 100, 1.0)
        self.assertLess(float(gp.mean()) / 23.0 * 100, 0.1)
        np.testing.assert_allclose(prizes.values() / 23.0, v / 23.0, atol=0.005)
        jam = lambda s: s.range_freqs().get("all-in", 0.0)
        self.assertLess(jam(prizes), jam(chips))


class TestTournamentAdvice(unittest.TestCase):
    def test_a_tournament_3way_flop_is_solved_in_prizes(self):
        """A 3-way flop on the bubble is solved in prize equity (the note says so; EVs from folding now, its own 0),
        and a library spot's flop is solved live, not read from the chip library."""
        from gto import native
        R = lambda pos, action, to=None: {"pos": pos, "action": action, **({"to": to} if to is not None else {})}
        adv = native.advisor("fast")
        st = {"players": 4, "bb": 1, "stacks": {"CO": 40, "BTN": 25, "SB": 8, "BB": 25}, "hero": "BB", "hole": "Qh9h",
              "board": "Kh7d2c", "preflop": [R("CO", "raise", 2), R("BTN", "call"), R("SB", "fold"), R("BB", "call")],
              "payouts": [50, 30, 20]}
        st["flop"] = [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 2.0},
                      {"pos": "BTN", "action": "call"}]
        r = adv.advise(st)
        self.assertTrue(r.get("solved"), r.get("notes"))
        self.assertTrue(any("prize equity" in n and "3-way" in n for n in r["notes"]), r["notes"])
        ev = {a["action"]: a["ev_bb"] for a in r["actions"]}
        self.assertEqual(ev["fold"], 0.0)
        pre = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"), R("BB", "call")]
        lib = {"players": 6, "hero": "BB", "hole": "Qh9h", "board": "Kc7d2h", "preflop": pre, "flop": []}
        self.assertTrue(adv.advise(lib).get("precomputed"))
        r = adv.advise(dict(lib, payouts=[50, 30, 20], budget=10))   # (Fast's 5 s left 17 iterations on a loaded machine)
        self.assertTrue(r.get("solved"), r.get("notes")); self.assertFalse(r.get("precomputed"))
        self.assertTrue(any("prize equity" in n for n in r["notes"]), r["notes"])
