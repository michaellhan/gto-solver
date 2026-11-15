"""Exploit mode in multiway pots: the pool's strategy at every opponent decision of the 3- and 4-way solves
(gto/exploit.py MWPoolLock) with hero best-responding, and the betting rules following the pool (Advisor._multiway)."""
import time
import unittest
from types import SimpleNamespace as N
from unittest import mock

import numpy as np

from gto import exploit as ex
from gto import advisor as A
from gto.advisor import Advisor, _keep, _MW_SHADE, FOLD_SHADE
from gto.cards import ALL_COMBOS, cards
from gto.multiway_solver import MultiwaySolver, MWConfig
from gto.solver import Config
from tests.test_multiway_solver import _ranges

R = lambda p, a, amt=None: dict({"pos": p, "action": a}, **({"amount": amt} if amt else {}))
PRE = [R("UTG", "fold"), R("HJ", "fold"), R("CO", "raise", 2.5), R("BTN", "call"), R("SB", "fold"), R("BB", "call")]
CHK = [R(p, "check") for p in ("BB", "CO", "BTN")]
PRE4 = [R("UTG", "fold"), R("HJ", "raise", 2.5), R("CO", "call"), R("BTN", "call"), R("SB", "fold"), R("BB", "call")]
CHK4 = [R(p, "check") for p in ("BB", "HJ", "CO", "BTN")]


def _advise(hole, board, opp=None, rules=False, four=False):
    """BB leads into the field on the river after every street checked through (a 3-way pot, or 4-way with `four`):
    the 3-way (4-way) solve to a fixed iteration count, or the betting rules."""
    adv = Advisor(Config(depth=1), time_budget=1e9)
    adv.mw_max_iters = 150
    adv.mw_trials = 1500
    if rules:
        adv.mw_solve_streets = ()
    chk = CHK4 if four else CHK
    st = {"players": 6, "hero": "BB", "hole": hole, "board": board, "stack": 100, "preflop": PRE4 if four else PRE,
          "flop": chk, "turn": chk, "river": []}
    if opp:
        st["opponents"] = opp
    r = adv.advise(st)
    assert "actions" in r, r
    return r, {a["action"]: a for a in r["actions"]}


class TestMWPoolLock(unittest.TestCase):
    def _lock(self, board, info):
        """A lock on one stub decision of player 1 facing a bet (fold, call: no raise to cut), its annotation given."""
        live = np.array([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in board and b not in board])
        lock = ex.MWPoolLock(N(hands=N(active=live)), [1])
        node = N(player=1, names=["fold", "call"], bd=N(boards=[board]))
        lock._info = {id(node): info}
        return lock, node, len(live)

    def test_multiway_rows_and_heads_up_rows(self):
        """A street three or more began: the online pool defends as at equilibrium (the over-fold data is heads-up) and
        the live one calls more; a street two began: the heads-up rows; committed: neither."""
        river = tuple(cards("Qs7c3d8h2s"))
        for n_in, comm, prof, want in ((2, False, "online", np.interp(0.5, ex.FOLD_X, ex.FOLD_SHIFT["river"])),
                                       (3, False, "online", 0.0),
                                       (3, False, "live", np.interp(0.5, ex.FOLD_X, ex.LIVE["MW_FOLD_SHIFT"]["river"])),
                                       (3, True, "live", 0.0), (2, True, "online", 0.0)):
            lock, node, n = self._lock(river, ("river", True, True, 0.5, n_in, comm))
            t = ex.set_profile(prof)
            try:
                out = lock(node, np.full((1, 2, n), 0.5, dtype=np.float32), np.ones((1, n)))
            finally:
                ex.reset_profile(t)
            self.assertAlmostEqual(float(out[0, 0].mean()) - 0.5, want, places=3, msg=(n_in, comm, prof))
        self.assertLess(np.interp(0.5, ex.FOLD_X, ex.LIVE["MW_FOLD_SHIFT"]["river"]), -0.04)
        # a later street's batch (hand buckets): committed per row, by the line each row came from
        lock, node, _ = self._lock(river, ("river", True, True, 0.5, 2, np.array([True, False])))
        node.bd = None
        out = lock(node, np.full((2, 2, 8), 0.5, dtype=np.float32), np.ones((2, 8)))
        self.assertAlmostEqual(float(out[0, 0].mean()), 0.5, places=5)
        self.assertGreater(float(out[1, 0].mean()), 0.6)
        # hero's own decisions are never locked
        lock, node, n = self._lock(river, ("river", True, True, 0.5, 2, False))
        node.player = 0
        s = np.full((1, 2, n), 0.5, dtype=np.float32)
        self.assertIs(lock(node, s, np.ones((1, n))), s)

    def test_annotations_over_a_turn_solve(self):
        """MWPoolLock.info over a 3-way turn solve with the river by hand buckets: the turn's decisions are faced by three
        players, the river batches after a turn fold by two; a turn raise commits its maker on the river below it."""
        s = MultiwaySolver(cards("Kh7d2c9s"), _ranges(), 8.0, 97.5, cfg=MWConfig(later="buckets", buckets=8))
        s.run(iters=3)
        lock = ex.MWPoolLock(s, [1, 2])
        info = lock.info()
        self.assertEqual(len(info), len(s.decisions))
        root_street = [v for d, v in ((d, info[id(d)]) for d in s.decisions if d.bd is not None)]
        self.assertTrue(root_street and all(v[0] == "turn" and v[4] == 3 for v in root_street))
        later = [(d, info[id(d)]) for d in s.decisions if d.bd is None]
        self.assertTrue(all(v[0] == "river" for _, v in later))
        self.assertEqual({v[4] for _, v in later}, {2, 3})
        # hero bets, player 1 raises: player 1 is committed on every river row below that line, not on the others
        bet = s.root.children[s.root.names.index(next(nm for nm in s.root.names if nm.startswith("bet")))]
        self.assertEqual(bet.player, 1)
        raised = [e for e, c in lock._exit_comm.items() if 1 in c]
        self.assertTrue(raised)
        mixed = [v[5] for d, v in later if d.player == 1 and isinstance(v[5], np.ndarray) and v[5].any() and not v[5].all()]
        self.assertTrue(mixed)
        # a facing decision's bet as a share of the pot before it: hero's half-pot bet
        self.assertAlmostEqual(info[id(bet)][3], 0.5, delta=0.07)

    def test_best_response_evs(self):
        """action_br_evs: with the strategies as they are it is the best response's (never under the average play's EVs);
        with the pool's leaks it differs."""
        s = MultiwaySolver(cards("Kh7d2c9s4h"), _ranges(), 8.0, 97.5)
        s.run(iters=40)
        ev = s.action_evs()
        br = s.action_br_evs(lambda node, sig, reach: sig)
        w = s.reach0[0][0] > 0
        self.assertGreaterEqual(float((br - ev)[:, w].min()), -1e-4)
        t = ex.set_profile("live")
        try:
            pool = s.action_br_evs(ex.MWPoolLock(s, [1, 2]))
        finally:
            ex.reset_profile(t)
        self.assertGreater(float(np.abs(pool - br)[:, w].max()), 0.5)
        self.assertIsNone(s._lock)   # (only for the pass)


class TestMWPoolAdvice(unittest.TestCase):
    def test_live_pool_value_bets_thin_and_stops_bluffing(self):
        """The 3-way river solve against live players, who call more multiway: KdQd's thin lead into two (top pair on
        Kh7d2c9s4h, which GTO checks) bets, and 6h5h's bluff stops paying. Before, exploit mode played every multiway
        pot at equilibrium (a note said so) and both answers were GTO's."""
        g, ga = _advise("KdQd", "Kh7d2c9s4h")
        self.assertEqual(g["recommended"], "check", g["actions"])
        r, a = _advise("KdQd", "Kh7d2c9s4h", "live")
        self.assertEqual(r["recommended"], "bet 50%", r["actions"])
        self.assertGreater(a["bet 50%"]["ev_bb"], a["check"]["ev_bb"] + 0.3, r["actions"])
        self.assertEqual(a["check"]["gto_freq"], ga["check"]["freq"])   # GTO % is GTO mode's answer
        self.assertTrue(any(n.startswith("exploit mode: CO and BTN play like typical live players") for n in r["notes"]))
        self.assertFalse(any("played at equilibrium" in n for n in r["notes"]), r["notes"])
        self.assertIn("ev_tie", r)
        _, online = _advise("6h5h", "Kh7d2c9s4c", "pool")
        r, live = _advise("6h5h", "Kh7d2c9s4c", "live")
        self.assertEqual(r["recommended"], "check", r["actions"])
        self.assertLess(live["bet 50%"]["ev_bb"], -0.2)
        self.assertGreater(online["bet 50%"]["ev_bb"], live["bet 50%"]["ev_bb"] + 0.3)

    def test_bluff_into_three_live_callers(self):
        """4-way (the N-player solve on the river): a bluff into three live players loses more than one into three online
        players, and the online pool's leads still differ from GTO's."""
        _, online = _advise("6h5h", "Kh7d2c9s4c", "pool", four=True)
        r, live = _advise("6h5h", "Kh7d2c9s4c", "live", four=True)
        self.assertTrue(r.get("solved"), r["notes"])
        self.assertEqual(r["recommended"], "check")
        self.assertLess(live["bet 50%"]["ev_bb"], online["bet 50%"]["ev_bb"] - 0.5)
        self.assertTrue(any("HJ, CO and BTN play like typical live players" in n for n in r["notes"]), r["notes"])

    def test_betting_rules_follow_the_pool(self):
        """The betting rules (no solve): against live players the opponents fold together less (exploit.MW_RULES_SHADE)
        and bluff less, so KdQd's river lead bets and bluffs lose more; the online pool folds as the rules do."""
        g, ga = _advise("KdQd", "Kh7d2c9s4h", rules=True)
        r, a = _advise("KdQd", "Kh7d2c9s4h", "live", rules=True)
        self.assertEqual(g["recommended"], "check", g["actions"])
        self.assertTrue(r["recommended"].startswith("bet"), r["actions"])
        self.assertTrue(any("typical live players in the betting rules" in n for n in r["notes"]), r["notes"])
        _, gb = _advise("6h5h", "Kh7d2c9s4c", rules=True)
        _, lb = _advise("6h5h", "Kh7d2c9s4c", "live", rules=True)
        self.assertLess(lb["bet 66%"]["ev_bb"], gb["bet 66%"]["ev_bb"] - 0.5)

    def test_rules_shade_and_bluff_keep(self):
        self.assertAlmostEqual(_keep(5.0, 10.0, 2), 1.0 - (FOLD_SHADE / 3) ** 0.5)
        t = _MW_SHADE.set(ex.LIVE["MW_RULES_SHADE"])
        try:
            self.assertGreater(_keep(5.0, 10.0, 2), 1.0 - (FOLD_SHADE / 3) ** 0.5)
            self.assertAlmostEqual(_keep(5.0, 10.0, 2, 1.0), 1.0 - (1 / 3) ** 0.5)   # an explicit shade stays
        finally:
            _MW_SHADE.reset(t)
        self.assertEqual(ex.MW_RULES_SHADE, FOLD_SHADE)   # the online pool: the rules' own


class TestSharedBudget(unittest.TestCase):
    """Exploit mode's two answers after the flop (the GTO one for the GTO % under each action, then the exploit one)
    share one answer's budget (advisor.GTO_SHARE). Each had the whole budget, so since the GTO % became GTO mode's own
    answer (3f695a58, and multiway in 97e6ecf4) an exploit answer took twice a GTO one: Quick 3-way 3.1-4.0 s against
    1.4-1.9, Fast 3-way flop 11.3 s against 5.0, a heads-up turn 10.1 against 5.0."""

    STATE = {"players": 6, "hero": "BB", "hole": "KdQd", "board": "Kh7d2c9s4h", "stack": 100, "preflop": PRE,
             "flop": CHK, "turn": CHK, "river": []}

    def _calls(self, state, gto_secs=0.0, budget=0.6):
        """The budgets each answer was asked with (_postflop stubbed): [(pool, time_budget, prior_budget, mw_budget)]."""
        adv = Advisor(Config(depth=1), time_budget=budget, prior_budget=budget / 2)
        adv.mw_budget = 2 * budget
        calls = []

        def post(self_, *a, pool=False, locks=None, **k):
            calls.append((pool, self_.time_budget, self_.prior_budget, self_.mw_budget))
            if not pool:
                time.sleep(gto_secs)
            return {"actions": [{"action": "check", "freq": 1.0, **({"gto_freq": 0.4} if pool else {})},
                                {"action": "bet 50%", "freq": 0.0, **({"gto_freq": 0.6} if pool else {})}],
                    "notes": [], "solved": True, "recommended": "check"}
        with mock.patch.object(Advisor, "_postflop", post):
            r = adv.advise(state)
        self.assertEqual((adv.time_budget, adv.prior_budget, adv.mw_budget), (budget, budget / 2, 2 * budget))
        return calls, r

    def test_the_gto_answer_gets_its_share_and_the_exploit_one_the_rest(self):
        calls, r = self._calls(dict(self.STATE, opponents="live"), gto_secs=0.3)
        (gp, gt, gpb, gm), (ep, et, epb, em) = calls
        self.assertFalse(gp)
        self.assertTrue(ep)
        self.assertAlmostEqual(gt, 0.6 * A.GTO_SHARE)
        self.assertAlmostEqual(gpb, 0.3 * A.GTO_SHARE)
        self.assertAlmostEqual(gm, 1.2 * A.GTO_SHARE)
        # the GTO answer took half the nominal budget (max of time_budget and mw_budget: 1.2 s): the rest
        self.assertAlmostEqual(et / 0.6, em / 1.2, places=6)
        self.assertGreater(et, 0.6 * (1 - A.GTO_SHARE) - 1e-9)
        self.assertLess(et, 0.6 * (1 - 0.3 / 1.2) + 1e-9)
        self.assertEqual([a["gto_freq"] for a in r["actions"]], [1.0, 0.0])   # (the GTO answer's mix)
        # a GTO answer read off a library at once: the exploit one keeps (nearly) the whole budget
        calls, _ = self._calls(dict(self.STATE, opponents="pool"))
        self.assertGreater(calls[1][1], 0.6 * 0.95)
        # GTO mode: one answer with the whole budget
        calls, _ = self._calls(dict(self.STATE))
        self.assertEqual(calls, [(False, 0.6, 0.3, 1.2)])

    def test_quick_shows_no_gto_percent(self):
        """Quick: no second answer (it is a solve too, and a quick answer has ~2 s): no GTO %, a note says so."""
        calls, r = self._calls(dict(self.STATE, opponents="live", quick=True))
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0])
        self.assertTrue(all("gto_freq" not in a for a in r["actions"]))
        self.assertTrue(any(n.startswith("GTO % isn't shown at Quick") for n in r["notes"]), r["notes"])

    def test_exploit_answer_takes_about_a_gto_answers_time(self):
        """The same spot against GTO and against live players, each solved to its budget (3-way river, Fast's sizes):
        the exploit answer takes about the GTO one's time, not twice it (it took 3.1 s against 1.3)."""
        def secs(opp):
            adv = Advisor(Config(depth=1), time_budget=0.8)
            adv.mw_max_iters = 10 ** 6
            adv.mw_max_expl = 1e9          # (a loaded machine's short solve still answers: the time is what's checked)
            t = time.time()
            r = adv.advise(dict(self.STATE, **({"opponents": opp} if opp else {})))
            self.assertTrue(r.get("solved"), r.get("notes"))
            return time.time() - t, r
        g, _ = secs(None)
        e, r = secs("live")
        self.assertLess(e, 1.35 * g + 0.25, (e, g))
        self.assertTrue(all("gto_freq" in a for a in r["actions"]), r["notes"])

    def test_a_shortened_multiway_solve_runs_on_with_more_time(self):
        """A 3-way solve that ran out of a share of the budget (exploit mode's GTO answer) runs on when GTO mode asks the
        spot with the whole budget, instead of answering from the shortened solve as a cached solve that ran out does."""
        adv = Advisor(Config(depth=1), time_budget=0.05)
        adv.mw_max_iters = 10 ** 6
        adv.mw_max_expl = 1e9
        adv.advise(dict(self.STATE))
        (sol,) = adv._mw_cache.values()
        self.assertTrue(sol.out_of_time)
        before = sol.iters
        adv.advise(dict(self.STATE))                 # (the same budget: the same solve, as before)
        self.assertEqual(sol.iters, before)
        adv.time_budget = 0.6
        adv.advise(dict(self.STATE))
        self.assertGreater(sol.iters, before)

    def test_a_kind_of_gto_solve_too_slow_for_its_share_is_not_tried_again(self):
        """A multiway GTO answer that didn't settle in its share: no GTO %, no betting rules run for it (their time is
        the answer's), and the advisor doesn't solve that kind (street, players, budget) for the GTO % again."""
        adv = Advisor(Config(depth=1), time_budget=1e9)
        adv.mw_max_iters, adv.mw_trials = 3, 200
        adv.mw_max_expl = 1e-9                       # (every solve "unsettled")
        seen = []
        rules, run = Advisor._multiway, Advisor._mw_run

        def rules_(self_, *a, **k):
            seen.append(("rules", self_._gto_pass))
            return rules(self_, *a, **k)

        def run_(self_, *a, **k):
            seen.append(("solve", self_._gto_pass))
            return run(self_, *a, **k)
        with mock.patch.object(Advisor, "_multiway", rules_), mock.patch.object(Advisor, "_mw_run", run_):
            adv.advise(dict(self.STATE, opponents="live"))
            self.assertEqual(seen, [("solve", True), ("solve", False), ("rules", False)])
            self.assertEqual([k[:2] for k in adv._gto_unsettled], [("river", 3)])
            seen.clear()
            r = adv.advise(dict(self.STATE, board="Kh7d2c9s5h", opponents="live"))   # (another river: not tried)
            self.assertEqual(seen, [("solve", False), ("rules", False)])
        self.assertFalse(Advisor._gto_unsettled)   # (the class's stays empty)
        self.assertIn("actions", r)


class TestCoreAllinTable(unittest.TestCase):
    def test_the_cores_table_is_numpys(self):
        """The best-response pass (numpy) reads the heads-up all-in table the compiled core built for the solve instead of
        building it again (0.8 s of a Fast 3-way flop's 1.1 s pass)."""
        from gto import mwcore
        if mwcore.lib() is None:
            self.skipTest("no compiled core")
        s = MultiwaySolver(cards("Kh7d2c"), _ranges(), 8.0, 20.0, cfg=MWConfig(later="buckets", buckets=6,
                                                                              bucket_river_step=12))
        s.run(iters=2)
        prog = getattr(s, "_mwc", None)
        if prog is None or not prog.rt:
            self.skipTest("the solve isn't in the core, or has no all-ins")
        core = s._allin_table()
        self.assertIs(core, s._table)
        del s._table
        s._mwc = None
        mine = s._allin_table()
        self.assertEqual(core.shape, mine.shape)
        self.assertLess(float(np.abs(core - mine).max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
