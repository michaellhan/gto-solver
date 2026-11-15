"""Exploit mode in multiway pots: the pool's strategy at every opponent decision of the 3- and 4-way solves
(gto/exploit.py MWPoolLock) with hero best-responding, and the betting rules following the pool (Advisor._multiway)."""
import unittest
from types import SimpleNamespace as N

import numpy as np

from gto import exploit as ex
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


if __name__ == "__main__":
    unittest.main()
