"""Advisor.instant on the turn after a library flop whose spot stores no turn files: the nearest spot that does
(advisor._near_turn_spots: the same preflop raiser in the same seat, single-raised, the pot and the ranges near) answers
the turn's first decision and in position after a check at once from its stored turn, marked approximate and naming
that spot, with no EVs (its pot's and ranges', not the hand's), until the turn's own solve has run NEAR_TURN_UNTIL
iterations; Study's whole range the same way where it stores NEAR_TURN_HELD of it; prefetched with the entry. A spot
with turn files of its own reads only its own, as before."""
import functools
import importlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from gto import advisor as A
from gto import flop_library as fl
from gto import native
from gto.cards import cards as parse_cards, cards_str, ALL_COMBOS, COMBO_INDEX
from gto.cli import parse_actions
from gto.solver import solve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
build = importlib.import_module("build_flop_library")

HAND = next(s for s in fl.SPOTS if s[0] == "srp_utg_bb")   # (the hand's spot: an entry with no turn file)
SRC = next(s for s in fl.SPOTS if s[0] == "srp_hj_bb")     # (the near spot: its entry with its turn file)
CANON = "Kc7d2h"
BOARD = "Kh7s2d"
X = lambda p: {"pos": p, "action": "check"}


def advisor(lib):
    a = native.advisor("fast", budget=0.5, prior_budget=0.5)
    a.library = lib
    return a


@functools.lru_cache(maxsize=None)
def _flop_solve(line):
    s = fl.spot_setup(line)
    return s, solve(parse_cards(CANON), s["full0"], s["full1"], s["pot"], s["stack"], cfg=fl.library_config(),
                    time_budget=1e9, max_iters=60)


class _Near(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        for (sid, _, line), turns in ((HAND, False), (SRC, True)):
            s, sol = _flop_solve(line)
            build.write_entry(cls.dir, sid, CANON, sol, dict(build._meta(sid, s), board=CANON), with_turns=turns)
        build.write_index(cls.dir, [HAND, SRC])
        cls.lib = fl.FlopLibrary.from_dir(cls.dir)
        cls.spots = {s["id"]: s for s in cls.lib.index()["spots"]}
        cls.turns = cls.lib.turns(cls.spots[SRC[0]], CANON)
        cls.setup = _flop_solve(HAND[2])[0]
        cls.smap = fl.canonical(parse_cards(BOARD))[1]
        cls.hand = {"players": 6, "stack": 100, "preflop": parse_actions(HAND[2]), "turn": [], "river": []}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def canon(self, c):
        return c // 4 * 4 + self.smap[c % 4]

    def hole_in(self, p, turn, k=3):
        """A combo of player p's in the hand's range that the near spot's turn file stores (checked-through flop)."""
        nd = fl.turn_node(self.turns, ["check", "check"], [] if p == 0 else ["check"], self.canon(parse_cards(turn)[0]))
        inv = {self.canon(c): c for c in range(52)}
        board = set(parse_cards(BOARD + turn))
        r = self.setup["full0" if p == 0 else "full1"]
        got = [cards_str(sorted((inv[a], inv[b]), reverse=True)) for a, b in (ALL_COMBOS[i] for i in nd["combos"])
               if r[COMBO_INDEX[(a, b)]] > 0.5]
        return [h for h in got if not set(parse_cards(h)) & board][k]

    def stored(self, at, hole, turn):
        nd = fl.turn_node(self.turns, ["check", "check"], at, self.canon(parse_cards(turn)[0]))
        i = COMBO_INDEX[tuple(sorted(self.canon(c) for c in parse_cards(hole)))]
        j = int(np.searchsorted(nd["combos"], i))
        self.assertEqual(nd["combos"][j], i)
        return nd["names"], nd["strat"][:, j]


class TestNearTurnSpots(_Near):
    def test_which_spots(self):
        """The UTG's open reads the HJ's (both single-raised, the opener in position, the same pot); the HJ's, which
        stores its own, reads nothing else; a spot the index doesn't hold, nothing."""
        self.assertEqual([s["id"] for s in A._near_turn_spots(self.lib, HAND[0])], [SRC[0]])
        self.assertEqual(A._near_turn_spots(self.lib, SRC[0]), [])
        self.assertEqual(A._near_turn_spots(self.lib, "nope"), [])
        with mock.patch.object(A, "NEAR_TURN_DIST", 0.1):   # (the ranges too far apart)
            self.assertEqual(A._near_turn_spots(fl.FlopLibrary.from_dir(self.dir), HAND[0]), [])
        lib = fl.FlopLibrary.from_dir(self.dir)   # (another depth: the stack-to-pot ratio out of NEAR_TURN_POT)
        next(s for s in lib.index()["spots"] if s["id"] == HAND[0])["stack"] = 37.5
        self.assertEqual(A._near_turn_spots(lib, HAND[0]), [])

    def test_the_sites_library(self):
        """The site's library: the flatted single-raised opens in position read the BTN's (srp_btn_bb stores every
        flop's turns); the opener out of position (no such spot stores turns) and the 3bet pots (their stored turns
        give up more than the turn's live solve does at 50 iterations: NEAR_TURN_RAISES) read nothing else."""
        lib = fl.default_library()
        if lib is None or "srp_btn_bb" not in {s["id"] for s in lib.index()["spots"]}:
            self.skipTest("no library")
        ids = lambda sid: [s["id"] for s in A._near_turn_spots(lib, sid)]
        for sid in ("srp_co_bb", "srp_hj_bb", "srp_utg_bb", "srp_btn_sb", "srp_co_sb"):
            self.assertEqual(ids(sid)[:1], ["srp_btn_bb"], sid)
        for sid in ("srp_btn_bb", "srp_hj_co", "srp_sb_bb", "3bp_bb_btn", "3bp_btn_co"):
            self.assertEqual(ids(sid), [], sid)


class TestInstantNearTurn(_Near):
    def test_first_to_act_and_after_a_check(self):
        """The big blind first on the turn after the flop checks through, and the opener after its check: the near
        spot's stored mix of the hand's combo, nothing solved, marked approximate, naming that spot, no EVs."""
        oop, ip = self.setup["oop"], self.setup["ip"]
        for p, hero, acts, at in ((0, oop, [], []), (1, ip, [X(oop)], ["check"])):
            hole = self.hole_in(p, "9c")
            with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
                res = advisor(self.lib).instant(dict(self.hand, hero=hero, hole=hole, flop=[X(oop), X(ip)], turn=acts,
                                                     board=BOARD + "9c"))
            names, mix = self.stored(at, hole, "9c")
            self.assertEqual(res["approximate"]["source"], "store")
            self.assertIn("HJ vs BB single-raised pot", res["approximate"]["label"])
            self.assertIn("the nearest spot that stores its turns", res["approximate"]["label"])
            self.assertEqual([a["action"] for a in res["actions"]], names)
            self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0005))
            self.assertNotIn("range_ev_bb", res)
            self.assertFalse(any("range_ev_bb" in a for a in res["actions"]))
            self.assertTrue(any("this turn's own solve replaces it" in n for n in res["notes"]))

    def test_until_the_turns_own_solve_has_run_a_while(self):
        """The turn's own solve in hand: the near spot answers until that has run NEAR_TURN_UNTIL iterations (not
        STORE_UNTIL: from the 50th the solve answers as well), then the solve does."""
        oop, ip = self.setup["oop"], self.setup["ip"]
        st = dict(self.hand, hero=oop, hole=self.hole_in(0, "9c"), flop=[X(oop), X(ip)], board=BOARD + "9c")
        adv = advisor(self.lib)
        adv.hu_settled = None
        adv.advise(st)
        own = next(s for k, s in adv._cache.items() if len(k[2]) == 4)
        own.settled_at = None
        adv.time_budget = 1e3
        n = own.solver.iters
        self.assertGreaterEqual(n, A.HU_CHECKS[0])
        with mock.patch.object(A, "NEAR_TURN_UNTIL", n + 1), mock.patch.object(A, "STORE_UNTIL", n + 1):
            self.assertEqual(adv.instant(st)["approximate"]["source"], "store")
        with mock.patch.object(A, "NEAR_TURN_UNTIL", n), mock.patch.object(A, "STORE_UNTIL", n + 1):
            self.assertEqual(adv.instant(st)["approximate"]["source"], "solve")

    def test_not_without_a_near_spot(self):
        """With the near spot's entry gone (no turns anywhere), or the ranges too far apart: no answer at once."""
        oop, ip = self.setup["oop"], self.setup["ip"]
        st = dict(self.hand, hero=oop, hole=self.hole_in(0, "9c"), flop=[X(oop), X(ip)], board=BOARD + "9c")
        self.assertIsNotNone(advisor(self.lib).instant(st))
        with mock.patch.object(A, "NEAR_TURN_DIST", 0.1):
            self.assertIsNone(advisor(fl.FlopLibrary.from_dir(self.dir)).instant(st))
        d = tempfile.mkdtemp()
        try:
            s, sol = _flop_solve(HAND[2])
            build.write_entry(d, HAND[0], CANON, sol, dict(build._meta(HAND[0], s), board=CANON), with_turns=False)
            build.write_index(d, [HAND])
            self.assertIsNone(advisor(fl.FlopLibrary.from_dir(d)).instant(st))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_study(self):
        """Study at the turn's first decision: the actor's whole range off the near spot's stored turn (each combo's
        mix the file's), marked approximate and naming it, no EVs; not where the file stores under NEAR_TURN_HELD of
        the range."""
        oop, ip = self.setup["oop"], self.setup["ip"]
        st = dict(self.hand, study=True, flop=[X(oop), X(ip)], board=BOARD + "9c")
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            res = advisor(self.lib).instant(st)
        self.assertEqual(res["study"], "node")
        self.assertEqual(res["approximate"]["source"], "store")
        self.assertIn("HJ vs BB single-raised pot", res["approximate"]["label"])
        self.assertTrue(all(a["ev_bb"] is None for a in res["actions"]))
        self.assertIsNone(res["range_ev_bb"])
        with mock.patch.object(A, "NEAR_TURN_HELD", 1.01):
            res = advisor(self.lib).instant(st)
            self.assertTrue(res is None or res["approximate"]["source"] != "store")

    def test_prefetched_with_the_entry(self):
        """Once the flop is known, the prefetch's dry run asks for the hand's entry and the near spot's turn file (not
        the near spot's entry), and keeps nothing as missing."""
        asked = []

        def fetch(p, real=self.lib._fetch):
            if p.startswith("index"):
                return real(p)
            asked.append(p)
            return None
        rec = fl.FlopLibrary(fetch)
        advisor(rec).prefetch(dict(self.hand, hero=self.setup["oop"], hole="AhAs", flop=[], board=BOARD))
        h, hs = self.spots[HAND[0]]["flops"][CANON], self.spots[SRC[0]]["flops"][CANON]
        self.assertIn(f"{HAND[0]}/{CANON}.bin?v={h}", asked)
        self.assertIn(f"{SRC[0]}/{CANON}.turn.bin?v={hs}", asked)
        self.assertNotIn(f"{SRC[0]}/{CANON}.bin?v={hs}", asked)
        self.assertFalse([k for k, e in rec._entries.items() if e is None])


if __name__ == "__main__":
    unittest.main()
