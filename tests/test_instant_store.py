"""Advisor.instant on a library flop's turn: the precomputed flop solve's own turn strategies, stored beside its entry
(flop_library's turn files), answer the turn's first decision and in position after a check at once
(advisor.instant_from_store), marked approximate; the turn's own solve still answers the request. Prefetched with the
entry once the flop is known (Advisor.prefetch)."""
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

SPOT = next(s for s in fl.SPOTS if s[0] == "srp_utg_bb")   # (the narrowest single-raised ranges: quick to solve)
CANON = "Kc7d2h"
BOARD = "Kh7s2d"            # (the same flop in other suits: read through the suit map)
C = lambda p: {"pos": p, "action": "call"}
X = lambda p: {"pos": p, "action": "check"}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}


def advisor(lib):
    a = native.advisor("fast", budget=0.5, prior_budget=0.5)
    a.library = lib
    return a


class TestInstantFromStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sid, _, line = SPOT
        cls.setup = s = fl.spot_setup(line)
        sol = solve(parse_cards(CANON), s["full0"], s["full1"], s["pot"], s["stack"], cfg=fl.library_config(),
                    time_budget=1e9, max_iters=60)
        meta = dict(build._meta(sid, s), board=CANON)
        cls.dirs = []
        cls.libs = []
        for turns in (True, False):   # (with its turn file; and an entry without one)
            d = tempfile.mkdtemp()
            build.write_entry(d, sid, CANON, sol, meta, with_turns=turns)
            build.write_index(d, [SPOT])
            cls.dirs.append(d)
            cls.libs.append(fl.FlopLibrary.from_dir(d))
        cls.lib = cls.libs[0]
        cls.spot = cls.lib.index()["spots"][0]
        cls.turns = cls.lib.turns(cls.spot, CANON)
        cls.smap = fl.canonical(parse_cards(BOARD))[1]
        cls.hand = {"players": 6, "stack": 100, "preflop": parse_actions(line), "turn": [], "river": []}

    @classmethod
    def tearDownClass(cls):
        for d in cls.dirs:
            shutil.rmtree(d, ignore_errors=True)

    def canon(self, c):
        return c // 4 * 4 + self.smap[c % 4]

    def stored(self, path, at, hole, turn):
        """The stored mix of `hole` (on BOARD) at a turn decision, read straight off the turn file."""
        nd = fl.turn_node(self.turns, path, at, self.canon(parse_cards(turn)[0]))
        i = COMBO_INDEX[tuple(sorted(self.canon(c) for c in parse_cards(hole)))]
        j = int(np.searchsorted(nd["combos"], i))
        self.assertEqual(nd["combos"][j], i)
        return nd["names"], nd["strat"][:, j], nd["played"]

    def hole_in(self, p, turn, k=0):
        """A combo of player p's that the turn file stores on the checked-through flop (on BOARD)."""
        nd = fl.turn_node(self.turns, ["check", "check"], [] if p == 0 else ["check"], self.canon(parse_cards(turn)[0]))
        inv = {self.canon(c): c for c in range(52)}
        board = set(parse_cards(BOARD + turn))
        got = [cards_str(sorted((inv[a], inv[b]), reverse=True)) for a, b in (ALL_COMBOS[i] for i in nd["combos"])]
        got = [h for h in got if not set(parse_cards(h)) & board]
        return got[k]

    def test_first_to_act_and_after_a_check(self):
        """The big blind first on the turn after the flop checks through, and the opener after its check: the stored
        mix of the hand's combo (through the suit map), the bet at the hand's own pot, nothing solved, marked
        approximate; on the turn class's played card and on another card of its class."""
        oop, ip = self.setup["oop"], self.setup["ip"]
        flop = [X(oop), X(ip)]
        inv = {self.canon(c): c for c in range(52)}
        played = next(parse_cards(p)[0] for p, _ in self.turns["head"]["classes"] if p[0] == "9")
        other = next(c for c in parse_cards("9c9d9h9s") if c != inv[played])
        for turn in (cards_str([inv[played]]), cards_str([other])):
            for p, hero, acts, at in ((0, oop, [], []), (1, ip, [X(oop)], ["check"])):
                hole = self.hole_in(p, turn, 3)
                adv = advisor(self.lib)
                with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
                    res = adv.instant(dict(self.hand, hero=hero, hole=hole, flop=flop, turn=acts, board=BOARD + turn))
                names, mix, pl = self.stored(["check", "check"], at, hole, turn)
                self.assertEqual(pl, played)
                self.assertEqual(res["approximate"]["source"], "store")
                self.assertIn("precomputed flop solve's view of the turn", res["approximate"]["label"])
                self.assertEqual([a["action"] for a in res["actions"]], names)
                self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0005))
                bet = next(a for a in res["actions"] if a["action"] == "bet 75%")
                self.assertAlmostEqual(bet["amount_bb"], round(0.75 * res["pot_bb"], 2), places=2)
                self.assertNotIn("solver", res)
                self.assertTrue(any("this turn's own solve replaces it" in n for n in res["notes"]))

    def test_the_line_the_flop_took(self):
        """A flop bet and call: the stored turn of that line (its pot larger), not the checked-through one's."""
        oop, ip = self.setup["oop"], self.setup["ip"]
        pot = self.setup["pot"]
        path = ["check", "bet 33%", "call"]
        line = next(ln for ln in self.turns["lines"] if ln["path"] == path)
        inv = {self.canon(c): c for c in range(52)}
        turn = "9c"
        board = set(parse_cards(BOARD + turn))
        nd = fl.turn_node(self.turns, path, [], self.canon(parse_cards(turn)[0]))
        hole = next(h for h in (cards_str(sorted((inv[a], inv[b]), reverse=True)) for a, b in (ALL_COMBOS[i] for i in nd["combos"]))
                    if not set(parse_cards(h)) & board)
        res = advisor(self.lib).instant(dict(self.hand, hero=oop, hole=hole, board=BOARD + turn,
                                             flop=[X(oop), B(ip, round(0.33 * pot, 2)), C(oop)]))
        self.assertEqual(res["approximate"]["source"], "store")
        _, mix, _ = self.stored(path, [], hole, turn)
        self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0005))
        self.assertGreater(line["pot"], next(ln for ln in self.turns["lines"] if ln["path"] == ["check", "check"])["pot"])

    def test_none_where_it_isnt_stored(self):
        """Facing a turn bet (not stored), an entry without a turn file, the river: no approximate answer from it."""
        oop, ip = self.setup["oop"], self.setup["ip"]
        flop = [X(oop), X(ip)]
        hole = self.hole_in(1, "9c", 3)
        facing = dict(self.hand, hero=ip, hole=hole, flop=flop, turn=[B(oop, 3)], board=BOARD + "9c")
        self.assertIsNone(advisor(self.lib).instant(facing))
        first = dict(self.hand, hero=ip, hole=hole, flop=flop, turn=[X(oop)], board=BOARD + "9c")
        self.assertIsNotNone(advisor(self.lib).instant(first))
        self.assertIsNone(advisor(self.libs[1]).instant(first))
        river = dict(first, turn=[X(oop), X(ip)], river=[X(oop)], board=BOARD + "9c3s")
        res = advisor(self.lib).instant(river)
        self.assertTrue(res is None or res["approximate"]["source"] != "store")

    def test_the_turns_own_solve_answers_the_request(self):
        """The request itself still solves the turn: its answer is the live solve's, not the stored one's."""
        oop, ip = self.setup["oop"], self.setup["ip"]
        hole = self.hole_in(0, "9c", 3)
        st = dict(self.hand, hero=oop, hole=hole, flop=[X(oop), X(ip)], board=BOARD + "9c")
        adv = advisor(self.lib)
        self.assertEqual(adv.instant(st)["approximate"]["source"], "store")
        res = adv.advise(st)
        self.assertNotIn("approximate", res)
        self.assertGreater(res["solver"]["iterations"], 0)
        self.assertIn("bet 33%", [a["action"] for a in res["actions"]])   # (the turn's own sizes)

    def test_before_the_turns_own_solve_until_it_has_run_a_while(self):
        """The turn's own solve in hand (a presolve's, or a request's stopped part way): the stored turn answers until
        that has run STORE_UNTIL iterations, then the solve does (instant_from_own)."""
        oop, ip = self.setup["oop"], self.setup["ip"]
        st = dict(self.hand, hero=oop, hole=self.hole_in(0, "9c", 3), flop=[X(oop), X(ip)], board=BOARD + "9c")
        adv = advisor(self.lib)
        adv.hu_settled = None
        adv.advise(st)
        own = next(s for k, s in adv._cache.items() if len(k[2]) == 4)
        own.settled_at = None
        adv.time_budget = 1e3   # (short of its budget: not an answer that's quick anyway)
        n = own.solver.iters
        self.assertGreaterEqual(n, A.HU_CHECKS[0])
        with mock.patch.object(A, "STORE_UNTIL", n + 1):
            self.assertEqual(adv.instant(st)["approximate"]["source"], "store")
        with mock.patch.object(A, "STORE_UNTIL", n):
            self.assertEqual(adv.instant(st)["approximate"]["source"], "solve")

    def test_prefetched_with_the_entry(self):
        """Once the flop is known, the prefetch's dry run asks for the flop's entry and its turn file where the index
        lists one (the browser's worker fetches what it asks for ahead), and keeps neither as missing."""
        for lib, want in ((self.lib, True), (self.libs[1], False)):
            asked = []

            def fetch(p, real=lib._fetch):
                if p.startswith("index"):
                    return real(p)
                asked.append(p)
                return None   # (as the worker's dry run: recorded, not read)
            rec = fl.FlopLibrary(fetch)
            adv = advisor(rec)
            adv.prefetch(dict(self.hand, hero=self.setup["oop"], hole="AhAs", flop=[], board=BOARD))
            h = self.spot["flops"][CANON]
            self.assertIn(f"{SPOT[0]}/{CANON}.bin?v={h}", asked)
            self.assertEqual(f"{SPOT[0]}/{CANON}.turn.bin?v={h}" in asked, want, asked)
            self.assertFalse([k for k, e in rec._entries.items() if e is None])


if __name__ == "__main__":
    unittest.main()
