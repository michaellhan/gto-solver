"""What the browser's worker does ahead of a request (gto/static/worker.js): Advisor.prefetch names the library files
a flop's answers read without reading them (the worker fetches them ahead), and advisor.warm_step builds the
preflop equity tables a slice at a time while the worker is idle."""
import os
import unittest

import numpy as np

from gto import advisor as A
from gto import native
from gto.flop_library import DIR, FlopLibrary
from gto.mw_library import INDEX as MW_INDEX, MWLibrary

F = lambda p: {"pos": p, "action": "fold"}
C = lambda p: {"pos": p, "action": "call"}
X = lambda p: {"pos": p, "action": "check"}
R = lambda p, to: {"pos": p, "action": "raise", "to": to}
SRP = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), C("BB")]
MW = [F("UTG"), F("HJ"), R("CO", 2.5), C("BTN"), F("SB"), C("BB")]


class Fetch:
    """The worker's libFetch over the shipped files: while `record` is a list (a prefetch's dry run) an entry's path
    is noted there and not read; the indexes are always read."""

    def __init__(self):
        self.record, self.read = None, []

    def __call__(self, p):
        if self.record is not None and not p.endswith("index.json"):
            self.record.append(p)
            return None
        self.read.append(p)
        try:
            with open(os.path.join(DIR, p.split("?")[0]), "rb") as f:
                return f.read()
        except OSError:
            return None


def state(**kw):
    return {"players": 6, "stack": 100, "preflop": SRP, "flop": [], "turn": [], "river": [], **kw}


@unittest.skipUnless(os.path.exists(os.path.join(DIR, "index.json")) and os.path.exists(os.path.join(DIR, MW_INDEX)),
                     "no library built")
class TestPrefetch(unittest.TestCase):
    def advisor(self):
        f = Fetch()
        adv = native.advisor("fast", budget=0.3, prior_budget=0.2)
        adv.library = FlopLibrary(f)
        adv.mw_library = MWLibrary(f)
        return adv, f

    def dry(self, adv, f, st):
        f.record = []
        try:
            adv.prefetch(st)
            return f.record
        finally:
            f.record = None

    def test_names_the_flops_entry_and_reads_it_later(self):
        """A heads-up flop the library holds: the dry run names its entry and its turn file (the hero not first to
        act: asked as the player who is), reads none, and leaves nothing marked missing; the answer then reads the
        entry and is precomputed."""
        adv, f = self.advisor()
        st = state(hero="BTN", hole="AsQs", board="Kc7d4h")
        names = self.dry(adv, f, st)
        self.assertEqual([p.split("?")[0] for p in names], ["srp_btn_bb/Kc7d4h.bin", "srp_btn_bb/Kc7d4h.turn.bin"])
        self.assertEqual([p for p in f.read if not p.endswith("index.json")], [])
        self.assertNotIn(None, adv.library._entries.values())
        res = adv.advise(dict(st, flop=[X("BB")]))
        self.assertTrue(res.get("precomputed"), res.get("notes"))
        self.assertIn(names[0], f.read)

    def test_three_way_flop(self):
        """A 3-way flop's entry is named too (the 3-way library's)."""
        adv, f = self.advisor()
        names = self.dry(adv, f, state(preflop=MW, hero="BB", hole="Qh9h", board="Kh7d2c"))
        self.assertTrue(names and all(p.startswith("mw_co_btn_bb/") for p in names), names)
        self.assertNotIn(None, adv.mw_library._entries.values())

    def test_nothing_before_the_flop_or_off_the_library(self):
        """Preflop there is nothing to name; nor in a pot no library holds (here a limped one); the hand's turn and
        river don't matter (the flop's entries and turn files are named)."""
        adv, f = self.advisor()
        self.assertEqual(self.dry(adv, f, state(hero="BTN", hole="AsQs", board="")), [])
        limp = [F("UTG"), F("HJ"), F("CO"), C("BTN"), F("SB"), X("BB")]
        self.assertEqual(self.dry(adv, f, state(preflop=limp, hero="BTN", hole="AsQs", board="Kc7d4h")), [])
        later = self.dry(adv, f, state(hero="BTN", hole="AsQs", board="Kc7d4h9s", flop=[X("BB"), X("BTN")]))
        self.assertEqual([p.split("?")[0] for p in later], ["srp_btn_bb/Kc7d4h.bin", "srp_btn_bb/Kc7d4h.turn.bin"])


class TestWarmStep(unittest.TestCase):
    def test_slices_build_the_same_tables(self):
        """warm_step's slices build the same boards and lookups as building them at once, then report done."""
        def reset():
            A._TABLES.update(rng=None, boards=[], index=[])
            for fn in (A._preflop_boards, A._board_index, A._board_index_all):
                fn.cache_clear()
        try:
            reset()
            whole = (A._preflop_boards().copy(), [x[0].copy() for x in A._board_index()], A._board_index_all()[2][3].copy())
            reset()
            steps = 0
            while not A.warm_step(0.01):
                steps += 1
                self.assertLess(steps, 2000)
            self.assertGreater(steps, 3)
            self.assertTrue(A.warm_step())
            np.testing.assert_array_equal(A._preflop_boards(), whole[0])
            self.assertEqual(len(A._board_index()), len(whole[1]))
            for a, b in zip(A._board_index(), whole[1]):
                np.testing.assert_array_equal(a[0], b)
            np.testing.assert_array_equal(A._board_index_all()[2][3], whole[2])
        finally:
            reset()


if __name__ == "__main__":
    unittest.main()
