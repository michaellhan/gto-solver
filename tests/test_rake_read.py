"""A raked table's flop of a library spot: an answer at once off the spot's unraked entry (advisor.instant_from_unraked),
marked approximate with what that read gives up (flop_library.RAKE_READ), while the flop's own solve with the rake
runs and replaces it. With rake on the advisor re-prices the preflop calls by the rake they go on to pay, so the
callers' ranges are narrower than the library spot's and its fingerprints don't find it: the spot is found by the
ranges the same preflop line brings without rake, and read for the raked ranges' combos (FlopLibrary.solution's
`unraked`). Nothing changes without rake. On the shipped library."""
import unittest

import numpy as np

from gto import advisor as A
from gto import flop_library as fl
from gto import native
from gto.cards import cards as parse_cards
from gto.cli import parse_actions
from gto.solver import MEMBER_FULL

LINES = dict((s[0], s[2]) for s in fl.SPOTS)
BOARD = "Kh7s2d"            # (Kc7d2h canonically: stored by every spot)
RAKE = {"rake": 0.1, "rake_cap": 3}
X = lambda p: {"pos": p, "action": "check"}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}


def advisor(speed="fast"):
    return native.advisor(speed, budget=0.5, prior_budget=0.5)


class TestRakeRead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = fl.default_library()
        ids = {s["id"] for s in (cls.lib.index().get("spots", []) if cls.lib else [])}
        if not {"srp_btn_bb", "3bp_bb_btn"} <= ids:
            raise unittest.SkipTest("the shipped library doesn't hold the spots")

    def hand(self, sid="srp_btn_bb", **kw):
        return dict({"players": 6, "stack": 100, "preflop": parse_actions(LINES[sid]), "board": BOARD}, **kw)

    def test_solution_reads_the_unraked_entry(self):
        """FlopLibrary.solution with a raked tree: None (another tree), unless `unraked`, which reads the entry with
        the same mixes and marks the solution; without rake `unraked` changes nothing."""
        s = fl.spot_setup(LINES["srp_btn_bb"])
        board = parse_cards(BOARD)
        args = (s["full0"], s["full1"], board, s["full0"], s["full1"], s["pot"], s["stack"])
        plain = fl.library_config(s["pot"], s["stack"])
        raked = fl.library_config(s["pot"], s["stack"])
        raked.rake_rate, raked.rake_cap = 0.1, 3.0
        base = self.lib.solution(*args, plain)
        self.assertIsNone(self.lib.solution(*args, raked))
        got = self.lib.solution(*args, raked, unraked=True)
        self.assertEqual(got.unraked, (0.1, 3.0))
        self.assertIsNone(base.unraked)
        np.testing.assert_array_equal(got.root.avg_strategy0(), base.root.avg_strategy0())
        self.assertIn("10% rake capped at 3bb", got.unraked_text())
        same = self.lib.solution(*args, plain, unraked=True)
        self.assertIsNone(same.unraked)
        np.testing.assert_array_equal(same.root.avg_strategy0(), base.root.avg_strategy0())
        raked.rake_cap = None
        self.assertEqual(self.lib.solution(*args, raked, unraked=True).unraked, (0.1, None))

    def test_instant_on_a_raked_flop(self):
        """With rake, a library flop answers at once off the entry, at either speed: your hand's stored mix (as the
        library's answer without rake), first to act, facing a bet and in position after a check, labelled with the
        rake and what the read gives up; no EVs. The full answer still solves the flop live with the rake."""
        for speed in ("fast", "accurate"):
            adv = advisor(speed)
            for st in (self.hand(hero="BB", hole="Qd7d"),
                       self.hand(hero="BB", hole="Qd7d", flop=[X("BB"), B("BTN", 1.82)]),
                       self.hand(hero="BTN", hole="AhQh", flop=[X("BB")]),
                       self.hand("3bp_bb_btn", hero="BB", hole="AhQh")):
                plain = adv.advise(st)
                self.assertIn("precomputed", plain)
                self.assertIsNone(adv.instant(st))   # (without rake the library answers in full, at once)
                r = adv.instant(dict(st, **RAKE))
                self.assertIsNotNone(r, st)
                self.assertEqual(r["approximate"]["source"], "unraked")
                self.assertIn("10% rake capped at 3bb", r["approximate"]["label"])
                self.assertIn("without rake", r["approximate"]["label"])
                self.assertTrue(r["notes"][0].startswith("approximate, at once:"), r["notes"])
                kind = lambda res: [a["action"].split(" to ")[0] for a in res["actions"]]   # (raises: the tree's)
                self.assertEqual(kind(r), kind(plain))
                self.assertLess(max(abs(a["freq"] - b["freq"]) for a, b in zip(r["actions"], plain["actions"])), 0.0011)
                self.assertTrue(all("ev_bb" not in a for a in r["actions"]))
        r = advisor().advise(dict(self.hand(hero="BB", hole="Qd7d"), **RAKE))
        self.assertNotIn("precomputed", r)
        self.assertNotIn("approximate", r)

    def test_not_read(self):
        """No read: rake_library off; a hand the spot's range doesn't hold; ranges of your own (no instant answer);
        Study (its own reads); once the flop's own raked solve has run RAKE_UNTIL iterations (its answer, then)."""
        st = dict(self.hand(hero="BB", hole="Qd7d"), **RAKE)
        adv = advisor()
        adv.rake_library = False
        self.assertIsNone(adv.instant(st))
        self.assertIsNone(advisor().instant(dict(st, hole="3c2h")))   # (the big blind doesn't defend 32o)
        r = advisor().instant(dict(self.hand(study=True), **RAKE))
        self.assertTrue(r is None or r.get("approximate", {}).get("source") != "unraked")
        adv = advisor()
        full = adv.advise(st)
        self.assertGreaterEqual(full["solver"]["iterations"], A.RAKE_UNTIL)
        r = adv.instant(st)
        self.assertTrue(r is None or r["approximate"]["source"] != "unraked", r and r["approximate"])

    def test_the_label(self):
        """The label gives the measured numbers of the nearest rake measured (RAKE_READ); a 9-handed table's flop, read
        off the 6-max spot (the library's borrow), says so too."""
        self.assertEqual(fl.rake_read(0.1, 1.0)[0], (0.1, 1.0))
        self.assertEqual(fl.rake_read(0.1, 2.5)[0], (0.1, 3.0))     # ($5 at $1/$2)
        self.assertEqual(fl.rake_read(0.1, 0.5)[0], (0.1, 1.0))     # ($5 at $5/$10)
        self.assertEqual(fl.rake_read(0.1, None)[0], (0.1, 3.0))
        self.assertEqual(fl.rake_read(0.05, 4.0)[0], (0.05, 3.0))
        nine = [{"pos": p, "action": "fold"} for p in ("UTG", "UTG1", "UTG2", "LJ", "HJ", "CO")] + \
            parse_actions("BTN:r2.5,SB:f,BB:c")
        r = advisor().instant(dict(self.hand(hero="BB", hole="Qd7d", players=9, preflop=nine), rake=0.1, rake_cap=1))
        self.assertEqual(r["approximate"]["source"], "unraked")
        lab = r["approximate"]["label"]
        m = fl.RAKE_READ[(0.1, 1.0)]
        for s in ("10% rake capped at 1bb", "60 flops at 10% capped at 1bb", f"gives up {m['loss']:.2f}% of the pot",
                  "no EVs", "that spot read as this one"):
            self.assertIn(s, lab)

    def test_a_tournament_has_no_rake(self):
        """A tournament is played without rake whatever the request's rake: with payouts or bounties, or "tournament"
        (an imported tournament hand's, hh.js). Without payouts (chip EV) the library answers in full, as with no
        rake; with them no raked read either."""
        st = self.hand(hero="BB", hole="Qd7d")
        adv = advisor()
        plain = adv.advise(st)
        t = dict(st, tournament={"id": "3141592653", "name": "$22 Hot", "level": 7}, **RAKE)
        self.assertIsNone(adv.instant(t))
        got = adv.advise(t)
        self.assertIn("precomputed", got)
        self.assertEqual([(a["action"], a["freq"], a.get("ev_bb")) for a in got["actions"]],
                         [(a["action"], a["freq"], a.get("ev_bb")) for a in plain["actions"]])
        for k in ({"payouts": [50, 30, 20]}, {"bounties": {"fmt": "ko", "all": 10}}):
            r = advisor().instant(dict(st, **k, **RAKE))
            self.assertTrue(r is None or r["approximate"]["source"] != "unraked", (k, r and r["approximate"]))

    def test_the_raked_ranges(self):
        """The read is for the raked table's ranges: the big blind's defence narrower than the spot's (re-priced by the
        rake), the button's open the same, each of the raked combos in the entry's."""
        adv = advisor()
        seen = {}
        orig = fl.FlopLibrary.solution

        def spy(lib, full0, full1, board, r0, r1, *a, **k):
            if k.get("unraked"):
                seen["r"] = (full0, full1, r0, r1)
            return orig(lib, full0, full1, board, r0, r1, *a, **k)
        fl.FlopLibrary.solution = spy
        try:
            self.assertEqual(adv.instant(dict(self.hand(hero="BB", hole="Qd7d"), **RAKE))["approximate"]["source"],
                             "unraked")
        finally:
            fl.FlopLibrary.solution = orig
        full0, full1, r0, r1 = seen["r"]
        s = fl.spot_setup(LINES["srp_btn_bb"])
        self.assertEqual([fl.range_fp(full0), fl.range_fp(full1)], [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])])
        bm = np.prod([1.0 - MEMBER_FULL[c] for c in parse_cards(BOARD)], axis=0)
        self.assertLess(r0.sum(), 0.6 * (s["full0"] * bm).sum())       # (218 combos into the flop, against 576)
        self.assertGreater(r1.sum(), 0.85 * (s["full1"] * bm).sum())   # (less your cards)
        self.assertTrue(np.all(s["full0"][r0 > 0] > 0) and np.all(s["full1"][r1 > 0] > 0))


if __name__ == "__main__":
    unittest.main()
