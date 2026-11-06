"""Advisor.instant in 3-way pots (the advice's early panel, your hand alone): the reads Study shows the whole range
at once, for your combo. A 3-way flop the 3-way library doesn't store, read off its stored flops most like it at the
decisions it stores (Advisor._study_mw_near); a later decision on a street whose earlier one your advice solved, off
that solve; the turn's first decision off the flop's live solve (Advisor._study_mw_prev). The mix alone (no EVs), marked
approximate, until the node's own solve replaces it; none on a flop the library stores (its answer is at once anyway),
at the turn's later decisions before a turn solve, or on a river. At 8- and 9-max a 3-way pot a library spot's ranges
are a few combos from is read as that spot's (MWLibrary.borrow), marked borrowed: its stored flops' answers in full,
the others' read off them. A pot neither held nor borrowed (antes; 6-max's other opens and flats) is read off the
nearest spot at once only (MWLibrary.stand_in), marked approximate."""
import gc
import unittest

from gto import advisor as A
from gto.native import advisor

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}]
PRE8 = [{"pos": "UTG1", "action": "fold"}, {"pos": "LJ", "action": "fold"}]
PRE9 = PRE8 + [{"pos": "UTG2", "action": "fold"}]
X = lambda p: {"pos": p, "action": "check"}
ST = {"players": 6, "stack": 100, "preflop": PRE, "board": "Kd8c3h", "flop": [], "turn": [], "river": []}


def held(r, k=0):
    """A combo Study's read holds for its actor (in range), the k-th."""
    return [x[0] for rows in r["combos"].values() for x in rows if x[1] > 0.4][k]


class TestInstantMultiway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = advisor("fast")
        cls.adv.mw_budget = 2.5   # (the flop's live 3-way solves: Fast's 5 s natively)

    @classmethod
    def tearDownClass(cls):
        del cls.adv
        gc.collect()

    def early(self, r, street, source):
        self.assertEqual((r["street"], r["approximate"]["source"]), (street, source))
        self.assertTrue(r["multiway"])
        self.assertAlmostEqual(sum(a["freq"] for a in r["actions"]), 1.0, delta=0.01)
        self.assertEqual(r["recommended"], max(r["actions"], key=lambda a: a["freq"])["action"])
        self.assertFalse(any("ev_bb" in a for a in r["actions"]))
        self.assertIn(f"this {street}'s own solve replaces it", r["notes"][0])

    def test_flop_near(self):
        """First to act and checked to once and twice on a flop the 3-way library doesn't store: each combo's mix
        Study's read gives it (the same stored flops), nothing solved; none for a combo off the range, on a flop the
        library stores, or after a bet."""
        adv = advisor("fast")
        for hero, flop in (("SB", []), ("BB", [X("SB")]), ("BTN", [X("SB"), X("BB")])):
            s = adv.instant(dict(ST, study=True, flop=flop))
            rows = {x[0]: x for rows in s["combos"].values() for x in rows}
            for k in (0, 7):
                hole = held(s, k)
                r = adv.instant(dict(ST, hero=hero, hole=hole, flop=flop))
                self.early(r, "flop", "near")
                self.assertEqual(r["approximate"], s["approximate"])
                self.assertEqual((r["hero"], r["position"]), (hero, s["position"]))
                self.assertEqual([a["action"] for a in r["actions"]], s["grid"]["actions"])
                for a, f in zip(r["actions"], rows[hole][2]):
                    self.assertAlmostEqual(a["freq"], f, delta=0.002)
        self.assertEqual(adv._mw_lines, {})   # (nothing solved)
        off = next(x[0] for rows in s["combos"].values() for x in rows if x[1] == 0) if any(
            x[1] == 0 for rows in s["combos"].values() for x in rows) else None
        if off is not None:
            self.assertIsNone(adv.instant(dict(ST, hero="BTN", hole=off, flop=[X("SB"), X("BB")])))
        self.assertIsNone(adv.instant(dict(ST, hero="SB", hole="KcQc", board="Kd7c2h")))
        self.assertIsNone(adv.instant(dict(ST, hero="BB", hole="KcQc", flop=[{"pos": "SB", "action": "bet", "to": 2.5}])))

    def test_later_and_the_turn(self):
        """After your advice on the flop (its live 3-way solve): facing a bet after your check, that solve's play of
        the line since; the turn's first decision, the flop solve's view of the turn (hand buckets, one bet size); at
        8-max the same (no library there: the flop solved live). None at the turn's later decisions before a turn
        solve, or on a river."""
        for st in (ST, dict(ST, players=8, preflop=PRE8 + PRE)):
            adv = self.adv
            hand = dict(st, hero="SB", hole="Td9d", board="Kd8c3h")
            root = adv.advise(hand)
            self.assertTrue(root.get("solved"), root.get("notes"))
            self.assertIsNone(adv.instant(hand))   # (its own solve in hand: quick anyway)
            pot = root["pot_bb"]
            facing = dict(hand, flop=[X("SB"), X("BB"), {"pos": "BTN", "action": "bet", "amount": round(pot / 3, 2)}])
            r = adv.instant(facing)
            self.early(r, "flop", "solve")
            self.assertRegex(r["approximate"]["label"], r"^this street's 3-way solve from SB's decision \(\d+ iterations")
            self.assertEqual([a["action"] for a in r["actions"]][:2], ["fold", "call"])
            turn = dict(hand, board="Kd8c3h2s", flop=[X("SB"), X("BB"), X("BTN")])
            r = adv.instant(turn)
            self.early(r, "turn", "prev")
            self.assertEqual([a["action"] for a in r["actions"]], ["check", "bet 50%"])
            self.assertRegex(r["approximate"]["label"], r"flop solve's view of the turn \(by card class and \d+ hand buckets")
            self.assertEqual((r["hero"], r["villain"]), ("SB", "BB and BTN"))
            self.assertIsNone(adv.instant(dict(turn, hero="BB", hole="AhKh", turn=[X("SB")])))
            self.assertIsNone(adv.instant(dict(turn, board="Kd8c3h2s7s", turn=[X("SB"), X("BB"), X("BTN")])))

    def test_instant_gate(self):
        """Exploit mode, locks and the other requests beyond the equilibrium stay without an answer at once."""
        for extra in ({"opponents": "pool"}, {"locks": {"BB": {"fold": 10}}}, {"jam": True}, {"sizes": {"flop": [50]}}):
            self.assertIsNone(advisor("fast").instant(dict(ST, hero="SB", hole="KcQc", **extra)), extra)


class TestBorrowMultiway(unittest.TestCase):
    """A 3-way pot of another table size read as a library spot's (MWLibrary.borrow): the same pot and stack, the
    three ranges within MW_BORROW_DIST and MW_BORROW_COVER of the spot's."""

    @classmethod
    def setUpClass(cls):
        from gto import mw_library as ML
        cls.ML, cls.lib = ML, ML.default_library()

    def setup(self, line, players):
        return self.ML.spot_setup(line, players=players)

    def test_borrow(self):
        """8- and 9-max BTN open, SB and BB call: the 6-max spot's (distance ~0.14); not at 6-max, without a table
        size (ranges of your own), where the ranges are too far apart (8-max CO open, SB and BB call), or in another
        pot (antes)."""
        from gto.cards import cards
        lib, ML = self.lib, self.ML
        for players, pre in ((8, "UTG:f,UTG1:f,LJ:f,"), (9, "UTG:f,UTG1:f,UTG2:f,LJ:f,")):
            s = self.setup(pre + "HJ:f,CO:f,BTN:r2.5,SB:c,BB:c", players)
            b = lib.borrow(s["ranges"], s["pot"], s["stack"], players)
            self.assertEqual(b["spot"]["id"], "mw_btn_sb_bb")
            self.assertLess(sum(b["dist"]), 0.3)
            self.assertGreaterEqual(min(b["cover"]), ML.MW_BORROW_COVER)
            sol = lib.solution(s["ranges"], cards("Kc7d2h"), s["pot"], s["stack"], players=players)
            self.assertEqual(sol.info()["borrowed"]["spot"], "mw_btn_sb_bb")
            self.assertRegex(sol.info()["borrowed"]["text"], r"^this table's ranges differ from that spot's by [\d.]+%")
            self.assertEqual(sol.label, "6-max BTN open, SB and BB call")
            self.assertIsNone(lib.solution(s["ranges"], cards("Kc7d2h"), s["pot"], s["stack"]))
            near = lib.near(s["ranges"], cards("Kd8c3h"), s["pot"], s["stack"], players=players)
            self.assertEqual(near.borrowed["spot"], "mw_btn_sb_bb")
        own = self.setup("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:c", 6)
        self.assertIsNone(lib.borrow(own["ranges"], own["pot"], own["stack"], 8))   # (the library holds it)
        sol = lib.solution(own["ranges"], cards("Kc7d2h"), own["pot"], own["stack"], players=6)
        self.assertIsNone(sol.borrowed)
        far = self.setup("UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:c", 8)
        self.assertIsNone(lib.borrow(far["ranges"], far["pot"], far["stack"], 8))
        six = self.setup("UTG:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:c", 6)
        self.assertIsNone(lib.borrow(six["ranges"], six["pot"], six["stack"], 6))   # (6-max: its own pot, solved)
        ante = ML.spot_setup("UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:c", players=8, ante=0.125)
        self.assertEqual(ante["pot"], 8.5)   # (another pot: no spot has it)
        self.assertIsNone(lib.borrow(ante["ranges"], ante["pot"], ante["stack"], 8))

    def test_advice(self):
        """The advice at 8-max: on a stored flop the spot's precomputed answer, EVs and all, marked borrowed (the page
        shows "precomputed · borrowed"); on one it doesn't store the answer at once off its stored flops, naming the
        spot, each combo's mix the one Study shows."""
        adv = advisor("fast")
        st = dict(ST, players=8, preflop=PRE8 + PRE, board="Kc7d2h")
        r = adv.advise(dict(st, hero="SB", hole="KhJh"))
        self.assertEqual(r["precomputed"]["borrowed"]["spot"], "mw_btn_sb_bb")
        self.assertTrue(r["solved"] and all("ev_bb" in a for a in r["actions"]))
        self.assertTrue(any("approximate: that spot read as this one" in n for n in r["notes"]))
        off = adv.advise(dict(st, hero="SB", hole="KdQd"))   # (not in that spot's SB range: its stored EVs would be random play)
        self.assertNotIn("precomputed", off)
        self.assertTrue(any("isn't in your range" in n for n in off["notes"]))
        st = dict(st, board="Kd8c3h")
        s = adv.instant(dict(st, study=True))
        self.assertRegex(s["approximate"]["label"], r"in the library's 6-max BTN open, SB and BB call, that spot read "
                                                    r"as this one \(this table's ranges differ")
        rows = {x[0]: x for rows in s["combos"].values() for x in rows}
        hole = held(s, 3)
        r = adv.instant(dict(st, hero="SB", hole=hole))
        self.assertEqual(r["approximate"], s["approximate"])
        for a, f in zip(r["actions"], rows[hole][2]):
            self.assertAlmostEqual(a["freq"], f, delta=0.002)

    def test_prefetch(self):
        """The worker's dry run at the flop names the borrowed spot's entry (and reads nothing)."""
        from gto import native
        from gto.flop_library import FlopLibrary
        from gto.mw_library import MWLibrary
        from tests.test_prefetch import Fetch
        f = Fetch()
        adv = native.advisor("fast", budget=0.3, prior_budget=0.2)
        adv.library, adv.mw_library = FlopLibrary(f), MWLibrary(f)
        f.record = []
        adv.prefetch(dict(ST, players=8, preflop=PRE8 + PRE, board="Kc7d2h", hero="BB", hole="Qh9h"))
        names, f.record = f.record, None
        self.assertEqual([p.split("?")[0] for p in names], ["mw_btn_sb_bb/Kc7d2h.bin"])
        self.assertEqual([p for p in f.read if not p.endswith("index.json")], [])


PRE_HJ = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "raise", "to": 2.5},
          {"pos": "CO", "action": "fold"}, {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"},
          {"pos": "BB", "action": "call"}]
PRE_UH = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "call"},
          {"pos": "CO", "action": "fold"}, {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"},
          {"pos": "BB", "action": "call"}]


class TestStandInMultiway(unittest.TestCase):
    """A 3-way pot neither held nor borrowed (antes: another pot; 6-max's other opens and flats) read approximately off
    the nearest spot (MWLibrary.stand_in): the answer at once only, marked approximate, no EVs."""

    @classmethod
    def setUpClass(cls):
        from gto import mw_library as ML
        cls.ML, cls.lib = ML, ML.default_library()

    def test_stand_in(self):
        """6-max with antes, HJ open, BTN and BB call: HJ's spot at another pot; 6-max UTG open, HJ and BB call: UTG's
        spot at the same pot. On a flop the spot stores its mixes, on another its stored flops most like it; never
        the stored answer itself (solution), with ranges of your own, or where the ranges are too far apart."""
        from gto.cards import cards
        lib, ML = self.lib, self.ML
        ante = ML.spot_setup("UTG:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:c", ante=0.125)
        six = ML.spot_setup("UTG:r2.5,HJ:c,CO:f,BTN:f,SB:f,BB:c")
        for s, sid, text in ((ante, "mw_hj_ip_bb", r"^its pot 8bb with 97.5 behind, this one 8.75bb with 97.4; the "
                                                   r"ranges differ by [\d.]+%"),
                             (six, "mw_utg_btn_bb", r"^the same pot; the ranges differ by [\d.]+%")):
            b = lib.stand_in(s["ranges"], s["pot"], s["stack"])
            self.assertEqual(b["spot"]["id"], sid)
            self.assertLessEqual(sum(b["dist"]), ML.MW_STAND_DIST)
            self.assertIsNone(lib.borrow(s["ranges"], s["pot"], s["stack"], 6))
            self.assertIsNone(lib.solution(s["ranges"], cards("Kc7d2h"), s["pot"], s["stack"], players=6))
            sol = lib.near(s["ranges"], cards("Kc7d2h"), s["pot"], s["stack"], players=6)    # (a stored flop)
            self.assertTrue(sol.approx["stored"])
            self.assertEqual(sol.approx["near"], [("Kc7d2h", 0)])
            self.assertEqual(sol.borrowed["spot"], sid)
            self.assertRegex(sol.borrowed_text(), text)
            self.assertFalse(sol.action_evs(sol.root).any())
            near = lib.near(s["ranges"], cards("Kd8c3h"), s["pot"], s["stack"], players=6)
            self.assertNotIn("stored", near.approx)
            self.assertEqual(len(near.approx["near"]), 3)
            self.assertEqual(near.borrowed["spot"], sid)
            self.assertIsNone(lib.near(s["ranges"], cards("Kd8c3h"), s["pot"], s["stack"]))   # (ranges of your own)
        far = ML.spot_setup("UTG:f,HJ:r2.5,CO:f,BTN:f,SB:c,BB:c")    # (HJ open, SB and BB call: 3.3 from BTN's)
        self.assertIsNone(lib.stand_in(far["ranges"], far["pot"], far["stack"]))
        self.assertIsNone(lib.near(far["ranges"], cards("Kd8c3h"), far["pot"], far["stack"], players=6))

    def test_advice(self):
        """The answer at once with antes: on a flop the spot stores its mix for the hand, marked approximate with the
        spot and the pot named, the one Study shows; the advice's own answer is the live solve's (no precomputed)."""
        adv = advisor("fast")
        st = dict(ST, preflop=PRE_HJ, board="Kc7d2h", ante=0.125)
        s = adv.instant(dict(st, study=True))
        self.assertRegex(s["approximate"]["label"], r"^the precomputed 3-way solve of this flop in the library's 6-max "
                                                    r"HJ open, CO or BTN and BB call, that spot read for this pot "
                                                    r"\(its pot 8bb")
        rows = {x[0]: x for rows in s["combos"].values() for x in rows}
        hole = held(s, 2)
        r = adv.instant(dict(st, hero="BB", hole=hole))
        self.assertEqual(r["approximate"], s["approximate"])
        self.assertTrue(all("ev_bb" not in a or a["ev_bb"] is None for a in r["actions"]))
        for a, f in zip(r["actions"], rows[hole][2]):
            self.assertAlmostEqual(a["freq"], f, delta=0.002)
        st6 = dict(ST, preflop=PRE_UH, board="Kd8c3h")
        s = adv.instant(dict(st6, study=True))
        self.assertRegex(s["approximate"]["label"], r"most similar flops \(.*\), by card substitution in the "
                                                    r"library's 6-max UTG open, BTN and BB call, that spot read for "
                                                    r"this pot \(the same pot")


if __name__ == "__main__":
    unittest.main()
