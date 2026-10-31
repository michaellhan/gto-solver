"""Advisor.instant on a library spot's flop that the library doesn't store: its decisions read off the spot's stored
flops most like it (gto/flop_approx.py: same pairing and suit pattern, nearest ranks, card substitution, averaged),
marked approximate (advisor.instant_from_near); the flop's own solve still answers the request. The stored flops it
reads are prefetched once the flop is known (Advisor.prefetch). A heads-up pot the library holds no spot for is read
off the spot whose ranges are nearest (flop_approx.near_spot). On the shipped library."""
import json
import unittest
from unittest import mock

import numpy as np

from gto import advisor as A
from gto import flop_approx as fa
from gto import flop_library as fl
from gto import native
from gto.cards import cards as parse_cards, cards_str, COMBO_INDEX
from gto.cli import parse_actions

SID = "srp_hj_co"                     # (stored on 552 flops: the opens flatted behind)
LINE = dict((s[0], s[2]) for s in fl.SPOTS)[SID]
BOARD = "KhJd6c"                      # (KcJd6h canonically: not stored; read off KcJd5h, KcJd7h, KcQd6h)
X = lambda p: {"pos": p, "action": "check"}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}
C = lambda p: {"pos": p, "action": "call"}


def advisor(lib=None):
    a = native.advisor("fast", budget=0.5, prior_budget=0.5)
    if lib is not None:
        a.library = lib
    return a


class _Lib(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = fl.default_library()
        cls.spot = next((s for s in (cls.lib.index().get("spots", []) if cls.lib else []) if s["id"] == SID), None)
        if cls.spot is None:
            raise unittest.SkipTest("the shipped library doesn't hold " + SID)
        cls.setup = fl.spot_setup(LINE)
        cls.hand = {"players": 6, "stack": 100, "preflop": parse_actions(LINE), "board": BOARD, "turn": [], "river": []}

    def expected(self, board, hole, path, spot=None, skip=()):
        """The mix instant_from_near should give `hole` at the stored node `path`: its images' stored mixes on the K
        nearest stored flops, averaged with weights 1 / (1 + distance); `spot` another spot's (near_spot), whose
        entry of the flop, where it stores it, is the one read; skip: flops left out (the image another hand class)."""
        b = parse_cards(board)
        canon, smap = fl.canonical(b)
        c = [x // 4 * 4 + smap[x % 4] for x in parse_cards(hole)]
        tot, wsum, names = 0.0, 0.0, None
        sp = spot or self.spot
        nb = fa.neighbours(sorted(sp["flops"]), list(canon), fa.K + 1)
        nb = nb[:1] if spot is not None and nb[0][2] == 0 else [x for x in nb if x[2] > 0][:fa.K]
        for nm, m, d in nb:
            if nm in skip:
                continue
            e = self.lib.entry(sp, nm)
            nd = next((n for n in e["nodes"] if n["path"] == list(path)), None)
            if nd is None:
                continue
            i = COMBO_INDEX[tuple(sorted(int(m[x]) for x in c))]
            mask = e["masks"][nd["player"]]
            if not mask[i]:
                continue
            col = int(np.searchsorted(np.nonzero(mask)[0], i))
            w = 1.0 / (1.0 + d)
            tot = tot + w * nd["strat"][:, col]
            wsum += w
            names = nd["names"]
        return names, tot / wsum


class TestFlopApprox(_Lib):
    def test_card_map(self):
        """A bijection of the cards taking the flop onto the stored flop, keeping its rank order and pairing, the
        other ranks in order; None for another pairing or suit pattern."""
        t, s = parse_cards("KcJd6h"), parse_cards("KcJd5h")
        m, d = fa.card_map(t, s)
        self.assertEqual(sorted(m.tolist()), list(range(52)))
        self.assertEqual(sorted(int(m[c]) for c in t), sorted(s))
        self.assertEqual(d, 1)
        self.assertEqual(cards_str([m[c] for c in parse_cards("6s5s")]), "5s6s")   # (bottom pair, the kicker below)
        self.assertEqual(cards_str([m[c] for c in parse_cards("AhQh")]), "AhQh")
        self.assertIsNone(fa.card_map(t, parse_cards("KcJc5h")))     # two-tone
        self.assertIsNone(fa.card_map(t, parse_cards("KcKd5h")))     # paired
        self.assertIsNone(fa.card_map(parse_cards("KcJc6h"), parse_cards("KcJd5d")))   # the suited pair differs

    def test_neighbours(self):
        """The nearest stored flops of the same pattern, least distance first; a stored flop's own entry isn't one."""
        canon = list(fl.canonical(parse_cards(BOARD))[0])
        nb = fa.neighbours(sorted(self.spot["flops"]), canon, 3)
        self.assertEqual([n for n, _, _ in nb], ["KcJd5h", "KcJd7h", "KcQd6h"])
        self.assertEqual([d for _, _, d in nb], [1, 1, 1])
        self.assertNotIn(cards_str(canon), self.spot["flops"])
        src = fa.sources(self.lib, self.spot, parse_cards("KcJd5h"))
        self.assertNotIn("KcJd5h", [s[3] for s in src])
        self.assertEqual(len(src), fa.K)


class TestInstantFromNear(_Lib):
    def setUp(self):   # (the averaged mixes themselves: the learned correction's tested below)
        p = mock.patch.object(fa, "theta", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def test_first_to_act(self):
        """The opener first to act on a flop not stored: its combo's images' mixes averaged, marked approximate, the
        flops named, nothing solved; the bets at the hand's pot."""
        adv = advisor()
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            res = adv.instant(dict(self.hand, hero="HJ", hole="AhQd", flop=[]))
        self.assertEqual(res["approximate"]["source"], "near")
        self.assertIn("KhJd5c, KhJd7c, KhQd6c", res["approximate"]["label"])   # (KcJd5h, ... in the hand's suits)
        names, mix = self.expected(BOARD, "AhQd", [])
        self.assertEqual([a["action"] for a in res["actions"]], names)
        self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0006))
        bet = next(a for a in res["actions"] if a["action"] == "bet 75%")
        self.assertAlmostEqual(bet["amount_bb"], round(0.75 * self.setup["pot"], 2), places=2)
        self.assertNotIn("solver", res)
        self.assertTrue(any("this flop's own solve replaces it" in n for n in res["notes"]))

    def test_hand_class(self):
        """An image of another hand class doesn't count where one of the same class does: 8h9h on KhJd6c is no pair
        and a backdoor straight draw, as its images on KcJd5h and KcQd6h, but on KcJd7h a gutshot."""
        canon = list(fl.canonical(parse_cards(BOARD))[0])
        cmap = fl.combo_map(fl.canonical(parse_cards(BOARD))[1])
        i = int(cmap[COMBO_INDEX[tuple(sorted(parse_cards("8h9h")))]])
        here = fa.combo_classes(canon, np.array([i]))[0]
        there = {nm: fa.combo_classes(parse_cards(nm), fa.combo_image(m)[[i]])[0]
                 for nm, m, _ in fa.neighbours(sorted(self.spot["flops"]), canon, 3)}
        self.assertEqual(there, {"KcJd5h": here, "KcJd7h": there["KcJd7h"], "KcQd6h": here})
        self.assertNotEqual(there["KcJd7h"], here)
        res = advisor().instant(dict(self.hand, hero="HJ", hole="8h9h", flop=[]))
        names, mix = self.expected(BOARD, "8h9h", [], skip=("KcJd7h",))
        self.assertEqual([a["action"] for a in res["actions"]], names)
        self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0006))
        _, every = self.expected(BOARD, "8h9h", [])
        self.assertFalse(np.allclose(mix, every, atol=0.01))

    def test_in_position_and_facing_a_bet(self):
        """The caller after a check, and the opener facing its bet (the size read at the tree's nearest)."""
        adv = advisor()
        res = adv.instant(dict(self.hand, hero="CO", hole="JhTh", flop=[X("HJ")]))
        names, mix = self.expected(BOARD, "JhTh", ["check"])
        self.assertEqual(res["approximate"]["source"], "near")
        self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0006))
        pot = self.setup["pot"]
        res = adv.instant(dict(self.hand, hero="HJ", hole="AdJc", flop=[X("HJ"), B("CO", round(0.75 * pot, 2))]))
        names, mix = self.expected(BOARD, "AdJc", ["check", "bet 75%"])
        self.assertEqual([a["action"].split()[0] for a in res["actions"]], ["fold", "call", "raise"])
        self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0006))
        self.assertAlmostEqual(res["to_call_bb"], round(0.75 * pot, 2), places=2)

    def test_none_where_it_doesnt_apply(self):
        """A stored flop (the library answers it at once), a hand no stored flop's range holds (the caller's
        queens: it 3bets them), a pot the library doesn't hold, the turn: no approximate answer from it."""
        adv = advisor()
        self.assertIn("KcJd5h", self.spot["flops"])
        self.assertIsNone(adv.instant(dict(self.hand, hero="HJ", hole="AhQd", board="KcJd5h", flop=[])))
        self.assertIsNone(adv.instant(dict(self.hand, hero="CO", hole="QhQd", flop=[X("HJ")])))
        limped = dict(self.hand, preflop=parse_actions("UTG:f,HJ:f,CO:f,BTN:f,SB:c,BB:x"), hero="SB", hole="AhQd",
                      flop=[])
        self.assertIsNone(adv.instant(limped))
        turn = dict(self.hand, hero="HJ", hole="AhQd", board=BOARD + "2s", flop=[X("HJ"), X("CO")])
        res = adv.instant(turn)
        self.assertTrue(res is None or res["approximate"]["source"] != "near")

    def test_far_flops_arent_read(self):
        """Stored flops further than MAX_DIST aren't read: none near enough, no answer."""
        with mock.patch.object(fa, "MAX_DIST", 0):
            self.assertIsNone(advisor().instant(dict(self.hand, hero="HJ", hole="AhQd", flop=[])))

    def test_the_request_still_solves(self):
        """The request itself solves the flop live: its answer is the solve's."""
        st = dict(self.hand, hero="HJ", hole="AhQd", flop=[])
        adv = advisor()
        self.assertEqual(adv.instant(st)["approximate"]["source"], "near")
        res = adv.advise(st)
        self.assertNotIn("approximate", res)
        self.assertNotIn("precomputed", res)
        self.assertGreater(res["solver"]["iterations"], 0)

    def test_before_the_flops_own_solve_until_it_has_run_a_while(self):
        """The flop's own solve in hand (a request's stopped part way): the stored flops answer until it has run
        NEAR_UNTIL iterations, then the solve does (instant_from_own)."""
        st = dict(self.hand, hero="HJ", hole="AhQd", flop=[])
        adv = advisor()
        adv.hu_settled = None
        adv.advise(st)
        own = next(s for k, s in adv._cache.items() if len(k[2]) == 3)
        own.settled_at = None
        adv.time_budget = 1e3   # (short of its budget: not an answer that's quick anyway)
        n = own.solver.iters
        self.assertGreaterEqual(n, A.HU_CHECKS[0])
        with mock.patch.object(A, "NEAR_UNTIL", n + 1):
            self.assertEqual(adv.instant(st)["approximate"]["source"], "near")
        with mock.patch.object(A, "NEAR_UNTIL", n):
            self.assertEqual(adv.instant(st)["approximate"]["source"], "solve")

    def test_prefetched(self):
        """Once the flop is known, the prefetch's dry run asks for the stored flops the answer reads (the browser's
        worker fetches them ahead) and keeps none as missing."""
        asked = []

        def fetch(p, real=self.lib._fetch):
            if p.startswith("index"):
                return real(p)
            asked.append(p)
            return None
        rec = fl.FlopLibrary(fetch)
        adv = advisor(rec)
        adv.prefetch(dict(self.hand, hero="HJ", hole="AhQd", flop=[]))
        for nm in ("KcJd5h", "KcJd7h", "KcQd6h"):
            self.assertIn(f"{SID}/{nm}.bin?v={self.spot['flops'][nm]}", asked)
        self.assertFalse([k for k, e in rec._entries.items() if e is None])



class TestCorrection(_Lib):
    def srcs_feats(self, hole):
        canon, smap = fl.canonical(parse_cards(BOARD))
        i = int(fl.combo_map(smap)[COMBO_INDEX[tuple(sorted(parse_cards(hole)))]])
        srcs = [(self.lib.entry(self.spot, nm), m, 1.0 / (1 + d))
                for nm, m, d in fa.neighbours(sorted(self.spot["flops"]), list(canon), fa.K)]
        return list(canon), srcs, i

    def test_applied(self):
        """A library spot's flop not stored: facing a bet hero's averaged mix takes the learned correction
        (correct_mix of the class average, gto/flop_approx_model.py's weights), a mix still; the first decisions (the
        root, after a check) stay as averaged."""
        th = fa.theta()
        if th is None:
            self.skipTest("no gto/flop_approx_model.py")
        pot = self.setup["pot"]
        moved = []
        for hero, hole, flop, path, corrected in (
                ("HJ", "AdJc", [X("HJ"), B("CO", round(0.75 * pot, 2))], ("check", "bet 75%"), True),
                ("CO", "JhTh", [B("HJ", round(0.33 * pot, 2))], ("bet 33%",), True),
                ("HJ", "AhQd", [], (), False), ("CO", "JhTh", [X("HJ")], ("check",), False)):
            res = advisor().instant(dict(self.hand, hero=hero, hole=hole, flop=flop))
            canon, srcs, i = self.srcs_feats(hole)
            plain = fa.predict(srcs, board=canon, combos=[i])[path]
            f = fa.correction_features(canon, srcs, np.array([i]))
            want = fa.correct_mix(th, plain["names"], path, plain["player"], plain["strat"][:, [i]], f)[:, 0]
            self.assertEqual(bool(th["on"][fa.node_kind(path, plain["player"])]), corrected, path)
            if not corrected:
                self.assertEqual(want.tolist(), plain["strat"][:, i].tolist())
            got = np.array([a["freq"] for a in res["actions"]])
            self.assertTrue(np.allclose(got, want, atol=0.0006), (hole, path, got, want))
            self.assertAlmostEqual(float(want.sum()), 1.0, places=6)
            self.assertEqual(fa.predict(srcs, board=canon, combos=[i], correct=True)[path]["strat"][:, i].round(6).tolist(),
                             want.round(6).tolist())
            moved.append(not np.allclose(want, plain["strat"][:, i], atol=0.002))
        self.assertTrue(moved[0] or moved[1])   # (it moves something)

    def test_other_spot_and_depth(self):
        """Facing a bet a pot read off another spot takes the correction (BB 3bets the HJ, read off BB 3bet vs CO's
        stored flops most like KhJd6c, which it doesn't store, and its own entry of KhJd5c), as does one at another
        depth read off its 100bb stored flops most like the flop (80bb BB 3bet vs BTN on KhJd6c); not the 100bb entry
        of the flop itself (KhJd5c)."""
        th = fa.theta()
        if th is None:
            self.skipTest("no gto/flop_approx_model.py")
        bb3 = "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c"
        holes = ("AhQd", "JhTh", "9h9s", "AhKc", "Ah5h", "QdTd")
        moved = []
        for line, depth, sid, hero, board, stored, corrected in (
                (HJ_BB3, 100, "3bp_bb_co", "HJ", BOARD, False, True),
                (HJ_BB3, 100, "3bp_bb_co", "HJ", "KhJd5c", True, True),
                (bb3, 80, "3bp_bb_btn", "BTN", BOARD, False, True),
                (bb3, 80, "3bp_bb_btn", "BTN", "KhJd5c", True, False)):
            sp = next(x for x in self.lib.index()["spots"] if x["id"] == sid)
            st = fl.spot_setup(line, 6, depth)
            canon, smap = fl.canonical(parse_cards(board))
            self.assertEqual(fl.cards_str(canon) in sp["flops"], stored)
            srcs = [(self.lib.entry(sp, nm), m, 1.0 / (1 + d))
                    for nm, m, d in fa.neighbours(sorted(sp["flops"]), list(canon), 1 if stored else fa.K)]
            hand = {"players": 6, "stack": depth, "preflop": parse_actions(line), "board": board, "turn": [],
                    "river": [], "hero": hero, "flop": [B("BB", round(0.33 * st["pot"], 2))]}
            adv, answered = advisor(), 0
            for hole in holes:
                res = adv.instant(dict(hand, hole=hole))
                if res is None:   # (a hand the caller's range doesn't hold)
                    continue
                answered += 1
                self.assertEqual(res["approximate"]["source"], "near")
                i = int(fl.combo_map(smap)[COMBO_INDEX[tuple(sorted(parse_cards(hole)))]])
                plain = fa.predict(srcs, board=list(canon), combos=[i])[("bet 33%",)]
                want = plain["strat"][:, i]
                if corrected:
                    f = fa.correction_features(list(canon), srcs, np.array([i]))
                    want = fa.correct_mix(th, plain["names"], ("bet 33%",), plain["player"], plain["strat"][:, [i]],
                                          f)[:, 0]
                    moved.append(not np.allclose(want, plain["strat"][:, i], atol=0.002))
                got = np.array([a["freq"] for a in res["actions"]])
                self.assertTrue(np.allclose(got, want, atol=0.0006), (sid, board, hole, got, want))
            self.assertGreater(answered, 2, (sid, board))
            if corrected:
                self.assertTrue(any(moved), (sid, board))
                moved = []

    def test_mixes(self):
        """Every combo's corrected mix is a mix; combos no source holds stay unanswered (NaN); without the board (a
        stored flop's own entry) nothing is corrected."""
        if fa.theta() is None:
            self.skipTest("no gto/flop_approx_model.py")
        canon, srcs, _ = self.srcs_feats("AhQd")
        c, u = fa.predict(srcs, board=canon, correct=True), fa.predict(srcs, board=canon)
        for path, o in c.items():
            ok = ~np.isnan(u[path]["strat"][0])
            self.assertTrue(np.array_equal(ok, ~np.isnan(o["strat"][0])))
            self.assertTrue(np.allclose(o["strat"][:, ok].sum(axis=0), 1.0))
            self.assertTrue((o["strat"][:, ok] >= 0).all())
        a, b = fa.predict(srcs, correct=True), fa.predict(srcs)
        for path in a:
            self.assertTrue(np.array_equal(np.nan_to_num(a[path]["strat"]), np.nan_to_num(b[path]["strat"])))


HJ_BB3 = "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:f,BB:r10,HJ:c"     # (BB 3bets the HJ: no spot of its own; read off 3bp_bb_co)


class TestNearSpot(_Lib):
    """A heads-up pot the library holds no spot for, read off the spot whose ranges are nearest (near_spot): the
    same pot and stack, each range covered by NEAR_COVER, the least distance; at another depth, the same pot."""

    def test_which_spot(self):
        lines = {HJ_BB3: "3bp_bb_co", "UTG:r2.5,HJ:f,CO:f,BTN:f,SB:f,BB:r10,UTG:c": "3bp_bb_co",
                 "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:r10,BB:f,HJ:c": "3bp_sb_co",
                 "UTG:r2.5,HJ:f,CO:f,BTN:r7.5,SB:f,BB:f,UTG:c": "3bp_co_hj",
                 "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:c,BB:f": "srp_co_sb",
                 "UTG:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:f": None,                       # (srp_hj_co covers 80%)
                 "UTG:f,HJ:f,CO:r2.5,BTN:r7.5,SB:f,BB:f,CO:r20,BTN:c": None}       # (a 4bet pot: no pot like it)
        for line, sid in lines.items():
            st = fl.spot_setup(line)
            self.assertIsNone(self.lib.spot(fl.range_fp(st["full0"]), fl.range_fp(st["full1"]), st["pot"], st["stack"]))
            sp = fa.near_spot(self.lib, st["full0"], st["full1"], st["pot"], st["stack"])
            self.assertEqual(sp and sp["id"], sid, line)
            if sp is not None:
                ss = fl.spot_setup(sp["line"])
                for t, u in ((st["full0"], ss["full0"]), (st["full1"], ss["full1"])):
                    self.assertGreaterEqual(fa.range_cover(t, u), fa.NEAR_COVER)

    def test_instant(self):
        """The 3bettor first to act: on a flop the nearest spot stores, its entry's mix (the spot named); on one it
        doesn't, its K stored flops most like it; the caller after a check."""
        sp = next(s for s in self.lib.index()["spots"] if s["id"] == "3bp_bb_co")
        st = fl.spot_setup(HJ_BB3)
        hand = {"players": 6, "stack": 100, "preflop": parse_actions(HJ_BB3), "turn": [], "river": []}
        self.assertGreater(st["full0"][COMBO_INDEX[tuple(sorted(parse_cards("AhKd")))]], 0)
        adv = advisor()
        for board, stored in (("KhJd5c", True), ("KhJd6c", False)):
            self.assertEqual(fl.cards_str(fl.canonical(parse_cards(board))[0]) in sp["flops"], stored)
            with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
                res = adv.instant(dict(hand, board=board, hero="BB", hole="AhKd", flop=[]))
            self.assertEqual(res["approximate"]["source"], "near")
            self.assertIn("the library's BB 3bet vs CO", res["approximate"]["label"])
            self.assertEqual("precomputed solve of this flop" in res["approximate"]["label"], stored)
            names, mix = self.expected(board, "AhKd", [], spot=sp)
            self.assertEqual([a["action"] for a in res["actions"]], names)
            self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0006))
            bet = next(a for a in res["actions"] if a["action"] == "bet 33%")
            self.assertAlmostEqual(bet["amount_bb"], round(0.33 * st["pot"], 2), places=2)
            self.assertTrue(any("this flop's own solve replaces it" in n for n in res["notes"]))
        res = adv.instant(dict(hand, board="KhJd6c", hero="HJ", hole="QhQd", flop=[X("BB")]))
        names, mix = self.expected("KhJd6c", "QhQd", ["check"], spot=sp)
        self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0006))

    def test_depths(self):
        """Another stack depth: the same spot at 100bb where the stack-to-pot ratio is within NEAR_SPR of its own and
        at least NEAR_MIN_SPR; none at 30bb or 250bb, nor in a 3bet pot at 60bb (SPR 2.4) or 200bb (ratio 2.11)."""
        srp, bb3 = "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c"
        cases = {(srp, 30): None, (srp, 40): "srp_btn_bb", (srp, 60): "srp_btn_bb", (srp, 150): "srp_btn_bb",
                 (srp, 200): "srp_btn_bb", (srp, 250): None, (bb3, 60): None, (bb3, 80): "3bp_bb_btn",
                 (bb3, 175): "3bp_bb_btn", (bb3, 200): None}
        for (line, depth), sid in cases.items():
            st = fl.spot_setup(line, 6, depth)
            self.assertIsNone(self.lib.spot(fl.range_fp(st["full0"]), fl.range_fp(st["full1"]), st["pot"], st["stack"]))
            sp = fa.near_spot(self.lib, st["full0"], st["full1"], st["pot"], st["stack"])
            self.assertEqual(sp and sp["id"], sid, (line, depth))

    def test_instant_depth(self):
        """80bb: the BB first to act on a flop BTN vs BB stores reads its 100bb entry (the depth named); the BTN after
        a check too."""
        line = "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c"
        sp = next(s for s in self.lib.index()["spots"] if s["id"] == "srp_btn_bb")
        hand = {"players": 6, "stack": 80, "preflop": parse_actions(line), "board": BOARD, "turn": [], "river": []}
        adv = advisor()
        for hero, hole, flop, path in (("BB", "Qh9h", [], []), ("BTN", "AhQd", [X("BB")], ["check"])):
            with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
                res = adv.instant(dict(hand, hero=hero, hole=hole, flop=flop))
            self.assertEqual(res["approximate"]["source"], "near")
            label = res["approximate"]["label"]
            self.assertIn("the precomputed solve of this flop in the library's BTN vs BB", label)
            self.assertIn("solved deeper (a stack-to-pot ratio of 17.7, this pot's 14.1)", label)
            self.assertNotIn("nearest", label)
            names, mix = self.expected(BOARD, hole, path, spot=sp)
            self.assertEqual([a["action"] for a in res["actions"]], names)
            self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0006))
        self.assertIsNone(adv.instant(dict(hand, stack=30, hero="BB", hole="Qh9h", flop=[])))

    def test_none_without_one(self):
        """A pot no spot covers: no approximate answer."""
        line = "UTG:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:f"
        hand = {"players": 6, "stack": 100, "preflop": parse_actions(line), "board": BOARD, "turn": [], "river": [],
                "hero": "CO", "hole": "AhQd", "flop": []}
        self.assertIsNone(advisor().instant(hand))

    def test_prefetched(self):
        """The prefetch names the nearest spot's entry of the flop for the browser to fetch ahead."""
        asked = []

        def fetch(p, real=self.lib._fetch):
            if p.startswith("index"):
                return real(p)
            asked.append(p)
            return None
        sp = next(s for s in self.lib.index()["spots"] if s["id"] == "3bp_bb_co")
        adv = advisor(fl.FlopLibrary(fetch))
        adv.prefetch({"players": 6, "stack": 100, "preflop": parse_actions(HJ_BB3), "board": "KhJd5c", "turn": [],
                      "river": [], "hero": "BB", "hole": "AhKd", "flop": []})
        self.assertIn(f"3bp_bb_co/KcJd5h.bin?v={sp['flops']['KcJd5h']}", asked)



class TestStudyNear(_Lib):
    """Study (Advisor.instant with "study": no cards of anyone's own) on a flop the library doesn't store: the actor's
    whole range at once off the stored flops most like it (Advisor._study_near), the mix per combo as the advice's
    approximate answer gives each hand, marked approximate with what it borrowed; no EVs or equities; the flop's own
    solve still answers the request."""

    def study(self, **kw):
        return {**self.hand, "study": True, "flop": [], **kw}

    @staticmethod
    def row(res, hole):
        cards = set(parse_cards(hole))
        return next(r for rows in res["combos"].values() for r in rows if set(parse_cards(r[0])) == cards)

    def test_whole_range_at_once(self):
        """HJ first to act: every combo's mix that of its images on the nearest stored flops (8h9h's without
        KcJd7h's, where it's another hand class), the range's frequency per action, the flops named; nothing
        solved, no EVs; then the live solve's answer has the same seats, pot and sizes, without "approximate"."""
        adv = advisor()
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            res = adv.instant(self.study())
        json.dumps(res, allow_nan=False)
        self.assertEqual((res["study"], res["street"], res["actor"], res["opponent"]), ("node", "flop", "HJ", "CO"))
        self.assertEqual(res["approximate"], {"source": "near", "iterations": None, "label": "the precomputed solves of "
                         "the most similar flops (KhJd5c, KhJd7c, KhQd6c), by card substitution"})
        for hole, skip in (("AhQd", ()), ("8h9h", ("KcJd7h",)), ("JcTc", ())):
            names, mix = self.expected(BOARD, hole, [], skip=skip)
            self.assertEqual([a["action"] for a in res["actions"]], names)
            self.assertTrue(np.allclose(self.row(res, hole)[2], mix, atol=0.0006), hole)
        rows = [r for rs in res["combos"].values() for r in rs]
        self.assertTrue(all(r[3] is None and r[4] is None for r in rows))
        w = sum(r[1] for r in rows)
        self.assertAlmostEqual(res["range_combos"], w, delta=0.1)
        for i, a in enumerate(res["actions"]):
            self.assertAlmostEqual(a["freq"], sum(r[1] * r[2][i] for r in rows) / w, delta=0.002)
            self.assertIsNone(a["ev_bb"])
        self.assertTrue(all(c["ev"] is None and c["eq"] is None for c in res["grid"]["cells"].values()))
        self.assertEqual((res["range_ev_bb"], res["range_eq"], res["opponent_eq"]), (None, None, None))
        live = adv.study(self.study())
        self.assertEqual(set(res) - set(live), {"approximate"})
        for k in ("actor", "opponent", "position", "players", "board", "pot_bb", "to_call_bb", "stack_bb"):
            self.assertEqual(res[k], live[k], k)
        self.assertEqual([(a["action"], a["amount_bb"]) for a in res["actions"]],
                         [(a["action"], a["amount_bb"]) for a in live["actions"]])
        self.assertGreater(live["solver"]["iterations"], 0)
        self.assertIsNone(adv.instant(self.study()))   # (its own solve settled: the request's answer is quick)

    def test_in_position_and_facing_a_bet(self):
        """CO after a check, HJ facing its 75% bet (to call at the hand's pot), on 8c7d5h (read off 8c7d4h, 8c7d6h,
        8c6d5h, where the opener checks all of its range: its checking range is all of it); each combo's mix the
        advice's approximate answer for that hand."""
        adv, board = advisor(), "8c7d5h"
        first = adv.instant(self.study(board=board))
        self.assertEqual([a["freq"] for a in first["actions"]], [1.0, 0.0, 0.0])
        res = adv.instant(self.study(board=board, flop=[X("HJ")]))
        self.assertEqual((res["actor"], res["position"], res["approximate"]["source"]), ("CO", "IP", "near"))
        for hole in ("JhTh", "9h8h"):   # (each combo's mix the advice's approximate answer for it)
            mine = adv.instant(dict(self.hand, board=board, hero="CO", hole=hole, flop=[X("HJ")]))
            self.assertEqual([a["action"] for a in res["actions"]], [a["action"] for a in mine["actions"]])
            self.assertTrue(np.allclose(self.row(res, hole)[2], [a["freq"] for a in mine["actions"]], atol=0.0006))
        self.assertAlmostEqual(res["opponent_combos"], first["range_combos"], delta=0.5)
        pot = self.setup["pot"]
        res = adv.instant(self.study(board=board, flop=[X("HJ"), B("CO", round(0.75 * pot, 2))]))
        self.assertEqual([a["action"].split()[0] for a in res["actions"]], ["fold", "call", "raise"])
        mine = adv.instant(dict(self.hand, board=board, hero="HJ", hole="AdJc",
                                flop=[X("HJ"), B("CO", round(0.75 * pot, 2))]))
        self.assertTrue(np.allclose(self.row(res, "AdJc")[2], [a["freq"] for a in mine["actions"]], atol=0.0006))
        self.assertAlmostEqual(res["to_call_bb"], round(0.75 * pot, 2), places=2)
        self.assertLess(res["opponent_combos"], 0.5 * first["range_combos"])   # (the hands CO bets)

    def test_the_next_node_reads_the_flops_again(self):
        """The flop's next node reads the stored flops' prediction kept from its first (flop_approx.PRED_KEEP per
        library), made for the combos either range holds."""
        adv, calls, real = advisor(fl.FlopLibrary(self.lib._fetch)), [], fa.predict

        def predict(*a, **k):
            calls.append(len(k["combos"]))
            return real(*a, **k)
        with mock.patch.object(fa, "predict", side_effect=predict):
            first = adv.instant(self.study(board="8c7d5h"))
            nxt = adv.instant(self.study(board="8c7d5h", flop=[X("HJ")]))
        self.assertEqual(nxt["actor"], "CO")
        self.assertEqual(len(calls), 1)
        self.assertLess(calls[0], 0.3 * 1326)
        self.assertEqual(first["approximate"], nxt["approximate"])

    def test_borrowed_spot_and_depth(self):
        """A pot the library holds no spot for (BB 3bets HJ: BB 3bet vs CO's flops) and another depth (80bb: BTN vs
        BB's entry of the flop at 100bb): the spot and the depth borrowed named, as the advice's approximate answer
        names them."""
        line = "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c"
        adv = advisor()
        res = adv.instant({"players": 6, "stack": 80, "preflop": parse_actions(line), "board": BOARD, "flop": [],
                           "turn": [], "river": [], "study": True})
        self.assertEqual(res["actor"], "BB")
        self.assertEqual(res["approximate"]["label"], "the precomputed solve of this flop in the library's BTN vs BB "
                         "single-raised pot, solved deeper (a stack-to-pot ratio of 17.7, this pot's 14.1)")
        res = adv.instant({"players": 6, "stack": 100, "preflop": parse_actions(HJ_BB3), "board": "KhJd6c",
                           "flop": [], "turn": [], "river": [], "study": True})
        self.assertEqual(res["actor"], "BB")
        self.assertTrue(res["approximate"]["label"].startswith("the library's BB 3bet vs CO on the most similar "
                                                               "flops ("), res["approximate"]["label"])

    def test_none_where_it_doesnt_apply(self):
        """A stored flop (the library's answer is at once anyway), a pot the library doesn't hold, the turn after
        an unstored flop, no stored flop near enough, a line the stored flops rarely take (HJ checks 7% of its range on
        KhJd6c's: the library's own answer solves those live too): no answer at once."""
        adv = advisor()
        self.assertIsNone(adv.instant(self.study(flop=[X("HJ")])))
        self.assertIsNone(adv.instant(self.study(board="KcJd5h")))
        limped = dict(self.study(), preflop=parse_actions("UTG:f,HJ:f,CO:f,BTN:f,SB:c,BB:x"))
        self.assertIsNone(adv.instant(limped))
        self.assertIsNone(adv.instant(self.study(board=BOARD + "2s", flop=[X("HJ"), X("CO")])))
        with mock.patch.object(fa, "MAX_DIST", 0):
            self.assertIsNone(adv.instant(self.study()))

    def test_prefetched(self):
        """Study's prefetch (no seat or cards of its own) names the stored flops its answer at once reads."""
        asked = []

        def fetch(p, real=self.lib._fetch):
            if p.startswith("index"):
                return real(p)
            asked.append(p)
            return None
        rec = fl.FlopLibrary(fetch)
        advisor(rec).prefetch(self.study())
        for nm in ("KcJd5h", "KcJd7h", "KcQd6h"):
            self.assertIn(f"{SID}/{nm}.bin?v={self.spot['flops'][nm]}", asked)
        self.assertFalse([k for k, e in rec._entries.items() if e is None])


if __name__ == "__main__":
    unittest.main()


EIGHT_BTN = "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c"
EIGHT_SB = "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:f,SB:r3,BB:c"


class TestNearExact(_Lib):
    """Another table size's pot that plays as a library spot (the same pot and stack, each range within a few combos:
    flop_approx.near_exact) is read as the library's own solve: instant, with its EVs, marked borrowed; the combos off
    the spot's ranges are left out, and your hand off them is solved live."""

    def test_which_spot(self):
        cases = {(EIGHT_BTN, 8): "srp_btn_bb", (EIGHT_SB, 8): "srp_sb_bb",
                 ("UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c", 8): "srp_co_bb",
                 ("UTG:f,UTG1:f,LJ:f,HJ:r2.5,CO:f,BTN:f,SB:f,BB:c", 8): "srp_hj_bb",
                 ("UTG:f,UTG1:f,UTG2:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", 9): "srp_btn_bb",
                 ("UTG:r2.5,UTG1:f,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:c", 8): "srp_utg_bb",   # (0.46 apart)
                 ("UTG:f,UTG1:f,LJ:r2.5,HJ:f,CO:f,BTN:c,SB:f,BB:f", 8): "srp_utg_btn",
                 ("UTG:f,UTG1:f,LJ:r2.5,HJ:r7.5,CO:f,BTN:f,SB:f,BB:f,LJ:c", 8): "3bp_hj_utg",
                 ("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", 6): None,               # (the library's own spot)
                 ("UTG:r2.5,HJ:f,CO:f,BTN:r7.5,SB:f,BB:f,UTG:c", 6): None}      # (6-max's own pot: not borrowed)
        for (line, n), sid in cases.items():
            st = fl.spot_setup(line, n)
            ne = fa.near_exact(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], n)
            self.assertEqual(ne and ne["spot"]["id"], sid, line)
            if ne is not None:
                self.assertGreaterEqual(min(ne["cover"]), fa.NEAR_EXACT_COVER)
                self.assertLessEqual(sum(ne["dist"]), fa.NEAR_EXACT_DIST)
        # (the same 6-max pot as another table size's would read CO's 3bet of HJ: only another table size's spots)
        st = fl.spot_setup("UTG:r2.5,HJ:f,CO:f,BTN:r7.5,SB:f,BB:f,UTG:c", 6)
        self.assertEqual(fa.near_exact(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], 8)["spot"]["id"],
                         "3bp_co_hj")
        for game in ({"ante": 0.125}, {"straddle": True}, {"stack": 60}):   # (another pot or stack: never)
            st = fl.spot_setup("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", 6, **game)
            self.assertIsNone(fa.near_exact(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], 8), game)

    def test_advice(self):
        """8-max BTN vs BB: the BB first to act and the BTN after a check get the 6-max spot's stored mix and EVs,
        nothing solved, the spot and how far named; without library_borrow the flop's approximate read instead."""
        six = "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c"
        adv = advisor()
        for hero, hole, flop in (("BB", "Qh9h", []), ("BTN", "AhQd", [X("BB")])):
            got = {}
            for n, line in ((6, six), (8, EIGHT_BTN)):
                hand = {"players": n, "stack": 100, "preflop": parse_actions(line), "board": BOARD, "turn": [],
                        "river": [], "hero": hero, "hole": hole, "flop": flop}
                with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
                    got[n] = adv.advise(hand)
            r = got[8]
            self.assertEqual(r["precomputed"]["spot"], "srp_btn_bb")
            self.assertEqual(r["precomputed"]["label"], "6-max BTN vs BB single-raised pot")
            b = r["precomputed"]["borrowed"]
            self.assertLessEqual(sum(b["dist"]), fa.NEAR_EXACT_DIST)
            self.assertIn("this table's ranges differ from that spot's by 0.4% of their weight at most", b["text"])
            self.assertTrue(any("precomputed solution (6-max BTN vs BB" in x and "approximate: this table's ranges" in x
                                for x in r["notes"]), r["notes"])
            self.assertNotIn("borrowed", got[6]["precomputed"])
            self.assertEqual([a["action"] for a in r["actions"]], [a["action"] for a in got[6]["actions"]])
            for a, b6 in zip(r["actions"], got[6]["actions"]):
                self.assertAlmostEqual(a["freq"], b6["freq"], places=6)
                self.assertAlmostEqual(a.get("ev_bb", 0.0), b6.get("ev_bb", 0.0), places=6)
            self.assertIsNotNone(r["actions"][0].get("ev_bb"))
        hand = {"players": 8, "stack": 100, "preflop": parse_actions(EIGHT_BTN), "board": BOARD, "turn": [],
                "river": [], "hero": "BB", "hole": "Qh9h", "flop": []}
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            self.assertIsNone(adv.instant(hand))          # (the library's answer is the request's own, at once)
            adv.library_borrow = False
            res = adv.instant(hand)
        self.assertEqual(res["approximate"]["source"], "near")
        self.assertIn("the library's BTN vs BB single-raised pot", res["approximate"]["label"])

    def test_not_for_ranges_of_your_own(self):
        """Ranges typed in ("ranges") are solved as typed: an 8-max BTN vs BB flop with the BB's range entered (its
        own chart's, so the spot would be borrowed) goes to the flop's solve, not the 6-max spot."""
        from gto.cards import ALL_COMBOS
        st = fl.spot_setup(EIGHT_BTN, 8)
        self.assertIsNotNone(fa.near_exact(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], 8))
        text = ",".join(f"{cards_str(cb)}:{w:.4f}" for cb, w in zip(ALL_COMBOS, st["full0"]) if w > 1e-4)
        hand = {"players": 8, "stack": 100, "preflop": parse_actions(EIGHT_BTN), "board": BOARD, "turn": [],
                "river": [], "hero": "BB", "hole": "Qh9h", "flop": []}
        adv = advisor()
        with mock.patch.object(A, "solve", side_effect=RuntimeError("solved")) as solve:
            self.assertIn("borrowed", adv.advise(hand)["precomputed"])
            self.assertFalse(solve.called)
            res = adv.advise(dict(hand, ranges={"BB": text}))
            self.assertTrue(solve.called)     # (the solve was asked; its error here leaves the betting rules)
        self.assertNotIn("precomputed", res)

    def test_prefetched(self):
        """The prefetch names the borrowed spot's entry of the flop for the browser to fetch ahead."""
        asked = []

        def fetch(p, real=self.lib._fetch):
            if p.startswith("index"):
                return real(p)
            asked.append(p)
            return None
        sp = next(s for s in self.lib.index()["spots"] if s["id"] == "srp_btn_bb")
        adv = advisor(fl.FlopLibrary(fetch))
        adv.prefetch({"players": 8, "stack": 100, "preflop": parse_actions(EIGHT_BTN), "board": BOARD, "turn": [],
                      "river": [], "hero": "BB", "hole": "Qh9h", "flop": []})
        name = fl.cards_str(fl.canonical(parse_cards(BOARD))[0])
        self.assertIn(f"srp_btn_bb/{name}.bin?v={sp['flops'][name]}", asked)

    def test_study(self):
        """Study on an 8-max BTN vs BB flop: the BTN's whole range at once from the borrowed spot, EVs included."""
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            r = advisor().advise({"players": 8, "stack": 100, "study": True, "preflop": parse_actions(EIGHT_BTN),
                                  "board": BOARD, "flop": [X("BB")], "turn": [], "river": []})
        self.assertEqual((r["actor"], r["precomputed"]["spot"]), ("BTN", "srp_btn_bb"))
        self.assertIn("borrowed", r["precomputed"])
        self.assertEqual(r["pot_bb"], 5.5)
        self.assertAlmostEqual(sum(a["freq"] for a in r["actions"]), 1.0, delta=0.01)
        self.assertTrue(r["combos"]["AKo"] and all(x[3] is not None for x in r["combos"]["AKo"]))

    def test_combos_off_its_ranges(self):
        """8-max SB vs BB: the SB's K4o isn't in the 6-max spot's range: left out of the ranges read, but as your
        hand the borrowed spot answers nothing (the flop is solved live)."""
        st = fl.spot_setup(EIGHT_SB, 8)
        ne = fa.near_exact(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], 8)
        self.assertEqual(ne["spot"]["id"], "srp_sb_bb")
        ss = fl.spot_setup(ne["spot"]["line"])
        k4 = COMBO_INDEX[tuple(sorted(parse_cards("Kh4c")))]
        self.assertGreater(st["full0"][k4], 0)
        self.assertEqual(ss["full0"][k4], 0)
        board = parse_cards("Qd8s3h")
        cfg = fl.library_config(st["pot"], st["stack"])
        args = (st["full0"], st["full1"], board, st["full0"], st["full1"], st["pot"], st["stack"], cfg)
        self.assertIsNone(self.lib.solution(*args))                         # (not the library's own spot)
        self.assertIsNone(self.lib.solution(*args, borrow=ne, need=[k4]))   # (your hand off its range)
        sol = self.lib.solution(*args, borrow=ne)
        self.assertEqual(sol.borrowed["spot"], "srp_sb_bb")
        self.assertNotIn(k4, set(sol.ctx.active[sol.r0 > 0].tolist()))
        held = COMBO_INDEX[tuple(sorted(parse_cards("AhKd")))]
        self.assertIn(held, set(sol.ctx.active[sol.r0 > 0].tolist()))
        self.assertIsNotNone(self.lib.solution(*args, borrow=ne, need=[held]))


EIGHT_HJ_BTN = "UTG:f,UTG1:f,LJ:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:f"
EIGHT_CO_BTN = "UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:f"
EIGHT_UTG_UTG1 = "UTG:r2.5,UTG1:c,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:f"


class TestBiggerTable(_Lib):
    """At 8- and 9-max a pot of a library pot's size is borrowed with each range 90% held by the spot's
    (NEAR_EXACT_TABLE_COVER) and the ranges up to NEAR_EXACT_TABLE_DIST apart, and where no spot covers it by NEAR_COVER, read approximately off one covering it by
    NEAR_TABLE_COVER: the hands that spot's ranges hold (the rest solved live). 6-max and shorter tables as before."""

    def test_borrowed(self):
        cases = {(EIGHT_HJ_BTN, 8): "srp_hj_co",
                 ("UTG:r2.5,UTG1:f,LJ:f,HJ:f,CO:f,BTN:c,SB:f,BB:f", 8): "srp_utg_btn",
                 ("UTG:f,UTG1:f,LJ:r2.5,HJ:f,CO:r7.5,BTN:f,SB:f,BB:f,LJ:c", 8): "3bp_hj_utg",
                 ("UTG:f,UTG1:f,LJ:f,HJ:r2.5,CO:f,BTN:r7.5,SB:f,BB:f,HJ:c", 8): "3bp_co_hj",
                 ("UTG:f,UTG1:f,UTG2:f,LJ:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:f", 9): "srp_hj_co",
                 (EIGHT_UTG_UTG1, 8): "srp_utg_btn",
                 ("UTG:f,UTG1:r2.5,LJ:c,HJ:f,CO:f,BTN:f,SB:f,BB:f", 8): "srp_utg_btn",
                 ("UTG:f,UTG1:r2.5,UTG2:c,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:f", 9): "srp_utg_btn"}
        for (line, n), sid in cases.items():
            st = fl.spot_setup(line, n)
            ne = fa.near_exact(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], n)
            self.assertEqual(ne and ne["spot"]["id"], sid, line)
            self.assertGreaterEqual(min(ne["cover"]), fa.NEAR_EXACT_TABLE_COVER)
            self.assertLessEqual(sum(ne["dist"]), fa.NEAR_EXACT_TABLE_DIST)
        st = fl.spot_setup(EIGHT_UTG_UTG1, 8)
        ne = fa.near_exact(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], 8)
        self.assertGreater(sum(ne["dist"]), fa.NEAR_EXACT_DIST)        # (UTG1's cold calls: 0.73 from the BTN's)
        st = fl.spot_setup(EIGHT_HJ_BTN, 8)
        ne = fa.near_exact(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], 8)
        self.assertLess(min(ne["cover"]), fa.NEAR_EXACT_COVER)        # (the BTN's flats: 8.5% off CO's range)
        self.assertIsNone(fa.near_exact(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], 5))

    def test_which_spot(self):
        """Under NEAR_COVER: the 8-max CO's open flatted by the BTN off HJ vs CO, the LJ's flatted by the HJ off UTG vs
        HJ, the BTN's 3bet of UTG off the CO's of the HJ, 9-max's CO vs BTN too (tier 1); none at 6 players, none for
        UTG flatted by the HJ (cover 0.71)."""
        cases = [(EIGHT_CO_BTN, 8, "srp_hj_co"),
                 ("UTG:f,UTG1:f,LJ:r2.5,HJ:c,CO:f,BTN:f,SB:f,BB:f", 8, "srp_utg_hj"),
                 ("UTG:r2.5,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r7.5,SB:f,BB:f,UTG:c", 8, "3bp_co_hj"),
                 ("UTG:f,UTG1:f,UTG2:f,LJ:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:f", 9, "srp_hj_co"),
                 ("UTG:r2.5,UTG1:f,LJ:f,HJ:c,CO:f,BTN:f,SB:f,BB:f", 8, None)]
        for line, n, sid in cases:
            st = fl.spot_setup(line, n)
            args = (self.lib, st["full0"], st["full1"], st["pot"], st["stack"])
            sp = fa.near_spot(*args, players=n)
            self.assertEqual(sp and sp["id"], sid, line)
            self.assertIsNone(fa.near_spot(*args, players=6), line)
            self.assertIsNone(fa.near_exact(*args, n), line)
            if sid is not None:
                rows = {r["id"]: r for r in fa.near_candidates(*args, players=n)}
                self.assertEqual(rows[sid]["tier"], 1, line)
                self.assertLess(rows[sid]["cover"], fa.NEAR_COVER, line)
                self.assertGreaterEqual(rows[sid]["cover"], fa.NEAR_TABLE_COVER, line)

    def test_instant(self):
        """8-max CO vs BTN, the CO first to act: a hand HJ vs CO's HJ range holds gets that spot's flops' mix at once,
        approximately, nothing solved; one it doesn't (the 8-max CO opens wider) has no answer at once (solved live)."""
        st = fl.spot_setup(EIGHT_CO_BTN, 8)
        sp = next(s for s in self.lib.index()["spots"] if s["id"] == "srp_hj_co")
        ss = fl.spot_setup(sp["line"])
        board = BOARD
        dead = set(parse_cards(board))
        off = next(i for i, cb in enumerate(fl.ALL_COMBOS) if st["full0"][i] > 0.5 and ss["full0"][i] == 0 and
                   not dead & set(cb))
        hand = {"players": 8, "stack": 100, "preflop": parse_actions(EIGHT_CO_BTN), "board": board, "turn": [],
                "river": [], "hero": "CO", "flop": []}
        adv = advisor()
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            held = adv.instant(dict(hand, hole="AhKd"))
        self.assertEqual(held["approximate"]["source"], "near")
        self.assertIn("the library's HJ vs CO single-raised pot", held["approximate"]["label"])
        self.assertIn("the spot whose ranges are nearest this pot's", held["approximate"]["label"])
        self.assertNotIn("your hand isn't in that spot's range", held["approximate"]["label"])
        self.assertNotIn("precomputed", held)
        names, mix = self.expected(board, "AhKd", [], spot=sp)
        self.assertEqual([a["action"] for a in held["actions"]], names)
        self.assertTrue(np.allclose([a["freq"] for a in held["actions"]], mix, atol=0.0006))
        self.assertIsNone(adv.instant(dict(hand, hole=cards_str(fl.ALL_COMBOS[off]))))


STRADDLE_BTN = "HJ:f,CO:f,BTN:r5,SB:f,BB:f,UTG:c"     # (the straddler calls the BTN's raise: an 11.5 bb pot)
STRADDLE_SB = "HJ:f,CO:f,BTN:f,SB:r6,BB:f,UTG:c"      # (the straddler calls the SB's raise: 13 bb, 94 behind)


class TestOtherGame(_Lib):
    """A pot of another game (a straddle, antes, heads-up blinds: another pot size) is read off the library spot whose
    ranges are nearest, of the same stack-to-pot ratio band, where the same seat raised last preflop (near_spot's
    aggr), within NEAR_POT_DIST; raises named by their size in bb are matched across pots (flop_approx.norm_name)."""

    def test_which_spot(self):
        st = fl.spot_setup(STRADDLE_BTN, 6, straddle=True)
        self.assertEqual((st["oop"], st["ip"]), ("UTG", "BTN"))
        sp = fa.near_spot(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], aggr=1)
        self.assertEqual(sp and sp["id"], "srp_btn_bb")
        for aggr in (None, 0):          # (the raiser not known, or out of position: no other pot's spot)
            self.assertIsNone(fa.near_spot(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], aggr=aggr))
        rows = {r["id"]: r for r in fa.near_candidates(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], aggr=1)}
        self.assertIsNone(rows["srp_btn_bb"]["why"])
        self.assertEqual(rows["srp_sb_bb"]["why"], "pot")              # (the SB raised there: out of position)
        self.assertEqual(rows["3bp_bb_btn"]["why"], "pot")
        # 6-max 100bb's own lines are as they were: a squeeze pot isn't read off a 3bet pot of another size
        sq = fl.spot_setup("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:r10,BB:c,BTN:f")
        self.assertIsNone(fa.near_spot(self.lib, sq["full0"], sq["full1"], sq["pot"], sq["stack"], aggr=0))

    def test_tiers(self):
        """Where no spot covers each range by NEAR_POT_COVER, a nearer one covering less (tier 1: the ante's SB vs BB
        off SB vs BB, the straddler's call of the BB's raise off it too), else a single-raised pot the BB defends at
        about the same stack-to-pot ratio (tier 2: heads-up's raised pot and the ante's BTN vs BB off BTN vs BB, the
        ante's SB flatting the CO off CO vs BB and the HJ off HJ vs BB, 1.44 and 1.50 apart), else such a pot at
        another stack-to-pot ratio, read for the hands held alone (tier 3: a straddled SB vs UTG off SB vs BB, at 50bb
        with an ante the BB's defence of the CO off BTN vs BB, not the SB vs BB, whose raiser's range is 83% held); a
        lower tier wins over a nearer spot (the ante's UTG vs BB stays on HJ vs BB, not UTG vs BB's tier 2); outside
        them nothing (the ante's UTG vs HJ, 0.53 apart; heads-up's 3bet pot)."""
        cases = [("SB:r3,BB:c", 2, {}, "srp_btn_bb", 2),
                 ("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", 6, {"ante": 0.125}, "srp_btn_bb", 2),
                 ("UTG:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:f", 6, {"ante": 0.125}, "srp_co_bb", 2),
                 ("UTG:f,HJ:r2.5,CO:f,BTN:f,SB:c,BB:f", 6, {"ante": 0.125}, "srp_hj_bb", 2),
                 ("UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:c", 6, {"ante": 0.125}, "srp_sb_bb", 1),
                 ("HJ:f,CO:f,BTN:f,SB:f,BB:r5,UTG:c", 6, {"straddle": True}, "srp_sb_bb", 1),
                 (STRADDLE_SB, 6, {"straddle": True}, "srp_sb_bb", 3),
                 ("UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c", 6, {"ante": 0.125, "stack": 50}, "srp_btn_bb", 3),
                 ("UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:c", 6, {"ante": 0.125, "stack": 75}, None, None),
                 ("UTG:r2.5,HJ:f,CO:f,BTN:f,SB:f,BB:c", 6, {"ante": 0.125}, "srp_hj_bb", 0),
                 ("UTG:r2.5,HJ:c,CO:f,BTN:f,SB:f,BB:f", 6, {"ante": 0.125}, None, None),
                 ("SB:r3,BB:r10,SB:c", 2, {}, None, None)]
        for line, n, game, sid, tier in cases:
            st = fl.spot_setup(line, n, **game)
            last = [t.split(":")[0] for t in line.split(",") if t.split(":")[1].startswith("r")][-1]
            aggr = (st["oop"], st["ip"]).index(last)
            sp = fa.near_spot(self.lib, st["full0"], st["full1"], st["pot"], st["stack"], aggr=aggr)
            self.assertEqual(sp and sp["id"], sid, line)
            if sid is not None:
                rows = {r["id"]: r for r in fa.near_candidates(self.lib, st["full0"], st["full1"], st["pot"],
                                                               st["stack"], aggr=aggr)}
                self.assertEqual(rows[sid]["tier"], tier, line)
        rows = {r["id"]: r for r in fa.near_candidates(self.lib, st["full0"], st["full1"], st["pot"], st["stack"],
                                                       aggr=0)}
        self.assertEqual(rows["3bp_bb_btn"]["why"], "cover")          # (heads-up's 3bet pot: half its range unheld)

    def test_held_only(self):
        """Tier 3's read answers the hands its spot's ranges hold alone (near_fill): the straddled SB's raise called by
        the straddler, off SB vs BB (whose big blind holds 84% of the straddler's range) at less than half its stack-to-
        pot ratio. The SB gets the entry's mix at once, nothing solved; the straddler, facing the SB's bet with a hand
        the big blind's range there doesn't hold (a suited ace it 3bets), has no answer at once (solved live)."""
        st = fl.spot_setup(STRADDLE_SB, 6, straddle=True)
        self.assertEqual((st["oop"], st["ip"]), ("SB", "UTG"))
        args = (self.lib, st["full0"], st["full1"], st["pot"], st["stack"])
        self.assertEqual(fa.near_spot(*args, aggr=0)["id"], "srp_sb_bb")
        self.assertFalse(fa.near_fill(*args, aggr=0))
        ante = fl.spot_setup("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", 6, ante=0.125)   # (tier 2: filled)
        self.assertTrue(fa.near_fill(self.lib, ante["full0"], ante["full1"], ante["pot"], ante["stack"], aggr=1))
        sp = next(s for s in self.lib.index()["spots"] if s["id"] == "srp_sb_bb")
        board = "KcJd2h"
        self.assertIn(board, sp["flops"])
        sol = fa.approx_solution(*args[:3], parse_cards(board), st["full0"], st["full1"], st["pot"], st["stack"],
                                 fl.library_config(st["pot"], st["stack"]), aggr=0)
        self.assertFalse(any(f.any() for f in sol.filled.values()))
        known = sol.known[id(sol.root)]
        self.assertTrue(0.7 < known.mean() < 0.95)                      # (the straddler's hands the BB doesn't hold)
        hand = {"players": 6, "stack": 100, "straddle": True, "preflop": parse_actions(STRADDLE_SB), "board": board,
                "turn": [], "river": []}
        ss = fl.spot_setup(sp["line"])
        dead = set(parse_cards(board))
        off = next(i for i, cb in enumerate(fl.ALL_COMBOS) if st["full1"][i] > 0.5 and ss["full1"][i] == 0 and
                   not dead & set(cb))
        on = next(i for i, cb in enumerate(fl.ALL_COMBOS) if st["full1"][i] > 0.5 and ss["full1"][i] > 0.5 and
                  not dead & set(cb))
        adv = advisor()
        bet = [B("SB", round(0.33 * st["pot"], 2))]
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            res = adv.instant(dict(hand, hero="SB", hole="QhTh", flop=[]))
            held = adv.instant(dict(hand, hero="UTG", hole=cards_str(fl.ALL_COMBOS[on]), flop=bet))
        self.assertIn("another game's pot (6 bb, a stack-to-pot ratio of 16.2; this one 13 bb, 7.2)",
                      res["approximate"]["label"])
        names, mix = self.expected(board, "QhTh", [], spot=sp)
        self.assertEqual([a["action"] for a in res["actions"]], names)
        self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0006))
        self.assertNotIn("your hand isn't in that spot's range", held["approximate"]["label"])
        self.assertIsNone(adv.instant(dict(hand, hero="UTG", hole=cards_str(fl.ALL_COMBOS[off]), flop=bet)))

    def test_heads_up(self):
        """Heads-up's raised pot reads BTN vs BB's entry of the flop at once, approximately: a hand that spot's BB
        range holds gets its own mix there, one it doesn't (the heads-up big blind defends wider) its hand class's."""
        sp = next(s for s in self.lib.index()["spots"] if s["id"] == "srp_btn_bb")
        board = "KhJd5c"
        hand = {"players": 2, "stack": 100, "preflop": parse_actions("SB:r3,BB:c"), "board": board, "turn": [],
                "river": [], "hero": "BB", "flop": []}
        adv = advisor()
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            held = adv.instant(dict(hand, hole="Qh9h"))
            other = adv.instant(dict(hand, hole="7h2c"))
        self.assertIn("another game's pot (5.5 bb, a stack-to-pot ratio of 17.7; this one 6 bb, 16.2)",
                      held["approximate"]["label"])
        self.assertNotIn("your hand isn't in that spot's range", held["approximate"]["label"])
        names, mix = self.expected(board, "Qh9h", [], spot=sp)
        self.assertEqual([a["action"] for a in held["actions"]], names)
        self.assertTrue(np.allclose([a["freq"] for a in held["actions"]], mix, atol=0.0006))
        self.assertIn("your hand isn't in that spot's range: the average mix there of its hand class",
                      other["approximate"]["label"])
        self.assertAlmostEqual(sum(a["freq"] for a in other["actions"]), 1.0, places=3)

    def test_instant(self):
        """The straddler first to act: the entry of the flop in BTN vs BB, another game's pot named; the BTN's c-bet
        check-raised: the entry's node facing its raise (named in bb at its own pot)."""
        sp = next(s for s in self.lib.index()["spots"] if s["id"] == "srp_btn_bb")
        board = "KhJd5c"
        self.assertIn(fl.cards_str(fl.canonical(parse_cards(board))[0]), sp["flops"])
        hand = {"players": 6, "stack": 100, "straddle": True, "preflop": parse_actions(STRADDLE_BTN), "board": board,
                "turn": [], "river": []}
        adv = advisor()
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            res = adv.instant(dict(hand, hero="UTG", hole="Qh9h", flop=[]))
        label = res["approximate"]["label"]
        self.assertIn("the precomputed solve of this flop in the library's BTN vs BB single-raised pot", label)
        self.assertIn("another game's pot (5.5 bb, a stack-to-pot ratio of 17.7; this one 11.5 bb, 8.3)", label)
        names, mix = self.expected(board, "Qh9h", [], spot=sp)
        self.assertEqual([a["action"] for a in res["actions"]], names)
        self.assertTrue(np.allclose([a["freq"] for a in res["actions"]], mix, atol=0.0006))
        # the straddler checks and faces the c-bet: the entry's node, whose raise is named at its own pot's size
        e = self.lib.entry(sp, fl.cards_str(fl.canonical(parse_cards(board))[0]))
        nd = next(nd for nd in e["nodes"] if nd["path"] == ["check", "bet 33%"])
        st = fl.spot_setup(STRADDLE_BTN, 6, straddle=True)
        sol = fa.approx_solution(self.lib, st["full0"], st["full1"], parse_cards(board), st["full0"], st["full1"],
                                 st["pot"], st["stack"], fl.library_config(st["pot"], st["stack"]), aggr=1)
        node = fa.node_norm(sol.root, nd["path"])
        self.assertIn(id(node), sol.stored)
        raise_here = next(x for x in node.names if x.startswith("raise to"))
        self.assertNotIn(raise_here, nd["names"])                      # (another size in bb)
        self.assertEqual([fa.norm_name(x) for x in node.names], [fa.norm_name(x) for x in nd["names"]])
        self.assertEqual(fa.norm_name("bet 33%"), "bet 33%")
