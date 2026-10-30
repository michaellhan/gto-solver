"""Advisor.instant on a library spot's flop that the library doesn't store: its decisions read off the spot's stored
flops most like it (gto/flop_approx.py: same pairing and suit pattern, nearest ranks, card substitution, averaged),
marked approximate (advisor.instant_from_near); the flop's own solve still answers the request. The stored flops it
reads are prefetched once the flop is known (Advisor.prefetch). On the shipped library."""
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

    def expected(self, board, hole, path):
        """The mix instant_from_near should give `hole` at the stored node `path`: its images' stored mixes on the K
        nearest stored flops, averaged with weights 1 / (1 + distance)."""
        b = parse_cards(board)
        canon, smap = fl.canonical(b)
        c = [x // 4 * 4 + smap[x % 4] for x in parse_cards(hole)]
        tot, wsum, names = 0.0, 0.0, None
        for nm, m, d in fa.neighbours(sorted(self.spot["flops"]), list(canon), fa.K):
            e = self.lib.entry(self.spot, nm)
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


if __name__ == "__main__":
    unittest.main()
