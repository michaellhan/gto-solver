"""Advisor.instant: an approximate answer at once for a heads-up street whose own solve takes seconds, read off what
the hand has solved already (the browser's worker posts it before the request's own answer), and nothing solved."""
import gc
import unittest
from unittest import mock

import numpy as np

from gto import native
from gto.advisor import HU_CHECKS, _swap_combo
from gto.cards import cards as parse_cards, COMBO_INDEX
from gto import advisor as A
from gto.solver import Cancelled, TreeTooBig

F = lambda p: {"pos": p, "action": "fold"}
C = lambda p: {"pos": p, "action": "call"}
X = lambda p: {"pos": p, "action": "check"}
R = lambda p, to: {"pos": p, "action": "raise", "to": to}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}
LIMP = [F("UTG"), F("HJ"), F("CO"), F("BTN"), C("SB"), X("BB")]   # (a limped pot: no flop library, the flop solved live)
FLOP = {"players": 6, "stack": 100, "preflop": LIMP, "flop": [], "turn": [], "river": [], "hero": "SB", "hole": "QhJh",
        "board": "Ks7d2c"}
TURN = dict(FLOP, flop=[X("SB"), B("BB", 1.5), C("SB")], board="Ks7d2cTh")
FACING = dict(TURN, turn=[X("SB"), B("BB", 3)])
RIVER = dict(TURN, turn=[X("SB"), B("BB", 3), C("SB")], board="Ks7d2cTh5s")


def advisor():
    return native.advisor("fast", budget=1, prior_budget=0.5)


def solved(adv):
    return len(adv._cache), sum(s.solver.iters for s in adv._cache.values())


class TestInstant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = advisor()
        cls.adv.advise(FLOP)

    @classmethod
    def tearDownClass(cls):
        # (its cached solves: the suite runs in one process, where a class's advisor holds them to the end)
        del cls.adv
        gc.collect()

    def test_turn_from_the_flop_solve(self):
        """The turn read off the hand's flop solve (its turn by card class and hand buckets, one bet size): the
        mix sums to 1, the amounts are the hand's real pot's, and nothing is solved."""
        before = solved(self.adv)
        res = self.adv.instant(TURN)
        self.assertEqual(solved(self.adv), before)
        self.assertEqual(res["approximate"]["source"], "prev")
        self.assertIn("flop solve's view of the turn", res["approximate"]["label"])
        names = [a["action"] for a in res["actions"]]
        self.assertEqual(names[0], "check")
        self.assertIn("bet 75%", names)
        self.assertAlmostEqual(sum(a["freq"] for a in res["actions"]), 1.0, delta=0.005)
        self.assertEqual(res["pot_bb"], 5.0)
        self.assertEqual(next(a for a in res["actions"] if a["action"] == "bet 75%")["amount_bb"], 3.75)
        self.assertEqual(res["recommended"], max(res["actions"], key=lambda a: a["freq"])["action"])
        self.assertNotIn("solver", res)
        self.assertTrue(all("ev_bb" not in a for a in res["actions"]))

    def test_facing_a_bet_off_the_trees_size(self):
        """Facing a 60% bet the flop's turn has only 75% for: read at its 75%, the call the real one."""
        res = self.adv.instant(FACING)
        self.assertEqual([a["action"] for a in res["actions"]][:2], ["fold", "call"])
        self.assertEqual(res["actions"][1]["amount_bb"], 3.0)
        self.assertEqual(res["to_call_bb"], 3.0)
        raise_ = [a for a in res["actions"] if a["action"].startswith("raise")]
        self.assertTrue(all(a["action"] == f"raise to {a['amount_bb']:g}" and a["amount_bb"] > 6 for a in raise_))

    def test_none_without_a_solve_to_read(self):
        """Before the flop is solved (the narrowing would solve it) there is nothing to read; preflop and the flop
        itself answer quickly anyway."""
        fresh = advisor()
        self.assertIsNone(fresh.instant(TURN))
        self.assertEqual(solved(fresh), (0, 0))
        self.assertIsNone(fresh.instant(dict(FLOP, board="", hole="QhJh")))
        self.assertIsNone(self.adv.instant(FLOP))   # (its solve is cached: the answer is quick)

    def test_none_where_more_than_the_equilibrium_is_asked(self):
        for extra in ({"opponents": "pool"}, {"locks": {"BB": {"fold": 10}}}, {"study": True}, {"jam": True},
                      {"breakdown": True}, {"sizes": {"turn": [50]}}):
            self.assertIsNone(self.adv.instant(dict(TURN, **extra)), extra)

    def test_river_from_the_turn_solve(self):
        """After the turn's advice, the river read off that turn solve (its river by card class, per combo, at
        its river sizes: not the river root's 250%); then the river's own solve, cached, is quick: none."""
        adv = advisor()
        adv.advise(FLOP)
        adv.advise(TURN)
        adv.advise(FACING)
        before = solved(adv)
        res = adv.instant(RIVER)
        self.assertEqual(solved(adv), before)
        self.assertEqual(res["approximate"]["source"], "prev")
        self.assertIn("turn solve's view of the river", res["approximate"]["label"])
        names = [a["action"] for a in res["actions"]]
        self.assertEqual(names[0], "check")
        self.assertNotIn("bet 250%", names)
        full = adv.advise(RIVER)
        self.assertIn("bet 250%", [a["action"] for a in full["actions"]])
        self.assertIsNone(adv.instant(RIVER))

    def stopped_turn(self):
        """An advisor whose turn solve (TURN) the page stopped after HU_CHECKS[0] iterations."""
        adv = native.advisor("fast", budget=60, prior_budget=0.5)   # (stopped at HU_CHECKS[0], not by time)
        adv.advise(dict(FLOP, budget=1))
        adv.hu_settled = 1e-9   # (run in pieces: a check after HU_CHECKS[0] iterations, where it's stopped)
        stop = [False]
        adv.should_stop = lambda: stop[0]
        read = adv._expl

        def expl(sol):
            if sol.ctx.root_len == 4:
                stop[0] = True
            return read(sol)
        adv._expl = expl
        with self.assertRaises(Cancelled):
            adv.advise(TURN)
        return adv

    def test_the_streets_own_solve_stopped_part_way(self):
        """A turn solve stopped part way (the page stopped its request) is read as it stands: source "solve"."""
        adv = self.stopped_turn()
        res = adv.instant(TURN)
        self.assertEqual(res["approximate"]["source"], "solve")
        self.assertFalse(any("smaller game" in n for n in res["notes"]))
        self.assertGreaterEqual(res["approximate"]["iterations"], HU_CHECKS[0])
        self.assertEqual([a["action"] for a in res["actions"]][0], "check")
        self.assertIn("bet 33%", [a["action"] for a in res["actions"]])   # (the turn's own menu)

    def test_a_cut_solve_says_so(self):
        """Read off a turn solve cut to fit (Quick's cap, advisor.TRIMS): the answer says its game is smaller, as the
        solve's own answer does."""
        real, first = A.tree_cells, []

        def too_big(board, *a, **kw):   # (the full turn tree past the cap: solved at the first trim level)
            if len(board) == 4 and not first:
                first.append(1)
                raise TreeTooBig(30_000_000, kw["max_cells"])
            return real(board, *a, **kw)
        with mock.patch.object(A, "tree_cells", too_big):
            adv = self.stopped_turn()
        self.assertTrue(first)
        res = adv.instant(TURN)
        self.assertEqual(res["approximate"]["source"], "solve")
        cut = [n for n in res["notes"] if "smaller game" in n]
        self.assertEqual(len(cut), 1, res["notes"])
        self.assertTrue(cut[0].startswith(f"turn: solved in a smaller game than the full one, to fit this speed's time "
                                          f"and memory: {A.TRIMS[0]}"), cut[0])

    def test_facing_a_size_off_your_advices_tree(self):
        """Their bet after your turn advice off the tree it solved (50% where it has 33/75/125%): that solve, read at its
        nearest size (Advisor._hand_cfg), is the street's own; the answer then solves the turn with the size in."""
        adv = advisor()
        adv.advise(FLOP)
        adv.advise(TURN)
        before = solved(adv)
        res = adv.instant(dict(TURN, turn=[X("SB"), B("BB", 2.5)]))
        self.assertEqual(solved(adv), before)
        self.assertEqual(res["approximate"]["source"], "solve")
        self.assertIn("from your earlier decision", res["approximate"]["label"])
        self.assertEqual([a["action"] for a in res["actions"]][:2], ["fold", "call"])
        self.assertEqual(res["actions"][1]["amount_bb"], 2.5)

    def test_swap(self):
        """A class's played card stands for the card dealt: hero's combo through the swap of the two."""
        h = parse_cards("4hAs")
        self.assertEqual(_swap_combo(h, parse_cards("4h")[0], parse_cards("4c")[0]), COMBO_INDEX[tuple(sorted(parse_cards("4cAs")))])
        self.assertEqual(_swap_combo(h, parse_cards("5h")[0], parse_cards("5c")[0]), COMBO_INDEX[tuple(sorted(h))])


if __name__ == "__main__":
    unittest.main()
