"""Turn and river reports (Study's report at a closed flop or turn: Advisor._runouts, gto/runouts.py): each next
card's equity and chance, the report from a library flop's stored turns and from a live turn solve's rivers, held
(from what's in hand) or solved, matching what Study's own answers give."""
import gc
import itertools
import json
import os
import unittest

import numpy as np

from gto import flop_library as fl
from gto.cards import ALL_COMBOS, COMBO_INDEX, card_str, cards
from gto.evaluator import eval_boards
from gto.native import advisor
from gto.runouts import next_card_equities
from gto.solver import MEMBER_FULL

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
LIB = os.path.join(os.path.dirname(__file__), "..", "gto", "static", "flop_library")


def brute(board, w0, w1):
    """next_card_equities by hand: every pair of hands sharing no card with each other or the board, on every
    runout neither blocks."""
    dead = set(board)
    live = [i for i, cb in enumerate(ALL_COMBOS) if not set(cb) & dead and (w0[i] > 0 or w1[i] > 0)]
    out = {}
    for c in [c for c in range(52) if c not in dead]:
        rest = [[]] if len(board) == 4 else [[r] for r in range(52) if r not in dead and r != c]
        num = den = 0.0
        for more in rest:
            b = list(board) + [c] + more
            ok = [i for i in live if not set(ALL_COMBOS[i]) & set(b)]
            s = dict(zip(ok, eval_boards([b], np.array(ok))[0]))
            for i, j in itertools.product(ok, ok):
                if set(ALL_COMBOS[i]) & set(ALL_COMBOS[j]):
                    continue
                w = w0[i] * w1[j]
                den += w
                num += w * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
        out[c] = (num, den)
    return out


class TestEquities(unittest.TestCase):
    def test_against_brute_force(self):
        """Each card's equity (ties half, card removal between the two hands and the runout) and chance (the share
        of the hand pairs that leave it live), a river's and a turn's (every river under each turn card)."""
        rng = np.random.default_rng(3)
        for board, k in (("Kd7c2h5s", 60), ("9c6d6c", 14)):
            b = cards(board)
            w0, w1 = np.zeros(1326), np.zeros(1326)
            w0[rng.choice(1326, k, replace=False)] = rng.random(k)
            w1[rng.choice(1326, k, replace=False)] = rng.random(k)
            got_cards, eq, p = next_card_equities(b, w0, w1)
            want = brute(b, w0, w1)
            self.assertEqual(sorted(got_cards), sorted(want))
            dens = np.array([want[c][1] for c in got_cards])
            self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
            np.testing.assert_allclose(p, dens / dens.sum(), atol=1e-9)
            for c, e in zip(got_cards, eq):
                num, den = want[c]
                if den > 0:
                    self.assertAlmostEqual(float(e), num / den, places=9)
                else:
                    self.assertTrue(np.isnan(e))

    def test_no_pairs(self):
        """No pair of hands left (one range only blocks the other): NaN equity, no chance."""
        w0, w1 = np.zeros(1326), np.zeros(1326)
        w0[COMBO_INDEX[tuple(sorted(cards("AsAh")))]] = 1.0
        w1[COMBO_INDEX[tuple(sorted(cards("AsAd")))]] = 1.0
        _, eq, p = next_card_equities(cards("Kd7c2h5s"), w0, w1)
        self.assertTrue(np.isnan(eq).all())
        self.assertEqual(float(p.sum()), 0.0)


def reach(node, action):
    """(the acting seat, its reach after `action` per combo) from a Study node answer's combos: each combo's reach
    there (w) times its mix's share for the action."""
    a = node["grid"]["actions"].index(action)
    r = np.zeros(1326)
    for rows in node["combos"].values():
        for cb, w, f, *_ in rows:
            r[COMBO_INDEX[tuple(sorted(cards(cb)))]] = w * f[a]
    return node["actor"], r


def reach_at(node):
    """The acting seat's reach at a Study node, per combo."""
    r = np.zeros(1326)
    for rows in node["combos"].values():
        for cb, w, *_ in rows:
            r[COMBO_INDEX[tuple(sorted(cards(cb)))]] = w
    return r


class TestReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = advisor("fast")

    @classmethod
    def tearDownClass(cls):
        # (its cached solves: the suite runs in one process, where a class's advisor holds them to the end)
        del cls.adv
        gc.collect()

    def study(self, **st):
        return self.adv.advise(dict({"players": 6, "stack": 100, "study": True, "budget": 1.0}, **st))

    def check_cards(self, r, board, n):
        got = [c["card"] for c in r["cards"]]
        self.assertEqual(len(got), n)
        self.assertFalse(set(got) & {board[i:i + 2] for i in range(0, len(board), 2)})
        self.assertAlmostEqual(sum(c["p"] for c in r["cards"]), 1.0, delta=0.001)
        for c in r["cards"]:
            self.assertAlmostEqual(sum(c["eq"]), 1.0, delta=0.0002)
        for part in (r["first"], r["after_check"]):
            self.assertEqual(sorted(x for k in part["cards"] for x in k["cards"]), sorted(got))
            for k in part["cards"]:
                self.assertAlmostEqual(sum(k["f"]), 1.0, delta=0.01)
                self.assertGreaterEqual(k["w"], 0.0)

    def test_stored_turns(self):
        """A library flop whose entry has stored turns (3bet pot CO vs HJ, 9c6d6c): its turn report at once from
        the turn file, held or not, on the lines it holds: per turn card the stored strategies over the range Study's
        own flop answers give, each card's equity between the two ranges; no EVs (strategies only). A line it
        doesn't hold: skipped (held, with the lines it does) or solved live."""
        with open(os.path.join(LIB, "index.json")) as f:
            spot = next(s for s in json.load(f)["spots"] if s["id"] == "3bp_co_hj")
        pre = [{"pos": p, "action": "raise", "to": float(a[1:])} if a[0] == "r" else
               {"pos": p, "action": {"f": "fold", "c": "call", "x": "check"}[a[0]]}
               for p, a in (t.split(":") for t in spot["line"].split(","))]
        board = "9c6d6c"
        r = self.study(preflop=pre, board=board, flop=[{"pos": "HJ", "action": "check"}, {"pos": "CO", "action": "check"}],
                       runouts="held")
        self.assertEqual((r["study"], r["street"], r["skipped"]), ("runouts", "turn", "not solved yet"))
        paths = [ln["path"] for ln in r["lines"]]
        self.assertIn(["check", "bet 75%", "call"], paths)
        ln = r["lines"][paths.index(["check", "bet 75%", "call"])]
        self.assertEqual([(a["pos"], a["action"]) for a in ln["flop"]], [("HJ", "check"), ("CO", "bet"), ("HJ", "call")])
        self.assertGreater(ln["share"], 0.05)
        for held in ("held", True):
            r = self.study(preflop=pre, board=board, flop=ln["flop"], runouts=held)
            self.assertEqual((r["source"], r["street"], r["seats"], r["lines"][r["line"]]["path"]),
                             ("store", "turn", ["HJ", "CO"], ["check", "bet 75%", "call"]))
            self.assertLess(r["seconds"], 1.0)
        self.assertEqual((r["first"]["actor"], r["after_check"]["actor"]), ("HJ", "CO"))
        self.check_cards(r, board, 49)
        self.assertTrue(all(c["ev"] is None for c in r["cards"]))
        # the ranges entering the turn: Study's answers at CO's bet and HJ's call
        seat_b, r_co = reach(self.study(preflop=pre, board=board, flop=ln["flop"][:1]), "bet 75%")
        seat_c, r_hj = reach(self.study(preflop=pre, board=board, flop=ln["flop"][:2]), "call")
        self.assertEqual((seat_b, seat_c), ("CO", "HJ"))
        cs, eq, p = next_card_equities(cards(board), r_hj, r_co)
        by = {c["card"]: c for c in r["cards"]}
        for c, e, q in zip(cs, eq, p):
            self.assertAlmostEqual(by[card_str(c)]["eq"][0], float(e), delta=0.005)
            self.assertAlmostEqual(by[card_str(c)]["p"], float(q), delta=0.0005)
        # the first decision's mix per card: the file's strategies over HJ's range there
        t = self.adv.library.turns(spot, fl.cards_str(fl.canonical(cards(board))[0]))
        smap = fl.canonical(cards(board))[1]
        for row in r["first"]["cards"]:
            c = cards(row["card"])[0]
            st = fl.turn_strategies(t, ln["path"], (), c, smap)
            w = r_hj * (1.0 - MEMBER_FULL[c]) * ~np.isnan(st["strat"][0])
            want = (np.nan_to_num(st["strat"]) * w).sum(axis=1) / w.sum()
            np.testing.assert_allclose(row["f"], want, atol=0.02, err_msg=row["card"])

    def test_turn_solve_rivers(self):
        """A closed turn's river report, read off the turn's live solve: held, nothing until Study's turn answer
        solved it, then at once; per river card each player's EV (range against range: the two add up to the pot)
        and equity between the ranges Study's turn answers give."""
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "to": 1.8}, {"pos": "BB", "action": "call"}]
        board = "Qh8h4c9d"
        st = dict(preflop=PRE, board=board, flop=flop)
        check = [{"pos": "BB", "action": "check"}]
        r = self.study(**st, turn=check + [{"pos": "BTN", "action": "bet", "to": 3.0}, {"pos": "BB", "action": "call"}],
                       runouts="held")
        self.assertEqual((r["study"], r["street"], r["skipped"]), ("runouts", "river", "not solved yet"))
        self.assertNotIn("lines", r)
        b_node = self.study(**st, turn=check)    # (solves the turn: Study's answer, as the page asks it)
        bet = next(a for a in b_node["actions"] if a["action"].startswith("bet"))
        turn = check + [{"pos": "BTN", "action": "bet", "to": bet["amount_bb"]}, {"pos": "BB", "action": "call"}]
        c_node = self.study(**st, turn=turn[:2])
        r = self.study(**st, turn=turn, runouts="held")
        self.assertEqual((r["source"], r["street"], r["seats"]), ("solve", "river", ["BB", "BTN"]))
        self.assertLess(r["seconds"], 1.0)
        self.check_cards(r, board, 48)
        for c in r["cards"]:
            self.assertAlmostEqual(sum(c["ev"]), r["pot_bb"], delta=0.03)
            self.assertTrue(0.0 <= min(c["ev"]) <= r["pot_bb"])
        # equities between Study's ranges entering the river (BTN's: its answer's bet share, which the walk to BB's
        # call smooths a little where the solve rarely bets: Advisor._walk), both off the solve as the report reads it:
        # BTN's answer asked again, as BB's ran the turn's solve on (the flop's cached by then, the turn gets the time
        # its narrowing didn't spend: Advisor.spare_prior; on a loaded machine 81 iterations, then 325, and the first
        # answer's range was 0.012-0.022 off in equity where the solve's own is 0.0001)
        b_node = self.study(**st, turn=check)
        self.assertEqual(b_node["solver"]["iterations"], c_node["solver"]["iterations"])
        _, r_btn = reach(b_node, bet["action"])
        _, r_bb = reach(c_node, "call")
        cs, eq, p = next_card_equities(cards(board), r_bb, r_btn)
        by = {c["card"]: c for c in r["cards"]}
        diff = [abs(by[card_str(c)]["eq"][0] - float(e)) for c, e in zip(cs, eq)]
        self.assertLess(max(diff), 0.01)
        self.assertLess(float(np.mean(diff)), 0.004)
        evs = [c["ev"][0] for c in r["cards"]]
        self.assertGreater(max(evs) - min(evs), 0.5)   # (a river's EVs follow the card)
        # the same report solved (the cached solve: the same numbers)
        s = self.study(**st, turn=turn, runouts=True)
        self.assertEqual(s["first"], r["first"])
        self.assertEqual(s["cards"], r["cards"])


if __name__ == "__main__":
    unittest.main()
