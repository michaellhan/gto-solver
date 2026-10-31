"""Study on a library flop's turn answers at once from the flop's stored turn (Advisor.instant with "study":
Advisor._study_store) while the turn's own solve runs: the whole range's mix per hand class and combo, the range's
frequency per action, the opponent's range, marked approximate; a format-2 turn file adds its EVs (each action's and
the range's), a format-1 file none; no hand's own EVs or equities. Nothing is solved, and Study's own answer (the
turn's live solve) is as before."""
import gc
import json
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from gto import advisor as A
from gto import flop_library as fl
from gto.cards import cards as parse_cards, ALL_COMBOS, COMBO_INDEX
from tests.test_instant_store import SPOT, CANON, BOARD, X, B, C, advisor, build, _flop_solve
from gto.cli import parse_actions


class TestStudyStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sid, _, line = SPOT
        s, sol = _flop_solve()
        cls.setup = s
        meta = dict(build._meta(sid, s), board=CANON)
        cls.dirs, cls.libs = [], {}
        for name, turns, evs in (("f1", True, False), ("f2", True, True), ("none", False, False)):
            d = tempfile.mkdtemp()
            build.write_entry(d, sid, CANON, sol, meta, with_turns=turns, turn_evs=evs)
            build.write_index(d, [SPOT])
            cls.dirs.append(d)
            cls.libs[name] = fl.FlopLibrary.from_dir(d)
        cls.spot = cls.libs["f1"].index()["spots"][0]
        cls.smap = fl.canonical(parse_cards(BOARD))[1]
        oop, ip = s["oop"], s["ip"]
        cls.oop, cls.ip = oop, ip
        cls.hand = {"players": 6, "stack": 100, "preflop": parse_actions(line), "flop": [X(oop), X(ip)], "turn": [],
                    "river": [], "board": BOARD + "9c", "speed": "fast"}

    @classmethod
    def tearDownClass(cls):
        for d in cls.dirs:
            shutil.rmtree(d, ignore_errors=True)
        cls.libs = None
        gc.collect()

    def quick(self, lib, **kw):
        with mock.patch.object(A, "solve", side_effect=AssertionError("solved")):
            return advisor(self.libs[lib]).instant(dict(self.hand, study=True, **kw))

    def canon_combo(self, combo: str) -> int:
        return COMBO_INDEX[tuple(sorted(c // 4 * 4 + self.smap[c % 4] for c in parse_cards(combo)))]

    def test_the_whole_range_at_once(self):
        """The turn's first decision and in position after a check: every hand the stored turn holds, each combo's
        stored mix (through the suit map), the grid's cells their combos' mix by weight, the range's frequencies,
        the opponent's range (after a check: the hands the stored first decision checks), the bet at the hand's own
        pot; marked approximate, a format-1 file's strategies only."""
        t = self.libs["f1"].turns(self.spot, CANON)
        card = parse_cards("9c")[0]
        for hero, at, path in ((self.oop, [], ()), (self.ip, [X(self.oop)], ("check",))):
            res = self.quick("f1", turn=at)
            self.assertEqual((res["study"], res["street"], res["actor"]), ("node", "turn", hero))
            self.assertEqual(res["approximate"]["source"], "store")
            self.assertIn("precomputed flop solve's view of the turn (by card class, one bet size, 75%)", res["approximate"]["label"])
            self.assertEqual(res["solver"]["by"], "the flop library")
            nd = fl.turn_node(t, ["check", "check"], list(path), card // 4 * 4 + self.smap[card % 4])
            self.assertEqual([a["action"] for a in res["actions"]], nd["names"])
            self.assertAlmostEqual(sum(a["freq"] for a in res["actions"]), 1.0, places=2)
            self.assertAlmostEqual(sum(a["combos"] for a in res["actions"]), res["range_combos"], delta=0.2)
            bet = next(a for a in res["actions"] if a["action"] == "bet 75%")
            self.assertAlmostEqual(bet["amount_bb"], round(0.75 * res["pot_bb"], 2), places=2)
            # each combo's mix is the stored one; each cell its combos' by weight
            n = 0
            for cls, rows in res["combos"].items():
                cell = res["grid"]["cells"][cls]
                w = sum(r[1] for r in rows)
                for a in range(len(nd["names"])):
                    self.assertAlmostEqual(sum(r[1] * r[2][a] for r in rows) / w, cell["f"][a], delta=0.01)
                for r in rows:
                    j = int(np.searchsorted(nd["combos"], self.canon_combo(r[0])))
                    self.assertEqual(nd["combos"][j], self.canon_combo(r[0]))
                    self.assertTrue(np.allclose(r[2], nd["strat"][:, j], atol=0.0015), (r, nd["strat"][:, j]))
                    self.assertIsNone(r[3])
                    self.assertIsNone(r[4])
                    n += 1
                self.assertEqual((cell["ev"], cell["evm"], cell["eq"]), (None, None, None))
            self.assertGreater(n, 50)
            self.assertEqual([a["ev_bb"] for a in res["actions"]], [None] * len(nd["names"]))
            self.assertIsNone(res["range_ev_bb"])
            self.assertIsNone(res["range_eq"])
            self.assertIsNone(res["opponent_eq"])
            self.assertTrue(any("strategies only" in x for x in res["notes"]))
            self.assertEqual(json.loads(json.dumps(res)), res)
        # after the check the big blind's range is the hands its stored first decision checks
        first, after = self.quick("f1"), self.quick("f1", turn=[X(self.oop)])
        self.assertAlmostEqual(after["opponent_combos"], first["actions"][0]["combos"], delta=0.05 * first["range_combos"])
        self.assertLess(after["opponent_combos"], first["range_combos"])

    def test_format_2_evs(self):
        """A format-2 turn file: each action's EV (the hands of the range that take it, the turn class's) and the
        range's (at the first decision the file's range EV, after a check the actions' by frequency), scaled from
        the stored line's pot to the hand's; the same mix as format 1."""
        t = self.libs["f2"].turns(self.spot, CANON)
        card = parse_cards("9c")[0]
        cc = card // 4 * 4 + self.smap[card % 4]
        ln = next(x for x in t["lines"] if x["path"] == ["check", "check"])
        for at, path, p in (([], (), 0), ([X(self.oop)], ("check",), 1)):
            res, f1 = self.quick("f2", turn=at), self.quick("f1", turn=at)
            self.assertEqual([a["freq"] for a in res["actions"]], [a["freq"] for a in f1["actions"]])
            nd = fl.turn_node(t, ["check", "check"], list(path), cc)
            k = res["pot_bb"] / ln["pot"]
            for a, e in zip(res["actions"], nd["evs"]):
                if np.isfinite(e):
                    self.assertAlmostEqual(a["ev_bb"], round(float(e) * k, 2), places=2)
                else:
                    self.assertIsNone(a["ev_bb"])
            if not at:
                self.assertAlmostEqual(res["range_ev_bb"], round(float(fl.turn_range_evs(t, ["check", "check"], cc)[p]) * k, 2))
            else:
                w = [(a["freq"], a["ev_bb"]) for a in res["actions"] if a["ev_bb"] is not None and a["freq"] > 0]
                self.assertAlmostEqual(res["range_ev_bb"], sum(f * e for f, e in w) / sum(f for f, _ in w), places=1)
            self.assertTrue(any("no hand's own" in x for x in res["notes"]))
            self.assertTrue(all(r[3] is None for rows in res["combos"].values() for r in rows))

    def test_none_where_it_isnt_stored(self):
        """Facing a turn bet, an entry without a turn file, the river, the flop, preflop, a tournament: none."""
        self.assertIsNotNone(self.quick("f1"))
        self.assertIsNone(self.quick("f1", turn=[B(self.oop, 3)]))
        self.assertIsNone(self.quick("none"))
        self.assertIsNone(self.quick("f1", turn=[X(self.oop), X(self.ip)], river=[], board=BOARD + "9c3s"))
        self.assertIsNone(self.quick("f1", flop=[], turn=[], board=BOARD))
        self.assertIsNone(self.quick("f1", flop=[], turn=[], board=""))
        self.assertIsNone(self.quick("f1", payouts=[50, 30, 20]))
        # cards of anyone's own and the advice's settings are left out, as Study leaves them
        self.assertEqual(self.quick("f1", hero=self.oop, hole="AsAh", locks={}, opponents="gto")["range_combos"],
                         self.quick("f1")["range_combos"])
        # and without "study" the advice's own instant answer (needs hero's cards) is unchanged
        self.assertIsNone(advisor(self.libs["f1"]).instant(dict(self.hand)))

    def test_the_turns_own_solve_answers_study(self):
        """Nothing solved or cached by the answer at once; Study's own answer is the turn's live solve, with every
        hand's EVs and equities, and its sizes."""
        adv = advisor(self.libs["f1"])
        n = len(adv._cache)
        self.assertEqual(adv.instant(dict(self.hand, study=True))["approximate"]["source"], "store")
        self.assertEqual(len(adv._cache), n)
        res = adv.study(dict(self.hand))
        self.assertNotIn("approximate", res)
        self.assertNotIn("by", res["solver"])
        self.assertIn("bet 33%", [a["action"] for a in res["actions"]])
        self.assertIsNotNone(res["range_eq"])
        self.assertTrue(all(r[3] is not None for rows in res["combos"].values() for r in rows))
        # once it's in hand (settled), nothing at once: the request is quick anyway
        self.assertIsNone(adv.instant(dict(self.hand, study=True)))

    def test_turn_files_fetched_ahead_from_the_flop(self):
        """Study at a library flop's decision, answered (its entry read): the prefetch's dry run for the flop (as the
        browser's worker runs it, files recorded rather than read) asks for the entry's turn file, and keeps nothing
        as missing; before the flop's answer it has no entry to go by (no hero to run the advice for)."""
        real = self.libs["f1"]._fetch
        asked, rec = [], [False]

        def fetch(p):
            if rec[0] and not p.startswith("index"):
                asked.append(p)
                return None
            return real(p)
        lib = fl.FlopLibrary(fetch)
        adv = advisor(lib)
        flop = dict(self.hand, flop=[], turn=[], board=BOARD)
        rec[0] = True
        adv.prefetch(dict(flop, study=True))
        self.assertEqual(asked, [])
        rec[0] = False
        self.assertIn("precomputed", adv.study(flop))
        rec[0] = True
        adv.prefetch(dict(flop, study=True))
        h = self.spot["flops"][CANON]
        self.assertEqual(asked, [f"{SPOT[0]}/{CANON}.turn.bin?v={h}"])
        self.assertFalse([k for k, e in lib._entries.items() if e is None])


if __name__ == "__main__":
    unittest.main()
