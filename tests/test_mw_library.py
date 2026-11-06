"""The precomputed 3-way flop library (gto/mw_library.py, scripts/build_mw_library.py): entries that read back as
solved, in any suits, and the advisor answering a 3-way flop's first decisions from it at every speed (and only where
it holds the spot)."""
import importlib
import json
import re
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

from gto import flop_library as fl
from gto import mw_library as ml
from gto.advisor import Advisor, MW_LATER, MW_FLOP_BETS
from gto.cards import cards, cards_str, ALL_COMBOS, COMBO_INDEX
from gto.cli import parse_actions

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
build = importlib.import_module("build_mw_library")

SPOT = ml.SPOTS[0]           # CO opens, BTN and BB call
CANON = "Kc7d2h"
BOARD = "Kh7s2d"             # the same flop in other suits
ITERS = 20


def _hole_in(r, board, k=0, full=True):
    """The k-th combo with range r's top weight (else one off it) that the board doesn't block."""
    b = set(cards(board))
    idx = [i for i in np.nonzero(r >= 0.999 * r.max() if full else r <= 0)[0] if not set(ALL_COMBOS[i]) & b]
    return cards_str(ALL_COMBOS[idx[k]])


class TestNear(unittest.TestCase):
    """MWLibrary.near on the repo's 3-way library: a flop it doesn't store read off its stored flops most like it."""

    @classmethod
    def setUpClass(cls):
        cls.lib = ml.MWLibrary.from_dir()
        cls.setup = ml.spot_setup(SPOT[2])
        if not cls.lib.spot([fl.range_fp(r) for r in cls.setup["ranges"]], cls.setup["pot"], cls.setup["stack"]):
            raise unittest.SkipTest("the 3-way library doesn't hold the spot")

    def near(self, board):
        s = self.setup
        return self.lib.near(s["ranges"], cards(board), s["pot"], s["stack"])

    def test_reads_the_stored_decisions(self):
        sol = self.near("Kh8d3c")
        self.assertIsNotNone(sol)
        self.assertIsNone(self.lib.solution(self.setup["ranges"], cards("Kh8d3c"), self.setup["pot"], self.setup["stack"]))
        self.assertEqual(sorted(sol.nodes), sorted(tuple(p) for p in ml.STORED))
        self.assertEqual(len(sol.approx["near"]), 3)
        for nm, d in sol.approx["near"]:   # (rainbow, unpaired, a king high: in the hand's suits)
            b = cards(nm)
            self.assertTrue(0 < d <= 6 and len({c % 4 for c in b}) == 3 and len({c // 4 for c in b}) == 3)
        for path, node in sol.nodes.items():
            s = node.avg_strategy()[0]
            np.testing.assert_allclose(s.sum(axis=0), 1.0, atol=1e-5)
            self.assertTrue(sol.approx["known"][path].all())
            self.assertFalse(sol.action_evs(node).any())
        # a flop the library stores: its own solve answers, not this
        self.assertIsNone(self.near(CANON))

    def test_any_suits(self):
        # the same flop in other suits: each combo's mix that of its image
        a, b = self.near("Kh8d3c"), self.near("Ks8c3d")
        swap = {"h": "s", "d": "c", "c": "d", "s": "h"}
        col_b = {int(i): k for k, i in enumerate(b.hands.active)}
        img = [col_b[COMBO_INDEX[tuple(sorted(cards("".join(c[:-1] + swap[c[-1]] for c in
                                                             re.findall("..", cards_str(ALL_COMBOS[i]))))))]]
               for i in a.hands.active]
        for path in a.nodes:
            np.testing.assert_allclose(a.nodes[path].avg_strategy()[0], b.nodes[path].avg_strategy()[0][:, img], atol=1e-6)

    def test_left_out_flops(self):
        # stored flops read off the others (scripts/bench_mw_approx.py): the EV given up by their own EVs, % of the pot
        bench = importlib.import_module("bench_mw_approx")
        spot = self.lib.spot([fl.range_fp(r) for r in self.setup["ranges"]], self.setup["pot"], self.setup["stack"])
        losses = []
        for name in [CANON] + sorted(spot["flops"])[::70][:3]:
            got = bench.leave_one_out(self.lib, spot, name, [np.asarray(r) for r in self.setup["ranges"]])
            self.assertTrue(got is not None or name != CANON)   # (K72 rainbow has stored flops near; others may not)
            losses += [x[1] for x in got or ()]
            self.assertIn(name, spot["flops"])   # (put back)
        self.assertTrue(losses)
        self.assertLess(np.mean(losses), 0.5)
        self.assertGreater(min(losses), -0.05)


class TestMWLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        sid, label, line = SPOT
        cls.setup = s = ml.spot_setup(line)
        cls.sol = ml.solve_all(cards(CANON), s["ranges"], s["pot"], s["stack"])
        cls.sol.run(iters=ITERS)
        os.makedirs(os.path.join(cls.dir, sid))
        with open(os.path.join(cls.dir, sid, CANON + ".bin"), "wb") as f:
            f.write(ml.encode(cls.sol, dict(build._meta(sid, s), board=CANON, all_hands=True)))
        build.write_index(cls.dir, [SPOT])
        cls.lib = ml.MWLibrary.from_dir(cls.dir)
        cls.pre = parse_actions(line)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _libsol(self, board=CANON, lib=None, ranges=None, stack=None):
        s = self.setup
        return (lib or self.lib).solution(ranges or s["ranges"], cards(board), s["pot"], stack or s["stack"])

    def _advisor(self, **kw):
        adv = Advisor(time_budget=kw.get("budget", 0.5), prior_budget=0.3)
        adv.mw_library = self.lib
        return adv

    def test_setup_and_config(self):
        """The spot is 3-way (BB, CO, BTN to act), solved on native Accurate's 3-way flop game."""
        s = self.setup
        self.assertEqual(s["seats"], ["BB", "CO", "BTN"])
        self.assertEqual((s["pot"], s["stack"]), (8.0, 97.5))
        c = ml.library_config()
        self.assertEqual((c.later, c.buckets), (MW_LATER["later"], MW_LATER["buckets"]))
        self.assertEqual(c.bet["flop"], MW_FLOP_BETS["flop"])
        self.assertEqual(fl.flop_name(cards(BOARD)), CANON)

    def test_entry_reads_back_as_solved(self):
        """On the flop it was solved on (rainbow: each combo its own class), each stored node (first to act, checked to
        once and twice) holds the solve's strategy to within a step (1/Q) for every hand (off a range: the current
        strategy), its EVs to within the encoding's, and its best-response gains."""
        ls = self._libsol()
        self.assertIsNotNone(ls)
        self.assertEqual(sorted(ls.nodes), sorted(ml.STORED))
        self.assertEqual(ls.iters, ITERS)
        pos = {c: i for i, c in enumerate(self.sol.hands.active)}
        cols = np.array([pos[c] for c in ls.hands.active])
        step = fl.EV_STEP * self.setup["pot"]
        for path in ml.STORED:
            n0, R = ml._reach(self.sol, path)
            n1 = ls.nodes[path]
            self.assertEqual(n0.names, n1.names)
            self.assertEqual(n0.player, len(path))
            p = n0.player
            reach = R[p, 0][cols]
            want = np.where(reach > 1e-12, n0.avg_strategy()[0][:, cols], n0.strategy()[0][:, cols])
            self.assertLess(np.abs(want - n1.avg_strategy()[0]).max(), 1.01 / fl.Q, path)
            e0 = self.sol.action_evs(n0, R)[:, cols]
            e1 = ls.action_evs(n1)
            gap = e0.max(axis=0, keepdims=True) - e0
            self.assertTrue(np.all(np.abs(e0 - e1) <= 1.001 * step + 0.021 * gap), path)
            self.assertEqual(n1.amounts, n0.amounts)
        g = self.sol.br_gains() / self.sol.root_pot() * 100
        self.assertAlmostEqual(ls.exploitability(), max(g.mean(), 0.0), places=3)
        # every hand the flop doesn't block, for every player (all_hands)
        self.assertEqual(ls.hands.n, 1176)

    def test_other_suits(self):
        """The same flop in other suits reads the same entry, each combo taking its canonical combo's values."""
        a, b = self._libsol(), self._libsol(BOARD)
        cmap = fl.combo_map(fl.canonical(cards(BOARD))[1])
        pos = {c: i for i, c in enumerate(a.hands.active)}
        cols = np.array([pos[cmap[c]] for c in b.hands.active])
        for path in ml.STORED:
            np.testing.assert_allclose(b.nodes[path].avg_strategy()[0], a.nodes[path].avg_strategy()[0][:, cols])
        np.testing.assert_allclose(b.reach0[:, 0], a.reach0[:, 0][:, cols])

    def test_mismatch_is_ignored(self):
        """Other ranges, another stack, rake, a flop it doesn't hold, a tampered index: no library."""
        s = self.setup
        self.assertIsNone(self._libsol(ranges=[s["ranges"][0] * 0.9] + s["ranges"][1:]))
        self.assertIsNone(self._libsol(stack=s["stack"] - 10))
        self.assertIsNone(self._libsol("Qh8s3d"))
        self.assertIsNone(self.lib.solution(s["ranges"], cards(CANON), s["pot"], s["stack"], rake=True))
        d = tempfile.mkdtemp()
        try:
            shutil.copytree(self.dir, d, dirs_exist_ok=True)
            with open(os.path.join(d, ml.INDEX)) as f:
                idx = json.load(f)
            idx["tree"]["version"] = ml.MODEL_VERSION + 1
            with open(os.path.join(d, ml.INDEX), "w") as f:
                json.dump(idx, f)
            self.assertIsNone(self._libsol(lib=ml.MWLibrary.from_dir(d)))
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertIsNone(self._libsol(lib=ml.MWLibrary(lambda p: None)))

    def test_advisor_answers_from_library(self):
        """Each of the flop's first decisions is answered at once from the library, quick requests included, with the
        stored mix for the hero's hand (a hand off its range too, noted); a practice opponent's move after it is read
        off the same solve."""
        s = self.setup
        X = lambda p: {"pos": p, "action": "check"}
        flops = {"BB": [], "CO": [X("BB")], "BTN": [X("BB"), X("CO")]}
        cmap = fl.combo_map(fl.canonical(cards(BOARD))[1])
        ref = self._libsol()
        pos = {c: i for i, c in enumerate(ref.hands.active)}
        for i, seat in enumerate(s["seats"]):
            for full in (True, False):
                hole = _hole_in(s["ranges"][i], BOARD, 2, full=full)
                adv = self._advisor()
                t = time.time()
                r = adv.advise({"hero": seat, "hole": hole, "board": BOARD, "preflop": self.pre, "stack": 100,
                                "flop": flops[seat], "quick": True})
                self.assertLess(time.time() - t, 0.5)
                self.assertTrue(r.get("solved"), r.get("notes"))
                self.assertEqual(r["precomputed"]["iterations"], ITERS)
                self.assertEqual(r["position"], f"{i + 1} of 3 to act")
                ci = pos[cmap[COMBO_INDEX[tuple(sorted(cards(hole)))]]]
                want = ref.nodes[tuple("check" for _ in flops[seat])].avg_strategy()[0][:, ci]
                np.testing.assert_allclose([a["freq"] for a in r["actions"]], want, atol=0.006)
                self.assertEqual(any("isn't in your range" in n for n in r["notes"]), not full)
                self.assertEqual(all("ev_bb" in a for a in r["actions"]), full)   # (stored off-range EVs aren't read)
        # the next opponent's move, from the solve the hero's advice came from
        adv = self._advisor()
        st = {"hero": "BB", "hole": _hole_in(s["ranges"][0], BOARD), "board": BOARD, "preflop": self.pre, "stack": 100}
        adv.advise(st)
        mv = adv.play(dict(st, flop=[X("BB")]), "CO", _hole_in(s["ranges"][1], BOARD), seed=1)
        self.assertEqual(mv["street"], "flop")
        self.assertIn(mv["name"], ref.nodes[("check",)].names)

    def test_turn_narrowed_through_library(self):
        """A turn after the flop checks through 3-way: the flop's ranges follow the library's solve (as settled as
        its entry says: here one marked settled)."""
        s = self.setup
        lib = ml.MWLibrary.from_dir(self.dir)
        lib.index()
        e = lib.entry(lib._spots[next(iter(lib._spots))], CANON)
        e["head"]["gains"] = [0.05, 0.05, 0.05]
        adv = Advisor(time_budget=0.5, prior_budget=0.3)
        adv.mw_library = lib
        adv.mw_solve_streets = ()
        X = lambda p: {"pos": p, "action": "check"}
        r = adv.advise({"hero": "BB", "hole": _hole_in(s["ranges"][0], BOARD + "4c"), "board": BOARD + "4c",
                        "preflop": self.pre, "stack": 100, "flop": [X("BB"), X("CO"), X("BTN")]})
        self.assertTrue(any(n.startswith("flop: 3-way, ranges narrowed through the precomputed 3-way solve")
                            for n in r["notes"]), r["notes"])
        self.assertNotIn("precomputed", r)

    def test_not_used_where_it_does_not_apply(self):
        """After a bet (a node it doesn't store), with rake, custom sizes or ranges, exploit mode, a lock, another stack:
        not from the library (the rules or a live solve answer)."""
        s = self.setup
        hole = _hole_in(s["ranges"][1], BOARD)
        base = {"hero": "CO", "hole": hole, "board": BOARD, "preflop": self.pre, "stack": 100,
                "flop": [{"pos": "BB", "action": "check"}], "quick": True}
        adv = self._advisor()
        self.assertIn("precomputed", adv.advise(base))
        for st in (dict(base, flop=[{"pos": "BB", "action": "bet", "amount": 2.5}]), dict(base, rake=0.05, rake_cap=3),
                   dict(base, sizes={"flop": [50]}), dict(base, ranges={"BTN": "TT+, AK"}), dict(base, opponents="pool"),
                   dict(base, locks={"BTN": {"fold": 15}}), dict(base, stack=60)):
            r = self._advisor().advise(st)
            self.assertNotIn("precomputed", r, st)

    def test_library_only(self):
        """The session analyzer's library scope answers a 3-way flop the library holds, and skips the rest."""
        s = self.setup
        base = {"hero": "BB", "hole": _hole_in(s["ranges"][0], BOARD), "board": BOARD, "preflop": self.pre, "stack": 100,
                "library_only": True}
        self.assertIn("precomputed", self._advisor().advise(base))
        self.assertIn("skipped", self._advisor().advise(dict(base, board="Qh8s3d")))


class TestShippedMWLibrary(unittest.TestCase):
    def test_index_matches_the_charts(self):
        """The shipped 3-way library was built for the current charts and tree, and stays small."""
        path = os.path.join(ml.DIR, ml.INDEX)
        if not os.path.exists(path):
            self.skipTest("no 3-way library built")
        with open(path) as f:
            idx = json.load(f)
        self.assertEqual(fl._key_str(idx["tree"]), fl._key_str(ml.tree_key(ml.library_config())))
        lines = {s[0]: s[2] for s in ml.SPOTS}
        total = 0
        for sp in idx["spots"]:
            s = ml.spot_setup(lines[sp["id"]])
            self.assertEqual(sp["fp"], [fl.range_fp(r) for r in s["ranges"]],
                             f"{sp['id']}: the charts changed; rebuild with scripts/build_mw_library.py")
            self.assertEqual((sp["pot"], sp["stack"], sp["seats"]), (s["pot"], s["stack"], s["seats"]))
            for name in sp["flops"]:
                total += os.path.getsize(os.path.join(ml.DIR, sp["id"], name + ".bin"))
        self.assertLess(total, 6e6)


if __name__ == "__main__":
    unittest.main()
