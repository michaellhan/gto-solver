"""The precomputed flop library (gto/flop_library.py, scripts/build_flop_library.py): canonical flops, entries that
read back as solved, the advisor answering from it (and only where it holds the spot)."""
import hashlib
import importlib
import itertools
import json
import os
import random
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

from gto import flop_library as fl
from gto.advisor import Advisor
from gto.cards import cards, cards_str, ALL_COMBOS, COMBO_INDEX
from gto.cli import parse_actions
from gto.solver import solve, Config, MEMBER_FULL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
build = importlib.import_module("build_flop_library")

SPOT = next(s for s in fl.SPOTS if s[0] == "3bp_btn_co")   # narrow ranges: quick to solve; the caller checks
CANON = "Kc7d2h"
BOARD = "Kh7s2d"            # the same flop in other suits
ITERS = 120


def _hole_in(r, board, k=0):
    """The k-th combo with full weight in range r that the board doesn't block."""
    b = set(cards(board))
    idx = [i for i in np.nonzero(r >= 0.999)[0] if not set(ALL_COMBOS[i]) & b]
    return cards_str(ALL_COMBOS[idx[k]])


class TestCanonicalFlops(unittest.TestCase):
    def test_count_and_weights(self):
        fls = fl.all_flops()
        self.assertEqual(len(fls), 1755)
        self.assertEqual(sum(fls.values()), 22100)
        rep = fl.representative_flops(184)
        self.assertEqual(len(rep), 184)
        self.assertEqual(len(set(rep)), 184)
        self.assertTrue(all(f in fls for f in rep))
        # every suit pattern gets about its share (rainbow 40%, two-tone 55%, monotone 5% of flops)
        mono = sum(1 for f in rep if len({c % 4 for c in f}) == 1)
        self.assertTrue(5 <= mono <= 15, mono)
        # a report's weights: the representative set stands for every flop evenly; other sets for their own flops
        names = [cards_str(f) for f in rep]
        w = fl.report_weights(names)
        self.assertAlmostEqual(sum(w.values()), 22100)
        self.assertEqual(len(set(w.values())), 1)
        self.assertEqual(fl.report_weights(["Kc7d2h", "4c3c2c"]), {"Kc7d2h": 24.0, "4c3c2c": 4.0})
        self.assertEqual(sum(fl.report_weights([cards_str(f) for f in fls]).values()), 22100)

    def test_round_trip(self):
        """Every suit permutation of a flop has the same canonical flop, the suit map takes the flop there, and
        the combo map is a bijection that takes the flop's blocked combos to the canonical flop's."""
        rng = random.Random(3)
        for _ in range(40):
            f = tuple(rng.sample(range(52), 3))
            canon, smap = fl.canonical(f)
            self.assertEqual(tuple(sorted((c // 4 * 4 + smap[c % 4] for c in f), reverse=True)), canon)
            for p in itertools.permutations(range(4)):
                g = tuple(c // 4 * 4 + p[c % 4] for c in f)
                self.assertEqual(fl.canonical(g)[0], canon)
            m = fl.combo_map(smap)
            self.assertEqual(sorted(m), list(range(len(ALL_COMBOS))))
            blocked = MEMBER_FULL[list(f)].sum(axis=0) > 0
            cblocked = MEMBER_FULL[list(canon)].sum(axis=0) > 0
            self.assertTrue(np.array_equal(cblocked[m], blocked))
        self.assertEqual(fl.flop_name(cards("Ks7d2c")), "Kc7d2h")
        self.assertEqual(fl.flop_name(cards("2h7hKs")), fl.flop_name(cards("Kd7c2c")))

    def test_tree_key(self):
        """The native and browser Accurate trees are the library's; Fast is the same sizes, another flop model."""
        from gto.native import advisor
        lk = fl.tree_key(fl.library_config())
        acc, fast = (fl.tree_key(advisor(s, library=False).cfg) for s in ("accurate", "fast"))
        self.assertEqual(fl._key_str(acc), fl._key_str(lk))
        self.assertEqual(fl._key_str(fast["sizes"]), fl._key_str(lk["sizes"]))
        self.assertNotEqual(fl._key_str(fast["model"]), fl._key_str(lk["model"]))


class TestFlopLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        sid, label, line = SPOT
        cls.setup = fl.spot_setup(line)
        s = cls.setup
        cls.sol = solve(cards(CANON), s["full0"], s["full1"], s["pot"], s["stack"], cfg=fl.library_config(),
                        time_budget=1e9, max_iters=ITERS)
        meta = dict(build._meta(sid, s), board=CANON)
        os.makedirs(os.path.join(cls.dir, sid))
        with open(os.path.join(cls.dir, sid, CANON + ".bin"), "wb") as f:
            f.write(fl.encode(cls.sol, meta))
        build.write_index(cls.dir, [SPOT])
        cls.lib = fl.FlopLibrary.from_dir(cls.dir)
        cls.pre = parse_actions(line)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _libsol(self, board=CANON, cfg=None, lib=None, full=None):
        s = self.setup
        f0, f1 = full or (s["full0"], s["full1"])
        return (lib or self.lib).solution(f0, f1, cards(board), s["full0"], s["full1"], s["pot"], s["stack"],
                                          cfg or fl.library_config())

    def test_entry_reads_back_as_solved(self):
        """On the flop it was solved on, every stored node's strategy is the solve's to within a step (1/Q) and
        its EVs to within the encoding's (0.01 bb near the best action, 1.5% of the gap below it)."""
        ls = self._libsol()
        self.assertIsNotNone(ls)
        self.assertTrue(ls.exact)
        self.assertEqual(ls.iters, ITERS)
        paths = fl.stored_paths(self.sol.root, self.sol.r0, self.sol.r1)
        self.assertIn((), paths)
        self.assertIn(("check",), paths)
        pos = {c: i for i, c in enumerate(self.sol.ctx.active)}
        cols = np.array([pos[c] for c in ls.ctx.active])
        for path in paths:
            n0, n1 = fl.node_at(self.sol.root, path), fl.node_at(ls.root, path)
            self.assertEqual(n0.names, n1.names)
            p = n0.player
            r = self.sol.r0 if p == 0 else self.sol.r1
            m = r[cols] > 0
            s0, s1 = n0.avg_strategy0()[:, cols][:, m], n1.avg_strategy0()[:, m]
            self.assertLess(np.abs(s0 - s1).max(), 1.01 / fl.Q, path)
            a0, a1 = fl._reaches(self.sol.root, path, self.sol.r0, self.sol.r1)
            e0 = self.sol.solver.node_action_evs(n0, a0, a1)[:, cols][:, m]
            e1 = ls.solver.node_action_evs(n1)[:, m]
            gap = e0.max(axis=0, keepdims=True) - e0
            self.assertTrue(np.all(np.abs(e0 - e1) <= 0.011 + 0.016 * gap), path)
        self.assertEqual(ls.exploitability(), round(self.sol.exploitability(), 4))

    def test_ranges(self):
        """write_index writes the spot's ranges.json (the drills' hand weights): both ranges entering the flop in
        1/10000, fingerprinted as the index's, holding every combo of the entry's ranges."""
        with open(os.path.join(self.dir, SPOT[0], "ranges.json")) as f:
            rg = json.load(f)
        with open(os.path.join(self.dir, "index.json")) as f:
            sp = json.load(f)["spots"][0]
        s = self.setup
        self.assertEqual(rg["fp"], sp["fp"])
        for p, full in ((0, s["full0"]), (1, s["full1"])):
            w = np.array(rg["w"][p]) / 10000
            self.assertEqual(len(w), len(ALL_COMBOS))
            self.assertLess(np.abs(w - full).max(), 0.5e-4 + 1e-9)
            self.assertEqual(fl.range_fp(w), sp["fp"][p])
        with open(os.path.join(self.dir, SPOT[0], CANON + ".bin"), "rb") as f:
            e = fl.decode(f.read())
        for p in (0, 1):
            self.assertTrue(np.all(np.array(rg["w"][p])[e["masks"][p]] > 0))

    def test_report(self):
        """write_index writes the spot's report.json (hashed in the index): per flop, each stored node's range
        frequencies (the entry's, summing to 1), EVs summing to the pot, equities to 1, a hand in the acting player's
        range off the board; a rewrite keeps the lines of unchanged entries without reading them."""
        sid = SPOT[0]
        rp = os.path.join(self.dir, sid, "report.json")
        with open(rp, "rb") as f:
            data = f.read()
        with open(os.path.join(self.dir, "index.json")) as f:
            idx = json.load(f)
        self.assertEqual(idx["spots"][0]["report"], hashlib.sha1(data).hexdigest()[:10])
        rep = json.loads(data)
        s = self.setup
        self.assertEqual((rep["id"], rep["line"], rep["seats"], rep["pot"]), (sid, SPOT[2], [s["oop"], s["ip"]], s["pot"]))
        self.assertEqual([x["f"] for x in rep["flops"]], [CANON])
        line = rep["flops"][0]
        with open(os.path.join(self.dir, sid, CANON + ".bin"), "rb") as f:
            e = fl.decode(f.read())
        self.assertEqual(set(line["n"]), {"|".join(nd["path"]) for nd in e["nodes"]})
        for nd in e["nodes"]:
            k = "|".join(nd["path"])
            self.assertEqual(rep["nodes"][k], nd["names"])
            fr = line["n"][k]
            self.assertAlmostEqual(sum(fr), 1.0, delta=0.002)
            for x, nm in zip(fr, nd["names"]):
                self.assertAlmostEqual(x, nd["freqs"][nm], delta=0.001)
            hand = cards(line["h"][k])
            self.assertFalse(set(hand) & set(cards(CANON)))
            r = s["full0"] if nd["player"] == 0 else s["full1"]
            self.assertGreater(r[COMBO_INDEX[tuple(sorted(hand))]], 0.99)
        self.assertAlmostEqual(sum(line["ev"]), s["pot"], delta=0.011)
        self.assertAlmostEqual(sum(line["eq"]), 1.0, delta=0.0011)
        # the root's EV: out of position's range average of the solve's node values
        root = self.sol.root
        r0 = self.sol.r0
        v = (root.avg_strategy0() * self.sol.solver.node_action_evs(root, self.sol.r0, self.sol.r1)).sum(axis=0)
        self.assertAlmostEqual(line["ev"][0], float((r0 * v).sum() / r0.sum()), delta=0.03)
        # range equity: exact over every runout within half a point; a range against itself is 50%
        board = cards(CANON)
        bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        exact = fl.range_equity(board, s["full0"] * bm, s["full1"] * bm, step=1)
        self.assertAlmostEqual(line["eq"][0], exact, delta=0.005)
        self.assertAlmostEqual(fl.range_equity(board, s["full0"] * bm, s["full0"] * bm), 0.5, delta=1e-9)
        # a rewrite reuses the unchanged line (no entry read) and gives the same file
        def read(name):
            raise AssertionError("an unchanged entry was read again")
        again = fl.build_report(idx["spots"][0], s["full0"], s["full1"], read, rep)
        self.assertEqual(json.dumps(again, separators=(",", ":")).encode(), data)

    def test_reproduces_live_solve_in_other_suits(self):
        """The entry, read for the flop in other suits, is a live solve of that flop with the same tree and
        iterations, to within the noise of suit order (the turn classes' cards differ, and 120 iterations are far
        from settled per combo): range frequencies within 3 points at every stored node both players reach with 10%
        of their range (6 behind a line rarer than that), and at the root each
        combo's mix within 0.1 on average."""
        ls = self._libsol(BOARD)
        self.assertIsNotNone(ls)
        s = self.setup
        live = solve(cards(BOARD), s["full0"], s["full1"], s["pot"], s["stack"], cfg=fl.library_config(),
                     time_budget=1e9, max_iters=ITERS)
        self.assertTrue(np.array_equal(live.ctx.active, ls.ctx.active))
        for path in fl.stored_paths(self.sol.root, self.sol.r0, self.sol.r1):
            n0, n1 = fl.node_at(live.root, path), fl.node_at(ls.root, path)
            a0, a1 = fl._reaches(live.root, path, live.r0, live.r1)
            r = a0 if n0.player == 0 else a1
            w = r / r.sum()
            f0, f1 = n0.avg_strategy0() @ w, n1.avg_strategy0() @ w
            common = min(a0.sum() / live.r0.sum(), a1.sum() / live.r1.sum()) >= 0.1
            self.assertLess(np.abs(f0 - f1).max(), 0.03 if common else 0.06, (path, f0, f1))
            if path:
                continue
            m = (live.r0 if n0.player == 0 else live.r1) > 0
            self.assertLess(np.abs(n0.avg_strategy0()[:, m] - n1.avg_strategy0()[:, m]).sum(axis=0).mean() / 2, 0.1, path)

    def test_mismatch_is_ignored(self):
        """Other ranges (changed charts), another tree or model version, a tampered entry: no library."""
        s = self.setup
        self.assertIsNone(self._libsol(full=(s["full0"] * 0.9, s["full1"])))
        cfg = fl.library_config()
        cfg.bet_sizes["flop"] = [0.5, 0.75]
        self.assertIsNone(self._libsol(cfg=cfg))
        cfg = fl.library_config()
        cfg.rake_rate = 0.05
        self.assertIsNone(self._libsol(cfg=cfg))
        self.assertIsNone(self.lib.solution(s["full0"], s["full1"], cards("Qh8s3d"), s["full0"], s["full1"],
                                            s["pot"], s["stack"], fl.library_config()))   # a flop it doesn't hold
        self.assertIsNone(self.lib.solution(s["full0"], s["full1"], cards(BOARD), s["full0"], s["full1"],
                                            s["pot"], s["stack"] - 10, fl.library_config()))   # another stack
        # the index's fingerprint, or the model version, no longer what the code computes
        d = tempfile.mkdtemp()
        try:
            shutil.copytree(self.dir, d, dirs_exist_ok=True)
            with open(os.path.join(d, "index.json")) as f:
                idx = json.load(f)
            bad = json.loads(json.dumps(idx))
            bad["spots"][0]["fp"][0] = "0" * 16
            with open(os.path.join(d, "index.json"), "w") as f:
                json.dump(bad, f)
            self.assertIsNone(self._libsol(lib=fl.FlopLibrary.from_dir(d)))
            bad = json.loads(json.dumps(idx))
            bad["tree"]["model"]["version"] = fl.MODEL_VERSION + 1
            with open(os.path.join(d, "index.json"), "w") as f:
                json.dump(bad, f)
            self.assertIsNone(self._libsol(lib=fl.FlopLibrary.from_dir(d)))
            # an entry whose header doesn't match its index line (rebuilt for other charts, say)
            with open(os.path.join(d, "index.json"), "w") as f:
                json.dump(idx, f)
            head = dict(build._meta(SPOT[0], self.setup), board=CANON)
            head["fp"] = ["1" * 16, head["fp"][1]]
            with open(os.path.join(d, SPOT[0], CANON + ".bin"), "wb") as f:
                f.write(fl.encode(self.sol, head))
            self.assertIsNone(self._libsol(lib=fl.FlopLibrary.from_dir(d)))
            # stale entries are left out of a rebuilt index
            build.write_index(d, [SPOT])
            with open(os.path.join(d, "index.json")) as f:
                self.assertEqual(json.load(f)["spots"], [])
        finally:
            shutil.rmtree(d, ignore_errors=True)
        # no index at all
        self.assertIsNone(self._libsol(lib=fl.FlopLibrary(lambda p: None)))

    def test_advisor_answers_from_library(self):
        """The advisor's flop answer comes from the library where it holds the spot: instantly, at the stored
        iterations and exploitability, with the stored mix for the hero's hand; on the same tree (Accurate's
        model) or labelled as Accurate's model for Fast's. A turn spot narrows the flop through it."""
        s = self.setup
        hole = _hole_in(s["full0"], BOARD, 3)
        base = {"hero": s["oop"], "hole": hole, "board": BOARD, "preflop": self.pre, "stack": 100}
        fast = Config(n_turns=3, river_play=False, turn_classes=False, **{k: {a: list(b) for a, b in v.items()} for k, v in fl.LS.items()})
        for cfg, same in ((fl.library_config(), True), (fast, False)):
            adv = Advisor(cfg, time_budget=1.0, prior_budget=0.5)
            adv.library = self.lib
            t = time.time()
            r = adv.advise(base)
            self.assertLess(time.time() - t, 0.9)
            self.assertEqual(r["precomputed"]["same_tree"], same, r.get("notes"))
            self.assertEqual(r["solver"]["iterations"], ITERS)
            self.assertAlmostEqual(r["solver"]["exploitability_pct"], round(self.sol.exploitability(), 2), places=2)
            canon_combo = fl.combo_map(fl.canonical(cards(BOARD))[1])[COMBO_INDEX[tuple(sorted(cards(hole)))]]
            i = list(self.sol.ctx.active).index(canon_combo)
            want = self.sol.root.avg_strategy0()[:, i]
            got = [a["freq"] for a in r["actions"]]
            self.assertLess(np.abs(np.array(got) - want).max(), 0.011, (got, want))
            self.assertTrue(any("precomputed solution" in n for n in r["notes"]))
            # the in-position 3bettor after the caller checks: stored too (a lead, which the caller hardly ever
            # makes, isn't: STORE_LINE)
            ri = adv.advise(dict(base, hero=s["ip"], hole=_hole_in(s["full1"], BOARD, 5),
                                 flop=[{"pos": s["oop"], "action": "check"}]))
            self.assertIn("precomputed", ri)
            self.assertEqual([a["action"] for a in ri["actions"]][0], "check")
        # without the library the same spot is solved live: the budget, fewer iterations
        adv = Advisor(fl.library_config(), time_budget=1.0, prior_budget=0.5)
        t = time.time()
        r = adv.advise(base)
        self.assertGreater(time.time() - t, 0.9)
        self.assertNotIn("precomputed", r)
        self.assertGreater(r["solver"]["exploitability_pct"], self.sol.exploitability())
        # a turn spot: the flop's narrowing reads the library
        adv = Advisor(fast, time_budget=1.0, prior_budget=0.5)
        adv.library = self.lib
        r = adv.advise(dict(base, board=BOARD + "4c", flop=[{"pos": s["oop"], "action": "check"},
                                                          {"pos": s["ip"], "action": "check"}]))
        self.assertTrue(any(n.startswith("flop: precomputed solution") for n in r["notes"]), r["notes"])
        self.assertNotIn("precomputed", r)   # (the turn itself is solved live)

    def test_library_only(self):
        """"library_only" (the session analyzer's instant scope): a flop the library holds is answered from it, anything
        else is skipped at once, never solved; preflop answers as usual."""
        s = self.setup
        adv = Advisor(fl.library_config(), time_budget=5.0, prior_budget=5.0)
        adv.library = self.lib
        base = {"hero": s["oop"], "hole": _hole_in(s["full0"], BOARD, 3), "board": BOARD, "preflop": self.pre, "stack": 100,
                "library_only": True}
        self.assertIn("precomputed", adv.advise(base))
        t = time.time()
        for st in (dict(base, board=BOARD + "4c", flop=[{"pos": s["oop"], "action": "check"}, {"pos": s["ip"], "action": "check"}]),
                   dict(base, stack=60)):   # a turn; a stack the library doesn't hold
            self.assertIn("skipped", adv.advise(st))
        self.assertLess(time.time() - t, 2.0)
        self.assertNotIn("skipped", adv.advise(dict(base, board="", preflop=self.pre[:1], hero=self.pre[1]["pos"])))
        self.assertFalse(adv._lib_only)

    def test_not_used_where_it_does_not_apply(self):
        """Exploit mode, a nodelock on the villain, a custom range and a rare villain line (a lead the solve never makes) solve live."""
        s = self.setup
        adv = Advisor(fl.library_config(), time_budget=0.5, prior_budget=0.3)
        adv.library = self.lib
        base = {"hero": s["oop"], "hole": _hole_in(s["full0"], BOARD, 3), "board": BOARD, "preflop": self.pre, "stack": 100}
        self.assertNotIn("precomputed", adv.advise(dict(base, opponents="pool")))
        self.assertNotIn("precomputed", adv.advise(dict(base, ranges={s["ip"]: "TT+, AK"})))
        self.assertNotIn("precomputed", adv.advise(dict(base, locks={s["ip"]: {"fold": 15}})))
        lead = self.sol.root.avg_strategy0() @ (self.sol.r0 / self.sol.r0.sum())
        rare = [n for n, f in zip(self.sol.root.names, lead) if n.startswith("bet") and f < 0.05]
        if rare:   # in-position facing a lead the solve rarely makes
            amt = float(rare[0].split()[1][:-1]) / 100 * s["pot"]
            r = adv.advise(dict(base, hero=s["ip"], hole=_hole_in(s["full1"], BOARD, 5),
                                flop=[{"pos": s["oop"], "action": "bet", "amount": round(amt, 1)}]))
            self.assertNotIn("precomputed", r)


class TestShippedLibrary(unittest.TestCase):
    def test_index_matches_the_charts(self):
        """The shipped library (gto/static/flop_library) was built for the current charts and tree: every spot in
        its index has the fingerprint the charts give now, and its entries are small."""
        path = os.path.join(fl.DIR, "index.json")
        if not os.path.exists(path):
            self.skipTest("no library built")
        with open(path) as f:
            idx = json.load(f)
        self.assertEqual(fl._key_str(idx["tree"]), fl._key_str(fl.tree_key(fl.library_config())))
        lines = {s[0]: s[2] for s in fl.SPOTS}
        total = 0
        for sp in idx["spots"]:
            s = fl.spot_setup(lines[sp["id"]])
            self.assertEqual(sp["fp"], [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])],
                             f"{sp['id']}: the charts changed; rebuild with scripts/build_flop_library.py")
            self.assertEqual((sp["pot"], sp["stack"]), (s["pot"], s["stack"]))
            for name in sp["flops"]:
                size = os.path.getsize(os.path.join(fl.DIR, sp["id"], name + ".bin"))
                self.assertLess(size, 200_000)
                total += size
        self.assertLess(total, 15e6)

    def test_reports_match_the_index(self):
        """Each shipped spot's report.json (the page's flop reports) is the one the index hashes, holds the index's
        flops with their entry hashes, its weights sum to the 22100 flops, and its frequencies to 1."""
        path = os.path.join(fl.DIR, "index.json")
        if not os.path.exists(path):
            self.skipTest("no library built")
        with open(path) as f:
            idx = json.load(f)
        for sp in idx["spots"]:
            with open(os.path.join(fl.DIR, sp["id"], "report.json"), "rb") as f:
                data = f.read()
            self.assertEqual(sp["report"], hashlib.sha1(data).hexdigest()[:10], sp["id"])
            self.assertLess(len(data), 100_000)
            rep = json.loads(data)
            self.assertEqual({x["f"]: x["hash"] for x in rep["flops"]}, sp["flops"])
            self.assertAlmostEqual(sum(x["w"] for x in rep["flops"]), 22100, delta=1)
            for x in rep["flops"]:
                self.assertIn("", x["n"])
                for k, fr in x["n"].items():
                    self.assertEqual(len(fr), len(rep["nodes"][k]))
                    self.assertAlmostEqual(sum(fr), 1.0, delta=0.002, msg=(sp["id"], x["f"], k))
                self.assertAlmostEqual(sum(x["ev"]), sp["pot"], delta=0.011)

    def test_shipped_ranges_match_the_index(self):
        """Each shipped spot has its ranges.json (the page's drills), with the index's fingerprints."""
        path = os.path.join(fl.DIR, "index.json")
        if not os.path.exists(path):
            self.skipTest("no library built")
        with open(path) as f:
            idx = json.load(f)
        for sp in idx["spots"]:
            with open(os.path.join(fl.DIR, sp["id"], "ranges.json")) as f:
                rg = json.load(f)
            self.assertEqual(rg["fp"], sp["fp"], f"{sp['id']}: rebuild the index (scripts/build_flop_library.py --index-only)")
            for p in (0, 1):
                self.assertEqual(fl.range_fp(np.array(rg["w"][p]) / 10000), sp["fp"][p])


if __name__ == "__main__":
    unittest.main()
