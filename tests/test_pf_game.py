"""The preflop game solver (gto/pf_game.py)."""
import unittest
import numpy as np
from gto import pf_game as G
from gto.pf_classes import CLASSES, equity_matrix


class TestTree(unittest.TestCase):
    def test_six_max_lines(self):
        root = G.build()
        nodes = list(G.walk(root))
        # the UTG open, a BTN 3bet, CO folding... the opener facing it folds, calls or 4bets
        g = G.Game(root, G.SIX, G.flat_model())
        nd = g.find([("UTG", "raise"), ("BTN", "raise"), ("UTG", None)][:2])
        while g.positions[nd.player] != "UTG":
            nd = nd.children[nd.names.index("fold")]
        self.assertEqual(nd.names, ["fold", "call", "raise"])
        four = nd.children[2]
        self.assertAlmostEqual(four.line[-1][2], 7.5 * 2.3)
        # sizes: 2.5 opens, 3bb from the SB, 3x in position, 12bb from a blind, 10bb BB over the SB
        self.assertEqual(g.find([("CO", "raise")]).line[-1][2], 2.5)
        self.assertEqual(g.find([("SB", "raise")]).line[-1][2], 3.0)
        self.assertEqual(g.find([("CO", "raise"), ("BTN", "raise")]).line[-1][2], 7.5)
        self.assertEqual(g.find([("CO", "raise"), ("BB", "raise")]).line[-1][2], 12.0)
        self.assertEqual(g.find([("SB", "raise"), ("BB", "raise")]).line[-1][2], 10.0)
        # a squeeze adds one open per caller
        self.assertEqual(g.find([("CO", "raise"), ("BTN", "call"), ("BB", "raise")]).line[-1][2], 14.5)
        # the 5bet is all-in, and at most three players see a flop (two in a raised pot)
        self.assertTrue(all(len(t.active) <= 3 for t in nodes if t.kind == "show"))
        self.assertTrue(all(len(t.active) == 2 for t in nodes if t.kind == "show" and t.n_raises > 1))
        self.assertTrue(all(t.inv.max() <= 100 for t in nodes if t.kind != "dec"))

    def test_zero_sum(self):
        """Every terminal pays out what was put in (without rake): the seats' values sum to 0 heads-up, and
        nearly so with more players (card removal against each opponent is taken independently, and
        three-way pots are normalised over the ranges, not hand by hand)."""
        g = G.Game(G.build(["SB", "BB"]), ["SB", "BB"], G.flat_model())
        g.solve(20)
        self.assertAlmostEqual(float(g.values().sum()), 0.0, places=9)
        g = G.Game(G.build(G.SIX[2:]), G.SIX[2:], G.flat_model())
        g.solve(20)
        self.assertLess(abs(float(g.values().sum())), 0.04)


class TestSolve(unittest.TestCase):
    def test_heads_up_converges(self):
        g = G.Game(G.build(["SB", "BB"]), ["SB", "BB"], G.flat_model())
        g.solve(400)
        self.assertLess(g.exploitability(), 0.01)          # bb per hand, both seats
        # with every flop paying all-in equity (no postflop edge) the big blind defends nearly everything
        nd = g.find([("SB", "raise"), ("BB", None)][:1])
        s = g.strategy_at(nd)
        self.assertGreater(float((G.PRIOR * s["fold"]).sum()), -1e-9)
        self.assertLess(float((G.PRIOR * s["fold"]).sum()), 0.2)

    def test_model_from_tables_keeps_measurements(self):
        """The two-way decomposition gives back each side's per-bucket excess on average over the
        opponent's hands in that bucket, and the pair shares add to the pot."""
        from gto.pf_solve import bucket_of
        rng = np.random.default_rng(1)
        b = bucket_of()
        nb = int(b.max()) + 1
        xa = rng.normal(0, 0.05, (len(CLASSES), nb))
        wa = rng.random(len(CLASSES)); wc = rng.random(len(CLASSES))
        # a consistent caller table: its block means are minus the aggressor's, plus class effects
        onehot = np.eye(nb)[b]
        ma = (onehot * wa[:, None]).T @ xa / (onehot * wa[:, None]).sum(0)[:, None]   # (aggr bucket, caller bucket)
        noise = rng.normal(0, 0.05, (len(CLASSES), nb))
        mn = (onehot * wc[:, None]).T @ noise / (onehot * wc[:, None]).sum(0)[:, None]
        xc = -ma.T[b] + noise - mn[b]
        key = (2, True)
        m = G.EQRModel.from_tables({key + ("aggr",): xa, key + ("caller",): xc},
                                   {key + ("aggr",): wa, key + ("caller",): wc})
        s = m.share(key) - equity_matrix()
        for i in (0, 50, 100):
            for k in range(nb):
                cols = (b == k)
                got = (s[i, cols] * wc[cols]).sum() / wc[cols].sum()
                self.assertAlmostEqual(got, xa[i, k], delta=0.06)


class TestMeasuredRealization(unittest.TestCase):
    def test_single_raised_pot_matches_published(self):
        """The measured model's range realization in a button-vs-big-blind single-raised pot on the charts'
        ranges sits in the published solver reports' band (in position 1.12-1.18, big blind 0.78-0.88;
        docs/preflop_research.md 2.1), and per class suited connectors out-realize offsuit kings."""
        from gto import preflop as pf
        from gto.pf_classes import to_classes
        m = G.measured_model()
        if m is None:
            self.skipTest("no gto/pf_eqr.py")
        line = [{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO")] + [
            {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"},
            {"pos": "BB", "action": "call"}]
        R = pf.replay_ranges(line)
        btn, bb = to_classes(R["BTN"]), to_classes(R["BB"])
        eq = equity_matrix()
        r_bb = G.realization(m, 1, True, "caller", btn)
        r_btn = G.realization(m, 1, True, "aggr", bb)
        from gto.pf_classes import PAIR_COMPAT
        e_bb = (PAIR_COMPAT * (1 - eq).T * btn[None, :]).sum(1) / (PAIR_COMPAT * btn[None, :]).sum(1)
        e_btn = (PAIR_COMPAT * eq * bb[None, :]).sum(1) / (PAIR_COMPAT * bb[None, :]).sum(1)
        agg = lambda r, e, w: float((w * r * e).sum() / (w * e).sum())
        self.assertTrue(0.75 <= agg(r_bb, e_bb, bb) <= 0.92, agg(r_bb, e_bb, bb))
        self.assertTrue(1.08 <= agg(r_btn, e_btn, btn) <= 1.22, agg(r_btn, e_btn, btn))
        i = {c: CLASSES.index(c) for c in ("76s", "65s", "K5o", "K7o")}
        self.assertGreater(min(r_bb[i["76s"]], r_bb[i["65s"]]), max(r_bb[i["K5o"]], r_bb[i["K7o"]]))


class TestRakeRerank(unittest.TestCase):
    def test_raked_calls_keep_playable_hands(self):
        """With rake the big blind's calls of a button open are cut by realized equity (gto/pf_eqr.py), not
        all-in equity: suited connectors stay, weak offsuit kings and queens go, and the total is the
        all-in-equity rule's."""
        from gto import preflop as pf
        from gto.advisor import preflop_strategy
        from gto.ranges import parse_range
        if G.measured_model() is None:
            self.skipTest("no gto/pf_eqr.py")
        line = [{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO")] + [
            {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}]
        T = pf.Table(pf.SEATS[6], rake=(0.05, 0.6))
        s = preflop_strategy("BB", line, T, 100)
        call = lambda h: float(s["call"][parse_range(h) > 0].mean())
        self.assertGreater(min(call("76s"), call("54s")), 0.3)
        self.assertLess(max(call("K5o"), call("Q7o"), call("J5s")), 0.1)
        self.assertAlmostEqual(float(s["call"].mean()), 0.317, delta=0.01)


if __name__ == "__main__":
    unittest.main()
