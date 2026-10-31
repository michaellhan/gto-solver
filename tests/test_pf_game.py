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
        # the 5bet is all-in, and at most three players see a flop (two in a raised pot; three all-in)
        self.assertTrue(all(len(t.active) <= 3 for t in nodes if t.kind == "show"))
        self.assertTrue(all(len(t.active) == 2 for t in nodes if t.kind == "show" and t.n_raises > 1 and not t.allin))
        # facing a cold jam and a call a 3bettor may call too (a three-way all-in), not only fold
        g40 = G.Game(G.build(G.SIX, stack=40.0), G.SIX, G.flat_model(), stack=40.0)
        nd = g40.find([("BTN", "raise"), ("SB", "raise"), ("BB", "all-in"), ("BTN", "call")])
        self.assertEqual((g40.positions[nd.player], nd.names), ("SB", ["fold", "call"]))
        self.assertEqual(len(nd.children[1].active), 3)
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


class TestTreeOptions(unittest.TestCase):
    def test_small_blind_limp(self):
        """sb_limp: the small blind first in completes, the big blind checks or raises to Sizes.iso, the small
        blind re-raises Sizes.limp_3bet times that; no limp once anyone has opened, and heads-up too."""
        g = G.Game(G.build(sb_limp=True), G.SIX, G.flat_model())
        nd = g.root
        while g.positions[nd.player] != "SB":
            nd = nd.children[nd.names.index("fold")]
        self.assertEqual(nd.names, ["fold", "call", "raise"])
        bb = nd.children[1]
        self.assertEqual(bb.names, ["call", "raise"])
        self.assertEqual(bb.children[0].kind, "show")
        self.assertEqual(bb.children[0].inv.sum(), 2.0)
        self.assertEqual(bb.children[1].line[-1][2], 3.5)
        self.assertEqual(bb.children[1].children[2].line[-1][2], 10.5)
        self.assertEqual(g.find([("CO", "raise"), ("SB", None)][:1]).names, ["fold", "call", "raise"])
        h = G.Game(G.build(["SB", "BB"], sb_limp=True), ["SB", "BB"], G.flat_model())
        h.solve(200)
        self.assertLess(h.exploitability(), 0.02)

    def test_four_bet_by_position(self):
        sz = G.Sizes(four_x_ip=2.0, four_x_oop=2.7)
        g = G.Game(G.build(sizes=sz), G.SIX, G.flat_model())
        self.assertAlmostEqual(g.find([("CO", "raise"), ("BTN", "raise"), ("CO", "raise")]).line[-1][2], 7.5 * 2.7)
        self.assertAlmostEqual(g.find([("BTN", "raise"), ("BB", "raise"), ("BTN", "raise")]).line[-1][2], 24.0)

    def test_spot_model_picks_the_seats(self):
        """SpotModel: a flop between measured seats uses their model, else one with the same caller, else
        the fallback."""
        a = G.EQRModel(share_fn=lambda key: np.full((len(CLASSES), len(CLASSES)), 0.1))
        b = G.EQRModel(share_fn=lambda key: np.full((len(CLASSES), len(CLASSES)), 0.2))
        f = G.EQRModel(share_fn=lambda key: np.full((len(CLASSES), len(CLASSES)), 0.3))
        m = G.SpotModel({(1, True, "BTN", "BB"): a, (1, True, "CO", "SB"): b}, f)
        self.assertEqual(m.share((1, True, "BTN", "BB"))[0, 0], 0.1)
        self.assertEqual(m.share((1, True, "UTG", "SB"))[0, 0], 0.2)
        self.assertEqual(m.share((1, True, "UTG", "BTN"))[0, 0], 0.3)
        self.assertEqual(m.share((2, True, "BTN", "BB"))[0, 0], 0.3)


class TestGames(unittest.TestCase):
    """Other games (gto/pf_formats.py's solves): antes, a straddle, 7-9 handed tables, other depths."""

    def test_antes(self):
        """An ante is dead money: in every terminal's chips (folded seats lose it), out of the stack before the
        betting; the sizes are the same."""
        g = G.Game(G.build(ante=0.125), G.SIX, G.flat_model())
        walk = g.find([("UTG", "fold"), ("HJ", "fold"), ("CO", "fold"), ("BTN", "fold"), ("SB", "fold")])
        self.assertEqual(walk.kind, "fold")
        self.assertAlmostEqual(walk.inv.sum(), 1.5 + 6 * 0.125)
        self.assertEqual(g.find([("CO", "raise")]).line[-1][2], 2.5)
        jam = [t for t in G.walk(g.root) if t.kind == "show" and t.allin]
        self.assertTrue(jam and all(abs(t.inv.max() - 100.0) < 1e-9 for t in jam))
        g.solve(30)
        # the antes come back out of the pots: off zero only by card removal, as without them (0.102 and
        # 0.081 here, three-way all-ins' shares normalised over the ranges; antes kept would be 0.75)
        g0 = G.Game(G.build(), G.SIX, G.flat_model())
        g0.solve(30)
        self.assertLess(abs(float(g.values().sum()) - float(g0.values().sum())), 0.05)

    def test_straddle(self):
        """The first seat posts 2bb and acts last; opens are 2.5 straddles; the big blind can fold."""
        g = G.Game(G.build(straddle=True), G.SIX, G.flat_model())
        self.assertEqual(g.positions[g.root.player], "HJ")
        nd = g.find([("HJ", "raise")])
        self.assertEqual(nd.line[-1][2], 5.0)
        last = g.find([("HJ", "raise"), ("BB", "fold")])
        self.assertEqual(g.positions[last.player], "UTG")
        self.assertEqual(last.names, ["fold", "call", "raise"])
        # folded to the big blind, it faces the straddle's 2bb: fold or raise (it used to have only the raise)
        bb = g.find([("HJ", "fold"), ("CO", "fold"), ("BTN", "fold"), ("SB", "fold")])
        self.assertEqual((g.positions[bb.player], bb.names), ("BB", ["fold", "raise"]))
        # unstraddled, the big blind's walk is a terminal and checking behind a limp is free
        g = G.Game(G.build(sb_limp=True), G.SIX, G.flat_model())
        self.assertEqual(g.find([("SB", "call")]).names, ["call", "raise"])

    def test_seats_and_depths(self):
        """7-9 handed seats act postflop after the blinds in seating order; shallow, raises that put in 40% of the
        stack are all-in."""
        from gto.preflop import SEATS
        self.assertEqual(G.post_rank(G.SIX), [G.POST_ORDER.index(p) for p in G.SIX])
        self.assertEqual(G.post_rank(SEATS[8]), [2, 3, 4, 5, 6, 7, 0, 1])
        self.assertEqual(G.post_rank(["SB", "BB"]), [1, 0])
        g = G.Game(G.build(SEATS[8], max_players=2), SEATS[8], G.flat_model())
        self.assertEqual(g.find([("UTG1", "raise")]).line[-1][2], 2.5)
        g = G.Game(G.build(stack=20), G.SIX, G.flat_model(), stack=20)
        self.assertEqual(g.find([("CO", "raise"), ("BB", "all-in")]).line[-1][2], 20.0)
        self.assertEqual(g.find([("CO", "raise"), ("BTN", "raise")]).line[-1][2], 7.5)

    def test_library_model_depth_scale(self):
        """LibraryModel's scale plays another depth: each share's excess over equity times k."""
        m = G.library_model()
        if m is None:
            self.skipTest("no gto/pf_libvals.py")
        k = (1, True, "BTN", "BB")
        eq = equity_matrix()
        s = G.LibraryModel(scale={1: 0.5}).share(k)
        self.assertLess(np.abs(s - (eq + 0.5 * (m.share(k) - eq))).max(), 1e-9)
        self.assertLess(np.abs(G.LibraryModel(scale={2: 0.5}).share(k) - m.share(k)).max(), 1e-12)


class TestSolve(unittest.TestCase):
    def test_masses(self):
        """Game._masses (each seat's reach times PAIR_COMPAT, worked once a node for the seat whose reach changes
        there and carried down for the others) is the product at every node."""
        pos = ["CO", "BTN", "SB", "BB"]
        g = G.Game(G.build(pos, max_players=3), pos, G.flat_model())
        g.solve(3)
        reach = g._down(g.current())
        np.testing.assert_allclose(g._masses(reach), reach @ G.PAIR_COMPAT, rtol=1e-12, atol=1e-15)

    def test_average_power(self):
        """Game.avg_power weighs iteration t's strategy (times its own reach) by t ** avg_power in the average: the
        sums equal the strategies and reaches of a run replayed by hand. pf_study's solves average with t^3 and get
        nearer equilibrium than linear averaging in the same iterations."""
        from gto import pf_study as PS
        pos = ["SB", "BB"]
        for power in (1.0, 3.0):
            g = G.Game(G.build(pos), pos, G.flat_model())
            g.avg_power = power
            want = np.zeros_like(g.ssum)
            for t in (1, 2, 3):
                s = g.current()
                reach = g._down(s)
                for i, nd in enumerate(g.dec):
                    want[i] += t ** power * reach[nd, g.player[i]][None, :] * s[i]
                g.iterate(t)
            np.testing.assert_allclose(g.ssum, want, rtol=1e-12, atol=1e-15)
        self.assertEqual(PS.Spec(players=2).game().avg_power, PS.AVG_POWER)
        expl = {}
        for power in (1.0, PS.AVG_POWER):
            g = G.Game(G.build(pos), pos, G.flat_model())
            g.avg_power = power
            expl[power] = g.solve(60).exploitability()
        self.assertLess(expl[PS.AVG_POWER], expl[1.0])

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
        # (J5s, a suited non-connector, ranks with the suited hands since they carry the river correction's
        # +0.10: kept 69% at 0.6bb, 50% at 4bb)
        self.assertLess(max(call("K5o"), call("Q7o"), call("K7o"), call("K8o")), 0.1)
        self.assertAlmostEqual(float(s["call"].mean()), 0.317, delta=0.01)


if __name__ == "__main__":
    unittest.main()
