"""The one-size view (state "one_size"; Advisor._one_size_view, the page's Simple tab): a decision solved with one bet
size at it for each candidate and with every candidate to mix freely; the best size keeps the most EV for the player
betting, and each size's EV given up is the full menu's EV less its own, in % of the pot."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from gto import advisor as A
from gto.cards import COMBO_INDEX, cards
from gto.native import advisor
from gto.solver import N_FULL, Config, TreeBuilder, Ctx

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
X = lambda p: {"pos": p, "action": "check"}
# a river checked to the BB: flop and turn checked through, BTN vs BB single-raised (pot 5.5)
RIVER = dict(players=6, stack=100, hero="BB", hole="AhJh", board="Ks7d2cTh5s", preflop=PRE, flop=[X("BB"), X("BTN")],
             turn=[X("BB"), X("BTN")], river=[])


def rng(*hands):
    """A range (1326 weights) of these combos ("KhKd", ...)."""
    r = np.zeros(N_FULL)
    for h in hands:
        a, b = cards(h)
        r[COMBO_INDEX[(min(a, b), max(a, b))]] = 1.0
    return r


class TestCandidates(unittest.TestCase):
    def test_sizes(self):
        labels = lambda c: [lab for lab, _ in c]
        self.assertEqual(labels(A.one_size_candidates(10.0, 95.0, 2.5)), ["25%", "33%", "50%", "75%", "100%", "125%", "150%"])
        # the river twice the pot too
        self.assertEqual(labels(A.one_size_candidates(10.0, 95.0, 2.5, river=True))[-1], "200%")
        # a shallow stack (at most allin_spr pots behind): all-in, once, for every size that reaches the stack
        c = A.one_size_candidates(20.0, 25.0, 2.5)
        self.assertEqual(labels(c), ["25%", "33%", "50%", "75%", "100%", "all-in"])
        self.assertEqual(dict(c)["all-in"], 1.25)
        # a small pot: sizes that round to the same bet (half a big blind, at least 1) are one
        self.assertEqual(labels(A.one_size_candidates(2.0, 50.0, 2.5)), ["25%", "75%", "100%", "125%", "150%"])

    def test_summary(self):
        full = {"size": "full", "ev_bb": 10.0, "exploitability_pct": 0.1}
        rows = [{"size": "25%", "ev_bb": 9.0, "exploitability_pct": 0.1},
                {"size": "75%", "ev_bb": 9.96, "exploitability_pct": 0.1},
                {"size": "150%", "ev_bb": 9.97, "exploitability_pct": 0.1}]
        s = A.one_size_summary(full, rows, 20.0)
        self.assertEqual(s["best"], "150%")
        self.assertEqual([r["loss_pct"] for r in s["rows"]], [5.0, 0.2, 0.15])
        self.assertEqual(s["tied"], ["75%"])   # (0.01 bb apart: within 0.1% of the 20 bb pot)
        self.assertNotIn("loss_pct", rows[0])   # (the rows given aren't changed)
        s = A.one_size_summary(None, rows, 20.0)
        self.assertEqual((s["best"], [r["loss_pct"] for r in s["rows"]]), ("150%", [None, None, None]))
        self.assertEqual(A.one_size_summary(full, [], 20.0), {"rows": [], "best": None, "tied": []})


class TestRootBets(unittest.TestCase):
    def test_root_only(self):
        """Config.root_bets replaces the root's bets (and its low-SPR all-in) and nothing below it."""
        board = cards("Ks7d2cTh5s")
        r = rng("AhAd", "QhJc", "9s8s", "6h6d")
        cfg = Config(bet_sizes={"river": [0.33, 1.25]}, raise_sizes={"river": [0.75]})
        names = lambda c, pot, stack: TreeBuilder(Ctx(np.nonzero(r)[0], c), dead=list(board)).build(board, pot, stack)
        root = names(cfg, 10.0, 90.0)
        self.assertEqual(root.names, ["check", "bet 33%", "bet 125%"])
        one = cfg.copy()
        one.root_bets = [0.5]
        root1 = names(one, 10.0, 90.0)
        self.assertEqual(root1.names, ["check", "bet 50%"])
        self.assertEqual(root1.children[0].names, root.children[0].names)   # (the opponent's bets after a check)
        # shallow: the street adds all-in; root_bets says exactly what the root bets
        self.assertIn("all-in", names(cfg, 10.0, 20.0).names)
        self.assertEqual(names(one, 10.0, 20.0).names, ["check", "bet 50%"])
        one.root_bets = [A.ONE_SIZE_ALLIN]
        self.assertEqual(names(one, 10.0, 20.0).names, ["check", "all-in"])
        # its own cache key; the default config's key is as before
        self.assertNotEqual(one.sizes_key(), cfg.sizes_key())
        self.assertFalse(any(k and k[0] == "root_bets" for k in cfg.sizes_key() if isinstance(k, tuple)))


class TestCut(unittest.TestCase):
    """The view's speed-ups (README, One bet size: Speed): the full menu's solve is read at ONE_SIZE_CHECKS, its
    favourite sizes go first (one_size_order), and a size clearly behind the best stops early (_one_size_run's cut),
    none of which changes which size is best or tied."""
    BOARD = cards("KsQd7c4h2s")
    VALUE = ["KhKd", "KhKc", "KdKc"]
    AIR = [a + b for a in ("9s", "9h", "9d", "9c") for b in ("8s", "8h", "8d", "8c")]
    CATCH = [q + j for q in ("Qh", "Qc", "Qs") for j in ("Jh", "Jd", "Jc", "Js")]

    def solve(self, adv, sizes, cut=None, target=0.05):
        cfg = adv.cfg.copy()
        cfg.root_bets = list(sizes)
        cur = SimpleNamespace(pot_start=10.0, n_raises=0, name="river")
        return adv._one_size_solve(self.BOARD, rng(*self.VALUE, *self.AIR), rng(*self.CATCH), cur, 100.0, (0.0, 0.0), 0,
                                   cfg, target, cut=cut)

    def test_bounds_and_cut(self):
        adv = advisor("fast")
        s = self.solve(adv, [2.0])
        lo, hi = s.bounds
        row = adv._one_size_row(s, "200%", 0, (), 0.0)
        self.assertLessEqual(lo, row["ev_bb"] + 1e-3)
        self.assertGreaterEqual(hi, row["ev_bb"] - 1e-3)
        self.assertLess(hi - lo, 4 * row["exploitability_pct"] / 100 * 10.0 + 1e-3)
        # a quarter pot gives up ~7% of the pot here: against the 200% row it stops at the first read
        adv._one_rows = {(("spot",), "200%"): row}
        bar = adv._one_size_bar(("spot",), 0.05, 10.0)
        s25 = self.solve(adv, [0.25], cut=bar)
        self.assertEqual(s25.cut_at, A.HU_CHECKS[0])
        r25 = adv._one_size_row(s25, "25%", 0, (), 0.0)
        self.assertTrue(r25["cut"])
        self.assertNotIn("cut", row)
        self.assertGreater(r25["exploitability_pct"], 0.05)
        out = A.one_size_summary(None, [row, r25], 10.0)
        self.assertEqual((out["best"], out["tied"]), ("200%", []))
        self.assertNotIn("_lo", out["rows"][0])
        # 150% gives up 1% of the pot to 200% (3/19 of the pot x 2/3 less 1.5/2.5): behind by far more than the
        # target, it stops early too; a size as good as the best (the best's again) is never cut, and settles
        s150 = self.solve(adv, [1.5], cut=bar)
        self.assertIsNotNone(s150.cut_at)
        again = self.solve(adv, [2.0], cut=bar)
        self.assertIsNone(getattr(again, "cut_at", None))
        self.assertEqual(again.settled_at, 0.05)

    def test_order(self):
        labels = ["25%", "33%", "50%", "75%", "100%", "125%", "150%"]
        full = {"actions": ["check", "bet 25%", "bet 50%", "bet 150%"], "freq": [0.6, 0.0, 0.15, 0.25]}
        self.assertEqual(A.one_size_order(labels, full), ["150%", "50%", "125%", "100%", "75%", "33%", "25%"])
        self.assertEqual(A.one_size_order(["25%", "75%", "all-in"], {"actions": ["check", "all-in"], "freq": [0.5, 0.5]}),
                         ["all-in", "75%", "25%"])
        self.assertEqual(A.one_size_order(labels, {"actions": ["check"], "freq": [1.0]}), labels)


class TestKnownAnswer(unittest.TestCase):
    def test_polar_river(self):
        """The nuts or nothing against bluff-catchers on the river: the bettor's value grows with the bet (value
        V/(V+B) of the range wins pot x (1 + s/(1+s)) at bet s, the bluffs break even), so the best single size is the
        biggest, each size's EV is that formula's, and what a size gives up against the full menu is the formula's
        difference."""
        adv = advisor("fast")
        board = cards("KsQd7c4h2s")   # (no straight or flush can come: KK is the nuts)
        value = ["KhKd", "KhKc", "KdKc"]
        air = [a + b for a in ("9s", "9h", "9d", "9c") for b in ("8s", "8h", "8d", "8c")]
        R0, R1 = rng(*value, *air), rng(*[q + j for q in ("Qh", "Qc", "Qs") for j in ("Jh", "Jd", "Jc", "Js")])
        pot, stack = 10.0, 100.0
        cur = SimpleNamespace(pot_start=pot, n_raises=0, name="river")
        cands = A.one_size_candidates(pot, stack, 2.5, river=True)
        frac = dict(cands)
        rows, full = [], None
        for label in ["full"] + [c for c, _ in cands]:
            cfg = adv.cfg.copy()
            cfg.root_bets = [f for _, f in cands] if label == "full" else [frac[label]]
            sol = adv._one_size_solve(board, R0, R1, cur, stack, (0.0, 0.0), 0, cfg, 0.02)
            row = adv._one_size_row(sol, label, 0, (), 0.0)
            if label == "full":
                full = row
            else:
                rows.append(row)
                s = max(round(frac[label] * pot * 2) / 2, 1.0) / pot   # (the bet as the tree sizes it: 33% is 3.5)
                want = 3 / 19 * pot * (1 + s / (1 + s))
                self.assertAlmostEqual(row["ev_bb"], want, delta=0.02 * pot / 100 + 0.005, msg=label)
        out = A.one_size_summary(full, rows, pot)
        self.assertEqual(out["best"], "200%")
        loss = {r["size"]: r["loss_pct"] for r in out["rows"]}
        self.assertLess(abs(loss["200%"]), 0.05)
        self.assertAlmostEqual(loss["25%"], 3 / 19 * 100 * (2 / 3 - 0.2), delta=0.1)
        self.assertEqual(sorted(loss, key=loss.get, reverse=True), [c for c, _ in cands])   # (smaller gives up more)
        # the full menu bets the biggest size: all its value and as many bluffs as that size allows (2/3 of value)
        f = dict(zip(full["actions"], full["freq"]))
        self.assertAlmostEqual(f["bet 200%"], (3 + 2) / 19, delta=0.02)


class TestView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = advisor("fast")

    def ask(self, **kw):
        return self.adv.advise(dict(RIVER, **kw))

    def test_rows_and_cache(self):
        self.adv._one_rows.clear()   # (the other tests' rows)
        solves = []
        orig = A.Advisor._one_size_solve

        def counted(adv, *a, **k):
            solves.append(a[7].root_bets)
            return orig(adv, *a, **k)

        with mock.patch.object(A.Advisor, "_one_size_solve", counted):
            r = self.ask(one_size="list")
            o = r["one_size"]
            self.assertEqual(solves, [])
            self.assertEqual((o["street"], o["actor"], o["opponent"], o["position"]), ("river", "BB", "BTN", "OOP"))
            self.assertEqual(o["candidates"], ["25%", "33%", "50%", "75%", "100%", "125%", "150%", "200%"])
            self.assertEqual((o["full"], o["rows"], o["best"]), (None, [], None))
            self.assertIn("EV given up", o["definition"])
            o = self.ask(one_size="full")["one_size"]
            self.assertEqual(len(solves), 1)
            self.assertEqual(len(solves[0]), 8)
            self.assertEqual(o["full"]["actions"][0], "check")
            self.assertEqual(len(o["full"]["actions"]), 9)
            o = self.ask(one_size="50%")["one_size"]
            self.assertEqual(solves[-1], [0.5])
            self.assertEqual([x["size"] for x in o["rows"]], ["50%"])
            row = o["rows"][0]
            self.assertEqual(row["actions"], ["check", "bet 50%"])
            self.assertAlmostEqual(sum(row["freq"]), 1.0, delta=0.01)
            self.assertAlmostEqual(row["loss_pct"], (o["full"]["ev_bb"] - row["ev_bb"]) / o["pot_bb"] * 100, delta=0.002)
            self.assertEqual(o["hand"], "AJs")
            self.assertEqual(len(row["hand"]["freq"]), 2)
            self.assertEqual(set(row["grid"]["actions"]), {"check", "bet 50%"})
            # asked again: answered from what's kept, nothing solved; "all" solves only the sizes still missing
            n = len(solves)
            again = self.ask(one_size="50%")["one_size"]
            self.assertEqual(len(solves), n)
            self.assertEqual(again["rows"], o["rows"])
            o = self.ask(one_size="all")["one_size"]
            self.assertEqual(len(solves), n + 7)
            self.assertEqual([x["size"] for x in o["rows"]], o["candidates"])
            self.assertIn(o["best"], o["candidates"])
            best = next(x for x in o["rows"] if x["size"] == o["best"])
            # (the best gives up the least; nothing gives up much less than nothing, the solves' precision aside)
            self.assertEqual(best["loss_pct"], min(x["loss_pct"] for x in o["rows"]))
            self.assertGreater(min(x["loss_pct"] for x in o["rows"]), -2 * o["target_pct"])
            self.ask(one_size="all")
            self.assertEqual(len(solves), n + 7)
            # a size that isn't a candidate is solved too (and listed after them)
            o = self.ask(one_size=60)["one_size"]
            self.assertEqual(solves[-1], [0.6])
            self.assertEqual(o["rows"][-1]["size"], "60%")

    def test_default_answer_unchanged(self):
        """Without one_size the answer is the advice, as before: the view leaves no state behind."""
        a = self.ask()
        self.ask(one_size="25%")
        b = self.ask()
        self.assertNotIn("one_size", a)
        for k in ("recommended", "actions", "grid", "street"):
            self.assertEqual(a[k], b[k])

    def test_study(self):
        r = self.adv.advise({k: v for k, v in RIVER.items() if k not in ("hero", "hole")} | {"study": True,
                                                                                             "one_size": "25%"})
        o = r["one_size"]
        self.assertEqual(o["actor"], "BB")
        self.assertNotIn("hand", o)
        self.assertNotIn("hand", o["rows"][0])

    def test_refused(self):
        bet = dict(river=[{"pos": "BB", "action": "bet", "amount": 3}], hero="BTN", hole="AsQs")
        cases = {
            "preflop": dict(board="", flop=[], turn=[], preflop=PRE[:3], hero="BTN", hole="AsQs"),
            "face": bet,
            "pool": dict(opponents="pool"),
            "lock": dict(locks={"BTN": {"fold": 15}}),
            "multiway": dict(preflop=PRE[:4] + [{"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}],
                             flop=[X("SB"), X("BB"), X("BTN")], turn=[X("SB"), X("BB"), X("BTN")], hero="SB",
                             hole="AhJh"),
            "size": dict(one_size="huge"),
            "range": dict(one_size=900),
        }
        for name, kw in cases.items():
            r = self.ask(**{"one_size": "all", **kw})
            self.assertIn("error", r, name)
            self.assertNotIn("one_size", r, name)
            self.assertNotIn("recommended", r, name)   # (not the betting rules' advice in its place)


if __name__ == "__main__":
    unittest.main()
