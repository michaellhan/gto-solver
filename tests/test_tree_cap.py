"""A new heads-up solve's tree cut to fit (advisor.MAX_CELLS, TRIMS): under Advisor.max_cells (memory) and, on the
turn and river, what runs CAP_ITERS iterations in the budget at Advisor.cell_s; and the first iteration's
bailout (solver.TooSlow) where this device turns out slower than cell_s."""
import unittest
from unittest import mock

from gto import advisor as A
from gto import core, native
from gto.cards import cards
from gto.ranges import parse_range
from gto.solver import BucketEntry, Chance, Decision, TooSlow, TreeTooBig, solve, tree_cells

OOP_T = parse_range("66+,A9s+,KTs+,QTs+,JTs,T9s,98s,AJo+,KQo")
IP_T = parse_range("22+,A2s+,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,ATo+,KJo+,QJo")
F = lambda p: {"pos": p, "action": "fold"}
C = lambda p: {"pos": p, "action": "call"}
X = lambda p: {"pos": p, "action": "check"}
R = lambda p, to: {"pos": p, "action": "raise", "to": to}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}
SRP = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), C("BB")]
TURN = {"players": 6, "stack": 100, "preflop": SRP, "flop": [X("BB"), B("BTN", 1.8), C("BB")], "turn": [], "river": [],
        "hero": "BB", "hole": "QhJh", "board": "Ks7d2cTh"}


def regret_cells(sol):
    seen, out, todo = set(), 0, [sol.root]
    while todo:
        n = todo.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Decision):
            out += n.regret.size
            todo.extend(n.children)
        elif isinstance(n, (Chance, BucketEntry)):
            todo.append(n.child)
    return out


def turn_solve(**kw):
    return solve(cards("Ks7d2cTh"), OOP_T, IP_T, 20, 90, cfg=native.config("fast"), time_budget=1e9, **kw)


class TestTreeCap(unittest.TestCase):
    def test_cells_are_the_trees_regrets(self):
        """TreeBuilder.cells counts each decision's boards x actions x width, the regrets' size, and tree_cells the
        same without building the tree; past max_cells either stops (TreeTooBig) with the count so far."""
        sol = turn_solve(max_iters=0)
        self.assertEqual(sol.cells, regret_cells(sol))
        cfg = native.config("fast")
        self.assertEqual(tree_cells(cards("Ks7d2cTh"), OOP_T, IP_T, 20, 90, cfg=cfg), sol.cells)
        flop = solve(cards("Qs8d5c"), OOP_T, IP_T, 5.5, 97.5, cfg=cfg, time_budget=1e9, max_iters=0)
        self.assertEqual(tree_cells(cards("Qs8d5c"), OOP_T, IP_T, 5.5, 97.5, cfg=cfg), flop.cells)
        self.assertEqual(flop.cells, regret_cells(flop))
        with self.assertRaises(TreeTooBig):
            tree_cells(cards("Ks7d2cTh"), OOP_T, IP_T, 20, 90, cfg=cfg, max_cells=sol.cells - 1)
        self.assertIsNotNone(turn_solve(max_iters=0, max_cells=sol.cells))
        with self.assertRaises(TreeTooBig) as e:
            turn_solve(max_iters=0, max_cells=sol.cells // 2)
        self.assertGreater(e.exception.cells, sol.cells // 2)
        self.assertLessEqual(e.exception.cells, sol.cells)

    def test_too_slow_after_the_first_iteration(self):
        """max_per: the first iteration over it raises TooSlow with its time and the tree's cells, in the compiled
        core and in numpy; under it the solve runs on."""
        for use_core in ((True, False) if core.lib() is not None else (False,)):
            with self.subTest(core=use_core), \
                    mock.patch.object(core, "lib", return_value=core.lib() if use_core else None):
                with self.assertRaises(TooSlow) as e:
                    turn_solve(max_iters=5, max_per=0.0)
                full = turn_solve(max_iters=5, max_per=1e9)
                self.assertEqual(full.iters, 5)
                self.assertEqual(e.exception.cells, full.cells)
                self.assertGreater(e.exception.per, 0.0)

    def test_trim_levels(self):
        """Each level cuts more (TRIMS): every street's raises to one size, then the later streets' bets, then this
        street's, then one raise a street. The size kept is the one nearest 75% pot, or the hand's own this street
        (Config.keep_sizes); the config passed in is left as it was."""
        cfg = native.config("fast")
        cfg.raise_sizes["turn"] = [0.5, 0.66, 1.5]
        cfg.bet_sizes["turn"] = [0.33, 0.75, 1.25]
        cfg.bet_sizes["river"] = [0.33, 1.25]
        cfg.max_raises = 3
        lv = [A.Advisor._trim_cfg(cfg, "turn", k) for k in range(1, len(A.TRIMS) + 1)]
        self.assertEqual([c.raise_sizes["turn"] for c in lv], [[0.66]] * 4)
        self.assertEqual([c.bet_sizes["river"] for c in lv], [[0.33, 1.25], [0.33], [0.33], [0.33]])
        self.assertEqual([c.bet_sizes["turn"] for c in lv], [[0.33, 0.75, 1.25]] * 2 + [[0.75]] * 2)
        self.assertEqual([c.max_raises for c in lv], [3, 3, 3, 1])
        self.assertEqual(cfg.raise_sizes["turn"], [0.5, 0.66, 1.5])
        cfg.keep_sizes = {"bet": [0.33], "raise": [1.5], "aggr": 2}
        lv = [A.Advisor._trim_cfg(cfg, "turn", k) for k in range(1, len(A.TRIMS) + 1)]
        self.assertEqual([c.raise_sizes["turn"] for c in lv], [[1.5]] * 4)
        self.assertEqual([c.bet_sizes["turn"] for c in lv], [[0.33, 0.75, 1.25]] * 2 + [[0.33]] * 2)
        self.assertEqual(lv[-1].max_raises, 2)

    def test_cap_from_memory_and_time(self):
        """max_cells everywhere; on the turn and river also the tree whose CAP_ITERS iterations at cell_s fit the
        budget."""
        adv = native.advisor("fast")
        adv.max_cells, adv.cell_s = 5_000_000, 1e-8 / A.CAP_ITERS * 10
        self.assertEqual(adv._cells_cap(3, 0.1), 5_000_000)
        self.assertEqual(adv._cells_cap(4, 0.1), 1_000_000)
        self.assertEqual(adv._cells_cap(5, 10.0), 5_000_000)
        adv.max_cells = None
        self.assertEqual(adv._cells_cap(4, 2.0), 20_000_000)
        self.assertIsNone(adv._cells_cap(3, 2.0))

    def test_a_tree_past_the_cap_is_cut_and_noted(self):
        """The turn's tree over max_cells is solved cut (the first level under it), within the cap, and the answer
        says so; under it nothing changes."""
        adv = native.advisor("fast", budget=0.5, prior_budget=0.2)
        adv.cell_s = 1e-12   # (memory alone: the time cap has its own test)
        full = adv.advise(TURN)
        cells = full["solver"]["cells"]
        self.assertFalse(any("smaller game" in n for n in full["notes"]))
        adv = native.advisor("fast", budget=0.5, prior_budget=0.2)
        adv.max_cells, adv.cell_s = cells - 1, 1e-12
        res = adv.advise(TURN)
        self.assertLessEqual(res["solver"]["cells"], cells - 1)
        self.assertTrue(any(n.startswith("turn: solved in a smaller game than the full one") for n in res["notes"]),
                        res["notes"])
        self.assertTrue(res["recommended"])

    def test_too_slow_device_recalibrates_and_cuts(self):
        """A first iteration that says the floor's 10 would take SLOW x the budget: cell_s is set from it and the tree
        cut to fit the rest of the budget (the smallest level whatever its size, when none does)."""
        adv = native.advisor("fast", budget=1.0, prior_budget=0.2)
        adv.cell_s = 1e-12
        calls = []
        real = A.solve

        def slow_once(board, *a, **kw):
            calls.append([len(board), kw.get("max_cells"), kw.get("max_per"), kw["cfg"], None])
            if len(board) == 4 and [c[0] for c in calls].count(4) == 1:
                e = TooSlow(kw["max_per"] * 4)
                e.cells = 10_000_000
                raise e
            sol = real(board, *a, **kw)
            calls[-1][4] = sol.cells
            return sol
        with mock.patch.object(A, "solve", slow_once):
            res = adv.advise(TURN)
        turn = [c for c in calls if c[0] == 4]
        self.assertGreaterEqual(len(turn), 2)
        self.assertIsNotNone(turn[0][2])
        self.assertIsNone(turn[1][2])          # (one bailout a solve)
        self.assertLess(turn[1][4], 10_000_000)
        self.assertEqual(len(turn[1][3].raise_sizes["turn"]), 1)
        self.assertTrue(any("smaller game" in n for n in res["notes"]), res["notes"])
        self.assertGreater(adv.cell_s, 1e-9)


if __name__ == "__main__":
    unittest.main()
