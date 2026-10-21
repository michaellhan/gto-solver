"""Turn and river solves against reference solves, published solver figures and river theory.

The spots (scripts/turn_river_reference.py SPOTS) are BTN vs BB single-raised pots and a CO vs BTN 3bet pot after a flop
(and for river spots a turn) line, their ranges narrowed by long solves of those streets (tests/data/turn_river_spots.npz).
The references (tests/data/turn_river_refs.json) are this solver's every-river trees run to convergence: the rich menu
(turn and river bets of 33%, 75% and 125% pot, more on a river root), the one-size tree (75%) and native Accurate's.
"""
import json
import os
import unittest

import numpy as np

from gto.solver import Config, solve, lift_classes, _class_closure, Ctx, COMBO_TABLE, CA_FULL, CB_FULL
from scripts import turn_river_reference as T

REFS = json.load(open(T.REFS_FILE))


def one_size(frac):
    return dict(bet_sizes={s: [frac] for s in ("flop", "turn", "river")},
                raise_sizes={s: [0.75] for s in ("flop", "turn", "river")}, max_raises=1)


def facing(sol, bet=1):
    """The river's check-bet line: (OOP's reach after checking, IP's after betting, IP's node, OOP's node facing it)."""
    root = sol.root
    ipn = root.children[0]
    rc = sol.r0 * root.avg_strategy0()[0]
    rb = sol.r1 * ipn.avg_strategy0()[bet]
    return rc, rb, ipn, ipn.children[bet]


class TestRiverTheory(unittest.TestCase):
    """Closed forms that hold on river spots at equilibrium, checked on converged solves of real ranges.

    A bet of b x pot: the caller risks b to win 1 + b, so a pure bluff-catcher (beats every bluff, loses to all value)
    is indifferent when the bettor's bets hold a bluff share x with x (1 + b) = (1 - x) b, i.e. x = b / (1 + 2b)
    (a quarter at half pot, a third at pot, 40% at twice pot). A bluff with no showdown value wins the pot when the opponent folds and loses b
    when called, so it is indifferent when the defender folds b / (1 + b) of the range facing the bet: it defends
    MDF = 1 / (1 + b). Both hold exactly in the toy nuts-or-air game (tests/test_gto.py test_polarized_river_game); on
    a polar spot (BTN bet the flop and turn, BB check-called both: the BB capped, the BTN's range sets, two pair and
    air) they hold approximately: a bluff's alternative is checking (worth its showdown value, not 0) and hands sit
    between value and air. Where both ranges are condensed (the flop and turn checked through) the BB defends far
    less than MDF: the BTN's bluffs give up showdown value, and GTO Wizard finds the out-of-position defender folding
    more than MDF in general (blog.gtowizard.com/mdf-alpha: "consistently over-folds", e.g. QQ3 a 33% c-bet, MDF 75%,
    folds almost half). And at any equilibrium, a hand mixing two actions is indifferent between them."""

    _sols = {}

    @classmethod
    def sol(cls, spot, frac):
        key = (spot, frac)
        if key not in cls._sols:
            cls._sols[key] = T.solve_spot(spot, one_size(frac), 1500)
        return cls._sols[key]

    def stats(self, spot, frac):
        sol = self.sol(spot, frac)
        rc, rb, ipn, vs = facing(sol)
        sv = vs.avg_strategy0()
        b = vs.bets[1] / ipn.pot_now
        call = rc * sv[1]
        M = sol.ctx.showdown_matrix(tuple(sol.root.board)).astype(np.float64)
        eq = (M @ call + sol.ctx.compat @ call) / 2 / np.maximum(sol.ctx.compat @ call, 1e-12)
        bluff = float((rb * (eq < 0.5)).sum() / rb.sum())          # bets that lose to the calling range
        defend = float(1 - (sv[0] * rc).sum() / rc.sum())
        return sol, b, bluff, defend

    def test_polar_bluff_share_and_mdf(self):
        # r-k72-3-5 (Ks7d2c 3h 5c, bet 33% / call, bet 75% / call): measured 25.4 / 28.1 / 30.9 / 34.5% bluffs against
        # 25.3 / 29.9 / 35.8 / 40.0, defending 65.0 / 56.3 / 43.5 / 31.9 against MDF 66.2 / 57.3 / 44.3 / 33.3
        for spot in ("r-k72-3-5", "r-k72s-4-9s"):
            for frac in (0.5, 1.25, 2.0):
                _, b, bluff, defend = self.stats(spot, frac)
                self.assertLess(abs(bluff - b / (1 + 2 * b)), 0.065, f"{spot} {frac}: bluffs {bluff:.3f}")
                self.assertLess(abs(defend - 1 / (1 + b)), 0.03, f"{spot} {frac}: defends {defend:.3f}")
                # bigger bets: more bluffs in the bets, less defence
        self.assertLess(self.stats("r-k72-3-5", 0.5)[2], self.stats("r-k72-3-5", 2.0)[2])

    def test_condensed_defender_folds_more_than_mdf(self):
        # r-q83x-5-j (Qs8d3c 5h Jc, checked through): BB defends 37 / 30 / 25% against MDF 65 / 44 / 33
        for frac in (0.5, 1.25, 2.0):
            _, b, bluff, defend = self.stats("r-q83x-5-j", frac)
            self.assertLess(defend, 1 / (1 + b) - 0.07, f"{frac}: defends {defend:.3f}")
            self.assertLess(abs(bluff - b / (1 + 2 * b)), 0.065)

    def test_mixed_hands_are_indifferent(self):
        # a bluff-catcher mixing call and fold, and a losing hand mixing bet and check, gain nothing by either
        # (measured: at most 0.1% of the pot on the polar spot, 0.5% on the condensed one at 1500 iterations)
        for spot in ("r-k72-3-5", "r-q83x-5-j", "r-k72-3x-9"):
            for frac in (0.5, 2.0):
                sol, b, _, _ = self.stats(spot, frac)
                rc, rb, ipn, vs = facing(sol)
                sv, s1 = vs.avg_strategy0(), ipn.avg_strategy0()
                ev = sol.solver.node_action_evs(vs, rc, rb)
                mix = (sv[1] > 0.1) & (sv[0] > 0.1) & (rc > 1e-3 * rc.max())
                self.assertTrue(mix.any())
                self.assertLess(np.abs(ev[1] - ev[0])[mix].max() / ipn.pot_now, 0.01, f"{spot} {frac} catchers")
                ev1 = sol.solver.node_action_evs(ipn, rc, sol.r1)
                mixb = (s1[1] > 0.1) & (s1[0] > 0.1) & (sol.r1 > 1e-3 * sol.r1.max())
                self.assertLess(np.abs(ev1[1] - ev1[0])[mixb].max() / ipn.pot_now, 0.01, f"{spot} {frac} bettor")


class TestRiverClasses(unittest.TestCase):
    """Config.river_classes: the turn root's river as one card per class, each class standing for its cards by swaps."""

    def test_swaps_and_closure(self):
        board, r0, r1, pot, stack = T.load_spot("t-k72s-4s")
        bm = np.prod([1.0 - np.eye(52)[c][CA_FULL] - np.eye(52)[c][CB_FULL] for c in board], axis=0) > 0
        active = np.flatnonzero(((r0 > 0) | (r1 > 0)) & bm)
        ctx0 = Ctx(active, Config())
        cls = ctx0.turn_classes(tuple(board), set(board))
        self.assertEqual(sorted(c for _, cs in cls for c in cs), [c for c in range(52) if c not in board])
        closed = _class_closure(active, cls)
        self.assertTrue(set(active) <= set(closed))
        ctx = Ctx(closed, Config(river_classes=True))
        G, flat, starts, sizes = ctx.class_perm(cls)
        n = ctx.n
        for e in range(len(G)):
            g = G[e]
            # every active combo's image is in the space, and a swap is an involution
            ok = g < n
            self.assertTrue(ok[np.isin(closed, active)].all())
            self.assertTrue((g[g[ok]] == np.flatnonzero(ok)).all())
        # the played card's own row is the identity
        for k, (played, cs) in enumerate(cls):
            self.assertTrue((G[starts[k] + list(cs).index(played)] == np.arange(n)).all())

    def test_close_to_every_river(self):
        # same iterations: the class tree's frequencies within a point of every river's, its strategy (lifted into the
        # every-river tree) about as far from equilibrium there as every river's own, and the game zero-sum.
        # Measured at 100 iterations: exploitability 0.37 / 0.44 lifted / 0.37 every river on K72-3 (13 classes, 2-2.4x
        # faster), 0.28 / 0.35 / 0.28 on K72s-4s (22 classes, 1.4-1.7x)
        for spot in ("t-k72-3", "t-k72s-4s"):
            kw = dict(T.ONE)
            cls = T.solve_spot(spot, dict(kw, river_classes=True), 60)
            full = T.solve_spot(spot, kw, 60)
            a, b = T.stats(cls), T.stats(full)
            for k in T.compared(b):
                self.assertLess(abs(a.get(k, 0) - b[k]), 1.0, f"{spot} {k}: {a.get(k, 0):.1f} classes, {b[k]:.1f} every river")
            s = cls.solver
            x, y = cls.r0[None, :].astype(np.float32), cls.r1[None, :].astype(np.float32)
            v0 = float((s._cfr(cls.root, 0, x, y, update=False, use_avg=True)[0] * cls.r0).sum())
            v1 = float((s._cfr(cls.root, 1, x, y, update=False, use_avg=True)[0] * cls.r1).sum())
            self.assertLess(abs(v0 + v1) / float((s.compat_sum(y)[0] * cls.r0).sum()), 1e-4)
            lifted = T.solve_spot(spot, kw, 0)
            lift_classes(lifted, cls)
            e_cls, e_lift, e_full = cls.exploitability(), lifted.exploitability(), full.exploitability()
            self.assertLess(abs(e_cls - e_full), 0.3, spot)
            self.assertLess(e_lift - e_full, 0.3, f"{spot}: lifted {e_lift:.2f}, every river {e_full:.2f}")


if __name__ == "__main__":
    unittest.main()
