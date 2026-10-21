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

with open(T.REFS_FILE) as _f:
    REFS = json.load(_f)


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


# Published turn and river figures (6-max cash 100bb unless noted) next to this solver's richest trees. (spot, statistic,
# low, high, source); statistics as T.stats: "oop bet" the out-of-position player's leads at the root, "ip bet" the in-
# position player's bets once checked to, "ip overbet" its bets above pot. Few aggregates are published for turns and
# rivers (GTO Wizard's are behind an account; its blog shows single spots, mostly as images), so the bands are wide and
# "*" marks figures from a different but similar spot. Checked against the native Accurate model (TestTurnRiverReferences)
# and against the rich reference itself.
TURN_REFERENCES = [
    # GTO Wizard, "How and why you should use turn donk bets": BB leads the turn almost only when it pairs a low board card
    # or fills a straight; on K75 a 7 turn it leads ~11% at 100bb (16% at 40bb), mostly small. Ours leads K72 a 7 with
    # 33% pot more often (39% in the rich tree: the BTN c-bet this flop with its whole range at a third pot)
    ("t-k72-3", "oop bet", 0, 3, "GTO Wizard turn donks: none on a brick"),
    ("t-k72-7", "oop bet", 8, 45, "GTO Wizard K75-7: ~11% at 100bb *"),
    # ThinkGTO (Solver+, JJ4 BTN vs BB, 25/80/135% turn sizes): brick turn barrel 53.5% (7.7% overbets), an A 28% of which
    # 21% overbets; GTO Wizard: overbets where the turn favours the c-bettor's range
    ("t-k72-3", "ip bet", 30, 65, "ThinkGTO JJ4 brick 53.5% *"),
    ("t-k72-3", "ip overbet", 10, 40, "ThinkGTO JJ4 brick 7.7%, A 21% overbets *"),
    ("t-q83-a", "ip overbet", 15, 50, "ThinkGTO JJ4-A: overbets 21 of 28% bets *"),
    # GTO Wizard, "The turn probe bet" (T72r checked, BTN vs BB) and "The art of turn probing": BB probes most turns,
    # often large; gtogecko: K9 probes Q72-3 61%
    ("t-q83x-5", "oop bet", 20, 65, "GTO Wizard turn probing, gtogecko Q72-3 *"),
    ("t-q83x-5", "ip bet", 20, 60, "GTO Wizard delayed c-bets *"),
    # GTO Wizard, "Mastering turn play in 3-bet pots": the 3-bettor bets the turn 47-63% after a flop bet (SB vs BTN)
    ("t-3b-k72-4", "ip bet", 45, 85, "GTO Wizard 3bet pots 47-63% *"),
]
RIVER_REFERENCES = [
    # GTO Wizard, "Principles of river play": a polar river (bet, bet, brick) bets big and overbets/shoves; its MDF
    # article: the defender folds more than MDF when it is capped
    ("r-k72-3-5", "ip bet", 65, 90, "GTO Wizard principles of river play: polar ranges bet most of the range"),
    ("r-k72-3-5", "ip overbet", 30, 75, "GTO Wizard principles of river play: overbets and shoves"),
    ("r-k72-3-5", "oop vs bet: fold", 45, 70, "MDF and over-folding (blog.gtowizard.com/mdf-alpha)"),
    ("r-q83x-5-j", "oop vs bet: fold", 55, 80, "GTO Wizard MDF: the out-of-position defender over-folds"),
]


def shipped(speed, spot, iters):
    """The spot solved with gto/native.py's tree for `speed` (a river root with Advisor.root_sizes' menu)."""
    from gto import native
    cfg = native.config(speed)
    board, r0, r1, pot, stack = T.load_spot(spot)
    if len(board) == 5:
        for kind, v in native.ROOT_SIZES["river"].items():
            (cfg.bet_sizes if kind == "bet" else cfg.raise_sizes)["river"] = list(v)
    return solve(board, r0, r1, pot, stack, cfg=cfg, time_budget=1e9, max_iters=iters)


class TestTurnRiverReferences(unittest.TestCase):
    """The shipped turn and river models against the reference solves (scripts/turn_river_reference.py --measure has
    the per-budget numbers) and published figures (TURN_REFERENCES, RIVER_REFERENCES)."""

    _sols = {}
    FAST_ITERS, ACC_ITERS = 60, 150     # (native Fast gets ~60-80 turn iterations in 2.5 s, Accurate 200-300 in 30 s)

    @classmethod
    def sol(cls, speed, spot):
        key = (speed, spot)
        if key not in cls._sols:
            # (a river root solves in milliseconds an iteration: 600 is what Fast's 2.5 s gives)
            cls._sols[key] = shipped(speed, spot, 600 if spot.startswith("r-") else
                                     cls.FAST_ITERS if speed == "fast" else cls.ACC_ITERS)
        return cls._sols[key]

    def test_fast_turn_close_to_rich_reference(self):
        # native Fast's turn (75% and 125% pot, the river inside at 75%, river classes) at 60 iterations (~2.5 s) against
        # the rich reference: EVs within 2.6% of the pot per combo (measured 0.7-2.3; one size with every river at its
        # 42 iterations: 1.5-3.4), frequencies within 6 points on average, 11 where the reference leads or probes a
        # third of the pot, which Fast's menu lacks (K72-7, Q83 checked through: measured 9.8 and 8.3)
        refs = T.load_refs()
        for spot in ("t-k72-3", "t-k72s-4s", "t-k72-7", "t-q83-a", "t-q83x-5", "t-3b-k72-4"):
            sol = self.sol("fast", spot)
            ref = refs[spot]["rich"]
            err, worst = T.freq_error(T.stats(sol), ref["stats"])
            self.assertLess(err, 11.0 if spot in ("t-k72-7", "t-q83x-5") else 6.0, f"{spot}: {err:.1f} (worst {worst})")
            self.assertLess(T.ev_error(sol, ref["values"]), 2.6, spot)

    def test_accurate_turn_close_to_rich_reference(self):
        # 33/75/125% with river classes at 150 iterations (native Accurate gets ~200 in 30 s): frequencies within 3.5 points
        # of the rich reference on average, EVs within 0.8% of the pot (measured: 0.4-2.4 points, 0.26-0.53%; the old two
        # sizes with every river at the same cost: 3.6-8.8 points, 0.8-2.9%, mostly the missing overbet and small lead)
        refs = T.load_refs()
        for spot in ("t-k72-3", "t-k72-7", "t-q83-a"):
            sol = self.sol("accurate", spot)
            ref = refs[spot]["rich"]
            err, worst = T.freq_error(T.stats(sol), ref["stats"])
            self.assertLess(err, 3.5, f"{spot}: {err:.1f} (worst {worst})")
            self.assertLess(T.ev_error(sol, ref["values"]), 0.8, spot)

    def test_river_root_menu_close_to_rich(self):
        # a river root's 33/75/125/250% menu at 600 iterations: per-combo EVs within 0.75% of the pot of the six-size
        # reference (measured 0.11-0.61, the same at 2500 iterations: the menu, not convergence), frequencies within 2.5
        refs = T.load_refs()
        for spot in ("r-k72-3-5", "r-k72-3x-9", "r-k72s-4-9s", "r-q83x-5-j"):
            sol = self.sol("fast", spot)        # (both speeds' river roots play the same menu)
            ref = refs[spot]["rich"]
            self.assertLess(T.ev_error(sol, ref["values"]), 0.75, spot)
            self.assertLess(T.freq_error(T.stats(sol), ref["stats"])[0], 2.5, spot)

    def test_published(self):
        for spot, stat, lo, hi, src in TURN_REFERENCES + RIVER_REFERENCES:
            ref = REFS[spot]["rich"]["stats"].get(stat, 0.0)
            self.assertTrue(lo <= ref <= hi, f"reference {spot} {stat} {ref:.1f} not in [{lo}, {hi}] ({src})")
            # (Accurate where its solve is shared with the test above; Fast has no small lead for K72-7's)
            speeds = ["accurate"] if spot in ("t-k72-3", "t-k72-7", "t-q83-a") else ["fast"]
            for speed in speeds:
                v = T.stats(self.sol(speed, spot)).get(stat, 0.0)
                self.assertTrue(lo <= v <= hi, f"{speed} {spot} {stat} {v:.1f} not in [{lo}, {hi}] ({src})")

    def test_texture(self):
        # the pairing turn draws leads, the brick none; the A overcard on Q83 is overbet more than the brick on K72
        st = {s: REFS[s]["rich"]["stats"] for s in ("t-k72-3", "t-k72-7", "t-q83-a")}
        self.assertGreater(st["t-k72-7"]["oop bet"], st["t-k72-3"]["oop bet"] + 10)
        self.assertGreater(st["t-q83-a"]["ip overbet"], st["t-k72-3"]["ip overbet"])


if __name__ == "__main__":
    unittest.main()
