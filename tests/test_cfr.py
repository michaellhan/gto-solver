"""The regret minimizers (Config.cfr, gto.solver.CFR_SCHEMES) against toy poker games solved in closed form, played
through the real solver on real cards, and the solver's exploitability against best responses derived here.

Half-street games: the in-position player Y acts first on the river (to_act=1, so its check ends the hand) with one
bet of B into a pot of P, s = B / P, the bet all-in (stack = B), and X calls or folds. EVs in the solver's units:
what a hand takes from the pot, minus its own bets when it loses.

* AKQ (The Mathematics of Poker, ch. 14): each player holds one of AA > KK > QQ on a board neither pairs, the six
  cards disjoint, so the opponent holds either of the other two equally (card removal does it). Y bets A, checks K,
  bluffs Q with b = s / (1 + s) (X's K must be indifferent: b (P + B) = B); X calls with A, folds Q, calls K with
  c = (1 - s) / (1 + s) (Y's Q must be indifferent between a bluff and a check, worth 0: X's A always calls and its
  K c of the time, so -B / 2 + ((1 - c) P - c B) / 2 = 0). Y's EVs:
  A = P + c B / 2, K = P / 2, Q = 0.
* Clairvoyance (ch. 13): Y holds the nuts or air, X a bluff-catcher that beats air. Y bets every nut hand and
  bluffs s / (1 + s) of its bets; X calls 1 / (1 + s). X's EV: P x (the share of Y's hands that are air checking).
* [0, 1] game (ch. 16, Example 16.x, half street): hands uniform on [0, 1]. Y bets its top a and bottom b, X calls
  with its top c. X's threshold hand beats every bluff and loses to every value bet: b (P + B) = a B. Y's bluffing
  threshold b: (1 - c) P - c B = b P. Y's value threshold 1 - a: it gains B against X's calls it beats (c - a) and
  loses B against the a above it, so c = 2 a. Hence a = (1 + s) / (s + 2 (1 + s)^2), b = a s / (1 + s), c = 2 a
  (s = 1: 2/9, 1/9, 4/9). X's calls among its bluff-catchers [b, 1 - a] aren't unique, only their mass c - a.
  (Which leaves Y's middle hands some room as well.) Played on 1081 river combos whose strengths are replaced by a random order (the
  evaluator patched out): card removal is then a random 8% of the opponent's hands, the same at every strength.
"""
import unittest
from unittest import mock
import numpy as np
from gto.cards import cards, COMBO_INDEX, ALL_COMBOS
from gto.ranges import parse_range
from gto.solver import solve, Config, Ctx, CFR_SCHEMES, MEMBER_FULL, N_FULL

NONE = {s: [] for s in ("flop", "turn", "river")}
BOARD = cards("2c3d7h9sJd")


def half_street_cfg(s, cfr="dcfr"):
    return Config(bet_sizes={st: [s] for st in ("flop", "turn", "river")}, raise_sizes=NONE, allin_spr=0, cfr=cfr)


def idx(sol, combo):
    return list(sol.ctx.active).index(COMBO_INDEX[tuple(sorted(cards(combo)))])


AKQ = ("AsAh", "KsKh", "QsQh")


def akq(s, cfr="dcfr", iters=2000, P=10.0):
    r = parse_range(", ".join(AKQ))
    return solve(BOARD, r, r, pot=P, stack=s * P, to_act=1, cfg=half_street_cfg(s, cfr), time_budget=1e9, max_iters=iters)


def akq_exploitability(sol, P, B):
    """Independent best responses to the solve's average strategies, enumerating the six deals."""
    yb = {h: sol.root.avg_strategy0()[1, idx(sol, h)] for h in AKQ}                      # Y bets
    xc = {h: sol.root.children[1].avg_strategy0()[1, idx(sol, h)] for h in AKQ}          # X calls
    rank = {h: 2 - i for i, h in enumerate(AKQ)}
    sd = lambda a, b: 1.0 if rank[a] > rank[b] else -1.0
    v_y = v_x = 0.0
    for h in AKQ:
        opp = [o for o in AKQ if o != h]
        check = sum(sd(h, o) * P / 2 for o in opp)
        bet = sum(xc[o] * sd(h, o) * (P / 2 + B) + (1 - xc[o]) * P / 2 for o in opp)
        v_y += max(check, bet)
        checked = sum((1 - yb[o]) * sd(h, o) * P / 2 for o in opp)
        call = sum(yb[o] * sd(h, o) * (P / 2 + B) for o in opp)
        fold = sum(yb[o] * -P / 2 for o in opp)
        v_x += checked + max(call, fold)
    return (v_x + v_y) / 2 / 6 / P * 100


class TestAKQ(unittest.TestCase):
    def test_converges_to_closed_form(self):
        P = 10.0
        for s in (0.5, 0.75):
            B = s * P
            sol = akq(s)
            self.assertEqual(sol.root.names, ["check", "all-in"])
            ys = sol.root.avg_strategy0()[1]
            xs = sol.root.children[1].avg_strategy0()[1]
            b, c = s / (1 + s), (1 - s) / (1 + s)
            got = {h: float(ys[idx(sol, h)]) for h in AKQ}
            self.assertAlmostEqual(got["AsAh"], 1, places=2)
            self.assertAlmostEqual(got["KsKh"], 0, places=2)
            self.assertAlmostEqual(got["QsQh"], b, delta=0.01)
            self.assertAlmostEqual(float(xs[idx(sol, "AsAh")]), 1, places=2)
            self.assertAlmostEqual(float(xs[idx(sol, "QsQh")]), 0, places=2)
            self.assertAlmostEqual(float(xs[idx(sol, "KsKh")]), c, delta=0.01)
            ev = sol.solver.node_action_evs(sol.root, sol.r0, sol.r1)
            self.assertAlmostEqual(float(ev[1, idx(sol, "AsAh")]), P + c * B / 2, delta=0.02)
            self.assertAlmostEqual(float(ev[0, idx(sol, "KsKh")]), P / 2, delta=0.02)
            self.assertAlmostEqual(float(ev[0, idx(sol, "QsQh")]), 0, delta=0.02)
            self.assertAlmostEqual(float(ev[1, idx(sol, "QsQh")]), 0, delta=0.02)
            self.assertLess(sol.exploitability(), 0.05)

    def test_exploitability_is_exact(self):
        # the solver's best responses equal the enumeration above for strategies far from equilibrium: a few
        # iterations, and arbitrary strategy sums
        P = 10.0
        rng = np.random.default_rng(3)
        for s in (0.5, 1.0):
            for iters in (1, 3, 10):
                sol = akq(s, iters=iters)
                self.assertAlmostEqual(sol.exploitability(), akq_exploitability(sol, P, s * P), places=4)
            for _ in range(3):
                for node in (sol.root, sol.root.children[1]):
                    node.ssum[...] = rng.random(node.ssum.shape)
                e = sol.exploitability()
                self.assertGreater(e, 0.1)
                self.assertAlmostEqual(e, akq_exploitability(sol, P, s * P), places=4)

    def test_every_scheme_converges(self):
        for cfr in CFR_SCHEMES:
            sol = akq(0.5, cfr=cfr, iters=1000)
            self.assertLess(sol.exploitability(), 0.3, cfr)
            self.assertAlmostEqual(float(sol.root.avg_strategy0()[1, idx(sol, "QsQh")]), 1 / 3, delta=0.03, msg=cfr)


class TestClairvoyance(unittest.TestCase):
    def test_converges_to_closed_form(self):
        # Y: 3 nut combos (88 on K8532) and 12 air (T9o); X: QQ, which beats air only
        board = cards("Kh8d5c3s2h")
        P = 10.0
        for s in (0.5, 1.0, 2.0):
            sol = solve(board, parse_range("QQ"), parse_range("88, T9o"), pot=P, stack=s * P, to_act=1,
                        cfg=half_street_cfg(s), time_budget=1e9, max_iters=2000)
            act = sol.ctx.active
            air = parse_range("T9o")[act] > 0
            nuts = parse_range("88")[act] > 0
            bet = sol.root.avg_strategy0()[1]
            self.assertAlmostEqual(float(bet[nuts].mean()), 1, places=2)
            bluffs = float(bet[air].sum())
            self.assertAlmostEqual(bluffs / 3, s / (1 + s), delta=0.01)
            qq = parse_range("QQ")[act] > 0
            self.assertAlmostEqual(float(sol.root.children[1].avg_strategy0()[1][qq].mean()), 1 / (1 + s), delta=0.01)
            # X's value: the pot whenever air checks; the solver's per-combo utility is net of X's half of the pot
            u = sol.solver._cfr(sol.root, 0, sol.r0[None].astype(np.float32), sol.r1[None].astype(np.float32),
                                update=False, use_avg=True)[0]
            ev = float(u[qq].mean()) / 15 + P / 2
            self.assertAlmostEqual(ev, P * (12 - 3 * s / (1 + s)) / 15, delta=0.02)
            self.assertLess(sol.exploitability(), 0.05)


def uniform_game(s, cfr="dcfr", iters=600, P=10.0, seed=0):
    """The [0, 1] game on every river combo, strengths a random order. Returns (solve, strengths by active combo)."""
    order = np.random.default_rng(seed).permutation(N_FULL).astype(np.float64)

    def strength(self, board):
        live = MEMBER_FULL[list(board)][:, self.active].sum(axis=0) == 0
        return np.where(live, order[self.active], -1.0)

    r = np.ones(N_FULL)
    with mock.patch.object(Ctx, "strength", strength):
        sol = solve(BOARD, r, r, pot=P, stack=s * P, to_act=1, cfg=half_street_cfg(s, cfr), time_budget=1e9,
                    max_iters=iters)
        sol.solver.exploitability(sol.r0, sol.r1)     # (builds the showdown index while patched)
    st = order[sol.ctx.active]
    u = (np.argsort(np.argsort(st)) + 0.5) / len(st)   # each combo's place in [0, 1]
    return sol, st, u


def uniform_exploitability(sol, st, P, B):
    """Independent best responses with dense matrices (no Ranking, no compat sums): X is player 0, Y player 1."""
    ctx = sol.ctx
    C = np.array([[0.0 if set(ALL_COMBOS[a]) & set(ALL_COMBOS[b]) else 1.0 for b in ctx.active] for a in ctx.active])
    S = np.sign(st[:, None] - st[None, :]) * C        # S[i, j]: +1 if i beats j
    yb = sol.root.avg_strategy0()[1].astype(np.float64)
    xc = sol.root.children[1].avg_strategy0()[1].astype(np.float64)
    checked = S @ (1 - yb) * P / 2
    call = S @ yb * (P / 2 + B)
    fold = -(C @ yb) * P / 2
    v_x = (checked + np.maximum(call, fold)).sum()
    check = S @ np.ones(len(st)) * P / 2              # (S is antisymmetric: Y's j against X's i is -S[i, j])
    bet = S @ xc * (P / 2 + B) + C @ (1 - xc) * P / 2
    v_y = np.maximum(check, bet).sum()
    return (v_x + v_y) / 2 / C.sum() / P * 100


class TestUniformGame(unittest.TestCase):
    def test_converges_to_closed_form(self):
        P = 10.0
        for s in (0.5, 1.0):
            a = (1 + s) / (s + 2 * (1 + s) ** 2)
            b, c = a * s / (1 + s), 2 * a
            sol, st, u = uniform_game(s)
            yb = sol.root.avg_strategy0()[1]
            xc = sol.root.children[1].avg_strategy0()[1]
            self.assertAlmostEqual(float(yb.mean()), a + b, delta=0.01)
            self.assertAlmostEqual(float(xc.mean()), c, delta=0.01)
            # (card removal: near a threshold a hand that blocks the right hands crosses it; a few points of blur)
            m = 0.08
            self.assertGreater(float(yb[u > 1 - a + m].min()), 0.95)
            self.assertGreater(float(yb[u < b - m].min()), 0.95)
            # X's hands between b and 1 - a all beat every bluff and lose to every value bet: which of them call
            # isn't unique, only how many (c - a); above 1 - a calling is strictly better, below b folding. With
            # X's calls spread lower, a middle hand of Y's can come close to indifferent too (a few mix)
            self.assertLess(float(yb[(u > b + m) & (u < 1 - a - m)].mean()), 0.01)
            self.assertGreater(float(xc[u > 1 - a + m].min()), 0.95)
            self.assertLess(float(xc[u < b - m].max()), 0.05)
            self.assertLess(sol.exploitability(), 0.1)

    def test_exploitability_is_exact(self):
        P = 10.0
        for s, iters in ((1.0, 2), (0.5, 7), (1.0, 40)):
            sol, st, _ = uniform_game(s, iters=iters)
            e = sol.exploitability()
            self.assertGreater(e, 0.01)
            self.assertAlmostEqual(e, uniform_exploitability(sol, st, P, s * P), places=3)


class TestSchemesInTheSolver(unittest.TestCase):
    def test_merged_subtrees_match_plain_traversal(self):
        # the predictive schemes' last regrets are views into one buffer like the regrets: merged subtrees and the
        # strategy reuse between passes give the same numbers as walking every node on its own
        r_oop = parse_range("22+, A2s+, KTs+, QTs+, JTs, T9s, ATo+, KQo")
        r_ip = parse_range("22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A8o+, KTo+, QTo+")
        small = {"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]}
        for cfr in ("pcfr+", "pdcfr+", "hs-dcfr"):
            cfg = Config(n_rivers=4, bet_sizes=small, raise_sizes=small, cfr=cfr)
            fast = solve(cards("Ks7d2cTh"), r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=0)
            plain = solve(cards("Ks7d2cTh"), r_oop, r_ip, pot=5.5, stack=97.5, cfg=cfg, max_iters=0)
            self.assertTrue(any(len(m) > 1 for _, m in fast.solver._groups))
            fast.solver.batch_leaves = False
            for s in (fast.solver, plain.solver):
                s.horizon = 12
            fast.solver.run(fast.r0, fast.r1, time_budget=1e9, max_iters=12)
            s = plain.solver
            a, b = plain.r0[None, :].astype(np.float32), plain.r1[None, :].astype(np.float32)
            for _ in range(12):
                for p in (0, 1):
                    s._cfr(s.root, p, a, b, update=True, use_avg=False)
                s.iters += 1
                s._discount()
            for x, y in zip(fast.solver._nodes, plain.solver._nodes):
                self.assertTrue(np.array_equal(x.ssum, y.ssum), cfr)
                self.assertTrue(np.array_equal(x.regret, y.regret), cfr)
                if cfr != "hs-dcfr":
                    self.assertTrue(np.array_equal(x.pred, y.pred), cfr)
            if cfr != "hs-dcfr":
                self.assertTrue(any(np.any(x.pred != 0) for x in fast.solver._nodes), cfr)


if __name__ == "__main__":
    unittest.main()
