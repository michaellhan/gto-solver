"""Study's preflop solves (gto/pf_study.py): the page's game read off a Study request, a solve run in chunks, the
stored standard solves, and each node's answer, its EVs checked against a direct computation over class pairs (two
players left), triples (three) and factored sums over the opponents' classes (any number)."""
import functools
import itertools
import unittest

import numpy as np

from gto import pf_game as G
from gto import pf_study as PS
from gto.native import advisor
from gto.pf_classes import equity_matrix

RANKS = "23456789TJQKA"
BTN_OPEN = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
            {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}]


def combos(cls):
    return 6 if len(cls) == 2 else 4 if cls.endswith("s") else 12


def pair_weights():
    """Disjoint combo pairs by class pair, from the 1326 combos enumerated here (not pf_classes')."""
    def name(a, b):
        (r1, s1), (r2, s2) = sorted([divmod(a, 4), divmod(b, 4)], reverse=True)
        return RANKS[r1] * 2 if r1 == r2 else RANKS[r1] + RANKS[r2] + ("s" if s1 == s2 else "o")
    cs = list(itertools.combinations(range(52), 2))
    idx = np.array([PS.CLASSES.index(name(a, b)) for a, b in cs])
    m = np.array([(1 << a) | (1 << b) for a, b in cs], dtype=np.int64)
    one = np.zeros((len(cs), PS.K))
    one[np.arange(len(cs)), idx] = 1
    return one.T @ ((m[:, None] & m[None, :]) == 0).astype(float) @ one


W = pair_weights()


def direct_evs(sv, line):
    """Each class's EV of each action at a node two players are left at, by a recursion over pure class pairs (the
    other's actions on the way and both players' strategies below, the showdowns' pot shares), with no reaches,
    masses or counterfactual values: {class: [EV by tree action]} relative to folding now."""
    g = sv.game
    node, _, bets = sv.walk(line)
    s = sv._at(node.index)[0]
    p = node.player
    out = {g.positions.index(x) for x, a, _ in node.line if a == "fold"}
    o = next(q for q in range(g.n) if q != p and q not in out)
    row = {k: r for r, k in enumerate(g.terms)}
    share = {}
    for _, S, rows, a, c, pe in g.hu:
        sh = np.where(G.PAIR_COMPAT > 0, S / np.maximum(G.PAIR_COMPAT, 1e-300), 0.0)
        for j, r in enumerate(rows):
            share[int(r)] = (sh, int(a[j]), float(pe[j]))

    def val(nd):   # p's payoff, [p's class, o's class]
        if nd.kind != "dec":
            r = row[nd.index]
            if nd.kind == "fold":
                return np.full((PS.K, PS.K), g.pay[r, p])
            sh, a, pe = share[r]
            return (pe * sh if a == p else pe * (1 - sh.T)) - g.inv[r, p]
        d = g.dec_of[nd.index]
        vs = [val(c) for c in nd.children]
        if nd.player == p:
            return sum(s[d, a][:, None] * v for a, v in enumerate(vs))
        return sum(s[d, a][None, :] * v for a, v in enumerate(vs))
    ro, x = np.ones(PS.K), node
    while g.parent[x.index] >= 0:
        par = g.nodes_all[g.parent[x.index]]
        if par.player == o:
            ro = ro * s[g.dec_of[par.index], g.act[x.index]]
        x = par
    w = W * ro[None, :]
    inv = bets[g.positions[p]] + sv.spec.ante
    ev = [(w * val(c)).sum(1) / np.maximum(w.sum(1), 1e-300) + inv for c in node.children]
    return node, {cl: [ev[a][i] for a in range(len(ev))] for i, cl in enumerate(PS.CLASSES)}


def _path(g, s, t, stop=None):
    """Each seat's product of its own action probabilities on the way down to node t from `stop` (the root)."""
    P = np.ones((g.n, PS.K))
    x = g.nodes_all[t]
    while g.parent[x.index] >= 0 and x.index != stop:
        par = g.nodes_all[g.parent[x.index]]
        P[par.player] *= s[g.dec_of[par.index], g.act[x.index]]
        x = par
    return P


@functools.lru_cache(maxsize=1)
def _triples():
    """[i, j, l] combo triples with j's and l's combos off i's cards (from W), and a class's three-way share before
    normalising: corr x equity against each opponent."""
    n_in = np.array([combos(c) for c in PS.CLASSES], float)
    eq = equity_matrix()
    return (W[:, :, None] * W[:, None, :] / n_in[:, None, None],
            G.three_way_corr()[:, None, None] * eq[:, :, None] * eq[:, None, :])


def direct_evs3(sv, line):
    """Each class's EV of each action at a node three players are left at, by a sum over class triples at every
    terminal below (each player's action probabilities on the way, from the root to the node for the opponents and
    from the action down for all three; the payoffs: a fold's, a heads-up showdown's pot share, a three-way one's
    share of the pot, corr x equity x equity, normalised over the three ranges reaching that terminal and times the
    seat's realization), with no reaches, masses or counterfactual values. Card removal as the game models it: combo
    triples with both opponents' combos disjoint from the player's, from the 1326 combos enumerated here (each
    opponent's against the other's isn't modelled); the players folded before the node cancel out."""
    g = sv.game
    node, _, bets = sv.walk(line)
    s = sv._at(node.index)[0]
    p = node.player
    out = {g.positions.index(x) for x, a, _ in node.line if a == "fold"}
    o1, o2 = [q for q in range(g.n) if q != p and q not in out]
    row = {k: r for r, k in enumerate(g.terms)}
    hu, three = {}, {}
    for _, S, rows, a, c, pe in g.hu:
        sh = np.where(G.PAIR_COMPAT > 0, S / np.maximum(G.PAIR_COMPAT, 1e-300), 0.0)
        for j, r in enumerate(rows):
            hu[int(r)] = (sh, int(a[j]), int(c[j]), float(pe[j]))
    if g.three is not None:
        rows, act, fac, pe = g.three
        for j, r in enumerate(rows):
            three[int(r)] = ([int(x) for x in act[j]], [float(x) for x in fac[j]], float(pe[j]))
    T3, Q = _triples()
    P0 = _path(g, s, node.index)
    B = T3 * P0[o1][None, :, None] * P0[o2][None, None, :]
    inv = bets[g.positions[p]] + sv.spec.ante
    evs = []
    for ch in node.children:
        num = np.zeros(PS.K)
        for t in G.walk(ch):
            if t.kind == "dec":
                continue
            r = row[t.index]
            Pt = _path(g, s, t.index, stop=ch.index)
            w1, w2 = Pt[o1], Pt[o2]
            if t.kind == "fold" or not g.t_active[r, p]:
                v = g.pay[r, p] * np.einsum("ijl,j,l->i", B, w1, w2)
            elif r in hu:
                sh, a, c, pe = hu[r]
                pay = (pe * sh if a == p else pe * (1 - sh.T)) - g.inv[r, p]     # [p's class, the other's]
                v = np.einsum("ijl,j,l,ij->i" if o1 in (a, c) else "ijl,j,l,il->i", B, w1, w2, pay)
            else:
                act, fac, pe = three[r]
                Pr = _path(g, s, t.index)
                tot = mass = 0.0
                for k in range(3):
                    x, y = [act[m] for m in range(3) if m != k]
                    tot += np.einsum("ijl,i,j,l,ijl->", T3, Pr[act[k]], Pr[x], Pr[y], Q)
                    mass += np.einsum("ijl,i,j,l->", T3, Pr[act[k]], Pr[x], Pr[y]) / 3
                pay = pe * fac[act.index(p)] * (mass / tot if tot > 1e-300 else 1.0) * Q - g.inv[r, p]
                v = np.einsum("ijl,j,l,ijl->i", B, w1, w2, pay)
            num += Pt[p] * v
        evs.append(num / np.maximum(B.sum((1, 2)), 1e-300) + inv)
    return node, {cl: [evs[a][i] for a in range(len(evs))] for i, cl in enumerate(PS.CLASSES)}


def direct_evs_n(sv, line):
    """Each class's EV of each action at a node any number of players are left at, by sums over the opponents'
    classes at every terminal below, with no reaches, masses or counterfactual values. The game's card removal is
    pairwise with the player's class, so given the player's class the opponents' sums factor: an opponent not in the
    pot at a terminal adds its own sum (its combos off the player's, times its action probabilities from the root),
    one in a heads-up showdown the sum of the pot share against it, and a three-way showdown's share (corr x equity x
    equity, normalised over the three ranges reaching that terminal, times the seat's realization) splits into a sum
    for each of the two. Agrees with direct_evs and direct_evs3 (the double and triple sums) to 1e-11."""
    g = sv.game
    node, _, bets = sv.walk(line)
    s = sv._at(node.index)[0]
    p = node.player
    out = {g.positions.index(x) for x, a, _ in node.line if a == "fold"}
    opp = [q for q in range(g.n) if q != p and q not in out]
    n_in = np.array([combos(c) for c in PS.CLASSES], float)
    wn = W / n_in[:, None]                       # [i, j]: j's combos off one of i's
    E = wn * equity_matrix()
    corr = G.three_way_corr()
    row = {k: r for r, k in enumerate(g.terms)}
    hu, three = {}, {}
    for _, S, rows, a, c, pe in g.hu:
        sh = np.where(G.PAIR_COMPAT > 0, S / np.maximum(G.PAIR_COMPAT, 1e-300), 0.0)
        for j, r in enumerate(rows):
            hu[int(r)] = (sh, int(a[j]), float(pe[j]))
    if g.three is not None:
        rows, act, fac, pe = g.three
        for j, r in enumerate(rows):
            three[int(r)] = ([int(x) for x in act[j]], [float(x) for x in fac[j]], float(pe[j]))
    P0 = _path(g, s, node.index)
    den = np.prod([wn @ P0[q] for q in opp], axis=0)
    inv = bets[g.positions[p]] + sv.spec.ante
    evs = []
    for ch in node.children:
        num = np.zeros(PS.K)
        for t in G.walk(ch):
            if t.kind == "dec":
                continue
            r = row[t.index]
            Pt = _path(g, s, t.index, stop=ch.index)
            Pq = {q: P0[q] * Pt[q] for q in opp}
            own = {q: wn @ Pq[q] for q in opp}
            if t.kind == "fold" or not g.t_active[r, p]:
                v = g.pay[r, p] * np.prod([own[q] for q in opp], axis=0)
            else:
                act = [q for q in opp if g.t_active[r, q]]
                rest = np.prod([own[q] for q in opp if q not in act] or [np.ones(PS.K)], axis=0)
                if r in hu:
                    sh, a, pe = hu[r]
                    (o,) = act
                    pay = (pe * sh if a == p else pe * (1 - sh.T)) - g.inv[r, p]     # [p's class, o's]
                    v = ((pay * wn) @ Pq[o]) * rest
                else:
                    acts, fac, pe = three[r]
                    x, y = act
                    Pr = _path(g, s, t.index)            # the three's reaches from the root
                    tot = mass = 0.0
                    for k in range(3):
                        u, w = [acts[m] for m in range(3) if m != k]
                        tot += (Pr[acts[k]] * n_in * corr * (E @ Pr[u]) * (E @ Pr[w])).sum()
                        mass += (Pr[acts[k]] * n_in * (wn @ Pr[u]) * (wn @ Pr[w])).sum() / 3
                    norm = mass / tot if tot > 1e-300 else 1.0
                    v = (pe * fac[acts.index(p)] * norm * corr * (E @ Pq[x]) * (E @ Pq[y])
                         - g.inv[r, p] * own[x] * own[y]) * rest
            num += Pt[p] * v
        evs.append(num / np.maximum(den, 1e-300) + inv)
    return node, {cl: [evs[a][i] for a in range(len(evs))] for i, cl in enumerate(PS.CLASSES)}



def check_direct(t, sv, players, max_terms=None, min_reach=1e-4, limit=None, direct=None):
    """Every node `players` are left at that the solve reaches (each seat with `min_reach` of its hands; the first
    `limit` of them in the tree's order), its answer's EVs against the direct computation's to the answer's rounding
    (`direct`: by default the double sum with two left, the triple sum with three, the factored sums with more);
    returns how many."""
    n = 0
    g = sv.game
    reach = sv._values()[1]
    for nd in G.walk(g.root):
        if nd.kind != "dec" or g.n - sum(a == "fold" for _, a, _ in nd.line) != players or n == limit:
            continue
        if min(reach[nd.index, q].sum() for q in range(g.n)) < min_reach:
            continue
        if max_terms and sum(1 for x in G.walk(nd) if x.kind != "dec") > max_terms:
            continue
        line = [{"pos": x, "action": a, **({"to": v} if a not in ("fold", "call") else {})} for x, a, v in nd.line]
        r = sv.node(line)
        _, ev = (direct or (direct_evs if players == 2 else direct_evs3 if players == 3 else direct_evs_n))(sv, line)
        order = sorted(range(len(nd.names)), key=lambda a: {"raise": 0, "all-in": 0, "call": 1, "fold": 2}[nd.names[a]])
        for cl, c in r["grid"]["cells"].items():
            for j, a in enumerate(order):
                t.assertAlmostEqual(c["ev"][j], ev[cl][a], delta=0.0011, msg=(line, cl))
        n += 1
    return n


class TestSpec(unittest.TestCase):
    def test_spec_of(self):
        spec, why = PS.spec_of({"players": 6, "stack": 100})
        self.assertEqual((spec, why), (PS.Spec(), None))
        self.assertEqual(spec.key, "6max_100bb")
        spec, _ = PS.spec_of({"players": 6, "stack": 100, "rake": 0.05, "rake_cap": 3})
        self.assertEqual(spec.key, "6max_100bb_r5c3")
        self.assertIn(spec.key, PS.STORED)
        spec, _ = PS.spec_of({"players": 4, "stack": 40, "ante": 0.125, "straddle": True,
                              "stacks": {"UTG": 40, "BTN": 40}, "pf_sizes": {"open": 2.2}})
        self.assertEqual((spec.players, spec.stack, spec.ante, spec.straddle, spec.sizes), (4, 40.0, 0.125, True, (("open", 2.2),)))
        self.assertNotIn(spec.key, PS.STORED)
        # (amounts in chips with a big blind: as big blinds)
        self.assertEqual(PS.spec_of({"players": 6, "stack": 200, "bb": 2})[0].stack, 100.0)
        for st, part in (({"players": 8}, "too big"), ({"payouts": [0.5, 0.5]}, "tournament"),
                         ({"stacks": {"UTG": 50, "BTN": 100}}, "stacks equal"), ({"stack": 5}, "10 to 250"),
                         ({"pf_sizes": {"open": 9}}, "open must be"), ({"pf_sizes": {"limp": 1}}, "unknown")):
            spec, why = PS.spec_of(dict({"players": 6, "stack": 100}, **st))
            self.assertIsNone(spec)
            self.assertIn(part, why)

    def test_pack(self):
        sv = PS.Solve(PS.Spec(players=2)).run(5)
        g = sv.game
        blob = PS.pack(g, 1.5)
        self.assertEqual(blob[:4], b"PFS1")
        self.assertAlmostEqual(PS.stored_mbb(blob), 1.5, places=5)
        self.assertIsNone(PS.stored_mbb(PS.pack(g)))
        s, back = g.average(), PS.unpack(g, blob)
        reach = g._down(s)
        for d, k in enumerate(g.dec):
            on = reach[k, g.player[d]] > 1e-7
            np.testing.assert_allclose(back[d][:, on], s[d][:, on], atol=1.5 / 255)
        with self.assertRaises(ValueError):
            PS.unpack(PS.Spec(players=3).game(), blob)
        # a stored solve: done, its iterations and measure from the file
        st = PS.Solve(PS.Spec(players=2), blob)
        self.assertTrue(st.done)
        self.assertEqual(st.status()["mbb"], 1.5)
        self.assertEqual(st.status()["iterations"], PS.STORED_ITERS)


class TestHeadsUp(unittest.TestCase):
    """The heads-up game solved to the end (2000 iterations, ~1 s), every node's EVs checked directly."""
    @classmethod
    def setUpClass(cls):
        cls.sv = PS.Solve(PS.Spec(players=2))
        while not cls.sv.done:
            cls.sv.run(30)

    def test_solved(self):
        sv = self.sv
        self.assertEqual(sv.iters, PS.MAX_ITERS)   # (a small tree solves to the end)
        self.assertEqual(sv.target, 0.0)
        self.assertLess(sv.mbb, 0.8)

    def test_every_node(self):
        sv, n = self.sv, 0
        for nd in G.walk(sv.game.root):
            if nd.kind != "dec":
                continue
            line = [{"pos": x, "action": a, **({"to": t} if a != "fold" and a != "call" else {})} for x, a, t in nd.line]
            r = sv.node(line)
            self.assertEqual((r["study"], r["actor"], r["source"]), ("node", sv.game.positions[nd.player], "solver"))
            check_totals(self, r)
            _, ev = direct_evs(sv, line)
            order = sorted(range(len(nd.names)), key=lambda a: {"raise": 0, "all-in": 0, "call": 1, "fold": 2}[nd.names[a]])
            for cl, c in r["grid"]["cells"].items():
                self.assertEqual(len(c["ev"]), len(order))
                for j, a in enumerate(order):
                    self.assertAlmostEqual(c["ev"][j], ev[cl][a], delta=0.0011, msg=(line, cl, j))
                # folding is 0 (EVs are relative to folding now); the mix's EV is the mix of the EVs
                if "fold" in r["grid"]["actions"]:
                    self.assertEqual(c["ev"][r["grid"]["actions"].index("fold")], 0.0)
                self.assertAlmostEqual(c["evm"], sum(f * e for f, e in zip(c["f"], c["ev"])), delta=0.01)
            n += 1
        self.assertEqual(n, len(sv.game.dec))

    def test_equilibrium(self):
        """Solved to the end: at the nodes the range reaches often, a hand mixing two actions finds them about as
        good, and the range gives up next to nothing to its best actions."""
        sv, gaps = self.sv, []
        for nd in G.walk(sv.game.root):
            if nd.kind != "dec":
                continue
            r = sv.node([{"pos": x, "action": a, **({"to": t} if a not in ("fold", "call") else {})} for x, a, t in nd.line])
            if r["range_combos"] < 100 or any(o["combos"] < 100 for o in r["opponents"]):
                continue
            self.assertLess(r["regret_bb"], 0.01)
            self.assertFalse(r["rough"])
            for c in r["grid"]["cells"].values():
                on = [e for f, e in zip(c["f"], c["ev"]) if f > 0.1]
                if c["w"] > 0.2 and len(on) > 1:
                    gaps.append(max(on) - min(on))
        self.assertGreater(len(gaps), 10)
        self.assertLess(np.median(gaps), 0.02)
        self.assertLess(max(gaps), 0.25)

    def test_answer(self):
        r = self.sv.node([])
        self.assertEqual((r["actor"], r["pot_bb"], r["to_call_bb"], r["stack_bb"]), ("SB", 1.5, 0.5, 99.5))
        self.assertEqual(r["grid"]["actions"], ["raise", "call", "fold"])   # (the small blind may limp heads-up)
        self.assertEqual(r["actions"][0]["amount_bb"], 3.0)
        self.assertEqual(r["grid"]["cells"]["AA"]["f"][2], 0.0)
        self.assertGreater(r["range_ev_bb"], 0.0)
        self.assertEqual(r["opponents"], [])
        self.assertEqual(r["solver"]["iterations"], PS.MAX_ITERS)
        r = self.sv.node([{"pos": "SB", "action": "call"}])   # a limp: the big blind checks or raises
        self.assertEqual((r["actor"], r["to_call_bb"]), ("BB", 0.0))
        self.assertEqual(r["grid"]["actions"], ["raise", "check"])
        self.assertEqual([o["seat"] for o in r["opponents"]], ["SB"])
        # where the hand goes on: a flop to deal, or over
        self.assertEqual(self.sv.node([{"pos": "SB", "action": "call"}, {"pos": "BB", "action": "check"}]),
                         {"study": "deal", "street": "flop", "allin": False})
        self.assertEqual(self.sv.node([{"pos": "SB", "action": "fold"}]), {"study": "over", "street": "preflop", "winner": "BB"})
        # a raise of another size is played as the tree's, and said; a player out of turn is refused
        r = self.sv.node([{"pos": "SB", "action": "raise", "to": 2.5}])
        self.assertEqual(r["notes"], ["SB's raise to 2.5bb is played as the solve's size, 3bb"])
        self.assertIn("error", self.sv.node([{"pos": "BB", "action": "fold"}]))


def check_totals(t, r):
    acts = r["actions"]
    t.assertAlmostEqual(sum(a["freq"] for a in acts), 1.0, delta=0.002)
    t.assertEqual(r["grid"]["actions"], [a["action"] for a in acts])
    tot = sum(c["w"] * combos(k) for k, c in r["grid"]["cells"].items())
    t.assertAlmostEqual(tot, r["range_combos"], delta=0.005 * tot + 0.2)
    for i, a in enumerate(acts):
        share = sum(c["w"] * combos(k) * c["f"][i] for k, c in r["grid"]["cells"].items()) / tot
        t.assertAlmostEqual(share, a["freq"], delta=0.005)


class TestThreeHanded(unittest.TestCase):
    """A 3-handed game solved to the end (2000 iterations, ~3 s): every node it reaches, its EVs checked directly,
    where three players are left by a sum over class triples (three-way showdowns' normalised shares included)."""
    @classmethod
    def setUpClass(cls):
        cls.sv = PS.Solve(PS.Spec(players=3))
        while not cls.sv.done:
            cls.sv.run(30)

    def test_direct(self):
        self.assertGreater(check_direct(self, self.sv, 3), 15)
        self.assertGreater(check_direct(self, self.sv, 2), 20)

    def test_factored_sums(self):
        """direct_evs_n (the opponents' sums factored, as the game's pairwise card removal allows) equals the double
        and triple sums at every node the solve reaches."""
        g, n = self.sv.game, 0
        reach = self.sv._values()[1]
        for nd in G.walk(g.root):
            if nd.kind != "dec" or min(reach[nd.index, q].sum() for q in range(g.n)) < 1e-4:
                continue
            left = g.n - sum(a == "fold" for _, a, _ in nd.line)
            line = [{"pos": x, "action": a, **({"to": v} if a not in ("fold", "call") else {})} for x, a, v in nd.line]
            _, ev = direct_evs_n(self.sv, line)
            _, want = (direct_evs if left == 2 else direct_evs3)(self.sv, line)
            for cl in PS.CLASSES:
                np.testing.assert_allclose(ev[cl], want[cl], rtol=0, atol=1e-9, err_msg=str((line, cl)))
            n += 1
        self.assertGreater(n, 40)


class TestFourHanded(unittest.TestCase):
    """A 4-handed game solved 400 iterations: every node it reaches with all four players left, its EVs checked by
    the factored sums over the opponents' classes (direct_evs_n)."""
    @classmethod
    def setUpClass(cls):
        cls.sv = PS.Solve(PS.Spec(players=4))
        cls.sv.run(600, should_stop=lambda: cls.sv.iters >= 400)

    def test_direct(self):
        self.assertEqual(self.sv.iters, 400)
        self.assertGreater(check_direct(self, self.sv, 4), 15)


class TestCustomGames(unittest.TestCase):
    """Games set in the Preflop solve form, solved 300 iterations, every node each reaches checked by the factored
    sums: heads-up at 10bb (its limps), 3-handed at 15bb with antes, 4-handed with antes and a straddle, 4-handed at
    20bb raked 10% capped at 1bb. (Checked once the same way, too slow for the tests: heads-up and 3-handed at
    250bb, 5-handed raked with other sizes, 6-max at 40bb with antes, straddled, and with other sizes.)"""
    def test_direct(self):
        for kw in (dict(players=2, stack=10.0), dict(players=3, stack=15.0, ante=0.2),
                   dict(players=4, ante=0.125, straddle=True), dict(players=4, stack=20.0, rake=(0.1, 1.0))):
            sv = PS.Solve(PS.Spec(**kw))
            sv.run(600, should_stop=lambda: sv.iters >= 300)
            n = {k: check_direct(self, sv, k, direct=direct_evs_n) for k in range(2, kw["players"] + 1)}
            self.assertGreater(min(n.values()), 5, (kw, n))


class TestStall(unittest.TestCase):
    def test_rule(self):
        """A solve with a target stops once a check is no nearer than STALL times the one STALL_BACK iterations
        before, from STALL_FROM on; its status says so."""
        sv = PS.Solve(PS.Spec(players=5))
        self.assertEqual(sv.target, PS.TARGET_MBB)
        t, b = PS.STALL_FROM, PS.STALL_FROM - PS.STALL_BACK
        sv.iters = sv.expl_at = t
        sv.expl = [0.003] * 5
        for hist, want in (({b: 3.05, t: 3.0}, True), ({b: 3.2, t: 3.0}, False), ({t: 3.0}, False),
                           ({b - 100: 3.0, t: 3.0}, False)):
            sv.history = hist
            self.assertEqual((sv.stalled, sv.done, sv.status()["stalled"]), (want, want, want), hist)
        sv.history = {b: 3.05, t: 3.0}
        sv.target = 0.0                                   # (a small tree runs to MAX_ITERS)
        self.assertFalse(sv.done)
        sv.target, sv.iters, sv.expl_at = PS.TARGET_MBB, t + 1, t + 1      # (only as measured)
        self.assertFalse(sv.stalled)
        sv.iters = sv.expl_at = b
        sv.history = {b - PS.STALL_BACK: 3.05, b: 3.0}    # (before STALL_FROM)
        self.assertFalse(sv.done)

    def test_levels_off(self):
        """4-handed at 50bb with a 0.12 ante levels off at ~1.8 mbb (its three-way flops: tests/test_pf_game.py), so
        given a 1.5 target it stops where it gets no nearer (the 700th iteration, 1.89), its numbers about where
        2,000 iterations put them (1.75): at the first decisions a class's EVs 0.002-0.003 bb off on average (0.025 at
        most), facing the CO's open 0.004 (0.09), the actions' frequencies within half a point."""
        spec = PS.Spec(players=4, stack=50.0, ante=0.12)
        sv = PS.Solve(spec)
        sv.target = PS.TARGET_MBB                         # (4-handed solves run to MAX_ITERS: no target)
        while not sv.done:
            sv.run(30)
        self.assertTrue(sv.stalled)
        self.assertLess(sv.iters, 1000)
        self.assertGreater(sv.mbb, PS.TARGET_MBB)
        ref = PS.Solve(spec)
        while not ref.done:
            ref.run(30)
        self.assertEqual(ref.iters, PS.MAX_ITERS)
        self.assertFalse(ref.stalled)
        co, btn = {"pos": "CO", "action": "fold"}, {"pos": "BTN", "action": "fold"}
        for line, mean, worst in (([], 0.004, 0.04), ([co], 0.004, 0.04), ([co, btn], 0.004, 0.04),
                                  ([{"pos": "CO", "action": "raise"}], 0.006, 0.15)):
            a, b = sv.node(line), ref.node(line)
            d = [abs(x - y) for cl, c in b["grid"]["cells"].items() if c["w"] > 0.2
                 for x, y, f in zip(a["grid"]["cells"][cl]["ev"], c["ev"], c["f"]) if f >= 0.05]
            self.assertGreater(len(d), 100, line)
            self.assertLess(np.mean(d), mean, line)
            self.assertLess(max(d), worst, line)
            self.assertLess(max(abs(x["freq"] - y["freq"]) for x, y in zip(a["actions"], b["actions"])), 0.01, line)


class TestStored(unittest.TestCase):
    """The stored 6-max solve (gto/static/pf/6max_100bb.bin): loaded, answered, its EVs checked directly where two
    players are left and against the game's own values at the first decision."""
    @classmethod
    def setUpClass(cls):
        PS._HELD.clear()
        cls.sv = PS.held(PS.Spec())

    def test_loaded(self):
        sv = self.sv
        self.assertTrue(sv.stored and sv.done)
        self.assertLess(sv.mbb, 2.0)
        self.assertIs(PS.held(PS.Spec()), sv)
        self.assertIsNone(PS.held(PS.Spec(players=5)))   # (nothing stored or solved)

    def test_first_decision(self):
        sv = self.sv
        r = sv.node([])
        self.assertEqual((r["actor"], r["range_combos"], r["pot_bb"]), ("UTG", 1326.0, 1.5))
        check_totals(self, r)
        self.assertEqual(r["grid"]["actions"], ["raise", "fold"])
        cells = r["grid"]["cells"]
        self.assertGreater(cells["AA"]["f"][0], 0.99)
        self.assertLess(cells["72o"]["f"][0], 0.01)
        self.assertGreater(cells["AA"]["evm"], 2.0)
        # the range's EV is the game's own value for UTG (Game.values: counterfactual, so over the chance the
        # others' hands don't collide with it, the same for every class at the first decision)
        g = sv.game
        m = np.prod([G.PRIOR @ G.PAIR_COMPAT for _ in range(g.n - 1)], axis=0)
        self.assertAlmostEqual(float(np.ptp(m)), 0.0, places=9)
        self.assertAlmostEqual(r["range_ev_bb"], g.values(g.average())[0] / m[0], delta=0.002)

    def test_heads_up_nodes(self):
        """Every node two players are left at that the solve reaches, checked directly (exact there: the model's
        card removal is pairwise, and a player who folded on the way cancels out)."""
        sv, n = self.sv, 0
        reach = sv._values()[1]
        for nd in G.walk(sv.game.root):
            if nd.kind != "dec" or sv.game.n - sum(a == "fold" for _, a, _ in nd.line) != 2:
                continue
            if min(reach[nd.index, q].sum() for q in range(sv.game.n)) < 1e-4 or len(nd.line) > 9:
                continue
            line = [{"pos": x, "action": a, **({"to": t} if a not in ("fold", "call") else {})} for x, a, t in nd.line]
            r = sv.node(line)
            _, ev = direct_evs(sv, line)
            order = sorted(range(len(nd.names)), key=lambda a: {"raise": 0, "all-in": 0, "call": 1, "fold": 2}[nd.names[a]])
            for cl, c in r["grid"]["cells"].items():
                for j, a in enumerate(order):
                    self.assertAlmostEqual(c["ev"][j], ev[cl][a], delta=0.0011, msg=(line, cl))
            n += 1
        self.assertGreater(n, 20)

    def test_three_way_nodes(self):
        """Nodes three players are left at (three folded first), by the sum over class triples: the first 40 in the
        tree's order that each seat reaches with 1% of its hands, with at most 30 terminals below (BTN, SB and BB's
        3-way pots first). Without those limits every one of the 1,079 the solve reaches with 0.01% of each seat's
        hands matches (~2.5 minutes)."""
        self.assertEqual(check_direct(self, self.sv, 3, max_terms=30, min_reach=1e-2, limit=40), 40)

    def test_many_player_nodes(self):
        """Every node three to six players are left at that each seat reaches with 0.01% of its hands (the first
        decisions, the opens and calls before the blinds act, the blinds' 3-way pots: 9, 95, 489 and 1,079 of them
        with six, five, four and three left), by the factored sums over the opponents' classes."""
        n = {k: check_direct(self, self.sv, k, direct=direct_evs_n) for k in (3, 4, 5, 6)}
        self.assertGreater(min(n.values()), 0)
        self.assertGreater(n[3], 1000)
        self.assertGreater(n[4] + n[5] + n[6], 550)

    def test_btn_open(self):
        sv = self.sv
        r = sv.node(BTN_OPEN)
        self.assertEqual((r["actor"], r["pot_bb"], r["to_call_bb"], r["stack_bb"]), ("BB", 4.0, 1.5, 99.0))
        self.assertEqual(r["grid"]["actions"], ["raise", "call", "fold"])
        self.assertEqual([o["seat"] for o in r["opponents"]], ["BTN"])
        self.assertTrue(300 < r["opponents"][0]["combos"] < 700)
        check_totals(self, r)
        self.assertFalse(r["rough"])
        self.assertLess(r["regret_bb"], 0.02)
        # the range's EV over its actions' (each weighted by its hands' chance against BTN's range)
        self.assertGreater(r["range_ev_bb"], 0.0)
        evs = [a["ev_bb"] for a in r["actions"]]
        self.assertEqual(evs[2], 0.0)
        self.assertTrue(min(evs) - 0.01 <= r["range_ev_bb"] <= max(evs) + 0.01)
        # crumbs of the average strategy aren't shown: a class appears from LIVE_W of its combos
        self.assertTrue(all(c["w"] > PS.LIVE_W for c in r["grid"]["cells"].values()))

    def test_line_never_played(self):
        """A 0% move: the player is taken to make it with every hand it held (its range as before), so the EVs are
        numbers (not 0/0), said in the notes and flagged rough."""
        sv = self.sv
        cold4 = [{"pos": "UTG", "action": "raise", "to": 2.5}, {"pos": "HJ", "action": "raise", "to": 7.5},
                 {"pos": "CO", "action": "raise", "to": 17.25}]   # (a cold 4bet: the stored solve never makes one)
        self.assertLess(sv.node(cold4[:2])["actions"][0]["freq"], 0.0005)
        r2 = sv.node(cold4)
        self.assertEqual(r2["actor"], "BTN")
        self.assertTrue(r2["rough"])
        self.assertIn("the solve never brings CO here: the range is as before the 0% move", r2["notes"])
        self.assertEqual([o["seat"] for o in r2["opponents"]], ["UTG", "HJ", "CO"])
        self.assertGreater(r2["opponents"][2]["combos"], 100)
        for c in r2["grid"]["cells"].values():
            self.assertTrue(all(np.isfinite(c["ev"])))
            self.assertTrue(all(-101 < e < 301 for e in c["ev"]))   # (AA's jam can win three stacks)


class TestAdvisor(unittest.TestCase):
    """Advisor.study with "preflop_solve": the node from a solve in hand, an offer to solve, chunks, refusals."""
    @classmethod
    def setUpClass(cls):
        cls.adv = advisor("fast")

    def study(self, **st):
        return self.adv.advise(dict({"players": 6, "stack": 100, "study": True, "preflop_solve": True}, **st))

    def test_flow(self):
        PS._HELD.clear()
        r = self.study()   # the standard game: stored, at once
        self.assertEqual((r["study"], r["source"], r["actor"]), ("node", "solver", "UTG"))
        self.assertTrue(r["pf_solve"]["stored"])
        r = self.study(rake=0.05, rake_cap=3, preflop=BTN_OPEN)
        self.assertEqual((r["study"], r["actor"], r["pf_solve"]["key"]), ("node", "BB", "6max_100bb_r5c3"))
        # a game not solved: the offer, then chunks until done, then its nodes
        r = self.study(players=3)
        self.assertEqual(r["study"], "pf_unsolved")
        self.assertEqual(r["key"], "3max_100bb")
        self.assertTrue(r["game"].startswith("3 players, 100bb, no rake, opens 2.5bb"), r["game"])
        self.assertGreater(r["estimate_s"], 0)
        for _ in range(20):
            p = self.study(players=3, preflop_solve={"run": 2})
            self.assertEqual(p["study"], "pf_progress")
            if p["pf_solve"]["done"]:
                break
        self.assertTrue(p["pf_solve"]["done"])
        self.assertEqual(p["pf_solve"]["iterations"], PS.MAX_ITERS)
        r = self.study(players=3, preflop=[{"pos": "BTN", "action": "raise", "to": 2.5}])
        self.assertEqual((r["study"], r["actor"]), ("node", "SB"))
        self.assertFalse(r["pf_solve"]["stored"])
        # the charts' answer without "preflop_solve" (unchanged), the solve not asked after the flop
        r = self.adv.advise({"players": 6, "stack": 100, "study": True})
        self.assertNotEqual(r.get("source"), "solver")
        r = self.study(board="Kd7c2h", preflop=BTN_OPEN + [{"pos": "BB", "action": "call"}])
        self.assertNotEqual(r.get("source"), "solver")
        self.assertEqual(r["street"], "flop")

    def test_kept_and_back(self):
        """A finished solve packed for the page ({"export": true}) and handed back once the worker no longer holds it
        ({"load"}): the same mix and EVs (to the pack's 1/255 steps), marked as kept in the browser; a stored game,
        an unfinished solve or none give no blob; a bad blob leaves the game unsolved. Custom sizes ("pf_sizes")."""
        PS._HELD.clear()
        st = {"players": 3, "pf_sizes": {"open": 2.2, "four_x": 2.5}}
        self.assertEqual(self.study(preflop_solve={"export": True}, **st)["blob"], None)       # (none run)
        self.assertIsNone(self.study(preflop_solve={"export": True})["blob"])                  # (the stored game)
        p = self.study(preflop_solve={"run": 0.5}, **st)
        self.assertFalse(p["pf_solve"]["done"])
        self.assertIsNone(self.study(preflop_solve={"export": True}, **st)["blob"])            # (not finished)
        while not p["pf_solve"]["done"]:
            p = self.study(preflop_solve={"run": 5}, **st)
        line = [{"pos": "BTN", "action": "raise", "to": 2.2}]
        live = self.study(preflop=line, **st)
        self.assertEqual((live["actor"], live["pf_solve"]["key"]), ("SB", "3max_100bb_four_x2.5_open2.2"))
        self.assertIn("opens 2.2bb (SB 3), 3bets 3x / 4.8x from the blinds, 4bets 2.5x", live["pf_solve"]["label"])
        e = self.study(preflop_solve={"export": True}, **st)
        self.assertEqual((e["study"], e["key"], e["iterations"]), ("pf_export", "3max_100bb_four_x2.5_open2.2", PS.MAX_ITERS))
        self.assertLess(len(e["blob"]), 40000)
        PS._HELD.clear()
        self.assertEqual(self.study(preflop=line, **st)["study"], "pf_unsolved")
        bad = self.study(preflop=line, preflop_solve={"load": "bm90IGEgc29sdmU=", "iterations": 5}, **st)
        self.assertEqual(bad["study"], "pf_unsolved")
        back = self.study(preflop=line, preflop_solve={"load": e["blob"], "iterations": e["iterations"]}, **st)
        self.assertEqual((back["study"], back["actor"]), ("node", "SB"))
        self.assertTrue(back["pf_solve"]["cached"])
        self.assertFalse(back["pf_solve"]["stored"])
        self.assertTrue(back["pf_solve"]["done"])
        self.assertEqual((back["pf_solve"]["iterations"], back["pf_solve"]["mbb"]), (PS.MAX_ITERS, round(e["mbb"], 2)))
        worst = 0.0
        for cl, c in live["grid"]["cells"].items():
            b = back["grid"]["cells"][cl]
            worst = max(worst, max(abs(x - y) for x, y in zip(c["f"], b["f"])))
        self.assertLess(worst, 0.006)
        self.assertAlmostEqual(back["range_ev_bb"], live["range_ev_bb"], delta=0.01)
        # (held now: its nodes answer without the blob, and it isn't exported again)
        self.assertTrue(self.study(**st)["pf_solve"]["cached"])
        self.assertIsNone(self.study(preflop_solve={"export": True}, **st)["blob"])

    def test_refused(self):
        r = self.study(players=8)
        self.assertEqual((r["study"], r["street"]), ("pf_none", "preflop"))
        self.assertIn("too big", r["reason"])
        r = self.study(payouts=[0.5, 0.3, 0.2])
        self.assertEqual(r["study"], "pf_none")


if __name__ == "__main__":
    unittest.main()
