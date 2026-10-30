"""Bounty tournaments (gto/icm.py Bounties; the advisor's state "bounties"): the proportional model's values of a bust
(KO, PKO, mystery), the covering rule, push/fold, the heads-up and 3-way solvers' terminals, and the answers."""
import itertools
import json
import unittest
import numpy as np
from gto import icm
from gto.advisor import Advisor, MEMBER_FULL
from gto.cards import cards as parse_cards
from gto.ranges import parse_range
from gto.solver import solve, Config


def R(pos, action, to=None):
    return {"pos": pos, "action": action, **({"to": to} if to is not None else {})}


def X(pos, action, amount=None):
    return {"pos": pos, "action": action, **({"amount": amount} if amount is not None else {})}


SIX = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
# 6-max, no payouts: SB 20bb jams into BB 40bb (who covers), everyone else 30bb
JAM_STACKS = {"UTG": 30, "HJ": 30, "CO": 30, "BTN": 30, "SB": 20, "BB": 40}
SB_JAM = [R(p, "fold") for p in ("UTG", "HJ", "CO", "BTN")] + [R("SB", "all-in")]
S4 = {"CO": 40, "BTN": 25, "SB": 8, "BB": 25}


def call_share(r):
    """The share of the arriving range that calls, from an answer's grid."""
    g = r["grid"]
    ci = g["actions"].index("call")
    n = lambda cls: 6 if len(cls) == 2 else 4 if cls.endswith("s") else 12
    tot = sum(c["w"] * n(k) for k, c in g["cells"].items())
    return sum(c["w"] * n(k) * c["f"][ci] for k, c in g["cells"].items()) / tot


def note(r, start):
    return next(n for n in r["notes"] if n.startswith(start))


class TestModel(unittest.TestCase):
    # BTN 10bb all in, BB 30bb (covers) calls 9 more, SB 50bb folded its 0.5; one prize (winner-take-all) of 300 and
    # 50 on every head: 90 chips, 150 in bounties, so a unit of money is worth 1 / (300/90 + 150/90) = 0.2 chips
    ST = {"BTN": 10, "SB": 50, "BB": 30}
    PUT = {"BTN": 10, "SB": 0.5, "BB": 1}

    def model(self, fmt, **kw):
        return icm.Model(self.ST, [300], bounties=icm.Bounties(fmt, {p: 50 for p in self.ST}, **kw))

    def test_ko_winner_take_all_is_chips_plus_the_bounty(self):
        """KO, winner-take-all: every outcome is its chips, and a bust adds k (1 - s') B to the winner (s': its chip
        share after the hand; it owned that share of the busted player's bounty already, in the pool)."""
        m = self.model("ko")
        self.assertAlmostEqual(m.scale, 0.3)
        self.assertAlmostEqual(m.bounty_k, 0.2)
        win, lose = m.showdown(dict(self.PUT, BB=10), "BB", "BTN")
        bonus = 0.2 * (1 - 40.5 / 90) * 50                          # 5.5 chips: BB ends with 40.5 of the 90
        self.assertAlmostEqual(win["BB"], 10.5 + bonus)              # +10 from BTN, +0.5 from SB, and the bounty
        self.assertAlmostEqual(lose["BB"], -10.0)                    # (BB doesn't bust: nothing changes hands but chips)
        self.assertAlmostEqual(lose["BTN"], 10.5)
        self.assertAlmostEqual(win["BTN"], -10.0)                    # the busted player loses its chips, no more
        # the call: 9 to win the 20.5 (and the bounty) needs 9 / (20.5 + 5.5) = 34.6% against 9 / 20.5 = 43.9% in chips
        need, chips = m.call_need(self.PUT, "BB", "BTN", 9)
        self.assertAlmostEqual(need, 9 / 26)
        self.assertAlmostEqual(chips, 9 / 20.5)
        need0, _ = icm.Model(self.ST, [300]).call_need(self.PUT, "BB", "BTN", 9)
        self.assertAlmostEqual(need0, 9 / 20.5)                      # (winner-take-all without bounties: chips)
        self.assertAlmostEqual(m.bust_value("BB", "BTN")[1], 50 * (1 - 40 / 90))   # (the stacks before the hand)

    def test_pko_is_half_at_once_and_the_added_half_at_the_winners_chip_share(self):
        """PKO: B/2 paid at once, B/2 added to the winner's head, which stays in the pool and is worth the winner's
        chip share s' of it, less the share s' of B it owned before: B/2 + s' B/2 - s' B = (1 - s') B/2."""
        m = self.model("pko")
        win, _ = m.showdown(dict(self.PUT, BB=10), "BB", "BTN")
        s = 40.5 / 90
        self.assertAlmostEqual(win["BB"], 10.5 + 0.2 * (25 + s * 25 - s * 50))
        self.assertAlmostEqual(win["BB"], 10.5 + 2.75)
        self.assertAlmostEqual(m.call_need(self.PUT, "BB", "BTN", 9)[0], 9 / 23.25)
        self.assertEqual(m.bounty_pool, 150)                         # (the added half stays in play)

    def test_mystery_is_the_average_envelope(self):
        m = icm.Model(self.ST, [300], bounties=icm.Bounties("mystery", pool=900, left=30))
        self.assertEqual(m.bounty.envelope, 30)
        self.assertEqual(m.cash["BTN"], 30)
        self.assertEqual(m.bounty_pool, 900)
        k = 1 / (300 / 90 + 900 / 90)
        win, _ = m.showdown(dict(self.PUT, BB=10), "BB", "BTN")
        self.assertAlmostEqual(win["BB"], 10.5 + k * (1 - 40.5 / 90) * 30)
        # an envelope alone: the pool is one envelope for each bust still to come at the table
        self.assertEqual(icm.Model(self.ST, [300], bounties=icm.Bounties("mystery", envelope=30)).bounty_pool, 60)

    def test_covering_rule(self):
        """A bounty pays only when the winner covers the busted player; a split busts nobody."""
        m = self.model("ko")
        # BTN (10) beats BB (30): BB doesn't bust, so BTN wins chips only
        _, lose = m.showdown(dict(self.PUT, BB=10), "BB", "BTN")
        self.assertAlmostEqual(lose["BTN"], 10.5)
        self.assertIsNone(m.bust_value("BTN", "BB"))
        self.assertIsNotNone(m.bust_value("BB", "BTN"))
        # equal stacks: whoever wins busts the other
        eq = icm.Model({"A": 20, "B": 20, "C": 50}, [300], bounties=icm.Bounties("ko", {"A": 10, "B": 10, "C": 10}))
        wa, wb = eq.showdown({"A": 20, "B": 20}, "A", "B")
        self.assertAlmostEqual(wa["A"] - 20, wb["B"] - 20)
        self.assertGreater(wa["A"], 20.0)
        # heads-up solver terminals: a split busts nobody, so it is chips (a winner-take-all) for both
        hu = icm.HUPayoffs(eq, ("A", "B"), {"A": 0.0, "B": 0.0}, (0.0, 0.0))
        self.assertEqual(hu._end((20.0, 20.0), None), (0.0, 0.0))
        self.assertAlmostEqual(hu._end((20.0, 20.0), 0)[0], wa["A"])
        # a short stack's win in the solver doesn't take the covering player's bounty
        short = icm.HUPayoffs(m, ("BB", "BTN"), {"SB": 0.5}, (0.0, 0.0))
        self.assertAlmostEqual(short._end((10.0, 10.0), 1)[1], 10.5)

    def test_zero_bounties_change_nothing(self):
        for b in (icm.Bounties("ko", {p: 0 for p in S4}), icm.Bounties("pko"), icm.Bounties("mystery", envelope=0),
                  icm.Bounties("mystery", pool=0, left=10)):
            self.assertFalse(b)
            a, z = icm.Model(S4, [50, 30, 20]), icm.Model(S4, [50, 30, 20], bounties=b)
            self.assertIsNone(z.bounty)
            self.assertEqual(a.key, z.key)
            put = {"CO": 0, "BTN": 25, "SB": 0.5, "BB": 25}
            self.assertEqual(a.showdown(put, "BTN", "BB"), z.showdown(put, "BTN", "BB"))
            self.assertEqual(a.premium({"BTN": 2, "SB": 0.5, "BB": 1}, "BB", "BTN"),
                             z.premium({"BTN": 2, "SB": 0.5, "BB": 1}, "BB", "BTN"))


class TestSolvers(unittest.TestCase):
    def test_river_call_widens_against_a_covered_bounty(self):
        """BB (40bb) faces BTN's 22bb river jam (BTN all in, 25bb at the start: BB covers) with a fixed range: queens
        fold in chips and call against a 40 KO bounty on BTN, their EV up by their equity times k (1 - s') B."""
        board = parse_cards("Ks7d2c9h4s")
        r0 = parse_range("22+,A2s+,KTs+,QJs,ATo+,KJo+")
        r1 = parse_range("AK,KQ,KJs,77,99,44,22,T8s,65s,A3s,A5s,JTs")
        stacks = {"CO": 40, "BTN": 25, "SB": 8, "BB": 40}
        contrib = {"CO": 0.0, "BTN": 25.0, "SB": 0.5, "BB": 3.0}
        qq, share = [], []
        for b in (None, icm.Bounties("ko", {"CO": 10, "BTN": 40, "SB": 10, "BB": 10})):
            m = icm.Model(stacks, [100], (), b)
            cfg = Config(bet_sizes={"river": [0.5]}, raise_sizes={"river": [1.0]})
            cfg.icm = icm.HUPayoffs(m, ("BB", "BTN"), contrib, (3.25, 25.25))
            sol = solve(board, r0, r1, 6.5, 22.0, bets=(0.0, 22.0), to_act=0, cfg=cfg, time_budget=100, max_iters=200)
            qq.append(sol.for_combo(tuple(parse_cards("QhQd"))))
            share.append(sol.range_strategy(r0 * np.prod([1 - MEMBER_FULL[c] for c in board], axis=0))["call"])
        self.assertLess(qq[0]["call"][0], 0.001)
        self.assertGreater(qq[1]["call"][0], 0.999)
        # in chips (one prize) BB folds to 37 left, wins to 65.5 or loses to 15, so the call's EV gives QQ's equity
        eq = (qq[0]["call"][1] + 22.0) / 50.5
        k = 1 / (100 / 113 + 70 / 113)          # 113 chips, 100 in prizes, 70 in bounties
        self.assertAlmostEqual(qq[1]["call"][1] - qq[0]["call"][1], eq * k * (1 - 65.5 / 113) * 40, places=6)
        self.assertGreater(share[1], share[0] + 0.05)

    def test_three_way_terminals_pay_each_bust_to_who_takes_its_last_chips(self):
        """A 3-way all-in (A 10bb, B 20bb, C calls 20 of its 30; D folded 1): the 3-way solver's weights reproduce,
        for each player and each best pot won, the mean over the hands' orders of the outcome's value, with A's bounty
        to whoever wins the main pot and B's to the side pot's winner (A, the main pot's, doesn't cover B)."""
        stacks = {"A": 10, "B": 20, "C": 30, "D": 25}
        bty = icm.Bounties("ko", {"A": 5, "B": 8, "C": 3, "D": 4})
        m = icm.Model(stacks, [50, 30, 20], bounties=bty)
        mw = icm.MWPayoffs(m, ("A", "B", "C"), {"D": 1.0})
        layers = [(31.0, (0, 1, 2)), (20.0, (1, 2))]
        W = mw.weights([((0, 1, 2), [10.0, 20.0, 20.0], layers)])[0]
        put = {"A": 10.0, "B": 20.0, "C": 20.0, "D": 1.0}
        names = "ABC"
        G = {p: {} for p in range(3)}
        for perm in itertools.permutations(range(3)):
            main, side = perm[0], next(x for x in perm if x in (1, 2))
            won = {names[main]: 31.0}
            won[names[side]] = won.get(names[side], 0.0) + 20.0
            ko = []
            if main != 0:
                ko.append(("A", {names[main]: 1.0}))
            if 1 not in (main, side):
                ko.append(("B", {names[side]: 1.0}))
            v = m.value(put, won, ko)
            for p in range(3):
                best = 0 if main == p else 1 if side == p else None
                G[p].setdefault(best, []).append(v[names[p]])
        for p in range(3):
            for best, vals in G[p].items():
                ks = [j for j, (_, e) in enumerate(layers) if p in e and (best is not None and j >= best)]
                got = W[p][0] + sum(W[p][1 + j] for j in ks)
                self.assertAlmostEqual(got, float(np.mean(vals)), places=9, msg=(p, best))
        # C taking everything collects both bounties: more than the chips' worth of it in a model without them
        plain = icm.MWPayoffs(icm.Model(stacks, [50, 30, 20]), ("A", "B", "C"), {"D": 1.0}).weights(
            [((0, 1, 2), [10.0, 20.0, 20.0], layers)])[0]
        self.assertGreater(W[2][0] + W[2][1] + W[2][2], plain[2][0] + plain[2][1] + plain[2][2])


class TestAdvisor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = Advisor(time_budget=1.0, prior_budget=0.5)

    def bb_vs_jam(self, hole, bounties=None):
        return self.A.advise({"players": 6, "bb": 1, "stacks": JAM_STACKS, "hero": "BB", "hole": hole,
                              "preflop": SB_JAM, **({"bounties": bounties} if bounties else {})})

    def test_zero_bounties_change_nothing(self):
        """Zero bounties (any format) answer exactly as no bounties: preflop calls, shoves, opens, defence and a
        heads-up river solve (its solve the same cached one: the same prize model). All but "seconds" and "sampled"
        (a random draw from the mix: Qh9h's call is a hair under 100%, and once in a full run it drew the raise)."""
        jam = dict(players=4, bb=1, stacks={"CO": 30, "BTN": 20, "SB": 15, "BB": 5}, hero="SB", hole="AhJd",
                   preflop=[R("CO", "fold"), R("BTN", "all-in")], payouts=[50, 30, 20])
        spots = [jam, dict(jam, payouts=[1000, 600, 400], field={"players": 20, "chips": 500}),
                 dict(players=4, bb=1, stacks=S4, hero="BTN", hole="Kd9c", preflop=[R("CO", "fold")], payouts=[50, 30, 20]),
                 dict(players=4, bb=1, stacks=S4, hero="SB", hole="Kd9c", preflop=[R("CO", "fold"), R("BTN", "fold")],
                      payouts=[50, 30, 20]),
                 dict(players=4, bb=1, stacks=S4, hero="BB", hole="Jh8h",
                      preflop=[R("CO", "fold"), R("BTN", "raise", 2), R("SB", "fold")], payouts=[50, 30, 20]),
                 dict(players=4, bb=1, stacks=S4, hero="BB", hole="Qh9h", board="Kh7d2c9s4s", payouts=[50, 30, 20],
                      preflop=[R("CO", "fold"), R("BTN", "raise", 2), R("SB", "fold"), R("BB", "call")],
                      flop=[X("BB", "check"), X("BTN", "check")], turn=[X("BB", "check"), X("BTN", "check")],
                      river=[X("BB", "check"), X("BTN", "bet", 3)])]
        zeros = ({"format": "ko", "seats": 0}, {"format": "pko", "seats": {"BB": 0, "SB": 0}},
                 {"format": "mystery", "envelope": 0}, {"format": "ko", "seats": [0, 0, 0, 0]})
        for st in spots:
            a = Advisor(time_budget=1.0, prior_budget=0.5)
            want = a.advise(st)
            self.assertNotIn("error", want)
            want.pop("seconds"); want.pop("sampled", None)
            for z in zeros:
                got = a.advise(dict(st, bounties=z))
                got.pop("seconds"); got.pop("sampled", None)
                self.assertEqual(json.dumps(got, sort_keys=True), json.dumps(want, sort_keys=True), (st["hero"], z))

    def test_push_fold_call_widens_against_a_short_stack_with_a_bounty(self):
        """BB 40bb facing SB's 20bb jam, no payouts (chip EV: the prizes taken as the 60 in bounties): with 10 PKO
        on every head a unit is worth 1.5bb (k = 180 / (60 + 60)) and busting SB pays 5, worth (1 - 60/180) 5 x 1.5 =
        5.0bb, so the call of 19 needs 19 / (40 + 5) = 42.2% against 19 / 40 = 47.5%; A2o (42.5% against the jam)
        calls about half the time, where it folds in chips. With 30 KO on SB's head (10 on the others') busting SB is
        worth 22.5bb and the call needs 19 / 62.5 = 30.4%: BB calls nearly everything."""
        chips, pko, ko = self.bb_vs_jam("Ad2c"), self.bb_vs_jam("Ad2c", {"format": "pko", "seats": 10}), \
            self.bb_vs_jam("Ad2c", {"format": "ko", "seats": {**{p: 10 for p in SIX}, "SB": 30}})
        self.assertNotIn("tournament", chips)
        b = pko["tournament"]["bounties"]
        self.assertEqual((b["format"], b["model"], b["pool"], b["bb_per_unit"]), ("pko", "proportional", 60.0, 1.5))
        self.assertEqual(b["you_bust"]["SB"], {"bb": 5.0, "value": 3.3333, "cash": 5.0, "in_hand": True})
        self.assertIn("calling needs 42.2% equity in prize equity with the bounties against 47.5% in chips "
                      "(a bounty discount of 5.3%)", " ".join(pko["notes"]))
        self.assertIn("chip EV", pko["tournament"]["note"])
        self.assertAlmostEqual(19 / 45, 0.4222, places=4)
        self.assertEqual(chips["recommended"], "fold")
        self.assertGreater(pko["actions"][0]["freq"], 0.3)
        self.assertLess(abs(pko["equity"] - 19 / 45), 0.01)          # (A2o is about indifferent: it mixes)
        self.assertIn("30.4%", note(ko, "tournament: calling needs"))
        self.assertEqual(ko["tournament"]["bounties"]["you_bust"]["SB"]["bb"], 22.5)
        self.assertGreater(call_share(ko), 0.95)
        self.assertGreater(call_share(pko), call_share(chips) + 0.03)     # 21% -> 26% (SB jams tighter too)
        # Kd8c is below what PKO asks: still a fold
        self.assertEqual(self.bb_vs_jam("Kd8c", {"format": "pko", "seats": 10})["recommended"], "fold")

    def test_shoves_widen_when_covering_a_bounty(self):
        """SB 40bb covers BB 12bb: SB's push/fold jams more hands with BB's bounty to win, more the bigger it is."""
        stacks = dict(JAM_STACKS, SB=40, BB=12)
        line = SB_JAM[:4]

        def jams(b):
            r = self.A.advise({"players": 6, "bb": 1, "stacks": stacks, "hero": "SB", "hole": "Kd2c", "preflop": line,
                               **({"bounties": b} if b else {})})
            g = r["grid"]
            i = next(i for i, a in enumerate(g["actions"]) if a in ("all-in", "raise"))
            n = lambda cls: 6 if len(cls) == 2 else 4 if cls.endswith("s") else 12
            return sum(c["w"] * n(k) * c["f"][i] for k, c in g["cells"].items()) / sum(c["w"] * n(k) for k, c in g["cells"].items())
        base = jams(None)
        ten = jams({"format": "ko", "seats": 10})
        big = jams({"format": "ko", "seats": {**{p: 10 for p in SIX}, "BB": 30}})
        self.assertGreater(ten, base + 0.02)
        self.assertGreater(big, ten + 0.02)

    def test_three_way_river(self):
        """SB (30bb) with queens on the river: BTN (12bb) jams after SB and BB (30bb) check. With 80 on BTN's head SB
        raises all in some of the time to isolate BB (the side pot's call or fold) and win BTN's bounty."""
        st = {"players": 4, "bb": 1, "stacks": {"CO": 40, "BTN": 12, "SB": 30, "BB": 30}, "hero": "SB", "hole": "QsQc",
              "board": "Kh7d2c9s4h", "payouts": [50, 30, 20],
              "preflop": [R("CO", "fold"), R("BTN", "raise", 2), R("SB", "call"), R("BB", "call")],
              "flop": [X("SB", "check"), X("BB", "check"), X("BTN", "check")],
              "turn": [X("SB", "check"), X("BB", "check"), X("BTN", "check")],
              "river": [X("SB", "check"), X("BB", "check"), X("BTN", "all-in")]}
        plain = self.A.advise(st)
        ko = self.A.advise(dict(st, bounties={"format": "ko", "seats": {"CO": 20, "BTN": 80, "SB": 20, "BB": 20}}))
        for r in (plain, ko):
            self.assertNotIn("error", r)
            self.assertTrue(r.get("multiway"))
        f = lambda r, a: next(x["freq"] for x in r["actions"] if x["action"] == a)
        self.assertGreater(f(ko, "all-in"), f(plain, "all-in") + 0.2)
        b = ko["tournament"]["bounties"]
        self.assertTrue(b["you_bust"]["BTN"]["in_hand"])
        self.assertEqual(b["you_bust"]["BTN"]["cash"], 80.0)
        self.assertIn("bounties (each bust's to whoever wins the pot that takes the player's last chips)",
                      " ".join(ko["notes"]))

    def test_heads_up_postflop_carries_the_bounties(self):
        st = {"players": 4, "bb": 1, "stacks": S4, "hero": "BB", "hole": "7s6s", "board": "Kh7d2c",
              "preflop": [R("CO", "fold"), R("BTN", "raise", 2), R("SB", "fold"), R("BB", "call")],
              "flop": [X("BB", "check"), X("BTN", "bet", 3)], "payouts": [50, 30, 20],
              "bounties": {"format": "pko", "seats": 25}}
        r = self.A.advise(st)
        self.assertNotIn("error", r)
        b = r["tournament"]["bounties"]
        self.assertTrue(b["you_bust"]["BTN"]["in_hand"])        # 25 against 25: BB covers BTN (and BTN BB)
        self.assertIn("BTN", b["bust_you"])
        self.assertIn("bounties: PKO", note(r, "bounties: PKO"))
        # the one-size view isn't worked out with bounties
        r = self.A.advise(dict(st, flop=[X("BB", "check"), X("BTN", "check")], board="Kh7d2c5s", one_size="all"))
        self.assertIn("bounties", r["error"])

    def test_parsing(self):
        base = {"players": 6, "bb": 1, "stacks": JAM_STACKS, "hero": "BB", "hole": "Ad2c", "preflop": SB_JAM}
        for bad in ({"format": "bogus", "seats": 10}, {"format": "ko", "seats": {"XX": 5}}, {"format": "ko", "seats": -5},
                    {"format": "ko", "seats": [1, 2, 3]}, {"format": "mystery"}, {"format": "mystery", "pool": 100},
                    {"format": "ko", "seats": 10, "field": "x"}, "pko"):
            r = self.A.advise(dict(base, bounties=bad))
            self.assertIn("error", r, bad)
            self.assertIn("bount", r["error"])
        same = [self.A.advise(dict(base, bounties=b))["tournament"]["bounties"]["bounties"] for b in
                ({"format": "PKO", "seats": 10}, {"format": "progressive", "seats": [10] * 6},
                 {"format": "pko", "seats": {p: 10 for p in SIX}}, {"format": "pko", "seats": {p.lower(): "10" for p in SIX}})]
        self.assertTrue(all(s == same[0] for s in same))
        m = self.A.advise(dict(base, bounties={"format": "mystery", "pool": 900, "left": 30}))["tournament"]["bounties"]
        self.assertEqual((m["format"], m["pool"]), ("mystery", 900.0))
        # the command line's --bounties
        from gto.cli import parse_bounties
        self.assertEqual(parse_bounties("pko 10", SIX), {"format": "pko", "seats": "10"})
        self.assertEqual(parse_bounties("ko 10 SB=30 field=12", SIX),
                         {"format": "ko", "seats": {**{p: "10" for p in SIX}, "SB": "30"}, "field": "12"})
        self.assertEqual(parse_bounties("mystery 25", SIX), {"format": "mystery", "envelope": "25"})
        self.assertEqual(parse_bounties("mystery pool=900 left=30", SIX), {"format": "mystery", "pool": "900", "left": "30"})
        ko = self.A.advise(dict(base, bounties=parse_bounties("ko 10 SB=30", SIX)))
        self.assertEqual(ko["tournament"]["bounties"]["bounties"]["SB"], 30.0)
        self.assertEqual(ko["tournament"]["bounties"]["bounties"]["BB"], 10.0)


if __name__ == "__main__":
    unittest.main()
