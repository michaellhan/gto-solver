"""Bet sizes off the tree's abstraction, from one hand (6-max, CO 35bb, QcTdTc 3h: BTN AhJc's 95%-pot turn bet into
CO's check, jammed on): Config.force_allin merges a size that leaves under 15% of the pot behind into the all-in (the
browser's 125% turn bet left 1.5bb of 27); the "size" request puts the size you bet in the street's tree and marks its
row, so a review grades a 95% bet as itself, not as the tree's 75%, at SPRs from 1.3 (that turn) to 18 (a flop), on a
dry and a wet board, a value hand and a bluff; and the solve again after an opponent's rare line keeps the street's own
tree (its shallow sizes), where it took the base tree's."""
import unittest

from gto import native
from gto.cards import cards
from gto.cli import parse_actions
from gto.ranges import parse_range
from gto.solver import Config, solve

X = lambda p: {"pos": p, "action": "check"}
B = lambda p, a: {"pos": p, "action": "bet", "amount": a}
C = lambda p: {"pos": p, "action": "call"}
J = lambda p: {"pos": p, "action": "all-in"}
F = lambda p: {"pos": p, "action": "fold"}
PF = [F("UTG"), {"pos": "HJ", "action": "raise", "to": 3}, C("CO"), C("BTN"), F("SB"), F("BB")]
FL = [X("HJ"), B("CO", 5), C("BTN"), F("HJ")]
HAND = {"players": 6, "hero": "BTN", "hole": "AhJc", "stack": 100, "stacks": {"CO": 35}, "preflop": PF, "flop": FL,
        "board": "QcTdTc3h", "river": []}
SRP = parse_actions("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c")
RIVER = {"players": 6, "stack": 100, "preflop": SRP, "board": "Ks7d2cTh4s", "hero": "BTN",
         "flop": [X("BB"), B("BTN", 1.8), C("BB")], "turn": [X("BB"), X("BTN")], "river": [X("BB")]}
OOP = parse_range("TT+,AQs+,KQs,QJs,JTs,T9s,AQo+")
IP = parse_range("22+,ATs+,KTs+,QTs+,JTs,T9s,98s,ATo+,KJo+,QJo")


def adv():
    return native.advisor("fast", budget=1.5, prior_budget=1.5)


def row(res, **kw):
    return [a for a in res["actions"] if all(a.get(k) == v for k, v in kw.items())]


class TestForceAllin(unittest.TestCase):
    def root(self, cfg, pot=20.5, stack=27.0, **kw):
        return solve(cards("QcTdTc3h"), OOP, IP, pot, stack, cfg=cfg, time_budget=1e9, max_iters=0, **kw).root.names

    def cfg(self, **kw):
        cfg = native.config("fast")
        cfg.bet_sizes["turn"] = [0.33, 0.75, 1.25]
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    def test_near_stack_size_is_the_all_in(self):
        """125% of 20.5 leaves 1.4 of 27 behind (under 15% of the pot after it): the all-in, not a size beside it; 75%
        leaves 11.6 (over 15% of 51) and stays. Off (0, the flop library's trees): both, and the all-in."""
        self.assertEqual(native.config("fast").force_allin, 0.15)
        names = self.root(self.cfg())
        self.assertEqual([n for n in names if n.startswith("bet")], ["bet 33%", "bet 75%"])
        self.assertIn("all-in", names)
        self.assertIn("bet 125%", self.root(self.cfg(force_allin=0.0)))
        self.assertIn("bet 125%", self.root(self.cfg(), stack=90.0))   # (deep: 125% leaves 64.4 behind)

    def test_sizes_that_stay(self):
        """The root's exact bets (root_bets: the one-size view's) and a size the hand took that leaves that little
        (unforced: the 95% bet CO jammed over) stay; a size the hand took that leaves more doesn't split the cache key
        (keep_sizes: one tree, the turn's solve is the river's narrowing)."""
        self.assertIn("bet 125%", self.root(self.cfg(root_bets=[0.33, 1.25])))
        self.assertIn("bet 125%", self.root(self.cfg(unforced={"bet": [1.25]})))
        self.assertNotEqual(self.cfg(unforced={"bet": [1.25]}).sizes_key(), self.cfg().sizes_key())
        self.assertEqual(self.cfg(keep_sizes={"bet": [0.75]}).sizes_key(), self.cfg().sizes_key())
        cfg = adv()._cfg_with_real_sizes(self.turn([X("CO"), B("BTN", 19.5), J("CO")]), "CO", "BTN", 27.0)
        self.assertEqual(cfg.unforced, {"bet": [0.951]})   # (19.5 / 20.5, as the tree has it)
        self.assertIsNone(adv()._cfg_with_real_sizes(self.turn([X("CO"), B("BTN", 6.8)]), "CO", "BTN", 27.0).unforced)

    def turn(self, acts):
        from gto.advisor import StreetState
        return StreetState("turn", ["CO", "BTN"], ["CO", "BTN"], 20.5, {"CO": 8.0, "BTN": 8.0}, actions=list(acts))

    def test_raises(self):
        """Facing a bet, a raise that leaves under 15% of the pot behind is the all-in: one all-in, no raise beside it."""
        names = self.root(self.cfg(), pot=10.0, stack=27.0, bets=(6.0, 0.0), to_act=1)
        self.assertEqual(names.count("all-in"), 1)
        self.assertFalse(any(n.startswith("raise") and float(n.split()[-1]) > 23 for n in names if n[-1].isdigit()))

    def test_tree_key(self):
        """The solve cache tells the trees apart."""
        self.assertNotEqual(self.cfg().sizes_key(), self.cfg(force_allin=0.0).sizes_key())
        self.assertEqual(Config().force_allin, 0.0)


class TestSizeRequest(unittest.TestCase):
    def check_size(self, res, label, amount):
        """The size's row is the one marked, at that amount, and (the solve's EV: a size your hand takes at 0% can't
        beat what it takes) no better than the best row."""
        sized = row(res, size=True)
        self.assertEqual([(a["action"], a["amount_bb"]) for a in sized], [(label, amount)])
        best = max(a["ev_bb"] for a in res["actions"] if "ev_bb" in a)
        if sized[0]["freq"] < 0.01:
            self.assertLessEqual(sized[0]["ev_bb"], best + 0.3)
        return sized[0]

    def test_turn_spr_1_3(self):
        """The hand's turn: 19.5 into 20.5 is "bet 95%" (was graded as the 75% row), the shallow tree (33%, 75% swapped
        for 95%, all-in) has no 125% leaving 1.5bb, and the bluff gives up EV to the third-pot bet or the check."""
        res = adv().advise({**HAND, "turn": [X("CO")], "size": 19.5})
        self.assertEqual(res["street"], "turn")
        r = self.check_size(res, "bet 95%", 19.5)
        self.assertFalse(row(res, action="bet 125%"))
        self.assertFalse(row(res, action="bet 75%"))
        self.assertLess(r["ev_bb"], max(a["ev_bb"] for a in res["actions"] if a is not r) - 0.5)
        # without it the tree's sizes, nothing marked
        plain = adv().advise({**HAND, "turn": [X("CO")]})
        self.assertFalse(row(plain, size=True))
        self.assertTrue(row(plain, action="bet 75%"))

    def test_typical_player(self):
        """Against the typical player too: the row at its size, the best response's EV."""
        res = adv().advise({**HAND, "turn": [X("CO")], "size": 19.5, "opponents": "pool"})
        self.check_size(res, "bet 95%", 19.5)

    def test_river_overbet(self):
        """A 2x river overbet (SPR ~10), a missed draw and top pair: its own row; within 15% of a tree size, that size."""
        for hole in ("QhJh", "Kh9h"):
            res = adv().advise({**RIVER, "hole": hole, "size": 18})
            self.check_size(res, "bet 198%", 18.0)
        res = adv().advise({**RIVER, "hole": "QhJh", "size": 12})
        self.assertEqual([a["action"] for a in row(res, size=True)], ["bet 125%"])   # (11.5: within 15%)

    def test_flop_deep(self):
        """A deep flop (SPR 18, a wet board) the flop library holds: its tree keeps its own sizes, so a half-pot c-bet
        is its 55% (within 15%); a 145% one nothing in it is near isn't marked (the page grades it as the nearest)."""
        hand = {"players": 6, "stack": 100, "preflop": SRP, "board": "9h8h4c", "hero": "BTN", "hole": "AsKd",
                "flop": [X("BB")], "turn": [], "river": []}
        res = adv().advise({**hand, "size": 2.75})
        self.assertEqual(res["street"], "flop")
        self.assertEqual([a["amount_bb"] for a in row(res, size=True)], [3.0])
        res = adv().advise({**hand, "size": 8.0})
        self.assertFalse([a for a in row(res, size=True) if abs(a["amount_bb"] - 8.0) > 1.2])

    def test_not_a_bet(self):
        """A size facing a bet that isn't a raise, one past the stack, a jam: nothing marked."""
        a = adv()
        res = a.advise({**HAND, "turn": [X("CO"), B("BTN", 19.5), J("CO")], "size": 19.5})
        self.assertFalse(row(res, size=True))
        res = a.advise({**HAND, "turn": [X("CO")], "size": 40})
        self.assertFalse(row(res, size=True))


class TestRareResolveTree(unittest.TestCase):
    def test_keeps_street_sizes(self):
        """After CO's pot-sized turn lead (0% in the turn's solve: solved again from the node below it), the solve again
        is the street's tree, the shallow sizes with the lead's 100% swapped in, where it took the base tree's (33, 75,
        125% leaving 1.5bb)."""
        a = adv()
        cfgs, orig = [], a._solve_cached

        def spy(board, r0, r1, pot, stack, budget, bets=(0.0, 0.0), to_act=0, n_raises=0, cfg=None, **kw):
            cfgs.append((len(board), tuple(bets), cfg))
            return orig(board, r0, r1, pot, stack, budget, bets, to_act, n_raises, cfg, **kw)
        a._solve_cached = spy
        res = a.advise({**HAND, "turn": [B("CO", 20.5)]})
        street = [c for n, b, c in cfgs if n == 4 and max(b) == 0]
        again = [c for n, b, c in cfgs if n == 4 and max(b) > 0]
        self.assertTrue(street and again)
        self.assertTrue(any("solved again from here" in n for n in res["notes"]))
        for cfg in again:
            self.assertEqual(cfg.bet_sizes["turn"], street[0].bet_sizes["turn"])
            self.assertNotIn(1.25, cfg.bet_sizes["turn"])


if __name__ == "__main__":
    unittest.main()
