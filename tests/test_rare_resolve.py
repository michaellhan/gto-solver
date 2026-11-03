"""After a line an earlier street's solve rarely takes (advisor.RARE_LINE, read by the betting rules), the play after it
is that street solved again from the node (advisor.RARE_RESOLVE, Advisor._rare_resolver): on a live flop (HJ KhQh's 75%
bet on Ah2h7d, called) and on a library flop that doesn't store the node (the big blind's check in BB vs BTN's 3bet
pot on Kc7d4h, which the library has at 0%: the button's bet and the call are the new solve's, and the flop isn't
solved live from its start). A line the solve takes is walked as before, and a quick answer solves nothing again."""
import unittest

from gto import advisor as A, native
from gto.cli import parse_actions

BET = lambda p, a: {"pos": p, "action": "bet", "amount": a}
CALL = lambda p: {"pos": p, "action": "call"}
CHECK = lambda p: {"pos": p, "action": "check"}
HJ_BTN = parse_actions("UTG:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:f")
BB3_BTN = parse_actions("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c")


def spy(adv):
    """The heads-up solves `adv` asks for: (cards on the board, the bets at its root, the player to act there)."""
    calls, orig = [], adv._solve_cached

    def solve(board, r0, r1, pot, stack, budget, bets=(0.0, 0.0), to_act=0, *a, **kw):
        calls.append((len(board), tuple(round(float(b), 2) for b in bets), to_act))
        return orig(board, r0, r1, pot, stack, budget, bets, to_act, *a, **kw)
    adv._solve_cached = solve
    return calls


def rare(res):
    return [o for o in res.get("off_solution", []) if o["why"] == "rare"]


class TestRareResolve(unittest.TestCase):
    def hand(self, pre, board, hero, hole, flop, **kw):
        return {"players": 6, "stack": 100, "preflop": pre, "board": board, "hero": hero, "hole": hole,
                "flop": list(flop), "turn": [], "river": [], **kw}

    def test_live_flop(self):
        """HJ's 75% bet (~0% in the flop's solve), called: the BTN's answer to it is the flop solved again facing
        the bet, listed with the line."""
        adv = native.advisor("fast", budget=1.5, prior_budget=1.5)
        calls = spy(adv)
        res = adv.advise(self.hand(HJ_BTN, "Ah2h7d4d", "HJ", "KhQh", [BET("HJ", 5.0), CALL("BTN")]))
        self.assertEqual([(o["street"], o["pos"], o["action"], o.get("resolved")) for o in rare(res)],
                         [("flop", "HJ", "bet 75%", True)])
        self.assertIn((3, (0.0, 0.0), 0), calls)   # (the flop's own solve, which the bet is rare in)
        self.assertIn((3, (5.0, 0.0), 1), calls)   # (solved again: the BTN to act facing 5 bb)
        self.assertTrue(any(n.startswith("flop: the play after it solved again from there") for n in res["notes"]))
        self.assertEqual(res["street"], "turn")

    def test_solved_line_walked(self):
        """The third-pot bet the solve takes with most of HJ's range: no rare line, nothing solved again."""
        adv = native.advisor("fast", budget=1.5, prior_budget=1.5)
        calls = spy(adv)
        res = adv.advise(self.hand(HJ_BTN, "Ah2h7d4d", "HJ", "KhQh", [BET("HJ", 2.0), CALL("BTN")]))
        self.assertEqual(rare(res), [])
        self.assertEqual([c for c in calls if c[0] == 3], [(3, (0.0, 0.0), 0)])

    def test_switched_off(self):
        """Without RARE_RESOLVE the BTN's answer is read off the flop's solve, as before."""
        adv = native.advisor("fast", budget=1.5, prior_budget=1.5)
        calls = spy(adv)
        A.RARE_RESOLVE = False
        try:
            res = adv.advise(self.hand(HJ_BTN, "Ah2h7d4d", "HJ", "KhQh", [BET("HJ", 5.0), CALL("BTN")]))
        finally:
            A.RARE_RESOLVE = True
        self.assertEqual([o.get("resolved") for o in rare(res)], [None])
        self.assertEqual([c for c in calls if c[0] == 3], [(3, (0.0, 0.0), 0)])

    def test_library_flop(self):
        """BB vs BTN's 3bet pot on Kc7d4h: the library has the big blind checking 0% and doesn't store the button's
        answer. The check is listed (read by the rules), the flop from it solved with the button to act, never from its
        start, and the library's root read."""
        adv = native.advisor("fast", budget=1.5, prior_budget=1.5)
        calls = spy(adv)
        res = adv.advise(self.hand(BB3_BTN, "Kc7d4h9s", "BB", "QsQh",
                                   [CHECK("BB"), BET("BTN", 6.8), CALL("BB")]))
        self.assertEqual([(o["street"], o["pos"], o["action"], o.get("resolved")) for o in rare(res)],
                         [("flop", "BB", "check", True)])
        self.assertEqual([c for c in calls if c[0] == 3], [(3, (0.0, 0.0), 1)])
        self.assertTrue(any(n.startswith("flop: precomputed solution") for n in res["notes"]))

    def test_quick(self):
        """A quick answer solves no earlier street, again or at all: the library doesn't store the line, so the flop is
        narrowed by the betting rules, as before."""
        adv = native.advisor("fast", budget=1.5, prior_budget=1.5)
        calls = spy(adv)
        res = adv.advise(self.hand(BB3_BTN, "Kc7d4h9s", "BB", "QsQh",
                                   [CHECK("BB"), BET("BTN", 6.8), CALL("BB")], quick=True))
        self.assertEqual([c for c in calls if c[0] == 3], [])
        self.assertIn({"street": "flop", "why": "rules"}, res["off_solution"])


if __name__ == "__main__":
    unittest.main()
