"""A river line the solve takes with under RARE_LINE of the bettor's range is still its equilibrium when the solve has
settled (advisor.RARE_KEEP, Advisor._settled_line): the bettor's range is the solve's, not the rules' value-only range
for a pot-sized bet or more, which folded bluff-catchers the river's exact solve calls (scripts/bench_rare_lines.py).
The hands below are answered as the exact river solve on the same ranges answers them, across positions, stack depths,
pot types, lines (a lead after the turn checked through, a lead after a called c-bet, a bet after your check, a raise
of your lead) and hands (bluff-catchers that call, hands that fold, a value raise)."""
import types
import unittest

from gto import advisor as A, native
from gto.cli import parse_actions

R = lambda p, a, x=None: dict({"pos": p, "action": a}, **({"to": x} if a == "raise" else {"amount": x} if x is not None else {}))
X = lambda p: R(p, "check")
CO_BB5 = parse_actions("HJ:f,CO:r2.5,BTN:f,SB:f,BB:c")
BTN_BB = parse_actions("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c")
UTG_BTN = parse_actions("UTG:r2.5,HJ:f,CO:f,BTN:c,SB:f,BB:f")
SB_BB = parse_actions("UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:c")
BB3_BTN = parse_actions("UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r11,BTN:c")
XX = lambda o, i: [X(o), X(i)]


def hand(pre, board, hero, hole, flop, turn, river, players=6, stack=100):
    return {"players": players, "stack": stack, "preflop": pre, "board": board, "hero": hero, "hole": hole,
            "flop": flop, "turn": turn, "river": river}


class TestRareKeep(unittest.TestCase):
    def setUp(self):
        self.keep = A.RARE_KEEP

    def tearDown(self):
        A.RARE_KEEP = self.keep

    def answer(self, st):
        return native.advisor("fast").advise(st)

    def test_the_overbet_hand(self):
        """5-handed, CO's Ac8c on Ad5d4c 4s 9d, checked through, BB overbets 7.5 into 5.5 (4-6% of BB's range in the
        river's solve): A8 calls, as the river's exact solve calls it 100% (+1.5 bb; postflop-solver too), where the
        rules' value-only range folded it at -1.1. Only the call is asserted: how rare the overbet is here turns on how
        far the flop's and turn's live solves got in their time (with a quarter of Fast's 2.5 s it is a sliver, with
        half 10% or more and not rare), so a loaded machine moved the rest; test_calls and test_value_raise check the
        kept range and the rules' answer it replaces on spots that hold at a quarter to 2.4x the budget."""
        st = hand(CO_BB5, "Ad5d4c4s9d", "CO", "Ac8c", [X("BB"), X("CO")], [X("BB"), X("CO")],
                  [R("BB", "bet", 7.5)], players=5)
        self.assertEqual(self.answer(st)["recommended"], "call")

    def kept(self, res, sliver=False):
        offs = [o for o in res.get("off_solution", []) if o["street"] == "river" and o["why"] == "rare, settled"]
        self.assertEqual(len(offs), 1, res.get("off_solution"))
        self.assertEqual(bool(offs[0].get("sliver")), sliver)

    def keeps(self, st, rec, rules):
        """`st` answered with the solve's range kept (`rec`) and with the rules' range (`rules`)."""
        res = self.answer(st)
        self.kept(res)
        self.assertTrue(res["recommended"].startswith(rec), res["recommended"])
        A.RARE_KEEP = False
        self.assertTrue(self.answer(st)["recommended"].startswith(rules))
        A.RARE_KEEP = self.keep

    # The spots below hold with the time budget anywhere from a quarter of Fast's 2.5 s (a loaded machine) to 2.4x: the
    # line kept at 1.6-6.4% of the bettor's range, never a sliver, the same answer, the rules' answer the other way
    # every time. The exact river solves are on the ranges Fast reached the river with (0.0008-0.007% of the pot).
    def test_calls(self):
        """Hands the exact river solve calls 100%, which the rules' value-only range folded: BB 3bet v BTN, 100bb,
        Jc5c4s 8c Ts checked through, BB leads the pot (5% of its range): JdKs calls (+26.4 bb, 6.3 over its next);
        the same pot, TsJd6d 3h 9h, BB checks and BTN bets 2x pot (4%): TcQc calls (+2.1); 5-handed CO v BB,
        5s5c6d 6s Ks checked through, BB jams (2-6%): 5dAc calls (+19.4)."""
        self.keeps(hand(BB3_BTN, "Jc5c4s8cTs", "BTN", "JdKs", XX("BB", "BTN"), XX("BB", "BTN"),
                        [R("BB", "bet", 22.5)]), "call", "fold")
        self.keeps(hand(BB3_BTN, "TsJd6d3h9h", "BB", "TcQc", XX("BB", "BTN"), XX("BB", "BTN"),
                        [X("BB"), R("BTN", "bet", 45.0)]), "call", "fold")
        self.keeps(hand(CO_BB5, "5s5c6d6sKs", "CO", "5dAc", XX("BB", "CO"), XX("BB", "CO"), [R("BB", "all-in")],
                        players=5), "call", "fold")

    def test_value_raise(self):
        """SB v BB, 100bb, Qc5h8c Js 3c checked through, SB leads 2x pot (2-6% of its range): BB's 2c4c raises (the
        exact river solve raises 100%, +23.5 bb, 6.6 over calling), where against the rules' range it only called."""
        self.keeps(hand(SB_BB, "Qc5h8cJs3c", "BB", "2c4c", XX("SB", "BB"), XX("SB", "BB"), [R("SB", "bet", 12.0)]),
                   "raise", "call")

    def test_folds(self):
        """Hands the exact river solve folds still fold: SB v BB, 50bb, Kc3cTd 9h 8c, the SB leading 2x pot (1%), BB's
        5s6d (call -9.4 bb there); BB 3bet v BTN, 100bb, Qd3d2s Jd Ac, BB leading the pot (1.3%), BTN's JhKc (-3.6)."""
        res = self.answer(hand(SB_BB, "Kc3cTd9h8c", "BB", "5s6d", XX("SB", "BB"), XX("SB", "BB"),
                               [R("SB", "bet", 12.0)], stack=50))
        self.assertEqual(res["recommended"], "fold")
        res = self.answer(hand(BB3_BTN, "Qd3d2sJdAc", "BTN", "JhKc", XX("BB", "BTN"), XX("BB", "BTN"),
                               [R("BB", "bet", 22.5)]))
        self.assertEqual(res["recommended"], "fold")

    def test_sliver(self):
        """A line the settled solve keeps next to none of the range on (under RARE_KEEP_COMBOS) plays the solve's mix,
        without EVs: BTN v BB, 5s8d9d Th 8h, BTN's third-pot c-bet called, the turn checked through, BB leads 2x pot
        (0%): 4c4h folds (the exact solve 91%), where its EVs read call +26.6 bb against those few hands; 50bb, Jd2c4c
        5s 4d checked through, BB leads 136%: TdTh calls (99%). With the pool the rules' range answers, as before."""
        st = hand(BTN_BB, "5s8d9dTh8h", "BTN", "4c4h", [X("BB"), R("BTN", "bet", 2.0), R("BB", "call")],
                  XX("BB", "BTN"), [R("BB", "bet", 19.0)])
        res = self.answer(st)
        self.kept(res, sliver=True)
        self.assertEqual(res["recommended"], "fold")
        self.assertFalse(any("ev_bb" in a for a in res["actions"]), res["actions"])
        self.assertGreater(next(a["freq"] for a in res["actions"] if a["action"] == "fold"), 0.7)
        self.assertGreater(res["villain_range_combos"], 0.5)   # (the rules' range: the solve's is next to none)
        pool = self.answer(dict(st, opponents="pool"))
        self.assertFalse([o for o in pool.get("off_solution", []) if o["why"] == "rare, settled"])
        self.assertTrue(all("ev_bb" in a for a in pool["actions"]))
        res = self.answer(hand(BTN_BB, "Jd2c4c5s4d", "BTN", "TdTh", XX("BB", "BTN"), XX("BB", "BTN"),
                               [R("BB", "bet", 7.5)], stack=50))
        self.assertEqual(res["recommended"], "call")

    def test_settled_line(self):
        """Kept only on the river, for a solve within RARE_KEEP_EXPL of equilibrium; a sliver (under RARE_KEEP_COMBOS)
        only at equilibrium."""
        adv = native.advisor("fast")
        sol = lambda e: types.SimpleNamespace(iters=1, exploitability=lambda: e)
        self.assertTrue(adv._settled_line(sol(0.05), 3.0, "river"))
        self.assertFalse(adv._settled_line(sol(0.05), 3.0, "turn"))
        self.assertFalse(adv._settled_line(sol(0.05), 3.0, "flop"))
        self.assertFalse(adv._settled_line(sol(A.RARE_KEEP_EXPL + 0.5), 3.0, "river"))
        self.assertTrue(adv._settled_line(sol(0.05), A.RARE_KEEP_COMBOS / 2, "river"))           # (a sliver: its mix)
        self.assertFalse(adv._settled_line(sol(0.05), A.RARE_KEEP_COMBOS / 2, "river", True))    # (not for the pool)
        self.assertTrue(adv._settled_line(sol(0.05), 3.0, "river", True))
        A.RARE_KEEP = False
        self.assertFalse(adv._settled_line(sol(0.05), 3.0, "river"))


if __name__ == "__main__":
    unittest.main()
