"""The advisor's catch-all (Advisor.advise, Advisor._rules): a spot whose solve fails is answered by the chart, the
betting rules or equity, marked "fallback", never an error; a cancel still raises; Study answers preflop from the
chart and says it can't after the flop. And the spots scripts/coverage_fuzz.py found unanswered or answered with a
move the page can't play, answered properly (regressions)."""
import unittest
from unittest import mock

from gto.advisor import Advisor, replay_hand, _table_stacks, can_raise
from gto import preflop as pf
from gto.solver import Cancelled

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
       {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
FLOP = {"players": 6, "hero": "BB", "hole": "AhTh", "board": "Td7c2s", "stack": 100, "preflop": PRE,
        "flop": [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 2}],
        "budget": 1.0, "prior_budget": 0.3, "quick": True}
PREFLOP = {"players": 6, "hero": "BB", "hole": "AhTh", "board": "", "stack": 100, "preflop": PRE[:5]}


def _acts(line):
    """'UTG r2.5, HJ c, BB x, SB b3, BTN a, CO f' as the page's actions (as scripts/sanity_sweep.py _acts)."""
    names = {"c": "call", "x": "check", "f": "fold", "a": "all-in"}
    out = []
    for step in filter(None, (x.strip() for x in line.split(","))):
        pos, a = step.split()
        out.append({"pos": pos, "action": names[a]} if a in names else
                   {"pos": pos, "action": "raise", "to": float(a[1:])} if a[0] == "r" else
                   {"pos": pos, "action": "bet", "amount": float(a[1:])})
    return out


def boom(*a, **k):
    raise RuntimeError("injected")


def _fuzz():
    import importlib.util, os
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "coverage_fuzz.py")
    spec = importlib.util.spec_from_file_location("coverage_fuzz", path)
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    return cf


CF = _fuzz()


def legal(test, res, state):
    """The answer is one the page can play here: a recommendation among its actions, finite numbers, every action
    it plays a legal move (scripts/coverage_fuzz.py state_flags; a fallback isn't flagged here)."""
    test.assertEqual(CF.state_flags(res, state), [], (res.get("recommended"), res.get("actions"), res.get("error")))


class TestCatchAll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = Advisor(time_budget=1.0, prior_budget=0.3)

    def check(self, res, by, state):
        self.assertNotIn("error", res)
        self.assertEqual(res["fallback"], by)
        self.assertIn("injected", res["fallback_reason"])
        self.assertIn("boom", res["fallback_where"])
        self.assertTrue(res["notes"][0].startswith("the solver couldn't answer this spot"))
        self.assertIn(res["recommended"], [a["action"] for a in res["actions"]])
        legal(self, res, state)

    def test_postflop_rules(self):
        with mock.patch.object(Advisor, "_postflop", boom):
            res = self.adv.advise(FLOP)
        self.check(res, "rules", FLOP)
        self.assertEqual(res["street"], "flop")
        self.assertEqual(res["to_call_bb"], 2.0)
        self.assertNotEqual(res["recommended"], "fold")   # (top pair, top kicker, facing a third of the pot)
        self.assertFalse([n for n in res["notes"] if "not a solve here" in n])   # (the rules' own note for 5+ way pots)

    def test_postflop_equity_when_the_rules_fail_too(self):
        with mock.patch.object(Advisor, "_postflop", boom), mock.patch.object(Advisor, "_multiway", boom):
            res = self.adv.advise(FLOP)
        self.check(res, "equity", FLOP)
        self.assertIn("the betting rules", res["notes"][0])
        self.assertEqual([a["action"] for a in res["actions"]], ["fold", "call"])

    def test_preflop_chart(self):
        with mock.patch.object(Advisor, "_preflop", boom):
            res = self.adv.advise(PREFLOP)
        self.check(res, "chart", PREFLOP)
        self.assertIn("grid", res)
        self.assertAlmostEqual(sum(a["freq"] for a in res["actions"]), 1.0, delta=0.01)

    def test_preflop_equity_when_the_chart_fails_too(self):
        with mock.patch.object(Advisor, "_preflop", boom), mock.patch("gto.preflop.strategy_for", boom):
            res = self.adv.advise(PREFLOP)
        self.check(res, "equity", PREFLOP)

    def test_unusable_answer(self):
        bad = {"street": "flop", "actions": [{"action": "call", "freq": float("nan")}], "recommended": "call"}
        with mock.patch.object(Advisor, "_postflop", lambda *a, **k: dict(bad)):
            res = self.adv.advise(FLOP)
        self.assertEqual(res["fallback"], "rules")
        self.assertIn("freq is nan", res["fallback_reason"])
        with mock.patch.object(Advisor, "_postflop", lambda *a, **k: {"error": "no decision pending at this point in the tree"}):
            self.assertEqual(self.adv.advise(FLOP)["fallback"], "rules")

    def test_input_errors_stay_errors(self):
        # (the request's own mistakes: no fallback)
        for st in (dict(FLOP, hero="XX"), dict(FLOP, hole="AhTd"), dict(FLOP, hero="CO"), dict(FLOP, stack=-5),
                   dict(FLOP, sizes={"flop": ["x"]})):
            res = self.adv.advise(st)
            self.assertIn("error", res, st)
            self.assertNotIn("fallback_reason", res)

    def test_unreadable_state_is_an_error_not_an_exception(self):
        st = {k: v for k, v in FLOP.items() if k != "hole"}
        res = self.adv.advise(st)
        self.assertIn("error", res)
        self.assertIn("KeyError", res["fallback_reason"])

    def test_cancel_raises(self):
        def cancel(*a, **k):
            raise Cancelled()
        for where, st in (("_postflop", FLOP), ("_preflop", PREFLOP)):
            with mock.patch.object(Advisor, where, cancel):
                with self.assertRaises(Cancelled):
                    self.adv.advise(st)
                with self.assertRaises(Cancelled):
                    self.adv.study({k: v for k, v in st.items() if k not in ("hero", "hole")})
        with mock.patch.object(Advisor, "_postflop", boom), mock.patch.object(Advisor, "_multiway", cancel):
            with self.assertRaises(Cancelled):
                self.adv.advise(FLOP)

    def test_library_only_skips(self):
        with mock.patch.object(Advisor, "_postflop", boom):
            res = self.adv.advise(dict(FLOP, flop=[], library_only=True))
        self.assertIn("skipped", res)

    def test_study(self):
        st = {k: v for k, v in PREFLOP.items() if k not in ("hero", "hole")}
        with mock.patch.object(Advisor, "_preflop", boom):
            res = self.adv.study(st)
        self.assertEqual((res["study"], res["fallback"]), ("node", "chart"))
        self.assertAlmostEqual(sum(a["freq"] for a in res["actions"]), 1.0, delta=0.01)
        with mock.patch.object(Advisor, "_postflop", boom):
            res = self.adv.study({k: v for k, v in FLOP.items() if k not in ("hero", "hole")})
        self.assertIn("Study couldn't solve this spot", res["error"])
        self.assertIn("injected", res["fallback_reason"])

    def test_play(self):
        # a practice opponent's move when the solve for it fails: the seat's own (rules) advice
        st = dict(FLOP, hero="CO", hole="KsKd", flop=[{"pos": "BB", "action": "check"}])
        with mock.patch.object(Advisor, "_postflop", boom):
            mv = self.adv.play(dict(st, hero="BB", hole="AhTh"), "CO", "KsKd", seed=1)
        self.assertIn(mv["action"], ("check", "bet", "all-in"))


class TestFuzzFound(unittest.TestCase):
    """Spots scripts/coverage_fuzz.py found unanswered or answered with a move the page can't play."""
    @classmethod
    def setUpClass(cls):
        cls.adv = Advisor(time_budget=1.0, prior_budget=0.3)

    def answer(self, st):
        res = self.adv.advise(st)
        legal(self, res, st)
        self.assertNotIn("fallback", res)
        return res

    def test_short_all_in_after_the_ante(self):
        # a 1bb small blind posts a 0.15 ante: its all-in is to 0.85, not 1.0 (the raise-to was rounded to a half big
        # blind after it was held to the stack)
        st = {"players": 3, "hero": "SB", "hole": "QdJc", "board": "", "stack": 300.0,
              "stacks": {"BTN": 8.0, "SB": 1.0, "BB": 2.0}, "ante": 0.15, "rake": 0.05, "rake_cap": 3,
              "preflop": [{"pos": "BTN", "action": "raise", "to": 2.0}], "budget": 1.5, "prior_budget": 0.3,
              "speed": "quick"}
        res = self.answer(st)
        self.assertEqual(next(a["amount_bb"] for a in res["actions"] if a["action"] == "all-in"), 0.85)

    # a 5-way river with mixed stacks (7-handed, straddle, ante): the N-player solve plays everyone at the shortest stack
    # behind among those with chips (BTN's 77.5 after its call), so a deeper player's raise there was to 77.5 where the
    # smallest raise is to 144, and a deeper player facing that raise was priced at a call of 5 instead of 71.5
    RIVER5 = {"players": 7, "stack": 100.0, "ante": 0.5, "straddle": True, "board": "8h8s6h2sQh",
              "stacks": {"UTG": 184.5, "LJ": 81.0, "HJ": 165.5, "CO": 1.5, "BTN": 86.0, "SB": 132.0, "BB": 26.1},
              "preflop": [{"pos": "LJ", "action": "raise", "to": 5.0}, {"pos": "HJ", "action": "raise", "to": 8.0},
                          {"pos": "CO", "action": "call"}, {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"},
                          {"pos": "BB", "action": "fold"}, {"pos": "UTG", "action": "call"}, {"pos": "LJ", "action": "call"}],
              "flop": [{"pos": p, "action": "check"} for p in ("UTG", "LJ", "HJ", "BTN")],
              "turn": [{"pos": p, "action": "check"} for p in ("UTG", "LJ", "HJ", "BTN")],
              "river": [{"pos": "UTG", "action": "check"}, {"pos": "LJ", "action": "check"}, {"pos": "HJ", "action": "check"},
                        {"pos": "BTN", "action": "bet", "amount": 1.0}, {"pos": "UTG", "action": "call"},
                        {"pos": "LJ", "action": "all-in"}, {"pos": "HJ", "action": "call"}, {"pos": "BTN", "action": "call"}],
              "budget": 0.8, "prior_budget": 0.3, "speed": "quick", "quick": True}

    def test_multiway_raise_under_the_smallest(self):
        res = self.answer(dict(self.RIVER5, hero="UTG", hole="9hJh"))
        self.assertTrue(any("smallest raise you can make (to 144)" in n for n in res["notes"]), res["notes"])
        for a in res["actions"]:
            if a["action"].startswith(("raise", "bet")):
                self.assertGreaterEqual(a["amount_bb"], 144.0)

    def test_multiway_call_over_the_shortest_stack(self):
        st = dict(self.RIVER5, hero="HJ", hole="4c6s",
                  river=self.RIVER5["river"] + [{"pos": "UTG", "action": "raise", "to": 144.0}])
        res = self.answer(st)
        self.assertEqual(next(a["amount_bb"] for a in res["actions"] if a["action"] == "call"), 71.5)

    def test_three_way_raise_over_what_others_can_match(self):
        # a 3-way flop: the deep CO's raise as the 3-way solve plays it is all-in to 18 (the most another can match),
        # where the smallest raise is to 29: the betting rules answer, their raise shown at 29 (the same move). (The
        # browser worker's advisor, which solves a quick 3-way flop)
        st = {"players": 4, "hero": "CO", "hole": "5sJc", "board": "Jh3s2d", "stack": 100.0,
              "stacks": {"CO": 211.5, "BTN": 24.0, "SB": 22.3, "BB": 100.0},
              "preflop": _acts("CO r6, BTN c, SB c, BB f"), "flop": _acts("SB b14.5"),
              "budget": 1.5, "prior_budget": 0.3, "speed": "quick", "quick": True}
        w = CF.worker_advisors()["fast"]
        res = w.advise(st)
        legal(self, res, st)
        self.assertNotIn("fallback", res)
        res2 = w.advise(dict(st, stacks=dict(st["stacks"], BTN=40.0)))
        self.assertTrue(res2.get("solved"), res2.get("notes"))   # (with chips for the raise: the solve)
        self.assertTrue(any("smallest raise you can make (to 29)" in n for n in res["notes"]), res["notes"])
        self.assertIn("raise to 29", [a["action"] for a in res["actions"]])

    def test_rules_raise_held_to_the_deepest_opponent(self):
        # a 4-way river where the deep BB faces a bet: the betting rules play its raise to 56 (what the deepest
        # opponent can match), shown as the smallest raise the page takes, to 90
        st = {"players": 9, "hero": "BB", "hole": "7dKh", "board": "2d4c6h8hTd", "stack": 100.0, "stacks": {"BB": 166.5},
              "preflop": _acts("UTG c, UTG1 c, UTG2 f, LJ f, HJ f, CO c, BTN r3.5, SB f, BB c, UTG c, UTG1 c, CO c"),
              "flop": _acts("BB x, UTG x, UTG1 x, CO x, BTN b6.5, BB c, UTG c, UTG1 c, CO c"),
              "turn": _acts("BB x, UTG x, UTG1 x, CO x, BTN b34, BB c, UTG c, UTG1 c, CO f"),
              "river": _acts("BB x, UTG x, UTG1 x, BTN b45"), "budget": 0.8, "prior_budget": 0.3, "speed": "quick",
              "quick": True}
        res = self.answer(st)
        self.assertEqual([a["amount_bb"] for a in res["actions"] if a["action"].startswith("raise")], [90.0])

    def test_mw_fits(self):
        from gto.advisor import _mw_fits
        T = pf.Table(pf.SEATS[7], 0.5, True, None, 100.0, self.RIVER5["stacks"])
        st = replay_hand(T, 100.0, {k: self.RIVER5[k] for k in ("preflop", "flop", "turn", "river")}, 4)[-1]
        stk = _table_stacks(T, 100.0)
        rem = [p for p in st.order if p in st.active]
        self.assertTrue(_mw_fits(rem, st, stk, "UTG"))       # (every bet within BTN's 77.5)
        st.bets["UTG"], st.cur_bet = 144.0, 144.0
        self.assertFalse(_mw_fits(rem, st, stk, "HJ"))       # (HJ, deeper, faces 144)
        self.assertTrue(_mw_fits(rem, st, stk, "BTN"))       # (BTN can only call all-in: its price is exact)

    def test_rules_raise_of_the_whole_stack_is_all_in(self):
        # six in on the flop, four of them all-in: the big blind's raise over a 1.5 bet is its whole stack, 1.55, so
        # it's all-in (as "raise to 1.55" the page took it as a raise to 1.5, not above the bet)
        st = {"players": 8, "hero": "BB", "hole": "8c4d", "board": "2c9d6c", "stack": 10.0, "ante": 0.25, "straddle": True,
              "stacks": {"UTG": 1.5, "UTG1": 10.0, "LJ": 1.0, "HJ": 24.8, "CO": 2.0, "BTN": 151.5, "SB": 1.5, "BB": 6.8},
              "preflop": _acts("UTG1 r5, LJ c, HJ f, CO c, BTN f, SB c, BB c"), "flop": _acts("BB x, UTG1 b1.5"),
              "budget": 1.5, "prior_budget": 0.3, "speed": "quick", "quick": True}
        res = self.answer(st)
        self.assertEqual([(a["action"], a["amount_bb"]) for a in res["actions"] if a["action"] not in ("fold", "call")],
                         [("all-in", 1.55)])

    def test_straddle_option_after_an_all_in_for_less(self):
        # a 2bb stack's all-in under the straddle (1.875 after the ante) is a short call, not a jam to face: the
        # straddler, called by three, has nothing to call (it was answered call or fold, as facing a jam)
        st = {"players": 5, "hero": "HJ", "hole": "2h6s", "board": "", "stack": 100.0, "ante": 0.125, "straddle": True,
              "stacks": {"HJ": 12.0, "CO": 2.0, "BTN": 100.0, "SB": 14.5, "BB": 5.0},
              "preflop": _acts("CO a, BTN c, SB c, BB c"), "speed": "quick"}
        res = self.answer(st)
        self.assertNotIn("call", [a["action"] for a in res["actions"]])
        self.assertNotIn("fold", [a["action"] for a in res["actions"]])
        self.assertIn("check", [a["action"] for a in res["actions"]])

    def test_legal_to(self):
        from gto.advisor import _legal_to
        T = pf.Table(pf.SEATS[6], 0.0, False, None, 100.0)
        st = replay_hand(T, 100.0, {"preflop": [{"pos": "UTG", "action": "raise", "to": 2.25}]}, 1)[0]
        self.assertEqual(_legal_to(st, "HJ", 3.6, 50.0), 3.5)     # nearest half big blind (min 3.5)
        self.assertEqual(_legal_to(st, "HJ", 3.2, 50.0), 3.5)     # held to the smallest raise, 3.5
        self.assertEqual(_legal_to(st, "HJ", 6.25, 50.0), 6.0)    # half to even, as the rules always sized
        self.assertEqual(_legal_to(st, "HJ", 6.75, 50.0), 7.0)
        self.assertEqual(_legal_to(st, "HJ", 80.0, 50.0), 50.0)   # all-in
        self.assertEqual(_legal_to(st, "HJ", 3.0, 3.1), 3.1)      # all-in for less than a raise


if __name__ == "__main__":
    unittest.main()
