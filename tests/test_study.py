"""Study mode (Advisor.study, the page's solution browser): the hand as recorded, answered for whoever acts next
with their whole range: per hand class and combo the mix, EVs and equities, and what the hand needs next where
nobody acts."""
import gc
import unittest
from unittest import mock

import numpy as np

from gto import advisor as advisor_mod
from gto import solver as solver_mod
from gto.breakdown import draw_category, made_category
from gto.cards import COMBO_INDEX, cards
from gto.native import advisor

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
def combos(cls):
    """The class's combos: pairs 6, suited 4, offsuit 12."""
    return 6 if len(cls) == 2 else 4 if cls.endswith("s") else 12


class TestStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adv = advisor("fast")

    @classmethod
    def tearDownClass(cls):
        # (its cached solves: the suite runs in one process, where a class's advisor holds them to the end)
        del cls.adv
        gc.collect()

    def study(self, **st):
        return self.adv.advise(dict({"players": 6, "stack": 100, "study": True}, **st))

    def check_totals(self, r):
        acts = r["actions"]
        self.assertAlmostEqual(sum(a["freq"] for a in acts), 1.0, delta=0.01)
        self.assertEqual(r["grid"]["actions"], [a["action"] for a in acts])
        # the range's mix per action is the grid's, weighted by each class's combos
        tot = sum(c["w"] * combos(k) for k, c in r["grid"]["cells"].items())
        self.assertAlmostEqual(tot, r["range_combos"], delta=0.01 * tot + 1)
        for i, a in enumerate(acts):
            share = sum(c["w"] * combos(k) * c["f"][i] for k, c in r["grid"]["cells"].items()) / tot
            self.assertAlmostEqual(share, a["freq"], delta=0.01)

    def test_preflop_whole_range(self):
        r = self.study()
        self.assertEqual((r["study"], r["street"], r["actor"]), ("node", "preflop", "UTG"))
        self.assertEqual(r["range_combos"], 1326.0)
        self.check_totals(r)
        self.assertEqual(r["grid"]["cells"]["AA"]["f"][r["grid"]["actions"].index("raise")], 1.0)
        r = self.study(preflop=PRE[:4])
        self.assertEqual(r["actor"], "SB")
        self.assertIn("call", r["grid"]["actions"])
        self.check_totals(r)
        r = self.study(stack=10)   # (a jam: "all-in" in the grid as in the actions, its colour and legend)
        self.assertEqual(r["grid"]["actions"], ["all-in", "fold"])
        self.check_totals(r)

    def test_next_card_and_over(self):
        self.assertEqual(self.study(preflop=PRE), {"study": "deal", "street": "flop", "allin": False})
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "check"}]
        self.assertEqual(self.study(preflop=PRE, board="Kd7c2h", flop=flop)["street"], "turn")
        self.assertEqual(self.study(preflop=PRE[:5] + [{"pos": "BB", "action": "fold"}]),
                         {"study": "over", "street": "preflop", "winner": "BTN"})
        jam = PRE[:5] + [{"pos": "BB", "action": "all-in"}, {"pos": "BTN", "action": "call"}]
        self.assertEqual(self.study(preflop=jam), {"study": "deal", "street": "flop", "allin": True})
        self.assertEqual(self.study(preflop=jam, board="Kd7c2h4s9h")["study"], "over")
        self.assertIn("error", self.study(preflop=PRE[:3] + [{"pos": "BTN", "action": "raise", "to": 2.5}],
                                         board="Kd7c2h"))

    def test_flop_from_the_library(self):
        r = self.study(preflop=PRE, board="Kd7c2h", flop=[{"pos": "BB", "action": "check"}])
        self.assertEqual((r["actor"], r["opponent"], r["position"]), ("BTN", "BB", "IP"))
        self.assertEqual(r["precomputed"]["spot"], "srp_btn_bb")
        self.check_totals(r)
        self.assertEqual(r["pot_bb"], 5.5)
        A = len(r["actions"])
        # each class's numbers are its combos', weighted by their reach
        for cls in ("AKo", "77", "QJs"):
            c, cs = r["grid"]["cells"][cls], r["combos"][cls]
            w = np.array([x[1] for x in cs])
            self.assertAlmostEqual(w.sum() / combos(cls), c["w"], delta=0.002)
            for a in range(A):
                self.assertAlmostEqual((w * [x[2][a] for x in cs]).sum() / w.sum(), c["f"][a], delta=0.002)
                self.assertAlmostEqual((w * [x[3][a] for x in cs]).sum() / w.sum(), c["ev"][a], delta=0.01)
            self.assertTrue(all(len(x[0]) == 4 and 0 <= x[4] <= 1 for x in cs))
        # each combo's made hand and draw (the Breakdown's categories, for the page's filter)
        made, draws = [k for k, _ in r["categories"]["made"]], [k for k, _ in r["categories"]["draws"]]
        board = cards("Kd7c2h")
        for cls, want in (("AKo", "top_tk"), ("77", "set"), ("KQs", "top_gk"), ("AA", "overpair"), ("QJs", None)):
            for x in r["combos"].get(cls, []):
                hole = cards(x[0])
                self.assertEqual(made[x[5]], made_category(hole, board), x[0])
                self.assertEqual(draws[x[6]], draw_category(hole, board), x[0])
                if want:
                    self.assertEqual(made[x[5]], want, x[0])
        # no combo holds a board card; strong hands have the equity, and the two ranges' equities add up to one
        self.assertFalse(any("Kd" in x[0] or "7c" in x[0] for cs in r["combos"].values() for x in cs))
        self.assertGreater(r["grid"]["cells"]["77"]["eq"], 0.9)
        self.assertAlmostEqual(r["range_eq"] + r["opponent_eq"], 1.0, delta=0.01)
        self.assertAlmostEqual(r["opponent_combos"], sum(c["w"] * combos(k) for k, c in r["opponent_grid"]["cells"].items()),
                               delta=1.0)
        # at equilibrium the mix's EV is the best action's, within the solve's distance from it
        for k, c in r["grid"]["cells"].items():
            self.assertLess(max(c["ev"]) - c["evm"], 0.05 * r["pot_bb"] + 0.01, k)

    def test_rough_ev(self):
        """An action the range takes under advisor.RARE_BRANCH is marked "rough_ev", as the advice marks it for a hand
        (the opponent's answer to it is barely trained: the page shows its EVs with a "≈"); nothing else changes."""
        st = dict(preflop=PRE, board="Kd7c2h", flop=[{"pos": "BB", "action": "check"}])
        root, r = self.study(**dict(st, flop=[])), self.study(**st)
        with mock.patch.object(advisor_mod, "RARE_BRANCH", 0.3):   # (the same stored flop: BTN bets 75% 26%)
            t = self.study(**st)
            adv = self.adv.advise(dict({"players": 6, "stack": 100, "hero": "BTN", "hole": "AsJc"}, **st))
        for res, bar in ((root, advisor_mod.RARE_BRANCH), (r, advisor_mod.RARE_BRANCH), (t, 0.3)):
            for a in res["actions"]:
                self.assertIn(a.get("rough_ev"), (None, True))   # (the key only where it holds)
                share = a["combos"] / res["range_combos"]
                if abs(share - bar) > 0.002:
                    self.assertEqual("rough_ev" in a, share < bar, (a, bar))
        self.assertEqual([a.get("rough_ev") for a in root["actions"]], [None, True, True])   # (BB checks it all)
        self.assertEqual([a.get("rough_ev") for a in t["actions"]], [None, None, True])
        self.assertEqual([a.get("rough_ev") for a in t["actions"]], [a.get("rough_ev") for a in adv["actions"]])
        strip = lambda res: dict(res, seconds=None, actions=[{k: v for k, v in a.items() if k != "rough_ev"}
                                                             for a in res["actions"]])
        self.assertEqual(strip(r), strip(t))

    def test_hero_and_exploits_ignored(self):
        st = dict(preflop=PRE, board="Kd7c2h", flop=[{"pos": "BB", "action": "check"}])
        a = self.study(**st)
        b = self.study(hero="BB", hole="KhKs", opponents="pool", locks={"BB": {"fold_to_bet": 30}}, **st)
        self.assertEqual(a["grid"], b["grid"])
        self.assertEqual(b["actor"], "BTN")

    def test_tournament(self):
        """A tournament's payouts and bounties reach Study: preflop the charts' mix as the advice re-prices it for the
        whole range (ICM), after the flop the solves pay prize equity; the notes name the player to act. Without a
        tournament (no payouts, or one prize: chips are prizes) the answers are the same as before."""
        strip = lambda r: {k: v for k, v in r.items() if k != "seconds"}
        pay = [50, 30, 20]
        # BB facing BTN's open at 20bb: ICM's risk premium jams less (20% of hands in chips, 15.5% here: the rest call)
        chips = self.study(stack=20, preflop=PRE[:5])
        icm = self.study(stack=20, preflop=PRE[:5], payouts=pay)
        self.assertEqual((icm["actor"], icm["street"]), ("BB", "preflop"))
        self.assertEqual([a["action"] for a in icm["actions"]], ["all-in", "call", "fold"])
        self.assertAlmostEqual(sum(a["freq"] for a in icm["actions"]), 1.0, delta=0.01)
        jam = lambda r: next(a["freq"] for a in r["actions"] if a["action"] == "all-in")
        self.assertLess(jam(icm), jam(chips) - 0.03)
        self.assertTrue(icm["notes"][0].startswith("tournament: ICM, BB's prize equity"), icm["notes"][0])
        self.assertIn("equity_share", icm["tournament"])
        self.assertFalse([n for n in icm["notes"] if n.startswith("equity ")])   # (nothing about one hand)
        for same in ([], [100]):
            self.assertEqual(strip(self.study(stack=20, preflop=PRE[:5], payouts=same)), strip(chips), same)
        # bounties: the note is the actor's
        pko = self.study(stack=20, preflop=PRE[:5], bounties={"format": "pko", "seats": 10})
        self.assertTrue(pko["notes"][1].startswith("bounties: PKO"), pko["notes"][1])
        self.assertIn("BTN covers BB (BB's bounty 10)", pko["notes"][1])
        self.assertNotIn(" you", pko["notes"][1])
        # after the flop: the river solved in prize equity
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "check"}]
        st = dict(stack=30, preflop=PRE, board="Ks7d2cTh5s", flop=flop, turn=flop, budget=1.0)
        chips, icm = self.study(**st), self.study(payouts=pay, **st)
        self.assertEqual((icm["street"], icm["actor"]), ("river", "BB"))
        self.check_totals(icm)
        self.assertNotEqual(icm["grid"], chips["grid"])
        self.assertTrue(icm["notes"][0].startswith("tournament: ICM, BB's prize equity"))
        self.assertNotIn("tournament", chips)
        self.assertEqual(strip(self.study(payouts=[100], **st))["grid"], chips["grid"])

    def test_breakdown(self):
        r = self.study(preflop=PRE, board="Kd7c2h", breakdown=True)
        self.assertEqual(r["actor"], "BB")
        self.assertEqual(r["breakdown"]["actions"], r["grid"]["actions"])
        self.assertNotIn("mine", r["breakdown"]["hero"])   # (no hand of ours)

    def test_turn_solved_live(self):
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "to": 1.8},
                {"pos": "BB", "action": "call"}]
        r = self.study(preflop=PRE, board="Kd7c2h5s", flop=flop, turn=[{"pos": "BB", "action": "check"}], budget=1.0)
        self.assertEqual((r["street"], r["actor"]), ("turn", "BTN"))
        self.assertNotIn("precomputed", r)
        self.assertGreater(r["solver"]["iterations"], 0)
        self.check_totals(r)
        self.assertAlmostEqual(r["pot_bb"], 9.1, delta=0.01)
        self.assertTrue(r["categories"]["draws"])
        river = self.study(preflop=PRE, board="Kd7c2h5s9d", flop=flop, turn=[{"pos": "BB", "action": "check"},
                           {"pos": "BTN", "action": "check"}], budget=1.0)
        self.assertEqual((river["street"], river["categories"]["draws"]), ("river", []))   # (none on the river)

    def test_only_combos_in_the_range(self):
        """A solve leaves a combo out of a player's arithmetic where its range has no weight (Solver._support): its
        strategy is uniform and its EVs 0 there, so no answer may show one. Study lists the actor's combos in its range
        at the node (each in the solve's window), the grid and the breakdown weigh by it, and a hand of ours outside
        our range is solved at a near-zero weight (Advisor._with_hero), its mix and EVs worked out."""
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "to": 1.8},
                {"pos": "BB", "action": "call"}]
        st = dict(preflop=PRE, board="Kd7c2h5s", flop=flop, turn=[{"pos": "BB", "action": "check"}], budget=1.0)
        # (windows whatever they leave out: WINDOW_MAX 1, where a turn's hold its river classes' images; and a
        # new advisor, whose solves are its own)
        p = mock.patch.object(solver_mod, "WINDOW_MAX", 1.0)
        p.start()
        self.addCleanup(p.stop)
        adv = advisor("fast")
        r = adv.advise(dict({"players": 6, "stack": 100, "study": True, "breakdown": True}, **st))
        sol = list(adv._cache.values())[-1]
        S, active = sol.solver, list(sol.ctx.active)
        self.assertIsNotNone(S._perm)                          # (some combos left out)
        listed = 0
        for cls, rows in r["combos"].items():
            for row in rows:
                i = active.index(COMBO_INDEX[tuple(sorted(cards(row[0])))])
                self.assertTrue(S._win[1][i], row[0])           # (BTN: the turn solve's player 1)
                listed += 1
        self.assertGreater(listed, 50)
        self.assertLessEqual(set(r["combos"]), set(r["grid"]["cells"]))
        self.assertAlmostEqual(r["breakdown"]["hero"]["combos"], r["range_combos"], delta=0.2)
        # our hand outside our range: 7h2d isn't in BTN's opens
        adv = advisor("fast")
        a = adv.advise(dict({"players": 6, "stack": 100, "hero": "BTN", "hole": "7h2d"}, **st))
        self.assertTrue(any("isn't in your range" in n for n in a["notes"]), a["notes"])
        sol = list(adv._cache.values())[-1]
        i = list(sol.ctx.active).index(COMBO_INDEX[tuple(sorted(cards("7h2d")))])
        self.assertTrue(sol.solver._win[1][i])
        self.assertAlmostEqual(sum(x["freq"] for x in a["actions"]), 1.0, delta=0.01)
        self.assertTrue(any(abs(x["ev_bb"]) > 0.05 for x in a["actions"]))

    def test_runouts(self):
        """A closed flop's runout report: the turn's first decision on every turn card, and the second player's
        after a check, read off a live solve of the flop (the library's entries hold the flop only)."""
        flop = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "to": 1.8},
                {"pos": "BB", "action": "call"}]
        r = self.study(preflop=PRE, board="Kd7c2h", flop=flop, runouts=True, budget=1.0)
        self.assertEqual((r["study"], r["street"], r["first"]["actor"], r["after_check"]["actor"]),
                         ("runouts", "turn", "BB", "BTN"))
        self.assertGreater(r["solver"]["iterations"], 0)
        for part in (r["first"], r["after_check"]):
            self.assertEqual(part["actions"][0], "check")
            cards_in = sorted(c for k in part["cards"] for c in k["cards"])
            self.assertEqual(len(cards_in), 49)
            self.assertFalse({"Kd", "7c", "2h"} & set(cards_in))
            for k in part["cards"]:
                self.assertIn(k["card"], k["cards"])
                self.assertAlmostEqual(sum(k["f"]), 1.0, delta=0.01)
            self.assertAlmostEqual(sum(part["avg"]), 1.0, delta=0.01)
        bets = {k["card"]: sum(k["f"][1:]) for k in r["after_check"]["cards"]}
        self.assertLess(bets["7d"], max(bets.values()))   # (the board pairing the caller's 7s: the least barrelled)
        # a check-through flop has one too; a flop still being bet doesn't
        r = self.study(preflop=PRE, board="Kd7c2h", flop=[flop[0], {"pos": "BTN", "action": "check"}],
                       runouts=True, budget=1.0)
        self.assertEqual(r["first"]["actor"], "BB")
        self.assertIn("error", self.study(preflop=PRE, board="Kd7c2h", flop=flop[:2], runouts=True, budget=1.0))
        # a closed turn's: the river's first decision on every card, off a live solve of the turn
        turn = [{"pos": "BB", "action": "check"}, {"pos": "BTN", "action": "bet", "to": 5.6}, {"pos": "BB", "action": "call"}]
        r = self.study(preflop=PRE, board="Kd7c2h5s", flop=flop, turn=turn, runouts=True, budget=1.0)
        self.assertEqual((r["study"], r["street"], r["first"]["actor"], r["after_check"]["actor"]),
                         ("runouts", "river", "BB", "BTN"))
        for part in (r["first"], r["after_check"]):
            cards_in = sorted(c for k in part["cards"] for c in k["cards"])
            self.assertEqual(len(cards_in), 48)
            self.assertEqual(len(set(cards_in)), 48)
            self.assertFalse({"Kd", "7c", "2h", "5s"} & set(cards_in))
            for k in part["cards"]:
                self.assertAlmostEqual(sum(k["f"]), 1.0, delta=0.01)
            self.assertAlmostEqual(sum(part["avg"]), 1.0, delta=0.01)
        self.assertIn("error", self.study(preflop=PRE, board="Kd7c2h5s", flop=flop, turn=turn[:2], runouts=True,
                                          budget=1.0))

    def test_four_way_later_at_once(self):
        """Later 4-way nodes: Study's answer at once off the street's own earlier solve (Advisor._study_mw_prev), the
        ranges the node's own solve starts from (4-way streets narrowed by the betting rules). None at the first
        decision (no 4-way library), once the node's own solve is in hand, facing a bet (the 4-way tree gives only its
        first decision's player a raise) or on the turn's first decision (a 4-way flop's view of the turn was too far
        from the turn's own answer)."""
        adv = advisor("fast")
        pre = [PRE[0], {"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "call"},
               {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
        st = {"players": 6, "stack": 100, "preflop": pre, "board": "Js6cKs", "study": True}
        X = lambda p: {"pos": p, "action": "check"}
        with mock.patch.object(advisor_mod, "MW_STUDY_FLOP_BUDGET", 3.0):
            self.assertIsNone(adv.instant(st))
            root = adv.advise(st)
            self.assertEqual((root["actor"], len(root["opponents"])), ("BB", 3))
            x = dict(st, flop=[X("BB")])
            r = adv.instant(x)
            self.assertEqual((r["actor"], r["approximate"]["source"]), ("HJ", "solve"))
            self.assertRegex(r["approximate"]["label"], r"^this street's 4-way solve from BB's decision \(\d+ iterations")
            self.assertTrue(r["range_ev_bb"] is None and all(a["ev_bb"] is None for a in r["actions"]))
            self.check_totals(r)
            live = adv.advise(x)
            self.assertEqual([a["action"] for a in r["actions"]], [a["action"] for a in live["actions"]])
            for a, b in zip(r["opponents"], live["opponents"]):
                self.assertEqual(a["seat"], b["seat"])
                self.assertAlmostEqual(a["combos"], b["combos"], delta=0.01 * b["combos"] + 0.1)
            for a, b in zip(r["actions"], live["actions"]):
                self.assertAlmostEqual(a["freq"], b["freq"], delta=0.1)
            self.assertIsNone(adv.instant(x))
            # facing BB's bet: none (the root's tree gives HJ no raise there; the node's own solve does)
            bet = root["actions"][1]
            self.assertTrue(bet["action"].startswith("bet"))
            fb = dict(st, flop=[{"pos": "BB", "action": "bet", "to": bet["amount_bb"]}])
            self.assertIsNone(adv.instant(fb))
            self.assertIn("raise", " ".join(a["action"] for a in adv.advise(fb)["actions"]))
            # the turn's first decision: none (the flop's view of a 4-way turn)
            self.assertIsNone(adv.instant(dict(st, board="Js6cKs7h", flop=[X(p) for p in ("BB", "HJ", "CO", "BTN")])))
        del adv
        gc.collect()

    def test_three_way_later_at_once(self):
        """Later 3-way nodes on a flop the library doesn't store, and the turn: Study's answer at once (Advisor.instant
        with "study") read off the hand's earlier solve (Advisor._study_mw_prev): checked to, the first decision's
        solve (preferred to the stored flops most like it); at the turn's first decision the flop's (by hand buckets,
        one bet size), at its later ones the turn's first decision's once solved; the mix alone, the ranges as the
        node's own solve starts from them. None once the node's own solve is in hand, at the turn's later decisions
        before a turn solve, on a river, or on a library flop's turn (its entries hold the flop)."""
        adv = advisor("fast")   # (its own: no solves of the class's other tests in hand)
        adv.mw_budget = 2.5     # (the turn's solves: Fast's 5 s natively)
        pre = PRE[:3] + [{"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "call"},
                         {"pos": "BB", "action": "call"}]
        st = {"players": 6, "stack": 100, "preflop": pre, "board": "Kd8c3h", "study": True}
        X = lambda p: {"pos": p, "action": "check"}

        def no_values(r):
            self.assertTrue(r["range_ev_bb"] is None and r["range_eq"] is None and all(o["eq"] is None for o in r["opponents"]))
            self.assertTrue(all(a["ev_bb"] is None for a in r["actions"]))
            self.assertTrue(all(x[3] is None and x[4] is None for rows in r["combos"].values() for x in rows))
            self.check_totals(r)

        def alike(inst, live, tol):
            self.assertEqual([o["seat"] for o in inst["opponents"]], [o["seat"] for o in live["opponents"]])
            for a, b in zip(inst["opponents"], live["opponents"]):   # (the ranges the node's own solve starts from)
                self.assertAlmostEqual(a["combos"], b["combos"], delta=0.01 * b["combos"] + 0.1)
            self.assertAlmostEqual(inst["range_combos"], live["range_combos"], delta=0.02 * live["range_combos"] + 0.5)
            fi, fl = ({a["action"].split()[0]: 0.0 for a in r["actions"]} for r in (inst, live))
            for r, f in ((inst, fi), (live, fl)):
                for a in r["actions"]:
                    f[a["action"].split()[0]] += a["freq"]
            for k in set(fi) | set(fl):
                self.assertAlmostEqual(fi.get(k, 0.0), fl.get(k, 0.0), delta=tol, msg=(k, fi, fl))

        with mock.patch.object(advisor_mod, "MW_STUDY_FLOP_BUDGET", 3.0):
            root = adv.advise(st)
            self.assertTrue(root["multiway"] and not root["precomputed"])
            self.assertIsNone(adv.instant(st))   # (its own solve in hand: quick anyway)
            x = dict(st, flop=[X("SB")])
            r = adv.instant(x)
            self.assertEqual((r["actor"], r["approximate"]["source"]), ("BB", "solve"))
            self.assertRegex(r["approximate"]["label"], r"^this street's 3-way solve from SB's decision \(\d+ iterations")
            self.assertTrue(r["multiway"] and r["precomputed"] is None)
            no_values(r)
            live = adv.advise(x)
            self.assertEqual([a["action"] for a in r["actions"]], [a["action"] for a in live["actions"]])
            alike(r, live, 0.1)
            self.assertIsNone(adv.instant(x))
            # the turn: the flop's solve (BB's, the latest), by hand buckets at one bet size
            t = dict(st, board="Kd8c3h9s", flop=[X("SB"), X("BB"), X("BTN")])
            r = adv.instant(t)
            self.assertEqual((r["street"], r["actor"], r["approximate"]["source"]), ("turn", "SB", "prev"))
            self.assertEqual([a["action"] for a in r["actions"]], ["check", "bet 50%"])
            self.assertRegex(r["approximate"]["label"], r"flop solve's view of the turn \(by card class and 12 hand buckets")
            self.assertEqual(r["solver"]["iterations"], live["solver"]["iterations"])
            no_values(r)
            # the turn's later decisions: none off the flop's buckets (far from their own answers there), checked to
            # or facing a bet; once the turn's first decision is solved, its solve's
            self.assertIsNone(adv.instant(dict(t, turn=[X("SB")])))
            self.assertIsNone(adv.instant(dict(t, turn=[X("SB"), {"pos": "BB", "action": "bet", "to": 4.0}])))
            tl = adv.advise(t)
            self.assertEqual(tl["actor"], "SB")
            r = adv.instant(dict(t, turn=[X("SB")]))
            self.assertEqual((r["actor"], r["approximate"]["source"]), ("BB", "solve"))
            self.assertEqual([a["action"] for a in r["actions"]], [a["action"] for a in tl["actions"]])
            no_values(r)
            alike(r, adv.advise(dict(t, turn=[X("SB")])), 0.15)
            # none: a river, a library flop's turn
            self.assertIsNone(adv.instant(dict(t, board="Kd8c3h9s4c", turn=[X("SB"), X("BB"), X("BTN")])))
            lib = dict(st, board="Kd7c2h")
            self.assertTrue(adv.advise(lib)["precomputed"])
            self.assertIsNone(adv.instant(dict(lib, board="Kd7c2h9s", flop=[X("SB"), X("BB"), X("BTN")])))
        del adv
        gc.collect()

    def test_three_way_near_at_once(self):
        """A 3-way flop the 3-way library doesn't store: Study's answer at once (Advisor.instant with "study") read
        off the library's stored flops most like it (MWLibrary.near) at the decisions it stores, first to act and
        checked to: the mix alone (no EVs or equities), both opponents' ranges, the flops read; none on a flop the
        library stores (its answer is at once anyway), after a bet, or 4-way."""
        pre = PRE[:3] + [{"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "call"},
                         {"pos": "BB", "action": "call"}]
        st = {"players": 6, "stack": 100, "preflop": pre, "board": "Kd8c3h", "study": True}
        r = self.adv.instant(st)
        self.assertEqual((r["study"], r["street"], r["actor"], r["position"]), ("node", "flop", "SB", "1 of 3 to act"))
        self.assertTrue(r["multiway"])
        self.assertEqual(r["approximate"]["source"], "near")
        self.assertRegex(r["approximate"]["label"], r"3-way solves of the most similar flops \((\w{6}, ){2}\w{6}\), by card")
        self.assertIsNone(r["precomputed"])
        self.assertEqual([o["seat"] for o in r["opponents"]], ["BB", "BTN"])
        self.check_totals(r)
        self.assertTrue(r["range_ev_bb"] is None and r["range_eq"] is None and all(o["eq"] is None for o in r["opponents"]))
        self.assertTrue(all(a["ev_bb"] is None for a in r["actions"]))
        self.assertTrue(all(c["ev"] is None and c["eq"] is None for c in r["grid"]["cells"].values()))
        self.assertTrue(all(x[3] is None and x[4] is None for rows in r["combos"].values() for x in rows))
        root = dict({o["seat"]: o["combos"] for o in r["opponents"]}, SB=r["range_combos"])
        # checked to: BB, then BTN, the players before narrowed by the mixes read for them
        r = self.adv.instant(dict(st, flop=[{"pos": "SB", "action": "check"}]))
        self.assertEqual((r["actor"], r["approximate"]["source"]), ("BB", "near"))
        r = self.adv.instant(dict(st, flop=[{"pos": "SB", "action": "check"}, {"pos": "BB", "action": "check"}]))
        self.assertEqual(r["actor"], "BTN")
        opp = {o["seat"]: o["combos"] for o in r["opponents"]}
        self.assertLess(opp["SB"], root["SB"])
        self.assertLess(opp["BB"], root["BB"])
        self.check_totals(r)
        # none: a flop the library stores, a line through a bet, 4-way
        self.assertIsNone(self.adv.instant(dict(st, board="Kd7c2h")))
        self.assertIsNone(self.adv.instant(dict(st, flop=[{"pos": "SB", "action": "bet", "to": 2.5}])))
        pre4 = PRE[:2] + [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "call"},
                          {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}]
        self.assertIsNone(self.adv.instant(dict(st, preflop=pre4)))

    def test_three_way(self):
        """3-way pots after the flop: the 3-player solve's answer (the 3-way library's at its flops' checked-to
        nodes), each opponent's range, equity on every street (a share of the pot at showdown against the two other
        ranges, the rest of the board dealt), rare actions marked; 4-way by the 4-player solve (equity on the river
        only), 5+ refused."""
        pre = PRE[:3] + [{"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "call"},
                         {"pos": "BB", "action": "call"}]
        r = self.study(preflop=pre, board="Kd7c2h", breakdown=True)
        self.assertEqual((r["study"], r["street"], r["actor"], r["position"]), ("node", "flop", "SB", "1 of 3 to act"))
        self.assertTrue(r["multiway"] and r["precomputed"])
        self.assertEqual([o["seat"] for o in r["opponents"]], ["BB", "BTN"])
        self.check_totals(r)
        for o in r["opponents"]:
            tot = sum(c["w"] * combos(k) for k, c in o["grid"]["cells"].items())
            self.assertAlmostEqual(tot, o["combos"], delta=0.01 * tot + 1)
        eqs = [r["range_eq"]] + [o["eq"] for o in r["opponents"]]
        self.assertTrue(all(0.2 < e < 0.5 for e in eqs))
        self.assertAlmostEqual(sum(eqs), 1.0, delta=0.05)
        # each combo's (sets and better near the top, 7x and 2x behind Kx) and the classes' and opponents' cells
        eq_of = {x[0]: x[4] for rows in r["combos"].values() for x in rows}
        sets = [v for k, v in eq_of.items() if k[0] == k[2] and k[0] in "K72"]
        self.assertTrue(sets and min(sets) > 0.75, sets)
        self.assertTrue(all(c["eq"] is not None for c in r["grid"]["cells"].values()))
        self.assertTrue(all(c["eq"] is not None for o in r["opponents"] for c in o["grid"]["cells"].values()))
        bd = r["breakdown"]
        self.assertEqual(bd["runouts"], [1176, 1176])
        self.assertAlmostEqual(bd["hero"]["eq"], r["range_eq"], delta=0.002)
        self.assertEqual([round(v["eq"], 3) for v in bd["villains"]], [o["eq"] for o in r["opponents"]])
        self.assertTrue(any(n.startswith("per hand: each hand's mix and EVs on the flop are its own") and "16 a player"
                            in n for n in r["notes"]))
        # the next node: BB's, checked to, from the same solve; its bets are never taken (rough EVs: rough_ev)
        r2 = self.study(preflop=pre, board="Kd7c2h", flop=[{"pos": "SB", "action": "check"}])
        self.assertEqual((r2["actor"], [o["seat"] for o in r2["opponents"]]), ("BB", ["SB", "BTN"]))
        self.assertLess(r2["opponents"][0]["combos"], r["range_combos"])    # SB's checks
        self.assertGreater(r2["opponents"][0]["eq"], 0.2)
        rare = {a["action"]: bool(a.get("rough_ev")) for a in r2["actions"]}
        for a in r2["actions"]:
            if abs(a["freq"] - advisor_mod.RARE_BRANCH) > 0.001:
                self.assertEqual(rare[a["action"]], a["freq"] < advisor_mod.RARE_BRANCH, a)
        self.assertTrue(rare["bet 33%"])
        self.assertNotIn("rough_ev", next(a for a in r["actions"] if a["action"] == "check"))
        cc = [{"pos": p, "action": "check"} for p in ("SB", "BB", "BTN")]
        r = self.study(preflop=pre, board="Kd7c2h5s9c", flop=cc, turn=cc)
        self.assertEqual((r["study"], r["street"], r["actor"]), ("node", "river", "SB"))
        self.check_totals(r)
        eqs = [r["range_eq"]] + [o["eq"] for o in r["opponents"]]
        self.assertTrue(all(0 < e < 1 for e in eqs))
        self.assertAlmostEqual(sum(eqs), 1.0, delta=0.1)
        # (the runout report reads heads-up pots)
        self.assertIn("heads-up", self.adv.study(dict(players=6, stack=100, preflop=pre, board="Kd7c2h", flop=cc,
                                                      runouts=True))["error"])
        pre4 = PRE[:2] + [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "call"},
                          {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}]
        r = self.study(preflop=pre4, board="Kd7c2h")
        self.assertEqual((r["study"], r["street"], r["actor"], r["position"]), ("node", "flop", "SB", "1 of 4 to act"))
        self.assertEqual([o["seat"] for o in r["opponents"]], ["BB", "CO", "BTN"])
        self.check_totals(r)
        self.assertIsNone(r["range_eq"])
        self.assertTrue(all(o["eq"] is None for o in r["opponents"]))
        pre5 = PRE[:1] + [{"pos": "HJ", "action": "raise", "to": 2.5}, {"pos": "CO", "action": "call"},
                          {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "call"}, {"pos": "BB", "action": "call"}]
        self.assertIn("5 players", self.study(preflop=pre5, board="Kd7c2h")["error"])


if __name__ == "__main__":
    unittest.main()
