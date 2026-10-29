"""Per-seat stacks: the replay's all-ins for less, side pots, uncalled bets and incomplete raises, and the
advisor's use of each player's stack (effective stack preflop and heads-up, side pots multiway)."""
import unittest

from gto import preflop as pf
from gto.advisor import Advisor, replay_hand, can_raise, preflop_investments
from gto.solver import Config

MW_TRIALS = 600


def R(pos, action, to=None):
    a = {"pos": pos, "action": action}
    if to is not None:
        a["to" if action == "raise" else "amount"] = to
    return a


def F(*seats):
    return [R(p, "fold") for p in seats]


def T(**stacks):
    return pf.Table(pf.SEATS[6], stacks={p: stacks.get(p, 100.0) for p in pf.SEATS[6]})


class TestReplayStacks(unittest.TestCase):
    def test_short_allin_and_side_pot(self):
        # CO (30bb) jams, BTN and BB call; BB bets 20 on the flop and BTN calls beside the all-in player
        pre = F("UTG", "HJ") + [R("CO", "all-in"), R("BTN", "call")] + F("SB") + [R("BB", "call")]
        rep = replay_hand(T(CO=30), 100, {"preflop": pre, "flop": [R("BB", "bet", 20), R("BTN", "call")], "turn": []}, 3)
        self.assertEqual(rep[0].bets["CO"], 30)
        self.assertEqual(rep[0].allin, {"CO"})
        self.assertAlmostEqual(rep[1].pot_start, 90.5)
        self.assertTrue(rep[1].closed)
        self.assertAlmostEqual(rep[2].pot_start, 130.5)
        self.assertEqual(rep[2].players_start, ["BB", "CO", "BTN"])
        self.assertEqual(rep[2].to_act, "BB")

    def test_uncalled_bet_comes_back_and_the_action_ends(self):
        # BTN (100) jams over the 20bb big blind, who calls: 80 come back, nobody is left to bet on the flop
        pre = F("UTG", "HJ", "CO") + [R("BTN", "all-in")] + F("SB") + [R("BB", "call")]
        rep = replay_hand(T(BB=20), 100, {"preflop": pre, "flop": []}, 2)
        self.assertEqual(rep[0].returned, {"BTN": 80})
        self.assertEqual(rep[0].allin, {"BB"})
        self.assertAlmostEqual(rep[1].pot_start, 40.5)
        self.assertTrue(rep[1].closed)
        self.assertIsNone(rep[1].to_act)
        # a small blind all-in for less than the big blind: the big blind has nothing to call
        hu = pf.Table(pf.SEATS[2], stacks={"SB": 0.75, "BB": 100})
        st = replay_hand(hu, 100, {"preflop": [R("SB", "call")]}, 1)[0]
        self.assertTrue(st.closed)
        self.assertEqual(st.returned, {"BB": 0.25})

    def test_blind_that_takes_the_whole_stack_is_all_in(self):
        # a 0.5bb small blind is all-in by posting: it has no decision, and its 'call' (as a history may write it)
        # is skipped
        t3 = pf.Table(pf.SEATS[3], stacks={"BTN": 100, "SB": 0.5, "BB": 100})
        st = replay_hand(t3, 100, {"preflop": [R("BTN", "call")]}, 1)[0]
        self.assertIn("SB", st.allin)
        self.assertEqual(st.to_act, "BB")
        st = replay_hand(t3, 100, {"preflop": [R("BTN", "call"), R("SB", "call")]}, 1)[0]
        self.assertEqual(st.to_act, "BB")

    def test_incomplete_raise_does_not_reopen(self):
        pre = F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "call")] + F("SB") + [R("BB", "call")]
        flop = [R("BB", "bet", 4), R("CO", "raise", 8), R("BTN", "all-in")]   # 12bb BTN: all-in to 9.5, 1.5 more
        st = replay_hand(T(BTN=12), 100, {"preflop": pre, "flop": flop}, 2)[1]
        self.assertAlmostEqual(st.cur_bet, 9.5)
        self.assertAlmostEqual(st.min_raise, 4)
        self.assertEqual(st.to_act, "BB")
        self.assertTrue(can_raise(st, "BB"))       # BB faced CO's full raise
        st = replay_hand(T(BTN=12), 100, {"preflop": pre, "flop": flop + [R("BB", "call")]}, 2)[1]
        self.assertEqual(st.to_act, "CO")
        self.assertFalse(can_raise(st, "CO"))      # CO made the last full raise: call or fold
        # a full raise reopens it (BTN 18.5: all-in to 16, 8 more)
        st = replay_hand(T(BTN=18.5), 100, {"preflop": pre, "flop": flop + [R("BB", "call")]}, 2)[1]
        self.assertTrue(can_raise(st, "CO"))

    def test_preflop_investments_per_seat(self):
        inv, active = preflop_investments([R("UTG", "all-in"), R("HJ", "call")] + F("CO", "BTN", "SB", "BB"),
                                          {p: 100.0 for p in pf.POSITIONS} | {"UTG": 15.0})
        self.assertEqual((inv["UTG"], inv["HJ"]), (15.0, 15.0))
        self.assertEqual(active, ["UTG", "HJ"])


class TestAdvisorStacks(unittest.TestCase):
    def setUp(self):
        self.adv = Advisor(Config(depth=1), time_budget=1.0, prior_budget=0.5)
        self.adv.mw_trials = MW_TRIALS
        self.adv.mw_solve_streets = ()   # the betting rules (3-way pots are solved by default)

    def test_blind_all_in_seat_answers(self):
        # (was a ZeroDivisionError in the preflop pricing: the seat's raise risked nothing) preflop, and every later
        # street, whose ranges replay that seat's line
        for sb, extra in ((0.5, {}), (0.3, {}), (1.0, {"ante": 0.5})):
            base = {"players": 3, "stack": 100, "stacks": {"SB": sb}, "hero": "BB", "hole": "KsQd", **extra}
            out = self.adv.advise(dict(base, board="", preflop=[R("BTN", "call"), R("SB", "call")]))
            self.assertTrue(out.get("actions"), (sb, out))
            out = self.adv.advise(dict(base, board="3h2c5s", preflop=[R("BTN", "call"), R("SB", "call"), R("BB", "check")]))
            self.assertTrue(out.get("actions"), (sb, out))

    def test_stacks_input(self):
        base = {"players": 6, "hero": "BTN", "hole": "AhKd", "board": "", "preflop": F("UTG", "HJ", "CO")}
        self.assertIn("not a seat", self.adv.advise(dict(base, stacks={"XX": 50}))["error"])
        self.assertIn("positive", self.adv.advise(dict(base, stacks={"BB": -5}))["error"])
        # the old meaning of "stacks" (one effective stack) still reads the same where they're equal
        a = self.adv.advise(dict(base, stacks={"BTN": 100, "BB": 100}))
        b = self.adv.advise(dict(base, stack=100))
        self.assertEqual(a["actions"], b["actions"])

    def test_preflop_effective_stack(self):
        # BTN with 100bb against two 12bb blinds plays a 12bb game: push/fold, as with 12bb stacks all round
        base = {"players": 6, "hero": "BTN", "hole": "Ah5d", "board": "", "preflop": F("UTG", "HJ", "CO")}
        short = self.adv.advise(dict(base, stack=100, stacks={"SB": 12, "BB": 12}))
        equal = self.adv.advise(dict(base, stack=12))
        self.assertEqual(short["actions"], equal["actions"])
        self.assertEqual(short["actions"][0]["action"], "all-in")
        self.assertAlmostEqual(short["actions"][0]["amount_bb"], 12)
        # one deep blind behind keeps the 100bb game: a normal open
        deep = self.adv.advise(dict(base, stack=100, stacks={"BB": 12}))
        self.assertEqual(deep["actions"][0]["action"], "raise")
        self.assertAlmostEqual(deep["actions"][0]["amount_bb"], 2.5)
        # facing a 15bb jam: the call is 14bb (equity against a 15bb push/fold range)
        pre = F("UTG") + [R("HJ", "all-in")] + F("CO", "BTN", "SB")
        r = self.adv.advise({"players": 6, "hero": "BB", "hole": "AhJd", "board": "", "stack": 100, "stacks": {"HJ": 15},
                             "preflop": pre})
        call = next(a for a in r["actions"] if a["action"] == "call")
        self.assertAlmostEqual(call["amount_bb"], 14)
        self.assertTrue(any("15bb jam range: push/fold" in n for n in r["notes"]), r["notes"])

    def test_postflop_heads_up_effective_stack(self):
        # 30bb in the big blind against hero's 100bb button: the solve plays 27.5bb behind, as with 30bb stacks
        pre = F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB") + [R("BB", "call")]
        base = {"players": 6, "hero": "BTN", "hole": "AhKd", "board": "Kc7d2s", "preflop": pre, "flop": [R("BB", "check")],
                "budget": 1.0, "prior_budget": 0.5}
        r = self.adv.advise(dict(base, stack=100, stacks={"BB": 30}))
        self.assertNotIn("error", r)
        self.assertAlmostEqual(r["stack_bb"], 27.5)
        allin = [a for a in r["actions"] if a["action"] == "all-in"]
        self.assertTrue(not allin or allin[0]["amount_bb"] <= 27.5 + 1e-9, r["actions"])
        e = self.adv.advise(dict(base, stack=30))
        self.assertEqual(r["solver"]["tree_actions"], e["solver"]["tree_actions"])
        self.assertAlmostEqual(r["stack_bb"], e["stack_bb"])

    def test_multiway_side_pot(self):
        # CO is all-in for 10bb; BB (hero) and BTN bet 20 and 30 more beside it. On the river CO holds AA, which
        # beats hero's KK, and BTN a missed JT: hero loses the 30.5bb main pot and wins the 100bb side pot
        pre = F("UTG", "HJ") + [R("CO", "all-in"), R("BTN", "call")] + F("SB") + [R("BB", "call")]
        st = {"players": 6, "hero": "BB", "hole": "KhKd", "board": "Qs9h5d3c2s", "stack": 100, "stacks": {"CO": 10},
              "preflop": pre, "flop": [R("BB", "bet", 20), R("BTN", "call")], "turn": [R("BB", "bet", 30), R("BTN", "call")],
              "river": [], "ranges": {"CO": "AA", "BTN": "JTo", "BB": "KK"}}
        r = self.adv.advise(st)
        self.assertNotIn("error", r)
        self.assertTrue(r["multiway"])
        self.assertTrue(any("all-in for less" in n for n in r["notes"]), r["notes"])
        ev = {a["action"]: a["ev_bb"] for a in r["actions"]}
        # checking wins the side pot (and whatever BTN bluffs), never the main pot
        self.assertGreater(ev["check"], 90)
        self.assertLess(ev["check"], 100 + 30.5)
        # a bet faces BTN alone: betting stays available beside the all-in player (here both sizes are more than the
        # 40bb hero has left, so the bet is named all-in, 40 into BTN's 40, as the page takes it)
        bets = [a for a in r["actions"] if a["action"].startswith("bet") or a["action"] == "all-in"]
        self.assertTrue(bets, ev)
        self.assertTrue(all(a["amount_bb"] <= 40 + 1e-9 for a in bets), r["actions"])

    def test_three_way_solve_plays_each_stack_not_the_shortest(self):
        # a 12bb CO in a 3-way pot: each player bets its own stack as far as another can match it, the CO's 9.5bb
        # behind contesting the main pot (it was everyone at the CO's 9.5bb, the deep players' "all-in" a 9.5bb bet):
        # the BTN's menu is the deep players' (half the pot; no all-in, 12 pots behind); the short CO's all-in is its
        # whole stack
        adv = Advisor(Config(depth=1), time_budget=1e9)
        adv.mw_max_iters = 20
        pre = F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "call")] + F("SB") + [R("BB", "call")]
        st = {"players": 6, "hero": "BTN", "hole": "AsKd", "board": "Ks7d2cTh5s", "stack": 100, "stacks": {"CO": 12},
              "preflop": pre, "flop": [R("BB", "check"), R("CO", "check"), R("BTN", "check")],
              "turn": [R("BB", "check"), R("CO", "check"), R("BTN", "check")], "river": [R("BB", "check"), R("CO", "check")]}
        r = adv.advise(st)
        self.assertTrue(r.get("solved"), r.get("notes"))
        self.assertEqual([a["action"] for a in r["actions"]], ["check", "bet 50%"])
        self.assertEqual(r["stack_bb"], 97.5)
        # practice: the move is the solve's (a bet of half the pot, 4bb)
        for s in range(8):
            m = adv.play(st, "BTN", "AsKd", seed=s)
            self.assertIn(m["action"], ("check", "bet"))
            if m["action"] == "bet":
                self.assertAlmostEqual(m["to"], 4.0)
        # a bet of the CO's stack (off the solve's menu) is faced as made
        st_bet = dict(st, river=st["river"] + [R("BTN", "bet", 9.5)])
        r2 = adv.advise(dict(st_bet, hero="BB", hole="QcJc"))
        self.assertNotIn("error", r2)
        self.assertAlmostEqual(r2["to_call_bb"], 9.5)
        # a practice opponent's move is read off the BB's own solve of the river
        bb = dict(st, hero="BB", hole="QcJc", river=[])
        self.assertTrue(adv.advise(bb).get("solved"))
        n = len(adv._mw_cache)
        m = adv.play(dict(bb, river=st["river"]), "BTN", "AsKd", seed=1)
        self.assertEqual(len(adv._mw_cache), n)
        self.assertEqual([a["action"] for a in m["actions"]], ["check", "bet 50%"])
        r3 = adv.advise(dict(st, hero="CO", hole="QdQc", river=[R("BB", "check")]))
        self.assertTrue(r3.get("solved"))
        self.assertIn("all-in", [a["action"] for a in r3["actions"]])
        self.assertEqual(r3["stack_bb"], 9.5)

    def test_multiway_incomplete_raise_no_raise_option(self):
        # flop: hero (BB) bets 4, CO raises to 8, the 12bb BTN jams to 9.5: hero may raise (CO's raise was full)...
        pre = F("UTG", "HJ") + [R("CO", "raise", 2.5), R("BTN", "call")] + F("SB") + [R("BB", "call")]
        base = {"players": 6, "hero": "BB", "hole": "9h9d", "board": "Kc7d2s", "stack": 100, "stacks": {"BTN": 12},
                "preflop": pre}
        r = self.adv.advise(dict(base, flop=[R("BB", "bet", 4), R("CO", "raise", 8), R("BTN", "all-in")]))
        self.assertTrue(any(a["action"].startswith("raise") for a in r["actions"]), r["actions"])
        # ...but CO, back after hero's call, can only call or fold
        r = self.adv.advise(dict(base, hero="CO", hole="AhAd",
                                 flop=[R("BB", "bet", 4), R("CO", "raise", 8), R("BTN", "all-in"), R("BB", "call")]))
        self.assertNotIn("error", r)
        self.assertEqual(sorted(a["action"] for a in r["actions"]), ["call", "fold"])


if __name__ == "__main__":
    unittest.main()
