"""The web page's game states against the Python advisor.

tests/data/page_states.json holds random hands played through the page's own engine (written by
tests/js/solver_state.test.js: `UPDATE_FIXTURES=1 node --test 'tests/js/*.test.js'`), each spot with the state
the page would send to the solver and the page's view of it: whose turn, the pot, the bet to call, who is
still in. The advisor must replay every state to the same spot, and accept the ones where it's hero's turn.
"""
import json
import os
import unittest

from gto import preflop as pf
from gto.advisor import Advisor, replay_hand, _norm_actions, STREETS
from gto.solver import Config

HERE = os.path.dirname(__file__)
N_STREETS = {0: 1, 3: 2, 4: 3, 5: 4}


def load_spots():
    with open(os.path.join(HERE, "data", "page_states.json")) as f:
        return json.load(f)


class TestPageStates(unittest.TestCase):
    def test_replay_matches_the_page(self):
        spots = load_spots()
        self.assertGreater(len(spots), 100)
        for i, sp in enumerate(spots):
            st, page = sp["state"], sp["page"]
            with self.subTest(spot=i, hand=sp["hand"]):
                n = st["players"]
                table = pf.Table(pf.SEATS[n], st.get("ante", 0.0), st.get("straddle", False))
                streets = {s: _norm_actions(st[s], 1.0) for s in STREETS}
                last = replay_hand(table, st["stack"], streets, N_STREETS[len(st["board"]) // 2])[-1]
                self.assertEqual(last.name, page["street"])
                self.assertFalse(last.closed)
                self.assertEqual(last.to_act, page["to_act"])
                self.assertAlmostEqual(last.pot_start + sum(last.bets.values()), page["pot"], places=6)
                self.assertAlmostEqual(last.cur_bet, page["cur_bet"], places=6)
                self.assertEqual(sorted(last.active), sorted(page["active"]))
                self.assertEqual(sorted(last.allin), sorted(page["allin"]))

    def test_advisor_accepts_hero_spots(self):
        adv = Advisor(Config(depth=1), time_budget=0.5, prior_budget=0.3)
        # a quick sample (big-table preflop and multiway spots take seconds each): short-handed preflop
        # spots, the first heads-up spot on each later street and the smallest multiway one
        spots = [sp for sp in load_spots() if sp["page"]["to_act"] == sp["state"]["hero"]]
        pre = [sp for sp in spots if not sp["state"]["board"] and sp["state"]["players"] <= 6][:10]
        hu = {}
        for sp in spots:
            if sp["state"]["board"] and len(sp["page"]["active"]) == 2:
                hu.setdefault(sp["page"]["street"], sp)
        multi = sorted((sp for sp in spots if sp["state"]["board"] and len(sp["page"]["active"]) > 2),
                       key=lambda sp: (len(sp["page"]["active"]), sp["state"]["players"]))[:1]
        self.assertEqual(len(pre), 10)
        self.assertEqual(set(hu), {"flop", "turn", "river"})
        self.assertEqual(len(multi), 1)
        for sp in pre + list(hu.values()) + multi:
            st = dict(sp["state"], budget=0.5, prior_budget=0.3)
            with self.subTest(hand=sp["hand"], street=sp["page"]["street"]):
                r = adv.advise(st)
                self.assertNotIn("error", r, r.get("error"))
                self.assertEqual(r["street"], sp["page"]["street"])
                self.assertAlmostEqual(sum(a["freq"] for a in r["actions"]), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
