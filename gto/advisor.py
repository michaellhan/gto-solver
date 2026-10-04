"""Turn a game state into a recommendation.

Game state (all money in big blinds unless `bb` is given, then chips / bb):
{
  "players": 6,                                 # 2..9 (seat names from preflop.SEATS)
  "hero": "BTN", "hole": "AhKd", "board": "Ks7d2c",
  "stack": 100,                                 # effective stack (or "stacks": {pos: chips})
  "preflop": [{"pos": "CO", "action": "raise", "to": 2.5}, {"pos": "BTN", "action": "call"}, ...],
  "flop": [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 2.0}],
  "turn": [], "river": []
}
Actions: fold, check, call, bet (amount), raise (to), all-in.
"""
from __future__ import annotations
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .cards import cards as parse_cards, COMBO_INDEX, hand_class, ALL_COMBOS, cards_str
from .ranges import N_COMBOS, range_to_text
from . import preflop as pf
from .solver import solve, Config, Solution, Decision, Terminal, Chance, match_action, expand, MEMBER_FULL, _inv_of

DEFAULT_OPEN = {"SB": 3.0}
STREET_OF = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
STREETS = ["preflop", "flop", "turn", "river"]


def _norm_actions(acts: List[dict], bb: float) -> List[dict]:
    out = []
    for a in acts or []:
        d = dict(a)
        d["pos"] = str(d["pos"]).upper()
        act = str(d["action"]).lower().replace("allin", "all-in").replace("shove", "all-in").replace("jam", "all-in")
        act = {"x": "check", "f": "fold", "c": "call", "r": "raise", "b": "bet", "ai": "all-in"}.get(act, act)
        d["action"] = act
        for k in ("to", "amount"):
            if k in d and d[k] is not None:
                d[k] = float(d[k]) / bb
        out.append(d)
    return out


# ----------------------------------------------------------------------------- hand replay
@dataclass
class StreetState:
    name: str
    order: List[str]                 # acting order this street
    players_start: List[str]         # not folded at the start of the street, in acting order
    pot_start: float                 # chips in the middle at the start of the street
    inv_start: Dict[str, float]      # total invested per player at the start
    bets: Dict[str, float] = field(default_factory=dict)   # committed this street
    cur_bet: float = 0.0
    n_raises: int = 0
    acted: set = field(default_factory=set)
    active: set = field(default_factory=set)               # not folded after the actions so far
    allin: set = field(default_factory=set)
    to_act: Optional[str] = None
    closed: bool = False
    actions: List[dict] = field(default_factory=list)


def _showdown_share(st, idx, hands) -> float:
    """Hero's (combo idx) share of the pot at showdown against these opponent combos, strengths st."""
    best = max(st[j] for j in hands)
    if st[idx] != best:
        return float(st[idx] > best)
    return 1.0 / (1 + sum(st[j] == best for j in hands))


def replay_hand(positions: List[str], stack: float, streets: Dict[str, List[dict]], n_streets: int) -> List[StreetState]:
    """Replay preflop..current street. Sizes: bet=amount added this street, raise=to."""
    inv = {p: 0.0 for p in positions}
    active = set(positions)
    allin: set = set()
    out: List[StreetState] = []
    post = pf.postflop_order(positions)
    for si in range(n_streets):
        name = STREETS[si]
        pre = name == "preflop"
        order = list(positions) if pre else post
        st = StreetState(name, order, [p for p in order if p in active], sum(inv.values()), dict(inv))
        st.bets = {p: 0.0 for p in positions}
        if pre:
            if "SB" in positions:
                st.bets["SB"] = min(0.5, stack)
            st.bets["BB"] = min(1.0, stack)
            st.cur_bet = 1.0
        st.active, st.allin = set(active), set(allin)
        last = None
        for a in streets.get(name, []):
            p, act = a["pos"], a["action"]
            if p not in st.active or p in st.allin:
                continue
            remaining = stack - inv[p] - st.bets[p]
            if act == "fold":
                st.active.discard(p)
            elif act in ("check", "call"):
                st.bets[p] = min(st.cur_bet, st.bets[p] + remaining)
                if st.bets[p] + inv[p] >= stack - 1e-9:
                    st.allin.add(p)
                st.acted.add(p)
            elif act in ("bet", "raise", "all-in"):
                if act == "all-in":
                    to = st.bets[p] + remaining
                elif a.get("to") is not None:
                    to = a["to"]
                elif a.get("amount") is not None:
                    to = a["amount"]
                else:
                    to = _default_size(st, p, pre)
                to = min(to, st.bets[p] + remaining)
                st.bets[p] = to
                if to > st.cur_bet + 1e-9:
                    st.cur_bet = to
                    st.n_raises += 1
                    st.acted = {p}
                else:
                    st.acted.add(p)
                if st.bets[p] + inv[p] >= stack - 1e-9:
                    st.allin.add(p)
            st.actions.append(a)
            last = p
        actors = [p for p in st.active if p not in st.allin]
        st.closed = len(st.active) <= 1 or all(p in st.acted for p in actors)
        if not st.closed:
            start = (order.index(last) + 1) % len(order) if last else 0
            for k in range(len(order)):
                q = order[(start + k) % len(order)]
                if q in st.active and q not in st.allin and q not in st.acted:
                    st.to_act = q
                    break
        out.append(st)
        for p in positions:
            inv[p] += st.bets[p]
        active, allin = set(st.active), set(st.allin)
        if not st.closed:
            break
    return out


def _default_size(st: StreetState, p: str, pre: bool) -> float:
    if pre:
        if st.n_raises == 0:
            return DEFAULT_OPEN.get(p, 2.5)
        if st.n_raises == 1:
            return st.cur_bet * (4.0 if p in ("SB", "BB") else 3.0)
        return st.cur_bet * 2.3
    pot = st.pot_start + sum(st.bets.values())
    if st.cur_bet > st.bets[p]:
        return st.cur_bet + 0.75 * (pot + st.cur_bet - st.bets[p])
    return 0.66 * pot


def preflop_investments(actions: List[dict], stacks: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """Kept for compatibility: chips each player put in preflop and who is still in."""
    stack = min(stacks.values()) if stacks else 100.0
    st = replay_hand(pf.POSITIONS, stack, {"preflop": actions}, 1)[0]
    inv = {p: st.inv_start[p] + st.bets[p] for p in pf.POSITIONS}
    return inv, [p for p in pf.POSITIONS if p in st.active]


def _range_key(v: np.ndarray) -> str:
    return hashlib.md5(np.round(v, 6).tobytes()).hexdigest()[:12]


# ----------------------------------------------------------------------------- advisor
class Advisor:
    def __init__(self, cfg: Optional[Config] = None, time_budget: float = 4.0, prior_budget: float = 2.0,
                 prior_smoothing: float = 0.15):
        self.cfg = cfg or Config()
        self.time_budget = time_budget
        self.prior_budget = prior_budget
        self.prior_smoothing = prior_smoothing
        self._cache: Dict[tuple, Solution] = {}
        self.should_stop = None   # callable; when it returns true the running solve raises solver.Cancelled
        self.warm_from: Optional["Advisor"] = None   # another advisor (e.g. a faster config) whose solves seed ours
        self._by_spot: Dict[tuple, Solution] = {}    # latest solve per spot, whatever the ranges
        self.rng = random.Random()

    # ------------------------------------------------------------ public
    def advise(self, state: dict) -> dict:
        with np.errstate(all="ignore"):
            return self._advise(state)

    def _advise(self, state: dict) -> dict:
        t0 = time.time()
        bb = float(state.get("bb", 1.0))
        n = int(state.get("players", 6))
        if n not in pf.SEATS:
            return {"error": f"players must be 2..9, got {n}"}
        positions = pf.SEATS[n]
        hero = str(state["hero"]).upper()
        if hero not in positions:
            return {"error": f"hero seat {hero} is not one of {positions}"}
        hole = tuple(parse_cards(str(state["hole"])))
        board = parse_cards(str(state.get("board") or ""))
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
            return {"error": "need two hole cards and 0, 3, 4 or 5 board cards"}
        if state.get("stacks"):
            stack = min(float(v) for v in state["stacks"].values()) / bb
        else:
            stack = float(state.get("stack", 100.0)) / bb
        streets = {s: _norm_actions(state.get(s), bb) for s in STREETS}
        if not board:
            return self._preflop(hero, hole, streets["preflop"], positions, stack, t0)
        saved = self.time_budget, self.prior_budget
        if state.get("budget"):
            self.time_budget = max(0.5, min(120.0, float(state["budget"])))
        if state.get("prior_budget"):
            self.prior_budget = max(0.3, min(30.0, float(state["prior_budget"])))
        try:
            return self._postflop(hero, hole, board, stack, streets, positions, t0)
        finally:
            self.time_budget, self.prior_budget = saved

    # ------------------------------------------------------------ preflop
    def _preflop(self, hero, hole, preflop, positions, stack, t0):
        st = replay_hand(positions, stack, {"preflop": preflop}, 1)[0]
        if st.closed:
            return {"error": "preflop betting is already closed (deal the flop)"}
        if st.to_act != hero:
            return {"error": f"it is {st.to_act}'s turn, not {hero}'s"}
        strat = pf.hand_strategy(hero, hole, preflop, positions)
        remaining = stack - st.inv_start[hero] - st.bets[hero]
        to_call = min(st.cur_bet - st.bets[hero], remaining)
        size = min(_default_size(st, hero, True), st.bets[hero] + remaining)
        size = round(size * 2) / 2
        acts = []
        for k, v in strat.items():
            a = {"action": k, "freq": round(v, 3)}
            if k == "raise":
                if size >= st.bets[hero] + remaining - 1e-9:
                    a["action"] = "all-in"
                a["amount_bb"] = round(size, 2)
            elif k == "call":
                a["amount_bb"] = round(to_call, 2)
            acts.append(a)
        best = max(acts, key=lambda a: a["freq"])["action"]
        full = pf.strategy_for(hero, preflop, positions)
        names = list(full.keys())
        arriving = pf.replay_ranges(preflop, positions).get(hero, np.ones(N_COMBOS))
        return {
            "street": "preflop", "hero": hero, "hand": hand_class(hole), "players": len(positions),
            "actions": acts, "recommended": best, "sampled": self._sample(acts),
            "pot_bb": round(st.pot_start + sum(st.bets.values()), 2), "to_call_bb": round(to_call, 2),
            "grid": strategy_grid([full[k] for k in names], names, arriving),
            "notes": ["preflop: chart approximation"], "seconds": round(time.time() - t0, 3),
        }

    # ------------------------------------------------------------ postflop
    def _postflop(self, hero, hole, board, stack, streets, positions, t0):
        notes: List[str] = []
        n_streets = {3: 2, 4: 3, 5: 4}[len(board)]
        rep = replay_hand(positions, stack, streets, n_streets)
        if len(rep) < n_streets:
            return {"error": "an earlier street's betting is not finished", "notes": notes}
        cur = rep[-1]
        if hero not in cur.active:
            return {"error": "hero has folded"}
        if cur.closed:
            return {"error": "this street's betting is already closed (deal the next card)"}
        if cur.to_act != hero:
            return {"error": f"it is {cur.to_act}'s turn, not {hero}'s"}
        ranges = pf.replay_ranges(streets["preflop"], positions)
        remaining = [p for p in cur.order if p in cur.active]
        pot_now = cur.pot_start + sum(cur.bets.values())
        to_call = cur.cur_bet - cur.bets[hero]
        hero_remaining = stack - cur.inv_start[hero] - cur.bets[hero]

        live_opps = [p for p in remaining if p != hero]
        if len(live_opps) != 1 or cur.allin & set(remaining) - {hero} and len(remaining) > 2:
            return self._multiway(hero, hole, board, ranges, remaining, cur, stack, t0, notes)
        villain = live_opps[0]
        post = cur.order
        oop, ip = sorted([hero, villain], key=post.index)
        hero_p = 0 if hero == oop else 1
        r0, r1 = ranges[oop].copy(), ranges[ip].copy()
        for c in hole:  # villain can't hold hero's cards
            if hero_p == 0:
                r1 = r1 * (1.0 - MEMBER_FULL[c])
            else:
                r0 = r0 * (1.0 - MEMBER_FULL[c])

        hu_from_start = cur.players_start == remaining
        if hu_from_start:
            # narrow through earlier streets that were also heads-up between these two
            for st in rep[1:-1]:
                sub_board = board[:{"flop": 3, "turn": 4, "river": 5}[st.name]]
                if st.players_start != remaining:
                    notes.append(f"{st.name}: was multiway, ranges not narrowed")
                    continue
                eff = min(stack - st.inv_start[oop], stack - st.inv_start[ip])
                cfg = self._cfg_with_real_sizes(st, oop, ip, eff)
                sol = self._solve_cached(sub_board, r0, r1, st.pot_start, eff, self.prior_budget, cfg=cfg)
                node, c0, c1 = self._walk(sol, st.actions, oop, ip, st.pot_start, eff, notes, st.name, self.prior_smoothing)
                r0, r1 = expand(sol.ctx, c0), expand(sol.ctx, c1)
            eff = min(stack - cur.inv_start[oop], stack - cur.inv_start[ip])
            cfg = self._cfg_with_real_sizes(cur, oop, ip, eff)
            sol = self._solve_cached(board, r0, r1, cur.pot_start, eff, self.time_budget, cfg=cfg)
            node, n_r0, n_r1 = self._walk(sol, cur.actions, oop, ip, cur.pot_start, eff, notes, cur.name, 0.0)
        else:
            # pot became heads-up during this street: solve from the current node
            eff = min(stack - cur.inv_start[oop], stack - cur.inv_start[ip])
            sol = self._solve_cached(board, r0, r1, cur.pot_start, eff, self.time_budget,
                                     bets=(cur.bets[oop], cur.bets[ip]), to_act=hero_p, n_raises=cur.n_raises)
            node, n_r0, n_r1 = sol.root, sol.r0.copy(), sol.r1.copy()
            notes.append(f"{cur.name}: pot became heads-up this street; {villain}'s range is not narrowed by this street's actions")
        if not isinstance(node, Decision):
            return {"error": "no decision pending at this point in the tree", "notes": notes}
        if node.player != hero_p:
            notes.append("warning: tree expected the other player to act; reporting that node")
        evs = sol.solver.node_action_evs(node, n_r0, n_r1)
        strat = node.avg_strategy0()
        idx = COMBO_INDEX[(min(hole), max(hole))]
        pos_in_active = np.nonzero(sol.ctx.active == idx)[0]
        ci = int(pos_in_active[0]) if len(pos_in_active) else None
        acts_out = []
        for a, name in enumerate(node.names):
            f = float(strat[a, ci]) if ci is not None else 1.0 / len(node.names)
            ev = float(evs[a, ci]) if ci is not None else 0.0
            acts_out.append({"action": name, "freq": round(f, 3), "ev_bb": round(ev, 2),
                             "amount_bb": self._amount(name, node, hero_p)})
        if ci is None:
            notes.append("hero's hand is outside the chart range for this line; frequencies are uniform")
        best = max(acts_out, key=lambda a: a["freq"])["action"]
        hero_r = expand(sol.ctx, n_r0 if hero_p == 0 else n_r1)
        vill_r = expand(sol.ctx, n_r1 if hero_p == 0 else n_r0)
        w = hero_r / max(hero_r.sum(), 1e-12)
        sf = np.zeros((len(node.names), N_COMBOS)); sf[:, sol.ctx.active] = strat
        range_freq = {name: round(float((sf[a] * w).sum()), 3) for a, name in enumerate(node.names)}
        return {
            "street": cur.name, "hero": hero, "villain": villain, "position": "OOP" if hero_p == 0 else "IP",
            "players": len(positions), "hand": hand_class(hole), "board": cards_str(board),
            "pot_bb": round(node.pot_now, 2), "to_call_bb": round(min(to_call, hero_remaining), 2),
            "stack_bb": round(min(eff - max(node.bets), hero_remaining), 2),
            "actions": acts_out, "recommended": best, "sampled": self._sample(acts_out),
            "grid": strategy_grid([sf[a] for a in range(len(node.names))], list(node.names), hero_r),
            "range_action_freq": range_freq,
            "hero_range_combos": round(float(hero_r.sum()), 1),
            "villain_range_combos": round(float(vill_r.sum()), 1),
            "villain_range": range_to_text(vill_r, 0.05)[:400],
            "solver": {"iterations": sol.iters, "solve_seconds": round(sol.seconds, 2), "nodes": sol.n_nodes,
                       "active_combos": sol.ctx.n, "tree_actions": node.names,
                       # how far this street's solve is from equilibrium, % of the street's starting pot
                       "exploitability_pct": round(max(0.0, sol.exploitability()), 2)},
            "notes": notes, "seconds": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------ multiway
    def _multiway(self, hero, hole, board, ranges, remaining, cur: StreetState, stack, t0, notes):
        """3+ players still in: Monte Carlo vs every live range, not a solve. Frequencies are 0/1 (the
        highest EV action) and EVs are relative to folding, assuming the hand is checked down after
        this action. Against a bet or raise each opponent continues with the top of their range, sized
        so the field together folds as often as a pure bluff needs to break even."""
        from .evaluator import eval_board
        rng = random.Random(1)
        opps = [p for p in remaining if p != hero]
        dead = set(board) | set(hole)
        avail = [c for c in range(52) if c not in dead]
        idx = COMBO_INDEX[(min(hole), max(hole))]
        opp_ranges = []
        for p in opps:
            r = ranges.get(p, np.ones(N_COMBOS)).copy()
            for c in dead:
                r *= 1.0 - MEMBER_FULL[c]
            opp_ranges.append(r if r.sum() > 0 else np.ones(N_COMBOS))
        pot = cur.pot_start + sum(cur.bets.values())
        to_call = cur.cur_bet - cur.bets[hero]
        hero_rem = stack - cur.inv_start[hero] - cur.bets[hero]
        call_amt = min(to_call, hero_rem)
        n_opp = len(opps)
        # Our aggressive option: raise to 75% pot after calling, or bet 66% pot, to the nearest 0.5bb.
        if to_call > 1e-9:
            agg_to = min(round((cur.cur_bet + 0.75 * (pot + call_amt)) * 2) / 2, cur.bets[hero] + hero_rem)
            agg_name = f"raise to {agg_to:g}"
        else:
            agg_to = min(round(0.66 * pot * 2) / 2, hero_rem)
            agg_name = "bet 66%"
        hero_add = agg_to - cur.bets[hero]
        opp_add = [max(0.0, min(agg_to, stack - cur.inv_start[p]) - cur.bets[p]) for p in opps]
        # Who continues vs it: rank each range by average showdown strength over runouts, then keep
        # the top share c per opponent, with (1 - c) ** n_opp = hero_add / (pot + hero_add).
        alpha = hero_add / (pot + hero_add) if hero_add > 1e-9 else 1.0
        keep = 1.0 - alpha ** (1.0 / n_opp)
        tot = np.zeros(N_COMBOS); cnt = np.zeros(N_COMBOS)
        for _ in range(1 if len(board) == 5 else 40):
            st = eval_board(list(board) + rng.sample(avail, 5 - len(board)))
            ok = st >= 0
            tot[ok] += st[ok]; cnt[ok] += 1
        order = np.argsort(-np.where(cnt > 0, tot / np.maximum(cnt, 1), -1.0))
        conts = []
        for r in opp_ranges:
            cum = np.cumsum(r[order])
            cont = np.zeros(N_COMBOS, dtype=bool)
            cont[order[:int(np.searchsorted(cum, keep * cum[-1] - 1e-9)) + 1]] = True
            conts.append(cont)
        share_all = ev_agg = trials = 0.0
        for _ in range(400):
            run = list(board) + rng.sample(avail, 5 - len(board))
            st = eval_board(run)
            used = set(run) | set(hole)
            hands = []
            for r in opp_ranges:
                w = r.copy()
                for c in used:
                    w *= 1.0 - MEMBER_FULL[c]
                if w.sum() <= 0:
                    break
                j = rng.choices(range(N_COMBOS), weights=w, k=1)[0]
                used |= set(ALL_COMBOS[j]); hands.append(j)
            else:
                trials += 1
                share_all += _showdown_share(st, idx, hands)
                callers = [i for i, j in enumerate(hands) if conts[i][j]]
                if not callers:
                    ev_agg += pot
                else:
                    final = pot + hero_add + sum(opp_add[i] for i in callers)
                    ev_agg += _showdown_share(st, idx, [hands[i] for i in callers]) * final - hero_add
        trials = max(trials, 1)
        eq = float(share_all / trials)
        acts = []
        if to_call > 1e-9:
            need = call_amt / (pot + call_amt)
            acts.append({"action": "fold", "freq": 0.0, "ev_bb": 0.0})
            acts.append({"action": "call", "freq": 0.0, "ev_bb": round(eq * (pot + call_amt) - call_amt, 2), "amount_bb": round(call_amt, 2)})
            notes.append(f"pot odds need {need*100:.0f}% equity, you have {eq*100:.0f}%")
        else:
            acts.append({"action": "check", "freq": 0.0, "ev_bb": round(eq * pot, 2)})
        if hero_add > 1e-9:
            acts.append({"action": agg_name, "freq": 0.0, "ev_bb": round(float(ev_agg / trials), 2), "amount_bb": round(agg_to, 2)})
            notes.append(f"vs {agg_name} each opponent continues with the top {keep*100:.0f}% of their range")
        pick = max(range(len(acts)), key=lambda i: acts[i]["ev_bb"])
        acts[pick]["freq"] = 1.0
        return {"street": cur.name, "hero": hero, "hand": hand_class(hole), "board": cards_str(board),
                "players": len(remaining) + 0, "multiway": True, "opponents": opps,
                "equity_vs_field": round(eq, 3), "pot_bb": round(pot, 2), "to_call_bb": round(call_amt, 2),
                "actions": acts, "recommended": acts[pick]["action"], "sampled": acts[pick]["action"],
                "notes": notes + [f"{len(remaining)} players in the pot: the solver is heads-up only, this is an equity and pot-odds heuristic"],
                "seconds": round(time.time() - t0, 2)}

    # ------------------------------------------------------------ helpers
    def _solve_cached(self, board, r0, r1, pot, stack, budget, bets=(0.0, 0.0), to_act=0, n_raises=0, cfg=None):
        cfg = cfg or self.cfg
        spot = (tuple(board), round(pot, 2), round(stack, 2), tuple(round(b, 2) for b in bets), to_act, n_raises,
                cfg.sizes_key())
        key = (_range_key(r0), _range_key(r1)) + spot
        sol = self._cache.get(key)
        if sol is None:
            # Same spot solved by warm_from (ranges can differ a little: they come from its own earlier
            # streets): start from it and count its time against the budget, as extend() does below.
            warm = self.warm_from._by_spot.get(spot) if self.warm_from is not None else None
            sol = solve(board, r0, r1, pot, stack, bets=bets, to_act=to_act, n_raises=n_raises, cfg=cfg,
                        time_budget=max(budget - warm.seconds, 0.2 * budget) if warm else budget,
                        should_stop=self.should_stop, warm=warm)
            if warm is not None:
                sol.seconds += warm.seconds
            self._cache[key] = sol
            self._by_spot[spot] = sol
        elif sol.seconds < budget * 0.8 and sol.iters < 1000:
            sol.extend(budget - sol.seconds, should_stop=self.should_stop)        # keep going from the stored regrets instead of restarting
        return sol

    def _cfg_with_real_sizes(self, st: StreetState, oop: str, ip: str, stack: float) -> Config:
        """Tree config for this street where the bet/raise sizes actually taken replace the
        nearest abstract size (only when they differ by more than 15%), so range narrowing
        uses the real action instead of snapping it to the abstraction."""
        cfg = self.cfg.copy()
        bets = [0.0, 0.0]
        pot = st.pot_start
        for a in st.actions:
            if a["pos"] not in (oop, ip):
                continue
            p = 0 if a["pos"] == oop else 1
            act = a["action"]
            pot_total = pot + bets[0] + bets[1]
            if act == "bet" and (a.get("amount") or a.get("to")):
                amt = a.get("amount") or a.get("to")
                frac = amt / max(pot_total, 1e-9)
                _swap_nearest(cfg.bet_sizes[st.name], frac)
                bets[p] = amt
            elif act == "raise" and (a.get("to") or a.get("amount")):
                to = a.get("to") or a.get("amount")
                call_amt = bets[1 - p] - bets[p]
                frac = (to - bets[1 - p]) / max(pot_total + call_amt, 1e-9)
                if to < stack - 1e-9:
                    _swap_nearest(cfg.raise_sizes[st.name], frac)
                bets[p] = to
            elif act == "call":
                bets[p] = min(bets[1 - p], stack)
            elif act == "all-in":
                bets[p] = stack
        return cfg

    def _walk(self, sol: Solution, acts: List[dict], oop, ip, pot, stack, notes, street, smoothing=0.0):
        """Follow real actions through the solved tree, narrowing ranges."""
        node = sol.root
        r0, r1 = sol.r0.copy(), sol.r1.copy()
        bets = [0.0, 0.0]
        for a in acts:
            if not isinstance(node, Decision):
                break
            if a["pos"] not in (oop, ip):
                continue
            p = 0 if a["pos"] == oop else 1
            if p != node.player:
                notes.append(f"{street}: action order mismatch at {a['pos']} {a['action']}; ignoring")
                continue
            name = self._tree_action_name(node, a, p, bets, pot)
            k = match_action(node.names, name)
            s = node.avg_strategy0()[k]
            if smoothing > 0:
                s = (1.0 - smoothing) * s + smoothing / len(node.names)
            if p == 0:
                r0 = r0 * s
            else:
                r1 = r1 * s
            if node.names[k] != name:
                notes.append(f"{street}: mapped '{name}' to tree action '{node.names[k]}'")
            act = a["action"]
            if act in ("bet", "raise"):
                amt = a.get("to") if a.get("to") is not None else a.get("amount")
                if amt is None:
                    amt = node.children[k].bets[p] if isinstance(node.children[k], Decision) else bets[1 - p]
                bets[p] = float(amt)
            elif act == "all-in":
                bets[p] = stack
            elif act == "call":
                bets[p] = min(bets[1 - p], stack)
            node = node.children[k]
        return node, r0, r1

    def _tree_action_name(self, node: Decision, a: dict, p: int, bets, pot) -> str:
        act = a["action"]
        if act in ("check", "fold", "call", "all-in"):
            return act
        pot_total = pot + bets[0] + bets[1]
        if act == "bet":
            amt = a.get("amount") or a.get("to") or 0.0
            return f"bet {amt / max(pot_total, 1e-9) * 100:.0f}%"
        to = a.get("to") or a.get("amount") or 0.0
        return f"raise to {to:g}"

    @staticmethod
    def _amount(name: str, node: Decision, p: int) -> Optional[float]:
        child = dict(zip(node.names, node.children)).get(name)
        if name.startswith(("bet", "raise")) or name == "all-in":
            return round(child.bets[p], 2) if isinstance(child, Decision) else round(node.bets[p] + _stack_left(node, p), 2)
        if name == "call":
            return round(node.bets[1 - p] - node.bets[p], 2)
        return None

    def _sample(self, acts) -> str:
        r = self.rng.random(); acc = 0.0
        for a in acts:
            acc += a["freq"]
            if r <= acc:
                return a["action"]
        return acts[-1]["action"]


def _stack_left(node: Decision, p: int) -> float:
    for n, c in zip(node.names, node.children):
        if n == "all-in":
            return _inv_of(c, p) - _inv_of(node, p)
    return 0.0


def _swap_nearest(sizes: List[float], frac: float, tol: float = 0.15) -> None:
    """Replace the entry of `sizes` closest to `frac` with `frac` if it is more than `tol`
    (relative) away; otherwise keep the abstraction (keeps solves cache-friendly)."""
    if not sizes or frac <= 0:
        return
    i = min(range(len(sizes)), key=lambda k: abs(sizes[k] - frac))
    if abs(sizes[i] - frac) / max(sizes[i], 1e-9) > tol and frac not in sizes:
        sizes[i] = round(frac, 3)


RANKS = "AKQJT98765432"


def strategy_grid(strats: List[np.ndarray], names: List[str], weights: np.ndarray) -> dict:
    """13x13 hand-class grid: for each class, the range weight and per-action frequency."""
    cls_idx: Dict[str, List[int]] = {}
    for i, cb in enumerate(ALL_COMBOS):
        cls_idx.setdefault(hand_class(cb), []).append(i)
    out = {}
    for cls, idxs in cls_idx.items():
        idx = np.array(idxs)
        w = weights[idx]
        tot = float(w.sum())
        if tot <= 1e-12:
            out[cls] = {"w": 0.0, "f": [0.0] * len(names)}
            continue
        out[cls] = {"w": round(tot / len(idxs), 3),
                    "f": [round(float((s[idx] * w).sum() / tot), 3) for s in strats]}
    return {"actions": names, "cells": out}


def state_board(board) -> str:
    return cards_str(board)
