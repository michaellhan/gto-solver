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
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .cards import cards as parse_cards, COMBO_INDEX, hand_class, ALL_COMBOS, cards_str
from .ranges import N_COMBOS, range_to_text
from . import preflop as pf
from .pf_classes import push_fold
from .exploit import PoolLock, pool_preflop
from .solver import solve, Config, Solution, Decision, Terminal, Chance, match_action, expand, MEMBER_FULL, _inv_of

DEFAULT_OPEN = {"SB": 3.0}
FOLD_SHADE = 0.75   # multiway: opponents fold this share of what a balanced defence would
# The next four are fitted to the heads-up solver by scripts/calibrate_multiway.py (RAISE_EQ barely
# mattered there, so it stays a judgment call).
NARROW_SOFT = 0.3   # multiway narrowing: weight kept by hands the rules say would have played differently
OFF_RULE_MAX = 0.25  # ...but those hands are at most this share of the narrowed range (a big bet's rule range is small)
VALUE_SHARE = 0.35  # multiway: a bet's value range is this share of what would call it
RAISE_SHARE = 0.2   # multiway: facing a bet, this share of what continues (the strongest) raises
LEAD_EQ = 0.5       # multiway, later streets: we lead when our equity beats this vs every calling range
RAISE_EQ = 0.55     # multiway, later streets: we raise their bet when our equity beats this vs what continues
RARE_LINE = 0.10    # heads-up: widen the villain's range when the solve takes their line less often than this
UNSETTLED = 3.0     # heads-up: a solve this far from equilibrium (% pot) hasn't settled (the page's "converged" cutoff)
JAM_BLEND = 0.04    # preflop vs a jam: call frequency ramps from 0 to 1 over this much equity around the pot odds
PREFLOP_BOARDS = 400  # boards sampled for preflop all-in equity
PRICE_BLEND = 0.02  # preflop, off-chart price: continue frequency ramps over this much (scaled) equity around the pot odds
RAISE_BLEND = 0.03  # preflop, more players in: value-raise frequency ramps over this much equity when called
COMMIT = 0.4       # preflop: a raise that would put in this share of the stack goes all-in instead
JAM_SPR = 1.5      # ...as does one leaving less than this many pots behind once called
PUSH_FOLD_STACK = 12.0  # preflop: at or below this many bb, raising first in is all-in (push/fold)
CHART_STACK = 100.0  # the depth the preflop charts are written for
IMPLIED = {"pair": 0.06, "suited": 0.03}  # preflop calls: share of the stack behind (beyond the chart's) small pairs / suited hands expect to win later
DEPTH_TOL = 20.0   # ...a spot this many bb deeper or shallower than the charts re-prices calls
PRICE_TOL = 0.02    # ...and a spot within this much of the chart's pot odds, against the raisers only, is the chart's own
CUT_FLOOR = 0.05    # ...a worse price holds the chart's continuing hands to the equity of its weakest this share of them
EV_GAP = 1.0        # heads-up, unsettled solve: recommend the highest EV when it beats the top of the mix by this % of pot
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
    posted: Dict[str, float] = field(default_factory=dict)   # blinds (and straddle) posted this street


def _showdown_share(st, idx, hands) -> float:
    """Hero's (combo idx) share of the pot at showdown against these opponent combos, strengths st."""
    best = max(st[j] for j in hands)
    if st[idx] != best:
        return float(st[idx] > best)
    return 1.0 / (1 + sum(st[j] == best for j in hands))


def replay_hand(positions: List[str], stack: float, streets: Dict[str, List[dict]], n_streets: int) -> List[StreetState]:
    """Replay preflop..current street. Sizes: bet=amount added this street, raise=to."""
    ante = min(getattr(positions, "ante", 0.0), stack)
    inv = {p: ante for p in positions}          # antes: dead money in the pot from the start
    active = set(positions)
    allin: set = set()
    out: List[StreetState] = []
    post = pf.postflop_order(positions)
    for si in range(n_streets):
        name = STREETS[si]
        pre = name == "preflop"
        order = (positions.preflop_order() if hasattr(positions, "preflop_order") else list(positions)) if pre else post
        st = StreetState(name, order, [p for p in order if p in active], sum(inv.values()), dict(inv))
        st.bets = {p: 0.0 for p in positions}
        if pre:
            st.posted = positions.posted() if hasattr(positions, "posted") else (
                {"SB": 0.5, "BB": 1.0} if "SB" in positions else {"BB": 1.0})
            for p, b in st.posted.items():
                st.bets[p] = min(b, stack - inv[p])
            st.cur_bet = max(st.posted.values())
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


# ----------------------------------------------------------------------------- multiway range rules
# Not a solve: simple rules for how players with ranges act once 3+ are in the pot. Used by
# Advisor._multiway and to narrow ranges through multiway streets (see _narrow_street).

def _top(r: np.ndarray, frac: float, rank: np.ndarray) -> np.ndarray:
    """Mask of the strongest combos (in `rank` order) holding `frac` of range r's weight."""
    m = np.zeros(N_COMBOS, dtype=bool)
    if frac > 1e-9 and r.sum() > 0:
        cum = np.cumsum(r[rank])
        m[rank[:int(np.searchsorted(cum, frac * cum[-1] - 1e-9)) + 1]] = True
    return m


def _aggro(r, v, b, p, rank) -> np.ndarray:
    """A bet or raise of b into p: value with the top v of r plus balanced bluffs from the bottom,
    bluffs = value * b / (p + b)."""
    return _top(r, v, rank) | _top(r, min(v * b / (p + b), 1.0 - v), rank[::-1])


def _keep(b, p, n, shade=None) -> float:
    """Share of their range each of n players continues with against a bet of b into p: together
    they fold shade * b / (p + b) of the time (shade 1 = a balanced defence)."""
    shade = FOLD_SHADE if shade is None else shade
    return 1.0 - (shade * b / (p + b)) ** (1.0 / max(n, 1))


def _joint_top(rs: List[np.ndarray], fold: float, rank: np.ndarray) -> List[np.ndarray]:
    """What each of the ranges rs continues with against a bet they fold to together with
    probability `fold`: everyone keeps the hands above one strength bar (in `rank` order), set so
    the chance that every player's hand is below it is `fold`. A range of weak hands gives up more
    of itself than a strong one: after a lead and a call, the caller (who called with what beats
    the lead) keeps most of their range and the leader's bluffs fold. One range: _top(r, 1 - fold)."""
    below = np.ones(N_COMBOS + 1)      # below[m]: chance all hands are outside the top m combos
    for r in rs:
        c = np.cumsum(r[rank])
        if c[-1] > 0:
            below *= 1.0 - np.concatenate(([0.0], c)) / c[-1]
    m = int(np.argmax(below <= fold + 1e-9)) if fold < 1.0 else 0
    out = []
    for _ in rs:
        mask = np.zeros(N_COMBOS, dtype=bool)
        mask[rank[:m]] = True
        out.append(mask)
    return out


def _share(r, mask) -> float:
    """Share of range r's weight on mask."""
    t = r.sum()
    return float((r * mask).sum() / t) if t > 0 else 0.0


def _fold_together(costs, p, shade=None) -> float:
    """Chance the players paying `costs` to continue into a pot of p fold together (see _keep)."""
    return float(np.prod([1.0 - _keep(c, p, len(costs), shade) for c in costs]))


@lru_cache(maxsize=64)
def _street_rank(board: tuple):
    """(strength averaged over runouts, combos strongest first) on this board; -1 = blocked."""
    from .evaluator import eval_board
    rng = random.Random(sum(c * 53 ** i for i, c in enumerate(board)))
    avail = [c for c in range(52) if c not in board]
    tot = np.zeros(N_COMBOS); cnt = np.zeros(N_COMBOS)
    for _ in range(1 if len(board) == 5 else 24):
        s = eval_board(list(board) + rng.sample(avail, 5 - len(board)))
        ok = s >= 0
        tot[ok] += s[ok]; cnt[ok] += 1
    sc = np.where(cnt > 0, tot / np.maximum(cnt, 1), -1.0)
    return sc, np.argsort(-sc)


@lru_cache(maxsize=1)
def _preflop_boards():
    """Fixed random boards and every combo's strength on each (-1 = blocked), for preflop equity."""
    from .evaluator import eval_board
    rng = random.Random(1)
    return np.array([eval_board(rng.sample(range(52), 5)) for _ in range(PREFLOP_BOARDS)])


def _implied_w(h: str) -> float:
    """Implied-odds weight of a hand class: the ones that win big pots by hitting (pairs 22-99, suited
    hands that aren't two big cards); premiums and offsuit hands don't depend on it."""
    if len(h) == 2:
        return IMPLIED["pair"] if h[0] in "23456789" else 0.0
    if h.endswith("s") and not (h[0] in "AKQJT" and h[1] in "AKQJT"):
        return IMPLIED["suited"]
    return 0.0


_IMPLIED_W = np.array([_implied_w(hand_class(c)) for c in ALL_COMBOS])
_CARD_MASK = np.array([(1 << a) | (1 << b) for a, b in ALL_COMBOS], dtype=np.int64)


def _preflop_equity(villain: np.ndarray) -> np.ndarray:
    """All-in preflop equity of every combo against the weighted range `villain`, from the precomputed
    hand-class equity table (exact up to class-level card removal; board sampling was noisier than the
    gaps between neighbouring hands)."""
    from .pf_classes import equity_matrix, to_classes, PAIR_COMPAT, CLASS_OF
    w = PAIR_COMPAT * to_classes(villain)[None, :]
    den = w.sum(1)
    eq = np.where(den > 1e-12, (equity_matrix() * w).sum(1) / np.maximum(den, 1e-12), np.nan)
    return eq[CLASS_OF]


def _preflop_equity_multi(villains: List[np.ndarray]) -> np.ndarray:
    """Preflop equity of every combo against several weighted ranges at once: on each board, the product
    of the chances of beating each (ties count half; card removal between the villains is ignored)."""
    if len(villains) == 1:
        return _preflop_equity(villains[0])
    sts = _preflop_boards()
    vs = []
    for r in villains:
        v = np.nonzero(r > 1e-9)[0]
        vs.append((v, r[v], (_CARD_MASK[:, None] & _CARD_MASK[None, v]) == 0))
    n = sum(len(v) for v, _, _ in vs)
    num = np.zeros(N_COMBOS); den = np.zeros(N_COMBOS)
    for s in sts[: max(60, min(len(sts), int(6e7 / (N_COMBOS * max(n, 1)))))]:
        p = (s >= 0).astype(float)
        for v, w, ok in vs:
            sv = s[v]
            wl = (ok & (sv >= 0)[None, :]) * w
            d = wl.sum(1)
            win = (wl * ((s[:, None] > sv[None, :]) + 0.5 * (s[:, None] == sv[None, :]))).sum(1)
            p = p * np.where(d > 0, win / np.maximum(d, 1e-12), 0.0)
        num += p; den += s >= 0
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _narrow_street(ranges: Dict[str, np.ndarray], board, st: StreetState, stack: float, hero=None, hole=()) -> None:
    """Narrow each player's range (in place) by what they did on a street that started multiway,
    with the multiway rules: a check drops most of the hands that would have bet, a bet or raise
    keeps its value + bluff range, a call keeps what continues minus what would have raised.
    Hands a rule rules out keep some weight (players don't follow our rules), see _soften."""
    sc, rank = _street_rank(tuple(board))
    dead = [MEMBER_FULL[c] for c in hole]
    bets = {p: 0.0 for p in st.inv_start}
    cur = 0.0
    active = set(st.players_start)
    for a in st.actions:
        p, act = a["pos"], a["action"]
        if p not in active or p not in ranges:
            continue
        if act == "fold":
            active.discard(p)
            continue
        r = ranges[p] * (sc >= 0)
        for d in dead if p != hero else ():   # opponents can't hold our cards
            r = r * (1.0 - d)
        if r.sum() <= 0:
            continue
        n = len(active) - 1
        pot = st.pot_start + sum(bets.values())
        remaining = stack - st.inv_start[p] - bets[p]
        to_call = max(0.0, min(cur - bets[p], remaining))
        if act == "all-in":
            to = bets[p] + remaining
        elif act in ("bet", "raise"):
            to = a.get("to") if a.get("to") is not None else a.get("amount")
            to = min(float(to) if to is not None else _default_size_post(pot, cur, bets[p]), bets[p] + remaining)
        else:
            to = cur
        keep_m = _rule_mask(r, to, cur, to_call, pot, n, rank)
        cur = max(cur, to)
        bets[p] = max(bets[p], min(to, bets[p] + remaining))
        ranges[p] = r * _soften(r, keep_m)


def _soften(r, keep_m) -> np.ndarray:
    """Per-combo weights for range r after a line the rules play with keep_m: 1 on the rule's hands,
    NARROW_SOFT off them, lowered so the off-rule hands make up at most OFF_RULE_MAX of the result."""
    on, off = float(r[keep_m].sum()), float(r[~keep_m].sum())
    soft = NARROW_SOFT
    if off * soft > 0 and OFF_RULE_MAX < 1.0:
        soft = min(soft, OFF_RULE_MAX / (1.0 - OFF_RULE_MAX) * on / off)
    return np.where(keep_m, 1.0, soft)


def _rule_mask(r, to, cur, to_call, pot, n, rank) -> np.ndarray:
    """The part of range r the multiway rules play this way, with n opponents and `pot` in the
    middle: a bet or raise to `to` over the current bet `cur`, else a call of to_call (or a check)."""
    if to > cur + 1e-9:                       # bet or raise: value + balanced bluffs
        if to_call <= 1e-9:
            return _aggro(r, _keep(to - cur, pot, n, 1.0) * VALUE_SHARE, to - cur, pot, rank)
        return _aggro(r, _keep(to_call, pot, n) * RAISE_SHARE, to - cur, pot + to_call, rank)
    if to_call <= 1e-9:                       # check: not what would have bet
        b = 0.66 * pot
        return ~_aggro(r, _keep(b, pot, n, 1.0) * VALUE_SHARE, b, pot, rank)
    k = _keep(to_call, pot, n)                # call: continues, but wouldn't raise
    return _top(r, k, rank) & ~_top(r, k * RAISE_SHARE, rank)


def _default_size_post(pot, cur, mine) -> float:
    return cur + 0.75 * (pot + cur - mine) if cur > mine else 0.66 * pot


def _default_size(st: StreetState, p: str, pre: bool) -> float:
    if pre:
        # chips the other live players (not the last raiser) put in beyond their blinds: a caller of
        # the current bet adds one bet to the size (a squeeze), a limper 1bb (an iso-raise)
        last = next((a["pos"] for a in reversed(st.actions) if a["action"] in ("raise", "all-in", "bet")), None)
        extra = sum(max(0.0, st.bets[q] - st.posted.get(q, 0.0)) for q in st.active if q not in (p, last))
        if st.n_raises == 0:
            if extra > 0:   # over limpers: 3x (4x from the blinds) plus one bet per limper
                return st.cur_bet * (4.0 if p in st.posted else 3.0) + extra
            return DEFAULT_OPEN.get(p, 2.5) * st.cur_bet
        if st.n_raises == 1:
            return st.cur_bet * (4.0 if p in st.posted else 3.0) + extra
        return st.cur_bet * 2.3 + extra
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


# ----------------------------------------------------------------------------- adjusted preflop play
# Every player's preflop decisions get the same off-chart adjustments as hero's (Advisor._priced: price,
# players in, depth, antes), so the ranges replayed into later decisions match the advice. Cached per
# action line: each decision is worked out once.
_PF_CACHE: Dict[tuple, object] = {}


def _pf_key(kind, p, actions, positions, stack) -> tuple:
    acts = tuple((a["pos"], a["action"], a.get("to"), a.get("amount")) for a in actions)
    return (kind, p, acts, tuple(positions), getattr(positions, "ante", 0.0), getattr(positions, "straddle", False), stack)


def _pf_cached(key, fn):
    if key not in _PF_CACHE:
        if len(_PF_CACHE) > 512:
            _PF_CACHE.clear()
        _PF_CACHE[key] = fn()
    return _PF_CACHE[key]


def preflop_strategy(p: str, before: List[dict], positions, stack: float) -> Dict[str, np.ndarray]:
    """p's preflop strategy per combo after the actions `before`: the chart, adjusted like hero's."""
    def work():
        full = pf.strategy_for(p, before, positions)
        if "call" in full:
            st = replay_hand(positions, stack, {"preflop": before}, 1)[0]
            if not st.closed and st.to_act == p:
                full = Advisor._priced(p, before, positions, stack, st, full, [])
        return full
    return _pf_cached(_pf_key("s", p, before, positions, stack), work)


def preflop_ranges(actions: List[dict], positions, stack: float, pool_except: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Each still-active player's range after the preflop actions (pf.replay_ranges with the adjusted
    strategies). With `pool_except` (hero's seat), everyone else plays like a typical player pool
    (exploit.pool_preflop)."""
    def work():
        ranges = {p: np.ones(N_COMBOS) for p in positions}
        active = set(positions)
        for i, a in enumerate(actions):
            pos, act = a["pos"], a["action"]
            if pos not in ranges:
                continue
            if act == "fold":
                active.discard(pos)
                continue
            strat = preflop_strategy(pos, actions[:i], positions, stack)
            if pool_except is not None and pos != pool_except:
                strat = pool_strategy(strat, pos, actions[:i], ranges)
            key = "raise" if act == "all-in" else ("call" if act == "check" and "check" not in strat else act)
            if key in strat:
                ranges[pos] = ranges[pos] * strat[key]
            elif act == "call" and not any(h["action"] in ("raise", "all-in") for h in actions[:i]):
                ranges[pos] = ranges[pos] * pf.LIMP
        return {p: ranges[p] for p in active}
    return _pf_cached(_pf_key("r", pool_except, actions, positions, stack), work)


def pool_strategy(strat: Dict[str, np.ndarray], p: str, before: List[dict], ranges: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """p's preflop strategy `strat` (after the actions `before`) as a player-pool player would play it:
    exploit.pool_preflop, with p's equity against the raiser to pick the extra flats facing an open.
    `ranges`: everyone's range at that point."""
    raisers = [h["pos"] for h in before if h["action"] in ("raise", "all-in")]
    eq = _preflop_equity(ranges[raisers[0]]) if len(raisers) == 1 and raisers[0] in ranges else None
    return pool_preflop(strat, len(raisers), p, ranges.get(p, np.ones(N_COMBOS)), eq)


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
        ante = float(state.get("ante") or 0.0) / bb
        straddle = bool(state.get("straddle")) and n >= 3
        positions = pf.Table(pf.SEATS[n], ante, straddle)
        hero = str(state["hero"]).upper()
        if hero not in positions:
            return {"error": f"hero seat {hero} is not one of {positions}"}
        hole = tuple(parse_cards(str(state["hole"])))
        board = parse_cards(str(state.get("board") or ""))
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
            return {"error": "need two hole cards and 0, 3, 4 or 5 board cards"}
        if len(set(hole) | set(board)) != len(hole) + len(board):
            return {"error": "the same card appears twice in your hand and the board"}
        if state.get("stacks"):
            stack = min(float(v) for v in state["stacks"].values()) / bb
        else:
            stack = float(state.get("stack", 100.0)) / bb
        streets = {s: _norm_actions(state.get(s), bb) for s in STREETS}
        if not board:
            return self._preflop(hero, hole, streets["preflop"], positions, stack, t0,
                                 pool=state.get("opponents") == "pool")
        saved = self.time_budget, self.prior_budget
        if state.get("budget"):
            self.time_budget = max(0.5, min(120.0, float(state["budget"])))
        if state.get("prior_budget"):
            self.prior_budget = max(0.3, min(30.0, float(state["prior_budget"])))
        try:
            return self._postflop(hero, hole, board, stack, streets, positions, t0,
                                  pool=state.get("opponents") == "pool")
        finally:
            self.time_budget, self.prior_budget = saved

    # ------------------------------------------------------------ preflop
    def _preflop(self, hero, hole, preflop, positions, stack, t0, pool=False):
        st = replay_hand(positions, stack, {"preflop": preflop}, 1)[0]
        if st.closed:
            return {"error": "preflop betting is already closed (deal the flop)"}
        if st.to_act != hero:
            return {"error": f"it is {st.to_act}'s turn, not {hero}'s"}
        aggr = [a["pos"] for a in preflop if a["action"] in ("raise", "all-in", "bet")]
        if aggr and aggr[-1] in st.allin and aggr[-1] != hero:
            return self._vs_jam(hero, hole, preflop, positions, stack, st, t0, pool)
        full = pf.strategy_for(hero, preflop, positions)
        notes = ["preflop: chart approximation"]
        if "call" in full:
            full = self._priced(hero, preflop, positions, stack, st, full, notes, pool)
        elif pool:
            notes.append("exploit mode: opens follow the chart (the pool 3bets less but flats more, "
                         "so there's no clear edge in opening wider or tighter)")
        idx = COMBO_INDEX[(min(hole), max(hole))]
        strat = {k: float(v[idx]) for k, v in full.items()}
        remaining = stack - st.inv_start[hero] - st.bets[hero]
        to_call = min(st.cur_bet - st.bets[hero], remaining)
        size = min(_default_size(st, hero, True), st.bets[hero] + remaining)
        size = round(size * 2) / 2
        if "raise" in full and remaining > to_call + 1e-9:
            # all-in instead: the raise is most of the stack, or leaves under JAM_SPR pots behind once called
            pot = st.pot_start + sum(st.bets.values())
            pot_called = pot + 2 * size - st.bets[hero] - st.cur_bet
            short_open = st.n_raises == 0 and stack <= PUSH_FOLD_STACK
            if (size >= st.bets[hero] + remaining - 1e-9 or st.inv_start[hero] + size >= COMMIT * stack
                    or (stack - st.inv_start[hero] - size) < JAM_SPR * pot_called or short_open):
                size = st.bets[hero] + remaining
                if short_open or stack < CHART_STACK - DEPTH_TOL:
                    full = self._push(hero, preflop, positions, stack, st, size, full, notes, pool)
                else:
                    notes.append("the raise would commit the stack, so it goes all-in")
                strat = {k: float(v[idx]) for k, v in full.items()}
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
        names = list(full.keys())
        arriving = preflop_ranges(preflop, positions, stack).get(hero, np.ones(N_COMBOS))
        return {
            "street": "preflop", "hero": hero, "hand": hand_class(hole), "players": len(positions),
            "actions": acts, "recommended": best, "sampled": self._sample(acts),
            "pot_bb": round(st.pot_start + sum(st.bets.values()), 2), "to_call_bb": round(to_call, 2),
            "grid": strategy_grid([full[k] for k in names], names, arriving),
            "notes": notes, "seconds": round(time.time() - t0, 3),
        }

    @staticmethod
    def _push(hero, preflop, positions, stack, st, jam, full, notes, pool=False):
        """A raise that would commit much of the stack is an all-in: hero's range for it is the push/fold
        equilibrium against the players still in (the last raiser's range calls, if there is one); hands
        the chart continued with that don't shove call."""
        arr = preflop_ranges(preflop, positions, stack, hero if pool else None)
        last = next((a["pos"] for a in reversed(preflop) if a["action"] in ("raise", "all-in")), None)
        opp = [p for p in st.order if p in st.active and p != hero and p not in st.allin]
        call_rng = arr.get(last, np.ones(N_COMBOS)) if last else np.ones(N_COMBOS)
        caller_in = st.bets[last] if last else float(np.mean([st.bets[p] for p in opp]))
        pot = st.pot_start + sum(st.bets.values())
        push = push_fold(arr.get(hero, np.ones(N_COMBOS)), call_rng, jam, st.bets[hero], caller_in, pot,
                         max(1, len(opp)))
        out = dict(full)
        out["raise"] = push
        if "call" in full:   # what the chart continued with and doesn't shove calls
            out["call"] = np.minimum(full["call"] + full.get("raise", 0.0), 1.0 - push)
        if "check" in full:
            out["check"] = 1.0 - push
        else:
            out["fold"] = 1.0 - push - out.get("call", 0.0)
            out = pf.monotone(out)
        notes.append(f"a raise here commits too much of a {jam + st.inv_start[hero]:g}bb stack: all-in, "
                     f"range from the push/fold equilibrium against {len(opp)} player{'s' * (len(opp) > 1)}")
        return out

    @staticmethod
    def _priced(hero, preflop, positions, stack, st, full, notes, pool=False):
        """The charts assume a standard-sized raise, nobody else in the pot, 100bb and no antes. When the
        spot differs, re-draw hero's line against the chart's own standard line (same raisers at default
        sizes, everyone else folding, chart depth, no antes):
        - continuing: the chart's continuing share of hero's range there marks an equity bar (its
          weakest continuing hand's equity against the raisers, relative to that line's pot odds); hero
          continues where equity against everyone in, held to the same bar, beats the actual pot odds
          (with implied odds for small pairs and suited hands by how much deeper or shallower than the
          charts). A better price only adds calls, a worse one only removes them.
        - raising, when more players are in: each player's chart facing hero's raise gives their continuing
          range and how often they continue. The chart's raises split into value (the top of hero's range
          by equity when called, as many hands as it raises) and bluffs. A value raise stays only while its
          equity when called (against each continuing range, weighted by how often it continues) reaches
          the chart's weakest value raise heads-up; bluffs shrink by how much less often everyone folds.
        With `pool` (exploit mode) the other players' ranges and their responses to hero's raise are the
        player pool's (exploit.pool_preflop), so this runs even in the chart's own spot: calls follow
        equity against what the pool actually plays, and bluff raises grow (up to 1.5x) when the pool
        folds more often."""
        def priced(s, stk):
            opp = [p for p in s.order if p in s.active and p != hero and
                   (p in s.acted or s.bets[p] > s.posted.get(p, 0.0) + 1e-9)]
            call = min(s.cur_bet - s.bets[hero], stk - s.inv_start[hero] - s.bets[hero])
            pot = s.pot_start + sum(s.bets.values())
            return opp, call / (pot + call), call, pot, stk - s.inv_start[hero] - s.bets[hero] - call

        def facing_raise(line, opp, arr, positions, as_pool=False):
            # each player facing hero's raise, by their chart (or as the pool plays it): the chance
            # everyone folds, and hero's equity when called (against each continuing range, weighted by
            # how often it continues)
            line = line + [{"pos": hero, "action": "raise"}]
            if as_pool:   # hero's raising range, for the pool's extra flats against an open
                arr = dict(arr, **{hero: arr.get(hero, np.ones(N_COMBOS)) * full.get("raise", 1.0)})
            f, num, den = 1.0, np.zeros(N_COMBOS), 0.0
            for p in opp:
                w = arr[p]
                s_p = pf.strategy_for(p, line, positions)
                if as_pool:
                    s_p = pool_strategy(s_p, p, line, arr)
                cont = w * (1.0 - s_p.get("fold", 0.0))
                c = float(cont.sum() / max(w.sum(), 1e-12))
                f *= 1.0 - c
                if c > 1e-6:
                    num = num + c * _preflop_equity(cont)
                    den += c
            return f, num / max(den, 1e-12)

        def quantile(eq, w, share):
            ok = ~np.isnan(eq) & (w > 0)
            order = np.argsort(-eq[ok])
            cum = np.cumsum(w[ok][order])
            return float(eq[ok][order][min(np.searchsorted(cum, share * cum[-1]), len(cum) - 1)])

        std_line = [{"pos": a["pos"], "action": "fold"} if a["action"] == "call" and a["pos"] != hero
                    else {"pos": a["pos"], "action": a["action"]} for a in preflop]
        # the chart's own spot: its depth, no antes
        table = pf.Table(positions, 0.0, getattr(positions, "straddle", False))
        std = replay_hand(table, CHART_STACK, {"preflop": std_line}, 1)[0]
        if std.closed or std.to_act != hero:
            return full
        opp, need, call, pot, behind = priced(st, stack)
        opp_std, need_std, _, _, behind_std = priced(std, CHART_STACK)
        # implied odds: pairs and suited hands win more when they hit, by the stack behind. Deeper or
        # shallower than the charts moves their price by how their odds with those winnings change at
        # this price, so 20bb shallower costs a small pair a little of the implied odds it had at chart
        # depth, not all of them.
        depth = behind - behind_std
        implied = lambda b: call / np.maximum(pot + call + _IMPLIED_W * max(b, 0.0), call + 1e-9)
        need_v = need + implied(behind) - implied(behind_std)
        more = sorted(opp) != sorted(opp_std)
        reprice = abs(need - need_std) >= PRICE_TOL or abs(depth) >= DEPTH_TOL or pool
        if not (more or reprice):
            return full
        raise_c = full.get("raise", np.zeros(N_COMBOS))
        call_c = full["call"]
        arr_std = pf.replay_ranges(std_line, table)
        w = arr_std[hero]
        tot = max(w.sum(), 1e-12)
        eq_std = _preflop_equity_multi([arr_std[p] for p in opp_std])
        arr = preflop_ranges(preflop, positions, stack, hero if pool else None)
        eq = _preflop_equity_multi([arr[p] for p in opp])
        out = dict(full)
        msg = []
        value_n = bluff_n = raise_c
        value_c = raise_c
        r = float((w * raise_c).sum() / tot)
        if (more or pool) and r > 0.005:
            f, eq_c = facing_raise(preflop, opp, arr, positions, pool)
            f_std, eq_c_std = facing_raise(std_line, opp_std, arr_std, table)
            v_bar = quantile(eq_c_std, w, r)
            value_c = np.where(eq_c_std >= v_bar, raise_c, 0.0)
            bluff_c = raise_c - value_c
            keep = np.clip(0.5 + (eq_c - v_bar) / RAISE_BLEND, 0.0, 1.0)
            keep[np.isnan(keep)] = 0.0
            # only ever tighter than the chart: the charts' continuing ranges don't see the bigger
            # squeeze size, and capped callers would otherwise make value raises look wider
            value_n = np.minimum(value_c, keep)
            scale = min(1.5 if pool else 1.0, f / max(f_std, 1e-9))
            bluff_n = bluff_c * scale
            out["raise"] = np.minimum(1.0, value_n + bluff_n)
            msg.append(f"raises re-drawn for {'the player pool' if pool and not more else f'{len(opp)} players in'}: value raises keep {v_bar:.0%} equity against "
                       f"what continues (the chart's weakest value raise heads-up); bluffs x{scale:.2f} "
                       f"(everyone folds {f:.0%} vs {f_std:.0%})")
        raise_n = out.get("raise", raise_c)
        # continuing before the price: the chart's calls and value raises, kept bluffs, new value raises
        base = np.minimum(1.0, np.maximum(call_c + value_c + (raise_n - value_n if more or pool else raise_c - value_c), raise_n))
        new = base
        if reprice:
            q = float((w * (call_c + raise_c)).sum() / tot)
            if 0.02 < q < 0.98:
                # a better price adds hands the chart folds that clear its bar (the equity at its continuing share)
                realize = need_std / max(quantile(eq_std, w, q), 1e-6)
                add = np.clip(0.5 + (eq * realize - need_v) / PRICE_BLEND, 0.0, 1.0)
                add[np.isnan(add)] = 0.0
                # a worse price removes calls, never raises. The chart keeps playable hands (small pairs,
                # suited connectors) below that raw-equity bar, so its continuing hands are held to its
                # weakest ones instead: a slightly worse price trims the bottom, not every such hand.
                floor = quantile(eq_std, w * (call_c + raise_c), 1.0 - CUT_FLOOR)
                cut = np.clip(1.0 + (eq * need_std / max(floor, 1e-6) - need_v) / PRICE_BLEND, 0.0, 1.0)
                cut[np.isnan(cut)] = 0.0
                # a hand gains where the price is better, or (exploit mode) where it does better against the
                # pool's range than against the chart's
                better = need_v < need_std
                if pool:
                    better = better | (np.nan_to_num(eq, nan=0.0) > np.nan_to_num(eq_std, nan=0.0) + 1e-6)
                new = np.where(better, np.maximum(base, add), np.minimum(base, np.maximum(cut, raise_n)))
                deep = (f", {abs(depth):g}bb {'deeper' if depth > 0 else 'shallower'} than the charts "
                        f"(implied odds for pairs and suited hands)" if abs(depth) >= DEPTH_TOL else "")
                msg.insert(0, f"price {need:.0%} of the pot against {', '.join(opp)} (the chart assumes "
                              f"{need_std:.0%} against {', '.join(opp_std)}{deep}): calls re-drawn by equity "
                              f"against the pot odds" + (", against the player pool's ranges" if pool else ""))
        if "raise" in full:
            out["raise"] = np.minimum(raise_n, new)
        out["call"] = new - out.get("raise", 0.0)
        out["fold"] = 1.0 - new
        notes.extend(msg)
        return pf.monotone(out)

    def _vs_jam(self, hero, hole, preflop, positions, stack, st, t0, pool=False):
        """Facing a preflop all-in: call when equity against the jammer's range beats the pot odds.
        With `pool`, a jam over a raise comes from the player pool's raising range (premiums, mostly)."""
        ji = max(i for i, a in enumerate(preflop) if a["action"] in ("raise", "all-in", "bet"))
        jammer, before = preflop[ji]["pos"], preflop[:ji]
        prev = replay_hand(positions, stack, {"preflop": before}, 1)[0]
        pool_except = hero if pool else None
        arr = preflop_ranges(before, positions, stack, pool_except)
        rng = arr.get(jammer, np.ones(N_COMBOS))
        jam = st.bets[jammer]
        notes = ["preflop: facing an all-in, call when equity beats the pot odds"]
        if prev.n_raises <= 1:
            # open- or 3bet-jam: push/fold equilibrium against the players left to act (the opener, if
            # any, is the one who calls)
            opp = [p for p in prev.order if p in prev.active and p != jammer]
            caller = next((p for p in opp if prev.bets[p] >= prev.cur_bet - 1e-9 and prev.n_raises), None)
            call_rng = arr.get(caller, np.ones(N_COMBOS)) if caller else np.ones(N_COMBOS)
            caller_in = prev.bets[caller] if caller else float(np.mean([prev.bets[p] for p in opp]))
            pot_before = prev.pot_start + sum(prev.bets.values())
            push = push_fold(rng, call_rng, jam, prev.bets[jammer], caller_in, pot_before, len(opp))
            share = float((rng * push).sum() / max(rng.sum(), 1e-12))
            rng = rng * push
            notes.append(f"{jammer}'s {jam:g}bb jam range: push/fold equilibrium against {len(opp)} "
                         f"player{'s' * (len(opp) > 1)} left ({share:.0%} of their hands)")
        else:
            s_j = preflop_strategy(jammer, before, positions, stack)
            if pool:
                s_j = pool_strategy(s_j, jammer, before, arr)
                notes.append(f"exploit mode: {jammer}'s jam is the player pool's raising range")
            rng = rng * s_j.get("raise", np.ones(N_COMBOS))
        remaining = stack - st.inv_start[hero] - st.bets[hero]
        to_call = min(st.cur_bet - st.bets[hero], remaining)
        pot = st.pot_start + sum(st.bets.values())
        # hero's call only wins what the jammer can match
        win = pot - max(0.0, st.bets[jammer] - (st.bets[hero] + to_call))
        need = to_call / (win + to_call)
        # everyone who already called the jam is in the pot too: equity against all of them
        after = preflop_ranges(preflop, positions, stack, pool_except)
        callers = [a["pos"] for a in preflop[ji + 1:] if a["action"] in ("call", "all-in") and a["pos"] in st.active]
        eq = _preflop_equity_multi([rng] + [after[p] for p in callers])
        idx = COMBO_INDEX[(min(hole), max(hole))]
        call = np.clip(0.5 + (eq - need) / JAM_BLEND, 0.0, 1.0)
        call[np.isnan(call)] = 0.0
        if callers:
            notes.append(f"{', '.join(callers)} already called: equity is against {jammer} and them")
        behind = [p for p in st.active if p not in (hero, jammer) and p not in st.allin and p not in callers]
        if behind:
            notes.append(f"{', '.join(behind)} still to act behind (not counted)")
        notes.append(f"equity {eq[idx]:.1%} vs {need:.1%} needed")
        acts = [{"action": "call", "freq": round(float(call[idx]), 3), "amount_bb": round(to_call, 2)},
                {"action": "fold", "freq": round(1 - float(call[idx]), 3)}]
        arriving = after.get(hero, np.ones(N_COMBOS))
        return {
            "street": "preflop", "hero": hero, "hand": hand_class(hole), "players": len(positions),
            "actions": acts, "recommended": "call" if call[idx] >= 0.5 else "fold", "sampled": self._sample(acts),
            "pot_bb": round(pot, 2), "to_call_bb": round(to_call, 2), "equity": round(float(eq[idx]), 4),
            "grid": strategy_grid([call, 1 - call], ["call", "fold"], arriving),
            "notes": notes, "seconds": round(time.time() - t0, 3),
        }

    # ------------------------------------------------------------ postflop
    def _postflop(self, hero, hole, board, stack, streets, positions, t0, pool=False):
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
        # a copy: narrowing below replaces entries, and the cached dict is shared with later requests
        ranges = dict(preflop_ranges(streets["preflop"], positions, stack, hero if pool else None))
        # Streets that started with 3+ players can't be solved; narrow ranges there by rule.
        mw_streets = [st for st in rep[1:] if len(st.players_start) > 2]
        for st in mw_streets:
            _narrow_street(ranges, board[:{"flop": 3, "turn": 4, "river": 5}[st.name]], st, stack, hero, hole)
        if mw_streets:
            notes.append(f"{', '.join(st.name for st in mw_streets)}: multiway, ranges narrowed by "
                         "simple betting rules rather than a solve")
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
        widened: List[str] = []
        if hu_from_start:
            # narrow through earlier streets that were also heads-up between these two
            for st in rep[1:-1]:
                sub_board = board[:{"flop": 3, "turn": 4, "river": 5}[st.name]]
                if st.players_start != remaining:
                    continue   # multiway: narrowed above
                eff = min(stack - st.inv_start[oop], stack - st.inv_start[ip])
                cfg = self._cfg_with_real_sizes(st, oop, ip, eff)
                sol = self._solve_cached(sub_board, r0, r1, st.pot_start, eff, self.prior_budget, cfg=cfg)
                node, c0, c1 = self._walk(sol, st.actions, oop, ip, st.pot_start, eff, notes, st.name, self.prior_smoothing,
                                          board=sub_board, rules=(0, 1),
                                          lock=PoolLock(sol.ctx, 1 - hero_p) if pool else None)
                r0, r1 = expand(sol.ctx, c0), expand(sol.ctx, c1)
            eff = min(stack - cur.inv_start[oop], stack - cur.inv_start[ip])
            cfg = self._cfg_with_real_sizes(cur, oop, ip, eff)
            r0, r1 = self._with_hero(r0, r1, hero_p, hole, notes)
            sol = self._solve_cached(board, r0, r1, cur.pot_start, eff, self.time_budget, cfg=cfg)
            # The villain's actions this street are exact unless the solve rarely takes them: then
            # smooth as on earlier streets, so we aren't advised against a sliver of a range.
            node, n_r0, n_r1 = self._walk(sol, cur.actions, oop, ip, cur.pot_start, eff, notes, cur.name,
                                          self.prior_smoothing, only=1 - hero_p, widened=widened, board=board,
                                          lock=PoolLock(sol.ctx, 1 - hero_p) if pool else None)
        else:
            # pot became heads-up during this street: solve from the current node
            eff = min(stack - cur.inv_start[oop], stack - cur.inv_start[ip])
            r0, r1 = self._with_hero(r0, r1, hero_p, hole, notes)
            sol = self._solve_cached(board, r0, r1, cur.pot_start, eff, self.time_budget,
                                     bets=(cur.bets[oop], cur.bets[ip]), to_act=hero_p, n_raises=cur.n_raises)
            node, n_r0, n_r1 = sol.root, sol.r0.copy(), sol.r1.copy()
            notes.append(f"{cur.name}: pot became heads-up this street; solved from here")
        if not isinstance(node, Decision):
            return {"error": "no decision pending at this point in the tree", "notes": notes}
        if node.player != hero_p:
            notes.append("warning: tree expected the other player to act; reporting that node")
        if pool:
            # the best response to the pool (their equilibrium with its leaks), on later streets too:
            # each hand takes its highest-EV action
            evs = sol.solver.node_action_br_evs(node, n_r0, n_r1, PoolLock(sol.ctx, 1 - hero_p))
            strat = np.zeros_like(evs)
            strat[np.argmax(evs, axis=0), np.arange(evs.shape[1])] = 1.0
            notes.append(f"exploit mode: {villain} plays like a typical player pool (calls small bets too often, "
                         "folds too often to big turn and river bets, raises and bluffs less); EVs assume you "
                         "keep best-responding on later streets")
        else:
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
            if pool and ci is not None:
                acts_out[-1]["gto_freq"] = round(float(node.avg_strategy0()[a, ci]), 3)
        if ci is None:
            notes.append("hero's hand was not solved; frequencies are uniform")
        best = max(acts_out, key=lambda a: a["freq"])["action"]
        sampled = self._sample(acts_out)
        if widened and ci is not None and not pool:
            # The solve's mix assumes the villain almost never plays this way; against the range they
            # actually showed, go with the EVs.
            ev_best = max(acts_out, key=lambda a: a["ev_bb"])["action"]
            if ev_best != best:
                notes.append(f"recommending {ev_best}, the highest EV against {', '.join(widened)}'s rule-based range; "
                             f"the solve's mix ({best}) assumes they almost never play this way")
            best = sampled = ev_best
        expl = max(0.0, sol.exploitability())
        if pool and expl >= UNSETTLED:
            notes.append(f"the solve is {expl:.1f}% of the pot from equilibrium, so the pool model built on it "
                         "is rough: Accurate mode gives steadier exploits")
        if not pool and not widened and ci is not None and expl >= UNSETTLED:
            # Mixed actions have equal EVs only at equilibrium; short of it the mix lags behind, so
            # a clear EV gap says where the solve is heading.
            top = max(acts_out, key=lambda a: a["ev_bb"])
            mix_ev = next(a["ev_bb"] for a in acts_out if a["action"] == best)
            if top["action"] != best and top["ev_bb"] - mix_ev > EV_GAP / 100 * node.pot_now:
                notes.append(f"recommending {top['action']}, the highest EV: the solve is {expl:.1f}% of the pot "
                             f"from equilibrium, so its mix ({best} most often) hasn't caught up")
                best = sampled = top["action"]
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
            "actions": acts_out, "recommended": best, "sampled": sampled,
            "grid": strategy_grid([sf[a] for a in range(len(node.names))], list(node.names), hero_r),
            "range_action_freq": range_freq,
            "hero_range_combos": round(float(hero_r.sum()), 1),
            "villain_range_combos": round(float(vill_r.sum()), 1),
            "villain_range": range_to_text(vill_r, 0.05)[:400],
            "solver": {"iterations": sol.iters, "solve_seconds": round(sol.seconds, 2), "nodes": sol.n_nodes,
                       "active_combos": sol.ctx.n, "tree_actions": node.names,
                       # how far this street's solve is from equilibrium, % of the street's starting pot
                       "exploitability_pct": round(expl, 2)},
            "widened": widened,
            "notes": notes, "seconds": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------ multiway
    def _multiway(self, hero, hole, board, ranges, remaining, cur: StreetState, stack, t0, notes):
        """3+ players still in: Monte Carlo vs every live range, not a solve. Frequencies are 0/1 (the
        highest EV action); EVs are relative to folding now.

        Opponents follow simple range rules, with hands ranked by average strength over runouts:
        - Facing a bet or raise of size b into pot p, the n players who can continue fold together
          with probability FOLD_SHADE * b / (p + b): a quarter less often than a balanced defence
          (which would let a pure bluff break even), since real players call too much. They all
          keep the hands above one strength bar, so a range of weak hands (a lead into the field
          after checking twice) folds more of itself than a strong one (a call of that lead), and
          the strongest part of what each continues with raises instead.
        - Betting into us: value with the top half of what would call that bet (balanced, unshaded), plus bluffs from
          the bottom of the range at the balanced ratio, bluffs = value * b / (p + b).
        - Once someone bets or raises, the players not involved fold.
        We call or fold by our equity against the aggressor's range on the board as it stands (no
        peeking at later cards). Every line plays on to the river with one 66%-pot bet per later
        street, and we never bet or raise after this decision."""
        from .evaluator import eval_board
        rng = random.Random(1)
        opps = [p for p in remaining if p != hero]
        n_opp = len(opps)
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
        seq = cur.order[cur.order.index(hero) + 1:] + cur.order[:cur.order.index(hero)]
        after = sorted(range(n_opp), key=lambda i: seq.index(opps[i]))   # opponents in acting order after us
        cap = [stack - cur.inv_start[p] for p in opps]                   # most each opponent can have in this street
        accurate = self.time_budget >= 20

        # Runouts, shared by the strength rankings and the trials: from the flop every turn with a few
        # rivers each, from the turn every river.
        runs = []
        if len(board) == 5:
            runs = [(tuple(board), eval_board(board))]
        elif len(board) == 4:
            runs = [(tuple(board) + (c,), eval_board(list(board) + [c])) for c in avail]
        else:
            for t in avail:
                for c in rng.sample([c for c in avail if c != t], 8 if accurate else 4):
                    runs.append((tuple(board) + (t, c), eval_board(list(board) + [t, c])))

        def strength(sts):
            """(average strength, combos strongest first, the samples) over these runouts."""
            tot = np.zeros(N_COMBOS); cnt = np.zeros(N_COMBOS)
            for s in sts:
                ok = s >= 0
                tot[ok] += s[ok]; cnt[ok] += 1
            sc = np.where(cnt > 0, tot / np.maximum(cnt, 1), -1.0)
            return sc, np.argsort(-sc), sts

        now = strength([s for _, s in runs])
        turns = {}
        if len(board) == 3:
            by_turn = {}
            for run, s in runs:
                by_turn.setdefault(run[3], []).append(s)
            turns = {t: strength(sts) for t, sts in by_turn.items()}
        rivers = {}
        def info(run, n, st):
            if n == len(board):
                return now
            if n == 4:
                return turns[run[3]]
            if run not in rivers:
                rivers[run] = (st, np.argsort(-st), [st])
            return rivers[run]

        top, aggro, keep = _top, _aggro, _keep

        def equity(w, sts):
            num = den = 0.0
            for s in sts:
                ww = w * (s >= 0)
                num += ww[s < s[idx]].sum() + 0.5 * ww[s == s[idx]].sum()
                den += ww.sum()
            return num / den if den > 0 else 0.5

        def calls(w, sts, cost, p):
            """Do we call `cost` more into a pot of p (their chips included) against range w?"""
            return equity(w, sts) * (p + cost) > cost

        ranks = {}
        def calls_turn(w, sts, b, p, left):
            """Turn version of calls(): average over the river samples of calling b, then facing
            their river bet (value + balanced bluffs from what they bet the turn with) and calling
            or folding it, so a call that a river bet will push us off isn't counted at full equity."""
            P = p + b
            b2 = min(round(0.66 * P * 2) / 2, left - b)
            tot = 0.0
            for s in sts:
                ws = w * (s >= 0)
                if ws.sum() <= 0:
                    continue
                win = (s < s[idx]) + 0.5 * (s == s[idx])
                if b2 <= 1e-9:
                    tot += (ws * win).sum() / ws.sum() * P
                    continue
                if id(s) not in ranks:
                    ranks[id(s)] = np.argsort(-s)
                m = aggro(ws, keep(b2, P, 1, 1.0) * VALUE_SHARE, b2, P, ranks[id(s)])
                wm = ws * m
                ev_bet = (wm * win).sum() * (P + 2 * b2) - wm.sum() * b2
                tot += (max(ev_bet, 0.0) + (ws * ~m * win).sum() * P) / ws.sum()
            return tot / max(len(sts), 1) > b

        cache = {}

        def play_out(run, st, hands, live, p, put, matched):
            """Our result (relative to folding now) once this street's betting is over. live:
            [(opponent, their range now)]; p the pot; put what we've added this street; matched the
            bet everyone left has matched."""
            alive = [i for i, _ in live]
            rs = dict(live)
            left = {"hero": hero_rem - put, **{i: cap[i] - matched for i in alive}}
            for n in range(len(board) + 1, 6):
                sc, rank, sts = info(run, n, st)
                b = min(round(0.66 * p * 2) / 2, left["hero"], *(left[i] for i in alive))
                if b <= 1e-9:
                    continue
                # Decisions are cached on the turn (key: turn card and the ranges going in, which are
                # arrays that live for the whole call); river ones are cheap and run-specific.
                def cached(key, fn):
                    if key is None:
                        return fn()
                    if key not in cache:
                        cache[key] = fn()
                    return cache[key]
                ids = tuple(id(rs[i]) for i in alive)
                tkey = (run[3], tuple(alive), ids, round(p, 2)) if n == 4 else None
                # We lead for value when we're ahead of every range that would call it.
                def lead_fn():
                    ws = [rs[i] * (sc >= 0) for i in alive]
                    cs = dict(zip(alive, _joint_top(ws, _fold_together([b] * len(alive), p), rank)))
                    return all(equity(rs[i] * (sc >= 0) * cs[i], sts) > LEAD_EQ for i in alive), cs
                lead, cs = cached(tkey and ("lead",) + tkey, lead_fn)
                if lead:
                    callers = [i for i in alive if cs[i][hands[i]]]
                    if not callers:
                        return p - put
                    p += b * (1 + len(callers)); put += b; left["hero"] -= b
                    for i in callers:
                        left[i] -= b; rs[i] = rs[i] * (sc >= 0) * cs[i]
                    alive = callers
                    continue
                v = keep(b, p, len(alive), 1.0) * VALUE_SHARE
                for i in alive:
                    r = rs[i] * (sc >= 0)          # drop combos holding a card dealt since
                    key = tkey and (i,) + tkey
                    m = cached(key and ("bet",) + key, lambda: aggro(r, v, b, p, rank))
                    if m[hands[i]]:
                        w = r * m
                        # Raise to b + 75% of the pot after calling when we're ahead of what continues,
                        # else call or fold.
                        rto = min(round((b + 0.75 * (p + 2 * b)) * 2) / 2, left["hero"], left[i])
                        def raise_fn():
                            if rto <= b + 1e-9:
                                return False, None
                            c = top(w, keep(rto - b, p + 2 * b, 1), rank)
                            return equity(w * c, sts) > RAISE_EQ, c
                        ok, c = cached(key and ("raise",) + key, raise_fn)
                        if ok:
                            if not c[hands[i]]:
                                return p + b - put
                            p += 2 * rto; put += rto; left["hero"] -= rto; left[i] -= rto
                            alive = [i]; rs[i] = w * c
                            break
                        we_call = cached(key and ("call",) + key,
                                         lambda: calls_turn(w, sts, b, p + b, min(left["hero"], left[i])) if n == 4
                                         else calls(w, sts, b, p + b))
                        if not we_call:
                            return -put
                        p += 2 * b; put += b; left["hero"] -= b; left[i] -= b
                        alive = [i]; rs[i] = w
                        break
            return _showdown_share(st, idx, [hands[i] for i in alive]) * p - put

        sc0, rank0, sts0 = now
        trials = 0
        share_all = 0.0

        # Calling: players yet to act behind us fold, call or raise.
        if to_call > 1e-9:
            p_call = pot + call_amt
            behind = [i for i in after if cur.bets[opps[i]] < cur.cur_bet - 1e-9 and opps[i] not in cur.allin]
            matched = [i for i in range(n_opp) if i not in behind]
            beh = {}
            costs = [min(cur.cur_bet, cap[i]) - cur.bets[opps[i]] for i in behind]
            beh_conts = _joint_top([opp_ranges[i] for i in behind], _fold_together(costs, p_call), rank0)
            for i, cost, cont in zip(behind, costs, beh_conts):
                k = _share(opp_ranges[i], cont)
                rto = min(round((cur.cur_bet + 0.75 * (p_call + cost)) * 2) / 2, cap[i])
                hero_to = min(rto, cur.bets[hero] + hero_rem)
                extra = hero_to - cur.cur_bet
                raise_m = aggro(opp_ranges[i], k * RAISE_SHARE, rto - cur.cur_bet, p_call + cost, rank0) if extra > 1e-9 else None
                p_r = p_call + rto - cur.bets[opps[i]]
                beh[i] = (cost, cont, raise_m, rto, hero_to, extra, p_r,
                          raise_m is not None and calls(opp_ranges[i] * raise_m, sts0, extra, p_r))
            cont_r = {i: opp_ranges[i] * beh[i][1] for i in behind}
            raise_r = {i: opp_ranges[i] * beh[i][2] for i in behind if beh[i][2] is not None}

        # Betting or raising: to 75% pot after calling, or 66% pot, to the nearest 0.5bb.
        if to_call > 1e-9:
            agg_to = min(round((cur.cur_bet + 0.75 * (pot + call_amt)) * 2) / 2, cur.bets[hero] + hero_rem)
            agg_name = f"raise to {agg_to:g}"
        else:
            agg_to = min(round(0.66 * pot * 2) / 2, hero_rem)
            agg_name = "bet 66%"
        hero_add = agg_to - cur.bets[hero] if agg_to > cur.cur_bet + 1e-9 else 0.0   # no raise once a call is all-in
        if hero_add > 1e-9:
            opp_add = [max(0.0, min(agg_to, cap[i]) - cur.bets[opps[i]]) for i in range(n_opp)]
            # Players already all-in can't fold; the rest share one strength bar (see _joint_top).
            can_fold = [i for i in range(n_opp) if opp_add[i] > 1e-9]
            conts = [np.ones(N_COMBOS, dtype=bool) for _ in range(n_opp)]
            if can_fold:
                fold = _fold_together([hero_add] * n_opp, pot)
                for i, m in zip(can_fold, _joint_top([opp_ranges[i] for i in can_fold], fold, rank0)):
                    conts[i] = m
            k_agg = [_share(opp_ranges[i], conts[i]) for i in range(n_opp)]
            agg_cont_r = [opp_ranges[i] * conts[i] for i in range(n_opp)]
            rr = {}
            for i in range(n_opp):
                rto = min(round((agg_to + 0.75 * (pot + hero_add + opp_add[i])) * 2) / 2, cap[i])
                hero_to = min(rto, cur.bets[hero] + hero_rem)
                extra = hero_to - agg_to
                if extra > 1e-9:
                    m = aggro(opp_ranges[i], k_agg[i] * RAISE_SHARE, rto - agg_to, pot + hero_add + opp_add[i], rank0)
                    p_r = pot + hero_add + rto - cur.bets[opps[i]]
                    rr[i] = (m, opp_ranges[i] * m, hero_to, extra, p_r,
                             calls(opp_ranges[i] * m, sts0, extra, p_r))

        # Checking: an opponent behind may bet; we pick call or fold per bettor from the trials.
        bets = {}
        if to_call <= 1e-9:
            for i in after:
                if opps[i] in cur.acted or opps[i] in cur.allin:
                    continue
                b = min(round(0.66 * pot * 2) / 2, cap[i] - cur.bets[opps[i]], hero_rem)
                if b > 1e-9:
                    m = aggro(opp_ranges[i], keep(b, pot, n_opp, 1.0) * VALUE_SHARE, b, pot, rank0)
                    bets[i] = (b, m, opp_ranges[i] * m)

        cums = [np.cumsum(r) for r in opp_ranges]
        # Trials reuse the runouts, each pass with fresh opponent hands, and every action is scored on
        # the same trials. Stop once the gap between the two best actions is clear of the noise.
        min_trials, max_seconds = (1200, 5.0) if accurate else (400, 1.5)
        everyone = list(enumerate(opp_ranges))
        vals = {"call": [], "agg": [], "chk": []}   # per trial; "chk" holds (bettor or -1, value)
        t_mc = time.time()

        def summary():
            """{action: per-trial values} with each check-line bettor called only if that pays."""
            out = {}
            if to_call > 1e-9:
                out["fold"] = np.zeros(len(vals["call"])); out["call"] = np.array(vals["call"])
            else:
                tot = {}
                for i, v in vals["chk"]:
                    tot[i] = tot.get(i, 0.0) + v
                out["check"] = np.array([v if i < 0 or tot[i] > 0 else 0.0 for i, v in vals["chk"]])
            if hero_add > 1e-9:
                out["agg"] = np.array(vals["agg"])
            return out

        def gap():
            """(gap between the two best mean EVs, its standard error from paired trials)."""
            out = summary()
            if len(out) < 2 or trials < 2:
                return 0.0, 0.0
            a, b = sorted(out.values(), key=lambda x: -x.mean())[:2]
            d = a - b
            return float(d.mean()), float(d.std() / np.sqrt(len(d)))

        while True:
            if self.should_stop is not None and self.should_stop():
                break
            for run, st in runs:
                used = set(run) | set(hole)
                hands = []
                for cum in cums:
                    for _ in range(50):
                        j = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
                        j = min(j, N_COMBOS - 1)
                        if not (set(ALL_COMBOS[j]) & used):
                            break
                    else:
                        break
                    used |= set(ALL_COMBOS[j]); hands.append(j)
                if len(hands) < n_opp:
                    continue
                trials += 1
                share_all += _showdown_share(st, idx, hands)
                if to_call > 1e-9:
                    p, live, res = p_call, [(i, opp_ranges[i]) for i in matched], None
                    for i in behind:
                        cost, cont, raise_m, rto, hero_to, extra, p_r, we_call = beh[i]
                        if raise_m is not None and raise_m[hands[i]]:
                            res = (play_out(run, st, hands, [(i, raise_r[i])], p_r + extra, call_amt + extra, hero_to)
                                   if we_call else -call_amt)
                            break
                        if cont[hands[i]]:
                            p += cost; live.append((i, cont_r[i]))
                    vals["call"].append(res if res is not None else play_out(run, st, hands, live, p, call_amt, cur.cur_bet))
                else:
                    first = next((i for i in bets if bets[i][1][hands[i]]), None)
                    if first is None:
                        vals["chk"].append((-1, play_out(run, st, hands, everyone, pot, 0.0, cur.cur_bet)))
                    else:
                        b, _, r_b = bets[first]
                        vals["chk"].append((first, play_out(run, st, hands, [(first, r_b)], pot + 2 * b, b, b)))
                if hero_add > 1e-9:
                    res = None
                    for i in after:
                        if i in rr and rr[i][0][hands[i]]:
                            _, r_rr, hero_to, extra, p_r, we_call = rr[i]
                            res = (play_out(run, st, hands, [(i, r_rr)], p_r + extra, hero_add + extra, hero_to)
                                   if we_call else -hero_add)
                            break
                    if res is None:
                        callers = [i for i in range(n_opp) if conts[i][hands[i]]]
                        if not callers:
                            res = pot
                        else:
                            final = pot + hero_add + sum(opp_add[i] for i in callers)
                            res = play_out(run, st, hands, [(i, agg_cont_r[i]) for i in callers], final, hero_add, agg_to)
                    vals["agg"].append(res)
            if trials >= min_trials:
                d, se = gap()
                if se < max(0.25, abs(d) / 3) or time.time() - t_mc > max_seconds or trials >= 20000:
                    break
            elif time.time() - t_mc > 3 * max_seconds:
                break
        trials = max(trials, 1)
        eq = float(share_all / trials)
        ev = {k: float(v.mean()) if len(v) else 0.0 for k, v in summary().items()}
        d, se = gap()
        acts = []
        if to_call > 1e-9:
            need = call_amt / (pot + call_amt)
            acts.append({"action": "fold", "freq": 0.0, "ev_bb": 0.0})
            acts.append({"action": "call", "freq": 0.0, "ev_bb": round(ev["call"], 2), "amount_bb": round(call_amt, 2)})
            notes.append(f"pot odds need {need*100:.0f}% equity, you have {eq*100:.0f}% against everyone's whole range")
            if behind:
                notes.append(f"after a call, {', '.join(opps[i] for i in behind)} still to act")
        else:
            acts.append({"action": "check", "freq": 0.0, "ev_bb": round(ev["check"], 2)})
            if bets:
                tot = {}
                for i, v in vals["chk"]:
                    if i >= 0:
                        tot[i] = tot.get(i, 0.0) + v
                p_bet = sum(1 for i, _ in vals["chk"] if i >= 0) / trials
                called = [opps[i] for i, t in tot.items() if t > 0]
                notes.append(f"if you check, someone bets {p_bet*100:.0f}% of the time"
                             + (f"; you'd call {', '.join(called)}" if called else "; you'd fold to it"))
        if hero_add > 1e-9:
            acts.append({"action": agg_name, "freq": 0.0, "ev_bb": round(ev["agg"], 2), "amount_bb": round(agg_to, 2)})
            notes.append(f"vs {agg_name}: " + ", ".join(f"{opps[i]} continues with {k_agg[i]*100:.0f}%"
                                                         for i in range(n_opp)) + " of their range")
        pick = max(range(len(acts)), key=lambda i: acts[i]["ev_bb"])
        acts[pick]["freq"] = 1.0
        notes.append(f"{trials} trials; the best action leads the next by {d:.2f} ± {se:.2f} bb")
        return {"street": cur.name, "hero": hero, "hand": hand_class(hole), "board": cards_str(board),
                "players": len(remaining), "multiway": True, "opponents": opps,
                "equity_vs_field": round(eq, 3), "pot_bb": round(pot, 2), "to_call_bb": round(call_amt, 2),
                "actions": acts, "recommended": acts[pick]["action"], "sampled": acts[pick]["action"],
                "notes": notes + [f"{len(remaining)} players in the pot: the solver is heads-up only, this is a Monte Carlo heuristic"],
                "seconds": round(time.time() - t0, 2)}

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _with_hero(r0, r1, hero_p, hole, notes):
        """Our hand gets a negligible weight when the ranges say we'd never be here, so the solve
        still gives it a strategy and EVs (against the opponent's range) without changing the rest."""
        i = COMBO_INDEX[(min(hole), max(hole))]
        r = r0 if hero_p == 0 else r1
        if r[i] > 1e-9:
            return r0, r1
        r = r.copy(); r[i] = 1e-3
        notes.append("your hand isn't in your range for this line, so it was solved as a near-zero-weight combo")
        return (r, r1) if hero_p == 0 else (r0, r)

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
        uses the real action instead of snapping it to the abstraction. A size already taken
        this street is never swapped out by a later one, an all-in deeper than the tree's
        all-in SPR gets a size that reaches the stack, and the raise cap covers the raises made."""
        cfg = self.cfg.copy()
        bets = [0.0, 0.0]
        pot = st.pot_start
        fixed = {"bet": set(), "raise": set()}
        n_aggr = 0
        for a in st.actions:
            if a["pos"] not in (oop, ip):
                continue
            p = 0 if a["pos"] == oop else 1
            act = a["action"]
            pot_total = pot + bets[0] + bets[1]
            amt = a.get("amount") or a.get("to") if act == "bet" else a.get("to") or a.get("amount")
            if act in ("bet", "raise") and amt and (bets[p] + amt if act == "bet" else amt) >= stack - 1e-9:
                act = "all-in"   # a bet or raise of the whole stack
            if act == "bet" and amt:
                _swap_nearest(cfg.bet_sizes[st.name], amt / max(pot_total, 1e-9), fixed=fixed["bet"])
                bets[p] = amt
            elif act == "raise" and amt:
                call_amt = bets[1 - p] - bets[p]
                _swap_nearest(cfg.raise_sizes[st.name], (amt - bets[1 - p]) / max(pot_total + call_amt, 1e-9),
                              fixed=fixed["raise"])
                bets[p] = amt
            elif act == "call":
                bets[p] = min(bets[1 - p], stack)
            elif act == "all-in":
                # the tree offers all-in only at low SPR; a deeper jam would map to the biggest bet,
                # so swap in a size that reaches the stack (half a bb over, past the tree's rounding)
                if bets[1 - p] <= bets[p]:
                    _swap_nearest(cfg.bet_sizes[st.name], (stack - bets[p] + 0.5) / max(pot_total, 1e-9),
                                  fixed=fixed["bet"])
                else:
                    call_amt = bets[1 - p] - bets[p]
                    _swap_nearest(cfg.raise_sizes[st.name],
                                  (stack - bets[1 - p] + 0.5) / max(pot_total + call_amt, 1e-9), fixed=fixed["raise"])
                bets[p] = stack
            if act in ("bet", "raise", "all-in"):
                n_aggr += 1
        # past the cap the tree offers only all-in, so a 3bet that isn't one would be read as a jam
        cfg.max_raises = max(cfg.max_raises, n_aggr)
        return cfg

    def _walk(self, sol: Solution, acts: List[dict], oop, ip, pot, stack, notes, street, smoothing=0.0, only=None,
              widened=None, board=None, rules=None, lock=None):
        """Follow real actions through the solved tree, narrowing ranges. Smoothing mixes each
        strategy with uniform so a line the solve rarely takes doesn't shrink a range to nothing.
        With `only` (0 = OOP, 1 = IP) and `board`, that player's actions are exact unless the solve
        takes one with less than RARE_LINE of their range: then their range for it comes from the
        multiway betting rules instead (see _rule_mask), and their seat is appended to `widened`.
        `rules` (players, with `board`) gives the same fallback on an earlier street, where the other
        actions stay smoothed: a short prior solve can be far from equilibrium, and a line it rarely
        takes (a turn call it folds two pair to) would leave the range a sliver.
        `lock` (exploit mode) narrows its player's range by how the pool plays instead of the solve."""
        rules = (only,) if rules is None and only is not None else rules or ()
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
            name = self._tree_action_name(node, a, p, bets, pot, stack)
            k = match_action(node.names, name)
            rp = r0 if p == 0 else r1
            if lock is not None and p == lock.player:
                s = lock(node, node.avg_strategy(), rp[None, :])[0][k]
            else:
                s = node.avg_strategy0()[k]
            rare = (rp * s).sum() < RARE_LINE * max(rp.sum(), 1e-12)
            if p in rules and rare and board is not None and a["action"] != "fold":
                if widened is not None and p == only:
                    widened.append(a["pos"])
                notes.append(f"{street}: the solve rarely has {a['pos']} {a['action']} here "
                             f"({(rp * s).sum() / max(rp.sum(), 1e-12) * 100:.0f}% of their range), so their range "
                             "comes from simple betting rules instead")
                sc, rank = _street_rank(tuple(board))
                cur_bet, pot_total = max(bets), pot + bets[0] + bets[1]
                act_ = a["action"]
                amt = a.get("to") if a.get("to") is not None else a.get("amount")
                to = (stack if act_ == "all-in" else float(amt) if act_ in ("bet", "raise") and amt is not None
                      else node.children[k].bets[p] if act_ in ("bet", "raise") and isinstance(node.children[k], Decision)
                      else cur_bet)
                full = expand(sol.ctx, rp) * (sc >= 0)
                m = _rule_mask(full, to, cur_bet, max(0.0, cur_bet - bets[p]), pot_total, 1, rank)
                s = _soften(full, m)[sol.ctx.active]
            elif smoothing > 0 and only is None:
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

    def _tree_action_name(self, node: Decision, a: dict, p: int, bets, pot, stack: float = float("inf")) -> str:
        act = a["action"]
        if act in ("check", "fold", "call", "all-in"):
            return act
        pot_total = pot + bets[0] + bets[1]
        if act == "bet":
            amt = a.get("amount") or a.get("to") or 0.0
            if bets[p] + amt >= stack - 1e-9:
                return "all-in"
            return f"bet {amt / max(pot_total, 1e-9) * 100:.0f}%"
        to = a.get("to") or a.get("amount") or 0.0
        return "all-in" if to >= stack - 1e-9 else f"raise to {to:g}"

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


def _swap_nearest(sizes: List[float], frac: float, tol: float = 0.15, fixed: Optional[set] = None) -> None:
    """Replace the entry of `sizes` closest to `frac` with `frac` if it is more than `tol`
    (relative) away; otherwise keep the abstraction (keeps solves cache-friendly). Entries in
    `fixed` (indices of sizes taken earlier this street) are never replaced: with all of them
    taken, `frac` is added. The entry used is added to `fixed`."""
    fixed = set() if fixed is None else fixed
    if not sizes or frac <= 0:
        return
    near = min(range(len(sizes)), key=lambda k: abs(sizes[k] - frac))
    if abs(sizes[near] - frac) / max(sizes[near], 1e-9) <= tol or frac in sizes:
        fixed.add(sizes.index(frac) if frac in sizes else near)
        return
    free = [k for k in range(len(sizes)) if k not in fixed]
    if free:
        i = min(free, key=lambda k: abs(sizes[k] - frac))
        sizes[i] = round(frac, 3)
    else:
        sizes.append(round(frac, 3))
        i = len(sizes) - 1
    fixed.add(i)


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
