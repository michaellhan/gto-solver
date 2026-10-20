"""Sanity sweep: run many common spots through the advisor and flag outputs with the signatures of bugs
found before, so the next one shows up here rather than at the table.

Setups (SETUPS): 6-max 100bb carries most of the weight; also 20bb, 40bb and 200bb stacks, heads-up,
3-handed, 9-handed with an ante, and 6-max with a straddle (their preflop trees pruned to at most three
voluntary players).

Spots, per setup:
  preflop:  decisions in the charts' preflop lines (no open-limps, at most two callers per raise), drawn
            by how often each line happens; hero holding AA, KK, QQ, AKs and two random hands; some in
            exploit mode (opponents play like a player pool)
  postflop: lines that reach the flop heads-up (single-raised, 3bet, 4bet pots), random boards, hero's
            hand drawn from its own range (sometimes off it): flop first to act, facing a check or a bet,
            turn and river after checks, river facing a bet; 3-way pots (the multiway heuristic); some
            spots in exploit mode

Flags:
  error            the advisor returned an error or raised
  nan              an EV or frequency is NaN or infinite
  premium_fold     preflop: AA told to fold; KK below a 5bet; QQ/AKs to a single raise
  thin_range       a preflop range under preflop.MIN_LINE_COMBOS (8) that was not filled out (the solve
                   would play nearly face-up)
  face_up          postflop: a range under 3 combos before its player acts on the street
  ev_is_pot        postflop: a bet or all-in whose EV is exactly the pot (the villain always folds:
                   the signature of a face-up solve)
  strong_fold      postflop: a set or better (on an unpaired board) told to fold to a bet
  rec_ev_gap       the recommended action's EV trails the best by more than max(1bb, 10% of the pot)
  filled_range     (info) a preflop line left under MIN_LINE_COMBOS and was filled out: the safeguard
  unsettled        (info) postflop: the solve is more than 10% of the pot from equilibrium. Expected on
                   big flop ranges at short budgets: the site plays a smaller game in the browser, shows a
                   convergence badge and offers Accurate mode

Postflop flags are confirmed: each flagged spot is solved again with a longer budget (--confirm) on half
the cores, and only flags that reproduce are reported (short solves sharing the machine can be far from
equilibrium, which looks like a face-up range or a bet worth exactly the pot).

Quick mode (--quick) is what scripts/deploy.sh runs before every deploy (~3 min on 10 cores): a fixed
set, deterministic for --seed (quick_jobs), of 10 preflop decision nodes per setup (5 hands each; 6 more in
exploit mode in the default setup) and 10 postflop spots per setup rotating through the heads-up kinds,
with the default setup getting each heads-up kind 4 times, 3-way pots and exploit-mode spots. Its preflop
trees keep lines that happen at least 1e-4 of the time with at most three voluntary players (the full
6-max tree takes over a minute to walk), so on top it checks the rare lines (RARE_LINES, rare_jobs): ~30
hand-picked lines over the setups where earlier bugs hid (3bet and 4bet pots, cold 4bets, flat then 4bet,
a squeeze called twice or 4bet, 4bet and 5bet jams faced, limp-reraises, straddled 3bet pots), built
directly from their steps (build_line) rather than by walking the tree. Per line: its last two decision
nodes and the decision it ends at (5 hands each), the ranges a closed line reaches the flop with
(thin_range / filled_range), and in its pot a flop spot facing a bet and a turn spot heads-up (face_up,
ev_is_pot) or both 3-way kinds. The confirm pass re-solves only spots with a flag other than the info
ones, and the run exits 1 when any such flag remains (--strict does the same for other runs).

  python scripts/sanity_sweep.py                       # full sweep (~40 min on 8 cores)
  python scripts/sanity_sweep.py --pre 300 --post 200  # quicker
  python scripts/sanity_sweep.py --quick               # the pre-deploy check (~3 min, exit 1 on bug flags)
  python scripts/sanity_sweep.py --out report.md       # where to write the report (default sanity_report.md)
  python scripts/sanity_sweep.py --rare                # a full sweep plus the rare lines
  python scripts/sanity_sweep.py --full-report         # the same, report in reports/sanity_YYYY-MM-DD.md
                                                       # (for an occasional or nightly run; not scheduled)
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

# (tag, players, stack, ante per player, straddle, share of the spots)
SETUPS = [
    ("6p 100bb", 6, 100.0, 0.0, False, 0.55),
    ("6p 20bb", 6, 20.0, 0.0, False, 0.06),
    ("6p 40bb", 6, 40.0, 0.0, False, 0.06),
    ("6p 200bb", 6, 200.0, 0.0, False, 0.06),
    ("HU 100bb", 2, 100.0, 0.0, False, 0.06),
    ("3p 100bb", 3, 100.0, 0.0, False, 0.05),
    ("9p 100bb ante", 9, 100.0, 0.12, False, 0.10),
    ("6p straddle", 6, 100.0, 0.0, True, 0.06),
]
POOL_SHARE = 0.2      # spots in exploit mode (default setup only)


def _table(setup):
    from gto import preflop as pf
    _, n, _, ante, straddle, _ = setup
    return pf.Table(pf.SEATS[n], ante, straddle)


def _size(st, p, nr, T, stack):
    """Default raise-to for p at a node with nr raises in (open 2.5x / SB 3x the big bet, 3bet 3x in
    position / 4x out of position plus one raise per caller, 4bet 2.3x / 2.5x, then all-in)."""
    from gto import preflop as pf
    cur = st.cur_bet
    if nr == 0:
        return round(cur * (3.0 if p == "SB" else 2.5), 1)
    post = pf.postflop_order(list(T))
    others = [q for q in st.active if q != p and st.bets[q] >= cur - 1e-9]
    ip = all(post.index(p) > post.index(q) for q in others)
    if nr == 1:
        callers = sum(1 for q in st.active if q != p and 0 < st.bets[q] >= cur - 1e-9) - 1
        return round(cur * (3.0 if ip else 4.0) + max(callers, 0) * cur, 1)
    if nr == 2:
        return round(cur * (2.3 if ip else 2.5), 1)
    return stack


def preflop_lines(setup, max_in=None, min_reach=0.0):
    """(decision nodes, closed lines): every node of the setup's charts preflop tree as (actions, player
    to act, reach), and the lines that close with two or three players in and nobody all-in as (actions,
    reach); reach is how often the line happens (the product of each player's share of their range
    taking their action). `max_in` caps how many players put chips in voluntarily; `min_reach` drops
    lines rarer than that (quick mode: the full 6-max tree takes over a minute to walk)."""
    from gto import preflop as pf
    from gto.advisor import replay_hand
    T = _table(setup)
    stack = setup[2]
    nodes, closed = [], []

    def dfs(acts, ranges, reach):
        st = replay_hand(T, stack, {"preflop": acts}, 1)[0]
        if st.closed or st.to_act is None:
            if len(st.active) in (2, 3) and not st.allin:
                closed.append((list(acts), reach))
            return
        p = st.to_act
        nodes.append((list(acts), p, reach))
        strat = pf.strategy_for(p, acts, T)
        nr = st.n_raises
        since = 0
        for h in reversed(acts):
            if h["action"] in ("raise", "all-in"):
                break
            since += h["action"] == "call"
        voluntary = {a["pos"] for a in acts if a["action"] in ("call", "raise", "all-in")}
        for key, v in strat.items():
            if v.sum() < 1e-9:
                continue
            if key in ("call", "check") and nr == 0 and p not in ("SB", "BB") and not T.straddle:
                continue                      # no open-limps
            if key == "call" and nr > 0 and since >= 2:
                continue                      # at most two callers per raise
            if max_in and key != "fold" and p not in voluntary and len(voluntary) >= max_in:
                continue
            if key == "fold":
                a = {"pos": p, "action": "fold"}
            elif key in ("call", "check"):
                a = {"pos": p, "action": key}
            else:
                to = _size(st, p, nr, T, stack)
                a = ({"pos": p, "action": "all-in"} if to >= stack - st.inv_start[p] - 1e-9 or nr >= 3
                     else {"pos": p, "action": "raise", "to": to})
            w = ranges.get(p, np.ones_like(v))
            share = float((w * v).sum() / max(w.sum(), 1e-12))
            if reach * share < min_reach:
                continue
            dfs(acts + [a], dict(ranges, **{p: w * v}), reach * share)

    dfs([], {}, 1.0)
    return nodes, closed


def weighted_sample(items, weights, n, rng, power=0.5):
    """n distinct items, drawn with probability proportional to weight**power (softened, so rare lines
    still appear)."""
    if not items or n <= 0:
        return []
    w = np.array(weights, float) ** power
    k = min(n, int((w > 0).sum()))            # a small tree can have fewer reachable nodes than asked for
    if k == 0:
        return []
    idx = np.random.default_rng(rng.randrange(1 << 30)).choice(len(items), size=k, replace=False, p=w / w.sum())
    return [items[i] for i in idx]


def describe(acts):
    return ", ".join(f"{a['pos']} {a['action']}" + (f" {a['to']:g}" if "to" in a else "") +
                     (f" {a['amount']:g}" if "amount" in a else "") for a in acts if a["action"] != "fold")


def _finite(res):
    for a in res.get("actions", []):
        for k in ("freq", "ev_bb"):
            v = a.get(k)
            if v is not None and not math.isfinite(v):
                return False
    return True


def _base_state(setup, hero, hole, board, preflop):
    tag, n, stack, ante, straddle, _ = setup
    st = {"players": n, "hero": hero, "hole": hole, "board": board, "stack": stack, "preflop": preflop}
    if ante:
        st["ante"] = ante
    if straddle:
        st["straddle"] = True
    return st


def _ev_gap(res, info, flags):
    evs = [a for a in res.get("actions", []) if a.get("ev_bb") is not None]
    rec = res.get("recommended")
    if evs and rec:
        best = max(a["ev_bb"] for a in evs)
        rec_ev = next((a["ev_bb"] for a in evs if a["action"] == rec), None)
        if rec_ev is not None and best - rec_ev > max(1.0, 0.1 * res.get("pot_bb", 0)):
            flags.append(("rec_ev_gap", info))


def check_preflop(job):
    """Flags for one preflop decision node."""
    setup, acts, hero, hands, pool = job
    from gto.advisor import Advisor, preflop_ranges, hand_class
    from gto.cards import cards
    from gto.preflop import MIN_LINE_COMBOS
    tag = setup[0] + (" pool" if pool else "")
    T = _table(setup)
    adv = Advisor(time_budget=1.0, prior_budget=0.5)
    flags = []
    n_raises = sum(a["action"] in ("raise", "all-in") for a in acts)
    try:
        rng = preflop_ranges(acts, T, setup[2])
        for p in getattr(rng, "rebuilt", []):
            flags.append(("filled_range", {"tag": tag, "line": describe(acts), "player": p}))
        for p, r in rng.items():
            if r.sum() < MIN_LINE_COMBOS and p not in getattr(rng, "rebuilt", []):
                flags.append(("thin_range", {"tag": tag, "line": describe(acts), "player": p,
                                             "combos": round(float(r.sum()), 1)}))
    except Exception as e:   # noqa: BLE001 - a sweep reports everything
        flags.append(("error", {"tag": tag, "line": describe(acts), "where": "preflop_ranges", "error": repr(e)}))
    for hole in hands:
        state = _base_state(setup, hero, hole, "", acts)
        if pool:
            state["opponents"] = "pool"
        try:
            res = adv.advise(state)
        except Exception as e:   # noqa: BLE001
            flags.append(("error", {"tag": tag, "state": state, "error": repr(e)}))
            continue
        if "error" in res:
            flags.append(("error", {"tag": tag, "state": state, "error": res["error"]}))
            continue
        info = {"tag": tag, "state": state, "actions": res["actions"]}
        if not _finite(res):
            flags.append(("nan", info))
        hc = hand_class(tuple(cards(hole)))
        if res.get("recommended") == "fold" and (
                hc == "AA" or (hc == "KK" and n_raises <= 3) or (hc in ("QQ", "AKs") and n_raises <= 1)):
            flags.append(("premium_fold", info))
        _ev_gap(res, info, flags)
    return flags


def postflop_jobs(setup, closed, n, seed, multiway=False, pool_share=0.0, kinds=None):
    """n postflop spots of this setup from its closed lines (heads-up, or 3-way with `multiway`):
    (setup, state, kind, pool). `kinds` fixes the spot kinds, cycled in order (default: drawn at random,
    facing a flop bet twice as often)."""
    from gto.advisor import preflop_ranges, replay_hand
    from gto.cards import ALL_COMBOS, card_str
    from gto import preflop as pf
    rng = random.Random(seed)
    T = _table(setup)
    stack = setup[2]
    post = pf.postflop_order(list(T))
    jobs = []
    want = 3 if multiway else 2
    by_type = defaultdict(list)
    for l, reach in closed:
        st = replay_hand(T, stack, {"preflop": l}, 1)[0]
        if len(st.active) == want:
            by_type[min(sum(a["action"] in ("raise", "all-in") for a in l), 3)].append((l, reach))
    pools = [by_type[k] for k in sorted(by_type) if by_type[k]]
    if not pools:
        return []
    draw = (["first", "vs_bet"] if multiway else
            ["first", "vs_check", "vs_bet", "vs_bet", "turn", "river", "river_vs_bet"])
    while len(jobs) < n:
        pool_ = rng.choice(pools)
        l = rng.choices([x[0] for x in pool_], weights=[x[1] ** 0.5 for x in pool_])[0]
        st = replay_hand(T, stack, {"preflop": l}, 1)[0]
        players = sorted(st.active, key=post.index)
        ranges = preflop_ranges(l, T, stack)
        kind = kinds[len(jobs) % len(kinds)] if kinds else rng.choice(draw)
        oop, ip = players[0], players[-1]
        n_board = {"turn": 4, "river": 5, "river_vs_bet": 5}.get(kind, 3)
        board = rng.sample(range(52), n_board)
        pot = st.pot_start + sum(st.bets.values())
        flop, turn, river = [], [], []
        if multiway:
            if kind == "first":
                hero = players[0]
            else:
                hero = players[1]
                flop = [{"pos": players[0], "action": "bet", "amount": round(pot * rng.choice([0.33, 0.75]), 1)}]
        elif kind == "first":
            hero = oop
        elif kind == "vs_check":
            hero, flop = ip, [{"pos": oop, "action": "check"}]
        elif kind == "vs_bet":
            hero = rng.choice([oop, ip])
            amt = round(pot * rng.choice([0.33, 0.75]), 1)
            flop = ([{"pos": oop, "action": "bet", "amount": amt}] if hero == ip else
                    [{"pos": oop, "action": "check"}, {"pos": ip, "action": "bet", "amount": amt}])
        else:
            checks = lambda: [{"pos": oop, "action": "check"}, {"pos": ip, "action": "check"}]
            flop = checks()
            if kind == "turn":
                hero = oop
            else:
                turn = checks()
                if kind == "river":
                    hero = oop
                else:
                    hero = rng.choice([oop, ip])
                    amt = round(pot * rng.choice([0.5, 1.0]), 1)
                    river = ([{"pos": oop, "action": "bet", "amount": amt}] if hero == ip else
                             [{"pos": oop, "action": "check"}, {"pos": ip, "action": "bet", "amount": amt}])
        w = ranges[hero].copy()
        for c in board:
            w[[i for i, (a, b) in enumerate(ALL_COMBOS) if a == c or b == c]] = 0.0
        if rng.random() < 0.2 or w.sum() < 1e-9:
            ci = rng.choice([i for i, (a, b) in enumerate(ALL_COMBOS) if a not in board and b not in board])
        else:
            ci = rng.choices(range(len(w)), weights=w)[0]
        hole = "".join(card_str(c) for c in ALL_COMBOS[ci])
        state = _base_state(setup, hero, hole, "".join(card_str(c) for c in board), l)
        state.update({"flop": flop, "turn": turn, "river": river})
        jobs.append((setup, state, kind, rng.random() < pool_share))
    return jobs


def _hand_class_rank(hole, board):
    """treys rank class of hero's best hand (1 straight flush .. 9 high card), and whether the board pairs."""
    from treys import Card, Evaluator
    hand = [Card.new(hole[:2]), Card.new(hole[2:])]
    bd = [Card.new(board[i:i + 2]) for i in range(0, len(board), 2)]
    ev = Evaluator()
    cls = ev.get_rank_class(ev.evaluate(bd, hand))
    ranks = [board[i] for i in range(0, len(board), 2)]
    return cls, len(set(ranks)) < len(ranks)


def check_postflop(job):
    setup, state, kind, pool = job[:4]
    budget = job[4] if len(job) > 4 else 3.0
    from gto.advisor import Advisor
    from gto.solver import Config
    tag = setup[0] + (" pool" if pool else "")
    # 3s per street: the sweep runs a solve per core, so this is about what 1s gets on its own; the
    # confirm pass re-runs flagged spots with more
    # Config()'s flop model (river betting, a turn per rank class) needs ~100 iterations, far more than 3s of a
    # loaded core gives, so the sweep keeps the cheaper one it was tuned on: flags then mean the rules, not budget
    adv = Advisor(Config(river_play=False, turn_classes=False), time_budget=budget, prior_budget=budget / 3)
    st = dict(state, **({"opponents": "pool"} if pool else {}))
    flags = []
    try:
        res = adv.advise(st)
    except Exception as e:   # noqa: BLE001
        return [("error", {"tag": tag, "state": st, "error": repr(e)})]
    if "error" in res:
        return [("error", {"tag": tag, "state": st, "error": res["error"]})]
    if res.get("multiway"):
        tag += " 3-way"
    info = {"tag": tag, "state": st, "actions": res.get("actions"), "notes": res.get("notes")}
    if not _finite(res):
        flags.append(("nan", info))
    if res.get("to_call_bb", 0) > 0 and res.get("recommended") == "fold":
        cls, paired = _hand_class_rank(state["hole"], state["board"])
        if cls <= 6 and not paired:
            flags.append(("strong_fold", info))
    _ev_gap(res, info, flags)
    if res.get("multiway"):
        return flags
    hc, vc = res.get("hero_range_combos"), res.get("villain_range_combos")
    # a range counts only before its player acts this street: acting narrows it legitimately (a bet
    # with 2 combos of a 15-combo range is a real betting range; a rare check shrinks hero's, but
    # hero's EVs on this street don't depend on it)
    street = state.get({3: "flop", 4: "turn", 5: "river"}[len(state["board"]) // 2], [])
    hero_acted = any(a["pos"] == state["hero"] for a in street)
    villain_acted = any(a["pos"] != state["hero"] for a in street)
    if (vc is not None and vc < 3 and not villain_acted) or (hc is not None and hc < 3 and not hero_acted):
        flags.append(("face_up", dict(info, hero_combos=hc, villain_combos=vc)))
    pot = res.get("pot_bb", 0.0)
    evs_ = [a["ev_bb"] for a in res.get("actions", []) if a.get("ev_bb") is not None]
    spread = max(evs_) - min(evs_) if evs_ else 0.0
    check_ev = next((a["ev_bb"] for a in res.get("actions", []) if a["action"] == "check" and a.get("ev_bb") is not None), None)
    # (every option worth the pot, or checking worth it too, is hero beating the villain's whole range: a bet
    # winning exactly the pot is then no sign of a face-up solve; a short solve under load left one option
    # a little below the pot and flagged it)
    nuts = spread <= 0.05 or (check_ev is not None and abs(check_ev - pot) < 0.02)
    for a in (res.get("actions", []) if not nuts else []):
        if (a["action"].startswith("bet") or a["action"].startswith("raise") or a["action"] == "all-in") \
                and a.get("ev_bb") is not None and abs(a["ev_bb"] - pot) < 0.02 and a.get("freq", 0) > 0.5:
            flags.append(("ev_is_pot", info))
            break
    if res.get("solver", {}).get("exploitability_pct", 0) > 10:
        flags.append(("unsettled", dict(info, exploitability=res["solver"]["exploitability_pct"])))
    return flags


INFO = {"filled_range", "unsettled"}
POST_KINDS = ["first", "vs_check", "vs_bet", "turn", "river", "river_vs_bet"]
MULTIWAY_KINDS = ["first", "vs_bet"]

# quick mode (--quick): per setup, preflop decision nodes and postflop spots (with their kinds); the default
# setup also gets 3-way spots and exploit-mode (pool) nodes and spots. Lines rarer than QUICK_MIN_REACH are
# skipped and at most three players put chips in voluntarily, so the preflop trees take seconds to walk.
QUICK_MIN_REACH = 1e-4
QUICK_PRE, QUICK_PRE_POOL = 10, 6           # per setup; pool nodes in the default setup on top
QUICK_POST = 10                             # per non-default setup, kinds rotating through POST_KINDS
QUICK_POOL_KINDS = ["first", "vs_bet", "turn", "river_vs_bet"]
QUICK_REPEAT = 4                            # default setup: each heads-up and 3-way kind this many times, pool kinds twice


def quick_jobs(seed=7):
    """(preflop jobs, postflop jobs) of the quick sweep: a small fixed set (deterministic for a seed) that
    covers every setup, every heads-up postflop spot kind, 3-way pots and exploit mode, preflop and
    postflop."""
    from gto.cards import card_str
    rng = random.Random(seed)
    pre_jobs, post_jobs = [], []
    for si, setup in enumerate(SETUPS):
        default = setup[0] == "6p 100bb"
        nodes, closed = preflop_lines(setup, max_in=3, min_reach=QUICK_MIN_REACH)
        sample = weighted_sample(nodes, [r for _, _, r in nodes], QUICK_PRE + (QUICK_PRE_POOL if default else 0), rng)
        for i, (acts, hero, _) in enumerate(sample):
            a, b = rng.sample(range(52), 2)
            pre_jobs.append((setup, acts, hero, ["AsAd", "KsKd", "QsQd", "AhKh", card_str(a) + card_str(b)],
                             default and i >= QUICK_PRE))
        s = seed + 100 * si
        if default:
            post_jobs += postflop_jobs(setup, closed, len(POST_KINDS) * QUICK_REPEAT, s, kinds=POST_KINDS)
            post_jobs += postflop_jobs(setup, closed, len(MULTIWAY_KINDS) * QUICK_REPEAT, s + 1, multiway=True,
                                       kinds=MULTIWAY_KINDS)
            post_jobs += postflop_jobs(setup, closed, len(QUICK_POOL_KINDS) * 2, s + 2, pool_share=1.0,
                                       kinds=QUICK_POOL_KINDS)
        else:
            k = [POST_KINDS[(si * QUICK_POST + j) % len(POST_KINDS)] for j in range(QUICK_POST)]
            post_jobs += postflop_jobs(setup, closed, QUICK_POST, s, kinds=k)
    rare_pre, rare_post = rare_jobs(seed)
    return pre_jobs + rare_pre, post_jobs + rare_post


# Rare lines: hand-picked preflop lines, rarer than QUICK_MIN_REACH or with more than three voluntary
# players, of the kinds where earlier bugs hid (thin or empty ranges, face-up solves, KK folding to a
# jam). Each is (setup tag, kind, steps); a step is (seat, verb) with verb "r" (raise to the sweep's
# default size, all-in from the 5bet on or when it covers the stack), "j" (all-in), "c" (call; an
# open-limp when nobody raised) or "f" (fold). Seats not named fold when it's their turn. A line ends
# closed (its pot goes to the flop) or at a decision.
RARE_LINES = [
    ("6p 100bb", "3bet pot", [("CO", "r"), ("BTN", "r"), ("CO", "c")]),
    ("6p 100bb", "4bet pot", [("CO", "r"), ("BTN", "r"), ("CO", "r"), ("BTN", "c")]),
    ("6p 100bb", "4bet pot", [("BTN", "r"), ("BB", "r"), ("BTN", "r"), ("BB", "c")]),
    ("6p 100bb", "5bet jam faced", [("CO", "r"), ("BTN", "r"), ("CO", "r"), ("BTN", "j")]),
    ("6p 100bb", "5bet jam faced", [("BTN", "r"), ("BB", "r"), ("BTN", "r"), ("BB", "j")]),
    ("6p 100bb", "cold 4bet", [("UTG", "r"), ("CO", "r"), ("BTN", "r"), ("UTG", "f"), ("CO", "c")]),
    ("6p 100bb", "cold 4bet", [("BTN", "r"), ("SB", "r"), ("BB", "r")]),
    ("6p 100bb", "flat then 4bet", [("CO", "r"), ("BTN", "c"), ("BB", "r"), ("CO", "f"), ("BTN", "r")]),
    ("6p 100bb", "squeeze called", [("CO", "r"), ("BTN", "c"), ("BB", "r"), ("CO", "c"), ("BTN", "c")]),
    ("6p 100bb", "squeeze 4bet", [("CO", "r"), ("BTN", "c"), ("BB", "r"), ("CO", "r")]),
    ("6p 100bb", "limp-reraise", [("HJ", "c"), ("BTN", "r"), ("HJ", "r")]),
    ("HU 100bb", "limp-reraise", [("SB", "c"), ("BB", "r"), ("SB", "r")]),
    ("HU 100bb", "4bet pot", [("SB", "r"), ("BB", "r"), ("SB", "r"), ("BB", "c")]),
    ("HU 100bb", "5bet jam faced", [("SB", "r"), ("BB", "r"), ("SB", "r"), ("BB", "j")]),
    ("6p straddle", "straddled 3bet pot", [("HJ", "r"), ("UTG", "r"), ("HJ", "c")]),
    ("6p straddle", "straddled 3bet pot", [("HJ", "r"), ("BTN", "r"), ("HJ", "c")]),
    ("6p straddle", "4bet pot", [("CO", "r"), ("BTN", "r"), ("CO", "r"), ("BTN", "c")]),
    ("6p straddle", "limp-reraise", [("HJ", "c"), ("UTG", "r"), ("HJ", "r")]),
    ("6p 20bb", "4bet jam faced", [("CO", "r"), ("BTN", "r"), ("CO", "j")]),
    ("6p 40bb", "4bet jam faced", [("CO", "r"), ("BTN", "r"), ("CO", "j")]),
    ("6p 40bb", "flat then 4bet", [("CO", "r"), ("BTN", "c"), ("BB", "r"), ("CO", "f"), ("BTN", "j")]),
    ("6p 200bb", "4bet pot", [("CO", "r"), ("BTN", "r"), ("CO", "r"), ("BTN", "c")]),
    ("6p 200bb", "5bet jam faced", [("CO", "r"), ("BTN", "r"), ("CO", "r"), ("BTN", "j")]),
    ("6p 200bb", "cold 4bet", [("UTG", "r"), ("HJ", "r"), ("CO", "r"), ("UTG", "f"), ("HJ", "c")]),
    ("3p 100bb", "cold 4bet", [("BTN", "r"), ("SB", "r"), ("BB", "r")]),
    ("3p 100bb", "4bet pot", [("BTN", "r"), ("SB", "r"), ("BTN", "r"), ("SB", "c")]),
    ("9p 100bb ante", "cold 4bet", [("UTG", "r"), ("LJ", "r"), ("BTN", "r")]),
    ("9p 100bb ante", "4bet pot", [("UTG", "r"), ("CO", "r"), ("UTG", "r"), ("CO", "c")]),
    ("9p 100bb ante", "squeeze called", [("HJ", "r"), ("CO", "c"), ("BB", "r"), ("HJ", "c"), ("CO", "c")]),
]
RARE_KINDS = ["3bet pot", "4bet pot", "cold 4bet", "flat then 4bet", "squeeze called", "squeeze 4bet",
              "5bet jam faced", "4bet jam faced", "limp-reraise", "straddled 3bet pot"]
RARE_POST_KINDS = ["vs_bet", "turn"]        # postflop spots per heads-up rare pot (3-way: MULTIWAY_KINDS)
RARE_PRE_NODES = 2                          # decision nodes checked before a line's last steps


def build_line(setup, steps):
    """The preflop actions of a RARE_LINES template in this setup, as (actions, decision prefixes, reach,
    closed): the prefixes are the action lists at which each named seat acts, `reach` how often the
    charts play the line (an open-limp, off the charts, counts its share of preflop.LIMP) and `closed`
    whether the line goes to the flop. Raises a ValueError when a named seat doesn't get to act."""
    from gto import preflop as pf
    from gto.advisor import replay_hand
    T = _table(setup)
    stack = setup[2]
    acts, prefixes, ranges, reach = [], [], {}, 1.0

    def take(p, a, key, strat):
        nonlocal acts, reach
        v = strat.get(key)
        if v is None and key == "call" and not any(h["action"] in ("raise", "all-in") for h in acts):
            v = pf.LIMP
        if v is None:
            v = np.zeros_like(next(iter(strat.values())))
        w = ranges.get(p, np.ones_like(v))
        reach *= float((w * v).sum() / max(w.sum(), 1e-12))
        ranges[p] = w * v
        acts = acts + [a]

    for pos, verb in steps:
        while True:
            st = replay_hand(T, stack, {"preflop": acts}, 1)[0]
            if st.closed or st.to_act is None:
                raise ValueError(f"{setup[0]}: {pos} never acts in {describe(acts)}")
            if st.to_act == pos:
                break
            take(st.to_act, {"pos": st.to_act, "action": "fold"}, "fold", pf.strategy_for(st.to_act, acts, T))
        prefixes.append(list(acts))
        if verb == "r":
            to = _size(st, pos, st.n_raises, T, stack)
            a = ({"pos": pos, "action": "all-in"} if to >= stack - st.inv_start[pos] - 1e-9 or st.n_raises >= 3
                 else {"pos": pos, "action": "raise", "to": to})
        elif verb == "j":
            a = {"pos": pos, "action": "all-in"}
        else:
            a = {"pos": pos, "action": {"c": "call", "f": "fold"}[verb]}
        key = "raise" if a["action"] in ("raise", "all-in") else a["action"]
        take(pos, a, key, pf.strategy_for(pos, acts, T))
    st = replay_hand(T, stack, {"preflop": acts}, 1)[0]
    return acts, prefixes, reach, bool(st.closed or st.to_act is None)


def rare_jobs(seed=7, setups=None):
    """(preflop jobs, postflop jobs) for RARE_LINES (deterministic for a seed): per line its last
    RARE_PRE_NODES decision nodes (hero holding AA, KK, QQ, AKs and a random hand) and, if it ends at
    a decision, that node too; a closed line's ranges are checked (a job with no hands) and its pot gets
    postflop spots, RARE_POST_KINDS heads-up (flop facing a bet, turn) or MULTIWAY_KINDS 3-way. Jobs
    match check_preflop and check_postflop."""
    from gto.cards import card_str
    from gto.advisor import replay_hand
    by_tag = {s[0]: s for s in SETUPS}
    rng = random.Random(seed + 1)
    pre_jobs, post_jobs, seen = [], [], set()
    for li, (tag, kind, steps) in enumerate(RARE_LINES):
        setup = by_tag[tag]
        if setups is not None and setup not in setups:
            continue
        acts, prefixes, _, closed = build_line(setup, steps)
        nodes = prefixes[-RARE_PRE_NODES:] + ([] if closed else [acts])
        for node in nodes:
            key = (tag, json.dumps(node))
            if key in seen:
                continue
            seen.add(key)
            hero = replay_hand(_table(setup), setup[2], {"preflop": node}, 1)[0].to_act
            a, b = rng.sample(range(52), 2)
            pre_jobs.append((setup, node, hero, ["AsAd", "KsKd", "QsQd", "AhKh", card_str(a) + card_str(b)], False))
        if closed:
            pre_jobs.append((setup, acts, None, [], False))      # the ranges the pot reaches the flop with
            n_in = len(replay_hand(_table(setup), setup[2], {"preflop": acts}, 1)[0].active)
            if n_in in (2, 3):
                kinds = MULTIWAY_KINDS if n_in == 3 else RARE_POST_KINDS
                post_jobs += postflop_jobs(setup, [(acts, 1.0)], len(kinds), seed + 1000 + li,
                                           multiway=n_in == 3, kinds=kinds)
    return pre_jobs, post_jobs


def report(flags_pre, flags_post, n_pre, n_post, seconds, out, quick=False, rare=False):
    counts = defaultdict(lambda: [0, 0])
    by_tag = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)
    seen = set()
    for i, fl in enumerate((flags_pre, flags_post)):
        for name, ex in fl:
            if "line" in ex:                 # range flags: once per setup, line and player
                key = (name, ex.get("tag"), ex["line"], ex.get("player"))
                if key in seen:
                    continue
                seen.add(key)
            counts[name][i] += 1
            by_tag[name][ex.get("tag", "")] += 1
            if len(examples[name]) < 8:
                examples[name].append(ex)
    lines = ["# Sanity sweep" + (" (quick)" if quick else ""), "",
             f"{n_pre} preflop decision nodes (x {5 if quick else 6} hands) and {n_post} postflop spots, {seconds:.0f}s.", ""]
    if rare:
        lines += [f"Includes the {len(RARE_LINES)} rare lines (RARE_LINES: {', '.join(RARE_KINDS)}); "
                  "their closed lines are range checks with no hands.", ""]
    lines += ["| flag | preflop | postflop | by setup |", "|---|---|---|---|"]
    for name in sorted(counts, key=lambda k: (k in INFO, -sum(counts[k]))):
        tags = ", ".join(f"{t}: {c}" for t, c in sorted(by_tag[name].items(), key=lambda kv: -kv[1]))
        lines.append(f"| {name}{' (info)' if name in INFO else ''} | {counts[name][0]} | {counts[name][1]} | {tags} |")
    for name in sorted(examples, key=lambda k: (k in INFO, k)):
        lines += ["", f"## {name}", ""]
        for ex in examples[name]:
            if "state" in ex:
                s = ex["state"]
                head = f"- [{ex.get('tag')}] {s['hero']} {s['hole']}" + (f" on {s['board']}" if s.get("board") else "") + \
                       f": {describe(s['preflop'])}" + "".join(f" / {k} {describe(s[k])}" for k in ("flop", "turn", "river") if s.get(k))
            else:
                head = f"- [{ex.get('tag')}] {ex.get('line')}"
            rest = {k: v for k, v in ex.items() if k not in ("state", "line", "notes", "tag")}
            lines.append(head + ("  \n  `" + json.dumps(rest, default=str)[:400] + "`" if rest else ""))
            if "state" in ex:
                lines.append("  state: `" + json.dumps(ex["state"])[:700] + "`")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=int, default=1500, help="preflop decision nodes to check (over all setups)")
    ap.add_argument("--post", type=int, default=1400, help="postflop spots to check (over all setups)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="sanity_report.md")
    ap.add_argument("--confirm", type=float, default=10.0,
                    help="re-run flagged postflop spots with this budget per street (0: skip)")
    ap.add_argument("--quick", action="store_true",
                    help="the small fixed set run before each deploy (~3 min), rare lines included; implies --strict")
    ap.add_argument("--strict", action="store_true",
                    help="exit with status 1 when any flag other than the info ones remains")
    ap.add_argument("--rare", action="store_true",
                    help="add the RARE_LINES checks to a full run (quick mode always has them)")
    ap.add_argument("--full-report", nargs="?", const="reports", metavar="DIR",
                    help="full sweep plus the rare lines, report written to DIR/sanity_YYYY-MM-DD.md "
                         "(default DIR: reports); for an occasional or nightly run, nothing schedules it")
    args = ap.parse_args()
    if args.full_report:
        if args.quick:
            ap.error("--full-report runs the full sweep: drop --quick")
        os.makedirs(args.full_report, exist_ok=True)
        args.out = os.path.join(args.full_report, f"sanity_{time.strftime('%Y-%m-%d')}.md")
        args.rare = True
    strict = args.strict or args.quick
    t0 = time.time()
    if args.quick:
        pre_jobs, post_jobs = quick_jobs(args.seed)
    else:
        pre_jobs, post_jobs = full_jobs(args)
        if args.rare:
            rp, rq = rare_jobs(args.seed)
            pre_jobs, post_jobs = pre_jobs + rp, post_jobs + rq
    print(f"{len(pre_jobs)} preflop nodes, {len(post_jobs)} postflop spots", flush=True)
    with Pool(args.procs) as pool:
        fpre = [f for fl in pool.imap_unordered(check_preflop, pre_jobs, chunksize=1 if args.quick else 8) for f in fl]
        print(f"preflop done in {time.time() - t0:.0f}s", flush=True)
        fpost = [f for fl in pool.imap_unordered(check_postflop, post_jobs, chunksize=1 if args.quick else 2) for f in fl]
    print(f"postflop done in {time.time() - t0:.0f}s: {len(fpost)} flags", flush=True)
    if args.confirm > 0:
        # short solves sharing the machine can be far from equilibrium, which looks like a bug (a bet
        # worth exactly the pot, a range near empty): re-run each flagged spot with a longer budget on
        # half the cores and keep only the flags that reproduce. Quick mode re-runs only spots with a
        # flag other than the info ones (an unsettled flop is common and blocks nothing); info flags of
        # spots not re-run are kept as they came
        jobs = {}
        for name, ex in fpost:
            if "state" in ex and name not in ("error", "nan") and not (args.quick and name in INFO):
                st = dict(ex["state"]); pool_ = st.pop("opponents", None) == "pool"
                setup = next(s for s in SETUPS if ex["tag"].startswith(s[0]))
                jobs[json.dumps(ex["state"], sort_keys=True)] = (setup, st, "confirm", pool_, args.confirm)
        again = []
        if jobs:
            with Pool(max(1, args.procs // 2)) as pool:
                again = pool.map(check_postflop, list(jobs.values()), chunksize=1)
        confirmed = {(k, name) for k, fl in zip(jobs, again) for name, _ in fl}
        before = len(fpost)
        fpost = [(name, ex) for name, ex in fpost if name in ("error", "nan") or "state" not in ex
                 or json.dumps(ex["state"], sort_keys=True) not in jobs
                 or (json.dumps(ex["state"], sort_keys=True), name) in confirmed]
        print(f"confirm pass ({args.confirm:g}s solves, {len(jobs)} spots): {len(fpost)} of {before} "
              f"postflop flags reproduce", flush=True)
    counts = report(fpre, fpost, len(pre_jobs), len(post_jobs), time.time() - t0, args.out, quick=args.quick,
                    rare=args.quick or args.rare)
    print(f"done in {time.time() - t0:.0f}s; report in {args.out}")
    for name, (a, b) in sorted(counts.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {name}{' (info)' if name in INFO else ''}: preflop {a}, postflop {b}")
    bugs = blocking(counts)
    if strict:
        if bugs:
            print(f"FAIL: {', '.join(bugs)} (see {args.out})")
            sys.exit(1)
        print("OK: no flags other than info ones")


def blocking(counts):
    """The flags (by name) that fail a strict run: any but the info ones."""
    return sorted(name for name, c in counts.items() if name not in INFO and sum(c) > 0)


def full_jobs(args):
    rng = random.Random(args.seed)
    from gto.cards import card_str
    pre_jobs, post_jobs = [], []
    for setup in SETUPS:
        default = setup[0] == "6p 100bb"
        nodes, closed = preflop_lines(setup, max_in=None if default else 3)
        n_pre = int(round(args.pre * setup[5]))
        for acts, hero, _ in weighted_sample(nodes, [r for _, _, r in nodes], n_pre, rng):
            rand = []
            while len(rand) < 2:
                a, b = rng.sample(range(52), 2)
                rand.append(card_str(a) + card_str(b))
            pre_jobs.append((setup, acts, hero, ["AsAd", "KsKd", "QsQd", "AhKh"] + rand,
                             default and rng.random() < POOL_SHARE))
        n_post = int(round(args.post * setup[5]))
        if default:     # a share of the default setup's spots are 3-way pots
            post_jobs += postflop_jobs(setup, closed, n_post - n_post // 6, args.seed, pool_share=POOL_SHARE)
            post_jobs += postflop_jobs(setup, closed, n_post // 6, args.seed + 1, multiway=True)
        else:
            post_jobs += postflop_jobs(setup, closed, n_post, args.seed)
        print(f"{setup[0]}: {len(nodes)} nodes, {len(closed)} closed lines; {n_pre} preflop, {n_post} postflop", flush=True)
    return pre_jobs, post_jobs


if __name__ == "__main__":
    main()
