"""Sanity sweep: run many common spots through the advisor and flag outputs with the signatures of bugs
found before, so the next one shows up here rather than at the table.

Spots (6-max, 100bb):
  preflop:  every decision in the charts' preflop lines (no open-limps, at most two callers per raise),
            hero holding AA, KK, QQ, AKs and two random hands
  postflop: a sample of lines that reach the flop heads-up (single-raised, 3bet, 4bet pots, all seat
            pairs), random flops, hero's hand drawn from its own range (sometimes off it), deciding
            first, facing a check, or facing a bet; some turns after a checked flop

Flags:
  error            the advisor returned an error or raised
  nan              an EV or frequency is NaN or infinite
  premium_fold     preflop: AA told to fold; KK below a 5bet; QQ/AKs to a single raise
  filled_range     (info) a player's preflop line left them under preflop.MIN_LINE_COMBOS (8) and was
                   filled out with the hands that take the action: the safeguard working
  thin_range       a player's preflop range is under MIN_LINE_COMBOS and was not filled out (the solve
                   would play nearly face-up)
  face_up          postflop: a range under 3 combos before its player acts on the street
  ev_is_pot        postflop: a bet or all-in whose EV is exactly the pot (the villain always folds:
                   the signature of a face-up solve)
  strong_fold      postflop: a set or better (on an unpaired board) told to fold to a bet
  rec_ev_gap       the recommended action's EV trails the best by more than max(1bb, 10% of the pot)
  unsettled        (info) postflop: the solve is more than 10% of the pot from equilibrium. Expected
                   on big flop ranges at short budgets: the site plays a smaller game in the browser,
                   shows a convergence badge and offers Accurate mode

Postflop flags are confirmed: each flagged spot is solved again with a longer budget (--confirm) on
half the cores, and only flags that reproduce are reported (short solves sharing the machine can be far
from equilibrium, which looks like a face-up range or a bet worth exactly the pot).

  python scripts/sanity_sweep.py                       # full sweep (~30-40 min on 8 cores)
  python scripts/sanity_sweep.py --pre 200 --post 100  # quicker
  python scripts/sanity_sweep.py --out report.md       # where to write the report (default sanity_report.md)
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

STACK = 100.0
N_PLAYERS = 6


def _table():
    from gto import preflop as pf
    return pf.Table(pf.SEATS[N_PLAYERS])


def _size(st, p, nr, T):
    """Default raise-to for p at a node with nr raises in (open 2.5/3, 3bet 3x in position / 4x out of
    position plus one raise per caller, 4bet 2.3x / 2.5x, then all-in)."""
    from gto import preflop as pf
    cur = st.cur_bet
    if nr == 0:
        return 3.0 if p == "SB" else 2.5
    post = pf.postflop_order(list(T))
    others = [q for q in st.active if q != p and st.bets[q] >= cur - 1e-9]
    ip = all(post.index(p) > post.index(q) for q in others)
    if nr == 1:
        callers = sum(1 for q in st.active if q != p and 0 < st.bets[q] >= cur - 1e-9) - 1
        return round(cur * (3.0 if ip else 4.0) + max(callers, 0) * cur, 1)
    if nr == 2:
        return round(cur * (2.3 if ip else 2.5), 1)
    return STACK


def preflop_lines():
    """(decision nodes, closed heads-up lines): every node of the charts' preflop tree as (actions,
    player to act, reach), and the lines that close with exactly two players in as (actions, reach);
    reach is how often the line happens (the product of each player's share of their range taking
    their action)."""
    from gto import preflop as pf
    from gto.advisor import replay_hand
    T = _table()
    nodes, closed = [], []

    def dfs(acts, ranges, reach):
        st = replay_hand(T, STACK, {"preflop": acts}, 1)[0]
        if st.closed or st.to_act is None:
            if len(st.active) == 2 and not st.allin:
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
        for key, v in strat.items():
            if v.sum() < 1e-9:
                continue
            if key in ("call", "check") and nr == 0 and p not in ("SB", "BB"):
                continue                      # no open-limps
            if key == "call" and nr > 0 and since >= 2:
                continue                      # at most two callers per raise
            if key == "fold":
                a = {"pos": p, "action": "fold"}
            elif key in ("call", "check"):
                a = {"pos": p, "action": key}
            else:
                to = _size(st, p, nr, T)
                a = ({"pos": p, "action": "all-in"} if to >= STACK - st.inv_start[p] - 1e-9 or nr >= 3
                     else {"pos": p, "action": "raise", "to": to})
            # how often the line happens: each player's share of their range taking their action
            w = ranges.get(p, np.ones_like(v))
            share = float((w * v).sum() / max(w.sum(), 1e-12))
            dfs(acts + [a], dict(ranges, **{p: w * v}), reach * share)

    dfs([], {}, 1.0)
    return nodes, closed


def weighted_sample(items, weights, n, rng, power=0.5):
    """n distinct items, drawn with probability proportional to weight**power (softened, so rare lines
    still appear)."""
    w = np.array(weights, float) ** power
    idx = np.random.default_rng(rng.randrange(1 << 30)).choice(len(items), size=min(n, len(items)), replace=False,
                                                               p=w / w.sum())
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


def check_preflop(job):
    """Flags for one preflop decision node."""
    acts, hero, hands = job
    from gto.advisor import Advisor, preflop_ranges
    T = _table()
    adv = Advisor(time_budget=1.0, prior_budget=0.5)
    flags = []
    n_raises = sum(a["action"] in ("raise", "all-in") for a in acts)
    try:
        from gto.preflop import MIN_LINE_COMBOS
        rng = preflop_ranges(acts, T, STACK)
        for p in getattr(rng, "rebuilt", []):
            flags.append(("filled_range", {"line": describe(acts), "player": p}))
        for p, r in rng.items():
            if r.sum() < MIN_LINE_COMBOS and p not in getattr(rng, "rebuilt", []):
                flags.append(("thin_range", {"line": describe(acts), "player": p, "combos": round(float(r.sum()), 1)}))
    except Exception as e:   # noqa: BLE001 - a sweep reports everything
        flags.append(("error", {"line": describe(acts), "where": "preflop_ranges", "error": repr(e)}))
    for hole in hands:
        state = {"players": N_PLAYERS, "hero": hero, "hole": hole, "board": "", "stack": STACK, "preflop": acts}
        try:
            res = adv.advise(state)
        except Exception as e:   # noqa: BLE001
            flags.append(("error", {"state": state, "error": repr(e)}))
            continue
        if "error" in res:
            flags.append(("error", {"state": state, "error": res["error"]}))
            continue
        if not _finite(res):
            flags.append(("nan", {"state": state, "actions": res["actions"]}))
        from gto.advisor import hand_class
        from gto.cards import cards
        hc = hand_class(tuple(cards(hole)))
        rec = res.get("recommended")
        if rec == "fold" and (hc == "AA" or (hc == "KK" and n_raises <= 3) or (hc in ("QQ", "AKs") and n_raises <= 1)):
            flags.append(("premium_fold", {"state": state, "actions": res["actions"]}))
        evs = [a for a in res["actions"] if a.get("ev_bb") is not None]
        if evs and rec:
            best = max(a["ev_bb"] for a in evs)
            rec_ev = next((a["ev_bb"] for a in evs if a["action"] == rec), None)
            if rec_ev is not None and best - rec_ev > max(1.0, 0.1 * res.get("pot_bb", 0)):
                flags.append(("rec_ev_gap", {"state": state, "actions": res["actions"]}))
    return flags


def postflop_jobs(closed, n, seed):
    """n postflop spots from the closed heads-up lines: (state, kind) with kind first/vs_check/vs_bet/turn."""
    from gto.advisor import preflop_ranges, replay_hand
    from gto.cards import ALL_COMBOS, card_str
    from gto import preflop as pf
    rng = random.Random(seed)
    T = _table()
    post = pf.postflop_order(list(T))
    jobs = []
    # spread over pot types (single-raised, 3bet, 4bet+), lines within each drawn by how often they happen
    by_type = defaultdict(list)
    for l, reach in closed:
        by_type[min(sum(a["action"] in ("raise", "all-in") for a in l), 3)].append((l, reach))
    pools = [by_type[k] for k in sorted(by_type) if by_type[k]]
    while len(jobs) < n:
        pool_ = rng.choice(pools)
        l = rng.choices([x[0] for x in pool_], weights=[x[1] ** 0.5 for x in pool_])[0]
        st = replay_hand(T, STACK, {"preflop": l}, 1)[0]
        players = sorted(st.active, key=post.index)
        oop, ip = players
        ranges = preflop_ranges(l, T, STACK)
        kind = rng.choice(["first", "vs_check", "vs_bet", "vs_bet", "turn"])
        hero = oop if kind in ("first", "turn") else ip if kind == "vs_check" else rng.choice([oop, ip])
        deck = list(range(52))
        board = rng.sample(deck, 4 if kind == "turn" else 3)
        w = ranges[hero].copy()
        for c in board:
            w[[i for i, (a, b) in enumerate(ALL_COMBOS) if a == c or b == c]] = 0.0
        if rng.random() < 0.2 or w.sum() < 1e-9:
            free = [i for i, (a, b) in enumerate(ALL_COMBOS) if a not in board and b not in board]
            ci = rng.choice(free)
        else:
            ci = rng.choices(range(len(w)), weights=w)[0]
        hole = "".join(card_str(c) for c in ALL_COMBOS[ci])
        pot = st.pot_start + sum(st.bets.values())
        flop, turn = [], []
        if kind == "vs_check":
            flop = [{"pos": oop, "action": "check"}]
        elif kind == "vs_bet":
            size = rng.choice([0.33, 0.75])
            if hero == ip:
                flop = [{"pos": oop, "action": "bet", "amount": round(pot * size, 1)}]
            else:
                flop = [{"pos": oop, "action": "check"}, {"pos": ip, "action": "bet", "amount": round(pot * size, 1)}]
        elif kind == "turn":
            flop = [{"pos": oop, "action": "check"}, {"pos": ip, "action": "check"}]
        state = {"players": N_PLAYERS, "hero": hero, "hole": hole, "board": "".join(card_str(c) for c in board),
                 "stack": STACK, "preflop": l, "flop": flop, "turn": turn}
        jobs.append((state, kind))
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
    state, kind = job[:2]
    budget = job[2] if len(job) > 2 else 3.0
    from gto.advisor import Advisor
    # 3s per street: the sweep runs a solve per core, so this is about what 1s gets on its own; the
    # confirm pass re-runs flagged spots with more
    adv = Advisor(time_budget=budget, prior_budget=budget / 3)
    flags = []
    try:
        res = adv.advise(state)
    except Exception as e:   # noqa: BLE001
        return [("error", {"state": state, "error": repr(e)})]
    if "error" in res:
        return [("error", {"state": state, "error": res["error"]})]
    if not _finite(res):
        flags.append(("nan", {"state": state, "actions": res["actions"]}))
    info = {"state": state, "actions": res.get("actions"), "notes": res.get("notes")}
    if res.get("multiway"):
        return flags
    hc, vc = res.get("hero_range_combos"), res.get("villain_range_combos")
    # a range counts only before its player acts this street: acting narrows it legitimately (a bet
    # with 2 combos of a 15-combo range is a real betting range; a rare check shrinks hero's, but
    # hero's EVs on this street don't depend on it)
    street = state.get("turn" if len(state["board"]) == 8 else "flop", [])
    hero_acted = any(a["pos"] == state["hero"] for a in street)
    villain_acted = any(a["pos"] != state["hero"] for a in street)
    if (vc is not None and vc < 3 and not villain_acted) or (hc is not None and hc < 3 and not hero_acted):
        flags.append(("face_up", dict(info, hero_combos=hc, villain_combos=vc)))
    pot = res.get("pot_bb", 0.0)
    for a in res.get("actions", []):
        if (a["action"].startswith("bet") or a["action"].startswith("raise") or a["action"] == "all-in") \
                and a.get("ev_bb") is not None and abs(a["ev_bb"] - pot) < 0.02 and a.get("freq", 0) > 0.5:
            flags.append(("ev_is_pot", info))
            break
    if res.get("to_call_bb", 0) > 0 and res.get("recommended") == "fold":
        cls, paired = _hand_class_rank(state["hole"], state["board"])
        if cls <= 6 and not paired:
            flags.append(("strong_fold", info))
    evs = [a for a in res.get("actions", []) if a.get("ev_bb") is not None]
    rec = res.get("recommended")
    if evs and rec:
        best = max(a["ev_bb"] for a in evs)
        rec_ev = next((a["ev_bb"] for a in evs if a["action"] == rec), None)
        if rec_ev is not None and best - rec_ev > max(1.0, 0.1 * pot):
            flags.append(("rec_ev_gap", info))
    if res.get("solver", {}).get("exploitability_pct", 0) > 10:
        flags.append(("unsettled", dict(info, exploitability=res["solver"]["exploitability_pct"])))
    return flags


def report(flags_pre, flags_post, n_pre, n_post, seconds, out):
    counts = defaultdict(lambda: [0, 0])
    examples = defaultdict(list)
    seen = set()
    for i, fl in enumerate((flags_pre, flags_post)):
        for name, ex in fl:
            if "line" in ex:                 # range flags: once per line and player
                key = (name, ex["line"], ex.get("player"))
                if key in seen:
                    continue
                seen.add(key)
            counts[name][i] += 1
            if len(examples[name]) < 6:
                examples[name].append(ex)
    lines = ["# Sanity sweep", "",
             f"{n_pre} preflop decisions ({n_pre} nodes x hands) and {n_post} postflop spots, {seconds:.0f}s.", "",
             "| flag | preflop | postflop |", "|---|---|---|"]
    info = {"filled_range", "unsettled"}
    for name in sorted(counts, key=lambda k: (k in info, -sum(counts[k]))):
        lines.append(f"| {name}{' (info)' if name in info else ''} | {counts[name][0]} | {counts[name][1]} |")
    for name in sorted(examples):
        lines += ["", f"## {name}", ""]
        for ex in examples[name]:
            if "state" in ex:
                s = ex["state"]
                head = f"- {s['hero']} {s['hole']}" + (f" on {s['board']}" if s.get("board") else "") + \
                       f": {describe(s['preflop'])}" + (f" / flop {describe(s['flop'])}" if s.get("flop") else "")
            else:
                head = f"- {ex.get('line')}"
            rest = {k: v for k, v in ex.items() if k not in ("state", "line", "notes")}
            lines.append(head + ("  \n  `" + json.dumps(rest, default=str)[:400] + "`" if rest else ""))
            if "state" in ex:
                lines.append("  state: `" + json.dumps(ex["state"])[:600] + "`")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=int, default=1500, help="preflop decision nodes to check")
    ap.add_argument("--post", type=int, default=1200, help="postflop spots to check")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="sanity_report.md")
    ap.add_argument("--confirm", type=float, default=10.0,
                    help="re-run flagged postflop spots with this budget per street (0: skip)")
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(args.seed)
    nodes, closed = preflop_lines()
    nodes = weighted_sample(nodes, [r for _, _, r in nodes], args.pre, rng)
    from gto.cards import card_str
    pre_jobs = []
    for acts, hero, _ in nodes:
        rand = []
        while len(rand) < 2:
            a, b = rng.sample(range(52), 2)
            rand.append(card_str(a) + card_str(b))
        pre_jobs.append((acts, hero, ["AsAd", "KsKd", "QsQd", "AhKh"] + rand))
    post_jobs = postflop_jobs(closed, args.post, args.seed)
    print(f"{len(pre_jobs)} preflop nodes, {len(post_jobs)} postflop spots ({len(closed)} heads-up lines)", flush=True)
    with Pool(args.procs) as pool:
        fpre = [f for fl in pool.imap_unordered(check_preflop, pre_jobs, chunksize=8) for f in fl]
        print(f"preflop done in {time.time() - t0:.0f}s", flush=True)
        fpost = [f for fl in pool.imap_unordered(check_postflop, post_jobs, chunksize=2) for f in fl]
    print(f"postflop done in {time.time() - t0:.0f}s: {len(fpost)} flags", flush=True)
    if args.confirm > 0:
        # short solves sharing the machine can be far from equilibrium, which looks like a bug (a bet
        # worth exactly the pot, a range near empty): re-run each flagged spot with a longer budget on
        # half the cores and keep only the flags that reproduce
        spots = {json.dumps(ex["state"], sort_keys=True): ex["state"] for _, ex in fpost if "state" in ex}
        with Pool(max(1, args.procs // 2)) as pool:
            again = pool.map(check_postflop, [(st, "confirm", args.confirm) for st in spots.values()], chunksize=1)
        confirmed = {(json.dumps(st, sort_keys=True), name) for st, fl in zip(spots.values(), again) for name, _ in fl}
        dropped = len(fpost)
        fpost = [(name, ex) for name, ex in fpost
                 if "state" not in ex or (json.dumps(ex["state"], sort_keys=True), name) in confirmed]
        print(f"confirm pass ({args.confirm:g}s solves): {len(fpost)} of {dropped} postflop flags reproduce", flush=True)
    counts = report(fpre, fpost, len(pre_jobs), len(post_jobs), time.time() - t0, args.out)
    print(f"done in {time.time() - t0:.0f}s; report in {args.out}")
    for name, (a, b) in sorted(counts.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {name}: preflop {a}, postflop {b}")


if __name__ == "__main__":
    main()
