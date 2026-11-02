"""A realistic 6-max session as PokerStars hand histories, for timing and checking the session analyzer
(gto/static/session.js; scripts/bench_session.mjs runs it as the page does).

    python scripts/bench_session.py --hands 1000 --seed 1 --tables 8 --out DIR

writes DIR/table_1.txt ... DIR/table_8.txt (the hands dealt round-robin over the tables, as a night of 8 tables
saves them: one file a table). 6-max $0.50/$1 (amounts are big blinds), the button moving each hand; "Hero" sits in
seat 4 and so plays every position in turn. Stacks: Hero 100bb (auto top-up) 70% of hands, else 60-200bb; each
other seat 100bb 60% of hands, else 40-250bb. Preflop every seat plays the page's charts (advisor.preflop_strategy
at its effective stack, the chart's mix sampled with its own cards), the chart's sizes, and opens and 3bets of
another size now and then (2.0-3.0bb opens, a 3bet 10-20% off) as players do. After the flop a simple player:
its hand's equity against a random hand (12 sampled runouts) decides bet, check, call, raise or fold, with some
noise and the preflop raiser c-betting more; bets of 25-125% of the pot (a third and three quarters most often),
raises to 3x. Nobody plays GTO after the flop, so the analyzer has mistakes to find, Hero's among them.
Seeded: the same seed, the same files.
"""
from __future__ import annotations
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
from gto import preflop as pf  # noqa: E402
from gto.advisor import preflop_strategy, replay_hand, _default_size  # noqa: E402
from gto.cards import ALL_COMBOS, COMBO_INDEX, card_str  # noqa: E402
from gto.evaluator import eval_board  # noqa: E402

SEATS6 = pf.SEATS[6]                       # UTG HJ CO BTN SB BB: preflop order
STREET_NAMES = ["preflop", "flop", "turn", "river"]
HEADS = {1: "FLOP", 2: "TURN", 3: "RIVER"}
SIZES = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25]
SIZE_W = [0.08, 0.32, 0.14, 0.08, 0.24, 0.09, 0.05]


def money(x: float) -> str:
    s = f"{round(x + 1e-9, 2):.2f}"
    return "$" + (s[:-3] if s.endswith(".00") else s)


def equity(hole, board, rng, n=12) -> float:
    """hole's share of the pot against a random hand over n sampled runouts (ties half)."""
    dead = set(hole) | set(board)
    deck = [c for c in range(52) if c not in dead]
    idx = COMBO_INDEX[tuple(sorted(hole))]
    tot = 0.0
    for _ in range(n if len(board) < 5 else 1):
        run = list(board) + rng.sample(deck, 5 - len(board))
        s = eval_board(run)
        me = s[idx]
        ok = s >= 0
        ok[idx] = False
        others = s[ok]
        tot += (np.sum(others < me) + 0.5 * np.sum(others == me)) / max(1, len(others))
    return tot / (n if len(board) < 5 else 1)


def preflop_move(p, before, table, st, hole, rng):
    eff = _eff(table, p, st)
    strat = preflop_strategy(p, before, table, eff)
    ci = COMBO_INDEX[tuple(sorted(hole))]
    acts = [(k, float(v[ci])) for k, v in strat.items() if float(v[ci]) > 0]
    tot = sum(w for _, w in acts)
    x = rng.random() * tot
    for k, w in acts:
        x -= w
        if x <= 0:
            break
    if k in ("raise", "bet"):
        to = _default_size(st, p, True)
        if st.n_raises == 0 and rng.random() < 0.3 and to <= 3.5:
            to = rng.choice([2.0, 2.2, 2.3, 3.0])
        elif st.n_raises >= 1 and rng.random() < 0.25:
            to = round(to * rng.uniform(0.85, 1.15) * 2) / 2
        return {"pos": p, "action": "raise", "to": round(to, 2)}
    return {"pos": p, "action": k}


def _eff(table, p, st):
    stk = table.stacks or {q: table.depth for q in table}
    others = [stk[q] for q in st.active if q != p] or [stk[p]]
    return min(stk[p], max(others))


def postflop_move(p, st, table, hole, board, inv, pfr, rng):
    stk = (table.stacks or {})[p] if table.stacks else table.depth
    left = stk - inv[p] - st.bets[p]
    pot = st.pot_start + sum(st.bets.values())
    to_call = st.cur_bet - st.bets[p]
    e = equity(hole, board, rng)
    e += rng.gauss(0, 0.06)
    if to_call <= 1e-9:
        pb = 0.75 if e > 0.8 else 0.45 if e > 0.62 else 0.25 if e < 0.3 else 0.12
        if p == pfr and len(board) == 3:
            pb += 0.25
        if rng.random() < min(pb, 0.95) and left > 0:
            amt = round(pot * rng.choices(SIZES, SIZE_W)[0], 2)
            if amt >= left * 0.8:
                return {"pos": p, "action": "all-in"}
            return {"pos": p, "action": "bet", "to": max(1.0, amt)}
        return {"pos": p, "action": "check"}
    need = to_call / (pot + to_call)
    if e > 0.86 and rng.random() < 0.4 and st.n_raises < 3 or e < 0.25 and rng.random() < 0.04 and st.n_raises < 2:
        to = round(st.cur_bet * 3, 2)
        if to - st.bets[p] >= left * 0.7:
            return {"pos": p, "action": "all-in"}
        return {"pos": p, "action": "raise", "to": to}
    if e + 0.08 > need * 1.6 or rng.random() < 0.1:
        return {"pos": p, "action": "call"}
    return {"pos": p, "action": "fold"}


def deal_hand(k: int, rng: random.Random, table_name: str, base: int):
    btn_seat = k % 6 + 1
    # seat s (1-6) -> position: the button's seat is BTN, then SB, BB, UTG, HJ, CO clockwise
    ring = ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
    pos_of = {(btn_seat - 1 + i) % 6 + 1: ring[i] for i in range(6)}
    seat_of = {v: s for s, v in pos_of.items()}
    name = {pos_of[s]: ("Hero" if s == 4 else f"villain{s}") for s in range(1, 7)}
    hero = pos_of[4]
    stacks = {}
    for p in SEATS6:
        if p == hero:
            stacks[p] = 100.0 if rng.random() < 0.7 else float(rng.randint(60, 200))
        else:
            stacks[p] = 100.0 if rng.random() < 0.6 else float(rng.randint(40, 250))
    table = pf.Table(SEATS6, depth=100.0, stacks=stacks if len(set(stacks.values())) > 1 else None)
    deck = list(range(52))
    rng.shuffle(deck)
    holes = {p: (deck[2 * i], deck[2 * i + 1]) for i, p in enumerate(SEATS6)}
    board_all = deck[12:17]
    streets = {s: [] for s in STREET_NAMES}
    lines = []
    head = (f"PokerStars Hand #{base + k}:  Hold'em No Limit ($0.50/$1.00 USD) - 2026/11/0{1 + k % 2} "
            f"{18 + (k // 90) % 6}:{(k // 2) % 60:02d}:{(k * 7) % 60:02d} ET")
    lines += [head, f"Table '{table_name}' 6-max Seat #{btn_seat} is the button"]
    for s in range(1, 7):
        lines.append(f"Seat {s}: {name[pos_of[s]]} ({money(stacks[pos_of[s]])} in chips)")
    lines += [f"{name['SB']}: posts small blind $0.50", f"{name['BB']}: posts big blind $1", "*** HOLE CARDS ***",
              f"Dealt to Hero [{card_str(holes[hero][0])} {card_str(holes[hero][1])}]"]
    pfr = None
    for si, sname in enumerate(STREET_NAMES):
        board = board_all[:[0, 3, 4, 5][si]]
        if si:
            if si == 1:
                lines.append(f"*** FLOP *** [{' '.join(card_str(c) for c in board)}]")
            else:
                lines.append(f"*** {HEADS[si]} *** [{' '.join(card_str(c) for c in board[:-1])}] [{card_str(board[-1])}]")
        for _ in range(40):
            sts = replay_hand(table, 100.0, streets, si + 1)
            if len(sts) <= si:
                break
            st = sts[si]
            if st.closed or st.to_act is None:
                break
            p = st.to_act
            inv = {q: sum(x.bets.get(q, 0.0) for x in sts[:si]) for q in SEATS6}
            if si == 0:
                a = preflop_move(p, streets["preflop"], table, st, holes[p], rng)
            else:
                a = postflop_move(p, st, table, holes[p], board, inv, pfr, rng)
            stk = stacks[p]
            left = stk - inv[p] - st.bets[p]
            if a["action"] in ("raise", "bet") and a["to"] - st.bets[p] >= left - 1e-9:
                a = {"pos": p, "action": "all-in"}
            if a["action"] == "raise" and st.cur_bet <= 1e-9 and si:
                a["action"] = "bet"
            if a["action"] in ("raise", "bet") and a["to"] <= st.cur_bet + 1e-9:
                a = {"pos": p, "action": "call" if st.cur_bet > st.bets[p] else "check"}
            if a["action"] == "check" and st.cur_bet > st.bets[p] + 1e-9:
                a = {"pos": p, "action": "fold"}
            if a["action"] in ("call", "fold") and st.cur_bet <= st.bets[p] + 1e-9:
                a = {"pos": p, "action": "check"}
            streets[sname].append(a)
            lines.append(_text(a, st, left, name))
            if si == 0 and a["action"] in ("raise", "all-in"):
                pfr = p
        sts = replay_hand(table, 100.0, streets, si + 1)
        st = sts[-1]
        if len(st.active) <= 1:
            break
        live = [q for q in st.active if q not in st.allin]
        if len(live) <= 1 and si < 3:
            for sj in range(si + 1, 4):   # everyone all-in: the board runs out
                b = board_all[:[0, 3, 4, 5][sj]]
                lines.append(f"*** FLOP *** [{' '.join(card_str(c) for c in b)}]" if sj == 1 else
                             f"*** {HEADS[sj]} *** [{' '.join(card_str(c) for c in b[:-1])}] [{card_str(b[-1])}]")
            break
    lines.append("*** SUMMARY ***")
    return "\n".join(lines)


def _text(a, st, left, name) -> str:
    p, act = a["pos"], a["action"]
    n = name[p]
    if act == "fold":
        return f"{n}: folds"
    if act == "check":
        return f"{n}: checks"
    if act == "call":
        add = min(st.cur_bet - st.bets[p], left)
        return f"{n}: calls {money(add)}" + (" and is all-in" if add >= left - 1e-9 else "")
    if act == "all-in":
        to = st.bets[p] + left
        if st.cur_bet <= 1e-9:
            return f"{n}: bets {money(left)} and is all-in"
        if to <= st.cur_bet + 1e-9:
            return f"{n}: calls {money(left)} and is all-in"
        return f"{n}: raises {money(to - st.cur_bet)} to {money(to)} and is all-in"
    if act == "bet":
        return f"{n}: bets {money(a['to'])}"
    return f"{n}: raises {money(a['to'] - st.cur_bet)} to {money(a['to'])}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tables", type=int, default=8)
    ap.add_argument("--out", required=True, help="a folder: one file a table")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    files = [[] for _ in range(args.tables)]
    for k in range(args.hands):
        t = k % args.tables
        files[t].append(deal_hand(k // args.tables * 1 + t * 100000, rng, f"Synth {t + 1}", 260000000000))
    for t, hands in enumerate(files):
        with open(os.path.join(args.out, f"table_{t + 1}.txt"), "w") as f:
            f.write("\n\n\n".join(hands) + "\n")
    print(f"{args.hands} hands in {args.tables} files under {args.out}")


if __name__ == "__main__":
    main()
