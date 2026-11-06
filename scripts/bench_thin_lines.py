"""How the floor that fills thin lines (advisor._post_floor, preflop.thin_line) should fill them: a census of the lines it
fills, a reference for them, and the candidates against it, on heads-up and 3-way pots across pot types and streets.

  python scripts/bench_thin_lines.py census --out census.jsonl                               # every line the floor fills
  python scripts/bench_thin_lines.py reference --census census.jsonl --per-cell 2 --out r.jsonl  # ~1.5 h on one core
  python scripts/bench_thin_lines.py report --out r.jsonl                                    # the tables

A street-start solve can't be the reference: the rarer the line, the less of the range reaches it, and on the lines the
floor fills it often reaches none at all (the shape of a range of 0.00 combos changes with the iteration count). See
`constrained` for what stands in for it. Results: docs/multiway_research.md, "Thin lines".
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gto import native  # noqa: E402  (first: holds BLAS to one thread)
import numpy as np      # noqa: E402
import gto.advisor as A  # noqa: E402
from gto import preflop as pf  # noqa: E402
from gto.cards import cards  # noqa: E402
from gto import multiway_solver as MW  # noqa: E402

N = 1326
SIZES = (0.33, 0.75, 1.25)


def p_(pos, action, **kw):
    return {"pos": pos, "action": action, **kw}


def fold(*seats):
    return [p_(s, "fold") for s in seats]


# (name, table size, stack, preflop actions)
THREE_WAY = [
    ("srp-co", 6, 100, fold("UTG", "HJ") + [p_("CO", "raise", to=2.5), p_("BTN", "call"), p_("SB", "fold"), p_("BB", "call")]),
    ("srp-utg", 6, 100, [p_("UTG", "raise", to=2.5)] + fold("HJ", "CO", "BTN") + [p_("SB", "call"), p_("BB", "call")]),
    ("3bet-cold", 7, 100, fold("UTG", "LJ", "HJ") + [p_("CO", "raise", to=2.5), p_("BTN", "raise", to=7.5), p_("SB", "fold"),
                                                    p_("BB", "call"), p_("CO", "call")]),
    ("squeeze", 6, 100, [p_("UTG", "fold"), p_("HJ", "raise", to=2.5), p_("CO", "call"), p_("BTN", "raise", to=9.5),
                         p_("SB", "fold"), p_("BB", "fold"), p_("HJ", "call"), p_("CO", "call")]),
    ("srp-short", 6, 40, fold("UTG", "HJ") + [p_("CO", "raise", to=2.5), p_("BTN", "call"), p_("SB", "fold"), p_("BB", "call")]),
    ("3bet-deep", 7, 200, fold("UTG", "LJ", "HJ") + [p_("CO", "raise", to=2.5), p_("BTN", "raise", to=7.5), p_("SB", "fold"),
                                                    p_("BB", "call"), p_("CO", "call")]),
]
HEADS_UP = [
    ("srp-btn", 6, 100, fold("UTG", "HJ", "CO") + [p_("BTN", "raise", to=2.5), p_("SB", "fold"), p_("BB", "call")]),
    ("srp-utg", 6, 100, [p_("UTG", "raise", to=2.5)] + fold("HJ", "CO", "BTN", "SB") + [p_("BB", "call")]),
    ("3bet-ip", 6, 100, fold("UTG", "HJ") + [p_("CO", "raise", to=2.5), p_("BTN", "raise", to=7.5)] + fold("SB", "BB")
     + [p_("CO", "call")]),
    ("3bet-bb", 6, 100, fold("UTG", "HJ", "CO") + [p_("BTN", "raise", to=2.5), p_("SB", "fold"), p_("BB", "raise", to=10.0),
                                                  p_("BTN", "call")]),
    ("4bet", 6, 100, fold("UTG", "HJ", "CO") + [p_("BTN", "raise", to=2.5), p_("SB", "fold"), p_("BB", "raise", to=10.0),
                                               p_("BTN", "raise", to=23.0), p_("BB", "call")]),
    ("srp-short", 6, 40, fold("UTG", "HJ", "CO") + [p_("BTN", "raise", to=2.5), p_("SB", "fold"), p_("BB", "call")]),
]
FLOPS = ["Ks7s2h", "Kh7d2c", "Jh9h4c", "8s8d3c", "Qd6c5s", "As9d4h", "Td8s6s", "7c5c2d", "AhKdQc", "9s9d9c"]
TURNS = ["2c", "Qs", "5h"]     # the first of these not on the flop, then the next (as many as --cards)
RIVERS = ["3d", "Ah", "Tc"]


def table(n):
    return pf.Table(pf.SEATS[n])


def next_cards(board, deck, k):
    used = {board[i:i + 2] for i in range(0, len(board), 2)}
    return [c for c in deck if c not in used][:k]


def lines(n, stack, streets, name, n_st, sizes=SIZES):
    """Every line of street `name` from the first decision on (one bet and one raise at most, no all-ins), as (actions,
    closed): each node the street reaches, closed or with a player to act."""
    def walk(acts):
        st = A.replay_hand(table(n), stack, {**streets, name: acts}, n_st)[-1]
        if len(st.active) < 2:
            return
        if acts:
            yield list(acts), st.closed
        if st.closed or st.to_act is None:
            return
        p = st.to_act
        pot = st.pot_start + sum(st.bets.values())
        behind = stack - st.inv_start[p]
        if st.cur_bet <= st.bets[p] + 1e-9:
            yield from walk(acts + [p_(p, "check")])
            if st.n_raises == 0:
                for f in sizes:
                    amt = max(round(f * pot * 2) / 2, 1.0)
                    if amt < behind - 1e-9:
                        yield from walk(acts + [p_(p, "bet", amount=amt)])
        else:
            yield from walk(acts + [p_(p, "call")])
            if len(st.active) > 2:
                yield from walk(acts + [p_(p, "fold")])
            if st.n_raises < 2:
                to = st.cur_bet + round(0.75 * (pot + st.cur_bet - st.bets[p]) * 2) / 2
                if to < behind - 1e-9:
                    yield from walk(acts + [p_(p, "raise", to=to)])
    return walk([])


def short(acts):
    return " ".join(a["pos"] + ":" + a["action"][0] + (f"{a.get('amount', a.get('to')):g}" if "amount" in a or "to" in a else "")
                    for a in acts)


def live_mask(board):
    m = np.ones(N)
    for c in board:
        m = m * (1.0 - A.MEMBER_FULL[c])
    return m


def mw_street_ranges(n, stack, preflop, board, streets):
    """The 3-way rules through the earlier streets, as Advisor's multiway path (no hero: everyone floored): the ranges
    starting each street, with the replay."""
    pos = table(n)
    ranges = dict(A.preflop_ranges(preflop, pos, stack))
    names = [s for s in ("flop", "turn", "river") if s in streets]
    rep = A.replay_hand(pos, stack, {"preflop": preflop, **streets}, 1 + len(names)) if names else []
    stk = A._table_stacks(pos, stack)
    slow = {}
    for st in rep[1:]:
        before = dict(ranges)
        bk = board[:{"flop": 3, "turn": 4, "river": 5}[st.name]]
        A._narrow_street(ranges, bk, st, stk, None, (), slow)
        for p in st.players_start:
            if p in ranges and p in before:
                ranges[p] = A._post_floor(before[p], ranges[p], bk, p, st.name, [])
    return ranges, slow


def census_mw(args, out):
    for name, n, stack, pre in THREE_WAY:
        for f in FLOPS[:args.flops]:
            for street, priors in (("flop", [{}]),
                                   ("turn", [{"flop": fl} for fl in prior_lines(n, stack, pre, 1)]),
                                   ("river", [{"flop": fl, "turn": tl} for fl in prior_lines(n, stack, pre, 1)
                                              for tl in prior_lines(n, stack, pre, 2, {"flop": fl})])):
                for rest in cards_for(f, street, args.cards):
                    board = cards(f + rest)
                    for prior in priors:
                        t0 = time.time()
                        before, slow0 = mw_street_ranges(n, stack, pre, board, prior)
                        n_st = 1 + len(prior) + 1
                        k = 0
                        for acts, closed in lines(n, stack, {"preflop": pre, **prior}, street, n_st):
                            if closed and street == "river":
                                continue
                            st = A.replay_hand(table(n), stack, {"preflop": pre, **prior, street: acts}, n_st)[-1]
                            if len(st.players_start) != 3:
                                continue
                            ranges = dict(before)
                            A._narrow_street(ranges, board, st, A._table_stacks(table(n), stack), None, (), dict(slow0))
                            live = live_mask(board)
                            acted = {a["pos"] for a in acts}
                            thin = [(p, round(float((ranges[p] * live).sum()), 2), round(float((before[p] * live).sum()), 2))
                                    for p in st.players_start if p in acted and p in st.active
                                    and (ranges[p] * live).sum() < pf.MIN_LINE_COMBOS]
                            if thin:
                                k += 1
                                out.write(json.dumps({"kind": "3way", "spot": name, "board": f + rest, "street": street,
                                                      "prior": {s: short(v) for s, v in prior.items()}, "line": short(acts),
                                                      "closed": closed, "thin": thin}) + "\n")
                        out.flush()
                        print(f"3way {name} {f}{rest} {street} {[short(v) for v in prior.values()]}: {k} thin lines "
                              f"({time.time() - t0:.1f} s)", flush=True)


def prior_lines(n, stack, pre, si, prior=None):
    """The earlier street's lines the grid follows: checked through, and the first to bet called by everyone."""
    prior = prior or {}
    st = A.replay_hand(table(n), stack, {"preflop": pre, **prior}, si + 1 if False else 1 + len(prior))[-1]
    seats = list(st.active)
    order = [q for q in pf.postflop_order(table(n)) if q in seats]
    pot = st.pot_start + sum(st.bets.values()) if not prior else sum(
        A.replay_hand(table(n), stack, {"preflop": pre, **prior}, 1 + len(prior))[-1].inv_start.values())
    if prior:
        last = A.replay_hand(table(n), stack, {"preflop": pre, **prior}, 1 + len(prior))[-1]
        pot = last.pot_start + sum(last.bets.values())
    else:
        pot = st.pot_start + sum(st.bets.values())
    amt = round(0.5 * pot * 2) / 2
    return [[p_(q, "check") for q in order],
            [p_(order[0], "check")] * 0 + [p_(order[0], "bet", amount=amt)] + [p_(q, "call") for q in order[1:]]]


def cards_for(f, street, k):
    if street == "flop":
        return [""]
    turns = next_cards(f, TURNS, k)
    if street == "turn":
        return turns
    return [t + r for t in turns[:1] for r in next_cards(f + t, RIVERS, k)]


# ------------------------------------------------------------------ the fills
def fills(a, b, target=None):
    """The candidates for a thin line: `a` its range, `b` the street's start (both on the live cards), every one but
    `current` at target combos."""
    T = float(pf.MIN_LINE_COMBOS if target is None else target)
    out = {"raw": a, "current": pf.thin_line(a, b)[0]}

    def top_up(r):
        need = T - r.sum()
        room = np.clip(b - r, 0.0, None)
        return r + need / room.sum() * room if need > 1e-9 and room.sum() > 1e-9 else r
    out["uniform"] = top_up(a)
    lo, hi = 1.0, 1e9
    for _ in range(80):
        m = (lo * hi) ** 0.5
        lo, hi = (m, hi) if np.minimum(m * a, b).sum() < T else (lo, m)
    out["scale"] = top_up(np.minimum(hi * a, b))
    w = np.clip(a / np.where(b > 0, b, 1.0), 0.0, 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if (b * w ** m).sum() > T else (lo, m)
    out["temper"] = top_up(b * w ** hi)
    return out


def tv(x, y):
    return 0.5 * float(np.abs(x / x.sum() - y / y.sum()).sum())


# ------------------------------------------------------------------ the reference
class Root:
    """While a solve is built, its root's actions filtered: MW._actions' first call of the build."""

    def __init__(self, pred):
        self.pred, self.first = pred, True

    def __enter__(self):
        self.real = MW._actions

        def actions(*a, **k):
            out = self.real(*a, **k)
            if self.first:
                self.first = False
                out = self.pred(out)
            return out
        MW._actions = actions
        return self

    def __exit__(self, *e):
        MW._actions = self.real


def street_cfg(street, **kw):
    c = {"flop": dict(A.MW_LATER, buckets=12, bucket_river_step=3, bet={"flop": list(SIZES), "turn": 0.5, "river": 0.5}),
         "turn": dict(A.MW_LATER, buckets=12, bet={"flop": 0.33, "turn": list(SIZES), "river": 0.5}),
         "river": dict(bet={"flop": 0.33, "turn": 0.5, "river": list(SIZES)})}[street]
    return MW.MWConfig(**{**c, **kw})


def node_args(cur, stack):
    """The 3-way solve's root at a node of the street (Advisor._multiway_solve, even stacks, nobody all-in)."""
    players = [q for q in cur.order if q in cur.active]
    S = min(stack - cur.inv_start[q] for q in players)
    bets = [cur.bets[q] for q in players]
    pot = cur.pot_start + sum(b for q, b in cur.bets.items() if q not in cur.active)
    pending = {i for i, q in enumerate(players) if q not in cur.acted}
    return players, S, bets, pot, pending


def nearest(act):
    """The root's action of the tree for the real one `act`."""
    want = {"check": "check", "call": "call", "fold": "fold", "bet": "bet", "raise": "raise"}[act["action"]]
    to = act.get("to", act.get("amount"))

    def pick(rows):
        same = [x for x in rows if x[0].split()[0] == want]
        return min(same, key=lambda x: abs((x[1] or 0.0) - (to or 0.0))) if same else None
    return pick


def forced(board, rs, pot, S, bets, hp, n_raises, pending, cfg, act, iters):
    """Player hp's EV per combo for the action `act` at this node, with the solve's only way from its root that action:
    its whole range `rs[hp]` takes it, so everyone after it plays against that range."""
    pick = nearest(act)
    with Root(lambda rows: [pick(rows)]):
        sol = MW.MultiwaySolver(board, rs, pot, S, bets=bets, to_act=hp, n_raises=n_raises, pending=pending | {hp}, cfg=cfg)
    sol.run(iters=iters)
    return sol.expand(sol.action_evs()[0])


def otherwise(board, rs, pot, S, bets, hp, n_raises, pending, cfg, act, iters):
    """...and its best EV per combo by anything else there: the same node without that action, `rs[hp]` the rest."""
    pick = nearest(act)

    def drop(rows):
        x = pick(rows)
        return [r for r in rows if r[0] != x[0]]
    with Root(drop):
        sol = MW.MultiwaySolver(board, rs, pot, S, bets=bets, to_act=hp, n_raises=n_raises, pending=pending | {hp}, cfg=cfg)
    sol.run(iters=iters)
    return sol.expand(sol.action_evs().max(axis=0))


def constrained(board, others, b, target, cur, stack, p, act, street, iters=100, steps=6, seed=None, log=None):
    """The line's range if player p takes it with at least `target` combos of b and everyone plays their best given that:
    the equilibrium of the game with that constraint, by fictitious play. Each step solves the line with p on the range
    so far and the node without the line with the rest of b; p's best reply is every combo the line suits better than
    anything else (a line that needs no constraint is its range) and, under `target`, the combos giving up least by it
    (EV of the line less their best EV otherwise), the range moving to it by 2 / (t + 2)."""
    players, S, bets, pot, pending = node_args(cur, stack)
    hp = players.index(p)
    cfg = street_cfg(street)
    tiny = 1e-3 * b
    R = np.minimum(b, seed if seed is not None else fills(others[p], b, target)["uniform"])
    for t in range(steps):
        rs = [others[q] for q in players]
        rs[hp] = np.maximum(R, tiny)
        line = forced(board, rs, pot, S, bets, hp, cur.n_raises, pending, cfg, act, iters)
        rs[hp] = np.maximum(b - R, tiny)
        alt = otherwise(board, rs, pot, S, bets, hp, cur.n_raises, pending, cfg, act, iters)
        g = (alt - line)
        order = np.argsort(np.where(b > 0, g, np.inf), kind="stable")
        c = np.cumsum(b[order])
        k = max(int(np.searchsorted(c, target)), int(((g < 0) & (b > 0)).sum()))  # a bucket's EV reaches combos not in b
        br = np.zeros(N)
        br[order[:k]] = b[order[:k]]
        if k < N and (c[k - 1] if k else 0.0) < target:
            br[order[k]] = target - (c[k - 1] if k else 0.0)
        eta = 2.0 / (t + 2)
        R1 = (1 - eta) * R + eta * br
        if log is not None:
            log.append({"t": t, "move": round(tv(R1, R), 4), "cost": round(float((br * g).sum() / br.sum()), 3), "combos": round(float(br.sum()), 1),
                        "bind": bool(br.sum() <= target + 1e-6)})
        R = R1
    return R


def replay_at(n, stack, streets, street, acts, n_st):
    return A.replay_hand(table(n), stack, {**streets, street: acts}, n_st)[-1]


def after_line(n, stack, streets, street, acts, before, board, n_st, slow):
    st = replay_at(n, stack, streets, street, acts, n_st)
    r = dict(before)
    A._narrow_street(r, board, st, A._table_stacks(table(n), stack), None, (), dict(slow))
    return r, st


def decide(board, players, rs, pot, S, bets, hp, n_raises, pending, street, iters):
    """The solve hero's advice comes from at a node: per combo (1326) its range, mix and action EVs."""
    cfg = street_cfg(street)
    sol = MW.MultiwaySolver(board, rs, pot, S, bets=bets, to_act=hp, n_raises=n_raises, pending=pending | {hp}, cfg=cfg)
    sol.run(iters=iters)
    return sol.root.names, sol.expand(sol.reach0[hp, 0]), \
        np.stack([sol.expand(v) for v in sol.root.avg_strategy()[0]]), np.stack([sol.expand(v) for v in sol.action_evs()])


def given_up(ref, cand):
    """What hero's mix from `cand` gives up by the reference's EVs (bb per combo of its range); the share of the range
    whose most frequent action changes."""
    names, r, sg_r, ev = ref
    assert names == cand[0]
    sg = cand[2]
    w = r / r.sum()
    best = ev.max(axis=0)
    ok = r > 0
    loss = float((w * (best - (sg * ev).sum(axis=0)))[ok].sum())
    flip = float(w[ok][sg_r.argmax(axis=0)[ok] != sg.argmax(axis=0)[ok]].sum())
    return loss, flip


class Captured(BaseException):
    pass


def census_hu(args, out):
    adv = native.advisor("fast", library=False)
    real = A._post_floor
    for name, n, stack, pre in HEADS_UP:
        for f in FLOPS[:args.flops]:
            for street, priors in (("flop", [{}]),
                                   ("turn", [{"flop": fl} for fl in prior_lines(n, stack, pre, 1)]),
                                   ("river", [{"flop": fl, "turn": tl} for fl in prior_lines(n, stack, pre, 1)
                                              for tl in prior_lines(n, stack, pre, 2, {"flop": fl})])):
                if street == "river":
                    continue    # (a heads-up street's floor is read at the next street: none after the river)
                for rest in cards_for(f, street, args.cards):
                    board = f + rest
                    nxt = next_cards(board, ["3d", "Ah", "Tc", "6c"], 1)[0]
                    for prior in priors:
                        t0 = time.time()
                        k = 0
                        n_st = 1 + len(prior) + 1
                        for acts, closed in lines(n, stack, {"preflop": pre, **prior}, street, n_st):
                            if not closed:
                                continue
                            seen = {}

                            def hook(before, after, bd, pos, st_name, notes, off=None):
                                if st_name == street:
                                    live = np.ones(N)
                                    for c in bd:
                                        live = live * (1.0 - A.MEMBER_FULL[c])
                                    seen[pos] = (float((after * live).sum()), float((before * live).sum()))
                                    if len(seen) == 2:
                                        raise Captured()
                                return real(before, after, bd, pos, st_name, notes, off)
                            A._post_floor = hook
                            streets = {"preflop": pre, **prior, street: acts}
                            order = [q for q in pf.postflop_order(table(n)) if q in A.replay_hand(table(n), stack, streets, n_st)[-1].active]
                            hole = next_cards(board + nxt, ["2d", "3h", "4s", "5c", "6d", "7h"], 2)
                            st = {"players": n, "hero": order[0], "hole": "".join(hole), "board": board + nxt, "stack": stack,
                                  **streets, ("turn" if street == "flop" else "river"): []}
                            try:
                                adv.advise(st)
                            except Captured:
                                pass
                            finally:
                                A._post_floor = real
                            thin = [(p, round(v[0], 2), round(v[1], 2)) for p, v in seen.items() if v[0] < pf.MIN_LINE_COMBOS]
                            if thin:
                                k += 1
                                out.write(json.dumps({"kind": "hu", "spot": name, "board": board + nxt, "street": street,
                                                      "prior": {s: short(v) for s, v in prior.items()}, "line": short(acts),
                                                      "closed": True, "thin": thin}) + "\n")
                        out.flush()
                        print(f"hu {name} {board} {street} {[short(v) for v in prior.values()]}: {k} thin lines "
                              f"({time.time() - t0:.1f} s)", flush=True)


# ------------------------------------------------------------------ one line against the reference
SPOTS = {("3way", x[0]): x for x in THREE_WAY}
SPOTS.update({("hu", x[0]): x for x in HEADS_UP})


def find(n, stack, streets, street, n_st, text):
    return next(a for a, _ in lines(n, stack, streets, street, n_st) if short(a) == text)


def priors_of(ev, n, stack, pre):
    prior = {}
    for s_, text in ev["prior"].items():
        cands = prior_lines(n, stack, pre, 0, prior)
        prior[s_] = next(c for c in cands if short(c) == text)
    return prior


def hook_hu(adv, n, stack, streets, street, board, nxt, n_st, order):
    """Advisor's heads-up walk up to the floor on `street`: {seat: (street-start range, the line's range, live mask)}."""
    seen = {}
    real = A._post_floor

    def hook(before, after, bd, pos, st_name, notes, off=None):
        if st_name == street:
            seen[pos] = (before.copy(), after.copy(), live_mask(bd))
            if len(seen) == 2:
                raise Captured()
        return real(before, after, bd, pos, st_name, notes, off)
    A._post_floor = hook
    hole = next_cards(board + nxt, ["2d", "3h", "4s", "5c", "6d", "7h"], 2)
    st = {"players": n, "hero": order[0], "hole": "".join(hole), "board": board + nxt, "stack": stack, **streets,
          ("turn" if street == "flop" else "river"): []}
    try:
        adv.advise(st)
    except Captured:
        pass
    finally:
        A._post_floor = real
    return seen


def one(ev, args, adv=None):
    """Event `ev`: the candidates' ranges for its thin player against the constrained references, by total variation and
    by what the next decision of someone else gives up."""
    n, stack = SPOTS[(ev["kind"], ev["spot"])][1:3]
    pre = SPOTS[(ev["kind"], ev["spot"])][3]
    prior = priors_of(ev, n, stack, pre)
    street, cs = ev["street"], ev["board"]
    n_st = 2 + len(prior)
    streets = {"preflop": pre, **prior}
    acts = find(n, stack, streets, street, n_st, ev["line"])
    rounds = {"flop": 6, "turn": 8, "river": 10}[street]
    bk = cs[:rounds]
    board = cards(bk)
    live = live_mask(board)
    p = ev["p"]
    idx = max(i for i, a in enumerate(acts) if a["pos"] == p)
    cur = replay_at(n, stack, streets, street, acts[:idx], n_st)
    end = replay_at(n, stack, streets, street, acts, n_st)
    if ev["kind"] == "3way":
        before, slow = mw_street_ranges(n, stack, pre, board, prior)
        at = after_line(n, stack, streets, street, acts[:idx], before, board, n_st, slow)[0]
        fin = after_line(n, stack, streets, street, acts, before, board, n_st, slow)[0]
    else:
        order = [q for q in pf.postflop_order(table(n)) if q in end.active]
        got = hook_hu(adv, n, stack, {**streets, street: acts}, street, bk, cs[rounds:rounds + 2], n_st, order)
        before = {q: got[q][0] for q in got}
        fin = {q: got[q][1] for q in got}
        at = dict(before)
        A._narrow_street(at, board, cur, A._table_stacks(table(n), stack), None, (), {}, check_whole=True)
    b = before[p] * live
    a = fin[p] * live
    c = float(a.sum())
    if b.sum() <= pf.MIN_LINE_COMBOS or c <= 0:
        return {"skip": f"street range {b.sum():.1f}, line {c:.1f}"}

    def filled(rs, skip=()):
        out = {}
        for q in rs:
            x, y = rs[q] * live, before[q] * live
            out[q] = pf.thin_line(x, y)[0] if q not in skip and x.sum() < pf.MIN_LINE_COMBOS else x
        return out
    others = filled(at, (p,))
    t0 = time.time()
    refs, logs = {}, {}
    for name, T in (("8", pf.MIN_LINE_COMBOS), ("5", max(5.0, min(c, 5.0))), ("own", max(c, 2.0))):
        logs[name] = []
        refs[name] = constrained(board, others, b, T, cur, stack, p, acts[idx], street, iters=args.iters,
                                 seed=fills(a, b, T)["uniform"], log=logs[name])
    t1 = time.time()
    cand = fills(a, b)
    cand["whole"] = b
    cand["scale5"] = fills(a, b, max(5.0, c))["scale"]
    cand["scale6"] = fills(a, b, max(6.0, c))["scale"]
    sc = A._street_info(tuple(board))[0]
    strength = {k: float((v * sc).sum() / v.sum()) for k, v in {**cand, **{"ref" + k: v for k, v in refs.items()}}.items()}
    # the decision: the next player who isn't p, at this node or the next street's first
    if not end.closed:
        hero = end.to_act
        if hero == p:
            return {"skip": "the thin player acts next"}
        dboard, dstreet, node = board, street, end
        d_live = live
        players, S, bets, pot, pending = node_args(node, stack)
        n_raises = node.n_raises
    else:
        nxt = next_cards(bk, ["3d", "Ah", "Tc", "6c"], 1)[0] if ev["kind"] == "3way" else cs[rounds:rounds + 2]
        dboard = cards(bk + nxt)
        dstreet = {"flop": "turn", "turn": "river"}[street]
        d_live = live_mask(dboard)
        players = [q for q in pf.postflop_order(table(n)) if q in end.active]
        hero = next(q for q in players if q != p)
        S = min(stack - end.inv_start[q] - end.bets[q] for q in players)
        bets, n_raises = [0.0] * len(players), 0
        pot = end.pot_start + sum(end.bets.values())
        pending = set(range(len(players)))
        for q in players[:players.index(hero)]:
            pending.discard(players.index(q))
    hp = players.index(hero)
    base = filled(fin, (hero,))
    rs0 = [base[q] for q in players]

    def run(R, iters=None):
        rs = list(rs0)
        rs[players.index(p)] = R
        return decide(dboard, players, rs, pot, S, bets, hp, n_raises, pending, dstreet, iters or args.eval_iters)
    res = {k: run(v) for k, v in {**cand, **{"ref" + k: v for k, v in refs.items()}}.items()}
    pot_now = pot + sum(bets)
    out = {"p": p, "c": round(c, 2), "b": round(float(b.sum()), 2), "hero": hero, "pot": round(pot_now, 2),
           "strength": {k: round(v, 3) for k, v in strength.items()}, "secs": [round(t1 - t0, 1), round(time.time() - t1, 1)],
           "move": {k: logs[k][-1]["move"] for k in logs}, "cost": {k: logs[k][-1]["cost"] for k in logs},
           "ref_combos": {k: round(float(v.sum()), 1) for k, v in refs.items()},
           "hero_range": round(float((rs0[hp] * d_live).sum()), 1),
           "combos": {k: round(float(v.sum()), 1) for k, v in cand.items()}}
    for rn, R in refs.items():
        out["tv" + rn] = {k: round(tv(v, R), 4) for k, v in cand.items()}
        out["loss" + rn] = {}
        out["flip" + rn] = {}
        for k in cand:
            loss, flip = given_up(res["ref" + rn], res[k])
            out["loss" + rn][k] = round(loss / pot_now * 100, 4)
            out["flip" + rn][k] = round(flip, 4)
        again = run(R, args.eval_iters // 2)
        out["noise" + rn] = round(given_up(res["ref" + rn], again)[0] / pot_now * 100, 4)
    return out


def sample(paths, k, seed):
    """Up to k events from each cell (kind, spot, street, how thin, the last action a bet or raise or not), one thin player
    each, not a blind's check line left whole."""
    import random
    rng = random.Random(seed)
    cells = {}
    for path in paths:
        for row in open(path):
            e = json.loads(row)
            for p, c, b in e["thin"]:
                if b <= pf.MIN_LINE_COMBOS:
                    continue
                last = [x for x in e["line"].split() if x.startswith(p + ":")][-1].split(":")[1][0]
                thin = 0 if c < 2.5 else 1 if c < 5 else 2
                cells.setdefault((e["kind"], e["spot"], e["street"], thin, last in "br"), []).append({**e, "p": p})
    out = []
    for key in sorted(cells):
        rows = cells[key]
        rng.shuffle(rows)
        out += rows[:k]
    rng.shuffle(out)
    return out


def reference(args):
    evs = sample(args.census, args.per_cell, args.seed)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(x)["key"] for x in open(args.out)}
    key = lambda e: "|".join((e["kind"], e["spot"], e["board"], e["street"], json.dumps(e["prior"]), e["line"], e["p"]))
    todo = [e for e in evs if key(e) not in done]
    print(f"{len(evs)} events, {len(todo)} to do", flush=True)
    adv = None
    n = 0
    with open(args.out, "a") as out:
        for e in todo:
            if args.max and n >= args.max:
                break
            t0 = time.time()
            if e["kind"] == "hu" and adv is None:
                adv = native.advisor("fast", library=False)
            try:
                r = one(e, args, adv)
            except Exception as x:  # noqa: BLE001
                import traceback
                r = {"skip": "error " + repr(x)[:200]}
                traceback.print_exc()
            out.write(json.dumps({"key": key(e), **{k: e[k] for k in ("kind", "spot", "board", "street", "line", "closed")}, **r}) + "\n")
            out.flush()
            n += 1
            print(key(e)[:90], r.get("skip") or (r["c"], r["loss8"]["current"], r["loss8"]["scale"], r["loss5"]["current"], r["loss5"]["scale"]), f"{time.time() - t0:.0f} s", flush=True)


def report(path, boots=800):
    """Mean % of the pot the next decision gives up by each candidate against each reference, the candidates less the
    current filler with a 95% bootstrap interval over spot and flop (a flop's lines aren't independent)."""
    rows = {}
    for row in open(path):
        e = json.loads(row)
        if "loss8" in e and e["c"] < pf.MIN_LINE_COMBOS:  # a time-budget solve can leave a census line above it
            rows[e["key"]] = e
    rows = list(rows.values())
    rng = np.random.default_rng(1)

    def table(name, rs):
        if len(rs) < 3:
            return
        print(f"\n{name}: {len(rs)} lines")
        for r in ("8", "5", "own"):
            loss = {c: np.array([e["loss" + r][c] for e in rs]) for c in ("raw", "current", "scale", "uniform", "temper", "whole")}
            print(f"  T={r}: " + "  ".join(f"{c} {v.mean():.2f}" for c, v in loss.items()))
            spots = sorted({(e["spot"], e["board"][:6]) for e in rs})
            at = [[i for i, e in enumerate(rs) if (e["spot"], e["board"][:6]) == k] for k in spots]
            for c in ("raw", "scale", "whole"):
                d = loss[c] - loss["current"]
                means = [d[np.concatenate([at[j] for j in rng.integers(len(at), size=len(at))])].mean() for _ in range(boots)]
                print(f"    {c} - current {d.mean():+.3f} [{np.percentile(means, 2.5):+.3f}, {np.percentile(means, 97.5):+.3f}], median {np.median(d):+.3f}")
    table("all", rows)
    for kind in ("hu", "3way"):
        for street in ("flop", "turn", "river"):
            table(f"{kind} {street}", [e for e in rows if e["kind"] == kind and e["street"] == street])
    table("street range under 40 combos", [e for e in rows if e["b"] < 40])
    table("street range 40 combos or more", [e for e in rows if e["b"] >= 40])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["census", "reference", "report"])
    ap.add_argument("--census", nargs="*", default=[])
    ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--eval-iters", type=int, default=150)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default="thin_census.jsonl")
    ap.add_argument("--flops", type=int, default=10)
    ap.add_argument("--cards", type=int, default=2)
    ap.add_argument("--only", choices=["hu", "3way"])
    a = ap.parse_args()
    if a.mode == "report":
        return report(a.out)
    if a.mode == "reference":
        return reference(a)
    with open(a.out, "a") as out:
        if a.only != "hu":
            census_mw(a, out)
        if a.only != "3way":
            census_hu(a, out)


if __name__ == "__main__":
    main()
