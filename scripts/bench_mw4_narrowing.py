"""4-way flops: are the next street's ranges better read through the flop solve your advice came from (Advisor._mw_walk,
as a 3-way flop does) than through the betting rules (advisor._narrow_street)?

  python scripts/bench_mw4_narrowing.py ranges  --dir reports/mw4_narrowing          # the solves; ~6 min a spot-flop
  python scripts/bench_mw4_narrowing.py add --add "a50 b,a39 b"                      # lines with a size of their own
  python scripts/bench_mw4_narrowing.py answers --dir reports/mw4_narrowing          # each range set's next decisions
  python scripts/bench_mw4_narrowing.py report  --dir reports/mw4_narrowing          # the tables

Candidates, per line (a flop played to its end, or to a later flop decision of yours):
  rules      advisor._narrow_street and nothing else
  fast       the walk through Fast's solve (MW4_FAST_STREETS, FAST_ITERS iterations: what it reaches in its 5 s), smoothing
             and trust as the advisor plays them; fast-raw: followed fully, no smoothing
  quick      the same through a quick answer's solve (MW4_QUICK_FLOP, QUICK_ITERS iterations)
  accurate   the same through Accurate's (MW4_STREETS, ACCURATE_ITERS)
  ref-early  the reference at half its iterations: the noise floor of everything below

The reference can't be Accurate: it plays the same game as Fast (one size, nobody but the root's player raises), so it
would agree with Fast about whatever that game gets wrong. Its game has a raise for everyone on the flop and two sizes
there (33 and 75%; a line sized off the tree has that size in its reference: SIZED), 8 buckets on every third river,
solved from the flop's start to about 0.2% of the pot, and is followed fully and unsmoothed.

Two measures. Ranges: total variation between the normalized ranges on the turn's live cards, per player still in, against
the reference. Answers: your next decision solved from each candidate's ranges, its mix valued by the reference ranges'
own solve of that decision (the reference's mix less the candidate's, % of the pot), for hands drawn from your range by
weight; `ref-early` there is how much a change of ranges too small to matter moves the number. Results:
docs/multiway_research.md, "4-way flops".
"""
import argparse
import dataclasses
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gto import native  # noqa: E402  (first: holds BLAS to one thread)
import numpy as np      # noqa: E402
import gto.advisor as A  # noqa: E402
from gto import preflop as pf  # noqa: E402
from gto import multiway_solver as MW  # noqa: E402
from gto.cards import cards, cards_str, ALL_COMBOS  # noqa: E402
from gto.solver import MEMBER_FULL  # noqa: E402

N = 1326
REF_ITERS, FAST_ITERS, QUICK_ITERS, ACCURATE_ITERS = 250, 75, 30, 400
HANDS = 10   # hands per decision
SIZED = {"a39 b": (0.39, 0.75), "a25 b": (0.25, 0.75), "a50 b": (0.5, 1.0)}   # those lines' references' flop sizes


def p_(pos, action, **kw):
    return {"pos": pos, "action": action, **kw}


def fold(*seats):
    return [p_(s, "fold") for s in seats]


# (name, table size, stack, preflop actions)
SPOTS = {
    "limp-raise": (7, 100, [p_("UTG", "call"), p_("LJ", "raise", to=4.0), p_("HJ", "call")] + fold("CO", "BTN", "SB")
                   + [p_("BB", "call"), p_("UTG", "call")]),
    "hj-open": (6, 100, fold("UTG") + [p_("HJ", "raise", to=2.5), p_("CO", "call"), p_("BTN", "call"), p_("SB", "fold"),
                                       p_("BB", "call")]),
    "utg-open": (6, 100, [p_("UTG", "raise", to=2.5), p_("HJ", "fold"), p_("CO", "call"), p_("BTN", "call"),
                          p_("SB", "fold"), p_("BB", "call")]),
    "limped": (6, 100, [p_("UTG", "call"), p_("HJ", "call"), p_("CO", "fold"), p_("BTN", "call"), p_("SB", "fold"),
                        p_("BB", "check")]),
    "short": (6, 40, fold("UTG") + [p_("HJ", "raise", to=2.5), p_("CO", "call"), p_("BTN", "call"), p_("SB", "fold"),
                                    p_("BB", "call")]),
}
FLOPS = {"limp-raise": ["2s4d6c", "Kh7d2c", "Jh9h4c", "As9d4h"],
         "hj-open": ["Kh7d2c", "Jh9h4c", "8s8d3c", "7c5c2d"],
         "utg-open": ["As9d4h", "Td8s6s", "Qd6c5s", "Kh7d2c"],
         "limped": ["Qd6c5s", "9s8h3d", "Ah9h4h", "Kh7d2c"],
         "short": ["Kh7d2c", "Td8s6s"]}
TURNS = ["Qs", "3c", "6h", "9d"]     # the first two a flop leaves


def lines(order, pot):
    """{name: (hero, the flop's actions, kind)}: a, b, c, d in acting order. Kinds: tree (the solve's tree holds the line),
    near (its 33% stands for a bet near it), far (for a bet well over it), raise (no such action in the 4-way tree)."""
    a, b, c, d = order
    bet = lambda p, f: p_(p, "bet", amount=max(round(f * pot * 2) / 2, 1.0))
    x = lambda *ps: [p_(q, "check") for q in ps]
    c_ = lambda q: p_(q, "call")
    b33, b39, b75 = bet(a, 0.33), bet(a, 0.39), bet(a, 0.75)
    return {
        "a33 b": (a, [b33, c_(b)] + fold(c, d), "tree"),
        "a33 d": (a, [b33] + fold(b, c) + [c_(d)], "tree"),
        "a33 bc": (a, [b33, c_(b), c_(c)] + fold(d), "tree"),
        "ax b33 a": (a, x(a) + [bet(b, 0.33)] + fold(c, d) + [c_(a)], "tree"),
        "axx c33 a": (a, x(a, b) + [bet(c, 0.33)] + fold(d) + [c_(a)] + fold(b), "tree"),
        "axxx d33 a": (a, x(a, b, c) + [bet(d, 0.33), c_(a)] + fold(b, c), "tree"),
        "axxx d33 d": (d, x(a, b, c) + [bet(d, 0.33), c_(a)] + fold(b, c), "tree"),
        "xxc33 c": (c, x(a, b) + [bet(c, 0.33)] + fold(d) + [c_(a)] + fold(b), "tree"),
        "xxxx": (a, x(a, b, c, d), "tree"),
        "a25 b": (a, [bet(a, 0.25), c_(b)] + fold(c, d), "under"),
        "a39 b": (a, [b39, c_(b)] + fold(c, d), "near"),
        "a50 b": (a, [bet(a, 0.5), c_(b)] + fold(c, d), "mid"),
        "a75 b": (a, [b75, c_(b)] + fold(c, d), "far"),
        "a33 b-raise": (a, [b33, p_(b, "raise", to=3 * b33["amount"])] + fold(c, d) + [c_(a)], "raise"),
    }


def table(n):
    return pf.Table(pf.SEATS[n])


def state(spot, board, hero, hole, flop):
    """Your decision after `flop`: on it, with a flop's board; with a turn card, the turn's first (the seats before you
    check)."""
    n, stack, pre = SPOTS[spot]
    turn = []
    if len(cards(board)) > 3:
        st = A.replay_hand(table(n), stack, {"preflop": pre, "flop": flop, "turn": []}, 3)[-1]
        alive = [p for p in st.order if p in st.active]
        turn = [p_(q, "check") for q in alive[:alive.index(hero)]] if hero in alive else []
    return {"players": n, "stack": stack, "preflop": pre, "board": board, "hero": hero, "hole": hole, "flop": list(flop),
            "turn": turn}


class Stop(BaseException):
    """Out of Advisor.advise, past its catch-all, once the ranges are read."""


def read_ranges(adv, st):
    """The flop's narrowing as Advisor._advise does it at state st: the rules' ranges, the walk's (the advisor's own
    smoothing and trust, and fully followed unsmoothed), the ranges entering the flop."""
    real, trust, smooth, box = A.Advisor._mw_walk, A.trust_of, adv.prior_smoothing, {}

    def walk(ranges, before, board, st_, *a, **k):
        if st_.name != "flop":
            return real(adv, ranges, before, board, st_, *a, **k)
        rules, plain, full = dict(ranges), dict(ranges), dict(ranges)
        ok = real(adv, plain, before, board, st_, *a, **k)
        adv.prior_smoothing, A.trust_of = 0.0, lambda e, s: 1.0
        try:
            real(adv, full, before, board, st_, *a, **k)
        finally:
            adv.prior_smoothing, A.trust_of = smooth, trust
        box.update(rules=rules, walk=plain, raw=full, before=dict(before), ok=bool(ok))
        raise Stop()
    adv._mw_walk = walk
    try:
        adv.advise(dict(st))
    except Stop:
        pass
    finally:
        del adv._mw_walk
    return box


class FlopRaises:
    """While a reference is built: a bet and a raise on the flop for everyone (the 4-way game's one bet later on)."""

    def __enter__(self):
        self.real = MW._actions

        def actions(cfg, npl, street, *a, **k):
            return self.real(dataclasses.replace(cfg, max_raises=2 if street == "flop" else 1), npl, street, *a, **k)
        MW._actions = actions

    def __exit__(self, *e):
        MW._actions = self.real


def reference(flop, rs, pot, S, sizes=(0.33, 0.75)):
    cfg = MW.MWConfig(bet={"flop": list(sizes), "turn": 0.5, "river": 0.5}, max_raises=1, later="buckets", buckets=8,
                      bucket_river_step=3)
    with FlopRaises():
        return MW.MultiwaySolver(flop, rs, pot, S, cfg=cfg)


def ref_walk(sol, players, acts):
    """The solve's ranges through `acts` from its root, followed fully and unsmoothed, and each action's share of its
    player's range; None where the tree lacks an action."""
    R = sol.reach0[:, 0, :].astype(np.float64).copy()
    node, shares = sol.root, []
    for a in acts:
        if not isinstance(node, MW.Decision):
            break
        i = A._mw_pick(node, a)
        if i is None or players[node.player] != a["pos"]:
            return None
        p = node.player
        was = R[p].sum()
        R[p] = R[p] * node.avg_strategy()[0][i]
        shares.append([a["pos"], a["action"], float(R[p].sum() / max(was, 1e-12))])
        node = node.children[i]
    return {q: sol.expand(R[k]) for k, q in enumerate(players)}, shares


def hole_of(r, flop):
    """A hand of the range, for the solves' seat at the table (the ranges don't depend on it)."""
    m = r * np.prod([1.0 - MEMBER_FULL[c] for c in flop], axis=0)
    return cards_str(ALL_COMBOS[int(np.argmax(m))])


def speeds():
    fast = native.advisor("fast", library=False)
    fast.mw_max_iters, fast.mw_budget = FAST_ITERS, 1e6
    acc = native.advisor("accurate", library=False)
    acc.mw_max_iters, acc.time_budget = ACCURATE_ITERS, 1e6
    quick = native.advisor("fast", library=False)
    quick.mw_max_iters, quick.mw_budget = QUICK_ITERS, 1e6
    A.MW4_WALK_OFF, A.MW4_WALK_CHECKS = 1e9, True   # (the walk everywhere: where it stops is what this measures)
    return {"fast": fast, "accurate": acc, "quick": quick}


def hero_decisions(spot, flop, acts, hero):
    """The flop's actions before each decision of hero's with three or more still in (where a multiway solve stores)."""
    n, stack, pre = SPOTS[spot]
    out = []
    for i, a in enumerate(acts):
        if a["pos"] == hero:
            st = A.replay_hand(table(n), stack, {"preflop": pre, "flop": acts[:i]}, 2)[-1]
            if len(st.active) >= 3:
                out.append(i)
    return out


def solve_line(spot, flop, filler, nm, line, order, enter, holes, advs, late, early, arrays, meta):
    """A line's ranges under every candidate, into arrays and meta."""
    def keep(cand, rg):
        for p, r in rg.items():
            arrays[f"{nm}|{cand}|{p}"] = np.asarray(r, dtype=np.float32)

    t1 = time.time()
    hero, acts, kind = line
    hole = holes[hero]
    info = {"hero": hero, "kind": kind, "acts": acts, "hole": hole, "ok": {}, "solves": {}}
    for sp, adv in advs.items():
        # the flop's decisions as you'd have asked them (solved, stored in _mw_lines), then the turn's ranges
        for i in hero_decisions(spot, flop, acts, hero):
            res = adv.advise(dict(state(spot, flop, hero, hole, acts[:i]), **({"quick": True} if sp == "quick" else {})))
            if res.get("solver"):
                info["solves"].setdefault(sp, []).append([res["solver"].get("iterations"),
                                                          res["solver"].get("exploitability_pct")])
        box = read_ranges(adv, state(spot, flop + filler, hero, hole, acts))
        if sp == "fast":
            keep("rules", {p: box["rules"][p] for p in order})
            keep("fast", {p: box["walk"][p] for p in order})
            keep("fast-raw", {p: box["raw"][p] for p in order})
        else:
            keep(sp, {p: box["walk"][p] for p in order})
        info["ok"][sp] = box["ok"]
    keep("enter", {p: enter[p] for p in order})
    if late is None:
        info["ref"] = False
    else:
        keep("ref", late[0])
        info["ref"], info["shares"] = True, late[1]
        if early is not None:
            keep("ref-early", early[0])
    meta["lines"][nm] = info
    print(f"  {nm}: {info['ok']} {info['solves']} ({time.time() - t1:.0f} s)", flush=True)


def do_add(args):
    """Lines with a size of their own (SIZED) on the saved spot-flops: that size in the reference's tree."""
    only = args.spots.split(",") if args.spots else list(SPOTS)
    for spot in only:
        n, stack, pre = SPOTS[spot]
        for fi, flop in enumerate(FLOPS[spot]):
            fn = os.path.join(args.dir, f"{spot}-{flop}.npz")
            if fi not in args.rounds or not os.path.exists(fn):
                continue
            arrays, meta = dict(np.load(fn)), json.load(open(fn[:-4] + ".json"))
            todo = [nm for nm in args.add.split(",") if nm not in meta.get("sized", [])]
            if not todo:
                continue
            order, pot = meta["order"], meta["pot"]
            L = lines(order, pot)
            board = cards(flop)
            filler = next(c for c in ("2d", "3h", "4s", "5c") if cards(c)[0] not in board)
            enter = {p: arrays[f"a33 b|enter|{p}"].astype(np.float64) for p in order}
            st = A.replay_hand(table(n), stack, {"preflop": pre, "flop": []}, 2)[-1]
            stk = A._table_stacks(table(n), stack)
            S = min(stk[p] - st.inv_start[p] for p in order)
            rs = [enter[p] for p in order]
            holes = {p: hole_of(enter[p], board + cards(filler)) for p in order}
            advs = speeds()
            for nm in todo:
                t0 = time.time()
                alt = reference(board, rs, pot, S, sizes=SIZED[nm])
                alt.run(iters=REF_ITERS // 2)
                early = ref_walk(alt, order, L[nm][1])
                alt.run(iters=REF_ITERS - REF_ITERS // 2)
                solve_line(spot, flop, filler, nm, L[nm], order, enter, holes, advs, ref_walk(alt, order, L[nm][1]), early,
                           arrays, meta)
                meta.setdefault("sized", []).append(nm)
                print(f"{spot} {flop} {nm}: reference {(alt.br_gains() / alt.root_pot() * 100).round(2)} "
                      f"({time.time() - t0:.0f} s)", flush=True)
                del alt
            tmp = fn[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, fn)
            json.dump(meta, open(fn[:-4] + ".json", "w"))


def do_ranges(args):
    os.makedirs(args.dir, exist_ok=True)
    only = args.spots.split(",") if args.spots else list(SPOTS)
    for spot in only:
        n, stack, pre = SPOTS[spot]
        for fi, flop in enumerate(FLOPS[spot]):
            if fi not in args.rounds:
                continue
            fn = os.path.join(args.dir, f"{spot}-{flop}.npz")
            if os.path.exists(fn):
                continue
            t0 = time.time()
            st = A.replay_hand(table(n), stack, {"preflop": pre, "flop": []}, 2)[-1]
            order = [p for p in st.order if p in st.players_start]
            if len(order) != 4:
                raise SystemExit(f"{spot}: {order}")
            pot = st.pot_start
            L = lines(order, pot)
            board = cards(flop)
            filler = next(c for c in ("2d", "3h", "4s", "5c") if cards(c)[0] not in board)
            advs = speeds()
            probe = next(h for h in ("AsKs", "QcJc", "7h6h") if not set(cards(h)) & set(board + cards(filler)))
            first = read_ranges(advs["fast"], state(spot, flop + filler, order[0], probe, L["xxxx"][1]))
            enter = first["before"]
            stk = A._table_stacks(table(n), stack)
            S = min(stk[p] - st.inv_start[p] for p in order)
            rs = [enter[p] for p in order]
            ref = reference(board, rs, pot, S)
            ref.run(iters=REF_ITERS // 2)
            early = {nm: ref_walk(ref, order, acts) for nm, (_, acts, _) in L.items()}
            ref.run(iters=REF_ITERS - REF_ITERS // 2)
            late = {nm: ref_walk(ref, order, acts) for nm, (_, acts, _) in L.items()}
            gains = ref.br_gains() / ref.root_pot() * 100
            print(f"{spot} {flop}: reference {gains.round(2)} ({time.time() - t0:.0f} s)", flush=True)
            arrays, meta = {}, {"spot": spot, "flop": flop, "order": order, "pot": pot, "gains": gains.tolist(), "lines": {}}
            holes = {p: hole_of(enter[p], board + cards(filler)) for p in order}
            for nm in L:
                if nm not in SIZED:
                    solve_line(spot, flop, filler, nm, L[nm], order, enter, holes, advs, late[nm], early[nm], arrays, meta)
            np.savez_compressed(fn, **arrays)
            json.dump(meta, open(fn[:-4] + ".json", "w"))
            del ref, advs


def live(board):
    m = np.ones(N)
    for c in board:
        m = m * (1.0 - MEMBER_FULL[c])
    return m


def tv(x, y, m):
    x, y = x * m, y * m
    return 0.5 * float(np.abs(x / x.sum() - y / y.sum()).sum())


def floored(r, before, board, pos):
    return A._post_floor(before, r, board, pos, "flop", [])


def turn_adv():
    """The next decision's solve, stopped by how settled it is, not the clock (the machine is shared): 0.5% of the pot."""
    adv = native.advisor("fast", library=False, budget=60.0)
    adv.hu_settled, adv.hu_settled_from = 0.5, {k: 100 for k in A.HU_SETTLED_FROM}
    adv.mw_max_iters, adv.mw_budget = 150, 1e6
    return adv


def answers_for(adv, st, cand, hands):
    """Your answer for each hand at state st with the flop's ranges `cand`: {hand: {action: (freq, ev)}}."""
    real = A.Advisor._mw_walk

    def walk(ranges, before, board, st_, *a, **k):
        if st_.name != "flop":
            return real(adv, ranges, before, board, st_, *a, **k)
        ranges.update(cand)
        return True
    adv._mw_walk = walk
    out = {}
    try:
        for h in hands:
            res = adv.advise(dict(st, hole=h))
            if res.get("actions") and "ev_bb" in res["actions"][0]:
                out[h] = {a["action"]: (float(a["freq"]), float(a["ev_bb"] if a["ev_bb"] is not None else 0.0))
                          for a in res["actions"]}
    finally:
        del adv._mw_walk
    return out


def do_answers(args):
    with_ans = args.cands.split(",")
    out = os.path.join(args.dir, "results.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["spot"], r["flop"], r["line"], r["turn"]) for r in map(json.loads, open(out))}
    rows = open(out, "a")
    for fn in sorted(f for f in os.listdir(args.dir) if f.endswith(".npz")):
        z = np.load(os.path.join(args.dir, fn))
        meta = json.load(open(os.path.join(args.dir, fn[:-4] + ".json")))
        spot, flop, order = meta["spot"], meta["flop"], meta["order"]
        n, stack, pre = SPOTS[spot]
        fb = cards(flop)
        turns = [c for c in TURNS if cards(c)[0] not in fb][:args.turns]
        adv = turn_adv()
        for nm, info in meta["lines"].items():
            if args.lines and nm not in args.lines.split(","):
                continue
            hero, acts = info["hero"], info["acts"]
            for turn in turns:
                key = (spot, flop, nm, turn)
                if key in done:
                    continue
                t0 = time.time()
                board = fb + cards(turn)
                m = live(board)
                st = A.replay_hand(table(n), stack, {"preflop": pre, "flop": acts, "turn": []}, 3)[-1]
                alive = [p for p in order if p in st.active]
                if hero not in alive:
                    continue
                get = lambda c, p: z[f"{nm}|{c}|{p}"].astype(np.float64)
                cands = [c for c in ("rules", "fast", "fast-raw", "quick", "accurate", "ref-early", "ref")
                         if f"{nm}|{c}|{order[0]}" in z]
                row = {"spot": spot, "flop": flop, "line": nm, "turn": turn, "kind": info["kind"], "hero": hero,
                       "alive": alive, "ok": info["ok"], "solves": info["solves"], "shares": info.get("shares"),
                       "tv": {}, "combos": {}}
                if "ref" in cands:
                    ref = {p: floored(get("ref", p), get("enter", p), board, p) for p in alive}
                    for c in cands:
                        rg = {p: floored(get(c, p), get("enter", p), board, p) for p in alive}
                        row["tv"][c] = {p: tv(rg[p], ref[p], m) for p in alive}
                        row["combos"][c] = {p: float((rg[p] * m).sum()) for p in alive}
                if "ref" in cands:
                    rng = np.random.default_rng(zlib.crc32(f"{spot} {flop} {nm}".encode()))
                    w = get("ref", hero) * m
                    if len(alive) > 1 and hero in alive and (w > 0).sum() >= 1:
                        idx = rng.choice(N, size=min(HANDS, int((w > 0).sum())), replace=False, p=w / w.sum())
                        hands = [cards_str(ALL_COMBOS[int(i)]) for i in idx]
                        state_t = state(spot, flop + turn, hero, hands[0], acts)
                        to_go = {}
                        for c in with_ans:
                            if c in cands:
                                rg = {p: get(c, p) for p in order}
                                to_go[c] = answers_for(adv, state_t, rg, hands)
                        row["answers"] = to_go
                        row["pot"] = float(st.pot_start)
                rows.write(json.dumps(row) + "\n")
                rows.flush()
                print(f"{spot} {flop}{turn} {nm}: " + " ".join(
                    f"{c} {np.mean(list(row['tv'][c].values())):.3f}" for c in row["tv"]) + f" ({time.time() - t0:.0f} s)",
                      flush=True)
        del adv


def losses(row):
    """Per candidate, over the hands: the reference's mix less the candidate's valued at the reference's EVs (% of the
    pot), half the mix's L1 distance, the share of hands with the reference's top action, and the EV's move."""
    ans = row.get("answers") or {}
    ref = ans.get("ref")
    if not ref:
        return {}
    out = {}
    pot = row["pot"]
    for c, got in ans.items():
        L, D, T, M = [], [], [], []
        for h, q in ref.items():
            g = got.get(h)
            if g is None or set(g) != set(q):
                continue
            acts = list(q)
            sr = np.array([q[a][0] for a in acts])
            sc = np.array([g[a][0] for a in acts])
            Q = np.array([q[a][1] for a in acts])
            Qc = np.array([g[a][1] for a in acts])
            L.append(float((sr - sc) @ Q) / pot * 100)
            D.append(0.5 * float(np.abs(sr - sc).sum()))
            T.append(int(acts[int(np.argmax(sr))] == acts[int(np.argmax(sc))]))
            M.append(abs(float(sc @ Qc) - float(sr @ Q)) / pot * 100)
        if L:
            out[c] = {"loss": float(np.mean(L)), "mix": float(np.mean(D)), "top": float(np.mean(T)), "ev": float(np.mean(M))}
    return out


def boot(pairs, n=2000, seed=1):
    """The mean of pairs (group, difference) with a 95% interval over the groups."""
    rng = np.random.default_rng(seed)
    g = {}
    for k, d in pairs:
        g.setdefault(k, []).append(d)
    v = [np.mean(x) for x in g.values()]
    draws = [np.mean(rng.choice(v, size=len(v))) for _ in range(n)]
    return float(np.mean(v)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), len(v)


def do_report(args):
    rows = [json.loads(x) for x in open(os.path.join(args.dir, "results.jsonl"))]
    order = ["rules", "quick", "fast", "fast-raw", "accurate", "ref-early"]
    print(f"{len(rows)} rows, {len({(r['spot'], r['flop']) for r in rows})} spot-flops")
    groups = [("all on-tree", lambda r: r["kind"] == "tree"), ("heads-up next", lambda r: r["kind"] == "tree" and len(r["alive"]) == 2),
              ("3-way next", lambda r: r["kind"] == "tree" and len(r["alive"]) == 3),
              ("4-way next", lambda r: r["kind"] == "tree" and len(r["alive"]) == 4),
              ("25% -> 33%", lambda r: r["kind"] == "under"), ("39% -> 33%", lambda r: r["kind"] == "near"),
              ("50% -> 33%", lambda r: r["kind"] == "mid"), ("75% -> 33%", lambda r: r["kind"] == "far"),
              ("a raise", lambda r: r["kind"] == "raise")]
    print("\nRanges: mean TV from the reference, opponents of the hero still in")
    print(f"{'':16}" + "".join(f"{c:>10}" for c in order) + "      n")
    for nm, f in groups:
        sel = [r for r in rows if f(r) and r["tv"]]
        if not sel:
            continue
        cells = []
        for c in order:
            v = [np.mean([d for p, d in r["tv"][c].items() if p != r["hero"]]) for r in sel
                 if c in r["tv"] and any(p != r["hero"] for p in r["tv"][c])]
            cells.append(f"{np.mean(v):10.3f}" if v else f"{'-':>10}")
        print(f"{nm:16}" + "".join(cells) + f"{len(sel):7}")
    print("\nBy line (opponents' TV)")
    for ln in sorted({r["line"] for r in rows}):
        sel = [r for r in rows if r["line"] == ln and r["tv"]]
        cells = []
        for c in order:
            v = [np.mean([d for p, d in r["tv"][c].items() if p != r["hero"]]) for r in sel if c in r["tv"]]
            cells.append(f"{np.mean(v):10.3f}" if v else f"{'-':>10}")
        print(f"{ln:16}" + "".join(cells) + f"{len(sel):7}")
    print("\nAnswers: % of the pot given up to the reference ranges' solve (mix distance, top action, EV's move)")
    for nm, f in groups:
        sel = [(r, losses(r)) for r in rows if f(r) and r.get("answers")]
        sel = [(r, x) for r, x in sel if x]
        if not sel:
            continue
        print(f"{nm} ({len(sel)} decisions)")
        for c in order:
            v = [x[c] for _, x in sel if c in x]
            if v:
                print(f"  {c:10} loss {np.mean([d['loss'] for d in v]):7.3f}  mix {np.mean([d['mix'] for d in v]):.3f}  "
                      f"top {np.mean([d['top'] for d in v]):.2f}  ev {np.mean([d['ev'] for d in v]):.2f}")
        for c in ("fast", "accurate"):
            pairs = [((r["spot"], r["flop"]), x[c]["loss"] - x["rules"]["loss"]) for r, x in sel if c in x and "rules" in x]
            if pairs:
                m, lo, hi, k = boot(pairs)
                print(f"  {c} less rules: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {k} spot-flops")
    print("\nAnswers by line: % of the pot given up")
    print(f"{'':16}" + "".join(f"{c:>10}" for c in order) + "  fast-rules [95%]")
    for ln in sorted({r["line"] for r in rows}):
        sel = [(r, losses(r)) for r in rows if r["line"] == ln and r.get("answers")]
        sel = [(r, x) for r, x in sel if x]
        cells = [f"{np.mean([x[c]['loss'] for _, x in sel if c in x]):10.3f}" if any(c in x for _, x in sel) else f"{'-':>10}"
                 for c in order]
        m, lo, hi, _ = boot([((r["spot"], r["flop"]), x["fast"]["loss"] - x["rules"]["loss"]) for r, x in sel
                             if "fast" in x and "rules" in x])
        print(f"{ln:16}" + "".join(cells) + f"  {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\nBy the line's rarity in the reference (the rarest action's share of its player's range): opponents' TV")
    for lo, hi in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.4), (0.4, 1.01)):
        sel = [r for r in rows if r["kind"] == "tree" and r["tv"] and r.get("shares")
               and lo <= min(s[2] for s in r["shares"]) < hi]
        if sel:
            v = {c: np.mean([np.mean([d for p, d in r["tv"][c].items() if p != r["hero"]]) for r in sel if c in r["tv"]])
                 for c in order if any(c in r["tv"] for r in sel)}
            print(f"  {lo:.2f}-{hi:.2f}: " + "  ".join(f"{c} {x:.3f}" for c, x in v.items()) + f"  n {len(sel)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("ranges", "add", "answers", "report"))
    ap.add_argument("--add", default="a50 b,a39 b")
    ap.add_argument("--dir", default="reports/mw4_narrowing")
    ap.add_argument("--spots", default="")
    ap.add_argument("--rounds", type=lambda x: [int(i) for i in x.split(",")], default=[0, 1, 2, 3],
                    help="which of each spot's flops")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--lines", default="")
    ap.add_argument("--cands", default="rules,quick,fast,fast-raw,accurate,ref-early,ref")
    a = ap.parse_args()
    {"ranges": do_ranges, "add": do_add, "answers": do_answers, "report": do_report}[a.what](a)


if __name__ == "__main__":
    main()
