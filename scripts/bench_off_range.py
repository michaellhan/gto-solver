"""Does a hand outside its range lose anything by being solved at a near-zero weight (Advisor._with_hero, MW_OFF_RANGE)?

CFR works out a hand's value against the opponents' reach whatever the hand's own weight, so a hand at 1e-3 of a combo
plays a best response to the others' strategies. What it can't do is move them. This compares, spot by spot, the answer
the Advisor gives such a hand with the answer at a real weight, on the same tree and solve budget:

  near     the hand at a near-zero weight (what the Advisor does)
  combo    the hand itself at 1 combo, on the street answered
  class    its whole hand class at CLASS_W a combo, on the street answered
  history  the class and its neighbours (family: A4s -> A2s-A6s) at CLASS_W from before the flop, through every street
           the hand has played: the same earlier solves and rules, with those hands in the range they narrow
  control  the combo solve of ANOTHER probe, this hand still near-zero there: what a small change of range does to an
           answer that isn't the hand's own, so the noise floor of everything above
  tiny     (--arms) the hand at TINY_W of a combo: whether a weight a little over near-zero settles what a real one does
  wide     (--arms) a band of near-bar hands in from before the flop whatever you hold (WIDE_MASS of the range's combos,
           near_bar), the one repair that can't depend on your cards: what it does to these hands and to hands in range,
           and what it costs the solve

  python scripts/bench_off_range.py run --group hu --out hu.jsonl        # resumes: finished reads are skipped
  python scripts/bench_off_range.py run --group 3way --out mw.jsonl
  python scripts/bench_off_range.py report --out hu.jsonl,mw.jsonl
  python scripts/bench_off_range.py buckets     # the flop's worst reads with its later streets played per combo
  python scripts/bench_off_range.py hand        # the hand that asked (As4s, 7-handed), by arm

Scores: the top action (the mix's most frequent) changing; the regret of the near-zero answer's mix read at the real
weight's EVs, as a % of the pot (the number that matters: a flip between two actions with equal EVs costs nothing); each
action's EV difference; and how far everyone else's strategies move, as a reach-weighted distance. Results:
docs/off_range_research.md.
"""
import argparse
import dataclasses
import gzip
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gto import native  # noqa: E402  (first: holds BLAS to one thread)
import numpy as np      # noqa: E402
import gto.advisor as A  # noqa: E402
from gto import breakdown as BD  # noqa: E402
from gto import multiway_solver as MW  # noqa: E402
from gto import preflop as pf  # noqa: E402
from gto.cards import ALL_COMBOS, COMBO_INDEX, cards  # noqa: E402
from gto.solver import Decision  # noqa: E402

N = 1326
NEAR_HU = 1e-3     # as Advisor._with_hero
NEAR_MW = 2e-9     # just over MW_OFF_RANGE's 1e-9 test, so one solve serves every probe of a spot (the same near-zero)
CLASS_W = 0.25     # a combo's weight in the class and history arms
HU_TARGET = 0.5    # % of the pot: where a heads-up solve stops (the arms are compared solve for solve)
TINY_W = 0.05      # a combo's weight in the tiny arm
WIDE_MASS = 0.2    # the wide arm's band, as a share of the range's combos (each band combo at CLASS_W)
RANKS = "23456789TJQKA"


POOL = False       # --pool: answered against the typical player, as the page's cash hands are


def p_(pos, action, **kw):
    return {"pos": pos, "action": action, **kw}


def fold(*seats):
    return [p_(s, "fold") for s in seats]


def to_btn(*rest):
    return fold("UTG", "HJ", "CO") + list(rest)


# (name, table size, stack, preflop, hero)
HU = [
    ("srp-btn-bb", 6, 100, to_btn(p_("BTN", "raise", to=2.5), p_("SB", "fold"), p_("BB", "call")), "BB"),
    ("srp-btn-btn", 6, 100, to_btn(p_("BTN", "raise", to=2.5), p_("SB", "fold"), p_("BB", "call")), "BTN"),
    ("srp-utg-bb", 6, 100, [p_("UTG", "raise", to=2.5)] + fold("HJ", "CO", "BTN", "SB") + [p_("BB", "call")], "BB"),
    ("limp-bb", 6, 100, fold("UTG", "HJ", "CO", "BTN") + [p_("SB", "call"), p_("BB", "check")], "BB"),
    ("3bet-co", 6, 100, fold("UTG", "HJ") + [p_("CO", "raise", to=2.5), p_("BTN", "raise", to=7.5)] + fold("SB", "BB")
     + [p_("CO", "call")], "CO"),
    ("3bet-btn", 6, 100, to_btn(p_("BTN", "raise", to=2.5), p_("SB", "fold"), p_("BB", "raise", to=10.0),
                                p_("BTN", "call")), "BTN"),
    ("4bet-bb", 6, 100, to_btn(p_("BTN", "raise", to=2.5), p_("SB", "fold"), p_("BB", "raise", to=10.0),
                               p_("BTN", "raise", to=23.0), p_("BB", "call")), "BB"),
    ("short-bb", 6, 40, to_btn(p_("BTN", "raise", to=2.5), p_("SB", "fold"), p_("BB", "call")), "BB"),
]
THREE = [
    ("3way-co-bb", 6, 100, fold("UTG", "HJ") + [p_("CO", "raise", to=2.5), p_("BTN", "call"), p_("SB", "fold"), p_("BB", "call")], "BB"),
    ("3way-co-btn", 6, 100, fold("UTG", "HJ") + [p_("CO", "raise", to=2.5), p_("BTN", "call"), p_("SB", "fold"), p_("BB", "call")], "BTN"),
    ("3way-utg-bb", 6, 100, [p_("UTG", "raise", to=2.5)] + fold("HJ", "CO", "BTN") + [p_("SB", "call"), p_("BB", "call")], "BB"),
    ("3way-3bet-bb", 7, 100, fold("UTG", "LJ", "HJ") + [p_("CO", "raise", to=2.5), p_("BTN", "raise", to=7.5), p_("SB", "fold"),
                                                        p_("BB", "call"), p_("CO", "call")], "BB"),
    # the hand that asked: limped, raised, called twice behind: 4-way to the flop
    ("4way-limp-bb", 7, 100, [p_("UTG", "call"), p_("LJ", "raise", to=4), p_("HJ", "call")] + fold("CO", "BTN", "SB")
     + [p_("BB", "call"), p_("UTG", "call")], "BB"),
]
FLOPS = ["Ks7s2h", "Jh9h4c", "8s8d3c", "7c5c2d", "AhKdQc", "Td8s6s"]
TURNS = ["2c", "Qs", "5h"]
RIVERS = ["3d", "Ah", "Tc"]


def table(n):
    return pf.Table(pf.SEATS[n])


def nxt_card(board, deck):
    held = {board[i:i + 2] for i in range(0, len(board), 2)}
    return next(c for c in deck if c not in held)


def replay(n, stack, streets, name):
    k = {"flop": 2, "turn": 3, "river": 4}[name]
    return A.replay_hand(table(n), stack, streets, k)[-1]


def play_line(n, stack, streets, name, frac):
    """The street played to its end: the first player to act checks, the next bets `frac` of the pot and the rest call
    (frac 0: everyone checks)."""
    acts, bet = [], False
    while True:
        st = replay(n, stack, {**streets, name: acts}, name)
        if st.closed or st.to_act is None:
            return acts
        p = st.to_act
        pot = st.pot_start + sum(st.bets.values())
        if st.cur_bet > st.bets[p] + 1e-9:
            acts = acts + [p_(p, "call")]
        elif acts and frac and not bet:
            acts, bet = acts + [p_(p, "bet", amount=max(round(frac * pot * 2) / 2, 1.0))], True
        else:
            acts = acts + [p_(p, "check")]


def read_nodes(n, stack, streets, name, hero, frac):
    """The hero decisions a street's reads stop at: its first, and facing a bet of `frac` of the pot."""
    out, acts = [], []
    while True:
        st = replay(n, stack, {**streets, name: acts}, name)
        if st.closed or st.to_act is None or len(out) == 2:
            return out
        p = st.to_act
        pot = st.pot_start + sum(st.bets.values())
        if p == hero:
            out.append(("facing" if st.cur_bet > st.bets[p] + 1e-9 else "first", list(acts)))
            if st.cur_bet > st.bets[p] + 1e-9:
                return out
            if acts:       # (in position, checked to: the bet read is the other branch, from the top)
                return out + read_facing(n, stack, streets, name, hero, frac)
            acts = acts + [p_(p, "check")]
        elif st.cur_bet > st.bets[p] + 1e-9:
            acts = acts + [p_(p, "call")]
        elif hero in [a["pos"] for a in acts]:   # (hero checked: the next player bets)
            acts = acts + [p_(p, "bet", amount=max(round(frac * pot * 2) / 2, 1.0))]
        else:
            acts = acts + [p_(p, "check")]


def read_facing(n, stack, streets, name, hero, frac):
    """Hero in position, the bet in front of it: the first player leads."""
    acts = []
    while True:
        st = replay(n, stack, {**streets, name: acts}, name)
        if st.closed or st.to_act is None:
            return []
        p = st.to_act
        if p == hero:
            return [("facing", list(acts))] if st.cur_bet > st.bets[p] + 1e-9 else []
        pot = st.pot_start + sum(st.bets.values())
        acts = acts + ([p_(p, "call")] if st.cur_bet > st.bets[p] + 1e-9 else
                       [p_(p, "bet", amount=max(round(frac * pot * 2) / 2, 1.0))] if not acts else [p_(p, "check")])


def spots(group, flops):
    """(id, group, table size, stack, hero, streets before this one, board, street, reads)"""
    pots = HU if group == "hu" else THREE
    for i, (name, n, stack, pre, hero) in enumerate(pots):
        for j in range(flops):
            flop = FLOPS[(i + j * 2) % len(FLOPS)]
            k = (i + j) % 3
            turn = nxt_card(flop, TURNS[k:] + TURNS[:k])
            river = nxt_card(flop + turn, RIVERS[k:] + RIVERS[:k])
            streets = {"preflop": pre}
            frac = (0.33, 0.75)[(i + j) % 2]
            line = (frac, 0.0)[(i + j) % 2]    # an earlier street's bet, or checked through
            for street, board in (("flop", flop), ("turn", flop + turn), ("river", flop + turn + river)):
                st = {"flop": [], "turn": [], "river": [], **streets}
                reads = read_nodes(n, stack, st, street, hero, frac)
                if reads and group != "hu":
                    reads = reads[:1] if (i + j) % 2 else reads[-1:]   # (3-way: each read is a solve of its own)
                yield (f"{name}/{flop}/{street}", "4way" if name.startswith("4way") else group, n, stack, hero,
                       dict(streets), board, street, reads)
                played = play_line(n, stack, st, street, line)
                streets[street] = played
                if len(replay(n, stack, {**st, street: played}, street).active) < 2:
                    break


def bin_of(hole, board):
    made, draw = BD.made_category(hole, board), BD.draw_category(hole, board)
    if made in ("quads", "full_house", "flush", "straight", "set", "trips", "two_pair", "overpair"):
        return "strong", made
    if made.startswith("top"):
        return "top pair", made
    if made in ("middle", "weak"):
        return "lower pair", made
    return ("draw" if draw in ("combo", "fd", "oesd") else "gutshot" if draw == "gutshot" else "air"), made


def probes(rng_in, board, k, seed):
    """Up to k off-range hands of the board's hand types: one a type, classes rather than combos equally likely."""
    rnd = random.Random(seed)
    dead = set(cards(board))
    by = {}
    for i, cb in enumerate(ALL_COMBOS):
        if rng_in[i] <= 1e-9 and not dead & set(cb):
            by.setdefault(bin_of(cb, cards(board))[0], {}).setdefault(A.hand_class(cb), []).append(i)
    out, order = [], ["strong", "top pair", "lower pair", "draw", "gutshot", "air"]
    while len(out) < k and any(by.get(b) for b in order):
        for b in order:
            if by.get(b) and len(out) < k:
                cls = rnd.choice(sorted(by[b]))
                out.append((b, cls, rnd.choice(by[b].pop(cls))))
    return out


def family(cls):
    """A hand class and its neighbours: pairs a rank either way; else the same top card, kicker within two (A4s: A2s-A6s)."""
    if len(cls) == 2:
        r = RANKS.index(cls[0])
        return [RANKS[x] * 2 for x in range(max(r - 1, 0), min(r + 2, 13))]
    hi, lo, shape = RANKS.index(cls[0]), RANKS.index(cls[1]), cls[2]
    return [cls[0] + RANKS[x] + shape for x in range(max(lo - 2, 0), min(lo + 3, hi))]


class Harness:
    """One Advisor for a street of a hand, its ranges patched by arm: the earlier streets' solves are the same solves."""

    def __init__(self, n, stack, hero, streets, board):
        self.n, self.stack, self.hero, self.board = n, stack, hero, board
        self.streets = {"flop": [], "turn": [], "river": [], **streets}
        adv = self.adv = native.advisor("fast", budget=40.0, prior_budget=10.0, library=False)
        adv.hu_settled = HU_TARGET
        adv.mw_budget = 30.0
        adv.cache_mb = 500
        self.w, self.before = {}, None
        self.seen, self.walked = None, None
        self._cached, self._wlk, self._mw = adv._solve_cached, adv._walk, adv._multiway_solve
        adv._with_hero = self.with_hero
        adv._as_advised = lambda r0, r1, hp, hole: (r0, r1)
        adv._solve_cached = self.solve_spy
        adv._walk = self.walk_spy
        adv._multiway_solve = self.mw_spy

    def with_hero(self, r0, r1, hero_p, hole, notes):
        r = (r0 if hero_p == 0 else r1).copy()
        for i, w in self.w.items():
            r[i] = max(r[i], w)
        return (r, r1) if hero_p == 0 else (r0, r)

    def solve_spy(self, *a, **k):
        self.seen = self._cached(*a, **k)
        return self.seen

    def walk_spy(self, sol, *a, **k):
        out = self._wlk(sol, *a, **k)
        self.walked = (sol, out)
        return out

    def mw_spy(self, hero, hole, board, ranges, *a, **k):
        ranges = dict(ranges)
        r = ranges[hero].copy()
        for i, w in self.w.items():
            r[i] = max(r[i], w)
        ranges[hero] = r
        return self._mw(hero, hole, board, ranges, *a, **k)

    def state(self, hole, acts, street):
        return {"players": self.n, "stack": self.stack, "hero": self.hero, "hole": hole, "board": self.board,
                **self.streets, street: list(acts), **({"opponents": "pool"} if POOL else {})}

    def view(self, res, hero_p=None):
        """Who holds what at the node just answered: (hero's strategy, its reach, [(the next player's strategy, reach,
        weight)]) over all combos, for the distances; None where the answer isn't a solve's."""
        if res.get("multiway"):
            sol = list(self.adv._mw_cache.values())[-1]
            root, hp = sol.root, sol.root.player

            def full(s):
                f = np.zeros((s.shape[0], N)); f[:, sol.hands.active] = s
                return f
            reach = [sol.expand(sol.reach0[p][0]).astype(float) for p in range(len(sol.reach0))]
            s = full(root.avg_strategy()[0])
            w = reach[hp] / max(reach[hp].sum(), 1e-12)
            kids = []
            for a, ch in enumerate(root.children):
                if isinstance(ch, MW.Decision) and ch.player != hp:
                    kids.append((a, full(ch.avg_strategy()[0]), reach[ch.player]))
            return s, reach[hp], kids, (s * w).sum(axis=1)
        if self.seen is None or self.walked is None:
            return None
        wsol, (node, c0, c1) = self.walked
        if wsol is not self.seen:
            node, c0, c1 = self.seen.root, self.seen.r0, self.seen.r1
        sol = self.seen
        if not isinstance(node, Decision):
            return None
        hp = node.player

        def full(s):
            f = np.zeros((s.shape[0], N)); f[:, sol.ctx.active] = s
            return f
        s, rh, rv = full(node.avg_strategy0()), A.expand(sol.ctx, c0 if hp == 0 else c1), A.expand(sol.ctx, c1 if hp == 0 else c0)
        w = rh / max(rh.sum(), 1e-12)
        kids = [(a, full(ch.avg_strategy0()), rv) for a, ch in enumerate(node.children)
                if isinstance(ch, Decision) and ch.player != hp]
        return s, rh, kids, (s * w).sum(axis=1)

    def ask(self, hi, acts, street):
        self.seen = self.walked = None
        hole = "".join(RANKS[c // 4] + "cdhs"[c % 4] for c in ALL_COMBOS[hi])
        t = time.time()
        res = self.adv.advise(self.state(hole, acts, street))
        return res, self.view(res), time.time() - t


def tv(sa, ra, sb, rb, skip):
    """The reach-weighted distance between two strategies of a range, % of it: the combos in `skip` out."""
    keep = np.ones(N, bool)
    keep[list(skip)] = False
    w = np.minimum(ra, rb) * keep
    if w.sum() <= 1e-9:
        return 0.0
    return float(100 * (w * 0.5 * np.abs(sa - sb).sum(axis=0)).sum() / w.sum())


def compare(base, other, skip):
    """(hero's range's distance, the next player's after each of hero's actions weighted by how often the range takes
    it, the biggest change in the range's frequency of an action, % points)."""
    if base is None or other is None or base[0].shape != other[0].shape:
        return None, None, None
    h = tv(base[0], base[1], other[0], other[1], skip)
    kids = {a: (s, r) for a, s, r in other[2]}
    v, tot = 0.0, 0.0
    for a, s, r in base[2]:
        if a in kids and kids[a][0].shape == s.shape:
            f = float(base[3][a])
            v += f * tv(s, r, kids[a][0], kids[a][1], ())
            tot += f
    return round(h, 3), round(v / tot, 3) if tot > 0 else None, round(float(100 * np.abs(base[3] - other[3]).max()), 3)


def line_of(res):
    return [[a["action"], a["freq"], a["ev_bb"]] for a in res.get("actions", [])]


def mass(res, view):
    return None if view is None else round(float(view[1].sum()), 2)


def run(args):
    global POOL
    POOL = args.pool
    arms = set(args.arms.split(","))
    done = set()
    if os.path.exists(args.out):
        for ln in open(args.out):
            r = json.loads(ln)
            done.add((r["spot"], r["node"], r["hand"], r["arm"]))
    out = open(args.out, "a")
    real_ranges = A.preflop_ranges
    all_spots = list(spots(args.group, args.flops))
    for s_i, (sid, group, n, stack, hero, streets, board, street, reads) in enumerate(all_spots):
        if args.only and args.only not in sid or street not in args.streets.split(","):
            continue
        if args.max_spots and s_i >= args.max_spots:
            break
        t_spot = time.time()
        pf_in = real_ranges(streets["preflop"], table(n), stack)[hero]
        ps = probes(pf_in, board, args.hands, sid)
        if not ps or not reads:
            continue
        near_w = NEAR_MW if group != "hu" else NEAR_HU
        base_w = {p[2]: near_w for p in ps}
        h = Harness(n, stack, hero, streets, board)    # (heads-up: one solve of the street serves every read of it)
        for kind, acts in reads:
            node = f"{kind}:{len(acts)}"
            h.w = dict(base_w)
            base = {}
            for b, cls, hi in ps:
                try:
                    res, view, secs = h.ask(hi, acts, street)
                except Exception as e:  # noqa: BLE001  (one bad read isn't the run)
                    print("skip", sid, node, cls, e, flush=True)
                    continue
                if "error" in res or not res.get("actions"):
                    print("no answer", sid, node, cls, res.get("error"), flush=True)
                    continue
                base[hi] = (res, view)
                key = cls + "/" + b
                if (sid, node, key, "near") not in done:
                    emit(out, sid, group, street, node, res, key, "near", secs, None, view, ps)
            if not base:
                continue

            def arm(name, weights, wrap=None, only=None):
                """One solve with these weights in, read for the probes it is about."""
                A.preflop_ranges = wrap or real_ranges
                try:
                    h2 = h if wrap is None else Harness(n, stack, hero, streets, board)
                    h2.w = {**base_w, **weights}
                    for b, cls, hi in ps:
                        key = cls + "/" + b
                        if hi not in base or (only is not None and hi not in only) or (sid, node, key, name) in done:
                            continue
                        res, view, secs = h2.ask(hi, acts, street)
                        if "error" in res:
                            continue
                        skip = [i for i, x in h2.w.items() if x > near_w] + list(wrap.added if wrap else ())
                        emit(out, sid, group, street, node, res, key, name, secs, base[hi], view, ps, skip)
                finally:
                    A.preflop_ranges = real_ranges

            for pi, (b, cls, hi) in enumerate(ps):
                if hi not in base:
                    continue
                key = cls + "/" + b
                if "combo" in arms and (sid, node, key, "combo") not in done:
                    next_hi = ps[(pi + 1) % len(ps)][2]
                    arm("combo", {hi: 1.0}, only={hi})
                    if next_hi != hi:     # (the same solve, the next probe still near-zero in it)
                        arm_control(h, out, sid, group, street, node, acts, ps, pi, base, base_w, hi, done)
                if "class" in arms and (sid, node, key, "class") not in done:
                    idx = A._class_index()[cls]
                    arm("class", {int(i): CLASS_W for i in idx if pf_in[i] <= 1e-9}, only={hi})
                if "history" in arms and (sid, node, key, "history") not in done and (args.history or street == "flop") \
                        and pi % args.history_every == 0:
                    add = [int(i) for c in family(cls) for i in A._class_index()[c] if pf_in[i] <= 1e-9]
                    arm("history", {}, wrap=with_family(real_ranges, hero, add), only={hi})
                if "tiny" in arms and (sid, node, key, "tiny") not in done:
                    arm("tiny", {hi: TINY_W}, only={hi})
            if "wide" in arms:
                wrap = with_family(real_ranges, hero, near_bar(pf_in))
                dead = set(cards(board))
                rnd = random.Random(sid)
                held = [i for i, cb in enumerate(ALL_COMBOS) if pf_in[i] > 0.2 and not dead & set(cb)]
                inside = [("in " + bin_of(ALL_COMBOS[i], cards(board))[0], A.hand_class(ALL_COMBOS[i]), i)
                          for i in rnd.sample(held, min(4, len(held)))]
                for b, cls, hi in inside:      # (hands in range, as they stand)
                    res, view, secs = h.ask(hi, acts, street)
                    if "error" not in res and res.get("actions"):
                        base[hi] = (res, view)
                        if (sid, node, cls + "/" + b, "near") not in done:
                            emit(out, sid, group, street, node, res, cls + "/" + b, "near", secs, None, view, ps)
                A.preflop_ranges = wrap
                try:
                    h2 = Harness(n, stack, hero, streets, board)
                    h2.w = dict(base_w)
                    for b, cls, hi in list(ps) + inside:
                        key = cls + "/" + b
                        if hi in base and (sid, node, key, "wide") not in done:
                            res, view, secs = h2.ask(hi, acts, street)
                            if "error" not in res:
                                emit(out, sid, group, street, node, res, key, "wide", secs, base[hi], view, ps, wrap.added)
                finally:
                    A.preflop_ranges = real_ranges
        print(f"{s_i + 1}/{len(all_spots)} {sid} {time.time() - t_spot:.0f}s", flush=True)


def near_bar(pf_in):
    """The off-range combos closest to the range's weak end by strength (the weakest tenth of it by weight), enough of
    them for WIDE_MASS of its combos at CLASS_W. No cards of anyone's in it."""
    st = pf._strength()
    w = np.asarray(pf_in, dtype=float)
    order = np.argsort(st, kind="stable")
    cum = np.cumsum(w[order])
    bar = float(st[order[min(int(np.searchsorted(cum, 0.1 * cum[-1])), N - 1)]])
    off = np.nonzero(w <= 1e-9)[0]
    off = off[np.argsort(np.abs(st[off] - bar), kind="stable")]
    return [int(i) for i in off[:int(round(WIDE_MASS * w.sum() / CLASS_W))]]


def arm_control(h, out, sid, group, street, node, acts, ps, pi, base, base_w, hi, done):
    b2, cls2, h2i = ps[(pi + 1) % len(ps)]
    key = cls2 + "/" + b2
    if h2i not in base:
        return
    h.w = {**base_w, hi: 1.0}
    res, view, secs = h.ask(h2i, acts, street)
    if "error" not in res:
        emit(out, sid, group, street, node, res, key, "control", secs, base[h2i], view, ps, [hi])


def with_family(real, hero, add):
    def wrap(actions, positions, stack, pool_except=None, locks=None):
        out = real(actions, positions, stack, pool_except, locks)
        new = type(out)(out)
        new.rebuilt = list(getattr(out, "rebuilt", []))
        r = np.array(new[hero], dtype=float)
        r[add] = np.maximum(r[add], CLASS_W)
        new[hero] = r
        return new
    wrap.added = add
    return wrap


def emit(out, sid, group, street, node, res, key, arm, secs, base, view, ps, skip=()):
    rest = compare(base[1], view, skip) if base is not None else (None, None, None)
    out.write(json.dumps({"spot": sid, "group": group, "street": street, "node": node, "hand": key, "arm": arm,
                          "pot": res.get("pot_bb"), "hero_combos": mass(res, view),
                          "villain_combos": res.get("villain_range_combos") or sum((res.get("opponent_range_combos") or {}).values()),
                          "acts": line_of(res), "rest": rest, "secs": round(secs, 2),
                          "iters": (res.get("solver") or {}).get("iterations"),
                          "active": (res.get("solver") or {}).get("active_combos"),
                          "expl": (res.get("solver") or {}).get("exploitability_pct")}) + "\n")
    out.flush()


# ------------------------------------------------------------ the report
def score(near, x):
    """(top action changed, top by EV changed, regret of near's mix at x's EVs, regret of near's top action, mean |EV
    difference|, largest, each % of the pot) or None where the two aren't the same menu."""
    a, b = near["acts"], x["acts"]
    if [q[0] for q in a] != [q[0] for q in b]:
        return None
    pot = max(x["pot"] or near["pot"], 1e-9)
    top = max(range(len(a)), key=lambda i: a[i][1])
    top_x = max(range(len(b)), key=lambda i: b[i][1])
    ev_top = max(range(len(a)), key=lambda i: a[i][2])
    ev_top_x = max(range(len(b)), key=lambda i: b[i][2])
    best = max(q[2] for q in b)
    mix = sum(p[1] * q[2] for p, q in zip(a, b)) / max(sum(p[1] for p in a), 1e-9)
    dev = [abs(p[2] - q[2]) / pot * 100 for p, q in zip(a, b)]
    # (a flip that costs under a tenth of a percent of the pot is two tied actions: counted apart)
    # the other way round: the real weight's mix, played against the ranges that leave the hand out
    back = (max(q[2] for q in a) - sum(p[1] * q[2] for p, q in zip(b, a)) / max(sum(p[1] for p in b), 1e-9)) / pot * 100
    return (top != top_x, ev_top != ev_top_x, (best - mix) / pot * 100, (best - b[top][2]) / pot * 100,
            float(np.mean(dev)), float(max(dev)), abs(a[top][1] - b[top][1]) * 100, back)


def boot(xs, spots_of, stat, n=400, seed=7):
    """The statistic's 90% interval, spots drawn with replacement: reads of a spot aren't independent."""
    rnd = random.Random(seed)
    by = {}
    for x, s in zip(xs, spots_of):
        by.setdefault(s, []).append(x)
    keys = sorted(by)
    vals = []
    for _ in range(n):
        pick = [x for k in (rnd.choice(keys) for _ in keys) for x in by[k]]
        vals.append(stat(pick))
    vals.sort()
    return vals[int(0.05 * n)], vals[int(0.95 * n)]


def rows_for(reads, arm, keep=lambda r: True):
    near = {(r["spot"], r["node"], r["hand"]): r for r in reads if r["arm"] == "near"}
    out = []
    for r in reads:
        if r["arm"] != arm or not keep(r):
            continue
        n = near.get((r["spot"], r["node"], r["hand"]))
        s = n and score(n, r)
        if s:
            out.append((s, r, n))
    return out


def line(label, rows):
    if not rows:
        return f"{label:<30}    0"
    S = [r[0] for r in rows]
    sp = [r[1]["spot"] for r in rows]
    mix = [s[2] for s in S]
    flip = [float(s[0]) for s in S]
    costly = [float(s[0] and s[3] > 0.5) for s in S]
    lo, hi = boot(mix, sp, lambda v: float(np.mean(v)))
    hv = [r[1]["rest"][0] for r in rows if r[1]["rest"] and r[1]["rest"][0] is not None]
    vv = [r[1]["rest"][1] for r in rows if r[1]["rest"] and r[1]["rest"][1] is not None]
    return (f"{label:<30} {len(rows):>4} {100 * np.mean(flip):>5.0f}% {100 * np.mean(costly):>5.0f}% "
            f"{np.mean(mix):>6.2f} [{lo:.2f},{hi:.2f}] {np.median(mix):>5.2f} {np.percentile(mix, 90):>5.2f} {max(mix):>6.2f} "
            f"{np.mean([s[7] for s in S]):>5.2f} "
            f"{np.mean([s[4] for s in S]):>5.2f} {np.percentile([s[5] for s in S], 90):>5.2f} "
            f"{np.mean(hv) if hv else 0:>5.2f} {np.mean(vv) if vv else 0:>5.2f}")


HEAD = (f"{'':<30} {'n':>4} {'flip':>6} {'cost':>6} {'regret mean [90%]':>20} {'med':>5} {'p90':>5} {'max':>6} {'back':>5} "
        f"{'|dEV|':>5} {'p90':>5} {'hero':>5} {'vill':>5}")


def grade(cost):
    return 0 if cost <= 3.0 else 1 if cost <= 10.0 else 2     # (the page's EV grades: good, a little off, a mistake)


def verdicts(rows, arm):
    """What the answer's reader sees. Grades: each action's EV gap to the top, % of the pot, near-zero against the real
    weight's. Close (the page's 3%): when near-zero's top EV leads the next by more, does the real weight agree?"""
    if not rows:
        return
    same = tot = over = under = 0
    lead, flips, loss = 0, 0, []
    within, w_loss = 0, []
    gaps = []
    for s, r, n in rows:
        a, b = n["acts"], r["acts"]
        pot = max(r["pot"], 1e-9)
        ta, tb = max(q[2] for q in a), max(q[2] for q in b)
        for p, q in zip(a, b):
            ga, gb = grade((ta - p[2]) / pot * 100), grade((tb - q[2]) / pot * 100)
            tot += 1
            same += ga == gb
            over += ga > gb
            under += ga < gb
            gaps.append(((ta - p[2]) - (tb - q[2])) / pot * 100)
        best = sorted(p[2] for p in a)
        top = max(range(len(a)), key=lambda i: a[i][2])
        cost = (tb - b[top][2]) / pot * 100
        if len(best) > 1 and (best[-1] - best[-2]) / pot * 100 > 3.0:
            lead += 1
            flips += max(range(len(b)), key=lambda i: b[i][2]) != top
            loss.append(cost)
        else:
            within += 1
            w_loss.append(cost)
    print(f"{arm}: grades of {tot} actions: same {100 * same / tot:.1f}%, near-zero harsher {100 * over / tot:.1f}%, kinder "
          f"{100 * under / tot:.1f}%; its EV gap less the real one's: mean {np.mean(gaps):+.2f}, p10 {np.percentile(gaps, 10):+.2f}, "
          f"p90 {np.percentile(gaps, 90):+.2f}")
    print(f"   near-zero's top EV by over 3% of the pot: {lead} reads, the real top differs on {flips}, top's cost mean "
          f"{np.mean(loss) if loss else 0:.2f} max {max(loss) if loss else 0:.2f}; within 3%: {within} reads, cost mean "
          f"{np.mean(w_loss) if w_loss else 0:.2f} max {max(w_loss) if w_loss else 0:.2f}")


def costs(reads):
    near = {(r["spot"], r["node"], r["hand"]): r for r in reads if r["arm"] == "near"}
    ra = []
    for r in reads:
        n = near.get((r["spot"], r["node"], r["hand"]))
        if r["arm"] == "wide" and n and n.get("active") and r.get("active") and n.get("iters") and r.get("iters"):
            ra.append(r["active"] / n["active"])
    if ra:
        print(f"wide: the solve's combos x{np.median(ra):.2f} (median), x{np.percentile(ra, 90):.2f} (p90)")


def report(args):
    reads = [json.loads(ln) for f in args.out.split(",") for ln in (gzip.open(f, "rt") if f.endswith(".gz") else open(f))]
    print(f"{len(reads)} reads, {len({r['spot'] for r in reads})} spots, "
          f"{len({(r['spot'], r['node'], r['hand']) for r in reads if r['arm'] == 'near'})} hand-nodes, "
          f"{sum(r['secs'] for r in reads) / 60:.0f} min of asking")
    print("flip: top action changes; cost: it changes and that costs > 0.5% of the pot; regret: the near-zero mix at the "
          "real weight's EVs, % of the pot; back: the real weight's mix at near-zero's EVs; |dEV|: per action; hero / vill: how far the rest of the range moves, % of it\n")
    for arm in ("control", "tiny", "combo", "class", "history", "wide"):
        print(f"--- {arm}\n{HEAD}")
        for g in ("hu", "3way", "4way"):
            for st in ("flop", "turn", "river"):
                line_rows = rows_for(reads, arm, lambda r: r["group"] == g and r["street"] == st)
                if line_rows:
                    print(line(f"{g} {st}", line_rows))
        rows = rows_for(reads, arm)
        if not rows:
            continue
        print(line("all", rows))
        for lo, hi in ((0, 20), (20, 60), (60, 2000)):
            print(line(f"hero range {lo}-{hi} combos", [x for x in rows if x[1]["hero_combos"] and lo <= x[1]["hero_combos"] < hi]))
        for b in ("strong", "top pair", "lower pair", "draw", "gutshot", "air"):
            print(line(b, [x for x in rows if x[1]["hand"].endswith("/" + b)]))
        for k in ("first", "facing"):
            print(line(k, [x for x in rows if x[1]["node"].startswith(k)]))
        for b in ("in strong", "in top pair", "in lower pair", "in draw", "in gutshot", "in air"):
            if any(x[1]["hand"].endswith("/" + b) for x in rows):
                print(line(b, [x for x in rows if x[1]["hand"].endswith("/" + b)]))
        print()
    for arm in ("combo", "class", "history"):
        verdicts(rows_for(reads, arm), arm)
    costs(reads)
    rows = sorted(rows_for(reads, "class") + rows_for(reads, "combo"), key=lambda x: -x[0][2])
    print("worst reads (regret % pot, mix and top)")
    for s, r, n in rows[:12]:
        print(f"  {s[2]:.2f} {s[3]:.2f} {r['arm']:<7} {r['spot']} {r['node']} {r['hand']} {r['hero_combos']}c pot {r['pot']}: "
              f"near {n['acts']} -> {r['acts']}")


def pot_of(name):
    return next(x for x in HU if x[0] == name)


def combo_of(hole):
    c = cards(hole)
    return COMBO_INDEX[(min(c), max(c))]


def print_read(tag, res, view):
    print(f"{tag:<28}", [(a["action"], round(a["freq"], 3), a["ev_bb"]) for a in res["actions"]],
          "hero", round(float(view[1].sum()), 1) if view else None, res["solver"].get("iterations"), flush=True)


def per_combo(h):
    """h's solves with the flop's later streets played per combo instead of by Fast's hand buckets."""
    def solve(*x, **k):
        x = list(x)
        if len(x) > 9 and x[9] is not None:
            x[9] = dataclasses.replace(x[9], river_buckets=None, turn_buckets=False)
        else:
            k["cfg"] = dataclasses.replace(k.get("cfg") or h.adv.cfg, river_buckets=None, turn_buckets=False)
        h.seen = h._cached(*x, **k)
        return h.seen
    h.adv._solve_cached = solve


def buckets(_args):
    """Is a flop's near-zero error the weight or the buckets? Each hand at near-zero and at 1 combo, by buckets and per combo."""
    bet = lambda seat, amount: [p_("BB", "check"), p_(seat, "bet", amount=amount)]
    for name, board, hero, hole, acts in (("4bet-bb", "8s8d3c", "BB", "Ad8h", []), ("4bet-bb", "8s8d3c", "BB", "8h3h", []),
                                          ("4bet-bb", "Ks7s2h", "BB", "7c2c", bet("BTN", 15.5)),
                                          ("3bet-btn", "Jh9h4c", "BTN", "Jd4d", [p_("BB", "bet", amount=9.0)]),
                                          ("srp-btn-bb", "Ks7s2h", "BB", "7c2c", []), ("srp-btn-bb", "8s8d3c", "BB", "Ah8h", [])):
        _, n, stack, pre, _ = pot_of(name)
        hi = combo_of(hole)
        print(name, board, hole)
        for tag, w, combos in (("near-zero, buckets", NEAR_HU, False), ("1 combo, buckets", 1.0, False),
                               ("near-zero, per combo", NEAR_HU, True), ("1 combo, per combo", 1.0, True)):
            h = Harness(n, stack, hero, {"preflop": pre}, board)
            h.w = {hi: w}
            if combos:
                per_combo(h)
            res, view, _ = h.ask(hi, acts, "flop")
            print_read("  " + tag, res, view)


def the_hand(_args):
    """The hand that asked: 7-handed, UTG limps, LJ raises to 4, HJ, BB and UTG call, BB bets 6.5 on 2s4d6c, LJ calls, the 6h."""
    global CLASS_W
    pre = [p_("UTG", "call"), p_("LJ", "raise", to=4), p_("HJ", "call")] + fold("CO", "BTN", "SB") + [p_("BB", "call"), p_("UTG", "call")]
    streets = {"preflop": pre, "flop": [p_("BB", "bet", amount=6.5), p_("LJ", "call"), p_("HJ", "fold"), p_("UTG", "fold")]}
    board, hi, idx, real = "2s4d6c6h", combo_of("As4s"), A._class_index(), A.preflop_ranges
    h = Harness(7, 100, "BB", streets, board)
    for tag, w in (("near-zero", {hi: NEAR_HU}), ("A4s at 1 combo", {hi: 1.0}), ("class", {int(i): CLASS_W for i in idx["A4s"]})):
        h.w = w
        res, view, _ = h.ask(hi, [], "turn")
        print_read(tag, res, view)
    in_range = real(pre, table(7), 100)["BB"]
    add = [int(i) for k in family("A4s") for i in idx[k] if in_range[i] <= 1e-9]
    for CLASS_W in (0.25, 0.5, 1.0):
        A.preflop_ranges = with_family(real, "BB", add)
        h = Harness(7, 100, "BB", streets, board)
        h.w = {hi: NEAR_HU}
        res, view, _ = h.ask(hi, [], "turn")
        print_read(f"A2s-A6s at {CLASS_W} from preflop", res, view)
    A.preflop_ranges = real


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report", "buckets", "hand"))
    ap.add_argument("--out", default="off_range.jsonl", help="report: several, comma-separated")
    ap.add_argument("--group", default="hu", choices=("hu", "3way"))
    ap.add_argument("--flops", type=int, default=2)
    ap.add_argument("--hands", type=int, default=5)
    ap.add_argument("--history", action="store_true", help="the history arm on the turn and river too (a solve of the flop each)")
    ap.add_argument("--history-every", type=int, default=1, help="the history arm on every nth hand of a spot")
    ap.add_argument("--arms", default="combo,class,history", help="and tiny, wide")
    ap.add_argument("--pool", action="store_true", help="against the typical player (the page's cash hands), not equilibrium")
    ap.add_argument("--only", default=None, help="spots with this in their id")
    ap.add_argument("--streets", default="river,turn,flop")
    ap.add_argument("--max-spots", type=int, default=0)
    a = ap.parse_args()
    {"run": run, "report": report, "buckets": buckets, "hand": the_hand}[a.cmd](a)
