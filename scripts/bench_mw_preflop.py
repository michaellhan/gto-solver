"""A reference for multiway preflop calls: each hand class's EV of calling, from flop solves, against what the advice
(Advisor._priced) continues with. No published solve covers limpers and cold calls, so the price and equity rules in
_priced had nothing to be held to; this gives them something.

For each spot (SPOTS) the players' ranges entering the flop are frozen once (setup: the advice's at the time it ran, kept in
the cache and in the data file, so changing the advice doesn't move its own reference), and for each way the players
still to act can answer hero's call (branches) a sample of flops is solved with the multiway solver on the site's game
(3-way: MW_LATER, MW_FLOP_BETS; 4 and more: MW4_STREETS, one bet a street):
- Hero's range is what it calls with, every class it had before its action added at EPS: a hand outside the range gets
  its counterfactual value (the trick in scripts/pf_equilibrium.py). A player still to act has its whole range at EPS
  the same way, so its own answer is measured too.
- EV(call) of a class = its share of the pot over the flops - the chips it adds. Where a limper (or an opener behind a
  3bet, or a seat behind hero) still has to act, each branch "these fold, those continue with what continues" is weighted by
  how often it happens; a branch where one of them raises counts as losing the call (hero folds: right for the
  marginal hands this is about, too low for the best ones).
- The flop sample's noise is mostly which flops a hand hit, so a class's share is its all-in equity over all boards
  (advisor._preflop_equity_multi, as the advice reads it) plus its excess (share - equity on the same deals, flop by
  flop). The excess is what the solve adds, and it hardly varies with the flop's cards hitting the hand.
- Two rounds: round 1 (fewer flops and iterations) takes the advice's calls; round 2 (other flops) the +EV hands of round 1,
  hero's and those of the players still to act (never the advice's), and it is what's reported.
- Each flop is one job and one small npz in the cache, so a stopped run resumes where it stopped.

The river: the solve plays the turn and river by hand buckets, the river on the turn's boards without its card, so a
flush that comes in on the river isn't paid as it should be (--river-check measures that, and gto/advisor.py RIVER_FIX is
the heads-up form of it). `river_fix` in the report is EV with that correction applied to the class's equity share.

    python scripts/bench_mw_preflop.py --minutes 9            # solve what's left (resumable, one process)
    python scripts/bench_mw_preflop.py --wide --minutes 9     # the same with every other seat's range a bit wider
    python scripts/bench_mw_preflop.py --freeze               # data file from the cache, the advice as it is as "before"
    python scripts/bench_mw_preflop.py --report               # the table, the advice as it is now beside it
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import gto.native  # noqa: F401  (one BLAS thread per process)
import numpy as np

EPS = 1e-3
CACHE = os.path.join(ROOT, ".pf_cache", "mw_preflop")
DATA = os.path.join(ROOT, "docs", "mw_preflop_reference.json")
K = 169
RK = (0.05, 4.0)
RAMP = 0.3         # bb of EV over which a hand goes into the next round's range
MIN_BRANCH = 0.06  # a branch rarer than this isn't solved (the others share its weight)
MIN_CONT = 0.08    # round 1 takes at least this share of a seat-to-act's range continuing, so a branch measures it
WIDEN = 0.5        # --wide: each other seat's range grows by this share of its combos, the next best by equity
ROUNDS = {1: dict(flops=4, offset=0.25, iters3=100, iters4=100), 2: dict(flops=10, offset=0.75, iters3=200, iters4=150)}
THIN = 0.25        # a branch weighing less than this gets half the flops (its noise counts by its weight)

# name: (seats, hero, the line to hero (folds filled in), rake, what a published solve says)
SPOTS = {
    # the hand that found it, and the same with nobody owing
    "bg7": (7, "BB", "UTG call, LJ raise 4, HJ call", None, ""),
    "bg7_nolimp": (7, "BB", "LJ raise 4, HJ call", None, ""),
    # a raise over limpers, heads-up behind it
    "iso1": (6, "BB", "UTG call, BTN raise 4", None, ""),
    "iso2": (6, "BB", "UTG call, HJ call, BTN raise 5", None, ""),
    "iso_big": (6, "BB", "UTG call, CO raise 6", None, ""),
    # limp + raise + call
    "lrc6": (6, "BB", "UTG call, HJ raise 4, CO call", None, ""),
    "lrc6_btn": (6, "BB", "HJ call, CO raise 4, BTN call", None, ""),
    "lrc6_sb": (6, "BB", "HJ call, BTN raise 4, SB call", None, ""),
    "lrc2": (6, "BB", "UTG call, HJ call, CO raise 5, BTN call", None, ""),
    # an open and a cold call: the charts' spots, anchored where a solve is published
    "cc_btn_sb": (6, "BB", "BTN raise 2.5, SB call", RK,
                  "Poker Academy NL200: calls 16.1%, raises 12.7% at a 20% price"),
    "cc_co_btn": (6, "BB", "CO raise 2.5, BTN call", None, "GTO Wizard NL500 40bb: raises 9.8%"),
    "cc_utg_co": (6, "BB", "UTG raise 2.5, CO call", None, ""),
    "cc_4way": (6, "BB", "UTG raise 2.5, HJ call, BTN call", None, ""),
    # an opener behind a 3bet
    "cold3": (6, "BB", "CO raise 2.5, BTN raise 7.5", None, ""),
    # the small blind, the big blind behind it
    "sb_iso": (6, "SB", "CO call, BTN raise 4", None, ""),
}


def table(spot):
    from gto import preflop as pf
    return pf.Table(pf.SEATS[SPOTS[spot][0]], rake=SPOTS[spot][3])


def full_line(T, hero, text):
    """The line to hero, with the folds in between."""
    from gto.advisor import replay_hand
    from tests.preflop_refs import _line
    pre = _line(text, T)
    while (st := replay_hand(T, 100, {"preflop": pre}, 1)[0]).to_act != hero:
        pre.append({"pos": st.to_act, "action": "fold"})
    return pre, st


def widen(w, frac=WIDEN):
    """w plus the next best hands by equity against w, frac of its combos more."""
    from gto.advisor import _preflop_equity
    from gto.pf_classes import CLASS_OF
    eq = np.nan_to_num(_preflop_equity(w), nan=0.0)
    room = np.bincount(CLASS_OF, 1.0 - w, K)
    size = np.bincount(CLASS_OF, np.ones(len(w)), K)
    eqc = np.bincount(CLASS_OF, eq, K) / size
    add, left = np.zeros(K), frac * float(w.sum())
    for c in np.argsort(-eqc):
        take = min(left, room[c])
        add[c] = take / size[c]
        left -= take
        if left <= 1e-9:
            break
    return np.minimum(1.0, w + add[CLASS_OF])


def best_of(w, combos):
    """`combos` of w, the best hands by equity against random hands, by whole classes."""
    from gto.advisor import _preflop_equity
    from gto.pf_classes import CLASS_OF
    eq = np.nan_to_num(_preflop_equity(np.ones(len(w))), nan=0.0)
    out = np.zeros(len(w))
    for c in np.argsort(-(np.bincount(CLASS_OF, eq, K))):
        out[CLASS_OF == c] = w[CLASS_OF == c]
        if out.sum() >= combos:
            break
    return out


def setup(spot, wide=False):
    """The spot frozen: everyone's ranges and what each seat still to act does. Saved; later runs read that."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"setup_{spot}{'_wide' if wide else ''}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["meta"].item(), {k[2:]: z[k] for k in z.files if k.startswith("r_")}
    from gto import preflop as pf
    from gto.advisor import preflop_ranges, preflop_strategy, replay_hand
    n, hero, text, rake, _ = SPOTS[spot]
    T = table(spot)
    pre, st = full_line(T, hero, text)
    call = st.cur_bet - st.bets[hero]
    arr = preflop_ranges(pre, T, 100)
    s = preflop_strategy(hero, pre, T, 100)
    # who answers hero's call
    line = pre + [{"pos": hero, "action": "call"}]
    arr2 = preflop_ranges(line, T, 100)
    pend, rng = [], {f"seat_{p}": arr[p].copy() for p in arr if p != hero}
    rng["hero_all"] = arr[hero].copy()
    rng["hero_call"] = arr[hero] * s["call"]
    while True:
        st2 = replay_hand(T, 100, {"preflop": line}, 1)[0]
        if st2.closed:
            break
        p = st2.to_act
        sp = preflop_strategy(p, line, T, 100)
        w = arr2[p]
        tot = max(float(w.sum()), 1e-12)
        cont = w * sp.get("call", 0.0)
        if float(cont.sum()) / tot < MIN_CONT:   # (the advice has it folding: its best hands, so the solve can say)
            cont = best_of(w, MIN_CONT * tot)
        pend.append(dict(pos=p, call=float(cont.sum()) / tot, raise_=float((w * sp.get("raise", 0.0)).sum() / tot),
                         owed=float(st2.cur_bet - st2.bets[p])))
        rng[f"pend_all_{p}"] = w.copy()
        rng[f"pend_cont_{p}"] = cont
        rng[f"pend_raise_{p}"] = w * sp.get("raise", 0.0)
        line = line + [{"pos": p, "action": "fold"}]
    closed = replay_hand(T, 100, {"preflop": pre + [{"pos": hero, "action": "call"}]}, 1)[0]
    level = float(closed.cur_bet)
    # who is in: everyone left but those still to answer
    pending = [d["pos"] for d in pend]
    live = [p for p in closed.active if p != hero and p not in pending]
    if wide:
        for p in live:
            rng[f"seat_{p}"] = widen(rng[f"seat_{p}"])
        for p in pending:
            rng[f"pend_cont_{p}"] = np.minimum(rng[f"pend_all_{p}"], widen(rng[f"pend_cont_{p}"]))
    order = pf.postflop_order(list(T))
    meta = dict(spot=spot, hero=hero, call=float(call), level=level, live=live, pend=pend, order=order,
                dead=float(closed.pot_start + sum(closed.bets.values()) - level * (1 + len(live))
                           - sum(closed.bets[p] for p in pending)),
                line=[dict(a) for a in pre], wide=wide)
    np.savez(path, meta=np.array(meta, dtype=object), **{f"r_{k}": v for k, v in rng.items()})
    return meta, rng


def branches(meta, rates=None):
    """[(the seats that continue, weight)], and the weight of someone raising. `rates`: {seat: continue rate} of the
    round before, for the advice's."""
    pend = meta["pend"]
    out = [((), 1.0)]
    no_raise = 1.0
    for d in pend:
        c = (rates or {}).get(d["pos"], d["call"])
        r = d["raise_"]
        f = max(1.0 - c - r, 0.0)
        no_raise *= c + f
        out = [(k + (d["pos"],), w * c) for k, w in out] + [(k, w * f) for k, w in out]
    keep = [(k, w) for k, w in out if w >= MIN_BRANCH * no_raise]
    tot = sum(w for _, w in keep)
    return [(k, w * no_raise / tot) for k, w in keep], 1.0 - no_raise


def flops(n, offset):
    """n flops at equal spacing over the textures (build_realization.flop_sample's strata), equally weighted."""
    path = os.path.join(CACHE, "flops.npz")
    if os.path.exists(path):
        z = np.load(path)
        f, w = z["f"], z["w"]
    else:
        import build_realization as B
        got = B.flop_sample(272)
        f, w = np.array([x[0] for x in got]), np.array([x[1] for x in got])
        np.savez(path, f=f, w=w)
    cum = np.cumsum(w) / w.sum()
    idx = np.unique(np.searchsorted(cum, (np.arange(n) + offset) / n))
    return [tuple(int(c) for c in f[i]) for i in idx]


def branch_flops(rnd, weight):
    cfg = ROUNDS[rnd]
    return flops(cfg["flops"] if weight >= THIN or rnd == 1 else cfg["flops"] // 2, cfg["offset"])


def mw_config(n_players, rake):
    from gto import advisor as A
    from gto.multiway_solver import MWConfig
    r = dict(rake_rate=rake[0], rake_cap=rake[1]) if rake else {}
    if n_players <= 3:
        return MWConfig(**dict(A.MW_LATER, bet=A.MW_FLOP_BETS), **r)
    return MWConfig(bet=A.MW4_BETS, max_raises=1, **A.MW4_STREETS["flop"], **r)


def class_sums(s, p):
    """Player p on this flop, per class: its value (bb, chips-weighted by the deals), the same deals' value checked
    down (all-in equity x the pot), and the deals' mass; each combo counted alike."""
    from gto.mwcore import EV
    from gto.pf_classes import CLASS_OF
    R = s.reach0[:, 0, :]
    v = s._core_eval(s.root, EV, p, R)
    if v is None:
        v = s._pass(s.reach0, "avg")[p][0]
    act = s.hands.active
    m = s._deal_mass(s.reach0, p)[0]
    leaf = s.builder._runout([s.board], tuple(range(s.np)), s.pot_start, [0.0] * s.np)
    e = s._walk(leaf, s.reach0, "avg")[p][0]
    c = CLASS_OF[act]
    return np.bincount(c, v, K), np.bincount(c, e, K), np.bincount(c, m, K)


def job_path(spot, wide, rnd, K_seats, flop):
    tag = "+".join(K_seats) or "none"
    return os.path.join(CACHE, f"{spot}{'_wide' if wide else ''}_r{rnd}_{tag}_{'-'.join(map(str, flop))}.npz")


def solve_job(path, board, ranges, players, pot, stack, iters, rake):
    from gto.multiway_solver import MultiwaySolver
    cfg = mw_config(len(players), rake)
    cfg.leaf_rivers = 4
    t0 = time.time()
    s = MultiwaySolver(board, ranges, pot, stack, cfg=cfg)
    s.run(iters=iters)
    gap = s.nash_conv()
    out = {}
    for p, name in enumerate(players):
        out[f"v_{name}"], out[f"e_{name}"], out[f"m_{name}"] = class_sums(s, p)
    np.savez(path + ".tmp.npz", gap=gap, seconds=time.time() - t0, iters=iters, **out)
    os.replace(path + ".tmp.npz", path)
    return gap


def branch_game(meta, rng, seats, hero_w, pend_w):
    """(the players in postflop order, their ranges, the pot, the stack behind) of the branch where `seats` continue."""
    players = [p for p in meta["order"] if p == meta["hero"] or p in meta["live"] or p in seats]
    rs = []
    for p in players:
        if p == meta["hero"]:
            rs.append(hero_w)
        elif p in seats:
            rs.append(pend_w[p])
        else:
            rs.append(rng[f"seat_{p}"])
    # the pot: the chips of everyone who folded, and the level from each player left (hero's call in it)
    pot = meta["dead"] + meta["level"] * len(players)
    for d in meta["pend"]:
        if d["pos"] not in seats:                # a seat still to act that folds leaves what it had in
            pot += meta["level"] - d["owed"]
    return players, rs, pot, 100.0 - meta["level"]


_ROUND: dict = {}


def round_inputs(spot, wide, rnd):
    """Hero's range and each seat-to-act's for this round, with their rates. Round 1: the advice's. Round 2: round 1's
    +EV hands (None until round 1 is done)."""
    if (spot, wide, rnd) not in _ROUND or _ROUND[spot, wide, rnd] is None:
        _ROUND[spot, wide, rnd] = _round_inputs(spot, wide, rnd)
    return _ROUND[spot, wide, rnd]


def _round_inputs(spot, wide, rnd):
    meta, rng = setup(spot, wide)
    w_all = rng["hero_all"]
    if rnd == 1:
        hero_w = rng["hero_call"] + EPS * w_all
        pend_w = {d["pos"]: rng[f"pend_cont_{d['pos']}"] + EPS * rng[f"pend_all_{d['pos']}"] for d in meta["pend"]}
        return meta, rng, hero_w, pend_w, None
    res = estimate(spot, wide, 1)
    if res is None or any(b["flops"] < ROUNDS[1]["flops"] for b in res["branches"]):
        return None
    from gto.pf_classes import CLASS_OF
    ramp = lambda ev: np.clip(0.5 + ev / RAMP, 0.0, 1.0)[CLASS_OF]
    hero_w = w_all * (ramp(res["ev"]) + EPS)
    pend_w, rates = {}, {}
    for d in meta["pend"]:
        p = d["pos"]
        a = rng[f"pend_all_{p}"]
        ev = res["pend"].get(p)
        if ev is None:
            pend_w[p] = rng[f"pend_cont_{p}"] + EPS * a
            continue
        c = a * ramp(ev)
        c = np.maximum(c - rng[f"pend_raise_{p}"], 0.0)
        pend_w[p] = c + EPS * a
        rates[p] = float(c.sum() / max(a.sum(), 1e-12))
    return meta, rng, hero_w, pend_w, rates


def jobs(spot, wide, rnd):
    got = round_inputs(spot, wide, rnd)
    if got is None:
        return []
    meta, rng, hero_w, pend_w, rates = got
    cfg = ROUNDS[rnd]
    out = []
    for seats, weight in branches(meta, rates)[0]:
        players, rs, pot, stack = branch_game(meta, rng, seats, hero_w, pend_w)
        iters = cfg["iters3"] if len(players) <= 3 else cfg["iters4"]
        for f in branch_flops(rnd, weight):
            path = job_path(spot, wide, rnd, seats, f)
            if not os.path.exists(path):
                out.append((path, f, rs, players, pot, stack, iters, SPOTS[spot][3]))
    return out


def estimate(spot, wide, rnd, resample=None):
    """Per class hero's EV of calling (and a seat-to-act's of its own call) from the flops solved so far, None until a
    branch has flops. `resample`: a numpy generator, each branch's flops drawn again with it (the bootstrap)."""
    from gto.pf_classes import CLASS_OF
    got = round_inputs(spot, wide, rnd)
    if got is None:
        return None
    meta, rng, hero_w, pend_w, rates = got
    hero = meta["hero"]
    br, p_raise = branches(meta, rates)
    ev = np.zeros(K) - p_raise * meta["call"]
    pend_ev = {d["pos"]: np.zeros(K) for d in meta["pend"]}
    pend_w_sum = {d["pos"]: 0.0 for d in meta["pend"]}
    info = []
    gaps, secs = [], 0.0
    for seats, weight in br:
        players, rs, pot, stack = branch_game(meta, rng, seats, hero_w, pend_w)
        recs = []
        fl = branch_flops(rnd, weight)
        draw = [1] * len(fl) if resample is None else np.bincount(resample.integers(0, len(fl), len(fl)), minlength=len(fl))
        for f, n in zip(fl, draw):
            path = job_path(spot, wide, rnd, seats, f)
            if os.path.exists(path):
                z = np.load(path)
                if n:
                    recs.append((n, z))
                gaps.append(float(z["gap"]))
                secs += float(z["seconds"])
        if not recs:
            return None

        def excess(name):
            num = sum(n * (z[f"v_{name}"] - z[f"e_{name}"]) for n, z in recs)
            den = sum(n * z[f"m_{name}"] for n, z in recs)
            return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0), \
                np.where(den > 0, sum(n * z[f"e_{name}"] for n, z in recs) / np.maximum(den, 1e-300), 0.0)

        def share(name):
            from gto.advisor import _preflop_equity_multi, _preflop_equity
            key = ("eq", spot, wide, rnd, seats, name)
            if key not in _EQ:
                bare = [(rng[f"seat_{q}"] if q in meta["live"] else
                         (hero_w if q == hero else pend_w[q])) for q in players if q != name]
                _EQ[key] = np.nan_to_num(_preflop_equity_multi(bare) if len(bare) > 1 else _preflop_equity(bare[0]))
                _EQ[key] = np.bincount(CLASS_OF, _EQ[key], K) / np.bincount(CLASS_OF, np.ones(len(CLASS_OF)), K)
            x, e = excess(name)
            rake = SPOTS[spot][3]
            net = pot if not rake else pot - min(rake[0] * pot, rake[1] if rake[1] is not None else 1e9)
            return x + net * _EQ[key], x, e / max(pot, 1e-9), _EQ[key]

        sh, x, eflop, eq = share(hero)
        ev += weight * (sh - meta["call"])
        info.append(dict(seats=list(seats), weight=weight, players=players, pot=pot, flops=len(recs), excess=x, eq=eq,
                         eq_flop=eflop))
        for d in meta["pend"]:
            p = d["pos"]
            if p in seats:
                pend_ev[p] = pend_ev[p] + weight * (share(p)[0] - d["owed"])
                pend_w_sum[p] += weight
    pend = {p: v / pend_w_sum[p] for p, v in pend_ev.items() if pend_w_sum[p] > 0}
    return dict(ev=ev, pend=pend, branches=info, p_raise=p_raise, gap=float(np.mean(gaps)), seconds=secs, meta=meta,
                rng=rng, rates=rates)


_EQ: dict = {}


def run(args):
    t0 = time.time()
    limit = args.minutes * 60
    names = args.spots.split(",") if args.spots else list(SPOTS)
    for rnd in (1, 2):
        for spot in names:
            for j in jobs(spot, args.wide, rnd):
                if time.time() - t0 > limit:
                    print("out of time", flush=True)
                    return
                gap = solve_job(*j)
                print(f"{time.time() - t0:6.0f}s r{rnd} {os.path.basename(j[0])} gap {gap:.2f}%", flush=True)
    print("all solved", flush=True)


def hero_weights(rng):
    """Combos of each class hero can hold here."""
    from gto.pf_classes import to_classes
    return to_classes(rng["hero_all"])


def advice(spot, res):
    """The advice as the code stands: (call, raise) per class as a share of the hero's combos of the class."""
    from gto.advisor import preflop_strategy
    from gto.pf_classes import to_classes, CLASSES
    from gto import preflop as pf
    T = table(spot)
    pre = res["meta"]["line"]
    s = preflop_strategy(res["meta"]["hero"], pre, T, 100)
    w = res["rng"]["hero_all"]
    den = np.maximum(to_classes(w), 1e-12)
    return to_classes(w * s["call"]) / den, to_classes(w * s.get("raise", np.zeros(pf.N_COMBOS))) / den


def score(ev, call, raise_, n):
    """(share continuing, EV given up per hand in bb, and split: calling -EV hands, folding +EV hands). A raise isn't judged
    by the EV of a call."""
    fold = np.maximum(1.0 - call - raise_, 0.0)
    bad = call * np.maximum(-ev, 0.0)
    missed = fold * np.maximum(ev, 0.0)
    tot = n.sum()
    return float(((call + raise_) * n).sum() / tot), float((bad * n).sum() / tot), float((missed * n).sum() / tot)


def bootstrap(spot, wide, rnd=2, n=60, seed=5):
    """Spread of the share of hands with +EV calls over resamples of the flops."""
    r = np.random.default_rng(seed)
    shares, evs = [], []
    wt = None
    for _ in range(n):
        res = estimate(spot, wide, rnd, r)
        if res is None:
            return None
        if wt is None:
            wt = hero_weights(res["rng"])
        shares.append(float(((res["ev"] > 0) * wt).sum() / wt.sum()))
        evs.append(res["ev"])
    return float(np.std(shares)), np.std(evs, axis=0)


CHECK = ("A4s", "A5s", "A9s", "K9s", "T9s", "76s", "54s", "33", "66", "99", "AJo", "KQo", "K9o")


def freeze(args):
    from gto.pf_classes import CLASSES
    path = args.data
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            out = json.load(fh)
    out.setdefault("settings", {}).update(
        eps=EPS, rounds={str(k): v for k, v in ROUNDS.items()}, widen=WIDEN, min_branch=MIN_BRANCH, classes=list(CLASSES))
    for spot in SPOTS:
        for wide in (False, True):
            res = estimate(spot, wide, 2)
            if res is None or any(b["flops"] < 0.8 * len(branch_flops(2, b["weight"])) for b in res["branches"]):
                continue
            key = spot + ("_wide" if wide else "")
            n = hero_weights(res["rng"])
            sd, se = bootstrap(spot, wide)
            call, raise_ = advice(spot, res)
            prev = out.get("spots", {}).get(key, {})
            out.setdefault("spots", {})[key] = {
                "seats": SPOTS[spot][0], "hero": SPOTS[spot][1], "line": SPOTS[spot][2], "rake": SPOTS[spot][3],
                "published": SPOTS[spot][4], "call": res["meta"]["call"], "wide": wide,
                "pending": res["meta"]["pend"], "rates": res["rates"], "p_raise": res["p_raise"],
                "branches": [dict(seats=b["seats"], weight=round(b["weight"], 4), players=b["players"], pot=b["pot"],
                                  flops=b["flops"]) for b in res["branches"]],
                "gap_pct_pot": round(res["gap"], 3), "solve_seconds": round(res["seconds"]),
                "combos": [round(float(x), 2) for x in n],
                "ev": [round(float(x), 3) for x in res["ev"]], "se": [round(float(x), 3) for x in se],
                "share": round(float(((res["ev"] > 0) * n).sum() / n.sum()), 4), "share_sd": round(sd, 4),
                "advisor_before": prev.get("advisor_before") or {"call": [round(float(x), 3) for x in call],
                                                                  "raise": [round(float(x), 3) for x in raise_]},
            }
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"{path}: {os.path.getsize(path) / 1024:.0f} KB, {len(out.get('spots', {}))} spots")


def report(args):
    from gto.pf_classes import CLASSES
    with open(args.data) as fh:
        D = json.load(fh)
    idx = {c: i for i, c in enumerate(CLASSES)}
    tot = np.zeros(3)
    n_sp = 0
    print(f"{'spot':16s} {'+EV calls':>10s} {'advice':>7s} {'before':>7s}   EV lost bb/hand now (before)   "
          f"{' '.join(f'{c:>5s}' for c in ('A4s', 'A9s', '76s', '33', 'K9o'))}")
    for key, r in D["spots"].items():
        spot = key[:-5] if key.endswith("_wide") else key
        if key != spot and not args.wide:
            continue
        ev, n = np.array(r["ev"]), np.array(r["combos"])
        res = estimate(spot, r["wide"], 2)
        call, raise_ = advice(spot, res) if res else (np.zeros(K), np.zeros(K))
        a = score(ev, call, raise_, n)
        b = score(ev, np.array(r["advisor_before"]["call"]), np.array(r["advisor_before"]["raise"]), n)
        print(f"{key:16s} {r['share'] * 100:6.1f}±{r['share_sd'] * 100:3.1f} {a[0] * 100:7.1f} {b[0] * 100:7.1f}   "
              f"{a[1] + a[2]:.3f} ({b[1] + b[2]:.3f}) {'':10s}  "
              + " ".join(f"{ev[idx[c]]:5.2f}" for c in ("A4s", "A9s", "76s", "33", "K9o")))
        tot += np.array([a[1] + a[2], b[1] + b[2], 1.0])
    print(f"mean EV lost per hand: now {tot[0] / tot[2]:.3f} bb, before {tot[1] / tot[2]:.3f} bb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=9.0)
    ap.add_argument("--spots")
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()
    if args.freeze:
        freeze(args)
    elif args.report:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
