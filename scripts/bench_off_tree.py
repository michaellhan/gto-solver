"""Facing a bet or raise of a size the library's flop tree doesn't have: what each answer gives up against a long solve
that has the size in its tree.

The library's flop entries store the decisions facing a 33% and a 75% bet and a 50% raise (of the pot after calling).
A bet of another size (x) is answered here, at the decision facing it, by:
  near, interp, mdf, prop   read off the entry's decisions facing the stored sizes (gto/off_tree.py);
  fast:N                    the flop solved live in the browser's Fast tree with x in it (the advice's live solve:
                            the stored size nearest x replaced by it, Advisor._cfg_with_real_sizes), N iterations;
and priced by the reference: the flop solved in the browser's Accurate tree with x in it (the same menu), --iters
iterations (natively, one thread; a reference neither the library nor Fast's tree reads). EV given up at the decision
= the reference mix's EV - the answer's, each combo priced by the reference's EVs per action, weighted by the player's
reach there, % of the pot entering the flop; frequency error = the range's action frequencies apart (total variation,
points) and each combo's mix apart (reach-weighted total variation). The entry's own decisions facing 33% and 75%
priced by the reference solved with the library's menu are the floor (what the library gives up on its own tree
against this reference).

Cases: bet:X (the player out of position checks and the one in position bets X% in single-raised pots; in 3bet pots
the one out of position bets X% first), raise:R (a 33% bet raised R% of the pot after calling).

    python scripts/bench_off_tree.py OUT.jsonl --spots srp_btn_bb,3bp_bb_btn --n 16 \
        --cases bet:25,bet:50,bet:100,bet:150,raise:30,raise:85,raise:130
    python scripts/bench_off_tree.py OUT.jsonl --report
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import random
import sys
import time

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("GTO_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gto.native  # noqa: E402,F401
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import off_tree as OT  # noqa: E402
from gto import solver as S  # noqa: E402
from gto.cards import cards as parse_cards  # noqa: E402
import bench_flop_approx as BA  # noqa: E402
import bench_flop_seed as BS  # noqa: E402

N = 1326
METHODS = ("near", "interp", "mdf", "prop")


def texture_sample(names, n, seed):
    """n of the spot's stored flops, one from each of n equal slices of them in texture order (suits, pairing,
    ranks), a random one in each slice."""
    keys = sorted(names, key=lambda nm: fl._texture(tuple(parse_cards(nm))))
    rng = random.Random(seed)
    out = []
    for i in range(n):
        lo, hi = i * len(keys) // n, max(i * len(keys) // n + 1, (i + 1) * len(keys) // n)
        out.append(keys[rng.randrange(lo, hi)])
    return out


def first_bettor(e):
    """0 when the player out of position bets first (checks under 60% of their range at the root: a 3bet pot's
    c-bet), else 1 (the one in position bets after a check)."""
    root = next(nd for nd in e["head"]["nodes"] if not nd["path"])
    fr = root["freqs"] if isinstance(root["freqs"], dict) else dict(zip(root["names"], root["freqs"]))
    return 0 if fr.get("check", 0.0) < 0.6 else 1


def swapped(sizes, x):
    s = list(sizes)
    i = min(range(len(s)), key=lambda k: abs(s[k] - x))
    if abs(s[i] - x) / s[i] > 0.15:
        s[i] = round(x, 3)
    return s


def pick(names, kind, x=None):
    """The child name of `kind` ("bet"/"raise") nearest x (a pot share, for bets)."""
    c = [nm for nm in names if nm.startswith(kind)]
    if not c:
        return None
    if x is None or kind == "raise":
        return c[0]
    return min(c, key=lambda nm: abs(float(nm.split()[-1].rstrip("%")) / 100 - x))


def ref_path(root, case, x, fb, b0=None):
    """The path to the decision facing the case's action in a solve's tree (a raise: of the bet nearest b0)."""
    if case == "bet":
        path = [] if fb == 0 else ["check"]
        n = fl.node_at(root, path)
        return tuple(path + [pick(n.names, "bet", x)])
    path = ([] if fb == 0 else ["check"])
    n = fl.node_at(root, path)
    path.append(pick(n.names, "bet", b0 or 0.33))
    n = fl.node_at(root, path)
    return tuple(path + [pick(n.names, "raise")])


def price(node, pot):
    """(C, T) at a decision: what its player puts in to call, the pot before it."""
    p = node.player
    return float(max(node.bets) - node.bets[p]), float(node.pot_now)


def ref_node(sol, path):
    """The reference's decision: {names, strat, evs, reach (each (.., 1326)), C, T, freq (the line's share of the
    bettor's range)}."""
    n = fl.node_at(sol.root, path)
    r0, r1 = fl._reaches(sol.root, path, sol.r0, sol.r1)
    act = sol.ctx.active
    ev = sol.solver.node_action_evs(n, r0, r1)
    st = n.avg_strategy0()
    p = n.player
    full = lambda a: (lambda o: (o.__setitem__((slice(None), act), a), o)[1])(np.full((a.shape[0], N), np.nan))
    reach = np.zeros(N)
    reach[act] = r0 if p == 0 else r1
    # the bettor's share of their range on this line (their reach here over their reach at the line's start)
    q, q0 = (r1, sol.r1) if p == 0 else (r0, sol.r0)
    pa = fl.node_at(sol.root, path[:-1])
    a0, a1 = fl._reaches(sol.root, path[:-1], sol.r0, sol.r1)
    qb = a1 if p == 0 else a0
    C, T = price(n, None)
    return {"names": list(n.names), "strat": full(st), "evs": full(ev), "reach": reach, "C": C, "T": T,
            "freq": float(q.sum() / max(qb.sum(), 1e-12)), "player": p}


def entry_node(e, reaches, path, tree_root):
    """A library entry's stored decision as off_tree's source (over 1326 combos, NaN off the entry's range)."""
    nd = next((x for x in e["nodes"] if tuple(x["path"]) == tuple(path)), None)
    if nd is None:
        return None
    m = e["masks"][nd["player"]]
    idx = np.nonzero(m)[0]
    st = np.full((len(nd["names"]), N), np.nan)
    ev = np.full((len(nd["names"]), N), np.nan)
    st[:, idx], ev[:, idx] = nd["strat"], nd["evs"]
    C, T = price(fl.node_at(tree_root, path), None)
    return {"names": list(nd["names"]), "strat": st, "evs": ev, "reach": reaches[tuple(path)][0], "C": C, "T": T,
            "player": nd["player"]}


def loss(ref, P, pot):
    """(EV given up % of the pot entering the flop, range frequency TV points, combo TV points) of the mix P (A, 1326)
    at the reference's decision."""
    S_, V = ref["strat"], ref["evs"]
    w = ref["reach"]
    act = w > 0
    S_, V, w, P = S_[:, act], V[:, act], w[act], P[:, act].copy()
    bad = np.isnan(P[0])
    if bad.any():
        fill = np.nansum(P * w[None, :], axis=1) / max(float(w[~bad].sum()), 1e-12)
        if not np.isfinite(fill).all() or fill.sum() <= 0:
            fill = np.full(len(P), 1.0 / len(P))
        P[:, bad] = (fill / fill.sum())[:, None]
    P = P / np.maximum(P.sum(axis=0, keepdims=True), 1e-12)
    W = max(float(w.sum()), 1e-12)
    g = float((w * ((S_ - P) * V).sum(axis=0)).sum() / W / pot * 100)
    rf = float(0.5 * np.abs(((P - S_) * w).sum(axis=1)).sum() / W * 100)
    cf = float((w * 0.5 * np.abs(P - S_).sum(axis=0)).sum() / W * 100)
    return round(g, 4), round(rf, 2), round(cf, 2)


def lib_tree(e, pot, stack):
    cfg = fl.library_config(pot, stack)
    from types import SimpleNamespace
    board = parse_cards(e["head"]["board"])
    cfg.depth = 1
    ctx = SimpleNamespace(active=np.arange(N), n=N, cfg=cfg, root_len=3)
    return S.TreeBuilder(ctx, dead=list(board)).build(tuple(board), pot, stack)


def run_one(lib, spot, s, name, cases, iters, cks):
    pot, stack = float(spot["pot"]), float(spot["stack"])
    board = parse_cards(name)
    e = lib.entry(spot, name)
    reaches = BA.node_reaches(e, s["full0"], s["full1"])
    fb = first_bettor(e)
    troot = lib_tree(e, pot, stack)
    base = [] if fb == 0 else ["check"]
    out = {"spot": spot["id"], "flop": name, "iters": iters, "cases": {}}

    def solve(cfgname, bets, raises, it):
        cfg = BS.config(cfgname)
        cfg.bet_sizes["flop"] = list(bets)
        cfg.raise_sizes["flop"] = list(raises)
        t0 = time.time()
        sol = S.solve(board, s["full0"], s["full1"], pot, stack, cfg=cfg, time_budget=1e9, max_iters=it)
        return sol, time.time() - t0

    # the floor: the entry's own decisions against the reference solved with the library's menu
    ref_std, secs = solve("accurate", [0.33, 0.75], [0.5], iters)
    floor = {}
    for path in [tuple(base + ["bet 33%"]), tuple(base + ["bet 75%"]), tuple(base + ["bet 33%", None])]:
        if path[-1] is None:
            n = fl.node_at(ref_std.root, path[:-1])
            path = path[:-1] + (pick(n.names, "raise"),)
        src = entry_node(e, reaches, path, troot)
        if src is None:
            continue
        try:
            ref = ref_node(ref_std, path)
        except (ValueError, IndexError, AttributeError):
            continue
        floor["|".join(path)] = loss(ref, OT.onto(ref["names"], OT.by_kind(src["names"], src["strat"])), pot)
    out["floor"] = floor
    out["ref_std"] = {"expl": round(ref_std.exploitability(), 4), "s": round(secs, 1)}
    for case in cases:
        kind_, v = case.split(":")
        x = float(v) / 100
        if kind_ == "bet":
            menus = {"swap": (swapped([0.33, 0.75], x), [0.5]), "only": ([x], [0.5])}
            srcs = [entry_node(e, reaches, tuple(base + [b]), troot) for b in ("bet 33%", "bet 75%")]
            b0 = None
        else:
            # (the raise of the c-bet the entry stores a raise of: 33% where it does, else 75%)
            srcs, b0 = [], None
            for b in (0.33, 0.75):
                n = fl.node_at(troot, base + [pick(fl.node_at(troot, base).names, "bet", b)])
                src = entry_node(e, reaches, tuple(base + [pick(fl.node_at(troot, base).names, "bet", b),
                                                           pick(n.names, "raise")]), troot)
                if src is not None:
                    srcs, b0 = [src], b
                    break
            menus = {"swap": ([0.33, 0.75], [x])}
        srcs = [x_ for x_ in srcs if x_ is not None]
        row = {"refs": {}, "m": {}}
        out["cases"][case] = row
        if not srcs:
            row["error"] = "no stored source"
            continue
        refs = {}
        for rk, (bets, raises) in menus.items():
            ref_sol, secs = solve("accurate", bets, raises, iters)
            try:
                path = ref_path(ref_sol.root, kind_, x, fb, b0)
                ref = ref_node(ref_sol, path)
            except (ValueError, IndexError, AttributeError, TypeError) as ex:
                row["refs"][rk] = {"error": str(ex)}
                continue
            refs[rk] = ref
            row["refs"][rk] = {"path": list(path), "names": ref["names"], "C": ref["C"], "T": ref["T"],
                               "line_freq": round(ref["freq"], 4), "expl": round(ref_sol.exploitability(), 4),
                               "s": round(secs, 1),
                               "freq": {nm: round(float(np.nansum(ref["strat"][a] * ref["reach"]) /
                                                        max(ref["reach"].sum(), 1e-12)), 4)
                                        for a, nm in enumerate(ref["names"])}}
            row["m"][rk] = {}
            for m in METHODS:
                mix = OT.answer(srcs, ref["C"], ref["T"], m, reach=ref["reach"])
                row["m"][rk][m] = loss(ref, OT.onto(ref["names"], mix), pot)
        if not refs:
            continue
        # the advice's live solve: Fast's tree with x in it (the stored size nearest replaced), cold
        bets, raises = menus["swap"]
        cfg = BS.config("fast")
        cfg.bet_sizes["flop"], cfg.raise_sizes["flop"] = list(bets), list(raises)
        t0 = time.time()
        sol = S.solve(board, s["full0"], s["full1"], pot, stack, cfg=cfg, time_budget=1e9, max_iters=0)
        el = time.time() - t0
        for ck in cks:
            t0 = time.time()
            sol.solver.run(sol.r0, sol.r1, time_budget=1e9, max_iters=ck, min_iters=1)
            el += time.time() - t0
            sol.iters = sol.solver.iters
            try:
                n = fl.node_at(sol.root, ref_path(sol.root, kind_, x, fb, b0))
            except (ValueError, IndexError, AttributeError, TypeError):
                continue
            st = np.full((len(n.names), N), np.nan)
            st[:, sol.ctx.active] = n.avg_strategy0()
            mix = OT.by_kind(n.names, st)
            for rk, ref in refs.items():
                row["m"][rk][f"fast:{ck}"] = loss(ref, OT.onto(ref["names"], mix), pot) + (round(el, 2),)
    return out


def solve_sources(sol, base, kind_, b0=None):
    """A solve's own decisions facing each of its sizes of the case's action (its bets after `base`, or its raises
    of the bet nearest b0) as off_tree's sources, their EVs worked out."""
    n = fl.node_at(sol.root, base)
    paths = ([tuple(base) + (nm,) for nm in n.names if nm.startswith("bet")] if kind_ in ("bet", "lead") else [])
    if kind_ == "raise":
        bn = pick(n.names, "bet", b0 or 0.33)
        c = fl.node_at(sol.root, list(base) + [bn])
        paths = [tuple(base) + (bn, nm) for nm in c.names if nm.startswith("raise")]
    out = []
    for path in paths:
        d = fl.node_at(sol.root, path)
        if not isinstance(d, S.Decision):
            continue
        out.append(ref_node(sol, path))
    return out


def later_spot(lib, spot, s, name, street, rng):
    """(board, r0, r1, pot, stack) of a turn after the entry's flop line check, bet 33% (75% where 33% isn't stored),
    call, the turn card random; a river: that turn solved in Accurate's tree (400 iterations) and played check, bet
    75%, call, the river card random. None where the entry doesn't store the line."""
    e = lib.entry(spot, name)
    nodes = {tuple(n["path"]): n for n in e["nodes"]}
    fb = first_bettor(e)
    base = () if fb == 0 else ("check",)

    def row(path, nm):
        nd = nodes.get(path)
        if nd is None or nm not in nd["names"]:
            return None
        o = np.zeros(N)
        o[np.nonzero(e["masks"][nd["player"]])[0]] = nd["strat"][nd["names"].index(nm)]
        return o
    pot, stack = float(spot["pot"]), float(spot["stack"])
    troot = lib_tree(e, pot, stack)
    flop = parse_cards(name)
    for b in ("bet 33%", "bet 75%"):
        call = row(base + (b,), "call")
        bet = row(base, b)
        if call is None or bet is None:
            continue
        chk = row((), "check") if fb == 1 else None
        if fb == 1 and chk is None:
            continue
        rs = [s["full0"].copy(), s["full1"].copy()]
        if fb == 1:
            rs[0] *= chk * call
            rs[1] *= bet
        else:
            rs[0] *= bet
            rs[1] *= call
        nd = fl.node_at(troot, list(base) + [b])
        C, T = price(nd, None)
        tpot = T + C
        tstack = float(stack - (tpot - pot) / 2)
        break
    else:
        return None
    left = [c for c in range(52) if c not in flop]
    turn = rng.choice(left)
    board = list(flop) + [turn]
    bm = np.prod([1.0 - S.MEMBER_FULL[c] for c in board], axis=0)
    rs = [r * bm for r in rs]
    if street == "turn":
        return board, rs[0], rs[1], tpot, tstack, "|".join(base + (b, "call"))
    sol = S.solve(board, rs[0], rs[1], tpot, tstack, cfg=BS.config("accurate"), time_budget=1e9, max_iters=400)
    path = ["check", pick(fl.node_at(sol.root, ["check"]).names, "bet", 0.75)]
    a0, a1 = fl._reaches(sol.root, path, sol.r0, sol.r1)
    n = fl.node_at(sol.root, path)
    k = n.names.index("call")
    a0 = a0 * n.avg_strategy0()[k]
    river = rng.choice([c for c in left if c != turn])
    board = board + [river]
    bm = np.prod([1.0 - S.MEMBER_FULL[c] for c in board], axis=0)
    nd = n.children[k]
    rpot = float(max(n.bets) * 2 + tpot)
    return (board, S.expand(sol.ctx, a0) * bm, S.expand(sol.ctx, a1) * bm, rpot, tstack - max(n.bets),
            "|".join(base + (b, "call")) + " / check|" + path[1] + "|call")


def run_later(lib, spot, s, name, street, cases, iters, src_iters, cks, seed):
    rng = random.Random(f"{seed}{name}{street}")
    sp = later_spot(lib, spot, s, name, street, rng)
    if sp is None:
        return None
    board, r0, r1, pot, stack, line = sp
    std = {"turn": [0.33, 0.75, 1.25], "river": [0.33, 0.75, 1.25, 2.5]}[street]
    out = {"spot": spot["id"], "flop": name, "street": street, "board": fl.cards_str(board), "line": line,
           "pot": round(pot, 2), "stack": round(stack, 2), "iters": iters, "cases": {}}

    def cfg_of(name_, bets=None, raises=None):
        c = BS.config(name_)
        if street == "river":
            c.bet_sizes["river"] = list(std)
        if bets is not None:
            c.bet_sizes[street] = list(bets)
        if raises is not None:
            c.raise_sizes[street] = list(raises)
        return c

    def solve(c, it):
        t0 = time.time()
        sol = S.solve(board, r0, r1, pot, stack, cfg=c, time_budget=1e9, max_iters=it)
        return sol, time.time() - t0
    # the hand's own solve of the street with the speed's menu (what Advisor.instant reads at its nearest)
    src_sol, secs = solve(cfg_of("fast"), src_iters)
    out["src"] = {"expl": round(src_sol.exploitability(), 4), "s": round(secs, 1)}
    for case in cases:
        kind_, v = case.split(":")
        x = float(v) / 100
        base = ["check"] if kind_ in ("bet", "raise") else []
        if kind_ == "raise":
            menus = {"swap": (std, [x])}
        else:
            menus = {"swap": (swapped(std, x), None), "only": ([x], None)}
        try:
            srcs = solve_sources(src_sol, base, "bet" if kind_ == "lead" else kind_, 0.75)
        except (ValueError, IndexError, AttributeError):
            srcs = []
        row = {"refs": {}, "m": {}}
        out["cases"][case] = row
        if not srcs:
            row["error"] = "no source"
            continue
        refs = {}
        for rk, (bets, raises) in menus.items():
            ref_sol, secs = solve(cfg_of("accurate", bets, raises), iters)
            try:
                path = (list(base) + [pick(fl.node_at(ref_sol.root, base).names, "bet", x)] if kind_ != "raise" else
                        list(base) + [pick(fl.node_at(ref_sol.root, base).names, "bet", 0.75)])
                if kind_ == "raise":
                    path.append(pick(fl.node_at(ref_sol.root, path).names, "raise"))
                ref = ref_node(ref_sol, tuple(path))
            except (ValueError, IndexError, AttributeError, TypeError) as ex:
                row["refs"][rk] = {"error": str(ex)}
                continue
            refs[rk] = (ref, tuple(path))
            row["refs"][rk] = {"path": path, "names": ref["names"], "C": ref["C"], "T": ref["T"],
                               "line_freq": round(ref["freq"], 4), "expl": round(ref_sol.exploitability(), 4),
                               "s": round(secs, 1),
                               "freq": {nm: round(float(np.nansum(ref["strat"][a] * ref["reach"]) /
                                                        max(ref["reach"].sum(), 1e-12)), 4)
                                        for a, nm in enumerate(ref["names"])}}
            row["m"][rk] = {}
            for m in METHODS:
                mix = OT.answer(srcs, ref["C"], ref["T"], m, reach=ref["reach"])
                row["m"][rk][m] = loss(ref, OT.onto(ref["names"], mix), pot)
        if not refs:
            continue
        bets, raises = menus["swap"]
        sol = S.solve(board, r0, r1, pot, stack, cfg=cfg_of("fast", bets if kind_ != "raise" else None, raises),
                      time_budget=1e9, max_iters=0)
        el = 0.0
        for ck in cks:
            t0 = time.time()
            sol.solver.run(sol.r0, sol.r1, time_budget=1e9, max_iters=ck, min_iters=1)
            el += time.time() - t0
            for rk, (ref, path) in refs.items():
                try:
                    n = fl.node_at(sol.root, path)
                except (ValueError, IndexError, AttributeError):
                    continue
                st = np.full((len(n.names), N), np.nan)
                st[:, sol.ctx.active] = n.avg_strategy0()
                row["m"][rk][f"fast:{ck}"] = loss(ref, OT.onto(ref["names"], OT.by_kind(n.names, st)), pot) + (round(el, 2),)
    return out


def report(path, min_line=0.02):
    rows = [json.loads(x) for x in open(path)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["spot"], r.get("street", "flop"))].append(r)
    for (sid, street), rs in by.items():
        print(f"{sid} {street}: {len(rs)} spots (reference: Accurate tree, {rs[0]['iters']} iterations)")
        fl_ = collections.defaultdict(list)
        for r in rs:
            for k, v in r.get("floor", {}).items():
                fl_[k].append(v)
        for k, v in fl_.items():
            a = np.array(v)
            print(f"  floor {k:28s} {len(a):3d} given up {a[:, 0].mean():.3f} (median {np.median(a[:, 0]):.3f}), "
                  f"range freq {a[:, 1].mean():.1f}, combo {a[:, 2].mean():.1f}")
        cases = sorted({c for r in rs for c in r["cases"]}, key=lambda c: (c.split(":")[0], float(c.split(":")[1])))
        for c in cases:
            for rk in ("swap", "only"):
                ok = [r["cases"][c] for r in rs if c in r["cases"] and rk in r["cases"][c].get("m", {})]
                lf = np.array([o["refs"][rk]["line_freq"] for o in ok])
                ok = [o for o in ok if o["refs"][rk]["line_freq"] >= min_line]
                if not ok:
                    continue
                print(f"  {c} vs {rk} menu: {len(ok)} flops of {len(lf)} with the line at {min_line * 100:.0f}%+ of the "
                      f"bettor's range (taken by {lf.mean() * 100:.0f}% on average), ref expl "
                      f"{np.mean([o['refs'][rk]['expl'] for o in ok]):.3f}")
                for m in ok[0]["m"][rk]:
                    a = np.array([o["m"][rk][m][:3] for o in ok if m in o["m"][rk]])
                    t = [o["m"][rk][m][3] for o in ok if m in o["m"][rk] and len(o["m"][rk][m]) > 3]
                    print(f"    {m:10s} given up {a[:, 0].mean():6.3f} (median {np.median(a[:, 0]):.3f}, 90th "
                          f"{np.percentile(a[:, 0], 90):.3f}), range freq {a[:, 1].mean():5.1f}, combo "
                          f"{a[:, 2].mean():5.1f}" + (f"   {np.mean(t):.2f} s" if t else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--spots", default="srp_btn_bb,3bp_bb_btn")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--cases", default="bet:25,bet:50,bet:100,bet:150,raise:30,raise:85,raise:130")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--checkpoints", default="50,100,200,300,500")
    ap.add_argument("--street", default="flop", help="flop, or turn / river: after the entry's flop line, a random card")
    ap.add_argument("--src-iters", type=int, default=300, help="(turn, river) the hand's own solve's iterations")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.out)
    lib = fl.FlopLibrary.from_dir()
    spots = {x["id"]: x for x in lib.index()["spots"]}
    done = set()
    if os.path.exists(a.out):
        done = {(d["spot"], d["flop"]) for d in map(json.loads, open(a.out))}
    cks = [int(x) for x in a.checkpoints.split(",")]
    for sid in a.spots.split(","):
        spot = spots[sid]
        s = BA.setup(sid)
        for name in texture_sample(spot["flops"], a.n, a.seed):
            if (sid, name) in done:
                continue
            t0 = time.time()
            if a.street == "flop":
                r = run_one(lib, spot, s, name, a.cases.split(","), a.iters, cks)
            else:
                r = run_later(lib, spot, s, name, a.street, a.cases.split(","), a.iters, a.src_iters, cks, a.seed)
                if r is None:
                    continue
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(sid, name, f"{time.time() - t0:.0f}s",
                  {c: {m: v[0] for m, v in d.get("m", {}).get("swap", {}).items() if m in ("near", "mdf", "fast:300")}
                   for c, d in r["cases"].items()}, flush=True)
    report(a.out)


if __name__ == "__main__":
    main()
