"""A turn after a library flop whose entry has no turn file, read off another spot's stored turn: what that gives up.

A library flop's turn answers at once from the entry's turn file (flop_library.encode_turns, advisor.instant_from_store)
where the spot has one; the spots solved before turn files (and the pots another table size borrows from them) wait
for the turn's live solve. Here a case's turn (a preflop line in a game, the flop line the hand took, a turn card) is
answered off a spot that has turn files on the same flop (the source), and priced:

* reference: the turn solved natively in the Accurate tree (native.config("accurate"), --iters iterations, one
  thread) with the two ranges the case's own library entry narrows the flop line to (its stored flop strategies:
  what the advisor solves the turn with);
* near: the source's stored turn for the case's combos (fl.turn_strategies on the same flop line, the card read as its
  class's played card), each action at the live tree's nearest; over the combos the source stores ("unseen": the
  reach-weighted share it doesn't, which the advisor leaves to the live solve);
* own: the case's own spot's stored turn where it has one (the floor: what a stored turn gives up for its own pot);
* fast50 / fast100: the case's turn solved live in Fast's tree to 50 / 100 iterations (what replaces an answer at
  once: advisor.STORE_UNTIL).

Priced at the turn's first decision and in position after a check (the nodes a turn file holds): each combo's mix by
the reference's EVs per action (bench_flop_approx.given_up's measure), the acting player's reach, % of the turn pot.
One JSON line a (case, flop, line, turn) to OUT; resumable.

    python scripts/bench_turn_near.py OUT.jsonl --cases 6max_co_bb,8max_co_bb --n 6
    python scripts/bench_turn_near.py OUT.jsonl --report
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
import gto.native as gn  # noqa: E402
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import flop_approx as fa  # noqa: E402
from gto.advisor import match_action  # noqa: E402
from gto.cards import ALL_COMBOS, cards as parse_cards, cards_str  # noqa: E402
from gto.solver import solve, Decision, MEMBER_FULL  # noqa: E402

# case -> (Format tag, preflop line in that game, the source spot with turn files)
CASES = {
    "6max_btn_bb": ("", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", "srp_btn_bb"),
    "6max_co_bb": ("", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c", "srp_btn_bb"),
    "6max_hj_bb": ("", "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:f,BB:c", "srp_btn_bb"),
    "6max_utg_bb": ("", "UTG:r2.5,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_btn_bb"),
    "6max_btn_sb": ("", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:f", "srp_btn_bb"),
    "6max_co_sb": ("", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:f", "srp_btn_bb"),
    "8max_co_bb": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c", "srp_btn_bb"),
    "8max_lj_bb": ("8max", "UTG:f,UTG1:f,LJ:r2.5,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_btn_bb"),
    "9max_utg_bb": ("9max", "UTG:r2.5,UTG1:f,UTG2:f,LJ:f,HJ:f,CO:f,BTN:f,SB:f,BB:c", "srp_btn_bb"),
    "6max_3bp_bb_co": ("", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:r10,CO:c", "3bp_bb_co"),
    "6max_3bp_bb_btn": ("", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c", "3bp_bb_co"),
    "6max_3bp_co_hj": ("", "UTG:f,HJ:r2.5,CO:r7.5,BTN:f,SB:f,BB:f,HJ:c", "3bp_co_hj"),
    "6max_3bp_btn_co": ("", "UTG:f,HJ:f,CO:r2.5,BTN:r7.5,SB:f,BB:f,CO:c", "3bp_co_hj"),
    "6max_3bp_hj_utg": ("", "UTG:r2.5,HJ:r7.5,CO:f,BTN:f,SB:f,BB:f,UTG:c", "3bp_co_hj"),
    "8max_3bp_bb_btn": ("8max", "UTG:f,UTG1:f,LJ:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c", "3bp_bb_co"),
    "6max_3bp_sb_btn": ("", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:r10,BB:f,BTN:c", "3bp_bb_co"),
    "6max_3bp_sb_co": ("", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:r10,BB:f,CO:c", "3bp_bb_co"),
}
RANKS = "23456789TJQKA"


def case_setup(lib, case):
    """(setup, the library spot the advisor reads the case's flop off, borrow dict or None, the source spot)."""
    tag, line, src = CASES[case]
    fmt = fl.Format.parse(tag)
    s = fl.spot_setup(line, fmt.players, fmt.stack, fmt.ante, fmt.straddle)
    spots = {x["id"]: x for x in lib.index()["spots"]}
    sp = lib.spot(fl.range_fp(s["full0"]), fl.range_fp(s["full1"]), s["pot"], s["stack"])
    borrow = None
    if sp is None:
        borrow = fa.near_exact(lib, s["full0"], s["full1"], s["pot"], s["stack"], fmt.players)
        sp = borrow and borrow["spot"]
    return s, sp, borrow, spots[src]


def flop_lines(sol):
    """[(path, r0, r1, share)] of a LibrarySolution: the flop lines its stored nodes take to the turn (no raise), with
    the ranges narrowed by its stored strategies, most reached first (share of the hand pairs, card removal aside)."""
    out = []
    tot = float(sol.r0.sum() * sol.r1.sum())

    def walk(n, path, a0, a1):
        if not isinstance(n, Decision) or id(n) not in sol.stored:
            return
        st = n.avg_strategy()[0].astype(np.float64)
        for k, nm in enumerate(n.names):
            if nm.startswith("raise") or nm == "all-in" or nm == "fold":
                continue
            b0, b1 = (a0 * st[k], a1) if n.player == 0 else (a0, a1 * st[k])
            if nm == "call" or (nm == "check" and path):
                out.append((path + (nm,), b0, b1, float(b0.sum() * b1.sum()) / tot))
            else:
                walk(n.children[k], path + (nm,), b0, b1)
    walk(sol.root, (), sol.r0.astype(np.float64), sol.r1.astype(np.float64))
    return sorted(out, key=lambda x: -x[3])


def line_pot(path, pot):
    """The turn pot after a flop line of checks, bets (a share of the pot) and a call."""
    bet = 0.0
    for nm in path:
        if nm.startswith("bet"):
            bet = float(nm.split()[-1].rstrip("%")) / 100.0 * pot
    return pot + 2 * bet


def full(active, v):
    f = np.zeros(len(ALL_COMBOS))
    f[active] = v
    return f


def given_up(names, S, V, w, pred_names, P):
    """(EV given up, chips per unit reach; the unseen share of w) of a predicted mix P (A', 1326 NaN where none, its
    actions pred_names) at a node with names, the reference mix S and EVs V (A, n) and reach w (n) over the same
    combos (P already restricted to them)."""
    Q = np.zeros((len(names), P.shape[1]))
    for a, nm in enumerate(pred_names):
        Q[match_action(list(names), nm)] += np.nan_to_num(P[a])
    bad = np.isnan(P[0])
    ww = np.where(bad, 0.0, w)
    Q = Q / np.maximum(Q.sum(axis=0, keepdims=True), 1e-12)
    true_ev, pred_ev = (S * V).sum(axis=0), (Q * V).sum(axis=0)
    W = max(float(ww.sum()), 1e-12)
    return float((ww * (true_ev - pred_ev)).sum() / W), float(w[bad].sum() / max(float(w.sum()), 1e-12))


def price_turn(ref, at_nodes, preds, pot):
    """{method: {at: (given up % of the turn pot, unseen)}} for the reference turn solve `ref` at its first decision
    and in position after a check; preds {method: fn(at, player) -> (names, (A, 1326)) or None}."""
    out = collections.defaultdict(dict)
    act = ref.ctx.active
    r0, r1 = ref.r0.astype(np.float64), ref.r1.astype(np.float64)
    root = ref.root
    for at in at_nodes:
        if at == ():
            node, a0, a1 = root, r0, r1
        else:
            if "check" not in root.names:
                continue
            k = root.names.index("check")
            s0 = root.avg_strategy()[0][k].astype(np.float64)
            node, a0, a1 = root.children[k], r0 * s0, r1
        if not isinstance(node, Decision):
            continue
        S = node.avg_strategy()[0].astype(np.float64)
        V = ref.solver.node_action_evs(node, a0, a1)
        w = a0 if node.player == 0 else a1
        for m, fn in preds.items():
            got = fn(at, node.player)
            if got is None:
                continue
            names, P = got
            loss, unseen = given_up(node.names, S, V, w, names, P[:, act])
            out[m][" ".join(at) or "first"] = (round(loss / pot * 100, 4), round(unseen, 4))
    return out


def live_pred(sol):
    """A live turn solve's mix as a prediction: fn(at, player) -> (names, (A, 1326))."""
    def fn(at, p):
        node = sol.root
        if at:
            if "check" not in node.names:
                return None
            node = node.children[node.names.index("check")]
        if not isinstance(node, Decision) or node.player != p:
            return None
        P = np.full((len(node.names), len(ALL_COMBOS)), np.nan)
        P[:, sol.ctx.active] = node.avg_strategy()[0]
        return node.names, P
    return fn


def store_pred(t, path, card):
    """A turn file's stored mix on the flop line `path` (the canonical flop: no suit map) as a prediction."""
    ident = list(range(4))

    def fn(at, p):
        st = fl.turn_strategies(t, path, at, card, ident)
        if st is None or st["player"] != p:
            return None
        return st["names"], st["strat"]
    return fn


def run(a):
    lib = fl.FlopLibrary.from_dir()
    acc, fast = gn.config("accurate"), gn.config("fast")
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            r = json.loads(x)
            done.add((r["case"], r["flop"], r["line"], r["turn"]))
    for case in a.cases.split(","):
        s, sp, borrow, src = case_setup(lib, case)
        if sp is None:
            print(case, "no library spot", file=sys.stderr)
            continue
        has = src.get("turns")
        names = sorted(n for n in sp["flops"] if n in src["flops"] and (has is True or (has and n in has)))
        random.Random(a.seed).shuffle(names)
        cfg = fl.library_config(s["pot"], s["stack"])
        for name in names[:a.n]:
            board = parse_cards(name)
            bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
            sol = lib.solution(s["full0"], s["full1"], board, s["full0"] * bm, s["full1"] * bm, s["pot"], s["stack"],
                               cfg, borrow=borrow)
            t_src = lib.turns(src, name)
            t_own = lib.turns(sp, name)
            if sol is None or t_src is None:
                continue
            rng = random.Random(f"{a.seed}{case}{name}")
            src_paths = {tuple(ln["path"]) for ln in t_src["lines"]}
            lines = [x for x in flop_lines(sol) if x[0] in src_paths][:a.lines]
            for path, r0, r1, share in lines:
                deck = [c for c in range(52) if c not in board]
                for card in rng.sample(deck, a.cards):
                    key = (case, name, " ".join(path), cards_str([card]))
                    if key in done:
                        continue
                    pot_t = line_pot(path, s["pot"])
                    stack_t = s["stack"] - (pot_t - s["pot"]) / 2
                    b4 = list(board) + [card]
                    f0, f1 = full(sol.ctx.active, r0), full(sol.ctx.active, r1)
                    t0 = time.time()
                    ref = solve(b4, f0, f1, pot_t, stack_t, cfg=acc, time_budget=1e9, max_iters=a.iters)
                    t_ref = time.time() - t0
                    preds = {"near": store_pred(t_src, path, card)}
                    if t_own is not None:
                        preds["own"] = store_pred(t_own, path, card)
                    for it in (50, 100):
                        preds[f"fast{it}"] = live_pred(solve(b4, f0, f1, pot_t, stack_t, cfg=fast, time_budget=1e9,
                                                             max_iters=it))
                    got = price_turn(ref, [(), ("check",)], preds, pot_t)
                    row = {"case": case, "spot": sp["id"], "src": src["id"], "flop": name, "line": " ".join(path),
                           "turn": cards_str([card]), "share": round(share, 4), "pot": pot_t, "iters": ref.iters,
                           "expl": round(ref.exploitability(), 4), "secs": round(t_ref, 1), "res": got}
                    with open(a.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)


def report(a):
    rows = [json.loads(x) for x in open(a.out)]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for m, d in r["res"].items():
            for at, (loss, unseen) in d.items():
                by[r["case"]][(m, at)].append((loss, unseen))
    for case, d in by.items():
        n = len({(r["flop"], r["line"], r["turn"]) for r in rows if r["case"] == case})
        src = next(r["src"] for r in rows if r["case"] == case)
        cells = []
        for m in ("near", "own", "fast50", "fast100"):
            for at in ("first", "check"):
                v = d.get((m, at))
                if v:
                    L = [x[0] for x in v]
                    cells.append(f"{m}/{at} {np.mean(L):.3f} (med {np.median(L):.3f}, max {max(L):.2f}"
                                 + (f", unseen {np.mean([x[1] for x in v]):.2f}" if m in ("near", "own") else "") + ")")
        print(f"{case} <- {src} ({n} turns): " + "; ".join(cells))
    print("\n| case <- source (turns) | near: first | near: in position after a check | unseen | Fast at 50 "
          "iterations | at 100 | near at most Fast's 50th |")
    print("|---|---|---|---|---|---|---|")
    for case, d in by.items():
        n = len({(r["flop"], r["line"], r["turn"]) for r in rows if r["case"] == case})
        src = next(r["src"] for r in rows if r["case"] == case)
        m = lambda k, at: np.mean([x[0] for x in d[(k, at)]]) if d.get((k, at)) else float("nan")
        mx = lambda k, at: max(x[0] for x in d[(k, at)]) if d.get((k, at)) else float("nan")
        wins = [sum(a[0] <= b[0] for a, b in zip(d[("near", at)], d[("fast50", at)])) for at in ("first", "check")]
        print(f"| {case} <- {src} ({n}) | {m('near', 'first'):.3f} ({mx('near', 'first'):.2f}) | "
              f"{m('near', 'check'):.3f} ({mx('near', 'check'):.2f}) | "
              + " / ".join(f"{np.mean([x[1] for x in d[('near', at)]]):.2f}" for at in ("first", "check"))
              + f" | {m('fast50', 'first'):.3f} / {m('fast50', 'check'):.3f} | {m('fast100', 'first'):.3f} / "
              f"{m('fast100', 'check'):.3f} | {wins[0]} / {wins[1]} of {len(d[('near', 'first')])} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--n", type=int, default=6, help="flops a case")
    ap.add_argument("--lines", type=int, default=2, help="flop lines a flop (most reached)")
    ap.add_argument("--cards", type=int, default=1, help="turn cards a line")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    report(a) if a.report else run(a)


if __name__ == "__main__":
    main()
