"""Derive the preflop equilibrium: equity realization measured with the postflop solver, the full preflop
game solved with it (gto/pf_game.py), and the two iterated to a fixed point.

Each round:
  1. measure: for each standard heads-up flop spot (SPOTS: single-raised, 3bet and 4bet pots, the aggressor
     in and out of position, several seat pairs each) solve a stratified sample of flops (build_realization's
     flop_sample) on the ranges that reach it (round 0: the charts, gto/preflop.py; later rounds: the previous
     round's preflop solve), every class of the range each player had before its last action added at a tiny
     weight (EPS), so that hands outside the range get their counterfactual value too (a hand the opponent
     never expects plays its own best strategy against the opponent's equilibrium one: what it would earn if
     added). Per class and opponent strength bucket: the pot share (EV over the pot, later bets included) and
     the all-in equity.
  2. fit: per (pot type, aggressor in position, role) a smooth model of the excess share (share - equity) on
     hand features (fit_eqr), with each class's own mean residual shrunk in -> gto/pf_eqr.py.
  3. solve: the preflop game with the model (pf_game.Game), DAMP-averaged with the previous round's model.
Solves are cached per round, spot and flop (--cache), so an interrupted run resumes where it stopped.

  python scripts/pf_equilibrium.py --rounds 3                      # the whole loop
  python scripts/pf_equilibrium.py --round 1 --stage measure       # one stage of one round
  python scripts/pf_equilibrium.py --report                        # the solve against tests/preflop_refs.py
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import gto.native  # noqa: F401  (one BLAS thread per process)
import numpy as np

EPS = 1e-3
DEFAULT_CACHE = os.path.join(ROOT, ".pf_cache")

# (name, line to the flop in pf_game actions, (pot type, aggressor in position))
SPOTS = [
    ("BTN-BB", [("BTN", "raise"), ("BB", "call")], (1, True)),
    ("CO-BB", [("CO", "raise"), ("BB", "call")], (1, True)),
    ("UTG-BB", [("UTG", "raise"), ("BB", "call")], (1, True)),
    ("BTN-SB", [("BTN", "raise"), ("SB", "call"), ("BB", "fold")], (1, True)),
    ("SB-BB", [("SB", "raise"), ("BB", "call")], (1, False)),
    ("CO-BTN", [("CO", "raise"), ("BTN", "call"), ("SB", "fold"), ("BB", "fold")], (1, False)),
    ("HJ-CO", [("HJ", "raise"), ("CO", "call"), ("BTN", "fold"), ("SB", "fold"), ("BB", "fold")], (1, False)),
    ("CO-BTN3", [("CO", "raise"), ("BTN", "raise"), ("SB", "fold"), ("BB", "fold"), ("CO", "call")], (2, True)),
    ("UTG-CO3", [("UTG", "raise"), ("CO", "raise"), ("BTN", "fold"), ("SB", "fold"), ("BB", "fold"),
                 ("UTG", "call")], (2, True)),
    ("SB-BB3", [("SB", "raise"), ("BB", "raise"), ("SB", "call")], (2, True)),
    ("BTN-BB3", [("BTN", "raise"), ("BB", "raise"), ("BTN", "call")], (2, False)),
    ("BTN-SB3", [("BTN", "raise"), ("SB", "raise"), ("BB", "fold"), ("BTN", "call")], (2, False)),
    ("CO-BB3", [("CO", "raise"), ("BB", "raise"), ("CO", "call")], (2, False)),
    ("CO-BTN4", [("CO", "raise"), ("BTN", "raise"), ("SB", "fold"), ("BB", "fold"), ("CO", "raise"),
                 ("BTN", "call")], (3, False)),
    ("SB-BB4", [("SB", "raise"), ("BB", "raise"), ("SB", "raise"), ("BB", "call")], (3, False)),
    ("BTN-BB4", [("BTN", "raise"), ("BB", "raise"), ("BTN", "raise"), ("BB", "call")], (3, True)),
    ("BTN-SB4", [("BTN", "raise"), ("SB", "raise"), ("BB", "fold"), ("BTN", "raise"), ("SB", "call")], (3, True)),
]


def game(model=None, rake=None):
    from gto import pf_game as G
    return G.Game(G.build(), G.SIX, model or G.flat_model(), rake=rake)


def terminal(g, line):
    """The terminal after `line` (everyone not named folding), and the decision nodes on the way."""
    node, path = g.root, []
    for posn, act in line:
        while node.kind == "dec" and g.positions[node.player] != posn:
            path.append((node, "fold"))
            node = node.children[node.names.index("fold")]
        path.append((node, act))
        node = node.children[node.names.index(act)]
    assert node.kind == "show", f"{line} doesn't end in a flop"
    return node, path


def spot_ranges(g, reach, line):
    """(oop seat, ip seat, aggressor seat, combo ranges at the flop, each player's range before its last
    action, pot, stack behind) on `line`, from reaches (N, n, K) of a preflop solve."""
    from gto.pf_classes import CLASS_OF
    from gto.pf_game import PRIOR
    t, path = terminal(g, line)
    a = t.aggressor
    c = [q for q in t.active if q != a][0]
    before = {}
    for node, act in path:                 # the last decision of each player on the way
        if node.player in (a, c):
            before[node.player] = node
    combo = lambda w: (w / PRIOR)[CLASS_OF]
    rng = {q: combo(reach[t.index, q]) for q in (a, c)}
    pre = {q: combo(reach[before[q].index, q]) for q in (a, c)}
    oop, ip = sorted((a, c), key=lambda q: g.post_rank[q])
    pot = float(t.inv.sum())
    return oop, ip, a, rng, pre, pot, 100.0 - float(t.inv.max())


def chart_ranges(g, line):
    """The same from the charts (gto/preflop.py) replayed along the line."""
    from gto import preflop as pf
    t, path = terminal(g, line)
    a = t.aggressor
    c = [q for q in t.active if q != a][0]
    acts, before = [], {}
    T = pf.Table(pf.SEATS[6])
    for (node, act), (posn, act2, to) in zip(path, t.line):
        if node.player in (a, c):
            before[node.player] = pf.replay_ranges(acts, T).get(g.positions[node.player])
        acts.append({"pos": posn, "action": act2 if act2 != "all-in" else "raise", **({"to": to} if to else {})})
    R = pf.replay_ranges(acts, T)
    rng = {q: R[g.positions[q]] for q in (a, c)}
    oop, ip = sorted((a, c), key=lambda q: g.post_rank[q])
    return oop, ip, a, rng, before, float(t.inv.sum()), 100.0 - float(t.inv.max())


def post_config():
    """The postflop tree for the measurements: the native fast speed's (gto/native.py: flop bets 33% and 75%
    pot, one turn and river size, the turn and river played by 32 hand buckets per player), about 15x
    cheaper an iteration than the per-combo turn and river, within a few points of them on flop strategy."""
    from gto.native import SPEEDS
    from gto.solver import Config
    sizes = SPEEDS["fast"][0]
    return Config(bet_sizes={k: list(v) for k, v in sizes["bet_sizes"].items()},
                  raise_sizes={k: list(v) for k, v in sizes["raise_sizes"].items()}, **sizes["flop_model"])


def measure_job(job):
    """One flop solve -> per player (oop, ip): (3, K, buckets) sums over its combos of the opponent's
    compatible mass in each bucket times [pot share, preflop equity, 1] (each combo counted alike, whatever
    its own weight: in range or at EPS)."""
    path, board, r_oop, r_ip, pot, stack, budget = job
    if os.path.exists(path):
        return path
    from gto.solver import solve, Config
    from gto.pf_classes import CLASS_OF, equity_vs, to_classes
    from gto.pf_solve import bucket_of, K
    # a fixed number of iterations (not a time budget: the machine's load would set the accuracy)
    sol = solve(board, r_oop, r_ip, pot, stack, cfg=post_config(), time_budget=1e9, max_iters=int(budget))
    s, r0, r1, act = sol.solver, sol.r0, sol.r1, sol.ctx.active
    bfull = bucket_of()[CLASS_OF]
    nb = int(bfull.max()) + 1
    out = []
    s.sampling = False
    for p, r_opp_full in ((0, r_ip), (1, r_oop)):
        acc = np.zeros((3, K, nb))
        r_opp = r1 if p == 0 else r0
        for b in range(nb):
            mb = (bfull[act] == b).astype(float)
            ro = r_opp * mb
            if ro.sum() < 1e-12:
                continue
            rr = (r0, ro) if p == 0 else (ro, r1)
            u = s._cfr(sol.root, p, rr[0][None, :], rr[1][None, :], update=False, use_avg=True)[0]
            cs = s.compat_sum(ro[None, :])[0]
            share = np.where(cs > 1e-15, u / np.maximum(cs, 1e-15) + pot / 2, 0.0) / pot
            w = np.zeros(len(r_oop)); sh = np.zeros(len(r_oop))
            w[act] = cs / max(r_opp.sum(), 1e-15)   # the bucket's compatible mass, a share of the whole range
            sh[act] = share
            eq = np.nan_to_num(equity_vs(r_opp_full * (bfull == b)), nan=0.0)
            acc[0, :, b] = to_classes(w * sh)
            acc[1, :, b] = to_classes(w * eq)
            acc[2, :, b] = to_classes(w)
        out.append(acc)
    np.savez(path + ".tmp.npz", oop=out[0], ip=out[1], iters=sol.iters)
    os.replace(path + ".tmp.npz", path)
    return path


def measure(rnd, reach, args, g):
    """Run (or resume) the flop solves of round `rnd`; returns the table: per (pot type, aggressor in
    position, role): summed share, equity and weight arrays (K, buckets) and the class weights of the
    ranges before the last action (for fitting)."""
    import build_realization as B
    from gto.pf_classes import to_classes
    flops = B.flop_sample(args.flops)
    d = os.path.join(args.cache, f"round{rnd}_f{args.flops}_i{args.budget}")
    os.makedirs(d, exist_ok=True)
    jobs, meta = [], []
    for name, line, key in SPOTS:
        oop, ip, a, rng, pre, pot, stack = chart_ranges(g, line) if reach is None else spot_ranges(g, reach, line)
        r_oop = rng[oop] + EPS * pre[oop]
        r_ip = rng[ip] + EPS * pre[ip]
        for k, (f, fw, half, fwh) in enumerate(flops):
            path = os.path.join(d, f"{name}_{k}.npz")
            jobs.append((path, f, r_oop, r_ip, pot, stack, args.budget))
            meta.append((name, key, oop == a, fw, to_classes(pre[oop]), to_classes(pre[ip]),
                         to_classes(rng[oop]), to_classes(rng[ip])))
    todo = [j for j in jobs if not os.path.exists(j[0])]
    print(f"round {rnd}: {len(jobs)} flop solves, {len(todo)} to run", flush=True)
    t0 = time.time()
    if todo:
        with Pool(args.procs) as pool:
            for i, _ in enumerate(pool.imap_unordered(measure_job, todo)):
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    table = {}
    for (path, *_), (name, key, oop_aggr, fw, pre_oop, pre_ip, rng_oop, rng_ip) in zip(jobs, meta):
        z = np.load(path)
        for side, pre, rg in (("oop", pre_oop, rng_oop), ("ip", pre_ip, rng_ip)):
            role = "aggr" if (side == "oop") == oop_aggr else "caller"
            k = key + (role,)
            ent = table.setdefault(k, {"acc": 0.0, "pre": 0.0, "rng": 0.0, "spots": {}})
            ent["rng"] = ent["rng"] + fw * rg / rg.sum()
            # each spot counts by the flop weights alone (spots pooled equally)
            ent["acc"] = ent["acc"] + fw * z[side]
            ent["spots"][name] = ent["spots"].get(name, 0.0) + fw * z[side]
            ent["pre"] = ent["pre"] + fw * pre / pre.sum()
    return table


# ----------------------------------------------------------------------------- fit
from gto.pf_game import EQR_FEATURES as FEATURES, eqr_features as features  # noqa: E402


SHRINK_W0 = 1.0   # a class's own mean residual counts by w / (w + SHRINK_W0) (w: its weight over a typical class's)


def fit_eqr(table):
    """Per key: model coefficients, the class residuals (shrunk) and fit stats."""
    from gto.pf_solve import _bucket_equity
    E = _bucket_equity()
    X = features(E)
    out = {}
    for k, ent in sorted(table.items()):
        sh, eq, w = ent["acc"]
        x = np.where(w > 1e-12, (sh - eq) / np.maximum(w, 1e-12), 0.0)     # measured excess per class, bucket
        # weight: the class's share of the range before the action (every class of it measured alike) times
        # the bucket's compatible mass
        wt = ent["pre"][:, None] * w / np.maximum(w.sum(1, keepdims=True), 1e-12)
        wt = wt / wt.sum()
        msk = wt > 1e-9
        sw = np.sqrt(wt[msk])
        beta = np.linalg.lstsq(X[msk] * sw[:, None], x[msk] * sw, rcond=None)[0]
        pred = X @ beta
        res = x - pred
        mean = (wt * x).sum()
        r2 = 1 - (wt * res ** 2).sum() / max((wt * (x - mean) ** 2).sum(), 1e-12)
        wc = wt.sum(1)
        typ = np.median(wc[wc > 0]) if (wc > 0).any() else 1.0
        rc = np.where(wc > 0, (wt * res).sum(1) / np.maximum(wc, 1e-12), 0.0)
        lam = (wc / typ) / (wc / typ + SHRINK_W0)
        out[k] = {"w": ent["rng"], "coef": beta, "resid": lam * rc, "r2": float(r2),
                  "realization": float(((wt * sh / np.maximum(w, 1e-12)).sum()) / max((wt * eq / np.maximum(w, 1e-12)).sum(), 1e-12))}
    return out


def excess_tables(fit):
    """Per key: (K, buckets) excess share of the fitted model (pf_game.excess_from_fit)."""
    from gto.pf_game import excess_from_fit
    return {k: excess_from_fit(v["coef"], v["resid"]) for k, v in fit.items()}


def blend_fits(new, old, damp):
    """The damped fit: coefficients, residuals and range weights averaged (the excess is linear in the first
    two, so this averages the tables too)."""
    if old is None:
        return new
    return {k: {f: (damp * np.asarray(v[f]) + (1 - damp) * np.asarray(old[k][f]) if f in ("coef", "resid", "w") else v[f])
                for f in v} for k, v in new.items()}


def write_eqr(fit, path, note):
    """gto/pf_eqr.py: the fitted model's coefficients, class residuals and range weights per key."""
    keys = sorted(fit)
    lines = ['"""Generated by scripts/pf_equilibrium.py: equity realization for the preflop game (gto/pf_game.py), measured',
             'with the postflop solver. Per (pot type 1/2/3 = single-raised/3bet/4bet, aggressor in position, role "aggr"',
             'or "caller"): a hand\'s excess pot share over its all-in equity against each opponent strength bucket',
             '(pf_solve.bucket_of) is COEF on pf_game.EQR_FEATURES of its equity against the bucket plus the class',
             'residual RESID (pf_classes.CLASSES order); W holds the class weights of the range it was measured on',
             '(pf_game.measured_model combines a key\'s two roles into pair shares).',
             note,
             'Pre-action range realization (pot share over equity) by key: ' +
             ", ".join(f"{k[0]}{'ip' if k[1] else 'oop'}-{k[2]} {fit[k]['realization']:.2f}" for k in keys) + '."""',
             "COEF = {"]
    for k in keys:
        lines.append(f"    {k!r}: {[round(float(x), 4) for x in fit[k]['coef']]!r},")
    lines.append("}")
    lines.append("RESID = {")
    for k in keys:
        lines.append(f"    {k!r}: {[round(float(x), 4) for x in fit[k]['resid']]!r},")
    lines.append("}")
    lines.append("W = {")
    for k in keys:
        w = np.asarray(fit[k]["w"], float)
        w = w / w.sum()
        lines.append(f"    {k!r}: {[round(float(x), 5) for x in w]!r},")
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- report
def report(g, verbose=False):
    """The solve against tests/preflop_refs.py: per reference, the solve's fold/call/raise shares on the
    reference's line (unraked solves are compared as they are)."""
    from tests.preflop_refs import REFS, HANDS
    from gto.pf_game import PRIOR
    from gto.pf_classes import CLASS_INDEX
    from gto.ranges import parse_range
    from gto.pf_classes import CLASS_OF
    reach = g.reach()
    errs, n_miss, rows = [], 0, []
    for r in REFS:
        line = [tuple(a.split()[:2]) for a in r["line"].split(", ")] if r["line"] else []
        node = g.root
        try:
            for posn, act in line + [(r["hero"], None)]:
                while node.kind == "dec" and g.positions[node.player] != posn:
                    node = node.children[node.names.index("fold")]
                if act is not None:
                    node = node.children[node.names.index("call" if act == "call" else "raise")]
        except ValueError:
            continue
        s = g.strategy_at(node)
        w = PRIOR if r["of"] == "hands" else reach[node.index, node.player]
        w = w / w.sum()
        got = {k if k != "all-in" else "raise": float((w * v).sum()) for k, v in s.items()}
        bad = {k: (got.get(k, 0), v) for k, v in r["want"].items() if abs(got.get(k, 0) - v) > r["tol"] + 1e-9}
        if not r["uncertain"]:
            errs += [abs(got.get(k, 0) - v) for k, v in r["want"].items()]
            n_miss += bool(bad)
        rows.append((r["name"], got, r["want"], bad, r["uncertain"]))
    nb = 0
    hands = []
    for h in HANDS:
        name, hero, line, hand, action, lo, hi = h[:7]
        line = [tuple(a.split()[:2]) for a in line.split(", ")] if line else []
        node = g.root
        for posn, act in line + [(hero, None)]:
            while node.kind == "dec" and g.positions[node.player] != posn:
                node = node.children[node.names.index("fold")]
            if act is not None:
                node = node.children[node.names.index("call" if act == "call" else "raise")]
        s = g.strategy_at(node)
        m = parse_range(hand) > 0
        key = action if action in s else ("all-in" if action == "raise" and "all-in" in s else action)
        v = float(np.mean(s.get(key, np.zeros(len(PRIOR)))[CLASS_OF][m]))
        ok = lo - 1e-9 <= v <= hi + 1e-9
        nb += not ok
        hands.append((name, action, v, lo, hi, ok))
    return rows, (float(np.mean(errs)) if errs else 0.0, n_miss), hands, nb


def print_report(g):
    rows, (mae, n_miss), hands, nb = report(g)
    for name, got, want, bad, unc in rows:
        cell = lambda k: f"{got.get(k, 0) * 100:5.1f}/{want[k] * 100:5.1f}" if k in want else " " * 11
        print(f"{name:34s} fold {cell('fold')}  call {cell('call')}  raise {cell('raise')}  "
              f"{'MISS' if bad else 'ok'}{'?' if unc else ''}")
    print(f"certain references outside tolerance {n_miss}, mean abs error {mae * 100:.2f} points")
    for name, action, v, lo, hi, ok in hands:
        print(f"  {name:36s} {action} {v * 100:5.1f}% (want {lo * 100:.0f}-{hi * 100:.0f}) {'ok' if ok else 'MISS'}")
    print(f"hand facts outside range {nb}")


# ----------------------------------------------------------------------------- driver
def load_fit(cache, rnd):
    """Round rnd's damped fit (dfit), or its own fit for older caches."""
    for name in (f"dfit{rnd}.pkl", f"fit{rnd}.pkl"):
        path = os.path.join(cache, name)
        if os.path.exists(path):
            return pickle.load(open(path, "rb"))
    return None


def load_model(cache, rnd):
    from gto import pf_game as G
    fit = load_fit(cache, rnd)
    return G.EQRModel.from_tables(excess_tables(fit), {k: np.asarray(v["w"]) for k, v in fit.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--round", type=int)
    ap.add_argument("--stage", choices=("measure", "solve"))
    ap.add_argument("--flops", type=int, default=68)
    ap.add_argument("--budget", type=int, default=120, help="postflop solver iterations a flop")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--damp", type=float, default=0.5, help="weight of the new round's fit against the previous")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--rake", default="", help="rate,cap for the solve (e.g. 0.05,4)")
    ap.add_argument("--tag", default="", help="suffix for the solve's file (e.g. a solve without rake)")
    ap.add_argument("--report", type=int, help="print round N's solve against the references")
    ap.add_argument("--write", type=int, help="write round N's damped fit to gto/pf_eqr.py")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    rake = tuple(float(x) for x in args.rake.split(",")) if args.rake else None
    g = game(rake=rake)
    if args.write is not None:
        fit = load_fit(args.cache, args.write)
        write_eqr(fit, os.path.join(ROOT, "gto", "pf_eqr.py"),
                  f"Round {args.write} of the fixed point ({args.flops} flops a spot, {args.budget} iterations a flop).")
        return
    if args.report is not None:
        g.set_model(load_model(args.cache, args.report))
        g.load(os.path.join(args.cache, f"solve{args.report}{args.tag}.npz"))
        print_report(g)
        return
    rounds = [args.round] if args.round is not None else list(range(args.rounds))
    for rnd in rounds:
        stages = [args.stage] if args.stage else ["measure", "solve"]
        if "measure" in stages:
            reach = None
            if rnd > 0:
                g.set_model(load_model(args.cache, rnd - 1))
                g.load(os.path.join(args.cache, f"solve{rnd - 1}.npz"))
                reach = g.reach()
            table = measure(rnd, reach, args, g)
            pickle.dump(table, open(os.path.join(args.cache, f"table{rnd}.pkl"), "wb"))
            fit = fit_eqr(table)
            pickle.dump(fit, open(os.path.join(args.cache, f"fit{rnd}.pkl"), "wb"))
            dfit = blend_fits(fit, load_fit(args.cache, rnd - 1) if rnd > 0 else None, args.damp)
            pickle.dump(dfit, open(os.path.join(args.cache, f"dfit{rnd}.pkl"), "wb"))
            for k in sorted(fit):
                print(f"  {k}: realization {fit[k]['realization']:.3f}, R^2 {fit[k]['r2']:.2f}", flush=True)
        if "solve" in stages:
            g.set_model(load_model(args.cache, rnd))
            g.regret[:] = 0
            g.ssum[:] = 0
            t0 = time.time()
            g.solve(args.iters, verbose=True, every=500)
            g.save(os.path.join(args.cache, f"solve{rnd}{args.tag}.npz"))
            print(f"round {rnd} solved in {time.time() - t0:.0f}s", flush=True)
            print_report(g)


if __name__ == "__main__":
    main()
