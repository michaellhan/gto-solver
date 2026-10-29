"""Which part of the flop library's model costs its strategies EV: the flop solved again with one part of the model
changed, the decisions a library entry stores (the root, IP after a check, OOP facing a c-bet after checking) locked
into postflop-solver's full game of the library's tree (every turn and river card), once per player as in
bench_postflop_solver.py's flop mode: the EV that player gives up, in % of the pot.

  python scripts/bench_flop_abstraction.py solve lib,rd64 3bp_bb_btn:AcJd3h 3bp_bb_btn:Qc5h5d   # ours (cached)
  python scripts/bench_flop_abstraction.py measure lib,rd64,stored 3bp_bb_btn:AcJd3h            # theirs, the losses

Variants (from library_config() without the pot, the single-raised pots' model):
  lib      one turn per rank class (turn_classes), the river bet blind on the turn's boards
  all      every turn card
  rs       the river by 32 river-strength buckets (still on the turn's boards)
  rdN      the river dealt, N strength buckets on each card (Config.river_dealt; the 3bet pots' entries are rd64)
  rcN      the river dealt per combo, N river cards under each turn class (depth 3; rc48 is every card)
  stored   the library's entry as shipped (needs the flop in the library)
Ours are solved to --ours (% of the pot, in our own game) or 1200 iterations, checked every 100, or with @N (rd64@200)
to N iterations exactly (the library's build stops at --min-iters once under its target); theirs to --target,
their equilibrium cached per flop. Records go to reports/postflop_solver/abstraction.jsonl, our solves to
reports/postflop_solver/abstraction/. A single-raised flop's full game takes 2-5 GB and minutes on 6 threads.
"""
import argparse, json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bench_postflop_solver as B
from gto import flop_library as fl
from gto.cards import cards as parse_cards
from gto.solver import Ctx, TreeBuilder, solve

DIR = os.path.join(B.OUT, "abstraction")


def variant_cfg(v):
    cfg = fl.library_config()
    if v == "rs":
        cfg.river_strength, cfg.river_strength_spr = 32, 0.0
    elif v.startswith("rd"):
        cfg.river_dealt = int(v[2:] or 64)
    elif v.startswith("rc"):
        cfg.depth, cfg.n_rivers_from_flop = 3, int(v[2:] or 48)
    elif v == "all":
        cfg.turn_classes, cfg.n_turns = False, 49
    elif v != "lib":
        raise SystemExit(f"unknown variant {v}")
    return cfg


def path(sid, flop_s, v):
    return os.path.join(DIR, f"{sid}_{flop_s}_{v}.pkl")


def do_solve(v, ours, job):
    sid, flop_s = job.split(":")
    s = fl.spot_setup(B.SP[sid])
    board = parse_cards(flop_s)
    t0 = time.time()
    if v == "stored":
        ls = fl.default_library().solution(s["full0"], s["full1"], board, s["full0"], s["full1"], s["pot"], s["stack"],
                                           fl.library_config())
        if ls is None:
            print("not in the library:", job, flush=True)
            return
        nodes = {" ".join(strs): (pl, strat) for strs, _, strat, pl in B.stored_nodes(ls, ls.root)}
        rec = dict(active=ls.ctx.active, r0=ls.r0, r1=ls.r1, nodes=nodes, iters=ls.info().get("iterations"),
                   expl=ls.exploitability(), secs=0.0)
    else:
        base, _, at = v.partition("@")
        sol = solve(board, s["full0"], s["full1"], s["pot"], s["stack"], cfg=variant_cfg(base), time_budget=1e9,
                    max_iters=int(at) if at else 200)
        ex = sol.exploitability()
        while not at and ex > ours and sol.iters < 1200:
            sol.extend(1e9, max_iters=sol.iters + 100)
            ex = sol.exploitability()
        nodes = {" ".join(strs): (n.player, n.avg_strategy()[0]) for strs, n in B.street_nodes(sol.root)}
        rec = dict(active=sol.ctx.active, r0=sol.r0, r1=sol.r1, nodes=nodes, iters=sol.iters, expl=ex,
                   secs=time.time() - t0)
    with open(path(sid, flop_s, v), "wb") as f:
        pickle.dump(rec, f)
    print(job, v, "iterations", rec["iters"], "expl %.4f" % rec["expl"], "secs %.0f" % rec["secs"], flush=True)


def do_measure(variants, target, job):
    sid, flop_s = job.split(":")
    s = fl.spot_setup(B.SP[sid])
    f0, f1, pot, stack = s["full0"], s["full1"], s["pot"], s["stack"]
    board = parse_cards(flop_s)
    cfg = fl.library_config()
    ls = fl.default_library().solution(f0, f1, board, f0, f1, pot, stack, cfg)
    if ls is None:
        print("not in the library (its stored nodes are the ones locked):", job, flush=True)
        return
    stored_paths = {" ".join(strs) for strs, _, _, _ in B.stored_nodes(ls, ls.root)}
    ctx = Ctx(ls.ctx.active, cfg)
    ctx.root_len = 3
    full = TreeBuilder(ctx, dead=list(board)).build(tuple(board), pot, stack)
    fnodes = {" ".join(strs): (strs, n) for strs, n in B.street_nodes(full)}
    base = B.spec(board, f0, f1, pot, stack, full)
    ef = os.path.join(DIR, f"eq_{sid}_{flop_s}.json")
    eq = json.load(open(ef)) if os.path.exists(ef) else None
    if eq is None or eq["expl_pct"] > target:   # (a tighter one cached is kept)
        eq = B.their_solve(base, target, 3000, compress=True)
        eq = {k: eq[k] for k in ("root", "acts", "strat", "player", "ev_cur", "iters", "expl_pct", "secs")}
        eq["target"] = target
        with open(ef, "w") as f:
            json.dump(eq, f)
    head = {"spot": sid, "flop": flop_s, "their_iters": eq["iters"], "their_expl": round(eq["expl_pct"], 4),
            "their_secs": round(eq["secs"], 1)}
    for v in variants:
        if not os.path.exists(path(sid, flop_s, v)):
            print("not solved (run solve first):", job, v, flush=True)
            continue
        with open(path(sid, flop_s, v), "rb") as f:
            r = pickle.load(f)
        active = r["active"]
        keep = [np.nonzero(r["r0"] > 0)[0], np.nonzero(r["r1"] > 0)[0]]
        nodes = [(fnodes[p][0], fnodes[p][1], strat, pl) for p, (pl, strat) in r["nodes"].items() if p in stored_paths]
        t = time.time()
        loss, info = B.ev_losses(base, eq, nodes, active, keep, pot, target, compress=True)
        strat = np.zeros((len(full.names), len(B.HANDS)))
        strat[:, active] = r["nodes"][""][1]
        rec = dict(head, variant=v, ours_iters=r["iters"], ours_expl=round(float(r["expl"]), 4),
                   ours_secs=round(r["secs"], 1), ev_loss=loss, locked=info, root=B.root_compare(full, strat, eq, f0),
                   wall=round(time.time() - t, 1), their_threads=B.THREADS)
        print(json.dumps(rec), flush=True)
        with open(os.path.join(B.OUT, "abstraction.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("solve", "measure"))
    ap.add_argument("variants", help="comma separated: lib, all, rs, rdN, rcN, stored")
    ap.add_argument("jobs", nargs="+", help="spot:flop pairs")
    ap.add_argument("--ours", type=float, default=0.05, help="solve: our solves' exploitability, %% of the pot")
    ap.add_argument("--target", type=float, default=0.05, help="measure: their solves' exploitability")
    ap.add_argument("--threads", type=int, default=B.THREADS, help="their solves' threads (rayon)")
    a = ap.parse_args()
    B.THREADS = a.threads
    os.makedirs(DIR, exist_ok=True)
    for job in a.jobs:
        for v in a.variants.split(","):
            if a.mode == "solve":
                do_solve(v, a.ours, job)
        if a.mode == "measure":
            if not os.path.exists(B.BIN):
                sys.exit(f"no harness at {B.BIN}: build scripts/pfbench (its Cargo.toml) or set PFBENCH")
            do_measure(a.variants.split(","), a.target, job)
