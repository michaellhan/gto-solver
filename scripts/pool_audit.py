"""Exploit-mode audit: sampled heads-up postflop spots against the player pool, each scored by the pool's rules with
exploit.py's switches off and on, on the same solve (so solve noise doesn't read as a difference), and flagged
where the pool's best response looks like an artefact of the model rather than a leak a real pool has.

Per spot it records the hero's hand's best-response action and EVs, and over hero's whole range at the decision the
share taking aggressive actions (bets, raises, all-in): at equilibrium, against the pool with every switch off, with
each switch on alone, and with all on (the advice). Flags:
  changed      a switch changes the hand's best action, or the range's aggressive share by 5+ points
  overexploit  against the pool (all switches on) the range is aggressive 15+ points and 2x more than at equilibrium
               while facing a bet (how the AcQd raise to 48 over a donk overbet showed: 11% of BTN's range raising at
               equilibrium, 49% against the pool)

    python scripts/pool_audit.py --n 60 --seed 3 --out reports/pool_audit.md
"""
from __future__ import annotations
import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

AGG = ("bet", "raise", "all-in")
SWITCHES = ("tree_commit", "river_cut_folds")


def variants():
    off = {k: False for k in SWITCHES}
    out = {"off": off}
    for k in SWITCHES:
        out[k] = dict(off, **{k: True})
    out["on"] = {k: True for k in SWITCHES}
    return out


def install(record):
    """Wraps Solver.node_action_br_evs: with a pool lock it also scores each variant into `record` (one entry per
    call: the hero decision's names, per variant the EVs (A, N), the reach and equilibrium strategy)."""
    from gto import solver as S
    from gto.exploit import PoolLock
    orig = S.Solver.node_action_br_evs

    def wrapped(self, node, r0, r1, lock=None):
        if not isinstance(lock, PoolLock):
            return orig(self, node, r0, r1, lock)
        out = {}
        for name, sw in variants().items():
            lk = copy.copy(lock)
            for k, v in sw.items():
                setattr(lk, k, v)
            out[name] = np.asarray(orig(self, node, r0, r1, lk), dtype=np.float64)
        reach = np.asarray(r0 if node.player == 0 else r1, dtype=np.float64).ravel()
        record.append({"names": list(node.names), "evs": out, "reach": reach, "eq": np.asarray(node.avg_strategy0()),
                       "active": list(self.ctx.active_list), "pot": float(node.pot_now)})
        return out["on"]
    S.Solver.node_action_br_evs = wrapped


def br_share(evs, reach, names):
    """Share of the reaching range whose best action is aggressive (ties split evenly)."""
    best = evs >= evs.max(axis=0, keepdims=True) - 1e-9
    w = best / best.sum(axis=0, keepdims=True)
    agg = np.array([n.startswith(AGG) for n in names])
    return float((w[agg] * reach).sum() / max(reach.sum(), 1e-12))


def eq_share(eq, reach, names):
    agg = np.array([n.startswith(AGG) for n in names])
    return float((eq[agg] * reach).sum() / max(reach.sum(), 1e-12))


def spots(n, seed):
    """n pool-mode heads-up spots from the sanity sweep's generator: 6-max 100bb most, 3-handed and 40bb too, the
    kinds weighted to bets faced (where raises are)."""
    import sanity_sweep as sw
    rng = random.Random(seed)
    setups = {s[0]: s for s in sw.SETUPS}
    picks = [("6p 100bb", 0.6), ("3p 100bb", 0.25), ("6p 40bb", 0.15)]
    kinds = ["vs_bet", "river_vs_bet", "vs_check", "turn", "river", "first", "vs_bet", "river_vs_bet"]
    jobs = []
    for tag, share in picks:
        if tag not in setups:
            continue
        setup = setups[tag]
        _, closed = sw.preflop_lines(setup, max_in=2, min_reach=1e-4)
        jobs += sw.postflop_jobs(setup, closed, max(1, round(n * share)), rng.randrange(1 << 30), kinds=kinds)
    return jobs[:n]


def run(job, budget):
    from gto.native import advisor
    setup, state, kind = job[:3]
    st = dict(state, opponents="pool")
    record = []
    install(record)
    adv = advisor("fast", 2, budget, budget / 3)
    t0 = time.time()
    res = adv.advise(st)
    if "error" in res or not record:
        return {"setup": setup[0], "kind": kind, "state": st, "error": res.get("error", "no pool best response"),
                "seconds": time.time() - t0}
    from gto.cards import COMBO_INDEX, cards
    rec = record[-1]
    names = rec["names"]
    ci = COMBO_INDEX[tuple(sorted(cards(state["hole"])))]
    col = rec["active"].index(ci) if ci in rec["active"] else None
    row = {"setup": setup[0], "kind": kind, "state": st, "names": names, "pot": rec["pot"],
           "facing": res.get("to_call_bb", 0) > 0, "seconds": time.time() - t0,
           "eq_agg": eq_share(rec["eq"], rec["reach"], names)}
    for v, evs in rec["evs"].items():
        row[v] = {"agg": br_share(evs, rec["reach"], names),
                  "hand": None if col is None else [float(x) for x in evs[:, col]]}
    flags = []
    for v in SWITCHES + ("on",):
        a, b = row["off"], row[v]
        hand_change = a["hand"] is not None and int(np.argmax(a["hand"])) != int(np.argmax(b["hand"])) and \
            max(b["hand"]) - b["hand"][int(np.argmax(a["hand"]))] > 0.005 * rec["pot"]
        if hand_change or abs(b["agg"] - a["agg"]) >= 0.05:
            flags.append(f"changed:{v}")
    on = row["on"]["agg"]
    if row["facing"] and on - row["eq_agg"] >= 0.15 and on >= 2 * row["eq_agg"]:
        flags.append("overexploit")
    row["flags"] = flags
    return row


def describe(r):
    st = r["state"]
    line = "; ".join(f"{s}: " + ",".join(f"{a['pos']} {a['action']}{' ' + str(a.get('amount') or a.get('to') or '') if a['action'] in ('bet', 'raise') else ''}"
                                          for a in st.get(s, [])) for s in ("flop", "turn", "river") if st.get(s))
    return f"{r['setup']} {r['kind']}: {st['hero']} {st['hole']} on {st['board']} ({line or 'first'})"


def report(rows, out, seconds):
    lines = ["# Exploit-mode audit", "", f"{len(rows)} spots in {seconds:.0f}s. Aggressive share = hero's range whose "
             "best response bets or raises; 'off' = every exploit.py switch off, 'on' = the advice.", ""]
    errs = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    for flag in ["overexploit"] + [f"changed:{v}" for v in SWITCHES + ("on",)]:
        hit = [r for r in ok if flag in r["flags"]]
        lines.append(f"## {flag}: {len(hit)}")
        for r in hit:
            def hand(v):
                h = r[v]["hand"]
                return "-" if h is None else f"{r['names'][int(np.argmax(h))]} " + "/".join(f"{x:+.2f}" for x in h)
            lines.append(f"- {describe(r)}: actions {', '.join(r['names'])}; aggressive eq {r['eq_agg']:.0%}, off "
                         f"{r['off']['agg']:.0%}, " + ", ".join(f"{v} {r[v]['agg']:.0%}" for v in SWITCHES + ("on",)) +
                         f"; hand off {hand('off')}, on {hand('on')}")
        lines.append("")
    if errs:
        lines.append(f"## errors: {len(errs)}")
        lines += [f"- {describe(r)}: {r['error']}" for r in errs]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def _one(args):
    job, budget = args
    try:
        return run(job, budget)
    except Exception as e:   # noqa: BLE001
        return {"setup": job[0][0], "kind": job[2], "state": job[1], "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--budget", type=float, default=2.5)
    ap.add_argument("--out", default="pool_audit.md")
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    jobs = spots(args.n, args.seed)
    # one spot per child process: the solver's caches grow over many spots past what the machine's memory guard allows
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool, open(args.out + ".jsonl", "w") as log:
        for i, r in enumerate(pool.imap(_one, [(j, args.budget) for j in jobs])):
            rows.append(r)
            log.write(json.dumps(r, default=float) + "\n")
            log.flush()
            print(f"{i + 1}/{len(jobs)} {describe(r)} {r.get('flags', r.get('error'))}", flush=True)
    report(rows, args.out, time.time() - t0)
    print("report in", args.out)


if __name__ == "__main__":
    main()
