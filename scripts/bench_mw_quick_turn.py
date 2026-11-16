"""Quick's 3-way turn game: what settles in a quick answer's time, and whether it beats the betting rules.

Takes 3-way turn decisions a quick answer met in a coverage run (coverage_fuzz.py jsonl; the ones the rules answered
first), and for each spot solves a reference (24 buckets split by value, Accurate's turn sizes, 200 iterations) and
each candidate game for as many iterations as fit its budget in CPU time (its tables, then iterations at the measured
CPU per iteration, so the count doesn't depend on machine load), and plays the betting rules (GTO mode, as Quick plays
them) on hands drawn from the advised player's range. Per hand, by the reference's EVs: what the most frequent action
gives up and what the mix gives up (each action mapped to the reference tree's action of its kind at the nearest
size), % of the pot, weighted by the range; and each candidate's own exploitability (mean best-response gain).

  python scripts/bench_mw_quick_turn.py .work/mw-audit/live_11_audit.jsonl out.json --spots 14 --hands 30 \\
      --budget 1.0 'q8={"buckets":8}' 'q6s1={"buckets":6,"bet":{"flop":0.33,"turn":[0.5],"river":0.5}}'
  python scripts/bench_mw_quick_turn.py ... --sum

A candidate is MWConfig fields as JSON over advisor.MW_QUICK_TURN. Results are kept in out.json per spot, so a run
resumes; --budget is the solve's (exploit mode's GTO pass leaves the exploit answer about two thirds of Quick's 1.5 s).
"""
import sys, os, time, json, argparse, dataclasses
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import gto.native  # noqa: F401  (before numpy: BLAS on one thread)
import numpy as np
import gto.advisor as A
import gto.multiway_solver as M
from gto.cards import ALL_COMBOS, cards_str

REF = {"buckets": 24, "bucket_power": 1.0, "bucket_eq": 0.0, "bet": A.MW_TURN_BETS}


class _Got(Exception):
    pass


def kind(nm):
    return nm.split()[0]


def records(path, n):
    rules, solved = [], []
    for l in open(path):
        r = json.loads(l)
        t = r.get("tags") or {}
        if t.get("street") == "turn" and t.get("in_pot") == 3 and t.get("speed") == "quick" and not t.get("allin_in") \
                and "state" in r:
            (rules if r.get("by") == "rules" else solved).append(r)
    k = min(len(rules), max(n - n // 4, n - len(solved)))
    return rules[:k] + solved[:n - k]


def capture(adv, st):
    """The 3-way turn solve's arguments the advisor builds for this state (its own game's MWConfig)."""
    got, orig = {}, M.MultiwaySolver.__init__

    def init(self, *a, **kw):
        if len(a[0]) != 4:
            return orig(self, *a, **kw)
        got["a"], got["k"] = a, {k: v for k, v in kw.items() if k not in ("deadline", "t_init")}
        raise _Got()
    M.MultiwaySolver.__init__ = init
    try:
        adv.advise(st)
    except _Got:
        pass
    finally:
        M.MultiwaySolver.__init__ = orig
    return got


def solve_cpu(args, kw, cfg, budget=None, iters=None):
    """A solve run for `iters` iterations, or as many as fit `budget` CPU seconds after its tables."""
    c0 = time.process_time()
    s = M.MultiwaySolver(*args, **{**kw, "cfg": cfg})
    tables = time.process_time() - c0
    if iters is None:
        c1 = time.process_time()
        s.run(iters=3)
        per = (time.process_time() - c1) / 3
        iters = int(max(3, min(700, (budget - tables) / per))) if budget > tables else 0
        if iters > 3:
            s.run(iters=iters - 3)
    else:
        per = None
        s.run(iters=iters)
    return s, tables, per, iters


def run(a):
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    games = [(g.split("=", 1)[0], json.loads(g.split("=", 1)[1])) for g in a.games]
    from gto import native
    adv = native.advisors()["fast"]
    rules = native.advisors()["fast"].__class__(adv.cfg, time_budget=1e9)
    rules.mw_solve_streets = ()
    rng = np.random.default_rng(a.seed)
    for r in records(a.records, a.spots):
        sid = f"{r['hand']}:{r['k']}"
        st = {k: v for k, v in r["state"].items() if k != "opponents"}
        got = capture(adv, dict(st))
        if "a" not in got:
            print(sid, "no 3-way turn solve here", flush=True)
            continue
        args, kw = got["a"], got["k"]
        base = dataclasses.replace(kw["cfg"], **{k: v for k, v in A.MW_QUICK_TURN.items()})
        d = res.setdefault(sid, {"by": r.get("by")})
        if "ref" not in d:
            ref, *_ = solve_cpu(args, kw, dataclasses.replace(base, **REF), iters=a.ref_iters)
            d["ref"] = {"names": list(ref.root.names), "amounts": [float(x or 0) for x in ref.root.amounts],
                        "ev": ref.action_evs().tolist(), "pot": ref.root_pot(), "p": int(ref.root.player),
                        "w": ref.reach0[ref.root.player, 0].tolist(), "active": ref.hands.active.tolist(),
                        "expl": float(ref.br_gains().mean() / ref.root_pot() * 100)}
            json.dump(res, open(a.out, "w"))
        R = d["ref"]
        w = np.array(R["w"])
        live = np.where(w > 0)[0]
        if "hands" not in d:
            d["hands"] = [int(x) for x in rng.choice(live, size=min(a.hands, len(live)), replace=False, p=w[live] / w[live].sum())]
        for name, g in games:
            key = f"{name}@{a.budget:g}"
            if key in d:
                continue
            s, tables, per, iters = solve_cpu(args, kw, dataclasses.replace(base, **g), budget=a.budget)
            out = {"tables": tables, "per": per, "iters": iters}
            if iters:
                out.update(names=list(s.root.names), amounts=[float(x or 0) for x in s.root.amounts],
                           sig=s.root.avg_strategy()[0].tolist(),
                           expl=float(np.mean([x for i, x in enumerate(s.br_gains()) if i not in s.cfg.allin]) / s.root_pot() * 100))
            d[key] = out
            print(f"{sid} {key}: tables {tables:.2f} s, {per and per * 1000 or 0:.0f} ms an iteration, {iters} iterations, "
                  f"{out.get('expl', float('nan')):.1f}% of the pot from equilibrium", flush=True)
            json.dump(res, open(a.out, "w"))
        if "rules" not in d:
            rows = []
            for ci in d["hands"]:
                h = cards_str(ALL_COMBOS[R["active"][ci]])     # (the solve's hands: the combos the board leaves)
                rr = rules.advise(dict(st, hole=h))
                if "actions" not in rr:
                    print(sid, h, "rules:", {k: rr[k] for k in rr if k in ("error", "notes")}, flush=True)
                    rr = {"actions": []}
                rows.append([[x["action"], x["freq"], x.get("amount_bb")] for x in rr["actions"]])
            d["rules"] = rows
            json.dump(res, open(a.out, "w"))
        print(sid, "done", flush=True)


def summary(a):
    res = json.load(open(a.out))
    names = sorted({k for d in res.values() for k in d if "@" in k})
    print(f"{'game':14s} spots  iters (min-max)  settled<10%  own expl  top pick gives up  mix gives up  (% of the pot)")
    for nm in names + ["rules"]:
        pick, mix, its, ex, ok, n = [], [], [], [], 0, 0
        for sid, d in res.items():
            if "ref" not in d or (nm not in d) or "rules" not in d:
                continue
            R = d["ref"]
            ev, pot = np.array(R["ev"]), R["pot"]
            hands = d["hands"]
            n += 1
            if nm == "rules":
                for ci, row in zip(hands, d["rules"]):
                    rmap = mapper_names(R, [x[0] for x in row], [x[2] for x in row])
                    best = ev[:, ci].max()
                    loss = [best - ev[m, ci] if m is not None else 0.0 for m in rmap]
                    f = np.array([x[1] for x in row])
                    pick.append(loss[int(np.argmax(f))] / pot * 100)
                    mix.append(float(np.dot(f / f.sum(), loss)) / pot * 100)
                ok += 1
                continue
            g = d[nm]
            its.append(g["iters"])
            if not g["iters"]:
                continue
            ex.append(g["expl"])
            ok += g["expl"] < 10
            m = mapper_names(R, g["names"], g["amounts"])
            sig = np.array(g["sig"])
            for ci in hands:
                best = ev[:, ci].max()
                loss = np.array([best - ev[x, ci] if x is not None else 0.0 for x in m])
                f = sig[:, ci]
                pick.append(loss[int(np.argmax(f))] / pot * 100)
                mix.append(float(np.dot(f / max(f.sum(), 1e-12), loss)) / pot * 100)
        if not n:
            continue
        itxt = f"{np.median(its):5.0f} ({min(its)}-{max(its)})" if its else "      -      "
        print(f"{nm:14s} {n:4d}  {itxt:15s}  {ok:3d}/{n:<3d}      {np.mean(ex) if ex else float('nan'):6.1f}%"
              f"   {np.mean(pick):8.2f}         {np.mean(mix):8.2f}")


def mapper_names(R, names, amounts):
    rn, ra = R["names"], R["amounts"]
    out = []
    for nm, amt in zip(names, amounts):
        same = [i for i, x in enumerate(rn) if kind(x) == kind(nm)]
        out.append(min(same, key=lambda i: abs(ra[i] - (amt or 0))) if same else None)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records"); ap.add_argument("out"); ap.add_argument("games", nargs="*")
    ap.add_argument("--spots", type=int, default=14); ap.add_argument("--hands", type=int, default=30)
    ap.add_argument("--budget", type=float, default=1.0); ap.add_argument("--ref-iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3); ap.add_argument("--sum", action="store_true")
    a = ap.parse_intermixed_args()
    summary(a) if a.sum else run(a)
