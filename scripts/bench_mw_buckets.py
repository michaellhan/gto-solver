"""Multiway flop bucket games scored per combo against a finer game, on spots from a coverage run.

A game's exploitability is measured inside its own abstraction, so it can't see the abstraction's error: 3-way, BTN
5d5c facing a lead on 4c5h2h read 0.02% from equilibrium in 6 buckets with every hand's call worth about 0 (55 -0.16 bb,
a wheel +0.99) where 24 buckets give +13.2 and +23.1. This takes the flop decisions of 3- and 4-way pots from a
coverage_fuzz.py jsonl (nobody all-in), solves each in every game for the same iterations and compares the root
player's per-combo EVs with the first game's (the reference): for the reference's best action, weighted by the range,
and over the range's top tenth by value (the hands whose EVs collapse), in % of the pot; and the reference's EV of each
game's mix against its own ("mix loss").

  python scripts/bench_mw_buckets.py capture .work/coverage/live_11.jsonl spots.pkl --n3 14 --n4 10
  python scripts/bench_mw_buckets.py run spots.pkl out.json 'ref={"buckets":24,"bucket_river_step":3,"four":{"buckets":16}}' \
      'q6={"buckets":6,"bucket_river_step":6}' 'q6p1={"buckets":6,"bucket_river_step":6,"bucket_power":1}'
  python scripts/bench_mw_buckets.py sum out.json ref q6 q6p1

A game is MWConfig fields as JSON over the spot's own (the advisor's Fast flop game), "four" a 4-way pot's own
overrides, "iters" its iterations (200). Results are kept in out.json per spot and game, so a run resumes.
"""
import sys, os, time, json, pickle, dataclasses, argparse
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import gto.multiway_solver as M


class _Got(Exception):
    pass


def capture(a):
    import coverage_fuzz as CF
    recs = [r for r in map(json.loads, open(a.records)) if r.get("tags", {}).get("street") == "flop" and "state" in r
            and r["tags"].get("in_pot") in (3, 4) and not r["tags"].get("allin_in")]
    np.random.default_rng(a.seed).shuffle(recs)
    cap = {}

    def init(self, *args, **kw):
        cap["a"], cap["k"] = args, dict(kw)
        raise _Got()
    M.MultiwaySolver.__init__ = init
    spots, want, got = [], {3: a.n3, 4: a.n4}, {3: 0, 4: 0}
    for r in recs:
        npl = r["tags"]["in_pot"]
        if got[npl] >= want[npl]:
            continue
        st = {k: v for k, v in r["state"].items() if k not in ("quick", "budget", "prior_budget")}
        st["speed"] = "fast"
        adv = CF.advisors("server")["fast"]
        adv.time_budget = 60
        cap.clear()
        try:
            adv.advise(st)
        except _Got:
            pass
        if "a" not in cap or len(cap["a"][1]) != npl:
            continue
        kw = {k: v for k, v in cap["k"].items() if k not in ("deadline", "t_init")}
        spots.append({"id": f"{r['hand']}:{r['k']}", "np": npl, "a": cap["a"], "k": kw})
        got[npl] += 1
        print(spots[-1]["id"], f"{npl}-way", flush=True)
    pickle.dump(spots, open(a.spots, "wb"))


def run(a):
    spots = pickle.load(open(a.spots, "rb"))
    games = [(g.split("=", 1)[0], json.loads(g.split("=", 1)[1])) for g in a.games]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for sp in spots:
        for name, g in games:
            key = f"{sp['id']}|{name}"
            if key in res:
                continue
            g = dict(g)
            iters, four = g.pop("iters", 200), g.pop("four", None)
            if sp["np"] == 4 and four:
                g.update(four)
            cfg = dataclasses.replace(sp["k"]["cfg"], **g)
            t = time.time()
            s = M.MultiwaySolver(*sp["a"], **{**sp["k"], "cfg": cfg})
            s.run(iters=iters)
            p = s.root.player
            res[key] = {"names": list(s.root.names), "ev": s.action_evs().tolist(), "sig": s.root.avg_strategy()[0].tolist(),
                        "w": s.reach0[p, 0].tolist(), "pot": s.root_pot(), "np": sp["np"],
                        "expl": float(s.br_gains().mean() / s.root_pot() * 100), "build": s.build_seconds,
                        "secs": time.time() - t}
            print(f"{key}: {res[key]['expl']:.2f}% from equilibrium, tables {s.build_seconds:.2f} s, {time.time() - t:.1f} s",
                  flush=True)
            json.dump(res, open(a.out, "w"))


def summary(a):
    res = json.load(open(a.out))
    ref, names = a.games[0], a.games[1:]
    for npl in (3, 4, None):
        print(f"-- {npl}-way" if npl else "-- all")
        for nm in names:
            err, strong, loss, expl, worst = [], [], [], [], 0.0
            for sid in sorted({k.split("|")[0] for k in res}):
                R, V = res.get(f"{sid}|{ref}"), res.get(f"{sid}|{nm}")
                if not R or not V or V["names"] != R["names"] or (npl and R["np"] != npl):
                    continue
                w, evr, sr, pot = np.array(R["w"]), np.array(R["ev"]), np.array(R["sig"]), R["pot"]
                ev, sg = np.array(V["ev"]), np.array(V["sig"])
                mix = (sr * evr).sum(axis=0)
                e = np.abs(np.take_along_axis(ev - evr, evr.argmax(axis=0)[None], 0)[0]) / pot * 100
                live = w > 0
                top = live & (mix >= np.quantile(mix[live], 0.9))
                W = w / w.sum()
                err.append((W * e).sum()); strong.append(e[top].mean())
                loss.append((W * (mix - (sg * evr).sum(axis=0))).sum() / pot * 100)
                expl.append(V["expl"]); worst = max(worst, e[live].max())
            if err:
                print(f"{nm:12s} n={len(err):2d}  EV {np.mean(err):5.2f}% of the pot (top tenth {np.mean(strong):5.2f}, "
                      f"worst combo {worst:5.1f})  mix loss {np.mean(loss):5.2f}% (worst spot {np.max(loss):.2f})  "
                      f"own expl {np.mean(expl):.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("records"); c.add_argument("spots")
    c.add_argument("--n3", type=int, default=14); c.add_argument("--n4", type=int, default=10); c.add_argument("--seed", type=int, default=5)
    r = sub.add_parser("run"); r.add_argument("spots"); r.add_argument("out"); r.add_argument("games", nargs="+")
    s = sub.add_parser("sum"); s.add_argument("out"); s.add_argument("games", nargs="+")
    a = ap.parse_args()
    {"capture": capture, "run": run, "sum": summary}[a.cmd](a)
