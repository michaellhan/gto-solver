"""Warm-starting a live flop solve from the precomputed library's most similar stored flop (README, Precomputed
flop library: "Flops not stored"). A spot the library holds, on a flop it doesn't store, is solved live; this
seeds that solve from the stored flop most like it and measures exploitability (% of the pot) at iteration
checkpoints against a cold start, on 12 flops of a spot the library doesn't store (5 rainbow, 5 two-tone, a
monotone and a paired one, seeded draw).

The similar flop: the same pairing and suit pattern, ranks nearest (summed rank gaps). Its strategies come across by
card substitution: the target flop's cards onto the stored flop's, the other ranks in order, the suits as the two
patterns pair them, so every combo has a substitute (KcJd6h from KcJd5h: a 6 is played as the 5, a 5 as the 4...).
At each stored node the tree has too, a combo's regrets are its substitute's stored strategy times --scale x the
pot x the other player's range (so the first iterations play it), and its strategy sum that strategy once.

    python scripts/bench_library_warm.py OUT.jsonl --config accurate --modes cold,warm:10,warm:30 --jobs 4
    python scripts/bench_library_warm.py OUT.jsonl --config accurate --noblas --checkpoints 10,20,40,60,80
    python scripts/bench_library_warm.py OUT.jsonl --report

--config: gto.native's fast / accurate, or browser_accurate (the worker's Accurate: 64 buckets over every river).
--noblas runs the browser's kernels (solver.NO_BLAS). Each line of OUT is one (flop, config, mode): its checkpoints
(iterations, seconds, exploitability); a run skips lines already there. --library: a library folder other than the
shipped one (the stored flops change as the library grows: freeze a copy, e.g. git archive, for comparable runs).
"""
from __future__ import annotations
import argparse
import collections
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Dict, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gto.native as gn  # noqa: E402  (holds BLAS to one thread before numpy loads)
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import solver as S  # noqa: E402
from gto.cards import ALL_COMBOS, COMBO_INDEX, cards as parse_cards, cards_str  # noqa: E402


def flop_card_map(target: Sequence[int], source: Sequence[int]) -> Optional[Tuple[np.ndarray, int]]:
    """(m, distance): m[c] the source card standing for target card c (a bijection of the 52 cards taking the target
    flop onto the source flop, rank order and pairing kept, suits as the two flops' suit patterns pair them),
    distance the flops' summed rank gaps; None when their pairing or suit pattern differ."""
    t = sorted(target, key=lambda c: -c)
    best = None
    for pi in itertools.permutations(range(3)):
        s = [source[i] for i in pi]
        if any((t[i] // 4 > t[j] // 4) != (s[i] // 4 > s[j] // 4) or (t[i] // 4 == t[j] // 4) != (s[i] // 4 == s[j] // 4)
               for i in range(3) for j in range(3)):
            continue
        suit: Dict[int, int] = {}
        if any(suit.setdefault(t[i] % 4, s[i] % 4) != s[i] % 4 for i in range(3)) or len(set(suit.values())) != len(suit):
            continue
        rest = [x for x in range(4) if x not in suit.values()]
        for x in range(4):
            if x not in suit:
                suit[x] = rest.pop(0)
        rank = {t[i] // 4: s[i] // 4 for i in range(3)}
        rank.update(zip([r for r in range(13) if r not in rank], [r for r in range(13) if r not in rank.values()]))
        d = sum(abs(t[i] // 4 - s[i] // 4) for i in range(3))
        if best is None or d < best[1]:
            best = (np.array([rank[c // 4] * 4 + suit[c % 4] for c in range(52)]), d)
    return best


def similar_flop(names: Sequence[str], board: Sequence[int]):
    """(name, card map, distance): the stored flop most like `board` (least distance, then the higher cards' gaps)."""
    best = None
    for nm in names:
        src = parse_cards(nm)
        r = flop_card_map(board, src)
        if r is None:
            continue
        key = (r[1], [abs(a // 4 - b // 4) for a, b in zip(sorted(board, reverse=True), sorted(src, reverse=True))])
        if best is None or key < best[0]:
            best = (key, nm, r[0])
    return None if best is None else (best[1], best[2], best[0][0])


def warm_from_similar(lib: fl.FlopLibrary, sol, full0, full1, pot: float, stack: float, scale: float) -> Optional[dict]:
    """Seeds a flop solve not yet iterated (solve(..., max_iters=0)) of a spot the library holds, on a flop it
    doesn't store, from its most similar stored flop. {'src', 'distance', 'nodes'} or None."""
    spot = lib.spot(fl.range_fp(full0), fl.range_fp(full1), pot, stack)
    if spot is None:
        return None
    canon, smap = fl.canonical(list(sol.root.board))
    if cards_str(canon) in spot.get("flops", {}):
        return None
    sim = similar_flop(sorted(spot.get("flops", {})), list(canon))
    e = lib.entry(spot, sim[0]) if sim else None
    if e is None:
        return None
    card = sim[1][np.array([c // 4 * 4 + smap[c % 4] for c in range(52)])]   # actual -> canonical -> stored flop
    act = sol.ctx.active
    src = np.array([COMBO_INDEX[(min(x, y), max(x, y))] for x, y in
                    zip(card[[ALL_COMBOS[i][0] for i in act]], card[[ALL_COMBOS[i][1] for i in act]])])
    reach, n_seeded = (sol.r0, sol.r1), 0
    for nd in e["nodes"]:
        try:
            n = fl.node_at(sol.root, nd["path"])
        except (ValueError, IndexError, AttributeError):
            continue
        if not isinstance(n, S.Decision) or list(n.names) != nd["names"]:
            continue
        p, m = nd["player"], e["masks"][nd["player"]]
        pos = np.full(len(ALL_COMBOS), -1)
        pos[np.nonzero(m)[0]] = np.arange(int(m.sum()))
        col = pos[src]
        ok = (col >= 0) & (reach[p] > 0)
        st = nd["strat"][:, col[ok]]
        n.regret[0][:, ok] = scale * pot * float(reach[1 - p].sum()) * st
        n.ssum[0][:, ok] = st * reach[p][ok]
        n_seeded += 1
    return {"src": sim[0], "distance": sim[2], "nodes": n_seeded}


LINES = {s[0]: s[2] for s in fl.SPOTS}


def config(name: str) -> S.Config:
    if name == "browser_accurate":       # gto/static/worker.js
        return S.Config(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8, river_classes=True,
                        river_buckets=64, turn_buckets=True, bucket_river_step=1,
                        bet_sizes={"flop": [0.33, 0.75], "turn": [0.33, 0.75, 1.25], "river": [0.33, 0.75, 1.25]},
                        raise_sizes={"flop": [0.5], "turn": [0.75], "river": [0.75]})
    return gn.config(name)


def pick_flops(lib: fl.FlopLibrary, sid: str, seed: int = 3) -> list:
    stored = set(next(s for s in lib.index()["spots"] if s["id"] == sid)["flops"])
    cand = [cards_str(f) for f in fl.all_flops() if cards_str(f) not in stored]
    random.Random(seed).shuffle(cand)
    want = {"rainbow": 5, "twotone": 5, "mono": 1, "paired": 1}
    out = []
    for f in cand:
        b = parse_cards(f)
        k = "paired" if len({c // 4 for c in b}) < 3 else \
            {1: "rainbow", 2: "twotone", 3: "mono"}[max(sum(1 for c in b if c % 4 == s) for s in range(4))]
        if want.get(k, 0) > 0:
            out.append(f)
            want[k] -= 1
    return out


def run(job):
    sid, name, cfg_name, noblas, mode, cks, libdir = job
    S.NO_BLAS = noblas
    s = fl.spot_setup(LINES[sid])
    sol = S.solve(parse_cards(name), s["full0"], s["full1"], s["pot"], s["stack"], cfg=config(cfg_name),
                  time_budget=1e9, max_iters=0)
    info = {}
    if mode.startswith("warm"):
        info = warm_from_similar(fl.FlopLibrary.from_dir(libdir), sol, s["full0"], s["full1"], s["pot"], s["stack"],
                                 float(mode.split(":")[1])) or {}
    pts, secs = [], 0.0
    for ck in cks:
        t0 = time.time()
        sol.solver.run(sol.r0, sol.r1, time_budget=1e9, max_iters=ck, min_iters=1)
        secs += time.time() - t0
        sol.iters = sol.solver.iters
        pts.append((ck, round(secs, 2), round(sol.exploitability(), 3)))
    return dict(sid=sid, flop=name, cfg=cfg_name, blas=int(not noblas), mode=mode, pts=pts, **info)


def report(path: str) -> None:
    rows = [json.loads(x) for x in open(path)]
    groups = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        groups[(r["sid"], r["cfg"], r["blas"])][r["flop"]][r["mode"]] = r
    for (sid, cfg, blas), flops in sorted(groups.items()):
        modes = sorted({m for d in flops.values() for m in d}, key=lambda m: (m != "cold", m))
        cks = [p[0] for p in next(iter(next(iter(flops.values())).values()))["pts"]]
        print(f"{sid} {cfg} {'BLAS' if blas else 'NO_BLAS'}: exploitability (% of the pot, geometric mean of "
              f"{len(flops)} flops) at iterations {cks}; seconds to the last")
        for m in modes:
            ex = np.array([[p[2] for p in d[m]["pts"]] for d in flops.values() if m in d])
            gm = np.exp(np.log(np.maximum(ex, 1e-3)).mean(axis=0))
            secs = np.mean([d[m]["pts"][-1][1] for d in flops.values() if m in d])
            line = f"  {m:9s} " + " ".join(f"{x:7.2f}" for x in gm) + f"   {secs:6.1f} s"
            if m != "cold":
                c = np.array([[p[2] for p in d["cold"]["pts"]] for d in flops.values() if m in d])
                worse = (ex > c * 1.02).sum(axis=0)
                line += "   flops worse than cold: " + " ".join(str(int(x)) for x in worse)
            print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--spot", default="srp_btn_bb")
    ap.add_argument("--config", default="accurate")
    ap.add_argument("--modes", default="cold,warm:10,warm:30")
    ap.add_argument("--checkpoints", default="5,10,20,40,80,120")
    ap.add_argument("--noblas", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--library", default=fl.DIR)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.out)
    cks = [int(x) for x in a.checkpoints.split(",")]
    flops = pick_flops(fl.FlopLibrary.from_dir(a.library), a.spot)
    done = set()
    if os.path.exists(a.out):
        for x in open(a.out):
            d = json.loads(x)
            done.add((d["sid"], d["flop"], d["cfg"], d["blas"], d["mode"]))
    jobs = [(a.spot, f, a.config, a.noblas, m, cks, a.library) for f in flops for m in a.modes.split(",")
            if (a.spot, f, a.config, int(not a.noblas), m) not in done]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for r in pool.imap_unordered(run, jobs):
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r["flop"], r["mode"], r["pts"][-1], flush=True)
    report(a.out)


if __name__ == "__main__":
    main()
