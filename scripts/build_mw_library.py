"""Builds the precomputed 3-way flop library (gto/mw_library.py) offline: every spot in mw_library.SPOTS on a set of
canonical flops, one 3-player solve from the flop's start on the library's game (native Accurate's), run until its
players' mean best-response gain is under --target % of the pot (checked every --check iterations from --min-iters,
at most --max-iters).

    python scripts/build_mw_library.py --jobs 8                            # every spot: REP_FLOPS (184) + EXTRA_FLOPS
    python scripts/build_mw_library.py --boards Kh7d2c,9s8s5d              # given flops
    python scripts/build_mw_library.py --index-only                        # rewrite mw_index.json from the entries

Resumable as scripts/build_flop_library.py: an entry already on disk for the current charts and tree is kept. The
index is rewritten every --index-every entries and at the end. One progress line per entry (spot, flop, iterations,
exploitability, seconds, bytes).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gto.native  # noqa: F401,E402  (holds BLAS to one thread before numpy loads)
from gto import flop_library as fl  # noqa: E402
from gto import mw_library as ml  # noqa: E402
from gto.cards import cards as parse_cards, cards_str  # noqa: E402

_SETUPS = {}


def _setup(line):
    if line not in _SETUPS:
        _SETUPS[line] = ml.spot_setup(line)
    return _SETUPS[line]


def _meta(sid, s):
    return {"spot": sid, "fp": [fl.range_fp(r) for r in s["ranges"]], "pot": s["pot"], "stack": s["stack"],
            "tree": ml.tree_key(ml.library_config()), "seats": s["seats"]}


def current(path: str, meta: dict) -> bool:
    """Whether the entry at path was built for these ranges and this tree."""
    try:
        with open(path, "rb") as f:
            head = ml.decode(f.read())["head"]
    except (OSError, ValueError):
        return False
    return head.get("players") == 3 and head.get("all_hands") is True and head.get("fp") == meta["fp"] and head.get("tree") == fl.tree_hash(meta["tree"])


def solve_entry(job):
    """Solves one (spot, flop) and writes its entry. Returns a progress line."""
    sid, line, name, out, a = job
    s = _setup(line)
    meta = dict(_meta(sid, s), board=name, all_hands=True)
    t0 = time.time()
    sol = ml.solve_all(parse_cards(name), s["ranges"], s["pot"], s["stack"])
    sol.run(iters=a["min_iters"])
    expl = sol.br_gains().mean() / sol.root_pot() * 100
    while expl > a["target"] and sol.iters < a["max_iters"]:
        sol.run(iters=min(a["check"], a["max_iters"] - sol.iters))
        expl = sol.br_gains().mean() / sol.root_pot() * 100
    data = ml.encode(sol, meta)
    path = os.path.join(out, sid, name + ".bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)
    return f"{sid} {name} iters={sol.iters} expl={expl:.3f}% secs={time.time() - t0:.0f} bytes={len(data)}"


def write_index(out: str, spots) -> dict:
    """mw_index.json from the entries on disk that match the current charts and tree."""
    idx = {"format": ml.FORMAT, "tree": ml.tree_key(ml.library_config()), "spots": []}
    stale = total = 0
    for sid, label, line in spots:
        s = _setup(line)
        meta = _meta(sid, s)
        d = os.path.join(out, sid)
        flops = {}
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".bin"):
                    continue
                p = os.path.join(d, fn)
                if not current(p, meta):
                    stale += 1
                    continue
                with open(p, "rb") as f:
                    data = f.read()
                total += len(data)
                flops[fn[:-4]] = hashlib.sha1(data).hexdigest()[:10]
        if flops:
            idx["spots"].append({"id": sid, "label": label, "line": line, "seats": s["seats"], "fp": meta["fp"],
                                 "pot": s["pot"], "stack": s["stack"], "flops": flops})
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, ml.INDEX + ".tmp")
    with open(tmp, "w") as f:
        json.dump(idx, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(out, ml.INDEX))
    n = sum(len(sp["flops"]) for sp in idx["spots"])
    print(f"index: {len(idx['spots'])} spots, {n} flops, {total / 1e6:.2f} MB of entries"
          + (f" ({stale} stale entries left out)" if stale else ""), flush=True)
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default="", help="comma separated spot ids (default: all)")
    ap.add_argument("--flops", default="", help="how many representative flops per spot (default: the spot's "
                    "mw_library.SPOT_FLOPS, else REP_FLOPS)")
    ap.add_argument("--boards", default="", help="comma separated flops instead (any suits: each is canonicalized)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--min-iters", type=int, default=200)
    ap.add_argument("--max-iters", type=int, default=800)
    ap.add_argument("--check", type=int, default=100, help="iterations between best-response checks")
    ap.add_argument("--target", type=float, default=0.1, help="stop under this mean best-response gain (%% of the pot)")
    ap.add_argument("--out", default=ml.DIR)
    ap.add_argument("--limit", type=int, default=0, help="solve at most this many entries this run")
    ap.add_argument("--index-every", type=int, default=10)
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args(argv)
    want = [x for x in a.spots.split(",") if x]
    spots = [s for s in ml.SPOTS if not want or s[0] in want]
    if want and len(spots) != len(want):
        ap.error(f"unknown spot in {want}; known: {[s[0] for s in ml.SPOTS]}")
    if a.index_only:
        write_index(a.out, ml.SPOTS)
        return
    jobs = []
    opts = {"min_iters": a.min_iters, "max_iters": a.max_iters, "check": a.check, "target": a.target}
    for sid, label, line in spots:
        meta = _meta(sid, _setup(line))
        if a.boards:
            names = list(dict.fromkeys(fl.flop_name(parse_cards(b)) for b in a.boards.split(",") if b))
        else:
            n = int(a.flops) if a.flops else ml.SPOT_FLOPS.get(sid, ml.REP_FLOPS)
            names = list(dict.fromkeys([cards_str(f) for f in fl.representative_flops(n)] + ml.EXTRA_FLOPS))
        jobs += [(sid, line, nm, a.out, opts) for nm in names
                 if not current(os.path.join(a.out, sid, nm + ".bin"), meta)]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} entries to solve ({len(spots)} spots, the rest on disk), {a.jobs} processes", flush=True)
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for done, msg in enumerate(pool.imap_unordered(solve_entry, jobs), 1):
                el = time.time() - t0
                print(f"[{done}/{len(jobs)} {el / 60:.0f} min, ~{el / done * (len(jobs) - done) / 60:.0f} min left] {msg}",
                      flush=True)
                if done % a.index_every == 0:
                    write_index(a.out, ml.SPOTS)
    write_index(a.out, ml.SPOTS)


if __name__ == "__main__":
    main()
