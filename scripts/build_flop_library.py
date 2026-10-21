"""Builds the precomputed flop library (gto/flop_library.py) offline: every spot in flop_library.SPOTS on a set of
canonical flops, solved with the library's tree (the browser's Accurate config) until it is within --target % of
the pot from equilibrium (checked every --check iterations from --min-iters, at most --max-iters).

    python scripts/build_flop_library.py --jobs 5                       # every spot, 184 flops each
    python scripts/build_flop_library.py --spots srp_btn_bb --flops all # one spot on all 1755 flops
    python scripts/build_flop_library.py --spots srp_btn_bb --boards Ks7d2c,Qs8s3d   # given flops
    python scripts/build_flop_library.py --index-only                   # rewrite index.json from the entries

Each index rewrite also writes <spot>/report.json (flop_library.build_report: per flop, the range frequencies at each
stored node, both players' EV and equity; the page's flop reports), recomputing only the flops whose entries changed.

Resumable: an entry already on disk for the current charts and tree is kept, so a stopped run picks up where it was
(run it again with the same arguments). An entry whose spot's ranges or tree no longer match (the charts or the
library's config changed) is solved again, and left out of the index until it is. The index is rewritten every
--index-every entries and at the end. Progress lines go to stdout (one per entry: spot, flop, iterations,
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
from gto.cards import cards as parse_cards, cards_str  # noqa: E402
from gto.solver import solve  # noqa: E402

_SETUPS = {}


def _setup(line):
    if line not in _SETUPS:
        _SETUPS[line] = fl.spot_setup(line)
    return _SETUPS[line]


def _meta(sid, s):
    return {"spot": sid, "fp": [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])], "pot": s["pot"],
            "stack": s["stack"], "tree": fl.tree_key(fl.library_config()), "seats": [s["oop"], s["ip"]]}


def current(path: str, meta: dict) -> bool:
    """Whether the entry at path was built for these ranges and this tree."""
    try:
        with open(path, "rb") as f:
            head = fl.decode(f.read())["head"]
    except (OSError, ValueError):
        return False
    return (head.get("format") == fl.FORMAT and head.get("fp") == meta["fp"]
            and fl._key_str(head.get("tree")) == fl._key_str(meta["tree"]))


def solve_entry(job):
    """Solves one (spot, flop) and writes its entry. Returns a progress line."""
    sid, line, name, out, a = job
    s = _setup(line)
    meta = dict(_meta(sid, s), board=name)
    board = parse_cards(name)
    t0 = time.time()
    sol = solve(board, s["full0"], s["full1"], s["pot"], s["stack"], cfg=fl.library_config(), time_budget=1e9,
                max_iters=a["min_iters"])
    expl = sol.exploitability()
    while expl > a["target"] and sol.iters < a["max_iters"]:
        sol.extend(1e9, max_iters=min(sol.iters + a["check"], a["max_iters"]))
        expl = sol.exploitability()
    data = fl.encode(sol, meta)
    path = os.path.join(out, sid, name + ".bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return f"{sid} {name} iters={sol.iters} expl={expl:.3f}% secs={time.time() - t0:.0f} bytes={len(data)}"


def write_report(out: str, sp: dict, s: dict) -> str:
    """<spot>/report.json (flop_library.build_report: the spot's strategy across its flops, for the page's flop
    reports), reusing the flops of the one on disk whose entries are unchanged; returns its content hash."""
    path = os.path.join(out, sp["id"], "report.json")
    try:
        with open(path) as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = None

    def read(name):
        with open(os.path.join(out, sp["id"], name + ".bin"), "rb") as f:
            return f.read()
    rep = fl.build_report(sp, s["full0"], s["full1"], read, old)
    data = json.dumps(rep, separators=(",", ":")).encode()
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)
    return hashlib.sha1(data).hexdigest()[:10]


def write_ranges(out: str, sp: dict, s: dict) -> None:
    """<spot>/ranges.json (flop_library.spot_ranges: the ranges entering the flop, for the page's drills)."""
    path = os.path.join(out, sp["id"], "ranges.json")
    with open(path + ".tmp", "w") as f:
        json.dump(fl.spot_ranges(sp["id"], s["full0"], s["full1"]), f, separators=(",", ":"))
    os.replace(path + ".tmp", path)


def write_index(out: str, spots) -> dict:
    """index.json from the entries on disk that match the current charts and tree (and each spot's report.json)."""
    idx = {"format": fl.FORMAT, "tree": fl.tree_key(fl.library_config()), "spots": []}
    stale = 0
    total = 0
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
            sp = {"id": sid, "label": label, "line": line, "seats": [s["oop"], s["ip"]],
                  "fp": meta["fp"], "pot": s["pot"], "stack": s["stack"], "flops": flops}
            sp["report"] = write_report(out, sp, s)
            write_ranges(out, sp, s)
            idx["spots"].append(sp)
    tmp = os.path.join(out, "index.json.tmp")
    os.makedirs(out, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(idx, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(out, "index.json"))
    n = sum(len(sp["flops"]) for sp in idx["spots"])
    print(f"index: {len(idx['spots'])} spots, {n} flops, {total / 1e6:.2f} MB of entries"
          + (f" ({stale} stale entries left out)" if stale else ""), flush=True)
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default="", help="comma separated spot ids (default: all, most common first)")
    ap.add_argument("--flops", default="184", help="how many representative flops per spot, or 'all' (1755)")
    ap.add_argument("--boards", default="", help="comma separated flops instead (any suits: each is canonicalized)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--min-iters", type=int, default=200)
    ap.add_argument("--max-iters", type=int, default=500)
    ap.add_argument("--check", type=int, default=100, help="iterations between exploitability checks")
    ap.add_argument("--target", type=float, default=0.15, help="stop under this exploitability (%% of the pot)")
    ap.add_argument("--out", default=fl.DIR)
    ap.add_argument("--limit", type=int, default=0, help="solve at most this many entries this run")
    ap.add_argument("--index-every", type=int, default=20)
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args(argv)
    want = [x for x in a.spots.split(",") if x]
    spots = [s for s in fl.SPOTS if not want or s[0] in want]
    if want and len(spots) != len(want):
        ap.error(f"unknown spot in {want}; known: {[s[0] for s in fl.SPOTS]}")
    if a.index_only:
        write_index(a.out, fl.SPOTS)
        return
    if a.boards:
        names = list(dict.fromkeys(fl.flop_name(parse_cards(b)) for b in a.boards.split(",") if b))
    else:
        names = [cards_str(f) for f in fl.representative_flops(1755 if a.flops == "all" else int(a.flops))]
    jobs = []
    opts = {"min_iters": a.min_iters, "max_iters": a.max_iters, "check": a.check, "target": a.target}
    for sid, label, line in spots:
        meta = _meta(sid, _setup(line))
        for nm in names:
            if not current(os.path.join(a.out, sid, nm + ".bin"), meta):
                jobs.append((sid, line, nm, a.out, opts))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} entries to solve ({len(spots)} spots x {len(names)} flops, the rest on disk), "
          f"{a.jobs} processes", flush=True)
    t0 = time.time()
    done = 0
    if jobs:
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for msg in pool.imap_unordered(solve_entry, jobs):
                done += 1
                el = time.time() - t0
                print(f"[{done}/{len(jobs)} {el / 60:.0f} min, ~{el / done * (len(jobs) - done) / 60:.0f} min left] {msg}",
                      flush=True)
                if done % a.index_every == 0:
                    write_index(a.out, fl.SPOTS)
    write_index(a.out, fl.SPOTS)


if __name__ == "__main__":
    main()
