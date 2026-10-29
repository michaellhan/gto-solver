"""Builds the precomputed flop library (gto/flop_library.py) offline: every spot in flop_library.SPOTS on a set of
canonical flops, solved with the library's tree (the browser's Accurate config) until it is within --target % of
the pot from equilibrium (checked every --check iterations from --min-iters, at most --max-iters).

    python scripts/build_flop_library.py --jobs 5                       # every spot, its SPOT_FLOPS (else 184)
    python scripts/build_flop_library.py --spots srp_btn_bb --flops all # one spot on all 1755 flops
    python scripts/build_flop_library.py --spots srp_btn_bb --boards Ks7d2c,Qs8s3d   # given flops
    python scripts/build_flop_library.py --index-only                   # rewrite index.json from the entries
    python scripts/build_flop_library.py --recode                       # format-1 entries to format 2, unsolved
    python scripts/build_flop_library.py --format 40bb --flops 184 --jobs 5   # another game's spots (FORMAT_SPOTS)

Spots of another game (flop_library.FORMAT_SPOTS: a stack depth, ante, table size or straddle; ids like
srp_btn_bb_40bb) are built only when named, by --format TAG or --spots; the index lists every game's spots on disk.

Each index rewrite also writes <spot>/report.json (flop_library.build_report: per flop, the range frequencies at each
stored node, both players' EV and equity; the page's flop reports), recomputing only the flops whose entries changed.

Resumable: an entry already on disk for the current charts and tree is kept, so a stopped run picks up where it was
(run it again with the same arguments). An entry whose spot's ranges or tree no longer match (the charts or the
library's config changed) is solved again, and left out of the index until it is; but for one solved with the river
blind (flop_library.old_tree(), before every spot dealt it), which the index keeps listing (a spot's "old") until it
is solved again, and --keep-old leaves for a later run (a spot's flops not on disk first). The index is rewritten every
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


def _setup(line, fmt=fl.DEFAULT_FORMAT):
    """The spot a line leads to in a game (flop_library.Format), as the advisor plays it."""
    if (line, fmt) not in _SETUPS:
        _SETUPS[(line, fmt)] = fl.spot_setup(line, fmt.players, fmt.stack, fmt.ante, fmt.straddle)
    return _SETUPS[(line, fmt)]


def _spot(sp) -> tuple:
    """(id, label, line, Format) of a spot (flop_library.all_spots'; SPOTS' (id, label, line) is the default game's)."""
    return tuple(sp) if len(sp) > 3 else (*sp, fl.DEFAULT_FORMAT)


def _meta(sid, s):
    return {"spot": sid, "fp": [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])], "pot": s["pot"],
            "stack": s["stack"], "tree": fl.tree_key(fl.library_config(s["pot"], s["stack"])),
            "seats": [s["oop"], s["ip"]]}


def old_tree(meta: dict):
    """The spot's old tree (flop_library.old_tree()) if it isn't the spot's tree now, else None."""
    old = fl.old_tree()
    return None if fl.tree_hash(old) == fl.tree_hash(meta["tree"]) else old


def current(path: str, meta: dict, tree: dict = None) -> bool:
    """Whether the entry at path was built for these ranges and this tree (`tree`, else the spot's)."""
    try:
        with open(path, "rb") as f:
            head = fl.decode(f.read())["head"]
    except (OSError, ValueError):
        return False
    return (head.get("format") == fl.FORMAT and head.get("fp") == meta["fp"]
            and head.get("tree") == fl.tree_hash(tree or meta["tree"]))


def solve_entry(job):
    """Solves one (spot, flop) and writes its entry. Returns a progress line."""
    sid, line, fmt, name, out, a = job
    s = _setup(line, fl.Format(**fmt))
    meta = dict(_meta(sid, s), board=name)
    board = parse_cards(name)
    t0 = time.time()
    sol = solve(board, s["full0"], s["full1"], s["pot"], s["stack"], cfg=fl.library_config(s["pot"], s["stack"]),
                time_budget=1e9, max_iters=a["min_iters"])
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


def recode(out: str, spots) -> None:
    """Re-encodes every format-1 entry on disk in the current format (flop_library.recode: from its stored values,
    without solving again), in place; an entry for other ranges than the spot's now is left as it is (stale)."""
    n = before = after = 0
    for sid, label, line, fmt in map(_spot, spots):
        d = os.path.join(out, sid)
        if not os.path.isdir(d):
            continue
        s = _setup(line, fmt)
        fp = [fl.range_fp(s["full0"]), fl.range_fp(s["full1"])]
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".bin"):
                continue
            p = os.path.join(d, fn)
            with open(p, "rb") as f:
                data = f.read()
            if data[:len(fl.MAGIC_V1)] != fl.MAGIC_V1 or fl.decode(data)["head"].get("fp") != fp:
                continue
            new = fl.recode(data, s["full0"], s["full1"])
            with open(p + ".tmp", "wb") as f:
                f.write(new)
            os.replace(p + ".tmp", p)
            n, before, after = n + 1, before + len(data), after + len(new)
    print(f"recoded {n} entries: {before / 1e6:.2f} MB -> {after / 1e6:.2f} MB", flush=True)


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
    """index.json from the entries on disk that match the current charts and tree (and each spot's report.json).
    A spot of another game than the default carries it ("game": flop_library.Format's fields)."""
    idx = {"format": fl.FORMAT, "tree": fl.tree_key(fl.library_config()), "spots": []}
    stale = n_old = 0
    total = 0
    for sid, label, line, fmt in map(_spot, spots):
        d = os.path.join(out, sid)
        if not os.path.isdir(d):
            continue
        s = _setup(line, fmt)
        meta = _meta(sid, s)
        old = old_tree(meta)
        flops, olds = {}, []
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".bin"):
                    continue
                p = os.path.join(d, fn)
                if not current(p, meta):
                    if old is None or not current(p, meta, old):
                        stale += 1
                        continue
                    olds.append(fn[:-4])
                with open(p, "rb") as f:
                    data = f.read()
                total += len(data)
                flops[fn[:-4]] = hashlib.sha1(data).hexdigest()[:10]
        if flops:
            sp = {"id": sid, "label": label, "line": line, "seats": [s["oop"], s["ip"]],
                  "fp": meta["fp"], "pot": s["pot"], "stack": s["stack"], "tree": meta["tree"], "flops": flops}
            if olds:
                sp["old_tree"], sp["old"] = old, olds
                n_old += len(olds)
            if fmt != fl.DEFAULT_FORMAT:
                sp["game"] = fmt.as_dict()
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
          + (f", {n_old} on the old tree" if n_old else "") + (f" ({stale} stale entries left out)" if stale else ""),
          flush=True)
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default="", help="comma separated spot ids (default: all, most common first)")
    ap.add_argument("--format", default="", help="comma separated game tags (flop_library.FORMAT_SPOTS: 40bb, ...): "
                    "their spots (default: the default game's; with --spots, the named spots of every game)")
    ap.add_argument("--flops", default="", help="how many representative flops per spot, or 'all' (1755); default: "
                    "the spot's flop_library.SPOT_FLOPS, else REP_FLOPS")
    ap.add_argument("--boards", default="", help="comma separated flops instead (any suits: each is canonicalized)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--min-iters", type=int, default=200)
    ap.add_argument("--max-iters", type=int, default=1000)   # (paired and trips boards settle slowest: 600-800)
    ap.add_argument("--check", type=int, default=100, help="iterations between exploitability checks")
    ap.add_argument("--target", type=float, default=0.15, help="stop under this exploitability (%% of the pot)")
    ap.add_argument("--out", default=fl.DIR)
    ap.add_argument("--limit", type=int, default=0, help="solve at most this many entries this run")
    ap.add_argument("--index-every", type=int, default=20)
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--keep-old", action="store_true", help="leave the entries on the old tree (the river blind) for a "
                    "later run: solve the flops not on disk first")
    ap.add_argument("--recode", action="store_true", help="re-encode format-1 entries (no solving), then the index")
    a = ap.parse_args(argv)
    want = [x for x in a.spots.split(",") if x]
    tags = [x for x in a.format.split(",") if x]
    unknown = [t for t in tags if t not in fl.FORMAT_SPOTS and t != "default"]
    if unknown:
        ap.error(f"unknown game {unknown}; known: {list(fl.FORMAT_SPOTS)}")
    every = fl.all_spots()
    pick = [t if t != "default" else "" for t in tags] or ([""] if not want else None)
    spots = [s for s in fl.all_spots(pick) if not want or s[0] in want]
    if want and len(spots) != len(want):
        ap.error(f"unknown spot in {want}; known: {[s[0] for s in every]}")
    if a.recode:
        recode(a.out, spots)
    if a.index_only or a.recode:
        write_index(a.out, every)
        return
    def names_for(sid):
        if a.boards:
            return list(dict.fromkeys(fl.flop_name(parse_cards(b)) for b in a.boards.split(",") if b))
        n = 1755 if a.flops == "all" else int(a.flops) if a.flops else fl.SPOT_FLOPS.get(sid, fl.REP_FLOPS)
        return [cards_str(f) for f in fl.representative_flops(n)]
    jobs = []
    opts = {"min_iters": a.min_iters, "max_iters": a.max_iters, "check": a.check, "target": a.target}
    n_flops = 0
    for sid, label, line, fmt in spots:
        meta = _meta(sid, _setup(line, fmt))
        old = old_tree(meta) if a.keep_old else None
        names = names_for(sid)
        n_flops += len(names)
        for nm in names:
            p = os.path.join(a.out, sid, nm + ".bin")
            if not current(p, meta) and not (old is not None and current(p, meta, old)):
                jobs.append((sid, line, fmt.as_dict(), nm, a.out, opts))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} entries to solve ({len(spots)} spots, {n_flops} flops, the rest on disk), "
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
                    write_index(a.out, every)
    write_index(a.out, every)


if __name__ == "__main__":
    main()
