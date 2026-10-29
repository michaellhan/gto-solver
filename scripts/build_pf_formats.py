"""Solves the preflop game of other formats and writes their chart moves (gto/pf_formats.py) to gto/pf_format_data.py.

    python scripts/build_pf_formats.py --solve 20bb,40bb,a0.125 --dir DIR     # solve (or keep) each game, DIR/<tag>.npz
    python scripts/build_pf_formats.py --write --dir DIR                       # the moves from the solves in DIR

A tag is a flop_library.Format's ('' the 6-max 100bb game, '40bb', 'a0.125', '40bb_a0.125', '8max', 'straddle',
'2max', '2max_40bb'; 'mp2' appended: at most two players to a flop, the 7-9 handed games' tree; 'limp': the small
blind may limp, the heads-up games' limp moves). Each game is
pf_game's tree for that table (its default sizes, raked as the charts are: RAKE) with the flop library's realization
(pf_game.LibraryModel, advisor._river_fix at RIVER_FIX), another depth's scaled by DEPTH_SCALE (the excess over equity measured there by
live solves against 100bb's on the same flops: scripts/pf_depth_values.py). One process, one thread.
--write writes the moves of SETS (pf_formats' grids): 150bb, 200bb and heads-up 200bb solve (--solve 150bb,200bb,
2max_200bb,2max_200bb_limp) but aren't written (pf_formats.DEPTHS: deeper plays 100bb's moves and the rules' implied
odds).
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import gto.native  # noqa: F401,E402
import numpy as np  # noqa: E402
from gto import pf_game as G  # noqa: E402
from gto import pf_formats as PF  # noqa: E402
from gto.flop_library import Format  # noqa: E402

RIVER_FIX = (1.0, 0.5)          # advisor._river_fix's scale in single-raised and 3bet pots (as the 100bb solve)
RAKE = (0.05, 4.0)              # every game is raked as the solves the charts were fit to (5% capped at 4bb), so a
                                # move is measured in the charts' own game (unraked, the margins move much more: the
                                # big blind's defence against the button +34 points with a 0.125bb ante, raked +24)
# {depth: {pot type: k}}: each share's excess over equity at that depth is k times 100bb's (pf_depth_values.py --fit:
# live solves of BTN vs BB, SB vs BB and BB vs BTN's 3bet pot on 12 of the 184 representative flops at 150
# iterations against the library's at 100bb; linear in depth between, 1 at 100). Single-raised pots barely move
# (BTN vs BB 0.90 / 0.98 / 0.96, SB vs BB 0.88 / 0.98 / 0.95 at 20 / 40 / 200bb), 3bet pots realize less as the
# stack shrinks to the pot (0.48 at 20bb, 0.76 at 40, 1.03 at 200); 4bet pots (type 3, no measure) as 3bet pots
DEPTH_SCALE = {20.0: {1: 0.890, 2: 0.475, 3: 0.475},
               40.0: {1: 0.977, 2: 0.762, 3: 0.762},
               100.0: {1: 1.0, 2: 1.0, 3: 1.0},
               200.0: {1: 0.956, 2: 1.034, 3: 1.034}}
ITERS = 1500

# the moves: (dimension, sub-key, game, base game)
SETS = ([("depth", d, f"{d:g}bb", "") for d in PF.DEPTHS if d != 100.0] +
        [("hu_depth", d, f"2max_{d:g}bb", "2max") for d in PF.HU_DEPTHS if d != 100.0] +
        [("ante", d, (f"{d:g}bb_" if d != 100.0 else "") + f"a{PF.ANTE:g}", f"{d:g}bb" if d != 100.0 else "")
         for d in PF.ANTE_DEPTHS] +
        [("players", 7.0, "7max_mp2", "mp2"), ("players", 8.0, "8max_mp2", "mp2"), ("straddle", 100.0, "straddle", "")] +
        [("limp", d, f"2max_{d:g}bb_limp" if d != 100.0 else "2max_limp", f"2max_{d:g}bb" if d != 100.0 else "2max")
         for d in PF.HU_DEPTHS])


def depth_scale(depth: float) -> dict:
    ds = sorted(DEPTH_SCALE)
    x = min(max(depth, ds[0]), ds[-1])
    lo, hi = max(d for d in ds if d <= x), min(d for d in ds if d >= x)
    t = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return {k: (1 - t) * DEPTH_SCALE[lo].get(k, 1.0) + t * DEPTH_SCALE[hi].get(k, 1.0) for k in (1, 2, 3)}


def parse(tag: str):
    """(Format, max players to a flop, whether the small blind may limp) of a tag."""
    mp, limp = 3, False
    parts = [p for p in tag.split("_") if p]
    if "mp2" in parts:
        mp = 2
        parts.remove("mp2")
    if "limp" in parts:
        limp = True
        parts.remove("limp")
    return Format.parse("_".join(parts)), mp, limp


def game(tag: str):
    from gto.advisor import _river_fix
    from gto.preflop import SEATS
    fmt, mp, limp = parse(tag)
    pos = SEATS[fmt.players]
    model = G.LibraryModel(river_fix=_river_fix(), river_scale={1: RIVER_FIX[0], 2: RIVER_FIX[1]},
                           scale=depth_scale(fmt.stack))
    root = G.build(pos, stack=fmt.stack, ante=fmt.ante, straddle=fmt.straddle, max_players=mp, sb_limp=limp)
    return G.Game(root, pos, model, stack=fmt.stack, rake=RAKE)


def path(d: str, tag: str) -> str:
    return os.path.join(d, (tag or "6max_100bb") + ".npz")


def solved(d: str, tag: str):
    g = game(tag)
    g.load(path(d, tag))
    return g


REACH = 0.05                    # a class's move counts in full where both solves play the line with at least this
                                # share of its combos, less below (none where either never reaches the decision:
                                # an unreached class's strategy is arbitrary, the 3bettor's 72o facing a 4bet)


def decisions(g) -> dict:
    """{line_key: ({action: per-class strategy}, per-class reach)} of a solved game's chart decisions ('all-in' as
    'raise'; the reach: the share of each class's combos the player to act brings to the decision, the product of
    its own actions' probabilities along the line, walked per decision so no (nodes, seats, classes) array is held:
    the 8-handed game's would be ~1 GB)."""
    out = {}
    avg = g.average()
    for k in g.dec:
        nd = g.nodes_all[k]
        key = PF.node_key(g.positions[nd.player], nd.line)
        if key is None:
            continue
        s = avg[g.dec_of[k]]
        acts = {}
        for a, nm in enumerate(nd.names):
            nm = "raise" if nm == "all-in" else nm
            acts[nm] = acts.get(nm, 0.0) + s[a]
        r = np.ones(len(s[0]))
        c = k
        while g.parent[c] >= 0:
            par = g.parent[c]
            if g.nodes_all[par].player == nd.player:
                r = r * avg[g.dec_of[par], g.act[c]]
            c = par
        out[key] = (acts, r)
    return out


def moves(dg: dict, db: dict, positions, dim: str) -> dict:
    """The moves of a game (its decisions(), dg, at `positions`) against its base game's (db): per chart decision,
    the class strategy minus the base's at the decision whose charts it plays (the same seats for depth, ante and
    limps; preflop.chart_key's for players and straddle), for raise and call (fold is what's left), weighted by how
    surely both solves reach it (REACH)."""
    from gto.preflop import Table
    out = {}
    table = Table(list(positions), 0.0, dim == "straddle", None)
    for key, (acts, rg) in dg.items():
        hero, seq = key.split("|")
        actions = [{"pos": x.split(":")[0], "action": "raise"} for x in seq.split(",") if x]
        bkey = key if dim in ("depth", "ante", "hu_depth", "limp") else PF.analog(hero, actions, table)
        if bkey is None or bkey not in db:
            continue
        b, rb = db[bkey]
        w = np.clip(np.minimum(rg, rb) / REACH, 0.0, 1.0)
        mv = {}
        for a in ("raise", "call"):
            if a in acts or a in b:
                mv[a] = w * (np.asarray(acts.get(a, 0.0)) - np.asarray(b.get(a, 0.0)))
        out[key] = mv
    return out


def game_moves(d: str, tag: str, base: str, dim: str) -> dict:
    """moves() of two solved games, one loaded at a time."""
    g = solved(d, tag)
    dg, positions = decisions(g), g.positions
    del g
    gc.collect()
    b = solved(d, base)
    db = decisions(b)
    del b
    gc.collect()
    return moves(dg, db, positions, dim)


def write(d: str, out: str, keep=()):
    """The moves of every game in SETS solved in d (and of those named in `keep`, (dimension, sub-key), unsolved there:
    their entries in the module now at `out`, as they are) to `out`."""
    lines = ['"""Generated by scripts/build_pf_formats.py: the preflop charts\' moves in other games (gto/pf_formats.py).',
             'DELTAS[dimension][sub-key] = (index, data): per decision and action, each class\'s (pf_classes.CLASSES)',
             'move from the base game\'s solve to this game\'s, in 1% steps (pf_formats.encode)."""', "DELTAS = {"]
    by = {}
    current = {}
    if keep and os.path.exists(out):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_current_moves", out)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        current = mod.DELTAS
    for dim, sub, tag, base in SETS:
        if not (os.path.exists(path(d, tag)) and os.path.exists(path(d, base))):
            if sub in current.get(dim, {}) and (dim, sub) in keep:
                by.setdefault(dim, {})[sub] = current[dim][sub]
                print(f"{dim} {sub:g}: {tag or '6max_100bb'} or {base or '6max_100bb'} not solved, KEPT from {out}")
            else:
                print(f"{dim} {sub:g}: {tag or '6max_100bb'} or {base or '6max_100bb'} not solved, skipped")
            continue
        mv = game_moves(d, tag, base, dim)
        by.setdefault(dim, {})[sub] = PF.encode(mv)
        print(f"{dim} {sub:g}: {len(mv)} decisions, mean |move| "
              f"{np.mean([np.abs(v).mean() for m in mv.values() for v in m.values()]):.3f}", flush=True)
    for dim, subs in by.items():
        lines.append(f"    {dim!r}: {{")
        for sub, (index, data) in subs.items():
            lines.append(f"        {sub!r}: ({index!r},")
            lines += [f"            {data[i:i + 100]!r}" for i in range(0, len(data), 100)] or ["            ''"]
            lines[-1] += "),"
        lines.append("    },")
    lines.append("}")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out} ({os.path.getsize(out) / 1e3:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", default="", help="comma separated tags ('6max' for the 6-max 100bb game)")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "gto", "pf_format_data.py"))
    ap.add_argument("--keep", default="", help="dimension:sub-key,... to keep from --out's module where unsolved")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    for tag in [t for t in a.solve.split(",") if t]:
        tag = "" if tag == "6max" else tag
        p = path(a.dir, tag)
        if os.path.exists(p):
            print(f"{tag or '6max_100bb'}: solved already", flush=True)
            continue
        g = game(tag)
        t0 = time.time()
        g.solve(a.iters)
        g.save(p)
        print(f"{tag or '6max_100bb'}: {len(g.dec)} decisions, {a.iters} iterations in {time.time() - t0:.0f}s, "
              f"exploitability {g.exploitability():.4f} bb/hand", flush=True)
    if a.write:
        keep = [(k.split(":")[0], float(k.split(":")[1])) for k in a.keep.split(",") if k]
        write(a.dir, a.out, keep)


if __name__ == "__main__":
    main()
