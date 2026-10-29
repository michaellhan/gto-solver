"""Per-class flop values read from the flop library, for the preflop game (gto/pf_game.py, LibraryModel).

For every spot of a library directory (its index.json) and every stored flop, an entry holds each combo's EV per
action at the stored flop decisions (flop_library.encode: chips relative to folding there, the pot's share as the
flop starts, later bets included; no rake). Per combo this takes:
  * out of position: its root value, sum over actions of strategy x EV;
  * in position: its value after each of the root actions, weighted by how often the out-of-position range takes
    that action against this combo (its card removal): after a check or a lead the in-position player's stored
    node (strategy x EV); a lead the library didn't store (STORE_LINE: the out-of-position player takes it with
    under 5% of its range) is valued as the check is, shifted so the two players' values there add up to the pot
    (the game is constant-sum without rake).
Each combo's share of the pot is weighted by how often it meets this flop (the flop's weight among all 22100, times
the opponent's range mass that doesn't collide with the combo or the flop: the deal's own probability), and summed
per hand class. The values are of the ranges the library was solved for (the charts' ranges entering the flop);
a class outside a range has none.

    python scripts/pf_library_values.py [--lib DIR ...] [--spots a,b] [--out gto/pf_libvals.py]

Several --lib directories: a spot is read from the first that holds the most flops of it. The output module holds
per spot its seats, pot, stack, flop count, and per player the class shares (the deal-weighted mean pot share), the
class deal weights and the class weights of the range entering the flop.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import gto.native  # noqa: F401,E402  (one BLAS thread)
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto.cards import cards as parse_cards  # noqa: E402
from gto.pf_classes import CLASS_OF, COMPAT, to_classes  # noqa: E402
from gto.solver import MEMBER_FULL  # noqa: E402

K = int(CLASS_OF.max()) + 1


def entry_values(e: dict, full0: np.ndarray, full1: np.ndarray, pot: float):
    """(per player: combo indices, pot share per combo, deal weight per combo, zero-sum gap) of one decoded entry."""
    board = parse_cards(e["head"]["board"])
    bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
    w = [full0 * bm, full1 * bm]
    idx = [np.nonzero(m)[0] for m in e["masks"]]
    nodes = {tuple(nd["path"]): nd for nd in e["nodes"]}
    root = nodes[()]
    assert root["player"] == 0
    v0 = (root["strat"] * root["evs"]).sum(0)
    C = COMPAT[np.ix_(idx[1], idx[0])].astype(np.float64)       # (ip combos, oop combos)
    w0, w1 = w[0][idx[0]], w[1][idx[1]]
    den1 = C @ w0                                                 # ip combo: the oop mass it can meet
    cs0 = C.T @ w1                                                # oop combo: the ip mass it can meet
    v1 = np.zeros(len(idx[1]))
    stored, missing = {}, []
    for a, nm in enumerate(root["names"]):
        pa = (C @ (w0 * root["strat"][a])) / np.maximum(den1, 1e-300)
        child = nodes.get((nm,))
        if child is not None and child["player"] == 1:
            stored[nm] = (child["strat"] * child["evs"]).sum(0)
            v1 += pa * stored[nm]
        else:
            missing.append((a, pa))
    base = stored.get("check", next(iter(stored.values()), None))
    for a, pa in missing:
        # an action whose ip node isn't stored: ip's value there is the check's (else another stored node's), shifted
        # so both players' values over the pairs that meet there add up to the pot (oop's is the root's EV of it)
        m = w0 * root["strat"][a]
        pairs = C * m[None, :] * w1[:, None]
        tot = pairs.sum()
        if tot < 1e-12:
            continue
        mean0 = float((pairs.sum(0) * root["evs"][a]).sum() / tot)
        b = base if base is not None else np.zeros(len(idx[1]))
        mean1 = float((pairs.sum(1) * b).sum() / tot)
        v1 += pa * (b + (pot - mean0 - mean1))
    # the two values over every pair that meets add up to the pot (no rake): a check of the decoding
    P = C * w0[None, :] * w1[:, None]
    gap = float(((P.sum(0) * v0).sum() + (P.sum(1) * v1).sum()) / P.sum() - pot)
    return [(idx[0], v0 / pot, cs0), (idx[1], v1 / pot, den1)], gap


def flop_set(flops) -> list:
    """The flops to read: the largest representative set (1755, 552 or 184: each holds the next) the spot has whole, so
    a spot still building is read on the flops it has finished of a set (weighted as that set), else all it has."""
    have = set(flops)
    for n in (1755, 552, 184):
        names = [fl.cards_str(f) for f in fl.representative_flops(n)]
        if all(x in have for x in names):
            return sorted(names)
    return sorted(flops)


def spot_values(lib: str, spot: dict, ranges: dict, verbose=False):
    full0 = np.asarray(ranges["w"][0], float) / 10000.0
    full1 = np.asarray(ranges["w"][1], float) / 10000.0
    names = flop_set(spot["flops"])
    wts = fl.report_weights(names)
    acc = [np.zeros(K), np.zeros(K)]
    wacc = [np.zeros(K), np.zeros(K)]
    gaps = []
    t0 = time.time()
    for nm in names:
        with open(os.path.join(lib, spot["id"], nm + ".bin"), "rb") as f:
            e = fl.decode(f.read())
        per, gap = entry_values(e, full0, full1, float(spot["pot"]))
        gaps.append(gap)
        for p, (ix, share, dw) in enumerate(per):
            d = wts[nm] * dw * (full0, full1)[p][ix]
            acc[p] += np.bincount(CLASS_OF[ix], weights=d * share, minlength=K)
            wacc[p] += np.bincount(CLASS_OF[ix], weights=d, minlength=K)
    out = {"seats": spot["seats"], "pot": float(spot["pot"]), "stack": float(spot["stack"]), "line": spot.get("line"),
           "flops": len(names), "fp": spot["fp"]}
    for p, side in ((0, "oop"), (1, "ip")):
        out[side] = {"share": np.where(wacc[p] > 0, acc[p] / np.maximum(wacc[p], 1e-300), 0.0),
                     "w": wacc[p] / max(wacc[p].sum(), 1e-300),
                     "range": to_classes(full0 if p == 0 else full1)}
    g = np.abs(gaps)
    if verbose:
        r = [out[s]["w"] @ out[s]["share"] for s in ("oop", "ip")]
        print(f"{spot['id']}: {len(names)} flops in {time.time() - t0:.0f}s; range shares oop {r[0]:.4f} ip {r[1]:.4f} "
              f"(sum {r[0] + r[1]:.4f}); zero-sum gap per entry mean {g.mean():.4f} max {g.max():.4f} bb", flush=True)
    return out


def write_module(vals: dict, path: str, note: str):
    lines = ['"""Generated by scripts/pf_library_values.py: per-class flop values from the flop library, for the',
             'preflop game (gto/pf_game.py LibraryModel). Per spot: seats [oop, ip], pot, stack (behind), the flops read,',
             'and per player (oop, ip) SHARE[i]: hand class i\'s mean share of the pot as the flop starts (later bets',
             'included, no rake; weighted by how often each combo meets each flop), W[i]: the class\'s share of those',
             'deals, RANGE[i]: the class\'s combos in the range entering the flop (pf_classes.CLASSES order).',
             note + '"""', "SPOTS = {"]
    for sid in vals:
        v = vals[sid]
        lines.append(f"    {sid!r}: {{")
        lines.append(f"        'seats': {list(v['seats'])!r}, 'pot': {v['pot']!r}, 'stack': {v['stack']!r}, "
                     f"'flops': {v['flops']!r}, 'line': {v['line']!r},")
        for side in ("oop", "ip"):
            s = v[side]
            lines.append(f"        {side!r}: {{")
            lines.append(f"            'share': {[round(float(x), 5) for x in s['share']]!r},")
            lines.append(f"            'w': {[round(float(x), 7) for x in s['w']]!r},")
            lines.append(f"            'range': {[round(float(x), 4) for x in s['range']]!r},")
            lines.append("        },")
        lines.append("    },")
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", default=None, help="library directory (repeatable)")
    ap.add_argument("--spots", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "gto", "pf_libvals.py"))
    a = ap.parse_args()
    libs = a.lib or [fl.DIR]
    want = [x for x in a.spots.split(",") if x]
    best = {}
    for lib in libs:
        try:
            idx = json.load(open(os.path.join(lib, "index.json")))
        except (OSError, ValueError):
            continue
        for sp in idx.get("spots", []):
            if want and sp["id"] not in want:
                continue
            if sp["id"] not in best or len(flop_set(sp["flops"])) > len(flop_set(best[sp["id"]][1]["flops"])):
                best[sp["id"]] = (lib, sp)
    vals = {}
    for sid, (lib, sp) in best.items():
        ranges = json.load(open(os.path.join(lib, sid, "ranges.json")))
        assert ranges["fp"] == sp["fp"], sid
        vals[sid] = spot_values(lib, sp, ranges, verbose=True)
    order = [s[0] for s in fl.SPOTS if s[0] in vals] + sorted(s for s in vals if s not in {x[0] for x in fl.SPOTS})
    write_module({s: vals[s] for s in order}, a.out,
                 "Read from " + ", ".join(f"{s} ({vals[s]['flops']} flops)" for s in order) + ".")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
