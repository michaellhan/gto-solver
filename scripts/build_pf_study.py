"""Solves the preflop games Study answers at once (gto/pf_study.py STORED) and writes each one's average strategy to
gto/static/pf/<key>.bin (pf_study.pack: ~300 KB).

    python scripts/build_pf_study.py [--keys 6max_100bb,6max_100bb_r5c3] [--iters 1500] [--out gto/static/pf]

One process, one thread (BLAS held to one by gto.native); ~3 minutes a 6-max game natively. Prints each game's
exploitability (mbb/hand a seat) every 250 iterations and of the stored (quantized) strategy at the end.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gto.native  # noqa: F401,E402
from gto import pf_study as PS  # noqa: E402

SPECS = {"6max_100bb": PS.Spec(), "6max_100bb_r5c3": PS.Spec(rake=(0.05, 3.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default=",".join(PS.STORED))
    ap.add_argument("--iters", type=int, default=PS.STORED_ITERS)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "gto", "static", "pf"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for key in a.keys.split(","):
        spec = SPECS[key]
        assert spec.key == key, (spec.key, key)
        g = spec.game()
        t0 = time.time()
        for t in range(1, a.iters + 1):
            g.iterate(t)
            if t % 250 == 0:
                e = g.exploitability(per_seat=True)
                print(f"{key} {t}: {1000 * sum(e) / len(e):.2f} mbb/hand a seat, {time.time() - t0:.0f} s", flush=True)
        s = PS.Solve(spec, PS.pack(g))
        s.measure()
        blob = PS.pack(g, s.mbb)
        with open(os.path.join(a.out, key + ".bin"), "wb") as f:
            f.write(blob)
        print(f"{key}: {len(blob)} bytes, stored strategy {s.mbb:.2f} mbb/hand a seat", flush=True)


if __name__ == "__main__":
    main()
