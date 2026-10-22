"""The charts against published 6-max 100bb cash preflop solver frequencies (tests/preflop_refs.py).

    python scripts/preflop_reference.py [-v]     # -v: sources too

Prints each spot's fold/call/raise shares, chart against published, and the mean absolute error over
the certain references (points per share).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tests.preflop_refs import REFS, HANDS, RAKED_HANDS, measure, measure_hand, measure_raked_hand, misses


def main():
    errs, n_miss = [], 0
    for r in REFS:
        got = measure(r)
        bad = misses(r, got)
        cell = lambda k: f"{got.get(k, 0) * 100:5.1f}/{r['want'][k] * 100:5.1f}" if k in r["want"] else " " * 11
        flag = ("MISS" if bad else "ok") + ("?" if r["uncertain"] else "")
        print(f"{r['name']:34s} {r['of']:5s} fold {cell('fold')}  call {cell('call')}  raise {cell('raise')}  "
              f"±{r['tol'] * 100:.0f} {flag}")
        if "-v" in sys.argv:
            print(f"    {r['source']}" + (f"; {r['note']}" if r["note"] else ""))
        if not r["uncertain"]:
            errs += [abs(got.get(k, 0) - v) for k, v in r["want"].items()]
            n_miss += bool(bad)
    print(f"certain references: {sum(not r['uncertain'] for r in REFS)}, outside tolerance {n_miss}, "
          f"mean abs error {sum(errs) / max(len(errs), 1) * 100:.2f} points")
    n_bad = 0
    for h in HANDS:
        v = measure_hand(h)
        ok = h[5] - 1e-9 <= v <= h[6] + 1e-9
        n_bad += not ok
        print(f"  {h[0]:36s} {h[4]} {v * 100:5.1f}% (want {h[5] * 100:.0f}-{h[6] * 100:.0f}) {'ok' if ok else 'MISS'}"
              + (f"  {h[7]}" if "-v" in sys.argv else ""))
    print(f"hand facts: {len(HANDS)}, outside range {n_bad}")
    n_bad = 0
    for h in RAKED_HANDS:
        v = measure_raked_hand(h)
        ok = h[5] - 1e-9 <= v <= h[6] + 1e-9
        n_bad += not ok
        print(f"  {h[0]:36s} {h[4]} {v * 100:5.1f}% (want {h[5] * 100:.0f}-{h[6] * 100:.0f}) {'ok' if ok else 'MISS'}"
              + (f"  {h[7]}" if "-v" in sys.argv else ""))
    print(f"raked hand facts (at their rake, through the advice): {len(RAKED_HANDS)}, outside range {n_bad}")


if __name__ == "__main__":
    main()
