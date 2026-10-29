"""The advice in other games (gto/pf_formats.py's charts) against published solver frequencies
(tests/preflop_refs.py FORMAT_REFS: 8-max 100bb opens, 40bb MTT and cash spots).

    python scripts/preflop_format_reference.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tests.preflop_refs import FORMAT_REFS, measure_format, misses


def main():
    errs, n_miss = [], 0
    for r in FORMAT_REFS:
        got = measure_format(r)
        bad = misses(r, got)
        cell = lambda k: f"{got.get(k, 0) * 100:5.1f}/{r['want'][k] * 100:5.1f}" if k in r["want"] else " " * 11
        print(f"{r['name']:40s} fold {cell('fold')}  call {cell('call')}  raise {cell('raise')}  "
              f"±{r['tol'] * 100:.0f} {'MISS' if bad else 'ok'}{'?' if r['uncertain'] else ''}")
        if not r["uncertain"]:
            errs += [abs(got.get(k, 0) - v) for k, v in r["want"].items()]
            n_miss += bool(bad)
    print(f"certain references: {sum(not r['uncertain'] for r in FORMAT_REFS)}, outside tolerance {n_miss}, "
          f"mean abs error {sum(errs) / max(len(errs), 1) * 100:.2f} points")


if __name__ == "__main__":
    main()
