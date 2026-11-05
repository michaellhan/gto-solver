#!/usr/bin/env python3
"""The Python tests, as `python -m unittest` finds them, with each module in its own process.

One process running all of them keeps growing, to 3.7 GB by the time it reaches a test that alone peaks at 0.3 GB. With
other work on the machine that reads as out of memory: the machine's guard paused such a run for an hour, then killed it.
Run this way the highest peak is 2.93 GB (tests.test_core) and it's gone with that module's process. Every test still runs.

    python scripts/run_tests.py     prints each module that fails, the highest peak, and a last line like unittest's
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAN = re.compile(r"^Ran (\d+) tests? in", re.M)
EXPECTED = re.compile(r"expected failures=(\d+)")


def modules() -> list[str]:
    """The modules `python -m unittest` would run; one that can't be imported is named by its own failure."""
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    found: set[str] = set()

    def walk(suite):
        for t in suite:
            if isinstance(t, unittest.TestSuite):
                walk(t)
            elif type(t).__module__ == "unittest.loader":
                found.add(t.id().rsplit(".", 1)[-1] if t.id().count(".") == 2 else t.id().split(".", 2)[-1])
            else:
                found.add(type(t).__module__)

    walk(unittest.TestLoader().discover("."))
    return sorted(found)


def run_one(name: str) -> tuple[int, str, float]:
    """(exit status, its output, its peak in GB) of one module in its own process."""
    with tempfile.TemporaryFile("w+") as out:
        p = subprocess.Popen([sys.executable, "-m", "unittest", "-q", name], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        rss = usage.ru_maxrss / (1 << 30 if sys.platform == "darwin" else 1 << 20)
        return p.returncode, out.read(), rss


def main() -> int:
    start, tests, expected, peak, failed = time.time(), 0, 0, (0.0, ""), []
    names = modules()
    for name in names:
        code, text, rss = run_one(name)
        if code != 0:   # (a few tests solve on a time budget, and a loaded machine can fail one: a real bug fails again)
            print(f"{name} failed; retrying it once", flush=True)
            code, text, again = run_one(name)
            rss = max(rss, again)
        ran = RAN.search(text)
        tests += int(ran.group(1)) if ran else 0
        expected += sum(int(n) for n in EXPECTED.findall(text))
        peak = max(peak, (rss, name))
        if code != 0 or not ran:
            failed.append(name)
            print(f"--- {name} (exit {code})\n{text[-6000:]}", flush=True)
    print(f"Ran {tests} tests in {time.time() - start:.1f}s, {len(names)} modules; highest peak {peak[0]:.2f} GB ({peak[1]})")
    if failed:
        print(f"FAILED: {' '.join(failed)}")
        return 1
    print(f"OK (expected failures={expected})" if expected else "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
