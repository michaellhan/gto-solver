"""The browser solver's Python code in one file: solver.zip, which gto/static/worker.js fetches (alongside Pyodide, not
after it) and unpacks into Pyodide's file system. It holds every gto/*.py (gto/__init__.py too: inside a zip the leading
underscore GitHub Pages won't serve doesn't matter), the compiled CFR core (gto/core.wasm, scripts/build_core.py) and
treys, the hand evaluator's lookup tables (pure Python, MIT licensed: its LICENSE goes along), taken from this
interpreter's install, so the page needs neither micropip nor PyPI.
The zip is deterministic (sorted names, fixed timestamps): the same sources give the same bytes and hash, which
scripts/deploy.sh puts in the worker's version.

    python scripts/build_bundle.py OUT.zip      # prints the zip's sha1
"""
from __future__ import annotations
import glob
import hashlib
import os
import sys
import zipfile
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "solver.zip"   # worker.js fetches this name


def bundle_files() -> List[Tuple[str, str]]:
    """(name in the zip, source path), sorted by name."""
    import treys
    out = [(f"gto/{os.path.basename(p)}", p) for p in glob.glob(os.path.join(ROOT, "gto", "*.py"))]
    # the compiled CFR core's modules (gto/core.py), which worker.js instantiates from the unpacked bundle (core-mt.wasm,
    # with threads, when the page is cross-origin isolated), and the multiway core's (gto/mwcore.py)
    for name in ("core.wasm", "core-mt.wasm", "mwcore.wasm", "mwcore-mt.wasm"):
        wasm = os.path.join(ROOT, "gto", name)
        if os.path.exists(wasm):
            out.append((f"gto/{name}", wasm))
    tdir = os.path.dirname(treys.__file__)
    out += [(f"treys/{os.path.basename(p)}", p) for p in glob.glob(os.path.join(tdir, "*.py"))]
    lic = glob.glob(os.path.join(os.path.dirname(tdir), "treys-*.dist-info", "LICENSE*"))
    if lic:
        out.append(("treys/LICENSE", lic[0]))
    return sorted(out)


def build(path: str) -> str:
    """Writes the zip to path; returns its sha1."""
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, src in bundle_files():
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(src, "rb") as f:
                z.writestr(info, f.read(), compresslevel=9)
    os.replace(tmp, path)
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(build(sys.argv[1]))
