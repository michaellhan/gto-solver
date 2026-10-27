"""Builds the compiled CFR core's browser module, gto/core.wasm, from core/gtocore.c (gto/core.py; the native library
is built from the same source on first use). It needs clang with the wasm32 target and wasm-ld: on a Mac Homebrew's
llvm and lld (brew install llvm lld), found there or through $WASM_CC. The module carries the source's sha1
(core_source), which tests/test_core.py checks against the source, so a change to the C needs this run again:

    python scripts/build_core.py         # prints the module's size and the source's sha1
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gto import core  # noqa: E402

OUT = os.path.join(ROOT, "gto", "core.wasm")


def clang() -> str:
    for c in (os.environ.get("WASM_CC"), "/opt/homebrew/opt/llvm/bin/clang", "/usr/local/opt/llvm/bin/clang",
              shutil.which("clang")):
        if c and os.path.exists(c):
            return c
    sys.exit("no clang: install llvm and lld (wasm-ld), or set WASM_CC")


def main():
    sha = core.source_hash()
    env = dict(os.environ)
    # wasm-ld: Homebrew's lld is a formula of its own, beside llvm's clang
    env["PATH"] = os.pathsep.join(p for p in ("/opt/homebrew/opt/lld/bin", "/usr/local/opt/lld/bin", env.get("PATH", "")) if p)
    tmp = OUT + ".tmp"
    # -msimd128: 128-bit SIMD, in every current browser (Chrome 91, Firefox 89, Safari 16.4); where it isn't the module
    # fails to load and the solver runs in numpy
    subprocess.run([clang(), "--target=wasm32", "-O3", "-msimd128", "-nostdlib", f'-DCORE_SOURCE="{sha}"',
                    "-Wl,--no-entry", "-Wl,--strip-all", "-o", tmp, core.SRC], check=True, env=env)
    os.replace(tmp, OUT)
    print(f"{OUT}: {os.path.getsize(OUT)} bytes, source {sha}")


if __name__ == "__main__":
    main()
