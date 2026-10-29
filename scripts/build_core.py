"""Builds the compiled CFR core's browser modules from core/gtocore.c (gto/core.py; the native library is built from
the same source on first use): gto/core.wasm, and gto/core-mt.wasm, the same with threads (shared memory, which
worker.js gives it and its helper workers when the page is cross-origin isolated: gto/static/sw.js); and
gto/mwcore.wasm, the multiway core (core/mwcore.c, gto/mwcore.py). It needs clang with the wasm32 target and wasm-ld: on a Mac Homebrew's
llvm and lld (brew install llvm lld), found there or through $WASM_CC. The module carries the source's sha1
(core_source), which tests/test_core.py checks against the source, so a change to the C needs this run again:

    python scripts/build_core.py         # prints the modules' sizes and the source's sha1
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gto import core, mwcore  # noqa: E402

OUT = os.path.join(ROOT, "gto", "core.wasm")
OUT_MT = os.path.join(ROOT, "gto", "core-mt.wasm")
# the threaded module's memory is worker.js's (imported, shared: SharedArrayBuffer) and at most this big (a shared
# memory declares its maximum; a browser reserves it, it isn't used until the arena grows into it)
MAX_MT = 2 << 30


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
    cc = [clang(), "--target=wasm32", "-O3", "-msimd128", "-nostdlib", f'-DCORE_SOURCE="{sha}"', "-Wl,--no-entry",
          "-Wl,--strip-all"]
    subprocess.run(cc + ["-o", tmp, core.SRC], check=True, env=env)
    os.replace(tmp, OUT)
    # threads: wasm atomics (Chrome 74, Firefox 79, Safari 14.1); the helpers set their own stack pointer (core_stack)
    tmp = OUT_MT + ".tmp"
    subprocess.run(cc + ["-matomics", "-mbulk-memory", "-Wl,--import-memory", "-Wl,--shared-memory",
                         f"-Wl,--max-memory={MAX_MT}", "-Wl,--export=__stack_pointer", "-o", tmp, core.SRC],
                   check=True, env=env)
    os.replace(tmp, OUT_MT)
    for p in (OUT, OUT_MT):
        print(f"{p}: {os.path.getsize(p)} bytes, source {sha}")
    # the multiway core (gto/mwcore.py): one thread; bulk memory for the copies and fills clang emits
    msha = mwcore.source_hash()
    tmp = mwcore.WASM + ".tmp"
    subprocess.run(cc[:5] + ["-mbulk-memory", f'-DCORE_SOURCE="{msha}"'] + cc[6:] + ["-o", tmp, mwcore.SRC],
                   check=True, env=env)
    os.replace(tmp, mwcore.WASM)
    print(f"{mwcore.WASM}: {os.path.getsize(mwcore.WASM)} bytes, source {msha}")


if __name__ == "__main__":
    main()
