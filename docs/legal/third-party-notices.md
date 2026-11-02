# DRAFT: Third-party Notices

> **DRAFT for review by a lawyer. Not published, not linked from the site.**
> Written from `.work/monetize/paywall.md` (section 5) and from what the site ships today (November 2026). Text in
> [brackets] is still to fill in. Re-check this list whenever the shipped files or their versions change.

GTO Solver uses the following third-party software. Each is the work of its own authors and is used under its own
licence, which these terms don't change.

## Shipped with the site

### treys 0.1.8 (MIT)

The hand evaluator's lookup tables. Its Python files are included in the solver bundle (`solver.zip`), together with
its licence file (`treys/LICENSE`), which the build adds (`scripts/build_bundle.py`).

```
Copyright (c) 2013 Will Drevo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Loaded by your browser from a public CDN (not redistributed by us)

### Pyodide 0.26.4 (Mozilla Public License 2.0)

The Python runtime compiled to WebAssembly that the solver runs on, loaded unmodified from
https://cdn.jsdelivr.net/pyodide/v0.26.4/full/. Source: https://github.com/pyodide/pyodide. Licence:
https://www.mozilla.org/en-US/MPL/2.0/. Pyodide includes CPython (Python Software Foundation License) and other
components under their own licences, listed in its distribution.

### NumPy 1.26.4 (BSD 3-Clause)

Loaded as Pyodide's NumPy package from the same CDN. Copyright (c) 2005-2023, NumPy Developers. Licence:
https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt. NumPy's wheel bundles further components under their own
licences, listed in that file.

## Services

- **Anthropic Claude API**: writes the "Explain with AI" explanations (a service, not software we distribute).
- **Cloudflare Workers**: runs the explanation proxy.
- **GitHub Pages** and **jsDelivr**: host the site and the runtime.

## Not shipped: development tools only

These are used to build or test the project and are not part of what the site serves:

- **postflop-solver** by b-inary (AGPL-3.0), used only by the benchmark harness `scripts/pfbench`. Keep it out of
  anything shipped: shipping it, or code derived from it, would bring the AGPL's obligations with it.
- **wrangler** (Cloudflare, MIT OR Apache-2.0 [confirm]), the deploy tool for the explanation proxy.
- **clang / LLVM** (Apache-2.0 with LLVM exceptions), which compiles the solver core to WebAssembly. The core is our
  own C code built without a C library (`-nostdlib`), so no runtime library is bundled into the modules.

## Our own code

The solver, the page and the precomputed data are Copyright (c) 2026 [Michael Han]. All rights reserved
[or the licence chosen in paywall.md section 5]. [Add the same copyright line and a NOTICE file inside solver.zip.]
