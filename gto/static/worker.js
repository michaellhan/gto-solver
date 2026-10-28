// Runs the Python solver in the browser with Pyodide. Used when no local server is present.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const T0 = performance.now();

// The solver's Python code (every gto/*.py) and treys (the evaluator's lookup tables) in one zip, built by
// scripts/build_bundle.py: no micropip, no PyPI, one request instead of 16 in a row. Its hash is in the worker's
// version (?v=, scripts/deploy.sh), so a new bundle is a new URL; it's requested before Pyodide's script even loads and
// kept in the Cache API under that URL, so a later visit reads it without a request (GitHub Pages' HTTP cache would
// revalidate it after 10 minutes). Beside it the Cache API keeps the bytecode the first visit compiled (solver-pyc.zip,
// made here, never requested): compiling the startup modules took ~0.15 s of their import.
const VERSION = new URLSearchParams(self.location ? self.location.search : "").get("v") || "dev";
const BUNDLE = `solver.zip?v=${VERSION}`, PYC = `solver-pyc.zip?v=${VERSION}`;
// Only a deployed version (deploy.sh's 10-hex-digit hash) is cached: a copy served without one could change under it.
const cacheP = (async () => {
  if (!/^[0-9a-f]{10}$/.test(VERSION)) return null;
  try { return await caches.open("gto-solver"); } catch (e) { return null; }   // none: insecure origins, some private modes
})();
async function fromCache(key) {
  try { const c = await cacheP, hit = c && await c.match(key); return hit ? await hit.arrayBuffer() : null; } catch (e) { return null; }
}
async function toCache(key, buf) {   // replaces the versions before this one
  try {
    const c = await cacheP; if (!c) return;
    const name = key.split("?")[0];
    for (const k of await c.keys()) if (new URL(k.url).pathname.endsWith("/" + name)) await c.delete(k);
    await c.put(key, new Response(buf, { headers: { "Content-Type": "application/zip" } }));
  } catch (e) {}
}
const bundle = (async () => {
  const [hit, pyc] = await Promise.all([fromCache(BUNDLE), fromCache(PYC)]);
  if (hit) return { buf: hit, pyc, cached: true };
  const r = await fetch(BUNDLE);
  if (!r.ok) throw new Error(`could not load the solver code (${BUNDLE}: ${r.status})`);
  const buf = await r.arrayBuffer();
  toCache(BUNDLE, buf.slice(0));
  return { buf, pyc: null, cached: false };
})();
bundle.catch(() => {});   // awaited (and reported) in the setup below
// numpy's wheel (12 MB, the biggest download) is requested now, alongside the runtime, so that loadPackage finds it in
// the HTTP cache (jsdelivr caches versioned files for a year) rather than starting it once the runtime is up. The name
// is pyodide-lock.json's for 0.26.4; were it wrong, this request would just fail and loadPackage fetch numpy itself.
fetch(PYODIDE + "numpy-1.26.4-cp312-cp312-pyodide_2024_0_wasm32.whl").then((r) => r.arrayBuffer()).catch(() => {});
importScripts(PYODIDE + "pyodide.js");

// The precomputed flop library (gto/flop_library.py): the Python side asks for its index once and for an entry
// (<spot>/<flop>.bin, 1.5-6 KB) only when a flop spot it holds comes up. A synchronous request (allowed in a
// worker) because the solver runs synchronously; a missing or unreadable file is just no library.
self.libFetch = (path) => {
  try {
    const x = new XMLHttpRequest();
    x.open("GET", `flop_library/${path}`, false);
    x.responseType = "arraybuffer";
    x.send();
    return x.status === 200 ? new Uint8Array(x.response) : null;
  } catch (e) {
    return null;
  }
};

// The core's threads: core-mt.wasm with its memory shared (a SharedArrayBuffer: the page must be cross-origin isolated,
// which sw.js arranges) and up to 7 helper workers, each the same module on the same memory waiting in core_helper for
// rows of a chance node's subtree (core/gtocore.c); a solve's numbers are the same on any number of threads. Without
// isolation, or where the threaded module won't load, core.wasm alone.
const HELPERS = Math.min(7, ((self.navigator && navigator.hardwareConcurrency) || 1) - 1);
const HELPER = `onmessage = async (e) => {
  const { mod, memory, i } = e.data;
  const ex = (await WebAssembly.instantiate(mod, { env: { memory } })).exports;
  ex.__stack_pointer.value = ex.core_stack(i);   // (every instance starts on the same stack)
  ex.core_helper(i);                              // runs rows of the solves' chance nodes; never returns
};`;
async function loadCore(py) {
  if (self.crossOriginIsolated && HELPERS > 0) {
    try {
      const mod = await WebAssembly.compile(py.FS.readFile("/app/gto/core-mt.wasm"));
      const memory = new WebAssembly.Memory({ initial: 256, maximum: 32768, shared: true });
      const ex = Object.assign({}, (await WebAssembly.instantiate(mod, { env: { memory } })).exports, { memory });
      const url = URL.createObjectURL(new Blob([HELPER], { type: "text/javascript" }));
      for (let i = 0; i < HELPERS; i++) new Worker(url).postMessage({ mod, memory, i });
      ex.core_threads(HELPERS + 1);
      return ex;
    } catch (e) {}
  }
  try {
    return (await WebAssembly.instantiate(py.FS.readFile("/app/gto/core.wasm"))).instance.exports;
  } catch (e) {
    return null;
  }
}

// The setup, in the stages the page shows; their seconds come back with "Solving…" (setup: for
// scripts/bench_browser.mjs and the console).
const ready = (async () => {
  const setup = {}; let t = T0;
  const lap = (k) => { const n = performance.now(); setup[k] = +((n - t) / 1000).toFixed(2); t = n; };
  postMessage({ status: "Loading the solver runtime (1 of 3: Python and numpy, about 25 MB, cached after the first visit)…" });
  const py = await loadPyodide({ indexURL: PYODIDE });
  lap("runtime");
  postMessage({ status: "Loading numpy (2 of 3)…" });
  await py.loadPackage(["numpy"], { messageCallback: () => {} });
  lap("numpy");
  postMessage({ status: "Starting the solver (3 of 3)…" });
  const { buf, pyc, cached } = await bundle;
  Object.assign(setup, { bundle_cached: cached, pyc_cached: !!pyc });
  py.unpackArchive(buf, "zip", { extractDir: "/app" });
  if (pyc) py.unpackArchive(pyc, "zip", { extractDir: "/app" });
  // the compiled CFR core (gto/core.py; in the bundle, ~25 KB): turn, river and flop solves run their iterations in
  // it. Where it can't load (a browser without wasm SIMD) the solver runs in numpy as before.
  self.gtoCore = await loadCore(py);
  setup.core = !!self.gtoCore;
  setup.threads = self.gtoCore ? self.gtoCore.core_threads(0) : 0;
  // the sources get a fixed mtime (a .pyc is valid while its source's mtime and size match) and the startup imports
  // write their bytecode (Pyodide turns that off), which goes to the Cache API once the first request has its answer
  py.runPython("import os, sys, json\n"
             + "for d, _, fs in os.walk('/app'):\n"
             + "    for f in fs:\n"
             + "        if f.endswith('.py'): os.utime(os.path.join(d, f), (1577836800, 1577836800))\n"
             + "sys.path.insert(0, '/app'); sys.dont_write_bytecode = False\n"
             + "import gto.advisor, gto.solver, gto.flop_library\n"
             + "sys.dont_write_bytecode = True");
  if (!pyc) setTimeout(() => {   // after the requests waiting on the setup (they continue before any timer)
    try {
      py.runPython("import zipfile\n"
                 + "with zipfile.ZipFile('/tmp/pyc.zip', 'w', zipfile.ZIP_DEFLATED) as z:\n"
                 + "    for d, _, fs in os.walk('/app'):\n"
                 + "        for f in fs:\n"
                 + "            if f.endswith('.pyc'): z.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), '/app'))");
      toCache(PYC, py.FS.readFile("/tmp/pyc.zip"));
    } catch (e) {}
  }, 0);
  lap("imports");
  // Pyodide numpy is 10-30x slower per iteration than native on the flop (its leaf products have no BLAS), ~1.5x on the
  // turn (vectorised river showdowns), so the browser plays a smaller game. Turn, river and flop solves run their
  // iterations in the compiled core (gto/core.wasm, above): 5.5-6.7x numpy's iterations on the turn, the river's solve
  // 0.5 s for numpy's 10-13 (scripts/bench_browser.mjs, README Compiled core), a flop 2-16x (6.6x on average:
  // scripts/bench_flop_core.py). Two flop bet sizes; inside a flop solve one turn and river size (Config.later_sizes).
  // From the turn (scripts/turn_river_reference.py: 6 turn spots against every-river solves run to convergence with a
  // 33/75/125% menu; README Accuracy vs speed), the river is one card per class (river_classes: 13 cards on a rainbow
  // turn, 22-24 with a flush suit, each standing for its cards through suit swaps; within ~0.1% of the pot of every
  // river at the same iterations, 2-3x cheaper). The old Fast's 8 fixed rivers settled (~450 iterations) into a game
  // where only those rivers come: per-combo EVs 5-10% of the pot from its own tree with every river.
  // Fast (8 s): a third, three quarters and 125% pot on the turn, the river inside at a third and 125% (gto/native.py):
  //   ~105 iterations in the core on a loaded machine (0.3-0.8% of the pot from equilibrium), per-combo EVs 0.34-0.55%
  //   of the pot from the rich reference on five single-raised turns. Before the core it played 33/125% with the river
  //   at 75%, ~90-150 iterations in numpy; converged in the core (~280) that menu was 0.58-1.48% off (the old 8 rivers:
  //   3.8-9.5; 75% alone, converged: 0.6-3.8).
  // Accurate (60 s): a third, three quarters and 125% on the turn and river, ~180-360 iterations in numpy (now ~450-490
  //   in the core): 0.08-0.33% (one size with every river: 0.6-3.8).
  // A river root bets 33/75/125/250% pot (Advisor.root_sizes): 0.1-0.6% of the pot per combo from a six-size menu,
  //   where 75% alone was 1.2-7.2 (polar rivers overbet half the range).
  // From the flop both play one turn per rank class with the river bet on, the turn and river by hand buckets
  //   (river_buckets, turn_buckets; README, Accuracy vs speed), the river in single-raised pots by 32 buckets of river
  //   strength (river_strength: a draw strong where it hits, weak where it misses). Fast: 32 buckets over every third river, 130-420
  //   iterations in 8 s in numpy, 229-986 in the compiled core on a loaded machine (scripts/bench_browser.mjs,
  //   bench_flop_core.py). Accurate: 64 over every river, 520-870 iterations in 60 s in numpy (0.03-0.05% of the pot
  //   from equilibrium in its game), its 1000 in 16-31 s in the core (0.01%). Against a solve over every turn (12 flops, scripts/bench_solver.py --flop) at 400 iterations
  //   the c-bet is 2.5 / 1.8 points off on average, the size split 4.1 / 2.3, BB's check 1.6 / 1.3. Accurate played
  //   the turn and river per combo until measured here: 54 iterations in 60 s on a rainbow flop (7.5 points of
  //   c-bet, 12.7 of split, 8.5 of check), ~25-30 on a two-tone one (at 25: 14.4, 18.4, 18.4) where, still 10% of
  //   the pot from equilibrium, it took a second budget (132 s for the page's 60, a 590 MB heap). Its earlier-street solves
  //   (12 s, for range narrowing) play its own flop model, so a hand's flop advice narrows its turn without a solve.
  // With 4 or more threads (the core's helpers, above) Fast's turn plays Accurate's tree, the river inside at
  //   33/75/125% and shallow turns too (gto/native.py THREADED_FAST: threads give 2.5-3x the iterations, and past ~200
  //   of Fast's the river's 75% is worth more than the iterations it costs: five single-raised turns 0.12% of the pot
  //   off per combo at 300 iterations, 0.07 at 450, where Fast's own tree settles at 0.34).
  const THREADED = setup.threads >= 4;
  const FAST_RIVER = THREADED ? "[0.33, 0.75, 1.25]" : "[0.33, 1.25]";
  const RS = "[setattr(a, 'root_sizes', {'river': {'bet': [0.33, 0.75, 1.25, 2.5]}}) for a in advs.values()]; ";
  const RAISES = "raise_sizes={'flop': [0.5], 'turn': [0.75], 'river': [0.75]}";
  py.runPython("import sys, json; sys.path.insert(0, '/app'); from gto.advisor import Advisor; from gto.solver import Config; "
             + `advs = {'fast': Advisor(Config(n_turns=3, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4, `
             + `river_buckets=32, turn_buckets=True, bucket_river_step=3, river_classes=True, river_strength=32, `
             + `bet_sizes={'flop': [0.33, 0.75], 'turn': [0.33, 0.75, 1.25], 'river': ${FAST_RIVER}}, ${RAISES})), `
             + `'accurate': Advisor(Config(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8, river_classes=True, `
             + `river_buckets=64, turn_buckets=True, bucket_river_step=1, river_strength=32, `
             + `bet_sizes={'flop': [0.33, 0.75], 'turn': [0.33, 0.75, 1.25], 'river': [0.33, 0.75, 1.25]}, ${RAISES}))}; `
             + RS
             // Accurate picks up from a Fast solve of the same spot (not the reverse: Fast's few seconds can't re-fit an
             // Accurate solve to its own tree; the page shows Accurate's answer instead)
             + "advs['accurate'].warm_from = advs['fast']; "
             // Fast's turn also gets the narrowing flop's 3 s when that flop costs nothing (from the library or the hand's
             // flop advice; Advisor.spare_prior, as natively): the page's bar counts them anyway
             + "advs['fast'].spare_prior = True; "
             // a shallow turn (at most 4.5 x the pot behind: a 3bet pot's) bets 33/75% in Fast, not 33/125% (gto/native.py
             // FAST_SPR_SIZES: four 3bet turns 1.96% of the pot off per combo -> 0.80 at 8 s, frequencies 6.8 points -> 2.0);
             // with threads Accurate's tree there too
             + (THREADED ? "" : "advs['fast'].spr_sizes = {'turn': (4.5, {'bet': [0.33, 0.75], 'river': [0.75]})}; ")
             // both speeds answer a flop spot the library holds from it (Accurate's flop model, solved offline)
             + "from gto.flop_library import FlopLibrary; from js import libFetch; "
             + "lib = FlopLibrary(lambda p: (lambda r: None if r is None else r.to_bytes())(libFetch(p))); "
             + "advs['fast'].library = advs['accurate'].library = lib; "
             // and a 3-way flop's first decisions the 3-way library holds (gto/mw_library.py: mw_index.json, its entries
             // ~4 KB), at every speed, Practice's quick answers included
             + "from gto.mw_library import MWLibrary; mwlib = MWLibrary(lib._fetch); "
             + "advs['fast'].mw_library = advs['accurate'].mw_library = mwlib; "
             // 3-way pots (measured, scripts/bench_browser.mjs): the 3-player solver on the river in Fast (0.13-0.16 s an
             // iteration: 41-48 in 8 s, 0.3-0.4% of the pot from equilibrium); Fast's turns and flops in a smaller game
             // (advisor.MW_FAST_STREETS: 6 buckets, every sixth river, the flop betting half the pot) in its 8 s: a flop
             // first to act 51 iterations, 2.6% of the pot; checked to 85, 0.44%; a turn 110, 0.15%. Accurate's in its 60 s, the
             // later streets bet by 12 hand buckets with the turn's tables over every third river, the flop betting a third or
             // three quarters of the pot (numpy has no BLAS here: a flop iteration 0.34 s checked to, 161 in the 60 s, 0.27%;
             // 0.7-0.95 s first to act, 56-77, 3-6%; the 60 s cover the tables and the gains and EVs read off after). Turns
             // (0.18 s) and rivers stop once settled (advisor.MW_SETTLED): 200 iterations in 34 s, 100 in 15 s, 0.07%.
             + "from gto.advisor import MW_FAST_STREETS; advs['fast'].mw_solve_streets = ('flop', 'turn', 'river'); "
             + "advs['fast'].mw_street_config = {s: dict(c) for s, c in MW_FAST_STREETS.items()}; "
             + "advs['accurate'].mw_solve_streets = ('flop', 'turn', 'river'); "
             + "advs['accurate'].mw_street_config = {s: {'later': 'buckets', 'buckets': 12, 'bucket_river_step': 3} for s in ('flop', 'turn')}; "
             + "advs['accurate'].mw_street_config['flop']['bet'] = {'flop': [0.33, 0.75], 'turn': 0.5, 'river': 0.5}; "
             // 4+ way rivers (and two all-in) by the N-player solve (scripts/bench_browser.mjs): 4-way 2.5 s at Fast, 50
             // iterations, 0.08% of the pot from equilibrium (Quick 1.8 s, 0.27%); 5-way 7.6 s, 0.1% (Quick 5 iterations,
             // 5.4%); 6-way ~3 s an iteration, so Fast leaves it to the rules and Accurate's 60 s solve it
             + "advs['fast'].mw_nway_max = 5; advs['accurate'].mw_nway_max = 6; "
             // the solves kept for reuse, both speeds' under one budget (advisor.CACHE_MB): a hand's Fast flop, turn and
             // river hold ~380 MB here, Accurate's turn and river ~470; kept without bound the heap reached its 2 GB in 12
             // answers over 5 hands and every later solve failed for want of memory
             + "advs['accurate']._held = advs['fast']._held; advs['fast'].cache_mb = advs['accurate'].cache_mb = 700");
  lap("advisors");
  setup.total = +((performance.now() - T0) / 1000).toFixed(2);
  postMessage({ status: "Solving…", setup });
  return py;
})();

// A solve can only be cancelled before it starts (runPython blocks this thread); the page terminates
// the worker to stop one that is running.
const cancelled = new Set();
onmessage = async (e) => {
  if (e.data.cancel != null) { cancelled.add(e.data.cancel); return; }
  const { id, state, play } = e.data;   // play: {seat, hole}, a practice opponent's move (Advisor.play)
  try {
    const py = await ready;
    if (cancelled.delete(id)) return postMessage({ id, result: { cancelled: true } });
    py.globals.set("state_json", JSON.stringify(state));
    py.globals.set("play_json", JSON.stringify(play || null));
    const out = py.runPython("st = json.loads(state_json); pl = json.loads(play_json); adv = advs.get(st.get('speed'), advs['fast']); "
                           + "json.dumps(adv.play(st, pl['seat'], pl['hole'], pl.get('seed')) if pl else adv.advise(st))");
    postMessage({ id, result: JSON.parse(out) });
  } catch (err) {
    postMessage({ id, error: String(err).split("\n").slice(-3).join(" ") });
  }
};
