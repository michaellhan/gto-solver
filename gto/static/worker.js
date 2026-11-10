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
const LIBRARY = new URLSearchParams(self.location ? self.location.search : "").get("lib") || "flop_library/";   // (the page's LIB_BASE)
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
// (a failure here is the setup's, reported as any other: this script goes on, so its onmessage is there to answer)
let scriptError = null;
try { importScripts(PYODIDE + "pyodide.js"); } catch (err) { scriptError = err; }

// The precomputed flop library (gto/flop_library.py): the Python side asks for its index once and for an entry
// (<spot>/<flop>.bin, 1.5-6 KB) only when a flop spot it holds comes up. The solver runs synchronously, so it reads
// them synchronously: from LIB, the files fetched ahead, else by a synchronous request (allowed in a worker); a missing
// or unreadable file is just no library. Fetched ahead: both indexes (index.json 117 KB gzipped, 0.8-4.9 s on a cold
// CDN; mw_index.json), requested now beside the runtime, and a flop's entries, with their turn files where the index
// lists them (<spot>/<flop>.turn.bin, ~10-25 KB: the turn's approximate answer at once), once the page says the flop is
// known (a "prefetch" message: Advisor.prefetch's dry run names them, RECORD); a request waits for those still in flight
// (whenAhead) rather than asking again. A synchronous request blocked the worker for the index on a session's first
// flop and for each entry (0.13-0.67 s each from here, through the CDN), where the answer itself takes 10-40 ms.
const LIB = new Map();   // path -> Uint8Array (fetched) or a Promise (in flight)
function libAhead(path) {
  if (!LIB.has(path)) {
    if (LIB.size >= 64) for (const [k, v] of LIB) if (v instanceof Uint8Array && !k.endsWith("index.json")) { LIB.delete(k); break; }   // (one never asked for)
    const p = fetch(LIBRARY + path).then((r) => r.ok ? r.arrayBuffer() : null)
      .then((b) => { if (b) LIB.set(path, new Uint8Array(b)); else LIB.delete(path); }, () => { LIB.delete(path); });
    LIB.set(path, p);
  }
}
libAhead("index.json");
libAhead("mw_index.json");
async function whenAhead() {   // (until none is in flight: a prefetch can start more while a request waits)
  for (let p; (p = [...LIB.values()].filter((v) => v instanceof Promise)).length; ) await Promise.all(p);
}
let RECORD = null;   // the paths a prefetch's dry run asks for (entries: not read now; the indexes always are)
self.libFetch = (path) => {
  const hit = LIB.get(path);
  if (hit instanceof Uint8Array) { LIB.delete(path); return hit; }   // (Python keeps what it decodes)
  if (RECORD && !path.endsWith("index.json")) { RECORD.push(path); return null; }
  try {
    const x = new XMLHttpRequest();
    x.open("GET", LIBRARY + path, false);
    x.responseType = "arraybuffer";
    x.send();
    return x.status === 200 ? new Uint8Array(x.response) : null;
  } catch (e) {
    return null;
  }
};
// A file of the site's own beside this worker (gto/pf_study.py's stored preflop solves, pf/<game>.bin, ~300 KB),
// read synchronously as the library's entries are; null where it's missing.
self.siteFetch = (path) => {
  try {
    const x = new XMLHttpRequest();
    x.open("GET", path, false);
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
// isolation, or where the threaded module won't load, core.wasm alone. The multiway core the same way (mwcore-mt.wasm,
// its own memory and helpers, in mw_helper: a pass's root-street leaves, core/mwcore.c). A helper that fails (a trap:
// core_helper throws) marks itself in the core's memory (core_dead), where the solve waiting on its rows reads it: that
// solve's core_run throws in turn instead of the worker waiting for good. Then gto/core.py (gto/mwcore.py) starts the
// core again (reload: a new instance on a fresh memory with new helpers, the old ones ended) and runs the solve again
// in it, from the solver's arrays as they were before the failed run; numpy runs it if it fails again.
const HELPERS = Math.min(7, ((self.navigator && navigator.hardwareConcurrency) || 1) - 1);
const HELPER = `onmessage = async (e) => {
  const { mod, memory, i, pre } = e.data;         // (pre: the module's exports' prefix, core_ or mw_)
  const ex = (await WebAssembly.instantiate(mod, { env: { memory } })).exports;
  ex.__stack_pointer.value = ex[pre + "stack"](i);   // (every instance starts on the same stack)
  const dead = new Int32Array(memory.buffer, ex[pre + "dead"](0), 1);
  try {
    ex[pre + "helper"](i);                        // runs the solves' jobs; returns only by failing
  } catch (err) {
    Atomics.store(dead, 0, i + 1);
  }
};`;
let helperURL = null;
// the module started (its exports, the memory among them, and reload()); `name`: the global it's kept in
function start(mod, pre, mt, name) {
  let ex, helpers = [];
  if (mt) {
    const memory = new WebAssembly.Memory({ initial: 256, maximum: 32768, shared: true });
    ex = Object.assign({}, new WebAssembly.Instance(mod, { env: { memory } }).exports, { memory });
    helperURL = helperURL || URL.createObjectURL(new Blob([HELPER], { type: "text/javascript" }));
    for (let i = 0; i < HELPERS; i++) {
      const w = new Worker(helperURL);
      w.postMessage({ mod, memory, i, pre });
      helpers.push(w);
    }
    ex[pre + "threads"](HELPERS + 1);
  } else {
    ex = Object.assign({}, new WebAssembly.Instance(mod).exports);
  }
  ex.reload = () => {
    for (const w of helpers) w.terminate();
    return (self[name] = start(mod, pre, mt, name));
  };
  return ex;
}
async function loadCore(py, file, pre, name) {
  if (self.crossOriginIsolated && HELPERS > 0) {
    try {
      return start(await WebAssembly.compile(py.FS.readFile(file.replace(".wasm", "-mt.wasm"))), pre, true, name);
    } catch (e) {}
  }
  try {
    return start(await WebAssembly.compile(py.FS.readFile(file)), pre, false, name);
  } catch (e) {
    return null;
  }
}

// The setup, in the stages the page shows; their seconds come back with "Solving…" (setup: for
// scripts/bench_browser.mjs and the console).
const ready = (async () => {
  if (scriptError) throw scriptError;
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
  self.gtoCore = await loadCore(py, "/app/gto/core.wasm", "core_", "gtoCore");
  setup.core = !!self.gtoCore;
  setup.threads = self.gtoCore ? self.gtoCore.core_threads(0) : 0;
  // the multiway core (gto/mwcore.py, ~120 KB): 3- and 4-way flop and turn solves' iterations, with the same helpers'
  // number of threads
  self.mwCore = await loadCore(py, "/app/gto/mwcore.wasm", "mw_", "mwCore");
  setup.mwcore = !!self.mwCore;
  setup.mwthreads = self.mwCore && self.mwCore.mw_threads ? self.mwCore.mw_threads(0) : 0;
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
  // a bet or raise leaving under 15% of the pot behind once called is the all-in (Config.force_allin; gto/native.py)
  const RAISES = "raise_sizes={'flop': [0.5], 'turn': [0.75], 'river': [0.75]}, force_allin=0.15";
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
             // Fast's heads-up solves stop once settled (Advisor.hu_settled: the first of advisor.HU_CHECKS where the
             //   solve is that close to equilibrium, % of the pot, from the 300th iteration on: advisor.HU_SETTLED_FROM)
             //   instead of running the whole budget. With threads (Accurate's turn tree) at 0.2: ~300 iterations, which
             //   against the rich reference on the five single-raised turns is 0.12% of the pot off per combo (0.07 at
             //   450, 0.21 at 200; README, Accuracy vs speed). On one thread Fast's own tree, whose menu leaves it 0.34
             //   off however long it runs (0.35 at 200), at 0.3, from the 200th: ~200 iterations. A river or a 3bet
             //   pot's turn settles at 300 of its ~1000, within 0.1 of the 1000th's per-combo EVs.
             //   With threads a turn runs on to its 600th first: its exploitability is under 0.2 by the 300th (2-3 s of
             //   its ~11), but its per-combo EVs are still moving there. Against the same tree at 2000 iterations (nine
             //   of scripts/turn_river_reference.py's turns, scripts/turn_convergence.py) the five single-raised turns
             //   are 0.19% of the pot off per combo at 300, 0.08 at 600, 0.04 at 1000 (frequencies 0.71, 0.24, 0.11
             //   points); the four 3bet turns 0.24 (one 0.58), 0.05 and 0.02 (0.40, 0.06, 0.03).
             + `advs['fast'].hu_settled = ${THREADED ? 0.2 : 0.3}; `
             + (THREADED ? "advs['fast'].hu_settled_from.update(turn=600, shallow=600); "
                         : "advs['fast'].hu_settled_from['turn'] = 200; ")
             // a heads-up answer after the flop 10 times further from equilibrium than a speed's are (advisor.OVER_TARGET:
             // Fast's settled stop, Accurate's 0.1% of the pot) says it's approximate: its solve ran short (the device
             // busy or slow, a big tree, or the compiled core failed twice and numpy answered: 5-7% on a Fast turn)
             + `advs['fast'].expl_target = ${THREADED ? 0.2 : 0.3}; advs['accurate'].expl_target = 0.1; `
             // a live flop or turn shows its answer so far while it solves (Advisor.progress, first_look: the solve
             //   stopped and read at these iterations, where it has at least advisor.LOOK_MIN_S left to where it next
             //   may stop, and isn't settled there; the answer the same): Fast at the 100th (32 turns: the mix 15
             //   points per combo off a 1000-iteration solve's, 0.02% of the pot given up; the answer, settled at
             //   300-400, 4.4 and 0.005) and at the 200th, 300th and 400th where it runs on past them (8.0 and 0.007,
             //   5.2 and 0.005, 3.6 and 0.004), Accurate at the 100th, 300th and 600th (1.8 and 0.004) of its 1000
             + "advs['fast'].first_look = {'flop': (100, 200, 300, 400), 'turn': (100, 200, 300, 400)}; "
             + "advs['accurate'].first_look = {'flop': (100, 300, 600), 'turn': (100, 300, 600)}; "
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
             // 3-way pots (measured, scripts/bench_browser.mjs): the 3-player solver on every street, its iterations in
             // the multiway core (gto/mwcore.wasm; numpy took 0.7-0.95 s a first-to-act flop iteration here, 56-77 in
             // Accurate's 60 s, 3-6% of the pot from equilibrium). Accurate plays native's game (advisor.MW_LATER, 16
             // hand buckets on the later streets, the flop betting a third or three quarters of the pot, the turn
             // 33/75/125%): a flop first to act (Kh8d3c, CO opens, BTN and BB call) 400 iterations in 13 s, 0.05% of the
             // pot. Fast a smaller flop game (advisor.MW_FAST_STREETS: 12 buckets, every third river) in its 8 s: 203
             // iterations, 0.25%; checked to twice 400, 0.03%; a turn 400, 0.04%. Quick's 1.8 s (advisor.MW_QUICK_FLOP,
             // MW_QUICK_TURN): a flop 114 iterations, 0.43%; a turn 87, 0.58%. Without the core (it failed to load)
             // Accurate keeps the old smaller game (12 buckets, every third river), Fast 6 buckets, and 4-way flops and
             // turns are left to the rules. Rivers and turns stop once settled (advisor.MW_SETTLED). 4-way flops and
             // turns: the 4-player solve (advisor.MW4_STREETS; Fast MW4_FAST_STREETS), in the core too: a flop (Kh7d2c,
             // HJ opens, CO, BTN and BB call) Fast 200 iterations in 3.8 s, 0.03% of the pot; a turn 100 in 0.4 s, 0.04%.
             // Quick's 4-way flop is the rules' (mw4_quick_flop: its tables took 1.6 s, then 15 iterations, 9.4%).
             + "from gto.advisor import MW_FAST_STREETS, MW4_FAST_STREETS; from gto import mwcore as _mwc; "
             + "advs['fast'].mw_solve_streets = advs['accurate'].mw_solve_streets = ('flop', 'turn', 'river'); "
             + "advs['fast'].mw_street_config = {s: dict(c) for s, c in MW_FAST_STREETS.items()}; "
             + "advs['fast'].mw4_street_config = {s: dict(c) for s, c in MW4_FAST_STREETS.items()}; "
             + "_slow = _mwc.lib() is None; "
             + "advs['accurate'].mw_street_config = {s: dict(advs['accurate'].mw_street_config[s], buckets=12, bucket_river_step=3) for s in ('flop', 'turn')} if _slow else advs['accurate'].mw_street_config; "
             + "advs['fast'].mw_street_config = {s: dict(c, buckets=6, bucket_river_step=6) for s, c in advs['fast'].mw_street_config.items()} if _slow else advs['fast'].mw_street_config; "
             + "advs['fast'].mw4_street_config = None if _slow else advs['fast'].mw4_street_config; "
             + "advs['accurate'].mw4_street_config = None if _slow else advs['accurate'].mw4_street_config; "
             + "advs['fast'].mw4_quick_flop = None; "
             // 4+ way rivers (and two all-in) by the N-player solve (scripts/bench_browser.mjs): 4-way 2.5 s at Fast, 50
             // iterations, 0.08% of the pot from equilibrium (Quick 1.8 s, 0.27%); 5-way 7.6 s, 0.1% (Quick 5 iterations,
             // 5.4%); 6-way ~3 s an iteration, so Fast leaves it to the rules and Accurate's 60 s solve it
             + "advs['fast'].mw_nway_max = 5; advs['accurate'].mw_nway_max = 6; "
             // the solves kept for reuse, both speeds' under one budget (advisor.CACHE_MB): a hand's Fast flop, turn and
             // river hold ~380 MB here, Accurate's turn and river ~470; kept without bound the heap reached its 2 GB in 12
             // answers over 5 hands and every later solve failed for want of memory
             + "advs['accurate']._held = advs['fast']._held; advs['fast'].cache_mb = advs['accurate'].cache_mb = 700");
  // a solve stops when the page says so (stopNow, below): Advisor.should_stop, read between the core's chunks of
  // iterations (~0.1 s); the request then answers {cancelled: true}
  py.globals.set("stop_now", stopNow);
  py.runPython("from gto.solver import Cancelled\n"
             + "for _a in advs.values(): _a.should_stop = lambda: bool(stop_now())\n"
             + "def answer(st, pl):\n"
             + "    adv = advs.get(st.get('speed'), advs['fast'])\n"
             + "    try:\n"
             + "        return json.dumps(adv.play(st, pl['seat'], pl['hole'], pl.get('seed')) if pl else adv.advise(st))\n"
             + "    except Cancelled:\n"
             + "        return json.dumps({'cancelled': True})\n"
             // a request marked "early" (the page's own advice): an approximate answer first where one is in hand at once
             // (Advisor.instant: the previous street's solve read for this street, ~0.1 s), posted before the solve
             + "def early(st):\n"
             + "    try:\n"
             + "        r = advs.get(st.get('speed'), advs['fast']).instant(st)\n"
             + "    except Exception:\n"
             + "        r = None\n"
             + "    return json.dumps(r) if r else ''\n"
             // ...and then, where the street is solved live, its answer so far at each of the speed's looks
             // (Advisor.progress: the solve stops there, is read and goes on; {partial} each, else {result}: answered in
             // full on the way, or stopped)
             + "def progress(st):\n"
             + "    try:\n"
             + "        r = advs.get(st.get('speed'), advs['fast']).progress(st)\n"
             + "    except Cancelled:\n"
             + "        r = {'result': {'cancelled': True}}\n"
             + "    except Exception:\n"
             + "        r = None\n"
             + "    return json.dumps(r) if r else ''\n"
             // a presolve (the benchmark's, scripts/bench_browser.mjs --presolve; the page asks for none): the solve your next
             // decision on the street reads, made now and kept (Advisor.presolve); the request that follows goes on from it
             + "def presolve(st, seat):\n"
             + "    try:\n"
             + "        r = advs.get(st.get('speed'), advs['fast']).presolve(st, seat)\n"
             + "    except Cancelled:\n"
             + "        return json.dumps({'cancelled': True})\n"
             + "    return json.dumps(r or {'skipped': True})");
  lap("advisors");
  setup.total = +((performance.now() - T0) / 1000).toFixed(2);
  postMessage({ status: "Solving…", setup });
  return py;
})();

// Idle time (no request waiting): Advisor-side work an answer would otherwise wait for (advisor.warm_step: the
// preflop equity tables, 0.5 s the first time a multiway preflop spot is priced, then the modules answers import on
// first use), ~50 ms at a time so a request that comes waits at most that.
let busy = 0, warmDone = false, warmTimer = null;
function idle() {
  if (warmDone || warmTimer) return;
  warmTimer = setTimeout(async () => {
    warmTimer = null;
    if (busy) return;
    try { warmDone = (await ready).runPython("from gto.advisor import warm_step; warm_step(0.05)"); } catch (e) { warmDone = true; }
    idle();
  }, 30);
}
ready.then(idle, () => {});
// A load that fails is said at once ({loadFailed}: the page offers Try again), not only to the requests that come after
// (each still answers {id, error}). The script's own failure also stays an error on the worker, as it was before
// the try above: Lizard's worker-transport hears of it by onerror.
ready.catch((err) => {
  postMessage({ loadFailed: String(err && err.message || err) });
  if (err === scriptError && typeof reportError === "function") reportError(err);
});

// Stopping a request. Before it starts: a "cancel" message (the page's, while the runtime loads). Once it runs it
// blocks this thread, so messages wait: where the page is cross-origin isolated it hands over a SharedArrayBuffer
// ("stop": every request up to the id in STOP[0] is stopped, and the one in STOP[1]), which the running solve reads
// between chunks of iterations (stopNow, Advisor.should_stop) and stops within ~0.1 s, the worker keeping its solves,
// library entries and tables; elsewhere the page terminates the worker (the runtime reloads from cache, ~1.5 s, and
// the hand's earlier streets are solved again).
const cancelled = new Set();
let STOP = null, CUR = 0;   // (CUR: the request running)
function stopNow() { return STOP !== null && (Atomics.load(STOP, 0) >= CUR || Atomics.load(STOP, 1) === CUR); }
onmessage = async (e) => {
  if (e.data.ping != null) return postMessage({ pong: e.data.ping });   // (the page: this thread isn't stuck)
  if (e.data.cancel != null) { cancelled.add(e.data.cancel); return; }
  if (e.data.stop) { STOP = new Int32Array(e.data.stop); return; }
  if (e.data.prefetch) {   // the page's hand has a flop: its library entries, fetched ahead (above)
    try {
      const py = await ready;
      await whenAhead();   // (the indexes: the dry run reads them)
      py.globals.set("state_json", JSON.stringify(e.data.prefetch));
      RECORD = [];
      py.runPython("advs['fast'].prefetch(json.loads(state_json))");
      for (const path of RECORD) libAhead(path);
    } catch (err) {} finally { RECORD = null; }
    return;
  }
  // play: {seat, hole}, a practice opponent's move (Advisor.play); early: post an approximate answer first ({id, partial});
  // presolve: the opponent's seat, to act (Advisor.presolve: {presolved, iterations}, {skipped} or {cancelled})
  const { id, state, play, early, presolve } = e.data;
  busy++;
  try {
    const py = await ready;
    if (String(state.board || "").length >= 6) await whenAhead();   // (a preflop answer needs no library file)
    CUR = id;
    if (cancelled.delete(id) || stopNow()) return postMessage({ id, result: { cancelled: true } });
    py.globals.set("state_json", JSON.stringify(state));
    py.globals.set("play_json", JSON.stringify(play || null));
    if (presolve) {
      py.globals.set("pre_seat", String(presolve));
      return postMessage({ id, result: JSON.parse(py.runPython("presolve(json.loads(state_json), pre_seat)")) });
    }
    if (early && !play) {
      const r = py.runPython("early(json.loads(state_json))");
      if (r) postMessage({ id, partial: JSON.parse(r) });
      for (let k = 0; k < 4; k++) {   // (the street's own solve so far, at each look: Advisor.progress)
        const p = py.runPython("progress(json.loads(state_json))");
        if (!p) break;
        const o = JSON.parse(p);
        if (o.result) return postMessage({ id, result: o.result });
        postMessage({ id, partial: o.partial });
      }
    }
    postMessage({ id, result: JSON.parse(py.runPython("answer(json.loads(state_json), json.loads(play_json))")) });
  } catch (err) {
    postMessage({ id, error: String(err).split("\n").slice(-3).join(" ") });
  } finally {
    busy--;
    idle();
  }
};
