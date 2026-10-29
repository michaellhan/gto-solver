// The browser's solver, timed: runs gto/static/worker.js itself (its Python setup, configs and request handler,
// so the benchmark can't drift from the page) under Pyodide in Node, one request at a time, and reports each
// request's wall time against the page's budget (index.html BUDGETS: the street's budget plus each earlier postflop
// street's narrowing budget, what the page's progress bar counts down), the iterations and exploitability the
// answer reports, the time outside the solve loops, and the wasm heap (peak; Pyodide's memory never shrinks).
//
// Pyodide 0.26.4 (the worker's) is installed in a directory of its own and named with --pyodide:
//
//   mkdir -p ~/pyo && (cd ~/pyo && npm install --ignore-scripts pyodide@0.26.4)
//   node scripts/bench_browser.mjs --pyodide ~/pyo [--spots flop-lib,turn] [--speeds fast,accurate] [--json out.json]
//   node scripts/bench_browser.mjs --list
//
// --worker FILE runs an edited copy of worker.js instead (a config to try before changing the page's).
// --session runs every spot in one worker, one after another, as a page asks them over a session. --idle S waits S
// seconds after the setup before the first request (the worker warms up while idle); --prefetch S sends the page's
// prefetch (the flop's library entries fetched ahead) S seconds before each request with a flop; BENCH_NET_MS gives
// each library file that latency, and each line counts the requests' own synchronous fetches (sync-fetches).
// Each spot runs in a fresh worker (a child process: no solve cache carried between spots, like a page load);
// --warm runs Fast then Accurate in the same worker, as when the page is switched to Accurate after a Fast answer
// (Accurate starts from Fast's solve; the page's bar then counts the Accurate budget less Fast's seconds).
// numpy is read from the pyodide package's directory; the solver's code from solver.zip, built here as deploy.sh
// builds it (scripts/build_bundle.py, with the repo's .venv Python or $PYTHON). Each spot's first line gives the
// worker's setup by stage, the worker's own timings (runtime, numpy, imports, advisors).
// Times are wall clock on this machine: a browser tab is single-threaded too, but other load moves them.
import { spawnSync, execFileSync } from "child_process";
import os from "os";
import fs from "fs";
import path from "path";
import vm from "vm";
import { createRequire } from "module";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = path.join(ROOT, "gto", "static");

// the page's budgets in the browser (index.html BUDGETS.wasm): [street, each earlier postflop street]
// (BENCH_BUDGETS='{"fast":[1.5,0.25]}' tries others, BENCH_STATE='{"quick":true}' adds request fields, e.g. Practice's;
// children inherit both)
// BENCH_NET_MS: each library file's request takes this long (the CDN measured 130-670 ms an entry, 0.8-4.9 s the index)
const NET_MS = +(process.env.BENCH_NET_MS || 0);
const BUDGETS = { fast: [8, 3], accurate: [60, 12], ...JSON.parse(process.env.BENCH_BUDGETS || "{}") };

const F = (pos) => ({ pos, action: "fold" }), C = (pos) => ({ pos, action: "call" }), X = (pos) => ({ pos, action: "check" });
const R = (pos, to) => ({ pos, action: "raise", to }), B = (pos, amount) => ({ pos, action: "bet", amount });
const SRP = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), C("BB")];
const MW = [F("UTG"), F("HJ"), R("CO", 2.5), C("BTN"), F("SB"), C("BB")];
const MW4 = [F("UTG"), R("HJ", 2.5), C("CO"), C("BTN"), F("SB"), C("BB")];
const CO_BB = [F("UTG"), F("HJ"), R("CO", 2.5), F("BTN"), F("SB"), C("BB")];
const LIMP = [F("UTG"), F("HJ"), F("CO"), F("BTN"), C("SB"), X("BB")];   // (a limped pot: not in the flop library, solved live)
const BB_3BET = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), R("BB", 10), C("BTN")];
const base = (o) => ({ players: 6, stack: 100, preflop: SRP, flop: [], turn: [], river: [], ...o });
const SRP_FLOP = [X("BB"), B("BTN", 1.8), C("BB")];                 // a third-pot c-bet called
const SRP_TURN = [X("BB"), B("BTN", 6), C("BB")];                   // two thirds called
// name: request state (the page's payload without budget / prior_budget / speed)
const SPOTS = {
  "preflop": base({ hero: "CO", hole: "AhJd", board: "", preflop: [F("UTG"), F("HJ")] }),
  "preflop-mw": base({ hero: "BB", hole: "Th9h", board: "", preflop: [F("UTG"), F("HJ"), R("CO", 2.5), C("BTN"), F("SB")] }),  // priced 3-way
  "flop-lib": base({ hero: "BTN", hole: "AsQs", board: "Kc7d4h", flop: [X("BB")] }),          // in the flop library
  "flop-lib-co": base({ hero: "BB", hole: "AhQd", board: "Qh9c4d", preflop: CO_BB }),            // in it: CO vs BB, the root
  "flop-lib-3bp": base({ hero: "BB", hole: "AhQd", board: "Kc7d4h", preflop: BB_3BET }),        // a 3bet pot's root (the
                                                    // BB's check there is rare: the BTN's answer to it is solved live)
  "flop-3bp": base({ hero: "BB", hole: "AhQd", board: "Ks7d2c", preflop: BB_3BET }),           // not in it: solved
  "flop": base({ hero: "BTN", hole: "AsQs", board: "Ks7d2c", flop: [X("BB")] }),              // (in it since BTN vs BB has every flop)
  "flop-2tone": base({ hero: "BB", hole: "JcTc", board: "9h8h6c" }),                          // not in it, two-tone
  "turn-lib": base({ hero: "BB", hole: "QhJh", board: "Kc7d4h9s", flop: SRP_FLOP }),         // the flop narrowed by the library
  "turn": base({ hero: "BB", hole: "QhJh", board: "Ks7d2cTh", flop: SRP_FLOP }),             // (its flop in the library now)
  "turn-3bp": base({ hero: "BB", hole: "AhQd", board: "Ks7d2cTh", preflop: BB_3BET, flop: [X("BB"), B("BTN", 6.8), C("BB")] }),  // the flop solved to narrow
  // a limped pot asked street by street (--session flop-limp,turn-limp,river-limp: each street's narrowing reads the
  // solve the one before made)
  "flop-limp": base({ hero: "SB", hole: "QhJh", board: "Ks7d2c", preflop: LIMP }),
  "turn-limp": base({ hero: "SB", hole: "QhJh", board: "Ks7d2cTh", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")] }),
  "river-limp": base({ hero: "SB", hole: "QhJh", board: "Ks7d2cTh5s", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")],
                       turn: [X("SB"), B("BB", 3), C("SB")] }),
  "river": base({ hero: "BB", hole: "QhJh", board: "Ks7d2cTh5s", flop: SRP_FLOP, turn: SRP_TURN }),
  "mw-flop": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c", preflop: MW }),                // first to act
  "mw-flop-x": base({ hero: "BTN", hole: "AhJd", board: "Kh7d2c", preflop: MW, flop: [X("BB"), X("CO")] }),
  "mw-turn": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c9s", preflop: MW, flop: [X("BB"), X("CO"), X("BTN")] }),
  "mw-river": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c9s4h", preflop: MW, flop: [X("BB"), X("CO"), X("BTN")],
                     turn: [X("BB"), X("CO"), X("BTN")] }),
  "mw4-flop": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c", preflop: MW4 }),               // 4-way, first to act
  "mw4-turn": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c9s", preflop: MW4, flop: [X("BB"), X("HJ"), X("CO"), X("BTN")] }),
  "exploit": base({ hero: "BB", hole: "QhJh", board: "Ks7d2cTh", flop: SRP_FLOP, opponents: "pool" }),
  "lock": base({ hero: "BB", hole: "QhJh", board: "Ks7d2cTh", flop: SRP_FLOP, locks: { BTN: { fold: 15, bluff: 30 } } }),
  "raked": base({ hero: "BTN", hole: "AsQs", board: "Kc7d4h", flop: [X("BB")], rake: 0.05, rake_cap: 3 }),
};
const STREET_IDX = { 0: 0, 3: 1, 4: 2, 5: 3 };

function arg(name, dflt) {
  const i = process.argv.indexOf("--" + name);
  return i < 0 ? dflt : process.argv[i + 1];
}
const flag = (name) => process.argv.includes("--" + name);

// the page's estimate (index.html analyze(): the bar's "of ~N s")
function pageBudget(state, speed, fastSeconds) {
  const street = STREET_IDX[state.board.length / 2];
  if (street === 0) return null;
  const [b, p] = BUDGETS[speed];
  let est = b + p * Math.max(0, street - 1);
  if (speed === "accurate" && fastSeconds) est = Math.round(Math.max(est * 0.2, est - fastSeconds));
  return est;
}

// worker.js in a sandbox with the browser APIs it uses, served from the repo (deploy.sh puts gto/*.py and the flop
// library next to the page)
function buildBundle() {
  const venv = path.join(ROOT, ".venv", "bin", "python");
  const python = process.env.PYTHON || (fs.existsSync(venv) ? venv : "python3");
  const out = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "gto-bundle-")), "solver.zip");
  execFileSync(python, [path.join(ROOT, "scripts", "build_bundle.py"), out]);
  return out;
}

async function startWorker(pyodideDir, bundlePath) {
  const require = createRequire(path.join(path.resolve(pyodideDir), "noop.js"));
  const { loadPyodide } = require("pyodide");
  let py = null, onMsg = null;
  const ctx = {
    console, setTimeout, clearTimeout, URL, URLSearchParams, TextDecoder, TextEncoder, Uint8Array, ArrayBuffer, JSON, Promise,
    Error, performance, WebAssembly, importScripts: () => {},
    loadPyodide: async () => (py = await loadPyodide()),   // the npm package's own files, not the CDN's
    fetch: async (url) => {
      if (/^https?:/.test(url)) return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };   // the numpy prefetch
      if (url.startsWith("flop_library/")) {   // the library's files fetched ahead (BENCH_NET_MS: a CDN's latency)
        ctx.aheadFetches++;
        await new Promise((r) => setTimeout(r, NET_MS));
      }
      const f = url.startsWith("solver.zip") ? bundlePath : url.startsWith("flop_library/") ? path.join(STATIC, url.split("?")[0])
        : path.join(ROOT, url.split("?")[0]);
      if (!fs.existsSync(f)) return { ok: false, status: 404 };
      const buf = fs.readFileSync(f);
      return { ok: true, status: 200, text: async () => buf.toString("utf8"),
               arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length) };
    },
    XMLHttpRequest: class {
      open(method, url) { this.url = url; }
      send() {
        const t = performance.now();
        while (performance.now() - t < NET_MS);   // (synchronous: the worker waits)
        const f = path.join(STATIC, this.url.split("?")[0]);
        if (fs.existsSync(f)) { this.status = 200; const b = fs.readFileSync(f); this.response = b.buffer.slice(b.byteOffset, b.byteOffset + b.length); ctx.libFetches++; }
        else this.status = 404;
      }
    },
    libFetches: 0, aheadFetches: 0,
    postMessage: (m) => onMsg && onMsg(m),
  };
  ctx.self = ctx;
  vm.createContext(ctx);
  const src = fs.readFileSync(arg("worker", path.join(STATIC, "worker.js")), "utf8");
  const t0 = performance.now();
  // Python's `from js import ...` reads Node's global, not the sandbox's: the library's fetch and the cores the setup
  // instantiates are read through to the sandbox's from the start (the setup asks for the multiway core, gto/mwcore.py
  // lib(), and keeps the answer: set only after it, the bench ran every multiway solve in numpy)
  for (const k of ["libFetch", "gtoCore", "mwCore"]) Object.defineProperty(globalThis, k, { get: () => ctx[k], configurable: true });
  vm.runInContext(src, ctx, { filename: "worker.js" });
  // the setup has run once the worker reports it is solving
  let stages = null;
  await new Promise((res, rej) => {
    onMsg = (m) => { if (m.status === "Solving…") { stages = m.setup || null; res(); } if (m.error) rej(new Error(m.error)); };
  });
  const setup = (performance.now() - t0) / 1000;
  const heap = () => py._module.HEAPU8.buffer.byteLength / 2 ** 20;
  let id = 0;
  const prefetch = (state) => ctx.onmessage({ data: { prefetch: state } });
  const request = (state) => new Promise((res) => {
    const my = ++id;
    onMsg = (m) => { if (m.id === my) res(m); };
    ctx.onmessage({ data: { id: my, state } });
  });
  return { setup, stages, heap, request, prefetch, ctx };
}

// one spot's requests in one worker (--session: every spot's, one after another in the same worker, as a page
// asks them over a session: the library's index fetched once, the solves kept); prints JSON lines
async function runOne(spotList, speeds, pyodideDir, bundlePath) {
  const w = await startWorker(pyodideDir, bundlePath);
  console.log(JSON.stringify({ spot: spotList.join(","), setup: +w.setup.toFixed(2), stages: w.stages, heap_mb: +w.heap().toFixed(0) }));
  // --idle S: the first request S seconds after the setup (the time a person takes to act: the worker warms up then)
  if (arg("idle")) await new Promise((r) => setTimeout(r, 1000 * +arg("idle")));
  for (const spot of spotList) await runSpot(w, spot, speeds);
}
async function runSpot(w, spot, speeds) {
  let fastSeconds = null;
  for (const speed of speeds) {
    const [budget, prior_budget] = BUDGETS[speed];
    const state = { ...SPOTS[spot], budget, prior_budget, speed, ...JSON.parse(process.env.BENCH_STATE || "{}") };
    // --prefetch S: the page's prefetch as the flop is known, S seconds before the request (the time the hand takes
    // to reach your decision)
    if (arg("prefetch") && state.board.length >= 6) {
      w.prefetch({ ...state, board: state.board.slice(0, 6), flop: [], turn: [], river: [] });
      await new Promise((r) => setTimeout(r, 1000 * +arg("prefetch")));
    }
    const [x0, a0] = [w.ctx.libFetches, w.ctx.aheadFetches];
    const t0 = performance.now();
    const m = await w.request(state);
    const wall = (performance.now() - t0) / 1000;
    const r = m.result || {}, s = r.solver || {};
    const est = pageBudget(state, speed, fastSeconds);
    console.log(JSON.stringify({
      spot, speed, wall: +wall.toFixed(2), page_budget: est, over_pct: est ? Math.round((wall / est - 1) * 100) : null,
      iterations: s.iterations ?? null, solve_s: s.solve_seconds ?? null, expl_pct: s.exploitability_pct ?? null, cells: s.cells ?? null,
      outside_s: s.solve_seconds != null ? +(wall - s.solve_seconds).toFixed(2) : null,
      precomputed: !!r.precomputed, heap_mb: +w.heap().toFixed(0), lib_fetches: w.ctx.libFetches - x0, lib_ahead: w.ctx.aheadFetches - a0,
      street: r.street, recommended: r.recommended, error: m.error || r.error || null,
      notes: (r.notes || []).filter((n) => process.env.BENCH_ALL_NOTES || /equilibrium|solved|iterations|rules/.test(n)).map((n) => n.slice(0, 160)),
    }));
    if (speed === "fast") fastSeconds = r.seconds;
  }
}

if (flag("child")) {
  await runOne(arg("spot").split(","), arg("speeds").split(","), arg("pyodide"), arg("bundle"));
  process.exit(0);
}
if (flag("list")) {
  for (const [k, v] of Object.entries(SPOTS)) console.log(k.padEnd(12), JSON.stringify(v));
  process.exit(0);
}
const pyodideDir = arg("pyodide", process.env.PYODIDE_DIR);
if (!pyodideDir) { console.error("--pyodide DIR (where pyodide@0.26.4 is installed) is required"); process.exit(2); }
const spots = arg("spots", Object.keys(SPOTS).join(",")).split(",");
const speeds = arg("speeds", "fast,accurate").split(",");
const bundlePath = buildBundle();
const rows = [];
for (const spot of spots) if (!SPOTS[spot]) { console.error(`unknown spot ${spot} (--list)`); process.exit(2); }
for (const spot of flag("session") ? [spots.join(",")] : spots) {
  // a fresh worker per spot (and per speed unless --warm)
  for (const group of flag("warm") ? [speeds] : speeds.map((s) => [s])) {
    const p = spawnSync(process.execPath, [fileURLToPath(import.meta.url), "--child", "--spot", spot, "--speeds", group.join(","),
                                           "--pyodide", pyodideDir, "--bundle", bundlePath, ...(arg("worker") ? ["--worker", arg("worker")] : []), ...["prefetch", "idle"].flatMap((k) => (arg(k) ? ["--" + k, arg(k)] : []))], { encoding: "utf8", maxBuffer: 1 << 26 });
    for (const line of p.stdout.split("\n").filter((l) => l.startsWith("{"))) {
      const r = JSON.parse(line);
      if (r.speed) rows.push(r);
      console.log(r.speed ? `${r.spot.padEnd(11)} ${r.speed.padEnd(8)} ${String(r.wall).padStart(6)} s of ~${r.page_budget ?? "-"} ` +
                            `(${r.over_pct == null ? "-" : (r.over_pct > 0 ? "+" : "") + r.over_pct + "%"})  it ${r.iterations ?? "-"}  ` +
                            `expl ${r.expl_pct ?? "-"}%  solve ${r.solve_s ?? "-"} s  outside ${r.outside_s ?? "-"} s  heap ${r.heap_mb} MB` +
                            `${r.cells ? "  cells " + (r.cells / 1e6).toFixed(1) + "M" : ""}` +
                            `${r.precomputed ? "  library" : ""}${r.lib_fetches ? "  sync-fetches " + r.lib_fetches : ""}${r.error ? "  ERROR " + r.error : ""}` +
                            (r.notes.length ? "\n             " + r.notes.join("\n             ") : "")
                          : `${r.spot}: worker setup ${r.setup} s${r.stages ? " (" + Object.entries(r.stages).filter(([k]) => k !== "total").map(([k, v]) => `${k} ${v}`).join(", ") + ")" : ""}, heap ${r.heap_mb} MB`);
    }
    if (p.status !== 0) console.error(p.stderr.split("\n").slice(-15).join("\n"));
  }
}
const out = arg("json");
if (out) fs.writeFileSync(out, JSON.stringify(rows, null, 1));
