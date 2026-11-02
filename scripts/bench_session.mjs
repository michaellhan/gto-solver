// The session analyzer in the browser, timed: hand histories imported and asked as the Analyze session panel asks
// them (gto/static/session.js's own prepare and runner over index.html, loaded as the tests load it: the page's
// requests at the browser's Fast budgets), each answered by gto/static/worker.js under Pyodide in Node (the setup of
// scripts/bench_browser.mjs, copied). Prints the time per street and the hands per minute of the scope, and with
// --deep the Go deeper pass after a Quick triage (the flagged decisions at Fast) and how often its grades and the
// triage's agree.
//
//   mkdir -p ~/pyo && (cd ~/pyo && npm install --ignore-scripts pyodide@0.26.4)
//   python scripts/bench_session.py --hands 1000 --out /tmp/s1000        # (a 6-max session's hand histories)
//   node scripts/bench_session.mjs --pyodide ~/pyo --hh /tmp/s1000 [--scope triage|all|library|preflop] [--every N]
//                                  [--deep] [--every-deep] [--json out.json]
//
// --every N asks every Nth hand of the session (all its decisions, in order, as the panel does: a turn after its
// flop, whose solve the triage reads); the hands per minute count those hands. BENCH_THREADS=N: a cross-origin
// isolated page on an N-core machine (the live site), the worker's helper threads as worker_threads. The worker's
// solve cache is held to 200 MB (the page's is 700) to keep this process near 1 GB.
import { execFileSync } from "child_process";
import { Worker as Thread } from "worker_threads";
import os from "os";
import fs from "fs";
import path from "path";
import vm from "vm";
import { createRequire } from "module";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = path.join(ROOT, "gto", "static");
const require = createRequire(import.meta.url);
const arg = (name, dflt) => { const i = process.argv.indexOf("--" + name); return i < 0 ? dflt : process.argv[i + 1]; };
const flag = (name) => process.argv.includes("--" + name);

function buildBundle() {
  const venv = path.join(ROOT, ".venv", "bin", "python");
  const python = process.env.PYTHON || (fs.existsSync(venv) ? venv : "python3");
  const out = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "gto-bundle-")), "solver.zip");
  execFileSync(python, [path.join(ROOT, "scripts", "build_bundle.py"), out]);
  return out;
}

// worker.js in a sandbox with the browser APIs it uses, the repo's files served to it (scripts/bench_browser.mjs's)
async function startWorker(pyodideDir, bundlePath) {
  const req = createRequire(path.join(path.resolve(pyodideDir), "noop.js"));
  const { loadPyodide } = req("pyodide");
  let py = null, onMsg = null;
  const libFile = (url) => path.join(STATIC, url.split("?")[0]);
  const ctx = {
    console, setTimeout, clearTimeout, URL, URLSearchParams, TextDecoder, TextEncoder, Uint8Array, ArrayBuffer, JSON, Promise,
    Error, performance, WebAssembly, importScripts: () => {},
    loadPyodide: async () => (py = await loadPyodide()),
    fetch: async (url) => {
      if (/^https?:/.test(url)) return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
      const f = url.startsWith("solver.zip") ? bundlePath : url.startsWith("flop_library/") ? libFile(url) : path.join(ROOT, url.split("?")[0]);
      if (!fs.existsSync(f)) return { ok: false, status: 404 };
      const buf = fs.readFileSync(f);
      return { ok: true, status: 200, text: async () => buf.toString("utf8"), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length) };
    },
    XMLHttpRequest: class {
      open(method, url) { this.url = url; }
      send() {
        const f = libFile(this.url);
        if (fs.existsSync(f)) { this.status = 200; const b = fs.readFileSync(f); this.response = b.buffer.slice(b.byteOffset, b.byteOffset + b.length); }
        else this.status = 404;
      }
    },
    postMessage: (m) => onMsg && onMsg(m),
  };
  const NT = +(process.env.BENCH_THREADS || 0);
  if (NT > 0) {
    const blobs = new Map();
    Object.assign(ctx, {
      crossOriginIsolated: true, navigator: { hardwareConcurrency: NT },
      Blob: class { constructor(parts) { this.text = parts.join(""); } },
      URL: class extends URL { static createObjectURL(b) { const u = `blob:bench/${blobs.size}`; blobs.set(u, b.text); return u; } },
      Worker: class {
        constructor(url) { this.src = blobs.get(String(url)); }
        postMessage(d) {
          const t = this.t = new Thread(`const { workerData } = require("worker_threads");\n${this.src}\nonmessage({ data: workerData });`, { eval: true, workerData: d });
          t.on("error", () => {});
          t.unref();
        }
        terminate() { if (this.t) this.t.terminate(); }
      },
    });
  }
  ctx.self = ctx;
  vm.createContext(ctx);
  for (const k of ["libFetch", "gtoCore", "mwCore"]) Object.defineProperty(globalThis, k, { get: () => ctx[k], configurable: true });
  const t0 = performance.now();
  vm.runInContext(fs.readFileSync(path.join(STATIC, "worker.js"), "utf8"), ctx, { filename: "worker.js" });
  await new Promise((res, rej) => { onMsg = (m) => { if (m.status === "Solving…") res(); if (m.error) rej(new Error(m.error)); }; });
  const setup = (performance.now() - t0) / 1000;
  py.runPython("advs['fast'].cache_mb = advs['accurate'].cache_mb = 200");
  let id = 0;
  const heap = () => py._module.HEAPU8.buffer.byteLength / 2 ** 20;
  const request = (state) => new Promise((res) => {
    const my = ++id;
    onMsg = (m) => { if (m.id === my && !m.partial) res(m.error ? { error: m.error } : m.result); };
    ctx.onmessage({ data: { id: my, state } });
  });
  return { setup, heap, request };
}

const pyodideDir = arg("pyodide", process.env.PYODIDE_DIR), hh = arg("hh");
if (!pyodideDir || !hh) { console.error("--pyodide DIR (pyodide@0.26.4) and --hh FILE_OR_DIR are required"); process.exit(2); }
const scope = arg("scope", "triage"), every = +arg("every", 1);
const files = fs.statSync(hh).isDirectory() ? fs.readdirSync(hh).sort().map((f) => path.join(hh, f)) : [hh];

// the page and the analyzer, as the tests load them (tests/js/page.js: index.html in a sandbox, no server: the
// browser's budgets)
const { loadPage } = require(path.join(ROOT, "tests", "js", "page.js"));
const page = loadPage({});
for (const f of ["hh.js", "session.js"]) vm.runInContext(fs.readFileSync(path.join(STATIC, f), "utf8"), page.ctx, { filename: f });
await page.ready();
const SS = page.ctx.Session;
const prep = await SS.prepare(files.map((f) => ({ name: path.basename(f), text: fs.readFileSync(f, "utf8") })),
                              (t) => page.ctx.importHand(t), null, 200, async () => {});
const hands = prep.hands.filter((h, k) => k % every === 0);
const sess = SS.newSession({ ...prep, hands }, scope, "bench");
const w = await startWorker(pyodideDir, buildBundle());
const host = { ...SS.pageHost(), busy: () => false, advise: (p) => w.request(JSON.parse(JSON.stringify(p))) };
const budgets = host.budgets();
console.log(`${prep.hands.length} hands imported, ${hands.length} asked (every ${every}), ${sess.items.length} decisions at ${scope}; `
            + `worker setup ${w.setup.toFixed(1)} s, threads ${process.env.BENCH_THREADS || 1}, Fast budgets ${JSON.stringify(budgets)}`);
const pass = async (s, label) => {
  const t0 = Date.now(), n0 = s.next;
  await SS.runner(s, host, { saveEvery: 1e12 }).run();
  const sec = (Date.now() - t0) / 1000, by = {};
  for (const [k, [t, n]] of Object.entries(s.timing)) by[k] = { n, s: +t.toFixed(1), each: +(t / n).toFixed(2) };
  console.log(`${label}: ${s.next - n0} decisions in ${sec.toFixed(1)} s, ${(hands.length / (sec / 60)).toFixed(0)} hands/min `
              + `(heap ${w.heap().toFixed(0)} MB)\n  ${JSON.stringify(by)}`);
  return { decisions: s.next - n0, sec: +sec.toFixed(1), hands_per_min: +(hands.length / (sec / 60)).toFixed(1), timing: by };
};
const out = { scope, every, hands: hands.length, threads: +(process.env.BENCH_THREADS || 1), setup: +w.setup.toFixed(1) };
out.first = await pass(sess, scope);
const rp = SS.report(sess);
out.conf = rp.conf; out.flagged = rp.flagged;
console.log(`  confidence ${JSON.stringify(rp.conf)}, flagged ${rp.flagged}, errors ${sess.results.filter((r) => r.status === "error").length}`);
if (scope === "triage" && (flag("deep") || flag("every-deep"))) {
  const d = SS.deepen(sess, flag("every-deep"));
  d.timing = {};
  out.deep = await pass(d, "deeper (" + (flag("every-deep") ? "every" : "flagged") + ")");
  out.agree = SS.agreement(d.results);
  console.log(`  triage vs Fast on the same decisions: ${JSON.stringify(out.agree)}`);
}
if (arg("json")) fs.writeFileSync(arg("json"), JSON.stringify(out, null, 1));
process.exit(0);
