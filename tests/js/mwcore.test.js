// The multiway core's threaded module (gto/mwcore-mt.wasm, built by scripts/build_core.py from core/mwcore.c): its
// memory shared with helpers (worker.js's own helper script, here on worker_threads), which run a pass's root-street
// leaves. A 3-player game small enough to write by hand (each player's decision in turn, then a fold leaving one of
// them the pot, its deal masses over the others' reaches) comes out the same, bit for bit, on 1 thread and on 4, and
// the helpers ran some of its leaves; a helper that fails ends the solve's wait instead of hanging it. (The Python
// tests check the native build of the same source, threads and all, against numpy on real trees.)
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..", "..");
const MAGIC = 0x6d776332;

// Players 0, 1 and 2 each pick one of two actions (RDEC: regrets, sums, a kept strategy and its flag per decision),
// then a fold (RFOLD) leaves one of them the pot: 8 leaves, a pass's tasks. n combos, some sharing a card.
function mwGame(ex, n) {
  const MAGIC = 0x6d776332;
  ex.mw_reset();
  const base = ex.mw_base() >>> 0;
  const put = (words, Arr) => {
    const off = ex.mw_alloc(words.length * 4) >>> 0;
    new Arr(ex.memory.buffer, base + off, words.length).set(words);
    return off;
  };
  const f = (x) => new Int32Array(new Float32Array([x]).buffer)[0];
  const ca = put([...Array(n).keys()].map((i) => i), Int32Array);
  const cb = put([...Array(n).keys()].map((i) => i + 5), Int32Array);
  const R0 = put([0, 1, 2].flatMap((p) => [...Array(n).keys()].map((i) => (i * (p + 3)) % 4 === 1 ? 0 : 1 + ((i + p) % 3))), Float32Array);
  const ndec = 7, len = ndec * 2 * n;
  const regret = put(new Array(len).fill(0), Float32Array), ssum = put(new Array(len).fill(0), Float32Array);
  const sig = put(new Array(len).fill(0), Float32Array), flags = put(new Array(ndec).fill(0), Int32Array);
  const arrs = put([regret, ssum, len], Int32Array);
  // the records after the header's 26 words: decision d at depth q (player q), its children next
  const words = [];
  let dec = 0, leaf = 0;
  const node = (q) => {
    const at = 26 + words.length;
    if (q === 3) {
      const l = leaf++;
      words.push(2, l % 3, f(3 + l), f(1 + (l % 2)), f(1), f(0.5 * l));
      return at;
    }
    const d = dec++;
    words.push(1, q, 2, regret + 8 * n * d, ssum + 8 * n * d, sig + 8 * n * d, d, 0, 0);
    const a = node(q + 1), b = node(q + 1);
    words[at - 26 + 7] = a;
    words[at - 26 + 8] = b;
    return at;
  };
  const root = node(0);
  const prog = put([MAGIC, 3, n, 1, 1, root, ca, cb, 0, 0, 0, 0, 1, arrs, 0, R0, 0, 0, 0, flags, ndec, 0, 0, 0, 0, 0, ...words], Int32Array);
  const iters = 40;
  const t = [...Array(iters).keys()].map((i) => i + 1);
  const fac = put(t.flatMap((x) => [x ** 1.5 / (x ** 1.5 + 1), (x / (x + 1)) ** 2]), Float32Array);
  const out = ex.mw_alloc(4 * n) >>> 0;
  return {
    run: () => ex.mw_run(prog, iters, fac),
    // regrets and sums, then each player's best response and value on the average strategies (BR 1, EV 2)
    read: () => {
      const got = [...new Int32Array(ex.memory.buffer, base + regret, 2 * len)];
      for (const mode of [1, 2]) {
        for (let p = 0; p < 3; p++) {
          if (ex.mw_eval(prog, root, mode, p, R0, out, 0) !== 0) throw new Error("bad program");
          got.push(...new Int32Array(ex.memory.buffer, base + out, n));
        }
      }
      return got;
    },
  };
}

const helperPage = () => {
  // worker.js's helper (a browser worker: its onmessage, called here with what worker.js posts it)
  const src = fs.readFileSync(path.join(ROOT, "gto", "static", "worker.js"), "utf8");
  const helper = src.match(/const HELPER = `([\s\S]*?)`;/)[1];
  return `const { workerData } = require("node:worker_threads");\n${helper}\nonmessage({ data: workerData });`;
};

test("mwcore-mt: threads give the same numbers, and the helpers run leaves (worker.js's helper)", async () => {
  const { Worker } = require("node:worker_threads");
  const mod = await WebAssembly.compile(fs.readFileSync(path.join(ROOT, "gto", "mwcore-mt.wasm")));
  const memory = new WebAssembly.Memory({ initial: 256, maximum: 32768, shared: true });
  const ex = Object.assign({}, (await WebAssembly.instantiate(mod, { env: { memory } })).exports, { memory });
  assert.equal(ex.mw_version(), MAGIC);
  const runs = [];
  for (const T of [1, 4]) {
    assert.equal(ex.mw_threads(T), T);
    const g = mwGame(ex, 12);
    assert.equal(g.run(), 40);
    runs.push(g.read());
  }
  assert.equal(ex.mw_helped(), 0);   // (no helpers yet: this thread ran every leaf)
  assert.ok(runs[0].some((x) => x !== 0));
  assert.deepEqual(runs[1], runs[0]);
  const page = helperPage();
  const helpers = [0, 1, 2].map((i) => new Worker(page, { eval: true, workerData: { mod, memory, i, pre: "mw_" } }));
  try {
    // (a run takes milliseconds: on a busy machine it can be over before a helper is scheduled, so again until one ran)
    for (let k = 0; k < 500 && !ex.mw_helped(); k++) {
      const g = mwGame(ex, 12);
      assert.equal(g.run(), 40);
      assert.deepEqual(g.read(), runs[0]);
      if (k % 10 === 9) await new Promise((ok) => setTimeout(ok, 10));
    }
    assert.ok(ex.mw_helped() > 0, "the helpers ran no leaves");
    const dead = new Int32Array(memory.buffer, ex.mw_dead(0), 1);
    assert.equal(Atomics.load(dead, 0), 0);
  } finally {
    await Promise.all(helpers.map((w) => w.terminate()));
  }
});

// Helper 0 fails (mw_fail, a test's: a trap holding a task) on 2 threads and on 8, the solving thread a worker thread of
// its own, so a solve that waits for good fails the test at the time limit instead of hanging it: its mw_run throws
// (gto/mwcore.py: Failed, and numpy runs the solve).
const SOLVER = `const { parentPort, workerData: { mod, memory, T } } = require("node:worker_threads");
${mwGame.toString()}
const ex = Object.assign({}, new WebAssembly.Instance(mod, { env: { memory } }).exports, { memory });
ex.mw_threads(T);
const pause = new Int32Array(new SharedArrayBuffer(4));
const dead = new Int32Array(memory.buffer, ex.mw_dead(0), 1);
let runs = 0, threw = null;
ex.mw_fail(0, 1);
for (; runs < 2000 && !threw; runs++) {
  const g = mwGame(ex, 12);
  try { g.run(); } catch (e) { threw = String(e); }
  if (runs % 10 === 9) Atomics.wait(pause, 0, 0, 5);   // (the helpers scheduled)
}
parentPort.postMessage({ runs, threw, dead: Atomics.load(dead, 0), helped: ex.mw_helped() });`;

test("mwcore-mt: a helper that fails ends the solve's wait instead of hanging it", async () => {
  const { Worker } = require("node:worker_threads");
  const page = helperPage();
  const mod = await WebAssembly.compile(fs.readFileSync(path.join(ROOT, "gto", "mwcore-mt.wasm")));
  for (const T of [2, 8]) {
    const memory = new WebAssembly.Memory({ initial: 256, maximum: 32768, shared: true });
    const helpers = [...Array(T - 1).keys()].map((i) => new Worker(page, { eval: true, workerData: { mod, memory, i, pre: "mw_" } }));
    for (const w of helpers) w.on("error", () => {});
    const solver = new Worker(SOLVER, { eval: true, workerData: { mod, memory, T } });
    let timer;
    try {
      const got = await Promise.race([
        new Promise((ok, no) => { solver.once("message", ok); solver.once("error", no); }),
        new Promise((ok) => { timer = setTimeout(() => ok({ hung: true }), 60000); }),
      ]);
      assert.ok(!got.hung, `${T} threads: the solve waited for good on the failed helper's task`);
      assert.match(got.threw || "", /unreachable/, `${T} threads: no run failed (${got.runs} runs, ${got.helped} leaves helped)`);
      assert.equal(got.dead, 1, `${T} threads: helper 0 marked itself`);
    } finally {
      clearTimeout(timer);
      await Promise.all([solver, ...helpers].map((w) => w.terminate()));
    }
  }
});

// After a trap the core starts again (gto/core.py and gto/mwcore.py restart: worker.js's start, ex.reload): a new
// instance on a fresh memory, the old helpers terminated and new ones started on it, and the page's handle (self.mwCore)
// the new one. Helper 0 of the first fails, its solve throws; the reloaded one solves clean, its helpers running leaves,
// the same numbers as one thread. (In a worker thread of its own, so that a wait for good fails at the time limit.)
const RELOAD = `const { parentPort, workerData: { mod, page, start } } = require("node:worker_threads");
${mwGame.toString()}
const { Worker: Thread } = require("node:worker_threads");
let made = 0, terminated = 0;
// a browser Worker on worker_threads: the helper script run with what start posts it
class Worker {
  constructor(url) { made++; }
  postMessage(data) { this.t = new Thread(page, { eval: true, workerData: data }); this.t.on("error", () => {}); }
  terminate() { terminated++; if (this.t) this.t.terminate(); }
}
const self = globalThis, HELPERS = 3, HELPER = "";
const URL = { createObjectURL: () => "blob:helper" }, Blob = class {};
let helperURL = null;
const go = new Function("Worker", "self", "HELPERS", "HELPER", "URL", "Blob", "helperURL",
  start + "\\nreturn start;")(Worker, self, HELPERS, HELPER, URL, Blob, helperURL);
const pause = new Int32Array(new SharedArrayBuffer(4));
const ref = (() => {   // one thread
  const memory = new WebAssembly.Memory({ initial: 256, maximum: 32768, shared: true });
  const ex = Object.assign({}, new WebAssembly.Instance(mod, { env: { memory } }).exports, { memory });
  ex.mw_threads(1);
  const g = mwGame(ex, 12);
  g.run();
  return g.read();
})();
const first = self.mwCore = go(mod, "mw_", true, "mwCore");
first.mw_fail(0, 1);
let threw = null, runs = 0;
for (; runs < 2000 && !threw; runs++) {
  try { mwGame(first, 12).run(); } catch (e) { threw = String(e); }
  if (runs % 10 === 9) Atomics.wait(pause, 0, 0, 5);
}
const dead = Atomics.load(new Int32Array(first.memory.buffer, first.mw_dead(0), 1), 0);
const helpersBefore = made;
const second = first.reload();
const out = { threw, dead, helpersBefore, terminated, made, handle: self.mwCore === second,
              fresh: second.memory !== first.memory && second.memory.buffer !== first.memory.buffer, same: true };
for (let k = 0; k < 500 && (k < 3 || !second.mw_helped()); k++) {
  const g = mwGame(second, 12);
  if (g.run() !== 40) out.same = false;
  const got = g.read();
  if (got.length !== ref.length || got.some((x, i) => x !== ref[i])) out.same = false;
  if (k % 10 === 9) Atomics.wait(pause, 0, 0, 5);
}
out.helped = second.mw_helped();
out.dead2 = Atomics.load(new Int32Array(second.memory.buffer, second.mw_dead(0), 1), 0);
parentPort.postMessage(out);`;

test("worker.js's start: a reload is a new instance on a fresh memory with new helpers, and it solves clean", async () => {
  const { Worker } = require("node:worker_threads");
  const src = fs.readFileSync(path.join(ROOT, "gto", "static", "worker.js"), "utf8");
  const start = src.match(/^function start\(mod[\s\S]*?\n}\n/m)[0];
  const mod = await WebAssembly.compile(fs.readFileSync(path.join(ROOT, "gto", "mwcore-mt.wasm")));
  const solver = new Worker(RELOAD, { eval: true, workerData: { mod, page: helperPage(), start } });
  let timer;
  try {
    const got = await Promise.race([
      new Promise((ok, no) => { solver.once("message", ok); solver.once("error", no); }),
      new Promise((ok) => { timer = setTimeout(() => ok({ hung: true }), 90000); }),
    ]);
    assert.ok(!got.hung, "a solve waited for good");
    assert.match(got.threw || "", /unreachable/, "the first instance's failing helper failed no solve");
    assert.equal(got.dead, 1);
    assert.equal(got.helpersBefore, 3);
    assert.equal(got.terminated, 3);     // (the first instance's helpers)
    assert.equal(got.made, 6);           // (and three new ones)
    assert.ok(got.handle, "self.mwCore isn't the new instance");
    assert.ok(got.fresh, "the new instance shares the old memory");
    assert.ok(got.same, "the new instance's solves differ from one thread's");
    assert.ok(got.helped > 0, "the new helpers ran no leaves");
    assert.equal(got.dead2, 0);
  } finally {
    clearTimeout(timer);
    await solver.terminate();
  }
});
