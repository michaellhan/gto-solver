// The compiled CFR core's browser module (gto/core.wasm, built by scripts/build_core.py from core/gtocore.c): built from
// the current source, its exports, and a game small enough to write by hand solved and evaluated in it (the Python
// tests check the native build of the same source against numpy on real trees).
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..", "..");
const MAGIC = 0x67746f31;
const load = async () => (await WebAssembly.instantiate(fs.readFileSync(path.join(ROOT, "gto", "core.wasm")))).instance.exports;

test("built from the current source", async () => {
  const ex = await load();
  const sha = crypto.createHash("sha1").update(fs.readFileSync(path.join(ROOT, "core", "gtocore.c"))).digest("hex");
  const at = ex.core_source();
  assert.equal(Buffer.from(new Uint8Array(ex.memory.buffer, at, 40)).toString(), sha);
  assert.equal(ex.core_version(), MAGIC);
});

test("solves and evaluates a hand-written game", async () => {
  // two combos sharing no card; player 0 picks between two folds, one paying it 1 and one -1
  const ex = await load();
  ex.core_reset();
  const base = ex.core_base() >>> 0;
  const put = (words, Arr) => {
    const off = ex.core_alloc(words.length * 4) >>> 0;
    new Arr(ex.memory.buffer, base + off, words.length).set(words);
    return off;
  };
  const f = (x) => new Int32Array(new Float32Array([x]).buffer)[0];
  const ca = put([0, 2], Int32Array), cb = put([1, 3], Int32Array);
  const r0 = put([1, 1], Float32Array), r1 = put([1, 1], Float32Array);
  const regret = put([0, 0, 0, 0], Float32Array), ssum = put([0, 0, 0, 0], Float32Array);
  const prog = put([MAGIC, 2, ca, cb, 10, r0, r1, regret, ssum, 4,
                    1, 0, 2, 1, 2, regret, ssum, 19, 23,      // DECISION: player 0, 2 actions, 1 board, 2 combos
                    2, 1, f(1), f(-1),                          // FOLD: player 0 wins 1
                    2, 1, f(-1), f(1)], Int32Array);           // FOLD: player 0 loses 1
  const fac = put(new Array(40).fill(0).map((_, i) => (i % 4 === 0 ? 0 : 1)), Float32Array);   // no discounting
  assert.equal(ex.core_run(prog, 10, fac), 10);
  const ss = new Float32Array(ex.memory.buffer, base + ssum, 4);
  // uniform on the first iteration, then all on the winning fold
  assert.deepEqual([...ss], [9.5, 9.5, 0.5, 0.5]);
  const out = ex.core_alloc(8) >>> 0;
  assert.equal(ex.core_eval(prog, 10, 0, r0, r1, out, 1), 0);    // average strategies: 0.95 - 0.05
  assert.deepEqual([...new Float32Array(ex.memory.buffer, base + out, 2)].map((x) => +x.toFixed(5)), [0.9, 0.9]);
  assert.equal(ex.core_eval(prog, 10, 0, r0, r1, out, 2), 0);    // best response
  assert.deepEqual([...new Float32Array(ex.memory.buffer, base + out, 2)], [1, 1]);
  assert.equal(ex.core_run(regret, 1, fac), -1);                 // not a program
});

// gto/core-mt.wasm, the same with threads: its memory shared with helpers (worker.js's, here worker_threads), which take
// rows of each chance node's subtree. A game with a chance node of 40 rows comes out the same, bit for bit, on 1
// thread and on 4, and the helpers ran some of its rows.
const HELPER = `const { parentPort, workerData: { mod, memory, i } } = require("node:worker_threads");
const ex = new WebAssembly.Instance(mod, { env: { memory } }).exports;
ex.__stack_pointer.value = ex.core_stack(i);
parentPort.postMessage("ready");
ex.core_helper(i);`;

// A deterministic game: player 0 folds (+-0.5) or deals one of K cards (masks block some combos on some cards), after
// which player 1 picks one of two folds; n combos sharing no card.
function chanceGame(ex, n, K) {
  ex.core_reset();
  const base = ex.core_base() >>> 0;
  const put = (words, Arr) => {
    const off = ex.core_alloc(words.length * 4) >>> 0;
    new Arr(ex.memory.buffer, base + off, words.length).set(words);
    return off;
  };
  const f = (x) => new Int32Array(new Float32Array([x]).buffer)[0];
  const ca = put([...Array(n).keys()].map((i) => 2 * i), Int32Array), cb = put([...Array(n).keys()].map((i) => 2 * i + 1), Int32Array);
  const r0 = put([...Array(n).keys()].map((i) => 1 + (i % 3)), Float32Array);
  const r1 = put([...Array(n).keys()].map((i) => 1 + ((i * 7) % 5)), Float32Array);
  const len = 2 * n + K * 2 * n;
  const regret = put(new Array(len).fill(0), Float32Array), ssum = put(new Array(len).fill(0), Float32Array);
  const masks = [];
  for (let k = 0; k < K; k++) for (let i = 0; i < n; i++) masks.push((i + 2 * k) % 7 === 0 ? 0 : 1);
  const norm = [...Array(n).keys()].map((i) => masks.filter((_, x) => x % n === i).reduce((a, b) => a + b, 0));
  const m = put(masks, Float32Array), nm = put(norm, Float32Array);
  const root = 10, fold = root + 9, ch = fold + 4, dec = ch + 7, fa = dec + 9, fb = fa + 4;
  const prog = put([MAGIC, n, ca, cb, root, r0, r1, regret, ssum, len,
                    1, 0, 2, 1, n, regret, ssum, ch, fold,                       // DECISION: player 0, deal or fold
                    2, 1, f(0.5), f(-0.5),                                      // FOLD
                    5, 1, K, dec, m, m, nm,                                     // CHANCE: K cards
                    1, 1, 2, K, n, regret + 8 * n, ssum + 8 * n, fa, fb,        // DECISION: player 1, K rows
                    2, K, f(1), f(-1),                                          // FOLD
                    2, K, f(-2), f(2)], Int32Array);
  const fac = put(new Array(4 * 200).fill(0).map((_, i) => [0, 1, 0.5, 0.9][i % 4]), Float32Array);
  return { prog, fac, root, read: () => [...new Int32Array(ex.memory.buffer, base + regret, 2 * len)] };
}

test("core-mt: threads give the same numbers, and the helpers run rows", async () => {
  const { Worker } = require("node:worker_threads");
  const mod = await WebAssembly.compile(fs.readFileSync(path.join(ROOT, "gto", "core-mt.wasm")));
  const memory = new WebAssembly.Memory({ initial: 256, maximum: 32768, shared: true });
  const ex = Object.assign({}, (await WebAssembly.instantiate(mod, { env: { memory } })).exports, { memory });
  const sha = crypto.createHash("sha1").update(fs.readFileSync(path.join(ROOT, "core", "gtocore.c"))).digest("hex");
  assert.equal(Buffer.from(new Uint8Array(memory.buffer, ex.core_source(), 40)).toString(), sha);
  const runs = [];
  for (const T of [1, 4]) {
    assert.equal(ex.core_threads(T), T);
    const g = chanceGame(ex, 24, 40);
    assert.equal(ex.core_run(g.prog, 200, g.fac), 200);
    runs.push(g.read());
  }
  assert.equal(ex.core_helped(), 0);   // (no helpers yet: this thread ran every row)
  const helpers = [1, 2, 3].map((i) => new Worker(HELPER, { eval: true, workerData: { mod, memory, i: i - 1 } }));
  try {
    await Promise.all(helpers.map((w) => new Promise((ok) => w.once("message", ok))));
    // (a run takes milliseconds: on a busy machine it can be over before a helper is scheduled, so again until one ran rows)
    for (let k = 0; k < 500 && !ex.core_helped(); k++) {
      const g = chanceGame(ex, 24, 40);
      assert.equal(ex.core_run(g.prog, 200, g.fac), 200);
      assert.deepEqual(g.read(), runs[0]);
      if (k % 10 === 9) await new Promise((ok) => setTimeout(ok, 10));
    }
    assert.ok(ex.core_helped() > 0, "the helpers ran no rows");
  } finally {
    await Promise.all(helpers.map((w) => w.terminate()));
  }
  assert.ok(runs[0].some((x) => x !== 0));
  assert.deepEqual(runs[1], runs[0]);
});

// A helper that fails (a trap in its rows: a bug, say, that runs past its lane) must not leave the solve waiting for
// good on the rows it held. worker.js's own helper script marks it in the core's memory, the thread waiting on the job's
// rows traps on that, and its core_run throws (gto/core.py: CoreError, and numpy runs the solve). Here the core fails
// helper 0 (core_fail, a test's) on its first chunk, on 2 threads and on 8, the solving thread a worker thread of its
// own, so a solve that waits for good fails the test at the time limit instead of hanging it.
const SOLVER = `const { parentPort, workerData: { mod, memory, T } } = require("node:worker_threads");
const MAGIC = ${MAGIC};
${chanceGame.toString()}
const ex = Object.assign({}, new WebAssembly.Instance(mod, { env: { memory } }).exports, { memory });
ex.core_threads(T);
const pause = new Int32Array(new SharedArrayBuffer(4));
const dead = new Int32Array(memory.buffer, ex.core_dead(0), 1);
let runs = 0, threw = null;
ex.core_fail(0, 1);
for (; runs < 2000 && !threw; runs++) {
  const g = chanceGame(ex, 24, 40);
  try { ex.core_run(g.prog, 50, g.fac); } catch (e) { threw = String(e); }
  if (runs % 10 === 9) Atomics.wait(pause, 0, 0, 5);   // (the helpers scheduled)
}
parentPort.postMessage({ runs, threw, dead: Atomics.load(dead, 0), helped: ex.core_helped() });`;

test("core-mt: a helper that fails ends the solve's wait instead of hanging it (worker.js's helper)", async () => {
  const { Worker } = require("node:worker_threads");
  const src = fs.readFileSync(path.join(ROOT, "gto", "static", "worker.js"), "utf8");
  const helper = src.match(/const HELPER = `([\s\S]*?)`;/)[1];
  // (the page's helper is a browser worker: its onmessage, called here with what worker.js posts it)
  const HELPER_PAGE = `const { workerData } = require("node:worker_threads");\n${helper}\nonmessage({ data: workerData });`;
  const mod = await WebAssembly.compile(fs.readFileSync(path.join(ROOT, "gto", "core-mt.wasm")));
  for (const T of [2, 8]) {
    const memory = new WebAssembly.Memory({ initial: 256, maximum: 32768, shared: true });
    const helpers = [...Array(T - 1).keys()].map((i) => new Worker(HELPER_PAGE, { eval: true, workerData: { mod, memory, i, pre: "core_" } }));
    for (const w of helpers) w.on("error", () => {});   // (a helper's own error goes nowhere in the browser either)
    const solver = new Worker(SOLVER, { eval: true, workerData: { mod, memory, T } });
    let timer;
    try {
      const got = await Promise.race([
        new Promise((ok, no) => { solver.once("message", ok); solver.once("error", no); }),
        new Promise((ok) => { timer = setTimeout(() => ok({ hung: true }), 60000); }),
      ]);
      assert.ok(!got.hung, `${T} threads: the solve waited for good on the failed helper's rows`);
      assert.match(got.threw || "", /unreachable/, `${T} threads: no run failed (${got.runs} runs, ${got.helped} rows helped)`);
      assert.equal(got.dead, 1, `${T} threads: helper 0 marked itself`);
    } finally {
      clearTimeout(timer);
      await Promise.all([solver, ...helpers].map((w) => w.terminate()));
    }
  }
});
