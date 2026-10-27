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
