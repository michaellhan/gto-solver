// The browser worker's request loop (gto/static/worker.js), run in a vm with a stand-in for Pyodide: what it posts for
// a request marked "early" (the page's own advice): the approximate answer at once (early()), then the street's own
// solve so far at each of its looks (progress(): Advisor.progress), then the answer.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.join(__dirname, "..", "..");
const SRC = fs.readFileSync(path.join(ROOT, "gto", "static", "worker.js"), "utf8");
const tick = () => new Promise((ok) => setTimeout(ok, 0));
const plain = (x) => JSON.parse(JSON.stringify(x));   // (the vm's objects, compared as this realm's)

// worker.js with `answers` standing in for the Python side: by the call's name (early, progress, answer, presolve), a
// list of what each call returns in turn (a JSON string; '' for none). `ran`: every runPython's code.
function loadWorker(answers) {
  const posted = [], ran = [];
  const py = {
    globals: { set() {} },
    FS: { readFile() { throw new Error("no core here"); } },   // (loadCore: no compiled core, numpy answers)
    loadPackage: async () => {},
    unpackArchive() {},
    runPython(code) {
      ran.push(code);
      const m = code.match(/^(\w+)\(json\.loads/);
      if (m) return (answers[m[1]] || []).shift() ?? "";
      return code.includes("warm_step") ? true : undefined;
    },
  };
  const ctx = {
    console, performance, setTimeout, clearTimeout, URLSearchParams, Promise, JSON, Map, Set, Math, String, Error,
    fetch: async () => ({ ok: true, status: 200, arrayBuffer: async () => new ArrayBuffer(0) }),
    importScripts() {},
    loadPyodide: async () => py,
    postMessage: (m) => posted.push(m),
  };
  ctx.self = ctx;
  vm.createContext(ctx);
  vm.runInContext(SRC, ctx);
  return { posted, ran, send: (data) => ctx.onmessage({ data }) };
}

const STATE = { board: "Ks7d2cTh", hero: "BB", hole: "QhJh" };
const part = (iters) => JSON.stringify({ partial: { recommended: "check", approximate: { source: "solve", iterations: iters } } });

test("a live street's looks are posted as partials, then the answer progress() came to", async () => {
  const w = loadWorker({ early: [JSON.stringify({ approximate: { source: "store" } })],
                         progress: [part(100), part(300), JSON.stringify({ result: { recommended: "bet 75%" } })] });
  await w.send({ id: 7, state: STATE, early: true });
  const mine = w.posted.filter((m) => m.id === 7);
  assert.deepEqual(mine.map((m) => m.partial ? m.partial.approximate.source + (m.partial.approximate.iterations || "") : m.result.recommended),
                   ["store", "solve100", "solve300", "bet 75%"]);
  assert.equal(w.ran.filter((c) => c.startsWith("answer(")).length, 0);   // (progress answered: not solved again)
});

test("where progress() has nothing (no look, an error) the answer is asked for as before; a play or an ask not marked early never looks", async () => {
  const w = loadWorker({ early: [""], progress: [""], answer: [JSON.stringify({ recommended: "check" })] });
  await w.send({ id: 1, state: STATE, early: true });
  assert.deepEqual(plain(w.posted.filter((m) => m.id === 1)), [{ id: 1, result: { recommended: "check" } }]);
  const v = loadWorker({ progress: [part(100)], answer: [JSON.stringify({ recommended: "fold" }), JSON.stringify({ action: "call" })] });
  await v.send({ id: 2, state: STATE });
  await v.send({ id: 3, state: STATE, early: true, play: { seat: "BTN", hole: "AhAd" } });
  assert.deepEqual(plain(v.posted.filter((m) => m.id)), [{ id: 2, result: { recommended: "fold" } }, { id: 3, result: { action: "call" } }]);
  assert.equal(v.ran.filter((c) => c.startsWith("progress(")).length, 0);
});

test("at most four looks, then the answer", async () => {
  const w = loadWorker({ progress: Array(6).fill(part(100)), answer: [JSON.stringify({ recommended: "check" })] });
  await w.send({ id: 4, state: STATE, early: true });
  const mine = w.posted.filter((m) => m.id === 4);
  assert.equal(mine.filter((m) => m.partial).length, 4);
  assert.deepEqual(plain(mine.at(-1)), { id: 4, result: { recommended: "check" } });
});

test("the speeds' looks: Fast at the 100th iteration, Accurate at the 100th and 300th (flop and turn)", async () => {
  const w = loadWorker({});
  await w.send({ ping: 1 });
  for (let k = 0; k < 5 && !w.ran.some((c) => c.includes("first_look")); k++) await tick();
  const cfg = w.ran.find((c) => c.includes("first_look"));
  assert.ok(cfg.includes("advs['fast'].first_look = {'flop': (100,), 'turn': (100,)}"));
  assert.ok(cfg.includes("advs['accurate'].first_look = {'flop': (100, 300), 'turn': (100, 300)}"));
  assert.match(w.ran.find((c) => c.includes("def progress")), /def progress\(st\):\n    try:\n        r = advs\.get\(st\.get\('speed'\), advs\['fast'\]\)\.progress\(st\)\n    except Cancelled:\n        r = \{'result': \{'cancelled': True\}\}\n    except Exception:\n        r = None/);
});
