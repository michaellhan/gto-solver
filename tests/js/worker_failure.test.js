// The page when its solver won't load: a plain message and a Try again that starts a new worker, in place of "Loading…"
// for good, wherever a request was waiting (the advice, the equity, a whole-hand review), and for a load that goes quiet.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const MESSAGE = "The solver couldn't load. Check your connection and try again.";
const flush = async () => { for (let i = 0; i < 6; i++) await new Promise(r => setImmediate(r)); };
async function asking(extra = {}) {
  const p = loadPage({ protocol: "https:", isolated: true, ...extra });
  await p.ready();
  p.workers.at(-1).onmessage({ data: { status: "Loading runtime…" } });
  return p;
}
const w = (p) => p.workers.at(-1);
const result = (p) => p.byId("result").innerHTML;

test("the worker says its load failed: the message, and Try again starts a new one", async () => {
  const p = await asking();
  const n = p.workers.length, first = w(p);
  first.onmessage({ data: { loadFailed: "NetworkError" } });
  await flush();
  assert.equal(first.terminated, true);
  assert.equal(p.val("worker"), null);
  p.run("lastKey = null; resultKey = null; renderResult({ error: LOAD_FAILED, loadFailed: true });");
  assert.ok(result(p).includes(MESSAGE));
  assert.match(result(p), /id="solverRetry"[^>]*>Try again</);
  assert.equal(p.workers.length, n);          // (nothing restarts until it's asked to)
  p.byId("solverRetry").onclick();
  assert.equal(p.workers.length, n + 1);
  assert.equal(p.val("loadFailed"), false);
  assert.equal(w(p).terminated, false);
  w(p).onmessage({ data: { status: "Solving…" } });
  assert.equal(p.val("workerStatus"), "Solving…");   // (the string itself: Lizard's API)
});

test("a boot error with no word from the worker (its script wouldn't load) is the same failure", async () => {
  const p = await asking();
  let answer = null;
  p.ctx.__got = (r) => { answer = r; };
  p.run('pending[77] = __got;');
  w(p).onerror({ message: "NetworkError" });
  assert.equal(p.val("worker"), null);
  assert.equal(answer.error, MESSAGE);
  assert.equal(answer.loadFailed, true);
  assert.equal(p.val("Object.keys(pending).length"), 0);
});

test("after the runtime is up an error is still the solve's own (no restart, no load message)", async () => {
  const p = await asking();
  w(p).onmessage({ data: { status: "Solving…" } });
  let answer = null;
  p.ctx.__got = (r) => { answer = r; };
  p.run('pending[78] = __got;');
  w(p).onerror({ message: "out of memory" });
  assert.equal(answer.error, "solver worker failed: out of memory");
  assert.notEqual(p.val("worker"), null);
});

test("a load that goes quiet for 3 minutes gives up; each word from the worker restarts the clock; once running it's off", async () => {
  let timers = [];
  const p = await asking();
  p.ctx.setTimeout = (f, ms) => { timers.push({ f, ms }); return timers.length; };
  p.ctx.clearTimeout = () => {};
  w(p).onmessage({ data: { status: "Loading runtime…" } });
  const t = timers.at(-1);
  assert.equal(t.ms, 180000);
  w(p).onmessage({ data: { status: "Loading solver…" } });
  const t2 = timers.at(-1);
  assert.notEqual(t2, t);
  const running = w(p);
  t2.f();
  assert.equal(running.terminated, true);
  assert.equal(p.val("loadFailed"), true);
  // a fresh worker that reaches Solving… arms no clock
  p.run("startWorker()");
  timers = [];
  w(p).onmessage({ data: { status: "Solving…" } });
  assert.equal(timers.filter(x => x.ms === 180000).length, 0);
  // and an old worker's clock doesn't touch the new one
  t.f();
  assert.notEqual(p.val("worker"), null);
});

test("a view over the page (My hands) gets the toast, with its own Try again above the view", async () => {
  const p = await asking();
  p.ctx.__got = () => {};
  p.run('pending[79] = __got; overlay = "hands"; document.getElementById("toast").hidden = true;');
  w(p).onmessage({ data: { loadFailed: "x" } });
  const toast = p.byId("toast");
  assert.ok(toast.textContent.includes(MESSAGE) || toast.innerHTML.includes(MESSAGE));
  assert.equal(toast.style.zIndex, 100);
  const n = p.workers.length;
  toast.children.at(-1).onclick();
  assert.equal(p.workers.length, n + 1);
  assert.equal(toast.hidden, true);
});

test("the library's prefetch doesn't restart a worker that failed", async () => {
  const p = await asking();
  w(p).onmessage({ data: { loadFailed: "x" } });
  const n = p.workers.length;
  p.run("prefetchLibrary()");
  assert.equal(p.workers.length, n);
});

const OPENED = acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c");
const WHOLE = state({ cards: ["Ah", "Kh", "Kd", "7s", "2c", "9h", "3d"], preflop: OPENED, flop: acts("BB x, BTN b3, BB c"), turn: acts("BB x, BTN x"), river: acts("BB b8, BTN c") });

test("a review waiting on the solver stops at the failure, keeps the hand, and Try again runs the review again", async () => {
  const p = await asking();
  p.S = WHOLE;
  const done = p.run("reviewHand()");
  await flush();
  assert.ok(p.posted.some(m => m.state));
  w(p).onmessage({ data: { loadFailed: "x" } });
  await done; await flush();
  assert.equal(p.val("review"), null);
  assert.equal(p.S.cards[6], "3d");           // (the whole hand back on the table)
  assert.ok(result(p).includes(MESSAGE));
  assert.equal(p.run("document.body.dataset.s"), "review");   // (a phone keeps the panel and its message up, not under the table)
  assert.equal(p.byId("doneNew").hidden, true);               // (Try again is what's asked; New hand is on the setup line)
  const n = p.workers.length, before = p.posted.length;
  p.byId("solverRetry").onclick();
  await flush();
  assert.equal(p.workers.length, n + 1);
  assert.ok(p.val("review") !== null);        // (the review, from its first decision)
  assert.ok(p.posted.length > before);
});

test("the equity panel: the same message, and its Try again asks again", async () => {
  const p = loadPage({ protocol: "https:", isolated: true });
  await p.ready();
  p.S = { ...state({ cards: ["Ah", "Kd", "Ks", "7d", "2c"], preflop: OPENED, flop: acts("BB x, BTN x") }), finished: true, reviewed: true };
  await flush();
  const asks = () => p.posted.filter(m => m.state && m.state.equity);
  assert.equal(asks().length, 1);
  w(p).onmessage({ data: { loadFailed: "x" } });
  await flush();
  assert.ok(p.byId("equity").innerHTML.includes(MESSAGE));
  assert.match(p.byId("equity").innerHTML, /id="equityRetry"[^>]*>Try again</);
  const n = p.workers.length;
  p.byId("equityRetry").onclick();
  await flush();
  assert.equal(p.workers.length, n + 1);
  assert.equal(asks().length, 2);
});
