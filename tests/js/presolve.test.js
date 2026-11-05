// Pre-solves are gone from the main screen (review first): while a hand is entered the worker is asked nothing, whoever
// is to act, so nothing solves ahead of your decision. What stays from these tests: a stopped solve that never answers doesn't hold the next request up.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const flush = () => new Promise(r => setImmediate(r));
// BTN (you) vs BB, single-raised, BB to act first on the flop
const OOP_TO_ACT = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c") };

async function solving(opts = { isolated: true }) {
  const p = loadPage(opts);
  await p.ready();
  p.workers[0].onmessage({ data: { status: "Solving…" } });   // (the runtime is up)
  const flag = opts.isolated ? new Int32Array(p.posted.find((m) => m.stop).stop) : null;
  const pres = () => p.posted.filter((m) => m.presolve);
  const asks = () => p.posted.filter((m) => m.state && !m.presolve);
  return { p, flag, pres, asks };
}

test("your opponent to act: the worker is asked nothing, and your move asks nothing either", async () => {
  const { p, pres, asks } = await solving();
  p.S = state(OOP_TO_ACT);
  await flush();
  p.run("renderAll()");
  await flush();
  p.act({ action: "check" });   // BB checks: your turn, heads-up on the flop, where a presolve went on to your advice
  await flush();
  assert.deepEqual([pres().length, asks().length], [0, 0]);
  assert.match(p.byId("result").innerHTML, /Finish the hand to see how each decision compares with the solver\./);
});

test("hidden and shown again, or left: nothing starts", async () => {
  const { p, flag, pres } = await solving();
  p.S = state(OOP_TO_ACT);
  await flush();
  p.setHidden(true);
  await flush();
  p.setHidden(false);
  await flush();
  assert.equal(pres().length, 0);
  assert.deepEqual([...flag], [0, 0]);
  assert.equal(p.ctx.presolveFor, undefined);   // (the page has no pre-solve left to start)
});

test("a stopped solve that never answers (stuck where it can't read the stop): after 15 s the worker is restarted and the advice waiting behind it goes to the new one", async () => {
  for (const stuck of [false, true]) {
    const { p, asks } = await solving();
    const timers = [];
    p.ctx.setTimeout = (f, ms) => { timers.push({ f, ms }); return timers.length; };
    p.spot(state({ ...OOP_TO_ACT, flop: acts("BB x") }));
    await flush();
    const first = asks().at(-1);
    p.spot(state({ ...OOP_TO_ACT, flop: acts("BB b2") }));   // another decision: stops that solve and asks for this one, which waits behind it
    await flush();
    const ask = asks().at(-1), pings = p.posted.filter((m) => m.ping != null);
    assert.notEqual(ask.id, first.id);
    assert.equal(pings.length, 1);
    const watch = timers.filter((t) => t.ms === 15000);
    assert.equal(watch.length, 1);
    if (!stuck) p.workers[0].onmessage({ data: { pong: pings[0].ping } });   // (the worker went on: it isn't stuck)
    const before = p.posted.length;
    watch[0].f();
    await flush();
    if (!stuck) {
      assert.equal(p.workers.length, 1);
      assert.equal(p.workers[0].terminated, false);
      continue;
    }
    assert.equal(p.workers.length, 2);
    assert.equal(p.workers[0].terminated, true);
    const again = p.posted.slice(before);
    assert.ok(again.some((m) => m.stop));   // (the new worker's own stop flag)
    assert.deepEqual(again.filter((m) => m.state).map((m) => m.id), [ask.id]);   // the advice, not the solve stopped
    assert.equal(again.find((m) => m.state).early, true);
    assert.ok(!again.some((m) => m.id === first.id));
    p.workers[1].onmessage({ data: { status: "Solving…" } });
    p.workers[1].onmessage({ data: { id: ask.id, result: { street: "flop", hero: "BTN", hand: "AKs", recommended: "check", actions: [{ action: "check", freq: 1 }] } } });
    await flush();
    assert.match(p.byId("result").innerHTML, /<span class="big">check<\/span>/);
  }
});
