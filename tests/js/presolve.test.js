// Speculative pre-solves (index.html presolveFor): while your opponent is to act heads-up after the flop, the page asks
// the browser's worker for the solve your next decision reads ({id, state, presolve: seat}: Advisor.presolve), one at a
// time, only while the page is visible and nothing else is asked, and stops it (the shared stop flag) the moment you
// enter their move, hide the tab or leave; its answer is never shown.
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

test("your opponent to act: the worker presolves the street with your cards, once, and your move stops it", async () => {
  const { p, flag, pres, asks } = await solving();
  p.S = state(OOP_TO_ACT);
  await flush();
  assert.equal(pres().length, 1);
  const pre = pres()[0];
  assert.equal(pre.presolve, "BB");
  assert.deepEqual(pre.state, p.payload());
  assert.equal(pre.early, undefined);
  assert.equal(asks().length, 0);
  p.run("renderAll()");
  await flush();
  assert.equal(pres().length, 1);   // (in flight: not asked again)
  // BB checks: the presolve is stopped through the shared flag, then your advice is asked for
  p.act({ action: "check" });
  await flush();
  assert.equal(flag[1], pre.id);
  assert.equal(flag[0], 0);
  const ask = asks().at(-1);
  assert.ok(ask.id > pre.id);
  assert.equal(ask.early, true);
  assert.equal(pres().length, 1);
  // its late reply is dropped; the advice's is shown
  p.workers[0].onmessage({ data: { id: pre.id, result: { cancelled: true } } });
  p.workers[0].onmessage({ data: { id: ask.id, result: { street: "flop", hero: "BTN", hand: "AKs", recommended: "check", actions: [{ action: "check", freq: 1 }] } } });
  await flush();
  assert.match(p.byId("result").innerHTML, /<span class="big">check<\/span>/);
});

test("a presolve that ran its budget isn't asked again; one stopped is, when the spot comes back", async () => {
  const { p, pres } = await solving();
  p.S = state(OOP_TO_ACT);
  await flush();
  p.workers[0].onmessage({ data: { id: pres()[0].id, result: { presolved: "flop", iterations: 300 } } });
  await flush();
  p.run("renderAll()");
  await flush();
  assert.equal(pres().length, 1);
  assert.equal(p.byId("result").innerHTML.includes("300"), false);   // (never shown)
  // the turn, BB to act: a new one; stopped by a new hand, then the spot again: asked again
  p.S = state({ ...OOP_TO_ACT, cards: [...OOP_TO_ACT.cards, "9c"], flop: acts("BB x, BTN b2, BB c") });
  await flush();
  assert.equal(pres().length, 2);
  assert.equal(pres()[1].state.board, "Kd7s2c9c");
  p.workers[0].onmessage({ data: { id: pres()[1].id, result: { cancelled: true } } });
  await flush();
  p.run("renderAll()");
  await flush();
  assert.equal(pres().length, 3);
});

test("only while the page is visible: hidden or left, it's stopped; shown again, it starts again", async () => {
  const { p, flag, pres } = await solving();
  p.S = state(OOP_TO_ACT);
  await flush();
  const first = pres()[0];
  p.setHidden(true);
  assert.equal(flag[1], first.id);
  p.run("renderAll()");
  await flush();
  assert.equal(pres().length, 1);   // (nothing while hidden)
  p.setHidden(false);
  await flush();
  assert.equal(pres().length, 2);
  p.fire("pagehide");
  assert.equal(flag[1], pres()[1].id);
});

test("never while another request is asked, and not where it doesn't apply", async () => {
  const { p, pres, asks } = await solving();
  // your own advice still solving when you act: it runs on (the street's solve), no presolve beside it
  p.S = state({ ...OOP_TO_ACT, flop: acts("BB x") });
  await flush();
  const ask = asks().at(-1);
  p.act({ action: "bet", to: 3 });
  await flush();
  assert.equal(pres().length, 0);
  p.workers[0].onmessage({ data: { id: ask.id, result: { street: "flop", hero: "BTN", hand: "AKs", recommended: "check", actions: [{ action: "check", freq: 1 }] } } });
  await flush();
  assert.equal(pres().length, 1);   // (once it's answered)
  assert.equal(pres()[0].presolve, "BB");
  const n = () => pres().length;
  const none = async (setup, extra, why) => {
    const before = n();
    p.run(setup);
    p.S = { ...state({ ...OOP_TO_ACT, cards: [...OOP_TO_ACT.cards, "3h"], flop: acts("BB x, BTN x") }), ...extra };
    await flush();
    assert.equal(n(), before, why);
    p.S = state(OOP_TO_ACT);   // (a different spot, so the next case's is new)
    await flush();
  };
  await none("OPP = 'pool'", {}, "against the pool");
  await none("OPP = 'gto'", { sizes: { turn: [50] } }, "custom sizes");
  await none("", { locks: { BB: { fold: 10 } } }, "a lock");
  await none("", { ranges: { BB: "QQ+" } }, "a custom range");
  await none("", { optSizes: true }, "size optimization");
  await none("", { payouts: [50, 30, 20] }, "a tournament");
  await none("overlay = 'study'", {}, "Study open");
  await none("overlay = null; window.Practice = new Proxy({ on: true }, { get: (t, k) => k in t ? t[k] : () => null })", {}, "Practice (its own look-ahead)");
  p.run("window.Practice = undefined");
  // a 3-way pot, and you to act
  const before = n();
  p.S = state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO r2.5, BTN c, SB f, BB c") });
  await flush();
  assert.equal(n(), before);
});

test("no presolve where the page can't stop one (not cross-origin isolated) or the runtime is still loading", async () => {
  const { p, pres } = await solving({});
  p.S = state(OOP_TO_ACT);
  await flush();
  assert.equal(pres().length, 0);
  const q = loadPage({ isolated: true });
  await q.ready();
  q.S = state(OOP_TO_ACT);
  await flush();
  assert.equal(q.posted.filter((m) => m.presolve).length, 0);
});
