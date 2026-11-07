// A hand off its range (index.html's checkRanges): once the answer is shown the page asks the same spot under the other
// ranges, one at a time and behind the answer, and says whether they agree. Hands in range ask nothing, the next spot stops
// them, a review doesn't start them, and what's been asked isn't asked again.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const OPENED = acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c");
const HAND = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: OPENED, flop: acts("BB x") };
const answer = (m, rec, x = {}) => ({ data: { id: m.id, result: { street: "flop", hero: "BTN", hand: "AKs", recommended: rec, pot_bb: 5.5,
  actions: [{ action: "check", freq: rec === "check" ? 1 : 0, ev_bb: 2 }, { action: "bet 33%", amount_bb: 1.8, freq: rec === "check" ? 0 : 1, ev_bb: 2.5 }], notes: [], ...x } } });
const OFF = { off_range: { weight: 0.001 } };
const solves = (p) => p.posted.filter(m => m.state);
const followUps = (p) => solves(p).filter(m => m.state.background);
const line = (p) => p.byId("rangesCheck").innerHTML;
async function page(st = HAND, opt = { isolated: true }) {
  const p = loadPage(opt);
  await p.ready();
  if (p.workers[0]) p.workers[0].onmessage({ data: { status: "Solving…" } });
  p.S = state(st);
  await flush();
  return p;
}
const reply = async (p, m, rec, x) => { p.workers.at(-1).onmessage(answer(m, rec, x)); await flush(); };

test("a hand in its range asks nothing more", async () => {
  const p = await page();
  await reply(p, solves(p)[0], "check");
  assert.equal(solves(p).length, 1);
  assert.equal(p.present("rangesCheck"), null);
});

test("off its range: the chart's ranges, then wider, one at a time, each after the last, and they differ", async () => {
  const p = await page();
  assert.equal(solves(p)[0].state.opponents, "pool");
  await reply(p, solves(p)[0], "check", OFF);
  assert.match(p.byId("result").innerHTML, /<table/);   // (the answer is up first)
  assert.match(line(p), /being checked under other ranges \(1 of 3\)/);
  const f = followUps(p);
  assert.equal(f.length, 1);
  assert.equal(f[0].state.opponents, undefined);
  assert.equal(f[0].state.wider, undefined);
  await reply(p, f[0], "bet 33%");
  assert.match(line(p), /\(2 of 3\)/);
  const g = followUps(p);
  assert.equal(g.length, 2);
  assert.equal(g[1].state.wider, true);
  assert.equal(g[1].state.opponents, undefined);
  await reply(p, g[1], "bet 33%");
  assert.match(line(p), /the answer depends on the ranges/);
  assert.match(line(p), /check against a typical player, bet 33% at the charts' ranges and if your range is wider\./);
  assert.match(line(p), /gives up at most 0\.5 bb/);
  assert.match(p.byId("result").innerHTML, /<table/);
});

test("a flip by a hair says so: it isn't an answer that depends on the ranges", async () => {
  const p = await page();
  const near = (v) => ({ actions: [{ action: "check", freq: 0, ev_bb: 2 }, { action: "bet 33%", amount_bb: 1.8, freq: 1, ev_bb: 2 + v }] });
  await reply(p, solves(p)[0], "check", OFF);                      // (pot 5.5: 3% is 0.165 bb)
  await reply(p, followUps(p)[0], "bet 33%", near(0.1));
  await reply(p, followUps(p)[1], "check", near(-0.1));
  assert.doesNotMatch(line(p), /depends/);
  assert.match(line(p), /pick different plays, but it matters little: check against a typical player and if your range is wider, bet 33% at the charts' ranges\./);
  assert.match(line(p), /gives up at most 0\.1 bb/);
});

test("a play the other range hasn't got: nothing to weigh, so it depends", async () => {
  const p = await page();
  await reply(p, solves(p)[0], "bet 75%", OFF);
  await reply(p, followUps(p)[0], "check");
  await reply(p, followUps(p)[1], "check");
  assert.match(line(p), /the answer depends on the ranges/);
  assert.doesNotMatch(line(p), /gives up/);
});

test("they agree", async () => {
  const p = await page();
  await reply(p, solves(p)[0], "check", OFF);
  await reply(p, followUps(p)[0], "check");
  await reply(p, followUps(p)[1], "check");
  assert.match(line(p), /all 3 ranges tried say the same: <b>check<\/b>/);
});

test("a hand left its range on this street: wider has nothing to widen, so the chart alone", async () => {
  const p = await page();
  await reply(p, solves(p)[0], "check", { off_range: { weight: 0, wider: false } });
  assert.equal(followUps(p).length, 1);
  await reply(p, followUps(p)[0], "check");
  assert.equal(followUps(p).length, 1);
  assert.match(line(p), /both ranges tried say the same/);
});

test("the next spot stops them, at once, and its answer isn't held up", async () => {
  const p = await page();
  await reply(p, solves(p)[0], "check", OFF);
  const f = followUps(p);
  assert.equal(f.length, 1);
  p.S = state({ ...HAND, flop: acts("BB b2") }); await flush();     // (BB bets: a new spot for BTN)
  const main = solves(p).at(-1);
  assert.equal(main.state.background, undefined);
  assert.notEqual(main.id, f[0].id);
  assert.equal(p.run("Atomics.load(STOP, 1)"), f[0].id);
  await reply(p, main, "check");
  assert.equal(followUps(p).length, 1);                            // (nothing else from the spot it left)
  assert.equal(p.present("rangesCheck"), null);
});

test("a spot seen again asks nothing: its answers are kept", async () => {
  const p = await page();
  await reply(p, solves(p)[0], "check", OFF);
  await reply(p, followUps(p)[0], "bet 33%");
  await reply(p, followUps(p)[1], "check");
  const n = solves(p).length, before = line(p);
  p.run("renderResult(lastResult, resultKey)"); await flush();
  assert.equal(solves(p).length, n);
  assert.equal(line(p), before);
});

test("a follow-up that fails drops out of the line", async () => {
  const p = await page();
  await reply(p, solves(p)[0], "check", OFF);
  p.workers.at(-1).onmessage({ data: { id: followUps(p)[0].id, error: "nope" } }); await flush();
  await reply(p, followUps(p)[1], "bet 33%");
  assert.match(line(p), /the answer depends on the ranges/);
  assert.doesNotMatch(line(p), /charts' ranges/);
});

test("your own ranges or locks are the assumption: nothing is checked", async () => {
  const p = await page();
  p.S = { ...state(HAND), ranges: { BB: "22+" } }; await flush();
  await reply(p, solves(p).at(-1), "check", OFF);
  assert.equal(followUps(p).length, 0);
});

test("a browser with no stop flag doesn't start them: stopping one would restart the worker", async () => {
  const p = await page(HAND, {});
  await reply(p, solves(p)[0], "check", OFF);
  assert.equal(followUps(p).length, 0);
  assert.equal(p.present("rangesCheck"), null);
});

test("rules answers and preflop aren't checked", async () => {
  const p = await page();
  await reply(p, solves(p)[0], "check", { ...OFF, multiway: true });
  assert.equal(followUps(p).length, 0);
});

test("a tournament's own answer is the charts': wider is all it asks", async () => {
  const p = await page();
  p.S = { ...state(HAND), tournament: true }; await flush();
  const main = solves(p).at(-1);
  assert.equal(main.state.opponents, undefined);
  await reply(p, main, "check", OFF);
  assert.equal(followUps(p).length, 1);
  assert.equal(followUps(p)[0].state.wider, true);
  assert.match(line(p), /\(1 of 2\)/);
});

test("a review asks none; opening a decision then checks that one", async () => {
  const p = await page({ ...HAND, flop: acts("BB x, BTN b3, BB c") });
  p.run("S.finished = true; renderAll()"); await flush();
  assert.equal(solves(p).length, 0);
  p.run("reviewHand()"); await flush();
  for (let i = 0; i < 8 && p.run("reviewing()"); i++) {
    const m = solves(p).at(-1);
    if (m && !m.done) { m.done = true; await reply(p, m, "check", OFF); }
    await flush();
  }
  assert.ok(solves(p).length >= 1);
  assert.equal(followUps(p).length, 0);
  p.run("stepTo('flop', 1)"); await flush();
  const f = followUps(p);
  assert.equal(f.length, 1);
  assert.equal(f[0].state.flop.length, 1);
});
