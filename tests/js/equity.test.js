// Your equity on a reviewed hand (index.html's updateEquity): once a hand is reviewed the page asks the advisor
// ("equity": true) if the betting closed with you still in (the next card not entered, everyone all-in, the river's
// showdown), and shows the answer with the review. It asks nothing while a hand is entered or just finished, on a
// decision stepped to, on your turn or after your fold.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const flush = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const PRE = acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c");
const ANSWER = { showdown: true, street: "flop", next: "turn", allin: false, hero: "BTN", hand: "AKo", board: "Ks7d2c",
  pot_bb: 5.5, equity: 0.898, win: 0.897, tie: 0.001, share_bb: 4.94, runouts: 1081, exact: true, notes: [],
  villains: [{ seat: "BB", equity: 0.898, combos: 413.5, range: "AQo, ...", grid: { actions: ["in range"], cells: {} } }] };
const asks = (p) => p.posted.filter(m => m.state && m.state.equity);
const reviewed = (o) => ({ ...state(o), finished: true, reviewed: true });

async function page(s) {
  const p = loadPage();
  await p.ready();
  p.S = s;
  await flush();
  return p;
}

test("a reviewed hand whose flop was checked through: the page asks for your equity and shows it", async () => {
  const p = await page(reviewed({ cards: ["Ah", "Kd", "Ks", "7d", "2c"], preflop: PRE, flop: acts("BB x, BTN x") }));
  const q = asks(p);
  assert.equal(q.length, 1);
  assert.equal(q[0].state.board, "Ks7d2c");
  assert.equal(p.byId("equity").hidden, false);
  assert.match(p.byId("equity").innerHTML, /Working out your equity/);
  p.workers.at(-1).onmessage({ data: { id: q[0].id, result: ANSWER } });
  await flush();
  const h = p.byId("equity").innerHTML;
  assert.match(h, /89\.8%/);
  assert.match(h, /BB's range \(\d+(\.\d+)? combos\), over all 1081 runouts\./);
  assert.match(h, /4\.94 of the 5\.5 bb pot/);
  assert.match(h, /over all 1081 runouts/);
  // the same spot again (a re-render) asks nothing more; the answer is kept for it
  p.run("renderAll()"); await flush();
  assert.equal(asks(p).length, 1);
});

test("all-in preflop, reviewed: asked with the board still to come, and at the showdown", async () => {
  const pre = acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB a, BTN c");
  const p = await page(reviewed({ cards: ["Qs", "Qd"], preflop: pre }));
  assert.equal(asks(p).length, 1);
  p.S = reviewed({ cards: ["Qs", "Qd", "Ks", "7d", "2c", "5h", "9s"], preflop: pre }); await flush();
  assert.deepEqual(asks(p).map(m => m.state.board), ["", "Ks7d2c5h9s"]);
});

test("only once it's reviewed: nothing while a hand is entered, on Finish hand, or on a decision stepped to", async () => {
  const hand = { cards: ["Ah", "Kd", "Ks", "7d", "2c"], preflop: PRE, flop: acts("BB x, BTN x") };
  const p = await page(state(hand));                       // the betting closed, the turn not entered
  assert.equal(asks(p).length, 0);
  assert.equal(p.byId("equity").hidden, true);
  p.S = { ...state(hand), finished: true }; await flush();  // finished: Keep entering could still reopen it
  assert.equal(asks(p).length, 0);
  assert.equal(p.byId("equity").hidden, true);
  p.S = reviewed(hand); await flush();
  assert.equal(asks(p).length, 1);
  p.workers.at(-1).onmessage({ data: { id: asks(p)[0].id, result: ANSWER } }); await flush();
  assert.equal(p.byId("equity").hidden, false);
  p.run('stepTo("flop", 1)'); await flush();
  assert.equal(p.byId("equity").hidden, true);              // (that decision's advice, not the hand's end)
  p.run("stepStop()"); await flush();
  assert.equal(p.byId("equity").hidden, false);
  assert.equal(asks(p).length, 1);                          // (kept)
});

test("a decision opened while it runs: the equity request is stopped, and asked again back on the hand", async () => {
  const hand = { cards: ["Ah", "Kd", "Ks", "7d", "2c"], preflop: PRE, flop: acts("BB x, BTN x") };
  const p = await page(reviewed(hand));
  assert.equal(asks(p).length, 1);
  p.run('stepTo("flop", 1)'); await flush();
  assert.equal(p.val("eqReq"), null);
  p.run("stepStop()"); await flush();
  assert.equal(p.byId("equity").hidden, false);
});

test("no equity on your turn, after your fold, or before the street closes", async () => {
  for (const s of [
    reviewed({ cards: ["Ah", "Kd", "Ks", "7d", "2c"], preflop: PRE, flop: acts("BB x") }),               // your turn
    reviewed({ cards: ["Ah", "Kd"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB r10, BTN f") }),  // folded
    reviewed({ cards: ["Ah", "Kd"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5") }),                       // still betting
  ]) {
    const p = await page(s);
    assert.equal(asks(p).length, 0);
    assert.equal(p.byId("equity").hidden, true);
  }
});
