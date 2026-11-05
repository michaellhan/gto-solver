// Solving in the middle of a hand (index.html: liveSpot, live): each of your turns in a hand being entered is solved as it
// comes, the answer goes when the spot does, and the hand stays as open as it was.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const OPENED = acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c");
const YOUR_FLOP = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: OPENED, flop: acts("BB x") };
const THEIR_FLOP = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: OPENED };
const answer = (m, x = {}) => ({ data: { id: m.id, result: { street: "flop", hero: "BTN", hand: "AKs", recommended: "check", pot_bb: 5.5,
  actions: [{ action: "check", freq: 0.7, ev_bb: 2 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.3, ev_bb: 1.9 }], notes: [], ...x } } });
async function page(st) {
  const p = loadPage({ isolated: true });
  await p.ready();
  if (p.workers[0]) p.workers[0].onmessage({ data: { status: "Solving…" } });   // (the runtime is up)
  p.S = state(st);
  await flush();
  return p;
}
const solves = (p) => p.posted.filter(m => m.state);
const WAITING = /Your turns are solved as you reach them/;

test("your turn is solved without being asked, once, and the hand stays as it was", async () => {
  const p = await page(YOUR_FLOP);
  const asked = solves(p);
  assert.equal(asked.length, 1);
  assert.equal(asked[0].state.board, "Kd7s2c");
  assert.equal(asked[0].state.speed, "fast");
  assert.deepEqual(asked[0].state.flop, [{ pos: "BB", action: "check" }]);
  assert.equal(p.byId("reviewTitle").textContent, "Solver");
  assert.match(p.byId("result").innerHTML, /Solving the flop/);
  p.workers.at(-1).onmessage(answer(asked[0])); await flush();
  assert.match(p.byId("result").innerHTML, /70%/);
  assert.equal(p.run("document.body.dataset.s"), "enter");   // (the table and dock stay)
  assert.ok(p.buttons().some(b => /^Check/.test(b.innerHTML)));
  assert.equal(p.buttons().some(b => b.id === "solveSpot"), false);
  p.run("renderAll()"); await flush();
  assert.equal(solves(p).length, 1);                         // (a re-render asks nothing more)
  assert.equal(p.val("S.reviewed"), undefined);              // (no lock: it can be undone and entered on)
  assert.equal(p.val("S.finished"), undefined);
  p.run("undo()"); await flush();
  assert.equal(p.S.actions.flop.length, 0);
});

test("nothing is solved at someone else's turn, while a card is needed, or on a finished hand", async () => {
  const p = await page(THEIR_FLOP);
  assert.equal(p.byId("reviewTitle").textContent, "Review");
  assert.match(p.byId("result").innerHTML, WAITING);
  p.S = state({ cards: ["Ah", "Kh"], preflop: OPENED });   // (the flop to deal)
  await flush(); p.run("renderAll()");
  assert.ok(p.sim().waiting);
  p.S = { ...state(YOUR_FLOP), finished: true };
  await flush(); p.run("renderAll()");
  await p.run("analyze(simulate())"); await flush();       // (even called directly)
  assert.deepEqual(solves(p), []);
});

test("recording your action takes the answer away, and your next turn is solved when it comes", async () => {
  const p = await page(YOUR_FLOP);
  p.workers.at(-1).onmessage(answer(solves(p)[0])); await flush();
  p.button("Bet 33%").onclick(); await flush();
  assert.match(p.byId("result").innerHTML, WAITING);
  assert.equal(p.byId("reviewTitle").textContent, "Review");
  p.button("Call").onclick(); p.run('placeCard("9h")'); await flush();
  assert.equal(solves(p).length, 1);
  p.button("Check").onclick(); await flush();   // BB checks the turn: your turn again
  assert.equal(solves(p).length, 2);
  assert.equal(solves(p)[1].state.board, "Kd7s2c9h");
  assert.match(p.byId("result").innerHTML, /Solving the turn/);
});

test("leaving a spot mid-solve stops it, and its answer isn't shown", async () => {
  const p = await page(YOUR_FLOP);
  const m = solves(p)[0];
  p.button("Check").onclick(); await flush();
  assert.equal(p.val("solving"), false);
  assert.match(p.byId("result").innerHTML, WAITING);
  p.workers.at(-1).onmessage(answer(m)); await flush();
  assert.match(p.byId("result").innerHTML, WAITING);
  assert.equal(p.val("live"), false);
});

test("back at a spot already answered, the answer comes from the page's cache", async () => {
  const p = await page(YOUR_FLOP);
  p.workers.at(-1).onmessage(answer(solves(p)[0])); await flush();
  p.button("Check").onclick(); await flush();
  p.run("undo()"); await flush();
  assert.equal(solves(p).length, 1);
  assert.match(p.byId("result").innerHTML, /70%/);
});

test("your action before the solver is ready: the spot you left isn't solved", async () => {
  const p = loadPage({ isolated: true });
  p.S = state(YOUR_FLOP);             // (your turn, the page still finding its backend)
  p.button("Check").onclick();
  await p.ready(); await flush();
  assert.deepEqual(solves(p), []);
  assert.equal(p.val("solving"), false);
});

test("Finish hand at your turn goes to the review, without the answer left on the panel", async () => {
  const p = await page(YOUR_FLOP);
  p.workers.at(-1).onmessage(answer(solves(p)[0])); await flush();
  p.run("finishHand()"); await flush();
  assert.equal(p.val("live"), false);
  assert.equal(p.byId("reviewTitle").textContent, "Review");
  assert.doesNotMatch(p.byId("result").innerHTML, /70%/);
});

test("a turn solved during the hand isn't solved again by its review", async () => {
  const p = await page(YOUR_FLOP);
  p.workers.at(-1).onmessage(answer(solves(p)[0])); await flush();
  p.button("Check").onclick(); await flush();                  // your check, recorded
  p.run("finishHand()"); await flush();
  const before = solves(p).length, done = p.run("reviewHand()");
  for (let n = 0; n < 40; n++) {   // (your preflop raise still has to be solved)
    await flush();
    const m = solves(p).at(-1);
    if (m && !m.answered && m.state.board === "") { m.answered = true; p.workers.at(-1).onmessage(answer(m, { street: "preflop" })); }
    if (p.val("review") === null) break;
  }
  await done; await flush();
  assert.deepEqual(solves(p).slice(before).filter(m => !m.state.equity).map(m => m.state.board), [""]);   // (the hand's equity follows the review)
  assert.equal(p.S.actions.flop.find(a => a.pos === "BTN").g.m, "check");
});

test("on a phone the page stays where it is: scrolling to each answer would take the dock away", async () => {
  const p = await page(THEIR_FLOP);
  let scrolled = 0;
  p.run('window.matchMedia = () => ({ matches: true })');
  p.byId("pAdvice").scrollIntoView = () => scrolled++;
  p.button("Check").onclick(); await flush();   // BB checks: your turn
  p.workers.at(-1).onmessage(answer(solves(p)[0])); await flush();
  assert.match(p.byId("result").innerHTML, /70%/);
  assert.equal(scrolled, 0);
});

test("a failed solve says so, and an undo and the same spot ask again", async () => {
  const p = await page(YOUR_FLOP);
  p.workers.at(-1).onmessage({ data: { id: solves(p)[0].id, result: { error: "no answer" } } }); await flush();
  assert.match(p.byId("result").innerHTML, /no answer/);
  p.run("undo()"); await flush();
  p.button("Check").onclick(); await flush();
  assert.equal(solves(p).length, 2);
});
