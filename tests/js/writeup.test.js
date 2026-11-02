// Write up a hand (gto/static/writeup.js, loaded the first time its button is clicked): the game, your seat and
// cards, the action seat by seat (amounts in dollars with stakes; skipping to a seat folds or checks the ones before
// it), each street's cards typed, undo, the draft kept in the browser (store.js, or localStorage without it), and Review hand putting the hand on the
// page's table and grading your decisions. The page's own hand and its advice are left alone while you write.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const flush = () => new Promise(r => setImmediate(r));
const flushAll = async () => { for (let i = 0; i < 5; i++) await flush(); };
const j = (x) => JSON.parse(JSON.stringify(x));

async function opened(opts) {
  const p = loadPage(opts);
  await p.ready();
  p.click("writeupBtn");
  await flushAll();
  assert.ok(p.scripts.includes("writeup.js"));
  return p;
}
const W = (p) => p.val("WriteUp.state()");
const call = (p, code) => p.val(`WriteUp.${code}`);

test("opening loads writeup.js once, in the page's game; the page's hand is left alone", async () => {
  const p = await opened();
  assert.equal(p.scripts.filter(s => s === "writeup.js").length, 1);
  p.click("writeupBtn"); await flushAll();
  assert.equal(p.scripts.filter(s => s === "writeup.js").length, 1);   // (loaded once)
  const w = W(p);
  assert.equal(w.players, 6); assert.equal(w.hero, "BTN"); assert.equal(w.stack, 100);
  assert.deepEqual(w.actions, { preflop: [], flop: [], turn: [], river: [] });
  assert.match(call(p, "gameText()"), /^big blinds · 6-handed · 100bb stacks · no rake$/);
  // the page's hand: a spot of its own, unchanged by the write-up's entries
  p.S = state({ cards: ["Ah", "Kh"], preflop: acts("UTG f, HJ r2.5") });
  const before = j(p.S), pay = JSON.stringify(p.payload());
  call(p, "setHole('7c7d')");
  call(p, "record({ action: 'fold' })");
  assert.deepEqual(j(p.S), before);
  assert.equal(JSON.stringify(p.payload()), pay);
  assert.deepEqual(p.val("POS"), ["UTG", "HJ", "CO", "BTN", "SB", "BB"]);
  assert.deepEqual(W(p).actions.preflop, [{ pos: "UTG", action: "fold" }]);
});

test("a $2/$5 hand, 9-handed with a straddle: skips, dollar bets, cards, the line, undo", async () => {
  const p = await opened();
  call(p, "setStakes('2-5')");
  call(p, "setGame({ players: 9 })");
  call(p, "setGame({ stack: '$1,000' })");
  call(p, "setGame({ straddle: true })");
  call(p, "setRake('10-live')");
  let w = W(p);
  assert.deepEqual(w.stakes, { sb: 2, bb: 5 });
  assert.equal(w.stack, 200); assert.equal(w.straddle, true);
  assert.equal(w.rake, 10); assert.equal(w.rakeCap, 1);
  assert.equal(JSON.parse(p.storage.get("gto-stakes")).bb, 5);   // (remembered for the page's new hands too)
  assert.match(call(p, "gameText()"), /^\$2\/\$5 \(small blind played as \$2\.50\) · 9-handed · \$1,000 stacks · \$10 straddle · 10% rake \(cap \$5\)$/);
  call(p, "setGame({ hero: 'BTN' })");
  assert.equal(call(p, "setHole('ah kd')"), null);
  assert.deepEqual(W(p).cards.slice(0, 2), ["Ah", "Kd"]);
  // preflop: the straddle (UTG) acts last; the seats up to yours can be skipped to
  assert.deepEqual(call(p, "skipTargets()"), ["UTG2", "LJ", "HJ", "CO", "BTN"]);
  call(p, "skipTo('HJ')");
  assert.equal(call(p, "record({ amount: '$35' })"), null);   // HJ raises to $35 (7bb)
  call(p, "record({ action: 'fold' })");                       // CO
  call(p, "record({ action: 'call' })");                       // you
  assert.deepEqual(call(p, "skipTargets()"), ["BB", "UTG"]);
  call(p, "skipTo('UTG')");                                    // the blinds fold
  call(p, "record({ action: 'call' })");                       // the straddle calls
  w = W(p);
  assert.deepEqual(w.actions.preflop.map(a => [a.pos, a.action, a.to]),
    [["UTG1", "fold", undefined], ["UTG2", "fold", undefined], ["LJ", "fold", undefined], ["HJ", "raise", 7], ["CO", "fold", undefined],
     ["BTN", "call", undefined], ["SB", "fold", undefined], ["BB", "fold", undefined], ["UTG", "call", undefined]]);
  // the flop: typed, a card already out refused
  assert.match(call(p, "deal('Kc 7d Ah')"), /Ah is already in your hand/);
  assert.match(call(p, "deal('Kc 7d')"), /three cards/);
  assert.equal(call(p, "deal('Kc 7d 2s')"), null);
  assert.deepEqual(W(p).cards.slice(2, 5), ["Kc", "7d", "2s"]);
  // UTG first after the flop; checks to HJ, who bets $50; a raise under the minimum is refused in dollars
  assert.deepEqual(call(p, "skipTargets()"), ["HJ", "BTN"]);
  call(p, "record({ action: 'check' })");
  call(p, "record({ amount: '50' })");
  assert.equal(call(p, "record({ amount: '60' })"), "The smallest raise is to $100.");
  call(p, "record({ amount: '150' })");
  call(p, "record({ action: 'fold' })");
  call(p, "record({ action: 'call' })");
  assert.match(call(p, "deal('3h 4h')"), /turn's card/);
  call(p, "deal('3h')");
  call(p, "record({ action: 'check' })");
  call(p, "record({ amount: '250' })");
  call(p, "record({ action: 'fold' })");
  w = W(p);
  assert.deepEqual(w.actions.flop, [{ pos: "UTG", action: "check" }, { pos: "HJ", action: "bet", to: 10 }, { pos: "BTN", action: "raise", to: 30 },
                                     { pos: "UTG", action: "fold" }, { pos: "HJ", action: "call" }]);
  assert.deepEqual(w.actions.turn, [{ pos: "HJ", action: "check" }, { pos: "BTN", action: "bet", to: 50 }, { pos: "HJ", action: "fold" }]);
  const h = p.run("WriteUp.html()");
  assert.match(h, /HJ raises to \$35/);
  assert.match(h, /UTG calls \$25/);           // (the straddle's $10 in already)
  assert.match(h, /BTN raises to \$150/);
  assert.match(h, /HJ calls \$100/);
  assert.match(h, /<b>Hand over<\/b>/);
  assert.match(h, /data-review(?! disabled)/);
  // undo: the last action, then a street's cards
  call(p, "undo()");
  assert.equal(W(p).actions.turn.length, 2);
  call(p, "undo()"); call(p, "undo()");
  assert.equal(W(p).cards[5], "3h");
  call(p, "undo()");
  assert.equal(W(p).cards[5], null);
  assert.equal(W(p).actions.flop.length, 5);
  assert.match(p.run("WriteUp.html()"), /Deal the turn/);
  // kept as a draft: a reload finds it
  const q = loadPage({ storage: Object.fromEntries(p.storage) });
  await q.ready();
  q.click("writeupBtn"); await flushAll();
  assert.deepEqual(W(q), W(p));
});

test("the buttons and prompt for whoever is to act, in dollars and in big blinds", async () => {
  const p = await opened();
  call(p, "setStakes('1-3')");
  call(p, "setGame({ stack: '300' })");
  call(p, "skipTo('CO')");
  call(p, "record({ amount: '15' })");   // CO raises to $15 = 5bb
  let h = p.run("WriteUp.html()");
  assert.match(h, /<b class="me">You \(BTN\)<\/b> to act on the preflop · pot \$19\.50 <span class="wu-bb">6\.5bb<\/span> · \$15 <span class="wu-bb">5bb<\/span> to call/);
  assert.match(h, /data-act="fold"[^>]*>Fold</);
  assert.match(h, /data-act="call"[^>]*>Call \$15</);
  assert.match(h, /data-act="all-in"[^>]*>All-in \$300</);
  assert.match(h, /data-form="bet".*Raise to<\/span>\$<input/);
  assert.match(h, /small blind played as \$1\.50/);
  // big blinds: the same spot's amounts in bb
  call(p, "setStakes('off')");
  h = p.run("WriteUp.html()");
  assert.match(h, /Call 5bb/);
  assert.match(h, /pot 6\.5bb · 5bb to call/);
  call(p, "record({ amount: '15' })");
  assert.deepEqual(W(p).actions.preflop.at(-1), { pos: "BTN", action: "raise", to: 15 });
});

test("cards: the ways they're typed, and the ones refused", async () => {
  const p = await opened();
  assert.deepEqual(p.val("WriteUp.parseCards('A♥K♦').cards"), ["Ah", "Kd"]);
  assert.deepEqual(p.val("WriteUp.parseCards('10h 9H').cards"), ["Th", "9h"]);
  assert.deepEqual(p.val("WriteUp.parseCards('AsKs, Qs').cards"), ["As", "Ks", "Qs"]);
  assert.equal(p.val("WriteUp.parseCards('AxKs').junk"), "Ax");
  assert.match(call(p, "setHole('Ah')"), /two cards/);
  assert.match(call(p, "setHole('AhAh')"), /different/);
  assert.match(call(p, "setHole('AhKz')"), /two cards/);
  assert.equal(call(p, "setHole('Qc Jc')"), null);
});

test("Review hand: blocked until you've a decision; then on the table, graded, and a new draft in the same game", async () => {
  const p = await opened();
  call(p, "setStakes('2-5')");
  assert.equal(await p.run("WriteUp.review()"), false);
  assert.match(call(p, "error()"), /two cards/);
  call(p, "setHole('JdJc')");
  call(p, "skipTo('BTN')");
  assert.equal(await p.run("WriteUp.review()"), false);
  assert.match(call(p, "error()"), /nothing of yours to review/);
  assert.match(p.run("WriteUp.html()"), /data-review disabled/);
  call(p, "record({ amount: '12.50' })");   // you open to $12.50 (2.5bb)
  call(p, "skipTo('BB')");
  call(p, "record({ action: 'call' })");
  call(p, "deal('9s 5d 2c')");
  call(p, "record({ action: 'check' })");
  const w = W(p);
  assert.equal(await p.run("WriteUp.review()"), true);
  await flushAll();
  const S = JSON.parse(p.storage.get("handreview"));   // (the page's S is each decision's spot while it reviews)
  assert.deepEqual(S.cards, w.cards);
  assert.deepEqual(S.actions, w.actions);
  assert.deepEqual(S.stakes, { sb: 2, bb: 5 });
  assert.equal(p.val("review && review.n"), 1);                // the review is grading your one decision
  assert.ok(p.posted.some(m => m.state && m.state.hole === "JdJc" && m.state.preflop.length === 3));
  assert.equal(p.val("document.body.children.find(e => e.id === 'writeupBox').hidden"), true);
  // the next write-up: no cards or actions, the same game and seat
  const n = W(p);
  assert.deepEqual(n.actions, { preflop: [], flop: [], turn: [], river: [] });
  assert.deepEqual(n.cards, [null, null, null, null, null, null, null]);
  assert.deepEqual([n.stakes, n.hero, n.players], [{ sb: 2, bb: 5 }, "BTN", 6]);
});

test("Open on table: the hand on the page without a review; a table size change starts the action over", async () => {
  const p = await opened();
  call(p, "setHole('8h8d')");
  call(p, "record({ action: 'fold' })");
  call(p, "setGame({ players: 8 })");
  assert.equal(W(p).players, 8);
  assert.deepEqual(W(p).actions.preflop, []);
  assert.deepEqual(W(p).cards.slice(0, 2), ["8h", "8d"]);
  call(p, "skipTo('BTN')");
  assert.equal(await p.run("WriteUp.review(true)"), true);
  await flushAll();
  assert.equal(p.S.players, 8);
  assert.equal(p.S.actions.preflop.length, 5);
  assert.equal(p.val("review"), null);
});

test("with the page's storage (store.js): the draft and the stakes are its records", async () => {
  const until = async (f) => { for (let i = 0; i < 400; i++) { const v = await f(); if (v) return v; await flush(); } throw new Error("timed out"); };
  const p = await opened({ store: true });
  await p.run("GTOStore.ready");
  call(p, "setStakes('1-3')");
  call(p, "setHole('AhKd')");
  const rec = (k) => p.run(`GTOStore.record("kv", ${JSON.stringify(k)})`);
  const stakes = await until(() => rec("gto-stakes"));
  assert.deepEqual(JSON.parse(stakes.data.raw), { sb: 1, bb: 3 });
  const draft = await until(async () => { const r = await rec("gto-writeup"); return r && JSON.parse(r.data.raw).cards[0] === "Ah" && r; });
  assert.deepEqual(JSON.parse(draft.data.raw).stakes, { sb: 1, bb: 3 });
  // the page's own pick goes there too, and its next new hand takes it
  p.byId("stakes").onchange({ target: { value: "5-10" } });
  await until(async () => { const r = await rec("gto-stakes"); return r && JSON.parse(r.data.raw).bb === 10; });
  p.click("newhand");
  assert.deepEqual(p.S.stakes, { sb: 5, bb: 10 });
});
