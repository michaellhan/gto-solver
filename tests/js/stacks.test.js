// Per-seat stacks (S.stacks): the engine's all-ins for less, side pots, uncalled bets, incomplete raises and the
// end of the action once players are all-in; the setup's seat stacks, the state sent to the solver, share links
// and the copied hand text.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const label = (b) => b.innerHTML.replace(/<.*$/, "");
function page(st, stacks) { const p = loadPage(); p.S = { ...st, ...(stacks ? { stacks } : {}) }; return p; }
const BOARD = ["Ah", "Kh", "Kd", "7s", "2c", "9h", "3d"];

test("a short stack's all-in: calls for less, the side pot, and the pot shown main and side", () => {
  // CO has 30bb: its jam is called by the button and the big blind, who bet on beside it
  const p = page(state({ cards: BOARD.slice(0, 6), preflop: acts("UTG f, HJ f, CO a, BTN c, SB f, BB c") }), { CO: 30 });
  let s = p.sim();
  assert.deepEqual([s.current, s.toAct, s.pot, s.allin], ["flop", "BB", 90.5, ["CO"]]);
  assert.deepEqual(p.val("potsNow(simulate())"), [{ amount: 90.5, eligible: ["CO", "BTN", "BB"] }]);
  p.act({ action: "bet", to: 20 }); p.act({ action: "call" });
  s = p.sim();
  assert.deepEqual([s.current, s.toAct, s.pot], ["turn", "BB", 130.5]);
  assert.deepEqual(p.val("potsNow(simulate())"), [{ amount: 90.5, eligible: ["CO", "BTN", "BB"] }, { amount: 40, eligible: ["BTN", "BB"] }]);
  assert.match(p.byId("felt").children.at(-1).innerHTML, /pot 130.5 bb<\/div><div class="side">main 90.5 · side 40<\/div>/);
  // each seat's chips behind show beside its name while the stacks differ
  const seat = (q) => p.byId("felt").children.find(e => e.innerHTML.startsWith(`<span class="pos">${q}`)).innerHTML;
  assert.match(seat("CO"), /<span class="stk"[^>]*>0<\/span>/);
  assert.match(seat("BTN"), /<span class="stk"[^>]*>50<\/span>/);
  // BB jams the turn, BTN calls: the side pot grows, the main pot stays
  p.act({ action: "all-in" }); p.act({ action: "call" });
  s = p.sim();
  assert.deepEqual([s.waiting, s.runout, s.pot, s.allin], ["river", true, 230.5, ["CO", "BTN", "BB"]]);
  assert.deepEqual(p.val("potsNow(simulate())").map(x => x.amount), [90.5, 140]);
});

test("a bet nobody can call in full comes back once the street closes", () => {
  // hero (BTN, 100bb) jams over the 20bb big blind, who calls: 80bb come back and nobody is left to bet
  const p = page(state({ cards: BOARD.slice(0, 2), preflop: acts("UTG f, HJ f, CO f, BTN a, SB f, BB c") }), { BB: 20 });
  const s = p.sim();
  assert.deepEqual([s.waiting, s.runout, s.pot, s.allin, s.streets.preflop.returned], ["flop", true, 40.5, ["BB"], { BTN: 80 }]);
  assert.equal(s.invested.BTN, 20);
  p.run('placeCard("Kd"); placeCard("7s"); placeCard("2c")');
  assert.deepEqual([p.sim().waiting, p.sim().cur], ["turn", null]);   // no flop betting: one player has chips
  // a short stack all-in for less than the big blind: the big blind has nothing to call and the street is over
  const q = page(state({ players: 2, hero: "BB", cards: BOARD.slice(0, 2), preflop: acts("SB c") }), { SB: 0.75 });
  assert.deepEqual([q.sim().waiting, q.sim().pot, q.sim().allin], ["flop", 1.5, ["SB"]]);
  assert.deepEqual(q.sim().streets.preflop.returned, { BB: 0.25 });
  // heads-up, the short stack jams and is called: the hand runs out
  const r = page(state({ players: 2, hero: "BB", cards: BOARD.slice(0, 2), preflop: acts("SB a, BB c") }), { SB: 10 });
  assert.deepEqual([r.sim().waiting, r.sim().pot, r.sim().runout], ["flop", 20, true]);
});

test("a hand that ends on a fold to a bet: the uncalled part goes back, the pot shown is the pot won", () => {
  // flop: BB bets 6, BTN raises to 20.5, BB folds: 14.5 of the raise was never called
  const p = page(state({ cards: BOARD.slice(0, 5), preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB b6, BTN r20.5, BB f") }));
  const s = p.sim();
  assert.deepEqual([s.handOver, s.pot, s.invested.BTN, s.streets.flop.returned], [true, 17.5, 8.5, { BTN: 14.5 }]);
  p.run("renderAll()");
  assert.match(p.byId("felt").children.at(-1).innerHTML, /pot 17.5 bb/);
  assert.match(p.byId("felt").children.find(e => e.innerHTML.startsWith('<span class="pos">BTN')).innerHTML, /in 8.5/);
  // everyone folds to the big blind: its blind isn't called past the small blind's
  const q = page(state({ hero: "BB", cards: BOARD.slice(0, 2), preflop: acts("UTG f, HJ f, CO f, BTN f, SB f") }));
  assert.deepEqual([q.sim().handOver, q.sim().pot, q.sim().streets.preflop.returned], [true, 1, { BB: 0.5 }]);
});

test("an all-in raise for less than a full raise doesn't reopen the betting", () => {
  // flop: BB bets 4, CO raises to 8 (a full raise of 4), the 12bb button jams to 9.5 (1.5 more: not a full raise)
  const pre = acts("UTG f, HJ f, CO r2.5, BTN c, SB f, BB c");
  const p = page(state({ hero: "CO", cards: BOARD.slice(0, 6), preflop: pre, flop: acts("BB b4, CO r8, BTN a") }), { BTN: 12 });
  let s = p.sim();
  assert.deepEqual([s.toAct, s.cur.curBet, s.cur.minRaise], ["BB", 9.5, 4]);
  assert.deepEqual(p.val("[...simulate().cur.locked].sort()"), ["BTN", "CO"]);
  // BB faced CO's full raise: it may raise again, at least by that raise (to 13.5)
  assert.equal(p.act({ action: "bet", to: 12 }), "The smallest raise is to 13.5bb.");
  assert.equal(p.act({ action: "call" }), null);
  // back to CO, who made the last full raise: call or fold only
  s = p.sim();
  assert.equal(s.toAct, "CO");
  assert.equal(p.act({ action: "bet", to: 30 }), "CO can't raise: the all-in for less than a full raise doesn't reopen the betting.");
  const labels = p.buttons().map(label);
  assert.ok(labels.includes("Call") && labels.includes("Fold"), labels.join());
  assert.ok(!labels.some(l => /^(Raise|All-in)/.test(l)), labels.join());
  assert.equal(p.act({ action: "all-in" }), null);   // an all-in that can't raise is a call
  assert.deepEqual(p.S.actions.flop.at(-1), { pos: "CO", action: "call" });
  s = p.sim();
  assert.deepEqual([s.current, s.toAct, s.pot], ["turn", "BB", 8 + 3 * 9.5]);
  // a full raise reopens it: BTN jams to 16 over CO's 8 (8 more), so CO may raise again
  const q = page(state({ hero: "CO", cards: BOARD.slice(0, 5), preflop: pre, flop: acts("BB b4, CO r8, BTN a, BB c") }), { BTN: 18.5 });
  assert.equal(q.sim().cur.curBet, 16);
  assert.equal(q.act({ action: "bet", to: 40 }), null);
  assert.deepEqual(q.S.actions.flop.at(-1), { pos: "CO", action: "raise", to: 40 });
});

test("the smallest bet and raise: a full raise, all-in for less allowed", () => {
  const p = page(state({ cards: ["Ah", "Kh"], preflop: acts("UTG r3") }));
  assert.equal(p.act({ action: "bet", to: 4.5 }), "The smallest raise is to 5bb.");
  assert.equal(p.act({ action: "bet", to: 5 }), null);
  // a 4bb stack may go all-in for less than the minimum raise
  const q = page(state({ cards: ["Ah", "Kh"], preflop: acts("UTG r3") }), { HJ: 4 });
  assert.equal(q.act({ action: "bet", to: 4 }), null);
  assert.deepEqual(q.S.actions.preflop.at(-1), { pos: "HJ", action: "all-in" });
  // postflop the smallest bet is the big blind
  const r = page(state({ cards: BOARD.slice(0, 5), preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c") }));
  assert.equal(r.act({ action: "bet", to: 0.5 }), "The smallest bet is 1bb.");
});

test("nobody left with chips: no raise, the all-in button calls", () => {
  // BB (100bb) faces the 15bb button's jam with everyone else out: call or fold
  const p = page(state({ hero: "BB", cards: ["Ah", "Kh"], preflop: acts("UTG f, HJ f, CO f, BTN a, SB f") }), { BTN: 15 });
  const labels = p.buttons().map(label);
  assert.deepEqual(labels.filter(l => /^(Raise|All-in|Call|Fold)/.test(l)), ["Fold", "Call"]);
  assert.match(p.act({ action: "bet", to: 40 }), /can't raise: nobody left has chips to call it/);
  assert.equal(p.act({ action: "all-in" }), null);
  assert.deepEqual(p.S.actions.preflop.at(-1), { pos: "BB", action: "call" });
});

test("setup: seat stacks, the summary, players and new hand keep them, the stack box", () => {
  const p = loadPage();
  p.S = state({});
  assert.equal(p.byId("seatStacks").hidden, true);
  p.click("stacksEdit");
  assert.equal(p.byId("seatStacks").hidden, false);
  const box = (q) => [...p.byId("seatStackIns").walk()].find(e => e.dataset && e.dataset.p === q);
  assert.equal(box("CO").value, "100");
  box("CO").onchange({ target: { value: "35" } });
  box("BB").onchange({ target: { value: "250" } });
  assert.deepEqual(p.S.stacks, { CO: 35, BB: 250 });
  assert.equal(p.val("STACK('CO')"), 35);
  p.run("editSetup = false; S.cards[0] = 'Ah'; S.cards[1] = 'Kh'; renderAll()");
  assert.match(p.byId("setupSum").innerHTML, /<b>100bb<\/b> stacks \(CO 35, BB 250\)/);
  box("CO").onchange({ target: { value: "100" } });   // back to the table's stack: no entry
  assert.deepEqual(p.S.stacks, { BB: 250 });
  // the table's stack changes, the seats with their own keep it; one equal to it is dropped
  p.byId("stack").value = "250"; p.byId("stack").onchange({ target: p.byId("stack") });
  assert.equal(p.S.stack, 250);
  assert.equal(p.S.stacks, undefined);
  p.run("S.stacks = { SB: 40, UTG: 60 }; save(); renderAll()");
  p.run("document.getElementById('players').children[2].onclick()");   // 4-handed: UTG isn't a seat
  assert.deepEqual([p.S.players, p.S.stacks], [4, { SB: 40 }]);
  p.click("newhand");
  assert.deepEqual(p.S.stacks, { SB: 40 });
  p.click("stacksSame");
  assert.equal(p.S.stacks, undefined);
  assert.equal(p.byId("seatStacks").hidden, true);
});

test("the state sent to the solver carries every seat's stack only when they differ", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state({ cards: ["Ah", "Kh"] });
  assert.equal("stacks" in p.payload(), false);
  p.S = { ...state({ cards: ["Ah", "Kh"] }), stacks: { CO: 35 } };
  assert.deepEqual(p.payload().stacks, { UTG: 100, HJ: 100, CO: 35, BTN: 100, SB: 100, BB: 100 });
  assert.equal(p.payload().stack, 100);
});

test("share links and saved hands carry the stacks; old ones without play equal stacks; bad entries are dropped", () => {
  const st = { ...state({ cards: ["Ah", "Kh"], preflop: acts("UTG a") }), stacks: { UTG: 12 } };
  const enc = (x) => Buffer.from(JSON.stringify(x)).toString("base64");
  let p = loadPage({ hash: "#h=" + enc(st) });
  assert.deepEqual(p.S.stacks, { UTG: 12 });
  assert.equal(p.sim().cur.curBet, 12);
  p = loadPage({ hash: "#h=" + enc(state({ cards: ["Ah", "Kh"], preflop: acts("UTG a") })) });   // an old link
  assert.equal(p.S.stacks, undefined);
  assert.equal(p.sim().cur.curBet, 100);
  p = loadPage({ hash: "#h=" + enc({ ...st, stacks: { UTG: -3, XX: 50, BB: "abc", SB: 100, CO: 30 } }) });
  assert.deepEqual(p.S.stacks, { CO: 30 });
  p = loadPage({ hash: "#h=" + enc({ ...st, stacks: "nope" }) });
  assert.equal(p.S.stacks, undefined);
});


