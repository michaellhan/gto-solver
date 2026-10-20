// Practice mode (gto/static/practice.js) with a stubbed solver: dealing, opponents moving through the
// solver's play requests, grading your decisions, the showdown and the running score.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage } = require("./page.js");

const PRACTICE = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "static", "practice.js"), "utf8");
const flush = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const ADVICE = { street: "flop", hero: "BTN", hand: "AKs", recommended: "call",
  actions: [{ action: "fold", freq: 0.2, ev_bb: 0 }, { action: "check", freq: 0.3, ev_bb: 0.4 }, { action: "call", freq: 0.5, ev_bb: 1 }] };

// a page in browser (worker) mode with practice.js loaded and a seeded Math.random
async function practicePage(seed = 7, storage) {
  const p = loadPage({ storage });
  await p.ready();
  p.run(`Math.random = (() => { let s = ${seed}; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();`);
  vm.runInContext(PRACTICE, p.ctx, { filename: "practice.js" });
  p.run("Practice.set(true)");
  return p;
}
// Answer the solver's requests: play requests with villain(message), advice requests with ADVICE.
// Returns the play requests answered.
function answer(p, done, villain) {
  const plays = [];
  for (const m of p.posted) {
    if (m.id == null || done.has(m.id)) continue;
    done.add(m.id);
    if (m.play) plays.push(m);
    p.workers.at(-1).onmessage({ data: { id: m.id, result: m.play ? villain(m) : ADVICE } });
  }
  return plays;
}
const passive = (m) => ({ pos: m.play.seat, action: "call" });   // calls a bet, checks otherwise

// Play the hand out: opponents per villain(), you per hero(sim) once the advice is in.
async function playHand(p, villain = passive, hero = () => ({ action: "call" })) {
  const done = new Set(), plays = [];
  for (let i = 0; i < 200 && !p.val("Practice.state.hand.result"); i++) {
    await flush();
    plays.push(...answer(p, done, villain));
    await flush();
    const sim = p.sim();
    if (sim.toAct === p.S.hero && p.val("resultKey === JSON.stringify(payload())")) assert.equal(p.act(hero(sim)), null);
  }
  return plays;
}

test("practice deals a random seat and hands, and the opponents' hands and the board never repeat a card", async () => {
  const p = await practicePage();
  const h = p.val("Practice.state.hand"), S = p.S;
  assert.equal(h.hero, S.hero);
  assert.deepEqual(h.hole, S.cards.slice(0, 2));
  assert.deepEqual(Object.keys(h.hands).sort(), p.val("POS").filter(x => x !== S.hero).sort());
  const all = [...h.hole, ...Object.values(h.hands).flat(), ...h.board];
  assert.equal(new Set(all).size, all.length);
  assert.equal(all.length, 2 * 6 + 5);
  assert.ok(p.ctx.document.body.className.includes("practice"));
  assert.equal(p.byId("practiceBar").hidden, false);
  // different seeds give different seats over a few deals
  const seats = new Set();
  for (let s = 1; s < 9; s++) seats.add((await practicePage(s * 1000 + 3)).S.hero);
  assert.ok(seats.size >= 3, [...seats].join());
});

test("opponents move by asking the solver for their own hand's move; you are graded; showdown and score", async () => {
  const p = await practicePage(11);
  const plays = await playHand(p);
  const h = p.val("Practice.state.hand"), S = p.S;
  assert.ok(plays.length > 0);
  for (const m of plays) {
    assert.notEqual(m.play.seat, S.hero);
    assert.equal(m.play.hole, h.hands[m.play.seat].join(""));
    assert.equal(m.state.hero, S.hero);                  // the state is the page's own payload (your seat and cards)
    assert.equal(m.state.hole, S.cards[0] + S.cards[1]);
  }
  // everyone checks or calls down: the whole board is dealt from the deck dealt up front, and it goes to showdown
  assert.deepEqual(S.cards.slice(2), h.board);
  assert.equal(p.sim().handOver, true);
  assert.equal(h.result.shown.length, 5);
  assert.ok(h.result.best);
  const mine = Object.values(S.actions).flat().filter(a => a.pos === S.hero);
  assert.ok(mine.length >= 4 && mine.every(a => a.g), "each of your decisions is graded");
  const sc = p.val("Practice.state.score");
  assert.equal(sc.hands, 1);
  assert.equal(sc.decisions, mine.length);
  assert.equal(sc.net, h.result.net);
  // 6 players in for 1bb each: the pot is 6bb, so you net +5, -1 or a split
  const win = h.result.winners;
  assert.equal(h.result.net, win.includes(S.hero) ? Math.round((6 / win.length - 1) * 100) / 100 : -1);
  assert.deepEqual(JSON.parse(p.storage.get("gto-practice-score")), sc);
  // the opponents' cards show on the table at showdown
  assert.deepEqual(p.val(`Practice.shown("${h.result.shown[0]}")`), h.hands[h.result.shown[0]]);
  assert.match(p.byId("practiceBar").innerHTML, /You win|You lose|Split pot/);
  // Next hand (or N) deals again; the score is kept
  p.run("Practice.next()"); await flush();
  assert.equal(p.val("Practice.state.hand.result"), null);
  assert.notDeepEqual(p.S.cards.slice(0, 2), S.cards.slice(0, 2));
  assert.equal(p.val("Practice.state.score.hands"), 1);
});

test("your fold ends the hand; you can't act before the spot is solved; opponents' bets and raises", async () => {
  const p = await practicePage(5);
  await flush();
  const done = new Set();
  // the first raise from an opponent is a 3bb raise, then they call
  let raised = false;
  const villain = (m) => {
    const s = m.state.preflop;
    if (!raised && !s.some(a => a.action === "raise")) { raised = true; return { pos: m.play.seat, action: "raise", to: 3, name: "raise" }; }
    return { pos: m.play.seat, action: "call" };
  };
  for (let i = 0; i < 50 && p.sim().toAct !== p.S.hero; i++) { await flush(); answer(p, done, villain); await flush(); }
  const sim = p.sim();
  assert.equal(sim.toAct, p.S.hero);
  if (!p.val("resultKey === JSON.stringify(payload())")) assert.match(p.act({ action: "fold" }), /being solved/);
  await flush(); answer(p, done, villain); await flush();
  const before = p.S.actions.preflop.length;
  assert.equal(p.act({ action: "fold" }), null);
  assert.equal(p.S.actions.preflop.length, before + 1);
  assert.ok(p.S.actions.preflop.at(-1).g, "graded");
  const s2 = p.sim();
  assert.equal(s2.handOver, true);
  const h = p.val("Practice.state.hand");
  assert.equal(h.result.folded, true);
  assert.ok(h.result.net <= 0);
  if (raised) assert.ok(p.S.actions.preflop.some(a => a.action === "raise" && a.to === 3 && a.pos !== p.S.hero));
  // no more requests once the hand is over
  const n = p.posted.length; await flush(); assert.equal(p.posted.length, n);
});

test("a move that arrives after the hand moved on is dropped, and the practice hand resumes after a reload", async () => {
  const p = await practicePage(21);
  await flush();
  const S0 = p.S;
  if (p.sim().toAct !== S0.hero) {
    p.run("Practice.next()"); await flush();   // a new deal while the opponent's request is out
    const stale = p.posted.filter(m => m.play)[0], before = JSON.stringify(p.S.actions);
    p.workers.at(-1).onmessage({ data: { id: stale.id, result: { pos: stale.play.seat, action: "fold" } } });
    await flush();
    assert.equal(JSON.stringify(p.S.actions), before);
  }
  const storage = Object.fromEntries(p.storage);
  const q = loadPage({ storage });
  await q.ready();
  vm.runInContext(PRACTICE, q.ctx, { filename: "practice.js" });
  const hand = q.val("Practice.state.hand");
  q.run("Practice.set(true)");
  assert.deepEqual(q.val("Practice.state.hand"), hand);   // the same hand, not a new deal
  q.run("Practice.set(false)");
  assert.equal(q.byId("practiceBar").hidden, true);
  assert.ok(!q.ctx.document.body.className.includes("practice"));
});

test("hand strength: categories and kickers", async () => {
  const p = await practicePage();
  const r = (s) => p.val(`Practice.rank7(${JSON.stringify(s.split(" "))})`);
  assert.deepEqual(r("Ah Kh Qh Jh Th 2c 3d"), [8, 14]);
  assert.deepEqual(r("5s 4d 3c 2h Ah Kd Kc"), [4, 5]);              // the wheel beats a pair
  assert.deepEqual(r("Ks Kd Kc 7h 7d 7c 2s"), [6, 13, 7]);
  assert.deepEqual(r("2h 3h 9h Jh Kh Ks Kd"), [5, 13, 11, 9, 3, 2]);
  assert.deepEqual(r("As Ad Ks Kd Qs Qd 2c"), [2, 14, 13, 12]);
  assert.deepEqual(r("9s 9d 5c 7h Jd Kc 2s"), [1, 9, 13, 11, 7]);
  assert.deepEqual(r("Ah Ad Ac As 2h 3h 4h"), [7, 14, 4]);
});

test("your fold that closes the street ends the hand, with others still in", async () => {
  // the big blind folds to an open and a call: no flop is dealt and the opponents don't play on
  let p, seed = 1;
  do p = await practicePage(seed++); while (p.S.hero !== "BB");
  const done = new Set();
  let opened = false;
  const villain = (m) => {
    if (!opened) { opened = true; return { pos: m.play.seat, action: "raise", to: 3, name: "raise 3" }; }
    return { pos: m.play.seat, action: "call" };
  };
  for (let i = 0; i < 50 && p.sim().toAct !== "BB"; i++) { await flush(); answer(p, done, villain); await flush(); }
  for (let i = 0; i < 5 && !p.val("resultKey === JSON.stringify(payload())"); i++) { await flush(); answer(p, done, villain); await flush(); }
  assert.equal(p.act({ action: "fold" }), null);
  p.run("renderAll()"); await flush();
  const sim = p.sim();
  assert.equal(sim.handOver, true);
  assert.equal(sim.waiting, null);
  assert.deepEqual(p.S.cards.slice(2), [null, null, null, null, null]);
  const h = p.val("Practice.state.hand");
  assert.deepEqual([h.result.folded, h.result.net], [true, -1]);
  const n = p.posted.length; await flush(); answer(p, done, villain); await flush();
  assert.equal(p.posted.length, n);
  assert.equal(p.val("Practice.state.score.hands"), 1);
});

test("rake is taken from the called pot: a bet nobody calls goes back unraked", async () => {
  const p = loadPage();
  await p.ready();
  p.run(`Math.random = (() => { let s = 3; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();`);
  p.run("S.players = 2; S.rake = 5; S.rakeCap = 3; setPositions(2); save();");
  vm.runInContext(PRACTICE, p.ctx, { filename: "practice.js" });
  p.run("Practice.set(true)");
  // limp-check to the flop, you bet 2bb there and the opponent folds: 2bb called pot, 5% rake
  const villain = (m) => ({ pos: m.play.seat, action: (m.state.flop || []).some(a => a.action === "bet") ? "fold" : "call" });
  await playHand(p, villain, (sim) => p.S.cards[2] && !(sim.cur.curBet > 0) ? { action: "bet", to: 2 } : { action: "call" });
  const h = p.val("Practice.state.hand");
  assert.deepEqual(p.S.actions.flop.map(a => a.action), p.S.hero === "BB" ? ["bet", "fold"] : ["check", "bet", "fold"]);
  assert.equal(h.result.net, 0.9);   // 4bb pot less 0.1 rake on the 2 called, less your 3
});

test("practice deals keep the Advanced setup (sizes, ranges, size optimization), as New hand does", async () => {
  const p = loadPage();
  await p.ready();
  p.run(`S.sizes = { flop: [50] }; S.ranges = { BTN: "QQ+", BB: "22+" }; S.optSizes = true; save();`);
  vm.runInContext(PRACTICE, p.ctx, { filename: "practice.js" });
  p.run("Practice.set(true)");
  for (let k = 0; k < 2; k++) {
    const S = p.S;
    assert.deepEqual([S.sizes, S.ranges, S.optSizes], [{ flop: [50] }, { BTN: "QQ+", BB: "22+" }, true]);
    p.run("Practice.next()"); await flush();
  }
});

test("per-seat stacks: practice deals keep them, and a short all-in's side pot is paid to the best hand among the players in it", async () => {
  const p = await practicePage(9);
  p.run(`S.stacks = { BTN: 10 }; save(); Practice.next();`); await flush();
  assert.deepEqual(p.S.stacks, { BTN: 10 });
  // 3-handed: the button (10bb) jams, both blinds call and bet the flop: a 30bb main pot for the button's aces,
  // the 40bb side pot between the blinds for your kings (you put in 30)
  p.run(`S = { ...fresh(3), hero: "SB", stack: 100, stacks: { BTN: 10 }, cards: ["Kh", "Kd", "2c", "7d", "9h", "Js", "3s"],
    actions: { preflop: [{ pos: "BTN", action: "all-in" }, { pos: "SB", action: "call" }, { pos: "BB", action: "call" }],
               flop: [{ pos: "SB", action: "bet", to: 20 }, { pos: "BB", action: "call" }],
               turn: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }], river: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }] } };
    setPositions(3);
    Practice.state.hand = { hero: "SB", players: 3, hole: ["Kh", "Kd"], hands: { BTN: ["Ah", "Ad"], BB: ["Qh", "Qd"] }, board: ["2c", "7d", "9h", "Js", "3s"], result: null };
    Practice.state.busy = false; Practice.state.error = "";`);
  const sim = p.sim();
  assert.equal(sim.handOver, true);
  assert.equal(sim.pot, 70);
  assert.deepEqual(p.val("potsNow(simulate())"), [{ amount: 30, eligible: ["BTN", "SB", "BB"] }, { amount: 40, eligible: ["SB", "BB"] }]);
  p.run("Practice.tick(simulate())");
  const r = p.val("Practice.state.hand.result");
  assert.equal(r.net, 10);
  assert.deepEqual(r.pots, [{ amount: 30, eligible: ["BTN", "SB", "BB"], winners: ["BTN"] }, { amount: 40, eligible: ["SB", "BB"], winners: ["SB"] }]);
  assert.match(p.byId("practiceBar").innerHTML, /You win a side pot/);
  assert.match(p.byId("practiceBar").innerHTML, /main pot 30: BTN · side pot 40: you/);
  // the button's aces against your kings alone: you lose the main pot and win nothing else
  p.run(`Practice.state.hand = { ...Practice.state.hand, hands: { BTN: ["Ah", "Ad"], BB: ["As", "Ac"] }, result: null }; Practice.tick(simulate())`);
  const r2 = p.val("Practice.state.hand.result");
  assert.equal(r2.net, -30);
  assert.deepEqual(r2.pots.map(x => x.winners), [["BTN", "BB"], ["BB"]]);
  assert.match(p.byId("practiceBar").innerHTML, /You lose/);
});
