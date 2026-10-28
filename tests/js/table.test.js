// Practice at a table (Your seat: Table, the default): the button moving one seat a hand, each seat's chips
// carried over (rebuys, Reset stacks), the showdown paying main, side and split pots to the best hands, and the
// run-out after your fold, which never changes your score. Stubbed solver, seeded deals, fixed decks.
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

async function tablePage(seed = 7, storage = {}, setup = "") {
  const p = loadPage({ storage });
  await p.ready();
  p.run(`Math.random = (() => { let s = ${seed}; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();`);
  if (setup) p.run(setup);
  vm.runInContext(PRACTICE, p.ctx, { filename: "practice.js" });
  p.run("Practice.set(true)");
  return p;
}
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
const passive = (m) => ({ pos: m.play.seat, action: "call" });
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
const T = (p) => p.val("Practice.state.table");

test("the button moves one seat clockwise a hand: your position runs BTN, CO, HJ … UTG, BB, SB at 2 to 9 players", async () => {
  const p = await tablePage();
  const want = { 2: ["SB", "BB"], 3: ["BTN", "BB", "SB"], 4: ["BTN", "CO", "BB", "SB"], 5: ["BTN", "CO", "HJ", "BB", "SB"],
    6: ["BTN", "CO", "HJ", "UTG", "BB", "SB"], 7: ["BTN", "CO", "HJ", "LJ", "UTG", "BB", "SB"],
    8: ["BTN", "CO", "HJ", "LJ", "UTG1", "UTG", "BB", "SB"], 9: ["BTN", "CO", "HJ", "LJ", "UTG2", "UTG1", "UTG", "BB", "SB"] };
  for (let n = 2; n <= 9; n++) {
    const seq = []; for (let b = 0; b < n; b++) seq.push(p.val(`Practice.posAt(0, ${b}, ${n})`));
    assert.deepEqual(seq, want[n], `${n}-handed`);
    for (let b = 0; b < n; b++) {   // every seat has its own position; the one after the button's is the small blind
      const row = []; for (let k = 0; k < n; k++) row.push(p.val(`Practice.posAt(${k}, ${b}, ${n})`));
      assert.equal(new Set(row).size, n);
      assert.equal(row[b], n === 2 ? "SB" : "BTN");
      if (n > 2) assert.equal(row[(b + 1) % n], "SB");
    }
  }
});

test("at the table: Table is the default seat, the button and your position move each hand, the D chip with it, kept over a reload", async () => {
  const p = await tablePage(3);
  assert.equal(p.val("Practice.state.seat"), "table");
  assert.match(p.byId("pxSeat").innerHTML, /data-v="table"[^>]*>Table<\/button>.*data-v="random"/);
  const heroes = [], btns = [];
  for (let k = 0; k < 7; k++) {
    const h = p.val("Practice.state.hand");
    heroes.push(p.S.hero); btns.push(T(p).btn);
    assert.equal(h.tbl.btn, T(p).btn);
    assert.equal(p.S.hero, p.val(`Practice.posAt(0, ${T(p).btn}, 6)`));
    // the D chip sits on the button's seat, drawn in its place around the table (your seat at the bottom)
    p.run("renderTable(simulate())");
    const seats = p.byId("felt").children.filter(e => /seat/.test(e.className));
    const d = seats.filter(e => e.innerHTML.includes('class="dbtn"'));
    assert.equal(d.length, 1);
    assert.match(d[0].innerHTML, /^<span class="pos">BTN/);
    await playHand(p, passive, () => ({ action: "fold" }));
    p.run("Practice.next()"); await flush();
  }
  for (let k = 1; k < 7; k++) assert.equal(btns[k], (btns[k - 1] + 1) % 6);
  const order = ["BTN", "CO", "HJ", "UTG", "BB", "SB"], i0 = order.indexOf(heroes[0]);
  assert.deepEqual(heroes, heroes.map((_, k) => order[(i0 + k) % 6]));
  // a reload keeps the table: the same hand, button and stacks
  const q = await tablePage(99, Object.fromEntries(p.storage));
  assert.deepEqual(T(q), T(p));
  assert.equal(q.S.hero, p.S.hero);
  assert.deepEqual(q.val("Practice.state.hand.hole"), p.val("Practice.state.hand.hole"));
  // a hand left before its end (a seat picked, a setup change) doesn't move the button
  const b = T(q).btn; q.run("Practice.next()"); await flush();
  assert.equal(T(q).btn, b);
});

test("every seat's chips carry over; a short seat rebuys; Reset stacks; the solver gets each seat's stack", async () => {
  const p = await tablePage(11);
  assert.deepEqual(T(p).seats, [100, 100, 100, 100, 100, 100]);
  await playHand(p);   // everyone calls down: 6bb to the best hand(s)
  const h = p.val("Practice.state.hand"), t = T(p), n = 6;
  const pos = (k) => p.val(`Practice.posAt(${k}, ${h.tbl.btn}, 6)`);
  assert.equal(t.seats[0], 100 + h.result.net);
  for (let k = 0; k < n; k++) assert.equal(t.seats[k], Math.round((100 - 1 + (h.result.pays[pos(k)] || 0)) * 100) / 100);
  assert.equal(Math.round(t.seats.reduce((a, b) => a + b, 0) * 100) / 100, 600);   // (no rake: no chips made or lost)
  // the winners' seats show what they won and their chips after the hand
  p.run("renderTable(simulate())");
  const won = p.byId("felt").children.filter(e => /\bwon\b/.test(e.className));
  assert.equal(won.length, h.result.winners.length);
  assert.match(won[0].innerHTML, /wins \d/);
  // the next hand: the button moved on, each seat with its own chips, sent to the solver as per-seat stacks
  p.run("Practice.next()"); await flush();
  const h2 = p.val("Practice.state.hand");
  assert.equal(h2.tbl.btn, (h.tbl.btn + 1) % n);
  for (let k = 0; k < n; k++) assert.equal(p.val(`STACK(Practice.posAt(${k}, ${h2.tbl.btn}, 6))`), t.seats[k]);
  const sent = p.val("payload()");
  assert.deepEqual(sent.stacks, Object.fromEntries(p.val("POS").map(q => [q, p.val(`STACK("${q}")`)])));
  assert.match(p.byId("practiceBar").innerHTML, new RegExp(`Your stack <b>${String(t.seats[0]).replace(".", "\\.")} bb</b>`));
  // a seat under a quarter of the starting stack rebuys to it at the next deal; a stack typed in the setup is taken over
  p.run("Practice.state.table.seats[3] = 20; Practice.state.table.seats[4] = 30");
  p.run("Practice.next()"); await flush();
  const h3 = p.val("Practice.state.hand");
  assert.equal(h3.tbl.stacks[3], 100);
  assert.equal(h3.tbl.stacks[4], 30);
  assert.deepEqual(h3.tbl.rebuys, [p.val(`Practice.posAt(3, ${h3.tbl.btn}, 6)`)]);
  assert.match(p.byId("practiceBar").innerHTML, /rebought to 100/);
  const s5 = p.val(`Practice.posAt(5, ${h3.tbl.btn}, 6)`);
  p.run(`S.stacks = { ...S.stacks, ${s5}: 250 }; save(); renderAll()`);   // (the setup's seat stacks: a new hand once the seat thinking has moved)
  const seen = new Set(); for (let i = 0; i < 5; i++) { await flush(); answer(p, seen, passive); }
  assert.equal(T(p).seats[5], 250);
  assert.equal(p.val(`STACK("${s5}")`), 250);
  // Reset stacks: everyone back to the setup's stack, a new hand
  p.ctx.confirm = () => true;
  p.run("Practice.rebuyAll()"); await flush();
  assert.deepEqual(T(p).seats, [100, 100, 100, 100, 100, 100]);
  assert.equal(p.S.stacks, undefined);
  // a new starting stack in the setup is a new table
  p.run("S.stack = 40; save(); renderAll()");
  for (let i = 0; i < 5; i++) { await flush(); answer(p, seen, passive); }
  assert.deepEqual([T(p).start, T(p).seats], [40, [40, 40, 40, 40, 40, 40]]);
});

test("Start over (and Reset practice in the ⋯ menu): the score, mistakes, habits and the table cleared, a new hand; asked first", async () => {
  const p = await tablePage(5);
  for (let k = 0; k < 2; k++) { await playHand(p); p.run("Practice.next()"); await flush(); }
  const before = p.val("Practice.state");
  assert.equal(before.score.hands, 2);
  assert.ok(before.review.length > 0 && Object.keys(before.habits).length > 0);
  assert.notDeepEqual(T(p).seats, [100, 100, 100, 100, 100, 100]);
  p.run("renderAll()");
  assert.match(p.byId("practiceBar").innerHTML, /<button class="small ghost" id="pxReset"[^>]*>Start over<\/button>/);
  // declined: nothing changes
  p.ctx.confirm = () => false;
  assert.equal(p.val("Practice.startOver()"), false);
  assert.equal(p.val("Practice.state.score.hands"), 2);
  const seq = T(p).seq, hole = p.val("Practice.state.hand.hole");
  // from the ⋯ menu: everything cleared in this browser too, a new table (every seat at the setup's stack), Hand 1
  p.ctx.confirm = () => true;
  p.click("resetPractice");
  await flush();
  const after = p.val("Practice.state");
  assert.deepEqual([after.score.hands, after.score.decisions, after.score.net, after.review, after.habits], [0, 0, 0, [], {}]);
  assert.deepEqual([T(p).seats, T(p).seq], [[100, 100, 100, 100, 100, 100], 1]);
  assert.notEqual(T(p).seq, seq);
  assert.notDeepEqual(after.hand.hole, hole);
  assert.equal(after.hand.result, null);
  for (const k of ["score", "review", "habits"]) assert.deepEqual(JSON.parse(p.storage.get("gto-practice-" + k)), after[k]);
  assert.deepEqual(JSON.parse(p.storage.get("gto-practice-table")).seats, [100, 100, 100, 100, 100, 100]);
  assert.equal(p.byId("pxHandNo").textContent, "Hand 1");
  assert.equal(p.val("Practice.state.seat"), "table");   // (your seat and the drill stay)
});

// a fixed hand at the table, played to its end: 3-handed, every action given
function fixed(p, { stacks, actions, hole, hands, board, btn = 0 }) {
  p.run(`S = { ...fresh(3), hero: Practice.posAt(0, ${btn}, 3), stack: 100, ${stacks ? `stacks: ${JSON.stringify(stacks)},` : ""} cards: ${JSON.stringify([...hole, ...board])},
      actions: ${JSON.stringify({ preflop: [], flop: [], turn: [], river: [], ...actions })} };
    setPositions(3);
    Practice.state.table = { n: 3, start: 100, btn: ${btn}, seats: [0, 1, 2].map(k => STACK(Practice.posAt(k, ${btn}, 3))), seq: 5 };
    Practice.state.hand = { hero: S.hero, players: 3, hole: ${JSON.stringify(hole)}, hands: ${JSON.stringify(hands)}, board: ${JSON.stringify(board)}, result: null,
      tbl: { seq: 5, btn: ${btn}, stacks: Practice.state.table.seats.slice(), rebuys: [] } };
    Practice.state.busy = false; Practice.state.error = ""; Practice.tick(simulate());`);
  return p.val("Practice.state.hand.result");
}

test("showdown: everyone's cards, the winning hand named, a split pot and side pots paid to the best hands, the chips moved", async () => {
  const p = await tablePage(5);
  // you're the button (seat 0): all three check it down; your queen-high flush beats the small blind's two pair
  let r = fixed(p, { hole: ["Qh", "4h"], hands: { SB: ["Kc", "7c"], BB: ["2d", "3s"] }, board: ["Kd", "7d", "9h", "2h", "Jh"],
    actions: { preflop: [{ pos: "BTN", action: "call" }, { pos: "SB", action: "call" }, { pos: "BB", action: "check" }],
      flop: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }],
      turn: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }],
      river: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }] } });
  assert.deepEqual([r.winners, r.net, r.best], [["BTN"], 2, "a flush"]);
  assert.equal(r.says, "Your Q-high flush beats SB's two pair, kings and sevens");
  assert.deepEqual(T(p).seats, [102, 99, 99]);
  assert.deepEqual(p.val("Practice.shown('SB')"), ["Kc", "7c"]);
  assert.deepEqual(p.val("Practice.shown('BB')"), ["2d", "3s"]);
  assert.match(p.byId("practiceBar").innerHTML, /Your Q-high flush beats SB's two pair, kings and sevens/);
  // the board plays for two of you: a split pot (the third's pair of twos loses), the chips halved
  r = fixed(p, { hole: ["2c", "3c"], hands: { SB: ["4d", "5d"], BB: ["2s", "8d"] }, board: ["Ah", "Kd", "Qs", "Jc", "Th"],
    actions: { preflop: [{ pos: "BTN", action: "raise", to: 3 }, { pos: "SB", action: "call" }, { pos: "BB", action: "call" }],
      flop: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }],
      turn: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }],
      river: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }] } });
  assert.deepEqual(r.winners, ["BTN", "SB", "BB"]);   // (the board's straight: all three)
  assert.equal(r.net, 0);
  assert.equal(r.says, "Split pot: you, SB and BB tie with A-high straight");
  r = fixed(p, { hole: ["Ah", "Kh"], hands: { SB: ["Ad", "Kd"], BB: ["2s", "8d"] }, board: ["As", "Kc", "7s", "4c", "3h"],
    actions: { preflop: [{ pos: "BTN", action: "raise", to: 3 }, { pos: "SB", action: "call" }, { pos: "BB", action: "call" }],
      flop: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }],
      turn: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }],
      river: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }] } });
  assert.deepEqual([r.winners, r.net], [["BTN", "SB"], 1.5]);
  assert.equal(r.says, "Split pot: you and SB tie with two pair, aces and kings");
  assert.deepEqual(T(p).seats, [101.5, 101.5, 97]);
  assert.match(p.byId("practiceBar").innerHTML, /Split pot/);
  // side pots: the button (you, 10bb) is all-in; the blinds bet on: you win the 30bb main pot, the small blind the side pot
  r = fixed(p, { stacks: { BTN: 10 }, hole: ["Ah", "Ad"], hands: { SB: ["Kh", "Kd"], BB: ["Qh", "Qd"] }, board: ["2c", "7d", "9h", "Js", "3s"],
    actions: { preflop: [{ pos: "BTN", action: "all-in" }, { pos: "SB", action: "call" }, { pos: "BB", action: "call" }],
      flop: [{ pos: "SB", action: "bet", to: 20 }, { pos: "BB", action: "call" }],
      turn: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }], river: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }] } });
  assert.deepEqual(r.pots, [{ amount: 30, eligible: ["BTN", "SB", "BB"], winners: ["BTN"] }, { amount: 40, eligible: ["SB", "BB"], winners: ["SB"] }]);
  assert.deepEqual(r.pays, { BTN: 30, SB: 40 });
  assert.equal(r.net, 20);
  assert.deepEqual(T(p).seats, [30, 110, 70]);   // (your 10 is 30; the small blind 100 - 30 + 40; the big blind 100 - 30)
  // the same with the big blind's kings beating the small blind's queens for the side pot, the main pot split two ways
  r = fixed(p, { stacks: { BTN: 10 }, hole: ["Ah", "Kc"], hands: { SB: ["Ad", "Ks"], BB: ["Qh", "Qd"] }, board: ["2c", "7d", "9h", "Js", "3s"],
    actions: { preflop: [{ pos: "BTN", action: "all-in" }, { pos: "SB", action: "call" }, { pos: "BB", action: "call" }],
      flop: [{ pos: "SB", action: "bet", to: 20 }, { pos: "BB", action: "call" }],
      turn: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }], river: [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }] } });
  assert.deepEqual(r.pots.map(x => x.winners), [["BB"], ["BB"]]);
  assert.deepEqual(T(p).seats, [0, 70, 140]);
  assert.equal(r.says, "BB's pair of queens beats your A-high");
  // the next deal: the busted seat (yours) rebuys
  p.run("Practice.state.seat = 'table'; Practice.next()"); await flush();
  assert.equal(p.val("Practice.state.hand.tbl.stacks[0]"), 100);
});

test("hand names: categories, and the kicker when two hands share a name", async () => {
  const p = await tablePage();
  const name = (a, b) => p.val(`Practice.handName(Practice.rank7(${JSON.stringify(a.split(" "))})${b ? `, Practice.rank7(${JSON.stringify(b.split(" "))})` : ""})`);
  assert.equal(name("Ah Kh Qh Jh Th 2c 3d"), "royal flush");
  assert.equal(name("9h 8h 7h 6h 5h 2c 3d"), "9-high straight flush");
  assert.equal(name("Ks Kd Kc 7h 7d 7c 2s"), "full house, kings full of sevens");
  assert.equal(name("5s 4d 3c 2h Ah Kd Kc"), "5-high straight");
  assert.equal(name("Ts 9d 8c 7h 6h Kd 2c"), "10-high straight");
  assert.equal(name("9s 9d 9c 7h Jd Kc 2s"), "three nines");
  assert.equal(name("As Ad Ac Ah 2h 3h 4h"), "four aces");
  assert.equal(name("As Ad Kc 7h Jd 4c 2s", "Ah Ac Qc 7h Jd 4c 2s"), "pair of aces, K kicker");
  assert.equal(name("Ac 9d 7c 5h 3d 2c Ts"), "A-high");
});

test("Run it out after your fold: the others play on to the end, everyone's cards shown, the table's chips as it went, your score unchanged", async () => {
  // the button at seat 3: you (seat 0) are UTG, first to act; you fold, five others still in
  const p = await tablePage(21, { "gto-practice-table": JSON.stringify({ n: 6, start: 100, btn: 3, seats: [100, 100, 100, 100, 100, 100], seq: 0 }) });
  p.run("Practice.next()"); await flush();
  assert.equal(p.S.hero, "UTG");
  await playHand(p, passive, () => ({ action: "fold" }));
  const h = p.val("Practice.state.hand"), score = p.val("Practice.state.score"), result = h.result;
  assert.equal(result.folded, true);
  assert.equal(p.val("Practice.runLabel()"), "Run it out");
  p.run("renderAll()");
  assert.match(p.byId("practiceBar").innerHTML, /id="pxRun2"[^>]*>Run it out</);
  assert.ok(p.byId("actionArea").children.some(e => e.className === "pxend"));
  p.run("Practice.runOut()");
  assert.equal(p.val("Practice.running"), true);
  const done = new Set(p.posted.map(m => m.id)), plays = [];
  for (let i = 0; i < 200 && p.val("Practice.running"); i++) { await flush(); plays.push(...answer(p, done, passive)); await flush(); }
  assert.equal(p.val("Practice.running"), false);
  // each opponent's move asked from its own seat and cards
  assert.ok(plays.length >= 5);
  for (const m of plays) { assert.equal(m.state.hero, m.play.seat); assert.equal(m.state.hole, m.play.hole); }
  const run = p.val("Practice.state.hand.run"), board = h.board;
  assert.deepEqual(p.S.cards.slice(2), board);
  assert.equal(run.done, true);
  assert.equal(run.live.length, 5);
  assert.ok(run.says);
  for (const q of ["HJ", "CO", "BTN", "SB", "BB"]) assert.deepEqual(p.val(`Practice.shown("${q}")`), h.hands[q]);
  // your result and score are as they were at your fold
  assert.deepEqual(p.val("Practice.state.score"), score);
  assert.deepEqual(p.val("Practice.state.hand.result"), result);
  assert.deepEqual(JSON.parse(p.storage.get("gto-practice-score")), score);
  // the table's chips: you lost nothing (UTG, no blind), the others as the run-out went (5bb to the winners)
  const t = T(p);
  assert.equal(t.seats[0], 100);
  assert.equal(Math.round(t.seats.reduce((a, b) => a + b, 0) * 100) / 100, 600);
  const pos = (k) => p.val(`Practice.posAt(${k}, 3, 6)`);
  for (let k = 1; k < 6; k++) assert.equal(t.seats[k], Math.round((99 + (run.pays[pos(k)] || 0)) * 100) / 100);
  p.run("renderAll()");
  assert.match(p.byId("practiceBar").innerHTML, /Run out/);
  assert.equal(p.val("Practice.runLabel()"), null);
  // the next hand: the button moves on as usual
  p.run("Practice.next()"); await flush();
  assert.equal(T(p).btn, 4);
  assert.deepEqual(p.val("Practice.state.score"), score);
});

test("without a run-out the others' chips go as if checked down; Show hands when only one is left", async () => {
  const p = await tablePage(23, { "gto-practice-table": JSON.stringify({ n: 6, start: 100, btn: 3, seats: [100, 100, 100, 100, 100, 100], seq: 0 }) });
  p.run("Practice.next()"); await flush();
  await playHand(p, passive, () => ({ action: "fold" }));
  // you folded UTG before anyone acted: the blinds' chips are the pot (0.5 each between them, the big blind's other 0.5 back)
  const t = T(p), sb = 4, bb = 5;
  assert.equal(t.seats[0], 100);
  assert.equal(t.seats[1] + t.seats[2] + t.seats[3], 300);
  assert.equal(t.seats[sb] + t.seats[bb], 200);
  // (not shown as won on the felt: the others' hand isn't over; their chips show as they stand)
  p.run("renderTable(simulate())");
  const felt = p.byId("felt").children.filter(e => /seat/.test(e.className));
  assert.ok(felt.every(e => !/\bwon\b/.test(e.className)));
  assert.match(felt.find(e => e.innerHTML.startsWith('<span class="pos">SB')).innerHTML, /stack behind">99.5</);
  // heads-up: you fold to a bet, one opponent left: Show hands deals the rest of the board and shows the cards
  const q = await tablePage(29, { "gto-practice-table": JSON.stringify({ n: 2, start: 100, btn: 1, seats: [100, 100], seq: 0 }) }, "S.players = 2; setPositions(2); save();");
  q.run("Practice.next()"); await flush();
  assert.equal(q.S.hero, "BB");
  await playHand(q, (m) => ({ pos: m.play.seat, action: "raise", to: 3, name: "raise 3" }), () => ({ action: "fold" }));
  assert.equal(q.val("Practice.runLabel()"), "Show hands");
  const sc = q.val("Practice.state.score");
  q.run("Practice.runOut()"); await flush();
  assert.deepEqual(q.S.cards.slice(2), q.val("Practice.state.hand.board"));
  assert.deepEqual(q.val("Practice.shown('SB')"), q.val("Practice.state.hand.hands.SB"));
  assert.deepEqual(T(q).seats, [99, 101]);
  assert.deepEqual(q.val("Practice.state.score"), sc);
});
