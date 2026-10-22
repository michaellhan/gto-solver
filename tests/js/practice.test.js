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

test("the coach: the verdict on your last decision with the solver's mix, then each decision in the hand's summary", async () => {
  const p = await practicePage(11);
  const done = new Set();
  for (let i = 0; i < 50 && !(p.sim().toAct === p.S.hero && p.val("resultKey === JSON.stringify(payload())")); i++) { await flush(); answer(p, done, passive); await flush(); }
  assert.match(p.byId("practiceBar").innerHTML, /Your turn/);
  assert.equal(p.act({ action: "call" }), null);   // the solver's top action (50%)
  p.run("renderAll()");
  const log = p.val("Practice.state.hand.log");
  assert.equal(log.length, 1);
  assert.deepEqual(log[0].mix.map(a => [a.n, a.f]), [["fold", 0.2], ["check", 0.3], ["call", 0.5]]);
  const html = p.byId("practiceBar").innerHTML;
  assert.match(html, /class="verdict">Best move</);
  assert.match(html, /preflop · you <b>called<\/b>/);
  assert.match(html, /no EV lost/);
  assert.match(html, /call <b class="num">50%<\/b><span class="tag">you<\/span>/);   // the mix, at a glance
  assert.match(html, /<details class="cx-more"><summary>Details<\/summary>.*The solver's top choice here/);   // why and the EVs fold away
  assert.match(html, /\+1.00 bb/);
  // after the flop you check when you can: in the mix (30%), 0.6bb below calling's EV
  await playHand(p, passive, (sim) => sim.cur.curBet > sim.cur.bets[p.S.hero] ? { action: "call" } : { action: "check" });
  const h = p.val("Practice.state.hand"), mine = Object.values(p.S.actions).flat().filter(a => a.pos === p.S.hero);
  assert.equal(h.log.length, mine.length);
  assert.ok(mine.some(a => a.action === "check"));
  assert.match(p.byId("practiceBar").innerHTML, /<span class="pill"[^>]*>Good<\/span><span class="num neg">−0.60 bb/);
  assert.match(p.byId("practiceBar").innerHTML, /This hand · \d+ of \d+ in the solver's strategy/);
  // Next hand takes the action buttons' place
  const next = p.byId("actionArea").children.find(b => b.id === "pxNext");
  assert.match(next.innerHTML, /Next hand<kbd>N<\/kbd>/);
  assert.equal(p.byId("actionArea").children.filter(b => b.className === "acts").length, 0);
  assert.equal(p.val("Practice.state.score.good"), mine.length);   // check and call are both in the mix
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

test("your fold mid-street ends the hand with the street's bets (a straddle too) still in front of the others", async () => {
  // straddled table, seats with their own stacks: you (HJ, first after the straddle) fold preflop
  const p = loadPage();
  await p.ready();
  vm.runInContext(PRACTICE, p.ctx, { filename: "practice.js" });
  p.run("Practice.state.on = true; Practice.state.hand = { hole: ['Ah', 'Kh'], hands: {} }");
  p.S = { players: 6, hero: "HJ", stack: 100, stacks: { BTN: 150 }, straddle: true, cards: ["7h", "2c", null, null, null, null, null],
    actions: { preflop: [{ pos: "HJ", action: "fold" }], flop: [], turn: [], river: [] } };
  const sim = p.sim();
  assert.deepEqual([sim.handOver, sim.pot, sim.invested.UTG, sim.invested.BB], [true, 3.5, 2, 1]);
  p.run("renderTable(simulate())");
  const seat = (q) => p.byId("felt").children.find(e => e.innerHTML.startsWith(`<span class="pos">${q}`)).innerHTML;
  assert.match(seat("UTG"), /<span class="stk"[^>]*>98<\/span>.*in 2/);
  assert.match(seat("SB"), /<span class="stk"[^>]*>99.5<\/span>.*in 0.5/);
  assert.match(seat("BTN"), /<span class="stk"[^>]*>150<\/span>/);
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

test("an import's toast (Review whole hand) goes when practice starts, and a review never runs on a practice hand", async () => {
  const p = loadPage();
  await p.ready();
  p.run(`toast("Imported PokerStars hand", null, 15000)`);
  assert.equal(p.byId("toast").hidden, false);
  vm.runInContext(PRACTICE, p.ctx, { filename: "practice.js" });
  p.run("Practice.set(true)");
  assert.equal(p.byId("toast").hidden, true);
  await playHand(p);
  const hand = JSON.stringify(p.S);
  // the toast's button, clicked before it was hidden: rewinding the practice hand to each of your decisions
  // would have the practice loop deal over it or move the other seats
  p.run("var seen = []; const __an = analyze; analyze = (sim) => { seen.push(JSON.stringify(S)); return __an(sim); };");
  await p.run("reviewHand()");
  assert.deepEqual(p.val("seen"), []);
  assert.equal(p.val("review"), null);
  assert.equal(JSON.stringify(p.S), hand);
});

test("practice setup and states: a fixed or random seat, the buttons wait for the solve, GTO opponents, the full analysis on request", async () => {
  const p = await practicePage(13, { "gto-opponents": "pool" });
  // a fixed seat: every deal from it, kept in the browser; Random again. The picker is a row of the site's
  // buttons (a radio group), not the browser's select: a click on one picks it
  const sel = p.byId("pxSeat"), pick = (seat) => sel.onclick({ target: { closest: () => ({ dataset: { seat } }) } });
  assert.match(sel.innerHTML, /<button type="button" role="radio" class="rnd" data-seat="random"[^>]*>Random<\/button>.*data-seat="BTN"[^>]*>BTN<\/button>/);
  assert.doesNotMatch(sel.innerHTML, /<select|<option/);
  pick("CO"); await flush();
  for (let k = 0; k < 3; k++) { assert.equal(p.S.hero, "CO"); p.run("Practice.next()"); await flush(); }
  assert.equal(JSON.parse(p.storage.get("gto-practice-seat")), "CO");
  pick("random"); await flush();
  const seats = new Set(); for (let k = 0; k < 8; k++) { p.run("Practice.next()"); await flush(); seats.add(p.S.hero); }
  assert.ok(seats.size >= 3, [...seats].join());
  // practice opponents play GTO: the pool setting isn't sent while practising
  assert.ok(p.posted.length && p.posted.every(m => !(m.state || m).opponents));
  // your turn before the spot is solved: "Solving your spot…", the buttons disabled; then "Your turn"
  const done = new Set(), plays = () => { for (const m of p.posted) if (m.play && !done.has(m.id)) { done.add(m.id); p.workers.at(-1).onmessage({ data: { id: m.id, result: passive(m) } }); } };
  for (let i = 0; i < 50 && p.sim().toAct !== p.S.hero; i++) { await flush(); plays(); await flush(); }
  assert.equal(p.sim().toAct, p.S.hero);
  {   // (only the opponents' requests answered so far: your spot's solve is still out)
    assert.ok(!p.val("resultKey === JSON.stringify(payload())"));
    const acts = p.byId("actionArea").children.find(e => /acts/.test(e.className));
    assert.match(acts.className, /wait/);
    assert.ok(acts.children.every(b => b.disabled));
    assert.match(p.byId("practiceBar").innerHTML, /Solving your spot…/);
  }
  for (let i = 0; i < 5 && !p.val("resultKey === JSON.stringify(payload())"); i++) { await flush(); answer(p, done, passive); await flush(); }
  const acts = p.byId("actionArea").children.find(e => /acts/.test(e.className));
  assert.ok(!/wait/.test(acts.className) && acts.children.every(b => !b.disabled));
  assert.match(p.byId("practiceBar").innerHTML, /Your turn/);
  // the solver's full analysis of your last decision shows only when asked for, and never on your turn
  assert.equal(p.act({ action: "call" }), null); p.run("renderAll()");
  assert.ok(!p.ctx.document.body.className.includes("pxdetail"));
  p.run("Practice.state.detail = true; renderAll()");
  assert.ok(p.ctx.document.body.className.includes("pxdetail"));
  p.run("Practice.next()"); await flush();
  assert.ok(!p.ctx.document.body.className.includes("pxdetail"));   // (nothing to show on a new hand)
});

// the solver checks or calls half the time (whichever you have), both at the best EV
const EVEN = { ...ADVICE, actions: [{ action: "fold", freq: 0.1, ev_bb: 0 }, { action: "check", freq: 0.5, ev_bb: 1 }, { action: "call", freq: 0.5, ev_bb: 1 }] };
const passMe = (sim) => sim.cur.curBet > sim.cur.bets[sim.toAct] ? { action: "call" } : { action: "check" };
// your first decision, its spot solved (the opponents check or call)
async function toMyTurn(p, done, reply = (m) => m.play ? passive(m) : ADVICE) {
  for (let i = 0; i < 50 && !(p.sim().toAct === p.S.hero && p.val("resultKey === JSON.stringify(payload())")); i++) {
    await flush();
    for (const m of p.posted) if (m.id != null && !done.has(m.id)) { done.add(m.id); p.workers.at(-1).onmessage({ data: { id: m.id, result: reply(m) } }); }
    await flush();
  }
  assert.equal(p.sim().toAct, p.S.hero);
}

test("looking ahead: once your spot is solved, the line where you check or call is asked for while you decide, and used when you take it", async () => {
  const p = await practicePage(11);
  const done = new Set(), reply = (m) => m.play ? passive(m) : EVEN;
  await toMyTurn(p, done, reply);
  const hero = p.S.hero, street = p.sim().current, mine = passMe(p.sim());
  await flush();
  // requests for the hand after your check or call (the browser looks ahead at Fast when the solver plays that
  // half the time or more), the page's own hand untouched
  const ahead = p.posted.filter(m => m.state && m.state[street].some(a => a.pos === hero));
  assert.ok(ahead.length >= 1, "a request goes out before you act");
  assert.deepEqual(ahead[0].state[street].find(a => a.pos === hero), { pos: hero, ...mine });
  assert.equal(p.S.actions[street].filter(a => a.pos === hero).length, 0, "the page's hand is untouched");
  // answer the look-ahead to its end (your next spot), then call: nothing is asked twice
  const answerAll = () => { for (const m of p.posted) if (m.id != null && !done.has(m.id)) { done.add(m.id); p.workers.at(-1).onmessage({ data: { id: m.id, result: reply(m) } }); } };
  for (let i = 0; i < 20; i++) { await flush(); answerAll(); }
  assert.equal(p.act(mine), null);
  for (let i = 0; i < 20 && !(p.sim().toAct === hero && p.val("resultKey === JSON.stringify(payload())")); i++) { await flush(); answerAll(); await flush(); }
  assert.equal(p.sim().toAct, hero);
  assert.ok(p.val("resultKey === JSON.stringify(payload())"), "your next spot is answered already");
  const key = (m) => JSON.stringify([m.state, m.play || null]), keys = p.posted.filter(m => m.id != null).map(key);
  assert.equal(new Set(keys).size, keys.length, "no request twice: the opponents' moves and your spot came from the look-ahead");
});

test("looking ahead in the browser only when the solver checks or calls your hand half the time or more; another move stops it", async () => {
  const RARE = { ...ADVICE, actions: [{ action: "fold", freq: 0.6, ev_bb: 0 }, { action: "check", freq: 0.4, ev_bb: 1 }, { action: "call", freq: 0.4, ev_bb: 1 }] };
  const p = await practicePage(11);
  const done = new Set();
  await toMyTurn(p, done, (m) => m.play ? passive(m) : RARE);
  await flush(); await flush();
  const street = p.sim().current;
  assert.ok(!p.posted.some(m => m.state && m.state[street].some(a => a.pos === p.S.hero)), "a check or call 40% of the time: no look-ahead");
  // at 50% it looks ahead; your fold then ends the hand and the look-ahead with it
  const q = await practicePage(11), d2 = new Set();
  await toMyTurn(q, d2, (m) => m.play ? passive(m) : EVEN);
  await flush();
  const st2 = q.sim().current;
  assert.ok(q.posted.some(m => m.state && m.state[st2].some(a => a.pos === q.S.hero)));
  assert.equal(q.act({ action: "fold" }), null);
  for (let i = 0; i < 5; i++) { await flush(); answer(q, d2, passive); }
  const m = q.posted.length; await flush(); answer(q, d2, passive); await flush();
  assert.equal(q.posted.length, m, "nothing more is asked once the hand is over");
  assert.ok(q.val("Practice.state.hand.result"));
});

test("Review mistakes: your decisions that gave up the most EV across hands, each opening its verdict; the dock's one-line verdict", async () => {
  const p = await practicePage(11);
  // before any decision the dock's line says where the verdict will be
  assert.match(p.byId("pxStrip").innerHTML, /Hand 1 · each decision is graded here/);
  const done = new Set();
  await toMyTurn(p, done);
  const hole = p.S.cards.slice(0, 2);
  p.key("n"); await flush();
  assert.deepEqual(p.S.cards.slice(0, 2), hole, "a stray N mid-hand doesn't throw the hand away");
  assert.equal(p.act({ action: "fold" }), null);   // 20% in the mix, 1bb below calling's EV
  p.run("renderAll()");
  assert.match(p.byId("pxStrip").innerHTML, /You folded/);
  const r = p.val("Practice.state.review");
  assert.equal(r.length, 1);
  assert.deepEqual([r[0].n, r[0].seat, r[0].rec, r[0].g.l], [1, p.S.hero, { action: "fold" }, 1]);
  assert.deepEqual(r[0].hole, p.S.cards.slice(0, 2));
  // a second hand: a call (the solver's best) is no mistake; the dock's line carries its verdict
  p.run("Practice.next()"); await flush();
  await toMyTurn(p, done, (m) => m.play ? passive(m) : EVEN);
  assert.equal(p.act(passMe(p.sim())), null); p.run("renderAll()");   // (in the mix at the best EV: no mistake)
  assert.match(p.byId("pxStrip").innerHTML, /(Best move|Good)<\/b><span class="muted">preflop · you (called|checked)/);
  let html = p.byId("practiceBar").innerHTML;
  assert.match(html, /<button class="small rev" id="pxRev"[^>]*>Review mistakes<span class="n">1<\/span><\/button>/);
  assert.doesNotMatch(html, /cx-rev/);
  // closed, a wide screen lists your recent decisions, newest first
  assert.match(html, /<div class="cx-recent">.*<li class="pick" data-i="1"[^>]*>.*<li class="pick" data-i="0"/);
  p.run("Practice.state.rev = true; renderAll()");   // (Review mistakes: the fake DOM can't find the bar's buttons)
  html = p.byId("practiceBar").innerHTML;
  assert.match(html, /<ul class="cx-list cx-rev"><li class="pick" data-i="0"[^>]*><span class="st">#1 preflop<\/span>.*fold.*−1.00 bb/);
  assert.equal((html.match(/<li class="pick/g) || []).length, 1, "only the mistake");
  p.run("Practice.state.pick = 0; renderAll()");
  html = p.byId("practiceBar").innerHTML;
  assert.match(html, /<li class="spot">.*<span class="verdict">Good<\/span>.*you <b>folded<\/b>/);
  assert.doesNotMatch(html.split('<li class="spot">')[1], /pxDetail/, "the full analysis is for your last decision only");
  // kept with the score in this browser; Reset clears both
  assert.equal(JSON.parse(p.storage.get("gto-practice-review")).length, 2);
  p.run("confirm = () => true");
  p.run("Practice.state.review = []; renderAll()");
  assert.doesNotMatch(p.byId("practiceBar").innerHTML, /pxRev/);
});
