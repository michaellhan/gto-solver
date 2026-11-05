// Review first (index.html's gate: handDone, reviewing, analyze): while a hand is entered your decisions are recorded
// and nothing goes to the solver. A hand is finished when it ended, when you folded, or when it was marked so, and each
// decision you recorded can then be reviewed, all together (reviewHand) or one by one (stepTo). Every route onto the
// table keeps to that.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage, state, acts, HTML_PATH } = require("./page.js");

const STATIC = path.join(__dirname, "..", "..", "gto", "static");
const HH = (f) => fs.readFileSync(path.join(__dirname, "..", "data", "hh", f), "utf8");
const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const j = (x) => JSON.parse(JSON.stringify(x));
const asked = (p) => p.posted.filter(m => m.state || m.presolve || m.prefetch || m.play);
const nothingAsked = (p) => assert.deepEqual(asked(p), []);
const OPENED = acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c");
const FLOP = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: OPENED, flop: acts("BB x") };
// BB checks, you bet 3, BB calls; BB checks, you check; BB bets, you call: four decisions, the hand over
const WHOLE = { cards: ["Ah", "Kh", "Kd", "7s", "2c", "9h", "3d"], preflop: OPENED, flop: acts("BB x, BTN b3, BB c"), turn: acts("BB x, BTN x"), river: acts("BB b8, BTN c") };
const answer = (m, x = {}) => ({ data: { id: m.id, result: { street: "flop", hero: "BTN", hand: "AKs", recommended: "check", pot_bb: 5.5,
  actions: [{ action: "check", freq: 0.7, ev_bb: 2 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.3, ev_bb: 1.9 }], notes: [], ...x } } });
async function page(opts = { isolated: true }) {
  const p = loadPage(opts);
  await p.ready();
  if (p.workers[0]) p.workers[0].onmessage({ data: { status: "Solving…" } });   // (the runtime is up: everything a pre-solve waited for)
  return p;
}
async function reviewed(st, opts) {   // the whole hand reviewed: each decision answered as it's asked
  const p = await page(opts);
  p.S = st;
  const done = p.run("reviewHand()");
  const solves = () => p.posted.filter(x => x.state && !x.state.equity);   // (the hand's equity follows the review)
  for (let n = 0; n < 60 && solves().length < 4 + 1; n++) {
    await flush();
    const m = solves().at(-1);
    if (m && !m.answered) { m.answered = true; p.workers.at(-1).onmessage(answer(m)); }
    if (p.val("review") === null) break;
  }
  await done; await flush();
  return p;
}
const ids = (p) => p.buttons().map(b => b.id).filter(Boolean);
const press = (p, id) => (p.buttons().find(b => b.id === id) || p.byId(id)).onclick();

test("a whole hand entered, button by button: the worker is asked nothing, on any street, whoever is to act", async () => {
  const p = await page();
  p.run('placeCard("Ah"); placeCard("Kh")');
  const check = async () => { await flush(); nothingAsked(p); assert.match(p.byId("result").innerHTML, /Finish the hand to see how each decision compares with the solver\./); };
  await check();
  assert.ok(p.buttons().every(b => !/best/.test(b.className)));
  p.button("Fold").onclick(); await check();     // UTG
  p.button("Fold to me").onclick(); await check();   // to you
  p.button("Raise to 2.5").onclick(); await check();
  p.button("Fold").onclick(); p.button("Call").onclick(); await flush();   // SB, BB
  for (const c of ["Kd", "7s", "2c"]) p.run(`placeCard("${c}")`);
  await check();
  p.button("Check").onclick(); await check();    // BB
  assert.match(p.buttons().find(b => /Check/.test(b.innerHTML)).innerHTML, /^Check/);   // your turn: the plain buttons, no mix
  assert.doesNotMatch(p.byId("actionArea").innerHTML, /·\s*\d+%/);
  assert.match(p.byId("actionArea").children[0].innerHTML, /Your action/);
  p.button("Bet 33%").onclick(); await check();
  p.button("Call").onclick(); p.run('placeCard("9h")'); await check();
  p.button("Check").onclick(); p.button("Check").onclick(); p.run('placeCard("3d")'); await check();
  p.button("Check").onclick(); p.button("Check").onclick();
  await flush();
  assert.equal(p.sim().handOver, true);
  nothingAsked(p);   // (over, it waits for you to ask)
  assert.deepEqual(ids(p), ["reviewAll"]);
  assert.equal(p.byId("reviewCount").textContent, "Your decisions");
  assert.equal((p.byId("adviceNote").innerHTML.match(/Not reviewed yet/g) || []).length, 4);
  assert.equal(p.buttons().find(b => b.id === "reviewAll").className, "nexth");
  assert.match(fs.readFileSync(HTML_PATH, "utf8"), /#actionArea \.nexth \{ display: block; width: 100%;/);   // (the page's main action, not a plain small button)
});

test("the panel shows nothing of a hand being entered, no grade or mix whatever it carries", async () => {
  const p = await page();
  const st = j(state(FLOP));
  st.actions.preflop[3].g = { m: "raise 2.5", f: 0.6, b: "raise 2.5", bf: 0.6, l: 0 };   // (a hand from before review first)
  p.S = st;
  await flush();
  assert.equal(p.byId("adviceNote").hidden, true);
  assert.match(p.byId("result").innerHTML, /Nothing is solved until then/);
  const line = [...p.byId("timeline").walk()].map(e => e.innerHTML).join(" ");
  assert.match(line, /BTN raises 2\.5/);
  assert.doesNotMatch(line, /class="g /);
  assert.equal(p.run("document.body.dataset.s"), "enter");   // (a phone keeps the table and dock first)
  nothingAsked(p);
});

test("rewinding a hand being entered brings no advice, nor do undo, a reload, or a saved or shared unfinished hand", async () => {
  const p = await page();
  p.S = state({ ...WHOLE, river: acts("BB b8") });   // you to act on the river
  await flush();
  p.run('rewindTo("flop", 1)');
  await flush();
  assert.equal(p.sim().toAct, "BTN");
  assert.deepEqual(p.val("S.actions.flop"), [{ pos: "BB", action: "check" }]);
  assert.match(p.byId("toast").innerHTML, /Back to your flop decision\. Enter it again\./);
  nothingAsked(p);
  p.run("undo()"); p.act({ action: "check" });
  await flush();
  nothingAsked(p);
  // a reload at your turn
  const again = await page({ isolated: true, storage: Object.fromEntries(p.storage) });
  await flush();
  assert.equal(again.sim().toAct, "BTN");
  nothingAsked(again);
  // a link to an unfinished hand, at load and in a tab already open; the page's loader
  const b64 = Buffer.from(JSON.stringify(state(FLOP))).toString("base64");
  const link = await page({ isolated: true, hash: "#h=" + b64 });
  await flush();
  assert.equal(link.sim().toAct, "BTN");
  nothingAsked(link);
  again.location.hash = "#h=" + b64; again.fire("hashchange");
  await flush();
  await again.run(`loadHandState(${JSON.stringify(state(FLOP))})`);
  await flush();
  assert.equal(again.sim().toAct, "BTN");
  nothingAsked(again);
});

test("the page has one way to ask for a hand's advice, and it's shut on a hand being entered", async () => {
  const p = await page();
  p.S = state(FLOP);
  await p.run("analyze(simulate())");   // (even called directly)
  await p.run("reviewHand()");          // (and a stale toast's Review whole hand)
  await flush();
  nothingAsked(p);
  assert.equal(p.val("review"), null);
});

test("Finish hand at your turn: what you recorded is reviewed, the decision you didn't record isn't", async () => {
  const p = await page();
  p.S = state(FLOP);   // your flop decision to come
  assert.ok(ids(p).includes("finishHand"));
  p.key("e");
  await flush();
  assert.equal(p.val("S.finished"), true);
  assert.deepEqual(ids(p), ["reviewAll", "keepEntering"]);
  assert.match(p.byId("actionArea").children[0].innerHTML, /Hand finished/);
  assert.match(p.byId("adviceNote").innerHTML, /Your flop decision wasn't recorded, so it isn't reviewed\./);
  assert.equal(p.run("document.body.dataset.s"), "enter");
  assert.ok(p.byId("felt").children.every(s => !/toact/.test(s.className) && !/to act/.test(s.innerHTML)));   // (nobody's to act now)
  nothingAsked(p);
  const done = p.run("reviewHand()");
  await flush();
  const asks = p.posted.filter(m => m.state);
  assert.equal(asks.length, 1);
  assert.equal(asks[0].state.board, "");            // your preflop open
  p.workers.at(-1).onmessage(answer(asks[0], { street: "preflop", actions: [{ action: "fold", freq: 0.4 }, { action: "raise 2.5", amount_bb: 2.5, freq: 0.6 }] }));
  await done; await flush();
  assert.equal(p.posted.filter(m => m.state).length, 1);   // nothing for the flop, then or after
  assert.equal(p.sim().toAct, "BTN");
  assert.equal(p.val("S.actions.flop.length"), 1);
  assert.equal(p.byId("reviewCount").textContent, "1 of 1 in the solver's mix");
  assert.match(p.byId("adviceNote").innerHTML, /Your flop decision wasn't recorded/);
  // nor by a hand-made step to it
  p.run('rewindTo("flop", 1)'); p.run('stepTo("flop", 0)');   // (past the end, and BB's check)
  await flush();
  assert.equal(p.val("stepView"), null);
  assert.equal(p.posted.filter(m => m.state).length, 1);
});

test("Finish hand goes back: before any advice Keep entering (or Undo) reopens it; a card isn't asked for once it's finished", async () => {
  const p = await page();
  p.S = state({ cards: FLOP.cards.slice(0, 2), preflop: OPENED });   // the flop to deal
  assert.equal(p.run("needCard(simulate())"), true);
  p.key("e");
  assert.equal(p.val("S.finished"), true);
  assert.equal(p.run("needCard(simulate())"), false);
  assert.equal(p.act({ action: "check" }), "Deal the next card first.");
  press(p, "keepEntering");
  assert.equal(p.val("S.finished"), undefined);
  assert.equal(p.run("needCard(simulate())"), true);
  p.S = state(FLOP); p.run("finishHand()");
  assert.equal(p.act({ action: "check" }), "This hand is finished.");
  p.run("undo()");
  assert.equal(p.val("S.finished"), undefined);
  assert.equal(p.val("S.actions.flop.length"), 1);
  p.run("undo()");
  assert.equal(p.val("S.actions.flop.length"), 0);   // (the next Undo takes an action back)
  nothingAsked(p);
});

test("your fold finishes the hand: nothing more to enter, and the review is there", async () => {
  const p = await page();
  p.S = state({ cards: FLOP.cards.slice(0, 2), preflop: acts("UTG r2.5, HJ c, CO f") });
  p.button("Fold").onclick();
  await flush();
  assert.equal(p.sim().handOver, false);
  assert.match(p.byId("actionArea").children[0].innerHTML, /You folded/);
  assert.deepEqual(ids(p), ["reviewAll"]);
  assert.equal(p.byId("felt").children.filter(s => /clickable/.test(s.className)).length, 0);
  assert.equal(p.run("needCard(simulate())"), false);
  nothingAsked(p);
});

test("a decision of a finished hand: its advice, the whole hand kept; a reload brings the hand back, not the spot", async () => {
  const p = await page();
  p.S = state(WHOLE);
  p.run('rewindTo("flop", 1)');
  await flush();
  assert.equal(p.val("stepView.s"), "flop");
  assert.deepEqual(p.val("S.actions.flop"), [{ pos: "BB", action: "check" }]);
  assert.equal(p.val("S.actions.turn.length"), 0);
  assert.equal(p.val("S.cards[5]"), null);
  const m = p.posted.filter(x => x.state).at(-1);
  assert.equal(m.state.board, "Kd7s2c");
  p.workers.at(-1).onmessage(answer(m));
  await flush();
  assert.match(p.byId("result").innerHTML, /<th>Solver<\/th>.*<tr class="best">.*<\/i>Check/s);
  assert.deepEqual(ids(p), []);   // (the dock is shut: the panel's Back, Previous and Next are the way on)
  assert.equal(p.byId("pAction").hidden, true);
  assert.equal(p.byId("stepTop").hidden, false);
  assert.equal(p.byId("stepNav").hidden, false);
  assert.ok(!p.byId("stepPrev").hidden && !p.byId("stepNext").hidden);
  assert.match(p.byId("stepHead").innerHTML, /<h2 class="street">Flop · /);
  assert.equal(p.byId("stepOf").textContent, "Decision 2 of 4");
  assert.equal(p.run("document.body.dataset.s"), "decision");   // (a phone shows this panel alone)
  // saved whole, and no move can be recorded from here
  assert.equal(JSON.parse(p.storage.get("handreview")).actions.river.length, 2);
  assert.equal(p.act({ action: "check" }), "This hand is finished.");
  // Next, Previous (and the arrow keys); past the ends nothing
  press(p, "stepNext");
  await flush();
  assert.equal(p.val("stepView.s"), "turn");
  p.key("ArrowLeft");
  await flush();
  assert.equal(p.val("stepView.s"), "flop");
  assert.equal(p.posted.filter(x => x.state && x.state.board === "Kd7s2c" && x.state.flop.length === 1).length, 1);   // (the answer in hand)
  // a reload: the whole hand, no decision's advice asked (a reviewed hand's equity is)
  const q = await page({ isolated: true, storage: Object.fromEntries(p.storage) });
  await flush();
  assert.equal(q.val("stepView"), null);
  assert.equal(q.sim().handOver, true);
  assert.deepEqual(asked(q).map(m => !!m.state.equity), [true]);
  // back to the hand: Undo here, as the button
  p.run("undo()");
  await flush();
  assert.equal(p.val("stepView"), null);
  assert.equal(p.val("S.actions.river.length"), 2);
  assert.equal(p.byId("result").className, "empty");
});

test("once its advice has been seen a hand stays as entered: no Undo, no cards, a rewind only looks", async () => {
  const p = await reviewed(state(WHOLE));
  const whole = j(p.val("S.actions"));
  assert.equal(p.val("S.reviewed"), true);
  assert.equal(p.byId("reviewCount").textContent, "3 of 3 in the solver's mix");   // (the fixture answer has no call, so the river stays ungraded)
  assert.match(p.byId("adviceNote").innerHTML, /Reviewed, no grade/);
  assert.equal(p.byId("reviewAgain").hidden, false);
  assert.equal(p.byId("result").className, "empty");   // (no decision's advice left under the whole hand)
  p.run("undo()");
  assert.deepEqual(j(p.val("S.actions")), whole);
  assert.match(p.byId("toast").innerHTML, /stays as entered/);
  p.run("removeCard(6)");
  assert.equal(p.val("S.cards[6]"), "3d");
  assert.equal(p.byId("editCards").hidden, true);
  p.run('rewindTo("flop", 1)');
  await flush();
  assert.equal(p.val("stepView.s"), "flop");
  assert.deepEqual(j(p.val("stepView.full.actions")), whole);
  assert.equal(p.posted.filter(m => m.state && !m.state.equity).length, 4);   // (the review's answer, not a new solve)
  p.run("stepStop()");
  // the mark survives a reload and a link
  const q = await page({ isolated: true, storage: Object.fromEntries(p.storage) });
  q.run("undo()");
  assert.deepEqual(j(q.val("S.actions")), whole);
  // a new hand is a new hand
  p.click("newhand");
  assert.equal(p.val("S.reviewed"), undefined);
  assert.equal(p.val("S.finished"), undefined);
});

test("a review cancelled before any advice leaves the hand open; one cancelled after keeps the line", async () => {
  const p = await page();
  p.S = { ...state(FLOP), finished: true };
  const done = p.run("reviewHand()");
  await flush();
  p.run('review.cancel = "restore"; cancelSolve()');
  p.workers.at(-1).onmessage({ data: { id: p.posted.filter(m => m.state).at(-1).id, result: { cancelled: true } } });
  await done; await flush();
  assert.equal(p.val("S.reviewed"), undefined);
  assert.ok(ids(p).includes("keepEntering"));
  // a review shows no approximate answer, its rows fill in as it goes: none seen, so none counts
  const q = await page();
  q.S = state(WHOLE);
  const d2 = q.run("reviewHand()");
  await flush();
  const m = q.posted.filter(x => x.state).at(-1);
  const partial = { street: "flop", hero: "BTN", recommended: "check", approximate: { source: "prev", label: "x" }, actions: [{ action: "check", freq: 1 }] };
  q.workers.at(-1).onmessage({ data: { id: m.id, partial } });
  await flush();
  assert.doesNotMatch(q.byId("result").children.map(c => c.innerHTML).join(""), /Approximate/);
  q.run('review.cancel = "restore"; cancelSolve()');
  q.workers.at(-1).onmessage({ data: { id: m.id, result: { cancelled: true } } });
  await d2; await flush();
  assert.equal(q.val("S.reviewed"), undefined);
  // on a decision opened it shows, and that's advice seen
  const r = await page();
  r.S = state(WHOLE);
  r.run('stepTo("flop", 1)');
  await flush();
  const o = r.posted.filter(x => x.state).at(-1);
  r.workers.at(-1).onmessage({ data: { id: o.id, partial } });
  await flush();
  assert.match(r.byId("result").children.map(c => c.innerHTML).join(""), /Approximate/);
  assert.equal(r.val("stepView.full.reviewed"), true);
});

// The review's run against answers: ans(message, nth solve)
async function runReview(p, ans) {
  const done = p.run("reviewHand()");
  const solves = () => p.posted.filter(x => x.state && !x.state.equity);
  for (let n = 0; n < 60; n++) {
    await flush();
    const m = solves().at(-1);
    if (m && !m.answered) { m.answered = true; p.workers.at(-1).onmessage(ans(m, solves().length)); }
    if (p.val("review") === null) break;
  }
  await done; await flush();
}
const failed = (m) => ({ data: { id: m.id, error: "solver error: boom" } });

test("one decision's solve fails in a review: the review ends, that row has no grade, and the page goes on", async () => {
  const p = await page();
  p.S = state(WHOLE);
  await runReview(p, (m, n) => n === 2 ? failed(m) : answer(m));
  assert.equal(p.val("review"), null);
  assert.equal(p.val("solving"), false);
  assert.equal(p.val("S.actions.flop[1].g"), null);
  assert.ok(p.val("S.actions.preflop.find(a => a.pos === S.hero).g"));
  assert.match(p.byId("adviceNote").innerHTML, /<b>Flop<\/b> · bet 3<\/span><span class="l"><\/span><span class="s">Reviewed, no grade<\/span>/);
  p.run("renderAll()");                                 // (a later render reads the rows again)
  assert.equal(p.run("document.body.dataset.s"), "review");
  assert.equal(p.byId("reviewAgain").hidden, false);    // (and it can be asked again)
  p.run("reviewHand()");
  await flush();
  assert.ok(p.val("review") !== null);
});

test("every solve fails in a review: no advice was seen, so the hand can be reviewed again", async () => {
  const p = await page();
  p.S = state(WHOLE);
  await runReview(p, failed);
  assert.equal(p.val("review"), null);
  assert.equal(p.val("S.reviewed"), undefined);
  assert.equal(p.run("document.body.dataset.s"), "enter");
  assert.equal(p.byId("reviewAgain").hidden, true);
  assert.deepEqual(p.buttons().map(b => b.id), ["reviewAll"]);
  const before = p.posted.filter(x => x.state).length;
  p.key("r");
  await flush();
  assert.ok(p.val("review") !== null);                  // (R goes to it as well)
  assert.ok(p.posted.filter(x => x.state).length > before);
});

test("the betting rules' hand with no grade to give stays reviewed: no second way to review it", async () => {
  const p = await page();
  p.S = state(FLOP);
  p.run("finishHand()");
  await runReview(p, (m) => answer(m, { multiway: true, actions: [{ action: "check", freq: 1 }] }));
  assert.equal(p.val("S.reviewed"), true);
  assert.equal(p.byId("reviewAgain").hidden, false);
  assert.deepEqual(p.buttons().map(b => b.id).filter(x => x === "reviewAll"), []);
});

test("during a review the total waits: no 0.00 before the costly decision is graded", async () => {
  const p = await page();
  p.S = state(WHOLE);
  const done = p.run("reviewHand()");
  await flush();
  p.workers.at(-1).onmessage(answer(p.posted.filter(x => x.state).at(-1)));
  await flush();
  assert.ok(p.val("review.k") >= 2);
  assert.doesNotMatch(p.byId("adviceNote").innerHTML, /<strong/);
  p.run('review.cancel = "restore"; cancelSolve()');
  await done; await flush();
  assert.match(p.byId("adviceNote").innerHTML, /<strong/);
});

test("focus goes with the screen: into a decision to its Back, back to the row; Prev and Next keep theirs; the count is announced", async () => {
  const p = await page();
  p.S = state(WHOLE);
  await runReview(p, (m) => answer(m));
  p.run('stepTo("flop", 1)');
  assert.equal(p.focused(), p.byId("stepBack"));
  p.byId("stepNext").focus();
  p.run("stepBy(1)");
  assert.equal(p.focused(), p.byId("stepNext"));
  assert.match(fs.readFileSync(HTML_PATH, "utf8"), /id="reviewCount" aria-live="polite"/);
  const q = await page();
  q.S = state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: OPENED, flop: acts("BB x, BTN x") });
  q.button("Finish hand").onclick();
  assert.equal(q.focused(), q.buttons().find(b => b.id === "reviewAll"));
});

test("the setup editor stops the dock floating over the page", async () => {
  const p = await page();
  p.S = state({ cards: ["Ah", "Kh"] });
  assert.equal(p.run('document.body.classList.contains("editing")'), false);
  p.byId("setupEdit").onclick();
  assert.equal(p.run('document.body.classList.contains("editing")'), true);
  assert.match(fs.readFileSync(HTML_PATH, "utf8"), /body\.editing \.dock \{ position: static; \}/);
  p.byId("setupDone").onclick();
  assert.equal(p.run('document.body.classList.contains("editing")'), false);
});

test("a saved lock goes with a decision's request; Randomise and recording off the advice are gone", async () => {
  const html = fs.readFileSync(HTML_PATH, "utf8");
  assert.doesNotMatch(html, /randomise|Randomise/);
  assert.doesNotMatch(html, /data-act=|applyAction/);
  const p = await page();
  p.spot({ ...state(FLOP), locks: { BB: { fold: 15 } } });
  await flush();
  assert.deepEqual(j(p.posted.filter(m => m.state).at(-1).state.locks), { BB: { fold: 15 } });
});

test("a hand hh.js read reviews straight away", async () => {
  const p = await page({ isolated: true });
  vm.runInContext(fs.readFileSync(path.join(STATIC, "hh.js"), "utf8"), p.ctx, { filename: "hh.js" });
  p.ctx.__t = HH("stars_6max_srp.txt");
  await p.run("loadHandState(importHand(__t).state)");
  await flush();
  assert.equal(p.val("S.finished"), true);
  assert.equal(p.byId("reviewCount").textContent, "Your decisions");
  assert.ok(ids(p).includes("reviewAll") && !ids(p).includes("keepEntering"));   // (it's all there: nothing more to enter)
  nothingAsked(p);   // (it waits to be asked)
  p.run('rewindTo("flop", 1)');
  await flush();
  assert.equal(p.posted.filter(m => m.state).length, 1);
  assert.equal(p.posted.filter(m => m.state)[0].state.board, "Qs8d3c");
  // cut short by the history (the flop only) it's finished all the same, and a mid-hand import asks no card
  const st = p.run("importHand").call(null, HH("stars_6max_srp.txt")).state;
  assert.equal(st.finished, true);
});

test("the flags: true or nothing on what's loaded, left out of a hand's identity in My hands", async () => {
  const p = await page();
  assert.equal(p.val(`cleanState(${JSON.stringify({ ...state(FLOP), finished: "yes", reviewed: 1 })}).finished`), undefined);
  assert.equal(p.val(`cleanState(${JSON.stringify({ ...state(FLOP), finished: "yes", reviewed: 1 })}).reviewed`), undefined);
  assert.equal(p.val(`cleanState(${JSON.stringify({ ...state(FLOP), finished: true, reviewed: true })}).finished`), true);
  const { createStore } = require(path.join(STATIC, "store.js"));
  const { fakeIndexedDB, fakeLocalStorage } = require("./idb_shim.js");
  const H = createStore({ indexedDB: fakeIndexedDB(), localStorage: fakeLocalStorage() }).hands;
  const st = state(FLOP);
  assert.equal(H.sig({ ...st, finished: true, reviewed: true }), H.sig(st));
});

test("a share link carries a finished hand as finished, and from a decision looked at the whole hand", async () => {
  const p = await page();
  p.S = state(WHOLE);
  p.run('rewindTo("turn", 1)');
  await flush();
  const b64 = p.run("shareLink()").match(/#h=([A-Za-z0-9+/=]+)/)[1];
  const st = JSON.parse(Buffer.from(b64, "base64").toString());
  assert.equal(st.actions.river.length, 2);
  const q = await page({ isolated: true, hash: "#h=" + b64 });
  await flush();
  assert.equal(q.sim().handOver, true);
  assert.deepEqual(ids(q), ["reviewAll"]);
  nothingAsked(q);
});

test("My hands: an imported or reviewed hand opens finished, one left part way opens as it was, without advice", async () => {
  const p = loadPage({ store: true, isolated: true });
  await p.ready();
  await p.run("GTOStore.ready");
  p.workers[0].onmessage({ data: { status: "Solving…" } });
  vm.runInContext(fs.readFileSync(path.join(STATIC, "hands.js"), "utf8"), p.ctx, { filename: "hands.js" });
  const open = async (st, source) => {
    const rec = j(await p.run(`GTOStore.hands.save(${JSON.stringify(st)}, ${JSON.stringify({ source })})`));
    await p.run(`(async () => MyHands.pageHost().open(await GTOStore.hands.get(${JSON.stringify(rec.id)})))()`);
    await flush();
    return rec;
  };
  await open(state(FLOP), "table");
  assert.equal(p.sim().toAct, "BTN");
  assert.match(p.byId("toast").innerHTML, /It isn't finished: enter the rest, or press Finish hand, to review it\./);
  assert.equal(p.byId("toast").children.filter(e => e.textContent === "Review whole hand").length, 0);
  nothingAsked(p);
  await open(state({ ...FLOP, cards: ["9s", "9d", "Kd", "7s", "2c"] }), "import");   // (kept before hands were marked finished)
  assert.equal(p.val("S.finished"), true);
  assert.equal(p.byId("toast").children.filter(e => e.textContent === "Review whole hand").length, 1);
  nothingAsked(p);
  p.click("newhand");
});

test("a review that ends with the betting closed asks for your equity once, after the decisions; a hand being entered never does", async () => {
  const p = await reviewed(state(WHOLE));
  await flush();
  const eq = p.posted.filter(m => m.state && m.state.equity);
  assert.equal(eq.length, 1);
  assert.equal(eq[0].state.board, "Kd7s2c9h3d");
  assert.ok(p.posted.indexOf(eq[0]) > p.posted.findLastIndex(m => m.state && !m.state.equity));
  const q = await page();
  q.S = state({ ...FLOP, flop: acts("BB x, BTN x") });
  await flush();
  nothingAsked(q);
  assert.equal(q.byId("equity").hidden, true);
});


// The Review panel's first-visit line: the download while it's happening, in a browser that hasn't got it yet.
test("the first visit's download shows while the solver loads, not on a later visit, a server, or once it's up", async () => {
  const ready = (p) => { p.run("resetAdvice(); updateAdviceState()"); return p.byId("solverReady").hidden ? null : p.byId("solverReady").textContent; };
  const up = (p) => p.workers.at(-1).onmessage({ data: { status: "Solving…" } });
  const first = loadPage();
  await first.ready();
  assert.equal(ready(first), "Getting the solver ready, about 9 MB on first visit.");
  first.workers.at(-1).onmessage({ data: { status: "Loading numpy (2 of 3)…" } });
  assert.equal(ready(first), "Getting the solver ready, about 9 MB on first visit.");
  up(first);
  assert.equal(ready(first), null);
  // the worker's bundle in the Cache API, of any version: this browser has been here
  let asked = null;
  const later = loadPage({ caches: { match: async (...a) => { asked = a; return {}; } } });
  await later.ready(); await flush();
  assert.equal(ready(later), null);
  assert.equal(asked[0], "solver.zip"); assert.equal(asked[1].ignoreSearch, true);
  // none there, or the Cache API refusing: a first visit
  for (const match of [async () => undefined, async () => { throw new Error("denied"); }]) {
    const p = loadPage({ caches: { match } });
    await p.ready(); await flush();
    assert.ok(ready(p));
  }
  const server = loadPage({ server: true });
  await server.ready(); await flush();
  assert.equal(ready(server), null);
  // a load that failed says so where it's asked for, not here
  const failed = loadPage();
  await failed.ready();
  failed.workers.at(-1).onerror({ message: "x" });
  assert.equal(ready(failed), null);
});
