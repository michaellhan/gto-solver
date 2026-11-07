// A hand outside your range with its top EVs together (Advisor's off_range and close): the answer reads as close, a move
// among them isn't a mistake in the grade, the review, Copy hand or My hands, and the solver's numbers stay.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage, state, acts, HTML_PATH } = require("./page.js");

const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const OFF = "your hand isn't in your range for this line, so it was solved as a near-zero-weight combo";
// The hand that found it (7-handed, A4s, 2s4d6c 6h after a bet and a call): 100% on an edge of 1.3% of the pot
const TURN = { street: "turn", hero: "BB", villain: "LJ", hand: "A4s", pot_bb: 29.5, to_call_bb: 0, recommended: "bet 75%", sampled: "bet 75%",
  off_range: true, close: { actions: ["bet 75%", "bet 33%", "check"], within_pct: 3, gap_pct: 2.2 }, notes: [OFF], solver: { exploitability_pct: 0.4, iterations: 200 },
  actions: [{ action: "check", freq: 0, ev_bb: 6.24 }, { action: "bet 33%", amount_bb: 9.5, freq: 0, ev_bb: 6.51 }, { action: "bet 75%", amount_bb: 22, freq: 1, ev_bb: 6.89 }],
  grid: { actions: ["check", "bet 33%", "bet 75%"], cells: {} } };
const POOL = { ...TURN, recommended: "bet 33%", sampled: "bet 33%", close: { actions: ["bet 33%", "bet 75%"], within_pct: 3, gap_pct: 1.3 },
  actions: [{ action: "check", freq: 0, ev_bb: 8.41, gto_freq: 0 }, { action: "bet 33%", amount_bb: 9.5, freq: 1, ev_bb: 9.75, gto_freq: 0 }, { action: "bet 75%", amount_bb: 22, freq: 0, ev_bb: 9.37, gto_freq: 1 }] };
const SPOT = { players: 7, hero: "BB", stack: 100, cards: ["As", "4s", "2s", "4d", "6c", "6h", null], actions: { preflop: [], flop: [], turn: [], river: [] } };
const grade = (p, r, rec, tournament = false) => { p.run(`S.tournament = ${tournament}`); return p.val(`gradeWith(${JSON.stringify(r)}, ${JSON.stringify(rec)})`); };

test("a close move isn't a mistake: ✓, no bb lost, the EVs kept, at equilibrium and against the typical player", () => {
  const p = loadPage();
  const g = grade(p, TURN, { action: "check" }, true);                                   // (a tournament: by the mix, 0%)
  assert.deepEqual([g.m, g.f, g.k, g.l, g.cl, g.e, g.o, g.c, g.cg], ["check", 0, 1, 0, 0.65, 6.24, 6.89, TURN.close.actions, 2.2]);
  assert.equal(p.val(`gradeClass(${JSON.stringify(g)})`), "ok");
  assert.equal(p.val(`gradeMark(${JSON.stringify(g)})`), "✓");
  assert.equal(p.val(`inMix(${JSON.stringify(g)})`), true);
  assert.match(p.run(`lossHtml(${JSON.stringify(g)})`), /0\.00 bb/);
  assert.equal(p.run(`verdictText(${JSON.stringify(g)})`), "Close: bet 75%, bet 33% or check, within 2.2% of the pot. 0.65 bb behind the top EV. Your hand isn't in your range here, so the solver has no real mix for it: only its EVs count.");
  const t = grade(p, POOL, { action: "bet", to: 22 });                                   // (cash: by the EV given up)
  assert.deepEqual([t.p, t.k, t.l, t.cl, t.h], [1, 1, 0, 0.38, 1.3]);
  assert.equal(p.val(`gradeClass(${JSON.stringify(t)})`), "ok");
});

test("the betting rules' answer to such a hand reads the same", () => {
  const p = loadPage();
  const g = grade(p, { ...TURN, multiway: true }, { action: "check" });
  assert.deepEqual([g.h, g.k, g.l, g.cl, g.p], [2.2, 1, 0, 0.65, undefined]);
  assert.equal(p.run(`decisionLine(${JSON.stringify({ pos: "BB", action: "check", g })}, 29.5)`), "  Me: check. Betting rules: close, bet 75%, bet 33% or check (within 2.2% of the pot).");
});

test("a move outside the close ones is graded as ever, and says what was close", () => {
  const p = loadPage();
  const t = grade(p, POOL, { action: "check" });                                         // (4.5% of the pot, against the pool)
  assert.deepEqual([t.k, t.l, t.h, t.c], [undefined, 1.34, 4.5, POOL.close.actions]);
  assert.equal(p.val(`gradeClass(${JSON.stringify(t)})`), "meh");
  assert.match(p.run(`verdictText(${JSON.stringify(t)})`), /the best play is bet 33%\. 1\.34 bb lost, 4\.5% of the pot\. Close: bet 33% or bet 75%\.$/);
});

test("an answer that isn't marked close is graded as it was", () => {
  const p = loadPage();
  const { off_range, close, ...plain } = TURN;
  const g = grade(p, plain, { action: "check" }, true);
  assert.deepEqual([g.k, g.c, g.l, g.b], [undefined, undefined, 0.65, "bet 75%"]);
  assert.equal(p.val(`gradeClass(${JSON.stringify(g)})`), "bad");
});

test("the mix can lag the EVs: the grade goes by the answer's top EV, not the 100%", () => {
  const p = loadPage();
  const r = { ...TURN, recommended: "bet 33%", sampled: "bet 33%", close: undefined, pot_bb: 8,
    actions: [{ action: "check", freq: 1, ev_bb: 1.92 }, { action: "bet 33%", amount_bb: 2.5, freq: 0, ev_bb: 3.83 }] };
  const top = grade(p, r, { action: "bet", to: 2.5 }, true), mix = grade(p, r, { action: "check" }, true);
  assert.deepEqual([top.b, top.bf, top.k, top.l], ["bet 33%", 0, 1, 0]);
  assert.equal(p.val(`gradeClass(${JSON.stringify(top)})`), "ok");
  assert.match(p.run(`verdictText(${JSON.stringify(top)})`), /^The highest EV\./);
  assert.deepEqual([mix.b, mix.k, mix.l], ["bet 33%", undefined, 1.91]);
});

test("the answer on the table: Close first, the solver's numbers under it, every close action in bold", () => {
  const p = loadPage();
  p.S = SPOT;
  p.run(`renderResult(${JSON.stringify(TURN)}, JSON.stringify(payload()))`);
  const h = p.byId("result").innerHTML;
  assert.match(h, /<div class="verdict"><div class="hd">Close: bet 75%, bet 33% or check<\/div><p>Within 2\.2% of the pot\. Your hand isn't in your range here/);
  assert.ok(h.indexOf("Close:") < h.indexOf('class="acttbl"'));
  assert.match(h, /100%/);                                                               // (the solver's number stays)
  assert.equal((h.match(/<tr class="best">/g) || []).length, 3);
  const { off_range, close, ...plain } = TURN;
  p.run(`renderResult(${JSON.stringify(plain)}, JSON.stringify(payload()))`);
  assert.doesNotMatch(p.byId("result").innerHTML, /Close:/);
  assert.equal((p.byId("result").innerHTML.match(/<tr class="best">/g) || []).length, 1);
});

test("Copy hand says close, at your turn and in a decision", () => {
  const p = loadPage();
  p.S = SPOT;
  p.run(`lastResult = ${JSON.stringify(TURN)}; resultKey = JSON.stringify(payload())`);
  assert.equal(p.run("shownAdvice()"), "  Solver: close, bet 75%, bet 33% or check (within 2.2% of the pot). Mix bet 75% (100%)");
  const line = (a) => p.run(`decisionLine(${JSON.stringify(a)}, 29.5)`);
  const g = grade(p, TURN, { action: "check" }, true);
  assert.equal(line({ pos: "BB", action: "check", g }), "  Me: check. Solver: close, bet 75%, bet 33% or check (within 2.2% of the pot).");
  const t = grade(p, POOL, { action: "check" });
  assert.equal(line({ pos: "BB", action: "check", g: t }), "  Me: check. Against a typical player: close, bet 33% or bet 75% (within 1.3% of the pot). Lost 1.34bb.");
});

test("a review's row, count and total: a close move is good and costs nothing", async () => {
  const p = loadPage();
  await p.ready();
  const hand = { ...state({ cards: ["As", "4s", "2s", "4d", "6c", "6h", "Jd"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB b2, BTN c"), turn: acts("BB x, BTN x"), river: acts("BB x, BTN x") }), hero: "BB", finished: true };
  p.S = hand;
  const g = grade(p, TURN, { action: "check" }, true);
  p.run(`S.actions.turn[0].g = ${JSON.stringify(g)}; S.reviewed = true; renderAll()`);
  await flush();
  const n = p.byId("adviceNote").innerHTML;
  assert.match(n, /Close: bet 75%, bet 33% or check/);
  assert.match(n, /class="i g ok">✓/);
  assert.doesNotMatch(n, /−0\.65 bb/);
});

test("My hands reads the same: ✓ with no gradeClass to hand, the close actions in its title", () => {
  const MH = require(path.join(path.dirname(HTML_PATH), "hands.js"));
  const st = { players: 6, hero: "BB", cards: ["As", "4s", "2s", "4d", "6c", "6h", null], stack: 100, actions: { preflop: [], flop: [], river: [],
    turn: [{ pos: "BB", action: "check", g: { m: "check", f: 0, b: "bet 75%", bf: 1, l: 0, cl: 0.65, e: 6.24, o: 6.89, k: 1, c: ["bet 75%", "check"], cg: 2.2 } }] } };
  assert.match(MH.lineHtml(st), /<span class="mh-g ok" title="close: bet 75% or check">✓/);
  assert.ok(fs.existsSync(HTML_PATH));
});
