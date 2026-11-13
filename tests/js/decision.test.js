// The right panel's two answers (index.html: rowHtml, updateAdviceState; verdictHtml, actionTable, sourceOf, rangeHtml,
// renderStepHead): the review of a finished hand and a decision of it, each saying where its numbers came from.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const { loadPage, state, acts, HTML_PATH } = require("./page.js");

const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const HAND = { cards: ["Ah", "Kd", "Ks", "7d", "2c", "5h"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x, BTN b2, BB c"), turn: acts("BB x, BTN x") };
const PRE = { street: "preflop", hero: "BTN", hand: "AKo", recommended: "raise", notes: ["preflop: chart approximation"],
  actions: [{ action: "raise", amount_bb: 2.5, freq: 1 }, { action: "fold", freq: 0 }],
  grid: { actions: ["raise", "fold"], cells: { AKo: { w: 1, f: [1, 0] }, "72o": { w: 1, f: [0, 1] } } } };
const FLOP = { street: "flop", hero: "BTN", villain: "BB", hand: "AKo", recommended: "bet 33%", pot_bb: 5.5, precomputed: { label: "BTN vs BB", exploitability_pct: 0.11 },
  actions: [{ action: "check", freq: 0.04, ev_bb: 8.25 }, { action: "bet 33%", amount_bb: 2, freq: 0.63, ev_bb: 8.25 }, { action: "bet 75%", amount_bb: 4, freq: 0.33, ev_bb: 8.25 }] };
const TURN = { street: "turn", hero: "BTN", villain: "BB", hand: "AKo", recommended: "bet 125%", pot_bb: 9.5, solver: { exploitability_pct: 0.08, iterations: 300 },
  actions: [{ action: "check", freq: 0, ev_bb: 13.34 }, { action: "bet 33%", amount_bb: 3, freq: 0, ev_bb: 13.72, rough_ev: true }, { action: "bet 75%", amount_bb: 7, freq: 0, ev_bb: 14.01 }, { action: "bet 125%", amount_bb: 12, freq: 1, ev_bb: 14.08 }],
  villain_range_combos: 172.5, grid: { actions: ["check", "bet 33%", "bet 75%", "bet 125%"], cells: { AKo: { w: 0.5, f: [0.5, 0, 0, 0.5] }, KK: { w: 1, f: [1, 0, 0, 0] } } } };
async function page(opts = { isolated: true }) {
  const p = loadPage(opts);
  await p.ready();
  if (p.workers[0]) p.workers[0].onmessage({ data: { status: "Solving…" } });
  return p;
}
async function reviewed(st = { ...state(HAND), finished: true, tournament: true }) {   // (a tournament: at equilibrium)
  const p = await page();
  p.S = st;
  await flush();
  const line = [...p.byId("timeline").walk()].map(e => e.innerHTML).join(" ");
  assert.doesNotMatch(line, /class="g /);   // (before the review, no marks)
  const done = p.run("reviewHand()");
  for (let n = 0; n < 60; n++) {
    await flush();
    const m = p.posted.filter(x => x.state && !x.state.equity).at(-1);
    if (m && !m.answered) { m.answered = true; p.workers.at(-1).onmessage({ data: { id: m.id, result: [PRE, 0, 0, FLOP, TURN][m.state.board.length / 2] } }); }
    if (m && p.val("review") === null) break;
  }
  await done; await flush();
  return p;
}

test("the review: the count, the hand and its total, a row a decision, the marks on the timeline, and nothing more stored", async () => {
  const p = await reviewed();
  assert.equal(p.byId("reviewCount").textContent, "2 of 3 in the solver's mix");
  assert.equal(p.run("document.body.dataset.s"), "review");
  const n = p.byId("adviceNote").innerHTML;
  assert.match(n, /<div class="sum hand"><span><span class="inl">A<i class="r">♥︎<\/i> K<i class="r">♦︎<\/i><\/span> on <span class="inl">K<i>♠︎<\/i> /);
  assert.match(n, /<strong class="bad">−0\.74 bb<\/strong>/);
  assert.match(n, /<span class="i g ok">✓<\/span><span class="t"><b>Preflop<\/b> · raise to 2\.5<\/span><span class="l dim">no EV<\/span><span class="s">Solver 100%, from the chart<\/span>/);
  assert.match(n, /<span class="i g ok">✓<\/span><span class="t"><b>Flop<\/b> · bet 33%<\/span><span class="l">0\.00 bb<\/span><span class="s">Solver 63%, precomputed<\/span>/);
  assert.match(n, /<span class="i g bad">✗<\/span><span class="t"><b>Turn<\/b> · check<\/span><span class="l bad">−0\.74 bb<\/span><span class="s">Solver 0%, it bets 125%\. Solved live<\/span>/);
  assert.equal((n.match(/<button class="rowb"/g) || []).length, 3);
  const line = [...p.byId("timeline").walk()].map(e => e.innerHTML).join(" ");
  assert.match(line, /class="g ok"/);
  assert.match(line, /class="g bad"/);
  assert.equal(p.byId("doneBtns").hidden, false);
  assert.equal(p.byId("doneNew").hidden, false);
  assert.equal(p.byId("pAction").hidden, true);
  // where an answer came from isn't saved: what's stored is what was
  const stored = JSON.parse(p.storage.get("handreview"));
  assert.deepEqual(Object.keys(stored.actions.turn.find(a => a.pos === "BTN").g).sort(), ["b", "bf", "e", "f", "l", "m", "o"]);
  const q = await page({ isolated: true, storage: Object.fromEntries(p.storage) });
  await flush();
  const again = q.byId("adviceNote").innerHTML;
  assert.match(again, /Solver 100%, from the chart/);   // (a chart's answer is known without it)
  assert.match(again, /<span class="s">Solver 63%<\/span>/);
  assert.match(again, /<span class="s">Solver 0%, it bets 125%<\/span>/);
});

test("a cash hand is graded against the typical player: by the EV given up, with GTO under each action", async () => {
  const p = await reviewed({ ...state(HAND), finished: true });
  assert.equal(p.payload().opponents, "pool");
  assert.match(p.byId("reviewCount").textContent, /of 3 good against a typical player$/);
  const n = p.byId("adviceNote").innerHTML;
  assert.match(n, /Best against a typical player/);                                 // (a tie: your action counts as the best)
  assert.match(n, /Against a typical player, bet 125% is best/);
  assert.doesNotMatch(n, /in the solver's mix|Solver 0%/);
  const g = p.val("S.actions.turn[1].g");
  assert.deepEqual([g.p, g.l, g.b], [1, 0.74, "bet 125%"]);
  assert.equal(p.run(`gradeClass(${JSON.stringify(g)})`), "meh");                   // (7.8% of the pot)
  assert.match(p.run(`verdictText(${JSON.stringify(g)})`), /Against a typical player the best play is bet 125%\. 0\.74 bb lost, 7\.8% of the pot\./);
  assert.match(p.run("handText()"), /Against a typical player: bet 125%\. Lost 0\.74bb\./);
  // EVs inside the answer's ev_tie are tied, and the advice's own action is the best of a tie
  const graded = (m, tie) => p.run(`gradeWith({ pot_bb: 10, recommended: "bet 33%", ${tie == null ? "" : `ev_tie: ${tie}, `}actions: [
    { action: "check", freq: 0, ev_bb: 9.6 }, { action: "bet 33%", freq: 1, ev_bb: 9.95, amount_bb: 7 }, { action: "bet 75%", freq: 0, ev_bb: 9.97, amount_bb: 15 }] }, ${JSON.stringify(m)})`);
  const small = { action: "bet", to: 7 }, big = { action: "bet", to: 15 };
  let t = graded(small, 0.1);
  assert.deepEqual([t.b, t.l, t.o], ["bet 33%", 0, 9.95]);
  t = graded(big, 0.1);                                                       // (the other side of the tie has lost nothing either)
  assert.deepEqual([t.b, t.l], ["bet 75%", 0]);
  t = graded({ action: "check" }, 0.1);
  assert.deepEqual([t.b, t.l, t.h], ["bet 33%", 0.35, 3.5]);
  t = graded(small, 0.01);                                                      // (past the tie the top EV is the best)
  assert.deepEqual([t.b, t.l], ["bet 75%", 0.02]);
  t = graded(small);                                                            // (an answer without one, as before)
  assert.deepEqual([t.b, t.l], ["bet 75%", 0.02]);
  // a hand you'd already reviewed keeps its grade until you review it again
  assert.equal(p.run(`gradeClass({ f: 0.2, l: 1 })`), "ok");
  p.run('rewindTo("turn", 1)');
  await flush();
  const h = p.byId("result").innerHTML;
  const withGto = { ...TURN, actions: TURN.actions.map((a, i) => ({ ...a, gto_freq: i === 3 ? 1 : 0 })) };
  assert.match(p.run(`actionTable(${JSON.stringify(withGto)}, null)`), /<small>GTO 100%<\/small>/);
  assert.match(h, /Against a typical online player, who folds too much to bets/);
});

test("the typical player's numbers are cash: a tournament stays at equilibrium, its grades by the mix", async () => {
  const p = await reviewed();
  assert.equal(p.payload().opponents, undefined);
  assert.equal("p" in p.val("S.actions.flop[1].g"), false);
});

test("a decision: where you are, the verdict, the table with your action, where it came from, your range", async () => {
  const p = await reviewed();
  p.run('rewindTo("turn", 1)');
  await flush();
  assert.equal(p.run("document.body.dataset.s"), "decision");
  assert.equal(p.byId("stepOf").textContent, "Decision 3 of 3");
  assert.equal(p.byId("stepHead").innerHTML.replace(/︎/g, ""), `<h2 class="street">Turn · <span class="inl">K<i>♠</i> 7<i class="r">♦</i> 2<i>♣</i> 5<i class="r">♥</i></span></h2><p class="spot">Pot 9.5 bb. BB checks.</p>`);
  assert.deepEqual([p.byId("stepPrev").textContent, p.byId("stepPrev").hidden, p.byId("stepNext").hidden], ["‹ Flop", false, true]);
  const h = p.byId("result").innerHTML;
  assert.match(h, /<div class="verdict"><div class="hd"><span class="g bad">✗<\/span> You checked<\/div><p>The solver bets 125% every time\. 0\.74 bb lost\.<\/p><\/div>/);
  assert.match(h, /<th>With <span class="inl">/);
  assert.ok(h.indexOf("Bet 125%") < h.indexOf("Bet 75%"));   // (most played first, then best)
  assert.match(h, /<tr class="best"><td><div class="an"><i class="sw" style="--k:var\(--agg2\)"><\/i>Bet 125% <small>12 bb<\/small><\/div><\/td><td class="n">100%<\/td><td class="n">14\.08<\/td><\/tr>/);
  assert.match(h, /<\/i>Check <span class="you">you<\/span><\/div><\/td><td class="n">0%<\/td><td class="n">13\.34<\/td>/);
  assert.match(h, /≈ 13\.72/);
  assert.match(h, /<p class="meta">Solved live, 0\.08% of the pot from equilibrium\. ≈ is a rough EV\.<\/p>/);
  assert.match(h, /<h2>Your range: 12 combos vs BB's 172\.5<\/h2>/);   // (half of AKo's 12 and KK's 6: the grid's own count)
  assert.match(h, /<div class="mix"[^>]*><div style="--k:var\(--passive\);flex:\d/);
  assert.match(h, /<span><i class="sw" style="--k:var\(--passive\)"><\/i>Check <b>75\.0%<\/b><\/span>/);
  assert.equal((h.match(/<button class="cell/g) || []).length, 169);
  assert.match(h, /<button class="cell in me" data-h="AKo" aria-label="AKo"><span class="st" style="opacity:0\.650"><i style="--k:var\(--passive\);width:50\.0%"><\/i><i style="--k:var\(--agg2\);width:50\.0%"><\/i><\/span><span class="lbl">AKo<\/span><\/button>/);
  assert.match(h, /<button class="cell" data-h="72o"[^>]*><span class="st" style="opacity:0\.300"><\/span>/);
  // the line under the grid
  p.ctx.__g = JSON.stringify(TURN.grid);
  assert.equal(p.val(`readLine(JSON.parse(__g), "AKo", "AKo", false)`), "<b>AKo</b> (all combos) · 50% in range · Check 50% · Bet 125% 50%");
  assert.equal(p.val(`readLine(JSON.parse(__g), "KK", "AKo", false)`), "<b>KK</b> · 100% in range · Check 100%");
  assert.equal(p.val(`readLine(JSON.parse(__g), "72o", "AKo", false)`), "<b>72o</b> isn't in your range.");
  // your hand outside your range (a preflop chart answers it anyway): drawn from grid.hero, dashed, and said so
  p.ctx.__g = JSON.stringify({ ...TURN.grid, hero: [0.5, 0, 0, 0.5] });
  assert.equal(p.val(`readLine(JSON.parse(__g), "72o", "72o", true)`), "<b>72o</b> isn't in your range for this line, so it's drawn apart (dashed) · Check 50% · Bet 125% 50%");
  assert.equal(p.val(`readLine(JSON.parse(__g), "J8s", "72o", true)`), "<b>J8s</b> isn't in your range.");
  assert.match(p.val(`gridCells(JSON.parse(__g), "72o")`), /<button class="cell out me" data-h="72o" aria-label="72o"><span class="st" style="opacity:0\.650"><i style="--k:var\(--passive\);width:50\.0%"><\/i><i style="--k:var\(--agg2\);width:50\.0%"><\/i><\/span>/);
  assert.match(p.val(`gridCells(JSON.parse(__g), "AKo")`), /<button class="cell" data-h="72o"[^>]*><span class="st" style="opacity:0\.300"><\/span>/);
  // Back to review
  p.byId("stepBack").onclick();
  await flush();
  assert.equal(p.val("stepView"), null);
  assert.equal(p.run("document.body.dataset.s"), "review");
  assert.equal(p.byId("stepTop").hidden, true);
});

test("a right play reads fine, a preflop decision has no EV and shows the chart", async () => {
  const p = await reviewed();
  p.run('rewindTo("flop", 1)');
  await flush();
  let h = p.byId("result").innerHTML;
  assert.match(h, /<span class="g ok">✓<\/span> You bet 33%<\/div><p>In the solver's mix at 63%\. 0\.00 bb lost\.<\/p>/);
  assert.match(h, /<p class="meta">Precomputed, 0\.11% of the pot from equilibrium\.<\/p>/);
  assert.equal(p.byId("stepPrev").textContent, "‹ Preflop");
  assert.equal(p.byId("stepNext").textContent, "Turn ›");
  p.run('rewindTo("preflop", 3)');
  await flush();
  h = p.byId("result").innerHTML;
  assert.equal(p.byId("stepHead").innerHTML, `<h2 class="street">Preflop</h2><p class="spot">UTG, HJ and CO fold. Pot 1.5 bb.</p>`);
  assert.match(h, /You raised to 2\.5<\/div><p>The chart raises this hand 100% of the time\.<\/p>/);
  assert.match(h, /<th>Solver<\/th><\/tr>/);   // (no EV column)
  assert.match(h, /Raise <small>to 2\.5 bb<\/small> <span class="you">you<\/span>/);
  assert.match(h, /<p class="meta">From the preflop chart, not a solve, so there is no EV\.<\/p>/);
  assert.match(h, /<h2>The chart for BTN<\/h2>/);
  assert.doesNotMatch(h, /class="mix"/);
  assert.doesNotMatch(h, /<b>\d+\.\d%<\/b>/);       // (a key without shares)
  assert.doesNotMatch(h, /<details>/);              // (the chart note says nothing the source line doesn't)
  assert.equal(p.byId("stepPrev").hidden, true);
  p.key("Escape");                                  // (its label on Back to review)
  assert.notEqual(p.run("document.body.dataset.s"), "decision");
});

test("the verdict's sentence, and the row's, for every kind of grade", () => {
  const p = loadPage();
  const v = (g) => p.val(`verdictText(${JSON.stringify(g)})`);
  assert.equal(v({ m: "check", f: 0, b: "bet 125%", bf: 1, l: 0.74 }), "The solver bets 125% every time. 0.74 bb lost.");
  assert.equal(v({ m: "check", f: 0, b: "bet 75%", bf: 0.6, l: 0.2 }), "The solver bets 75% 60% of the time. 0.20 bb lost.");
  assert.equal(v({ m: "bet 33%", f: 0.63, b: "bet 33%", bf: 0.63, l: 0 }), "In the solver's mix at 63%. 0.00 bb lost.");
  assert.equal(v({ m: "bet 33%", f: 0.05, b: "check", bf: 0.95, l: 0.1 }), "Rare in the solver's mix, at 5%. 0.10 bb lost.");
  assert.equal(v({ m: "bet 33%", f: 0.2, b: "check", bf: 0.8, l: 0, rv: 1 }), "In the solver's mix at 20%. ≈ 0.00 bb lost.");
  assert.equal(v({ m: "raise", f: 1, b: "raise", bf: 1, l: null }), "The chart raises this hand 100% of the time.");
  assert.equal(v({ m: "call", f: 0.01, b: "fold", bf: 0.99, l: null }), "The chart folds this hand 99% of the time. It calls it 1%.");
  assert.equal(v({ m: "check", f: 0, b: "check", bf: 0, l: 0.5, h: 5 }), "The betting rules pick check. 0.50 bb lost, 5% of the pot.");
  assert.match(v({ m: "check", f: 0, b: "bet 33%", bf: 1, l: 1, r: 1 }), /Rough solve: treat this as a hint, not a verdict\.$/);
  // cut-offs as coded: in the mix from 15%, rare from 3%
  const c = (f) => p.val(`gradeClass({ f: ${f} })`);
  assert.deepEqual([c(0.15), c(0.149), c(0.03), c(0.029)], ["ok", "meh", "meh", "bad"]);
});

test("the source line says plainly where the numbers came from, whatever the answer", () => {
  const p = loadPage();
  const line = (r) => p.val(`sourceOf(${JSON.stringify({ street: "turn", actions: [], ...r })})`);
  assert.deepEqual(line({ solver: { exploitability_pct: 0.13 } }), ["Solved live, 0.13% of the pot from equilibrium.", "solved live"]);
  assert.deepEqual(line({ solver: { exploitability_pct: 0.8 }, solved: true, opponents: ["CO", "BTN"] }), ["Solved live, 3-way, 0.80% of the pot from equilibrium.", "solved live"]);
  assert.deepEqual(line({ precomputed: { exploitability_pct: 0.11 } }), ["Precomputed, 0.11% of the pot from equilibrium.", "precomputed"]);
  assert.deepEqual(line({ precomputed: { exploitability_pct: 0.5 }, solved: true, opponents: ["CO", "BTN"] }), ["Precomputed, 3-way, 0.50% of the pot from equilibrium.", "precomputed"]);
  assert.deepEqual(line({ street: "preflop" }), ["From the preflop chart, not a solve, so there is no EV.", "from the chart"]);
  assert.deepEqual(line({ street: "preflop", actions: [{ action: "fold", ev_bb: 0 }] }), ["From the preflop chart, not a solve.", "from the chart"]);
  assert.deepEqual(line({ multiway: true, equity_vs_field: 0.4 }), ["Betting rules, not a solve: too many players in for one here. 40% equity against the field.", "betting rules"]);
  assert.deepEqual(line({ fallback: "rules" }), ["The solver couldn't answer this spot, so this is the betting rules' answer, not a solve.", "not a solve"]);
  assert.deepEqual(line({ fallback: "equity" }), ["The solver couldn't answer this spot, so this is a rough equity estimate, not a solve.", "not a solve"]);
  assert.deepEqual(line({ fallback: "chart" }), ["The solver couldn't answer this spot, so this is the preflop chart's mix, not a solve.", "not a solve"]);
});

test("the states the mock doesn't draw keep the structure: no grade, no grid, several opponents", async () => {
  const p = await page();
  p.S = { ...state(HAND), finished: true };
  await flush();
  p.run('rewindTo("turn", 1)');
  await flush();
  const m = p.posted.filter(x => x.state).at(-1);
  // the betting rules: no grid, the panel says so; a decision without a grade says what the rules pick
  p.workers.at(-1).onmessage({ data: { id: m.id, result: { street: "turn", hero: "BTN", hand: "AKo", multiway: true, recommended: "check", opponents: ["BB", "SB", "CO"], equity_vs_field: 0.4, notes: [],
    actions: [{ action: "check", freq: 1, ev_bb: 3 }, { action: "bet 33%", amount_bb: 3, freq: 0, ev_bb: 2 }] } } });
  await flush();
  let h = p.byId("result").innerHTML;
  assert.match(h, /<div class="hd">You checked<\/div><p>No grade for this decision\. The betting rules' pick is check\.<\/p>/);
  assert.match(h, /<th>Rules<\/th>/);
  assert.match(h, /Betting rules, not a solve/);
  assert.match(h, /No range grid here: this answer is for your hand alone\./);
  // a 3-way solve: each opponent's range
  p.ctx.__res = JSON.stringify({ street: "turn", hero: "BTN", hand: "AKo", multiway: true, solved: true, opponents: ["CO", "BB"], opponent_range_combos: { CO: 88.25, BB: 120 }, recommended: "check", solver: { exploitability_pct: 0.5 },
    actions: [{ action: "check", freq: 1, ev_bb: 3 }], grid: { actions: ["check"], cells: { AKo: { w: 1, f: [1] } } }, notes: [] });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  h = p.byId("result").innerHTML;
  assert.match(h, /<h2>Your range: 12 combos vs CO's 88\.3 and BB's 120<\/h2>/);
  assert.equal(p.val(`andList(["UTG", "HJ", "BTN"])`), "UTG, HJ and BTN");
  // the same seat named twice by the widened ranges: once
  p.ctx.__res = JSON.stringify({ street: "turn", hero: "BTN", hand: "AKo", recommended: "check", widened: ["BB", "BB"], solver: { exploitability_pct: 0.5 }, actions: [{ action: "check", freq: 1, ev_bb: 3 }] });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.match(p.byId("result").innerHTML, /The solve rarely has BB play this way/);
});

test("a move the chart leaves out gets its 0% row; the spot line and the sizes the review names", () => {
  const p = loadPage();
  p.ctx.__res = JSON.stringify({ street: "preflop", recommended: "fold", actions: [{ action: "fold", freq: 1 }, { action: "raise 2.5", amount_bb: 2.5, freq: 0 }] });
  const h = p.run(`actionTable(JSON.parse(__res), "call")`);
  assert.match(h, /<\/i>Call <span class="you">you<\/span><\/div><\/td><td class="n">0%<\/td>/);
  const line = (a) => p.val(`spotLine(${JSON.stringify(acts(a))})`);
  assert.equal(line("UTG f, HJ f, CO f"), "UTG, HJ and CO fold");
  assert.equal(line("BB x"), "BB checks");
  assert.equal(line("BB b2, BTN r7, BB a"), "BB bets 2, BTN raises to 7, BB goes all-in");
  assert.equal(line("UTG x, HJ x, CO b3, BTN c"), "UTG and HJ check, CO bets 3, BTN calls");
  assert.equal(p.val(`betPct({ action: "bet", to: 2 }, { m: "bet 33%" }, 5.5)`), "33");
  assert.equal(p.val(`betPct({ action: "bet", to: 2.5 }, { m: "bet 33%" }, 5.5)`), null);   // (another size reads as its amount)
  assert.ok(!p.val(`betPct({ action: "check" }, { m: "check" }, 5.5)`));
  assert.equal(p.val("evText(-0.001)"), "0.00");
  assert.equal(p.val("evText(-0.41)"), "−0.41");
});

test("your equity's notes are what bears on it, not how the line was read", () => {
  const p = loadPage();
  const eq = { showdown: true, equity: 0.569, win: null, tie: null, share_bb: 114.04, pot_bb: 200.5, runouts: null, next: "flop", allin: true, street: "preflop",
    villains: [{ seat: "BB", combos: 75.8, equity: 0.569 }], notes: ["flop: mapped 'bet 36%' to tree action 'bet 33%'", "flop: precomputed solution (x)", "against several ranges their hands are taken as independent"] };
  const h = p.val(`equityHtml(${JSON.stringify(eq)})`);
  assert.match(h, /<span>Your equity at the end<\/span><b>56\.9%<\/b>/);
  assert.match(h, /<p class="note">About 114\.04 of the 200\.5 bb pot\. Against BB's range \(75\.8 combos\), from the preflop hand-class equity table\.<\/p>/);
  assert.match(h, /<details><summary>Details<\/summary><p class="meta">against several ranges their hands are taken as independent<\/p><\/details>$/);
  assert.doesNotMatch(h, /<details open|mapped|precomputed/);
  // nothing left to fold away: the one line
  const bare = p.val(`equityHtml(${JSON.stringify({ ...eq, notes: eq.notes.slice(0, 2) })})`);
  assert.doesNotMatch(bare, /<details/);
});

test("the action table: a dash for an action with no EV, so your row's cell isn't blank", () => {
  const p = loadPage();
  const res = { street: "turn", hero: "BB", actions: [{ action: "check", freq: 1, ev_bb: 18 }] };
  const h = p.val(`actionTable(${JSON.stringify(res)}, "all-in")`);
  assert.match(h, /<span class="you">you<\/span><\/div><\/td><td class="n">0%<\/td><td class="n">–<\/td><\/tr>/);
});

test("the count goes with the verdict: rare in the mix is in the mix, a ~ the solver never plays isn't", async () => {
  const p = loadPage();
  await p.ready();
  const mix = (g) => p.val(`inMix(${JSON.stringify(g)})`);
  assert.equal(mix({ f: 0.04, l: 0.1 }), true);       // (~, "Rare in the solver's mix, at 4%")
  assert.equal(mix({ f: 0.5, l: 0 }), true);
  assert.equal(mix({ f: 0, l: 0 }), false);           // (~: no EV given up, but "The solver bets every time")
  assert.equal(mix({ f: 0, l: 1, r: 1 }), false);     // (~: a rough solve)
  assert.equal(mix({ f: 0, l: 0, h: 1 }), true);      // (the betting rules: by EV)
  assert.equal(mix({ f: 0, l: 1, h: 20 }), false);
});

test("the grid's labels have a dark edge in every scheme, and a focused cell a ring selection doesn't take", () => {
  const html = fs.readFileSync(HTML_PATH, "utf8");
  assert.match(html, /\.cell\.in \.lbl \{ color: #fff; text-shadow: 0 0 2px #000, 0 0 2px #000, 0 0 2px #000; \}/);
  const sel = html.indexOf(".cell.sel { outline"), focus = html.indexOf(".grid .cell:focus-visible");
  assert.ok(sel > 0 && focus > sel);
});
