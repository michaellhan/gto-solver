// Study mode (gto/static/study.js, loaded on demand): the hand walked node by node, the requests it sends, the line
// to step back along, typed and random cards, and the HTML of the grid, the actions and a hand's combos.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const S = require(path.join(__dirname, "..", "..", "gto", "static", "study.js"));
const LINE = "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c";
const SPOT = S.fromSpot({ id: "srp_btn_bb", line: LINE });
const cellsOf = (h) => h.match(/class="sy-cell[^"]*"/g);

// a flop node shaped like Advisor.study's: BTN to act after BB checks on Kd7c2h
const NODE = {
  study: "node", street: "flop", actor: "BTN", opponent: "BB", position: "IP", players: 2, board: "Kd7c2h",
  pot_bb: 5.5, to_call_bb: 0, stack_bb: 97.5,
  actions: [{ action: "check", amount_bb: null, freq: 0.5, combos: 200, ev_bb: 3.1 },
            { action: "bet 33%", amount_bb: 2, freq: 0.4, combos: 160, ev_bb: 4.2 },
            { action: "bet 75%", amount_bb: 4, freq: 0.1, combos: 40, ev_bb: 6 }],
  grid: { actions: ["check", "bet 33%", "bet 75%"], cells: {
    AA: { w: 1, f: [0.2, 0.3, 0.5], ev: [10, 11, 12], evm: 11.3, eq: 0.9 },
    KQo: { w: 0.5, f: [0.6, 0.4, 0], ev: [5, 5.5, 4], evm: 5.2, eq: 0.8 },
    "72s": { w: 1, f: [1, 0, 0], ev: [-0.02, -0.5, -1], evm: -0.02, eq: 0.6 } } },
  combos: { AA: [["AsAh", 1, [0, 0, 1], [10, 11, 12.5], 0.91], ["AdAc", 1, [1, 0, 0], [9, 8, 7], 0.88]] },
  range_combos: 400, range_ev_bb: 4, range_eq: 0.55, opponent_combos: 300, opponent_eq: 0.45,
  opponent_grid: { actions: ["in range"], cells: { "72s": { w: 0.25, f: [1], eq: 0.7 }, QJs: { w: 1, f: [1], eq: 0.3 } } },
  solver: { iterations: 400, solve_seconds: 0, exploitability_pct: 0.11 }, precomputed: { label: "BTN vs BB" }, notes: [],
};

test("a spot's preflop line and the page's hand become states", () => {
  assert.deepEqual(S.parseLine(LINE), [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "fold" }, { pos: "CO", action: "fold" },
    { pos: "BTN", action: "raise", to: 2.5 }, { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }]);
  assert.deepEqual(S.parseLine(""), []);
  assert.deepEqual(SPOT, { players: 6, stack: 100, board: "", preflop: S.parseLine(LINE), flop: [], turn: [], river: [] });
  // a spot of another game (the index's "game": flop_library.Format) starts in that game
  assert.deepEqual(S.fromSpot({ id: "srp_btn_bb_40bb_a0.125", line: LINE, game: { players: 6, stack: 40, ante: 0.125, straddle: false } }),
                   { players: 6, stack: 40, ante: 0.125, board: "", preflop: S.parseLine(LINE), flop: [], turn: [], river: [] });
  assert.deepEqual(S.fromSpot({ id: "x", line: LINE, game: { players: 8, stack: 100, ante: 0, straddle: true } }).straddle, true);
  // the table's hand: without hero, hole cards or the advice's settings; streets copied
  const p = { hero: "BTN", hole: "AsKs", players: 6, stack: 100, ante: 0.1, board: "Kd7c2h", preflop: SPOT.preflop,
              flop: [{ pos: "BB", action: "check" }], speed: "fast", locks: { BB: {} }, stacks: null };
  const st = S.fromPayload(p);
  assert.deepEqual(Object.keys(st).sort(), ["ante", "board", "flop", "players", "preflop", "stack"]);
  assert.notEqual(st.flop[0], p.flop[0]);
  assert.deepEqual(S.request(st, "accurate"), { ...st, study: true, speed: "accurate" });
  assert.equal(S.streetOf(st), "flop");
  assert.equal(S.streetOf(S.fresh()), "preflop");
  assert.equal(S.streetOf(S.deal(st, "5s9h")), "river");
  // a tournament's payouts, other tables and bounties stay (the page sends them only when set)
  const t = { ...p, payouts: [50, 30, 20], field: { players: 20, chips: 600 }, bounties: { format: "pko", seats: { BB: 10 } } };
  assert.deepEqual(Object.keys(S.fromPayload(t)).sort(), ["ante", "board", "bounties", "field", "flop", "payouts", "players", "preflop", "stack"]);
  assert.deepEqual(S.request(S.fromPayload(t), "fast").payouts, [50, 30, 20]);
});

test("an action played goes on the street it was answered for", () => {
  assert.deepEqual(["check", "call", "fold", "bet 33%", "raise to 7", "raise", "all-in"].map(S.kindOf),
                   ["check", "call", "fold", "bet", "raise", "raise", "all-in"]);
  const st = S.deal(SPOT, "Kd7c2h");
  const a = S.play(st, { street: "flop", actor: "BB" }, { action: "check", amount_bb: null });
  assert.deepEqual(a.flop, [{ pos: "BB", action: "check" }]);
  assert.deepEqual(SPOT.flop, []);                                   // (states aren't changed in place)
  const b = S.play(a, NODE, NODE.actions[1]);
  assert.deepEqual(b.flop[1], { pos: "BTN", action: "bet", to: 2 });
  const c = S.play(b, { street: "flop", actor: "BB" }, { action: "raise to 7", amount_bb: 7 });
  assert.deepEqual(c.flop[2], { pos: "BB", action: "raise", to: 7 });
  assert.deepEqual(S.play(c, NODE, { action: "all-in", amount_bb: 97.5 }).flop[3], { pos: "BTN", action: "all-in" });
  assert.deepEqual(S.play(SPOT, { street: "preflop", actor: "UTG" }, { action: "raise", amount_bb: 2.5 }).preflop.at(-1),
                   { pos: "UTG", action: "raise", to: 2.5 });
});

test("the line: street headings, actions, preflop folds in a row as one step, and back", () => {
  const st = S.play(S.deal(SPOT, "Kd7c2h"), { street: "flop", actor: "BB" }, { action: "check" });
  const ln = S.line(st);
  assert.deepEqual(ln.map(x => x.kind === "street" ? x.street + ":" + x.cards : x.label),
                   ["preflop:", "UTG–CO fold", "BTN raise 2.5", "SB fold", "BB call", "flop:Kd7c2h", "BB check"]);
  // each step is the state it was taken from
  assert.deepEqual(ln[1].st.preflop, []);
  assert.deepEqual(ln[2].st.preflop.map(a => a.pos), ["UTG", "HJ", "CO"]);
  assert.equal(ln[5].st.board, "Kd7c2h");
  assert.deepEqual(ln[5].st.flop, []);
  assert.deepEqual(ln[0].st, { ...st, board: "", preflop: [], flop: [] });
  // back: the last action's decision, then before the flop was dealt, then preflop's actions
  const b1 = S.back(st);
  assert.deepEqual(b1, { ...st, flop: [] });
  const b2 = S.back(b1);
  assert.deepEqual(b2, { ...st, board: "", flop: [] });
  assert.deepEqual(S.back(b2).preflop.map(a => a.pos), ["UTG", "HJ", "CO", "BTN", "SB"]);
  assert.equal(S.back(S.fresh()), null);
  assert.deepEqual(S.back({ ...S.fresh(), preflop: [{ pos: "UTG", action: "fold" }] }).preflop, []);
  // turn and river headings carry their card; a missing state is an empty line
  const r = S.deal(S.deal(S.deal(SPOT, "Kd7c2h"), "5s"), "9h");
  assert.deepEqual(S.line(r).filter(x => x.kind === "street").map(x => x.cards), ["", "Kd7c2h", "5s", "9h"]);
  assert.equal(S.back(r).board, "Kd7c2h5s");
  assert.deepEqual(S.line(null), []);
});

test("typed cards are read and checked against the board; random ones are off it", () => {
  const st = S.deal(SPOT, "Kd7c2h");
  assert.deepEqual(S.readCards(SPOT, "kd 7C,2h", 3), { cards: "Kd7c2h" });
  assert.deepEqual(S.readCards(st, "10s", 1), { cards: "Ts" });
  assert.match(S.readCards(st, "5s9h", 1).error, /a card/);
  assert.match(S.readCards(SPOT, "Kd7c", 3).error, /3 cards/);
  assert.match(S.readCards(st, "5x", 1).error, /rank .* suit/);
  assert.match(S.readCards(st, "Kd", 1).error, /already out/);
  assert.match(S.readCards(SPOT, "KdKd2h", 3).error, /already out/);
  let i = 0;
  const rnd = () => [0, 0.999, 0.5][i++ % 3];
  const cs = S.randomCards(st, 3, rnd);
  assert.equal(cs.length, 6);
  assert.equal(cs.slice(0, 2), "Ac");
  const got = cs.match(/../g);
  assert.equal(new Set(got).size, 3);
  assert.ok(got.every(c => !["Kd", "7c", "2h"].includes(c)));
  for (let k = 0; k < 50; k++) assert.ok(!["Kd", "7c", "2h"].includes(S.randomCards(st, 1)));
});

test("action labels, EQR and the cell numbers", () => {
  assert.equal(S.label({ action: "check", amount_bb: null }), "Check");
  assert.equal(S.label({ action: "bet 33%", amount_bb: 1.815 }), "Bet 33% · 1.82bb");
  assert.equal(S.label({ action: "raise to 7", amount_bb: 7 }), "Raise to 7bb");
  assert.equal(S.label({ action: "raise", amount_bb: 7.5 }), "Raise to 7.5bb");
  assert.equal(S.label({ action: "call", amount_bb: 2 }), "Call 2bb");
  assert.equal(S.label({ action: "all-in", amount_bb: 97.5 }), "All-in 97.5bb");
  assert.equal(S.eqr({ evm: 2.75, eq: 0.5 }, 5.5), 1);
  assert.equal(S.eqr({ evm: 1, eq: 0 }, 5.5), null);
  assert.equal(S.eqr({ evm: null, eq: 0.5 }, 5.5), null);
  const c = NODE.grid.cells["72s"];
  assert.equal(S.cellValue(c, "ev", NODE, null).txt, "0.0");             // (-0.02 shows as 0, not -0.0)
  assert.equal(S.cellValue(c, "ev", NODE, 2).txt, "-1.0");
  assert.equal(S.cellValue(c, "equity", NODE, null).txt, "60");
  assert.equal(S.cellValue(c, "eqr", NODE, null).txt, (-0.02 / (0.6 * 5.5)).toFixed(2));
  assert.equal(S.cellValue({ w: 1, f: [1] }, "ev", NODE, null), null);
  assert.equal(S.cellValue(c, "strategy", NODE, null), null);
  assert.deepEqual([S.classAt(0, 0), S.classAt(0, 1), S.classAt(1, 0), S.classAt(12, 12)], ["AA", "AKs", "AKo", "22"]);
  assert.deepEqual(["AA", "AKs", "AKo"].map(S.combosOf), [6, 4, 12]);
});

test("the grid: 169 classes, the range's mix stacked, a focused action, numbers, and the opponent's range", () => {
  const ui = { view: "strategy", who: "me", sel: "KQo", focus: null };
  let h = S.gridHtml(NODE, ui);
  const cells = cellsOf(h);
  assert.equal(cells.length, 169);
  assert.equal(cells.filter(c => / out/.test(c)).length, 166);
  assert.match(h, /class="sy-cell sel" data-cls="KQo"/);
  assert.match(h, /data-cls="KQo" style=""><span class="sy-fill" style="height:50.0%">(<i [^>]*><\/i>){2}<\/span>/);
  assert.match(h, /data-cls="AA" style=""><span class="sy-fill" style="height:100.0%">(<i [^>]*><\/i>){3}<\/span>/);
  // an action hovered: only its share of each hand
  h = S.gridHtml(NODE, { ...ui, focus: 2 });
  assert.match(h, /data-cls="AA" style=""><span class="sy-fill" style="height:100.0%"><i style="width:50.0%[^>]*><\/i><\/span>/);
  assert.match(h, /data-cls="KQo" style=""><span class="sy-fill" style="height:50.0%"><\/span>/);
  // EV: shaded with the number (the focused action's EV)
  h = S.gridHtml(NODE, { ...ui, view: "ev" });
  assert.match(h, /class="sy-grid sy-num"/);
  assert.match(h, /data-cls="AA" style="background:hsl\([^"]+"><span class="sy-v">11\.3<\/span>/);
  assert.match(S.gridHtml(NODE, { ...ui, view: "ev", focus: 0 }), /data-cls="AA"[^>]*><span class="sy-v">10\.0<\/span>/);
  assert.match(S.gridHtml(NODE, { ...ui, view: "equity" }), /data-cls="AA"[^>]*><span class="sy-v">90<\/span>/);
  // the opponent's range: its weights (strategy and EV views alike), or its equity
  h = S.gridHtml(NODE, { ...ui, who: "opp", view: "ev", focus: 1 });
  assert.equal(cellsOf(h).filter(c => !/ out/.test(c)).length, 2);
  assert.match(h, /data-cls="72s" style=""><span class="sy-fill" style="height:25.0%"><i style="width:100%;background:var\(--accent\)"><\/i>/);
  assert.match(S.gridHtml(NODE, { ...ui, who: "opp", view: "equity" }), /data-cls="QJs"[^>]*><span class="sy-v">30<\/span>/);
  // a node without an opponent's grid (preflop) shows the actor's whatever "who" says
  assert.equal(cellsOf(S.gridHtml({ ...NODE, opponent_grid: undefined }, { ...ui, who: "opp" })).filter(c => !/ out/.test(c)).length, 3);
});

test("the actions table, the range's totals, a hand's combos and the status lines", () => {
  const ui = { view: "strategy", who: "me", sel: null, focus: 1 };
  let h = S.actionsHtml(NODE, ui);
  assert.equal((h.match(/<tr data-act=/g) || []).length, 3);
  assert.match(h, /<tr data-act="1" tabindex="0" class="focus" title="Play Bet 33% · 2bb">/);
  assert.match(h, /40\.0%<\/td><td class="n">160\.0<\/td><td class="n">\+4\.20<\/td>/);
  assert.match(h, /BTN's range: <b>400<\/b> combos · EV <b>\+4\.00<\/b>bb · equity <b>55\.0%<\/b> · EQR <b>1\.32<\/b>/);
  assert.match(h, /BB: 300 combos · equity 45\.0%/);
  const pre = { ...NODE, street: "preflop", range_ev_bb: null, opponent: null };
  h = S.actionsHtml(pre, ui);
  assert.doesNotMatch(h, />EV</);
  assert.doesNotMatch(h, /equity/);
  // a hand: its mix and EVs, then its combos best first
  h = S.detailHtml(NODE, ui, "AA");
  assert.match(h, /<b>AA<\/b> <span class="sy-mut">6 of 6 combos<\/span> · EV <b>\+11\.30<\/b> · eq <b>90\.0%<\/b>/);
  const rows = h.match(/<tr title="[^"]*"/g);
  assert.equal(rows.length, 2);
  assert.match(rows[0], /check 0% \(EV \+10\.00\), bet 33% 0% \(EV \+11\.00\), bet 75% 100% \(EV \+12\.50\)/);
  assert.match(rows[1], /check 100% \(EV \+9\.00\)/);
  assert.match(S.detailHtml(NODE, ui, "KQo"), /6 of 12 combos/);
  assert.doesNotMatch(S.detailHtml(NODE, ui, "KQo"), /<table/);
  assert.match(S.detailHtml(NODE, ui, "32o"), /not in BTN's range here/);
  assert.match(S.detailHtml(NODE, { ...ui, who: "opp" }, "72s"), /1 of 4 combos<\/span> · equity <b>70\.0%<\/b><\/div>$/);
  assert.match(S.detailHtml(NODE, { ...ui, who: "opp" }, "AA"), /not in BB's range here/);
  assert.match(S.detailHtml(NODE, ui, null), /a hand for its combos/);
  // status, the solver's line, the line's buttons
  assert.equal(S.statusHtml(NODE), "<b>BTN</b> to act (IP) · pot 5.5bb · 97.5bb behind");
  assert.equal(S.statusHtml({ ...NODE, to_call_bb: 2, position: null }), "<b>BTN</b> to act · pot 5.5bb · 2bb to call · 97.5bb behind");
  assert.equal(S.statusHtml({ study: "deal", street: "turn" }), "");
  assert.equal(S.solverHtml(NODE), "Precomputed (BTN vs BB): 400 iterations, 0.11% of the pot from equilibrium.");
  assert.match(S.solverHtml({ ...NODE, precomputed: null, solver: { iterations: 90, solve_seconds: 3.456, exploitability_pct: 0.5 } }),
               /^Solved live in 3\.46 s: 90 iterations, 0\.5% of the pot/);
  assert.doesNotMatch(S.solverHtml({ ...NODE, precomputed: null, solver: { iterations: 90, solve_seconds: 3.456, exploitability_pct: 0.5 } }), /sy-rough/);
  // a live solve cut short is flagged: 3% of the pot "meh", 10% "bad"
  assert.match(S.solverHtml({ ...NODE, precomputed: null, solver: { iterations: 12, solve_seconds: 9, exploitability_pct: 4 } }), /<span class="sy-rough meh">Cut short/);
  assert.match(S.solverHtml({ ...NODE, precomputed: null, solver: { iterations: 12, solve_seconds: 9, exploitability_pct: 24 } }), /sy-rough bad/);
  assert.match(S.solverHtml({ street: "preflop" }), /charts' mix/);
  h = S.lineHtml(S.deal(SPOT, "Kd7c2h"), NODE);
  assert.equal((h.match(/data-step=/g) || []).length, 6);
  assert.match(h, /<button class="sy-a" data-step="1" title="Back to this decision">UTG–CO fold<\/button>/);
  assert.match(h, /<span class="sy-now">BTN \?<\/span>$/);
  assert.doesNotMatch(S.lineHtml(SPOT, { study: "deal" }), /sy-now/);
});

test("the range builder: a blank paint, brushes, the score, and its grid, panel and detail", () => {
  assert.equal(S.passiveOf(["fold", "call", "raise"]), 0);
  assert.equal(S.passiveOf(["check", "bet 33%"]), 0);
  assert.equal(S.passiveOf(["raise", "call"]), 1);
  const p = S.blankPaint(NODE);
  assert.deepEqual(p.f, { AA: [1, 0, 0], KQo: [1, 0, 0], "72s": [1, 0, 0] });
  // a brush sets its action's share, the rest keeping its proportions (all on one action: to the passive one, or
  // spread when the brush is the passive action)
  const near = (a, b) => assert.deepEqual(a.map(x => +x.toFixed(6)), b);
  near(S.brush([1, 0, 0], 2, 1, 0), [0, 0, 1]);
  near(S.brush([1, 0, 0], 1, 0.5, 0), [0.5, 0.5, 0]);
  near(S.brush([0.5, 0.5, 0], 2, 0.5, 0), [0.25, 0.25, 0.5]);
  near(S.brush([0, 0, 1], 2, 0.5, 0), [0.5, 0, 0.5]);
  near(S.brush([1, 0, 0], 0, 0.5, 0), [0.5, 0.25, 0.25]);
  // all checked: AA (6 combos) 80% off, KQo (half its 12) 40%, 72s right: 1 - (6 x 0.8 + 6 x 0.4) / 16
  const sc = S.scoreBuild(NODE, p);
  assert.equal(sc.combos, 16);
  assert.ok(Math.abs(sc.acc - 0.55) < 1e-9);
  assert.ok(Math.abs(sc.ev_loss_bb - 9 / 16) < 1e-9);          // AA 11.3 - 10 = 1.3, KQo 5.2 - 5 = 0.2, a combo
  assert.ok(Math.abs(sc.ev_loss_pct - 100 * 9 / 16 / 5.5) < 1e-9);
  near(sc.mine, [1, 0, 0]);
  near(sc.gto, [0.55, 0.2625, 0.1875]);
  const exact = { f: Object.fromEntries(Object.entries(NODE.grid.cells).map(([k, c]) => [k, [...c.f]])), score: null };
  assert.equal(S.scoreBuild(NODE, exact).acc, 1);
  assert.equal(S.scoreBuild(NODE, exact).ev_loss_bb, 0);
  // preflop: no EVs, accuracy only
  const pre = { ...NODE, street: "preflop", grid: { actions: ["fold", "raise"], cells: { AA: { w: 1, f: [0, 1] }, "72o": { w: 1, f: [1, 0] } } } };
  const ps = S.scoreBuild(pre, S.blankPaint(pre));
  assert.equal(ps.ev_loss_bb, null);
  assert.ok(Math.abs(ps.acc - 12 / 18) < 1e-9);
  assert.equal(S.grade(0.95), "Excellent");
  assert.equal(S.grade(0.55), "Off");
  // the grid: painted mixes; once scored the solver's above yours, outlined by the difference
  const ui = { sel: null, brush: 1, amt: 0.5, tally: { n: 0, acc: 0 } };
  let g = S.buildGridHtml(NODE, p, ui);
  assert.equal(cellsOf(g).length, 169);
  assert.equal(cellsOf(g).filter(c => !/ out/.test(c)).length, 3);
  assert.match(g, /sy-build"/);
  const scored = { ...p, score: sc };
  g = S.buildGridHtml(NODE, scored, ui);
  assert.match(g, /class="sy-cell pair e2" data-cls="AA"/);
  assert.match(g, /class="sy-cell e1" data-cls="KQo"/);
  assert.match(g, /class="sy-cell" data-cls="72s"/);
  assert.equal((g.match(/sy-half top/g) || []).length, 3);
  // the panel: brushes and amounts, then the score, each action both ways and the biggest differences
  let h = S.buildPanelHtml(NODE, p, ui);
  assert.equal((h.match(/data-brush=/g) || []).length, 3);
  assert.match(h, /data-brush="1" class="active"/);
  assert.match(h, /data-amt="0.5" class="active"/);
  assert.match(h, /Unpainted hands check/);
  assert.match(h, /data-score/);
  h = S.buildPanelHtml(NODE, scored, { ...ui, tally: { n: 2, acc: 1.5 } });
  assert.match(h, /<b>55\.0%<\/b> accurate · Off/);
  assert.match(h, /0\.563<\/b>bb a combo \(10\.23% of the pot\)/);
  assert.match(h, /data-cls="AA" class="ghost">AA <small>80% · −1\.30bb/);
  assert.ok(h.indexOf('data-cls="AA"') < h.indexOf('data-cls="KQo"'));
  assert.doesNotMatch(h, /data-cls="72s"/);
  assert.match(h, /data-newflop/);
  assert.match(h, /2 ranges, 75\.0% accurate/);
  assert.doesNotMatch(S.buildPanelHtml({ ...NODE, board: "Kd7c2h5s" }, scored, ui), /data-newflop/);
  // a hand's detail: yours, then the solver's with its EVs
  assert.match(S.buildDetailHtml(NODE, p, "AA"), /Yours<\/div><div class="sy-mix"><span style="--c:[^"]*">check <b>100%/);
  assert.doesNotMatch(S.buildDetailHtml(NODE, p, "AA"), /Solver/);
  assert.match(S.buildDetailHtml(NODE, scored, "AA"), /bet 75% <b>50%<\/b> <small>\+12\.00/);
  assert.match(S.buildDetailHtml(NODE, scored, "QJs"), /not in BTN's range/);
});

test("the filter: one made hand or draw, the grid, the range's totals and the combos over its hands", () => {
  // combo rows tagged with their made hand and draw (indices into categories)
  const node = { ...NODE, categories: { made: [["set", "Set"], ["overpair", "Overpair"], ["top_tk", "Top pair, top kicker"]],
                                        draws: [["fd", "Flush draw"], ["none", "No draw"]] },
    combos: { AA: [["AsAh", 1, [0, 0, 1], [10, 11, 12], 0.9, 1, 1], ["AdAc", 1, [1, 0, 0], [9, 8, 7], 0.88, 1, 0]],
              KQo: [["KsQh", 1, [0.5, 0.5, 0], [5, 6, 4], 0.8, 2, 1], ["KhQd", 0.5, [1, 0, 0], [4, 4, 4], 0.7, 2, 1]],
              "72s": [["7s2s", 1, [0, 0, 1], [3, 2, 8], 0.95, 0, 1]] } };
  assert.deepEqual(S.catsOf(node).map(c => [c.key, c.label, c.combos]),
                   [["m:set", "Set", 1], ["m:overpair", "Overpair", 2], ["m:top_tk", "Top pair, top kicker", 1.5], ["d:fd", "Flush draw", 1]]);
  assert.deepEqual(S.catsOf(NODE), []);                    // (preflop, or an answer without them)
  const near = (a, b) => assert.deepEqual(a.map(x => x == null ? x : +x.toFixed(6)), b);
  const op = S.filterRes(node, "m:overpair");
  assert.deepEqual(Object.keys(op.grid.cells), ["AA"]);
  const aa = op.grid.cells.AA;
  near([aa.w, aa.evm, aa.eq], [+(2 / 6).toFixed(6), 10.5, 0.89]);
  near(aa.f, [0.5, 0, 0.5]);
  near(aa.ev, [9.5, 9.5, 9.5]);
  near(op.actions.map(a => a.freq), [0.5, 0, 0.5]);
  near(op.actions.map(a => a.ev_bb), [9, null, 12]);       // (none for an action no hand of it takes)
  assert.deepEqual(op.actions.map(a => a.action), ["check", "bet 33%", "bet 75%"]);
  near([op.range_combos, op.range_ev_bb, op.range_eq], [2, 10.5, 0.89]);
  assert.deepEqual(op.filter, { key: "m:overpair", label: "Overpair", of: 400 });
  const tp = S.filterRes(node, "m:top_tk").grid.cells.KQo;
  near([tp.w, ...tp.f], [0.125, +(2 / 3).toFixed(6), +(1 / 3).toFixed(6), 0]);
  const fd = S.filterRes(node, "d:fd");
  assert.deepEqual(fd.combos, { AA: [node.combos.AA[1]] });
  assert.equal(S.filterRes(node, "d:oesd"), node);          // not a category here (draws on the river): every hand
  assert.equal(S.filterRes(node, null), node);
  // the select, the grid, the totals and the detail
  const sel = S.filterHtml(node, "m:overpair");
  assert.match(sel, /<option value="">All hands<\/option><optgroup label="Made hands"><option value="m:set">Set · 0\.3%/);
  assert.match(sel, /<option value="m:overpair" selected>Overpair · 0\.5%<\/option>/);
  assert.match(sel, /<optgroup label="Draws"><option value="d:fd">Flush draw · 0\.3%<\/option><\/optgroup>/);
  assert.doesNotMatch(sel, /No draw/);
  assert.equal(S.filterHtml(NODE, null), "");
  assert.equal(cellsOf(S.gridHtml(op, { view: "strategy", who: "me" })).filter(c => / out/.test(c)).length, 168);
  const t = S.actionsHtml(op, {});
  assert.match(t, /BTN's <b>overpair<\/b>: <b>2<\/b> of 400 combos \(0\.5%\) · EV <b>\+10\.50<\/b>bb/);
  assert.match(t, /data-unfilter/);
  assert.doesNotMatch(S.actionsHtml(node, {}), /data-unfilter/);
  const d = S.detailHtml(fd, { who: "me" }, "AA");
  assert.match(d, /1 of 6 combos/);
  assert.equal((d.match(/<tr title=/g) || []).length, 1);
});

test("the runout report: every turn (or river) card by rank and suit, the board's blank, the mix and the bet share, two players", () => {
  // shaped like Advisor.study's {"runouts": true} on Kd7c2h after check, bet, call: two classes stand for the 49 cards
  const off = new Set(["Kd", "7c", "2h"]), all = [];
  for (const r of "AKQJT98765432") for (const s of "shdc") if (!off.has(r + s)) all.push(r + s);
  const sevens = all.filter(c => c[0] === "7"), rest = all.filter(c => c[0] !== "7");
  const part = (actor, f7, fr) => ({ actor, actions: ["check", "bet 75%"], avg: [0.6, 0.4],
    cards: [{ card: "7d", cards: sevens, f: f7 }, { card: "Qh", cards: rest, f: fr }] });
  const RO = { study: "runouts", street: "turn", first: part("BB", [0.9, 0.1], [1, 0]), after_check: part("BTN", [0.8, 0.2], [0.45, 0.55]),
               solver: { iterations: 400, exploitability_pct: 0.05, by: "hand buckets" }, notes: [] };
  const h = S.runoutsHtml(RO, "first", "Kd7c2h");
  assert.equal((h.match(/class="sy-roc[" ]/g) || []).length, 52);
  assert.equal((h.match(/class="sy-roc off"/g) || []).length, 3);
  assert.equal((h.match(/data-card="/g) || []).length, 49);
  assert.match(h, /data-card="7s" title="7s: check 90% · bet 75% 10%/);
  assert.match(h, /data-rowho="first" class="active">BB</);
  assert.match(h, /data-rowho="after" class="">BTN after a check</);
  assert.match(h, /how often BB bets/);
  assert.match(h, /the turn by hand buckets/);
  const a = S.runoutsHtml(RO, "after", "Kd7c2h");
  assert.match(a, /data-card="Ah" title="Ah: check 45% · bet 75% 55%/);
  assert.match(a, /<span class="sy-rov">55%<\/span>/);
  // no check to follow (the first player bet or there's no second decision): one tab
  assert.doesNotMatch(S.runoutsHtml({ ...RO, after_check: null }, "after", "Kd7c2h"), /data-rowho="after"/);
  // the river's, off a turn solve: every card its own (48 of them), the board's four blank
  const rivers = all.filter(c => c !== "5s").map(c => ({ card: c, cards: [c], f: c[0] === "5" ? [0.7, 0.3] : [1, 0] }));
  const RR = { ...RO, street: "river", first: { actor: "BB", actions: ["check", "bet 50%"], avg: [0.95, 0.05], cards: rivers },
               after_check: null, solver: { iterations: 300, exploitability_pct: 0.1, by: "every hand" } };
  const rv = S.runoutsHtml(RR, "first", "Kd7c2h5s");
  assert.equal((rv.match(/class="sy-roc off"/g) || []).length, 4);
  assert.equal((rv.match(/data-card="/g) || []).length, 48);
  assert.match(rv, /data-card="5h" title="5h: check 70% · bet 50% 30%/);
  assert.match(rv, /live solve of the turn/);
  assert.doesNotMatch(rv, /hand buckets/);
  assert.doesNotMatch(rv, /sy-rough/);
  // a solve cut short says so
  const rough = S.runoutsHtml({ ...RR, solver: { iterations: 13, exploitability_pct: 38.58, by: "every hand" } }, "first", "Kd7c2h5s");
  assert.match(rough, /class="sy-rough bad">Only 13 iterations: 38\.58% of the pot from equilibrium/);
  assert.match(S.runoutsHtml({ ...RR, solver: { iterations: 90, exploitability_pct: 4.2 } }, "first", "Kd7c2h5s"), /sy-rough meh/);
});

test("a 3-way node: a toggle per opponent, their ranges and combos, no equity views before the river", () => {
  const MW = { ...NODE, actor: "BB", opponent: "SB and BTN", position: "2 of 3 to act", multiway: true, range_eq: null,
    opponent_grid: undefined, opponent_combos: undefined, opponent_eq: undefined, precomputed: null,
    opponents: [{ seat: "SB", combos: 30, eq: null, grid: { actions: ["in range"], cells: { "72s": { w: 0.25, f: [1], eq: null } } } },
                { seat: "BTN", combos: 500, eq: null, grid: { actions: ["in range"], cells: { AA: { w: 1, f: [1], eq: null } } } }] };
  assert.deepEqual(S.oppsOf(MW).map(o => [o.key, o.seat]), [["opp", "SB"], ["opp1", "BTN"]]);
  assert.deepEqual(S.oppsOf(NODE).map(o => [o.key, o.seat, o.combos]), [["opp", "BB", 300]]);
  assert.deepEqual(S.oppsOf({ ...NODE, street: "preflop", opponent: null }), []);
  let h = S.whoHtml(MW, "opp1");
  assert.deepEqual(h.match(/data-who="[^"]+"/g), ['data-who="me"', 'data-who="opp"', 'data-who="opp1"']);
  assert.match(h, /data-who="opp1" class="active">BTN</);
  // a "who" the node lacks (a heads-up node after a 3-way one) shows the actor's
  assert.match(S.whoHtml(NODE, "opp1"), /data-who="me" class="active">BTN</);
  assert.deepEqual(S.viewsOf(MW), ["strategy", "ev", "breakdown"]);
  assert.deepEqual(S.viewsOf({ ...MW, range_eq: 0.4 }), ["strategy", "ev", "equity", "eqr", "breakdown"]);
  // heads-up where the actor can bet or raise: Simple too (the one-size view, simple.js); facing a bet with no raise not
  assert.deepEqual(S.viewsOf(NODE), ["strategy", "ev", "equity", "eqr", "breakdown", "simple"]);
  assert.deepEqual(S.viewsOf({ ...NODE, actions: [{ action: "fold" }, { action: "call" }, { action: "raise to 20" }] }), ["strategy", "ev", "equity", "eqr", "breakdown", "simple"]);
  assert.deepEqual(S.viewsOf({ ...NODE, actions: [{ action: "fold" }, { action: "call" }] }), ["strategy", "ev", "equity", "eqr", "breakdown"]);
  assert.deepEqual(S.viewsOf({ ...NODE, street: "preflop" }), ["strategy"]);
  // the second opponent's range in the grid and the detail
  const ui = { view: "strategy", who: "opp1", sel: null, focus: null };
  assert.equal(cellsOf(S.gridHtml(MW, ui)).filter(c => !/ out/.test(c)).length, 1);
  assert.match(S.gridHtml(MW, ui), /class="sy-cell pair" data-cls="AA"/);
  assert.match(S.detailHtml(MW, ui, "72s"), /not in BTN's range here/);
  assert.match(S.detailHtml(MW, { ...ui, who: "opp" }, "72s"), /1 of 4 combos<\/span><\/div>$/);
  // the totals: EV without equity, both opponents' combos
  h = S.actionsHtml(MW, { ...ui, who: "me" });
  assert.match(h, /BB's range: <b>400<\/b> combos · EV <b>\+4\.00<\/b>bb<br>SB: 30 combos<br>BTN: 500 combos<\/div>$/);
  assert.doesNotMatch(h, /equity/);
  h = S.actionsHtml({ ...MW, range_eq: 0.4, opponents: MW.opponents.map((o, i) => ({ ...o, eq: [0.25, 0.35][i] })) }, ui);
  assert.match(h, /equity <b>40\.0%<\/b> · EQR <b>1\.82<\/b><br>SB: 30 combos · equity 25\.0%<br>BTN: 500 combos · equity 35\.0%/);
});
