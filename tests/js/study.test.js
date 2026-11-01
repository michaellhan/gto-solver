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
  // another table size's pot read off a library spot whose ranges are a few combos away: said so
  assert.equal(S.solverHtml({ ...NODE, precomputed: { ...NODE.precomputed, label: "6-max BTN vs BB", borrowed: { text: "ranges 0.4% apart" } } }),
               "Precomputed (6-max BTN vs BB; approximate, that spot read as this one: ranges 0.4% apart): 400 iterations, 0.11% of the pot from equilibrium.");
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

test("an action the range almost never takes (rough_ev) shows its EVs with a \"≈\" and the tip, nowhere else", () => {
  const R = { ...NODE, actions: NODE.actions.map((a, i) => i === 2 ? { ...a, rough_ev: true } : a) };
  const ui = { view: "strategy", who: "me", sel: null, focus: null };
  const tip = /title="The range almost never takes this, so how the opponent answers it is barely solved/;
  // the range's per action: bet 75%'s EV alone
  let h = S.actionsHtml(R, ui);
  assert.match(h, /<td class="n sy-rough" title="The range almost never[^"]*">≈ \+6\.00<\/td>/);
  assert.equal((h.match(/≈/g) || []).length, 1);
  assert.match(h, /<td class="n">\+4\.20<\/td>/);
  // a class's mix and its combos' tips
  h = S.detailHtml(R, ui, "AA");
  assert.match(h, /bet 75% <b>50%<\/b> <small class="sy-rough" title="The range almost never[^"]*">≈ \+12\.00<\/small>/);
  assert.match(h, /check <b>20%<\/b> <small>\+10\.00<\/small>/);
  const rows = h.match(/<tr title="[^"]*"/g);
  assert.match(rows[0], /bet 33% 0% \(EV \+11\.00\), bet 75% 100% \(EV ≈ \+12\.50\)\. ≈: The range almost never/);
  // the EV view focused on it: "≈" in each cell and the tip on the grid; on another action or the mix's EV, none
  assert.equal(S.cellValue(R.grid.cells.AA, "ev", R, 2).txt, "≈12.0");
  assert.equal(S.cellValue(R.grid.cells.AA, "ev", R, 1).txt, "11.0");
  assert.equal(S.cellValue(R.grid.cells.AA, "ev", R, null).txt, "11.3");
  assert.match(S.gridHtml(R, { ...ui, view: "ev", focus: 2 }), tip);
  assert.doesNotMatch(S.gridHtml(R, { ...ui, view: "ev", focus: 1 }), /≈|title=/);
  assert.doesNotMatch(S.gridHtml(R, { ...ui, view: "strategy", focus: 2 }), /≈/);
  // Build's solver mix once scored
  const p = S.blankPaint(R); p.score = S.scoreBuild(R, p);
  assert.match(S.buildDetailHtml(R, p, "AA"), /<small class="sy-rough"[^>]*>≈ \+12\.00<\/small>/);
  // without the flag nothing changes; an action under a twentieth of a combo has no EV and no "≈"
  assert.doesNotMatch(S.actionsHtml(NODE, ui) + S.detailHtml(NODE, ui, "AA") + S.gridHtml(NODE, { ...ui, view: "ev", focus: 2 }), /≈|sy-rough/);
  const none = { ...R, actions: R.actions.map((a, i) => i === 2 ? { ...a, ev_bb: null } : a) };
  assert.match(S.actionsHtml(none, ui), /<td class="n">—<\/td>/);
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

// a library flop's turn answered at once from its stored turn (Advisor.instant with "study"): CO to act first on
// 9c6d6c-As after HJ checks, bets and calls; strategies only (a format-1 file), the hands' EVs and equities none
const QUICK = {
  ...NODE, street: "turn", actor: "HJ", opponent: "CO", position: "OOP", board: "9c6d6cAs", pot_bb: 41.25, stack_bb: 80.25,
  actions: [{ action: "check", amount_bb: null, freq: 0.9, combos: 18, ev_bb: null },
            { action: "bet 75%", amount_bb: 30.94, freq: 0.1, combos: 2, ev_bb: null }],
  grid: { actions: ["check", "bet 75%"], cells: {
    AA: { w: 0.5, f: [0.5, 0.5], ev: null, evm: null, eq: null }, KQo: { w: 0.25, f: [1, 0], ev: null, evm: null, eq: null } } },
  combos: { AA: [["AhAd", 1, [0, 1], null, null, 3, 0], ["AdAc", 0.5, [1, 0], null, null, 3, 0]] },
  range_combos: 20, range_ev_bb: null, range_eq: null, opponent_eq: null,
  categories: { made: [["none", "Nothing"], ["pair", "Pair"], ["twopair", "Two pair"], ["overpair", "Overpair"]], draws: [["none", "No draw"]] },
  approximate: { source: "store", label: "the precomputed flop solve's view of the turn (by card class, one bet size, 75%)", iterations: 300 },
  solver: { iterations: 300, exploitability_pct: 0.12, by: "the flop library" }, precomputed: undefined,
  notes: ["the stored turn holds strategies only, so no EVs until the turn's own solve"],
};

test("an answer at once: the strategy alone, the mix without EVs, what it's read off", () => {
  assert.deepEqual(S.viewsOf(QUICK), ["strategy"]);
  assert.deepEqual(S.viewsOf({ ...QUICK, approximate: undefined, range_eq: 0.5 }), ["strategy", "ev", "equity", "eqr", "breakdown", "simple"]);
  const ui = { view: "strategy", who: "me", focus: null, sel: null };
  assert.match(S.gridHtml(QUICK, ui), /data-cls="AA" style=""><span class="sy-fill" style="height:50.0%">/);
  const a = S.actionsHtml(QUICK, ui);
  assert.match(a, /Bet 75% · 30.94bb<\/td><td class="n">.*10.0%<\/td><td class="n">2.0<\/td><td class="n">—<\/td>/);
  assert.match(a, /HJ's range: <b>20<\/b> combos<br>CO: 300 combos</);
  assert.doesNotMatch(a, /EV <b>/);
  // a hand's combos: by weight, their mix, no EV or equity columns
  const d = S.detailHtml(QUICK, ui, "AA");
  assert.match(d, /<th>Combo<\/th><th>Mix<\/th><\/tr>/);
  assert.doesNotMatch(d, /EV|Eq|NaN|null/);
  assert.ok(d.indexOf(`title="check 0%, bet 75% 100%"`) > 0);
  assert.ok(d.indexOf(`title="check 0%, bet 75% 100%"`) < d.indexOf(`title="check 100%, bet 75% 0% · 50% of it in the range"`));
  // a format-2 file's EVs: each action's and the range's
  const e = { ...QUICK, actions: QUICK.actions.map((x, i) => ({ ...x, ev_bb: [12.5, 20.1][i] })), range_ev_bb: 13.26 };
  assert.match(S.actionsHtml(e, ui), /<td class="n">\+20.10<\/td>.*EV <b>\+13.26<\/b>bb/s);
  assert.equal(S.solverHtml(QUICK), "Approximate, at once: the precomputed flop solve's view of the turn (by card class, one bet size, 75%) "
    + "(300 iterations, 0.12% of the pot from equilibrium on the flop). The turn's own solve is running and replaces it: then each hand's EVs, equities, the Breakdown and Simple.");
  assert.equal(S.approxHtml(QUICK), S.solverHtml(QUICK));
  // the filter: the category's mix, no EVs
  const f = S.filterRes(QUICK, S.catsOf(QUICK)[0].key);
  assert.equal(f.range_combos, 1.5);
  assert.deepEqual(f.grid.cells.AA.f.map(x => +x.toFixed(3)), [0.333, 0.667]);
  assert.equal(f.grid.cells.AA.ev, null);
  assert.equal(f.range_ev_bb, null);
  assert.deepEqual(f.actions.map(x => x.ev_bb), [null, null]);
  assert.doesNotMatch(S.detailHtml(f, ui, "AA"), /NaN/);
});

// Study in a fake page (as runouts.test.js's), its solver answering `node` and, for a turn, `quick` at once first;
// `hold` keeps each request open until resolved by hand; `globals`: the page's (SPEED: the speed it opens at)
async function studyPage(quick, live, globals = {}) {
  const vm = require("node:vm"), fs = require("node:fs");
  const body = { innerHTML: "", querySelector: (s) => s === ".sy-stage" ? body : null, insertAdjacentHTML(_, h) { this.innerHTML = h + this.innerHTML; } };
  const el = () => ({ dataset: {}, style: {}, hidden: true, innerHTML: "", setAttribute() {}, addEventListener() {}, appendChild: (c) => c,
                      focus() {}, contains: () => true, querySelector: (s) => s === ".sy-body" ? body : s === ".sy-card" ? { focus() {} } : null });
  const asked = [], open = [], stopped = [], prefetched = [];
  const ctx = vm.createContext({ console, setInterval, clearInterval, setTimeout, clearTimeout,
    document: { getElementById: () => null, createElement: el, head: { appendChild() {} }, body: { appendChild() {} }, activeElement: null },
    localStorage: { getItem: () => null, setItem() {} }, fetch: async () => ({ ok: false }), addEventListener() {},
    studyPrefetch: (st) => prefetched.push(st),
    simpleStop: (p) => { stopped.push(p && p.reqId); const o = open.find(x => x.id === (p && p.reqId)); if (o) o.done({ cancelled: true }); return true; },
    studyAsk: (q, onPartial) => {
      const id = asked.push({ q, early: !!onPartial });
      let done;
      const p = new Promise(r => { done = r; });
      p.reqId = id;
      open.push({ id, q, done });
      if (onPartial && quick) setTimeout(() => onPartial(quick(q)), 0);
      return p;
    } });
  ctx.window = ctx;
  Object.assign(ctx, globals);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "..", "gto", "static", "study.js"), "utf8"), ctx, { filename: "study.js" });
  const tick = () => new Promise(r => setTimeout(r, 5));
  const page = { asked, open, stopped, prefetched, tick, html: () => body.innerHTML, ctx,
    // answer request i (1-based) with the live answer
    land: async (i) => { open.find(x => x.id === i).done(live(asked[i - 1].q)); await tick(); } };
  return page;
}

test("Study on a library flop's turn: the stored answer at once, the live one replacing it; playing on it stops the solve", async () => {
  const flop = [{ pos: "HJ", action: "check" }, { pos: "CO", action: "bet", to: 12.38 }, { pos: "HJ", action: "call" }];
  const at = { players: 6, stack: 100, board: "9c6d6cAs", preflop: [], flop, turn: [], river: [] };
  const LIVE = { ...QUICK, approximate: undefined, actions: [{ action: "check", amount_bb: null, freq: 1, combos: 20, ev_bb: 4.8 },
    { action: "bet 33%", amount_bb: 13.61, freq: 0, combos: 0, ev_bb: null }, { action: "bet 75%", amount_bb: 30.94, freq: 0, combos: 0, ev_bb: null }],
    grid: { actions: ["check", "bet 33%", "bet 75%"], cells: { AA: { w: 0.5, f: [1, 0, 0], ev: [5, 4, 3], evm: 5, eq: 0.8 } } }, combos: {},
    range_ev_bb: 4.8, range_eq: 0.5, solver: { iterations: 1000, solve_seconds: 1.1, exploitability_pct: 0.01 }, notes: [] };
  const p = await studyPage((q) => ({ ...QUICK, actor: q.turn.length ? "CO" : "HJ" }), (q) => q.board.length === 8 ? LIVE : { study: "deal", street: "turn" });
  p.ctx.STUDY_AT = { state: at, spot: "3bp_co_hj" };
  const opened = p.ctx.Study.open();
  await p.tick();
  // asked with an answer at once (a turn), shown at once, not dimmed, the solve said to run
  assert.deepEqual(p.asked.map(x => [x.q.turn.length, x.early]), [[0, true]]);
  assert.match(p.html(), /Approximate, at once: the turn's own solve is running/);
  assert.match(p.html(), /<span class="sy-now">HJ \?<\/span>/);
  assert.doesNotMatch(p.html(), /sy-stage busy|sy-wait|data-view="ev"|data-build/);
  assert.match(p.html(), /Approximate, at once: the precomputed flop solve's view of the turn/);
  // the live answer replaces it
  await p.land(1);
  await opened;
  assert.doesNotMatch(p.html(), /Approximate/);
  assert.match(p.html(), /data-view="ev"/);
  assert.match(p.html(), /Solved live in 1.1 s/);
  assert.equal(p.stopped.length, 0);   // (nothing left behind)
});

test("an action played on an answer at once stops its solve; the next node's answer at once comes first", async () => {
  const flop = [{ pos: "HJ", action: "check" }, { pos: "CO", action: "bet", to: 12.38 }, { pos: "HJ", action: "call" }];
  const at = { players: 6, stack: 100, board: "9c6d6cAs", preflop: [], flop, turn: [], river: [] };
  const p = await studyPage((q) => ({ ...QUICK, actor: q.turn.length ? "CO" : "HJ" }), () => ({ study: "over" }));
  p.ctx.STUDY_AT = { state: at, spot: "3bp_co_hj" };
  p.ctx.Study.open();
  await p.tick();
  assert.match(p.html(), /<span class="sy-now">HJ \?<\/span>/);
  // HJ checks on the answer at once (a click on its row): its solve is stopped, CO's node asked, its answer at once shown
  p.ctx.Study.act({ act: "0" });
  await p.tick();
  assert.deepEqual(p.stopped, [1]);
  assert.deepEqual(p.asked.map(x => [x.q.turn.length, x.early]), [[0, true], [1, true]]);
  assert.match(p.html(), /<span class="sy-now">CO \?<\/span>/);
  assert.match(p.html(), /Approximate, at once: the turn's own solve is running/);
  // back to the start of the hand (no turn: asked without one), CO's solve stopped as well
  p.ctx.Study.act({ step: "0" });
  await p.tick();
  assert.equal(p.asked.at(-1).early, false);
  assert.deepEqual(p.stopped, [1, 2]);
});

test("a flop's library files: at Fast fetched once a library flop's answer is in; at Accurate before its first answer, its answer at once asked for", async () => {
  const at = { players: 6, stack: 100, board: "9c6d6c", preflop: [], flop: [], turn: [], river: [] };
  const flopNode = (q) => q.board.length === 6 ? { ...NODE, actor: q.flop.length ? "CO" : "HJ", precomputed: q.board === "9c6d6c" ? { label: "CO 3bet vs HJ" } : undefined } : { study: "over" };
  // Fast: the flop asked without an answer at once, its turn files fetched once its answer (a library flop's) is in
  let p = await studyPage(null, flopNode);
  p.ctx.STUDY_AT = { state: at, spot: "3bp_co_hj" };
  let opened = p.ctx.Study.open();
  await p.tick();
  assert.deepEqual(p.asked.map(x => [x.q.flop.length, x.early]), [[0, false]]);
  assert.equal(p.prefetched.length, 0);
  await p.land(1);
  await opened;
  assert.equal(p.prefetched.length, 1);
  assert.equal(p.prefetched[0].board, "9c6d6c");
  assert.equal(p.prefetched[0].study, true);
  // the next node on the same flop: not fetched again; preflop: none, and no answer at once asked for
  p.ctx.Study.act({ act: "0" });
  await p.tick();
  await p.land(2);
  p.ctx.Study.act({ step: "0" });
  await p.tick();
  assert.deepEqual(p.asked.map(x => [x.q.flop.length, x.early]), [[0, false], [1, false], [0, false]]);
  assert.equal(p.prefetched.length, 1);
  // a live flop at Fast: none
  p = await studyPage(null, flopNode);
  p.ctx.STUDY_AT = { state: { ...at, board: "Kc9d4h" }, spot: "3bp_co_hj" };
  opened = p.ctx.Study.open();
  await p.tick();
  await p.land(1);
  await opened;
  assert.equal(p.prefetched.length, 0);
  // Accurate: fetched before the answer lands (its entry and turn files, or the stored flops nearest it), once a
  // flop, and the flop's requests ask for an answer at once (one read off the stored flops nearest it: none here)
  p = await studyPage(null, flopNode, { SPEED: "accurate" });
  p.ctx.STUDY_AT = { state: at, spot: "3bp_co_hj" };
  opened = p.ctx.Study.open();
  await p.tick();
  assert.equal(p.prefetched.length, 1);
  assert.equal(p.prefetched[0].board, "9c6d6c");
  assert.deepEqual(p.asked.map(x => [x.q.flop.length, x.early]), [[0, true]]);
  await p.land(1);
  await opened;
  p.ctx.Study.act({ act: "0" });
  await p.tick();
  assert.deepEqual(p.asked.map(x => [x.q.flop.length, x.early]), [[0, true], [1, true]]);
  await p.land(2);
  assert.equal(p.prefetched.length, 1);
});

// a flop the library doesn't store, answered at once off the stored flops most like it (Advisor._study_near): the
// mix without EVs, what it borrowed, no solve stats of its own
const NEAR = {
  ...QUICK, street: "flop", board: "Kc9d4h", pot_bb: 22, stack_bb: 89,
  approximate: { source: "near", label: "the precomputed solves of the most similar flops (Kc9d3h, Kc8d4h, Kc9c4d), by card substitution", iterations: null },
  solver: { iterations: null, exploitability_pct: null, by: "the flop library" },
  notes: ["read off stored flops by card substitution: no EVs or equities until the flop's own solve"],
};

test("Study on a flop the library doesn't store: the near flops' answer at once, what it borrowed, the live one replacing it", async () => {
  assert.equal(S.approxHtml(NEAR), "Approximate, at once: the precomputed solves of the most similar flops (Kc9d3h, Kc8d4h, Kc9c4d), by card "
    + "substitution. The flop's own solve is running and replaces it: then each hand's EVs, equities, the Breakdown and Simple.");
  assert.deepEqual(S.viewsOf(NEAR), ["strategy"]);
  const at = { players: 6, stack: 100, board: "Kc9d4h", preflop: [], flop: [], turn: [], river: [] };
  const LIVE = { ...NODE, actor: "HJ", board: "Kc9d4h", precomputed: undefined, solver: { iterations: 400, solve_seconds: 2.4, exploitability_pct: 0.3 } };
  const p = await studyPage((q) => ({ ...NEAR, actor: "HJ" }), () => LIVE, { SPEED: "accurate" });
  p.ctx.STUDY_AT = { state: at, spot: "3bp_co_hj" };
  const opened = p.ctx.Study.open();
  await p.tick();
  assert.deepEqual(p.asked.map(x => x.early), [true]);
  assert.match(p.html(), /Approximate, at once: the flop's own solve is running/);
  assert.match(p.html(), /most similar flops \(Kc9d3h, Kc8d4h, Kc9c4d\), by card substitution\. The flop's own solve/);
  assert.doesNotMatch(p.html(), /iterations, |null|NaN|undefined|EV: bb won/);
  await p.land(1);
  await opened;
  assert.doesNotMatch(p.html(), /Approximate/);
  assert.match(p.html(), /Solved live in 2.4 s/);
});

// a 3-way flop the 3-way library doesn't store, answered at once off its stored flops most like it
// (Advisor._study_mw_near): asked for at Fast too, where a heads-up flop isn't
test("Study on a 3-way flop: an answer at once asked for at Fast too, the note without Simple", async () => {
  const pre = [["UTG", "f"], ["HJ", "f"], ["CO", "f"], ["BTN", "r2.5"], ["SB", "c"], ["BB", "c"]].map(([pos, a]) =>
    a[0] === "r" ? { pos, action: "raise", to: 2.5 } : { pos, action: { f: "fold", c: "call" }[a] });
  assert.equal(S.sawFlop({ players: 6, preflop: pre }), 3);
  assert.equal(S.sawFlop({ players: 6, preflop: [...pre.slice(0, 4), { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }] }), 2);
  const MWN = { ...NEAR, multiway: true, actor: "SB", opponent: "BB and BTN", board: "Kd8c3h",
    approximate: { source: "near", label: "the precomputed 3-way solves of the most similar flops (Kd8c4h, Kd9c3h, Kd7c2h), by card substitution", iterations: null } };
  assert.match(S.approxHtml(MWN), /replaces it: then each hand's EVs, equities and the Breakdown\.$/);
  const LIVE = { ...NODE, actor: "SB", board: "Kd8c3h", multiway: true, precomputed: undefined, solver: { iterations: 200, solve_seconds: 15, exploitability_pct: 0.5 } };
  const p = await studyPage(() => MWN, () => LIVE);
  p.ctx.STUDY_AT = { state: { players: 6, stack: 100, board: "Kd8c3h", preflop: pre, flop: [], turn: [], river: [] }, spot: "" };
  const opened = p.ctx.Study.open();
  await p.tick();
  assert.deepEqual(p.asked.map(x => x.early), [true]);
  assert.match(p.html(), /Approximate, at once: the flop's own solve is running/);
  assert.match(p.html(), /3-way solves of the most similar flops \(Kd8c4h, Kd9c3h, Kd7c2h\)/);
  await p.land(1);
  await opened;
  assert.doesNotMatch(p.html(), /Approximate/);
});

// a later 3-way node answered at once off the hand's earlier solve (Advisor._study_mw_prev): this street's from an
// earlier decision (its stats in the label), or on the turn the flop's (its stats after the label, on the flop)
test("Study's 3-way answer at once off the hand's earlier solve: the note", () => {
  const own = { ...NEAR, multiway: true, street: "turn", actor: "BB", solver: { iterations: null, exploitability_pct: null },
    approximate: { source: "solve", label: "this street's 3-way solve from SB's decision (270 iterations, 0.07% of the pot from equilibrium), on the line since", iterations: null } };
  assert.match(S.approxHtml(own), /^Approximate, at once: this street's 3-way solve from SB's decision \(270 iterations, 0\.07% of the pot from equilibrium\), on the line since\. The turn's own solve is running/);
  const prev = { ...own, actor: "SB", solver: { iterations: 400, exploitability_pct: 0.07 },
    approximate: { source: "prev", label: "the flop solve's view of the turn (by card class and 12 hand buckets a player, one bet size)", iterations: 400 } };
  assert.match(S.approxHtml(prev), /one bet size\) \(400 iterations, 0\.07% of the pot from equilibrium on the flop\)\. The turn's own solve is running and replaces it: then each hand's EVs, equities and the Breakdown\.$/);
});

// a live multiway solve's progress while Study waits on it: its seconds of the speed's budget (a flop's 15 at least),
// a bar; on the answer at once's line too; none heads-up. A 3-way deal's Turn report has strategies only
test("Study on a live 3-way solve: its seconds of up to its budget, a bar, none heads-up; a 3-way Turn report", async () => {
  const pre = [["UTG", "f"], ["HJ", "f"], ["CO", "f"], ["BTN", "r2.5"], ["SB", "c"], ["BB", "c"]].map(([pos, a]) =>
    a[0] === "r" ? { pos, action: "raise", to: 2.5 } : { pos, action: { f: "fold", c: "call" }[a] });
  const at = { players: 6, stack: 100, board: "Kd8c3h", preflop: pre, flop: [], turn: [], river: [] };
  assert.equal(S.inPot(at), 3);
  assert.equal(S.inPot({ ...at, flop: [{ pos: "SB", action: "check" }, { pos: "BB", action: "fold" }] }), 2);
  assert.equal(S.mwSeconds(at, "fast"), 15);
  assert.equal(S.mwSeconds({ ...at, board: "Kd8c3h9s" }, "fast"), 8);
  assert.equal(S.mwSeconds({ ...at, board: "Kd8c3h9s" }, "accurate"), 60);
  assert.equal(S.mwSeconds({ ...at, preflop: [...pre.slice(0, 4), { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }] }, "fast"), null);
  assert.equal(S.mwSeconds({ ...at, board: "" }, "fast"), null);
  const LIVE = { ...NODE, actor: "SB", board: "Kd8c3h", multiway: true, precomputed: undefined, solver: { iterations: 400, solve_seconds: 13, exploitability_pct: 0.05 } };
  // no answer at once: the count beside the spinner, and the bar
  let p = await studyPage(null, () => LIVE);
  p.ctx.STUDY_AT = { state: at, spot: "" };
  let opened = p.ctx.Study.open();
  await p.tick();
  assert.match(p.html(), /Solving the 3-way flop… <span class="num" data-sytime>0\.\d s of up to ~15 s<\/span><\/div><div class="prog"><div data-sybar style="width:\d+\.\d%"><\/div><\/div>/);
  await p.land(1);
  await opened;
  assert.doesNotMatch(p.html(), /data-sytime|data-sybar/);
  // with the answer at once: the count on its line
  const MWN = { ...NEAR, multiway: true, actor: "SB", opponent: "BB and BTN", board: "Kd8c3h",
    approximate: { source: "near", label: "the precomputed 3-way solves of the most similar flops (Kd8c4h, Kd9c3h, Kd7c2h), by card substitution", iterations: null } };
  p = await studyPage(() => MWN, () => LIVE);
  p.ctx.STUDY_AT = { state: at, spot: "" };
  opened = p.ctx.Study.open();
  await p.tick();
  assert.match(p.html(), /Approximate, at once: the flop's own solve is running <span class="num" data-sytime>0\.\d s of up to ~15 s<\/span>/);
  await p.land(1);
  await opened;
  // a 3-way deal: a Turn report of the first decision's strategy; heads-up with EVs and equities
  const closed = [{ pos: "SB", action: "check" }, { pos: "BB", action: "check" }, { pos: "BTN", action: "check" }];
  p = await studyPage(null, () => ({ study: "deal", street: "turn" }));
  p.ctx.STUDY_AT = { state: { ...at, flop: closed }, spot: "" };
  opened = p.ctx.Study.open();
  await p.tick();
  await p.land(1);
  await opened;
  assert.match(p.html(), /Deal the turn/);
  assert.match(p.html(), /data-runouts title="The turn's first decision on every card: strategy \(3-way\)"/);
  p = await studyPage(null, () => ({ study: "deal", street: "turn" }));
  p.ctx.STUDY_AT = { state: { ...at, preflop: [...pre.slice(0, 4), { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }], flop: closed.slice(1) }, spot: "" };
  opened = p.ctx.Study.open();
  await p.tick();
  await p.land(1);
  await opened;
  assert.match(p.html(), /data-runouts title="The turn's first decision on every card: strategy, EV and equity"/);
  // none where four saw the flop, though three are left (the report reads pots 3-way from the flop)
  const pre4 = [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "raise", to: 2.5 }, { pos: "CO", action: "call" }, { pos: "BTN", action: "call" },
                { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }];
  p = await studyPage(null, () => ({ study: "deal", street: "turn" }));
  p.ctx.STUDY_AT = { state: { ...at, preflop: pre4, flop: [{ pos: "BB", action: "check" }, { pos: "HJ", action: "bet", to: 3.5 }, { pos: "CO", action: "call" },
                                                           { pos: "BTN", action: "call" }, { pos: "BB", action: "fold" }] }, spot: "" };
  opened = p.ctx.Study.open();
  await p.tick();
  await p.land(1);
  await opened;
  assert.match(p.html(), /Deal the turn/);
  assert.doesNotMatch(p.html(), /data-runouts/);
  // heads-up: the spinner alone
  p = await studyPage(null, () => LIVE);
  p.ctx.STUDY_AT = { state: { ...at, preflop: [...pre.slice(0, 4), { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }] }, spot: "" };
  opened = p.ctx.Study.open();
  await p.tick();
  assert.match(p.html(), /<span class="spinner"><\/span>Solving…<\/div>/);
  assert.doesNotMatch(p.html(), /data-sytime/);
  await p.land(1);
  await opened;
});

// a river after a turn Study solved live: that solve's view of the river at once (Advisor._study_prev)
const PREV = {
  ...QUICK, street: "river", board: "Ks7d2cTh5s", pot_bb: 12.5, stack_bb: 93.75,
  approximate: { source: "prev", label: "the turn solve's view of the river (by card class, its river sizes)", iterations: 122 },
  solver: { iterations: 122, exploitability_pct: 0.54, by: "the turn's solve" },
  notes: ["read off the turn's solve, which plays the river at its own sizes and a river class's cards alike: no EVs or equities until the river's own solve"],
};

test("Study on a turn or river after the street before was solved live: that solve's answer at once, its stats, the live one replacing it", async () => {
  assert.equal(S.approxHtml(PREV), "Approximate, at once: the turn solve's view of the river (by card class, its river sizes) (122 iterations, "
    + "0.54% of the pot from equilibrium on the turn). The river's own solve is running and replaces it: then each hand's EVs, equities, the Breakdown and Simple.");
  assert.match(S.approxHtml({ ...PREV, street: "turn" }), /from equilibrium on the flop\)\. The turn's own solve/);
  const flop = [{ pos: "HJ", action: "check" }, { pos: "CO", action: "bet", to: 4 }, { pos: "HJ", action: "call" }];
  const turn = [{ pos: "HJ", action: "check" }, { pos: "CO", action: "check" }];
  const at = { players: 6, stack: 100, board: "Ks7d2cTh5s", preflop: [], flop, turn, river: [] };
  const LIVE = { ...NODE, actor: "HJ", street: "river", board: "Ks7d2cTh5s", precomputed: undefined, solver: { iterations: 300, solve_seconds: 0.9, exploitability_pct: 0.2 } };
  const p = await studyPage((q) => ({ ...PREV, actor: "HJ" }), () => LIVE);
  p.ctx.STUDY_AT = { state: at, spot: "srp_hj_co" };
  const opened = p.ctx.Study.open();
  await p.tick();
  // at Fast too: a river asks for its answer at once
  assert.deepEqual(p.asked.map(x => [x.q.board, x.early]), [["Ks7d2cTh5s", true]]);
  assert.match(p.html(), /Approximate, at once: the river's own solve is running/);
  assert.match(p.html(), /the turn solve's view of the river \(by card class, its river sizes\) \(122 iterations, 0\.54% of the pot from equilibrium on the turn\)/);
  assert.doesNotMatch(p.html(), /null|NaN|undefined|EV: bb won|data-view="ev"/);
  await p.land(1);
  await opened;
  assert.doesNotMatch(p.html(), /Approximate/);
  assert.match(p.html(), /Solved live in 0.9 s/);
});

// ---- the Solver source preflop (gto/pf_study.py through Advisor.study's "preflop_solve"): the solve's mix and EVs
const PF_NODE = {
  study: "node", street: "preflop", actor: "BB", players: 6, board: "", pot_bb: 4, to_call_bb: 1.5, stack_bb: 99,
  actions: [{ action: "raise", amount_bb: 11, freq: 0.12, combos: 60, ev_bb: 1.2 }, { action: "call", amount_bb: null, freq: 0.5, combos: 250, ev_bb: 0.6 },
            { action: "fold", amount_bb: null, freq: 0.38, combos: 190, ev_bb: 0 }],
  grid: { actions: ["raise", "call", "fold"], cells: {
    AA: { w: 1, f: [1, 0, 0], ev: [9.1, 6.2, 0], evm: 9.1, eq: null }, "72o": { w: 1, f: [0, 0.1, 0.9], ev: [-1.1, -0.02, 0], evm: -0.002, eq: null } } },
  range_combos: 1326, range_ev_bb: 0.554, range_eq: null, opponent: "BTN",
  opponents: [{ seat: "BTN", combos: 560, grid: { actions: [], cells: { AA: { w: 1 } } } }],
  solver: { by: "the preflop solve", iterations: 1500, mbb: 1.34, exploitability_bb: null },
  pf_solve: { key: "6max_100bb", label: "6 players, 100bb, no rake", iterations: 1500, done: true, stored: true, mbb: 1.34, target_mbb: 2 },
  source: "solver", notes: [], regret_bb: 0.004, rough: false,
};

test("the Solver source preflop: its request, the EV column and view, the solve's line", () => {
  const st = S.fresh ? { ...SPOT, preflop: SPOT.preflop.slice(0, 5) } : null;
  assert.equal(S.request(st, "fast", "solver").preflop_solve, true);
  assert.equal(S.request(st, "fast", "charts").preflop_solve, undefined);
  assert.equal(S.request(S.deal(SPOT, "Kd7c2h"), "fast", "solver").preflop_solve, undefined);   // (after the flop: the solves as ever)
  assert.deepEqual(S.viewsOf(PF_NODE), ["strategy", "ev"]);
  const ui = { view: "strategy", who: "me", sel: null, focus: null };
  let h = S.actionsHtml(PF_NODE, ui);
  assert.match(h, />EV</);
  assert.match(h, /12\.0%<\/td><td class="n">60\.0<\/td><td class="n">\+1\.20<\/td>/);
  assert.match(h, /BB's range: <b>1326<\/b> combos · EV <b>\+0\.55<\/b>bb/);
  assert.match(S.detailHtml(PF_NODE, ui, "AA"), /EV <b>\+9\.10<\/b>/);
  assert.equal(S.solverHtml(PF_NODE), "Preflop solve of this game (6 players, 100bb, no rake): precomputed, 1500 iterations, 1.34 mbb/hand from equilibrium "
    + "(a seat's average). The solve's own equilibrium (its flops valued by the flop library's realization), not the charts: the advice plays the charts, which fit published solves better.");
  const live = { ...PF_NODE, pf_solve: { ...PF_NODE.pf_solve, stored: false, done: false, iterations: 300, mbb: 12.5 } };
  assert.match(S.solverHtml(live), /^Preflop solve of this game \(6 players, 100bb, no rake\): 300 iterations, 12\.5 mbb\/hand .*Not finished: the numbers can still move/);
  assert.match(S.solverHtml({ ...PF_NODE, rough: true, regret_bb: 0.8 }), /A line the solve rarely or never plays: the mix here is barely solved \(the range gives up 0\.8bb a hand/);
  assert.doesNotMatch(S.solverHtml(PF_NODE), /rarely/);
  // a solve that levelled off above its target says so (and one kept in the browser, short of the cap above it)
  const stalled = { ...PF_NODE, pf_solve: { ...PF_NODE.pf_solve, stored: false, done: true, stalled: true, iterations: 1100, mbb: 3.01, target_mbb: 1.5, max_iterations: 2000 } };
  assert.match(S.solverHtml(stalled), /: 1100 iterations, 3\.01 mbb\/hand from equilibrium \(a seat's average\); it stopped there, getting no nearer in 200 iterations \(the model's three-way pots keep some games' solves above the 1\.5 mbb target, though their numbers settle\)\. The solve's own/);
  const kept = { ...stalled.pf_solve, stalled: undefined, cached: true };
  assert.match(S.solverHtml({ ...stalled, pf_solve: kept }), /solved earlier in this browser, 1100 iterations, .*it stopped there/);
  for (const p of [{ ...stalled.pf_solve, stalled: false }, { ...kept, iterations: 2000 }, { ...kept, mbb: 1.2 }, { ...stalled.pf_solve, done: false }, PF_NODE.pf_solve])
    assert.doesNotMatch(S.solverHtml({ ...stalled, pf_solve: p }), /stopped there/, JSON.stringify(p));
  // the charts' answer keeps its mix without EVs
  assert.deepEqual(S.viewsOf({ ...PF_NODE, source: undefined }), ["strategy"]);
  assert.doesNotMatch(S.actionsHtml({ ...PF_NODE, source: undefined, range_ev_bb: null }, ui), />EV</);
});

test("Study's Solver source: a game not solved yet offered, solved in chunks with its progress, the node asked again after each, Stop", async () => {
  const CHARTS = { ...PF_NODE, actor: "UTG", source: undefined, opponents: [], opponent: null, range_ev_bb: null, solver: undefined, pf_solve: undefined,
                   actions: PF_NODE.actions.map(a => ({ ...a, ev_bb: null })) };
  const UNSOLVED = { study: "pf_unsolved", street: "preflop", game: "3 players, 100bb, no rake", key: "3max_100bb", estimate_s: 3 };
  let chunks = 0;
  const live = (q) => q.preflop_solve && q.preflop_solve.run
    ? { study: "pf_progress", street: "preflop", pf_solve: { ...PF_NODE.pf_solve, stored: false, done: false, iterations: 100 * ++chunks, mbb: 20 / chunks, target_mbb: 2 } }
    : q.preflop_solve ? (chunks ? { ...PF_NODE, actor: "UTG", pf_solve: { ...PF_NODE.pf_solve, stored: false, done: false, iterations: 100 * chunks } } : UNSOLVED)
    : CHARTS;
  const p = await studyPage(null, live);
  p.ctx.STUDY_AT = { state: { players: 3, stack: 100, board: "", preflop: [], flop: [], turn: [], river: [] } };
  const opened = p.ctx.Study.open();
  await p.tick();
  await p.land(1);
  await opened;
  assert.equal(p.asked[0].q.preflop_solve, undefined);
  assert.match(p.html(), /<button data-pf="charts" class="active">Charts<\/button><button data-pf="solver" class="">Solver<\/button>/);
  // the Solver source: the game's not solved, so the offer
  p.ctx.Study.act({ pf: "solver" });
  await p.tick();
  assert.equal(p.asked[1].q.preflop_solve, true);
  await p.land(2);
  assert.match(p.html(), /Solve this game's preflop<\/b>: 3 players, 100bb, no rake\./);
  assert.match(p.html(), /About 5 s here/);
  assert.match(p.html(), /<button data-pfsolve>Solve<\/button>/);
  // solving: a chunk, then the node asked again (its answer from the solve so far), then the next chunk
  p.ctx.Study.act({ pfsolve: "" });
  await p.tick();
  assert.equal(p.asked[2].q.preflop_solve.run, 4);
  assert.equal(p.asked[2].q.players, 3);
  assert.match(p.html(), /Solving this game's preflop: <b>0<\/b> iterations/);
  assert.match(p.html(), /data-pfstop/);
  await p.land(3);
  assert.equal(p.asked[3].q.preflop_solve, true);
  await p.land(4);
  assert.equal(p.asked[4].q.preflop_solve.run, 4);
  assert.match(p.html(), /Solving this game's preflop: <b>100<\/b> iterations, 20 mbb\/hand from equilibrium/);
  assert.match(p.html(), /<b>UTG<\/b> to act/);
  assert.match(p.html(), /Not finished: the numbers can still move/);
  // Stop: the chunk running ends it, the node asked once more, the line gone
  p.ctx.Study.act({ pfstop: "" });
  assert.match(p.html(), /stopping…/);
  await p.land(5);
  assert.equal(p.asked.length, 6);
  assert.equal(p.asked[5].q.preflop_solve, true);
  await p.land(6);
  assert.doesNotMatch(p.html(), /Solving this game's preflop|data-pfstop/);
  assert.match(p.html(), /200 iterations/);
  assert.equal(p.asked.length, 6);
  // back to the charts: their answer (in hand) again
  p.ctx.Study.act({ pf: "charts" });
  await p.tick();
  assert.equal(p.asked.length, 6);
  assert.match(p.html(), /<button data-pf="charts" class="active">/);
  assert.match(p.html(), /charts' mix/);
});

test("Study's Solver source on a game the solve doesn't take: why, and the charts still there", async () => {
  const p = await studyPage(null, (q) => q.preflop_solve ? { study: "pf_none", street: "preflop", reason: "8-handed preflop is too big to solve here (up to 7 players)" }
    : { ...PF_NODE, source: undefined, range_ev_bb: null });
  p.ctx.STUDY_AT = { state: { players: 8, stack: 100, board: "", preflop: [], flop: [], turn: [], river: [] } };
  const opened = p.ctx.Study.open();
  await p.tick();
  await p.land(1);
  await opened;
  p.ctx.Study.act({ pf: "solver" });
  await p.tick();
  await p.land(2);
  assert.match(p.html(), /The preflop solve doesn't take this game: 8-handed preflop is too big to solve here \(up to 7 players\)\. Charts answers it\./);
  assert.match(p.html(), /data-pf="charts"/);
  assert.doesNotMatch(p.html(), /data-pfsolve/);
});

test("the Preflop solve's game: the form's fields read and clamped, its state (sizes where not the defaults), the game key, the form", () => {
  assert.deepEqual(S.readGame({}), S.PF_GAME);
  const g = S.readGame({ players: "3", stack: "40", ante: "0.1", straddle: true, rake: "5", cap: "3", open: "2.2", sb_open: "", ip_3bet: "9", blind_3bet: "x", four_x: "2.5" });
  assert.deepEqual(g, { players: 3, stack: 40, ante: 0.1, straddle: true, rake: 5, cap: 3, open: 2.2, sb_open: 3, ip_3bet: 5, blind_3bet: 4.8, four_x: 2.5 });
  assert.equal(S.readGame({ players: 2, straddle: true }).straddle, false);   // (no straddle heads-up)
  assert.equal(S.readGame({ players: 9 }).players, 6);                       // (2-6 in the browser)
  assert.equal(S.readGame({ players: 7 }, 7).players, 7);
  assert.equal(S.readGame({ stack: 5000 }).stack, 250);
  assert.deepEqual(S.gameState(g), { players: 3, stack: 40, board: "", preflop: [], flop: [], turn: [], river: [], ante: 0.1, straddle: true,
                                     rake: 0.05, rake_cap: 3, pf_sizes: { open: 2.2, ip_3bet: 5, four_x: 2.5 } });
  assert.deepEqual(S.gameState(S.PF_GAME), S.fresh());
  assert.equal(S.gameState({ rake: 0, cap: 3 }).rake_cap, undefined);
  // the solver's request carries the sizes preflop; the charts' and later streets' don't
  const st = S.gameState(g);
  assert.deepEqual(S.request(st, "fast", "solver").pf_sizes, { open: 2.2, ip_3bet: 5, four_x: 2.5 });
  assert.equal(S.request(st, "fast", "charts").pf_sizes, undefined);
  assert.equal(S.request(S.deal(st, "Kd7c2h"), "fast", "solver").pf_sizes, undefined);
  // the game key: a line played in it is the same game; another size isn't
  const played = S.play(st, { street: "preflop", actor: "BTN" }, { action: "raise to 2.2bb", amount_bb: 2.2 });
  assert.equal(S.gameKey(played), S.gameKey(st));
  assert.notEqual(S.gameKey(S.gameState({ ...g, open: 2.5 })), S.gameKey(st));
  // the table's game
  assert.deepEqual(S.gameOf({ players: 9, stacks: { UTG: 60, BB: 60 }, ante: 0.2, rake: 0.05, rake_cap: 4 }),
                   { ...S.PF_GAME, stack: 60, ante: 0.2, rake: 5, cap: 4 });
  const h = S.gameHtml(g);
  assert.match(h, /<option value="3" selected>3<\/option>/);
  assert.match(h, /<option value="2">2 \(heads-up\)<\/option>/);
  assert.doesNotMatch(h, /value="7"/);
  assert.match(h, /name="open" value="2.2" min="2" max="5" step="0.1"/);
  assert.match(h, /name="straddle" checked>/);
  assert.equal((h.match(/data-ctl="pfg"/g) || []).length, 11);
  assert.match(S.gameHtml({ players: 2 }), /name="straddle" disabled>/);
});

test("Preflop solve from the menu: the form's game asked of the solve, a change solving another game (the running solve stopped), the solve kept in the browser and handed back", async () => {
  const kept = new Map(), puts = [];
  const cache = { get: async (k) => kept.get(k) || null, put: async (k, v) => { puts.push(k); kept.set(k, v); }, del: async (k) => { kept.delete(k); } };
  const UNSOLVED = (q) => ({ study: "pf_unsolved", street: "preflop", game: `${q.players} players`, key: `${q.players}max_${q.stack}bb`, estimate_s: 3 });
  let chunks = 0, loaded = null;
  const node = (q, extra = {}) => ({ ...PF_NODE, actor: "UTG", pf_solve: { ...PF_NODE.pf_solve, key: `${q.players}max_${q.stack}bb`, stored: false, ...extra } });
  const live = (q) => q.preflop_solve && q.preflop_solve.run ? { study: "pf_progress", street: "preflop", pf_solve: { iterations: 100 * ++chunks, mbb: 1, target_mbb: 0, max_iterations: 200, done: chunks >= 2 } }
    : q.preflop_solve && q.preflop_solve.export ? { study: "pf_export", street: "preflop", key: `${q.players}max_${q.stack}bb`, iterations: 200, mbb: 0.9, blob: "QUJD" }
    : q.preflop_solve && q.preflop_solve.load ? (loaded = q.preflop_solve, node(q, { cached: true, done: true, iterations: q.preflop_solve.iterations }))
    : q.preflop_solve ? (chunks >= 2 && q.players === 4 ? node(q, { done: true, iterations: 200 }) : UNSOLVED(q))
    : { ...PF_NODE, source: undefined, range_ev_bb: null };
  const p = await studyPage(null, live, { studyPfCache: cache, WORKER_VERSION: "w1",
    studyHand: () => ({ players: 4, stack: 100, hero: "BB", rake: 0, board: "" }) });
  const opened = p.ctx.Study.open({ view: "preflop" });
  await p.tick();
  assert.equal(p.asked[0].q.preflop_solve, true);
  assert.equal(p.asked[0].q.players, 4);   // (the table's game, the first time)
  await p.land(1);
  await opened;
  assert.match(p.html(), /<option value="pfgame" selected>Preflop solve: a game you set<\/option>/);
  assert.match(p.html(), /class="sy-pfg"/);
  assert.match(p.html(), /Solve this game's preflop<\/b>: 4 players\./);
  // solved: two chunks (the node asked after the first), then the node again and the solve kept
  p.ctx.Study.act({ pfsolve: "" });
  await p.tick();
  await p.land(2); await p.land(3); await p.land(4);
  assert.equal(p.asked[4].q.preflop_solve, true);
  await p.land(5);
  assert.equal(JSON.stringify(p.asked[5].q.preflop_solve), '{"export":true}');
  await p.land(6);
  assert.deepEqual(puts, ["w1/4max_100bb"]);
  assert.equal(kept.get("w1/4max_100bb").blob, "QUJD");
  assert.equal(kept.get("w1/4max_100bb").it, 200);
  // another game from the form: its first decision (an offer); a solve started there stops when the form changes again
  p.ctx.Study.setGame({ players: 3, stack: 100 });
  await p.tick();
  assert.equal(p.asked[6].q.players, 3);
  await p.land(7);
  assert.match(p.html(), /Solve this game's preflop<\/b>: 3 players\./);
  p.ctx.Study.act({ pfsolve: "" });
  await p.tick();
  assert.equal(JSON.stringify(p.asked[7].q.preflop_solve), '{"run":4}');
  p.ctx.Study.setGame({ players: 4, stack: 100 });   // (back to the solved game: its node in hand, the 3-handed solve stopped)
  await p.tick();
  assert.equal(p.ctx.Study.state.pfRun, null);
  assert.equal(p.asked.length, 8);
  assert.match(p.html(), /<b>UTG<\/b> to act/);
  await p.land(8);   // (the stopped chunk of the 3-handed game ends: nothing more asked for it)
  assert.equal(p.asked.length, 8);
  // the same game again: nothing asked
  p.ctx.Study.setGame({ players: "4", stack: "100" });
  await p.tick();
  assert.equal(p.asked.length, 8);
  p.ctx.Study.close();
  // the page opened again (a new worker, holding no solve): the game's solve from the browser's cache, handed back
  chunks = 0;
  const q = await studyPage(null, live, { studyPfCache: cache, WORKER_VERSION: "w1", studyHand: () => ({ players: 4, stack: 100 }) });
  const o2 = q.ctx.Study.open({ view: "preflop" });
  await q.tick();
  await q.land(1);
  assert.equal(JSON.stringify(q.asked[1].q.preflop_solve), '{"load":"QUJD","iterations":200}');
  assert.equal(q.asked[1].q.players, 4);
  await q.land(2);
  await o2;
  assert.equal(JSON.stringify(loaded), '{"load":"QUJD","iterations":200}');
  assert.match(q.html(), /solved earlier in this browser, 200 iterations/);
  assert.match(q.html(), /<b>UTG<\/b> to act/);
  // another worker version: not this solver's, so not handed back
  const r = await studyPage(null, live, { studyPfCache: cache, WORKER_VERSION: "w2", studyHand: () => ({ players: 4, stack: 100 }) });
  const o3 = r.ctx.Study.open({ view: "preflop" });
  await r.tick();
  await r.land(1);
  await o3;
  assert.equal(r.asked.length, 1);
  assert.match(r.html(), /Solve this game's preflop/);
  // a copy the worker refuses (another tree's) is dropped from the cache
  kept.set("w2/4max_100bb", { blob: "bad", it: 5 });
  const u = await studyPage(null, (x) => x.preflop_solve ? UNSOLVED(x) : live(x), { studyPfCache: cache, WORKER_VERSION: "w2", studyHand: () => ({ players: 4, stack: 100 }) });
  const o4 = u.ctx.Study.open({ view: "preflop" });
  await u.tick();
  await u.land(1);
  await u.land(2);
  await o4;
  assert.equal(JSON.stringify(u.asked[1].q.preflop_solve), '{"load":"bad","iterations":5}');
  assert.equal(kept.has("w2/4max_100bb"), false);
  assert.match(u.html(), /Solve this game's preflop/);
});

test("the Solver source's buttons are among the panel's clickable data attributes", () => {
  const src = require("node:fs").readFileSync(path.join(__dirname, "..", "..", "gto", "static", "study.js"), "utf8");
  const sel = /const data = \(t\) => \{ const e = t\.closest\("([^"]+)"\)/.exec(src)[1].split(",");
  for (const a of ["[data-pf]", "[data-pfsolve]", "[data-pfstop]"]) assert.ok(sel.includes(a), a);
});

test("the size menus: boxes read, the sizes a menu asks for, sent after the flop only", () => {
  assert.deepEqual(S.readSizes("33 75/125%"), [33, 75, 125]);
  assert.deepEqual(S.readSizes("75, 33, 33"), [33, 75]);
  assert.deepEqual(S.readSizes("  "), []);
  assert.match(S.readSizes("3").error, /from 5 to 500/);
  assert.match(S.readSizes("x").error, /from 5 to 500/);
  assert.match(S.readSizes("10 20 30 40 50 60").error, /at most 5/);
  // a blank box, or the standard sizes typed, is that street's standard; none at all: the standard trees (null)
  assert.deepEqual(S.customSizes({ flop: "25 50 100", turn: "33 75 125", river: "", raise: "" }), { flop: [25, 50, 100] });
  assert.equal(S.customSizes({ flop: "33 75", raise: "50" }), null);
  assert.match(S.customSizes({ turn: "1000" }).error, /^Turn:/);
  assert.equal(S.sizesOf("standard", { flop: [25] }), null);
  assert.deepEqual(S.sizesOf("more"), { flop: [25, 33, 50, 75, 125] });
  assert.deepEqual(S.sizesOf("custom", { turn: [50] }), { turn: [50] });
  const flop = S.deal(SPOT, "Kd7c2h");
  assert.deepEqual(S.request(flop, "fast", "charts", S.sizesOf("more")).sizes, { flop: [25, 33, 50, 75, 125] });
  assert.equal(S.request(flop, "fast", "charts", null).sizes, undefined);
  assert.equal(S.request(SPOT, "fast", "solver", S.sizesOf("more")).sizes, undefined);   // (preflop: the charts' or the solve's own)
  // the note: what each street bets and what the menu costs
  assert.match(S.sizeNote("standard", null, "fast"), /^flop 33% \/ 75% · turn 33% \/ 75% \/ 125% .*The flop library's flops answer at once\.$/);
  assert.match(S.sizeNote("more", S.sizesOf("more"), "fast"), /^flop 25% \/ 33% \/ 50% \/ 75% \/ 125% .*every flop is solved live here: about 1-4 s at Fast/);
  assert.match(S.sizeNote("more", S.sizesOf("more"), "accurate"), /about 5-13 s at Accurate/);
  assert.match(S.sizeNote("custom", { flop: [50] }, "fast"), /solved live here: up to 8 s at Fast/);
  assert.match(S.sizeNote("custom", { turn: [50, 100] }, "fast"), /turn 50% \/ 100% .*flops still answer at once; the turn solves with these sizes/);
  // the controls: the select, the custom boxes (the standard as placeholders), an error
  const h = S.sizesHtml("custom", { flop: "25 50" }, "Flop: at most 5 sizes", "fast");
  assert.match(h, /<select id="sySizes" data-ctl="sizes"[^>]*>.*<option value="custom" selected>Custom…<\/option>/);
  assert.match(h, /<input name="flop" value="25 50" placeholder="33 75"/);
  assert.match(h, /<input name="raise" value="" placeholder="50"/);
  assert.match(h, /<div class="sy-err">Flop: at most 5 sizes<\/div>/);
  assert.doesNotMatch(S.sizesHtml("more", {}, null, "fast"), /data-sizes|sy-err/);
  // the note is on the sizes applied, not the boxes' text typed since
  assert.match(S.sizesHtml("custom", { flop: "60" }, null, "fast", null), /sy-sznote[^>]*>flop 33% \/ 75% /);
  assert.match(S.sizesHtml("custom", { flop: "60" }, null, "fast", { flop: [60] }), /sy-sznote[^>]*>flop 60% · /);
});

test("Study's size menu: the node asked again with the menu's sizes, a custom error shown without asking", async () => {
  const p = await studyPage(null, (q) => ({ ...NODE, actions: q.sizes ? [...NODE.actions, { action: "bet 125%", amount_bb: 6.9, freq: 0, combos: 0, ev_bb: null }] : NODE.actions }));
  p.ctx.STUDY_AT = { state: S.deal(SPOT, "Kd7c2h"), spot: "srp_btn_bb" };
  const opened = p.ctx.Study.open();
  await p.tick();
  await p.land(1);
  await opened;
  assert.equal(p.asked[0].q.sizes, undefined);
  assert.match(p.html(), /<option value="standard" selected>Standard<\/option>/);
  await Promise.all([p.ctx.Study.setSizes("more"), p.tick().then(() => p.land(2))]);
  assert.deepEqual(JSON.parse(JSON.stringify(p.asked[1].q.sizes)), { flop: [25, 33, 50, 75, 125] });
  assert.match(p.html(), /every flop is solved live here/);
  // Custom with nothing applied: the standard trees (the first answer, cached), its boxes shown
  await p.ctx.Study.setSizes("custom");
  await p.tick();
  assert.equal(p.asked.length, 2);
  assert.match(p.html(), /<form data-sizes/);
  // a bad box: said, not asked; a good one asked
  await p.ctx.Study.applyText({ flop: "2", turn: "", river: "", raise: "" });
  assert.equal(p.asked.length, 2);
  assert.match(p.html(), /Flop: percentages of the pot from 5 to 500/);
  await Promise.all([p.ctx.Study.applyText({ flop: "25 50 100", turn: "", river: "", raise: "" }), p.tick().then(() => p.land(3))]);
  assert.deepEqual(JSON.parse(JSON.stringify(p.asked[2].q.sizes)), { flop: [25, 50, 100] });
  assert.doesNotMatch(p.html(), /sy-err/);
  assert.equal(p.ctx.Study.state.sizes, "custom");
});
