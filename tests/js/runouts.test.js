// Turn and river reports (gto/static/runouts.js, loaded by Study on demand): what each card does to the board, the
// rows joining the report's parts with its cards, filters, sorting, the averages and groups, and the table's HTML.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const R = require(path.join(__dirname, "..", "..", "gto", "static", "runouts.js"));
const S = require(path.join(__dirname, "..", "..", "gto", "static", "study.js"));
globalThis.Runouts = R;   // (Study's report reads it from the page)

const deck = (board) => { const off = new Set(board.match(/../g)), out = []; for (const r of "AKQJT98765432") for (const s of "shdc") if (!off.has(r + s)) out.push(r + s); return out; };

test("card classes: a flush, a straight, pairing the board, overcards, undercards and bricks", () => {
  const c = (b, x) => R.cardClass(b, x);
  // Kd7c2h: rainbow and disconnected
  assert.equal(c("Kd7c2h", "As"), "over");
  assert.equal(c("Kd7c2h", "Kh"), "pair");
  assert.equal(c("Kd7c2h", "2s"), "pair");
  assert.equal(c("Kd7c2h", "Qs"), "under");
  assert.equal(c("Kd7c2h", "5s"), "under");   // (no five ranks hold 5, 7 and 2... 3-7 holds 5 and 7 only)
  // 9c6d6c: a club completes the flush (two on board), cards around 9-6 a straight, the low cards bricks
  assert.equal(c("9c6d6c", "Ac"), "flush");
  assert.equal(c("9c6d6c", "6h"), "pair");
  assert.equal(c("9c6d6c", "8s"), "straight");
  assert.equal(c("9c6d6c", "Ts"), "straight");
  assert.equal(c("9c6d6c", "5h"), "straight");
  assert.equal(c("9c6d6c", "Jh"), "over");
  assert.equal(c("9c6d6c", "3s"), "brick");
  assert.equal(c("9c6d6c", "2d"), "brick");
  // the ace plays low: A on 5-4-x, a wheel's
  assert.equal(c("5d4c9h", "Ah"), "straight");
  assert.equal(c("5d4c9h", "Ks"), "over");
  // the flush before the pair: the turn of a flush suit pairing the board
  assert.equal(c("Kd7d2h", "7d"), "flush");
  // a river report: the turn's card is on the board too (four cards)
  assert.equal(c("Kd7c2h5s", "4h"), "straight");
  assert.equal(c("Kd7c2h5s", "Th"), "under");
  for (const x of deck("Qh8h4c9d")) assert.ok(R.CLASSES[c("Qh8h4c9d", x)], x);
});

// a turn report shaped like Advisor.study's {"runouts": ...} on Kd7c2h: the flop solve's classes (the 7s, the rest),
// a check by BB then BTN's bets; per card its chance, equities and EVs
const all = deck("Kd7c2h"), sevens = all.filter(c => c[0] === "7"), rest = all.filter(c => c[0] !== "7");
const part = (actor, f7, fr) => ({ actor, actions: ["check", "bet 75%"], avg: [0.6, 0.4],
  cards: [{ card: "7d", cards: sevens, f: f7, w: 30 }, { card: "Qh", cards: rest, f: fr, w: 60 }] });
const RO = { study: "runouts", street: "turn", first: part("BB", [0.9, 0.1], [1, 0]), after_check: part("BTN", [0.8, 0.2], [0.45, 0.55]),
             seats: ["BB", "BTN"], pot_bb: 9.5, source: "solve",
             cards: all.map(c => ({ card: c, p: 1 / 49, eq: c[0] === "A" ? [0.4, 0.6] : [0.55, 0.45], ev: c[0] === "A" ? [3.5, 6] : [4.5, 5] })),
             solver: { iterations: 400, exploitability_pct: 0.05, by: "hand buckets" }, notes: [] };

test("rows: each card off the board with its class's mix, its chance, equities and EVs", () => {
  const rs = R.rows(RO, "first", "Kd7c2h");
  assert.equal(rs.length, 49);
  const s7 = rs.find(r => r.card === "7s"), ah = rs.find(r => r.card === "Ah");
  assert.deepEqual([s7.cls, s7.f, s7.w, s7.ev], ["pair", [0.9, 0.1], 30, [4.5, 5]]);
  assert.deepEqual([ah.cls, ah.f, ah.rank, ah.suit, ah.eq], ["over", [1, 0], "A", "h", [0.4, 0.6]]);
  assert.deepEqual(R.rows(RO, "after", "Kd7c2h").find(r => r.card === "7s").f, [0.8, 0.2]);
  assert.deepEqual(R.rows({ ...RO, after_check: null }, "after", "Kd7c2h").find(r => r.card === "7s").f, [0.9, 0.1]);   // (no check to follow)
  // filter, sort (cards high to low, an action's frequency), groups and their averages
  assert.equal(R.filter(rs, { cls: "pair" }).length, 9);
  assert.equal(R.filter(rs, { rank: "A" }).length, 4);
  assert.deepEqual(R.sort(rs, "card", true).slice(0, 5).map(r => r.card), ["As", "Ah", "Ad", "Ac", "Ks"]);
  assert.deepEqual(R.sort(rs, "a1", true).slice(0, 3).map(r => r.card), ["7s", "7h", "7d"]);
  assert.equal(R.sort(rs, "ev0", false)[0].rank, "A");
  const g = R.groups(rs, "cls");
  assert.deepEqual(g.map(x => [x.g, x.n]), [["over", 4], ["under", 36], ["pair", 9]]);
  const pair = g.find(x => x.g === "pair");   // the 7s (w 30) and the K and 2 (w 60): the mix by the range on each
  assert.ok(Math.abs(pair.f[1] - (3 * 30 * 0.1) / (3 * 30 + 6 * 60)) < 1e-9);
  const avg = R.average(rs);
  assert.ok(Math.abs(avg.p - 1) < 1e-9);
  assert.ok(Math.abs(avg.ev[0] - (4 * 3.5 + 45 * 4.5) / 49) < 1e-9);
  assert.ok(Math.abs(avg.eq[1] - (4 * 0.6 + 45 * 0.45) / 49) < 1e-9);
  assert.deepEqual(R.groups(rs, "suit").map(x => x.n), [13, 12, 12, 12]);
  assert.equal(R.groups(rs, "rank")[0].label, "As");
});

test("the table: a summary, groups, every card dealt on a click, the seats' EVs and equities; a stored turn's EVs blank", () => {
  const ui = { who: "first", sort: "card", desc: true, group: "", filter: {} };
  const h = R.html(RO, ui, "Kd7c2h");
  assert.equal((h.match(/<tr data-card="/g) || []).length, 49);
  assert.match(h, /<th data-rosort="ev0"[^>]*>BB EV<\/th>/);
  assert.match(h, /<th data-rosort="eq1"[^>]*>BTN eq<\/th>/);
  assert.match(h, /<th data-rosort="card" class="ro-c" tabindex="0" aria-sort="descending">Turn ▾<\/th>/);
  assert.match(h, /All 49 cards<small>100% of turns<\/small>/);
  assert.match(h, /<tr data-card="Ah"[^>]*>.*?<small>overcard<\/small>.*?<td class="n">100<\/td><td class="n">0<\/td><td class="n">3.50<\/td><td class="n">6.00<\/td><td class="n ro-eq">40.0<\/td><td class="n ro-eq">60.0<\/td><\/tr>/);
  const gh = R.html(RO, { ...ui, group: "cls", filter: { cls: "pair" } }, "Kd7c2h");
  assert.match(gh, /9 cards<small>18% of turns<\/small>/);
  assert.match(gh, /<tr class="ro-grp" data-rogroup="cls:pair"/);
  assert.equal((gh.match(/<tr data-card="/g) || []).length, 9);
  // in position after a check
  assert.match(R.html(RO, { ...ui, who: "after" }, "Kd7c2h"), /<th class="ro-b">BTN after a check<\/th>/);
  // the flop library's stored turns: no EVs, a note saying so, the flop lines they hold
  const ST = { ...RO, source: "store", cards: RO.cards.map(c => ({ ...c, ev: null })), solver: { iterations: 300, exploitability_pct: 0.139, by: "the flop library" },
               lines: [{ flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 7.13 }, { pos: "BB", action: "call" }], share: 0.3, path: [] },
                       { flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "check" }], share: 0.25, path: [] }], line: 1 };
  const sh = R.html(ST, ui, "Kd7c2h");
  assert.match(sh, /<td class="n">—<\/td><td class="n">—<\/td><td class="n ro-eq">40.0<\/td>/);
  assert.match(R.noteHtml(ST), /stored turns hold strategies only, so there are no EVs/);
  assert.match(R.noteHtml(RO), /live solve of the flop .*the turn by hand buckets/);
  assert.match(R.noteHtml({ ...RO, street: "river", solver: { by: "every hand" } }), /live solve of the turn/);
  const c = R.controlsHtml(ST, ui);
  assert.match(c, /<option value="1" selected>Flop: check, check \(25%\)<\/option>/);
  assert.match(c, /<option value="0">Flop: check, bet 7.13, call \(30%\)<\/option>/);
  assert.doesNotMatch(R.controlsHtml(RO, ui), /data-ctl="roline"/);
  // stored turns with the flop solve's EVs (a format-2 turn file): each card's EVs as a live report's, each action's
  // EV for the hands that take it in its bar's tip (none where nobody does), and the note says so
  const evPart = (actor) => ({ ...part(actor, [0.9, 0.1], [1, 0]), cards: [{ card: "7d", cards: sevens, f: [0.9, 0.1], w: 30, ev: [4.12, 6.3] },
                                                                       { card: "Qh", cards: rest, f: [1, 0], w: 60, ev: [4.4, null] }] });
  const SE = { ...ST, cards: RO.cards, first: evPart("BB"), after_check: evPart("BTN") };
  const eh = R.html(SE, ui, "Kd7c2h");
  assert.match(eh, /<tr data-card="Ah"[^>]*>.*?<td class="n">3.50<\/td><td class="n">6.00<\/td><td class="n ro-eq">40.0<\/td>/);
  assert.match(eh, /<tr data-card="7s"[^>]*>.*?title="Check 90.0%, EV 4.12bb"><\/i><i [^>]*title="Bet 75% 10.0%, EV 6.30bb">/);
  assert.match(eh, /<tr data-card="Ah"[^>]*>.*?title="Check 100.0%, EV 4.40bb">/);
  assert.match(eh, /<tr class="ro-sum">.*?title="Check [\d.]+%"><\/i>/);   // (averages: no action EVs)
  assert.deepEqual(R.rows(SE, "first", "Kd7c2h").find(r => r.card === "Qh").fe, [4.4, null]);
  assert.equal(R.rows(RO, "first", "Kd7c2h").find(r => r.card === "Qh").fe, null);
  assert.match(R.noteHtml(SE), /EV in bb: each range's share of the pot \(9.50bb pot\) on the flop solve's strategies, by turn class .*the EV of each action for the hands that take it/);
  assert.doesNotMatch(R.noteHtml(SE), /no EVs/);
});

test("Study's report: the table or the cards, the solve offered when nothing's in hand, its seconds", () => {
  const ui = { view: "table", who: "first", sort: "card", desc: true, group: "", filter: {} };
  const t = S.reportHtml(RO, ui, "Kd7c2h");
  assert.match(t, /data-roview="table" class="active">Table/);
  assert.match(t, /data-rowho="after" class="">BTN after a check/);
  assert.match(t, /class="ro-t"/);
  const c = S.reportHtml(RO, { ...ui, view: "cards" }, "Kd7c2h");
  assert.match(c, /data-roview="cards" class="active">Cards/);
  assert.match(c, /class="sy-rogrid"/);
  assert.doesNotMatch(c, /class="ro-t"/);
  // nothing in hand: a solve of the street, its seconds as measured in the browser (here unthreaded: not isolated)
  const k = S.skippedHtml({ skipped: "not solved yet" }, "turn", "fast");
  assert.match(k, /data-rosolve>Solve the flop \(Fast, ~3 s\)/);
  assert.match(S.skippedHtml({ skipped: "not solved yet" }, "river", "accurate"), /Solve the turn \(Accurate, ~31 s\)/);
  assert.equal(S.estimate("river", "fast"), 7.5);
  // threaded (the site, cross-origin isolated): 8 threads as measured, between by the part threads split
  assert.deepEqual([1, 2, 4, 8, 16].map(n => S.estimate("river", "accurate", n)), [31, 17.6, 10.9, 7.5, 7.5]);
  assert.deepEqual(["turn", "river"].map(st => S.estimate(st, "fast", 8)), [1.5, 3]);
  assert.equal(S.estimate("turn", "accurate", 8), 1.5);
  // a library flop's stored turns that don't hold the line: the lines they do
  const l = S.skippedHtml({ skipped: "not solved yet", lines: [{ flop: [{ pos: "HJ", action: "check" }, { pos: "CO", action: "bet", to: 12.5 }, { pos: "HJ", action: "call" }], share: 0.3047 }] }, "turn", "fast");
  assert.match(l, /don't hold this flop line/);
  assert.match(l, /<option value="0">Flop: check, bet 12.5, call \(30%\)<\/option>/);
});

// a 3-way turn report (Advisor._mw_runouts): the first decision's strategy only
const RO3 = { study: "runouts", street: "turn", multiway: true, first: part("SB", [0.7, 0.3], [0.6, 0.4]), after_check: null,
              seats: ["SB", "BB", "BTN"], pot_bb: 7.5, source: "solve",
              cards: all.map(c => ({ card: c, p: 1 / 49, eq: null, ev: null })),
              solver: { iterations: 270, exploitability_pct: 0.4, by: "hand buckets" }, notes: [] };

test("a 3-way report: the first decision's strategy only (one tab, no EV or equity columns), its note, its solve's seconds", () => {
  assert.equal(R.partOf(RO3, "after").actor, "SB");   // (no part after a check: the first)
  assert.deepEqual(R.rows(RO3, "first", "Kd7c2h").find(r => r.card === "7s").f, [0.7, 0.3]);
  const ui = { who: "first", sort: "card", desc: true, group: "cls", filter: {} };
  const h = R.html(RO3, ui, "Kd7c2h");
  assert.match(h, /<th class="ro-b">SB<\/th>/);
  assert.doesNotMatch(h, /data-rosort="ev0"|data-rosort="eq0"|ro-eq/);
  assert.match(h, /<tr data-card="Ah"[^>]*>.*?<td class="n">60<\/td><td class="n">40<\/td><\/tr>/);
  assert.match(h, /<tr class="ro-grp" data-rogroup="cls:over"[^>]*>.*?<td class="n">60<\/td><td class="n">40<\/td><\/tr>/);
  assert.match(R.noteHtml(RO3), /^Read off a live 3-way solve of the flop \(270 iterations, 0.4% of the pot from equilibrium\), which plays the turn \(7.50bb pot\) by hand buckets at one bet size.*The first decision's strategy only: no EVs or equities 3-way, nor the decisions after it/);
  // Study's: one tab over the table or the cards grid
  const t = S.reportHtml(RO3, { view: "table", ...ui }, "Kd7c2h");
  assert.match(t, /<span class="segment"><button data-rowho="first" class="active">SB<\/button><\/span>/);
  const c = S.reportHtml(RO3, { ...ui, view: "cards" }, "Kd7c2h");
  assert.match(c, /data-card="Ah" title="Ah: check 60% · bet 75% 40%/);
  assert.match(c, /how often SB bets\. Read off a live 3-way solve of the flop .*The first decision's strategy only \(no EVs or equities 3-way, nor the decisions after it\)/);
  // its solve, a 3-way flop's: up to its budget
  assert.match(S.skippedHtml({ skipped: "not solved yet" }, "turn", "fast", undefined, 15), /data-rosolve>Solve the flop \(Fast, up to ~15 s\)/);
  // a river report: the second player's after a check too (two tabs)
  const RV = { ...RO3, street: "river", after_check: part("BB", [0.8, 0.2], [0.75, 0.25]) };
  assert.match(R.noteHtml(RV), /^Read off a live 3-way solve of the turn .*the river \(7.50bb pot\).*Strategies only: no EVs or equities 3-way\. Each card's chance/);
  const v = S.reportHtml(RV, { ...ui, who: "after", view: "cards" }, "Kd7c2h5s");
  assert.match(v, /data-rowho="first" class="">SB<\/button><button data-rowho="after" class="active">BB after a check<\/button><\/span>/);
  assert.match(v, /how often BB bets\. Read off a live 3-way solve of the turn .*Strategies only \(no EVs or equities 3-way\)\. Click/);
});

// the flop reports' Turns: the lines a library flop's stored turns hold (Advisor._store_lines), most reached first
const HJ_CO = [
  { flop: [{ pos: "HJ", action: "check" }, { pos: "CO", action: "bet", to: 12.38 }, { pos: "HJ", action: "call" }], share: 0.3047, path: ["check", "bet 75%", "call"] },
  { flop: [{ pos: "HJ", action: "check" }, { pos: "CO", action: "bet", to: 5.45 }, { pos: "HJ", action: "call" }], share: 0.2413, path: ["check", "bet 33%", "call"] },
  { flop: [{ pos: "HJ", action: "check" }, { pos: "CO", action: "bet", to: 5.45 }, { pos: "HJ", action: "raise", to: 19.5 }, { pos: "CO", action: "call" }], share: 0.0287,
    path: ["check", "bet 33%", "raise to 19.5", "call"] }];

test("a view's line: the most reached held line through its decision, the bet faced first, -1 where none", () => {
  assert.equal(R.pickLine(HJ_CO, "oop"), 0);
  assert.equal(R.pickLine(HJ_CO, "ip"), 0);
  assert.equal(R.pickLine(HJ_CO, "vsbet", "bet 75%"), 0);
  assert.equal(R.pickLine(HJ_CO, "vsbet", "bet 33%"), 1);   // (the view's own bet before a more reached line's)
  assert.equal(R.pickLine(HJ_CO, "vsbet", "bet 50%"), 0);   // (a size no line holds: any bet after a check)
  assert.equal(R.pickLine(HJ_CO, "vsbet"), 0);
  // out of position's bets: its first decision only; checked through: in position's after a check, not facing a bet
  const L = [{ share: 0.4, path: ["bet 33%", "call"] }, { share: 0.2, path: ["check", "check"] }, { share: 0.1, path: ["check", "bet 75%", "raise to 30", "call"] }];
  assert.deepEqual(["oop", "ip", "vsbet"].map(v => R.pickLine(L, v, "bet 33%")), [0, 1, 2]);
  assert.deepEqual(["oop", "ip", "vsbet"].map(v => R.pickLine(L.slice(0, 1), v)), [0, -1, -1]);
  assert.deepEqual(["oop", "ip", "vsbet"].map(v => R.pickLine(L.slice(0, 2), v)), [0, 1, -1]);
  assert.equal(R.pickLine([], "oop"), -1);
  assert.equal(R.pickLine(undefined, "ip"), -1);
  // (unsorted lines: the most reached still)
  assert.equal(R.pickLine([HJ_CO[2], HJ_CO[1], HJ_CO[0]], "ip"), 2);
});

test("the report's header names the line; with no line through the view, the solve comes first", () => {
  const state = { board: "9c6d6c", flop: HJ_CO[0].flop, turn: [], river: [] };
  const h = S.roHeadHtml({ street: "turn" }, state);
  assert.equal(h, `<div class="sy-rohead">The turn report after the flop's <b>HJ check, CO bet 12.38, HJ call</b></div>`);
  assert.match(S.roHeadHtml({ street: "turn" }, state, { label: "CO's decision after a check", share: 0.3047 }),
    /<b>HJ check, CO bet 12.38, HJ call<\/b>: the most reached flop line through CO's decision after a check that the flop library's stored turns hold \(30% of the time\)/);
  assert.match(S.roHeadHtml({ street: "river" }, { ...state, turn: [{ pos: "HJ", action: "check" }, { pos: "CO", action: "check" }] }),
    /The river report after the turn's <b>HJ check, CO check<\/b>/);
  // none through the view: the stored turns hold others (still offered), or the library holds no turns for the flop
  const n = S.skippedHtml({ skipped: "not solved yet", lines: HJ_CO.slice(0, 1) }, "turn", "fast", { label: "HJ facing a bet after a check" });
  assert.match(n, /hold no flop line through HJ facing a bet after a check: solve the flop for this line's turn report/);
  assert.ok(n.indexOf("data-rosolve") < n.indexOf("data-ctl=\"roline\""));
  assert.match(S.skippedHtml({ skipped: "not solved yet" }, "turn", "fast", { label: "HJ's first decision" }),
    /The flop library doesn't hold this flop's turns: solve the flop for the turn report\..*data-rosolve>Solve the flop/);
});

// Study in a page of its own (a fake DOM: the panel's body), opened from the flop reports' Turns (STUDY_AT), against
// a solver that holds the stored turns' lines `lines` on 9c6d6c
async function studyAt(at, lines) {
  const vm = require("node:vm"), fs = require("node:fs");
  const body = { innerHTML: "", querySelector: () => null, insertAdjacentHTML() {} };
  const el = () => ({ dataset: {}, style: {}, hidden: true, innerHTML: "", setAttribute() {}, addEventListener() {}, appendChild: (c) => c,
                      focus() {}, contains: () => true, querySelector: (s) => s === ".sy-body" ? body : s === ".sy-card" ? { focus() {} } : null });
  const asked = [];
  const ctx = vm.createContext({ console, setInterval, clearInterval, setTimeout, clearTimeout,
    document: { getElementById: () => null, createElement: el, head: { appendChild() {} }, body: { appendChild() {} }, activeElement: null },
    localStorage: { getItem: () => null, setItem() {} }, fetch: async () => ({ ok: false }), addEventListener() {},
    studyAsk: async (q) => {
      asked.push(q);
      if (!q.runouts) return { study: "deal", street: "turn" };
      const i = lines.findIndex(l => JSON.stringify(l.flop) === JSON.stringify(q.flop));
      return i < 0 ? { study: "runouts", street: "turn", skipped: "not solved yet", ...(lines.length ? { lines } : {}) }
        : { ...RO, seats: ["HJ", "CO"], source: "store", lines, line: i };
    } });
  ctx.window = ctx;
  for (const f of ["runouts.js", "study.js"]) vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "..", "gto", "static", f), "utf8"), ctx, { filename: f });
  ctx.STUDY_AT = { state: { players: 6, stack: 100, board: "9c6d6c", preflop: [], flop: [{ pos: "HJ", action: "check" }, { pos: "CO", action: "check" }], turn: [], river: [] },
                   spot: "3bp_co_hj", report: true, ...at };
  await ctx.Study.open();
  return { html: body.innerHTML, asked, state: JSON.parse(JSON.stringify(ctx.Study.state)) };   // (out of the page's realm)
}

test("Turns in Study: the most reached held line through the view, named; else the view's own line with the solve first", async () => {
  // HJ never checks back here: checked through isn't held; HJ's first decision lands on the most reached line
  const a = await studyAt({ view: "oop", label: "HJ's first decision" }, HJ_CO);
  assert.deepEqual(a.state.ro.res && a.state.ro.res.line, 0);
  assert.deepEqual(a.state.state.flop, HJ_CO[0].flop);
  assert.match(a.html, /The turn report after the flop's <b>HJ check, CO bet 12.38, HJ call<\/b>: the most reached flop line through HJ's first decision that the flop library's stored turns hold \(30% of the time\)/);
  assert.match(a.html, /class="ro-t"/);
  assert.deepEqual(a.asked.map(q => [q.runouts || "node", q.flop.length]), [["node", 2], ["held", 2], ["node", 3], ["held", 3]]);   // (nothing solved)
  // facing the 33% bet: that bet's line, though the 75% one is reached more
  const b = await studyAt({ view: "vsbet", bet: "bet 33%", label: "HJ facing bet 33% after a check" }, HJ_CO);
  assert.equal(b.state.ro.res.line, 1);
  assert.match(b.html, /<b>HJ check, CO bet 5.45, HJ call<\/b>: the most reached flop line through HJ facing bet 33% after a check/);
  // Study already on the line: no move, the report at once
  const c = await studyAt({ view: "ip", label: "CO's decision after a check", state: { players: 6, stack: 100, board: "9c6d6c", preflop: [], flop: HJ_CO[0].flop, turn: [], river: [] } }, HJ_CO);
  assert.equal(c.state.ro.res.line, 0);
  assert.equal(c.asked.length, 2);
  // no line through the view (CO only bets after HJ's bet here): the line Study opened at, its solve first, the others offered
  const d = await studyAt({ view: "ip", label: "CO's decision after a check" }, [{ flop: [{ pos: "HJ", action: "bet", to: 5.45 }, { pos: "CO", action: "call" }], share: 0.4, path: ["bet 33%", "call"] }]);
  assert.ok(d.state.ro.skipped && d.state.ro.none);
  assert.deepEqual(d.state.state.flop, [{ pos: "HJ", action: "check" }, { pos: "CO", action: "check" }]);
  assert.match(d.html, /hold no flop line through CO's decision after a check: solve the flop for this line's turn report.*data-rosolve>Solve the flop.*or a line they hold/);
  // a flop without stored turns: the solve
  const e = await studyAt({ view: "oop", label: "HJ's first decision" }, []);
  assert.match(e.html, /The flop library doesn't hold this flop's turns: solve the flop for the turn report/);
  assert.doesNotMatch(e.html, /data-ctl="roline"/);
});
