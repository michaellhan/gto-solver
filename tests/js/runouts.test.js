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
  // nothing in hand: a solve of the street, at the speed's budget (a river's: the turn's and the flop's)
  const k = S.skippedHtml({ skipped: "not solved yet" }, "turn", "fast");
  assert.match(k, /data-rosolve>Solve the flop \(Fast, ~8 s\)/);
  assert.match(S.skippedHtml({ skipped: "not solved yet" }, "river", "accurate"), /Solve the turn \(Accurate, ~72 s\)/);
  assert.equal(S.estimate("river", "fast"), 11);
  // a library flop's stored turns that don't hold the line: the lines they do
  const l = S.skippedHtml({ skipped: "not solved yet", lines: [{ flop: [{ pos: "HJ", action: "check" }, { pos: "CO", action: "bet", to: 12.5 }, { pos: "HJ", action: "call" }], share: 0.3047 }] }, "turn", "fast");
  assert.match(l, /don't hold this flop line/);
  assert.match(l, /<option value="0">Flop: check, bet 12.5, call \(30%\)<\/option>/);
});
