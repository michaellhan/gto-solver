// Hand-history import (gto/static/hh.js, loaded on demand) and the whole-hand review it leads to.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage, acts, pageScript } = require("./page.js");

const HH_JS = path.join(__dirname, "..", "..", "gto", "static", "hh.js");
const { parseHandHistory, HH_SEATS } = require(HH_JS);
const HH = (f) => fs.readFileSync(path.join(__dirname, "..", "data", "hh", f + ".txt"), "utf8");
const j = (x) => JSON.parse(JSON.stringify(x));
const flush = () => new Promise(r => setImmediate(r));
const flushAll = async () => { for (let i = 0; i < 5; i++) await flush(); };

// a page with hh.js loaded (as the browser does once the dialog opens) and a hand imported through the dialog
async function imported(text, opts) {
  const p = loadPage(opts);
  vm.runInContext(fs.readFileSync(HH_JS, "utf8"), p.ctx, { filename: "hh.js" });
  await p.ready();
  p.click("importHand");
  p.byId("hhText").value = text;
  await p.click("hhGo");
  return p;
}

// What each history should become: the page state's setup and actions, and where the page ends up.
const CASES = {
  stars_6max_srp: { players: 6, hero: "BTN", stack: 100, cards: ["Ah", "Qd", "Qs", "8d", "3c", "7h", "2s"],
    preflop: "UTG f, HJ f, CO f, BTN r2.5, SB f, BB c", flop: "BB x, BTN b1.8, BB c", turn: "BB x, BTN b6.1, BB c", river: "BB x, BTN b14.5, BB f", pot: 2.5 + 2.5 + 0.5 + 3.6 + 12.2 + 14.5 },
  // a sitting-out seat isn't dealt in: five-handed; hero (120bb) is capped at the 100bb of the player who stayed in
  stars_3bet_pot: { players: 5, hero: "BB", stack: 100, cards: ["Ac", "Kh", "Kd", "9c", "4h", "2s", "7c"],
    preflop: "HJ f, CO r2.5, BTN f, SB f, BB r11, CO c", flop: "BB b7, CO r22, BB c", turn: "BB x, CO a, BB c", river: "", pot: 200.5 },
  gg_limped_multiway: { players: 6, hero: "BB", stack: 100, cards: ["7s", "6s", "Ts", "8h", "2s", "Kd", null],
    preflop: "UTG c, HJ f, CO c, BTN f, SB c, BB x", flop: "SB x, BB x, UTG b2, CO f, SB c, BB r10.5, UTG f, SB c", turn: "SB x, BB b16, SB f", river: "", pot: 4 + 2 + 21 + 16 },
  gg_allin_preflop: { players: 6, hero: "SB", stack: 100, cards: ["Ks", "Kh", "Jd", "5c", "2h", "9s", "Kc"],
    preflop: "UTG f, HJ r2.5, CO f, BTN r7.5, SB r18, BB f, HJ f, BTN a, SB c", flop: "", turn: "", river: "", pot: 201 + 2.5 },
  stars_9max_mtt_ante: { players: 9, hero: "CO", stack: 80, ante: 0.1, cards: ["Ah", "Td", "Ac", "9h", "9d", "5s", "Qh"],
    preflop: "UTG f, UTG1 f, UTG2 f, LJ r2.2, HJ f, CO c, BTN f, SB f, BB c", flop: "BB x, LJ b2, CO c, BB f", turn: "LJ x, CO b5, LJ c", river: "LJ x, CO x", pot: 22 },
  stars_heads_up: { players: 2, hero: "BB", stack: 100, cards: ["9c", "9d", "Ts", "8s", "2c", "4h", "3d"],
    preflop: "SB r2.5, BB r9, SB c", flop: "BB x, SB b6, BB c", turn: "BB x, SB x", river: "BB b20, SB f", pot: 50 },
  gg_straddle: { players: 6, hero: "BB", stack: 100, straddle: true, cards: ["Jc", "Jd", "8c", "4d", "2s", "Ah", null],
    preflop: "HJ f, CO r7, BTN f, SB f, BB r26, UTG f, CO c", flop: "BB b17.5, CO c", turn: "BB x, CO b39, BB f", river: "", pot: 26 + 26 + 0.5 + 2 + 35 + 39 },
};

for (const [name, c] of Object.entries(CASES)) {
  test(`imports ${name}`, async () => {
    const p = await imported(HH(name));
    const S = j(p.S);
    assert.equal(p.byId("hhErr").textContent, "");
    assert.equal(p.byId("importBox").hidden, true);
    assert.deepEqual([S.players, S.hero, S.stack, S.ante, !!S.straddle], [c.players, c.hero, c.stack, c.ante, !!c.straddle]);
    assert.deepEqual(S.cards, c.cards);
    for (const s of ["preflop", "flop", "turn", "river"]) assert.deepEqual(S.actions[s], j(acts(c[s])), s);
    const sim = p.sim();
    assert.equal(sim.handOver, true);
    assert.ok(Math.abs(sim.pot - c.pot) < 1e-9, `pot ${sim.pot}`);
    // the setup inputs follow the imported hand, it's saved, and the hand review offers to grade it
    assert.equal(p.byId("stack").value, c.stack);
    assert.deepEqual(JSON.parse(p.storage.get("handreview")), S);
    assert.match(p.byId("adviceNote").innerHTML, /id="reviewAll"/);
    assert.match(p.byId("toast").innerHTML, new RegExp(`you are ${c.players === 2 && c.hero === "SB" ? "BTN/SB" : c.hero} with ${c.cards[0]}${c.cards[1]}`));
  });
}

test("parsed details: sites, seats, names and amounts in big blinds", () => {
  const h = parseHandHistory(HH("stars_3bet_pot"));
  assert.equal(h.site, "PokerStars"); assert.equal(h.bb, 2); assert.equal(h.heroName, "HeroName");
  assert.deepEqual(h.seats.map(s => [s.seat, s.pos, s.stack]), [[1, "CO", 100], [2, "BTN", 156.2], [3, "SB", 94], [4, "BB", 120], [6, "HJ", 100]]);
  assert.deepEqual(h.actions.find(a => a.act === "bet" && a.allIn), { street: "turn", pos: "CO", act: "bet", to: 67, allIn: true });
  assert.equal(parseHandHistory(HH("gg_limped_multiway")).site, "GGPoker");
  const t = parseHandHistory(HH("stars_9max_mtt_ante"));
  assert.equal(t.bb, 300); assert.equal(t.ante, 0.1); assert.equal(t.seats.find(s => s.seat === 3).pos, "HJ");   // sitting out but dealt in (posted the ante)
  // Windows line endings, a trailing hand and a "10" card are fine
  const two = parseHandHistory(HH("stars_heads_up").replace(/\n/g, "\r\n") + "\r\n\r\n" + HH("stars_6max_srp"));
  assert.equal(two.players, 2);
  assert.deepEqual(two.warnings, ["2 hands pasted: imported the first."]);
  assert.deepEqual(parseHandHistory(HH("stars_heads_up").replace("[9c 9d]", "[10c 9d]")).hole, ["Tc", "9d"]);
});

test("the page's seat names are the importer's", () => {
  assert.deepEqual(j(loadPage().val("SEATS")), j(HH_SEATS));
});

test("errors: no hero cards, not Hold'em, not a hand history", async () => {
  assert.throws(() => parseHandHistory(HH("stars_6max_srp").replace("Dealt to HeroName [Ah Qd]\n", "")), /hole cards aren't in this history/);
  assert.throws(() => parseHandHistory(HH("stars_6max_srp").replace("Hold'em No Limit", "Omaha Pot Limit")), /Only Hold'em/);
  assert.throws(() => parseHandHistory(HH("stars_6max_srp").replace("[Ah Qd]", "[Ah Qd 7c 2c]")), /Only Hold'em/);
  assert.throws(() => parseHandHistory("hello"), /No players found/);
  // in the dialog the message shows and the hand on the page is kept
  const p = await imported("UTG raises, I call, flop A72");
  assert.match(p.byId("hhErr").textContent, /No players found/);
  assert.equal(p.byId("importBox").hidden, false);
  assert.deepEqual(p.S.cards, [null, null, null, null, null, null, null]);
});

test("a short all-in: equal stacks on the page, the shorter player checks through", async () => {
  // a third player all-in for 30bb with two 100bb players behind; the hero (BTN) folds the turn
  const text = HH("stars_6max_srp")
    .replace("tilt_proof: folds", "tilt_proof: raises $27 to $30 and is all-in").replace("Seat 3: tilt_proof ($100 in chips)", "Seat 3: tilt_proof ($30 in chips)")
    .replace("HeroName: raises $1.50 to $2.50", "HeroName: calls $30").replace("OldTAG: calls $1.50", "OldTAG: calls $29")
    .replace(/\*\*\* FLOP \*\*\*[\s\S]*\*\*\* SUMMARY/, "*** FLOP *** [Qs 8d 3c]\nOldTAG: checks\nHeroName: checks\n*** TURN *** [Qs 8d 3c] [7h]\nOldTAG: bets $20\nHeroName: folds\n*** SUMMARY");
  const p = await imported(text);
  const S = j(p.S);
  assert.deepEqual(S.actions.preflop, j(acts("UTG f, HJ f, CO r30, BTN c, SB f, BB c")));
  assert.deepEqual(S.actions.flop, j(acts("BB x, CO x, BTN x")));
  assert.deepEqual(S.actions.turn, j(acts("BB b20, CO c, BTN f")));
  assert.match(p.byId("toast").innerHTML, /CO went all-in for less/);
});

test("hh.js is loaded only when the import dialog opens", async () => {
  assert.doesNotMatch(pageScript(), /parseHandHistory/);
  const p = loadPage();
  const added = [];
  p.ctx.document.head = { appendChild: (s) => added.push(s) };
  await p.ready();
  p.click("importHand");
  assert.equal(p.byId("importBox").hidden, false);
  assert.equal(added.length, 1);
  assert.match(added[0].src, /^hh\.js\?v=/);
  p.click("importHand");   // opening it again doesn't load it twice
  assert.equal(added.length, 1);
});

// ---- review the whole hand
function answer(p, st) {   // a made-up solve of the spot the worker was sent
  const pre = st.board === "";
  const actions = pre ? [{ action: "fold", freq: 0.1 }, { action: "check", freq: 0.1 }, { action: "call", freq: 0.2 }, { action: "raise 2.5", amount_bb: 2.5, freq: 0.6 }]
    : [{ action: "check", freq: 0.25, ev_bb: 3 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.7, ev_bb: 3.4 }, { action: "fold", freq: 0.05, ev_bb: 0 }];
  return { street: pre ? "preflop" : "postflop", hero: st.hero, hand: "AQo", recommended: actions[pre ? 3 : 1].action, actions };
}
async function respond(p) {
  const m = p.posted.at(-1);
  p.workers.at(-1).onmessage({ data: { id: m.id, result: answer(p, m.state) } });
  await flushAll();
  return m.state;
}

test("review whole hand: each decision is solved in order with progress, graded, then the hand review", async () => {
  const p = await imported(HH("stars_6max_srp"));
  const full = j(p.S);
  const done = p.run("reviewHand()");
  const sent = [];
  for (let k = 1; k <= 4; k++) {
    await flushAll();
    assert.match(p.byId("adviceNote").innerHTML, new RegExp(`Reviewing decision ${k} of 4`));
    sent.push(await respond(p));
  }
  await done;
  // the spots: each one just before your decision, with the board up to its street, at the Fast budget
  assert.deepEqual(sent.map(s => [s.board, s.preflop.length, s.flop.length, s.turn.length, s.river.length, s.speed]),
    [["", 3, 0, 0, 0, "fast"], ["Qs8d3c", 6, 1, 0, 0, "fast"], ["Qs8d3c7h", 6, 3, 1, 0, "fast"], ["Qs8d3c7h2s", 6, 3, 3, 1, "fast"]]);
  const S = j(p.S);
  assert.deepEqual(S.cards, full.cards);
  const g = ["preflop", "flop", "turn", "river"].map(s => S.actions[s].find(a => a.pos === "BTN").g);
  assert.deepEqual(g.map(x => [x.m, x.f, x.l]), [["raise 2.5", 0.6, null], ["bet 33%", 0.7, 0], ["bet 33%", 0.7, 0], ["bet 33%", 0.7, 0]]);
  assert.match(p.byId("adviceNote").innerHTML, /Hand review<\/b> · 4 of 4 decisions in the solver's mix/);
  assert.doesNotMatch(p.byId("adviceNote").innerHTML, /reviewAll/);
  assert.deepEqual(JSON.parse(p.storage.get("handreview")), S);
});

test("review whole hand respects Accurate, can be cancelled, and multiway spots are reviewed without a grade", async () => {
  const p = await imported(HH("gg_limped_multiway"), { storage: { "gto-speed": "accurate" } });
  const done = p.run("reviewHand()");
  await flushAll();
  assert.equal(p.posted.at(-1).state.speed, "accurate");
  await respond(p);   // preflop check
  const m = p.posted.at(-1);   // the flop check: three players in
  p.workers.at(-1).onmessage({ data: { id: m.id, result: { street: "flop", hero: "BB", hand: "76s", multiway: true, recommended: "check", opponents: ["SB", "UTG", "CO"], equity_vs_field: 0.3, actions: [{ action: "check", freq: 1 }] } } });
  await flushAll();
  assert.match(p.byId("adviceNote").innerHTML, /Reviewing decision 3 of 4/);
  const pending = p.posted.at(-1);
  p.run('review.cancel = "restore"; cancelSolve()');   // the Cancel link; the worker (still loading here) skips the solve
  p.workers.at(-1).onmessage({ data: { id: pending.id, result: { cancelled: true } } });
  await done;
  const S = j(p.S), mine = ["preflop", "flop", "turn"].flatMap(s => S.actions[s].filter(a => a.pos === "BB"));
  assert.equal(S.actions.turn.length, 3);   // the whole hand is back
  assert.equal(mine[0].g.m, "check");
  assert.equal(mine[1].g, null);              // multiway: reviewed, no grade
  assert.ok(!("g" in mine[2]) && !("g" in mine[3]));
  assert.match(p.byId("adviceNote").innerHTML, /id="reviewAll"/);   // what's left can still be reviewed
});
