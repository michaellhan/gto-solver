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
const HH_DIR = path.join(__dirname, "..", "data", "hh");
const HH = (f) => fs.readFileSync(path.join(HH_DIR, /\.\w+$/.test(f) ? f : f + ".txt"), "utf8");
const j = (x) => JSON.parse(JSON.stringify(x));
const r2 = (x) => Math.round(x * 100) / 100;
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
// (pot: the pot won, as the history's "Total pot": a bet nobody called goes back)
const CASES = {
  stars_6max_srp: { players: 6, hero: "BTN", stack: 100, cards: ["Ah", "Qd", "Qs", "8d", "3c", "7h", "2s"],
    preflop: "UTG f, HJ f, CO f, BTN r2.5, SB f, BB c", flop: "BB x, BTN b1.8, BB c", turn: "BB x, BTN b6.1, BB c", river: "BB x, BTN b14.5, BB f", pot: 2.5 + 2.5 + 0.5 + 3.6 + 12.2 },
  // a sitting-out seat isn't dealt in: five-handed; hero has 120bb, the CO who stays in 100bb (all-in on the turn)
  stars_3bet_pot: { players: 5, hero: "BB", stack: 120, stacks: { HJ: 100, CO: 100, BTN: 156.2, SB: 94 }, cards: ["Ac", "Kh", "Kd", "9c", "4h", "2s", "7c"],
    preflop: "HJ f, CO r2.5, BTN f, SB f, BB r11, CO c", flop: "BB b7, CO r22, BB c", turn: "BB x, CO a, BB c", river: "", pot: 200.5 },
  gg_limped_multiway: { players: 6, hero: "BB", stack: 100, cards: ["7s", "6s", "Ts", "8h", "2s", "Kd", null],
    preflop: "UTG c, HJ f, CO c, BTN f, SB c, BB x", flop: "SB x, BB x, UTG b2, CO f, SB c, BB r10.5, UTG f, SB c", turn: "SB x, BB b16, SB f", river: "", pot: 4 + 2 + 21 },
  gg_allin_preflop: { players: 6, hero: "SB", stack: 124, cards: ["Ks", "Kh", "Jd", "5c", "2h", "9s", "Kc"],
    preflop: "UTG f, HJ r2.5, CO f, BTN r7.5, SB r18, BB f, HJ f, BTN a, SB c", flop: "", turn: "", river: "", pot: 201 + 2.5 },
  stars_9max_mtt_ante: { players: 9, hero: "CO", stack: 80, ante: 0.1, cards: ["Ah", "Td", "Ac", "9h", "9d", "5s", "Qh"],
    preflop: "UTG f, UTG1 f, UTG2 f, LJ r2.2, HJ f, CO c, BTN f, SB f, BB c", flop: "BB x, LJ b2, CO c, BB f", turn: "LJ x, CO b5, LJ c", river: "LJ x, CO x", pot: 22 },
  stars_heads_up: { players: 2, hero: "BB", stack: 100, cards: ["9c", "9d", "Ts", "8s", "2c", "4h", "3d"],
    preflop: "SB r2.5, BB r9, SB c", flop: "BB x, SB b6, BB c", turn: "BB x, SB x", river: "BB b20, SB f", pot: 30 },
  gg_straddle: { players: 6, hero: "BB", stack: 124, straddle: true, cards: ["Jc", "Jd", "8c", "4d", "2s", "Ah", null],
    preflop: "HJ f, CO r7, BTN f, SB f, BB r26, UTG f, CO c", flop: "BB b17.5, CO c", turn: "BB x, CO b39, BB f", river: "", pot: 26 + 26 + 0.5 + 2 + 35 },
  // names with brackets ("x (2)"), a colon, "raises", unicode; 2c/5c blinds (a 0.4bb small blind)
  stars_zoom_names: { players: 6, hero: "UTG", stack: 128, cards: ["Td", "Tc", "9s", "4c", "2d", "Kh", null],
    preflop: "UTG r2.6, HJ f, CO r9, BTN f, SB f, BB f, UTG c", flop: "UTG x, CO b6, UTG c", turn: "UTG x, CO a, UTG f", river: "", pot: 19.5 + 12 },
  // run twice (the first board is the hand's), all-in insurance and cash-drop lines; the big blind's 45bb all-in
  // is a raise for less, the hero's jam over it is called by the 125bb button: a 45bb main pot three ways
  gg_run_twice_cashout: { players: 6, hero: "HJ", stack: 100, stacks: { CO: 90, BTN: 125, BB: 45 }, cards: ["Ah", "Ad", "2c", "7d", "9s", "Jh", "Kd"],
    preflop: "UTG f, HJ r2.5, CO f, BTN r8.5, SB f, BB a, HJ a, BTN c", flop: "", turn: "", river: "", pot: 245.5 },
  stars_run_twice: { players: 4, hero: "BTN", stack: 100, cards: ["As", "Ks", "Kd", "8s", "3s", "2h", "9c"],
    preflop: "CO r2.5, BTN r7.5, SB f, BB f, CO c", flop: "CO x, BTN b5, CO a, BTN c", turn: "", river: "", pot: 201.5 },
  // every player's cards in "Dealt to" lines: the hero is "Hero"
  gg_all_cards_shown: { players: 4, hero: "SB", stack: 100, cards: ["Ks", "Kh", "Jd", "5c", "2h", null, null],
    preflop: "CO f, BTN f, SB r3, BB c", flop: "SB b2, BB f", turn: "", river: "", pot: 6 },
  stars_bounty_allin: { players: 5, hero: "BB", stack: 50, ante: 0.12, cards: ["Jh", "Js", "3d", "5h", "9c", "Td", "Ad"],
    preflop: "HJ f, CO a, BTN f, SB f, BB c", flop: "", turn: "", river: "", pot: 50.86 },
};

// the other sites: the same hands in each site's format become the same page state
const SAME = { "888_6max_srp": "stars_6max_srp", "ignition_6max_srp": "stars_6max_srp", "888_9max_mtt_ante": "stars_9max_mtt_ante",
  winamax_mtt_ante: "stars_9max_mtt_ante", acr_9max_mtt_ante: "stars_9max_mtt_ante", party_6max_allin: "gg_allin_preflop",
  party_mtt_bounty_allin: "stars_bounty_allin", ignition_mtt_allin: "stars_bounty_allin", winamax_3bet_pot: "stars_3bet_pot",
  "ipoker_3bet_pot.xml": "stars_3bet_pot", acr_run_twice: "stars_run_twice" };
for (const [f, like] of Object.entries(SAME)) CASES[f] = CASES[like];

for (const [name, c] of Object.entries(CASES)) {
  test(`imports ${name}`, async () => {
    const p = await imported(HH(name));
    const S = j(p.S);
    assert.equal(p.byId("hhErr").textContent, "");
    assert.equal(p.byId("importBox").hidden, true);
    assert.deepEqual([S.players, S.hero, S.stack, S.ante, !!S.straddle], [c.players, c.hero, c.stack, c.ante, !!c.straddle]);
    if (c.stacks) assert.deepEqual(S.stacks, c.stacks);
    assert.deepEqual(S.cards, c.cards);
    for (const s of ["preflop", "flop", "turn", "river"]) assert.deepEqual(S.actions[s], j(acts(c[s])), s);
    const sim = p.sim();
    assert.equal(sim.handOver, true);
    assert.ok(Math.abs(sim.pot - c.pot) < 1e-9, `pot ${sim.pot}`);
    // the setup inputs follow the imported hand, it's saved, and the hand review offers to grade it
    assert.equal(p.byId("stack").value, c.stack);
    assert.deepEqual(JSON.parse(p.storage.get("handreview")), S);
    assert.match(p.buttons().find(b => b.id === "reviewAll").innerHTML, /^Review whole hand/);
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

test("other sites: detected, seats, amounts and the hero", () => {
  const sites = Object.keys(SAME).map(f => parseHandHistory(HH(f)).site);
  assert.deepEqual(sites, ["888poker", "Ignition/Bovada", "888poker", "Winamax", "WPN/ACR", "partypoker", "partypoker", "Ignition/Bovada", "Winamax", "iPoker", "WPN/ACR"]);
  for (const [f, like] of Object.entries(SAME)) {   // the parsed hand is the PokerStars/GG one but for the names and seat numbers
    const strip = (h) => ({ ...h, site: 0, heroName: 0, tournament: !!h.tournament, seats: h.seats.map(s => [s.pos, s.stack]).sort() });
    assert.deepEqual(strip(parseHandHistory(HH(f))), strip(parseHandHistory(HH(like))), f);
  }
  // 888's tournament amounts with dots for thousands; Ignition's hero is "[ME]" among everyone's cards, its seat numbers aren't the table's order
  assert.equal(parseHandHistory(HH("888_9max_mtt_ante")).seats[0].stack, r2(18250 / 300));
  const ig = parseHandHistory(HH("ignition_mtt_allin"));
  assert.deepEqual([ig.heroName, ig.hole, ig.seats.map(s => s.name + "=" + s.pos).join()], ["Big Blind [ME]", ["Jh", "Js"], "UTG+1=CO,Dealer=BTN,Small Blind=SB,Big Blind [ME]=BB,UTG=HJ"]);
  // iPoker: a session of several hands imports the first you're seated in; entities in names, comma decimals
  const x = HH("ipoker_3bet_pot.xml"), game = x.match(/<game\b[\s\S]*<\/game>/)[0];
  const two = parseHandHistory(x.replace(game, game.replace(/HeroName/g, "someone_else") + "\n" + game));
  assert.deepEqual([two.heroName, two.warnings], ["HeroName", ["2 hands pasted: imported the first."]]);
  const odd = parseHandHistory(x.replace(/grinder_pl/g, "R&amp;D").replace(/€(\d+)\.(\d\d)/g, "€$1,$2"));
  assert.deepEqual([odd.seats[0].name, odd.seats[0].stack, odd.actions.find(a => a.pos === "CO" && a.allIn).to], ["R&D", 100, 67]);
  // a trailing hand of another site is cut off; party's "players : 6/6" isn't the stakes
  assert.deepEqual(parseHandHistory(HH("888_6max_srp") + "\n\n" + HH("888_9max_mtt_ante")).warnings, ["2 hands pasted: imported the first."]);
  assert.equal(parseHandHistory(HH("party_6max_allin").replace("$0.25/$0.50 USD ", "$50 USD fastforward ")).bb, 0.5);
});

test("mutated histories never crash the importer", async () => {
  // lines dropped, cut, swapped, duplicated or with an amount changed, in every fixture: a clear Error or a playable state
  const p = await imported("");
  const texts = fs.readdirSync(HH_DIR).map(f => HH(f));
  let seed = 7, ok = 0;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648, pick = (a) => a[Math.floor(rnd() * a.length)];
  for (let it = 0; it < 1500; it++) {
    let lines = pick(texts).split("\n");
    for (let k = 1 + Math.floor(rnd() * 3); k > 0 && lines.length; k--) {
      const r = rnd(), i = Math.floor(rnd() * lines.length);
      if (r < 0.3) lines.splice(i, 1);
      else if (r < 0.45) lines = lines.slice(0, i);
      else if (r < 0.55) lines = lines.slice(i);
      else if (r < 0.7) { const t = Math.floor(rnd() * lines.length); [lines[i], lines[t]] = [lines[t], lines[i]]; }
      else if (r < 0.85) lines[i] = lines[i].replace(/[\d.]+/, String(Math.floor(rnd() * 500) / (rnd() < 0.5 ? 1 : 100)));
      else lines.splice(i, 0, pick(lines));
    }
    p.ctx.__t = lines.join("\n");
    let r;
    try {
      r = p.run("(() => { const r = importHand(__t); S = r.state; setPositions(S.players); const s = simulate(); return { st: r.state, pot: s.pot }; })()");
    } catch (e) {
      assert.equal(e.constructor.name, "Error", `${e.stack}\n${p.ctx.__t}`);
      continue;
    }
    const cards = r.st.cards.filter(Boolean);
    assert.ok(r.st.stack > 0 && isFinite(r.pot) && r.st.cards[0] && r.st.cards[1] && new Set(cards).size === cards.length, p.ctx.__t);
    ok++;
  }
  assert.ok(ok > 300, `${ok} imported`);
});

test("odd histories: no button line, mixed games, short deck, a hero without a seat line", () => {
  // a paste that starts at the seat lines: the button is the seat before the blinds, no dead-button warning
  const part = parseHandHistory(HH("stars_6max_srp").split("\n").slice(2).join("\n"));
  assert.deepEqual(part.warnings, []);
  assert.equal(part.hero, "BTN");
  assert.deepEqual(part.seats.map(s => s.pos), parseHandHistory(HH("stars_6max_srp")).seats.map(s => s.pos));
  const mixed = HH("stars_6max_srp").replace("Hold'em No Limit ($0.50/$1.00 USD)", "Mixed NLH/PLO (Hold'em No Limit, $0.50/$1.00 USD)");
  assert.equal(parseHandHistory(mixed).hero, "BTN");
  assert.throws(() => parseHandHistory(mixed.replace("(Hold'em No Limit,", "(Omaha Pot Limit,")), /Only Hold'em/);
  assert.throws(() => parseHandHistory(HH("stars_6max_srp").replace("Hold'em No Limit", "6+ Hold'em No Limit")), /Only Hold'em/);
  assert.match(parseHandHistory(HH("stars_zoom_names")).warnings[0], /small blind is 0.4bb/);
  assert.throws(() => parseHandHistory(HH("stars_6max_srp").replace("Dealt to HeroName", "Dealt to Hero Name")), /no "Seat N: Hero Name/);
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

test("a short all-in: every seat plays its own stack, the all-in player is out of the betting", async () => {
  // a third player all-in for 30bb with two 100bb players behind; the hero (BTN) folds the turn
  const text = HH("stars_6max_srp")
    .replace("tilt_proof: folds", "tilt_proof: raises $27 to $30 and is all-in").replace("Seat 3: tilt_proof ($100 in chips)", "Seat 3: tilt_proof ($30 in chips)")
    .replace("HeroName: raises $1.50 to $2.50", "HeroName: calls $30").replace("OldTAG: calls $1.50", "OldTAG: calls $29")
    .replace(/\*\*\* FLOP \*\*\*[\s\S]*\*\*\* SUMMARY/, "*** FLOP *** [Qs 8d 3c]\nOldTAG: checks\nHeroName: checks\n*** TURN *** [Qs 8d 3c] [7h]\nOldTAG: bets $20\nHeroName: folds\n*** SUMMARY");
  const p = await imported(text);
  const S = j(p.S);
  assert.deepEqual(S.stacks, { CO: 30, HJ: 143.27, SB: 62.4 });
  assert.deepEqual(S.actions.preflop, j(acts("UTG f, HJ f, CO a, BTN c, SB f, BB c")));
  assert.deepEqual(S.actions.flop, j(acts("BB x, BTN x")));
  assert.deepEqual(S.actions.turn, j(acts("BB b20, BTN f")));
  const sim = p.sim();   // BB's uncalled turn bet comes back; BB and CO go to showdown for the 90.5bb pot
  assert.deepEqual([sim.waiting, sim.pot, sim.allin, sim.active], ["river", 90.5, ["CO"], ["CO", "BB"]]);
  assert.doesNotMatch(p.byId("toast").innerHTML, /all-in for less|equal stacks/);
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
  assert.match(p.buttons().find(b => b.id === "reviewAll").innerHTML, /^Review again/);   // (nothing left: at other settings, say)
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
  assert.match(p.buttons().find(b => b.id === "reviewAll").innerHTML, /^Review whole hand/);   // what's left can still be reviewed
});

test("the import dialog names every site the importer reads", () => {
  const html = fs.readFileSync(require("./page.js").HTML_PATH, "utf8");
  const dialog = html.match(/<div id="importBox"[\s\S]*?<\/div>/)[0].replace(/<[^>]*>/g, " ");
  const sites = new Set(fs.readdirSync(HH_DIR).map(f => parseHandHistory(HH(f)).site));
  assert.ok(sites.size >= 8);
  for (const s of sites) assert.ok(dialog.includes(s), `${s} isn't named in the dialog: ${dialog.trim()}`);
});

test("tournaments: the header's id, name, buy-in and level (Roman or not); a cash game has none", () => {
  const T = (f) => j(parseHandHistory(HH(f)).tournament || null);
  assert.deepEqual(T("stars_9max_mtt_ante"), { id: "3712345678", buyin: "$10+$1", level: 7 });
  assert.deepEqual(T("stars_bounty_allin"), { id: "3800000000", buyin: "$4.40+$4.40+$1.20", level: 4 });
  assert.deepEqual(T("winamax_mtt_ante"), { id: "812345678", name: "Monster Stack", buyin: "9€+1€", level: 7 });
  assert.deepEqual(T("888_9max_mtt_ante"), { id: "198765432", buyin: "$10+$1" });
  assert.deepEqual(T("acr_9max_mtt_ante"), { id: "33339185", level: 7 });
  assert.deepEqual(T("ignition_mtt_allin"), { id: "M-20241002003411", buyin: "$8.80+$1.20", level: 4 });
  assert.deepEqual(T("party_mtt_bounty_allin"), { id: "301084002", buyin: "$8.8+$1.2" });
  for (const f of ["stars_6max_srp", "gg_straddle", "888_6max_srp", "winamax_3bet_pot", "ipoker_3bet_pot.xml"]) assert.equal(T(f), null, f);
  // GGPoker's tournament header, and iPoker's tournament tags
  const gg = HH("gg_straddle").replace(/^Poker Hand #\w+: [^\n]*/, "Poker Hand #TM2345678901: Tournament #123456789, Bounty Hunters $10 Hold'em No Limit - Level10(300/600(80)) - 2024/06/15 20:31:44");
  assert.deepEqual(j(parseHandHistory(gg).tournament), { id: "123456789", level: 10 });
  const ip = HH("ipoker_3bet_pot.xml").replace("<gametype>", "<tournamentcode>777123</tournamentcode><tournamentname>Sunday \"Big\"</tournamentname><gametype>");
  assert.deepEqual(j(parseHandHistory(ip).tournament), { id: "777123", name: "Sunday 'Big'" });
});

test("a tournament's hand takes no rake, whatever the page's Rake: its tournament is kept and the page sends none", async () => {
  const withRake = { storage: { handreview: JSON.stringify({ players: 6, hero: "BTN", stack: 100, cards: [null, null, null, null, null, null, null],
    actions: { preflop: [], flop: [], turn: [], river: [] }, rake: 10, rakeCap: 1 }) } };
  const t = await imported(HH("stars_9max_mtt_ante"), withRake);
  assert.deepEqual(j(t.S.tournament), { id: "3712345678", buyin: "$10+$1", level: 7 });
  assert.equal("rake" in t.S, false);
  assert.equal(t.payload().rake, undefined);
  // a cash game's hand takes the page's rake, as before
  const c = await imported(HH("stars_6max_srp"), withRake);
  assert.equal("tournament" in c.S, false);
  assert.equal(c.S.rake, 10); assert.equal(c.S.rakeCap, 1);
});
