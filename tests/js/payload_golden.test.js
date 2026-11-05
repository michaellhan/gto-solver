// What the page sends to the solver, pinned. tests/data/payload_golden.json holds what payload() returned on the code
// before the cut-down (3b3aa950) for every state in tests/data/page_states.json and for saved hands with each setting
// (stacks, ante, straddle, rake, stakes, payouts, the field, bounties, sizes, ranges, size optimization, locks, the
// Accurate speed), reached the ways a hand reaches the page: this browser's saved hand, a #h= link, a decision of a
// reviewed hand. Removing a control removes the input, not the setting, so every entry must keep answering the same.
//
// One exception, and only one: the Fast / Accurate control has left and the page always solves at Fast, so the entries
// recorded at Accurate (with gto-speed still stored here, which the page no longer reads) expect speed "fast" and Fast's
// two budgets. Nothing else may differ.
//
// UPDATE_GOLDEN=1 node --test tests/js/payload_golden.test.js adds entries for new cases. It never changes an entry
// that's there: a different payload for a saved hand is a bug, or a decision to take back to the file by hand.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage, state, acts } = require("./page.js");

const GOLDEN = path.join(__dirname, "..", "data", "payload_golden.json");
const flush = () => new Promise(r => setImmediate(r));

// page_states.json's state is the request format; the page's own hand from it
function handOf(st) {
  const board = (st.board || "").match(/../g) || [], hole = (st.hole || "").match(/../g) || [];
  const cards = [hole[0] || null, hole[1] || null, ...board];
  while (cards.length < 7) cards.push(null);
  const act = (a) => a.action === "bet" ? { pos: a.pos, action: "bet", to: a.amount } : a;
  const out = { players: st.players, hero: st.hero, stack: st.stack, cards, actions: Object.fromEntries(["preflop", "flop", "turn", "river"].map(s => [s, st[s].map(act)])) };
  for (const k of ["stacks", "ante", "straddle"]) if (st[k] !== undefined) out[k] = st[k];
  return out;
}

const FLOP = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") };
const PRE = { cards: ["Ah", "Kh"], preflop: acts("UTG f, HJ f, CO f") };
const TURN = { cards: ["Ah", "Kh", "Kd", "7s", "2c", "9h"], preflop: FLOP.preflop, flop: acts("BB x, BTN b2, BB c"), turn: acts("BB x") };
const LOCK = { fold: -20, bluff: -50, raise: -50 };
const ALL = { stacks: { BB: 60, SB: 250 }, ante: 0.25, straddle: true, rake: 5, rakeCap: 3, stakes: { sb: 2, bb: 5 }, payouts: [50, 30, 20], fieldN: 40, fieldAvg: 55,
  bounty: { fmt: "pko", all: 10, seats: { BB: 25 }, field: 12 }, sizes: { flop: [33, 75], turn: [50], river: [100], raise: [60] }, ranges: { BB: "22+,A2s+,KTo+", BTN: "AA" },
  optSizes: true, locks: { BB: LOCK, BTN: { fold: 10 } } };
const S = (name, spot, extra, opts) => ({ name, state: state(spot), extra: extra || {}, ...opts });
const CASES = [
  S("plain flop", FLOP), S("plain preflop", PRE), S("plain turn", TURN),
  S("stack 40", FLOP, { stack: 40 }), S("stacks", FLOP, { stacks: { BB: 60, SB: 250 } }), S("stacks preflop", PRE, { stacks: { BB: 60, UTG: 300 } }),
  S("ante", FLOP, { ante: 0.25 }), S("ante preflop", PRE, { ante: 1 }),
  S("straddle", { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("HJ f, CO f, BTN r5, SB f, BB f, UTG c"), flop: acts("UTG x") }, { straddle: true }),
  S("straddle preflop", { cards: ["Ah", "Kh"], preflop: acts("HJ f, CO f") }, { straddle: true }),
  S("rake and cap", FLOP, { rake: 5, rakeCap: 3 }), S("rake without a cap", FLOP, { rake: 4.5 }), S("rake preflop", PRE, { rake: 5, rakeCap: 3 }),
  S("stakes", FLOP, { stakes: { sb: 2, bb: 5 } }), S("stakes with a stack and an ante", FLOP, { stakes: { sb: 1, bb: 3 }, stack: 120, ante: 0.5 }),
  S("payouts", FLOP, { payouts: [50, 30, 20] }), S("payouts preflop", PRE, { payouts: [50, 30, 20] }), S("one payout, nothing at stake", FLOP, { payouts: [100] }),
  S("payouts and the field", FLOP, { payouts: [50, 30, 20], fieldN: 40, fieldAvg: 55 }), S("the field without an average", FLOP, { payouts: [50, 30, 20], fieldN: 40 }),
  S("payouts over rake", FLOP, { payouts: [50, 30, 20], rake: 5, rakeCap: 3 }), S("an imported tournament hand over rake", FLOP, { tournament: true, rake: 5, rakeCap: 3 }),
  S("KO bounty", FLOP, { bounty: { fmt: "ko", all: 10 } }), S("PKO, a seat's own, the field", FLOP, { bounty: { fmt: "pko", all: 10, seats: { BB: 25 }, field: 12 }, payouts: [100], fieldN: 20, fieldAvg: 40 }),
  S("mystery envelope", FLOP, { bounty: { fmt: "mystery", env: 30 }, payouts: [50, 30] }), S("mystery pool", FLOP, { bounty: { fmt: "mystery", pool: 900, left: 30 } }),
  S("a bounty format with nothing at stake", FLOP, { bounty: { fmt: "ko" } }),
  S("sizes", FLOP, { sizes: { flop: [33, 75], turn: [50], river: [100], raise: [60] } }), S("sizes preflop", PRE, { sizes: { flop: [33, 75] } }),
  S("ranges", FLOP, { ranges: { BB: "22+,A2s+,KTo+" } }), S("ranges preflop", PRE, { ranges: { BB: "22+" } }), S("hero's own range", FLOP, { ranges: { BTN: "AA" } }),
  S("size optimization", FLOP, { optSizes: true }), S("size optimization preflop", PRE, { optSizes: true }),
  S("locks", FLOP, { locks: { BB: LOCK } }), S("locks preflop", PRE, { locks: { BB: LOCK } }), S("hero's own lock", FLOP, { locks: { BTN: LOCK } }),
  S("graded and reviewed", { ...FLOP, flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 2, g: { c: "ok", l: 0 } }] }, { finished: true, reviewed: true, payouts: [50, 30, 20] }),
  S("everything flop", FLOP, ALL), S("everything turn", TURN, ALL), S("everything preflop", PRE, ALL),
  S("everything 9-handed", { players: 9, hero: "CO", cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, UTG1 f, UTG2 f, LJ f, HJ f, CO r2.5, BTN f, SB f, BB c"), flop: acts("BB x") }, { ...ALL, stacks: { LJ: 40 }, locks: { BB: LOCK }, ranges: { BB: "22+" } }),
  S("heads-up, a straddle and an ante ignored", { players: 2, hero: "BB", cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("SB r2.5, BB c"), flop: acts("BB x") }, { straddle: true, ante: 0 }),
  S("accurate flop", FLOP, {}, { speed: "accurate" }), S("accurate preflop", PRE, {}, { speed: "accurate" }), S("accurate turn", TURN, {}, { speed: "accurate" }),
  S("accurate everything", FLOP, ALL, { speed: "accurate" }),
  S("local server fast", FLOP, {}, { server: true }), S("local server accurate", FLOP, {}, { server: true, speed: "accurate" }),
].map(c => ({ name: c.name, state: { ...c.state, ...c.extra }, ...(c.speed ? { speed: c.speed } : {}), ...(c.server ? { server: true } : {}) }));

async function boot(c, how) {
  const storage = { ...(c.speed ? { "gto-speed": c.speed } : {}), ...(how === "saved" ? { handreview: JSON.stringify(c.state) } : {}) };
  const hash = how === "link" ? "#h=" + btoa(unescape(encodeURIComponent(JSON.stringify(c.state)))) : "";
  const p = loadPage({ storage, hash, server: !!c.server });
  await p.ready(); await flush();
  if (how === "direct") p.S = c.state;
  if (how === "decision") {
    // the hand as a review shows a decision of it (the actions after it played out and stepped back over)
    p.spot(c.state);
    await flush();
  }
  return p;
}

// The payload a golden entry expects now: the entry, or (the exception above) at Fast.
const FAST = { wasm: [8, 3], server: [2.5, 2.5] };
function expected(g) {
  if (g.payload.speed === "fast") return g.payload;
  const [budget, prior_budget] = FAST[g.server ? "server" : "wasm"];
  return { ...g.payload, budget, prior_budget, speed: "fast" };
}
function same(got, g, how) {
  const want = expected(g);
  assert.deepEqual(got, want, `${g.name} (${how})`);
  assert.deepEqual(Object.keys(got), Object.keys(want), `${g.name} (${how}): the keys' order (a payload is the cache's key)`);
}

if (process.env.UPDATE_GOLDEN) test("recording tests/data/payload_golden.json (new entries only)", async () => {
  const before = fs.existsSync(GOLDEN) ? JSON.parse(fs.readFileSync(GOLDEN, "utf8")) : [];
  const have = new Set(before.map(g => g.name)), out = [...before];
  const p = loadPage();
  await p.ready();
  JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "page_states.json"), "utf8")).forEach((sp, i) => {
    const name = `page_states ${i}`;
    if (have.has(name)) return;
    p.S = handOf(sp.state);
    out.push({ name, state: handOf(sp.state), payload: p.payload() });
  });
  for (const c of CASES) {
    if (have.has(c.name)) continue;
    out.push({ ...c, payload: (await boot(c, "direct")).payload() });
  }
  fs.writeFileSync(GOLDEN, "[\n" + out.map(g => JSON.stringify(g)).join(",\n") + "\n]\n");
});
else {
  const golden = JSON.parse(fs.readFileSync(GOLDEN, "utf8"));
  const pages = golden.filter(g => g.name.startsWith("page_states ")), hands = golden.filter(g => !g.name.startsWith("page_states "));

  test("the golden file covers page_states.json and every case, and each setting reaches a payload", () => {
    assert.equal(pages.length, JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "page_states.json"), "utf8")).length);
    assert.deepEqual(hands.map(g => g.name), CASES.map(c => c.name));
    const of = (name) => hands.find(g => g.name === name).payload;
    for (const [name, key] of [["stacks", "stacks"], ["ante", "ante"], ["straddle", "straddle"], ["rake and cap", "rake"], ["rake and cap", "rake_cap"], ["payouts", "payouts"],
      ["payouts and the field", "field"], ["KO bounty", "bounties"], ["sizes", "sizes"], ["ranges", "ranges"], ["size optimization", "optimize_sizes"], ["locks", "locks"]])
      assert.ok(key in of(name), `${name}: no ${key}`);
    assert.equal(of("accurate flop").speed, "accurate");
    assert.equal(of("accurate flop").budget, 60);
    assert.equal(of("local server accurate").budget, 30);
    assert.ok(!("rake" in of("payouts over rake")) && !("rake" in of("an imported tournament hand over rake")));
  });

  test("every state of page_states.json sends what it did", async () => {
    const p = loadPage();
    await p.ready();
    for (const g of pages) { p.S = g.state; same(p.payload(), g, "direct"); }
  });

  for (const how of ["saved", "link", "direct", "decision"])
    test(`a saved hand with each setting sends what it did: ${how}`, async () => {
      for (const g of hands) same((await boot(g, how)).payload(), g, how);
    });
}
