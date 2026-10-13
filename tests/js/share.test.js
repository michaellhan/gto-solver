// Share links (#h=<base64 JSON of the page state>) and the copied hand text.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const j = (x) => JSON.parse(JSON.stringify(x));
const flush = () => new Promise(r => setImmediate(r));
const linkOf = (text) => text.match(/^Open this spot: (.*)$/m)[1];

// A real link copied from the site (6-max, BTN KsKd, 4bet pot, flop 7d2s5s, BB checked).
const REAL = "eyJwbGF5ZXJzIjo2LCJoZXJvIjoiQlROIiwic3RhY2siOjEwMCwiY2FyZHMiOlsiS3MiLCJLZCIsIjdkIiwiMnMiLCI1cyIsbnVsbCxudWxsXSwiYWN0aW9ucyI6eyJwcmVmbG9wIjpbeyJwb3MiOiJVVEciLCJhY3Rpb24iOiJmb2xkIn0seyJwb3MiOiJISiIsImFjdGlvbiI6ImZvbGQifSx7InBvcyI6IkNPIiwiYWN0aW9uIjoicmFpc2UiLCJ0byI6Mi41fSx7InBvcyI6IkJUTiIsImFjdGlvbiI6ImNhbGwiLCJnIjp7Im0iOiJjYWxsIiwiZiI6MCwiYiI6InJhaXNlIiwiYmYiOjEsImwiOm51bGx9fSx7InBvcyI6IlNCIiwiYWN0aW9uIjoiZm9sZCJ9LHsicG9zIjoiQkIiLCJhY3Rpb24iOiJyYWlzZSIsInRvIjoxMH0seyJwb3MiOiJDTyIsImFjdGlvbiI6ImZvbGQifSx7InBvcyI6IkJUTiIsImFjdGlvbiI6InJhaXNlIiwidG8iOjIzLCJnIjp7Im0iOiJyYWlzZSIsImYiOjEsImIiOiJyYWlzZSIsImJmIjoxLCJsIjpudWxsfX0seyJwb3MiOiJCQiIsImFjdGlvbiI6ImNhbGwifV0sImZsb3AiOlt7InBvcyI6IkJCIiwiYWN0aW9uIjoiY2hlY2sifV0sInR1cm4iOltdLCJyaXZlciI6W119LCJzdHJhZGRsZSI6ZmFsc2V9";

test("decodes a real link from the site", async () => {
  const p = loadPage({ hash: "#h=" + REAL });
  await p.ready();
  const S = j(p.S);
  assert.equal(S.players, 6);
  assert.equal(S.hero, "BTN");
  assert.equal(S.stack, 100);
  assert.deepEqual(S.cards, ["Ks", "Kd", "7d", "2s", "5s", null, null]);
  assert.equal(S.straddle, false);
  assert.deepEqual(S.actions.flop, [{ pos: "BB", action: "check" }]);
  assert.equal(S.actions.preflop.length, 9);
  assert.deepEqual(S.actions.preflop[7], { pos: "BTN", action: "raise", to: 23, g: { m: "raise", f: 1, b: "raise", bf: 1, l: null } });
  // the hash is cleared, the state is kept for reloads, and it's hero's flop decision
  assert.deepEqual(p.replaced, ["/gto-solver/"]);
  assert.deepEqual(JSON.parse(p.storage.get("handreview")), S);
  const sim = p.sim();
  assert.equal(sim.current, "flop");
  assert.equal(sim.toAct, "BTN");
  assert.equal(sim.pot, 23 + 23 + 2.5 + 0.5);   // both 4bets, CO's dead open and the SB
  const pl = p.payload();
  assert.equal(pl.hole, "KsKd");
  assert.equal(pl.board, "7d2s5s");
  assert.equal(pl.straddle, undefined);   // straddle: false is not sent
  assert.deepEqual(pl.flop, [{ pos: "BB", action: "check" }]);
  assert.deepEqual(pl.preflop[7], { pos: "BTN", action: "raise", to: 23 });   // grades stay on the page
});

const SPOTS = {
  "preflop, empty board": state({ cards: ["Ah", "Kh"], preflop: acts("UTG f, HJ r2.5") }),
  "no cards yet": state({}),
  "heads-up flop": state({ players: 2, hero: "BB", cards: ["9c", "9d", "Ts", "8s", "2c"], preflop: acts("SB r2.5, BB c"), flop: acts("BB x, SB b1.5") }),
  "straddled 9-handed turn with an ante": state({ players: 9, hero: "UTG", ante: 0.12, straddle: true, stack: 250,
    cards: ["Qs", "Qd", "Js", "7h", "2d", "Ac"], preflop: acts("UTG1 r6, UTG2 f, LJ f, HJ f, CO f, BTN f, SB f, BB f, UTG c"), flop: acts("UTG x, UTG1 b4"), turn: [] }),
  "3-handed river": state({ players: 3, hero: "SB", stack: 40, cards: ["5h", "4h", "3h", "2h", "Kc", "Kd", "6s"],
    preflop: acts("BTN r2.5, SB c, BB c"), flop: acts("SB x, BB x, BTN b2.5, SB c, BB f"), turn: acts("SB x, BTN x"), river: acts("SB b10, BTN a") }),
};
for (const n of [2, 3, 4, 5, 7, 8]) SPOTS[`${n}-handed preflop`] = state({ players: n, hero: n === 2 ? "SB" : "BTN", cards: ["Tc", "Td"] });

for (const [name, st] of Object.entries(SPOTS)) {
  test(`share link round trip: ${name}`, async () => {
    const a = loadPage();
    a.S = st;
    const link = linkOf(a.run("handText()"));
    assert.match(link, /^https:\/\/www\.michael-han\.com\/gto-solver\/#h=[A-Za-z0-9+/=]+$/);
    const b = loadPage({ hash: link.slice(link.indexOf("#")) });
    assert.deepEqual(j(b.S), j(a.S));
    assert.deepEqual(j(b.S), st);
    await a.ready(); await b.ready();
    assert.deepEqual(b.payload(), a.payload());
    assert.deepEqual(j(b.sim()), j(a.sim()));
  });
}

test("share link survives non-ASCII text in the state", () => {
  const a = loadPage();
  a.S = { ...state({ cards: ["Ah", "Kh"] }), note: "Ünïcödé ♠♥" };
  const b = loadPage({ hash: linkOf(a.run("handText()")).replace(/^[^#]*/, "") });
  assert.equal(b.S.note, "Ünïcödé ♠♥");
});

test("a broken or foreign link is ignored and the saved hand stays", () => {
  const saved = JSON.stringify(state({ cards: ["2c", "3c"] }));
  for (const hash of ["#h=notbase64!!", "#h=" + Buffer.from("{not json").toString("base64"), "#h=" + Buffer.from('{"players":12,"cards":[],"actions":{}}').toString("base64"), "#x=1"]) {
    const p = loadPage({ hash, storage: { handreview: saved } });
    assert.deepEqual(p.S.cards.slice(0, 2), ["2c", "3c"], hash);
  }
});

// Bug: a state missing a street's action list (or with a seat that isn't at the table, or a card twice)
// was accepted as is, saved, and crashed simulate() on every later page load.
test("a malformed link is repaired, not saved as a page-breaking state", async () => {
  const bad = { players: "6", hero: "UTG7", stack: -3, cards: ["As", "As", "Kd", "zz", "2c", "3c"],
    actions: { preflop: [...acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), { pos: "XX", action: "fold" }, { pos: "BB", action: "dance" }], flop: acts("BB x, BTN x") } };
  const p = loadPage({ hash: "#h=" + Buffer.from(JSON.stringify(bad)).toString("base64") });
  const S = j(p.S);
  assert.equal(S.players, 6);
  assert.equal(S.hero, "BTN");
  assert.equal(S.stack, 100);
  assert.deepEqual(S.cards, ["As", null, "Kd", null, "2c", "3c", null]);
  assert.deepEqual(S.actions.preflop, acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"));
  assert.deepEqual(S.actions.turn, []);
  assert.deepEqual(S.actions.river, []);
  // and the same repair applies to what this browser saved
  const q = loadPage({ storage: { handreview: JSON.stringify(bad) } });
  assert.deepEqual(j(q.S), S);
  p.run("renderAll()");
});

test("a corrupt saved hand falls back to a fresh one", () => {
  for (const saved of ["{oops", "null", '"text"', '{"players":6}']) {
    const p = loadPage({ storage: { handreview: saved } });
    assert.deepEqual(j(p.S), state({}), saved);
  }
});

// ---- the copied hand text
function graded(f, l, extra = {}) { return { m: "x", f, b: "raise", bf: 0.6, l, ...extra }; }

test("hand text: header, streets with boards, verdict marks, link and state JSON", async () => {
  const p = loadPage();
  const st = state({ players: 6, hero: "CO", stack: 100, ante: 0.1, straddle: true, cards: ["Ah", "Qh", "Qs", "8d", "3c", "2h"],
    preflop: acts("HJ f, CO r6, BTN f, SB f, BB f, UTG c"), flop: acts("UTG x, CO b4, UTG c"), turn: acts("UTG x") });
  st.actions.preflop[1].g = graded(0.62, 0);
  st.actions.flop[1].g = graded(0.1, 0.42);
  p.S = st;
  await p.ready();
  const lines = p.run("handText()").split("\n");
  assert.equal(lines[0], "GTO Solver hand: 6-handed, 100bb effective, 0.1bb ante, UTG straddles 2bb, I am CO with AhQh");
  assert.equal(lines[1], "Preflop: HJ folds, CO raises to 6 [✓ solver 62%], BTN folds, SB folds, BB folds, UTG calls");
  assert.equal(lines[2], "Flop (Qs 8d 3c): UTG checks, CO bets 4 [~ solver 10%, -0.42bb], UTG calls");
  // pot: 6 antes 0.6 + SB 0.5 + BB 1 + 6 + 6 + 4 + 4
  assert.equal(lines[3], "Turn (2h): UTG checks — CO to act, pot 22.1bb");
  assert.match(lines[4], /^Open this spot: https:\/\/www\.michael-han\.com\/gto-solver\/#h=/);
  assert.equal(lines[5], "State JSON: " + JSON.stringify(p.payload()));
  assert.equal(lines.length, 6);
});

test("hand text: heads-up seat label, unknown cards, a street waiting for action", () => {
  const p = loadPage();
  p.S = state({ players: 2, hero: "SB", stack: 50 });
  const lines = p.run("handText()").split("\n");
  assert.equal(lines[0], "GTO Solver hand: 2-handed, 50bb effective, I am BTN/SB with ??");
  assert.equal(lines[1], "Preflop: BTN/SB to act, pot 1.5bb");
});

test("hand text: the solver's advice", async () => {
  const p = loadPage();
  p.S = state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") });
  await p.ready();
  p.ctx.__res = JSON.stringify({ street: "flop", position: "IP", hero: "BTN", hand: "AKs", board: "Kd7s2c", pot_bb: 5.5, recommended: "bet 33%",
    actions: [{ action: "check", freq: 0.25, ev_bb: 4.1 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.75, ev_bb: 4.3 }],
    range_action_freq: { check: 0.4, "bet 33%": 0.6 }, villain: "BB", villain_range: "AA,KK", villain_range_combos: 12.5,
    solver: { iterations: 300, solve_seconds: 2.4, active_combos: 400, tree_actions: ["check", "bet 33%"], exploitability_pct: 1.2 }, notes: ["a", "b"] });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  const lines = p.run("handText()").split("\n");
  assert.deepEqual(lines.slice(2, 9), [
    "Flop (Kd 7s 2c): BB checks — BTN to act, pot 5.5bb",
    "Advice (flop, IP): bet 33%",
    "  check 25.0% EV +4.10 | bet 33% 1.8bb 75.0% EV +4.30",
    "  whole range: check 40%, bet 33% 60%",
    "  BB range (12.5 combos): AA,KK",
    "  solver: 300 iterations, 2.4s, 400 combos, sizes check, bet 33%",
    "  notes: a; b",
  ]);
  assert.match(lines[9], /^Open this spot: /);
  // and against the pool, the GTO frequencies are listed too
  p.ctx.__res = JSON.stringify({ street: "flop", recommended: "check", actions: [{ action: "check", freq: 1, gto_freq: 0.4 }, { action: "bet 75%", amount_bb: 4.1, freq: 0, gto_freq: 0.6 }] });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  const l2 = p.run("handText()").split("\n");
  assert.equal(l2[3], "Advice (flop, vs a typical player pool): check");
  assert.equal(l2[4], "  check 100.0% (GTO 40%) | bet 75% 4.1bb 0.0% (GTO 60%)");
});

test("copy hand puts the hand text on the clipboard", async () => {
  const p = loadPage();
  p.S = state({ cards: ["Ah", "Kh"] });
  await p.ready();
  await p.byId("copyhand").onclick();
  assert.equal(p.clipboard.text, p.run("handText()"));
  assert.equal(p.byId("copyhand").textContent, "Copied");
  await flush();
});
