// Share links (#h=<base64 JSON of the page state>) and the Share button that copies one.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const j = (x) => JSON.parse(JSON.stringify(x));
const flush = () => new Promise(r => setImmediate(r));

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
    const link = a.run("shareLink()");
    assert.match(link, /^https:\/\/www\.michael-han\.com\/gto-solver\/#h=[A-Za-z0-9+/=]+$/);
    const b = loadPage({ hash: link.slice(link.indexOf("#")) });
    assert.deepEqual(j(b.S), j(a.S));
    assert.deepEqual(j(b.S), st);
    await a.ready(); await b.ready();
    assert.deepEqual(b.payload(), a.payload());
    assert.deepEqual(j(b.sim()), j(a.sim()));
  });
}

test("a link opened in the tab the page is already showing (only the hash changes) loads the spot", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state({ cards: ["2c", "7d"], preflop: acts("UTG f") });
  p.location.hash = "#h=" + REAL;
  p.fire("hashchange");
  await flush(); await flush();
  const S = j(p.S);
  assert.deepEqual(S.cards, ["Ks", "Kd", "7d", "2s", "5s", null, null]);
  assert.equal(S.actions.preflop.length, 9);
  assert.deepEqual(p.replaced, ["/gto-solver/"]);
  assert.deepEqual(JSON.parse(p.storage.get("handreview")), S);
  // a hash that isn't a link leaves the hand alone
  p.location.hash = "#foo";
  p.fire("hashchange");
  await flush();
  assert.deepEqual(j(p.S), S);
});

test("share link survives non-ASCII text in the state", () => {
  const a = loadPage();
  a.S = { ...state({ cards: ["Ah", "Kh"] }), note: "Ünïcödé ♠♥" };
  const b = loadPage({ hash: a.run("shareLink()").replace(/^[^#]*/, "") });
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

test("no GTO column or note on an answer with no lock", async () => {
  const q = loadPage();
  q.S = state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") });
  await q.ready();
  q.ctx.__res = JSON.stringify({ street: "flop", recommended: "check", actions: [{ action: "check", freq: 1, ev_bb: 4 }] });
  q.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.doesNotMatch(q.byId("result").innerHTML, /GTO's frequency|GTO %/);
});

test("a raise's amount isn't repeated when its name already says it, in the advice rows", async () => {
  const p = loadPage();
  p.S = state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x, BTN b2, BB r7") });
  await p.ready();
  p.ctx.__res = JSON.stringify({ street: "flop", recommended: "raise to 24.5", actions: [{ action: "fold", freq: 0.2 }, { action: "call", amount_bb: 5, freq: 0.3 },
    { action: "raise to 24.5", amount_bb: 24.5, freq: 0.4 }, { action: "raise to 30bb", amount_bb: 30, freq: 0.05 }, { action: "all-in", amount_bb: 97.5, freq: 0.05 }] });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  const html = p.byId("result").innerHTML;
  assert.match(html, /<\/i>Raise <small>to 24\.5 bb<\/small><\/div>/);
  assert.match(html, /<\/i>Raise <small>to 30 bb<\/small><\/div>/);
  assert.match(html, /<\/i>Call <small>5 bb<\/small><\/div>/);
  assert.match(html, /<\/i>All-in <small>97\.5 bb<\/small><\/div>/);
});

test("Share puts the hand's link on the clipboard", async () => {
  const p = loadPage();
  p.S = state({ cards: ["Ah", "Kh"] });
  await p.ready();
  await p.byId("share").onclick();
  assert.equal(p.clipboard.text, p.run("shareLink()"));
  assert.match(p.clipboard.text, /^https:\/\/www\.michael-han\.com\/gto-solver\/#h=/);
  assert.equal(p.byId("share").textContent, "Copied");
  await flush();
});

// Bug: Safari allows clipboard writes only during the click, so an await before the write fell through to the prompt.
test("Share works where the clipboard needs the user's gesture (Safari), even before the backend is known", async () => {
  const p = loadPage({ gesture: true });
  await p.click("share");
  assert.equal(p.clipboard.via, "writeText");
  assert.equal(p.clipboard.text, p.run("shareLink()"));
  assert.deepEqual(p.clipboard.prompts, []);
  await flush();
});

test("Share falls back to execCommand, then a prompt, when the clipboard API refuses", async () => {
  let p = loadPage({ clipboard: "denied" });
  await p.ready();
  await p.click("share");
  assert.equal(p.clipboard.via, "execCommand");
  assert.equal(p.clipboard.text, p.run("shareLink()"));
  assert.equal(p.byId("share").textContent, "Copied");
  // nothing works: the link is offered in a prompt to copy by hand
  p = loadPage({ clipboard: "denied", gesture: true });
  await p.ready();
  await p.click("share");
  assert.equal(p.clipboard.text, null);
  assert.deepEqual(p.clipboard.prompts, [p.run("shareLink()")]);
  assert.equal(p.byId("share").textContent, "Share");
  await flush();
});
