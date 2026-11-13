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

test("Copy hand: the table, your cards and the actions so far, in a few lines", () => {
  const p = loadPage();
  p.S = { players: 6, hero: "BTN", stack: 100, cards: ["Ah", "Kh", "Kd", "7s", "2c", null, null], actions: {
    preflop: [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "raise", to: 2.5 }, { pos: "CO", action: "fold" }, { pos: "BTN", action: "call" }, { pos: "SB", action: "fold" }, { pos: "BB", action: "fold" }],
    flop: [{ pos: "HJ", action: "bet", to: 2 }], turn: [], river: [] } };
  assert.equal(p.run("handText()"), [
    "6-handed, 100bb stacks, I am BTN with AhKh",
    "Preflop: UTG folds, HJ raises to 2.5, CO folds. BTN (me) to act, pot 4bb, 2.5bb to call",
    "  Me: call. Solver: no answer yet",
    "Then: SB folds, BB folds",
    "Flop (Kd 7s 2c): HJ bets 2. BTN (me) to act, pot 8.5bb, 2bb to call"].join("\n"));
  p.run("S.actions.flop = []; S.cards = ['Ah', 'Kh', null, null, null, null, null]; S.actions.preflop = []; S.finished = false");
  assert.equal(p.run("handText()"), "6-handed, 100bb stacks, I am BTN with AhKh\nPreflop: UTG to act, pot 1.5bb, 1bb to call");
  // an answer stays with the decision it answers: a later 3bet starts a line of its own, with the spot facing it
  p.S = { players: 8, hero: "BTN", stack: 100, cards: ["Ad", "Kc", null, null, null, null, null], actions: {
    preflop: [{ pos: "UTG", action: "raise", to: 3 }, { pos: "UTG1", action: "call" }, { pos: "LJ", action: "fold" }, { pos: "HJ", action: "fold" }, { pos: "CO", action: "fold" },
      { pos: "BTN", action: "call", adv: { m: "call", f: 0.43, b: "raise", ba: 12, bf: 0.57, l: 0 } }, { pos: "SB", action: "raise", to: 18 }, { pos: "BB", action: "fold" }, { pos: "UTG", action: "fold" }, { pos: "UTG1", action: "fold" }],
    flop: [], turn: [], river: [] } };
  p.run(`lastResult = ${JSON.stringify({ street: "preflop", pot_bb: 28, actions: [{ action: "call", freq: 1 }, { action: "fold", freq: 0 }] })}; resultKey = JSON.stringify(payload())`);
  assert.equal(p.run("handText()"), [
    "8-handed, 100bb stacks, I am BTN with AdKc",
    "Preflop: UTG raises to 3, UTG1 calls, LJ folds, HJ folds, CO folds. BTN (me) to act, pot 7.5bb, 3bb to call",
    "  Me: call (solver 43%). Solver: raise to 12 (57%).",
    "Then: SB raises to 18, BB folds, UTG folds, UTG1 folds. BTN (me) to act, pot 28bb, 15bb to call",
    "  Me: to decide. Solver: call (100%)"].join("\n"));
});

test("Copy hand: each of your decisions beside the solver's answer, which may differ", () => {
  const p = loadPage();
  const line = (a) => p.run(`decisionLine(${JSON.stringify(a)}, 10)`);
  const call = { pos: "BTN", action: "call" };
  assert.equal(line({ ...call, g: { m: "call", f: 0.62, b: "call", bf: 0.62, l: 0, e: 1, o: 1 } }), "  Me: call. Solver: call (62%).");
  assert.equal(line({ ...call, g: { m: "call", f: 0.38, b: "raise to 8", bf: 0.62, l: 0.12, e: 1, o: 1.12 } }), "  Me: call (solver 38%). Solver: raise to 8 (62%). Lost 0.12bb.");
  // (no grade yet: the answer that was on screen; a review's grade goes first)
  const seen = { m: "fold", f: 0, b: "call", bf: 1, l: 1.5 }, seenRow = { pos: "BTN", action: "fold", adv: seen };
  assert.equal(line(seenRow), "  Me: fold (solver 0%). Solver: call (100%). Lost 1.50bb.");
  assert.equal(line({ ...seenRow, g: { ...seen, f: 0.5, bf: 0.5, l: 0 } }), "  Me: fold (solver 50%). Solver: call (50%).");
  // a size the solver doesn't have, graded as its nearest; a rough EV; a rough solve; the betting rules; a chart
  assert.equal(line({ pos: "BTN", action: "bet", to: 5, g: { m: "bet 33%", f: 0.2, b: "check", bf: 0.8, l: 0.3, rv: 1, r: 1 } }), "  Me: bet 5 (solver 20% as its bet 33%). Solver: check (80%). Lost ≈ 0.30bb. Rough solve.");
  assert.equal(line({ ...call, g: { m: "call", f: 0, b: "raise to 9", bf: 1, l: 2, h: 20, e: 1, o: 3 } }), "  Me: call. Betting rules: raise to 9. Lost 2.00bb.");
  assert.equal(line({ ...call, g: { m: "call", f: 1, b: "call", bf: 1, l: null } }), "  Me: call. Solver: call (100%).");
  assert.equal(line({ ...call, g: { m: "call", f: 0, b: "raise", ba: 7.5, bf: 1, l: null } }), "  Me: call (solver 0%). Solver: raise to 7.5 (100%).");
  assert.equal(line({ pos: "BTN", action: "raise", to: 8, g: { m: "raise", f: 1, b: "raise", ba: 7.5, bf: 1, l: null } }), "  Me: raise to 8. Solver: raise to 7.5 (100%).");
  assert.equal(line({ ...call, g: null }), "  Me: call. Solver: no grade");
});

test("Copy hand: at your turn, the answer on screen; and your action keeps it", () => {
  const p = loadPage();
  p.S = { players: 6, hero: "UTG", stack: 100, cards: ["Ah", "Kh", null, null, null, null, null], actions: { preflop: [], flop: [], turn: [], river: [] } };
  const res = { street: "preflop", pot_bb: 1.5, actions: [{ action: "fold", freq: 0.25 }, { action: "raise to 2.5", amount_bb: 2.5, freq: 0.75 }, { action: "all-in", freq: 0 }] };
  p.run(`lastResult = ${JSON.stringify(res)}; resultKey = JSON.stringify(payload())`);
  assert.equal(p.run("handText()").split("\n").at(-1), "  Me: to decide. Solver: raise to 2.5 (75%), fold (25%)");
  p.run("act({ action: 'fold' })");
  const a = p.run("S.actions.preflop[0]");
  assert.equal(a.adv.b, "raise to 2.5");
  p.run(`S.actions.preflop = []; lastResult = ${JSON.stringify({ ...res, actions: [{ action: "fold", freq: 0.25 }, { action: "raise", amount_bb: 2.5, freq: 0.75 }] })}; resultKey = JSON.stringify(payload())`);
  assert.equal(p.run("handText()").split("\n").at(-1), "  Me: to decide. Solver: raise to 2.5 (75%), fold (25%)");
  p.run("act({ action: 'fold' })");
  assert.equal("g" in a, false);   // (not a grade: the hand isn't reviewed)
  assert.equal(p.run("handText()").split("\n")[2], "  Me: fold (solver 25%). Solver: raise to 2.5 (75%).");
  // someone else's turn, or an answer for another spot: nothing kept
  p.run("S.actions.preflop = []; resultKey = 'other'; act({ action: 'fold' })");
  assert.equal("adv" in p.run("S.actions.preflop[0]"), false);
});

test("Copy hand copies the text and says so", async () => {
  const p = loadPage();
  await p.ready();
  await p.click("copyhand");
  assert.equal(p.clipboard.text, p.run("handText()"));
  assert.equal(p.byId("copyhand").textContent, "Copied");
  await flush();
});

