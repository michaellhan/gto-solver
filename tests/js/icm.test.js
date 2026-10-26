// Tournament settings: the prizes still to be paid (S.payouts) and the players at other tables (S.fieldN of them,
// S.fieldAvg bb on average), kept in the state, sent to the solver as payouts and field (two prizes or more: the advice
// is then in prize equity) and carried by share links.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage, state, acts } = require("./page.js");

const j = (x) => JSON.parse(JSON.stringify(x));
const linkOf = (text) => text.match(/^Open this spot: (.*)$/m)[1];
const PRE = { cards: ["Ah", "Jd"], preflop: acts("UTG f, HJ f, CO f, BTN r2") };

test("no tournament by default: nothing sent, the field hidden", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(PRE);
  const pl = p.payload();
  assert.equal("payouts" in pl, false);
  assert.equal("field" in pl, false);
  assert.equal(p.byId("payouts").value, "");
  assert.equal(p.byId("fieldBox").hidden, true);
});

test("payouts typed in: sent with two prizes or more, the field once both its numbers are in", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(PRE);
  p.byId("payouts").onchange({ target: { value: "50, 30 / 20" } });
  assert.deepEqual(j(p.S.payouts), [50, 30, 20]);
  assert.deepEqual(j(p.payload().payouts), [50, 30, 20]);
  assert.equal(p.byId("fieldBox").hidden, false);
  assert.equal(JSON.parse(p.storage.get("handreview")).payouts.length, 3);   // saved with the hand
  p.byId("fieldN").onchange({ target: { value: "12.7" } });
  assert.equal(p.S.fieldN, 12);
  assert.equal("field" in p.payload(), false);   // no average stack yet
  p.byId("fieldAvg").onchange({ target: { value: "35" } });
  assert.deepEqual(j(p.payload().field), { players: 12, chips: 420 });
  p.byId("fieldAvg").onchange({ target: { value: "0" } });
  assert.equal("fieldAvg" in p.S, false);
  assert.equal("field" in p.payload(), false);
  // one prize is a cash game's all-or-nothing: not a tournament, nothing sent, the field dropped
  p.byId("payouts").onchange({ target: { value: "100" } });
  assert.equal("payouts" in p.payload(), false);
  assert.equal(p.byId("fieldBox").hidden, true);
  p.byId("payouts").onchange({ target: { value: "" } });
  assert.equal("payouts" in p.S, false); assert.equal("fieldN" in p.S, false);
});

test("a payouts change drops the shown advice", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(PRE);
  p.byId("result").className = "advice"; p.byId("result").innerHTML = "old advice";
  p.byId("payouts").onchange({ target: { value: "50 30 20" } });
  assert.equal(p.byId("result").className, "empty");
});

test("the tournament survives a change of players and a new hand", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state(PRE), payouts: [50, 30, 20], fieldN: 9, fieldAvg: 30 };
  p.run('document.getElementById("players").children[3].onclick()');   // 5 players
  assert.equal(p.S.players, 5);
  assert.deepEqual([j(p.S.payouts), p.S.fieldN, p.S.fieldAvg], [[50, 30, 20], 9, 30]);
  p.byId("newhand").onclick();
  assert.deepEqual([j(p.S.payouts), p.S.fieldN, p.S.fieldAvg], [[50, 30, 20], 9, 30]);
});

test("share link round trip with a tournament, and the copied hand says it", async () => {
  const T = { ...state(PRE), payouts: [50, 30, 20] };
  for (const st of [T, { ...T, fieldN: 20, fieldAvg: 25 }, state(PRE)]) {
    const a = loadPage();
    a.S = st;
    const text = a.run("handText()");
    const b = loadPage({ hash: linkOf(text).replace(/^[^#]*/, "") });
    assert.deepEqual(j(b.S), j(st));
    await a.ready(); await b.ready();
    assert.deepEqual(b.payload(), a.payload());
    const first = text.split("\n")[0];
    if (st.fieldN) assert.match(first, /, tournament, payouts 50\/30\/20 \(\+20 at other tables\), I am /);
    else if (st.payouts) assert.match(first, /, tournament, payouts 50\/30\/20, I am /);
    else assert.doesNotMatch(first, /tournament/);
  }
});

test("bad tournament values in a link or storage are dropped", () => {
  for (const [payouts, fieldN, fieldAvg, want] of [
    ["50 30 x 20", 5, 30, { payouts: [50, 30, 20], fieldN: 5, fieldAvg: 30 }],
    [[50, -3, "20"], -1, "x", { payouts: [50, 20] }],
    ["abc", 5, 30, {}], [null, 5, 30, {}], [{}, 5, 30, {}],
    [[60, 40], 3.9, "25", { payouts: [60, 40], fieldN: 3, fieldAvg: 25 }]]) {
    const st = { ...state(PRE), payouts, fieldN, fieldAvg };
    const p = loadPage({ hash: "#h=" + Buffer.from(JSON.stringify(st)).toString("base64") });
    const S = j(p.S);
    const got = {};
    for (const k of ["payouts", "fieldN", "fieldAvg"]) if (S[k] !== undefined) got[k] = S[k];
    assert.deepEqual(got, want, JSON.stringify([payouts, fieldN, fieldAvg]));
    const p2 = loadPage({ storage: { handreview: JSON.stringify(st) } });
    assert.deepEqual(j(p2.S), S);
  }
});

test("the advisor reads the tournament keys the page sends", () => {
  const src = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "advisor.py"), "utf8");
  for (const k of ["payouts", "field"]) assert.ok(src.includes(`state.get("${k}")`), k);
  const body = src.slice(src.indexOf("def _tournament("));
  assert.ok(body.includes('"players"') && body.includes('"chips"'));
});
