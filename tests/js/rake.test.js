// The rake setting: a select (off / 5% cap 3bb / custom) in the setup panel, kept in the state (S.rake in
// percent, S.rakeCap in bb), sent to the solver as rake (a fraction) and rake_cap (bb), and carried by share links.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage, state, acts } = require("./page.js");

const j = (x) => JSON.parse(JSON.stringify(x));
const FLOP = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") };
const withRake = (st, rake, rakeCap) => ({ ...st, rake, ...(rakeCap !== undefined ? { rakeCap } : {}) });

test("rake is off by default and not sent", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  const pl = p.payload();
  assert.equal("rake" in pl, false);
  assert.equal("rake_cap" in pl, false);
  assert.equal(p.byId("rake").value, "off");
  assert.equal(p.byId("rakeCustom").hidden, true);
});

test("payload: rake as a fraction of the pot, the cap in bb, no cap key without a cap", async () => {
  const p = loadPage();
  await p.ready();
  p.S = withRake(state(FLOP), 5, 3);
  assert.equal(p.payload().rake, 0.05);
  assert.equal(p.payload().rake_cap, 3);
  p.S = withRake(state(FLOP), 4.5);
  assert.equal(p.payload().rake, 0.045);
  assert.equal("rake_cap" in p.payload(), false);
  p.S = withRake(state(FLOP), 0, 3);   // a zero rate is no rake, whatever the cap
  assert.equal("rake" in p.payload(), false);
  assert.equal("rake_cap" in p.payload(), false);
});

test("the rake select: preset, custom percent and cap, off", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  p.byId("rake").onchange({ target: { value: "5-3" } });
  assert.equal(p.S.rake, 5); assert.equal(p.S.rakeCap, 3);
  assert.equal(p.byId("rake").value, "5-3");
  assert.equal(p.byId("rakeCustom").hidden, true);
  assert.deepEqual([p.payload().rake, p.payload().rake_cap], [0.05, 3]);
  // the setting is saved with the hand
  assert.equal(JSON.parse(p.storage.get("handreview")).rake, 5);
  p.byId("rake").onchange({ target: { value: "custom" } });   // starts from the current rake
  assert.equal(p.byId("rakeCustom").hidden, false);
  assert.equal(p.byId("rake").value, "custom");
  assert.equal(p.byId("rakePct").value, "5"); assert.equal(p.byId("rakeCap").value, "3");
  p.byId("rakePct").onchange({ target: { value: "6.5" } });
  p.byId("rakeCap").onchange({ target: { value: "4" } });
  assert.equal(p.S.rake, 6.5); assert.equal(p.S.rakeCap, 4);
  assert.deepEqual([p.payload().rake, p.payload().rake_cap], [0.065, 4]);
  p.byId("rakeCap").onchange({ target: { value: "0" } });     // no cap
  assert.equal("rakeCap" in p.S, false);
  assert.equal("rake_cap" in p.payload(), false);
  p.byId("rakePct").onchange({ target: { value: "80" } });    // held to 25%
  assert.equal(p.S.rake, 25);
  p.byId("rake").onchange({ target: { value: "off" } });
  assert.equal("rake" in p.S, false); assert.equal("rakeCap" in p.S, false);
  assert.equal("rake" in p.payload(), false);
  assert.equal(p.byId("rakeCustom").hidden, true);
  // custom from off: 5% cap 3bb to edit
  p.byId("rake").onchange({ target: { value: "custom" } });
  assert.equal(p.S.rake, 5); assert.equal(p.S.rakeCap, 3);
  assert.equal(p.byId("rake").value, "custom");
});

test("a rake change drops the shown advice (it was for another game)", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  p.byId("result").className = "advice"; p.byId("result").innerHTML = "old advice";
  p.byId("rake").onchange({ target: { value: "5-3" } });
  assert.equal(p.byId("result").className, "empty");
});

test("rake survives a change of players and a new hand", async () => {
  const p = loadPage();
  await p.ready();
  p.S = withRake(state({ ...FLOP, hero: "BB" }), 5, 3);
  p.run('document.getElementById("players").children[7].onclick()');   // 9 players
  assert.equal(p.S.players, 9); assert.equal(p.S.rake, 5); assert.equal(p.S.rakeCap, 3);
  p.byId("newhand").onclick();
  assert.equal(p.S.rake, 5); assert.equal(p.S.rakeCap, 3);
  p.S = state(FLOP);                                                 // without rake, none appears
  p.byId("newhand").onclick();
  assert.equal("rake" in p.S, false); assert.equal("rakeCap" in p.S, false);
});

test("share link round trip with rake", async () => {
  for (const st of [withRake(state(FLOP), 5, 3), withRake(state(FLOP), 7.5), state(FLOP)]) {
    const a = loadPage();
    a.S = st;
        const b = loadPage({ hash: a.run("shareLink()").replace(/^[^#]*/, "") });
    assert.deepEqual(j(b.S), st);
    await a.ready(); await b.ready();
    assert.deepEqual(b.payload(), a.payload());
    assert.equal(b.byId("rake").value, !st.rake ? "off" : st.rakeCap === 3 && st.rake === 5 ? "5-3" : "custom");
  }
});

test("a bad rake in a link or storage is dropped", () => {
  for (const [rake, rakeCap, want] of [["abc", 3, {}], [40, 3, {}], [-1, 3, {}], [null, 3, {}], [5, -2, { rake: 5 }],
                                       [5, "x", { rake: 5 }], ["5", "3", { rake: 5, rakeCap: 3 }]]) {
    const st = { ...state(FLOP), rake, rakeCap };
    const p = loadPage({ hash: "#h=" + Buffer.from(JSON.stringify(st)).toString("base64") });
    const S = p.S;
    assert.deepEqual({ ...(S.rake !== undefined ? { rake: S.rake } : {}), ...(S.rakeCap !== undefined ? { rakeCap: S.rakeCap } : {}) },
                     want, JSON.stringify([rake, rakeCap]));
    const p2 = loadPage({ storage: { handreview: JSON.stringify(st) } });
    assert.deepEqual(j(p2.S), j(S));
  }
});

test("the advisor reads the rake keys the page sends", () => {
  const src = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "advisor.py"), "utf8");
  const body = src.slice(src.indexOf("def _advise("), src.indexOf("def _preflop("));
  for (const k of ["rake", "rake_cap"]) assert.ok(body.includes(`state.get("${k}")`), k);
});
