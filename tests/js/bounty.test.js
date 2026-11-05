// What a saved hand may hold of bounty tournaments, which have no control now (S.bounty): the format (KO, PKO, mystery),
// every seat's bounty and the seats' own, the other tables' average, a mystery tournament's envelopes; kept in the state,
// sent to the solver as "bounties" (gto/advisor.py _bounties) once something is at stake (with the payouts, one prize too,
// and the field), carried by share links and saved with the hand.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage, state, acts } = require("./page.js");

const j = (x) => JSON.parse(JSON.stringify(x));
const PRE = { cards: ["Ah", "Jd"], preflop: acts("UTG f, HJ f, CO f, BTN r2") };
const SIX = ["UTG", "HJ", "CO", "BTN", "SB", "BB"];
const every = (v) => Object.fromEntries(SIX.map(p => [p, v]));

test("no bounties by default: nothing sent", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(PRE);
  assert.equal("bounties" in p.payload(), false);
});

test("a hand's PKO: the format alone sends nothing, a bounty sends every seat's, with the payouts and field", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state(PRE), bounty: { fmt: "pko" } };
  assert.equal("bounties" in p.payload(), false);   // (nothing at stake yet)
  p.S = { ...state(PRE), bounty: { fmt: "pko", all: 10 } };
  assert.deepEqual(j(p.payload().bounties), { format: "pko", seats: every(10) });
  assert.equal("payouts" in p.payload(), false);   // no payouts: the solver's chip EV rule
  assert.deepEqual(j(JSON.parse(p.storage.get("handreview")).bounty), { fmt: "pko", all: 10 });   // saved with the hand
  // one prize (winner-take-all) goes with the bounties; the field once its numbers are in, with its average bounty
  p.S = { ...state(PRE), bounty: { fmt: "pko", all: 10, field: 7.5 }, payouts: [500], fieldN: 12, fieldAvg: 30 };
  const pl = p.payload();
  assert.deepEqual(j(pl.payouts), [500]);
  assert.deepEqual(j(pl.field), { players: 12, chips: 360 });
  assert.deepEqual(j(pl.bounties), { format: "pko", seats: every(10), field: 7.5 });
  // a seat's own bounty
  p.S = { ...state(PRE), bounty: { fmt: "pko", all: 10, seats: { SB: 20 } } };
  assert.equal(p.payload().bounties.seats.SB, 20);
  assert.equal(p.payload().bounties.seats.BB, 10);
});

test("mystery bounties: the average envelope, or the pool and the players left", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state(PRE), bounty: { fmt: "mystery", pool: 900 } };
  assert.equal("bounties" in p.payload(), false);   // (the pool needs the players left)
  p.S = { ...state(PRE), bounty: { fmt: "mystery", pool: 900, left: 30 } };
  assert.deepEqual(j(p.payload().bounties), { format: "mystery", pool: 900, left: 30 });
  p.S = { ...state(PRE), bounty: { fmt: "mystery", env: 25, pool: 900, left: 30 } };
  assert.deepEqual(j(p.payload().bounties), { format: "mystery", envelope: 25, pool: 900, left: 30 });
});

test("bounties don't carry to the next hand", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state(PRE), bounty: { fmt: "ko", all: 5, seats: { SB: 12, UTG: 1 }, field: 6 } };
  p.run('document.getElementById("players").children[3].onclick()');   // 5 players
  assert.equal(p.S.players, 5);
  assert.equal(p.S.bounty, undefined);
  p.S = { ...state(PRE), bounty: { fmt: "ko", all: 5, seats: { SB: 12, UTG: 1 }, field: 6 } };
  p.byId("newhand").onclick();
  assert.equal(p.S.bounty, undefined);
});

test("share link round trip with bounties", async () => {
  const B = { ...state(PRE), bounty: { fmt: "pko", all: 10, seats: { SB: 20 } } };
  for (const st of [B, { ...B, payouts: [50, 30, 20] }, { ...state(PRE), bounty: { fmt: "mystery", env: 25 } }]) {
    const a = loadPage();
    a.S = st;
        const b = loadPage({ hash: a.run("shareLink()").replace(/^[^#]*/, "") });
    assert.deepEqual(j(b.S), j(st));
    await a.ready(); await b.ready();
    assert.deepEqual(b.payload(), a.payload());
    assert.ok(b.payload().bounties);
  }
});

test("bad bounty values in a link or storage are dropped", () => {
  for (const [bounty, want] of [
    [{ fmt: "pko", all: "10", seats: { SB: 20, XX: 5, BB: -1, CO: "x", BTN: 10 } }, { fmt: "pko", all: 10, seats: { SB: 20 } }],
    [{ fmt: "bogus", all: 10 }, undefined], ["pko", undefined], [null, undefined],
    [{ fmt: "ko", all: -5, field: "x" }, { fmt: "ko" }],
    [{ fmt: "mystery", env: "abc", pool: 900, left: 30.7, all: 3 }, { fmt: "mystery", pool: 900, left: 30 }]]) {
    const st = { ...state(PRE), bounty };
    const p = loadPage({ hash: "#h=" + Buffer.from(JSON.stringify(st)).toString("base64") });
    assert.deepEqual(j(p.S).bounty, want, JSON.stringify(bounty));
    const p2 = loadPage({ storage: { handreview: JSON.stringify(st) } });
    assert.deepEqual(j(p2.S), j(p.S));
  }
});

test("the advisor reads the bounty keys the page sends", () => {
  const src = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "advisor.py"), "utf8");
  assert.ok(src.includes('state.get("bounties")'));
  const body = src.slice(src.indexOf("def _bounties("));
  for (const k of ['"format"', '"seats"', '"field"', '"envelope"', '"pool"', '"left"']) assert.ok(body.includes(k), k);
});
