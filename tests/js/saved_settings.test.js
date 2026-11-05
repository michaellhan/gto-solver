// Removing a control removes the input, not the setting. A hand keeps its settings and payload() keeps sending them
// (payload_golden.test.js pins that); a setting with no control shows in a one-line banner and isn't carried to the next
// hand, so New hand is its off switch; opening a saved hand changes nothing in it; a saved "pool" is refused, because it's
// in the browser, not the hand. Lizard skips a hand with payouts or bounties on import (classic.ts), so a hand that lost
// them in a load would turn up there as a plain hand: these tests keep the record as it was.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage, state, acts } = require("./page.js");

const STATIC = path.join(__dirname, "..", "..", "gto", "static");
const MH = require(path.join(STATIC, "hands.js"));
const j = (x) => JSON.parse(JSON.stringify(x));
const flush = async () => { for (let k = 0; k < 20; k++) await new Promise(r => setImmediate(r)); };
async function until(f, n = 400) { for (let i = 0; i < n; i++) { const v = await f(); if (v) return v; await new Promise(r => setImmediate(r)); } throw new Error("timed out"); }

const SPOT = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") };
const played = (o = {}) => ({ ...state(SPOT), ...o });
const LOCK = { fold: -20, bluff: -50, raise: -50 };
const TOURNEY = { payouts: [50, 30, 20], fieldN: 40, fieldAvg: 55 };
const BOUNTY = { bounty: { fmt: "pko", all: 10, seats: { BB: 25 }, field: 12 } };
const ADV = { sizes: { flop: [33, 75] }, ranges: { BB: "22+,A2s+" }, optSizes: true };
const CARRIED = { stack: 80, stacks: { BB: 60 }, ante: 0.25, straddle: true, rake: 5, rakeCap: 3, stakes: { sb: 2, bb: 5 } };
const NOT_CARRIED = ["payouts", "fieldN", "fieldAvg", "bounty", "sizes", "ranges", "optSizes", "locks"];
const ALL = { ...CARRIED, ...TOURNEY, ...BOUNTY, ...ADV, locks: { BB: LOCK } };
// Lizard's rule for a classic hand, copied (lizard web/src/features/analyze/classic.ts, classicHand)
const lizardSkips = (st) => Number(st.ante) > 0 || !!st.straddle || !!st.payouts || !!st.bounty;

async function storePage(opts = {}) {
  const p = loadPage({ store: true, ...opts });
  await p.ready();
  await p.run("GTOStore.ready");
  p.hands = async () => j(await p.run("GTOStore.hands.list()"));
  return p;
}
const saved = async (p, st) => {   // a hand in My hands, as New hand saves it, with the table a new hand
  p.S = st;
  p.click("newhand");
  return (await until(async () => { const h = await p.hands(); return h.length && h; }))[0];
};
const openRecord = async (p, id) => {
  vm.runInContext(fs.readFileSync(path.join(STATIC, "hands.js"), "utf8"), p.ctx, { filename: "hands.js" });
  await p.run(`(async () => MyHands.pageHost().open(await GTOStore.hands.get(${JSON.stringify(id)})))()`);
  await flush();
};
const banner = (p) => p.byId("savedSettings");

test("the banner names each setting with no control, and points new spots to Lizard, not this hand", async () => {
  const p = loadPage();
  await p.ready();
  p.S = played();
  assert.equal(banner(p).hidden, true);
  assert.equal(banner(p).innerHTML, "");
  p.S = played(TOURNEY);
  assert.equal(banner(p).hidden, false);
  assert.match(banner(p).innerHTML, /^Saved with this hand: tournament, payouts 50\/30\/20 \(\+40 at other tables\)\. New hand clears it\./);
  assert.match(banner(p).innerHTML, /<a href="\/lizard\/tournaments">/);
  assert.doesNotMatch(banner(p).innerHTML, /\/lizard\/solve/);
  p.S = played(BOUNTY);
  assert.match(banner(p).innerHTML, /PKO bounties 10 \(BB 25\)/);
  assert.match(banner(p).innerHTML, /<a href="\/lizard\/tournaments">/);
  for (const [o, word] of [[{ sizes: ADV.sizes }, /custom sizes/], [{ ranges: ADV.ranges }, /custom BB range/], [{ optSizes: true }, /optimized sizes/]]) {
    p.S = played(o);
    assert.match(banner(p).innerHTML, word);
    assert.match(banner(p).innerHTML, /<a href="\/lizard\/solve">/);
    assert.doesNotMatch(banner(p).innerHTML, /\/lizard\/tournaments/);
  }
  p.S = played({ locks: { BB: LOCK } });   // read-only, and it has nowhere to point
  assert.match(banner(p).innerHTML, /^Saved with this hand: BB locked \(calling station\)\. New hand clears it\.$/);
  p.S = played({ ...TOURNEY, ...ADV, locks: { BB: LOCK } });
  assert.match(banner(p).innerHTML, /New hand clears them\. For a new spot, use <a href="\/lizard\/tournaments">Lizard's tournaments<\/a> or <a href="\/lizard\/solve">Lizard's solver<\/a>\.$/);
  assert.equal(banner(p).innerHTML.includes("\n"), false);
});

test("the banner shows what is sent: mid-hand too, and not what has no effect", async () => {
  const p = loadPage();
  await p.ready();
  p.S = played({ locks: { BB: LOCK } });
  assert.match(banner(p).innerHTML, /BB locked/);
  for (const o of [{ locks: { BTN: LOCK } }, { ranges: { BTN: "AA" } }, { payouts: [100] }, { bounty: { fmt: "ko" } }]) {
    p.S = played(o);
    assert.equal(banner(p).hidden, true, JSON.stringify(o));
  }
  for (const o of [CARRIED, { stakes: { sb: 2, bb: 5 } }, { tournament: true }]) {   // (the rest has a control, or its own label)
    p.S = played(o);
    assert.equal(banner(p).hidden, true, JSON.stringify(o));
  }
});

test("the banner stays through the review", async () => {
  const p = loadPage();
  await p.ready();
  p.spot(played(TOURNEY));
  await flush();
  assert.equal(p.run("!!stepView"), true);
  assert.match(banner(p).innerHTML, /payouts 50\/30\/20/);
});

test("New hand and a new table size drop what has no control and keep what has one", async () => {
  const p = loadPage();
  await p.ready();
  const check = (want) => {
    const s = p.S;
    for (const k of NOT_CARRIED) assert.equal(k in s, false, k);
    assert.deepEqual(j({ stack: s.stack, ante: s.ante, straddle: s.straddle, rake: s.rake, rakeCap: s.rakeCap, stakes: s.stakes }), want);
    assert.equal(banner(p).hidden, true);
  };
  const want = { stack: 80, ante: 0.25, straddle: true, rake: 5, rakeCap: 3, stakes: { sb: 2, bb: 5 } };
  p.S = played(ALL);
  p.click("newhand");
  check(want);
  assert.deepEqual(j(p.S.stacks), { BB: 60 });
  assert.deepEqual(p.S.cards.filter(Boolean), []);
  p.S = played(ALL);
  p.run('document.getElementById("players").children[7].onclick()');   // 9 players
  assert.equal(p.S.players, 9);
  check(want);
  assert.deepEqual(j(p.S.stacks), { BB: 60 });
  assert.deepEqual(p.S.cards.slice(0, 2), ["Ah", "Kh"]);               // (a new table size keeps the cards, as it did)
});

test("changing your seat in a hand keeps its settings: it's the same hand", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state({ cards: ["Ah", "Kh"] }), ...TOURNEY };
  p.run('document.getElementById("seats").children[0].onclick()');
  assert.equal(p.S.hero, "UTG");
  assert.deepEqual(j(p.S.payouts), TOURNEY.payouts);
});

test("each setting survives the page's repair of a hand, with the hand's signature unchanged", async () => {
  const p = await storePage();
  for (const o of [{ stacks: { BB: 60 } }, { ante: 0.25 }, { straddle: true }, { rake: 5, rakeCap: 3 }, { stakes: { sb: 2, bb: 5 } }, { payouts: [50, 30, 20] }, TOURNEY, BOUNTY,
    { sizes: ADV.sizes }, { ranges: ADV.ranges }, { optSizes: true }, { locks: { BB: LOCK } }, ALL]) {
    p.ctx.__tmp = JSON.stringify(played(o));
    assert.equal(p.val("GTOStore.hands.sig(cleanState(JSON.parse(__tmp)))"), p.val("GTOStore.hands.sig(JSON.parse(__tmp))"), JSON.stringify(o));
  }
});

test("a hand opened from My hands isn't modified: the record, its signature and its time stay; New hand then has none of it", async () => {
  for (const o of [TOURNEY, BOUNTY, ADV, { locks: { BB: LOCK } }, ALL]) {
    const p = await storePage();
    const rec = await saved(p, played(o));
    for (const k of Object.keys(o)) assert.deepEqual(rec.data.state[k], o[k], k);
    await openRecord(p, rec.id);
    await flush();
    assert.deepEqual((await p.hands())[0], rec);
    assert.match(banner(p).innerHTML, /^Saved with this hand: /);
    p.click("newhand");
    await flush();
    for (const k of NOT_CARRIED) assert.equal(k in p.S, false, k);
    const after = await p.hands();
    assert.equal(after.length, 1);
    assert.deepEqual(after[0].data.state, rec.data.state);
    assert.equal(after[0].data.sig, rec.data.sig);
  }
});

test("a link to a hand you own with payouts: the banner says so, New hand leaves one record, payouts kept, and Lizard still skips it", async () => {
  const seed = await storePage();
  const rec = await saved(seed, played(TOURNEY));
  const hash = "#h=" + btoa(JSON.stringify(rec.data.state));
  const p = await storePage({ hash, indexedDB: seed.ctx.indexedDB, storage: Object.fromEntries(seed.storage) });
  await flush();
  assert.deepEqual(j(p.S.payouts), TOURNEY.payouts);
  assert.match(banner(p).innerHTML, /payouts 50\/30\/20/);
  p.click("newhand");
  await flush();
  assert.equal("payouts" in p.S, false);
  const hs = await p.hands();
  assert.equal(hs.length, 1);
  assert.deepEqual(hs[0].data.state.payouts, TOURNEY.payouts);
  assert.equal(lizardSkips(hs[0].data.state), true);
  assert.equal(lizardSkips(p.S), false);   // (the next hand isn't skipped for a setting it never had)
});

test("the hand on the table at the upgrade keeps its settings over a reload, then New hand leaves one record", async () => {
  const seed = await storePage();
  const rec = await saved(seed, played(ALL));
  const storage = Object.fromEntries(seed.storage);
  storage.handreview = JSON.stringify(rec.data.state);
  storage["gto-hands-table"] = JSON.stringify({ id: rec.id, sig: rec.data.sig, spot: false, source: "table" });
  const p = await storePage({ indexedDB: seed.ctx.indexedDB, storage });
  await flush();
  assert.equal(p.S.bounty.fmt, "pko");
  assert.deepEqual((await p.hands())[0], rec);                     // (loading wrote no record)
  assert.equal(JSON.parse(p.storage.get("handreview")).payouts.length, 3);
  assert.doesNotMatch(p.byId("toast").innerHTML, /carry/);
  p.click("newhand");
  await flush();
  const hs = await p.hands();
  assert.equal(hs.length, 1);
  assert.deepEqual(hs[0].data.state, rec.data.state);
});

test("a stale gto-hands-table on load writes no second record", async () => {
  const seed = await storePage();
  const rec = await saved(seed, played(TOURNEY));
  const storage = { ...Object.fromEntries(seed.storage), handreview: JSON.stringify(rec.data.state), "gto-hands-table": JSON.stringify({ id: "gone", sig: "stale", spot: false }) };
  const p = await storePage({ indexedDB: seed.ctx.indexedDB, storage });
  await flush();
  assert.equal((await p.hands()).length, 1);
  p.click("newhand");
  await flush();
  assert.equal((await p.hands()).length, 1);
});

test("an exported file's hand comes back as the same hand: merged, not added", async () => {
  const p = await storePage();
  const rec = await saved(p, played(ALL));
  p.ctx.__file = JSON.stringify([{ state: rec.data.state, source: "file" }]);
  const r = await p.run("GTOStore.hands.importFile(JSON.parse(__file), cleanState)");
  assert.deepEqual(j(r), { added: 0, merged: 1, skipped: 0 });
  assert.deepEqual((await p.hands()).map(h => h.id), [rec.id]);
  assert.deepEqual((await p.hands())[0].data.state, rec.data.state);
});

const GRADE = { cls: "good", ev: 1.5 };
const withGrade = (st) => { const x = j(st); x.actions.preflop[3].g = GRADE; return x; };
const fileOf = (st, o = {}) => JSON.stringify({ format: "gto-solver-hands", version: 1, hands: [{ state: st, source: "file", tags: [], note: "", ...o }] });

test("an older export over a hand reviewed since keeps its grades: only its tags and note come in", async () => {
  const p = await storePage();
  const rec = await saved(p, played());
  await p.run(`GTOStore.hands.update(${JSON.stringify(rec.id)}, { tags: ["mine"], state: ${JSON.stringify(withGrade(rec.data.state))} })`);
  const before = (await p.hands())[0];
  assert.deepEqual(before.data.state.actions.preflop[3].g, GRADE);
  assert.equal(before.data.reviewed, true);
  p.ctx.__file = fileOf(rec.data.state);
  assert.deepEqual(j(await p.run("GTOStore.hands.importFile(JSON.parse(__file), cleanState)")), { added: 0, merged: 1, skipped: 0 });
  assert.deepEqual((await p.hands())[0], before);                  // (nothing to take: the record isn't written)
  p.ctx.__file = fileOf(rec.data.state, { tags: ["theirs"], note: "from the file" });
  assert.deepEqual(j(await p.run("GTOStore.hands.importFile(JSON.parse(__file), cleanState)")), { added: 0, merged: 1, skipped: 0 });
  const after = (await p.hands())[0];
  assert.deepEqual(after.data.state, before.data.state);
  assert.equal(after.data.reviewed, true);
  assert.deepEqual(after.data.tags, ["mine", "theirs"]);
  assert.equal(after.data.note, "from the file");
  p.ctx.__file = fileOf(rec.data.state, { note: "another" });
  await p.run("GTOStore.hands.importFile(JSON.parse(__file), cleanState)");
  assert.equal((await p.hands())[0].data.note, "from the file");   // (a note here stays)
});

test("an export with grades fills in a hand here that has none; a hand here with grades keeps its own", async () => {
  const p = await storePage();
  const rec = await saved(p, played());
  p.ctx.__file = fileOf(withGrade(rec.data.state));
  assert.deepEqual(j(await p.run("GTOStore.hands.importFile(JSON.parse(__file), cleanState)")), { added: 0, merged: 1, skipped: 0 });
  const after = (await p.hands())[0];
  assert.deepEqual(after.data.state.actions.preflop[3].g, GRADE);
  assert.equal(after.data.reviewed, true);
  const other = withGrade(rec.data.state); other.actions.preflop[3].g = { cls: "bad", ev: -3 };
  p.ctx.__file = fileOf(other);
  await p.run("GTOStore.hands.importFile(JSON.parse(__file), cleanState)");
  assert.deepEqual((await p.hands())[0].data.state.actions.preflop[3].g, GRADE);
});

test("two entries of one hand in a file are one record", async () => {
  const p = await storePage();
  const st = j(played());
  p.ctx.__file = JSON.stringify({ format: "gto-solver-hands", version: 1, hands: [{ state: st, tags: ["a"] }, { state: st, tags: ["b"] }] });
  assert.deepEqual(j(await p.run("GTOStore.hands.importFile(JSON.parse(__file), cleanState)")), { added: 1, merged: 1, skipped: 0 });
  const hs = await p.hands();
  assert.equal(hs.length, 1);
  assert.deepEqual(hs[0].data.tags, ["a", "b"]);
});

// Records cleanState would change (older or foreign ones: the page writes none of these itself).
const REPAIRED = {
  "an action by a seat not at the table": (s) => { s.actions.flop.push({ pos: "LJ", action: "check" }); },
  "a card twice": (s) => { s.cards[3] = s.cards[0]; },
  "a rake out of range": (s) => { s.rake = 30; s.rakeCap = 3; },
  "field numbers with no payouts": (s) => { s.fieldN = 40; s.fieldAvg = 55; },
  "a lock of an unknown kind": (s) => { s.locks = { BB: { fold: -20, nope: 1 } }; },
  "a range for a seat not at the table": (s) => { s.ranges = { LJ: "AA" }; },
  "five cards in the list": (s) => { s.cards = s.cards.slice(0, 5); },
};
const putRecord = async (p, st, o = {}) => {
  p.ctx.__rec = JSON.stringify({ state: st, sig: p.val(`GTOStore.hands.sig(${JSON.stringify(st)})`), tags: ["t"], note: "n", source: "table", reviewed: false, ...o });
  await p.run('GTOStore.put("hands", "old", JSON.parse(__rec))');
  return (await p.hands())[0];
};
const repaired = (name) => { const s = j(played()); REPAIRED[name](s); return s; };

for (const name of Object.keys(REPAIRED))
  test(`opening a record the page would repair leaves it as it was: ${name}`, async () => {
    const p = await storePage();
    const rec = await putRecord(p, repaired(name));
    assert.notEqual(p.val("GTOStore.hands.sig(S)"), rec.data.sig);
    await openRecord(p, rec.id);
    assert.deepEqual((await p.hands())[0], rec);
    assert.notEqual(p.val("GTOStore.hands.sig(S)"), rec.data.sig, "the table has the cleaned copy");
    // over a reload too, and New hand leaves one record, not rewritten
    const q = await storePage({ storage: Object.fromEntries(p.storage), indexedDB: p.ctx.indexedDB });
    await flush();
    q.click("newhand");
    await flush();
    assert.deepEqual(j(await q.hands()), [rec]);
  });

test("a repaired record opened twice stays one untouched record", async () => {
  const p = await storePage();
  const rec = await putRecord(p, repaired("a card twice"));
  await openRecord(p, rec.id);
  await openRecord(p, rec.id);
  assert.deepEqual(await p.hands(), [rec]);
});

test("a repaired record played on from, or reviewed, is the hand you changed: its record takes it, and only then", async () => {
  const p = await storePage();
  const rec = await putRecord(p, repaired("a rake out of range"));
  await openRecord(p, rec.id);
  p.run('S.actions.flop.push({ pos: "BTN", action: "check" }); save();');
  p.click("newhand");
  await flush();
  let hs = await p.hands();
  assert.deepEqual(hs.map(h => h.id), ["old"]);
  assert.equal(hs[0].data.state.actions.flop.length, 2);
  assert.equal("rake" in hs[0].data.state, false);
  assert.deepEqual([hs[0].data.tags, hs[0].data.note], [["t"], "n"]);

  const q = await storePage();
  const r2 = await putRecord(q, repaired("field numbers with no payouts"));
  await openRecord(q, r2.id);
  q.run("S.actions.preflop[3].g = { cls: 'good', ev: 1 };");
  await q.run('keepHand("review", S)');
  await flush();
  hs = await q.hands();
  assert.deepEqual(hs.map(h => h.id), ["old"]);
  assert.equal(hs[0].data.reviewed, true);
  assert.equal(q.val("GTOStore.hands.table().opened === undefined"), true);
});

test("a hand marked finished only to open it isn't written back either", async () => {
  const p = await storePage();
  const rec = await putRecord(p, j(played()), { source: "review" });
  await openRecord(p, rec.id);
  assert.equal(p.S.finished, true);
  p.click("newhand");
  await flush();
  assert.deepEqual(await p.hands(), [rec]);
});

test("a clean record opened leaves no copy in gto-hands-table, and My hands doesn't offer to save a repaired one again", async () => {
  const p = await storePage();
  const rec = await saved(p, played());
  await openRecord(p, rec.id);
  assert.equal("opened" in JSON.parse(p.storage.get("gto-hands-table")), false);
  const q = await storePage();
  const r2 = await putRecord(q, repaired("a card twice"));
  await q.run('loadJs("hands.js?v=" + HN_VERSION, () => window.MyHands).then(() => MyHands.useHost(MyHands.pageHost()))');
  await q.run('MyHands.act({ savetable: "" })');                   // (the list read: an empty table saves nothing)
  await q.run('MyHands.pageHost().open(MyHands.recs[0])');
  await flush();
  assert.equal(q.run("MyHands.tableHtml()"), "");
  q.run('S.actions.flop.push({ pos: "BTN", action: "check" }); save();');
  assert.match(q.run("MyHands.tableHtml()"), /data-savetable/);
});

test("reviewing again sends the hand's settings: no stored grade decides what the numbers were against", async () => {
  const p = loadPage();
  await p.ready();
  p.spot(played({ ...TOURNEY, ...ADV, locks: { BB: LOCK } }));
  await flush();
  const st = p.posted.filter(m => m.state).at(-1).state;
  assert.deepEqual([st.payouts, st.sizes, st.ranges, st.optimize_sizes, st.locks], [TOURNEY.payouts, ADV.sizes, ADV.ranges, true, { BB: LOCK }]);
});

test("a hand not started when the page loads drops them, with one notice; no record is touched", async () => {
  const blank = { ...state({ cards: ["Ah", "Kh"] }), ...TOURNEY, ...ADV, ...CARRIED };
  const raw = JSON.stringify(blank);
  const p = await storePage({ storage: { handreview: raw } });
  await flush();
  assert.equal(p.storage.get("handreview"), raw);                  // (the frozen copy's: only what you enter here changes it)
  assert.equal(await p.run('GTOStore.ready.then(() => GTOStore.getRaw("handreview"))'), raw);
  for (const k of NOT_CARRIED) assert.equal(k in p.S, false, k);
  assert.equal(p.S.stack, 80);                                     // (what has a control stays)
  assert.equal(p.S.rake, 5);
  assert.deepEqual(p.S.cards.slice(0, 2), ["Ah", "Kh"]);
  assert.match(p.byId("toast").innerHTML, /Settings from your last hand no longer carry over/);
  assert.equal(p.byId("toast").hidden, false);
  assert.equal(banner(p).hidden, true);
  assert.deepEqual(await p.hands(), []);
  const again = await storePage({ storage: Object.fromEntries(p.storage), indexedDB: p.ctx.indexedDB });
  assert.equal(again.byId("toast").innerHTML, "");                 // (once, though handreview still holds them)
  assert.equal(again.storage.get("handreview"), raw);
  for (const k of NOT_CARRIED) assert.equal(k in again.S, false, k);
  again.run('document.getElementById("seats").children[0].onclick()');   // (the first thing entered here saves the hand without them)
  assert.equal("payouts" in JSON.parse(again.storage.get("handreview")), false);
  // one with nothing in effect goes without a word
  const quiet = await storePage({ storage: { handreview: JSON.stringify({ ...state({}), payouts: [100] }) } });
  assert.equal("payouts" in quiet.S, false);
  assert.equal(quiet.byId("toast").innerHTML, "");
});

test("a link's hand with no action yet isn't touched: only a hand from the last visit is", async () => {
  const st = { ...state({ cards: ["Ah", "Kh"] }), ...TOURNEY };
  const p = loadPage({ hash: "#h=" + btoa(JSON.stringify(st)) });
  await p.ready(); await flush();
  assert.deepEqual(j(p.S.payouts), TOURNEY.payouts);
  assert.match(banner(p).innerHTML, /payouts/);
});

for (const serviceWorker of [false, true])
  test(`a link's hand with no action yet keeps its settings over a reload${serviceWorker ? ": the service worker's, on a first visit" : ""}; New hand then drops them`, async () => {
    const st = { ...state({ cards: ["Ah", "Kh"] }), ...TOURNEY, ...ADV, locks: { BB: LOCK } };
    const p = await storePage({ hash: "#h=" + btoa(JSON.stringify(st)), serviceWorker });
    await flush();
    assert.equal(p.run("isolating"), serviceWorker);
    assert.deepEqual(j(p.S.payouts), TOURNEY.payouts);
    const q = await storePage({ storage: Object.fromEntries(p.storage), indexedDB: p.ctx.indexedDB });
    await flush();
    for (const k of NOT_CARRIED.filter(k => k !== "bounty")) assert.equal(k in q.S, true, k);
    assert.deepEqual(j(JSON.parse(q.storage.get("handreview")).payouts), TOURNEY.payouts);
    assert.match(banner(q).innerHTML, /payouts 50\/30\/20/);
    assert.doesNotMatch(q.byId("toast").innerHTML, /carry/);
    q.click("newhand");
    await flush();
    for (const k of NOT_CARRIED) assert.equal(k in q.S, false, k);
    const r = await storePage({ storage: { ...Object.fromEntries(q.storage), handreview: JSON.stringify({ ...q.S, ...TOURNEY }) }, indexedDB: q.ctx.indexedDB });
    await flush();
    assert.equal("payouts" in r.S, false);                         // (no longer the link's hand: the rule for a blank hand again)
  });

for (const store of [false, true])
  test(`a saved "pool" is refused: not sent, not rewritten, said once (${store ? "with the store" : "localStorage"})`, async () => {
    const mk = async (storage, o = {}) => { const p = store ? await storePage({ storage, ...o }) : loadPage({ storage }); if (!store) await p.ready(); await flush(); return p; };
    const p = await mk({ "gto-opponents": "pool" });
    p.spot(played());
    await flush();
    assert.equal(p.payload().opponents, undefined);
    assert.equal(p.posted.filter(m => m.state).at(-1).state.opponents, undefined);
    assert.match(p.byId("toast").innerHTML, /no longer solves against a player pool/);
    assert.equal(p.storage.get("gto-opponents"), "pool");
    if (store) assert.equal(await p.run('GTOStore.ready.then(() => GTOStore.getRaw("gto-opponents"))'), "pool");
    const q = await mk(Object.fromEntries(p.storage), store ? { indexedDB: p.ctx.indexedDB } : {});
    assert.equal(q.byId("toast").innerHTML, "");                   // (once)
    assert.equal(q.storage.get("gto-opponents"), "pool");
    assert.equal(q.payload().opponents, undefined);
  });

test("no notice for a saved gto, and none while the page is about to reload for its service worker", async () => {
  const p = loadPage({ storage: { "gto-opponents": "gto" } });
  await p.ready();
  assert.equal(p.byId("toast").innerHTML, "");
  assert.equal(p.storage.has("gto-pool-refused"), false);
  const q = loadPage({ storage: { "gto-opponents": "pool" }, serviceWorker: true });
  await q.ready();
  assert.equal(q.run("isolating"), true);
  assert.equal(q.storage.has("gto-pool-refused"), false);          // (kept for the load after it)
});

test("the setup line says cash, except for a tournament; My hands marks these hands", async () => {
  const p = loadPage();
  await p.ready();
  p.S = played();
  assert.match(p.byId("setupSum").innerHTML, /<b>6-handed<\/b> <span class="muted">·<\/span> cash/);
  p.S = played(TOURNEY);
  assert.doesNotMatch(p.byId("setupSum").innerHTML, /cash/);
  p.S = played({ tournament: true });
  assert.doesNotMatch(p.byId("setupSum").innerHTML, /cash/);
  const game = (o) => MH.summary(played(o)).game;
  assert.equal(game({}), "6-handed, 100bb");
  assert.equal(game({ payouts: [50, 30] }), "6-handed, 100bb, tournament settings");
  assert.equal(game({ bounty: { fmt: "ko", all: 5 } }), "6-handed, 100bb, tournament settings");
  assert.equal(game({ locks: { BB: LOCK } }), "6-handed, 100bb, custom settings");
  assert.equal(game(ALL), "6-handed, 80bb, 0.25 ante, straddle, tournament settings, custom settings");
});
