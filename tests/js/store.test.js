// The site's storage layer (gto/static/store.js) over an in-memory IndexedDB (idb_shim.js): the first load's migration
// of what the site kept before (localStorage keys as the page and its views wrote them, the analyzer's IndexedDB
// "gto-session" and its localStorage fallback), the localStorage mirror kept in step (a key written or removed behind
// the store's back, a mirror lost whole, a full localStorage), the records' envelope (id, schema, created / updated,
// tombstones), the fallback backends, the Anthropic API key kept out of every record and backup, backup and restore
// (merging, never deleting), and My hands' records (the same hand once, a hand played on, imports and exports).
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { createStore } = require(path.join(__dirname, "..", "..", "gto", "static", "store.js"));
const { fakeIndexedDB, seedOldDb, fakeLocalStorage } = require("./idb_shim.js");

const KEY = "sk-ant-api03-Zx9wQ_secretSECRETsecret-0123456789";
// a clock the test moves
function clock(t = 1_700_000_000_000) { const c = () => c.t; c.t = t; c.tick = (d = 1000) => (c.t += d); return c; }
const open = async (env) => { const st = createStore({ broadcast: false, events: false, ...env }); await st.ready; return st; };
// a hand as the page keeps it (S)
function hand({ hero = "BTN", cards = ["Ah", "Kd"], preflop, flop = [], turn = [], river = [], ...rest } = {}) {
  const c = [...cards]; while (c.length < 7) c.push(null);
  return { players: 6, hero, stack: 100, cards: c, ...rest,
           actions: { preflop: preflop || [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "fold" }, { pos: "CO", action: "fold" }, { pos: "BTN", action: "raise", to: 2.5 }, { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }], flop, turn, river } };
}
// localStorage as a browser that has used the site keeps it (each key in the shape its code writes)
const OLD_LS = () => ({
  handreview: JSON.stringify(hand({ cards: ["Ah", "Kd", "Qs", "7h", "2c"], flop: [{ pos: "BB", action: "check" }] })),
  fourcolor: "1", "gto-speed": "accurate", "gto-opponents": "pool", "gto-sound": "0", "gto-layout": JSON.stringify({ cols: [320, 1, 280] }),
  "gto-practice": "1", "gto-practice-score": JSON.stringify({ hands: 41, decisions: 97, ok: 80, lost: 6.25 }),
  "gto-practice-review": JSON.stringify([{ key: "BTN|AKo|flop", n: 2 }]), "gto-practice-seat": "CO",
  "gto-drills": JSON.stringify({ spot: "btn_bb", view: "vsbet", filter: {}, hist: { btn_bb: [1, 0, 1] } }),
  "gto-study": JSON.stringify({ spot: "srp_btn_bb", speed: "fast", view: "strategy" }),
  "gto-reports": JSON.stringify({ spot: "srp_btn_bb", sort: "ev" }), "gto-mt-score": JSON.stringify({ hands: 3 }),
  "gto-some-new-thing": "x",                         // (a key added by a later version: taken in too)
  "gto-anthropic-key": KEY, "gto-explain-own": "1",  // (never records)
  "other-site": "not ours",
});
const SESSION = { v: 1, created: 1699990000000, name: "Tuesday", scope: "all", status: "paused", items: [{ id: 0, hand: 0, s: "flop", i: 1 }], hands: [{ id: "h1", text: "PokerStars Hand #1: ..." }] };
const HISTORY = [{ id: 1699990000000, fp: "abc", name: "Tuesday", date: 1699990000000, hands: 120, decisions: 300 },
                 { id: 1699900000000, fp: "def", name: "Monday", date: 1699900000000, hands: 80, decisions: 210 }];

test("first load: every localStorage key and the analyzer's database are taken in, nothing lost or changed", async () => {
  const init = OLD_LS(), ls = fakeLocalStorage(init), idb = fakeIndexedDB(), now = clock();
  await seedOldDb(idb, "gto-session", "kv", { current: SESSION, history: HISTORY });
  const st = await open({ localStorage: ls, indexedDB: idb, now });
  assert.equal(st.backend, "indexeddb");
  const kv = Object.fromEntries((await st.list("kv")).map(r => [r.id, r]));
  for (const [k, v] of Object.entries(init)) {
    if (k === "gto-anthropic-key" || k === "gto-explain-own" || k === "other-site") assert.equal(kv[k], undefined, k);
    else assert.equal(kv[k].data.raw, v, k);
    assert.equal(ls.getItem(k), v, `localStorage keeps ${k}`);   // (the mirror: left where it was)
  }
  const r = kv.handreview;
  assert.deepEqual(Object.keys(r).sort(), ["created", "data", "dev", "id", "updated", "v"]);
  assert.equal(r.v, st.SCHEMA); assert.equal(r.created, now.t); assert.equal(r.updated, now.t); assert.equal(r.dev, st.device);
  // the analyzer: its session in progress a blob, its history a record per session (ids as strings), newest first
  assert.deepEqual(await st.blob("session-current").get(), SESSION);
  assert.deepEqual(await st.sessions.get(), HISTORY);
  assert.deepEqual((await st.list("sessions")).map(x => x.id).sort(), ["1699900000000", "1699990000000"]);
  // the old database is left as it was
  assert.deepEqual(idb.dump("gto-session", "kv"), { current: SESSION, history: HISTORY });
  // and the page's reads give what they gave before
  assert.equal(st.getRaw("gto-speed"), "accurate");
  assert.deepEqual(st.get("gto-practice-score"), JSON.parse(init["gto-practice-score"]));
  assert.equal(st.getRaw("gto-anthropic-key"), KEY);   // (the key: still where Practice reads it)
});

test("the migration runs once: a session deleted after it isn't taken in again", async () => {
  const ls = fakeLocalStorage(), idb = fakeIndexedDB();
  await seedOldDb(idb, "gto-session", "kv", { current: SESSION, history: HISTORY });
  const a = await open({ localStorage: ls, indexedDB: idb });
  await a.sessions.set([HISTORY[0]]);
  await a.blob("session-current").del();
  const b = await open({ localStorage: ls, indexedDB: idb });
  assert.deepEqual(await b.sessions.get(), [HISTORY[0]]);
  assert.equal(await b.blob("session-current").get(), null);
  // the deleted one is a tombstone (a later sync can tell it from one never seen)
  const dead = (await b.list("sessions", { deleted: true })).find(r => r.id === "1699900000000");
  assert.equal(dead.deleted, true); assert.equal(dead.data, undefined);
  assert.equal(b.device, a.device);   // (this browser's id, kept)
});

test("the analyzer's localStorage fallback is taken in, and no empty database is made for it", async () => {
  const ls = fakeLocalStorage({ "gto-session": JSON.stringify(SESSION), "gto-session-history": JSON.stringify(HISTORY) }), idb = fakeIndexedDB();
  const st = await open({ localStorage: ls, indexedDB: idb });
  assert.deepEqual(await st.blob("session-current").get(), SESSION);
  assert.equal((await st.sessions.get()).length, 2);
  assert.equal(idb.has("gto-session"), false);
  assert.equal(st.getRaw("gto-session") != null, true);          // (left as it was)
  assert.equal((await st.list("kv")).some(r => r.id.startsWith("gto-session")), false);   // (not a kv record too)
});

test("kv: the mirror at once, the record just after; created stays, updated moves; a removed key is a tombstone", async () => {
  const ls = fakeLocalStorage(), idb = fakeIndexedDB(), now = clock();
  const st = await open({ localStorage: ls, indexedDB: idb, now });
  st.set("gto-drills", { spot: "a" });
  assert.equal(ls.getItem("gto-drills"), '{"spot":"a"}');
  assert.deepEqual(st.get("gto-drills"), { spot: "a" });
  await st.flush();
  const t0 = now.t;
  assert.equal((await st.record("kv", "gto-drills")).created, t0);
  now.tick(5000); st.set("gto-drills", { spot: "b" }); await st.flush();
  let r = await st.record("kv", "gto-drills");
  assert.equal(r.created, t0); assert.equal(r.updated, t0 + 5000); assert.deepEqual(JSON.parse(r.data.raw), { spot: "b" });
  now.tick(); st.set("gto-drills", { spot: "b" }); await st.flush();   // (the same value: no new write)
  assert.equal((await st.record("kv", "gto-drills")).updated, t0 + 5000);
  st.remove("gto-drills"); await st.flush();
  assert.equal(ls.getItem("gto-drills"), null); assert.equal(st.get("gto-drills"), null);
  r = await st.record("kv", "gto-drills");
  assert.equal(r.deleted, true); assert.equal(r.data, undefined); assert.equal(r.created, t0);
  // and in the database itself
  assert.equal(idb.dump("gto-store", "kv")["gto-drills"].deleted, true);
  // the key is never a record, set through the store or not
  st.setRaw("gto-anthropic-key", KEY); await st.flush();
  assert.equal(ls.getItem("gto-anthropic-key"), KEY);
  assert.equal(await st.record("kv", "gto-anthropic-key"), null);
});

test("reconcile: a key written or removed behind the store's back, and a mirror lost whole", async () => {
  const ls = fakeLocalStorage({ "gto-study": '{"spot":"x"}', "gto-speed": "fast" }), idb = fakeIndexedDB();
  await open({ localStorage: ls, indexedDB: idb });
  ls.setItem("gto-study", '{"spot":"y"}');     // (code that doesn't use the store)
  ls.removeItem("gto-speed");
  ls.setItem("gto-practice-hint", "1");
  let st = await open({ localStorage: ls, indexedDB: idb });
  assert.equal(JSON.parse((await st.record("kv", "gto-study")).data.raw).spot, "y");
  assert.equal((await st.record("kv", "gto-speed")).deleted, true);
  assert.equal((await st.record("kv", "gto-practice-hint")).data.raw, "1");
  // site data cleared but not the database (the sentinel gone with the rest): written back, nothing deleted
  ls.clear();
  st = await open({ localStorage: ls, indexedDB: idb });
  assert.equal(ls.getItem("gto-study"), '{"spot":"y"}');
  assert.equal(ls.getItem("gto-practice-hint"), "1");
  assert.equal(ls.getItem("gto-speed"), null);
  assert.equal((await st.record("kv", "gto-study")).deleted, undefined);
  // before a backup too
  ls.setItem("gto-reports", '{"sort":"n"}');
  const b = await st.backup();
  assert.equal(b.stores.kv.find(r => r.id === "gto-reports").data.raw, '{"sort":"n"}');
});

test("a full localStorage: the record keeps the value, read back now and at the next load, mirrored once there's room", async () => {
  const ls = fakeLocalStorage({ handreview: JSON.stringify(hand()) }), idb = fakeIndexedDB();
  let st = await open({ localStorage: ls, indexedDB: idb });
  const big = JSON.stringify(hand({ cards: ["Qs", "Qd", "9c", "8c", "2h"], flop: [{ pos: "BB", action: "check" }] }));
  ls.full = true;
  assert.doesNotThrow(() => st.setRaw("handreview", big));
  assert.equal(ls.getItem("handreview"), null);    // (not the older hand left in its place)
  assert.equal(st.getRaw("handreview"), big);
  await st.flush();
  assert.equal(st.getRaw("handreview"), big);
  st = await open({ localStorage: ls, indexedDB: idb });   // (still full: not taken for a deletion)
  assert.equal((await st.record("kv", "handreview")).data.raw, big);
  assert.equal(st.getRaw("handreview"), big);
  ls.full = false;
  st = await open({ localStorage: ls, indexedDB: idb });
  assert.equal(ls.getItem("handreview"), big);
  assert.equal((await st.record("kv", "handreview")).nomirror, undefined);
});

test("without IndexedDB the records are kept in localStorage, without either in memory", async () => {
  for (const idb of [null, fakeIndexedDB({ failOpen: true })]) {
    const ls = fakeLocalStorage({ "gto-speed": "fast", "gto-session-history": JSON.stringify(HISTORY) });
    let st = await open({ localStorage: ls, indexedDB: idb });
    assert.equal(st.backend, "localStorage");
    await st.hands.save(hand());
    st = await open({ localStorage: ls, indexedDB: idb });
    assert.equal((await st.hands.list()).length, 1);
    assert.equal((await st.sessions.get()).length, 2);
    assert.equal(st.getRaw("gto-speed"), "fast");
    assert.ok(JSON.parse(ls.getItem("gto-store:hands")));
  }
  const st = await open({ localStorage: null, indexedDB: null });
  assert.equal(st.backend, "memory");
  st.set("gto-study", { a: 1 });
  assert.deepEqual(st.get("gto-study"), { a: 1 });
  await st.flush();
  assert.deepEqual(st.get("gto-study"), { a: 1 });
  st.remove("gto-study");
  assert.equal(st.get("gto-study"), null);
});

test("a backup holds everything but the API key: not as a record, not inside a note or a setting", async () => {
  const init = OLD_LS(), ls = fakeLocalStorage(init), idb = fakeIndexedDB();
  await seedOldDb(idb, "gto-session", "kv", { current: SESSION, history: HISTORY });
  const st = await open({ localStorage: ls, indexedDB: idb });
  const h = await st.hands.save(hand(), { note: `my key is ${KEY} oops` });
  st.set("gto-study", { custom: "key " + KEY }); await st.flush();
  const b = await st.backup();
  const text = JSON.stringify(b);
  assert.equal(text.includes(KEY), false);
  assert.equal(text.includes("gto-anthropic-key"), false);
  assert.equal(text.includes("gto-explain-own"), false);
  assert.equal(text.includes("other-site"), false);
  assert.equal(b.format, "gto-solver-backup"); assert.equal(b.schema, 1); assert.equal(b.device, st.device);
  assert.deepEqual(b.counts, { kv: b.stores.kv.length, hands: 1, sessions: 2, blobs: 1 });
  assert.equal(b.stores.hands[0].data.note, "my key is [removed] oops");
  assert.ok(b.stores.kv.length >= 14);
  // the store itself is unchanged (only the file is scrubbed)
  assert.equal((await st.hands.get(h.id)).data.note.includes(KEY), true);
  assert.equal(ls.getItem("gto-anthropic-key"), KEY);
});

test("restore: into a new browser everything comes back; here the newer stays and nothing is deleted", async () => {
  const now = clock();
  const a = await open({ localStorage: fakeLocalStorage(OLD_LS()), indexedDB: fakeIndexedDB(), now });
  await a.hands.save(hand(), { tags: ["3bet"], note: "fold?" });
  await a.sessions.set(HISTORY);
  await a.blob("session-current").set(SESSION);
  const b = await a.backup();
  // a new browser: everything added, the settings in its localStorage
  const ls2 = fakeLocalStorage(), c = await open({ localStorage: ls2, indexedDB: fakeIndexedDB(), now });
  const r = await c.restore(JSON.parse(JSON.stringify(b)));
  assert.equal(r.added, b.counts.kv + 1 + 2 + 1); assert.equal(r.updated, 0);
  assert.equal(ls2.getItem("gto-speed"), "accurate");
  assert.equal(ls2.getItem("gto-anthropic-key"), null);
  assert.deepEqual((await c.hands.list())[0].data.tags, ["3bet"]);
  assert.deepEqual(await c.sessions.get(), HISTORY);
  assert.deepEqual(await c.blob("session-current").get(), SESSION);
  // the same backup again: all kept
  assert.equal((await c.restore(b)).added, 0);
  // here: a newer setting stays, an older one is replaced, a hand deleted here comes back, one added here stays
  now.tick(); c.setRaw("gto-speed", "fast"); await c.flush();
  const old = { ...b, stores: { ...b.stores, kv: b.stores.kv.map(x => x.id === "gto-sound" ? { ...x, updated: now.t + 10, data: { raw: "1" } } : x) } };
  await c.hands.remove((await c.hands.list())[0].id);
  const mine = await c.hands.save(hand({ hero: "CO", cards: ["7s", "7d"] }));
  const r2 = await c.restore(old);
  assert.equal(ls2.getItem("gto-speed"), "fast");
  assert.equal(ls2.getItem("gto-sound"), "1");
  assert.equal(r2.stores.kv.updated, 1);
  assert.equal(r2.stores.hands.added, 1);
  const hs = await c.hands.list();
  assert.equal(hs.length, 2); assert.ok(hs.some(x => x.id === mine.id));
  // the same hand under another id (saved in both browsers): one record, its tags joined, its note kept or filled
  const d = await open({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB(), now });
  const here = await d.hands.save(hand(), { tags: ["mine"] });
  const r3 = await d.restore(b);
  const only = await d.hands.list();
  assert.equal(only.length, 1); assert.equal(only[0].id, here.id);
  assert.deepEqual(only[0].data.tags, ["mine", "3bet"]); assert.equal(only[0].data.note, "fold?");
  assert.equal(r3.stores.hands.updated, 1);
  await assert.rejects(d.restore({ format: "gto-solver-hands", hands: [] }), /backup/);
  await assert.rejects(d.restore(null), /backup/);
  // a backup can't bring the key in, nor a key that isn't the site's
  const evil = { ...b, stores: { kv: [{ id: "gto-anthropic-key", created: 1, updated: 1, data: { raw: "sk-x" } }, { id: "other", created: 1, updated: 1, data: { raw: "y" } }] } };
  const ls3 = fakeLocalStorage(), e = await open({ localStorage: ls3, indexedDB: fakeIndexedDB() });
  assert.equal((await e.restore(evil)).added, 0);
  assert.equal(ls3.getItem("gto-anthropic-key"), null); assert.equal(ls3.getItem("other"), null);
});

test("hands: the same hand is one record (grades aside); a hand played on updates its record; a new hand is new", async () => {
  const st = await open({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB() });
  const H = st.hands, a = hand();
  assert.equal(await H.save(hand({ preflop: [] })), null);   // (no action: nothing to keep)
  const r1 = await H.save(a);
  assert.equal(r1.data.source, "table"); assert.equal(r1.data.reviewed, false); assert.deepEqual(r1.data.tags, []);
  const graded = JSON.parse(JSON.stringify(a)); graded.actions.preflop[3].g = { f: 0.92, l: 0 };
  assert.equal(H.sig(graded), H.sig(a));
  const r2 = await H.save(graded);
  assert.equal(r2.id, r1.id); assert.equal(r2.data.reviewed, true);
  // played on: the flop and a check (same setup, each street's actions the old ones and more)
  const more = hand({ cards: ["Ah", "Kd", "Qs", "7h", "2c"], flop: [{ pos: "BB", action: "check" }] });
  assert.equal(H.extendsHand(a, more), true); assert.equal(H.extendsHand(more, a), false);
  assert.equal(H.extendsHand(a, hand({ cards: ["Ah", "Qd"] })), false);
  assert.equal(H.extendsHand(a, { ...more, stack: 50 }), false);
  const r3 = await H.save(more, { id: r1.id });
  assert.equal(r3.id, r1.id); assert.equal(r3.created, r1.created); assert.equal(r3.data.reviewed, false);
  // without the id it's another hand (the table gives the id)
  const r4 = await H.save(hand({ cards: ["Ah", "Kd", "Js", "7h", "2c"], flop: [{ pos: "BB", action: "check" }] }));
  assert.notEqual(r4.id, r1.id);
  assert.equal((await H.list()).length, 2);
  // the table: the last saved
  assert.equal(H.table().id, r4.id);
});

test("hands.keep: the page's hooks (a hand replaced, reviewed, imported; an untouched drills spot isn't kept)", async () => {
  const st = await open({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB() });
  const H = st.hands;
  // a drills spot put on the table, then replaced untouched
  const spot = hand({ hero: "BB" });
  H.onTable(spot, { spot: true });
  assert.equal(await H.keep("replaced", spot), null);
  // played on from it: kept, as a hand played here
  const played = hand({ hero: "BB", cards: ["Ah", "Kd", "Qs", "7h", "2c"], flop: [{ pos: "BB", action: "check" }] });
  const k = await H.keep("replaced", played);
  assert.equal(k.data.source, "table");
  assert.equal(H.table().spot, true);   // ("replaced": the table is the next hand's; the page says which)
  // from a link, replaced
  const linked = hand({ hero: "CO", cards: ["9s", "9d"] });
  H.onTable(linked, { source: "link" });
  assert.equal((await H.keep("replaced", linked)).data.source, "link");
  // reviewed: the table is that record now
  const rv = hand({ hero: "SB", cards: ["5s", "4s"] }); rv.actions.preflop[4] = { pos: "SB", action: "call", g: { f: 0.2, l: 0.1 } };
  H.onTable(rv, {});
  const kr = await H.keep("review", rv);
  assert.equal(kr.data.reviewed, true); assert.equal(H.table().id, kr.id);
  // imported, with its history (the site's hand number read from it)
  const hh = "PokerStars Hand #251234567890: Hold'em No Limit ($0.25/$0.50 USD)\nTable 'X' 6-max";
  const ki = await H.keep("import", hand({ hero: "HJ", cards: ["Tc", "Td"] }), { site: "pokerstars", hh });
  assert.equal(ki.data.source, "import"); assert.equal(ki.data.site, "pokerstars"); assert.equal(ki.data.hhId, "251234567890"); assert.equal(ki.data.hh, hh);
  // keep copies the hand at once (the page changes S right after)
  const live = hand({ hero: "UTG", cards: ["2s", "2d"] }), p = H.keep("replaced", live);
  live.cards[0] = "3s"; live.actions.preflop = [];
  assert.equal((await p).data.state.cards[0], "2s");
  assert.equal((await H.list()).length, 5);
  assert.equal(H.handNumber('<game gamecode="8812345" >'), "8812345");
  assert.equal(H.handNumber("Poker Hand #HD2093847: Hold'em No Limit"), "HD2093847");
});

test("hands: tags, a note, delete and undo (in its place in the list), export and import", async () => {
  const now = clock(), st = await open({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB(), now });
  const H = st.hands;
  const a = await H.save(hand()); now.tick();
  const b = await H.save(hand({ hero: "CO", cards: ["7s", "7d"] })); now.tick();
  assert.deepEqual((await H.list()).map(r => r.id), [b.id, a.id]);   // (newest first)
  await H.update(a.id, { tags: ["bluff", " bluff", "", "river"], note: "x" });
  let r = await H.get(a.id);
  assert.deepEqual(r.data.tags, ["bluff", "river"]); assert.equal(r.data.note, "x"); assert.ok(r.updated > r.created);
  // a tag and a note written at once (each reads the record, then writes it): both kept
  await Promise.all([H.update(b.id, { tags: ["a"] }), H.update(b.id, { note: "n" }), H.save(hand({ hero: "CO", cards: ["7s", "7d"] }), { tags: ["c"] })]);
  assert.deepEqual((await H.get(b.id)).data.tags, ["a", "c"]); assert.equal((await H.get(b.id)).data.note, "n");
  await H.remove(a.id);
  assert.deepEqual((await H.list()).map(x => x.id), [b.id]);
  assert.equal((await st.record("hands", a.id)).deleted, true);
  await H.restore(r);
  assert.deepEqual((await H.list()).map(x => x.id), [b.id, a.id]);
  assert.equal((await H.get(a.id)).data.note, "x");
  // export, then into another browser: the same hands, tags and notes; again: merged, not doubled
  const file = JSON.parse(JSON.stringify(H.exportFile(await H.list())));
  assert.equal(file.format, "gto-solver-hands"); assert.equal(file.hands.length, 2);
  const c = await open({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB() });
  assert.deepEqual(await c.hands.importFile(file), { added: 2, merged: 0, skipped: 0 });
  assert.deepEqual(await c.hands.importFile(file), { added: 0, merged: 2, skipped: 0 });
  const got = await c.hands.list();
  assert.deepEqual(got.find(x => x.data.sig === a.data.sig).data.tags, ["bluff", "river"]);
  // a state's JSON, a list of them, one with no action (skipped), and the page's clean applied
  const out = await c.hands.importFile([hand({ hero: "BB", cards: ["Js", "Th"] }), hand({ preflop: [] })], (s) => ({ ...s, cleaned: true }));
  assert.deepEqual(out, { added: 1, merged: 0, skipped: 1 });
  assert.equal((await c.hands.list()).find(x => x.data.state.hero === "BB").data.state.cleaned, true);
  assert.equal((await c.hands.list()).find(x => x.data.state.hero === "BB").data.source, "file");
  await assert.rejects(c.hands.importFile({ foo: 1 }), /no hands/);
});

test("sessions and blobs: session.js's { get, set, del } over the records", async () => {
  const st = await open({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB() });
  assert.deepEqual(await st.sessions.get(), []);
  await st.sessions.set(HISTORY);
  assert.deepEqual(await st.sessions.get(), HISTORY);
  await st.sessions.set([{ ...HISTORY[1], name: "Mon" }]);
  assert.deepEqual(await st.sessions.get(), [{ ...HISTORY[1], name: "Mon" }]);
  await st.sessions.del();
  assert.deepEqual(await st.sessions.get(), []);
  const cur = st.blob("session-current");
  assert.equal(await cur.get(), null);
  await cur.set(SESSION); assert.deepEqual(await cur.get(), SESSION);
  await cur.set({ ...SESSION, status: "done" }); assert.equal((await cur.get()).status, "done");
  await cur.del(); assert.equal(await cur.get(), null);
});

test("a change is told to listeners (and to the other tabs)", async () => {
  const st = await open({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB() });
  const seen = [];
  const off = st.onChange((c) => seen.push(c));
  await st.hands.save(hand());
  st.set("gto-study", { a: 1 }); await st.flush();
  off();
  await st.hands.save(hand({ hero: "CO" }));
  assert.deepEqual(seen, [{ stores: ["hands"], remote: false }, { stores: ["kv"], remote: false }]);
});
