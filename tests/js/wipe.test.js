// "Delete everything this site keeps" (GTOStore.wipe, My hands): every key the site's code uses is in store.js's one
// list (this file greps gto/static for them), what the site keeps goes, what other apps keep on the origin stays, a
// failed IndexedDB falls back and says what stayed, nothing comes back on the next load, and the page reports what it did.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const STATIC = path.join(__dirname, "..", "..", "gto", "static");
const { createStore, SITE_KEYS, SITE_DATABASES, SITE_CACHES } = require(path.join(STATIC, "store.js"));
const MH = require(path.join(STATIC, "hands.js"));
const { fakeIndexedDB, seedOldDb, fakeLocalStorage } = require("./idb_shim.js");

// the string literals in gto/static that aren't storage keys, and what each is instead
const NOT_KEYS = {
  "gto-solver-backup": "a backup file's format", "gto-solver-hands": "a My hands file's format", "gto-solver-": "a download's file name",
  "gto-": "the prefix reconcile takes keys in by", "gto-store:": "the prefix of the gto-store:<store> keys, each in SITE_KEYS",
  "gto-session-": "the prefix of the analyzer's fallback, whose one key is gto-session-history",
  "gto-store": "a database", "gto-pf": "a database", "gto-solver": "a cache", "gto-solver-classic": "a cache",
};
// every storage key a text uses: each "gto-…" literal, and whatever the storage calls name, directly or by a constant
function scan(text) {
  const out = new Set();
  for (const m of text.matchAll(/["'`](gto-[A-Za-z0-9:_-]*)["'`$]/g)) out.add(m[1]);
  const con = new Map([...text.matchAll(/\b([A-Z][A-Z_]*)\s*=\s*"([^"]+)"/g)].map(m => [m[1], m[2]]));
  const CALL = /(?:\b(?:getItem|setItem|removeItem|getRaw|setRaw)|\b(?:KV|pageStore|GTOStore|store\(\))\.(?:set|get|remove)|\bsaveJSON)\(\s*(?:"([^"]+)"|'([^']+)'|([A-Z][A-Z_]*)\b)/g;
  for (const m of text.matchAll(CALL)) {
    const k = m[1] || m[2] || con.get(m[3]);
    if (k) out.add(k); else out.add("?" + m[3]);   // (a constant with no literal here: a key this scan can't see)
  }
  for (const m of text.matchAll(/function \w*[kK]ey\(\)[^\n]*/g)) for (const q of m[0].matchAll(/"([^"]+)"/g)) out.add(q[1]);   // (stateKey())
  return out;
}
const SOURCES = fs.readdirSync(STATIC).filter(f => /\.(js|html)$/.test(f));

test("every storage key in gto/static is in store.js's SITE_KEYS (a new key can't escape the delete)", () => {
  const seen = new Set();
  for (const f of SOURCES) {
    const text = fs.readFileSync(path.join(STATIC, f), "utf8");
    for (const k of scan(text)) if (!SITE_KEYS.includes(k) && !(k in NOT_KEYS)) assert.fail(`${f} uses the storage key "${k}", which isn't in SITE_KEYS (store.js)`);
    for (const k of scan(text)) seen.add(k);
  }
  // the scan finds what it must, the ones with no gto- prefix and the ones in constants among them
  for (const k of ["handreview", "fourcolor", "gto-anthropic-key", "gto-explain-own", "gto-session-run", "gto-writeup", "gto-stakes", "gto-welcome", "gto-isolate", "gto-hands-ui", "gto-practice-review", "gto-store-mirror"]) assert.ok(seen.has(k), k);
  // and it would catch these
  for (const [src, k] of [['localStorage.setItem("gto-brand-new", 1)', "gto-brand-new"], ['pageStore.setRaw("plainkey", x)', "plainkey"],
                          ['const NEWKEY = "other"; sessionStorage.getItem(NEWKEY)', "other"], ['KV.set("a-key", v)', "a-key"], ['const K = `gto-x${n}`', "gto-x"]])
    assert.ok(scan(src).has(k), src);
  // the names of the databases and caches: exactly these, each in the source that uses it
  assert.deepEqual(Object.keys(SITE_DATABASES).sort(), ["gto-pf", "gto-session", "gto-store"]);
  assert.deepEqual(SITE_DATABASES["gto-pf"], ["solves"]);
  assert.deepEqual(SITE_CACHES, ["gto-solver", "gto-solver-classic"]);
  assert.match(fs.readFileSync(path.join(STATIC, "study.js"), "utf8"), /indexedDB\.open\("gto-pf", 1\)/);
  assert.match(fs.readFileSync(path.join(STATIC, "worker.js"), "utf8"), /caches\.open\("gto-solver"\)/);
});

test("no storage key is another app's, and nothing clears the origin", () => {
  // the keys the other folders of the site repo and Lizard use (read 2026-11-04): their prefixes
  const OTHERS = /^(lizard|tp\.|lb\.|hardest:|bj:|c2|pgstyle:|preflop-|humanizer:|sheephav-|initials-|supabase\.)/;
  for (const k of SITE_KEYS) assert.ok(!OTHERS.test(k), k);
  for (const k of SITE_KEYS) assert.ok(k.startsWith("gto-") || ["handreview", "fourcolor"].includes(k), k);
  for (const f of SOURCES) {
    const text = fs.readFileSync(path.join(STATIC, f), "utf8").replace(/^\s*\/\/.*$/gm, "");   // (the code, not the comment that says so)
    assert.doesNotMatch(text, /(localStorage|sessionStorage)\.clear\(/, f);
    assert.doesNotMatch(text, /indexedDB\.databases\(/, f);
    assert.doesNotMatch(text, /caches\.keys\(/, f);
  }
});

const KEY = "sk-ant-api03-Zx9wQ_secretSECRETsecret-0123456789";
const HAND = { players: 6, hero: "BTN", stack: 100, cards: ["Ah", "Kd", null, null, null, null, null], actions: { preflop: [{ pos: "BTN", action: "raise", to: 2.5 }], flop: [], turn: [], river: [] } };
const OTHER_APPS = { "lizard.solveTimes.v1": "[1]", "lizard:anything": "x", "lizard.practiceHabits.v1": "{}", "hardest:theme": "dark", "bj:stats": "{}", "preflop-lab-stats-v1": "{}", "tp.me": "me", "humanizer:key": "theirs", "sheephav-draft": "text" };
const fakeCaches = (names) => { const c = { names: new Set(names), deleted: [], async delete(n) { c.deleted.push(n); return c.names.delete(n); } }; return c; };
async function site(opts = {}) {
  const idb = opts.idb === undefined ? fakeIndexedDB() : opts.idb, ls = fakeLocalStorage({
    handreview: JSON.stringify(HAND), fourcolor: "1", "gto-speed": "accurate", "gto-practice-score": "{\"hands\":3}", "gto-drills": "{}", "gto-mt": "1", "gto-welcome": "seen", "gto-stakes": "null", "gto-writeup": "{}",
    "gto-a-key-from-an-older-version": "x", "gto-anthropic-key": KEY, "gto-explain-own": "1", "gto-session-run": "{}", ...OTHER_APPS, ...(opts.ls || {}) });
  const ss = fakeLocalStorage({ "gto-isolate": "1", "gto-session-run": "tab", "lizard.sw-reloaded": "1", "lb.place": "1" });
  const caches = fakeCaches(["gto-solver", "gto-solver-classic", "lizard:solver", "other"]);
  if (idb && !opts.bare) {
    await seedOldDb(idb, "gto-session", "kv", { current: { v: 1, name: "old" }, history: [{ id: 1, date: 1, name: "old one" }] });
    await seedOldDb(idb, "gto-pf", "solves", { a: { at: 1 } });
    await seedOldDb(idb, "lizard", "things", { mine: 1 });
  }
  const st = createStore({ broadcast: false, events: false, localStorage: ls, sessionStorage: ss, indexedDB: idb, caches, wipeWait: 5, ...(opts.env || {}) });
  await st.ready;
  await st.hands.save(HAND, { table: false });
  await st.put("sessions", "s2", { id: "s2", date: 2 });
  await st.blob("session-current").set({ v: 1 });
  await st.flush();
  return { st, idb, ls, ss, caches };
}
const reload = async ({ ls, ss, idb, caches }) => { const st = createStore({ broadcast: false, events: false, localStorage: ls, sessionStorage: ss, indexedDB: idb, caches }); await st.ready; return st; };

test("wipe: everything the site keeps goes, with its old copies, and other apps' data stays", async () => {
  const s = await site(), r = await s.st.wipe();
  assert.equal(r.ok, true); assert.deepEqual(r.failed, []);
  assert.equal(r.deleted.hands, 1); assert.equal(r.deleted.sessions, 2); assert.equal(r.deleted.blobs, 1); assert.ok(r.deleted.kv >= 9);
  assert.equal(r.aiKey, true);
  assert.deepEqual(r.databases.sort(), ["gto-pf", "gto-session", "gto-store"]);
  assert.deepEqual(r.caches.sort(), ["gto-solver", "gto-solver-classic"]);
  // localStorage: only the other apps' keys are left, and the old key the store had taken in went too
  assert.deepEqual([...s.ls.map.keys()].sort(), Object.keys(OTHER_APPS).sort());
  assert.deepEqual(Object.fromEntries(s.ls.map), OTHER_APPS);
  assert.deepEqual([...s.ss.map.keys()].sort(), ["lb.place", "lizard.sw-reloaded"]);
  for (const k of ["gto-speed", "handreview", "gto-anthropic-key", "gto-a-key-from-an-older-version"]) assert.ok(r.keys.includes(k), k);
  // the databases: empty, the other app's as it was; only the two named caches were asked for
  for (const [n, stores] of Object.entries(SITE_DATABASES)) for (const t of stores) if (t !== "meta") assert.deepEqual(s.idb.dump(n, t), {}, n + " " + t);
  assert.deepEqual(s.idb.dump("lizard", "things"), { mine: 1 });
  assert.deepEqual(s.caches.deleted, ["gto-solver", "gto-solver-classic"]);
  assert.deepEqual([...s.caches.names].sort(), ["lizard:solver", "other"]);
});

test("wipe: nothing comes back on the next load, and the page works afterwards", async () => {
  const s = await site();
  await s.st.wipe();
  const st = await reload(s);
  assert.deepEqual(await st.hands.list(), []);
  assert.deepEqual(await st.list("kv"), []);
  assert.deepEqual(await st.list("sessions"), []);
  assert.equal(await st.blob("session-current").get(), null);
  const b = await st.backup();
  assert.deepEqual(b.counts, { kv: 0, hands: 0, sessions: 0, blobs: 0 });
  // the database still opens, and keeps what's saved from here on
  st.setRaw("gto-speed", "fast");
  await st.hands.save(HAND, { table: false });
  await st.flush();
  assert.equal((await reload(s).then(x => x.hands.list())).length, 1);
  assert.equal(s.ls.getItem("gto-speed"), "fast");
  // and a backup made before restores into it
  const t = await site(), bak = await t.st.backup();
  await t.st.wipe();
  const st2 = await reload(t), out = await st2.restore(bak);
  assert.equal(out.stores.hands.added, 1);
  assert.equal((await st2.hands.list()).length, 1);
});

test("wipe: the page's open connections don't stop it (the store's own, and Study's on gto-pf, which never closes)", async () => {
  const idb = fakeIndexedDB(), s = await site({ idb });
  const study = await new Promise((ok) => { const q = idb.open("gto-pf", 1); q.onsuccess = () => ok(q.result); });
  idb.blockDelete = true;   // (a deleteDatabase would wait for it)
  const r = await s.st.wipe();
  assert.equal(r.ok, true);
  assert.deepEqual(idb.dump("gto-pf", "solves"), {});
  assert.equal(study.closed, false);
});

test("wipe: until the page loads again nothing the page still holds is written back", async () => {
  const s = await site();
  await s.st.wipe();
  s.st.setRaw("handreview", JSON.stringify(HAND));
  s.st.set("gto-practice-score", { hands: 9 });
  await s.st.hands.keep("replaced", HAND);
  await s.st.flush();
  assert.equal(s.ls.getItem("handreview"), null);
  assert.equal(s.ls.getItem("gto-practice-score"), null);
  assert.deepEqual(s.idb.dump("gto-store", "hands"), {});
  assert.deepEqual(s.idb.dump("gto-store", "kv"), {});
});

test("wipe: a database whose stores won't clear is deleted by name instead", async () => {
  const idb = fakeIndexedDB(), s = await site({ idb });
  idb.failClear = true;
  const r = await s.st.wipe();
  idb.failClear = false;
  assert.equal(r.records, true); assert.deepEqual(r.databases.includes("gto-store"), true);
  assert.deepEqual(r.failed.map(f => f.what).sort(), ["gto-pf", "gto-session"]);   // (those two have no other way: said, not hidden)
  assert.equal(r.ok, false);
  assert.equal(s.ls.getItem("gto-speed"), null);
  assert.deepEqual(await (await reload(s)).hands.list(), []);
});

for (const [name, set] of [["fails", (i) => { i.failDelete = true; }], ["is blocked by another tab", (i) => { i.blockDelete = true; }]]) {
  test(`wipe: where IndexedDB refuses and its delete ${name}, nothing is half deleted and the report says so`, async () => {
    const idb = fakeIndexedDB(), s = await site({ idb });
    idb.failClear = true; set(idb);
    const r = await s.st.wipe();
    idb.failClear = idb.failDelete = idb.blockDelete = false;
    assert.equal(r.ok, false); assert.equal(r.records, false);
    assert.ok(r.failed.some(f => f.what === "records"));
    assert.deepEqual(r.deleted, {});
    // the records and their mirror are both still here; only what's never a record went, and the report says that
    assert.equal(s.ls.getItem("gto-speed"), "accurate");
    assert.equal(s.ls.getItem("handreview"), JSON.stringify(HAND));
    assert.notEqual(s.ls.getItem("gto-store-mirror"), null);
    assert.equal(s.ls.getItem("gto-anthropic-key"), null);
    assert.equal((await s.st.hands.list()).length, 1);
    assert.equal((await (await reload(s)).hands.list()).length, 1);
    const text = MH.wipeText(r);
    assert.match(text, /^Deleted your saved AI key/);
    assert.match(text, /Couldn't delete your saved hands, sessions, progress and settings/);
    assert.match(text, /They're still here/);
    // the page goes on saving, and a second try does the rest
    s.st.setRaw("gto-speed", "fast");
    assert.equal(s.ls.getItem("gto-speed"), "fast");
    const r2 = await s.st.wipe();
    assert.equal(r2.records, true); assert.equal(r2.ok, true);
    assert.deepEqual(await (await reload(s)).hands.list(), []);
  });
}

test("wipe: without IndexedDB the records are the localStorage fallback's, and those go; in memory too", async () => {
  const s = await site({ idb: null });
  assert.equal(s.st.backend, "localStorage");
  assert.ok(JSON.parse(s.ls.getItem("gto-store:hands")));
  const r = await s.st.wipe();
  assert.equal(r.ok, true); assert.equal(r.deleted.hands, 1);
  for (const k of s.ls.map.keys()) assert.ok(!k.startsWith("gto-"), k);
  assert.deepEqual(await (await reload(s)).hands.list(), []);
  const m = await site({ idb: null, env: { fallback: "memory", localStorage: null } });
  assert.equal(m.st.backend, "memory");
  assert.equal((await m.st.wipe()).deleted.hands, 1);
  assert.deepEqual(await m.st.hands.list(), []);
});

test("wipe: an IndexedDB that wouldn't open is reported: what it holds can't be said to be gone", async () => {
  const idb = fakeIndexedDB({ failOpen: true }), s = await site({ idb, bare: true });
  assert.equal(s.st.backend, "localStorage");
  const r = await s.st.wipe();
  assert.equal(r.records, true); assert.equal(r.ok, false);
  assert.deepEqual(r.failed.map(f => f.what).sort(), ["gto-pf", "gto-session", "gto-store"]);
  assert.match(MH.wipeText(r), /Couldn't delete whatever this browser's IndexedDB holds/);
});

test("wipe: a tab that saves again in the meantime is reported, not hidden", async () => {
  const idb = fakeIndexedDB(), s = await site({ idb });
  s.ss.removeItem = (k) => { if (k === "gto-speed") idb.poke("gto-store", "hands", "x", { id: "x", data: {} }); };
  const r = await s.st.wipe();
  assert.equal(r.ok, false);
  assert.deepEqual(r.failed.map(f => f.what), ["tab"]);
  assert.match(MH.wipeText(r), /Another tab of this site saved something again/);
});

test("wipe: other tabs hear of it: they stop writing and the page reloads; a tab of the frozen copy just reads again", async () => {
  const idb = fakeIndexedDB(), ls = fakeLocalStorage();
  const a = createStore({ events: false, localStorage: ls, indexedDB: idb }), b = createStore({ events: false, localStorage: ls, indexedDB: idb });
  await a.ready; await b.ready;
  const heard = new Promise(res => b.onChange(res));
  await a.hands.save(HAND, { table: false });
  await heard;
  const wiped = new Promise(res => b.onChange((c) => { if (c.wiped) res(c); }));
  assert.equal((await a.wipe()).ok, true);
  assert.deepEqual({ ...(await wiped), stores: null }, { stores: null, remote: true, wiped: true });
  b.setRaw("gto-speed", "accurate");
  assert.equal(ls.getItem("gto-speed"), null);
  assert.deepEqual(await b.hands.list(), []);
  // the page's listener
  assert.match(fs.readFileSync(path.join(STATIC, "index.html"), "utf8"), /onChange\(\(c\) => \{ if \(c\.wiped && c\.remote\) location\.reload\(\); \}\)/);
});

test("the panel: one control, a confirm that says plainly what goes, Back up first, and a report", async () => {
  const s = await site();
  MH.useHost({ store: () => s.st });
  assert.match(MH.wipeHtml(null), /data-wipe>Delete everything this site keeps</);
  await MH.act({ wipe: "" });
  const c = MH.wipeHtml(MH.wiping);
  for (const t of ["saved hands", "sessions", "drill and practice progress", "settings", "saved AI key", "cached solver files", "/gto-solver-classic/", "shares this storage", "Lizard's own data stays", "Lizard could no longer import these hands", "can't be undone"]) assert.ok(c.includes(t), t);
  assert.match(c, /<button data-backup>Back up first<\/button>/);
  assert.match(c, /data-wipego>Delete everything</);
  assert.match(c, /data-wipecancel>Cancel</);
  await MH.act({ wipecancel: "" });
  assert.equal(MH.wiping, null);
  assert.equal((await s.st.hands.list()).length, 1);   // (nothing went without the confirm)
  await MH.act({ wipe: "" });
  await MH.act({ wipego: "" });
  const h = MH.wipeHtml(MH.wiping);
  assert.match(h, /<b>Deleted<\/b>/);
  assert.match(h, /Deleted 1 hand, 2 analyzed sessions, \d+ settings and saved states, a session in progress, your saved AI key, the cached solver files\./);
  assert.match(h, /data-reload/);
  assert.doesNotMatch(h, /data-wipego/);
  assert.deepEqual(MH.recs, []);
  // a report that isn't all good offers it again, and to close unless the records went
  assert.match(MH.wipeHtml({ report: { ok: false, records: false, failed: [{ what: "records", why: "x" }] } }), /data-wipego>Try again[\s\S]*data-wipecancel>Close/);
  assert.match(MH.wipeHtml({ report: { ok: false, records: true, deleted: { hands: 1 }, failed: [{ what: "gto-pf", why: "x" }] } }), /Not everything was deleted[\s\S]*data-wipego>Try again[\s\S]*data-reload/);
  // a store that throws is a report too
  MH.useHost({ store: () => ({ wipe: async () => { throw new Error("no"); }, hands: { list: async () => [] } }) });
  await MH.act({ wipego: "" });
  assert.equal(MH.wiping.report.ok, false);
  await MH.act({ wipecancel: "" });
});

test("Back up first keeps the confirm open and says what's in the file", async () => {
  const s = await site(), urls = [];
  globalThis.URL.createObjectURL = () => "blob:x"; globalThis.URL.revokeObjectURL = () => {};
  globalThis.document = { createElement: () => ({ click() { urls.push(1); }, remove() {} }), body: { appendChild() {} } };
  try {
    MH.useHost({ store: () => s.st });
    await MH.act({ wipe: "" });
    await MH.act({ backup: "" });
    assert.equal(urls.length, 1);
    assert.match(MH.wipeHtml(MH.wiping), /Backed up 1 hand, 2 analyzed sessions/);
    assert.match(MH.wipeHtml(MH.wiping), /data-wipego/);
    await MH.act({ wipecancel: "" });
  } finally { delete globalThis.document; }
});

test("the panel: a second click while a delete runs starts no second delete", async () => {
  let runs = 0, done;
  MH.useHost({ store: () => ({ wipe: () => { runs++; return new Promise((r) => { done = r; }); }, hands: { list: async () => [] } }) });
  await MH.act({ wipe: "" });
  const first = MH.act({ wipego: "" });
  await MH.act({ wipego: "" });
  assert.equal(runs, 1);
  done({ ok: true, records: true, deleted: { hands: 1 }, failed: [] });
  await first;
  assert.equal(MH.wiping.report.deleted.hands, 1);
});
