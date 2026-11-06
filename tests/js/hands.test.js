// My hands (gto/static/hands.js) and the page's hooks into the storage layer (store.js): the hand on the table kept when
// New hand, a link or an import takes its place (an untouched spot isn't), an imported hand with its history, a
// hand opened from My hands saved back into its own record; and the panel's pure parts
// (a hand's summary and line, search, filters, tags, the import's kinds, the rows' HTML escaped); a record the page
// can't open.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage, state } = require("./page.js");
const { fakeIndexedDB, fakeLocalStorage } = require("./idb_shim.js");

const STATIC = path.join(__dirname, "..", "..", "gto", "static");
const MH = require(path.join(STATIC, "hands.js"));
const { createStore } = require(path.join(STATIC, "store.js"));
const HH = (f) => fs.readFileSync(path.join(__dirname, "..", "data", "hh", f), "utf8");
const j = (x) => JSON.parse(JSON.stringify(x));

const OPEN = [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "fold" }, { pos: "CO", action: "fold" }, { pos: "BTN", action: "raise", to: 2.5 }, { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }];
const played = (o = {}) => state({ hero: "BTN", cards: ["Ah", "Kd", "Qs", "7h", "2c"], preflop: OPEN, flop: [{ pos: "BB", action: "check" }], ...o });
// wait for the store's asynchronous writes (each IndexedDB request is answered on its own turn)
async function until(f, n = 400) { for (let i = 0; i < n; i++) { const v = await f(); if (v) return v; await new Promise(r => setImmediate(r)); } throw new Error("timed out"); }
async function storePage(opts = {}) {
  const p = loadPage({ store: true, ...opts });
  await p.ready();
  await p.run("GTOStore.ready");
  p.hands = async () => j(await p.run("GTOStore.hands.list()"));
  return p;
}

test("New hand keeps the hand on the table in My hands (handreview is no longer its only copy); an empty one isn't kept", async () => {
  const p = await storePage();
  p.S = played();
  p.click("newhand");
  const hs = await until(async () => { const h = await p.hands(); return h.length && h; });
  assert.equal(hs.length, 1);
  assert.deepEqual(hs[0].data.state.cards.slice(0, 5), ["Ah", "Kd", "Qs", "7h", "2c"]);
  assert.equal(hs[0].data.source, "table");
  assert.equal(JSON.parse(p.storage.get("handreview")).actions.preflop.length, 0);   // (the table: a new hand)
  // New hand again on the empty hand: nothing more
  p.click("newhand");
  await until(async () => true, 50);
  assert.equal((await p.hands()).length, 1);
  // the record is in the page's IndexedDB and the setting mirrored as before
  assert.equal(p.storage.get("gto-store-mirror") != null, true);
});

test("a new table size keeps the hand on the table in My hands first, as New hand does; an empty one isn't kept", async () => {
  const p = await storePage();
  p.S = played(); p.run("save()");
  p.run('document.getElementById("players").children[7].onclick()');   // 9 players
  assert.equal(p.S.players, 9);
  const recs = await until(async () => { const r = await p.hands(); return r.length && r; });
  assert.equal(recs.length, 1);
  assert.equal(recs[0].data.state.players, 6);
  assert.deepEqual(recs[0].data.state.actions, j(played().actions));
  assert.deepEqual(j(await p.run("GTOStore.hands.table()")).id, null);   // (what's on the table now isn't that record)
  p.run('document.getElementById("players").children[4].onclick()');     // 6 again: nothing recorded on the 9-handed table
  await p.run("GTOStore.flush()");
  assert.equal((await p.hands()).length, 1);
});

test("a hand opened from My hands is saved back into its own record when it's replaced, not as a copy", async () => {
  const p = await storePage();
  p.S = played();
  p.click("newhand");
  const [rec] = await until(async () => { const h = await p.hands(); return h.length && h; });
  await p.run(`GTOStore.hands.update(${JSON.stringify(rec.id)}, { tags: ["river"], note: "check back?" })`);
  vm.runInContext(fs.readFileSync(path.join(STATIC, "hands.js"), "utf8"), p.ctx, { filename: "hands.js" });
  await p.run(`(async () => MyHands.pageHost().open(await GTOStore.hands.get(${JSON.stringify(rec.id)})))()`);
  assert.deepEqual(p.S.cards.slice(0, 5), rec.data.state.cards.slice(0, 5));
  assert.equal(JSON.parse(p.storage.get("gto-hands-table")).id, rec.id);
  assert.match(p.byId("toast").innerHTML, /Opened your hand: BTN vs BB, AhKd/);
  // played on (the turn) and replaced: the same record, longer, its tags and note kept
  p.S = { ...p.S, cards: [...p.S.cards.slice(0, 5), "9s", null], actions: { ...p.S.actions, flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "check" }] } };
  p.click("newhand");
  const hs = await until(async () => { const h = await p.hands(); return h[0] && h[0].data.state.cards[5] && h; });
  assert.equal(hs.length, 1); assert.equal(hs[0].id, rec.id);
  assert.deepEqual(hs[0].data.tags, ["river"]); assert.equal(hs[0].data.note, "check back?");
});

test("a spot the frozen copy put on the table isn't kept untouched; played on, it is", async () => {
  const p = await storePage();
  const spot = state({ hero: "BB", cards: ["Js", "Ts", "9c", "8d", "2h"], preflop: OPEN });
  await p.run(`loadHandState(${JSON.stringify(spot)})`);
  p.run("GTOStore.hands.onTable(S, { spot: true })");
  assert.equal(JSON.parse(p.storage.get("gto-hands-table")).spot, true);
  p.click("newhand");
  await until(async () => true, 80);
  assert.equal((await p.hands()).length, 0);
  await p.run(`loadHandState(${JSON.stringify(spot)})`);
  p.run("GTOStore.hands.onTable(S, { spot: true })");
  p.act({ action: "check" });
  p.click("newhand");
  const hs = await until(async () => { const h = await p.hands(); return h.length && h; });
  assert.equal(hs[0].data.state.actions.flop[0].action, "check");
});

test("a #h= link: the hand it replaces is kept, and the linked hand when it's replaced in turn", async () => {
  const p = await storePage();
  p.S = played();
  const linked = played({ hero: "BB", cards: ["9s", "9d", "Qs", "7h", "2c"] });
  p.location.hash = "#h=" + p.run(`encodeState(${JSON.stringify(linked)})`);
  p.fire("hashchange");
  await until(async () => (await p.hands()).length === 1);
  assert.equal(p.S.hero, "BB");
  assert.equal(JSON.parse(p.storage.get("gto-hands-table")).source, "link");
  p.click("newhand");
  const hs = await until(async () => { const h = await p.hands(); return h.length === 2 && h; });
  assert.deepEqual(hs.map(h => h.data.source).sort(), ["link", "table"]);
});

// ---- the panel's pure parts
// the panel's actions over the real page's host, records in the store as another page or version left them
async function panelWith(recs) {
  const p = await storePage();
  await p.run('loadJs("hands.js?v=" + HN_VERSION, () => window.MyHands).then(() => MyHands.useHost(MyHands.pageHost()))');
  p.ctx.__recs = recs;
  for (const i of recs.keys()) await p.run(`GTOStore.put("hands", "r${i}", __recs[${i}])`);
  p.list = () => p.run('MyHands.act({ savetable: "" })');   // (the list read again: an empty table saves nothing)
  await p.list();
  // a download: what the file says
  p.ctx.Blob = class { constructor(parts) { p.file = JSON.parse(parts.join("")); this.size = 1; } };
  p.ctx.URL = { createObjectURL: () => "blob:x", revokeObjectURL: () => {} };
  return p;
}
const NOT_OPENABLE = [{ source: "drill", state: null }, { source: "table", state: {} }, { state: { players: 11, cards: [], actions: {} }, note: "the river" },
                      { source: "import", site: "Other", hh: "a history", state: { players: 6, cards: "AhKd", actions: null } }];

test("a record this page can't open lists, exports, backs up and deletes, and Open says so with a link to the frozen copy", async () => {
  const p = await panelWith(NOT_OPENABLE);
  const has = (id) => p.run(`MyHands.recs.some(r => r.id === "${id}")`);
  assert.equal(p.run("MyHands.recs.length"), NOT_OPENABLE.length);
  const list = p.run("MyHands.listHtml ? MyHands.recs.map(r => MyHands.rowHtml(r, true)).join('') : ''");
  assert.equal(list.match(/data-row=/g).length, NOT_OPENABLE.length);
  assert.equal(p.run('MyHands.filterRecs(MyHands.recs, { q: "river" }).length'), 1);
  p.S = played();
  const table = p.storage.get("handreview");
  for (const key of ["open", "link"]) for (const i of NOT_OPENABLE.keys()) {
    await p.run(`MyHands.act({ clear: "" }); MyHands.act({ ${key}: "r${i}" })`);
    assert.match(p.run("MyHands.msg"), /can't open that hand/, `${key} ${i}`);
    assert.match(p.run("MyHands.msg"), /<a href="\/gto-solver-classic\/hands\/">the frozen copy<\/a>/);
  }
  assert.equal(p.storage.get("handreview"), table, "the table is as it was");
  assert.equal(p.clipboard.text, null, "no link copied");
  // the list as a file, with no link for a hand that can't open; one hand; a backup
  await p.run('MyHands.act({ export: "" })');
  assert.equal(p.file.hands.length, NOT_OPENABLE.length);
  assert.ok(p.file.hands.every(h => !("link" in h)));
  assert.deepEqual(p.file.hands.find(h => h.site === "Other").state, NOT_OPENABLE[3].state);
  assert.equal(p.file.hands.find(h => h.site === "Other").hh, "a history");
  await p.run('MyHands.act({ json: "r1" })');
  assert.deepEqual(p.file.hands[0].state, {});
  assert.equal(j(await p.run("GTOStore.backup()")).counts.hands, NOT_OPENABLE.length);
  // kept as they were: nothing here rewrote them
  const kept = await p.hands();
  for (const [i, data] of NOT_OPENABLE.entries()) assert.deepEqual(kept.find(r => r.id === `r${i}`).data, data);
  await p.run('MyHands.act({ del: "r0" })');
  assert.match(p.run("MyHands.msg"), /Deleted/);
  assert.equal(has("r0"), false);
  await p.run('MyHands.act({ undo: "" })');
  assert.equal(has("r0"), true);
});

test("every kind a record is saved as opens on the table, a hand imported in the frozen copy with its history too", async () => {
  const st = played();
  const recs = ["table", "review", "link", "file", "import"].map(source => ({ source, state: st, ...(source === "import" ? { site: "PokerStars", hh: "a history" } : {}) }));
  const p = await panelWith(recs);
  for (const i of recs.keys()) {
    p.S = state({ hero: "CO" });
    await p.run(`MyHands.act({ open: "r${i}" })`);
    assert.deepEqual(p.S.cards.slice(0, 5), ["Ah", "Kd", "Qs", "7h", "2c"], recs[i].source);
    assert.doesNotMatch(p.run("MyHands.msg"), /can't open/, recs[i].source);
  }
});

test("a typical-player grade is counted by the share of the pot given up", () => {
  const st = played({ river: [] });
  st.actions.flop.push({ pos: "BTN", action: "bet", to: 2, g: { f: 0, l: 0.2, h: 3.6, p: 1 } }, { pos: "BB", action: "raise", to: 8 }, { pos: "BTN", action: "fold", g: { f: 0, l: 3, h: 20, p: 1 } });
  assert.deepEqual(MH.summary(st).graded, { ok: 0, meh: 1, bad: 1 });
});

test("summary: who, the cards, the pot, how far it went, the grades", () => {
  const st = played({ river: [] });
  st.actions.flop.push({ pos: "BTN", action: "bet", to: 2, g: { f: 0.6, l: 0 } }, { pos: "BB", action: "raise", to: 8 }, { pos: "BTN", action: "fold", g: { f: 0.01, l: 1.234 } });
  const s = MH.summary(st);
  assert.equal(s.title, "BTN vs BB"); assert.deepEqual(s.hole, ["Ah", "Kd"]); assert.deepEqual(s.board, ["Qs", "7h", "2c"]);
  assert.equal(s.pot, "single-raised pot"); assert.equal(s.folded, "flop"); assert.equal(s.street, "flop");
  assert.equal(s.decisions, 3); assert.equal(s.nGraded, 2); assert.deepEqual(s.graded, { ok: 1, meh: 0, bad: 1 }); assert.equal(s.lost, 1.23);
  assert.equal(s.game, "6-handed, 100bb");
  const hu = MH.summary(state({ players: 2, hero: "SB", cards: ["2s", "2d"], preflop: [{ pos: "SB", action: "raise", to: 2.5 }, { pos: "BB", action: "raise", to: 9 }, { pos: "SB", action: "call" }] }));
  assert.equal(hu.title, "BTN/SB vs BB"); assert.equal(hu.pot, "3-bet pot");
  const mw = MH.summary(state({ hero: "BB", cards: ["7c", "6c"], preflop: [{ pos: "UTG", action: "call" }, { pos: "HJ", action: "fold" }, { pos: "CO", action: "call" }, { pos: "BTN", action: "fold" }, { pos: "SB", action: "call" }, { pos: "BB", action: "check" }] }));
  assert.equal(mw.title, "BB, 4-way"); assert.equal(mw.pot, "limped pot");
  assert.deepEqual(MH.summary(null).hole, []);   // (a broken record doesn't throw)
  // all in on the turn, the river dealt: it went to the river
  assert.equal(MH.summary(played({ cards: ["Ah", "Kd", "Qs", "7h", "2c", "9d", "3s"], turn: [{ pos: "BB", action: "all-in" }, { pos: "BTN", action: "call" }] })).street, "river");
  assert.match(MH.lineHtml(st), /Flop <span class="mh-c s">Q♠/);
  assert.match(MH.lineHtml(st), /<b>BTN folds<\/b> <span class="mh-g bad"[^>]*>✗ −1\.23/);
});

test("search, the kind and tag filters, tag counts", () => {
  const rec = (st, d = {}) => ({ id: Math.random().toString(36), created: 1, data: { state: st, tags: [], note: "", source: "table", ...d } });
  const a = rec(played(), { tags: ["bluff", "river"], note: "Villain was tanking", reviewed: true });
  const b = rec(state({ hero: "CO", cards: ["7s", "7d"], preflop: [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "raise", to: 2.5 }, { pos: "CO", action: "raise", to: 8 }] }), { source: "import", site: "GGPoker", tags: ["river"] });
  const all = [a, b];
  const q = (x) => MH.filterRecs(all, { q: x }).map(r => r.id);
  assert.deepEqual(q("AKo"), [a.id]); assert.deepEqual(q("ah kd"), [a.id]); assert.deepEqual(q("akkd"), []);
  assert.deepEqual(q("77"), [b.id]); assert.deepEqual(q("qs7h2c"), [a.id]); assert.deepEqual(q("3-bet"), [b.id]);
  assert.deepEqual(q("#bluff"), [a.id]); assert.deepEqual(q("tanking villain"), [a.id]); assert.deepEqual(q("gg"), [b.id]);
  assert.deepEqual(q("btn vs bb"), [a.id]); assert.deepEqual(q("#river").length, 2); assert.deepEqual(q(""), [a.id, b.id]);
  assert.deepEqual(MH.filterRecs(all, { kind: "reviewed" }).map(r => r.id), [a.id]);
  assert.deepEqual(MH.filterRecs(all, { kind: "unreviewed" }).map(r => r.id), [b.id]);
  assert.deepEqual(MH.filterRecs(all, { kind: "import" }).map(r => r.id), [b.id]);
  assert.deepEqual(MH.filterRecs(all, { tag: "bluff" }).map(r => r.id), [a.id]);
  assert.deepEqual(MH.tagCounts(all), [["river", 2], ["bluff", 1]]);
  assert.equal(MH.cleanTag("  #3bet, pots "), "3bet pots");
});

test("rows and lists: escaped, the empty and no-match cases, a page at a time", () => {
  const evil = '<img src=x onerror="alert(1)">';
  const r = { id: 'x"><b>', created: Date.UTC(2026, 10, 2, 12, 0), data: { state: played(), tags: [evil], note: evil, source: "import", site: evil } };
  for (const h of [MH.rowHtml(r, false), MH.rowHtml(r, true), MH.toolbarHtml([r], {})]) {
    assert.doesNotMatch(h, /<img|"><b>/);
    assert.match(h, /&lt;img/);
  }
  assert.match(MH.rowHtml(r, true), /<textarea data-note="x&quot;&gt;&lt;b&gt;"/);
  assert.match(MH.listHtml([], {}, 50, null), /No hands yet/);
  assert.match(MH.listHtml([r], { q: "zzz" }, 50, null), /No hand matches/);
  const many = Array.from({ length: 120 }, (_, i) => ({ ...r, id: "h" + i }));
  const h = MH.listHtml(many, {}, 50, null);
  assert.equal((h.match(/<li /g) || []).length, 50);
  assert.match(h, /Show 50 more \(70 left\)/);
});

test("a restored file's kinds: a My hands file, a backup", () => {
  assert.equal(MH.parseImport('{"format":"gto-solver-hands","hands":[]}').kind, "hands");
  assert.equal(MH.parseImport('[{"players":6}]').kind, "hands");
  assert.equal(MH.parseImport('{"format":"gto-solver-backup","stores":{}}').kind, "backup");
  assert.match(MH.parseImport("{nope").error, /valid JSON/);
  assert.match(MH.parseImport("  ").error, /Nothing/);
  assert.equal(MH.backupText({ counts: { kv: 1, hands: 3, sessions: 0, blobs: 1 } }), "3 hands, 1 setting or saved state, a session in progress");
  // what isn't this page's says whose it is
  const bk = MH.backupHtml(); assert.match(bk, /what the frozen copy at\s+\/gto-solver-classic\/ keeps \(analyzed sessions, Practice, the drills, Study\)/);
  assert.doesNotMatch(bk.slice(0, bk.indexOf("frozen copy")), /Practice|drills|Study|Explain/);
  assert.equal(MH.fileName("backup", Date.UTC(2026, 10, 2, 12)), "gto-solver-backup-2026-11-02.json");
});

test("store and panel together: an export imported elsewhere, links and hand histories through importFile", async () => {
  const st = createStore({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB(), broadcast: false, events: false });
  await st.ready;
  const rec = await st.hands.save(played(), { tags: ["x"] });
  const file = st.hands.exportFile([rec]);
  const p = MH.parseImport(JSON.stringify(file));
  const other = createStore({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB(), broadcast: false, events: false });
  assert.deepEqual(await other.hands.importFile(p.obj), { added: 1, merged: 0, skipped: 0 });
  const [got] = await other.hands.list();
  assert.deepEqual(got.data.tags, ["x"]); assert.equal(got.data.source, "table");
  assert.equal(MH.searchText(got).includes("#x"), true);
});

test("an export carries each hand's #h= link, and an entry with only its link imports as that hand", async () => {
  const enc = (s) => Buffer.from(JSON.stringify(s), "utf8").toString("base64");
  const dec = (b) => JSON.parse(Buffer.from(b, "base64").toString("utf8"));
  const st = createStore({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB(), broadcast: false, events: false });
  await st.ready;
  const a = await st.hands.save(played(), { tags: ["x"] });
  const b = await st.hands.save(played({ hero: "BB", cards: ["Js", "Th", "Qs", "7h", "2c"] }));
  const file = j(st.hands.exportFile([a, b], (s) => "https://example.test/gto-solver/#h=" + enc(s)));
  assert.equal(file.hands.length, 2);
  for (const h of file.hands) assert.deepEqual(dec(h.link.split("#h=")[1]), h.state);
  assert.equal(j(st.hands.exportFile([a])).hands[0].link, undefined);   // (no link without the page's encoder)
  const other = createStore({ localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB(), broadcast: false, events: false });
  await other.ready;
  const links = { format: file.format, hands: [{ link: file.hands[0].link, tags: ["y"] }, { link: file.hands[1].link }, { link: "#h=!!!" }, { link: "no link" }] };
  assert.deepEqual(await other.hands.importFile(links, null, dec), { added: 2, merged: 0, skipped: 2 });
  const got = await other.hands.list();
  assert.deepEqual(got.map(r => r.data.sig).sort(), [a.data.sig, b.data.sig].sort());
  assert.deepEqual(got.find(r => r.data.sig === a.data.sig).data.tags, ["y"]);
  // the panel's export goes through the page's link (pageHost().link: the solver's URL and encodeState)
  const src = fs.readFileSync(path.join(STATIC, "hands.js"), "utf8");
  assert.equal((src.match(/H\.exportFile\(\[?\w+\]?, linkFor\)/g) || []).length, 2);
  assert.match(src, /linkFor = \(st\) => host\.clean\(st\) \? host\.link\(st\) : undefined/);
  assert.match(src, /importFile\(p\.obj, host\.clean, host\.decode\)/);
});
