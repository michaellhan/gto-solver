// My hands (gto/static/hands.js) and the page's hooks into the storage layer (store.js): the hand on the table kept when
// New hand, a link or an import takes its place (an untouched spot isn't), an imported hand with its history, a
// hand opened from My hands saved back into its own record, Study at your last decision; and the panel's pure parts
// (a hand's summary and line, search, filters, tags, the import's kinds, the rows' HTML escaped).
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

test("a spot put on the table isn't kept untouched; played on, it is", async () => {
  const p = await storePage();
  const spot = state({ hero: "BB", cards: ["Js", "Ts", "9c", "8d", "2h"], preflop: OPEN });
  await p.run(`loadHandState(${JSON.stringify(spot)}, "your write-up")`);
  assert.equal(JSON.parse(p.storage.get("gto-hands-table")).spot, true);
  p.click("newhand");
  await until(async () => true, 80);
  assert.equal((await p.hands()).length, 0);
  await p.run(`loadHandState(${JSON.stringify(spot)}, "your write-up")`);
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

test("an imported hand is kept with its site, hand number and history", async () => {
  const p = await storePage();
  const text = HH("stars_6max_srp.txt");
  p.byId("hhText").value = text;
  await p.click("hhGo");
  const hs = await until(async () => { const h = await p.hands(); return h.length && h; });
  assert.equal(hs.length, 1);
  const d = hs[0].data;
  assert.equal(d.source, "import"); assert.equal(d.site, "PokerStars"); assert.equal(d.hhId, "249012345678");
  assert.equal(d.hh, text.trim());
  assert.deepEqual(d.state.cards.slice(0, 2), ["Ah", "Qd"]);
  // the same history again: the same record
  p.byId("hhText").value = text;
  await p.click("hhGo");
  await until(async () => true, 80);
  assert.equal((await p.hands()).length, 1);
});

test("Study from My hands: the hand on the table, Study at your last decision", async () => {
  const p = await storePage();
  vm.runInContext(fs.readFileSync(path.join(STATIC, "hands.js"), "utf8"), p.ctx, { filename: "hands.js" });
  p.run("globalThis.__views = []; openView = (n) => { __views.push(n); return Promise.resolve(); };");
  const st = played({ flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 3 }, { pos: "BB", action: "call" }], cards: ["Ah", "Kd", "Qs", "7h", "2c", "9d"], turn: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "check" }] });
  const rec = j(await p.run(`GTOStore.hands.save(${JSON.stringify({ ...st, finished: true })})`));
  await p.run(`(async () => MyHands.pageHost().study(await GTOStore.hands.get(${JSON.stringify(rec.id)})))()`);
  assert.deepEqual(p.val("__views"), ["study"]);
  const at = p.val("STUDY_AT");
  assert.equal(at.spot, "table");
  assert.equal(at.state.board, "Qs7h2c9d");
  assert.equal(at.state.turn.length, 1);          // (BB's check: your check back is the decision)
  assert.equal(at.state.flop.length, 3);
  assert.equal(p.S.actions.turn.length, 2);       // (the table keeps the whole hand)
});

test("Study from My hands: a hand still being entered stays on the table, and says why", async () => {
  const p = await storePage();
  vm.runInContext(fs.readFileSync(path.join(STATIC, "hands.js"), "utf8"), p.ctx, { filename: "hands.js" });
  p.run("globalThis.__views = []; globalThis.__toasts = []; openView = (n) => { __views.push(n); return Promise.resolve(); }; toast = (m) => __toasts.push(m);");
  const rec = j(await p.run(`GTOStore.hands.save(${JSON.stringify(played())})`));
  await p.run(`(async () => MyHands.pageHost().study(await GTOStore.hands.get(${JSON.stringify(rec.id)})))()`);
  assert.deepEqual(p.val("__views"), []);
  assert.equal(p.val("globalThis.STUDY_AT"), undefined);
  assert.ok(p.val("__toasts").some(m => /once it's finished/.test(m)));
  assert.equal(p.S.actions.flop.length, 1);       // (it's on the table, as it was)
});

// ---- the panel's pure parts
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

test("an import's kinds: a My hands file, a backup, links, hand histories", () => {
  assert.equal(MH.parseImport('{"format":"gto-solver-hands","hands":[]}').kind, "hands");
  assert.equal(MH.parseImport('[{"players":6}]').kind, "hands");
  assert.equal(MH.parseImport('{"format":"gto-solver-backup","stores":{}}').kind, "backup");
  assert.deepEqual(MH.parseImport("https://www.michael-han.com/gto-solver/#h=eyJhIjoxfQ==\nand #h=eyJiIjoyfQ==").links, ["eyJhIjoxfQ==", "eyJiIjoyfQ=="]);
  assert.equal(MH.parseImport(HH("stars_6max_srp.txt")).kind, "hh");
  assert.match(MH.parseImport("{nope").error, /valid JSON/);
  assert.match(MH.parseImport("  ").error, /Nothing/);
  assert.equal(MH.backupText({ counts: { kv: 1, hands: 3, sessions: 0, blobs: 1 } }), "3 hands, 1 setting or saved state, a session in progress");
  assert.equal(MH.fileName("backup", Date.UTC(2026, 10, 2, 12)), "gto-solver-backup-2026-11-02.json");
});

test("lastDecision: the hand rewound to your last action (the cards after it gone)", () => {
  const st = played({ cards: ["Ah", "Kd", "Qs", "7h", "2c", "9d", "3s"], flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 3 }, { pos: "BB", action: "call" }], turn: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "check" }], river: [{ pos: "BB", action: "bet", to: 5 }] });
  const at = MH.lastDecision(st);
  assert.deepEqual(at.actions.turn, [{ pos: "BB", action: "check" }]); assert.deepEqual(at.actions.river, []);
  assert.deepEqual(at.cards, ["Ah", "Kd", "Qs", "7h", "2c", "9d", null]);
  assert.equal(st.actions.river.length, 1);   // (a copy)
  assert.equal(MH.lastDecision(state({ hero: "BB", preflop: [{ pos: "UTG", action: "fold" }] })), null);
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
  assert.equal((src.match(/H\.exportFile\(\[?\w+\]?, host\.link\)/g) || []).length, 2);
  assert.match(src, /importFile\(p\.obj, host\.clean, host\.decode\)/);
});
