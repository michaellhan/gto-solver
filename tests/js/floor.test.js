// The floor: My hands (hands.js) and the store (store.js) are what stays when the page is cut down, so neither may lean
// on the views that are to leave. A Restore of a backup holding all four stores works with those files gone.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage } = require("./page.js");
const { createStore } = require("../../gto/static/store.js");
const { fakeIndexedDB, fakeLocalStorage } = require("./idb_shim.js");
const STATIC = path.join(__dirname, "..", "..", "gto", "static");

const LEAVING = ["practice.js", "session.js", "drills.js", "reports.js", "explain.js", "study.js", "writeup.js"];
const GLOBALS = { "practice.js": "Practice", "session.js": "Session", "drills.js": "Drills", "reports.js": "FlopReports", "explain.js": "Explain", "study.js": "Study", "writeup.js": "WriteUp" };
// the code alone: a comment may say where a thing comes from
const code = (s) => s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "").replace(/\s\/\/ .*$/gm, "");
function deps(text) {
  const c = code(text), out = [];
  for (const f of LEAVING) {
    const re = f.replace(".", "\\.");
    if (new RegExp(re).test(c)) out.push(f);
    if (new RegExp(`\\b(window|globalThis|self)\\.${GLOBALS[f]}\\b|\\b${GLOBALS[f]}\\.[a-zA-Z]`).test(c)) out.push(GLOBALS[f]);
    if (new RegExp(`require\\([^)]*${f.replace(".js", "")}`).test(c)) out.push("require " + f);
  }
  return out;
}

test("hands.js and store.js need none of the views that are to leave", () => {
  for (const f of ["hands.js", "store.js"]) {
    assert.deepEqual(deps(fs.readFileSync(path.join(STATIC, f), "utf8")), [], f);
  }
});

test("the guard catches what it's for", () => {
  for (const [src, hit] of [['loadJs("practice.js?v=1", () => 1)', "practice.js"], ["window.Session.load()", "Session"], ["globalThis.Drills", "Drills"], ["Explain.open(x)", "Explain"],
                            ['s.src = "reports.js"', "reports.js"], ['const w = require("./writeup.js")', "writeup.js"], ['loadJs("study.js", () => globalThis.Study)', "study.js"]])
    assert.ok(deps(src).includes(hit), src);
  assert.deepEqual(deps('// the analyzer (session.js) writes this\nconst a = 1; // and practice.js\n/* Study.open */'), []);
});

async function fullBackup() {
  const st = createStore({ broadcast: false, events: false, localStorage: fakeLocalStorage(), indexedDB: fakeIndexedDB() });
  await st.ready;
  st.setRaw("gto-practice-score", '{"hands":7}');
  await st.hands.save({ players: 6, hero: "BTN", stack: 100, cards: ["Ah", "Kd", null, null, null, null, null], actions: { preflop: [{ pos: "BTN", action: "raise", to: 2.5 }], flop: [], turn: [], river: [] } }, { table: false });
  await st.put("sessions", "s1", { id: "s1", date: 1, name: "One night" });
  await st.blob("session-current").set({ v: 1, name: "going" });
  await st.flush();
  const b = await st.backup();
  assert.equal(b.counts.kv, 1);
  assert.deepEqual([b.counts.hands, b.counts.sessions, b.counts.blobs], [1, 1, 1]);
  return b;
}

test("a Restore of all four stores works with those files gone", async () => {
  const b = await fullBackup();
  const p = loadPage({ store: true, scripts: Object.fromEntries(LEAVING.map(f => [f, null])) });
  await p.ready();
  await p.run("GTOStore.ready");
  await p.run('loadJs("hands.js?v=" + HN_VERSION, () => window.MyHands).then(() => MyHands.useHost(MyHands.pageHost()))');
  await p.run(`MyHands.importText(${JSON.stringify(JSON.stringify(b))}, "backup.json")`);
  await p.run('MyHands.act({ restorego: "" })');
  const list = JSON.parse(await p.run("GTOStore.hands.list().then(JSON.stringify)"));
  assert.equal(list.length, 1);
  assert.equal(list[0].data.state.hero, "BTN");
  const sessions = JSON.parse(await p.run('GTOStore.list("sessions").then(JSON.stringify)'));
  assert.equal(sessions[0].data.name, "One night");
  assert.equal(await p.run('GTOStore.blob("session-current").get().then(x => x.name)'), "going");
  assert.equal(p.run('GTOStore.getRaw("gto-practice-score")'), '{"hands":7}');
  // and none of the seven was so much as asked for
  for (const f of LEAVING) assert.ok(!p.scripts.includes(f), f);
});
