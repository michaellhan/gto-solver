// The help pages (gto/static/help.js: Getting started, About & accuracy, Glossary, Changelog, Fair play), their URLs
// and menu entries in index.html, the answer badges that open the Glossary, and the first visit's chooser (#welcome).
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage, state, acts } = require("./page.js");
const Help = require("../../gto/static/help.js");

const HTML = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "static", "index.html"), "utf8");
const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };

// stand-ins for the views' scripts (the real ones need a DOM); Help reports a tab picked and closing as help.js does
const SCRIPTS = {
  "help.js": `globalThis.Help = { opened: [], page: null, open(page) { this.opened.push(page); this.page = page; this.term = globalThis.HELP_TERM; globalThis.HELP_TERM = null; },
    show(page, term) { if (term) this.term = term; if (page !== this.page) { this.page = page; viewSwitch(page); } },
    close() { const p = this.page; this.page = null; viewClosed(p); }, get shown() { return this.page; } };`,
  "session.js": `globalThis.Session = { opened: 0, open() { this.opened++; return Promise.resolve(); }, close() { viewClosed("session"); } };`,
};
const SEEN = { "gto-welcome": "seen" };   // (a visitor who has answered the chooser)
const page = (opts = {}) => loadPage({ scripts: SCRIPTS, storage: SEEN, ...opts });

test("each help page builds, its links go to pages, glossary entries and tools that exist", () => {
  assert.deepEqual(Help.PAGES.map(([p]) => p), ["start", "about", "glossary", "changelog", "fairplay"]);
  const p = page();
  const views = p.val("Object.keys(VIEWS)");
  const terms = new Set(Help.TERMS.map(([id]) => id));
  assert.equal(terms.size, Help.TERMS.length, "a glossary id is used twice");
  for (const [id, label] of Help.PAGES) {
    const h = Help.pageHtml(id);
    assert.ok(h.length > 800, id);
    assert.equal(p.val(`ROUTES[${JSON.stringify(id)}]`), label, id);   // (the tab's name is the URL's title)
    for (const [, t] of h.matchAll(/data-term="([^"]+)"/g)) assert.ok(terms.has(t), `${id}: no glossary entry ${t}`);
    for (const [, g] of h.matchAll(/data-page="([^"]+)"/g)) assert.ok(Help.PAGES.some(([x]) => x === g), `${id}: no page ${g}`);
    for (const [, g] of h.matchAll(/data-go="([^"]+)"/g)) assert.ok(g === "practice" || views.includes(g), `${id}: the page can't open ${g}`);
  }
  const g = Help.pageHtml("glossary");
  for (const id of terms) assert.match(g, new RegExp(`<dt id="hp-${id}">`));
  assert.equal(Help.pageHtml("nope"), Help.pageHtml("start"));
  assert.match(Help.tabsHtml("about"), /<button role="tab" data-page="about" aria-selected="true" class="active">About &amp; accuracy<\/button>/);
  // every glossary word an answer's badge names (index.html's data-term) has its entry
  const badge = [...HTML.matchAll(/data-term="([a-z]+)"/g)].map(m => m[1]);
  for (const t of [...badge, "approximate", "exploitability", "borrowed", "precomputed", "offsolved", "heuristic"]) assert.ok(terms.has(t), t);
});

test("the help pages say what the page does: the badge's and the grades' thresholds are index.html's", () => {
  const p = page();
  const conv = (x) => p.val(`convergence({ solver: { exploitability_pct: ${x} } }).label`);
  assert.deepEqual([conv(2.99), conv(3), conv(9.99), conv(10)], ["converged", "fair", "fair", "rough"]);
  assert.match(Help.aboutHtml(), /converged<\/span> under 3%, <span class="hp-b meh">fair<\/span> under 10%/);
  const grade = (g) => p.val(`gradeClass(${JSON.stringify(g)})`);
  assert.deepEqual([grade({ f: 0.15 }), grade({ f: 0.149 }), grade({ f: 0.03 }), grade({ f: 0.029 }), grade({ f: 0.01, r: 1 })], ["ok", "meh", "meh", "bad", "meh"]);
  const gl = Help.glossaryHtml();
  assert.match(gl, /✓ \(the solver's mix takes it at least 15% of the time\), ~ \(3-15%\) or ✗ \(under 3%\); a rough solve, or a line that gave up no EV, never grades ✗/);
  assert.deepEqual([p.val("HEUR_OK"), p.val("HEUR_MEH")], [3, 10]);
  assert.match(gl, /within 3% of the pot of the best is ✓, within 10% is ~/);
  assert.match(Help.fairplayHtml(), /Do not use it during a hand you are playing, online or live/);
  assert.match(Help.aboutHtml(), /postflop-solver/);
});

test("a help page's URL opens it; its tabs move the URL without closing; back and forward switch pages", async () => {
  const p = page({ path: "/gto-solver/about/" });
  await p.ready(); await flush();
  assert.deepEqual(p.scripts, ["help.js"]);
  assert.deepEqual(p.val("Help.opened"), ["about"]);
  assert.equal(p.ctx.document.title, "About & accuracy · GTO Solver");
  assert.deepEqual(p.pushed, []);
  p.run("Help.show('glossary')");                 // a tab picked
  assert.deepEqual(p.pushed, ["/gto-solver/glossary/"]);
  assert.equal(p.val("overlay"), "glossary");
  assert.equal(p.ctx.document.title, "Glossary · GTO Solver");
  assert.deepEqual(p.val("Help.opened"), ["about"]);   // (not closed and opened again)
  p.popTo("/gto-solver/about/"); await flush();   // back
  assert.equal(p.val("overlay"), "about");
  assert.equal(p.val("Help.page"), "about");
  p.popTo("/gto-solver/"); await flush();
  assert.equal(p.val("overlay"), null);
  assert.equal(p.val("Help.page"), null);
  assert.deepEqual(p.pushed, ["/gto-solver/glossary/"]);
});

test("the ⋯ menu's Help entries and the header's ? open the pages; another entry while one is open is a tab", async () => {
  const p = page();
  await p.ready(); await flush();
  p.click("helpBtn"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/start/"]);
  assert.deepEqual(p.val("Help.opened"), ["start"]);
  p.click("helpFairplay"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/start/", "/gto-solver/fairplay/"]);
  assert.deepEqual(p.val("Help.opened"), ["start"]);
  assert.equal(p.val("Help.page"), "fairplay");
  p.run("Help.close()");
  assert.deepEqual(p.pushed, ["/gto-solver/start/", "/gto-solver/fairplay/", "/gto-solver/"]);
  for (const [id, route] of [["helpStart", "start"], ["helpAbout", "about"], ["helpGlossary", "glossary"], ["helpChangelog", "changelog"]]) {
    p.click(id); await flush();
    assert.equal(p.location.pathname, `/gto-solver/${route}/`, id);
    p.run("Help.close()");
  }
  assert.equal(p.byId("menu").hidden, true);
});

test("an answer's badge opens the Glossary at its word", async () => {
  const p = page();
  await p.ready(); await flush();
  const click = (term) => { for (const f of p.byId("result").listeners.click) f({ target: { closest: (sel) => sel === "[data-term]" ? { dataset: { term } } : null } }); };
  click("offsolved"); await flush();
  assert.deepEqual(p.val("Help.opened"), ["glossary"]);
  assert.equal(p.val("Help.term"), "offsolved");
  assert.equal(p.location.pathname, "/gto-solver/glossary/");
  click("eqr"); await flush();                    // (already open: scrolled to the word, not opened again)
  assert.deepEqual(p.val("Help.opened"), ["glossary"]);
  assert.equal(p.val("Help.term"), "eqr");
  // a click elsewhere on the advice opens nothing
  for (const f of p.byId("result").listeners.click) f({ target: { closest: () => null } });
  assert.deepEqual(p.val("Help.opened"), ["glossary"]);
});

test("the help pages' buttons open the tools", async () => {
  const p = page();
  await p.ready(); await flush();
  p.run("helpGo('practice')"); await flush();
  assert.equal(p.val("practicing()"), true);
  p.run("helpGo('practice')"); await flush();     // (already on: stays on)
  assert.equal(p.val("practicing()"), true);
  p.run("helpGo('import')"); await flush();
  assert.equal(p.byId("importBox").hidden, false);
  assert.equal(p.val("overlay"), "import");
});

test("a first visit at the solver's URL shows the chooser once; Not now dismisses it for good", async () => {
  const p = loadPage({ scripts: SCRIPTS });
  await p.ready(); await flush();
  assert.equal(p.byId("welcome").hidden, false);
  assert.equal(p.storage.get("gto-welcome"), "due");
  assert.deepEqual(p.pushed, []);
  assert.equal(p.storage.get("gto-speed"), "fast");      // (stored after the chooser was decided)
  // a reload before answering shows it again
  const again = loadPage({ scripts: SCRIPTS, storage: Object.fromEntries(p.storage) });
  await again.ready();
  assert.equal(again.byId("welcome").hidden, false);
  p.click("welcomeClose");
  assert.equal(p.byId("welcome").hidden, true);
  assert.equal(p.storage.get("gto-welcome"), "seen");
  assert.deepEqual(p.pushed, []);
  const q = loadPage({ scripts: SCRIPTS, storage: Object.fromEntries(p.storage) });
  await q.ready(); await flush();
  assert.equal(q.byId("welcome").hidden, true);
});

test("the chooser isn't shown to a returning visitor, at a view's URL or with a link's spot", async () => {
  const r = loadPage({ scripts: SCRIPTS, storage: { "gto-speed": "accurate" } });   // (here before the chooser existed)
  await r.ready();
  assert.equal(r.byId("welcome").hidden, true);
  assert.equal(r.storage.get("gto-welcome"), "seen");
  const v = loadPage({ scripts: SCRIPTS, path: "/gto-solver/session/" });
  await v.ready(); await flush();
  assert.equal(v.byId("welcome").hidden, true);
  assert.equal(v.storage.get("gto-welcome"), "seen");
  const a = page();
  await a.ready();
  a.S = state({ hero: "CO", cards: ["Ah", "Kd"], preflop: acts("UTG f, HJ f") });
  const link = a.run("handText()").match(/^Open this spot: (.*)$/m)[1];
  const l = loadPage({ scripts: SCRIPTS, hash: link.slice(link.indexOf("#")) });
  await l.ready(); await flush();
  assert.equal(l.S.hero, "CO");
  assert.equal(l.byId("welcome").hidden, true);
  assert.equal(l.storage.get("gto-welcome"), "seen");
});

test("the chooser's choices open Practice, the session analyzer and the help pages, and dismiss it", async () => {
  const go = async (id) => { const p = loadPage({ scripts: SCRIPTS }); await p.ready(); await flush(); p.click(id); await flush();
    assert.equal(p.byId("welcome").hidden, true, id); assert.equal(p.storage.get("gto-welcome"), "seen", id); return p; };
  const x = await go("welcomePractice");
  assert.equal(x.val("practicing()"), true);
  assert.deepEqual(x.pushed, ["/gto-solver/practice/"]);
  const s = await go("welcomeSession");
  assert.deepEqual(s.scripts, ["hh.js", "session.js"]);
  assert.deepEqual(s.pushed, ["/gto-solver/session/"]);
  for (const [id, route] of [["welcomeStart", "start"], ["welcomeAbout", "about"], ["welcomeFair", "fairplay"]]) {
    const h = await go(id);
    assert.deepEqual(h.pushed, [`/gto-solver/${route}/`], id);
  }
  // Review a hand: the solver as it is, with a hint and a way to paste a hand history instead
  const r = await go("welcomeReview");
  assert.deepEqual(r.pushed, []);
  assert.equal(r.byId("toast").hidden, false);
  assert.match(r.byId("toast").innerHTML, /Pick your seat and cards in the setup panel, then enter the hand you played, seat by seat\. When it's finished, review it\./);
  const b = r.byId("toast").children.find(e => e.textContent === "Or import a hand history");
  b.onclick(); await flush();
  assert.equal(r.byId("importBox").hidden, false);
  assert.deepEqual(r.pushed, ["/gto-solver/import/"]);
  // Escape dismisses it too
  const e = loadPage({ scripts: SCRIPTS }); await e.ready();
  e.key("Escape");
  assert.equal(e.byId("welcome").hidden, true);
  assert.equal(e.storage.get("gto-welcome"), "seen");
});

test("the chooser changes nothing the advice does", async () => {
  const hand = state({ hero: "BTN", cards: ["Ah", "Kd"], preflop: acts("UTG f, HJ f, CO f") });
  const run = async (storage) => {
    const p = loadPage({ scripts: SCRIPTS, storage }); await p.ready(); await flush();
    const empty = p.byId("result").innerHTML;
    p.S = hand; await flush();
    return { empty, sent: p.posted.filter(m => m.state).map(m => m.state), payload: p.payload() };
  };
  const first = await run({}), seen = await run(SEEN);
  assert.deepEqual(first, seen);
  assert.match(first.empty, /When it's finished, the solver reviews each decision you made/);
  assert.deepEqual(first.sent, []);
});
