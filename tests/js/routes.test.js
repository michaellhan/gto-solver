// The views' URLs (index.html's ROUTES): a URL opens its view, opening and closing a view pushes its URL, back and
// forward switch views, share links built on any view's URL are the solver's, and the route table is the one
// gto/routes.py (scripts/deploy.sh, gto/server.py) reads.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { loadPage, state, acts } = require("./page.js");

const ROOT = path.join(__dirname, "..", "..");
const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const linkOf = (text) => text.match(/^Open this spot: (.*)$/m)[1];

// stand-ins for the views' scripts (the real ones need a DOM): they record opening and report closing as the real ones do
const SCRIPTS = { "hands.js": `globalThis.MyHands = { opened: 0, closed: 0, open() { this.opened++; return Promise.resolve(); }, close() { this.closed++; viewClosed("hands"); } };`,
                  "session.js": `globalThis.Session = { opened: 0, open() { this.opened++; return Promise.resolve(); }, close() { viewClosed("session"); } };`,
                  "study.js": `globalThis.Study = { opened: 0, open(o) { this.opened++; this.as = o && o.view || "study"; return Promise.resolve(); }, close() { viewClosed(this.as); } };` };
const page = (opts = {}) => loadPage({ scripts: SCRIPTS, ...opts });

test("paths parse to a view and the solver's root", () => {
  const p = page();
  const r = (x) => p.val(`parseRoute(${JSON.stringify(x)})`);
  assert.deepEqual(r("/gto-solver/"), { root: "/gto-solver/", view: "solver" });
  assert.deepEqual(r("/gto-solver/index.html"), { root: "/gto-solver/", view: "solver" });
  assert.deepEqual(r("/gto-solver/practice/"), { root: "/gto-solver/", view: "practice" });
  assert.deepEqual(r("/gto-solver/practice"), { root: "/gto-solver/", view: "practice" });
  assert.deepEqual(r("/gto-solver/practice/index.html"), { root: "/gto-solver/", view: "practice" });
  assert.deepEqual(r("/gto-solver/session/"), { root: "/gto-solver/", view: "session" });
  assert.deepEqual(r("/"), { root: "/", view: "solver" });                    // the local server
});

test("the solver's URL: nothing pushed, Practice stays off even if it was on last time", async () => {
  const p = page({ storage: { "gto-practice": "1" } });
  await p.ready(); await flush();
  assert.deepEqual(p.pushed, []);
  assert.deepEqual(p.replaced, []);
  assert.ok(!p.scripts.includes("practice.js"));
  assert.equal(p.val("practicing()"), false);
  assert.equal(p.ctx.document.title, "GTO Solver");
});

test("/practice/ opens Practice; its button and back/forward switch between it and the solver", async () => {
  const p = page({ path: "/gto-solver/practice/" });
  await p.ready(); await flush();
  assert.ok(p.scripts.includes("practice.js"));
  assert.equal(p.val("practicing()"), true);
  assert.equal(p.ctx.document.title, "Practice · GTO Solver");
  assert.deepEqual(p.pushed, []);                    // (opened at its own URL)
  await p.click("practiceBtn"); await flush();       // Exit practice
  assert.equal(p.val("practicing()"), false);
  assert.deepEqual(p.pushed, ["/gto-solver/"]);
  assert.equal(p.ctx.document.title, "GTO Solver");
  p.popTo("/gto-solver/practice/"); await flush();   // back
  assert.equal(p.val("practicing()"), true);
  p.popTo("/gto-solver/"); await flush();            // forward
  assert.equal(p.val("practicing()"), false);
  assert.deepEqual(p.pushed, ["/gto-solver/"]);      // (back and forward push nothing)
  await p.click("practiceBtn"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/", "/gto-solver/practice/"]);
});

test("the query string stays on every view's URL", async () => {
  const p = page({ search: "?debug=1" });
  await p.ready();
  await p.click("practiceBtn"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/practice/?debug=1"]);
});

test("a view: opening pushes its URL, closing pushes the solver's, back and forward open and close it", async () => {
  const p = page();
  await p.ready();
  p.click("myHands"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/hands/"]);
  assert.equal(p.ctx.document.title, "My hands · GTO Solver");
  p.run("MyHands.close()");
  assert.deepEqual(p.pushed, ["/gto-solver/hands/", "/gto-solver/"]);
  assert.equal(p.ctx.document.title, "GTO Solver");
  p.popTo("/gto-solver/hands/"); await flush();         // back: open again
  assert.equal(p.val("MyHands.opened"), 2);
  assert.equal(p.val("overlay"), "hands");
  p.popTo("/gto-solver/"); await flush();
  assert.equal(p.val("MyHands.closed"), 2);
  assert.equal(p.val("overlay"), null);
  assert.equal(p.pushed.length, 2);
});

test("a view's URL opened directly opens it", async () => {
  const r = page({ path: "/gto-solver/hands/" });
  await r.ready(); await flush();
  assert.equal(r.val("MyHands.opened"), 1);
  assert.equal(r.location.pathname, "/gto-solver/hands/");
  assert.deepEqual(r.pushed, []);
  const s = page({ path: "/gto-solver/session/" });
  await s.ready(); await flush();
  assert.deepEqual(s.scripts, ["hh.js", "session.js"]);
  assert.equal(s.val("Session.opened"), 1);
  assert.equal(s.ctx.document.title, "Analyze session · GTO Solver");
  const y = page({ path: "/gto-solver/study/" });
  await y.ready(); await flush();
  assert.deepEqual(y.scripts, ["study.js"]);
  assert.equal(y.val("Study.opened"), 1);
  assert.equal(y.ctx.document.title, "Study · GTO Solver");
  y.run("Study.close()");
  assert.deepEqual(y.pushed, ["/gto-solver/"]);
  const f = page({ path: "/gto-solver/preflop/" });   // (Study as the menu's Preflop solve: the game form)
  await f.ready(); await flush();
  assert.deepEqual(f.scripts, ["study.js"]);
  assert.equal(f.val("Study.as"), "preflop");
  assert.equal(f.ctx.document.title, "Preflop solve · GTO Solver");
  f.run("Study.close()");
  assert.deepEqual(f.pushed, ["/gto-solver/"]);
  const i = page({ path: "/gto-solver/import/" });
  await i.ready(); await flush();
  assert.equal(i.byId("importBox").hidden, false);
  i.click("hhCancel");
  assert.equal(i.byId("importBox").hidden, true);
  assert.deepEqual(i.pushed, ["/gto-solver/"]);
});

test("a view over Practice returns to /practice/ when closed; back to the solver turns Practice off", async () => {
  const p = page({ path: "/gto-solver/practice/" });
  await p.ready(); await flush();
  p.click("myHands"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/hands/"]);
  p.run("MyHands.close()");
  assert.deepEqual(p.pushed, ["/gto-solver/hands/", "/gto-solver/practice/"]);
  assert.equal(p.val("practicing()"), true);
  p.popTo("/gto-solver/hands/"); await flush();   // back to My hands: Practice stays on under it
  assert.equal(p.val("overlay"), "hands");
  assert.equal(p.val("practicing()"), true);
  p.popTo("/gto-solver/"); await flush();
  assert.equal(p.val("overlay"), null);
  assert.equal(p.val("practicing()"), false);
});

test("the menu's range builder opens Study at its URL, asking for its Build mode", async () => {
  const p = page({ path: "/gto-solver/" });
  await p.ready(); await flush();
  p.click("rangeBuilder"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/study/"]);
  assert.equal(p.val("Study.opened"), 1);
  assert.equal(p.val("window.STUDY_BUILD"), true);
});

test("the menu's Preflop solve opens Study at /preflop/ as its game form; Study from the menu after it is Study again", async () => {
  const p = page({ path: "/gto-solver/" });
  await p.ready(); await flush();
  p.click("pfSolve"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/preflop/"]);
  assert.equal(p.val("Study.as"), "preflop");
  assert.equal(p.val("overlay"), "preflop");
  p.click("study"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/preflop/", "/gto-solver/", "/gto-solver/study/"]);   // (as between any two views)
  assert.equal(p.val("Study.as"), "study");
  assert.equal(p.val("Study.opened"), 2);
});

test("the solver's page pins its base to the root, so a view's URL pushed doesn't move its scripts", async () => {
  // (Study opened from the menu at /gto-solver/study/ had loaded /gto-solver/study/study.js, a 404)
  const p = page({ path: "/gto-solver/" });
  await p.ready();
  assert.equal(p.val("document.head.children[0].tagName"), "BASE");
  assert.equal(p.val("document.head.children[0].href"), "/gto-solver/");
});

test("a view's script that fails to load leaves its URL", async () => {
  const p = loadPage({ scripts: { ...SCRIPTS, "session.js": null } });
  await p.ready();
  p.click("analyzeSession"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/session/", "/gto-solver/"]);
  assert.match(p.err(), /Couldn't load the session analyzer/);
});

test("share links built on a view's URL are the solver's, and open there (old links too)", async () => {
  const a = page({ path: "/gto-solver/hands/" });
  await a.ready(); await flush();
  a.S = state({ hero: "CO", cards: ["Ah", "Kd"], preflop: acts("UTG f, HJ f") });
  const link = linkOf(a.run("handText()"));
  assert.match(link, /^https:\/\/www\.michael-han\.com\/gto-solver\/#h=/);
  const hash = link.slice(link.indexOf("#"));
  for (const where of ["/gto-solver/", "/gto-solver/index.html", "/gto-solver/practice/"]) {
    const b = page({ path: where, hash });
    await b.ready(); await flush();
    assert.equal(b.S.hero, "CO", where);
    assert.deepEqual(b.S.cards.slice(0, 2), ["Ah", "Kd"], where);
    assert.equal(b.val("practicing()"), false, where);   // (a link opened at /practice/ shows its spot, in the solver)
    assert.equal(b.location.pathname, "/gto-solver/", where);   // (index.html too: the solver's URL)
    assert.equal(b.location.hash, "");
    assert.deepEqual(b.pushed, [], where);
  }
});

test("the route table is the one gto/routes.py reads, and deploy.sh writes each route's page after the stamps", () => {
  const html = fs.readFileSync(path.join(ROOT, "gto", "static", "index.html"), "utf8");
  const p = page();
  const py = process.env.PYTHON || (fs.existsSync(path.join(ROOT, ".venv", "bin", "python")) ? path.join(ROOT, ".venv", "bin", "python") : "python3");
  const out = execFileSync(py, ["-c", "import json, sys; from gto import routes; print(json.dumps(routes.routes(sys.stdin.read())))"],
    { cwd: ROOT, input: html, encoding: "utf8" });
  const [names, spot] = JSON.parse(out);
  assert.deepEqual(names, p.val("Object.keys(ROUTES)"));
  assert.deepEqual(spot, p.val("SPOT_ROUTES"));
  assert.match(html, /^const ROUTES = \{.*\};$/m);
  // each label is the ⋯ menu's / header's (without its ellipsis)
  for (const [id, r] of [["practiceBtn", "practice"], ["analyzeSession", "session"], ["study", "study"], ["pfSolve", "preflop"], ["importHand", "import"],
                         ["helpStart", "start"], ["helpAbout", "about"], ["helpGlossary", "glossary"], ["helpChangelog", "changelog"], ["helpFairplay", "fairplay"], ["myHands", "hands"]]) {
    const label = html.match(new RegExp(`<button id="${id}"[^>]*>([^<]*)</button>`))[1].replace("…", "");
    assert.equal(p.val(`ROUTES[${JSON.stringify(r)}]`), label, id);
  }
  const deploy = fs.readFileSync(path.join(ROOT, "scripts", "deploy.sh"), "utf8");
  const gen = deploy.indexOf('-m gto.routes "$WORK/site/gto-solver"');
  assert.ok(gen > 0, "deploy.sh doesn't run gto.routes on the site folder");
  // every version constant the page has is stamped, from one list of its modules, before the library is published
  const modules = deploy.match(/^MODULES="\n([\s\S]*?)^"$/m)[1].trim().split("\n").map(l => l.split(/\s+/));
  assert.deepEqual(["WORKER_VERSION", ...modules.map(m => m[0])].sort(), [...html.matchAll(/^const (\w+_VERSION) = /gm)].map(m => m[1]).sort());
  for (const [, ...files] of modules) for (const f of files) assert.ok(fs.existsSync(path.join(ROOT, "gto", "static", f)), f);
  const stamped = Math.max(deploy.indexOf("stamp WORKER_VERSION"), deploy.indexOf('stamp "$constant"'));
  assert.ok(deploy.indexOf("stamp WORKER_VERSION") > 0 && deploy.indexOf('stamp "$constant"') > 0, "deploy.sh doesn't stamp the versions");
  assert.ok(stamped < gen, "the versions are stamped after the routes' pages are copied");
  assert.ok(stamped < deploy.indexOf('/scripts/deploy_library.sh"'), "the library is published before the versions are stamped: a stamp that fails would leave it published");
  const lib = deploy.indexOf('const LIB_BASE = "../gto-solver-library/"');   // (the library's own site, scripts/deploy_library.sh)
  assert.ok(lib > 0 && lib < gen, "LIB_BASE isn't stamped before the routes' pages are copied");
  assert.ok(gen < deploy.indexOf("git add -A"));
});
