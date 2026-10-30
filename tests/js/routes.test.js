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

// stand-ins for the views' scripts (the real ones need a DOM): they record open(spot) and report a spot / closing as
// the real ones do
const stub = (g, name, label = (s) => s + " label") => `globalThis.${g} = { opened: [], open(spot) { this.opened.push(spot || null);
  const s = spot || "srp_btn_bb"; return Promise.resolve().then(() => viewSpot("${name}", s, (${label})(s))); },
  close() { this.closed = (this.closed || 0) + 1; viewClosed("${name}"); } };`;
const SCRIPTS = { "reports.js": stub("FlopReports", "reports"), "drills.js": stub("Drills", "drills"),
                  "session.js": `globalThis.Session = { opened: 0, open() { this.opened++; return Promise.resolve(); }, close() { viewClosed("session"); } };`,
                  "study.js": `globalThis.Study = { opened: 0, open() { this.opened++; return Promise.resolve(); }, close() { viewClosed("study"); } };` };
const page = (opts = {}) => loadPage({ scripts: SCRIPTS, ...opts });

test("paths parse to a view and the solver's root", () => {
  const p = page();
  const r = (x) => p.val(`parseRoute(${JSON.stringify(x)})`);
  assert.deepEqual(r("/gto-solver/"), { root: "/gto-solver/", view: "solver", spot: null });
  assert.deepEqual(r("/gto-solver/index.html"), { root: "/gto-solver/", view: "solver", spot: null });
  assert.deepEqual(r("/gto-solver/practice/"), { root: "/gto-solver/", view: "practice", spot: null });
  assert.deepEqual(r("/gto-solver/practice"), { root: "/gto-solver/", view: "practice", spot: null });
  assert.deepEqual(r("/gto-solver/practice/index.html"), { root: "/gto-solver/", view: "practice", spot: null });
  assert.deepEqual(r("/gto-solver/reports/3bp-bb-btn/"), { root: "/gto-solver/", view: "reports", spot: "3bp_bb_btn" });
  assert.deepEqual(r("/gto-solver/drills/srp-co-bb/"), { root: "/gto-solver/", view: "drills", spot: "srp_co_bb" });
  assert.deepEqual(r("/gto-solver/session/"), { root: "/gto-solver/", view: "session", spot: null });
  assert.deepEqual(r("/"), { root: "/", view: "solver", spot: null });                    // the local server
  assert.deepEqual(r("/reports/srp-btn-bb/"), { root: "/", view: "reports", spot: "srp_btn_bb" });
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

test("the flop reports: opening pushes /reports/, the spot shown replaces it, closing pushes the solver's", async () => {
  const p = page();
  await p.ready();
  p.click("flopReports"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/reports/"]);
  assert.deepEqual(p.replaced, ["/gto-solver/reports/srp-btn-bb/"]);
  assert.equal(p.ctx.document.title, "srp_btn_bb label · Flop reports · GTO Solver");
  p.run("viewSpot('reports', '3bp_sb_btn', 'SB 3bet vs BTN')");   // another spot picked in the panel
  assert.equal(p.location.pathname, "/gto-solver/reports/3bp-sb-btn/");
  assert.equal(p.pushed.length, 1);
  p.run("FlopReports.close()");
  assert.deepEqual(p.pushed, ["/gto-solver/reports/", "/gto-solver/"]);
  assert.equal(p.ctx.document.title, "GTO Solver");
  p.popTo("/gto-solver/reports/srp-co-bb/"); await flush();         // back: open again, on that spot
  assert.deepEqual(p.val("FlopReports.opened"), [null, "srp_co_bb"]);
  assert.equal(p.val("overlay"), "reports");
  p.popTo("/gto-solver/"); await flush();
  assert.equal(p.val("FlopReports.closed"), 2);
  assert.equal(p.val("overlay"), null);
  assert.equal(p.pushed.length, 2);
});

test("a view's URL opened directly opens it, on its spot", async () => {
  const r = page({ path: "/gto-solver/reports/3bp-bb-btn/" });
  await r.ready(); await flush();
  assert.deepEqual(r.val("FlopReports.opened"), ["3bp_bb_btn"]);
  assert.equal(r.location.pathname, "/gto-solver/reports/3bp-bb-btn/");
  assert.deepEqual(r.pushed, []);
  const d = page({ path: "/gto-solver/drills/srp-co-bb/" });
  await d.ready(); await flush();
  assert.deepEqual(d.scripts, ["reports.js", "drills.js"]);
  assert.deepEqual(d.val("Drills.opened"), ["srp_co_bb"]);
  assert.equal(d.ctx.document.title, "srp_co_bb label · Drills · GTO Solver");
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
  p.click("drills"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/drills/"]);
  assert.equal(p.location.pathname, "/gto-solver/drills/srp-btn-bb/");
  p.run("Drills.close()");
  assert.deepEqual(p.pushed, ["/gto-solver/drills/", "/gto-solver/practice/"]);
  assert.equal(p.val("practicing()"), true);
  p.popTo("/gto-solver/drills/srp-btn-bb/"); await flush();   // back to the drills: Practice stays on under them
  assert.equal(p.val("overlay"), "drills");
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
  const a = page({ path: "/gto-solver/reports/srp-btn-bb/" });
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
  for (const [id, r] of [["practiceBtn", "practice"], ["drills", "drills"], ["analyzeSession", "session"], ["study", "study"], ["flopReports", "reports"], ["importHand", "import"]]) {
    const label = html.match(new RegExp(`<button id="${id}"[^>]*>([^<]*)</button>`))[1].replace("…", "");
    assert.equal(p.val(`ROUTES[${JSON.stringify(r)}]`), label, id);
  }
  const deploy = fs.readFileSync(path.join(ROOT, "scripts", "deploy.sh"), "utf8");
  const gen = deploy.indexOf('-m gto.routes "$WORK/site/gto-solver"');
  assert.ok(gen > 0, "deploy.sh doesn't run gto.routes on the site folder");
  for (const v of ["WORKER", "HH", "RP", "DR", "BD", "SS", "SY", "RO", "SP", "MT"]) assert.ok(deploy.indexOf(`const ${v}_VERSION = \\"$`) < gen, `${v}_VERSION is stamped after the routes' pages are copied`);
  const lib = deploy.indexOf('const LIB_BASE = "../gto-solver-library/"');   // (the library's own site, scripts/deploy_library.sh)
  assert.ok(lib > 0 && lib < gen, "LIB_BASE isn't stamped before the routes' pages are copied");
  assert.ok(gen < deploy.indexOf("git add -A"));
});
