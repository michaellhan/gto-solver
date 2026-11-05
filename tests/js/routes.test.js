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

// stand-ins for the views' scripts (the real ones need a DOM): they record opening and report closing as the real ones do
const SCRIPTS = { "hands.js": `globalThis.MyHands = { opened: 0, closed: 0, open() { this.opened++; return Promise.resolve(); }, close() { this.closed++; viewClosed("hands"); } };` };
const page = (opts = {}) => loadPage({ scripts: SCRIPTS, ...opts });

test("paths parse to a view and the solver's root", () => {
  const p = page();
  const r = (x) => p.val(`parseRoute(${JSON.stringify(x)})`);
  assert.deepEqual(r("/gto-solver/"), { root: "/gto-solver/", view: "solver" });
  assert.deepEqual(r("/gto-solver/index.html"), { root: "/gto-solver/", view: "solver" });
  assert.deepEqual(r("/gto-solver/hands/"), { root: "/gto-solver/", view: "hands" });
  assert.deepEqual(r("/gto-solver/hands"), { root: "/gto-solver/", view: "hands" });
  assert.deepEqual(r("/gto-solver/hands/index.html"), { root: "/gto-solver/", view: "hands" });
  assert.deepEqual(r("/"), { root: "/", view: "solver" });                    // the local server
});

test("the solver's URL: nothing pushed", async () => {
  const p = page();
  await p.ready(); await flush();
  assert.deepEqual(p.pushed, []);
  assert.deepEqual(p.replaced, []);
  assert.equal(p.ctx.document.title, "GTO Solver");
});

test("the query string stays on every view's URL", async () => {
  const p = page({ search: "?debug=1" });
  await p.ready();
  await p.click("myHands"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/hands/?debug=1"]);
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
});

test("the solver's page pins its base to the root, so a view's URL pushed doesn't move its scripts", async () => {
  // (a view's script would load from under its URL, /gto-solver/hands/hands.js, a 404)
  const p = page({ path: "/gto-solver/" });
  await p.ready();
  assert.equal(p.val("document.head.children[0].tagName"), "BASE");
  assert.equal(p.val("document.head.children[0].href"), "/gto-solver/");
});

test("a view's script that fails to load leaves its URL", async () => {
  const p = loadPage({ scripts: { ...SCRIPTS, "hands.js": null } });
  await p.ready();
  p.click("myHands"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/hands/", "/gto-solver/"]);
  assert.match(p.err(), /Couldn't load My hands/);
});

test("share links built on a view's URL are the solver's, and open there (old links too)", async () => {
  const a = page({ path: "/gto-solver/hands/" });
  await a.ready(); await flush();
  a.S = state({ hero: "CO", cards: ["Ah", "Kd"], preflop: acts("UTG f, HJ f") });
  const link = a.run("shareLink()");
  assert.match(link, /^https:\/\/www\.michael-han\.com\/gto-solver\/#h=/);
  const hash = link.slice(link.indexOf("#"));
  for (const where of ["/gto-solver/", "/gto-solver/index.html"]) {
    const b = page({ path: where, hash });
    await b.ready(); await flush();
    assert.equal(b.S.hero, "CO", where);
    assert.deepEqual(b.S.cards.slice(0, 2), ["Ah", "Kd"], where);
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
  for (const [id, r] of [["helpStart", "start"], ["helpAbout", "about"], ["helpGlossary", "glossary"], ["helpChangelog", "changelog"], ["helpFairplay", "fairplay"], ["myHands", "hands"]]) {
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
