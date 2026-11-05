// About (gto/static/help.js): its URL, the header's About and Lizard, and that its statements are what the page does.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage } = require("./page.js");
const Help = require("../../gto/static/help.js");

const HTML = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "static", "index.html"), "utf8");
const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };

// a stand-in for help.js (the real one needs a DOM), reporting closing as it does
const SCRIPTS = {
  "help.js": `globalThis.Help = { opens: 0, isOpen: false, open() { this.opens++; this.isOpen = true; },
    close() { this.isOpen = false; viewClosed("about"); } };`,
};
const page = (opts = {}) => loadPage({ scripts: SCRIPTS, ...opts });

test("About builds, its title is its URL's, and Fair play has gone", () => {
  assert.ok(Help.aboutHtml().length > 800);
  assert.equal(page().val("ROUTES.about"), "About");
  assert.equal(page().val("ROUTES.fairplay"), undefined);
  assert.doesNotMatch(Help.aboutHtml(), /Fair play|data-page/);
});

test("the pages say what the page does: the source line's and the grades' thresholds are index.html's", () => {
  const p = page();
  const conv = (x) => p.val(`convergence({ solver: { exploitability_pct: ${x} } }).cls`);
  assert.deepEqual([conv(2.99), conv(3), conv(9.99), conv(10)], ["ok", "meh", "meh", "bad"]);
  const about = Help.aboutHtml();
  assert.match(about, /source line gives it: under 3% is close to a full solver, and above that\s+the line says the solve hasn't settled/);
  const grade = (g) => p.val(`gradeClass(${JSON.stringify(g)})`);
  assert.deepEqual([grade({ f: 0.15 }), grade({ f: 0.149 }), grade({ f: 0.03 }), grade({ f: 0.029 }), grade({ f: 0.01, r: 1 })], ["ok", "meh", "meh", "bad", "meh"]);
  assert.match(about, /✓ \(the solver's mix takes it at least 15% of the time\), ~ \(3-15%\) or ✗ \(under 3%\); a rough solve, or a line that gave up no EV, never grades ✗/);
  assert.deepEqual([p.val("HEUR_OK"), p.val("HEUR_MEH")], [3, 10]);
  // the times are the page's estimate: the street's budget plus a narrowing budget for each earlier postflop street
  const [own, prior] = p.val("BUDGETS").wasm;
  assert.match(Help.aboutHtml(), new RegExp(`a live flop takes about ${own} s a decision, a turn ${own + prior} s and a river ${own + 2 * prior} s`));
  assert.match(about, /within 3% of the pot of the best is ✓, within 10% is ~/);
  assert.match(about, /postflop-solver/);
});

test("About keeps its statements: what stays on the device, the way out, the frozen copy and Lizard", () => {
  const about = Help.aboutHtml();
  assert.match(about, /nothing you enter is uploaded/);
  assert.match(about, /jsDelivr/);
  assert.match(about, /GitHub Pages/);
  assert.match(about, /Back up everything/);
  assert.match(about, /Restore/);
  assert.match(about, /Delete everything this site keeps/);
  assert.match(about, /href="\/gto-solver-classic\/"/);
  assert.match(about, /href="\/lizard\/"/);
  // what left isn't described as here
  for (const word of [/Accurate/, /Pool/, /Practice mode/, /Glossary/, /Getting started/, /Advanced/]) assert.doesNotMatch(about, word);
});

test("the header: Hands, About, Lizard, and no menu", () => {
  const head = HTML.slice(HTML.indexOf("<header>"), HTML.indexOf("</header>"));
  assert.match(head, /<button id="myHands"[^>]*>Hands<\/button>/);
  assert.match(head, /<button id="helpAbout"[^>]*>About<\/button>/);
  assert.match(head, /<a href="\/lizard\/">Lizard →<\/a>/);
  assert.deepEqual([...head.matchAll(/<(?:button id="(\w+)"|a href)/g)].map(m => m[1]), ["myHands", "helpAbout", undefined]);
  assert.doesNotMatch(HTML, /id="menu|id="welcome|data-term/);
});

test("About's URL opens it; back and forward close and open it", async () => {
  const p = page({ path: "/gto-solver/about/" });
  await p.ready(); await flush();
  assert.deepEqual(p.scripts, ["help.js"]);
  assert.equal(p.val("Help.opens"), 1);
  assert.equal(p.ctx.document.title, "About · GTO Solver");
  assert.deepEqual(p.pushed, []);
  p.popTo("/gto-solver/"); await flush();
  assert.equal(p.val("overlay"), null);
  assert.equal(p.val("Help.isOpen"), false);
  p.popTo("/gto-solver/about/"); await flush();
  assert.equal(p.val("overlay"), "about");
  assert.equal(p.val("Help.isOpen"), true);
  assert.deepEqual(p.pushed, []);
});

test("the header's About opens it and closing goes back to the solver's URL", async () => {
  const p = page();
  await p.ready(); await flush();
  p.click("helpAbout"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/about/"]);
  assert.equal(p.val("Help.opens"), 1);
  p.run("Help.close()");
  assert.deepEqual(p.pushed, ["/gto-solver/about/", "/gto-solver/"]);
});

test("a first visit opens on the page itself, with nothing in the way", async () => {
  const p = page();
  await p.ready(); await flush();
  assert.deepEqual(p.pushed, []);
  assert.match(p.byId("result").innerHTML, /Finish the hand to see how each decision compares with the solver\./);
  assert.equal(p.storage.has("gto-welcome"), false);
});
