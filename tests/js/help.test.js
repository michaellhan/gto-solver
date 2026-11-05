// About and Fair play (gto/static/help.js): their URLs, the header's About and Lizard, and that each page's statements
// are what the page does.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage } = require("./page.js");
const Help = require("../../gto/static/help.js");

const HTML = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "static", "index.html"), "utf8");
const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };

// a stand-in for help.js (the real one needs a DOM), reporting a link followed and closing as it does
const SCRIPTS = {
  "help.js": `globalThis.Help = { opened: [], page: null, open(page) { this.opened.push(page); this.page = page; },
    show(page) { if (page !== this.page) { this.page = page; viewSwitch(page); } },
    close() { const p = this.page; this.page = null; viewClosed(p); }, get shown() { return this.page; } };`,
};
const page = (opts = {}) => loadPage({ scripts: SCRIPTS, ...opts });

test("About and Fair play build, each page's title is its URL's, and each links to the other", () => {
  assert.deepEqual(Help.PAGES.map(([p]) => p), ["about", "fairplay"]);
  const p = page();
  for (const [id, label] of Help.PAGES) {
    const h = Help.pageHtml(id);
    assert.ok(h.length > 800, id);
    assert.equal(p.val(`ROUTES[${JSON.stringify(id)}]`), label, id);
    for (const [, g] of h.matchAll(/data-page="([^"]+)"/g)) assert.ok(Help.PAGES.some(([x]) => x === g), `${id}: no page ${g}`);
  }
  assert.match(Help.aboutHtml(), /data-page="fairplay"/);
  assert.match(Help.fairplayHtml(), /data-page="about"/);
  assert.equal(Help.pageHtml("nope"), Help.pageHtml("about"));
});

test("the pages say what the page does: the badge's and the grades' thresholds are index.html's", () => {
  const p = page();
  const conv = (x) => p.val(`convergence({ solver: { exploitability_pct: ${x} } }).label`);
  assert.deepEqual([conv(2.99), conv(3), conv(9.99), conv(10)], ["converged", "fair", "fair", "rough"]);
  const about = Help.aboutHtml();
  assert.match(about, /converged<\/span> under 3%, <span class="hp-b meh">fair<\/span> under 10%/);
  const grade = (g) => p.val(`gradeClass(${JSON.stringify(g)})`);
  assert.deepEqual([grade({ f: 0.15 }), grade({ f: 0.149 }), grade({ f: 0.03 }), grade({ f: 0.029 }), grade({ f: 0.01, r: 1 })], ["ok", "meh", "meh", "bad", "meh"]);
  assert.match(about, /✓ \(the solver's mix takes it at least 15% of the time\), ~ \(3-15%\) or ✗ \(under 3%\); a rough solve, or a line that gave up no EV, never grades ✗/);
  assert.deepEqual([p.val("HEUR_OK"), p.val("HEUR_MEH")], [3, 10]);
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
  const fair = Help.fairplayHtml();
  assert.match(fair, /Do not use it during a hand you are playing, online or live/);
  assert.match(fair, /gets no advice until it's finished/);
  assert.match(fair, /no screen capture, table reading or connection\s+to any poker client/);
  // what left isn't described as here
  for (const word of [/Accurate/, /Pool/, /Practice mode/, /Glossary/, /Getting started/, /Advanced/]) assert.doesNotMatch(about + fair, word);
});

test("the header: Hands, About, Lizard, and no menu", () => {
  const head = HTML.slice(HTML.indexOf("<header>"), HTML.indexOf("</header>"));
  assert.match(head, /<button id="myHands"[^>]*>Hands<\/button>/);
  assert.match(head, /<button id="helpAbout"[^>]*>About<\/button>/);
  assert.match(head, /<a href="\/lizard\/">Lizard →<\/a>/);
  assert.deepEqual([...head.matchAll(/<(?:button id="(\w+)"|a href)/g)].map(m => m[1]), ["myHands", "helpAbout", undefined]);
  assert.doesNotMatch(HTML, /id="menu|id="welcome|data-term/);
});

test("a page's URL opens it; a link moves the URL without closing; back and forward switch pages", async () => {
  const p = page({ path: "/gto-solver/about/" });
  await p.ready(); await flush();
  assert.deepEqual(p.scripts, ["help.js"]);
  assert.deepEqual(p.val("Help.opened"), ["about"]);
  assert.equal(p.ctx.document.title, "About · GTO Solver");
  assert.deepEqual(p.pushed, []);
  p.run("Help.show('fairplay')");                 // a link followed
  assert.deepEqual(p.pushed, ["/gto-solver/fairplay/"]);
  assert.equal(p.val("overlay"), "fairplay");
  assert.equal(p.ctx.document.title, "Fair play · GTO Solver");
  assert.deepEqual(p.val("Help.opened"), ["about"]);   // (not closed and opened again)
  p.popTo("/gto-solver/about/"); await flush();   // back
  assert.equal(p.val("overlay"), "about");
  assert.equal(p.val("Help.page"), "about");
  p.popTo("/gto-solver/"); await flush();
  assert.equal(p.val("overlay"), null);
  assert.equal(p.val("Help.page"), null);
  assert.deepEqual(p.pushed, ["/gto-solver/fairplay/"]);
});

test("the header's About opens it and closing goes back to the solver's URL", async () => {
  const p = page();
  await p.ready(); await flush();
  p.click("helpAbout"); await flush();
  assert.deepEqual(p.pushed, ["/gto-solver/about/"]);
  assert.deepEqual(p.val("Help.opened"), ["about"]);
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
