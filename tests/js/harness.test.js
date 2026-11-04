// The harness itself (page.js): it loads the page and nothing else, and it can't make up an element, which is what used
// to hide broken wiring: a removed element's unguarded line throws in a browser and passed here.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { loadPage, pageScript, HTML_PATH, RUNTIME_IDS, staticIds, unknownIds } = require("./page.js");

const STATIC = path.dirname(HTML_PATH);
const sources = () => fs.readdirSync(STATIC).filter(f => f.endsWith(".js")).map(f => [f, fs.readFileSync(path.join(STATIC, f), "utf8")]).concat([["index.html", pageScript()]]);

test("the page loads with only index.html beside it: no module is read at load", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gto-page-"));
  try {
    fs.copyFileSync(HTML_PATH, path.join(dir, "index.html"));
    const code = `const { loadPage } = require(${JSON.stringify(path.join(__dirname, "page.js"))}); (async () => { const p = loadPage(); await p.ready(); console.log(p.val("S.players")); })();`;
    const out = execFileSync(process.execPath, ["-e", code], { env: { ...process.env, GTO_PAGE: path.join(dir, "index.html") }, encoding: "utf8" });
    assert.equal(out.trim(), "6");
  } finally { fs.rmSync(dir, { recursive: true }); }
  assert.doesNotMatch(fs.readFileSync(path.join(__dirname, "page.js"), "utf8"), /require\("\.\/practice_src/);
});

test("exactly one bare inline script", () => {
  assert.equal([...fs.readFileSync(HTML_PATH, "utf8").matchAll(/<script>/g)].length, 1);
  assert.ok(pageScript().length > 1000);   // (pageScript throws on any other number)
});

test("an id that isn't the page's is a failure, not an element, even inside the page's own try/catch", async () => {
  const p = loadPage();
  await p.ready();
  assert.throws(() => p.run(`document.getElementById("notThere")`), /no element with id "notThere"/);
  assert.throws(() => p.byId("notThere"), /no element with id "notThere"/);
  assert.equal(p.run(`(() => { try { document.getElementById("notThere"); return "swallowed"; } catch (e) { return "swallowed"; } })()`), "swallowed");
  assert.deepEqual(unknownIds.splice(0), ["notThere", "notThere", "notThere"]);   // (what fails a test whose page swallowed it; emptied: this one meant it)
});

test("a lookup finds what's there, as a browser would, and null for a listed id that isn't there yet", async () => {
  const p = loadPage();
  await p.ready();
  assert.equal(p.run(`document.getElementById("newhand").tagName`), "DIV");            // (in the markup)
  assert.equal(p.run(`document.getElementById("solveMsg")`), null);                    // (listed; not rendered yet)
  assert.equal(p.run(`document.getElementById("mhStyle")`), null);                     // (so a module's "is my style in yet?" is answered no)
  p.run(`document.getElementById("result").innerHTML = '<span id="solveMsg">Solving</span>'`);
  assert.ok(p.run(`document.getElementById("solveMsg")`));                             // (written into an element)
  p.run(`document.getElementById("result").innerHTML = ""`);
  assert.equal(p.run(`document.getElementById("solveMsg")`), null);                    // (and gone with that HTML)
  p.run(`var __c = document.createElement("div"); __c.id = "customBox"`);
  assert.ok(p.run(`document.getElementById("customBox") === __c`));                    // (the very element given the id)
  p.run(`__c.remove()`);
  assert.equal(p.run(`document.getElementById("customBox")`), null);
});

test("every id gto/static looks up is in the page's markup or RUNTIME_IDS", () => {
  const known = new Set([...staticIds(), ...Object.values(RUNTIME_IDS).flat()]);
  for (const [f, src] of sources())
    for (const m of src.matchAll(/getElementById\(\s*["']([^"']+)["']\s*\)/g)) {
      if (m[1].endsWith("_")) continue;   // (a prefix: "sz_" + k, the id in the markup)
      assert.ok(known.has(m[1]), `${f} looks up "${m[1]}", which is neither in index.html's markup nor in RUNTIME_IDS (tests/js/page.js)`);
    }
});

test("RUNTIME_IDS has no leftovers: each entry is still created by its file, and none is in the markup", () => {
  const src = new Map(sources()), inMarkup = staticIds();
  for (const [f, ids] of Object.entries(RUNTIME_IDS)) {
    assert.ok(src.has(f), `RUNTIME_IDS lists ${f}, which is gone: its ids go with it (tests/js/page.js)`);
    for (const id of ids) {
      assert.ok(src.get(f).includes(id.replace(/\d$/, "")), `RUNTIME_IDS: ${f} no longer has "${id}"`);
      assert.ok(!inMarkup.has(id), `RUNTIME_IDS: "${id}" is in the markup`);
    }
  }
});
