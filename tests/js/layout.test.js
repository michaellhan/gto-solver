// The page's layout code (folding panels, the on-screen keyboard offset) and copying on iOS.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage } = require("./page.js");

const flush = () => new Promise(r => setImmediate(r));
const title = (p, id) => p.byId(id).querySelector("h2[data-fold]");
const press = (el, key) => { let stopped = false; for (const f of el.listeners.keydown || []) f({ key, target: el, preventDefault() {}, stopPropagation() { stopped = true; } }); return stopped; };

test("a panel folds from the keyboard as well as by a click, and says whether it's open", () => {
  const p = loadPage({ layout: true });
  const h = title(p, "pTable");
  assert.equal(h["aria-expanded"], "true");
  assert.equal(press(h, "Enter"), true);
  assert.ok(p.byId("pTable").classList.contains("collapsed"));
  assert.equal(h["aria-expanded"], "false");
  assert.deepEqual(JSON.parse(p.storage.get("gto-layout")).folded, { pTable: true });
  press(h, " ");
  assert.ok(!p.byId("pTable").classList.contains("collapsed"));
  assert.equal(h["aria-expanded"], "true");
  // other keys are left to the page's shortcuts
  assert.equal(press(h, "n"), false);
  assert.ok(!p.byId("pTable").classList.contains("collapsed"));
  for (const f of title(p, "pHistory").listeners.click) f({ target: title(p, "pHistory") });
  assert.ok(p.byId("pHistory").classList.contains("collapsed"));
  // a folded panel stays folded on reload
  const q = loadPage({ layout: true, storage: { "gto-layout": p.storage.get("gto-layout") } });
  assert.ok(q.byId("pHistory").classList.contains("collapsed"));
  assert.equal(title(q, "pHistory")["aria-expanded"], "false");
});

// Bug: on iOS Safari the keyboard shrinks only the visual viewport, so the action dock (stuck to the bottom of
// the layout viewport) sat under the keyboard while a custom bet size was typed.
test("the action dock is lifted above the on-screen keyboard", () => {
  const p = loadPage({ layout: true });
  assert.equal(p.rootProps["--kb"], undefined);
  p.setViewport({ height: 470 });                     // keyboard up: 330px of the 800px layout viewport covered
  assert.equal(p.rootProps["--kb"], "330px");
  p.setViewport({ offsetTop: 200 });                  // the browser scrolled the visual viewport down by 200px
  assert.equal(p.rootProps["--kb"], "130px");
  p.setViewport({ height: 800, offsetTop: 0 });       // keyboard down
  assert.equal(p.rootProps["--kb"], "0px");
  p.setViewport({ height: 400, scale: 2 });           // pinch zoom, not a keyboard
  assert.equal(p.rootProps["--kb"], "0px");
  p.setViewport({ height: 780, scale: 1 });           // a toolbar's worth: not a keyboard
  assert.equal(p.rootProps["--kb"], "0px");
});

// Bug: iOS Safari ignores select() on a textarea, so the execCommand fallback copied nothing there.
test("the execCommand fallback selects the text the way iOS needs", async () => {
  const p = loadPage({ clipboard: "denied", iosSelect: true });
  await p.ready();
  await p.click("share");
  assert.equal(p.clipboard.via, "execCommand");
  assert.equal(p.clipboard.text, p.run("shareLink()"));
  assert.deepEqual(p.clipboard.prompts, []);
  await flush();
});
