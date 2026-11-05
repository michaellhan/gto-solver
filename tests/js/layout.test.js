// The on-screen keyboard offset and copying on iOS.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage } = require("./page.js");

const flush = () => new Promise(r => setImmediate(r));

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
