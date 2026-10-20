// The setup's Advanced disclosure: custom postflop bet sizes (S.sizes, % of the pot), ranges entering the flop
// by seat (S.ranges) and size optimization (S.optSizes); sent as sizes / ranges / optimize_sizes from the flop
// on, kept with the hand and by share links, and the villain's range grid in the advice's Details.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage, state, acts } = require("./page.js");

const j = (x) => JSON.parse(JSON.stringify(x));
const linkOf = (text) => text.match(/^Open this spot: (.*)$/m)[1];
const FLOP = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") };
const ADV = { sizes: { flop: [33, 125], raise: [60] }, ranges: { BB: "22+,A2s+,KTo+" }, optSizes: true };
const rangeBoxes = (p) => [...p.byId("rangeIns").walk()].filter(e => e.tagName === "INPUT");

test("unset by default: nothing sent, empty boxes", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  for (const k of ["sizes", "ranges", "optimize_sizes"]) assert.equal(k in p.payload(), false, k);
  assert.equal(p.byId("sz_flop").value, "");
  assert.equal(p.byId("optSizes").checked, false);
  assert.deepEqual(rangeBoxes(p).map(i => i.dataset.p), ["UTG", "HJ", "CO", "SB", "BB"]);   // everyone but you
  assert.equal(p.byId("adv").open, false);
});

test("the inputs set the state and the payload; sent from the flop on only", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  p.byId("sz_flop").onchange({ target: { value: "33, 125% 33 7000 x" } });   // duplicates, junk and > 500% dropped
  p.byId("sz_raise").onchange({ target: { value: "60" } });
  p.byId("sz_turn").onchange({ target: { value: "10 20 30 40" } });          // at most 3
  p.byId("optSizes").onchange({ target: { checked: true } });
  rangeBoxes(p).find(i => i.dataset.p === "BB").onchange({ target: { value: " 22+,A2s+,KTo+ " } });
  assert.deepEqual(j(p.S.sizes), { flop: [33, 125], turn: [10, 20, 30], raise: [60] });
  assert.deepEqual(p.payload().sizes, { flop: [33, 125], turn: [10, 20, 30], raise: [60] });
  assert.deepEqual(p.payload().ranges, { BB: "22+,A2s+,KTo+" });
  assert.equal(p.payload().optimize_sizes, true);
  assert.equal(JSON.parse(p.storage.get("handreview")).optSizes, true);   // saved with the hand
  assert.equal(p.byId("sz_flop").value, "33 125");
  assert.equal(rangeBoxes(p).find(i => i.dataset.p === "BB").value, "22+,A2s+,KTo+");
  // clearing a box goes back to the default
  p.byId("sz_turn").onchange({ target: { value: "" } });
  p.byId("optSizes").onchange({ target: { checked: false } });
  assert.deepEqual(j(p.S.sizes), { flop: [33, 125], raise: [60] });
  assert.equal("optSizes" in p.S, false);
  assert.equal("optimize_sizes" in p.payload(), false);
  // preflop the solve doesn't use them, so they aren't sent (the preflop advice and its cache stay the same)
  p.S = { ...state({ cards: ["Ah", "Kh"] }), ...ADV };
  for (const k of ["sizes", "ranges", "optimize_sizes"]) assert.equal(k in p.payload(), false, k);
  // your own seat's range isn't sent (its box hides while you sit there)
  p.S = { ...state(FLOP), ranges: { BTN: "AA", BB: "KK" } };
  assert.deepEqual(p.payload().ranges, { BB: "KK" });
});

test("a change drops the shown advice; kept by a new hand, and by a change of players for seats still there", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  p.byId("result").className = "advice"; p.byId("result").innerHTML = "old advice";
  p.byId("sz_flop").onchange({ target: { value: "50" } });
  assert.equal(p.byId("result").className, "empty");
  p.S = { ...state({ ...FLOP, hero: "BB" }), ...ADV, ranges: { UTG1: "AA", BTN: "QQ+" } };
  p.byId("newhand").onclick();
  assert.deepEqual(j(p.S.sizes), ADV.sizes); assert.equal(p.S.optSizes, true);
  p.run('document.getElementById("players").children[7].onclick()');   // 9 players
  assert.deepEqual(j(p.S.ranges), { BTN: "QQ+" }) ;
  p.run('document.getElementById("players").children[1].onclick()');   // 3 players: BTN SB BB
  assert.deepEqual(j(p.S.ranges), { BTN: "QQ+" });
  p.run('document.getElementById("players").children[0].onclick()');   // heads-up: no BTN seat
  assert.equal("ranges" in p.S, false);
  assert.deepEqual(j(p.S.sizes), ADV.sizes);
});

test("share link round trip, the summary and the copied hand say so", async () => {
  const a = loadPage();
  a.S = { ...state(FLOP), ...ADV };
  const text = a.run("handText()");
  const b = loadPage({ hash: linkOf(text).replace(/^[^#]*/, "") });
  assert.deepEqual(j(b.S), j(a.S));
  await a.ready(); await b.ready();
  assert.deepEqual(b.payload(), a.payload());
  assert.equal(b.byId("adv").open, true);
  assert.match(text.split("\n")[0], /, custom sizes, custom BB range, optimized sizes, I am BTN/);
  b.run("editSetup = false; renderAll()");
  assert.match(b.byId("setupSum").innerHTML, /custom sizes/);
});

test("bad Advanced fields in a link or storage are dropped", () => {
  const st = { ...state(FLOP), sizes: { flop: ["x", -5, 33], turn: "75", river: [900], bogus: [50] }, ranges: { BB: 7, XX: "AA", CO: "  ", SB: "KK+" }, optSizes: "yes" };
  const p = loadPage({ hash: "#h=" + Buffer.from(JSON.stringify(st)).toString("base64") });
  assert.deepEqual(j(p.S.sizes), { flop: [33], turn: [75] });
  assert.deepEqual(j(p.S.ranges), { SB: "KK+" });
  assert.equal("optSizes" in p.S, false);
  const p2 = loadPage({ storage: { handreview: JSON.stringify({ ...st, sizes: "33", ranges: ["AA"] }) } });
  assert.equal("sizes" in p2.S, false); assert.equal("ranges" in p2.S, false);
});

test("the villain's range grid and the size usage render in Details", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  const cells = { AA: { w: 1, f: [1] }, AKs: { w: 0.5, f: [1] }, "72o": { w: 0, f: [0] } };
  p.ctx.__res = JSON.stringify({ street: "flop", hero: "BTN", villain: "BB", hand: "AKs", recommended: "check", villain_range: "AA, AKs:0.50",
    villain_range_combos: 8, villain_grid: { actions: ["in range"], cells }, size_usage: { check: 0.2, "33%": 0.5, "125%": 0.3 }, sizes_kept: ["33%", "125%"],
    actions: [{ action: "check", freq: 1, ev_bb: 1 }] });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  const h = p.byId("result").innerHTML;
  assert.match(h, /<div class="vgrid">.*BB's range by hand/s);
  assert.equal((h.match(/class="cell/g) || []).length, 169);   // one grid (no hero grid in this result)
  assert.match(h, /data-h="AKs" style="opacity:0.625"/);
  assert.match(h, /check 20% · 33% 50% · 125% 30%; kept 33%, 125%/);
  assert.match(p.run("handText()"), /size optimization: check 20% · 33% 50% · 125% 30%; kept 33%, 125%/);
});

test("the advisor reads the Advanced keys the page sends", () => {
  const src = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "advisor.py"), "utf8");
  const body = src.slice(src.indexOf("def _advise("), src.indexOf("def _preflop("));
  for (const k of ["sizes", "ranges", "optimize_sizes"]) assert.ok(body.includes(`state.get("${k}")`), k);
});
