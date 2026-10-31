// The advice's Breakdown and Equity views (gto/static/breakdown.js, loaded on demand): the request (the advice's state
// with "breakdown": true, asked once per spot), each view's HTML, and the page's tabs and lazy loading.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage } = require("./page.js");

const BD_JS = path.join(__dirname, "..", "..", "gto", "static", "breakdown.js");
const B = require(BD_JS);

// a payload shaped like gto/breakdown.py's breakdown()
const BD = {
  actions: ["check", "bet 33%", "bet 75%"], pot: 5.5, nut_eq: 0.8,
  hero: { who: "BTN", combos: 100, eq: 0.55, ev: 3.6, nuts: 0.2, mine: ["top_tk", "none"],
          curve: [0.95, 0.8, 0.6, 0.4, 0.2],
          made: [{ id: "set", label: "Set", c: 10, eq: 0.95, f: [0, 0.4, 0.6], ev: 13.5 },
                 { id: "top_tk", label: "Top pair, top kicker", c: 20, eq: 0.85, f: [0.1, 0.5, 0.4], ev: 8 },
                 { id: "nothing", label: "No made hand", c: 70, eq: 0.35, f: [0.5, 0.3, 0.2], ev: 1.5 }],
          draws: [{ id: "gutshot", label: "Gutshot", c: 30, eq: 0.4, f: [0.2, 0.5, 0.3], ev: 3 },
                  { id: "none", label: "No draw", c: 70, eq: 0.6, f: [0.4, 0.3, 0.3], ev: 4 }] },
  villains: [{ who: "BB", combos: 80, eq: 0.45, ev: 1.9, nuts: 0.1, curve: [0.9, 0.7, 0.5, 0.3, 0.1],
               made: [{ id: "middle", label: "Middle pair", c: 30, eq: 0.6 }, { id: "nothing", label: "No made hand", c: 50, eq: 0.3 }],
               draws: [{ id: "none", label: "No draw", c: 80, eq: 0.45 }] }],
};
const j = (x) => JSON.parse(JSON.stringify(x));

test("the request: the advice's state with the flag, asked once per spot", async () => {
  B.cache.clear();
  const asked = [];
  const ask = async (st) => { asked.push(st); return { recommended: "check", breakdown: BD }; };
  const key = JSON.stringify({ hero: "BTN", board: "Qs8d3c" });
  const a = await B.request(key, ask), b = await B.request(key, ask);
  assert.deepEqual(asked, [{ hero: "BTN", board: "Qs8d3c", breakdown: true }]);
  assert.equal(a.breakdown, BD); assert.equal(b, a);
  // cancelled or failed: an error, and the next open asks again
  const k2 = JSON.stringify({ hero: "BB" });
  assert.match((await B.request(k2, async () => ({ cancelled: true }))).error, /cancelled/);
  assert.match((await B.request(k2, async () => ({ error: "boom" }))).error, /boom/);
  assert.match((await B.request(JSON.stringify({ x: 1 }), async () => ({ breakdown: null }))).error, /betting rules/);
  assert.equal((await B.request(k2, async () => ({ breakdown: BD }))).breakdown, BD);
});

test("the Breakdown view: the hero's mix per category, the villain's make-up", () => {
  const h = B.breakdownHtml(BD, { who: 0, part: "made", sel: null });
  assert.equal((h.match(/class="bd-row[^"]*" data-row=/g) || []).length, 3);
  assert.match(h, /You \(BTN\)/);
  assert.match(h, /data-row="top_tk"[^>]*>.*?YOU/);        // the hero's own hand
  // bars as long as the share against the largest category, split by the actions
  assert.match(h, /data-row="set".*?class="bd-bar" style="width:14\.3%"><i style="width:40\.0%;background:var\(--accent\)"><\/i><i style="width:60\.0%/);
  assert.match(h, /20\.0%<\/span><span class="n">85%<\/span><span class="n">\+8\.00<\/span>/);
  assert.match(h, /Top pair, top kicker<\/b> · 20\.0 combos \(20\.0%\)/);   // the readout: the hero's row
  const d = B.breakdownHtml(BD, { who: 0, part: "draws", sel: "gutshot" });
  assert.match(d, /data-part="draws" class="active"/);
  assert.match(d, /<b>Gutshot<\/b> · 30\.0 combos \(30\.0%\)/);
  const v = B.breakdownHtml(BD, { who: 1, part: "made", sel: null });
  assert.match(v, /BB isn't the one to act/);
  assert.doesNotMatch(v, /bd-legend|YOU/);
  assert.match(v, /noev/);
  // the river: no draws, no toggle
  const r = B.breakdownHtml({ ...BD, hero: { ...BD.hero, draws: [] } }, { who: 0, part: "draws", sel: null });
  assert.doesNotMatch(r, /data-part=/);
  assert.match(r, /data-row="set"/);
});

test("the Equity view: both curves, averages and the nut advantage; none 3-way", () => {
  const h = B.equityHtml(BD);
  assert.equal((h.match(/<polyline/g) || []).length, 2);
  assert.match(h, /55\.0%.*\+3\.60 bb.*20\.0%/s);
  assert.match(h, /BTN has the range advantage \(55\.0% equity\)\. BTN has the nut advantage: 10\.0 points more/);
  const pts = h.match(/points="([^"]*)"/)[1].split(" ").map(p => p.split(",").map(Number));
  assert.equal(pts.length, 5);
  assert.ok(pts.every((p, i) => !i || (p[0] > pts[i - 1][0] && p[1] >= pts[i - 1][1])), "left to right, falling");
  const mw = { ...BD, villains: [{ ...BD.villains[0], curve: undefined }, { ...BD.villains[0], who: "CO", curve: undefined }] };
  assert.match(B.equityHtml(mw), /heads-up spots/);
});

test("the views say what the equities are over: the river's board, every river card, a sample of the flop's runouts", () => {
  const note = (runouts, view) => {
    const h = view === "made" ? B.breakdownHtml({ ...BD, runouts }, { who: 0, part: "made", sel: null }) : B.equityHtml({ ...BD, runouts });
    return h.match(/<div class="bd-note">([^<]*)<\/div>/)[1];
  };
  assert.match(note([1, 1]), /against the other range \(on the river, no cards to come\)/);
  assert.match(note([48, 48]), /\(every river card dealt out\)/);
  assert.match(note([392, 1176]), /\(over an even sample of 392 of the 1176 turn and river runouts\)/);
  assert.match(note([1176, 1176]), /\(every runout dealt out\)/);
  assert.match(note([1, 1], "made"), /Equity is against the other range, on the river, no cards to come\./);
  assert.match(note([392, 1176], "made"), /Equity is against the other range, over an even sample of 392 of the 1176 turn and river runouts\./);
  assert.doesNotMatch(note(undefined), /every runout/);
  // 3-way (advisor._study_mw_node): against both other ranges, as a share of the pot
  const mw = (runouts) => B.breakdownHtml({ ...BD, runouts, villains: [BD.villains[0], { ...BD.villains[0], who: "CO" }] },
                                          { who: 0, part: "made", sel: null }).match(/<div class="bd-note">([^<]*)<\/div>/)[1];
  assert.match(mw([1176, 1176]), /Equity is against the other ranges \(a share of the pot at showdown\), every runout dealt out\./);
  assert.match(mw([48, 48]), /other ranges \(a share of the pot at showdown\), every river card dealt out\./);
});

const RES = { street: "flop", hero: "BTN", villain: "BB", hand: "AQo", board: "Qs8d3c", recommended: "bet 75%",
  solver: { exploitability_pct: 0.4, iterations: 50, solve_seconds: 2, active_combos: 300, tree_actions: ["check", "bet 33%", "bet 75%"] },
  actions: [{ action: "check", freq: 0, ev_bb: 3 }, { action: "bet 33%", freq: 0, ev_bb: 3.5 }, { action: "bet 75%", freq: 1, ev_bb: 4 }],
  grid: { actions: ["check", "bet 33%", "bet 75%"], cells: { AQo: { w: 1, f: [0, 0, 1] } } } };

function page(server) {
  const p = loadPage({ server });
  p.run("wireGrid = () => {}");   // (the fake DOM has no grid element to wire)
  p.ctx.__res = JSON.stringify(RES);
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  return p;
}

test("the advice panel: Grid / Breakdown / Equity tabs postflop, the grid inside its own view", () => {
  const p = page(), h = p.byId("result").innerHTML;
  assert.match(h, /class="segment vtabs"><button data-v="grid" class="active">Grid<\/button><button data-v="breakdown" class="">Breakdown<\/button><button data-v="equity" class="">Equity<\/button><button data-v="simple" class="">Simple<\/button><\/div><div class="bdview" hidden><\/div><div class="gridview">/);
  // 3-way solves: no equity tab; the betting rules' answers and preflop: no tabs
  p.ctx.__res = JSON.stringify({ ...RES, multiway: true, solved: true, opponents: ["BB", "CO"] });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.match(p.byId("result").innerHTML, /Breakdown<\/button><\/div>/);
  for (const r of [{ ...RES, multiway: true, solved: false, opponents: ["BB", "CO"], equity_vs_field: 0.5 }, { ...RES, street: "preflop" }]) {
    p.ctx.__res = JSON.stringify(r);
    p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
    assert.doesNotMatch(p.byId("result").innerHTML, /vtabs/);
  }
});

// a stand-in for the result panel's elements showView reaches with selectors (the fake DOM doesn't parse HTML)
function panel(p) {
  p.run(`var __box = document.createElement("div"), __grid = document.createElement("div"), __tabs = [];
    for (const v of ["grid", "breakdown", "equity"]) { const b = document.createElement("button"); b.dataset.v = v; __tabs.push(b); }
    var __el = { querySelector: (s) => s === ".bdview" ? __box : s === ".gridview" ? __grid : null, querySelectorAll: () => __tabs };
    var __head = []; document.head = { appendChild: (s) => { __head.push(s); return s; } };`);
}

test("opening Breakdown loads breakdown.js once, then asks the solver with the flag", async () => {
  const p = page(true);
  await p.ready();
  panel(p);
  p.run(`showView(__el, "breakdown", JSON.parse(__res), JSON.stringify(payload()))`);
  assert.deepEqual(p.val("[__grid.hidden, __box.hidden, __tabs.map(b => b.className)]"), [true, false, ["", "active", ""]]);
  assert.equal(p.run("__head.length"), 1);
  assert.equal(p.run("__head[0].src"), "breakdown.js?v=1");
  assert.equal(p.posted.length, 0);                              // nothing asked before the module is in
  vm.runInContext(fs.readFileSync(BD_JS, "utf8"), p.ctx, { filename: "breakdown.js" });
  p.run("__head[0].onload()");
  await new Promise(r => setImmediate(r));
  const st = p.posted.at(-1).state;
  assert.equal(st.breakdown, true);
  assert.deepEqual({ ...st, breakdown: undefined }, { ...p.payload(), breakdown: undefined });
  assert.match(p.run("__box.innerHTML"), /Breaking down/);
  // the Grid tab: back to the grid, nothing loaded or asked
  const n = p.posted.length;
  p.run(`showView(__el, "grid", JSON.parse(__res), JSON.stringify(payload()))`);
  assert.deepEqual(p.val("[__grid.hidden, __box.hidden, __head.length]"), [false, true, 1]);
  assert.equal(p.posted.length, n);
  // the view picked stays: the next advice opens on it (the module already in, so no second script)
  p.run(`showView(__el, "equity", JSON.parse(__res), JSON.stringify(payload()))`);
  assert.equal(p.run("VIEW"), "equity");
  assert.equal(p.run("__head.length"), 1);
});

test("the view renders the answer in the page", async () => {
  const p = page();
  panel(p);
  vm.runInContext(fs.readFileSync(BD_JS, "utf8"), p.ctx, { filename: "breakdown.js" });
  p.ctx.__bd = JSON.stringify(BD);
  await p.run(`window.Breakdown.show(__box, "breakdown", JSON.parse(__res), "{}", async () => ({ breakdown: JSON.parse(__bd) }))`);
  assert.match(p.run("__box.innerHTML"), /data-row="top_tk"/);
  await p.run(`window.Breakdown.show(__box, "equity", JSON.parse(__res), "{}", async () => { throw new Error("cached") })`);
  assert.match(p.run("__box.innerHTML"), /polyline/);
});

test("in Study a hero's category offers the grid filtered to it", () => {
  const got = [];
  const h = B.breakdownHtml(BD, { who: 0, part: "made", sel: "set" }, { filter: (k) => got.push(k) });
  assert.match(h, /<button class="ghost bd-go" data-filter="m:set">Show these hands in the grid<\/button>/);
  assert.match(B.breakdownHtml(BD, { who: 0, part: "draws", sel: "gutshot" }, { filter: () => {} }), /data-filter="d:gutshot"/);
  assert.doesNotMatch(B.breakdownHtml(BD, { who: 0, part: "draws", sel: "none" }, { filter: () => {} }), /data-filter/);
  assert.doesNotMatch(B.breakdownHtml(BD, { who: 1, part: "made", sel: "middle" }, { filter: () => {} }), /data-filter/);
  assert.doesNotMatch(B.breakdownHtml(BD, { who: 0, part: "made", sel: "set" }), /data-filter/);   // (the advice panel)
});
