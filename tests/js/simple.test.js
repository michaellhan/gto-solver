// The one-size view (gto/static/simple.js, loaded on demand: the advice's Simple tab and Study's): a spot's run (the
// full menu first, then each size, one request a solve), its verdict (the best size, what each gives up), stopping
// and carrying on, the panel's HTML, and the page's tab, loading and request.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage } = require("./page.js");

const SP_JS = path.join(__dirname, "..", "..", "gto", "static", "simple.js");
const S = require(SP_JS);

const grid = (n) => ({ actions: n, cells: { AKs: { w: 1, f: n.map((_, i) => (i === n.length - 1 ? 1 : 0)) } } });
const row = (size, ev, x = 0.05) => ({ size, actions: ["check", `bet ${size}`], freq: [0.4, 0.6], ev_bb: ev, iterations: 300,
                                      exploitability_pct: x, seconds: 0.5, grid: grid(["check", `bet ${size}`]) });
const FULL = { ...row("full", 3.0), actions: ["check", "bet 33%", "bet 75%"], freq: [0.4, 0.2, 0.4], grid: grid(["check", "bet 33%", "bet 75%"]) };
const META = { street: "turn", actor: "BTN", opponent: "BB", position: "IP", board: "Ks7d2cTh", pot_bb: 10, stack_bb: 95,
               candidates: ["33%", "75%", "150%"], asked: "full", definition: "best = ...", target_pct: 0.1 };
const EV = { "33%": 2.9, "75%": 2.99, "150%": 2.95 };
// a solver answering one_size requests: each its row and every row asked before (as the advisor keeps them)
function solver() {
  const asked = [], kept = {};
  const ask = async (st) => {
    asked.push(st.one_size);
    if (st.one_size === "full") kept.full = FULL; else kept[st.one_size] = row(st.one_size, EV[st.one_size]);
    const { full, ...rows } = kept;
    return { one_size: { ...META, asked: st.one_size, full: full || null, rows: Object.values(rows) } };
  };
  return { ask, asked };
}
const KEY = JSON.stringify({ hero: "BTN", board: "Ks7d2cTh", speed: "accurate", budget: 60, prior_budget: 12 });

test("a spot's run: the speed's fields aside, the full menu first, then each size, once", async () => {
  S.runs.clear();
  assert.equal(S.baseKey(KEY), JSON.stringify({ hero: "BTN", board: "Ks7d2cTh" }));
  assert.equal(S.baseKey({ hero: "BTN", one_size: "50%", speed: "fast", quick: true }), JSON.stringify({ hero: "BTN" }));
  const sv = solver(), seen = [];
  const run = await S.start(KEY, sv.ask, (r) => seen.push(r.status));
  assert.deepEqual(sv.asked, ["full", "33%", "75%", "150%"]);
  assert.equal(run.status, "done");
  assert.equal(seen[0], "running"); assert.equal(seen.at(-1), "done");
  // the same spot at another speed is the same run: nothing asked again
  await S.start(JSON.stringify({ hero: "BTN", board: "Ks7d2cTh", speed: "fast", budget: 8 }), sv.ask);
  assert.equal(sv.asked.length, 4);
  const sum = S.summarize(run);
  assert.equal(sum.best, "75%");
  assert.deepEqual(sum.rows.map(r => r.loss_pct), [1, 0.1, 0.5]);   // ((3.0 - EV) / 10 bb pot, % of it)
  assert.deepEqual(sum.tied, []);
});

test("the verdict: ties within the solves' precision, no losses before the full menu", () => {
  const run = S.runOf(JSON.stringify({ t: 1 }));
  S.merge(run, { ...META, full: null, rows: [row("33%", 2.9), { ...row("75%", 2.995), loss_pct: 9 }] });
  let s = S.summarize(run);
  assert.equal(s.best, "75%");
  assert.deepEqual(s.rows.map(r => r.loss_pct), [null, null]);
  assert.equal(S.nextOf(run), "full");
  S.merge(run, { ...META, full: FULL, rows: [row("150%", 2.99)] });
  s = S.summarize(run);
  assert.deepEqual(s.tied, ["150%"]);   // (0.005 bb apart: inside 0.05% of a 10 bb pot)
  assert.deepEqual(s.rows.map(r => r.size), ["33%", "75%", "150%"]);
  assert.equal(S.nextOf(run), null);
});

test("the order: after the full menu, the sizes in the answer's order (the likely best first), else the candidates'", () => {
  const run = S.runOf(JSON.stringify({ order: 1 }));
  S.merge(run, { ...META, full: FULL, rows: [] });
  assert.equal(S.nextOf(run), "33%");
  S.merge(run, { ...META, order: ["150%", "75%", "33%"], full: FULL, rows: [] });
  assert.equal(S.nextOf(run), "150%");
  S.merge(run, { ...META, order: ["150%", "75%", "33%"], full: FULL, rows: [row("150%", 2.95)] });
  assert.equal(S.nextOf(run), "75%");
});

test("a size stopped early, clearly behind the best: its loss approximate, and the note says why", () => {
  const run = S.runOf(JSON.stringify({ cut: 1 }), "turn");
  S.merge(run, { ...META, full: FULL, rows: [row("75%", 2.99), row("150%", 2.95)] });
  run.status = "done";
  assert.doesNotMatch(S.panelHtml(run), /≈/);
  S.merge(run, { ...META, full: FULL, rows: [{ ...row("33%", 2.77, 0.42), cut: true }] });
  const h = S.panelHtml(run);
  assert.match(h, /<td class="n"><span title="Stopped once clearly behind the best: within 0\.42% of the pot">≈2\.3%<\/span><\/td><\/tr>/);
  assert.match(h, /; ≈: a size whose solve stopped early, once it was clearly behind the best\.<\/div>/);
  assert.equal(S.summarize(run).best, "75%");
});

test("stopping: the request in flight is stopped, what's in hand stays, the run carries on from there", async () => {
  S.runs.clear();
  const sv = solver(), stopped = [];
  let hold = null;
  const ask = (st) => {
    if (st.one_size === "75%" && !hold) {   // (this one hangs until stopped)
      const p = new Promise((res) => { hold = res; });
      p.reqId = 7;
      return p;
    }
    return sv.ask(st);
  };
  const going = S.start(KEY, ask);
  await new Promise((r) => setImmediate(r));
  const run = S.runOf(KEY);
  assert.equal(run.cur, "75%");
  S.stop(null, (p) => { stopped.push(p.reqId); hold({ cancelled: true }); });
  await going;
  assert.deepEqual(stopped, [7]);
  assert.equal(run.status, "stopped");
  assert.equal(run.userStopped, false);
  assert.deepEqual(Object.keys(run.rows), ["33%"]);
  await S.start(KEY, ask);
  assert.deepEqual(sv.asked, ["full", "33%", "75%", "150%"]);
  assert.equal(run.status, "done");
  // stopped and started again before the stopped request's answer is back (new advice from the cache shows the tab
  // again at once): the late {cancelled} belongs to the old loop and leaves the new one running
  S.runs.clear();
  const sv2 = solver();
  let late = null;
  const ask2 = (st) => {
    if (st.one_size === "75%" && !late) return new Promise((res) => { late = res; });
    return sv2.ask(st);
  };
  const first = S.start(KEY, ask2);
  await new Promise((r) => setImmediate(r));
  const run2 = S.runOf(KEY);
  S.stop(null, () => {});
  const again = S.start(KEY, ask2);
  late({ cancelled: true });
  await first; await again;
  assert.equal(run2.status, "done");
  assert.deepEqual(Object.keys(run2.rows).sort(), ["150%", "33%", "75%"]);
  // a request that fails: the run says so and stops there
  const bad = S.runOf(JSON.stringify({ bad: 1 }));
  await S.start(bad.key, async () => ({ error: "the one-size view is for heads-up pots: 3 players are in" }));
  assert.equal(bad.status, "error");
  assert.match(S.panelHtml(bad), /heads-up pots/);
});

test("the time left: the estimate first, then the pace of the answers", () => {
  const run = S.runOf(JSON.stringify({ eta: 1 }), "flop");
  assert.equal(S.etaOf(run, "wasm"), 8 * S.EST.wasm.flop);   // (7 sizes and the full menu)
  S.merge(run, { ...META, street: "flop", full: FULL, rows: [row("33%", 2.9)] });
  run.secs = [5, 1];
  assert.equal(S.etaOf(run, "wasm"), 2 * 1);
});

test("the panel: a flop waits to be asked, then progress, the verdict, the sizes and the strategy", () => {
  const flop = S.runOf(JSON.stringify({ flop: 1 }), "flop");
  let h = S.panelHtml(flop, { mode: "wasm" });
  assert.match(h, /<button data-os="start">Find the best size<\/button>/);
  assert.match(h, /about \d+ s in all/);
  const run = S.runOf(JSON.stringify({ panel: 1 }), "turn");
  run.status = "running"; run.cur = "75%";
  S.merge(run, { ...META, hand: "AKs", hole: "AhKh", full: FULL, rows: [{ ...row("33%", 2.9), hand: { freq: [0.25, 0.75], ev: [1.5, 2.25] } }] });
  h = S.panelHtml(run, { mode: "wasm" });
  assert.match(h, /Solving 75% · 2 of 4 done/);
  assert.match(h, /data-os="stop"/);
  assert.match(h, /Best so far: <b>33%<\/b>/);
  assert.match(h, /<tr class="os-wait"><td>75%<\/td><td class="n" colspan="3"><span class="spinner"><\/span>/);
  assert.match(h, /Your AKs \(AhKh\): check 25% <span class="muted">\+1\.50 bb<\/span> · bet 33% 75%/);
  assert.match(h, /data-h="AKs"/);
  S.merge(run, { ...META, full: FULL, rows: [row("75%", 2.99), row("150%", 2.95)] });
  run.status = "done"; run.cur = null;
  h = S.panelHtml(run, { mode: "wasm" });
  assert.doesNotMatch(h, /data-os=/);
  assert.match(h, /Best single size: <b>75%<\/b> · gives up <b>0\.10%<\/b> of the pot against every size mixed/);
  assert.match(h, /<tr data-size="75%" class="best sel" tabindex="0"><td><i style="background:var\(--agg1\)"><\/i>75% <span class="os-tag">best<\/span><\/td><td class="n">60%<\/td><td class="n">\+2\.99<\/td><td class="n">0\.10%<\/td><\/tr>/);
  assert.match(h, /<tr data-size="full" class="os-full" tabindex="0"><td>Every size, mixed<\/td><td class="n">60%<\/td>/);
  assert.match(h, /each solve is within 0\.1% of the pot of equilibrium/);
  // a row picked: its strategy below; a size within precision is no loss below zero
  run.sel = "full";
  assert.match(S.panelHtml(run), /<b>Every size<\/b>: check 40% · bet 33% 20% · bet 75% 40%/);
  S.merge(run, { ...META, full: FULL, rows: [row("150%", 3.004)] });
  assert.match(S.panelHtml(run), /data-size="150%" class="best[^"]*"[^>]*>.*?<td class="n">0\.00%<\/td>/);
  assert.equal(S.colorOf("all-in"), "var(--allin)");
  assert.equal(S.colorOf("bet 200%"), "var(--agg3)");
  assert.equal(S.colorOf("check"), "var(--passive)");
});

test("facing a bet: the raise view, sizes by the multiple of the bet, each with what it raises to", () => {
  assert.equal(S.kindOf({ actions: [{ action: "fold" }, { action: "call" }, { action: "raise to 20.3" }] }), "raise");
  assert.equal(S.kindOf({ actions: [{ action: "check" }, { action: "bet 33%" }] }), "bet");
  const run = S.runOf(JSON.stringify({ raise: 1 }), "turn", "raise");
  let h = S.panelHtml(run);
  assert.match(h, /<b>One raise size<\/b>/);
  assert.match(h, /5 solves, about \d+ s/);   // (four multiples and the full menu, before the answer names them)
  const R = (size, ev, to) => ({ size, actions: ["fold", "call", `raise to ${to}`], freq: [0.3, 0.5, 0.2], ev_bb: ev, iterations: 300,
                                 exploitability_pct: 0.05, seconds: 1, grid: grid(["fold", "call", `raise to ${to}`]) });
  const full = { ...R("full", 2.0, 16.8), actions: ["fold", "call", "raise to 16.8", "raise to 20.3"], freq: [0.3, 0.5, 0.1, 0.1],
                 grid: grid(["fold", "call", "raise to 16.8", "raise to 20.3"]) };
  S.merge(run, { ...META, kind: "raise", to_call_bb: 6.8, candidates: ["2.5x", "3x"], order: ["3x", "2.5x"],
                 raises: { "2.5x": { to: 16.8, pct: 44 }, "3x": { to: 20.3, pct: 59 } }, full, rows: [R("2.5x", 1.9, 16.8), R("3x", 1.99, 20.3)] });
  run.status = "done";
  h = S.panelHtml(run);
  assert.match(h, /Best single raise size: <b>3x<\/b> <span class="muted">to 20\.3<\/span> · gives up <b>0\.10%<\/b> of the pot against every raise size mixed/);
  assert.match(h, /<th>Raise<\/th><th class="n">Raises<\/th>/);
  // each row: its swatch by its size after calling, what it raises to, the share that raises (not folds or calls)
  assert.match(h, /<tr data-size="2\.5x" class="" tabindex="0"><td><i style="background:var\(--agg1\)"><\/i>2\.5x <span class="muted">to 16\.8<\/span><\/td><td class="n">20%<\/td><td class="n">\+1\.90<\/td><td class="n">1\.00%<\/td><\/tr>/);
  assert.match(h, /<td>Every size, mixed<\/td><td class="n">20%<\/td>/);
  assert.match(h, /Sizes are multiples of the bet faced/);
  assert.match(h, /Best: the raise size that keeps the most EV for the player facing the bet/);
  assert.match(h, /<b>3x<\/b>: fold 30% · call 50% · raise to 20\.3 20%/);
});

test("a tournament: the panel says the EVs are prize equity and shows the tournament's notes", () => {
  const run = S.runOf(JSON.stringify({ tourney: 1 }), "river");
  const full = { ...row("full", 3.0), actions: ["check", "bet 33%"], grid: grid(["check", "bet 33%"]) };
  S.merge(run, { ...META, full, rows: [row("33%", 2.99)] });
  assert.doesNotMatch(S.panelHtml(run), /prize equity/);
  const note = "tournament: ICM, your prize equity 16.7% of what's left to pay; bubble factor up to 1.39 (a chip lost costs 1.39x what one won gains)";
  S.merge(run, { ...META, tournament: [note, "bounties: KO <10>"], full, rows: [] });
  const h = S.panelHtml(run);
  assert.match(h, /In a tournament every solve pays prize equity \(ICM\), so EVs are chips worth as much prize equity\.<br>tournament: ICM, your prize equity 16\.7%/);
  assert.match(h, /<br>bounties: KO &lt;10&gt;<\/div>/);   // (escaped)
  assert.ok(run.rows["33%"]);   // (rows kept across answers)
});

// ---- the page
const RES = { street: "turn", hero: "BTN", villain: "BB", hand: "AQo", board: "Ks7d2cTh", recommended: "bet 75%",
  solver: { exploitability_pct: 0.1, iterations: 300, solve_seconds: 2, active_combos: 300, tree_actions: ["check", "bet 33%", "bet 75%"] },
  actions: [{ action: "check", freq: 0, ev_bb: 3 }, { action: "bet 33%", freq: 0, ev_bb: 3.5 }, { action: "bet 75%", freq: 1, ev_bb: 4 }],
  grid: { actions: ["check", "bet 33%", "bet 75%"], cells: { AQo: { w: 1, f: [0, 0, 1] } } } };
function page(server) {
  const p = loadPage({ server });
  p.run("wireGrid = () => {}");
  p.ctx.__res = JSON.stringify(RES);
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  return p;
}
const tabs = (p) => [...p.byId("result").innerHTML.matchAll(/data-v="(\w+)"/g)].map(m => m[1]);

test("the advice panel offers Simple heads-up after the flop where you can bet or raise, at equilibrium", () => {
  const p = page();
  assert.deepEqual(tabs(p), ["grid", "breakdown", "equity", "simple"]);
  // facing a bet: where you can raise (the raise view), not where calling or folding is all there is
  for (const raise of ["raise to 16.5", "all-in"]) {
    p.ctx.__res = JSON.stringify({ ...RES, actions: [{ action: "fold", freq: 0 }, { action: "call", freq: 1 }, { action: raise, freq: 0 }] });
    p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
    assert.ok(tabs(p).includes("simple"), raise);
  }
  const facing = { ...RES, actions: [{ action: "fold", freq: 0 }, { action: "call", freq: 1 }] };
  for (const r of [facing, { ...RES, multiway: true, solved: true, opponents: ["BB", "CO"] }, { ...RES, locked: { BB: { fold: 10 } } }]) {
    p.ctx.__res = JSON.stringify(r);
    p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
    assert.ok(!tabs(p).includes("simple"), JSON.stringify(r.actions));
  }
  p.ctx.__res = JSON.stringify(RES);
  p.run(`renderResult(JSON.parse(__res), JSON.stringify({ ...payload(), opponents: "pool" }))`);
  assert.ok(!tabs(p).includes("simple"));
});

test("opening Simple loads simple.js once and asks at Fast, one size a request; new advice stops it", async () => {
  const p = page();
  await p.ready();
  p.run(`var __box = document.createElement("div"), __grid = document.createElement("div"), __tabs = [];
    for (const v of ["grid", "breakdown", "equity", "simple"]) { const b = document.createElement("button"); b.dataset.v = v; __tabs.push(b); }
    var __el = { querySelector: (s) => s === ".bdview" ? __box : s === ".gridview" ? __grid : null, querySelectorAll: () => __tabs };
    var __head = []; document.head = { appendChild: (s) => { __head.push(s); return s; } };`);
  p.run(`showView(__el, "simple", JSON.parse(__res), JSON.stringify({ ...payload(), speed: "accurate", budget: 60 }))`);
  assert.deepEqual(p.val("[__grid.hidden, __box.hidden, __tabs.map(b => b.className)]"), [true, false, ["", "", "", "active"]]);
  assert.equal(p.run("__head[0].src"), "simple.js?v=1");
  const n = p.posted.length;
  vm.runInContext(fs.readFileSync(SP_JS, "utf8"), p.ctx, { filename: "simple.js" });
  p.run("__head[0].onload()");
  await new Promise(r => setImmediate(r));
  const st = p.posted.at(-1).state;
  assert.equal(p.posted.length, n + 1);
  assert.equal(st.one_size, "full");
  assert.deepEqual([st.speed, st.budget, st.prior_budget], ["fast", 8, 3]);   // (the browser's Fast budgets)
  assert.deepEqual({ ...st, one_size: undefined, speed: undefined, budget: undefined, prior_budget: undefined },
                   { ...p.payload(), one_size: undefined, speed: undefined, budget: undefined, prior_budget: undefined });
  assert.match(p.run("__box.innerHTML"), /Solving every size/);
  assert.equal(p.run("[...window.Simple.runs.values()][0].status"), "running");
  // the page moves on to another spot: the run stops (its request stopped through the shared flag), its answers stay
  p.run(`window.Simple.stop(null, simpleStop)`);
  assert.equal(p.run("[...window.Simple.runs.values()][0].status"), "stopped");
});
