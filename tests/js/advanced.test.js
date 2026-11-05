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

test("a change drops the shown advice; neither a new hand nor a change of players keeps these", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  p.byId("result").className = "advice"; p.byId("result").innerHTML = "old advice";
  p.byId("sz_flop").onchange({ target: { value: "50" } });
  assert.equal(p.byId("result").className, "empty");
  p.S = { ...state({ ...FLOP, hero: "BB" }), ...ADV, ranges: { UTG1: "AA", BTN: "QQ+" } };
  p.byId("newhand").onclick();
  for (const k of ["sizes", "ranges", "optSizes", "locks"]) assert.equal(k in p.S, false, k);
  p.S = { ...state({ ...FLOP, hero: "BB" }), ...ADV, ranges: { UTG1: "AA", BTN: "QQ+" } };
  p.run('document.getElementById("players").children[7].onclick()');   // 9 players
  for (const k of ["sizes", "ranges", "optSizes", "locks"]) assert.equal(k in p.S, false, k);
});

test("share link round trip, and the summary says so", async () => {
  const a = loadPage();
  a.S = { ...state(FLOP), ...ADV };
    const b = loadPage({ hash: a.run("shareLink()").replace(/^[^#]*/, "") });
  assert.deepEqual(j(b.S), j(a.S));
  await a.ready(); await b.ready();
  assert.deepEqual(b.payload(), a.payload());
  assert.equal(b.byId("adv").open, true);
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
});

test("a precomputed answer is badged; one borrowed from another table size's library spot says so and how far", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  const pre = { spot: "srp_btn_bb", label: "BTN vs BB single-raised pot", iterations: 300, same_tree: false };
  const res = { street: "flop", hero: "BTN", hand: "AKs", recommended: "check", actions: [{ action: "check", freq: 1, ev_bb: 1 }], precomputed: pre };
  p.ctx.__res = JSON.stringify(res);
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  let h = p.byId("result").innerHTML;
  assert.match(h, /<span class="conv ok" title="Solved offline \(BTN vs BB single-raised pot, 300 iterations\) with Accurate's flop model: instant, at either speed.">precomputed<\/span>/);
  const text = "this table's ranges differ from that spot's by 0.4% of their weight at most";
  p.ctx.__res = JSON.stringify({ ...res, precomputed: { ...pre, label: "6-max " + pre.label, borrowed: { spot: "srp_btn_bb", cover: [1, 0.9996], dist: [0.0088, 0.0025], text } } });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  h = p.byId("result").innerHTML;
  assert.match(h, /<span class="conv meh" title="Solved offline \(6-max BTN vs BB single-raised pot, 300 iterations\)[^"]* Approximate: that spot read as this one \(this table's ranges differ from that spot's by 0\.4% of their weight at most\)\.">precomputed · borrowed<\/span>/);
});

test("an answer whose ranges left the solve earlier in the hand is badged off the solved line, with why", async () => {
  // (gto/advisor.py off_solution: HJ KhQh's 75% flop bet the solve takes with 0% of its range, read by the betting
  // rules; a line that left a few combos, filled out; an unsettled earlier solve; a Quick answer's rules)
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  const off = [{ street: "flop", pos: "HJ", why: "rare", action: "bet 75%", share: 0 }, { street: "flop", pos: "BTN", why: "filled", combos: 6.7 },
    { street: "turn", why: "unsettled", trust: 40 }, { street: "turn", why: "rules" }, { street: "flop", pos: "BTN", why: "rare", action: "raise to 13", share: 4.2 }];
  const res = { street: "river", hero: "HJ", hand: "KQs", recommended: "fold", actions: [{ action: "fold", freq: 1, ev_bb: 0 }, { action: "call", freq: 0, ev_bb: -62.5 }],
    solver: { iterations: 1000, solve_seconds: 0.03, active_combos: 153, tree_actions: ["fold", "call"], exploitability_pct: 0.01 } };
  p.ctx.__res = JSON.stringify({ ...res, off_solution: off });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  const h = p.byId("result").innerHTML;
  assert.match(h, /<span class="conv ok" title="[^"]*">converged · 0\.0% pot<\/span><span class="conv meh" title="Off the solved line: [^"]*not the whole hand's equilibrium\.">off the solved line<\/span>/);
  assert.match(h, /Off the solved line: HJ's bet 75% on the flop is never in the solve, so their range from there is the betting rules'; BTN's flop line left 6\.7 combos, so their range is filled out with the rest of their flop range; the turn solve hadn't settled, so ranges follow it 40% and the betting rules the rest; the turn is narrowed by the betting rules, not solved; BTN's raise to 13 on the flop is only 4\.2% in the solve, so their range from there is the betting rules'\. This street's solve is the best answer to those ranges/);
  assert.match(p.val(`offText({ off_solution: [{ street: "preflop", pos: "CO", why: "filled" }] })`), /^CO's preflop line isn't in the charts/);
  // the play after a rare line solved again from the node (Advisor._rare_resolver)
  assert.equal(p.val(`offText({ off_solution: [{ street: "flop", pos: "BB", why: "rare", action: "check", share: 0, resolved: true }] })`),
    "BB's check on the flop is never in the solve, so their range from there is the betting rules' and the play after it is solved again from there");
  assert.equal(p.val(`gradeWith(JSON.parse(__res), { action: "fold" }).x`), 1);
  // on the solved line: neither, and the grade has no x
  p.ctx.__res = JSON.stringify(res);
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.doesNotMatch(p.byId("result").innerHTML, /off the solved line|Off the solved line/);
  assert.equal(p.val(`"x" in gradeWith(JSON.parse(__res), { action: "fold" })`), false);
  assert.equal(p.val("offSolution({ off_solution: [] })"), false);
});

test("custom sizes with size optimization: the hint says they compete in its menu, the advice whether they were kept", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  p.byId("sz_flop").onchange({ target: { value: "40 120" } });
  assert.equal(p.byId("optSizesHint").hidden, true);   // (no optimization)
  p.byId("optSizes").onchange({ target: { checked: true } });
  assert.equal(p.byId("optSizesHint").hidden, false);
  p.byId("sz_flop").onchange({ target: { value: "" } });
  p.byId("sz_raise").onchange({ target: { value: "60" } });
  assert.equal(p.byId("optSizesHint").hidden, true);   // (a raise size isn't a street's bet menu)
  const res = { street: "flop", recommended: "check", size_usage: { check: 0.2, "33%": 0.5, "40%": 0.3 }, sizes_kept: ["33%", "40%"], sizes_custom: ["40%", "120%"],
    actions: [{ action: "check", freq: 1, ev_bb: 1 }] };
  p.ctx.__res = JSON.stringify(res);
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.match(p.byId("result").innerHTML, /kept 33%, 40% \(your sizes 40%, 120% were in the menu\)/);
  p.ctx.__res = JSON.stringify({ ...res, sizes_kept: ["33%"] });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.match(p.byId("result").innerHTML, /kept 33% \(your sizes 40%, 120% were in the menu, not kept\)/);
});

test("the advisor reads the Advanced keys the page sends", () => {
  const src = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "advisor.py"), "utf8");
  const body = src.slice(src.indexOf("def _advise("), src.indexOf("def _preflop("));
  for (const k of ["sizes", "ranges", "optimize_sizes"]) assert.ok(body.includes(`state.get("${k}")`), k);
});

// Locks (S.locks, sent as "locks"): an opponent's tendencies, re-priced against preflop and best-responded to after the
// flop (gto/exploit.py)
const lockRow = (p, seat) => [...p.byId("lockIns").children].find(r => r.children[0].dataset.p === seat);
const lockSel = (p, seat) => lockRow(p, seat).children[0];
const lockBox = (p, seat, k) => [...lockRow(p, seat).walk()].find(e => e.tagName === "INPUT" && e.dataset.k === k);

test("locks: a profile per opponent, sent preflop and after, and its numbers editable", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state(FLOP), finished: true };   // (locks are for a finished hand's review)
  p.run("renderAll()");
  assert.equal("locks" in p.payload(), false);
  assert.deepEqual([...p.byId("lockIns").children].map(r => r.children[0].dataset.p), ["UTG", "HJ", "CO", "SB", "BB"]);
  assert.equal(lockSel(p, "BB").value, "");
  assert.equal(lockBox(p, "BB", "fold"), undefined);       // GTO: no inputs
  lockSel(p, "BB").onchange({ target: { value: "station" } });
  assert.deepEqual(j(p.S.locks), { BB: { fold: -20, bluff: -50, raise: -50 } });
  assert.deepEqual(j(p.payload().locks), { BB: { fold: -20, bluff: -50, raise: -50 } });
  assert.equal(lockSel(p, "BB").value, "station");
  assert.match(p.run("ADV_TEXT()"), /BB locked \(calling station\)/);
  // a number edited: Custom, the rest kept; out of bounds clamped, a cleared box dropped
  lockBox(p, "BB", "next_fold").onchange({ target: { value: "160" } });
  lockBox(p, "BB", "raise").onchange({ target: { value: "" } });
  assert.deepEqual(j(p.S.locks), { BB: { fold: -20, bluff: -50, next_fold: 100 } });
  assert.equal(lockSel(p, "BB").value, "custom");
  assert.match(p.run("ADV_TEXT()"), /BB locked \(fold -20, bluff -50, next fold 100%\)/);
  // Custom from GTO shows the inputs without locking anything yet
  lockSel(p, "SB").onchange({ target: { value: "custom" } });
  assert.equal("SB" in p.S.locks, false);
  lockBox(p, "SB", "bet").onchange({ target: { value: "40" } });
  assert.deepEqual(j(p.S.locks.SB), { bet: 40 });
  // back to GTO: unlocked
  lockSel(p, "SB").onchange({ target: { value: "" } });
  assert.deepEqual(Object.keys(p.S.locks), ["BB"]);
  // preflop the locks are sent too (they act there), the other Advanced fields aren't; your own seat's lock never is
  p.S = { ...state({ cards: ["Ah", "Kh"] }), locks: { BB: { fold: 10 } }, sizes: { flop: [50] } };
  assert.deepEqual(j(p.payload().locks), { BB: { fold: 10 } });
  assert.equal("sizes" in p.payload(), false);
  p.S = { ...state(FLOP), locks: { BTN: { fold: 10 }, BB: { bet: -20 } } };
  assert.deepEqual(j(p.payload().locks), { BB: { bet: -20 } });
});

test("locks: share link round trip, and the summary and the advice say so", async () => {
  const a = loadPage();
  a.S = { ...state(FLOP), finished: true, locks: { BB: { fold: 15 }, SB: { fold: 5, next_bet: 0 } } };
    const b = loadPage({ hash: a.run("shareLink()").replace(/^[^#]*/, "") });
  assert.deepEqual(j(b.S), j(a.S));
  await a.ready(); await b.ready();
  assert.deepEqual(b.payload(), a.payload());
  assert.equal(b.byId("adv").open, true);
  b.run("editSetup = false; renderAll()");
  assert.match(b.byId("setupSum").innerHTML, /BB locked \(overfolds to bets\)/);
  b.ctx.__res = JSON.stringify({ street: "flop", hero: "BTN", villain: "BB", hand: "AKs", recommended: "bet 75%", locked: { BB: { fold: 15 } },
    actions: [{ action: "check", freq: 0, gto_freq: 0.4, ev_bb: 1 }, { action: "bet 75%", freq: 1, gto_freq: 0.6, ev_bb: 2 }], notes: [] });
  b.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.match(b.byId("result").innerHTML, /Against your lock on BB \(overfolds to bets\): the best response/);
  assert.match(b.byId("result").innerHTML, /GTO 40%/);

  // preflop: re-priced against the lock, the chart's mix as GTO %
  b.ctx.__res = JSON.stringify({ street: "preflop", hero: "BTN", hand: "AKs", recommended: "raise", locked: { BB: { fold: 15 } },
    actions: [{ action: "raise", freq: 1, gto_freq: 0.8, amount_bb: 2.5 }, { action: "fold", freq: 0, gto_freq: 0.2 }], notes: [] });
  b.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.match(b.byId("result").innerHTML, /Against your lock on BB \(overfolds to bets\): your line re-priced against it, with GTO's frequency under each \(the preflop chart's mix\)/);
});

test("locks: bad fields in a link are dropped or clamped", () => {
  const st = { ...state(FLOP), locks: { BB: { fold: "x", bluff: 300, bogus: 5, bet: 0, next_fold: 50 }, XX: { fold: 10 }, SB: { fold: true }, CO: 7 } };
  const p = loadPage({ hash: "#h=" + Buffer.from(JSON.stringify(st)).toString("base64") });
  assert.deepEqual(j(p.S.locks), { BB: { bluff: 100, next_fold: 50 } });
  const p2 = loadPage({ storage: { handreview: JSON.stringify({ ...st, locks: "BB" }) } });
  assert.equal("locks" in p2.S, false);
});

test("locks: the page's keys, bounds and presets are the advisor's", () => {
  const src = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "exploit.py"), "utf8");
  const block = src.slice(src.indexOf("LOCK_KEYS = {"), src.indexOf("\n}", src.indexOf("LOCK_KEYS = {")));
  const py = Object.fromEntries([...block.matchAll(/"(\w+)": \((-?\d+), (-?\d+)\)/g)].map(m => [m[1], [+m[2], +m[3]]]));
  const p = loadPage();
  assert.deepEqual(j(p.run("LOCK_KEYS")), py);
  for (const [k] of p.run("LOCK_INPUTS")) assert.ok(k in py, k);
  const pre = src.slice(src.indexOf("LOCK_PRESETS = {"), src.indexOf("LOCK_WORDS"));
  for (const [k, [, v]] of Object.entries(p.run("LOCK_PRESETS")))
    assert.ok(pre.includes(`"${k}": ${JSON.stringify(v).replace(/"(\w+)":/g, '"$1": ').replace(/,/g, ", ")}`), k);
  const adv = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "advisor.py"), "utf8");
  const body = adv.slice(adv.indexOf("def _advise("), adv.indexOf("def _preflop("));
  assert.ok(body.includes('state.get("locks")'));
});
