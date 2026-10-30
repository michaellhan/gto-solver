// Flop reports (gto/static/reports.js, loaded on demand): textures, the rows of each view, filters, sorting, the
// weighted group averages, the table's HTML, and a click on a flop loading that hand into the page.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage } = require("./page.js");

const RP_JS = path.join(__dirname, "..", "..", "gto", "static", "reports.js");
const R = require(RP_JS);
const LIB = path.join(__dirname, "..", "..", "gto", "static", "flop_library");
const j = (x) => JSON.parse(JSON.stringify(x));

// a small report shaped like flop_library.build_report's
const REP = {
  format: 1, id: "srp_btn_bb", label: "BTN vs BB single-raised pot", line: "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c",
  seats: ["BB", "BTN"], pot: 5.5, stack: 97.5,
  nodes: { "": ["check", "bet 33%", "bet 75%"], check: ["check", "bet 33%", "bet 75%"],
           "check|bet 33%": ["fold", "call", "raise to 7"], "check|bet 75%": ["fold", "call", "raise to 11"] },
  flops: [
    { f: "Ac4d3h", w: 100, ev: [2.1, 3.4], eq: [0.45, 0.55], n: { "": [0.9, 0.05, 0.05], check: [0.5, 0.2, 0.3],
      "check|bet 33%": [0.4, 0.5, 0.1], "check|bet 75%": [0.5, 0.4, 0.1] }, h: { "": "Jd7d", check: "KhAd", "check|bet 33%": "Kc7d", "check|bet 75%": "Ks7d" } },
    { f: "9c8c7c", w: 50, ev: [2.8, 2.7], eq: [0.52, 0.48], n: { "": [1, 0, 0], check: [0.8, 0.2, 0], "check|bet 33%": [0.3, 0.6, 0.1] },
      h: { "": "6d5d", check: "AsAh", "check|bet 33%": "Tc9d" } },
    { f: "Kc7d7h", w: 50, ev: [1.9, 3.6], eq: [0.41, 0.59], n: { "": [1, 0, 0] }, h: { "": "QsQd" } },
  ],
};

test("textures", () => {
  const t = (f) => { const x = R.texture(f); return [x.hi, x.pair, x.suits, x.conn]; };
  assert.deepEqual(t("Ac4d3h"), ["A", "unpaired", "rainbow", "conn"]);     // A-3-4: the wheel
  assert.deepEqual(t("9c8c7c"), ["9-", "unpaired", "mono", "conn"]);
  assert.deepEqual(t("Kc7d7h"), ["K", "paired", "rainbow", "dry"]);
  assert.deepEqual(t("Kc9d2c"), ["K", "unpaired", "twotone", "semi"]);
  assert.deepEqual(t("Kc8d2h"), ["K", "unpaired", "rainbow", "dry"]);
  assert.deepEqual(t("5c5d5h"), ["9-", "trips", "rainbow", "dry"]);
  assert.deepEqual(t("QcJdTh"), ["Q", "unpaired", "rainbow", "conn"]);
});

test("each view's rows: its node's frequencies by column, null where the entry doesn't store it", () => {
  assert.deepEqual(R.columns(REP, "oop"), ["check", "bet 33%", "bet 75%"]);
  assert.deepEqual(R.columns(REP, "vsbet"), ["fold", "call", "raise"]);
  const ip = R.rows(REP, "ip");
  assert.deepEqual(ip.map(r => r.freqs), [[0.5, 0.2, 0.3], [0.8, 0.2, 0], null]);
  const vb = R.rows(REP, "vsbet");
  // facing in position's most common bet: 75% on A43 (30% against 20%), 33% on 987
  assert.deepEqual(vb.map(r => [r.bet, r.freqs]), [["bet 75%", [0.5, 0.4, 0.1]], ["bet 33%", [0.3, 0.6, 0.1]], [null, null]]);
});

test("filter, sort and weighted averages", () => {
  const rs = R.rows(REP, "oop");
  assert.deepEqual(R.filter(rs, { pair: "unpaired" }).map(r => r.f), ["Ac4d3h", "9c8c7c"]);
  assert.deepEqual(R.filter(rs, { suits: "rainbow", hi: "K" }).map(r => r.f), ["Kc7d7h"]);
  assert.deepEqual(R.sort(rs, "flop", true).map(r => r.f), ["Ac4d3h", "Kc7d7h", "9c8c7c"]);
  assert.deepEqual(R.sort(rs, "flop", false).map(r => r.f), ["9c8c7c", "Kc7d7h", "Ac4d3h"]);
  assert.deepEqual(R.sort(rs, "a0", true).map(r => r.f), ["Kc7d7h", "9c8c7c", "Ac4d3h"]);   // ties: high flop first
  assert.deepEqual(R.sort(rs, "ev1", true).map(r => r.f), ["Kc7d7h", "Ac4d3h", "9c8c7c"]);
  const a = R.average(rs);
  assert.equal(a.w, 200);
  assert.ok(Math.abs(a.freqs[0] - (0.9 * 100 + 50 + 50) / 200) < 1e-12);
  assert.ok(Math.abs(a.ev[0] - (2.1 * 100 + 2.8 * 50 + 1.9 * 50) / 200) < 1e-12);
  // frequencies average over the flops that store the node only
  const b = R.average(R.rows(REP, "ip"));
  assert.ok(Math.abs(b.freqs[0] - (0.5 * 100 + 0.8 * 50) / 150) < 1e-12);
  const g = R.groups(rs, "suits");
  assert.deepEqual(g.map(x => [x.g, x.n, x.w]), [["rainbow", 2, 150], ["mono", 1, 50]]);
});

test("the table: headers sort, rows load, groups filter", () => {
  const st = { view: "oop", sort: "a1", desc: true, group: "pair", filter: {} };
  const h = R.html(REP, st);
  assert.match(h, /data-sort="a1"[^>]*aria-sort="descending"/);
  for (const k of ["flop", "a0", "a1", "a2", "ev0", "ev1", "eq0", "eq1"]) assert.match(h, new RegExp(`data-sort="${k}"`));
  assert.deepEqual([...h.matchAll(/data-flop="(\w+)"/g)].map(m => m[1]), ["Ac4d3h", "Kc7d7h", "9c8c7c"]);
  assert.deepEqual([...h.matchAll(/data-group="([\w:]+)"/g)].map(m => m[1]), ["pair:unpaired", "pair:paired"]);
  assert.match(h, /All flops/);
  const f = R.html(REP, { ...st, filter: { suits: "mono" }, group: "" });
  assert.deepEqual([...f.matchAll(/data-flop="(\w+)"/g)].map(m => m[1]), ["9c8c7c"]);
  assert.match(f, /1 flops/);
  assert.match(R.html(REP, { ...st, view: "ip" }), /not stored/);
});

test("the controls change the view, sort, filters", () => {
  R._set({ spots: [{ id: REP.id, label: REP.label, report: "x" }] }, REP);
  Object.assign(R.state, { view: "oop", sort: "flop", desc: true, group: "", filter: {} });
  R.act({ view: "vsbet" }); assert.equal(R.state.view, "vsbet");
  R.act({ sort: "ev0" }); assert.deepEqual([R.state.sort, R.state.desc], ["ev0", true]);
  R.act({ sort: "ev0" }); assert.equal(R.state.desc, false);
  R.act({ ctl: "group", value: "conn" }); assert.equal(R.state.group, "conn");
  R.act({ group: "conn:dry" }); assert.deepEqual([R.state.filter.conn, R.state.group], ["dry", ""]);
  R.act({ ctl: "f:suits", value: "rainbow" }); assert.deepEqual(j(R.state.filter), { conn: "dry", suits: "rainbow" });
});

test("the hand state for a flop at each view", () => {
  const pre = [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "fold" }, { pos: "CO", action: "fold" },
               { pos: "BTN", action: "raise", to: 2.5 }, { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }];
  assert.deepEqual(R.stateFor(REP, "Ac4d3h", "oop"), { players: 6, hero: "BB", stack: 100,
    cards: ["Jd", "7d", "Ac", "4d", "3h", null, null], actions: { preflop: pre, flop: [], turn: [], river: [] } });
  const ip = R.stateFor(REP, "Ac4d3h", "ip");
  assert.deepEqual([ip.hero, ip.cards.slice(0, 2), ip.actions.flop], ["BTN", ["Kh", "Ad"], [{ pos: "BB", action: "check" }]]);
  const vb = R.stateFor(REP, "Ac4d3h", "vsbet");
  assert.deepEqual([vb.hero, vb.cards.slice(0, 2), vb.actions.flop], ["BB", ["Ks", "7d"],
    [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 4.13 }]]);
  // a spot of another game (flop_library.Format: the index's and report's "game") is loaded in that game
  const g = R.stateFor({ ...REP, game: { players: 6, stack: 40, ante: 0.125, straddle: false } }, "Ac4d3h", "oop");
  assert.deepEqual([g.players, g.stack, g.ante, g.straddle], [6, 40, 0.125, undefined]);
});

test("a flop's Turns: Study at the flop with the view's line closed, for its turn report", () => {
  const pre = R.stateFor(REP, "Ac4d3h", "oop").actions.preflop;
  // out of position first: checked through; in position's view and facing a bet: the most common bet, called
  assert.deepEqual(R.studyStateFor(REP, "Ac4d3h", "oop"), { players: 6, stack: 100, board: "Ac4d3h", preflop: pre,
    flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "check" }], turn: [], river: [] });
  for (const v of ["ip", "vsbet"])
    assert.deepEqual(R.studyStateFor(REP, "Ac4d3h", v).flop, [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 4.13 }, { pos: "BB", action: "call" }]);
  assert.equal(R.turnLine(REP, "Kc7d7h", "vsbet").text, "check, check");   // (no bet stored after a check)
  assert.equal(R.studyStateFor(REP, "2c2d2h", "oop"), null);
  const g = R.studyStateFor({ ...REP, game: { players: 6, stack: 40, ante: 0.125 } }, "Ac4d3h", "oop");
  assert.deepEqual([g.stack, g.ante], [40, 0.125]);
  // a button on every flop's row, not the summary's or the groups'
  const h = R.html(REP, { view: "vsbet", sort: "flop", desc: true, group: "pair", filter: {} });
  assert.deepEqual([...h.matchAll(/data-turns="(\w+)"/g)].map(m => m[1]), ["Ac4d3h", "Kc7d7h", "9c8c7c"]);
  assert.match(h, /title="The turn report in Study: every turn card after the most reached flop line through BB facing bet 75% after a check">Turns<\/button>/);
  // the view Study picks the line for (Runouts.pickLine): the decision, and facing a bet the bet the view shows
  assert.deepEqual(R.turnView(REP, "Ac4d3h", "oop"), { view: "oop", label: "BB's first decision" });
  assert.deepEqual(R.turnView(REP, "Ac4d3h", "ip"), { view: "ip", label: "BTN's decision after a check" });
  assert.deepEqual(R.turnView(REP, "Ac4d3h", "vsbet"), { view: "vsbet", bet: "bet 75%", label: "BB facing bet 75% after a check" });
  assert.deepEqual(R.turnView(REP, "Kc7d7h", "vsbet"), { view: "vsbet", label: "BB facing a bet after a check" });
});

test("a flop's Turns opens Study there for the view (STUDY_AT)", async () => {
  const p = loadPage();
  vm.runInContext(fs.readFileSync(RP_JS, "utf8"), p.ctx, { filename: "reports.js" });
  await p.ready();
  p.run("FlopReports")._set({ spots: [] }, REP);
  Object.assign(p.run("FlopReports").state, { view: "vsbet", spot: "srp_btn_bb" });
  p.run("globalThis.openView = (v) => { globalThis.opened = v; }");
  await p.run("FlopReports").act({ turns: "Ac4d3h" });
  assert.equal(p.val("opened"), "study");
  const at = JSON.parse(JSON.stringify(p.val("STUDY_AT")));
  assert.deepEqual([at.spot, at.report, at.view, at.bet, at.label], ["srp_btn_bb", true, "vsbet", "bet 75%", "BB facing bet 75% after a check"]);
  assert.deepEqual(at.state.flop, [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 4.13 }, { pos: "BB", action: "call" }]);
});

// click-to-load in the page: the hand replaces the page's, the hero is to act at the flop node
async function loaded(rep, f, view) {
  const p = loadPage();
  vm.runInContext(fs.readFileSync(RP_JS, "utf8"), p.ctx, { filename: "reports.js" });
  await p.ready();
  p.run("FlopReports")._set({ spots: [] }, rep);
  Object.assign(p.run("FlopReports").state, { view });
  await p.run("FlopReports").act({ flop: f });
  return p;
}

test("a click on a flop loads that hand into the page", async () => {
  const p = await loaded(REP, "9c8c7c", "vsbet");
  const S = p.S, sim = p.sim();
  assert.deepEqual([S.players, S.hero, S.stack, S.cards], [6, "BB", 100, ["Tc", "9d", "9c", "8c", "7c", null, null]]);
  assert.deepEqual(S.actions.flop, [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 1.82 }]);
  assert.deepEqual([sim.current, sim.toAct, sim.active], ["flop", "BB", ["BTN", "BB"]]);
  assert.match(p.byId("toast").innerHTML, /Loaded BTN vs BB single-raised pot, 9c8c7c: you are BB with Tc9d/);
  // the solver is asked about that spot
  const st = p.payload();
  assert.deepEqual([st.hero, st.hole, st.board], ["BB", "Tc9d", "9c8c7c"]);
});

// the shipped reports: every spot's, as the build wrote them (scripts/build_flop_library.py)
test("the shipped reports load, render and load a hand for every view", async () => {
  const idx = JSON.parse(fs.readFileSync(path.join(LIB, "index.json"), "utf8"));
  assert.ok(idx.spots.length > 0);
  for (const sp of idx.spots) {
    assert.ok(sp.report, `${sp.id}: no report in the index`);
    const rep = JSON.parse(fs.readFileSync(path.join(LIB, sp.id, "report.json"), "utf8"));
    assert.deepEqual(rep.flops.map(x => x.f).sort(), Object.keys(sp.flops).sort());
    for (const v of Object.keys(R.VIEWS)) {
      const rs = R.rows(rep, v), a = R.average(rs);
      assert.ok(Math.abs(a.freqs.reduce((x, y) => x + y, 0) - 1) < 0.005, `${sp.id} ${v}`);
      assert.equal((R.html(rep, { view: v, sort: "flop", desc: true, group: "suits", filter: {} }).match(/data-flop=/g) || []).length, rs.length);
    }
  }
  const rep = JSON.parse(fs.readFileSync(path.join(LIB, idx.spots[0].id, "report.json"), "utf8"));
  for (const v of Object.keys(R.VIEWS)) {
    const f = rep.flops.find(x => R.nodeOf(rep, x, v)).f;
    const p = await loaded(rep, f, v), S = p.S, sim = p.sim();
    assert.equal(sim.toAct, S.hero, v);
    assert.equal(sim.current, "flop");
    assert.ok(S.cards[0] && S.cards[1] && !S.cards.slice(2, 5).includes(S.cards[0]) && !S.cards.slice(2, 5).includes(S.cards[1]));
  }
});
