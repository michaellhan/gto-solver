// Drills (gto/static/drills.js, loaded on demand): decoding the library's entries, the (flop, node) draws of each
// drill, hands dealt by range weight (in range, off the board), grading (mixed spots too), the score, the keys.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

globalThis.FlopReports = require(path.join(__dirname, "..", "..", "gto", "static", "reports.js"));
const D = require(path.join(__dirname, "..", "..", "gto", "static", "drills.js"));
const LIB = path.join(__dirname, "..", "..", "gto", "static", "flop_library");
const INDEX = JSON.parse(fs.readFileSync(path.join(LIB, "index.json"), "utf8"));
const RANKS = "23456789TJQKA", SUITS = "cdhs";
const cardInt = (s) => RANKS.indexOf(s[0]) * 4 + SUITS.indexOf(s[1]);
// a seeded generator (mulberry32)
const rng = (seed) => () => { seed = (seed + 0x6D2B79F5) | 0; let t = seed; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const load = (sid, f) => D.decode(fs.readFileSync(path.join(LIB, sid, f + ".bin")));
const ranges = (sid) => JSON.parse(fs.readFileSync(path.join(LIB, sid, "ranges.json"), "utf8"));
const report = (sid) => JSON.parse(fs.readFileSync(path.join(LIB, sid, "report.json"), "utf8"));

const REP = {
  id: "srp_btn_bb", label: "BTN vs BB single-raised pot", line: "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", seats: ["BB", "BTN"], pot: 5.5, stack: 97.5,
  nodes: { "": ["check", "bet 33%", "bet 75%"], check: ["check", "bet 33%", "bet 75%"],
           "check|bet 33%": ["fold", "call", "raise to 7"], "check|bet 75%": ["fold", "call", "raise to 11"] },
  flops: [
    { f: "Ac4d3h", w: 100, n: { "": [0.9, 0.05, 0.05], check: [0.5, 0.2, 0.3], "check|bet 33%": [0.4, 0.5, 0.1], "check|bet 75%": [0.5, 0.4, 0.1] } },
    { f: "9c8c7c", w: 50, n: { "": [1, 0, 0], check: [0.8, 0.2, 0], "check|bet 33%": [0.3, 0.6, 0.1] } },
    { f: "Kc7d7h", w: 50, n: { "": [1, 0, 0] } },
  ],
};

test("decoded entries: each stored node's strategy, weighted by the range there, gives the solve's frequencies", async () => {
  for (const sp of INDEX.spots.slice(0, 3)) {
    const rg = ranges(sp.id);
    for (const f of Object.keys(sp.flops).slice(0, 3)) {
      const e = await load(sp.id, f);
      assert.equal(e.head.board, f);
      for (const nd of e.nodes) {
        assert.equal(nd.strat.length, nd.names.length);
        const w = D.handWeights(e, nd.path, rg), tot = w.reduce((a, b) => a + b, 0);
        // (facing a bet after checking, the weights carry the hand's stored check frequency, in 1% steps: where the
        // range seldom checks, as a 3bettor, that rounding moves the facing range's frequencies a few points, and
        // where it checks a combo or two in all, anything)
        const tol = nd.path.length > 1 ? (nd.combos < 5 ? 1 : 0.06) : 0.01;
        nd.names.forEach((n, a) => {
          const fq = nd.strat[a].reduce((s, x, j) => s + x * w[j], 0) / tot;
          assert.ok(Math.abs(fq - nd.freqs[n]) < tol, `${sp.id} ${f} ${nd.path} ${n}: ${fq} vs ${nd.freqs[n]}`);
        });
        for (let j = 0; j < w.length; j++) assert.ok(Math.abs(nd.strat.reduce((s, r) => s + r[j], 0) - 1) < 1e-9);
        if (nd.fold0) assert.ok(nd.evs[0].every(x => x === 0));
      }
    }
  }
});

test("format 2: a value per suit class (two-tone: the two suits off the flop swap), the ranges' combos, the EV step", async () => {
  const sid = "srp_btn_bb", rg = ranges(sid), sp = INDEX.spots.find(s => s.id === sid);
  assert.equal(INDEX.format, 2);
  const f = Object.keys(sp.flops).find(x => x[1] === x[3] && x[3] !== x[5] && new Set([x[0], x[2], x[4]]).size === 3);
  const e = await load(sid, f), board = new Set([f.slice(0, 2), f.slice(2, 4), f.slice(4, 6)].map(cardInt));
  assert.deepEqual(e.head.ev, [Math.round(sp.pot * 0.01 * 1e4) / 1e4, 32, 1.04]);
  // the flop's suits: one twice, one once; the two left swap
  const fs = [f[1], f[3], f[5]].map(x => SUITS.indexOf(x)), off = [0, 1, 2, 3].filter(x => !fs.includes(x));
  const swap = (c) => (c & 3) === off[0] ? c - off[0] + off[1] : (c & 3) === off[1] ? c - off[1] + off[0] : c;
  for (const p of [0, 1]) {
    const want = D.COMBOS.flatMap(([a, b], i) => rg.w[p][i] > 0 && !board.has(a) && !board.has(b) ? [i] : []);
    assert.deepEqual(e.combos[p], want);
  }
  let pairs = 0;
  for (const nd of e.nodes) {
    const pos = new Map(e.combos[nd.player].map((c, j) => [c, j]));
    e.combos[nd.player].forEach((ci, j) => {
      const [a, b] = D.COMBOS[ci].map(swap), k = pos.get(D.COMBOS.findIndex(([x, y]) => x === Math.min(a, b) && y === Math.max(a, b)));
      assert.ok(k !== undefined);
      if (k !== j) pairs++;
      nd.strat.forEach((r) => assert.equal(r[k], r[j]));
      nd.evs.forEach((r) => assert.equal(r[k], r[j]));
    });
  }
  assert.ok(pairs > 500, String(pairs));
});

test("each drill's (flop, node) draws: stored nodes only, the texture filter, bets by how often they're made", () => {
  assert.deepEqual(D.candidates(REP, "oop", {}).map(c => [c.f, c.path, c.w]), [["Ac4d3h", [], 100], ["9c8c7c", [], 50], ["Kc7d7h", [], 50]]);
  assert.deepEqual(D.candidates(REP, "ip", {}).map(c => c.f), ["Ac4d3h", "9c8c7c"]);
  assert.deepEqual(D.candidates(REP, "oop", { suits: "mono" }).map(c => c.f), ["9c8c7c"]);
  assert.deepEqual(D.candidates(REP, "oop", { pair: "paired", hi: "K" }).map(c => c.f), ["Kc7d7h"]);
  assert.deepEqual(D.candidates(REP, "oop", { suits: "mono", pair: "paired" }), []);
  const vb = D.candidates(REP, "vsbet", {}).map(c => [c.f, c.path.join("|"), Math.round(c.w * 100) / 100]);
  // A43: 33% bet 20% of the time, 75% 30% (the flop's weight split 2:3); 987: only the 33% bet is stored
  assert.deepEqual(vb, [["Ac4d3h", "check|bet 33%", 40], ["Ac4d3h", "check|bet 75%", 60], ["9c8c7c", "check|bet 33%", 50]]);
  // every shipped spot has draws for each drill
  for (const sp of INDEX.spots) for (const v of Object.keys(D.VIEWS)) assert.ok(D.candidates(report(sp.id), v, {}).length > 0, `${sp.id} ${v}`);
});

test("weighted picks", () => {
  const r = rng(1), n = [0, 0, 0];
  for (let i = 0; i < 6000; i++) n[D.pick([1, 0, 2], r)]++;
  assert.equal(n[1], 0);
  assert.ok(Math.abs(n[2] / n[0] - 2) < 0.2, String(n));
  assert.equal(D.pick([0, 0], r), -1);
});

test("hands are dealt from the acting player's range at the node, off the board, by weight", async () => {
  const sid = "srp_btn_bb", rep = report(sid), rg = ranges(sid), r = rng(7);
  for (const view of Object.keys(D.VIEWS)) {
    const cs = D.candidates(rep, view, {});
    for (let i = 0; i < 6; i++) {
      const c = cs[D.pick(cs.map(x => x.w), r)], e = await load(sid, c.f);
      const w = D.handWeights(e, c.path, rg), nd = D.nodeAt(e, c.path);
      const counts = new Map();
      for (let k = 0; k < 300; k++) {
        const pr = D.problem(rep, e, c, rg, r);
        assert.equal(pr.hero, rep.seats[nd.player]);
        const ci = D.COMBOS.findIndex(x => x[0] === pr.combo[0] && x[1] === pr.combo[1]), j = e.combos[nd.player].indexOf(ci);
        assert.ok(j >= 0 && w[j] > 0, "in the range at the node");
        assert.ok(rg.w[nd.player][ci] > 0);
        // the shown cards: one suit relabelling of the canonical flop and the combo, no card twice
        const shown = [...pr.hand, ...pr.board];
        assert.equal(new Set(shown).size, 5);
        const canon = [c.f.slice(0, 2), c.f.slice(2, 4), c.f.slice(4)];
        pr.board.forEach((x, t) => assert.equal(x[0], canon[t][0]));
        assert.deepEqual(pr.freqs, nd.strat.map(row => row[j]));
        assert.equal(pr.names, nd.names);
        counts.set(j, (counts.get(j) || 0) + 1);
      }
      // the most common hand comes up about as often as its weight says
      const tot = w.reduce((a, b) => a + b, 0), [jt, nt] = [...counts].sort((a, b) => b[1] - a[1])[0];
      assert.ok(nt / 300 < w[jt] / tot * 3 + 0.03, `${view} ${c.f}: ${nt}/300 for weight ${w[jt] / tot}`);
    }
  }
  // the weights: none on combos that touch the board; facing a bet after checking, scaled by the hand's check
  const e = await load(sid, "Ac4d3h"), board = ["Ac", "4d", "3h"].map(cardInt);
  for (const p of [[], ["check"], ["check", "bet 33%"]]) {
    const nd = D.nodeAt(e, p), w = D.handWeights(e, p, rg);
    e.combos[nd.player].forEach((ci, j) => { if (D.COMBOS[ci].some(x => board.includes(x))) assert.equal(w[j], 0); });
  }
  const w0 = D.handWeights(e, [], rg), w2 = D.handWeights(e, ["check", "bet 33%"], rg), root = D.nodeAt(e, []);
  w2.forEach((x, j) => assert.ok(Math.abs(x - w0[j] * root.strat[0][j]) < 1e-12));
});

test("pot at the node", () => {
  assert.deepEqual(D.potAt(REP, []), { pot: 5.5, bet: 0 });
  assert.deepEqual(D.potAt(REP, ["check", "bet 75%"]), { pot: 9.63, bet: 4.13 });
});

test("grading: the best EV, or a played mix, is correct; the EV given up is against the best", () => {
  const pr = { names: ["fold", "call", "raise to 7"], pot: 7.3, freqs: [0, 0.62, 0.38], evs: [0, 1.2, 1.18] };
  let g = D.grade(pr, 1);
  assert.deepEqual([g.verdict, g.ok, g.mixed, g.loss, g.best], ["Best", true, true, 0, 1]);
  g = D.grade(pr, 2);                                     // played 38%: correct, with the small EV gap shown
  assert.deepEqual([g.verdict, g.ok, g.mixed], ["Correct", true, true]);
  assert.ok(Math.abs(g.loss - 0.02) < 1e-9);
  g = D.grade(pr, 0);                                     // 1.2bb of a 7.3bb pot: 16%
  assert.deepEqual([g.verdict, g.ok], ["Blunder", false]);
  assert.ok(Math.abs(g.lossPct - 1.2 / 7.3 * 100) < 1e-9);
  const pure = { names: ["check", "bet 33%"], pot: 5.5, freqs: [0.95, 0.05], evs: [2, 1.95] };
  g = D.grade(pure, 1);                                   // 5% isn't a mix: 0.05bb, under 2% of the pot
  assert.deepEqual([g.verdict, g.ok, g.mixed], ["Inaccuracy", false, false]);
  assert.equal(D.grade({ ...pure, evs: [2, 1.8] }, 1).verdict, "Mistake");
  // equal EVs (the library's step) count as best, whatever the frequency
  assert.equal(D.grade({ ...pure, evs: [2, 1.99] }, 1).verdict, "Best");
});

test("score bookkeeping: accuracy, EV lost, streaks, the last 100", () => {
  let s = D.emptyScore();
  const ok = { ok: true, loss: 0, lossPct: 0 }, bad = { ok: false, loss: 0.5, lossPct: 10 };
  for (const g of [ok, ok, bad, ok, ok, ok]) s = D.addScore(s, g);
  assert.deepEqual([s.n, s.ok, s.streak, s.best, s.loss, s.lossPct], [6, 5, 3, 3, 0.5, 10]);
  assert.deepEqual(s.recent, [1, 1, 0, 1, 1, 1]);
  const m = D.summary(s);
  assert.ok(Math.abs(m.acc - 5 / 6) < 1e-12 && Math.abs(m.loss - 0.5 / 6) < 1e-12);
  for (let i = 0; i < 150; i++) s = D.addScore(s, bad);
  assert.equal(s.recent.length, 100);
  assert.deepEqual([s.n, s.streak, s.best], [156, 0, 3]);
  assert.deepEqual(D.addScore(undefined, ok).n, 1);       // a drill with no history yet
});

test("keys: 1-n answer, Space or Enter next once answered, Escape closes", () => {
  const pr = { names: ["fold", "call", "raise to 7"] };
  assert.deepEqual(D.keyAction("2", { pr, g: null }), { answer: 1 });
  assert.equal(D.keyAction("4", { pr, g: null }), null);
  assert.equal(D.keyAction(" ", { pr, g: null }), null);
  assert.equal(D.keyAction("1", { pr, g: { a: 0 } }), null);
  assert.deepEqual(D.keyAction(" ", { pr, g: { a: 0 } }), { next: true });
  assert.deepEqual(D.keyAction("Enter", { pr, g: { a: 0 } }), { next: true });
  assert.equal(D.keyAction("1", { pr: null, g: null }), null);   // still dealing
  assert.deepEqual(D.keyAction("Escape", { pr: null, g: null }), { close: true });
});

test("the solver's state for a problem: the preflop line, the flop action to the node, hero's cards", () => {
  const pr = { hero: "BB", hand: ["Kc", "6s"], board: ["Qh", "Th", "4h"], path: ["check", "bet 75%"] };
  const st = D.stateFor(REP, pr);
  assert.equal(st.hero, "BB");
  assert.deepEqual(st.cards, ["Kc", "6s", "Qh", "Th", "4h", null, null]);
  assert.deepEqual(st.actions.flop, [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 4.13 }]);
  assert.deepEqual(st.actions.preflop.slice(3), [{ pos: "BTN", action: "raise", to: 2.5 }, { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }]);
  assert.deepEqual(D.stateFor(REP, { ...pr, hero: "BTN", path: ["check"] }).actions.flop, [{ pos: "BB", action: "check" }]);
});
