// The session analyzer (gto/static/session.js, loaded on demand after hh.js): a file of many hands split per site,
// the hands it skips (and why), the queue of decisions (pause, resume, stop, the page cancelling a request) on a
// stubbed solver, the request and grade it makes through the page, the report's sums, the mistakes and the exports.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage } = require("./page.js");
const { synthSession } = require("./hh_synth.js");

const STATIC = path.join(__dirname, "..", "..", "gto", "static");
const { parseHandHistory, splitHands } = require(path.join(STATIC, "hh.js"));
const SS = require(path.join(STATIC, "session.js"));
const HH_DIR = path.join(__dirname, "..", "data", "hh");
const FILES = fs.readdirSync(HH_DIR).sort();
const HH = (f) => fs.readFileSync(path.join(HH_DIR, f), "utf8");
const TXT = FILES.filter(f => f.endsWith(".txt"));
const j = (x) => JSON.parse(JSON.stringify(x));

// a page with hh.js and session.js loaded, as the menu item loads them
async function page(opts) {
  const p = loadPage(opts);
  for (const f of ["hh.js", "session.js"]) vm.runInContext(fs.readFileSync(path.join(STATIC, f), "utf8"), p.ctx, { filename: f });
  await p.ready();
  return p;
}
// two iPoker hands in one session file: the fixture's game again under another game code
const twoIpoker = () => { const x = HH("ipoker_3bet_pot.xml"), g = x.match(/<game\b[\s\S]*?<\/game>/)[0]; return x.replace(g, g + "\n" + g.replace(/gamecode="\d+"/, 'gamecode="999"')); };
const OMAHA = HH("stars_6max_srp.txt").replace("Hold'em No Limit", "Omaha Pot Limit").replace("PokerStars Hand #249012345678", "PokerStars Hand #249012345600").replace("[Ah Qd]", "[Ah Qd Kc 2s]");
const OBSERVED = HH("stars_6max_srp.txt").replace("PokerStars Hand #249012345678", "PokerStars Hand #249012345601").replace(/^Dealt to .*\n/m, "");

test("splitHands: every site's hand alone is one hand, and in a file of all of them each parses as it does alone", () => {
  for (const f of FILES) {
    const parts = splitHands(HH(f));
    assert.equal(parts.length, 1, f);
    assert.deepEqual(j(parseHandHistory(parts[0])), j(parseHandHistory(HH(f))), f);
  }
  // concatenated as a session file (blank lines, CRLF, a preamble before the first)
  const all = "Exported by a tracker\r\n\r\n" + TXT.map(f => HH(f).replace(/\n/g, "\r\n")).join("\r\n\r\n\r\n");
  const parts = splitHands(all);
  assert.equal(parts.length, TXT.length);
  parts.forEach((t, k) => assert.deepEqual(j(parseHandHistory(t)), j(parseHandHistory(HH(TXT[k]))), TXT[k]));
  // the same site back to back, no blank line between hands
  assert.equal(splitHands(HH("888_6max_srp.txt") + HH("888_9max_mtt_ante.txt")).length, 2);
  assert.equal(splitHands(HH("party_6max_allin.txt") + "\n" + HH("party_mtt_bounty_allin.txt")).length, 2);
  // iPoker: a <session> per <game>, each with the session's <general> (stakes, nickname)
  const xp = splitHands(twoIpoker());
  assert.equal(xp.length, 2);
  const a = parseHandHistory(xp[0]), b = parseHandHistory(xp[1]);
  assert.deepEqual(j(a), j(parseHandHistory(HH("ipoker_3bet_pot.xml"))));
  assert.deepEqual([b.heroName, b.bb, b.warnings], [a.heroName, a.bb, []]);
  assert.equal(splitHands("").length, 0);
});

test("a single paste of several hands still imports the first, with the same warning", () => {
  const two = parseHandHistory(HH("stars_6max_srp.txt") + "\n\n" + HH("stars_3bet_pot.txt"));
  assert.deepEqual(two.warnings, ["2 hands pasted: imported the first."]);
  assert.deepEqual({ ...j(two), warnings: [] }, j(parseHandHistory(HH("stars_6max_srp.txt"))));
  assert.deepEqual(parseHandHistory(twoIpoker()).warnings, ["2 hands pasted: imported the first."]);
});

test("hand numbers per site", () => {
  const ids = Object.fromEntries(FILES.map(f => [f, SS.handId(splitHands(HH(f))[0])]));
  assert.equal(ids["stars_6max_srp.txt"], "249012345678");
  assert.equal(ids["gg_allin_preflop.txt"], "HD2051234567");
  assert.equal(ids["888_6max_srp.txt"], "1234567890");
  assert.equal(ids["ipoker_3bet_pot.xml"], "3858162574");
  assert.ok(Object.values(ids).every(Boolean), JSON.stringify(ids));
});

test("prepare: a session file across sites, synthetic hands and ones to skip, counted by why", async () => {
  const p = await page();
  const synth = synthSession(60, 3);
  const text = [...TXT.map(HH), synth, OMAHA, OBSERVED, HH("stars_6max_srp.txt") /* again */].join("\n\n");
  const prog = [];
  const prep = await p.ctx.Session.prepare(text, (t) => p.ctx.importHand(t), (k, n) => prog.push([k, n]), 25, async () => {});
  const want = TXT.length + 60 + 3;
  assert.equal(prep.total, want);
  assert.deepEqual(j(prep.skipped), { nonHoldem: 1, noHero: 1, noDecisions: prep.skipped.noDecisions, duplicate: 1, failed: 0 });
  // a walk to hero's big blind is the only hand without a decision of hero's
  const walks = splitHands(synth).filter(t => /HeroName: posts big blind/.test(t) && !/HeroName: (folds|calls|raises|checks|bets)/.test(t)).length;
  assert.equal(prep.skipped.noDecisions, walks);
  assert.equal(prep.hands.length, want - 3 - walks);
  assert.deepEqual(prog.map(x => x[1]), [want, want, want, want].slice(0, Math.floor((want - 1) / 25)));
  // each hand is the page's own import of its text, numbered as in the file
  const h = prep.hands.find(x => x.id === "249012345678");
  assert.deepEqual(j(h.state), j(p.ctx.importHand(HH("stars_6max_srp.txt")).state));
  assert.equal(h.n, TXT.indexOf("stars_6max_srp.txt") + 1);
  assert.deepEqual([h.site, h.hero, h.hole], ["PokerStars", "BTN", "AhQd"]);
  // a hand the importer can't read is listed with its message
  const bad = await p.ctx.Session.prepare(HH("stars_6max_srp.txt").replace(/^Seat \d.*\n/gm, ""), (t) => p.ctx.importHand(t), null, 100, async () => {});
  assert.equal(bad.skipped.failed, 1);
  assert.match(bad.errors[0].msg, /No players found/);
});

test("decisions, scopes and spot states", () => {
  const st = { players: 6, hero: "BTN", stack: 100, cards: ["Ah", "Qd", "Qs", "8d", "3c", "7h", "2s"],
               actions: { preflop: [{ pos: "UTG", action: "fold" }, { pos: "BTN", action: "raise", to: 2.5 }, { pos: "BB", action: "call" }],
                          flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 1.8 }, { pos: "BB", action: "call" }],
                          turn: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "check" }], river: [{ pos: "BB", action: "bet", to: 5 }, { pos: "BTN", action: "fold" }] } };
  assert.deepEqual(SS.decisions(st), [["preflop", 1], ["flop", 1], ["turn", 1], ["river", 1]]);
  const hands = [{ state: st }, { state: st }];
  assert.deepEqual(SS.buildItems(hands, "preflop").map(x => [x.h, x.s]), [[0, "preflop"], [1, "preflop"]]);
  assert.deepEqual(SS.buildItems(hands, "library").map(x => x.s), ["preflop", "flop", "preflop", "flop"]);
  assert.equal(SS.buildItems(hands, "all").length, 8);
  const sp = SS.spotState(st, "turn", 1);
  assert.deepEqual(sp.cards, ["Ah", "Qd", "Qs", "8d", "3c", "7h", null]);
  assert.deepEqual([sp.actions.turn.length, sp.actions.river.length, sp.actions.flop.length], [1, 0, 3]);
  assert.equal(st.actions.turn.length, 2);   // (a copy)
});

test("preflop situations and postflop spots", () => {
  const A = (s) => s ? s.split(",").map(t => { const [pos, a] = t.trim().split(" "); return { pos, action: { f: "fold", c: "call", r: "raise", a: "all-in", x: "check" }[a[0]], ...(a[0] === "r" ? { to: +a.slice(1) } : {}) }; }) : [];
  const cases = [
    ["", "UTG", "RFI"], ["UTG f, HJ f", "CO", "RFI"], ["UTG c", "BTN", "vs limp"], ["UTG c, HJ f, CO f, BTN f, SB c", "BB", "BB vs limp"],
    ["UTG r2.5", "HJ", "vs open"], ["UTG r2.5, HJ c", "BTN", "vs open + call"], ["UTG r2.5, HJ f, CO f, BTN f", "SB", "SB vs open"],
    ["UTG r2.5, HJ f, CO f, BTN f, SB f", "BB", "BB defence"], ["UTG r2.5, HJ c, CO f, BTN f, SB f", "BB", "BB defence"],
    ["UTG r2.5, HJ r8, CO f, BTN f, SB f, BB f", "UTG", "vs 3bet"], ["UTG r2.5, HJ r8", "CO", "cold vs 3bet"],
    ["UTG r2.5, HJ r8, CO f, BTN f, SB f, BB f, UTG r20", "HJ", "vs 4bet+"], ["UTG c, HJ r4, CO f, BTN f, SB f, BB f", "UTG", "limp vs raise"],
    ["UTG a", "HJ", "vs open"], ["UTG a, HJ a", "CO", "vs open + call"],   // (an all-in after an all-in calls)
  ];
  for (const [pre, hero, want] of cases) assert.equal(SS.preflopSituation(A(pre), hero), want, `${pre} / ${hero}`);
  const st = { hero: "BB", actions: { preflop: A("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: [{ pos: "BB", action: "bet", to: 2 }, { pos: "BTN", action: "call" }], turn: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 4 }], river: [] } };
  const t = SS.classify(st, "turn", 0, { toCall: 0, pot: 9.5, ip: false, players: 2, raises: 1 });
  assert.deepEqual([t.role, t.spot, t.flopAggr, t.act, t.cls, t.facing], ["caller", "SRP · OOP · first to act", true, "check", "passive", false]);
  const f = SS.classify({ ...st, hero: "BTN" }, "flop", 1, { toCall: 2, pot: 7.5, ip: true, players: 2, raises: 2 });
  assert.deepEqual([f.role, f.spot, f.cls], ["PFR", "3bet pot · IP · facing bet", "passive"]);
  assert.equal(SS.classify(st, "flop", 0, { toCall: 0, pot: 5.5, ip: false, players: 3, raises: 0 }).spot, "Multiway · OOP · first to act");
  const pf = SS.classify(st, "preflop", 5, { toCall: 1.5, pot: 4, ip: true, players: 6, raises: 1 });
  assert.deepEqual([pf.pf, pf.act, pf.spot], ["BB defence", "call", undefined]);
});

// ---- the queue on a stubbed solver
function stubSession(nHands = 3) {
  const st = { players: 6, hero: "BTN", stack: 100, cards: ["Ah", "Qd", "Qs", "8d", "3c", "7h", "2s"],
               actions: { preflop: [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "fold" }, { pos: "CO", action: "fold" }, { pos: "BTN", action: "raise", to: 2.5 }, { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }],
                          flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", to: 1.8 }, { pos: "BB", action: "call" }], turn: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "check" }], river: [] } };
  const hands = Array.from({ length: nHands }, (_, k) => ({ n: k + 1, id: String(100 + k), site: "PokerStars", hero: "BTN", hole: "AhQd", state: st }));
  return SS.newSession({ hands, skipped: {}, errors: [], total: nHands }, "all", "test.txt");
}
// a host whose solver answers when the test says so; grades like the page (the matched action's freq, EV loss)
function stubHost() {
  const asked = [], waiting = [];
  const H = {
    asked, waiting,
    spot: (st) => ({ payload: { board: st.cards.slice(2).filter(Boolean).join(""), n: Object.values(st.actions).flat().length }, pot: 5, toCall: 0, ip: true, players: 2, raises: 1 }),
    advise: (req) => new Promise(res => { asked.push(req); waiting.push(res); }),
    grade: (res, rec) => { const m = res.actions.find(a => a.action.startsWith(rec.action)) || res.actions[0], top = Math.max(...res.actions.map(a => a.ev_bb));
      return { m: m.action, f: m.freq, b: res.actions[0].action, bf: res.actions[0].freq, l: Math.round((top - m.ev_bb) * 100) / 100, e: m.ev_bb, o: top }; },
    answer(res) { waiting.shift()(res || { actions: [{ action: "check", freq: 0.6, ev_bb: 1 }, { action: "bet 33%", freq: 0.4, ev_bb: 0.75 }, { action: "raise", freq: 0, ev_bb: 0.5 }] }); },
  };
  return H;
}
const tick = async () => { for (let i = 0; i < 6; i++) await new Promise(r => setImmediate(r)); };

test("the queue: in order, pause after the decision in flight, resume, stop drops it, a cancelled request is asked again", async () => {
  const sess = stubSession(3), H = stubHost(), saved = [];
  let t = 0;
  const R = SS.runner(sess, H, { save: async (s) => saved.push(s.next), now: () => t, saveEvery: 0 });
  assert.equal(sess.items.length, 9);   // preflop, flop, turn per hand
  const p1 = R.run();
  await tick();
  assert.equal(R.running, true);
  assert.deepEqual(H.asked.map(a => a.n), [3]);   // hand 1's preflop decision: the 3 actions before it
  t += 1500; H.answer(); await tick();
  t += 4000; H.answer(); await tick();
  assert.equal(sess.next, 2);
  R.pause(); assert.equal(R.stopping, "pause");
  t += 6000; H.answer(); await p1;   // the decision in flight is kept
  assert.deepEqual([sess.next, sess.status, R.running], [3, "paused", false]);
  assert.deepEqual(sess.results.map(r => [r.h, r.s, r.sec]), [[0, "preflop", 1.5], [0, "flop", 4], [0, "turn", 6]]);
  assert.deepEqual(sess.timing, { preflop: [1.5, 1], flop: [4, 1], turn: [6, 1] });
  assert.ok(saved.includes(3));
  // resumed: a request the page cancelled (its own solve took the worker) is asked again
  const p2 = R.run(); await tick();
  H.answer({ cancelled: true }); await tick();
  assert.equal(H.asked.length, 5); assert.equal(sess.next, 3);
  H.answer({ skipped: "not a flop-library spot" }); await tick();   // (an answer the scope skipped)
  H.answer({ error: "boom" }); await tick();
  R.cancel(); H.answer(); await p2;   // stopped: the answer in flight is dropped
  assert.deepEqual([sess.next, sess.status], [5, "stopped"]);
  assert.deepEqual(sess.results.slice(3).map(r => r.status), ["skipped", "error"]);
  // three cancellations in a row: an error, the queue goes on
  const p3 = R.run(); await tick();
  for (let k = 0; k < 3; k++) { H.answer({ cancelled: true }); await tick(); }
  assert.deepEqual([sess.results[5].status, sess.results[5].err, sess.next], ["error", "cancelled", 6]);
  for (let k = 0; k < 3; k++) { H.answer(); await tick(); }
  await p3;
  assert.deepEqual([sess.next, sess.status], [9, "done"]);
  await R.run();   // nothing left: no request
  assert.equal(H.waiting.length, 0);
});

test("the queue in the flop-library scope asks flops with library_only, and widening asks only what's left", async () => {
  const sess = stubSession(2); sess.scope = "library"; sess.items = SS.buildItems(sess.hands, "library");
  const H = stubHost(), R = SS.runner(sess, H);
  const p = R.run();
  for (let k = 0; k < 4; k++) { await tick(); H.answer(k === 1 ? { skipped: "not a flop-library spot" } : undefined); }
  await p;
  assert.deepEqual(H.asked.map(a => !!a.library_only), [false, true, false, true]);
  assert.deepEqual(sess.results.map(r => r.status), ["ok", "skipped", "ok", "ok"]);
  const w = SS.widen(sess, "all");
  assert.deepEqual(w.items.slice(w.next).map(x => [x.h, x.s]), [[0, "flop"], [0, "turn"], [1, "turn"]]);
  assert.deepEqual([w.results.length, w.status, w.scope], [3, "paused", "all"]);
});

test("ETA: each street's budget until two answers are timed, then their average", () => {
  const sess = stubSession(2);
  // server Fast [2.5, 2.5]: preflop 0.2, flop 3, turn 5.5
  assert.equal(SS.eta(sess, [2.5, 2.5]), 2 * (0.2 + 3 + 5.5));
  assert.equal(SS.estimate("river", "all", [8, 3]), 8 + 6 + 0.5);
  assert.equal(SS.estimate("flop", "library", [8, 3]), 0.4);
  sess.next = 3; sess.timing = { preflop: [0.1, 1], flop: [2, 2], turn: [9, 1] };
  assert.equal(SS.eta(sess, [2.5, 2.5]), 0.2 + 1 + 5.5);
});

// ---- the report
function graded(h, s, extra, g) { return { h, s, i: 0, street: s, status: "ok", mix: { fold: 0.2, passive: 0.3, aggr: 0.5 }, ...extra, ...(g ? { g } : {}) }; }
test("the report: EV lost, bb/100, per street, position, spot and preflop situation, frequencies, mistakes", () => {
  const sess = stubSession(4);
  sess.hands[2].state = { ...sess.hands[2].state, hero: "BB" };
  const G = (f, l, bf = 0.7) => ({ m: "x", f, b: "best", bf, l, ...(l != null ? { e: 0, o: l } : {}) });
  sess.results = [
    graded(0, "preflop", { pos: "BTN", pf: "RFI", cls: "aggr" }, G(1, null)),
    graded(0, "flop", { pos: "BTN", spot: "SRP · IP · checked to", role: "PFR", facing: false, cls: "aggr", mix: { fold: 0, passive: 0.25, aggr: 0.75 } }, G(0.6, 0.1)),
    graded(0, "turn", { pos: "BTN", spot: "SRP · IP · checked to", role: "PFR", facing: false, flopAggr: true, cls: "passive", mix: { fold: 0, passive: 0.1, aggr: 0.9 } }, G(0.1, 1.5)),
    graded(1, "preflop", { pos: "BTN", pf: "RFI", cls: "fold" }, G(0.01, null, 0.99)),
    graded(1, "flop", { pos: "BTN", spot: "SRP · IP · checked to", role: "PFR", facing: false, cls: "passive", mix: { fold: 0, passive: 0.5, aggr: 0.5 } }, G(0.5, 0.4)),
    graded(2, "preflop", { pos: "BB", pf: "BB defence", cls: "fold", mix: { fold: 0.8, passive: 0.2, aggr: 0 } }, G(0.8, null)),
    graded(2, "flop", { pos: "BB", spot: "SRP · OOP · facing bet", role: "caller", facing: true, cls: "fold", mix: { fold: 0.02, passive: 0.9, aggr: 0.08 } }, G(0.02, 2.25)),
    { h: 3, s: "flop", i: 0, street: "flop", pos: "BTN", status: "skipped" },
    { h: 3, s: "turn", i: 0, street: "turn", pos: "BTN", spot: "Multiway · IP · checked to", status: "ungraded", mix: { fold: 0, passive: 1, aggr: 0 } },
    { h: 3, s: "river", i: 0, street: "river", pos: "BTN", status: "error", err: "x" },
  ];
  sess.next = sess.results.length;
  const rp = SS.report(sess, 3);
  assert.deepEqual([rp.hands, rp.handsDone, rp.done, rp.graded, rp.ev, rp.lost], [4, 4, 10, 7, 4, 4.25]);
  assert.equal(rp.bb100, 106.25);   // 4.25 bb over 4 hands
  assert.deepEqual([rp.ok, rp.bad, rp.skipped, rp.ungraded, rp.errors], [4, 2, 1, 1, 1]);
  const row = (rows, k) => { const r = rows.find(x => x.key === k); return r && [r.n, r.graded, r.ev, r.lost, r.ok, r.bad]; };
  assert.deepEqual(rp.byStreet.map(r => r.key), ["preflop", "flop", "turn"]);   // (in street order; the skipped and failed aren't counted)
  assert.deepEqual(row(rp.byStreet, "flop"), [3, 3, 3, 2.75, 2, 1]);
  assert.deepEqual(row(rp.byStreet, "turn"), [2, 1, 1, 1.5, 0, 0]);
  assert.deepEqual(row(rp.byPos, "BB"), [2, 2, 1, 2.25, 1, 1]);
  assert.deepEqual(rp.byPos.map(r => r.key), ["HJ", "BTN", "BB"].filter(k => rp.byPos.some(r => r.key === k)));
  assert.deepEqual(row(rp.bySpot, "SRP · IP · checked to"), [3, 3, 3, 2, 2, 0]);
  assert.equal(rp.bySpot[0].key, "SRP · OOP · facing bet");   // (most EV lost first)
  assert.equal(rp.bySpot.find(r => r.key === "SRP · IP · checked to").avg, 0.667);
  assert.deepEqual(rp.byPf.map(r => [r.key, r.n, r.ok, r.ev]), [["RFI", 2, 1, 0], ["BB defence", 1, 1, 0]]);
  const fq = Object.fromEntries(rp.freqs.map(f => [f.label, [f.n, Math.round(f.you * 100), Math.round(f.gto * 100)]]));
  assert.deepEqual(fq["Open (RFI)"], [2, 50, 50]);
  assert.deepEqual(fq["Flop c-bet"], [2, 50, 63]);
  assert.deepEqual(fq["Turn barrel"], [1, 0, 90]);
  assert.deepEqual(fq["Flop vs c-bet: fold"], [1, 100, 2]);
  assert.deepEqual(fq["BB defence: fold"], [1, 100, 80]);
  assert.equal(fq["River bet"], undefined);   // (no graded river decision)
  // the biggest mistakes by EV lost (top 3); preflop chart misses apart, the clearest first
  assert.deepEqual(rp.mistakes.map(r => [r.h, r.s, r.g.l]), [[2, "flop", 2.25], [0, "turn", 1.5], [1, "flop", 0.4]]);
  assert.deepEqual(rp.offChart.map(r => [r.h, r.s]), [[1, "preflop"]]);
});

test("exports: a CSV row per decision asked, and JSON with the summary", () => {
  const sess = stubSession(1);
  sess.hands[0].id = 'a,"b"';
  sess.results = [graded(0, "preflop", { pos: "BTN", pf: "RFI", act: "raise to 2.5", cls: "aggr", src: "chart", sec: 0.1 }, { m: "raise", f: 0.9, b: "raise", bf: 0.9, l: null }),
                  graded(0, "flop", { pos: "BTN", spot: "SRP · IP · checked to", act: "bet 1.8", cls: "aggr", src: "library", pot: 5.5, facing: false, ip: true }, { m: "bet 33%", f: 0.2, b: "check", bf: 0.8, l: 0.31, e: 1.2, o: 1.51 })];
  const csv = SS.toCSV(sess).trim().split("\n");
  assert.equal(csv.length, 3);
  assert.equal(csv[0], "hand,hand_id,site,position,hole,street,board,situation,action,solver_freq,best_action,best_freq,ev_bb,best_ev_bb,ev_lost_bb,grade,source,status,pot_bb,facing_bet,in_position,seconds");
  assert.equal(csv[1], '1,"a,""b""",PokerStars,BTN,AhQd,preflop,,RFI,raise to 2.5,0.9,raise,0.9,,,,ok,chart,ok,,,,0.1');
  assert.equal(csv[2], '1,"a,""b""",PokerStars,BTN,AhQd,flop,Qs8d3c,SRP · IP · checked to,bet 1.8,0.2,check,0.8,1.2,1.51,0.31,ok,library,ok,5.5,false,true,');
  const js = JSON.parse(SS.toJSON(sess));
  assert.deepEqual([js.scope, js.summary.lost, js.summary.graded, js.decisions.length, js.summary.mistakes[0].ev_lost_bb], ["all", 0.31, 2, 2, 0.31]);
  assert.equal(js.decisions[1].board, "Qs8d3c");
});

test("the page host: the request is the page's own for that decision (at Fast, GTO), graded as the page grades it", async () => {
  const p = await page({ storage: { "gto-opponents": "pool", "gto-speed": "accurate" } });
  const imp = p.ctx.importHand(HH("stars_6max_srp.txt")), host = p.ctx.Session.pageHost();
  const before = p.val("S");
  const sp = j(host.spot(p.ctx.Session.spotState(imp.state, "flop", 1)));
  assert.deepEqual(p.val("S"), before);   // the page's hand is untouched
  // the same spot loaded into the page: its payload, at Fast's budgets, without the pool
  p.S = p.ctx.Session.spotState(imp.state, "flop", 1);
  const want = p.payload();
  delete want.opponents;
  assert.deepEqual(sp.payload, { ...want, budget: 8, prior_budget: 3, speed: "fast" });
  assert.deepEqual([sp.pot, sp.toCall, sp.ip, sp.players, sp.raises], [5.5, 0, true, 2, 1]);
  assert.equal(host.spot(p.ctx.Session.spotState(imp.state, "flop", 0)), null);   // (BB's turn there, not hero's)
  const res = { street: "flop", actions: [{ action: "check", freq: 0.3, ev_bb: 2 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.7, ev_bb: 2.5 }], solver: { exploitability_pct: 1 } };
  assert.deepEqual(j(host.grade(res, { action: "check" })), { m: "check", f: 0.3, b: "bet 33%", bf: 0.7, l: 0.5, e: 2, o: 2.5 });
  assert.equal(host.grade({ multiway: true, actions: res.actions }, { action: "check" }), null);   // the betting rules: no grade
});

test("a limp the preflop chart leaves out (open-limp, SB completes) is graded off the chart, not left ungraded", async () => {
  const p = await page();
  const host = p.ctx.Session.pageHost();
  // the advisor's chart answer for the small blind folded to: raise or fold, no limp
  const sb = { street: "preflop", actions: [{ action: "raise", freq: 0, amount_bb: 3 }, { action: "fold", freq: 1 }] };
  assert.deepEqual(j(host.grade(sb, { action: "call" })), { m: "call", f: 0, b: "fold", bf: 1, l: null });
  assert.deepEqual(j(p.ctx.gradeWith(sb, { action: "call" })), { m: "call", f: 0, b: "fold", bf: 1, l: null });
  assert.equal(p.run(`gradeMark(${JSON.stringify(p.ctx.gradeWith(sb, { action: "call" }))})`), "✗");
  // an answer listing the action grades it by its own frequency as before
  const bb = { street: "preflop", actions: [{ action: "raise", freq: 0.2, amount_bb: 4 }, { action: "check", freq: 0.8 }] };
  assert.deepEqual(j(host.grade(bb, { action: "check" })), { m: "check", f: 0.8, b: "check", bf: 0.8, l: null });
  // postflop a missing passive action stays ungraded (the solver lists every legal one)
  assert.equal(host.grade({ street: "flop", actions: [{ action: "check", freq: 1, ev_bb: 1 }] }, { action: "call" }), null);
});

test("a preflop jam the chart doesn't list is graded as the chart's raise (or 0% where it never raises), with a note", async () => {
  const p = await page();
  const host = p.ctx.Session.pageHost();
  // facing an open: the chart 3bets to 7.5 or folds, no all-in
  const vsOpen = { street: "preflop", actions: [{ action: "raise to 7.5", amount_bb: 7.5, freq: 0.4 }, { action: "fold", freq: 0.6 }] };
  const g = j(host.grade(vsOpen, { action: "all-in" }));
  assert.deepEqual(g, { m: "raise to 7.5", f: 0.4, b: "fold", bf: 0.6, l: null, v: "all-in" });
  assert.deepEqual(j(p.ctx.gradeWith(vsOpen, { action: "all-in" })), g);
  assert.match(p.run(`gradeTip(${JSON.stringify(g)})`), /Your line \(all-in\) 40% \(graded as the chart's raise to 7\.5: it has no all-in here\)/);
  // a chart that only folds or calls: the jam is off it
  const calls = { street: "preflop", actions: [{ action: "call", freq: 0.7 }, { action: "fold", freq: 0.3 }] };
  assert.deepEqual(j(host.grade(calls, { action: "all-in" })), { m: "all-in", f: 0, b: "call", bf: 0.7, l: null });
  assert.deepEqual(j(host.grade(calls, { action: "raise", to: 9 })), { m: "raise to 9", f: 0, b: "call", bf: 0.7, l: null });
  // a chart that only jams: a smaller raise is graded as its jam
  const jams = { street: "preflop", actions: [{ action: "all-in", amount_bb: 20, freq: 0.5 }, { action: "fold", freq: 0.5 }] };
  assert.deepEqual(j(host.grade(jams, { action: "raise", to: 9 })), { m: "all-in", f: 0.5, b: "all-in", bf: 0.5, l: null, v: "raise to 9" });
  // listed actions as before; postflop an unlisted jam stays ungraded
  assert.equal(j(host.grade({ ...vsOpen, actions: [...vsOpen.actions, { action: "all-in", amount_bb: 100, freq: 0 }] }, { action: "all-in" })).v, undefined);
  assert.equal(host.grade({ street: "flop", actions: [{ action: "check", freq: 1, ev_bb: 1 }, { action: "bet 33%", amount_bb: 2, freq: 0, ev_bb: 0.5 }] }, { action: "all-in" }), null);
  // the hand review and the copied hand say how it was graded
  p.S = { players: 6, hero: "CO", stack: 100, cards: ["Ah", "Kd", null, null, null, null, null],
    actions: { preflop: [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "raise", to: 2.5 }, { pos: "CO", action: "all-in", g }, { pos: "BTN", action: "fold" }, { pos: "SB", action: "fold" }, { pos: "BB", action: "fold" }, { pos: "HJ", action: "fold" }], flop: [], turn: [], river: [] } };
  p.run("renderAll()");
  assert.match(p.byId("adviceNote").innerHTML + p.byId("result").innerHTML, /<b>all-in<\/b> · solver 40% <span class="muted">as the chart's raise to 7.5<\/span>/);
  assert.match(p.run("handText()"), /CO all-in \[✓ solver 40% as the chart's raise to 7.5\]/);
});

test("the report counts decisions left ungraded by why: the betting rules' multiway answers apart from the rest", () => {
  const sess = stubSession(1);
  sess.results = [{ h: 0, s: "turn", i: 0, street: "turn", pos: "BTN", status: "ungraded", src: "rules" },
                  { h: 0, s: "flop", i: 0, street: "flop", pos: "BTN", status: "ungraded", src: "solve" }];
  const rp = SS.report(sess);
  assert.deepEqual([rp.ungraded, rp.rules], [2, 1]);
});

test("a synthetic session end to end through the page: every decision asked once, in order, and graded", async () => {
  const p = await page({ server: true });
  const S = p.ctx.Session, prep = await S.prepare(synthSession(50, 11), (t) => p.ctx.importHand(t), null, 100, async () => {});
  const sess = S.newSession(prep, "all", "synth.txt"), host = S.pageHost();
  const asked = [];
  host.advise = async (req) => { asked.push(req); return { actions: [{ action: "fold", freq: 0.1, ev_bb: 0 }, { action: "check", freq: 0.25, ev_bb: 0.2 }, { action: "call", freq: 0.25, ev_bb: 0.2 }, { action: req.board ? "bet 75%" : "raise", amount_bb: 3, freq: 0.3, ev_bb: 0.3 }, { action: "all-in", freq: 0.1, ev_bb: 0.1 }] }; };
  await S.runner(sess, host).run();
  assert.equal(sess.status, "done");
  assert.equal(asked.length, sess.items.length);
  assert.ok(sess.items.length > 80, String(sess.items.length));
  assert.ok(sess.results.every(r => r.status === "ok"), JSON.stringify(sess.results.find(r => r.status !== "ok")));
  // each request is for hero's hand on that street's board
  sess.results.forEach((r, k) => { const h = sess.hands[r.h]; assert.equal(asked[k].hole, h.hole); assert.equal(asked[k].board.length, 2 * { preflop: 0, flop: 3, turn: 4, river: 5 }[r.s]); });
  const rp = S.report(sess);
  assert.equal(rp.handsDone, prep.hands.length);
  assert.ok(rp.byPf.some(r => r.key === "RFI") && rp.byPf.some(r => r.key === "BB defence") && rp.bySpot.length >= 4);
  assert.ok(rp.freqs.some(f => f.label === "Flop c-bet"));
});

// ---- across sessions: summaries kept, summed, leaks ranked
function reported(created, name, rows) {
  const sess = stubSession(rows.hands || 10);
  sess.created = created; sess.name = name;
  sess.results = rows.results; sess.next = sess.results.length;
  return sess;
}
const G2 = (f, l) => ({ m: "x", f, b: "best", bf: 0.7, l, ...(l != null ? { e: 0, o: l } : {}) });
const many = (k, s, extra, g) => Array.from({ length: k }, (_, n) => graded(n % 10, s, extra, g));

test("history: a session's summary, kept newest first, one per session", () => {
  assert.equal(SS.summarize(stubSession(2)), null);   // nothing asked yet: nothing to keep
  const a = reported(1000, "a.txt", { results: [...many(4, "flop", { pos: "BTN", spot: "SRP · IP · checked to", role: "PFR", facing: false, cls: "aggr" }, G2(0.5, 0.25))] });
  const sa = SS.summarize(a), rp = SS.report(a);
  assert.deepEqual([sa.id, sa.name, sa.scope, sa.hands, sa.decisions, sa.graded, sa.ev, sa.lost, sa.bb100], [1000, "a.txt", "all", 4, 4, 4, 4, 1, rp.bb100]);
  assert.deepEqual(j(sa.bySpot), j(rp.bySpot.map(({ key, n, graded, ok, bad, ev, lost }) => ({ key, n, graded, ok, bad, ev, lost }))));
  const b = SS.summarize(reported(2000, "b.txt", { hands: 6, results: many(2, "preflop", { pos: "BTN", pf: "RFI", cls: "fold" }, G2(0.01, null)) }));
  let h = SS.remember(SS.remember([], sa), b);
  assert.deepEqual(h.map(x => x.id), [2000, 1000]);
  h = SS.remember(h, { ...sa, hands: 9 });            // the same session again: replaced, not added
  assert.deepEqual(h.map(x => [x.id, x.hands]), [[2000, 2], [1000, 9]]);
  assert.deepEqual(SS.remember(null, null), []);
  assert.deepEqual(SS.remember([null, { x: 1 }, sa], null).map(x => x.id), [1000]);   // (anything else stored is dropped)
  // the same hands analyzed again (another session, any scope): the newer replaces the older
  const again = reported(3000, "b-again.txt", { hands: 6, results: many(2, "preflop", { pos: "BTN", pf: "RFI", cls: "fold" }, G2(0.01, null)) });
  again.scope = "preflop";
  const sg = SS.summarize(again);
  assert.equal(sg.fp, b.fp);
  assert.notEqual(sa.fp, SS.summarize(reported(4000, "c", { hands: 3, results: many(2, "flop", {}, G2(0.5, 0)) })).fp);
  assert.deepEqual(SS.remember(h, sg).map(x => x.id), [3000, 1000]);   // (b.txt's entry gone)
  const lots = Array.from({ length: 250 }, (_, k) => ({ ...sa, id: k, date: k, fp: String(k) }));
  assert.equal(SS.remember(lots, b).length, 200);
});

test("history: sums over sessions, frequencies weighted by decisions, each session's bb/100 oldest first", () => {
  const spot = { pos: "BTN", spot: "SRP · IP · checked to", role: "PFR", facing: false, cls: "aggr", mix: { fold: 0, passive: 0.5, aggr: 0.5 } };
  const a = SS.summarize(reported(1000, "a.txt", { results: [...many(10, "flop", spot, G2(0.5, 0.1)), ...many(10, "preflop", { pos: "BTN", pf: "RFI", cls: "aggr" }, G2(0.9, null))] }));
  const b = SS.summarize(reported(2000, "b.txt", { results: many(30, "flop", { ...spot, cls: "passive" }, G2(0.5, 0.2)) }));
  const tr = SS.trends([b, a]);
  assert.deepEqual([tr.sessions, tr.hands, tr.decisions, tr.ev, tr.lost], [2, 20, 50, 40, 7]);
  assert.equal(tr.bb100, 35);                                    // 7bb over 20 hands
  assert.deepEqual(tr.byStreet.map(r => [r.key, r.n, r.ev, r.lost]), [["preflop", 10, 0, 0], ["flop", 40, 40, 7]]);
  assert.deepEqual(tr.bySpot.map(r => [r.key, r.n, r.lost, r.avg]), [["SRP · IP · checked to", 40, 7, 0.175]]);
  const cbet = tr.freqs.find(f => f.label === "Flop c-bet");    // you bet 10 of 40; the solver half of each
  assert.deepEqual([cbet.n, cbet.you, cbet.gto], [40, 0.25, 0.5]);
  assert.deepEqual(tr.series.map(x => [x.id, x.bb100]), [[1000, 10], [2000, 60]]);
  assert.deepEqual(SS.trends([]).sessions, 0);
});

test("history: leaks past their thresholds, the clearest first", () => {
  const spot = (k, cls, l) => many(k, "flop", { pos: "BB", spot: "SRP · OOP · facing bet", role: "caller", facing: true, cls, mix: { fold: 0.3, passive: 0.65, aggr: 0.05 } }, G2(0.3, l));
  const pf = (k, f) => many(k, "preflop", { pos: "BB", pf: "BB defence", cls: "fold", mix: { fold: 0.3, passive: 0.65, aggr: 0.05 } }, G2(f, null));
  // folding to a c-bet 100% of the time where the solver folds 30% (20 decisions), losing 0.6bb a decision there,
  // and BB defence folds the chart plays 1% of the time (20 decisions, 0 in the mix)
  const s1 = SS.summarize(reported(1, "x", { results: [...spot(20, "fold", 0.6), ...pf(20, 0.01)] }));
  const L = SS.leaks(SS.trends([s1]));
  assert.deepEqual(L.map(l => [l.kind, l.key]).sort(), [["freq", "BB defence: fold"], ["freq", "Flop vs c-bet: fold"], ["pf", "BB defence"], ["spot", "SRP · OOP · facing bet"]]);
  assert.ok(L.every((l, k) => !k || L[k - 1].score >= l.score));
  // under the thresholds: none
  const s2 = SS.summarize(reported(2, "y", { results: [...spot(5, "fold", 0.6), ...pf(5, 0.01)] }));
  assert.deepEqual(SS.leaks(SS.trends([s2])), []);
  // the page's list: each leak said, the sessions with a Forget button, the empty note
  const html = SS.historyHtml([s1, s2], "Across your sessions");
  assert.match(html, /Flop vs c-bet: fold<\/b>: you 100%, GTO 30% on the same hands/);
  assert.match(html, /BB defence<\/b>: 0% of your actions in the chart's mix/);
  assert.match(html, /SRP · OOP · facing bet<\/b>: −0\.60bb a decision/);
  assert.equal((html.match(/data-forget="/g) || []).length, 2);
  assert.match(SS.historyHtml([s2], "t"), /No clear leak yet/);
  assert.equal(SS.historyHtml([], "t"), "");
});

test("history storage: its own key beside the session's", async () => {
  const mem = new Map();
  global.localStorage = { getItem: (k) => mem.has(k) ? mem.get(k) : null, setItem: (k, v) => mem.set(k, v), removeItem: (k) => mem.delete(k) };
  try {
    const cur = SS.lsStore("gto-session"), hist = SS.lsStore("gto-session-history");
    await cur.set({ v: 1 }); await hist.set([{ id: 1 }]);
    assert.deepEqual(await hist.get(), [{ id: 1 }]);
    await cur.del();
    assert.deepEqual(await hist.get(), [{ id: 1 }]);
  } finally { delete global.localStorage; }
});
