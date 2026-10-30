// Multi-table practice (gto/static/multitable.js, loaded by practice.js when Tables goes above 1) with a stubbed solver:
// the Tables picker, each table's own hand, the scheduler (the table that needs you in front, the queue's order,
// instant answers first, the other tables' solves before look-aheads, a long solve giving way, solves stopped when
// their hand ends), the session's score across tables, persistence and the view's HTML.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage } = require("./page.js");

const STATIC = path.join(__dirname, "..", "..", "gto", "static");
const PRACTICE = fs.readFileSync(path.join(STATIC, "practice.js"), "utf8");
const MT = fs.readFileSync(path.join(STATIC, "multitable.js"), "utf8");
const flush = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const ZERO = { hands: 0, decisions: 0, best: 0, good: 0, evDecisions: 0, evLost: 0, net: 0 };
const DECK = []; for (const r of "AKQJT98765432") for (const s of "shdc") DECK.push(r + s);
const SEATS6 = ["UTG", "HJ", "CO", "BTN", "SB", "BB"];
const acts = (s) => s ? s.split(", ").map(x => { const [pos, a] = x.split(" "); return a === "f" ? { pos, action: "fold" } : a === "c" ? { pos, action: "call" }
  : a === "x" ? { pos, action: "check" } : a[0] === "r" ? { pos, action: "raise", to: +a.slice(1) } : { pos, action: "bet", to: +a.slice(1) }; }) : [];
const SRP = "UTG f, HJ f, CO f, BTN r2.5, SB f, BB c";

// a table as multitable.js keeps it in this browser: your seat and cards, the board (shown: how much of it is dealt)
function table({ hero = "BTN", hole, board = [], shown = 0, preflop = "", flop = "", turn = "", river = "", drill = "any", n = 1 }) {
  board = [...board];
  const used = new Set([...hole, ...board]), rest = DECK.filter(c => !used.has(c));
  while (board.length < 5) board.push(rest.pop());
  const hands = {}; for (const p of SEATS6) if (p !== hero) hands[p] = [rest.pop(), rest.pop()];
  const cards = [...hole, ...board.slice(0, shown)]; while (cards.length < 7) cards.push(null);
  return { st: { players: 6, hero, stack: 100, cards, actions: { preflop: acts(preflop), flop: acts(flop), turn: acts(turn), river: acts(river) } },
    hand: { hero, players: 6, hole, hands, board, result: null, log: [], n, drill }, stats: { ...ZERO } };
}
// a page practising on several tables (multitable.js loaded first, so its timings are set before it starts): no pause
// between hands, no guard after the table in front changes, the clock __clock
async function mtPage({ seed = 7, tables, n = 2, storage = {}, opts = {}, guard = 0 } = {}) {
  const st = { "gto-practice-seat": '"random"', "gto-practice-tables": String(n), "gto-mt-drill": '"any"', ...storage };
  if (tables) st["gto-mt"] = JSON.stringify({ n, focus: 0, tables });
  const p = loadPage({ ...opts, storage: st });
  await p.ready();
  p.run(`Math.random = (() => { let s = ${seed}; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();`);
  p.run("globalThis.__clock = 0");
  vm.runInContext(PRACTICE, p.ctx, { filename: "practice.js" });
  vm.runInContext(MT, p.ctx, { filename: "multitable.js" });
  p.run(`MultiTable.OPT.pause = 0; MultiTable.OPT.guard = ${guard}; MultiTable.OPT.now = () => globalThis.__clock;`);
  p.run("Practice.set(true)");
  await flush();
  return p;
}
const clock = (p, t) => p.run(`globalThis.__clock = ${t}`);
// the requests not answered yet; answer one (or each) with fn(message) -> result
const open = (p, done) => p.posted.filter(m => m.id != null && m.state && !done.has(m.id));
function reply(p, done, m, result) { done.add(m.id); p.workers.at(-1).onmessage({ data: { id: m.id, result } }); }
async function answerAll(p, done, fn, rounds = 1) {
  for (let r = 0; r < rounds; r++) { await flush(); for (const m of open(p, done)) reply(p, done, m, fn(m)); await flush(); }
}
const T = (p) => p.val("MultiTable.state.tables.map(t => ({ id: t.id, st: t.st, hand: t.hand, stats: t.stats, ready: MultiTable.ready(t), since: t.since, err: t.err }))");
const focus = (p) => p.val("MultiTable.state.focus");
const inFlight = (p) => p.val("MultiTable.state.flight && { key: MultiTable.state.flight.job.key, id: MultiTable.state.flight.id, ahead: !!MultiTable.state.flight.job.ahead, t: MultiTable.state.flight.job.t.id }");
// whose request a message is: the table whose hero holds its hole cards
const tableOf = (p, m) => T(p).findIndex(t => t.hand && t.hand.hole.join("") === m.state.hole);
const passive = (m) => ({ pos: m.play.seat, action: "call" });
const ADVICE = { street: "flop", recommended: "call",
  actions: [{ action: "fold", freq: 0.2, ev_bb: 0 }, { action: "check", freq: 0.3, ev_bb: 0.4 }, { action: "call", freq: 0.5, ev_bb: 1 }] };
const FOLD_ONLY = { street: "preflop", recommended: "fold", actions: [{ action: "fold", freq: 1 }] };
const stub = (advice = ADVICE, villain = passive) => (m) => m.play ? villain(m) : advice;

test("Tables: 1 is Practice as before (multitable.js never loaded); 2-4 load it and play that many tables, each its own hand; back to 1 resumes the page's hand", async () => {
  const p = loadPage({ storage: { "gto-practice-seat": '"random"', "gto-mt-drill": '"any"' } });
  await p.ready();
  vm.runInContext(PRACTICE, p.ctx, { filename: "practice.js" });
  p.run("Practice.set(true)"); await flush();
  assert.equal(p.val("Practice.state.tables"), 1);
  assert.ok(!p.scripts.includes("multitable.js"));
  assert.equal(p.byId("practiceBar").hidden, false);
  assert.match(p.byId("pxTables").innerHTML, /data-v="1".*data-v="4"/);
  const single = p.S;
  p.run("Practice.setTables(3)"); await flush(8);
  assert.ok(p.scripts.includes("multitable.js"));
  assert.equal(p.val("MultiTable.on"), true);
  assert.equal(p.storage.get("gto-practice-tables"), "3");
  assert.ok(p.ctx.document.body.className.split(" ").includes("mt"));
  const ts = T(p);
  assert.equal(ts.length, 3);
  for (const t of ts) {   // each table its own deck: no card twice at a table
    const all = [...t.hand.hole, ...Object.values(t.hand.hands).flat(), ...t.hand.board];
    assert.equal(new Set(all).size, 17);
    assert.deepEqual(t.st.cards.slice(0, 2), t.hand.hole);
  }
  assert.notDeepEqual(ts[0].hand.hole, ts[1].hand.hole);
  // the page's own hand waits as it was, and its keys go to the tables (N doesn't deal the page a new hand)
  assert.deepEqual(p.S.cards, single.cards);
  p.key("n"); await flush();
  assert.deepEqual(p.S.cards, single.cards);
  assert.match(p.byId("mtTop").innerHTML, /data-tables="3"[^>]*>3</);
  // back to one table: the view goes, Practice's coach and the page's hand come back
  p.run("Practice.setTables(1)"); await flush();
  assert.equal(p.val("MultiTable.on"), false);
  assert.ok(!p.ctx.document.body.className.split(" ").includes("mt"));
  assert.equal(p.byId("mtView").hidden, true);
  assert.deepEqual(p.S.cards, single.cards);
  assert.equal(p.byId("practiceBar").hidden, false);
});

test("the table that needs you comes to the front; tables queue in the order they started waiting; S skips, a click picks one", async () => {
  const tables = [
    table({ hero: "BTN", hole: ["Ah", "Kh"], board: ["Kd", "7s", "2c"], shown: 3, preflop: SRP }),                 // BB (an opponent) acts first on the flop
    table({ hero: "BTN", hole: ["Qs", "Qc"], board: ["Jd", "8s", "3c"], shown: 3, preflop: SRP, flop: "BB x" }),   // your flop spot (a ~2 s solve)
    table({ hero: "UTG", hole: ["9h", "9d"] }),                                                                     // your preflop spot (instant)
  ];
  const p = await mtPage({ tables, n: 3 }), done = new Set();
  const answerNext = async (t, result) => {
    await flush();
    const ms = open(p, done);
    assert.equal(ms.length, 1, "one request at a time");
    clock(p, t); reply(p, done, ms[0], result(ms[0])); await flush();
    return ms[0];
  };
  // instant first (table 3's preflop), then your spot at table 2 before table 1's opponent (yours before an opponent's)
  let m = await answerNext(1, stub());
  assert.equal(tableOf(p, m), 2); assert.equal(m.state.board, "");
  assert.equal(focus(p), 2, "the only table waiting comes to the front");
  m = await answerNext(2, stub());
  assert.equal(tableOf(p, m), 1); assert.ok(!m.play);
  m = await answerNext(3, (m) => ({ pos: m.play.seat, action: "check" }));
  assert.equal(tableOf(p, m), 0); assert.equal(m.play.seat, "BB");
  m = await answerNext(4, stub());   // (your spot at table 1 after BB's check: the street is solved, so it's instant)
  assert.equal(tableOf(p, m), 0); assert.ok(!m.play);
  // the queue: table 3 (waiting since 1), 2 (since 2), 1 (since 4); the front stays put while it waits for you
  assert.deepEqual(p.val("MultiTable.queue().map(t => t.id)"), [2, 1, 0]);
  assert.equal(focus(p), 2);
  assert.match(p.byId("mtT2").className, /\bcur\b/);
  assert.match(p.byId("mtT2").className, /\bwait\b/);
  assert.match(p.byId("mtT0").innerHTML, /Your turn <span class="mt-q"[^>]*>3</);
  assert.match(p.byId("mtTabs").innerHTML, /data-t="0"[^>]*>1<span class="dot"><\/span><span class="mt-q">3</);
  // you act at the front: the next in the queue (table 2, waiting longer than table 1) comes forward
  clock(p, 10);
  assert.equal(p.run("MultiTable.act({ action: 'call' })"), null);
  assert.equal(focus(p), 1);
  // S: the next table waiting, this one to the back of the queue
  p.key("s");
  assert.equal(focus(p), 0);
  assert.deepEqual(p.val("MultiTable.queue().map(t => t.id)"), [0, 1]);
  // a click on a table brings it to the front, also one that isn't waiting for you (it stays until you act or pick another)
  p.run("MultiTable.click({ t: '2' })");
  assert.equal(focus(p), 2);
  p.run("MultiTable.pump()");
  assert.equal(focus(p), 2);
  assert.match(p.byId("mtDock").innerHTML, /Table 1 is waiting for you/);
  assert.equal(p.run("MultiTable.act({ action: 'call' })"), "It's not your turn at this table.");
});

test("after the table in front changes by itself, keys and buttons wait a moment (a double press doesn't act at the next table)", async () => {
  const tables = [table({ hero: "UTG", hole: ["Ah", "Kh"] }), table({ hero: "UTG", hole: ["Qs", "Qc"] })];
  const p = await mtPage({ tables, guard: 350 }), done = new Set();
  await answerAll(p, done, stub(FOLD_ONLY), 3);
  assert.equal(focus(p), 0);
  assert.equal(p.val("MultiTable.queue().length"), 2);
  clock(p, 1000);
  p.key("f");   // folds at table 1: table 2 comes to the front
  assert.equal(focus(p), 1);
  assert.equal(p.val("MultiTable.state.focusAt"), 1000);
  p.key("f");   // (at once: ignored)
  assert.equal(p.val("MultiTable.ready(MultiTable.state.tables[1])"), true);
  clock(p, 1400);
  p.key("f");
  assert.equal(p.val("MultiTable.ready(MultiTable.state.tables[1])"), false);
});

test("instant answers go first, and a long solve in flight gives way to one (stopped, its solve kept, and asked again)", async () => {
  const tables = [
    table({ hero: "BTN", hole: ["Ah", "Kh"], board: ["Kd", "7s", "2c", "9h"], shown: 4, preflop: SRP, flop: "BB x, BTN x", turn: "BB x" }),   // your turn spot: long at Accurate
    table({ hero: "UTG", hole: ["Qs", "Qc"] }),                                                                                              // your preflop spot: instant
  ];
  const p = await mtPage({ tables, storage: { "gto-speed": "accurate", "gto-mt-seat": '"UTG"' }, opts: { isolated: true } }), done = new Set();
  const flag = new Int32Array(p.posted.find(m => m.stop).stop);
  await flush();
  let ms = open(p, done);
  assert.equal(ms.length, 1);
  assert.equal(tableOf(p, ms[0]), 1, "the preflop chart's answer goes before the turn's solve");
  assert.equal(p.val("MultiTable.estimate(MultiTable.state.tables[0], JSON.parse(MultiTable.state.tables[0].turnKey))"), 60);
  reply(p, done, ms[0], FOLD_ONLY); await flush();
  ms = open(p, done);
  assert.equal(ms.length, 1);
  const long = ms[0];
  assert.equal(tableOf(p, long), 0);
  assert.equal(long.state.board, "Kd7s2c9h");
  assert.equal(p.val("MultiTable.group(MultiTable.state.flight.job)"), 4);
  // you fold at table 2: its next hand's preflop is instant, so the turn's solve is stopped for it
  assert.equal(p.run("MultiTable.act({ action: 'fold' })"), null);
  await flush();
  assert.equal(flag[1], long.id, "the long solve was stopped");
  assert.equal(p.workers.length, 1, "the worker (and the solve it keeps) stays");
  assert.equal(p.val("MultiTable.state.met.stops"), 1);
  ms = open(p, done).filter(m => m.id !== long.id);
  assert.equal(ms.length, 1);
  assert.equal(tableOf(p, ms[0]), 1);
  assert.equal(ms[0].state.board, "");
  // once table 2 waits for you again, the turn is asked again (the worker goes on from where it stopped)
  done.add(long.id);
  reply(p, done, ms[0], FOLD_ONLY); await flush();
  ms = open(p, done);
  assert.equal(ms.length, 1);
  assert.equal(tableOf(p, ms[0]), 0);
  assert.deepEqual(ms[0].state, long.state);
  // without a stop flag (the worker can't be stopped) nothing is stopped and the look-aheads wait at Accurate
  const q = await mtPage({ tables, storage: { "gto-speed": "accurate", "gto-mt-seat": '"UTG"' } }), qd = new Set();
  await flush();
  let [m] = open(q, qd); reply(q, qd, m, FOLD_ONLY); await flush();
  [m] = open(q, qd);
  assert.equal(tableOf(q, m), 0);
  assert.equal(q.run("MultiTable.act({ action: 'fold' })"), null); await flush();
  assert.equal(q.val("MultiTable.state.met.stops"), 0);
  assert.deepEqual(open(q, qd).map(x => x.id), [m.id], "the turn's solve runs on; table 2's preflop waits for it");
  assert.deepEqual(q.val("MultiTable.aheads()"), []);
});

test("while you act, the other tables' requests go first, then the look-aheads at the table in front, then the others' look-aheads", async () => {
  const MIX = { street: "flop", recommended: "check", actions: [{ action: "check", freq: 0.5, ev_bb: 1 }, { action: "bet 50%", amount_bb: 3, freq: 0.5, ev_bb: 1 }] };
  const tables = [
    table({ hero: "BB", hole: ["Ah", "Kh"], board: ["Kd", "7s", "2c"], shown: 3, preflop: SRP }),                               // your flop spot (BB first)
    table({ hero: "BB", hole: ["Qs", "Qc"], board: ["Jd", "8s", "3c", "4h"], shown: 4, preflop: SRP, flop: "BB x, BTN x" }),   // your turn spot
  ];
  const p = await mtPage({ tables }), done = new Set();
  const reply2 = (m) => m.state.flop.length && m.state.flop.at(-1).action === "bet" ? { pos: m.play.seat, action: "call" } : { pos: m.play.seat, action: "check" };
  await flush();
  let ms = open(p, done);
  assert.equal(tableOf(p, ms[0]), 0);
  reply(p, done, ms[0], MIX); await flush();
  assert.equal(focus(p), 0);
  ms = open(p, done);
  assert.equal(ms.length, 1);
  assert.equal(tableOf(p, ms[0]), 1, "table 2's own spot before table 1's look-ahead");
  assert.deepEqual(ms[0].state.flop, [{ pos: "BB", action: "check" }, { pos: "BTN", action: "check" }]);
  reply(p, done, ms[0], MIX); await flush();
  // both wait for you: the look-aheads at the table in front (table 1) first: BTN's reply to each of your likeliest
  // moves (quick: the flop is solved), then your turn spot after each
  const seen = [];
  for (let i = 0; i < 8; i++) {
    ms = open(p, done);
    assert.equal(ms.length, 1);
    if (tableOf(p, ms[0]) !== 0) break;
    assert.equal(p.val("MultiTable.state.flight.job.ahead"), true);
    seen.push(ms[0]);
    reply(p, done, ms[0], ms[0].play ? reply2(ms[0]) : MIX); await flush();
  }
  assert.deepEqual(seen.slice(0, 2).map(m => m.play && m.play.seat), ["BTN", "BTN"]);
  assert.deepEqual(seen.slice(0, 2).map(m => m.state.flop[0].action).sort(), ["bet", "check"]);
  assert.deepEqual(seen.slice(2).map(m => [!!m.play, m.state.board]), [[false, "Kd7s2c" + T(p)[0].hand.board[3]], [false, "Kd7s2c" + T(p)[0].hand.board[3]]]);
  // then table 2's (BTN's reply on the turn)
  assert.equal(tableOf(p, ms[0]), 1);
  assert.equal(ms[0].play.seat, "BTN");
  assert.equal(ms[0].state.turn.length, 1);
  // either line you take at table 1, BTN's reply and your next spot are in hand: no request, your turn at once
  const before = p.posted.length;
  clock(p, 5);
  assert.equal(p.run("MultiTable.act({ action: 'bet', to: 3 })"), null);
  await flush();
  assert.deepEqual(T(p)[0].st.actions.flop.map(a => a.pos + " " + a.action), ["BB bet", "BTN call"]);
  assert.equal(T(p)[0].st.cards.filter(Boolean).length, 6, "the turn dealt");
  assert.equal(T(p)[0].ready, true);
  assert.equal(p.posted.length, before);
  // (table 2 has waited longer: it comes forward, table 1 behind it)
  assert.equal(focus(p), 1);
  assert.deepEqual(p.val("MultiTable.queue().map(t => t.id)"), [1, 0]);
});

test("a look-ahead is stopped when its table's hand ends (you fold instead), and a table's request when the table goes", async () => {
  const FACING = { street: "flop", recommended: "call", actions: [{ action: "fold", freq: 0.3, ev_bb: 0 }, { action: "call", freq: 0.4, ev_bb: 0.2 }, { action: "raise 9", amount_bb: 9, freq: 0.3, ev_bb: 0.2 }] };
  const facing = (hole) => table({ hero: "BTN", hole, board: ["Kd", "7s", "2c"], shown: 3, preflop: SRP, flop: "BB b3" });
  const p = await mtPage({ tables: [facing(["Ah", "Kh"]), facing(["Qs", "Qc"])], opts: { isolated: true } }), done = new Set();
  const flag = new Int32Array(p.posted.find(m => m.stop).stop);
  for (let i = 0; i < 2; i++) { await flush(); const [m] = open(p, done); reply(p, done, m, FACING); }
  await flush();
  const [ahead] = open(p, done);
  assert.equal(tableOf(p, ahead), 0);
  assert.equal(p.val("MultiTable.state.flight.job.ahead"), true);
  assert.equal(p.run("MultiTable.act({ action: 'fold' })"), null);
  await flush();
  assert.equal(flag[1], ahead.id, "its look-ahead is stopped");
  assert.equal(p.val("MultiTable.state.score.hands"), 1);
  assert.equal(T(p)[0].stats.hands, 1);
  assert.notEqual(T(p)[0].hand.hole.join(""), "AhKh", "table 1 dealt its next hand");
  // fewer tables: the request of a table that goes is stopped
  const pre = (hole) => table({ hero: "UTG", hole });
  const q = await mtPage({ tables: [pre(["Ah", "Kh"]), pre(["Qs", "Qc"]), facing(["2s", "2c"])], n: 3, opts: { isolated: true } }), qd = new Set();
  const qf = new Int32Array(q.posted.find(m => m.stop).stop);
  for (let i = 0; i < 2; i++) { await flush(); const [m] = open(q, qd); reply(q, qd, m, FOLD_ONLY); }
  await flush();
  const [m] = open(q, qd);
  assert.equal(tableOf(q, m), 2);
  q.run("Practice.setTables(2)"); await flush();
  assert.equal(q.val("MultiTable.state.tables.length"), 2);
  assert.equal(qf[1], m.id);
  assert.equal(q.storage.get("gto-practice-tables"), "2");
  assert.match(q.byId("mtView").innerHTML, /mt-grid n2/);
});

test("every decision graded into one session score, each table's results and Review mistakes across tables; kept over a reload", async () => {
  // the solver almost never calls: each of your calls is a mistake giving up 1 bb
  const BAD = { street: "flop", recommended: "fold", actions: [{ action: "fold", freq: 0.6, ev_bb: 0 }, { action: "check", freq: 0.39, ev_bb: 0 }, { action: "call", freq: 0.01, ev_bb: -1 }, { action: "bet 50%", amount_bb: 3, freq: 0.0, ev_bb: -1 }] };
  const p = await mtPage({ seed: 11 }), done = new Set();
  for (let i = 0; i < 400 && p.val("MultiTable.state.score.hands") < 6; i++) {
    await answerAll(p, done, stub(BAD));
    if (p.val("MultiTable.ready(MultiTable.state.tables[MultiTable.state.focus])")) assert.equal(p.run("MultiTable.act({ action: 'call' })"), null);
  }
  const M = p.val("({ score: MultiTable.state.score, review: MultiTable.state.review, tables: MultiTable.state.tables.map(t => t.stats) })");
  assert.ok(M.score.hands >= 6);
  for (const k of ["hands", "decisions", "best", "good", "evDecisions"]) assert.equal(M.score[k], M.tables.reduce((s, t) => s + t[k], 0), k);
  assert.ok(Math.abs(M.score.net - M.tables.reduce((s, t) => s + t.net, 0)) < 1e-6);
  assert.ok(M.tables.every(t => t.hands >= 1), "both tables played");
  assert.deepEqual([...new Set(M.review.map(e => e.table))].sort(), [1, 2]);
  assert.ok(M.review.every(e => e.g && e.ctx && e.hole.length === 2 && e.n >= 1));
  assert.ok(M.review.some(e => e.g.l === 1), "a call that gives up 1 bb");
  // the side: the score, each table's row, the last decision's verdict, Review mistakes from both tables
  const side = p.byId("mtSide").innerHTML;
  assert.match(side, new RegExp(`<b class="num">${M.score.hands}</b> hands`));
  assert.match(side, /<table class="mt-tbls">.*<td>1<\/td>.*<td>2<\/td>/s);
  assert.match(side, /Table \d · last decision/);
  assert.match(side, /data-rev="1"[^>]*>Review mistakes<span class="n">\d+<\/span>/);
  p.run("MultiTable.click({ rev: '1' })");
  const rev = p.byId("mtSide").innerHTML;
  assert.match(rev, /Your biggest mistakes · all tables/);
  assert.match(rev, /T1 #\d+ /); assert.match(rev, /T2 #\d+ /);
  p.run("MultiTable.click({ pick: '0' })");
  // a reload: the tables' hands and the session come back
  const hands = T(p).map(t => t.hand.hole);
  p.run("Practice.setTables(1)");
  const q = loadPage({ storage: Object.fromEntries(p.storage) });
  await q.ready();
  vm.runInContext(PRACTICE, q.ctx, { filename: "practice.js" });
  vm.runInContext(MT, q.ctx, { filename: "multitable.js" });
  q.run("Practice.set(true)"); q.run("Practice.setTables(2)"); await flush();
  assert.deepEqual(q.val("MultiTable.state.score"), M.score);
  assert.equal(q.val("MultiTable.state.review.length"), M.review.length);
  assert.deepEqual(q.val("MultiTable.state.tables.map(t => t.hand.hole)"), hands);
  assert.deepEqual(q.val("MultiTable.state.tables.map(t => t.stats)"), M.tables);
  // Start over: the session cleared, every table dealt again
  q.run("globalThis.confirm = () => true; MultiTable.startOver()");
  assert.deepEqual(q.val("MultiTable.state.score"), ZERO);
  assert.equal(q.val("MultiTable.state.review.length"), 0);
  assert.deepEqual(JSON.parse(q.storage.get("gto-mt-score")).score, ZERO);
});

test("fewer tables set the others aside, their hand and results kept for when there are as many again (also over a reload)", async () => {
  const tables = [["Ah", "Kh"], ["Qs", "Qc"], ["Jd", "Js"], ["9h", "9d"]].map(hole => table({ hero: "UTG", hole }));
  tables[3].stats = { ...ZERO, hands: 5, decisions: 7, good: 6, best: 5, net: 12.5 };
  const p = await mtPage({ tables, n: 4 });
  p.run("Practice.setTables(2)"); await flush();
  assert.deepEqual(p.val("MultiTable.state.tables.map(t => t.hand.hole.join(''))"), ["AhKh", "QsQc"]);
  assert.equal(JSON.parse(p.storage.get("gto-mt")).tables[3].hand.hole.join(""), "9h9d");
  assert.doesNotMatch(p.byId("mtSide").innerHTML, /<td>4<\/td>/);
  // a reload at 2 tables, then 4 again: tables 3 and 4 as they were
  const q = loadPage({ storage: Object.fromEntries(p.storage) });
  await q.ready();
  vm.runInContext(PRACTICE, q.ctx, { filename: "practice.js" });
  vm.runInContext(MT, q.ctx, { filename: "multitable.js" });
  q.run("Practice.set(true)"); await flush();
  assert.equal(q.val("MultiTable.state.tables.length"), 2);
  q.run("Practice.setTables(4)"); await flush();
  assert.deepEqual(q.val("MultiTable.state.tables.map(t => t.hand.hole.join(''))"), ["AhKh", "QsQc", "JdJs", "9h9d"]);
  assert.equal(q.val("MultiTable.state.tables[3].stats.hands"), 5);
  assert.match(q.byId("mtSide").innerHTML, /<td>4<\/td><td>5<\/td><td>86%<\/td>/);
});

test("the view: compact tables in a grid, the front one outlined with the action dock and its keys; on a phone one table and a strip of tabs", async () => {
  const FACING = { street: "flop", recommended: "call", actions: [{ action: "fold", freq: 0.3, ev_bb: 0 }, { action: "call", freq: 0.4, ev_bb: 0.2 }, { action: "raise 9", amount_bb: 9, freq: 0.3, ev_bb: 0.2 }] };
  const tables = [
    table({ hero: "BTN", hole: ["Ah", "Kh"], board: ["Kd", "7s", "2c"], shown: 3, preflop: SRP, flop: "BB b3" }),
    table({ hero: "UTG", hole: ["Qs", "Qc"] }),
    table({ hero: "BB", hole: ["9h", "9d"], board: ["Jd", "8s", "3c"], shown: 3, preflop: SRP }),
    table({ hero: "SB", hole: ["2s", "2c"] }),
  ];
  const p = await mtPage({ tables, n: 4 }), done = new Set();
  const view = p.byId("mtView").innerHTML;
  assert.match(view, /<div class="mt-grid n4" id="mtGrid">(<div class="mt-t" id="mtT\d" data-t="\d"><\/div>){4}<\/div>/);
  assert.match(view, /id="mtTabs" role="tablist"/);
  assert.match(view, /id="mtDock"/);
  // nobody waits for you yet: the dock says so, each table says what it's doing
  assert.match(p.byId("mtDock").innerHTML, /No table needs you yet: the solver is on 4 tables/);
  assert.match(p.byId("mtT3").innerHTML, /<span class="spinner"><\/span>UTG is thinking…/);
  assert.match(p.byId("mtT0").innerHTML, /Solving your spot…/);
  // table 1 answered (facing BB's bet): picked, it's in front with the dock
  for (let i = 0; i < 16 && !T(p)[0].ready; i++) { await flush(); const [m] = open(p, done); if (!m) break; reply(p, done, m, tableOf(p, m) === 0 ? FACING : stub()(m)); await flush(); }
  await flush();
  p.run("MultiTable.click({ t: '0' })");
  const t = p.byId("mtT0");
  assert.match(t.className, /^mt-t wait cur$/);
  for (const i of [1, 2, 3]) assert.doesNotMatch(p.byId("mtT" + i).className, /\bcur\b/);
  assert.match(t.innerHTML, /<span class="nm">Table 1<\/span><span class="sub">#1 · BTN<\/span>/);
  assert.equal((t.innerHTML.match(/class="mt-seat/g) || []).length, 6);
  assert.match(t.innerHTML, /class="mt-seat me ta"/);
  assert.equal((t.innerHTML.match(/class="bc /g) || []).length, 5 + 2, "five board slots and your two cards");
  assert.match(t.innerHTML, /pot 8.5 bb/);
  assert.match(t.innerHTML, /<span class="made">pair of kings<\/span>/);
  const dock = p.byId("mtDock").innerHTML;
  assert.match(dock, /Table 1 · your turn<\/span><span class="pot">BTN · flop · pot 8.5 bb · 3 to call · need 26%/);
  assert.match(dock, /data-act="0">Fold<small>|data-act="0">Fold<kbd>F<\/kbd>/);
  assert.match(dock, /data-act="0">Fold.*<kbd>F<\/kbd>.*data-act="1">Call<small>3 bb<\/small><kbd>C<\/kbd>.*Raise to 9.*<kbd>1<\/kbd>.*your size….*<kbd>R<\/kbd>.*All-in.*<kbd>A<\/kbd>/s);
  p.key("r");
  assert.match(p.byId("mtDock").innerHTML, /<input type="number" id="mtAmt"/);
  p.key("Escape");
  assert.doesNotMatch(p.byId("mtDock").innerHTML, /mtAmt/);
  // the phone's tabs: one per table, the front selected, the waiting ones marked
  const tabs = p.byId("mtTabs").innerHTML;
  assert.equal((tabs.match(/class="mt-tab /g) || []).length, 4);
  assert.match(tabs, /class="mt-tab wait" role="tab" aria-selected="true" data-t="0"/);
  assert.match(tabs, /class="mt-tab think" role="tab" aria-selected="false" data-t="2" title="Table 3: Solving your spot…">3<span class="spinner">/);
  // a phone shows the front table alone under the tabs, the dock stuck to the bottom; nothing wider than the screen
  assert.match(MT, /@media \(max-width: 560px\) \{[^@]*\.mt-tabs \{ display: flex;[^@]*\.mt-grid \{ grid-template-columns: minmax\(0, 1fr\) !important; \}[^@]*\.mt-t:not\(\.cur\) \{ display: none; \}[^@]*\.mt-dock \{ position: sticky; bottom: 0;/);
  assert.match(MT, /\.mt-grid \{ display: grid; grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/);
  // the top bar: Tables, Spots (Instant flops first), Seat
  const top = p.byId("mtTop").innerHTML;
  assert.match(top, /aria-checked="true" data-tables="4"/);
  assert.match(top, /data-drill="lib"[^>]*>Instant flops</);
  assert.match(top, /aria-checked="true" data-drill="any"/);
  assert.match(top, /aria-checked="true" data-seat="random"/);
  assert.match(top, /<b>Any hand<\/b>: preflop and the library's flops answer at once; other flops, turns and rivers take ~2 s each/);
  // the side: the score and a row per table
  assert.equal((p.byId("mtSide").innerHTML.match(/<tr class="(cur)?"><td>\d<\/td>/g) || []).length, 4);
});

test("Instant flops: preflop played for you, each table dealt until a heads-up flop the library holds; its flop answers count as instant", async () => {
  const idx = JSON.parse(fs.readFileSync(path.join(STATIC, "flop_library", "index.json"), "utf8"));
  // the page's canonical flop names are the library's: each stored flop names itself, and so does any suit swap of it
  const p0 = await mtPage();
  const names = Object.keys(idx.spots.find(s => s.id === "srp_btn_bb").flops);
  assert.equal(names.length, 1755);
  p0.ctx.__names = names;
  assert.deepEqual(p0.val(`__names.filter(nm => { const cs = [nm.slice(0, 2), nm.slice(2, 4), nm.slice(4, 6)], sw = { c: "h", d: "s", h: "c", s: "d" };
    return MultiTable.flopName(cs) !== nm || MultiTable.flopName(cs.map(c => c[0] + sw[c[1]]).reverse()) !== nm; })`), []);
  // opponents (and your preflop) by the BTN-vs-BB line: everyone folds to BTN, who opens to 2.5, BB calls; then checks
  const LINE = { UTG: "fold", HJ: "fold", CO: "fold", BTN: "raise", SB: "fold", BB: "call" };
  const villain = (m) => m.state.board ? { pos: m.play.seat, action: "check" } : LINE[m.play.seat] === "raise" ? { pos: m.play.seat, action: "raise", to: 2.5 } : { pos: m.play.seat, action: LINE[m.play.seat] };
  const p = loadPage({ storage: { "gto-practice-seat": '"random"', "gto-practice-tables": "2", "gto-mt-drill": '"lib"' } });
  await p.ready();
  p.run("Math.random = (() => { let s = 5; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();");
  vm.runInContext(PRACTICE, p.ctx, { filename: "practice.js" });
  vm.runInContext(MT, p.ctx, { filename: "multitable.js" });
  p.ctx.__idx = idx;
  p.run("MultiTable.OPT.pause = 0; MultiTable.setLib(__idx)");
  p.run("Practice.set(true)"); await flush();
  const done = new Set();
  for (let i = 0; i < 400 && !T(p).every(t => t.ready); i++) await answerAll(p, done, stub(ADVICE, villain));
  for (const t of T(p)) {
    assert.ok(t.ready, `table ${t.id + 1} waits for you`);
    assert.equal(t.hand.drill, "lib");
    assert.ok(["BTN", "BB"].includes(t.hand.hero));
    assert.deepEqual(t.st.actions.preflop.map(a => `${a.pos} ${a.action}${a.to ? a.to : ""}`), ["UTG fold", "HJ fold", "CO fold", "BTN raise2.5", "SB fold", "BB call"]);
    assert.equal(t.st.cards.filter(Boolean).length, 5);
    assert.ok(names.includes(p.val(`MultiTable.flopName(${JSON.stringify(t.st.cards.slice(2, 5))})`)));
  }
  // your flop spot there is an instant answer (the library), a turn a quick solve
  const est = (f) => p.val(`(() => { const t = MultiTable.state.tables[0], q = JSON.parse(t.turnKey); return MultiTable.estimate(t, ${f}); })()`);
  assert.ok(est("q") < 0.5);
  assert.equal(est("{ ...q, board: q.board + '2d', flop: [] }"), 2);
  assert.match(p.byId("mtTop").innerHTML, /<b>Instant flops<\/b>: preflop is played for you/);
  // outside its game (an ante) Instant flops is Any hand, and says so
  p.run("S.ante = 0.1; MultiTable.render()");
  assert.match(p.byId("mtTop").innerHTML, /Instant flops needs the standard game/);
  assert.match(p.byId("mtTop").innerHTML, /data-drill="lib" title="Needs the standard game[^"]*" disabled/);
});

test("waits are measured: per decision, the seconds from your turn to its buttons", async () => {
  const tables = [table({ hero: "UTG", hole: ["Ah", "Kh"] }), table({ hero: "UTG", hole: ["Qs", "Qc"] })];
  const p = await mtPage({ tables }), done = new Set();
  await flush();
  clock(p, 500);
  let [m] = open(p, done); reply(p, done, m, FOLD_ONLY); await flush();
  clock(p, 1200);
  [m] = open(p, done); reply(p, done, m, FOLD_ONLY); await flush();
  const mt = p.val("MultiTable.metrics()");
  assert.equal(mt.decisions, 2);
  assert.deepEqual(mt.waits.map(w => w.s), [0.5, 1.2]);
  assert.equal(mt.median, 1.2);
});
