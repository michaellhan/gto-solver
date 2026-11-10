// The state the page sends to the solver (payload()): its keys match what gto/advisor.py reads, the worker
// and the local server receive it as built, and the speed and opponents toggles set its budget fields.
// Also writes tests/data/page_states.json: hands played through the page with its own engine's view of
// each spot, which tests/test_page_states.py replays through the Python advisor.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { loadPage, state, acts } = require("./page.js");

const ROOT = path.join(__dirname, "..", "..");
const flush = () => new Promise(r => setImmediate(r));
const FLOP = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") };

// keys the Python side reads from the state: Advisor._advise's state.get("k") / state["k"], and the worker's speed
function advisorKeys() {
  const src = fs.readFileSync(path.join(ROOT, "gto", "advisor.py"), "utf8");
  const body = src.slice(src.indexOf("def _advise("), src.indexOf("def _preflop("));
  const keys = new Set([...body.matchAll(/state(?:\.get\(|\[)"(\w+)"/g)].map(m => m[1]));
  for (const s of ["preflop", "flop", "turn", "river"]) if (body.includes("for s in STREETS")) keys.add(s);
  const worker = fs.readFileSync(path.join(ROOT, "gto", "static", "worker.js"), "utf8");
  for (const m of worker.matchAll(/st\.get\('(\w+)'\)/g)) keys.add(m[1]);
  return keys;
}

test("the advisor reads every key the page sends", async () => {
  const keys = advisorKeys();
  for (const k of ["players", "hero", "hole", "board", "stack", "stacks", "bb", "ante", "straddle", "budget", "prior_budget", "opponents", "speed", "flop"])
    assert.ok(keys.has(k), `advisor key list is missing ${k}: ${[...keys]}`);
  const p = loadPage();
  await p.ready();
  for (const st of [state(FLOP), state({ ...FLOP, ante: 0.25, straddle: true, players: 9, hero: "UTG" }), state({ players: 2, hero: "SB", cards: ["2c", "2d"] })]) {
    p.S = st;
    for (const k of Object.keys(p.payload())) assert.ok(keys.has(k), `the page sends ${k}, which the advisor ignores`);
  }
});

test("payload: cards, board, actions in the advisor's format, optional fields only when set", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state({ ...FLOP, flop: acts("BB x, BTN b2, BB r7, BTN a") });
  const pl = p.payload();
  assert.deepEqual(pl, {
    players: 6, hero: "BTN", hole: "AhKh", board: "Kd7s2c", stack: 100,
    preflop: [{ pos: "UTG", action: "fold" }, { pos: "HJ", action: "fold" }, { pos: "CO", action: "fold" }, { pos: "BTN", action: "raise", to: 2.5 }, { pos: "SB", action: "fold" }, { pos: "BB", action: "call" }],
    flop: [{ pos: "BB", action: "check" }, { pos: "BTN", action: "bet", amount: 2 }, { pos: "BB", action: "raise", to: 7 }, { pos: "BTN", action: "all-in" }],
    turn: [], river: [], budget: 8, prior_budget: 3, speed: "fast", opponents: "pool",
  });
  p.S = state({ ...FLOP, ante: 0.1, straddle: true });
  assert.equal(p.payload().ante, 0.1);
  assert.equal(p.payload().straddle, true);
  p.S = state({ ...FLOP, players: 2, hero: "BB", preflop: acts("SB r2.5, BB c"), straddle: true, ante: 0 });
  assert.equal("straddle" in p.payload(), false);   // no straddle heads-up
  assert.equal("ante" in p.payload(), false);
  p.S = state({ cards: ["Ah", "Kh", "Kd", "7s", "2c", null, "9h"], preflop: FLOP.preflop });
  assert.equal(p.payload().board, "Kd7s2c");
  p.S = state({});
  assert.equal(p.payload().hole, "");   // was 0 (null + null) in the copied "State JSON"
});

test("reviewing a decision the worker gets the state as built, at the Fast budget", async () => {
  const p = loadPage();
  await p.ready();
  p.spot(state(FLOP));
  await flush();
  const m = p.posted.at(-1);
  assert.deepEqual(m.state, p.payload());
  assert.equal(typeof m.id, "number");
  assert.equal(p.workers.length, 1);
  assert.match(p.workers[0].url, /^worker\.js\?v=/);
  assert.deepEqual([m.state.budget, m.state.prior_budget, m.state.speed], [8, 3, "fast"]);
  // the worker's answer is shown; no move to record, the panel leads to the other decisions
  p.workers[0].onmessage({ data: { id: m.id, result: { street: "flop", hero: "BTN", hand: "AKs", recommended: "check", actions: [{ action: "check", freq: 0.7 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.3 }] } } });
  await flush();
  assert.match(p.byId("result").innerHTML, /<tr class="best">.*?<\/i>Check/s);
  assert.equal(p.byId("pAction").hidden, true);
  assert.ok(!p.byId("stepTop").hidden && !p.byId("stepNav").hidden);
});

test("your turn on a hand being entered, the solver still loading: the panel says so, not that it's waiting for you", () => {
  const q = loadPage();
  q.S = state(FLOP);
  q.run("resetAdvice(); updateAdviceState()");
  assert.match(q.byId("result").innerHTML, /Getting the solver ready/);
  assert.doesNotMatch(q.byId("result").innerHTML, /Your turn|Waiting for your turn/);
});

test("a solve of a spot you've left doesn't hold up the next one: it's stopped and the new spot asked at once", async () => {
  const p = loadPage();
  await p.ready();
  p.spot(state(FLOP));
  await flush();
  const first = p.posted.at(-1);
  p.workers[0].onmessage({ data: { status: "Solving…" } });   // the runtime is up and solving that spot
  p.click("newhand");
  p.spot(state({ cards: ["9s", "9d", "Qc", "8h", "3d"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") }));
  await flush(); await flush();
  const asked = p.posted.at(-1);
  assert.notEqual(asked.id, first.id);
  assert.deepEqual(asked.state, p.payload());
  assert.equal(p.workers[0].terminated, true);   // the old solve blocked the worker: it was restarted
  assert.match(p.byId("result").innerHTML, /Solving the flop/);
  // the old solve's answer, if it still came, is dropped
  p.workers[0].onmessage && p.workers[0].onmessage({ data: { id: first.id, result: { street: "flop", recommended: "check", actions: [{ action: "check", freq: 1 }] } } });
  await flush();
  assert.match(p.byId("result").innerHTML, /Solving the flop/);
  // with the same spot still in flight, more renders ask nothing more (the new worker is told the flop once, to
  // fetch its library entries ahead: prefetchLibrary)
  const asks = () => p.posted.filter((m) => !m.prefetch).length;
  const n = asks();
  p.run("renderAll()");
  await flush();
  assert.equal(asks(), n);
  p.run("renderAll()");
  await flush();
  assert.equal(asks(), n);
  // (New hand stopped the solve there, so only the new worker is told the flop)
  assert.equal(p.posted.filter((m) => m.prefetch && m.prefetch.board === "Qc8h3d").length, 1);
});

test("cross-origin isolated, a solve of a spot you've left is stopped through memory shared with the worker, which keeps running", async () => {
  const p = loadPage({ isolated: true });
  await p.ready();
  const told = p.posted.filter((m) => m.stop);
  assert.equal(told.length, 1);
  assert.equal(Object.prototype.toString.call(told[0].stop), "[object SharedArrayBuffer]");
  const flag = new Int32Array(told[0].stop);
  const asks = () => p.posted.filter((m) => m.state);
  p.spot(state(FLOP));
  await flush();
  const first = asks().at(-1);
  p.workers[0].onmessage({ data: { status: "Solving…" } });
  assert.deepEqual([...flag], [0, 0]);
  p.click("newhand");
  p.spot(state({ cards: ["9s", "9d", "Qc", "8h", "3d"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") }));
  await flush(); await flush();
  const asked = asks().at(-1);
  assert.deepEqual(asked.state, p.payload());
  assert.equal(flag[0], first.id);              // every request up to the one left is stopped...
  assert.ok(asked.id > flag[0]);                // ...not the new one
  assert.equal(p.workers.length, 1);
  assert.equal(p.workers[0].terminated, false); // (not restarted: it keeps the hand's solves)
  assert.match(p.byId("result").innerHTML, /Solving the flop/);
  // the stopped request's answer, when it comes, is dropped; the new one's is shown
  p.workers[0].onmessage({ data: { id: first.id, result: { cancelled: true } } });
  await flush();
  assert.match(p.byId("result").innerHTML, /Solving the flop/);
  p.workers[0].onmessage({ data: { id: asked.id, result: { street: "flop", hero: "BTN", hand: "99", recommended: "check", actions: [{ action: "check", freq: 1 }] } } });
  await flush();
  assert.match(p.byId("result").innerHTML, /<tr class="best">.*?<\/i>Check/s);
  // one request alone (the equity's): stopRequests(id)
  const a = p.run("window.__a = callAdvise({ x: 1 }); __a.reqId"), b = p.run("window.__b = callAdvise({ x: 2 }); __b.reqId");
  assert.equal(p.run(`stopRequests(${a})`), true);
  assert.deepEqual([flag[0], flag[1]], [first.id, a]);
  assert.equal((await p.run("__a")).cancelled, true);
  p.workers[0].onmessage({ data: { id: b, result: { ok: 1 } } });
  assert.equal((await p.run("__b")).ok, 1);
  // not isolated: no flag, and stopping a running solve restarts the worker (the test above)
  const q = loadPage();
  await q.ready();
  assert.equal(q.posted.filter((m) => m.stop).length, 0);
  assert.equal(q.run("stopRequests()"), false);
});

test("at your turn or reviewing, once the flop is known the worker is told it, once a flop, to fetch the library's entries for it ahead; anyone else's turn fetches nothing", async () => {
  const p = loadPage();
  await p.ready();
  const told = () => p.posted.filter((m) => m.prefetch).map((m) => m.prefetch);
  p.S = state({ ...FLOP, flop: [] });   // BB to act: nothing told
  await flush();
  assert.equal(told().length, 0);
  p.spot(state({ cards: FLOP.cards.slice(0, 2), preflop: acts("UTG f, HJ f, CO f") }));   // preflop: nothing to fetch
  await flush();
  assert.equal(told().length, 0);
  p.spot(state(FLOP));   // your flop decision: the flop told
  await flush();
  assert.equal(told().length, 1);
  p.spot(state({ ...FLOP, cards: [...FLOP.cards, "9h"], flop: acts("BB x, BTN x"), turn: acts("BB x") }));   // your next decision on that flop: not told again
  await flush();
  assert.equal(told().length, 1);
  p.spot(state({ ...FLOP, cards: ["Ah", "Kh", "Qc", "8h", "3d", "2s"], flop: acts("BB x, BTN x"), turn: acts("BB x") }));   // another flop (a turn too: the flop's own)
  await flush();
  assert.equal(told().length, 2);
  assert.deepEqual(told().map((x) => [x.board, x.flop.length, x.turn.length]), [["Kd7s2c", 0, 0], ["Qc8h3d", 0, 0]]);
});

test("the page always solves at Fast: a stored speed isn't read or changed, and none is written", async () => {
  const html = require("node:fs").readFileSync(require("./page.js").HTML_PATH, "utf8");
  assert.doesNotMatch(html, /solvectl|gto-speed/);
  for (const stored of ["quick", "fast", "accurate"]) {
    const q = loadPage({ storage: { "gto-speed": stored } });
    await q.ready();
    q.S = state(FLOP);
    assert.deepEqual([q.payload().budget, q.payload().prior_budget, q.payload().speed, q.payload().quick], [8, 3, "fast", undefined]);
    assert.equal(q.storage.get("gto-speed"), stored);   // (Lizard and the frozen copy read it)
  }
  const p = loadPage();
  await p.ready();
  assert.equal(p.storage.has("gto-speed"), false);
  const local = loadPage({ server: true });
  await local.ready();
  local.S = state(FLOP);
  assert.deepEqual([local.payload().budget, local.payload().prior_budget, local.payload().speed], [2.5, 2.5, "fast"]);
});

test("the backend probe finds a file on the static site (no 404 logged on every load), and the server answers it", async () => {
  const p = loadPage();
  await p.ready();
  assert.equal(p.run("MODE"), "wasm");
  assert.equal(p.fetched.length, 1);
  assert.deepEqual(p.notFound, []);
  // the file goes to the site with the page, and the local server answers the same path itself
  const probe = p.fetched[0];
  assert.match(fs.readFileSync(path.join(ROOT, "scripts", "deploy.sh"), "utf8"), new RegExp(`gto/static/${probe.replace(".", "\\.")}"`));
  const handler = fs.readFileSync(path.join(ROOT, "gto", "server.py"), "utf8").match(/def do_GET[\s\S]*?\n\n/)[0];
  const prefix = handler.match(/self\.path\.startswith\("([^"]+)"\)/)[1];
  assert.ok(("/" + probe).startsWith(prefix), `the server answers ${prefix}, the page asks ${probe}`);
  const q = loadPage({ server: true });
  await q.ready();
  assert.equal(q.run("MODE"), "server");
});

test("over https (the static site) the worker starts with the page, not after the backend probe; over http it waits for it", async () => {
  const p = loadPage({ protocol: "https:" });
  assert.equal(p.workers.length, 1);   // before the probe has answered
  await p.ready();
  assert.equal(p.run("MODE"), "wasm");
  assert.equal(p.workers.length, 1);   // the same worker: not started twice
  assert.equal(p.workers[0].terminated, false);
  const q = loadPage();                // http: the local server's page
  assert.equal(q.workers.length, 0);
  await q.ready();
  assert.equal(q.workers.length, 1);
  const s = loadPage({ protocol: "https:", server: true });   // a local server behind an https proxy: the worker goes
  await s.ready();
  assert.equal(s.run("MODE"), "server");
  assert.equal(s.workers[0].terminated, true);
  assert.equal(s.run("worker"), null);
});

test("with the local server the state is POSTed at the server's budgets", async () => {
  const p = loadPage({ server: true });
  await p.ready();
  assert.equal(p.run("MODE"), "server");
  p.spot(state(FLOP));
  await flush();
  const pl = p.payload();
  assert.deepEqual([pl.budget, pl.prior_budget], [2.5, 2.5]);
  assert.deepEqual(p.posted.at(-1).state, pl);
  assert.equal(p.workers.length, 0);
});

// ---- hands played through the page, for the Python cross-check
function rng(seed) {   // mulberry32
  return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function playHands(nHands, seed, unequal = false) {
  const R = rng(seed), pick = (a) => a[Math.floor(R() * a.length)];
  const deck0 = [];
  for (const r of "AKQJT98765432") for (const s of "shdc") deck0.push(r + s);
  const out = [];
  for (let h = 0; h < nHands; h++) {
    const players = 2 + Math.floor(R() * 8);
    const p = loadPage();
    const seats = p.val(`SEATS[${players}]`);
    const deck = [...deck0].sort(() => R() - 0.5);
    const st = state({ players, hero: pick(seats), stack: pick([15, 40, 100, 100, 250]), cards: deck.slice(0, 2),
      ante: R() < 0.25 ? pick([0.1, 0.125, 1]) : undefined, straddle: players >= 3 && R() < 0.25 ? true : undefined });
    // per-seat stacks: some seats short or deep (side pots, all-ins for less, incomplete raises)
    if (unequal) st.stacks = Object.fromEntries(seats.filter(() => R() < 0.6).map(q => [q, pick([3, 8, 15, 22.5, 40, 60, 150, 300])]));
    p.S = st;
    let next = 2;
    for (let step = 0; step < 80; step++) {
      const sim = p.sim();
      if (sim.handOver) break;
      if (sim.waiting) { p.run(`placeCard("${deck[next++]}")`); continue; }
      const info = sim.cur, who = sim.toAct;
      if (who === p.S.hero || R() < 0.2) out.push({ hand: h + (unequal ? 1000 : 0), state: (({ opponents, budget, prior_budget, speed, ...rest }) => rest)(p.payload()), page: { street: sim.current, to_act: who, pot: sim.pot, cur_bet: info.curBet, active: sim.active, allin: sim.allin } });
      const facing = info.curBet - info.bets[who] > 1e-9, x = R();
      let a;
      if (facing) a = x < 0.35 ? { action: "fold" } : x < 0.75 ? { action: "call" } : x < 0.95 ? { action: "bet", to: info.curBet * (2 + 2 * R()) } : { action: "all-in" };
      else a = x < 0.55 ? { action: "check" } : x < 0.95 ? { action: "bet", to: info.curBet + Math.max(1, info.pot * (0.25 + R())) } : { action: "all-in" };
      let err = p.act(a);
      if (err && /can't raise|smallest raise/.test(err)) err = p.act({ action: "call" });   // (a raise the rules don't allow)
      assert.equal(err, null, JSON.stringify({ a, sim }));
    }
  }
  return out;
}

test("hands played through the page: fixture for the Python advisor (tests/data/page_states.json)", () => {
  const spots = [...playHands(70, 20261012), ...playHands(40, 20261020, true)];
  assert.ok(spots.length > 100);
  for (const s of spots) {   // basic invariants of the page's engine
    assert.ok(s.page.active.includes(s.page.to_act));
    assert.ok(!s.page.allin.includes(s.page.to_act));
    assert.ok(s.page.pot > 0);
    assert.equal(s.state.board.length / 2, { preflop: 0, flop: 3, turn: 4, river: 5 }[s.page.street]);
  }
  const file = path.join(ROOT, "tests", "data", "page_states.json");
  const text = JSON.stringify(spots, null, 0).replace(/\},\{"hand"/g, '},\n{"hand"') + "\n";
  if (process.env.UPDATE_FIXTURES || !fs.existsSync(file)) fs.writeFileSync(file, text);
  assert.equal(fs.readFileSync(file, "utf8"), text, "the page's engine changed: rerun with UPDATE_FIXTURES=1 and check tests/test_page_states.py");
});

test("a jam the answer's tree has no all-in for is asked again with jam: true and graded against that answer", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  // a deep flop: the tree's bets are a third and three quarters of the pot, no all-in
  const deep = { street: "flop", hero: "BTN", hand: "AKs", recommended: "bet 75%", pot_bb: 5.5,
    actions: [{ action: "check", freq: 0.1, ev_bb: 2.4 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.1, ev_bb: 2.6 }, { action: "bet 75%", amount_bb: 4, freq: 0.8, ev_bb: 2.7 }] };
  const jam = { ...deep, actions: [...deep.actions, { action: "all-in", amount_bb: 97.5, freq: 0, ev_bb: -2.3 }] };
  const move = "gradedMove(simulate(), { action: 'all-in' })";
  assert.equal(p.val(`jamOffTree(${JSON.stringify(deep)}, ${move})`), true);
  assert.equal(p.val(`jamOffTree(${JSON.stringify(jam)}, ${move})`), false);   // (an answer with an all-in grades it at once)
  const key = JSON.stringify(p.payload());
  const graded = p.run(`jamGrade(${JSON.stringify(key)}, ${move})`);
  await flush();
  const again = p.posted.find(x => x.state && x.state.jam);
  assert.ok(again, "the spot asked again with jam: true");
  assert.deepEqual({ ...again.state, jam: undefined }, { ...JSON.parse(key), jam: undefined });   // (the spot before the jam)
  p.workers[0].onmessage({ data: { id: again.id, result: jam } });
  const g = await graded;
  assert.deepEqual([g.m, g.f, g.l], ["all-in", 0, 5]);
  assert.equal(p.run(`gradeClass(${JSON.stringify(g)})`), "bad");
});

test("a bet well off the answer's sizes is asked again with size: and graded as its own row, not the nearest size's", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  // the turn tree's sizes are 33% and 75% of a 20.5bb pot; you bet 19.5 (95%)
  const r = { street: "flop", hero: "BTN", hand: "AJo", recommended: "bet 33%", pot_bb: 20.5,
    actions: [{ action: "check", freq: 0.4, ev_bb: 7.9 }, { action: "bet 33%", amount_bb: 6.8, freq: 0.6, ev_bb: 8.3 },
      { action: "bet 75%", amount_bb: 15.4, freq: 0, ev_bb: 7.6, rough_ev: true }, { action: "all-in", amount_bb: 27, freq: 0, ev_bb: 1.2 }] };
  const sized = { ...r, actions: [r.actions[0], r.actions[1], { action: "bet 95%", amount_bb: 19.5, freq: 0, ev_bb: 7.1, size: true, rough_ev: true }, r.actions[3]] };
  assert.equal(p.val(`sizeOffTree(${JSON.stringify(r)}, { action: "bet", to: 19.5 })`), true);
  assert.equal(p.val(`sizeOffTree(${JSON.stringify(sized)}, { action: "bet", to: 19.5 })`), false);   // (asked with it already)
  // within 15% of a size the tree has: graded as it at once
  assert.equal(p.val(`sizeOffTree(${JSON.stringify(r)}, { action: "bet", to: 7.2 })`), false);
  assert.equal(p.val(`sizeOffTree(${JSON.stringify(r)}, { action: "bet", to: 15 })`), false);
  assert.equal(p.val(`sizeOffTree(${JSON.stringify(r)}, { action: "check" })`), false);
  assert.equal(p.val(`sizeOffTree({ ...${JSON.stringify(r)}, street: "preflop" }, { action: "bet", to: 19.5 })`), false);
  assert.equal(p.val(`sizeOffTree({ ...${JSON.stringify(r)}, multiway: true }, { action: "bet", to: 19.5 })`), false);
  // without it the 95% bet is graded as the 75% row's 7.6
  assert.equal(p.val(`gradeWith(${JSON.stringify(r)}, { action: "bet", to: 19.5 })`).m, "bet 75%");
  const key = JSON.stringify(p.payload());
  const graded = p.run(`sizeGrade(${JSON.stringify(key)}, { action: "bet", to: 19.5 })`);
  await flush();
  const again = p.posted.find(x => x.state && x.state.size != null);
  assert.ok(again, "the spot asked again with size");
  assert.equal(again.state.size, 19.5);
  assert.deepEqual({ ...again.state, size: undefined }, { ...JSON.parse(key), size: undefined });
  p.workers[0].onmessage({ data: { id: again.id, result: sized } });
  const g = await graded;
  assert.deepEqual([g.m, g.e, g.l, g.rv], ["bet 95%", 7.1, 1.2, 1]);   // (its own row: its EV, rough as the answer has it)
});

test("after the flop a raise the tree has only an all-in for is graded as its all-in; a 4+-way jam is asked again too", async () => {
  const p = await (async () => { const q = loadPage(); await q.ready(); return q; })();
  p.run("S.tournament = true");   // (at equilibrium: graded by its mix)
  // facing a 4bet: fold, call or all-in
  const r = { street: "flop", pot_bb: 60, actions: [{ action: "fold", freq: 0.3, ev_bb: 0 }, { action: "call", freq: 0.5, amount_bb: 23, ev_bb: 4 },
    { action: "all-in", freq: 0.2, amount_bb: 97.5, ev_bb: 3.9 }] };
  const g = p.val(`gradeWith(${JSON.stringify(r)}, { action: "raise", to: 80 })`);
  assert.deepEqual([g.m, g.f, g.v], ["all-in", 0.2, "raise to 80"]);
  assert.match(p.run(`gradeTip(${JSON.stringify(g)})`), /graded as the solver's all-in: it has no smaller raise here/);
  // the betting rules (4+ players) list no all-in: a jam there is asked again
  const rules = { street: "flop", multiway: true, pot_bb: 10, actions: [{ action: "check", freq: 1, ev_bb: 1 }, { action: "bet 33%", freq: 0, ev_bb: 0.8, amount_bb: 3.5 }] };
  assert.equal(p.val(`jamOffTree(${JSON.stringify(rules)}, { action: "all-in" })`), true);
  assert.equal(p.val(`jamOffTree(${JSON.stringify(rules)}, { action: "check" })`), false);
  assert.equal(p.val(`jamOffTree({ ...${JSON.stringify(rules)}, street: "preflop" }, { action: "all-in" })`), false);
});

test("an all-in nobody can match more of is graded as the call (or check) it is", async () => {
  const p = loadPage();
  await p.ready();
  // BTN with 20bb behind faces BB's 30bb bet: all-in is a call for less
  p.S = { ...state({ ...FLOP, flop: acts("BB b30") }), stacks: { BTN: 22.5 } };
  assert.deepEqual(p.val('gradedMove(simulate(), { action: "all-in" })'), { action: "call" });
  // deep, facing a third pot: a real jam
  p.S = state({ ...FLOP, flop: acts("BB b2") });
  assert.deepEqual(p.val('gradedMove(simulate(), { action: "all-in" })'), { action: "all-in" });
  // the only other player is all-in for less than the bet to call: nobody can match more, so it's the call
  p.S = { ...state({ ...FLOP, flop: acts("BB a") }), stacks: { BB: 12.5 } };
  assert.deepEqual(p.val('gradedMove(simulate(), { action: "all-in" })'), { action: "call" });
  assert.deepEqual(p.val('gradedMove(simulate(), { action: "bet", to: 5 })'), { action: "bet", to: 5 });
});

test("an approximate answer the worker has at once is shown under the progress bar, then replaced by the request's own", async () => {
  const p = loadPage();
  await p.ready();
  const asks = () => p.posted.filter((m) => m.state);
  p.spot(state(FLOP));
  await flush();
  const ask = asks().at(-1);
  assert.equal(ask.early, true);   // (the page's own advice asks for one)
  p.workers[0].onmessage({ data: { status: "Solving…" } });
  const result = p.byId("result");
  const approx = { street: "flop", hero: "BTN", recommended: "bet 75%", approximate: { source: "prev", label: "the preflop's view" },
                   actions: [{ action: "check", freq: 0.4, amount_bb: null }, { action: "bet 75%", freq: 0.6, amount_bb: 4.13 }] };
  p.workers[0].onmessage({ data: { id: ask.id, partial: approx } });
  await flush();
  const box = result.children.at(-1);
  assert.equal(box.id, "early");
  assert.match(box.innerHTML, /<div class="hd">Bet 75%<\/div><p>Approximate, at once: the preflop's view\. The flop's own solve, running above, replaces it\.<\/p>/);
  assert.match(box.innerHTML, /<td class="n">60%<\/td>/);
  assert.match(box.innerHTML, /<small>4\.13 bb<\/small>/);
  assert.match(result.innerHTML, /Solving the flop/);   // (the progress stays above it)
  assert.equal(p.run("lastResult"), null);                // not an answer to grade or record
  // read off a solve cut to fit Quick: the box says its game is smaller
  const cut = "flop: solved in a smaller game than the full one, to fit this speed's time and memory: one raise size";
  p.workers[0].onmessage({ data: { id: ask.id, partial: { ...approx, notes: ["approximate, at once: x", cut] } } });
  await flush();
  assert.equal(result.children.filter((c) => c.id === "early").length, 1);
  assert.ok(result.children.at(-1).innerHTML.includes(`<p class="meta">${cut}</p>`));
  assert.doesNotMatch(result.children.at(-1).innerHTML, /approximate, at once: x/);
  p.workers[0].onmessage({ data: { id: ask.id, result: { street: "flop", hero: "BTN", hand: "AKs", recommended: "check", actions: [{ action: "check", freq: 1 }] } } });
  await flush();
  assert.equal(result.children.length, 0);
  assert.match(result.innerHTML, /<tr class="best">.*?<\/i>Check/s);
  assert.doesNotMatch(result.innerHTML, /approximate/);
  // a partial for a request no longer pending is dropped
  p.workers[0].onmessage({ data: { id: ask.id, partial: approx } });
  await flush();
  assert.equal(result.children.length, 0);
});

test("an approximate turn answer from a format-2 turn file shows the stored range EVs, labelled as the range's", async () => {
  const p = loadPage();
  await p.ready();
  p.spot(state(FLOP));
  await flush();
  const ask = p.posted.filter((m) => m.state).at(-1);
  p.workers[0].onmessage({ data: { status: "Solving…" } });
  const result = p.byId("result");
  const approx = { street: "turn", hero: "BB", recommended: "check", pot_bb: 9.5, approximate: { source: "store", label: "the precomputed flop solve's view of the turn (by card class, one bet size, 75%)" },
                   actions: [{ action: "check", freq: 0.7, amount_bb: null }, { action: "bet 75%", freq: 0.3, amount_bb: 7.13 }] };
  // format 1: no EV column (the mix only)
  p.workers[0].onmessage({ data: { id: ask.id, partial: approx } });
  await flush();
  let box = result.children.at(-1);
  assert.match(box.innerHTML, /<th>Solver<\/th><\/tr>/);
  assert.doesNotMatch(box.innerHTML, /range EV|Range EV/);
  // format 2: each action's range EV and both ranges', never as the hand's own EV
  p.workers[0].onmessage({ data: { id: ask.id, partial: { ...approx, range_ev_bb: 4.54, villain_range_ev_bb: 4.96,
    actions: [{ ...approx.actions[0], range_ev_bb: 3.2 }, { ...approx.actions[1], range_ev_bb: -0.41 }] } } });
  await flush();
  box = result.children.at(-1);
  assert.equal(result.children.filter((c) => c.id === "early").length, 1);
  assert.match(box.innerHTML, /<th title="The EV of the hands in your range that take this action \(not your hand's own\)[^"]*">Range EV, bb<\/th>/);
  assert.match(box.innerHTML, /<td class="n">3\.20<\/td>/);
  assert.match(box.innerHTML, /<td class="n">−0\.41<\/td>/);
  assert.match(box.innerHTML, /range EV: yours \+4\.54 bb · theirs \+4\.96 bb of the 9\.5 bb pot/);
  assert.match(box.innerHTML, /not your hand's own; your hand's EVs come with the solve/);
  assert.doesNotMatch(box.innerHTML, />EV, bb</);
  // in position after a check: the actions' range EVs, no range line (one action nobody takes: no EV)
  p.workers[0].onmessage({ data: { id: ask.id, partial: { ...approx, actions: [{ ...approx.actions[0], range_ev_bb: 2.5 }, { ...approx.actions[1], range_ev_bb: null }] } } });
  await flush();
  box = result.children.at(-1);
  assert.match(box.innerHTML, /Range EV/);
  assert.doesNotMatch(box.innerHTML, /range EV: yours/);
  assert.match(box.innerHTML, /7\.13 bb<\/small><\/div><\/td><td class="n">30%<\/td><td class="n">–<\/td><\/tr>/);
  assert.equal(p.run("lastResult"), null);
});

test("an action your range almost never takes (rough_ev) shows its EV with a ≈ and the tip; graded the same, its EV lost marked", async () => {
  const p = loadPage();
  await p.ready();
  p.spot(state(FLOP));
  await flush();
  const ask = p.posted.filter((m) => m.state).at(-1);
  const res = { street: "flop", hero: "BTN", hand: "AKo", recommended: "bet 75%", pot_bb: 5.5,
    actions: [{ action: "check", freq: 0.6, ev_bb: 4.88, amount_bb: null }, { action: "bet 33%", freq: 0, ev_bb: 4.5, amount_bb: 2 },
      { action: "bet 75%", freq: 0.4, ev_bb: 4.9, amount_bb: 4 }] };
  const rough = { ...res, actions: res.actions.map(a => a.action === "bet 33%" ? { ...a, rough_ev: true } : a) };
  p.workers[0].onmessage({ data: { id: ask.id, result: res } });
  await flush();
  const result = p.byId("result");
  assert.doesNotMatch(result.innerHTML, /≈|rough/);
  const plain = result.innerHTML;
  p.spot(state({ ...FLOP, cards: ["Ah", "Kc", "Kd", "7s", "2c"] }));   // (another hand: a request of its own)
  await flush();
  p.workers[0].onmessage({ data: { id: p.posted.filter((m) => m.state).at(-1).id, result: rough } });
  await flush();
  const html = result.innerHTML;
  assert.match(html, /<td class="n" title="Your range almost never takes this, so how the opponent answers it is barely solved: its EV is rough, usually lower than exact\.">≈ 4\.50<\/td>/);
  assert.match(html, /<td class="n">4\.88<\/td>/);   // (the others as before)
  assert.match(html, /≈ is a rough EV\./);
  assert.equal((html.match(/≈/g) || []).length, (plain.match(/≈/g) || []).length + 2);
  // grading: the same verdict and EV lost; the EV lost and the move's EV carry a ≈ and the tip
  const g0 = p.val(`gradeWith(${JSON.stringify(res)}, { action: "bet", to: 2 })`);
  const g = p.val(`gradeWith(${JSON.stringify(rough)}, { action: "bet", to: 2 })`);
  assert.deepEqual({ ...g, rv: undefined }, { ...g0, rv: undefined });
  assert.equal(g.rv, 1);
  assert.equal(g0.rv, undefined);
  assert.equal(p.val(`gradeClass(${JSON.stringify(g)})`), p.val(`gradeClass(${JSON.stringify(g0)})`));
  assert.equal(p.val(`gradeWith(${JSON.stringify(rough)}, { action: "check" })`).rv, undefined);   // (your move's own EV is what counts)
  assert.match(p.run(`lossHtml(${JSON.stringify(g)})`), /<span class="l bad" title="Your range almost never[^"]*">≈ −0\.40 bb<\/span>/);
  assert.equal(p.run(`lossHtml(${JSON.stringify(g0)})`), `<span class="l bad">−0.40 bb</span>`);
  assert.match(p.run(`verdictText(${JSON.stringify(g)})`), / ≈ 0\.40 bb lost, /);
  assert.match(p.run(`verdictText(${JSON.stringify(g0)})`), /[^≈] 0\.40 bb lost, /);
  assert.match(p.run(`gradeTip(${JSON.stringify(g)})`), /EV ≈ \+4\.50 bb vs \+4\.90 bb \(−0\.40 bb, 7\.3% of the pot\)\. Your range almost never takes this/);
  assert.doesNotMatch(p.run(`gradeTip(${JSON.stringify(g0)})`), /≈|almost never/);
});
