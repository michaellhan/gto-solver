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
    turn: [], river: [], budget: 8, prior_budget: 3, speed: "fast",
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

test("on your turn the worker gets the state as built, at the Fast budget", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  await flush();
  const m = p.posted.at(-1);
  assert.deepEqual(m.state, p.payload());
  assert.equal(typeof m.id, "number");
  assert.equal(p.workers.length, 1);
  assert.match(p.workers[0].url, /^worker\.js\?v=/);
  assert.deepEqual([m.state.budget, m.state.prior_budget, m.state.speed], [8, 3, "fast"]);
  // the worker's answer is shown and the buttons switch to the solver's sizes
  p.workers[0].onmessage({ data: { id: m.id, result: { street: "flop", hero: "BTN", hand: "AKs", recommended: "check", actions: [{ action: "check", freq: 0.7 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.3 }] } } });
  await flush();
  assert.match(p.byId("result").innerHTML, /<span class="big">check<\/span>/);
  assert.ok(p.buttons().some(b => b.innerHTML.startsWith('Bet 2<small>33% pot<span class="fq"> · 30%</span>')));
});

test("on your turn before the solve starts (the page still finding its backend) the panel doesn't say it waits for your turn", async () => {
  const p = loadPage();
  p.S = state(FLOP);
  p.run("resetAdvice(); updateAdviceState()");   // as the page is right after a reload, before modeReady settles
  assert.doesNotMatch(p.byId("result").innerHTML, /Waiting for your turn/);
  assert.match(p.byId("result").innerHTML, /Your turn/);
  p.S = state({ ...FLOP, flop: [] });   // BB to act: it does wait
  p.run("resetAdvice(); updateAdviceState()");
  assert.match(p.byId("result").innerHTML, /Waiting for your turn/);
});

test("a solve of a spot you've left doesn't hold up your next turn: it's stopped and the new spot asked at once", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  await flush();
  const first = p.posted.at(-1);
  p.workers[0].onmessage({ data: { status: "Solving…" } });   // the runtime is up and solving that spot
  p.click("newhand");
  p.S = state({ cards: ["9s", "9d", "Qc", "8h", "3d"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") });
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
  // with the same spot still in flight, more renders ask nothing more
  const n = p.posted.length;
  p.run("renderAll()");
  await flush();
  assert.equal(p.posted.length, n);
});

test("Quick asks for quick answers, about 2 s by street, and is remembered", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  await flush();
  await p.byId("solvectl").children[0].onclick();
  await flush();
  const pl = p.payload();
  assert.deepEqual([pl.speed, pl.quick, pl.budget, pl.prior_budget], ["quick", true, ...p.val("QUICK_BUDGETS.flop")]);
  assert.equal(p.storage.get("gto-speed"), "quick");
  assert.deepEqual(p.posted.at(-1).state, pl);
  const q = loadPage({ storage: { "gto-speed": "quick" } });
  await q.ready();
  assert.equal(q.val("SPEED"), "quick");
  assert.match(q.byId("solvectl").children[0].className, /active/);
});

test("Accurate and Pool toggles change the state sent, and are remembered", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  await flush();
  const n = p.posted.length;
  await p.byId("solvectl").children[2].onclick();   // (Quick, Fast, Accurate)
  await p.byId("oppctl").children[1].onclick();
  await flush();
  const pl = p.payload();
  assert.deepEqual([pl.budget, pl.prior_budget, pl.speed, pl.opponents], [60, 12, "accurate", "pool"]);
  assert.equal(p.storage.get("gto-speed"), "accurate");
  assert.equal(p.storage.get("gto-opponents"), "pool");
  assert.ok(p.workers[0].terminated === false || p.workers.length > 1);
  assert.deepEqual(p.posted.at(-1).state, pl);
  assert.ok(p.posted.length > n);
  const q = loadPage({ storage: { "gto-speed": "accurate", "gto-opponents": "pool" } });
  await q.ready();
  q.S = state(FLOP);
  assert.deepEqual([q.payload().speed, q.payload().opponents], ["accurate", "pool"]);
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
  p.S = state(FLOP);
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
      if (who === p.S.hero || R() < 0.2) out.push({ hand: h + (unequal ? 1000 : 0), state: (({ budget, prior_budget, speed, ...rest }) => rest)(p.payload()), page: { street: sim.current, to_act: who, pot: sim.pot, cur_bet: info.curBet, active: sim.active, allin: sim.allin } });
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
  await flush();
  const m = p.posted.at(-1);
  // a deep flop: the tree's bets are a third and three quarters of the pot, no all-in
  const deep = { street: "flop", hero: "BTN", hand: "AKs", recommended: "bet 75%", pot_bb: 5.5,
    actions: [{ action: "check", freq: 0.1, ev_bb: 2.4 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.1, ev_bb: 2.6 }, { action: "bet 75%", amount_bb: 4, freq: 0.8, ev_bb: 2.7 }] };
  p.workers[0].onmessage({ data: { id: m.id, result: deep } });
  await flush();
  const n = p.posted.length;
  assert.equal(p.act({ action: "all-in" }), null);
  const rec = () => p.val("S.actions.flop.at(-1)");
  assert.equal(rec().g, undefined);   // not gradeable yet
  const again = p.posted.slice(n).find(x => x.state && x.state.jam);
  assert.ok(again, "the spot asked again with jam: true");
  assert.deepEqual({ ...again.state, jam: undefined }, { ...m.state, jam: undefined });   // (the spot before the jam)
  p.workers[0].onmessage({ data: { id: again.id, result: { ...deep, actions: [...deep.actions, { action: "all-in", amount_bb: 97.5, freq: 0, ev_bb: -2.3 }] } } });
  await flush();
  assert.deepEqual([rec().g.m, rec().g.f, rec().g.l], ["all-in", 0, 5]);
  assert.equal(p.val("gradeClass(S.actions.flop.at(-1).g)"), "bad");
  // an answer with an all-in grades the jam at once, with no second request (another hand: not the cached spot)
  p.S = state({ ...FLOP, cards: ["Qh", "Qd", "Kd", "7s", "2c"] });
  await flush();
  const m2 = p.posted.at(-1);
  p.workers[0].onmessage({ data: { id: m2.id, result: { ...deep, actions: [...deep.actions, { action: "all-in", amount_bb: 97.5, freq: 0, ev_bb: -2.3 }] } } });
  await flush();
  const n2 = p.posted.length;
  assert.equal(p.act({ action: "all-in" }), null);
  assert.equal(rec().g.m, "all-in");
  assert.ok(!p.posted.slice(n2).some(x => x.state && x.state.jam));
});

test("after the flop a raise the tree has only an all-in for is graded as its all-in; a 4+-way jam is asked again too", async () => {
  const p = await (async () => { const q = loadPage(); await q.ready(); return q; })();
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
