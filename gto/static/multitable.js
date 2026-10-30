// Multi-table practice, loaded by practice.js the first time Tables goes above 1: two to four tables at once against
// GTO opponents, each with its own hand, board, pot, seats and your cards, in a view of its own in place of the page's
// table and coach. The table that needs you comes to the front (outlined, with the action buttons, the size box and
// the keys); tables that wait queue in the order they started waiting. Every decision is graded as at one table, into
// one session score, each table's results and Review mistakes across all of them; every table's hand and the session
// are kept in this browser over a reload.
//
// Solves: the solver answers one request at a time (the browser's worker, or the local server's lock), so this file
// keeps one in flight and picks the next by what it unblocks: answers that are instant (a preflop chart, a flop from the
// library, a street already solved) first, then the other tables' short solves (your spot, an opponent's move: ~2 s at
// Fast), then a look-ahead for the table in front (your likeliest moves played on, as at one table), then long solves
// (Accurate), then the other waiting tables' look-aheads. A long or speculative solve gives way to an instant one where
// it can be stopped (the local server; the browser's worker when the page shares a stop flag with it): a stopped solve
// stays cached and goes on from there when asked again. One that a table no longer needs (its hand ended, you took
// another line, Start over) is stopped the same way.
"use strict";
(() => {
  const X = () => window.Practice.mt;
  const STORE = "gto-mt", SCORE = "gto-mt-score", DRILL = "gto-mt-drill", SEAT = "gto-mt-seat";
  const read = (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } };
  const store = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} };
  const ZERO = { hands: 0, decisions: 0, best: 0, good: 0, evDecisions: 0, evLost: 0, net: 0 };
  const REVIEW_KEEP = 60, REVIEW_SHOW = 6, RECENT = 8, CACHE_KEEP = 400, DRILL_TRIES = 400;
  // seconds: an answer under INSTANT goes first; under SHORT it's a quick solve (Practice's ~2 s at Fast)
  const INSTANT = 0.5, SHORT = 4;
  // ms: the pause after a hand before its table deals again; the keys and buttons ignored for GUARD after the table in
  // front changed by itself (a double press meant for the last table doesn't land on the next); the clock (tests set these)
  const OPT = { pause: 1600, guard: 350, now: () => (typeof performance !== "undefined" && performance.now ? performance.now() : Date.now()) };
  const now = () => OPT.now();
  const r2 = (x) => Math.round(x * 100) / 100;
  const clone = (x) => JSON.parse(JSON.stringify(x));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  // ---- the spots: Instant flops (the default: preflop played for you, each table dealt until a heads-up flop the flop
  // library holds), Any hand, or Practice's drills
  const DRILLS = { lib: "Instant flops", any: "Any hand", srp: "Raised pot", "3bet": "3-bet pot", bvb: "Blind vs blind" };
  const DRILL_TIP = { lib: "Preflop is played for you; each table is dealt until a heads-up flop the precomputed library holds, so flops answer at once",
    any: "Every hand from the start: preflop and the library's flops answer at once, other flops, turns and rivers are solved while you play the other tables",
    srp: "Heads-up after one preflop raise and a call: preflop is played for you", "3bet": "Heads-up after a 3-bet and a call: preflop is played for you",
    bvb: "Small blind against big blind, folded to them: preflop is played for you" };
  const SEEK = { lib: "Dealing an instant flop…", srp: "Dealing a raised pot…", "3bet": "Dealing a 3-bet pot…", bvb: "Dealing blind vs blind…" };

  // tables[i]: { id, st (the table's hand as the page's S), hand: { hero, players, hole, hands, board, result, log, n
  // (this table's hand number), drill }, stats (this table's score), last (its last hand's result, a line) } and, not
  // kept: res/resKey (the advice for your spot, by payload), turnKey/turnAt (your spot, since when), since (ready since:
  // the queue's order), nextAt (deals again then), tries (a drill's deals), err, solved (streets the solver has
  // answered for this hand: their next requests are instant), jams (jams to grade again), cands (Instant flops' lines)
  const saved = read(STORE) || {}, sc0 = read(SCORE) || {};
  const M = { on: false, n: 2, tables: [], focus: 0, focusAt: -1e9, pin: false, err: "", custom: false, dock: null,
    drill: Object.keys(DRILLS).includes(read(DRILL)) ? read(DRILL) : "lib", seat: read(SEAT) || "random",
    score: { ...ZERO, ...(sc0.score || {}) }, review: Array.isArray(sc0.review) ? sc0.review : [], rev: false, pick: null, lastId: sc0.lastId || null,
    spare: [], flight: null, adv: new Map(), moves: new Map(), charts: new Map(), bad: new Set(), seen: new Map(), lib: null, libState: "", seq: 0, timer: null, skel: 0, savedAt: -1e9, saveTimer: null,
    met: { waits: [], solves: [], you: 0, solve: 0, pause: 0, t: null, state: null, stops: 0, charted: 0 } };

  function newTable(i, s) {
    const st = s && s.st ? cleanState(s.st) : null, h = s && s.hand;
    const hand = st && h && Array.isArray(h.hole) && h.hands && Array.isArray(h.board) && Array.isArray(h.log) && h.hero === st.hero && h.players === st.players
      && st.cards[0] === h.hole[0] && st.cards[1] === h.hole[1] ? h : null;
    return { id: i, st: hand ? st : null, hand, stats: { ...ZERO, ...(s && s.stats || {}) }, last: s && s.last || null,
      res: null, resKey: null, turnKey: null, turnAt: null, since: null, nextAt: 0, tries: 0, err: "", solved: new Set(), jams: [], cands: null };
  }
  // every table's hand and the session into this browser: at once after a decision or a hand's end (now), else at most
  // twice a second (a drill's deals flash by)
  function persist(now_) {
    clearTimeout(M.saveTimer); M.saveTimer = null;
    if (!now_ && now() - M.savedAt < 500) { M.saveTimer = setTimeout(() => persist(true), 500); return; }
    M.savedAt = now();
    const kept = (t) => ({ st: t.st, hand: t.hand, stats: t.stats, last: t.last });
    store(STORE, { n: M.n, focus: M.focus, tables: [0, 1, 2, 3].map(i => M.tables[i] ? kept(M.tables[i]) : M.spare[i] || null) });
    store(SCORE, { score: M.score, review: M.review, lastId: M.lastId });
  }
  // run f with the page's S set to a table's hand (simulate(), payload() and act() read the page's S; act() saves it,
  // so the page's own is saved again after: the loop's end, save())
  function within(st, f) {
    const keep = S;
    S = st; if (POS.length !== st.players) setPositions(st.players);
    try { return f(); } finally { S = keep; if (POS.length !== keep.players) setPositions(keep.players); }
  }

  // ---- the flop library's index: which preflop lines and flops it holds (Instant flops; and an instant answer's estimate)
  // canonical flop name, as gto/flop_library.py canonical(): the least of the flop's 24 suit permutations, high to low
  const R13 = "23456789TJQKA", SU = "cdhs";
  const PERMS = []; for (const a of [0, 1, 2, 3]) for (const b of [0, 1, 2, 3]) for (const c of [0, 1, 2, 3]) for (const d of [0, 1, 2, 3]) if (new Set([a, b, c, d]).size === 4) PERMS.push([a, b, c, d]);
  function flopName(cards) {
    const ints = cards.map(c => R13.indexOf(c[0]) * 4 + SU.indexOf(c[1]));
    let best = null;
    for (const p of PERMS) {
      const t = ints.map(c => (c >> 2) * 4 + p[c & 3]).sort((a, b) => b - a);
      if (!best || t[0] < best[0] || (t[0] === best[0] && (t[1] < best[1] || (t[1] === best[1] && t[2] < best[2])))) best = t;
    }
    return best.map(c => R13[c >> 2] + SU[c & 3]).join("");
  }
  const tok = (a) => `${a.pos}:${a.action === "fold" ? "f" : a.action === "call" ? "c" : a.action === "check" ? "x" : a.action === "all-in" ? "a" : "r" + fmt(a.to)}`;
  function setLib(idx) {
    M.lib = (idx && Array.isArray(idx.spots) ? idx.spots : []).filter(s => !s.format && s.line && s.flops).map(s => {
      const toks = s.line.split(",");
      return { id: s.id, label: s.label, toks, players: toks.filter(x => !x.endsWith(":f")).map(x => x.split(":")[0]), flops: new Set(Object.keys(s.flops)) };
    });
    M.libState = M.lib.length ? "ok" : "failed";
  }
  function loadLib() {
    if (M.libState) return;
    M.libState = "loading";
    Promise.resolve().then(() => fetch(LIB_BASE + "index.json", { cache: "no-cache" })).then(r => r.ok ? r.json() : null).catch(() => null)
      .then(idx => { setLib(idx); pump(); });
  }
  // the library answers this game: six players, 100bb each, nothing else (no ante, straddle, rake, tournament, custom
  // sizes, ranges or locks: a request with any key beyond these)
  const STD = new Set(["players", "hero", "hole", "board", "stack", "preflop", "flop", "turn", "river", "budget", "prior_budget", "speed", "quick"]);
  const libGame = () => S.players === 6 && +S.stack === 100 && Object.keys(within({ ...S, cards: fresh().cards, actions: fresh().actions }, payload)).every(k => STD.has(k));
  // the spots dealt: Instant flops outside its game is Any hand
  const drillNow = () => M.drill === "lib" && (!libGame() || M.libState === "failed") ? "any" : M.drill;
  // a flop the library holds for this preflop line (an instant answer)
  function libFlop(t, req) {
    if (!M.lib || !libGame()) return false;
    const line = t.st.actions.preflop.map(tok).join(), name = flopName([req.board.slice(0, 2), req.board.slice(2, 4), req.board.slice(4, 6)]);
    return M.lib.some(s => s.toks.join() === line && s.flops.has(name));
  }

  // ---- preflop played for you (Instant flops and the drills, in the standard game): every seat's move drawn here from
  // the charts' mix at its node, as the solver's draw does (Advisor.play: the chart's mix for that hand), so dealing
  // until a hand reaches the spot asks the solver once per preflop node all session (Study's answer there: each hand
  // class's mix, Advisor._study_preflop) instead of once per seat per hand; a node it can't answer, or a hand its
  // answer doesn't hold, is asked per seat as before
  const chartKey = (req) => JSON.stringify([req.players, req.stack, req.preflop]);
  const chartReq = (req) => ({ players: req.players, stack: req.stack, board: "", preflop: clone(req.preflop), flop: [], turn: [], river: [],
                               study: true, speed: req.speed || "fast" });
  const classOf = (h) => {
    const [a, b] = R13.indexOf(h[0][0]) >= R13.indexOf(h[1][0]) ? h : [h[1], h[0]];
    return a[0] === b[0] ? a[0] + b[0] : a[0] + b[0] + (a[1] === b[1] ? "s" : "o");
  };
  // a seat's move from a chart answer, in Advisor.play's shape ({pos, action, to, name, freq, street, actions}), or null
  function chartMove(ch, seat, hole) {
    const c = ch && ch.actor === seat && ch.grid && ch.grid.cells[classOf(hole)];
    if (!c || !Array.isArray(c.f) || c.f.length !== ch.actions.length) return null;
    const acts = ch.actions.map((a, i) => ({ action: a.action, amount_bb: a.amount_bb, freq: c.f[i] }));
    const live = acts.filter(a => a.freq > 0), from = live.length ? live : acts;
    let r = Math.random() * from.reduce((s, a) => s + a.freq, 0), pick = from[from.length - 1];
    for (const a of from) { r -= a.freq; if (r <= 0) { pick = a; break; } }
    const out = { pos: seat, action: pick.action, name: pick.action, freq: pick.freq, street: "preflop", actions: acts };
    if (/^(bet|raise)/.test(pick.action)) { out.action = pick.action.split(" ")[0]; out.to = r2(+pick.amount_bb); }
    return out;
  }
  const charting = (t, sim) => autoPre(t, sim) && libGame();

  // ---- dealing: a random seat (or the one picked) each hand, the setup's stacks, ante, straddle, rake and the rest
  function deal(t) {
    const page = S, n = page.players, seats = SEATS[n], d = drillNow();
    const deck = []; for (const r of RANKS) for (const s of SUITS) deck.push(r + s);
    for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
    const pool = d === "bvb" ? seats.filter(p => p === "SB" || p === "BB") : seats;
    const hero = pool.includes(M.seat) ? M.seat : pool[Math.floor(Math.random() * pool.length)];
    const hands = {}; for (const p of seats) hands[p] = [deck.pop(), deck.pop()];
    const base = { ...page }; for (const k of ["cards", "actions", "hero", "stacks", "players"]) delete base[k];
    const st = { ...base, ...fresh(n), hero, stack: page.stack, ...cleanStacks(page, seats) };
    st.cards[0] = hands[hero][0]; st.cards[1] = hands[hero][1];
    const hole = hands[hero]; delete hands[hero];
    t.st = st;
    t.hand = { hero, players: n, hole, hands, board: deck.slice(0, 5), result: null, log: [], n: t.stats.hands + 1, drill: d };
    t.res = t.resKey = t.turnKey = t.turnAt = t.since = null; t.err = ""; t.jams = []; t.solved = new Set();
    t.cands = d === "lib" && M.lib ? M.lib.filter(s => s.players.includes(hero) && s.flops.has(flopName(t.hand.board.slice(0, 3)))) : null;
    if (d === "any" && t.id === M.focus) sfx("deal");
  }
  // a drill's preflop has gone another way (the table deals again)
  function offDrill(t, sim) {
    const d = t.hand.drill;
    if (d === "any" || S.cards[2]) return false;
    const pre = S.actions.preflop;
    if (d === "lib") {
      const toks = pre.map(tok), on = (t.cands || []).filter(c => toks.every((x, i) => c.toks[i] === x));
      if (!on.length || sim.handOver || !sim.active.has(S.hero)) return true;
      return sim.waiting === "flop" && !on.some(c => c.toks.length === toks.length);
    }
    const raises = pre.filter(a => a.action === "raise" || a.action === "bet").length, want = { srp: 1, "3bet": 2 }[d], blinds = ["SB", "BB"];
    if (!sim.active.has(S.hero) || pre.some(a => a.action === "all-in") || (want && raises > want)) return true;
    if (d === "bvb" && (!blinds.includes(S.hero) || pre.some(a => !blinds.includes(a.pos) && a.action !== "fold"))) return true;
    if (sim.handOver) return true;
    return sim.waiting === "flop" && (sim.active.size !== 2 || (want != null && raises !== want));
  }
  const autoPre = (t, sim) => t.hand.drill !== "any" && sim.current === "preflop";
  const seeking = (t) => !!t.hand && !t.hand.result && t.hand.drill !== "any" && !t.st.cards[2];

  // ---- the end of a hand: the showdown, its result and the scores
  function count(sc, gs, net) {
    const ev = gs.filter(g => g.l != null);
    sc.hands++; sc.decisions += gs.length; sc.best += gs.filter(g => g.m === g.b).length; sc.good += gs.filter(g => gradeClass(g) === "ok").length;
    sc.evDecisions += ev.length; sc.evLost += ev.reduce((s, g) => s + g.l, 0); sc.net = r2(sc.net + net);
  }
  const headline = (r) => r.folded ? "You folded" : !r.winners.includes(S.hero) ? "You lose" : r.winners.length > 1 ? "Split pot" : "You win";
  function finish(t, sim) {
    const h = t.hand, x = X();
    const inv = { ...sim.invested }, board = boardCards(), live = POS.filter(p => sim.active.has(p)), folded = !sim.active.has(S.hero);
    const cards = (p) => p === S.hero ? h.hole : h.hands[p];
    // (your fold with others in: their chips go as if they checked it down, as at one table)
    const st = folded && live.length > 1 ? x.settle(inv, live, new Set(live), h.board, cards, true) : x.settle(inv, live, sim.allin, board, cards, board.length >= 3);
    const showdown = !folded && live.length > 1, net = r2((folded ? 0 : st.pays[S.hero] || 0) - inv[S.hero]);
    h.result = { net, winners: folded ? [] : st.winners, shown: showdown ? live.filter(p => p !== S.hero) : [], folded,
      best: showdown ? x.CATS[st.val[st.winners[0]][0]] : null, pays: folded && live.length > 1 ? {} : st.pays, ...(showdown ? { says: x.says(st) } : {}) };
    const gs = h.log.map(e => e.g).filter(Boolean);
    count(t.stats, gs, net); count(M.score, gs, net);
    t.last = { n: h.n, net, text: headline(h.result), says: h.result.says || "", seat: S.hero };
    t.nextAt = now() + OPT.pause; t.tries = 0; M.force = true;   // (the loop's timer deals it then)
    if (!folded && net !== 0) sfx(net > 0 ? "win" : "lose", 0.25);
  }

  // ---- a table's next step: as far as it goes without asking the solver (deal, the next street, an opponent's move or
  // your advice already in hand), then the request it waits on: { t, kind: "you" | "opp", key, req, play, est }, or null
  function step(t) {
    for (let k = 0; k < 4096; k++) {   // (a drill's deals drawn here: up to DRILL_TRIES hands of a few moves each)
      if (t.err) return null;
      if (!t.hand || !t.st || (t.hand.result && now() >= t.nextAt && !t.jams.length)) {
        if (drillNow() === "lib" && !M.lib) { loadLib(); return null; }   // (the library's index first)
        deal(t); continue;
      }
      if (t.hand.result) return null;
      const r = within(t.st, () => {
        const sim = simulate();
        if (offDrill(t, sim)) return "redeal";
        if (sim.handOver) { finish(t, sim); return "over"; }
        if (sim.waiting) {
          for (let i = 2; i < 2 + NEED[sim.waiting]; i++) if (!S.cards[i]) S.cards[i] = t.hand.board[i - 2];
          if (sim.waiting === "flop") { t.tries = 0; prefetchLibrary(); }
          if (t.id === M.focus) sfx("card");
          return "next";
        }
        if (!sim.cur) return "stuck";
        const p = sim.toAct, req = payload(), key = JSON.stringify(req);
        if (p === S.hero && !autoPre(t, sim)) {   // your turn: the spot's advice, to grade your move by
          if (t.turnKey !== key) { t.turnKey = key; t.turnAt = now(); t.since = null; t.res = t.resKey = null; }
          if (t.resKey === key) return "ready";
          const res = M.adv.get(key);
          if (!res) return { t, kind: "you", key, req, street: sim.current, est: estimate(t, req) };
          t.res = res; t.resKey = key; t.since = now();
          M.met.waits.push({ s: (t.since - t.turnAt) / 1000, street: sim.current, table: t.id + 1 });
          return "ready";
        }
        const mk = key + "|" + p, hole = p === S.hero ? t.hand.hole : t.hand.hands[p];
        let m = M.moves.get(mk);
        if (!m && charting(t, sim)) {   // (preflop played for you: the node's chart, asked once)
          const ck = chartKey(req), ch = M.charts.get(ck);
          if (ch === undefined) return { t, kind: "chart", key: "chart|" + ck, ck, req: chartReq(req), street: "preflop", est: 0.05 };
          m = chartMove(ch, p, hole);
          if (m) M.met.charted++;
        }
        if (!m) return { t, kind: "opp", key: mk, req, play: { seat: p, hole: hole.join("") }, street: sim.current, est: estimate(t, req) };
        M.moves.delete(mk);
        const e = act({ ...X().move(m), auto: p === S.hero });
        if (e) { t.err = `${SEAT_LABEL(p, S.players)}'s move (${m.name || m.action}) was rejected: ${e}`; return "stuck"; }
        if (t.id === M.focus && !seeking(t)) X().heard(S.actions[sim.current].at(-1));
        return "next";
      });
      if (r === "redeal") {
        if (++t.tries > DRILL_TRIES) { t.err = `No ${DRILLS[t.hand.drill].toLowerCase()} came up in ${DRILL_TRIES} deals: try another seat or spot.`; return null; }
        deal(t); continue;
      }
      if (r === "next" || r === "over") continue;   // (over: the pause, or with none the next hand at once)
      if (typeof r === "string") return null;
      return r;
    }
    return null;
  }
  // how long a request takes, roughly (s): it orders the queue (instant first) and keeps long solves from blocking
  const streetKey = (q) => JSON.stringify([q.hero, q.board, q.preflop, q.board.length >= 8 ? q.flop : 0, q.board.length >= 10 ? q.turn : 0, q.speed]);
  function estimate(t, req) {
    const nb = req.board.length / 2;
    if (nb < 3) return 0.05;                              // a preflop chart
    if (t.solved.has(streetKey(req))) return 0.2;         // the street's solve is in hand (a reply heads-up reads it)
    if (nb === 3 && libFlop(t, req)) return 0.3;          // a library flop
    const street = nb === 3 ? "flop" : nb === 4 ? "turn" : "river";
    if (SPEED !== "accurate") return { flop: 2, turn: 2, river: 1.4 }[street];   // Practice's quick answers
    return (MODE === "server" ? 30 : 60) * (street === "river" ? 0.4 : 1);
  }
  const ready = (t) => !!(t && t.hand && !t.hand.result && !t.err && t.resKey && t.resKey === t.turnKey);
  // the tables waiting for you, in the order they started waiting
  const queue = () => M.tables.filter(ready).sort((a, b) => a.since - b.since || a.id - b.id);
  const front = () => M.tables[M.focus] || null;
  // the table in front: kept while it waits for you (or you picked it), else the first in the queue; right after you
  // act the first in the queue, also when the table you acted at is waiting again at once (it waits behind the others)
  function refocus() {
    const f = front(), next = M.next; M.next = false;
    if (f && !next && (ready(f) || (M.pin && !(f.hand && f.hand.result)))) return;
    const q = queue();
    if (q.length && q[0] !== f) { M.focus = q[0].id; M.focusAt = now(); M.custom = false; M.err = ""; }
    M.pin = false;
  }

  // ---- looking ahead, as at one table: at each table waiting for you, your likeliest moves played on to the next
  // request that takes a solve (an opponent's reply, your next spot), its answer kept for when you take that line
  function walk(t, rec) {
    const st = { ...t.st, actions: clone(t.st.actions), cards: t.st.cards.slice() };
    return within(st, () => {
      if (act({ ...rec, auto: true })) return null;
      for (let k = 0; k < 24; k++) {
        const sim = simulate();
        if (sim.handOver || (!sim.cur && !sim.waiting)) return null;
        if (sim.waiting) { for (let i = 2; i < 2 + NEED[sim.waiting]; i++) if (!S.cards[i]) S.cards[i] = t.hand.board[i - 2]; continue; }
        const p = sim.toAct, req = payload(), key = JSON.stringify(req);
        if (p === S.hero) return M.adv.has(key) || M.bad.has(key) ? null : { t, kind: "you", ahead: true, key, req, street: sim.current, est: estimate(t, req) };
        const mk = key + "|" + p, m = M.moves.get(mk);
        if (!m) return M.bad.has(mk) ? null : { t, kind: "opp", ahead: true, key: mk, req, play: { seat: p, hole: t.hand.hands[p].join("") }, street: sim.current, est: estimate(t, req) };
        if (act(X().move(m))) return null;
      }
      return null;
    });
  }
  function aheads() {
    if (MODE !== "server" && !STOP && SPEED === "accurate") return [];   // (unstoppable: a wrong guess would block for long)
    const out = [], f = front();
    for (const t of [...(ready(f) ? [f] : []), ...queue().filter(x => x !== f)]) {
      const lines = within(t.st, () => X().likely(t.res, simulate()));
      lines.forEach((c, i) => { const j = walk(t, c.rec); if (j && j.est < SHORT) out.push({ ...j, g: t === f ? 3 : 5, rank: i }); });
    }
    return out;
  }
  // a request's priority group: 1 instant, 2 a short solve, 3 a look-ahead at the table in front, 4 a long solve,
  // 5 the other tables' look-aheads; then yours before an opponent's (look-aheads: the quickest, then your likeliest
  // line: each line's next request before a line's one after), then the longest waiting
  const group = (j) => j.ahead ? j.g : j.kind === "jam" ? 2 : j.est < INSTANT ? 1 : j.est < SHORT ? 2 : 4;
  const KIND = { you: 0, opp: 1, chart: 1, jam: 2 };
  const age = (j) => { if (!M.seen.has(j.key)) M.seen.set(j.key, now()); return M.seen.get(j.key); };
  const byPriority = (a, b) => group(a) - group(b) || (a.ahead && b.ahead ? a.est - b.est || a.rank - b.rank : KIND[a.kind] - KIND[b.kind])
    || age(a) - age(b) || a.t.id - b.t.id;
  function jobs() {
    const out = [];
    for (const t of M.tables) {
      const j = step(t); if (j) out.push(j);
      for (const x of t.jams) out.push({ t, kind: "jam", key: x.key, req: x.req, jam: x, est: 2 });
    }
    return out;
  }
  // stop the request in flight, where it can be (its solve stays cached, to go on from there if asked again)
  function stopFlight() {
    const f = M.flight;
    if (!f || f.stopping) return false;
    if (MODE === "server") { f.stopping = true; M.met.stops++; fetch("cancel", { method: "POST" }).catch(() => {}); return true; }
    if (f.id != null && STOP && stopRequests(f.id)) { f.stopping = true; M.met.stops++; return true; }
    return false;
  }
  function launch(j) {
    const t0 = now();
    let c; try { c = callAdvise(j.req, j.play); } catch (e) { c = Promise.resolve({ error: String(e) }); }
    const fl = M.flight = { job: j, id: c.reqId, t0, stopping: false };
    Promise.resolve(c).catch(e => ({ error: "Request failed: " + e })).then(res => {
      if (M.flight === fl) M.flight = null;
      took(j, res || { error: "no answer" }, (now() - t0) / 1000);
      pump();
    });
  }
  const trim = (m) => { while (m.size > CACHE_KEEP) m.delete(m.keys().next().value); };
  // an answer: into the caches the tables read (a stop that came too late leaves a whole answer: kept too)
  function took(j, res, s) {
    M.seen.delete(j.key);
    if (res.cancelled) return;
    if (j.kind === "chart") {   // (a node it can't answer: its seats asked one by one)
      M.charts.set(j.ck, !res.error && res.study === "node" && res.street === "preflop" && res.grid && Array.isArray(res.actions) ? res : null);
      return trim(M.charts);
    }
    if (res.error) {
      if (j.ahead) { M.bad.add(j.key); if (M.bad.size > 500) M.bad.clear(); }
      else if (j.kind === "jam") j.t.jams = j.t.jams.filter(x => x !== j.jam);
      else if (M.tables[j.t.id] === j.t) j.t.err = `${j.kind === "you" ? "Your spot" : j.play.seat + "'s move"} failed: ${res.error}`;
      return;
    }
    M.met.solves.push({ s, est: j.est, kind: j.kind, ahead: !!j.ahead, street: j.street || "" });
    if (M.met.solves.length > 2000) M.met.solves.shift();
    if (j.kind === "jam") return regrade(j, res);
    if (j.play) { if (!M.moves.has(j.key)) M.moves.set(j.key, res); trim(M.moves); }
    else { if (!M.adv.has(j.key)) M.adv.set(j.key, res); trim(M.adv); }
    j.t.solved.add(streetKey(j.req));
  }
  // ---- the loop: every table stepped, the table in front picked, and the solver given its next request
  function pump() {
    if (!M.on) return;
    clearTimeout(M.timer); M.timer = null;
    if (typeof overlay !== "undefined" && overlay) { M.timer = setTimeout(pump, 1000); return render(); }   // (a view open over Practice)
    if (!MODE) { modeReady.then(pump); return render(); }
    const js = jobs();
    refocus();
    metric(js);
    const ah = aheads(), all = [...js, ...ah].sort(byPriority), f = M.flight;
    if (f && !f.stopping) {
      const same = all.find(j => j.key === f.job.key);
      if (!same) stopFlight();   // nobody needs it now (its hand ended, its line left)
      else {
        f.job = same;   // (a look-ahead your move made real keeps its place, as the real request)
        const best = all.find(j => j.key !== f.job.key);
        if (best && group(best) <= 2 && group(best) < group(same) && group(same) >= 3) stopFlight();   // something instant or short waits behind it
      }
    }
    if (!M.flight && all.length) launch(all[0]);
    const later = M.tables.filter(t => t.hand && t.hand.result && t.nextAt > now()).map(t => t.nextAt);
    if (later.length) M.timer = setTimeout(pump, Math.min(...later) - now() + 30);
    if (M.seen.size > 1000) M.seen.clear();
    save(); persist(M.force); M.force = false;
    render();
  }
  // where the time goes: some table waiting for you (you), none but a solve under way (solve), between hands (pause)
  function metric(js) {
    const m = M.met, t = now(), st = M.tables.some(ready) ? "you" : js.length || M.flight ? "solve" : "pause";
    if (m.t != null && m.state) m[m.state] += t - m.t;
    m.t = t; m.state = st;
  }

  // ---- your move at the table in front: graded against its spot's advice, as at one table
  function heroAct(a) {
    const t = front();
    if (!ready(t)) return "It's not your turn at this table.";
    if (now() - M.focusAt < OPT.guard) return "guard";
    const x = X();
    const err = within(t.st, () => {
      const sim = simulate(), street = sim.current, info = sim.cur, facing = info.curBet - info.bets[S.hero] > 1e-9;
      const ctx = x.spot(street), before = S.actions[street].filter(b => b.action !== "fold");
      const line = before.length ? before.map(b => `${SEAT_LABEL(b.pos, S.players)} ${x.ACT(b)}`).join(", ") : street === "preflop" ? "folded to you" : "first to act";
      const e0 = act({ ...a, auto: true });
      if (e0) return e0;
      const entry = S.actions[street].at(-1), rec = { action: entry.action, ...(entry.to != null ? { to: entry.to } : {}) };
      const move = gradedMove(sim, rec), off = jamOffTree(t.res, move), g = off ? null : gradeWith(t.res, move);
      if (g) entry.g = g;
      x.habit(facing, rec.action);
      const id = Date.now().toString(36) + (++M.seq).toString(36);
      const e = { id, street, rec, g, mix: x.mixOf(t.res), ctx, table: t.id + 1 };
      t.hand.log.push(e);
      M.review.push({ ...e, n: t.hand.n, seat: S.hero, hole: t.hand.hole, board: boardCards(), line });
      M.review = M.review.slice(-REVIEW_KEEP); M.lastId = id; M.pick = null;
      if (off) t.jams.push({ key: "jam|" + t.resKey, req: { ...JSON.parse(t.resKey), jam: true }, id, move });
      x.heard(rec);
      const gc = g && gradeClass(g);
      if (gc === "ok" || gc === "bad") sfx(gc === "ok" ? "good" : "bad", 0.2);
      return null;
    });
    if (err) { save(); M.err = err; render(); return err; }
    t.res = t.resKey = t.since = null; M.custom = false; M.err = ""; M.pin = false; M.next = true;
    pump(); persist(true);
    return null;
  }
  // a jam deeper than its answer's tree, asked again with all-in in it (index.html jamGrade): its verdict, and the
  // scores if its hand has already been counted
  function regrade(j, res) {
    const t = j.t, x = j.jam;
    t.jams = t.jams.filter(y => y !== x);
    const g = gradeWith(res, x.move);
    if (!g) return;
    const e = t.hand && t.hand.log.find(y => y.id === x.id), r = M.review.find(y => y.id === x.id), mix = X().mixOf(res);
    for (const y of [e, r]) if (y && !y.g) { y.g = g; y.mix = mix || y.mix; }
    if (e && t.hand.result) for (const s of [t.stats, M.score]) {
      s.decisions++; s.best += g.m === g.b ? 1 : 0; s.good += gradeClass(g) === "ok" ? 1 : 0;
      if (g.l != null) { s.evDecisions++; s.evLost += g.l; }
    }
    persist(true);
  }

  // ---- the view
  const CSS = `
  body.mt main { display: none !important; }
  #mtView { padding: 12px 16px calc(16px + env(safe-area-inset-bottom)); max-width: 1560px; margin: 0 auto; }
  .mt-top { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; margin-bottom: 10px; }
  .mt-top .grp { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .mt-top .lbl { font-size: 12px; color: var(--muted); white-space: nowrap; }
  .mt-top .cx-seats { margin: 0; flex-wrap: nowrap; }
  .mt-top .cx-seats button { padding: 5px 9px; }
  .mt-note { width: 100%; font-size: 12px; color: var(--muted); line-height: 1.45; }
  .mt-note b { color: var(--text); font-weight: 600; }
  .mt-body { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 14px; align-items: start; }
  .mt-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  @media (min-width: 1300px) { .mt-grid.n3 { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  .mt-t { position: relative; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r); padding: 6px 8px 8px; cursor: pointer; min-width: 0; transition: border-color .15s, box-shadow .15s; }
  .mt-t:hover { border-color: var(--line2); }
  .mt-t.wait { border-color: rgba(242,181,68,.55); }
  .mt-t.cur { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), 0 0 18px rgba(242,181,68,.18); cursor: default; }
  .mt-t .hd { display: flex; align-items: center; gap: 8px; font-size: 12px; min-height: 22px; }
  .mt-t .hd .nm { font-weight: 700; }
  .mt-t .hd .sub { color: var(--muted); }
  .mt-t .hd .sts { margin-left: auto; display: flex; align-items: center; gap: 6px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
  .mt-t .hd .sts .spinner { width: 11px; height: 11px; margin: 0; }
  .mt-t.wait .hd .sts, .mt-t.cur.wait .hd .sts { color: var(--accent); font-weight: 700; }
  .mt-q { display: inline-grid; place-items: center; min-width: 18px; height: 18px; padding: 0 4px; border-radius: 9px; background: var(--accent); color: #1a1200; font-size: 11px; font-weight: 800; }
  .mt-felt { position: relative; height: var(--mt-h, clamp(170px, 26vh, 240px)); }
  .mt-grid.n2 { --mt-h: clamp(210px, 36vh, 330px); }
  .mt-felt::before { content: ""; position: absolute; inset: 14% 8%; border-radius: 999px; background: radial-gradient(ellipse at center, #1d533b 0%, #163c2b 70%, #112f20 100%); border: 5px solid #241c16; box-shadow: inset 0 0 24px rgba(0,0,0,.45); }
  .mt-t.seek .mt-felt { opacity: .55; }
  .mt-seat { position: absolute; transform: translate(-50%, -50%); z-index: 2; background: var(--panel2); border: 1px solid var(--line); border-radius: 7px; padding: 1px 4px 2px; min-width: 48px; text-align: center; font-size: 10.5px; line-height: 1.3; font-variant-numeric: tabular-nums; }
  .mt-seat .p { font-weight: 700; font-size: 11px; }
  .mt-seat .k { color: var(--muted); font-weight: 400; margin-left: 3px; }
  .mt-seat .a { display: block; color: var(--muted); min-height: 13px; }
  .mt-seat.f { opacity: .38; }
  .mt-seat.me { border-color: var(--accent); }
  .mt-seat.ta { border-color: var(--accent); }
  .mt-seat.ta.th { animation: think 1.4s ease-in-out infinite; }
  .mt-seat.won { border-color: var(--good); }
  .mt-seat.won .a { color: var(--good); font-weight: 700; }
  .mt-seat .hc { display: flex; justify-content: center; gap: 2px; margin: 1px 0; font-weight: 700; font-size: 11.5px; }
  .mt-seat .hc > span { background: #f5f5f0; border-radius: 3px; padding: 0 2px; line-height: 1.3; color: #111; }
  .mt-seat .dlr { position: absolute; right: -6px; bottom: -6px; width: 15px; height: 15px; border-radius: 50%; background: #f5f5f0; color: #111; font-size: 9px; font-weight: 800; line-height: 15px; }
  .mt-center { position: absolute; left: 50%; top: 47%; transform: translate(-50%, -50%); text-align: center; z-index: 1; }
  .mt-center .board { display: flex; gap: 3px; justify-content: center; }
  .mt-center .bc { width: 25px; height: 35px; font-size: 12px; }
  .mt-center .pot { margin-top: 3px; font-size: 11px; color: #d2e2d7; font-variant-numeric: tabular-nums; }
  .mt-foot { display: flex; align-items: center; gap: 10px; min-height: 40px; font-size: 12px; }
  .mt-foot .mine { display: flex; gap: 2px; flex: none; }
  .mt-foot .mine .bc { width: 27px; height: 38px; font-size: 13px; }
  .mt-foot .info { min-width: 0; display: flex; flex-direction: column; gap: 1px; }
  .mt-foot .info > * { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .mt-foot .made b { font-weight: 600; }
  .mt-foot .err { color: var(--bad); white-space: normal; }
  .mt-foot .res b { font-size: 13px; }
  .mt-pill { font-size: 11px; font-weight: 700; color: var(--c); }
  .mt-dock { margin-top: 10px; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r); padding: 10px 12px 6px; }
  .mt-dock .prompt { margin-bottom: 6px; }
  .mt-dock .acts button { min-height: 54px; font-size: 15px; max-width: none; min-width: max-content; padding: 8px 30px 8px 12px; }
  .mt-dock .acts button kbd { top: 5px; right: 5px; font-size: 11px; line-height: 15px; padding: 0 4px; color: var(--muted); border-color: var(--line2); }
  .mt-dock .custom { display: flex; }
  .mt-dock .idle { color: var(--muted); font-size: 13px; padding: 10px 0 12px; display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
  .mt-dock .hint { font-size: 11.5px; color: var(--faint); margin: 2px 0 4px; }
  .mt-dock .err { color: var(--bad); font-size: 12px; min-height: 0; }
  .mt-side { min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r); padding: 12px 14px 14px; }
  .mt-side .cx-stats { margin-top: 0; padding-top: 0; border-top: 0; }
  .mt-side .cx-tools { display: flex; gap: 8px; margin-top: 10px; }
  .mt-tbls { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .mt-tbls th { font-weight: 400; color: var(--muted); text-align: right; padding: 3px 4px; font-size: 11px; }
  .mt-tbls td { text-align: right; padding: 4px; border-top: 1px solid var(--line); white-space: nowrap; }
  .mt-tbls th:first-child, .mt-tbls td:first-child { text-align: left; }
  .mt-tbls td.lst { color: var(--muted); max-width: 90px; overflow: hidden; text-overflow: ellipsis; }
  .mt-tbls tr.cur td:first-child { color: var(--accent); font-weight: 700; }
  .mt-tabs { display: none; }
  @media (max-width: 1100px) { .mt-body { grid-template-columns: minmax(0, 1fr); } }
  @media (max-width: 560px) {
    #mtView { padding: 8px 10px calc(10px + env(safe-area-inset-bottom)); }
    .mt-top { gap: 6px 10px; }
    .mt-top .grp { width: 100%; }
    .mt-top .cx-seats { flex: 1; flex-wrap: wrap; }
    .mt-top .cx-seats button { padding: 5px 7px; }
    .mt-note:not(.warn) { display: none; }
    .mt-tabs { display: flex; gap: 6px; position: sticky; top: 0; z-index: 6; background: var(--bg); padding: 6px 0; margin-bottom: 4px; }
    .mt-tab { flex: 1; display: flex; align-items: center; justify-content: center; gap: 6px; min-height: 40px; padding: 4px; font-weight: 700; }
    .mt-tab .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--line2); }
    .mt-tab.wait .dot { background: var(--accent); }
    .mt-tab.done .dot { background: var(--muted); }
    .mt-tab .spinner { width: 10px; height: 10px; margin: 0; }
    .mt-tab[aria-selected="true"] { border-color: var(--accent); color: var(--accent); }
    .mt-tab.wait:not([aria-selected="true"]) { background: rgba(242,181,68,.12); }
    .mt-grid { grid-template-columns: minmax(0, 1fr) !important; }
    .mt-t:not(.cur) { display: none; }
    .mt-t { --mt-h: clamp(200px, 34vh, 260px); }
    .mt-dock { position: sticky; bottom: 0; z-index: 5; box-shadow: 0 -8px 20px rgba(0,0,0,.45); padding: 8px 10px 4px; }
    .mt-dock .acts button { min-width: 0; flex: 1 1 28%; padding: 8px 22px 8px 8px; font-size: 14px; }
    .mt-dock .acts button kbd { display: none; }
    .mt-dock .hint { display: none; }
  }`;
  function view() {
    let box = document.getElementById("mtView");
    if (!box) {
      box = document.createElement("div"); box.id = "mtView";
      const main = document.querySelector && document.querySelector("main");
      if (main && main.after) main.after(box); else document.body.appendChild(box);
    }
    if (!document.getElementById("mtStyle")) { const s = document.createElement("style"); s.id = "mtStyle"; s.textContent = CSS; document.head.appendChild(s); }
    if (!box.mtWired) { box.mtWired = true; wire(box); }
    if (M.skel !== M.n) {   // the frame, once per table count: each part below is filled on its own (a part that hasn't changed keeps its focus)
      M.skel = M.n;
      box.innerHTML = `<div class="mt-top" id="mtTop"></div><div class="mt-tabs" id="mtTabs" role="tablist" aria-label="Tables"></div><div class="mt-body"><div class="mt-left">`
        + `<div class="mt-grid n${M.n}" id="mtGrid">${M.tables.map(t => `<div class="mt-t" id="mtT${t.id}" data-t="${t.id}"></div>`).join("")}</div>`
        + `<div class="mt-dock" id="mtDock" aria-live="polite"></div></div><div class="mt-side" id="mtSide"></div></div>`;
      for (const id of ["mtTop", "mtTabs", "mtDock", "mtSide", ...M.tables.map(t => "mtT" + t.id)]) { const el = document.getElementById(id); if (el) el.mtHtml = null; }
    }
    return box;
  }
  const part = (id, html, cls) => {
    const el = document.getElementById(id); if (!el) return;
    if (cls != null && el.className !== cls) el.className = cls;
    if (el.mtHtml !== html) { el.innerHTML = html; el.mtHtml = html; }
  };
  const radio = (name, opts, cur, off = []) => `<div class="cx-seats" role="radiogroup" aria-label="${name}">` + opts.map(([v, l, tip]) =>
    `<button type="button" role="radio" aria-checked="${String(v) === String(cur)}" data-${name}="${v}" title="${esc(tip)}"${off.includes(v) ? " disabled" : ""}>${l}</button>`).join("") + `</div>`;
  function topHtml() {
    const seats = SEATS[S.players], d = drillNow(), pool = d === "bvb" ? ["SB", "BB"] : seats;
    const libOff = !libGame() || M.libState === "failed";
    const tables = `<div class="grp"><span class="lbl">Tables</span>${radio("tables", [1, 2, 3, 4].map(n => [n, n, n === 1 ? "One table: back to the page's own hand" : `${n} tables at once`]), M.n)}</div>`;
    const spots = `<div class="grp"><span class="lbl">Spots</span>${radio("drill", Object.entries(DRILLS).map(([k, l]) => [k, l, k === "lib" && libOff ? "Needs the standard game: 6 players, 100bb, no ante, straddle, rake, tournament or advanced settings" : DRILL_TIP[k]]), d, libOff ? ["lib"] : [])}</div>`;
    const seat = `<div class="grp"><span class="lbl">Seat</span>${radio("seat", [["random", "Random", "A random seat at each table, each hand"], ...seats.map(p => [p, SEAT_LABEL(p, seats.length), `Every hand from ${SEAT_LABEL(p, seats.length)}`])], pool.includes(M.seat) ? M.seat : "random", seats.filter(p => !pool.includes(p)))}</div>`;
    const quick = SPEED !== "accurate";
    let note = d === "lib" ? `<b>Instant flops</b>: preflop is played for you and each table is dealt until a heads-up flop the library holds, so flops answer at once; turns (~2 s) and rivers (~1 s) are solved while you play the other tables.`
      : d === "any" ? `<b>Any hand</b>: preflop and the library's flops answer at once; other flops, turns and rivers take ${quick ? "~2 s" : "longer"} each, solved while you play the other tables (a table waiting on one shows a spinner).`
      : `<b>${DRILLS[d]}</b>: preflop is played for you; the library's flops answer at once, other flops, turns and rivers take ${quick ? "~2 s" : "longer"} each.`;
    if (M.drill === "lib" && d !== "lib") note = `Instant flops needs the standard game (6 players, 100bb, no ante, straddle, rake, tournament or advanced settings), so this is <b>Any hand</b>: other flops, turns and rivers take ${quick ? "~2 s" : "longer"} each.`;
    if (!quick) note += ` At <b>Accurate</b> a solve takes up to ${MODE === "server" ? "30 s" : "a minute"}, so tables wait longer: <b>Fast</b> suits several tables.`;
    const setup = `${S.players} players · ${fmt(S.stack)} bb${S.ante ? ` · ante ${fmt(S.ante)}` : ""}${RAKE() ? " · rake" : ""}${PAYOUTS().length ? " · tournament" : ""}`;
    note += ` <span class="muted">${setup}: the setup is changed at 1 table.</span>`;
    return tables + spots + seat + `<div class="mt-note${!quick || (M.drill === "lib" && d !== "lib") ? " warn" : ""}">${note}</div>`;
  }
  // what a table is doing, for its header and the phone's tab
  function stateOf(t) {
    if (!t.hand) return { cls: "think", text: M.libState === "loading" ? "Loading the flop library…" : "Dealing…", spin: true };
    if (t.err) return { cls: "err", text: "Something went wrong" };
    if (t.hand.result) return { cls: "done", text: "Next hand…" };
    if (seeking(t)) return { cls: "think seek", text: SEEK[t.hand.drill], spin: true };
    if (ready(t)) { const q = queue(), i = q.indexOf(t); return { cls: "wait", text: "Your turn", q: q.length > 1 ? i + 1 : 0 }; }
    const sim = within(t.st, simulate);
    if (sim.toAct === t.st.hero) return { cls: "think", text: "Solving your spot…", spin: true };
    return { cls: "think", text: sim.toAct ? `${SEAT_LABEL(sim.toAct, t.st.players)} is thinking…` : "Dealing…", spin: true };
  }
  function tableHtml(t, s) {
    const head = `<div class="hd"><span class="nm">Table ${t.id + 1}</span>${t.hand ? `<span class="sub">#${t.hand.n} · ${SEAT_LABEL(t.hand.hero, t.hand.players)}</span>` : ""}`
      + `<span class="sts">${s.spin ? '<span class="spinner"></span>' : ""}${s.text}${s.q ? ` <span class="mt-q" title="Your place in the queue">${s.q}</span>` : ""}</span></div>`;
    if (!t.hand) return head + `<div class="mt-felt"></div><div class="mt-foot"></div>`;
    const x = X(), h = t.hand, r = h.result;
    return within(t.st, () => {
      const sim = simulate(), n = POS.length, hi = Math.max(0, POS.indexOf(S.hero));
      const seats = POS.map((p, i) => {
        const th = (90 + (i - hi) * 360 / n) * Math.PI / 180, folded = !sim.active.has(p);
        const bet = sim.cur ? sim.cur.bets[p] : 0, total = sim.invested[p] + bet, behind = Math.max(0, STACK(p) - total);
        const show = p === S.hero ? h.hole : r && r.shown.includes(p) ? h.hands[p] : null, won = r && r.winners.includes(p);
        const amt = won ? `wins ${fmt((r.pays || {})[p] || 0)}` : folded ? "fold" : sim.allin.has(p) ? "all-in" : bet > 0 ? fmt(bet) : "";
        const cls = "mt-seat" + (p === S.hero ? " me" : "") + (folded && !show ? " f" : "") + (sim.toAct === p && !r ? " ta" + (p !== S.hero || !ready(t) ? " th" : "") : "") + (won ? " won" : "");
        return `<div class="${cls}" style="left:${(50 + 42 * Math.cos(th)).toFixed(1)}%;top:${(50 + 38 * Math.sin(th)).toFixed(1)}%"><span class="p">${SEAT_LABEL(p, n)}</span><span class="k">${fmt(behind)}</span>`
          + (show ? `<span class="hc">${cardHtml(show[0])}${cardHtml(show[1])}</span>` : "") + `<span class="a">${amt}</span>`
          + (p === (n === 2 ? "SB" : "BTN") ? '<span class="dlr" title="The dealer button">D</span>' : "") + `</div>`;
      }).join("");
      const board = S.cards.slice(2).map(c => c ? `<div class="bc ${c[1]}">${c[0]}<span>${SUIT_SYM[c[1]]}</span></div>` : `<div class="bc empty"></div>`).join("");
      const felt = `<div class="mt-felt">${seats}<div class="mt-center"><div class="board">${board}</div><div class="pot">pot ${fmt(sim.pot)} bb</div></div></div>`;
      const b = boardCards(), last = h.log[h.log.length - 1];
      let info;
      if (t.err) info = `<span class="err">${esc(t.err)}</span><span><button class="small" data-retry="${t.id}">Retry</button></span>`;
      else if (r) info = `<span class="res"><b>${headline(r)}</b> <span class="num ${r.net > 0 ? "pos" : r.net < 0 ? "neg" : ""}">${x.money(r.net)}</span></span>`
        + `<span class="muted">${esc(r.says || (r.best ? `${r.winners.includes(S.hero) ? "with" : "against"} ${r.best}` : ""))}</span>`;
      else {
        info = b.length >= 3 ? `<span class="made">${x.handName(x.rank7([...h.hole, ...b]))}</span>` : `<span class="made muted">${sim.current || ""}</span>`;
        if (last) { const v = x.verdict(last.g); info += `<span><span class="mt-pill" style="--c:${v.c}">${v.label}</span> <span class="muted">${last.street} · you ${x.YOU(last.rec)}</span></span>`; }
      }
      return head + felt + `<div class="mt-foot"><div class="mine">${x.big(h.hole, true)}</div><div class="info">${info}</div></div>`;
    });
  }
  function tabsHtml(states) {
    return M.tables.map((t, i) => { const s = states[i];
      return `<button type="button" class="mt-tab ${s.cls}" role="tab" aria-selected="${t.id === M.focus}" data-t="${t.id}" title="Table ${t.id + 1}: ${esc(s.text)}">${t.id + 1}`
        + (s.spin ? '<span class="spinner"></span>' : `<span class="dot"></span>`) + (s.q ? `<span class="mt-q">${s.q}</span>` : "") + `</button>`; }).join("");
  }
  // the action dock: the table in front's buttons (the solver's sizes once its spot is solved, as at one table)
  function dockModel(t) {
    return within(t.st, () => {
      const sim = simulate(), info = sim.cur, p = S.hero, street = sim.current;
      const myBet = info.bets[p], toCall = info.curBet - myBet, remaining = STACK(p) - sim.invested[p] - myBet, facing = toCall > 1e-9, callAmt = Math.min(toCall, remaining);
      const res = t.res && !t.res.error && !heuristic(t.res) ? t.res : null, acts = [];
      if (facing) { acts.push({ label: "Fold", cls: "fold", a: { action: "fold" }, key: "F" }); acts.push({ label: "Call", cls: "pass", a: { action: "call" }, sub: fmt(callAmt) + " bb", key: "C" }); }
      else acts.push({ label: "Check", cls: "pass", a: { action: "check" }, key: "C" });
      const raiseOk = canRaise(sim, p), can = remaining > toCall + 1e-9 && raiseOk, minTo = minRaiseTo(info), verb = facing || info.curBet > 0 ? "Raise to" : "Bet";
      let k = 0;
      if (can && res) {
        for (const x of res.actions) {
          if (!/^(bet|raise)/.test(x.action) || x.amount_bb == null) continue;
          const v = legalTo(x.amount_bb, info);
          if (!(v < myBet + remaining - 1e-9) || acts.some(a => a.a && a.a.to === v)) continue;
          const tag = /%$/.test(x.action) ? x.action.replace(/^bet /, "") + " pot" : street === "preflop" && info.nRaises === 0 ? "open" : (v / info.curBet).toFixed(1) + "x";
          acts.push({ label: `${verb} ${fmt(v)}`, cls: "agg", a: { action: "bet", to: v }, sub: tag, key: String(++k) });
        }
      } else if (can) {
        const half = (x) => Math.round(x * 2) / 2;
        let sizes;
        if (street === "preflop") { const v = half(preflopSize(sim, p)); sizes = [[v, info.nRaises === 0 ? "open" : fmt(v / info.curBet) + "x"]]; }
        else if (!facing) sizes = [[half(info.pot / 3), "⅓ pot"], [half(info.pot * 0.66), "⅔ pot"], [half(info.pot), "pot"]];
        else sizes = [[half(info.curBet + 0.75 * (info.pot + toCall)), "¾ pot"], [half(info.curBet + info.pot + toCall), "pot"]];
        for (const [v, l] of sizes) if (v < myBet + remaining - 1e-9 && v >= minTo - 1e-9) acts.push({ label: `${verb} ${fmt(v)}`, cls: "agg", a: { action: "bet", to: v }, sub: l, key: String(++k) });
      }
      if (can) acts.push({ label: verb, cls: "agg", custom: true, sub: "your size…", key: "R" });
      if (remaining > 0 && (raiseOk || remaining <= toCall + 1e-9)) acts.push({ label: "All-in", cls: "agg", a: { action: "all-in" }, sub: fmt(myBet + remaining) + " bb", key: "A" });
      return { t: t.id, acts, street, pot: info.pot, callAmt, facing, verb, need: facing ? callAmt / (info.pot + callAmt) : 0 };
    });
  }
  function dockHtml() {
    const t = front(), q = queue();
    if (!ready(t)) {
      M.dock = null;
      const busy = M.tables.filter(x => x.hand && !x.hand.result && !x.err).length;
      const msg = q.length ? `Table ${q[0].id + 1} is waiting for you: <button class="small" data-t="${q[0].id}">Go<kbd>S</kbd></button>`
        : !t ? "" : t.err ? `Table ${t.id + 1} needs a retry.` : busy ? `<span class="spinner"></span>No table needs you yet: the solver is on ${busy === 1 ? "the other table" : `${busy} tables`}.` : `<span class="spinner"></span>Dealing the next hands…`;
      return `<div class="idle">${msg}</div>`;
    }
    const d = M.dock = dockModel(t), seat = SEAT_LABEL(t.st.hero, t.st.players), more = q.filter(x => x !== t).length;
    const prompt = `<div class="prompt"><span class="who hero">Table ${t.id + 1} · your turn</span><span class="pot">${seat} · ${d.street} · pot ${fmt(d.pot)} bb${d.facing ? ` · ${fmt(d.callAmt)} to call · need ${(d.need * 100).toFixed(0)}%` : ""}</span></div>`;
    const acts = `<div class="acts">` + d.acts.map((a, i) => `<button type="button" class="${a.cls}" data-act="${i}">${a.label}${a.sub ? `<small>${a.sub}</small>` : ""}<kbd>${a.key}</kbd></button>`).join("") + `</div>`;
    const custom = M.custom ? `<div class="custom"><span class="muted">${d.verb === "Bet" ? "bet" : "raise to"}</span><input type="number" id="mtAmt" step="0.5" min="0" inputmode="decimal"><span class="muted">bb</span><button class="small" data-apply="1">Apply</button></div>` : "";
    const hint = `<div class="hint">${more ? `${more} more waiting · S next table · ` : ""}F fold · C check/call · 1-${Math.max(1, d.acts.filter(a => /^\d$/.test(a.key)).length)} sizes · R your size · A all-in</div>`;
    return prompt + acts + custom + (M.err && M.err !== "guard" ? `<div class="err">${esc(M.err)}</div>` : "") + hint;
  }
  // the side: the session's score, the last decision's verdict, each table's results, the recent decisions or mistakes
  const lostOf = (e) => e.g && e.g.l != null ? e.g.l : 0;
  function mistakes() {
    const x = X(), bad = M.review.map((e, i) => ({ e, i })).filter(({ e }) => e.g && e.g.m !== e.g.b && (lostOf(e) > 0.005 || ["meh", "bad"].includes(x.verdict(e.g).k)));
    return bad.sort((a, b) => lostOf(b.e) - lostOf(a.e) || a.e.g.f - b.e.g.f || b.i - a.i).slice(0, REVIEW_SHOW);
  }
  function row(e, i) {
    const x = X(), v = x.verdict(e.g), on = M.pick === i;
    const li = `<li class="pick${on ? " on" : ""}" data-pick="${i}" tabindex="0" role="button" aria-expanded="${on}" title="Show this decision's verdict"><span class="st">T${e.table} #${e.n} ${e.street}</span>`
      + `<span><span class="hole">${e.hole.map(cardHtml).join("")}</span> ${x.ACT(e.rec)}</span><span class="pill" style="--c:${v.c}">${v.label}</span>`
      + `<span class="num ${x.lostCls(e)}">${e.g && e.g.l != null ? (lostOf(e) > 0.005 ? "−" + e.g.l.toFixed(2) + " bb" : "0 bb") : ""}</span></li>`;
    return li + (on ? `<li class="spot"><div class="muted">Table ${e.table} · ${SEAT_LABEL(e.seat, S.players)} · ${e.board.length ? e.board.map(cardHtml).join(" ") + " · " : ""}${esc(e.line)}</div>${x.feedback(e, { past: true })}</li>` : "");
  }
  function sideHtml() {
    const x = X(), sc = M.score, acc = sc.decisions ? x.pct(sc.good / sc.decisions) : "–", avg = sc.evDecisions ? (sc.evLost / sc.evDecisions).toFixed(2) : "–";
    const nm = mistakes().length;
    let out = `<div class="cx-stats" title="Your multi-table session, all tables together"><span><b class="num">${sc.hands}</b> hands</span>`
      + `<span title="Decisions the solver's strategy includes (it plays them 15% of the time or more)"><b class="num">${acc}</b> accuracy</span>`
      + `<span title="Average EV given up against the solver's best action, per decision that has an EV (after the flop)"><b class="num">${avg}</b> bb lost / decision</span>`
      + `<span><b class="num ${sc.net > 0 ? "pos" : sc.net < 0 ? "neg" : ""}">${sc.hands ? x.money(r2(sc.net)) : "–"}</b> won</span></div>`
      + `<div class="cx-tools">${M.review.length ? `<button class="small rev" data-rev="1" aria-expanded="${M.rev}">${M.rev ? "Hide mistakes" : "Review mistakes"}${nm ? `<span class="n">${nm}</span>` : ""}</button>` : ""}`
      + `<button class="small ghost" data-reset="1" title="Clear the session's score and mistakes and deal new hands at every table">Start over</button></div>`;
    const last = M.lastId && M.review.find(e => e.id === M.lastId);
    if (last) out += x.feedback(last, { label: `Table ${last.table} · last decision`, old: ready(front()) });
    else out += `<div class="cx-fb empty">Each decision is graded here once you act: the verdict, the EV it gave up and the solver's mix, whichever table it was at.</div>`;
    out += `<div class="cx-h">Tables</div><table class="mt-tbls"><tr><th>Table</th><th>Hands</th><th>Accuracy</th><th>Lost / dec.</th><th>Won</th><th>Last hand</th></tr>`
      + M.tables.map(t => { const s = t.stats;
        return `<tr class="${t.id === M.focus ? "cur" : ""}"><td>${t.id + 1}</td><td>${s.hands}</td><td>${s.decisions ? x.pct(s.good / s.decisions) : "–"}</td><td>${s.evDecisions ? (s.evLost / s.evDecisions).toFixed(2) : "–"}</td>`
          + `<td class="num ${s.net > 0 ? "pos" : s.net < 0 ? "neg" : ""}">${s.hands ? x.money(r2(s.net)) : "–"}</td><td class="lst" title="${t.last ? esc(`#${t.last.n} ${t.last.text} ${x.money(t.last.net)}${t.last.says ? ": " + t.last.says : ""}`) : ""}">${t.last ? `${esc(t.last.text)} ${x.money(t.last.net)}` : ""}</td></tr>`; }).join("") + `</table>`;
    if (M.rev) {
      const xs = mistakes();
      out += xs.length ? `<div class="cx-h">Your biggest mistakes · all tables · last ${REVIEW_KEEP} decisions</div><ul class="cx-list cx-rev">${xs.map(({ e, i }) => row(e, i)).join("")}</ul>`
        : `<div class="cx-empty cx-rev">No mistakes in your last ${REVIEW_KEEP} decisions so far.</div>`;
    } else {
      const xs = M.review.map((e, i) => ({ e, i })).slice(-RECENT).reverse();
      out += `<div class="cx-h">Recent decisions · all tables</div>` + (xs.length ? `<ul class="cx-list">${xs.map(({ e, i }) => row(e, i)).join("")}</ul>` : `<div class="cx-empty">Your decisions at every table are listed here as you play.</div>`);
    }
    return out;
  }
  function render() {
    if (!M.on) return;
    const box = view(), states = M.tables.map(stateOf);
    part("mtTop", topHtml());
    part("mtTabs", tabsHtml(states));
    M.tables.forEach((t, i) => part("mtT" + t.id, tableHtml(t, states[i]), `mt-t ${states[i].cls}${t.id === M.focus ? " cur" : ""}`));
    const amt = document.getElementById("mtAmt"), kept = amt && amt.value, had = amt && document.activeElement === amt;
    part("mtDock", dockHtml());
    const amt2 = M.custom && document.getElementById("mtAmt");
    if (amt2 && amt2 !== amt) { if (kept) amt2.value = kept; if (had || M.focusAmt) { amt2.focus && amt2.focus(); M.focusAmt = false; } }
    part("mtSide", sideHtml());
    box.hidden = false;
  }
  // clicks and keys on the view (one listener each, the parts' HTML redrawn under them)
  function wire(box) {
    box.addEventListener("click", (e) => {
      const el = e.target && e.target.closest && e.target.closest("[data-act],[data-apply],[data-tables],[data-drill],[data-seat],[data-retry],[data-rev],[data-reset],[data-pick],[data-ai],[data-t]");
      if (!el || el.disabled) return;
      click(el.dataset, el);
    });
    box.addEventListener("keydown", (e) => {
      if (e.target && e.target.id === "mtAmt") { if (e.key === "Enter") { e.preventDefault(); applyCustom(); } else if (e.key === "Escape") { M.custom = false; render(); } return; }
      const li = e.target && e.target.closest && e.target.closest("[data-pick]");
      if (li && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); click(li.dataset, li); }
    });
  }
  function click(d, el) {
    if (d.act != null) {
      const a = M.dock && M.dock.t === M.focus && M.dock.acts[+d.act];
      if (!a) return;
      if (a.custom) { M.custom = !M.custom; M.focusAmt = M.custom; return render(); }
      return heroAct(a.a);
    }
    if (d.apply) return applyCustom();
    if (d.tables) return window.Practice.setTables(+d.tables);
    if (d.drill) { if (d.drill !== M.drill) { M.drill = d.drill; store(DRILL, M.drill); if (M.drill === "lib") loadLib(); redealAll(); } return; }
    if (d.seat) { if (d.seat !== M.seat) { M.seat = d.seat; store(SEAT, M.seat); redealAll(); } return; }
    if (d.retry != null) { const t = M.tables[+d.retry]; if (t) { t.err = ""; t.tries = 0; } return pump(); }
    if (d.rev) { M.rev = !M.rev; M.pick = null; return render(); }
    if (d.reset) return startOver();
    if (d.pick != null) { M.pick = M.pick === +d.pick ? null : +d.pick; return render(); }
    if (d.ai) { const f = el.closest && el.closest(".cx-ai"), i = f && f.querySelector("input.cx-key"); return X().aiClick(d.ai, d.k, i ? i.value : undefined); }
    if (d.t != null) return pick(+d.t);
  }
  // a table picked by a click or a tab: in front until you act there (or pick another)
  function pick(i) {
    if (!M.tables[i] || i === M.focus) return;
    M.focus = i; M.pin = true; M.custom = false; M.err = ""; M.focusAt = -1e9;
    render();
  }
  function applyCustom() {
    const el = document.getElementById("mtAmt"), v = parseFloat(el && el.value);
    return heroAct({ action: "bet", to: v });
  }
  // S: the next table waiting (the one in front goes to the back of the queue); N: finished tables deal now
  function key(e) {
    const k = e.key, low = String(k).toLowerCase();
    if (k === "Escape") { if (M.custom) { M.custom = false; render(); } return; }
    if (low === "s") { e.preventDefault && e.preventDefault(); return skip(); }
    if (low === "n") { for (const t of M.tables) if (t.hand && t.hand.result) t.nextAt = 0; return pump(); }
    const t = front();
    if (!ready(t) || !M.dock || M.dock.t !== t.id) return;
    const want = low === "x" ? "c" : low, a = M.dock.acts.find(x => x.key.toLowerCase() === want);
    if (!a) return;
    e.preventDefault && e.preventDefault();
    if (a.custom) { M.custom = !M.custom; M.focusAmt = M.custom; return render(); }
    return heroAct(a.a);
  }
  function skip() {
    const f = front(), q = queue().filter(t => t !== f);
    if (!q.length) return;
    if (ready(f)) f.since = Math.max(now(), ...q.map(t => t.since)) + 1e-3;   // (to the back of the queue)
    M.focus = q[0].id; M.pin = false; M.custom = false; M.err = ""; M.focusAt = -1e9;
    render();
  }
  function redealAll() {
    for (const t of M.tables) { t.hand = null; t.st = null; t.err = ""; t.tries = 0; }
    pump();
  }
  function startOver() {
    if (typeof confirm === "function" && !confirm("Start the multi-table session over? Its score, mistakes and every table's results are cleared, and every table deals a new hand.")) return false;
    M.score = { ...ZERO }; M.review = []; M.lastId = null; M.rev = false; M.pick = null; M.spare = [];
    for (const t of M.tables) { t.stats = { ...ZERO }; t.last = null; }
    redealAll(); persist(true);
    return true;
  }

  // ---- on and off: Practice's Tables (2-4) starts it, 1 or Exit practice stops it; the tables' hands wait meanwhile
  function start(n) {
    n = Math.max(2, Math.min(4, Math.floor(+n) || 2));
    const first = !M.tables.length;
    M.n = n;
    if (first) { M.spare = Array.isArray(saved.tables) ? saved.tables.slice(0, 4) : []; if (saved.focus != null) M.focus = +saved.focus || 0; }
    // fewer tables: the others set aside, their hand and results as they were, until there are as many again
    for (const t of M.tables.slice(n)) M.spare[t.id] = { st: t.st, hand: t.hand, stats: t.stats, last: t.last };
    M.tables = M.tables.slice(0, n);
    while (M.tables.length < n) { const i = M.tables.length; M.tables.push(newTable(i, M.spare[i])); M.spare[i] = null; }
    if (M.focus >= n) M.focus = 0;
    M.on = true;
    document.body.classList.add("mt");
    if (drillNow() === "lib" || M.drill === "lib") loadLib();
    pump();
  }
  function stop() {
    if (!M.on) return;
    M.on = false; clearTimeout(M.timer); M.timer = null;
    stopFlight();
    persist(true);
    document.body.classList.remove("mt");
    const box = document.getElementById("mtView"); if (box) box.hidden = true;
  }
  function tick() { if (M.on) pump(); }
  // how the waits went: per decision, the seconds from your turn to its buttons (the spot's answer in hand), and the
  // session's time split: some table waiting for you, none but a solve under way, between hands
  function metrics() {
    const w = M.met.waits.map(x => x.s).sort((a, b) => a - b), q = (p) => w.length ? r2(w[Math.min(w.length - 1, Math.floor(p * w.length))]) : null;
    const tot = M.met.you + M.met.solve + M.met.pause;
    return { decisions: w.length, instant: w.filter(s => s < 0.05).length, median: q(0.5), p90: q(0.9), max: w.length ? r2(w[w.length - 1]) : null,
      youMs: Math.round(M.met.you), solveMs: Math.round(M.met.solve), pauseMs: Math.round(M.met.pause), solveShare: tot ? r2(M.met.solve / tot) : null,
      stops: M.met.stops, solves: M.met.solves.length, charted: M.met.charted, charts: M.charts.size, waits: M.met.waits.slice(-200) };
  }

  window.MultiTable = { get on() { return M.on; }, start, stop, tick, key, startOver, pump, act: heroAct, pick, skip, metrics,
    // (tests)
    state: M, OPT, queue, ready, flopName, setLib, step, estimate, group, aheads, render, click };
})();
