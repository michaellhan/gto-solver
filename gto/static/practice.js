// Practice mode, loaded by the page's Practice button on first use: hands against GTO opponents.
// It drives the page's own engine: the other seats' hands and the board are dealt up front from one deck, and
// each opponent's move is drawn from the GTO mix for its actual hand (Advisor.play: the preflop chart, then the
// postflop solve, through the worker or POST /play). Your decisions are graded as in a hand review; the advice
// stays hidden on your turn and shows once you've acted. Your seat (Your seat): Table (the default: a real ring
// game, you at one seat, the button moving one seat each hand and every seat's chips carried over), Random, or
// one position every hand. While you decide, the lines you are likeliest to take are asked for ahead (see ahead()).
// The running score, the table and your last decisions (Review mistakes) are kept in this browser; Start over clears them.
// Tables (1-4): above one, multitable.js (loaded the first time) plays that many tables at once in its own view.
"use strict";
(() => {
  const ON = "gto-practice", HABITS = "gto-practice-habits", STATE = "gto-practice-state", HAND = "gto-practice-hand", SCORE = "gto-practice-score", SEAT = "gto-practice-seat", REVIEW = "gto-practice-review", DRILL = "gto-practice-drill", HINT = "gto-practice-hint", TABLE = "gto-practice-table", TABLES = "gto-practice-tables";
  // (kept through the page's storage, store.js, which keeps each key in localStorage too; localStorage itself without it.
  // The API key below stays in localStorage alone: never a record, a backup or a sync)
  const KV = globalThis.GTOStore || null;
  const read = (k) => { try { return JSON.parse(KV ? KV.getRaw(k) : localStorage.getItem(k)); } catch (e) { return null; } };
  const store = (k, v) => { try { if (KV) KV.set(k, v); else localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} };
  // (explain.js, which the page loads first: a hand's strength and a decision's words, shared with a finished hand's review)
  const X = window.Explain, { rank7, cmp, handName, CATS, mixOf, verdict, pct, evTxt, why, heroRead, spot, keyOf, md } = X;
  const ZERO = { hands: 0, decisions: 0, best: 0, good: 0, evDecisions: 0, evLost: 0, net: 0 };
  // hand: { hero, players, hole, hands: {seat: [c, c]}, board: [5 cards], result }
  // seat: "table", "random" or a position; table: the ring game (see deal()); more: the verdict's details open; detail: your range shown (see range);
  // review: your last decisions (REVIEW_KEEP), for Review mistakes; rev: that list open; pick: the one shown in it;
  // drill: the kind of pot dealt (DRILLS); tries: deals so far looking for it; hint: the first-run hint dismissed
  const P = { on: false, hand: read(HAND), score: { ...ZERO, ...(read(SCORE) || {}) }, habits: read(HABITS) || {}, busy: false, status: "", error: "",
    seat: read(SEAT) || "table", table: read(TABLE), more: !!read("gto-practice-more"), detail: !!read("gto-practice-detail"),
    review: Array.isArray(read(REVIEW)) ? read(REVIEW) : [], rev: false, pick: null,
    drill: read(DRILL) || "any", tries: 0, hint: !!read(HINT), range: null, tables: [2, 3, 4].includes(read(TABLES)) ? read(TABLES) : 1 };
  // tables: how many tables at once (above 1: multitable.js's view, mt() while it's on)
  // range: { id, res, label } for your last decision, kept in memory while its solve is the page's advice (lastResult):
  // how the solver plays your whole range there (the advice panel's grid, nothing else)
  P.running = !!(P.hand && P.hand.run && !P.hand.run.done);   // (a run-out after your fold, resumed after a reload)
  const REVIEW_KEEP = 60, REVIEW_SHOW = 6;
  if (P.score.decisions && !(read(SCORE) || {}).good) P.score.good = P.score.best;   // (a score from before "good" was kept)

  // ---- the pot to the best hands: chips in by seat (inv), the players still in (live), those all-in (allin: each
  // wins at most its stake from each player); each side pot goes to the best hand among the players in it, split
  // evenly on a tie, all of them short the rake (pots that saw a flop: raked) in proportion; a bet nobody matched
  // goes back; a tournament's pots (payouts or bounties set, or an imported tournament hand's) aren't raked, whatever
  // the Rake setting, as the page asks the solver. Returns { pays: {seat: chips won}, pots: [{amount, eligible, winners}],
  // winners, val: {seat: rank7} }.
  function settle(inv, live, allin, board, cards, raked) {
    let pot = POS.reduce((t, p) => t + (inv[p] || 0), 0);
    const most = POS.map(p => inv[p] || 0).sort((a, b) => b - a), uncalled = most[0] - (most[1] || 0);
    if (raked && RAKE() && !TOURNEY() && !S.tournament) pot -= Math.min((pot - uncalled) * RAKE() / 100, RAKE_CAP() || Infinity);
    const r2 = (x) => Math.round(x * 100) / 100, pays = {};
    if (live.length < 2) { if (live[0]) pays[live[0]] = r2(pot); return { pays, pots: [], winners: live, val: {} }; }
    const val = Object.fromEntries(live.map(p => [p, rank7([...cards(p), ...board])]));
    const top = (ps) => { const b = ps.reduce((b, p) => cmp(val[p], val[b]) > 0 ? p : b); return ps.filter(p => cmp(val[p], val[b]) === 0); };
    const side = sidePots(Object.fromEntries(POS.map(p => [p, inv[p] || 0])), live, allin);
    const total = side.reduce((t, x) => t + x.amount, 0), keep = total > 0 ? pot / total : 1;
    const pots = side.map(x => { const w = top(x.eligible); for (const p of w) pays[p] = (pays[p] || 0) + x.amount * keep / w.length;
      return { amount: r2(x.amount * keep), eligible: x.eligible, winners: w }; });
    for (const p in pays) pays[p] = r2(pays[p]);
    return { pays, pots, winners: top(live), val };
  }
  // a showdown in a line: "CO's Q-high flush beats your two pair, kings and sevens"
  function says(st) {
    const who = (p) => p === S.hero ? "your" : `${SEAT_LABEL(p, S.players)}'s`, w = st.winners, v = st.val;
    if (!w.length || !v[w[0]]) return "";
    if (w.length > 1) return `Split pot: ${names(w)} tie with ${handName(v[w[0]])}`;
    const rest = Object.keys(v).filter(p => !w.includes(p)), lo = rest.reduce((b, p) => b && cmp(v[b], v[p]) >= 0 ? b : p, null);
    const top = handName(v[w[0]], v[lo]);
    return lo ? `${who(w[0])} ${top} ${/^(three|four) /.test(top) ? "beat" : "beats"} ${who(lo)} ${handName(v[lo], v[w[0]])}`.replace(/^./, (c) => c.toUpperCase()) : "";
  }

  // ---- the table (Your seat: Table): a ring game. You keep seat 0; the button moves one seat clockwise each hand
  // played out, so your position goes BTN, CO, HJ … UTG, BB, SB, BTN; each seat's chips carry over from hand to
  // hand, and a seat under a quarter of the starting stack (S.stack) rebuys to it. P.table: { n, start, btn (the
  // button's seat), seats: [chips by seat], seq (hands dealt) }; each hand keeps where it started (hand.tbl).
  const REBUY = 0.25;
  const tableMode = () => P.seat === "table" && P.drill === "any";
  // seat k's position with the button at seat b, n-handed (the seats run clockwise, as SEATS[n] and the action)
  const posAt = (k, b, n) => { const s = SEATS[n], i = s.indexOf(n === 2 ? "SB" : "BTN"); return s[(((i + k - b) % n) + n) % n]; };
  const seatOf = (p, b, n) => { for (let k = 0; k < n; k++) if (posAt(k, b, n) === p) return k; return -1; };
  // the hand's chips into the table: each seat's stack at the start of the hand, plus what it won, less what it put in
  function toTable(h, pays, inv) {
    const T = P.table, t = h.tbl;
    if (!t || !T || T.seq !== t.seq) return;   // (not the table's latest hand)
    T.seats = t.stacks.map((x, k) => { const p = posAt(k, t.btn, T.n); return Math.max(0, Math.round((x + (pays[p] || 0) - (inv[p] || 0)) * 100) / 100); });
    store(TABLE, T);
  }
  // the table for the next hand: the button moved on if the last hand was played out (a stack typed in the setup
  // meanwhile taken over), short stacks topped up. Returns the hand's start: { seq, btn, stacks, rebuys }
  function seatUp() {
    const n = S.players, h = P.hand;
    let T = P.table;
    if (!T || T.n !== n || T.start !== S.stack || !Array.isArray(T.seats) || T.seats.length !== n)
      T = P.table = { n, start: S.stack, btn: Math.floor(Math.random() * n), seats: Array(n).fill(S.stack), seq: 0 };
    else if (h && h.tbl && h.tbl.seq === T.seq) {
      if (h.result) T.btn = (T.btn + 1) % n;
      else POS.forEach(p => { const k = seatOf(p, h.tbl.btn, n); if (Math.abs(STACK(p) - h.tbl.stacks[k]) > 1e-9) T.seats[k] = STACK(p); });
    }
    const rebuys = [];
    T.seats = T.seats.map((x, k) => { if (x < T.start * REBUY - 1e-9) { rebuys.push(posAt(k, T.btn, n)); return T.start; } return x; });
    T.seq++; store(TABLE, T);
    return { seq: T.seq, btn: T.btn, stacks: T.seats.slice(), rebuys };
  }
  // Reset stacks: every seat back to the starting stack, from a new hand
  function rebuyAll() { if (P.table) { P.table.seats = P.table.seats.map(() => P.table.start); store(TABLE, P.table); } next(); }
  // Start over (the score's button, or Reset practice in the ⋯ menu): the score, your last decisions, what you
  // usually do (habits) and the table (every seat's chips, the button) cleared, from a new hand; your seat, the drill
  // and the setup stay. Asked first.
  function startOver() {
    if (mt()) return window.MultiTable.startOver();   // (several tables: their session's own Start over)
    if (typeof confirm === "function" && !confirm("Start practice over? Your score, mistakes and the table's stacks are cleared, and a new hand is dealt.")) return false;
    P.score = { ...ZERO }; P.review = []; P.habits = {}; P.table = null; P.rev = false; P.pick = null;
    store(SCORE, P.score); store(REVIEW, P.review); store(HABITS, P.habits); store(TABLE, null);
    next();
    return true;
  }

  // ---- dealing
  // the table (stacks, ante, straddle, rake; at the table the starting stack too, though every seat has its own):
  // a change in the setup deals a new hand rather than alter this one
  const chips = () => JSON.stringify([POS.map(STACK), ANTE(), STRADDLER(), RAKE(), RAKE_CAP(), ...(tableMode() ? [S.stack] : [])]);
  function deal() {
    const deck = []; for (const r of RANKS) for (const s of SUITS) deck.push(r + s);
    for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
    const tbl = tableMode() ? seatUp() : null, pool = seatsFor(P.drill), hands = {};
    const hero = tbl ? posAt(0, tbl.btn, S.players) : pool.includes(P.seat) ? P.seat : pool[Math.floor(Math.random() * pool.length)];
    for (const p of POS) hands[p] = [deck.pop(), deck.pop()];
    // (as New hand: seats keep their stacks; at the table, each its own chips)
    const stacks = tbl ? cleanStacks({ stack: S.stack, stacks: Object.fromEntries(POS.map(p => [p, tbl.stacks[seatOf(p, tbl.btn, S.players)]])) }, POS) : cleanStacks(S, POS);
    S = { ...fresh(S.players), hero, stack: S.stack, ante: S.ante, straddle: S.straddle, rake: S.rake, rakeCap: S.rakeCap, payouts: S.payouts, fieldN: S.fieldN, fieldAvg: S.fieldAvg, ...cleanBounty(S, POS), ...cleanAdv(S, POS), ...stacks };
    S.cards[0] = hands[hero][0]; S.cards[1] = hands[hero][1];
    const hole = hands[hero]; delete hands[hero];
    P.hand = { hero, players: S.players, hole, hands, board: deck.slice(0, 5), result: null, log: [], chips: chips(), ...(tbl ? { tbl } : {}) };
    P.error = ""; P.running = false; store(HAND, P.hand); save(); resetAdvice();
    if (!drilling()) sfx("deal");   // (a drill's deals flash by until its spot comes up: its flop is heard instead)
    A.moves.clear(); A.key = null; A.gen++;
  }
  // the hand on the page is the one dealt here (not changed by New hand, the setup or a share link)
  const ours = () => { const h = P.hand; return !!h && h.hero === S.hero && h.players === S.players && S.cards[0] === h.hole[0] && S.cards[1] === h.hole[1]
    && POS.every(p => p === h.hero || h.hands[p]) && S.cards.slice(2).every((c, i) => !c || c === h.board[i]) && (!h.chips || h.chips === chips()); };

  // ---- drills: only one kind of pot. The preflop is played for you too (from the chart, as the opponents'), and
  // the hand dealt again until it reaches the flop heads-up, you in it, as that kind of pot; you play from the flop.
  const DRILLS = { any: "Any hand", srp: "Raised pot", "3bet": "3-bet pot", bvb: "Blind vs blind" };
  const DRILL_TIP = { any: "Every hand from the start", srp: "Heads-up after one preflop raise and a call: preflop is played for you, you play from the flop",
    "3bet": "Heads-up after a 3-bet and a call: preflop is played for you, you play from the flop", bvb: "Small blind against big blind, folded to them: preflop is played for you, you play from the flop" };
  const DRILL_TRIES = 400;
  const seatsFor = (d) => d === "bvb" ? POS.filter(p => p === "SB" || p === "BB") : POS;
  const drilling = () => P.on && P.drill !== "any";
  // your preflop is played for you (the drill's preflop, until the flop)
  const auto = (sim) => drilling() && !!sim.cur && sim.current === "preflop" && sim.toAct === S.hero;
  // while the drill's preflop is played (the deals flash by): what the coach and the dock say, else null
  const seeking = () => drilling() && P.hand && !P.hand.result && !S.cards[2] ? `Dealing a ${DRILLS[P.drill].toLowerCase()}…` : null;
  // the drill's preflop has gone another way (dealt again), else false (still on course, or on the flop)
  function offDrill(sim) {
    if (!drilling() || S.cards[2]) return false;
    const pre = S.actions.preflop, raises = pre.filter(a => a.action === "raise" || a.action === "bet").length;
    const want = { srp: 1, "3bet": 2 }[P.drill], blinds = seatsFor("bvb");
    if (!sim.active.has(S.hero) || pre.some(a => a.action === "all-in") || (want && raises > want)) return true;
    if (P.drill === "bvb" && (!blinds.includes(S.hero) || pre.some(a => !blinds.includes(a.pos) && a.action !== "fold"))) return true;
    if (sim.handOver) return true;
    return sim.waiting === "flop" && (sim.active.size !== 2 || (want != null && raises !== want));
  }
  function dealStreet() {
    const sim = simulate();
    for (let i = 2; i < 2 + NEED[sim.waiting]; i++) if (!S.cards[i]) S.cards[i] = P.hand.board[i - 2];
    save(); sfx("card");
  }

  // ---- an opponent's move: the GTO mix for its hand, drawn by the solver
  async function villain() {
    await modeReady;
    const sim = simulate(), p = sim.toAct, before = JSON.stringify([S.actions, S.cards]), mine = p === S.hero;
    P.status = seeking() || `${SEAT_LABEL(p, S.players)} is thinking…`; bar(sim);
    const req = P.running ? { ...payload(), hero: p, hole: cards(p).join("") } : payload();   // (a run-out: from the seat's own view)
    let res;
    try { res = (!mine && await fromAhead(JSON.stringify(req) + "|" + p)) || await callAdvise(req, { seat: p, hole: (mine ? P.hand.hole : P.hand.hands[p]).join("") }); }
    catch (e) { res = { error: "Request failed: " + e }; }
    P.status = "";
    if (!P.on || before !== JSON.stringify([S.actions, S.cards])) return;   // the hand moved on while it thought
    if (res.cancelled) return;                                           // asked again on the next render
    if (res.error) { P.error = `${p}'s move failed: ${res.error}`; return; }
    const e = act({ ...move(res), auto: mine });
    if (e) P.error = `${p}'s move (${res.name}) was rejected: ${e}`;
    else if (!seeking()) heard(S.actions[sim.current].at(-1));
  }

  // ---- looking ahead: on your turn, once your spot is solved, the lines you're likeliest to take are played on
  // (the opponents' replies drawn as they will be, the next street dealt), each as far as the next request that takes
  // a solve: your own next spot (kept in the page's cache) or an opponent's move (kept here, used if you do take that
  // line). Moves the solve already holds (heads-up, a reply on your own tree) cost nothing, so the wait goes into the
  // time you take to decide. Your likeliest moves: the solver's mix for your hand, blended half and half with what you
  // usually do in that kind of spot (habits: fold / check or call / bet or raise, facing a bet or not) once you've
  // decided it a few times; up to AHEAD_LINES of them, each at least AHEAD_MIN likely, a fold never (it ends your hand).
  // A look-ahead your move made useless is stopped: by the local server, or by the browser's worker where the page
  // shares a stop flag with it (stopRequests: cross-origin isolated, as the site is under its service worker). A
  // browser that can't stop one looks ahead only at Fast or Quick, where Practice's answers take about 2 s (a wrong
  // guess makes the next request wait that long at most).
  const AHEAD_LINES = 2, AHEAD_MIN = 0.2, HABIT_MIN = 5;
  const A = { key: null, gen: 0, moves: new Map(), out: null };   // out: {k, p} the look-ahead request in flight
  const clone = (x) => JSON.parse(JSON.stringify(x));
  const kindOf = (action) => action === "fold" ? "fold" : action === "check" || action === "call" ? "pass" : "aggr";
  // what you've done, by spot kind: {facing|open: {fold, pass, aggr}} (kept in this browser)
  function habit(facing, action) {
    const h = P.habits[facing ? "facing" : "open"] = P.habits[facing ? "facing" : "open"] || { fold: 0, pass: 0, aggr: 0 };
    h[kindOf(action)]++; store(HABITS, P.habits);
  }
  // your moves at this spot, likeliest first: [{rec, p}] (rec as the page records it)
  function likely(res, sim) {
    const info = sim.cur, facing = info.curBet - info.bets[S.hero] > 1e-9;
    const legal = (res.actions || []).filter(a => a.action !== "fold" && (facing ? a.action !== "check" : a.action !== "call"));
    const h = P.habits[facing ? "facing" : "open"], n = h ? h.fold + h.pass + h.aggr : 0;
    const byKind = {}; for (const a of res.actions || []) byKind[kindOf(a.action)] = (byKind[kindOf(a.action)] || 0) + a.freq;
    return legal.map(a => {
      const k = kindOf(a.action);
      const p = n >= HABIT_MIN ? 0.5 * a.freq + 0.5 * (h[k] / n) * (byKind[k] > 0 ? a.freq / byKind[k] : 0) : a.freq;
      const rec = k === "pass" ? { action: facing ? "call" : "check" } : a.action === "all-in" ? { action: "all-in" }
        : a.amount_bb != null ? { action: facing || info.curBet > 0 ? "raise" : "bet", to: legalTo(a.amount_bb, info) } : null;
      return { rec, p };
    }).filter(c => c.rec && c.p >= AHEAD_MIN).sort((x, y) => y.p - x.p).slice(0, AHEAD_LINES);
  }
  // run f with the page's hand set to st's actions and cards (act() and payload() read the page's S)
  function within(st, f) {
    const keep = [S.actions, S.cards]; S.actions = st.actions; S.cards = st.cards;
    try { return f(); } finally { S.actions = keep[0]; S.cards = keep[1]; save(); }
  }
  function ask(k, req, play) {
    const c = callAdvise(req, play), p = c.catch(e => ({ error: String(e) }));
    A.out = { k, p, id: c.reqId }; p.then(() => { if (A.out && A.out.p === p) A.out = null; });
    return p;
  }
  async function ahead() {
    const key = JSON.stringify(payload());
    if (A.key === key || !lastResult || lastResult.error) return;
    A.key = key; const gen = ++A.gen;
    if (MODE !== "server" && !STOP && SPEED === "accurate") return;
    const sim0 = simulate();
    for (const c of likely(lastResult, sim0)) {
      if (gen !== A.gen) return;
      await line(gen, sim0.current, c.rec);
    }
  }
  // one line: your move, then the hand played on to the next request that takes a solve
  async function line(gen, street, rec) {
    const st = { actions: clone(S.actions), cards: S.cards.slice() };
    st.actions[street].push({ pos: S.hero, ...rec });
    // the hand on the page is still where the line starts, or on it (you haven't gone another way, it hasn't ended)
    const sig = (acts) => STREETS.map(s => acts[s].map(a => `${a.pos} ${a.action} ${a.to || ""}`));
    const onLine = () => { const r = sig(S.actions), h = sig(st.actions);
      return ours() && !simulate().handOver && r.every((x, i) => x.every((y, j) => y === h[i][j])); };
    for (let n = 0; n < 16; n++) {
      if (gen !== A.gen || !P.on || !onLine()) return;
      const step = within(st, () => {
        const sim = simulate();
        if (sim.handOver || (!sim.cur && !sim.waiting)) return null;
        if (sim.waiting) { for (let i = 2; i < 2 + NEED[sim.waiting]; i++) if (!st.cards[i]) st.cards[i] = P.hand.board[i - 2]; return "dealt"; }
        return { seat: sim.toAct, req: payload() };
      });
      if (!step) return;
      if (step === "dealt") continue;
      const k = JSON.stringify(step.req);
      if (step.seat === S.hero) {   // your next spot: into the page's cache, where analyze() finds it
        if (cache.has(k)) return;
        const res = await ask(k, step.req);
        if (!res.error && !res.cancelled && !cache.has(k)) cache.set(k, res);
        return;
      }
      const mk = k + "|" + step.seat;
      let res = A.moves.get(mk);
      if (!res) {
        res = await ask(mk, step.req, { seat: step.seat, hole: P.hand.hands[step.seat].join("") });
        if (res.error || res.cancelled) return;
        if (!A.moves.has(mk)) A.moves.set(mk, res);
        res = A.moves.get(mk);
      }
      if (gen !== A.gen || !onLine()) return;
      if (within(st, () => act(move(res)))) return;   // (refused: stop here)
    }
  }
  // Before a real request (an opponent's move, your spot's advice): the look-ahead's answer when it asked the same,
  // else null once a look-ahead in flight is out of the way (the local server's solve stopped).
  async function fromAhead(k) {
    if (A.moves.has(k)) { const r = A.moves.get(k); A.moves.delete(k); return r; }
    const o = A.out;
    if (!o) return null;
    if (o.k === k) { const r = await o.p; if (r.error || r.cancelled) return null; A.moves.delete(k); return r; }
    A.gen++;   // you went another way: the look-ahead stops (a local server's solve now, the browser's where it can stop
    // one; else it runs out)
    if (MODE === "server") { try { await fetch("cancel", { method: "POST" }); } catch (e) {} await o.p; }
    else if (o.id != null && stopRequests(o.id)) await o.p;
    return null;
  }
  // an action's sound: a knock for a check, chips for a call, more for a bet or raise, a stack for an all-in
  const heard = (a) => sfx({ raise: "bet", "all-in": "allin" }[a.action] || a.action);
  // an opponent's move as the page records it (a size under the minimum raise is lifted to it, rather than refused)
  const move = (res) => res.action === "bet" || res.action === "raise" ? { action: "bet", to: res.to, clamp: true } : { action: res.action };

  // ---- the end of the hand: showdown, your result and the running score
  function finish(sim) {
    const h = P.hand;
    if (h.result) return;
    const inv = { ...sim.invested };   // (you folded mid-street: that street's bets are in it too)
    const board = boardCards(), live = POS.filter(p => sim.active.has(p)), folded = !sim.active.has(S.hero);
    // your fold with others still in: unless you run it out, the others' chips go as if they checked it down from
    // there (each wins at most what it put in from each player, on the board dealt up front)
    const st = folded && live.length > 1 ? settle(inv, live, new Set(live), h.board, cards, true) : settle(inv, live, sim.allin, board, cards, board.length >= 3);
    const showdown = !folded && live.length > 1;
    const net = Math.round(((folded ? 0 : st.pays[S.hero] || 0) - inv[S.hero]) * 100) / 100;
    h.result = { net, winners: folded ? [] : st.winners, shown: showdown ? live.filter(p => p !== S.hero) : [], best: showdown ? CATS[st.val[st.winners[0]][0]] : null,
      folded, pays: folded && live.length > 1 ? {} : st.pays, ...(showdown ? { says: says(st) } : {}), ...(showdown && st.pots.length > 1 ? { pots: st.pots } : {}) };
    toTable(h, st.pays, inv);
    const mine = []; for (const s of STREETS) for (const a of S.actions[s]) if (a.pos === S.hero && a.g) mine.push(a.g);
    const sc = P.score, ev = mine.filter(g => g.l != null);
    sc.hands++; sc.decisions += mine.length; sc.best += mine.filter(g => g.m === g.b).length; sc.good += mine.filter(g => gradeClass(g) === "ok").length;
    sc.evDecisions += ev.length; sc.evLost += ev.reduce((t, g) => t + g.l, 0); sc.net += net;
    store(HAND, h); store(SCORE, sc);
    if (!folded && net !== 0) sfx(net > 0 ? "win" : "lose", 0.25);   // (after the last action's sound)
    if (!P.hint) { P.hint = true; store(HINT, 1); }   // (the first-run hint: once a hand is played, it's known)
  }
  const cards = (p) => p === S.hero ? P.hand.hole : P.hand.hands[p];

  // ---- after your fold: Show hands (everyone's cards, the rest of the board) or, with others still in, Run it out
  // (the opponents play the hand on by their strategies, to its real end, and the table's chips go as it did).
  // Your result and the score stay as they were at your fold.
  const runLabel = () => { const h = P.hand, r = h && h.result; if (!P.on || !r || !r.folded || h.run || !ours()) return null;
    return simulate().active.size > 1 ? "Run it out" : "Show hands"; };
  function runOut() {
    const h = P.hand, label = runLabel();
    if (!label) return;
    h.run = { done: label === "Show hands" };
    if (h.run.done) { for (let i = 2; i < 7; i++) S.cards[i] = h.board[i - 2]; save(); }
    else P.running = true;
    store(HAND, h); renderAll();
  }
  // the run-out has reached the end of the hand: its showdown, and the table's chips as it went
  function endRun(sim) {
    const h = P.hand, inv = { ...sim.invested }, live = POS.filter(p => sim.active.has(p)), board = boardCards();
    const st = settle(inv, live, sim.allin, board, cards, board.length >= 3);
    P.running = false;
    h.run = { done: true, live, winners: st.winners, pays: st.pays, ...(live.length > 1 ? { says: says(st) } : {}), ...(st.pots.length > 1 ? { pots: st.pots } : {}) };
    toTable(h, st.pays, inv); store(HAND, h);
  }

  // ---- your decisions: each one's verdict and the solver's mix for the spot, kept with the hand
  const ACT = (a) => a.action === "bet" ? `bet ${fmt(a.to)}` : a.action === "raise" ? `raise to ${fmt(a.to)}` : a.action;
  const YOU = (a) => ({ fold: "folded", check: "checked", call: "called", "all-in": "went all-in" })[a.action] || (a.action === "bet" ? `bet ${fmt(a.to)}` : `raised to ${fmt(a.to)}`);
  const evCell = (a) => a.ev === undefined ? `<span class="num"></span>` : a.r ? `<span class="num rough" title="${ROUGH_EV_TIP}">≈ ${evTxt(a.ev)} bb</span>` : `<span class="num">${evTxt(a.ev)} bb</span>`;
  function graded(street, rec, res) {
    if (!P.hand) return null;
    { const c = simulate().cur; if (c) habit(c.curBet - c.bets[S.hero] > 1e-9, rec.action); }   // (before the move: what it faced)
    const mix = mixOf(res);
    const id = Date.now().toString(36) + (++AI.seq).toString(36);   // (the explanation kept in memory; not Math.random, the deal's)
    const e = { id, street, rec: { action: rec.action, ...(rec.to != null ? { to: rec.to } : {}) }, g: rec.g || null, mix, ctx: spot(street) };
    // (the others still in: one or two by name, more as a count)
    const vs = [...simulate().active].filter(p => p !== S.hero).map(p => SEAT_LABEL(p, S.players));
    P.range = res && !res.error && !heuristic(res) && res.grid ? { id, res, label: `Your range · ${street} · ${SEAT_LABEL(S.hero, S.players)}`
      + (vs.length ? ` vs ${vs.length <= 2 ? vs.join(", ") : vs.length + " players"}` : "") } : null;
    (P.hand.log = P.hand.log || []).push(e);
    store(HAND, P.hand);
    // for Review mistakes: the spot (hand number, seat, cards, board, the street's action before you) with it
    const before = S.actions[street].filter(a => a.action !== "fold"), line = before.length
      ? before.map(a => `${SEAT_LABEL(a.pos, S.players)} ${ACT(a)}`).join(", ") : street === "preflop" ? "folded to you" : "first to act";
    P.review.push({ ...e, n: P.score.hands + 1, seat: S.hero, hole: P.hand.hole, board: boardCards(), line });
    P.review = P.review.slice(-REVIEW_KEEP); P.pick = null; store(REVIEW, P.review);
    heard(rec);   // your move, then its verdict: a chime when it's in the solver's mix, a low note for a mistake
    const gc = rec.g && gradeClass(rec.g);
    if (gc === "ok" || gc === "bad") sfx(gc === "ok" ? "good" : "bad", 0.2);
    return e;
  }
  // a decision graded after the fact (a jam deeper than the answer's tree, solved again with all-in in it: index.html
  // jamGrade): its verdict and mix, in the hand's log and Review mistakes, and the score if the hand already counted
  function regraded(e, g, res) {
    if (!e || e.g) return;
    e.g = g; e.mix = mixOf(res) || e.mix;
    const r = P.review.find(x => x.id === e.id); if (r) { r.g = g; r.mix = e.mix; store(REVIEW, P.review); }
    const h = P.hand && P.hand.log && P.hand.log.includes(e) ? P.hand : null;
    if (h) store(HAND, h);
    if (h && h.result) {   // (finish() counted the hand's grades before this one came in)
      const sc = P.score;
      sc.decisions++; sc.best += g.m === g.b ? 1 : 0; sc.good += gradeClass(g) === "ok" ? 1 : 0;
      if (g.l != null) { sc.evDecisions++; sc.evLost += g.l; }
      store(SCORE, sc);
    }
    const gc = gradeClass(g);
    if (gc === "ok" || gc === "bad") sfx(gc === "ok" ? "good" : "bad", 0.2);
    renderAll();
  }
  const ICON = { best: "★", good: "✓", meh: "?!", bad: "✕", none: "–" };
  // your range at decision e, while its solve is still the advice panel's
  const rangeOf = (e) => P.range && e && P.range.id === e.id && P.range.res === lastResult ? P.range : null;
  // the EV a decision gave up: red for an inaccuracy or a mistake; muted for a move in the solver's strategy (its
  // mixed actions are worth the same in theory: a small gap there is the solve's noise more than an error)
  // the EV a decision gave up as text: "≈" when your move is one your range almost never takes (rv: its EV is rough)
  const lostTxt = (g) => `${g.rv ? "≈ " : ""}−${g.l.toFixed(2)} bb`;
  const lostCls = (e) => !(e.g && e.g.l > 0.005) ? "muted" : ["best", "good"].includes(verdict(e.g).k) ? "soft" : "neg";
  // the verdict on one decision: the verdict, what you did and the EV it cost, the solver's mix as a bar with
  // its frequencies; why and the EVs fold away (Details), with how the solver plays your whole range there.
  // o.label: headed "Last decision · street" (the coach's card); o.old: you're on a new decision now (faded);
  // o.past: a decision from the review list (no range: its solve is gone)
  function feedback(e, o = {}) {
    const v = verdict(e.g), g = e.g, past = o.past;
    const cost = g && g.l != null ? (g.l > 0.005 ? `<span class="cost num ${lostCls(e)}" title="EV given up against the solver's best action${g.rv ? ". " + ROUGH_EV_TIP : ""}">${lostTxt(g)}</span>` : `<span class="cost num pos">no EV lost</span>`) : "";
    let mix = "", table = "";
    if (e.mix && e.mix.length && g && g.h != null) {   // the betting rules: their EVs, no mix (0/1)
      const you = (a) => a.n === g.m;
      setColors(e.mix.map(a => a.n));
      table = `<div class="cx-acts"><span class="hd">Action</span><span class="hd num"></span><span class="hd num">EV</span>`
        + e.mix.map(a => `<span class="n${you(a) ? " you" : ""}" style="--c:${COLORS(a.n)}">${a.n}</span><span class="num"></span><span class="num">${a.ev !== undefined ? evTxt(a.ev) + " bb" : ""}</span>`).join("") + `</div>`;
    } else if (e.mix && e.mix.length) {
      const you = (a) => g && a.n === g.m, leg = g && !e.mix.some(you) ? [...e.mix, { n: g.m, f: 0 }] : e.mix;   // (your move, also when the solve hasn't it)
      setColors(leg.map(a => a.n));
      const ev = e.mix.some(a => a.ev !== undefined);
      mix = `<div class="cx-mix">${e.mix.filter(a => a.f >= 0.0005).map(a => `<div style="width:${(a.f * 100).toFixed(1)}%;background:${COLORS(a.n)}" title="${a.n} ${pct(a.f)}">${a.f >= 0.2 ? `<span>${a.n}${/%$/.test(a.n) ? " · " : " "}${pct(a.f)}</span>` : ""}</div>`).join("")}</div>`
        + `<div class="cx-leg">${leg.map(a => `<span class="${you(a) ? "you" : a.f < 0.005 ? "muted" : ""}" style="--c:${COLORS(a.n)}">${a.n} <b class="num">${pct(a.f)}</b>${you(a) ? '<span class="tag">you</span>' : ""}</span>`).join("")}</div>`;
      if (ev) table = `<div class="cx-acts"><span class="hd">Action</span><span class="hd num">How often</span><span class="hd num">EV</span>`
        + e.mix.map(a => `<span class="n${you(a) ? " you" : ""}" style="--c:${COLORS(a.n)}">${a.n}</span><span class="num${you(a) ? " you" : ""}">${pct(a.f)}</span>${evCell(a)}`).join("") + `</div>`;
    }
    const lbl = o.label ? `<div class="lbl">${o.label} · ${e.street}</div>` : "";
    return `<div class="cx-fb${o.old ? " old" : ""}" style="--c:${v.c}">${lbl}<div class="head"><span class="ico" aria-hidden="true">${ICON[v.k]}</span><span class="verdict">${v.label}</span><span class="what">${o.label ? "" : e.street + " · "}you <b>${YOU(e.rec)}</b></span>${cost}</div>${mix}`
      + `<details class="cx-more"${P.more ? " open" : ""}><summary>Details</summary><div class="why">${why(e)}${g && g.r ? ' <span class="muted">(rough solve: a hint, not a verdict)</span>' : ""}${g && g.x ? ' <span class="muted">(off the solved line: earlier actions the solve rarely takes left its ranges, so this grades the best answer to the ranges the hand leaves)</span>' : ""}</div>${table}${aiHtml(e)}`
      + (!past && rangeOf(e) ? `<button class="cx-rangeb" id="pxDetail" aria-expanded="${P.detail}">${P.detail ? "Hide your range" : "See how the solver plays your whole range"}</button>` : "") + `</details></div>`;
  }


  // ---- explaining a decision (explain.js): Practice's row of buttons, drawn on the coach's cards
  // (a decision kept from before the spot was: what the review list has, else this hand's)
  const prompt = (e) => X.prompt(e, P.hand || {});
  // re-render the coach (several tables: their view's side)
  const coach = () => { if (mt()) window.MultiTable.render(); else if (P.on) bar(simulate()); };
  const panel = X.panel({ promptOf: prompt, show: coach }), AI = panel.state, aiHtml = panel.html, aiClick = panel.click, explain = panel.explain;

  // ---- the coach panel: what's happening now, the verdict on your last decision, the result, the score
  const money = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x)) + " bb";
  const names = (ps) => ps.map(p => p === S.hero ? "you" : SEAT_LABEL(p, S.players)).join(", ").replace(/, ([^,]*)$/, " and $1");
  const big = (cs, sm) => cs.map(c => `<div class="bc ${c[1]}${sm ? " sm" : ""}">${c[0]}<span>${SUIT_SYM[c[1]]}</span></div>`).join("");
  // with side pots, which of the pots you were in you won
  function headline(r) {
    if (!r.pots) return r.winners.includes(S.hero) ? (r.winners.length > 1 ? "Split pot" : "You win") : "You lose";
    const mine = r.pots.filter(x => x.eligible.includes(S.hero)), won = mine.filter(x => x.winners.includes(S.hero));
    return !won.length ? "You lose" : won.length === mine.length ? (won.some(x => x.winners.length > 1) ? "Split pot" : "You win")
      : won.includes(r.pots[0]) ? "You win the main pot" : "You win a side pot";
  }
  // your turn (or, until the spot is solved so your move can be graded, "Solving your spot"), an opponent
  // thinking, dealing; your cards and the board beside it (on a phone the table and the dock say the same: hidden)
  function now(sim) {
    const h = P.hand, seat = SEAT_LABEL(S.hero, S.players), board = boardCards();
    const boardRow = board.length ? `<div class="cx-board">${big(board, true)}</div>` : "";
    // the best hand your cards make with the board so far: "Your hand: two pair, kings and sevens"
    const made = board.length >= 3 && h.hole && h.hole.length === 2 ? `<div class="cx-made">Your hand: <b>${handName(rank7([...h.hole, ...board]))}</b></div>` : "";
    let cls = "", t, sub;
    if (P.error) { cls = " err"; t = "Something went wrong"; sub = `<span class="err">${P.error}</span> <button class="link" id="pxRetry">Retry</button>`; }
    else if (P.status) { t = `<span class="spinner"></span>${P.status}`; sub = `${seat} · ${sim.current || ""}`; }
    else if (sim.toAct === S.hero && sim.cur) {
      const info = sim.cur, toCall = Math.min(info.curBet - info.bets[S.hero], STACK(S.hero) - sim.invested[S.hero] - info.bets[S.hero]);
      const ready = resultKey === JSON.stringify(payload());
      cls = ready ? " mine" : " wait"; t = ready ? "Your turn" : `<span class="spinner"></span>Solving your spot…`;
      sub = `${seat} · ${sim.current} · pot ${fmt(info.pot)} bb${toCall > 1e-9 ? ` · <b>${fmt(toCall)} bb to call</b>` : ""}`;
    } else { t = "Dealing…"; sub = `${seat}${sim.current ? " · " + sim.current : ""}`; }
    return `<div class="cx-now${cls}"><div class="cx-cards">${big(h.hole)}</div><div><div class="t">${t}</div><div class="sub">${sub}</div>${boardRow}${made}</div></div>`;
  }
  function summary(r) {
    const h = P.hand, log = h.log || [];
    const shown = r.shown.map(p => `<span class="${r.winners.includes(p) ? "win" : ""}"><b>${SEAT_LABEL(p, S.players)}</b>${big(h.hands[p], true)}</span>`).join("");
    const mineShown = r.shown.length && !r.folded ? `<span class="${r.winners.includes(S.hero) ? "win" : ""}"><b>You</b>${big(h.hole, true)}</span>` : "";
    let out = `<div class="cx-res ${r.net > 0 ? "won" : r.net < 0 ? "lost" : ""}"><div><span class="big">${r.folded ? "You folded" : headline(r)}</span><span class="net num ${r.net > 0 ? "pos" : r.net < 0 ? "neg" : ""}">${money(r.net)}</span></div>`
      + (r.shown.length ? `<div class="cx-show">${mineShown}${shown}</div>` : "")
      + (r.says ? `<div class="says">${r.says}</div>` : "")
      + (r.best && !r.says ? `<div class="pots">${names(r.winners)} ${r.winners.length > 1 || r.winners[0] === S.hero ? "win" : "wins"} with ${r.best}</div>` : "")
      + (r.pots ? `<div class="pots">${r.pots.map((x, i) => `${i ? "side" : "main"} pot ${fmt(x.amount)}: ${names(x.winners)}`).join(" · ")}</div>` : "")
      + (r.folded ? runPart(h) : "") + `</div>`;
    if (log.length) out += feedback(log[log.length - 1], { label: "Your last decision" });   // (as after any other)
    if (log.length > 1) {
      const lost = log.reduce((t, e) => t + (e.g && e.g.l || 0), 0), good = log.filter(e => ["best", "good"].includes(verdict(e.g).k)).length;
      out += `<div class="cx-h">This hand · ${good} of ${log.length} in the solver's strategy${lost > 0.005 ? ` · <span class="num neg">−${lost.toFixed(2)} bb</span> EV lost` : ""}</div><ul class="cx-list">`
        + log.map(e => { const v = verdict(e.g); return `<li title="${e.g ? gradeTip(e.g) : ""}"><span class="st">${e.street}</span><span>${ACT(e.rec)} <span class="muted">· ${e.g && e.g.h != null ? "rules' best: " + e.g.b : "solver " + (e.g ? pct(e.g.f) : "–")}</span></span><span class="pill" style="--c:${v.c}">${v.label}</span><span class="num ${lostCls(e)}">${e.g && e.g.l != null ? (e.g.l > 0.005 ? lostTxt(e.g) : "0 bb") : ""}</span></li>`; }).join("") + `</ul>`;
    }
    return out;
  }
  // at the table: your seat's chips (and who rebought this hand), Reset stacks
  function tableLine() {
    const h = P.hand, t = h && h.tbl, T = P.table;
    if (!tableMode() || !t || !T || !ours()) return "";
    const mine = h.result && !P.running && T.seq === t.seq ? T.seats[0] : t.stacks[0], re = (t.rebuys || []).map(p => p === S.hero ? "you" : SEAT_LABEL(p, S.players));
    return `<div class="cx-tbl"><span>Your stack <b>${fmt(mine)} bb</b>${re.length ? ` · ${re.join(", ")} rebought to ${fmt(T.start)}` : " · the button moves each hand"}</span>`
      + `<button class="small ghost" id="pxRebuy" title="Every seat back to ${fmt(T.start)} bb (the setup's stack), from a new hand">Reset stacks</button></div>`;
  }
  // after your fold: the run-out (or Show hands) as it went, else its button
  function runPart(h) {
    const run = h.run, label = runLabel();
    if (label) return `<div class="cx-run"><button class="small" id="pxRun2" title="${label === "Run it out" ? "Play the hand on without you and show everyone's cards (your score doesn't change)" : "Show everyone's cards and the rest of the board"}">${label}</button></div>`;
    if (!run) return "";
    if (!run.done) return `<div class="cx-run muted"><span class="spinner"></span>Running it out…</div>`;
    const live = run.live || [];
    const shown = live.map(p => `<span class="${run.winners.includes(p) ? "win" : ""}"><b>${SEAT_LABEL(p, S.players)}</b>${big(h.hands[p], true)}</span>`).join("");
    const w = run.winners || [];
    return `<div class="cx-run"><div class="cx-h">Run out</div>${shown ? `<div class="cx-show">${shown}</div>` : ""}`
      + (run.says ? `<div class="says">${run.says}</div>` : "")
      + (w.length ? `<div class="pots">${names(w)} ${w.length > 1 ? "win" : "wins"} ${fmt(w.reduce((t, p) => t + (run.pays[p] || 0), 0))} bb</div>` : "")
      + (run.pots ? `<div class="pots">${run.pots.map((x, i) => `${i ? "side" : "main"} pot ${fmt(x.amount)}: ${names(x.winners)}`).join(" · ")}</div>` : "") + `</div>`;
  }
  // the score, one line
  function stats() {
    const sc = P.score, acc = sc.decisions ? pct(sc.good / sc.decisions) : "–";
    const avg = sc.evDecisions ? (sc.evLost / sc.evDecisions).toFixed(2) : "–";
    return `<div class="cx-stats" title="Your practice score in this browser"><span><b class="num">${sc.hands}</b> hands</span>`
      + `<span title="Decisions the solver's strategy includes (it plays them 15% of the time or more)"><b class="num">${acc}</b> accuracy</span>`
      + `<span title="Average EV given up against the solver's best action, per decision that has an EV (after the flop)"><b class="num">${avg}</b> bb lost / decision</span>`
      + `<span><b class="num ${sc.net > 0 ? "pos" : sc.net < 0 ? "neg" : ""}">${sc.hands ? money(Math.round(sc.net * 100) / 100) : "–"}</b> won</span>`
      + `</div><div class="cx-tools">`
      + (P.review.length ? `<button class="small rev" id="pxRev" aria-expanded="${P.rev}">${P.rev ? "Hide mistakes" : "Review mistakes"}${mistakes().length ? `<span class="n">${mistakes().length}</span>` : ""}</button>` : "")
      + `<button class="small ghost" id="pxReset" title="Clear your score, mistakes and the table's stacks, and deal a new hand (also Reset practice in the ⋯ menu)">Start over</button></div>`;
  }
  // Review mistakes: of your last REVIEW_KEEP decisions, the REVIEW_SHOW that gave up the most EV (then the
  // rarest in the solver's mix: preflop has no EVs), each opening its verdict
  const lostOf = (e) => e.g && e.g.l != null ? e.g.l : 0;
  function mistakes() {
    const bad = P.review.map((e, i) => ({ e, i })).filter(({ e }) => e.g && e.g.m !== e.g.b && (lostOf(e) > 0.005 || ["meh", "bad"].includes(verdict(e.g).k)));
    return bad.sort((a, b) => lostOf(b.e) - lostOf(a.e) || a.e.g.f - b.e.g.f || b.i - a.i).slice(0, REVIEW_SHOW);
  }
  // a review row: hand number and street, your cards and action, the verdict and the EV it gave up; a click opens its spot
  function reviewRow(e, i) {
    const v = verdict(e.g);
    const row = `<li class="pick${P.pick === i ? " on" : ""}" data-i="${i}" tabindex="0" role="button" aria-expanded="${P.pick === i}" title="Show this decision's verdict"><span class="st">#${e.n} ${e.street}</span>`
      + `<span><span class="hole">${e.hole.map(cardHtml).join("")}</span> ${ACT(e.rec)}</span><span class="pill" style="--c:${v.c}">${v.label}</span>`
      + `<span class="num ${lostCls(e)}">${e.g && e.g.l != null ? (lostOf(e) > 0.005 ? lostTxt(e.g) : "0 bb") : ""}</span></li>`;
    return row + (P.pick === i ? `<li class="spot"><div class="muted">${SEAT_LABEL(e.seat, S.players)} · ${e.board.length ? e.board.map(cardHtml).join(" ") + " · " : ""}${e.line}</div>${feedback(e, { past: true })}</li>` : "");
  }
  // on a wide screen, under the score (the coach's column has the room): your last decisions, newest first
  const RECENT = 8;
  function recent() {
    if (P.rev) return "";
    const xs = P.review.map((e, i) => ({ e, i })).slice(-RECENT).reverse();
    return `<div class="cx-recent"><div class="cx-h">Recent decisions</div>` + (xs.length ? `<ul class="cx-list">${xs.map(({ e, i }) => reviewRow(e, i)).join("")}</ul>`
      : `<div class="cx-empty">Your decisions are listed here as you play, each opening its verdict.</div>`) + `</div>`;
  }
  function reviewList() {
    const xs = mistakes();
    if (!P.rev) return recent();
    if (!xs.length) return `<div class="cx-empty cx-rev">No mistakes in your last ${REVIEW_KEEP} decisions so far.</div>`;
    return `<div class="cx-h">Your biggest mistakes · last ${REVIEW_KEEP} decisions</div><ul class="cx-list cx-rev">${xs.map(({ e, i }) => reviewRow(e, i)).join("")}</ul>`;
  }
  // a row of the site's buttons, one picked (a radio group: arrows move, Enter, Space or a click picks); built once
  // per set of options and only its state changed after, so a re-render keeps its focus. opts: [[value, label,
  // title, class]]; now: the one outlined (this hand's seat, with Random); off: the ones that can't be picked
  function radios(box, opts, cur, pick, now, off = []) {
    const want = opts.map(o => o[0]).join();
    if (box.dataset.opts !== want) {
      box.innerHTML = opts.map(([v, l, t, c]) => `<button type="button" role="radio" class="${c || ""}" data-v="${v}" title="${t}">${l}</button>`).join("");
      box.dataset.opts = want;
    }
    const bs = box.querySelectorAll ? [...box.querySelectorAll("[data-v]")] : [];
    for (const b of bs) { const on = b.dataset.v === cur; b.setAttribute("aria-checked", on); b.tabIndex = on ? 0 : -1; b.classList.toggle("cur", b.dataset.v === now); b.disabled = off.includes(b.dataset.v); }
    box.onclick = (e) => { const b = e.target.closest && e.target.closest("[data-v]"); if (b && b.dataset.v !== cur && !b.disabled) pick(b.dataset.v); };
    box.onkeydown = (e) => {
      const ok = bs.filter(b => !b.disabled), i = ok.indexOf(e.target), d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (i < 0 || !d) return;
      e.preventDefault(); e.stopPropagation(); ok[(i + d + ok.length) % ok.length].focus();
    };
  }
  // above the panel: your seat (Table first, then Random) and the drill (the kind of pot); a drill deals a random
  // seat (a blind-vs-blind drill, a blind) or the one picked
  function pickers() {
    const pool = seatsFor(P.drill), cur = tableMode() ? "table" : pool.includes(P.seat) ? P.seat : "random";
    radios(document.getElementById("pxSeat"), [["table", "Table", P.drill === "any" ? "A ring game: you keep your seat, the button moves one seat each hand and the chips carry over" : "A drill deals a random seat", "rnd"],
      ["random", "Random", P.drill === "bvb" ? "A random blind each hand" : "A random seat each hand", "rnd"],
      ...POS.map(p => [p, SEAT_LABEL(p, POS.length), `Every hand from ${SEAT_LABEL(p, POS.length)}`])], cur,
      (v) => { P.seat = v; store(SEAT, P.seat); next(); }, ours() ? S.hero : null, [...(P.drill === "any" ? [] : ["table"]), ...POS.filter(p => !pool.includes(p))]);
    radios(document.getElementById("pxDrill"), Object.entries(DRILLS).map(([k, l]) => [k, l, DRILL_TIP[k]]), P.drill,
      (v) => { P.drill = v; store(DRILL, v); next(); });
    tablesPicker(document.getElementById("pxTables"));
    const n = document.getElementById("pxHandNo"); n.textContent = `Hand ${P.score.hands + (P.hand && P.hand.result ? 0 : 1)}`;
  }
  // ---- Tables: 1 (this file's own table) or 2-4 at once (multitable.js, loaded the first time: its own view in
  // place of the page's table and coach, every table's hand and the session kept in this browser; this table's hand
  // waits as it was)
  const mt = () => !!(window.MultiTable && window.MultiTable.on);
  const TABLES_TIP = { 1: "One table: the hand on the page", 2: "Two tables at once", 3: "Three tables at once", 4: "Four tables at once" };
  function tablesPicker(box) {
    if (box) radios(box, [1, 2, 3, 4].map(n => [String(n), String(n), TABLES_TIP[n]]), String(P.tables), (v) => setTables(+v));
  }
  const loadMt = () => loadJs("multitable.js?v=" + MT_VERSION, () => window.MultiTable);
  // n tables: above 1 the multi-table view starts (or takes the new count); 1 goes back to this table's hand
  // Going from the one table to several: what it asked the solver and no longer needs (your spot's advice, an
  // opponent's move, a look-ahead) is stopped first, so the tables' first requests don't wait behind it: the local
  // server's solve, or the browser's requests where the page shares a stop flag with the worker (their solves kept).
  // The one table asks again when it's back.
  function quietOne() {
    A.gen++; A.key = null; A.out = null;
    if (MODE === "server") return fetch("cancel", { method: "POST" }).catch(() => {});
    if (typeof stopRequests === "function") stopRequests();
    return Promise.resolve();
  }
  function setTables(n) {
    const was = P.tables;
    P.tables = [2, 3, 4].includes(n) ? n : 1; store(TABLES, P.tables);
    const quiet = P.on && was === 1 && P.tables > 1 && !mt() ? quietOne() : null;
    if (P.tables > 1) return Promise.all([loadMt(), quiet]).then(() => { if (P.on && P.tables > 1) window.MultiTable.start(P.tables); })
      .catch(e => { P.error = "Couldn't load multi-table practice: " + (e && e.message || e); renderAll(); });
    if (mt()) window.MultiTable.stop();
    renderAll();
    return Promise.resolve();
  }
  // a phone: the coach sits under the table, often below the fold, so the action dock carries one line for it: the
  // verdict on your last decision (the result once the hand is over); a tap brings the coach's card into view.
  // Always there while practising, so the dock never changes height.
  function strip(sim) {
    const el = document.getElementById("pxStrip"); if (!el) return;
    const h = P.hand, r = h && h.result, log = (h && h.log) || [], last = log[log.length - 1];
    let html = P.hint ? `<span class="muted fit">Hand ${P.score.hands + (r ? 0 : 1)} · each decision is graded here</span>`
      : `<span class="fit">Your seat is marked <b>you</b> · the others play GTO</span>`;   // (first run)
    if (r) html = `<b>${r.folded ? "You folded" : headline(r)}</b><span class="num ${r.net > 0 ? "pos" : r.net < 0 ? "neg" : ""}">${money(r.net)}</span>`
      + (log.length ? `<span class="muted fit">· ${log.filter(e => ["best", "good"].includes(verdict(e.g).k)).length} of ${log.length} in the solver's strategy</span>` : "");
    else if (last) {
      const v = verdict(last.g), g = last.g, top = last.mix && last.mix.length ? last.mix.reduce((b, a) => a.f > b.f ? a : b) : null;
      html = (sim && sim.cur && sim.toAct === S.hero ? `<span class="muted">Last:</span>` : "") + `<b style="color:${v.c}">${v.label}</b><span class="muted">${last.street} · you ${YOU(last.rec)}</span>`
        + (g && g.l != null && g.l > 0.005 ? `<span class="num neg">${lostTxt(g)}</span>` : "")
        + (top && v.k !== "best" ? `<span class="muted fit">· ${g && g.h != null ? `rules' best: ${top.n}` : `solver: ${top.n} ${pct(top.f)}`}</span>` : "");
    }
    el.innerHTML = html + `<span class="go">›</span>`;
    el.style.setProperty && el.style.setProperty("--c", r ? (r.net > 0 ? "var(--good)" : r.net < 0 ? "var(--bad)" : "var(--line2)") : last ? verdict(last.g).c : "var(--line)");
    el.onclick = () => { const t = document.querySelector && (document.querySelector("#practiceBar .cx-hint, #practiceBar .cx-fb:not(.empty)") || document.getElementById("pCoach")); if (t && t.scrollIntoView) t.scrollIntoView({ behavior: "smooth", block: "start" }); };
  }
  function bar(sim) {
    const el = document.getElementById("practiceBar"); el.hidden = !P.on; if (!P.on) return;
    pickers();
    const h = P.hand, r = h && h.result, log = (h && h.log) || [];
    let body = "";
    const again = !r && sim.cur && sim.toAct === S.hero && !P.status;   // (a new decision of yours: the last one's verdict fades)
    if (h) body = r ? summary(r) : now(sim) + (log.length ? feedback(log[log.length - 1], { label: "Last decision", old: again })
      : `<div class="cx-fb empty">Each decision is graded here once you act: the verdict, the EV it gave up and the solver's mix.</div>`);
    const hint = P.hint ? "" : `<div class="cx-hint"><div>You play the seat marked <b>you</b>; the others play GTO by themselves.`
      + ` Act with the action buttons<span class="hoveronly"> or their keys (F, C, 1‑3, R, A; N for the next hand)</span>.</div><button class="small" id="pxHint">Got it</button></div>`;
    const html = hint + tableLine() + body + stats() + reviewList();
    if (el.pxHtml !== html) {   // (unchanged: kept, with its focus and open Details; a key being typed survives a change)
      const ki = el.querySelector ? el.querySelector("input.cx-key") : null, kv = ki && ki.value, kf = ki && document.activeElement === ki;
      el.innerHTML = html; el.pxHtml = html;
      const kn = (kv || kf) && el.querySelector("input.cx-key"); if (kn) { kn.value = kv || ""; if (kf) kn.focus(); }
    }
    strip(sim);
    const hp = document.getElementById("pHistory"); if (hp && hp.scrollHeight > hp.clientHeight) hp.scrollTop = hp.scrollHeight;
    // how the solver plays your whole range at your last decision, under the coach, when asked for
    const rg = P.detail && rangeOf(log[log.length - 1]);
    document.body.classList.toggle("pxdetail", !!rg);
    const rh = document.getElementById("pxRangeHd"); if (rh && rg) rh.textContent = rg.label;
    document.body.classList.toggle("pxseek", !!seeking() && !P.error);   // (the drill's deals flash by, dimmed)
    const q = (sel) => el.querySelector ? el.querySelector(sel) : null, on = (id, f) => { const b = q("#" + id); if (b) b.onclick = f; };
    on("pxHint", () => { P.hint = true; store(HINT, 1); renderAll(); });
    on("pxRetry", () => { P.error = ""; P.tries = 0; renderAll(); });
    on("pxDetail", () => {
      P.detail = !P.detail; store("gto-practice-detail", P.detail); renderAll();
      const pa = P.detail && document.getElementById("pAdvice"); if (pa && pa.scrollIntoView) pa.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    on("pxReset", startOver);
    on("pxRev", () => { P.rev = !P.rev; P.pick = null; renderAll(); });
    on("pxRebuy", () => { if (typeof confirm === "function" && !confirm("Put every seat back to the starting stack and deal a new hand?")) return; rebuyAll(); });
    on("pxRun2", runOut);
    if (el.querySelectorAll) el.querySelectorAll("li.pick").forEach(li => {
      li.onclick = () => { P.pick = P.pick === +li.dataset.i ? null : +li.dataset.i; renderAll(); const t = el.querySelector(`li.pick[data-i="${li.dataset.i}"]`); if (t) t.focus(); };
      li.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); li.onclick(); } };
    });
    if (el.querySelectorAll) panel.wire(el);
    const d = q("details.cx-more"); if (d) d.ontoggle = () => { P.more = d.open; store("gto-practice-more", P.more); };
  }
  // ---- the loop: each render, if it's not your turn, deal the street or move the opponent to act
  function schedule(f) {
    P.busy = true;
    Promise.resolve().then(f).catch(e => { P.error = String(e); }).then(() => { P.busy = false; renderAll(); });
  }
  function tick(sim) {
    // (several tables: this table's hand waits, also while multitable.js loads)
    if (P.on && P.tables > 1) { if (mt()) return window.MultiTable.tick(); return bar(sim); }
    bar(sim);
    if (!P.on) return;
    if (P.busy || P.error) return;
    if (!ours()) return schedule(deal);
    if (offDrill(sim)) {   // (dealt again until the drill's spot comes up, within reason)
      if (++P.tries <= DRILL_TRIES) return schedule(deal);
      P.error = `No ${DRILLS[P.drill].toLowerCase()} came up in ${DRILL_TRIES} deals: try another seat or drill.`; return bar(sim);
    }
    if (sim.handOver) { if (!P.hand.result || P.running) { if (P.running) endRun(sim); else finish(sim); bar(sim); renderTable(sim); renderAction(sim); } return; }
    if (sim.waiting) return schedule(dealStreet);
    if (sim.cur && (sim.toAct !== S.hero || auto(sim))) schedule(villain);
    else if (sim.cur && resultKey === JSON.stringify(payload())) ahead().catch(() => {});   // your turn, solved: look ahead
  }
  function next() { P.tries = 0; editSetup = false; schedule(deal); }   // (a new hand folds the setup; a change in it deals one, open)
  function set(on) {
    on = !!on; const was = P.on;
    // the first time, Practice starts from the solver's setup without its hand; a practice hand still on the solver's
    // page (from before the two kept their own) goes back to Practice, and the solver keeps no hand or table chips of it
    let seed = null;
    if (on && !was && read(STATE) === null) {
      const blank = { ...S, cards: fresh().cards, actions: fresh().actions };
      if (ours()) { seed = S; S = blank; delete S.stacks; } else seed = blank;
    }
    if (on !== was) save();   // (the mode being left, under its own key)
    P.on = on; store(ON, P.on ? 1 : 0); A.gen++; A.key = null;
    if (on !== was) swapState(seed);
    document.body.classList.toggle("practice", P.on);
    const b = document.getElementById("practiceBtn"); b.classList.toggle("active", P.on); b.textContent = P.on ? "Exit practice" : "Practice";
    if (P.on) document.getElementById("toast").hidden = true;   // e.g. an import's "Review whole hand", for the hand dealt over
    if (!P.on && mt()) window.MultiTable.stop();
    if (P.on && !ours()) deal();
    renderAll();
    if (P.on && P.tables > 1) setTables(P.tables);   // (the multi-table view, as it was left)
  }
  // a seat's cards once the hand is over (shown at showdown; everyone's after Show hands or Run it out)
  const shown = (p) => P.on && P.hand && P.hand.result && (P.hand.run || P.hand.result.shown.includes(p)) && ours() ? P.hand.hands[p] : null;
  // at the table, a seat on the felt: its chips once the hand's over (null: the engine's stack behind) and what it
  // collected, or null away from the table
  function seatView(p) {
    const h = P.hand;
    if (!P.on || !h || !h.tbl || !ours() || !P.table || P.table.seq !== h.tbl.seq) return null;
    // (after your fold the others' chips show as they stand until a run-out: the hand isn't over for them)
    const r = h.result, end = r && !P.running && (!r.folded || !!h.run || simulate().active.size < 2), pays = end ? (h.run && h.run.pays) || r.pays || {} : {};
    return { after: end ? P.table.seats[seatOf(p, h.tbl.btn, S.players)] : null, won: pays[p] || 0 };
  }

  window.Practice = { get on() { return P.on; }, get running() { return P.running && !mt(); }, set, tick, next, shown, seatView, runLabel, runOut, rebuyAll, startOver,
    graded, regraded, rank7, handName, settle, posAt, fromAhead, auto, seeking, state: P, ahead: A, setTables,
    ai: { prompt, spot, heroRead, click: aiClick, explain, keyOf, state: AI, md },
    // for multitable.js: the grading, the verdict cards, the showdown and the look-ahead's likely lines
    mt: { likely, habit, spot, feedback, verdict, settle, says, handName, rank7, money, big, ACT, YOU, mixOf, lostCls, ICON, pct, why,
      seatsFor, DRILLS, DRILL_TIP, DRILL_TRIES, CATS, move, heard, radios, aiClick, AHEAD_LINES } };
})();
