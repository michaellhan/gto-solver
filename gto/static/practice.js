// Practice mode, loaded by the page's Practice button on first use: random hands against GTO opponents.
// It drives the page's own engine: your seat and cards are random, the other seats' hands and the board are
// dealt up front, and each opponent's move is drawn from the GTO mix for its actual hand (Advisor.play: the
// preflop chart, then the postflop solve, through the worker or POST /play). Your decisions are graded as
// in a hand review; the advice stays hidden on your turn and shows once you've acted. The running score is
// kept in this browser.
"use strict";
(() => {
  const ON = "gto-practice", HAND = "gto-practice-hand", SCORE = "gto-practice-score";
  const read = (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } };
  const store = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} };
  const ZERO = { hands: 0, decisions: 0, best: 0, evDecisions: 0, evLost: 0, net: 0 };
  // hand: { hero, players, hole, hands: {seat: [c, c]}, board: [5 cards], result }
  const P = { on: false, hand: read(HAND), score: { ...ZERO, ...(read(SCORE) || {}) }, busy: false, status: "", error: "" };

  // ---- 7-card hand strength: [category, tie-breakers...], compared left to right
  const CATS = ["high card", "pair", "two pair", "three of a kind", "straight", "flush", "full house", "four of a kind", "straight flush"];
  function rank7(cs) {
    const v = (c) => 14 - RANKS.indexOf(c[0]), cnt = {}, suits = {};
    for (const c of cs) { cnt[v(c)] = (cnt[v(c)] || 0) + 1; (suits[c[1]] = suits[c[1]] || []).push(v(c)); }
    const straight = (vals) => { const s = new Set(vals); if (s.has(14)) s.add(1); for (let h = 14; h >= 5; h--) if ([0, 1, 2, 3, 4].every(k => s.has(h - k))) return h; return 0; };
    const flush = Object.values(suits).find(x => x.length >= 5);
    if (flush && straight(flush)) return [8, straight(flush)];
    const g = Object.entries(cnt).map(([r, n]) => [n, +r]).sort((a, b) => b[0] - a[0] || b[1] - a[1]);
    const kick = (not, n) => Object.keys(cnt).map(Number).filter(r => !not.includes(r)).sort((a, b) => b - a).slice(0, n);
    if (g[0][0] === 4) return [7, g[0][1], ...kick([g[0][1]], 1)];
    if (g[0][0] === 3 && g[1] && g[1][0] >= 2) return [6, g[0][1], g[1][1]];
    if (flush) return [5, ...flush.sort((a, b) => b - a).slice(0, 5)];
    const st = straight(Object.keys(cnt).map(Number));
    if (st) return [4, st];
    if (g[0][0] === 3) return [3, g[0][1], ...kick([g[0][1]], 2)];
    if (g[0][0] === 2 && g[1][0] === 2) return [2, g[0][1], g[1][1], ...kick([g[0][1], g[1][1]], 1)];
    if (g[0][0] === 2) return [1, g[0][1], ...kick([g[0][1]], 3)];
    return [0, ...kick([], 5)];
  }
  const cmp = (a, b) => { for (let i = 0; i < Math.max(a.length, b.length); i++) if ((a[i] || 0) !== (b[i] || 0)) return (a[i] || 0) - (b[i] || 0); return 0; };

  // ---- dealing
  function deal() {
    const deck = []; for (const r of RANKS) for (const s of SUITS) deck.push(r + s);
    for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
    const hero = POS[Math.floor(Math.random() * POS.length)], hands = {};
    for (const p of POS) hands[p] = [deck.pop(), deck.pop()];
    S = { ...fresh(S.players), hero, stack: S.stack, ante: S.ante, straddle: S.straddle, rake: S.rake, rakeCap: S.rakeCap, ...cleanAdv(S, POS) };   // (as New hand)
    S.cards[0] = hands[hero][0]; S.cards[1] = hands[hero][1];
    const hole = hands[hero]; delete hands[hero];
    P.hand = { hero, players: S.players, hole, hands, board: deck.slice(0, 5), result: null };
    P.error = ""; store(HAND, P.hand); save(); resetAdvice();
  }
  // the hand on the page is the one dealt here (not changed by New hand, the setup or a share link)
  const ours = () => { const h = P.hand; return !!h && h.hero === S.hero && h.players === S.players && S.cards[0] === h.hole[0] && S.cards[1] === h.hole[1]
    && POS.every(p => p === h.hero || h.hands[p]) && S.cards.slice(2).every((c, i) => !c || c === h.board[i]); };
  function dealStreet() {
    const sim = simulate();
    for (let i = 2; i < 2 + NEED[sim.waiting]; i++) if (!S.cards[i]) S.cards[i] = P.hand.board[i - 2];
    save();
  }

  // ---- an opponent's move: the GTO mix for its hand, drawn by the solver
  async function villain() {
    await modeReady;
    const sim = simulate(), p = sim.toAct, before = JSON.stringify([S.actions, S.cards]);
    P.status = `${SEAT_LABEL(p, S.players)} is thinking…`; bar(sim);
    let res;
    try { res = await callAdvise(payload(), { seat: p, hole: P.hand.hands[p].join("") }); }
    catch (e) { res = { error: "Request failed: " + e }; }
    P.status = "";
    if (!P.on || before !== JSON.stringify([S.actions, S.cards])) return;   // the hand moved on while it thought
    if (res.cancelled) return;                                           // asked again on the next render
    if (res.error) { P.error = `${p}'s move failed: ${res.error}`; return; }
    const e = act(res.action === "bet" || res.action === "raise" ? { action: "bet", to: res.to } : { action: res.action });
    if (e) P.error = `${p}'s move (${res.name}) was rejected: ${e}`;
  }

  // ---- the end of the hand: showdown, your result and the running score
  function finish(sim) {
    const h = P.hand;
    if (h.result) return;
    const open = sim.streets[sim.current] && !sim.streets[sim.current].closed;   // you folded mid-street
    const inv = (p) => sim.invested[p] + (open ? sim.streets[sim.current].bets[p] : 0);
    const board = boardCards(), live = POS.filter(p => sim.active.has(p));
    let pot = POS.reduce((t, p) => t + inv(p), 0);
    // rake (pots that saw a flop) is on the called pot: a bet or raise nobody matched goes back untouched
    const most = POS.map(inv).sort((a, b) => b - a), uncalled = most[0] - (most[1] || 0);
    if (board.length >= 3 && RAKE()) pot -= Math.min((pot - uncalled) * RAKE() / 100, RAKE_CAP() || Infinity);
    const cards = (p) => p === S.hero ? h.hole : h.hands[p];
    let winners = live, shown = [], best = null;
    if (!sim.active.has(S.hero)) winners = [];
    else if (live.length > 1) {
      const val = Object.fromEntries(live.map(p => [p, rank7([...cards(p), ...board])]));
      best = live.reduce((b, p) => cmp(val[p], val[b]) > 0 ? p : b);
      winners = live.filter(p => cmp(val[p], val[best]) === 0);
      shown = live.filter(p => p !== S.hero);
      best = CATS[val[best][0]];
    }
    const net = Math.round(((winners.includes(S.hero) ? pot / winners.length : 0) - inv(S.hero)) * 100) / 100;
    h.result = { net, winners, shown, best, folded: !sim.active.has(S.hero) };
    const mine = []; for (const s of STREETS) for (const a of S.actions[s]) if (a.pos === S.hero && a.g) mine.push(a.g);
    const sc = P.score, ev = mine.filter(g => g.l != null);
    sc.hands++; sc.decisions += mine.length; sc.best += mine.filter(g => g.m === g.b).length;
    sc.evDecisions += ev.length; sc.evLost += ev.reduce((t, g) => t + g.l, 0); sc.net += net;
    store(HAND, h); store(SCORE, sc);
  }

  // ---- the practice bar (in the action dock)
  const money = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x)) + " bb";
  function bar(sim) {
    const el = document.getElementById("practiceBar"); el.hidden = !P.on; if (!P.on) return;
    const h = P.hand, r = h && h.result, sc = P.score;
    let msg;
    if (P.error) msg = `<span class="err">${P.error}</span> <button class="link" id="pxRetry">Retry</button>`;
    else if (r) msg = `<b>${r.folded ? "You folded" : r.winners.includes(S.hero) ? (r.winners.length > 1 ? "Split pot" : "You win") : "You lose"}</b> <span class="num ${r.net > 0 ? "pos" : r.net < 0 ? "neg" : ""}">${money(r.net)}</span>`
      + (r.shown.length ? ` · ${r.shown.map(p => `${SEAT_LABEL(p, S.players)} ${h.hands[p].map(cardHtml).join("")}`).join(" · ")}${r.best ? ` · ${r.winners.map(p => p === S.hero ? "you" : SEAT_LABEL(p, S.players)).join(" and ")} with ${r.best}` : ""}` : "")
      + ` <button class="small" id="pxNext">Next hand <kbd>N</kbd></button>`;
    else if (P.status) msg = `<span class="spinner"></span>${P.status}`;
    else if (sim.toAct === S.hero) msg = resultKey !== JSON.stringify(payload()) ? `<span class="spinner"></span>Your move (solving the spot so it can be graded)` : "Your move. The advice shows once you've acted.";
    else msg = "";
    const avg = sc.evDecisions ? `${(sc.evLost / sc.evDecisions).toFixed(2)} bb lost/decision` : "";
    const best = sc.decisions ? `${Math.round(sc.best / sc.decisions * 100)}% best` : "";
    el.innerHTML = `<span class="pxmsg">${msg}</span><span class="pxscore muted" title="Practice score in this browser: hands played, average EV given up per decision with an EV (postflop), share of decisions that took the solver's top action, and chips won">${[`${sc.hands} hands`, avg, best, sc.hands ? money(Math.round(sc.net * 100) / 100) : ""].filter(Boolean).join(" · ")} <button class="link" id="pxReset">reset</button></span>`;
    const on = (id, f) => { const b = el.querySelector ? el.querySelector("#" + id) : null; if (b) b.onclick = f; };
    on("pxRetry", () => { P.error = ""; renderAll(); });
    on("pxNext", () => next());
    on("pxReset", () => { P.score = { ...ZERO }; store(SCORE, P.score); renderAll(); });
  }

  // ---- the loop: each render, if it's not your turn, deal the street or move the opponent to act
  function schedule(f) {
    P.busy = true;
    Promise.resolve().then(f).catch(e => { P.error = String(e); }).then(() => { P.busy = false; renderAll(); });
  }
  function tick(sim) {
    bar(sim);
    if (!P.on) return;
    if (P.busy || P.error) return;
    if (!ours()) return schedule(deal);
    if (sim.handOver) { if (!P.hand.result) { finish(sim); bar(sim); renderTable(sim); } return; }
    if (sim.waiting) return schedule(dealStreet);
    if (sim.cur && sim.toAct !== S.hero) schedule(villain);
  }
  function next() { schedule(deal); }
  function set(on) {
    P.on = !!on; store(ON, P.on ? 1 : 0);
    document.body.classList.toggle("practice", P.on);
    const b = document.getElementById("practiceBtn"); b.classList.toggle("active", P.on);
    if (P.on && !ours()) deal();
    renderAll();
  }
  // a seat's cards once the hand is over (shown at showdown)
  const shown = (p) => P.on && P.hand && P.hand.result && P.hand.result.shown.includes(p) && ours() ? P.hand.hands[p] : null;

  window.Practice = { get on() { return P.on; }, set, tick, next, shown, rank7, state: P };
})();
