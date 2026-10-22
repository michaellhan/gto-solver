// Practice mode, loaded by the page's Practice button on first use: random hands against GTO opponents.
// It drives the page's own engine: your seat and cards are random, the other seats' hands and the board are
// dealt up front, and each opponent's move is drawn from the GTO mix for its actual hand (Advisor.play: the
// preflop chart, then the postflop solve, through the worker or POST /play). Your decisions are graded as
// in a hand review; the advice stays hidden on your turn and shows once you've acted. Your seat is random or
// the one picked in the coach panel. The running score is kept in this browser.
"use strict";
(() => {
  const ON = "gto-practice", HAND = "gto-practice-hand", SCORE = "gto-practice-score", SEAT = "gto-practice-seat";
  const read = (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } };
  const store = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} };
  const ZERO = { hands: 0, decisions: 0, best: 0, good: 0, evDecisions: 0, evLost: 0, net: 0 };
  // hand: { hero, players, hole, hands: {seat: [c, c]}, board: [5 cards], result }
  // seat: "random" or a position; more: the verdict's details open; detail: the solver's full analysis shown
  const P = { on: false, hand: read(HAND), score: { ...ZERO, ...(read(SCORE) || {}) }, busy: false, status: "", error: "",
    seat: read(SEAT) || "random", more: !!read("gto-practice-more"), detail: !!read("gto-practice-detail") };
  if (P.score.decisions && !(read(SCORE) || {}).good) P.score.good = P.score.best;   // (a score from before "good" was kept)

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
    const hero = POS.includes(P.seat) ? P.seat : POS[Math.floor(Math.random() * POS.length)], hands = {};
    for (const p of POS) hands[p] = [deck.pop(), deck.pop()];
    S = { ...fresh(S.players), hero, stack: S.stack, ante: S.ante, straddle: S.straddle, rake: S.rake, rakeCap: S.rakeCap, ...cleanAdv(S, POS), ...cleanStacks(S, POS) };   // (as New hand: seats keep their stacks)
    S.cards[0] = hands[hero][0]; S.cards[1] = hands[hero][1];
    const hole = hands[hero]; delete hands[hero];
    P.hand = { hero, players: S.players, hole, hands, board: deck.slice(0, 5), result: null, log: [] };
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
    // (clamp: a size under the minimum raise is lifted to it, rather than refused)
    const e = act(res.action === "bet" || res.action === "raise" ? { action: "bet", to: res.to, clamp: true } : { action: res.action });
    if (e) P.error = `${p}'s move (${res.name}) was rejected: ${e}`;
  }

  // ---- the end of the hand: showdown, your result and the running score
  function finish(sim) {
    const h = P.hand;
    if (h.result) return;
    const inv = (p) => sim.invested[p];   // (you folded mid-street: that street's bets are in it too)
    const board = boardCards(), live = POS.filter(p => sim.active.has(p));
    let pot = POS.reduce((t, p) => t + inv(p), 0);
    // rake (pots that saw a flop) is on the called pot: a bet or raise nobody matched goes back untouched
    const most = POS.map(inv).sort((a, b) => b - a), uncalled = most[0] - (most[1] || 0);
    if (board.length >= 3 && RAKE()) pot -= Math.min((pot - uncalled) * RAKE() / 100, RAKE_CAP() || Infinity);
    const cards = (p) => p === S.hero ? h.hole : h.hands[p];
    let winners = live, shown = [], best = null, won = pot / Math.max(1, live.length), pots = null;
    if (!sim.active.has(S.hero)) { winners = []; won = 0; }
    else if (live.length > 1) {
      const val = Object.fromEntries(live.map(p => [p, rank7([...cards(p), ...board])]));
      const top = (ps) => { const b = ps.reduce((b, p) => cmp(val[p], val[b]) > 0 ? p : b); return ps.filter(p => cmp(val[p], val[b]) === 0); };
      winners = top(live);
      shown = live.filter(p => p !== S.hero);
      best = CATS[val[winners[0]][0]];
      // side pots: each all-in player's stake caps what they win from everyone; each pot goes to the best hand
      // among the players in it, all of them short the rake in proportion
      const all = Object.fromEntries(POS.map(p => [p, inv(p)])), side = sidePots(all, live, sim.allin);
      const total = side.reduce((t, x) => t + x.amount, 0), keep = total > 0 ? pot / total : 1;
      won = 0;
      pots = side.map(x => { const w = top(x.eligible); if (w.includes(S.hero)) won += x.amount * keep / w.length; return { amount: Math.round(x.amount * keep * 100) / 100, eligible: x.eligible, winners: w }; });
      if (pots.length < 2) pots = null;
    }
    const net = Math.round((won - inv(S.hero)) * 100) / 100;
    h.result = { net, winners, shown, best, folded: !sim.active.has(S.hero), ...(pots ? { pots } : {}) };
    const mine = []; for (const s of STREETS) for (const a of S.actions[s]) if (a.pos === S.hero && a.g) mine.push(a.g);
    const sc = P.score, ev = mine.filter(g => g.l != null);
    sc.hands++; sc.decisions += mine.length; sc.best += mine.filter(g => g.m === g.b).length; sc.good += mine.filter(g => gradeClass(g) === "ok").length;
    sc.evDecisions += ev.length; sc.evLost += ev.reduce((t, g) => t + g.l, 0); sc.net += net;
    store(HAND, h); store(SCORE, sc);
  }

  // ---- your decisions: each one's verdict and the solver's mix for the spot, kept with the hand
  const ACT = (a) => a.action === "bet" ? `bet ${fmt(a.to)}` : a.action === "raise" ? `raise to ${fmt(a.to)}` : a.action;
  const YOU = (a) => ({ fold: "folded", check: "checked", call: "called", "all-in": "went all-in" })[a.action] || (a.action === "bet" ? `bet ${fmt(a.to)}` : `raised to ${fmt(a.to)}`);
  function graded(street, rec, res) {
    if (!P.hand) return;
    const mix = res && !res.error && res.actions ? res.actions.map(a => ({ n: a.action, f: a.freq, ...(a.ev_bb !== undefined ? { ev: a.ev_bb } : {}) })) : null;
    (P.hand.log = P.hand.log || []).push({ street, rec: { action: rec.action, ...(rec.to != null ? { to: rec.to } : {}) }, g: rec.g || null, mix });
    store(HAND, P.hand);
  }
  // the verdict: best (the solver's top action), good (in its mix), inaccuracy (rare), mistake (almost never)
  function verdict(g) {
    if (!g) return { k: "none", label: "Not graded", c: "var(--muted)" };
    const cls = gradeClass(g);
    if (g.m === g.b) return { k: "best", label: "Best move", c: "var(--good)" };
    if (cls === "ok") return { k: "good", label: "Good", c: "var(--good)" };
    if (cls === "meh") return { k: "meh", label: "Inaccuracy", c: "var(--meh)" };
    return { k: "bad", label: "Mistake", c: "var(--bad)" };
  }
  const pct = (f) => `${Math.round(f * 100)}%`;
  const evTxt = (x) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(2)}`;
  function why(e) {
    const g = e.g, v = verdict(g);
    if (!g) return "This spot isn't solved (4 or more players), so this decision isn't graded.";
    const as = g.v ? ` (graded as its ${g.m})` : "";
    if (v.k === "best") return g.f >= 0.995 ? `The solver always plays this here.` : `The solver's top choice here: it plays this ${pct(g.f)} of the time.`;
    if (v.k === "good") return `The solver mixes this in ${pct(g.f)} of the time${as}; its top choice is ${g.b} (${pct(g.bf)}).`;
    return `The solver ${g.f < 0.005 ? "never plays this" : `plays this only ${pct(g.f)} of the time`}${as}; it prefers ${g.b} (${pct(g.bf)}).`;
  }
  // the verdict on one decision: the verdict, what you did and the EV it cost, the solver's mix as a bar with
  // its frequencies; why and the EVs fold away (Details), as does the solver's full analysis of the spot
  function feedback(e) {
    const v = verdict(e.g), g = e.g;
    const cost = g && g.l != null ? (g.l > 0.005 ? `<span class="cost num neg" title="EV given up against the solver's best action">−${g.l.toFixed(2)} bb</span>` : `<span class="cost num pos">no EV lost</span>`) : "";
    let mix = "", table = "";
    if (e.mix && e.mix.length) {
      setColors(e.mix.map(a => a.n));
      const ev = e.mix.some(a => a.ev !== undefined), you = (a) => g && a.n === g.m;
      mix = `<div class="cx-mix">${e.mix.map(a => `<div style="width:${(a.f * 100).toFixed(1)}%;background:${COLORS(a.n)}" title="${a.n} ${pct(a.f)}"></div>`).join("")}</div>`
        + `<div class="cx-leg">${e.mix.map(a => `<span class="${you(a) ? "you" : ""}" style="--c:${COLORS(a.n)}">${a.n} <b class="num">${pct(a.f)}</b>${you(a) ? '<span class="tag">you</span>' : ""}</span>`).join("")}</div>`;
      if (ev) table = `<div class="cx-acts"><span class="hd">Action</span><span class="hd num">How often</span><span class="hd num">EV</span>`
        + e.mix.map(a => `<span class="n${you(a) ? " you" : ""}" style="--c:${COLORS(a.n)}">${a.n}</span><span class="num${you(a) ? " you" : ""}">${pct(a.f)}</span><span class="num">${a.ev !== undefined ? evTxt(a.ev) + " bb" : ""}</span>`).join("") + `</div>`;
    }
    return `<div class="cx-fb" style="--c:${v.c}"><div class="head"><span class="verdict">${v.label}</span><span class="what">${e.street} · you <b>${YOU(e.rec)}</b></span>${cost}</div>${mix}`
      + `<details class="cx-more"${P.more ? " open" : ""}><summary>Details</summary><div class="why">${why(e)}${g && g.r ? ' <span class="muted">(rough solve: a hint, not a verdict)</span>' : ""}</div>${table}`
      + `<button class="link" id="pxDetail">${P.detail ? "Hide" : "Show"} the solver's full analysis</button></details></div>`;
  }

  // ---- the coach panel: what's happening now, the verdict on your last decision, the result, the score
  const money = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x)) + " bb";
  const names = (ps) => ps.map(p => p === S.hero ? "you" : SEAT_LABEL(p, S.players)).join(" and ");
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
    let cls = "", t, sub;
    if (P.error) { cls = " err"; t = "Something went wrong"; sub = `<span class="err">${P.error}</span> <button class="link" id="pxRetry">Retry</button>`; }
    else if (P.status) { t = P.status; sub = `${seat} · ${sim.current || ""}`; }
    else if (sim.toAct === S.hero && sim.cur) {
      const info = sim.cur, toCall = Math.min(info.curBet - info.bets[S.hero], STACK(S.hero) - sim.invested[S.hero] - info.bets[S.hero]);
      const ready = resultKey === JSON.stringify(payload());
      cls = ready ? " mine" : " wait"; t = ready ? "Your turn" : `<span class="spinner"></span>Solving your spot…`;
      sub = `${seat} · ${sim.current} · pot ${fmt(info.pot)} bb${toCall > 1e-9 ? ` · <b>${fmt(toCall)} bb to call</b>` : ""}`;
    } else { t = "Dealing…"; sub = `${seat}${sim.current ? " · " + sim.current : ""}`; }
    return `<div class="cx-now${cls}"><div class="cx-cards">${big(h.hole)}</div><div><div class="t">${t}</div><div class="s">${sub}</div>${boardRow}</div></div>`;
  }
  function summary(r) {
    const h = P.hand, log = h.log || [];
    const shown = r.shown.map(p => `<span class="${r.winners.includes(p) ? "win" : ""}"><b>${SEAT_LABEL(p, S.players)}</b>${big(h.hands[p], true)}</span>`).join("");
    const mineShown = r.shown.length && !r.folded ? `<span class="${r.winners.includes(S.hero) ? "win" : ""}"><b>You</b>${big(h.hole, true)}</span>` : "";
    let out = `<div class="cx-res"><div><span class="big">${r.folded ? "You folded" : headline(r)}</span><span class="net num ${r.net > 0 ? "pos" : r.net < 0 ? "neg" : ""}">${money(r.net)}</span></div>`
      + (r.shown.length ? `<div class="cx-show">${mineShown}${shown}</div>` : "")
      + (r.best ? `<div class="pots">${names(r.winners)} ${r.winners.length > 1 || r.winners[0] === S.hero ? "win" : "wins"} with ${r.best}</div>` : "")
      + (r.pots ? `<div class="pots">${r.pots.map((x, i) => `${i ? "side" : "main"} pot ${fmt(x.amount)}: ${names(x.winners)}`).join(" · ")}</div>` : "") + `</div>`;
    if (log.length) out += feedback(log[log.length - 1]);   // your last decision's verdict, as after any other
    if (log.length > 1) {
      const lost = log.reduce((t, e) => t + (e.g && e.g.l || 0), 0), good = log.filter(e => ["best", "good"].includes(verdict(e.g).k)).length;
      out += `<div class="cx-h">This hand · ${good} of ${log.length} in the solver's strategy${lost > 0.005 ? ` · <span class="num neg">−${lost.toFixed(2)} bb</span> EV lost` : ""}</div><ul class="cx-list">`
        + log.map(e => { const v = verdict(e.g); return `<li title="${e.g ? gradeTip(e.g) : ""}"><span class="st">${e.street}</span><span>${ACT(e.rec)} <span class="muted">· solver ${e.g ? pct(e.g.f) : "–"}</span></span><span class="pill" style="--c:${v.c}">${v.label}</span><span class="num ${e.g && e.g.l > 0.005 ? "neg" : "muted"}">${e.g && e.g.l != null ? (e.g.l > 0.005 ? "−" + e.g.l.toFixed(2) + " bb" : "0 bb") : ""}</span></li>`; }).join("") + `</ul>`;
    }
    return out;
  }
  // the score, one line
  function stats() {
    const sc = P.score, acc = sc.decisions ? pct(sc.good / sc.decisions) : "–";
    const avg = sc.evDecisions ? (sc.evLost / sc.evDecisions).toFixed(2) : "–";
    return `<div class="cx-stats" title="Your practice score in this browser"><span><b class="num">${sc.hands}</b> hands</span>`
      + `<span title="Decisions the solver's strategy includes (it plays them 15% of the time or more)"><b class="num">${acc}</b> accuracy</span>`
      + `<span title="Average EV given up against the solver's best action, per decision that has an EV (after the flop)"><b class="num">${avg}</b> bb lost / decision</span>`
      + `<span><b class="num ${sc.net > 0 ? "pos" : sc.net < 0 ? "neg" : ""}">${sc.hands ? money(Math.round(sc.net * 100) / 100) : "–"}</b> won</span>`
      + `<button class="link" id="pxReset">Reset</button></div>`;
  }
  // the seat picker above the panel (static, so a re-render doesn't close it while open): Random or a seat
  function seatPicker() {
    const sel = document.getElementById("pxSeat"), want = ["random", ...POS].join();
    if (sel.dataset.opts !== want) {
      sel.innerHTML = `<option value="random">Random</option>` + POS.map(p => `<option value="${p}">${SEAT_LABEL(p, POS.length)}</option>`).join("");
      sel.dataset.opts = want;
    }
    sel.value = POS.includes(P.seat) ? P.seat : "random";
    sel.onchange = () => { P.seat = sel.value; store(SEAT, P.seat); next(); };
    const n = document.getElementById("pxHandNo"); n.textContent = `Hand ${P.score.hands + (P.hand && P.hand.result ? 0 : 1)}`;
  }
  function bar(sim) {
    const el = document.getElementById("practiceBar"); el.hidden = !P.on; if (!P.on) return;
    seatPicker();
    const h = P.hand, r = h && h.result, log = (h && h.log) || [];
    let body = "";
    if (h) body = r ? summary(r) : now(sim) + (log.length ? feedback(log[log.length - 1]) : "");   // (until you act again)
    el.innerHTML = body + stats();
    // the solver's full analysis of your last decision, under the coach, when asked for
    document.body.classList.toggle("pxdetail", P.detail && log.length > 0);
    const q = (sel) => el.querySelector ? el.querySelector(sel) : null, on = (id, f) => { const b = q("#" + id); if (b) b.onclick = f; };
    on("pxRetry", () => { P.error = ""; renderAll(); });
    on("pxDetail", () => { P.detail = !P.detail; store("gto-practice-detail", P.detail); renderAll(); });
    on("pxReset", () => { if (typeof confirm === "function" && !confirm("Reset your practice score?")) return; P.score = { ...ZERO }; store(SCORE, P.score); renderAll(); });
    const d = q("details.cx-more"); if (d) d.ontoggle = () => { P.more = d.open; store("gto-practice-more", P.more); };
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
    const b = document.getElementById("practiceBtn"); b.classList.toggle("active", P.on); b.textContent = P.on ? "Exit practice" : "Practice";
    if (P.on) document.getElementById("toast").hidden = true;   // e.g. an import's "Review whole hand", for the hand dealt over
    if (P.on && !ours()) deal();
    renderAll();
  }
  // a seat's cards once the hand is over (shown at showdown)
  const shown = (p) => P.on && P.hand && P.hand.result && P.hand.result.shown.includes(p) && ours() ? P.hand.hands[p] : null;

  window.Practice = { get on() { return P.on; }, set, tick, next, shown, graded, rank7, state: P };
})();
