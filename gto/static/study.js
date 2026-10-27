// Study mode, loaded by index.html the first time "Study" is opened from the ⋯ menu: a solution browser. A hand is
// walked node by node with no cards of your own: at each decision the solver answers for whoever acts with every hand
// of their range (gto/advisor.py, Advisor.study): the 13x13 grid by strategy, EV, equity or EQR, each class's combos,
// the range's mix and EV per action, the opponent's range; an action clicked plays it, a card picked (or a random one)
// deals the next street, the line above steps back. It starts from a precomputed spot's preflop line (the flop
// library's, instant on its flops), the start of a hand, or the hand on the page's table. The pure parts (states,
// requests, the HTML) are what the Node tests call.
(function () {
  "use strict";
  const RANKS = "AKQJT98765432", SUITS = "cdhs", SYM = { s: "♠", h: "♥", d: "♦", c: "♣" };
  const STREETS = ["preflop", "flop", "turn", "river"];
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const pct = (x, d = 0) => x == null ? "—" : (x * 100).toFixed(d) + "%";
  const sgn = (x, d = 2) => x == null ? "—" : (x > 0 ? "+" : "") + x.toFixed(d);
  const num = (x) => String(Math.round(x * 100) / 100);
  const classAt = (i, j) => i === j ? RANKS[i] + RANKS[j] : i < j ? RANKS[i] + RANKS[j] + "s" : RANKS[j] + RANKS[i] + "o";
  const combosOf = (cls) => cls.length === 2 ? 6 : cls[2] === "s" ? 4 : 12;
  const cardsOf = (s) => (String(s).match(/../g) || []);
  const cardHtml1 = (c) => typeof cardHtml === "function" ? cardHtml(c) : `<span class="${c[1]}">${c[0]}${SYM[c[1]]}</span>`;
  // the page's colours for actions (index.html's COLORS, bets shaded by size once setColors has the names)
  const color = (n) => typeof COLORS === "function" ? COLORS(n) : n === "fold" ? "var(--fold)" : n === "check" || n === "call" ? "var(--passive)"
    : n === "all-in" ? "var(--allin)" : "var(--agg1)";

  // ---- the hand being studied: {players, stack, (stacks, ante, straddle, rake, rake_cap,) board, preflop, flop, turn, river}
  const fresh = (players = 6, stack = 100) => ({ players, stack, board: "", preflop: [], flop: [], turn: [], river: [] });
  // a spot's preflop line as the flop library's index writes it ("UTG:f,...,BTN:r2.5,SB:f,BB:c")
  const parseLine = (line) => String(line || "").split(",").filter(Boolean).map(t => {
    const [pos, a] = t.split(":");
    return a[0] === "r" ? { pos, action: "raise", to: parseFloat(a.slice(1)) } : { pos, action: { f: "fold", c: "call", x: "check" }[a[0]] };
  });
  const fromSpot = (sp) => ({ ...fresh(), preflop: parseLine(sp.line) });
  // the page's own hand (its advice payload) without hero, hole cards and the advice's own settings
  const SETUP = ["players", "stack", "stacks", "ante", "straddle", "rake", "rake_cap", "board", ...STREETS];
  const fromPayload = (p) => Object.fromEntries(SETUP.filter(k => p[k] != null).map(k => [k, STREETS.includes(k) ? (p[k] || []).map(a => ({ ...a })) : p[k]]));
  const streetOf = (st) => ({ 0: "preflop", 3: "flop", 4: "turn", 5: "river" })[cardsOf(st.board).length];
  // the solver's request for the node: Advisor.study
  const request = (st, speed) => ({ ...fromPayload(st), study: true, speed });
  // what an advice action's name plays as: check, call, fold, bet, raise or all-in
  const kindOf = (n) => n === "all-in" ? "all-in" : (/^(check|call|fold|bet|raise)/.exec(n) || [null, n])[1];
  // the state after the actor takes advice action a (amount_bb: a bet's or raise's total this street)
  function play(st, res, a) {
    const kind = kindOf(a.action), s = res.street;
    const act = { pos: res.actor, action: kind, ...(kind === "bet" || kind === "raise" ? { to: a.amount_bb } : {}) };
    return { ...st, [s]: [...(st[s] || []), act] };
  }
  const deal = (st, cards) => ({ ...st, board: st.board + cards });
  // n random cards off the board
  function randomCards(st, n, rnd = Math.random) {
    const used = new Set(cardsOf(st.board)), deck = [];
    for (const r of RANKS) for (const s of SUITS) if (!used.has(r + s)) deck.push(r + s);
    let out = "";
    for (let i = 0; i < n; i++) out += deck.splice(Math.floor(rnd() * deck.length), 1)[0];
    return out;
  }
  // typed cards ("kd 7c 2H", "Td") normalized for the board, or {error}
  function readCards(st, text, n) {
    const t = String(text).replace(/[\s,]+/g, "").replace(/10/g, "T");
    const cs = t.match(/../g) || [];
    if (cs.length !== n || cs.join("") !== t) return { error: `type ${n === 1 ? "a card" : n + " cards"}, like ${n === 1 ? "5s" : "Kd7c2h"}` };
    const out = cs.map(c => c[0].toUpperCase() + c[1].toLowerCase());
    if (out.some(c => !RANKS.includes(c[0]) || !SUITS.includes(c[1]))) return { error: "cards are a rank (A K Q J T 9-2) and a suit (s h d c)" };
    const used = new Set(cardsOf(st.board));
    if (new Set(out).size !== n || out.some(c => used.has(c))) return { error: "that card is already out" };
    return { cards: out.join("") };
  }
  const actLabel = (a) => {
    const n = a.action === "all-in" ? "all-in" : a.action, to = a.to ?? a.amount;
    return `${a.pos} ${n}${to != null && n !== "call" ? " " + num(to) : ""}`;
  };
  // one step back: the last action's decision, or before a street's cards were dealt
  function back(st) {
    const items = line(st), last = items[items.length - 1];
    if (!last || last.kind === "street" && last.street === "preflop") return null;
    if (last.kind === "act") return last.st;
    return { ...st, board: cardsOf(st.board).slice(0, { flop: 0, turn: 3, river: 4 }[last.street]).join("") };
  }
  // the line so far, to step back along: per street a heading (its state before any action) and its actions (each
  // the state it was taken from); preflop folds in a row are one step
  function line(st) {
    const out = [];
    if (!st) return out;
    for (const s of STREETS) {
      const need = { preflop: 0, flop: 3, turn: 4, river: 5 }[s], cs = cardsOf(st.board);
      if (cs.length < need) break;
      const base = { ...st, board: cs.slice(0, need).join("") };
      for (const t of STREETS.slice(STREETS.indexOf(s))) base[t] = [];
      for (const t of STREETS.slice(0, STREETS.indexOf(s))) base[t] = st[t];
      out.push({ kind: "street", street: s, cards: s === "preflop" ? "" : s === "flop" ? cs.slice(0, 3).join("") : cs[need - 1], st: base });
      const acts = st[s] || [];
      for (let i = 0; i < acts.length; i++) {
        const prev = out[out.length - 1];
        const at = { ...base, [s]: acts.slice(0, i) };
        if (s === "preflop" && acts[i].action === "fold" && prev.kind === "act" && prev.fold) { prev.label = `${prev.first}–${acts[i].pos} fold`; continue; }
        out.push({ kind: "act", label: actLabel(acts[i]), st: at, fold: s === "preflop" && acts[i].action === "fold", first: acts[i].pos });
      }
    }
    return out;
  }
  // an action's label: "Bet 33% · 1.8bb", "Raise to 7.5bb", "Call 2bb", "All-in 97.5bb"
  function label(a) {
    const n = a.action, cap = n[0].toUpperCase() + n.slice(1), amt = a.amount_bb;
    if (amt == null) return cap;
    if (/\d/.test(n)) return /%/.test(n) ? `${cap} · ${num(amt)}bb` : cap + (/bb$/.test(n) ? "" : "bb");
    return n === "raise" ? `Raise to ${num(amt)}bb` : `${cap} ${num(amt)}bb`;
  }
  const eqr = (c, pot) => c.eq > 0 && c.evm != null ? c.evm / (c.eq * pot) : null;

  // ---- the grid: a cell per hand class; strategy: its mix stacked, as tall as its share of the class's combos; ev,
  // equity, eqr: shaded red to green with the number; the opponent's range: its weight and equity
  const VIEWS = { strategy: "Strategy", ev: "EV", equity: "Equity", eqr: "EQR" };
  const hue = (t) => `hsl(${Math.round(Math.max(0, Math.min(1, t)) * 120)} 50% 34%)`;
  function cellValue(c, view, res, focus) {
    if (view === "ev") { const v = focus != null ? c.ev[focus] : c.evm; return v == null ? null : { v, t: 0.5 + v / (2 * res.pot_bb), txt: (Math.abs(v) < 0.05 ? 0 : v).toFixed(1) }; }
    if (view === "equity") return c.eq == null ? null : { v: c.eq, t: c.eq, txt: (c.eq * 100).toFixed(0) };
    if (view === "eqr") { const v = eqr(c, res.pot_bb); return v == null ? null : { v, t: (v - 0.4) / 1.2, txt: v.toFixed(2) }; }
    return null;
  }
  function gridHtml(res, ui) {
    const opp = ui.who === "opp" && res.opponent_grid, g = opp ? res.opponent_grid : res.grid, names = g.actions;
    const view = opp && ui.view !== "equity" ? "strategy" : ui.view, focus = opp ? null : ui.focus;
    let h = `<div class="sy-grid${view === "strategy" ? "" : " sy-num"}">`;
    for (let i = 0; i < 13; i++) for (let j = 0; j < 13; j++) {
      const cls = classAt(i, j), c = g.cells[cls];
      let inner = "", style = "";
      if (c && c.w > 0) {
        const ht = (Math.min(1, c.w) * 100).toFixed(1);
        if (view === "strategy") {
          const segs = opp ? `<i style="width:100%;background:var(--accent)"></i>` : names.map((n, a) => c.f[a] > 0.0005 && (focus == null || focus === a)
            ? `<i style="width:${(c.f[a] * 100).toFixed(1)}%;background:${color(n)}"></i>` : "").join("");
          inner = `<span class="sy-fill" style="height:${ht}%">${segs}</span>`;
        } else {
          const v = cellValue(c, view, res, focus);
          if (v) { style = `background:${hue(v.t)}`; inner = `<span class="sy-v">${v.txt}</span>`; }
        }
      }
      h += `<div class="sy-cell${i === j ? " pair" : ""}${!c || !(c.w > 0) ? " out" : ""}${ui.sel === cls ? " sel" : ""}" data-cls="${cls}" style="${style}">${inner}<span class="sy-l">${cls}</span></div>`;
    }
    return h + `</div>`;
  }
  // the actions: the range's frequency, its combos and the EV of the hands that take it; a row plays it
  function actionsHtml(res, ui) {
    const tot = res.range_combos;
    let h = `<table class="sy-acts"><thead><tr><th>Action</th><th class="n">Range</th><th class="n">Combos</th>${res.street === "preflop" ? "" : `<th class="n" title="The EV in bb of the hands that take it, relative to folding now">EV</th>`}</tr></thead><tbody>`;
    res.actions.forEach((a, i) => {
      h += `<tr data-act="${i}" tabindex="0" class="${ui.focus === i ? "focus" : ""}" title="Play ${esc(label(a))}"><td><i class="sy-sw" style="background:${color(a.action)}"></i>${esc(label(a))}</td>`
        + `<td class="n"><span class="sy-mini"><span style="width:${(a.freq * 100).toFixed(1)}%;background:${color(a.action)}"></span></span>${pct(a.freq, 1)}</td>`
        + `<td class="n">${a.combos.toFixed(1)}</td>${res.street === "preflop" ? "" : `<td class="n">${sgn(a.ev_bb)}</td>`}</tr>`;
    });
    h += `</tbody></table><div class="sy-bar">${res.actions.map(a => a.freq > 0.0005 ? `<i style="width:${(a.freq * 100).toFixed(2)}%;background:${color(a.action)}" title="${esc(label(a))} ${pct(a.freq, 1)}"></i>` : "").join("")}</div>`;
    h += `<div class="sy-tot">${res.actor}'s range: <b>${num(tot)}</b> combos`
      + (res.range_ev_bb != null ? ` · EV <b>${sgn(res.range_ev_bb)}</b>bb · equity <b>${pct(res.range_eq, 1)}</b> · EQR <b>${(res.range_ev_bb / (res.range_eq * res.pot_bb)).toFixed(2)}</b>` : "")
      + (res.opponent ? `<br>${res.opponent}: ${num(res.opponent_combos)} combos · equity ${pct(res.opponent_eq, 1)}` : "") + `</div>`;
    return h;
  }
  // a class's detail: its mix and numbers, then each combo of it (sorted by EV)
  function detailHtml(res, ui, cls) {
    if (!cls) return `<div class="sy-hint">${typeof window === "object" && "ontouchstart" in window ? "Tap" : "Hover or click"} a hand for its combos.</div>`;
    const opp = ui.who === "opp" && res.opponent_grid, g = opp ? res.opponent_grid : res.grid, c = g.cells[cls], names = g.actions;
    const n = combosOf(cls);
    if (!c || !(c.w > 0)) return `<div class="sy-dh"><b>${cls}</b> <span class="sy-mut">not in ${opp ? res.opponent + "'s" : res.actor + "'s"} range here</span></div>`;
    let h = `<div class="sy-dh"><b>${cls}</b> <span class="sy-mut">${num(c.w * n)} of ${n} combos</span>`;
    if (opp) return h + (c.eq != null ? ` · equity <b>${pct(c.eq, 1)}</b>` : "") + `</div>`;
    h += c.eq != null ? ` · EV <b>${sgn(c.evm)}</b> · eq <b>${pct(c.eq, 1)}</b> · EQR <b>${eqr(c, res.pot_bb).toFixed(2)}</b>` : "";
    h += `</div><div class="sy-mix">${names.map((nm, a) => c.f[a] >= 0.005 ? `<span style="--c:${color(nm)}">${esc(nm)} <b>${pct(c.f[a])}</b>${c.ev ? ` <small>${sgn(c.ev[a])}</small>` : ""}</span>` : "").join("")}</div>`;
    const cs = (res.combos || {})[cls];
    if (!cs || !cs.length) return h;
    const evOf = (x) => x[2].reduce((t, f, a) => t + f * x[3][a], 0);
    h += `<table class="sy-cb"><thead><tr><th>Combo</th><th>Mix</th><th class="n">EV</th><th class="n">Eq</th></tr></thead><tbody>`;
    for (const x of [...cs].sort((a, b) => evOf(b) - evOf(a))) {
      h += `<tr title="${names.map((nm, a) => `${nm} ${pct(x[2][a])} (EV ${sgn(x[3][a])})`).join(", ")}${x[1] < 0.999 ? ` · ${pct(x[1])} of it in the range` : ""}"><td class="sy-c">${cardsOf(x[0]).map(cardHtml1).join("")}${x[1] < 0.999 ? `<small>${pct(x[1])}</small>` : ""}</td>`
        + `<td><span class="sy-mini wide">${names.map((nm, a) => x[2][a] > 0.0005 ? `<span style="width:${(x[2][a] * 100).toFixed(1)}%;background:${color(nm)}"></span>` : "").join("")}</span></td>`
        + `<td class="n">${sgn(evOf(x))}</td><td class="n">${pct(x[4])}</td></tr>`;
    }
    return h + `</tbody></table>`;
  }
  function lineHtml(st, res) {
    let h = "";
    line(st).forEach((it, i) => {
      if (it.kind === "street") h += `<button class="sy-st" data-step="${i}" title="This street's first decision">${it.street === "preflop" ? "Preflop" : it.street[0].toUpperCase() + it.street.slice(1) + " " + cardsOf(it.cards).map(cardHtml1).join("")}</button>`;
      else h += `<button class="sy-a" data-step="${i}" title="Back to this decision">${esc(it.label)}</button>`;
    });
    if (res && res.study === "node") h += `<span class="sy-now">${esc(res.actor)} ?</span>`;
    return h;
  }
  function statusHtml(res) {
    if (res.study !== "node") return "";
    const bits = [`<b>${esc(res.actor)}</b> to act${res.position ? ` (${res.position})` : ""}`, `pot ${num(res.pot_bb)}bb`];
    if (res.to_call_bb > 0) bits.push(`${num(res.to_call_bb)}bb to call`);
    if (res.stack_bb != null) bits.push(`${num(res.stack_bb)}bb behind`);
    return bits.join(" · ");
  }
  function solverHtml(res) {
    if (res.street === "preflop") return "Preflop: the charts' mix (an approximation: no EVs or equities).";
    const s = res.solver || {}, p = res.precomputed;
    const how = p ? `Precomputed (${esc(p.label)})` : `Solved live in ${num(s.solve_seconds || 0)} s`;
    return `${how}: ${s.iterations} iterations, ${num(s.exploitability_pct)}% of the pot from equilibrium.`;
  }

  // ---- the panel (browser)
  let box = null, index = null, res = null, err = null, busy = false, gen = 0;
  const ST = { state: null, spot: "", speed: typeof SPEED === "string" && SPEED !== "quick" ? SPEED : "fast", view: "strategy", who: "me", sel: null, hover: null, focus: null };
  try { const s = JSON.parse(localStorage.getItem("gto-study")) || {}; for (const k of ["state", "spot", "speed", "view"]) if (s[k] != null) ST[k] = s[k]; } catch (e) {}
  const store = () => { try { localStorage.setItem("gto-study", JSON.stringify({ state: ST.state, spot: ST.spot, speed: ST.speed, view: ST.view })); } catch (e) {} };
  const answers = new Map();   // request (JSON) -> its answer's promise
  const ask = (st) => typeof globalThis.studyAsk === "function" ? globalThis.studyAsk(st) : Promise.reject(new Error("no solver"));
  async function go(state) {
    ST.state = state; ST.sel = ST.hover = ST.focus = null; err = null; store();
    const key = JSON.stringify(request(state, ST.speed)), g = ++gen;
    if (!answers.has(key)) answers.set(key, Promise.resolve(ask(JSON.parse(key))).then(r => {
      if (!r || r.cancelled || r.error) answers.delete(key);
      return r || { error: "no answer" };
    }, e => { answers.delete(key); return { error: String(e && e.message || e) }; }));
    busy = true; draw();
    const r = await answers.get(key);
    if (g !== gen) return;
    busy = false;
    if (r.cancelled) { err = "The request was cancelled."; res = null; }
    else if (r.error) { err = r.error; res = null; }
    else { err = null; res = r; res.key = key; }
    if (res && res.study === "node" && typeof setColors === "function") setColors(res.actions.map(a => a.action));
    draw();
  }
  function controls() {
    const spots = (index && index.spots || []).map(s => [s.id, s.label]);
    const opts = [["", "New hand (6-max, 100bb)"], ...spots, ["table", "The hand on the table"]];
    return `<div class="sy-ctl"><select data-ctl="spot" aria-label="Start from">${opts.map(([v, t]) => `<option value="${v}"${v === ST.spot ? " selected" : ""}>${esc(t)}</option>`).join("")}</select>`
      + `<span class="segment">${["fast", "accurate"].map(m => `<button data-speed="${m}" class="${m === ST.speed ? "active" : ""}">${m === "fast" ? "Fast" : "Accurate"}</button>`).join("")}</span>`
      + `<button class="ghost" data-back ${back(ST.state) ? "" : "disabled"} title="Back one step (Backspace)">← Back</button></div>`;
  }
  function body() {
    if (err) return `<div class="sy-msg sy-err">${esc(err)}</div>`;
    if (!res) return `<div class="sy-msg"><span class="spinner"></span>Solving…</div>`;
    if (res.study === "deal") {
      const n = res.street === "flop" ? 3 : 1;
      return `<div class="sy-deal"><div>${res.allin ? "Nobody can bet any more. " : ""}Deal the ${res.street}:</div><form data-deal><input name="cards" autocomplete="off" spellcheck="false" placeholder="${n === 3 ? "e.g. Kd7c2h" : "e.g. 5s"}" aria-label="The ${res.street}">`
        + `<button type="submit">Deal</button><button type="button" data-random>Random</button></form><div class="sy-err" data-dealerr></div></div>`;
    }
    if (res.study === "over") return `<div class="sy-msg">${res.winner ? `Hand over: everyone else folded, ${esc(res.winner)} wins the pot.` : "Hand over: the river's betting is done, the hands go to showdown."} Step back along the line above to look at another branch.</div>`;
    const pre = res.street === "preflop", cls = ST.hover || ST.sel;
    const views = pre ? ["strategy"] : [...Object.keys(VIEWS), "breakdown"];
    const view = views.includes(ST.view) ? ST.view : "strategy";
    const ui = { ...ST, view };
    let h = `<div class="sy-status">${statusHtml(res)}</div><div class="sy-main"><div class="sy-left"><div class="sy-ctl">`
      + `<span class="segment">${views.map(v => `<button data-view="${v}" class="${v === view ? "active" : ""}">${VIEWS[v] || "Breakdown"}</button>`).join("")}</span>`
      + (res.opponent_grid && view !== "breakdown" ? `<span class="segment">${["me", "opp"].map(w => `<button data-who="${w}" class="${w === ST.who ? "active" : ""}">${esc(w === "me" ? res.actor : res.opponent)}</button>`).join("")}</span>` : "")
      + `</div>`;
    h += view === "breakdown" ? `<div class="sy-bd"></div>` : gridHtml(res, ui);
    h += `</div><div class="sy-right">${actionsHtml(res, ui)}<div class="sy-detail">${view === "breakdown" ? "" : detailHtml(res, ui, cls)}</div></div></div>`;
    h += `<div class="sy-note">${solverHtml(res)} ${pre ? "" : `EV: bb won from here relative to folding now. Equity: against ${esc(res.opponent)}'s range with the rest of the board dealt. EQR: EV over equity × pot.`}`
      + `${(res.notes || []).length ? `<br>${res.notes.map(esc).join("<br>")}` : ""}</div>`;
    return h;
  }
  function draw() {
    if (!box) return;
    const b = box.querySelector(".sy-body");
    // (while solving: the last decision dimmed under a spinner; the spinner alone after a deal or an error)
    const dim = busy && res && res.study === "node";
    b.innerHTML = controls() + `<div class="sy-line">${lineHtml(ST.state, res && !busy ? res : null)}</div>` + `<div class="sy-stage${dim ? " busy" : ""}">${busy && !dim ? `<div class="sy-msg"><span class="spinner"></span>Solving…</div>` : body()}</div>`;
    if (dim) b.querySelector(".sy-stage").insertAdjacentHTML("afterbegin", `<div class="sy-wait"><span class="spinner"></span>Solving…</div>`);
    const bd = b.querySelector(".sy-bd");
    if (bd && res && !busy) showBreakdown(bd);
    const inp = b.querySelector("[data-deal] input"); if (inp && !busy && !("ontouchstart" in window)) inp.focus();
  }
  function showBreakdown(el) {
    const go2 = () => window.Breakdown.show(el, "breakdown", res, res.key, (st) => ask(st));
    if (window.Breakdown) return go2();
    el.innerHTML = `<div class="sy-msg"><span class="spinner"></span>Loading…</div>`;
    const p = typeof loadJs === "function" ? loadJs("breakdown.js?v=" + (typeof BD_VERSION === "string" ? BD_VERSION : "1"), () => window.Breakdown) : Promise.reject(new Error("no loader"));
    p.then(go2, () => { el.innerHTML = `<div class="sy-msg sy-err">Couldn't load the breakdown (breakdown.js).</div>`; });
  }
  // hover, focus and a hand picked: the cells (in place: a tap's click lands on the cell it started on) and the detail
  function redrawGrid() {
    if (!box || !res || res.study !== "node" || busy) return;
    const view = res.street === "preflop" ? "strategy" : ST.view, ui = { ...ST, view };
    const g = box.querySelector(".sy-grid"), d = box.querySelector(".sy-detail");
    if (g) {
      const t = document.createElement("div"); t.innerHTML = gridHtml(res, ui);
      const now = t.firstElementChild;
      g.className = now.className;
      [...now.children].forEach((c, i) => { const o = g.children[i]; o.className = c.className; o.setAttribute("style", c.getAttribute("style")); o.innerHTML = c.innerHTML; });
    }
    if (d && view !== "breakdown") d.innerHTML = detailHtml(res, ui, ST.hover || ST.sel);
    box.querySelectorAll(".sy-acts tr[data-act]").forEach(r => r.classList.toggle("focus", +r.dataset.act === ST.focus));
  }
  function startFrom(spot) {
    ST.spot = spot;
    if (spot === "table") {
      const p = typeof globalThis.studyHand === "function" ? globalThis.studyHand() : null;
      return go(p ? fromPayload(p) : fresh());
    }
    const sp = spot && index && index.spots.find(s => s.id === spot);
    return go(sp ? fromSpot(sp) : fresh());
  }
  function stepBack() { const b = back(ST.state); if (b) go(b); }
  function act(d, e) {
    if (d.close != null) return close();
    if (d.speed) { if (d.speed !== ST.speed) { ST.speed = d.speed; return go(ST.state); } return; }
    if (d.back != null) return stepBack();
    if (d.step != null) return go(line(ST.state)[+d.step].st);
    if (d.view) { ST.view = d.view; store(); return draw(); }
    if (d.who) { ST.who = d.who; ST.sel = ST.hover = null; return draw(); }
    if (d.random != null && res && res.study === "deal") return go(deal(ST.state, randomCards(ST.state, res.street === "flop" ? 3 : 1)));
    if (d.act != null && res && res.study === "node" && !busy) return go(play(ST.state, res, res.actions[+d.act]));
    if (d.cls) { ST.sel = ST.sel === d.cls ? null : d.cls; ST.hover = null; return redrawGrid(); }
  }
  function close() {
    gen++; busy = false;
    if (box) box.hidden = true; const m = document.getElementById("menuBtn"); if (m) m.focus();
    if (globalThis.viewClosed) globalThis.viewClosed("study");
  }
  const CSS = `
  .sy { position: fixed; inset: 0; z-index: 90; background: rgba(0,0,0,.55); display: flex; justify-content: center; align-items: flex-start; padding: 28px 16px 16px; }
  .sy-card { background: var(--panel2); color: var(--text); border: 1px solid var(--line2); border-radius: var(--r); width: min(1120px, 100%); max-height: calc(100dvh - 44px); display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,.5); }
  .sy-head { display: flex; align-items: baseline; gap: 10px; padding: 10px 12px 6px; }
  .sy-sub { color: var(--muted); font-size: 12px; flex: 1; }
  .sy-x { margin-left: auto; padding: 2px 8px; }
  .sy-body { display: flex; flex-direction: column; gap: 8px; padding: 0 12px 12px; min-height: 0; overflow: auto; }
  .sy-ctl { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .sy-ctl select { background: var(--bg); color: var(--text); border: 1px solid var(--line); border-radius: 7px; padding: 5px 6px; font: inherit; font-size: 12px; max-width: 100%; }
  .sy-ctl .segment button { font-size: 12px; padding: 5px 9px; }
  .sy-line { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; font-size: 12px; }
  .sy-line button { font-size: 12px; padding: 3px 7px; border-radius: 6px; background: var(--panel); border: 1px solid var(--line); color: var(--text); }
  .sy-line button.sy-st { font-weight: 600; background: var(--bg); }
  .sy-line button:disabled { opacity: 1; color: var(--muted); }
  .sy-line .sy-now { color: var(--accent); font-weight: 600; padding: 3px 4px; }
  .sy-stage { position: relative; }
  .sy-stage.busy .sy-main, .sy-stage.busy .sy-status { opacity: .45; pointer-events: none; }
  .sy-wait { position: absolute; left: 50%; top: 40px; transform: translateX(-50%); z-index: 5; background: var(--panel); border: 1px solid var(--line2); border-radius: 8px; padding: 8px 14px; font-size: 13px; }
  .sy-status { font-size: 13px; color: var(--muted); margin-bottom: 6px; }
  .sy-status b { color: var(--text); }
  .sy-main { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 14px; align-items: start; }
  .sy-left .sy-ctl { margin-bottom: 6px; }
  .sy-grid { display: grid; grid-template-columns: repeat(13, 1fr); gap: 2px; max-width: 560px; }
  .sy-cell { aspect-ratio: 1; position: relative; background: #111418; border-radius: 3px; overflow: hidden; cursor: pointer; font-size: 10px; }
  .sy-cell.out { opacity: .35; }
  .sy-cell.sel { outline: 2px solid var(--accent); outline-offset: -1px; z-index: 2; }
  .sy-cell:hover { outline: 2px solid #fff; outline-offset: -1px; z-index: 3; }
  .sy-fill { position: absolute; left: 0; right: 0; bottom: 0; display: flex; pointer-events: none; }
  .sy-fill i { display: block; height: 100%; }
  .sy-l { position: absolute; left: 2px; top: 1px; color: #fff; text-shadow: 0 0 3px #000; font-weight: 600; pointer-events: none; z-index: 1; }
  .sy-cell.pair .sy-l { color: #ffe9b0; }
  .sy-v { position: absolute; right: 2px; bottom: 1px; font-size: 9px; color: #fff; opacity: .9; pointer-events: none; font-variant-numeric: tabular-nums; }
  .sy-acts { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
  .sy-acts th { color: var(--muted); font-weight: 500; text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--line2); font-size: 12px; }
  .sy-acts td { padding: 6px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  .sy-acts .n { text-align: right; }
  .sy-acts tbody tr { cursor: pointer; }
  .sy-acts tbody tr:hover td, .sy-acts tbody tr.focus td { background: var(--line); }
  .sy-sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; vertical-align: -1px; }
  .sy-mini { display: inline-flex; width: 46px; height: 7px; background: var(--line); border-radius: 2px; overflow: hidden; margin-right: 6px; vertical-align: 1px; }
  .sy-mini.wide { width: 100%; min-width: 70px; margin: 0; }
  .sy-mini span { display: block; height: 100%; }
  .sy-bar { display: flex; height: 10px; border-radius: 3px; overflow: hidden; background: var(--line); margin-top: 8px; }
  .sy-bar i { display: block; height: 100%; }
  .sy-tot { font-size: 12px; color: var(--muted); margin-top: 6px; line-height: 1.6; }
  .sy-tot b { color: var(--text); font-weight: 600; }
  .sy-detail { margin-top: 12px; border-top: 1px solid var(--line); padding-top: 8px; font-size: 12px; }
  .sy-dh { margin-bottom: 6px; }
  .sy-dh b { font-size: 13px; }
  .sy-mut, .sy-hint { color: var(--muted); }
  .sy-mix { display: flex; flex-wrap: wrap; gap: 4px 10px; margin-bottom: 8px; }
  .sy-mix span::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 2px; background: var(--c); margin-right: 5px; }
  .sy-mix small { color: var(--muted); }
  .sy-cb { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  .sy-cb th { color: var(--muted); font-weight: 500; text-align: left; padding: 2px 5px; font-size: 11px; }
  .sy-cb td { padding: 3px 5px; border-top: 1px solid var(--line); }
  .sy-cb .n { text-align: right; white-space: nowrap; }
  .sy-cb .sy-c { white-space: nowrap; font-weight: 600; }
  .sy-cb .sy-c small { color: var(--faint); font-weight: 400; margin-left: 4px; }
  .sy-note, .sy-msg { color: var(--muted); font-size: 11px; line-height: 1.5; margin-top: 10px; }
  .sy-msg { padding: 24px 4px; font-size: 13px; }
  .sy-err { color: var(--bad); }
  .sy-deal { padding: 18px 4px; font-size: 13px; display: flex; flex-direction: column; gap: 8px; }
  .sy-deal form { display: flex; gap: 6px; flex-wrap: wrap; }
  .sy-deal input { background: var(--bg); color: var(--text); border: 1px solid var(--line); border-radius: 7px; padding: 6px 8px; font: inherit; width: 150px; text-transform: none; }
  .sy-bd { min-height: 200px; }
  @media (max-width: 760px) {
    .sy { padding: 0; }
    .sy-card { max-height: 100dvh; height: 100dvh; border-radius: 0; border: 0; width: 100%; }
    .sy-sub { display: none; }
    .sy-main { grid-template-columns: minmax(0, 1fr); }
    .sy-cell { font-size: 7px; }
    .sy-l { left: 1px; top: 0; }
    .sy-v { display: none; }
    .sy-ctl select { font-size: 16px; flex: 1 1 100%; }
    .sy-deal input { font-size: 16px; }
  }`;
  async function open() {
    if (!document.getElementById("syStyle")) {
      const s = document.createElement("style"); s.id = "syStyle"; s.textContent = CSS; document.head.appendChild(s);
    }
    if (!box) {
      box = document.createElement("div"); box.id = "studyBox"; box.className = "sy"; box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true"); box.setAttribute("aria-label", "Study");
      box.innerHTML = `<div class="sy-card"><div class="sy-head"><b>Study</b><span class="sy-sub">the solver's strategy for a whole range, node by node: click an action to play it</span><button class="ghost sy-x" data-close aria-label="Close">✕</button></div><div class="sy-body"></div></div>`;
      document.body.appendChild(box);
      const data = (t) => { const e = t.closest("[data-close],[data-speed],[data-back],[data-step],[data-view],[data-who],[data-random],[data-act],[data-cls]"); return e && !e.disabled ? e.dataset : null; };
      box.addEventListener("click", (e) => { if (e.target === box) return close(); const d = data(e.target); if (d) act(d, e); });
      box.addEventListener("submit", (e) => {
        e.preventDefault();
        if (!res || res.study !== "deal") return;
        const r = readCards(ST.state, e.target.elements.cards.value, res.street === "flop" ? 3 : 1);
        if (r.error) { const m = box.querySelector("[data-dealerr]"); if (m) m.textContent = r.error; return; }
        go(deal(ST.state, r.cards));
      });
      box.addEventListener("change", (e) => { if (e.target.dataset.ctl === "spot") startFrom(e.target.value); });
      box.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { e.stopPropagation(); close(); }
        else if (e.key === "Backspace" && !/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) { e.preventDefault(); stepBack(); }
        else if ((e.key === "Enter" || e.key === " ") && e.target.matches("tr[data-act]")) { e.preventDefault(); act(e.target.dataset); }
      });
      // hover: a hand's combos, an action's hands
      box.addEventListener("mouseover", (e) => {
        const c = e.target.closest(".sy-cell"), a = e.target.closest("tr[data-act]");
        const h = c ? c.dataset.cls : null, f = a ? +a.dataset.act : null;
        if (h !== ST.hover || f !== ST.focus) { ST.hover = h; ST.focus = f; redrawGrid(); }
      });
      box.addEventListener("mouseleave", () => { ST.hover = ST.focus = null; redrawGrid(); });
    }
    box.hidden = false; res = null; err = null; draw();
    try { index = index || await fetch("flop_library/index.json", { cache: "no-cache" }).then(r => r.ok ? r.json() : null); } catch (e) { index = null; }
    if (ST.spot === "table") ST.spot = "";
    await go(ST.state && ST.state.preflop ? ST.state : (index && index.spots.length ? (ST.spot = index.spots[0].id, fromSpot(index.spots[0])) : fresh()));
    const t = box.querySelector("select, button"); if (t) t.focus();
  }

  const api = { fresh, parseLine, fromSpot, fromPayload, streetOf, request, kindOf, play, deal, randomCards, readCards, line, back, label,
                eqr, cellValue, gridHtml, actionsHtml, detailHtml, lineHtml, statusHtml, solverHtml, classAt, combosOf,
                open, close, state: ST };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Study = api;
})();
