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
  // ---- the range builder: the actor's strategy painted hand by hand, then scored against the solver's. A paint is
  // {f: {class: mix}, score}; a brush plays action a at share t of a hand, the rest of its mix keeping its proportions
  const AMOUNTS = [1, 0.75, 0.5, 0.25];
  // what an unpainted hand plays: fold, else check or call
  const passiveOf = (names) => { const i = names.indexOf("fold"); return i >= 0 ? i : Math.max(0, names.findIndex(n => n === "check" || n === "call")); };
  function blankPaint(res) {
    const g = res.grid, d = passiveOf(g.actions), f = {};
    for (const [cls, c] of Object.entries(g.cells)) if (c.w > 0) f[cls] = g.actions.map((_, a) => a === d ? 1 : 0);
    return { f, score: null };
  }
  function brush(mix, a, t, d) {
    const out = mix.map((x, i) => i === a ? 0 : x), rest = out.reduce((s, x) => s + x, 0);
    if (rest > 1e-9) for (let i = 0; i < out.length; i++) out[i] *= (1 - t) / rest;
    else if (t < 1 && a !== d) out[d] = 1 - t;
    else if (t < 1) for (let i = 0; i < out.length; i++) if (i !== a) out[i] = (1 - t) / (out.length - 1);
    out[a] = t;
    return out;
  }
  // a paint against the solver's mix, every hand by its combos in the range: accuracy (one minus the share of the
  // range played differently), the EV given up (bb a combo, where the answer has EVs: not preflop), each action's
  // share of the range both ways, and per hand its difference and loss
  function scoreBuild(res, paint) {
    const g = res.grid, n = g.actions.length, mine = new Array(n).fill(0), gto = new Array(n).fill(0), cells = {};
    let W = 0, off = 0, loss = 0, evs = true;
    for (const [cls, c] of Object.entries(g.cells)) {
      if (!(c.w > 0)) continue;
      const m = paint.f[cls] || g.actions.map((_, a) => a === passiveOf(g.actions) ? 1 : 0), wc = c.w * combosOf(cls);
      const d = 0.5 * m.reduce((s, x, a) => s + Math.abs(x - c.f[a]), 0);
      let l = null;
      if (c.ev) { const ev = (f) => f.reduce((s, x, a) => s + x * c.ev[a], 0); l = Math.max(0, ev(c.f) - ev(m)); loss += wc * l; } else evs = false;
      cells[cls] = { off: d, loss: l, combos: wc };
      W += wc; off += wc * d;
      m.forEach((x, a) => { mine[a] += wc * x; gto[a] += wc * c.f[a]; });
    }
    const per = (x) => W ? x / W : 0;
    return { acc: W ? 1 - off / W : 1, ev_loss_bb: evs && W ? loss / W : null, ev_loss_pct: evs && W && res.pot_bb ? 100 * loss / W / res.pot_bb : null,
             mine: mine.map(per), gto: gto.map(per), combos: W, cells };
  }
  const grade = (acc) => acc >= 0.9 ? "Excellent" : acc >= 0.8 ? "Good" : acc >= 0.65 ? "Fair" : "Off";
  const segsHtml = (f, names) => names.map((n, a) => f[a] > 0.0005 ? `<i style="width:${(f[a] * 100).toFixed(1)}%;background:${color(n)}"></i>` : "").join("");
  // the grid while building: each hand's painted mix (as tall as its share of the class); once scored, the solver's
  // mix above yours, outlined by how far apart they are
  function buildGridHtml(res, paint, ui) {
    const g = res.grid, names = g.actions, sc = paint.score;
    let h = `<div class="sy-grid sy-build${sc ? " scored" : ""}">`;
    for (let i = 0; i < 13; i++) for (let j = 0; j < 13; j++) {
      const cls = classAt(i, j), c = g.cells[cls], on = c && c.w > 0, m = paint.f[cls];
      let inner = "", k = "";
      if (on && sc) {
        inner = `<span class="sy-half top">${segsHtml(c.f, names)}</span><span class="sy-half bot">${segsHtml(m, names)}</span>`;
        const d = sc.cells[cls].off;
        k = d >= 0.5 ? " e2" : d >= 0.2 ? " e1" : "";
      } else if (on) inner = `<span class="sy-fill" style="height:${(Math.min(1, c.w) * 100).toFixed(1)}%">${segsHtml(m, names)}</span>`;
      h += `<div class="sy-cell${i === j ? " pair" : ""}${on ? "" : " out"}${k}${ui.sel === cls ? " sel" : ""}" data-cls="${cls}">${inner}<span class="sy-l">${cls}</span></div>`;
    }
    return h + `</div>`;
  }
  function buildPanelHtml(res, paint, ui) {
    const names = res.grid.actions, sc = paint.score, now = sc || scoreBuild(res, paint);
    const bar = (f) => `<div class="sy-bar">${names.map((n, a) => f[a] > 0.0005 ? `<i style="width:${(f[a] * 100).toFixed(2)}%;background:${color(n)}" title="${esc(n)} ${pct(f[a], 1)}"></i>` : "").join("")}</div>`;
    if (!sc) {
      let h = `<div class="sy-bh">Paint ${esc(res.actor)}'s strategy: pick an action, then click or drag over hands. Unpainted hands ${esc(names[passiveOf(names)])}.</div>`;
      h += `<div class="sy-brushes">${names.map((n, a) => `<button data-brush="${a}" class="${a === ui.brush ? "active" : ""}" title="Key ${a + 1}"><i class="sy-sw" style="background:${color(n)}"></i>${esc(n)}</button>`).join("")}</div>`;
      h += `<div class="sy-ctl"><span class="segment">${AMOUNTS.map(t => `<button data-amt="${t}" class="${t === ui.amt ? "active" : ""}">${t * 100}%</button>`).join("")}</span>`
        + `<button class="ghost" data-fill title="Paint every hand with this brush">Fill all</button><button class="ghost" data-reset>Reset</button></div>`;
      h += `<div class="sy-tot">Your range: ${names.map((n, a) => `${esc(n)} <b data-mine="${a}">${pct(now.mine[a], 1)}</b>`).join(" · ")}</div><div data-minebar>${bar(now.mine)}</div>`;
      return h + `<div class="sy-ctl sy-go"><button data-score>Check my range</button><button class="ghost" data-build>Show the solution</button></div>`;
    }
    let h = `<div class="sy-score"><b>${pct(sc.acc, 1)}</b> accurate · ${grade(sc.acc)}`
      + (sc.ev_loss_bb != null ? `<br><span class="sy-mut">EV given up: <b>${sc.ev_loss_bb.toFixed(3)}</b>bb a combo (${sc.ev_loss_pct.toFixed(2)}% of the pot)</span>` : "") + `</div>`;
    h += `<table class="sy-acts"><thead><tr><th>Action</th><th class="n">Yours</th><th class="n">Solver</th></tr></thead><tbody>`
      + names.map((n, a) => `<tr><td><i class="sy-sw" style="background:${color(n)}"></i>${esc(n)}</td><td class="n">${pct(sc.mine[a], 1)}</td><td class="n">${pct(sc.gto[a], 1)}</td></tr>`).join("")
      + `</tbody></table><div class="sy-mut sy-key">Each hand: the solver's mix above, yours below; outlined where they differ by 20% or more (red: 50%).</div>`;
    const miss = Object.entries(sc.cells).filter(([, x]) => x.off >= 0.05).sort((a, b) => b[1].off * b[1].combos - a[1].off * a[1].combos).slice(0, 6);
    if (miss.length) h += `<div class="sy-miss"><div class="sy-mut">Biggest differences:</div>${miss.map(([cls, x]) => `<button data-cls="${cls}" class="ghost">${cls} <small>${pct(x.off)}${x.loss != null ? ` · −${x.loss.toFixed(2)}bb` : ""}</small></button>`).join("")}</div>`;
    if (ui.tally && ui.tally.n) h += `<div class="sy-mut sy-key">This session: ${ui.tally.n} range${ui.tally.n > 1 ? "s" : ""}, ${pct(ui.tally.acc / ui.tally.n, 1)} accurate on average.</div>`;
    return h + `<div class="sy-ctl sy-go"><button data-retry>Try again</button>${cardsOf(res.board || "").length === 3 ? `<button class="ghost" data-newflop title="The same spot on a random flop">New flop</button>` : ""}<button class="ghost" data-build>Show the solution</button></div>`;
  }
  // a hand's detail while building: your mix, and once scored the solver's with each action's EV
  function buildDetailHtml(res, paint, cls) {
    if (!cls) return `<div class="sy-hint">${paint.score ? "Hover or click a hand to compare." : "Hover a hand for its mix."}</div>`;
    const g = res.grid, c = g.cells[cls], names = g.actions, m = paint.f[cls];
    if (!c || !(c.w > 0)) return `<div class="sy-dh"><b>${cls}</b> <span class="sy-mut">not in ${esc(res.actor)}'s range here</span></div>`;
    const mix = (f, ev) => `<div class="sy-mix">${names.map((nm, a) => f[a] >= 0.005 ? `<span style="--c:${color(nm)}">${esc(nm)} <b>${pct(f[a])}</b>${ev ? ` <small>${sgn(ev[a])}</small>` : ""}</span>` : "").join("")}</div>`;
    let h = `<div class="sy-dh"><b>${cls}</b> <span class="sy-mut">${num(c.w * combosOf(cls))} of ${combosOf(cls)} combos</span></div><div class="sy-mut">Yours</div>${mix(m)}`;
    if (paint.score) {
      const x = paint.score.cells[cls];
      h += `<div class="sy-mut">Solver${c.ev ? " (EV of each action)" : ""}</div>${mix(c.f, c.ev)}<div class="sy-mut">${pct(x.off)} of it played differently${x.loss != null ? `, ${x.loss.toFixed(3)}bb a combo given up` : ""}</div>`;
    }
    return h;
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
  const ST = { state: null, spot: "", speed: typeof SPEED === "string" && SPEED !== "quick" ? SPEED : "fast", view: "strategy", who: "me", sel: null, hover: null, focus: null,
               build: false, brush: 0, amt: 1, tally: { n: 0, acc: 0 } };
  try { const s = JSON.parse(localStorage.getItem("gto-study")) || {}; for (const k of ["state", "spot", "speed", "view", "build"]) if (s[k] != null) ST[k] = s[k]; } catch (e) {}
  const store = () => { try { localStorage.setItem("gto-study", JSON.stringify({ state: ST.state, spot: ST.spot, speed: ST.speed, view: ST.view, build: ST.build })); } catch (e) {} };
  // the range builder's paints, by the answer they're of
  const paints = new Map();
  const building = () => ST.build && res && res.study === "node" && !busy;
  const paintOf = (r) => { if (!paints.has(r.key)) paints.set(r.key, blankPaint(r)); return paints.get(r.key); };
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
    if (res && res.study === "node" && ST.brush >= res.grid.actions.length) ST.brush = 0;
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
    const buildBtn = `<button class="ghost${ST.build ? " active" : ""}" data-build title="Paint the strategy yourself, then score it (B)">${ST.build ? "✎ Building" : "✎ Build"}</button>`;
    if (ST.build) {
      const paint = paintOf(res), ui = { ...ST };
      return `<div class="sy-status">${statusHtml(res)}</div><div class="sy-main"><div class="sy-left"><div class="sy-ctl">${buildBtn}</div>`
        + buildGridHtml(res, paint, ui) + `</div><div class="sy-right"><div class="sy-bpanel">${buildPanelHtml(res, paint, ui)}</div>`
        + `<div class="sy-detail">${buildDetailHtml(res, paint, cls)}</div></div></div><div class="sy-note">${solverHtml(res)}</div>`;
    }
    const views = pre ? ["strategy"] : [...Object.keys(VIEWS), "breakdown"];
    const view = views.includes(ST.view) ? ST.view : "strategy";
    const ui = { ...ST, view };
    let h = `<div class="sy-status">${statusHtml(res)}</div><div class="sy-main"><div class="sy-left"><div class="sy-ctl">`
      + `<span class="segment">${views.map(v => `<button data-view="${v}" class="${v === view ? "active" : ""}">${VIEWS[v] || "Breakdown"}</button>`).join("")}</span>`
      + (res.opponent_grid && view !== "breakdown" ? `<span class="segment">${["me", "opp"].map(w => `<button data-who="${w}" class="${w === ST.who ? "active" : ""}">${esc(w === "me" ? res.actor : res.opponent)}</button>`).join("")}</span>` : "")
      + buildBtn + `</div>`;
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
    // (a redraw drops the focused button: the panel keeps the focus, so its keys still work)
    if (!box.hidden && !box.contains(document.activeElement)) box.querySelector(".sy-card").focus({ preventScroll: true });
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
    if (ST.build) {
      const paint = paintOf(res), g = box.querySelector(".sy-grid"), d = box.querySelector(".sy-detail");
      if (g) {
        const t = document.createElement("div"); t.innerHTML = buildGridHtml(res, paint, ST);
        [...t.firstElementChild.children].forEach((c, i) => { const o = g.children[i]; o.className = c.className; o.innerHTML = c.innerHTML; });
      }
      if (d) d.innerHTML = buildDetailHtml(res, paint, ST.hover || ST.sel);
      return;
    }
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
  // a hand painted with the brush (not once scored): its cell and the range's totals redrawn in place
  function paintCell(cls) {
    if (!building()) return;
    const paint = paintOf(res), c = res.grid.cells[cls];
    if (paint.score || !c || !(c.w > 0)) return;
    paint.f[cls] = brush(paint.f[cls], ST.brush, ST.amt, passiveOf(res.grid.actions));
    redrawGrid();
    const now = scoreBuild(res, paint);
    box.querySelectorAll("[data-mine]").forEach(b => { b.textContent = pct(now.mine[+b.dataset.mine], 1); });
    const t = document.createElement("div"); t.innerHTML = buildPanelHtml(res, paint, ST);
    const mb = box.querySelector("[data-minebar]"), nb = t.querySelector("[data-minebar]");
    if (mb && nb) mb.innerHTML = nb.innerHTML;
  }
  function newFlop() {
    const base = { ...ST.state, board: "", flop: [], turn: [], river: [] };
    return go(deal(base, randomCards(base, 3)));
  }
  function act(d, e) {
    if (d.close != null) return close();
    if (d.speed) { if (d.speed !== ST.speed) { ST.speed = d.speed; return go(ST.state); } return; }
    if (d.back != null) return stepBack();
    if (d.step != null) return go(line(ST.state)[+d.step].st);
    if (d.view) { ST.view = d.view; store(); return draw(); }
    if (d.who) { ST.who = d.who; ST.sel = ST.hover = null; return draw(); }
    if (d.random != null && res && res.study === "deal") return go(deal(ST.state, randomCards(ST.state, res.street === "flop" ? 3 : 1)));
    if (d.act != null && res && res.study === "node" && !busy) return go(play(ST.state, res, res.actions[+d.act]));
    if (d.build != null) { ST.build = !ST.build; ST.sel = ST.hover = ST.focus = null; store(); return draw(); }
    if (building()) {
      const paint = paintOf(res);
      if (d.brush != null) { ST.brush = +d.brush; return draw(); }
      if (d.amt != null) { ST.amt = +d.amt; return draw(); }
      if (d.reset != null) { paints.set(res.key, blankPaint(res)); return draw(); }
      if (d.fill != null) { for (const k of Object.keys(paint.f)) paint.f[k] = brush(paint.f[k], ST.brush, ST.amt, passiveOf(res.grid.actions)); return draw(); }
      if (d.score != null) { paint.score = scoreBuild(res, paint); ST.tally.n++; ST.tally.acc += paint.score.acc; return draw(); }
      if (d.retry != null) { paint.score = null; return draw(); }
      if (d.newflop != null) return newFlop();
      if (d.cls) { if (paint.score) { ST.sel = ST.sel === d.cls ? null : d.cls; ST.hover = null; redrawGrid(); } return; }
    }
    if (d.cls) { ST.sel = ST.sel === d.cls ? null : d.cls; ST.hover = null; return redrawGrid(); }
  }
  function close() {
    gen++; busy = false;
    if (box) box.hidden = true; const m = document.getElementById("menuBtn"); if (m) m.focus();
    if (globalThis.viewClosed) globalThis.viewClosed("study");
  }
  const CSS = `
  .sy { position: fixed; inset: 0; z-index: 90; background: rgba(0,0,0,.55); display: flex; justify-content: center; align-items: flex-start; padding: 28px 16px 16px; }
  .sy-card:focus { outline: none; }
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
  .sy-build { touch-action: none; user-select: none; -webkit-user-select: none; }
  .sy-build .sy-cell { cursor: crosshair; }
  .sy-build.scored .sy-cell { cursor: pointer; }
  .sy-half { position: absolute; left: 0; right: 0; height: 50%; display: flex; pointer-events: none; }
  .sy-half.top { top: 0; } .sy-half.bot { bottom: 0; border-top: 1px solid rgba(0,0,0,.6); }
  .sy-half i { display: block; height: 100%; }
  .sy-cell.e1 { box-shadow: inset 0 0 0 2px #e0b341; }
  .sy-cell.e2 { box-shadow: inset 0 0 0 2px var(--bad); }
  .sy-bh { font-size: 13px; color: var(--muted); margin-bottom: 8px; line-height: 1.5; }
  .sy-brushes { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
  .sy-brushes button { font-size: 13px; padding: 6px 10px; background: var(--panel); border: 1px solid var(--line); color: var(--text); border-radius: 7px; }
  .sy-brushes button.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .sy-go { margin-top: 12px; }
  .sy-score { font-size: 14px; margin-bottom: 10px; line-height: 1.6; }
  .sy-score > b { font-size: 22px; }
  .sy-key { font-size: 11px; margin-top: 6px; line-height: 1.5; }
  .sy-miss { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin-top: 8px; font-size: 12px; }
  .sy-miss button { font-size: 12px; padding: 3px 7px; }
  .sy-miss small { color: var(--muted); }
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
      box.innerHTML = `<div class="sy-card" tabindex="-1"><div class="sy-head"><b>Study</b><span class="sy-sub">the solver's strategy for a whole range, node by node: click an action to play it</span><button class="ghost sy-x" data-close aria-label="Close">✕</button></div><div class="sy-body"></div></div>`;
      document.body.appendChild(box);
      const data = (t) => { const e = t.closest("[data-close],[data-speed],[data-back],[data-step],[data-view],[data-who],[data-random],[data-act],[data-cls],[data-build],[data-brush],[data-amt],[data-reset],[data-fill],[data-score],[data-retry],[data-newflop]"); return e && !e.disabled ? e.dataset : null; };
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
        else if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
        else if (e.key === "b" || e.key === "B") { e.stopPropagation(); act({ build: "" }); }
        else if (building() && /^[1-9]$/.test(e.key) && +e.key <= res.grid.actions.length) { e.stopPropagation(); act({ brush: String(+e.key - 1) }); }
      });
      // painting: a press on a hand paints it, and a drag (mouse, pen or finger) every hand on its path (in steps of
      // a few pixels between two moves: a quick drag skips cells)
      let painting = null;
      const paintAt = (x, y, done) => {
        const el = document.elementFromPoint(x, y), c = el && el.closest(".sy-build .sy-cell");
        if (c && !done.has(c.dataset.cls)) { done.add(c.dataset.cls); paintCell(c.dataset.cls); }
      };
      box.addEventListener("pointerdown", (e) => {
        const c = e.target.closest(".sy-build .sy-cell");
        if (!c || !building() || paintOf(res).score) return;
        e.preventDefault(); painting = { x: e.clientX, y: e.clientY, done: new Set() }; paintAt(e.clientX, e.clientY, painting.done);
      });
      box.addEventListener("pointermove", (e) => {
        if (!painting) return;
        const { x, y } = painting, n = Math.max(1, Math.ceil(Math.hypot(e.clientX - x, e.clientY - y) / 6));
        for (let i = 1; i <= n; i++) paintAt(x + (e.clientX - x) * i / n, y + (e.clientY - y) * i / n, painting.done);
        painting.x = e.clientX; painting.y = e.clientY;
      });
      window.addEventListener("pointerup", () => { painting = null; });
      window.addEventListener("pointercancel", () => { painting = null; });
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
    if (globalThis.STUDY_BUILD) {   // (the menu's Range builder: a random flop of the spot, building)
      globalThis.STUDY_BUILD = false; ST.build = true;
      const sp = index && (index.spots.find(s => s.id === ST.spot) || index.spots[0]);
      if (sp) { ST.spot = sp.id; ST.state = fromSpot(sp); await newFlop(); const t = box.querySelector("select, button"); if (t) t.focus(); return; }
    }
    await go(ST.state && ST.state.preflop ? ST.state : (index && index.spots.length ? (ST.spot = index.spots[0].id, fromSpot(index.spots[0])) : fresh()));
    const t = box.querySelector("select, button"); if (t) t.focus();
  }

  const api = { fresh, parseLine, fromSpot, fromPayload, streetOf, request, kindOf, play, deal, randomCards, readCards, line, back, label,
                eqr, cellValue, gridHtml, actionsHtml, detailHtml, lineHtml, statusHtml, solverHtml, classAt, combosOf,
                passiveOf, blankPaint, brush, scoreBuild, grade, buildGridHtml, buildPanelHtml, buildDetailHtml,
                open, close, state: ST };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Study = api;
})();
