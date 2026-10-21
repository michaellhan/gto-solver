// Drills, loaded by index.html the first time "Drills" is opened from the ⋯ menu (after reports.js, whose texture
// classification the filter uses): one precomputed spot's decision drilled hand after hand, answered at once from
// the flop library. A flop is drawn by weight (each stands for its share of the 22100) among the spot's flops that
// store the node and pass the texture filter, a hand from the acting player's range there (its weight entering the
// flop, off the board, times its own frequency of the line so far), in a random suit relabelling. The player picks
// an action; the stored mix and EVs for that hand grade it. The entry (<spot>/<flop>.bin, gto/flop_library.py's
// encode) is decoded here; the weights entering the flop are <spot>/ranges.json. The pure parts (decoding, the
// deal, grading, the score, keys) are what the Node tests call.
(function () {
  "use strict";
  const RANKS = "23456789TJQKA", SUITS = "cdhs";
  const cardStr = (c) => RANKS[c >> 2] + SUITS[c & 3];
  const cardInt = (s) => RANKS.indexOf(s[0]) * 4 + SUITS.indexOf(s[1]);
  const COMBOS = [];                                  // cards.ALL_COMBOS: (a, b), a < b, in order
  for (let a = 0; a < 52; a++) for (let b = a + 1; b < 52; b++) COMBOS.push([a, b]);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const FR = () => globalThis.FlopReports || require("./reports.js");

  // ---- entries (flop_library.pack, format 2: MAGIC, then zlib of a u32 header length, the JSON header, the body).
  // A value per suit-isomorphic combo: the combos the canonical flop's suit stabilizer maps to each other share it
  // (each class stored once, at its least combo index; per node in hand order: ranks high to low, suited first)
  const MAGIC = "GTOFL3", FORMAT = 2;
  async function inflate(u8) {
    if (typeof DecompressionStream === "function") {
      const s = new Blob([u8]).stream().pipeThrough(new DecompressionStream("deflate"));
      return new Uint8Array(await new Response(s).arrayBuffer());
    }
    return new Uint8Array(require("node:zlib").inflateSync(u8));
  }
  const CIDX = new Map(COMBOS.map(([a, b], i) => [a * 52 + b, i]));
  const PERMS = [];
  (function perms(p, rest) { if (!rest.length) PERMS.push(p); rest.forEach((x, i) => perms([...p, x], rest.filter((_, j) => j !== i))); })([], [0, 1, 2, 3]);
  // rep[i]: the least combo index of combo i's class under the board's suit stabilizer (flop_library.iso_reps)
  function isoReps(board) {
    const key = [...board].sort((x, y) => x - y).join(), rep = COMBOS.map((_, i) => i);
    for (const p of PERMS) {
      if (board.map(c => (c >> 2) * 4 + p[c & 3]).sort((x, y) => x - y).join() !== key) continue;
      COMBOS.forEach(([a, b], i) => { const x = (a >> 2) * 4 + p[a & 3], y = (b >> 2) * 4 + p[b & 3], j = CIDX.get(Math.min(x, y) * 52 + Math.max(x, y)); if (j < rep[i]) rep[i] = j; });
    }
    return rep;
  }
  // combo indices in hand order (flop_library._hand_order): higher rank, lower rank (high to low), suited first, index
  const handOrder = (idx) => idx.slice().sort((i, j) => {
    const [a, b] = COMBOS[i], [c, d] = COMBOS[j];
    return (d >> 2) - (b >> 2) || (c >> 2) - (a >> 2) || ((a & 3) !== (b & 3)) - ((c & 3) !== (d & 3)) || i - j;
  });
  // the inflated body -> {head, combos: [oop, ip] (combo indices in each range, ascending), nodes: [{path, player,
  // names, fold0, freqs {name: range frequency}, strat [A][k], evs [A][k]}]}: k the acting player's combos, in the
  // order of combos[player]
  function parse(raw) {
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength), hl = dv.getUint32(0, true);
    const head = JSON.parse(new TextDecoder().decode(raw.subarray(4, 4 + hl)));
    if (head.format !== FORMAT) throw new Error(`flop library entry format ${head.format}`);
    const Q = head.q, [step, lin, base] = head.ev, dValue = (c) => c < lin ? c * step : lin * step * Math.pow(base, c - lin);
    const rep = isoReps(cardsOf(head.board).map(cardInt)), own = rep.flatMap((r, i) => r === i ? [i] : []);
    let o = 4 + hl;
    const nb = Math.ceil(own.length / 8), masks = [], combos = [[], []];
    for (const p of [0, 1]) {
      const m = new Uint8Array(COMBOS.length);
      own.forEach((c, k) => { m[c] = (raw[o + (k >> 3)] >> (7 - (k & 7))) & 1; });
      o += nb;
      masks.push(m);
      for (let i = 0; i < COMBOS.length; i++) if (m[rep[i]]) combos[p].push(i);
    }
    const nodes = head.nodes.map((nd) => {
      const stored = handOrder(own.filter(c => masks[nd.player][c])), pos = new Map(stored.map((c, k) => [c, k]));
      const col = combos[nd.player].map(i => pos.get(rep[i]));
      const k = stored.length, A = nd.names.length, strat = [], evs = [];
      for (let a = 0; a < A - 1; a++) { strat.push(Array.from(raw.subarray(o, o + k))); o += k; }
      strat.push(strat.length ? strat[0].map((_, j) => Math.max(Q - strat.reduce((s, r) => s + r[j], 0), 0)) : new Array(k).fill(Q));
      for (let j = 0; j < k; j++) { const t = strat.reduce((s, r) => s + r[j], 0) || 1; for (const r of strat) r[j] /= t; }
      const v = new Array(k);
      for (let j = 0; j < k; j++) { const x = raw[o + j] | (raw[o + k + j] << 8); v[j] = (x > 32767 ? x - 65536 : x) * step; }
      o += 2 * k;
      if (nd.fold0) evs.push(new Array(k).fill(0));
      for (let r = nd.fold0 ? 1 : 0; r < A; r++) { evs.push(v.map((x, j) => x - dValue(raw[o + j]))); o += k; }
      const freqs = Object.fromEntries(nd.names.map((n, a) => [n, nd.freqs[a]]));
      return { ...nd, freqs, strat: strat.map(r => col.map(c => r[c])), evs: evs.map(r => col.map(c => r[c])) };
    });
    return { head, combos, nodes };
  }
  async function decode(buf) {
    const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    if (new TextDecoder().decode(u8.subarray(0, MAGIC.length)) !== MAGIC) throw new Error("not a flop library entry");
    return parse(await inflate(u8.subarray(MAGIC.length)));
  }
  const nodeAt = (e, path) => e.nodes.find(n => n.path.join("|") === path.join("|")) || null;

  // ---- the drills: a spot's node, as the reports' views (out of position first; in position after a check; out of
  // position facing a bet after checking, any stored size, drawn by how often in position bets it)
  const VIEWS = { oop: (s) => `${s.seats[0]} first to act`, ip: (s) => `${s.seats[1]} after ${s.seats[0]} checks`,
                  vsbet: (s) => `${s.seats[0]} checks, faces a bet` };
  // the (flop, node) pairs a drill draws from, with their weights: [{f, path, w}]
  function candidates(rep, view, filter) {
    const out = [];
    for (const fl of rep.flops) {
      if (!Object.entries(filter || {}).every(([k, v]) => !v || FR().texture(fl.f)[k] === v)) continue;
      if (view === "oop" && fl.n[""]) out.push({ f: fl.f, path: [], w: fl.w });
      else if (view === "ip" && fl.n.check) out.push({ f: fl.f, path: ["check"], w: fl.w });
      else if (view === "vsbet") {
        const ks = Object.keys(fl.n).filter(k => k.startsWith("check|")), cn = rep.nodes.check || [];
        const fq = ks.map(k => fl.n.check ? fl.n.check[cn.indexOf(k.slice(6))] || 0 : 0), t = fq.reduce((a, b) => a + b, 0);
        ks.forEach((k, i) => out.push({ f: fl.f, path: k.split("|"), w: fl.w * (t > 0 ? fq[i] / t : 1 / ks.length) }));
      }
    }
    return out.filter(c => c.w > 0);
  }
  function pick(ws, rng) {
    const t = ws.reduce((a, b) => a + b, 0);
    let x = rng() * t;
    for (let i = 0; i < ws.length; i++) { x -= ws[i]; if (x < 0 && ws[i] > 0) return i; }
    for (let i = ws.length - 1; i >= 0; i--) if (ws[i] > 0) return i;
    return -1;
  }
  // the acting player's hand weights at a node (by column of combos[player]): the range entering the flop (1/10000),
  // none touching the board, times the player's own frequency of each earlier action on the path
  function handWeights(e, path, ranges) {
    const nd = nodeAt(e, path);
    if (!nd) return null;
    const p = nd.player, board = new Set(cardsOf(e.head.board).map(cardInt));
    const w = e.combos[p].map(ci => board.has(COMBOS[ci][0]) || board.has(COMBOS[ci][1]) ? 0 : ranges.w[p][ci] / 10000);
    for (let i = 0; i < path.length; i++) {
      const anc = nodeAt(e, path.slice(0, i));
      if (anc && anc.player === p) { const a = anc.names.indexOf(path[i]); w.forEach((x, j) => w[j] = x * anc.strat[a][j]); }
    }
    return w;
  }
  const cardsOf = (f) => [f.slice(0, 2), f.slice(2, 4), f.slice(4, 6)];
  const relabel = (c, perm) => c[0] + SUITS[perm[SUITS.indexOf(c[1])]];
  function shuffle(rng) { const p = [0, 1, 2, 3]; for (let i = 3; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; }
  // pot at the node and the bet faced (bb): the bet sizes are fractions of the pot ("bet 33%")
  function potAt(rep, path) {
    const b = path.find(x => /^bet /.test(x)), bet = b ? Math.round(parseFloat(b.slice(4)) / 100 * rep.pot * 100) / 100 : 0;
    return { pot: Math.round((rep.pot + bet) * 100) / 100, bet };
  }
  // one problem: the hand dealt at a drawn (flop, node); null if the entry has no hand there
  function problem(rep, e, cand, ranges, rng) {
    const nd = nodeAt(e, cand.path), w = handWeights(e, cand.path, ranges);
    if (!nd || !w) return null;
    const j = pick(w, rng);
    if (j < 0) return null;
    const perm = shuffle(rng), [a, b] = COMBOS[e.combos[nd.player][j]];
    const hand = [cardStr(b), cardStr(a)].map(c => relabel(c, perm));
    const { pot, bet } = potAt(rep, cand.path);
    return { spot: rep.id, flop: cand.f, path: cand.path, player: nd.player, hero: rep.seats[nd.player],
             board: cardsOf(cand.f).map(c => relabel(c, perm)), hand, combo: COMBOS[e.combos[nd.player][j]], names: nd.names,
             freqs: nd.strat.map(r => r[j]), evs: nd.evs.map(r => r[j]), pot, bet };
  }

  // ---- grading, as GTO Wizard's trainer: an action is correct when it's the highest-EV one (within the library's
  // EV step) or the hand plays it MIX of the time or more; the EV given up is against the best action
  const MIX = 0.1, TOL = 0.011;
  function grade(pr, a) {
    const best = Math.max(...pr.evs), loss = Math.max(0, best - pr.evs[a]), lossPct = loss / pr.pot * 100;
    const mixed = pr.freqs.filter(f => f >= MIX).length > 1;
    const verdict = loss <= TOL ? "Best" : pr.freqs[a] >= MIX ? "Correct" : lossPct < 2 ? "Inaccuracy" : lossPct < 8 ? "Mistake" : "Blunder";
    return { a, best: pr.evs.indexOf(best), loss: loss <= TOL ? 0 : loss, lossPct: loss <= TOL ? 0 : lossPct, ok: verdict === "Best" || verdict === "Correct", mixed, verdict };
  }
  // a score: {n, ok, loss (bb, summed), lossPct (summed), streak, best, recent (the last 100 as 0/1)}
  const emptyScore = () => ({ n: 0, ok: 0, loss: 0, lossPct: 0, streak: 0, best: 0, recent: [] });
  function addScore(s, g) {
    const s2 = { ...emptyScore(), ...s };
    const streak = g.ok ? s2.streak + 1 : 0;
    return { n: s2.n + 1, ok: s2.ok + (g.ok ? 1 : 0), loss: s2.loss + g.loss, lossPct: s2.lossPct + g.lossPct, streak,
             best: Math.max(s2.best, streak), recent: [...s2.recent, g.ok ? 1 : 0].slice(-100) };
  }
  const summary = (s) => s.n ? { acc: s.ok / s.n, loss: s.loss / s.n, lossPct: s.lossPct / s.n } : { acc: 0, loss: 0, lossPct: 0 };
  // keys: 1-9 answer, Space/Enter next once answered, Escape closes
  function keyAction(key, st) {
    if (key === "Escape") return { close: true };
    if (st.pr && st.g == null && /^[1-9]$/.test(key) && +key <= st.pr.names.length) return { answer: +key - 1 };
    if (st.pr && st.g != null && (key === " " || key === "Enter")) return { next: true };
    return null;
  }

  // the page state for the problem (the solver's view of the same spot): preflop line, flop, flop action, hero
  function stateFor(rep, pr) {
    const pre = rep.line.split(",").map(t => {
      const [pos, a] = t.split(":");
      return a[0] === "r" ? { pos, action: "raise", to: parseFloat(a.slice(1)) } : { pos, action: { f: "fold", c: "call", x: "check" }[a[0]] };
    });
    const [oop, ip] = rep.seats, flop = [];
    for (const x of pr.path) flop.push(x === "check" ? { pos: flop.length ? ip : oop, action: "check" } : { pos: flop.length ? ip : oop, action: "bet", to: potAt(rep, [x]).bet });
    return { players: 6, hero: pr.hero, stack: 100, cards: [...pr.hand, ...pr.board, null, null], actions: { preflop: pre, flop, turn: [], river: [] } };
  }

  // ---- the panel (browser)
  const KEY = "gto-drills";
  let box = null, index = null, rep = null, ranges = null, entries = {}, busy = false;
  const ST = { spot: null, view: "vsbet", filter: {}, pr: null, g: null, session: emptyScore(), hist: {} };
  try { const s = JSON.parse(localStorage.getItem(KEY)) || {}; Object.assign(ST, { spot: s.spot || null, view: s.view || "vsbet", filter: s.filter || {}, hist: s.hist || {} }); } catch (e) {}
  const store = () => { try { localStorage.setItem(KEY, JSON.stringify({ spot: ST.spot, view: ST.view, filter: ST.filter, hist: ST.hist })); } catch (e) {} };
  const hkey = () => `${ST.spot}/${ST.view}`;
  const get = async (p, bin) => { const r = await fetch("flop_library/" + p); if (!r.ok) throw new Error(`couldn't load ${p} (${r.status})`); return bin ? r.arrayBuffer() : r.json(); };
  const spotOf = (id) => index.spots.find(s => s.id === id);
  async function entry(f) {
    const k = ST.spot + "/" + f;
    if (!entries[k]) { const e = await decode(await get(`${ST.spot}/${f}.bin?v=${spotOf(ST.spot).flops[f]}`, true)); entries = { ...(Object.keys(entries).length > 40 ? {} : entries), [k]: e }; }
    return entries[k];
  }
  async function deal() {
    if (busy) return; busy = true; ST.g = null;
    try {
      const cs = candidates(rep, ST.view, ST.filter);
      ST.pr = null; ST.err = cs.length ? null : "No flop in this spot stores this decision with that texture: widen the filter.";
      for (let tries = 0; cs.length && !ST.pr && tries < 8; tries++) {
        const c = cs[pick(cs.map(x => x.w), Math.random)];
        ST.pr = problem(rep, await entry(c.f), c, ranges, Math.random);
      }
    } catch (err) { ST.err = "Couldn't deal: " + (err.message || err); }
    busy = false; draw();
  }
  function answer(a) {
    if (!ST.pr || ST.g) return;
    ST.g = grade(ST.pr, a); ST.session = addScore(ST.session, ST.g); ST.hist[hkey()] = addScore(ST.hist[hkey()], ST.g); store(); draw();
    const n = box.querySelector("[data-next]"); if (n) n.focus();
  }
  async function pickSpot(id) {
    ST.spot = id; rep = ranges = null; ST.pr = null; ST.session = emptyScore(); draw();
    const sp = spotOf(id);
    [rep, ranges] = await Promise.all([get(`${id}/report.json?v=${sp.report}`), get(`${id}/ranges.json?v=${sp.fp.join("")}`)]);
    if (String(ranges.fp) !== String(sp.fp)) throw new Error("the spot's ranges.json is out of date");
    store(); return deal();
  }

  const label = (n) => n[0].toUpperCase() + n.slice(1);
  const fmt = (x) => (Math.round(x * 100) / 100).toString();
  function actLabel(pr, n) {
    if (/^bet /.test(n)) return `${label(n)} <small>${fmt(parseFloat(n.slice(4)) / 100 * pr.pot)}bb</small>`;
    if (n === "call") return `Call <small>${fmt(pr.bet)}bb</small>`;
    return esc(label(n));
  }
  const cardH = (c) => `<span class="dr-c ${c[1]}">${c[0]}${{ s: "♠", h: "♥", d: "♦", c: "♣" }[c[1]]}</span>`;
  const color = (n, names) => {
    if (n === "fold") return "var(--fold)";
    if (n === "check" || n === "call") return "var(--passive)";
    const agg = names.filter(c => /^(bet|raise)/.test(c)), sh = agg.length === 1 ? [1] : [0, 2, 3];
    return `var(--agg${sh[Math.min(agg.indexOf(n), sh.length - 1)]})`;
  };
  function situation(pr) {
    const [o, p] = rep.seats, b = pr.path.find(x => /^bet /.test(x));
    const line = pr.path.length === 0 ? `${o} to act first` : b ? `${o} checks, ${p} bets ${b.slice(4)} (${fmt(pr.bet)}bb)` : `${o} checks`;
    return `${esc(line)} · pot ${fmt(pr.pot)}bb`;
  }
  function scoreHtml() {
    const s = summary(ST.session), h = ST.hist[hkey()], hs = h && summary(h);
    const rec = h && h.recent.length ? Math.round(h.recent.reduce((a, b) => a + b, 0) / h.recent.length * 100) : null;
    return `<div class="dr-score"><span><b>${ST.session.n ? Math.round(s.acc * 100) : "—"}%</b> accuracy <small>${ST.session.ok}/${ST.session.n}</small></span>`
      + `<span><b>${fmt(s.loss)}</b>bb <small>(${s.lossPct.toFixed(1)}% pot)</small> avg loss</span><span><b>${ST.session.streak}</b> streak <small>best ${ST.session.best}</small></span></div>`
      + (h && h.n ? `<div class="dr-hist">This drill, all sessions: ${h.n} hands, ${Math.round(hs.acc * 100)}% (last ${h.recent.length}: ${rec}%), ${hs.lossPct.toFixed(1)}% pot avg loss, best streak ${h.best} <button class="link" data-reset>Reset</button></div>` : "");
  }
  function stageHtml() {
    if (ST.err) return `<div class="dr-msg">${esc(ST.err)}</div>`;
    const pr = ST.pr, g = ST.g;
    if (!pr) return `<div class="dr-msg">Dealing…</div>`;
    const btns = pr.names.map((n, i) => {
      const cls = !g ? "" : i === g.a ? (g.ok ? " dr-ok" : " dr-bad") : pr.freqs[i] >= MIX || i === g.best ? " dr-gto" : "";
      return `<button data-a="${i}" class="dr-act${cls}"${g ? " disabled" : ""}><kbd>${i + 1}</kbd>${actLabel(pr, n)}</button>`;
    }).join("");
    let fb = "";
    if (g) {
      const rows = pr.names.map((n, i) => `<tr class="${i === g.a ? "dr-me" : ""}"><td>${actLabel(pr, n)}</td><td class="dr-bw"><span class="dr-bar"><i style="width:${(pr.freqs[i] * 100).toFixed(1)}%;background:${color(n, pr.names)}"></i></span></td><td class="n">${(pr.freqs[i] * 100).toFixed(0)}%</td><td class="n">${pr.evs[i].toFixed(2)}</td></tr>`).join("");
      fb = `<div class="dr-fb"><div class="dr-verdict ${g.ok ? "good" : "bad"}"><b>${g.verdict}</b>${g.mixed ? ` <span class="dr-tag">mixed</span>` : ""}`
        + `<span>${g.loss > 0 ? `EV loss ${fmt(g.loss)}bb (${g.lossPct.toFixed(1)}% pot)` : "no EV lost"}</span></div>`
        + `<table class="dr-t"><thead><tr><th>Action</th><th>GTO mix</th><th class="n">Freq</th><th class="n">EV bb</th></tr></thead><tbody>${rows}</tbody></table>`
        + `<div class="dr-next"><button class="link" data-study>Open in the solver</button><button data-next class="active">Next hand <kbd>Space</kbd></button></div></div>`;
    }
    return `<div class="dr-sit">${situation(pr)}</div><div class="dr-cards"><div><small>Board</small><span>${pr.board.map(cardH).join("")}</span></div>`
      + `<div><small>You (${esc(pr.hero)})</small><span>${pr.hand.map(cardH).join("")}</span></div></div><div class="dr-acts">${btns}</div>${fb}`;
  }
  function controls() {
    const sel = (id, opts, cur) => `<select data-ctl="${id}" aria-label="${id}">${opts.map(([v, t]) => `<option value="${v}"${v === cur ? " selected" : ""}>${esc(t)}</option>`).join("")}</select>`;
    const L = FR().LABELS, G = { hi: "high card", suits: "suits", pair: "pairing", conn: "connectedness" };
    const f = (k) => sel("f:" + k, [["", "Any " + G[k]], ...Object.entries(L[k])], ST.filter[k] || "");
    const nf = Object.values(ST.filter).filter(Boolean).length;
    return `<div class="dr-ctl">${sel("spot", index.spots.filter(s => s.report).map(s => [s.id, s.label]), ST.spot)}`
      + `<span class="segment">${Object.keys(VIEWS).map(v => `<button data-view="${v}" class="${v === ST.view ? "active" : ""}">${esc(VIEWS[v](spotOf(ST.spot)))}</button>`).join("")}</span></div>`
      + `<details class="dr-flt"${nf ? " open" : ""}><summary>Texture filter${nf ? ` (${nf})` : ""}</summary><div class="dr-ctl">${f("hi")}${f("suits")}${f("pair")}${f("conn")}</div></details>`;
  }
  function draw() {
    if (!box) return;
    const body = box.querySelector(".dr-body");
    if (!index || !rep) { body.innerHTML = `<div class="dr-msg">Loading…</div>`; return; }
    body.innerHTML = controls() + `<div class="dr-stage">${stageHtml()}</div>` + scoreHtml()
      + `<div class="dr-note">Keys: 1-${ST.pr ? ST.pr.names.length : 3} answer, Space or Enter for the next hand. Correct: the best EV, or an action the hand plays ${MIX * 100}% of the time or more. Flops drawn by how often they come, hands by the range's weight at this point.</div>`;
  }
  function act(d) {
    if (d.a != null) return answer(+d.a);
    if (d.next != null) return deal();
    if (d.view) { ST.view = d.view; ST.session = emptyScore(); store(); return deal(); }
    if (d.reset != null) { delete ST.hist[hkey()]; store(); return draw(); }
    if (d.study != null) { const st = stateFor(rep, ST.pr); close(); return window.loadHandState && window.loadHandState(st, `${rep.label}, ${ST.pr.board.join("")}`); }
    if (d.ctl === "spot") return pickSpot(d.value).catch(err => { ST.err = err.message; draw(); });
    if (d.ctl && d.ctl.startsWith("f:")) { ST.filter = { ...ST.filter, [d.ctl.slice(2)]: d.value }; ST.session = emptyScore(); store(); return deal(); }
  }
  function onKey(e) {
    if (!box || box.hidden || e.metaKey || e.ctrlKey || e.altKey) return;
    if (/select|input|textarea/i.test(e.target.tagName) && e.key !== "Escape") return;
    if (e.key === "Tab") return;
    const r = keyAction(e.key, ST);
    e.stopPropagation();                        // the page's own shortcuts stay off while the drills are open
    if (!r) return;
    e.preventDefault();
    if (r.close) close(); else if (r.next) deal(); else answer(r.answer);
  }
  function close() { if (box) box.hidden = true; const m = document.getElementById("menuBtn"); if (m) m.focus(); }
  const CSS = `
  .dr { position: fixed; inset: 0; z-index: 90; background: rgba(0,0,0,.55); display: flex; justify-content: center; align-items: flex-start; padding: 40px 16px 16px; }
  .dr-card { background: var(--panel2); color: var(--text); border: 1px solid var(--line2); border-radius: var(--r); width: min(620px, 100%); max-height: calc(100dvh - 56px); display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,.5); }
  .dr-head { display: flex; align-items: baseline; gap: 10px; padding: 10px 12px 6px; }
  .dr-sub { color: var(--muted); font-size: 12px; flex: 1; }
  .dr-x { margin-left: auto; padding: 2px 8px; }
  .dr-body { display: flex; flex-direction: column; gap: 10px; padding: 0 12px 12px; overflow: auto; }
  .dr-ctl { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .dr-ctl select { font-size: 12px; padding: 5px 6px; max-width: 100%; }
  .dr-ctl .segment { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
  .dr-ctl .segment button { font-size: 12px; padding: 5px 9px; border-radius: 0; border: 0; }
  .dr-flt summary { cursor: pointer; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
  .dr-stage { border: 1px solid var(--line); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; gap: 12px; }
  .dr-sit { color: var(--muted); font-size: 13px; }
  .dr-cards { display: flex; gap: 28px; flex-wrap: wrap; }
  .dr-cards > div { display: flex; flex-direction: column; gap: 4px; }
  .dr-cards small { color: var(--faint); font-size: 11px; }
  .dr-c { display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 54px; margin-right: 4px; border-radius: 5px; background: #f5f5f0; color: #111; font-weight: 700; font-size: 17px; }
  .dr-acts { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 6px; }
  .dr-act { padding: 10px 8px; font-weight: 600; text-align: left; }
  .dr-act small { color: var(--muted); font-weight: 400; }
  .dr-act kbd { margin: 0 6px 0 0; }
  .dr-act:disabled { opacity: .55; }
  .dr-act.dr-ok { opacity: 1; border-color: var(--good); box-shadow: inset 0 0 0 1px var(--good); }
  .dr-act.dr-bad { opacity: 1; border-color: var(--bad); box-shadow: inset 0 0 0 1px var(--bad); }
  .dr-act.dr-gto { opacity: .9; border-color: var(--line2); }
  .dr-verdict { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; font-size: 14px; }
  .dr-verdict.good b { color: var(--good); } .dr-verdict.bad b { color: var(--bad); }
  .dr-verdict span { color: var(--muted); font-size: 12px; }
  .dr-verdict .dr-tag { border: 1px solid var(--line2); border-radius: 9px; padding: 0 6px; font-size: 11px; }
  .dr-t { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; margin-top: 6px; }
  .dr-t th { color: var(--muted); font-weight: 500; text-align: left; padding: 3px 6px; border-bottom: 1px solid var(--line2); }
  .dr-t td { padding: 4px 6px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  .dr-t .n { text-align: right; } .dr-t small { color: var(--muted); }
  .dr-t tr.dr-me td { background: var(--line); }
  .dr-bw { width: 45%; }
  .dr-bar { display: block; height: 10px; border-radius: 3px; background: var(--bg); overflow: hidden; }
  .dr-bar i { display: block; height: 100%; }
  .dr-next { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; }
  .dr-score { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; font-variant-numeric: tabular-nums; }
  .dr-score small { color: var(--muted); }
  .dr-hist, .dr-note { color: var(--muted); font-size: 11px; line-height: 1.45; }
  .dr-msg { padding: 20px 4px; color: var(--muted); font-size: 13px; }
  @media (max-width: 600px) {
    .dr { padding: 0; }
    .dr-card { max-height: 100dvh; height: 100dvh; border-radius: 0; border: 0; width: 100%; }
    .dr-sub { display: none; }
    .dr-ctl select { font-size: 16px; flex: 1 1 40%; }
    .dr-ctl .segment { width: 100%; }
    .dr-ctl .segment button { flex: 1 1 30%; padding: 7px 4px; }
    .dr-c { width: 36px; height: 48px; font-size: 16px; }
    .dr-acts { grid-template-columns: 1fr 1fr; }
  }`;
  async function open() {
    if (!document.getElementById("drStyle")) { const s = document.createElement("style"); s.id = "drStyle"; s.textContent = CSS; document.head.appendChild(s); }
    if (!box) {
      box = document.createElement("div"); box.id = "drillsBox"; box.className = "dr"; box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true"); box.setAttribute("aria-label", "Drills");
      box.innerHTML = `<div class="dr-card"><div class="dr-head"><b>Drills</b><span class="dr-sub">one flop decision, hand after hand, graded against the precomputed solutions</span><button class="ghost dr-x" data-close aria-label="Close">✕</button></div><div class="dr-body"></div></div>`;
      document.body.appendChild(box);
      box.addEventListener("click", (e) => {
        if (e.target === box) return close();
        const t = e.target.closest("[data-close],[data-a],[data-next],[data-view],[data-reset],[data-study]");
        if (!t || t.disabled) return;
        if (t.dataset.close != null) close(); else act(t.dataset);
      });
      box.addEventListener("change", (e) => { const c = e.target.dataset.ctl; if (c) act({ ctl: c, value: e.target.value }); });
      window.addEventListener("keydown", onKey, true);
    }
    box.hidden = false; draw();
    try {
      index = index || await get("index.json");
      if (index.format !== FORMAT) { index = null; throw new Error("the library changed: reload the page"); }
      const ids = (index.spots || []).filter(s => s.report).map(s => s.id);
      if (!ids.length) throw new Error("no precomputed spots");
      if (!rep) await pickSpot(ids.includes(ST.spot) ? ST.spot : ids[0]);
    } catch (err) { box.querySelector(".dr-body").innerHTML = `<div class="dr-msg">Couldn't load the drills: ${esc(err.message || err)}</div>`; }
    const t = box.querySelector(".dr-act") || box.querySelector("select"); if (t) t.focus();
  }

  const api = { decode, parse, nodeAt, candidates, pick, handWeights, problem, potAt, grade, addScore, emptyScore, summary,
                keyAction, stateFor, COMBOS, VIEWS, MIX, open, state: ST };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Drills = api;
})();
