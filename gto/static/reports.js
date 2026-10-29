// Flop reports, loaded by index.html the first time "Flop reports" is opened from the ⋯ menu: a precomputed spot's
// flop strategy across all its flops (flop_library/<spot>/report.json, written with the library's index by
// scripts/build_flop_library.py), per flop and averaged by texture group (each flop weighted by how many of the 22100
// flops it stands for), sortable and filterable; a click on a flop loads that hand into the page. The pure parts
// (texture, filter, sort, groups, the HTML, the hand state) are what the Node tests call.
(function () {
  const LIB = typeof LIB_BASE === "string" ? LIB_BASE : "flop_library/";   // where the library is (index.html's LIB_BASE)
  "use strict";
  const RANKS = "23456789TJQKA";
  const rv = (c) => RANKS.indexOf(c[0]);
  const cardsOf = (f) => [f.slice(0, 2), f.slice(2, 4), f.slice(4, 6)];
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const pct = (x, d = 0) => x == null ? "—" : (x * 100).toFixed(d);

  // A flop's texture: high card, pairing, suits, connectedness (any straight with two hole cards: connected;
  // two ranks within a straight's span: semi; neither: disconnected; paired and trips flops count their distinct ranks).
  function texture(f) {
    const cs = cardsOf(f), r = cs.map(rv).sort((a, b) => b - a), su = new Set(cs.map(c => c[1])).size;
    const d = [...new Set(r)], lo = d.map(x => x === 12 ? -1 : x);   // the ace plays low too
    const span = (xs) => Math.max(...xs) - Math.min(...xs) <= 4;
    const near = (xs) => xs.some((a, i) => xs.some((b, j) => j > i && Math.abs(a - b) <= 4));
    const conn = d.length === 3 && (span(d) || span(lo)) ? "conn" : near(d) || near(lo) ? "semi" : "dry";
    const high = RANKS[r[0]];
    return { high, hi: "AKQJT".includes(high) ? high : "9-", pair: d.length === 3 ? "unpaired" : d.length === 2 ? "paired" : "trips",
             suits: su === 1 ? "mono" : su === 2 ? "twotone" : "rainbow", conn, ranks: r };
  }
  const LABELS = {
    suits: { rainbow: "Rainbow", twotone: "Two-tone", mono: "Monotone" },
    pair: { unpaired: "Unpaired", paired: "Paired", trips: "Trips" },
    conn: { conn: "Connected", semi: "Semi-connected", dry: "Disconnected" },
    hi: { A: "A high", K: "K high", Q: "Q high", J: "J high", T: "T high", "9-": "9 high or lower" },
  };
  const GROUPS = { hi: "High card", suits: "Suits", pair: "Pairing", conn: "Connectedness" };

  // The views: who acts at which stored node. vsbet: out of position facing in position's most common bet after
  // checking (fold / call / raise, whatever the size).
  const VIEWS = { oop: { label: (s) => `${s.seats[0]} first`, player: 0 }, ip: { label: (s) => `${s.seats[1]} vs check`, player: 1 },
                  vsbet: { label: (s) => `${s.seats[0]} vs bet`, player: 0 } };
  const short = (n) => n.replace(/^raise.*/, "raise");
  // the view's node on a flop: {key, names (as the view's columns), freqs} or null where the entry doesn't store it
  function nodeOf(rep, fl, view) {
    if (view === "oop") return fl.n[""] ? { key: "", names: rep.nodes[""], freqs: fl.n[""] } : null;
    if (view === "ip") return fl.n.check ? { key: "check", names: rep.nodes.check, freqs: fl.n.check } : null;
    // (in position's response to a check isn't stored where out of position hardly checks; its bets may be)
    const ck = fl.n.check, cn = rep.nodes.check || [];
    let best = null;
    for (const k of Object.keys(fl.n).filter(k => k.startsWith("check|")).sort()) {
      const nm = k.slice(6), f = ck ? ck[cn.indexOf(nm)] || 0 : 0;
      if (!best || f > best.f) best = { k, nm, f };
    }
    return best && { key: best.k, names: rep.nodes[best.k].map(short), freqs: fl.n[best.k], bet: best.nm };
  }
  // the view's columns: action names in order (vsbet: fold, call, raise)
  function columns(rep, view) {
    if (view !== "vsbet") return (view === "oop" ? rep.nodes[""] : rep.nodes.check) || [];
    const out = [];
    for (const k in rep.nodes) if (k.startsWith("check|")) for (const n of rep.nodes[k].map(short)) if (!out.includes(n)) out.push(n);
    return out;
  }
  // rows: one per flop {f, w, t (texture), ev, eq, freqs (by column, null if not stored), bet, hand}
  function rows(rep, view) {
    const cols = columns(rep, view);
    return rep.flops.map(fl => {
      const nd = nodeOf(rep, fl, view);
      const freqs = nd ? cols.map(c => { const i = nd.names.indexOf(c); return i < 0 ? 0 : nd.freqs[i]; }) : null;
      return { f: fl.f, w: fl.w, t: texture(fl.f), ev: fl.ev, eq: fl.eq, freqs, bet: nd && nd.bet, key: nd && nd.key, hand: nd && fl.h[nd.key] };
    });
  }
  const matches = (r, flt) => Object.entries(flt || {}).every(([k, v]) => !v || r.t[k] === v);
  const filter = (rs, flt) => rs.filter(r => matches(r, flt));
  // sort keys: "flop" (ranks high to low), "w", "ev0", "ev1", "eq0", "eq1", "a<i>" (the i-th column's frequency)
  function sortKey(r, key) {
    if (key === "flop") return r.t.ranks[0] * 169 + r.t.ranks[1] * 13 + r.t.ranks[2];
    if (key === "w") return r.w;
    const m = key.match(/^(ev|eq|a)(\d+)$/);
    if (!m) return 0;
    const src = m[1] === "a" ? r.freqs : r[m[1]];
    return src ? src[+m[2]] : -Infinity;
  }
  function sort(rs, key, desc) {
    const s = desc ? -1 : 1;
    return [...rs].sort((a, b) => (sortKey(a, key) - sortKey(b, key)) * s || (sortKey(b, "flop") - sortKey(a, "flop")));
  }
  // weighted averages over rows: {n, w, freqs, ev, eq}; frequencies over the flops that store the view's node
  function average(rs) {
    const out = { n: rs.length, w: 0, freqs: null, ev: [0, 0], eq: [0, 0] };
    let wf = 0;
    for (const r of rs) {
      out.w += r.w;
      for (const i of [0, 1]) { out.ev[i] += r.w * r.ev[i]; out.eq[i] += r.w * r.eq[i]; }
      if (r.freqs) { out.freqs = out.freqs || r.freqs.map(() => 0); r.freqs.forEach((x, i) => out.freqs[i] += r.w * x); wf += r.w; }
    }
    for (const i of [0, 1]) { out.ev[i] /= out.w || 1; out.eq[i] /= out.w || 1; }
    if (out.freqs) out.freqs = out.freqs.map(x => x / wf);
    return out;
  }
  function groups(rs, by) {
    const order = Object.keys(LABELS[by]);
    return order.map(g => ({ g, label: LABELS[by][g], ...average(rs.filter(r => r.t[by] === g)) })).filter(x => x.n);
  }

  // the page's colours for actions: bets and raises shaded by size like the advice's bars
  function color(n, cols) {
    if (n === "fold") return "var(--fold)";
    if (n === "check" || n === "call") return "var(--passive)";
    if (n === "all-in") return "var(--allin)";
    const agg = cols.filter(c => /^(bet|raise)/.test(c));
    const sh = agg.length === 1 ? [1] : agg.length === 2 ? [0, 2] : [0, 1, 2, 3];
    return `var(--agg${sh[Math.min(agg.indexOf(n), sh.length - 1)] ?? 1})`;
  }
  const label = (n) => n[0].toUpperCase() + n.slice(1);
  function bar(freqs, cols) {
    if (!freqs) return `<span class="rp-na">not stored</span>`;
    return `<span class="rp-bar">${freqs.map((x, i) => x > 0.001 ? `<i style="width:${(x * 100).toFixed(2)}%;background:${color(cols[i], cols)}" title="${esc(label(cols[i]))} ${pct(x, 1)}%"></i>` : "").join("")}</span>`;
  }
  const flopHtml = (f) => cardsOf(f).map(c => typeof cardHtml === "function" ? cardHtml(c)
    : `<span class="${c[1]}">${c[0]}${{ s: "♠", h: "♥", d: "♦", c: "♣" }[c[1]]}</span>`).join("");

  // the table: a summary row, the group averages (when grouped), the flops; th[data-sort] sorts, tr[data-flop]
  // loads, tr[data-group] filters to that group
  function html(rep, st) {
    const cols = columns(rep, st.view), all = rows(rep, st.view), rs = sort(filter(all, st.filter), st.sort, st.desc);
    const tot = all.reduce((a, r) => a + r.w, 0), [o, p] = rep.seats;
    const arrow = (k) => st.sort === k ? (st.desc ? " ▾" : " ▴") : "";
    const th = (k, t, cls = "") => `<th data-sort="${k}" class="${cls}" tabindex="0" aria-sort="${st.sort === k ? (st.desc ? "descending" : "ascending") : "none"}">${t}${arrow(k)}</th>`;
    const nums = (a) => `${(a.freqs || cols.map(() => null)).map(x => `<td class="n">${pct(x)}</td>`).join("")}`
      + `<td class="n">${a.ev[0].toFixed(2)}</td><td class="n">${a.ev[1].toFixed(2)}</td><td class="n rp-eq">${pct(a.eq[0], 1)}</td><td class="n rp-eq">${pct(a.eq[1], 1)}</td>`;
    const head = `<tr>${th("flop", "Flop", "rp-f")}<th class="rp-b">Strategy</th>${cols.map((c, i) => th("a" + i, esc(label(c)), "n")).join("")}`
      + `${th("ev0", `${o} EV`, "n")}${th("ev1", `${p} EV`, "n")}${th("eq0", `${o} eq`, "n rp-eq")}${th("eq1", `${p} eq`, "n rp-eq")}</tr>`;
    const sum = average(rs);
    let body = `<tr class="rp-sum"><td class="rp-f">${rs.length === all.length ? "All flops" : `${rs.length} flops`}<small>${pct(sum.w / tot, 0)}% of flops</small></td><td class="rp-b">${bar(sum.freqs, cols)}</td>${nums(sum)}</tr>`;
    if (st.group) for (const g of groups(rs, st.group))
      body += `<tr class="rp-grp" data-group="${st.group}:${g.g}" tabindex="0"><td class="rp-f">${esc(g.label)}<small>${pct(g.w / tot, 1)}%</small></td><td class="rp-b">${bar(g.freqs, cols)}</td>${nums(g)}</tr>`;
    for (const r of rs)
      body += `<tr data-flop="${r.f}" tabindex="0" title="Load this flop${r.bet ? ` (facing ${esc(r.bet)})` : ""}"><td class="rp-f"><span class="rp-cards">${flopHtml(r.f)}</span>${r.bet ? `<small>vs ${esc(r.bet.replace("bet ", ""))}</small>` : ""}</td><td class="rp-b">${bar(r.freqs, cols)}</td>${nums({ freqs: r.freqs, ev: r.ev, eq: r.eq })}</tr>`;
    return `<table class="rp-t"><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }

  // The page state for a flop of the report at the view's node: the spot's preflop line, the canonical flop, the
  // flop action up to the node, the acting player as hero with the report's hand for it.
  function stateFor(rep, f, view) {
    const fl = rep.flops.find(x => x.f === f);
    if (!fl) return null;
    const nd = nodeOf(rep, fl, view), [oop, ip] = rep.seats;
    const pre = rep.line.split(",").map(t => {
      const [pos, a] = t.split(":");
      return a[0] === "r" ? { pos, action: "raise", to: parseFloat(a.slice(1)) } : { pos, action: { f: "fold", c: "call", x: "check" }[a[0]] };
    });
    const flop = view === "oop" ? [] : [{ pos: oop, action: "check" }];
    if (view === "vsbet" && nd) flop.push({ pos: ip, action: "bet", to: Math.round(parseFloat(nd.bet.match(/[\d.]+/)[0]) / 100 * rep.pot * 100) / 100 });
    const hand = nd ? fl.h[nd.key] : null;
    const hole = hand ? [hand.slice(0, 2), hand.slice(2, 4)] : [null, null];
    const g = rep.game || {};     // (a spot of another game than 6-max 100bb: {players, stack, ante, straddle})
    return { players: g.players || 6, hero: VIEWS[view].player ? ip : oop, stack: g.stack || 100, ...(g.ante ? { ante: g.ante } : {}),
             ...(g.straddle ? { straddle: true } : {}), cards: [...hole, ...cardsOf(f), null, null], actions: { preflop: pre, flop, turn: [], river: [] } };
  }

  // ---- the panel (browser)
  let box = null, index = null, rep = null;
  const ST = { spot: null, view: "oop", sort: "flop", desc: true, group: "", filter: {} };
  try { Object.assign(ST, JSON.parse(localStorage.getItem("gto-reports")) || {}); } catch (e) {}
  const store = () => { try { localStorage.setItem("gto-reports", JSON.stringify(ST)); } catch (e) {} };
  const get = async (p) => { const r = await fetch(LIB + p, { cache: "no-cache" }); if (!r.ok) throw new Error(`couldn't load ${p} (${r.status})`); return r.json(); };

  function controls() {
    const sel = (id, opts, cur) => `<select data-ctl="${id}">${opts.map(([v, t]) => `<option value="${v}"${v === cur ? " selected" : ""}>${esc(t)}</option>`).join("")}</select>`;
    const spots = (index.spots || []).filter(s => s.report).map(s => [s.id, s.label]);
    const f = (k) => sel("f:" + k, [["", "Any " + GROUPS[k].toLowerCase()], ...Object.entries(LABELS[k])], ST.filter[k] || "");
    return `<div class="rp-ctl">${sel("spot", spots, ST.spot)}<span class="segment">${Object.keys(VIEWS).map(v =>
      `<button data-view="${v}" class="${v === ST.view ? "active" : ""}">${esc(VIEWS[v].label(rep))}</button>`).join("")}</span></div>
      <div class="rp-ctl">${f("hi")}${f("suits")}${f("pair")}${f("conn")}${sel("group", [["", "No groups"], ...Object.entries(GROUPS).map(([k, t]) => [k, "By " + t.toLowerCase()])], ST.group)}</div>`;
  }
  function draw() {
    if (!box) return;
    const body = box.querySelector(".rp-body");
    if (!rep) { body.innerHTML = `<div class="rp-msg">Loading…</div>`; return; }
    body.innerHTML = controls() + `<div class="rp-scroll">${html(rep, ST)}</div>`
      + `<div class="rp-note">${esc(rep.label)}, 100bb, flops up to suits, each weighted by the flops it stands for. Frequencies are the range's; EV in bb at the start of the flop (pot ${rep.pot}bb), equity range against range. ${"ontouchstart" in window ? "Tap" : "Click"} a flop to load it with a hand that mixes there.</div>`;
  }
  async function pick(id) {
    ST.spot = id; rep = null; draw();
    const sp = index.spots.find(s => s.id === id);
    rep = await get(`${id}/report.json?v=${sp.report}`); draw(); store();
    if (globalThis.viewSpot) globalThis.viewSpot("reports", id, rep.label);   // (the page's URL: /reports/<spot>/)
  }
  function act(d) {
    if (d.view) { ST.view = d.view; }
    else if (d.sort) { if (ST.sort === d.sort) ST.desc = !ST.desc; else { ST.sort = d.sort; ST.desc = true; } }
    else if (d.group) { const [k, v] = d.group.split(":"); ST.filter = { ...ST.filter, [k]: v }; ST.group = ""; }
    else if (d.flop) { const st = stateFor(rep, d.flop, ST.view); close(); return window.loadHandState && window.loadHandState(st, `${rep.label}, ${d.flop}`); }
    else if (d.ctl === "spot") { store(); return pick(d.value); }
    else if (d.ctl === "group") ST.group = d.value;
    else if (d.ctl && d.ctl.startsWith("f:")) ST.filter = { ...ST.filter, [d.ctl.slice(2)]: d.value };
    store(); draw();
  }
  function close() {
    if (box) box.hidden = true; const m = document.getElementById("menuBtn"); if (m) m.focus();
    if (globalThis.viewClosed) globalThis.viewClosed("reports");   // (the page's URL back to the solver's)
  }
  // styles on the page's tokens (--panel, --line, --text…), so the panel follows the page's theme
  const CSS = `
  .rp { position: fixed; inset: 0; z-index: 90; background: rgba(0,0,0,.55); display: flex; justify-content: center; align-items: flex-start; padding: 40px 16px 16px; }
  .rp-card { background: var(--panel2); color: var(--text); border: 1px solid var(--line2); border-radius: var(--r); width: min(980px, 100%); max-height: calc(100dvh - 56px); display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,.5); }
  .rp-head { display: flex; align-items: baseline; gap: 10px; padding: 10px 12px 6px; }
  .rp-sub { color: var(--muted); font-size: 12px; flex: 1; }
  .rp-x { margin-left: auto; padding: 2px 8px; }
  .rp-body { display: flex; flex-direction: column; gap: 8px; padding: 0 12px 12px; min-height: 0; }
  .rp-ctl { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .rp-ctl select { background: var(--bg); color: var(--text); border: 1px solid var(--line); border-radius: 7px; padding: 5px 6px; font: inherit; font-size: 12px; max-width: 100%; }
  .rp-ctl .segment { display: inline-flex; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
  .rp-ctl .segment button { font-size: 12px; padding: 5px 9px; }
  .rp-scroll { overflow: auto; min-height: 0; border: 1px solid var(--line); border-radius: 8px; }
  .rp-t { border-collapse: collapse; width: 100%; font-size: 12px; font-variant-numeric: tabular-nums; }
  .rp-t th { position: sticky; top: 0; z-index: 2; background: var(--panel2); color: var(--muted); font-weight: 500; text-align: left; padding: 6px 7px; border-bottom: 1px solid var(--line2); white-space: nowrap; cursor: pointer; user-select: none; }
  .rp-t th[aria-sort="ascending"], .rp-t th[aria-sort="descending"] { color: var(--accent); }
  .rp-t td { padding: 4px 7px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  .rp-t .n { text-align: right; }
  .rp-t .rp-f { position: sticky; left: 0; z-index: 1; background: var(--panel2); min-width: 74px; }
  .rp-t th.rp-f { z-index: 3; }
  .rp-t td.rp-f small { display: block; color: var(--faint); font-size: 10px; line-height: 1.1; }
  .rp-t .rp-b { width: 34%; min-width: 120px; }
  .rp-t tbody tr[data-flop], .rp-t tr.rp-grp { cursor: pointer; }
  .rp-t tbody tr[data-flop]:hover td, .rp-t tr.rp-grp:hover td { background: var(--line); }
  .rp-t tr.rp-sum td { font-weight: 600; background: var(--panel); }
  .rp-t tr.rp-grp td { background: var(--panel); }
  .rp-t tr.rp-grp td.rp-f, .rp-t tr.rp-sum td.rp-f { background: var(--panel); }
  .rp-cards span { font-weight: 600; margin-right: 2px; }
  .rp-bar { display: flex; height: 12px; border-radius: 3px; overflow: hidden; background: var(--line); }
  .rp-bar i { display: block; height: 100%; }
  .rp-na { color: var(--faint); font-size: 11px; }
  .rp-note, .rp-msg { color: var(--muted); font-size: 11px; line-height: 1.45; }
  .rp-msg { padding: 20px 4px; font-size: 13px; }
  @media (max-width: 600px) {
    .rp { padding: 0; }
    .rp-card { max-height: 100dvh; height: 100dvh; border-radius: 0; border: 0; width: 100%; }
    .rp-sub, .rp-t .rp-eq { display: none; }
    .rp-ctl select { font-size: 16px; flex: 1 1 40%; }
    .rp-t .rp-b { min-width: 90px; }
  }`;
  // spot: a spot's id to show (the page's /reports/<spot>/), else the one last picked
  async function open(spot) {
    if (!document.getElementById("rpStyle")) {
      const s = document.createElement("style"); s.id = "rpStyle"; s.textContent = CSS; document.head.appendChild(s);
    }
    if (!box) {
      box = document.createElement("div"); box.id = "reportsBox"; box.className = "rp"; box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true"); box.setAttribute("aria-label", "Flop reports");
      box.innerHTML = `<div class="rp-card"><div class="rp-head"><b>Flop reports</b><span class="rp-sub">the precomputed spots' flop strategy across every flop</span><button class="ghost rp-x" data-close aria-label="Close">✕</button></div><div class="rp-body"></div></div>`;
      document.body.appendChild(box);
      const data = (t) => { const e = t.closest("[data-close],[data-view],[data-sort],[data-group],[data-flop]"); return e && (e.dataset.close != null ? { close: 1 } : e.dataset); };
      box.addEventListener("click", (e) => { if (e.target === box) return close(); const d = data(e.target); if (d) { if (d.close) close(); else act(d); } });
      box.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { e.stopPropagation(); close(); }
        else if ((e.key === "Enter" || e.key === " ") && e.target.matches("[data-sort],[data-group],[data-flop]")) { e.preventDefault(); act(e.target.dataset); }
      });
      box.addEventListener("change", (e) => { const c = e.target.dataset.ctl; if (c) act({ ctl: c, value: e.target.value }); });
    }
    box.hidden = false; draw();
    try {
      index = index || await get("index.json");
      const ids = (index.spots || []).filter(s => s.report).map(s => s.id);
      if (!ids.length) throw new Error("no precomputed spots");
      const want = ids.includes(spot) ? spot : ids.includes(ST.spot) ? ST.spot : ids[0];
      if (!rep || want !== ST.spot) await pick(want); else if (globalThis.viewSpot) globalThis.viewSpot("reports", ST.spot, rep.label);
    } catch (err) { box.querySelector(".rp-body").innerHTML = `<div class="rp-msg">Couldn't load the flop reports: ${esc(err.message || err)}</div>`; }
    const t = box.querySelector("select, button"); if (t) t.focus();
  }

  const api = { texture, rows, columns, nodeOf, filter, sort, average, groups, html, stateFor, VIEWS, LABELS,
                open, close, act, state: ST, _set: (i, r) => { index = i; rep = r; } };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.FlopReports = api;
})();
