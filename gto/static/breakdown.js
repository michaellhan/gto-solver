// Breakdown and Equity views of the advice panel, loaded by index.html the first time either tab is opened. Each asks
// the solver again with the advice's own state plus "breakdown": true (gto/breakdown.py), answered from the solve the
// advice came from, so no new solve: per hand category (made hands, draws) its share of the range, the strategy mix
// and EV of the player to act and its equity; each range's equity curve, average equity and EV and its nut share.
// The pure parts (request, the HTML of each view) are what the Node tests call.
(function () {
  "use strict";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const pct = (x, d = 0) => x == null ? "—" : (x * 100).toFixed(d) + "%";
  const evs = (x) => x == null ? "—" : (x > 0 ? "+" : "") + x.toFixed(2);
  // the page's action colours (index.html's COLORS) where it has them
  const color = (n) => (typeof COLORS === "function" ? COLORS(n) : "var(--accent)");
  const VILLAIN = "#6fa8ff";   // the villain's curve and bars (the hero's are the accent)

  // the advice's state with the flag: {breakdown} or a reason it couldn't come (cancelled, an error, none here)
  const cache = new Map();   // key (the advice's payload) -> Promise of that answer
  function request(key, ask) {
    if (!cache.has(key)) {
      const p = Promise.resolve(ask({ ...JSON.parse(key), breakdown: true })).then((r) => {
        if (!r || r.cancelled || r.error) { cache.delete(key); return { error: r && r.error ? r.error : "The request was cancelled; open the tab again." }; }
        return r.breakdown ? { breakdown: r.breakdown } : { error: "No breakdown here: this spot was answered by betting rules, not a solve." };
      }, (e) => { cache.delete(key); return { error: "Request failed: " + e }; });
      cache.set(key, p);
    }
    return cache.get(key);
  }

  const ST = { who: 0, part: "made", sel: null };   // the player shown (0 hero), made hands or draws, the row picked
  const players = (bd) => [bd.hero, ...bd.villains];
  // what the equities are over (bd.runouts: [dealt, of all]): none to come on the river, every river card on the
  // turn, on the flop usually an even sample of the turn and river runouts
  function runText(bd) {
    const [n, all] = bd.runouts || [];
    if (!all) return "over the runouts";
    if (all === 1) return "on the river, no cards to come";
    if (n >= all) return all === 48 ? "every river card dealt out" : "every runout dealt out";
    return `over an even sample of ${n} of the ${all} turn and river runouts`;
  }

  // the Breakdown view: a row per category, its bar as long as its share of the range (against the largest), split
  // by the hero's actions (a villain's: plain), with the share, equity and EV
  function breakdownHtml(bd, st = ST, opts = {}) {
    const ps = players(bd), p = ps[Math.min(st.who, ps.length - 1)], hero = p === bd.hero;
    const parts = bd.hero.draws.length ? ["made", "draws"] : ["made"];
    const part = parts.includes(st.part) ? st.part : "made", rows = p[part] || [];
    const tot = rows.reduce((t, r) => t + r.c, 0) || 1, max = Math.max(...rows.map(r => r.c), 1e-9);
    const mine = hero && bd.hero.mine ? bd.hero.mine[part === "made" ? 0 : 1] : null;
    let h = `<div class="bd-ctl"><div class="segment">${ps.map((x, i) => `<button data-who="${i}" class="${x === p ? "active" : ""}">${i || !bd.hero.mine ? esc(x.who) : `You (${esc(x.who)})`}</button>`).join("")}</div>`;
    if (parts.length > 1) h += `<div class="segment">${parts.map(x => `<button data-part="${x}" class="${x === part ? "active" : ""}">${x === "made" ? "Made hands" : "Draws"}</button>`).join("")}</div>`;
    h += `</div>`;
    if (hero) h += `<div class="bd-legend">${bd.actions.map(n => `<span style="--c:${color(n)}">${esc(n)}</span>`).join("")}</div>`;
    else h += `<div class="bd-note">${esc(p.who)} isn't the one to act, so this is the range's make-up${p.eq != null ? " and equity" : ""}.</div>`;
    const hasEq = rows.some(r => r.eq != null), hasEv = hero && rows.some(r => r.ev != null);
    h += `<div class="bd-rows${hasEq ? "" : " noeq"}${hasEv ? "" : " noev"}"><div class="bd-row bd-head"><span>Category</span><span>Share of range</span><span class="n">%</span>${hasEq ? '<span class="n">Eq</span>' : ""}${hasEv ? '<span class="n">EV</span>' : ""}</div>`;
    rows.forEach((r, i) => {
      const segs = hero && r.f ? r.f.map((f, a) => f > 0 ? `<i style="width:${(f * 100).toFixed(1)}%;background:${color(bd.actions[a])}"></i>` : "").join("")
        : `<i style="width:100%;background:${p === bd.hero ? "var(--accent)" : VILLAIN}"></i>`;
      h += `<div class="bd-row${r.id === mine ? " me" : ""}${st.sel === r.id ? " sel" : ""}" data-row="${r.id}" tabindex="0"><span class="bd-l">${esc(r.label)}${r.id === mine ? '<b class="bd-you">YOU</b>' : ""}</span>`
        + `<span class="bd-b"><span class="bd-bar" style="width:${(r.c / max * 100).toFixed(1)}%">${segs}</span></span>`
        + `<span class="n">${pct(r.c / tot, 1)}</span>${hasEq ? `<span class="n">${pct(r.eq)}</span>` : ""}${hasEv ? `<span class="n">${evs(r.ev)}</span>` : ""}</div>`;
    });
    h += `</div>`;
    const r = rows.find(x => x.id === st.sel) || rows.find(x => x.id === mine);
    if (r) h += `<div class="bd-read"><b>${esc(r.label)}</b> · ${r.c.toFixed(1)} combos (${pct(r.c / tot, 1)})${r.eq != null ? ` · equity ${pct(r.eq, 1)}` : ""}${r.ev != null && hero ? ` · EV ${evs(r.ev)} bb` : ""}`
      + (hero && r.f ? `<br>${bd.actions.map((n, a) => `<span class="a" style="--c:${color(n)}">${esc(n)} ${pct(r.f[a])}</span>`).join(" ")}` : "")
      + (hero && opts.filter && r.id !== "none" ? `<br><button class="ghost bd-go" data-filter="${part === "made" ? "m" : "d"}:${esc(r.id)}">Show these hands in the grid</button>` : "") + `</div>`;
    h += `<div class="bd-note">${part === "made" ? "Made hands by what the hole cards add (a pair on the board is everyone's); top pair's kicker is top, good (the next two) or weak." : "Draws: a flush draw is four to a suit with a hole card; open-ended counts double gutshots; combo is a flush draw with a straight draw; backdoor is three to a flush or straight on the flop."}${hasEq ? ` Equity is against the other range${bd.villains.length > 1 ? "s (a share of the pot at showdown)" : ""}, ${runText(bd)}.` : ""}</div>`;
    return h;
  }

  // the Equity view: each range's equities sorted high to low against its share of the range (GTO Wizard's
  // equity distribution), with the averages, EV and the share at the nut threshold
  function equityHtml(bd) {
    const ps = players(bd).filter(p => p.curve && p.curve.length);
    if (ps.length < 2) return `<div class="bd-msg">The equity view is for heads-up spots: with more players in, each hand's equity against the field isn't split per opponent here.</div>`;
    const W = 320, H = 180, L = 30, B = 20, T = 8, R = 14, x = (f) => L + f * (W - L - R), y = (e) => T + (1 - e) * (H - T - B);
    const col = (i) => i ? VILLAIN : "var(--accent)";
    let svg = `<svg class="bd-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Equity distribution of both ranges">`;
    for (const e of [0, 0.25, 0.5, 0.75, 1]) svg += `<line x1="${L}" x2="${W - R}" y1="${y(e)}" y2="${y(e)}" class="bd-gl${e === 0.5 ? " mid" : ""}"/><text x="${L - 4}" y="${y(e) + 3}" text-anchor="end">${e * 100}</text>`;
    for (const f of [0, 0.25, 0.5, 0.75, 1]) svg += `<text x="${x(f)}" y="${H - 6}" text-anchor="middle">${f * 100}%</text>`;
    svg += `<line x1="${L}" x2="${W - R}" y1="${y(bd.nut_eq)}" y2="${y(bd.nut_eq)}" class="bd-nut"/>`;
    ps.forEach((p, i) => {
      const n = p.curve.length - 1;
      svg += `<polyline fill="none" stroke="${col(i)}" stroke-width="2" stroke-linejoin="round" points="${p.curve.map((e, k) => `${x(k / n).toFixed(1)},${y(e).toFixed(1)}`).join(" ")}"/>`;
    });
    svg += `<line class="bd-cur" x1="0" x2="0" y1="${T}" y2="${H - B}" visibility="hidden"/></svg>`;
    const stat = (p, i) => `<div class="bd-card" style="--c:${col(i)}"><div class="bd-who">${i || !bd.hero.mine ? "" : "You · "}${esc(p.who)} <small>${p.combos} combos</small></div>`
      + `<div class="bd-stats"><span><small>Equity</small><b>${pct(p.eq, 1)}</b></span><span><small>EV</small><b>${p.ev != null ? evs(p.ev) + " bb" : "—"}</b></span><span title="Share of the range with ${pct(bd.nut_eq)} equity or more"><small>≥${pct(bd.nut_eq)}</small><b>${pct(p.nuts, 1)}</b></span></div></div>`;
    const [a, b] = ps, lead = (a.nuts - b.nuts) * 100;
    const nutText = Math.abs(lead) < 1 ? "Neither range has a nut advantage" : `${esc(lead > 0 ? a.who : b.who)} has the nut advantage: ${Math.abs(lead).toFixed(1)} points more of the range at ${pct(bd.nut_eq)}+ equity`;
    const eqText = `${esc(a.eq >= b.eq ? a.who : b.who)} has the range advantage (${pct(Math.max(a.eq, b.eq), 1)} equity)`;
    return `<div class="bd-cards">${ps.map(stat).join("")}</div><div class="bd-chart">${svg}<div class="bd-tip"></div></div>`
      + `<div class="bd-sum">${eqText}. ${nutText}.</div>`
      + `<div class="bd-note">Each line is a range's hands sorted from most to least equity against the other range (${runText(bd)}); the dashed line marks ${pct(bd.nut_eq)}. EV is in big blinds of the pot as it stands (the villain's: the pot less yours${ps[1].ev == null ? "; not shown with rake" : ""}).</div>`;
  }

  // the chart's readout: both ranges' equity at the pointer's share of the range
  function wireChart(box, bd) {
    const svg = box.querySelector(".bd-svg"), tip = box.querySelector(".bd-tip"), cur = box.querySelector(".bd-cur");
    if (!svg || !tip) return;
    const ps = players(bd).filter(p => p.curve && p.curve.length);
    const move = (e) => {
      const r = svg.getBoundingClientRect(), f = Math.max(0, Math.min(1, ((e.clientX - r.left) / r.width * 320 - 30) / 276));
      cur.setAttribute("x1", 30 + f * 276); cur.setAttribute("x2", 30 + f * 276); cur.setAttribute("visibility", "visible");
      tip.innerHTML = `top ${pct(f)} of each range: ` + ps.map((p, i) => `<span style="color:${i ? VILLAIN : "var(--accent)"}">${esc(p.who)} ${pct(p.curve[Math.round(f * (p.curve.length - 1))])}</span>`).join(" · ");
    };
    svg.addEventListener("pointermove", move); svg.addEventListener("pointerdown", move);
    svg.addEventListener("pointerleave", () => { cur.setAttribute("visibility", "hidden"); tip.innerHTML = ""; });
  }

  const CSS = `
  .bd-ctl { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 8px; }
  .bd-ctl .segment button { font-size: 12px; padding: 5px 9px; }
  .bd-legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; margin-bottom: 6px; }
  .bd-legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; background: var(--c); }
  .bd-rows { font-size: 12px; font-variant-numeric: tabular-nums; }
  .bd-row { display: grid; grid-template-columns: minmax(96px, 32%) minmax(0, 1fr) 46px 38px 46px; gap: 6px; align-items: center; padding: 3px 4px; border-radius: 5px; cursor: pointer; }
  .bd-rows.noev .bd-row { grid-template-columns: minmax(96px, 32%) minmax(0, 1fr) 46px 38px; }
  .bd-rows.noeq.noev .bd-row { grid-template-columns: minmax(96px, 32%) minmax(0, 1fr) 46px; }
  .bd-rows.noeq:not(.noev) .bd-row { grid-template-columns: minmax(96px, 32%) minmax(0, 1fr) 46px 46px; }
  .bd-row:hover, .bd-row.sel { background: var(--line); }
  .bd-row.me .bd-l { color: var(--accent); }
  .bd-head { color: var(--muted); cursor: default; }
  .bd-head:hover { background: none; }
  .bd-row .n { text-align: right; }
  .bd-l { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bd-you { font-size: 8px; letter-spacing: .08em; background: var(--accent); color: #1a1200; border-radius: 3px; padding: 1px 3px; margin-left: 5px; vertical-align: 1px; }
  .bd-b { display: block; min-width: 0; }
  .bd-bar { display: flex; height: 12px; border-radius: 3px; overflow: hidden; min-width: 2px; background: var(--line2); }
  .bd-bar i { display: block; height: 100%; }
  .bd-read { margin-top: 8px; font-size: 12px; line-height: 1.6; min-height: 2.6em; }
  .bd-go { font-size: 11px; padding: 2px 8px; margin-top: 4px; }
  .bd-read .a::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin: 0 4px 0 6px; background: var(--c); }
  .bd-note { color: var(--muted); font-size: 11px; line-height: 1.5; margin-top: 8px; }
  .bd-msg { color: var(--muted); font-size: 13px; padding: 14px 2px; }
  .bd-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 4px 0 8px; }
  .bd-card { border: 1px solid var(--line); border-left: 3px solid var(--c); border-radius: 7px; padding: 6px 8px; background: var(--panel2); min-width: 0; }
  .bd-who { font-size: 12px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bd-who small { color: var(--muted); font-weight: 400; }
  .bd-stats { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 2px; font-variant-numeric: tabular-nums; }
  .bd-stats span { display: flex; flex-direction: column; }
  .bd-stats small { color: var(--muted); font-size: 10px; }
  .bd-stats b { font-size: 14px; }
  .bd-chart { position: relative; }
  .bd-svg { width: 100%; max-width: 560px; height: auto; display: block; touch-action: pan-y; }
  .bd-svg text { fill: var(--faint); font-size: 9px; font-variant-numeric: tabular-nums; }
  .bd-gl { stroke: var(--line); stroke-width: 1; }
  .bd-gl.mid { stroke: var(--line2); }
  .bd-nut { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
  .bd-cur { stroke: var(--muted); stroke-width: 1; }
  .bd-tip { font-size: 12px; min-height: 1.5em; font-variant-numeric: tabular-nums; }
  .bd-sum { font-size: 12px; margin-top: 4px; }
  @media (max-width: 420px) {
    .bd-row { grid-template-columns: minmax(84px, 36%) minmax(0, 1fr) 40px 32px 40px; gap: 4px; font-size: 11px; }
    .bd-rows.noev .bd-row { grid-template-columns: minmax(84px, 36%) minmax(0, 1fr) 40px 32px; }
    .bd-stats { gap: 8px; }
  }`;

  // show a view ("breakdown" or "equity") of the advice res (payload key) in box; ask: the page's solver call
  // opts.filter(key): a hero's category row offers it (Study: the grid filtered to "m:<id>" or "d:<id>")
  async function show(box, view, res, key, ask, opts = {}) {
    if (!document.getElementById("bdStyle")) {
      const s = document.createElement("style"); s.id = "bdStyle"; s.textContent = CSS; document.head.appendChild(s);
    }
    const want = key + "|" + view;
    box.dataset.want = want;
    box.innerHTML = `<div class="bd-msg"><span class="spinner"></span>Breaking down the ranges…</div>`;
    const out = await request(key, ask);
    if (box.dataset.want !== want) return;   // another view or newer advice since
    if (out.error) { box.innerHTML = `<div class="bd-msg">${esc(out.error)}</div>`; return; }
    const bd = out.breakdown;
    const draw = () => {
      box.innerHTML = view === "equity" ? equityHtml(bd) : breakdownHtml(bd, ST, opts);
      if (view === "equity") wireChart(box, bd);
    };
    box.onclick = (e) => {
      const t = e.target.closest("[data-who],[data-part],[data-row],[data-filter]");
      if (!t) return;
      if (t.dataset.filter) return opts.filter(t.dataset.filter);
      if (t.dataset.who != null) { ST.who = +t.dataset.who; ST.sel = null; }
      else if (t.dataset.part) { ST.part = t.dataset.part; ST.sel = null; }
      else ST.sel = ST.sel === t.dataset.row ? null : t.dataset.row;
      draw();
    };
    box.onkeydown = (e) => { if ((e.key === "Enter" || e.key === " ") && e.target.dataset && e.target.dataset.row) { e.preventDefault(); e.target.click(); } };
    draw();
  }

  const api = { request, show, breakdownHtml, equityHtml, cache, state: ST };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Breakdown = api;
})();
