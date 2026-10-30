// Turn and river reports, loaded by Study (study.js) the first time one is opened: at a closed flop or turn,
// Advisor.study's "runouts" answer as a table of every next card: the first decision's range strategy on it (or in
// position's after a check), each player's EV and equity there, per card and averaged by what the card does to the
// board, by rank or by suit; sortable and filterable like the flop reports (reports.js), a click on a card deals it.
// The pure parts (card classes, rows, filter, sort, groups, the HTML) are what the Node tests call.
(function () {
  "use strict";
  const RANKS = "23456789TJQKA", SUITS = "shdc", SYM = { s: "♠", h: "♥", d: "♦", c: "♣" };
  const rv = (c) => RANKS.indexOf(c[0]);
  const cardsOf = (b) => String(b || "").match(/../g) || [];
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const ok = (x) => x != null && isFinite(x);
  const pct = (x, d = 0) => ok(x) ? (x * 100).toFixed(d) : "—";
  const bb = (x) => ok(x) ? x.toFixed(2) : "—";
  const cardHtml1 = (c) => typeof cardHtml === "function" ? cardHtml(c) : `<span class="${c[1]}">${c[0]}${SYM[c[1]]}</span>`;
  const color = (n) => typeof COLORS === "function" ? COLORS(n) : n === "fold" ? "var(--fold)" : n === "check" || n === "call" ? "var(--passive)"
    : n === "all-in" ? "var(--allin)" : "var(--agg1)";
  const label = (n) => n[0].toUpperCase() + n.slice(1);

  // What a card does to the board, the first of these that holds: completes a flush (a third card of its suit on the
  // board, or more), completes a straight (a straight's five ranks hold it and two board ranks or more, the ace low
  // too), pairs the board, an overcard (above every board card), an undercard (below the top card, above the
  // lowest) or a brick (below every board card).
  const CLASSES = { over: "Overcards", under: "Undercards", brick: "Bricks", pair: "Pairs the board", straight: "Completes a straight", flush: "Completes a flush" };
  function cardClass(board, card) {
    const bs = cardsOf(board);
    if (bs.filter(c => c[1] === card[1]).length >= 2) return "flush";
    const r = rv(card), ranks = [...new Set(bs.map(rv))];
    if (ranks.includes(r)) return "pair";
    const at = (x) => x === 12 ? [12, -1] : [x];
    for (let s = -1; s <= 8; s++) {
      const inW = (x) => at(x).some(y => y >= s && y <= s + 4);
      if (inW(r) && ranks.filter(inW).length >= 2) return "straight";
    }
    return r > Math.max(...ranks) ? "over" : r < Math.min(...ranks) ? "brick" : "under";
  }
  const SHORT = { over: "overcard", under: "undercard", brick: "brick", pair: "pairs", straight: "straight", flush: "flush" };
  const GROUPS = { cls: "What it does", rank: "Rank", suit: "Suit" };
  const LABELS = { cls: CLASSES, rank: Object.fromEntries([...RANKS].reverse().map(r => [r, r === "T" ? "Tens" : r + "s"])),
                   suit: { s: "♠ Spades", h: "♥ Hearts", d: "♦ Diamonds", c: "♣ Clubs" } };
  // (the groups' order: an object's digit keys would come first)
  const ORDER = { cls: Object.keys(CLASSES), rank: [...RANKS].reverse(), suit: [...SUITS] };

  // the part shown: "first" (the street's first decision) or "after" (in position after a check)
  const partOf = (ro, who) => who === "after" && ro.after_check ? ro.after_check : ro.first;
  // rows: one per card off the board {card, cls, rank, suit, f (the part's mix there; null: none), w (its player's
  // range there), fe (each action's EV for the hands that take it: a stored turn with EVs; else null), p (the card's
  // chance), eq, ev ([out of position, in position]; null: none)}
  function rows(ro, who, board) {
    const part = partOf(ro, who), by = {};
    for (const c of part.cards) for (const x of c.cards) by[x] = c;
    const cards = ro.cards || Object.keys(by).map(card => ({ card, p: null, eq: null, ev: null }));
    return cards.map(x => {
      const c = by[x.card];
      return { card: x.card, cls: cardClass(board, x.card), rank: x.card[0], suit: x.card[1], f: c ? c.f : null,
               w: c ? (c.w ?? 1) : 0, fe: c && c.ev ? c.ev : null, p: x.p, eq: x.eq, ev: x.ev };
    });
  }
  const filter = (rs, flt) => rs.filter(r => Object.entries(flt || {}).every(([k, v]) => !v || r[k] === v));
  // sort keys: "card" (rank, then suit), "a<i>" (the i-th action's frequency), "ev0", "ev1", "eq0", "eq1"
  function sortKey(r, key) {
    if (key === "card") return rv(r.card) * 4 + (3 - SUITS.indexOf(r.suit));
    const m = String(key).match(/^(ev|eq|a)(\d+)$/);
    if (!m) return 0;
    const src = m[1] === "a" ? r.f : r[m[1]];
    return src && ok(src[+m[2]]) ? src[+m[2]] : -Infinity;
  }
  function sort(rs, key, desc) {
    const s = desc ? -1 : 1;
    return [...rs].sort((a, b) => (sortKey(a, key) - sortKey(b, key)) * s || (sortKey(b, "card") - sortKey(a, "card")));
  }
  // averages over rows: the mix by the acting range on each card (w), EV and equity by the cards' chances (p)
  function average(rs) {
    const out = { n: rs.length, p: 0, f: null, ev: null, eq: null };
    let wf = 0, pe = 0, pq = 0;
    for (const r of rs) {
      const p = ok(r.p) ? r.p : 1 / rs.length;
      out.p += p;
      if (r.f && r.w > 0) { out.f = out.f || r.f.map(() => 0); r.f.forEach((x, i) => out.f[i] += r.w * x); wf += r.w; }
      if (r.ev) { out.ev = out.ev || [0, 0]; r.ev.forEach((x, i) => out.ev[i] += p * x); pe += p; }
      if (r.eq) { out.eq = out.eq || [0, 0]; r.eq.forEach((x, i) => out.eq[i] += p * x); pq += p; }
    }
    if (out.f) out.f = out.f.map(x => x / wf);
    if (out.ev) out.ev = out.ev.map(x => x / pe);
    if (out.eq) out.eq = out.eq.map(x => x / pq);
    return out;
  }
  function groups(rs, by) {
    return ORDER[by].map(g => ({ g, label: LABELS[by][g], ...average(rs.filter(r => r[by] === g)) })).filter(x => x.n);
  }
  // fe: each action's EV (a card's row of a stored turn with EVs), in its part's tip
  function bar(f, acts, fe) {
    if (!f) return `<span class="ro-na">—</span>`;
    const ev = (i) => fe && ok(fe[i]) ? `, EV ${bb(fe[i])}bb` : "";
    return `<span class="ro-bar">${f.map((x, i) => x > 0.001 ? `<i style="width:${(x * 100).toFixed(2)}%;background:${color(acts[i])}" title="${esc(label(acts[i]))} ${pct(x, 1)}%${ev(i)}"></i>` : "").join("")}</span>`;
  }

  // the table: a summary row, the group averages (when grouped), the cards; th[data-rosort] sorts, tr[data-card]
  // deals, tr[data-rogroup] filters to that group. ui: {who, sort, desc, group, filter}
  function html(ro, ui, board) {
    const part = partOf(ro, ui.who), acts = part.actions, all = rows(ro, ui.who, board);
    const rs = sort(filter(all, ui.filter), ui.sort || "card", ui.desc !== false);
    const [o, p] = ro.seats || ["OOP", "IP"];
    const arrow = (k) => ui.sort === k ? (ui.desc !== false ? " ▾" : " ▴") : "";
    const th = (k, t, cls = "") => `<th data-rosort="${k}" class="${cls}" tabindex="0" aria-sort="${ui.sort === k ? (ui.desc !== false ? "descending" : "ascending") : "none"}">${t}${arrow(k)}</th>`;
    const nums = (a) => `${(a.f || acts.map(() => null)).map(x => `<td class="n">${pct(x)}</td>`).join("")}`
      + `<td class="n">${bb(a.ev && a.ev[0])}</td><td class="n">${bb(a.ev && a.ev[1])}</td><td class="n ro-eq">${pct(a.eq && a.eq[0], 1)}</td><td class="n ro-eq">${pct(a.eq && a.eq[1], 1)}</td>`;
    const head = `<tr>${th("card", ro.street === "river" ? "River" : "Turn", "ro-c")}<th class="ro-b">${esc(part.actor)}${ui.who === "after" ? " after a check" : ""}</th>`
      + `${acts.map((n, i) => th("a" + i, esc(label(n)), "n")).join("")}${th("ev0", `${esc(o)} EV`, "n")}${th("ev1", `${esc(p)} EV`, "n")}`
      + `${th("eq0", `${esc(o)} eq`, "n ro-eq")}${th("eq1", `${esc(p)} eq`, "n ro-eq")}</tr>`;
    const sum = average(rs);
    let body = `<tr class="ro-sum"><td class="ro-c">${rs.length === all.length ? `All ${all.length} cards` : `${rs.length} card${rs.length === 1 ? "" : "s"}`}<small>${pct(sum.p, 0)}% of ${ro.street}s</small></td><td class="ro-b">${bar(sum.f, acts)}</td>${nums(sum)}</tr>`;
    if (ui.group) for (const g of groups(rs, ui.group))
      body += `<tr class="ro-grp" data-rogroup="${ui.group}:${g.g}" tabindex="0" title="Only these cards"><td class="ro-c">${esc(g.label)}<small>${g.n} card${g.n === 1 ? "" : "s"}, ${pct(g.p, 1)}%</small></td><td class="ro-b">${bar(g.f, acts)}</td>${nums(g)}</tr>`;
    for (const r of rs)
      body += `<tr data-card="${r.card}" tabindex="0" title="Deal ${r.card}"><td class="ro-c"><span class="ro-card">${cardHtml1(r.card)}</span><small>${SHORT[r.cls]}</small></td>`
        + `<td class="ro-b">${bar(r.f, acts, r.fe)}</td>${nums(r)}</tr>`;
    return `<div class="ro-scroll"><table class="ro-t"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
  }
  // the controls over the table: grouping, the class filter and (a stored turn) the flop lines it holds
  function controlsHtml(ro, ui) {
    const sel = (id, opts, cur, aria) => `<select data-ctl="${id}" aria-label="${aria}">${opts.map(([v, t]) => `<option value="${v}"${String(v) === String(cur) ? " selected" : ""}>${esc(t)}</option>`).join("")}</select>`;
    let h = sel("rogroup", [["", "No groups"], ...Object.entries(GROUPS).map(([k, t]) => [k, "By " + t.toLowerCase()])], ui.group || "", "Group the cards")
      + sel("rofilter", [["", "Every card"], ...Object.entries(CLASSES)], (ui.filter || {}).cls || "", "Only cards that");
    if (ro.lines && ro.lines.length > 1) h += linesSelect(ro.lines, ro.line);
    return h;
  }
  // a stored turn's flop lines (Advisor._store_lines) as a select: each line's actions and how often the flop
  // solve takes it
  const lineLabel = (ln) => ln.flop.map(a => `${a.action === "all-in" ? "all-in" : a.action}${a.to != null ? " " + a.to : ""}`).join(", ");
  const linesSelect = (lines, cur) => `<select data-ctl="roline" aria-label="Flop line">${lines.map((ln, i) =>
    `<option value="${i}"${i === cur ? " selected" : ""}>Flop: ${esc(lineLabel(ln))} (${pct(ln.share, 0)}%)</option>`).join("")}</select>`;
  // the flop line a flop report's view opens the turn report on (reports.js's Turns): the most reached of the lines
  // held that pass through the view's decision: out of position's first (any line), in position's after a check (a
  // check first), out of position's facing a bet after a check (check, a bet, a response: the view's own size first,
  // else any); -1 where none does
  function pickLine(lines, view, bet) {
    const through = (p) => view === "ip" ? p[0] === "check" && p.length > 1
      : view === "vsbet" ? p[0] === "check" && /^bet /.test(p[1] || "") && p.length > 2 : p.length > 0;
    const best = (want) => (lines || []).reduce((b, ln, i) => want(ln.path || []) && (b < 0 || ln.share > lines[b].share) ? i : b, -1);
    const i = view === "vsbet" && bet ? best(p => through(p) && p[1] === bet) : -1;
    return i >= 0 ? i : best(through);
  }

  // what the numbers are, under the table
  function noteHtml(ro) {
    const s = ro.solver || {}, river = ro.street === "river", pot = ro.pot_bb != null ? ` (${bb(ro.pot_bb)}bb pot)` : "";
    const eq = `equity range against range${river ? "" : " (every river dealt)"}, each card's chance given both ranges`;
    if (ro.source === "store") {
      // (a turn file with the flop solve's EVs, format 2: EVs as a live report's, each action's in its bar's tip)
      const evs = (ro.cards || []).some(c => c.ev);
      return `From the flop library's stored turns: the precomputed flop solve's turn strategy (${s.iterations} iterations, ${s.exploitability_pct}% of the pot from equilibrium), `
        + `its turn sizes and each turn class's cards alike (a rank's cards, a flush suit's apart), mixes in 5% steps. `
        + (evs ? `EV in bb: each range's share of the pot${pot} on the flop solve's strategies, by turn class (the two add up to it), and in each bar's tip the EV of each action for the hands that take it; `
          : `These stored turns hold strategies only, so there are no EVs (—); `)
        + `${eq}. Groups average the mix by the acting range on each card${evs ? ", EV and equity by the cards' chances" : ""}. A turn you deal is solved on its own, with more sizes. Click a card to deal it.`;
    }
    return `Read off a live solve of the ${river ? "turn" : "flop"} (${s.iterations} iterations, ${s.exploitability_pct}% of the pot from equilibrium${s.by === "hand buckets" ? `, the ${esc(ro.street)} by hand buckets` : ""}), `
      + (river ? "which plays the river with its own few sizes: a river you deal is solved on its own, with its own menu of sizes. "
        : "which plays one turn bet size and each turn card's class alike (a rank's cards, a flush suit's apart): a turn you deal is solved on its own, with more sizes. ")
      + `EV in bb: each range's share of the pot${pot} on the solve's strategies (the two add up to it); ${eq}. Groups average the mix by the acting range on each card, EV and equity by the cards' chances. Click a card to deal it.`;
  }

  // styles on the page's tokens (--panel, --line, --text…), as the flop reports'
  const CSS = `
  .ro-scroll { overflow: auto; max-height: min(56vh, 520px); border: 1px solid var(--line); border-radius: 8px; }
  .ro-t { border-collapse: collapse; width: 100%; font-size: 12px; font-variant-numeric: tabular-nums; }
  .ro-t th { position: sticky; top: 0; z-index: 2; background: var(--panel2); color: var(--muted); font-weight: 500; text-align: left; padding: 6px 7px; border-bottom: 1px solid var(--line2); white-space: nowrap; cursor: pointer; user-select: none; }
  .ro-t th[aria-sort="ascending"], .ro-t th[aria-sort="descending"] { color: var(--accent); }
  .ro-t td { padding: 4px 7px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  .ro-t .n { text-align: right; }
  .ro-t .ro-c { position: sticky; left: 0; z-index: 1; background: var(--panel2); min-width: 70px; }
  .ro-t th.ro-c { z-index: 3; }
  .ro-t td.ro-c small { display: block; color: var(--faint); font-size: 10px; line-height: 1.1; }
  .ro-t .ro-b { width: 32%; min-width: 110px; }
  .ro-t tbody tr[data-card], .ro-t tr.ro-grp { cursor: pointer; }
  .ro-t tbody tr[data-card]:hover td, .ro-t tr.ro-grp:hover td { background: var(--line); }
  .ro-t tr.ro-sum td, .ro-t tr.ro-sum td.ro-c { font-weight: 600; background: var(--panel); }
  .ro-t tr.ro-grp td, .ro-t tr.ro-grp td.ro-c { background: var(--panel); }
  .ro-card span { font-weight: 600; }
  .ro-bar { display: flex; height: 12px; border-radius: 3px; overflow: hidden; background: var(--line); }
  .ro-bar i { display: block; height: 100%; }
  .ro-na { color: var(--faint); font-size: 11px; }
  @media (max-width: 600px) { .ro-t .ro-eq { display: none; } .ro-t .ro-b { min-width: 80px; } }`;
  function style() {
    if (typeof document === "undefined" || document.getElementById("roStyle")) return;
    const s = document.createElement("style"); s.id = "roStyle"; s.textContent = CSS; document.head.appendChild(s);
  }

  const api = { cardClass, CLASSES, GROUPS, LABELS, partOf, rows, filter, sort, sortKey, average, groups, html, controlsHtml,
                lineLabel, linesSelect, pickLine, noteHtml, style };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Runouts = api;
})();
