// The one-size view ("Simple"): at a heads-up decision after the flop where the player to act can bet, the street
// solved from there with one bet size at that decision, for each candidate (25% to 150% of the pot, the river 200%
// too, all-in where the tree offers it), and with every candidate to mix freely (the full menu); facing a bet, the
// same with one raise size (2.5x, 3x, 4x and 5x the bet, all-in where the tree offers it). The best single size
// is the one that keeps the most EV for the player betting (or raising); each size's EV given up is the full menu's EV
// less its own, in % of the pot (gto/advisor.py Advisor._one_size_view; README, One bet size). Loaded by index.html the
// first time the advice's Simple tab or Study's Simple view is opened.
// The page asks one solve per request (the request's state plus "one_size": "full", then each size) so it shows
// progress, can stop between or during solves, and keeps what it has: the runs here per spot and the advisor's own
// rows (a size asked again answers at once). The pure parts (the run's bookkeeping, the verdict, the HTML) are what the
// Node tests call.
(function () {
  "use strict";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const pct = (x, d = 0) => x == null ? "—" : (x * 100).toFixed(d) + "%";
  const evs = (x) => x == null ? "—" : (x > 0 ? "+" : "") + x.toFixed(2);
  const now = () => (typeof performance === "object" ? performance.now() : Date.now()) / 1000;
  // the request fields that pick the speed and its budgets: the page's ask sets them (the view is solved at Fast), so a
  // spot's run is the same at either speed
  const SPEED_KEYS = ["speed", "budget", "prior_budget", "quick", "early"];
  // seconds a solve takes, before any has answered here (the browser with threads, the local server; measured with
  // scripts/bench_browser.mjs and bench_one_size.py): the flop's estimate is what its opt-in button says
  const EST = { wasm: { flop: 0.3, turn: 1.5, river: 0.05 }, server: { flop: 0.2, turn: 1, river: 0.05 } };
  const N_SIZES = { flop: 7, turn: 7, river: 8 };   // candidates before all-in (advisor.ONE_SIZES, ONE_SIZES_RIVER)
  const N_RAISES = 4;                                // ...facing a bet (advisor.ONE_RAISES)
  // an advice's decision: "raise" facing a bet (fold and call to choose from), else "bet"
  const kindOf = (res) => res && Array.isArray(res.actions) && !res.actions.some((a) => a.action === "check")
    && res.actions.some((a) => a.action === "fold") ? "raise" : "bet";

  // ---- the runs: one per spot (the advice's state less its speed), kept for the page's life
  const runs = new Map();
  function baseKey(key) {
    const st = typeof key === "string" ? JSON.parse(key) : { ...key };
    for (const k of SPEED_KEYS) delete st[k];
    delete st.one_size;
    return JSON.stringify(st);
  }
  function runOf(key, street, kind) {
    const k = baseKey(key);
    if (!runs.has(k)) runs.set(k, { key: k, street: street || null, kind: kind || "bet", status: "idle", meta: null, candidates: null,
                                    full: null, rows: {}, cur: null, p: null, error: null, secs: [], sel: null, userStopped: false });
    return runs.get(k);
  }
  // the next solve a run needs: the full menu first (its answer names the candidates and the order to solve them in:
  // the sizes the full menu bets most first, so the solves of the sizes clearly behind them stop early), then each size
  const nextOf = (run) => !run.full ? "full" : ((run.meta && run.meta.order) || run.candidates || []).find((c) => !run.rows[c]) || null;
  const doneOf = (run) => (run.full ? 1 : 0) + (run.candidates || []).filter((c) => run.rows[c]).length;
  const totalOf = (run) => run.candidates ? run.candidates.length + 1 : (run.kind === "raise" ? N_RAISES : N_SIZES[run.street || "turn"]) + 1;
  // an answer's one_size into the run: its spot, the candidates, the full menu's row and every size's row it has
  function merge(run, o) {
    const { rows, full, best, tied, ...meta } = o;
    run.meta = meta; run.street = meta.street || run.street; run.kind = meta.kind || run.kind || "bet";
    run.candidates = meta.candidates || run.candidates;
    if (full) run.full = full;
    for (const r of rows || []) { const { loss_pct, ...row } = r; run.rows[row.size] = row; }
    return run;
  }
  // the verdict over the rows in hand (as gto/advisor.py one_size_summary): each size's EV given up against the full
  // menu (loss_pct, % of the pot; null before the full menu's row), the best (the highest EV) and the sizes tied with
  // it (within the precision of the two solves: the larger of their exploitabilities)
  function summarize(run) {
    const pot = run.meta ? run.meta.pot_bb : 1, full = run.full;
    const order = [...(run.candidates || []), ...Object.keys(run.rows).filter((s) => !(run.candidates || []).includes(s))];
    const rows = order.filter((s) => run.rows[s]).map((s) => {
      const r = run.rows[s];
      return { ...r, loss_pct: full ? Math.round((full.ev_bb - r.ev_bb) / Math.max(pot, 1e-9) * 1e5) / 1e3 : null };
    });
    if (!rows.length) return { rows, best: null, tied: [] };
    const top = rows.reduce((a, r) => r.ev_bb > a.ev_bb ? r : a);
    const tol = (r) => Math.max(r.exploitability_pct || 0, top.exploitability_pct || 0) / 100 * pot;
    return { rows, best: top.size, tied: rows.filter((r) => r !== top && top.ev_bb - r.ev_bb <= tol(r)).map((r) => r.size) };
  }
  // seconds left: the solves still to do at the pace the run's answers came (the full menu's first one aside: it
  // carries the narrowing of the streets before), else the estimate
  function etaOf(run, mode) {
    const left = totalOf(run) - doneOf(run);
    const per = run.secs.length > 1 ? run.secs.slice(1).reduce((a, b) => a + b, 0) / (run.secs.length - 1)
      : (EST[mode] || EST.wasm)[run.street || "turn"];
    return Math.max(0, left * per);
  }

  // Runs a spot's solves one request at a time until every size is in, it's stopped or one fails. ask(state) is the
  // page's solver call (a promise of the answer; {cancelled} once stopped); onUpdate(run) after each answer (the run's
  // latest: a view shown again while it runs takes the updates over).
  async function start(key, ask, onUpdate) {
    const run = runOf(key);
    if (onUpdate) run.onUpdate = onUpdate;
    if (run.status === "running") return run;
    run.status = "running"; run.error = null; run.userStopped = false;
    const gen = run.gen = (run.gen || 0) + 1;   // (a run stopped and started again before its last answer came back:
    const upd = () => { if (run.onUpdate) run.onUpdate(run); };   // that answer is the old loop's, left alone)
    for (let label; (label = nextOf(run));) {
      run.cur = label;
      upd();
      const t0 = now();
      let r;
      try { run.p = ask({ ...JSON.parse(run.key), one_size: label }); r = await run.p; }
      catch (e) { r = { error: "Request failed: " + (e && e.message || e) }; }
      if (run.gen !== gen) return run;                   // started again meanwhile: the new loop carries on
      run.p = null; run.cur = null;
      if (run.status !== "running") return run;          // stopped meanwhile
      if (!r || r.cancelled) { run.status = "stopped"; upd(); return run; }
      if (r.error || !r.one_size) { run.status = "error"; run.error = r.error || "No answer."; upd(); return run; }
      run.secs.push(now() - t0);
      merge(run, r.one_size);
      upd();
    }
    run.status = "done"; upd();
    return run;
  }
  // Stops the runs in flight (or `key`'s): the request running is stopped through stopReq(promise) (the page's
  // stopRequests or POST /cancel); what's in hand stays, and showing the spot again carries on (unless user: the Stop
  // button, then its Continue button does)
  function stop(key, stopReq, user = false) {
    const k = key == null ? null : baseKey(key);
    for (const run of runs.values()) {
      if (run.status !== "running" || (k && run.key !== k)) continue;
      run.status = "stopped"; run.userStopped = user;
      if (run.p && stopReq) { try { stopReq(run.p); } catch (e) {} }
    }
  }

  // ---- the HTML
  // bets by size: light to dark as in the page's shading; all-in its own
  function colorOf(name) {
    if (name === "check" || name === "call") return "var(--passive)";
    if (name === "fold") return "var(--fold)";
    if (/all-in/.test(name)) return "var(--allin)";
    const m = /([\d.]+)%/.exec(name), x = m ? +m[1] : 75;
    return `var(--agg${x <= 33 ? 0 : x <= 75 ? 1 : x <= 125 ? 2 : 3})`;
  }
  // the share of the range that bets (or raises: neither checks, folds nor calls)
  const betShare = (r) => r.actions.reduce((t, n, i) => t + (n === "check" || n === "fold" || n === "call" ? 0 : r.freq[i]), 0);
  const sizeName = (s) => s === "full" ? "Every size" : s === "all-in" ? "All-in" : s;
  // a row's swatch: a bet by its size; a raise by its size after calling (the answer's raises: {to, pct})
  const swatch = (run, s) => {
    const x = run.kind === "raise" && run.meta && run.meta.raises && run.meta.raises[s];
    return colorOf(s === "all-in" ? "all-in" : x ? `raise ${x.pct}%` : "bet " + s);
  };
  const raiseTo = (run, s) => {
    const x = run.kind === "raise" && run.meta && run.meta.raises && run.meta.raises[s];
    return x && s !== "all-in" ? ` <span class="muted">to ${x.to}</span>` : "";
  };
  const mixText = (r) => r.actions.map((n, i) => r.freq[i] >= 0.005 ? `${esc(n)} ${pct(r.freq[i])}` : "").filter(Boolean).join(" · ");
  const RANKS = "AKQJT98765432";
  const classAt = (i, j) => i === j ? RANKS[i] + RANKS[i] : i < j ? RANKS[i] + RANKS[j] + "s" : RANKS[j] + RANKS[i] + "o";
  // a size's strategy over the range (the page's grid: each class's mix stacked, faint where the range has little)
  function gridHtml(grid, hand) {
    let h = `<div class="grid os-grid">`;
    for (let i = 0; i < 13; i++) for (let j = 0; j < 13; j++) {
      const cls = classAt(i, j), c = grid.cells[cls] || { w: 0, f: [] }, me = cls === hand;
      const op = c.w <= 0 ? 0.08 : 0.25 + 0.75 * Math.min(1, c.w);
      const stack = c.w > 0 ? grid.actions.map((n, a) => `<div style="width:${(c.f[a] * 100).toFixed(1)}%;background:${colorOf(n)}"></div>`).join("") : "";
      h += `<div class="cell${i === j ? " pair" : ""}${me ? " me" : ""}" data-h="${cls}" style="opacity:${op}"><div class="stack">${stack}</div><span class="lbl">${cls}</span>${me ? '<span class="you">YOU</span>' : ""}</div>`;
    }
    return h + `</div>`;
  }
  function readoutHtml(grid, cls, hand) {
    if (!cls) return "";
    const c = grid.cells[cls];
    return `<b>${esc(cls)}${cls === hand ? " (you)" : ""}</b>` + (c && c.w > 0
      ? grid.actions.map((n, a) => c.f[a] >= 0.005 ? `<span class="a" style="--c:${colorOf(n)}">${esc(n)} ${pct(c.f[a])}</span>` : "").join("")
      : `<span class="muted">not in the range here</span>`);
  }
  // the progress line: the solves done of all, the one running, the time left; Stop / Continue / the flop's opt-in
  function progressHtml(run, mode) {
    const n = totalOf(run), d = doneOf(run), eta = Math.round(etaOf(run, mode));
    const s = run.street || "turn";
    if (run.status === "idle" && s === "flop")
      return `<div class="os-go"><div>A flop solve per size: ${n} solves of the flop (every size alone and all of them mixed), about ${eta} s in all.</div>`
        + `<button data-os="start">Find the best size</button></div>`;
    if (run.status === "error")
      return `<div class="os-go"><div class="err">${esc(run.error)}</div>${d < n ? `<button data-os="start">Try again</button>` : ""}</div>`;
    if (run.status === "done") return "";
    const bar = `<div class="prog"><div style="width:${(d / n * 100).toFixed(1)}%"></div></div>`;
    if (run.status === "running")
      return `<div class="os-prog"><span class="spinner"></span><span>Solving ${run.cur ? sizeName(run.cur).toLowerCase() : "…"} · ${d} of ${n} done${d ? ` · about ${eta} s left` : ""}</span>`
        + `<button class="ghost" data-os="stop">Stop</button></div>${bar}`;
    return `<div class="os-prog"><span>${d ? `Stopped: ${d} of ${n} solved, about ${eta} s left.` : `${n} solves, about ${eta} s.`}</span><button data-os="start">${d ? "Continue" : "Solve"}</button></div>${bar}`;
  }
  // the panel: the verdict, a row per size (its bet share, the range's EV, the EV it gives up), the size picked's
  // strategy (its mix, your hand's, the grid)
  function panelHtml(run, opts = {}) {
    const m = run.meta || {}, sum = summarize(run), full = run.full;
    const raise = run.kind === "raise", one = raise ? "raise size" : "size";
    let h = `<div class="os-head"><b>One ${raise ? "raise" : "bet"} size</b><span class="muted"> · ${m.actor ? `${esc(m.actor)}'s ${esc(m.street)} decision` : `the ${esc(run.street || "")} decision`}${m.pot_bb != null ? ` · pot ${m.pot_bb} bb, ${m.stack_bb} behind` : ""} · solved at Fast</span></div>`;
    h += progressHtml(run, opts.mode || "wasm");
    if (!sum.rows.length && !full) return h;
    const best = sum.rows.find((r) => r.size === sum.best);
    if (best && full) {
      const all = run.candidates && run.candidates.every((c) => run.rows[c]);
      h += `<div class="os-best">${all ? `Best single ${one}` : "Best so far"}: <b>${esc(sizeName(best.size))}</b>${raiseTo(run, best.size)}`
        + (sum.tied.length ? ` <span class="muted">(as good within the solves' precision: ${sum.tied.map(sizeName).map(esc).join(", ")})</span>` : "")
        + ` · gives up <b>${Math.max(0, best.loss_pct).toFixed(2)}%</b> of the pot against every ${one} mixed</div>`;
    }
    const sel = run.sel && (run.sel === "full" ? full : run.rows[run.sel]) ? run.sel : sum.best || (full ? "full" : null);
    const tip = `EV given up: the range's EV with every ${one} to mix freely less its EV with this size alone, % of the pot`;
    h += `<table class="os-table"><thead><tr><th>${raise ? "Raise" : "Size"}</th><th class="n">${raise ? "Raises" : "Bets"}</th><th class="n" title="The range's EV from here, bb">EV</th><th class="n" title="${tip}">Gives up</th></tr></thead><tbody>`;
    if (full) h += `<tr data-size="full" class="os-full${sel === "full" ? " sel" : ""}" tabindex="0"><td>Every size, mixed</td><td class="n">${pct(betShare(full))}</td><td class="n">${evs(full.ev_bb)}</td><td class="n">—</td></tr>`;
    const known = new Set(sum.rows.map((r) => r.size));
    for (const r of sum.rows) {
      // (a size whose solve stopped once it was clearly behind the best: its EV given up as precise as it was then)
      const loss = r.loss_pct == null ? "—" : r.cut ? `<span title="Stopped once clearly behind the best: within ${r.exploitability_pct}% of the pot">≈${Math.max(0, r.loss_pct).toFixed(1)}%</span>`
        : `${r.loss_pct < 0 ? "0.00" : r.loss_pct.toFixed(2)}%`;
      h += `<tr data-size="${esc(r.size)}" class="${r.size === sum.best ? "best" : ""}${sum.tied.includes(r.size) ? " tied" : ""}${sel === r.size ? " sel" : ""}" tabindex="0">`
        + `<td><i style="background:${swatch(run, r.size)}"></i>${esc(sizeName(r.size))}${raiseTo(run, r.size)}${r.size === sum.best ? ' <span class="os-tag">best</span>' : ""}</td>`
        + `<td class="n">${pct(betShare(r))}</td><td class="n">${evs(r.ev_bb)}</td><td class="n">${loss}</td></tr>`;
    }
    for (const c of run.candidates || []) if (!known.has(c))
      h += `<tr class="os-wait"><td>${esc(sizeName(c))}</td><td class="n" colspan="3">${run.cur === c ? '<span class="spinner"></span>' : "…"}</td></tr>`;
    h += `</tbody></table>`;
    const r = sel === "full" ? full : run.rows[sel];
    if (r) {
      h += `<div class="os-strat"><b>${esc(sizeName(r.size))}</b>: ${mixText(r)}</div>`;
      if (r.hand && m.hand) h += `<div class="os-hand">Your ${esc(m.hand)}${m.hole ? ` (${esc(m.hole)})` : ""}: ${r.actions.map((n, i) => r.hand.freq[i] >= 0.005 ? `${esc(n)} ${pct(r.hand.freq[i])} <span class="muted">${evs(r.hand.ev[i])} bb</span>` : "").filter(Boolean).join(" · ")}</div>`;
      h += `<div class="rangewrap">${gridHtml(r.grid, m.hand)}<div class="rangeside"><div class="readout"></div>`
        + `<div class="legend">${r.grid.actions.map((n) => `<span style="--c:${colorOf(n)}">${esc(n)}</span>`).join("")}</div></div></div>`;
    }
    h += `<div class="os-note">Each size is a solve of the ${esc(m.street || run.street || "street")} from this decision with the same ranges, the opponent adapting to it; only this decision's ${raise ? "raise" : "bet"} changes (the rest of the tree is the advice's). `
      + (raise ? `Sizes are multiples of the bet faced (3x: a raise to three times it). ` : "")
      + `Best: the ${one} that keeps the most EV for the player ${raise ? "facing the bet" : "betting"}. Gives up: the range's EV with every ${one} to mix freely less with that size alone, in % of the pot`
      + `${m.target_pct != null ? `; each solve is within ${m.target_pct}% of the pot of equilibrium, so sizes closer than that are a tie` : ""}`
      + `${sum.rows.some((r) => r.cut) ? "; ≈: a size whose solve stopped early, once it was clearly behind the best" : ""}.`
      + (m.tournament ? ` In a tournament every solve pays prize equity (ICM), so EVs are chips worth as much prize equity.<br>${m.tournament.map(esc).join("<br>")}` : "")
      + `</div>`;
    return h;
  }

  const CSS = `
  .os-head { font-size: 13px; margin: 2px 0 8px; }
  .os-prog { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); flex-wrap: wrap; }
  .os-prog button, .os-go button { font-size: 12px; padding: 4px 12px; }
  .os-go { display: flex; flex-direction: column; align-items: flex-start; gap: 8px; font-size: 13px; color: var(--muted); padding: 6px 0 10px; }
  .os-best { margin: 10px 0 8px; font-size: 13px; line-height: 1.5; }
  .os-table { border-collapse: collapse; width: 100%; max-width: 440px; font-size: 12px; font-variant-numeric: tabular-nums; }
  .os-table th { text-align: left; color: var(--muted); font-weight: 500; padding: 3px 6px; border-bottom: 1px solid var(--line); }
  .os-table td { padding: 4px 6px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  .os-table .n { text-align: right; }
  .os-table tbody tr[data-size] { cursor: pointer; }
  .os-table tbody tr[data-size]:hover { background: var(--panel2); }
  .os-table tr.sel { background: var(--panel2); box-shadow: inset 2px 0 0 var(--accent); }
  .os-table tr.best td:first-child, .os-table tr.tied td:first-child { font-weight: 600; }
  .os-table tr.os-full td { color: var(--muted); }
  .os-table tr.os-wait td { color: var(--faint); }
  .os-table td i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; }
  .os-tag { font-size: 10px; color: var(--accent); border: 1px solid var(--accent); border-radius: 4px; padding: 0 4px; margin-left: 4px; }
  .os-strat, .os-hand { font-size: 12px; margin-top: 10px; line-height: 1.5; }
  .os-note { color: var(--muted); font-size: 11px; line-height: 1.5; margin-top: 10px; }
  .os-grid { margin-top: 0; }`;

  // Shows the view of the advice `res` (its request: key) in box, starting or carrying on its run (a flop's waits for
  // its button); ask: the page's solver call at Fast; opts.stop(promise) stops a request; opts.mode "wasm" or "server".
  function show(box, res, key, ask, opts = {}) {
    if (typeof document === "object" && !document.getElementById("osStyle")) {
      const s = document.createElement("style"); s.id = "osStyle"; s.textContent = CSS; document.head.appendChild(s);
    }
    const run = runOf(key, res && res.street, kindOf(res));
    box.dataset.want = run.key;
    const draw = () => {
      if (box.dataset.want !== run.key) return;   // another view or spot since
      box.innerHTML = panelHtml(run, opts);
      const r = run.sel === "full" ? run.full : run.rows[run.sel || summarize(run).best];
      const out = box.querySelector(".readout"), g = box.querySelector(".os-grid");
      if (r && out && g) {
        const hand = run.meta && run.meta.hand;
        out.innerHTML = readoutHtml(r.grid, hand, hand);
        g.onmouseover = g.onclick = (e) => { const c = e.target.closest(".cell"); if (c) out.innerHTML = readoutHtml(r.grid, c.dataset.h, hand); };
        g.onmouseleave = () => { out.innerHTML = readoutHtml(r.grid, hand, hand); };
      }
    };
    run.onUpdate = draw;
    const go = () => start(run.key, ask, draw);
    box.onclick = (e) => {
      const b = e.target.closest("[data-os]");
      if (b) { if (b.dataset.os === "stop") { stop(run.key, opts.stop, true); draw(); } else go(); return; }
      const t = e.target.closest("tr[data-size]");
      if (t) { run.sel = t.dataset.size; draw(); }
    };
    box.onkeydown = (e) => { if ((e.key === "Enter" || e.key === " ") && e.target.dataset && e.target.dataset.size) { e.preventDefault(); e.target.click(); } };
    draw();
    // turns and rivers start (or carry on) by themselves; a flop once asked for, and nothing the Stop button stopped
    if ((run.status === "idle" && run.street !== "flop") || (run.status === "stopped" && !run.userStopped)) go();
    return run;
  }

  const api = { runs, baseKey, runOf, nextOf, merge, summarize, etaOf, start, stop, show, panelHtml, progressHtml, gridHtml, colorOf, kindOf, EST };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Simple = api;
})();
