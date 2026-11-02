// Session analyzer, loaded by index.html (after hh.js) the first time "Analyze session…" is opened from the ⋯ menu:
// a file of many hand histories (any site hh.js reads) is split into hands (splitHands), each imported as the page
// imports one (importHand: non-Hold'em hands, hands without your cards, duplicates and unreadable ones are counted and
// skipped), and every one of your decisions in the chosen scope is asked of the page's solver at Fast and graded as
// a whole-hand review grades it (gradeWith), one after another in a queue that can be paused, resumed (also after a
// reload: the session is kept in IndexedDB) or stopped. The report: EV lost (bb, bb/100) by street, position,
// postflop spot and preflop situation, your frequencies against the solver's on the same decisions, the biggest
// mistakes (a click opens that hand at that decision in the page), and the decisions as CSV or JSON. Each session's
// summary is also kept (its own IndexedDB entry, "history") and summed across sessions: the leaks that show over
// many sessions, and each session's EV lost over time.
// Everything but open() and the page host is pure and takes its host as an argument (the Node tests' stub solver).
(function () {
  "use strict";
  const STREETS = ["preflop", "flop", "turn", "river"], NEED = { preflop: 0, flop: 3, turn: 4, river: 5 };
  const SCOPES = { preflop: "Preflop only", library: "Flop-library spots (instant)", all: "All streets" };
  const STORE_KEY = "current", HISTORY_KEY = "history", HISTORY_MAX = 200, MIX_OK = 0.15, MIX_OFF = 0.03;
  const r2 = (x) => Math.round(x * 100) / 100;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const HH = () => typeof globalThis.splitHands === "function" ? globalThis : require("./hh.js");
  const clone = (x) => JSON.parse(JSON.stringify(x));

  // ---- reading a file
  // the site's hand number ("PokerStars Hand #2498…", "Poker Hand #HD20…", "HandId: #1887…", "#Game No : 1234", iPoker's gamecode)
  function handId(text) {
    const x = String(text).match(/gamecode="([^"]+)"/);
    if (x) return x[1];
    const head = String(text).split("\n").slice(0, 3).join(" ");
    const m = head.match(/(?:Hand|Game)\s*(?:No|Id)?\s*[:#]*\s*#?\s*([A-Z]*\d[\w-]*)/i);
    return m ? m[1] : "";
  }
  const skipKind = (msg) => /Only Hold'em/i.test(msg) ? "nonHoldem" : /hole cards aren't|has hole cards but no/i.test(msg) ? "noHero" : "failed";
  // your decisions in a hand state: [street, index in that street's actions]
  function decisions(st) {
    const out = [];
    for (const s of STREETS) (st.actions[s] || []).forEach((a, i) => { if (a.pos === st.hero) out.push([s, i]); });
    return out;
  }
  // The hands of a file: {hands: [{n, id, site, hero, hole, state, warn}], skipped: {nonHoldem, noHero, noDecisions,
  // duplicate, failed}, errors: [{n, id, msg}], total}. imp is the page's importHand. Async, yielding every chunk
  // hands so a big file doesn't freeze the page (progress(k, total) after each chunk).
  async function prepare(text, imp, progress, chunk = 100, pause = () => new Promise(r => setTimeout(r, 0))) {
    const parts = HH().splitHands(text), hands = [], errors = [], seen = new Set();
    const skipped = { nonHoldem: 0, noHero: 0, noDecisions: 0, duplicate: 0, failed: 0 };
    for (let k = 0; k < parts.length; k++) {
      if (k && k % chunk === 0) { if (progress) progress(k, parts.length); await pause(); }
      const t = parts[k], id = handId(t);
      let r;
      try { r = imp(t); } catch (e) {
        const msg = e && e.message || String(e), kind = skipKind(msg);
        skipped[kind]++; if (kind === "failed") errors.push({ n: k + 1, id, msg });
        continue;
      }
      const key = `${r.hand.site}|${id}|${r.hand.hole.join("")}`;
      if (id && seen.has(key)) { skipped.duplicate++; continue; }
      seen.add(key);
      if (!decisions(r.state).length) { skipped.noDecisions++; continue; }
      hands.push({ n: k + 1, id, site: r.hand.site, hero: r.hand.hero, hole: r.hand.hole.join(""), state: r.state,
                   ...(r.warnings.length ? { warn: r.warnings.join(" ") } : {}) });
    }
    return { hands, skipped, errors, total: parts.length };
  }
  // the decisions a scope asks about: preflop only; preflop and flops the library may hold (asked "library_only",
  // so anything else is skipped, not solved); everything
  const inScope = (scope, s) => scope === "all" || s === "preflop" || scope === "library" && s === "flop";
  function buildItems(hands, scope) {
    const out = [];
    hands.forEach((h, k) => decisions(h.state).forEach(([s, i]) => { if (inScope(scope, s)) out.push({ h: k, s, i }); }));
    return out;
  }
  // the page state at one of your decisions: the actions before it, the board up to its street
  function spotState(st, s, i) {
    const x = clone(st);
    x.actions[s] = x.actions[s].slice(0, i);
    for (const t of STREETS.slice(STREETS.indexOf(s) + 1)) x.actions[t] = [];
    for (let c = 2 + NEED[s]; c < 7; c++) x.cards[c] = null;
    return x;
  }

  // ---- what kind of decision it is
  const AGG = (a) => a.action === "raise" || a.action === "bet" || a.action === "all-in";
  // preflop: who raised, in order (an all-in after an all-in is taken as a call), and the calls before the first raise
  function preflopRaises(acts) {
    const raises = [];
    let limps = 0, callers = 0, shove = false;
    for (const a of acts) {
      const agg = a.action === "raise" || a.action === "bet" || a.action === "all-in" && !shove;
      if (a.action === "all-in") shove = true;
      if (agg) { raises.push(a.pos); callers = 0; } else if (a.action === "call" || a.action === "all-in") { if (raises.length) callers++; else limps++; }
    }
    return { raises, limps, callers };
  }
  // the preflop situation of a decision, from the actions before it
  function preflopSituation(before, hero) {
    const { raises, limps, callers } = preflopRaises(before), n = raises.length;
    if (!n) return limps ? (hero === "BB" ? "BB vs limp" : "vs limp") : "RFI";
    if (n === 1) return before.some(a => a.pos === hero) ? "limp vs raise" : hero === "BB" ? "BB defence" : hero === "SB" ? "SB vs open" : callers ? "vs open + call" : "vs open";
    if (n === 2) return raises[0] === hero ? "vs 3bet" : "cold vs 3bet";
    return "vs 4bet+";
  }
  const POT_TYPE = (n) => n <= 0 ? "Limped" : n === 1 ? "SRP" : n === 2 ? "3bet pot" : "4bet+ pot";
  // sp: the page's view of the spot (host.spot): facing a bet, in position, players left, preflop raises
  function classify(st, s, i, sp) {
    const hero = st.hero, pre = st.actions.preflop, rec = st.actions[s][i];
    const out = { street: s, pos: hero, act: actLabel(rec), cls: actClass(rec), facing: sp.toCall > 1e-9, pot: r2(sp.pot) };
    if (s === "preflop") { out.pf = preflopSituation(pre.slice(0, i), hero); return out; }
    const { raises } = preflopRaises(pre), pfr = raises[raises.length - 1];
    out.role = pfr === hero ? "PFR" : pfr ? "caller" : "limped";
    out.ip = !!sp.ip; out.multi = sp.players > 2; out.potType = POT_TYPE(sp.raises);
    out.spot = `${out.multi ? "Multiway" : out.potType} · ${out.ip ? "IP" : "OOP"} · ${out.facing ? "facing bet" : out.ip ? "checked to" : "first to act"}`;
    if (s === "turn") out.flopAggr = (st.actions.flop || []).some(a => a.pos === hero && AGG(a));
    return out;
  }
  const actLabel = (a) => a.action === "bet" ? `bet ${r2(a.to)}` : a.action === "raise" ? `raise to ${r2(a.to)}` : a.action;
  const actClass = (a) => a.action === "fold" ? "fold" : a.action === "check" || a.action === "call" ? "passive" : "aggr";
  // the hand's mix at the decision, by kind of action
  function mixOf(res) {
    const m = { fold: 0, passive: 0, aggr: 0 };
    for (const a of res.actions || []) m[/^fold/.test(a.action) ? "fold" : /^(check|call)/.test(a.action) ? "passive" : "aggr"] += a.freq || 0;
    return { fold: r2(m.fold * 100) / 100, passive: r2(m.passive * 100) / 100, aggr: r2(m.aggr * 100) / 100 };
  }
  const gradeClass = (g) => g.f >= MIX_OK ? "ok" : g.f >= MIX_OFF || g.r ? "meh" : "bad";

  // ---- the queue
  function newSession(prep, scope, name) {
    return { v: 1, name: name || "", created: Date.now(), scope, hands: prep.hands, skipped: prep.skipped, errors: prep.errors,
             total: prep.total, items: buildItems(prep.hands, scope), next: 0, results: [], timing: {}, status: "ready" };
  }
  // the same session in a wider scope: the decisions it didn't ask yet (and the flops "library_only" skipped)
  function widen(sess, scope) {
    const done = new Set(sess.results.filter(r => r.status !== "skipped").map(r => `${r.h}|${r.s}|${r.i}`));
    const keep = sess.results.filter(r => r.status !== "skipped");
    const todo = buildItems(sess.hands, scope).filter(it => !done.has(`${it.h}|${it.s}|${it.i}`));
    return { ...sess, scope, results: keep, items: [...keep.map(r => ({ h: r.h, s: r.s, i: r.i })), ...todo], next: keep.length, status: todo.length ? "paused" : "done" };
  }
  // one decision: the page's request for the spot (at Fast), the answer, the grade
  async function processItem(sess, it, host) {
    const hand = sess.hands[it.h], played = hand.state.actions[it.s][it.i];
    const base = { h: it.h, s: it.s, i: it.i };
    const sp = host.spot(spotState(hand.state, it.s, it.i));
    if (!sp) return { ...base, street: it.s, pos: hand.state.hero, act: actLabel(played), cls: actClass(played), status: "error", err: "the hand doesn't reach this decision" };
    Object.assign(base, classify(hand.state, it.s, it.i, sp));
    const rec = played.action === "all-in" && sp.allinAs ? { action: sp.allinAs } : played;
    const req = { ...sp.payload, ...(sess.scope === "library" && it.s === "flop" ? { library_only: true } : {}) };
    let res;
    try { res = await host.advise(req); } catch (e) { res = { error: String(e && e.message || e) }; }
    if (!res || res.cancelled) return { retry: true };
    if (res.skipped) return { ...base, status: "skipped", why: res.skipped };
    if (res.error) return { ...base, status: "error", err: String(res.error) };
    // a jam deeper than the answer's tree: asked again with all-in at every decision of the street (index.html jamOffTree)
    if (host.jamOffTree && host.jamOffTree(res, rec) && !host.grade(res, rec)) {
      let r2; try { r2 = await host.advise({ ...req, jam: true }); } catch (e) { r2 = null; }
      if (r2 && !r2.error && !r2.cancelled && r2.actions) res = r2;
    }
    const g = host.grade(res, rec);
    return { ...base, status: g ? "ok" : "ungraded", ...(g ? { g } : {}), mix: mixOf(res),
             src: res.precomputed ? "library" : it.s === "preflop" ? "chart" : res.multiway && !res.solved ? "rules" : "solve" };
  }
  // A runner over a session: run() works through its items until done, paused or stopped. A request the page
  // cancelled (its own solve took the solver) is asked again, up to 3 times. save(sess) is called every saveEvery ms
  // and when it stops; update() after each decision.
  function runner(sess, host, { update = () => {}, save = async () => {}, now = () => Date.now(), saveEvery = 2000 } = {}) {
    let stop = null, running = false, lastSave = 0, current = null;
    const R = {
      get running() { return running; }, get stopping() { return stop; }, get current() { return current; }, sess,
      async run() {
        if (running || sess.next >= sess.items.length) return;
        running = true; stop = null; sess.status = "running"; update();
        try {
          if (host.ready) await host.ready();
          while (sess.next < sess.items.length && !stop) {
            const it = sess.items[sess.next], t0 = now();
            current = it; update();
            const r = await processItem(sess, it, host);
            if (stop === "cancel") break;   // stopped: the answer in flight is dropped
            if (r.retry) { it.tries = (it.tries || 0) + 1; if (it.tries < 3) continue; Object.assign(r, { h: it.h, s: it.s, i: it.i, status: "error", err: "cancelled" }); delete r.retry; }
            r.sec = Math.round((now() - t0)) / 1000;
            const tm = sess.timing[it.s] || (sess.timing[it.s] = [0, 0]);
            tm[0] += r.sec; tm[1]++;
            sess.results.push(r); sess.next++;
            if (now() - lastSave >= saveEvery) { lastSave = now(); await save(sess); }
            update();
          }
        } finally {
          running = false; current = null;
          sess.status = sess.next >= sess.items.length ? "done" : stop === "cancel" ? "stopped" : "paused";
          stop = null; await save(sess); update();
        }
      },
      pause() { if (running) { stop = "pause"; update(); } },
      cancel() { if (running) stop = "cancel"; else if (sess.status !== "done") sess.status = "stopped"; update(); },
    };
    return R;
  }
  // seconds a decision takes at Fast before any is measured: the street's budget, plus each earlier postflop
  // street's narrowing budget; a library-only flop and preflop's chart are instant
  function estimate(s, scope, budgets) {
    const [b, p] = budgets || [2.5, 2.5];
    if (s === "preflop") return 0.2;
    if (s === "flop") return scope === "library" ? 0.4 : b + 0.5;
    return b + p * (s === "turn" ? 1 : 2) + 0.5;
  }
  // seconds left: each street's measured average once it has two answers, else the estimate
  function eta(sess, budgets) {
    let t = 0;
    for (let k = sess.next; k < sess.items.length; k++) {
      const s = sess.items[k].s, m = sess.timing[s];
      t += m && m[1] >= 2 ? m[0] / m[1] : estimate(s, sess.scope, budgets);
    }
    return t;
  }
  const scopeEta = (hands, scope, budgets) => buildItems(hands, scope).reduce((t, it) => t + estimate(it.s, scope, budgets), 0);
  const fmtDur = (s) => s < 1 ? "under a second" : s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.round(s / 60)} min` : `${Math.floor(s / 3600)} h ${Math.round(s % 3600 / 60)} min`;

  // ---- the report
  function tally(rows) {
    const graded = rows.filter(r => r.g), ev = graded.filter(r => r.g.l != null);
    const lost = r2(ev.reduce((t, r) => t + r.g.l, 0));
    return { n: rows.length, graded: graded.length, ev: ev.length, lost, avg: ev.length ? Math.round(lost / ev.length * 1000) / 1000 : null,
             ok: graded.filter(r => gradeClass(r.g) === "ok").length, bad: graded.filter(r => gradeClass(r.g) === "bad").length };
  }
  function groupBy(rows, key, order) {
    const m = new Map();
    for (const r of rows) { const k = key(r); if (k == null) continue; if (!m.has(k)) m.set(k, []); m.get(k).push(r); }
    const out = [...m].map(([k, rs]) => ({ key: k, ...tally(rs) }));
    return order ? out.sort((a, b) => (order.indexOf(a.key) + 1 || 99) - (order.indexOf(b.key) + 1 || 99)) : out.sort((a, b) => b.lost - a.lost || b.n - a.n);
  }
  // your frequency of a kind of action against the solver's for the same hands at the same decisions
  const FREQS = [
    ["Open (RFI)", "aggr", (r) => r.pf === "RFI"],
    ["BB defence: fold", "fold", (r) => r.pf === "BB defence"],
    ["BB defence: 3bet", "aggr", (r) => r.pf === "BB defence"],
    ["SB vs open: fold", "fold", (r) => r.pf === "SB vs open"],
    ["vs 3bet: fold", "fold", (r) => r.pf === "vs 3bet"],
    ["vs 3bet: 4bet", "aggr", (r) => r.pf === "vs 3bet"],
    ["Flop c-bet", "aggr", (r) => r.street === "flop" && r.role === "PFR" && !r.facing && !r.multi],
    ["Flop vs c-bet: fold", "fold", (r) => r.street === "flop" && r.role === "caller" && r.facing && !r.multi],
    ["Flop vs c-bet: raise", "aggr", (r) => r.street === "flop" && r.role === "caller" && r.facing && !r.multi],
    ["Turn barrel", "aggr", (r) => r.street === "turn" && r.role === "PFR" && r.flopAggr && !r.facing && !r.multi],
    ["Turn vs bet: fold", "fold", (r) => r.street === "turn" && r.facing],
    ["River bet", "aggr", (r) => r.street === "river" && !r.facing],
    ["River vs bet: fold", "fold", (r) => r.street === "river" && r.facing],
  ];
  function freqs(results) {
    const rows = results.filter(r => r.g && r.mix);
    return FREQS.map(([label, cls, f]) => {
      const rs = rows.filter(f);
      if (!rs.length) return null;
      return { label, cls, n: rs.length, you: rs.filter(r => r.cls === cls).length / rs.length, gto: rs.reduce((t, r) => t + r.mix[cls], 0) / rs.length };
    }).filter(Boolean);
  }
  const PF_ORDER = ["RFI", "vs limp", "BB vs limp", "vs open", "vs open + call", "SB vs open", "BB defence", "limp vs raise", "vs 3bet", "cold vs 3bet", "vs 4bet+"];
  function report(sess, top = 20) {
    const res = sess.results, done = res.filter(r => r.status === "ok" || r.status === "ungraded");
    const all = tally(done), handsDone = new Set(res.map(r => r.h)).size;
    const mistakes = done.filter(r => r.g && r.g.l > 0.005).sort((a, b) => b.g.l - a.g.l).slice(0, top);
    const offChart = done.filter(r => r.g && r.g.l == null && r.g.f < MIX_OFF).sort((a, b) => b.g.bf - a.g.bf).slice(0, top);
    const seats = [...new Set(sess.hands.map(h => h.state.players))].sort((a, b) => b - a), POS = (globalThis.HH_SEATS || HH().HH_SEATS);
    return {
      hands: sess.hands.length, handsDone, decisions: sess.items.length, done: res.length, ...all,
      bb100: handsDone ? r2(all.lost / handsDone * 100) : null,
      skipped: res.filter(r => r.status === "skipped").length, errors: res.filter(r => r.status === "error").length,
      ungraded: res.filter(r => r.status === "ungraded").length, rules: res.filter(r => r.status === "ungraded" && r.src === "rules").length,
      byStreet: groupBy(done, r => r.street, STREETS),
      byPos: groupBy(done, r => r.pos, [...new Set(seats.flatMap(n => POS[n] || []))]),
      bySpot: groupBy(done.filter(r => r.spot), r => r.spot),
      byPf: groupBy(done.filter(r => r.pf), r => r.pf, PF_ORDER),
      freqs: freqs(done), mistakes, offChart,
    };
  }
  // ---- export
  const boardAt = (st, s) => st.cards.slice(2, 2 + NEED[s]).filter(Boolean).join("");
  const COLS = ["hand", "hand_id", "site", "position", "hole", "street", "board", "situation", "action", "solver_freq", "best_action", "best_freq",
                "ev_bb", "best_ev_bb", "ev_lost_bb", "grade", "source", "status", "pot_bb", "facing_bet", "in_position", "seconds"];
  function rowsOf(sess) {
    return sess.results.map(r => {
      const h = sess.hands[r.h], g = r.g || {};
      return { hand: h.n, hand_id: h.id, site: h.site, position: h.state.hero, hole: h.hole, street: r.s, board: boardAt(h.state, r.s),
               situation: r.pf || r.spot || "", action: r.act || "", solver_freq: g.f, best_action: g.b, best_freq: g.bf, ev_bb: g.e, best_ev_bb: g.o,
               ev_lost_bb: g.l, grade: r.g ? gradeClass(r.g) : "", source: r.src || "", status: r.status, pot_bb: r.pot, facing_bet: r.facing,
               in_position: r.ip, seconds: r.sec };
    });
  }
  function toCSV(sess) {
    const cell = (v) => v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
    return [COLS.join(","), ...rowsOf(sess).map(r => COLS.map(c => cell(r[c])).join(","))].join("\n") + "\n";
  }
  function toJSON(sess) {
    const rp = report(sess, 50), strip = (r) => ({ hand: sess.hands[r.h].n, hand_id: sess.hands[r.h].id, street: r.s, action: r.act, situation: r.pf || r.spot, ev_lost_bb: r.g.l, solver_freq: r.g.f, best_action: r.g.b });
    return JSON.stringify({ name: sess.name, scope: sess.scope, created: new Date(sess.created).toISOString(), file: { hands: sess.total, analyzed: sess.hands.length, skipped: sess.skipped },
      summary: { ...rp, mistakes: rp.mistakes.map(strip), offChart: rp.offChart.map(strip) }, decisions: rowsOf(sess) }, null, 1);
  }

  // ---- across sessions: a summary of each analyzed session (newest first, at most HISTORY_MAX), summed
  // the hands a session holds, as one short string (the same file analyzed again, in any scope, has the same)
  function handsFp(hands) {
    const keys = hands.map(h => `${h.site}|${h.id}|${h.hole}|${h.state && h.state.hero}`).sort().join("\n");
    let a = 0x811c9dc5, b = 0;
    for (let k = 0; k < keys.length; k++) { const c = keys.charCodeAt(k); a = Math.imul(a ^ c, 16777619) >>> 0; b = (Math.imul(b, 31) + c) >>> 0; }
    return `${hands.length}:${a.toString(36)}${b.toString(36)}`;
  }
  const slim = (rows) => rows.map(({ key, n, graded, ok, bad, ev, lost }) => ({ key, n, graded, ok, bad, ev, lost }));
  function summarize(sess) {
    const rp = report(sess, 0);
    if (!rp.done) return null;
    return { id: sess.created, fp: handsFp(sess.hands), name: sess.name || "", scope: sess.scope, date: sess.created, hands: rp.handsDone, decisions: rp.done,
             graded: rp.graded, ok: rp.ok, bad: rp.bad, ev: rp.ev, lost: rp.lost, bb100: rp.bb100,
             byStreet: slim(rp.byStreet), byPos: slim(rp.byPos), byPf: slim(rp.byPf), bySpot: slim(rp.bySpot),
             freqs: rp.freqs.map(({ label, cls, n, you, gto }) => ({ label, cls, n, you, gto })) };
  }
  // the history with this summary in, in place of an older one of the same session or the same hands (a file analyzed
  // again is counted once)
  function remember(list, sum) {
    const L = Array.isArray(list) ? list.filter(x => x && x.id != null) : [];
    if (!sum) return L;
    return [sum, ...L.filter(x => x.id !== sum.id && !(sum.fp && x.fp === sum.fp))].sort((a, b) => b.date - a.date).slice(0, HISTORY_MAX);
  }
  // rows of the same key summed over sessions, the most EV lost first (or in `order`)
  function mergeRows(groups, order) {
    const m = new Map();
    for (const g of groups) for (const r of g || []) {
      const t = m.get(r.key) || { key: r.key, n: 0, graded: 0, ok: 0, bad: 0, ev: 0, lost: 0 };
      for (const k of ["n", "graded", "ok", "bad", "ev"]) t[k] += r[k] || 0;
      t.lost = r2(t.lost + (r.lost || 0));
      m.set(r.key, t);
    }
    const out = [...m.values()].map(t => ({ ...t, avg: t.ev ? Math.round(t.lost / t.ev * 1000) / 1000 : null }));
    return order ? out.sort((a, b) => (order.indexOf(a.key) + 1 || 99) - (order.indexOf(b.key) + 1 || 99)) : out.sort((a, b) => b.lost - a.lost || b.n - a.n);
  }
  // Leaks over the sessions, the clearest first: a frequency of yours 10+ points off the solver's on the same hands (15+
  // decisions), a preflop situation where under 80% of your actions are in the chart's mix (15+ graded), a postflop spot
  // losing 0.5bb+ a decision (10+ with EVs). Each scored by how far off times the square root of its decisions.
  const LEAK = { freq: [0.1, 15], pf: [0.8, 15], spot: [0.5, 10] };
  function leaks(tr) {
    const out = [];
    for (const f of tr.freqs) {
      const d = f.you - f.gto;
      if (f.n >= LEAK.freq[1] && Math.abs(d) >= LEAK.freq[0])
        out.push({ kind: "freq", key: f.label, n: f.n, you: f.you, gto: f.gto, score: Math.abs(d) * Math.sqrt(f.n) });
    }
    for (const r of tr.byPf) {
      if (r.graded >= LEAK.pf[1] && r.ok / r.graded < LEAK.pf[0])
        out.push({ kind: "pf", key: r.key, n: r.graded, inMix: r.ok / r.graded, score: (1 - r.ok / r.graded) * Math.sqrt(r.graded) });
    }
    for (const r of tr.bySpot) {
      if (r.ev >= LEAK.spot[1] && r.avg >= LEAK.spot[0])
        out.push({ kind: "spot", key: r.key, n: r.ev, avg: r.avg, lost: r.lost, score: r.avg / 2 * Math.sqrt(r.ev) });
    }
    return out.sort((a, b) => b.score - a.score);
  }
  function trends(list) {
    const L = Array.isArray(list) ? list : [];
    const sum = (k) => L.reduce((t, x) => t + (x[k] || 0), 0);
    const hands = sum("hands"), lost = r2(sum("lost")), fm = new Map();
    for (const x of L) for (const f of x.freqs || []) {
      const t = fm.get(f.label) || { label: f.label, cls: f.cls, n: 0, you: 0, gto: 0 };
      t.n += f.n; t.you += f.you * f.n; t.gto += f.gto * f.n;
      fm.set(f.label, t);
    }
    const order = FREQS.map(f => f[0]);
    const tr = { sessions: L.length, hands, decisions: sum("decisions"), graded: sum("graded"), ok: sum("ok"), bad: sum("bad"), ev: sum("ev"), lost,
                 bb100: hands ? r2(lost / hands * 100) : null,
                 byStreet: mergeRows(L.map(x => x.byStreet), STREETS), byPos: mergeRows(L.map(x => x.byPos)),
                 byPf: mergeRows(L.map(x => x.byPf), PF_ORDER), bySpot: mergeRows(L.map(x => x.bySpot)),
                 freqs: [...fm.values()].map(t => ({ ...t, you: t.you / t.n, gto: t.gto / t.n })).sort((a, b) => order.indexOf(a.label) - order.indexOf(b.label)),
                 // oldest first: each session's EV lost and share in the mix, for the trend
                 series: L.slice().reverse().map(x => ({ id: x.id, date: x.date, name: x.name, hands: x.hands, bb100: x.ev ? x.bb100 : null,
                                                         inMix: x.graded ? x.ok / x.graded : null })) };
    tr.leaks = leaks(tr);
    return tr;
  }

  // ---- storage: the session in IndexedDB (localStorage if that's missing); a big session is a few MB
  function idbStore(db = "gto-session", key = STORE_KEY) {
    const open = () => new Promise((res, rej) => {
      const q = indexedDB.open(db, 1);
      q.onupgradeneeded = () => q.result.createObjectStore("kv"); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error);
    });
    const tx = async (mode, f) => {
      const d = await open();
      return new Promise((res, rej) => { const t = d.transaction("kv", mode), q = f(t.objectStore("kv")); t.oncomplete = () => { d.close(); res(q.result); }; t.onerror = t.onabort = () => { d.close(); rej(t.error); }; });
    };
    return { get: () => tx("readonly", s => s.get(key)), set: (v) => tx("readwrite", s => s.put(v, key)), del: () => tx("readwrite", s => s.delete(key)) };
  }
  function lsStore(key = "gto-session") {
    return { get: async () => { try { return JSON.parse(localStorage.getItem(key)); } catch (e) { return null; } },
             set: async (v) => { try { localStorage.setItem(key, JSON.stringify(v)); } catch (e) {} },
             del: async () => { try { localStorage.removeItem(key); } catch (e) {} } };
  }
  // (the page's storage, store.js, where it's loaded: the session in progress a blob, the history a record each)
  const defaultStore = (key = STORE_KEY) => globalThis.GTOStore ? (key === STORE_KEY ? globalThis.GTOStore.blob("session-current") : globalThis.GTOStore.sessions)
    : typeof indexedDB === "object" && indexedDB ? idbStore("gto-session", key) : lsStore(key === STORE_KEY ? "gto-session" : "gto-session-" + key);

  // ---- the page: its importer, engine, solver and grading (globals of index.html and hh.js)
  function pageHost() {
    /* global importHand, modeReady, BUDGETS, MODE, S, cleanState, setPositions, simulate, payload, callAdvise, heuristic, gradeWith, jamOffTree, gradedMove, loadHandState, rewindTo */
    return {
      importHand: (t) => importHand(t),
      ready: () => modeReady,
      budgets: () => MODE ? BUDGETS[MODE].fast : null,
      // the solver request at a decision (as the page sends it, at Fast, against GTO opponents) and the spot's shape
      spot(st) {
        const old = S;
        S = cleanState(clone(st)); setPositions(S.players);
        try {
          const sim = simulate();
          if (!sim.cur || sim.toAct !== S.hero) return null;
          const [budget, prior_budget] = BUDGETS[MODE].fast, live = sim.cur.order.filter(p => sim.active.has(p));
          const p = { ...payload(), budget, prior_budget, speed: "fast" }; delete p.opponents;
          // (an all-in here that nobody can match more of is graded as the call or check it is)
          const jam = gradedMove(sim, { action: "all-in" });
          return { payload: p, ...(jam.action !== "all-in" ? { allinAs: jam.action } : {}), pot: sim.cur.pot, toCall: sim.cur.curBet - sim.cur.bets[S.hero], ip: live[live.length - 1] === S.hero,
                   players: live.length, raises: sim.streets.preflop.nRaises };
        } finally { S = old; setPositions(old.players); }
      },
      advise: (p) => callAdvise(p),
      grade: (res, rec) => gradeWith(res, rec),
      jamOffTree: (res, rec) => jamOffTree(res, rec),
      // the whole hand, rewound to the decision (its Undo brings the rest back)
      open: async (st, s, i) => { await loadHandState(clone(st)); rewindTo(s, i); },
    };
  }

  // ---- the panel
  let box = null, host = null, store = null, hstore = null, history = [], clearArm = false, sess = null, run = null, draft = null, reading = null, scope = "library", drawT = null, loaded = false;
  // (a redraw between a press and its release would lose the click: not while a pointer is down on the panel)
  function schedule() { if (drawT) return; drawT = setTimeout(() => { drawT = null; if (box && box._down) schedule(); else draw(); }, 400); }
  // (a session that isn't running goes into the history too: its summary as it stands)
  const save = (x) => {
    if (x.status !== "running") keep(x);
    return store ? store.set(clone(x)).catch(() => {}) : Promise.resolve();
  };
  function keep(x) {
    const sum = summarize(x);
    if (!sum) return;
    history = remember(history, sum);
    if (hstore) hstore.set(clone(history)).catch(() => {});
  }
  function start(s) {
    sess = s; run = runner(sess, host, { update: schedule, save });
    return run.run();
  }
  const pct = (x) => x == null ? "—" : `${Math.round(x * 100)}%`;
  const bb = (x) => x == null ? "—" : x > 0.005 ? `<span class="num neg">−${x.toFixed(2)}</span>` : `<span class="num pos">0.00</span>`;
  const cardH = (c) => `<span class="ss-c ${c[1]}">${c[0]}${{ s: "♠", h: "♥", d: "♦", c: "♣" }[c[1]]}</span>`;
  const cardsH = (s) => (String(s).match(/../g) || []).map(cardH).join("");
  function table(title, rows, first, note) {
    if (!rows.length) return "";
    return `<h3>${esc(title)}</h3>${note ? `<div class="ss-note">${note}</div>` : ""}<div class="ss-tw"><table class="ss-t"><thead><tr><th>${esc(first)}</th><th class="n">Decisions</th><th class="n" title="in the solver's mix (15%+)">In mix</th><th class="n" title="EV given up, bb">EV lost</th><th class="n" title="per decision with EVs">Avg</th></tr></thead><tbody>`
      + rows.map(r => `<tr><td>${esc(r.key)}</td><td class="n">${r.n}</td><td class="n">${r.graded ? pct(r.ok / r.graded) : "—"}</td><td class="n">${r.ev ? bb(r.lost) : "<span class=muted>—</span>"}</td><td class="n">${r.avg == null ? "—" : r.avg.toFixed(2)}</td></tr>`).join("")
      + `</tbody></table></div>`;
  }
  function freqTable(freqs, title) {
    if (!freqs.length) return "";
    return `<h3>${esc(title)}</h3><div class="ss-note">On the same decisions with the same hands: how often you took the action, and the solver's average frequency for those hands.</div><div class="ss-tw"><table class="ss-t"><thead><tr><th>Node</th><th class="n">n</th><th class="n">You</th><th class="n">GTO</th><th class="n">Diff</th></tr></thead><tbody>`
      + freqs.map(f => { const d = Math.round((f.you - f.gto) * 100); return `<tr><td>${esc(f.label)}</td><td class="n">${f.n}</td><td class="n">${pct(f.you)}</td><td class="n">${pct(f.gto)}</td><td class="n ${Math.abs(d) >= 10 ? "neg" : ""}">${d > 0 ? "+" : ""}${d}</td></tr>`; }).join("") + `</tbody></table></div>`;
  }
  const leakText = (l) => l.kind === "freq" ? `<b>${esc(l.key)}</b>: you ${pct(l.you)}, GTO ${pct(l.gto)} on the same hands <small>(${l.n} decisions)</small>`
    : l.kind === "pf" ? `<b>${esc(l.key)}</b>: ${pct(l.inMix)} of your actions in the chart's mix <small>(${l.n} decisions)</small>`
    : `<b>${esc(l.key)}</b>: −${l.avg.toFixed(2)}bb a decision <small>(${l.n} decisions, −${l.lost.toFixed(1)}bb in all)</small>`;
  // each session's bb/100 lost, oldest to newest (two sessions with EVs or more)
  function sparkline(series) {
    const pts = series.filter(x => x.bb100 != null);
    if (pts.length < 2) return "";
    const W = 240, H = 40, hi = Math.max(...pts.map(x => x.bb100), 1e-9);
    const xy = pts.map((x, k) => `${(k / (pts.length - 1) * (W - 8) + 4).toFixed(1)},${(H - 4 - x.bb100 / hi * (H - 8)).toFixed(1)}`).join(" ");
    return `<div class="ss-sparkbox"><svg class="ss-spark" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="bb/100 lost per session, oldest to newest"><polyline points="${xy}" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>`
      + `<span class="ss-note">bb/100 lost per session, oldest to newest (lower is better)</span></div>`;
  }
  const dateOf = (t) => { const d = new Date(t); return isNaN(d) ? "" : d.toISOString().slice(0, 10); };
  // the history: leaks over every session kept (the current one as it stands), each session, the sums
  function historyHtml(list, title) {
    if (!list.length) return "";
    const tr = trends(list), tile = (v, l) => `<div class="ss-tile"><b>${v}</b><small>${l}</small></div>`;
    let h = `<h2 class="ss-h2">${esc(title)}</h2><div class="ss-note">Kept in this browser: a summary of every session analyzed here (not the hands themselves).</div>`
      + `<div class="ss-tiles">${tile(tr.sessions, tr.sessions === 1 ? "session" : "sessions")}${tile(tr.hands, "hands")}`
      + tile(tr.ev ? `−${tr.bb100.toFixed(1)}` : "—", "bb/100 hands") + tile(pct(tr.graded ? tr.ok / tr.graded : null), "in the solver's mix") + `</div>`;
    h += `<h3>Leaks</h3>` + (tr.leaks.length
      ? `<ol class="ss-leaks">${tr.leaks.slice(0, 8).map(l => `<li>${leakText(l)}</li>`).join("")}</ol>`
      : `<div class="ss-note">No clear leak yet. One shows once a frequency is 10+ points off the solver's over 15+ decisions, a preflop situation has under 80% of your actions in the chart's mix over 15+, or a postflop spot loses 0.5bb+ a decision over 10+.</div>`);
    h += `<h3>Sessions</h3>${sparkline(tr.series)}<div class="ss-tw"><table class="ss-t"><thead><tr><th>Date</th><th>File</th><th class="n">Hands</th><th class="n">In mix</th><th class="n">bb/100</th><th></th></tr></thead><tbody>`
      + list.map(x => `<tr><td>${dateOf(x.date)}</td><td>${esc(x.name || "—")} <small>${esc(SCOPES[x.scope] || "")}</small></td><td class="n">${x.hands}</td><td class="n">${pct(x.graded ? x.ok / x.graded : null)}</td>`
        + `<td class="n">${x.ev && x.bb100 != null ? bb(x.bb100) : "—"}</td><td class="n">${sess && x.id === sess.created ? "<small>this one</small>" : `<button class="link" data-forget="${x.id}" title="Remove this session from the history">Forget</button>`}</td></tr>`).join("")
      + `</tbody></table></div>`;
    h += table("By street, all sessions", tr.byStreet, "Street") + table("Preflop situations, all sessions", tr.byPf, "Situation")
      + table("Postflop spots, all sessions", tr.bySpot, "Spot") + freqTable(tr.freqs, "Your frequencies vs GTO, all sessions");
    h += `<div class="ss-row">${clearArm ? `<span class="ss-note">Forget every session's summary?</span><button data-clearhist="yes">Yes, clear the history</button><button class="link" data-clearhist="no">Cancel</button>`
      : `<button class="link" data-clearhist="ask">Clear history</button>`}</div>`;
    return h;
  }
  function decisionRow(r, k) {
    const h = sess.hands[r.h], g = r.g;
    return `<tr class="ss-go" data-k="${k}" title="Open this hand at this decision"><td>${cardsH(h.hole)}</td><td>${esc(h.state.hero)}</td><td>${esc(r.s)}${r.s !== "preflop" ? ` ${cardsH(boardAt(h.state, r.s))}` : ""}<div class="ss-sub">${esc(r.pf || r.spot || "")}</div></td>`
      + `<td><b>${esc(r.act)}</b> <small>${pct(g.f)}${g.v ? ` as the chart's ${esc(g.m)}` : ""}</small><div class="ss-sub">solver: ${esc(g.b)} ${pct(g.bf)}</div></td><td class="n">${g.l != null ? bb(g.l) : `<span class="g bad">off chart</span>`}</td></tr>`;
  }
  function reportHtml() {
    const rp = report(sess);
    if (!rp.done) return "";
    const tile = (v, l, t) => `<div class="ss-tile"${t ? ` title="${esc(t)}"` : ""}><b>${v}</b><small>${l}</small></div>`;
    let h = `<div class="ss-tiles">${tile(`${rp.handsDone}<small>/${rp.hands}</small>`, "hands")}${tile(rp.graded, "decisions graded")}`
      + tile(rp.ev ? `−${rp.lost.toFixed(2)}` : "—", "bb EV lost", `over ${rp.ev} decisions with EVs (postflop; preflop answers are the charts, graded by frequency only)`)
      + tile(rp.bb100 != null && rp.ev ? `−${rp.bb100.toFixed(1)}` : "—", "bb/100 hands") + tile(pct(rp.graded ? rp.ok / rp.graded : null), "in the solver's mix")
      + tile(rp.bad, "off the mix (<3%)") + `</div>`;
    const extra = [rp.skipped && `${rp.skipped} flop decisions not in the library (skipped)`, rp.rules && `${rp.rules} multiway decisions answered by the betting rules (not graded)`,
      rp.ungraded > rp.rules && `${rp.ungraded - rp.rules} decisions whose action the answer doesn't list (not graded)`, rp.errors && `${rp.errors} failed`].filter(Boolean);
    if (extra.length) h += `<div class="ss-note">${extra.join(" · ")}</div>`;
    h += table("By street", rp.byStreet, "Street") + table("By position", rp.byPos, "Position")
      + table("Preflop situations", rp.byPf, "Situation", "Preflop answers come from the charts: no EVs, graded by how often the solver plays your action.")
      + table("Postflop spots", rp.bySpot, "Spot", "Pot type, your position, and whether you faced a bet.");
    h += freqTable(rp.freqs, "Your frequencies vs GTO");
    const list = (title, rows, note) => rows.length ? `<h3>${title}</h3><div class="ss-note">${note}</div><div class="ss-tw"><table class="ss-t ss-m"><thead><tr><th>Hand</th><th>Seat</th><th>Street</th><th>You / solver</th><th class="n">EV lost</th></tr></thead><tbody>`
      + rows.map(r => decisionRow(r, sess.results.indexOf(r))).join("") + `</tbody></table></div>` : "";
    h += list("Biggest mistakes", rp.mistakes, "By EV given up. Click one to open the hand at that decision.");
    h += list("Preflop: off-chart plays", rp.offChart, "Actions the charts play under 3% of the time with that hand.");
    return h;
  }
  function inputHtml() {
    const est = (sc) => { if (!draft) return ""; const n = buildItems(draft.hands, sc).length; return `<small>${n} decisions · ~${fmtDur(scopeEta(draft.hands, sc, host.budgets()))}</small>`; };
    const sk = draft && draft.skipped, skl = sk ? [["nonHoldem", "not Hold'em"], ["noHero", "without your cards"], ["noDecisions", "no decision of yours"], ["duplicate", "duplicates"], ["failed", "unreadable"]].filter(([k]) => sk[k]).map(([k, l]) => `${sk[k]} ${l}`) : [];
    return `<label class="ss-drop" data-drop><input type="file" data-file accept=".txt,.xml,.log,.hh,text/plain,text/xml"><b>Drop a hand-history file here</b> or <u>choose a file</u><small>PokerStars, GGPoker, 888, partypoker, Winamax, WPN/ACR, Ignition/Bovada, iPoker: a whole session, any number of hands</small></label>`
      + `<textarea data-paste rows="4" placeholder="…or paste the hand histories here" spellcheck="false"></textarea>`
      + (reading ? `<div class="ss-note">${esc(reading)}</div>` : "")
      + (draft ? `<div class="ss-found"><b>${draft.hands.length} hand${draft.hands.length === 1 ? "" : "s"} to analyze</b>${draft.name ? ` from ${esc(draft.name)}` : ""} (${draft.total} in the file${skl.length ? `; skipped: ${skl.join(", ")}` : ""})`
        + `${draft.errors.length ? `<details><summary>${draft.errors.length} unreadable</summary>${draft.errors.slice(0, 20).map(e => `<div class="ss-sub">hand ${e.n}${e.id ? ` (#${esc(e.id)})` : ""}: ${esc(e.msg)}</div>`).join("")}</details>` : ""}</div>`
        + `<div class="ss-scopes" role="radiogroup">${Object.entries(SCOPES).map(([k, l]) => `<button role="radio" aria-checked="${k === scope}" data-scope="${k}" class="${k === scope ? "active" : ""}">${l}${est(k)}</button>`).join("")}</div>`
        + `<div class="ss-note">Each decision is solved at Fast, as a hand review would. Flop-library spots are the precomputed flops (heads-up, and the first decision of the commonest 3-way pots): instant, the rest of the flops skipped. All streets solves every turn and river too.</div>`
        + `<div class="ss-row"><button class="active" data-go${draft.hands.length ? "" : " disabled"}>Analyze</button></div>` : "");
  }
  function runHtml() {
    const n = sess.items.length, k = sess.next, rn = run && run.running, cur = run && run.current;
    const st = rn ? (run.stopping ? "Pausing after this decision…" : cur ? `Hand ${sess.hands[cur.h].n}${sess.hands[cur.h].id ? ` (#${esc(sess.hands[cur.h].id)})` : ""}, ${cur.s}` : "Starting…")
      : { done: "Done.", paused: "Paused.", stopped: "Stopped.", ready: "Ready." }[sess.status] || "";
    const sec = eta(sess, host.budgets()), left = sess.status === "done" || sess.status === "stopped" || !n ? "" : sec < 1 ? " · almost done" : ` · about ${fmtDur(sec)} left`;
    let btns = rn ? `<button data-pause${run.stopping ? " disabled" : ""}>Pause</button><button data-stop>Stop</button>`
      : sess.status === "paused" || sess.status === "ready" || sess.status === "stopped" && k < n ? `<button class="active" data-resume>${sess.status === "ready" ? "Start" : "Resume"}</button>` : "";
    if (!rn && sess.results.length) btns += `<button data-csv>Export CSV</button><button data-json>Export JSON</button>`;
    if (!rn && sess.status === "done" && sess.scope !== "all") btns += `<button data-widen title="Also ask the decisions this scope left out">${sess.scope === "preflop" ? "Add the flop-library spots" : "Add all streets"}</button>`;
    if (!rn) btns += `<button class="link" data-new>New session</button>`;
    const all = remember(history, summarize(sess));
    return `<div class="ss-prog"><div class="ss-bar"><i style="width:${n ? (k / n * 100).toFixed(1) : 0}%"></i></div><div class="ss-st"><span>${k} of ${n} decisions${left}</span><span>${st}</span></div>`
      + `<div class="ss-sub">${esc(sess.name || "Session")} · ${sess.hands.length} hands · ${SCOPES[sess.scope]}</div></div><div class="ss-row">${btns}</div>` + reportHtml()
      + (all.some(x => x.id !== sess.created) ? historyHtml(all, "Across your sessions") : "");
  }
  function draw() {
    if (!box || box.hidden) return;
    const body = box.querySelector(".ss-body"), y = body.scrollTop;
    body.innerHTML = !loaded ? `<div class="ss-note">Loading…</div>` : sess ? runHtml() : inputHtml() + historyHtml(history, "Your sessions so far");
    body.scrollTop = y;
  }
  async function readText(text, name) {
    draft = null; reading = "Reading the hands…"; draw();
    try {
      draft = await prepare(text, host.importHand, (k, n) => { reading = `Reading the hands… ${k} of ${n}`; draw(); });
      draft.name = name || "";
      reading = draft.total ? null : "No hand histories found in that.";
    } catch (e) { reading = "Couldn't read that: " + (e.message || e); }
    draw();
  }
  function download(name, text, type) {
    const a = document.createElement("a"), url = URL.createObjectURL(new Blob([text], { type }));
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const fileBase = () => (sess.name || "session").replace(/\.[^.]*$/, "").replace(/[^\w.-]+/g, "_") + "-analysis";
  async function act(d) {
    if (d.scope) { scope = d.scope; return draw(); }
    if (d.go != null && draft) { const s = newSession(draft, scope, draft.name); draft = null; return start(s); }
    if (d.resume != null && sess) return start(sess);
    if (d.pause != null && run) return run.pause();
    if (d.stop != null && sess) { if (run && run.running) run.cancel(); else { sess.status = "stopped"; await save(sess); } return draw(); }
    if (d.widen != null && sess) return start(widen(sess, sess.scope === "preflop" ? "library" : "all"));
    if (d.new != null) { if (run && run.running) return; keep(sess); sess = run = null; if (store) await store.del().catch(() => {}); return draw(); }
    if (d.csv != null) return download(fileBase() + ".csv", toCSV(sess), "text/csv");
    if (d.json != null) return download(fileBase() + ".json", toJSON(sess), "application/json");
    if (d.k != null) { const r = sess.results[+d.k], h = sess.hands[r.h]; close(); return host.open(h.state, r.s, r.i); }
    if (d.forget != null) { history = history.filter(x => String(x.id) !== d.forget); if (hstore) await hstore.set(clone(history)).catch(() => {}); return draw(); }
    if (d.clearhist != null) {
      clearArm = d.clearhist === "ask";
      if (d.clearhist === "yes") { history = []; if (hstore) await hstore.del().catch(() => {}); }
      return draw();
    }
  }
  function close() {
    if (box) box.hidden = true; const m = document.getElementById("menuBtn"); if (m) m.focus();
    if (globalThis.viewClosed) globalThis.viewClosed("session");   // (the page's URL back to the solver's)
  }
  const CSS = `
  .ss { position: fixed; inset: 0; z-index: 90; background: rgba(0,0,0,.55); display: flex; justify-content: center; align-items: flex-start; padding: 40px 16px 16px; }
  .ss-card { background: var(--panel2); color: var(--text); border: 1px solid var(--line2); border-radius: var(--r); width: min(760px, 100%); max-height: calc(100dvh - 56px); display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,.5); }
  .ss-head { display: flex; align-items: baseline; gap: 10px; padding: 10px 12px 6px; }
  .ss-head .sub { color: var(--muted); font-size: 12px; flex: 1; }
  .ss-x { margin-left: auto; padding: 2px 8px; }
  .ss-body { display: flex; flex-direction: column; gap: 10px; padding: 0 12px 12px; overflow: auto; }
  .ss-body > * { flex-shrink: 0; }
  .ss-body h3 { font-size: 13px; margin: 8px 0 0; }
  .ss-drop { display: flex; flex-direction: column; align-items: center; gap: 4px; text-align: center; border: 1.5px dashed var(--line2); border-radius: 8px; padding: 18px 10px; cursor: pointer; font-size: 13px; }
  .ss-drop.over { border-color: var(--accent); background: var(--line); }
  .ss-drop small, .ss-note, .ss-sub { color: var(--muted); font-size: 11px; line-height: 1.45; }
  .ss-drop input { position: absolute; width: 1px; height: 1px; opacity: 0; }
  .ss-body textarea { width: 100%; box-sizing: border-box; font: 12px/1.4 ui-monospace, monospace; resize: vertical; }
  .ss-found { font-size: 13px; }
  .ss-found summary { cursor: pointer; color: var(--muted); font-size: 12px; }
  .ss-scopes { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
  .ss-scopes button { display: flex; flex-direction: column; gap: 2px; text-align: left; padding: 8px 10px; }
  .ss-scopes small { color: var(--muted); font-weight: 400; } .ss-scopes .active small { color: inherit; opacity: .75; }
  .ss-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .ss-prog { display: flex; flex-direction: column; gap: 4px; }
  .ss-bar { height: 8px; border-radius: 4px; background: var(--bg); overflow: hidden; }
  .ss-bar i { display: block; height: 100%; background: var(--accent); transition: width .2s; }
  .ss-st { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; font-size: 12px; font-variant-numeric: tabular-nums; }
  .ss-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(105px, 1fr)); gap: 6px; }
  .ss-tile { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; display: flex; flex-direction: column; gap: 2px; }
  .ss-tile b { font-size: 18px; font-variant-numeric: tabular-nums; } .ss-tile b small { font-size: 12px; color: var(--muted); font-weight: 400; }
  .ss-tile > small { color: var(--muted); font-size: 11px; }
  .ss-tw { overflow-x: auto; }
  .ss-t { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  .ss-t th { color: var(--muted); font-weight: 500; text-align: left; padding: 3px 6px; border-bottom: 1px solid var(--line2); white-space: nowrap; }
  .ss-t td { padding: 4px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
  .ss-t .n { text-align: right; white-space: nowrap; } .ss-t td.neg { color: var(--bad); }
  .ss-t small { color: var(--muted); }
  .ss-go { cursor: pointer; } .ss-go:hover td { background: var(--line); }
  .ss-c { display: inline-block; min-width: 1.5em; padding: 0 2px; margin-right: 2px; border-radius: 3px; background: #f5f5f0; color: #111; font-weight: 700; text-align: center; font-size: 11px; line-height: 1.5; }
  .ss-c.h { color: #c62828; } .ss-c.d { color: #c62828; }
  body.four .ss-c.d { color: #1565c0; } body.four .ss-c.c { color: #2e7d32; }
  .ss-m td:first-child { white-space: nowrap; }
  .ss-h2 { font-size: 14px; margin: 14px 0 0; padding-top: 10px; border-top: 1px solid var(--line2); }
  .ss-leaks { margin: 0; padding-left: 20px; font-size: 12px; line-height: 1.6; } .ss-leaks small { color: var(--muted); }
  .ss-sparkbox { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; } .ss-spark { color: var(--bad); display: block; max-width: 100%; }
  @media (max-width: 600px) {
    .ss { padding: 0; }
    .ss-card { max-height: 100dvh; height: 100dvh; border-radius: 0; border: 0; width: 100%; }
    .ss-head .sub { display: none; }
    .ss-scopes { grid-template-columns: 1fr; }
    .ss-body textarea { font-size: 16px; }
    .ss-tiles { grid-template-columns: repeat(2, 1fr); }
    .ss-t { font-size: 11px; } .ss-t th, .ss-t td { padding: 3px 4px; }
  }`;
  async function open(h) {
    host = h || host || pageHost();
    if (!store) try { store = defaultStore(); } catch (e) { store = lsStore(); }
    if (!hstore) try { hstore = defaultStore(HISTORY_KEY); } catch (e) { hstore = lsStore("gto-session-" + HISTORY_KEY); }
    if (!document.getElementById("ssStyle")) { const s = document.createElement("style"); s.id = "ssStyle"; s.textContent = CSS; document.head.appendChild(s); }
    if (!box) {
      box = document.createElement("div"); box.id = "sessionBox"; box.className = "ss"; box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true"); box.setAttribute("aria-label", "Analyze session");
      box.innerHTML = `<div class="ss-card"><div class="ss-head"><b>Analyze session</b><span class="sub">every decision of a session's hands, solved and graded</span><button class="ghost ss-x" data-close aria-label="Close">✕</button></div><div class="ss-body"></div></div>`;
      document.body.appendChild(box);
      box.addEventListener("click", (e) => {
        if (e.target === box) return close();
        const t = e.target.closest("[data-close],[data-scope],[data-go],[data-resume],[data-pause],[data-stop],[data-widen],[data-new],[data-csv],[data-json],[data-k],[data-forget],[data-clearhist]");
        if (!t || t.disabled) return;
        if (t.dataset.close != null) close(); else act(t.dataset);
      });
      box.addEventListener("change", (e) => { const f = e.target.dataset && e.target.dataset.file != null && e.target.files[0]; if (f) f.text().then(t => readText(t, f.name)); });
      box.addEventListener("input", (e) => { if (e.target.dataset && e.target.dataset.paste != null) { clearTimeout(box._t); const v = e.target.value; box._t = setTimeout(() => v.trim() && readText(v, ""), 400); } });
      box.addEventListener("pointerdown", () => { box._down = true; });
      window.addEventListener("pointerup", () => { box._down = false; }); window.addEventListener("pointercancel", () => { box._down = false; });
      box.addEventListener("dragover", (e) => { e.preventDefault(); const d = box.querySelector("[data-drop]"); if (d) d.classList.add("over"); });
      box.addEventListener("dragleave", (e) => { if (e.target.closest && e.target.closest("[data-drop]")) e.target.closest("[data-drop]").classList.remove("over"); });
      box.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer && e.dataTransfer.files[0]; if (f && !sess) f.text().then(t => readText(t, f.name)); });
      // the page's own shortcuts stay off while the panel is open
      window.addEventListener("keydown", (e) => { if (box.hidden) return; e.stopPropagation(); if (e.key === "Escape") close(); }, true);
    }
    box.hidden = false; draw();
    if (!loaded) {
      try { await host.ready(); } catch (e) {}
      if (!sess) { try { const s = await store.get(); if (s && s.v === 1 && s.items) { sess = s; if (s.status === "running" || s.status === "ready") sess.status = "paused"; } } catch (e) {} }
      try { history = remember(await hstore.get(), null); } catch (e) { history = []; }
      loaded = true; draw();
    }
  }

  const api = { handId, decisions, prepare, buildItems, inScope, spotState, preflopRaises, preflopSituation, classify, mixOf, newSession, widen,
                processItem, runner, estimate, eta, report, tally, freqs, toCSV, toJSON, rowsOf, lsStore, idbStore, pageHost, open, close, SCOPES,
                summarize, remember, mergeRows, trends, leaks, historyHtml, get history() { return history; },
                get session() { return sess; }, get runner_() { return run; } };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Session = api;
})();
