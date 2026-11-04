// Session analyzer, loaded by index.html (after hh.js) the first time "Analyze session…" is opened from the ⋯ menu:
// files of many hand histories (any site hh.js reads; several files or a whole folder at once, a hand in two of them
// counted once) are split into hands (splitHands), each imported as the page imports one (importHand: non-Hold'em
// hands, hands without your cards, duplicates and unreadable ones are counted and skipped), and every one of your
// decisions in the chosen scope is asked of the page's solver and graded as a whole-hand review grades it
// (gradeWith), one after another in a queue that can be paused, resumed (also after a reload: the session is kept in
// IndexedDB, and one left running picks up where it stopped, in the background) or stopped; it steps aside while the
// page asks the solver itself. Quick triage answers each decision from what is in hand at once (Advisor.triage: the
// charts, the flop library, reads off stored solves, else a Quick solve), each with how far to trust it; "Go deeper"
// then asks the flagged ones again at Fast, the biggest pots first. The report: EV lost (bb, bb/100) by street,
// position, postflop spot and preflop situation, your frequencies against the solver's on the same decisions, the
// session's leaks with their hands, the decisions filtered by street, position, pot
// type, EV lost and tags (a click opens that hand at that decision in the page), and the decisions as CSV or JSON.
// Each session's summary is also kept (its own IndexedDB entry, "history") and summed across sessions: the leaks that
// show over many sessions, and each session's EV lost over time.
// Everything but open() and the page host is pure and takes its host as an argument (the Node tests' stub solver).
(function () {
  "use strict";
  const STREETS = ["preflop", "flop", "turn", "river"], NEED = { preflop: 0, flop: 3, turn: 4, river: 5 };
  const SCOPES = { preflop: "Preflop only", library: "Flop-library spots (instant)", triage: "Quick triage, every street", all: "All streets at Fast" };
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
  // The hands of a file, or of several ([{name, text}], read in order: a hand in two of them, the same site, number
  // and cards, is counted once): {hands: [{n, id, site, hero, hole, state, warn, f}], skipped: {nonHoldem, noHero,
  // noDecisions, duplicate, failed}, errors: [{n, id, msg, file}], total, files: [{name, total, hands}] (several)}.
  // n is the hand's place in its file, f its file's index (several). imp is the page's importHand. Async, yielding
  // every chunk hands so a big file doesn't freeze the page (progress(k, total) after each chunk).
  async function prepare(input, imp, progress, chunk = 100, pause = () => new Promise(r => setTimeout(r, 0))) {
    const files = typeof input === "string" ? [{ name: "", text: input }] : Array.from(input || []), multi = files.length > 1;
    const parts = [], hands = [], errors = [], seen = new Set(), per = files.map(f => ({ name: f.name || "", total: 0, hands: 0 })), tours = new Map();
    files.forEach((f, fi) => HH().splitHands(String(f.text || "")).forEach((t, k) => parts.push([fi, k, t])));
    const skipped = { nonHoldem: 0, noHero: 0, noDecisions: 0, duplicate: 0, failed: 0 };
    for (let j = 0; j < parts.length; j++) {
      if (j && j % chunk === 0) { if (progress) progress(j, parts.length); await pause(); }
      const [fi, k, t] = parts[j], id = handId(t), at = multi ? { f: fi } : {};
      per[fi].total++;
      let r;
      try { r = imp(t); } catch (e) {
        const msg = e && e.message || String(e), kind = skipKind(msg);
        skipped[kind]++; if (kind === "failed") errors.push({ n: k + 1, id, msg, ...(multi ? { file: per[fi].name } : {}) });
        continue;
      }
      const key = `${r.hand.site}|${id}|${r.hand.hole.join("")}`;
      if (id && seen.has(key)) { skipped.duplicate++; continue; }
      seen.add(key);
      if (!decisions(r.state).length) { skipped.noDecisions++; continue; }
      per[fi].hands++;
      // a tournament's hands (hh.js: its id, name, buy-in, level), kept together for its payouts (entered once)
      const tr = r.hand.tournament, tk = tr ? `${r.hand.site}|${tr.id || "?"}` : null;
      if (tk) {
        if (!tours.has(tk)) tours.set(tk, { key: tk, site: r.hand.site, id: tr.id, ...(tr.name ? { name: tr.name } : {}), ...(tr.buyin ? { buyin: tr.buyin } : {}), hands: 0 });
        const o = tours.get(tk); o.hands++;
        if (tr.level != null) o.levels = o.levels ? [Math.min(o.levels[0], tr.level), Math.max(o.levels[1], tr.level)] : [tr.level, tr.level];
      }
      hands.push({ n: k + 1, ...at, id, site: r.hand.site, hero: r.hand.hero, hole: r.hand.hole.join(""), state: r.state,
                   ...(tk ? { t: tk } : {}), ...(r.warnings.length ? { warn: r.warnings.join(" ") } : {}) });
    }
    return { hands, skipped, errors, total: parts.length, ...(multi ? { files: per } : {}), ...(tours.size ? { tourneys: [...tours.values()] } : {}) };
  }
  // the hand-history files among those picked or dropped (a folder's: its .txt, .xml, .log and .hh files)
  const HH_FILE = /\.(txt|xml|log|hh)$/i;
  const hhFiles = (list) => Array.from(list || []).filter(f => f && (HH_FILE.test(f.name || "") || /^text\//.test(f.type || "")));
  // a session's name from its files: the one file's, else the folder's (when all come from one) or "N files"
  function filesName(files) {
    if (!files.length) return "";
    if (files.length === 1) return files[0].name || "";
    const dirs = new Set(files.map(f => String(f.path || f.webkitRelativePath || "").split("/").slice(0, -1).join("/")));
    const d = dirs.size === 1 ? [...dirs][0] : "";
    return d ? `${d.split("/").pop()} (${files.length} files)` : `${files.length} files`;
  }
  // the decisions a scope asks about: preflop only; preflop and flops the library may hold (asked "library_only",
  // so anything else is skipped, not solved); everything (Quick triage, asked "triage"; or at Fast)
  const inScope = (scope, s) => scope === "all" || scope === "triage" || s === "preflop" || scope === "library" && s === "flop";
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
  const gradeClass = (g) => g.f >= MIX_OK ? "ok" : g.f >= MIX_OFF || g.r || g.l === 0 ? "meh" : "bad";

  // ---- a tournament's payouts, entered once for all its hands (tset: {key: {payouts, fieldN, fieldAvg}}): the prizes
  // still to pay, first place first, and the players left at other tables with their average stack (bb), as the
  // page's Payouts row takes them; each of its hands is then solved in prize equity (ICM) from two prizes up
  const payList = (v) => (Array.isArray(v) ? v : String(v ?? "").split(/[\s,;\/]+/)).map(parseFloat).filter(x => x > 0 && isFinite(x)).slice(0, 200);
  function tourSet(t) {
    const pay = payList(t && t.payouts);
    if (!pay.length) return null;
    return { payouts: pay, ...(+t.fieldN > 0 ? { fieldN: Math.min(10000, Math.floor(+t.fieldN)) } : {}), ...(+t.fieldAvg > 0 ? { fieldAvg: +t.fieldAvg } : {}) };
  }
  function withPayouts(hands, tset) {
    if (!tset) return hands;
    return hands.map(h => { const t = h.t && tourSet(tset[h.t]); return t ? { ...h, state: { ...h.state, ...t } } : h; });
  }
  // ---- the queue
  function newSession(prep, scope, name) {
    const hands = withPayouts(prep.hands, prep.tset);
    return { v: 1, name: name || "", created: Date.now(), scope, hands, skipped: prep.skipped, errors: prep.errors,
             total: prep.total, ...(prep.files ? { files: prep.files.map(f => f.name) } : {}),
             ...(prep.tourneys ? { tourneys: prep.tourneys.map(t => ({ ...t, ...(tourSet((prep.tset || {})[t.key]) || {}) })) } : {}),
             items: buildItems(hands, scope), next: 0, results: [], timing: {}, status: "ready" };
  }
  // ---- Quick triage, then deeper: the triage answers worth asking again at Fast. Not the charts' or the library's
  // own spot (Fast answers those the same); the rest where the triage grade isn't in the mix, the EV it lost is
  // DEEP_LOSS of the pot or more, the answer is low confidence (a read off the nearest stored flops or an earlier
  // street's solve, a rougher Quick solve, the betting rules: measured, a third of those reads Fast grades
  // otherwise), or it couldn't grade (an error, an action the answer doesn't list). every: each triage answer Fast
  // could answer otherwise.
  const DEEP_LOSS = 0.02;
  const sameAtFast = (r) => (r.src === "chart" || r.src === "library") && r.tc === "high";
  const deepened = (r) => !!r.t || r.deep;
  function flagged(r, every) {
    if (deepened(r) || r.status === "skipped" || !r.tc && r.status !== "error") return false;   // (not a triage answer)
    if (sameAtFast(r)) return false;
    if (every || r.status === "error" || r.status === "ungraded" && r.src !== "rules") return true;
    if (!r.g) return false;
    return r.tc === "low" || gradeClass(r.g) !== "ok" || r.g.l != null && r.pot > 0 && r.g.l >= DEEP_LOSS * r.pot;
  }
  // the session with the flagged decisions (every: all that Fast could answer otherwise) queued again at Fast, the
  // hands with the biggest flagged pots first, each hand's in the order played (a turn's solve at Fast answers its
  // river in a fraction of a second, where the river alone took 2.6 s in the browser); each answer replaces its
  // triage one, which it keeps (t) for the agreement
  function deepen(sess, every) {
    const queued = new Set(sess.items.slice(sess.next).filter(it => it.deep).map(it => it.k));
    const ks = sess.results.map((r, k) => k).filter(k => !queued.has(k) && flagged(sess.results[k], every)), top = {};
    for (const k of ks) { const r = sess.results[k]; top[r.h] = Math.max(top[r.h] || 0, r.pot || 0); }
    ks.sort((a, b) => { const x = sess.results[a], y = sess.results[b]; return top[y.h] - top[x.h] || x.h - y.h || a - b; });
    const items = ks.map(k => { const r = sess.results[k]; return { h: r.h, s: r.s, i: r.i, deep: 1, k }; });
    const all = [...sess.items, ...items];
    return { ...sess, items: all, status: sess.next < all.length ? "paused" : sess.status };
  }
  const flaggedRows = (sess, every) => {
    const queued = new Set(sess.items.slice(sess.next).filter(it => it.deep).map(it => it.k));
    return sess.results.filter((r, k) => !queued.has(k) && flagged(r, every));
  };
  const flaggedCount = (sess, every) => flaggedRows(sess, every).length;
  // the triage answer a deeper one replaced, in brief
  const triageOf = (r) => ({ src: r.src, tc: r.tc, status: r.status, ...(r.g ? { c: gradeClass(r.g), f: r.g.f, l: r.g.l, b: r.g.b } : {}) });
  // How often the deeper answers graded your action as the triage did: {n, same (the same grade class), worse (Fast's
  // grade worse), better}
  function agreement(results) {
    const rs = results.filter(r => r.t && r.g && r.t.c), O = { ok: 0, meh: 1, bad: 2 };
    const out = { n: rs.length, same: 0, worse: 0, better: 0 };
    for (const r of rs) { const d = O[gradeClass(r.g)] - O[r.t.c]; out[d === 0 ? "same" : d > 0 ? "worse" : "better"]++; }
    return out;
  }
  // the same session in a wider scope: the decisions it didn't ask yet (and the flops "library_only" skipped)
  function widen(sess, scope) {
    const done = new Set(sess.results.filter(r => r.status !== "skipped").map(r => `${r.h}|${r.s}|${r.i}`));
    const keep = sess.results.filter(r => r.status !== "skipped");
    const todo = buildItems(sess.hands, scope).filter(it => !done.has(`${it.h}|${it.s}|${it.i}`));
    return { ...sess, scope, results: keep, items: [...keep.map(r => ({ h: r.h, s: r.s, i: r.i })), ...todo], next: keep.length, status: todo.length ? "paused" : "done" };
  }
  // one decision: the page's request for the spot (at Fast; Quick triage's "triage" request; a deeper one at Fast),
  // the answer, the grade
  async function processItem(sess, it, host) {
    const hand = sess.hands[it.h], played = hand.state.actions[it.s][it.i];
    const base = { h: it.h, s: it.s, i: it.i };
    const sp = host.spot(spotState(hand.state, it.s, it.i));
    if (!sp) return { ...base, street: it.s, pos: hand.state.hero, act: actLabel(played), cls: actClass(played), status: "error", err: "the hand doesn't reach this decision" };
    Object.assign(base, classify(hand.state, it.s, it.i, sp), sp.toCall > 1e-9 ? { bet: r2(sp.toCall) } : {});
    const rec = played.action === "all-in" && sp.allinAs ? { action: sp.allinAs } : played;
    const req = { ...sp.payload, ...(it.deep ? {} : sess.scope === "library" && it.s === "flop" ? { library_only: true } : sess.scope === "triage" ? { triage: true } : {}) };
    let res;
    try { res = await host.advise(req); } catch (e) { res = { error: String(e && e.message || e) }; }
    if (!res || res.cancelled) return { retry: true, ...(res && res.yielded ? { yielded: true } : {}) };
    const tri = sess.scope === "triage" && !it.deep ? { tc: res.triage ? res.triage.confidence : "low" } : {};
    if (res.skipped) return { ...base, status: "skipped", why: res.skipped };
    if (res.error) return { ...base, ...tri, status: "error", err: String(res.error) };
    // a jam deeper than the answer's tree: asked again with all-in at every decision of the street (index.html jamOffTree)
    if (host.jamOffTree && host.jamOffTree(res, rec) && !host.grade(res, rec)) {
      let r2; try { r2 = await host.advise({ ...req, jam: true }); } catch (e) { r2 = null; }
      if (r2 && !r2.error && !r2.cancelled && r2.actions) res = r2;
    }
    const g = host.grade(res, rec), t = res.triage;
    return { ...base, status: g ? "ok" : "ungraded", ...(g ? { g } : {}), mix: mixOf(res),
             src: t ? t.source : res.precomputed ? "library" : it.s === "preflop" ? "chart" : res.multiway && !res.solved ? "rules" : "solve",
             ...tri, ...(t && t.label ? { tl: t.label } : {}) };
  }
  // A runner over a session: run() works through its items until done, paused or stopped. While the page asks the
  // solver itself (host.busy: its advice, Practice, Study) the queue waits (host.idle between looks), and a request
  // of ours it stopped for that (yielded) is asked again once it's done; one the page cancelled otherwise is asked
  // again up to 3 times. A deeper item's answer replaces the triage one (results[k]), keeping it (t). save(sess) is
  // called every saveEvery ms and when it stops; update() after each decision.
  function runner(sess, host, { update = () => {}, save = async () => {}, now = () => Date.now(), saveEvery = 2000 } = {}) {
    let stop = null, running = false, lastSave = 0, current = null;
    const busy = () => !!(host.busy && host.busy()), idle = host.idle || (() => new Promise(r => setTimeout(r, 300)));
    const R = {
      get running() { return running; }, get stopping() { return stop; }, get current() { return current; }, sess,
      get waiting() { return running && busy(); },
      async run() {
        if (running || sess.next >= sess.items.length) return;
        running = true; stop = null; sess.status = "running"; update();
        try {
          if (host.ready) await host.ready();
          while (sess.next < sess.items.length && !stop) {
            if (busy()) { update(); while (busy() && !stop) await idle(); if (stop) break; }
            const it = sess.items[sess.next], t0 = now();
            current = it; update();
            const r = await processItem(sess, it, host);
            if (stop === "cancel") break;   // stopped: the answer in flight is dropped
            if (r.retry) {
              if (r.yielded || busy()) continue;
              it.tries = (it.tries || 0) + 1; if (it.tries < 3) continue;
              Object.assign(r, { h: it.h, s: it.s, i: it.i, status: "error", err: "cancelled" }); delete r.retry; delete r.yielded;
            }
            r.sec = Math.round((now() - t0)) / 1000;
            const tk = timingKey(it), tm = sess.timing[tk] || (sess.timing[tk] = [0, 0]);
            tm[0] += r.sec; tm[1]++;
            if (it.deep && sess.results[it.k]) sess.results[it.k] = { ...r, t: triageOf(sess.results[it.k]) };
            else sess.results.push(r);
            sess.next++;
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
  // seconds a decision takes at Fast before any is measured: a share of the street's budget plus each earlier
  // postflop street's narrowing budget (FAST_SHARE: a 1,000-hand session's averages in the browser, where the solves
  // stop once they reach the target: flop 1.6 s, turn 7.8, river 0.1 after its turn, 2.6 alone; natively 1.9, 4.6,
  // 0.3); a library-only flop and preflop's chart are instant. Quick triage: what a 1,000-hand
  // session averaged natively and in the browser (most of it read at once, the rest a Quick solve: TRIAGE_BUDGETS).
  // threads: the browser's solver threads (pageHost.threads); at 8 the same session's every 20th hand took
  // AT_8 of the time at Fast (flop 0.28 s, turn 2.2, river 0.05) and TRIAGE_8 at triage (a Quick solve keeps its
  // budget), in between geometrically
  const TRIAGE_S = { preflop: 0.02, flop: 0.6, turn: 1.2, river: 0.2 }, TRIAGE_8 = { preflop: 1, flop: 0.38, turn: 0.92, river: 0.33 };
  const AT_8 = { flop: 0.17, turn: 0.28, river: 0.5 };
  function estimate(s, scope, budgets, threads = 1) {
    const [b, p] = budgets || [2.5, 2.5], w = Math.min(1, Math.max(0, ((threads || 1) - 1) / 7));
    if (s === "preflop") return scope === "triage" ? TRIAGE_S.preflop : 0.2;
    if (scope === "triage") return TRIAGE_S[s] * TRIAGE_8[s] ** w;
    if (s === "flop" && scope === "library") return 0.4;
    return (FAST_SHARE[s] * (b + p * (s === "flop" ? 0 : s === "turn" ? 1 : 2)) + 0.3) * AT_8[s] ** w;
  }
  const FAST_SHARE = { flop: 0.3, turn: 0.7, river: 0.15 };
  // (a deeper item's time is kept apart from the triage's on the same street)
  const timingKey = (it) => it.deep ? "fast " + it.s : it.s;
  // seconds left: each street's measured average once it has two answers, else the estimate
  function eta(sess, budgets, threads = 1) {
    let t = 0;
    for (let k = sess.next; k < sess.items.length; k++) {
      const it = sess.items[k], m = sess.timing[timingKey(it)];
      t += m && m[1] >= 2 ? m[0] / m[1] : estimate(it.s, it.deep ? "all" : sess.scope, budgets, threads);
    }
    return t;
  }
  const scopeEta = (hands, scope, budgets, threads = 1) => buildItems(hands, scope).reduce((t, it) => t + estimate(it.s, scope, budgets, threads), 0);
  const nth = () => host && host.threads ? host.threads() : 1;
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
  // The report, over the decisions a filter keeps (filterRows; the session's leaks, the triage's agreement and what
  // it flagged are over all of them)
  function report(sess, top = 20, F = null) {
    const res = sess.results, all0 = res.filter(r => r.status === "ok" || r.status === "ungraded"), done = filterRows(all0, F);
    const all = tally(done), handsDone = new Set(res.map(r => r.h)).size;
    const mistakes = done.filter(r => r.g && r.g.l > 0.005).sort((a, b) => b.g.l - a.g.l).slice(0, top);
    const offChart = done.filter(r => r.g && r.g.l == null && r.g.f < MIX_OFF).sort((a, b) => b.g.bf - a.g.bf).slice(0, top);
    const seats = [...new Set(sess.hands.map(h => h.state.players))].sort((a, b) => b - a), POS = (globalThis.HH_SEATS || HH().HH_SEATS);
    const byPf = groupBy(done.filter(r => r.pf), r => r.pf, PF_ORDER), bySpot = groupBy(done.filter(r => r.spot), r => r.spot), fq = freqs(done);
    const unf = F ? { freqs: freqs(all0), byPf: groupBy(all0.filter(r => r.pf), r => r.pf, PF_ORDER), bySpot: groupBy(all0.filter(r => r.spot), r => r.spot) } : { freqs: fq, byPf, bySpot };
    const conf = {}, srcs = {};
    for (const r of res) { if (r.tc) conf[r.tc] = (conf[r.tc] || 0) + 1; if (r.src && !r.t) srcs[r.src] = (srcs[r.src] || 0) + 1; }
    return {
      hands: sess.hands.length, handsDone, decisions: sess.items.length, done: res.length, ...all, filtered: !!F, kept: done.length,
      bb100: handsDone ? r2(all.lost / handsDone * 100) : null,
      skipped: res.filter(r => r.status === "skipped").length, errors: res.filter(r => r.status === "error").length,
      ungraded: res.filter(r => r.status === "ungraded").length, rules: res.filter(r => r.status === "ungraded" && r.src === "rules").length,
      byStreet: groupBy(done, r => r.street, STREETS),
      byPos: groupBy(done, r => r.pos, [...new Set(seats.flatMap(n => POS[n] || []))]),
      bySpot, byPf, freqs: fq, mistakes, offChart,
      leaks: leaks(unf).map(l => ({ ...l, rows: all0.filter(leakTest(l)).length })),
      conf, srcs, agree: agreement(res), flagged: flaggedCount(sess), reaskable: flaggedCount(sess, true),
      deepDone: res.filter(r => r.t).length, deepLeft: sess.items.slice(sess.next).filter(it => it.deep).length,
    };
  }
  // ---- the decisions by street, position, pot type, EV lost and tags; a leak's decisions
  const POT_TYPES = ["SRP", "3bet pot", "4bet+ pot", "Limped", "Multiway"];
  const lossPct = (r) => r.g && r.g.l != null && r.pot > 0 ? r.g.l / r.pot * 100 : null;
  // (under 2% of the pot an inaccuracy, under 8% a mistake, more a blunder)
  const LOSS = {
    lost: ["Any EV lost", (r) => r.g.l > 0.005],
    p2: ["2%+ of the pot (mistakes)", (r) => lossPct(r) >= 2],
    p8: ["8%+ of the pot (blunders)", (r) => lossPct(r) >= 8],
    bb1: ["1bb or more", (r) => r.g.l >= 1],
    bb5: ["5bb or more", (r) => r.g.l >= 5],
  };
  const TAGS = ["IP", "OOP", "facing a bet", "PFR", "caller", "multiway", "all-in", "off the mix", "flagged", "checked at Fast", "low confidence"];
  function tagsOf(r) {
    const t = [];
    if (r.s !== "preflop") {
      t.push(r.ip ? "IP" : "OOP");
      if (r.facing) t.push("facing a bet");
      if (r.role === "PFR" || r.role === "caller") t.push(r.role);
      if (r.multi) t.push("multiway");
    }
    if (/^all-in/.test(r.act || "")) t.push("all-in");
    if (r.g && gradeClass(r.g) === "bad") t.push("off the mix");
    if (flagged(r)) t.push("flagged");
    if (r.t) t.push("checked at Fast");
    else if (r.tc === "low") t.push("low confidence");
    return t;
  }
  // a leak's decisions: those of the frequency, the preflop situation or the postflop spot
  function leakTest(l) {
    if (l.kind === "freq") { const f = FREQS.find(x => x[0] === l.key); return f ? (r) => !!r.mix && f[2](r) : () => false; }
    return l.kind === "pf" ? (r) => r.pf === l.key : (r) => r.spot === l.key;
  }
  // F: {street, pos, pot (POT_TYPES: after the flop), loss (LOSS), tags: [all of them], leak: {kind, key}}
  function filterRows(rows, F) {
    if (!F) return rows;
    const L = F.leak ? leakTest(F.leak) : null, tags = F.tags || [];
    return rows.filter(r => (!F.street || r.s === F.street) && (!F.pos || r.pos === F.pos)
      && (!F.pot || r.s !== "preflop" && (F.pot === "Multiway" ? r.multi : !r.multi && r.potType === F.pot))
      && (!F.loss || !!(r.g && r.g.l != null && LOSS[F.loss] && LOSS[F.loss][1](r)))
      && (!tags.length || tags.every(t => tagsOf(r).includes(t))) && (!L || L(r)));
  }
  const filterOn = (F) => !!(F && (F.street || F.pos || F.pot || F.loss || (F.tags && F.tags.length) || F.leak));
  // ---- export
  const boardAt = (st, s) => st.cards.slice(2, 2 + NEED[s]).filter(Boolean).join("");
  const COLS = ["hand", "hand_id", "site", "position", "hole", "street", "board", "situation", "action", "solver_freq", "best_action", "best_freq",
                "ev_bb", "best_ev_bb", "ev_lost_bb", "grade", "source", "status", "pot_bb", "facing_bet", "in_position", "seconds",
                "confidence", "triage_source", "triage_grade", "triage_ev_lost_bb", "tags", "file", "tournament"];
  function rowsOf(sess) {
    return sess.results.map(r => {
      const h = sess.hands[r.h], g = r.g || {}, t = r.t || {};
      return { hand: h.n, hand_id: h.id, site: h.site, position: h.state.hero, hole: h.hole, street: r.s, board: boardAt(h.state, r.s),
               situation: r.pf || r.spot || "", action: r.act || "", solver_freq: g.f, best_action: g.b, best_freq: g.bf, ev_bb: g.e, best_ev_bb: g.o,
               ev_lost_bb: g.l, grade: r.g ? gradeClass(r.g) : "", source: r.src || "", status: r.status, pot_bb: r.pot, facing_bet: r.facing,
               in_position: r.ip, seconds: r.sec, confidence: r.t ? "deep" : r.tc || "", triage_source: t.src || "", triage_grade: t.c || "",
               triage_ev_lost_bb: t.l, tags: tagsOf(r).join("; "), file: sess.files && h.f != null ? sess.files[h.f] : "", tournament: h.t ? h.t.split("|").slice(1).join("|") : "" };
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
    /* global importHand, modeReady, BUDGETS, MODE, S, cleanState, setPositions, simulate, payload, callAdvise, heuristic, gradeWith, jamOffTree, gradedMove, loadHandState, rewindTo, solving, pending, stopRequests */
    const mine = new Map();   // our requests in flight: id -> stopped for the page
    // the page asking the solver itself: its advice solving, or a request of its own waiting in the browser's worker
    // (Practice, Study, a Breakdown; not ours)
    const busy = () => {
      try {
        if (typeof solving !== "undefined" && solving) return true;
        return MODE === "wasm" && typeof pending === "object" && Object.keys(pending).some(k => !mine.has(+k));
      } catch (e) { return false; }
    };
    const step = () => { for (const [id, stopped] of mine) if (!stopped && busy() && stopRequests(id)) mine.set(id, true); };
    return {
      busy,
      importHand: (t) => importHand(t),
      ready: () => modeReady,
      budgets: () => MODE ? BUDGETS[MODE].fast : null,
      // the browser's solver threads (worker.js: up to 7 helpers where the page is cross-origin isolated); the local
      // server's are its own (taken as one)
      threads: () => MODE === "wasm" && globalThis.crossOriginIsolated && typeof navigator === "object" ? Math.max(1, Math.min(8, navigator.hardwareConcurrency || 1)) : 1,
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
      // (ours is stopped once the page asks, looked at every 150 ms: the browser's worker's shared stop flag, ~0.1 s;
      // asked again after)
      step,
      advise(p) {
        const q = callAdvise(p), id = q && q.reqId;
        if (id == null) return q;
        mine.set(id, false);
        const t = setInterval(step, 150);
        return q.then(r => (r && r.cancelled && mine.get(id) ? { cancelled: true, yielded: true } : r))
          .finally(() => { clearInterval(t); mine.delete(id); });
      },
      grade: (res, rec) => gradeWith(res, rec),
      jamOffTree: (res, rec) => jamOffTree(res, rec),
      // the whole hand, a hand played, at the decision's advice (Back to the hand brings the rest)
      open: async (st, s, i) => { await loadHandState({ ...clone(st), finished: true }); rewindTo(s, i); },
    };
  }

  // ---- the panel
  let box = null, host = null, store = null, hstore = null, history = [], clearArm = false, sess = null, run = null, draft = null, reading = null, scope = "triage", drawT = null, loaded = false;
  let F = {}, showN = 25, loading = null, wasRunning = false, beat = null;
  // (a redraw between a press and its release would lose the click: not while a pointer is down on the panel)
  function schedule() { menuMark(); if (drawT) return; drawT = setTimeout(() => { drawT = null; if (box && box._down) schedule(); else draw(); }, 400); }
  // A session running is marked in localStorage (RUN_KEY: this tab's id and the time, every few seconds), so a reload
  // picks it up where it stopped (index.html loads this script then and calls resume()); one another tab is running
  // (marked under STALE_MS ago) isn't started twice
  const RUN_KEY = "gto-session-run", STALE_MS = 60000;
  const tabId = (() => { try { let t = sessionStorage.getItem(RUN_KEY); if (!t) { t = Math.random().toString(36).slice(2); sessionStorage.setItem(RUN_KEY, t); } return t; } catch (e) { return "tab"; } })();
  function runMark(on) {
    try { if (on) localStorage.setItem(RUN_KEY, JSON.stringify({ tab: tabId, at: Date.now() })); else localStorage.removeItem(RUN_KEY); } catch (e) {}
  }
  function elsewhere() {   // another tab running the session now
    try { const m = JSON.parse(localStorage.getItem(RUN_KEY)); return !!(m && m.tab !== tabId && Date.now() - m.at < STALE_MS); } catch (e) { return false; }
  }
  // the ⋯ menu's item shows how far a session running in the background is
  function menuMark() {
    const m = typeof document === "object" && document.getElementById ? document.getElementById("analyzeSession") : null;
    if (!m) return;
    if (m.dataset.label == null) m.dataset.label = m.textContent;
    const on = run && run.running && sess && sess.items.length;
    m.textContent = on ? `${m.dataset.label} ${Math.floor(sess.next / sess.items.length * 100)}%` : m.dataset.label;
  }
  // (a session that isn't running goes into the history too: its summary as it stands). Saved every 2 s while it
  // runs, less often where the copy takes longer (a big session: at most 2% of the time)
  let saveCost = 0, savedAt = 0;
  const save = (x) => {
    if (x.status !== "running") keep(x);
    else if (Date.now() - savedAt < saveCost * 50) return Promise.resolve();
    const t0 = Date.now(), c = clone(x);
    saveCost = Date.now() - t0; savedAt = Date.now();
    return store ? store.set(c).catch(() => {}) : Promise.resolve();
  };
  function keep(x) {
    const sum = summarize(x);
    if (!sum) return;
    history = remember(history, sum);
    if (hstore) hstore.set(clone(history)).catch(() => {});
  }
  function start(s) {
    sess = s; run = runner(sess, host, { update: schedule, save });
    clearInterval(beat); runMark(true);
    beat = setInterval(() => { if (run && run.running) runMark(true); else { clearInterval(beat); runMark(false); menuMark(); } }, 5000);
    return run.run().finally(() => { if (!(run && run.running)) { clearInterval(beat); runMark(false); menuMark(); } });
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
  // how far to trust an answer: the triage's confidence, or checked at Fast (what the triage said in its title)
  const CONF = { high: "ok", medium: "meh", low: "bad" };
  const confChip = (r) => r.t ? `<span class="ss-tc ok" title="Asked again at Fast. Quick triage: ${esc(r.t.c ? `${r.t.c === "ok" ? "in the mix" : r.t.c === "meh" ? "a rare play" : "off the mix"}${r.t.l != null ? `, −${r.t.l.toFixed(2)}bb` : ""}` : r.t.status)} (${esc(r.t.src || "")}, ${esc(r.t.tc || "")} confidence)">Fast</span>`
    : r.tc ? `<span class="ss-tc ${CONF[r.tc] || ""}" title="${esc(`Quick triage, ${r.tc} confidence: ${r.tl || r.src || ""}`)}">${esc(r.tc)}</span>` : "";
  function decisionRow(r, k) {
    const h = sess.hands[r.h], g = r.g || {}, file = sess.files && h.f != null ? ` · ${sess.files[h.f]}` : "";
    return `<tr class="ss-go" data-k="${k}" title="Open this hand at this decision (hand ${h.n}${h.id ? ` #${esc(h.id)}` : ""}${esc(file)})"><td>${cardsH(h.hole)}</td><td>${esc(h.state.hero)}</td><td>${esc(r.s)}${r.s !== "preflop" ? ` ${cardsH(boardAt(h.state, r.s))}` : ""}<div class="ss-sub">${esc(r.pf || r.spot || "")}</div></td>`
      + `<td><b>${esc(r.act)}</b> ${r.g ? `<small>${pct(g.f)}${g.v ? ` as the chart's ${esc(g.m)}` : ""}</small><div class="ss-sub">solver: ${esc(g.b)} ${pct(g.bf)} ${confChip(r)}</div>` : `<div class="ss-sub">${esc(r.err || "not graded")} ${confChip(r)}</div>`}</td>`
      + `<td class="n">${!r.g ? "—" : g.l != null ? bb(g.l) : g.f < MIX_OFF ? `<span class="g bad">off chart</span>` : "<small>no EVs</small>"}</td></tr>`;
  }
  const listHtml = (title, rows, note, more) => rows.length ? `<h3>${title}</h3>${note ? `<div class="ss-note">${note}</div>` : ""}<div class="ss-tw"><table class="ss-t ss-m"><thead><tr><th>Hand</th><th>Seat</th><th>Street</th><th>You / solver</th><th class="n">EV lost</th></tr></thead><tbody>`
    + rows.map(r => decisionRow(r, sess.results.indexOf(r))).join("") + `</tbody></table></div>${more ? `<div class="ss-row"><button class="link" data-more>Show ${Math.min(25, more)} more of ${more}</button></div>` : ""}` : "";
  // the decisions the filter keeps, the most EV lost first, then those off the mix
  const byLoss = (a, b) => (b.g ? b.g.l == null ? 0 : b.g.l : -1) - (a.g ? a.g.l == null ? 0 : a.g.l : -1) || (a.g ? a.g.f : 1) - (b.g ? b.g.f : 1);
  function filterHtml(rp) {
    const sel = (k, label, opts) => `<select data-filter="${k}" aria-label="${label}"><option value="">${label}</option>${opts.map(([v, t]) => `<option value="${esc(v)}"${F[k] === v ? " selected" : ""}>${esc(t)}</option>`).join("")}</select>`;
    const pos = [...new Set(sess.hands.map(h => h.state.hero))], order = rp.byPos.map(r => r.key);
    pos.sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
    const tags = TAGS.filter(t => sess.results.some(r => r.status !== "skipped" && tagsOf(r).includes(t)) || (F.tags || []).includes(t));
    return `<div class="ss-flt"><div class="ss-row">${sel("street", "Any street", STREETS.map(s => [s, s[0].toUpperCase() + s.slice(1)]))}${sel("pos", "Any position", pos.map(p => [p, p]))}`
      + `${sel("pot", "Any pot", POT_TYPES.map(p => [p, p]))}${sel("loss", "Any EV lost", Object.entries(LOSS).map(([k, [t]]) => [k, t]))}`
      + (filterOn(F) ? `<button class="link" data-clearf>Clear the filter</button>` : "") + `</div>`
      + `<div class="ss-tags">${tags.map(t => `<button class="ss-tag${(F.tags || []).includes(t) ? " active" : ""}" data-tag="${esc(t)}" aria-pressed="${(F.tags || []).includes(t)}">${esc(t)}</button>`).join("")}</div>`
      + (F.leak ? `<div class="ss-note">The leak's decisions: <b>${esc(F.leak.key)}</b> <button class="link" data-leakoff>show all</button></div>` : "") + `</div>`;
  }
  function triageHtml(rp) {
    if (sess.scope !== "triage") return "";
    const c = rp.conf, a = rp.agree, n = (c.high || 0) + (c.medium || 0) + (c.low || 0);
    let h = `<div class="ss-note ss-tri"><b>Quick triage</b>: ${n} decisions, ${c.high || 0} high confidence (the charts, the flop library), ${c.medium || 0} medium (reads off stored solves, settled Quick solves), ${c.low || 0} low.`;
    if (rp.flagged) h += ` ${rp.flagged} flagged: not in the mix, ${Math.round(DEEP_LOSS * 100)}%+ of the pot lost, or not graded.`;
    if (rp.deepLeft) h += ` ${rp.deepLeft} queued at Fast.`;
    if (a.n) h += ` Asked again at Fast: ${a.n}; the same grade on ${pct(a.same / a.n)} (Fast's worse on ${a.worse}, better on ${a.better}).`;
    return h + `</div>`;
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
    h += triageHtml(rp);
    if (rp.leaks.length) h += `<h3>This session's leaks</h3><ol class="ss-leaks">${rp.leaks.slice(0, 8).map((l, k) => `<li>${leakText(l)} <button class="link" data-leak="${k}">${l.rows} hands</button></li>`).join("")}</ol>`;
    const fp = filterOn(F) ? report(sess, 0, F) : rp;
    h += `<h2 class="ss-h2">${filterOn(F) ? `${fp.kept} decisions` : "All decisions"}</h2>` + filterHtml(rp);
    h += table("By street", fp.byStreet, "Street") + table("By position", fp.byPos, "Position")
      + table("Preflop situations", fp.byPf, "Situation", "Preflop answers come from the charts: no EVs, graded by how often the solver plays your action.")
      + table("Postflop spots", fp.bySpot, "Spot", "Pot type, your position, and whether you faced a bet.");
    h += freqTable(fp.freqs, "Your frequencies vs GTO");
    if (filterOn(F)) {
      const rows = filterRows(sess.results.filter(r => r.status === "ok" || r.status === "ungraded"), F).sort(byLoss);
      h += listHtml("Decisions", rows.slice(0, showN), "The most EV lost first. Click one to open the hand at that decision.", Math.max(0, rows.length - showN));
      if (!rows.length) h += `<div class="ss-note">No decision passes the filter.</div>`;
    } else {
      h += listHtml("Biggest mistakes", rp.mistakes, "By EV given up. Click one to open the hand at that decision.");
      h += listHtml("Preflop: off-chart plays", rp.offChart, "Actions the charts play under 3% of the time with that hand.");
    }
    return h;
  }
  // the draft's tournaments, each with its payouts to enter once (read on change, kept in draft.tset)
  function tourHtml() {
    const ts = draft && draft.tourneys || [];
    if (!ts.length) return "";
    const v = (k, f) => esc(((draft.tset || {})[k] || {})[f] ?? "");
    return `<div class="ss-tour"><b>${ts.length === 1 ? "A tournament" : `${ts.length} tournaments`}</b><div class="ss-note">The prizes still to pay, first place first ("500 300 200"), price every chip in prize equity (ICM) in each of its hands, with the players left at other tables and their average stack once two or more are paid, as the page's Payouts row does. Empty: chips, as a cash game.</div>`
      + ts.map(t => `<div class="ss-trow"><span>${esc(t.name || `#${t.id || "?"}`)}${t.name && t.id && t.id !== t.name ? ` <small>#${esc(t.id)}</small>` : ""}${t.buyin ? ` · ${esc(t.buyin)}` : ""} · ${t.hands} hand${t.hands === 1 ? "" : "s"}${t.levels ? ` · level ${t.levels[0]}${t.levels[1] !== t.levels[0] ? `–${t.levels[1]}` : ""}` : ""}</span>`
        + `<input type="text" data-tour="${esc(t.key)}" data-tf="payouts" value="${v(t.key, "payouts")}" placeholder="payouts left" title="The prizes still to pay, first place first" aria-label="The prizes still to pay, first place first">`
        + `<input type="number" min="0" step="1" data-tour="${esc(t.key)}" data-tf="fieldN" value="${v(t.key, "fieldN")}" placeholder="others left" title="Players left at other tables" aria-label="Players left at other tables">`
        + `<input type="number" min="0" step="1" data-tour="${esc(t.key)}" data-tf="fieldAvg" value="${v(t.key, "fieldAvg")}" placeholder="avg bb" title="Their average stack, in big blinds" aria-label="Their average stack, in big blinds"></div>`).join("") + `</div>`;
  }
  function inputHtml() {
    const est = (sc) => { if (!draft) return ""; const n = buildItems(draft.hands, sc).length; return `<small>${n} decisions · ~${fmtDur(scopeEta(draft.hands, sc, host.budgets(), nth()))}</small>`; };
    const sk = draft && draft.skipped, skl = sk ? [["nonHoldem", "not Hold'em"], ["noHero", "without your cards"], ["noDecisions", "no decision of yours"], ["duplicate", "duplicates"], ["failed", "unreadable"]].filter(([k]) => sk[k]).map(([k, l]) => `${sk[k]} ${l}`) : [];
    const nf = draft && draft.files ? draft.files.length : 1;
    return `<label class="ss-drop" data-drop><input type="file" data-file multiple accept=".txt,.xml,.log,.hh,text/plain,text/xml"><b>Drop hand-history files or a folder here</b> or <u>choose files</u><small>PokerStars, GGPoker, 888, partypoker, Winamax, WPN/ACR, Ignition/Bovada, iPoker: whole sessions, any number of hands and files (a hand in two files counts once)</small></label>`
      + `<div class="ss-row"><label class="ss-folder"><input type="file" data-file webkitdirectory multiple><u>Choose a folder</u></label><span class="ss-note">its .txt, .xml, .log and .hh files, and its folders'</span></div>`
      + `<textarea data-paste rows="4" placeholder="…or paste the hand histories here" spellcheck="false"></textarea>`
      + (reading ? `<div class="ss-note">${esc(reading)}</div>` : "")
      + (draft ? `<div class="ss-found"><b>${draft.hands.length} hand${draft.hands.length === 1 ? "" : "s"} to analyze</b>${draft.name ? ` from ${esc(draft.name)}` : ""} (${draft.total} in ${nf === 1 ? "the file" : `${nf} files`}${skl.length ? `; skipped: ${skl.join(", ")}` : ""})`
        + `${draft.errors.length ? `<details><summary>${draft.errors.length} unreadable</summary>${draft.errors.slice(0, 20).map(e => `<div class="ss-sub">${e.file ? `${esc(e.file)}, ` : ""}hand ${e.n}${e.id ? ` (#${esc(e.id)})` : ""}: ${esc(e.msg)}</div>`).join("")}</details>` : ""}</div>`
        + tourHtml()
        + `<div class="ss-scopes" role="radiogroup">${Object.entries(SCOPES).map(([k, l]) => `<button role="radio" aria-checked="${k === scope}" data-scope="${k}" class="${k === scope ? "active" : ""}">${l}${est(k)}</button>`).join("")}</div>`
        + `<div class="ss-note">Flop-library spots are the precomputed flops (heads-up, and the first decision of the commonest 3-way pots): instant, the rest of the flops skipped. Quick triage grades every decision: from the charts and the library, else read at once off stored solves, else a Quick solve, each with how far to trust it; then "Go deeper" asks the flagged ones again at Fast. All streets at Fast solves each decision as a hand review would.</div>`
        + `<div class="ss-row"><button class="active" data-go${draft.hands.length ? "" : " disabled"}>Analyze</button></div>` : "");
  }
  function tourText(s) {
    const ts = s.tourneys || [];
    if (!ts.length) return "";
    const icm = ts.filter(t => t.payouts && t.payouts.length >= 2);
    return ` · ${ts.length === 1 ? `tournament ${esc(ts[0].name || "#" + (ts[0].id || "?"))}` : `${ts.length} tournaments`}${icm.length ? `, ${ts.length === 1 ? "" : icm.length + " "}in prize equity (payouts ${icm.length === 1 ? icm[0].payouts.join("/") : "entered"})` : ", in chips"}`;
  }
  function runHtml() {
    const n = sess.items.length, k = sess.next, rn = run && run.running, cur = run && run.current;
    const st = rn ? (run.stopping ? "Pausing after this decision…" : run.waiting ? "Waiting while the page uses the solver…" : cur ? `Hand ${sess.hands[cur.h].n}${sess.hands[cur.h].id ? ` (#${esc(sess.hands[cur.h].id)})` : ""}, ${cur.s}${cur.deep ? " at Fast" : ""}` : "Starting…")
      : { done: "Done.", paused: "Paused.", stopped: "Stopped.", ready: "Ready." }[sess.status] || "";
    const sec = eta(sess, host.budgets(), nth()), left = sess.status === "done" || sess.status === "stopped" || !n ? "" : sec < 1 ? " · almost done" : ` · about ${fmtDur(sec)} left`;
    let btns = rn ? `<button data-pause${run.stopping ? " disabled" : ""}>Pause</button><button data-stop>Stop</button>`
      : sess.status === "paused" || sess.status === "ready" || sess.status === "stopped" && k < n ? `<button class="active" data-resume>${sess.status === "ready" ? "Start" : "Resume"}</button>` : "";
    if (!rn && sess.scope === "triage" && k >= n) {
      const fl = flaggedRows(sess), every = flaggedRows(sess, true), t = (rs) => fmtDur(rs.reduce((x, r) => x + estimate(r.s, "all", host.budgets(), nth()), 0));
      if (fl.length) btns += `<button class="active" data-deep="flagged" title="Ask the flagged decisions again at Fast, the hands with the biggest pots first">Go deeper: ${fl.length} flagged <small>~${t(fl)}</small></button>`;
      if (every.length > fl.length) btns += `<button data-deep="every" title="Ask every decision the triage didn't answer from the charts or the library again at Fast">All ${every.length} at Fast <small>~${t(every)}</small></button>`;
    }
    if (!rn && sess.results.length) btns += `<button data-csv>Export CSV</button><button data-json>Export JSON</button>`;
    if (!rn && sess.status === "done" && (sess.scope === "preflop" || sess.scope === "library")) btns += `<button data-widen title="Also ask the decisions this scope left out">${sess.scope === "preflop" ? "Add the flop-library spots" : "Add all streets"}</button>`;
    if (!rn) btns += `<button class="link" data-new>New session</button>`;
    const all = remember(history, summarize(sess)), other = !rn && sess.status === "paused" && elsewhere();
    return `<div class="ss-prog"><div class="ss-bar"><i style="width:${n ? (k / n * 100).toFixed(1) : 0}%"></i></div><div class="ss-st"><span>${k} of ${n} decisions${left}</span><span>${st}</span></div>`
      + `<div class="ss-sub">${esc(sess.name || "Session")} · ${sess.hands.length} hands · ${SCOPES[sess.scope]}${tourText(sess)}${rn ? " · goes on with this panel closed" : ""}</div></div>`
      + (other ? `<div class="ss-note">Running in another tab of this page.</div>` : "") + `<div class="ss-row">${btns}</div>` + reportHtml()
      + (all.some(x => x.id !== sess.created) ? historyHtml(all, "Across your sessions") : "");
  }
  function draw() {
    if (!box || box.hidden) return;
    const body = box.querySelector(".ss-body"), y = body.scrollTop;
    body.innerHTML = !loaded ? `<div class="ss-note">Loading…</div>` : sess ? runHtml() : inputHtml() + historyHtml(history, "Your sessions so far");
    body.scrollTop = y;
  }
  // files picked, dropped or a folder's: the hand-history ones read in order of their paths (one file of any name)
  async function readFiles(list) {
    const all = Array.from(list || []), files = hhFiles(all).length ? hhFiles(all) : all.length === 1 ? all : [];
    if (!files.length) { reading = "No hand-history files there (.txt, .xml, .log, .hh)."; return draw(); }
    const at = (f) => String(f.path || f.webkitRelativePath || f.name || "");
    files.sort((a, b) => at(a).localeCompare(at(b)));
    draft = null; reading = files.length > 1 ? `Reading ${files.length} files…` : "Reading the hands…"; draw();
    try {
      const texts = [];
      for (const f of files) texts.push({ name: f.name || "", text: await f.text() });
      draft = await prepare(texts.length === 1 ? texts[0].text : texts, host.importHand, (k, n) => { reading = `Reading the hands… ${k} of ${n}`; draw(); });
      draft.name = filesName(files);
      reading = draft.total ? null : "No hand histories found in that.";
    } catch (e) { reading = "Couldn't read that: " + (e.message || e); }
    draw();
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
  // a drop's files, a folder's walked (its entries read while the drop event lasts)
  async function droppedFiles(entries, files) {
    if (!entries.some(e => e && e.isDirectory)) return files;
    const out = [];
    const walk = async (e) => {
      if (e.isFile) { const f = await new Promise((res, rej) => e.file(res, rej)); try { Object.defineProperty(f, "path", { value: e.fullPath }); } catch (x) {} out.push(f); }
      else if (e.isDirectory) {
        const rd = e.createReader();
        for (;;) { const batch = await new Promise((res, rej) => rd.readEntries(res, rej)); if (!batch.length) break; for (const c of batch) await walk(c); }
      }
    };
    for (const e of entries) if (e) await walk(e);
    return out;
  }
  function download(name, text, type) {
    const a = document.createElement("a"), url = URL.createObjectURL(new Blob([text], { type }));
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const fileBase = () => (sess.name || "session").replace(/\.[^.]*$/, "").replace(/[^\w.-]+/g, "_") + "-analysis";
  async function act(d) {
    if (d.scope) { scope = d.scope; return draw(); }
    if (d.go != null && draft) { const s = newSession(draft, scope, draft.name); draft = null; F = {}; showN = 25; return start(s); }
    if (d.resume != null && sess) return start(sess);
    if (d.pause != null && run) return run.pause();
    if (d.stop != null && sess) { if (run && run.running) run.cancel(); else { sess.status = "stopped"; await save(sess); } return draw(); }
    if (d.widen != null && sess) return start(widen(sess, sess.scope === "preflop" ? "library" : "all"));
    if (d.deep != null && sess && !(run && run.running)) return start(deepen(sess, d.deep === "every"));
    if (d.new != null) { if (run && run.running) return; keep(sess); sess = run = null; F = {}; if (store) await store.del().catch(() => {}); return draw(); }
    if (d.csv != null) return download(fileBase() + ".csv", toCSV(sess), "text/csv");
    if (d.json != null) return download(fileBase() + ".json", toJSON(sess), "application/json");
    if (d.k != null) { const r = sess.results[+d.k], h = sess.hands[r.h]; close(); return host.open(h.state, r.s, r.i); }
    if (d.leak != null) { const l = report(sess, 0).leaks[+d.leak]; if (l) { F = { ...F, leak: { kind: l.kind, key: l.key } }; showN = 25; } return draw(); }
    if (d.leakoff != null) { F = { ...F, leak: null }; return draw(); }
    if (d.filter) { F = { ...F, [d.filter]: d.value || "" }; showN = 25; return draw(); }
    if (d.tag) { const t = F.tags || []; F = { ...F, tags: t.includes(d.tag) ? t.filter(x => x !== d.tag) : [...t, d.tag] }; showN = 25; return draw(); }
    if (d.clearf != null) { F = {}; showN = 25; return draw(); }
    if (d.more != null) { showN += 25; return draw(); }
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
  .ss-scopes { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
  .ss-tour { margin: 8px 0; padding: 8px 10px; border: 1px solid var(--line2); border-radius: 8px; font-size: 13px; }
  .ss-trow { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 6px; } .ss-trow span { flex: 1 1 200px; }
  .ss-trow input[type=text] { width: 130px; } .ss-trow input[type=number] { width: 100px; }
  .ss-folder { font-size: 12px; cursor: pointer; } .ss-folder input { position: absolute; width: 1px; height: 1px; opacity: 0; }
  .ss-flt { display: flex; flex-direction: column; gap: 6px; } .ss-flt select { font-size: 12px; padding: 5px 6px; max-width: 100%; }
  .ss-tags { display: flex; flex-wrap: wrap; gap: 4px; }
  .ss-tag { font-size: 11px; padding: 2px 8px; border-radius: 10px; } .ss-tag.active { border-color: var(--accent); background: var(--line); }
  .ss-tc { display: inline-block; font-size: 10px; padding: 0 5px; border-radius: 8px; border: 1px solid var(--line2); margin-left: 4px; }
  .ss-tc.ok { color: var(--good); } .ss-tc.meh { color: var(--muted); } .ss-tc.bad { color: var(--bad); }
  .ss-tri { border-left: 2px solid var(--line2); padding-left: 8px; }
  .ss-row button small { color: var(--muted); font-weight: 400; margin-left: 4px; }
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
    .ss-scopes { grid-template-columns: 1fr 1fr; }
    .ss-flt select { font-size: 16px; flex: 1 1 40%; }
    .ss-body textarea { font-size: 16px; }
    .ss-tiles { grid-template-columns: repeat(2, 1fr); }
    .ss-t { font-size: 11px; } .ss-t th, .ss-t td { padding: 3px 4px; }
  }`;
  function init(h) {
    host = h || host || pageHost();
    if (!store) try { store = defaultStore(); } catch (e) { store = lsStore(); }
    if (!hstore) try { hstore = defaultStore(HISTORY_KEY); } catch (e) { hstore = lsStore("gto-session-" + HISTORY_KEY); }
  }
  // the session kept (one saved while it ran goes on: wasRunning) and the history
  function load() {
    return loading = loading || (async () => {
      try { await host.ready(); } catch (e) {}
      if (!sess) {
        try {
          const s = await store.get();
          if (s && s.v === 1 && s.items) { sess = s; wasRunning = s.status === "running"; if (s.status === "running" || s.status === "ready") sess.status = "paused"; }
        } catch (e) {}
      }
      try { history = remember(await hstore.get(), null); } catch (e) { history = []; }
      loaded = true; draw();
    })();
  }
  // a session left running (the page closed or reloaded mid-way) picks up where it stopped, unless another tab has it
  function goOn() {
    if (!wasRunning || !sess || run && run.running || sess.next >= sess.items.length || elsewhere()) return null;
    wasRunning = false;
    return start(sess);
  }
  // the page, loaded with a session left running (index.html, RUN_KEY): it goes on in the background, the panel closed
  async function resume(h) {
    init(h);
    await load();
    return goOn();
  }
  async function open(h) {
    init(h);
    if (!document.getElementById("ssStyle")) { const s = document.createElement("style"); s.id = "ssStyle"; s.textContent = CSS; document.head.appendChild(s); }
    if (!box) {
      box = document.createElement("div"); box.id = "sessionBox"; box.className = "ss"; box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true"); box.setAttribute("aria-label", "Analyze session");
      box.innerHTML = `<div class="ss-card"><div class="ss-head"><b>Analyze session</b><span class="sub">every decision of a session's hands, solved and graded</span><button class="ghost ss-x" data-close aria-label="Close">✕</button></div><div class="ss-body"></div></div>`;
      document.body.appendChild(box);
      box.addEventListener("click", (e) => {
        if (e.target === box) return close();
        const t = e.target.closest("[data-close],[data-scope],[data-go],[data-resume],[data-pause],[data-stop],[data-widen],[data-deep],[data-new],[data-csv],[data-json],[data-k],[data-forget],[data-clearhist],[data-leak],[data-leakoff],[data-tag],[data-clearf],[data-more]");
        if (!t || t.disabled) return;
        if (t.dataset.close != null) close(); else act(t.dataset);
      });
      box.addEventListener("change", (e) => {
        const ds = e.target.dataset || {};
        if (ds.filter) return act({ filter: ds.filter, value: e.target.value });
        if (ds.tour != null && draft) { const t = (draft.tset = draft.tset || {})[ds.tour] = { ...(draft.tset[ds.tour] || {}) }; t[ds.tf] = e.target.value; return; }
        if (ds.file != null && e.target.files && e.target.files.length) { readFiles(e.target.files); e.target.value = ""; }
      });
      box.addEventListener("input", (e) => { if (e.target.dataset && e.target.dataset.paste != null) { clearTimeout(box._t); const v = e.target.value; box._t = setTimeout(() => v.trim() && readText(v, ""), 400); } });
      box.addEventListener("pointerdown", () => { box._down = true; });
      window.addEventListener("pointerup", () => { box._down = false; }); window.addEventListener("pointercancel", () => { box._down = false; });
      box.addEventListener("dragover", (e) => { e.preventDefault(); const d = box.querySelector("[data-drop]"); if (d) d.classList.add("over"); });
      box.addEventListener("dragleave", (e) => { if (e.target.closest && e.target.closest("[data-drop]")) e.target.closest("[data-drop]").classList.remove("over"); });
      box.addEventListener("drop", (e) => {
        e.preventDefault();
        const dt = e.dataTransfer;
        if (!dt || sess) return;
        // (a folder's entries only while the event lasts)
        const entries = Array.from(dt.items || []).map(i => i.webkitGetAsEntry ? i.webkitGetAsEntry() : null), files = Array.from(dt.files || []);
        droppedFiles(entries, files).then(readFiles, (err) => { reading = "Couldn't read that: " + (err && err.message || err); draw(); });
      });
      // the page's own shortcuts stay off while the panel is open
      window.addEventListener("keydown", (e) => { if (box.hidden) return; e.stopPropagation(); if (e.key === "Escape") close(); }, true);
    }
    box.hidden = false; draw();
    await load();
    goOn();
    draw();
  }

  const api = { handId, decisions, prepare, buildItems, inScope, spotState, preflopRaises, preflopSituation, classify, mixOf, newSession, widen,
                processItem, runner, estimate, eta, report, tally, freqs, toCSV, toJSON, rowsOf, lsStore, idbStore, pageHost, open, close, resume, SCOPES,
                hhFiles, filesName, withPayouts, tourSet, flagged, deepen, flaggedCount, agreement, filterRows, filterOn, tagsOf, leakTest, LOSS, TAGS, POT_TYPES, DEEP_LOSS, RUN_KEY,
                summarize, remember, mergeRows, trends, leaks, historyHtml, get history() { return history; },
                get session() { return sess; }, get runner_() { return run; } };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Session = api;
})();
