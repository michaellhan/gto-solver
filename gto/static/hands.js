// My hands, loaded by index.html the first time it opens (the ⋯ menu's My hands…, /gto-solver/hands/): the hands kept
// in this browser by store.js (GTOStore.hands): each hand you played on the table (kept when New hand or another hand
// takes its place), reviewed (Review whole hand, with its grades) or imported. Search, a filter by kind and tag, tags and
// a note on each, delete (Undo); a hand opens on the table (to review it) or in Study at your last decision, and copies
// as its #h= link; the list (or one hand) exports as JSON, and such a file, #h= links or hand histories import into it.
// Below the list, a backup of everything the site keeps in this browser and its restore (GTOStore.backup / restore).
// Everything but the panel (open) and the page host is pure and tested in Node.
(function () {
  "use strict";
  const STREETS = ["preflop", "flop", "turn", "river"];
  const SUIT = { s: "♠", h: "♥", d: "♦", c: "♣" };
  const KINDS = [["", "All"], ["reviewed", "Reviewed"], ["unreviewed", "Not reviewed"], ["import", "Imported"]];
  const PAGE = 50, UI_KEY = "gto-hands-ui";
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const clone = (x) => JSON.parse(JSON.stringify(x));
  const okCard = (c) => typeof c === "string" && /^[AKQJT98765432][shdc]$/.test(c);
  const fmt = (x) => (Math.round(x * 100) / 100).toString();
  const cardH = (c) => `<span class="mh-c ${c[1]}">${c[0]}${SUIT[c[1]]}</span>`;
  const cardsH = (cs) => cs.map(cardH).join("");
  // a grade's class as the page's gradeClass (✓ in the solver's mix 15%+, ~ 3%+ or a rough solve, ✗ less; the betting
  // rules' grades by the EV share given up)
  /* global gradeClass */
  const gcls = (g) => typeof gradeClass === "function" ? gradeClass(g)
    : g.h != null ? (g.h <= 0.03 ? "ok" : g.h <= 0.1 ? "meh" : "bad") : g.f >= 0.15 ? "ok" : g.f >= 0.03 || g.r ? "meh" : "bad";
  const MARK = { ok: "✓", meh: "~", bad: "✗" };

  // ---- a hand at a glance: who, what, how far, how it graded
  function summary(st) {
    st = st && typeof st === "object" ? st : {};
    const n = +st.players || 6, hero = st.hero || "", lab = (p) => n === 2 && p === "SB" ? "BTN/SB" : p;
    const cards = Array.isArray(st.cards) ? st.cards : [];
    const hole = [cards[0], cards[1]].filter(okCard), board = cards.slice(2, 7).filter(okCard);
    const A = (s) => st.actions && Array.isArray(st.actions[s]) ? st.actions[s].filter(a => a && typeof a === "object") : [];
    const pre = A("preflop");
    const raises = pre.filter(a => a.action === "raise" || a.action === "all-in").length;
    const pot = raises === 0 ? (pre.some(a => a.action === "call") ? "limped pot" : "") : ["single-raised pot", "3-bet pot", "4-bet pot"][raises - 1] || "5-bet+ pot";
    const post = STREETS.slice(1).flatMap(A);
    const out = new Set(pre.filter(a => a.action === "fold").map(a => a.pos));
    // the opponents: those who played after the flop, else those still in preflop (a call, raise or all-in, not folded)
    let vs = [...new Set(post.map(a => a.pos))].filter(p => p !== hero);
    if (!vs.length) vs = [...new Set(pre.filter(a => a.action !== "fold" && !out.has(a.pos)).map(a => a.pos))].filter(p => p !== hero);
    let folded = null;
    for (const s of STREETS) if (A(s).some(a => a.pos === hero && a.action === "fold")) { folded = s; break; }
    // how far it went: the last street with action, or dealt (an all-in run out)
    const dealt = board.length >= 5 ? "river" : board.length === 4 ? "turn" : board.length === 3 ? "flop" : "preflop";
    const acted = [...STREETS].reverse().find(s => A(s).length) || "preflop";
    const last = STREETS.indexOf(dealt) > STREETS.indexOf(acted) ? dealt : acted;
    const mine = STREETS.flatMap(s => A(s).filter(a => a.pos === hero).map(a => ({ ...a, s })));
    const graded = { ok: 0, meh: 0, bad: 0 };
    let lost = 0;
    for (const a of mine) if (a.g && typeof a.g === "object") { graded[gcls(a.g)]++; if (+a.g.l > 0) lost += +a.g.l; }
    const nGraded = graded.ok + graded.meh + graded.bad;
    const title = `${lab(hero)}${vs.length === 1 ? " vs " + lab(vs[0]) : vs.length > 1 ? `, ${vs.length + 1}-way` : ""}`;
    const game = `${n}-handed, ${fmt(+st.stack || 100)}bb${+st.ante > 0 ? `, ${fmt(+st.ante)} ante` : ""}${st.straddle ? ", straddle" : ""}`;
    return { hero, heroLabel: lab(hero), hole, board, players: n, pot, vs: vs.map(lab), folded, street: last, decisions: mine.length,
             graded, nGraded, lost: Math.round(lost * 100) / 100, title, game };
  }
  const word = (a) => a.action === "bet" ? `bets ${fmt(+a.to)}` : a.action === "raise" ? `raises to ${fmt(+a.to)}` : a.action === "all-in" ? "all-in"
    : a.action === "fold" ? "folds" : a.action === "check" ? "checks" : a.action === "call" ? "calls" : esc(a.action);
  // the line, street by street, your actions in bold with their grades
  function lineHtml(st) {
    const s = summary(st), cards = Array.isArray(st.cards) ? st.cards : [], n = s.players, lab = (p) => n === 2 && p === "SB" ? "BTN/SB" : p;
    const out = [];
    for (const k of STREETS) {
      const acts = st.actions && Array.isArray(st.actions[k]) ? st.actions[k].filter(a => a && typeof a === "object") : [];
      const bd = k === "flop" ? cards.slice(2, 5) : k === "turn" ? cards.slice(5, 6) : k === "river" ? cards.slice(6, 7) : [];
      if (!acts.length && !bd.filter(okCard).length) continue;
      const items = acts.map(a => {
        const t = `${esc(lab(a.pos))} ${word(a)}`;
        if (a.pos !== st.hero) return t;
        const g = a.g && typeof a.g === "object" ? a.g : null, c = g && gcls(g);
        return `<b>${t}</b>${g ? ` <span class="mh-g ${c}" title="the solver plays it ${Math.round((+g.f || 0) * 100)}%${+g.l > 0 ? `, ${fmt(+g.l)} bb given up` : ""}">${MARK[c]}${+g.l > 0.005 ? ` −${fmt(+g.l)}` : ""}</span>` : ""}`;
      });
      out.push(`<div class="mh-st"><span class="mh-sn">${k[0].toUpperCase() + k.slice(1)}${bd.filter(okCard).length ? " " + cardsH(bd.filter(okCard)) : ""}</span> ${items.join(", ") || "<small>—</small>"}</div>`);
    }
    return out.join("");
  }
  const day = (t) => { const d = new Date(t); return isNaN(d) ? "" : d.toISOString().slice(0, 10); };
  const when = (t) => {
    const d = new Date(t); if (isNaN(d)) return "";
    const p = (x) => String(x).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  };
  const SRC = { table: "played here", link: "from a link", review: "played here", import: "imported", file: "from a file" };
  // what a search reads: positions, cards (Ah Kd, AhKd, AKo), the board, the kind of pot, tags, the note, the site
  function searchText(rec) {
    const d = rec.data || {}, s = summary(d.state), h = s.hole;
    const cls = h.length === 2 ? (() => { const r = "AKQJT98765432", [a, b] = [...h].sort((x, y) => r.indexOf(x[0]) - r.indexOf(y[0])); return a[0] === b[0] ? a[0] + b[0] : a[0] + b[0] + (a[1] === b[1] ? "s" : "o"); })() : "";
    return [s.title, s.heroLabel, ...s.vs, h.join(""), h.join(" "), cls, s.board.join(""), s.board.join(" "), s.pot, s.street, s.game, s.folded ? "folded" : "",
            d.reviewed ? "reviewed" : "", SRC[d.source] || "", d.site || "", d.hhId || "", ...(d.tags || []).map(t => "#" + t), d.note || ""].join(" ").toLowerCase();
  }
  function matches(rec, q) { const t = searchText(rec); return String(q || "").toLowerCase().split(/\s+/).filter(Boolean).every(w => t.includes(w)); }
  function filterRecs(recs, ui = {}) {
    return recs.filter(r => r && r.data && (!ui.kind || (ui.kind === "reviewed" ? r.data.reviewed : ui.kind === "unreviewed" ? !r.data.reviewed : ui.kind === "import" ? r.data.source === "import" || r.data.source === "file" : true))
      && (!ui.tag || (r.data.tags || []).includes(ui.tag)) && (!ui.q || matches(r, ui.q)));
  }
  function tagCounts(recs) {
    const m = new Map(); for (const r of recs) for (const t of (r.data && r.data.tags) || []) m.set(t, (m.get(t) || 0) + 1);
    return [...m].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }
  const cleanTag = (t) => String(t || "").replace(/[,\s]+/g, " ").trim().replace(/^#+/, "").trim().slice(0, 40);

  function rowHtml(rec, isOpen) {
    const d = rec.data || {}, s = summary(d.state), id = esc(rec.id);
    const grades = s.nGraded ? `<span class="mh-grades" title="your decisions graded: in the solver's mix, a little off, a mistake">${["ok", "meh", "bad"].filter(k => s.graded[k]).map(k => `<span class="mh-g ${k}">${MARK[k]}${s.graded[k]}</span>`).join(" ")}${s.lost > 0.005 ? ` <span class="mh-lost">−${fmt(s.lost)} bb</span>` : ""}</span>`
      : s.decisions ? `<span class="mh-sub">${s.decisions} decision${s.decisions > 1 ? "s" : ""}, not reviewed</span>` : "";
    const tags = (d.tags || []).map(t => `<button class="mh-tag" data-tag="${esc(t)}" title="Only hands tagged ${esc(t)}">#${esc(t)}</button>`).join("");
    const head = `<div class="mh-top"><span class="mh-when">${esc(when(rec.created))}</span><b>${esc(s.title)}</b>${s.hole.length ? `<span class="mh-hole">${cardsH(s.hole)}</span>` : ""}`
      + `${s.board.length ? `<span class="mh-board">${cardsH(s.board)}</span>` : ""}<span class="mh-sub">${esc([s.pot, s.folded ? `folded ${s.folded === "preflop" ? "preflop" : "on the " + s.folded}` : s.street !== "preflop" ? "to the " + s.street : "", d.site ? `${d.site}${d.hhId ? " #" + d.hhId : ""}` : ""].filter(Boolean).join(" · "))}</span>`
      + `${grades}${tags}</div>`;
    const note = d.note && !isOpen ? `<div class="mh-note1">${esc(d.note.split("\n")[0].slice(0, 140))}</div>` : "";
    const acts = `<div class="mh-acts"><button data-open="${id}" title="Put this hand on the table, to see it or review it">Open</button>`
      + `<button data-study="${id}" title="Study this hand at your last decision: the solver's strategy for every hand of the range">Study</button>`
      + `<button data-link="${id}" title="Copy a link that opens this hand (#h=)">Copy link</button>`
      + `<button class="ghost" data-toggle="${id}" aria-expanded="${isOpen ? "true" : "false"}">${isOpen ? "Less" : "Tags, note…"}</button></div>`;
    const more = isOpen ? `<div class="mh-more">${lineHtml(d.state || {})}<div class="mh-sub">${esc(s.game)} · ${esc(SRC[d.source] || d.source || "")}</div>`
      + `<div class="mh-tagrow">${(d.tags || []).map(t => `<span class="mh-chip">#${esc(t)}<button class="link" data-rmtag="${esc(t)}" data-id="${id}" aria-label="Remove tag ${esc(t)}">×</button></span>`).join("")}`
      + `<input type="text" data-addtag="${id}" list="mhTags" placeholder="add a tag, Enter" maxlength="40" aria-label="Add a tag"></div>`
      + `<textarea data-note="${id}" rows="2" placeholder="A note: reads on villain, what you were thinking…" aria-label="Note">${esc(d.note || "")}</textarea>`
      + `<div class="mh-acts"><button class="ghost" data-json="${id}" title="Download this hand as a JSON file">Download JSON</button><button class="ghost mh-del" data-del="${id}">Delete</button></div></div>` : "";
    return `<li class="mh-row${isOpen ? " open" : ""}" data-row="${id}">${head}${note}${acts}${more}</li>`;
  }
  function listHtml(recs, ui, shown, openId) {
    const L = filterRecs(recs, ui);
    if (!recs.length) return `<div class="mh-empty">No hands yet. Every hand you enter is kept here when you start a new one (New hand), and so are the hands you review or import.</div>`;
    if (!L.length) return `<div class="mh-empty">No hand matches. <button class="link" data-clear>Show every hand</button></div>`;
    return `<ul class="mh-list">${L.slice(0, shown).map(r => rowHtml(r, r.id === openId)).join("")}</ul>`
      + (L.length > shown ? `<button class="ghost mh-moreb" data-more>Show ${Math.min(PAGE, L.length - shown)} more (${L.length - shown} left)</button>` : "");
  }
  function toolbarHtml(recs, ui) {
    const n = (k) => filterRecs(recs, { kind: k }).length;
    const kinds = KINDS.map(([k, t]) => `<button data-kind="${k}" class="${(ui.kind || "") === k ? "active" : ""}">${t} <small>${n(k)}</small></button>`).join("");
    const tags = tagCounts(recs);
    return `<div class="mh-tools"><input type="search" data-q value="${esc(ui.q || "")}" placeholder="Search: cards (AKs, Ah), positions, board, tags, notes…" aria-label="Search your hands">`
      + `<span class="segment">${kinds}</span></div>`
      + (tags.length ? `<div class="mh-tags">${tags.map(([t, c]) => `<button class="mh-tag${ui.tag === t ? " active" : ""}" data-tag="${esc(t)}">#${esc(t)} <small>${c}</small></button>`).join("")}</div>` : "")
      + `<datalist id="mhTags">${tags.map(([t]) => `<option value="${esc(t)}">`).join("")}</datalist>`;
  }

  // ---- files
  // what an import's text is: a My hands export or a state's JSON, a backup, #h= links, or hand histories
  function parseImport(text) {
    const t = String(text || "").trim();
    if (!t) return { error: "Nothing to import." };
    if (t[0] === "{" || t[0] === "[") {
      let obj; try { obj = JSON.parse(t); } catch (e) { return { error: "That isn't valid JSON." }; }
      if (obj && obj.format === "gto-solver-backup") return { kind: "backup", obj };
      return { kind: "hands", obj };
    }
    const links = [...t.matchAll(/#h=([A-Za-z0-9+/=]+)/g)].map(m => m[1]);
    if (links.length) return { kind: "links", links };
    return { kind: "hh", text: t };
  }
  const fileName = (what, t) => `gto-solver-${what}-${day(t)}.json`;
  function backupText(b) {
    const c = b.counts || Object.fromEntries(Object.entries(b.stores || {}).map(([k, v]) => [k, (v || []).length]));
    const parts = [[c.hands, "hand", "hands"], [c.sessions, "analyzed session", "analyzed sessions"], [c.kv, "setting or saved state", "settings and saved states"]]
      .filter(([x]) => x).map(([x, one, many]) => `${x} ${x === 1 ? one : many}`);
    return (parts.join(", ") || "nothing") + (c.blobs ? ", a session in progress" : "");
  }
  function download(name, obj) {
    const blob = new Blob([JSON.stringify(obj, null, 1)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    return blob.size;
  }

  // ---- the page (globals of index.html; session.js's host does the same)
  function pageHost() {
    // the hand on the table, as that record (the page's cleaned copy of it kept, should it differ: a file's hand)
    const onTable = async (rec) => {
      const H = globalThis.GTOStore.hands, st = clone(rec.data.state);
      if (rec.data.source === "import" || rec.data.source === "review") st.finished = true;   // (kept before hands were marked finished)
      await loadHandState(st);
      if (H.sig(S) !== rec.data.sig) await H.update(rec.id, { state: S });
      H.onTable(S, { id: rec.id, source: rec.data.source });
    };
    /* global S, cleanState, setPositions, payload, loadHandState, rewindTo, reviewHand, toast, encodeState, decodeState, ROOT, openView, loadJs, SY_VERSION, copyText, loadHH, splitHands, importHand, simulate, handDone, heldHand */
    return {
      store: () => globalThis.GTOStore,
      table: () => clone(heldHand()),
      clean: (st) => { try { return cleanState(clone(st)); } catch (e) { return null; } },
      link: (st) => location.origin + ROOT + "#h=" + encodeState(st),
      decode: (b64) => decodeState(b64),
      copy: (t) => copyText(t),
      async open(rec) {
        await onTable(rec);
        const ungraded = STREETS.some(s => S.actions[s].some(a => a.pos === S.hero && !a.g));
        const s = summary(S);
        const done = handDone(simulate());
        toast(`Opened your hand: ${esc(s.title)}${s.hole.length ? ", " + s.hole.join("") : ""}.${done ? "" : " It isn't finished: enter the rest, or press Finish hand, to review it."}`, null, 10000);
        if (ungraded && done) {
          const t = document.getElementById("toast"), b = document.createElement("button"); b.className = "link"; b.textContent = "Review whole hand";
          b.onclick = () => { t.hidden = true; reviewHand(); }; t.appendChild(b);
        }
      },
      // Study at your last decision (the table keeps the whole hand)
      async study(rec) {
        await onTable(rec);
        await loadJs("study.js?v=" + SY_VERSION, () => globalThis.Study);
        const at = lastDecision(S), old = S;
        let p = null;
        try { if (at) { S = at; setPositions(S.players); } p = payload(); } finally { S = old; setPositions(old.players); }
        globalThis.STUDY_AT = { spot: "table", state: globalThis.Study.fromPayload(p) };
        return openView("study");
      },
      async importHH(text) {
        await loadHH();
        return splitHands(text).map(h => { try { const r = importHand(h); return { state: r.state, site: r.hand.site, hh: h }; } catch (e) { return { error: e.message || String(e) }; } });
      },
    };
  }
  // the hand rewound to your last decision (null if you have none): what Study opens
  function lastDecision(st) {
    let at = null;
    for (const s of STREETS) (st.actions[s] || []).forEach((a, i) => { if (a.pos === st.hero) at = [s, i]; });
    if (!at) return null;
    const [s, i] = at, out = clone(st), k = STREETS.indexOf(s);
    out.actions[s] = out.actions[s].slice(0, i);
    for (const t of STREETS.slice(k + 1)) out.actions[t] = [];
    const keep = { preflop: 2, flop: 5, turn: 6, river: 7 }[s];
    for (let c = keep; c < 7; c++) out.cards[c] = null;
    return out;
  }

  // ---- the panel
  let deleted = null, box = null, host = null, recs = [], ui = { q: "", kind: "", tag: "" }, shown = PAGE, openId = null, msg = "", draft = null, unsub = null, noteT = null, loaded = false;
  const store = () => host.store();
  // the panel's edits to a hand, one after another (a tag and a note written at once don't undo each other)
  let edits = Promise.resolve();
  const upd = (id, patch) => (edits = edits.then(() => store().hands.update(id, patch)).catch(() => null));
  const saveUi = () => { try { store().set(UI_KEY, { kind: ui.kind, tag: ui.tag }); } catch (e) {} };
  const say = (m) => { msg = m; const el = box && box.querySelector(".mh-msg"); if (el) { el.innerHTML = m; el.hidden = !m; } };
  async function refresh() { try { recs = await store().hands.list(); } catch (e) { recs = []; } loaded = true; draw(); }
  function tableHtml() {
    let st = null; try { st = host.table(); } catch (e) {}
    const H = store().hands;
    if (!st || !H.worth(st)) return "";
    const sig = H.sig(st), saved = recs.some(r => r.data && r.data.sig === sig);
    if (saved) return "";
    const s = summary(st);
    return `<div class="mh-table">On the table: <b>${esc(s.title)}</b> ${cardsH(s.hole)} ${cardsH(s.board)} <span class="mh-sub">not saved yet (it's kept when you start a new hand)</span> <button data-savetable>Save it now</button></div>`;
  }
  function backupHtml() {
    if (draft) return `<div class="mh-backup"><b>Restore this backup?</b><div class="mh-sub">Exported ${esc(when(Date.parse(draft.exported)))}${draft.device && draft.device === store().device ? " from this browser" : " from another browser"}: ${esc(backupText(draft))}.
      What's missing here is added and, of a record in both, the newer one stays: nothing here is deleted. Settings take effect when the page loads again.</div>
      <div class="mh-acts"><button class="active" data-restorego>Restore</button><button class="ghost" data-restorecancel>Cancel</button></div></div>`;
    return `<div class="mh-backup"><b>Backup</b><div class="mh-sub">Everything this site keeps in this browser in one file: these hands, your analyzed sessions, Practice's score and
      mistakes, the drills' progress, Study's and the page's settings. Your Anthropic API key (Explain with AI) is never in it.</div>
      <div class="mh-acts"><button data-backup>Back up everything</button><label class="mh-file"><input type="file" data-restorefile accept=".json,application/json"><span>Restore from a backup…</span></label></div></div>`;
  }
  function draw() {
    if (!box || box.hidden) return;
    const body = box.querySelector(".mh-body");
    if (!loaded) { body.innerHTML = `<div class="mh-sub">Loading…</div>`; return; }
    const n = filterRecs(recs, ui).length;
    body.innerHTML = tableHtml() + toolbarHtml(recs, ui)
      + `<div class="mh-msg" role="status"${msg ? "" : " hidden"}>${msg}</div>`
      + `<div class="mh-listbox">${listHtml(recs, ui, shown, openId)}</div>`
      + `<div class="mh-io"><span class="mh-sub mh-count">${countText(n)}</span>`
      + `<button class="ghost" data-export${n ? "" : " disabled"} title="Download the hands listed as a JSON file">${exportText(n)}</button>`
      + `<label class="mh-file ghost"><input type="file" data-importfile accept=".json,.txt,.xml,.log,application/json,text/plain"><span>Import a file…</span></label>`
      + `<button class="ghost" data-pastebox>Paste…</button></div>`
      + `<div class="mh-paste" hidden><textarea data-paste rows="3" placeholder="Paste hand histories, #h= links or a My hands JSON export"></textarea><div class="mh-acts"><button data-pastego>Import</button></div></div>`
      + backupHtml();
  }
  const countText = (n) => `${n === recs.length ? `${recs.length} hand${recs.length === 1 ? "" : "s"}` : `${n} of ${recs.length} hands`} · kept in this browser only`;
  const exportText = (n) => `Export ${n === recs.length ? "all" : n} as JSON`;
  function drawList() {
    const el = box.querySelector(".mh-listbox"); if (el) el.innerHTML = listHtml(recs, ui, shown, openId);
    const n = filterRecs(recs, ui).length, c = box.querySelector(".mh-count"), x = box.querySelector("[data-export]");
    if (c) c.textContent = countText(n);
    if (x) { x.textContent = exportText(n); x.disabled = !n; }
  }
  const recOf = (id) => recs.find(r => r.id === id);

  async function importText(text, name) {
    const p = parseImport(text);
    if (p.error) return say(esc(p.error));
    if (p.kind === "backup") { draft = p.obj; draw(); return say("That's a backup of everything: restore it below."); }
    let r;
    try {
      if (p.kind === "hands") r = await store().hands.importFile(p.obj, host.clean, host.decode);
      else if (p.kind === "links") {
        const states = p.links.map(b => { try { return host.decode(b); } catch (e) { return null; } }).filter(Boolean);
        r = await store().hands.importFile(states.map(state => ({ state, source: "link" })), host.clean);
        r.skipped += p.links.length - states.length;
      } else {
        const hs = await host.importHH(text);
        const ok = hs.filter(h => h.state);
        r = await store().hands.importFile(ok.map(h => ({ state: h.state, source: "import", site: h.site, hh: h.hh, hhId: store().hands.handNumber(h.hh) })), host.clean);
        r.skipped += hs.length - ok.length;
        if (!ok.length && hs.length) return say(esc(`No hand could be read${hs[0].error ? ": " + hs[0].error : "."}`));
        if (!hs.length) return say("No hand history found in that text.");
      }
    } catch (e) { return say(esc(e.message || String(e))); }
    await refresh();
    say(esc(`${name ? name + ": " : ""}${r.added} hand${r.added === 1 ? "" : "s"} added${r.merged ? `, ${r.merged} already here` : ""}${r.skipped ? `, ${r.skipped} skipped (no action, or unreadable)` : ""}.`));
  }
  async function act(d, el) {
    const H = store().hands;
    if (d.clear != null) { ui = { q: "", kind: "", tag: "" }; shown = PAGE; saveUi(); return draw(); }
    if (d.kind != null) { ui.kind = d.kind; shown = PAGE; saveUi(); return draw(); }
    if (d.tag != null) { ui.tag = ui.tag === d.tag ? "" : d.tag; shown = PAGE; saveUi(); return draw(); }
    if (d.more != null) { shown += PAGE; return drawList(); }
    if (d.toggle != null) { openId = openId === d.toggle ? null : d.toggle; return drawList(); }
    if (d.savetable != null) { const r = await H.save(host.table(), { source: "table" }); await refresh(); return say(r ? "Saved the hand on the table." : "Nothing to save: the hand has no action yet."); }
    if (d.open != null) { const r = recOf(d.open); if (!r) return; close(); return host.open(r); }
    if (d.study != null) { const r = recOf(d.study); if (!r) return; close(); return host.study(r); }
    if (d.link != null) { const r = recOf(d.link); if (!r) return; const ok = await host.copy(host.link(r.data.state)); return say(ok ? "Link copied: it opens this hand on the solver's page." : "Couldn't copy the link."); }
    if (d.json != null) { const r = recOf(d.json); if (r) download(fileName("hand", Date.now()), H.exportFile([r], host.link)); return; }
    if (d.export != null) { const L = filterRecs(recs, ui); if (!L.length) return; const size = download(fileName("hands", Date.now()), H.exportFile(L, host.link)); return say(esc(`Exported ${L.length} hand${L.length === 1 ? "" : "s"} (${Math.max(1, Math.round(size / 1024))} KB).`)); }
    if (d.del != null) {
      const r = recOf(d.del); if (!r) return;
      await H.remove(r.id); openId = null; deleted = r; await refresh();
      return say(`Deleted ${esc(summary(r.data.state).title)}. <button class="link" data-undo>Undo</button>`);
    }
    if (d.undo != null) { const r = deleted; deleted = null; if (!r) return; await H.restore(r); await refresh(); return say(`Brought back ${esc(summary(r.data.state).title)}.`); }
    if (d.rmtag != null) { const r = recOf(d.id); if (!r) return; r.data.tags = (r.data.tags || []).filter(t => t !== d.rmtag); drawKeep(); return upd(r.id, { tags: r.data.tags }); }
    if (d.pastebox != null) { const p = box.querySelector(".mh-paste"); p.hidden = !p.hidden; if (!p.hidden) p.querySelector("textarea").focus(); return; }
    if (d.pastego != null) { const t = box.querySelector("[data-paste]"); return importText(t.value, ""); }
    if (d.backup != null) {
      const b = await store().backup(), size = download(fileName("backup", Date.now()), b);
      return say(esc(`Backed up ${backupText(b)} (${Math.max(1, Math.round(size / 1024))} KB). Keep the file somewhere safe; Restore from a backup brings it back, here or in another browser.`));
    }
    if (d.restorecancel != null) { draft = null; return draw(); }
    if (d.restorego != null) {
      const b = draft; draft = null;
      let r; try { r = await store().restore(b); } catch (e) { draw(); return say(esc(e.message || String(e))); }
      await refresh();
      return say(esc(`Restored: ${r.added} added, ${r.updated} updated, ${r.kept} already here.`) + (r.stores.kv && (r.stores.kv.added || r.stores.kv.updated) ? ` <button class="link" data-reload>Reload the page</button> for the settings.` : ""));
    }
    if (d.reload != null) return location.reload();
  }
  // the panel drawn again, the box being typed in still focused with what's in it
  function drawKeep() {
    const a = document.activeElement, ds = a && box.contains(a) ? a.dataset || {} : {};
    const sel = ds.addtag != null ? `[data-addtag="${CSS.escape(ds.addtag)}"]` : ds.q != null ? "[data-q]" : null, v = a && a.value;
    draw();
    const b = sel && box.querySelector(sel); if (b) { b.value = v; b.focus(); }
  }
  // a tag added or removed: the list's copy at once (a second tag typed before the first is written joins it), the
  // record after
  function addTag(id, el) {
    const ts = String(el.value).split(",").map(cleanTag).filter(Boolean); el.value = "";
    const r = recOf(id); if (!r) return;
    const have = new Set((r.data.tags || []).map(x => x.toLowerCase())), add = ts.filter(t => !have.has(t.toLowerCase()) && have.add(t.toLowerCase()));
    if (!add.length) return;
    r.data.tags = [...(r.data.tags || []), ...add];
    drawKeep();
    return upd(id, { tags: r.data.tags });
  }
  function saveNote(id, text) {
    const r = recOf(id); if (!r) return;
    r.data.note = text;   // (the list's copy: no redraw while typing)
    clearTimeout(noteT); noteT = setTimeout(() => { noteT = null; upd(id, { note: text }); }, 400);
  }
  const CSS_TEXT = `
  .mh { position: fixed; inset: 0; z-index: 90; background: rgba(0,0,0,.55); display: flex; justify-content: center; align-items: flex-start; padding: 40px 16px 16px; }
  .mh-card { background: var(--panel2); color: var(--text); border: 1px solid var(--line2); border-radius: var(--r); width: min(820px, 100%); max-height: calc(100dvh - 56px); display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,.5); }
  .mh-head { display: flex; align-items: baseline; gap: 10px; padding: 10px 12px 6px; }
  .mh-head .sub { color: var(--muted); font-size: 12px; flex: 1; }
  .mh-x { margin-left: auto; padding: 2px 8px; }
  .mh-body { display: flex; flex-direction: column; gap: 8px; padding: 0 12px 12px; overflow: auto; }
  .mh-body > * { flex-shrink: 0; }
  .mh-sub, .mh-empty { color: var(--muted); font-size: 12px; line-height: 1.45; }
  .mh-empty { padding: 18px 4px; }
  .mh-tools { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .mh-tools input[type=search] { flex: 1 1 240px; min-width: 0; }
  .mh-tags { display: flex; gap: 4px; flex-wrap: wrap; }
  .mh-tag { padding: 1px 7px; font-size: 11px; border-radius: 10px; } .mh-tag small { color: var(--muted); }
  .mh-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; }
  .mh-row { border-bottom: 1px solid var(--line); padding: 7px 2px; display: flex; flex-direction: column; gap: 4px; }
  .mh-row.open { background: var(--line); border-radius: 6px; padding: 7px 8px; }
  .mh-top { display: flex; gap: 6px 8px; align-items: center; flex-wrap: wrap; font-size: 13px; }
  .mh-when { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; min-width: 9.5em; }
  .mh-hole, .mh-board { white-space: nowrap; }
  .mh-c { display: inline-block; min-width: 1.5em; padding: 0 2px; margin-right: 2px; border-radius: 3px; background: #f5f5f0; color: #111; font-weight: 700; text-align: center; font-size: 11px; line-height: 1.5; }
  .mh-c.h, .mh-c.d { color: #c62828; } body.four .mh-c.d { color: #1565c0; } body.four .mh-c.c { color: #2e7d32; }
  .mh-g { font-size: 11px; font-weight: 700; white-space: nowrap; } .mh-g.ok { color: var(--good); } .mh-g.meh { color: var(--meh, #d9a400); } .mh-g.bad { color: var(--bad); }
  .mh-lost { color: var(--bad); font-size: 11px; font-variant-numeric: tabular-nums; }
  .mh-note1 { color: var(--muted); font-size: 12px; font-style: italic; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .mh-acts { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .mh-acts button, .mh-io button, .mh-file span { font-size: 12px; padding: 3px 9px; }
  .mh-more { display: flex; flex-direction: column; gap: 6px; padding-top: 4px; }
  .mh-st { font-size: 12px; line-height: 1.5; } .mh-sn { color: var(--muted); display: inline-block; min-width: 4.5em; }
  .mh-tagrow { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }
  .mh-chip { font-size: 11px; border: 1px solid var(--line2); border-radius: 10px; padding: 0 4px 0 8px; display: inline-flex; align-items: center; gap: 2px; }
  .mh-chip button { padding: 0 4px; }
  .mh-tagrow input { width: 12em; font-size: 12px; }
  .mh-more textarea, .mh-paste textarea { width: 100%; box-sizing: border-box; font: 12px/1.4 inherit; resize: vertical; }
  .mh-del { color: var(--bad); margin-left: auto; }
  .mh-table { font-size: 12px; border: 1px dashed var(--line2); border-radius: 8px; padding: 6px 10px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .mh-msg { font-size: 12px; border-left: 3px solid var(--accent); padding: 4px 8px; background: var(--line); border-radius: 4px; }
  .mh-io { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; padding-top: 4px; border-top: 1px solid var(--line2); }
  .mh-io .mh-sub { flex: 1 1 auto; }
  .mh-file { position: relative; display: inline-flex; cursor: pointer; }
  .mh-file input { position: absolute; width: 1px; height: 1px; opacity: 0; }
  .mh-file span { border: 1px solid var(--line2); border-radius: 6px; background: var(--panel); }
  .mh-file:focus-within span { outline: 2px solid var(--accent); }
  .mh-backup { display: flex; flex-direction: column; gap: 6px; padding-top: 10px; border-top: 1px solid var(--line2); font-size: 13px; }
  .mh-moreb { align-self: center; font-size: 12px; margin-top: 6px; }
  @media (max-width: 600px) {
    .mh { padding: 0; }
    .mh-card { max-height: 100dvh; height: 100dvh; border-radius: 0; border: 0; width: 100%; }
    .mh-head .sub { display: none; }
    .mh-when { min-width: 0; }
    .mh-tools input[type=search], .mh-more textarea, .mh-paste textarea, .mh-tagrow input { font-size: 16px; }
  }`;
  function close() {
    if (box) box.hidden = true;
    if (noteT) { clearTimeout(noteT); noteT = null; const t = box && box.querySelector("textarea[data-note]"); if (t) upd(t.dataset.note, { note: t.value }); }
    const m = document.getElementById("menuBtn"); if (m) m.focus();
    if (globalThis.viewClosed) globalThis.viewClosed("hands");
  }
  async function open(h) {
    host = h || host || pageHost();
    if (!store()) throw new Error("this browser's storage didn't load (store.js)");
    try { const u = store().get(UI_KEY); if (u) ui = { q: "", kind: u.kind || "", tag: u.tag || "" }; } catch (e) {}
    if (!document.getElementById("mhStyle")) { const s = document.createElement("style"); s.id = "mhStyle"; s.textContent = CSS_TEXT; document.head.appendChild(s); }
    if (!box) {
      box = document.createElement("div"); box.id = "handsBox"; box.className = "mh"; box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true"); box.setAttribute("aria-label", "My hands");
      box.innerHTML = `<div class="mh-card"><div class="mh-head"><b>My hands</b><span class="sub">every hand you've entered, reviewed or imported, kept in this browser</span><button class="ghost mh-x" data-close aria-label="Close">✕</button></div><div class="mh-body"></div></div>`;
      document.body.appendChild(box);
      const keys = "[data-close],[data-clear],[data-kind],[data-tag],[data-more],[data-toggle],[data-savetable],[data-open],[data-study],[data-link],[data-json],[data-export],[data-del],[data-rmtag],[data-pastebox],[data-pastego],[data-backup],[data-restorego],[data-restorecancel],[data-reload],[data-undo]";
      box.addEventListener("click", (e) => {
        if (e.target === box) return close();
        const t = e.target.closest(keys);
        if (!t || t.disabled) return;
        if (t.dataset.close != null) return close();
        act(t.dataset, t);
      });
      box.addEventListener("input", (e) => {
        const d = e.target.dataset || {};
        if (d.q != null) { ui.q = e.target.value; shown = PAGE; clearTimeout(box._q); box._q = setTimeout(drawList, 120); }
        else if (d.note != null) saveNote(d.note, e.target.value);
      });
      // the page's own shortcuts stay off while the panel is open (keys in it stop here; any other key at the window)
      box.addEventListener("keydown", (e) => {
        const d = e.target.dataset || {};
        e.stopPropagation();
        if (e.key === "Escape") close();
        else if (d.addtag != null && (e.key === "Enter" || e.key === ",")) { e.preventDefault(); addTag(d.addtag, e.target); }
        else if (d.paste != null && e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); importText(e.target.value, ""); }
      });
      box.addEventListener("change", (e) => {
        const d = e.target.dataset || {}, f = e.target.files && e.target.files[0];
        if (!f) return;
        if (d.importfile != null) f.text().then(t => importText(t, f.name));
        else if (d.restorefile != null) f.text().then(t => {
          let obj = null; try { obj = JSON.parse(t); } catch (err) {}
          if (!obj || obj.format !== "gto-solver-backup") { if (obj && (obj.format === "gto-solver-hands" || Array.isArray(obj))) return importText(t, f.name); return say("That isn't a GTO Solver backup file."); }
          draft = obj; draw();
        });
        e.target.value = "";
      });
      window.addEventListener("keydown", (e) => { if (box.hidden || box.contains(e.target)) return; e.stopPropagation(); if (e.key === "Escape") close(); }, true);
      unsub = store().onChange((c) => { if (c.remote && c.stores.includes("hands") && !box.hidden) refresh(); });
    }
    box.hidden = false; msg = ""; draft = null; shown = PAGE; draw();
    await store().ready;
    await refresh();
    const q = box.querySelector("[data-q]"); if (q) q.focus();
  }

  const api = { summary, lineHtml, searchText, matches, filterRecs, tagCounts, cleanTag, rowHtml, listHtml, toolbarHtml, parseImport, fileName, backupText,
                lastDecision, pageHost, open, close, act, importText, get recs() { return recs; }, get ui() { return ui; }, PAGE };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.MyHands = api;
})();
