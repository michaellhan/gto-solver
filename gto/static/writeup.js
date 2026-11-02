// Write up a hand, loaded by index.html the first time its "Write up a hand…" opens: a hand you played, entered after
// the session on your phone, then reviewed. The game (stakes, players, stacks in dollars, straddle, rake) and your seat
// and cards, then the action seat by seat: whoever is to act folds, checks, calls or bets (an amount in dollars with
// stakes), a seat further on is skipped to (everyone before it folds, or checks), and each street's cards are typed
// when it's dealt. Review hand loads it on the table and grades each of your decisions (the page's reviewHand).
// The draft is a hand like the page's own (S), kept apart in the page's storage, "gto-writeup" (store.js; a reload keeps it), and the
// page's engine plays it (withHand: simulate, recordOf); the next write-up starts in the last one's game.
(function () {
  "use strict";
  const KEY = "gto-writeup";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const cap = (s) => s[0].toUpperCase() + s.slice(1);
  // the game a write-up keeps from the last one (or the page's hand, the first time): not its cards or actions, nor a
  // tournament's or the advice's settings
  const GAME = ["players", "hero", "stack", "stacks", "straddle", "rake", "rakeCap", "stakes"];
  const BOARD = { flop: [2, 3, 4], turn: [5], river: [6] };

  let W = null;            // the hand being written up
  let box = null, err = "", gameOpen = null, perSeat = false, stakesCustom = false, rakeCustom = false;

  // a new hand in game g's game (a state like the page's, cleaned by its cleanState)
  function blank(g) {
    g = g || {};
    const n = SEATS[g.players] ? g.players : 6, w = fresh(n);
    for (const k of GAME) if (g[k] !== undefined && g[k] !== null) w[k] = JSON.parse(JSON.stringify(g[k]));
    if (!g.stakes) delete w.stakes;
    return cleanState(w);
  }
  function restore() {
    let w = null;
    try { w = JSON.parse(getRaw(KEY) || "null"); } catch (e) {}
    try { return w ? cleanState(w) : null; } catch (e) { return null; }
  }
  // (the page's storage, store.js; localStorage itself without it)
  const KV = globalThis.GTOStore || null;
  const getRaw = (k) => { try { return KV ? KV.getRaw(k) : localStorage.getItem(k); } catch (e) { return null; } };
  const setRaw = (k, v) => { try { if (KV) KV.setRaw(k, v); else localStorage.setItem(k, v); } catch (e) {} };
  const store = () => setRaw(KEY, JSON.stringify(W));
  const hand = () => W || (W = restore() || blank(S));
  // the page's engine and helpers on the write-up's hand (they read S and the seats)
  const run = (f) => withHand(hand(), f);

  // cards typed as "AhKd", "Ah Kd", "ah kd", "10h", "A♥K♦": [cards], and whatever else was typed
  function parseCards(t) {
    const out = [], s = String(t || "").replace(/♠/g, "s").replace(/♥/g, "h").replace(/♦/g, "d").replace(/♣/g, "c");
    const rest = s.replace(/(10|[2-9tjqka])\s*([shdc])/gi, (_, r, su) => { out.push((r === "10" ? "T" : r.toUpperCase()) + su.toLowerCase()); return ""; });
    return { cards: out, junk: rest.replace(/[\s,.\-/]/g, "") };
  }
  const used = (except) => new Set(W.cards.filter((c, i) => c && !except.includes(i)));

  // ---- the game
  function setGame(patch) {
    hand(); err = "";
    const had = STREETS.some(s => W.actions[s].length);
    run(() => {
      if ("players" in patch && SEATS[patch.players] && patch.players !== W.players) {
        const cards = W.cards; W = blank({ ...W, players: patch.players, stacks: null }); W.cards = cards;
      }
      if ("hero" in patch && SEATS[W.players].includes(patch.hero)) W.hero = patch.hero;
      if ("stakes" in patch) {
        const k = cleanStakes(patch.stakes);
        // (10% capped at $5 stays capped at $5)
        const live = STAKES() && RAKE() === 10 && Math.abs(RAKE_CAP() - RAKE_LIVE / STAKES().bb) < 1e-9;
        if (k) W.stakes = k; else delete W.stakes;
        if (live && k) W.rakeCap = RAKE_LIVE / k.bb;
        setRaw("gto-stakes", JSON.stringify(k));   // (the page's new hands too)
      }
      if ("stack" in patch) { const v = amountIn(patch.stack); if (v > 0) W.stack = v; else err = "Type the stacks" + (W.stakes ? " in dollars." : " in big blinds."); }
      if ("seatStack" in patch) {
        const [p, x] = patch.seatStack, v = amountIn(x), o = { ...(W.stacks || {}) };
        if (v > 0 && v !== W.stack) o[p] = v; else delete o[p];
        W.stacks = o;
      }
      if ("straddle" in patch) W.straddle = !!patch.straddle;
      if ("rake" in patch) {
        const [pct, capBb] = patch.rake || [0, 0];
        W.rake = Math.min(25, Math.max(0, +pct || 0)); W.rakeCap = Math.max(0, +capBb || 0);
      }
      // a new straddle or table size starts the action over (who acts when changes)
      if (had && ("straddle" in patch || "players" in patch)) for (const s of STREETS) W.actions[s] = [];
      W = cleanState({ ...W, ...cleanStacks(W, SEATS[W.players]) });
      if (!W.stacks || !Object.keys(W.stacks).length) delete W.stacks;
      if (!(+W.rake > 0)) { delete W.rake; delete W.rakeCap; } else if (!(+W.rakeCap > 0)) delete W.rakeCap;
      if (!W.straddle) delete W.straddle;
    });
    store(); draw();
    return err || null;
  }
  // the rake's presets as the page has them ("10-live": 10% capped at $5, with stakes)
  const rakeMode = () => run(() => {
    if (!RAKE()) return rakeCustom ? "custom" : "off";
    const k = Object.keys(RAKE_PRESETS).find(x => { const r = RAKE_PRESETS[x](); return r && r[0] === RAKE() && Math.abs(r[1] - RAKE_CAP()) < 1e-9; });
    return k && !rakeCustom ? k : "custom";
  });
  function setRake(mode, pct, capIn) {
    rakeCustom = mode === "custom";
    if (mode === "off") return setGame({ rake: [0, 0] });
    if (mode === "custom") return setGame({ rake: [pct != null ? parseFloat(pct) : (run(RAKE) || 5), capIn != null ? run(() => amountIn(capIn)) || 0 : (run(RAKE_CAP) || run(() => STAKES() ? RAKE_LIVE / STAKES().bb : 3))] });
    const r = run(() => RAKE_PRESETS[mode] && RAKE_PRESETS[mode]());
    return r ? setGame({ rake: r }) : null;
  }
  function setStakes(v, sb, bb) {
    stakesCustom = v === "custom";
    if (v === "off") return setGame({ stakes: null });
    if (STAKES_PRESETS[v]) return setGame({ stakes: { sb: STAKES_PRESETS[v][0], bb: STAKES_PRESETS[v][1] } });
    const k = sb != null ? { sb: parseFloat(sb), bb: parseFloat(bb) } : (W && W.stakes) || { sb: 1, bb: 2 };
    if (!cleanStakes(k)) { err = "The blinds: a small blind no bigger than the big blind, e.g. 1 and 3."; draw(); return err; }
    return setGame({ stakes: k });
  }
  function setHole(text) {
    hand(); err = "";
    const { cards, junk } = parseCards(text);
    if (junk || cards.length !== 2) err = "Your two cards, rank then suit: AhKd, 7s7c, Td9d.";
    else if (cards[0] === cards[1]) err = "Two different cards.";
    else if (cards.some(c => used([0, 1]).has(c))) err = `${cards.find(c => used([0, 1]).has(c))} is on the board.`;
    else { W.cards[0] = cards[0]; W.cards[1] = cards[1]; }
    store(); draw();
    return err || null;
  }

  // ---- the action
  const sim = () => run(() => simulate());
  // whoever is to act does a ({action, to?}: to in big blinds; amount, a typed one: dollars with stakes)
  function record(a) {
    hand(); err = "";
    const e = run(() => {
      const s = simulate();
      const r = recordOf(s, a.amount !== undefined ? { action: "bet", to: amountIn(a.amount) } : a);
      if (typeof r === "string") return inDollars(r);
      S.actions[s.current].push({ pos: s.toAct, ...r });
      return null;
    });
    err = e || ""; store(); draw();
    return e;
  }
  // (the engine's messages give amounts in big blinds: with stakes, dollars)
  const inDollars = (m) => STAKES() ? m.replace(/(\d+(?:\.\d+)?)bb\b/g, (_, x) => DOLLARS(+x)) : m;
  // the seats the action can be skipped to: each one after whoever is to act this street, up to your turn (the seats
  // before it fold, or check where nobody has bet)
  function skipTargets() {
    return run(() => {
      const s0 = simulate();
      if (!s0.cur || s0.toAct === S.hero) return [];
      S = JSON.parse(JSON.stringify(S));
      const out = [];
      for (let i = 0; i < 10; i++) {
        const s = simulate();
        if (!s.cur || s.current !== s0.current) break;
        if (i) out.push(s.toAct);
        if (s.toAct === S.hero) break;
        const r = recordOf(s, { action: s.cur.curBet - s.cur.bets[s.toAct] > 1e-9 ? "fold" : "check" });
        if (typeof r === "string") break;
        S.actions[s.current].push({ pos: s.toAct, ...r });
      }
      return out;
    });
  }
  function skipTo(p) {
    hand(); err = "";
    if (!skipTargets().includes(p)) { err = `The action can't skip to ${p} from here.`; draw(); return err; }
    run(() => {
      for (let i = 0; i < 10; i++) {
        const s = simulate();
        if (!s.cur || s.toAct === p) break;
        const r = recordOf(s, { action: s.cur.curBet - s.cur.bets[s.toAct] > 1e-9 ? "fold" : "check" });
        if (typeof r === "string") break;
        S.actions[s.current].push({ pos: s.toAct, ...r });
      }
    });
    store(); draw();
    return null;
  }
  // the next street's cards ("Kd 7s 2c", one card on the turn and river)
  function deal(text) {
    hand(); err = "";
    const s = sim();
    if (!s.waiting) { err = s.handOver ? "The hand is over." : "Finish the betting first."; draw(); return err; }
    const need = BOARD[s.waiting].length, { cards, junk } = parseCards(text), on = used(BOARD[s.waiting]);
    if (junk || cards.length !== need) err = need === 3 ? "The flop's three cards: Kd 7s 2c." : `The ${s.waiting}'s card: Qh.`;
    else if (new Set(cards).size !== cards.length) err = "Each card once.";
    else if (cards.some(c => on.has(c))) err = `${cards.find(c => on.has(c))} is already ${W.cards.indexOf(cards.find(c => on.has(c))) < 2 ? "in your hand" : "on the board"}.`;
    else BOARD[s.waiting].forEach((i, k) => { W.cards[i] = cards[k]; });
    store(); draw();
    return err || null;
  }
  // the last thing entered: an action, or a street's cards
  function undo() {
    hand(); err = "";
    for (const st of ["river", "turn", "flop"]) {
      if (W.actions[st].length) { W.actions[st].pop(); store(); draw(); return true; }
      if (BOARD[st].some(i => W.cards[i])) { BOARD[st].forEach(i => { W.cards[i] = null; }); store(); draw(); return true; }
    }
    const ok = !!W.actions.preflop.pop();
    store(); draw(); return ok;
  }
  // a new hand in the same game (and seat)
  function restart() { W = blank(hand()); err = ""; gameOpen = null; store(); draw(); }

  const heroActed = () => STREETS.some(s => W.actions[s].some(a => a.pos === W.hero));
  function blockedWhy(review) {
    hand();
    if (!W.cards[0] || !W.cards[1]) return "Type your two cards first.";
    if (review && !heroActed()) return "Record the action up to your decisions first: there's nothing of yours to review yet.";
    return null;
  }
  // Review hand: the hand on the page's table, each of your decisions solved and graded; or just opened there
  async function review(onTable) {
    const why = blockedWhy(!onTable);
    if (why) { err = why; draw(); return false; }
    const st = JSON.parse(JSON.stringify(W));
    close();
    W = blank(st); err = ""; gameOpen = null; store();
    await loadHandState(st, "your write-up");
    if (!onTable) reviewHand();
    return true;
  }

  // ---- the HTML (all built with the hand in place of the page's: the page's helpers read it)
  const label = (p) => SEAT_LABEL(p, S.players);
  const moneyT = (x) => STAKES() ? DOLLARS(x) : fmt(x) + "bb";
  const money = (x) => STAKES() ? `${DOLLARS(x)} <span class="wu-bb">${fmt(x)}bb</span>` : `${fmt(x)}bb`;
  // what one action put in ("calls $15", "raises to $60", "all-in $500"), sm: the hand just before it
  function verbOf(a, sm) {
    const info = sm.cur, p = a.pos;
    const left = info ? STACK(p) - sm.invested[p] - info.bets[p] : 0;
    if (a.action === "fold") return "folds";
    if (a.action === "check") return "checks";
    if (a.action === "call") return `calls ${moneyT(info ? Math.min(info.curBet - info.bets[p], left) : 0)}`;
    if (a.action === "all-in") return `all-in ${moneyT(info ? info.bets[p] + left : 0)}`;
    return `${a.action === "bet" ? "bets" : "raises to"} ${moneyT(a.to)}`;
  }
  function gameText() {
    const k = STAKES();
    return [k ? STAKES_TEXT() + (Math.abs(2 * k.sb - k.bb) > 1e-9 ? ` (small blind played as ${usd(k.bb / 2)})` : "") : "big blinds", `${S.players}-handed`,
            `${k ? DOLLARS(S.stack) : fmt(S.stack) + "bb"} stacks${UNEQUAL() ? ` (${POS.filter(p => STACK(p) !== S.stack).map(p => `${label(p)} ${moneyT(STACK(p))}`).join(", ")})` : ""}`,
            ...(STRADDLER() ? [`${k ? DOLLARS(2) : "2bb"} straddle`] : []), RAKE() ? RAKE_TEXT() : "no rake"].join(" · ");
  }
  function gameHtml() {
    const k = STAKES(), sm = !k ? "off" : (Object.keys(STAKES_PRESETS).find(x => STAKES_PRESETS[x][0] === k.sb && STAKES_PRESETS[x][1] === k.bb && !stakesCustom) || "custom");
    const rm = rakeMode();
    const opts = (list, v) => list.map(([val, t]) => `<option value="${val}"${val === v ? " selected" : ""}>${esc(t)}</option>`).join("");
    const seats = Array.from({ length: 8 }, (_, i) => i + 2).map(n => `<button type="button" data-players="${n}" class="${n === S.players ? "active" : ""}">${n}</button>`).join("");
    const unit = k ? "$" : "bb";
    return `<div class="wu-game">
      <label class="wu-f">Stakes <select data-ctl="stakes">${opts([["off", "big blinds"], ...Object.keys(STAKES_PRESETS).map(x => [x, `$${STAKES_PRESETS[x][0]}/$${STAKES_PRESETS[x][1]}`]), ["custom", "custom $"]], sm)}</select></label>
      ${sm === "custom" ? `<span class="wu-f">$<input type="text" inputmode="decimal" data-ctl="sb" value="${k ? k.sb : ""}" aria-label="Small blind in dollars">/ $<input type="text" inputmode="decimal" data-ctl="bb" value="${k ? k.bb : ""}" aria-label="Big blind in dollars"></span>` : ""}
      ${k && Math.abs(2 * k.sb - k.bb) > 1e-9 ? `<span class="wu-note">small blind played as ${usd(k.bb / 2)}</span>` : ""}
      <div class="wu-f">Players <span class="segment">${seats}</span></div>
      <label class="wu-f">Stacks ${k ? "$" : ""}<input type="text" inputmode="decimal" data-ctl="stack" value="${amountOut(S.stack)}" aria-label="Stacks in ${k ? "dollars" : "big blinds"}">${k ? `<span class="wu-bb">${fmt(S.stack)}bb</span>` : "bb"}</label>
      <button type="button" class="link" data-perseat>${perSeat || UNEQUAL() ? "all equal" : "per seat…"}</button>
      ${perSeat || UNEQUAL() ? `<div class="wu-seatstacks">${POS.map(p => `<label class="wu-f">${esc(label(p))} ${k ? "$" : ""}<input type="text" inputmode="decimal" data-ctl="seatstack" data-p="${p}" value="${amountOut(STACK(p))}" aria-label="${p}'s stack in ${unit === "$" ? "dollars" : "big blinds"}"></label>`).join("")}</div>` : ""}
      ${S.players >= 3 ? `<label class="wu-f"><input type="checkbox" data-ctl="straddle"${S.straddle ? " checked" : ""}> ${k ? DOLLARS(2) : "2bb"} straddle (${esc(POS[0])})</label>` : ""}
      <label class="wu-f">Rake <select data-ctl="rake">${opts([["off", "off"], ["5-3", `5%, cap ${k ? DOLLARS(3) : "3bb"}`], ...(k ? [["10-live", `10%, cap ${usd(RAKE_LIVE)}`]] : []), ["custom", "custom"]], rm)}</select></label>
      ${rm === "custom" ? `<span class="wu-f"><input type="text" inputmode="decimal" data-ctl="rakepct" value="${RAKE() ? fmt(RAKE()) : ""}" aria-label="Rake, percent">%, cap ${k ? "$" : ""}<input type="text" inputmode="decimal" data-ctl="rakecap" value="${RAKE_CAP() ? amountOut(RAKE_CAP()) : ""}" aria-label="Rake cap">${k ? "" : "bb"}</span>` : ""}
    </div>`;
  }
  function youHtml() {
    const hole = S.cards[0] && S.cards[1] ? S.cards[0] + S.cards[1] : "";
    return `<div class="wu-you"><span class="wu-lab">You</span><span class="wu-seats">${POS.map(p => `<button type="button" data-hero="${p}" class="${p === S.hero ? "active" : ""}">${esc(label(p))}</button>`).join("")}</span>
      <label class="wu-f">cards <input type="text" data-ctl="hole" value="${hole}" placeholder="AhKd" autocomplete="off" autocapitalize="off" autocorrect="off" spellcheck="false" aria-label="Your two cards" class="wu-cards"></label>
      ${hole ? `<span class="wu-hole">${cardHtml(S.cards[0])}${cardHtml(S.cards[1])}</span>` : ""}</div>`;
  }
  function lineHtml() {
    const full = S, rows = [];
    STREETS.forEach((st, si) => {
      const acts = full.actions[st], cards = st === "preflop" ? [] : BOARD[st].map(i => full.cards[i]).filter(Boolean);
      if (!acts.length && !cards.length) return;
      const items = []; let pot = 0;
      for (let i = 0; i <= acts.length; i++) {
        S = { ...full, actions: Object.fromEntries(STREETS.map((s, k) => [s, k < si ? full.actions[s] : k === si ? acts.slice(0, i) : []])) };
        const sm = simulate();
        if (!i) pot = sm.pot;
        if (i === acts.length) break;
        const a = acts[i];
        items.push(`<span class="wu-a${a.pos === full.hero ? " me" : ""}${["bet", "raise", "all-in"].includes(a.action) ? " agg" : ""}">${esc(label(a.pos))} ${verbOf(a, sm)}</span>`);
      }
      S = full;
      rows.push(`<div class="wu-st"><span class="wu-sth"><b>${cap(st)}</b>${cards.length ? " " + cards.map(cardHtml).join(" ") : ""} <span class="wu-pot">pot ${moneyT(pot)}</span></span>${items.length ? `<span class="wu-acts">${items.join("")}</span>` : ""}</div>`);
    });
    S = full;
    return rows.length ? `<div class="wu-line">${rows.join("")}</div>` : "";
  }
  function nowHtml() {
    const s = simulate();
    if (s.handOver) return `<div class="wu-now"><div class="wu-who"><b>Hand over</b> · pot ${money(s.pot)}</div></div>`;
    if (s.waiting) {
      const n = BOARD[s.waiting].length;
      return `<form class="wu-now wu-deal" data-form="deal"><div class="wu-who"><b>Deal the ${s.waiting}</b> · pot ${money(s.pot)}${s.runout ? " · everyone's all-in" : ""}</div>
        <div class="wu-row"><input type="text" name="cards" data-ctl="board" placeholder="${n === 3 ? "Kd 7s 2c" : "Qh"}" autocomplete="off" autocapitalize="off" autocorrect="off" spellcheck="false" aria-label="The ${s.waiting}" class="wu-cards"><button type="submit">Deal</button></div></form>`;
    }
    const info = s.cur, p = s.toAct, me = p === S.hero;
    const toCall = info.curBet - info.bets[p], left = STACK(p) - s.invested[p] - info.bets[p], facing = toCall > 1e-9;
    const callAmt = Math.min(toCall, left), allTo = info.bets[p] + left, raiseOk = canRaise(s, p);
    const verb = facing || info.curBet > 0 ? "Raise to" : "Bet";
    const targets = skipTargets();
    const skip = targets.length ? `<div class="wu-skip"><span class="wu-lab">${facing || s.current === "preflop" ? "Folds to" : "Checks to"}</span>${targets.map(q => `<button type="button" data-skip="${q}" class="${q === S.hero ? "me" : ""}">${q === S.hero ? "you" : esc(label(q))}</button>`).join("")}</div>` : "";
    const btns = [
      ...(facing ? [`<button type="button" data-act="fold" class="fold">Fold</button>`, `<button type="button" data-act="call" class="pass">Call ${moneyT(callAmt)}</button>`]
                 : [`<button type="button" data-act="check" class="pass">Check</button>`]),
      ...(left > 0 && (raiseOk || left <= toCall + 1e-9) ? [`<button type="button" data-act="all-in" class="agg">All-in ${moneyT(allTo)}</button>`] : []),
    ].join("");
    const bet = left > toCall + 1e-9 && raiseOk ? `<form class="wu-row wu-bet" data-form="bet"><span class="wu-lab">${verb}</span>${STAKES() ? "$" : ""}<input type="text" inputmode="decimal" name="amt" data-ctl="amt" placeholder="${STAKES() ? DOLLARS(minRaiseTo(info)).slice(1) : fmt(minRaiseTo(info))}" autocomplete="off" aria-label="${verb} amount in ${STAKES() ? "dollars" : "big blinds"}">${STAKES() ? "" : "bb"}<button type="submit" class="agg">${verb}</button></form>` : "";
    const folded = !s.active.has(S.hero) ? `<div class="wu-note">You folded: review your decisions now, or carry on.</div>` : "";
    return `<div class="wu-now"><div class="wu-who"><b class="${me ? "me" : ""}">${me ? "You (" + esc(label(p)) + ")" : esc(label(p))}</b> to act on the ${s.current} · pot ${money(info.pot)}${facing ? ` · ${money(callAmt)} to call` : ""}</div>
      ${folded}${skip}<div class="wu-acts2">${btns}</div>${bet}</div>`;
  }
  function html() {
    return run(() => {
      const any = STREETS.some(s => S.actions[s].length), open = gameOpen === null ? !any : gameOpen;
      const rv = !blockedWhy(true), tb = !blockedWhy(false);
      return `<div class="wu-card" tabindex="-1">
        <div class="wu-head"><b>Write up a hand</b><button type="button" class="ghost wu-x" data-close aria-label="Close">×</button></div>
        <div class="wu-body">
          <button type="button" class="wu-sum" data-game aria-expanded="${open}"><span>${esc(gameText())}</span><span class="wu-edit">${open ? "done" : "edit"}</span></button>
          ${open ? gameHtml() : ""}
          ${youHtml()}
          ${lineHtml()}
          ${nowHtml()}
          <div class="wu-err" role="alert">${esc(err)}</div>
        </div>
        <div class="wu-foot"><button type="button" data-undo${any || S.cards.slice(2).some(Boolean) ? "" : " disabled"}>Undo</button><button type="button" class="ghost" data-restart>New hand</button>
          <span class="wu-sp"></span><button type="button" data-table${tb ? "" : " disabled"} title="Put this hand on the table, without reviewing it">Open on table</button>
          <button type="button" class="active" data-review${rv ? "" : " disabled"} title="${rv ? "Put this hand on the table and grade each of your decisions (the solver solves each in turn)" : esc(blockedWhy(true))}">Review hand</button></div>
      </div>`;
    });
  }
  // (not again from inside: replacing a box being typed in blurs it, and the browser's change event for it would set
  // the same thing and redraw under this one)
  let drawing = false;
  function draw() {
    if (!box || box.hidden || drawing) return;
    const f = document.activeElement, ctl = f && f.dataset && f.dataset.ctl;
    drawing = true;
    try { box.innerHTML = html(); } finally { drawing = false; }
    // (an amount or cards box being typed in keeps the focus)
    if (ctl && ["amt", "board"].includes(ctl) && box.querySelector) { const e = box.querySelector(`[data-ctl="${ctl}"]`); if (e && e.focus) e.focus(); }
  }

  const CSS = `
  .wu { position: fixed; inset: 0; z-index: 90; background: rgba(0,0,0,.55); display: flex; justify-content: center; align-items: flex-start; padding: 28px 16px 16px; }
  .wu-card { background: var(--panel2); color: var(--text); border: 1px solid var(--line2); border-radius: var(--r); width: min(640px, 100%); max-height: calc(100dvh - 44px); display: flex; flex-direction: column; box-shadow: 0 12px 40px rgba(0,0,0,.5); }
  .wu-card:focus { outline: none; }
  .wu-head { display: flex; align-items: center; gap: 10px; padding: 10px 12px 4px; }
  .wu-x { margin-left: auto; padding: 2px 10px; font-size: 18px; }
  .wu-body { display: flex; flex-direction: column; gap: 10px; padding: 4px 12px 10px; overflow: auto; min-height: 0; }
  .wu-sum { display: flex; gap: 8px; justify-content: space-between; text-align: left; font-size: 13px; background: var(--panel); }
  .wu-edit { color: var(--accent); white-space: nowrap; }
  .wu-game { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; font-size: 13px; color: var(--muted); }
  .wu-f { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
  .wu-f input[type=text] { width: 76px; }
  .wu-game .segment button { padding: 5px 9px; min-width: 32px; }
  .wu-seatstacks { display: flex; flex-wrap: wrap; gap: 6px 12px; flex-basis: 100%; }
  .wu-bb { color: var(--faint); font-size: 11px; }
  .wu-note { color: var(--muted); font-size: 12px; }
  .wu-you { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 13px; }
  .wu-lab { color: var(--muted); font-size: 12px; }
  .wu-seats { display: flex; flex-wrap: wrap; gap: 4px; }
  .wu-seats button, .wu-skip button { padding: 5px 9px; font-size: 12px; }
  .wu-cards { width: 110px; text-transform: none; letter-spacing: .5px; }
  .wu-hole { font-size: 18px; font-weight: 600; }
  .wu-line { display: flex; flex-direction: column; gap: 6px; font-size: 13px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .wu-st { display: flex; flex-direction: column; gap: 3px; }
  .wu-sth { display: flex; gap: 6px; align-items: baseline; }
  .wu-pot { color: var(--faint); font-size: 11px; margin-left: auto; }
  .wu-acts { display: flex; flex-wrap: wrap; gap: 4px; }
  .wu-a { background: var(--panel2); border: 1px solid var(--line); border-radius: 6px; padding: 2px 7px; font-size: 12px; }
  .wu-a.me { border-color: var(--accent); }
  .wu-a.agg { color: var(--agg0); }
  .wu-now { display: flex; flex-direction: column; gap: 8px; }
  .wu-who { font-size: 14px; }
  .wu-who b.me { color: var(--accent); }
  .wu-skip { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
  .wu-skip button.me { border-color: var(--accent); }
  .wu-acts2 { display: flex; flex-wrap: wrap; gap: 6px; }
  .wu-acts2 button { flex: 1 1 30%; min-height: 44px; font-weight: 600; }
  .wu-acts2 .fold { color: var(--fold); } .wu-acts2 .pass { color: var(--passive); } .wu-acts2 .agg, .wu-bet .agg { color: var(--agg1); }
  .wu-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .wu-bet input { width: 96px; }
  .wu-bet button { min-height: 44px; }
  .wu-err { color: var(--bad); font-size: 12px; min-height: 1em; }
  .wu-err:empty { display: none; }
  .wu-foot { display: flex; gap: 6px; align-items: center; padding: 8px 12px 12px; border-top: 1px solid var(--line); flex-wrap: wrap; }
  .wu-sp { flex: 1; }
  @media (max-width: 600px) {
    .wu { padding: 0; }
    .wu-card { border-radius: 0; border: 0; width: 100%; max-height: none; height: 100dvh; }
    .wu-foot button { min-height: 44px; }
    .wu-foot [data-review] { flex: 1 1 100%; order: 9; }
  }`;

  function open() {
    if (!document.getElementById("wuStyle")) { const s = document.createElement("style"); s.id = "wuStyle"; s.textContent = CSS; document.head.appendChild(s); }
    hand(); err = "";
    if (!box) {
      box = document.createElement("div"); box.id = "writeupBox"; box.className = "wu"; box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true"); box.setAttribute("aria-label", "Write up a hand");
      document.body.appendChild(box);
      const d = (t) => t && t.closest ? t.closest("[data-close],[data-game],[data-players],[data-perseat],[data-hero],[data-act],[data-skip],[data-undo],[data-restart],[data-table],[data-review]") : null;
      box.addEventListener("click", (e) => {
        if (e.target === box) return close();
        const t = d(e.target); if (!t || t.disabled) return;
        const ds = t.dataset;
        if ("close" in ds) close();
        else if ("game" in ds) { gameOpen = !(gameOpen === null ? !STREETS.some(s => W.actions[s].length) : gameOpen); draw(); }
        else if (ds.players) setGame({ players: +ds.players });
        else if ("perseat" in ds) { if (perSeat || run(UNEQUAL)) { perSeat = false; W.stacks = {}; setGame({}); } else { perSeat = true; draw(); } }
        else if (ds.hero) setGame({ hero: ds.hero });
        else if (ds.act) record({ action: ds.act });
        else if (ds.skip) skipTo(ds.skip);
        else if ("undo" in ds) undo();
        else if ("restart" in ds) restart();
        else if ("table" in ds) review(true);
        else if ("review" in ds) review(false);
      });
      box.addEventListener("submit", (e) => {
        e.preventDefault();
        const f = e.target, kind = f.dataset && f.dataset.form;
        if (kind === "bet") record({ amount: f.elements.amt.value });
        else if (kind === "deal") deal(f.elements.cards.value);
      });
      box.addEventListener("change", (e) => {
        const t = e.target, c = t.dataset && t.dataset.ctl, v = t.value;
        if (c === "stakes") setStakes(v);
        else if (c === "sb" || c === "bb") { const q = (k) => box.querySelector(`[data-ctl="${k}"]`).value; setStakes("custom", q("sb"), q("bb")); }
        else if (c === "stack") setGame({ stack: v });
        else if (c === "seatstack") setGame({ seatStack: [t.dataset.p, v] });
        else if (c === "straddle") setGame({ straddle: t.checked });
        else if (c === "rake") setRake(v);
        else if (c === "rakepct" || c === "rakecap") setRake("custom", box.querySelector('[data-ctl="rakepct"]').value, box.querySelector('[data-ctl="rakecap"]').value);
        else if (c === "hole") setHole(v);
      });
      // (two cards typed are taken at once, as are a street's)
      box.addEventListener("input", (e) => {
        const t = e.target, c = t.dataset && t.dataset.ctl;
        if (c === "hole") { const r = parseCards(t.value); if (!r.junk && r.cards.length === 2 && r.cards[0] !== r.cards[1]) setHole(t.value); }
        else if (c === "board") { const s = sim(), r = parseCards(t.value); if (s.waiting && !r.junk && r.cards.length === BOARD[s.waiting].length) deal(t.value); }
      });
      // its keys aren't the page's shortcuts (F folds the table's hand, N starts a new one)
      box.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Escape") close(); });
    }
    box.hidden = false; draw();
    const c = box.querySelector && box.querySelector(".wu-card"); if (c && c.focus) c.focus();
  }
  function close() { if (box) box.hidden = true; }

  const api = { open, close, state: () => hand(), blank, parseCards, setGame, setStakes, setRake, setHole, record, skipTargets, skipTo,
                deal, undo, restart, review, html, gameText: () => run(gameText), error: () => err, KEY };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.WriteUp = api;
})();
