// Hand-history import, loaded by index.html the first time it's used. A pasted PokerStars or GGPoker/Natural8
// hand history (other sites' "Seat N: name (stack)" / "name: raises 2 to 3" text mostly parses too) becomes the
// page's hand state. parseHandHistory is pure (the Node tests call it directly); importHand replays the actions
// through the page's own engine (simulate), so what it returns is a hand the page can play and review.
(function () {
  "use strict";
  const SEATS = {   // as the page's: preflop acting order, the blinds last
    2: ["SB", "BB"], 3: ["BTN", "SB", "BB"], 4: ["CO", "BTN", "SB", "BB"], 5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"], 7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"], 9: ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"] };
  const STREETS = ["preflop", "flop", "turn", "river"];
  const AMT = String.raw`[^\d\s]*([\d,]*\.?\d+)`;   // "$1,234.50", "€0.5", "1500"
  const num = (s) => parseFloat(String(s).replace(/,/g, ""));
  const r2 = (x) => Math.round(x * 100) / 100;
  const card = (c) => { const m = String(c).match(/^(10|[2-9TJQKA])([shdc])$/i); return m ? (m[1] === "10" ? "T" : m[1].toUpperCase()) + m[2].toLowerCase() : null; };
  const cardsIn = (l) => (l.match(/\[[^\]]*\]/g) || []).flatMap(b => b.slice(1, -1).trim().split(/\s+/)).map(card);

  function parseHandHistory(text) {
    let lines = String(text || "").replace(/\r/g, "").split("\n").map(l => l.trim()).filter(Boolean);
    const warnings = [];
    const starts = lines.map((l, i) => /^(PokerStars\b.*Hand #|Poker Hand #)/i.test(l) ? i : -1).filter(i => i >= 0);
    if (starts.length > 1) { lines = lines.slice(starts[0], starts[1]); warnings.push(`${starts.length} hands pasted: imported the first.`); }
    const head = lines.find(l => /Hold'?em|Omaha|Hand #/i.test(l)) || lines[0] || "";
    // a mixed game names this hand's game in brackets: "Mixed NLH/PLO (Hold'em No Limit, $0.10/$0.25)"
    const game = (head.match(/\bMixed\b[^(]*\(([^)]*)\)/i) || [, head])[1];
    if (/Omaha|PLO|Stud|Razz|Badugi|Courchevel|6\+|Short ?Deck/i.test(game) || /\bDraw\b/i.test(game) && !/Hold'?em/i.test(game))
      throw new Error("Only Hold'em hands can be imported.");
    if (/\bLimit\b/i.test(game) && !/No ?Limit/i.test(game)) warnings.push("Not a no-limit hand: it's solved as no-limit.");
    const site = /PokerStars/i.test(head) ? "PokerStars" : /^Poker Hand #/i.test(head) ? "GGPoker" : "generic";

    const seats = [];
    for (const l of lines) {
      if (/^\*\*\*/.test(l)) break;   // seat lines in the summary say who won, not their stacks
      // the stack is the last "(…)" opening with an amount, so a name with brackets ("x (2)") keeps them
      const m = l.match(new RegExp(String.raw`^Seat (\d+): (.+) \(\s*${AMT}(?: in chips)?[^)]*\)`, "i"));
      if (m && !seats.some(s => s.seat === +m[1])) seats.push({ seat: +m[1], name: m[2].trim(), chips: num(m[3]) });
    }
    if (seats.length < 2) throw new Error("No players found. Paste a whole hand history, from its first line (\"PokerStars Hand #…\" or \"Poker Hand #…\") down; it needs the \"Seat 1: name ($100 in chips)\" lines.");
    const names = seats.map(s => s.name).sort((a, b) => b.length - a.length);
    const who = (l) => names.find(n => l.startsWith(n + ":") || l.startsWith(n + " "));

    let street = null, hero = null, hole = null, board = [];
    const posts = [], acts = [], dealt = new Set();
    for (const l of lines) {
      let m;
      if ((m = l.match(/^\*\*\*\s*(.*?)\s*\*\*\*/))) {
        const t = m[1].toUpperCase();
        // run it twice: the first board is played out, the second ignored
        street = /HOLE|PRE-?FLOP/.test(t) ? "preflop" : /SECOND|2ND/.test(t) ? "skip" : /FLOP/.test(t) ? "flop" : /TURN/.test(t) ? "turn" : /RIVER/.test(t) ? "river" : "done";
        if (street === "flop" || street === "turn" || street === "river") board = cardsIn(l);
        continue;
      }
      if ((m = l.match(/^Dealt to (.+?)\s*(\[.*\])?$/i))) {
        dealt.add(m[1]);
        const cs = m[2] ? cardsIn(m[2]) : [];
        if (cs.length > 2) throw new Error("Only Hold'em hands can be imported.");
        // GGPoker's own exports can show every player's cards; its hero is always "Hero"
        if (cs.length === 2 && cs.every(Boolean) && (!hole || m[1] === "Hero" && hero !== "Hero")) { hero = m[1]; hole = cs; }
        continue;
      }
      if (!board.length && (m = l.match(/^Board\s*(\[.*\])/i))) { board = cardsIn(m[1]); continue; }
      const name = who(l);
      if (!name || street === "done" || street === "skip") continue;
      const rest = l.slice(name.length).replace(/^:?\s*/, ""), allIn = /all[- ]in/i.test(rest);
      if ((m = rest.match(new RegExp(String.raw`^posts (?:the )?(ante|small & big blinds?|small blind|big blind|(?:a )?straddle)\s+${AMT}`, "i")))) {
        const k = m[1].toLowerCase();
        posts.push({ name, kind: k === "ante" ? "ante" : k.startsWith("small &") ? "dead" : k === "small blind" ? "sb" : k === "big blind" ? "bb" : "straddle", amt: num(m[2]) });
      } else if (!street) continue;
      else if (/^folds/i.test(rest)) acts.push({ street, name, act: "fold" });
      else if (/^checks/i.test(rest)) acts.push({ street, name, act: "check" });
      else if ((m = rest.match(new RegExp(String.raw`^calls\s+${AMT}`, "i")))) acts.push({ street, name, act: "call", amt: num(m[1]), allIn });
      else if ((m = rest.match(new RegExp(String.raw`^(?:bets|(?:goes |is )?all[- ]in)\s+${AMT}`, "i")))) acts.push({ street, name, act: "bet", amt: num(m[1]), allIn });
      else if ((m = rest.match(new RegExp(String.raw`^raises\s+(?:${AMT}\s+)?to\s+${AMT}`, "i")))) acts.push({ street, name, act: "raise", to: num(m[2]), allIn });
    }
    if (!hole) throw new Error("Your hole cards aren't in this history (no \"Dealt to you [Ah Kd]\" line). Import a hand you played, as your poker client saved it.");
    if (!seats.some(s => s.name === hero)) throw new Error(`"${hero}" has hole cards but no "Seat N: ${hero} (…)" line: paste the whole hand history.`);
    if (!board.every(Boolean)) board = [];

    // who was dealt in: anyone who posted, acted or (GG) has a "Dealt to" line; sitting-out seats did none of these
    const inHand = new Set([...posts.map(p => p.name), ...acts.map(a => a.name), ...dealt, hero]);
    const order = seats.filter(s => inHand.has(s.name)).sort((a, b) => a.seat - b.seat), n = order.length;
    if (n < 2 || n > 9) throw new Error(`${n} players in this hand: the page plays 2 to 9.`);
    const bb = (() => {
      const m = head.match(new RegExp(String.raw`\(\s*${AMT}\s*\/\s*${AMT}`));
      const p = posts.find(x => x.kind === "bb");
      return m && num(m[2]) > 0 ? num(m[2]) : p ? p.amt : 0;
    })();
    if (!(bb > 0)) throw new Error("Couldn't find the big blind (neither in the first line's \"(0.50/1)\" nor a \"posts big blind\" line).");
    const sbPost = posts.find(p => p.kind === "sb");
    if (sbPost && Math.abs(sbPost.amt / bb - 0.5) > 0.01) warnings.push(`The small blind is ${r2(sbPost.amt / bb)}bb: the page plays a half-big-blind small blind.`);

    // seats: clockwise from the button, the big blind is the first player after it who posted one (a new player
    // can post a big blind elsewhere); the page's names run from the first seat after the big blind round to it
    const bm = lines.join("\n").match(/Seat #(\d+) is the button/i);
    const bbPost = new Set(posts.filter(p => p.kind === "bb").map(p => p.name)), toBB = n === 2 ? 1 : 2;
    let bi, bbi;
    if (bm) {
      bi = order.findIndex(s => s.seat >= +bm[1]); if (bi < 0) bi = 0;
      if (order[bi].seat !== +bm[1]) bi = (bi + n - 1) % n;   // the button seat is empty or sat out: the last one before it
      bbi = (bi + toBB) % n;
      for (let k = 1; k <= n; k++) if (bbPost.has(order[(bi + k) % n].name)) { bbi = (bi + k) % n; break; }
    } else {   // no button line (a partial paste): the big blind's poster, else the seat after the small blind's
      const sb = order.findIndex(s => posts.some(p => p.kind === "sb" && p.name === s.name));
      bbi = order.findIndex(s => bbPost.has(s.name));
      if (bbi < 0) bbi = sb >= 0 ? (sb + 1) % n : toBB % n;
      bi = (bbi - toBB + n) % n;
    }
    const pos = {};
    for (let k = 0; k < n; k++) pos[order[(bbi + 1 + k) % n].name] = SEATS[n][k];
    if (n > 2 && pos[order[bi].name] !== "BTN") warnings.push("The button isn't next to the blinds (a dead button?): positions are counted from the big blind.");

    // amounts as each player's total on the street ("to"), in big blinds; all-in also when the chips run out
    const left = Object.fromEntries(order.map(s => [s.name, s.chips])), anteOf = {};
    let put = {}, antes = 0, straddle = false;
    for (const p of posts) {
      if (!pos[p.name]) continue;
      if (p.kind === "ante") { antes += p.amt; left[p.name] -= p.amt; anteOf[p.name] = (anteOf[p.name] || 0) + p.amt; continue; }
      const live = p.kind === "dead" ? bb : p.amt;
      left[p.name] -= p.amt; put[p.name] = Math.max(put[p.name] || 0, live);
      if (p.kind === "straddle") {
        if (n >= 3 && pos[p.name] === SEATS[n][0] && Math.abs(p.amt / bb - 2) < 0.01) straddle = true;
        else warnings.push(`${pos[p.name]}'s straddle isn't a 2bb straddle from the first seat, the only kind the page plays: it's left out.`);
      } else if (p.kind === "dead" || p.kind === "bb" && pos[p.name] !== "BB") warnings.push(`${pos[p.name]} posted a blind out of turn: the page leaves it out.`);
    }
    let cur = null;
    const actions = [];
    for (const a of acts) {
      if (!pos[a.name]) continue;
      if (a.street !== cur) { if (cur) put = {}; cur = a.street; }
      const before = put[a.name] || 0;
      const to = a.act === "raise" ? a.to : a.act === "fold" || a.act === "check" ? before : before + a.amt;
      left[a.name] -= to - before; put[a.name] = to;
      actions.push({ street: a.street, pos: pos[a.name], act: a.act, ...(a.act === "fold" || a.act === "check" ? {} : { to: r2(to / bb) }), ...(a.allIn || left[a.name] <= 1e-9 && to > before ? { allIn: true } : {}) });
    }
    // every seat plays its own stack. The page takes the same ante from everyone (the average), so a seat's stack
    // is set for what it has behind after the antes to be the history's (a big-blind ante paid by one player).
    const ante = Math.round(antes / n / bb * 1e4) / 1e4;
    const stacks = Object.fromEntries(order.map(s => [pos[s.name], r2((s.chips - (anteOf[s.name] || 0)) / bb + ante)]));
    // (effective: yours against the deepest opponent who didn't fold, for the summary)
    const opp = order.filter(s => s.name !== hero), folded = new Set(acts.filter(a => a.act === "fold").map(a => a.name));
    const deep = [opp.filter(s => !folded.has(s.name)), opp].find(x => x.length);
    const effective = Math.min(stacks[pos[hero]], Math.max(...deep.map(s => stacks[pos[s.name]])));
    return { site, bb, players: n, hero: pos[hero], heroName: hero, stack: stacks[pos[hero]], stacks, effective, ante, straddle,
             hole, board: board.slice(0, 5), seats: order.map(s => ({ seat: s.seat, name: s.name, pos: pos[s.name], stack: r2(s.chips / bb) })), actions, warnings };
  }

  // The page state for a parsed hand: every seat with its own stack, and the actions through the page's engine one
  // by one, as if entered at the table (a short all-in, its side pot and a bet nobody could call in full play out
  // there as at the table). If the two disagree the import stops there.
  function importHand(text) {
    const h = parseHandHistory(text), warnings = [...h.warnings];
    const st = { players: h.players, hero: h.hero, stack: h.stack, stacks: { ...h.stacks },
                 cards: [...h.hole, ...h.board, null, null, null, null, null].slice(0, 7),
                 actions: { preflop: [], flop: [], turn: [], river: [] }, ...(h.ante ? { ante: h.ante } : {}), ...(h.straddle ? { straddle: true } : {}),
                 ...(+S.rake > 0 ? { rake: S.rake, ...(+S.rakeCap > 0 ? { rakeCap: S.rakeCap } : {}) } : {}) };
    const old = S;
    S = st; setPositions(st.players);
    try {
      for (const a of h.actions) {
        const sim = simulate();
        if (!sim.cur || sim.current !== a.street || sim.toAct !== a.pos) {
          warnings.push(`The page's betting and the history part ways at ${a.pos}'s ${a.street} action: the hand stops there.`); break;
        }
        const info = sim.cur, toCall = info.curBet - info.bets[a.pos];
        // an all-in is the stack: one the page has a little off the site's (rounding, a blind it leaves out) is set to it
        if (a.allIn && a.to > 0) st.stacks[a.pos] = r2(sim.invested[a.pos] + a.to);
        const cap = STACK(a.pos) - sim.invested[a.pos], passive = { action: toCall > 1e-9 ? "call" : "check" };
        const rec = a.act === "fold" ? { action: "fold" } : a.act === "check" || a.act === "call" ? passive
          : a.to >= cap - 0.005 ? { action: "all-in" } : a.to <= info.curBet + 1e-9 ? passive : { action: info.curBet > 0 ? "raise" : "bet", to: a.to };
        S.actions[a.street].push({ pos: a.pos, ...rec });
      }
    } finally { S = old; setPositions(old.players); }
    st.stack = st.stacks[h.hero];
    const own = Object.fromEntries(Object.entries(st.stacks).filter(([, v]) => v !== st.stack));
    if (Object.keys(own).length) st.stacks = own; else delete st.stacks;
    return { state: st, hand: h, warnings };
  }

  const api = { parseHandHistory, importHand, HH_SEATS: SEATS };
  if (typeof module === "object" && module.exports) module.exports = api; else Object.assign(globalThis, api);
})();
