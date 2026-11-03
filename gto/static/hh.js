// Hand-history import, loaded by index.html the first time it's used. A pasted hand history from PokerStars,
// GGPoker/Natural8, 888poker, partypoker, Winamax, WPN/ACR, Ignition/Bovada or iPoker (XML) becomes the page's hand
// state: the other sites' text is rewritten into PokerStars-like lines first (norm), which one parser reads.
// parseHandHistory is pure (the Node tests call it directly); importHand replays the actions through
// the page's own engine (simulate), so what it returns is a hand the page can play and review.
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
  const amts = (s) => (String(s).match(/\d[\d,]*(?:\.\d+)?|\.\d+/g) || []).map(num);   // every amount in "raises [$1.50 USD] to $2"
  const card = (c) => { const m = String(c).match(/^(10|[2-9TJQKA])([shdc])$/i); return m ? (m[1] === "10" ? "T" : m[1].toUpperCase()) + m[2].toLowerCase() : null; };
  const cardsIn = (l) => (l.match(/\[[^\]]*\]/g) || []).flatMap(b => b.slice(1, -1).split(/[\s,]+/).filter(Boolean)).map(card);   // "[Ah Qd]", "[ Ah, Qd ]"
  const STREET = (t) => /HOLE|PRE-?FLOP|DOWN CARDS/.test(t) ? "preflop" : /SECOND|2ND|\b2$/.test(t) ? "skip" : /FLOP/.test(t) ? "flop"
    : /TURN/.test(t) ? "turn" : /RIVER/.test(t) ? "river" : /SHOW ?DOWN|SUMMARY|RUN IT/.test(t) ? "done" : null;
  const header = (l) => { const m = l.match(/^\*{2,}\s*(.*?)\s*\*{2,}/); return m && STREET(m[1].toUpperCase()); };
  const START = /^(PokerStars\b.*Hand #|Poker Hand #|Winamax Poker\b|(Game )?Hand #\d|(Ignition|Bovada|Bodog)\b.*Hand #|#Game No\b|\*{5}.*Hand History for Game|Game #\d+ starts)/i;
  const SITES = [["PokerStars", /PokerStars/i], ["GGPoker", /^Poker Hand #/m], ["Winamax", /^Winamax/m], ["888poker", /888|Pacific/i],
    ["partypoker", /Hand History for Game|^Game #\d+ starts/im], ["Ignition/Bovada", /^(Ignition|Bovada|Bodog)\b/m], ["WPN/ACR", /^(Game )?Hand #\d/m]];

  // iPoker's XML (Bet365, William Hill, Unibet's old client…): the first <game>, as text. Cards are suit first ("HA",
  // "C10"); action types: 0 fold, 1/2 blinds, 3 call, 4 check, 5 bet, 6 raise by, 7 all-in (chips added), 15 ante, 23 raise to.
  function ipoker(x, warnings) {
    const ent = (s) => s.replace(/&(amp|lt|gt|quot|apos);/g, (_, k) => ({ amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" })[k]);
    const at = (s, k) => ent((s.match(new RegExp(String.raw`\b${k}="([^"]*)"`)) || [])[1] || "");
    const tag = (s, k) => ent((s.match(new RegExp(`<${k}>([^<]*)</${k}>`)) || [])[1] || "");
    const money = (s) => { s = s.replace(/[^\d.,]/g, ""); return /,\d\d?$/.test(s) ? s.replace(/\./g, "").replace(",", ".") : s.replace(/,/g, ""); };
    const cards = (s) => s.trim().split(/\s+/).map(c => c.slice(1) + c[0]);
    const games = x.match(/<game\b[\s\S]*?<\/game>/g) || [], hint = tag(x, "nickname"), bb = tag(x, "bigblind");
    const g = games.find(g => g.includes(`name="${hint}"`)) || games[0] || x;   // the first hand you were dealt into
    if (games.length > 1) warnings.push(`${games.length} hands pasted: imported the first.`);
    const tc = tag(x, "tournamentcode"), tn = tag(x, "tournamentname");
    const out = [`iPoker Hand #${at(g, "gamecode")}: ${tag(x, "gametype")}${bb ? ` (${money(tag(x, "smallblind"))}/${money(bb)})` : ""}${tc ? ` Tournament #${tc}${tn ? ` "${tn.replace(/"/g, "'")}"` : ""}` : ""}`];
    for (const p of g.match(/<player\b[^>]*>/g) || []) {
      out.push(`Seat ${at(p, "seat")}: ${at(p, "name")} (${money(at(p, "chips"))})`);
      if (at(p, "dealer") === "1") out.push(`Seat #${at(p, "seat")} is the button`);
    }
    const board = [], HEAD = { 1: "HOLE CARDS", 2: "FLOP", 3: "TURN", 4: "RIVER" };
    const ACT = { 0: "folds", 1: "posts small blind ", 2: "posts big blind ", 3: "calls ", 4: "checks", 5: "bets ", 6: "raises ", 7: "is all-in ", 15: "posts ante ", 23: "raises to " };
    for (const [, no, body] of g.matchAll(/<round\s+no="(\d+)"[^>]*>([\s\S]*?)<\/round>/g)) {
      const items = [...body.matchAll(/<cards\b([^>]*)>([^<]*)<\/cards>|<action\b([^>]*)>/g)];
      for (const [, a, cs] of items) if (a && !at(a, "player")) board.push(...cards(cs));
      if (HEAD[no]) out.push(`*** ${HEAD[no]} *** ${+no > 1 ? `[${board.join(" ")}]` : ""}`);
      for (const [, a, cs, act] of items) {
        const t = act && ACT[at(act, "type")];
        if (a && at(a, "player")) out.push(`Dealt to ${at(a, "player")}${/^[\sX]*$/i.test(cs) ? "" : ` [${cards(cs).join(" ")}]`}`);
        else if (t) out.push(`${at(act, "player")}: ${t}${/ $/.test(t) ? money(at(act, "sum")) : ""}`);
      }
    }
    return { lines: out, site: "iPoker", hint };
  }

  // A history's lines (trimmed, no blank ones) as one array per hand: a hand starts at a site's first header line
  // (a header within 3 lines of another is the same hand's: 888's "#Game No" then its "***** … Hand History"),
  // anything before the first one stays with it
  function handLines(text) {
    const lines = String(text || "").replace(/\r/g, "").split("\n").map(l => l.trim()).filter(Boolean);
    const starts = lines.map((l, i) => START.test(l) ? i : -1).filter((i, k, a) => i >= 0 && !a.slice(Math.max(0, k - 3), k).some(j => j >= 0));
    if (starts.length < 2) return [lines];
    return starts.map((s, k) => lines.slice(k ? s : 0, starts[k + 1]));
  }
  // A file of many hands as one text per hand, each as parseHandHistory takes it: iPoker's XML a <session> per
  // <game> with the <general> before it (the stakes, your nickname), the other sites' text cut at each hand's header
  function splitHands(text) {
    text = String(text || "");
    if (/<game\b/i.test(text)) {
      const out = [];
      let gen = "";
      for (const [m] of text.matchAll(/<general\b[\s\S]*?<\/general>|<game\b[\s\S]*?<\/game>/g))
        if (m.startsWith("<general")) gen = m; else out.push(`<session>${gen}${m}</session>`);
      return out;
    }
    return handLines(text).map(ls => ls.join("\n")).filter(Boolean);
  }

  // the other sites' lines as PokerStars-like ones, and the hand's site
  function norm(text, warnings) {
    if (/<game\b/i.test(text)) return ipoker(String(text), warnings);
    const hands = handLines(text);
    let lines = hands[0];
    if (hands.length > 1) { lines = lines.slice(lines.findIndex(l => START.test(l))); warnings.push(`${hands.length} hands pasted: imported the first.`); }
    const pre = lines.slice(0, 3).join("\n"), site = (SITES.find(([, re]) => re.test(pre)) || ["generic"])[0], all = lines.join(" ");
    // dots for thousands (888's "1.183.764 $"), when no amount has cents
    if (/\d\.\d{3}(?!\d)/.test(all) && !/\d\.\d\d?(?!\d)/.test(all)) lines = lines.map(l => l.replace(/(\d)\.(?=\d{3}(?!\d))/g, "$1"));
    // Ignition/Bovada name players by position ("UTG+1 [ME] : Raises $2 to $2"); its seat numbers aren't the table's order, the listing is
    let k = 0;
    if (site === "Ignition/Bovada") lines = lines.map(l => /^Seat \d+:/i.test(l) ? l.replace(/^Seat \d+/i, () => `Seat ${++k}`)
      : l.replace(/^(.+?)\s+:\s+Card dealt to a spot (\[.*\])/i, "Dealt to $1 $2")
         .replace(/^(.+?)\s+:\s+(Small Blind|Big Blind|Ante chip)\b/i, (_, n, b) => `${n}: posts ${/ante/i.test(b) ? "ante" : b.toLowerCase()}`)
         .replace(/All-in\(raise[^)]*\)\s*(.*)$/i, "raises $1 and is all-in"));
    return { lines, site, hint: "" };
  }

  // a tournament's hand (the lines before the seats): its id (the site's number; Winamax's from the table's name), name,
  // buy-in and level, or null for a cash game
  const ROMAN = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };
  const roman = (s) => [...s.toUpperCase()].reduceRight((o, c) => { const v = ROMAN[c]; return { n: o.n + (v < o.top ? -v : v), top: Math.max(o.top, v) }; }, { n: 0, top: 0 }).n;
  function tourney(lines) {
    const pre = lines.slice(0, Math.max(1, lines.findIndex(l => /^Seat \d+:/i.test(l)))).join("\n");
    if (!/Tournament|Tourney|\bMTT\b|\bSNG\b|Sit ?(&|n|'n'|and) ?Go|Spin ?(&|and) ?Go/i.test(pre)) return null;
    const name = (pre.match(/Tournament\s*(?:#[\w-]+\s*)?"([^"]+)"/i) || [])[1];
    const id = (pre.match(/Tournament\s*(?:#|ID\s*:?)\s*([\w-]*\d[\w-]*)/i) || pre.match(/^Table:?\s*'[^'(]*\((\d+)\)/im) || [])[1] || name || null;
    const buyin = (pre.match(/(?:[$€£]\s?)?\d+(?:\.\d+)?\s?[€£]?(?:\s?\+\s?(?:[$€£]\s?)?\d+(?:\.\d+)?\s?[€£]?){1,2}/) || [])[0];
    const lv = (pre.match(/\bLevel\s*:?\s*([IVXLCDM]+|\d+)(?![\w.])/i) || [])[1];
    return { id, ...(name ? { name } : {}), ...(buyin ? { buyin: buyin.replace(/\s+/g, "") } : {}), ...(lv ? { level: /^\d+$/.test(lv) ? +lv : roman(lv) } : {}) };
  }

  function parseHandHistory(text) {
    const warnings = [], { lines, site, hint } = norm(text, warnings);
    const head = lines.find(l => /Hold'?em|Omaha|Hand #/i.test(l)) || lines[0] || "";
    // a mixed game names this hand's game in brackets: "Mixed NLH/PLO (Hold'em No Limit, $0.10/$0.25)"
    const game = (head.match(/\bMixed\b[^(]*\(([^)]*)\)/i) || [, head])[1];
    if (/Omaha|PLO|Stud|Razz|Badugi|Courchevel|6\+|Short ?Deck/i.test(game) || /\bDraw\b/i.test(game) && !/Hold'?em/i.test(game))
      throw new Error("Only Hold'em hands can be imported.");
    if (/\bLimit\b/i.test(game) && !/No ?Limit/i.test(game)) warnings.push("Not a no-limit hand: it's solved as no-limit.");

    const seats = [];
    for (const l of lines) {
      if (header(l)) break;   // seat lines in the summary say who won, not their stacks
      // the stack is the last "(…)" opening with an amount, so a name with brackets ("x (2)") keeps them
      const m = l.match(new RegExp(String.raw`^Seat (\d+): (.+) \(\s*${AMT}(?: in chips)?[^)]*\)`, "i"));
      if (m && !seats.some(s => s.seat === +m[1])) seats.push({ seat: +m[1], name: m[2].trim(), chips: num(m[3]) });
    }
    if (seats.length < 2) throw new Error("No players found. Paste a whole hand history, from its first line (\"PokerStars Hand #…\", \"Poker Hand #…\"…) down; it needs the \"Seat 1: name ($100)\" lines.");
    const names = seats.map(s => s.name).sort((a, b) => b.length - a.length);
    const who = (l) => names.find(n => l.startsWith(n + ":") || l.startsWith(n + " "));

    let street = null, hero = null, hole = null, board = [];
    const posts = [], acts = [], dealt = new Set();
    for (const l of lines) {
      let m;
      if (/^\*{2,}/.test(l)) {
        // run it twice: the first board is played out, the second ignored. 888 and party show only the new card.
        street = header(l) || (street ? "done" : null);
        const k = { flop: 3, turn: 4, river: 5 }[street], cs = cardsIn(l);
        if (k) board = cs.length >= k ? cs : [...board.slice(0, k - cs.length), ...cs];
        continue;
      }
      if ((m = l.match(/^Dealt to (.+?)(?:\s*(\[[^\]]*\]))?$/i))) {
        dealt.add(m[1]);
        const cs = m[2] ? cardsIn(m[2]) : [];
        if (cs.length > 2) throw new Error("Only Hold'em hands can be imported.");
        // GGPoker's own exports can show every player's cards, its hero is "Hero"; Ignition's is "… [ME]"
        const me = (n) => n === hint || n === "Hero" || /\[ME\]$/.test(n);
        if (cs.length === 2 && cs.every(Boolean) && (!hole || me(m[1]) && !me(hero))) { hero = m[1]; hole = cs; }
        continue;
      }
      if (!board.length && (m = l.match(/^Board:?\s*(\[.*\])/i))) { board = cardsIn(m[1]); continue; }
      const name = who(l);
      if (!name || street === "done" || street === "skip") continue;
      // amounts are what the action adds ("calls 2", 888/party/iPoker's "raises [$2]"), except a raise "to"
      const rest = l.slice(name.length).replace(/^\s*:?\s*/, ""), allIn = /all[- ]in/i.test(rest), a = amts(rest);
      if ((m = rest.match(/^posts (?:the )?(ante|small & big blinds?|small blind|big blind|(?:a )?straddle)\b(.*)/i))) {
        const k = m[1].toLowerCase(), amt = amts(m[2])[0];
        if (amt != null) posts.push({ name, kind: k === "ante" ? "ante" : k.startsWith("small &") ? "dead" : k === "small blind" ? "sb" : k === "big blind" ? "bb" : "straddle", amt });
      } else if (!street) continue;
      else if (/^folds/i.test(rest)) acts.push({ street, name, act: "fold" });
      else if (/^checks/i.test(rest)) acts.push({ street, name, act: "check" });
      else if (!a.length) continue;
      else if (/^calls?\b/i.test(rest)) acts.push({ street, name, act: "call", amt: a[0], allIn });
      else if (/^(bets|(goes |is )?all[- ]in)\b/i.test(rest)) acts.push({ street, name, act: "bet", amt: a[0], allIn });
      else if (/^raises\b/i.test(rest)) acts.push({ street, name, act: "raise", ...(/\bto\b/i.test(rest) ? { to: a.at(-1) } : { amt: a[0] }), allIn });
    }
    if (!hole) throw new Error("Your hole cards aren't in this history (no \"Dealt to you [Ah Kd]\" line). Import a hand you played, as your poker client saved it.");
    if (!seats.some(s => s.name === hero)) throw new Error(`"${hero}" has hole cards but no "Seat N: ${hero} (…)" line: paste the whole hand history.`);
    if (!board.every(Boolean) || new Set([...hole, ...board]).size < 2 + board.length) board = [];

    // who was dealt in: anyone who posted, acted or (GG) has a "Dealt to" line; sitting-out seats did none of these
    const inHand = new Set([...posts.map(p => p.name), ...acts.map(a => a.name), ...dealt, hero]);
    const order = seats.filter(s => inHand.has(s.name)).sort((a, b) => a.seat - b.seat), n = order.length;
    if (n < 2 || n > 9) throw new Error(`${n} players in this hand: the page plays 2 to 9.`);
    const bb = (() => {   // the stakes before the seats ("($0.50/$1)", "$0.50/$1 Blinds", Winamax's "(ante/sb/bb)"), else the big blind posted
      const pre = lines.slice(0, Math.max(1, lines.findIndex(l => /^Seat \d+:/i.test(l)))).join(" ").replace(/\d{4}[/-]\d\d?[/-]\d\d?|\d\d?\/\d\d?\/\d{2,4}|players\s*:\s*\d+\/\d+/gi, "");   // not dates, party's "players : 6/6"
      const m = pre.match(new RegExp(String.raw`${AMT}[^\d/]{0,6}\/\s*${AMT}(?:[^\d/]{0,6}\/\s*${AMT})?`));
      const p = posts.find(x => x.kind === "bb");
      return m && num(m[3] || m[2]) > 0 ? num(m[3] || m[2]) : p ? p.amt : 0;
    })();
    if (!(bb > 0)) throw new Error("Couldn't find the big blind (neither the stakes, \"($0.50/$1)\", nor a \"posts big blind\" line).");
    const sbPost = posts.find(p => p.kind === "sb");
    if (sbPost && Math.abs(sbPost.amt / bb - 0.5) > 0.01) warnings.push(`The small blind is ${r2(sbPost.amt / bb)}bb: the page plays a half-big-blind small blind.`);

    // seats: clockwise from the button, the big blind is the first player after it who posted one (a new player
    // can post a big blind elsewhere); the page's names run from the first seat after the big blind round to it
    const bm = lines.join("\n").match(/Seat #?(\d+) is the button/i);
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
      const to = a.act === "raise" && a.to != null ? a.to : a.act === "fold" || a.act === "check" ? before : before + a.amt;
      // an all-in "bet" into a bet (party's "is all-In [$46]", Ignition's "All-in $39") raises or calls
      const top = Math.max(0, ...Object.values(put)), act = a.act === "bet" && top > 0 ? (to > top ? "raise" : "call") : a.act;
      left[a.name] -= to - before; put[a.name] = to;
      actions.push({ street: a.street, pos: pos[a.name], act, ...(a.act === "fold" || a.act === "check" ? {} : { to: r2(to / bb) }), ...(a.allIn || left[a.name] <= 1e-9 && to > before ? { allIn: true } : {}) });
    }
    // every seat plays its own stack. The page takes the same ante from everyone (the average), so a seat's stack
    // is set for what it has behind after the antes to be the history's (a big-blind ante paid by one player).
    const ante = Math.round(antes / n / bb * 1e4) / 1e4;
    const stacks = Object.fromEntries(order.map(s => [pos[s.name], r2((s.chips - (anteOf[s.name] || 0)) / bb + ante)]));
    // (effective: yours against the deepest opponent who didn't fold, for the summary)
    const opp = order.filter(s => s.name !== hero), folded = new Set(acts.filter(a => a.act === "fold").map(a => a.name));
    const deep = [opp.filter(s => !folded.has(s.name)), opp].find(x => x.length);
    const effective = Math.min(stacks[pos[hero]], Math.max(...deep.map(s => stacks[pos[s.name]])));
    const tour = tourney(lines);
    return { site, bb, players: n, hero: pos[hero], heroName: hero, stack: stacks[pos[hero]], stacks, effective, ante, straddle,
             hole, board: board.slice(0, 5), seats: order.map(s => ({ seat: s.seat, name: s.name, pos: pos[s.name], stack: r2(s.chips / bb) })), actions, warnings,
             ...(tour ? { tournament: tour } : {}) };
  }

  // The page state for a parsed hand: every seat with its own stack, and the actions through the page's engine one
  // by one, as if entered at the table (a short all-in, its side pot and a bet nobody could call in full play out
  // there as at the table). If the two disagree the import stops there.
  function importHand(text) {
    const h = parseHandHistory(text), warnings = [...h.warnings];
    const st = { finished: true, players: h.players, hero: h.hero, stack: h.stack, stacks: { ...h.stacks },   // (a hand played: its review is open)
                 cards: [...h.hole, ...h.board, null, null, null, null, null].slice(0, 7),
                 actions: { preflop: [], flop: [], turn: [], river: [] }, ...(h.ante ? { ante: h.ante } : {}), ...(h.straddle ? { straddle: true } : {}),
                 // (a tournament's hand keeps its tournament and takes no rake, whatever the page's: the page sends none for it)
                 ...(h.tournament ? { tournament: h.tournament } : +S.rake > 0 ? { rake: S.rake, ...(+S.rakeCap > 0 ? { rakeCap: S.rakeCap } : {}) } : {}) };
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

  const api = { parseHandHistory, importHand, splitHands, HH_SEATS: SEATS };
  if (typeof module === "object" && module.exports) module.exports = api; else Object.assign(globalThis, api);
})();
