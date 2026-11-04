// Explaining a decision: Copy for AI and Explain with AI, for Practice's cards (practice.js) and a decision of a finished
// hand in review (index.html, showExplain). The page loads it on first use: before practice.js, which reads the hand
// strength and a decision's words from here, and when a reviewed decision's advice first shows, so a review needs no
// Practice. It runs on the page's engine and grading (S, simulate, gradeClass, amtOf, fmt, ...), as practice.js does.
// Nothing leaves the page until the Explain with AI button is clicked.
"use strict";
(() => {
  // ---- 7-card hand strength: [category, tie-breakers...], compared left to right
  const CATS = ["high card", "a pair", "two pair", "three of a kind", "a straight", "a flush", "a full house", "four of a kind", "a straight flush"];
  function rank7(cs) {
    const v = (c) => 14 - RANKS.indexOf(c[0]), cnt = {}, suits = {};
    for (const c of cs) { cnt[v(c)] = (cnt[v(c)] || 0) + 1; (suits[c[1]] = suits[c[1]] || []).push(v(c)); }
    const straight = (vals) => { const s = new Set(vals); if (s.has(14)) s.add(1); for (let h = 14; h >= 5; h--) if ([0, 1, 2, 3, 4].every(k => s.has(h - k))) return h; return 0; };
    const flush = Object.values(suits).find(x => x.length >= 5);
    if (flush && straight(flush)) return [8, straight(flush)];
    const g = Object.entries(cnt).map(([r, n]) => [n, +r]).sort((a, b) => b[0] - a[0] || b[1] - a[1]);
    const kick = (not, n) => Object.keys(cnt).map(Number).filter(r => !not.includes(r)).sort((a, b) => b - a).slice(0, n);
    if (g[0][0] === 4) return [7, g[0][1], ...kick([g[0][1]], 1)];
    if (g[0][0] === 3 && g[1] && g[1][0] >= 2) return [6, g[0][1], g[1][1]];
    if (flush) return [5, ...flush.sort((a, b) => b - a).slice(0, 5)];
    const st = straight(Object.keys(cnt).map(Number));
    if (st) return [4, st];
    if (g[0][0] === 3) return [3, g[0][1], ...kick([g[0][1]], 2)];
    if (g[0][0] === 2 && g[1][0] === 2) return [2, g[0][1], g[1][1], ...kick([g[0][1], g[1][1]], 1)];
    if (g[0][0] === 2) return [1, g[0][1], ...kick([g[0][1]], 3)];
    return [0, ...kick([], 5)];
  }
  const cmp = (a, b) => { for (let i = 0; i < Math.max(a.length, b.length); i++) if ((a[i] || 0) !== (b[i] || 0)) return (a[i] || 0) - (b[i] || 0); return 0; };
  // a rank7() value in words: "Q-high flush", "two pair, kings and sevens"; against a hand of the same name (vs),
  // with the kicker that decides it
  const RW = { 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes", 7: "sevens", 8: "eights", 9: "nines", 10: "tens", 11: "jacks", 12: "queens", 13: "kings", 14: "aces" };
  const HI = (x) => x === 10 ? "10" : "  23456789TJQKA"[x];
  function handName(v, vs) {
    const [c, a, b] = v;
    const n = [`${HI(a)}-high`, `pair of ${RW[a]}`, `two pair, ${RW[a]} and ${RW[b]}`, `three ${RW[a]}`, `${HI(a)}-high straight`, `${HI(a)}-high flush`,
      `full house, ${RW[a]} full of ${RW[b]}`, `four ${RW[a]}`, a === 14 ? "royal flush" : `${HI(a)}-high straight flush`][c];
    if (!vs || vs[0] !== c || handName(vs) !== n) return n;
    const i = v.findIndex((x, k) => k > 0 && x !== vs[k]);
    return i > 0 ? `${n}, ${HI(v[i])} kicker` : n;
  }

  // ---- a decision in words: the solver's mix, the verdict and why
  // (r: an action your range almost never takes, whose EV is rough: the answer's rough_ev, shown with a "≈")
  const mixOf = (res) => res && !res.error && res.actions ? res.actions.map(a => ({ n: a.action, f: a.freq, ...(a.ev_bb !== undefined ? { ev: a.ev_bb } : {}), ...(a.rough_ev ? { r: 1 } : {}), ...(amtOf(a) != null ? { a: amtOf(a) } : {}) })) : null;
  // the verdict: best (the solver's top action), good (in its mix), inaccuracy (rare), mistake (almost never)
  function verdict(g) {
    if (!g) return { k: "none", label: "Not graded", c: "var(--muted)" };
    const cls = gradeClass(g);
    if (g.m === g.b) return { k: "best", label: "Best move", c: "var(--good)" };
    if (cls === "ok") return { k: "good", label: "Good", c: "var(--good)" };
    if (cls === "meh") return { k: "meh", label: "Inaccuracy", c: "var(--meh)" };
    return { k: "bad", label: "Mistake", c: "var(--bad)" };
  }
  const pct = (f) => `${Math.round(f * 100)}%`;
  const evTxt = (x) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(2)}`;
  function why(e) {
    const g = e.g, v = verdict(g);
    if (!g) return "This multiway spot has no EVs to grade by, so this decision isn't graded.";
    if (g.h != null) {   // (a multiway spot left to the betting rules: graded by EV)
      const rules = "No solve for this multiway spot: graded by the betting rules' EVs, a heuristic against every live range.";
      return g.m === g.b ? `${rules} Your move has the highest EV.` : `${rules} Its best is ${g.b}; yours gives up ${g.h}% of the pot.`;
    }
    const as = g.v ? ` (graded as its ${g.m})` : "";
    // (x: the answer's ranges left the solve earlier in the hand, so its mix is the best answer to those ranges)
    const on = g.x ? "Against the ranges this line leaves, the" : "The";
    if (v.k === "best") return g.f >= 0.995 ? `${on} solver always plays this here.` : `${on} solver's top choice here: it plays this ${pct(g.f)} of the time.`;
    if (v.k === "good") return `${on} solver mixes this in ${pct(g.f)} of the time${as}; its top choice is ${g.b} (${pct(g.bf)}).`;
    return `${on} solver ${g.f < 0.005 ? "never plays this" : `plays this only ${pct(g.f)} of the time`}${as}; it prefers ${g.b} (${pct(g.bf)}).`;
  }

  // ---- explaining a decision. Copy for AI puts the decision on the clipboard as a prompt for any AI chat;
  // Explain with AI streams Claude's answer to that prompt into the card. With EXPLAIN_PROXY set (the Cloudflare Worker
  // in cloudflare/explain-proxy, which holds the API key and fixes the model, system prompt and token cap; a few free
  // explanations a day per visitor) it goes there by default, with no key; otherwise, or when the user chose "Use my
  // own key", it goes from this browser to the Claude Messages API with their own API key (kept in this browser's
  // localStorage, sent only to api.anthropic.com). Each answer is kept in memory per decision (AI.cache).
  // EXPLAIN_PROXY: the deployed worker's URL, e.g. "https://gto-explain.<subdomain>.workers.dev/explain"; "" = own key only
  const EXPLAIN_PROXY = "https://gto-explain.combatrobotics.workers.dev/explain";
  const AI_KEY = "gto-anthropic-key", AI_OWN = "gto-explain-own", AI_URL = "https://api.anthropic.com/v1/messages", AI_MODEL = "claude-sonnet-5-5";
  const AI_SYSTEM = "You coach no-limit hold'em players who are learning from a GTO solver. The user sends one decision from a practice hand, "
    + "with the solver's frequencies and EVs for their exact hand. Treat those numbers as ground truth: never contradict them or invent other "
    + "frequencies or EVs.\n\nReply in exactly this shape, 70 words at most in all:\n"
    + "**Verdict:** one sentence, at most 20 words: was their move right, and what does the solver prefer.\n"
    + "- Exactly three bullets, each one sentence of at most 15 words: the three reasons that matter most here (ranges, equity, blockers, board texture "
    + "or sizing; only those that matter).\n"
    + "**Takeaway:** one rule they can reuse, at most 15 words.\n\n"
    + "Plain words, numbers only from the hand. Take Hero's made hand, draws, outs and position from the HERO'S HAND lines: they are exact, "
    + "so never work them out yourself. Don't mention data the hand doesn't have (such as missing EVs). No preamble, no restating the hand, no headings or tables. With no solver "
    + "numbers, coach the spot from sound principles and say it isn't solved.";
  // (the worker in cloudflare/explain-proxy sends this same prompt, its tests check they match)
  const aiKey = () => { try { return localStorage.getItem(AI_KEY) || ""; } catch (e) { return ""; } };
  // the user chose their own key over the free proxy ("Use my own key", or saving a key in the form)
  const ownChosen = () => { try { return localStorage.getItem(AI_OWN) === "1"; } catch (e) { return false; } };
  const setOwn = (on) => { try { if (on) localStorage.setItem(AI_OWN, "1"); else localStorage.removeItem(AI_OWN); } catch (e) {} };
  const viaProxy = () => !!EXPLAIN_PROXY && !(ownChosen() && aiKey());
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  // the model's text as paragraphs, **bold** kept, anything else as text
  // (the reasons cut at three, the most the coach's shape asks for: the model sometimes adds a fourth, weaker one)
  const MAX_BULLETS = 3;
  const md = (t) => { let n = 0; return mdText(String(t).split("\n").filter(l => !/^\s*[-*•]\s+/.test(l) || ++n <= MAX_BULLETS).join("\n")); };
  const mdText = (t) => esc(t).split(/\n{2,}/).map(p => p.trim()).filter(Boolean)
    .map(p => {
      const inline = (x) => x.replace(/^#+\s*/gm, "").replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
      const lines = p.split("\n"), item = /^\s*[-*•]\s+/;
      // (a paragraph of reasons as a list, so a long one wraps under its own text)
      if (lines.every(l => item.test(l))) return `<ul>${lines.map(l => `<li>${inline(l.replace(item, ""))}</li>`).join("")}</ul>`;
      return `<p>${inline(p).replace(item, "• ").replace(/\n\s*[-*•]\s+/g, "\n• ").replace(/\n/g, "<br>")}</p>`;
    }).join("");
  const hash = (s) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7).toString(36);
  const keyOf = (e) => e.id || "h" + hash(JSON.stringify([e.n, e.street, e.hole, e.rec, e.line]));
  const bb =(x) => `${fmt(x)}bb`;
  const Street = (s) => s[0].toUpperCase() + s.slice(1);

  // Hero's hand read from the cards, for the prompt (a model left to work out draws gets them wrong): what Hero has
  // made, each draw with the cards that complete it, the board's texture and Hero's position after the flop.
  // hole: Hero's two cards; board: the cards out; active: the seats still in
  const SUIT_NAME = { s: "spades", h: "hearts", d: "diamonds", c: "clubs" };
  const RANK_OF = (x) => x === 1 ? "A" : "  23456789TJQKA"[x];
  function heroRead(hole, board, active, hero) {
    const out = [], val = (c) => 14 - RANKS.indexOf(c[0]), river = board.length >= 5;
    const others = ORDER_POST.filter(p => p !== hero && active.has(p)), last = ORDER_POST.filter(p => active.has(p)).at(-1);
    if (others.length) out.push(last === hero ? `Position: Hero is in position (acts last after the flop) against ${others.join(", ")}`
      : `Position: Hero is out of position against ${others.filter(p => ORDER_POST.indexOf(p) > ORDER_POST.indexOf(hero)).join(", ")}`);
    if (!board.length) {
      const [a, b] = [...hole].sort((x, y) => val(y) - val(x));
      out.push(`Hero's hand: ${a[0]}${b[0]}${a[0] === b[0] ? " (a pocket pair)" : a[1] === b[1] ? "s (suited)" : "o (offsuit)"}`);
      return out;
    }
    const mine = rank7([...hole, ...board]), rb = rank7(board), onBoard = board.length >= 5 && cmp(rb, mine) === 0;
    // (a pair or trips that is all the board's: Hero's cards are only kickers)
    const boards = !onBoard && mine[0] > 0 && rb[0] === mine[0] && rb[1] === mine[1] && (mine[0] !== 2 || rb[2] === mine[2]);
    out.push(`Hero's made hand: ${handName(mine)}${onBoard ? " (all on the board: Hero only plays the board)" : boards ? " (on the board: Hero's cards are only kickers)" : ""}`);
    // texture: suits and pairs on the board
    const bs = {}; for (const c of board) bs[c[1]] = (bs[c[1]] || 0) + 1;
    const most = Math.max(...Object.values(bs)), ms = Object.keys(bs).find(k => bs[k] === most);
    const paired = new Set(board.map(c => c[0])).size < board.length;
    out.push(`Board: ${most >= 3 ? `${most} ${SUIT_NAME[ms]} (a flush is possible)` : most === 2 ? `two ${SUIT_NAME[ms]}` : "rainbow"}${paired ? ", paired" : ""}`);
    if (river || mine[0] >= 4) return out;   // (no cards to come, or already a straight or better)
    // flush draws: four (or on the flop three, backdoor) of a suit, at least one of them Hero's
    for (const st of Object.keys(SUIT_NAME)) {
      const n = [...hole, ...board].filter(c => c[1] === st).length, h = hole.filter(c => c[1] === st);
      if (!h.length || n < 3) continue;
      if (n === 4) out.push(`Draw: flush draw in ${SUIT_NAME[st]} (Hero's ${h.join(" ")}; 9 outs)`);
      else if (n === 3 && board.length === 3 && h.length >= 1) out.push(`Draw: backdoor flush draw in ${SUIT_NAME[st]} (needs two more)`);
    }
    // straight draws: the ranks that would give Hero a straight it doesn't have with the board alone
    const outs = [];
    for (let r = 2; r <= 14; r++) {
      const c = RANKS[14 - r] + "x";
      const with_ = rank7([...hole, ...board, c]), boardOnly = rank7([...board, c]);
      if (with_[0] === 4 && !(boardOnly[0] === 4 && boardOnly[1] === with_[1])) outs.push(RANK_OF(r));
    }
    if (outs.length) out.push(`Draw: ${outs.length >= 2 ? "straight draw, 8 outs" : "gutshot, 4 outs"}: any ${outs.join(" or ")} makes Hero a straight`);
    if (!out.some(l => l.startsWith("Draw:"))) out.push("Draw: none");
    return out;
  }
  // The spot as text, kept with each decision (the page's hand moves on): the game, your seat and cards, every
  // action so far with its size and the pot, and the decision (the pot, what's to call, the stacks behind).
  // vs: who the hand was against and what the solver's numbers assume (a hand played: not Practice's GTO opponents)
  function spot(street, vs = "The opponents play the solver's GTO strategy") {
    const n = S.players, hero = S.hero, keep = S.actions, board = boardCards();
    const who = (p) => p === hero ? `Hero (${SEAT_LABEL(p, n)})` : SEAT_LABEL(p, n);
    // the hand as it stood before the k-th action of street s
    const at = (s, k) => {
      const i = STREETS.indexOf(s);
      S.actions = Object.fromEntries(STREETS.map((t, j) => [t, j < i ? keep[t] : j === i ? keep[t].slice(0, k) : []]));
      try { return simulate(); } finally { S.actions = keep; }
    };
    const game = [`No-limit hold'em ${S.tournament || TOURNEY() ? "tournament" : "cash game"}, ${n}-handed, ${UNEQUAL() ? `stacks ${POS.map(p => `${SEAT_LABEL(p, n)} ${bb(STACK(p))}`).join(", ")}` : `${bb(S.stack)} stacks`}`,
      `Blinds 0.5/1bb${ANTE() ? `, ${bb(ANTE())} ante each` : ""}${STRADDLER() ? `, ${SEAT_LABEL(STRADDLER(), n)} straddles 2bb` : ""}, ${RAKE() ? RAKE_TEXT() + " (pots that see a flop)" : "no rake"}${ICM() ? `; ${ICM_TEXT()}: the grading is in prize equity (ICM)${BOUNTY() ? " and bounties" : ""}` : BOUNTY() ? `; ${ICM_TEXT()}: the grading counts the bounties` : ""}`,
      ...(ADV_TEXT() ? [`Solver settings: ${ADV_TEXT()}`] : []), vs];
    const action = [];
    for (const s of STREETS.slice(0, STREETS.indexOf(street) + 1)) {
      const cards = s === "preflop" ? "" : " " + board.slice(s === "flop" ? 0 : NEED[s] - 1, NEED[s]).join(" ");
      const parts = keep[s].map((a, k) => {
        const sim = at(s, k), info = sim.cur, p = a.pos, pot = sim.pot;
        const left = info ? STACK(p) - sim.invested[p] - info.bets[p] : 0;
        const says = a.action === "fold" ? "folds" : a.action === "check" ? "checks"
          : a.action === "call" ? `calls ${bb(info ? Math.min(info.curBet, info.bets[p] + left) : 0)}`
          : a.action === "all-in" ? `goes all-in (${bb(info ? info.bets[p] + left : 0)})`
          : a.action === "bet" || !info || info.curBet < 1e-9 ? `bets ${bb(a.to)} (${Math.round(a.to / pot * 100)}% of ${bb(pot)})` : `raises to ${bb(a.to)}`;
        return `${who(p)} ${says}`;
      });
      action.push(`${Street(s)}${cards} (pot ${bb(at(s, 0).pot)}): ${parts.join(", ") || "no action yet"}`);
    }
    const sim = simulate(), info = sim.cur || { pot: sim.pot, curBet: 0, bets: {} }, mine = info.bets[hero] || 0;
    const toCall = Math.max(0, Math.min(info.curBet - mine, STACK(hero) - sim.invested[hero] - mine));
    const behind = [hero, ...POS.filter(p => p !== hero && sim.active.has(p))].map(p => `${who(p)} ${bb(STACK(p) - sim.invested[p] - (info.bets[p] || 0))}`);
    return { game, hand: `Hero: ${SEAT_LABEL(hero, n)} with ${S.cards[0]} ${S.cards[1]}`, action, read: heroRead([S.cards[0], S.cards[1]], board, sim.active, hero),
      at: `${Street(street)}, Hero to act. Pot ${bb(info.pot)}${toCall > 1e-9 ? `, ${bb(toCall)} to call` : ", nothing to call"}. Stacks behind: ${behind.join(", ")}`,
      pot: info.pot, toCall };
  }
  // the decision as a prompt for an AI: the spot, what you did, the solver's mix with EVs, the verdict, the ask
  // (e.played: a hand played at a table, in review, not a practice hand; h: the hand, for a decision kept from before
  // the spot was: what the review list has, else this hand's)
  function prompt(e, h = {}) {
    const seat = e.seat || h.hero, hole = e.hole || h.hole || [];
    const g = e.g, v = verdict(g), c = e.ctx || { game: [`No-limit hold'em, ${S.players}-handed`], hand: `Hero: ${seat ? SEAT_LABEL(seat, S.players) : "?"} with ${hole.join(" ")}`,
      action: [e.board ? `Board: ${e.board.join(" ") || "none yet"}` : "", e.line ? `${Street(e.street)} action before Hero: ${e.line}` : ""].filter(Boolean), at: `${Street(e.street)}, Hero to act` };
    const r = e.rec, f = (x) => x > 0 && x < 0.005 ? "<1%" : pct(x);
    const did = r.action === "bet" ? `bet ${bb(r.to)}${c.pot ? ` (${Math.round(r.to / c.pot * 100)}% pot)` : ""}` : r.action === "raise" ? `raise to ${bb(r.to)}`
      : r.action === "call" && c.toCall ? `call ${bb(c.toCall)}` : r.action;
    const L = [`${e.played ? "I'm reviewing a no-limit hold'em hand I played, with a GTO solver" : "I'm practising no-limit hold'em against a GTO solver"}. Please explain one of my decisions.`, "", "GAME", ...c.game.map(x => "- " + x), "- " + c.hand, "",
      "ACTION (sizes in bb; the pot at the start of each street)", ...c.action, "",
      ...(c.read ? ["HERO'S HAND (worked out from the cards, exact: use these, don't recompute them)", ...c.read.map(x => "- " + x), ""] : []),
      "DECISION", "- " + c.at,
      `- I chose: ${did}${g && g.m !== r.action ? ` (graded as the solver's "${g.m}")` : ""}`];
    if (e.mix && e.mix.length && g && g.h != null) {
      L.push("- No solver solution for this multiway spot: the betting rules (a heuristic against every live range) give these EVs for my hand (bb):");
      for (const a of e.mix) L.push(`  - ${a.n}${a.a != null ? ` (${bb(a.a)})` : ""}: EV ${a.ev !== undefined ? evTxt(a.ev).replace("−", "-") : "?"}${a.n === g.m ? "  <- my move" : ""}`);
    } else if (e.mix && e.mix.length) {
      L.push("- The solver's strategy for my exact hand (how often, EV in bb):");
      for (const a of e.mix) L.push(`  - ${a.n}${a.a != null ? ` (${bb(a.a)})` : ""}: ${f(a.f)}${a.ev !== undefined ? `, EV ${evTxt(a.ev).replace("−", "-")}${a.r ? " (rough: my range almost never takes this)" : ""}` : ""}${g && a.n === g.m ? "  <- my move" : ""}`);
      if (!e.mix.some(a => a.ev !== undefined)) L.push("  (preflop: the solver's chart gives frequencies, no EVs)");
    } else L.push("- The solver doesn't solve this multiway spot, so it has no mix here");
    if (g) {
      L.push(g.h != null ? `- Highest EV by the betting rules: ${g.b}` : `- The solver's top action: ${g.b} (${f(g.bf)})`);
      L.push(`- EV lost against the best action: ${g.l != null ? bb(g.l) : "not known (no EVs preflop)"}`);
      L.push(`- Verdict: ${v.label}. ${why(e)}`);
      if (g.r) L.push("- Note: this solve hadn't fully converged, so close frequencies are rough");
      if (g.x) L.push("- Note: off the solved line: an earlier action the solve rarely takes left its ranges, so the frequencies are the best answer to the ranges the hand leaves");
    } else L.push(`- Verdict: ${v.label}`);
    // (short and skimmable: a verdict, two or three reasons, one takeaway)
    const shape = "Answer briefly: a one-sentence verdict, three short reasons (ranges, equity, blockers, board texture or bet sizing, "
      + "whichever matter here) and one takeaway I can reuse. Under 70 words.";
    if (!(e.mix && e.mix.length)) L.push("", "How should I think about this spot, and was my play reasonable? " + shape);
    else L.push("", "Why is the solver's play better than mine here (or, if mine is in its mix, why do both work)? Treat the solver's numbers above "
      + "as ground truth and don't contradict them. " + shape);
    return L.join("\n");
  }

  // an error answer from the API as one line for the card
  function aiError(status, msg) {
    if (status === 401) return "Anthropic didn't accept this API key (401). Check it, or enter another.";
    if (status === 403) return `This key isn't allowed to do that (403)${msg ? ": " + msg : ""}.`;
    if (status === 429) return "Too many requests, or this key is out of credit (429). Wait a moment and try again.";
    if (status === 529 || status === 503) return `Anthropic's API is busy right now (${status}). Try again in a moment.`;
    return `The request failed (${status})${msg ? ": " + msg : ""}.`;
  }
  // an error answer from the proxy (its JSON {error, retryAfter, scope}) for the card; quota: offer the key form
  function proxyError(status, j) {
    j = j || {};
    if (status === 429 && j.scope === "minute") return { err: `Too many explanations in a minute. Try again in ${j.retryAfter || 60}s.` };
    if (status === 429) return { err: j.scope === "global" ? "Today's free explanations are used up — add your own key" : "Daily free explanations used up — add your own key", quota: true };
    if (status === 403) return { err: "This page can't use the free explanations (403) — add your own key", quota: true };
    return { err: j.error ? String(j.error) : `The explanation service failed (${status}).` };
  }
  // One host's row of buttons and its answers, each kept in memory per decision (state.cache): Practice has one, a finished
  // hand's review another. promptOf(e): decision e as the prompt; show(): draw the row again
  function panel({ promptOf, show }) {
    // cache: id -> { s: "run" | "done" | "err", t: text so far, err, bad (the key was refused), via: "proxy" | "key",
    // quota (the free explanations are used up: offer the key form) }; open: ids whose
    // explanation (or key form) is shown; form: the id whose key form is open; entries: id -> decision, as rendered
    const AI = { cache: {}, open: {}, form: null, entries: {}, seq: 0, timer: null };
    // Inside Details: Copy for AI and Explain with AI, then the explanation (or the key form) when open
    function aiHtml(e) {
      const k = keyOf(e), c = AI.cache[k], open = AI.open[k];
      AI.entries[k] = e;
      let out = `<div class="cx-aib"><button class="small" data-ai="copy" data-k="${k}" title="Copy this decision as a prompt to paste into any AI chat">Copy for AI</button>`
        + `<button class="small" data-ai="explain" data-k="${k}" aria-expanded="${!!open}" title="${EXPLAIN_PROXY ? "Claude explains this decision" : "Claude explains this decision (with your own Anthropic API key)"}">${open ? "Hide explanation" : "Explain with AI"}</button></div>`;
      if (!open) return out;
      if (AI.form === k) return out + `<div class="cx-ai form"><label for="pxKey">Your Anthropic API key</label><div class="row">`
        + `<input id="pxKey" class="cx-key" type="password" placeholder="sk-ant-…" autocomplete="off" spellcheck="false" data-k="${k}">`
        + `<button class="small" data-ai="save" data-k="${k}">Save &amp; explain</button></div>`
        + (AI.formErr ? `<p class="err">${esc(AI.formErr)}</p>` : "")
        + `<p class="note">It's kept only in this browser (localStorage) and sent only to api.anthropic.com, from this page; each explanation is a small charge to your Anthropic account. `
        + `<a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer">Get a key</a></p><button class="link" data-ai="cancel" data-k="${k}">Cancel</button></div>`;
      if (!c) return out;
      out += `<div class="cx-ai${c.s === "err" ? " err" : ""}" aria-live="polite">`
        + (c.t ? md(c.t) : c.s === "run" ? `<p class="muted"><span class="spinner"></span>Asking Claude…</p>` : "")
        + (c.s === "err" ? `<p class="err">${esc(c.err)}</p><div class="acts2">${c.quota ? "" : `<button class="small" data-ai="retry" data-k="${k}">Try again</button>`}`
          + `${c.bad ? `<button class="small" data-ai="change" data-k="${k}">Enter another key</button>` : ""}${c.quota ? `<button class="small" data-ai="own" data-k="${k}">Add your own key</button>` : ""}</div>` : "")
        + `<div class="ft"><span>Claude ${AI_MODEL.replace("claude-", "").replace(/-(\d)-(\d)$/, " $1.$2").replace(/^./, (x) => x.toUpperCase())} · from the solver's numbers above</span>`
        + (EXPLAIN_PROXY && c.via === "proxy" && !c.quota ? `<button class="link" data-ai="own" data-k="${k}">Use my own key</button>` : "")
        + (EXPLAIN_PROXY && c.via === "key" ? `<button class="link" data-ai="free" data-k="${k}">Use free explanations</button>` : "")
        + (aiKey() ? `<button class="link" data-ai="forget" data-k="${k}">Forget key</button>` : "") + `</div></div>`;
      return out;
    }
    // draw the row again (at most every 50 ms while an answer streams in)
    function redraw(now) {
      if (now) { clearTimeout(AI.timer); AI.timer = null; show(); return; }
      if (AI.timer) return;
      AI.timer = setTimeout(() => { AI.timer = null; show(); }, 50);
    }
    // stream Claude's explanation of decision k into AI.cache[k] (Server-Sent Events: text_delta pieces)
    async function explain(k) {
      const e = AI.entries[k], key = aiKey(), proxy = viaProxy();
      if (!e) return;
      if (!proxy && !key) { AI.form = k; AI.open[k] = true; return redraw(true); }
      const c = AI.cache[k] = { s: "run", t: "", via: proxy ? "proxy" : "key" };
      AI.open[k] = true; redraw(true);
      let res;
      try {
        // the proxy takes only the prompt (it sets the model, system prompt and token cap itself)
        res = proxy ? await fetch(EXPLAIN_PROXY, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: promptOf(e).slice(0, 8000) }) })
          : await fetch(AI_URL, { method: "POST", headers: { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true" }, body: JSON.stringify({ model: AI_MODEL, max_tokens: 700, stream: true,
            thinking: { type: "between_tools" }, output_config: { effort: "low" }, system: AI_SYSTEM, messages: [{ role: "user", content: promptOf(e) }] }) });
      } catch (err) {
        Object.assign(c, { s: "err", err: proxy ? "Couldn't reach the explanation service. Check your connection (or a content blocker) and try again."
          : "Couldn't reach api.anthropic.com. Check your connection (or a content blocker) and try again." }); return redraw(true);
      }
      if (!res.ok) {
        let j = null; try { j = await res.json(); } catch (err) {}
        if (proxy) Object.assign(c, { s: "err", ...proxyError(res.status, j) });
        else Object.assign(c, { s: "err", err: aiError(res.status, j && j.error && j.error.message || ""), bad: res.status === 401 });
        return redraw(true);
      }
      let buf = "", stop = null;
      const take = (block) => {
        const data = block.split("\n").filter(l => l.startsWith("data:")).map(l => l.slice(5).trim()).join("");
        if (!data) return;
        let m; try { m = JSON.parse(data); } catch (err) { return; }
        if (m.type === "content_block_delta" && m.delta && m.delta.type === "text_delta") { c.t += m.delta.text; redraw(); }
        else if (m.type === "message_delta" && m.delta && m.delta.stop_reason) stop = m.delta.stop_reason;
        else if (m.type === "error") throw new Error(m.error && m.error.message || "the API stopped with an error");
      };
      const feed = (chunk, end) => {
        buf = (buf + chunk).replace(/\r\n?/g, "\n");
        let i; while ((i = buf.indexOf("\n\n")) >= 0) { const b = buf.slice(0, i); buf = buf.slice(i + 2); take(b); }
        if (end && buf.trim()) { take(buf); buf = ""; }
      };
      try {
        if (res.body && res.body.getReader) {
          const rd = res.body.getReader(), dec = new TextDecoder();
          for (;;) { const { done, value } = await rd.read(); if (done) break; feed(dec.decode(value, { stream: true })); }
          feed(dec.decode(), true);
        } else feed(await res.text(), true);
      } catch (err) {
        Object.assign(c, { s: "err", err: `The answer stopped early: ${err && err.message || err}` }); return redraw(true);
      }
      if (stop === "refusal" && !c.t) Object.assign(c, { s: "err", err: "Claude declined to explain this one." });
      else if (!c.t) Object.assign(c, { s: "err", err: "The answer came back empty. Try again." });
      else { if (stop === "max_tokens") c.t += " …"; c.s = "done"; }
      redraw(true);
    }
    // the buttons of the AI row (a, the button's data-ai; value: the key typed, for save)
    function aiClick(a, k, value) {
      if (a === "copy") {
        const e = AI.entries[k]; if (!e) return;
        return copyText(promptOf(e)).then(ok => { if (ok) toast("Copied", null, 2000); });
      }
      if (a === "explain") {
        if (AI.open[k]) { AI.open[k] = false; if (AI.form === k) AI.form = null; return redraw(true); }
        const c = AI.cache[k];
        if (c && c.s !== "err") { AI.open[k] = true; return redraw(true); }
        return explain(k);
      }
      if (a === "save") {
        const inp = document.getElementById("pxKey"), v = String(value != null ? value : inp && inp.value || "").trim();
        if (!/^sk-ant-\S{8,}$/.test(v)) { AI.formErr = "That doesn't look like an Anthropic API key (they start with sk-ant-)."; return redraw(true); }
        try { localStorage.setItem(AI_KEY, v); } catch (err) { AI.formErr = "This browser won't store the key (private mode?)."; return redraw(true); }
        setOwn(true); AI.form = null; AI.formErr = ""; return explain(k);
      }
      if (a === "cancel") { AI.form = null; AI.formErr = ""; AI.open[k] = false; return redraw(true); }
      if (a === "retry") return explain(k);
      if (a === "change") { delete AI.cache[k]; AI.form = k; AI.open[k] = true; return redraw(true); }
      // Use my own key (or Add your own key, when the free ones are used up): the saved key, else the key form
      if (a === "own") { delete AI.cache[k]; AI.open[k] = true; if (aiKey()) { setOwn(true); return explain(k); } AI.form = k; return redraw(true); }
      if (a === "free") { setOwn(false); delete AI.cache[k]; return explain(k); }
      if (a === "forget") { try { localStorage.removeItem(AI_KEY); } catch (err) {} setOwn(false); for (const x of Object.keys(AI.cache)) if (AI.cache[x].s === "err") delete AI.cache[x]; toast("API key forgotten", null, 2500); return redraw(true); }
    }
    // the row's buttons, and Enter in the key form, in the element it was drawn into
    function wire(el) {
      const typed = (b) => { const f = b.closest && b.closest(".cx-ai"), i = f && f.querySelector("input.cx-key"); return i ? i.value : undefined; };
      el.querySelectorAll("[data-ai]").forEach(b => { b.onclick = () => aiClick(b.dataset.ai, b.dataset.k, typed(b)); });
      el.querySelectorAll("input.cx-key").forEach(i => { i.onkeydown = (ev) => { if (ev.key === "Enter") { ev.preventDefault(); aiClick("save", i.dataset.k, i.value); } }; });
    }
    return { html: aiHtml, explain, click: aiClick, wire, state: AI };
  }

  window.Explain = { rank7, cmp, handName, CATS, mixOf, verdict, pct, evTxt, why, heroRead, spot, prompt, keyOf, hash, md, panel };
})();
