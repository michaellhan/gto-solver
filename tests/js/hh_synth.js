// Synthetic PokerStars sessions for the session analyzer's tests (and a file to try it with in a browser:
// node tests/js/hh_synth.js 60 > session.txt). 6-max $0.50/$1, everyone $100 deep, the button moving each hand;
// hero ("HeroName") plays every seat in turn. Preflop: folds to an opener who raises to 2.5 (3 from the small
// blind), one player continues (calls or 3bets to 7.5, 10 out of position), the opener calls or folds a 3bet; a
// walk now and then. Heads-up after the flop: checks, bets of a third or three quarters of the pot, calls, folds
// and raises to 3x, capped at the stacks (an all-in runs the board out). Seeded: the same seed, the same file.
"use strict";

const rng = (seed) => () => { seed = (seed + 0x6D2B79F5) | 0; let t = seed; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"], POST = ["SB", "BB", "UTG", "HJ", "CO", "BTN"];
const money = (x) => "$" + (Math.round(x * 100) / 100).toFixed(2).replace(/\.00$/, "");

function synthHand(k, R, base = 250000000000) {
  const pick = (a) => a[Math.floor(R() * a.length)];
  const deck = [];
  for (const r of "23456789TJQKA") for (const s of "cdhs") deck.push(r + s);
  for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
  const btn = k % 6 + 1, seatOf = {}, name = {};
  ORDER.forEach((p, i) => { seatOf[p] = (btn - 1 + [3, 4, 5, 0, 1, 2][i]) % 6 + 1; });
  const heroSeat = 4;
  for (const p of ORDER) name[p] = seatOf[p] === heroSeat ? "HeroName" : `villain${seatOf[p]}`;
  const hero = ORDER.find(p => name[p] === "HeroName");
  const L = [`PokerStars Hand #${base + k}:  Hold'em No Limit ($0.50/$1.00 USD) - 2024/05/0${1 + k % 9} 2${k % 4}:${String(k % 60).padStart(2, "0")}:00 ET`,
             `Table 'Synth' 6-max Seat #${btn} is the button`];
  for (let s = 1; s <= 6; s++) L.push(`Seat ${s}: ${ORDER.map(p => name[p]).find((n, i) => seatOf[ORDER[i]] === s)} ($100 in chips)`);
  L.push(`${name.SB}: posts small blind $0.50`, `${name.BB}: posts big blind $1`, "*** HOLE CARDS ***", `Dealt to HeroName [${deck[0]} ${deck[1]}]`);
  const board = deck.slice(2, 7), left = Object.fromEntries(ORDER.map(p => [p, 100]));
  let bets = { SB: 0.5, BB: 1 }, high = 1;
  const line = (p, t) => L.push(`${name[p]}: ${t}`);
  const put = (p, to) => { const add = to - (bets[p] || 0); left[p] -= add; bets[p] = to; return add; };
  const raise = (p, to) => { to = Math.min(to, (bets[p] || 0) + left[p]); const inc = to - high; put(p, to); high = to; line(p, `raises ${money(inc)} to ${money(to)}${left[p] <= 1e-9 ? " and is all-in" : ""}`); };
  const call = (p) => { const to = Math.min(high, (bets[p] || 0) + left[p]); const add = put(p, to); line(p, `calls ${money(add)}${left[p] <= 1e-9 ? " and is all-in" : ""}`); };
  const folded = new Set(), fold = (p) => { folded.add(p); line(p, "folds"); };
  // preflop
  const o = R() < 0.9 ? Math.floor(R() * 5) : null;   // the opener's index in ORDER, or a walk
  let inHand = [];
  if (o == null) { for (const p of ORDER.slice(0, 5)) fold(p); }
  else {
    const opener = ORDER[o];
    for (const p of ORDER.slice(0, o)) fold(p);
    raise(opener, opener === "SB" ? 3 : 2.5);
    const rest = ORDER.slice(o + 1);
    const defender = rest.includes(hero) && hero !== opener ? hero : R() < 0.6 ? pick(rest) : null;
    let threeBet = false;
    for (const p of rest) {
      if (p !== defender) { fold(p); continue; }
      const x = R();
      if (x < 0.25) { raise(p, p === "SB" || p === "BB" ? 10 : 7.5); threeBet = true; } else if (x < 0.7) call(p); else fold(p);
    }
    if (threeBet) { if (R() < 0.55) call(opener); else fold(opener); }
    inHand = [opener, defender].filter(p => p && !folded.has(p));
  }
  // after the flop, heads-up
  if (inHand.length === 2) {
    const [oop, ip] = [...inHand].sort((a, b) => POST.indexOf(a) - POST.indexOf(b));
    let pot = ORDER.reduce((t, p) => t + (bets[p] || 0), 0);   // (the blinds are in bets)
    const streets = [["FLOP", 3], ["TURN", 4], ["RIVER", 5]];
    let over = false, allIn = false;
    for (const [st, n] of streets) {
      L.push(n === 3 ? `*** FLOP *** [${board.slice(0, 3).join(" ")}]` : `*** ${st} *** [${board.slice(0, n - 1).join(" ")}] [${board[n - 1]}]`);
      if (allIn || over) continue;
      bets = {}; high = 0;
      const bet = (p) => { const amt = Math.min(Math.round(pot * (R() < 0.5 ? 0.33 : 0.75) * 100) / 100, left[oop], left[ip]); put(p, amt); high = amt; line(p, `bets ${money(amt)}${left[p] <= 1e-9 ? " and is all-in" : ""}`); };
      const respond = (p, q) => {   // p faces q's bet: fold, call or raise to 3x (q then calls or folds)
        const x = R();
        if (x < 0.35) { fold(p); return "fold"; }
        if (x < 0.9 || left[p] <= high - (bets[p] || 0) || left[q] <= 1e-9) { call(p); return "call"; }
        raise(p, Math.round(high * 3 * 100) / 100);
        if (R() < 0.5) { fold(q); return "fold"; }
        call(q); return "call";
      };
      let r = null;
      if (R() < 0.3) { bet(oop); r = respond(ip, oop); }
      else { line(oop, "checks"); if (R() < 0.55) { bet(ip); r = respond(oop, ip); } else line(ip, "checks"); }
      pot += (bets[oop] || 0) + (bets[ip] || 0);
      if (r === "fold") over = true;
      if (left[oop] <= 1e-9 || left[ip] <= 1e-9) allIn = true;
      if (over) break;
    }
  }
  L.push("*** SUMMARY ***");
  return L.join("\n");
}

// n hands from a seed, separated by blank lines as PokerStars writes them
function synthSession(n, seed = 1, base) {
  const R = rng(seed), out = [];
  for (let k = 0; k < n; k++) out.push(synthHand(k, R, base));
  return out.join("\n\n\n") + "\n";
}

module.exports = { synthSession, synthHand, rng };
if (require.main === module) process.stdout.write(synthSession(+process.argv[2] || 60, +process.argv[3] || 1));
