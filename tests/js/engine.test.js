// The page's hand engine: seats and acting order, blinds/antes/straddle, recording actions and their
// validation, pot and bet-size math, undo/rewind, the seat shortcuts and the keyboard.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const label = (b) => b.innerHTML.replace(/<.*$/, "");
const sub = (b) => (b.innerHTML.match(/<small>(.*?)<\/small>/) || [])[1] || "";
function page(st) { const p = loadPage(); p.S = st; return p; }

test("seats and acting order for 2-9 players, with and without a straddle", () => {
  const p = loadPage();
  const order = (n, straddle) => {
    p.S = state({ players: n, hero: n === 2 ? "SB" : "BB", straddle });
    return p.val("[POS, ORDER_POST, simulate().streets.preflop.order, STRADDLER()]");
  };
  assert.deepEqual(order(2), [["SB", "BB"], ["BB", "SB"], ["SB", "BB"], null]);
  assert.deepEqual(order(2, true), [["SB", "BB"], ["BB", "SB"], ["SB", "BB"], null]);   // no straddle heads-up
  assert.deepEqual(order(3), [["BTN", "SB", "BB"], ["SB", "BB", "BTN"], ["BTN", "SB", "BB"], null]);
  assert.deepEqual(order(3, true), [["BTN", "SB", "BB"], ["SB", "BB", "BTN"], ["SB", "BB", "BTN"], "BTN"]);
  assert.deepEqual(order(6), [["UTG", "HJ", "CO", "BTN", "SB", "BB"], ["SB", "BB", "UTG", "HJ", "CO", "BTN"], ["UTG", "HJ", "CO", "BTN", "SB", "BB"], null]);
  assert.deepEqual(order(6, true)[2], ["HJ", "CO", "BTN", "SB", "BB", "UTG"]);
  const nine = order(9, true);
  assert.deepEqual(nine[0], ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]);
  assert.deepEqual(nine[1], ["SB", "BB", "UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN"]);
  assert.equal(nine[3], "UTG");
  for (const n of [4, 5, 7, 8]) assert.equal(order(n)[0].length, n);
  assert.equal(p.run('SEAT_LABEL("SB", 2)'), "BTN/SB");
  assert.equal(p.run('SEAT_LABEL("SB", 3)'), "SB");
});

test("blinds, antes and a straddle: pot, the bet to call and who acts first", () => {
  let s = page(state({ cards: ["Ah", "Kh"] })).sim();
  assert.equal(s.pot, 1.5); assert.equal(s.toAct, "UTG"); assert.equal(s.cur.curBet, 1);
  s = page(state({ cards: ["Ah", "Kh"], ante: 0.125 })).sim();
  assert.equal(s.pot, 1.5 + 6 * 0.125); assert.equal(s.invested.BTN, 0.125);
  s = page(state({ cards: ["Ah", "Kh"], straddle: true })).sim();
  assert.equal(s.pot, 3.5); assert.equal(s.toAct, "HJ"); assert.equal(s.cur.curBet, 2); assert.equal(s.cur.bets.UTG, 2);
  s = page(state({ players: 2, hero: "SB", cards: ["Ah", "Kh"] })).sim();
  assert.equal(s.toAct, "SB");
});

test("the big blind gets its option, then the street closes and waits for the flop", () => {
  const p = page(state({ cards: ["Ah", "Kh"], preflop: acts("UTG f, HJ c, CO f, BTN f, SB c") }));
  let s = p.sim();
  assert.equal(s.toAct, "BB"); assert.equal(s.cur.curBet, 1);
  assert.equal(p.act({ action: "check" }), null);
  s = p.sim();
  assert.equal(s.waiting, "flop"); assert.equal(s.cur, null); assert.equal(s.pot, 3);
  assert.equal(p.act({ action: "fold" }), "Deal the next card first.");
});

test("streets: postflop order, closing, all-in runout and the hand ending", () => {
  const base = { cards: ["Ah", "Kh", "Kd", "7s", "2c", "9h", "3d"], preflop: acts("UTG f, HJ f, CO r2.5, BTN c, SB f, BB c") };
  let s = page(state(base)).sim();
  assert.equal(s.current, "flop"); assert.equal(s.toAct, "BB"); assert.equal(s.pot, 8);
  s = page(state({ ...base, flop: acts("BB x, CO b4, BTN r12, BB f, CO c") })).sim();
  assert.equal(s.current, "turn"); assert.equal(s.toAct, "CO"); assert.equal(s.pot, 8 + 12 + 12);   // BB's check-fold adds nothing
  assert.deepEqual(s.active, ["CO", "BTN"]);
  s = page(state({ ...base, flop: acts("BB x, CO a, BTN c, BB f") })).sim();
  assert.equal(s.runout, true); assert.equal(s.handOver, true); assert.equal(s.pot, 2.5 + 100 + 100 + 0.5 - 0);
  assert.deepEqual(s.allin, ["CO", "BTN"]);
  s = page(state({ ...base, cards: base.cards.slice(0, 5), flop: acts("BB x, CO a, BTN c, BB f") })).sim();
  assert.equal(s.waiting, "turn"); assert.equal(s.runout, true);
  s = page(state({ ...base, flop: acts("BB b3, CO f, BTN f") })).sim();
  assert.equal(s.handOver, true); assert.equal(s.current, "flop"); assert.deepEqual(s.active, ["BB"]);
  s = page(state({ ...base, flop: acts("BB x, CO x, BTN x"), turn: acts("BB x, CO x, BTN x"), river: acts("BB x, CO x, BTN x") })).sim();
  assert.equal(s.handOver, true); assert.equal(s.current, "river"); assert.equal(s.pot, 8);
});

test("recording actions: validation, rounding and labels", () => {
  const p = page(state({ cards: ["Ah", "Kh"] }));
  assert.equal(p.act({ action: "check" }), "UTG can't check, there's 1bb to call.");
  assert.equal(p.act({ action: "bet", to: 1 }), "1bb isn't a raise, the bet is 1bb.");
  assert.equal(p.act({ action: "bet", to: NaN }), "Enter an amount.");
  assert.equal(p.act({ action: "bet", to: 0 }), "Enter an amount.");
  assert.equal(p.act({ action: "bet", to: 2.7 }), null);                     // rounded to the half bb
  assert.deepEqual(p.S.actions.preflop.at(-1), { pos: "UTG", action: "raise", to: 2.5 });
  assert.equal(p.act({ action: "bet", to: 150 }), null);                     // more than the stack: all-in
  assert.deepEqual(p.S.actions.preflop.at(-1), { pos: "HJ", action: "all-in" });
  for (let i = 0; i < 4; i++) p.act({ action: "fold" });
  assert.equal(p.act({ action: "call" }), null);                             // UTG calls the jam: hand runs out
  assert.equal(p.sim().waiting, "flop");
  const q = page(state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c") }));
  assert.equal(q.act({ action: "call" }), null);                             // nothing to call: a check
  assert.deepEqual(q.S.actions.flop.at(-1), { pos: "BB", action: "check" });
  assert.equal(q.act({ action: "bet", to: 2 }), null);                       // first in postflop: a bet
  assert.deepEqual(q.S.actions.flop.at(-1), { pos: "BTN", action: "bet", to: 2 });
  assert.equal(q.act({ action: "bet", to: 6 }), null);
  assert.deepEqual(q.S.actions.flop.at(-1), { pos: "BB", action: "raise", to: 6 });
  const r = page(state({ cards: ["Ah", "Kh"], preflop: acts("UTG f, HJ f, CO f, BTN f, SB f") }));
  assert.equal(r.sim().handOver, true);
  assert.equal(r.act({ action: "check" }), "The hand is over.");
});

test("a call for less than the bet puts the caller all-in", () => {
  const p = page(state({ stack: 20, cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c") }));
  p.act({ action: "bet", to: 15 });   // BB 15 into 5.5 with 17.5 behind
  p.act({ action: "all-in" });        // BTN 17.5
  const s = p.sim();
  assert.equal(s.cur.curBet, 17.5); assert.equal(s.toAct, "BB");
  p.act({ action: "call" });
  assert.deepEqual(p.sim().allin, ["BTN", "BB"]);
  assert.equal(p.sim().pot, 40.5);
});

test("action buttons: preflop sizes, postflop sizes, raises and pot odds", () => {
  const sizes = (p) => p.buttons().filter(b => b.className.startsWith("agg")).map(b => `${label(b)} (${sub(b)})`);
  let p = page(state({ cards: ["Ah", "Kh"] }));
  assert.deepEqual(sizes(p), ["Raise to 2.5 (open)", "All-in 100 ()"]);
  assert.match(p.byId("customBox").innerHTML, /<input id="customAmt" inputmode="decimal" data-key="r"[^>]* placeholder="Raise to, bb"/);   // (R goes to it)
  p = page(state({ hero: "SB", cards: ["Ah", "Kh"], preflop: acts("UTG f, HJ f, CO f, BTN f") }));
  assert.deepEqual(sizes(p)[0], "Raise to 3 (open)");                          // SB opens 3x
  p = page(state({ cards: ["Ah", "Kh"], preflop: acts("UTG r2.5") }));
  assert.deepEqual(sizes(p)[0], "Raise to 7.5 (3x)");
  p = page(state({ hero: "SB", cards: ["Ah", "Kh"], preflop: acts("UTG r2.5, HJ f, CO f, BTN f") }));
  assert.deepEqual(sizes(p)[0], "Raise to 10 (4x)");                           // blinds 3bet bigger
  p = page(state({ cards: ["Ah", "Kh"], straddle: true }));
  assert.deepEqual(sizes(p)[0], "Raise to 5 (open)");                          // 2.5x the straddle
  p = page(state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c") }));
  assert.deepEqual(sizes(p), ["Bet 33% (2 bb)", "Bet 75% (4 bb)", "All-in 97.5 ()"]);
  assert.match(p.byId("felt").children.at(-1).innerHTML, /pot 5.5 bb/);                      // (the pot is on the table,
  assert.doesNotMatch(p.byId("actionArea").children[0].innerHTML, /pot/);                    // not repeated with nothing to call)
  assert.match(p.byId("actionArea").children[0].innerHTML, /BB to act/);
  assert.match(p.byId("customBox").innerHTML, /placeholder="Other amount, bb"/);
  p.act({ action: "bet", to: 4 });
  assert.deepEqual(sizes(p).slice(0, 2), ["Raise to 14 (¾ pot)", "Raise to 17.5 (pot)"]);   // 4 + ¾(9.5 + 4) = 14.125; 4 + (9.5 + 4)
  assert.equal(sub(p.button("Call")), "4 bb");
  assert.match(p.byId("actionArea").children[0].innerHTML, /pot 9.5 bb · 4 to call · .*need 30%/);   // 4 / 13.5
});

// Bug: the default raise ignored limpers and callers (2.5bb over two limpers). The sizes match the
// advisor's _default_size: 3x (4x from the blinds) plus 1bb a limper / one bet a caller of the open.
test("preflop raise sizes over limpers and callers match the advisor", () => {
  const sizes = (p) => p.buttons().filter(b => b.className.startsWith("agg")).map(b => `${label(b)} (${sub(b)})`);
  const first = (pre, extra) => sizes(page(state({ hero: "HJ", cards: ["Ah", "Kh"], preflop: acts(pre), ...extra })))[0];   // (HJ: in every hand here)
  assert.equal(first("UTG c"), "Raise to 4 (iso)");                            // 3 + 1 limper
  assert.equal(first("UTG c, HJ c"), "Raise to 5 (iso)");                      // 3 + 2 limpers
  assert.equal(first("UTG c, HJ f, CO f, BTN f", { hero: "UTG" }), "Raise to 5 (iso)");        // SB: 4x + 1
  assert.equal(first("UTG c, HJ f, CO f, BTN f, SB c", { hero: "UTG" }), "Raise to 5.5 (iso)"); // BB: 4x + 1 + SB's 0.5
  assert.equal(first("HJ c", { straddle: true }), "Raise to 8 (iso)");         // 3x the straddle + 2
  assert.equal(first("UTG r2.5, HJ c"), "Raise to 10 (4x)");                   // squeeze: 3x + the caller's 2.5
  assert.equal(first("UTG r2.5, HJ c, CO f, BTN f"), "Raise to 12.5 (5x)");    // from the SB: 4x + 2.5
  assert.equal(first("UTG r2.5, HJ r7.5, CO f, BTN f, SB f, BB f"), "Raise to 17.5 (2.33x)");  // 4bet: 2.3x (17.25, rounded to 0.5)
});

// Bug: the preflop label was worked out from the size before it was rounded to 0.5bb, so a 4bet of
// 2.3 x 7.5 = 17.25 showed "2.3x" on a button that raised to 17.5 (2.33x).
test("raise-size labels describe the rounded size the button uses", () => {
  const sizes = (p) => p.buttons().filter(b => b.className.startsWith("agg")).map(b => [label(b), sub(b)]);
  for (const pre of ["UTG r2.5, HJ r7.5, CO f, BTN f, SB f, BB f", "UTG r2.5, HJ r8, CO f, BTN f, SB f, BB f", "UTG r3, HJ c"]) {
    const p = page(state({ hero: "UTG", cards: ["Ah", "Kh"], preflop: acts(pre) }));
    const [lab, tag] = sizes(p)[0];
    const to = parseFloat(lab.replace("Raise to ", "")), cur = p.sim().streets.preflop.curBet;
    assert.equal(to * 2 % 1, 0, lab);
    assert.equal(tag, (Math.round(to / cur * 100) / 100) + "x", `${pre}: ${lab} (${tag})`);
  }
});

// Bug: facing a postflop bet, the page offered 2.5x the bet while the advisor's default raise (used when a
// raise is recorded without a size) is the bet plus 75% of the pot after calling it (_default_size).
test("postflop default bet and raise sizes match the advisor", () => {
  const sizes = (p) => p.buttons().filter(b => b.className.startsWith("agg")).map(b => `${label(b)} (${sub(b)})`);
  const pre = acts("UTG f, HJ f, CO r2.5, BTN c, SB f, BB c");   // pot 8
  // heads-up: BB bets 6 into 8; the advisor raises to 6 + 0.75 * (14 + 6) = 21
  let p = page(state({ hero: "CO", cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: pre, flop: acts("BB b6") }));
  assert.deepEqual(sizes(p).slice(0, 2), ["Raise to 21 (¾ pot)", "Raise to 26 (pot)"]);
  // a caller in between: pot 20 after BB's bet and CO's call, BTN raises to 6 + 0.75 * (20 + 6) = 25.5
  p = page(state({ hero: "BTN", cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: pre, flop: acts("BB b6, CO c") }));
  assert.deepEqual(sizes(p).slice(0, 2), ["Raise to 25.5 (¾ pot)", "Raise to 32 (pot)"]);
  // facing a raise after our bet: we put in 4, BB raised to 12 (pot 5.5 + 16 = 21.5, 8 to call);
  // 12 + 0.75 * (21.5 + 8) = 34.125, rounded to 34
  p = page(state({ hero: "CO", cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO r2.5, BTN f, SB f, BB c"), flop: acts("BB x, CO b4, BB r12") }));
  assert.deepEqual(sizes(p).slice(0, 2), ["Raise to 34 (¾ pot)", "Raise to 41.5 (pot)"]);
  // unopened: 33% and 75% of the pot, to the half big blind
  p = page(state({ hero: "BB", cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: pre }));
  assert.deepEqual(sizes(p).slice(0, 2), ["Bet 33% (2.5 bb)", "Bet 75% (6 bb)"]);
});

test("undo removes the last action, across streets", () => {
  const p = page(state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") }));
  p.run("undo()");
  assert.deepEqual(p.S.actions.flop, []);
  p.run("undo()");
  assert.equal(p.S.actions.preflop.length, 5);
  assert.equal(p.sim().toAct, "BB");
  assert.deepEqual(p.S.cards.slice(2, 5), ["Kd", "7s", "2c"]);   // the board stays
  const q = page(state({}));
  q.run("undo()");
  assert.deepEqual(q.S, state({}));
});

test("rewinding to one of your decisions clears later actions and cards", () => {
  const p = page(state({ cards: ["Ah", "Kh", "Kd", "7s", "2c", "9h"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"),
    flop: acts("BB x, BTN b2, BB c"), turn: acts("BB x") }));
  p.run('rewindTo("flop", 1)');
  assert.deepEqual(p.S.actions.flop, acts("BB x"));
  assert.deepEqual(p.S.actions.turn, []);
  assert.deepEqual(p.S.cards, ["Ah", "Kh", "Kd", "7s", "2c", null, null]);
  assert.equal(p.sim().toAct, "BTN");
  p.run('rewindTo("preflop", 3)');
  assert.deepEqual(p.S.cards, ["Ah", "Kh", null, null, null, null, null]);
  assert.equal(p.S.actions.preflop.length, 3);
});

test("clicking a seat folds everyone before it (checks when there's nothing to call)", () => {
  const p = page(state({ hero: "BB", cards: ["Ah", "Kh"] }));
  assert.equal(p.run('advanceTo("BTN")'), null);
  assert.deepEqual(p.S.actions.preflop, acts("UTG f, HJ f, CO f"));
  p.act({ action: "bet", to: 2.5 });
  assert.equal(p.run('advanceTo("UTG")'), "UTG already folded.");
  const q = page(state({ hero: "BTN", cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ c, CO c, BTN c, SB f, BB x") }));
  assert.equal(q.run('advanceTo("CO")'), null);
  assert.deepEqual(q.S.actions.flop, acts("BB x, HJ x"));
});

// Bug: clicking a seat after yours folded (or checked) your hand for you, without advice or a grade.
test("clicking a seat after yours stops at your turn", () => {
  const p = page(state({ hero: "CO", cards: ["Ah", "Kh"] }));
  assert.equal(p.run('advanceTo("BB")'), "It's your turn first.");
  assert.deepEqual(p.S.actions.preflop, acts("UTG f, HJ f"));
  assert.equal(p.sim().toAct, "CO");
  assert.deepEqual(p.sim().active, ["UTG", "HJ", "CO", "BTN", "SB", "BB"].filter(x => x !== "UTG" && x !== "HJ"));
});

test("the 'fold to me' / 'everyone checks' shortcut", () => {
  const p = page(state({ hero: "BTN", cards: ["Ah", "Kh"] }));
  assert.equal(label(p.button("Fold to me")), "Fold to me");
  p.key("s");
  assert.deepEqual(p.S.actions.preflop, acts("UTG f, HJ f, CO f"));
  p.act({ action: "bet", to: 2.5 });
  p.button("Everyone else folds").click();
  assert.equal(p.sim().handOver, true);
  const q = page(state({ hero: "BB", cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ c, CO c, BTN f, SB f, BB x"), flop: acts("BB x") }));
  q.button("Everyone checks").click();
  assert.equal(q.sim().waiting, "turn");
});

// Bug: "Everyone else folds" folded the big blind's free option after limps (BB could check).
test("'everyone else folds' checks a player who has nothing to call", () => {
  const p = page(state({ hero: "BTN", cards: ["Ah", "Kh"], preflop: acts("UTG f, HJ c, CO f, BTN c") }));
  assert.equal(p.sim().toAct, "SB");
  p.button("Everyone else folds").click();
  assert.deepEqual(p.S.actions.preflop.slice(4), acts("SB f, BB x"));
  assert.equal(p.sim().waiting, "flop");
  assert.deepEqual(p.sim().active, ["HJ", "BTN", "BB"]);
});

test("cards: placing, refusing duplicates, removing and the board order", () => {
  const p = page(state({}));
  p.run('placeCard("Ah"); placeCard("Ah")');
  assert.deepEqual(p.S.cards, ["Ah", null, null, null, null, null, null]);
  assert.match(p.byId("toast").innerHTML, /A♥ is already in your hand/);
  p.run('placeCard("Kh")');
  assert.deepEqual(p.S.cards.slice(0, 2), ["Ah", "Kh"]);
  p.run('removeCard(0)');
  assert.equal(p.S.cards[0], null);
  p.run('placeCard("Qd")');
  assert.deepEqual(p.S.cards.slice(0, 2), ["Qd", "Kh"]);
});

// Bug: with an earlier board card removed the later ones slid down, so a turn card was treated as part of
// the flop (and sent to the solver as the board) while the slot waited for its replacement.
test("a board with a gap uses only the cards before the gap", () => {
  const p = page(state({ cards: ["Ah", "Kh", "Kd", "7s", "2c", "9h", "3d"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"),
    flop: acts("BB x, BTN x"), turn: acts("BB x") }));
  assert.equal(p.sim().current, "turn");
  p.run("removeCard(5)");   // the turn
  assert.equal(p.run('boardCards().join(" ")'), "Kd 7s 2c");
  assert.equal(p.sim().waiting, "turn");
  assert.equal(p.payload().board, "Kd7s2c");
  p.run("removeCard(3)");   // a flop card
  assert.equal(p.run('boardCards().join(" ")'), "Kd");
  assert.equal(p.sim().waiting, "flop");
  p.run('placeCard("7s")'); p.run('placeCard("9h")');
  assert.equal(p.run('boardCards().join(" ")'), "Kd 7s 2c 9h 3d");
});

test("typing cards: rank then suit, 10 for T, Esc and Backspace cancel", () => {
  const p = page(state({}));
  p.key("a"); p.key("h");
  p.key("1"); p.key("0"); p.key("c");
  assert.deepEqual(p.S.cards.slice(0, 2), ["Ah", "Tc"]);
  p.key("k"); p.key("Escape"); p.key("s");
  assert.equal(p.S.cards[2], null);
  p.key("1"); p.key("s");     // "1" then a suit: cancelled
  assert.equal(p.S.cards[2], null);
  p.key("q"); p.key("Backspace"); p.key("d");
  assert.equal(p.S.cards[2], null);
  p.key("a"); p.key("h");     // a duplicate is refused
  assert.equal(p.S.cards[2], null);
});

test("keyboard actions: F, C, X, A, digits, U", () => {
  const p = page(state({ hero: "BB", cards: ["Ah", "Kh"] }));
  p.key("f"); p.key("c"); p.key("1");
  assert.deepEqual(p.S.actions.preflop, acts("UTG f, HJ c, CO r4"));   // over a limper: 3x plus 1bb (the advisor's iso size)
  p.key("u");
  assert.deepEqual(p.S.actions.preflop, acts("UTG f, HJ c"));
  p.key("Backspace");
  assert.deepEqual(p.S.actions.preflop, acts("UTG f"));
  p.key("a");
  assert.deepEqual(p.S.actions.preflop.at(-1), { pos: "HJ", action: "all-in" });
  const q = page(state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c") }));
  q.key("x");
  assert.deepEqual(q.S.actions.flop, acts("BB x"));
});

test("setup: players, stack, ante and straddle inputs; new hand", () => {
  const p = page(state({ hero: "CO", stack: 60, ante: 0.1, straddle: true, cards: ["Ah", "Kh"], preflop: acts("HJ f") }));
  p.run('document.getElementById("players").children[7].onclick()');   // 9 players
  let S = p.S;
  assert.equal(S.players, 9); assert.equal(S.hero, "BTN"); assert.equal(S.stack, 60); assert.equal(S.ante, 0.1); assert.equal(S.straddle, true);
  assert.deepEqual(S.cards.slice(0, 2), ["Ah", "Kh"]); assert.deepEqual(S.actions.preflop, []);
  assert.equal(p.val("POS").length, 9);
  p.run('document.getElementById("players").children[0].onclick()');   // heads-up: hero is the SB
  assert.equal(p.S.hero, "SB");
  p.byId("ante").onchange({ target: { value: "-1" } });
  assert.equal(p.S.ante, 0);
  p.byId("straddle").onchange({ target: { checked: false } });
  assert.equal(p.S.straddle, false);
  p.byId("stack").onchange({ target: { value: "" } });
  assert.equal(p.S.stack, 100);
  p.byId("stack").onchange({ target: { value: "35.5" } });
  assert.equal(p.S.stack, 35.5);
  p.act({ action: "fold" });
  p.byId("newhand").onclick();
  S = p.S;
  assert.equal(S.hero, "SB"); assert.equal(S.stack, 35.5); assert.deepEqual(S.cards, [null, null, null, null, null, null, null]);
  assert.deepEqual(S.actions.preflop, []);
});

// Bug: a negative stack typed into the stack box was taken as is (every seat "all-in" before acting).
test("the stack box refuses non-positive stacks", () => {
  const p = page(state({ cards: ["Ah", "Kh"] }));
  p.byId("stack").onchange({ target: { value: "-20" } });
  assert.ok(p.S.stack > 0);
  assert.equal(p.sim().toAct, "UTG");
});

test("grades: marks, rough solves and the EV given up", () => {
  const p = page(state({ cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") }));
  const g = (o) => p.val(`[gradeClass(${JSON.stringify(o)}), gradeMark(${JSON.stringify(o)})]`);
  assert.deepEqual(g({ f: 0.5 }), ["ok", "✓"]);
  assert.deepEqual(g({ f: 0.15 }), ["ok", "✓"]);
  assert.deepEqual(g({ f: 0.05 }), ["meh", "~"]);
  assert.deepEqual(g({ f: 0.01 }), ["bad", "✗"]);
  assert.deepEqual(g({ f: 0.01, r: 1 }), ["meh", "~"]);
  assert.deepEqual(g({ f: 0, l: 0 }), ["meh", "~"]);       // (off the mix, but it gave up nothing)
  assert.deepEqual(g({ f: 0, l: 0.01 }), ["bad", "✗"]);
  assert.deepEqual(g({ f: 0, l: null }), ["bad", "✗"]);    // (a preflop chart's: no EV to go by)
  assert.equal(p.run("sgn(-1.234)"), "−1.23");
  assert.equal(p.run("sgn(0)"), "0.00");
  // a decision is graded against the answer for this exact spot
  p.ctx.__res = JSON.stringify({ street: "flop", recommended: "bet 33%", solver: { exploitability_pct: 12, iterations: 50, solve_seconds: 2, active_combos: 300, tree_actions: ["check", "bet 33%", "bet 75%"] },
    actions: [{ action: "check", freq: 0.3, ev_bb: 4.0 }, { action: "bet 33%", amount_bb: 1.8, freq: 0.6, ev_bb: 4.25 }, { action: "bet 75%", amount_bb: 4.1, freq: 0.1, ev_bb: 4.2 }] });
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.deepEqual(p.val('gradeFor({ action: "bet", to: 4 })'), { m: "bet 75%", f: 0.1, b: "bet 33%", bf: 0.6, l: 0.05, e: 4.2, o: 4.25, r: 1 });
  assert.deepEqual(p.val('gradeFor({ action: "check" })'), { m: "check", f: 0.3, b: "bet 33%", bf: 0.6, l: 0.25, e: 4, o: 4.25, r: 1 });
  // but not when the answer is for another spot
  p.run("resultKey = 'another spot'");
  assert.equal(p.val('gradeFor({ action: "check" })'), null);
});

test("a solved 3-way pot is shown and graded like a heads-up solve; the betting rules' answer isn't graded", () => {
  const st = state({ hero: "BB", cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO r2.5, BTN c, SB f, BB c") });
  const p = page(st);
  assert.equal(p.sim().toAct, "BB");
  const res = { street: "flop", hero: "BB", multiway: true, solved: true, opponents: ["CO", "BTN"], recommended: "check",
    solver: { exploitability_pct: 0.8, iterations: 120, solve_seconds: 20, active_combos: 600, tree_actions: ["check", "bet 33%"] },
    actions: [{ action: "check", freq: 0.7, ev_bb: 3.1 }, { action: "bet 33%", amount_bb: 2.5, freq: 0.3, ev_bb: 3.0 }], notes: [] };
  p.ctx.__res = JSON.stringify(res);
  p.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  const html = p.byId("result").innerHTML;
  assert.ok(html.includes("3 players in · 3-way solve"));
  assert.ok(html.includes("Solver mix") && !html.includes("heuristic, not a solve"));
  assert.deepEqual(p.val('gradeFor({ action: "check" })'), { m: "check", f: 0.7, b: "check", bf: 0.7, l: 0, e: 3.1, o: 3.1 });
  // the betting rules (4+ players, or a 3-way spot the solve doesn't take) stay ungraded
  const q = page(st);
  q.ctx.__res = JSON.stringify({ ...res, solved: undefined, equity_vs_field: 0.4, recommended: "check",
    actions: [{ action: "check", freq: 1, ev_bb: 3.1 }, { action: "bet 33%", amount_bb: 2.5, freq: 0, ev_bb: 3.0 }] });
  q.run("renderResult(JSON.parse(__res), JSON.stringify(payload()))");
  assert.ok(q.byId("result").innerHTML.includes("heuristic, not a solve"));
  assert.equal(q.val('gradeFor({ action: "check" })'), null);
});

test("convergence labels", () => {
  const p = loadPage();
  const c = (x) => p.val(`convergence({ solver: { exploitability_pct: ${x} } })`);
  assert.equal(c(1).cls, "ok"); assert.equal(c(3).cls, "meh"); assert.equal(c(9.9).cls, "meh"); assert.equal(c(10).cls, "bad");
  assert.equal(p.val("convergence({})"), null);
  // a solve that ran short (solver.approximate: gto/advisor.py _flag_short), under 3% all the same
  const a = p.val("convergence({ solver: { exploitability_pct: 2.4, approximate: true, target_pct: 0.2 } })");
  assert.deepEqual([a.cls, a.label, a.target], ["bad", "approximate", 0.2]);
  assert.match(p.val("convTip(" + JSON.stringify(a) + ")"), /Answers here are 0.2% or less: this solve ran short/);
  assert.match(p.val("convTip(convergence({ solver: { exploitability_pct: 1 } }))"), /Under 3% is close to a full solver/);
});

test("bet sizes are shaded light to dark by size", () => {
  const p = loadPage();
  p.run('setColors(["check", "bet 125%", "bet 75%", "bet 33%", "all-in"])');
  assert.deepEqual(p.val('["bet 33%", "bet 75%", "bet 125%", "all-in", "fold", "call", "check"].map(COLORS)'),
    ["var(--agg0)", "var(--agg1)", "var(--agg2)", "var(--allin)", "var(--fold)", "var(--passive)", "var(--passive)"]);
  p.run('setColors(["fold", "call", "raise 2.5x", "raise 4x"])');
  assert.deepEqual(p.val('["raise 2.5x", "raise 4x"].map(COLORS)'), ["var(--agg0)", "var(--agg2)"]);
  p.run('setColors(["check", "bet 50%"])');
  assert.equal(p.run('COLORS("bet 50%")'), "var(--agg1)");
  assert.equal(p.run('gridClass(0, 0) + gridClass(0, 1) + gridClass(1, 0)'), "AAAKsAKo");
});
