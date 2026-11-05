// Live stakes: a select (big blinds / $1/$2 / $1/$3 / $2/$5 / $5/$10 / custom) in the setup panel, kept in the state
// (S.stakes {sb, bb} in dollars) and remembered for new hands ("gto-stakes": store.js, or localStorage without it). With stakes the stack,
// ante, per-seat stacks, bets and the rake's cap are typed in dollars and shown in dollars beside the big blinds; the
// page keeps every amount in big blinds, so what the solver is sent doesn't change.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { loadPage, state, acts } = require("./page.js");

const PRE = { cards: ["Ah", "Kh"], preflop: acts("UTG f, HJ r2.5, CO f") };   // BTN (you) to act, facing 2.5bb
const FLOP = { cards: ["Ah", "Kh", "Kd", "7s", "2c"], preflop: acts("UTG f, HJ f, CO f, BTN r2.5, SB f, BB c"), flop: acts("BB x") };
const pick = (p, v) => p.byId("stakes").onchange({ target: { value: v } });
const custom = (p) => p.byId("actionArea").children.find(e => e.id === "customBox");
const html = (p) => [...p.byId("actionArea").walk()].map(e => e.innerHTML).join(" | ");

test("no stakes by default: big blinds, as before", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  assert.equal(p.byId("stakes").value, "off");
  assert.equal(p.byId("stakesCustom").hidden, true);
  assert.equal(p.byId("stackPre").hidden, true);
  assert.equal(p.byId("stackUnit").textContent, "bb");
  assert.equal(p.byId("straddleTxt").textContent, "straddle");
  assert.equal(p.byId("stack").value, 100);
  assert.equal("stakes" in p.S, false);
  assert.match(p.byId("felt").children.at(-1).innerHTML, /pot 5\.5 bb</);
});

test("a preset: saved, remembered, the stack and ante in dollars, the payload unchanged", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state({ ...FLOP, ante: 0.2, straddle: true }) };
  const before = JSON.stringify(p.payload());
  pick(p, "2-5");
  assert.deepEqual(p.S.stakes, { sb: 2, bb: 5 });
  assert.deepEqual(JSON.parse(p.storage.get("gto-stakes")), { sb: 2, bb: 5 });
  assert.equal(JSON.parse(p.storage.get("handreview")).stakes.bb, 5);
  assert.equal(JSON.stringify(p.payload()), before);   // (stakes change how amounts are typed and shown, not the spot)
  assert.equal(p.byId("stakes").value, "2-5");
  assert.equal(p.byId("stack").value, "500");
  assert.equal(p.byId("stackPre").hidden, false);
  assert.equal(p.byId("stackUnit").textContent, "= 100bb");
  assert.equal(p.byId("ante").value, "1");
  assert.equal(p.byId("anteUnit").textContent, "= 0.2bb");
  assert.equal(p.byId("straddleTxt").textContent, "$10 straddle");
  assert.equal(p.byId("stakesNote").textContent, "small blind played as $2.50");
  // typed in dollars ("$" and "," allowed), kept in big blinds
  p.byId("stack").value = "$1,250"; p.byId("stack").onchange({ target: p.byId("stack") });
  assert.equal(p.S.stack, 250);
  assert.equal(p.byId("stack").value, "1250");
  p.byId("ante").onchange({ target: { value: "2.5" } });
  assert.equal(p.S.ante, 0.5);
  assert.equal(p.payload().stack, 250); assert.equal(p.payload().ante, 0.5);
  // the summary names the stakes and the dollars
  assert.match(p.byId("setupSum").innerHTML, /<b>\$2\/\$5<\/b>.*<b>250bb<\/b> \(\$1,250\).*0\.5bb \(\$2\.50\) ante/);
  // $1/$2: the small blind is half the big one, no note
  pick(p, "1-2");
  assert.equal(p.byId("stakesNote").textContent, "");
  assert.equal(p.byId("stack").value, "500");
  // a new hand, another table size and a fresh page keep the stakes
  p.click("newhand");
  assert.deepEqual(p.S.stakes, { sb: 1, bb: 2 });
  const q = loadPage({ storage: { "gto-stakes": JSON.stringify({ sb: 5, bb: 10 }) } });
  await q.ready();
  assert.deepEqual(q.S.stakes, { sb: 5, bb: 10 });
  assert.equal(q.byId("stack").value, "1000");
  // back to big blinds
  pick(p, "off");
  assert.equal("stakes" in p.S, false);
  assert.equal(p.storage.get("gto-stakes"), "null");
  assert.equal(p.byId("stack").value, 250);
  assert.equal(p.byId("stackUnit").textContent, "bb");
});

test("custom stakes: from the current ones, bad entries ignored", async () => {
  const p = loadPage();
  await p.ready();
  p.S = state(FLOP);
  pick(p, "custom");
  assert.deepEqual(p.S.stakes, { sb: 1, bb: 2 });
  assert.equal(p.byId("stakes").value, "custom");
  assert.equal(p.byId("stakesCustom").hidden, false);
  p.byId("stakeSb").value = "3"; p.byId("stakeBb").value = "6"; p.byId("stakeBb").onchange();
  assert.deepEqual(p.S.stakes, { sb: 3, bb: 6 });
  assert.equal(p.byId("stakes").value, "custom");   // (stays open while it's being edited)
  p.byId("stakeSb").value = "10"; p.byId("stakeSb").onchange();   // a small blind bigger than the big one
  assert.deepEqual(p.S.stakes, { sb: 3, bb: 6 });
  p.byId("stakeBb").value = "0"; p.byId("stakeBb").onchange();
  assert.deepEqual(p.S.stakes, { sb: 3, bb: 6 });
  // a saved or linked hand with bad stakes plays in big blinds
  p.S = { ...state(FLOP), stakes: { sb: 5, bb: 2 } };
  p.run("S = cleanState(S)");
  assert.equal("stakes" in p.S, false);
});

test("bets in dollars: the buttons, the prompt, a typed size, the hand panel", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state(PRE), stakes: { sb: 2, bb: 5 } };
  const prompt = p.byId("actionArea").children[0].innerHTML;
  assert.match(prompt, /pot 4 bb \(\$20\) · 2\.5 \(\$12\.50\) to call/);
  assert.match(html(p), /Call<small>\$12\.50 · 2\.5bb<\/small>/);
  assert.match(html(p), /All-in \$500/);
  // the default raise sizes, labelled in dollars with the big blinds under them
  const raise = p.buttons().find(b => /^Raise to \$/.test(b.innerHTML));
  assert.ok(raise, html(p));
  assert.match(raise.innerHTML, /^Raise to \$\d[\d.,]*<small>\d[\d.]*bb · /);
  // the typed size is in dollars, rounded to the page's half big blind ($2.50 steps here)
  assert.match(custom(p).innerHTML, /<input id="customAmt" inputmode="decimal"[^>]* placeholder="Raise to, \$"/);
  assert.equal(p.val("amountIn('$40')"), 8);
  assert.equal(p.val("amountIn('41')"), 8.2);
  p.byId("customAmt").value = "40";
  custom(p).children.find(e => e.tagName === "BUTTON").onclick();
  assert.deepEqual(p.S.actions.preflop.at(-1), { pos: "BTN", action: "raise", to: 8 });
  // the hand panel and the felt
  const tl = [...p.byId("timeline").walk()].map(e => e.innerHTML).join(" ");
  assert.match(tl, /HJ raises 2\.5 \(\$12\.50\)/);
  assert.match(tl, /BTN raises 8 \(\$40\)/);
  const felt = p.byId("felt").children;
  assert.match(felt.at(-1).innerHTML, /pot 12 bb \(\$60\)/);
  assert.ok(felt.some(e => /<span class="amt">\$40</.test(e.innerHTML)));
  // in big blinds, the same spot reads as before
  p.S = state(PRE);
  assert.match(p.byId("actionArea").children[0].innerHTML, /pot 4 bb · 2\.5 to call/);
  assert.match(html(p), /Call<small>2\.5 bb<\/small>/);
  p.byId("customAmt").value = "8";
  custom(p).children.find(e => e.tagName === "BUTTON").onclick();
  assert.deepEqual(p.S.actions.preflop.at(-1), { pos: "BTN", action: "raise", to: 8 });
});

test("per-seat stacks in dollars", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state(FLOP), stakes: { sb: 1, bb: 3 } };
  p.run("setSeatStack('CO', '$150')");
  assert.equal(p.S.stacks.CO, 50);
  assert.equal(p.payload().stacks.CO, 50);
});

test("the live rake preset: 10% capped at $5, kept at $5 when the stakes change", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state(FLOP), stakes: { sb: 2, bb: 5 } };
  p.byId("rake").onchange({ target: { value: "10-live" } });
  assert.equal(p.S.rake, 10); assert.equal(p.S.rakeCap, 1);
  assert.deepEqual([p.payload().rake, p.payload().rake_cap], [0.1, 1]);
  assert.equal(p.byId("rake").value, "10-live");
  pick(p, "1-2");
  assert.equal(p.S.rakeCap, 2.5);
  assert.equal(p.byId("rake").value, "10-live");
  assert.match(p.run("RAKE_TEXT()"), /^10% rake \(cap \$5\)$/);
  // a custom cap is typed in dollars
  p.byId("rake").onchange({ target: { value: "custom" } });
  assert.equal(p.byId("rakeCap").value, "5");
  assert.equal(p.byId("rakeCapUnit").textContent, "2.5bb");
  p.byId("rakeCap").onchange({ target: { value: "4" } });
  assert.equal(p.S.rakeCap, 2);
  // in big blinds the preset isn't offered, and a cap stays where it is
  pick(p, "off");
  assert.equal(p.S.rakeCap, 2);
  assert.equal(p.val("RAKE_PRESETS['10-live']()"), null);
});

test("share links carry the stakes; table size changes keep them", async () => {
  const a = loadPage();
  await a.ready();
  a.S = { ...state(FLOP), stakes: { sb: 1, bb: 3 } };
  const b = loadPage({ hash: a.run("shareLink()").replace(/^[^#]*/, "") });
  await b.ready();
  assert.deepEqual(b.S.stakes, { sb: 1, bb: 3 });
  assert.equal(b.byId("stakes").value, "1-3");
  assert.equal(b.byId("stack").value, "300");
  const n9 = b.byId("players").children.find(e => e.textContent === 9);
  n9.onclick();
  assert.equal(b.S.players, 9);
  assert.deepEqual(b.S.stakes, { sb: 1, bb: 3 });
});

test("a tournament has no rake, whatever the Rake setting: payouts set, or an imported tournament hand", async () => {
  const p = loadPage();
  await p.ready();
  p.S = { ...state(FLOP), stakes: { sb: 2, bb: 5 }, rake: 10, rakeCap: 1 };
  assert.deepEqual([p.payload().rake, p.payload().rake_cap], [0.1, 1]);
  p.S = { ...state(FLOP), rake: 10, rakeCap: 1, payouts: [50, 30, 20] };
  assert.equal(p.payload().rake, undefined);
  assert.deepEqual(p.payload().payouts, [50, 30, 20]);
  p.S = { ...state(FLOP), rake: 10, rakeCap: 1, tournament: { id: "3141592653", name: "$22 Hot", level: 7 } };   // (hh.js's)
  assert.equal(p.payload().rake, undefined);
  assert.equal(p.payload().rake_cap, undefined);
});
