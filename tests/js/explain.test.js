// Practice mode's mistake explanations (gto/static/practice.js): Copy for AI (the decision as a prompt) and
// Explain with AI (the Claude Messages API, streamed, with the user's own key), against a stubbed solver and fetch.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { loadPage } = require("./page.js");

const PRACTICE = fs.readFileSync(path.join(__dirname, "..", "..", "gto", "static", "practice.js"), "utf8");
const flush = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); };
const ADVICE = { street: "flop", hero: "BTN", hand: "AKs", recommended: "call",
  actions: [{ action: "fold", freq: 0.2, ev_bb: 0 }, { action: "check", freq: 0.3, ev_bb: 0.4 }, { action: "call", freq: 0.5, ev_bb: 1 }] };
const API = "https://api.anthropic.com/v1/messages";
const KEY = "sk-ant-api03-obviously-fake-key-for-tests";
const PROXY = "https://gto-explain.example.workers.dev/explain";

// a practice page at your first decision; fetches to the API go to api(url, init) (recorded in p.apiCalls)
async function atDecision(storage = {}, api = null, { proxy = "" } = {}) {
  const p = loadPage({ storage });
  await p.ready();
  p.run("Math.random = (() => { let s = 11; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();");
  p.ctx.TextDecoder = TextDecoder;
  p.apiCalls = [];
  const own = p.ctx.fetch;
  p.ctx.fetch = (url, init) => String(url).startsWith("https://api.anthropic.com") || (proxy && String(url) === proxy) ? (p.apiCalls.push({ url: String(url), init }), api(url, init)) : own(url, init);
  // (EXPLAIN_PROXY set to the test's proxy, or "" for your own key only, as before the worker's deploy)
  const src = PRACTICE.replace(/const EXPLAIN_PROXY = "[^"]*";/, `const EXPLAIN_PROXY = ${JSON.stringify(proxy)};`);
  assert.ok(src.includes(`const EXPLAIN_PROXY = ${JSON.stringify(proxy)};`), "EXPLAIN_PROXY in practice.js");
  vm.runInContext(src, p.ctx, { filename: "practice.js" });
  p.run("Practice.set(true)");
  const done = new Set();
  for (let i = 0; i < 50 && !(p.sim().toAct === p.S.hero && p.val("resultKey === JSON.stringify(payload())")); i++) {
    await flush();
    for (const m of p.posted) {
      if (m.id == null || done.has(m.id)) continue;
      done.add(m.id);
      p.workers.at(-1).onmessage({ data: { id: m.id, result: m.play ? { pos: m.play.seat, action: "call" } : ADVICE } });
    }
    await flush();
  }
  assert.equal(p.sim().toAct, p.S.hero);
  return p;
}
const lastKey = (p) => p.val("Practice.ai.keyOf(Practice.state.hand.log.at(-1))");
const html = (p) => { p.run("renderAll()"); return p.byId("practiceBar").innerHTML; };
// an SSE response whose body arrives in the given pieces (split anywhere, even inside an event)
function sse(pieces) {
  const enc = new TextEncoder(); let i = 0;
  return { ok: true, status: 200, body: { getReader: () => ({ read: async () => i < pieces.length ? { done: false, value: enc.encode(pieces[i++]) } : { done: true } }) } };
}
const ev = (type, data) => `event: ${type}\r\ndata: ${JSON.stringify({ type, ...data })}\r\n\r\n`;

test("Copy for AI: the decision as a prompt with the game, the action, the solver's mix and EVs, the verdict and the ask", async () => {
  const p = await atDecision();
  const S = p.S, hole = S.cards.slice(0, 2), seat = S.hero, before = S.actions.preflop.filter(a => a.pos !== S.hero);
  assert.equal(p.act({ action: "fold" }), null);   // in the mix (20%) but 1bb worse than call: Good, 1.00bb lost
  const k = lastKey(p);
  const text = p.val(`Practice.ai.prompt(Practice.state.hand.log.at(-1))`);
  assert.match(text, /^I'm practising no-limit hold'em against a GTO solver/);
  assert.match(text, /GAME\n- No-limit hold'em cash game, 6-handed, 100bb stacks\n- Blinds 0\.5\/1bb, no rake\n/);
  assert.ok(text.includes(`- Hero: ${seat} with ${hole[0]} ${hole[1]}`), text);
  assert.match(text, /ACTION \(sizes in bb; the pot at the start of each street\)\nPreflop \(pot 1\.5bb\): /);
  for (const a of before) assert.ok(text.includes(`${a.pos} ${a.action === "call" ? "calls 1bb" : a.action + "s"}`), a.pos);
  assert.match(text, /DECISION\n- Preflop, Hero to act\. Pot [\d.]+bb, [\d.]+bb to call\. Stacks behind: /);
  assert.match(text, /- I chose: fold\n/);
  assert.match(text, /  - fold: 20%, EV 0\.00  <- my move\n  - check: 30%, EV \+0\.40\n  - call: 50%, EV \+1\.00\n/);
  assert.match(text, /- The solver's top action: call \(50%\)\n- EV lost against the best action: 1bb\n- Verdict: Good\. The solver mixes this in 20% of the time; its top choice is call \(50%\)\./);
  assert.match(text, /Why is the solver's play better than mine here.*ground truth.*one-sentence verdict, two or three short reasons \(ranges, equity, blockers, board texture or bet sizing.*one takeaway.*Under 70 words\.$/s);
  // the buttons are in Details; a click copies it and says so
  assert.match(html(p), new RegExp(`<button class="small" data-ai="copy" data-k="${k}"[^>]*>Copy for AI</button><button class="small" data-ai="explain" data-k="${k}" aria-expanded="false"[^>]*>Explain with AI</button>`));
  await p.run(`Practice.ai.click("copy", ${JSON.stringify(k)})`);
  assert.equal(p.clipboard.text, text);
  assert.match(p.byId("toast").innerHTML, /Copied/);
  assert.equal(p.apiCalls.length, 0);
});

test("Explain with AI with no key saved opens the key form; a bad key is refused, a good one is kept and used", async () => {
  const p = await atDecision({}, async () => sse([ev("message_stop", {})]));
  p.act({ action: "call" });
  const k = lastKey(p);
  await p.run(`Practice.ai.click("explain", ${JSON.stringify(k)})`);
  let h = html(p);
  assert.match(h, /<input id="pxKey" class="cx-key" type="password"/);
  assert.match(h, /kept only in this browser \(localStorage\) and sent only to api\.anthropic\.com/);
  assert.match(h, /href="https:\/\/console\.anthropic\.com\/settings\/keys" target="_blank" rel="noopener noreferrer">Get a key</);
  assert.match(h, /data-ai="explain"[^>]*aria-expanded="true"[^>]*>Hide explanation</);
  assert.equal(p.apiCalls.length, 0);
  await p.run(`Practice.ai.click("save", ${JSON.stringify(k)}, "hello")`);
  assert.match(html(p), /doesn't look like an Anthropic API key/);
  assert.equal(p.storage.has("gto-anthropic-key"), false);
  await p.run(`Practice.ai.click("save", ${JSON.stringify(k)}, " ${KEY} ")`);
  assert.equal(p.storage.get("gto-anthropic-key"), KEY);
  assert.equal(p.apiCalls.length, 1);
  assert.doesNotMatch(html(p), /cx-key/);
  // Cancel closes the form
  const q = await atDecision();
  q.act({ action: "call" });
  const k2 = lastKey(q);
  await q.run(`Practice.ai.click("explain", ${JSON.stringify(k2)})`);
  await q.run(`Practice.ai.click("cancel", ${JSON.stringify(k2)})`);
  assert.doesNotMatch(html(q), /cx-key/);
});

test("Explain with AI streams Claude's answer into the card, safely, once per decision; Forget key", async () => {
  const pieces = [ev("message_start", { message: { id: "msg_1" } }) + ev("content_block_start", { index: 0, content_block: { type: "text", text: "" } }),
    ev("content_block_delta", { index: 0, delta: { type: "text_delta", text: "Calling keeps **weak hands** in" } }).slice(0, 40),
    ev("content_block_delta", { index: 0, delta: { type: "text_delta", text: "Calling keeps **weak hands** in" } }).slice(40),
    ev("content_block_delta", { index: 0, delta: { type: "text_delta", text: " <script>x</script>.\n\nTakeaway: defend wide." } }),
    ev("content_block_stop", { index: 0 }) + ev("message_delta", { delta: { stop_reason: "end_turn" }, usage: { output_tokens: 20 } }) + ev("message_stop", {})];
  const p = await atDecision({ "gto-anthropic-key": KEY }, async () => sse(pieces));
  p.act({ action: "fold" });
  const k = lastKey(p);
  await p.run(`Practice.ai.click("explain", ${JSON.stringify(k)})`);
  await flush();
  assert.equal(p.apiCalls.length, 1);
  const { url, init } = p.apiCalls[0], body = JSON.parse(init.body);
  assert.equal(url, API);   // (the key is never in the URL)
  assert.equal(init.method, "POST");
  assert.deepEqual({ ...init.headers }, { "content-type": "application/json", "x-api-key": KEY, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" });
  assert.equal(body.model, "claude-sonnet-5-5");
  assert.equal(body.max_tokens, 700);
  assert.deepEqual(body.thinking, { type: "between_tools" });   // (no extended thinking eating the 700)
  assert.equal(body.stream, true);
  assert.match(body.system, /ground truth/);
  assert.equal(body.messages[0].content, p.val("Practice.ai.prompt(Practice.state.hand.log.at(-1))"));
  const c = p.val(`Practice.ai.state.cache[${JSON.stringify(k)}]`);
  assert.equal(c.s, "done");
  assert.equal(c.t, "Calling keeps **weak hands** in <script>x</script>.\n\nTakeaway: defend wide.");
  const h = html(p);
  assert.ok(h.includes('<div class="cx-ai" aria-live="polite"><p>Calling keeps <b>weak hands</b> in &lt;script&gt;x&lt;/script&gt;.</p><p>Takeaway: defend wide.</p>'), h);
  assert.ok(!h.includes(KEY));
  assert.match(h, /Claude Sonnet 5\.5/);
  // hidden and shown again: from memory, no new request
  await p.run(`Practice.ai.click("explain", ${JSON.stringify(k)})`);
  assert.doesNotMatch(html(p), /cx-ai"/);
  await p.run(`Practice.ai.click("explain", ${JSON.stringify(k)})`);
  assert.match(html(p), /Takeaway: defend wide/);
  assert.equal(p.apiCalls.length, 1);
  // Forget key
  assert.match(html(p), /data-ai="forget"[^>]*>Forget key</);
  await p.run(`Practice.ai.click("forget", ${JSON.stringify(k)})`);
  assert.equal(p.storage.has("gto-anthropic-key"), false);
  assert.doesNotMatch(html(p), /Forget key/);
});

test("Explain with AI: a refused key (401), a rate limit (429) and no network say so in the card, with Try again", async () => {
  let status = 401;
  const p = await atDecision({ "gto-anthropic-key": KEY }, async () => status ? { ok: false, status, json: async () => ({ type: "error", error: { type: "authentication_error", message: "invalid x-api-key" } }) } : Promise.reject(new TypeError("Failed to fetch")));
  p.act({ action: "call" });
  const k = JSON.stringify(lastKey(p));
  await p.run(`Practice.ai.click("explain", ${k})`);
  let h = html(p);
  assert.match(h, /<div class="cx-ai err"[^>]*>.*Anthropic didn't accept this API key \(401\)/s);
  assert.match(h, /data-ai="retry"[^>]*>Try again<.*data-ai="change"[^>]*>Enter another key</s);
  assert.ok(!h.includes(KEY));
  await p.run(`Practice.ai.click("change", ${k})`);
  assert.match(html(p), /class="cx-key"/);
  status = 429;
  await p.run(`Practice.ai.click("save", ${k}, ${JSON.stringify(KEY)})`);
  h = html(p);
  assert.match(h, /out of credit \(429\)/);
  assert.doesNotMatch(h, /Enter another key/);
  status = 0;
  await p.run(`Practice.ai.click("retry", ${k})`);
  assert.match(html(p), /Couldn't reach api\.anthropic\.com/);
  assert.equal(p.apiCalls.length, 3);
  // a mid-stream error event keeps what came and says it stopped
  status = null;
  const q = await atDecision({ "gto-anthropic-key": KEY }, async () => sse([ev("content_block_delta", { delta: { type: "text_delta", text: "Partial" } }), ev("error", { error: { type: "overloaded_error", message: "Overloaded" } })]));
  q.act({ action: "call" });
  await q.run(`Practice.ai.click("explain", ${JSON.stringify(lastKey(q))})`);
  assert.match(html(q), /<p>Partial<\/p><p class="err">The answer stopped early: Overloaded<\/p>/);
});

test("the prompt on a later street: the board by street, each bet with the pot, what's to call; Review mistakes carries it", async () => {
  const p = await atDecision();
  const h = p.val("Practice.state.hand"), S = p.S;
  assert.equal(p.act({ action: "call" }), null);
  const e = p.val("Practice.state.review.at(-1)");
  assert.equal(p.val("Practice.ai.keyOf(Practice.state.review.at(-1))"), lastKey(p));   // (the same decision, the same explanation)
  assert.equal(e.ctx.hand, `Hero: ${S.hero} with ${h.hole[0]} ${h.hole[1]}`);
  assert.ok(p.val("Practice.ai.prompt(Practice.state.review.at(-1))").includes(e.ctx.at));
  // heads-up with an ante and rake, on the turn facing a bet (the page's state set directly, as the hand would have it)
  const ctx = p.val(`(() => { S = { ...fresh(2), hero: "BB", stack: 100, ante: 0.1, rake: 5, rakeCap: 3, cards: ["Ah", "Qh", "Ks", "7d", "2c", "9h", null],
    actions: { preflop: [{ pos: "SB", action: "raise", to: 2.5 }, { pos: "BB", action: "call" }], flop: [{ pos: "BB", action: "check" }, { pos: "SB", action: "bet", to: 1.8 }, { pos: "BB", action: "call" }],
      turn: [{ pos: "BB", action: "check" }, { pos: "SB", action: "bet", to: 6 }], river: [] } }; setPositions(2); return Practice.ai.spot("turn"); })()`);
  assert.deepEqual(ctx.game.slice(0, 2), ["No-limit hold'em cash game, 2-handed, 100bb stacks", "Blinds 0.5/1bb, 0.1bb ante each, 5% rake (cap 3bb) (pots that see a flop)"]);
  assert.equal(ctx.hand, "Hero: BB with Ah Qh");
  assert.deepEqual(ctx.action, [
    "Preflop (pot 1.7bb): BTN/SB raises to 2.5bb, Hero (BB) calls 2.5bb",
    "Flop Ks 7d 2c (pot 5.2bb): Hero (BB) checks, BTN/SB bets 1.8bb (35% of 5.2bb), Hero (BB) calls 1.8bb",
    "Turn 9h (pot 8.8bb): Hero (BB) checks, BTN/SB bets 6bb (68% of 8.8bb)"]);
  assert.equal(ctx.at, "Turn, Hero to act. Pot 14.8bb, 6bb to call. Stacks behind: Hero (BB) 95.6bb, BTN/SB 89.6bb");
  const text = p.val(`Practice.ai.prompt({ street: "turn", rec: { action: "call" }, g: null, mix: null, ctx: ${JSON.stringify(ctx)} })`);
  assert.match(text, /- I chose: call 6bb\n- The solver doesn't solve this multiway spot.*\n- Verdict: Not graded\n/s);
});

test("EXPLAIN_PROXY is the deployed worker's /explain, so explanations are free by default; no API key in the page", () => {
  assert.match(PRACTICE, /\n  const EXPLAIN_PROXY = "https:\/\/gto-explain\.[a-z0-9-]+\.workers\.dev\/explain";\n/);
  assert.ok(!/sk-ant-api\d/.test(PRACTICE), "no API key in practice.js");
});

test("with the proxy: Explain with AI streams from it with no key, sending only the prompt; Use my own key and back", async () => {
  const pieces = [ev("content_block_delta", { index: 0, delta: { type: "text_delta", text: "Free **answer**." } }),
    ev("message_delta", { delta: { stop_reason: "end_turn" } }) + ev("message_stop", {})];
  const p = await atDecision({}, async () => sse(pieces), { proxy: PROXY });
  p.act({ action: "fold" });
  const k = JSON.stringify(lastKey(p));
  assert.match(html(p), /data-ai="explain"[^>]*title="Claude explains this decision">Explain with AI</);
  await p.run(`Practice.ai.click("explain", ${k})`);
  await flush();
  assert.equal(p.apiCalls.length, 1);
  const { url, init } = p.apiCalls[0];
  assert.equal(url, PROXY);
  assert.equal(init.method, "POST");
  assert.deepEqual({ ...init.headers }, { "content-type": "application/json" });
  assert.deepEqual(JSON.parse(init.body), { prompt: p.val("Practice.ai.prompt(Practice.state.hand.log.at(-1))") });
  let h = html(p);
  assert.doesNotMatch(h, /cx-key/);
  assert.match(h, /<p>Free <b>answer<\/b>\.<\/p>/);
  assert.match(h, /data-ai="own"[^>]*>Use my own key</);
  assert.doesNotMatch(h, /Forget key|Use free explanations/);
  // Use my own key with none saved: the key form; saving it switches to the API with the key, and it's remembered
  await p.run(`Practice.ai.click("own", ${k})`);
  assert.match(html(p), /class="cx-key"/);
  assert.equal(p.apiCalls.length, 1);
  await p.run(`Practice.ai.click("save", ${k}, ${JSON.stringify(KEY)})`);
  await flush();
  assert.equal(p.apiCalls.length, 2);
  assert.equal(p.apiCalls[1].url, API);
  assert.equal(p.apiCalls[1].init.headers["x-api-key"], KEY);
  assert.equal(p.storage.get("gto-explain-own"), "1");
  h = html(p);
  assert.match(h, /data-ai="free"[^>]*>Use free explanations<.*data-ai="forget"[^>]*>Forget key</s);
  // back to the free ones (the key stays saved, unused)
  await p.run(`Practice.ai.click("free", ${k})`);
  await flush();
  assert.equal(p.apiCalls.at(-1).url, PROXY);
  assert.equal(p.storage.has("gto-explain-own"), false);
  assert.equal(p.storage.get("gto-anthropic-key"), KEY);
  // a key saved but not chosen doesn't take over; Use my own key with one saved uses it straight away
  const q = await atDecision({ "gto-anthropic-key": KEY }, async () => sse(pieces), { proxy: PROXY });
  q.act({ action: "call" });
  const k2 = JSON.stringify(lastKey(q));
  await q.run(`Practice.ai.click("explain", ${k2})`);
  await flush();
  assert.equal(q.apiCalls.at(-1).url, PROXY);
  await q.run(`Practice.ai.click("own", ${k2})`);
  await flush();
  assert.equal(q.apiCalls.at(-1).url, API);
  assert.doesNotMatch(html(q), /cx-key/);
  // chosen before (gto-explain-own): the key from the start
  const r = await atDecision({ "gto-anthropic-key": KEY, "gto-explain-own": "1" }, async () => sse(pieces), { proxy: PROXY });
  r.act({ action: "call" });
  await r.run(`Practice.ai.click("explain", ${JSON.stringify(lastKey(r))})`);
  await flush();
  assert.equal(r.apiCalls.length, 1);
  assert.equal(r.apiCalls[0].url, API);
});

test("with the proxy: the daily limit (429) offers your own key, a minute's limit says wait, 403 and no network say so", async () => {
  let answer;
  const p = await atDecision({}, async () => answer(), { proxy: PROXY });
  p.act({ action: "call" });
  const k = JSON.stringify(lastKey(p));
  const err = (status, body) => () => ({ ok: false, status, json: async () => body });
  answer = err(429, { error: "Daily free explanations used up. Add your own key, or come back tomorrow.", retryAfter: 3600, scope: "day" });
  await p.run(`Practice.ai.click("explain", ${k})`);
  let h = html(p);
  assert.match(h, /<div class="cx-ai err"[^>]*>.*Daily free explanations used up — add your own key/s);
  assert.match(h, /data-ai="own"[^>]*>Add your own key</);
  assert.doesNotMatch(h, /data-ai="retry"|Use my own key/);
  await p.run(`Practice.ai.click("own", ${k})`);
  assert.match(html(p), /class="cx-key"/);
  answer = () => sse([ev("content_block_delta", { delta: { type: "text_delta", text: "With my key." } })]);
  await p.run(`Practice.ai.click("save", ${k}, ${JSON.stringify(KEY)})`);
  await flush();
  assert.equal(p.apiCalls.at(-1).url, API);
  assert.match(html(p), /<p>With my key\.<\/p>/);
  // the global cap, the per-minute limit, a refused origin, a proxy error and no network
  const q = await atDecision({}, async () => answer(), { proxy: PROXY });
  q.act({ action: "call" });
  const k2 = JSON.stringify(lastKey(q));
  const cases = [
    [err(429, { error: "x", retryAfter: 100, scope: "global" }), /Today's free explanations are used up — add your own key/, true],
    [err(429, { error: "x", retryAfter: 60, scope: "minute" }), /Too many explanations in a minute\. Try again in 60s\./, false],
    [err(403, { error: "This origin may not use the explanation service." }), /can't use the free explanations \(403\) — add your own key/, true],
    [err(503, { error: "Claude is busy right now. Try again in a moment.", retryAfter: 30 }), /Claude is busy right now\. Try again in a moment\./, false],
    [err(502, null), /The explanation service failed \(502\)\./, false],
    [() => Promise.reject(new TypeError("Failed to fetch")), /Couldn't reach the explanation service/, false]];
  for (const [a, re, quota] of cases) {
    answer = a;
    await q.run(`Practice.ai.click("retry", ${k2})`);
    h = html(q);
    assert.match(h, re);
    assert.equal(/data-ai="retry"/.test(h), !quota, String(re));
    assert.equal(/>Add your own key</.test(h), quota, String(re));
    assert.doesNotMatch(h, /Enter another key/);
  }
  assert.ok(q.apiCalls.every(c => c.url === PROXY));
});
