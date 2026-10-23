// Explain with AI proxy (Cloudflare Worker; the entry point is src/index.js). Practice mode's "Explain with AI" (gto/static/practice.js)
// POSTs {prompt} here; the worker builds the Claude Messages API request itself (fixed model, system prompt, token cap,
// streamed) with the ANTHROPIC_API_KEY secret and streams the Server-Sent Events straight back. The client chooses
// only the prompt text. Only the site's origins (ALLOWED_ORIGINS) may call it, and each IP and the whole proxy have
// limits: RATE_LIMITER (per IP per minute, the Workers Rate Limiting binding) and daily counters in the COUNTS KV
// namespace (per IP and in total), so the key's cost stays bounded.

export const UPSTREAM = "https://api.anthropic.com/v1/messages";
export const MODEL = "claude-sonnet-5-5";
// (with extended thinking off: thinking: between_tools. Sonnet 5.5 thinks by default, and its thinking counts toward
// max_tokens, so it could use up the 700 before the answer)
export const MAX_TOKENS = 700;
export const MAX_PROMPT = 8000;
// (the same coach prompt as practice.js's AI_SYSTEM, which the own-key path sends; test/index.test.js keeps them equal)
export const SYSTEM = "You coach no-limit hold'em players who are learning from a GTO solver. The user sends one decision from a practice hand, "
  + "with the solver's frequencies and EVs for their exact hand. Treat those numbers as ground truth: never contradict them or invent other "
  + "frequencies or EVs.\n\nReply in exactly this shape, 70 words at most in all:\n"
  + "**Verdict:** one sentence, at most 20 words: was their move right, and what does the solver prefer.\n"
  + "- Exactly three bullets, each one sentence of at most 15 words: the three reasons that matter most here (ranges, equity, blockers, board texture "
  + "or sizing; only those that matter).\n"
  + "**Takeaway:** one rule they can reuse, at most 15 words.\n\n"
  + "Plain words, numbers only from the hand. Take Hero's made hand, draws, outs and position from the HERO'S HAND lines: they are exact, "
    + "so never work them out yourself. Don't mention data the hand doesn't have (such as missing EVs). No preamble, no restating the hand, no headings or tables. With no solver "
  + "numbers, coach the spot from sound principles and say it isn't solved.";

const DEFAULT_ORIGINS = "https://www.michael-han.com,https://michael-han.com,http://localhost:*,http://127.0.0.1:*";

// the allowed origins: exact ones, or "scheme://host:*" for any port on that host
export function originAllowed(origin, list) {
  if (!origin) return false;
  for (const raw of String(list || DEFAULT_ORIGINS).split(",")) {
    const o = raw.trim();
    if (!o) continue;
    if (o.endsWith(":*")) {
      const base = o.slice(0, -2);
      if (origin === base || new RegExp("^" + base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ":\\d{1,5}$").test(origin)) return true;
    } else if (origin === o) return true;
  }
  return false;
}

const cors = (origin) => ({ "Access-Control-Allow-Origin": origin, "Vary": "Origin" });
function json(status, body, origin, extra = {}) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...(origin ? cors(origin) : {}), ...extra } });
}
const tooMany = (origin, error, retryAfter, scope) =>
  json(429, { error, retryAfter, scope }, origin, { "Retry-After": String(retryAfter) });

// seconds until the next UTC midnight (when the daily counters roll over)
function untilMidnight(now) {
  const d = new Date(now); d.setUTCHours(24, 0, 0, 0);
  return Math.max(1, Math.ceil((d - now) / 1000));
}
// the IP as a short hash (the KV keys never hold a raw address)
async function ipKey(ip) {
  const h = await crypto.subtle.digest("SHA-256", new TextEncoder().encode("gto-explain:" + ip));
  return [...new Uint8Array(h).slice(0, 12)].map(b => b.toString(16).padStart(2, "0")).join("");
}
const num = (v, d) => { const n = parseInt(v, 10); return Number.isFinite(n) && n >= 0 ? n : d; };

// an upstream error as a clean error for the page (never the upstream body or headers)
export function upstreamError(status) {
  if (status === 401 || status === 403) return { status: 502, error: "The explanation service is misconfigured (proxy misconfigured). Try again later." };
  if (status === 429 || status === 529 || status === 503 || status === 500 || status === 502 || status === 504)
    return { status: 503, error: "Claude is busy right now. Try again in a moment.", retryAfter: 30 };
  if (status === 400 || status === 413) return { status: 502, error: "Claude couldn't take this request." };
  return { status: 502, error: `The explanation service failed (${status}).` };
}

// the worker's fetch handler (src/index.js exports it as the module's default)
export const handler = {
  async fetch(request, env, ctx) {
    const url = new URL(request.url), origin = request.headers.get("Origin") || "";
    if (!originAllowed(origin, env.ALLOWED_ORIGINS)) return json(403, { error: "This origin may not use the explanation service." }, null);
    if (url.pathname !== "/explain") return json(404, { error: "Not found." }, origin);
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: { ...cors(origin), "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "content-type", "Access-Control-Max-Age": "86400" } });
    }
    if (request.method !== "POST") return json(405, { error: "Use POST." }, origin, { Allow: "POST, OPTIONS" });
    if (!/^application\/json\b/i.test(request.headers.get("content-type") || "")) return json(415, { error: "Send JSON." }, origin);
    let body;
    try { body = JSON.parse(await request.text()); } catch (e) { return json(400, { error: "The body isn't valid JSON." }, origin); }
    const prompt = body && typeof body === "object" ? body.prompt : undefined;
    if (typeof prompt !== "string" || !prompt.trim()) return json(400, { error: "Send {prompt: string}." }, origin);
    if (prompt.length > MAX_PROMPT) return json(413, { error: `The prompt is too long (at most ${MAX_PROMPT} characters).` }, origin);
    if (!env.ANTHROPIC_API_KEY) return json(502, { error: "The explanation service is misconfigured (proxy misconfigured). Try again later." }, origin);

    // limits: a minute's burst per IP, then the day's count per IP and in total (UTC days)
    const ip = request.headers.get("CF-Connecting-IP") || "unknown", id = await ipKey(ip);
    if (env.RATE_LIMITER) {
      const { success } = await env.RATE_LIMITER.limit({ key: id });
      if (!success) return tooMany(origin, "Too many explanations in a minute. Wait a moment and try again.", 60, "minute");
    }
    if (env.COUNTS) {
      const now = Date.now(), day = new Date(now).toISOString().slice(0, 10), wait = untilMidnight(now);
      const kIp = `ip:${day}:${id}`, kAll = `all:${day}`;
      const [nIp, nAll] = (await Promise.all([env.COUNTS.get(kIp), env.COUNTS.get(kAll)])).map(v => num(v, 0));
      if (nIp >= num(env.DAILY_PER_IP, 20)) return tooMany(origin, "Daily free explanations used up. Add your own key, or come back tomorrow.", wait, "day");
      if (nAll >= num(env.DAILY_GLOBAL, 1000)) return tooMany(origin, "Today's free explanations are used up for everyone. Add your own key, or come back tomorrow.", wait, "global");
      // (KV isn't atomic: a burst can overshoot by a few, which the per-minute limit keeps small)
      const ttl = { expirationTtl: 2 * 86400 };
      const put = Promise.all([env.COUNTS.put(kIp, String(nIp + 1), ttl), env.COUNTS.put(kAll, String(nAll + 1), ttl)]);
      if (ctx && ctx.waitUntil) ctx.waitUntil(put); else await put;
    }

    let res;
    try {
      res = await fetch(UPSTREAM, { method: "POST", headers: { "content-type": "application/json", "x-api-key": env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01" }, body: JSON.stringify({ model: MODEL, max_tokens: MAX_TOKENS, stream: true,
        thinking: { type: "between_tools" }, output_config: { effort: "low" }, system: SYSTEM, messages: [{ role: "user", content: prompt }] }) });
    } catch (e) {
      return json(502, { error: "Couldn't reach Claude. Try again in a moment." }, origin);
    }
    if (!res.ok) {
      try { await res.body?.cancel(); } catch (e) {}
      const m = upstreamError(res.status);
      return json(m.status, m.retryAfter ? { error: m.error, retryAfter: m.retryAfter } : { error: m.error }, origin, m.retryAfter ? { "Retry-After": String(m.retryAfter) } : {});
    }
    return new Response(res.body, { status: 200, headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache, no-store",
      ...cors(origin) } });
  },
};
