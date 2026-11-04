# Explain with AI proxy

A Cloudflare Worker for **Explain with AI** in Practice mode and a finished hand's review (`gto/static/explain.js`, `EXPLAIN_PROXY`), so it works
without each visitor's own API key. `POST /explain` with JSON `{"prompt": "..."}` (at most 8000 characters) from an
allowed origin; the worker sends it to the Claude Messages API with its own fixed model (`claude-sonnet-5-5`), coach
system prompt and `max_tokens` (700), streamed, and passes the Server-Sent Events back.

- **Origins**: `ALLOWED_ORIGINS` in `wrangler.toml` (exact origins, or `http://localhost:*` for any port); others get 403.
- **Limits**: per IP 5 a minute (`[[ratelimits]]` binding `RATE_LIMITER`) and `DAILY_PER_IP` (20) a UTC day, and
  `DAILY_GLOBAL` (1000) a day in total (KV namespace `COUNTS`, keyed by a hash of the IP). Over a limit: 429
  `{"error", "retryAfter", "scope": "minute" | "day" | "global"}` with `Retry-After`.
- **Errors**: Anthropic's 401/403 become 502 "proxy misconfigured", 429/5xx/529 become 503 "busy, try again"
  (`retryAfter: 30`); nothing of Anthropic's response (body, headers) is passed on except the stream itself.
- **Secret**: `ANTHROPIC_API_KEY` (a Console API key, `sk-ant-api03-...`), set with `npx wrangler secret put ANTHROPIC_API_KEY`.
  For `npx wrangler dev`, put a key in `.dev.vars` (`ANTHROPIC_API_KEY=...`; git-ignored) if you want real answers.

Deploy (the full steps are in the repo README's "Explain with AI proxy"):

```sh
npm install
npx wrangler login
npx wrangler kv namespace create COUNTS     # paste the id into wrangler.toml
npx wrangler secret put ANTHROPIC_API_KEY
npx wrangler deploy                         # prints https://gto-explain.<subdomain>.workers.dev
```

Tests (Node's test runner, fetch and the bindings mocked; no key, no network): `npm test`.
