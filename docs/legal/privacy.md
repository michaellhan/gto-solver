# DRAFT: Privacy Policy

> **DRAFT for review by a lawyer. Not in effect, not published, not linked from the site.**
> Written from `.work/monetize/paywall.md` (section 5) and from what the site's code does today (November 2026).
> Text in [brackets] is a decision or a detail still to fill in. Part A describes the site as it is now. Part B
> describes the planned paid tier (accounts, sync, payments) and must be checked against what is actually built
> before it is published. When Part B ships, the site's and README's "nothing you enter is uploaded" must change too.

Last updated: [date]

This policy explains what information GTO Solver (https://www.michael-han.com/gto-solver/, "the Service") handles,
why, and your choices. The Service is run by [legal entity name] ("we", "us"), [postal address], [contact email].

## Part A: the Service today (free, no accounts)

### A1. The solver runs in your browser

The solver is downloaded to your browser and runs there (Python compiled to WebAssembly). The hands you enter, the
hand histories you paste or import, your settings and your practice and drill history are **stored only on your
device**, in your browser's local storage, IndexedDB and cache. We don't receive them. Clearing your browser's site
data for michael-han.com deletes them. The Service sets no cookies and uses no analytics or advertising trackers.

What the site keeps on your device, for reference: settings (speed, opponents, layout, four-colour deck, sound,
whether you've seen the welcome screen); Practice and drill hands, scores and habits; opponent profiles and reports;
imported session hand histories and their analysis; cached solves and study answers; and, only if you enter one,
your own Anthropic API key (see A3).

### A2. Hosting and the files your browser downloads

- **GitHub Pages** (GitHub, Inc.) hosts the site. Like any web host it receives your IP address, browser details and
  the pages requested, and may keep them in its logs under GitHub's privacy statement. We don't receive those logs.
- **jsDelivr** (a public CDN) serves the Python runtime (Pyodide) and numpy the solver needs. It receives your IP
  address and browser details when your browser downloads them, under jsDelivr's privacy policy.

### A3. "Explain with AI" (Practice and the review of a finished hand)

This is the only feature that sends anything you do to a server.

- **Using our proxy (default).** When you click *Explain with AI*, your browser sends a short text summary of that
  decision (the game, your seat and cards, the board and the betting, what you chose, and the solver's mix, EVs and
  verdict for it), a hand you practised or one you played and are reviewing, to our proxy on **Cloudflare Workers**,
  which forwards that text alone to **Anthropic** (the Claude API) and returns the explanation. Nothing else from
  your device is sent, and nothing at all until you click. The summary names seats by position, not by player name.
  - To limit abuse and cost, the proxy counts requests per IP address: up to 5 a minute and 20 a day per address, and
    1000 a day in total. The daily counters are stored under a one-way hash (SHA-256) of your IP address, not the
    address itself, and expire after 2 days. The per-minute limit uses Cloudflare's rate-limiting service.
  - The proxy keeps no other record of requests. Cloudflare processes the request (including your IP address) to
    run the proxy, under its own policies. [Confirm the Worker's logging/observability settings in the Cloudflare
    dashboard.]
  - Anthropic processes the text to produce the explanation, under its commercial terms. [Confirm and state
    Anthropic's API data-retention period and that API inputs aren't used for training by default.]
- **Using your own key.** If you enter your own Anthropic API key, it is stored only in your browser, and the
  summary goes straight from your browser to Anthropic (api.anthropic.com) under your own account with Anthropic. It
  doesn't pass through us.

### A4. Contact

If you email us, we receive your email address and message and use them only to reply. [State retention, e.g. kept
for up to 2 years.]

## Part B: the paid tier *(planned; not built yet)*

### B1. What we collect

| What | Why | Kept for |
|---|---|---|
| Email address | your account: sign-in links, receipts, important notices | while your account is open, then [30] days |
| Google profile basics (name, email, picture), if you sign in with Google | sign-in | as above |
| Saved hands, notes and stats, **only if you turn on sync** | to show them on your other devices | until you delete them or close your account |
| Query log: account id, spot, board and time of the library reads and AI explanations you request | fair play (Terms section 4) and preventing scraping and account sharing | 90 days |
| Hashed IP address (SHA-256) and request counts | rate limits and abuse prevention | [2 days] |
| Session cookie | keeping you signed in | until you sign out or it expires ([duration]) |
| Subscription status (plan, renewal date) from our reseller | to unlock paid features | while your account is open |

We don't receive or store your card or bank details: our reseller [MoR] collects payment.

### B2. Fair-play checks

If a poker site, card room or player investigating fair play asks, we may confirm whether and when a given spot was
looked up from an account, using the query log, as the Terms of Service describe. We disclose only what is needed to
answer. [Decide the process and whether the user is told.]

### B3. Who processes your information

- **Cloudflare** (hosting, accounts database, storage, logs) as our processor.
- **[Email provider]** (sign-in emails and receipts) as our processor.
- **Anthropic** (AI explanations: the summary text only) as our processor.
- **[MoR, e.g. Paddle / Lemon Squeezy]** as the reseller and Merchant of Record: it collects payment, tax details
  and billing address and is responsible for that information under its own privacy policy.
- [Google, if Sign in with Google is offered: it tells us your name, email and picture when you choose it.]

We don't sell your information or share it for advertising. We may disclose information if the law requires it.

### B4. Cookies

We use a single cookie that is strictly necessary to keep you signed in, so we don't show a cookie banner. We use no
analytics, advertising or tracking cookies.

### B5. Your rights

You can, at any time and without charge:

- **access** and **export** your data: [account page > Export, which downloads a JSON file];
- **correct** it: [account page];
- **delete** it, closing your account: [account page > Delete account]. We delete your account data within [30]
  days, and it leaves our backups within [90] days. Query-log entries expire after 90 days. Our reseller keeps
  transaction records it must keep by law.

Depending on where you live (for example the EU, the UK or California) you may have further rights, such as to object
or to complain to a data-protection authority. Write to [contact email]. [Lawyer: legal bases under GDPR (contract
for the account; legitimate interests for fair-play logs and rate limits), a CCPA/CPRA notice, and whether an EU/UK
representative is needed.]

### B6. International transfers

We and our processors may process information in the United States and other countries. Where the law requires it we
rely on [the EU-US Data Privacy Framework / standard contractual clauses] for transfers.

### B7. Children

The Service is for adults (18+). We don't knowingly collect information from anyone under 18; if you think we have,
write to us and we'll delete it.

## Changes

If we change this policy in a way that matters, we'll say so on the site [and by email to account holders] before
the change takes effect.

## Contact

[Legal entity name], [postal address], [contact email].
