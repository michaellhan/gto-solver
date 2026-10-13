// Loads gto/static/index.html's inline script into a Node vm context with a minimal fake DOM, so the
// page's logic (hand simulation, action recording, share links, the state sent to the solver, the
// copied hand text) can be tested without a browser. Each loadPage() call is a fresh page.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

// GTO_PAGE=<file> tests another copy of the page (e.g. an older version, to see a test catch its bug)
const HTML_PATH = process.env.GTO_PAGE || path.join(__dirname, "..", "..", "gto", "static", "index.html");

function pageScript() {
  const html = fs.readFileSync(HTML_PATH, "utf8");
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  if (scripts.length !== 1) throw new Error(`expected one inline <script> in index.html, found ${scripts.length}`);
  return scripts[0];
}

class ClassList {
  constructor(el) { this.el = el; }
  _get() { return this.el.className.split(/\s+/).filter(Boolean); }
  _set(a) { this.el.className = a.join(" "); }
  add(...c) { const a = this._get(); for (const x of c) if (!a.includes(x)) a.push(x); this._set(a); }
  remove(...c) { this._set(this._get().filter(x => !c.includes(x))); }
  contains(c) { return this._get().includes(c); }
  toggle(c, force) {
    const on = force === undefined ? !this.contains(c) : !!force;
    if (on) this.add(c); else this.remove(c);
    return on;
  }
}

class El {
  constructor(tag, id) {
    this.tagName = (tag || "div").toUpperCase(); this.id = id || "";
    this.className = ""; this.classList = new ClassList(this);
    this.style = {}; this.dataset = {}; this.children = []; this.listeners = {};
    this._html = ""; this.textContent = ""; this.hidden = false; this.value = ""; this.checked = false;
    this.disabled = false; this.title = ""; this.onclick = null; this.onchange = null;
  }
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = String(v); this.children = []; }
  appendChild(c) { this.children.push(c); return c; }
  append(...xs) { for (const x of xs) if (typeof x === "string") this._html += x; else this.children.push(x); }
  insertAdjacentHTML(_, h) { this._html += h; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  querySelector() { return null; }
  querySelectorAll(sel) { return sel === "button" ? [...this.walk()].filter(e => e.tagName === "BUTTON") : []; }
  focus() {} blur() {} click() { if (this.onclick) this.onclick({ target: this }); }
  setAttribute(k, v) { this[k] = String(v); }
  select() { El.selected = this; }
  remove() { this.removed = true; }
  get offsetWidth() { return 0; }
  // all descendants (appended elements only; innerHTML is not parsed)
  *walk() { for (const c of this.children) { yield c; yield* c.walk(); } }
}

// Options: hash (location.hash), storage (initial localStorage entries), server (true: /health answers ok,
// so the page uses the local server's budgets and POSTs to /advise instead of the worker).
function loadPage(opts = {}) {
  const els = new Map();
  const byId = (id) => { if (!els.has(id)) els.set(id, new El("div", id)); return els.get(id); };
  const keydown = [];
  const storage = new Map(Object.entries(opts.storage || {}));
  const posted = [];      // messages the page sent to the solver worker / server
  const workers = [];
  // The clipboard. opts.clipboard: "text" (default: navigator.clipboard.writeText only), "item" (also
  // write() with ClipboardItem, whose data may be a promise, as Safari and recent Chrome), "denied" (the
  // clipboard API always refuses). opts.gesture: like Safari, clipboard writes and execCommand("copy") work
  // only while a user gesture is being handled (page.click() runs a click handler as one, synchronously).
  const clipboard = { text: null, gesture: false, prompts: [], via: null };
  const cbMode = opts.clipboard || "text";
  const allowed = () => cbMode !== "denied" && (!opts.gesture || clipboard.gesture);
  const fakeClipboard = {
    writeText: async (t) => { if (!allowed()) throw new Error("NotAllowedError"); clipboard.text = t; clipboard.via = "writeText"; },
  };
  class ClipboardItem { constructor(items) { this.items = items; } }
  if (cbMode === "item") fakeClipboard.write = (items) => {
    if (!allowed()) return Promise.reject(new Error("NotAllowedError"));
    return (async () => { clipboard.text = await (await items[0].items["text/plain"]).text(); clipboard.via = "write"; })();
  };
  const location = { hash: opts.hash || "", origin: "https://www.michael-han.com", pathname: "/gto-solver/", search: "" };
  const replaced = [];

  class FakeWorker {
    constructor(url) { this.url = url; this.onmessage = null; this.onerror = null; this.terminated = false; workers.push(this); }
    postMessage(m) { posted.push(JSON.parse(JSON.stringify(m))); }
    terminate() { this.terminated = true; }
  }
  // the header's speed and opponents toggles (static markup in the page)
  for (const [id, key, vals] of [["solvectl", "m", ["fast", "accurate"]], ["oppctl", "o", ["gto", "pool"]]])
    for (const v of vals) { const b = new El("button"); b.dataset[key] = v; byId(id).appendChild(b); }
  const body = new El("body", "body");
  const document = {
    body,
    getElementById: byId,
    createElement: (t) => new El(t),
    // the page only looks up action buttons by data-key ("#actionArea [data-key=..]"); everything else is absent
    querySelector(sel) {
      const m = sel.match(/^#(\w+) \[data-key="(.*)"\]$/);
      if (!m) return null;
      for (const e of byId(m[1]).walk()) if (e.dataset.key === m[2]) return e;
      return null;
    },
    addEventListener: (t, f) => { if (t === "keydown") keydown.push(f); },
    execCommand(cmd) {
      if (cmd !== "copy" || !El.selected || (opts.gesture && !clipboard.gesture)) return false;
      clipboard.text = El.selected.value; clipboard.via = "execCommand"; return true;
    },
  };
  El.selected = null;
  const ctx = {
    document, location, console,
    history: { replaceState: (_s, _t, url) => { replaced.push(url); location.hash = ""; } },
    navigator: { clipboard: fakeClipboard },
    ...(cbMode === "item" ? { ClipboardItem } : {}),
    Blob,
    localStorage: {
      getItem: (k) => storage.has(k) ? storage.get(k) : null,
      setItem: (k, v) => storage.set(k, String(v)),
      removeItem: (k) => storage.delete(k),
    },
    performance: { now: () => 0 },
    // timers never fire: the page only uses them for toasts and the solve progress clock
    setTimeout: () => 0, clearTimeout: () => {}, setInterval: () => 0, clearInterval: () => {},
    fetch: async (url, init) => {
      if (!opts.server) throw new Error("no server");
      if (String(url).startsWith("health")) return { ok: true, json: async () => ({ ok: true }) };
      if (String(url).startsWith("advise")) { posted.push({ state: JSON.parse(init.body) }); return new Promise(() => {}); }
      return { ok: true, json: async () => ({}) };
    },
    Worker: FakeWorker,
    CSS: { escape: (s) => String(s).replace(/["\\]/g, "\\$&") },
    btoa, atob,
    prompt: (msg, text) => { clipboard.prompts.push(text); return null; },
  };
  ctx.window = ctx;
  vm.createContext(ctx);
  vm.runInContext(pageScript(), ctx, { filename: "index.html" });
  // run code in the page's global scope (its top-level let/const are visible here)
  const run = (code) => vm.runInContext(code, ctx);
  // the value of an expression as a plain Node object (the page's own objects come from another realm)
  const val = (code) => { const t = run(`JSON.stringify(${code})`); return t === undefined ? undefined : JSON.parse(t); };
  const page = {
    ctx, run, val, els, byId, storage, posted, workers, clipboard, location, replaced,
    get S() { return val("S"); },
    set S(v) { ctx.__tmp = JSON.stringify(v); run("S = JSON.parse(__tmp); setPositions(S.players); save(); renderAll();"); },
    ready: () => run("modeReady"),
    // simulate(), with its sets as arrays in seat order
    sim: () => val("(() => { const s = simulate(); return { ...s, active: POS.filter(p => s.active.has(p)), allin: POS.filter(p => s.allin.has(p)) }; })()"),
    payload: () => val("payload()"),
    // press a key on the page (no target input focused)
    key(k) { const e = { key: k, metaKey: false, ctrlKey: false, altKey: false, target: { tagName: "BODY" }, preventDefault() {} }; for (const f of keydown) f(e); },
    // the buttons currently in the action panel, by label
    buttons() { return [...byId("actionArea").walk()].filter(e => e.tagName === "BUTTON"); },
    button(label) {
      const b = page.buttons().find(e => e.innerHTML.replace(/<[^>]*>.*$/, "").trim() === label || e.innerHTML.startsWith(label));
      if (!b) throw new Error(`no "${label}" button; have: ${page.buttons().map(e => e.innerHTML.replace(/<.*/, "")).join(", ")}`);
      return b;
    },
    // record an action for whoever is to act (as the page's buttons do); returns the page's error, if any
    act(a) { ctx.__tmp = JSON.stringify(a); return run("(() => { const e = act(JSON.parse(__tmp)); renderAll(); return e; })()"); },
    err: () => byId("err").textContent,
    // click an element as the user would: its handler runs inside a user gesture, which ends when the
    // handler returns (anything it awaits runs after the gesture); returns what the handler returned
    click(id) {
      clipboard.gesture = true;
      try { return byId(id).onclick({ target: byId(id) }); } finally { clipboard.gesture = false; }
    },
  };
  return page;
}

// A state like the page's own (S), built from a compact description.
function state({ players = 6, hero = "BTN", stack = 100, cards = [], ante, straddle, preflop = [], flop = [], turn = [], river = [] } = {}) {
  const c = [...cards]; while (c.length < 7) c.push(null);
  const s = { players, hero, stack, cards: c, actions: { preflop, flop, turn, river } };
  if (ante !== undefined) s.ante = ante;
  if (straddle !== undefined) s.straddle = straddle;
  return s;
}
// "UTG f, HJ r2.5, CO c, BTN x, SB b3, BB a" -> action list
function acts(str) {
  if (!str) return [];
  return str.split(",").map(t => t.trim()).filter(Boolean).map(t => {
    const [pos, a] = t.split(/\s+/);
    const kind = { f: "fold", x: "check", c: "call", a: "all-in", r: "raise", b: "bet" }[a[0]];
    const out = { pos, action: kind };
    if (kind === "raise" || kind === "bet") out.to = parseFloat(a.slice(1));
    return out;
  });
}

module.exports = { loadPage, state, acts, pageScript, HTML_PATH };
