// Loads gto/static/index.html's inline script into a Node vm context with a minimal fake DOM, so the
// page's logic (hand simulation, action recording, share links, the state sent to the solver, the
// copied hand text) can be tested without a browser. Each loadPage() call is a fresh page.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

// Node 24.14's V8 sometimes segfaults (about 1 run in 70 with the suite run 16 times in parallel; the test
// file's process dies with SIGSEGV and node --test reports "fail 1") in a GC that starts at the stack
// check of a function's Sparkplug (baseline) code: ClearStaleLeftTrimmedPointerVisitor reads a bad slot in
// BaselineOutOfLinePrologue's frame. Each loadPage() compiles the page afresh in a new context, so its
// functions are baseline-compiled over and over. Without Sparkplug (interpreter, then the optimizing
// tiers) the crash is gone (0 in 1900 parallel runs, against about 28 before); the suite is no slower.
require("node:v8").setFlagsFromString("--no-sparkplug");

// GTO_PAGE=<file> tests another copy of the page (e.g. an older version, to see a test catch its bug)
const HTML_PATH = process.env.GTO_PAGE || path.join(__dirname, "..", "..", "gto", "static", "index.html");

// The ids the page finds in its own HTML, and the ones it (or a module it loads) creates itself: a lookup of any other id
// is a test failure, not an element made up on the spot, which hid broken wiring. A lookup of one of these while it isn't
// there answers null, as a browser does. tests/js/harness.test.js checks this list against gto/static both ways, so an
// entry goes with its code.
const RUNTIME_IDS = {
  "index.html": ["customAmt", "customBox", "equityRetry", "solveBar", "solveMsg", "solveTime", "solverRetry"],
  "hands.js": ["mhStyle"], "help.js": ["hpStyle"],
};
const RUNTIME = new Set(Object.values(RUNTIME_IDS).flat());
// (the markup without the page's one inline script: its templates are the ids created at runtime)
const staticIds = () => new Set([...fs.readFileSync(HTML_PATH, "utf8").replace(/<script>[\s\S]*?<\/script>/, "").matchAll(/\sid="([^"]+)"/g)].map(m => m[1]));
// A lookup a page's own try/catch swallowed still fails the test it happened in
const unknownIds = [];
if (/\.test\.js$/.test(process.argv[1] || ""))
  require("node:test").afterEach(() => { if (unknownIds.length) throw new Error(`no element with id: ${unknownIds.splice(0).join(", ")} (not in index.html or RUNTIME_IDS, page.js)`); });

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
  constructor(tag, id, reg) {
    this.reg = reg; this._id = id || "";
    this.tagName = (tag || "div").toUpperCase();
    this.className = ""; this.classList = new ClassList(this);
    this.style = {}; this.dataset = {}; this.children = []; this.listeners = {};
    this._html = ""; this.textContent = ""; this.hidden = false; this.value = ""; this.checked = false;
    this.disabled = false; this.title = ""; this.onclick = null; this.onchange = null;
  }
  get id() { return this._id; }
  set id(v) { this._id = String(v); if (this.reg && v) this.reg.named.set(this._id, this); }   // (document.getElementById finds it)
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = String(v); this.children = []; this._wrote(this._html); }
  // ids in written HTML, which isn't parsed: there while this element's HTML says so
  _wrote(h) { if (this.reg) for (const m of h.matchAll(/\sid="([^"]+)"/g)) this.reg.written.set(m[1], this); }
  appendChild(c) { this.children.push(c); return c; }
  prepend(c) { this.children.unshift(c); }
  append(...xs) { for (const x of xs) if (typeof x === "string") { this._html += x; this._wrote(x); } else this.children.push(x); }
  insertAdjacentHTML(_, h) { this._html += h; this._wrote(h); }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  // appended descendants the page looks up by selector: the fold titles ("h2[data-fold]") and by id ("#early")
  querySelector(sel) {
    if (sel.startsWith("#")) return [...this.walk()].find(e => e.id === sel.slice(1)) || null;
    return sel === "h2[data-fold]" ? [...this.walk()].find(e => e.tagName === "H2" && "fold" in e.dataset) || null : null;
  }
  querySelectorAll(sel) { return sel === "button" ? [...this.walk()].filter(e => e.tagName === "BUTTON") : []; }
  focus() {} blur() {} click() { if (this.onclick) this.onclick({ target: this }); }
  setAttribute(k, v) { this[k] = String(v); }
  getAttribute(k) { return k in this ? this[k] : null; }
  closest() { return null; }
  // iOS Safari ignores select() on a textarea; only a selection range selects its text (El.iosSelect)
  select() { if (!El.iosSelect) El.selected = this; }
  setSelectionRange() { El.selected = this; }
  remove() { this.removed = true; }
  get offsetWidth() { return 0; }
  // all descendants (appended elements only; innerHTML is not parsed)
  *walk() { for (const c of this.children) { yield c; yield* c.walk(); } }
}

// Options: hash (location.hash), path (location.pathname, default /gto-solver/: e.g. /gto-solver/practice/), search,
// scripts (file name -> source run instead of gto/static's file when the page loads that script), protocol (location.protocol, default http:), storage (initial localStorage entries), server (true: /health answers ok,
// so the page uses the local server's budgets and POSTs to /advise instead of the worker), isolated (true: the page is
// cross-origin isolated, so it can share a SharedArrayBuffer with its worker), serviceWorker (true: a first visit whose
// service worker is registering, so the page is about to reload under it).
function loadPage(opts = {}) {
  const els = new Map(), staticNow = staticIds(), reg = { named: new Map(), written: new Map() };
  // the element with this id, if the page has one now: in its markup, given the id, or written into an element's HTML
  const present = (id) => {
    const n = reg.named.get(id), w = reg.written.get(id);
    if (n && !n.removed) return n;
    if (!staticNow.has(id) && !(w && !w.removed && w._html.includes(`id="${id}"`))) return null;
    if (!els.has(id)) els.set(id, new El("div", id, reg));
    return els.get(id);
  };
  const byId = (id) => {
    const el = present(id);
    if (el || RUNTIME.has(id)) return el;
    unknownIds.push(id);
    throw new Error(`no element with id "${id}": it's neither in index.html nor in RUNTIME_IDS (tests/js/page.js)`);
  };
  const keydown = [];
  const storage = new Map(Object.entries(opts.storage || {}));
  const posted = [];      // messages the page sent to the solver worker / server
  const workers = [];
  const fetched = [], notFound = [];   // URLs the page fetched; those the static site doesn't have (404s)
  // The clipboard. opts.clipboard: "text" (default: navigator.clipboard.writeText), "denied" (the
  // clipboard API always refuses). opts.gesture: like Safari, clipboard writes and execCommand("copy") work
  // only while a user gesture is being handled (page.click() runs a click handler as one, synchronously).
  const clipboard = { text: null, gesture: false, prompts: [], via: null };
  const cbMode = opts.clipboard || "text";
  const allowed = () => cbMode !== "denied" && (!opts.gesture || clipboard.gesture);
  const fakeClipboard = {
    writeText: async (t) => { if (!allowed()) throw new Error("NotAllowedError"); clipboard.text = t; clipboard.via = "writeText"; },
  };
  const location = { hash: opts.hash || "", origin: "https://www.michael-han.com", pathname: opts.path || "/gto-solver/", search: opts.search || "", protocol: opts.protocol || "http:" };
  const replaced = [], pushed = [];   // the URLs history.replaceState / pushState were given
  const goTo = (url) => {   // (the page passes absolute paths)
    const u = new URL(String(url), location.origin + location.pathname);
    location.pathname = u.pathname; location.search = u.search; location.hash = u.hash;
  };
  const scripts = [];   // the scripts the page loaded (document.head.appendChild), by file name
  const docListeners = {};   // document events other than keydown (visibilitychange), by type

  class FakeWorker {
    constructor(url) { this.url = url; this.onmessage = null; this.onerror = null; this.terminated = false; workers.push(this); }
    postMessage(m) { posted.push(m.stop ? { stop: m.stop } : JSON.parse(JSON.stringify(m))); }   // (the stop flag: shared, as is)
    terminate() { this.terminated = true; }
  }
  // the header's speed and opponents toggles (static markup in the page)
  for (const [id, key, vals] of [["solvectl", "m", ["fast", "accurate"]], ["oppctl", "o", ["gto", "pool"]]])
    if (staticNow.has(id)) for (const v of vals) { const b = new El("button"); b.dataset[key] = v; byId(id).appendChild(b); }
  const body = new El("body", "body");
  // a <script src> appended to the head runs (the page's own file, or opts.scripts') after the current task, then onload
  const head = new El("head");
  head.appendChild = (el) => {
    head.children.push(el);
    if (el.tagName !== "SCRIPT" || !el.src) return el;
    const name = String(el.src).split("?")[0];
    scripts.push(name);
    setImmediate(() => {
      const f = path.join(path.dirname(HTML_PATH), name);
      const src = opts.scripts && name in opts.scripts ? opts.scripts[name] : fs.existsSync(f) ? fs.readFileSync(f, "utf8") : null;
      if (src == null) return el.onerror && el.onerror(new Error("404"));
      vm.runInContext(src, ctx, { filename: name });
      if (el.onload) el.onload();
    });
    return el;
  };
  const document = {
    body, head, title: "GTO Solver",
    getElementById: byId,
    createElement: (t) => new El(t, "", reg),
    // the page only looks up action buttons by data-key ("#actionArea [data-key=..]"); everything else is absent
    querySelector(sel) {
      const m = sel.match(/^#(\w+) \[data-key="(.*)"\]$/);
      if (!m) return null;
      for (const e of byId(m[1]).walk()) if (e.dataset.key === m[2]) return e;
      return null;
    },
    addEventListener: (t, f) => { if (t === "keydown") keydown.push(f); else (docListeners[t] = docListeners[t] || []).push(f); },
    execCommand(cmd) {
      if (cmd !== "copy" || !El.selected || (opts.gesture && !clipboard.gesture)) return false;
      clipboard.text = El.selected.value; clipboard.via = "execCommand"; return true;
    },
  };
  El.selected = null; El.iosSelect = !!opts.iosSelect;
  // opts.layout: run the page's layout code (it only runs where window has events): the fold titles of the
  // Table and Hand panels, the root element's style properties and a visualViewport whose events the test fires
  const rootProps = {}, winListeners = {}, vvListeners = {};
  const viewport = { height: 800, offsetTop: 0, scale: 1, addEventListener: (t, f) => (vvListeners[t] = vvListeners[t] || []).push(f) };
  if (opts.layout) {
    for (const id of ["pTable", "pHistory"]) {
      const h = new El("h2"); h.dataset.fold = ""; h.parentElement = byId(id); byId(id).appendChild(h);
    }
    document.documentElement = {
      clientHeight: 800,
      style: { setProperty: (k, v) => { rootProps[k] = v; }, removeProperty: (k) => { delete rootProps[k]; } },
    };
    document.querySelectorAll = (sel) => sel === "h2[data-fold]" ? ["pTable", "pHistory"].map(id => byId(id).querySelector(sel)) : [];
    const addEv = document.addEventListener;
    document.addEventListener = (t, f) => { if (t === "keydown") addEv(t, f); };
  }
  const ctx = {
    document, location, console,
    ...(opts.isolated ? { crossOriginIsolated: true } : {}),
    history: {
      replaceState: (_s, _t, url) => { replaced.push(url); goTo(url); },
      pushState: (_s, _t, url) => { pushed.push(url); goTo(url); },
    },
    navigator: { clipboard: fakeClipboard, ...(opts.serviceWorker ? { serviceWorker: { controller: null, register: () => new Promise(() => {}), ready: new Promise(() => {}) } } : {}) },
    ...(opts.serviceWorker ? { isSecureContext: true, sessionStorage: { getItem: () => null, setItem: () => {} } } : {}),
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
      fetched.push(String(url));
      if (!opts.server) {
        // the static site: the page's own files are there (a file that isn't is a 404, which browsers log as an error)
        const f = path.join(path.dirname(HTML_PATH), String(url).split("?")[0]);
        if (/^[\w.]+$/.test(String(url)) && fs.existsSync(f)) return { ok: true, status: 200, json: async () => JSON.parse(fs.readFileSync(f, "utf8")) };
        if (/^[\w.]+$/.test(String(url))) { notFound.push(String(url)); return { ok: false, status: 404, json: async () => { throw new Error("not JSON"); } }; }
        throw new Error("no server");
      }
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
  ctx.addEventListener = (t, f) => (winListeners[t] = winListeners[t] || []).push(f);
  if (opts.layout) ctx.visualViewport = viewport;
  vm.createContext(ctx);
  // opts.store: the page's storage layer (store.js, its <script src> before the inline script) over opts.indexedDB (an
  // idb_shim.js database; a new one if not given). Without it the page keeps its state in localStorage itself.
  if (opts.store) {
    ctx.indexedDB = opts.indexedDB || require("./idb_shim.js").fakeIndexedDB();
    vm.runInContext(fs.readFileSync(path.join(path.dirname(HTML_PATH), "store.js"), "utf8"), ctx, { filename: "store.js" });
  }
  vm.runInContext(pageScript(), ctx, { filename: "index.html" });
  // run code in the page's global scope (its top-level let/const are visible here)
  const run = (code) => vm.runInContext(code, ctx);
  // the value of an expression as a plain Node object (the page's own objects come from another realm)
  const val = (code) => { const t = run(`JSON.stringify(${code})`); return t === undefined ? undefined : JSON.parse(t); };
  const page = {
    ctx, run, val, els, byId, present, storage, posted, workers, fetched, notFound, clipboard, location, replaced, pushed, scripts, rootProps, viewport,
    // back or forward to a URL (an absolute path): location changes, then popstate
    popTo(url) { goTo(url); page.fire("popstate", { state: null }); },
    // fire a window event (e.g. "hashchange" after setting location.hash)
    fire(t, e = {}) { for (const f of winListeners[t] || []) f(e); },
    // the tab hidden or shown (document.hidden, then visibilitychange)
    setHidden(h) { document.hidden = h; for (const f of docListeners.visibilitychange || []) f({}); },
    // change the visual viewport (e.g. the on-screen keyboard opening) and fire its events
    setViewport(v) { Object.assign(viewport, v); for (const t of ["resize", "scroll"]) for (const f of vvListeners[t] || []) f({}); },
    get S() { return val("S"); },
    set S(v) { ctx.__tmp = JSON.stringify(v); run("S = JSON.parse(__tmp); setPositions(S.players); save(); renderAll();"); },
    ready: () => run("modeReady"),
    // the hand st, you to act, as its review shows that decision: the hand played on from there (you fold), finished,
    // that decision looked at. This is how advice reaches the page. S is then the hand up to it.
    spot(st) {
      ctx.__tmp = JSON.stringify(st);
      run(`(() => { if (stepView) stepStop(); S = JSON.parse(__tmp); setPositions(S.players); const s = simulate().current;
        S.actions[s].push({ pos: S.hero, action: "fold" }); S.finished = true; save(); stepTo(s, S.actions[s].length - 1); })()`);
    },
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

module.exports = { loadPage, state, acts, pageScript, HTML_PATH, RUNTIME_IDS, staticIds, unknownIds };
