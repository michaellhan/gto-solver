// The site's local storage (loaded by the page before its own script): everything this browser keeps for you, in one
// place and one shape, ready to be synced to an account later. IndexedDB "gto-store" holds records
//   { id, v: schema, created, updated, dev: this browser's id, data } (a deleted one: { id, v, created, updated, dev,
//   deleted: true }, a tombstone, so a later sync can tell a deletion from a record never seen)
// in four stores:
//   kv        each small setting or state the page and its views keep (the solver's hand "handreview", the speed, and what
//             the frozen copy keeps: Practice's score and review, the drills' progress, Study's preferences...): id = its localStorage key, data
//             { raw: the string }. The page reads these synchronously while it starts, so each is also kept in
//             localStorage under its own key (the mirror, exactly what was there before): set() writes both, get()
//             reads the mirror. A key written to localStorage directly (code that doesn't use the store yet) is taken
//             in at the next load and before a backup (reconcile).
//   hands     My hands: a saved hand each (hands below)
//   sessions  the frozen copy's session analyzer's summary of each session (id: the session's)
//   blobs     big state kept whole: that analyzer's session in progress ("session-current")
// On the first load it takes in what the site kept before (migrate): every localStorage key above (left where it is:
// it's the mirror) and the analyzer's IndexedDB "gto-session" (its session in progress and history; the old database is
// left as it was). The frozen copy's solved preflop games (IndexedDB "gto-pf") and the worker's files (Cache API) are caches of
// what can be solved or fetched again and stay where they are.
// The Anthropic API key (the frozen copy's Explain with AI, "gto-anthropic-key") and the choice to use it ("gto-explain-own")
// never become records: they stay in localStorage only, and a backup never holds the key, even inside a note.
// Without IndexedDB (some private modes) the records are kept in localStorage ("gto-store:<store>"), else in memory.
// wipe() deletes all of it, by name (SITE_KEYS, SITE_DATABASES, SITE_CACHES): My hands' "Delete everything this site keeps".
(function () {
  "use strict";
  const DB_NAME = "gto-store", DB_VERSION = 1, SCHEMA = 1;
  const STORES = ["kv", "hands", "sessions", "blobs"];
  // the localStorage keys the site has kept (any other "gto-" key is taken in too)
  const KNOWN = ["handreview", "fourcolor", "gto-speed", "gto-opponents", "gto-sound", "gto-layout", "gto-reports", "gto-drills", "gto-study",
    "gto-practice", "gto-practice-habits", "gto-practice-state", "gto-practice-hand", "gto-practice-score", "gto-practice-seat",
    "gto-practice-review", "gto-practice-drill", "gto-practice-hint", "gto-practice-table", "gto-practice-tables", "gto-practice-more",
    "gto-practice-detail", "gto-mt", "gto-mt-score", "gto-mt-drill", "gto-mt-seat", "gto-hands-table", "gto-hands-ui"];
  // (and the session analyzer's "a run is going in this tab" mark, gto-session-run: a lock between tabs, not state)
  const LOCAL_ONLY = new Set(["gto-anthropic-key", "gto-explain-own", "gto-session-run"]);
  const AI_KEY = "gto-anthropic-key";
  // the analyzer's localStorage fallback (no IndexedDB): its session in progress and history, moved to their stores
  const MOVED = { "gto-session": ["blobs", "session-current"], "gto-session-history": ["sessions", null] };
  const SENTINEL = "gto-store-mirror";   // set once the mirror is the store's: a key missing from it was removed
  const DEVICE_ONLY = new Set(["gto-layout", "gto-hands-table"]);   // backed up, never synced (this screen's column widths; which record the hand on this table is)
  const BACKUP = "gto-solver-backup", HANDS_FILE = "gto-solver-hands";
  const STREETS = ["preflop", "flop", "turn", "river"];
  // Everything this site keeps in this browser, by name, for wipe(). The origin serves other sites (Lizard, and the 22
  // other folders of the site repo), so nothing here clears the origin: no localStorage.clear(), no database or cache
  // that isn't named. SITE_KEYS is every localStorage and sessionStorage key gto/static's code uses and the frozen
  // copy's (/gto-solver-classic/, scripts/publish_classic.sh); tests/js/wipe.test.js greps gto/static and fails on a
  // key that isn't here, and pins the list: the keys of a module gone from gto/static stay, as the frozen copy writes them.
  const SITE_KEYS = [...KNOWN, ...LOCAL_ONLY, ...Object.keys(MOVED), SENTINEL,
    ...[...STORES, "meta"].map(s => "gto-store:" + s),   // (the records, without IndexedDB)
    "gto-welcome", "gto-stakes", "gto-writeup", "gto-pool-refused",   // (taken in as any "gto-" key is)
    "gto-isolate", "gto-isolate-classic"];               // (sessionStorage: the service worker's once-a-tab mark)
  const SITE_DATABASES = { [DB_NAME]: [...STORES, "meta"], "gto-session": ["kv"], "gto-pf": ["solves"] };
  const SITE_CACHES = ["gto-solver", "gto-solver-classic"];

  const clone = (x) => x === undefined ? undefined : JSON.parse(JSON.stringify(x));
  const parse = (s) => { if (s == null) return null; try { return JSON.parse(s); } catch (e) { return null; } };
  function uid() {
    try { if (globalThis.crypto && crypto.randomUUID) return crypto.randomUUID(); } catch (e) {}
    const h = () => Math.floor(Math.random() * 0x10000).toString(16).padStart(4, "0");
    return `${h()}${h()}-${h()}-4${h().slice(1)}-${(8 + Math.floor(Math.random() * 4)).toString(16)}${h().slice(1)}-${h()}${h()}${h()}`;
  }
  // a short hash of a string (two 32-bit FNV/rolling hashes), as the analyzer's file fingerprint
  function hash(s) {
    let a = 0x811c9dc5, b = 0;
    for (let k = 0; k < s.length; k++) { const c = s.charCodeAt(k); a = Math.imul(a ^ c, 16777619) >>> 0; b = (Math.imul(b, 31) + c) >>> 0; }
    return a.toString(36) + b.toString(36);
  }
  // JSON with every object's keys sorted (the same hand saved twice gives the same text)
  function canon(x) {
    if (Array.isArray(x)) return "[" + x.map(v => v === undefined ? "null" : canon(v)).join(",") + "]";
    if (x && typeof x === "object") return "{" + Object.keys(x).filter(k => x[k] !== undefined).sort().map(k => JSON.stringify(k) + ":" + canon(x[k])).join(",") + "}";
    return JSON.stringify(x === undefined ? null : x);
  }

  // ---- where records are kept: IndexedDB, else localStorage, else memory
  function idbBackend(idb) {
    let dbp = null;
    const open = () => dbp = dbp || new Promise((res, rej) => {
      let q; try { q = idb.open(DB_NAME, DB_VERSION); } catch (e) { return rej(e); }
      q.onupgradeneeded = () => { const d = q.result; for (const s of [...STORES, "meta"]) if (!d.objectStoreNames.contains(s)) d.createObjectStore(s, { keyPath: "id" }); };
      q.onsuccess = () => { const d = q.result; d.onversionchange = () => { d.close(); dbp = null; }; res(d); };
      q.onerror = () => { dbp = null; rej(q.error || new Error("IndexedDB refused")); };
    });
    const done = (t) => new Promise((res, rej) => { t.oncomplete = () => res(); t.onerror = t.onabort = () => rej(t.error || new Error("transaction failed")); });
    return {
      kind: "indexeddb", open,
      async getAll(store) { const d = await open(), t = d.transaction(store, "readonly"), q = t.objectStore(store).getAll(); await done(t); return q.result || []; },
      async get(store, id) { const d = await open(), t = d.transaction(store, "readonly"), q = t.objectStore(store).get(id); await done(t); return q.result || null; },
      async write(batch) {   // [{ store, rec }] in one transaction
        if (!batch.length) return;
        const d = await open(), t = d.transaction([...new Set(batch.map(b => b.store))], "readwrite");
        for (const b of batch) t.objectStore(b.store).put(b.rec);
        await done(t);
      },
      async clear(stores) { const d = await open(), t = d.transaction(stores, "readwrite"); for (const s of stores) t.objectStore(s).clear(); await done(t); },
      // the database itself, where its stores wouldn't clear: this tab's connection closed first, the others' close on
      // versionchange (above; a connection that doesn't keeps it open, which is reported after `wait` ms)
      async drop(wait) {
        try { if (dbp) (await dbp).close(); } catch (e) {}
        dbp = null;
        await new Promise((res, rej) => {
          const t = setTimeout(() => rej(new Error("another tab has it open")), wait);
          const end = (f, x) => { clearTimeout(t); f(x); };
          let q; try { q = idb.deleteDatabase(DB_NAME); } catch (e) { return end(rej, e); }
          q.onsuccess = () => end(res);
          q.onerror = () => end(rej, q.error || new Error("IndexedDB refused"));
        });
      },
    };
  }
  function mapBackend(load, save, kind) {   // localStorage ("gto-store:<store>", a JSON object by id) or memory
    const m = {};
    const get = (s) => m[s] = m[s] || load(s) || {};
    return {
      kind, open: async () => {},
      async getAll(s) { return Object.values(get(s)).map(clone); },
      async get(s, id) { const r = get(s)[id]; return r ? clone(r) : null; },
      async write(batch) {
        const touched = new Set();
        for (const b of batch) { get(b.store)[b.rec.id] = clone(b.rec); touched.add(b.store); }
        for (const s of touched) save(s, m[s]);
      },
      async clear(stores) { for (const s of stores) { m[s] = {}; save(s, m[s]); } },
    };
  }
  // an old IndexedDB database's object store, read whole (null if the database isn't there: opening it doesn't create it)
  function readOldDb(idb, name, store) {
    return new Promise((res) => {
      let q; try { q = idb.open(name); } catch (e) { return res(null); }
      let created = false;
      q.onupgradeneeded = () => { created = true; try { q.transaction.abort(); } catch (e) {} };
      q.onerror = () => res(null);
      q.onsuccess = () => {
        const d = q.result;
        if (created || !d.objectStoreNames.contains(store)) { d.close(); return res(null); }
        const out = {}, t = d.transaction(store, "readonly"), os = t.objectStore(store), keys = os.getAllKeys(), vals = os.getAll();
        t.oncomplete = () => { (keys.result || []).forEach((k, i) => out[k] = vals.result[i]); d.close(); res(out); };
        t.onerror = t.onabort = () => { d.close(); res(null); };
      };
    });
  }

  // a named database's stores emptied, whoever else has it open (the frozen copy's Study keeps "gto-pf" open, and nothing closes it on
  // versionchange, so deleting it would wait for that tab). True; false if it isn't there (it isn't created: an open that
  // would is aborted, as in readOldDb); rejects if it can't be done.
  function clearDb(idb, name, stores) {
    return new Promise((res, rej) => {
      let q; try { q = idb.open(name); } catch (e) { return rej(e); }
      let created = false;
      q.onupgradeneeded = () => { created = true; try { q.transaction.abort(); } catch (e) {} };
      q.onerror = () => created ? res(false) : rej(q.error || new Error("IndexedDB refused"));
      q.onsuccess = () => {
        const d = q.result, have = stores.filter(s => d.objectStoreNames.contains(s));
        if (created || !have.length) { d.close(); return res(false); }
        let t; try { t = d.transaction(have, "readwrite"); for (const s of have) t.objectStore(s).clear(); } catch (e) { d.close(); return rej(e); }
        t.oncomplete = () => { d.close(); res(true); };
        t.onerror = t.onabort = () => { d.close(); rej(t.error || new Error("IndexedDB refused")); };
      };
    });
  }

  function createStore(env = {}) {
    const ls = env.localStorage === undefined ? (() => { try { return globalThis.localStorage || null; } catch (e) { return null; } })() : env.localStorage;
    const idb = env.indexedDB === undefined ? (() => { try { return globalThis.indexedDB || null; } catch (e) { return null; } })() : env.indexedDB;
    const ss = env.sessionStorage === undefined ? (() => { try { return globalThis.sessionStorage || null; } catch (e) { return null; } })() : env.sessionStorage;
    const cs = env.caches === undefined ? (() => { try { return globalThis.caches || null; } catch (e) { return null; } })() : env.caches;
    const now = env.now || (() => Date.now());
    const lsGet = (k) => { try { return ls ? ls.getItem(k) : null; } catch (e) { return null; } };
    const lsSet = (k, v) => { try { if (!ls) return false; ls.setItem(k, v); return true; } catch (e) { return false; } };
    const lsDel = (k) => { try { if (ls) ls.removeItem(k); } catch (e) {} };
    const lsKeys = () => { const out = []; try { if (ls && typeof ls.key === "function") for (let i = 0; i < ls.length; i++) out.push(ls.key(i)); } catch (e) {} return out; };
    const adoptable = (k) => typeof k === "string" && !LOCAL_ONLY.has(k) && !(k in MOVED) && !k.startsWith("gto-store")
      && (KNOWN.includes(k) || k.startsWith("gto-"));

    let backend = null, meta = null, isReady = false;
    const cache = {};                 // store -> Map(id -> record), once read
    const pending = new Map();        // "store\0id" -> { store, id, data (null: delete), t }
    const listeners = new Set();
    let timer = null, flushing = Promise.resolve();
    let sealed = false;               // wipe() took the records: nothing is written again until the page loads again
    const chan = (() => { try { return typeof BroadcastChannel === "function" && env.broadcast !== false ? new BroadcastChannel(DB_NAME) : null; } catch (e) { return null; } })();
    if (chan && typeof chan.unref === "function") chan.unref();   // (Node: a channel doesn't keep the process up)

    async function load(store) {
      if (!cache[store]) { const m = new Map(); for (const r of await backend.getAll(store)) if (r && r.id != null) m.set(r.id, r); cache[store] = m; }
      return cache[store];
    }
    function changed(stores, remote, wiped) {
      for (const f of listeners) try { f({ stores, remote: !!remote, ...(wiped ? { wiped } : {}) }); } catch (e) {}
      if (!remote && chan) try { chan.postMessage({ stores, ...(wiped ? { wiped } : {}) }); } catch (e) {}
    }
    const seal = () => { sealed = true; if (timer) { clearTimeout(timer); timer = null; } pending.clear(); };
    if (chan) chan.onmessage = (e) => {   // another tab wrote: what it changed is read again
      const stores = (e.data && e.data.stores || []).filter(s => STORES.includes(s)), wiped = !!(e.data && e.data.wiped);
      if (wiped) seal();
      for (const s of stores) delete cache[s];
      if (stores.length) changed(stores, true, wiped);
    };

    // a record of data (null: a tombstone) over the one before it
    function record(id, data, base, t, extra) {
      const created = base && !base.deleted && base.created ? base.created : t;
      return data === null ? { id, v: SCHEMA, created, updated: t, dev: meta.device, deleted: true }
        : { id, v: SCHEMA, created, updated: t, dev: meta.device, ...(extra || {}), data };
    }
    const same = (store, base, data) => base && !base.deleted && (store === "kv" ? base.data && base.data.raw === data.raw : canon(base.data) === canon(data));
    function queue(store, id, data, t, extra) {
      if (sealed) return;
      pending.set(store + "\0" + id, { store, id, data, t: t || now(), ...(extra || {}) });
      if (!timer && isReady) timer = setTimeout(() => { timer = null; flush(); }, 30);
    }
    // the queued writes, in one transaction (after the migration: the store's own come after what it took in)
    function flush() {
      flushing = flushing.then(async () => {
        if (!isReady || !pending.size) return;
        const items = [...pending.values()]; pending.clear();
        const batch = [];
        for (const it of items) {
          const m = await load(it.store), base = m.get(it.id) || null;
          if (it.data === null ? !base || base.deleted : it.store !== "blobs" && same(it.store, base, it.data) && !base.nomirror === !it.nomirror) continue;
          const rec = record(it.id, it.data, base, it.t);
          if (it.created && it.data !== null) rec.created = it.created;   // (a deleted hand brought back: where it was)
          if (it.nomirror && it.data !== null) rec.nomirror = true;
          m.set(it.id, rec); batch.push({ store: it.store, rec });
        }
        try { await backend.write(batch); } catch (e) { for (const b of batch) delete cache[b.store]; }
        if (batch.length) changed([...new Set(batch.map(b => b.store))]);
      }).catch(() => {});
      return flushing;
    }
    const flushNow = () => { if (timer) { clearTimeout(timer); timer = null; } return flush(); };
    try {
      if (typeof globalThis.addEventListener === "function" && env.events !== false) {
        globalThis.addEventListener("pagehide", flushNow);
        if (typeof document === "object" && document && typeof document.addEventListener === "function")
          document.addEventListener("visibilitychange", () => { if (document.hidden) flushNow(); });
      }
    } catch (e) {}

    // ---- the mirror and the records agree: a key written straight to localStorage is taken in, a key the store
    // removed is gone from both; a mirror lost as a whole (its sentinel gone too) is written back from the records
    async function reconcile() {
      const kv = await load("kv"), t = now(), batch = [];
      if (!ls) return 0;   // (no localStorage: the records are all there is)
      const queued = (id) => pending.get("kv\0" + id);
      if (lsGet(SENTINEL) == null) {
        for (const r of kv.values()) if (!r.deleted && !queued(r.id) && lsGet(r.id) == null && r.data && typeof r.data.raw === "string") lsSet(r.id, r.data.raw);
        if (!lsSet(SENTINEL, meta.device) || lsGet(SENTINEL) == null) return 0;   // (a mirror that can't be trusted to hold a key: nothing deleted by it)
      }
      const keys = new Set([...KNOWN, ...lsKeys()].filter(adoptable));
      for (const r of kv.values()) keys.add(r.id);
      for (const k of keys) {
        if (queued(k) || !adoptable(k)) continue;
        const raw = lsGet(k), base = kv.get(k) || null;
        if (raw == null) {
          if (!base || base.deleted) continue;
          // (a value localStorage had no room for: the record is the copy; mirrored again once there's room)
          if (base.nomirror) { if (base.data && typeof base.data.raw === "string" && lsSet(k, base.data.raw)) batch.push({ store: "kv", rec: record(k, base.data, base, t) }); continue; }
          batch.push({ store: "kv", rec: record(k, null, base, t) }); continue;
        }
        if (!base || base.deleted || !base.data || base.data.raw !== raw || base.nomirror) batch.push({ store: "kv", rec: record(k, { raw }, base, t) });
      }
      for (const b of batch) kv.set(b.rec.id, b.rec);
      await backend.write(batch);
      return batch.length;
    }
    // ---- the first load: what the site kept before the store (once; the old copies are left as they were)
    async function migrate() {
      const done = meta.migrated || (meta.migrated = {});
      const batch = [], t = now();
      const put = (store, id, data) => batch.push({ store, rec: record(String(id), data, null, t) });
      if (!done.session) {
        let old = idb ? await readOldDb(idb, "gto-session", "kv") : null;
        if (!old) old = {};
        const lsCur = parse(lsGet("gto-session")), lsHist = parse(lsGet("gto-session-history"));
        const cur = old.current || lsCur, hist = [...(Array.isArray(old.history) ? old.history : []), ...(Array.isArray(lsHist) ? lsHist : [])];
        if (cur && typeof cur === "object") put("blobs", "session-current", cur);
        const seen = new Set();
        for (const x of hist) if (x && x.id != null && !seen.has(String(x.id))) { seen.add(String(x.id)); put("sessions", x.id, x); }
        done.session = t;
      }
      for (const b of batch) (await load(b.store)).set(b.rec.id, b.rec);
      await backend.write(batch);
      return batch.length;
    }
    async function init() {
      const pick = async (b) => { await b.open(); await b.getAll("meta"); return b; };
      const lsMap = ls && env.fallback !== "memory" ? mapBackend((s) => parse(lsGet("gto-store:" + s)), (s, m) => lsSet("gto-store:" + s, JSON.stringify(m)), "localStorage") : null;
      const mem = mapBackend(() => null, () => {}, "memory");
      for (const b of [idb && idbBackend(idb), lsMap, mem]) { if (!b) continue; try { backend = await pick(b); break; } catch (e) {} }
      meta = (await backend.get("meta", "meta")) || { id: "meta", schema: SCHEMA, device: uid(), created: now(), migrated: {} };
      try { await migrate(); } catch (e) {}
      try { await reconcile(); } catch (e) { try { await load("kv"); } catch (e2) {} }
      meta.schema = SCHEMA;
      try { await backend.write([{ store: "meta", rec: meta }]); } catch (e) {}
      isReady = true;
      await flushNow();
    }
    const ready = init().catch(() => { isReady = true; });

    // ---- small state (kv): synchronous, the mirror read and written at once, the record written just after
    // (without localStorage, or a value it had no room for: the records)
    function getRaw(k) {
      const raw = lsGet(k);
      if (raw != null) return raw;
      const p = pending.get("kv\0" + k); if (p) return p.data ? p.data.raw : null;
      const r = cache.kv && cache.kv.get(k); return r && !r.deleted && r.data && (!ls || r.nomirror) ? r.data.raw : null;
    }
    function setRaw(k, raw) {
      raw = String(raw);
      if (sealed) return;
      if (LOCAL_ONLY.has(k)) { lsSet(k, raw); return; }
      const full = !lsSet(k, raw) && !!ls;
      if (full) lsDel(k);   // (localStorage full: not an older value left in the mirror)
      queue("kv", k, { raw }, 0, full ? { nomirror: true } : null);
    }
    function remove(k) { lsDel(k); if (!LOCAL_ONLY.has(k)) queue("kv", k, null); }

    // ---- records (hands, sessions, blobs): asynchronous
    async function list(store, opts = {}) { await ready; await flushNow(); const m = await load(store); return [...m.values()].filter(r => opts.deleted || !r.deleted).map(clone); }
    async function getRec(store, id) { await ready; await flushNow(); const r = (await load(store)).get(id); return r ? clone(r) : null; }
    async function put(store, id, data, created) { await ready; queue(store, String(id), clone(data), 0, { created }); await flushNow(); return getRec(store, String(id)); }
    async function del(store, id) { await ready; queue(store, String(id), null); await flushNow(); }
    function scrub(x, key) {
      if (!key || key.length < 8) return x;
      if (typeof x === "string") return x.split(key).join("[removed]");
      if (Array.isArray(x)) return x.map(v => scrub(v, key));
      if (x && typeof x === "object") { const o = {}; for (const k of Object.keys(x)) o[k] = scrub(x[k], key); return o; }
      return x;
    }
    async function backup() {
      await ready; await flushNow(); await reconcile().catch(() => {});
      const out = { format: BACKUP, schema: SCHEMA, exported: new Date(now()).toISOString(), device: meta.device, counts: {}, stores: {} };
      for (const s of STORES) {
        out.stores[s] = (await list(s)).filter(r => !(s === "kv" && (LOCAL_ONLY.has(r.id) || !adoptable(r.id))));
        out.counts[s] = out.stores[s].length;
      }
      return scrub(out, lsGet(AI_KEY));
    }
    const valid = (r) => r && typeof r === "object" && typeof r.id === "string" && r.id && !r.deleted && r.data !== undefined && r.data !== null;
    // a backup's records merged in: one missing here (or deleted here) comes back, of two the newer stays; nothing here
    // is deleted. Settings (kv) take effect when the page is loaded again. Hands already here (the same hand under
    // another id) aren't added twice: the backup's tags join, its note fills an empty one.
    async function restore(obj) {
      if (!obj || obj.format !== BACKUP || !obj.stores || typeof obj.stores !== "object") throw new Error("That isn't a GTO Solver backup file.");
      await ready; await flushNow();
      const t = now(), out = { added: 0, updated: 0, kept: 0, stores: {} }, batch = [];
      const sigs = new Map();
      for (const r of (await load("hands")).values()) if (!r.deleted && r.data) sigs.set(handSig(r.data.state), r);
      for (const s of STORES) {
        const recs = Array.isArray(obj.stores[s]) ? obj.stores[s] : [], m = await load(s), c = out.stores[s] = { added: 0, updated: 0, kept: 0 };
        for (const r0 of recs) {
          if (!valid(r0) || s === "kv" && (!adoptable(r0.id) || typeof r0.data.raw !== "string")) continue;
          const { nomirror: _nm, ...r1 } = clone(r0), r = { ...r1, v: SCHEMA }, here = m.get(r.id);
          r.created = +r.created || t; r.updated = +r.updated || t; r.dev = r.dev || "backup";
          if (s === "hands") {
            const dup = r.data && r.data.state && sigs.get(handSig(r.data.state));
            if (dup && dup.id !== r.id) {
              const d = dup.data, tags = [...new Set([...(d.tags || []), ...(r.data.tags || [])])], note = d.note || r.data.note || "";
              if (tags.length !== (d.tags || []).length || note !== (d.note || "")) { const n = record(dup.id, { ...d, tags, note }, dup, t); m.set(n.id, n); batch.push({ store: s, rec: n }); c.updated++; }
              else c.kept++;
              continue;
            }
          }
          if (!here || here.deleted) { if (here) r.updated = Math.max(r.updated, here.updated + 1, t); m.set(r.id, r); batch.push({ store: s, rec: r }); c.added++; }
          else if (r.updated > here.updated && canon(r.data) !== canon(here.data)) { m.set(r.id, r); batch.push({ store: s, rec: r }); c.updated++; }
          else { c.kept++; continue; }
          if (s === "kv" && !lsSet(r.id, r.data.raw)) { lsDel(r.id); if (ls) r.nomirror = true; }
          if (s === "hands" && r.data.state) sigs.set(handSig(r.data.state), r);
        }
        out.added += c.added; out.updated += c.updated; out.kept += c.kept;
      }
      await backend.write(batch);
      if (batch.length) changed([...new Set(batch.map(b => b.store))]);
      return out;
    }

    // ---- Delete everything this site keeps: what's here and in the other named places, nothing else on the origin.
    // The records and the mirror go together or not at all. A load takes a mirror key back in as a record and writes a
    // lost mirror back from the records (reconcile), so with the records still there the mirror stays and the page goes
    // on as it was; only the keys that are never records go (the API key). The report says what went and what didn't:
    //   { ok, records: the records and the mirror went, deleted: { hands, sessions, kv, blobs }, aiKey, keys, databases,
    //     caches, failed: [{ what: "records", a database's or a cache's name, or "tab", why }] }
    // Once the records are gone the store is sealed until the page loads again (and other tabs of this code with it,
    // onChange's { wiped }): what the page holds in memory, the hand on the table first, isn't written back. A tab of the
    // frozen copy can't be told more than to read again, so the page asks for other tabs to be closed first.
    async function wipe() {
      await ready;
      seal();
      await flushing;
      const out = { ok: false, records: false, deleted: {}, aiKey: false, keys: [], databases: [], caches: [], failed: [] };
      const fail = (what, e) => out.failed.push({ what, why: e && e.message || String(e || "") });
      const counts = {}, ids = [];
      for (const s of STORES) {
        counts[s] = 0;
        try { const m = await load(s); counts[s] = [...m.values()].filter(r => !r.deleted).length; if (s === "kv") ids.push(...m.keys()); } catch (e) {}
      }
      const whole = SITE_DATABASES[DB_NAME];
      try { await backend.clear(whole); out.records = true; }
      catch (e) {
        if (!backend.drop) fail("records", e);
        else try { await backend.drop(env.wipeWait || 5000); out.records = true; } catch (e2) { fail("records", e2); }
      }
      for (const s of STORES) delete cache[s];
      if (out.records) {
        out.deleted = counts;
        out.databases.push(DB_NAME);
        meta = { ...meta, device: uid() };   // (what's been migrated stays marked: nothing is taken in again)
        try { await backend.write([{ store: "meta", rec: meta }]); } catch (e) {}
      }
      if (idb) {
        // (running without IndexedDB, it wouldn't open: whatever it holds couldn't be read, so couldn't be said to be gone)
        if (backend.kind !== "indexeddb") try { if (await clearDb(idb, DB_NAME, whole)) out.databases.push(DB_NAME); } catch (e) { fail(DB_NAME, e); }
        for (const name of Object.keys(SITE_DATABASES)) {
          if (name === DB_NAME) continue;
          try { if (await clearDb(idb, name, SITE_DATABASES[name])) out.databases.push(name); } catch (e) { fail(name, e); }
        }
      }
      out.aiKey = lsGet(AI_KEY) != null;
      for (const k of out.records ? new Set([...SITE_KEYS, ...ids.filter(adoptable)]) : LOCAL_ONLY) {
        if (lsGet(k) != null) out.keys.push(k);
        lsDel(k);
        try { if (ss) ss.removeItem(k); } catch (e) {}
      }
      // (a tab that wrote in the meantime: this is what's there now, not what was cleared)
      if (out.records) try { for (const s of STORES) if ((await backend.getAll(s)).length) { fail("tab", new Error("another tab saved again")); break; } } catch (e) {}
      if (cs) for (const name of SITE_CACHES) { try { if (await cs.delete(name)) out.caches.push(name); } catch (e) { fail(name, e); } }
      out.ok = !out.failed.length;
      sealed = out.records;
      changed(STORES, false, out.records);
      return out;
    }

    // ---- My hands: the hands you've played on the table, reviewed or imported, each a record of
    //   { state: the page's hand (S: setup, cards, actions with their grades), sig: the hand without its grades (the
    //     same hand saved again is the same record), source: "table" | "review" | "import" | "file" | "link",
    //     reviewed: graded decisions in it, tags: [], note: "", site / hh: an imported hand's site and history }
    // The page tells it (gto-hands-table) which hand is on the table: the record it is (opened from My hands, saved,
    // imported), or a spot put there (the frozen copy's write-ups, drills and reports: kept only once you play on from it).
    const TABLE = "gto-hands-table";
    function handSig(st) {
      if (!st || typeof st !== "object") return "";
      const { finished, reviewed, ...rest } = st, s = { ...rest, actions: {} };   // (a hand's review, not the hand)
      for (const k of STREETS) s.actions[k] = (st.actions && Array.isArray(st.actions[k]) ? st.actions[k] : []).map(a => { const { g, ...rest } = a || {}; return rest; });
      return hash(canon(s));
    }
    // the site's hand number in a hand history ("PokerStars Hand #2498…", "Poker Hand #HD20…", iPoker's gamecode)
    function handNumber(text) {
      const x = String(text).match(/gamecode="([^"]+)"/);
      if (x) return x[1];
      const m = String(text).split("\n").slice(0, 3).join(" ").match(/(?:Hand|Game)\s*(?:No|Id)?\s*[:#]*\s*#?\s*([A-Z]*\d[\w-]*)/i);
      return m ? m[1] : null;
    }
    const actionsOf = (st) => STREETS.flatMap(k => st && st.actions && Array.isArray(st.actions[k]) ? st.actions[k] : []);
    const worth = (st) => !!st && typeof st === "object" && actionsOf(st).length > 0;
    const graded = (st) => actionsOf(st).some(a => a && a.g);
    // b is a plays on from a (the same setup and cards so far, each street's actions a's and then more)
    function extendsHand(a, b) {
      if (!a || !b) return false;
      const setup = (s) => { const { cards, actions, finished, reviewed, ...r } = s; return canon(r); };
      if (setup(a) !== setup(b)) return false;
      for (let i = 0; i < 7; i++) if (a.cards && a.cards[i] && (!b.cards || a.cards[i] !== b.cards[i])) return false;
      const bare = (x) => { const { g, ...r } = x || {}; return canon(r); };
      return STREETS.every(k => { const x = (a.actions || {})[k] || [], y = (b.actions || {})[k] || []; return x.length <= y.length && x.every((v, i) => bare(v) === bare(y[i])); });
    }
    async function saveNow(st, opts) {
      await ready;
      const sig = handSig(st), m = await load("hands");
      let base = opts.id ? m.get(opts.id) : null;
      if (base && (base.deleted || !base.data || !(base.data.sig === sig || extendsHand(base.data.state, st)))) base = null;
      if (!base) for (const r of m.values()) if (!r.deleted && r.data && r.data.sig === sig) { base = r; break; }
      const d = base ? base.data : { tags: [], note: "", source: opts.source || "table" };
      const data = { ...d, state: st, sig, reviewed: graded(st) || !!d.reviewed && d.sig === sig };
      if (opts.source === "import") data.source = "import";   // (a hand you'd entered, imported from its history after)
      for (const k of ["site", "hh", "hhId"]) if (opts[k]) data[k] = opts[k];
      if (opts.tags) data.tags = [...new Set([...(data.tags || []), ...opts.tags])];
      if (opts.note && !data.note) data.note = opts.note;
      const id = base ? base.id : uid();
      const rec = await put("hands", id, data);
      if (opts.table !== false) hands.onTable(st, { id, source: data.source });
      return rec;
    }
    async function updateNow(id, patch) {
      const r = await getRec("hands", id);
      if (!r || r.deleted) return null;
      const data = { ...r.data };
      if (Array.isArray(patch.tags)) data.tags = [...new Set(patch.tags.map(x => String(x).trim()).filter(Boolean))];
      if (typeof patch.note === "string") data.note = patch.note;
      if (patch.state) { data.state = clone(patch.state); data.sig = handSig(patch.state); data.reviewed = graded(patch.state); }
      return put("hands", id, data);
    }
    // (a hand's writes one after another: a save and an edit at once, each read then written, don't undo each other)
    let handsQ = Promise.resolve();
    const serial = (f) => { const p = handsQ.then(f, f); handsQ = p.catch(() => {}); return p; };
    const hands = {
      sig: handSig, worth, extendsHand, handNumber,
      table: () => parse(getRaw(TABLE)) || {},
      // the hand now on the table: the record it is (id), a spot (spot), or where it came from (source)
      onTable(st, opts = {}) { setRaw(TABLE, JSON.stringify({ id: opts.id || null, sig: handSig(st), spot: !!opts.spot, ...(opts.source ? { source: opts.source } : {}) })); },
      async list() { return (await list("hands")).sort((a, b) => b.created - a.created); },
      get: (id) => getRec("hands", id),
      // save a hand: into the record it already is (same hand, or one it plays on from), else a new one
      save(st, opts = {}) { if (!worth(st)) return Promise.resolve(null); st = clone(st); return serial(() => saveNow(st, opts)); },
      update: (id, patch) => serial(() => updateNow(id, patch)),
      // the page's hooks: "replaced" (the hand on the table is about to go: kept unless it's an untouched spot), "review" (a whole-hand review finished), "import" (an imported hand). The hand is copied at once.
      keep(how, st, meta = {}) {
        if (!worth(st)) return Promise.resolve(null);
        st = clone(st);
        const t = hands.table(), sig = handSig(st);
        if (how === "replaced" && t.spot && t.sig === sig) return Promise.resolve(null);
        const source = how === "import" ? "import" : meta.source || (t.spot ? "table" : t.source) || "table";
        const hh = typeof meta.hh === "string" && meta.hh.trim() ? meta.hh.trim().slice(0, 20000) : null;
        return hands.save(st, { ...meta, hh, hhId: meta.hhId || (hh ? handNumber(hh) : null), id: t.id, source,
                                table: how !== "replaced" && meta.table !== false }).catch(() => null);
      },
      remove: (id) => del("hands", id),
      // (Undo after a delete: the record as it was, back under its id)
      restore: (rec) => put("hands", rec.id, rec.data, +rec.created || 0),
      // a My hands export file ({ format: "gto-solver-hands", hands: [{ state, link, tags, note, ... }] }) or a list of
      // states; an entry with only its #h= link (decode: the page's decodeState) is that link's hand
      async importFile(obj, clean, decode) {
        const list0 = obj && obj.format === HANDS_FILE && Array.isArray(obj.hands) ? obj.hands : Array.isArray(obj) ? obj : obj && obj.players && obj.actions ? [obj] : null;
        if (!list0) throw new Error("That file has no hands in it (a My hands export, or a hand's state JSON).");
        const out = { added: 0, merged: 0, skipped: 0 };
        const before = new Set((await hands.list()).map(r => r.id));
        for (const h0 of list0) {
          let h = h0 && h0.state ? h0 : { state: h0 };
          if (h0 && !h0.state && !h0.players && typeof h0.link === "string") {
            const m = h0.link.match(/#h=([A-Za-z0-9+/=]+)/);
            let st = null; try { st = decode && m ? decode(m[1]) : null; } catch (e) {}
            h = { ...h0, state: st };
          }
          const st = clean ? clean(h.state) : h.state;
          if (!worth(st)) { out.skipped++; continue; }
          const rec = await hands.save(st, { source: h.source || "file", tags: Array.isArray(h.tags) ? h.tags : [], note: typeof h.note === "string" ? h.note : "",
                                             site: h.site, hh: h.hh, hhId: h.hhId, table: false });
          if (rec && before.has(rec.id)) out.merged++; else if (rec) { out.added++; before.add(rec.id); }
        }
        return out;
      },
      // (linkOf: each hand's #h= link, the page's, written beside its state; the stored Anthropic key's text, should a
      // note or a history hold it, is scrubbed as in a backup, the link made from the scrubbed state)
      exportFile(recs, linkOf) {
        recs = scrub(recs.map(r => ({ id: r.id, created: r.created, data: r.data })), lsGet(AI_KEY));
        return { format: HANDS_FILE, version: 1, exported: new Date(now()).toISOString(),
                 hands: recs.map(r => ({ id: r.id, created: new Date(r.created).toISOString(), source: r.data.source, tags: r.data.tags || [], note: r.data.note || "",
                                         ...(r.data.site ? { site: r.data.site } : {}), ...(r.data.hhId ? { hhId: r.data.hhId } : {}), ...(r.data.hh ? { hh: r.data.hh } : {}),
                                         ...(linkOf ? { link: linkOf(r.data.state) } : {}), state: r.data.state })) };
      },
    };

    return {
      ready, SCHEMA, STORES, KNOWN, LOCAL_ONLY, DEVICE_ONLY, BACKUP, HANDS_FILE, SITE_KEYS, SITE_DATABASES, SITE_CACHES,
      get backend() { return backend ? backend.kind : null; }, get device() { return meta ? meta.device : null; },
      get: (k) => parse(getRaw(k)), getRaw, set: (k, v) => setRaw(k, JSON.stringify(v)), setRaw, remove,
      list, record: getRec, put, del, flush: flushNow, reconcile: () => ready.then(reconcile),
      backup, restore, wipe, hands, onChange: (f) => { listeners.add(f); return () => listeners.delete(f); },
      _: { canon, hash, pending, cache, meta: () => meta },
    };
  }

  const api = { createStore, SITE_KEYS, SITE_DATABASES, SITE_CACHES };
  if (typeof module === "object" && module.exports) module.exports = api;
  else { globalThis.GTOStoreLib = api; try { globalThis.GTOStore = createStore(); } catch (e) {} }
})();
