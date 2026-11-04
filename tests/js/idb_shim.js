// A small in-memory IndexedDB for the Node tests (no npm packages): open with versions and onupgradeneeded (an upgrade
// whose transaction is aborted leaves no database, as the browser does), object stores with a keyPath or out-of-line
// keys, transactions that complete once their requests are answered (each answered on its own turn, in order) and
// roll back on abort, and get / getAll / getAllKeys / put / delete / clear. Values are structured clones.
// fakeIndexedDB({ failOpen: true }) refuses every open (a browser whose IndexedDB is off); on the factory, failClear makes a
// store's clear() fail, failDelete a deleteDatabase, blockDelete leaves one waiting for good (a connection that won't close).
"use strict";

const tick = (f) => setImmediate(f);
const err = (name, message) => Object.assign(new Error(message || name), { name });

class Request {
  constructor(source) { this.source = source; this.result = undefined; this.error = null; this.readyState = "pending"; this.onsuccess = null; this.onerror = null; }
}

class Transaction {
  constructor(db, names, mode) {
    this.db = db; this.mode = mode; this.names = names; this.queue = []; this.done = false; this.error = null;
    this.oncomplete = null; this.onerror = null; this.onabort = null;
    // a readwrite transaction rolls back on abort: the stores as they were
    this.saved = mode === "readonly" ? null : new Map(names.map(n => [n, new Map(db._stores.get(n).data)]));
    tick(() => this._run());   // (one chain answers the requests in order; it ends, committed, when none is left)
  }
  objectStore(name) {
    if (!this.names.includes(name)) throw err("NotFoundError", `no object store ${name} in this transaction`);
    return new ObjectStore(this, name);
  }
  _req(fn) {
    if (this.done || this.aborting) throw err("TransactionInactiveError");
    const r = new Request(this); this.queue.push([r, fn]);
    return r;
  }
  _run() {
    if (this.done || this.aborting) return;
    const next = this.queue.shift();
    if (!next) { this.done = true; if (this.oncomplete) this.oncomplete({ target: this }); return; }
    const [r, fn] = next;
    try { r.result = fn(); r.readyState = "done"; if (r.onsuccess) r.onsuccess({ target: r }); }
    catch (e) { r.error = e; r.readyState = "done"; if (r.onerror) r.onerror({ target: r }); return this._abort(e); }
    tick(() => this._run());
  }
  _abort(e) {
    if (this.done) return;
    this.done = true; this.error = e || err("AbortError");
    if (this.saved) for (const [n, m] of this.saved) this.db._stores.get(n).data = m;
    if (this.onabort) this.onabort({ target: this });
    if (this._onAbortInternal) this._onAbortInternal();
  }
  abort() { this.queue = []; this.aborting = true; tick(() => { this.aborting = false; this._abort(err("AbortError")); }); }
}

class ObjectStore {
  constructor(tx, name) { this.tx = tx; this.name = name; }
  get _s() { return this.tx.db._stores.get(this.name); }
  _key(v, key) {
    const kp = this._s.keyPath;
    if (kp) { if (key !== undefined) throw err("DataError", "a key given for a store with a keyPath"); const k = v && v[kp]; if (k === undefined || k === null) throw err("DataError", "no key in the value"); return k; }
    if (key === undefined) throw err("DataError", "no key");
    return key;
  }
  _write() { if (this.tx.mode === "readonly") throw err("ReadOnlyError"); }
  get(key) { return this.tx._req(() => { const v = this._s.data.get(key); return v === undefined ? undefined : structuredClone(v); }); }
  getAll() { return this.tx._req(() => [...this._s.data.keys()].sort(cmp).map(k => structuredClone(this._s.data.get(k)))); }
  getAllKeys() { return this.tx._req(() => [...this._s.data.keys()].sort(cmp)); }
  put(v, key) { this._write(); const k = this._key(v, key), c = structuredClone(v); return this.tx._req(() => { this._s.data.set(k, c); return k; }); }
  add(v, key) { this._write(); const k = this._key(v, key), c = structuredClone(v); return this.tx._req(() => { if (this._s.data.has(k)) throw err("ConstraintError"); this._s.data.set(k, c); return k; }); }
  delete(key) { this._write(); return this.tx._req(() => { this._s.data.delete(key); }); }
  clear() { this._write(); return this.tx._req(() => { if (this.tx.db._rec.factory && this.tx.db._rec.factory.failClear) throw err("UnknownError", "clear failed"); this._s.data.clear(); }); }
}
const cmp = (a, b) => (typeof a === typeof b ? (a < b ? -1 : a > b ? 1 : 0) : typeof a === "number" ? -1 : 1);

class DB {
  constructor(rec) { this._rec = rec; this.name = rec.name; this.closed = false; this.onversionchange = null; }
  get version() { return this._rec.version; }
  get _stores() { return this._rec.stores; }
  get objectStoreNames() { const names = [...this._rec.stores.keys()].sort(); return { contains: (n) => this._rec.stores.has(n), length: names.length, item: (i) => names[i], [Symbol.iterator]: () => names[Symbol.iterator]() }; }
  createObjectStore(name, opts = {}) {
    if (!this._upgrading) throw err("InvalidStateError", "not in an upgrade");
    if (this._rec.stores.has(name)) throw err("ConstraintError", `store ${name} exists`);
    this._rec.stores.set(name, { keyPath: opts.keyPath || null, data: new Map() });
    return new ObjectStore(this._upgrading, name);
  }
  deleteObjectStore(name) { this._rec.stores.delete(name); }
  transaction(names, mode = "readonly") {
    if (this.closed) throw err("InvalidStateError", "the database is closed");
    names = Array.isArray(names) ? names : [names];
    for (const n of names) if (!this._rec.stores.has(n)) throw err("NotFoundError", `no object store ${n}`);
    return new Transaction(this, names, mode);
  }
  close() { this.closed = true; }
}

function fakeIndexedDB(opts = {}) {
  const dbs = new Map();   // name -> { name, version, stores: Map(name -> { keyPath, data }) }
  const factory = {
    opened: [],   // the names open() was asked for, in order
    open(name, version) {
      factory.opened.push(name);
      const q = new Request(null); q.onupgradeneeded = null; q.onblocked = null; q.transaction = null;
      tick(() => {
        if (opts.failOpen || factory.failOpen) { q.error = err("UnknownError", "IndexedDB is off"); if (q.onerror) q.onerror({ target: q }); return; }
        let rec = dbs.get(name);
        const existed = !!rec, old = existed ? rec.version : 0;
        const v = version === undefined ? (existed ? rec.version : 1) : version;
        if (!(v >= 1) || v < old) { q.error = err("VersionError"); if (q.onerror) q.onerror({ target: q }); return; }
        if (!existed) { rec = { name, version: v, stores: new Map(), factory }; dbs.set(name, rec); }
        const db = new DB(rec);
        if (v > old) {
          const before = new Map([...rec.stores].map(([n, s]) => [n, s]));
          rec.version = v;
          const tx = new Transaction(db, [], "versionchange");
          db._upgrading = tx; q.transaction = tx; q.result = db;
          let aborted = false;
          tx._onAbortInternal = () => {
            aborted = true;
            if (!existed) dbs.delete(name); else { rec.version = old; rec.stores = before; }
            q.error = err("AbortError"); if (q.onerror) q.onerror({ target: q });
          };
          try { if (q.onupgradeneeded) q.onupgradeneeded({ target: q, oldVersion: old, newVersion: v }); } catch (e) { tx._abort(e); return; }
          tx.oncomplete = () => { db._upgrading = null; q.transaction = null; if (!aborted && q.onsuccess) q.onsuccess({ target: q }); };
          return;
        }
        q.result = db; if (q.onsuccess) q.onsuccess({ target: q });
      });
      return q;
    },
    deleteDatabase(name) {
      const q = new Request(null);
      if (factory.blockDelete) return q;
      tick(() => {
        if (factory.failDelete) { q.error = err("UnknownError", "delete failed"); if (q.onerror) q.onerror({ target: q }); return; }
        dbs.delete(name); if (q.onsuccess) q.onsuccess({ target: q });
      });
      return q;
    },
    async databases() { return [...dbs.values()].map(d => ({ name: d.name, version: d.version })); },
    // (tests) a database's store as a plain object, key -> value
    dump(name, store) { const r = dbs.get(name); if (!r || !r.stores.has(store)) return null; return Object.fromEntries([...r.stores.get(store).data].map(([k, v]) => [k, structuredClone(v)])); },
    has: (name) => dbs.has(name),
    // (tests) a value written straight into a database's store, as another tab's write would land
    poke(name, store, key, value) { dbs.get(name).stores.get(store).data.set(key, value); },
  };
  return factory;
}

// An old database as the site's code wrote it (session.js's "gto-session": an out-of-line "kv" store), for migrations
async function seedOldDb(idb, name, store, entries) {
  await new Promise((res, rej) => {
    const q = idb.open(name, 1);
    q.onupgradeneeded = () => q.result.createObjectStore(store);
    q.onerror = () => rej(q.error);
    q.onsuccess = () => {
      const d = q.result, t = d.transaction(store, "readwrite");
      for (const [k, v] of Object.entries(entries)) t.objectStore(store).put(v, k);
      t.oncomplete = () => { d.close(); res(); }; t.onerror = () => rej(t.error);
    };
  });
}

// localStorage as a browser has it (getItem, setItem, removeItem, key, length); `full` makes setItem throw (quota)
function fakeLocalStorage(init = {}) {
  const m = new Map(Object.entries(init));
  return {
    map: m, full: false,
    getItem: (k) => m.has(k) ? m.get(k) : null,
    setItem(k, v) { if (this.full) throw err("QuotaExceededError"); m.set(String(k), String(v)); },
    removeItem: (k) => { m.delete(k); },
    key: (i) => [...m.keys()][i] ?? null,
    get length() { return m.size; },
    clear: () => m.clear(),
  };
}

module.exports = { fakeIndexedDB, seedOldDb, fakeLocalStorage };
