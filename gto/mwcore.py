"""The multiway core (core/mwcore.c): MultiwaySolver.run's training iterations in C for the solves whose later streets
are bet on by hand buckets (MWConfig.later = "buckets": every 3-way flop and turn the advisor solves), natively (a
shared library built from the source on first use with the system's C compiler) and in the browser (gto/mwcore.wasm,
which worker.js instantiates before the solver loads: js.mwCore). Python still builds the tree and the bucket tables
and reads strategies off it; the core takes the tree, the tables, the reaches, the regrets and the strategy sums, runs
iterations in chunks (the time budget and should_stop checked between them), and the regrets and sums come back to the
solver's arrays after each run. It also runs the passes after a solve: each player's value on the average strategies,
its best response, and a root-street node's action EVs.

The arithmetic is the numpy solver's in float32, so their strategies agree to rounding (tests/test_multiway_solver.py).
One program at a time lives in the core's arena: another solver's replaces it, and a solver whose program is gone lays
its tree out again from its arrays. GTO_MWCORE=0 turns the core off (the numpy passes).
"""
from __future__ import annotations
import hashlib
import os
import subprocess
import sys
import time
import weakref

import numpy as np

MAGIC = 0x6d776332
RDEC, RFOLD, REXIT, BDEC, BTERM, BEXIT, BTERMW = 1, 2, 3, 4, 5, 6, 7
ALT, BR, EV = 0, 1, 2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "core", "mwcore.c")
WASM = os.path.join(ROOT, "gto", "mwcore.wasm")
CHUNK_S = 0.1          # seconds of iterations a call (between the budget and should_stop checks)
MAX_CHUNK = 256        # iterations a call at most (the discount factors' buffer)
MAX_K = 32             # hand buckets at most (the kernels' accumulators)
ARENA_MAX = 1 << 30

_lib = None
_gen = 0


class Unsupported(Exception):
    """A solve the core doesn't run (MultiwaySolver.run runs it in numpy)."""


def source_hash() -> str:
    with open(SRC, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def build_native(out_dir: str = None) -> str:
    """Compiles core/mwcore.c into a shared library (once per source: its hash is in the name); returns its path."""
    out_dir = out_dir or os.path.join(ROOT, "core", "build")
    path = os.path.join(out_dir, f"mwcore-{source_hash()[:12]}.so")
    if not os.path.exists(path):
        os.makedirs(out_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        cc = os.environ.get("CC", "cc")
        subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", tmp, SRC], check=True, capture_output=True)
        os.replace(tmp, path)
    return path


class _Native:
    def __init__(self, path: str):
        import ctypes
        self.ctypes = ctypes
        L = ctypes.CDLL(path)
        L.mw_alloc.restype, L.mw_alloc.argtypes = ctypes.c_uint32, [ctypes.c_uint32]
        L.mw_base.restype, L.mw_base.argtypes = ctypes.c_void_p, []
        L.mw_reset.restype, L.mw_reset.argtypes = None, []
        L.mw_run.restype, L.mw_run.argtypes = ctypes.c_int32, [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        L.mw_eval.restype = ctypes.c_int32
        L.mw_eval.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32] + [ctypes.c_uint32] * 3
        if L.mw_version() != MAGIC:
            raise RuntimeError("mwcore version mismatch")
        self.L = L
        self.base = 0

    def reset(self):
        self.L.mw_reset()
        self.base = self.L.mw_base()

    def alloc(self, nbytes: int) -> int:
        return self.L.mw_alloc(nbytes)

    def write(self, off: int, a: np.ndarray):
        a = np.ascontiguousarray(a)
        self.ctypes.memmove(self.base + off, a.ctypes.data, a.nbytes)

    def read(self, off: int, out: np.ndarray):
        self.ctypes.memmove(out.ctypes.data, self.base + off, out.nbytes)

    def run(self, prog: int, iters: int, factors: int) -> int:
        return self.L.mw_run(prog, iters, factors)

    def eval(self, prog: int, at: int, mode: int, p: int, R: int, out: int, kids: int) -> int:
        return self.L.mw_eval(prog, at, mode, p, R, out, kids)


class _Wasm:
    """The core in the browser: the module's exports (worker.js), its arena in its own memory, copied to and from."""

    def __init__(self, ex):
        if int(ex.mw_version()) & 0xffffffff != MAGIC:
            raise RuntimeError("mwcore version mismatch")
        self.ex = ex
        self.base = 0

    def reset(self):
        self.ex.mw_reset()
        self.base = int(self.ex.mw_base()) & 0xffffffff

    def alloc(self, nbytes: int) -> int:
        return int(self.ex.mw_alloc(nbytes)) & 0xffffffff

    def _view(self, off: int, nbytes: int):
        from js import Uint8Array      # (the memory's buffer changes when it grows: a fresh view every time)
        return Uint8Array.new(self.ex.memory.buffer, self.base + off, nbytes)

    def write(self, off: int, a: np.ndarray):
        a = np.ascontiguousarray(a)
        self._view(off, a.nbytes).assign(a.reshape(-1).view(np.uint8))

    def read(self, off: int, out: np.ndarray):
        self._view(off, out.nbytes).assign_to(out.reshape(-1).view(np.uint8))

    def run(self, prog: int, iters: int, factors: int) -> int:
        return int(self.ex.mw_run(prog, iters, factors))

    def eval(self, prog: int, at: int, mode: int, p: int, R: int, out: int, kids: int) -> int:
        return int(self.ex.mw_eval(prog, at, mode, p, R, out, kids))


def _load():
    if os.environ.get("GTO_MWCORE", "1") == "0":
        return None
    if sys.platform == "emscripten":
        try:
            from js import mwCore
            return _Wasm(mwCore) if mwCore else None
        except Exception:
            return None
    try:
        L = _Native(build_native())
    except Exception:
        return None
    return L


def lib():
    """The core, or None (no compiler or module, or GTO_MWCORE=0)."""
    global _lib
    if _lib is None:
        _lib = _load() or False
    return _lib or None


def _f(x) -> int:
    return int(np.array(x, dtype=np.float32).view(np.int32))


def _mask(players) -> int:
    m = 0
    for p in players:
        m |= 1 << p
    return m


class Program:
    """A MultiwaySolver's tree (bucket mode) as the core's program, its arrays in the core's arena (reset first)."""

    def __init__(self, L, s):
        global _gen
        from . import multiway_solver as MW
        self.MW = MW
        cfg = s.cfg
        if s.tabs is None or not s.levels or not cfg.alternate or s.np not in (2, 3, 4) \
                or (s._pw and s.np != 4):
            raise Unsupported("not a bucketed 2-, 3- or 4-player solve")
        if s.tabs.K > MAX_K or len(cfg.allin) > 1:
            raise Unsupported("buckets or all-ins")
        self.L, self.s = L, weakref.ref(s)
        L.reset()
        _gen += 1
        self.gen = _gen
        self.bytes = 0
        tb = s.tabs
        self.np, self.n, self.K, self.C = s.np, s.hands.n, tb.K, tb.C
        self.nflag = 0                # records with a current strategy kept (its flag: 1 while it's current)
        self.at = {}                  # id(root-street node) -> its record
        self._bitc = {}
        # every decision's regrets, sums and kept strategies in three runs (one copy each way: push, pull)
        self.decs = list(s.decisions)
        sizes = np.array([d.regret.size for d in self.decs], dtype=np.int64)
        self.total = int(sizes.sum())
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]) * 4
        self.reg = self._space(4 * self.total)
        self.ssum = self._space(4 * self.total)
        sig = self._space(4 * self.total)
        self._dec = {id(d): (self.reg + int(o), self.ssum + int(o), sig + int(o)) for d, o in zip(self.decs, starts)}
        self.push(flags=False)
        self.w = [MAGIC, s.np, self.n, self.K, self.C, 0,
                  self._up(s.hands.ca, np.int32), self._up(s.hands.cb, np.int32),
                  self._up(tb.masks, np.float32), self._up(tb.wmask, np.float32),
                  self._up(np.ascontiguousarray(tb.ids), np.int32), 0, 0, 0,
                  _mask(cfg.allin), self._up(s.reach0[:, 0, :], np.float32), 0, 0, 0, 0, 0, 0]
        if s.np == 4:      # (the root street's deals' weight: pairwise, each player's overlap correction)
            self.w[21] = self._up(np.concatenate([s._pw_corr(p) for p in range(4)]), np.float32)
        # each player's combos per class sorted by bucket (to_buckets sums a bucket's run): the order, each bucket's
        # start, and the class's mask in that order
        ids = np.asarray(tb.ids)
        perm = np.argsort(ids, axis=2, kind="stable")
        seg = np.zeros((npl := s.np, self.C, self.K + 1), dtype=np.int32)
        for i in range(npl):
            for t in range(self.C):
                seg[i, t] = np.searchsorted(ids[i, t][perm[i, t]], np.arange(self.K + 1))
        self.w[16] = self._up(perm, np.int32)
        self.w[17] = self._up(seg, np.int32)
        self.w[18] = self._up(np.take_along_axis(np.broadcast_to(tb.masks, ids.shape), perm, axis=2), np.float32)
        table = np.zeros(npl + (1 << npl) * npl, dtype=np.int64)
        for (key, p), T in tb.tp.items():
            table[p if key == ("J",) else npl + _mask(key[1]) * npl + p] = self._up(T, np.float32)
        self.w[11] = self._up(table.astype(np.uint32).view(np.int32), np.int32)
        # each root Exit's block in its level-0 group, each BExit block's in its next street's
        self._exit = {}
        for g in s.levels[0]:
            for i, (e, _) in enumerate(g.blocks):
                self._exit[id(e)] = (g, i)
        self._bx = {}
        for lvl in s.levels[1:]:
            for g in lvl:
                for i, (bx, j) in enumerate(g.blocks):
                    self._bx[(id(bx), j)] = (g, i)
        self.w[5] = self._root(s.root)
        # (the discount: one run of regrets and one of sums)
        arrs = np.array([[self.reg, self.ssum, self.total]], dtype=np.int64)
        self.w[12] = 1
        self.flags = self._up(np.zeros(max(self.nflag, 1), dtype=np.int32), np.int32)
        self.w[19] = self.flags
        self.w[20] = self.nflag
        self.w[13] = self._up(arrs.astype(np.uint32).view(np.int32), np.int32)
        words = np.array(self.w, dtype=np.int64)
        self.prog = self._up(words.astype(np.uint32).view(np.int32), np.int32)
        self.fac = L.alloc(8 * MAX_CHUNK)
        self.R = L.alloc(4 * npl * self.n)
        self.out = L.alloc(4 * self.n)
        self.kids = L.alloc(4 * self.n * 16)
        self.w = self._bitc = None

    # -- layout
    def _up(self, a, dtype) -> int:
        arr = np.ascontiguousarray(a, dtype=dtype)
        self.bytes += arr.nbytes
        if self.bytes > ARENA_MAX:
            raise Unsupported("over ARENA_MAX")
        off = self.L.alloc(max(arr.nbytes, 4))
        self.L.write(off, arr)
        return off

    def _arrays(self, d):
        """The decision's regret, ssum and current-strategy offsets (the last filled in by the core)."""
        hit = self._dec.get(id(d))
        if hit is None:
            raise Unsupported("a decision not in the solver's list")
        return hit

    def _space(self, nbytes) -> int:
        self.bytes += nbytes
        if self.bytes > ARENA_MAX:
            raise Unsupported("over ARENA_MAX")
        return self.L.alloc(max(nbytes, 4))

    def _bits(self, node, attr):
        """A bucket node's per-row numbers as float32 bit patterns (int32), once a node (every block reads them)."""
        key = (id(node), attr)
        hit = self._bitc.get(key)
        if hit is None:
            hit = self._bitc[key] = np.ascontiguousarray(getattr(node, attr), dtype=np.float32).view(np.int32)
        return hit

    def _flag(self) -> int:
        self.nflag += 1
        return self.nflag - 1

    def _root(self, node) -> int:
        MW, w, s = self.MW, self.w, self.s()
        at = len(w)
        self.at[id(node)] = at
        if isinstance(node, MW.Decision):
            if node.bd.B != 1:
                raise Unsupported("root street boards")
            A = len(node.children)
            reg, ss, sg = self._arrays(node)
            w += [RDEC, node.player, A, reg, ss, sg, self._flag()] + [0] * A
            for a, ch in enumerate(node.children):
                kid = self._root(ch)
                w[at + 7 + a] = kid
            return at
        if isinstance(node, MW.Terminal):
            if len(node.live) != 1 or node.bd.B != 1:
                raise Unsupported("a root-street showdown")
            if node.W is not None:    # (a tournament's: each player's value, as minus its chips in)
                v = node.W[:, 0].copy()
                v[node.live[0]] += node.W[node.live[0], 1]
                w += [RFOLD, node.live[0], _f(0.0)] + [_f(-x) for x in v]
                return at
            w += [RFOLD, node.live[0], _f(node.pot - node.rake)] + [_f(x) for x in node.c]
            return at
        if isinstance(node, MW.Exit):
            g, i = self._exit[id(node)]
            # (a player folded on the root street: minus its chips in, or a tournament's minus its fold's value)
            w += [REXIT, _mask(node.live), 0] + [_f(-node.F[p] if p in node.F else x) for p, x in enumerate(node.c)]
            w[at + 2] = self._bucket(g.tree, i)
            return at
        raise Unsupported(type(node).__name__)

    def _bucket(self, node, i) -> int:
        MW, w, C = self.MW, self.w, self.C
        at = len(w)
        r = i * C
        if isinstance(node, MW.BDecision):
            A = len(node.children)
            reg, ss, sg = self._arrays(node)
            blk = 4 * C * A * self.K
            leafc = 0 if node.fold is None else self._bits(node.leaf, "c")[node.player][r] if node.leaf.W is None \
                else _f(-node.leaf.W[node.player, 0, r])
            w += [BDEC, node.player, A, reg + i * blk, ss + i * blk, -1 if node.fold is None else node.fold,
                  leafc, sg + i * blk, self._flag()] + [0] * A
            for a, ch in enumerate(node.children):
                kid = self._bucket(ch, i)
                w[at + 9 + a] = kid
            return at
        if isinstance(node, MW.BTerminal):
            if node.W is not None:    # (a tournament's weights per player: the deals' weight's, the main pot's, the side's)
                w += [BTERMW, _mask(node.live), _mask(node.side)] + self._bits(node, "W")[:, :, r].reshape(-1).tolist()
                return at
            # (a player all-in for less: the main pot's share of the row's pot, and the side pot's players)
            fm = float(node.fm[r]) if node.fm is not None else -1.0
            w += [BTERM, _mask(node.live), _mask(node.side), self._bits(node, "net")[r], _f(fm)] + \
                self._bits(node, "c")[:, r].tolist()
            return at
        if isinstance(node, MW.BExit):
            g, j = self._bx[(id(node), i)]
            w += [BEXIT, 0]
            w[at + 1] = self._bucket(g.tree, j)
            return at
        raise Unsupported(type(node).__name__)

    # -- use
    def valid(self, L, s) -> bool:
        return self.gen == _gen and self.L is L and self.s() is s

    def push(self, flags=True):
        """The solver's regrets and sums into the arena (after numpy changed them)."""
        for off, attr in ((self.reg, "regret"), (self.ssum, "ssum")):
            if self.total:
                self.L.write(off, np.concatenate([getattr(d, attr).reshape(-1) for d in self.decs]).astype(np.float32))
        if flags:
            self.L.write(self.flags, np.zeros(max(self.nflag, 1), dtype=np.int32))   # (the kept strategies are stale)

    def pull(self):
        """The core's regrets and sums back into the solver's arrays."""
        buf = np.zeros(self.total, dtype=np.float32)
        for off, attr in ((self.reg, "regret"), (self.ssum, "ssum")):
            if not self.total:
                continue
            self.L.read(off, buf)
            at = 0
            for d in self.decs:
                a = getattr(d, attr)
                a[...] = buf[at:at + a.size].reshape(a.shape)
                at += a.size

    def run(self, iters: int, t0: int) -> int:
        """`iters` iterations, the discount's t counting from t0 (the solver's iterations before them)."""
        s = self.s()
        a, g = s.cfg.alpha, s.cfg.gamma
        t = np.arange(t0 + 1, t0 + iters + 1, dtype=np.float64)
        fac = np.stack([t ** a / (t ** a + 1), (t / (t + 1)) ** g], axis=1).astype(np.float32)
        self.L.write(self.fac, fac)
        return self.L.run(self.prog, iters, self.fac)

    def eval(self, node, mode: int, p: int, R: np.ndarray, kids: bool = False):
        """p's values (n) at a root-street node for reaches R (np, n) there; with kids, each action's (A, n) too."""
        at = self.at.get(id(node))
        if at is None:
            raise Unsupported("not a root-street node")
        self.L.write(self.R, np.ascontiguousarray(np.reshape(R, (self.np, self.n)), dtype=np.float32))
        A = len(node.children) if kids else 0
        if A > 16:
            raise Unsupported("actions")
        if self.L.eval(self.prog, at, mode, p, self.R, self.out, self.kids if A else 0) != 0:
            raise RuntimeError("mwcore: bad program")
        v = np.zeros(self.n, dtype=np.float32)
        self.L.read(self.out, v)
        if not A:
            return v.astype(np.float64)
        k = np.zeros((A, self.n), dtype=np.float32)
        self.L.read(self.kids, k)
        return v.astype(np.float64), k.astype(np.float64)


def program(s):
    """The solver's program in the core (built, or rebuilt from its arrays when another replaced it), or None."""
    L = lib()
    if L is None or getattr(s, "_mwc_off", False):
        return None
    prog = getattr(s, "_mwc", None)
    if prog is not None and prog.valid(L, s):
        return prog
    try:
        prog = Program(L, s)
    except Unsupported:
        s._mwc_off = True
        return None
    s._mwc = prog
    return prog
