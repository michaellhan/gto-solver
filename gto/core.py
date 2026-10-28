"""The compiled CFR core (core/gtocore.c): Solver.run's training iterations in C, natively (a shared library built from
the source on first use with the system's C compiler) and in the browser (gto/core.wasm, which worker.js instantiates
before the solver loads: js.gtoCore). Python still builds the tree and reads strategies off it; the core takes the
tree, the reaches, the regrets and the strategy sums, runs iterations in chunks (the time budget and cancelling are
checked between them), and the regrets and sums come back. It also runs the passes after a solve: EVs on the average
strategies and best responses (Solver.node_action_evs, exploitability). The last solver's program stays in the core's
memory until another solver's replaces it.

It covers turn and river trees: decisions, folds, showdowns, all-in equity leaves, the river dealt card by card or by
class (Config.river_classes), rake and a tournament's payoffs; and flop trees whose turn and river are played by hand
buckets (Config.river_buckets: Fast's flop model, the browser's Accurate's): the bucket and river-strength entries
(BucketEntry, RiverEntry) with their tables (Ctx.bucket_table, Ctx.river_table, built in the core too: Tables) and
the bucket games' folds and showdowns. A tree with anything else (sampled cards) or a predictive scheme runs in numpy
as before, as does a solve while another holds the core (one at a time: its memory is one arena), and natively a tree
of many equity leaves (LEAF_ROWS_BLAS). The arithmetic is the numpy solver's in float32, so their strategies agree to
rounding. GTO_CORE=0 turns the core off.
"""
from __future__ import annotations
import hashlib
import os
import subprocess
import sys
import threading
import time
import weakref
import numpy as np

MAGIC = 0x67746f31
DECISION, FOLD, SHOWDOWN, LEAF, CHANCE, CLASSES = 1, 2, 3, 4, 5, 6
BUCKETS, RIVER, BFOLD, BLEAF = 7, 8, 9, 10      # a flop solve's turn and river by hand buckets
AVG, BR = 1, 2         # evaluate(): both players on their average strategies; p's best response
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "core", "gtocore.c")
CHUNK_S = 0.1          # seconds of iterations a call (between the budget and should_stop checks)
# Natively numpy multiplies every equity leaf's matrices by the opponent's reaches in one threaded BLAS product a pass
# (Solver.batch_leaves), which beats the core's one-thread products once leaves are most of the work: the library's
# flop (every turn class, the river by equity at 3917 leaf rows of 529 x 529) ran 40 iterations in 17.6 s in the core,
# 4.6 s in numpy. Trees with more leaf rows than this stay in numpy there (turns have 4-6; the browser has no BLAS).
LEAF_ROWS_BLAS = 256
ARENA_MAX = 1 << 30    # bytes a program may take: offsets are 32-bit, and a browser tab's memory runs out sooner
MAX_CHUNK = 1024       # iterations a call at most (the discount factors' buffer)

_lib = None            # the loaded core; False: none
_lock = threading.Lock()
_gen = 0               # programs built: each resets the arena, so only the latest is still there
_prog = None           # that program


class Unsupported(Exception):
    """A tree the core doesn't run (Solver.run runs it in numpy)."""


def source_hash() -> str:
    with open(SRC, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def build_native(out_dir: str = None) -> str:
    """Compiles core/gtocore.c into a shared library (once per source: its hash is in the name); returns its path."""
    out_dir = out_dir or os.path.join(ROOT, "core", "build")
    path = os.path.join(out_dir, f"gtocore-{source_hash()[:12]}.so")
    if not os.path.exists(path):
        os.makedirs(out_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        cc = os.environ.get("CC", "cc")
        subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", tmp, SRC], check=True, capture_output=True)
        os.replace(tmp, path)
    return path


class _Native:
    """The core as a shared library (ctypes): its arena is this process's memory."""

    def __init__(self, path: str):
        import ctypes
        self.ctypes = ctypes
        L = ctypes.CDLL(path)
        L.core_alloc.restype, L.core_alloc.argtypes = ctypes.c_uint32, [ctypes.c_uint32]
        L.core_base.restype, L.core_base.argtypes = ctypes.c_void_p, []
        L.core_run.restype = ctypes.c_int32
        L.core_run.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        L.core_reset.restype, L.core_reset.argtypes = None, []
        L.core_eval.restype = ctypes.c_int32
        L.core_eval.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32] + [ctypes.c_uint32] * 3 + [ctypes.c_int32]
        L.core_threads.restype, L.core_threads.argtypes = ctypes.c_int32, [ctypes.c_int32]
        L.core_table.restype, L.core_table.argtypes = ctypes.c_int32, [ctypes.c_uint32]
        if L.core_version() != MAGIC:
            raise RuntimeError("core version mismatch")
        self.L = L
        self.base = 0

    def reset(self):
        self.L.core_reset()
        self.base = self.L.core_base()

    def alloc(self, nbytes: int) -> int:
        return self.L.core_alloc(nbytes)

    def write(self, off: int, a: np.ndarray):
        self.ctypes.memmove(self.base + off, a.ctypes.data, a.nbytes)

    def read(self, off: int, out: np.ndarray):
        self.ctypes.memmove(out.ctypes.data, self.base + off, out.nbytes)

    def run(self, prog: int, iters: int, factors: int) -> int:
        return self.L.core_run(prog, iters, factors)

    def eval(self, prog: int, at: int, p: int, r0: int, r1: int, out: int, md: int) -> int:
        return self.L.core_eval(prog, at, p, r0, r1, out, md)

    def threads(self, t: int) -> int:
        return self.L.core_threads(t)

    def table(self, args: int) -> int:
        return self.L.core_table(args)


class _Wasm:
    """The core in the browser: the module's exports (worker.js), its arena in its own memory, copied to and from."""

    def __init__(self, ex):
        if int(ex.core_version()) & 0xffffffff != MAGIC:
            raise RuntimeError("core version mismatch")
        self.ex = ex
        self.base = 0

    def reset(self):
        self.ex.core_reset()
        self.base = int(self.ex.core_base()) & 0xffffffff

    def alloc(self, nbytes: int) -> int:
        return int(self.ex.core_alloc(nbytes)) & 0xffffffff

    def _view(self, off: int, nbytes: int):
        from js import Uint8Array      # (the memory's buffer changes when it grows: a fresh view every time)
        return Uint8Array.new(self.ex.memory.buffer, self.base + off, nbytes)

    def write(self, off: int, a: np.ndarray):
        self._view(off, a.nbytes).assign(a.reshape(-1).view(np.uint8))

    def read(self, off: int, out: np.ndarray):
        self._view(off, out.nbytes).assign_to(out.reshape(-1).view(np.uint8))

    def run(self, prog: int, iters: int, factors: int) -> int:
        return int(self.ex.core_run(prog, iters, factors))

    def eval(self, prog: int, at: int, p: int, r0: int, r1: int, out: int, md: int) -> int:
        return int(self.ex.core_eval(prog, at, p, r0, r1, out, md))

    def threads(self, t: int) -> int:
        return int(self.ex.core_threads(t))

    def table(self, args: int) -> int:
        return int(self.ex.core_table(args))


def _load():
    if os.environ.get("GTO_CORE", "1") == "0":
        return None
    if sys.platform == "emscripten":
        try:
            from js import gtoCore
            return _Wasm(gtoCore) if gtoCore else None
        except Exception:
            return None
    try:
        L = _Native(build_native())
    except Exception:
        return None
    L.threads(int(os.environ.get("GTO_THREADS", "1")))
    return L


def lib():
    """The core, or None (no compiler or module, or GTO_CORE=0)."""
    global _lib
    if _lib is None:
        _lib = _load() or False
    return _lib or None


def threads(t: int = 0) -> int:
    """The core's threads, this one and its helpers (natively GTO_THREADS, else 1; worker.js's in the browser): sets
    them to t (unless 0) once no solve holds the core, and returns the number. A solve's numbers don't depend on it."""
    L = lib()
    if L is None:
        return 1
    with _lock:
        return L.threads(t)


def _f(x) -> int:
    """A float's float32 bits as a program word."""
    return int(np.array(x, dtype=np.float32).view(np.int32))


def _scalar(x) -> float:
    """A terminal's value: a float, or a merged node's column of one value."""
    a = np.ravel(np.asarray(x, dtype=np.float64))
    if a.size == 0 or np.any(a != a[0]):
        raise Unsupported("a value per row")
    return float(a[0])


class Program:
    """A solver's tree as the core's program, its arrays in the core's arena (the arena is reset first). r0, r1: (n,)
    reaches at the root. (It holds the solver weakly: a solve the Advisor's cache drops is freed.)"""

    def __init__(self, L, solver, r0: np.ndarray, r1: np.ndarray):
        global _gen
        from .solver import Decision, Terminal, Chance, BucketEntry, RiverEntry
        self._types = (Decision, Terminal, Chance, BucketEntry, RiverEntry)
        self.L, self.s = L, weakref.ref(solver)
        self.w = []
        self._up = {}
        self._rank = {}
        self.at = {}             # id(node) -> its record
        self.leaf_rows = 0
        self.bytes = 0
        L.reset()
        _gen += 1
        self._tables(solver)
        self.gen = _gen
        ctx = solver.ctx
        self.n = ctx.n
        self.reg = self._upload(solver._regret, np.float32)
        self.ss = self._upload(solver._ssum, np.float32)
        self.r0, self.r1 = L.alloc(4 * self.n), L.alloc(4 * self.n)
        self.reaches(r0, r1)
        self.w = [MAGIC, ctx.n, self._upload(ctx.ca, np.int32), self._upload(ctx.cb, np.int32), 0,
                  self.r0, self.r1, self.reg, self.ss, solver._regret.size]
        self.w[4] = self._emit(solver.root)
        if solver.batch_leaves and self.leaf_rows > LEAF_ROWS_BLAS:
            raise Unsupported("equity leaves: numpy's batched BLAS products")
        words = np.array(self.w, dtype=np.int64)
        self.prog = self._upload(words.astype(np.uint32).view(np.int32), np.int32)
        self.fac = L.alloc(16 * MAX_CHUNK)
        self.ev = L.alloc(12 * self.n)     # an evaluation's reaches and utilities
        self.w = self._rank = self._up = None

    def _tables(self, solver):
        """The tables the tree's bucket entries, bucket games and equity leaves read (Ctx.bucket_table, river_table,
        _avg_showdown), built in the core while the arena is empty (Tables), before the tree is laid out."""
        Decision, Terminal, Chance, BucketEntry, RiverEntry = self._types
        ctx = solver.ctx
        ctx.tables = Tables(self.L)
        try:
            stack = [solver.root]
            while stack:
                node = stack.pop()
                if isinstance(node, Decision):
                    stack.extend(node.children)
                elif isinstance(node, Terminal):
                    if node.kind[0] in "br":
                        (solver._btab if node.kind[0] == "b" else solver._rtab)(node)
                    elif node.kind == "leaf" and (len(node.boards[0]) == 3 or
                                                  len(node.boards[0]) == 4 and ctx.root_len == 3):
                        solver._terminal_M(node)    # (every runout: no random draws, whose order would change)
                elif isinstance(node, (Chance, BucketEntry, RiverEntry)):
                    if isinstance(node, RiverEntry):      # (a BucketEntry too)
                        solver._rtab(node)
                    elif isinstance(node, BucketEntry):
                        solver._btab(node)
                    stack.append(node.child)
        finally:
            ctx.tables = None
            self.L.reset()

    def valid(self, L, solver) -> bool:
        """The solver's, and still in the arena."""
        return self.gen == _gen and self.L is L and self.s() is solver

    def reaches(self, r0, r1):
        for off, r in ((self.r0, r0), (self.r1, r1)):
            r = np.ascontiguousarray(np.ravel(r), dtype=np.float32)
            if r.size != self.n:
                raise Unsupported("reaches")
            self.L.write(off, r)

    def _upload(self, a, dtype, transpose: bool = False) -> int:
        """`a` as `dtype` in the arena (once per array), its last two axes swapped with `transpose`; its offset."""
        key = (id(a), np.dtype(dtype).str, transpose)
        hit = self._up.get(key)
        if hit is not None:
            return hit[0]
        arr = np.ascontiguousarray(np.swapaxes(a, -1, -2) if transpose else a, dtype=dtype)
        self.bytes += arr.nbytes
        if self.bytes > ARENA_MAX:
            raise Unsupported("over ARENA_MAX")
        off = self.L.alloc(arr.nbytes)
        self.L.write(off, arr)
        self._up[key] = (off, a)      # (a kept alive: its id stays its own)
        return off

    def _view(self, a: np.ndarray, flat: np.ndarray, off: int) -> int:
        """A node's regrets or sums (a view of the solver's flat buffer) as an offset into the buffer's copy."""
        d = a.ctypes.data - flat.ctypes.data
        if not a.flags.c_contiguous or d < 0 or d + a.nbytes > flat.nbytes:
            raise Unsupported("not a view of the flat buffers")
        return off + d

    def _emit(self, node) -> int:
        at = self._record(node)
        self.at[id(node)] = at
        return at

    def _record(self, node) -> int:
        Decision, Terminal, Chance, BucketEntry, RiverEntry = self._types
        s, w = self.s(), self.w
        at = len(w)
        if isinstance(node, Decision):
            if node.pred is not None:
                raise Unsupported("predictive")
            A = len(node.children)
            w += [DECISION, node.player, A, node.B, node.regret.shape[2], self._view(node.regret, s._regret, self.reg),
                  self._view(node.ssum, s._ssum, self.ss)] + [0] * A
            for a, c in enumerate(node.children):
                kid = self._emit(c)
                w[at + 7 + a] = kid
            return at
        if isinstance(node, Terminal):
            B = len(node.boards)
            if node.kind == "fold":
                w += [FOLD, B] + [_f(_scalar(node.fold_pay(p))) for p in (0, 1)]
                return at
            if node.kind not in ("showdown", "leaf", "bfold", "bleaf", "rfold", "rleaf"):
                raise Unsupported(node.kind)
            if node.kind[0] in "br":
                # a bucket game's: its tables (the entry's, shared by every node on the same boards)
                T, _, E, kap = s._btab(node) if node.kind[0] == "b" else s._rtab(node)
                head = [B, E.shape[-1], T]
                if node.kind[1:] == "fold":
                    w += [BFOLD] + head + [self._upload(kap, np.float32, True)] + \
                        [_f(_scalar(node.fold_pay(p))) for p in (0, 1)]
                    return at
                w += [BLEAF] + head + [self._upload(E, np.float32, True), self._upload(kap, np.float32, True)] + \
                    self._coef(node)
                return at
            coef = self._coef(node)
            if len(node.boards[0]) == 5:
                ref = self._ranking(s.ctx.ranking(node.boards))
                at = len(w)      # (after the ranking's record, when it was added)
                w += [SHOWDOWN, B, ref] + coef
            else:
                self.leaf_rows += B
                w += [LEAF, B, self._upload(s._terminal_M(node), np.float32)] + coef
            return at
        if isinstance(node, RiverEntry):
            T, X, _, _ = s._rtab(node)              # X (2, T, W, K)
            w += [RIVER, len(node.boards), X.shape[2], X.shape[3], T, 0, self._upload(X, np.float32),
                  self._upload(X, np.float32, True)]
            w[at + 5] = self._emit(node.child)
            return at
        if isinstance(node, BucketEntry):
            T, ids, _, _ = s._btab(node)            # ids (T, 2, n)
            w += [BUCKETS, len(node.boards), s.ctx.cfg.river_buckets, T, 0, self._upload(ids, np.int32)]
            w[at + 4] = self._emit(node.child)
            return at
        if isinstance(node, Chance):
            if node.sample is not None:
                raise Unsupported("sampled cards")
            B = len(node.boards)
            if node.perm is not None:
                G, flat, starts, sizes = node.perm
                w += [CLASSES, B, node.K, 0, self._upload(node.masks, np.float32), self._upload(node.norm, np.float32),
                      self._upload(G, np.int32), self._upload(flat, np.int32), self._upload(starts, np.int32),
                      self._upload(sizes, np.float32), len(G)]
            else:
                wt = node.masks if node.umasks is None else node.umasks
                w += [CHANCE, B, node.K, 0, self._upload(node.masks, np.float32), self._upload(wt, np.float32),
                      self._upload(node.norm, np.float32)]
            kid = self._emit(node.child)
            w[at + 3] = kid
            return at
        raise Unsupported(type(node).__name__)

    @staticmethod
    def _coef(node) -> list:
        """A showdown's or equity leaf's (a_p, c_p) for each player p: u_p = a_p * (its products with the opponent's
        reach) + c_p * (their compatible mass)."""
        coef = []
        for p in (0, 1):
            if node.pay is not None:
                coef += [_f(_scalar(node.pay[2 * p + 1])), _f(_scalar(node.pay[2 * p]))]
            elif node.rake is None:
                coef += [_f(_scalar(node.inv[p])), 0]
            else:
                half = _scalar(node.rake) / 2
                coef += [_f(_scalar(node.inv[p]) - half), _f(-half)]
        return coef

    def _ranking(self, R) -> int:
        """A Ranking's record (once per ranking)."""
        hit = self._rank.get(id(R))
        if hit is not None:
            return hit
        w = self.w
        at = len(w)
        up = self._upload
        w += [R.order.shape[0], self.n, up(R.order, np.int32), up(R.lo, np.int32), up(R.hi, np.int32),
              up(R.pcombo, np.int32), up(R.plo, np.int32), up(R.phi, np.int32), up(R.cstart, np.int32),
              up(R.cend, np.int32), up(R.live, np.float32)]
        self._rank[id(R)] = at
        return at

    def iterate(self, k: int, factors):
        f = np.array(factors, dtype=np.float32).reshape(-1)
        self.L.write(self.fac, f)
        if self.L.run(self.prog, k, self.fac) != k:
            raise RuntimeError("core run failed")

    def upload(self, solver, regrets: bool = True):
        """The solver's strategy sums (and regrets) into the arena (numpy may have run since)."""
        if regrets:
            self.L.write(self.reg, solver._regret)
        self.L.write(self.ss, solver._ssum)

    def download(self, solver):
        """The regrets and strategy sums back into the solver's buffers."""
        self.L.read(self.reg, solver._regret)
        self.L.read(self.ss, solver._ssum)


class Tables:
    """A flop solve's tables in the core (Ctx.tables while a Program is being built, the arena empty: each table's
    inputs, scratch and outputs in it, then released): 7-10x numpy's natively. _avg_showdown's are numpy's floats
    (exact counts); the bucket and river tables put every combo in numpy's buckets (their equities to 2^-20,
    solver.eq_key, ties in combo order) and sum in another order, E to float32 rounding (tests/test_core.py)."""

    def __init__(self, L):
        self.L = L

    def _put(self, a, dtype) -> int:
        a = np.ascontiguousarray(a, dtype=dtype)
        off = self.L.alloc(max(a.nbytes, 4))
        self.L.write(off, a)
        return off

    def _get(self, off, shape, dtype) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        self.L.read(off, out)
        return out

    def _run(self, words):
        args = self._put(np.array(words, dtype=np.int64).astype(np.int32), np.int32)
        if self.L.table(args) != 0:
            raise RuntimeError("core table failed")

    def _start(self, ctx, S):
        self.L.reset()
        S = np.asarray(S)
        if S.size and (S.max() > 32767 or S.min() < -1):
            raise Unsupported("strengths over int16")
        return [ctx.n, len(S), self._put(S, np.int32), self._put(ctx.ca, np.int32), self._put(ctx.cb, np.int32)]

    def avg_showdown(self, ctx, S) -> np.ndarray:
        """Ctx._avg_showdown on boards of strengths S (R, n)."""
        head = self._start(ctx, S)
        out = self.L.alloc(4 * ctx.n * ctx.n)
        self._run([1] + head + [out])
        return self._get(out, (ctx.n, ctx.n), np.float32)

    def bucket_table(self, ctx, S, w, K: int) -> tuple:
        """Ctx.bucket_table's (ids, E, kappa) on the rivers of strengths S, w: each player's root reach on the combos
        holding no turn card."""
        n = ctx.n
        head = self._start(ctx, S)
        ins = [self._put(w[q], np.float64) for q in (0, 1)]
        ids, E, kap = self.L.alloc(8 * n), self.L.alloc(8 * K * K), self.L.alloc(8 * K * K)
        self._run([2] + head + [K] + ins + [ids, E, kap])
        return (self._get(ids, (2, n), np.int32).astype(np.intp), self._get(E, (2, K, K), np.float32),
                self._get(kap, (2, K, K), np.float32))

    def river_table(self, ctx, S, r, K: int, tids, outer) -> tuple:
        """Ctx.river_table's (X, E, kappa) on the rivers of strengths S, r: each player's root reach on the combos
        holding no turn card, tids: the turn buckets (outer of them) or None."""
        n = ctx.n
        head = self._start(ctx, S)
        ins = [self._put(r[q], np.float64) for q in (0, 1)]
        t = self._put(tids, np.int32) if outer else 0
        W = outer or n
        X, E, kap = self.L.alloc(8 * W * K), self.L.alloc(8 * K * K), self.L.alloc(8 * K * K)
        self._run([3] + head + [K] + ins + [outer or 0, t, X, E, kap])
        return (self._get(X, (2, W, K), np.float32), self._get(E, (2, K, K), np.float32),
                self._get(kap, (2, K, K), np.float32))


def _program(L, solver, r0, r1, regrets: bool = True):
    """The solver's program with its regrets (unless not `regrets`), sums and root reaches r0, r1 uploaded: the one
    in the arena if it's this solver's, else built. None when the tree has something the core doesn't run, or more
    than the arena takes (the browser's memory may refuse to grow: the core traps): remembered, as a tree doesn't
    change, and numpy runs it."""
    global _prog
    if not solver._core_ok:
        return None
    try:
        if _prog is not None and _prog.valid(L, solver):
            _prog.upload(solver, regrets)
            _prog.reaches(r0, r1)
            return _prog
        _prog = None
        _prog = Program(L, solver, r0, r1)
    except Exception:
        _prog = None
        solver._core_ok = False
    return _prog


def evaluate(solver, jobs):
    """Each job (node, p, r0, r1, md): p's utilities (1, n) at `node` (of one board) for reaches r0, r1 (n,) there: md
    AVG with both players on their average strategies (Solver._cfr with use_avg), BR with p best-responding to the
    opponent's (Solver._best_response). None when the core can't (numpy runs them)."""
    L = lib()
    if L is None or not solver.use_core or solver.predictive or any(len(j[0].boards) != 1 for j in jobs):
        return None
    if not _lock.acquire(blocking=False):
        return None
    try:
        prog = _program(L, solver, jobs[0][2], jobs[0][3], regrets=False)
        if prog is None:
            return None
        n, out = prog.n, []
        for node, p, r0, r1, md in jobs:
            at = prog.at.get(id(node))
            if at is None:
                return None
            for off, r in ((prog.ev, r0), (prog.ev + 4 * n, r1)):
                r = np.ascontiguousarray(np.ravel(r), dtype=np.float32)
                if r.size != n:
                    return None
                L.write(off, r)
            if L.eval(prog.prog, at, p, prog.ev, prog.ev + 4 * n, prog.ev + 8 * n, md) != 0:
                return None
            u = np.empty((1, n), dtype=np.float32)
            L.read(prog.ev + 8 * n, u)
            out.append(u)
        return out
    finally:
        _lock.release()


def run(solver, r0, r1, t0: float, time_budget: float, max_iters: int, min_iters: int, should_stop, reserve) -> bool:
    """Solver.run's iterations in the core, with its budget rules (`reserve` included). False, with nothing run, when
    there's no core, it's busy, or the tree has something it doesn't run."""
    from .solver import Cancelled
    global _lib
    L = lib()
    if L is None or solver.predictive or not _lock.acquire(blocking=False):
        return False
    try:
        start = solver.iters
        prog = _program(L, solver, r0, r1)
        if prog is None:
            return False
        t1 = time.time()
        per = None
        try:
            while solver.iters < max_iters:
                if should_stop is not None and should_stop():
                    prog.download(solver)
                    raise Cancelled()
                done = solver.iters - start
                el = time.time() - t0
                if done >= min_iters and per is not None:
                    if reserve is not None and el + per * (1.0 + reserve) > time_budget:
                        break
                    if reserve is None and el > time_budget:
                        break
                k = 1 if per is None else min(max_iters - solver.iters, MAX_CHUNK, max(1, int(CHUNK_S / per)))
                if per is not None:
                    left = time_budget - el - (per * reserve if reserve is not None else 0.0)
                    k = min(k, max(1, min_iters - done, int(left / per)))
                factors = []
                for _ in range(k):
                    solver.iters += 1
                    factors.append(solver._factors())
                prog.iterate(k, factors)
                per = (time.time() - t1) / (solver.iters - start)
        except Cancelled:
            raise
        except Exception:
            # (the core failed mid-run: its iterations are dropped and numpy runs the solve)
            solver.iters = start
            _lib = False
            return False
        prog.download(solver)
        return True
    finally:
        _lock.release()
