/* The compiled CFR core: whole training iterations of gto/solver.py's Solver.run (gto/core.py builds the program and
 * runs it). The same source builds natively (a shared library, loaded with ctypes) and for the browser (gto/core.wasm,
 * instantiated by gto/static/worker.js beside Pyodide, and gto/core-mt.wasm, the same with threads), with no libc:
 * everything lives in one arena, the inputs copied in by gto/core.py and the traversal's scratch above them (allocated
 * and released in stack order).
 *
 * A program is int32 words: a header, then one record per tree node (a record's children are word indices), floats
 * stored as their bits, arrays as byte offsets into the arena. The traversal is Solver._cfr's with update=True: the
 * same arithmetic in float32, in the same order where it matters (sums over actions and cards run first to last), so
 * a solve's strategies agree with the Python solver's to float32 rounding.
 *
 *   header  [MAGIC, n, ca, cb, root, r0, r1, regret, ssum, flat_len]
 *   DECISION [1, player, A, B, W, regret, ssum, child_0 .. child_A-1]   regret, ssum: (B, A, W) blocks
 *   FOLD     [2, B, f0, f1]                     u_p = f_p * compat(ro)
 *   SHOWDOWN [3, B, ranking, a0, c0, a1, c1]    u_p = a_p * (sum of ro over weaker - stronger) + c_p * compat(ro)
 *   LEAF     [4, B, M, a0, c0, a1, c1]          the same with M (B, n, n) @ ro (an all-in's equity over the runouts)
 *   CHANCE   [5, B, K, child, masks, w, norm]   child rows b*K+k; masks, w (B*K, n); norm (B, n)
 *   CLASSES  [6, B, K, child, masks, norm, G, flat, starts, sizes, nG]   river classes (Chance.spread, Chance.gather)
 *   RANKING  [B, n, order, lo, hi, pcombo, plo, phi, cstart, cend, live]  (Ranking: (B, n) / (B, 2n) / (B, 2, n))
 * A flop solve's turn and river by hand buckets (Config.river_buckets, river_strength): below an entry the rows are K
 * buckets wide, row b's tables those of board b % T (T, the period of the entry's boards).
 *   BUCKETS  [7, B, K, T, child, ids]           BucketEntry: ids (T, 2, n), each combo's bucket by player (K: none)
 *   RIVER    [8, B, W, K, T, child, X, Xt]      RiverEntry: X (2, T, W, K), Xt its (2, T, K, W) transpose: rows W
 *                                              wide in, K wide below
 *   BFOLD    [9, B, K, T, kapt, f0, f1]         u_p = f_p * kap[p] @ ro
 *   BLEAF    [10, B, K, T, Et, kapt, a0, c0, a1, c1]   u_p = a_p * E[p] @ ro + c_p * kap[p] @ ro
 *   (kapt, Et: kap and E (2, T, K, K) transposed, each (K, K) matrix's columns contiguous)
 *
 * Every node runs on a range of its rows, lo..hi (a node's rows are its boards; a chance node's child has K rows per
 * row of its own): the reaches and utilities passed between nodes hold just those rows, the program's arrays (regrets,
 * masks, rankings) are indexed by the row itself. Rows never mix below a chance node, so with threads (core_threads)
 * the first chance node on the way down hands its child's rows out in chunks, to this thread and its helpers, each
 * with its own scratch: every row gets the same arithmetic whichever thread runs it, and a solve's numbers don't
 * depend on how many threads ran it.
 */
#include <stddef.h>
#include <stdint.h>

typedef float f32;
typedef int32_t i32;
typedef uint32_t u32;
typedef uint64_t u64;

#ifdef __wasm__
#define EXPORT(name) __attribute__((export_name(#name)))
extern unsigned char __heap_base;
#ifdef __wasm_atomics__
#define THREADS 1
#endif
#else
#define EXPORT(name) __attribute__((visibility("default")))
#include <sys/mman.h>
#include <pthread.h>
#include <sched.h>
#define THREADS 1
#endif

#define MAGIC 0x67746f31   /* "gto1" */
enum { DECISION = 1, FOLD = 2, SHOWDOWN = 3, LEAF = 4, CHANCE = 5, CLASSES = 6, BUCKETS = 7, RIVER = 8, BFOLD = 9,
       BLEAF = 10 };

/* ------------------------------------------------------------------ arena */
static unsigned char *base;

#ifdef __wasm__
static void ensure(size_t end) {
  if (!base) base = (unsigned char *)(((uintptr_t)&__heap_base + 63) & ~(uintptr_t)63);
  size_t need = (uintptr_t)base + end, have = __builtin_wasm_memory_size(0) * 65536;
  if (need > have && __builtin_wasm_memory_grow(0, (need - have + 65535) / 65536) == (size_t)-1) __builtin_trap();
}
#else
#define RESERVE ((size_t)1 << 35)   /* 32 GB of address space, committed as it's touched */
static size_t touched;
static void ensure(size_t end) {
  if (!base) {
    void *p = mmap(0, RESERVE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) __builtin_trap();
    base = p;
  }
  if (end > RESERVE) __builtin_trap();
  if (end > touched) touched = end;
}
#endif

/* A thread's scratch: allocated up to top (a helper's within its lane, at most lim); par: inside a job's rows (no
   further split); peak: the most it has held (the lanes' size is measured with it). */
typedef struct {
  size_t top, lim, peak;
  int par;
} Lane;

static Lane M;   /* the calling thread's: the arena itself */

static void *salloc(Lane *L, size_t bytes) {
  size_t at = (L->top + 63) & ~(size_t)63;
  if (L->lim) {
    if (at + bytes > L->lim) __builtin_trap();
  } else {
    ensure(at + bytes);
  }
  L->top = at + bytes;
  if (L->top > L->peak) L->peak = L->top;
  return base + at;
}

static size_t lane_size;   /* bytes a helper's chunk of rows may take: 0 until a serial iteration measured them */
static size_t measured;

/* Frees the arena (natively its pages go back to the system). */
EXPORT(core_reset) void core_reset(void) {
#ifndef __wasm__
  if (base && touched > ((size_t)64 << 20)) {
    madvise(base, touched, MADV_FREE);
    touched = 0;
  }
#endif
  M.top = M.peak = 0;
  lane_size = measured = 0;
}

/* `bytes` in the arena; returns its offset. */
EXPORT(core_alloc) u32 core_alloc(u32 bytes) {
  unsigned char *p = salloc(&M, bytes);
  return (u32)(p - base);
}

/* The arena's address (offsets are from it). */
EXPORT(core_base) uintptr_t core_base(void) {
  ensure(0);
  return (uintptr_t)base;
}

EXPORT(core_version) i32 core_version(void) { return MAGIC; }

/* The source's sha1 (scripts/build_core.py builds the module with it, so a test can tell a stale one) */
#ifndef CORE_SOURCE
#define CORE_SOURCE "dev"
#endif
static const char source[] = CORE_SOURCE;
EXPORT(core_source) const char *core_source(void) { return source; }

/* ------------------------------------------------------------------ program */
static const i32 *P;
static i32 n;
static const i32 *ca, *cb;

#define PF(off) ((f32 *)(base + (u32)(off)))
#define PI(off) ((i32 *)(base + (u32)(off)))
static inline f32 fbits(i32 w) {
  union { i32 i; f32 f; } u;
  u.i = w;
  return u.f;
}

typedef float v4 __attribute__((vector_size(16)));
typedef float v4u __attribute__((vector_size(16), aligned(4)));

/* the sum of x[0..m) in 16 lanes (a sum in one register waits on each add before it) */
static f32 vsum(const f32 *x, i32 m) {
  i32 m16 = m & ~15, j = 0;
  v4 s0 = {0, 0, 0, 0}, s1 = s0, s2 = s0, s3 = s0;
  for (; j < m16; j += 16) {
    s0 += *(const v4u *)(x + j);
    s1 += *(const v4u *)(x + j + 4);
    s2 += *(const v4u *)(x + j + 8);
    s3 += *(const v4u *)(x + j + 12);
  }
  v4 s = (s0 + s1) + (s2 + s3);
  f32 t = (s[0] + s[1]) + (s[2] + s[3]);
  for (; j < m; j++) t += x[j];
  return t;
}

/* ro: one row of the opponent's reach -> out: the mass of opponent combos sharing no card with each combo
   (Solver.compat_sum: the total, minus each of the combo's cards' combos, plus the combo itself); card: 52 floats */
static void compat(const f32 *ro, f32 *card, f32 *out) {
  for (int c = 0; c < 52; c++) card[c] = 0;
  f32 tot = vsum(ro, n), run = 0;
  /* the combos come sorted by their first card (gto/cards.py ALL_COMBOS), so its sums run in a register (adding
     through memory, each add waited on the one before) */
  i32 cur = n ? ca[0] : 0;
  for (i32 j = 0; j < n; j++) {
    if (ca[j] != cur) {
      card[cur] += run;
      cur = ca[j];
      run = 0;
    }
    run += ro[j];
    card[cb[j]] += ro[j];
  }
  card[cur] += run;
  for (i32 i = 0; i < n; i++) out[i] = tot - card[ca[i]] - card[cb[i]] + ro[i];
}

/* out[0] = 0, out[j + 1] = out[j] + ro[idx[j]] for j < m: four running sums over the quarters at once (one waits on
   each add before it), then each quarter's offset */
static void prefix(const f32 *ro, const i32 *idx, i32 m, f32 *out) {
  i32 q = m / 4, s1 = q, s2 = 2 * q, s3 = 3 * q;
  f32 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  out[0] = 0;
  for (i32 j = 0; j < q; j++) {
    a0 += ro[idx[j]];
    a1 += ro[idx[s1 + j]];
    a2 += ro[idx[s2 + j]];
    a3 += ro[idx[s3 + j]];
    out[j + 1] = a0;
    out[s1 + j + 1] = a1;
    out[s2 + j + 1] = a2;
    out[s3 + j + 1] = a3;
  }
  for (i32 j = s3 + q; j < m; j++) out[j + 1] = a3 += ro[idx[j]];
  f32 o1 = a0, o2 = o1 + a1, o3 = o2 + a2;
  for (i32 j = s1 + 1; j <= s2; j++) out[j] += o1;
  for (i32 j = s2 + 1; j <= s3; j++) out[j] += o2;
  for (i32 j = s3 + 1; j <= m; j++) out[j] += o3;
}

/* Ranking.utility for row b: sum over compatible opponent combos of reach * (+1 weaker, -1 stronger) */
static void ranking(Lane *L, const i32 *R, i32 b, const f32 *ro, f32 *u) {
  const i32 *order = PI(R[2]) + (size_t)b * n, *lo = PI(R[3]) + (size_t)b * n, *hi = PI(R[4]) + (size_t)b * n;
  const i32 *pcombo = PI(R[5]) + (size_t)b * 2 * n;
  const i32 *plo = PI(R[6]) + (size_t)b * 2 * n, *phi = PI(R[7]) + (size_t)b * 2 * n;
  const i32 *cstart = PI(R[8]), *cend = PI(R[9]);
  const f32 *live = PF(R[10]) + (size_t)b * n;
  size_t mark = L->top;
  f32 *cum = salloc(L, sizeof(f32) * (n + 1)), *cs = salloc(L, sizeof(f32) * (2 * n + 1));
  f32 *k = salloc(L, sizeof(f32) * 52);   /* each card's term: its pairs' sum before them and to their end */
  prefix(ro, order, n, cum);
  prefix(ro, pcombo, 2 * n, cs);
  for (i32 c = 0; c < 52; c++) k[c] = cs[cend[c]] + cs[cstart[c]];
  f32 total = cum[n];
  for (i32 i = 0; i < n; i++) {
    f32 x = cum[lo[i]];
    x += cum[hi[i]];
    x += k[ca[i]];
    x += k[cb[i]];
    x -= cs[plo[i]];
    x -= cs[phi[i]];
    x -= cs[plo[n + i]];
    x -= cs[phi[n + i]];
    x -= total;
    u[i] = x * live[i];
  }
  L->top = mark;
}

/* u = M @ ro for one (n, n) matrix: four 4-wide accumulators a row */
static void matvec(const f32 *M, const f32 *ro, f32 *u) {
  i32 n16 = n & ~15;
  for (i32 i = 0; i < n; i++) {
    const f32 *m = M + (size_t)i * n;
    v4 s0 = {0, 0, 0, 0}, s1 = s0, s2 = s0, s3 = s0;
    i32 j = 0;
    for (; j < n16; j += 16) {
      s0 += *(const v4u *)(m + j) * *(const v4u *)(ro + j);
      s1 += *(const v4u *)(m + j + 4) * *(const v4u *)(ro + j + 4);
      s2 += *(const v4u *)(m + j + 8) * *(const v4u *)(ro + j + 8);
      s3 += *(const v4u *)(m + j + 12) * *(const v4u *)(ro + j + 12);
    }
    v4 s = (s0 + s1) + (s2 + s3);
    f32 x = (s[0] + s[1]) + (s[2] + s[3]);
    for (; j < n; j++) x += m[j] * ro[j];
    u[i] = x;
  }
}

static void bterminal(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *ro, f32 *u);

/* p's utility (rows lo..hi, n) at a terminal; ro: the opponent's reach on those rows */
static void terminal(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *ro, f32 *u) {
  if (nd[0] == BFOLD || nd[0] == BLEAF) {
    bterminal(L, nd, p, lo, hi, ro, u);
    return;
  }
  size_t mark = L->top;
  f32 *cs = salloc(L, sizeof(f32) * n), *card = salloc(L, sizeof(f32) * 52);
  if (nd[0] == FOLD) {
    f32 f = fbits(nd[2 + p]);
    for (i32 b = lo; b < hi; b++) {
      f32 *ub = u + (size_t)(b - lo) * n;
      compat(ro + (size_t)(b - lo) * n, card, cs);
      for (i32 i = 0; i < n; i++) ub[i] = f * cs[i];
    }
  } else {
    f32 a = fbits(nd[3 + 2 * p]), c = fbits(nd[4 + 2 * p]);
    for (i32 b = lo; b < hi; b++) {
      const f32 *r = ro + (size_t)(b - lo) * n;
      f32 *ub = u + (size_t)(b - lo) * n;
      if (nd[0] == SHOWDOWN)
        ranking(L, P + nd[2], b, r, ub);
      else
        matvec(PF(nd[2]) + (size_t)b * n * n, r, ub);
      if (c != 0) {
        compat(r, card, cs);
        for (i32 i = 0; i < n; i++) ub[i] = a * ub[i] + c * cs[i];
      } else {
        for (i32 i = 0; i < n; i++) ub[i] = a * ub[i];
      }
    }
  }
  L->top = mark;
}

/* Regret matching (Decision.strategy): rows lo..hi of the (B, A, W) regrets -> their strategies; with `avg`, the
   strategy sums -> the average strategy (Decision.avg_strategy). (Every loop runs over a row's combos innermost, so it
   vectorizes; the sums over actions still add them first to last.) */
static void strategy(Lane *L, const f32 *regret, i32 lo, i32 hi, i32 A, i32 W, int avg, f32 *s) {
  f32 uni = (f32)(1.0 / A);
  size_t mark = L->top;
  f32 *tot = salloc(L, sizeof(f32) * W);
  for (i32 b = lo; b < hi; b++) {
    const f32 *r = regret + (size_t)b * A * W;
    f32 *sb = s + (size_t)(b - lo) * A * W;
    if (avg)
      for (i32 x = 0; x < A * W; x++) sb[x] = r[x];
    else
      for (i32 x = 0; x < A * W; x++) sb[x] = r[x] > 0 ? r[x] : 0;
    for (i32 i = 0; i < W; i++) tot[i] = sb[i];
    for (i32 a = 1; a < A; a++)
      for (i32 i = 0; i < W; i++) tot[i] += sb[a * W + i];
    for (i32 i = 0; i < W; i++) tot[i] = tot[i] == 0 ? 0 : 1 / tot[i];   /* (one division a combo) */
    for (i32 a = 0; a < A; a++)
      for (i32 i = 0; i < W; i++) sb[a * W + i] = tot[i] == 0 ? uni : sb[a * W + i] * tot[i];
  }
  L->top = mark;
}

static void cfr(Lane *L, i32 at, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out);
static void split(Lane *L, i32 child, int p, i32 lo, i32 hi, const f32 *c0, const f32 *c1, f32 *u);

/* TRAIN: a training pass (Solver._cfr, update=True); AVG: both players on their average strategies, no update (use_avg);
   BR: p best-responds to the opponent's average strategy (Solver._best_response) */
enum { TRAIN = 0, AVG = 1, BR = 2 };
static int mode = TRAIN;

static int is_terminal(i32 at) {
  i32 t = P[at];
  return t == FOLD || t == SHOWDOWN || t == LEAF || t == BFOLD || t == BLEAF;
}

static void decision(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out) {
  i32 player = nd[1], A = nd[2], W = nd[4];
  f32 *regret = PF(nd[5]), *ssum = PF(nd[6]);
  const i32 *kids = nd + 7;
  size_t RW = (size_t)(hi - lo) * W, mark = L->top;
  f32 *s = salloc(L, sizeof(f32) * RW * A);
  if (!(mode == BR && player == p)) strategy(L, mode == TRAIN ? regret : ssum, lo, hi, A, W, mode != TRAIN, s);
  if (player == p) {
    const f32 *ro = p == 0 ? r1 : r0, *rp = p == 0 ? r0 : r1;
    f32 *ut = salloc(L, sizeof(f32) * RW * A);   /* (A, rows, W): each action's utilities contiguous */
    for (i32 a = 0; a < A; a++) {
      if (is_terminal(kids[a]))
        terminal(L, P + kids[a], p, lo, hi, ro, ut + a * RW);
      else
        cfr(L, kids[a], p, lo, hi, r0, r1, ut + a * RW);
    }
    if (mode == BR) {
      for (size_t x = 0; x < RW; x++) out[x] = ut[x];
      for (i32 a = 1; a < A; a++)
        for (size_t x = 0; x < RW; x++) out[x] = ut[a * RW + x] > out[x] ? ut[a * RW + x] : out[x];
      L->top = mark;
      return;
    }
    for (i32 b = lo; b < hi; b++) {
      size_t j = (size_t)(b - lo);
      f32 *nu0 = out + j * W;
      for (i32 a = 0; a < A; a++) {
        const f32 *sa = s + (j * A + a) * W, *ua = ut + a * RW + j * W;
        if (a == 0)
          for (i32 i = 0; i < W; i++) nu0[i] = sa[i] * ua[i];
        else
          for (i32 i = 0; i < W; i++) nu0[i] += sa[i] * ua[i];
      }
      if (mode != TRAIN) continue;
      for (i32 a = 0; a < A; a++) {
        f32 *rg = regret + ((size_t)b * A + a) * W, *ss = ssum + ((size_t)b * A + a) * W;
        const f32 *sa = s + (j * A + a) * W, *ua = ut + a * RW + j * W, *nu = out + j * W;
        const f32 *rb = rp + j * W;
        for (i32 i = 0; i < W; i++) {
          rg[i] += ua[i] - nu[i];
          ss[i] += rb[i] * sa[i];
        }
      }
    }
  } else {
    /* the opponent acts: each action's reach, and the utilities summed over actions in order */
    const f32 *ro = p == 0 ? r1 : r0;
    f32 *reach = salloc(L, sizeof(f32) * RW), *u = salloc(L, sizeof(f32) * RW);
    for (i32 a = 0; a < A; a++) {
      for (i32 b = lo; b < hi; b++) {
        size_t j = (size_t)(b - lo);
        const f32 *sa = s + (j * A + a) * W, *rb = ro + j * W;
        f32 *x = reach + j * W;
        for (i32 i = 0; i < W; i++) x[i] = rb[i] * sa[i];
      }
      f32 *dst = a == 0 ? out : u;
      if (is_terminal(kids[a]))
        terminal(L, P + kids[a], p, lo, hi, reach, dst);
      else if (p == 0)
        cfr(L, kids[a], p, lo, hi, r0, reach, dst);
      else
        cfr(L, kids[a], p, lo, hi, reach, r1, dst);
      if (a > 0)
        for (size_t x = 0; x < RW; x++) out[x] += u[x];
    }
  }
  L->top = mark;
}

static void chance(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out) {
  i32 K = nd[2], child = nd[3];
  const f32 *masks = PF(nd[4]), *w = PF(nd[5]), *norm = PF(nd[6]);
  size_t mark = L->top, rows = (size_t)(hi - lo) * K;
  f32 *c0 = salloc(L, sizeof(f32) * rows * n), *c1 = salloc(L, sizeof(f32) * rows * n);
  f32 *u = salloc(L, sizeof(f32) * rows * n);
  for (i32 b = lo; b < hi; b++)
    for (i32 k = 0; k < K; k++) {
      size_t row = ((size_t)(b - lo) * K + k) * n, at = ((size_t)b * K + k) * n, rb = (size_t)(b - lo) * n;
      for (i32 i = 0; i < n; i++) {
        c0[row + i] = r0[rb + i] * masks[at + i];
        c1[row + i] = r1[rb + i] * masks[at + i];
      }
    }
  split(L, child, p, lo * K, hi * K, c0, c1, u);
  for (i32 b = lo; b < hi; b++) {
    f32 *o = out + (size_t)(b - lo) * n;
    const f32 *ub = u + (size_t)(b - lo) * K * n, *wb = w + (size_t)b * K * n;
    for (i32 i = 0; i < n; i++) o[i] = ub[i] * wb[i];
    for (i32 k = 1; k < K; k++)
      for (i32 i = 0; i < n; i++) o[i] += ub[(size_t)k * n + i] * wb[(size_t)k * n + i];
    for (i32 i = 0; i < n; i++) o[i] = o[i] / norm[(size_t)b * n + i];
  }
  L->top = mark;
}

/* Chance.spread of one player's reach (rows lo..hi) into the class rows c; rp: n + 1 floats */
static void spread(const i32 *nd, i32 lo, i32 hi, const f32 *r, f32 *c, f32 *rp) {
  i32 K = nd[2], nG = nd[10];
  const f32 *masks = PF(nd[4]), *sizes = PF(nd[9]);
  const i32 *G = PI(nd[6]), *starts = PI(nd[8]);
  for (i32 b = lo; b < hi; b++) {
    for (i32 i = 0; i < n; i++) rp[i] = r[(size_t)(b - lo) * n + i];
    rp[n] = 0;
    for (i32 k = 0; k < K; k++) {
      f32 *dst = c + ((size_t)(b - lo) * K + k) * n;
      i32 e0 = starts[k], e1 = k + 1 < K ? starts[k + 1] : nG;
      for (i32 i = 0; i < n; i++) dst[i] = rp[G[(size_t)e0 * n + i]];
      for (i32 e = e0 + 1; e < e1; e++)
        for (i32 i = 0; i < n; i++) dst[i] += rp[G[(size_t)e * n + i]];
      const f32 *m = masks + ((size_t)b * K + k) * n;
      for (i32 i = 0; i < n; i++) dst[i] *= m[i] / sizes[k];
    }
  }
}

/* river classes: Chance.spread into the class subtrees, Chance.gather back */
static void classes(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out) {
  i32 K = nd[2], child = nd[3], nG = nd[10];
  const f32 *masks = PF(nd[4]), *norm = PF(nd[5]);
  const i32 *flat = PI(nd[7]);
  size_t mark = L->top, rows = (size_t)(hi - lo) * K;
  f32 *c0 = salloc(L, sizeof(f32) * rows * n), *c1 = salloc(L, sizeof(f32) * rows * n);
  f32 *u = salloc(L, sizeof(f32) * rows * n), *rp = salloc(L, sizeof(f32) * (n + 1));
  f32 *um = salloc(L, sizeof(f32) * K * n);
  spread(nd, lo, hi, r0, c0, rp);
  spread(nd, lo, hi, r1, c1, rp);
  split(L, child, p, lo * K, hi * K, c0, c1, u);
  for (i32 b = lo; b < hi; b++) {
    const f32 *ub = u + (size_t)(b - lo) * K * n, *m = masks + (size_t)b * K * n;
    for (size_t x = 0; x < (size_t)K * n; x++) um[x] = ub[x] * m[x];
    f32 *o = out + (size_t)(b - lo) * n;
    for (i32 i = 0; i < n; i++) o[i] = um[flat[i]];
    for (i32 e = 1; e < nG; e++)
      for (i32 i = 0; i < n; i++) o[i] += um[flat[(size_t)e * n + i]];
    for (i32 i = 0; i < n; i++) o[i] = o[i] / norm[(size_t)b * n + i];
  }
  L->top = mark;
}

/* ------------------------------------------------------------------ hand buckets (a flop solve's turn and river) */
/* u = M @ x for an (R, C) matrix given as its transpose Mt (C, R): each of u's entries summed over x first to last
   (numpy's matmul without BLAS), 32 of them at a time in registers, adding a row of Mt a step. A zero in x adds
   nothing (+-0 to every sum), so its row is skipped: a bucket a strategy never takes there has no reach. */
static void tmatvec(const f32 *Mt, const f32 *x, f32 *u, i32 R, i32 C) {
  i32 r = 0;
  for (; r + 32 <= R; r += 32) {
    v4 a0 = {0, 0, 0, 0}, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (i32 j = 0; j < C; j++) {
      f32 xj = x[j];
      if (xj == 0) continue;
      v4 s = {xj, xj, xj, xj};
      const f32 *m = Mt + (size_t)j * R + r;
      a0 += *(const v4u *)(m) * s;
      a1 += *(const v4u *)(m + 4) * s;
      a2 += *(const v4u *)(m + 8) * s;
      a3 += *(const v4u *)(m + 12) * s;
      a4 += *(const v4u *)(m + 16) * s;
      a5 += *(const v4u *)(m + 20) * s;
      a6 += *(const v4u *)(m + 24) * s;
      a7 += *(const v4u *)(m + 28) * s;
    }
    *(v4u *)(u + r) = a0;
    *(v4u *)(u + r + 4) = a1;
    *(v4u *)(u + r + 8) = a2;
    *(v4u *)(u + r + 12) = a3;
    *(v4u *)(u + r + 16) = a4;
    *(v4u *)(u + r + 20) = a5;
    *(v4u *)(u + r + 24) = a6;
    *(v4u *)(u + r + 28) = a7;
  }
  for (; r + 4 <= R; r += 4) {
    v4 a = {0, 0, 0, 0};
    for (i32 j = 0; j < C; j++) {
      f32 xj = x[j];
      v4 s = {xj, xj, xj, xj};
      a += *(const v4u *)(Mt + (size_t)j * R + r) * s;
    }
    *(v4u *)(u + r) = a;
  }
  for (; r < R; r++) {
    f32 y = 0;
    for (i32 j = 0; j < C; j++) y += Mt[(size_t)j * R + r] * x[j];
    u[r] = y;
  }
}

/* p's bucket utilities (rows lo..hi, K) at a bucket game's terminal (Solver._bucket_terminal); ro: the opponent's
   masses on those rows */
static void bterminal(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *ro, f32 *u) {
  i32 K = nd[2], T = nd[3];
  size_t KK = (size_t)K * K, mark = L->top;
  if (nd[0] == BFOLD) {
    const f32 *kap = PF(nd[4]) + (size_t)p * T * KK;
    f32 f = fbits(nd[5 + p]);
    for (i32 b = lo; b < hi; b++) {
      f32 *ub = u + (size_t)(b - lo) * K;
      tmatvec(kap + (size_t)(b % T) * KK, ro + (size_t)(b - lo) * K, ub, K, K);
      for (i32 i = 0; i < K; i++) ub[i] = f * ub[i];
    }
    return;
  }
  const f32 *E = PF(nd[4]) + (size_t)p * T * KK, *kap = PF(nd[5]) + (size_t)p * T * KK;
  f32 a = fbits(nd[6 + 2 * p]), c = fbits(nd[7 + 2 * p]);
  f32 *cs = salloc(L, sizeof(f32) * K);
  for (i32 b = lo; b < hi; b++) {
    const f32 *rb = ro + (size_t)(b - lo) * K;
    f32 *ub = u + (size_t)(b - lo) * K;
    tmatvec(E + (size_t)(b % T) * KK, rb, ub, K, K);
    if (c != 0) {
      tmatvec(kap + (size_t)(b % T) * KK, rb, cs, K, K);
      for (i32 i = 0; i < K; i++) ub[i] = a * ub[i] + c * cs[i];
    } else {
      for (i32 i = 0; i < K; i++) ub[i] = a * ub[i];
    }
  }
  L->top = mark;
}

/* BucketEntry: each player's reach summed into its buckets (combos in order), the bucket game below on the same rows,
   and every combo getting its bucket's utility back (Solver._bucket_in, _bucket_out) */
static void buckets(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out) {
  i32 K = nd[2], T = nd[3], child = nd[4];
  const i32 *ids = PI(nd[5]);
  size_t mark = L->top, RK = (size_t)(hi - lo) * K;
  f32 *m[2] = {salloc(L, sizeof(f32) * RK), salloc(L, sizeof(f32) * RK)}, *u = salloc(L, sizeof(f32) * RK);
  const f32 *r[2] = {r0, r1};
  for (int q = 0; q < 2; q++)
    for (i32 b = lo; b < hi; b++) {
      const i32 *id = ids + ((size_t)(b % T) * 2 + q) * n;
      const f32 *rb = r[q] + (size_t)(b - lo) * n;
      f32 *mb = m[q] + (size_t)(b - lo) * K;
      for (i32 k = 0; k < K; k++) mb[k] = 0;
      for (i32 i = 0; i < n; i++)
        if (id[i] < K) mb[id[i]] += rb[i];
    }
  cfr(L, child, p, lo, hi, m[0], m[1], u);
  for (i32 b = lo; b < hi; b++) {
    const i32 *id = ids + ((size_t)(b % T) * 2 + p) * n;
    const f32 *ub = u + (size_t)(b - lo) * K;
    f32 *o = out + (size_t)(b - lo) * n;
    for (i32 i = 0; i < n; i++) o[i] = id[i] < K ? ub[id[i]] : 0;
  }
  L->top = mark;
}

/* RiverEntry: each player's reach (rows, W) into its river buckets through X (r @ X[q]), the bucket game below on the
   same rows, and the utilities back through X (X[p] @ u) (Solver._river_in, _river_out) */
static void river(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out) {
  i32 W = nd[2], K = nd[3], T = nd[4], child = nd[5];
  const f32 *X = PF(nd[6]), *Xt = PF(nd[7]);
  size_t mark = L->top, RK = (size_t)(hi - lo) * K, WK = (size_t)W * K;
  f32 *m[2] = {salloc(L, sizeof(f32) * RK), salloc(L, sizeof(f32) * RK)}, *u = salloc(L, sizeof(f32) * RK);
  const f32 *r[2] = {r0, r1};
  for (int q = 0; q < 2; q++)
    for (i32 b = lo; b < hi; b++)
      tmatvec(X + ((size_t)q * T + b % T) * WK, r[q] + (size_t)(b - lo) * W, m[q] + (size_t)(b - lo) * K, K, W);
  cfr(L, child, p, lo, hi, m[0], m[1], u);
  for (i32 b = lo; b < hi; b++)
    tmatvec(Xt + ((size_t)p * T + b % T) * WK, u + (size_t)(b - lo) * K, out + (size_t)(b - lo) * W, W, K);
  L->top = mark;
}

static void cfr(Lane *L, i32 at, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out) {
  const i32 *nd = P + at;
  switch (nd[0]) {
    case DECISION: decision(L, nd, p, lo, hi, r0, r1, out); break;
    case CHANCE: chance(L, nd, p, lo, hi, r0, r1, out); break;
    case CLASSES: classes(L, nd, p, lo, hi, r0, r1, out); break;
    case BUCKETS: buckets(L, nd, p, lo, hi, r0, r1, out); break;
    case RIVER: river(L, nd, p, lo, hi, r0, r1, out); break;
    default: terminal(L, nd, p, lo, hi, p == 0 ? r1 : r0, out); break;
  }
}

/* ------------------------------------------------------------------ threads */
/* A job: a chance node's child on rows lo..lo+rows, handed out `grain` rows at a time. The ticket holds the job's
   number and the next row to take (an odd number while the job is being set up), so a thread only takes rows of the
   job it read; done counts the rows finished. Natively the helpers are pthreads (core_threads starts them); in the
   browser, workers sharing the module's memory, each in core_helper. */
#define MAXT 32
static i32 nthreads = 1;   /* this thread and its helpers */

static struct {
  u64 ticket;
  i32 done, gen, sleepers;
  i32 child, p, lo, rows, grain, helpers;
  const f32 *c0, *c1;
  f32 *u;
  size_t lanes, lane;
} J;

#define LD(x) __atomic_load_n(&(x), __ATOMIC_RELAXED)
#define ST(x, v) __atomic_store_n(&(x), (v), __ATOMIC_RELAXED)

static inline void relax(void) {
#if defined(__aarch64__)
  __asm__ volatile("yield");
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#endif
}

#ifdef THREADS
#ifndef __wasm__
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static i32 started;
#endif

/* this thread's share of job `seq`: chunks of rows while there are any */
static i32 helped;   /* rows the helpers ran (core_helped) */

static void take(Lane *L, u32 seq, i32 child, int p, i32 lo, i32 rows, i32 grain, const f32 *c0, const f32 *c1,
                 f32 *u) {
  for (;;) {
    u64 t = __atomic_load_n(&J.ticket, __ATOMIC_ACQUIRE);
    i32 r = (i32)(u32)t;
    if ((u32)(t >> 32) != seq || r >= rows) return;
    if (__sync_val_compare_and_swap(&J.ticket, t, t + (u32)grain) != t) continue;   /* (no expected value in memory) */
    i32 e = rows - r < grain ? rows : r + grain;
    size_t mark = L->top;
    cfr(L, child, p, lo + r, lo + e, c0 + (size_t)r * n, c1 + (size_t)r * n, u + (size_t)r * n);
    L->top = mark;
    if (L != &M) __atomic_fetch_add(&helped, e - r, __ATOMIC_RELAXED);
    __atomic_fetch_add(&J.done, e - r, __ATOMIC_RELEASE);
  }
}

/* blocks until the ticket's job number isn't `last` (after spinning a while: jobs come every few milliseconds) */
static void await_job(u32 last) {
  for (int spin = 0; spin < 1 << 16; spin++) {
    if ((u32)(__atomic_load_n(&J.ticket, __ATOMIC_ACQUIRE) >> 32) != last) return;
    relax();
  }
#ifdef __wasm__
  i32 g = __atomic_load_n(&J.gen, __ATOMIC_SEQ_CST);
  __atomic_fetch_add(&J.sleepers, 1, __ATOMIC_SEQ_CST);
  if ((u32)(__atomic_load_n(&J.ticket, __ATOMIC_SEQ_CST) >> 32) == last)
    __builtin_wasm_memory_atomic_wait32(&J.gen, g, -1);
  __atomic_fetch_sub(&J.sleepers, 1, __ATOMIC_SEQ_CST);
#else
  pthread_mutex_lock(&mu);
  __atomic_fetch_add(&J.sleepers, 1, __ATOMIC_SEQ_CST);
  while ((u32)(__atomic_load_n(&J.ticket, __ATOMIC_SEQ_CST) >> 32) == last) pthread_cond_wait(&cv, &mu);
  __atomic_fetch_sub(&J.sleepers, 1, __ATOMIC_SEQ_CST);
  pthread_mutex_unlock(&mu);
#endif
}

static void wake(void) {
  __atomic_fetch_add(&J.gen, 1, __ATOMIC_SEQ_CST);
  if (!__atomic_load_n(&J.sleepers, __ATOMIC_SEQ_CST)) return;
#ifdef __wasm__
  __builtin_wasm_memory_atomic_notify(&J.gen, 0xffffffffu);
#else
  pthread_mutex_lock(&mu);
  pthread_cond_broadcast(&cv);
  pthread_mutex_unlock(&mu);
#endif
}

/* helper i: takes rows of every job it sees, in its lane of the job's scratch */
static Lane hl[MAXT];
static void helper(i32 i) {
  u32 last = 0;
  for (;;) {
    await_job(last);
    u64 t = __atomic_load_n(&J.ticket, __ATOMIC_ACQUIRE);
    u32 seq = (u32)(t >> 32);
    if (seq & 1) continue;   /* being set up */
    i32 child = LD(J.child), p = LD(J.p), lo = LD(J.lo), rows = LD(J.rows), grain = LD(J.grain);
    i32 helpers = LD(J.helpers);
    const f32 *c0 = LD(J.c0), *c1 = LD(J.c1);
    f32 *u = LD(J.u);
    size_t lanes = LD(J.lanes), lane = LD(J.lane);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if ((u32)(__atomic_load_n(&J.ticket, __ATOMIC_RELAXED) >> 32) != seq) continue;   /* (read across two jobs) */
    last = seq;
    if (i >= helpers) continue;
    Lane *L = &hl[i];   /* (not on the stack: see core_stack) */
    L->top = lanes + (size_t)i * lane, L->lim = lanes + (size_t)(i + 1) * lane, L->par = 1;
    take(L, seq, child, p, lo, rows, grain, c0, c1, u);
  }
}

#ifdef __wasm__
/* a browser helper's loop (worker.js's helpers call it once; it never returns). Each instance of the module starts
   with the same stack pointer, so a helper first sets its own to core_stack(i), a stack of its own. */
static unsigned char stacks[MAXT][16384] __attribute__((aligned(16)));
EXPORT(core_stack) u32 core_stack(i32 i) { return (u32)(uintptr_t)stacks[i < 0 || i >= MAXT ? 0 : i] + 16384; }
EXPORT(core_helper) void core_helper(i32 i) { helper(i); }
#else
static void *run_helper(void *i) {
  helper((i32)(intptr_t)i);
  return 0;
}
#endif
#endif

/* the rows of jobs the helpers have run (so a test, or the page, can tell they're running) */
EXPORT(core_helped) i32 core_helped(void) {
#ifdef THREADS
  return __atomic_load_n(&helped, __ATOMIC_RELAXED);
#else
  return 0;
#endif
}

/* this thread and t - 1 helpers (natively started here, in the browser by worker.js); returns the number (t 0: just
   that). */
EXPORT(core_threads) i32 core_threads(i32 t) {
  if (t < 1) return nthreads;
#ifdef THREADS
  if (t > MAXT) t = MAXT;
#ifndef __wasm__
  for (; started < t - 1; started++) {
    pthread_t th;
    if (pthread_create(&th, 0, run_helper, (void *)(intptr_t)started) != 0) break;
    pthread_detach(th);
  }
  if (t > started + 1) t = started + 1;
#endif
#else
  t = 1;
#endif
  if (t != nthreads) {
    nthreads = t;
    lane_size = measured = 0;   /* (the chunks change with the number) */
  }
  return nthreads;
}

/* a chance node's child on rows lo..hi: the first chance node on the way down (not already in a job's rows) runs it
   in chunks, one chunk at a time while measuring the scratch a chunk takes (an iteration, serially, once a program),
   then as a job */
static void split(Lane *L, i32 child, int p, i32 lo, i32 hi, const f32 *c0, const f32 *c1, f32 *u) {
  i32 rows = hi - lo;
  if (L->par || nthreads < 2 || rows < 2) {
    cfr(L, child, p, lo, hi, c0, c1, u);
    return;
  }
  i32 grain = (rows + 2 * nthreads - 1) / (2 * nthreads);
  if (!lane_size) {
    L->par = 1;
    for (i32 r = 0; r < rows; r += grain) {
      i32 e = rows - r < grain ? rows : r + grain;
      size_t mark = L->top;
      L->peak = mark;
      cfr(L, child, p, lo + r, lo + e, c0 + (size_t)r * n, c1 + (size_t)r * n, u + (size_t)r * n);
      if (L->peak - mark > measured) measured = L->peak - mark;
      L->top = mark;
    }
    L->par = 0;
    return;
  }
#ifdef THREADS
  i32 helpers = nthreads - 1;
  size_t mark = L->top;
  unsigned char *lanes = salloc(L, lane_size * helpers);
  u32 seq = (u32)(__atomic_load_n(&J.ticket, __ATOMIC_RELAXED) >> 32) + 2;
  __atomic_store_n(&J.ticket, ((u64)(seq - 1) << 32) | 0xffffffffu, __ATOMIC_SEQ_CST);
  ST(J.child, child), ST(J.p, p), ST(J.lo, lo), ST(J.rows, rows), ST(J.grain, grain), ST(J.helpers, helpers);
  ST(J.c0, c0), ST(J.c1, c1), ST(J.u, u), ST(J.lanes, (size_t)(lanes - base)), ST(J.lane, lane_size);
  ST(J.done, 0);
  __atomic_store_n(&J.ticket, (u64)seq << 32, __ATOMIC_SEQ_CST);
  wake();
  L->par = 1;
  take(L, seq, child, p, lo, rows, grain, c0, c1, u);
  L->par = 0;
  for (int spin = 0; __atomic_load_n(&J.done, __ATOMIC_ACQUIRE) < rows; spin++) {
    relax();
#ifndef __wasm__
    if (spin > 1 << 12) sched_yield();
#endif
  }
  L->top = mark;
#endif
}

static void load(u32 prog) {
  P = PI(prog);
  n = P[1];
  ca = PI(P[2]);
  cb = PI(P[3]);
}

static i32 rows_of(i32 at) { return P[at] == DECISION ? P[at + 3] : P[at + 1]; }

/* p's utilities (rows, n) at the node record `at` for reaches r0, r1 (rows, n) each: md AVG on the average strategies,
   BR p's best response (Solver.node_action_evs, _best_response). Returns 0, or -1 on a bad program. (Threaded from
   the root, whose chance nodes the lanes were measured on; below it, serially.) */
EXPORT(core_eval) i32 core_eval(u32 prog, i32 at, i32 p, u32 r0, u32 r1, u32 out, i32 md) {
  load(prog);
  if (P[0] != MAGIC) return -1;
  mode = md;
  M.par = at != P[4] || !lane_size;
  cfr(&M, at, p, 0, rows_of(at), PF(r0), PF(r1), PF(out));
  M.par = 0;
  mode = TRAIN;
  return 0;
}

/* `iters` training iterations of the program at `prog` (both players' passes, then the discount: factors holds
   [mode, pos, neg, ssum] per iteration, mode 1 flooring the regrets at 0 first). Returns the iterations run, or -1 on
   a bad program. */
EXPORT(core_run) i32 core_run(u32 prog, i32 iters, u32 factors) {
  load(prog);
  if (P[0] != MAGIC) return -1;
  mode = TRAIN;
  i32 root = P[4], B = rows_of(root);
  const f32 *r0 = PF(P[5]), *r1 = PF(P[6]);
  f32 *regret = PF(P[7]), *ssum = PF(P[8]);
  size_t len = (u32)P[9];
  const f32 *fac = PF(factors);
  for (i32 t = 0; t < iters; t++) {
    for (int p = 0; p < 2; p++) {
      size_t mark = M.top;
      f32 *u = salloc(&M, sizeof(f32) * B * n);
      cfr(&M, root, p, 0, B, r0, r1, u);
      M.top = mark;
    }
    if (!lane_size && measured) lane_size = (measured + 63) & ~(size_t)63;
    f32 mode = fac[4 * t], pos = fac[4 * t + 1], neg = fac[4 * t + 2], sf = fac[4 * t + 3];
    if (mode != 0) {
      for (size_t x = 0; x < len; x++) regret[x] = (regret[x] > 0 ? regret[x] : 0) * pos;
    } else {
      for (size_t x = 0; x < len; x++) regret[x] *= regret[x] > 0 ? pos : neg;
    }
    for (size_t x = 0; x < len; x++) ssum[x] *= sf;
  }
  return iters;
}
