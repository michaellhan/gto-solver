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
 *   header  [MAGIC, n, ca, cb, root, r0, r1, regret, ssum, flat_len, nstacks, stacks, trim, live0, live1, l0, l1,
 *            ngames, games]
 *   DECISION [1, player, A, B, W, regret, ssum, child_0 .. child_A-1]   regret, ssum: (B, A, W) blocks
 *   FOLD     [2, B, f0, f1]                     u_p = f_p * compat(ro)
 *   SHOWDOWN [3, B, ranking, a0, c0, a1, c1]    u_p = a_p * (sum of ro over weaker - stronger) + c_p * compat(ro)
 *   LEAF     [4, B, M, a0, c0, a1, c1, st, j]  the same with M (B, n, n) @ ro (an all-in's equity over the runouts);
 *                                              st -1, or its products batched: its stack, j its row there (below)
 *   CHANCE   [5, B, K, child, masks, w, norm]   child rows b*K+k; masks, w (B*K, n); norm (B, n)
 *   CLASSES  [6, B, K, child, masks, norm, G, flat, starts, sizes, nG]   river classes (Chance.spread, Chance.gather)
 *   RANKING  [B, n, oi, pi, rec, kslots, total, live]   a Ranking's tables as ranking() reads them (gto/core.py
 *                                              _ranking): oi (B, n4), pi (B, m4) uint16, rec (B, n, 8) uint16,
 *                                              kslots (2, 52), total a slot, live (B, n)
 * A flop solve's turn and river by hand buckets (Config.river_buckets, river_strength): below an entry the rows are K
 * buckets wide, row b's tables those of board b % T (T, the period of the entry's boards).
 *   BUCKETS  [7, B, K, T, child, ids, game, lists]   BucketEntry: ids (T, 2, n), each combo's bucket by player (K:
 *                                              none); game: a deferred game's reaches and utilities (below), or 0;
 *                                              lists (T, 2, K + 1 + n) with it: each bucket's combos (starts, combos)
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

/* bytes a helper's chunk of rows may take: 0 until a serial pass measured them. [0] a whole pass's (a training pass,
   an evaluation: the same scratch), [1] a deferred training pass's (below: its chunks a row at a time, less) */
static size_t lane_size[2], measured[2];
static int deferring;   /* (which: 1 in a deferred training pass) */
static void sized(void) {
  for (int k = 0; k < 2; k++)
    if (!lane_size[k] && measured[k]) lane_size[k] = (measured[k] + 63) & ~(size_t)63;
}

/* Frees the arena (natively its pages go back to the system). */
EXPORT(core_reset) void core_reset(void) {
#ifndef __wasm__
  if (base && touched > ((size_t)64 << 20)) {
    madvise(base, touched, MADV_FREE);
    touched = 0;
  }
#endif
  M.top = M.peak = 0;
  lane_size[0] = lane_size[1] = measured[0] = measured[1] = 0;
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
static int trim;   /* this pass's batched equity leaves trimmed to the ranges (below) */

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

/* four floats gathered: a[ix[0..3]] */
#define G4(a, ix) ((v4){(a)[(ix)[0]], (a)[(ix)[1]], (a)[(ix)[2]], (a)[(ix)[3]]})

/* ro: one row of the opponent's reach -> out: f times the mass of opponent combos sharing no card with each combo
   (Solver.compat_sum: the total, minus each of the combo's cards' combos, plus the combo itself); card: 52 floats.
   (Four combos at a time on the way out, each combo's arithmetic the same, the cards' sums gathered.) */
static void compat(const f32 *ro, f32 *card, f32 f, f32 *out) {
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
  v4 T = {tot, tot, tot, tot}, F = {f, f, f, f};
  i32 i = 0;
  for (; i + 4 <= n; i += 4) {
    v4 x = T - G4(card, ca + i);
    x -= G4(card, cb + i);
    x += *(const v4u *)(ro + i);
    *(v4u *)(out + i) = F * x;
  }
  for (; i < n; i++) out[i] = f * (tot - card[ca[i]] - card[cb[i]] + ro[i]);
}

/* The prefix sums of a row of reaches in two orders at once, cum over n entries and cs over 2n (Ranking.utility's
   quarter_cumsum): each order's sum runs in four quarters, each quarter's running sum from 0 and then the quarters'
   totals before it added, as numpy sums them. The quarters' sums advance side by side in one vector (both orders'
   chains together: each add waits on the one before), so the sums land interleaved: the sum to p (p of the order's
   entries) at slot(p), 0 for p = 0, 1 + 4 j + l for the j-th sum of quarter l (p = l q + j + 1, q a quarter's
   length), p itself in the tail past the quarters (gto/core.py's _slots). oi, pi: each order's entries, its quarters
   interleaved likewise (four uint16 to a u64), then its tail. */
#define U16(x, s) ((u32)((x) >> (s)) & 0xffff)
static void prefixes(const f32 *ro, const u64 *oi, const u64 *pi, f32 *cum, f32 *cs) {
  i32 q1 = n / 4, q2 = (2 * n) / 4, j = 0;
  v4 A = {0, 0, 0, 0}, C = A;
  cum[0] = 0;
  cs[0] = 0;
  for (; j < q1; j++) {
    u64 x = oi[j], y = pi[j];
    A += (v4){ro[U16(x, 0)], ro[U16(x, 16)], ro[U16(x, 32)], ro[U16(x, 48)]};
    C += (v4){ro[U16(y, 0)], ro[U16(y, 16)], ro[U16(y, 32)], ro[U16(y, 48)]};
    *(v4u *)(cum + 1 + 4 * j) = A;
    *(v4u *)(cs + 1 + 4 * j) = C;
  }
  for (; j < q2; j++) {
    u64 y = pi[j];
    C += (v4){ro[U16(y, 0)], ro[U16(y, 16)], ro[U16(y, 32)], ro[U16(y, 48)]};
    *(v4u *)(cs + 1 + 4 * j) = C;
  }
  const uint16_t *ot = (const uint16_t *)oi, *pt = (const uint16_t *)pi;
  f32 a3 = A[3], c3 = C[3];
  for (i32 t = 4 * q1; t < n; t++) cum[t + 1] = a3 += ro[ot[t]];
  for (i32 t = 4 * q2; t < 2 * n; t++) cs[t + 1] = c3 += ro[pt[t]];
  /* each quarter's sums plus the quarters' before it (adding 0 to the first: its sums are +0 or more, unchanged) */
  f32 o1 = A[0], o2 = o1 + A[1], o3 = o2 + A[2];
  f32 p1 = C[0], p2 = p1 + C[1], p3 = p2 + C[2];
  v4 off = {0, o1, o2, o3}, poff = {0, p1, p2, p3};
  for (j = 0; j < q1; j++) *(v4u *)(cum + 1 + 4 * j) += off;
  for (i32 t = 4 * q1; t < n; t++) cum[t + 1] += o3;
  for (j = 0; j < q2; j++) *(v4u *)(cs + 1 + 4 * j) += poff;
  for (i32 t = 4 * q2; t < 2 * n; t++) cs[t + 1] += p3;
}

/* Ranking.utility for row b: sum over compatible opponent combos of reach * (+1 weaker, -1 stronger), times live (0
   where the combo holds a board card), times a with `scaled`. Each combo's sum adds the same terms in the same order
   as numpy's, four combos at a time: its record (8 uint16: the sums to its weaker and its weaker-or-tied combos, its
   cards, and the sums to the same two in each card's list of pairs, as slots) in two u64, the terms gathered. */
static void ranking(Lane *L, const i32 *R, i32 b, const f32 *ro, f32 *u, int scaled, f32 a) {
  i32 n4 = (n + 3) & ~3, m4 = (2 * n + 3) & ~3;
  const u64 *oi = (const u64 *)(PI(R[2])) + (size_t)b * n4 / 4, *pi = (const u64 *)(PI(R[3])) + (size_t)b * m4 / 4;
  const u64 *rec = (const u64 *)(PI(R[4])) + (size_t)b * 2 * n;
  const i32 *kend = PI(R[5]), *kstart = kend + 52;
  const f32 *live = PF(R[7]) + (size_t)b * n;
  size_t mark = L->top;
  f32 *cum = salloc(L, sizeof(f32) * (n + 1)), *cs = salloc(L, sizeof(f32) * (2 * n + 1));
  f32 *k = salloc(L, sizeof(f32) * 52);   /* each card's term: its pairs' sum before them and to their end */
  prefixes(ro, oi, pi, cum, cs);
  for (i32 c = 0; c < 52; c++) k[c] = cs[kend[c]] + cs[kstart[c]];
  f32 total = cum[R[6]];
  v4 T = {total, total, total, total}, S = {a, a, a, a};
  i32 i = 0;
  for (; i + 4 <= n; i += 4) {
    const u64 *r = rec + 2 * i;
    u64 a0 = r[0], a1 = r[2], a2 = r[4], a3 = r[6], b0 = r[1], b1 = r[3], b2 = r[5], b3 = r[7];
    v4 x = {cum[U16(a0, 0)], cum[U16(a1, 0)], cum[U16(a2, 0)], cum[U16(a3, 0)]};
    x += (v4){cum[U16(a0, 16)], cum[U16(a1, 16)], cum[U16(a2, 16)], cum[U16(a3, 16)]};
    x += (v4){k[U16(a0, 32)], k[U16(a1, 32)], k[U16(a2, 32)], k[U16(a3, 32)]};
    x += (v4){k[U16(a0, 48)], k[U16(a1, 48)], k[U16(a2, 48)], k[U16(a3, 48)]};
    x -= (v4){cs[U16(b0, 0)], cs[U16(b1, 0)], cs[U16(b2, 0)], cs[U16(b3, 0)]};
    x -= (v4){cs[U16(b0, 16)], cs[U16(b1, 16)], cs[U16(b2, 16)], cs[U16(b3, 16)]};
    x -= (v4){cs[U16(b0, 32)], cs[U16(b1, 32)], cs[U16(b2, 32)], cs[U16(b3, 32)]};
    x -= (v4){cs[U16(b0, 48)], cs[U16(b1, 48)], cs[U16(b2, 48)], cs[U16(b3, 48)]};
    x -= T;
    x *= *(const v4u *)(live + i);
    if (scaled) x *= S;
    *(v4u *)(u + i) = x;
  }
  for (; i < n; i++) {
    u64 r0 = rec[2 * i], r1 = rec[2 * i + 1];
    f32 x = cum[U16(r0, 0)];
    x += cum[U16(r0, 16)];
    x += k[U16(r0, 32)];
    x += k[U16(r0, 48)];
    x -= cs[U16(r1, 0)];
    x -= cs[U16(r1, 16)];
    x -= cs[U16(r1, 32)];
    x -= cs[U16(r1, 48)];
    x -= total;
    x *= live[i];
    if (scaled) x *= a;
    u[i] = x;
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
    for (i32 b = lo; b < hi; b++) compat(ro + (size_t)(b - lo) * n, card, f, u + (size_t)(b - lo) * n);
  } else {
    f32 a = fbits(nd[3 + 2 * p]), c = fbits(nd[4 + 2 * p]);
    for (i32 b = lo; b < hi; b++) {
      const f32 *r = ro + (size_t)(b - lo) * n;
      f32 *ub = u + (size_t)(b - lo) * n;
      if (nd[0] == SHOWDOWN && c == 0) {   /* (a times it as it's written) */
        ranking(L, P + nd[2], b, r, ub, 1, a);
        continue;
      }
      if (nd[0] == SHOWDOWN) {
        ranking(L, P + nd[2], b, r, ub, 0, 1);
      } else if (nd[7] >= 0) {   /* its product, from the pass's batch */
        const i32 *st = PI(P[11]) + 10 * nd[7];
        i32 lp = trim ? P[15 + p] : n;
        const f32 *pu = PF(st[5]) + ((size_t)b * st[3] + nd[8]) * lp;
        if (trim) {
          const i32 *live = PI(P[13 + p]);
          for (i32 i = 0; i < n; i++) ub[i] = 0;
          for (i32 i = 0; i < lp; i++) ub[live[i]] = pu[i];
        } else {
          for (i32 i = 0; i < n; i++) ub[i] = pu[i];
        }
      } else {
        matvec(PF(nd[2]) + (size_t)b * n * n, r, ub);
      }
      if (c != 0) {
        compat(r, card, 1, cs);
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
    for (i32 a = 0; a < A; a++) {
      const f32 *ra = r + (size_t)a * W;
      f32 *sa = sb + (size_t)a * W;
      if (avg)
        for (i32 i = 0; i < W; i++) sa[i] = ra[i];
      else
        for (i32 i = 0; i < W; i++) sa[i] = ra[i] > 0 ? ra[i] : 0;
      if (a == 0)
        for (i32 i = 0; i < W; i++) tot[i] = sa[i];
      else
        for (i32 i = 0; i < W; i++) tot[i] += sa[i];
    }
    /* (one division a combo; a sum under float32's smallest normal counts as none: its reciprocal can overflow, and
       a zero regret times inf is NaN) */
    for (i32 i = 0; i < W; i++) tot[i] = tot[i] < 0x1p-126f ? 0 : 1 / tot[i];
    for (i32 a = 0; a < A; a++)   /* (the product taken either way: a load under a condition doesn't vectorize) */
      for (i32 i = 0; i < W; i++) {
        f32 v = sb[a * W + i] * tot[i];
        sb[a * W + i] = tot[i] == 0 ? uni : v;
      }
  }
  L->top = mark;
}

static void cfr(Lane *L, i32 at, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out);
static void split(Lane *L, i32 child, int p, i32 lo, i32 hi, const f32 *c0, const f32 *c1, f32 *u);

/* TRAIN: a training pass (Solver._cfr, update=True); AVG: both players on their average strategies, no update (use_avg);
   BR: p best-responds to the opponent's average strategy (Solver._best_response) */
enum { TRAIN = 0, AVG = 1, BR = 2 };
static int mode = TRAIN;
/* a training pass over deferred bucket games (below): COLLECT walks down storing their reaches, READ is the pass
   proper, reading their utilities. Natively only: the browser never solves a dealt river (the wasm builds leave them
   out) */
enum { WHOLE = 0, COLLECT = 1, READ = 2 };
static int phase = WHOLE;
#ifdef __wasm__
#define DEFER 0
#else
#define DEFER 1
#endif

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
  if (phase == COLLECT) {
    /* down only: the opponent's strategies into its reach, as below */
    const f32 *ro = p == 0 ? r1 : r0;
    f32 *reach = player == p ? 0 : salloc(L, sizeof(f32) * RW);
    if (player != p) strategy(L, regret, lo, hi, A, W, 0, s);
    for (i32 a = 0; a < A; a++) {
      if (is_terminal(kids[a])) continue;
      if (player == p) {
        cfr(L, kids[a], p, lo, hi, r0, r1, out);
        continue;
      }
      for (i32 b = lo; b < hi; b++) {
        size_t j = (size_t)(b - lo);
        const f32 *sa = s + (j * A + a) * W, *rb = ro + j * W;
        f32 *x = reach + j * W;
        for (i32 i = 0; i < W; i++) x[i] = rb[i] * sa[i];
      }
      cfr(L, kids[a], p, lo, hi, p == 0 ? r0 : reach, p == 0 ? reach : r1, out);
    }
    L->top = mark;
    return;
  }
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

/* a dealt river's chance node in a deferred pass (its child a deferred game's entry): COLLECT sums each player's reach
   on each card into the card's buckets (the entry's sums of the reaches chance() would pass it), READ takes each
   combo's utility on each card from its bucket and sums them over the cards as chance() does */
static void dealt(const i32 *nd, const i32 *bk, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out) {
  i32 K = nd[2], B = bk[1], KB = bk[2], T = bk[3];
  const f32 *w = PF(nd[5]), *norm = PF(nd[6]);
  const i32 *ids = PI(bk[5]);
  f32 *g = PF(bk[6]);
  if (phase == COLLECT) {
    /* (by the entry's lists: each bucket's combos in order, its sum in a register; a combo in a bucket isn't blocked
       by the card, its mask 1: gto/core.py checks) */
    const f32 *r[2] = {r0, r1};
    for (int q = 0; q < 2; q++)
      for (i32 b = lo; b < hi; b++) {
        const f32 *rb = r[q] + (size_t)(b - lo) * n;
        for (i32 k = 0; k < K; k++) {
          i32 row = b * K + k;
          const i32 *at = PI(bk[7]) + ((size_t)(row % T) * 2 + q) * (KB + 1 + n), *cs = at + KB + 1;
          f32 *mb = g + ((size_t)q * B + row) * KB;
          for (i32 j = 0; j < KB; j++) {
            f32 x = 0;
            for (i32 e = at[j]; e < at[j + 1]; e++) x += rb[cs[e]];
            mb[j] = x;
          }
        }
      }
    return;
  }
  const f32 *u = g + (size_t)2 * B * KB;
  for (i32 b = lo; b < hi; b++) {
    f32 *o = out + (size_t)(b - lo) * n;
    for (i32 k = 0; k < K; k++) {
      i32 row = b * K + k;
      const i32 *id = ids + ((size_t)(row % T) * 2 + p) * n;
      const f32 *ub = u + (size_t)row * KB, *wb = w + (size_t)row * n;
      if (k == 0)
        for (i32 i = 0; i < n; i++) {
          f32 x = id[i] < KB ? ub[id[i]] : 0;
          o[i] = x * wb[i];
        }
      else
        for (i32 i = 0; i < n; i++) {
          f32 x = id[i] < KB ? ub[id[i]] : 0;
          o[i] += x * wb[i];
        }
    }
    for (i32 i = 0; i < n; i++) o[i] = o[i] / norm[(size_t)b * n + i];
  }
}

static void chance(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out) {
  i32 K = nd[2], child = nd[3];
  if (phase != WHOLE && P[child] == BUCKETS && P[child + 6]) {
    dealt(nd, P + child, p, lo, hi, r0, r1, out);
    return;
  }
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
  if (phase == COLLECT) {
    L->top = mark;
    return;
  }
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
  for (i32 b = lo; b < hi && phase != COLLECT; b++) {
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

/* tmatvec's u = M @ x over x's nonzero entries alone, listed (their indices nz, values v, m of them): the same sums,
   without a branch an entry (most of a bucket game's reaches are 0, at no pattern) */
static void smatvec(const f32 *Mt, const i32 *nz, const f32 *v, i32 m, f32 *u, i32 R) {
  i32 r = 0;
  for (; r + 32 <= R; r += 32) {
    v4 a0 = {0, 0, 0, 0}, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (i32 k = 0; k < m; k++) {
      f32 xj = v[k];
      v4 s = {xj, xj, xj, xj};
      const f32 *mt = Mt + (size_t)nz[k] * R + r;
      a0 += *(const v4u *)(mt) * s;
      a1 += *(const v4u *)(mt + 4) * s;
      a2 += *(const v4u *)(mt + 8) * s;
      a3 += *(const v4u *)(mt + 12) * s;
      a4 += *(const v4u *)(mt + 16) * s;
      a5 += *(const v4u *)(mt + 20) * s;
      a6 += *(const v4u *)(mt + 24) * s;
      a7 += *(const v4u *)(mt + 28) * s;
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
    for (i32 k = 0; k < m; k++) {
      f32 xj = v[k];
      v4 s = {xj, xj, xj, xj};
      a += *(const v4u *)(Mt + (size_t)nz[k] * R + r) * s;
    }
    *(v4u *)(u + r) = a;
  }
  for (; r < R; r++) {
    f32 y = 0;
    for (i32 k = 0; k < m; k++) y += Mt[(size_t)nz[k] * R + r] * v[k];
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

/* A dealt river (Config.river_dealt) is a bucket game on every card under each turn, the same cards' tables (each row's
   E and kap, 32 KB a player at 64 buckets) under every turn line: a pass reaching them one entry at a time read every
   row's again at each of its terminals (all the rows at once: megabytes, out of cache) and again at each entry. So a
   training pass over a tree whose bucket games are just decisions and bucket terminals (gto/core.py's deferred games,
   the header's ngames records at games) runs in three phases: a COLLECT walk down stores each game's bucket reaches,
   the games are played a row at a time, every game on the row before the next row (the row's tables read once a
   pass, from cache at every terminal; games()), and the READ pass proper takes each game's utilities where the walk
   would have played it. Each row of each game gets the same arithmetic as in one walk. A game's storage: its reaches
   m0, m1 and p's utilities u, (B, K) each. Elsewhere (evaluations) a game is played BROWS rows at a time. */
#define BROWS 2

/* BucketEntry: each player's reach summed into its buckets (combos in order), the bucket game below on the same rows,
   and every combo getting its bucket's utility back (Solver._bucket_in, _bucket_out) */
static void buckets(Lane *L, const i32 *nd, int p, i32 lo, i32 hi, const f32 *r0, const f32 *r1, f32 *out) {
  i32 K = nd[2], T = nd[3], child = nd[4];
  const i32 *ids = PI(nd[5]);
  if (phase == COLLECT) return;   /* (no deferred game below an entry) */
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
  for (i32 c = lo; c < hi; c += BROWS) {   /* (a few rows at a time: their tables stay in cache through the game) */
    i32 e = hi - c < BROWS ? hi : c + BROWS;
    size_t o = (size_t)(c - lo) * K;
    cfr(L, child, p, c, e, m[0] + o, m[1] + o, u + o);
  }
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

/* ------------------------------------------------------------------ batched equity leaves */
/* A tree of many equity leaves (gto/core.py's LEAF_ROWS_BLAS: the library's flop, every turn line ending in one) runs
   a pass in two phases, as numpy's Solver._leaf_batch does: a walk down puts the opponent's reach at each leaf row
   into its matrix stack's rows, each matrix multiplies all of its rows at once (it is read once a pass, where a leaf
   at a time read it again for each), and the pass reads the leaves' products. The header's stacks: nstacks records
   [M, S, R, X, U, jlo, jhi, T] (M and T two words each: addresses, natively the solver's own arrays), M (S, n, n), X
   the reaches and U their products (S, R, n): leaf j's row b is row b * R + j there (a leaf's rows are its stack's
   matrices, its leaves numbered in program order, so a subtree's are a run of them); jlo..jhi: the leaves the walk
   reached. A training pass with `trim` multiplies just the combos in each player's range (live0, live1: l0 and l1 of
   them, the root reaches' nonzero ones): T = M[live0][:, live1] (S, l0, l1), the reaches gathered to the opponent's
   range (rows lq wide), the products to p's (lp wide). M is antisymmetric (an equity's win minus loss), so player 1's
   are -(X @ T), and the combos outside p's range get 0 (numpy's trim_leaves: their reach is 0, so their regrets never
   matter). Evaluations multiply the whole of M. Natively only: in the browser (no BLAS) gto/core.py leaves them one
   at a time, so the wasm builds go without it. */
#ifdef __wasm__
static void batch(i32 at, int p, i32 rows, const f32 *ro) { (void)at, (void)p, (void)rows, (void)ro; }
#else
#ifdef __APPLE__
#define BLAS 1   /* Accelerate (gto/core.py links it): its sgemm runs on the matrix units */
void cblas_sgemm(int order, int ta, int tb, int m, int nn, int k, float alpha, const float *a, int lda,
                 const float *b, int ldb, float beta, float *c, int ldc);
#endif
static i32 use_blas = 1;

/* 1: products by BLAS where there is one, 0: by the loops below; -1 just asks. Returns whether BLAS runs them. */
EXPORT(core_gemm) i32 core_gemm(i32 on) {
  if (on >= 0) use_blas = on;
#ifdef BLAS
  return use_blas;
#else
  return 0;
#endif
}

static inline const f32 *addr(const i32 *w) { return (const f32 *)(uintptr_t)((u64)(u32)w[0] | (u64)(u32)w[1] << 32); }

/* U (R, m) = X (R, k) @ A^T for A (m, k): each output a row of X dotted with a row of A, four rows of each at a time
   in 16 registers (a load feeds four products), A in blocks of 64 rows that stay in cache while every row of X
   passes */
static void gemm_nt(const f32 *X, const f32 *A, f32 *U, i32 R, i32 m, i32 k) {
#ifdef BLAS
  if (use_blas) {
    cblas_sgemm(101, 111, 112, R, m, k, 1.0f, X, k, A, k, 0.0f, U, m);   /* row-major, A transposed */
    return;
  }
#endif
  i32 k4 = k & ~3;
  for (i32 i0 = 0; i0 < m; i0 += 64) {
    i32 i1 = m - i0 < 64 ? m : i0 + 64;
    for (i32 r = 0; r < R; r += 4) {
      i32 rr = R - r < 4 ? R - r : 4;
      for (i32 i = i0; i < i1; i += 4) {
        i32 ii = i1 - i < 4 ? i1 - i : 4;
        if (rr == 4 && ii == 4) {
          v4 acc[4][4];
          for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) acc[a][b] = (v4){0, 0, 0, 0};
          for (i32 j = 0; j < k4; j += 4) {
            v4 xv[4], av[4];
            for (int a = 0; a < 4; a++) xv[a] = *(const v4u *)(X + (size_t)(r + a) * k + j);
            for (int b = 0; b < 4; b++) av[b] = *(const v4u *)(A + (size_t)(i + b) * k + j);
            for (int a = 0; a < 4; a++)
              for (int b = 0; b < 4; b++) acc[a][b] += xv[a] * av[b];
          }
          for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) {
              v4 v = acc[a][b];
              f32 t = (v[0] + v[1]) + (v[2] + v[3]);
              for (i32 j = k4; j < k; j++) t += X[(size_t)(r + a) * k + j] * A[(size_t)(i + b) * k + j];
              U[(size_t)(r + a) * m + i + b] = t;
            }
        } else {
          for (int a = 0; a < rr; a++)
            for (int b = 0; b < ii; b++) {
              const f32 *x = X + (size_t)(r + a) * k, *y = A + (size_t)(i + b) * k;
              v4 v = {0, 0, 0, 0};
              for (i32 j = 0; j < k4; j += 4) v += *(const v4u *)(x + j) * *(const v4u *)(y + j);
              f32 t = (v[0] + v[1]) + (v[2] + v[3]);
              for (i32 j = k4; j < k; j++) t += x[j] * y[j];
              U[(size_t)(r + a) * m + i + b] = t;
            }
        }
      }
    }
  }
}

/* U (R, m) = -(X (R, k) @ A) for A (k, m): four rows of U by 16 columns in 16 registers, a row of A's 16 columns
   loaded once for the four (each output summed over k first to last), A in blocks of 256 columns */
static void gemm_nn_neg(const f32 *X, const f32 *A, f32 *U, i32 R, i32 k, i32 m) {
#ifdef BLAS
  if (use_blas) {
    cblas_sgemm(101, 111, 111, R, m, k, -1.0f, X, k, A, m, 0.0f, U, m);
    return;
  }
#endif
  for (i32 c0 = 0; c0 < m; c0 += 256) {
    i32 c1 = m - c0 < 256 ? m : c0 + 256;
    for (i32 r = 0; r < R; r += 4) {
      i32 rr = R - r < 4 ? R - r : 4;
      i32 c = c0;
      for (; c + 16 <= c1 && rr == 4; c += 16) {
        v4 acc[4][4];
        for (int a = 0; a < 4; a++)
          for (int b = 0; b < 4; b++) acc[a][b] = (v4){0, 0, 0, 0};
        for (i32 j = 0; j < k; j++) {
          const f32 *y = A + (size_t)j * m + c;
          v4 av[4];
          for (int b = 0; b < 4; b++) av[b] = *(const v4u *)(y + 4 * b);
          for (int a = 0; a < 4; a++) {
            f32 x = X[(size_t)(r + a) * k + j];
            v4 xs = {x, x, x, x};
            for (int b = 0; b < 4; b++) acc[a][b] += xs * av[b];
          }
        }
        for (int a = 0; a < 4; a++)
          for (int b = 0; b < 4; b++) *(v4u *)(U + (size_t)(r + a) * m + c + 4 * b) = -acc[a][b];
      }
      for (int a = 0; a < rr; a++)
        for (i32 cc = c; cc < c1; cc++) {
          f32 t = 0;
          for (i32 j = 0; j < k; j++) t += X[(size_t)(r + a) * k + j] * A[(size_t)j * m + cc];
          U[(size_t)(r + a) * m + cc] = -t;
        }
    }
  }
}

/* a node the walk down goes into: one with batched leaves at or below it */
static int below(i32 at) {
  i32 t = P[at];
  return t == DECISION || t == CHANCE || t == CLASSES || (t == LEAF && P[at + 7] >= 0);
}

/* The walk down (Solver._down with `want`): the opponent's reach ro (rows lo..hi) at node `at`, times the opponent's
   strategies (cfr's: the regrets', in an evaluation the average) and the chance nodes' masks as cfr multiplies them,
   into the batched leaves' rows */
static void down(Lane *L, i32 at, int p, i32 lo, i32 hi, const f32 *ro) {
  const i32 *nd = P + at;
  size_t mark = L->top;
  if (nd[0] == DECISION) {
    i32 A = nd[2], W = nd[4], any = 0;
    const i32 *kids = nd + 7;
    for (i32 a = 0; a < A; a++) any |= below(kids[a]);
    if (!any) return;
    if (nd[1] == p) {
      for (i32 a = 0; a < A; a++)
        if (below(kids[a])) down(L, kids[a], p, lo, hi, ro);
      return;
    }
    size_t RW = (size_t)(hi - lo) * W;
    f32 *s = salloc(L, sizeof(f32) * RW * A), *reach = salloc(L, sizeof(f32) * RW);
    strategy(L, PF(nd[mode == TRAIN ? 5 : 6]), lo, hi, A, W, mode != TRAIN, s);
    for (i32 a = 0; a < A; a++) {
      if (!below(kids[a])) continue;
      for (i32 b = lo; b < hi; b++) {
        size_t j = (size_t)(b - lo);
        const f32 *sa = s + (j * A + a) * W, *rb = ro + j * W;
        f32 *x = reach + j * W;
        for (i32 i = 0; i < W; i++) x[i] = rb[i] * sa[i];
      }
      down(L, kids[a], p, lo, hi, reach);
    }
  } else if (nd[0] == CHANCE) {
    i32 K = nd[2];
    const f32 *masks = PF(nd[4]);
    if (!below(nd[3])) return;   /* (a river dealt by buckets) */
    f32 *c = salloc(L, sizeof(f32) * (size_t)(hi - lo) * K * n);
    for (i32 b = lo; b < hi; b++)
      for (i32 k = 0; k < K; k++) {
        size_t row = ((size_t)(b - lo) * K + k) * n, at = ((size_t)b * K + k) * n, rb = (size_t)(b - lo) * n;
        for (i32 i = 0; i < n; i++) c[row + i] = ro[rb + i] * masks[at + i];
      }
    down(L, nd[3], p, lo * K, hi * K, c);
  } else if (nd[0] == CLASSES) {
    i32 K = nd[2];
    if (!below(nd[3])) return;
    f32 *c = salloc(L, sizeof(f32) * (size_t)(hi - lo) * K * n), *rp = salloc(L, sizeof(f32) * (n + 1));
    spread(nd, lo, hi, ro, c, rp);
    down(L, nd[3], p, lo * K, hi * K, c);
  } else if (nd[0] == LEAF && nd[7] >= 0) {
    i32 *st = PI(P[11]) + 10 * nd[7], R = st[3], j = nd[8];
    i32 lq = trim ? P[16 - p] : n;
    const i32 *live = PI(P[14 - p]);
    f32 *X = PF(st[4]);
    for (i32 b = lo; b < hi; b++) {
      const f32 *r = ro + (size_t)(b - lo) * n;
      f32 *x = X + ((size_t)b * R + j) * lq;
      if (trim)
        for (i32 i = 0; i < lq; i++) x[i] = r[live[i]];
      else
        for (i32 i = 0; i < n; i++) x[i] = r[i];
    }
    if (j < st[6]) st[6] = j;
    if (j >= st[7]) st[7] = j + 1;
  }
  L->top = mark;
}

/* before a pass from node `at` (rows 0..rows): the batched leaves' products for the opponent's reach ro there */
static void batch(i32 at, int p, i32 rows, const f32 *ro) {
  i32 *st = PI(P[11]);
  if (!P[10]) return;
  trim = P[12] && mode == TRAIN;
  for (i32 s = 0; s < P[10]; s++) st[10 * s + 6] = st[10 * s + 3], st[10 * s + 7] = 0;
  down(&M, at, p, 0, rows, ro);
  i32 l0 = P[15], l1 = P[16], lp = trim ? P[15 + p] : n, lq = trim ? P[16 - p] : n;
  for (i32 s = 0; s < P[10]; s++, st += 10) {
    i32 S = st[2], R = st[3], jlo = st[6], jhi = st[7];
    f32 *X = PF(st[4]), *U = PF(st[5]);
    for (i32 m = 0; m < S && jlo < jhi; m++) {
      size_t row = (size_t)m * R + jlo;
      if (!trim)
        gemm_nt(X + row * n, addr(st) + (size_t)m * n * n, U + row * n, jhi - jlo, n, n);
      else if (p == 0)
        gemm_nt(X + row * lq, addr(st + 8) + (size_t)m * l0 * l1, U + row * lp, jhi - jlo, l0, l1);
      else
        gemm_nn_neg(X + row * lq, addr(st + 8) + (size_t)m * l0 * l1, U + row * lp, jhi - jlo, l0, l1);
    }
  }
}

#endif

/* The deferred games on a row play their terminals' products together: a walk down keeps each decision's strategies
   and each terminal's opponent reach, then each row's table (E or kap) multiplies all of the reaches that read it at
   once (read once, not once a terminal; tmatvec's sums, without its skipped zeros: they add nothing), and the walk back
   up takes the utilities and updates as decision() does. (Over GMAX terminals or decisions: played as cfr() would.) */
#define GMAX 64
typedef struct {
  i32 on, nt, nd, over, tc, dc;
  const i32 *t[GMAX];        /* each terminal's record, its opponent reach (rows, K), its products */
  const f32 *ro[GMAX];
  f32 *u[GMAX], *cs[GMAX];
  f32 *s[GMAX];              /* each decision's strategies */
} Game;

static void gdown(Lane *L, Game *G, i32 at, int p, i32 lo, i32 hi, const f32 *ro) {
  const i32 *nd = P + at;
  if (nd[0] != DECISION) {
    if (G->nt == GMAX) {
      G->over = 1;
      return;
    }
    G->t[G->nt] = nd, G->ro[G->nt++] = ro;
    return;
  }
  i32 player = nd[1], A = nd[2], W = nd[4];
  size_t RW = (size_t)(hi - lo) * W;
  if (G->nd == GMAX) {
    G->over = 1;
    return;
  }
  f32 *s = G->s[G->nd++] = salloc(L, sizeof(f32) * RW * A);
  strategy(L, PF(nd[5]), lo, hi, A, W, 0, s);
  for (i32 a = 0; a < A; a++) {
    if (player == p) {
      gdown(L, G, nd[7 + a], p, lo, hi, ro);
      continue;
    }
    f32 *reach = salloc(L, sizeof(f32) * RW);
    for (i32 b = lo; b < hi; b++) {
      size_t j = (size_t)(b - lo);
      const f32 *sa = s + (j * A + a) * W, *rb = ro + j * W;
      f32 *x = reach + j * W;
      for (i32 i = 0; i < W; i++) x[i] = rb[i] * sa[i];
    }
    gdown(L, G, nd[7 + a], p, lo, hi, reach);
  }
}

/* every game's terminals' products on row b, those reading the same table (E or kap) together, over each reach's
   nonzero entries */
static void gprod(Lane *L, Game *G, i32 ng, int p, i32 b) {
  size_t mark = L->top;
  i32 m = 0;
  for (i32 g = 0; g < ng; g++) m += 2 * G[g].nt;
  const f32 **tab = salloc(L, sizeof(f32 *) * m), **x = salloc(L, sizeof(f32 *) * m), **keys = salloc(L, sizeof(f32 *) * m);
  f32 **y = salloc(L, sizeof(f32 *) * m);
  i32 *key = salloc(L, sizeof(i32) * m), K = 0, nk = 0;
  m = 0;
  for (i32 g = 0; g < ng; g++)
    for (i32 t = 0; t < G[g].nt; t++) {
      const i32 *nd = G[g].t[t];
      i32 T = nd[3];
      size_t KK = (size_t)nd[2] * nd[2], at = (size_t)p * T * KK + (size_t)(b % T) * KK;
      K = nd[2];
      G[g].u[t] = salloc(L, sizeof(f32) * K), G[g].cs[t] = 0;
      tab[m] = PF(nd[4]) + at, x[m] = G[g].ro[t], y[m++] = G[g].u[t];
      if (nd[0] == BLEAF && fbits(nd[7 + 2 * p]) != 0) {
        G[g].cs[t] = salloc(L, sizeof(f32) * K);
        tab[m] = PF(nd[5]) + at, x[m] = G[g].ro[t], y[m++] = G[g].cs[t];
      }
    }
  for (i32 i = 0; i < m; i++) {
    i32 k = 0;
    while (k < nk && keys[k] != tab[i]) k++;
    if (k == nk) keys[nk++] = tab[i];
    key[i] = k;
  }
  i32 *nz = salloc(L, sizeof(i32) * K);
  f32 *v = salloc(L, sizeof(f32) * K);
  for (i32 k = 0; k < nk; k++)
    for (i32 i = 0; i < m; i++) {
      if (key[i] != k) continue;
      i32 c = 0;
      for (i32 j = 0; j < K; j++) {
        nz[c] = j, v[c] = x[i][j];
        c += x[i][j] != 0;
      }
      smatvec(keys[k], nz, v, c, y[i], K);
    }
  L->top = mark;
}

/* the walk back up: p's utilities (rows lo..hi, W) at node `at`, updating p's regrets and strategy sums (rp: p's
   reach, the game's: p's own actions don't change it) */
static void gup(Lane *L, Game *G, i32 at, int p, i32 lo, i32 hi, const f32 *rp, f32 *out) {
  const i32 *nd = P + at;
  if (nd[0] != DECISION) {
    i32 t = G->tc++, K = nd[2];
    const f32 *u = G->u[t], *cs = G->cs[t];
    size_t RK = (size_t)(hi - lo) * K;
    if (nd[0] == BFOLD) {
      f32 f = fbits(nd[5 + p]);
      for (size_t i = 0; i < RK; i++) out[i] = f * u[i];
    } else if (cs) {
      f32 a = fbits(nd[6 + 2 * p]), c = fbits(nd[7 + 2 * p]);
      for (size_t i = 0; i < RK; i++) out[i] = a * u[i] + c * cs[i];
    } else {
      f32 a = fbits(nd[6 + 2 * p]);
      for (size_t i = 0; i < RK; i++) out[i] = a * u[i];
    }
    return;
  }
  i32 player = nd[1], A = nd[2], W = nd[4];
  f32 *regret = PF(nd[5]), *ssum = PF(nd[6]);
  const i32 *kids = nd + 7;
  size_t RW = (size_t)(hi - lo) * W, mark = L->top;
  const f32 *s = G->s[G->dc++];
  if (player != p) {
    f32 *u = salloc(L, sizeof(f32) * RW);
    for (i32 a = 0; a < A; a++) {
      gup(L, G, kids[a], p, lo, hi, rp, a == 0 ? out : u);
      if (a > 0)
        for (size_t x = 0; x < RW; x++) out[x] += u[x];
    }
    L->top = mark;
    return;
  }
  f32 *ut = salloc(L, sizeof(f32) * RW * A);
  for (i32 a = 0; a < A; a++) gup(L, G, kids[a], p, lo, hi, rp, ut + a * RW);
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
  L->top = mark;
}

/* asks for row b of every decision's regrets (and p's strategy sums) in the game below `at` ahead of its walks: the
   games' decisions are hundreds of streams, more than the hardware follows */
static void ahead(i32 at, i32 b, int p) {
  const i32 *nd = P + at;
  if (nd[0] != DECISION || b >= nd[3]) return;
  i32 A = nd[2], W = nd[4];
  size_t len = sizeof(f32) * A * W;
  const char *r = (const char *)(PF(nd[5]) + (size_t)b * A * W), *s = (const char *)(PF(nd[6]) + (size_t)b * A * W);
  if (nd[1] == p)
    for (size_t x = 0; x < len; x += 128) __builtin_prefetch(r + x, 1, 3), __builtin_prefetch(s + x, 1, 3);
  else
    for (size_t x = 0; x < len; x += 128) __builtin_prefetch(r + x, 0, 3);
  for (i32 a = 0; a < A; a++) ahead(nd[7 + a], b, p);
}

/* the deferred bucket games on rows lo..hi (a game's rows: below its B), a row at a time: every game walked down on
   the row, their terminals' products, every game walked back up */
static void games(Lane *L, int p, i32 lo, i32 hi) {
  const i32 *at = PI(P[18]);
  i32 ng = P[17];
  for (i32 b = lo; b < hi; b++) {
    for (i32 x = 0; x < ng && b + 2 < hi; x++) ahead(P[at[x] + 4], b + 2, p);
    size_t mark = L->top;
    Game *G = salloc(L, sizeof(Game) * ng);
    for (i32 x = 0; x < ng; x++) {
      const i32 *nd = P + at[x];
      i32 B = nd[1], K = nd[2];
      f32 *g = PF(nd[6]);
      G[x].nt = G[x].nd = G[x].over = G[x].tc = G[x].dc = 0, G[x].on = b < B;
      if (!G[x].on) continue;
      gdown(L, &G[x], nd[4], p, b, b + 1, g + ((size_t)(p == 0 ? B : 0) + b) * K);
      if (G[x].over || !G[x].nt) {
        cfr(L, nd[4], p, b, b + 1, g + (size_t)b * K, g + ((size_t)B + b) * K, g + ((size_t)2 * B + b) * K);
        G[x].nt = G[x].on = 0;
      }
    }
    gprod(L, G, ng, p, b);
    for (i32 x = 0; x < ng; x++) {
      const i32 *nd = P + at[x];
      i32 B = nd[1], K = nd[2];
      f32 *g = PF(nd[6]);
      if (G[x].on) gup(L, &G[x], nd[4], p, b, b + 1, g + ((size_t)(p == 0 ? 0 : B) + b) * K, g + ((size_t)2 * B + b) * K);
    }
    L->top = mark;
  }
}

/* a job's rows lo..hi (from row r of its arrays): a chance node's child, or (child -1) the deferred games */
static void run(Lane *L, i32 child, int p, i32 lo, i32 hi, i32 r, const f32 *c0, const f32 *c1, f32 *u) {
  if (DEFER && child < 0) {
    games(L, p, lo, hi);
  } else if (phase != WHOLE) {
    /* (a deferred pass's walks a row at a time: a card's dealt rivers' masks and buckets, read under every line, stay
       in cache) */
    for (i32 x = lo; x < hi; x++) {
      size_t o = (size_t)(r + x - lo) * n;
      cfr(L, child, p, x, x + 1, c0 + o, c1 + o, u + o);
    }
  } else {
    cfr(L, child, p, lo, hi, c0 + (size_t)r * n, c1 + (size_t)r * n, u + (size_t)r * n);
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
    run(L, child, p, lo + r, lo + e, r, c0, c1, u);
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
    lane_size[0] = lane_size[1] = measured[0] = measured[1] = 0;   /* (the chunks change with the number) */
  }
  return nthreads;
}

/* a chance node's child on rows lo..hi: the first chance node on the way down (not already in a job's rows) runs it
   in chunks, one chunk at a time while measuring the scratch a chunk takes (an iteration, serially, once a program),
   then as a job */
static void split(Lane *L, i32 child, int p, i32 lo, i32 hi, const f32 *c0, const f32 *c1, f32 *u) {
  i32 rows = hi - lo;
  if (L->par || nthreads < 2 || rows < 2) {
    run(L, child, p, lo, hi, 0, c0, c1, u);
    return;
  }
  i32 grain = (rows + 2 * nthreads - 1) / (2 * nthreads);
  if (!lane_size[deferring]) {
    L->par = 1;
    for (i32 r = 0; r < rows; r += grain) {
      i32 e = rows - r < grain ? rows : r + grain;
      size_t mark = L->top;
      L->peak = mark;
      run(L, child, p, lo + r, lo + e, r, c0, c1, u);
      if (L->peak - mark > measured[deferring]) measured[deferring] = L->peak - mark;
      L->top = mark;
    }
    L->par = 0;
    return;
  }
#ifdef THREADS
  i32 helpers = nthreads - 1;
  size_t mark = L->top;
  unsigned char *lanes = salloc(L, lane_size[deferring] * helpers);
  u32 seq = (u32)(__atomic_load_n(&J.ticket, __ATOMIC_RELAXED) >> 32) + 2;
  __atomic_store_n(&J.ticket, ((u64)(seq - 1) << 32) | 0xffffffffu, __ATOMIC_SEQ_CST);
  ST(J.child, child), ST(J.p, p), ST(J.lo, lo), ST(J.rows, rows), ST(J.grain, grain), ST(J.helpers, helpers);
  ST(J.c0, c0), ST(J.c1, c1), ST(J.u, u), ST(J.lanes, (size_t)(lanes - base)), ST(J.lane, lane_size[deferring]);
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
   the root, whose chance nodes the lanes are measured on, serially the first time; below it, serially.) */
EXPORT(core_eval) i32 core_eval(u32 prog, i32 at, i32 p, u32 r0, u32 r1, u32 out, i32 md) {
  load(prog);
  if (P[0] != MAGIC) return -1;
  mode = md;
  M.par = at != P[4];
  batch(at, p, rows_of(at), PF(p == 0 ? r1 : r0));
  cfr(&M, at, p, 0, rows_of(at), PF(r0), PF(r1), PF(out));
  M.par = 0;
  sized();
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
  i32 root = P[4], B = rows_of(root), GB = 0;
  const f32 *r0 = PF(P[5]), *r1 = PF(P[6]);
  for (i32 x = 0; x < P[17]; x++) {
    i32 b = P[PI(P[18])[x] + 1];
    GB = b > GB ? b : GB;
  }
  f32 *regret = PF(P[7]), *ssum = PF(P[8]);
  size_t len = (u32)P[9];
  const f32 *fac = PF(factors);
  for (i32 t = 0; t < iters; t++) {
    for (int p = 0; p < 2; p++) {
      size_t mark = M.top;
      f32 *u = salloc(&M, sizeof(f32) * B * n);
      batch(root, p, B, p == 0 ? r1 : r0);
      if (DEFER && P[17]) {
        deferring = 1;
        phase = COLLECT;
        cfr(&M, root, p, 0, B, r0, r1, u);
        phase = WHOLE;
        split(&M, -1, p, 0, GB, 0, 0, 0);
        phase = READ;
        cfr(&M, root, p, 0, B, r0, r1, u);
        phase = WHOLE;
        deferring = 0;
      } else {
        cfr(&M, root, p, 0, B, r0, r1, u);
      }
      M.top = mark;
    }
    sized();
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

/* ------------------------------------------------------------------ a flop solve's tables */
/* Ctx._avg_showdown, Ctx.bucket_table and Ctx.river_table for one board (gto/core.py's Tables), built before a
   program is laid out, in the arena while it's empty. A table's arguments are int32 words at `args` (arrays as
   offsets):
     [1, n, R, S, ca, cb, L]                                       _avg_showdown: L (n, n)
     [2, n, R, S, ca, cb, K, w0, w1, ids, E, kap]                  bucket_table: ids (2, n), E and kap (2, K, K)
     [3, n, R, S, ca, cb, K, r0, r1, outer, tids, X, E, kap]       river_table: X (2, outer or n, K)
   S (R, n): the combos' strengths on the R boards (the turn's rivers; -1: holds a board card); ca, cb: each combo's
   cards; w0, w1, r0, r1 (n) float64: each player's root reach on the combos holding no turn card; tids (2, n): the
   turn buckets (bucket_table's ids, outer of them); E, kap, X float32. */
typedef double f64;
typedef short v8s __attribute__((vector_size(16)));
#define PD(off) ((f64 *)(base + (u32)(off)))

/* idx[0..m) sorted by key, ties kept in order (numpy's stable argsort); tmp: m ints */
static void t_sort(i32 *idx, i32 *tmp, const f64 *key, i32 m) {
  for (i32 w = 1; w < m; w *= 2) {
    for (i32 lo = 0; lo < m; lo += 2 * w) {
      i32 mid = lo + w < m ? lo + w : m, hi = lo + 2 * w < m ? lo + 2 * w : m, a = lo, b = mid, o = lo;
      while (a < mid && b < hi) tmp[o++] = key[idx[b]] < key[idx[a]] ? idx[b++] : idx[a++];
      while (a < mid) tmp[o++] = idx[a++];
      while (b < hi) tmp[o++] = idx[b++];
    }
    for (i32 x = 0; x < m; x++) idx[x] = tmp[x];
  }
}

static inline int t_compat(const i32 *A, const i32 *Bc, i32 i, i32 j) {
  return A[i] != A[j] && A[i] != Bc[j] && Bc[i] != A[j] && Bc[i] != Bc[j];
}

/* Ctx._avg_showdown: L[i, j] = (boards where i beats j - boards where j beats i) / max(boards where neither holds a
   board card, 1), times their compatibility; the counts are exact, so the floats are numpy's. Counted in int16, eight
   columns at a time, rows a block at a time (the block's counts stay in cache over the boards). */
static void t_showdown(i32 nn, i32 R, const i32 *S, const i32 *A, const i32 *Bc, f32 *L) {
  size_t mark = M.top;
  i32 np = (nn + 7) & ~7, RB = 8;
  short *s = salloc(&M, 2 * (size_t)R * np), *v = salloc(&M, 2 * (size_t)R * np);
  short *D = salloc(&M, 2 * (size_t)RB * np), *C = salloc(&M, 2 * (size_t)RB * np);
  for (i32 r = 0; r < R; r++)
    for (i32 j = 0; j < np; j++) {
      i32 x = j < nn ? S[(size_t)r * nn + j] : -1;
      s[(size_t)r * np + j] = (short)x;
      v[(size_t)r * np + j] = x >= 0 ? -1 : 0;   /* (-1: counted) */
    }
  for (i32 i0 = 0; i0 < nn; i0 += RB) {
    i32 i1 = i0 + RB < nn ? i0 + RB : nn;
    for (size_t x = 0; x < (size_t)RB * np; x++) D[x] = C[x] = 0;
    for (i32 r = 0; r < R; r++) {
      const v8s *sr = (const v8s *)(s + (size_t)r * np), *vr = (const v8s *)(v + (size_t)r * np);
      for (i32 i = i0; i < i1; i++) {
        short si = s[(size_t)r * np + i];
        if (si < 0) continue;
        v8s sv = {si, si, si, si, si, si, si, si};
        v8s *d = (v8s *)(D + (size_t)(i - i0) * np), *c = (v8s *)(C + (size_t)(i - i0) * np);
        for (i32 k = 0; k < np / 8; k++) {
          v8s a = sr[k], m = vr[k];
          d[k] += ((a > sv) - (a < sv)) & m;   /* +1 where i is stronger, -1 weaker (comparisons are -1 / 0) */
          c[k] -= m;
        }
      }
    }
    for (i32 i = i0; i < i1; i++) {
      const short *d = D + (size_t)(i - i0) * np, *c = C + (size_t)(i - i0) * np;
      f32 *Li = L + (size_t)i * nn;
      for (i32 j = 0; j < nn; j++)
        Li[j] = (f32)d[j] / (f32)(c[j] > 1 ? c[j] : 1) * (f32)t_compat(A, Bc, i, j);
    }
  }
  M.top = mark;
}

/* each combo's bucket (K: none) from its equity key eq (solver.eq_key: the equity to 2^-20, ties in combo order), among
   the combos with reach w > 0: K equal shares of the reach in order of equity, a combo in the share its reach's
   midpoint falls in (Ctx.bucket_table, river_table) */
static void t_bucket(i32 nn, const f64 *eq, const f64 *w, i32 K, i32 *ids, i32 *idx, i32 *tmp) {
  i32 m = 0;
  for (i32 i = 0; i < nn; i++) {
    ids[i] = K;
    if (w[i] > 0) idx[m++] = i;
  }
  t_sort(idx, tmp, eq, m);
  f64 tot = 0;
  for (i32 x = 0; x < m; x++) tot += w[idx[x]];
  tot = tot > 1e-12 ? tot : 1e-12;
  f64 cum = 0;
  for (i32 x = 0; x < m; x++) {
    f64 ws = w[idx[x]];
    cum += ws;
    i32 b = (i32)((cum - ws / 2) / tot * K);
    ids[idx[x]] = b < K - 1 ? b : K - 1;
  }
}

static void t_buckets(const i32 *a) {
  i32 nn = a[1], R = a[2], K = a[6];
  const i32 *S = PI(a[3]), *A = PI(a[4]), *Bc = PI(a[5]);
  const f64 *w[2] = {PD(a[7]), PD(a[8])};
  i32 *ids = PI(a[9]);
  f32 *E = PF(a[10]), *kap = PF(a[11]);
  size_t mark = M.top, KK = (size_t)K * K;
  f32 *L = salloc(&M, sizeof(f32) * (size_t)nn * nn);
  f64 *eq = salloc(&M, sizeof(f64) * nn), *acc = salloc(&M, sizeof(f64) * K);
  f64 *W[2] = {salloc(&M, sizeof(f64) * K), salloc(&M, sizeof(f64) * K)};
  f64 *num = salloc(&M, sizeof(f64) * KK), *pm = salloc(&M, sizeof(f64) * KK), *c1 = salloc(&M, sizeof(f64) * 52 * K);
  i32 *idx = salloc(&M, sizeof(i32) * nn), *tmp = salloc(&M, sizeof(i32) * nn);
  t_showdown(nn, R, S, A, Bc, L);
  /* each player's buckets, by equity against the opponent's root range: (L @ w) / its compatible mass */
  for (int q = 0; q < 2; q++) {
    const f64 *wo = w[1 - q];
    f64 tot = 0, card[52] = {0};
    for (i32 j = 0; j < nn; j++) {
      tot += wo[j];
      card[A[j]] += wo[j];
      card[Bc[j]] += wo[j];
    }
    for (i32 i = 0; i < nn; i++) {
      if (!(w[q][i] > 0)) continue;
      const f32 *Li = L + (size_t)i * nn;
      f64 x = 0;
      for (i32 j = 0; j < nn; j++) x += (f64)Li[j] * wo[j];
      f64 cm = tot - card[A[i]] - card[Bc[i]] + wo[i];
      eq[i] = __builtin_rint(x / (cm > 1e-12 ? cm : 1e-12) * 1048576.0);
    }
    t_bucket(nn, eq, w[q], K, ids + (size_t)q * nn, idx, tmp);
  }
  const i32 *id0 = ids, *id1 = ids + nn;
  for (int q = 0; q < 2; q++) {
    for (i32 k = 0; k < K; k++) W[q][k] = 0;
    for (i32 i = 0; i < nn; i++)
      if (ids[(size_t)q * nn + i] < K) W[q][ids[(size_t)q * nn + i]] += w[q][i];
  }
  /* the bucket pairs' sums of L and of the compatible pairs, weighted by the root reach */
  for (size_t x = 0; x < KK; x++) num[x] = pm[x] = 0;
  for (size_t x = 0; x < 52 * (size_t)K; x++) c1[x] = 0;
  for (i32 j = 0; j < nn; j++)
    if (id1[j] < K) {
      c1[(size_t)A[j] * K + id1[j]] += w[1][j];
      c1[(size_t)Bc[j] * K + id1[j]] += w[1][j];
    }
  for (i32 i = 0; i < nn; i++) {
    if (id0[i] >= K) continue;
    const f32 *Li = L + (size_t)i * nn;
    for (i32 k = 0; k < K; k++) acc[k] = 0;
    for (i32 j = 0; j < nn; j++)
      if (id1[j] < K) acc[id1[j]] += (f64)Li[j] * w[1][j];
    f64 w0 = w[0][i], *nr = num + (size_t)id0[i] * K, *pr = pm + (size_t)id0[i] * K;
    const f64 *ca_ = c1 + (size_t)A[i] * K, *cb_ = c1 + (size_t)Bc[i] * K;
    for (i32 k = 0; k < K; k++) {
      nr[k] += w0 * acc[k];
      pr[k] += w0 * (W[1][k] - ca_[k] - cb_[k]);
    }
    if (id1[i] < K) pr[id1[i]] += w0 * w[1][i];
  }
  for (i32 x = 0; x < K; x++)
    for (i32 y = 0; y < K; y++) {
      f64 den = W[0][x] * W[1][y];
      f64 e = den > 0 ? num[(size_t)x * K + y] / den : 0, p = den > 0 ? pm[(size_t)x * K + y] / den : 0;
      E[(size_t)x * K + y] = (f32)e;
      E[KK + (size_t)y * K + x] = (f32)-e;
      kap[(size_t)x * K + y] = (f32)p;
      kap[KK + (size_t)y * K + x] = (f32)p;
    }
  M.top = mark;
}

/* one board's sweep of the combos in order of strength (ord[from..nn): the live ones, -1s before them) with each
   combo's reach x: u, the reach weaker less the reach stronger among the combos sharing no card with it (ties: none),
   and cm, the reach of those combos (Ranking.utility, Solver.compat_sum) */
static void t_sweep(const i32 *ord, i32 from, i32 nn, const i32 *s, const i32 *A, const i32 *Bc, const f64 *x, f64 *u,
                    f64 *cm) {
  f64 T = 0, wk = 0, ct[52] = {0}, cw[52] = {0}, gc[52] = {0};
  for (i32 t = from; t < nn; t++) {
    i32 j = ord[t];
    T += x[j];
    ct[A[j]] += x[j];
    ct[Bc[j]] += x[j];
  }
  for (i32 t = from; t < nn;) {
    i32 e = t;
    f64 g = 0;
    for (; e < nn && s[ord[e]] == s[ord[t]]; e++) {
      i32 j = ord[e];
      g += x[j];
      gc[A[j]] += x[j];
      gc[Bc[j]] += x[j];
    }
    /* weaker - stronger = 2 weaker + tied - all, each less its share among the combos holding either card */
    f64 base_ = 2 * wk + g - T;
    for (i32 h = t; h < e; h++) {
      i32 i = ord[h], a = A[i], b = Bc[i];
      u[i] = base_ - (2 * cw[a] + gc[a] - ct[a]) - (2 * cw[b] + gc[b] - ct[b]);
      cm[i] = T - ct[a] - ct[b] + x[i];
    }
    for (i32 h = t; h < e; h++) {
      i32 j = ord[h];
      cw[A[j]] += x[j];
      cw[Bc[j]] += x[j];
      gc[A[j]] = gc[Bc[j]] = 0;
    }
    wk += g;
    t = e;
  }
}

static void t_river(const i32 *a) {
  i32 nn = a[1], R = a[2], K = a[6], outer = a[9];
  const i32 *S = PI(a[3]), *A = PI(a[4]), *Bc = PI(a[5]), *tids = outer ? PI(a[10]) : 0;
  const f64 *rr[2] = {PD(a[7]), PD(a[8])};
  f32 *X = PF(a[11]), *E = PF(a[12]), *kap = PF(a[13]);
  size_t mark = M.top, KK = (size_t)K * K, nK = (size_t)nn * K, K52 = (size_t)52 * K;
  f64 *w[2] = {salloc(&M, sizeof(f64) * nn), salloc(&M, sizeof(f64) * nn)};
  f64 *u = salloc(&M, sizeof(f64) * nn), *cm = salloc(&M, sizeof(f64) * nn), *key = salloc(&M, sizeof(f64) * nn);
  i32 *ord = salloc(&M, sizeof(i32) * nn), *idx = salloc(&M, sizeof(i32) * nn), *tmp = salloc(&M, sizeof(i32) * nn);
  i32 *ids[2] = {salloc(&M, sizeof(i32) * nn), salloc(&M, sizeof(i32) * nn)};
  f64 *P[2] = {salloc(&M, sizeof(f64) * nK), salloc(&M, sizeof(f64) * nK)};
  f64 *num = salloc(&M, sizeof(f64) * KK), *den = salloc(&M, sizeof(f64) * KK);
  f64 *Tk = salloc(&M, sizeof(f64) * K), *wk = salloc(&M, sizeof(f64) * K), *gk = salloc(&M, sizeof(f64) * K);
  f64 *bk = salloc(&M, sizeof(f64) * K);
  f64 *ctk = salloc(&M, sizeof(f64) * K52), *cwk = salloc(&M, sizeof(f64) * K52), *gck = salloc(&M, sizeof(f64) * K52);
  for (int q = 0; q < 2; q++)
    for (size_t x = 0; x < nK; x++) P[q][x] = 0;
  for (size_t x = 0; x < KK; x++) num[x] = den[x] = 0;
  for (i32 r = 0; r < R; r++) {
    const i32 *s = S + (size_t)r * nn;
    for (int q = 0; q < 2; q++)
      for (i32 i = 0; i < nn; i++) w[q][i] = s[i] >= 0 ? rr[q][i] : 0;
    for (i32 i = 0; i < nn; i++) {
      ord[i] = i;
      key[i] = s[i];
    }
    t_sort(ord, tmp, key, nn);
    i32 from = 0;
    while (from < nn && s[ord[from]] < 0) from++;
    /* each player's river buckets: K equal shares of its reach by equity against the opponent's */
    for (int q = 0; q < 2; q++) {
      t_sweep(ord, from, nn, s, A, Bc, w[1 - q], u, cm);
      for (i32 i = 0; i < nn; i++)
        key[i] = s[i] >= 0 ? __builtin_rint(u[i] / (cm[i] > 1e-12 ? cm[i] : 1e-12) * 1048576.0) : 0;
      t_bucket(nn, key, w[q], K, ids[q], idx, tmp);
      for (i32 i = 0; i < nn; i++)
        if (ids[q][i] < K) P[q][(size_t)i * K + ids[q][i]] += 1;
    }
    /* the bucket pairs' showdown sums and compatible masses on this river: the sweep with one column of the
       opponent's reach per bucket */
    const i32 *id0 = ids[0], *id1 = ids[1];
    const f64 *w0 = w[0], *w1 = w[1];
    for (i32 k = 0; k < K; k++) Tk[k] = wk[k] = gk[k] = 0;
    for (size_t x = 0; x < K52; x++) ctk[x] = cwk[x] = gck[x] = 0;
    for (i32 t = from; t < nn; t++) {
      i32 j = ord[t], k = id1[j];
      if (k >= K) continue;
      Tk[k] += w1[j];
      ctk[(size_t)A[j] * K + k] += w1[j];
      ctk[(size_t)Bc[j] * K + k] += w1[j];
    }
    for (i32 t = from; t < nn;) {
      i32 e = t;
      for (; e < nn && s[ord[e]] == s[ord[t]]; e++) {
        i32 j = ord[e], k = id1[j];
        if (k >= K) continue;
        gk[k] += w1[j];
        gck[(size_t)A[j] * K + k] += w1[j];
        gck[(size_t)Bc[j] * K + k] += w1[j];
      }
      for (i32 k = 0; k < K; k++) bk[k] = 2 * wk[k] + gk[k] - Tk[k];
      for (i32 h = t; h < e; h++) {
        i32 i = ord[h];
        if (id0[i] >= K) continue;
        size_t oa = (size_t)A[i] * K, ob = (size_t)Bc[i] * K;
        f64 x0 = w0[i], *nr = num + (size_t)id0[i] * K, *dr = den + (size_t)id0[i] * K;
        for (i32 k = 0; k < K; k++) {
          nr[k] += x0 * (bk[k] - (2 * cwk[oa + k] + gck[oa + k] - ctk[oa + k]) -
                         (2 * cwk[ob + k] + gck[ob + k] - ctk[ob + k]));
          dr[k] += x0 * (Tk[k] - ctk[oa + k] - ctk[ob + k]);
        }
        if (id1[i] < K) dr[id1[i]] += x0 * w1[i];
      }
      for (i32 h = t; h < e; h++) {
        i32 j = ord[h], k = id1[j];
        if (k >= K) continue;
        cwk[(size_t)A[j] * K + k] += w1[j];
        cwk[(size_t)Bc[j] * K + k] += w1[j];
        gck[(size_t)A[j] * K + k] = gck[(size_t)Bc[j] * K + k] = 0;
      }
      for (i32 k = 0; k < K; k++) {
        wk[k] += gk[k];
        gk[k] = 0;
      }
      t = e;
    }
  }
  /* P[q]: the share of the rivers putting each combo in each bucket; A[q] = root reach x P[q] (in place) */
  for (int q = 0; q < 2; q++)
    for (i32 i = 0; i < nn; i++) {
      f64 *p = P[q] + (size_t)i * K, c = 0;
      for (i32 k = 0; k < K; k++) c += p[k];
      c = c > 1 ? c : 1;
      for (i32 k = 0; k < K; k++) p[k] /= c;
    }
  if (outer) {
    f64 *Q = salloc(&M, sizeof(f64) * (size_t)outer * K), *wt = salloc(&M, sizeof(f64) * outer);
    for (int q = 0; q < 2; q++) {
      const i32 *t = tids + (size_t)q * nn;
      for (size_t x = 0; x < (size_t)outer * K; x++) Q[x] = 0;
      for (i32 b = 0; b < outer; b++) wt[b] = 0;
      for (i32 i = 0; i < nn; i++) {
        if (t[i] >= outer) continue;
        wt[t[i]] += rr[q][i];
        for (i32 k = 0; k < K; k++) Q[(size_t)t[i] * K + k] += rr[q][i] * P[q][(size_t)i * K + k];
      }
      for (i32 b = 0; b < outer; b++)
        for (i32 k = 0; k < K; k++)
          X[((size_t)q * outer + b) * K + k] = (f32)(Q[(size_t)b * K + k] / (wt[b] > 1e-300 ? wt[b] : 1e-300));
    }
  } else {
    for (int q = 0; q < 2; q++)
      for (size_t x = 0; x < nK; x++) X[(size_t)q * nK + x] = (f32)P[q][x];
  }
  for (int q = 0; q < 2; q++)
    for (i32 i = 0; i < nn; i++)
      for (i32 k = 0; k < K; k++) P[q][(size_t)i * K + k] *= rr[q][i];
  /* kappa: the compatible share of the opponent's bucket mass as X spreads it, (A0.T @ compat @ A1) / (M0 M1) */
  f64 *Mq[2] = {Tk, wk}, *c1 = ctk, *row = bk;
  for (int q = 0; q < 2; q++) {
    for (i32 k = 0; k < K; k++) Mq[q][k] = 0;
    for (i32 i = 0; i < nn; i++)
      for (i32 k = 0; k < K; k++) Mq[q][k] += P[q][(size_t)i * K + k];
  }
  for (size_t x = 0; x < K52; x++) c1[x] = 0;
  for (i32 i = 0; i < nn; i++)
    for (i32 k = 0; k < K; k++) {
      c1[(size_t)A[i] * K + k] += P[1][(size_t)i * K + k];
      c1[(size_t)Bc[i] * K + k] += P[1][(size_t)i * K + k];
    }
  f64 *kn = salloc(&M, sizeof(f64) * KK);
  for (size_t x = 0; x < KK; x++) kn[x] = 0;
  for (i32 i = 0; i < nn; i++) {
    const f64 *a1 = P[1] + (size_t)i * K, *a0 = P[0] + (size_t)i * K;
    const f64 *ca_ = c1 + (size_t)A[i] * K, *cb_ = c1 + (size_t)Bc[i] * K;
    for (i32 k = 0; k < K; k++) row[k] = Mq[1][k] - ca_[k] - cb_[k] + a1[k];
    for (i32 x = 0; x < K; x++) {
      if (a0[x] == 0) continue;
      f64 *kr = kn + (size_t)x * K;
      for (i32 k = 0; k < K; k++) kr[k] += a0[x] * row[k];
    }
  }
  for (i32 x = 0; x < K; x++)
    for (i32 y = 0; y < K; y++) {
      size_t o = (size_t)x * K + y;
      f64 mm = Mq[0][x] * Mq[1][y], kp = mm > 0 ? kn[o] / mm : 0;
      f64 sd = den[o] > 0 ? num[o] / den[o] : 0, e = sd * kp;
      E[o] = (f32)e;
      E[KK + (size_t)y * K + x] = (f32)-e;
      kap[o] = (f32)kp;
      kap[KK + (size_t)y * K + x] = (f32)kp;
    }
  M.top = mark;
}

/* one table (the arguments above); 0, or -1 on a bad kind */
EXPORT(core_table) i32 core_table(u32 args) {
  const i32 *a = PI(args);
  switch (a[0]) {
    case 1: t_showdown(a[1], a[2], PI(a[3]), PI(a[4]), PI(a[5]), PF(a[6])); return 0;
    case 2: t_buckets(a); return 0;
    case 3: t_river(a); return 0;
  }
  return -1;
}
