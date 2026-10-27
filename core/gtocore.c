/* The compiled CFR core: whole training iterations of gto/solver.py's Solver.run (gto/core.py builds the program and
 * runs it). The same source builds natively (a shared library, loaded with ctypes) and for the browser (gto/core.wasm,
 * instantiated by gto/static/worker.js beside Pyodide), with no libc: everything lives in one arena, the inputs copied
 * in by gto/core.py and the traversal's scratch above them (allocated and released in stack order).
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
 */
#include <stddef.h>
#include <stdint.h>

typedef float f32;
typedef int32_t i32;
typedef uint32_t u32;

#ifdef __wasm__
#define EXPORT(name) __attribute__((export_name(#name)))
extern unsigned char __heap_base;
#else
#define EXPORT(name) __attribute__((visibility("default")))
#include <sys/mman.h>
#endif

#define MAGIC 0x67746f31   /* "gto1" */
enum { DECISION = 1, FOLD = 2, SHOWDOWN = 3, LEAF = 4, CHANCE = 5, CLASSES = 6 };

/* ------------------------------------------------------------------ arena */
static unsigned char *base;
static size_t top;

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

static void *salloc(size_t bytes) {
  size_t at = (top + 63) & ~(size_t)63;
  ensure(at + bytes);
  top = at + bytes;
  return base + at;
}

/* Frees the arena (natively its pages go back to the system). */
EXPORT(core_reset) void core_reset(void) {
#ifndef __wasm__
  if (base && touched > ((size_t)64 << 20)) {
    madvise(base, touched, MADV_FREE);
    touched = 0;
  }
#endif
  top = 0;
}

/* `bytes` in the arena; returns its offset. */
EXPORT(core_alloc) u32 core_alloc(u32 bytes) {
  unsigned char *p = salloc(bytes);
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

/* ro: one row of the opponent's reach -> out: the mass of opponent combos sharing no card with each combo
   (Solver.compat_sum: the total, minus each of the combo's cards' combos, plus the combo itself) */
static void compat(const f32 *ro, f32 *out) {
  f32 card[52];
  for (int c = 0; c < 52; c++) card[c] = 0;
  f32 tot = 0, run = 0;
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
    tot += ro[j];
  }
  card[cur] += run;
  for (i32 i = 0; i < n; i++) out[i] = tot - card[ca[i]] - card[cb[i]] + ro[i];
}

/* Ranking.utility for row b: sum over compatible opponent combos of reach * (+1 weaker, -1 stronger) */
static void ranking(const i32 *R, i32 b, const f32 *ro, f32 *u) {
  const i32 *order = PI(R[2]) + (size_t)b * n, *lo = PI(R[3]) + (size_t)b * n, *hi = PI(R[4]) + (size_t)b * n;
  const i32 *pcombo = PI(R[5]) + (size_t)b * 2 * n;
  const i32 *plo = PI(R[6]) + (size_t)b * 2 * n, *phi = PI(R[7]) + (size_t)b * 2 * n;
  const i32 *cstart = PI(R[8]), *cend = PI(R[9]);
  const f32 *live = PF(R[10]) + (size_t)b * n;
  size_t mark = top;
  f32 *cum = salloc(sizeof(f32) * (n + 1)), *cs = salloc(sizeof(f32) * (2 * n + 1));
  cum[0] = 0;
  for (i32 j = 0; j < n; j++) cum[j + 1] = cum[j] + ro[order[j]];
  cs[0] = 0;
  for (i32 j = 0; j < 2 * n; j++) cs[j + 1] = cs[j] + ro[pcombo[j]];
  f32 total = cum[n];
  for (i32 i = 0; i < n; i++) {
    f32 x = cum[lo[i]];
    x += cum[hi[i]];
    x += cs[cend[ca[i]]] + cs[cstart[ca[i]]];
    x += cs[cend[cb[i]]] + cs[cstart[cb[i]]];
    x -= cs[plo[i]];
    x -= cs[phi[i]];
    x -= cs[plo[n + i]];
    x -= cs[phi[n + i]];
    x -= total;
    u[i] = x * live[i];
  }
  top = mark;
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

/* p's utility (B, n) at a terminal; ro: the opponent's reach (B, n) */
static void terminal(const i32 *nd, int p, const f32 *ro, f32 *u) {
  i32 B = nd[1];
  size_t mark = top;
  f32 *cs = salloc(sizeof(f32) * n);
  if (nd[0] == FOLD) {
    f32 f = fbits(nd[2 + p]);
    for (i32 b = 0; b < B; b++) {
      compat(ro + (size_t)b * n, cs);
      for (i32 i = 0; i < n; i++) u[(size_t)b * n + i] = f * cs[i];
    }
  } else {
    f32 a = fbits(nd[3 + 2 * p]), c = fbits(nd[4 + 2 * p]);
    for (i32 b = 0; b < B; b++) {
      const f32 *r = ro + (size_t)b * n;
      f32 *ub = u + (size_t)b * n;
      if (nd[0] == SHOWDOWN)
        ranking(P + nd[2], b, r, ub);
      else
        matvec(PF(nd[2]) + (size_t)b * n * n, r, ub);
      if (c != 0) {
        compat(r, cs);
        for (i32 i = 0; i < n; i++) ub[i] = a * ub[i] + c * cs[i];
      } else {
        for (i32 i = 0; i < n; i++) ub[i] = a * ub[i];
      }
    }
  }
  top = mark;
}

/* Regret matching (Decision.strategy): (B, A, W) regrets -> strategies; with `avg`, the strategy sums -> the average
   strategy (Decision.avg_strategy). (Every loop runs over a row's combos innermost, so it vectorizes; the sums over
   actions still add them first to last.) */
static void strategy(const f32 *regret, i32 B, i32 A, i32 W, int avg, f32 *s) {
  f32 uni = (f32)(1.0 / A);
  size_t mark = top;
  f32 *tot = salloc(sizeof(f32) * W);
  for (i32 b = 0; b < B; b++) {
    const f32 *r = regret + (size_t)b * A * W;
    f32 *sb = s + (size_t)b * A * W;
    if (avg)
      for (i32 x = 0; x < A * W; x++) sb[x] = r[x];
    else
      for (i32 x = 0; x < A * W; x++) sb[x] = r[x] > 0 ? r[x] : 0;
    for (i32 i = 0; i < W; i++) tot[i] = sb[i];
    for (i32 a = 1; a < A; a++)
      for (i32 i = 0; i < W; i++) tot[i] += sb[a * W + i];
    for (i32 a = 0; a < A; a++)
      for (i32 i = 0; i < W; i++) sb[a * W + i] = tot[i] == 0 ? uni : sb[a * W + i] / tot[i];
  }
  top = mark;
}

static void cfr(i32 at, int p, const f32 *r0, const f32 *r1, f32 *out);

/* TRAIN: a training pass (Solver._cfr, update=True); AVG: both players on their average strategies, no update (use_avg);
   BR: p best-responds to the opponent's average strategy (Solver._best_response) */
enum { TRAIN = 0, AVG = 1, BR = 2 };
static int mode = TRAIN;

static int is_terminal(i32 at) {
  i32 t = P[at];
  return t == FOLD || t == SHOWDOWN || t == LEAF;
}

static void decision(const i32 *nd, int p, const f32 *r0, const f32 *r1, f32 *out) {
  i32 player = nd[1], A = nd[2], B = nd[3], W = nd[4];
  f32 *regret = PF(nd[5]), *ssum = PF(nd[6]);
  const i32 *kids = nd + 7;
  size_t BW = (size_t)B * W, mark = top;
  f32 *s = salloc(sizeof(f32) * BW * A);
  if (!(mode == BR && player == p)) strategy(mode == TRAIN ? regret : ssum, B, A, W, mode != TRAIN, s);
  if (player == p) {
    const f32 *ro = p == 0 ? r1 : r0, *rp = p == 0 ? r0 : r1;
    f32 *ut = salloc(sizeof(f32) * BW * A);   /* (A, B, W): each action's utilities contiguous */
    for (i32 a = 0; a < A; a++) {
      if (is_terminal(kids[a]))
        terminal(P + kids[a], p, ro, ut + a * BW);
      else
        cfr(kids[a], p, r0, r1, ut + a * BW);
    }
    if (mode == BR) {
      for (size_t x = 0; x < BW; x++) out[x] = ut[x];
      for (i32 a = 1; a < A; a++)
        for (size_t x = 0; x < BW; x++) out[x] = ut[a * BW + x] > out[x] ? ut[a * BW + x] : out[x];
      top = mark;
      return;
    }
    for (i32 b = 0; b < B; b++) {
      f32 *nu0 = out + (size_t)b * W;
      for (i32 a = 0; a < A; a++) {
        const f32 *sa = s + ((size_t)b * A + a) * W, *ua = ut + a * BW + (size_t)b * W;
        if (a == 0)
          for (i32 i = 0; i < W; i++) nu0[i] = sa[i] * ua[i];
        else
          for (i32 i = 0; i < W; i++) nu0[i] += sa[i] * ua[i];
      }
      if (mode != TRAIN) continue;
      for (i32 a = 0; a < A; a++) {
        f32 *rg = regret + ((size_t)b * A + a) * W, *ss = ssum + ((size_t)b * A + a) * W;
        const f32 *sa = s + ((size_t)b * A + a) * W, *ua = ut + a * BW + (size_t)b * W, *nu = out + (size_t)b * W;
        const f32 *rb = rp + (size_t)b * W;
        for (i32 i = 0; i < W; i++) {
          rg[i] += ua[i] - nu[i];
          ss[i] += rb[i] * sa[i];
        }
      }
    }
  } else {
    /* the opponent acts: each action's reach, and the utilities summed over actions in order */
    const f32 *ro = p == 0 ? r1 : r0;
    f32 *reach = salloc(sizeof(f32) * BW), *u = salloc(sizeof(f32) * BW);
    for (i32 a = 0; a < A; a++) {
      for (i32 b = 0; b < B; b++) {
        const f32 *sa = s + ((size_t)b * A + a) * W, *rb = ro + (size_t)b * W;
        f32 *x = reach + (size_t)b * W;
        for (i32 i = 0; i < W; i++) x[i] = rb[i] * sa[i];
      }
      f32 *dst = a == 0 ? out : u;
      if (is_terminal(kids[a]))
        terminal(P + kids[a], p, reach, dst);
      else if (p == 0)
        cfr(kids[a], p, r0, reach, dst);
      else
        cfr(kids[a], p, reach, r1, dst);
      if (a > 0)
        for (size_t x = 0; x < BW; x++) out[x] += u[x];
    }
  }
  top = mark;
}

static void chance(const i32 *nd, int p, const f32 *r0, const f32 *r1, f32 *out) {
  i32 B = nd[1], K = nd[2], child = nd[3];
  const f32 *masks = PF(nd[4]), *w = PF(nd[5]), *norm = PF(nd[6]);
  size_t mark = top, rows = (size_t)B * K;
  f32 *c0 = salloc(sizeof(f32) * rows * n), *c1 = salloc(sizeof(f32) * rows * n), *u = salloc(sizeof(f32) * rows * n);
  for (i32 b = 0; b < B; b++)
    for (i32 k = 0; k < K; k++) {
      size_t row = ((size_t)b * K + k) * n;
      for (i32 i = 0; i < n; i++) {
        c0[row + i] = r0[(size_t)b * n + i] * masks[row + i];
        c1[row + i] = r1[(size_t)b * n + i] * masks[row + i];
      }
    }
  cfr(child, p, c0, c1, u);
  for (i32 b = 0; b < B; b++) {
    f32 *o = out + (size_t)b * n;
    for (i32 i = 0; i < n; i++) o[i] = u[(size_t)b * K * n + i] * w[(size_t)b * K * n + i];
    for (i32 k = 1; k < K; k++) {
      size_t row = ((size_t)b * K + k) * n;
      for (i32 i = 0; i < n; i++) o[i] += u[row + i] * w[row + i];
    }
    for (i32 i = 0; i < n; i++) o[i] = o[i] / norm[(size_t)b * n + i];
  }
  top = mark;
}

/* river classes: Chance.spread into the class subtrees, Chance.gather back */
static void classes(const i32 *nd, int p, const f32 *r0, const f32 *r1, f32 *out) {
  i32 B = nd[1], K = nd[2], child = nd[3], nG = nd[10];
  const f32 *masks = PF(nd[4]), *norm = PF(nd[5]), *sizes = PF(nd[9]);
  const i32 *G = PI(nd[6]), *flat = PI(nd[7]), *starts = PI(nd[8]);
  size_t mark = top, rows = (size_t)B * K;
  f32 *c[2] = {salloc(sizeof(f32) * rows * n), salloc(sizeof(f32) * rows * n)};
  f32 *u = salloc(sizeof(f32) * rows * n), *rp = salloc(sizeof(f32) * (n + 1)), *um = salloc(sizeof(f32) * K * n);
  const f32 *r[2] = {r0, r1};
  for (int q = 0; q < 2; q++)
    for (i32 b = 0; b < B; b++) {
      for (i32 i = 0; i < n; i++) rp[i] = r[q][(size_t)b * n + i];
      rp[n] = 0;
      for (i32 k = 0; k < K; k++) {
        f32 *dst = c[q] + ((size_t)b * K + k) * n;
        i32 e0 = starts[k], e1 = k + 1 < K ? starts[k + 1] : nG;
        for (i32 i = 0; i < n; i++) dst[i] = rp[G[(size_t)e0 * n + i]];
        for (i32 e = e0 + 1; e < e1; e++)
          for (i32 i = 0; i < n; i++) dst[i] += rp[G[(size_t)e * n + i]];
        const f32 *m = masks + ((size_t)b * K + k) * n;
        for (i32 i = 0; i < n; i++) dst[i] *= m[i] / sizes[k];
      }
    }
  cfr(child, p, c[0], c[1], u);
  for (i32 b = 0; b < B; b++) {
    const f32 *ub = u + (size_t)b * K * n, *m = masks + (size_t)b * K * n;
    for (size_t x = 0; x < (size_t)K * n; x++) um[x] = ub[x] * m[x];
    f32 *o = out + (size_t)b * n;
    for (i32 i = 0; i < n; i++) o[i] = um[flat[i]];
    for (i32 e = 1; e < nG; e++)
      for (i32 i = 0; i < n; i++) o[i] += um[flat[(size_t)e * n + i]];
    for (i32 i = 0; i < n; i++) o[i] = o[i] / norm[(size_t)b * n + i];
  }
  top = mark;
}

static void cfr(i32 at, int p, const f32 *r0, const f32 *r1, f32 *out) {
  const i32 *nd = P + at;
  switch (nd[0]) {
    case DECISION: decision(nd, p, r0, r1, out); break;
    case CHANCE: chance(nd, p, r0, r1, out); break;
    case CLASSES: classes(nd, p, r0, r1, out); break;
    default: terminal(nd, p, p == 0 ? r1 : r0, out); break;
  }
}

static void load(u32 prog) {
  P = PI(prog);
  n = P[1];
  ca = PI(P[2]);
  cb = PI(P[3]);
}

/* p's utilities (rows, n) at the node record `at` for reaches r0, r1 (rows, n) each: md AVG on the average strategies,
   BR p's best response (Solver.node_action_evs, _best_response). Returns 0, or -1 on a bad program. */
EXPORT(core_eval) i32 core_eval(u32 prog, i32 at, i32 p, u32 r0, u32 r1, u32 out, i32 md) {
  load(prog);
  if (P[0] != MAGIC) return -1;
  mode = md;
  cfr(at, p, PF(r0), PF(r1), PF(out));
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
  i32 root = P[4];
  const f32 *r0 = PF(P[5]), *r1 = PF(P[6]);
  f32 *regret = PF(P[7]), *ssum = PF(P[8]);
  size_t len = (u32)P[9];
  const f32 *fac = PF(factors);
  for (i32 t = 0; t < iters; t++) {
    for (int p = 0; p < 2; p++) {
      size_t mark = top;
      f32 *u = salloc(sizeof(f32) * n);
      cfr(root, p, r0, r1, u);
      top = mark;
    }
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
