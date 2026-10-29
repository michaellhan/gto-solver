/* The multiway core: whole training iterations of gto/multiway_solver.py's 3-player (and 2-player) solves whose later
 * streets are bet on by hand buckets (MWConfig.later = "buckets": every flop and turn solve the advisor runs), and the
 * passes after a solve (each player's value on the average strategies, its best response, an action's EVs). gto/mwcore.py
 * lays the solver's tree out as a program and runs it; the same source builds natively (a shared library, ctypes) and
 * for the browser (gto/mwcore.wasm, instantiated by gto/static/worker.js beside Pyodide), with no libc: everything
 * lives in one arena, the inputs copied in by gto/mwcore.py and a pass's scratch above them (stack order).
 *
 * The passes are MultiwaySolver._pass's (modes alt, br; and avg for one player), on the same tree, in float32: the root
 * street per combo (its decisions, its folds, its Exits into the bucket world), and below it every root line's later
 * streets by hand buckets, one block of C rows (the next card's classes) of the merged subtrees (_Group) at a time,
 * walked depth first: a pass needs no stored reaches. A terminal's value is a contraction of the showdown tensors over
 * the other players' bucket reaches; below another player's fold (three players) the tensors are first summed over the
 * folded player's reach, which no longer changes (MultiwaySolver._bterms' reduce and apply); below the traverser's own
 * fold its value is minus its chips in times the deals' weight.
 *
 * Tensors are laid out per player p with p's buckets innermost, (C, K_o0, K_o1, K_p) for the others o0 < o1, so every
 * contraction is a sum of rows of K scaled by the others' reaches (skipped where a reach is 0: a line most of a range
 * never takes costs little).
 *
 *   header [MAGIC, np, n, K, C, root, ca, cb, masks, wmask, ids, tens, narr, arrs, allin, reach0, perm, seg, mkp,
 *           flags, nflag, pwc]
 *   RDEC  [1, q, A, regret, ssum, sig, flag, child_0 .. child_A-1]  root street: regret, ssum, sig (A, n)
 *   RFOLD [2, w, net, c_0 .. c_np-1]                             a fold leaving w alone: u_p = ([p == w] net - c_p) m_p
 *   REXIT [3, live, child, c_0 ..]                               the root street's end (Exit), the bucket subtree below
 *   BDEC  [4, q, A, regret, ssum, fold, leafc, sig, flag, child_0 ..]  regret, ssum, sig (C, A, K); fold: its index or -1
 *   BTERM [5, live, side, net, fm, c_0 ..]                       fm: the main pot's share (an all-in player), or -1
 *   BEXIT [6, child]                                             the next street's subtree
 * (c: each player's chips in, as floats; arrs: here one run of every decision's regrets and one of their sums)
 * tens: per player p, J's tensor at [p] and live set L's shares at [np + L * np + p] (offsets; 0 none); ids (np, C, n)
 * each combo's bucket; masks, wmask (C, n); arrs (narr, 3): every decision's regret, ssum and length (the discount);
 * perm (np, C, n) each player's combos per class in bucket order, seg (np, C, K + 1) each bucket's first, mkp (np, C,
 * n) the class's mask in that order; flags (nflag, one a decision record): 1 while its sig is its regrets' current
 * strategy (computed at its first use after the discount or its player's pass, so a pass's strategies are the numpy
 * solver's, from the regrets as they stand), 2 while it is the average strategy (the evaluations'), 0 stale.
 * pwc (np, n): four players, each player's pairwise overlap correction (MultiwaySolver._pw_corr: the root street's
 * deals' weight is pairwise, _pw_masses); 0 with two or three (exact).
 *
 * Four players (nobody all-in) contract (C, K_o0, K_o1, K_o2, K_p) tensors over three reaches, with no fold reductions.
 */
#include <stddef.h>
#include <stdint.h>

typedef float f32;
typedef double f64;
typedef int32_t i32;
typedef uint32_t u32;

#ifdef __wasm__
#define EXPORT(name) __attribute__((export_name(#name)))
extern unsigned char __heap_base;
#else
#define EXPORT(name) __attribute__((visibility("default")))
#include <sys/mman.h>
#endif

#define MAGIC 0x6d776331 /* "mwc1" */
enum { RDEC = 1, RFOLD = 2, REXIT = 3, BDEC = 4, BTERM = 5, BEXIT = 6 };
enum { ALT = 0, BR = 1, EV = 2 };

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
#define RESERVE ((size_t)1 << 34) /* 16 GB of address space, committed as it's touched */
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

EXPORT(mw_reset) void mw_reset(void) {
#ifndef __wasm__
  if (base && touched > ((size_t)64 << 20)) {
    madvise(base, touched, MADV_FREE);
    touched = 0;
  }
#endif
  top = 0;
}

EXPORT(mw_alloc) u32 mw_alloc(u32 bytes) { return (u32)((unsigned char *)salloc(bytes) - base); }

EXPORT(mw_base) uintptr_t mw_base(void) {
  ensure(0);
  return (uintptr_t)base;
}

EXPORT(mw_version) i32 mw_version(void) { return MAGIC; }

#ifndef CORE_SOURCE
#define CORE_SOURCE "dev"
#endif
static const char source[] = CORE_SOURCE;
EXPORT(mw_source) const char *mw_source(void) { return source; }

/* ------------------------------------------------------------------ program */
static const i32 *W;
static int NP, N, K, C, ALLIN;
static const i32 *CA, *CB, *IDS, *PERM, *SEG;
static const f32 *MASKS, *WMASK, *MKP, *PWC;
static i32 *FLAGS;
static int NFLAG;
static const u32 *TENS;

#define PF(off) ((f32 *)(base + (u32)(off)))
#define PI(off) ((i32 *)(base + (u32)(off)))
static inline f32 fbits(i32 w) {
  union { i32 i; f32 f; } u;
  u.i = w;
  return u.f;
}
static inline int pop(int m) { return __builtin_popcount((unsigned)m); }

static void load(u32 prog) {
  W = PI(prog);
  NP = W[1];
  N = W[2];
  K = W[3];
  C = W[4];
  CA = PI(W[6]);
  CB = PI(W[7]);
  MASKS = PF(W[8]);
  WMASK = PF(W[9]);
  IDS = PI(W[10]);
  TENS = (const u32 *)PI(W[11]);
  ALLIN = W[14];
  PERM = PI(W[16]);
  SEG = PI(W[17]);
  MKP = PF(W[18]);
  FLAGS = PI(W[19]);
  NFLAG = W[20];
  PWC = W[21] ? PF(W[21]) : 0;
}

static const f32 *tensor(int key, int p) { /* key: -1 J, else a live set's mask */
  u32 off = TENS[key < 0 ? p : NP + key * NP + p];
  return off ? PF(off) : 0;
}

/* ------------------------------------------------------------------ kernels (k: K, a compile-time constant where it
   matters: the rows of K stay in registers) */
#define INL static inline __attribute__((always_inline))

/* the nonzero entries of x (k): their indices (branch-free), how many */
INL int nzlist(const f32 *x, int *idx, const int k) {
  int m = 0;
  for (int a = 0; a < k; a++) {
    idx[m] = a;
    m += x[a] != 0;
  }
  return m;
}

/* One row of a contraction over two others' reaches: v (k) = sum_{a, b} x0[a] x1[b] (w0 T0 + w1 T1 + w2 T2)[a][b]
   (T_j rows of the row's (k, k, k) block; up to three tensors with weights: J and the pots' shares). The reaches'
   zeros are skipped (a line most of a range never takes costs little); each tensor sums on its own accumulators, two
   for J alone, so the multiply-adds don't wait on each other. */
INL void con3_row(const f32 *restrict T0, const f32 *restrict T1, const f32 *restrict T2, f32 w0, f32 w1, f32 w2,
                  const f32 *restrict x0, const f32 *restrict x1, f32 *restrict v, const int k) {
  int i0[32], i1[32];
  int m0 = nzlist(x0, i0, k), m1 = nzlist(x1, i1, k);
  f32 c1[32];
  for (int e = 0; e < m1; e++) c1[e] = x1[i1[e]];
  f32 A0[32], A1[32], A2[32];
  for (int j = 0; j < k; j++) A0[j] = A1[j] = A2[j] = 0;
  if (T1 == 0) {
    for (int d = 0; d < m0; d++) {
      int a = i0[d];
      f32 c0 = x0[a];
      const f32 *ta = T0 + (size_t)a * k * k;
      int e = 0;
      for (; e + 1 < m1; e += 2) {
        f32 ca = c0 * c1[e], cb = c0 * c1[e + 1];
        const f32 *ra = ta + (size_t)i1[e] * k, *rb = ta + (size_t)i1[e + 1] * k;
        for (int j = 0; j < k; j++) {
          A0[j] += ca * ra[j];
          A1[j] += cb * rb[j];
        }
      }
      if (e < m1) {
        f32 ca = c0 * c1[e];
        const f32 *ra = ta + (size_t)i1[e] * k;
        for (int j = 0; j < k; j++) A0[j] += ca * ra[j];
      }
    }
    for (int j = 0; j < k; j++) v[j] = w0 * (A0[j] + A1[j]);
    return;
  }
  for (int d = 0; d < m0; d++) {
    int a = i0[d];
    f32 c0 = x0[a];
    size_t oa = (size_t)a * k * k;
    if (T2 == 0) {
      for (int e = 0; e < m1; e++) {
        f32 c = c0 * c1[e];
        size_t o = oa + (size_t)i1[e] * k;
        for (int j = 0; j < k; j++) {
          A0[j] += c * T0[o + j];
          A1[j] += c * T1[o + j];
        }
      }
    } else {
      for (int e = 0; e < m1; e++) {
        f32 c = c0 * c1[e];
        size_t o = oa + (size_t)i1[e] * k;
        for (int j = 0; j < k; j++) {
          A0[j] += c * T0[o + j];
          A1[j] += c * T1[o + j];
          A2[j] += c * T2[o + j];
        }
      }
    }
  }
  for (int j = 0; j < k; j++) v[j] = w0 * A0[j] + w1 * A1[j] + w2 * A2[j];
}

/* Two players, and apply() below a fold: v (k) = sum_a x0[a] (w0 T0 + w1 T1)[a] (rows of a (k, k) block) */
INL void con2_row(const f32 *restrict T0, const f32 *restrict T1, f32 w0, f32 w1, const f32 *restrict x0,
                  f32 *restrict v, const int k) {
  int i0[32];
  int m0 = nzlist(x0, i0, k);
  f32 A0[32], A1[32];
  for (int j = 0; j < k; j++) A0[j] = A1[j] = 0;
  if (T1 == 0) {
    int d = 0;
    for (; d + 1 < m0; d += 2) {
      f32 ca = x0[i0[d]], cb = x0[i0[d + 1]];
      const f32 *ra = T0 + (size_t)i0[d] * k, *rb = T0 + (size_t)i0[d + 1] * k;
      for (int j = 0; j < k; j++) {
        A0[j] += ca * ra[j];
        A1[j] += cb * rb[j];
      }
    }
    if (d < m0) {
      f32 ca = x0[i0[d]];
      const f32 *ra = T0 + (size_t)i0[d] * k;
      for (int j = 0; j < k; j++) A0[j] += ca * ra[j];
    }
    for (int j = 0; j < k; j++) v[j] = w0 * (A0[j] + A1[j]);
    return;
  }
  for (int d = 0; d < m0; d++) {
    f32 c = x0[i0[d]];
    size_t o = (size_t)i0[d] * k;
    for (int j = 0; j < k; j++) {
      A0[j] += c * T0[o + j];
      A1[j] += c * T1[o + j];
    }
  }
  for (int j = 0; j < k; j++) v[j] = w0 * A0[j] + w1 * A1[j];
}

/* Four players: v (k) = sum_{a, b, c} x0[a] x1[b] x2[c] (w0 T0 + w1 T1)[a][b][c] (rows of a (k, k, k, k) block) */
INL void con4_row(const f32 *restrict T0, const f32 *restrict T1, const f32 *restrict T2, f32 w0, f32 w1, f32 w2,
                  const f32 *restrict x0, const f32 *restrict x1, const f32 *restrict x2, f32 *restrict v, const int k) {
  int i0[32], i1[32], i2[32];
  int m0 = nzlist(x0, i0, k), m1 = nzlist(x1, i1, k), m2 = nzlist(x2, i2, k);
  f32 c2[32];
  for (int e = 0; e < m2; e++) c2[e] = x2[i2[e]];
  f32 A0[32], A1[32], B0[32], D0[32];
  for (int j = 0; j < k; j++) A0[j] = A1[j] = B0[j] = D0[j] = 0;
  for (int d = 0; d < m0; d++) {
    f32 c0 = x0[i0[d]];
    for (int e1 = 0; e1 < m1; e1++) {
      f32 c01 = c0 * x1[i1[e1]];
      size_t oab = ((size_t)i0[d] * k + i1[e1]) * k * k;
      if (T1 == 0) {
        int e = 0;
        for (; e + 1 < m2; e += 2) {
          f32 ca = c01 * c2[e], cb = c01 * c2[e + 1];
          const f32 *ra = T0 + oab + (size_t)i2[e] * k, *rb = T0 + oab + (size_t)i2[e + 1] * k;
          for (int j = 0; j < k; j++) {
            A0[j] += ca * ra[j];
            A1[j] += cb * rb[j];
          }
        }
        if (e < m2) {
          f32 ca = c01 * c2[e];
          const f32 *ra = T0 + oab + (size_t)i2[e] * k;
          for (int j = 0; j < k; j++) A0[j] += ca * ra[j];
        }
      } else if (T2 == 0) {
        for (int e = 0; e < m2; e++) {
          f32 c = c01 * c2[e];
          size_t o = oab + (size_t)i2[e] * k;
          for (int j = 0; j < k; j++) {
            A0[j] += c * T0[o + j];
            B0[j] += c * T1[o + j];
          }
        }
      } else { /* (a player all-in for less: the side pot's tensor too) */
        for (int e = 0; e < m2; e++) {
          f32 c = c01 * c2[e];
          size_t o = oab + (size_t)i2[e] * k;
          for (int j = 0; j < k; j++) {
            A0[j] += c * T0[o + j];
            B0[j] += c * T1[o + j];
            D0[j] += c * T2[o + j];
          }
        }
      }
    }
  }
  if (T1 == 0)
    for (int j = 0; j < k; j++) v[j] = w0 * (A0[j] + A1[j]);
  else if (T2 == 0)
    for (int j = 0; j < k; j++) v[j] = w0 * A0[j] + w1 * B0[j];
  else
    for (int j = 0; j < k; j++) v[j] = w0 * A0[j] + w1 * B0[j] + w2 * D0[j];
}

/* The tensors summed over a folded player's reach: M (k, k) [the other][p] = sum over f's buckets of xf * T, f the
   first (fo = 0) or the second (fo = 1) of p's two others; J's and a pot's tensor (T1, M1: or none) at once. */
INL void red_row(const f32 *restrict T0, const f32 *restrict T1, const f32 *restrict xf, int fo, f32 *restrict M0,
                 f32 *restrict M1, const int k) {
  int idx[32];
  int m = nzlist(xf, idx, k);
  f32 c[32];
  for (int e = 0; e < m; e++) c[e] = xf[idx[e]];
  /* fo = 0: M[b] = sum_a xf[a] T[a][b], a stride of k * k; fo = 1: M[a] = sum_b xf[b] T[a][b] */
  size_t step = fo == 0 ? (size_t)k * k : (size_t)k, row = fo == 0 ? (size_t)k : (size_t)k * k;
  for (int r = 0; r < k; r++) {
    f32 A0[32], A1[32], B0[32], B1[32];
    for (int j = 0; j < k; j++) A0[j] = A1[j] = B0[j] = B1[j] = 0;
    const f32 *t0 = T0 + r * row, *t1 = T1 ? T1 + r * row : 0;
    if (t1) {
      for (int e = 0; e < m; e++) {
        size_t o = idx[e] * step;
        for (int j = 0; j < k; j++) {
          A0[j] += c[e] * t0[o + j];
          B0[j] += c[e] * t1[o + j];
        }
      }
      for (int j = 0; j < k; j++) {
        M0[r * k + j] = A0[j];
        M1[r * k + j] = B0[j];
      }
    } else {
      int e = 0;
      for (; e + 1 < m; e += 2) {
        size_t oa = idx[e] * step, ob = idx[e + 1] * step;
        for (int j = 0; j < k; j++) {
          A0[j] += c[e] * t0[oa + j];
          A1[j] += c[e + 1] * t0[ob + j];
        }
      }
      if (e < m) {
        size_t oa = idx[e] * step;
        for (int j = 0; j < k; j++) A0[j] += c[e] * t0[oa + j];
      }
      for (int j = 0; j < k; j++) M0[r * k + j] = A0[j] + A1[j];
    }
  }
}

#define BY_K(call)                 \
  switch (K) {                     \
  case 4: call(4); break;          \
  case 6: call(6); break;          \
  case 8: call(8); break;          \
  case 10: call(10); break;        \
  case 12: call(12); break;        \
  case 16: call(16); break;        \
  default: call(K); break;         \
  }

/* p's values v (C, K) = sum over the others' buckets of w_j T_j times their reaches X (np, C, K) */
static void contract(const f32 *T0, const f32 *T1, const f32 *T2, f32 w0, f32 w1, f32 w2, int p, const f32 *X, f32 *v) {
  if (NP == 4) {
    size_t blk4 = (size_t)K * K * K * K;
    int o[3], m = 0;
    for (int j = 0; j < 4; j++)
      if (j != p) o[m++] = j;
    for (int t = 0; t < C; t++) {
      size_t off = t * blk4;
      const f32 *t0 = T0 + off, *t1 = T1 ? T1 + off : 0, *t2 = T2 ? T2 + off : 0;
      const f32 *x0 = X + ((size_t)o[0] * C + t) * K, *x1 = X + ((size_t)o[1] * C + t) * K,
                *x2 = X + ((size_t)o[2] * C + t) * K;
#define C4(k) con4_row(t0, t1, t2, w0, w1, w2, x0, x1, x2, v + (size_t)t * K, k)
      BY_K(C4)
#undef C4
    }
    return;
  }
  size_t blk = (size_t)K * K * (NP == 3 ? K : 1);
  int o0 = p == 0 ? 1 : 0, o1 = p == 2 ? 1 : 2;
  for (int t = 0; t < C; t++) {
    size_t o = t * blk;
    const f32 *t0 = T0 + o, *t1 = T1 ? T1 + o : 0, *t2 = T2 ? T2 + o : 0;
    const f32 *x0 = X + ((size_t)o0 * C + t) * K;
    f32 *vt = v + (size_t)t * K;
    if (NP == 3) {
      const f32 *x1 = X + ((size_t)o1 * C + t) * K;
#define C3(k) con3_row(t0, t1, t2, w0, w1, w2, x0, x1, vt, k)
      BY_K(C3)
#undef C3
    } else { /* (two players have no side pot: T2 is none) */
#define C2(k) con2_row(t0, t1, w0, w1, x0, vt, k)
      BY_K(C2)
#undef C2
    }
  }
}

/* Below another player f's fold (three players): MJ and MS (C, K, K), J's and the {p, q} pot's tensors summed over f's
   reach xf (C, K) */
typedef struct {
  int f, q;
  f32 *MJ, *MS;
} Fold;

static void reduce(const f32 *T0, const f32 *T1, int p, int f, const f32 *xf, f32 *M0, f32 *M1) {
  int o0 = p == 0 ? 1 : 0, fo = f == o0 ? 0 : 1;
  size_t blk = (size_t)K * K * K, mb = (size_t)K * K;
  for (int t = 0; t < C; t++) {
#define R3(k) red_row(T0 + t * blk, T1 ? T1 + t * blk : 0, xf + (size_t)t * K, fo, M0 + t * mb, T1 ? M1 + t * mb : 0, k)
    BY_K(R3)
#undef R3
  }
}

/* v (C, K) = sum_a xq[a] (w0 M0 + w1 M1)[a] */
static void apply(const f32 *M0, const f32 *M1, f32 w0, f32 w1, const f32 *xq, f32 *v) {
  for (int t = 0; t < C; t++) {
    size_t o = (size_t)t * K * K;
#define A2(k) con2_row(M0 + o, M1 ? M1 + o : 0, w0, w1, xq + (size_t)t * K, v + (size_t)t * K, k)
    BY_K(A2)
#undef A2
  }
}

/* ------------------------------------------------------------------ strategies */
/* sig (rows, A, w) by regret matching on r (rows, A, w): the positive part normalized, uniform where none (avg: the
   strategy sums normalized, uniform where they are 0). Vectorized along w, JB entries at a time. */
#define JB 64
static void normalize(const f32 *restrict r, f32 *restrict sig, int rows, int A, int w, int pos) {
  f32 u = 1.0f / A;
  for (int x = 0; x < rows; x++) {
    const f32 *rr = r + (size_t)x * A * w;
    f32 *ss = sig + (size_t)x * A * w;
    for (int j0 = 0; j0 < w; j0 += JB) {
      int n = w - j0 < JB ? w - j0 : JB;
      f32 sum[JB];
      for (int j = 0; j < n; j++) sum[j] = 0;
      for (int a = 0; a < A; a++) {
        const f32 *ra = rr + (size_t)a * w + j0;
        if (pos)
          for (int j = 0; j < n; j++) sum[j] += ra[j] > 0 ? ra[j] : 0;
        else
          for (int j = 0; j < n; j++) sum[j] += ra[j];
      }
      for (int a = 0; a < A; a++) {
        const f32 *ra = rr + (size_t)a * w + j0;
        f32 *sa = ss + (size_t)a * w + j0;
        for (int j = 0; j < n; j++) {
          f32 v = pos ? (ra[j] > 0 ? ra[j] : 0) : ra[j];
          f32 d = sum[j] > 0 ? sum[j] : 1;
          sa[j] = sum[j] > 0 ? v / d : u;
        }
      }
    }
  }
}
static void match(const f32 *r, f32 *sig, int rows, int A, int w) { normalize(r, sig, rows, A, w, 1); }
static void average(const f32 *sm, f32 *sig, int rows, int A, int w) { normalize(sm, sig, rows, A, w, 0); }

/* a decision's strategy in a pass, kept between passes: in training the current one (flag 1), else the average (2),
   computed where the kept one is the other or stale (0) */
static const f32 *strategy(int mode, const f32 *reg, const f32 *ssum, f32 *kept, i32 flag, int rows, int A, int w) {
  int want = mode == ALT ? 1 : 2;
  if (FLAGS[flag] != want) {
    if (want == 1) match(reg, kept, rows, A, w);
    else average(ssum, kept, rows, A, w);
    FLAGS[flag] = want;
  }
  return kept;
}

/* a decision's combine: v = the node player's value (sum over actions for another player's node); in mode ALT at
   p's own node its regrets and strategy sums are updated (X: p's reach there). va (A, rows, w), sig (rows, A, w). */
static void combine(int mode, int mine, int A, int rows, int w, const f32 *restrict va, const f32 *restrict sig,
                    f32 *restrict reg, f32 *restrict ssum, const f32 *restrict Xp, f32 *restrict v) {
  size_t m = (size_t)rows * w;
  if (!mine) {
    for (size_t j = 0; j < m; j++) v[j] = va[j];
    for (int a = 1; a < A; a++)
      for (size_t j = 0; j < m; j++) v[j] += va[a * m + j];
    return;
  }
  if (mode == BR) {
    for (size_t j = 0; j < m; j++) v[j] = va[j];
    for (int a = 1; a < A; a++)
      for (size_t j = 0; j < m; j++) v[j] = va[a * m + j] > v[j] ? va[a * m + j] : v[j];
    return;
  }
  for (int x = 0; x < rows; x++) {
    f32 *vx = v + (size_t)x * w;
    for (int j = 0; j < w; j++) vx[j] = 0;
    for (int a = 0; a < A; a++) {
      const f32 *vat = va + a * m + (size_t)x * w, *sg = sig + ((size_t)x * A + a) * w;
      for (int j = 0; j < w; j++) vx[j] += vat[j] * sg[j];
    }
    if (mode != ALT) continue;
    const f32 *xp = Xp + (size_t)x * w;
    for (int a = 0; a < A; a++) {
      const f32 *vat = va + a * m + (size_t)x * w, *sg = sig + ((size_t)x * A + a) * w;
      f32 *rg = reg + ((size_t)x * A + a) * w, *sm = ssum + ((size_t)x * A + a) * w;
      for (int j = 0; j < w; j++) {
        rg[j] += vat[j] - vx[j];
        sm[j] += xp[j] * sg[j];
      }
    }
  }
}

/* ------------------------------------------------------------------ the bucket world */
static void term(const i32 *w, const f32 *X, f32 *v, int p, const Fold *fd) {
  int live = w[1], side = w[2];
  f32 net = fbits(w[3]), fm = fbits(w[4]);
  f32 wJ = -fbits(w[5 + p]), wA = 0, wS = 0;
  int kA = -2, kS = -2;
  if (live >> p & 1) {
    f32 fa = fm >= 0 ? fm : 1, fs = fm >= 0 ? 1 - fm : 0;
    if (pop(live) > 1) {
      kA = live;
      wA = net * fa;
    } else {
      wJ += net * fa;
    }
    if (fs != 0 && (side >> p & 1)) {
      if (pop(side) > 1) {
        kS = side;
        wS = net * fs;
      } else {
        wJ += net * fs;
      }
    }
  }
  if (fd) { /* live within {p, q}: its side pot (if any) is p's alone or not p's */
    apply(fd->MJ, kA >= 0 ? fd->MS : 0, wJ, wA, X + (size_t)fd->q * C * K, v);
    return;
  }
  const f32 *T0 = tensor(-1, p), *T1 = kA >= 0 ? tensor(kA, p) : 0, *T2 = kS >= 0 ? tensor(kS, p) : 0;
  if (!T1 && T2) {
    T1 = T2;
    wA = wS;
    T2 = 0;
  }
  contract(T0, T1, T2, wJ, wA, wS, p, X, v);
}

static void bwalk(int node, const f32 *X, f32 *v, int mode, int p, const Fold *fd);

static void bdec(const i32 *w, const f32 *X, f32 *v, int mode, int p, const Fold *fd) {
  int q = w[1], A = w[2], fold = w[5];
  f32 *reg = PF(w[3]), *ssum = PF(w[4]);
  size_t mark = top, ck = (size_t)C * K;
  const f32 *sig = strategy(mode, reg, ssum, PF(w[7]), w[8], C, A, K);
  f32 *Xc = salloc(sizeof(f32) * ck * NP), *va = salloc(sizeof(f32) * ck * A);
  for (int a = 0; a < A; a++) {
    for (size_t j = 0; j < ck * NP; j++) Xc[j] = X[j];
    f32 *xq = Xc + (size_t)q * ck;
    for (int t = 0; t < C; t++)
      for (int k = 0; k < K; k++) xq[t * K + k] *= sig[((size_t)t * A + a) * K + k];
    f32 *vo = va + (size_t)a * ck;
    int kid = w[9 + a];
    if (a == fold) {
      if (q == p) { /* p's own fold: minus its chips in times the deals' weight, whatever follows */
        if (fd) apply(fd->MJ, 0, -fbits(w[6]), 0, Xc + (size_t)fd->q * ck, vo);
        else contract(tensor(-1, p), 0, 0, -fbits(w[6]), 0, 0, p, Xc, vo);
        continue;
      }
      if (!fd && NP == 3) {
        size_t m2 = top;
        Fold nf;
        nf.f = q;
        nf.q = 3 - p - q;
        nf.MJ = salloc(sizeof(f32) * ck * K);
        nf.MS = salloc(sizeof(f32) * ck * K);
        reduce(tensor(-1, p), tensor((1 << p) | (1 << nf.q), p), p, q, xq, nf.MJ, nf.MS);
        bwalk(kid, Xc, vo, mode, p, &nf);
        top = m2;
        continue;
      }
    }
    bwalk(kid, Xc, vo, mode, p, fd);
  }
  combine(mode, q == p, A, C, K, va, sig, reg, ssum, X + (size_t)p * ck, v);
  if (mode == ALT && q == p) FLAGS[w[8]] = 0;
  top = mark;
}

static void bwalk(int node, const f32 *X, f32 *v, int mode, int p, const Fold *fd) {
  const i32 *w = W + node;
  switch (w[0]) {
  case BDEC: bdec(w, X, v, mode, p, fd); break;
  case BTERM: term(w, X, v, p, fd); break;
  case BEXIT: bwalk(w[1], X, v, mode, p, fd); break;
  default: __builtin_trap();
  }
}

/* ------------------------------------------------------------------ the root street, per combo */
/* the reach over the combos sharing no card with each combo: the total, minus each of its cards' combos, plus itself */
static void compat(const f32 *x, f64 *card, f64 *out) {
  f64 tot = 0;
  for (int c = 0; c < 52; c++) card[c] = 0;
  for (int i = 0; i < N; i++) {
    card[CA[i]] += x[i];
    card[CB[i]] += x[i];
    tot += x[i];
  }
  for (int i = 0; i < N; i++) out[i] = tot - card[CA[i]] - card[CB[i]] + x[i];
}

/* The exact joint mass (MultiwaySolver._mass): per combo h, the sum of x(i) y(j) over the pairs of combos sharing no
   card with h or with each other: sum over i compatible with h of x(i) U_y(h + i), U_y(S) the y-reach of the combos
   sharing no card with the four cards S by inclusion-exclusion (y's total, minus each card's, plus each pair's) */
static f64 XM[52 * 52], YM[52 * 52];
static void mass3(const f32 *x, const f32 *y, f32 *m) {
  size_t mark = top;
  f64 *Cx = salloc(sizeof(f64) * 52), *Cy = salloc(sizeof(f64) * 52), *Cz = salloc(sizeof(f64) * 52),
      *Cw = salloc(sizeof(f64) * 52);
  f64 *Kx = salloc(sizeof(f64) * N), *Ky = salloc(sizeof(f64) * N), *Kz = salloc(sizeof(f64) * N),
      *Kw = salloc(sizeof(f64) * N);
  f32 *z = salloc(sizeof(f32) * N), *ww = salloc(sizeof(f32) * N);
  f64 *X = XM, *Y = YM; /* (zero but for the combos' entries, set here and cleared after) */
  compat(x, Cx, Kx);
  compat(y, Cy, Ky);
  for (int i = 0; i < N; i++) {
    z[i] = (f32)(x[i] * (Cy[CA[i]] + Cy[CB[i]]));
    ww[i] = x[i] * y[i];
    X[CA[i] * 52 + CB[i]] = X[CB[i] * 52 + CA[i]] = x[i];
    Y[CA[i] * 52 + CB[i]] = Y[CB[i] * 52 + CA[i]] = y[i];
  }
  /* (z in float64 would be exact; its float32 rounding is far under the float32 regrets' own) */
  compat(z, Cz, Kz);
  compat(ww, Cw, Kw);
  for (int i = 0; i < N; i++) {
    int e = CA[i], f = CB[i];
    f64 d = 0;
    for (int dd = 0; dd < 52; dd++) {
      if (dd == e || dd == f) continue;
      f64 avail = Cx[dd] - X[dd * 52 + e] - X[dd * 52 + f];
      d += (Y[e * 52 + dd] + Y[f * 52 + dd]) * avail;
    }
    m[i] = (f32)(Kx[i] * Ky[i] - Kz[i] + Kw[i] + d);
  }
  for (int i = 0; i < N; i++) X[CA[i] * 52 + CB[i]] = X[CB[i] * 52 + CA[i]] = Y[CA[i] * 52 + CB[i]] = Y[CB[i] * 52 + CA[i]] = 0;
  top = mark;
}

static void kmass(const f32 *x, f32 *m) {
  size_t mark = top;
  f64 *Cx = salloc(sizeof(f64) * 52), *Kx = salloc(sizeof(f64) * N);
  compat(x, Cx, Kx);
  for (int i = 0; i < N; i++) m[i] = (f32)Kx[i];
  top = mark;
}

/* p's deals' weight per combo: the others' compatible reach (four players: pairwise, MultiwaySolver._pw_masses: the
   product of the others' one-sided masses times p's overlap correction, scaled so its total is player 0's, the factor
   held to [0.5, 2]) */
static void deal_mass(const f32 *R, int p, f32 *m) {
  if (NP == 4) {
    size_t mark = top;
    f32 *ku = salloc(sizeof(f32) * 4 * N), *m0 = salloc(sizeof(f32) * N);
    for (int j = 0; j < 4; j++) kmass(R + (size_t)j * N, ku + (size_t)j * N);
    f64 tot[2] = {0, 0};
    for (int r = 0; r < 2; r++) {
      int q = r == 0 ? 0 : p;
      f32 *mq = r == 0 ? m0 : m;
      for (int h = 0; h < N; h++) {
        f32 x = PWC[(size_t)q * N + h];
        for (int j = 0; j < 4; j++)
          if (j != q) x *= ku[(size_t)j * N + h];
        mq[h] = x;
        tot[r] += (f64)R[(size_t)q * N + h] * x;
      }
    }
    f64 sc = tot[1] > 1e-280 ? tot[0] / tot[1] : 1.0;
    sc = sc < 0.5 ? 0.5 : sc > 2.0 ? 2.0 : sc;
    if (p != 0)
      for (int h = 0; h < N; h++) m[h] = (f32)(m[h] * sc);
    top = mark;
    return;
  }
  if (NP == 2) {
    kmass(R + (size_t)(1 - p) * N, m);
  } else {
    int o0 = p == 0 ? 1 : 0, o1 = p == 2 ? 1 : 2;
    mass3(R + (size_t)o0 * N, R + (size_t)o1 * N, m);
  }
}

/* each player's reach (C, K) per class and bucket: the sums of its combos' reaches times the class's mask, a bucket's
   combos in a run (PERM, SEG, MKP) */
static void to_buckets(const f32 *R, f32 *X) {
  for (int i = 0; i < NP; i++) {
    const f32 *r = R + (size_t)i * N;
    for (int t = 0; t < C; t++) {
      size_t o = ((size_t)i * C + t);
      const i32 *pm = PERM + o * N, *sg = SEG + o * (K + 1);
      const f32 *mk = MKP + o * N;
      f32 *x = X + o * K;
      for (int k = 0; k < K; k++) {
        f32 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int e = sg[k], end = sg[k + 1];
        for (; e + 3 < end; e += 4) {
          s0 += r[pm[e]] * mk[e];
          s1 += r[pm[e + 1]] * mk[e + 1];
          s2 += r[pm[e + 2]] * mk[e + 2];
          s3 += r[pm[e + 3]] * mk[e + 3];
        }
        for (; e < end; e++) s0 += r[pm[e]] * mk[e];
        x[k] = (s0 + s1) + (s2 + s3);
      }
    }
  }
}

static void from_buckets(const f32 *V, int p, f32 *v) {
  for (int h = 0; h < N; h++) v[h] = 0;
  for (int t = 0; t < C; t++) {
    const i32 *id = IDS + ((size_t)p * C + t) * N;
    const f32 *wm = WMASK + (size_t)t * N, *vt = V + (size_t)t * K;
    for (int h = 0; h < N; h++) v[h] += vt[id[h]] * wm[h];
  }
}

static void rwalk(int node, const f32 *R, f32 *v, int mode, int p, f32 *kids) {
  const i32 *w = W + node;
  size_t mark = top;
  switch (w[0]) {
  case RDEC: {
    int q = w[1], A = w[2];
    f32 *reg = PF(w[3]), *ssum = PF(w[4]);
    const f32 *sig = strategy(mode, reg, ssum, PF(w[5]), w[6], 1, A, N);
    f32 *Rc = salloc(sizeof(f32) * NP * N), *va = salloc(sizeof(f32) * A * N);
    for (int a = 0; a < A; a++) {
      for (size_t j = 0; j < (size_t)NP * N; j++) Rc[j] = R[j];
      for (int h = 0; h < N; h++) Rc[(size_t)q * N + h] *= sig[(size_t)a * N + h];
      rwalk(w[7 + a], Rc, va + (size_t)a * N, mode, p, 0);
    }
    combine(mode, q == p, A, 1, N, va, sig, reg, ssum, R + (size_t)p * N, v);
    if (mode == ALT && q == p) FLAGS[w[6]] = 0;
    if (kids)
      for (size_t j = 0; j < (size_t)A * N; j++) kids[j] = va[j];
    break;
  }
  case RFOLD: {
    f32 net = fbits(w[2]);
    f32 *m = salloc(sizeof(f32) * N);
    deal_mass(R, p, m);
    f32 k = (p == w[1] ? net : 0) - fbits(w[3 + p]);
    for (int h = 0; h < N; h++) v[h] = k * m[h];
    break;
  }
  case REXIT: {
    int live = w[1];
    f32 cp = fbits(w[3 + p]);
    size_t ck = (size_t)C * K;
    f32 *X = salloc(sizeof(f32) * ck * NP), *vb = salloc(sizeof(f32) * ck);
    to_buckets(R, X);
    if (!(live >> p & 1)) {
      contract(tensor(-1, p), 0, 0, -cp, 0, 0, p, X, vb);
    } else if (NP == 3 && pop(live) == 2) { /* one player folded on the root street: its reach is fixed below */
      Fold fd;
      fd.f = __builtin_ctz((unsigned)(7 & ~live));
      fd.q = 3 - p - fd.f;
      fd.MJ = salloc(sizeof(f32) * ck * K);
      fd.MS = salloc(sizeof(f32) * ck * K);
      reduce(tensor(-1, p), tensor(live, p), p, fd.f, X + (size_t)fd.f * ck, fd.MJ, fd.MS);
      bwalk(w[2], X, vb, mode, p, &fd);
    } else {
      bwalk(w[2], X, vb, mode, p, 0);
    }
    from_buckets(vb, p, v);
    break;
  }
  default: __builtin_trap();
  }
  top = mark;
}

/* ------------------------------------------------------------------ entry points */
/* p's values (n) at the root-street node `at` for the reaches R (np, n) at it: mode BR p's best response to the others'
   average strategies, EV everyone on their average strategies, ALT a training pass of p's (current strategies, p's
   regrets and strategy sums updated); kids (A, n): with an RDEC, each action's (0: not wanted). Returns 0, or -1. */
EXPORT(mw_eval) i32 mw_eval(u32 prog, i32 at, i32 mode, i32 p, u32 R, u32 out, u32 kids) {
  load(prog);
  if (W[0] != MAGIC) return -1;
  rwalk(at, PF(R), PF(out), mode, p, kids ? PF(kids) : 0);
  return 0;
}

/* `iters` training iterations from the root (each player's pass but an all-in's, then the discount: factors holds
   [pos, ssum] per iteration, negative regrets halved). Returns the iterations run, or -1 on a bad program. */
EXPORT(mw_run) i32 mw_run(u32 prog, i32 iters, u32 factors) {
  load(prog);
  if (W[0] != MAGIC) return -1;
  const f32 *R0 = PF(W[15]), *fac = PF(factors);
  const i32 *arrs = PI(W[13]);
  int narr = W[12];
  for (int it = 0; it < iters; it++) {
    for (int p = 0; p < NP; p++) {
      if (ALLIN >> p & 1) continue;
      size_t mark = top;
      f32 *v = salloc(sizeof(f32) * N);
      rwalk(W[5], R0, v, ALT, p, 0);
      top = mark;
    }
    f32 pos = fac[2 * it], sf = fac[2 * it + 1];
    for (int a = 0; a < narr; a++) {
      f32 *reg = PF(arrs[3 * a]), *ssum = PF(arrs[3 * a + 1]);
      size_t len = (u32)arrs[3 * a + 2];
      for (size_t x = 0; x < len; x++) reg[x] *= reg[x] > 0 ? pos : 0.5f;
      for (size_t x = 0; x < len; x++) ssum[x] *= sf;
    }
    for (int j = 0; j < NFLAG; j++) FLAGS[j] = 0;
  }
  return iters;
}
