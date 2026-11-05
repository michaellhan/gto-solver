/* The multiway core: whole training iterations of gto/multiway_solver.py's 3-player (and 2-player) solves whose later
 * streets are bet on by hand buckets (MWConfig.later = "buckets": every flop and turn solve the advisor runs), and the
 * passes after a solve (each player's value on the average strategies, its best response, an action's EVs). gto/mwcore.py
 * lays the solver's tree out as a program and runs it; the same source builds natively (a shared library, ctypes) and
 * for the browser (gto/mwcore.wasm, instantiated by gto/static/worker.js beside Pyodide, and gto/mwcore-mt.wasm, the
 * same with threads: below), with no libc: everything lives in one arena, the inputs copied in by gto/mwcore.py and a
 * pass's scratch above them (stack order; a helper thread's in a lane of its own).
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
 *   BTERMW [7, live, side, wJ_0, wA_0, wS_0, wJ_1 ..]             a tournament's terminal: p's value is wJ_p times the
 *                                                                deals' weight plus wA_p (wS_p) times its share of
 *                                                                the main (side) pot (MultiwaySolver._icm_setup)
 *   RSHOW [8, live, npot, b_0 .. b_np-1, (mask_k, a_k0 .. a_k,np-1) x npot]  a river root's showdown: show()
 *   RRUN  [9, as RSHOW]                                          a root-street all-in, cards to come: runout()
 * (c: each player's chips in, as floats; arrs: here one run of every decision's regrets and one of their sums)
 * tens: per player p, J's tensor at [p] and live set L's shares at [np + L * np + p] (offsets; 0 none); ids (np, C, n)
 * each combo's bucket; masks, wmask (C, n); arrs (narr, 3): every decision's regret, ssum and length (the discount);
 * perm (np, C, n) each player's combos per class in bucket order, seg (np, C, K + 1) each bucket's first, mkp (np, C,
 * n) the class's mask in that order; flags (nflag, one a decision record): 1 while its sig is its regrets' current
 * strategy (computed at its first use after the discount or its player's pass, so a pass's strategies are the numpy
 * solver's, from the regrets as they stand), 2 while it is the average strategy (the evaluations'), 0 stale.
 * pwc (np, n): four players, each player's pairwise overlap correction (MultiwaySolver._pw_corr: the root street's
 * deals' weight is pairwise, _pw_masses); 0 with two or three (exact).
 * A river root (three players, no later streets: no buckets, C = K = 0) has showdowns on the root street, per combo
 * with exact card removal, and the header's words after pwc: str (n) each combo's strength, ord (n) the combos by
 * strength, gend (n) the end of each one's tie group in that order, sm (52, 52) the strength of the combo {c, d} at
 * [c][d] (int32); 0 for any other program. After them rb, the rows walked for a root-street all-in of three (runouts),
 * rord (rb, n) the combos by strength on each, rgend (rb, n) the end of each one's tie group in that order, rmk (rb, n)
 * the combos each leaves live; rt the heads-up table's space (n, n float32), rf every runout, rfs (rf, n) strengths on
 * each (-1: the combo holds a card of it), rpair how many of them a compatible pair of combos sees, rdone the table's
 * built; 0 with no all-in. A solve with no line left to bet on has C = K = 0.
 *
 * Four players (nobody all-in) contract (C, K_o0, K_o1, K_o2, K_p) tensors over three reaches, with no fold reductions.
 */
#include <stddef.h>
#include <stdint.h>

typedef float f32;
typedef double f64;
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
#ifdef THREADS
#define MAXT 16   /* this thread and its helpers, at most */
#else
#define MAXT 1
#endif

#define MAGIC 0x6d776334 /* "mwc3" */
enum { RDEC = 1, RFOLD = 2, REXIT = 3, BDEC = 4, BTERM = 5, BEXIT = 6, BTERMW = 7, RSHOW = 8, RRUN = 9 };
enum { ALT = 0, BR = 1, EV = 2 };

/* ------------------------------------------------------------------ arena */
static unsigned char *base;

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

/* A thread's scratch: allocated up to top, a helper's within its lane (at most lim); id: the thread (its XM, YM) */
typedef struct {
  size_t top, lim;
  int id;
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
  return base + at;
}

static u32 lane_prog;   /* the program the helpers' lanes are sized for (lane_bytes; 0: none yet) */
static size_t lane_bytes;

EXPORT(mw_reset) void mw_reset(void) {
#ifndef __wasm__
  if (base && touched > ((size_t)64 << 20)) {
    madvise(base, touched, MADV_FREE);
    touched = 0;
  }
#endif
  M.top = 0;
  lane_prog = 0;
}

EXPORT(mw_alloc) u32 mw_alloc(u32 bytes) { return (u32)((unsigned char *)salloc(&M, bytes) - base); }

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
static const i32 *STR, *ORD, *GEND, *SM;   /* (a river root's) */
static int RB;                             /* (a root-street all-in's runouts) */
static const i32 *RORD, *RGEND;
static const f32 *RMK, *RT;
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

/* MultiwaySolver._allin_table: T[q][h], combo h's share of a pot against combo q over every runout (a win 1, a tie a
   half, 0 if they share a card), counted in half-wins and then divided by the runouts a compatible pair sees. */
static void equity_table(void) {
  int rows = W[31];
  const i32 *st = PI(W[32]);
  i32 *T = PI(W[30]);
  for (size_t j = 0; j < (size_t)N * N; j++) T[j] = 0;
  for (int b = 0; b < rows; b++) {
    const i32 *s = st + (size_t)b * N;
    for (int q = 0; q < N; q++) {
      i32 sq = s[q], *t = T + (size_t)q * N;
      if (sq < 0) continue;
      for (int h = 0; h < N; h++) t[h] += (s[h] > sq) + (s[h] >= sq);
    }
  }
  f32 sc = 0.5f / (f32)W[33];
  f32 *F = PF(W[30]);
  for (int q = 0; q < N; q++)
    for (int h = 0; h < N; h++) {
      size_t o = (size_t)q * N + h;
      F[o] = (CA[q] == CA[h] || CA[q] == CB[h] || CB[q] == CA[h] || CB[q] == CB[h]) ? 0.0f : (f32)T[o] * sc;
    }
  ((i32 *)W)[34] = 1;
}

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
  STR = W[22] ? PI(W[22]) : 0;
  ORD = W[23] ? PI(W[23]) : 0;
  GEND = W[24] ? PI(W[24]) : 0;
  SM = W[25] ? PI(W[25]) : 0;
  RB = W[26];
  RORD = W[27] ? PI(W[27]) : 0;
  RGEND = W[28] ? PI(W[28]) : 0;
  RMK = W[29] ? PF(W[29]) : 0;
  RT = W[30] ? PF(W[30]) : 0;
  if (RT && !W[34]) equity_table();
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
  f32 wJ, wA = 0, wS = 0;
  int kA = -2, kS = -2;
  if (w[0] == BTERMW) {
    const i32 *pw = w + 3 + 3 * p;
    wJ = fbits(pw[0]);
    if (live >> p & 1) {
      f32 a = fbits(pw[1]), s = fbits(pw[2]);
      if (pop(live) > 1) {
        kA = live;
        wA = a;
      } else {
        wJ += a;
      }
      if (s != 0 && (side >> p & 1)) {
        if (pop(side) > 1) {
          kS = side;
          wS = s;
        } else {
          wJ += s;
        }
      }
    }
  } else {
    wJ = -fbits(w[5 + p]);
    if (live >> p & 1) {
      f32 net = fbits(w[3]), fm = fbits(w[4]);
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

static void bwalk(Lane *L, int node, const f32 *X, f32 *v, int mode, int p, const Fold *fd);

static void bdec(Lane *L, const i32 *w, const f32 *X, f32 *v, int mode, int p, const Fold *fd) {
  int q = w[1], A = w[2], fold = w[5];
  f32 *reg = PF(w[3]), *ssum = PF(w[4]);
  size_t mark = L->top, ck = (size_t)C * K;
  const f32 *sig = strategy(mode, reg, ssum, PF(w[7]), w[8], C, A, K);
  f32 *Xc = salloc(L, sizeof(f32) * ck * NP), *va = salloc(L, sizeof(f32) * ck * A);
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
        size_t m2 = L->top;
        Fold nf;
        nf.f = q;
        nf.q = 3 - p - q;
        nf.MJ = salloc(L, sizeof(f32) * ck * K);
        nf.MS = salloc(L, sizeof(f32) * ck * K);
        reduce(tensor(-1, p), tensor((1 << p) | (1 << nf.q), p), p, q, xq, nf.MJ, nf.MS);
        bwalk(L, kid, Xc, vo, mode, p, &nf);
        L->top = m2;
        continue;
      }
    }
    bwalk(L, kid, Xc, vo, mode, p, fd);
  }
  combine(mode, q == p, A, C, K, va, sig, reg, ssum, X + (size_t)p * ck, v);
  if (mode == ALT && q == p) FLAGS[w[8]] = 0;
  L->top = mark;
}

static void bwalk(Lane *L, int node, const f32 *X, f32 *v, int mode, int p, const Fold *fd) {
  const i32 *w = W + node;
  switch (w[0]) {
  case BDEC: bdec(L, w, X, v, mode, p, fd); break;
  case BTERM:
  case BTERMW: term(w, X, v, p, fd); break;
  case BEXIT: bwalk(L, w[1], X, v, mode, p, fd); break;
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
static f64 XM[MAXT][52 * 52], YM[MAXT][52 * 52];   /* (a thread's own) */
static void mass3(Lane *L, const f32 *x, const f32 *y, f32 *m) {
  size_t mark = L->top;
  f64 *Cx = salloc(L, sizeof(f64) * 52), *Cy = salloc(L, sizeof(f64) * 52), *Cz = salloc(L, sizeof(f64) * 52),
      *Cw = salloc(L, sizeof(f64) * 52);
  f64 *Kx = salloc(L, sizeof(f64) * N), *Ky = salloc(L, sizeof(f64) * N), *Kz = salloc(L, sizeof(f64) * N),
      *Kw = salloc(L, sizeof(f64) * N);
  f32 *z = salloc(L, sizeof(f32) * N), *ww = salloc(L, sizeof(f32) * N);
  f64 *X = XM[L->id], *Y = YM[L->id]; /* (zero but for the combos' entries, set here and cleared after) */
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
  L->top = mark;
}

static void kmass(Lane *L, const f32 *x, f32 *m) {
  size_t mark = L->top;
  f64 *Cx = salloc(L, sizeof(f64) * 52), *Kx = salloc(L, sizeof(f64) * N);
  compat(x, Cx, Kx);
  for (int i = 0; i < N; i++) m[i] = (f32)Kx[i];
  L->top = mark;
}

/* p's deals' weight per combo: the others' compatible reach (four players: pairwise, MultiwaySolver._pw_masses: the
   product of the others' one-sided masses times p's overlap correction, scaled so its total is player 0's, the factor
   held to [0.5, 2]) */
static void deal_mass(Lane *L, const f32 *R, int p, f32 *m) {
  if (NP == 4) {
    size_t mark = L->top;
    f32 *ku = salloc(L, sizeof(f32) * 4 * N), *m0 = salloc(L, sizeof(f32) * N);
    for (int j = 0; j < 4; j++) kmass(L, R + (size_t)j * N, ku + (size_t)j * N);
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
    f64 sc = tot[1] > 1e-280 && tot[1] > 1e-9 * (tot[0] < 0 ? -tot[0] : tot[0]) ? tot[0] / tot[1] : 1.0;
    sc = sc < 0.5 ? 0.5 : sc > 2.0 ? 2.0 : sc;
    if (p != 0)
      for (int h = 0; h < N; h++) m[h] = (f32)(m[h] * sc);
    L->top = mark;
    return;
  }
  if (NP == 2) {
    kmass(L, R + (size_t)(1 - p) * N, m);
  } else {
    int o0 = p == 0 ? 1 : 0, o1 = p == 2 ? 1 : 2;
    mass3(L, R + (size_t)o0 * N, R + (size_t)o1 * N, m);
  }
}

/* ------------------------------------------------------------------ a river root's showdowns */
/* RSHOW: p's value per combo h: b_p times the deals' weight m(h) plus a_kp times p's share of each pot k it contests
   (mask_k: the players contesting it; MultiwaySolver._term, exact), summed over the others' hands compatible with h and
   with each other, by inclusion-exclusion over the cards two of them could share. With x and z the others' reaches
   (q's, r's), y = x z, P_x(h, c) for a card c not h's the x-reach of the combos holding c that share no card with h
   (U: all of them, L: those weaker than h, E: tied, W = L + E / 2), and K_x(h) the same over every combo (the heads-up
   kernels),
     m        = K_U(x) K_U(z) - sum_c P_U(x) P_U(z) + K_U(y)
     3 live:    K_W(x) K_W(z) + K_E(x) K_E(z) / 12 - sum_c [P_W(x) P_W(z) + P_E(x) P_E(z) / 12] + K_L(y) + K_E(y) / 3
     p and q:   K_W(x) K_U(z) - sum_c P_W(x) P_U(z) + K_L(y) + K_E(y) / 2      (r's hand still holds its cards)
   (pairs of hands sharing one card are counted once by the sum over c, the same hand twice: K(y) puts it back once; a
   share is 1 against weaker hands, 1/2 against one tie, 1/3 against two: W_x W_z + E_x E_z / 12). One walk up the
   combos by strength keeps each card's reach weaker than the current tie group (L) and in it (T); a combo's P's are
   those less the combos holding one of its own cards (the pair matrices' rows e and f), summed over the 52 cards four
   lanes at a time, the tie terms (E: nonzero only on the tie group's cards) apart. O(52 n), no tables. */
typedef f32 v4 __attribute__((vector_size(16)));
typedef i32 vi4 __attribute__((vector_size(16)));
INL f64 hsum(v4 a) { return (f64)((a[0] + a[1]) + (a[2] + a[3])); }

static void show(Lane *L, const i32 *w, const f32 *R, f32 *v, int p) {
  int live = w[1], npot = w[2];
  f32 bp = fbits(w[3 + p]);
  if (!(live >> p & 1)) { /* p folded: its chips in (a weight) times the deals' weight */
    deal_mass(L, R, p, v);
    for (int h = 0; h < N; h++) v[h] *= bp;
    return;
  }
  size_t mark = L->top;
  int q = p == 0 ? 1 : 0, r = p == 2 ? 1 : 2;
  const f32 *x = R + (size_t)q * N, *z = R + (size_t)r * N;
  const i32 *pots = w + 3 + NP;
  f32 *Xq = salloc(L, sizeof(f32) * 52 * 52), *Xr = salloc(L, sizeof(f32) * 52 * 52);
  f64 *A = salloc(L, sizeof(f64) * 52 * 9);
  f64 *Lx = A, *Tx = A + 52, *Lz = A + 104, *Tz = A + 156, *Ly = A + 208, *Ty = A + 260, *Cx = A + 312, *Cz = A + 364,
      *Cy = A + 416;
  f32 *lx = salloc(L, sizeof(f32) * 52 * 5), *lz = lx + 52, *cx = lx + 104, *cz = lx + 156, *mk = lx + 208;
  i32 *gc = salloc(L, sizeof(i32) * 52);   /* the tie group's cards */
  unsigned char *ing = salloc(L, 52);
  for (int j = 0; j < 52 * 52; j++) Xq[j] = Xr[j] = 0;
  for (int c = 0; c < 52 * 9; c++) A[c] = 0;
  for (int c = 0; c < 52; c++) lx[c] = lz[c] = 0, mk[c] = 1, ing[c] = 0;
  f64 tx = 0, tz = 0, ty = 0;
  for (int i = 0; i < N; i++) {
    int a = CA[i], b = CB[i];
    f64 xi = x[i], zi = z[i], yi = xi * zi;
    Xq[a * 52 + b] = Xq[b * 52 + a] = x[i];
    Xr[a * 52 + b] = Xr[b * 52 + a] = z[i];
    Cx[a] += xi, Cx[b] += xi, Cz[a] += zi, Cz[b] += zi, Cy[a] += yi, Cy[b] += yi;
    tx += xi, tz += zi, ty += yi;
  }
  for (int c = 0; c < 52; c++) cx[c] = (f32)Cx[c], cz[c] = (f32)Cz[c];
  const v4 *lx4 = (const v4 *)lx, *lz4 = (const v4 *)lz, *cx4 = (const v4 *)cx, *cz4 = (const v4 *)cz,
           *mk4 = (const v4 *)mk;
  const vi4 one = {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000};   /* (1.0f's bits) */
  f64 XLx = 0, XLz = 0, XLy = 0;
  for (int i = 0; i < N;) {
    int end = GEND[i], ng = 0;
    f64 XEx = 0, XEz = 0, XEy = 0;
    for (int j = i; j < end; j++) {
      int g = ORD[j], a = CA[g], b = CB[g];
      f64 xg = x[g], zg = z[g], yg = xg * zg;
      Tx[a] += xg, Tx[b] += xg, Tz[a] += zg, Tz[b] += zg, Ty[a] += yg, Ty[b] += yg;
      XEx += xg, XEz += zg, XEy += yg;
      if (!ing[a]) ing[a] = 1, gc[ng++] = a;
      if (!ing[b]) ing[b] = 1, gc[ng++] = b;
    }
    for (int j = i; j < end; j++) {
      int h = ORD[j], e = CA[h], f = CB[h];
      i32 s = STR[h];
      f64 xh = x[h], zh = z[h], yh = xh * zh;
      f64 kux = tx - Cx[e] - Cx[f] + xh, kuz = tz - Cz[e] - Cz[f] + zh, kuy = ty - Cy[e] - Cy[f] + yh;
      f64 klx = XLx - Lx[e] - Lx[f], klz = XLz - Lz[e] - Lz[f], kly = XLy - Ly[e] - Ly[f];
      f64 kex = XEx - Tx[e] - Tx[f] + xh, kez = XEz - Tz[e] - Tz[f] + zh, key = XEy - Ty[e] - Ty[f] + yh;
      /* the sums over the cards: the weaker and the compatible parts, four cards a lane */
      const v4 *qe = (const v4 *)(Xq + e * 52), *qf = (const v4 *)(Xq + f * 52), *re = (const v4 *)(Xr + e * 52),
               *rf = (const v4 *)(Xr + f * 52);
      const vi4 *se = (const vi4 *)(SM + e * 52), *sf = (const vi4 *)(SM + f * 52);
      const vi4 s4 = {s, s, s, s};
      v4 aLL = {0, 0, 0, 0}, aUU = aLL, aLU = aLL, aUL = aLL;
      mk[e] = mk[f] = 0;
      for (int c = 0; c < 13; c++) {
        v4 le = (v4)((se[c] < s4) & one), lf = (v4)((sf[c] < s4) & one);
        v4 plx = (lx4[c] - qe[c] * le - qf[c] * lf) * mk4[c];
        v4 plz = lz4[c] - re[c] * le - rf[c] * lf;
        v4 pux = (cx4[c] - qe[c] - qf[c]) * mk4[c];
        v4 puz = cz4[c] - re[c] - rf[c];
        aLL += plx * plz;
        aUU += pux * puz;
        aLU += plx * puz;
        aUL += pux * plz;
      }
      mk[e] = mk[f] = 1;
      /* the tie terms, on the tie group's cards */
      f64 tLE = 0, tEE = 0, tEU = 0, tUE = 0;
      for (int k = 0; k < ng; k++) {
        int c = gc[k];
        if (c == e || c == f) continue;
        f32 xe = Xq[e * 52 + c], xf = Xq[f * 52 + c], ze = Xr[e * 52 + c], zf = Xr[f * 52 + c];
        i32 sec = SM[e * 52 + c], sfc = SM[f * 52 + c];
        f32 ee = sec == s, ef = sfc == s, le = sec < s, lf = sfc < s;
        f32 pex = (f32)Tx[c] - xe * ee - xf * ef, pez = (f32)Tz[c] - ze * ee - zf * ef;
        f32 plx = lx[c] - xe * le - xf * lf, plz = lz[c] - ze * le - zf * lf;
        f32 pux = cx[c] - xe - xf, puz = cz[c] - ze - zf;
        tLE += plx * pez + pex * plz, tEE += pex * pez, tEU += pex * puz, tUE += pux * pez;
      }
      f64 dWW = hsum(aLL) + 0.5 * tLE + 0.25 * tEE, dWU = hsum(aLU) + 0.5 * tEU, dUW = hsum(aUL) + 0.5 * tUE;
      f64 kwx = klx + 0.5 * kex, kwz = klz + 0.5 * kez;
      f64 m = kux * kuz - hsum(aUU) + kuy;
      f64 val = bp * m;
      for (int k = 0; k < npot; k++) {
        const i32 *pk = pots + (size_t)k * (1 + NP);
        int mask = pk[0];
        if (!(mask >> p & 1)) continue;
        f64 win;
        if (pop(mask) == 1) win = m;
        else if (pop(mask) == 3) win = kwx * kwz + kex * kez / 12 - dWW - tEE / 12 + kly + key / 3;
        else if (mask >> q & 1) win = kwx * kuz - dWU + kly + 0.5 * key;
        else win = kwz * kux - dUW + kly + 0.5 * key;
        val += fbits(pk[1 + p]) * win;
      }
      v[h] = (f32)val;
    }
    for (int k = 0; k < ng; k++) {
      int c = gc[k];
      Lx[c] += Tx[c], Lz[c] += Tz[c], Ly[c] += Ty[c];
      Tx[c] = Tz[c] = Ty[c] = 0;
      lx[c] = (f32)Lx[c], lz[c] = (f32)Lz[c];
      ing[c] = 0;
    }
    XLx += XEx, XLz += XEz, XLy += XEy;
    i = end;
  }
  L->top = mark;
}

/* ------------------------------------------------------------------ a root-street all-in's runouts */
/* RRUN: nobody can bet, so p's value per combo h is the deals' weight m(h) (deal_mass, as a fold's) times b_p plus
   a_kp times h's own share of each pot k it contests, over the runouts (MultiwaySolver._term_runout). Heads-up the
   table (equity_table): RT x the other's reach (weighted by what the hands of the players out of that pot leave it)
   over its compatible reach, O(n^2) with a thin range cheaper. Against
   two others q and r the rows: W_q W_r + E_q E_r / 12 over U_q U_r (W = weaker + tied / 2 of the reach, independent
   given h and the row), each summed over the rows h is live on before dividing: one walk up the combos by strength
   per row and player, O(rows n). Three players: a pot's shares scaled to add up to every deal's. */
static void runout_kernels(const f32 *x, const f32 *mk, const i32 *ord, const i32 *gend, f32 *y, f64 *cc, f64 *ku,
                           f64 *kw, f64 *ke) {
  f64 tot = 0;
  f64 *Cc = cc, *Lc = cc + 52, *Tc = cc + 104;
  for (int c = 0; c < 156; c++) cc[c] = 0;
  for (int i = 0; i < N; i++) {
    y[i] = x[i] * mk[i];
    Cc[CA[i]] += y[i];
    Cc[CB[i]] += y[i];
    tot += y[i];
  }
  f64 XL = 0;
  for (int i = 0; i < N;) {
    int end = gend[i];
    f64 XE = 0;
    for (int j = i; j < end; j++) {
      int g = ord[j];
      Tc[CA[g]] += y[g];
      Tc[CB[g]] += y[g];
      XE += y[g];
    }
    for (int j = i; j < end; j++) {
      int h = ord[j], e = CA[h], f = CB[h];
      f64 kl = XL - Lc[e] - Lc[f], k = XE - Tc[e] - Tc[f] + y[h];
      ku[h] = tot - Cc[e] - Cc[f] + y[h];
      kw[h] = kl + 0.5 * k;
      ke[h] = k;
    }
    for (int j = i; j < end; j++) {
      int g = ord[j], a = CA[g], b = CB[g];
      Lc[a] += Tc[a], Tc[a] = 0;
      Lc[b] += Tc[b], Tc[b] = 0;
    }
    XL += XE;
    i = end;
  }
}

static void runout(Lane *L, const i32 *w, const f32 *R, f32 *v, int p) {
  int npot = w[2];
  const i32 *pots = w + 3 + NP;
  size_t mark = L->top, stride = 1 + (size_t)NP, nn = (size_t)npot * NP * N;
  f32 *M = salloc(L, sizeof(f32) * (size_t)NP * N), *y = salloc(L, sizeof(f32) * N), *x = salloc(L, sizeof(f32) * N);
  f64 *val = salloc(L, sizeof(f64) * N), *cc = salloc(L, sizeof(f64) * 156);
  f64 *num = salloc(L, sizeof(f64) * nn), *den = salloc(L, sizeof(f64) * nn);
  f64 *T = salloc(L, sizeof(f64) * 3 * (size_t)NP * N);
  int tabs = 0, masses = 1 << p;            /* whose rows' kernels; whose deals' weights (three: to scale the shares) */
  for (int k = 0; k < npot; k++) {
    int mask = pots[(size_t)k * stride];
    if (!(mask >> p & 1) || pop(mask) < 2) continue;
    if (pop(mask) > 2) tabs |= NP == 3 ? mask : mask & ~(1 << p);
    if (NP == 3) masses |= mask;
  }
  for (int j = 0; j < NP; j++)
    if (masses >> j & 1) deal_mass(L, R, j, M + (size_t)j * N);
  for (size_t j = 0; j < nn; j++) num[j] = den[j] = 0;
  for (int k = 0; k < npot; k++) {  /* heads-up: the table, linear in the other's reach */
    int mask = pots[(size_t)k * stride];
    if (!(mask >> p & 1) || pop(mask) != 2) continue;
    for (int l = 0; l < NP; l++) {
      if (!(mask >> l & 1) || (NP == 3 ? 0 : l != p)) continue;
      f64 *nk = num + ((size_t)k * NP + l) * N, *dk = den + ((size_t)k * NP + l) * N;
      int o = __builtin_ctz((unsigned)(mask & ~(1 << l)));
      for (int h = 0; h < N; h++) x[h] = R[(size_t)o * N + h];
      for (int j = 0; j < NP; j++) /* a player out of this pot still holds cards: o's weight given them */
        if (j != l && j != o) {
          kmass(L, R + (size_t)j * N, y);
          for (int h = 0; h < N; h++) x[h] *= y[h];
        }
      for (int h = 0; h < N; h++) y[h] = 0;
      for (int q = 0; q < N; q++) {
        f32 xq = x[q];
        const f32 *t = RT + (size_t)q * N;
        if (xq > 0)
          for (int h = 0; h < N; h++) y[h] += xq * t[h];
      }
      for (int h = 0; h < N; h++) nk[h] = y[h];
      kmass(L, x, y);
      for (int h = 0; h < N; h++) dk[h] = y[h];
    }
  }
  for (int b = 0; tabs && b < RB; b++) {
    const i32 *ord = RORD + (size_t)b * N, *gend = RGEND + (size_t)b * N;
    const f32 *mk = RMK + (size_t)b * N;
    for (int j = 0; j < NP; j++)
      if (tabs >> j & 1) {
        f64 *t = T + 3 * (size_t)j * N;
        runout_kernels(R + (size_t)j * N, mk, ord, gend, y, cc, t, t + N, t + 2 * (size_t)N);
      }
    for (int k = 0; k < npot; k++) {
      int mask = pots[(size_t)k * stride];
      if (!(mask >> p & 1) || pop(mask) < 3) continue;
      for (int l = 0; l < NP; l++) {
        if (!(mask >> l & 1) || (NP != 3 && l != p)) continue;
        f64 *nk = num + ((size_t)k * NP + l) * N, *dk = den + ((size_t)k * NP + l) * N;
        int o[3], no = 0;
        for (int j = 0; j < NP; j++)
          if (j != l && (mask >> j & 1)) o[no++] = j;
        const f64 *t0 = T + 3 * (size_t)o[0] * N, *t1 = T + 3 * (size_t)o[1] * N;
        if (no == 2) {
          const f64 *e0 = t0 + 2 * (size_t)N, *e1 = t1 + 2 * (size_t)N;
          for (int h = 0; h < N; h++) {
            nk[h] += (t0[N + h] * t1[N + h] + e0[h] * e1[h] / 12.0) * mk[h];
            dk[h] += t0[h] * t1[h] * mk[h];
          }
        } else {
          const f64 *t2 = T + 3 * (size_t)o[2] * N;
          for (int h = 0; h < N; h++) {
            nk[h] += t0[N + h] * t1[N + h] * t2[N + h] * mk[h];
            dk[h] += t0[h] * t1[h] * t2[h] * mk[h];
          }
        }
      }
    }
  }
  f32 bp = fbits(w[3 + p]);
  for (int h = 0; h < N; h++) val[h] = bp;
  for (int k = 0; k < npot; k++) {
    const i32 *pk = pots + (size_t)k * stride;
    int mask = pk[0];
    if (!(mask >> p & 1)) continue;
    f64 a = fbits(pk[1 + p]);
    if (pop(mask) == 1) {
      for (int h = 0; h < N; h++) val[h] += a;
      continue;
    }
    f64 tot = 0, whole = 0;
    for (int l = 0; l < NP; l++) {
      if (!(mask >> l & 1) || (NP != 3 && l != p)) continue;
      f64 *nk = num + ((size_t)k * NP + l) * N, *dk = den + ((size_t)k * NP + l) * N; /* num becomes the share */
      const f32 *rl = R + (size_t)l * N, *ml = M + (size_t)l * N;
      f64 tl = 0, wl = 0;
      for (int h = 0; h < N; h++) {
        f64 s = dk[h] > 0 ? nk[h] / dk[h] : 0;
        s = s < 0 ? 0 : s > 1 ? 1 : s;
        nk[h] = s;
        tl += (f64)rl[h] * ml[h] * s;
        wl += (f64)rl[h] * ml[h];
      }
      tot += tl;
      if (!whole) whole = wl;
    }
    if (NP == 3 && tot > 1e-300) a *= whole / tot;
    const f64 *sp = num + ((size_t)k * NP + p) * N;
    for (int h = 0; h < N; h++) val[h] += a * sp[h];
  }
  const f32 *mp = M + (size_t)p * N;
  for (int h = 0; h < N; h++) v[h] = (f32)(val[h] * mp[h]);
  L->top = mark;
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

/* A root-street leaf: p's values v (n) for the reaches R (np, n) there: a fold's, or an Exit's (the bucket subtree
   below it, from each player's reach per class and bucket) */
static void leaf(Lane *L, const i32 *w, const f32 *R, f32 *v, int mode, int p) {
  size_t mark = L->top;
  if (w[0] == RFOLD) {
    f32 net = fbits(w[2]);
    f32 *m = salloc(L, sizeof(f32) * N);
    deal_mass(L, R, p, m);
    f32 k = (p == w[1] ? net : 0) - fbits(w[3 + p]);
    for (int h = 0; h < N; h++) v[h] = k * m[h];
  } else if (w[0] == REXIT) {
    int live = w[1];
    f32 cp = fbits(w[3 + p]);
    size_t ck = (size_t)C * K;
    f32 *X = salloc(L, sizeof(f32) * ck * NP), *vb = salloc(L, sizeof(f32) * ck);
    to_buckets(R, X);
    if (!(live >> p & 1)) {
      contract(tensor(-1, p), 0, 0, -cp, 0, 0, p, X, vb);
    } else if (NP == 3 && pop(live) == 2) { /* one player folded on the root street: its reach is fixed below */
      Fold fd;
      fd.f = __builtin_ctz((unsigned)(7 & ~live));
      fd.q = 3 - p - fd.f;
      fd.MJ = salloc(L, sizeof(f32) * ck * K);
      fd.MS = salloc(L, sizeof(f32) * ck * K);
      reduce(tensor(-1, p), tensor(live, p), p, fd.f, X + (size_t)fd.f * ck, fd.MJ, fd.MS);
      bwalk(L, w[2], X, vb, mode, p, &fd);
    } else {
      bwalk(L, w[2], X, vb, mode, p, 0);
    }
    from_buckets(vb, p, v);
  } else if (w[0] == RSHOW) {
    show(L, w, R, v, p);
  } else if (w[0] == RRUN) {
    runout(L, w, R, v, p);
  } else {
    __builtin_trap();
  }
  L->top = mark;
}

/* With helper threads (mw_threads) a pass runs its root-street leaves as one job: a walk down (COLLECT) keeps each
   leaf's reaches (a task), the job runs every task in some thread's own scratch (a lane), and the pass proper (READ)
   takes their values where the leaves are. A leaf's subtree is its own: its decisions' regrets, sums, kept strategies
   and flags are its own records' (mwcore.py: a root Exit's block of its group), the tensors are only read, and a
   fold's deal masses use the thread's own XM, YM; so the tasks write nothing in common, and each one's arithmetic is
   the serial pass's: the floats are the same on any number of threads. The root street's decisions are walked twice,
   their strategies kept from the first (the same ones), and updated on READ's walk, after their children's values, as
   the serial pass does. */
enum { WHOLE, COLLECT, READ };
static int phase;
typedef struct {
  i32 node;
  f32 *R, *v;   /* the reaches there (np, n), its values (n) */
} Task;
static Task *tasks;
static i32 ntasks, maxtasks, task_next;
static f32 *task_area;

static void rwalk(Lane *L, int node, const f32 *R, f32 *v, int mode, int p, f32 *kids) {
  const i32 *w = W + node;
  if (w[0] != RDEC) {
    if (phase == COLLECT) {
      if (ntasks == maxtasks) __builtin_trap();
      Task *t = &tasks[ntasks++];
      t->node = node;
      t->R = task_area, t->v = task_area + (size_t)NP * N;
      task_area += (size_t)(NP + 1) * N;
      for (size_t j = 0; j < (size_t)NP * N; j++) t->R[j] = R[j];
    } else if (phase == READ) {
      if (task_next >= ntasks || tasks[task_next].node != node) __builtin_trap();   /* (the walks differ: a bug) */
      const f32 *tv = tasks[task_next++].v;
      for (int h = 0; h < N; h++) v[h] = tv[h];
    } else {
      leaf(L, w, R, v, mode, p);
    }
    return;
  }
  size_t mark = L->top;
  int q = w[1], A = w[2];
  f32 *reg = PF(w[3]), *ssum = PF(w[4]);
  const f32 *sig = strategy(mode, reg, ssum, PF(w[5]), w[6], 1, A, N);
  f32 *Rc = salloc(L, sizeof(f32) * NP * N), *va = salloc(L, sizeof(f32) * A * N);
  for (int a = 0; a < A; a++) {
    for (size_t j = 0; j < (size_t)NP * N; j++) Rc[j] = R[j];
    for (int h = 0; h < N; h++) Rc[(size_t)q * N + h] *= sig[(size_t)a * N + h];
    rwalk(L, w[7 + a], Rc, va + (size_t)a * N, mode, p, 0);
  }
  if (phase != COLLECT) {
    combine(mode, q == p, A, 1, N, va, sig, reg, ssum, R + (size_t)p * N, v);
    if (mode == ALT && q == p) FLAGS[w[6]] = 0;
    if (kids)
      for (size_t j = 0; j < (size_t)A * N; j++) kids[j] = va[j];
  }
  L->top = mark;
}

static i32 leaves(i32 node) {
  const i32 *w = W + node;
  if (w[0] != RDEC) return 1;
  i32 m = 0;
  for (int a = 0; a < w[2]; a++) m += leaves(w[7 + a]);
  return m;
}

/* the bytes a task can take (a helper's lane holds the largest): bdec's scratch on a subtree's deepest line, each step
   with a fold's reduction (more than any line takes); an Exit's own; a fold's deal masses (mass3's, kmass's: more than
   either). salloc's alignment: 64 bytes an allocation. Exact sizes depend on the tree alone, not on the reaches. */
static size_t need_b(i32 node) {
  const i32 *w = W + node;
  size_t ck = (size_t)C * K, al = 64;
  if (w[0] == BEXIT) return need_b(w[1]);
  if (w[0] != BDEC) return 0;
  size_t kid = 0;
  for (int a = 0; a < w[2]; a++) {
    size_t k = need_b(w[9 + a]);
    if (k > kid) kid = k;
  }
  return 4 * al + 4 * ck * NP + 4 * ck * (size_t)w[2] + 2 * 4 * ck * K + kid;
}
static size_t need(i32 node) {
  const i32 *w = W + node;
  size_t ck = (size_t)C * K, al = 64, n = (size_t)N;
  if (w[0] == REXIT) return 4 * al + 4 * ck * NP + 4 * ck + 2 * 4 * ck * K + need_b(w[2]);
  if (w[0] == RFOLD) return 16 * al + 4 * n + 4 * 5 * n + 8 * 52 * 4 + 8 * 4 * n + 4 * 2 * n;
  if (w[0] == RSHOW) /* (a fold's, and its own) */
    return 16 * al + 4 * n + 4 * 5 * n + 8 * 52 * 4 + 8 * 4 * n + 4 * 2 * n + 8 * al + 2 * 4 * 52 * 52 + 8 * 52 * 9 +
           4 * 52 * 6 + 52;
  if (w[0] == RRUN) /* (a fold's, and its own) */
    return 16 * al + 4 * n + 4 * 5 * n + 8 * 52 * 4 + 8 * 4 * n + 4 * 2 * n + 8 * al + 4 * ((size_t)NP + 2) * n +
           8 * n + 8 * 156 + 2 * 8 * (size_t)w[2] * NP * n + 8 * 3 * (size_t)NP * n;
  size_t m = 0;
  for (int a = 0; a < w[2]; a++) {
    size_t k = need(w[7 + a]);
    if (k > m) m = k;
  }
  return m;
}

/* ------------------------------------------------------------------ threads */
/* A job: the pass's tasks, handed out one at a time. The ticket holds the job's number and the next task to take (an
   odd number while the job is being set up), so a thread only takes tasks of the job it read; done counts the tasks
   finished. Natively the helpers are pthreads (mw_threads starts them); in the browser, workers sharing the module's
   memory (mwcore-mt.wasm), each in mw_helper. As gtocore.c's (core/gtocore.c: its threads, the same scheme). */
static i32 nthreads = 1;   /* this thread and its helpers */
static i32 dead, fail_i, fail_n, helped;

#ifdef THREADS
static struct {
  u64 ticket;
  i32 done, gen, sleepers;
  i32 mode, p, rows, helpers;
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

#ifndef __wasm__
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static i32 started;
#endif
static Lane hl[MAXT];   /* the helpers' lanes */

/* this thread's share of job `seq`: tasks while there are any */
static void take(Lane *L, u32 seq, int mode, int p, i32 rows) {
  for (;;) {
    u64 t = __atomic_load_n(&J.ticket, __ATOMIC_ACQUIRE);
    i32 r = (i32)(u32)t;
    if ((u32)(t >> 32) != seq || r >= rows) return;
    if (__sync_val_compare_and_swap(&J.ticket, t, t + 1) != t) continue;
    if (LD(fail_n) && L == &hl[LD(fail_i) & (MAXT - 1)] && __atomic_sub_fetch(&fail_n, 1, __ATOMIC_RELAXED) == 0)
      __builtin_trap();   /* (mw_fail: a test's helper failing with a task held) */
    Task *k = &tasks[r];
    leaf(L, W + k->node, k->R, k->v, mode, p);
    if (L != &M) __atomic_fetch_add(&helped, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&J.done, 1, __ATOMIC_RELEASE);
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

/* helper i: takes tasks of every job it sees, in its lane */
static void helper(i32 i) {
  u32 last = 0;
  for (;;) {
    await_job(last);
    u64 t = __atomic_load_n(&J.ticket, __ATOMIC_ACQUIRE);
    u32 seq = (u32)(t >> 32);
    if (seq & 1) continue;   /* being set up */
    i32 mode = LD(J.mode), p = LD(J.p), rows = LD(J.rows), helpers = LD(J.helpers);
    size_t lanes = LD(J.lanes), lane = LD(J.lane);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if ((u32)(__atomic_load_n(&J.ticket, __ATOMIC_RELAXED) >> 32) != seq) continue;   /* (read across two jobs) */
    last = seq;
    if (i >= helpers) continue;
    Lane *L = &hl[i];
    L->top = lanes + (size_t)i * lane, L->lim = lanes + (size_t)(i + 1) * lane, L->id = i + 1;
    take(L, seq, mode, p, rows);
  }
}

#ifdef __wasm__
/* a browser helper's loop (worker.js's helpers call it once; it never returns), on a stack of its own (mw_stack) */
static unsigned char stacks[MAXT][65536] __attribute__((aligned(16)));
EXPORT(mw_stack) u32 mw_stack(i32 i) { return (u32)(uintptr_t)stacks[i < 0 || i >= MAXT ? 0 : i] + 65536; }
EXPORT(mw_helper) void mw_helper(i32 i) { helper(i); }
#else
static void *run_helper(void *i) {
  helper((i32)(intptr_t)i);
  return 0;
}
#endif

/* the tasks as a job: this thread takes tasks too (in the arena), then waits for the rest */
static void job(int mode, int p) {
  i32 helpers = nthreads - 1;
  size_t mark = M.top;
  unsigned char *lanes = salloc(&M, lane_bytes * (size_t)helpers);
  u32 seq = (u32)(__atomic_load_n(&J.ticket, __ATOMIC_RELAXED) >> 32) + 2;
  __atomic_store_n(&J.ticket, ((u64)(seq - 1) << 32) | 0xffffffffu, __ATOMIC_SEQ_CST);
  ST(J.mode, mode), ST(J.p, p), ST(J.rows, ntasks), ST(J.helpers, helpers);
  ST(J.lanes, (size_t)(lanes - base)), ST(J.lane, lane_bytes);
  ST(J.done, 0);
  __atomic_store_n(&J.ticket, (u64)seq << 32, __ATOMIC_SEQ_CST);
  wake();
  take(&M, seq, mode, p, ntasks);
  for (int spin = 0; __atomic_load_n(&J.done, __ATOMIC_ACQUIRE) < ntasks; spin++) {
    relax();
    if (LD(dead)) __builtin_trap();   /* (a helper failed holding a task: it won't come) */
#ifndef __wasm__
    if (spin > 1 << 12) sched_yield();
#endif
  }
  M.top = mark;
}
#endif

/* a pass from the root-street node `at`: with helpers its leaves as a job (above), else depth first */
static void pass(i32 at, const f32 *R, f32 *v, int mode, int p, f32 *kids) {
#ifdef THREADS
  if (nthreads > 1 && W[at] == RDEC) {
    u32 prog = (u32)((const unsigned char *)W - base);
    if (lane_prog != prog) {
      lane_bytes = (need(W[5]) + 63) & ~(size_t)63;
      lane_prog = prog;
    }
    size_t mark = M.top;
    i32 nt = leaves(at);
    tasks = salloc(&M, sizeof(Task) * (size_t)nt);
    task_area = salloc(&M, sizeof(f32) * (size_t)nt * (size_t)(NP + 1) * N);
    maxtasks = nt, ntasks = task_next = 0;
    phase = COLLECT;
    rwalk(&M, at, R, v, mode, p, kids);
    phase = WHOLE;
    job(mode, p);
    phase = READ;
    rwalk(&M, at, R, v, mode, p, kids);
    phase = WHOLE;
    if (task_next != ntasks) __builtin_trap();
    M.top = mark;
    return;
  }
#endif
  rwalk(&M, at, R, v, mode, p, kids);
}

/* ------------------------------------------------------------------ entry points */
/* p's values (n) at the root-street node `at` for the reaches R (np, n) at it: mode BR p's best response to the others'
   average strategies, EV everyone on their average strategies, ALT a training pass of p's (current strategies, p's
   regrets and strategy sums updated); kids (A, n): with an RDEC, each action's (0: not wanted). Returns 0, or -1. */
EXPORT(mw_eval) i32 mw_eval(u32 prog, i32 at, i32 mode, i32 p, u32 R, u32 out, u32 kids) {
  load(prog);
  if (W[0] != MAGIC) return -1;
  pass(at, PF(R), PF(out), mode, p, kids ? PF(kids) : 0);
  return 0;
}

/* `iters` training iterations from the root (each player's pass but an all-in's, then the discount: factors holds
   [pos, neg, ssum] per iteration: positive regrets x pos, negative x neg, sums x ssum). Returns the iterations run,
   or -1 on a bad program. */
EXPORT(mw_run) i32 mw_run(u32 prog, i32 iters, u32 factors) {
  load(prog);
  if (W[0] != MAGIC) return -1;
  const f32 *R0 = PF(W[15]), *fac = PF(factors);
  const i32 *arrs = PI(W[13]);
  int narr = W[12];
  for (int it = 0; it < iters; it++) {
    for (int p = 0; p < NP; p++) {
      if (ALLIN >> p & 1) continue;
      size_t mark = M.top;
      f32 *v = salloc(&M, sizeof(f32) * N);
      pass(W[5], R0, v, ALT, p, 0);
      M.top = mark;
    }
    f32 pos = fac[3 * it], neg = fac[3 * it + 1], sf = fac[3 * it + 2];
    for (int a = 0; a < narr; a++) {
      f32 *reg = PF(arrs[3 * a]), *ssum = PF(arrs[3 * a + 1]);
      size_t len = (u32)arrs[3 * a + 2];
      for (size_t x = 0; x < len; x++) reg[x] *= reg[x] > 0 ? pos : neg;
      for (size_t x = 0; x < len; x++) ssum[x] *= sf;
    }
    for (int j = 0; j < NFLAG; j++) FLAGS[j] = 0;
  }
  return iters;
}

/* this thread and t - 1 helpers (natively started here, in the browser by worker.js); returns the number (t 0: just
   that). A solve's numbers don't depend on it. */
EXPORT(mw_threads) i32 mw_threads(i32 t) {
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
  nthreads = t;
  return nthreads;
}

/* the address of the word a failed helper sets (worker.js: its number + 1; the job waiting on its task traps on it),
   and for tests: i > 0 sets it to i */
EXPORT(mw_dead) u32 mw_dead(i32 i) {
  if (i > 0) __atomic_store_n(&dead, i, __ATOMIC_SEQ_CST);
  return (u32)(uintptr_t)&dead;
}

/* tests: helper i traps holding the task of its nth from now (n 0: none) */
EXPORT(mw_fail) i32 mw_fail(i32 i, i32 n) {
  __atomic_store_n(&fail_i, i, __ATOMIC_RELAXED);
  __atomic_store_n(&fail_n, n, __ATOMIC_RELAXED);
  return n;
}

/* the tasks the helpers have run (so a test, or the page, can tell they're running) */
EXPORT(mw_helped) i32 mw_helped(void) { return __atomic_load_n(&helped, __ATOMIC_RELAXED); }
