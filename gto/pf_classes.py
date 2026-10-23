"""The 169 preflop hand classes, and push/fold equilibrium ranges for short-stack jams."""
from __future__ import annotations
import base64
import zlib
from functools import lru_cache
import numpy as np
from .cards import ALL_COMBOS, hand_class

CLASSES = sorted({hand_class(cb) for cb in ALL_COMBOS})
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
CLASS_OF = np.array([CLASS_INDEX[hand_class(cb)] for cb in ALL_COMBOS])
_MASK = np.array([(1 << a) | (1 << b) for a, b in ALL_COMBOS], dtype=np.int64)
COMPAT = (_MASK[:, None] & _MASK[None, :]) == 0                 # combo pairs with no shared card
N_IN = np.bincount(CLASS_OF).astype(float)                       # combos per class


def class_sums(m: np.ndarray) -> np.ndarray:
    """Sum a (1326, 1326) combo-pair matrix into (169, 169) class pairs (add.at, not matmul,
    which raises spurious warnings with macOS Accelerate)."""
    rows = np.zeros((len(CLASSES), m.shape[1]))
    np.add.at(rows, CLASS_OF, m)
    out = np.zeros((len(CLASSES), len(CLASSES)))
    np.add.at(out.T, CLASS_OF, rows.T)
    return out


# share of combo pairs between two classes that don't share a card (class_sums(COMPAT) exactly, integer counts: summed
# over runs of the combos sorted by class, ~3x faster than add.at, which is time the browser's worker spends at startup)
_ORDER = np.argsort(CLASS_OF, kind="stable")
_STARTS = np.searchsorted(CLASS_OF[_ORDER], np.arange(len(CLASSES)))
PAIR_COMPAT = (np.add.reduceat(np.add.reduceat(COMPAT[_ORDER][:, _ORDER].astype(np.int64), _STARTS, axis=0), _STARTS, axis=1)
               .astype(float) / np.outer(N_IN, N_IN))


@lru_cache(maxsize=1)
def equity_matrix() -> np.ndarray:
    """eq[a, b]: all-in equity of class a against class b."""
    from .pf_equity import DATA
    raw = np.frombuffer(zlib.decompress(base64.b64decode("".join(DATA))), dtype="<u2")
    k = len(CLASSES)
    return raw.reshape(k, k) / 65535.0


def to_classes(r: np.ndarray) -> np.ndarray:
    """Combo weights -> combos per class."""
    return np.bincount(CLASS_OF, weights=r, minlength=len(CLASSES))


def equity_vs(villain: np.ndarray) -> np.ndarray:
    """All-in preflop equity of every combo against the weighted range `villain`, from the precomputed
    hand-class equity table (exact up to class-level card removal; board sampling was noisier than the
    gaps between neighbouring hands)."""
    w = PAIR_COMPAT * to_classes(villain)[None, :]
    den = w.sum(1)
    eq = np.where(den > 1e-12, (equity_matrix() * w).sum(1) / np.maximum(den, 1e-12), np.nan)
    return eq[CLASS_OF]


@lru_cache(maxsize=1)
def _eq_compat() -> np.ndarray:
    """equity_matrix() times PAIR_COMPAT: equity against a class, weighted by the pairs of combos that can meet."""
    return equity_matrix() * PAIR_COMPAT


def push_fold(jam_range: np.ndarray, call_range: np.ndarray, jam: float, jammer_in: float, caller_in: float,
              pot: float, n_callers: int = 1, iters: int = 400, rake=None, alt=None) -> np.ndarray:
    """Push/fold equilibrium by fictitious play, in hand classes.

    The jammer (range jam_range, combos) has jammer_in in the pot of `pot` and shoves to `jam`; each of
    n_callers opponents (range call_range, caller_in already in) calls or folds. Callers are treated
    alike and independently; only one of them is assumed to call. Returns the jammer's push frequency
    per combo. `rake`: (rate, cap in bb or None), or None for no rake. A called jam runs out a board, so
    it sees the flop and its pot is raked (as Advisor._vs_jam prices a call of an all-in); a jam that
    takes the pot uncalled ends preflop and isn't ("no flop, no drop"). `alt`: per combo, the EV (over
    folding) of the jammer's best other option, a flat call of the raise it faces: a hand jams only when the
    jam beats it (push/fold against folding alone jammed a big blind's 25bb rejam over a button open with
    30% of hands, where the solves flat most of them)."""
    alt_c = np.zeros(len(CLASSES)) if alt is None else np.bincount(CLASS_OF, weights=alt, minlength=len(CLASSES)) / N_IN
    wj, wc = to_classes(jam_range), to_classes(call_range)
    final = 2 * jam + pot - jammer_in - caller_in
    if rake and rake[0] > 0:
        r = rake[0] * final
        final -= r if rake[1] is None else min(r, rake[1])   # what the winner collects
    # (each iteration's sums over the other class are matrix-vector products with the card-removal shares: ~5x
    # faster than the (169, 169) products summed, which a many-way jam line called 50 times over)
    # (matmul raises spurious warnings with macOS Accelerate: see class_sums)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _fictitious_play(wj, wc, alt_c, jam, jammer_in, caller_in, final, pot, n_callers, iters)[CLASS_OF]


def _fictitious_play(wj, wc, alt_c, jam, jammer_in, caller_in, final, pot, n_callers, iters):
    push = np.ones(len(CLASSES))
    avg_p = np.zeros(len(CLASSES)); avg_c = np.zeros(len(CLASSES))
    eqc = _eq_compat()
    tot = np.maximum(PAIR_COMPAT @ wc, 1e-12)
    for t in range(1, iters + 1):
        # callers: call when equity against the pushing range pays for the call
        x = push * wj
        e_c = (eqc @ x) / np.maximum(PAIR_COMPAT @ x, 1e-12)
        call = (e_c * final - (jam - caller_in) > 0).astype(float)
        avg_c += (call - avg_c) / t
        # jammer: push when it beats folding (0) given the callers' average strategy
        y = avg_c * wc
        q = PAIR_COMPAT @ y
        p_call1 = q / tot
        p_nocall = (1 - p_call1) ** n_callers
        e_j = (eqc @ y) / np.maximum(q, 1e-12)
        ev = p_nocall * pot + (1 - p_nocall) * (e_j * final - (jam - jammer_in))
        push = (ev > alt_c).astype(float)
        avg_p += (push - avg_p) / t
    return avg_p
