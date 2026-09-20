"""7-card hand strength for many combos at once, vectorised with numpy.

Uses treys' lookup tables (Cactus Kev style: prime-product keyed ranks) but
evaluates all 21 five-card subsets of every combo+board in one numpy pass.
Strength is 7463 - treys rank, so higher is better; combos overlapping the
board get -1.
"""
from __future__ import annotations
from itertools import combinations
from typing import Sequence
import numpy as np
from treys import Card
from treys.lookup import LookupTable
from .cards import ALL_COMBOS, card_str

_T = LookupTable()
_TREYS = np.array([Card.new(card_str(c)) for c in range(52)], dtype=np.int64)
_PRIME = _TREYS & 0xFF                      # low byte is the rank prime
_SUIT = _TREYS & 0xF000
_RANKBITS = _TREYS >> 16

# sorted prime-product -> rank tables
_fk = np.array(sorted(_T.flush_lookup)); _fv = np.array([_T.flush_lookup[k] for k in _fk])
_uk = np.array(sorted(_T.unsuited_lookup)); _uv = np.array([_T.unsuited_lookup[k] for k in _uk])
# prime product for every 13-bit rank pattern with exactly five bits (flush hands)
_RB_PRIME = np.zeros(1 << 13, dtype=np.int64)
for bits in range(1 << 13):
    if bin(bits).count("1") == 5:
        _RB_PRIME[bits] = Card.prime_product_from_rankbits(bits)

_SUBSETS = np.array(list(combinations(range(7), 5)))   # (21, 5)
_CA = np.array([a for a, b in ALL_COMBOS]); _CB = np.array([b for a, b in ALL_COMBOS])


def _rank_5(sets: np.ndarray) -> np.ndarray:
    """sets: (..., 5) card ints 0..51 -> treys rank (1 best .. 7462)."""
    flush = np.bitwise_and.reduce(_SUIT[sets], axis=-1) != 0
    prod = np.prod(_PRIME[sets], axis=-1)
    unsuited = _uv[np.clip(np.searchsorted(_uk, prod), 0, len(_uk) - 1)]
    rb = np.bitwise_or.reduce(_RANKBITS[sets], axis=-1)
    fprod = _RB_PRIME[rb]
    fl = _fv[np.clip(np.searchsorted(_fk, fprod), 0, len(_fk) - 1)]
    return np.where(flush, fl, unsuited)


def eval_board(board: Sequence[int], combos: Sequence[int] = None) -> np.ndarray:
    """Strength for combos (indices into ALL_COMBOS, default all 1326) on this 5-card
    board. Combos overlapping the board get -1."""
    board = np.asarray(sorted(board), dtype=np.int64)
    idx = np.arange(1326) if combos is None else np.asarray(list(combos), dtype=np.int64)
    a, b = _CA[idx], _CB[idx]
    seven = np.concatenate([a[:, None], b[:, None], np.broadcast_to(board, (len(idx), 5))], axis=1)  # (n, 7)
    sets = seven[:, _SUBSETS]                                   # (n, 21, 5)
    ranks = _rank_5(sets).min(axis=1)
    out = (7463 - ranks).astype(np.int32)
    blocked = np.isin(a, board) | np.isin(b, board)
    out[blocked] = -1
    return out
