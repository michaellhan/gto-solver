"""7-card hand strength for many combos at once, vectorised with numpy.

Uses treys' lookup tables (Cactus Kev style: prime-product keyed ranks). On a fixed board a hand's
best five cards without a flush depend only on the two hole cards' ranks, so the 21 five-card subsets
are evaluated for the 169 rank pairs instead of all 1326 combos; flushes come from a table of the best
flush (straight flush first) in any set of suited ranks, for the suits the board has three or more of.
The same ranks as evaluating every subset of every combo, about 8x faster.
Strength is 7463 - treys rank, so higher is better; combos overlapping the board get -1.
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
_RANK = np.array([int(x).bit_length() - 1 for x in _RANKBITS])          # 0 = deuce .. 12 = ace
_SUIT_I = np.array([(int(x) >> 12).bit_length() - 1 for x in _SUIT])    # 0..3
_RANK_PRIME = np.array([int(_PRIME[_RANK == r][0]) for r in range(13)], dtype=np.int64)
_CRA, _CRB, _CSA, _CSB = _RANK[_CA], _RANK[_CB], _SUIT_I[_CA], _SUIT_I[_CB]
_PAIRS = np.array([(a, b) for a in range(13) for b in range(13)])    # hole rank pairs


def _flush_table() -> np.ndarray:
    """Best flush rank (treys, 1 best) for every 13-bit set of suited ranks; no flush (under five) = 2**20."""
    bits = np.arange(1 << 13, dtype=np.int64)
    count = np.array([bin(b).count("1") for b in range(1 << 13)])
    top, c = bits.copy(), count.copy()
    while (c > 5).any():                    # the five highest ranks: drop the lowest bit until five are left
        drop = c > 5
        top = np.where(drop, top & (top - 1), top); c = c - drop
    sf = np.zeros_like(bits)                # the highest straight inside the set (wheel last, lowest)
    for m in [0b1000000001111] + [0b11111 << i for i in range(9)]:
        sf = np.where((bits & m) == m, m, sf)
    five = np.where(sf > 0, sf, top)
    out = np.full(1 << 13, 1 << 20, dtype=np.int64)
    ok = count >= 5
    out[ok] = _fv[np.searchsorted(_fk, _RB_PRIME[five[ok]])]
    return out


_FLUSH = _flush_table()


def eval_board(board: Sequence[int], combos: Sequence[int] = None) -> np.ndarray:
    """Strength for combos (indices into ALL_COMBOS, default all 1326) on this 5-card
    board. Combos overlapping the board get -1."""
    board = np.asarray(sorted(board), dtype=np.int64)
    idx = np.arange(1326) if combos is None else np.asarray(list(combos), dtype=np.int64)
    br, bs = _RANK[board], _SUIT_I[board]
    # without a flush: the best five of the board's ranks and each pair of hole ranks (a rank pair no
    # real combo can hold, five of a kind, is clipped; its combos are blocked)
    seven = np.concatenate([_PAIRS, np.broadcast_to(br, (169, 5))], axis=1)
    prod = np.prod(_RANK_PRIME[seven[:, _SUBSETS]], axis=-1)
    plain = _uv[np.minimum(np.searchsorted(_uk, prod), len(_uk) - 1)].min(axis=1).reshape(13, 13)
    ra, rb = _CRA[idx], _CRB[idx]
    ranks = plain[ra, rb]
    for s in range(4):
        on = bs == s
        if on.sum() >= 3:
            suited = int(np.bitwise_or.reduce(1 << br[on]))
            bits = suited | np.where(_CSA[idx] == s, 1 << ra, 0) | np.where(_CSB[idx] == s, 1 << rb, 0)
            ranks = np.minimum(ranks, _FLUSH[bits])
    out = (7463 - ranks).astype(np.int32)
    a, b = _CA[idx], _CB[idx]
    blocked = np.isin(a, board) | np.isin(b, board)
    out[blocked] = -1
    return out
