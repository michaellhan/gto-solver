"""7-card hand strength via treys, with caching. Higher = better (treys is inverted, we flip)."""
from __future__ import annotations
from functools import lru_cache
from typing import Sequence
import numpy as np
from treys import Card, Evaluator
from .cards import ALL_COMBOS, card_str

_ev = Evaluator()
_TREYS = [Card.new(card_str(c)) for c in range(52)]


@lru_cache(maxsize=None)
def _eval5plus(board: tuple, a: int, b: int) -> int:
    return 7463 - _ev.evaluate([_TREYS[a], _TREYS[b]], [_TREYS[c] for c in board])


def eval_board(board: Sequence[int], combos: Sequence[int] = None) -> np.ndarray:
    """Strength for combos (indices into ALL_COMBOS, default all 1326) on this 5-card
    board. Combos overlapping the board get -1."""
    bt = tuple(sorted(board))
    bs = set(bt)
    idx = range(1326) if combos is None else combos
    out = np.full(len(idx), -1, dtype=np.int32)
    for k, i in enumerate(idx):
        a, b = ALL_COMBOS[i]
        if a in bs or b in bs:
            continue
        out[k] = _eval5plus(bt, a, b)
    return out
