"""Card representation: ints 0..51, rank = c // 4 (0=2 .. 12=A), suit = c % 4."""
from __future__ import annotations
import itertools
from typing import List, Tuple

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card(s: str) -> int:
    s = s.strip()
    if len(s) != 2:
        raise ValueError(f"bad card {s!r}")
    r, su = s[0].upper(), s[1].lower()
    if r not in RANKS or su not in SUITS:
        raise ValueError(f"bad card {s!r}")
    return RANKS.index(r) * 4 + SUITS.index(su)


def cards(s: str) -> List[int]:
    """Parse 'AhKd' or 'Ah Kd' or 'Ah,Kd'."""
    s = s.replace(",", " ").replace("  ", " ").strip()
    if " " not in s:
        s = " ".join(s[i:i + 2] for i in range(0, len(s), 2))
    return [card(t) for t in s.split()]


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def cards_str(cs) -> str:
    return "".join(card_str(c) for c in cs)


def rank(c: int) -> int:
    return c // 4


def suit(c: int) -> int:
    return c % 4


ALL_CARDS = list(range(52))
# All 1326 two-card combos as (lo, hi) tuples, canonical order.
ALL_COMBOS: List[Tuple[int, int]] = [(a, b) for a, b in itertools.combinations(range(52), 2)]
COMBO_INDEX = {cb: i for i, cb in enumerate(ALL_COMBOS)}


def combo_str(cb: Tuple[int, int]) -> str:
    a, b = cb
    # print high card first
    if rank(a) < rank(b):
        a, b = b, a
    return card_str(a) + card_str(b)


def hand_class(cb: Tuple[int, int]) -> str:
    """'AKs', 'AKo', 'QQ' style label for a combo."""
    a, b = cb
    ra, rb = rank(a), rank(b)
    if ra < rb:
        ra, rb = rb, ra
    if ra == rb:
        return RANKS[ra] * 2
    return RANKS[ra] + RANKS[rb] + ("s" if suit(a) == suit(b) else "o")
