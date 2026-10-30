"""Turn and river reports (Study's report at a closed flop or turn, gto/static/runouts.js): the next street's first
decision on every card, with each card's range equity and its chance given the two ranges.

* next_card_equities: per card off the board, the first range's equity against the second on the board with that
  card (the rest dealt: every river under a turn card), card removal between the two hands, and the card's chance
  (the share of the hand pairs that leave it live). Every runout is counted, each 5-card board ranked once: a turn
  report's 49 x 48 runouts are the 1176 boards of two cards off the flop, each counted for both of its cards.
* store_parts: a library flop's stored turn (flop_library's turn file) read for the report: per turn card the
  acting player's range frequencies at the turn's first decision and in position after a check, each combo by its
  reach entering the turn.
"""
from __future__ import annotations
import itertools
from typing import List, Optional, Sequence, Tuple
import numpy as np
from .cards import ALL_COMBOS, card_str

BOARD_CHUNK = 192     # boards ranked at once (memory: ~BOARD_CHUNK x 52 x 51 floats per array of the card groups)


def _ranked(S: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Along the last axis of S (strengths) and W (weights), for each element the sum of W over the elements of a
    lower strength (L) and over those of the same strength, itself included (E)."""
    o = np.argsort(S, axis=-1, kind="stable")
    Ss = np.take_along_axis(S, o, -1)
    Ws = np.take_along_axis(W, o, -1)
    cs = np.cumsum(Ws, axis=-1)
    m = S.shape[-1]
    idx = np.broadcast_to(np.arange(m), S.shape)
    start = np.ones(S.shape, dtype=bool)
    start[..., 1:] = Ss[..., 1:] != Ss[..., :-1]
    end = np.ones(S.shape, dtype=bool)
    end[..., :-1] = start[..., 1:]
    first = np.maximum.accumulate(np.where(start, idx, 0), axis=-1)
    last = np.flip(np.minimum.accumulate(np.flip(np.where(end, idx, m - 1), -1), axis=-1), -1)
    Ls = np.take_along_axis(cs - Ws, first, -1)
    Es = np.take_along_axis(cs, last, -1) - Ls
    L, E = np.empty_like(Ls), np.empty_like(Es)
    np.put_along_axis(L, o, Ls, -1)
    np.put_along_axis(E, o, Es, -1)
    return L, E


def _pair_sums(S: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray, pos: np.ndarray):
    """Per board (rows of S: strengths of the combos, -1 where the board holds one of its cards): the pairs of hands
    of range a against range b that share no card, weighted (den), and a's share of them at showdown, ties half
    (num). groups (52, g): the combos holding each card (index m: none); pos (m, 2, 2): each combo's two cards and
    its place in their groups."""
    nb, m = S.shape
    live = S >= 0
    A, B = a[None, :] * live, b[None, :] * live
    L, E = _ranked(S, B)
    Sp = np.concatenate([S, np.full((nb, 1), -1, dtype=S.dtype)], axis=1)
    Bp = np.concatenate([B, np.zeros((nb, 1))], axis=1)
    Lg, Eg = _ranked(Sp[:, groups], Bp[:, groups])                      # (nb, 52, g)
    Wg = Bp[:, groups].sum(axis=2)                                       # (nb, 52)
    c1, p1, c2, p2 = pos[:, 0, 0], pos[:, 0, 1], pos[:, 1, 0], pos[:, 1, 1]
    lo = L - Lg[:, c1, p1] - Lg[:, c2, p2]
    eq = E - Eg[:, c1, p1] - Eg[:, c2, p2] + B
    tot = B.sum(axis=1, keepdims=True) - Wg[:, c1] - Wg[:, c2] + B
    return (A * (lo + 0.5 * eq)).sum(axis=1), (A * tot).sum(axis=1)


def next_card_equities(board: Sequence[int], w0: np.ndarray, w1: np.ndarray):
    """(cards off the board, w0's equity against w1 on each, each card's chance): equity on the board plus the card,
    the rest of the board dealt (every river under a turn card), ties half, each pair of hands counted on the runouts
    neither blocks; a card's chance is the share of the hand pairs that leave it live (they sum to one). NaN equity
    where no pair of hands is left."""
    from .evaluator import eval_boards
    board = list(board)
    dead = set(board)
    cards = [c for c in range(52) if c not in dead]
    keep = np.array([not (set(cb) & dead) for cb in ALL_COMBOS])
    idx = np.nonzero(((w0 > 0) | (w1 > 0)) & keep)[0]
    eq, p = np.full(len(cards), np.nan), np.zeros(len(cards))
    if not len(idx):
        return cards, eq, p
    a, b = np.asarray(w0, dtype=np.float64)[idx], np.asarray(w1, dtype=np.float64)[idx]
    m = len(idx)
    members: List[List[Tuple[int, int]]] = [[] for _ in range(52)]
    pos = np.zeros((m, 2, 2), dtype=np.int64)
    for k, ci in enumerate(idx):
        for j, c in enumerate(ALL_COMBOS[ci]):
            pos[k, j] = (c, len(members[c]))
            members[c].append(k)
    g = max(1, max(len(x) for x in members))
    groups = np.full((52, g), m, dtype=np.int64)
    for c, x in enumerate(members):
        groups[c, :len(x)] = x
    if len(board) == 4:
        runs = [(c,) for c in cards]
    elif len(board) == 3:
        runs = list(itertools.combinations(cards, 2))
    else:
        raise ValueError("a turn or river report: a flop or a turn")
    num, den = np.zeros(len(runs)), np.zeros(len(runs))
    for b0 in range(0, len(runs), BOARD_CHUNK):
        S = eval_boards([board + list(r) for r in runs[b0:b0 + BOARD_CHUNK]], idx).astype(np.int64)
        num[b0:b0 + len(S)], den[b0:b0 + len(S)] = _pair_sums(S, a, b, groups, pos)
    at = {c: i for i, c in enumerate(cards)}
    N, D = np.zeros(len(cards)), np.zeros(len(cards))
    for r, x, y in zip(runs, num, den):
        for c in r:
            N[at[c]] += x
            D[at[c]] += y
    eq = np.where(D > 0, N / np.maximum(D, 1e-300), np.nan)
    p = D / max(float(D.sum()), 1e-300)
    return cards, eq, p


def store_parts(t: dict, smap: Sequence[int], path: Sequence[str], reach: Sequence[np.ndarray], board: Sequence[int],
                seats: Sequence[str]) -> List[Optional[dict]]:
    """A library flop's stored turn (a decoded turn file, flop_library.decode_turns) on the flop line `path`, for the
    turn report: [the turn's first decision, in position's after a check] (None where the file doesn't hold it), each
    {"actor", "actions", "avg", "cards": [{"card", "cards": [card], "f", "w"}]}: per turn card off the board (the hand's
    own suits; smap: canonical()'s suit map of its flop) the acting player's range frequencies there, each stored
    combo by its reach entering the turn (reach: the two players', 1326 each, on the hand's suits) less the card
    (combos the file doesn't store, under its 0.5% of the most reached, are left out; w: that reach); avg: over every
    card, each by the reach live on it."""
    from .flop_library import combo_map, turn_node
    from .solver import MEMBER_FULL
    img = combo_map(smap)
    canon = []
    for r in reach:
        x = np.zeros(len(ALL_COMBOS))
        x[img] = np.asarray(r, dtype=np.float64)
        canon.append(x)
    mp = lambda c: c // 4 * 4 + smap[c % 4]
    cards = [c for c in range(52) if c not in set(board)]
    out: List[Optional[dict]] = []
    for at in ((), ("check",)):
        rows, F, W, names, player = [], [], [], None, None
        for c in cards:
            nd = turn_node(t, path, at, mp(c))
            if nd is None:
                break
            names, player = nd["names"], nd["player"]
            w = canon[player][nd["combos"]] * (1.0 - MEMBER_FULL[mp(c)][nd["combos"]])   # (dead on the card)
            tot = float(w.sum())
            f = (nd["strat"] * w[None, :]).sum(axis=1) / max(tot, 1e-12) if tot > 0 else \
                np.full(len(names), 1.0 / len(names))
            F.append(f)
            W.append(tot)
            rows.append({"card": card_str(c), "cards": [card_str(c)], "f": [round(float(x), 4) for x in f],
                         "w": round(tot, 3)})
        if len(rows) != len(cards):
            out.append(None)
            continue
        W_ = np.asarray(W)
        avg = (np.asarray(F) * W_[:, None]).sum(axis=0) / max(float(W_.sum()), 1e-12)
        out.append({"actor": seats[player], "actions": list(names), "avg": [round(float(x), 4) for x in avg],
                    "cards": rows})
    return out
