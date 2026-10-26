"""Tournament prize equity (ICM, the Malmuth-Harville model) and what it does to a pot's outcomes.

A player's chance of finishing first is their share of the chips in play; given who finished first, the rest
finish second in proportion to their chips among those left, and so on. Equity is the payouts weighted by those
chances. Exact over every finishing order by dynamic programming on the set of players already placed
(O(2^n n), n players: 9 at a table plus up to 3 groups of players elsewhere stay well under a millisecond).

A pot's outcomes (a fold, a win, a loss, a split) move chips between the players in it; their prize equity moves by
less than the chips do (a chip lost costs more than a chip won gains: the "bubble factor"), and the players not in
the pot gain whenever someone busts. The solvers take each outcome at its prize-equity value (Payoffs), so a call
that is +EV in chips can be -EV in prizes.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAX_PLAYERS = 12   # players in the model: the table's, and the rest of the field grouped (see field_stacks)


def equities(stacks: Sequence[float], payouts: Sequence[float]) -> np.ndarray:
    """Each player's prize equity (in the payouts' units) by Malmuth-Harville. Players with no chips get nothing
    (or the payouts below everyone still in, shared in order of bust-out: here, equally among them)."""
    s = np.asarray(stacks, dtype=float)
    pay = np.asarray(list(payouts), dtype=float)
    n = len(s)
    if n > MAX_PLAYERS:
        raise ValueError(f"ICM over {n} players: at most {MAX_PLAYERS}")
    alive = [i for i in range(n) if s[i] > 1e-12]
    out = np.zeros(n)
    k = len(alive)
    places = pay[:k] if len(pay) >= k else np.concatenate([pay, np.zeros(k - len(pay))])
    if k == 0:
        return out
    if k == 1:
        out[alive[0]] = places[0]
    else:
        out[alive] = _mh(tuple(float(s[i]) for i in alive), tuple(float(x) for x in places))
    dead = [i for i in range(n) if s[i] <= 1e-12]
    if dead and len(pay) > k:
        out[dead] = pay[k:k + len(dead)].sum() / len(dead) if len(pay[k:k + len(dead)]) else 0.0
    return out


@lru_cache(maxsize=1 << 16)
def _mh(s: Tuple[float, ...], pay: Tuple[float, ...]) -> np.ndarray:
    """Equities of players with stacks s (all > 0) for places paying pay (len(pay) <= len(s)): one layer of the
    finishing order per paid place, over the sets of players placed above it (only places that pay are needed)."""
    n = len(s)
    st = np.asarray(s)
    total = st.sum()
    eq = np.zeros(n)
    # layer: the sets of players who took the places so far (bit masks), the chance of each, the chips they held
    masks = np.zeros(1, dtype=np.int64)
    prob = np.ones(1)
    held = np.zeros(1)
    bits = np.int64(1) << np.arange(n, dtype=np.int64)
    for place, prize in enumerate(pay):
        left = total - held
        new_m, new_p, new_h = [], [], []
        for i in range(n):
            free = (masks & bits[i]) == 0
            if not free.any():
                continue
            q = prob[free] * st[i] / left[free]
            eq[i] += prize * q.sum()
            if place + 1 < len(pay):
                new_m.append(masks[free] | bits[i]); new_p.append(q); new_h.append(held[free] + st[i])
        if place + 1 >= len(pay):
            break
        m = np.concatenate(new_m)
        u, inv = np.unique(m, return_inverse=True)
        prob = np.bincount(inv, weights=np.concatenate(new_p), minlength=len(u))
        held = np.zeros(len(u))
        held[inv] = np.concatenate(new_h)
        masks = u
    return eq


def field_stacks(others: Sequence[float], groups: int) -> List[float]:
    """The players at other tables, for the model: `groups` stand-ins, each holding the average stack of an equal
    share of them by size (the biggest stacks' stand-in first), when there are more than that."""
    o = sorted((float(x) for x in others if x > 0), reverse=True)
    if len(o) <= groups:
        return o
    out = []
    for g in range(groups):
        part = o[g * len(o) // groups:(g + 1) * len(o) // groups]
        out.append(sum(part) / len(part))
    return out


def compress_payouts(payouts: Sequence[float], players: int, model_players: int) -> List[float]:
    """Payouts for a model of `model_players` standing in for `players`: model place j covers the real places
    ((j - 1) r, j r] (r = players / model_players) and pays their average, so the money still starts the same share
    of the way down the field (a 27-player field paying 12 modelled by 12 players pays 5.3 places)."""
    pay = sorted((float(x) for x in payouts), reverse=True) + [0.0] * max(0, players - len(payouts))
    r = players / model_players
    out = []
    for j in range(model_players):
        lo, hi = j * r, (j + 1) * r
        tot = 0.0
        for k in range(int(lo), min(int(np.ceil(hi)), players)):
            tot += pay[k] * (min(hi, k + 1) - max(lo, k))
        out.append(tot / r)
    while out and out[-1] <= 1e-12:
        out.pop()
    return out


class Model:
    """A tournament spot: the table's stacks (chips behind at the start of the hand, by seat), the other players'
    and the payouts still to be paid (first place first). ICM is exact over up to MAX_PLAYERS players; a bigger field
    is modelled by stand-ins (field_stacks) with the payouts spread over them (compress_payouts)."""

    def __init__(self, stacks: Dict[str, float], payouts: Sequence[float], others: Sequence[float] = ()):
        self.seats = list(stacks)
        self.stacks = {p: float(v) for p, v in stacks.items()}
        others = [float(x) for x in others if x > 0]
        pay = sorted((float(x) for x in payouts if x > 0), reverse=True)
        if not pay:
            raise ValueError("payouts: at least one prize")
        n_real = len(self.seats) + len(others)
        self.field = field_stacks(others, max(0, MAX_PLAYERS - len(self.seats)))
        n_model = len(self.seats) + len(self.field)
        self.approx = n_real > n_model
        self.players = n_real
        # (more prizes than players left: the extra ones are already won by everyone, so they don't move equity)
        self.payouts = compress_payouts(pay[:n_real], n_real, n_model) if self.approx else pay[:n_model]
        self.prize_pool = sum(self.payouts)
        # the real prizes left (a modelled field's equities are in them: each stand-in place pays the average of the
        # places it covers)
        self.real_pool = sum(pay[:n_real])
        self.base = self.equity_of(self.stacks)
        self.key = (tuple(sorted((p, round(v, 4)) for p, v in self.stacks.items())), tuple(round(x, 4) for x in self.field),
                    tuple(round(x, 6) for x in self.payouts))

    def equity_of(self, stacks: Dict[str, float]) -> Dict[str, float]:
        vals = [max(stacks[p], 0.0) for p in self.seats] + list(self.field)
        e = equities(vals, self.payouts)
        return {p: float(e[i]) for i, p in enumerate(self.seats)}

    def after(self, start_in: Dict[str, float], won: Dict[str, float]) -> Dict[str, float]:
        """Prize equity once the hand ends: each seat put start_in chips in (from its stack at the start of the
        hand, blinds and antes included) and takes `won` chips out of the pot."""
        st = {p: self.stacks[p] - start_in.get(p, 0.0) + won.get(p, 0.0) for p in self.seats}
        return self.equity_of(st)

    @property
    def scale(self) -> float:
        """Chips in play per unit of prize: prize equity times this is in chip equivalents."""
        return (sum(self.stacks.values()) + sum(self.field)) / self.prize_pool

    def value(self, put: Dict[str, float], won: Dict[str, float]) -> Dict[str, float]:
        """Each seat's prize equity once the hand ends (after(put, won)) less its equity at the start of the hand, in
        chip equivalents: in a winner-take-all, the chips won or lost exactly."""
        eq = self.after(put, won)
        k = self.scale
        return {p: (eq[p] - self.base[p]) * k for p in self.seats}

    def showdown(self, put: Dict[str, float], a: str, b: str):
        """a and b all in against each other with every seat's chips in `put` (the others' dead): (a's values when a
        wins, when b wins), by seat. The pot is what b can match of a's chips (and a of b's); the rest goes back."""
        m = min(put[a], put[b])
        pot = sum(min(v, m) if q in (a, b) else v for q, v in put.items())
        back = {q: put[q] - m for q in (a, b) if put[q] > m + 1e-12}
        out = []
        for w in (a, b):
            won = dict(back)
            won[w] = won.get(w, 0.0) + pot
            out.append(self.value(put, won))
        return out[0], out[1]

    def call_need(self, put: Dict[str, float], hero: str, jammer: str, to_call: float, taker: Optional[str] = None):
        """Facing jammer's all-in with `put` in (hero's call not yet), the equity hero's call of to_call needs in prize
        equity, and in chips: (icm, chips). Folding gives the pot to `taker` (the jammer by default)."""
        taker = taker or jammer
        F = self.value(put, {taker: sum(put.values())})[hero]
        after = dict(put)
        after[hero] = after.get(hero, 0.0) + to_call
        w, l = self.showdown(after, hero, jammer)
        W, L = w[hero], l[hero]
        icm = (F - L) / max(W - L, 1e-12)
        m = min(after[hero], after[jammer])
        pot = sum(min(v, m) if q in (hero, jammer) else v for q, v in after.items())
        add = min(to_call, max(m - put[hero], 0.0))
        return icm, add / max(pot, 1e-12)

    def premium(self, put: Dict[str, float], hero: str, villain: str) -> float:
        """The risk premium: how much more equity than chips call for hero needs to get the rest of the effective
        stack in against villain from here (put: everyone's chips in now), as a share of 1."""
        rest = min(self.stacks[hero] - put.get(hero, 0.0), self.stacks[villain] - put.get(villain, 0.0))
        if rest <= 1e-9:
            return 0.0
        jam = dict(put)
        jam[villain] = jam.get(villain, 0.0) + rest
        icm, chips = self.call_need(jam, hero, villain, rest)
        return max(0.0, icm - chips)

    def bubble_factor(self, hero: str, villain: str) -> float:
        """What hero risks against villain over what hero gains, in prize equity per chip, for a pot where both
        put in the shorter stack (the usual "bubble factor": 1 in a winner-take-all, above 1 near the money)."""
        x = min(self.stacks[hero], self.stacks[villain])
        if x <= 0:
            return 1.0
        win = self.after({hero: x, villain: x}, {hero: 2 * x})[hero] - self.base[hero]
        lose = self.base[hero] - self.after({hero: x, villain: x}, {villain: 2 * x})[hero]
        return lose / win if win > 1e-12 else float("inf")


class HUPayoffs:
    """A heads-up postflop solve's terminals in a tournament (solver Config.icm): the solver's chips in (inv, from
    the solve's root, where they are inv0) become the hand's end stacks and so each player's prize equity, in chip
    equivalents (the prize equity's change from the start of the hand times the chips in play over the prize pool:
    in a winner-take-all it is chips exactly).

    seats: the solve's two players (OOP, IP); contrib: every seat's chips in the pot at the solve's root (the two
    players' bets on this street included; folded players' are dead money)."""

    def __init__(self, model: Model, seats: Tuple[str, str], contrib: Dict[str, float], inv0: Tuple[float, float]):
        self.model, self.seats = model, tuple(seats)
        self.contrib = {p: float(contrib.get(p, 0.0)) for p in model.seats}
        self.inv0 = (float(inv0[0]), float(inv0[1]))
        chips = sum(model.stacks.values()) + sum(model.field)
        self.scale = chips / model.prize_pool
        self.key = (model.key, self.seats,
                    tuple(sorted((p, round(v, 4)) for p, v in self.contrib.items())), self.inv0)
        self._memo: Dict[tuple, tuple] = {}

    def _end(self, inv, winner: Optional[int]) -> Tuple[float, float]:
        """Both players' values (chip equivalents) when `winner` (0, 1; None: a split) takes the pot."""
        add = (inv[0] - self.inv0[0], inv[1] - self.inv0[1])
        put = dict(self.contrib)
        for i, p in enumerate(self.seats):
            put[p] = put.get(p, 0.0) + add[i]
        pot = sum(put.values())
        won = {self.seats[winner]: pot} if winner is not None else {self.seats[0]: pot / 2, self.seats[1]: pot / 2}
        eq = self.model.after(put, won)
        return tuple((eq[p] - self.model.base[p]) * self.scale for p in self.seats)

    def fold(self, inv, folder: int) -> Tuple[float, float]:
        k = ("f", round(float(inv[0]), 6), round(float(inv[1]), 6), folder)
        if k not in self._memo:
            self._memo[k] = self._end(inv, 1 - folder)
        return self._memo[k]

    def showdown(self, inv) -> Tuple[float, float, float, float]:
        """(A0, B0, A1, B1): player p's value is A_p + B_p (P(p wins) - P(p loses)); a split taken as the mean of
        a win and a loss (prize equity is concave in chips, so a split is worth slightly more: ties are rare)."""
        k = ("s", round(float(inv[0]), 6), round(float(inv[1]), 6))
        if k not in self._memo:
            w0 = self._end(inv, 0)    # player 0 wins
            w1 = self._end(inv, 1)    # player 1 wins
            self._memo[k] = ((w0[0] + w1[0]) / 2, (w0[0] - w1[0]) / 2, (w0[1] + w1[1]) / 2, (w1[1] - w0[1]) / 2)
        return self._memo[k]


def parse_payouts(spec) -> List[float]:
    """Payouts as a list or a comma-separated string ("50,30,20", "1000, 600, 400")."""
    if isinstance(spec, str):
        spec = [x for x in spec.replace(";", ",").split(",") if x.strip()]
    return [float(x) for x in spec]
