"""Tournament prize equity (ICM, the Malmuth-Harville model) and what it does to a pot's outcomes.

A player's chance of finishing first is their share of the chips in play; given who finished first, the rest
finish second in proportion to their chips among those left, and so on. Equity is the payouts weighted by those
chances. Exact over every finishing order by dynamic programming on the set of players already placed
(O(2^n n), n players: 9 at a table plus up to 3 groups of players elsewhere stay well under a millisecond).

A pot's outcomes (a fold, a win, a loss, a split) move chips between the players in it; their prize equity moves by
less than the chips do (a chip lost costs more than a chip won gains: the "bubble factor"), and the players not in
the pot gain whenever someone busts. The solvers take each outcome at its prize-equity value (Payoffs), so a call
that is +EV in chips can be -EV in prizes.

Bounty tournaments (Bounties: knockout, progressive knockout, mystery bounty) pay cash for each player busted, to
whoever wins the pot that takes their last chips, on top of the prizes. Bounties are valued by the proportional
model (GTO Wizard's "bounty power", ICMIZER's Proportional model): every player owns their chip share of the bounty
money still in play, so a chip is worth its prize equity plus the bounty pool over the chips in play, and a bust
moves the cash it pays out of that pool into the winner's pocket (Model.value).
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


_SUBSETS: Dict[tuple, list] = {}


def _subsets(n: int, places: int) -> list:
    """Malmuth-Harville's dynamic programme over (n, places) as matrices: per place, the sets of players placed above it
    (membership (n, M)) and every way to place one more (from set, player, to set) as one-hot maps."""
    key = (n, places)
    if key not in _SUBSETS:
        import itertools
        layers = []
        cur = [()]
        for k in range(places):
            index = {m: j for j, m in enumerate(cur)}
            nxt = sorted({tuple(sorted(m + (i,))) for m in cur for i in range(n) if i not in m}) if k + 1 < places else []
            nidx = {m: j for j, m in enumerate(nxt)}
            src, who, dst = [], [], []
            for m in cur:
                for i in range(n):
                    if i in m:
                        continue
                    src.append(index[m]); who.append(i)
                    dst.append(nidx[tuple(sorted(m + (i,)))] if nxt else 0)
            T = len(src)
            mem = np.zeros((n, len(cur)))
            for j, m in enumerate(cur):
                mem[list(m), j] = 1.0
            onehot_who = np.zeros((T, n)); onehot_who[np.arange(T), who] = 1.0
            onehot_dst = np.zeros((T, max(len(nxt), 1))); onehot_dst[np.arange(T), dst] = 1.0
            layers.append((mem, np.array(src), np.array(who), onehot_who, onehot_dst))
            cur = nxt
        _SUBSETS[key] = layers
    return _SUBSETS[key]


def equities_batch(S, payouts: Sequence[float], chunk: int = 4096) -> np.ndarray:
    """equities() for many stack vectors at once: S (B, n) -> (B, n), row by row the same numbers (a player with no
    chips gets the places below everyone still in, shared equally)."""
    S = np.maximum(np.asarray(S, dtype=float), 0.0)
    B, n = S.shape
    if n > MAX_PLAYERS:
        raise ValueError(f"ICM over {n} players: at most {MAX_PLAYERS}")
    pay = [float(x) for x in payouts]
    while pay and pay[-1] <= 0:
        pay.pop()
    out = np.zeros((B, n))
    if not pay or B == 0:
        return out
    P = min(len(pay), n)
    layers = _subsets(n, P)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):   # (matmul's spurious warnings on some BLAS)
        _mh_batch(S, pay, P, layers, chunk, out)
    return out


def _mh_batch(S, pay, P, layers, chunk, out):
    B, n = S.shape
    for a in range(0, B, chunk):
        s = S[a:a + chunk]
        alive = s > 1e-12
        s = np.where(alive, s, 0.0)
        total = s.sum(axis=1, keepdims=True)
        eq = np.zeros_like(s)
        prob = np.ones((len(s), 1))
        for k in range(P):
            mem, src, who, oh_who, oh_dst = layers[k]
            left = total - s @ mem                              # (b, M): chips not yet placed
            q = prob[:, src] * s[:, who] / np.where(left[:, src] > 1e-12, left[:, src], np.inf)
            eq += pay[k] * (q @ oh_who)
            if k + 1 < P:
                prob = q @ oh_dst
        k_alive = alive.sum(axis=1)
        # (fewer players with chips than places: the ones busted share the places below, as equities())
        for r in np.nonzero((k_alive < len(pay)) & (k_alive < n) & (k_alive > 0))[0]:
            kk = int(k_alive[r]); dead = ~alive[r]
            eq[r, dead] = sum(pay[kk:kk + int(dead.sum())]) / dead.sum()
        eq[k_alive == 0] = 0.0
        out[a:a + chunk] = eq


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


BOUNTY_FORMATS = {"ko": "KO", "pko": "PKO", "mystery": "mystery bounty"}
BUST_EPS = 1e-6     # (bb: a stack this close to nothing after the hand is busted)


class Bounties:
    """A bounty tournament's bounties, in the payouts' units (money).

    fmt: "ko" (knockout: the whole bounty goes to whoever busts the player), "pko" (progressive knockout: half is paid
    at once, half is added to the eliminator's own bounty, which stays in play) or "mystery" (each bust draws an
    envelope from a pool: worth the average envelope left).
    seats: each seat's bounty (ko, pko); field: the players at other tables' average bounty (None: the table's average).
    envelope: mystery's average envelope left; pool: the money left in envelopes (with envelope None: pool / left is the
    envelope, `left` the bounties still to be drawn). A mystery pool not given is envelope x (players left - 1): one
    envelope per bust still to come."""

    def __init__(self, fmt: str, seats: Optional[Dict[str, float]] = None, field: Optional[float] = None,
                 envelope: Optional[float] = None, pool: Optional[float] = None, left: Optional[float] = None):
        fmt = str(fmt).lower()
        if fmt not in BOUNTY_FORMATS:
            raise ValueError(f"bounty format {fmt!r}: one of {', '.join(BOUNTY_FORMATS)}")
        self.fmt = fmt
        self.seats = {p: max(0.0, float(v)) for p, v in (seats or {}).items()}
        self.field = None if field is None else max(0.0, float(field))
        if fmt == "mystery" and envelope is None and pool and left:
            envelope = float(pool) / float(left)
        self.envelope = None if envelope is None else max(0.0, float(envelope))
        self.pool = None if pool is None else max(0.0, float(pool))

    def __bool__(self):
        """Anything at stake: a bounty on someone at the table (or an envelope to draw)."""
        if self.fmt == "mystery":
            return bool(self.envelope)
        return any(v > 0 for v in self.seats.values())


class Model:
    """A tournament spot: the table's stacks (chips behind at the start of the hand, by seat), the other players'
    and the payouts still to be paid (first place first). ICM is exact over up to MAX_PLAYERS players; a bigger field
    is modelled by stand-ins (field_stacks) with the payouts spread over them (compress_payouts).

    bounties (Bounties): a bounty tournament. Each player owns their chip share of the bounty money in play (the pool,
    P: every bounty on a head, or a mystery's envelopes left; the proportional model), and a bust pays its cash (a KO
    bounty, half a PKO bounty, a mystery envelope) out of the pool to whoever wins the pot that takes the player's last
    chips (only a player who covers them can). A seat's value of an outcome is then its prize equity's change plus
    its share of the pool after (its chips over the chips in play) less before, plus the cash it won, in chip
    equivalents at chips / (prizes + pool) per unit (bounty_k: a chip is worth its prizes and its share of the pool;
    in a winner-take-all, chips exactly). The half of a PKO bounty added to the winner's head stays in the pool, so
    it is worth the winner's chip share of it after the hand (as everyone's). None, or nothing at stake: prizes only,
    the same numbers as without."""

    def __init__(self, stacks: Dict[str, float], payouts: Sequence[float], others: Sequence[float] = (),
                 bounties: Optional[Bounties] = None):
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
        self.bounty = None
        self.assumed_prizes = False     # (the advisor's: no payouts given, the prizes taken as the bounties in play)
        if bounties:
            self._bounty_setup(bounties, others)

    def _bounty_setup(self, b: Bounties, others: List[float]):
        """The bounty model's numbers: each seat's bounty and the cash its bust pays, the pool P in play, the chips in
        play (the table's and every other player's, not the model's stand-ins: a player's share of the pool is theirs
        of the real chips), and bounty_k, chip equivalents per unit of money."""
        self.bounty = b
        n_others = len(others)
        if b.fmt == "mystery":
            e = b.envelope or 0.0
            self.bounties = {p: e for p in self.seats}
            self.cash = dict(self.bounties)
            pool = b.pool if b.pool else e * max(self.players - 1, 1)
        else:
            self.bounties = {p: b.seats.get(p, 0.0) for p in self.seats}
            avg = b.field if b.field is not None else sum(self.bounties.values()) / max(len(self.seats), 1)
            self.cash = {p: v * (0.5 if b.fmt == "pko" else 1.0) for p, v in self.bounties.items()}
            pool = sum(self.bounties.values()) + n_others * avg
        self.bounty_pool = float(pool)
        self.chips_real = sum(self.stacks.values()) + sum(others)
        self.rho = self.bounty_pool / max(self.chips_real, 1e-12)      # the pool's money per chip
        self.bounty_k = 1.0 / (1.0 / self.scale + self.rho)              # chips per unit: GTO Wizard's bounty power
        self.key = self.key + (("bounty", b.fmt, tuple(sorted((p, round(v, 6)) for p, v in self.cash.items())),
                                round(self.bounty_pool, 6), round(self.chips_real, 4)),)

    def busts(self, put: Dict[str, float], won: Dict[str, float], winners) -> list:
        """The busts of an outcome (put, won as after()) for Model.value's `ko`: each seat with no chips left, its
        bounty to winners(seat) ({seat: share}: who won the pot that took its last chips)."""
        out = []
        for p in self.seats:
            if p in put and self.stacks[p] - put[p] + won.get(p, 0.0) <= BUST_EPS:
                w = winners(p)
                if w:
                    out.append((p, w))
        return out

    def _bounty_money(self, end: Dict[str, float], ko) -> Dict[str, float]:
        """Each seat's bounty money from an outcome (end: stacks after the hand; ko: [(busted seat, {winner: share})]):
        the cash it won plus its share of the pool after (end / chips in play x (P - the cash paid)) less before."""
        cash: Dict[str, float] = {}
        paid = 0.0
        for busted, winners in ko or ():
            c = self.cash.get(busted, 0.0)
            paid += c
            for w, sh in winners.items():
                cash[w] = cash.get(w, 0.0) + sh * c
        P, T = self.bounty_pool, self.chips_real
        return {p: cash.get(p, 0.0) + (max(end[p], 0.0) * (P - paid) - self.stacks[p] * P) / T for p in self.seats}

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

    def value(self, put: Dict[str, float], won: Dict[str, float], ko=()) -> Dict[str, float]:
        """Each seat's prize equity once the hand ends (after(put, won)) less its equity at the start of the hand, in
        chip equivalents: in a winner-take-all, the chips won or lost exactly. A bounty tournament adds each seat's
        bounty money (ko: the busts, [(seat, {winner: share})], Model.busts)."""
        eq = self.after(put, won)
        if self.bounty is None:
            k = self.scale
            return {p: (eq[p] - self.base[p]) * k for p in self.seats}
        end = {p: self.stacks[p] - put.get(p, 0.0) + won.get(p, 0.0) for p in self.seats}
        bm = self._bounty_money(end, ko)
        k = self.bounty_k
        return {p: (eq[p] - self.base[p] + bm[p]) * k for p in self.seats}

    def value_batch(self, put, won, kos=None) -> np.ndarray:
        """value() for many outcomes: put, won (B, seats) in self.seats' order -> (B, seats); kos: each outcome's
        busts (as value()'s ko, by seat name), for a bounty tournament."""
        st = np.array([self.stacks[p] for p in self.seats])[None, :] - np.asarray(put) + np.asarray(won)
        vals = np.concatenate([np.maximum(st, 0.0), np.broadcast_to(np.array(self.field, dtype=float), (len(st), len(self.field)))], axis=1)
        eq = equities_batch(vals, self.payouts)[:, :len(self.seats)]
        if self.bounty is None:
            return (eq - np.array([self.base[p] for p in self.seats])[None, :]) * self.scale
        bm = np.array([[d[p] for p in self.seats] for d in
                       (self._bounty_money(dict(zip(self.seats, row)), kos[i] if kos else ()) for i, row in enumerate(st))])
        return (eq - np.array([self.base[p] for p in self.seats])[None, :] + bm.reshape(eq.shape)) * self.bounty_k

    def showdown(self, put: Dict[str, float], a: str, b: str):
        """a and b all in against each other with every seat's chips in `put` (the others' dead): (a's values when a
        wins, when b wins), by seat. The pot is what b can match of a's chips (and a of b's); the rest goes back. In a
        bounty tournament the loser busts when the winner covers them, its bounty to the winner."""
        m = min(put[a], put[b])
        pot = sum(min(v, m) if q in (a, b) else v for q, v in put.items())
        back = {q: put[q] - m for q in (a, b) if put[q] > m + 1e-12}
        out = []
        for w in (a, b):
            won = dict(back)
            won[w] = won.get(w, 0.0) + pot
            ko = self.busts(put, won, lambda p, w=w: {w: 1.0} if p in (a, b) and p != w else None) \
                if self.bounty is not None else ()
            out.append(self.value(put, won, ko))
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
        call = rest
        if self.bounty is not None:
            # (both all in for the shorter stack, so the one covered busts when it loses and pays its bounty: the
            # rest behind alone leaves a seat with less in short of matching the other's chips)
            eff = min(self.stacks[hero], self.stacks[villain])
            jam[villain] = max(put.get(villain, 0.0), eff)
            call = max(eff - put.get(hero, 0.0), 0.0)
        icm, chips = self.call_need(jam, hero, villain, call)
        # (bounties can make it a discount: a bust of a player hero covers pays hero their bounty)
        return icm - chips if self.bounty is not None else max(0.0, icm - chips)

    def bust_value(self, hero: str, villain: str) -> Optional[Tuple[float, float]]:
        """What busting villain is worth to hero beyond the chips, all in against them (hero covering): (in chip
        equivalents, in money): the cash it pays less hero's share of it after the hand (hero owned its chip share of
        that bounty already, in the pool). None if hero doesn't cover villain or there's no bounty."""
        if self.bounty is None or self.stacks[hero] + BUST_EPS < self.stacks[villain] or self.stacks[villain] <= 0:
            return None
        c = self.cash.get(villain, 0.0)
        share = (self.stacks[hero] + self.stacks[villain]) / self.chips_real
        money = c * (1.0 - share)
        return money * self.bounty_k, money

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
        m = self.model
        if m.bounty is not None:
            # (a bounty tournament: the loser, all in, busts and its bounty goes to the winner; a split busts nobody)
            ko = m.busts(put, won, lambda p: {self.seats[winner]: 1.0} if winner is not None and p in self.seats else None)
            v = m.value(put, won, ko)
            return tuple(v[p] for p in self.seats)
        eq = m.after(put, won)
        return tuple((eq[p] - m.base[p]) * self.scale for p in self.seats)

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


class MWPayoffs:
    """A multiway postflop solve's terminals in a tournament (multiway_solver MWConfig.icm), in chip equivalents as
    HUPayoffs'. A pot contested by three or four players has more outcomes than a share can weigh: who wins when a
    player loses, and with a player all-in for less who takes the side pot. Each terminal's value is kept linear in
    the solver's shares (so the CFR's contractions stay as they are) by pricing every outcome at its prize equity and
    averaging over the orders of the live players' hands: player p's value is
        wJ m + sum_k w_k s_k
    with m the deals' weight and s_k p's share of pot layer k (the main pot, then the side pots, MultiwaySolver._layers:
    p wins layer k when its hand beats every other live hand whose chips reach it). Winning layer k means winning
    every layer above it p's chips reach, so the outcomes are "p's best layer won is k" (k = 0, 1, ...) or none:
    G_k is p's prize value of the first, averaged over the orders of the hands that give it, wJ = G_none and
    w_k = G_k - G_{k+1} (exact in chips: w_k is layer k's chips, wJ minus p's chips in). A player not live gets the
    mean over every order. What it leaves out is the bystander's part of a loss: which of the others wins when p
    loses is taken as even. (Values count from the start of the hand; the solver moves each player's to its own
    origin: MultiwaySolver._icm_setup.)

    seats: the solve's players (its order), by seat name; contrib: every seat's chips in at the solve's root (before
    its street; the folded seats' dead money with this street's bets), so a seat's chips in at a terminal are its
    contrib plus the solver's c (the root street's bets on)."""

    def __init__(self, model: Model, seats: Sequence[str], contrib: Dict[str, float]):
        self.model, self.seats = model, tuple(seats)
        self.contrib = {p: float(contrib.get(p, 0.0)) for p in model.seats}
        self.key = (model.key, self.seats, tuple(sorted((p, round(v, 4)) for p, v in self.contrib.items())))
        pos = {p: i for i, p in enumerate(model.seats)}
        self._ix = np.array([pos[p] for p in self.seats])
        self._put0 = np.array([self.contrib[p] for p in model.seats])
        self._memo: Dict[tuple, np.ndarray] = {}
        self.evals = 0          # outcomes priced (each one ICM over the table)

    def __repr__(self):
        return f"MWPayoffs({self.key!r})"

    def weights(self, states) -> List[np.ndarray]:
        """Per state (live, c, layers): live the players contesting the pot (solver indices), c each player's chips in
        from the root street's start, layers [(chips, players)] the pot's layers (MultiwaySolver._layers): W (np, 1 +
        len(layers)), each player's wJ then w_k (a player not live: wJ only, the mean over the orders)."""
        import itertools
        npl = len(self.seats)
        keys, todo = [], {}
        for live, c, layers in states:
            k = (tuple(live), tuple(round(float(x), 6) for x in c),
                 tuple((round(float(a), 6), tuple(e)) for a, e in layers))
            keys.append(k)
            if k not in self._memo and k not in todo:
                todo[k] = (tuple(live), [float(x) for x in c], [(float(a), tuple(e)) for a, e in layers])
        if todo:
            puts, wons, index, kos = [], [], [], []
            model = self.model
            bounty = model.bounty is not None
            stk = np.array([model.stacks[p] for p in model.seats])
            for k, (live, c, layers) in todo.items():
                put = self._put0.copy()
                put[self._ix] += np.asarray(c)
                tot = sum(a for a, _ in layers)
                sc = put.sum() / tot if tot > 1e-12 else 0.0     # (the layers' chips are the seats' in, as HUPayoffs)
                perms = list(itertools.permutations(live))
                for perm in perms:
                    won = np.zeros(len(put))
                    for amt, elig in layers:
                        w = next(x for x in perm if x in elig)
                        won[self._ix[w]] += amt * sc
                    puts.append(put); wons.append(won)
                    if bounty:
                        # a live player left with nothing busts: its bounty to whoever won the top layer its chips
                        # reach (the pot that took its last chips; that winner's chips reach it too, so covered it)
                        ko = []
                        for p in live:
                            if stk[self._ix[p]] - put[self._ix[p]] + won[self._ix[p]] <= BUST_EPS:
                                top = max(j for j, (_, e) in enumerate(layers) if p in e)
                                w = next(x for x in perm if x in layers[top][1])
                                if w != p:
                                    ko.append((self.seats[p], {self.seats[w]: 1.0}))
                        kos.append(ko)
                index.append((k, live, layers, perms))
            V = model.value_batch(np.array(puts), np.array(wons), kos if bounty else None)[:, self._ix]   # (outcomes, np)
            self.evals += len(V)
            at = 0
            for k, live, layers, perms in index:
                Vk = V[at:at + len(perms)]
                at += len(perms)
                L = len(layers)
                W = np.zeros((npl, 1 + L))
                W[:, 0] = Vk.mean(axis=0)
                for p in live:
                    top = max(j for j, (_, e) in enumerate(layers) if p in e)
                    # p's best layer won in each order: the first whose other players all rank below p
                    best = []
                    for perm in perms:
                        above = set(perm[:perm.index(p)])
                        best.append(next((j for j in range(top + 1) if not above & set(layers[j][1])), top + 1))
                    best = np.array(best)
                    G = [float(Vk[best == j, p].mean()) if (best == j).any() else 0.0 for j in range(top + 2)]
                    W[p, 0] = G[top + 1]
                    for j in range(top + 1):
                        W[p, 1 + j] = G[j] - G[j + 1]
                self._memo[k] = W
        return [self._memo[k] for k in keys]


def parse_payouts(spec) -> List[float]:
    """Payouts as a list or a comma-separated string ("50,30,20", "1000, 600, 400")."""
    if isinstance(spec, str):
        spec = [x for x in spec.replace(";", ",").split(",") if x.strip()]
    return [float(x) for x in spec]
