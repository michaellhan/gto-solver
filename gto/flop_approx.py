"""Approximate flop answers for a flop the library doesn't store, read off the spot's stored flops most like it.

A spot of the flop library (gto/flop_library.py) stores a representative set of flops (552 of the 1755 for the
flatted opens and the 3bet pots); another flop of the spot is solved live, and until that solve answers the page had
nothing to show. Here its stored decisions are read off the K stored flops most like it (`neighbours`: the same
pairing and suit pattern, the ranks nearest), each carried over by card substitution (`card_map`: the target flop's
cards onto the stored flop's, the other ranks in order, the suits as the two patterns pair them, so a combo's image
is the hand it plays like there: KcJd6h from KcJd5h plays a 6 as the 5, a 5 as the 6, 54 as 64), and their
strategies averaged per combo over the images that hold its hand class (`predict`: the made hand and the draw; all
of them where none does). A heads-up pot the library holds no spot for, or one at another stack depth, is read off
the spot whose ranges are nearest, where one covers it (`near_spot`), its entry of the same flop where it stores it.
Numpy only (the browser's worker runs it); nothing is solved.
"""
from __future__ import annotations
import itertools
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .cards import ALL_COMBOS, COMBO_INDEX, cards as parse_cards
from .solver import CA_FULL, CB_FULL, COMBO_TABLE

N_COMBOS = len(ALL_COMBOS)


def card_map(target: Sequence[int], source: Sequence[int]) -> Optional[Tuple[np.ndarray, int]]:
    """(m, distance): m[c] the source card standing for target card c (a bijection of the 52 cards taking the target
    flop onto the source flop, the flop's rank order and pairing kept, suits as the two flops' suit patterns pair
    them, the other ranks in order), distance the flops' summed rank gaps; None when their pairing or suit pattern
    differ."""
    t = sorted(target, key=lambda c: -c)
    best = None
    for pi in itertools.permutations(range(3)):
        s = [source[i] for i in pi]
        if any((t[i] // 4 > t[j] // 4) != (s[i] // 4 > s[j] // 4) or (t[i] // 4 == t[j] // 4) != (s[i] // 4 == s[j] // 4)
               for i in range(3) for j in range(i + 1, 3)):
            continue
        suit: Dict[int, int] = {}
        if any(suit.setdefault(t[i] % 4, s[i] % 4) != s[i] % 4 for i in range(3)) or len(set(suit.values())) != len(suit):
            continue
        rest = [x for x in range(4) if x not in suit.values()]
        for x in range(4):
            if x not in suit:
                suit[x] = rest.pop(0)
        rank = {t[i] // 4: s[i] // 4 for i in range(3)}
        rank.update(zip([r for r in range(13) if r not in rank], [r for r in range(13) if r not in rank.values()]))
        d = sum(abs(t[i] // 4 - s[i] // 4) for i in range(3))
        if best is None or d < best[1]:
            best = (np.array([rank[c // 4] * 4 + suit[c % 4] for c in range(52)], dtype=np.int64), d)
    return best


def _pattern(board: Sequence[int]) -> tuple:
    ranks = sorted((c // 4 for c in board), reverse=True)
    suits = sorted((sum(1 for c in board if c % 4 == s) for s in range(4)), reverse=True)
    return (3 - len(set(ranks)), suits[0])


def neighbours(names: Sequence[str], board: Sequence[int], k: int) -> List[Tuple[str, np.ndarray, int]]:
    """The k stored flops (names) most like `board`: [(name, card map, distance)], least distance first (then the
    higher cards' gaps), only flops of the same pairing and suit pattern."""
    pat = _pattern(board)
    hi = sorted(board, reverse=True)
    out = []
    for nm in names:
        src = parse_cards(nm)
        if _pattern(src) != pat:
            continue
        r = card_map(board, src)
        if r is None:
            continue
        key = (r[1], [abs(a // 4 - b // 4) for a, b in zip(hi, sorted(src, reverse=True))])
        out.append((key, nm, r[0]))
    out.sort(key=lambda x: x[0])
    return [(nm, m, key[0]) for key, nm, m in out[:k]]


def combo_image(m: np.ndarray) -> np.ndarray:
    """img[i]: the combo that combo i becomes under the card map m."""
    return COMBO_TABLE[m[CA_FULL], m[CB_FULL]]


K = 3                      # stored flops an answer reads (bench: 1 / 2 / 3 / 4 nearest)
MAX_DIST = 6               # ... at most this far (summed rank gaps): a trips board read off 4h4d4c for AhAdAc gave up 9-24%


def sources(lib, spot: dict, canon: Sequence[int], k: Optional[int] = None, max_dist: Optional[int] = None,
            same: bool = False) -> list:
    """[(decoded entry, card map, weight, name, distance)] of the spot's stored flops most like the canonical flop
    `canon` (not itself), weight 1 / (1 + distance), k (K) of them at most max_dist (MAX_DIST) away; the entries
    fetched through the library (lib.entry). same: another spot's (near_spot), whose own entry for the flop, where
    it stores it, is the one source."""
    k = K if k is None else k
    max_dist = MAX_DIST if max_dist is None else max_dist
    out = []
    for nm, m, d in neighbours(sorted(spot.get("flops", {})), canon, k + 1):
        if len(out) == k or d > max_dist:
            break
        if d == 0:
            if same:
                e = lib.entry(spot, nm)
                return [(e, m, 1.0, nm, 0)] if e is not None else []
            continue
        e = lib.entry(spot, nm)
        if e is not None:
            out.append((e, m, 1.0 / (1.0 + d), nm, d))
    return out


# A heads-up pot the library holds no spot for (the HJ's or the UTG's open flatted by the SB, the 3bet pots of the UTG
# and the HJ) is read off the library spot whose ranges are nearest, where one covers it (near_spot): the same pot and
# stack (so the same tree), each player's range there holding at least NEAR_COVER of theirs here (by weight) and the
# two ranges' L1 distances (each normalized) summing to at most NEAR_SPOT_DIST. scripts/bench_flop_approx.py's
# protocol "spot" (a stored spot read off another): the pairs inside both gave up 0.01-0.38% of the pot per flop
# (the source's entry of the same flop), those under 0.9 0.36-1.77.
# Another stack depth (the library's spots are 100bb deep) is read off a spot of the same pot whose stack-to-pot ratio
# is within NEAR_SPR of this pot's, where this pot's is at least NEAR_MIN_SPR (the hand's tree, built at its own stack:
# a node the entry stores with other actions, an all-in where the entry bets, isn't read). scripts/bench_flop_depth.py
# (ten spots, six flops each, every depth's flop solved in the Accurate tree and each combo's mix priced by its EVs, %
# of the pot): inside both and NEAR_COVER, at 60-150bb, the 100bb solve of the same flop gave up 0.10 (single-raised
# pots, mean) and 0.10 (3bet pots), the library's entry 0.23 and 0.22 (at 100bb, the measure's floor, 0.17 and 0.14);
# the entry at 40 and 50bb (ratio 0.38 and 0.49; nine single-raised spots, eight flops each, those inside NEAR_COVER)
# 0.30 and 0.32 (90th percentile 0.46, 0.47), at 175bb (1.77-1.83; eight spots, six flops) 0.35 (0.58), at 200bb
# (2.03) 0.25 (0.36: BTN vs BB and SB vs BB alone cover 90%); outside, 3bet pots at 200bb (2.11) 0.55-0.75, at 60bb
# (SPR 2.4) 1.3-1.9 with most nodes another tree's.
NEAR_COVER = 0.9
NEAR_SPOT_DIST = 2.0
NEAR_SPR = (0.35, 2.05)
NEAR_MIN_SPR = 3.0
# At a bigger table (8- and 9-max), a pot of the same pot and stack that no spot covers by NEAR_COVER (the CO's open
# flatted by the BTN, UTG's flatted by the CO, the LJ's by the HJ, UTG1's by the CO, the 3bets of UTG's open) is read
# off the nearest spot covering each range by NEAR_TABLE_COVER, at most NEAR_TABLE_DIST apart: the hands that spot's
# ranges hold (the rest are solved live; filled with their hand class's average, as for another game's pot, they gave
# up 0.38-0.50 in single-raised pots and 0.61-0.76 in the 3bet pots). scripts/bench_flop_formats.py (12 flops each, as
# below): the CO vs BTN off HJ vs CO (cover 0.75, distance 1.12) 0.275, UTG vs CO and LJ vs HJ and UTG1 vs CO off UTG vs
# HJ (0.81, 0.66; 0.86, 0.35; 0.90, 0.42) 0.249 / 0.217 / 0.210, the BTN's, CO's and HJ's 3bets of UTG off the CO's of
# the HJ and the HJ's of UTG (0.85-0.88, 0.60-0.75) 0.103-0.141 (worst flop 0.24-0.78 in all), against the live Fast
# solve's 0.170-0.197 at its 50th iteration, which replaces it; 13-28% of the hands (by reach) left to it. Left out:
# UTG vs HJ (0.71; 0.371, worst flop 1.57), the SB flatting UTG (0.71, 1.82; 0.405). Not at 6-max or a shorter
# table (near_candidates' players): those pots stay as they were.
NEAR_TABLE_COVER = 0.75
NEAR_TABLE_DIST = 1.2
# Another game's pot (antes or a straddle add dead money; heads-up the blinds play: a pot no spot has) is read off a
# spot of another pot where the same seat raised last preflop (near_spot's aggr: out of position, or in position, in
# both), the stack-to-pot ratio within NEAR_SPR as above, each range covered by NEAR_POT_COVER and the distances
# summing to at most NEAR_POT_DIST (6-max 100bb's squeezes and cold calls of a 3bet, 1.14-1.9 from the 3bet pots of
# another size, stay unread). The hands that spot's ranges don't hold play their hand class's average there
# (class_fill). scripts/bench_flop_formats.py (20 flops each, every game's flop solved at its own ranges in the
# Accurate tree, 300 iterations; each combo's mix priced by its EVs, % of the pot per flop, mean, the filled hands
# in): ante CO vs BB off BTN vs BB 0.21, straddled BTN vs UTG off BTN vs BB 0.19, CO vs UTG off HJ vs BB 0.17; the
# live Fast solve at its 50th iteration 0.19-0.21.
NEAR_POT_DIST = 1.0
NEAR_POT_COVER = 0.9
# Two more tiers, each read only where no spot of a tier before it qualifies (near_spot: the tier, then the distance):
# 1, ranges at most NEAR_POT_CLOSE_DIST apart that cover each of this pot's by NEAR_POT_CLOSE_COVER (the ante's SB vs
# BB and HJ vs BB, the blinds' 3bets of the BTN, the straddler's call of the BB's raise: 0.22-0.40 against the live
# solve's 0.18-0.33 at its 50th iteration, worst flop 0.30-0.64); 2, a single-raised pot the big blind defends
# whose stack-to-pot ratio is within NEAR_POT_WIDE_SPR of this one's, covering each range by NEAR_POT_WIDE_COVER and
# at most NEAR_POT_WIDE_DIST apart (heads-up's raised pot off BTN vs BB, cover 0.59, distance 1.36: 0.257, the live
# solve 0.199; the ante's BTN vs BB 0.234 against 0.198; the ante's SB flatting UTG's, the HJ's, the CO's and the
# BTN's opens off the big blind's defence of each, cover 0.91-0.96, distance 1.44-1.50: 0.176-0.220, worst flop
# 0.27-0.36, against the live solve's 0.218-0.246; nothing else lies 1.4-1.55 apart). Left out, measured worse:
# ranges 0.53-0.64 apart in the ante's UTG vs HJ (cover 0.84; 0.443, worst flop 0.96) and HJ's 3bet of UTG
# (0.87; 0.423, 0.93), a straddled BTN flatting the CO (0.79, 0.61; 0.371, 0.87), and the 3bet pots under 0.75 cover
# or 1.4 apart (above).
NEAR_POT_CLOSE_COVER = 0.75
NEAR_POT_CLOSE_DIST = 0.5
NEAR_POT_WIDE_COVER = 0.55
NEAR_POT_WIDE_DIST = 1.55
NEAR_POT_WIDE_SPR = (0.8, 1.25)
# A pot of another table size that plays as one the library holds (8- and 9-max BTN, SB, CO and HJ vs the BB, BTN and
# CO vs SB, the blinds' 3bets of the BTN and the CO, the BB's of the SB: the same pot and stack, the ranges entering the
# flop a few combos apart) is read as the library's own solve (near_exact: flop_library.FlopLibrary.solution with that
# spot: instant, its EVs and stored turns), marked borrowed, where each range there holds at least NEAR_EXACT_COVER of
# this one's (by weight; the combos it doesn't hold are left out, and your own hand off it is solved live) and the two
# ranges' distances (range_dist) sum to at most NEAR_EXACT_DIST. scripts/bench_flop_formats.py (as above, 20 flops
# each): the library's entry gave up as much for the 8-max pot as for its own 6-max one (BTN vs BB 0.120 / 0.119, SB
# vs BB 0.212 / 0.210, CO vs BB 0.128 / 0.128, HJ vs BB 0.126 / 0.127, BTN vs SB 0.164 / 0.165, CO vs SB 0.178 /
# 0.162, the 3bet pots 0.081-0.251 / 0.082-0.248); the 6-max pot's own solve read for the 8-max one 0.002-0.025
# (distance 0.011-0.145). Further out (12 flops each): UTG, UTG1 and LJ vs BB off UTG vs BB (distance 0.46, 0.57,
# 0.31) 0.132 / 0.130 / 0.123 against 6-max UTG's 0.124, LJ vs BTN, HJ vs CO and LJ vs CO off the flatted opens
# (0.21-0.25) 0.140-0.182 / 0.146-0.180, HJ's 3bet of LJ off HJ's of UTG (0.17) 0.064 / 0.057; the 6-max solve
# read 0.009-0.039. Only another table size's spots: at 6-max a pot the library doesn't hold is its own (BTN's 3bet
# of UTG, 0.29 from CO's of HJ, gives up 0.093 against that spot's own 0.053), read approximately and solved.
NEAR_EXACT_COVER = 0.97
NEAR_EXACT_DIST = 0.6
# At a bigger table (8- and 9-max) the cover may be NEAR_EXACT_TABLE_COVER (the hands off the spot's ranges, 5-12% of
# them by reach, are solved live). As above, 12 flops each: HJ vs BTN off HJ vs CO (cover 0.92, distance 0.35) 0.165
# against that spot's own 0.149, UTG and UTG1 vs BTN off UTG vs BTN (0.91, 0.45; 0.93, 0.45) 0.148 / 0.143 against
# 0.146; the 3bet pots of the LJ, HJ and UTG1 by the CO and BTN off the CO's of the HJ and the HJ's of UTG (0.91-0.97,
# 0.28-0.60) 0.067-0.114 against 0.053 / 0.057; the 6-max solve read for the 8-max pot 0.017-0.055 (worst flop 0.172).
# Shorter tables (2-5 players) keep NEAR_EXACT_COVER: unmeasured.
NEAR_EXACT_TABLE_COVER = 0.9
_SETUPS: Dict[tuple, Optional[dict]] = {}
_NEAR: Dict[tuple, Optional[str]] = {}
_EXACT: Dict[tuple, Optional[dict]] = {}


def _spot_ranges(spot: dict) -> Optional[dict]:
    """A library spot's ranges entering the flop (flop_library.spot_setup of its line, in its game: the index's
    "game" of another format's spot), None where they no longer match the fingerprints the index holds (the charts
    changed since it was built)."""
    from . import flop_library as fl
    line = spot.get("line")
    game = {k: v for k, v in (spot.get("game") or {}).items() if k in ("players", "stack", "ante", "straddle")}
    key = (line, tuple(sorted(game.items())))
    if key not in _SETUPS:
        try:
            st = fl.spot_setup(line, **game) if line else None
        except Exception:
            st = None
        if st is not None and list(spot.get("fp") or ()) != [fl.range_fp(st["full0"]), fl.range_fp(st["full1"])]:
            st = None
        _SETUPS[key] = st
    return _SETUPS[key]


def range_cover(t: np.ndarray, s: np.ndarray) -> float:
    """The share of range t's weight on combos range s holds."""
    tot = float(t.sum())
    return float((t * (s > 0)).sum()) / tot if tot > 0 else 0.0


def range_dist(t: np.ndarray, s: np.ndarray) -> float:
    """L1 distance of the two ranges, each normalized to 1 (0 to 2)."""
    return float(np.abs(t / max(float(t.sum()), 1e-12) - s / max(float(s.sum()), 1e-12)).sum())


def spot_aggressor(spot: dict) -> Optional[int]:
    """Who raised last preflop in a library spot (0 out of position, 1 in position; None: nobody, or not one of its
    two seats), from its line and seats."""
    seats = list(spot.get("seats") or ())
    last = None
    for tok in str(spot.get("line") or "").split(","):
        pos, _, act = tok.partition(":")
        if act.startswith("r"):
            last = pos
    return seats.index(last) if last in seats else None


def near_candidates(lib, full0: np.ndarray, full1: np.ndarray, pot: float, stack: float,
                    exclude: Optional[str] = None, aggr: Optional[int] = None, players: int = 6) -> List[dict]:
    """Every library spot as near_spot weighs it for a heads-up pot: {"id", "other" (another pot or stack), "cover"
    (the lesser of the two ranges' cover), "dist" (summed), "why" (None where it qualifies, else the first gate it
    fails: "pot" (another pot, and the same preflop raiser isn't in the same position or isn't known), "spr" (another
    stack-to-pot ratio outside NEAR_SPR or this pot's under NEAR_MIN_SPR), "ranges" (its charts changed), "cover",
    "dist"); "tier" where a lower one is read first (another game's pot's tiers; a bigger table's pot under NEAR_COVER:
    1)}. players: the hand's table size (over 6, the same pot and stack is also read at NEAR_TABLE_COVER)."""
    out = []
    for sp in lib.index().get("spots", []):
        if sp["id"] == exclude:
            continue
        same_pot = round(float(sp["pot"]), 2) == round(pot, 2)
        other = not same_pot or round(float(sp["stack"]), 2) != round(stack, 2)   # (another pot or depth)
        row = {"id": sp["id"], "other": other, "cover": None, "dist": None, "why": None}
        out.append(row)
        if not same_pot and (aggr is None or spot_aggressor(sp) != aggr):
            row["why"] = "pot"
            continue
        if other and (stack / pot < NEAR_MIN_SPR or not
                      NEAR_SPR[0] <= (stack / pot) / (float(sp["stack"]) / float(sp["pot"])) <= NEAR_SPR[1]):
            row["why"] = "spr"
            continue
        st = _spot_ranges(sp)
        if st is None:
            row["why"] = "ranges"
            continue
        row["cover"] = min(range_cover(full0, st["full0"]), range_cover(full1, st["full1"]))
        row["dist"] = range_dist(full0, st["full0"]) + range_dist(full1, st["full1"])
        if same_pot:
            if row["cover"] < NEAR_COVER:
                if players > 6 and not other and row["cover"] >= NEAR_TABLE_COVER and row["dist"] <= NEAR_TABLE_DIST:
                    row["tier"] = 1   # (a bigger table's pot of the same pot and stack: NEAR_TABLE_COVER)
                else:
                    row["why"] = "cover"
            elif row["dist"] > NEAR_SPOT_DIST:
                row["why"] = "dist"
            continue
        # another pot: the tiers' gates (0 first: near_spot prefers a lower tier to a nearer spot)
        c, d = row["cover"], row["dist"]
        q = (stack / pot) / (float(sp["stack"]) / float(sp["pot"]))
        wide = str(sp.get("line") or "").count(":r") == 1 and "BB" in (sp.get("seats") or ()) and \
            NEAR_POT_WIDE_SPR[0] <= q <= NEAR_POT_WIDE_SPR[1]   # (a single-raised pot the BB defends, as deep)
        if c >= NEAR_POT_COVER and d <= NEAR_POT_DIST:
            row["tier"] = 0
        elif c >= NEAR_POT_CLOSE_COVER and d <= NEAR_POT_CLOSE_DIST:
            row["tier"] = 1
        elif wide and c >= NEAR_POT_WIDE_COVER and d <= NEAR_POT_WIDE_DIST:
            row["tier"] = 2
        else:
            row["why"] = "cover" if c < min(NEAR_POT_COVER, NEAR_POT_CLOSE_COVER,
                                            NEAR_POT_WIDE_COVER if wide else 1.0) else "dist"
    return out


def near_spot(lib, full0: np.ndarray, full1: np.ndarray, pot: float, stack: float,
              exclude: Optional[str] = None, aggr: Optional[int] = None, players: int = 6) -> Optional[dict]:
    """The library spot nearest a heads-up pot it holds no spot for (ranges full0 out of position, full1 in position
    entering the flop; pot and stack): covering each range by NEAR_COVER, the ranges' summed distance least and at most
    NEAR_SPOT_DIST; one with the same pot and stack first, else (at least NEAR_MIN_SPR here) one whose stack-to-pot
    ratio is within NEAR_SPR of this pot's: of the same pot (another depth), or of another pot (another game's: antes,
    a straddle, heads-up) where the same seat raised last preflop (aggr: 0 out of position, 1 in position, None not
    known: no other pot). None where none is (near_candidates says why). exclude: a spot id left out
    (scripts/bench_flop_approx.py's protocol spot). players: the hand's table size (over 6, a spot of the same pot and
    stack covering it by NEAR_TABLE_COVER where none does by NEAR_COVER)."""
    from . import flop_library as fl
    key = (fl.range_fp(full0), fl.range_fp(full1), round(pot, 2), round(stack, 2), exclude, aggr, int(players))
    if key not in _NEAR:
        best = None
        for row in near_candidates(lib, full0, full1, pot, stack, exclude, aggr, players):
            if row["why"] is None and (best is None or (row["other"], row.get("tier", 0), row["dist"]) < best[0]):
                best = ((row["other"], row.get("tier", 0), row["dist"]), row["id"])
        _NEAR[key] = best and best[1]
    sid = _NEAR[key]
    return next((sp for sp in lib.index().get("spots", []) if sp["id"] == sid), None) if sid else None


def near_exact(lib, full0: np.ndarray, full1: np.ndarray, pot: float, stack: float,
               players: int = 6) -> Optional[dict]:
    """The library spot read as this heads-up pot's own (see NEAR_EXACT_DIST): {"spot", "cover" (each range's: the
    share of its weight the spot's range holds), "dist" (each range's range_dist)}, or None where the library holds
    this pot itself or no spot of the same pot and stack is that near. players: the hand's table size (only spots of
    another table size are read)."""
    from . import flop_library as fl
    key = (fl.range_fp(full0), fl.range_fp(full1), round(pot, 2), round(stack, 2), int(players))
    if key not in _EXACT:
        best = None
        if lib.spot(key[0], key[1], pot, stack) is None:
            for sp in lib.index().get("spots", []):
                if round(float(sp["pot"]), 2) != key[2] or round(float(sp["stack"]), 2) != key[3] or \
                        fl.Format.parse(sp.get("game") or "").players == key[4]:
                    continue
                st = _spot_ranges(sp)
                if st is None:
                    continue
                cover = (range_cover(full0, st["full0"]), range_cover(full1, st["full1"]))
                dist = (range_dist(full0, st["full0"]), range_dist(full1, st["full1"]))
                if min(cover) >= (NEAR_EXACT_TABLE_COVER if players > 6 else NEAR_EXACT_COVER) and \
                        sum(dist) <= NEAR_EXACT_DIST and \
                        (best is None or sum(dist) < sum(best["dist"])):
                    best = {"id": sp["id"], "cover": cover, "dist": dist}
        _EXACT[key] = best
    got = _EXACT[key]
    sp = got and next((x for x in lib.index().get("spots", []) if x["id"] == got["id"]), None)
    return {"spot": sp, "cover": got["cover"], "dist": got["dist"]} if sp else None


class ApproxSolution:
    """A flop not stored, read off stored flops like it (approx_solution), shaped for the advisor's walk like
    flop_library.LibrarySolution: the flop street's tree with the predicted nodes' strategies, `stored` their ids;
    no EVs, nothing solved."""

    def __init__(self, root, ctx, r0, r1, stored: set, known: dict, near: List[Tuple[str, int]], label: str,
                 other: bool = False, spr: Optional[Tuple[float, float]] = None, same_seats: bool = False,
                 pots: Optional[Tuple[float, float]] = None):
        self.root, self.names, self.ctx, self.r0, self.r1 = root, root.names, ctx, r0, r1
        self.stored = stored
        self.known = known                         # id(node) -> (n,) bool: the combos some source holds an image of
        self.near = near                           # [(stored flop in the hand's suits, distance)]
        self.label = label                         # the library spot read
        self.other = other                         # ...another spot's (near_spot): the hand's isn't in the library
        self.spr = spr                             # (the spot's stack-to-pot ratio, the hand's) at another depth
        self.same_seats = same_seats               # ...the hand's seats (the same spot, another depth)
        self.pots = pots                           # (the spot's pot, the hand's) in another game's pot (antes...)
        self.filled: Dict[int, np.ndarray] = {}    # id(node) -> (n,) bool: the combos class_fill answered
        self.iters = 0

    def covers(self, nodes) -> bool:
        return all(id(n) in self.stored for n in nodes)


def norm_name(name: str) -> str:
    """An action's name as a tree at another pot names it: a raise is named by its size in bb ("raise to 11" in a 5.5 bb
    pot is "raise to 12.5" in a 6.25 bb one) and the library's flop tree offers one raise size, so "raise"."""
    return "raise" if name.startswith("raise to ") else name


def node_norm(root, path: Sequence[str]):
    """The node of `root`'s tree a path names (norm_name's names), or None."""
    n = root
    for nm in path:
        names = [norm_name(x) for x in getattr(n, "names", ())]
        k = norm_name(nm)
        if k not in names:
            return None
        n = n.children[names.index(k)]
    return n


# predict's answers kept per library (the last PRED_KEEP flops and combos asked: Study's next node on a flop, or the
# advice's next decision with the same hand, reads its flop's again)
PRED_KEEP = 4


def approx_solution(lib, full0: np.ndarray, full1: np.ndarray, board: Sequence[int], r0: np.ndarray, r1: np.ndarray,
                    pot: float, stack: float, cfg, fetch_only: bool = False,
                    seats: Optional[Sequence[str]] = None,
                    hole: Optional[Sequence[int]] = None, aggr: Optional[int] = None,
                    players: int = 6) -> Optional[ApproxSolution]:
    """The flop decisions a library entry would store, for a flop of a library spot the library doesn't store, read
    off the spot's K stored flops most like it (sources, predict), or for a heads-up pot the library holds no spot
    for, off the nearest spot's (near_spot: its entry of the flop where it stores it, else its K most like it); or
    None: no library spot near (the ranges entering the flop `full0`, `full1`, pot and stack), another tree (cfg's
    sizes), a stored flop (the library answers) or no stored flop near enough. r0, r1: the ranges the advisor solves
    with. Each combo plays its images' average mix (those of its hand class); one no source holds has NaN (the
    answer for it: none).
    fetch_only: only read the sources' entries (None; the advisor's prefetch, which names them for the browser to
    fetch ahead). seats: the pot's (out of position, in position), for the label: the same spot at another depth.
    hole: the combo the answer is for (predict's hand class test made for it alone; default every combo). aggr: who
    raised last preflop (0 out of position, 1 in position; near_spot reads another pot's spot only where its own
    raiser sits there too). players: the hand's table size (near_spot's)."""
    from types import SimpleNamespace
    from . import flop_library as fl
    from .solver import Decision, TreeBuilder, MEMBER_FULL
    if len(board) != 3 or lib is None:
        return None
    idx = lib.index()
    if not idx.get("spots"):
        return None
    spot = lib.spot(fl.range_fp(full0), fl.range_fp(full1), pot, stack)
    other = spot is None
    if other:
        spot = near_spot(lib, full0, full1, pot, stack, aggr=aggr, players=players)
        if spot is None:
            return None
    tk, lk = fl.tree_key(cfg), lib.tree(spot)
    if fl._key_str(tk["sizes"]) != fl._key_str(lk.get("sizes")) or (lk.get("model") or {}).get("version") != fl.MODEL_VERSION:
        return None
    canon, smap = fl.canonical(board)
    if not other and fl.cards_str(canon) in spot.get("flops", {}):
        return None
    src = sources(lib, spot, list(canon), same=other)
    if not src or fetch_only:
        return None
    cmap = fl.combo_map(smap)                       # actual combo -> canonical combo
    bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
    r0f, r1f = r0 * bm, r1 * bm
    active = np.nonzero((r0f > 0) | (r1f > 0))[0]
    # the hand class test made for hero's combo, else for every combo either range holds (the tree holds no other:
    # Study's whole range, a third to a fifth of them); predict's answer kept for the flop's next node (PRED_KEEP);
    # the learned correction on every read of stored flops most like this one (the spot's own, another spot's, another
    # depth's), and on another spot's entry of the flop itself at the same depth (not another depth's: unmeasured)
    combos = np.unique(cmap[active]) if hole is None else np.array([cmap[COMBO_INDEX[(min(hole), max(hole))]]])
    corr = theta() is not None and (not other or any(d > 0 for *_, d in src)
                                    or round(float(spot["stack"]), 2) == round(stack, 2))
    # another game's pot: the hands its ranges don't hold play their hand class's average there (class_fill)
    fill = round(float(spot["pot"]), 2) != round(pot, 2)
    key = (spot["id"], fl.cards_str(canon), tuple((nm, d) for *_, nm, d in src), combos.tobytes(), corr,
           (fl.range_fp(full0), fl.range_fp(full1)) if fill else None)
    kept = lib.__dict__.setdefault("_approx_pred", OrderedDict())
    pred = kept.get(key)
    if pred is None:
        pred = predict([(e, m, w) for e, m, w, _, _ in src],
                       board=list(canon) if corr or any(d > 0 for *_, d in src) else None, combos=combos, correct=corr)
        if fill:
            held = {path: ~np.isnan(o["strat"][0]) for path, o in pred.items()}
            wc = (np.zeros(N_COMBOS), np.zeros(N_COMBOS))
            wc[0][cmap], wc[1][cmap] = full0, full1          # (the ranges on the canonical flop)
            class_fill(pred, list(canon), wc, combos=combos)
            for path, o in pred.items():
                o["filled"] = held[path] ^ ~np.isnan(o["strat"][0])
        kept[key] = pred
        while len(kept) > PRED_KEEP:
            kept.popitem(last=False)
    else:
        kept.move_to_end(key)
    tcfg = cfg.copy()
    tcfg.depth = 1
    ctx = SimpleNamespace(active=active, n=len(active), cfg=tcfg, root_len=3)
    root = TreeBuilder(ctx, dead=list(board)).build(tuple(board), pot, stack)
    stored, known, filled = set(), {}, {}
    for path, o in pred.items():
        n = node_norm(root, path)
        if not isinstance(n, Decision) or [norm_name(x) for x in n.names] != [norm_name(x) for x in o["names"]]:
            continue
        s = o["strat"][:, cmap[active]]
        known[id(n)] = ~np.isnan(s[0])
        if "filled" in o:
            filled[id(n)] = o["filled"][cmap[active]]
        s = np.where(np.isnan(s), 0.0, s).astype(np.float32)   # (no image: a zero row, avg_strategy's uniform; unread)
        n.ssum[0] = s
        n.regret[0] = s
        stored.add(id(n))
    if not stored:
        return None
    inv = [0] * 4                                   # canonical suit -> the hand's
    for a, b in enumerate(smap):
        inv[b] = a
    near = [(fl.cards_str(sorted((c // 4 * 4 + inv[c % 4] for c in parse_cards(nm)), reverse=True)), d)
            for _, _, _, nm, d in src]
    spr = None if round(float(spot["stack"]), 2) == round(stack, 2) else \
        (float(spot["stack"]) / float(spot["pot"]), stack / pot)
    same = seats is not None and list(spot.get("seats") or ()) == list(seats)
    pots = None if round(float(spot["pot"]), 2) == round(pot, 2) else (float(spot["pot"]), pot)
    sol = ApproxSolution(root, ctx, r0f[active], r1f[active], stored, known, near, spot.get("label", spot["id"]),
                         other, spr, same, pots)
    sol.filled = filled
    return sol


# A source's image of a combo counts only where it holds the same hand class there (gto/breakdown.py's categorize: the
# made hand and the draw) as the combo here, unless no source's image does (then all of them). Holding out stored
# flops (scripts/bench_flop_approx.py, knn3c against knn3, % of the pot per flop, mean / median / 90th percentile):
# the flatted opens' 600 0.24 / 0.13 / 0.42 -> 0.18 / 0.06 / 0.32, the 3bet pots' 960 0.24 / 0.16 / 0.50 -> 0.16 /
# 0.07 / 0.38, the vs-BB spots' 750 outside the 552 0.18 / 0.12 / 0.36 -> 0.11 / 0.05 / 0.25; facing a bet 0.56-0.67
# -> 0.40-0.43; better by over 0.01 on 78-85% of the flops, worse on 0.1-1.5%. Images of another class at 0.1 and 0.3
# of the weight did worse than none (200 flops of five spots: 0.120 none, 0.130, 0.145; 0.176 all).
CLASS_EPS = 0.0


def combo_classes(board: Sequence[int], combos: np.ndarray) -> np.ndarray:
    """Each combo's hand class on the board (breakdown.categorize's made hand and draw, joined)."""
    from .breakdown import categorize
    made, draws = categorize(list(board), combos)
    return np.array([m + "/" + d for m, d in zip(made, draws)])


def class_fill(pred: Dict[tuple, dict], board: Sequence[int], ranges: Sequence[np.ndarray],
               combos: Optional[Sequence[int]] = None) -> Dict[tuple, dict]:
    """predict()'s nodes with each combo no source holds (NaN: off the source's range) playing the average mix of
    the combos of its hand class here (breakdown.categorize's made hand and draw) that the sources do hold, weighted
    by the acting player's range (ranges: out of position, in position); the made hand's alone where no held combo
    shares the draw too; NaN where neither. combos: only these are filled (default every combo off the board).
    In place; returns pred."""
    dead = np.zeros(52, bool)
    dead[list(board)] = True
    live = np.nonzero(~(dead[CA_FULL] | dead[CB_FULL]))[0]
    made = draw = None
    want = None if combos is None else np.isin(live, np.asarray(list(combos), dtype=np.int64))
    for o in pred.values():
        S = o["strat"]
        bad = np.isnan(S[0, live])
        if want is not None:
            bad_t = bad & want
        else:
            bad_t = bad
        if not bad_t.any():
            continue
        if made is None:   # (once, where some combo needs it: ~35 ms natively)
            made, draw = class_codes(board, live)
        w = np.asarray(ranges[o["player"]], float)[live]
        good = ~bad & (w > 0)
        for key in (made * 8 + draw, made):
            for c in np.unique(key[bad_t]):
                tgt = bad_t & (key == c)
                src = good & (key == c)
                if not tgt.any() or not src.any():
                    continue
                avg = (S[:, live[src]] * w[src]).sum(axis=1) / float(w[src].sum())
                S[:, live[tgt]] = avg[:, None]
                bad_t &= ~tgt
    return pred


def class_codes(board: Sequence[int], combos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(made, draw): each combo's made hand and draw on the board as indices into breakdown.MADE and DRAWS."""
    from .breakdown import categorize, MADE, DRAWS
    made, draws = categorize(list(board), combos)
    mi = {m: i for i, (m, _) in enumerate(MADE)}
    di = {d: i for i, (d, _) in enumerate(DRAWS)}
    return (np.array([mi[x] for x in made], dtype=np.int64).reshape(-1),
            np.array([di[x] for x in draws], dtype=np.int64).reshape(-1))


# A learned correction of the averaged mixes (gto/flop_approx_model.py, written by scripts/train_flop_correction.py):
# each combo's mix p becomes softmax(log(p + CORR_EPS) + z), z per action a logistic model's: by the node (who acts,
# after which action) and the action's kind (check, bet up to half pot, bet more, fold, call, raise, all-in), terms for
# the combo's made hand and draw here, the share of its images (by weight) holding the same class there and the
# missing share times the made hand and draw again, the flop's ranks less the sources' (weighted; per rank, also per
# made hand), and log(p + CORR_EPS) itself (sharper or flatter). Trained on stored flops read off their spot's others
# to give up less EV (each combo's mix priced by the entry's EVs per action) while keeping the mixes as close to the
# entry's (3x the squared distance). Only the kinds of node the model holds are corrected: facing a bet or a raise
# (EV given up 36-44% lower, the mixes as close); the first decisions (the root, after a check) stay as averaged,
# where it gave up 8% more and mixed a little further (total variation 0.119 -> 0.122). README (A library spot's flop
# the library doesn't store) has the numbers.
CORR_EPS = 0.005
N_KINDS, N_TYPES = 16, 7
_THETA: Optional[dict] = None


def action_type(name: str) -> int:
    """An action's kind for the correction: check 0, a bet up to half pot 1, a bigger bet 2, fold 3, call 4, a raise
    5, all-in 6."""
    if name == "check":
        return 0
    if name.startswith("bet"):
        try:
            return 1 if float(name.split()[1].rstrip("%")) <= 50 else 2
        except (IndexError, ValueError):
            return 2
    return {"fold": 3, "call": 4, "all-in": 6}.get(name, 5)


def node_kind(path: Sequence[str], player: int) -> int:
    """A node's kind for the correction: the action it follows (action_type; 7 at the root) and who acts."""
    return (action_type(path[-1]) if len(path) else 7) * 2 + int(player)


def correction_features(board: Sequence[int], sources: Sequence[Tuple[dict, np.ndarray, float]],
                        idx: np.ndarray) -> dict:
    """The correction's inputs for the combos idx of `board` read off sources [(entry, card map, weight)]: made and
    draw (class_codes), agree (the sources' weight whose image holds the same made hand and draw there, a share),
    match (sources x combos: that image does) and dr (the flop's ranks high to low less the sources' weighted, / 4)."""
    tm, td = class_codes(board, idx)
    ws = np.array([float(w) for _, _, w in sources])
    match = np.zeros((len(sources), len(idx)), bool)
    rk = lambda b: np.sort(np.array([c // 4 for c in b], dtype=float))[::-1]
    dr = rk(board) * ws.sum()
    for k, (_, m, w) in enumerate(sources):
        there = [int(m[c]) for c in board]
        sm, sd = class_codes(there, combo_image(m)[idx])
        match[k] = (sm == tm) & (sd == td)
        dr = dr - w * rk(there)
    return {"made": tm, "draw": td, "match": match, "agree": (ws[:, None] * match).sum(0) / max(ws.sum(), 1e-12),
            "dr": dr / max(ws.sum(), 1e-12) / 4.0}


def theta() -> Optional[dict]:
    """The correction's weights (gto/flop_approx_model.py), {group: array (N_KINDS, N_TYPES, ...), 'on': the node
    kinds it corrects (N_KINDS,)}; None if absent."""
    global _THETA
    if _THETA is None:
        try:
            from . import flop_approx_model as M
        except ImportError:
            _THETA = {}
            return None
        import base64
        flat = np.frombuffer(base64.b64decode(M.DATA), dtype=np.float16).astype(np.float64)
        out, i = {}, 0
        for g, shape in M.SHAPES:
            n = len(M.KINDS) * int(np.prod(shape))
            full = np.zeros((N_KINDS, N_TYPES) + tuple(shape[1:]))
            full[M.KINDS] = flat[i:i + n].reshape((len(M.KINDS),) + tuple(shape))
            out[g] = full
            i += n
        out["on"] = np.zeros(N_KINDS, bool)        # (the kinds of node it corrects)
        out["on"][M.KINDS] = True
        _THETA = out
    return _THETA or None


def correction_z(th: dict, kind: int, types: np.ndarray, L: np.ndarray, f: dict) -> np.ndarray:
    """The correction's logits (A, n) for one node: types (A,) action kinds, L (A, n) log(p + CORR_EPS), f the
    combos' features (correction_features)."""
    t = types[:, None]
    made, draw, agree, dr = f["made"][None, :], f["draw"][None, :], f["agree"][None, :], f["dr"]
    z = th["b"][kind, t] + th["m"][kind, t, made] + th["d"][kind, t, draw] + th["a"][kind, t] * agree
    z = z + (th["r"][kind, types] @ dr)[:, None] + th["mr"][kind, t, made] @ dr
    z = z + th["g"][kind, t] * L + (th["md"][kind, t, made] + th["dd"][kind, t, draw]) * (1.0 - agree)
    return z


def correct_mix(th: dict, names: Sequence[str], path: Sequence[str], player: int, P: np.ndarray,
                f: dict) -> np.ndarray:
    """The corrected mixes (A, n) of mixes P (A, n; NaN columns kept); P itself at a kind of node the model doesn't
    correct."""
    ok = ~np.isnan(P).any(axis=0)
    out = P.copy()
    kind = node_kind(path, player)
    if not ok.any() or not th["on"][kind]:
        return out
    fo = {k: (v[:, ok] if k == "match" else v if k == "dr" else v[ok]) for k, v in f.items()}
    Q = P[:, ok] / np.maximum(P[:, ok].sum(axis=0, keepdims=True), 1e-12)
    L = np.log(Q + CORR_EPS)
    z = L + correction_z(th, kind, np.array([action_type(n) for n in names]), L, fo)
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    out[:, ok] = e / e.sum(axis=0, keepdims=True)
    return out


def predict(sources: Sequence[Tuple[dict, np.ndarray, float]], board: Optional[Sequence[int]] = None,
            combos: Optional[Sequence[int]] = None, correct: bool = False) -> Dict[tuple, dict]:
    """The stored decisions of a flop from decoded entries of flops like it: sources [(entry (flop_library.decode),
    card map (target card -> the entry's), weight)]. {path: {'player', 'names', 'strat' (A, 1326): each combo's mix,
    the weighted average of its images' stored mixes, NaN where no source holds an image, 'support' (1326): the
    weight behind it}}, for each node a source stores (with the same actions). board: the flop predicted (the
    sources' flops are its images); with it a combo averages only the images of the same hand class (combo_classes)
    where any source holds one, CLASS_EPS weight the others; combos: the combos that test is made for (the others
    average every image; default all off the board). correct: with board, those combos' mixes then take the learned
    correction (correct_mix; where gto/flop_approx_model.py is present)."""
    out: Dict[tuple, dict] = {}
    idx = feats = None
    if board is not None:
        dead = np.zeros(52, bool)
        dead[list(board)] = True
        idx = np.arange(N_COMBOS) if combos is None else np.asarray(combos, dtype=np.int64)
        idx = idx[~(dead[CA_FULL[idx]] | dead[CB_FULL[idx]])]
        feats = correction_features(board, sources, idx) if len(idx) else None
    for k, (e, m, w) in enumerate(sources):
        img = combo_image(m)
        wc = None
        if feats is not None:
            wc = np.full(N_COMBOS, w)
            wc[idx] = np.where(feats["match"][k], w, w * CLASS_EPS)
        for nd in e["nodes"]:
            p = nd["player"]
            mask = e["masks"][p]
            pos = np.full(N_COMBOS, -1)
            pos[np.nonzero(mask)[0]] = np.arange(int(mask.sum()))
            col = pos[img]
            ok = col >= 0
            key = tuple(nd["path"])
            o = out.get(key)
            if o is None:
                o = out[key] = {"player": p, "names": list(nd["names"]),
                                "sum": np.zeros((len(nd["names"]), N_COMBOS)), "support": np.zeros(N_COMBOS)}
                if wc is not None:
                    o["csum"], o["csup"] = np.zeros((len(nd["names"]), N_COMBOS)), np.zeros(N_COMBOS)
            elif o["names"] != list(nd["names"]) or o["player"] != p:
                continue
            x = nd["strat"][:, col[ok]]
            o["sum"][:, ok] += w * x
            o["support"][ok] += w
            if wc is not None:
                o["csum"][:, ok] += wc[ok] * x
                o["csup"][ok] += wc[ok]
    for o in out.values():
        s = o.pop("sum")
        sup = o["support"]
        o["strat"] = np.where(sup > 0, s / np.where(sup > 0, sup, 1.0), np.nan)
        if "csum" in o:
            cs, csup = o.pop("csum"), o.pop("csup")
            o["strat"] = np.where(csup > 1e-12, cs / np.where(csup > 1e-12, csup, 1.0), o["strat"])
    th = theta() if correct and feats is not None else None
    if th is not None:
        for path, o in out.items():
            o["strat"][:, idx] = correct_mix(th, o["names"], path, o["player"], o["strat"][:, idx], feats)
    return out
