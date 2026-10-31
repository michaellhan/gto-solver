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
_SETUPS: Dict[str, Optional[dict]] = {}
_NEAR: Dict[tuple, Optional[str]] = {}


def _spot_ranges(spot: dict) -> Optional[dict]:
    """A library spot's ranges entering the flop (flop_library.spot_setup of its line), None where they no longer
    match the fingerprints the index holds (the charts changed since it was built)."""
    from . import flop_library as fl
    line = spot.get("line")
    if line not in _SETUPS:
        try:
            st = fl.spot_setup(line) if line else None
        except Exception:
            st = None
        if st is not None and list(spot.get("fp") or ()) != [fl.range_fp(st["full0"]), fl.range_fp(st["full1"])]:
            st = None
        _SETUPS[line] = st
    return _SETUPS[line]


def range_cover(t: np.ndarray, s: np.ndarray) -> float:
    """The share of range t's weight on combos range s holds."""
    tot = float(t.sum())
    return float((t * (s > 0)).sum()) / tot if tot > 0 else 0.0


def range_dist(t: np.ndarray, s: np.ndarray) -> float:
    """L1 distance of the two ranges, each normalized to 1 (0 to 2)."""
    return float(np.abs(t / max(float(t.sum()), 1e-12) - s / max(float(s.sum()), 1e-12)).sum())


def near_spot(lib, full0: np.ndarray, full1: np.ndarray, pot: float, stack: float,
              exclude: Optional[str] = None) -> Optional[dict]:
    """The library spot nearest a heads-up pot it holds no spot for (ranges full0 out of position, full1 in position
    entering the flop; pot and stack): the same pot, covering each range by NEAR_COVER, the ranges' summed distance
    least and at most NEAR_SPOT_DIST; one with the same stack first, else (at least NEAR_MIN_SPR here) one whose
    stack-to-pot ratio is within NEAR_SPR of this pot's. None where none is. exclude: a spot id left out
    (scripts/bench_flop_approx.py's protocol spot)."""
    from . import flop_library as fl
    key = (fl.range_fp(full0), fl.range_fp(full1), round(pot, 2), round(stack, 2), exclude)
    if key not in _NEAR:
        best = None
        for sp in lib.index().get("spots", []):
            if sp["id"] == exclude or round(float(sp["pot"]), 2) != key[2]:
                continue
            other = round(float(sp["stack"]), 2) != key[3]   # (another depth)
            if other and (stack / pot < NEAR_MIN_SPR or not
                          NEAR_SPR[0] <= (stack / pot) / (float(sp["stack"]) / float(sp["pot"])) <= NEAR_SPR[1]):
                continue
            st = _spot_ranges(sp)
            if st is None:
                continue
            if min(range_cover(full0, st["full0"]), range_cover(full1, st["full1"])) < NEAR_COVER:
                continue
            d = range_dist(full0, st["full0"]) + range_dist(full1, st["full1"])
            if d <= NEAR_SPOT_DIST and (best is None or (other, d) < best[0]):
                best = ((other, d), sp["id"])
        _NEAR[key] = best and best[1]
    sid = _NEAR[key]
    return next((sp for sp in lib.index().get("spots", []) if sp["id"] == sid), None) if sid else None


class ApproxSolution:
    """A flop not stored, read off stored flops like it (approx_solution), shaped for the advisor's walk like
    flop_library.LibrarySolution: the flop street's tree with the predicted nodes' strategies, `stored` their ids;
    no EVs, nothing solved."""

    def __init__(self, root, ctx, r0, r1, stored: set, known: dict, near: List[Tuple[str, int]], label: str,
                 other: bool = False, spr: Optional[Tuple[float, float]] = None, same_seats: bool = False):
        self.root, self.names, self.ctx, self.r0, self.r1 = root, root.names, ctx, r0, r1
        self.stored = stored
        self.known = known                         # id(node) -> (n,) bool: the combos some source holds an image of
        self.near = near                           # [(stored flop in the hand's suits, distance)]
        self.label = label                         # the library spot read
        self.other = other                         # ...another spot's (near_spot): the hand's isn't in the library
        self.spr = spr                             # (the spot's stack-to-pot ratio, the hand's) at another depth
        self.same_seats = same_seats               # ...the hand's seats (the same spot, another depth)
        self.iters = 0

    def covers(self, nodes) -> bool:
        return all(id(n) in self.stored for n in nodes)


def approx_solution(lib, full0: np.ndarray, full1: np.ndarray, board: Sequence[int], r0: np.ndarray, r1: np.ndarray,
                    pot: float, stack: float, cfg, fetch_only: bool = False,
                    seats: Optional[Sequence[str]] = None,
                    hole: Optional[Sequence[int]] = None) -> Optional[ApproxSolution]:
    """The flop decisions a library entry would store, for a flop of a library spot the library doesn't store, read
    off the spot's K stored flops most like it (sources, predict), or for a heads-up pot the library holds no spot
    for, off the nearest spot's (near_spot: its entry of the flop where it stores it, else its K most like it); or
    None: no library spot near (the ranges entering the flop `full0`, `full1`, pot and stack), another tree (cfg's
    sizes), a stored flop (the library answers) or no stored flop near enough. r0, r1: the ranges the advisor solves
    with. Each combo plays its images' average mix (those of its hand class); one no source holds has NaN (the
    answer for it: none).
    fetch_only: only read the sources' entries (None; the advisor's prefetch, which names them for the browser to
    fetch ahead). seats: the pot's (out of position, in position), for the label: the same spot at another depth.
    hole: the combo the answer is for (predict's hand class test made for it alone; default every combo)."""
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
        spot = near_spot(lib, full0, full1, pot, stack)
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
    combos = None if hole is None else [int(cmap[COMBO_INDEX[(min(hole), max(hole))]])]
    pred = predict([(e, m, w) for e, m, w, _, _ in src],
                   board=list(canon) if any(d > 0 for *_, d in src) else None, combos=combos)
    bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
    r0f, r1f = r0 * bm, r1 * bm
    active = np.nonzero((r0f > 0) | (r1f > 0))[0]
    tcfg = cfg.copy()
    tcfg.depth = 1
    ctx = SimpleNamespace(active=active, n=len(active), cfg=tcfg, root_len=3)
    root = TreeBuilder(ctx, dead=list(board)).build(tuple(board), pot, stack)
    stored, known = set(), {}
    for path, o in pred.items():
        try:
            n = fl.node_at(root, path)
        except (ValueError, IndexError, AttributeError):
            continue
        if not isinstance(n, Decision) or list(n.names) != o["names"]:
            continue
        s = o["strat"][:, cmap[active]]
        known[id(n)] = ~np.isnan(s[0])
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
    return ApproxSolution(root, ctx, r0f[active], r1f[active], stored, known, near, spot.get("label", spot["id"]),
                          other, spr, same)


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


def predict(sources: Sequence[Tuple[dict, np.ndarray, float]], board: Optional[Sequence[int]] = None,
            combos: Optional[Sequence[int]] = None) -> Dict[tuple, dict]:
    """The stored decisions of a flop from decoded entries of flops like it: sources [(entry (flop_library.decode),
    card map (target card -> the entry's), weight)]. {path: {'player', 'names', 'strat' (A, 1326): each combo's mix,
    the weighted average of its images' stored mixes, NaN where no source holds an image, 'support' (1326): the
    weight behind it}}, for each node a source stores (with the same actions). board: the flop predicted (the
    sources' flops are its images); with it a combo averages only the images of the same hand class (combo_classes)
    where any source holds one, CLASS_EPS weight the others; combos: the combos that test is made for (the others
    average every image; default all off the board)."""
    out: Dict[tuple, dict] = {}
    idx = None
    if board is not None:
        dead = np.zeros(52, bool)
        dead[list(board)] = True
        idx = np.arange(N_COMBOS) if combos is None else np.asarray(combos, dtype=np.int64)
        idx = idx[~(dead[CA_FULL[idx]] | dead[CB_FULL[idx]])]
        tcls = combo_classes(board, idx) if len(idx) else None
    for e, m, w in sources:
        img = combo_image(m)
        wc = None
        if idx is not None and len(idx):
            wc = np.full(N_COMBOS, w)
            wc[idx] = np.where(combo_classes([int(m[c]) for c in board], img[idx]) == tcls, w, w * CLASS_EPS)
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
    return out
