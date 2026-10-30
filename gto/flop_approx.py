"""Approximate flop answers for a flop the library doesn't store, read off the spot's stored flops most like it.

A spot of the flop library (gto/flop_library.py) stores a representative set of flops (552 of the 1755 for the
flatted opens and the 3bet pots); another flop of the spot is solved live, and until that solve answers the page had
nothing to show. Here its stored decisions are read off the K stored flops most like it (`neighbours`: the same
pairing and suit pattern, the ranks nearest), each carried over by card substitution (`card_map`: the target flop's
cards onto the stored flop's, the other ranks in order, the suits as the two patterns pair them, so a combo's image
is the hand it plays like there: KcJd6h from KcJd5h plays a 6 as the 5, a 5 as the 6, 54 as 64), and their
strategies averaged per combo (`predict`). Numpy only (the browser's worker runs it); nothing is solved.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .cards import ALL_COMBOS, cards as parse_cards
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


def sources(lib, spot: dict, canon: Sequence[int], k: Optional[int] = None, max_dist: Optional[int] = None) -> list:
    """[(decoded entry, card map, weight, name, distance)] of the spot's stored flops most like the canonical flop
    `canon` (not itself), weight 1 / (1 + distance), k (K) of them at most max_dist (MAX_DIST) away; the entries
    fetched through the library (lib.entry)."""
    k = K if k is None else k
    max_dist = MAX_DIST if max_dist is None else max_dist
    out = []
    for nm, m, d in neighbours(sorted(spot.get("flops", {})), canon, k + 1):
        if len(out) == k or d > max_dist:
            break
        if d == 0:
            continue
        e = lib.entry(spot, nm)
        if e is not None:
            out.append((e, m, 1.0 / (1.0 + d), nm, d))
    return out


class ApproxSolution:
    """A flop not stored, read off stored flops like it (approx_solution), shaped for the advisor's walk like
    flop_library.LibrarySolution: the flop street's tree with the predicted nodes' strategies, `stored` their ids;
    no EVs, nothing solved."""

    def __init__(self, root, ctx, r0, r1, stored: set, known: dict, near: List[Tuple[str, int]], label: str):
        self.root, self.names, self.ctx, self.r0, self.r1 = root, root.names, ctx, r0, r1
        self.stored = stored
        self.known = known                         # id(node) -> (n,) bool: the combos some source holds an image of
        self.near = near                           # [(stored flop in the hand's suits, distance)]
        self.label = label
        self.iters = 0

    def covers(self, nodes) -> bool:
        return all(id(n) in self.stored for n in nodes)


def approx_solution(lib, full0: np.ndarray, full1: np.ndarray, board: Sequence[int], r0: np.ndarray, r1: np.ndarray,
                    pot: float, stack: float, cfg, fetch_only: bool = False) -> Optional[ApproxSolution]:
    """The flop decisions a library entry would store, for a flop of a library spot the library doesn't store, read
    off the spot's K stored flops most like it (sources, predict), or None: not a library spot (the ranges entering
    the flop `full0`, `full1`, pot and stack), another tree (cfg's sizes), a stored flop (the library answers) or no
    stored flop near enough. r0, r1: the ranges the advisor solves with. Each combo plays its images' average mix;
    one no source holds has NaN (the answer for it: none). fetch_only: only read the sources' entries (None; the
    advisor's prefetch, which names them for the browser to fetch ahead)."""
    from types import SimpleNamespace
    from . import flop_library as fl
    from .solver import Decision, TreeBuilder, MEMBER_FULL
    if len(board) != 3 or lib is None:
        return None
    idx = lib.index()
    if not idx.get("spots"):
        return None
    spot = lib.spot(fl.range_fp(full0), fl.range_fp(full1), pot, stack)
    if spot is None:
        return None
    tk, lk = fl.tree_key(cfg), lib.tree(spot)
    if fl._key_str(tk["sizes"]) != fl._key_str(lk.get("sizes")) or (lk.get("model") or {}).get("version") != fl.MODEL_VERSION:
        return None
    canon, smap = fl.canonical(board)
    if fl.cards_str(canon) in spot.get("flops", {}):
        return None
    src = sources(lib, spot, list(canon))
    if not src or fetch_only:
        return None
    pred = predict([(e, m, w) for e, m, w, _, _ in src])
    cmap = fl.combo_map(smap)                       # actual combo -> canonical combo
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
    return ApproxSolution(root, ctx, r0f[active], r1f[active], stored, known, near, spot.get("label", spot["id"]))


def predict(sources: Sequence[Tuple[dict, np.ndarray, float]]) -> Dict[tuple, dict]:
    """The stored decisions of a flop from decoded entries of flops like it: sources [(entry (flop_library.decode),
    card map (target card -> the entry's), weight)]. {path: {'player', 'names', 'strat' (A, 1326): each combo's mix,
    the weighted average of its images' stored mixes, NaN where no source holds an image, 'support' (1326): the
    weight behind it}}, for each node a source stores (with the same actions)."""
    out: Dict[tuple, dict] = {}
    for e, m, w in sources:
        img = combo_image(m)
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
            elif o["names"] != list(nd["names"]) or o["player"] != p:
                continue
            o["sum"][:, ok] += w * nd["strat"][:, col[ok]]
            o["support"][ok] += w
    for o in out.values():
        s = o.pop("sum")
        sup = o["support"]
        o["strat"] = np.where(sup > 0, s / np.where(sup > 0, sup, 1.0), np.nan)
    return out
