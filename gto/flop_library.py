"""Precomputed flop solutions: common heads-up spots (100bb, 6-max charts) solved offline with Accurate's flop model
(scripts/build_flop_library.py), so the advice on the flop is instant and settled at either speed.

A spot is the two players' ranges entering the flop (as the preflop charts replay them), the pot and the effective
stack; its flops are canonical under suit isomorphism (the ranges are suit-symmetric, so a flop's solve is any
suit-permuted flop's, with its combos permuted). Per spot and flop an entry stores the flop's first decisions (the
root, the in-position player after a check, the out-of-position player facing a bet after checking, the in-position
player facing a lead): each combo's average strategy (in 1/Q steps) and its EV per action, plus range aggregates.
Lines through a raise aren't stored and are solved live.

Files (gto/static/flop_library/, deployed next to the page): index.json lists the spots (their ranges'
fingerprints, pot, stack, label) with the tree they were solved on and, per spot, each solved flop's content hash;
<spot>/<flop>.bin is one entry. The browser's worker fetches the index once and an entry only when its spot comes up.
The library ignores itself where it doesn't apply: another tree (sizes, raises, rake, MODEL_VERSION), other ranges
(changed charts, other stacks or table sizes, a custom range) or a flop not in it.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import os
import struct
import zlib
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from .cards import ALL_COMBOS, COMBO_INDEX, cards_str, cards as parse_cards
from .solver import Config, Decision, TreeBuilder, MEMBER_FULL, FLOP_LEAF_RIVERS

FORMAT = 1
# Bump when the solver's flop game changes in a way the tree key can't see (the equity leaves, the turn classes,
# the river played blind): every stored entry is then ignored until rebuilt.
MODEL_VERSION = 1
MAGIC = b"GTOFL2"
Q = 100                    # strategy steps (1%)
STORE_LINE = 0.05          # a node is stored when the other player reaches it with this share of their range or more
EV_STEP = 0.02             # bb, the node value's step
D_LIN, D_BASE = 64, 1.03   # an action's EV below the best: 0.02 bb steps up to 1.28 bb, then 3% steps (to ~360 bb)
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "flop_library")

# The tree entries are solved on: the browser's Accurate config (gto/static/worker.js). The native Accurate plays the
# same flop game (its turn and river sizes differ, but inside a flop solve both play later_sizes, and its leaf
# counts give the same flop all-in leaf), so tree_key() is equal for both.
LS = dict(bet_sizes={"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]},
          raise_sizes={"flop": [0.5], "turn": [0.75], "river": [0.75]})


def library_config() -> Config:
    return Config(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8,
                  bet_sizes={k: list(v) for k, v in LS["bet_sizes"].items()},
                  raise_sizes={k: list(v) for k, v in LS["raise_sizes"].items()})


# The spots, most common first (6-max, 100bb, no ante or rake, the page's default sizes: 2.5bb opens, 3bb from the
# small blind, 3bets to 3x in position and 4x from the blinds). Preflop lines as the CLI writes them.
SPOTS = [
    ("srp_btn_bb", "BTN vs BB single-raised pot", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c"),
    ("srp_co_bb", "CO vs BB single-raised pot", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c"),
    ("3bp_bb_btn", "BB 3bet vs BTN", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c"),
    ("srp_sb_bb", "SB vs BB single-raised pot", "UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:c"),
    ("3bp_btn_co", "BTN 3bet vs CO", "UTG:f,HJ:f,CO:r2.5,BTN:r7.5,SB:f,BB:f,CO:c"),
    ("srp_hj_bb", "HJ vs BB single-raised pot", "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:f,BB:c"),
    ("srp_utg_bb", "UTG vs BB single-raised pot", "UTG:r2.5,HJ:f,CO:f,BTN:f,SB:f,BB:c"),
    ("3bp_sb_btn", "SB 3bet vs BTN", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:r10,BB:f,BTN:c"),
]


def spot_setup(line: str, players: int = 6, stack: float = 100.0) -> dict:
    """The flop spot a preflop line leads to, as the advisor replays it: {oop, ip, full0, full1 (their ranges
    entering the flop), pot, stack (effective, behind)}."""
    from . import preflop as pf
    from .advisor import preflop_ranges, replay_hand, _norm_actions
    from .cli import parse_actions
    acts = _norm_actions(parse_actions(line), 1.0)
    positions = pf.at_depth(pf.Table(pf.SEATS[players], 0.0, False, None), stack)
    ranges = preflop_ranges(acts, positions, stack)
    cur = replay_hand(positions, stack, {"preflop": acts, "flop": [], "turn": [], "river": []}, 2)[-1]
    live = [p for p in cur.order if p in cur.active]
    if len(live) != 2:
        raise ValueError(f"{line}: not heads-up to the flop")
    oop, ip = live
    eff = min(stack - cur.inv_start[oop], stack - cur.inv_start[ip])
    return {"oop": oop, "ip": ip, "full0": np.array(ranges[oop]), "full1": np.array(ranges[ip]),
            "pot": round(float(cur.pot_start), 2), "stack": round(float(eff), 2)}


# ----------------------------------------------------------------------------- canonical flops
_PERMS = list(itertools.permutations(range(4)))


def canonical(board: Sequence[int]) -> Tuple[tuple, tuple]:
    """(canonical flop, suit map) for a flop: the canonical flop is the least, as a tuple of its cards sorted high
    to low, of the flop's 24 suit permutations; map[s] is the canonical suit of suit s."""
    best = None
    for p in _PERMS:
        t = tuple(sorted((c // 4 * 4 + p[c % 4] for c in board), reverse=True))
        if best is None or t < best[0]:
            best = (t, p)
    return best


def flop_name(board: Sequence[int]) -> str:
    return cards_str(canonical(board)[0])


def combo_map(suit_map: Sequence[int]) -> np.ndarray:
    """m[i]: the combo that combo i becomes under the suit map."""
    m = np.empty(len(ALL_COMBOS), dtype=np.int64)
    for i, (a, b) in enumerate(ALL_COMBOS):
        x, y = a // 4 * 4 + suit_map[a % 4], b // 4 * 4 + suit_map[b % 4]
        m[i] = COMBO_INDEX[(min(x, y), max(x, y))]
    return m


def all_flops() -> Dict[tuple, int]:
    """Every canonical flop (1755) and how many of the 22100 flops it stands for."""
    out: Dict[tuple, int] = {}
    for f in itertools.combinations(range(52), 3):
        k = canonical(f)[0]
        out[k] = out.get(k, 0) + 1
    return out


def _texture(f: tuple) -> tuple:
    """Sort key grouping similar flops: suit pattern, pairing, then ranks high to low."""
    suits = sorted((sum(1 for c in f if c % 4 == s) for s in range(4)), reverse=True)
    ranks = sorted((c // 4 for c in f), reverse=True)
    paired = 3 - len(set(ranks))
    # (for a two-tone flop, which two ranks share the suit)
    suited_lo = 0
    if suits[0] == 2:
        s = max(range(4), key=lambda x: sum(1 for c in f if c % 4 == x))
        suited_lo = min(c // 4 for c in f if c % 4 == s) == ranks[2]
    return (suits[0], paired, ranks, suited_lo)


def representative_flops(n: int = 184) -> List[tuple]:
    """n canonical flops spread over the flop space by weight: sorted by texture (suit pattern, pairing, ranks),
    one flop at each 1/n step of the cumulative weight (systematic sampling), so each texture gets its share."""
    fl = all_flops()
    keys = sorted(fl, key=_texture)
    if n >= len(keys):
        return keys
    tot = sum(fl.values())
    out, acc, j = [], 0.0, 0
    for k in keys:
        lo, acc = acc, acc + fl[k]
        while j < n and lo <= (j + 0.5) * tot / n < acc:
            if not out or out[-1] != k:
                out.append(k)
            j += 1
    return out


# ----------------------------------------------------------------------------- keys
def range_fp(r: np.ndarray) -> str:
    """A range's fingerprint (weights to 1e-4)."""
    q = np.round(np.asarray(r, dtype=np.float64) * 10000).astype("<i4")
    return hashlib.sha1(q.tobytes()).hexdigest()[:16]


def _later(cfg: Config) -> dict:
    out = {}
    for kind, sizes in (("bet", cfg.bet_sizes), ("raise", cfg.raise_sizes)):
        for st in ("turn", "river"):
            v = (cfg.later_sizes or {}).get(kind, {}).get(st)
            out[f"{kind}_{st}"] = list(v if v is not None else sizes[st])
    return out


def tree_key(cfg: Config) -> dict:
    """What a flop solve's game depends on, beyond ranges, pot and stack. `sizes` must match for a library entry to
    stand for a solve of the tree; `model` is the flop model (Accurate's, or Fast's checked-down river on random
    turns), which an entry may stand in for with a note."""
    per = max(FLOP_LEAF_RIVERS, -(-cfg.leaf_turns * cfg.leaf_rivers_flop // 49))
    model = {"river_play": bool(cfg.river_play), "turn_classes": bool(cfg.turn_classes), "flop_leaf_rivers": per,
             "version": MODEL_VERSION}
    if not cfg.turn_classes:
        model.update(n_turns=cfg.n_turns, seed=cfg.seed, leaf_turns=cfg.leaf_turns)
    if getattr(cfg, "river_buckets", None) or getattr(cfg, "turn_buckets", False):
        # Fast's flop model: the turn and river by hand buckets (per combo otherwise, as the library is solved)
        model.update(river_buckets=cfg.river_buckets, turn_buckets=bool(cfg.turn_buckets),
                     bucket_river_step=cfg.bucket_river_step)
    sizes = {"flop_bet": list(cfg.bet_sizes["flop"]), "flop_raise": list(cfg.raise_sizes["flop"]), **_later(cfg),
             "max_raises": cfg.max_raises, "allin_spr": cfg.allin_spr, "depth": cfg.depth,
             "rake": [cfg.rake_rate, cfg.rake_cap] if cfg.rake_rate > 0 else None}
    return {"sizes": sizes, "model": model}


def _key_str(d) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


# ----------------------------------------------------------------------------- encoding
def stored_paths(root: Decision, r0=None, r1=None) -> List[tuple]:
    """The flop decisions an entry stores: the root, its decision children (in position after a check, or facing
    a lead) and the out-of-position player's decisions facing a bet after checking; with the reaches r0, r1, only
    those the other player gets to with STORE_LINE of their range (the advisor solves live past a line the solve
    rarely takes, RARE_LINE: a lead the big blind never makes is the solve's noise)."""
    out = [()]
    for nm, c in zip(root.names, root.children):
        if isinstance(c, Decision):
            out.append((nm,))
            if nm == "check":
                out += [(nm, n2) for n2, c2 in zip(c.names, c.children) if isinstance(c2, Decision) and n2 != "check"]
    if r0 is None:
        return out
    keep = []
    for path in out:
        n, a0, a1 = node_at(root, path), *_reaches(root, path, r0, r1)
        other, start = (a1, r1) if n.player == 0 else (a0, r0)
        if other.sum() >= STORE_LINE * max(start.sum(), 1e-12):
            keep.append(path)
    return keep


def node_at(root: Decision, path: Sequence[str]) -> Decision:
    n = root
    for nm in path:
        n = n.children[n.names.index(nm)]
    return n


def _reaches(root: Decision, path, r0, r1):
    n = root
    r0, r1 = r0.copy(), r1.copy()
    for nm in path:
        k = n.names.index(nm)
        s = n.avg_strategy0()[k]
        if n.player == 0:
            r0 = r0 * s
        else:
            r1 = r1 * s
        n = n.children[k]
    return r0, r1


def _d_code(d: np.ndarray) -> np.ndarray:
    lin = D_LIN * EV_STEP
    big = D_LIN + np.round(np.log(np.maximum(d, lin) / lin) / np.log(D_BASE))
    return np.clip(np.where(d < lin, np.round(d / EV_STEP), big), 0, 255).astype(np.uint8)


def _d_value(code: np.ndarray) -> np.ndarray:
    code = code.astype(np.float64)
    return np.where(code < D_LIN, code * EV_STEP, D_LIN * EV_STEP * D_BASE ** (code - D_LIN))


def _bits(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8)).tobytes()


def _unbits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))[:len(ALL_COMBOS)].astype(bool)


def encode(sol, meta: dict) -> bytes:
    """An entry from a flop solve (solver.Solution) of the canonical flop with the spot's full ranges. `meta`: spot,
    fp ([oop, ip]), pot, stack, tree (tree_key), and anything else to keep in the header."""
    ctx = sol.ctx
    full = [np.zeros(len(ALL_COMBOS)), np.zeros(len(ALL_COMBOS))]
    full[0][ctx.active], full[1][ctx.active] = sol.r0, sol.r1
    masks = [f > 0 for f in full]
    cols = [np.nonzero(m[ctx.active])[0] for m in masks]     # each player's combos, as columns of the solve
    body = [_bits(masks[0]), _bits(masks[1])]
    nodes = []
    for path in stored_paths(sol.root, sol.r0, sol.r1):
        n = node_at(sol.root, path)
        p = n.player
        r0, r1 = _reaches(sol.root, path, sol.r0, sol.r1)
        evs = sol.solver.node_action_evs(n, r0, r1)[:, cols[p]]
        st = n.avg_strategy0()[:, cols[p]]
        reach = (r0 if p == 0 else r1)[cols[p]]
        tot = max(float(reach.sum()), 1e-12)
        freqs = {nm: round(float((st[a] * reach).sum() / tot), 4) for a, nm in enumerate(n.names)}
        q = np.round(st[:-1] * Q).astype(np.int64)
        body.append(np.clip(q, 0, Q).astype(np.uint8).tobytes())
        v = evs.max(axis=0)
        vq = np.clip(np.round(v / EV_STEP), -32767, 32767).astype("<i2")
        body.append(np.frombuffer(vq.tobytes(), dtype=np.uint8).reshape(-1, 2).T.tobytes())   # low bytes, then high
        fold0 = n.names[0] == "fold" and bool(np.all(np.abs(evs[0]) < 1e-3))
        rows = range(1, len(n.names)) if fold0 else range(len(n.names))
        d = np.maximum(vq.astype(np.float64) * EV_STEP - evs[list(rows)], 0.0)
        body.append(_d_code(d).tobytes())
        nodes.append({"path": list(path), "player": p, "names": list(n.names), "fold0": fold0, "freqs": freqs,
                      "combos": round(float(reach.sum()), 2)})
    head = dict(meta, format=FORMAT, q=Q, nodes=nodes, iters=int(sol.iters), seconds=round(float(sol.seconds), 1),
                expl=round(float(sol.exploitability()), 4))
    hb = json.dumps(head, separators=(",", ":")).encode()
    return MAGIC + zlib.compress(struct.pack("<I", len(hb)) + hb + b"".join(body), 9)


def decode(data: bytes) -> dict:
    """{'head': header, 'masks': [oop, ip] (1326 bools), 'nodes': [{'path', 'player', 'names', 'strat' (A, n_p),
    'evs' (A, n_p)}]}, n_p each player's combos (ascending combo index, on the canonical flop)."""
    if data[:len(MAGIC)] != MAGIC:
        raise ValueError("not a flop library entry")
    raw = zlib.decompress(data[len(MAGIC):])
    hl = struct.unpack("<I", raw[:4])[0]
    head = json.loads(raw[4:4 + hl].decode())
    body = raw[4 + hl:]
    nb = (len(ALL_COMBOS) + 7) // 8
    masks = [_unbits(body[:nb]), _unbits(body[nb:2 * nb])]
    o = 2 * nb
    nodes = []
    for nd in head["nodes"]:
        k = int(masks[nd["player"]].sum())
        A = len(nd["names"])
        q = np.frombuffer(body, dtype=np.uint8, count=(A - 1) * k, offset=o).reshape(A - 1, k).astype(np.float64)
        o += (A - 1) * k
        last = np.maximum(Q - q.sum(axis=0), 0.0)
        st = np.vstack([q, last[None, :]])
        st /= np.maximum(st.sum(axis=0, keepdims=True), 1e-12)
        v = np.frombuffer(body, dtype=np.uint8, count=2 * k, offset=o).reshape(2, k).T.copy().view("<i2")[:, 0]
        v = v.astype(np.float64) * EV_STEP
        o += 2 * k
        rows = A - 1 if nd["fold0"] else A
        dc = np.frombuffer(body, dtype=np.uint8, count=rows * k, offset=o).reshape(rows, k)
        o += rows * k
        ev = v[None, :] - _d_value(dc)
        if nd["fold0"]:
            ev = np.vstack([np.zeros((1, k)), ev])
        nodes.append(dict(nd, strat=st, evs=ev))
    return {"head": head, "masks": masks, "nodes": nodes}


# ----------------------------------------------------------------------------- the advisor's view
class _Evs:
    """Stands in for Solution.solver: the stored EVs at stored nodes."""

    def __init__(self, evs: Dict[int, np.ndarray]):
        self._evs = evs

    def node_action_evs(self, node, r0=None, r1=None):
        return self._evs[id(node)]


class LibrarySolution:
    """A flop solve read from the library, shaped like solver.Solution for the advisor (its walk, its answer at a
    node): the flop street's tree (TreeBuilder, depth 1: the same flop decisions as the full tree) with the stored
    nodes' average strategies, the stored EVs, the stored exploitability."""

    def __init__(self, root, ctx, r0, r1, n_nodes, stored: set, evs: dict, head: dict, label: str, exact: bool):
        self.root, self.names, self.ctx, self.r0, self.r1 = root, root.names, ctx, r0, r1
        self.n_nodes = n_nodes
        self.stored = stored                       # ids of the stored nodes
        self.solver = _Evs(evs)
        self.iters = int(head["iters"])
        self.seconds = 0.0
        self._expl = float(head["expl"])
        self.head, self.label, self.exact = head, label, exact
        self.settled_once = True

    def exploitability(self) -> float:
        return self._expl

    def extend(self, *a, **k):
        return self

    def covers(self, nodes) -> bool:
        return all(id(n) in self.stored for n in nodes)

    def note(self) -> str:
        return (f"flop: precomputed solution ({self.label}, {self.iters} iterations, {self._expl:.2f}% of the pot from "
                f"equilibrium" + ("" if self.exact else "; Accurate's flop model, every turn rank with the river bet") + ")")

    def info(self) -> dict:
        return {"spot": self.head.get("spot"), "label": self.label, "board": self.head.get("board"),
                "iterations": self.iters, "exploitability_pct": round(self._expl, 2), "same_tree": self.exact}


class FlopLibrary:
    """The library, read through `fetch(path) -> bytes or None` (a path under the library's folder: 'index.json',
    '<spot>/<flop>.bin?v=<hash>'). Natively from_dir(); in the browser the worker passes a synchronous fetch."""

    def __init__(self, fetch: Callable[[str], Optional[bytes]]):
        self._fetch = fetch
        self._index = None
        self._spots: Dict[tuple, dict] = {}
        self._entries: Dict[tuple, Optional[dict]] = {}
        self._maps: Dict[tuple, np.ndarray] = {}

    @classmethod
    def from_dir(cls, path: str = DIR) -> "FlopLibrary":
        def fetch(p):
            fn = os.path.join(path, p.split("?")[0])
            try:
                with open(fn, "rb") as f:
                    return f.read()
            except OSError:
                return None
        return cls(fetch)

    def index(self) -> dict:
        if self._index is None:
            raw = None
            try:
                raw = self._fetch("index.json")
            except Exception:
                raw = None
            idx = {}
            if raw:
                try:
                    idx = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
                except ValueError:
                    idx = {}
            if idx.get("format") != FORMAT:
                idx = {"spots": []}
            self._index = idx
            for s in idx.get("spots", []):
                self._spots[(s["fp"][0], s["fp"][1], round(float(s["pot"]), 2), round(float(s["stack"]), 2))] = s
        return self._index

    def spot(self, fp0: str, fp1: str, pot: float, stack: float) -> Optional[dict]:
        self.index()
        return self._spots.get((fp0, fp1, round(pot, 2), round(stack, 2)))

    def entry(self, spot: dict, name: str) -> Optional[dict]:
        key = (spot["id"], name)
        if key not in self._entries:
            h = spot.get("flops", {}).get(name)
            e = None
            if h is not None:
                try:
                    raw = self._fetch(f"{spot['id']}/{name}.bin?v={h}")
                    e = decode(bytes(raw)) if raw else None
                except Exception:
                    e = None
            if e is not None and (e["head"].get("fp") != spot["fp"] or _key_str(e["head"].get("tree")) != _key_str(self.index().get("tree"))):
                e = None
            if len(self._entries) > 64:
                self._entries.clear()
            self._entries[key] = e
        return self._entries[key]

    def solution(self, full0: np.ndarray, full1: np.ndarray, board: Sequence[int], r0: np.ndarray, r1: np.ndarray,
                 pot: float, stack: float, cfg: Config) -> Optional[LibrarySolution]:
        """The stored solve for this flop spot, or None. full0, full1: the two ranges entering the flop (they pick
        the spot); r0, r1: the ranges the advisor solves with (the villain's less the hero's cards, which are dropped
        from the entry's), every combo of which the entry must hold; cfg: the advisor's tree for the street."""
        if len(board) != 3:
            return None
        idx = self.index()
        if not idx.get("spots"):
            return None
        tk, lk = tree_key(cfg), idx.get("tree") or {}
        if _key_str(tk["sizes"]) != _key_str(lk.get("sizes")) or (lk.get("model") or {}).get("version") != MODEL_VERSION:
            return None
        spot = self.spot(range_fp(full0), range_fp(full1), pot, stack)
        if spot is None:
            return None
        canon, smap = canonical(board)
        e = self.entry(spot, cards_str(canon))
        if e is None:
            return None
        mk = tuple(smap)
        if mk not in self._maps:
            self._maps[mk] = combo_map(smap)
        cmap = self._maps[mk]                       # actual combo -> canonical combo
        bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        r0f, r1f = r0 * bm, r1 * bm
        active = np.nonzero((r0f > 0) | (r1f > 0))[0]
        canon_active = cmap[active]
        cols = []
        for p, rf in ((0, r0f), (1, r1f)):
            m = e["masks"][p]
            if not m[canon_active[rf[active] > 0]].all():
                return None                         # a combo the entry doesn't hold (the hero's, off its range)
            pos = np.full(len(ALL_COMBOS), -1)
            pos[np.nonzero(m)[0]] = np.arange(int(m.sum()))
            cols.append(pos[canon_active])
        tcfg = cfg.copy()
        tcfg.depth = 1
        ctx = SimpleNamespace(active=active, n=len(active), cfg=tcfg, root_len=3)
        tb = TreeBuilder(ctx, dead=list(board))
        root = tb.build(tuple(board), pot, stack)
        stored, evs = set(), {}
        for nd in e["nodes"]:
            try:
                n = node_at(root, nd["path"])
            except (ValueError, IndexError, AttributeError):
                return None
            if not isinstance(n, Decision) or list(n.names) != nd["names"]:
                return None
            c = cols[nd["player"]]
            ok = c >= 0
            s = np.full((len(n.names), len(active)), 1.0 / len(n.names), dtype=np.float32)
            s[:, ok] = nd["strat"][:, c[ok]]
            ev = np.zeros((len(n.names), len(active)))
            ev[:, ok] = nd["evs"][:, c[ok]]
            n.ssum[0] = s
            n.regret[0] = s                          # (strategy() reads the regrets: the same mix)
            stored.add(id(n))
            evs[id(n)] = ev
        exact = _key_str(tk["model"]) == _key_str(lk.get("model"))
        return LibrarySolution(root, ctx, r0f[active], r1f[active], tb.n_nodes, stored, evs, e["head"],
                               spot.get("label", spot["id"]), exact)


_DEFAULT: Optional[FlopLibrary] = None


def default_library() -> Optional[FlopLibrary]:
    """The library in gto/static/flop_library (None when it has no index)."""
    global _DEFAULT
    if _DEFAULT is None and os.path.exists(os.path.join(DIR, "index.json")):
        _DEFAULT = FlopLibrary.from_dir(DIR)
    return _DEFAULT
