"""Precomputed flop solutions: common heads-up spots (100bb, 6-max charts) solved offline with Accurate's flop model
(scripts/build_flop_library.py), so the advice on the flop is instant and settled at either speed.

A spot is the two players' ranges entering the flop (as the preflop charts replay them), the pot and the effective
stack; its flops are canonical under suit isomorphism (the ranges are suit-symmetric, so a flop's solve is any
suit-permuted flop's, with its combos permuted). Per spot and flop an entry stores the flop's first decisions (the
root, the in-position player after a check, the out-of-position player facing a bet after checking, the in-position
player facing a lead) and the player facing a raise of those bets: each combo's average strategy (in 1/Q steps) and
its EV per action (in 1% of the pot), plus range aggregates; one value per suit class (combos the flop's own suit
permutations map to each other, pack()). Lines through a re-raise aren't stored and are solved live, as are the
raises of entries built before they were stored (stored_paths).

Files (gto/static/flop_library/, deployed next to the page): index.json lists the spots (their ranges'
fingerprints, pot, stack, label, the tree its entries were solved on) with the default tree and, per spot, each
solved flop's content hash;
<spot>/<flop>.bin is one entry (SPOT_FLOPS: how many flops a spot stores), <spot>/<flop>.turn.bin its turn file (the
flop solve's turn strategies, encode_turns; a spot's "turns" lists them). The browser's worker fetches the index once and an entry only when its spot comes up.
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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from .cards import ALL_COMBOS, COMBO_INDEX, cards_str, cards as parse_cards
from .solver import Config, Chance, Decision, TreeBuilder, MEMBER_FULL, FLOP_LEAF_RIVERS

FORMAT = 2                 # the index's and the entries' (1: per combo, GTOFL2; 2: per suit-isomorphic combo, GTOFL3)
# Bump when the solver's flop game changes in a way the tree key can't see (the equity leaves, the turn classes,
# the river played blind): every stored entry is then ignored until rebuilt.
MODEL_VERSION = 1
MAGIC = b"GTOFL3"
MAGIC_V1 = b"GTOFL2"       # format 1 (read only to re-encode it: recode())
Q = 100                    # strategy steps (1%)
STORE_LINE = 0.05          # a node is stored when the other player reaches it with this share of their range or more
EV_STEP = 0.01             # the node value's step, in pots (0.055 bb in a 5.5 bb pot, 0.2 bb in a 3bet pot)
D_LIN, D_BASE = 32, 1.04   # an action's EV below the best: EV_STEP steps up to 32% of the pot, then 4% steps
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "flop_library")

# The tree entries are solved on: the browser's Accurate config (gto/static/worker.js). The native Accurate plays the
# same flop game (its turn and river sizes differ, but inside a flop solve both play later_sizes, and its leaf
# counts give the same flop all-in leaf), so tree_key() is equal for both.
LS = dict(bet_sizes={"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]},
          raise_sizes={"flop": [0.5], "turn": [0.75], "river": [0.75]})


# Every spot is solved with the river dealt, by RIVER_DEALT buckets of strength on each card (Config.river_dealt).
# Played blind on the turn's boards, the river cost the 3bet pots' flop strategies 0.40% of the pot out of position
# and 0.29% in position against every card (four flops; the 3-bettor bet three quarters pot with 0-11% of its range
# where the full game's bets it with 16-55%), dealt 0.10 and 0.15 at 400 iterations, the shipped entries (200-300)
# 0.11 and 0.27, their root mixes 4-17 points per combo from the full game's where the blind river's were 16-55; a
# single-raised pot's 0.29 and 0.17 (BTN vs BB, JcTd8d), dealt 0.04 and 0.07 (README, Precomputed flop library). A
# single-raised entry takes 6-7x as long dealt (93-179 s of one core against 14-28 at 300 iterations; the same size).
# The entries solved blind before (old_tree()) stay in the index until each is solved again (a spot's old_tree, old).
RIVER_DEALT = 64


def library_config(pot: Optional[float] = None, stack: Optional[float] = None) -> Config:
    """The library's flop model; for a spot's pot and stack, with the river dealt (without them, the river bet blind on
    the turn's boards: the entries solved before, old_tree())."""
    deal = pot is not None and stack is not None
    return Config(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8,
                  bet_sizes={k: list(v) for k, v in LS["bet_sizes"].items()},
                  raise_sizes={k: list(v) for k, v in LS["raise_sizes"].items()},
                  river_dealt=RIVER_DEALT if deal else None)


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
    ("srp_hj_co", "HJ vs CO single-raised pot", "UTG:f,HJ:r2.5,CO:c,BTN:f,SB:f,BB:f"),
    ("srp_btn_sb", "BTN vs SB single-raised pot", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:f"),
    ("srp_utg_hj", "UTG vs HJ single-raised pot", "UTG:r2.5,HJ:c,CO:f,BTN:f,SB:f,BB:f"),
    ("srp_co_sb", "CO vs SB single-raised pot", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:f"),
    ("srp_utg_btn", "UTG vs BTN single-raised pot", "UTG:r2.5,HJ:f,CO:f,BTN:c,SB:f,BB:f"),
    ("3bp_sb_co", "SB 3bet vs CO", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:r10,BB:f,CO:c"),
    ("3bp_hj_utg", "HJ 3bet vs UTG", "UTG:r2.5,HJ:r7.5,CO:f,BTN:f,SB:f,BB:f,UTG:c"),
    ("3bp_bb_co", "BB 3bet vs CO", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:r10,CO:c"),
    ("3bp_co_hj", "CO 3bet vs HJ", "UTG:f,HJ:r2.5,CO:r7.5,BTN:f,SB:f,BB:f,HJ:c"),
    ("3bp_bb_sb", "BB 3bet vs SB", "UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:r10,SB:c"),
]
# How many representative flops each spot stores (representative_flops(n); an odd multiple of a set holds it: 552 =
# 3 x 184, 1656 = 3 x 552; 1755 is every flop): the opens the big blind calls every flop, by coverage per MB
# (scripts/flop_library_coverage.py --assume; a flop not stored was 53 of the 79 points of heads-up flop decisions
# the library missed): BTN vs BB on every flop +12.4 points for 4.5 MB, then SB vs BB +8.6 for 6.0 MB (its ranges'
# entries the biggest), HJ vs BB +7.8 for 3.5, UTG vs BB +6.1 for 3.1; the opens flatted behind or by the small
# blind 552 (+2.9 for 6.6 MB, every flop +4.3 more); the 3bet pots REP_FLOPS.
REP_FLOPS = 184
SPOT_FLOPS = {"srp_btn_bb": 1755, "srp_co_bb": 1755, "srp_hj_bb": 1755, "srp_utg_bb": 1755, "srp_sb_bb": 1755,
              "srp_hj_co": 552, "srp_btn_sb": 552, "srp_utg_hj": 552, "srp_co_sb": 552, "srp_utg_btn": 552,
              "3bp_bb_btn": 552, "3bp_btn_co": 552, "3bp_sb_btn": 552, "3bp_sb_co": 552, "3bp_hj_utg": 552,
              "3bp_bb_co": 552, "3bp_co_hj": 552, "3bp_bb_sb": 552}


# ----------------------------------------------------------------------------- formats
# The spots above are 6-max at 100bb without antes (the page's default game). A spot of another format (stack depth,
# ante, table size, straddle) is the same preflop line played in that game: its own ranges (the charts as the advice
# plays them there), pot and stack, so the advisor finds it by the same key (the two ranges' fingerprints, the pot and
# the stack: FlopLibrary.spot) and the page fetches it by its id like any other. Its id and label carry the format.
@dataclass(frozen=True)
class Format:
    """A game the library's spots are solved in: players at the table, the stack (bb, everyone's), the ante (bb a
    player, dead money) and whether the first seat straddles 2bb."""
    players: int = 6
    stack: float = 100.0
    ante: float = 0.0
    straddle: bool = False

    @property
    def tag(self) -> str:
        """'' for the default game, else what sets it apart: '40bb', '8max', 'a0.125', 'straddle', joined by '_'."""
        parts = ([f"{self.players}max"] if self.players != 6 else []) + \
            ([f"{self.stack:g}bb"] if self.stack != 100.0 else []) + \
            ([f"a{self.ante:g}"] if self.ante else []) + (["straddle"] if self.straddle else [])
        return "_".join(parts)

    @property
    def label(self) -> str:
        parts = ([f"{self.players}-max" if self.players > 2 else "heads-up"] if self.players != 6 else []) + \
            ([f"{self.stack:g}bb"] if self.stack != 100.0 else []) + \
            ([f"{self.ante:g}bb ante"] if self.ante else []) + (["straddled"] if self.straddle else [])
        return ", ".join(parts)

    @classmethod
    def parse(cls, tag: str) -> "Format":
        """The format a tag names ('' or 'default' is the default game)."""
        kw: dict = {}
        for p in [x for x in (tag or "").split("_") if x and x != "default"]:
            if p.endswith("max"):
                kw["players"] = int(p[:-3])
            elif p.endswith("bb"):
                kw["stack"] = float(p[:-2])
            elif p.startswith("a"):
                kw["ante"] = float(p[1:])
            elif p == "straddle":
                kw["straddle"] = True
            else:
                raise ValueError(f"unknown format part {p!r} in {tag!r}")
        return cls(**kw)

    def as_dict(self) -> dict:
        return {"players": self.players, "stack": self.stack, "ante": self.ante, "straddle": self.straddle}

    def table(self):
        """The game's table (preflop.Table) at its depth."""
        from . import preflop as pf
        return pf.at_depth(pf.Table(pf.SEATS[self.players], self.ante, self.straddle, None), self.stack)


DEFAULT_FORMAT = Format()
# Other formats' spots: per format tag, which of SPOTS it holds (each is that spot's preflop line played in the format:
# format_line). Built on request (scripts/build_flop_library.py --format TAG); until then the advisor solves them live.
FORMAT_SPOTS: Dict[str, List[str]] = {
    "40bb": ["srp_btn_bb", "srp_co_bb", "srp_sb_bb", "srp_hj_bb", "srp_utg_bb", "3bp_bb_btn"],
    "200bb": ["srp_btn_bb", "srp_co_bb", "srp_sb_bb", "srp_hj_bb", "srp_utg_bb", "3bp_bb_btn"],
}


def format_line(line: str, fmt: Format) -> str:
    """A spot's preflop line (the default game's, as SPOTS writes it) in another format: its raises and calls by the
    same seats at the same sizes, everyone else folding in that table's order."""
    if fmt == DEFAULT_FORMAT:
        return line
    from .advisor import replay_hand, _norm_actions
    from .cli import parse_actions
    table = fmt.table()
    want = [a for a in _norm_actions(parse_actions(line), 1.0) if a["action"] != "fold"]
    acts = []
    for a in want:
        if a["pos"] not in table:
            raise ValueError(f"{line}: no {a['pos']} at a {fmt.players}-handed table")
        while (st := replay_hand(table, fmt.stack, {"preflop": acts}, 1)[0]).to_act != a["pos"]:
            if st.closed or st.to_act is None:
                raise ValueError(f"{line}: {a['pos']} doesn't act in {fmt.tag or 'the default game'}")
            acts.append({"pos": st.to_act, "action": "fold"})
        acts.append(a)
    while not (st := replay_hand(table, fmt.stack, {"preflop": acts}, 1)[0]).closed and st.to_act is not None:
        acts.append({"pos": st.to_act, "action": "fold"})
    code = {"fold": "f", "call": "c", "check": "x"}
    return ",".join(f"{a['pos']}:{code[a['action']]}" if a["action"] in code else f"{a['pos']}:r{a['to']:g}"
                    for a in acts)


def all_spots(formats: Optional[Sequence[str]] = None) -> List[tuple]:
    """(id, label, line, Format) of every spot: SPOTS in the default game, then FORMAT_SPOTS's (id and label carrying
    the format). `formats`: only these tags ('' the default game)."""
    out = []
    if formats is None or "" in formats:
        out += [(sid, label, line, DEFAULT_FORMAT) for sid, label, line in SPOTS]
    base = {s[0]: s for s in SPOTS}
    for tag, ids in FORMAT_SPOTS.items():
        if formats is not None and tag not in formats:
            continue
        fmt = Format.parse(tag)
        for b in ids:
            sid, label, line = base[b]
            out.append((f"{sid}_{tag}", f"{label}, {fmt.label}", format_line(line, fmt), fmt))
    return out


def spot_setup(line: str, players: int = 6, stack: float = 100.0, ante: float = 0.0, straddle: bool = False) -> dict:
    """The flop spot a preflop line leads to, as the advisor replays it in a game (players, stack, ante, straddle:
    Format): {oop, ip, full0, full1 (their ranges entering the flop), pot, stack (effective, behind)}."""
    from . import preflop as pf
    from .advisor import preflop_ranges, replay_hand, _norm_actions
    from .cli import parse_actions
    acts = _norm_actions(parse_actions(line), 1.0)
    positions = pf.at_depth(pf.Table(pf.SEATS[players], ante, straddle, None), stack)
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
    if getattr(cfg, "river_dealt", None):
        model.update(river_dealt=cfg.river_dealt)      # the river dealt (the library's 3bet pots)
    if getattr(cfg, "river_strength", None):
        # the river by river-strength buckets (solver Config.river_strength; the library's river is on the turn's boards)
        model.update(river_strength=cfg.river_strength, bucket_river_step=cfg.bucket_river_step)
    sizes = {"flop_bet": list(cfg.bet_sizes["flop"]), "flop_raise": list(cfg.raise_sizes["flop"]), **_later(cfg),
             "max_raises": cfg.max_raises, "allin_spr": cfg.allin_spr, "depth": cfg.depth,
             "rake": [cfg.rake_rate, cfg.rake_cap] if cfg.rake_rate > 0 else None}
    return {"sizes": sizes, "model": model}


def old_tree() -> dict:
    """The tree the single-raised entries were solved on before every spot dealt the river (the river bet blind on the
    turn's boards): an index lists a spot's entries still on it as its "old", under "old_tree"."""
    return tree_key(library_config())


def _key_str(d) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


# ----------------------------------------------------------------------------- encoding
def stored_paths(root: Decision, r0=None, r1=None) -> List[tuple]:
    """The flop decisions an entry stores: the root, its decision children (in position after a check, or facing
    a lead), the out-of-position player's decisions facing a bet after checking, and the decisions facing a raise
    of any of those bets (a check-raise, a lead raised); with the reaches r0, r1, only those the other player gets
    to with STORE_LINE of their range (the advisor solves live past a line the solve rarely takes, RARE_LINE: a lead
    the big blind never makes is the solve's noise), a raise only of a bet that is stored."""
    out = [()]
    for nm, c in zip(root.names, root.children):
        if isinstance(c, Decision):
            out.append((nm,))
            if nm == "check":
                out += [(nm, n2) for n2, c2 in zip(c.names, c.children) if isinstance(c2, Decision) and n2 != "check"]
    for path in [p for p in out if p and p[-1] != "check"]:        # (facing a bet: its raises)
        n = node_at(root, path)
        out += [path + (n3,) for n3, c3 in zip(n.names, n.children)
                if isinstance(c3, Decision) and n3 not in ("fold", "call")]
    if r0 is None:
        return out
    keep = []
    for path in out:
        if raised(path) and path[:-1] not in keep:
            continue
        n, a0, a1 = node_at(root, path), *_reaches(root, path, r0, r1)
        other, start = (a1, r1) if n.player == 0 else (a0, r0)
        if other.sum() >= STORE_LINE * max(start.sum(), 1e-12):
            keep.append(path)
    return keep


def raised(path: Sequence[str]) -> bool:
    """A stored node facing a raise (a line with two bets in it): entries store these, the spots' reports don't."""
    return sum(a != "check" for a in path) >= 2


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


def _d_code(d: np.ndarray, step: float, lin_n: int = D_LIN, base: float = D_BASE) -> np.ndarray:
    lin = lin_n * step
    big = lin_n + np.round(np.log(np.maximum(d, lin) / lin) / np.log(base))
    return np.clip(np.where(d < lin, np.round(d / step), big), 0, 255).astype(np.uint8)


def _d_value(code: np.ndarray, step: float, lin_n: int = D_LIN, base: float = D_BASE) -> np.ndarray:
    code = code.astype(np.float64)
    return np.where(code < lin_n, code * step, lin_n * step * base ** (code - lin_n))


def _bits(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8)).tobytes()


def _unbits(b: bytes, n: int = len(ALL_COMBOS)) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))[:n].astype(bool)


def tree_hash(tree: dict) -> str:
    """An entry's tree (tree_key), as its header holds it: the index holds the tree itself."""
    return hashlib.sha1(_key_str(tree).encode()).hexdigest()[:12]


# Format 2 stores a value per suit-isomorphic combo: the canonical flop's suit permutations that keep it (two-tone:
# the two suits off the flop; monotone: the three; a pair's two suits) take a combo to equivalent ones (AhKh, AsKs
# on Kc7c2d), which the solve (and the ranges, and the game) can't tell apart. The solver's turn classes and leaf
# rivers are sampled by card, so its two numbers differ by noise (a 20-point mix gap for 18% of such pairs): the
# entry stores their average (strategies weighted by the hand's reach there), which is also what a solve in other
# suits gets nearer to. Stored combos, per node, in hand order: ranks high to low, suited first, then index (alike
# hands side by side compress better).
_ISO: Dict[tuple, np.ndarray] = {}


def iso_reps(board: Sequence[int]) -> np.ndarray:
    """rep[i]: the least combo index of combo i's class under the flop's suit stabilizer (the suit permutations that
    map the flop to itself); rep[i] == i for a class's stored combo."""
    key = tuple(sorted(board))
    if key not in _ISO:
        rep = np.arange(len(ALL_COMBOS))
        for p in _PERMS:
            if tuple(sorted(c // 4 * 4 + p[c % 4] for c in board)) == key:
                rep = np.minimum(rep, combo_map(p))
        _ISO[key] = rep
    return _ISO[key]


def _hand_order(idx: np.ndarray) -> np.ndarray:
    """idx (combo indices) sorted by hand: the higher rank, then the lower (high to low), suited first, index."""
    a = np.array([ALL_COMBOS[i] for i in idx]).reshape(-1, 2)
    hi, lo = np.maximum(a[:, 0] // 4, a[:, 1] // 4), np.minimum(a[:, 0] // 4, a[:, 1] // 4)
    off = (a[:, 0] % 4 != a[:, 1] % 4)
    return idx[np.lexsort((idx, off, -lo, -hi))]


def _layout(board, mask: np.ndarray):
    """(stored combos in hand order, col: each mask combo's (ascending index) column among them)."""
    rep = iso_reps(board)
    own = np.nonzero(mask & (rep == np.arange(len(rep))))[0]
    stored = _hand_order(own)
    pos = np.full(len(rep), -1)
    pos[stored] = np.arange(len(stored))
    return stored, pos[rep[np.nonzero(mask)[0]]]


def pack(head: dict, masks: Sequence[np.ndarray], nodes: Sequence[dict]) -> bytes:
    """An entry (format 2): head (spot, fp, pot, stack, tree (tree_key: stored as its hash), board, seats, iters,
    seconds, expl, ...), masks [oop, ip] (1326 bools), nodes [{path, player, names, fold0, freqs {name: range
    frequency}, combos, strat (A, n_p), evs (A, n_p), reach (n_p)}], n_p the player's mask combos ascending, reach
    their own reach at the node (weights the classes' strategy average)."""
    board = parse_cards(head["board"])
    rep = iso_reps(board)
    step = round(EV_STEP * float(head["pot"]), 4)
    body = []
    for m in masks:
        if not np.array_equal(m[rep], m):
            raise ValueError("a range that isn't suit-symmetric")
        body.append(_bits(m[rep == np.arange(len(rep))]))
    hn = []
    for nd in nodes:
        stored, col = _layout(board, masks[nd["player"]])
        k, A = len(stored), len(nd["names"])
        w = np.maximum(np.asarray(nd["reach"], dtype=np.float64), 0.0)
        ws = np.zeros(k)
        np.add.at(ws, col, w)
        cnt = np.bincount(col, minlength=k).astype(np.float64)
        st = np.zeros((A, k))
        ev = np.zeros((A, k))
        for a in range(A):
            np.add.at(st[a], col, nd["strat"][a] * w)
            np.add.at(ev[a], col, nd["evs"][a])
        flat = np.zeros((A, k))
        for a in range(A):
            np.add.at(flat[a], col, nd["strat"][a])
        st = np.where(ws > 1e-12, st / np.maximum(ws, 1e-300), flat / cnt)
        ev /= cnt
        q = np.clip(np.round(st[:-1] * Q), 0, Q).astype(np.uint8)
        body.append(q.tobytes())
        vq = np.clip(np.round(ev.max(axis=0) / step), -32767, 32767).astype("<i2")
        body.append(np.frombuffer(vq.tobytes(), dtype=np.uint8).reshape(-1, 2).T.tobytes())   # low bytes, then high
        rows = list(range(1, A) if nd["fold0"] else range(A))
        body.append(_d_code(np.maximum(vq.astype(np.float64) * step - ev[rows], 0.0), step, D_LIN, D_BASE).tobytes())
        hn.append({"path": list(nd["path"]), "player": nd["player"], "names": list(nd["names"]), "fold0": nd["fold0"],
                   "freqs": [nd["freqs"][n] for n in nd["names"]], "combos": nd["combos"]})
    h = {k: v for k, v in head.items() if k not in ("tree", "nodes", "format")}
    h.update(format=FORMAT, tree=tree_hash(head["tree"]) if isinstance(head.get("tree"), dict) else head.get("tree"),
             q=Q, ev=[step, D_LIN, D_BASE], nodes=hn)
    hb = json.dumps(h, separators=(",", ":")).encode()
    return MAGIC + zlib.compress(struct.pack("<I", len(hb)) + hb + b"".join(body), 9)


def encode(sol, meta: dict) -> bytes:
    """An entry from a flop solve (solver.Solution) of the canonical flop with the spot's full ranges. `meta`: spot,
    fp ([oop, ip]), pot, stack, tree (tree_key), board, and anything else to keep in the header."""
    ctx = sol.ctx
    full = [np.zeros(len(ALL_COMBOS)), np.zeros(len(ALL_COMBOS))]
    full[0][ctx.active], full[1][ctx.active] = sol.r0, sol.r1
    masks = [f > 0 for f in full]
    cols = [np.nonzero(m[ctx.active])[0] for m in masks]     # each player's combos, as columns of the solve
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
        fold0 = n.names[0] == "fold" and bool(np.all(np.abs(evs[0]) < 1e-3))
        nodes.append({"path": list(path), "player": p, "names": list(n.names), "fold0": fold0, "freqs": freqs,
                      "combos": round(float(reach.sum()), 2), "strat": st, "evs": evs, "reach": reach})
    head = dict(meta, iters=int(sol.iters), seconds=round(float(sol.seconds), 1), expl=round(float(sol.exploitability()), 4))
    return pack(head, masks, nodes)


def recode(data: bytes, full0: np.ndarray, full1: np.ndarray) -> bytes:
    """A format-1 entry (GTOFL2) in format 2, without solving again: its decoded strategies and EVs (1% and 0.02 bb
    steps), the class averages weighted by the reach the stored strategies give the spot's ranges (full0, full1:
    entering the flop) at each node. An entry already in format 2 comes back as it is."""
    if data[:len(MAGIC)] == MAGIC:
        return data
    e = _decode_v1(data)
    board = parse_cards(e["head"]["board"])
    bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
    full = [full0 * bm, full1 * bm]
    nodes = []
    for nd in e["nodes"]:
        p = nd["player"]
        m = e["masks"][p]
        reach = full[p][m].copy()
        for i in range(len(nd["path"])):
            anc = next((x for x in e["nodes"] if x["path"] == nd["path"][:i]), None)
            if anc is not None and anc["player"] == p:
                reach *= anc["strat"][anc["names"].index(nd["path"][i])]
        nodes.append(dict(nd, reach=reach))
    return pack(e["head"], e["masks"], nodes)


def _decode_v1(data: bytes) -> dict:
    """A format-1 entry: per combo, EVs in 0.02 bb steps (the gap below the best: 64 linear, then 3%)."""
    raw = zlib.decompress(data[len(MAGIC_V1):])
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
        st = np.vstack([q, np.maximum(Q - q.sum(axis=0), 0.0)[None, :]])
        st /= np.maximum(st.sum(axis=0, keepdims=True), 1e-12)
        v = np.frombuffer(body, dtype=np.uint8, count=2 * k, offset=o).reshape(2, k).T.copy().view("<i2")[:, 0]
        v = v.astype(np.float64) * 0.02
        o += 2 * k
        rows = A - 1 if nd["fold0"] else A
        dc = np.frombuffer(body, dtype=np.uint8, count=rows * k, offset=o).reshape(rows, k)
        o += rows * k
        ev = v[None, :] - _d_value(dc, 0.02, 64, 1.03)
        if nd["fold0"]:
            ev = np.vstack([np.zeros((1, k)), ev])
        nodes.append(dict(nd, strat=st, evs=ev))
    return {"head": head, "masks": masks, "nodes": nodes}


def decode(data: bytes) -> dict:
    """{'head': header, 'masks': [oop, ip] (1326 bools; a 3-way entry's: one per player, head 'players'), 'nodes': [{'path', 'player', 'names', 'fold0', 'freqs'
    {name: range frequency}, 'combos', 'strat' (A, n_p), 'evs' (A, n_p)}]}, n_p each player's combos (ascending
    combo index, on the canonical flop; a class's combos share its stored values)."""
    if data[:len(MAGIC_V1)] == MAGIC_V1:
        return _decode_v1(data)
    if data[:len(MAGIC)] != MAGIC:
        raise ValueError("not a flop library entry")
    raw = zlib.decompress(data[len(MAGIC):])
    hl = struct.unpack("<I", raw[:4])[0]
    head = json.loads(raw[4:4 + hl].decode())
    body = raw[4 + hl:]
    q_, (step, lin_n, base) = head["q"], head["ev"]
    board = parse_cards(head["board"])
    rep = iso_reps(board)
    own = rep == np.arange(len(rep))
    n_own = int(own.sum())
    nb = (n_own + 7) // 8
    masks = []
    n_masks = int(head.get("players", 2))         # (3-way entries, gto/mw_library.py: a mask per player)
    for p in range(n_masks):
        m = np.zeros(len(rep), dtype=bool)
        m[own] = _unbits(body[p * nb:(p + 1) * nb], n_own)
        masks.append(m[rep])
    o = n_masks * nb
    nodes = []
    for nd in head["nodes"]:
        stored, col = _layout(board, masks[nd["player"]])
        k, A = len(stored), len(nd["names"])
        q = np.frombuffer(body, dtype=np.uint8, count=(A - 1) * k, offset=o).reshape(A - 1, k).astype(np.float64)
        o += (A - 1) * k
        st = np.vstack([q, np.maximum(q_ - q.sum(axis=0), 0.0)[None, :]])
        st /= np.maximum(st.sum(axis=0, keepdims=True), 1e-12)
        v = np.frombuffer(body, dtype=np.uint8, count=2 * k, offset=o).reshape(2, k).T.copy().view("<i2")[:, 0]
        v = v.astype(np.float64) * step
        o += 2 * k
        rows = A - 1 if nd["fold0"] else A
        dc = np.frombuffer(body, dtype=np.uint8, count=rows * k, offset=o).reshape(rows, k)
        o += rows * k
        ev = v[None, :] - _d_value(dc, step, lin_n, base)
        if nd["fold0"]:
            ev = np.vstack([np.zeros((1, k)), ev])
        nodes.append(dict(nd, freqs=dict(zip(nd["names"], nd["freqs"])), strat=st[:, col], evs=ev[:, col]))
    return {"head": head, "masks": masks, "nodes": nodes}


# ----------------------------------------------------------------------------- precomputed turns
# An entry's turn file (<spot>/<flop>.turn.bin, beside it; GTOTN1): the flop solve's own turn strategies, so a turn
# on a library flop has an answer at once while a live solve settles it. Per flop line that TURN_LINE of the hand
# pairs entering the flop take to the turn (BTN vs BB on Kc7d2h five lines, 66% of the pairs: check-check 29%,
# check, bet 33%, call 23%, ...), the turn decisions TURN_NODES (the first, and in position after a check), per turn
# class (Ctx.turn_classes: the played card's strategy; 13 classes on a rainbow flop, 26 two-tone), for the acting
# player's combos that reach the line (TURN_REACH of its most reached) and don't hold the played card (which the flop
# solve doesn't play there): strategies in 1/TURN_Q steps (format 2: and the solve's EVs there, below). The flop solve
# bets and raises the turn 75% only. Another card of a class reads the played card's strategies as the flop solve
# plays it (turn_node). Written with the entry (scripts/build_flop_library.py), the header naming the entry's hash: a
# turn file is read only beside the entry it was solved with (the index lists those, a spot's "turns").
TURN_MAGIC = b"GTOTN1"
TURN_FORMAT = 1
TURN_LINE = 0.01           # a flop line is stored when this share of the hand pairs reach the turn on it
TURN_REACH = 0.005         # a combo is stored on a line when its reach there is this share of the player's most reached
TURN_Q = 20                # strategy steps (5%)
TURN_NODES = ((), ("check",))
# Format 2 (encode_turns(evs=True)): the flop solve's EVs on the turn too, after the strategies, so a format-1 reader
# (which reads the strategies the header lays out and stops there) reads the file as before; same magic, the header's
# "format": 2 and "ev": TURN_EV_STEP. Per line, per turn class: both players' range EV at the turn's first decision
# (range against range, each pair of hands that share no card by its two reaches, as Advisor._row_evs: the two add up
# to the pot), then per stored node, per class, the acting player's EV of each action (of the hands that take it,
# each by its reach and the pairs it makes); chips relative to folding now, both players on the solve's average
# strategies, every river dealt. int16 in a step of the line's turn pot, the header's "ev": TURN_EV_STEP, or the first
# of TURN_EV_STEPS the file's largest EV fits in (32767 steps: an all-in deep-stacked, 200bb behind a 5.5bb turn pot,
# is ~36x the pot, past 0.1%'s 32.7x); TURN_EV_NONE where nobody is there (an action under TURN_EV_MIN combos of the
# range). 8-12% more bytes, ~0.1-0.5 s a flop (3bet pots at 300 iterations:
# CO vs HJ 9c6d6c 8131 -> 9073 bytes, BB vs SB Kc7d2h 3861 -> 4225, BB vs CO Qc9c4d 16585 -> 17854). Written only
# when the build asks (scripts/build_flop_library.py --turn-evs); a format-1 file reads as before, without EVs.
TURN_EV_FORMAT = 2
TURN_EV_STEP = 0.001       # 0.1% of the turn pot (0.04 bb in a 3bet pot's 40 bb turn)
TURN_EV_STEPS = (0.001, 0.002, 0.005, 0.01)   # (up to 32.7, 65.5, 164, 328 times the pot; past the last: clipped)
TURN_EV_NONE = -32768
TURN_EV_MIN = 0.05         # (Study's: an action's EV is shown for a twentieth of a combo or more)


def entry_hash(entry: bytes) -> str:
    """The hash a turn file's header names its entry by."""
    return hashlib.sha1(entry).hexdigest()[:12]


def turn_lines(sol) -> List[tuple]:
    """The flop lines a solve takes to a turn decision: (flop actions, the turn's Chance node, r0 and r1 entering the
    turn, their share of the hand pairs entering the flop), most first."""
    s = sol.solver
    r0, r1 = sol.r0.astype(np.float64), sol.r1.astype(np.float64)

    def pairs(a0, a1):
        return float((s.compat_sum(a1[None, :].astype(np.float32))[0] * a0).sum())
    tot = max(pairs(r0, r1), 1e-300)
    out = []

    def walk(n, path, a0, a1):
        st = n.avg_strategy0()
        for k, (nm, c) in enumerate(zip(n.names, n.children)):
            b0, b1 = (a0 * st[k], a1) if n.player == 0 else (a0, a1 * st[k])
            if isinstance(c, Decision):
                walk(c, path + (nm,), b0, b1)
            elif isinstance(c, Chance) and isinstance(c.child, Decision):
                out.append((path + (nm,), c, b0, b1, pairs(b0, b1) / tot))
    walk(sol.root, (), r0, r1)
    return sorted(out, key=lambda x: -x[4])


def turn_evs(sol, ch, a0: np.ndarray, a1: np.ndarray, nodes: Sequence[tuple] = TURN_NODES) -> dict:
    """The EVs a flop solve has at a line's turn (ch: the line's turn Chance node, turn_lines; a0, a1: the two reaches
    entering it), per turn class (ch's rows: Ctx.turn_classes' order): {"range": (K, 2) both players' range EV at the
    turn's first decision, "nodes": {at: (K, A) the acting player's EV of each action at node_at(turn, at)}}, as format 2
    stores them (TURN_EV_FORMAT); in chips, NaN where nobody is there. A class's reach is the line's on its played card's
    live combos (ch.masks)."""
    from .solver import _inv_of
    s = sol.solver
    T = ch.child
    K = T.avg_strategy().shape[0]
    masks = np.asarray(ch.masks, dtype=np.float64).reshape(K, -1)
    R = [np.asarray(a0, dtype=np.float64)[None, :] * masks, np.asarray(a1, dtype=np.float64)[None, :] * masks]

    def cfv(node, p, r):
        """p's utility per combo at node (K, n), each over the pairs it makes (compat_sum), and those pair weights."""
        f32 = [np.ascontiguousarray(x, dtype=np.float32) for x in r]
        u = np.asarray(s._cfr(node, p, f32[0], f32[1], update=False, use_avg=True), dtype=np.float64)
        cs = np.asarray(s.compat_sum(f32[1 - p]), dtype=np.float64)
        return np.where(cs > 1e-12, u / np.where(cs > 1e-12, cs, 1.0), 0.0), cs

    was = getattr(s, "sampling", True)
    s.sampling = False                  # (an exact expectation: every river)
    try:
        rng = np.full((K, 2), np.nan)
        for p in (0, 1):
            ev, cs = cfv(T, p, R)
            w = R[p] * cs
            den = w.sum(axis=1)
            ok = den > 1e-9
            rng[ok, p] = (w * ev).sum(axis=1)[ok] / den[ok] + _inv_of(T, p, sol.ctx.cfg.icm)
        out = {}
        for at in nodes:
            n, r = T, list(R)
            try:
                for nm in at:
                    a = n.names.index(nm)
                    r[n.player] = r[n.player] * n.avg_strategy()[:, a, :]
                    n = n.children[a]
            except (ValueError, AttributeError):
                continue
            if not isinstance(n, Decision):
                continue
            q, st = n.player, n.avg_strategy()                 # (K, A, n)
            inv = _inv_of(n, q, sol.ctx.cfg.icm)
            evs = np.full((K, len(n.names)), np.nan)
            for a, c in enumerate(n.children):
                ev, cs = cfv(c, q, r)
                wa = r[q] * st[:, a, :]                         # the hands that take it
                w = wa * cs
                den = w.sum(axis=1)
                ok = (wa.sum(axis=1) >= TURN_EV_MIN) & (den > 1e-12)
                evs[ok, a] = (w * ev).sum(axis=1)[ok] / den[ok] + inv
            out[tuple(at)] = evs
    finally:
        s.sampling = was
    return {"range": rng, "nodes": out}


def _ev_q(x: np.ndarray, step: float) -> bytes:
    """int16 steps of x (clipped to +-32767: -32768 is TURN_EV_NONE, where x is NaN)."""
    q = np.where(np.isfinite(x), np.clip(np.round(np.nan_to_num(x) / step), -32767, 32767), TURN_EV_NONE)
    return q.astype("<i2").tobytes()


def encode_turns(sol, entry: bytes, nodes: Sequence[tuple] = TURN_NODES, evs: bool = False) -> Optional[bytes]:
    """The turn file of an entry (`entry`, encode()'s bytes) from its flop solve, or None when no line reaches the turn
    often enough (TURN_LINE). Header: format, board, entry (entry_hash), q, reach, classes [[played card, its class's
    cards]], lines [{path, share, pot, nodes [{at (turn actions to it), player, names}]}]. Body, per line: the two
    players' stored combos (1326 bits each); per node, per class, (A - 1) x k steps (the last action the rest), k the
    acting player's stored combos without the played card, ascending. evs: format 2 (TURN_EV_FORMAT), the solve's EVs
    after that (turn_evs): per line (K, 2) range EVs, then per node (K, A) action EVs, int16 in the header's "ev" of
    the line's pot (TURN_EV_STEP, or the first of TURN_EV_STEPS that holds the file's largest EV)."""
    ctx = sol.ctx
    act = np.asarray(ctx.active)
    head = decode(entry)["head"]
    classes = ctx.turn_classes(tuple(sol.root.boards[0]), sol.solver.dead)
    lines, body, tail = [], [], []
    for path, ch, a0, a1, share in turn_lines(sol):
        if share < TURN_LINE:
            break
        T = ch.child
        masks = []
        for a in (a0, a1):
            m = np.zeros(len(ALL_COMBOS), dtype=bool)
            m[act] = (a > 0) & (a >= TURN_REACH * float(a.max()))
            masks.append(m)
        hn, nb = [], []
        for at in nodes:
            try:
                n = node_at(T, at)
            except ValueError:
                continue
            if not isinstance(n, Decision):
                continue
            st = n.avg_strategy()                                 # (classes, A, n)
            if st.shape[0] != len(classes):
                raise ValueError("a turn node not batched by the turn classes")
            for k, (played, _) in enumerate(classes):
                cols = np.nonzero(masks[n.player][act] & (MEMBER_FULL[played][act] == 0))[0]
                nb.append(np.clip(np.round(st[k][:-1, cols] * TURN_Q), 0, TURN_Q).astype(np.uint8).tobytes())
            hn.append({"at": list(at), "player": n.player, "names": list(n.names)})
        if hn:
            pot = round(float(T.pot_now), 4)
            lines.append({"path": list(path), "share": round(share, 4), "pot": pot, "nodes": hn})
            body += [_bits(m) for m in masks] + nb
            if evs:
                e = turn_evs(sol, ch, a0, a1, [tuple(x["at"]) for x in hn])
                tail += [(e["range"], pot)] + [(e["nodes"][tuple(x["at"])], pot) for x in hn]
    if not lines:
        return None
    h = {"format": TURN_EV_FORMAT if evs else TURN_FORMAT, "board": head["board"], "entry": entry_hash(entry),
         "q": TURN_Q, "reach": TURN_REACH, "classes": [[cards_str([c]), cards_str(list(cs))] for c, cs in classes],
         "lines": lines}
    if evs:   # (the step: the first that holds the file's largest EV, in times its line's pot)
        big = max((float(np.nanmax(np.abs(x))) / pot for x, pot in tail if np.isfinite(x).any()), default=0.0)
        h["ev"] = next((s for s in TURN_EV_STEPS if big / s <= 32767), TURN_EV_STEPS[-1])
        tail = [_ev_q(x, h["ev"] * pot) for x, pot in tail]
    hb = json.dumps(h, separators=(",", ":")).encode()
    return TURN_MAGIC + zlib.compress(struct.pack("<I", len(hb)) + hb + b"".join(body) + b"".join(tail), 9)


def turns_head(data: bytes) -> dict:
    """A turn file's header alone."""
    if data[:len(TURN_MAGIC)] != TURN_MAGIC:
        raise ValueError("not a turn file")
    raw = zlib.decompress(data[len(TURN_MAGIC):])
    hl = struct.unpack("<I", raw[:4])[0]
    return json.loads(raw[4:4 + hl].decode())


def decode_turns(data: bytes) -> dict:
    """{'head', 'lines': [{'path', 'share', 'pot', 'masks': [oop, ip] (1326 bools), 'nodes': [{'at', 'player', 'names',
    'classes': per class (its stored combos (ascending indices), their strategy (A, k)), 'evs' (format 2: (K, A) the
    action EVs, chips; else None)}], 'evs' (format 2: (K, 2) the range EVs; else None)}]}; combos on the entry's
    (canonical) flop."""
    if data[:len(TURN_MAGIC)] != TURN_MAGIC:
        raise ValueError("not a turn file")
    raw = zlib.decompress(data[len(TURN_MAGIC):])
    hl = struct.unpack("<I", raw[:4])[0]
    head = json.loads(raw[4:4 + hl].decode())
    body = raw[4 + hl:]
    q_ = head["q"]
    played = [parse_cards(p)[0] for p, _ in head["classes"]]
    nb = (len(ALL_COMBOS) + 7) // 8
    o = 0
    lines = []
    for ln in head["lines"]:
        masks = [_unbits(body[o:o + nb]), _unbits(body[o + nb:o + 2 * nb])]
        o += 2 * nb
        nodes = []
        for nd in ln["nodes"]:
            A = len(nd["names"])
            per = []
            for c in played:
                idx = np.nonzero(masks[nd["player"]] & (MEMBER_FULL[c] == 0))[0]
                k = len(idx)
                q = np.frombuffer(body, dtype=np.uint8, count=(A - 1) * k, offset=o).reshape(A - 1, k).astype(np.float64)
                o += (A - 1) * k
                st = np.vstack([q, np.maximum(q_ - q.sum(axis=0), 0.0)[None, :]])
                per.append((idx, st / np.maximum(st.sum(axis=0, keepdims=True), 1e-12)))
            nodes.append(dict(nd, classes=per, evs=None))
        lines.append(dict(ln, masks=masks, nodes=nodes, evs=None))
    if head.get("ev"):          # (format 2: the EVs after every line's strategies)
        K = len(played)

        def ev(shape, step):
            nonlocal o
            x = np.frombuffer(body, dtype="<i2", count=shape[0] * shape[1], offset=o).reshape(shape)
            o += 2 * x.size
            return np.where(x == TURN_EV_NONE, np.nan, x * step)
        for ln in lines:
            step = float(head["ev"]) * float(ln["pot"])
            ln["evs"] = ev((K, 2), step)
            for nd in ln["nodes"]:
                nd["evs"] = ev((K, len(nd["names"])), step)
    return {"head": head, "lines": lines}


def turn_node(t: dict, path: Sequence[str], at: Sequence[str], card: int) -> Optional[dict]:
    """A stored turn decision of a decoded turn file: the flop line `path`, `at` the turn actions to it, `card` the
    turn (on the entry's flop): {'names', 'player', 'played' (its class's played card), 'combos' (ascending), 'strat'
    (A, k), 'evs' (format 2: (A,) the acting player's EV of each action on the class, chips, NaN where nobody takes
    it; else None)}, or None where the file doesn't hold it. Another card of the class reads the played card's strategies as
    the flop solve plays it: each combo the same combo's, and one holding the played card (stored for neither) that of
    its image holding `card` instead (dead on this turn). (Against a live Accurate turn on Kc7d2h this gives up 0.81 /
    0.63% of the pot on average, OOP / IP, the played card itself 0.50 / 0.67; the combos' suit-swap images 0.81 /
    0.93.)"""
    k = next((i for i, (_, cs) in enumerate(t["head"]["classes"]) if card in parse_cards(cs)), None)
    ln = next((x for x in t["lines"] if x["path"] == list(path)), None)
    if k is None or ln is None:
        return None
    nd = next((x for x in ln["nodes"] if x["at"] == list(at)), None)
    if nd is None:
        return None
    idx, st = nd["classes"][k]
    played = parse_cards(t["head"]["classes"][k][0])[0]
    if card != played:
        idx = np.array([COMBO_INDEX[tuple(sorted(played if c == card else c for c in ALL_COMBOS[i]))] for i in idx],
                       dtype=np.int64)
        o = np.argsort(idx)
        idx, st = idx[o], st[:, o]
    evs = None if nd.get("evs") is None else nd["evs"][k]
    return {"names": nd["names"], "player": nd["player"], "played": played, "combos": idx, "strat": st, "evs": evs}


def turn_range_evs(t: dict, path: Sequence[str], card: int) -> Optional[np.ndarray]:
    """Format 2: both players' range EV (2,) at the turn's first decision on the flop line `path` with `card` the turn
    (on the entry's flop; its class's), chips relative to folding now; None where the file holds no EVs (format 1) or
    not that line."""
    k = next((i for i, (_, cs) in enumerate(t["head"]["classes"]) if card in parse_cards(cs)), None)
    ln = next((x for x in t["lines"] if x["path"] == list(path)), None)
    if k is None or ln is None or ln.get("evs") is None:
        return None
    return ln["evs"][k]


def turn_strategies(t: dict, path: Sequence[str], at: Sequence[str], card: int, smap: Sequence[int]) -> Optional[dict]:
    """turn_node for every combo at once, on the hand's own suits: `card` the turn and `smap` canonical()'s suit map of
    the hand's flop (its suits to the entry's). {'names', 'player', 'played', 'strat' (A, 1326): each combo's stored
    mix (its image's on the entry's flop), NaN where the file stores none}, or None where it doesn't hold the node.
    (A live turn solve after a library flop starts from these: advisor.Advisor._store_seed.)"""
    nd = turn_node(t, path, at, card // 4 * 4 + smap[card % 4])
    if nd is None:
        return None
    img = combo_map(smap)                                   # (1326,) each combo's image on the entry's flop
    out = np.full((len(nd["names"]), len(ALL_COMBOS)), np.nan)
    stored = np.asarray(nd["combos"], dtype=np.int64)
    if len(stored):
        j = np.minimum(np.searchsorted(stored, img), len(stored) - 1)
        ok = stored[j] == img
        out[:, ok] = nd["strat"][:, j[ok]]
    return {"names": nd["names"], "player": nd["player"], "played": nd["played"], "strat": out}


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

    def __init__(self, root, ctx, r0, r1, n_nodes, stored: set, evs: dict, head: dict, label: str, exact: bool,
                 river_dealt: bool = False):
        self.root, self.names, self.ctx, self.r0, self.r1 = root, root.names, ctx, r0, r1
        self.n_nodes = n_nodes
        self.stored = stored                       # ids of the stored nodes
        self.solver = _Evs(evs)
        self.iters = int(head["iters"])
        self.seconds = 0.0
        self._expl = float(head["expl"])
        self.head, self.label, self.exact = head, label, exact
        self.river_dealt = river_dealt             # solved with the river dealt (library_config)
        self.settled_once = True

    def exploitability(self) -> float:
        return self._expl

    def extend(self, *a, **k):
        return self

    def covers(self, nodes) -> bool:
        return all(id(n) in self.stored for n in nodes)

    def note(self) -> str:
        return (f"flop: precomputed solution ({self.label}, {self.iters} iterations, {self._expl:.2f}% of the pot from "
                f"equilibrium" + ("" if self.exact else "; every turn rank and every river card" if self.river_dealt
                                  else "; Accurate's flop model, every turn rank with the river bet") + ")")

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

    def tree(self, spot: dict) -> dict:
        """The tree a spot's entries were solved on (tree_key): its own, else the index's."""
        return spot.get("tree") or self.index().get("tree") or {}

    def entry_tree(self, spot: dict, name: str) -> dict:
        """The tree one of a spot's entries was solved on: the spot's, or for a flop not solved again since the
        model changed, its old one (old_tree())."""
        return spot["old_tree"] if name in spot.get("old", ()) else self.tree(spot)

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
            if e is not None and (e["head"].get("fp") != spot["fp"] or e["head"].get("tree") != tree_hash(self.entry_tree(spot, name))):
                e = None
            if len(self._entries) > 64:
                self._entries.clear()
            self._entries[key] = e
        return self._entries[key]

    def turns(self, spot: dict, name: str) -> Optional[dict]:
        """An entry's turn file (decode_turns), or None: the spot doesn't list one (its "turns": a list of flops, or
        true for all), or it isn't the listed entry's."""
        has, h = spot.get("turns"), spot.get("flops", {}).get(name)
        if h is None or not (has is True or (has and name in has)):
            return None
        key = (spot["id"], name, "turns")
        if key not in self._entries:
            t = None
            try:
                raw = self._fetch(f"{spot['id']}/{name}.turn.bin?v={h}")
                t = decode_turns(bytes(raw)) if raw else None
            except Exception:
                t = None
            if t is not None and not t["head"].get("entry", "").startswith(h):
                t = None
            if len(self._entries) > 64:
                self._entries.clear()
            self._entries[key] = t
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
        spot = self.spot(range_fp(full0), range_fp(full1), pot, stack)
        if spot is None:
            return None
        tk, lk = tree_key(cfg), self.tree(spot)
        if _key_str(tk["sizes"]) != _key_str(lk.get("sizes")) or (lk.get("model") or {}).get("version") != MODEL_VERSION:
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
        ek = self.entry_tree(spot, cards_str(canon))
        exact = _key_str(tk["model"]) == _key_str(ek.get("model"))
        return LibrarySolution(root, ctx, r0f[active], r1f[active], tb.n_nodes, stored, evs, e["head"],
                               spot.get("label", spot["id"]), exact, bool((ek.get("model") or {}).get("river_dealt")))


_DEFAULT: Optional[FlopLibrary] = None


def default_library() -> Optional[FlopLibrary]:
    """The library in gto/static/flop_library (None when it has no index)."""
    global _DEFAULT
    if _DEFAULT is None and os.path.exists(os.path.join(DIR, "index.json")):
        _DEFAULT = FlopLibrary.from_dir(DIR)
    return _DEFAULT


# ----------------------------------------------------------------------------- the drills' ranges
# The page's drills (gto/static/drills.js) deal hands at a stored node by the acting player's range weight; an entry
# holds only which combos are in each range. <spot>/ranges.json, written with the index: both ranges entering the
# flop as 1326 integer weights in 1/10000 (combo order of cards.ALL_COMBOS), with their fingerprints (range_fp, of
# the same rounding), which the page checks against the index's.
def spot_ranges(spot_id: str, full0: np.ndarray, full1: np.ndarray) -> dict:
    w = [np.round(np.asarray(r, dtype=np.float64) * 10000).astype(int).tolist() for r in (full0, full1)]
    return {"format": 1, "id": spot_id, "fp": [range_fp(full0), range_fp(full1)], "w": w}


# ----------------------------------------------------------------------------- aggregate reports
# A spot's flop report (gto/static/reports.js, "Flop reports" in the page's menu): per stored flop, each stored
# node's range frequencies, both players' EV and equity at the root, and a hand to load per node, so the page shows
# the strategy across every flop without fetching the entries. <spot>/report.json, written with the index.
REPORT_FORMAT = 1
REPORT_RUNOUTS = 8          # equity over every 8th turn-river runout (~135 of 1081)


def report_weights(names: Sequence[str]) -> Dict[str, float]:
    """How many of the 22100 flops each stored flop stands for. The representative set (representative_flops(n))
    is a systematic sample by weight, so each of its flops stands for 22100 / n (its own suit count would count the
    weight twice); any other set (all 1755, given boards) stands for its own flops."""
    counts = all_flops()
    tot = sum(counts.values())
    keys = [canonical(parse_cards(n))[0] for n in names]
    if 0 < len(keys) < len(counts) and sorted(keys) == sorted(representative_flops(len(keys))):
        return {n: tot / len(names) for n in names}
    return {n: float(counts[k]) for n, k in zip(names, keys)}


def range_equity(board: Sequence[int], w0: np.ndarray, w1: np.ndarray, step: int = REPORT_RUNOUTS) -> float:
    """The first range's equity against the second on a flop (1326-combo weights), over every step-th runout."""
    from .evaluator import eval_boards
    rest = [c for c in range(52) if c not in set(board)]
    runs = list(itertools.combinations(rest, 2))[::step]
    i0, i1 = np.nonzero(w0 > 0)[0], np.nonzero(w1 > 0)[0]
    share = MEMBER_FULL[:, i0].T @ MEMBER_FULL[:, i1]                    # cards the two combos share
    W = w0[i0][:, None] * w1[i1][None, :] * (share == 0)
    num = den = 0.0
    for s in eval_boards([list(board) + list(r) for r in runs], np.concatenate([i0, i1])):
        a, b = s[:len(i0)], s[len(i0):]
        M = W * ((a >= 0)[:, None] & (b >= 0)[None, :])
        num += float((M * ((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :]))).sum())
        den += float(M.sum())
    return num / max(den, 1e-12)


def flop_report(e: dict, full0: np.ndarray, full1: np.ndarray, pot: float) -> dict:
    """One flop's line in its spot's report, from its decoded entry and the spot's ranges entering the flop: f (the
    canonical flop), n {path ('|'-joined): range frequencies in the node's action order}, ev [oop, ip] (bb, at the
    root: out of position's range average, in position the pot less it, as there's no rake), eq [oop, ip] (range
    equity), h {path: a hand of the acting player to load: its range's most mixed full-weight combo}. The nodes
    facing a raise are left out (the page's views are the first decisions)."""
    board = parse_cards(e["head"]["board"])
    bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
    w = [full0 * bm, full1 * bm]
    out = {"f": e["head"]["board"], "n": {}, "h": {}}
    for nd in e["nodes"]:
        if raised(nd["path"]):
            continue
        key = "|".join(nd["path"])
        fr = [nd["freqs"][nm] for nm in nd["names"]]
        out["n"][key] = [round(x / sum(fr), 3) for x in fr]
        m = e["masks"][nd["player"]]
        wp, st = w[nd["player"]][m], nd["strat"]
        ent = np.where(wp >= 0.999 * wp.max(), -(st * np.log(np.maximum(st, 1e-9))).sum(axis=0), -1.0)
        out["h"][key] = cards_str(ALL_COMBOS[int(np.nonzero(m)[0][int(np.argmax(ent))])][::-1])
    root, m0 = e["nodes"][0], e["masks"][0]
    ev0 = float((w[0][m0] * (root["strat"] * root["evs"]).sum(axis=0)).sum() / w[0][m0].sum())
    out["ev"] = [round(ev0, 2), round(pot - ev0, 2)]
    q = range_equity(board, w[0], w[1])
    out["eq"] = [round(q, 3), round(1 - q, 3)]
    return out


def build_report(spot: dict, full0: np.ndarray, full1: np.ndarray, read: Callable[[str], bytes],
                 old: Optional[dict] = None) -> dict:
    """A spot's report from its index entry (id, label, line, seats, pot, stack, flops {name: hash}); read(name)
    gives an entry's bytes. A flop of `old` (the report on disk) whose entry hash is unchanged is kept as it was.
    nodes {path: action names}: the stored nodes (a node's actions are the same on every flop)."""
    ok = bool(old) and old.get("format") == REPORT_FORMAT
    prev = {x["f"]: x for x in old["flops"]} if ok else {}
    nodes: Dict[str, List[str]] = dict(old["nodes"]) if ok else {}
    wts = report_weights(sorted(spot["flops"]))
    flops = []
    for nm in sorted(spot["flops"]):
        h = spot["flops"][nm]
        x = prev.get(nm)
        if x is None or x.get("hash") != h:
            e = decode(read(nm))
            for nd in e["nodes"]:
                k = "|".join(nd["path"])
                if raised(nd["path"]):
                    continue
                if nodes.setdefault(k, nd["names"]) != nd["names"]:
                    raise ValueError(f"{spot['id']} {nm}: node {k!r} has actions {nd['names']}, others {nodes[k]}")
            x = dict(flop_report(e, full0, full1, spot["pot"]), hash=h)
        x["w"] = round(wts[nm], 3)
        flops.append(x)
    return {"format": REPORT_FORMAT, "id": spot["id"], "label": spot.get("label", spot["id"]), "line": spot.get("line"),
            "seats": spot.get("seats"), "pot": spot["pot"], "stack": spot["stack"], "nodes": nodes, "flops": flops,
            **({"game": spot["game"]} if spot.get("game") else {})}


