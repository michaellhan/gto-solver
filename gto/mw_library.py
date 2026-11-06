"""Precomputed 3-way flop solutions: the commonest 3-way pots (100bb, 6-max charts) solved offline by the 3-player
solver (gto/multiway_solver.py) on native Accurate's flop game (advisor.MW_LATER, MW_FLOP_BETS) until settled
(scripts/build_mw_library.py), so a 3-way flop's first decision is a real solve at once, at every speed: a live
3-way flop solve needs ~8 s even in Fast's smaller game, and Practice's quick answers leave it to the betting rules.

A spot is the three players' ranges entering the flop (as the preflop charts replay them, postflop acting order), the
pot and the stack behind (everyone's the same); its flops are canonical under suit isomorphism as the heads-up
library's (gto/flop_library.py, whose entry format this reuses: one mask per player, head 'players': 3). One solve
from the flop's start per spot and flop stores the street's first decisions (STORED: first to act, checked to once,
checked to twice): each combo's average strategy and EV per action, plus range aggregates. The solve takes every hand
the flop doesn't block (multiway_solver's `include`: ~7% more time), so a hand off its player's range (a cold call
the charts 3bet or fold) is held too, by its current (regret-matching) strategy; its stored EVs aren't read (they have
it playing uniformly at its later decisions): a quick answer gives its mix alone, any other solves it live. The
masks then hold every hand. Lines through a bet are not stored: they are solved live (or, in a quick answer, left to the rules) as before.

Files (gto/static/flop_library/, deployed with the heads-up library): mw_index.json lists the spots (their ranges'
fingerprints, pot, stack, label) with the tree they were solved on and each solved flop's content hash;
<spot>/<flop>.bin is one entry. The library ignores itself where it doesn't apply: other ranges (changed charts,
other stacks or table sizes, a custom range), rake, custom flop sizes, or a flop not in it.
"""
from __future__ import annotations
import json
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .cards import ALL_COMBOS, cards as parse_cards, cards_str
from . import flop_library as fl

FORMAT = 1
# Bump when the 3-player solver's game changes in a way the tree key can't see: stored entries are then ignored.
MODEL_VERSION = 1
INDEX = "mw_index.json"
DIR = fl.DIR
# The decisions stored: the flop's first to act, the second after a check, the third after two
STORED = [(), ("check",), ("check", "check")]

# The spots (6-max, 100bb, no ante or rake, 2.5bb opens), with how often the charts deal each line (share of hands,
# folds included): BTN opens, SB and BB call 0.25%; HJ opens, CO or BTN and BB call 0.22% each;
# UTG opens, BTN and BB call 0.18%; CO opens, BTN and BB call 0.14% (UTG opens, HJ and BB call 0.08%). Preflop lines as
# the CLI writes them.
SPOTS = [
    ("mw_co_btn_bb", "CO open, BTN and BB call", "UTG:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:c"),
    ("mw_btn_sb_bb", "BTN open, SB and BB call", "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:c"),
    # (the CO's cold call over a HJ open is the BTN's: one game, matched by its ranges either way)
    ("mw_hj_ip_bb", "HJ open, CO or BTN and BB call", "UTG:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:c"),
    ("mw_utg_btn_bb", "UTG open, BTN and BB call", "UTG:r2.5,HJ:f,CO:f,BTN:c,SB:f,BB:c"),
    # listed for the build, stored once built (a spot without entries is left out of the index): of the flops seen
    # 3-way, UTG opens, HJ or CO and BB call 10.2% (one game: the HJ's and the CO's calls are the same range); CO opens,
    # SB and BB call 8.3%; HJ opens, SB and BB call 4.3%; UTG opens, SB and BB call 3.9% (the four above: 63.8%)
    ("mw_utg_ip_bb", "UTG open, HJ or CO and BB call", "UTG:r2.5,HJ:c,CO:f,BTN:f,SB:f,BB:c"),
    ("mw_co_sb_bb", "CO open, SB and BB call", "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:c,BB:c"),
    ("mw_hj_sb_bb", "HJ open, SB and BB call", "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:c,BB:c"),
    ("mw_utg_sb_bb", "UTG open, SB and BB call", "UTG:r2.5,HJ:f,CO:f,BTN:f,SB:c,BB:c"),
]
REP_FLOPS = 184
SPOT_FLOPS: Dict[str, int] = {}
# stored beside each spot's representative flops: K72 rainbow, the 3-way spots of scripts/bench_browser.mjs and the tests
EXTRA_FLOPS = ["Kc7d2h"]
# A 3-way pot of another table size (8- and 9-max: an open flatted and called by the big blind, the same pot and
# stack as a spot's) is read as that spot's (MWLibrary.borrow: solution and near with it), marked borrowed, where the
# three ranges' distances (flop_approx.range_dist) sum to at most MW_BORROW_DIST and each range there holds at least
# MW_BORROW_COVER of this one's (by weight; the entries hold every hand, so none is left out). Priced by each pot's own
# solve (scripts/bench_mw_formats.py, 6-10 stored flops a pot, % of the pot given up, mean (worst flop)): 8-max BTN
# open, SB and BB call (0.14 from the 6-max spot) 0.015 (0.06); the HJ's open flatted by the CO or the BTN (0.31 /
# 0.43) 0.028 / 0.046 (0.13); the LJ's by the BTN (0.36) 0.032 (0.11): under the live Fast solve of the pot's own
# ranges (0.06-0.08 on the spots' own pots at 200 iterations, the browser's 8 s). Further apart 0.09-0.17: UTG's
# flatted by the BTN (0.58) 0.088, by UTG1 (0.88) 0.098, the CO's by the BTN (0.90) 0.109, UTG's by the HJ (1.30)
# 0.172, the CO's by the SB (1.56) 0.129, as much as a spot's own pot read off its stored flops most like it
# (0.12-0.14) and more than the live solve (0.05-0.08 on those): read at once only, until it answers (MW_STAND_DIST).
MW_BORROW_DIST = 0.5
MW_BORROW_COVER = 0.9
# A 3-way pot neither held nor borrowed (6-max's other opens and flats, 8- and 9-max's further from a spot, antes:
# another pot) is read approximately off the nearest spot (MWLibrary.stand_in: near only, the answer at once while its
# own solve runs): any pot whose stack-to-pot ratio is within MW_STAND_SPR of this one's, the distances summing to at
# most MW_STAND_DIST and each range there holding at least MW_STAND_COVER of this one's; on a flop the spot stores, its
# entry's mixes, on another its stored flops most like it. scripts/bench_mw_formats.py (4-10 stored flops a pot, as
# borrow's): the stored mixes give up 0.08-0.17% of the pot (6-max UTG's open flatted by the HJ, 0.83 from UTG's
# spot, 0.090; the CO's by the SB, 1.58 from BTN's, 0.084; 8-max's above; with antes 0.091-0.168, the farthest 8-max
# UTG's flatted by UTG1, 1.80 and the BB's range 45% held, and 6-max's CO flatted by the SB, 1.95, 0.106; at another
# depth 0.044 at 120bb, BTN open, SB and BB call, the ratio 1.21x, and 0.024 at 85bb, CO open, BTN and BB call, 0.85x),
# the stored flops most like it 0.09-0.24, as the spots' own pots read that way (0.12-0.14); the live Fast solve that
# replaces them 0.05-0.09. Left out: the HJ's open flatted by the SB, 2.60 with antes and 3.26 without, 0.21 / 0.20
# (0.27 / 0.25 off the flops most like it, checked to twice 0.37-0.41); the straddle's pots (under half a spot's
# stack-to-pot ratio).
MW_STAND_DIST = 2.0
MW_STAND_COVER = 0.44
MW_STAND_SPR = (0.8, 1.25)
_SETUPS: Dict[str, Optional[dict]] = {}


def _spot_ranges(sp: dict) -> Optional[dict]:
    """A spot's own setup (spot_setup of its line, at 6-max), None where its ranges no longer match the fingerprints
    the index holds (the charts changed since it was built)."""
    if sp["id"] not in _SETUPS:
        try:
            st = spot_setup(sp["line"])
            ok = [fl.range_fp(r) for r in st["ranges"]] == list(sp["fp"])
        except Exception:
            st, ok = None, False
        _SETUPS[sp["id"]] = st if ok else None
    return _SETUPS[sp["id"]]


def library_config():
    """The game entries are solved on: native Accurate's 3-way flop (MWConfig: 16 hand buckets on the later streets,
    the flop betting a third or three quarters of the pot)."""
    from .advisor import MW_LATER, MW_FLOP_BETS
    from .multiway_solver import MWConfig
    return MWConfig(**dict(MW_LATER, bet={k: (list(v) if isinstance(v, list) else v) for k, v in MW_FLOP_BETS.items()}))


def tree_key(mcfg) -> dict:
    """What a 3-way flop solve's game depends on beyond ranges, pot and stack."""
    return {"bet": mcfg.bet, "raise_frac": mcfg.raise_frac, "max_raises": mcfg.max_raises, "allin_spr": mcfg.allin_spr,
            "later": mcfg.later, "buckets": mcfg.buckets, "classes": mcfg.classes,
            "bucket_river_step": mcfg.bucket_river_step, "exact": mcfg.exact,
            "rake": [mcfg.rake_rate, mcfg.rake_cap] if mcfg.rake_rate > 0 else None, "version": MODEL_VERSION}


def spot_setup(line: str, players: int = 6, stack: float = 100.0, ante: float = 0.0, straddle: bool = False) -> dict:
    """The 3-way flop spot a preflop line leads to, as the advisor replays it: {seats (postflop order), ranges (theirs
    entering the flop), pot, stack (behind, the shortest)}; in a game of `players`, `stack`, `ante` and `straddle`."""
    from . import preflop as pf
    from .advisor import preflop_ranges, replay_hand, _norm_actions
    from .cli import parse_actions
    acts = _norm_actions(parse_actions(line), 1.0)
    positions = pf.at_depth(pf.Table(pf.SEATS[players], ante, straddle, None), stack)
    ranges = preflop_ranges(acts, positions, stack)
    cur = replay_hand(positions, stack, {"preflop": acts, "flop": [], "turn": [], "river": []}, 2)[-1]
    live = [p for p in cur.order if p in cur.active]
    if len(live) != 3:
        raise ValueError(f"{line}: not 3-way to the flop")
    return {"seats": live, "ranges": [np.array(ranges[p]) for p in live], "pot": round(float(cur.pot_start), 2),
            "stack": round(float(min(stack - cur.inv_start[p] for p in live)), 2)}


# ----------------------------------------------------------------------------- encoding
def _reach(sol, path):
    """(node, reach (players, 1, n)) at the end of path, everyone on their average strategies."""
    from .multiway_solver import Decision
    node, R = sol.root, sol.reach0.copy()
    for nm in path:
        if not isinstance(node, Decision):
            raise ValueError(f"no decision at {path}")
        k = node.names.index(nm)
        R[node.player] = R[node.player] * node.avg_strategy()[:, k, :]
        node = node.children[k]
    return node, R


def encode(sol, meta: dict) -> bytes:
    """An entry from a 3-way flop solve (multiway_solver.MultiwaySolver from the flop's start) of the canonical flop
    with the spot's ranges. `meta`: spot, fp (three), pot, stack, tree (tree_key), board, seats."""
    from .multiway_solver import Decision
    active = sol.hands.active
    masks = []
    for p in range(sol.np):        # every hand solved (solve_all, meta all_hands), else each player's range
        m = np.zeros(len(ALL_COMBOS), dtype=bool)
        m[active if meta.get("all_hands") else active[sol.reach0[p, 0] > 0]] = True
        masks.append(m)
    cols = [np.nonzero(m[active])[0] for m in masks]
    nodes = []
    for path in STORED:
        try:
            node, R = _reach(sol, path)
        except (ValueError, IndexError):
            continue
        if not isinstance(node, Decision):
            continue
        p = node.player
        evs = sol.action_evs(node, R)[:, cols[p]]
        reach = R[p, 0][cols[p]]
        # (a hand its range doesn't reach here: the current strategy, as MultiwaySolver.hand_strategy)
        st = np.where(reach[None, :] > 1e-12, node.avg_strategy()[0][:, cols[p]], node.strategy()[0][:, cols[p]])
        tot = max(float(reach.sum()), 1e-12)
        freqs = {nm: round(float((st[a] * reach).sum() / tot), 4) for a, nm in enumerate(node.names)}
        nodes.append({"path": list(path), "player": p, "names": list(node.names), "fold0": False, "freqs": freqs,
                      "combos": round(float(reach.sum()), 2), "strat": st, "evs": evs, "reach": reach,
                      "amounts": [None if a is None else float(a) for a in node.amounts],
                      "pot_now": float(node.pot_now)})
    gains = sol.br_gains() / sol.root_pot() * 100
    head = dict(meta, players=sol.np, iters=int(sol.iters), seconds=round(float(sol.seconds), 1),
                expl=round(float(max(gains.mean(), 0.0)), 4), gains=[round(float(g), 4) for g in gains],
                acts={"|".join(nd["path"]): [nd["amounts"], nd["pot_now"]] for nd in nodes})
    return fl.pack(head, masks, nodes)


def solve_all(board, ranges, pot, stack, cfg=None):
    """A MultiwaySolver from the flop's start taking every hand the flop doesn't block (the library's solves)."""
    from .multiway_solver import MultiwaySolver, _MEM
    inc = [i for i in range(len(ALL_COMBOS)) if not _MEM[i, list(board)].any()]
    return MultiwaySolver(board, ranges, pot, stack, cfg=cfg or library_config(), include=inc)


def decode(data: bytes) -> dict:
    """flop_library.decode, with each node's amounts and pot (head 'acts')."""
    e = fl.decode(data)
    acts = e["head"].get("acts", {})
    for nd in e["nodes"]:
        nd["amounts"], nd["pot_now"] = acts.get("|".join(nd["path"]), [[None] * len(nd["names"]), 0.0])
    return e


# ----------------------------------------------------------------------------- the advisor's view
class _Unstored:
    """A node the entry doesn't hold (a line through a bet): walks stop there."""
    unstored = True


UNSTORED = _Unstored()


class _Hands:
    def __init__(self, active):
        self.active, self.n = active, len(active)


class LibraryMWSolution:
    """A 3-way flop solve read from the library, shaped like multiway_solver.MultiwaySolver for the advisor (its
    answer, practice opponents' moves and narrowing through the flop's stored lines: Advisor._mw_walk,
    _mw_actor_move): the stored decisions as multiway_solver.Decision nodes (their average strategies; a line the
    entry doesn't hold is UNSTORED), reach0, the stored EVs, the stored best-response gains, and the chips behind
    (`stacks`: the spot's, everyone the same)."""

    def __init__(self, nodes: Dict[tuple, object], evs: Dict[int, np.ndarray], active, reach0, head: dict, label: str,
                 seats: List[str], stack: Optional[float] = None):
        self.nodes, self._evs = nodes, evs
        self.root = nodes[()]
        self.hands = _Hands(active)
        self.reach0 = reach0
        self.np = reach0.shape[0]
        self.head, self.label, self.seats = head, label, seats
        self.iters = int(head["iters"])
        self.seconds = self.build_seconds = 0.0
        self.out_of_time, self.settled = False, True
        self._gains = np.array(head.get("gains") or [head["expl"]] * self.np, dtype=np.float64)
        self.pot_start = float(head["pot"])
        self.stacks = (float(stack if stack is not None else head.get("stack", 0.0)),) * self.np
        self.approx: Optional[dict] = None   # (MWLibrary.near's: read off stored flops like it, no EVs)
        self.borrowed: Optional[dict] = None  # (another table size's pot read as this spot's: MWLibrary.borrow)

    def root_pot(self):
        return self.pot_start

    def br_gains(self):
        """Each player's best-response gain as solved (bb), from the stored % of the pot."""
        return self._gains * self.pot_start / 100.0

    def exploitability(self) -> float:
        return float(max(self._gains.mean(), 0.0))

    def action_evs(self, node=None, R=None):
        return self._evs[id(node or self.root)]

    def combo_index(self, hole) -> Optional[int]:
        from .cards import COMBO_INDEX
        i = COMBO_INDEX[(min(hole), max(hole))]
        pos = np.nonzero(self.hands.active == i)[0]
        return int(pos[0]) if len(pos) else None

    def hand_strategy(self, ci: int, p: int, node=None) -> np.ndarray:
        return (node or self.root).avg_strategy()[0][:, ci]

    def reach_at(self, path):
        """Each player's reach (players, 1, n) at a stored node, through the stored strategies."""
        R = self.reach0.copy()
        node = self.root
        for nm in path:
            k = node.names.index(nm)
            R[node.player] = R[node.player] * node.avg_strategy()[:, k, :]
            node = node.children[k]
        return R

    def expand(self, v):
        out = np.zeros(len(ALL_COMBOS))
        out[self.hands.active] = v
        return out

    def borrowed_text(self) -> str:
        """How a borrowed spot (another table size's pot read as this one) or a stand-in (another pot read off it)
        differs from this pot."""
        b = self.borrowed
        moved = max(b["dist"]) / 2.0 * 100.0
        if "pot" in b and (abs(b["pot"] - self.pot_start) > 0.01 or abs(b["stack"] - self.stacks[0]) > 0.01):
            return (f"its pot {self.pot_start:g}bb with {round(self.stacks[0], 1):g} behind, this one {b['pot']:g}bb "
                    f"with {round(b['stack'], 1):g}; the ranges differ by {moved:.1f}% of their weight at most")
        if "pot" in b:
            return f"the same pot; the ranges differ by {moved:.1f}% of their weight at most"
        return f"this table's ranges differ from that spot's by {moved:.1f}% of their weight at most"

    def info(self) -> dict:
        return {"spot": self.head.get("spot"), "label": self.label, "board": self.head.get("board"),
                "iterations": self.iters, "exploitability_pct": round(self.exploitability(), 2), "same_tree": True,
                "players": 3, **({"borrowed": dict(self.borrowed, text=self.borrowed_text())} if self.borrowed else {})}


class MWLibrary:
    """The 3-way library, read through `fetch(path) -> bytes or None` as flop_library.FlopLibrary (natively
    from_dir(); in the browser the worker's synchronous fetch)."""

    def __init__(self, fetch: Callable[[str], Optional[bytes]]):
        self._fetch = fetch
        self._index = None
        self._spots: Dict[tuple, dict] = {}
        self._entries: Dict[tuple, Optional[dict]] = {}
        self._borrow: Dict[tuple, Optional[tuple]] = {}   # (MWLibrary.borrow's and stand_in's choices)

    @classmethod
    def from_dir(cls, path: str = DIR) -> "MWLibrary":
        return cls(fl.FlopLibrary.from_dir(path)._fetch)

    def index(self) -> dict:
        if self._index is None:
            try:
                raw = self._fetch(INDEX)
                idx = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw) if raw else {}
            except Exception:
                idx = {}
            if not isinstance(idx, dict) or idx.get("format") != FORMAT:
                idx = {"spots": []}
            self._index = idx
            for s in idx.get("spots", []):
                self._spots[(tuple(s["fp"]), round(float(s["pot"]), 2), round(float(s["stack"]), 2))] = s
        return self._index

    def spot(self, fps: Sequence[str], pot: float, stack: float) -> Optional[dict]:
        self.index()
        return self._spots.get((tuple(fps), round(pot, 2), round(stack, 2)))

    def borrow(self, ranges: Sequence[np.ndarray], pot: float, stack: float, players: int) -> Optional[dict]:
        """The spot read as this 3-way pot's own (see MW_BORROW_DIST): {"spot", "cover" (each range's: the share of
        its weight the spot's range holds), "dist" (each range's flop_approx.range_dist)}, or None where the library
        holds this pot itself, `players` (the hand's table size) is the library's own (6: a pot it doesn't hold is
        its own, solved), or no spot of the same pot and stack is that near."""
        from .flop_approx import range_cover, range_dist
        if len(ranges) != 3 or int(players) == 6:
            return None
        fps = tuple(fl.range_fp(r) for r in ranges)
        key = (fps, round(pot, 2), round(stack, 2))
        if key not in self._borrow:
            best = None
            if self.spot(fps, pot, stack) is None:
                for sp in self.index().get("spots", []):
                    if round(float(sp["pot"]), 2) != key[1] or round(float(sp["stack"]), 2) != key[2]:
                        continue
                    st = _spot_ranges(sp)
                    if st is None:
                        continue
                    cover = tuple(range_cover(np.asarray(x), y) for x, y in zip(ranges, st["ranges"]))
                    dist = tuple(range_dist(np.asarray(x), y) for x, y in zip(ranges, st["ranges"]))
                    if min(cover) >= MW_BORROW_COVER and sum(dist) <= MW_BORROW_DIST and \
                            (best is None or sum(dist) < sum(best[2])):
                        best = (sp["id"], cover, dist)
            self._borrow[key] = best
        got = self._borrow[key]
        sp = got and next((x for x in self.index().get("spots", []) if x["id"] == got[0]), None)
        return {"spot": sp, "cover": got[1], "dist": got[2]} if sp else None

    def stand_in(self, ranges: Sequence[np.ndarray], pot: float, stack: float) -> Optional[dict]:
        """The spot a 3-way pot neither held nor borrowed is read approximately off (see MW_STAND_DIST): as borrow's,
        plus "pot" and "stack" (this pot's), or None."""
        from .flop_approx import range_cover, range_dist
        if len(ranges) != 3 or pot <= 0:
            return None
        fps = tuple(fl.range_fp(r) for r in ranges)
        key = ("stand", fps, round(pot, 2), round(stack, 2))
        if key not in self._borrow:
            best = None
            for sp in self.index().get("spots", []):
                q = (stack / pot) / (float(sp["stack"]) / float(sp["pot"]))
                if not MW_STAND_SPR[0] <= q <= MW_STAND_SPR[1]:
                    continue
                st = _spot_ranges(sp)
                if st is None:
                    continue
                cover = tuple(range_cover(np.asarray(x), y) for x, y in zip(ranges, st["ranges"]))
                dist = tuple(range_dist(np.asarray(x), y) for x, y in zip(ranges, st["ranges"]))
                if min(cover) >= MW_STAND_COVER and sum(dist) <= MW_STAND_DIST and \
                        (best is None or sum(dist) < sum(best[2])):
                    best = (sp["id"], cover, dist)
            self._borrow[key] = best
        got = self._borrow[key]
        sp = got and next((x for x in self.index().get("spots", []) if x["id"] == got[0]), None)
        return {"spot": sp, "cover": got[1], "dist": got[2], "pot": round(float(pot), 2),
                "stack": round(float(stack), 2)} if sp else None

    def _pick(self, ranges, pot, stack, players) -> tuple:
        """(spot, borrow) of a 3-way pot: the library's own spot of its ranges, else the one borrow reads it as
        (with `players`, the table's size: None for ranges of your own), else (None, None)."""
        spot = self.spot([fl.range_fp(r) for r in ranges], pot, stack)
        if spot is not None or not players:
            return spot, None
        b = self.borrow(ranges, pot, stack, players)
        return (b["spot"], b) if b else (None, None)

    @staticmethod
    def _mark(sol, b):
        if sol is not None and b:
            sol.borrowed = {"spot": b["spot"]["id"], "cover": [round(float(c), 4) for c in b["cover"]],
                            "dist": [round(float(d), 4) for d in b["dist"]],
                            **({"pot": b["pot"], "stack": b["stack"]} if "pot" in b else {})}
            sol.label = f"6-max {sol.label}"    # (the library's own game: 6-max 100bb)
        return sol

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
            if e is not None and (e["head"].get("fp") != spot["fp"]
                                  or e["head"].get("tree") != fl.tree_hash(self.index().get("tree") or {})):
                e = None
            if len(self._entries) > 32:
                self._entries.clear()
            self._entries[key] = e
        return self._entries[key]

    def solution(self, ranges: Sequence[np.ndarray], board: Sequence[int], pot: float, stack: float,
                 rake: bool = False, players: Optional[int] = None) -> Optional[LibraryMWSolution]:
        """The stored solve of this 3-way flop, or None. ranges: the three players' ranges entering the flop, in
        postflop order (they pick the spot); rake: the hand is raked (the library isn't); players: the table's size,
        to read another table size's pot as a spot's (borrow; the solution's `borrowed` says so), None not to."""
        if len(board) != 3 or len(ranges) != 3 or rake:
            return None
        idx = self.index()
        if not idx.get("spots"):
            return None
        tk = idx.get("tree") or {}
        if tk.get("version") != MODEL_VERSION or tk.get("rake"):
            return None
        spot, bw = self._pick(ranges, pot, stack, players)
        if spot is None:
            return None
        canon, smap = fl.canonical(board)
        e = self.entry(spot, cards_str(canon))
        if e is None:
            return None
        return self._mark(self._from_entry(spot, e, ranges, board, smap), bw)

    def _from_entry(self, spot: dict, e: dict, ranges, board, smap) -> Optional[LibraryMWSolution]:
        """A decoded entry of the canonical flop read for the hand's flop `board` (smap: its suits' canonical ones)."""
        from .multiway_solver import Decision
        from .solver import MEMBER_FULL
        cmap = fl.combo_map(smap)                       # actual combo -> canonical combo
        bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        rf = [np.asarray(r, dtype=np.float64) * bm for r in ranges]
        cover = np.any(e["masks"], axis=0)[cmap]          # (actual combos the entry holds for some player)
        active = np.nonzero(np.any([r > 0 for r in rf], axis=0) | (cover & (bm > 0)))[0]
        canon_active = cmap[active]
        cols = []
        for p in range(3):
            m = e["masks"][p]
            if not m[canon_active[rf[p][active] > 0]].all():
                return None
            pos = np.full(len(ALL_COMBOS), -1)
            pos[np.nonzero(m)[0]] = np.arange(int(m.sum()))
            cols.append(pos[canon_active])
        n = len(active)
        nodes: Dict[tuple, object] = {}
        evs: Dict[int, np.ndarray] = {}
        for nd in e["nodes"]:
            d = Decision.__new__(Decision)
            A = len(nd["names"])
            c = cols[nd["player"]]
            ok = c >= 0
            s = np.full((1, A, n), 1.0 / A, dtype=np.float32)
            s[0][:, ok] = nd["strat"][:, c[ok]]
            ev = np.zeros((A, n))
            ev[:, ok] = nd["evs"][:, c[ok]]
            d.bd, d.player, d.names = None, nd["player"], list(nd["names"])
            d.children = [UNSTORED] * A
            d.ssum, d.regret = s, s.copy()
            d.pot_now, d.c, d.street = float(nd["pot_now"]), [0.0] * 3, "flop"
            d.amounts = list(nd["amounts"])
            nodes[tuple(nd["path"])] = d
            evs[id(d)] = ev
        if () not in nodes:
            return None
        for path, d in nodes.items():
            if path:
                par = nodes.get(path[:-1])
                if par is None or path[-1] not in par.names:
                    return None
                par.children[par.names.index(path[-1])] = d
        reach0 = np.stack([r[active] for r in rf])[:, None, :]
        return LibraryMWSolution(nodes, evs, active, reach0, e["head"], spot.get("label", spot["id"]),
                                 spot.get("seats") or [], spot.get("stack"))


    def near(self, ranges: Sequence[np.ndarray], board: Sequence[int], pot: float, stack: float,
             rake: bool = False, players: Optional[int] = None) -> Optional[LibraryMWSolution]:
        """A 3-way flop of a library spot the library doesn't store, read off the spot's stored flops most like it,
        as the heads-up library's (gto/flop_approx.py: the K nearest of the same pairing and suit pattern, each carried
        over by card substitution, a combo's mix the average of its images' that hold its hand class; not the learned
        correction, fit on heads-up flops): a LibraryMWSolution of the STORED decisions with those mixes, `approx` set
        ({"near": [(stored flop in the hand's suits, distance)], "known": {path: (n,) bool, the combos some source
        holds an image of}}), no EVs (zeros) or gains; None where the solve answers (a stored flop), the spot isn't
        held, or no stored flop is near enough. Leaving each stored flop out (scripts/bench_mw_approx.py), the
        STORED decisions read off the three nearest others give up 0.14% of the pot on average (by the flop's own
        EVs; median 0.11, 90th percentile 0.33). players: as solution's (another table size's pot read off a spot)."""
        from collections import OrderedDict
        from . import flop_approx as FA
        from .solver import MEMBER_FULL
        if len(board) != 3 or len(ranges) != 3 or rake:
            return None
        idx = self.index()
        tk = idx.get("tree") or {}
        if not idx.get("spots") or tk.get("version") != MODEL_VERSION or tk.get("rake"):
            return None
        spot, bw = self._pick(ranges, pot, stack, players)
        if spot is None and players:
            bw = self.stand_in(ranges, pot, stack)
            spot = bw and bw["spot"]
        if spot is None:
            return None
        canon, smap = fl.canonical(board)
        if cards_str(canon) in spot.get("flops", {}):
            if bw is None or "pot" not in bw:
                return None          # (the solve answers)
            e = self.entry(spot, cards_str(canon))    # (another pot's stand-in: the stored flop's mixes, no EVs)
            if e is None:
                return None
            nodes = [dict(nd, evs=np.zeros(nd["strat"].shape)) for nd in e["nodes"]]
            head = dict(e["head"], iters=0, expl=0.0, gains=[0.0] * 3)
            sol = self._from_entry(spot, {"head": head, "masks": e["masks"], "nodes": nodes}, ranges, board, smap)
            if sol is None:
                return None
            sol.approx = {"near": [(cards_str(board), 0)], "known": {}, "stored": True}
            return self._mark(sol, bw)
        src = FA.sources(self, spot, list(canon))
        if not src:
            return None
        cmap = fl.combo_map(smap)
        bm = np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0)
        combos = np.unique(cmap[np.flatnonzero(np.any([np.asarray(r) * bm > 0 for r in ranges], axis=0))])
        key = (spot["id"], cards_str(canon), tuple((nm, d) for *_, nm, d in src), combos.tobytes())
        kept = self.__dict__.setdefault("_near_pred", OrderedDict())
        pred = kept.get(key)
        if pred is None:
            pred = kept[key] = FA.predict([(x, m, w) for x, m, w, _, _ in src], board=list(canon), combos=combos)
            while len(kept) > FA.PRED_KEEP:
                kept.popitem(last=False)
        off = np.prod([1.0 - MEMBER_FULL[c] for c in canon], axis=0) > 0    # (canonical combos off the flop)
        e0 = src[0][0]
        nodes, known = [], {}
        for nd in e0["nodes"]:
            o = pred.get(tuple(nd["path"]))
            if o is None or o["names"] != list(nd["names"]) or o["player"] != nd["player"]:
                continue
            st = o["strat"][:, off]
            known[tuple(nd["path"])] = ~np.isnan(o["strat"][0])
            nodes.append(dict(nd, strat=np.where(np.isnan(st), 1.0 / len(nd["names"]), st), evs=np.zeros(st.shape)))
        head = dict(e0["head"], board=cards_str(canon), iters=0, expl=0.0, gains=[0.0] * 3)
        sol = self._from_entry(spot, {"head": head, "masks": [off] * 3, "nodes": nodes}, ranges, board, smap)
        if sol is None:
            return None
        inv = [0] * 4                                   # canonical suit -> the hand's
        for a, b in enumerate(smap):
            inv[b] = a
        ca = cmap[sol.hands.active]
        sol.approx = {"near": [(cards_str(sorted((c // 4 * 4 + inv[c % 4] for c in parse_cards(nm)), reverse=True)), d)
                               for *_, nm, d in src],
                      "known": {path: k[ca] for path, k in known.items()}}
        return self._mark(sol, bw)


_DEFAULT: Optional[MWLibrary] = None


def default_library() -> Optional[MWLibrary]:
    """The 3-way library in gto/static/flop_library (None when it has no index)."""
    global _DEFAULT
    if _DEFAULT is None and os.path.exists(os.path.join(DIR, INDEX)):
        _DEFAULT = MWLibrary.from_dir(DIR)
    return _DEFAULT
