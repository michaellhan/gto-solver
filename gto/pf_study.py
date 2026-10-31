"""Preflop solves for Study: pf_game's equilibrium for the page's own game, browsed node by node with every hand
class's mix and EVs (Study's "Solver" source preflop; its "Charts" source is the hand-written charts' mix).

The game (Spec, from the state as the page sends it: players, stack, ante, straddle, rake, and the raise sizes) is
gto/pf_game.py's tree for that table with its flops valued as scripts/build_pf_formats.py values the other games'
(pf_game.LibraryModel: the flop library's realization, advisor._river_fix's river correction, another depth's
excess over equity scaled by DEPTH_SCALE). Heads-up the small blind may also limp (pf_game's sb_limp), as the
heads-up charts' games do; from 7 players at most two see a flop (the 7-handed charts' games' tree), and 8-9 handed
trees are too big to hold (an 8-handed tree is 50k nodes: ~540 MB an array of reaches).

A solve runs in chunks of a few seconds the page asks for one after another (Advisor.study's "preflop_solve":
{"run": seconds}), each going on from the last (the worker keeps the Solve), until it is within TARGET_MBB of
equilibrium (mbb per hand per seat, what a seat gains on average by best-responding: Game.exploitability) or MAX_ITERS.
The standard games Study starts from (6-max 100bb, unraked and 5% capped at 3bb) are solved offline to 1500 iterations
and stored (gto/static/pf/<key>.bin, ~300 KB: each node's mix in 1/255 steps; scripts/build_pf_study.py) so they
answer at once.

Each node's answer reads the solve's average strategy and its counterfactual values: per class its share of the
range there (w), its mix (f), each action's EV (bb from here, relative to folding now: chips won net of what's in,
plus what folding gives up; the counterfactual value over the others' card-removal masses) and the mix's EV; per
action the range's frequency and combos (the mix's) and EV (each class weighted by its combos times its chance
against the others' ranges); the other players still in who have acted (their ranges); how much the range gives up
at the node to its best actions there (regret_bb), and "rough" where the line is one the solve barely plays. A
player the solve never brings to the node (a 0% move on the way) is taken to make that move with every hand it held
(Solve._at). EVs are the solve's own, at its equilibrium: they go with its mix, not the charts' (which the advice
plays and which fit published solves better: README, Preflop equilibrium). Checked against a direct computation
over class pairs wherever two players are left (tests/test_pf_study.py)."""
from __future__ import annotations
import base64
import os
import sys
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from . import pf_game as G
from .pf_classes import CLASSES, N_IN, PAIR_COMPAT

K = len(CLASSES)
MAX_PLAYERS = 7            # 8-9 handed trees don't fit (see the module's doc) ...
BROWSER_PLAYERS = 6        # ... and in the browser 7-handed (~15 minutes, over 1 GB of arrays) is left to native
TARGET_MBB = 2.0           # a solve stops this close to equilibrium (mbb/hand a seat, on average) ...
MAX_ITERS = 2000           # ... or after this many iterations
SMALL_WORK = 5000          # a tree this small (nodes x seats: 2-4 handed) solves to MAX_ITERS, seconds in all
CHECK_EVERY = 100          # iterations between exploitability checks (each ~2.5 up-and-down passes a seat)
STORED_ITERS = 1500
MAX_HELD = 3               # solves kept (a 6-max one holds ~60 MB)
ROUGH_BB = 0.05            # a node is flagged rough from this loss to its best actions (bb a hand of the range)
LIVE_W = 0.002             # a class shows at a node from this share of its combos (the average strategy leaves crumbs)

# {depth: {pot type: k}}: each share's excess over equity at that depth is k times 100bb's (scripts/pf_depth_values.py
# --fit: live solves of BTN vs BB, SB vs BB and BB vs BTN's 3bet pot on 12 of the 184 representative flops at 150
# iterations against the library's at 100bb; linear in depth between, 1 at 100). Single-raised pots barely move
# (BTN vs BB 0.90 / 0.98 / 0.96, SB vs BB 0.88 / 0.98 / 0.95 at 20 / 40 / 200bb), 3bet pots realize less as the
# stack shrinks to the pot (0.48 at 20bb, 0.76 at 40, 1.03 at 200); 4bet pots (type 3, no measure) as 3bet pots
DEPTH_SCALE = {20.0: {1: 0.890, 2: 0.475, 3: 0.475},
               40.0: {1: 0.977, 2: 0.762, 3: 0.762},
               100.0: {1: 1.0, 2: 1.0, 3: 1.0},
               200.0: {1: 0.956, 2: 1.034, 3: 1.034}}
RIVER_FIX = (1.0, 0.5)     # advisor._river_fix's scale in single-raised and 3bet pots (as build_pf_formats' games)


def depth_scale(depth: float) -> dict:
    ds = sorted(DEPTH_SCALE)
    x = min(max(depth, ds[0]), ds[-1])
    lo, hi = max(d for d in ds if d <= x), min(d for d in ds if d >= x)
    t = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return {k: (1 - t) * DEPTH_SCALE[lo].get(k, 1.0) + t * DEPTH_SCALE[hi].get(k, 1.0) for k in (1, 2, 3)}


def flop_model(stack: float) -> G.EQRModel:
    """The flop valuation the other games' solves use (build_pf_formats.game)."""
    from .advisor import _river_fix
    return G.LibraryModel(river_fix=_river_fix(), river_scale={1: RIVER_FIX[0], 2: RIVER_FIX[1]},
                          scale=depth_scale(stack))


# ----------------------------------------------------------------------------- the game
SIZE_KEYS = ("open", "sb_open", "ip_3bet", "blind_3bet", "four_x")
SIZE_LIMITS = {"open": (2.0, 5.0), "sb_open": (2.0, 5.0), "ip_3bet": (2.0, 5.0), "blind_3bet": (2.5, 6.0),
               "four_x": (1.8, 3.5)}


@dataclass(frozen=True)
class Spec:
    players: int = 6
    stack: float = 100.0
    ante: float = 0.0
    straddle: bool = False
    rake: Optional[Tuple[float, Optional[float]]] = None
    sizes: Tuple[Tuple[str, float], ...] = ()       # Sizes fields changed from the defaults

    @property
    def key(self) -> str:
        """A file-name-safe id: '6max_100bb', '6max_100bb_r5c3', '9max_40bb_a0.125_s'..."""
        k = f"{self.players}max_{self.stack:g}bb"
        if self.ante:
            k += f"_a{self.ante:g}"
        if self.straddle:
            k += "_s"
        if self.rake:
            k += f"_r{self.rake[0] * 100:g}" + (f"c{self.rake[1]:g}" if self.rake[1] else "")
        for n, v in self.sizes:
            k += f"_{n}{v:g}"
        return k

    def label(self) -> str:
        bits = [f"{self.players} players" if self.players > 2 else "heads-up", f"{self.stack:g}bb"]
        if self.ante:
            bits.append(f"{self.ante:g}bb ante")
        if self.straddle:
            bits.append("straddle")
        bits.append(f"rake {self.rake[0] * 100:g}%" + (f" cap {self.rake[1]:g}bb" if self.rake[1] else " no cap")
                    if self.rake else "no rake")
        sz = self.sizes_obj()
        bits.append(f"opens {sz.open:g}bb (SB {sz.sb_open:g}), 3bets {sz.ip_3bet:g}x / {sz.blind_3bet:g}x from the blinds, "
                    f"4bets {sz.four_x:g}x")
        return ", ".join(bits)

    def sizes_obj(self) -> G.Sizes:
        s = G.Sizes()
        for n, v in self.sizes:
            setattr(s, n, v)
        return s

    def positions(self) -> List[str]:
        from .preflop import SEATS
        return list(SEATS[self.players])

    def build(self) -> Tuple[G.Node, List[str]]:
        pos = self.positions()
        root = G.build(pos, stack=self.stack, sizes=self.sizes_obj(), max_players=3 if self.players <= 6 else 2,
                       ante=self.ante, straddle=self.straddle, sb_limp=self.players == 2)
        return root, pos

    def game(self) -> G.Game:
        root, pos = self.build()
        return G.Game(root, pos, flop_model(self.stack), stack=self.stack, rake=self.rake)


def spec_of(state: dict) -> Tuple[Optional[Spec], Optional[str]]:
    """The game of a Study request (amounts in big blinds, as the page sends them), or (None, why it can't be solved)."""
    bb = float(state.get("bb", 1.0))
    try:
        n = int(state.get("players", 6))
    except (TypeError, ValueError):
        return None, "players must be 2 to 9"
    if not 2 <= n <= 9:
        return None, "players must be 2 to 9"
    if n > MAX_PLAYERS:
        return None, f"{n}-handed preflop is too big to solve here (up to {MAX_PLAYERS} players)"
    if n > BROWSER_PLAYERS and sys.platform == "emscripten":
        return None, (f"{n}-handed preflop is too big to solve in the browser (up to {BROWSER_PLAYERS} players; "
                      f"{MAX_PLAYERS} with the local server, python -m gto.server)")
    if state.get("payouts") or state.get("bounties"):
        return None, "a tournament's preflop isn't solved here (the solve plays chips)"
    stacks = state.get("stacks")
    stack = float(state.get("stack") or 100.0) / bb
    if stacks:
        vals = [float(v) / bb for v in (stacks.values() if isinstance(stacks, dict) else stacks) if v is not None]
        if vals and max(vals) - min(vals) > 1e-6:
            return None, "the solve plays one stack depth for every seat (set the stacks equal)"
        if vals:
            stack = vals[0]
    if not 10.0 <= stack <= 250.0:
        return None, "the solve takes stacks of 10 to 250bb"
    ante = float(state.get("ante") or 0.0) / bb
    straddle = bool(state.get("straddle")) and n >= 3
    rate = max(0.0, min(0.25, float(state.get("rake") or 0.0)))
    cap = state.get("rake_cap")
    cap = float(cap) / bb if cap not in (None, "") and float(cap) > 0 else None
    sizes = []
    for k, v in sorted((state.get("pf_sizes") or {}).items()):
        if k not in SIZE_LIMITS:
            return None, f"unknown preflop size {k!r} (sizes: {', '.join(SIZE_KEYS)})"
        lo, hi = SIZE_LIMITS[k]
        v = round(float(v), 2)
        if not lo <= v <= hi:
            return None, f"preflop size {k} must be {lo:g} to {hi:g}"
        if abs(v - getattr(G.Sizes(), k)) > 1e-9:
            sizes.append((k, v))
    return Spec(n, round(stack, 2), round(ante, 4), straddle, (rate, cap) if rate > 0 else None, tuple(sizes)), None


# ----------------------------------------------------------------------------- the stored solves
STORED = ("6max_100bb", "6max_100bb_r5c3")


def pack(g: G.Game, mbb: float = float("nan")) -> bytes:
    """A solve's average strategy: per decision node (breadth-first) each action after the first, per class, in 1/255
    steps (the first is what's left), zlib'd; classes the player never brings to the node fold (their mix is
    arbitrary). With a 12-byte header: b'PFS1', the node count and the stored strategy's exploitability (mbb/hand a
    seat, float32; NaN where not measured)."""
    s = g.average()
    reach = g._down(s)
    nA = g.valid.sum(1)
    parts = []
    for d, k in enumerate(g.dec):
        x = np.round(s[d, 1:nA[d]] * 255).astype(np.uint8)
        x[:, reach[k, g.player[d]] <= 1e-7] = 0
        parts.append(x.ravel())
    raw = np.concatenate(parts).tobytes()
    return b"PFS1" + np.uint32(len(g.dec)).tobytes() + np.float32(mbb).tobytes() + zlib.compress(raw, 9)


def stored_mbb(blob: bytes) -> Optional[float]:
    """pack()'s exploitability in the header (None where not measured)."""
    x = float(np.frombuffer(blob[8:12], np.float32)[0])
    return None if x != x else x


def unpack(g: G.Game, blob: bytes) -> np.ndarray:
    """pack()'s strategy (D, A, K) for g's tree."""
    if blob[:4] != b"PFS1" or int(np.frombuffer(blob[4:8], np.uint32)[0]) != len(g.dec):
        raise ValueError("the stored preflop solve isn't this tree's")
    q = np.frombuffer(zlib.decompress(blob[12:]), np.uint8).astype(np.float64) / 255.0
    s = np.zeros((len(g.dec), g.A, K))
    nA = g.valid.sum(1)
    i = 0
    for d in range(len(g.dec)):
        m = (nA[d] - 1) * K
        x = q[i:i + m].reshape(nA[d] - 1, K)
        i += m
        s[d, 1:nA[d]] = x
        s[d, 0] = np.maximum(0.0, 1.0 - x.sum(0))
    if i != len(q):
        raise ValueError("the stored preflop solve's size doesn't match the tree")
    return s / np.maximum(s.sum(1, keepdims=True), 1e-12)


def _read_stored(key: str) -> Optional[bytes]:
    """The stored solve's file: in the browser from the site (pf/<key>.bin, the worker's siteFetch), else from
    gto/static/pf/."""
    if key not in STORED:
        return None
    if sys.platform == "emscripten":
        try:
            import js   # noqa: F401
            r = js.siteFetch(f"pf/{key}.bin")
            return None if r is None else r.to_bytes()
        except Exception:   # noqa: BLE001
            return None
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "pf", key + ".bin")
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


# ----------------------------------------------------------------------------- a solve
class Solve:
    """One game's solve, run in chunks (run()), and its nodes' answers (node())."""

    def __init__(self, spec: Spec, stored: Optional[bytes] = None):
        self.spec = spec
        self.game = spec.game()
        self.target = 0.0 if self.game.N * self.game.n < SMALL_WORK else TARGET_MBB
        self.iters = 0
        self.seconds = 0.0
        self.expl: Optional[List[float]] = None       # bb/hand each seat gains by best-responding
        self.expl_at = -1
        self.stored = False
        self.cached = False      # (a solve run earlier in the browser and kept there: load())
        self.stored_mbb: Optional[float] = None
        self._vals = None
        if stored is not None:
            s = unpack(self.game, stored)
            self.game.ssum = s.copy()
            self.game.regret = np.zeros_like(s)
            self.iters, self.stored, self.stored_mbb = STORED_ITERS, True, stored_mbb(stored)

    @property
    def mbb(self) -> Optional[float]:
        """Mean exploitability a seat, mbb/hand (a stored solve's as measured when it was stored)."""
        if self.expl is None:
            return self.stored_mbb
        return 1000.0 * sum(self.expl) / len(self.expl)

    @property
    def done(self) -> bool:
        return self.stored or self.iters >= MAX_ITERS or (self.mbb is not None and self.expl_at == self.iters
                                                          and self.mbb <= self.target)

    def measure(self):
        if self.expl_at != self.iters:
            self.expl = self.game.exploitability(per_seat=True)
            self.expl_at = self.iters

    def run(self, seconds: float, should_stop: Optional[Callable[[], bool]] = None) -> "Solve":
        """Iterate for about `seconds` (whole iterations; an exploitability check every CHECK_EVERY), stopping once
        done."""
        t0 = time.time()
        while not self.done:
            self.game.iterate(self.iters + 1)
            self.iters += 1
            self._vals = None
            if self.iters % CHECK_EVERY == 0 or self.iters >= MAX_ITERS:
                self.measure()
            if time.time() - t0 >= seconds or (should_stop is not None and should_stop()):
                break
        self.seconds += time.time() - t0
        return self

    def status(self) -> dict:
        return {"key": self.spec.key, "label": self.spec.label(), "iterations": self.iters, "done": self.done,
                "max_iterations": MAX_ITERS, "target_mbb": self.target, "stored": self.stored and not self.cached,
                "cached": self.cached,
                "mbb": None if self.mbb is None else round(self.mbb, 2),
                "measured_at": self.expl_at if self.expl is not None else (self.iters if self.mbb is not None else None),
                "seconds": round(self.seconds, 1)}

    # ---------------------------------------------------------------- answers
    def _values(self):
        """The average strategy, its reaches, masses and counterfactual values (cached until the next iteration)."""
        if self._vals is None:
            g = self.game
            s = g.average()
            reach = g._down(s)
            with np.errstate(all="ignore"):
                V = g._up(reach, s)
                M = g._masses(reach)
            self._vals = (s, reach, M, V)
        return self._vals

    def _at(self, k: int):
        """_values() for node k's answer, and the seats the solve never brings there (their reach 0: a line with a 0%
        move of theirs). Such a seat is taken to make its first 0% move on the way with every hand it holds there (the
        strategy above k changed for this answer alone), so its range at k is as before that move and the EVs aren't
        0/0 (and a player folded on the way doesn't take every hand out by card removal)."""
        s, reach, M, V = self._values()
        g = self.game
        low = [q for q in range(g.n) if reach[k, q].sum() < 1e-9]
        if not low:
            return s, reach, M, V, []
        path, x = [], k
        while g.parent[x] >= 0:
            path.append((int(g.parent[x]), int(g.act[x]), x))
            x = int(g.parent[x])
        path.reverse()
        s2, r2 = s.copy(), reach
        for _ in range(4 * len(path)):
            q = next((q for q in low if r2[k, q].sum() < 1e-9), None)
            if q is None:
                break
            par, a, ch = next((par, a, ch) for par, a, ch in path
                              if g.nodes_all[par].player == q and r2[ch, q].sum() < 1e-9 <= r2[par, q].sum())
            d = g.dec_of[par]
            s2[d] = 0.0
            s2[d, a] = 1.0
            r2 = g._down(s2)
        with np.errstate(all="ignore"):
            V2 = g._up(r2, s2)
            M2 = g._masses(r2)
        return s2, r2, M2, V2, [g.positions[q] for q in low]

    def walk(self, preflop: List[dict]) -> Tuple[Optional[G.Node], List[str], Dict[str, float]]:
        """The node a Study line reaches ({"pos", "action", "to"?} in big blinds), the notes on it (a raise of
        another size than the tree's), and each seat's chips in; (None, [why], {}) off the tree."""
        g = self.game
        node = g.root
        pos = g.positions
        bets = {p: 0.0 for p in pos}
        sb, bb = (pos[-2], pos[-1]) if len(pos) > 2 else (pos[0], pos[1])
        bets[sb] += 0.5
        bets[bb] += 1.0
        if self.spec.straddle and len(pos) >= 3:
            bets[pos[0]] += 2.0
        notes = []
        for a in preflop:
            if node.kind != "dec":
                return None, ["the line goes on past where the solve's betting ends"], {}
            who = pos[node.player]
            if str(a.get("pos")).upper() != who:
                return None, [f"{who} acts here in the solve's tree, not {a.get('pos')}"], {}
            act = str(a.get("action")).lower()
            want = {"fold": ("fold",), "call": ("call",), "check": ("call",), "raise": ("raise", "all-in"),
                    "bet": ("raise", "all-in"), "all-in": ("all-in", "raise")}.get(act, ())
            idx = next((node.names.index(w) for w in want if w in node.names), None)
            if idx is None:
                return None, [f"the solve's tree has no {act} for {who} here (it lets at most "
                              f"{3 if self.spec.players <= 6 else 2} players see a single-raised flop, two a raised one)"
                              if act in ("call", "check") else f"the solve's tree has no {act} for {who} here"], {}
            child = node.children[idx]
            if node.names[idx] in ("raise", "all-in", "call"):
                to = child.line[-1][2]
                want_to = a.get("to", a.get("amount"))
                if node.names[idx] != "call" and want_to is not None and abs(float(want_to) - to) > 0.051 \
                        and not (node.names[idx] == "all-in" and act == "all-in"):
                    notes.append(f"{who}'s raise to {float(want_to):g}bb is played as the solve's size, {to:g}bb")
                bets[who] = to
            node = child
        return node, notes, bets

    def node(self, preflop: List[dict]) -> dict:
        """Study's answer at the node the line reaches (see the module's doc), or {"error"}/{"study": "deal"/"over"}."""
        node, notes, bets = self.walk(preflop)
        if node is None:
            return {"error": notes[0]}
        g = self.game
        pos = g.positions
        if node.kind == "fold":
            return {"study": "over", "street": "preflop", "winner": pos[node.active[0]]}
        if node.kind == "show":
            return {"study": "deal", "street": "flop", "allin": bool(node.allin)}
        s, reach, M, V, never = self._at(node.index)
        d = int(g.dec_of[node.index])
        k = node.index
        p = node.player
        ante = self.spec.ante
        norm = np.ones(K)
        for q in range(g.n):
            if q != p:
                norm = norm * M[k, q]
        inv = bets[pos[p]] + ante
        cur = max(bets.values())
        nA = len(node.names)
        evs = np.zeros((nA, K))
        with np.errstate(all="ignore"):
            for a in range(nA):
                evs[a] = V[node.children[a].index, p] / np.maximum(norm, 1e-300) + inv
        w = reach[k, p] / G.PRIOR
        mix = s[d, :nA]
        live = w > LIVE_W
        if not live.any():   # (a range of crumbs, a line the solve all but never takes: its largest classes)
            live = w >= w.max() * 0.01
        evm = (mix * evs).sum(0)
        order = sorted(range(nA), key=lambda a: {"raise": 0, "all-in": 0, "call": 1, "fold": 2}[node.names[a]])
        names, amounts = [], []
        for a in order:
            nm = node.names[a]
            if nm == "call" and cur - bets[pos[p]] < 1e-9:
                nm = "check"
            names.append(nm)
            amounts.append(node.children[a].line[-1][2] if nm in ("raise", "all-in") else None)
        combos = w * N_IN
        tot = float(combos[live].sum())
        # (the range's and an action's EV weigh each class by its combos times its chance against the others' ranges,
        # card removal: what the player holds here, given what they hold; frequencies and combos are the mix's alone)
        held = combos * norm / max(float(norm[live].max()), 1e-300)
        acts = []
        for j, a in enumerate(order):
            c = combos * mix[a]
            share = float(c[live].sum())
            h = (held * mix[a])[live]
            acts.append({"action": names[j], "amount_bb": None if amounts[j] is None else round(amounts[j], 2),
                         "freq": round(share / max(tot, 1e-12), 4), "combos": round(share, 1),
                         "ev_bb": round(float((h * evs[a][live]).sum() / h.sum()), 3) if share > 1e-9 else None})
        cells = {}
        for i in np.nonzero(live)[0]:
            cells[CLASSES[i]] = {"w": round(float(min(1.0, w[i])), 4),
                                 "f": [round(float(mix[a, i]), 4) for a in order],
                                 "ev": [round(float(evs[a, i]), 3) for a in order],
                                 "evm": round(float(evm[i]), 3), "eq": None}
        # the other players still in who have acted (their ranges at this node)
        folded = {x for x, act, _ in node.line if act == "fold"}
        acted = []
        for x, act, _ in node.line:
            if act != "fold" and x not in folded and x not in acted and x != pos[p]:
                acted.append(x)
        opps = []
        for x in acted:
            q = pos.index(x)
            wq = reach[k, q] / G.PRIOR
            lq = wq > LIVE_W
            opps.append({"seat": x, "combos": round(float((wq * N_IN)[lq].sum()), 1),
                         "grid": {"actions": [], "cells": {CLASSES[i]: {"w": round(float(min(1.0, wq[i])), 4)}
                                                            for i in np.nonzero(lq)[0]}}})
        pot = sum(bets.values()) + ante * g.n
        st = self.status()
        # how far the range's mix here is from its best actions against the others' (bb a hand of the range: the
        # node's own share of exploitability, its continuation held), and whether the line is one the solve barely
        # plays: a range it never brings here, or one of under a combo
        regret = float((held * (evs.max(0) - evm))[live].sum() / max(float(held[live].sum()), 1e-300))
        if never:
            notes.append(f"the solve never brings {' and '.join(never)} here: "
                         + ("their ranges are" if len(never) > 1 else "the range is") + " as before the 0% move")
        thin = [o["seat"] for o in opps if o["combos"] < 1.0] + ([pos[p]] if tot < 1.0 else [])
        return {"study": "node", "street": "preflop", "actor": pos[p], "players": g.n, "board": "",
                "pot_bb": round(pot, 2), "to_call_bb": round(cur - bets[pos[p]], 2),
                "stack_bb": round(self.spec.stack - ante - bets[pos[p]], 2),
                "actions": acts, "grid": {"actions": names, "cells": cells}, "range_combos": round(tot, 1),
                "range_ev_bb": round(float((held * evm)[live].sum() / max(float(held[live].sum()), 1e-300)), 3),
                "range_eq": None,
                "opponents": opps, "opponent": " and ".join(acted) if acted else None,
                "solver": {"by": "the preflop solve", "iterations": self.iters,
                           "mbb": st["mbb"], "exploitability_bb": None if self.expl is None else round(sum(self.expl), 4)},
                "pf_solve": st, "source": "solver", "notes": notes, "regret_bb": round(regret, 3),
                "rough": bool(never or thin or regret >= ROUGH_BB)}


# ----------------------------------------------------------------------------- the solves held
_HELD: Dict[str, Solve] = {}


def held(spec: Spec, load: bool = True) -> Optional[Solve]:
    """The game's solve in hand: one run here, else a stored one (load), else None."""
    s = _HELD.get(spec.key)
    if s is None and load:
        blob = _read_stored(spec.key)
        if blob is not None:
            s = _keep(Solve(spec, blob))
    return s


def _keep(s: Solve) -> Solve:
    _HELD.pop(s.spec.key, None)
    _HELD[s.spec.key] = s
    while len(_HELD) > MAX_HELD:
        _HELD.pop(next(iter(_HELD)))
    return s


def solve(spec: Spec, seconds: float, should_stop: Optional[Callable[[], bool]] = None) -> Solve:
    """Runs the game's solve for about `seconds` more (started here if none is in hand)."""
    s = held(spec) or _keep(Solve(spec))
    _keep(s)
    return s.run(seconds, should_stop)


def export(spec: Spec) -> Optional[dict]:
    """A finished solve run here, packed for the page to keep (pack(), base64; Study keeps it in the browser's
    IndexedDB and hands it back with load() once the worker no longer holds it): {"key", "iterations", "mbb",
    "blob"}, or None (none run here, or not finished)."""
    s = held(spec, load=False)
    if s is None or s.stored or not s.done:
        return None
    mbb = s.mbb
    blob = pack(s.game, float("nan") if mbb is None else mbb)
    return {"key": spec.key, "iterations": s.iters, "mbb": None if mbb is None else round(mbb, 3),
            "blob": base64.b64encode(blob).decode("ascii")}


def load(spec: Spec, blob: str, iterations: int) -> Solve:
    """export()'s solve back in hand (the page's copy): its strategy as stored, its iterations and distance from
    equilibrium as measured when it was run. ValueError where it isn't this game's tree."""
    try:
        s = Solve(spec, base64.b64decode(blob))
    except zlib.error as e:
        raise ValueError(f"not a packed solve: {e}") from e
    s.iters, s.cached = int(iterations), True
    return _keep(s)


def estimate_seconds(spec: Spec, per_iter: Optional[float] = None) -> float:
    """Roughly how long a solve takes: iterations (~1050 to TARGET_MBB at 6-max; MAX_ITERS for a small tree) times
    the tree's iteration (measured: 0.12 s natively at 6-max, 0.35 s in Pyodide; in proportion to the tree's nodes
    and seats), plus the exploitability checks (~2.5 passes a seat each)."""
    root, pos = spec.build()
    work = sum(1 for _ in G.walk(root)) * len(pos)
    base = per_iter if per_iter is not None else (0.35 if sys.platform == "emscripten" else 0.12)
    it = base * work / (11010 * 6)
    n = MAX_ITERS if work < SMALL_WORK else 1050
    return it * n + it * 2.5 * len(pos) * (n // CHECK_EVERY)
