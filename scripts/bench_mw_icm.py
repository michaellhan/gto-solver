"""3-way pots in a tournament: how close the solver's prize payoffs (gto/icm.py MWPayoffs: each terminal's value linear in
the shares, the outcomes averaged over the orders of the live hands) come to exact ICM, on a 3-way river solved per
combo with every card removal counted.

The exact game pays every deal its own outcome: who wins the pot (ties split), every seat's stack after it, each
player's prize equity (Malmuth-Harville over the table). Both solves (chips and prizes) are played in it: each player's
value and its best response's (per combo, over the same tree), so what a solve's strategies give up in prizes. The
ranges are thinned (every k-th combo of each player's, by strength) so the deals can be summed one by one.

    PYTHONPATH=. python scripts/bench_mw_icm.py [--iters 600] [--keep 60]
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gto.advisor as A
from gto import icm
from gto import preflop as pf
from gto.cards import ALL_COMBOS, cards
from gto.evaluator import eval_board
from gto.multiway_solver import MWConfig, MultiwaySolver, Decision, Terminal

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
       {"pos": "BTN", "action": "call"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
SEATS = ("BB", "CO", "BTN")


class Exact:
    """The exact prize game on a river solve's tree: per terminal and set of tied winners, every player's value."""

    def __init__(self, s: MultiwaySolver, pay: icm.MWPayoffs, board, base):
        self.s, self.pay, self.base = s, pay, base     # (base: each player's origin, MultiwaySolver.icm_base)
        act = s.hands.active
        st = eval_board(board)[act]
        self.n = len(act)
        cs = [set(ALL_COMBOS[i]) for i in act]
        self.ok = np.array([[not (a & b) for b in cs] for a in cs], dtype=bool)    # (n, n) no shared card
        self.st = st
        npl = s.np
        R = s.reach0[:, 0, :]
        self.idx = [np.nonzero(R[p] > 0)[0] for p in range(npl)]
        # per triple of the players' combos (each in its range): compatible, and each player's strength
        i0, i1, i2 = self.idx
        self.comp = self.ok[np.ix_(i0, i1)][:, :, None] & self.ok[np.ix_(i0, i2)][:, None, :] \
            & self.ok[np.ix_(i1, i2)][None, :, :]
        self.S = [st[i0][:, None, None], st[i1][None, :, None], st[i2][None, None, :]]
        self._win = {}

    def winners(self, live):
        """{tuple of tied winners: (n0, n1, n2) mask of the triples where they win}."""
        if live not in self._win:
            best = np.maximum.reduce([np.broadcast_to(self.S[p], self.comp.shape) for p in live])
            at = {p: np.broadcast_to(self.S[p], self.comp.shape) == best for p in live}
            out = {}
            for k in range(1, len(live) + 1):
                for W in itertools.combinations(live, k):
                    m = self.comp.copy()
                    for p in live:
                        m &= at[p] if p in W else ~at[p]
                    if m.any():
                        out[W] = m
            self._win[live] = out
        return self._win[live]

    def values(self, t: Terminal):
        """{winners: each player's value (np,)}: the pot split among the winners, prize equity in chip equivalents
        counted from each player's origin as the prize solve counts it."""
        pay, npl = self.pay, self.s.np
        out = {}
        for W in self.winners(t.live):
            put = pay._put0.copy()
            put[pay._ix] += t.c
            won = np.zeros(len(put))
            for w in W:
                won[pay._ix[w]] += put.sum() / len(W)
            out[W] = pay.model.value_batch(put[None], won[None])[0, pay._ix] - self.base
        return out

    def term(self, t, X, p):
        """p's value per combo of its range at terminal t, the others' reaches X (np, n) there (unnormalised)."""
        npl = self.s.np
        q, r = [j for j in range(npl) if j != p]
        xq, xr = X[q][self.idx[q]], X[r][self.idx[r]]
        u = np.zeros(len(self.idx[p]))
        for W, V in self.values(t).items():
            m = self.winners(t.live)[W]
            # sum over the other two players' combos, p's axis kept
            T = np.einsum("abc,b,c->a" if p == 0 else "abc,a,c->b" if p == 1 else "abc,a,b->c", m,
                          *((xq, xr) if p == 0 else (xq, xr) if p == 1 else (xq, xr)))
            u += V[p] * T
        out = np.zeros(self.n)
        out[self.idx[p]] = u
        return out

    def walk(self, node, X, p, br):
        if isinstance(node, Terminal):
            return self.term(node, X, p)
        sig = node.avg_strategy()[0]
        vals = []
        for a, ch in enumerate(node.children):
            Xa = X.copy()
            if node.player != p:
                Xa[node.player] = X[node.player] * sig[a]
            vals.append(self.walk(ch, Xa, p, br))
        if node.player != p:
            return np.sum(vals, axis=0)
        V = np.stack(vals)
        return V.max(axis=0) if br else (V * sig).sum(axis=0)

    def mass(self):
        i0, i1, i2 = self.idx
        R = self.s.reach0[:, 0, :]
        return float(np.einsum("abc,a,b,c->", self.comp, R[0][i0], R[1][i1], R[2][i2]))

    def report(self):
        """Each player's value of the solve's strategies in the exact game, and its best response's gain (bb a deal)."""
        R = self.s.reach0[:, 0, :]
        m = self.mass()
        v, g = [], []
        for p in range(self.s.np):
            a = float((self.walk(self.s.root, R.copy(), p, False) * R[p]).sum()) / m
            b = float((self.walk(self.s.root, R.copy(), p, True) * R[p]).sum()) / m
            v.append(a); g.append(b - a)
        return np.array(v), np.array(g)


def thin(r, st, keep):
    """Every k-th combo of the range by strength, so `keep` are left (their weights kept)."""
    on = np.nonzero(r > 0)[0]
    on = on[np.argsort(st[on], kind="stable")]
    pick = on[np.linspace(0, len(on) - 1, min(keep, len(on))).round().astype(int)]
    out = np.zeros_like(r)
    out[pick] = r[pick]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--keep", type=int, default=60)
    ap.add_argument("--board", default="Kh7d2c9s4h")
    args = ap.parse_args()
    board = cards(args.board)
    Rp = A.preflop_ranges(PRE, pf.Table(pf.SEATS[6]), 100.0)
    st = eval_board(board)
    rs = [thin(np.asarray(Rp[p]), st, args.keep) for p in SEATS]
    # the money bubble: 4 left, 3 paid; the small blind (8bb) folded, the three in the pot 25bb at the hand's start
    # with 7.5 in each (the flop and turn bet and called): 23 in the pot, 17.5 behind
    cases = [("bubble 50/30/20, SB 8bb out of the pot", {"BB": 25.0, "CO": 25.0, "BTN": 25.0, "SB": 8.0}, [50, 30, 20]),
             ("bubble, BB the short stack (12bb)", {"BB": 12.0, "CO": 25.0, "BTN": 25.0, "SB": 20.0}, [50, 30, 20]),
             ("final table 6 left, 40/25/15/10/6/4", {"BB": 25.0, "CO": 25.0, "BTN": 25.0, "SB": 15.0, "UTG": 30.0,
                                                     "HJ": 40.0}, [40, 25, 15, 10, 6, 4])]
    for name, stacks, payouts in cases:
        contrib = {s: 7.5 for s in SEATS}
        contrib["SB"] = 0.5
        behind = min(stacks[s] for s in SEATS) - 7.5
        pot = sum(contrib.values())
        pay = icm.MWPayoffs(icm.Model(stacks, payouts), SEATS, contrib)
        sols = {}
        for label, P in (("chips", None), ("prizes", pay)):
            t0 = time.process_time()
            s = MultiwaySolver(board, rs, pot, behind, cfg=MWConfig(icm=P))
            s.run(iters=args.iters)
            sols[label] = (s, time.process_time() - t0)
        ex = {k: Exact(s, pay, board, sols["prizes"][0].icm_base) for k, (s, _) in sols.items()}
        print(f"== {name}: pot {pot:.1f}, {behind:.1f} behind, {args.keep} combos a player, {args.iters} iterations",
              flush=True)
        for label, (s, cpu) in sols.items():
            v, g = ex[label].report()
            own = s.br_gains() / s.root_pot() * 100
            line = (f"  {label:6s}: in exact prizes the players' values {np.round(v / pot * 100, 2)}% of the pot, their "
                    f"best responses gain {np.round(g / pot * 100, 3)}% (mean {g.mean() / pot * 100:.3f}%); own game's "
                    f"gains {np.round(own, 3)}%; {cpu:.1f}s")
            if label == "prizes":
                line += f"; the model's values {np.round(s.values() / pot * 100, 2)}%"
            root = s.root
            freqs = s.range_freqs()
            line += "; root " + ", ".join(f"{k} {x:.2f}" for k, x in freqs.items())
            print(line, flush=True)


if __name__ == "__main__":
    main()
