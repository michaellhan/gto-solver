"""Precomputed turns (flop_library.encode_turns): what a stored turn answer (the flop solve's turn node, dumped by
scripts/turn_store_probe.py) gives up against a live Accurate turn solve of the same spot (native.config("accurate"):
33/75/125% turn and river, river classes).
B = the live tree solved; then per player q, B solved again with q's stored node locked (OOP's first decision, or
IP's after a check): q's loss, % of the turn pot. The stored strategy is the class's played card's. On another card
of the class (--cards other / spade: another suit making a 2-flush with the flop, the suit the flop lacks) a map says
which stored combo a live combo reads (--maps): direct = the same combo (what the flop solve itself plays for every
card of a class; a combo holding the played card, which the flop solve doesn't play there, reads its swap image),
hybrid = the combo with the two cards' suits swapped, or the same combo where that one is dead on the
played card's board; a _q20 suffix rounds the stored strategy to 5% steps first. Numpy (GTO_CORE=0), so nodes lock.
One JSON line per (line, turn card) to OUT (resumable).

  python scripts/bench_turn_store.py DUMP.pkl OUT.jsonl --lines 2 --ranks A,7,4 --iters 400
  python scripts/bench_turn_store.py DUMP.pkl OUT.jsonl --cards other,spade --maps direct,hybrid,direct_q20
"""
import argparse, json, os, pickle, sys, time
import bench_menu_loss as M          # (sets GTO_CORE=0, locks Decision.strategy through M.LOCK)
import numpy as np
from gto import flop_library as fl, native
from gto.cards import ALL_COMBOS, cards_str, cards as parse_cards
from gto.solver import solve, MEMBER_FULL

ap = argparse.ArgumentParser()
ap.add_argument("dump"); ap.add_argument("out")
ap.add_argument("--lines", type=int, default=2)
ap.add_argument("--ranks", default="A,7,4")
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--swap", action="store_true", help="also another card of each class, mapped by suit swap")
ap.add_argument("--cards", default="", help="instead: other,spade (cards of the class besides the played one)")
ap.add_argument("--maps", default="swap", help="with --cards: direct,hybrid,direct_q20 ...")
ap.add_argument("--line-cards", default="", help="per line, the --cards for it: 0:other,spade;1:spade")
a = ap.parse_args()
D = pickle.load(open(a.dump, "rb"))
active = np.asarray(D["active"])
flop = parse_cards(D["flop"])
board3 = tuple(sorted(flop, reverse=True))
done = set()
if os.path.exists(a.out):
    done = {(r["line"], r["turn"]) for r in map(json.loads, open(a.out))}
RANKS = "23456789TJQKA"

def full(v):
    f = np.zeros(len(ALL_COMBOS)); f[active] = v; return f

def node_lock(sol, node, names, strat_full):
    """LOCK for `node` of the live solve: the stored strategy (names, (A, 1326)) on its actions, 0 elsewhere."""
    act = sol.ctx.active
    L = np.zeros((1, len(node.names), len(act)), dtype=np.float32)
    for i, nm in enumerate(node.names):
        if nm in names:
            L[0, i] = strat_full[names.index(nm)][act]
    tot = L.sum(axis=1, keepdims=True)
    L = np.where(tot > 1e-9, L / np.maximum(tot, 1e-12), np.float32(1.0 / len(node.names)) * (np.arange(len(node.names)) == 0)[None, :, None])
    return L

cfg = native.config("accurate")
flop_suits = {x % 4 for x in flop}
per_line = dict(x.split(":") for x in a.line_cards.split(";")) if a.line_cards else {}
for li, d in enumerate(D["dump"][:a.lines]):
    for rk in a.ranks.split(","):
        r = RANKS.index(rk)
        k = next(i for i, (pc, cs) in enumerate(D["classes"]) if pc // 4 == r)
        played, cs = D["classes"][k]
        want = per_line.get(str(li), a.cards)
        if want:
            cards = []
            for w in want.split(","):
                cand = [c for c in cs if c != played and c not in cards and ((c % 4 in flop_suits) == (w == "other"))]
                cards += cand[:1]
        else:
            cards = [played] + ([c for c in cs if c != played][:1] if a.swap else [])
        for c in cards:
            line, turn = " ".join(d["path"]), cards_str([c])
            if (line, turn) in done:
                continue
            perm = list(range(4)); perm[c % 4], perm[played % 4] = played % 4, c % 4
            cmap = fl.combo_map(perm)                     # combo on this card's board -> its image on the played card's
            first0 = np.zeros((len(d["first_names"]), len(ALL_COMBOS))); ip0 = np.zeros((len(d["ip_names"]), len(ALL_COMBOS)))
            first0[:, active] = d["first"][k]; ip0[:, active] = d["ip"][k]
            have = np.zeros(len(ALL_COMBOS), bool); have[active] = True
            have &= (1.0 - MEMBER_FULL[played]) > 0       # a combo the played card's board deals
            maps = {}
            for m in (a.maps.split(",") if a.cards or a.line_cards else ["swap"]):
                base, q20 = m.removesuffix("_q20"), m.endswith("_q20")
                if c == played:
                    idx = np.arange(len(ALL_COMBOS))
                elif base == "direct":                     # (a combo holding the played card has no strategy there:
                    idx = np.where(have, np.arange(len(ALL_COMBOS)), cmap)     # the flop solve gives it no reach)
                elif base == "swap":
                    idx = cmap
                else:                                      # hybrid
                    idx = np.where(have[cmap], cmap, np.arange(len(ALL_COMBOS)))
                f, i_ = first0[:, idx], ip0[:, idx]
                if q20:
                    f, i_ = [np.round(x / np.maximum(x.sum(0, keepdims=True), 1e-12) * 20) / 20 for x in (f, i_)]
                maps[m] = (f, i_)
            first, ip = next(iter(maps.values()))
            live = 1.0 - MEMBER_FULL[c]
            r0, r1 = full(d["r0"]) * live, full(d["r1"]) * live
            pot = d["pot"]; stack = D["setup"]["stack"] - (pot - D["setup"]["pot"]) / 2
            board = board3 + (c,)
            rec = dict(spot=D["spot"], flop=D["flop"], line=line, share=round(d["share"], 4), turn=turn,
                       played=cards_str([played]), iters=a.iters)
            M.LOCK.clear(); t0 = time.time()
            sb = solve(board, r0, r1, pot, stack, cfg=cfg, time_budget=1e9, max_iters=a.iters)
            eb = [M.ev_pct(sb, p) for p in (0, 1)]
            root = sb.solver.root
            w = sb.r0
            rec.update(b_expl=round(sb.exploitability(), 3), b_secs=round(time.time() - t0), b_names=list(root.names),
                       b_root=[round(float((root.avg_strategy0()[i] * w).sum() / w.sum()) * 100, 1) for i in range(len(root.names))],
                       stored_root=[round(float((first[i][sb.ctx.active] * w).sum() / w.sum()) * 100, 1) for i in range(len(d["first_names"]))],
                       stored_names=d["first_names"])
            del sb
            rec["loss"] = []
            jobs = [(m, q, names, st) for m, (f, i_) in maps.items() for q, (names, st) in enumerate(((d["first_names"], f), (d["ip_names"], i_)))]
            for m, q, names, st in jobs:
                M.LOCK.clear()
                sl = solve(board, r0, r1, pot, stack, cfg=cfg, time_budget=1e9, max_iters=1)
                for n in M.decisions(sl.solver.root, []):
                    n.regret[...] = 0; n.ssum[...] = 0
                node = sl.solver.root if q == 0 else sl.solver.root.children[sl.solver.root.names.index("check")]
                M.LOCK[id(node)] = node_lock(sl, node, names, st)
                sl.extend(1e9, max_iters=a.iters)
                if len(maps) > 1:
                    rec.setdefault("loss_by_map", {}).setdefault(m, []).append(round(eb[q] - M.ev_pct(sl, q), 3))
                else:
                    rec["loss"].append(round(eb[q] - M.ev_pct(sl, q), 3))
                del sl
            rec["secs"] = round(time.time() - t0)
            rec["maxrss_gb"] = round(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 2**30, 2)
            print(json.dumps(rec), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
