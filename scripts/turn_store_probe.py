"""Precomputed turns: one library entry solved as the build does (the river dealt, the core), then its turn
decisions: each flop line that reaches the turn (its reach), each turn node a store would hold (the first decision,
in position after a check, each facing a bet), per turn class, the node player's reached combos. Prints bytes for
store variants; dumps the turn roots of the top lines (ranges, the stored strategies per class) for
scripts/bench_turn_store.py.

  python scripts/turn_store_probe.py srp_btn_bb Kc7d2h 300 OUT.pkl [later_turn=0.33,0.75,1.25]
"""
import json, os, sys, time, zlib, pickle, resource
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gto.native  # noqa: F401
import numpy as np
from gto import flop_library as fl
from gto.cards import cards as parse_cards, cards_str
from gto.solver import solve, Decision, Chance, Terminal

sid, flop, iters, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
s = fl.spot_setup(dict((x, l) for x, _, l in fl.SPOTS)[sid])
cfg = fl.library_config(s["pot"], s["stack"])
for m in sys.argv[5:]:                  # a richer menu inside: later_turn=0.33,0.75,1.25 (later_raise_turn=, later_river=)
    k, v = m.split("=")
    kind, st = ("raise", k[12:]) if k.startswith("later_raise_") else ("bet", k[6:])
    cfg.later_sizes[kind][st] = [float(x) for x in v.split(",")]
t0 = time.time()
sol = solve(parse_cards(flop), s["full0"], s["full1"], s["pot"], s["stack"], cfg=cfg, time_budget=1e9, max_iters=iters)
print(f"solved {sol.iters} it, {sol.exploitability():.3f}% of the pot, {time.time() - t0:.0f} s, "
      f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB", flush=True)
ctx = sol.ctx
R0, R1 = sol.r0.astype(np.float64), sol.r1.astype(np.float64)
pair_tot = float((sol.solver.compat_sum(R1[None, :].astype(np.float32))[0] * R0).sum())

lines = []      # (path, r0, r1, chance)
def walk(n, path, r0, r1):
    st = n.avg_strategy0()
    for k, (nm, c) in enumerate(zip(n.names, n.children)):
        a0, a1 = (r0 * st[k], r1) if n.player == 0 else (r0, r1 * st[k])
        if isinstance(c, Decision):
            walk(c, path + (nm,), a0, a1)
        elif isinstance(c, Chance) and isinstance(c.child, Decision):
            lines.append((path + (nm,), a0, a1, c))
walk(sol.root, (), R0, R1)

def pairs(a0, a1):
    return float((sol.solver.compat_sum(a1[None, :].astype(np.float32))[0] * a0).sum()) / pair_tot

classes = ctx.turn_classes(sol.root.boards[0], sol.solver.dead)
K = len(classes)
rows = []
blob = {"q100": [], "q20": [], "q100_first": [], "q100_first_ip": []}
dump = []
for path, a0, a1, ch in sorted(lines, key=lambda x: -pairs(x[1], x[2])):
    T = ch.child
    share = pairs(a0, a1)
    nodes = [((), T)]
    for nm, c in zip(T.names, T.children):
        if isinstance(c, Decision):
            nodes.append(((nm,), c))
            if nm == "check":
                nodes += [((nm, n2), c2) for n2, c2 in zip(c.names, c.children) if isinstance(c2, Decision) and n2 != "check"]
    for seq, node in nodes:
        tag = " / ".join(seq) or "first"
        st = node.avg_strategy()                      # (K, A, n)
        p = node.player
        # the node player's reach per class: their turn-root reach x the class's live share x their turn actions on the way
        base = (a0 if p == 0 else a1)[None, :] * ch.masks.reshape(K, -1)
        if seq:
            m = T
            for nm in seq:
                k = m.names.index(nm)
                if m.player == p:
                    base = base * m.avg_strategy()[:, k, :]
                m = m.children[k]
        A = len(node.names)
        reached = base > 1e-6 * base.max()
        nr = int(reached.sum())
        for b in range(K):
            idx = np.nonzero(reached[b])[0]
            q = np.round(st[b][:A - 1, idx] * 100).astype(np.uint8)
            q20 = np.round(st[b][:A - 1, idx] * 20).astype(np.uint8)
            blob["q100"].append(q.tobytes()); blob["q20"].append(q20.tobytes())
            if tag == "first":
                blob["q100_first"].append(q.tobytes())
            if seq in ((), ("check",)):
                blob["q100_first_ip"].append(q.tobytes())
        rows.append(dict(line=" ".join(path), share=round(share, 4), node=tag, player=p, names=list(node.names), A=A,
                         reached_per_class=round(nr / K)))
    if len(dump) < 4 and share > 0.03:
        dump.append(dict(path=list(path), share=share, r0=a0, r1=a1, masks=ch.masks.reshape(K, -1),
                         first_names=list(T.names), first=T.avg_strategy(),
                         ip_names=list(T.children[T.names.index("check")].names),
                         ip=T.children[T.names.index("check")].avg_strategy(), pot=float(T.pot_now)))
for r in rows:
    print(json.dumps(r))
print("lines reaching the turn:", len(lines), "classes:", K, "covered share of pairs:",
      round(sum(pairs(a0, a1) for _, a0, a1, _ in lines), 3))
for k, v in blob.items():
    raw = b"".join(v)
    print(f"{k}: raw {len(raw) / 1024:.0f} KB, zlib {len(zlib.compress(raw, 9)) / 1024:.0f} KB")
with open(out, "wb") as f:
    pickle.dump(dict(spot=sid, flop=flop, active=ctx.active, classes=classes, setup={k: s[k] for k in ("pot", "stack")},
                     iters=sol.iters, dump=dump), f)
print(f"done {time.time() - t0:.0f} s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
