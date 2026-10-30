"""What the flop library's flop menu (bets 33/75%, raise 50%; turn and river 75% inside) costs the flop
strategies it stores, in our own abstraction (the library's model, numpy, GTO_CORE=0, so a node can be locked).

  A = the library's menu solved; B = a richer menu (--menu). B solved to equilibrium; then, per player q, B solved again
  with q's flop decisions locked to A's (B's extra actions at 0, nodes A doesn't have left free): q's EV loss, in % of
  the pot, is what q's flop strategy gives up against an opponent and a later game that use the richer menu.
  One JSON line a flop to --out; a flop already there is skipped (resumable). Waits while memory is under 30% free;
  stops if the process passes --max-gb.

  python scripts/bench_menu_loss.py srp_btn_bb Kc7d2h JcTd8d --menu flop=0.33,0.75,1.25 --iters 600 \
      --out reports/menu_loss.jsonl
"""
import argparse, json, os, resource, subprocess, sys, time
os.environ["GTO_CORE"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gto.native  # noqa: F401
import numpy as np
from gto import flop_library as fl
from gto import solver as S
from gto.cards import cards as parse_cards
import bench_postflop_solver as B

LOCK = {}
_strategy, _avg = S.Decision.strategy, S.Decision.avg_strategy
MAX_GB = 1.0


def _locked(orig):
    def f(self, sel=None):
        L = LOCK.get(id(self))
        if L is None:
            return orig(self, sel)
        return (L if sel is None else L[sel]).copy()
    return f


S.Decision.strategy, S.Decision.avg_strategy = _locked(_strategy), _locked(_avg)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def wait_memory():
    while True:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        free = int(out.rsplit(" ", 1)[-1].rstrip("%"))
        if free >= 30:
            return
        print(f"  memory {free}% free, waiting", flush=True)
        time.sleep(120)


def check_rss():
    if rss_gb() > MAX_GB:
        sys.exit(f"stopped: {rss_gb():.2f} GB over {MAX_GB} GB")


def decisions(node, out):
    if isinstance(node, S.Decision):
        out.append(node)
        for c in node.children:
            decisions(c, out)
    elif isinstance(node, (S.Chance, S.BucketEntry)):
        decisions(node.child, out)
    return out


def acts(node):
    return [B.act_str(B.act_of(node, nm, ch)) for nm, ch in zip(node.names, node.children)]


def ev_pct(sol, p):
    """p's EV at the root, both on their average strategies, % of the pot (exploitability's normalization)."""
    s = sol.solver
    a, b = sol.r0[None, :].astype(np.float32), sol.r1[None, :].astype(np.float32)
    s.sampling = False
    try:
        u = s._cfr(s.root, p, a, b, update=False, use_avg=True)
    finally:
        s.sampling = True
    pairs = float((s.compat_sum(b)[0] * sol.r0).sum())
    return float((u[0] * (sol.r0 if p == 0 else sol.r1)).sum()) / pairs / s.root.pot_now * 100


def run(board, s, cfg, iters, lock_from=None, q=None):
    wait_memory()
    LOCK.clear()
    t0 = time.time()
    if lock_from is None:
        sol = S.solve(board, s["full0"], s["full1"], s["pot"], s["stack"], cfg=cfg, time_budget=1e9, max_iters=iters)
        check_rss()
        return sol, 0, time.time() - t0
    sol = S.solve(board, s["full0"], s["full1"], s["pot"], s["stack"], cfg=cfg, time_budget=1e9, max_iters=1)
    for n in decisions(sol.solver.root, []):
        n.regret[...] = 0
        n.ssum[...] = 0
        if n.pred is not None:
            n.pred[...] = 0
    k = 0
    for strs, n in B.street_nodes(sol.solver.root):
        key = " ".join(strs)
        if n.player != q or key not in lock_from:
            continue
        a_acts, a_strat = lock_from[key]
        L = np.zeros((1, len(n.names), a_strat.shape[1]), dtype=np.float32)
        for i, x in enumerate(acts(n)):
            if x in a_acts:
                L[0, i] = a_strat[a_acts.index(x)]
        L /= np.maximum(L.sum(axis=1, keepdims=True), 1e-12)
        LOCK[id(n)] = L
        k += 1
    sol.extend(1e9, max_iters=iters)
    check_rss()
    return sol, k, time.time() - t0


def main():
    global MAX_GB
    ap = argparse.ArgumentParser()
    ap.add_argument("spot")
    ap.add_argument("flops", nargs="+")
    ap.add_argument("--menu", action="append", default=[], help="street=sizes (flop=0.33,0.75,1.25; raise_flop=0.5,1; later_turn=0.33,0.75,1.25; later_raise_river=0.5,1)")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gb", type=float, default=1.0)
    a = ap.parse_args()
    MAX_GB = a.max_gb
    menu = " ".join(a.menu)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["spot"], r["flop"], r["menu"], r["iters"]))
    line = dict((sid, l) for sid, _, l in fl.SPOTS)[a.spot]
    s = fl.spot_setup(line)
    for flop in a.flops:
        if (a.spot, flop, menu, a.iters) in done:
            continue
        board = parse_cards(flop)
        cfg_a, cfg_b = fl.library_config(), fl.library_config()
        for m in a.menu:
            k, v = m.split("=")
            sizes = [float(x) for x in v.split(",")]
            if k.startswith("later_"):           # the turn and river inside the flop solve: later_turn=, later_raise_river=
                kind, st = ("raise", k[12:]) if k.startswith("later_raise_") else ("bet", k[6:])
                cfg_b.later_sizes[kind][st] = sizes
            elif k.startswith("raise_"):
                cfg_b.raise_sizes[k[6:]] = sizes
            else:
                cfg_b.bet_sizes[k] = sizes
        sa, _, ta = run(board, s, cfg_a, a.iters)
        lock_from = {" ".join(strs): (acts(n), n.avg_strategy()[0]) for strs, n in B.street_nodes(sa.solver.root)}
        rec = dict(spot=a.spot, flop=flop, menu=menu, iters=a.iters, a_expl=round(sa.exploitability(), 4),
                   a_secs=round(ta), a_nodes=len(decisions(sa.solver.root, [])), a_ev=[round(ev_pct(sa, p), 4) for p in (0, 1)])
        del sa
        sb, _, tb = run(board, s, cfg_b, a.iters)
        eb = [ev_pct(sb, p) for p in (0, 1)]
        root = sb.solver.root
        w = (sb.r0 if root.player == 0 else sb.r1)
        rec.update(b_expl=round(sb.exploitability(), 4), b_secs=round(tb), b_nodes=len(decisions(root, [])),
                   b_ev=[round(x, 4) for x in eb],
                   b_root={k: round(float(v) * 100, 1) for k, v in zip(acts(root), (root.avg_strategy()[0] * w[None, :]).sum(axis=1) / w.sum())})
        del sb
        rec["locked"], rec["loss"] = [], []
        for q in (0, 1):
            sl, k, tl = run(board, s, cfg_b, a.iters, lock_from, q)
            e = ev_pct(sl, q)
            rec["locked"].append(dict(nodes=k, secs=round(tl), ev=round(e, 4)))
            rec["loss"].append(round(eb[q] - e, 4))
            del sl
        rec["maxrss_gb"] = round(rss_gb(), 2)
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
