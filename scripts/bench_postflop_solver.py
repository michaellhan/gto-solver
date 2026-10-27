"""Benchmark against b-inary's postflop-solver (github.com/b-inary/postflop-solver; Rust, DCFR, every card dealt).

Both solvers play the same game: our betting tree is handed to theirs line by line (scripts/pfbench, a small harness
around their crate; build it first, see its Cargo.toml), the same ranges, pot and stack (in chips of 0.01bb).

    python scripts/bench_postflop_solver.py river --boards 3 [--speeds fast,accurate]
    python scripts/bench_postflop_solver.py turn --boards 3
    python scripts/bench_postflop_solver.py flop [--target 0.15] srp_btn_bb:Kc7d2h 3bp_bb_btn:Tc2d2c ...
    python scripts/bench_postflop_solver.py summary         # the records so far, as tables

river, turn: the spots the Advisor solves in a hand. BTN vs BB, CO vs BB and a BB vs BTN 3bet pot, random flops of the
flop library with lines its strategy plays often (check-check where both checks are common, and the commonest c-bet
called), a random turn (and river, the turn checked through); hero is the player out of position. The Advisor answers
(native.advisor at that speed: its flop from the library, ranges narrowed through it, its own tree and budget) and its
solve of the street is taken from its cache. Theirs then
  - solves the same street exactly to --target (every river card where ours plays river classes on the turn),
  - solves it on one thread to the exploitability ours reached (the time their code needs for our accuracy),
  - solves it with every decision of one player on that street locked to ours, the rest (the river after a turn)
    played at equilibrium: the EV that player gives up by our strategy (ev_loss, % of the pot), once per player,
  - on the river, best-responds to our whole strategy (their_measure: our exploitability by their code),
and our root's range frequencies and per-combo mix (half the L1 distance, range-weighted, in points) are compared
with theirs.
flop: a library entry's stored nodes (the root, IP after a check, OOP facing a c-bet after checking) locked into
their full game of the library's tree (every turn and river card, where the entry played turn classes and the river
blind on the turn's boards), per player as above. Their flop game needs ~5 GB (compressed memory) and minutes a solve.

Each record is printed and appended to reports/postflop_solver/<mode>.jsonl. Times are wall seconds on this machine;
theirs use every core (rayon) unless noted, ours what the Advisor uses.
"""
import argparse, json, os, random, subprocess, sys, time
from types import SimpleNamespace
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np
from gto import flop_library as fl, native
from gto.cards import ALL_COMBOS, cards as parse_cards, cards_str
from gto.cli import parse_actions
from gto.solver import Chance, BucketEntry, Ctx, Decision, MEMBER_FULL, TreeBuilder

BIN = os.environ.get("PFBENCH", os.path.join(ROOT, "scripts", "pfbench", "target", "release", "pfbench"))
OUT = os.path.join(ROOT, "reports", "postflop_solver")
S = 100           # chips per bb (our bets are in 0.5bb steps)
HANDS = [cards_str(c) for c in ALL_COMBOS]
SP = dict((s[0], s[2]) for s in fl.SPOTS)


def chips(x):
    return int(round(x * S))


def run(spec, threads=None):
    env = dict(os.environ)
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    p = subprocess.run([BIN], input=json.dumps(spec), capture_output=True, text=True, env=env)
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout)


def act_of(node, name, child):
    """One of our actions as the harness's: {"t": kind, "a": the player's chips on the street after it}."""
    p = node.player
    if name in ("check", "call", "fold"):
        return {"t": name}
    amt = chips(child.bets[p])
    if name == "all-in":
        return {"t": "allin", "a": amt}
    return {"t": "raise" if node.bets[1 - p] > node.bets[p] + 1e-9 else "bet", "a": amt}


def act_str(a):
    return a["t"] if a["t"] in ("check", "call", "fold") else f"{a['t']} {a['a']}"


def below(node):
    """The decision under a chance node or a bucket/river entry (the same betting on every card)."""
    while isinstance(node, (Chance, BucketEntry)):
        node = node.child
    return node


def lines_of(root):
    """Every line of the tree ending in a bet, raise or all-in, each line's prefix first (their add_line)."""
    out = []

    def walk(node, line):
        node = below(node)
        if not isinstance(node, Decision):
            return
        for name, child in zip(node.names, node.children):
            a = act_of(node, name, child)
            if a["t"] in ("bet", "raise", "allin"):
                out.append(line + [a])
            walk(child, line + [a])
    walk(root, [])
    return out


def street_nodes(root):
    """(path strings, node) of every decision of the root's street (above a chance node)."""
    res = []

    def walk(node, strs):
        if not isinstance(node, Decision):
            return
        res.append((strs, node))
        for name, child in zip(node.names, node.children):
            walk(child, strs + [act_str(act_of(node, name, child))])
    walk(root, [])
    return res


def range_spec(r, board):
    rr = np.clip(r * np.prod([1.0 - MEMBER_FULL[c] for c in board], axis=0), 0, 1)
    idx = np.nonzero(rr > 0)[0]
    return [HANDS[i] for i in idx], [round(float(rr[i]), 6) for i in idx]


def lock_of(strs, node, active, strat, keep):
    """A node lock: the node's actions and `strat` (A, n compressed combos) for the combos `keep`."""
    return {"path": strs, "acts": [act_of(node, nm, ch) for nm, ch in zip(node.names, node.children)],
            "hands": [HANDS[active[j]] for j in keep],
            "strat": [[round(float(x), 5) for x in strat[a, keep]] for a in range(len(node.names))]}


def spec(board, f0, f1, pot, stack, root):
    return {"board": [cards_str([c]) for c in board], "oop": range_spec(f0, board), "ip": range_spec(f1, board),
            "pot": chips(pot), "stack": chips(stack), "lines": lines_of(root)}


def their_solve(base, target, iters=20000, threads=None, locks=(), compress=False):
    return run(dict(base, mode="solve", iters=iters, target_pct=target, locks=list(locks), compress=compress), threads)


def ev_losses(base, eq, nodes, active, keep, pot, target, compress=False):
    """Per player: their EV with that player's `nodes` locked to ours (the rest solved) below their equilibrium's."""
    loss, info = [], []
    for p in (0, 1):
        locks = [lock_of(strs, f, active, strat, keep[p]) for strs, f, strat, pl in nodes if pl == p]
        lk = their_solve(base, target, locks=locks, compress=compress)
        loss.append(round((eq["ev_cur"][p] - lk["ev_cur"][p]) / chips(pot) * 100, 4))
        info.append({"nodes": len(locks), "iters": lk["iters"], "expl": round(lk["expl_pct"], 4),
                     "secs": round(lk["secs"], 1)})
    return loss, info


def measure(base, nodes, active, keep):
    """Their best responses to our whole strategy (every node locked): (OOP's gain, IP's gain, exploitability)."""
    gains = []
    for lp in (1, 0):      # lock IP (OOP best-responds), then OOP
        locks = [lock_of(strs, f, active, strat, keep[lp]) for strs, f, strat, pl in nodes if pl == lp]
        gains.append(run(dict(base, mode="lock", lock_player=lp, locks=locks))["br_gain_pct"])
    return gains[0], gains[1], (gains[0] + gains[1]) / 2


def root_compare(root, strategy, theirs, r_root):
    """Our root strategy (A, 1326) against theirs: each action's range frequency and the range-weighted mix difference
    (half the L1 distance per combo, in points)."""
    acts, p = theirs["acts"], theirs["player"]
    ours = [act_str(act_of(root, nm, ch)) for nm, ch in zip(root.names, root.children)]
    hidx = {h: i for i, h in enumerate(HANDS)}
    idx = [hidx[h] if h in hidx else hidx[h[2:] + h[:2]] for h in theirs["root"][f"hands{p}"]]
    T = np.array(theirs["strat"])
    O = strategy[[ours.index(a) for a in acts]][:, idx]
    w = r_root[idx] / r_root[idx].sum()
    return {"acts": acts, "ours": [round(float(x) * 100, 1) for x in (O * w).sum(axis=1)],
            "theirs": [round(float(x) * 100, 1) for x in (T * w).sum(axis=1)],
            "mix_diff": round(float((np.abs(T - O).sum(axis=0) / 2 * w).sum() * 100), 2)}


def full_of(active, r):
    f = np.zeros(len(HANDS))
    f[active] = r
    return f


def write(mode, rec):
    print(json.dumps(rec), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{mode}.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def flop_lines(lib, setup, flop):
    """The flop lines the library's strategy plays often: check-check where both checks are common (>= 25%), and the
    commonest c-bet called (IP's after a check, or OOP's where OOP rarely checks). A rare line isn't answered from
    the library."""
    oop, ip, pot = setup["oop"], setup["ip"], setup["pot"]
    ls = lib.solution(setup["full0"], setup["full1"], flop, setup["full0"], setup["full1"], pot, setup["stack"],
                      fl.library_config())

    def freqs(n, r):
        return (n.avg_strategy0() * r).sum(axis=1) / r.sum()

    def commonest_bet(n, f):
        b = max((a for a, nm in enumerate(n.names) if nm.startswith("bet")), key=lambda a: f[a])
        return round(float(n.names[b].split()[1].rstrip("%")) / 100 * pot, 2)
    f0 = freqs(ls.root, ls.r0)
    n1 = ls.root.children[ls.root.names.index("check")]
    f1 = freqs(n1, ls.r1)
    lines = []
    if f0[0] >= 0.25 and f1[0] >= 0.25:
        lines.append(("x,x", [f"{oop}:x", f"{ip}:x"]))
    if f0[0] >= 0.25:
        lines.append(("x,b,c", [f"{oop}:x", f"{ip}:b{commonest_bet(n1, f1)}", f"{oop}:c"]))
    else:
        lines.append(("b,c", [f"{oop}:b{commonest_bet(ls.root, f0)}", f"{ip}:c"]))
    return lines


def bench_street(street, n_boards, speeds, target, seed):
    nb = {"turn": 4, "river": 5}[street]
    lib = fl.default_library()
    rng = random.Random(seed)
    for sid in ("srp_btn_bb", "srp_co_bb", "3bp_bb_btn"):
        setup = fl.spot_setup(SP[sid])
        oop, ip = setup["oop"], setup["ip"]
        names = sorted(next(s for s in lib.index()["spots"] if s["id"] == sid)["flops"])
        for _ in range(n_boards):
            flop = parse_cards(rng.choice(names))
            board = list(flop) + rng.sample([c for c in range(52) if c not in flop], nb - 3)
            for line, fa in flop_lines(lib, setup, flop):
                hole = next(h for h in HANDS if not set(parse_cards(h)) & set(board))
                for speed in speeds:
                    adv = native.advisor(speed)
                    st = {"players": 6, "bb": 1, "stack": 100, "board": cards_str(board), "hero": oop, "hole": hole,
                          "preflop": parse_actions(SP[sid]), "flop": parse_actions(",".join(fa)), "turn": [],
                          "river": []}
                    if street == "river":
                        st["turn"] = parse_actions(f"{oop}:x,{ip}:x")
                    t = time.time()
                    res = adv.advise(st)
                    secs = time.time() - t
                    keys = [k for k in adv._cache if len(k[2]) == nb]
                    if "error" in res or len(keys) != 1:
                        print("skipped", sid, cards_str(board), line, res.get("error"), len(keys), flush=True)
                        continue
                    k = keys[0]
                    sol = adv._cache[k]
                    pot, stack, bets, to_act = k[3:7]
                    if bets != (0.0, 0.0) or to_act != 0:
                        continue
                    active = sol.ctx.active
                    f0, f1 = full_of(active, sol.r0), full_of(active, sol.r1)
                    rec = {"spot": sid, "flop_line": line, "board": cards_str(board), "speed": speed,
                           "library_flop": not any(len(k2[2]) == 3 for k2 in adv._cache),
                           "answer_secs": round(secs, 2), "ours_iters": sol.iters, "ours_secs": round(sol.seconds, 2),
                           "ours_expl": round(sol.exploitability(), 4), "pot": pot, "stack": stack,
                           "combos": [int((sol.r0 > 0).sum()), int((sol.r1 > 0).sum())],
                           "sizes": sol.ctx.cfg.bet_sizes.get(street)}
                    base = spec(board, f0, f1, pot, stack, sol.root)
                    eq = their_solve(base, target)
                    rec.update(their_iters=eq["iters"], their_secs=round(eq["secs"], 2),
                               their_expl=round(eq["expl_pct"], 4), their_mem_gb=round(eq["mem_gb"], 3))
                    one = their_solve(base, max(rec["ours_expl"], target), threads=1)
                    rec.update(their_1t_secs_to_ours=round(one["secs"], 2), their_1t_iters_to_ours=one["iters"])
                    keep = [np.nonzero(sol.r0 > 0)[0], np.nonzero(sol.r1 > 0)[0]]
                    nodes = [(strs, n, n.avg_strategy()[0], n.player) for strs, n in street_nodes(sol.root)]
                    rec["ev_loss"], _ = ev_losses(base, eq, nodes, active, keep, pot, target)
                    rec["root"] = root_compare(sol.root, sol.strategy, eq, f0)
                    if street == "river":
                        g0, g1, ex = measure(base, nodes, active, keep)
                        rec.update(their_measure=round(ex, 4), br_gain=[round(g0, 3), round(g1, 3)])
                    write(street, rec)


def stored_nodes(ls, full_root):
    """(path strings, the full tree's node, the library's strategy, player) of every stored decision of an entry."""
    res = []

    def walk(n, f, strs):
        n, f = below(n), below(f)
        if not isinstance(n, Decision) or not isinstance(f, Decision):
            return
        assert list(n.names) == list(f.names), (n.names, f.names)
        if id(n) in ls.stored:
            res.append((strs, f, n.avg_strategy()[0], n.player))
        for nm, c, fc in zip(f.names, n.children, f.children):
            walk(c, fc, strs + [act_str(act_of(f, nm, fc))])
    walk(ls.root, full_root, [])
    return res


def bench_flop(jobs, target):
    lib = fl.default_library()
    for job in jobs:
        sid, flop_s = job.split(":")
        setup = fl.spot_setup(SP[sid])
        f0, f1, pot, stack = setup["full0"], setup["full1"], setup["pot"], setup["stack"]
        board = parse_cards(flop_s)
        cfg = fl.library_config()
        ls = lib.solution(f0, f1, board, f0, f1, pot, stack, cfg)
        if ls is None:
            print("not in the library:", job, flush=True)
            continue
        active = ls.ctx.active
        ctx = Ctx(active, cfg)
        ctx.root_len = 3
        full = TreeBuilder(ctx, dead=list(board)).build(tuple(board), pot, stack)
        base = spec(board, f0, f1, pot, stack, full)
        rec = {"spot": sid, "flop": flop_s, "lines": len(base["lines"]), "ours_expl": round(ls.exploitability(), 4),
               "ours_iters": ls.info().get("iterations")}
        t = time.time()
        eq = their_solve(base, target, 2000, compress=True)
        rec.update(their_iters=eq["iters"], their_secs=round(eq["secs"], 1), their_expl=round(eq["expl_pct"], 4),
                   their_mem_gb=round(eq["mem_gb"], 2))
        keep = [np.nonzero(ls.r0 > 0)[0], np.nonzero(ls.r1 > 0)[0]]
        nodes = stored_nodes(ls, full)
        rec["ev_loss"], rec["locked"] = ev_losses(base, eq, nodes, active, keep, pot, target, compress=True)
        for i, p in enumerate((0, 1)):
            rec["locked"][i]["paths"] = [" ".join(s) or "root" for s, _, _, pl in nodes if pl == p]
        strat = np.zeros((len(full.names), len(HANDS)))
        strat[:, active] = ls.root.avg_strategy()[0]
        rec["root"] = root_compare(full, strat, eq, f0)
        rec["wall"] = round(time.time() - t, 1)
        write("flop", rec)


def summary():
    def rng(xs, f="{:.3f}"):
        xs = sorted(xs)
        return f"{f.format(float(np.median(xs)))} ({f.format(xs[0])}-{f.format(xs[-1])})" if xs else ""
    for mode in ("river", "turn"):
        path = os.path.join(OUT, f"{mode}.jsonl")
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path)]
        print(f"\n{mode}: {len(recs)} records\n")
        print("| speed | n | ours: iterations, s | ours: % of the pot from equilibrium | their measure of it |"
              " EV given up, OOP / IP | root frequencies, max diff | mix diff | theirs: s to target (all cores)"
              " | theirs: s to ours (1 core) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for speed in sorted({r["speed"] for r in recs}):
            rs = [r for r in recs if r["speed"] == speed]
            meas = [abs(r["their_measure"] - r["ours_expl"]) for r in rs if "their_measure" in r]
            print(f"| {speed} | {len(rs)} | {rng([r['ours_iters'] for r in rs], '{:.0f}')}, "
                  f"{rng([r['ours_secs'] for r in rs], '{:.1f}')} | {rng([r['ours_expl'] for r in rs])} | "
                  + (f"within {max(meas):.4f}" if meas else "") + " | "
                  f"{rng([r['ev_loss'][0] for r in rs])} / {rng([r['ev_loss'][1] for r in rs])} | "
                  f"{rng([max(abs(a - b) for a, b in zip(r['root']['ours'], r['root']['theirs'])) for r in rs], '{:.1f}')} | "
                  f"{rng([r['root']['mix_diff'] for r in rs], '{:.1f}')} | {rng([r['their_secs'] for r in rs], '{:.2f}')} | "
                  f"{rng([r['their_1t_secs_to_ours'] for r in rs], '{:.2f}')} |")
    path = os.path.join(OUT, "flop.jsonl")
    if os.path.exists(path):
        print("\nflop\n")
        print("| spot | flop | library entry, % of the pot (its game) | EV given up, OOP / IP | root: ours / theirs |"
              " mix diff | their solves: iterations, % of the pot, s |")
        print("|---|---|---|---|---|---|---|")
        for l in open(path):
            r = json.loads(l)
            fr = ", ".join(f"{a} {o:.0f}/{t:.0f}" for a, o, t in zip(r["root"]["acts"], r["root"]["ours"], r["root"]["theirs"]))
            print(f"| {r['spot']} | {r['flop']} | {r['ours_expl']:.3f} | {r['ev_loss'][0]:.3f} / {r['ev_loss'][1]:.3f} | "
                  f"{fr} | {r['root']['mix_diff']:.1f} | {r['their_iters']}, {r['their_expl']:.3f}, {r['their_secs']:.0f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("river", "turn", "flop", "summary"))
    ap.add_argument("jobs", nargs="*", help="flop: spot:flop pairs (a flop the spot's library entry holds)")
    ap.add_argument("--boards", type=int, default=3, help="river/turn: random flops per spot")
    ap.add_argument("--speeds", default="fast,accurate")
    ap.add_argument("--target", type=float, default=None,
                    help="their solves' exploitability, %% of the pot (river 0.005, turn 0.05, flop 0.15)")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_intermixed_args()
    if a.mode == "summary":
        sys.exit(summary())
    if not os.path.exists(BIN):
        sys.exit(f"no harness at {BIN}: build scripts/pfbench (its Cargo.toml) or set PFBENCH")
    target = a.target if a.target is not None else {"river": 0.005, "turn": 0.05, "flop": 0.15}[a.mode]
    if a.mode == "flop":
        bench_flop(a.jobs, target)
    else:
        bench_street(a.mode, a.boards, a.speeds.split(","), target, a.seed)
