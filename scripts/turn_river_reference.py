"""Turn and river strategy against reference solves (tests/test_turn_river.py, tests/data/turn_river_refs.json).

The spots are BTN vs BB single-raised pots and a CO vs BTN 3bet pot after a flop (and for the river spots a turn)
line, their ranges narrowed by long solves of those streets (stored in tests/data/turn_river_spots.npz, so the tests
don't re-solve the flop):

    python scripts/turn_river_reference.py --ranges              # flop (and turn) solves -> the spots' ranges
    python scripts/turn_river_reference.py --reference           # rich trees to convergence -> the references
    python scripts/turn_river_reference.py --measure [--configs browser-fast,native-fast] [--spots t-k72-3]

--reference solves each spot three ways, every river card in the tree: the rich menu (turn and river bets of a
third, three quarters and 1.25 pot, raises of three quarters, the all-in; on a river root also half, double and
2.5 pot), the one-size tree the browser and native Fast played, and native Accurate's old two sizes. --measure solves
each spot with the browser's and the native configs (before: "browser-fast" ...; "-now": the current ones; and
candidates) at the iterations their budgets get and prints the differences from the reference of the same tree
(convergence and river sampling) and from the rich one (the size menu too):
aggregate frequencies (points of the range), the root players' EVs (% of the pot, per combo, range-weighted) and
the exploitability in its own tree. Budgets become iterations by each config's cost relative to the one-size
every-river tree on t-k72-3 measured in the same run (so a loaded machine slows both alike), and that tree's known
idle speed: 16.8 iterations a second natively (gto/native.py: Fast's 40-45 turn iterations in 2.5 s) and 11.6 in
Pyodide (README, Solver speed), each run with its kernels (BLAS natively, NO_BLAS for the browser). On a busy machine
the cost ratios still swing by +-30%; the README's table measured each config's cost on each spot twice and took the
faster. Run it with `nice`: a reference takes minutes per turn spot.
"""
import os, sys, time, json, argparse
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np
from gto import preflop as pf
from gto import solver as S
from gto.advisor import preflop_ranges
from gto.cards import cards
from gto.solver import Config, solve, Decision, Chance, N_FULL

DATA = os.path.join(ROOT, "tests", "data")
RANGES_FILE = os.path.join(DATA, "turn_river_spots.npz")
REFS_FILE = os.path.join(DATA, "turn_river_refs.json")
VALUES_FILE = os.path.join(DATA, "turn_river_ref_values.npz")   # each reference's per-combo root values (2, 1326)

SRP = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "fold"},
       {"pos": "BTN", "action": "raise", "to": 2.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
THREEBET = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
            {"pos": "BTN", "action": "raise", "to": 7.5}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "fold"},
            {"pos": "CO", "action": "call"}]
PRE = {"srp": (SRP, "BB", "BTN", 5.5, 97.5), "3bet": (THREEBET, "CO", "BTN", 16.5, 92.5)}
BC, XX = ["check", "bet 33%", "call"], ["check", "check"]
TURN_BC, TURN_XX = ["check", "bet 75%", "call"], ["check", "check"]
# name: (pot type, flop, flop line, turn, turn line or None, river or None, what it is)
SPOTS = {
    "t-k72-3": ("srp", "Ks7d2c", BC, "3h", None, None, "c-bet called, brick turn"),
    "t-k72s-4s": ("srp", "Ks7s2d", BC, "4s", None, None, "c-bet called, the flush completes"),
    "t-k72-7": ("srp", "Ks7d2c", BC, "7h", None, None, "c-bet called, the turn pairs the board"),
    "t-q83-a": ("srp", "Qs8d3c", BC, "Ah", None, None, "c-bet called, an overcard"),
    "t-q83x-5": ("srp", "Qs8d3c", XX, "5h", None, None, "flop checked through: the BB probes, the BTN delays"),
    "t-3b-k72-4": ("3bet", "Kd7c2h", BC, "4s", None, None, "3bet pot, c-bet called, brick turn"),
    "t-3b-k72-a": ("3bet", "Kd7c2h", BC, "Ad", None, None, "3bet pot, c-bet called, an overcard"),
    "t-3b-t94-2": ("3bet", "Th9h4c", BC, "2s", None, None, "3bet pot, two-tone middling flop, brick turn"),
    "t-3b-a85-8": ("3bet", "As8d5c", BC, "8h", None, None, "3bet pot, c-bet called, the turn pairs the board"),
    "r-k72-3-5": ("srp", "Ks7d2c", BC, "3h", TURN_BC, "5c", "bet, bet, brick river: polar bettor, capped catcher"),
    "r-k72-3x-9": ("srp", "Ks7d2c", BC, "3h", TURN_XX, "9d", "bet, check-check: condensed ranges"),
    "r-k72s-4-9s": ("srp", "Ks7s2d", BC, "4h", TURN_BC, "9s", "bet, bet, the flush completes on the river: blockers"),
    "r-q83x-5-j": ("srp", "Qs8d3c", XX, "5h", TURN_XX, "Jc", "checked to the river"),
}
FLOP_CFG = dict(bet_sizes={"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]},
                raise_sizes={"flop": [0.5], "turn": [0.75], "river": [0.75]})
ONE = dict(bet_sizes={"flop": [0.75], "turn": [0.75], "river": [0.75]},
           raise_sizes={"flop": [0.75], "turn": [0.75], "river": [0.75]})
TWO = dict(bet_sizes={"flop": [0.75], "turn": [0.5, 1.0], "river": [0.5, 1.0]},
           raise_sizes={"flop": [0.75], "turn": [0.75], "river": [0.75]})
RICH = dict(bet_sizes={"flop": [0.75], "turn": [0.33, 0.75, 1.25], "river": [0.33, 0.75, 1.25]},
            raise_sizes={"flop": [0.75], "turn": [0.75], "river": [0.75]})
RICH_RIVER = dict(bet_sizes={"flop": [0.75], "turn": [0.75], "river": [0.33, 0.5, 0.75, 1.25, 2.0, 2.5]},
                  raise_sizes={"flop": [0.75], "turn": [0.75], "river": [0.75]})
# the rich menu with a second raise size (about a 2.5x raise: 40% of the pot after the call) beside 75%
RICH_RAISE = dict(RICH, raise_sizes={"flop": [0.75], "turn": [0.4, 0.75], "river": [0.4, 0.75]})
RICH_RIVER_RAISE = dict(RICH_RIVER, raise_sizes={"flop": [0.75], "turn": [0.75], "river": [0.4, 0.75]})
# the trees each reference is solved in (every river card)
REF_TREES = {"rich": RICH, "one": ONE, "two": TWO, "richr": RICH_RAISE}
def _menu(turn, river, raise_=(0.75,)):
    return dict(bet_sizes={"flop": [0.75], "turn": list(turn), "river": list(river)},
                raise_sizes={"flop": [0.75], "turn": list(raise_), "river": list(raise_)})


# the river root's menu (Advisor.root_sizes): within 0.15% of the pot of RICH_RIVER's EVs on the river spots, where one
# size (75%) was up to 3.6% off and two (50%, 100%) up to 2.2%
ROOT_RIVER = [0.33, 0.75, 1.25, 2.5]
# configs measured: (tree kwargs, the reference tree of the same menu, budget in seconds, "native" or "browser");
# "_root_river": the river root's bet sizes when they differ from the tree's river (Advisor.root_sizes)
CONFIGS = {
    # before (static/worker.js, gto/native.py)
    "browser-fast": (dict(ONE, n_rivers=8), "one", 8.0, "browser"),
    "browser-accurate": (dict(ONE), "one", 60.0, "browser"),
    "native-fast": (dict(ONE), "one", 2.5, "native"),
    "native-accurate": (dict(TWO), "two", 30.0, "native"),
    # now (gto/native.py, static/worker.js)
    "browser-fast-now": (dict(_menu([0.33, 1.25], [0.75]), river_classes=True, _root_river=ROOT_RIVER), "rich", 8.0, "browser"),
    "browser-accurate-now": (dict(RICH, river_classes=True, _root_river=ROOT_RIVER), "rich", 60.0, "browser"),
    # (native Fast's turn gets 5 s where the flop costs nothing, 2.5 s when the request solves the flop too: -cold)
    "native-fast-now": (dict(_menu([0.33, 1.25], [0.75]), river_classes=True, _root_river=ROOT_RIVER), "rich", 5.0, "native"),
    "native-accurate-now": (dict(RICH, river_classes=True, _root_river=ROOT_RIVER), "rich", 30.0, "native"),
    # candidates: river classes, other menus, the river root's menu
    "browser-fast-classes": (dict(ONE, river_classes=True, _root_river=ROOT_RIVER), "one", 8.0, "browser"),
    "browser-accurate-classes": (dict(ONE, river_classes=True, _root_river=ROOT_RIVER), "one", 60.0, "browser"),
    "native-fast-classes": (dict(ONE, river_classes=True, _root_river=ROOT_RIVER), "one", 2.5, "native"),
    "native-accurate-classes": (dict(TWO, river_classes=True, _root_river=ROOT_RIVER), "two", 30.0, "native"),
    "native-fast-ob": (dict(_menu([0.75, 1.25], [0.75, 1.25]), river_classes=True, _root_river=ROOT_RIVER), "rich", 2.5, "native"),
    "native-accurate-rich": (dict(RICH, river_classes=True, _root_river=ROOT_RIVER), "rich", 30.0, "native"),
    "native-fast-s-ob": (dict(_menu([0.33, 1.25], [0.33, 1.25]), river_classes=True, _root_river=ROOT_RIVER), "rich", 2.5, "native"),
    "native-fast-m-ob": (dict(_menu([0.5, 1.25], [0.5, 1.25]), river_classes=True, _root_river=ROOT_RIVER), "rich", 2.5, "native"),
    "browser-fast-s-ob": (dict(_menu([0.33, 1.25], [0.33, 1.25]), river_classes=True, _root_river=ROOT_RIVER), "rich", 8.0, "browser"),
    "browser-fast-m-ob": (dict(_menu([0.5, 1.25], [0.5, 1.25]), river_classes=True, _root_river=ROOT_RIVER), "rich", 8.0, "browser"),
    "browser-fast-ob": (dict(_menu([0.75, 1.25], [0.75, 1.25]), river_classes=True, _root_river=ROOT_RIVER), "rich", 8.0, "browser"),
    "browser-accurate-ob": (dict(_menu([0.75, 1.25], [0.75, 1.25]), river_classes=True, _root_river=ROOT_RIVER), "rich", 60.0, "browser"),
    "browser-accurate-rich": (dict(RICH, river_classes=True, _root_river=ROOT_RIVER), "rich", 60.0, "browser"),
    # native Fast's turn before (75/125%, 2.5 s), and with the time the narrowing flop leaves (gto/native.py)
    "native-fast-prev": (dict(_menu([0.75, 1.25], [0.75]), river_classes=True, _root_river=ROOT_RIVER), "rich", 2.5, "native"),
    "native-fast-5s": (dict(_menu([0.75, 1.25], [0.75]), river_classes=True, _root_river=ROOT_RIVER), "rich", 5.0, "native"),
    "native-fast-rich-5s": (dict(_menu([0.33, 0.75, 1.25], [0.75]), river_classes=True, _root_river=ROOT_RIVER), "rich", 5.0, "native"),
    "native-fast-cold": (dict(_menu([0.33, 1.25], [0.75]), river_classes=True, _root_river=ROOT_RIVER), "rich", 2.5, "native"),
    # Fast's turn in a low-SPR pot (a 3bet pot's turn: 3.2 x the pot behind) and a second raise size
    **{f"{w}-fast-{tag}{sfx}": (dict(_menu(turn, [0.75]), river_classes=True, _root_river=ROOT_RIVER), "rich", b, w)
       for w, b0 in (("browser", 8.0), ("native", 5.0)) for sfx, b in (("", b0), ("-cold", 2.5 if w == "native" else None))
       if b is not None
       for tag, turn in (("33-75", [0.33, 0.75]), ("75-125", [0.75, 1.25]), ("75", [0.75]), ("rich", [0.33, 0.75, 1.25]))},
    **{f"{w}-{speed}-r2": (dict(_menu(turn, river, (0.4, 0.75)), river_classes=True, _root_river=ROOT_RIVER), "rich", b, w)
       for w, speed, turn, river, b in (("browser", "fast", [0.33, 1.25], [0.75], 8.0), ("native", "fast", [0.33, 1.25], [0.75], 5.0),
                                        ("browser", "accurate", [0.33, 0.75, 1.25], [0.33, 0.75, 1.25], 60.0),
                                        ("native", "accurate", [0.33, 0.75, 1.25], [0.33, 0.75, 1.25], 30.0))},
}
# idle iterations a second of the one-size every-river turn tree (see the docstring)
BASE_RATE = {"native": 16.8, "browser": 11.6}
BASE_SPOT = "t-k72-3"
clock = time.process_time


def _walk(sol, line, r0, r1):
    """Follow `line` (action names) from the root; returns (node, r0, r1) with the reaches multiplied in."""
    node, r0, r1 = sol.root, r0.copy(), r1.copy()
    for name in line:
        st = node.avg_strategy0()
        a = node.names.index(name)
        if node.player == 0:
            r0 = r0 * st[a]
        else:
            r1 = r1 * st[a]
        node = node.children[a]
    return node, r0, r1


def spot_ranges(name, cache):
    """(board, r_oop, r_ip, pot, stack) at the spot's root: the flop (and turn) lines played by long solves."""
    kind, flop, fline, turn, tline, river, _ = SPOTS[name]
    line, oop, ip, pot, stack = PRE[kind]
    rng = preflop_ranges(line, pf.Table(pf.SEATS[6]), 100)
    key = (kind, flop)
    if key not in cache:
        t0 = time.time()
        cache[key] = solve(cards(flop), rng[oop], rng[ip], pot, stack, cfg=Config(**FLOP_CFG), time_budget=1e9, max_iters=200)
        print(f"  flop {flop} ({kind}): {cache[key].iters} it {time.time() - t0:.0f}s, "
              f"exploitability {cache[key].exploitability():.2f}%", flush=True)
    sol = cache[key]
    node, a, b = _walk(sol, fline, sol.r0, sol.r1)
    r0, r1 = S.expand(sol.ctx, a), S.expand(sol.ctx, b)
    pot_t = _pot_after(sol, fline, pot)
    stack_t, pot = stack - (pot_t - pot) / 2, pot_t
    board = cards(flop) + cards(turn)
    if tline is None:
        return board, r0, r1, pot, stack_t
    key = (kind, flop, turn, tuple(fline))
    if key not in cache:
        t0 = time.time()
        cache[key] = solve(board, r0, r1, pot, stack_t, cfg=Config(**ONE), time_budget=1e9, max_iters=300)
        print(f"  turn {flop}{turn}: {cache[key].iters} it {time.time() - t0:.0f}s, "
              f"exploitability {cache[key].exploitability():.2f}%", flush=True)
    tsol = cache[key]
    node, a, b = _walk(tsol, tline, tsol.r0, tsol.r1)
    pot2 = _pot_after(tsol, tline, pot)
    return board + cards(river), S.expand(tsol.ctx, a), S.expand(tsol.ctx, b), pot2, stack_t - (pot2 - pot) / 2


def _pot_after(sol, line, pot):
    """The pot once `line` has closed the street."""
    node = sol.root
    for name in line:
        node = node.children[node.names.index(name)]
    while not isinstance(node, Decision):
        node = node.child
    return node.pot_now


def load_spot(name):
    """(board, OOP's range, IP's range, pot, stack) of a spot (from RANGES_FILE)."""
    with np.load(RANGES_FILE) as d:
        meta = json.loads(str(d["meta"]))[name]
        return (cards(meta["board"]), d[name + "/oop"].astype(np.float64), d[name + "/ip"].astype(np.float64),
                meta["pot"], meta["stack"])


def build_ranges(names):
    cache, arrays = {}, {}
    meta = {}
    if os.path.exists(RANGES_FILE):
        d = np.load(RANGES_FILE)
        meta = json.loads(str(d["meta"]))
        arrays = {k: d[k] for k in d.files if k != "meta"}
    for name in names:
        print(name, flush=True)
        board, r0, r1, pot, stack = spot_ranges(name, cache)
        from gto.cards import cards_str
        meta[name] = {"board": cards_str(board), "pot": pot, "stack": stack}
        arrays[name + "/oop"], arrays[name + "/ip"] = r0.astype(np.float32), r1.astype(np.float32)
        print(f"  {cards_str(board)} pot {pot} stack {stack}: {r0.sum():.0f} / {r1.sum():.0f} combos", flush=True)
    tmp = RANGES_FILE[:-4] + ".tmp.npz"     # (atomic: a --measure may be reading it)
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    os.replace(tmp, RANGES_FILE)


def tree_kw(name, tree):
    """The reference trees' kwargs for a spot: a river root gets the richer river menu."""
    if tree == "rich" and SPOTS[name][5] is not None:
        return RICH_RIVER
    if tree == "richr" and SPOTS[name][5] is not None:
        return RICH_RIVER_RAISE
    return REF_TREES[tree]


def stats(sol) -> dict:
    """Root-street aggregates (% of the acting player's range at the node): OOP's bets at the root; IP's bets after
    OOP checks, by size; each player's fold / call / raise facing the other's bets (pooled over the sizes, weighted by
    how often each comes, and against 75% pot when the tree has it); the root players' EVs (% of the pot)."""
    root, r0, r1 = sol.root, sol.r0, sol.r1
    out = {}
    st0 = root.avg_strategy0()
    pot = root.pot_now

    def freq(st, a, r):
        return float((st[a] * r).sum() / max(r.sum(), 1e-12) * 100)

    def bet_actions(node):
        return [a for a, n in enumerate(node.names) if n.startswith(("bet", "all-in"))]

    def facing(node, reach, resp, prefix):
        """Pooled response over node's bets reaching `resp` with reaches `reach[a]`."""
        tot = {}
        for a in bet_actions(node):
            ch = node.children[a]
            rr = reach[a]
            if rr.sum() <= 1e-9:
                continue
            s = ch.avg_strategy0()
            for b, m in enumerate(ch.names):
                k = "raise" if m.startswith("raise") or m == "all-in" else m
                v = float((s[b] * rr).sum())
                tot[k] = tot.get(k, 0.0) + v
                if node.names[a] == "bet 75%":
                    out[f"{prefix} vs 75: {k}"] = out.get(f"{prefix} vs 75: {k}", 0.0) + v / rr.sum() * 100
        z = sum(tot.values())
        for k, v in tot.items():
            out[f"{prefix} vs bet: {k}"] = v / max(z, 1e-12) * 100

    bets0 = bet_actions(root)
    out["oop bet"] = sum(freq(st0, a, r0) for a in bets0)
    for a in bets0:
        out[f"oop {root.names[a]}"] = freq(st0, a, r0)
    ipn = root.children[0]
    st1 = ipn.avg_strategy0()
    bets1 = bet_actions(ipn)
    out["ip bet"] = sum(freq(st1, a, r1) for a in bets1)
    for a in bets1:
        out[f"ip {ipn.names[a]}"] = freq(st1, a, r1)
    big = [a for a in bets1 if ipn.names[a] == "all-in" or float(ipn.names[a].split()[-1][:-1]) > 100]
    out["ip overbet"] = sum(freq(st1, a, r1) for a in big)
    # OOP facing IP's bets after checking: reaches r0 * check, each bet weighted by IP's mass betting it
    rc = r0 * st0[0]
    w1 = {a: (st1[a] * r1).sum() for a in bets1}
    facing(ipn, {a: rc * w1[a] / max(sum(w1.values()), 1e-12) for a in bets1}, None, "oop")
    # IP facing OOP's bets
    w0 = {a: (st0[a] * r0).sum() for a in bets0}
    facing(root, {a: r1 * w0[a] / max(sum(w0.values()), 1e-12) for a in bets0}, None, "ip")
    ev = root_values(sol)
    out["ev oop"], out["ev ip"] = float((ev[0] * r0).sum() / r0.sum()), float((ev[1] * r1).sum() / r1.sum())
    return out


def root_values(sol):
    """(2, N) each combo's value at the root under the average strategies: its chips at the end of the hand less
    what it had before the hand (the pot's share it ends with, net of what it put in), % of the root's pot."""
    s = sol.solver
    a, b = sol.r0[None, :].astype(np.float32), sol.r1[None, :].astype(np.float32)
    s.sampling = False
    out = []
    try:
        for p in (0, 1):
            u = s._cfr(sol.root, p, a, b, update=False, use_avg=True)[0]
            cs = s.compat_sum((b if p == 0 else a))[0]
            inv = S._inv_of(sol.root, p)
            out.append(np.where(cs > 1e-12, u / np.where(cs > 1e-12, cs, 1.0), 0.0) + inv)
    finally:
        s.sampling = True
    return np.array(out) / sol.root.pot_now * 100


def ev_error(sol, ref_vals) -> float:
    """Range-weighted mean |EV difference| per combo over both players, % of the pot (ref_vals: (2, 1326))."""
    v = root_values(sol)
    errs = []
    for p, r in ((0, sol.r0), (1, sol.r1)):
        full = S.expand(sol.ctx, v[p])
        rf = S.expand(sol.ctx, r)
        errs.append(float((np.abs(full - np.asarray(ref_vals[p])) * rf).sum() / rf.sum()))
    return float(np.mean(errs))


def solve_spot(name, kw, iters, budget=1e9):
    kw = dict(kw)
    root = kw.pop("_root_river", None)
    if root is not None and SPOTS[name][5] is not None:
        kw["bet_sizes"] = dict(kw["bet_sizes"], river=list(root))
    board, r0, r1, pot, stack = load_spot(name)
    return solve(board, r0, r1, pot, stack, cfg=Config(**kw), time_budget=budget, max_iters=iters)


def build_references(names, iters_turn, iters_river, trees):
    out = os.environ.get("REFS_OUT", REFS_FILE)     # (a second process writes its own files, merged after)
    vout = out[:-5] + "_values.npz" if out != REFS_FILE else VALUES_FILE
    refs = json.load(open(out)) if os.path.exists(out) else {}
    values = dict(np.load(vout)) if os.path.exists(vout) else {}
    for name in names:
        river = SPOTS[name][5] is not None
        for tree in trees:
            t0 = time.time()
            sol = solve_spot(name, tree_kw(name, tree), iters_river if river else iters_turn)
            ex = sol.exploitability()
            st = stats(sol)
            vals = root_values(sol)
            refs.setdefault(name, {})[tree] = {
                "stats": st, "iters": sol.iters, "exploitability": ex, "seconds": time.time() - t0}
            values[f"{name}/{tree}"] = np.array([S.expand(sol.ctx, vals[p]) for p in (0, 1)], dtype=np.float32)
            print(f"{name:12s} {tree:5s} {sol.iters:5d} it {time.time() - t0:6.0f}s expl {ex:.3f}% | "
                  + " ".join(f"{k} {v:.1f}" for k, v in st.items() if not k.startswith("ev")), flush=True)
            json.dump(refs, open(out, "w"), indent=0)
            np.savez_compressed(vout, **values)
    return refs


def per_iter_cpu(name, kw, noblas):
    """(setup seconds, CPU seconds per iteration) of a config on a spot."""
    old = S.NO_BLAS
    S.NO_BLAS = noblas
    try:
        t0 = clock()
        sol = solve_spot(name, kw, 2)
        setup = clock() - t0
        t0 = clock()
        per = 1e9
        for _ in range(2):
            t0 = clock()
            sol.solver.run(sol.r0, sol.r1, time_budget=1e9, max_iters=sol.solver.iters + 4, min_iters=4)
            per = min(per, (clock() - t0) / 4)
        return max(setup - per, 0.0), per
    finally:
        S.NO_BLAS = old


COMPARE = ("oop bet", "ip bet", "ip overbet", "oop vs bet: fold", "oop vs bet: call", "oop vs bet: raise",
           "ip vs bet: fold", "ip vs bet: call", "ip vs bet: raise")


def compared(ref) -> list:
    """The COMPARE statistics that are on the reference's path: a response to bets its player rarely faces
    (under 3% of the bettor's range) is left out."""
    return [k for k in COMPARE if k in ref and not (k.startswith("oop vs") and ref["ip bet"] < 3)
            and not (k.startswith("ip vs") and ref["oop bet"] < 3)]


def freq_error(st, ref):
    keys = [k for k in compared(ref) if k in st or k.endswith("overbet")]
    d = {k: abs(st.get(k, 0.0) - ref[k]) for k in keys}
    return float(np.mean(list(d.values()))), max(d.items(), key=lambda kv: kv[1])


def load_refs():
    """REFS_FILE, with REFS_MORE's spots (a second --reference process's file) merged in."""
    with open(REFS_FILE) as f:
        refs = json.load(f)
    with np.load(VALUES_FILE) as z:
        vals = dict(z)
    more = os.environ.get("REFS_MORE")
    if more:
        for k, v in json.load(open(more)).items():
            refs.setdefault(k, {}).update(v)
        vals.update(np.load(more[:-5] + "_values.npz"))
    for name, trees in refs.items():
        for tree, d in trees.items():
            d["values"] = vals[f"{name}/{tree}"]
    return refs


def budget_iters(name, kw, budget, where, base):
    """Iterations a config gets in `budget` seconds idle: its cost relative to the base tree's on BASE_SPOT measured now."""
    setup, per = per_iter_cpu(name, kw, where == "browser")
    scale = 1.0 / (BASE_RATE[where] * base[where])        # idle seconds per measured second
    return int(max(10, min(1000, (budget - setup * scale) / (per * scale))))


def measure(names, configs, against="rich"):
    """`against`: the richest reference compared with ("rich"; "richr": the rich menu with a second raise size)."""
    refs = load_refs()
    rows = []
    base = {w: per_iter_cpu(BASE_SPOT, ONE, w == "browser")[1] for w in ("native", "browser")}
    for name in names:
        for c in configs:
            kw, tree, budget, where = CONFIGS[c]
            if SPOTS[name][5] is not None and "_root_river" in kw:
                tree = "rich"
            iters = budget_iters(name, kw, budget, where, base)
            sol = solve_spot(name, kw, iters)
            st = stats(sol)
            ex = sol.exploitability()
            same, rich = refs[name][tree], refs[name][against]
            fe_same, worst_same = freq_error(st, same["stats"])
            fe_rich, worst_rich = freq_error(st, rich["stats"])
            row = dict(spot=name, config=c, iters=iters, expl=ex, fe_same=fe_same, fe_rich=fe_rich,
                       ev_same=ev_error(sol, same["values"]), ev_rich=ev_error(sol, rich["values"]),
                       worst_same=worst_same, worst_rich=worst_rich,
                       stats={k: round(v, 1) for k, v in st.items()})
            rows.append(row)
            print(f"{name:12s} {c:16s} {iters:4d} it expl {ex:5.2f}% | freq err same tree {fe_same:4.1f} (worst "
                  f"{worst_same[0]} {worst_same[1]:.1f}), rich {fe_rich:4.1f} ({worst_rich[0]} {worst_rich[1]:.1f}) | "
                  f"EV err {row['ev_same']:.2f} / {row['ev_rich']:.2f} % pot", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--spots")
    ap.add_argument("--trees", default="rich,one,two")
    ap.add_argument("--configs", default="browser-fast,browser-accurate,native-fast,native-accurate,browser-fast-now,browser-accurate-now,native-fast-now,native-accurate-now")
    ap.add_argument("--iters-turn", type=int, default=400)
    ap.add_argument("--iters-river", type=int, default=3000)
    ap.add_argument("--out")
    ap.add_argument("--against", default="rich")
    args = ap.parse_args()
    names = args.spots.split(",") if args.spots else list(SPOTS)
    if args.ranges:
        build_ranges(names)
    if args.reference:
        build_references(names, args.iters_turn, args.iters_river, args.trees.split(","))
    if args.measure:
        rows = measure(names, args.configs.split(","), args.against)
        if args.out:
            json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
