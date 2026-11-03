"""The play after a rare line on an earlier street (advisor.RARE_RESOLVE), and what it does to the next street's
advice, against the remaining game solved exactly.

  solve         the street's own solve at the node after the rare line (barely reached: barely trained)
  resolve       the street solved again from that node, with the ranges the walk reached it with (the advisor's:
                stopped once settled, RARE_RESOLVE_SETTLED)
  resolve-full  the same run to its 1000 iterations (--options solve,resolve,resolve-full)

Reference: the street from that node (the rare actor's range as the walk gave it, the responder's as it was) solved
by b-inary's postflop-solver (scripts/pfbench: every turn and river card, the tree from the node ours) to
--exact-target, its locks holding every decision before the node to the line taken, and walked exactly through the
rest of the street; then the next street solved from its start with those ranges in the advice's own tree (captured)
to --turn-iters (our solver). Each option's next-street answer is priced by that solve's action EVs at its first
decision and after a check there (EV given up, % of the pot at the street's start, reach-weighted over the player's
whole range), with range-frequency and per-combo total variation (points) and your hand's bb given up. "floor" is the
advice's tree with the reference's ranges at the solve option's iterations. Without --exact the street from the node
is our own solve in Accurate's flop model to --ref-iters. A line whose reply is rare at equilibrium too (the rules'
range unbalanced: the reply ~0%) has no equilibrium for the next street to be measured against; --summary counts the
lines whose reply the exact solve takes with at least 10% of the range.

  python scripts/bench_rare_reply.py OUT.jsonl [--speed fast] [--cases a,b] [--exact] [--exact-target 0.05]
         [--exact-threads 1] [--ref-iters 2000] [--turn-iters 1000] [--options solve,resolve]
  python scripts/bench_rare_reply.py --summary OUT.jsonl...
"""
import argparse, dataclasses, json, os, sys, time
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("GTO_THREADS", "1")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import bench_postflop_solver as BP  # noqa: E402
from gto import advisor as A, native, solver as S  # noqa: E402
from gto.cards import ALL_COMBOS, cards as parse_cards, cards_str, COMBO_INDEX  # noqa: E402
from gto.cli import parse_actions  # noqa: E402
from gto.solver import Decision  # noqa: E402

N = 1326
EXACT, EXACT_TARGET, EXACT_THREADS = False, 0.02, 1
B = lambda p, x: {"pos": p, "action": "bet", "amount": x}
R = lambda p, x: {"pos": p, "action": "raise", "to": x}
C = lambda p: {"pos": p, "action": "call"}
X = lambda p: {"pos": p, "action": "check"}
F = lambda p: {"pos": p, "action": "fold"}

PF = {
    "hj_btn": "UTG:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:f",       # live flops (not in the library)
    "co_btn": "UTG:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:f",
    "utg_co": "UTG:r2.5,HJ:f,CO:c,BTN:f,SB:f,BB:f",
    "btn3_hj": "UTG:f,HJ:r2.5,CO:f,BTN:r7.5,SB:f,BB:f,HJ:c",   # 3bet pot, the 3bettor in position
    "sb3_hj": "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:r10,BB:f,HJ:c",     # 3bet pot, the 3bettor out of position
    "btn_bb": "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c",        # library spots
    "co_bb": "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c",
    "bb3_btn": "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:r10,BTN:c",
}


def case(name, pf, board, hero, hole, flop, turn=(), river=(), nodes=None):
    return dict(name=name, pf=pf, board=board, hero=hero, hole=hole, flop=list(flop), turn=list(turn),
                river=list(river), nodes=nodes)


# Each: an earlier-street line with a rare action followed by a reply; the advice asked at the next street's first
# decision (hero to act there). pot 6.5 single-raised (2.5 + 2.5 + 1.5 blinds), 3bet pots 7.5x2+1.5 = 16.5 / 10x2+0.5.
CASES = [
    case("hj_btn_A27_75", "hj_btn", "Ah2h7d4d", "HJ", "KhQh", [B("HJ", 5.0), C("BTN")]),
    case("hj_btn_986_33", "hj_btn", "9h8h6c2s", "HJ", "TsTd", [B("HJ", 2.0), C("BTN")]),
    case("hj_btn_986_75", "hj_btn", "9h8h6c2s", "HJ", "AhKh", [B("HJ", 5.0), C("BTN")]),
    case("hj_btn_T63_75", "hj_btn", "Td6c3s2h", "HJ", "AcQc", [B("HJ", 5.0), C("BTN")]),
    case("hj_btn_552_c75", "hj_btn", "5c5d2h9s", "HJ", "AcKc", [X("HJ"), B("BTN", 5.0), C("HJ")]),
    case("co_btn_K72_75", "co_btn", "Kc7d2s4h", "CO", "AdKd", [B("CO", 5.0), C("BTN")]),
    case("co_btn_552_33", "co_btn", "5c5d2h9s", "CO", "AhKd", [B("CO", 2.0), C("BTN")]),
    case("utg_co_QJ4_c33", "utg_co", "QsJs4d8c", "UTG", "AdQd", [X("UTG"), B("CO", 2.0), C("UTG")]),
    case("utg_co_T63_c75", "utg_co", "Td6c3s2h", "UTG", "9c9d", [X("UTG"), B("CO", 5.0), C("UTG")]),
    case("utg_co_A27_c75", "utg_co", "Ah2h7d4d", "UTG", "KsKc", [X("UTG"), B("CO", 5.0), C("UTG")]),
    case("btn3_hj_A27_d33", "btn3_hj", "Ah2h7d4d", "HJ", "8s8c", [B("HJ", 5.5), C("BTN")]),
    case("btn3_hj_QJ4_c75", "btn3_hj", "QsJs4d8c", "HJ", "AhQh", [X("HJ"), B("BTN", 12.5), C("HJ")]),
    case("btn3_hj_T63_d75", "btn3_hj", "Td6c3s2h", "HJ", "JsJd", [B("HJ", 12.5), C("BTN")]),
    case("sb3_hj_A27_c33", "sb3_hj", "Ah2h7d4d", "SB", "QcQd", [X("SB"), B("HJ", 7.0), C("SB")]),
    case("sb3_hj_QJ4_c33", "sb3_hj", "QsJs4d8c", "SB", "KhKc", [X("SB"), B("HJ", 7.0), C("SB")]),
    case("sb3_hj_T63_75", "sb3_hj", "Td6c3s2h", "SB", "AsAd", [B("SB", 16.0), C("HJ")]),
    case("sb3_hj_K72_c33", "sb3_hj", "Kc7d2s4h", "SB", "AcKd", [X("SB"), B("HJ", 7.0), C("SB")]),
    # the reply a raise (the call above ~0% at equilibrium against the rules' range), then called
    case("hj_btn_A27_75r", "hj_btn", "Ah2h7d4d", "HJ", "KhQh", [B("HJ", 5.0), R("BTN", 13.0), C("HJ")]),
    case("hj_btn_986_33r", "hj_btn", "9h8h6c2s", "HJ", "TsTd", [B("HJ", 2.0), R("BTN", 7.0), C("HJ")]),
    # a library spot (CO vs BB holds every flop): CO's third-pot c-bet on Qh9c4d is rare (3%), BB (you) called
    case("co_bb_Q94_33", "co_bb", "Qh9c4d2s", "BB", "JdTd", [X("BB"), B("CO", 1.8), C("BB")]),
    # library 3bet pots: the big blind's (3bettor, out of position) check is 0% on most flops (the button's answer to
    # it not stored: solved live), the pot 20.5 (a third 6.8, three quarters 15.4)
    case("bb3_btn_K74_xbc", "bb3_btn", "Kc7d4h9s", "BB", "QsQh", [X("BB"), B("BTN", 6.8), C("BB")]),
    case("bb3_btn_A74_xbc", "bb3_btn", "Ac7d4hTs", "BB", "KcKd", [X("BB"), B("BTN", 6.8), C("BB")]),
    case("bb3_btn_A83_xbc", "bb3_btn", "Ac8d3h5s", "BB", "JcJd", [X("BB"), B("BTN", 6.8), C("BB")]),
    case("bb3_btn_K93_xbc", "bb3_btn", "Kc9d3h2c", "BB", "8s8h", [X("BB"), B("BTN", 6.8), C("BB")]),
    case("bb3_btn_KQ4_xbc", "bb3_btn", "KcQd4h7s", "BB", "AhJh", [X("BB"), B("BTN", 6.8), C("BB")]),
    case("bb3_btn_A22_xbc", "bb3_btn", "Ac2d2c9h", "BB", "KhQh", [X("BB"), B("BTN", 6.8), C("BB")]),
    case("bb3_btn_K82_x75c", "bb3_btn", "Kc8d2h5h", "BB", "TdTh", [X("BB"), B("BTN", 15.4), C("BB")]),
    case("bb3_btn_K74_xx", "bb3_btn", "Kc7d4h9s", "BB", "QsQh", [X("BB"), X("BTN")]),
    case("bb3_btn_AK4_xx", "bb3_btn", "AcKd4h8s", "BB", "JsJd", [X("BB"), X("BTN")]),
    case("bb3_btn_A83_xbc_ip", "bb3_btn", "Ac8d3h5s", "BTN", "AhQh", [X("BB"), B("BTN", 6.8), C("BB")], turn=[X("BB")]),
    case("bb3_btn_KQ4_xx_ip", "bb3_btn", "KcQd4h7s", "BTN", "TsTc", [X("BB"), X("BTN")], turn=[X("BB")]),
]


def walk_exact(sol, acts, oop, ip, pot, stack):
    """(node, r0, r1, shares) walked through the real actions with the solve's average strategy, no smoothing."""
    node, r = sol.root, [sol.r0.copy(), sol.r1.copy()]
    bets, shares = [0.0, 0.0], []
    for a in acts:
        p = 0 if a["pos"] == oop else 1
        assert isinstance(node, Decision) and node.player == p, (a, getattr(node, "names", None))
        name = A.Advisor._tree_action_name(node, a, p, bets, pot, stack)
        k = S.match_action(node.names, name)
        before = r[p].sum()
        r[p] = r[p] * node.avg_strategy0()[k]
        shares.append((a["pos"], node.names[k], float(r[p].sum() / max(before, 1e-12))))
        act = a["action"]
        if act in ("bet", "raise"):
            bets[p] = float(a.get("to") if a.get("to") is not None else a["amount"])
        elif act == "all-in":
            bets[p] = stack
        elif act == "call":
            bets[p] = min(bets[1 - p], stack)
        node = node.children[k]
    return node, r[0], r[1], shares


def full(ctx, v):
    return S.expand(ctx, v)


def tv(a, b):
    return float(0.5 * np.abs(a / max(a.sum(), 1e-12) - b / max(b.sum(), 1e-12)).sum())


def node_data(sol, node, r0, r1):
    """(names, strat (A,N) NaN off the solve, evs (A,N), reach of the node's player (N,))"""
    act = sol.ctx.active
    st = np.full((len(node.names), N), np.nan)
    ev = np.full((len(node.names), N), np.nan)
    st[:, act] = node.avg_strategy0()
    ev[:, act] = sol.solver.node_action_evs(node, r0, r1)
    reach = np.zeros(N)
    reach[act] = r0 if node.player == 0 else r1
    return list(node.names), st, ev, reach


def loss(ref_names, S_, V, w, names, P, pot):
    """EV given up (% pot), range frequency TV and per-combo TV (points) of mix P (names) at the reference's node."""
    Q = np.zeros((len(ref_names), N))
    for i, nm in enumerate(ref_names):
        if nm in names:
            Q[i] = P[names.index(nm)]
        else:
            Q[i] = 0.0
    act = w > 0
    S_, V, w, Q = S_[:, act], V[:, act], w[act], Q[:, act].copy()
    bad = np.isnan(Q).any(axis=0) | (np.nan_to_num(Q).sum(axis=0) <= 1e-9)
    if bad.any():
        fill = np.nansum(np.where(bad[None, :], 0, Q) * w[None, :], axis=1)
        fill = fill / fill.sum() if fill.sum() > 0 else np.full(len(Q), 1.0 / len(Q))
        Q[:, bad] = fill[:, None]
    Q = Q / np.maximum(Q.sum(axis=0, keepdims=True), 1e-12)
    W = max(float(w.sum()), 1e-12)
    g = float((w * ((S_ - Q) * V).sum(axis=0)).sum() / W / pot * 100)
    rf = float(0.5 * np.abs(((Q - S_) * w).sum(axis=1)).sum() / W * 100)
    cf = float((w * 0.5 * np.abs(Q - S_).sum(axis=0)).sum() / W * 100)
    return round(g, 4), round(rf, 2), round(cf, 2)


class Cap:
    def __init__(self):
        self.solves, self.resolves = [], []


CAP = Cap()
OPTIONS = ["solve", "resolve"]
SETTLED = A.RARE_RESOLVE_SETTLED
_orig_sc = A.Advisor._solve_cached
_orig_rr = A.Advisor._rare_resolver


def _sc(self, board, r0, r1, pot, stack, budget, bets=(0.0, 0.0), to_act=0, n_raises=0, cfg=None, **kw):
    t = time.time()
    sol = _orig_sc(self, board, r0, r1, pot, stack, budget, bets=bets, to_act=to_act, n_raises=n_raises, cfg=cfg, **kw)
    CAP.solves.append(dict(board=tuple(board), bets=tuple(bets), to_act=to_act, n_raises=n_raises,
                           cfg=cfg or self.cfg, sol=sol, pot=pot, stack=stack, secs=time.time() - t))
    return sol


def _rr(self, board, st, eff, cfg):
    f = _orig_rr(self, board, st, eff, cfg)
    if f is None:
        return None

    def g(R0, R1, bets, to_act, n_raises):
        t = time.time()
        sol, trust = f(R0, R1, bets, to_act, n_raises)
        CAP.resolves.append(dict(board=tuple(board), R0=R0.copy(), R1=R1.copy(), bets=bets, to_act=to_act,
                                 n_raises=n_raises, cfg=cfg, pot=st.pot_start, eff=eff, secs=time.time() - t,
                                 iters=int(sol.iters), expl=float(self._expl(sol)), trust=trust,
                                 oop_ip=None))
        return sol, trust
    return g



A.Advisor._solve_cached = _sc
A.Advisor._rare_resolver = _rr


HIDX = {}
for i, (a, b) in enumerate(ALL_COMBOS):
    HIDX[cards_str([a]) + cards_str([b])] = i
    HIDX[cards_str([b]) + cards_str([a])] = i


def prefix_acts(prefix, oop, stack):
    """Our real actions before the node as the harness's actions (chips on the street after each)."""
    out, bets = [], [0.0, 0.0]
    for a in prefix:
        p = 0 if a["pos"] == oop else 1
        t = a["action"]
        if t in ("check", "call", "fold"):
            if t == "call":
                bets[p] = bets[1 - p]
            out.append({"t": t})
        elif t == "all-in":
            bets[p] = stack
            out.append({"t": "allin", "a": BP.chips(stack)})
        else:
            amt = float(a.get("to") if a.get("to") is not None else a["amount"])
            out.append({"t": "raise" if bets[1 - p] > bets[p] + 1e-9 else "bet", "a": BP.chips(amt)})
            bets[p] = amt
    return out


def exact_reply(sub_root, board, R0, R1, pot, stack, prefix, rest, oop, target=0.02, iters=3000, threads=1,
                compress=False):
    """{"mem_gb", "iters", "expl", "nodes": [{"path", "player", "acts", "strat" (A, 1326), "reach" (1326)}]}: their
    equilibrium at the node and down `rest` (our real actions after the node). `sub_root`: our tree from the node
    (a Solution's root rooted there), whose lines (prefixed) are the tree after it."""
    pre = prefix_acts(prefix, oop, stack)
    lines = [pre[:i + 1] for i in range(len(pre)) if pre[i]["t"] in ("bet", "raise", "allin")]
    lines += [pre + l for l in BP.lines_of(sub_root)]
    sp = {"board": [cards_str([c]) for c in board], "oop": BP.range_spec(R0, board), "ip": BP.range_spec(R1, board),
          "pot": BP.chips(pot), "stack": BP.chips(stack), "lines": lines}
    # every prefix decision locked to the action taken, for every hand of its player
    locks, path, bets = [], [], [0, 0]
    ranges = (sp["oop"][0], sp["ip"][0])
    for i, (a, act) in enumerate(zip(prefix, pre)):
        p = 0 if a["pos"] == oop else 1
        # their actions at this node: the line's action and check/call/fold as the node allows
        facing = bets[1 - p] > bets[p]
        acts = ([{"t": "fold"}, {"t": "call"}] if facing else [{"t": "check"}])
        acts += [l[i] for l in lines if len(l) > i and l[:i] == pre[:i] and l[i]["t"] in ("bet", "raise", "allin")]
        uniq = []
        for x in acts:
            if x not in uniq:
                uniq.append(x)
        k = uniq.index(act)
        hands = ranges[p]
        strat = [[1.0 if j == k else 0.0] * len(hands) for j in range(len(uniq))]
        locks.append({"path": list(path), "acts": uniq, "hands": hands, "strat": strat})
        path.append(BP.act_str(act))
        if act["t"] in ("bet", "raise", "allin"):
            bets[p] = act["a"]
        elif act["t"] == "call":
            bets[p] = bets[1 - p]
    # the node, then each node down `rest` (our real actions after it, as the harness names them)
    more, node, bets_r = [], sub_root, list(sub_root.bets)
    from gto.advisor import Advisor
    from gto.solver import match_action
    for a in rest:
        pl = node.player
        k = match_action(node.names, Advisor._tree_action_name(node, a, pl, bets_r, pot, stack))
        more.append(BP.act_of(node, node.names[k], node.children[k]))
        node = node.children[k]
        if not hasattr(node, "bets"):
            break
        bets_r = list(node.bets)
    dumps = [list(path) + [BP.act_str(x) for x in more[:i]] for i in range(len(more))]
    out = BP.run(dict(sp, mode="solve", iters=iters, target_pct=target, locks=locks, compress=compress, dump=dumps),
              threads)
    nodes = []
    for nd in out["nodes"]:
        st = np.full((len(nd["acts"]), 1326), np.nan)
        reach = np.zeros(1326)
        idx = [HIDX[h] for h in nd["hands"]]
        st[:, idx] = np.array(nd["strat"])
        reach[idx] = np.array(nd["reach"])
        nodes.append({"path": nd["path"], "player": nd["player"], "acts": nd["acts"], "strat": st, "reach": reach})
    return {"mem_gb": out["mem_gb"], "iters": out["iters"], "expl": out["expl_pct"], "secs": out["secs"],
            "nodes": nodes}


def state(c, upto_turn=None):
    return {"players": 6, "stack": 100, "preflop": parse_actions(PF[c["pf"]]), "board": c["board"], "hero": c["hero"],
            "hole": c["hole"], "flop": c["flop"], "turn": c["turn"] if upto_turn is None else upto_turn,
            "river": c["river"]}


def accurate_model(cfg):
    return dataclasses.replace(cfg, river_buckets=None, turn_buckets=False, bucket_river_step=1, river_classes=True,
                               river_strength=native.RIVER_STRENGTH)


def run_case(c, speed, ref_iters, turn_iters, out):
    board = parse_cards(c["board"])
    street_len = len(board)
    earlier = "flop" if street_len == 4 else "turn"
    acts_earlier = c["flop"] if earlier == "flop" else c["turn"]
    rec = {"case": c["name"], "speed": speed}
    adv = native.advisor(speed)
    res_by = {}
    for opt in OPTIONS:
        A.RARE_RESOLVE = opt != "solve"
        A.RARE_RESOLVE_SETTLED = None if opt == "resolve-full" else SETTLED
        # (each option's re-solves from scratch: the cache would hand one option's to the next to go on from)
        for k in [k for k in adv._cache if len(k[2]) == len(board) - 1 and (tuple(k[5]) != (0.0, 0.0) or k[6] != 0)]:
            del adv._cache[k]
            adv._by_spot.pop(k[2:], None)
        CAP.solves.clear()
        n_res = len(CAP.resolves)
        t = time.time()
        res = adv.advise(state(c))
        dt = time.time() - t
        if "error" in res:
            rec["error"] = res["error"]
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(c["name"], "error", res["error"])
            return
        rare = [o for o in res.get("off_solution", []) if o["why"] == "rare" and o["street"] == earlier]
        cur = [s for s in CAP.solves if len(s["board"]) == street_len and s["bets"] == (0.0, 0.0)]
        if not cur:
            print(c["name"], opt, "no street solve captured")
            return
        sol = cur[-1]["sol"]
        hero_col = COMBO_INDEX[tuple(sorted(parse_cards(c["hole"])))]
        res_by[opt] = dict(sol=sol, cfg=cur[-1]["cfg"], pot=cur[-1]["pot"], stack=cur[-1]["stack"], res=res, secs=dt, rare=rare,
                           resolves=CAP.resolves[n_res:], fills=[o for o in res.get("off_solution", [])
                                                                 if o["why"] == "filled"])
        print(f"{c['name']} {speed} {opt}: {dt:.1f}s rare={[(o['pos'], o['action'], o['share']) for o in rare]} "
              f"rec={res['recommended']} villain {res.get('villain_range_combos')} hero {res.get('hero_range_combos')}"
              + (f" resolves={[(round(r['secs'], 2), r['iters'], round(r['expl'], 3)) for r in CAP.resolves[n_res:]]}"
                 if len(CAP.resolves) > n_res else ""))
    if not res_by["solve"]["rare"]:
        rec["skip"] = "no rare line"
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(c["name"], "no rare line: skipped")
        return
    rs = res_by["resolve"]["resolves"]
    if not rs:
        rec["skip"] = "no resolve (rare line ends the street)"
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(c["name"], "no resolve")
        return
    r0 = rs[0]
    # --- the reference: the earlier street from the node, Accurate's flop model (or the turn tree), long
    rcfg = accurate_model(r0["cfg"]) if earlier == "flop" else r0["cfg"]
    t = time.time()
    ref = S.solve(r0["board"], r0["R0"], r0["R1"], r0["pot"], r0["eff"], bets=r0["bets"], to_act=r0["to_act"],
                  n_raises=r0["n_raises"], cfg=rcfg, time_budget=3600, max_iters=ref_iters)
    ref_secs = time.time() - t
    ref_expl = ref.exploitability()
    # oop / ip from the earlier street's actions: the first actor of the street is OOP
    oop = acts_earlier[0]["pos"]
    ip = next(a["pos"] for a in acts_earlier if a["pos"] != oop)
    # the rest of the street after the rare action: the actions from the node on
    k_rare = None
    bets = [0.0, 0.0]
    for i, a in enumerate(acts_earlier):
        p = 0 if a["pos"] == oop else 1
        if (bets[0], bets[1]) == tuple(r0["bets"]) and p == r0["to_act"] and i > 0:
            k_rare = i
            break
        if a["action"] in ("bet", "raise"):
            bets[p] = float(a.get("to") if a.get("to") is not None else a["amount"])
        elif a["action"] == "call":
            bets[p] = bets[1 - p]
    rest = acts_earlier[k_rare:]
    root = ref.root
    rshare = {nm: float((full(ref.ctx, ref.root.avg_strategy0()[j]) * (r0["R0"] if root.player == 0 else r0["R1"])).sum())
              for j, nm in enumerate(root.names)}
    node, e0, e1, shares = walk_exact(ref, rest, oop, ip, r0["pot"], r0["eff"])
    REF0, REF1 = full(ref.ctx, e0), full(ref.ctx, e1)
    rec.update(ref_iters=int(ref.iters), ref_secs=round(ref_secs, 1), ref_expl=round(ref_expl, 4),
               ref_reply=rshare, ref_shares=shares, node_combos=[float(r0["R0"].sum()), float(r0["R1"].sum())],
               resolve_secs=[round(x["secs"], 2) for x in rs], resolve_iters=[x["iters"] for x in rs],
               resolve_expl=[round(x["expl"], 3) for x in rs])
    print(f"  ref {ref.iters} it {ref_secs:.1f}s expl {ref_expl:.4f}%: reply {({k: round(v, 1) for k, v in rshare.items()})}"
          f" shares {shares}  node combos {r0['R0'].sum():.1f}/{r0['R1'].sum():.1f}")
    if EXACT:
        t = time.time()
        ex = exact_reply(ref.root, r0["board"], r0["R0"], r0["R1"], r0["pot"], r0["eff"], acts_earlier[:k_rare],
                         rest, oop, target=EXACT_TARGET, iters=5000, threads=EXACT_THREADS, compress=True)
        X0, X1 = r0["R0"].copy(), r0["R1"].copy()
        xs, xr = [], {}
        for i, nd in enumerate(ex["nodes"]):
            pl = nd["player"]
            rp = X0 if pl == 0 else X1
            if i == 0:
                xr = {nm: float((np.nan_to_num(nd["strat"][j]) * rp).sum()) for j, nm in enumerate(nd["acts"])}
            # the action taken at this node: our tree's name for rest[i] mapped to theirs by order of the line
        node_r = ref.root
        bets_r = list(node_r.bets)
        for i, a in enumerate(rest):
            nd = ex["nodes"][i]
            k = S.match_action(node_r.names, A.Advisor._tree_action_name(node_r, a, node_r.player, bets_r, r0["pot"], r0["eff"]))
            nm = BP.act_str(BP.act_of(node_r, node_r.names[k], node_r.children[k]))
            j = nd["acts"].index(nm)
            pl = nd["player"]
            before = (X0 if pl == 0 else X1).sum()
            if pl == 0:
                X0 = X0 * np.nan_to_num(nd["strat"][j])
            else:
                X1 = X1 * np.nan_to_num(nd["strat"][j])
            xs.append((a["pos"], nm, float((X0 if pl == 0 else X1).sum() / max(before, 1e-12))))
            node_r = node_r.children[k]
            if hasattr(node_r, "bets"):
                bets_r = list(node_r.bets)
        rec.update(exact_iters=ex["iters"], exact_expl=round(ex["expl"], 4), exact_secs=round(time.time() - t, 1),
                   exact_mem=round(ex["mem_gb"], 3), exact_reply=xr, exact_shares=xs,
                   ours_vs_exact_tv=[round(tv(REF0, X0), 3), round(tv(REF1, X1), 3)])
        print(f"  exact {ex['iters']} it {time.time() - t:.1f}s expl {ex['expl']:.4f}% mem {ex['mem_gb']:.3f} GB: reply "
              f"{({k: round(v, 1) for k, v in xr.items()})} shares {xs}; ours vs exact tv {rec['ours_vs_exact_tv']}")
        REF0, REF1 = X0, X1
    # --- the next street's reference: from its start with the reference's ranges, the advice's own tree, long
    sol_s = res_by["solve"]["sol"]
    tcfg = res_by["solve"]["cfg"]
    # (hero's cards out of the villain's range, as the advice has them)
    hero_p = 0 if c["hero"] == oop else 1
    hc = parse_cards(c["hole"])
    from gto.solver import MEMBER_FULL
    keep = np.prod([1.0 - MEMBER_FULL[x] for x in hc], axis=0)
    REF = [REF0, REF1]
    REF[1 - hero_p] = REF[1 - hero_p] * keep
    pot_t, eff_t = res_by["solve"]["pot"], res_by["solve"]["stack"]
    t = time.time()
    tsol = S.solve(board, REF[0], REF[1], pot_t, eff_t, cfg=tcfg, time_budget=3600, max_iters=turn_iters)
    tsecs = time.time() - t
    texpl = tsol.exploitability()
    rec.update(turn_ref_iters=int(tsol.iters), turn_ref_secs=round(tsecs, 1), turn_ref_expl=round(texpl, 4))
    print(f"  next-street ref {tsol.iters} it {tsecs:.1f}s expl {texpl:.4f}%")
    # responder's / both ranges entering the street per option
    nodes = c["nodes"] or [[], [X(oop)]]   # (the street's first decision, and after a check there)
    rec["opts"] = {}
    # floor: the advice's street solve with the reference's ranges (as many iterations as the "solve" option's)
    fl = S.solve(board, REF[0], REF[1], pot_t, eff_t, cfg=tcfg, time_budget=3600, max_iters=int(sol_s.iters))
    res_by["floor"] = dict(sol=fl, res=res_by["solve"]["res"], secs=0.0, rare=[], fills=[])
    for opt, d in res_by.items():
        sol = d["sol"]
        o = {"secs": round(d["secs"], 2), "rare": [(x["pos"], x["action"], x["share"]) for x in d["rare"]],
             "fills": d["fills"], "rec": d["res"]["recommended"],
             "combos": [float(sol.r0.sum()), float(sol.r1.sum())],
             "tv": [round(tv(full(sol.ctx, sol.r0), REF[0]), 3), round(tv(full(sol.ctx, sol.r1), REF[1]), 3)]}
        o["nodes"] = []
        for path in nodes:
            try:
                rn, a0, a1, sh = walk_exact(tsol, path, oop, ip, pot_t, eff_t)
                on, b0, b1, _ = walk_exact(sol, path, oop, ip, pot_t, eff_t)
            except AssertionError:
                continue
            names, S_, V, w = node_data(tsol, rn, a0, a1)
            P = np.full((len(on.names), N), np.nan)
            P[:, sol.ctx.active] = on.avg_strategy0()
            g = loss(names, S_, V, w, list(on.names), P, pot_t)
            row = {"path": [x["action"] for x in path], "player": rn.player, "loss": g,
                   "ref_freq": {nm: round(float((S_[j][w > 0] * w[w > 0]).sum() / w.sum()), 3) for j, nm in enumerate(names)},
                   "opt_freq": {nm: round(float(np.nansum(P[j] * w) / w.sum()), 3) for j, nm in enumerate(on.names)},
                   "line_share": sh}
            if rn.player == hero_p:
                hc_i = hero_col
                if not np.isnan(S_[0, hc_i]) and not np.isnan(P[0, hc_i]):
                    ev_ref = {nm: round(float(V[j, hc_i]), 2) for j, nm in enumerate(names)}
                    got = {nm: round(float(P[j, hc_i]), 3) for j, nm in enumerate(on.names)}
                    ref_mix = {nm: float(S_[j, hc_i]) for j, nm in enumerate(names)}
                    # EV given up by the hand's mix against the reference's (its EVs per action), bb
                    lost = sum((ref_mix[nm] - got.get(nm, 0.0)) * ev_ref[nm] for nm in names)
                    row["hand"] = {"ref_mix": {k: round(v, 3) for k, v in ref_mix.items()}, "ref_ev": ev_ref,
                                   "opt_mix": got, "lost_bb": round(lost, 3)}
            o["nodes"].append(row)
        rec["opts"][opt] = o
        print(f"  {opt:8s} combos {o['combos'][0]:.1f}/{o['combos'][1]:.1f} tv {o['tv']} rec {o['rec']}  " +
              "  ".join(f"{'/'.join(n['path']) or 'root'}: {n['loss']}" + (f" hand -{n['hand']['lost_bb']}bb" if n.get('hand') else "")
                        for n in o["nodes"]))
    out.write(json.dumps(rec, default=float) + "\n"); out.flush()


def summary(files):
    """Per line the reply at equilibrium and each option's next-street EV given up; then over the lines whose reply is
    common at equilibrium (the exact solve's share >= 10%)."""
    recs = [json.loads(l) for f in files for l in open(f) if l.strip()]
    OPTS = []
    for r in recs:
        for o in r.get("opts", {}):
            if o not in OPTS:
                OPTS.append(o)
    rows = {o: {"root": [], "check": [], "hand": [], "tvB": []} for o in OPTS}
    for r in recs:
        if "opts" not in r:
            print(f"{r['case']:20s} {r.get('skip') or r.get('error') or r.get('exception')}")
            continue
        xs = r.get("exact_shares") or r.get("ref_shares")
        reply = xs[0][2] if xs else None
        common = reply is not None and reply >= 0.10
        tag = "common" if common else "RARE"
        print(f"{r['case']:20s} reply(exact) {reply * 100:5.1f}% [{tag}] ours {r['ref_shares'][0][2] * 100:5.1f}% "
              f"exact expl {r.get('exact_expl')} it {r.get('exact_iters')} mem {r.get('exact_mem')}  "
              f"next ref expl {r['turn_ref_expl']} it {r['turn_ref_iters']}; resolve {r['resolve_secs']}s {r['resolve_iters']}")
        for o in OPTS:
            d = r["opts"].get(o)
            if not d:
                continue
            ns = {"/".join(n["path"]) or "root": n for n in d["nodes"]}
            root, chk = ns.get("root"), ns.get("check")
            hand = next((n["hand"]["lost_bb"] for n in d["nodes"] if n.get("hand")), None)
            print(f"    {o:8s} combos {d['combos'][0]:6.1f}/{d['combos'][1]:6.1f} tv {d['tv']}  root {root['loss'] if root else '-'}"
                  f"  after check {chk['loss'] if chk else '-'}  hand {hand}")
            if common:
                if root:
                    rows[o]["root"].append(root["loss"][0])
                if chk:
                    rows[o]["check"].append(chk["loss"][0])
                if hand is not None:
                    rows[o]["hand"].append(hand)
                rows[o]["tvB"].append(max(d["tv"]))
    print("\nreplies common at equilibrium (exact share >= 10%): EV given up, % of the pot (mean / median / max)")
    for o in OPTS:
        for k in ("root", "check"):
            a = np.array(rows[o][k])
            if len(a):
                print(f"  {o:8s} {k:6s} n={len(a):2d} mean {a.mean():.3f} median {np.median(a):.3f} max {a.max():.3f}")
        a = np.array(rows[o]["tvB"])
        if len(a):
            print(f"  {o:8s} range TV (worse player) mean {a.mean():.3f}")
        a = np.array(rows[o]["hand"])
        if len(a):
            print(f"  {o:8s} hand bb lost mean {a.mean():.3f} max {a.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--summary", nargs="+", metavar="JSONL")
    ap.add_argument("--speed", default="fast")
    ap.add_argument("--cases", default="")
    ap.add_argument("--ref-iters", type=int, default=2000)
    ap.add_argument("--turn-iters", type=int, default=1000)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--exact-target", type=float, default=0.05)
    ap.add_argument("--exact-threads", type=int, default=1)
    ap.add_argument("--options", default="solve,resolve")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)
    want = set(a.cases.split(",")) if a.cases else None
    global EXACT, EXACT_TARGET, EXACT_THREADS, OPTIONS
    EXACT, EXACT_TARGET, EXACT_THREADS = a.exact, a.exact_target, a.exact_threads
    OPTIONS = a.options.split(",")
    assert OPTIONS[:2] == ["solve", "resolve"], "the reference is built from the solve and resolve options"
    with open(a.out, "a") as out:
        for c in CASES:
            if want and c["name"] not in want:
                continue
            try:
                run_case(c, a.speed, a.ref_iters, a.turn_iters, out)
            except Exception as e:
                import traceback; traceback.print_exc()
                out.write(json.dumps({"case": c["name"], "speed": a.speed, "exception": repr(e)}) + "\n"); out.flush()


if __name__ == "__main__":
    main()

