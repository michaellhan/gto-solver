"""Rare lines on the river: the advice facing a bet or raise its own solve takes with under RARE_LINE of the bettor's range,
against the river's exact equilibrium on the same ranges.

    python scripts/bench_rare_lines.py [--spots 40] [--seed 3] [--speed fast] [--out reports/rare_lines.jsonl]

Each spot is a hand (a preflop line, a stack, a random board, a flop and turn line that are common at equilibrium) whose
river ends with the villain betting or raising big (100-200% of the pot or all-in) into you, or leading small. The
Advisor answers it twice on one cache, with RARE_KEEP off (the rules' range for a rare line: value only for a pot-sized
bet or more) and on (a settled solve keeps its own range). The river is then solved from its start on the ranges the
Advisor reached it with (its solve's tree, the real sizes added) to 0.02% of the pot, and for each answer:
  - given_up: what the bettor wins more, in % of the pot at the river's start, by best-responding over the whole river
    to your whole range playing the advice's response at the node (your other decisions at equilibrium): how much
    the response can be exploited by betting there with other hands;
  - freq: your range's fold / call / raise there, against the exact solve's;
  - the hand's own answer against the exact solve's top action and EVs.
Records go to --out, one a spot; the summary prints at the end.
"""
import argparse, json, os, random, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from gto import native, advisor as AD   # (native first: one BLAS thread)
from gto.solver import solve, Decision, Config, match_action
from gto.cards import cards as parse_cards, COMBO_INDEX, ALL_COMBOS, cards_str, card_str
import numpy as np

R = lambda p, a, x=None: dict({"pos": p, "action": a}, **({"to": x} if a == "raise" else {"amount": x} if x is not None else {}))
F = lambda *ps: [R(p, "fold") for p in ps]

# (name, players, preflop, oop, ip, pot entering the flop at 100bb)
PREFLOP = [
    ("CO v BB, 5-handed", 5, F("HJ") + [R("CO", "raise", 2.5)] + F("BTN", "SB") + [R("BB", "call")], "BB", "CO"),
    ("BTN v BB", 6, F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB") + [R("BB", "call")], "BB", "BTN"),
    ("UTG v BTN", 6, [R("UTG", "raise", 2.5)] + F("HJ", "CO") + [R("BTN", "call")] + F("SB", "BB"), "UTG", "BTN"),
    ("SB v BB", 6, F("UTG", "HJ", "CO", "BTN") + [R("SB", "raise", 3)] + [R("BB", "call")], "SB", "BB"),
    ("BB 3bet v BTN", 6, F("UTG", "HJ", "CO") + [R("BTN", "raise", 2.5)] + F("SB") + [R("BB", "raise", 11),
                                                                                       R("BTN", "call")], "BB", "BTN"),
]
STACKS = (100, 100, 200, 50)
BIG = (1.0, 1.36, 2.0, "all-in")


def pot_of(pre, players):
    """Chips in after preflop (blinds folded included)."""
    put = {"SB": 0.5, "BB": 1.0}
    for a in pre:
        if a["action"] == "raise":
            put[a["pos"]] = a["to"]
        elif a["action"] == "call":
            put[a["pos"]] = max(put.values())
    return sum(put.values())


def make_spot(rng, i):
    name, players, pre, oop, ip = PREFLOP[i % len(PREFLOP)]
    stack = STACKS[(i // len(PREFLOP)) % len(STACKS)]
    deck = list(range(52))
    rng.shuffle(deck)
    board = deck[:5]
    pot = pot_of(pre, players)
    kind = rng.choice(["lead", "lead", "bet", "probe_raise", "cb_lead"])
    flop = [R(oop, "check"), R(ip, "check")]
    turn = [R(oop, "check"), R(ip, "check")]
    if kind == "cb_lead":
        c = round(0.33 * pot * 2) / 2
        flop = [R(oop, "check"), R(ip, "bet", c), R(oop, "call")]
        pot += 2 * c
    big = rng.choice(BIG)
    left = stack - (pot / 2)            # (both have put the same in)
    amt = left if big == "all-in" else min(left, round(big * pot * 2) / 2)
    act = "all-in" if amt >= left - 1e-9 else "bet"
    if kind in ("lead", "cb_lead"):       # OOP leads the river big into you (IP)
        river = [R(oop, act, None if act == "all-in" else amt)]
        hero = ip
    elif kind == "bet":                   # you check, IP bets big
        river = [R(oop, "check"), R(ip, act, None if act == "all-in" else amt)]
        hero = oop
    else:                                 # you lead a third, IP raises big
        b = round(0.33 * pot * 2) / 2
        to = left if big == "all-in" else min(left, round(b + big * (pot + 2 * b) * 2) / 2)
        river = [R(oop, "bet", b), R(ip, "all-in" if to >= left - 1e-9 else "raise", None if to >= left - 1e-9 else to)]
        hero = oop
    for x in river:
        if x["action"] == "all-in":
            x.pop("amount", None); x.pop("to", None)
    st = {"players": players, "hero": hero, "stack": stack, "board": cards_str(board), "preflop": pre, "flop": flop,
          "turn": turn, "river": river}
    return dict(name=name, kind=kind, big=big, stack=stack, state=st, hero_p=0 if hero == oop else 1)


def capture(adv, state):
    calls = []
    orig = AD.Advisor._solve_cached

    def spy(self, board, r0, r1, pot, stack, budget, bets=(0.0, 0.0), to_act=0, n_raises=0, cfg=None, **kw):
        sol = orig(self, board, r0, r1, pot, stack, budget, bets=bets, to_act=to_act, n_raises=n_raises, cfg=cfg, **kw)
        calls.append(dict(board=tuple(board), r0=np.array(r0), r1=np.array(r1), pot=pot, stack=stack, bets=tuple(bets),
                          to_act=to_act, cfg=cfg, sol=sol))
        return sol
    AD.Advisor._solve_cached = spy
    try:
        res = adv.advise(state)
    finally:
        AD.Advisor._solve_cached = orig
    return res, calls


def real_name(node, a, bets, pot, stack):
    return AD.Advisor._tree_action_name(node, a, node.player, bets, pot, stack)


def walk_to(sol, acts, oop, ip, pot, stack):
    """(node, r0, r1) after the river's actions, every decision by the solve's average strategy."""
    node, r0, r1 = sol.root, sol.r0.copy(), sol.r1.copy()
    bets = [0.0, 0.0]
    for a in acts:
        p = 0 if a["pos"] == oop else 1
        k = match_action(node.names, real_name(node, a, bets, pot, stack))
        s = node.avg_strategy0()[k]
        if p == 0:
            r0 = r0 * s
        else:
            r1 = r1 * s
        ch = node.children[k]
        bets = list(ch.bets) if isinstance(ch, Decision) else bets
        node = ch
    return node, r0, r1


def family(n):
    k = n.split()[0]
    return "raise" if k in ("raise", "all-in", "bet") else k


def response(sol, node, hero_p, rh_full):
    """{combo index: {family: freq}} of the hero's response at `node` of `sol`."""
    st = node.avg_strategy0()
    out = {}
    for j, c in enumerate(sol.ctx.active):
        if rh_full[int(c)] <= 0:
            continue
        d = {}
        for a, n in enumerate(node.names):
            d[family(n)] = d.get(family(n), 0.0) + float(st[a, j])
        out[int(c)] = d
    return out


def range_freq(resp, rh_full):
    tot, acc = 0.0, {}
    for c, d in resp.items():
        w = rh_full[c]
        tot += w
        for f, x in d.items():
            acc[f] = acc.get(f, 0.0) + w * x
    return {f: round(x / max(tot, 1e-12), 3) for f, x in acc.items()}


def exact_eval(start, state, hero_p, oop, ip, responses, target=0.02):
    """The river from its start, solved to `target`; per response {family: freq} by combo: the bettor's best-response gain
    (% of the pot) with the hero locked to it at the node."""
    cfg = (start["cfg"] or Config()).copy()
    pot = start["pot"]
    acts = state["river"]
    # the real sizes into the tree (exactly, beside the tree's own)
    bets, bs, rs = [0.0, 0.0], set(cfg.bet_sizes["river"]), set(cfg.raise_sizes["river"])
    for a in acts:
        p = 0 if a["pos"] == oop else 1
        tot = pot + bets[0] + bets[1]
        to = start["stack"] if a["action"] == "all-in" else (a.get("to") or a.get("amount"))
        if a["action"] in ("bet", "raise", "all-in") and to:
            if max(bets) <= 0:
                bs.add(round((to - bets[p]) / tot, 4))
            else:
                call = max(bets) - bets[p]
                rs.add(round((to - max(bets)) / (tot + call), 4))
            bets[p] = to
        elif a["action"] == "call":
            bets[p] = max(bets)
    cfg.bet_sizes = dict(cfg.bet_sizes, river=sorted(bs))
    cfg.raise_sizes = dict(cfg.raise_sizes, river=sorted(rs))
    cfg.allin_spr = max(cfg.allin_spr, 0.0)
    t = time.time()
    sol = solve(start["board"], start["r0"], start["r1"], pot, start["stack"], cfg=cfg, time_budget=120, max_iters=4000)
    expl = sol.solver.exploitability(sol.r0, sol.r1)
    node, r0, r1 = walk_to(sol, acts, oop, ip, pot, start["stack"])
    vp = 1 - hero_p
    rv_root = sol.r0 if vp == 0 else sol.r1
    bettor = (r0 if vp == 0 else r1)
    bettor_share = float(bettor.sum() / max(float(rv_root.sum()), 1e-12))
    rh = r1 if hero_p == 1 else r0
    rh_full = np.zeros(1326)
    rh_full[sol.ctx.active] = rh
    eq_resp = response(sol, node, hero_p, rh_full)
    evs = sol.solver.node_action_evs(node, r0, r1)
    a0, b0 = sol.r0[None, :].astype(np.float32), sol.r1[None, :].astype(np.float32)
    pairs = float((sol.solver.compat_sum(sol.r1[None, :])[0] * sol.r0).sum())
    rr = sol.r0 if vp == 0 else sol.r1
    u_eq = sol.solver._best_response(sol.root, vp, a0, b0)
    base = float((u_eq[0] * rr).sum())
    fam = [family(n) for n in node.names]
    gains = {}
    for key, resp in responses.items():
        sig = np.array(node.avg_strategy0(), dtype=np.float64)
        for j, c in enumerate(sol.ctx.active):
            d = resp.get(int(c))
            if not d:
                continue
            col = np.zeros(len(fam))
            for f, x in d.items():
                if f in fam:
                    col[fam.index(f)] += x
            if col.sum() > 0:
                sig[:, j] = col / col.sum()
        sig32 = sig.astype(np.float32)[None]
        lock = lambda nd, strat, r, _n=node, _s=sig32: _s if nd is _n else strat
        u = sol.solver._best_response(sol.root, vp, a0, b0, lock)
        gains[key] = round((float((u[0] * rr).sum()) - base) / pairs / sol.root.pot_now * 100, 3)
    return dict(sol=sol, node=node, expl=round(expl, 4), secs=round(time.time() - t, 1), bettor_share=round(bettor_share, 4),
                bettor_combos=round(float(bettor.sum()), 1), eq_freq=range_freq(eq_resp, rh_full), gains=gains,
                evs=evs, rh_full=rh_full, names=list(node.names))


def hero_hand(rng, start, hero_p, board):
    """A random combo of the hero's range entering the river, by its weight."""
    r = (start["r1"] if hero_p == 1 else start["r0"]).copy()
    dead = set(board)
    for i, (a, b) in enumerate(ALL_COMBOS):
        if a in dead or b in dead:
            r[i] = 0
    p = r / r.sum()
    i = int(rng.choice(len(p), p=p))
    return cards_str(ALL_COMBOS[i]), i


def run_spot(spot, speed, rng):
    st = dict(spot["state"])
    oop_ip = [x for x in (st["preflop"]) if x["action"] in ("raise", "call")]
    name, players, pre, oop, ip = next(p for p in PREFLOP if p[0] == spot["name"])
    hero_p = spot["hero_p"]
    adv = native.advisor(speed, 2, None, None)
    # a first pass with any hand of the range finds the river's start
    AD.RARE_KEEP = False
    probe = dict(st, hole=None)
    board = parse_cards(st["board"])
    deck = [c for c in range(52) if c not in board]
    st["hole"] = card_str(deck[0]) + card_str(deck[1])
    res, calls = capture(adv, st)
    starts = [c for c in calls if len(c["board"]) == 5 and c["bets"] == (0.0, 0.0)]
    if "error" in res or not starts:
        return {"skipped": res.get("error") or "no river solve"}
    hole, ci = hero_hand(rng, starts[-1], hero_p, board)
    st["hole"] = hole
    out = dict(name=spot["name"], kind=spot["kind"], big=spot["big"], stack=spot["stack"], board=st["board"],
               hero=st["hero"], hole=hole, river=st["river"])
    answers, resp = {}, {}
    for key, keep in (("rules", False), ("kept", True)):
        AD.RARE_KEEP = keep
        t = time.time()
        res, calls = capture(adv, st)
        if "error" in res:
            return {"skipped": res["error"]}
        starts = [c for c in calls if len(c["board"]) == 5 and c["bets"] == (0.0, 0.0)]
        resolves = [c for c in calls if len(c["board"]) == 5 and c["bets"] != (0.0, 0.0)]
        notes = [n for n in res.get("notes", []) if n.startswith("river")]
        if any("solved again from here" in n for n in notes) and resolves:
            rs = resolves[-1]
            rsol = rs["sol"]
            rh = (rs["r1"] if hero_p == 1 else rs["r0"]).copy()
            resp[key] = response(rsol, rsol.root, hero_p, rh)
            how = "solved from the node"
        else:
            s0 = starts[-1]
            node, r0, r1 = walk_to(s0["sol"], st["river"], oop, ip, s0["pot"], s0["stack"])
            rh = np.zeros(1326)
            rh[s0["sol"].ctx.active] = r1 if hero_p == 1 else r0
            resp[key] = response(s0["sol"], node, hero_p, rh)
            how = "the river's solve"
        answers[key] = dict(rec=res["recommended"], acts=[(a["action"], a["freq"], a.get("ev_bb")) for a in res["actions"]],
                            how=how, rare=any("rarely has" in n for n in notes),
                            settled=any("but it has settled" in n for n in notes), secs=round(time.time() - t, 1))
        start = starts[-1]
    ex = exact_eval(start, st, hero_p, oop, ip, resp)
    j = [i for i, c in enumerate(ex["sol"].ctx.active) if int(c) == ci]
    if j:
        j = j[0]
        st_ = ex["node"].avg_strategy0()
        out["exact_hand"] = {n: (round(float(st_[a, j]), 3), round(float(ex["evs"][a, j]), 2)) for a, n in enumerate(ex["names"])}
    for key in answers:
        rh_full = ex["rh_full"]
        answers[key]["freq"] = range_freq({c: d for c, d in resp[key].items() if rh_full[c] > 0} or resp[key], rh_full)
        answers[key]["given_up"] = ex["gains"][key]
    out.update(answers=answers, exact=dict(expl=ex["expl"], secs=ex["secs"], bettor_share=ex["bettor_share"],
                                           bettor_combos=ex["bettor_combos"], freq=ex["eq_freq"]))
    return out


def summary(recs):
    recs = [r for r in recs if "answers" in r]
    rare = [r for r in recs if r["answers"]["rules"]["rare"]]
    print(f"\n{len(recs)} spots, {len(rare)} with a rare river line in the solve")
    for label, sub in (("all", recs), ("rare", rare)):
        if not sub:
            continue
        for key in ("rules", "kept"):
            g = np.array([r["answers"][key]["given_up"] for r in sub])
            agree = np.mean([family(r["answers"][key]["rec"]) == family(max(r["exact_hand"], key=lambda n: r["exact_hand"][n][0]))
                             for r in sub if r.get("exact_hand")])
            fold = np.array([r["answers"][key]["freq"].get("fold", 0) - r["exact"]["freq"].get("fold", 0) for r in sub])
            print(f"  {label:5s} {key:6s} given up: mean {g.mean():6.2f}% median {np.median(g):5.2f}% max {g.max():6.2f}%"
                  f"  over 1%: {int((g > 1).sum()):2d}/{len(sub)}  fold freq vs exact {fold.mean():+.3f}"
                  f"  hand's action = exact's top {agree:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--speed", default="fast")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "rare_lines.jsonl"))
    ap.add_argument("--summary", action="store_true", help="summarize --out only")
    ap.add_argument("--only", default="", help="spot indices to run, comma separated (the rest are drawn and skipped)")
    ap.add_argument("--keep-combos", type=float, default=None, help="RARE_KEEP_COMBOS for the run")
    a = ap.parse_args()
    if a.summary:
        summary([json.loads(l) for l in open(a.out)])
        return
    rng = random.Random(a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.keep_combos is not None:
        AD.RARE_KEEP_COMBOS = a.keep_combos
    only = {int(x) for x in a.only.split(",") if x.strip()}
    recs = []
    for i in range(a.spots):
        spot = make_spot(rng, i)
        if only and i not in only:
            continue
        t = time.time()
        try:
            rec = run_spot(spot, a.speed, np.random.default_rng(a.seed * 1000 + i))
        except Exception as e:   # (a spot the tree can't take: say so and go on)
            rec = {"skipped": f"{type(e).__name__}: {e}"}
        rec.update(i=i, seed=a.seed, seconds=round(time.time() - t, 1))
        recs.append(rec)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if "answers" in rec:
            ans = rec["answers"]
            print(f"{i:3d} {rec['name']:18s} {rec['stack']:3d}bb {rec['kind']:11s} {str(rec['big']):6s} {rec['board']} "
                  f"{rec['hole']}  bettor {rec['exact']['bettor_share']*100:4.1f}%  rare={ans['rules']['rare']!s:5s} "
                  f"given up rules {ans['rules']['given_up']:6.2f}% kept {ans['kept']['given_up']:6.2f}%  "
                  f"hand {ans['rules']['rec']} / {ans['kept']['rec']} exact {rec.get('exact_hand')}", flush=True)
        else:
            print(f"{i:3d} skipped: {rec['skipped']}", flush=True)
    summary(recs)


if __name__ == "__main__":
    main()
