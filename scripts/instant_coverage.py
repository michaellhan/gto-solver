"""How many of a 6-max session's decisions are answered from precomputed data, street by street.

Every seat plays the preflop charts (scripts/flop_library_coverage.py's enumeration: each line's probability from
the chart frequencies, card removal between players ignored). A decision is any player's (the page asks for each
of yours, and Practice for each opponent's move; with the seats rotating, both have the same mix). By street:

* preflop: every decision the enumeration reaches (the charts answer all of them);
* flop, heads-up: the decisions of a flop by walking each stored entry's range frequencies (as
  flop_library_coverage.node_profile), answered where the spot, the flop and the node are stored; the flops a
  spot doesn't store (and the spots the library doesn't hold) are live solves with the spot's average profile, a
  library spot's answered approximately at once meanwhile ("+approx": read off its stored flops most like it,
  gto/flop_approx.py, at the nodes the spot's entries store), and a pot the library holds no spot for off the
  nearest spot's (flop_approx.near_spot: its stored flops, else those most like it);
* flop, 3-way: the first decisions the 3-way library stores (first to act, checked to once and twice), by walking
  its entries (at 8- and 9-max a pot read as a spot's, mw_library.MWLibrary.borrow); anything facing a bet is live
  (counted as two decisions a bet), and 4+ way flops are live; "+approx": a spot's flop not stored at those
  decisions off its stored flops most like it (MW_NEAR_HELD of them), a decision again after a bet off the player's
  earlier decision's live solve, and a 3-way turn's first decision off the flop's live solve (advisor.
  _study_mw_near, _study_mw_prev; Advisor.instant's answer for your hand); a pot neither held nor borrowed read off
  the nearest spot (MWLibrary.stand_in: its stored flops' mixes, the others' off those most like them);
* turn and river: how often the flop's betting reaches the turn with chips behind comes from the same walks (a
  call of an all-in reaches no further decision); the decisions a turn has, and how often it reaches the river,
  from --turns native Fast turn solves of the commonest spots (their turn and river streets walked by range
  frequencies; cached in --turn-json). A flop with a turn file (flop_library.encode_turns: the index's "turns", or
  assumed) answers the turn's first decision and the in-position player's after a check (the profile's
  turn_first) on the flop lines it stores (TURN_LINE of the pairs or more, by the entry's walk), in a borrowed pot
  for the hands the spot's ranges hold; a spot with none of its own, "+approx", off the nearest spot's that stores
  that flop's (advisor._near_turn_spots; NEAR_TURN_STORED of the decisions: the hands its file holds); nothing
  precomputed answers a river. "+approx" also: a flop not read off a
  library entry is solved live, and that solve answers every turn decision approximately at once (advisor.
  instant_from_prev: its own turn, by card class and hand buckets), as the turn's live solve answers every river
  decision (the river's own solve takes ~0.1-0.3 s); --why splits the turn decisions answered neither way (after a
  library flop with no turn file, its own or a near spot's; past a turn file's decisions: the turn's own solve, run
  from its first decision on; 3-way and more, solved live).

--assume projects a build (';' separated, a spot id of flop_library.all_spots() or mw_library.SPOTS, or a preflop
line as printed): id=n (n representative flops), id=n,turns (and turn files on the flops it adds), id=turns (turn
files on every flop it stores: the spot solved again). --candidate NAME ASSUME, repeated: each one's instant
decisions per 100 hands over the --assume'd baseline.

    python scripts/instant_coverage.py                      # uses the cached turn profile if present
    python scripts/instant_coverage.py --turns 12           # solves 12 turns natively first (~1 min of CPU)
    python scripts/instant_coverage.py --candidate flatted "srp_hj_co=1755,turns;srp_btn_sb=1755,turns" \
        --candidate btn_bb_turns "srp_btn_bb=turns"
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import mw_library as mwl  # noqa: E402
from gto import preflop as pf  # noqa: E402
from gto.advisor import preflop_strategy, preflop_ranges, replay_hand, _default_size  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flop_library_coverage import _line_str, RERAISE, rep_share  # noqa: E402

AGGR = ("bet", "raise", "all-in")
# the share of a turn's stored decisions (the first, and in position after a check) a near spot's turn file holds the
# hand's combo for (advisor._near_turn_spots; scripts/bench_turn_near.py's "unseen", single-raised pots: 3-15% of the
# reach first to act, 8% on average, 13-21% in position after a check, 16%; weighted 1 : 0.7)
NEAR_TURN_STORED = 0.89
# the share of a 3-way library spot's flops the library doesn't store that its stored flops answer at once
# (advisor._study_mw_near; scripts/bench_mw_approx.py: 16 of 740 stored flops left out in turn had none near enough)
MW_NEAR_HELD = 0.98


def enumerate_preflop(players=6, stack=100.0, min_prob=2e-5, ante=0.0, straddle=False):
    """(preflop decisions per hand, [(prob, line, setup, n_live)] for lines that see a flop). setup's "aggr": who
    raised last preflop, as an index into its "order" (None: nobody, or a player since folded)."""
    positions = pf.at_depth(pf.Table(pf.SEATS[players], ante, straddle, None), stack)
    out, dec = [], [0.0]

    def rec(acts, prob):
        st = replay_hand(positions, stack, {"preflop": acts}, 1)[0]
        if st.closed or st.to_act is None:
            live = [p for p in st.order if p in st.active]
            if len(live) >= 2:
                ranges = preflop_ranges(acts, positions, stack)
                order = [p for p in pf.postflop_order(positions) if p in live]
                eff = min(stack - st.inv_start[p] - st.bets[p] for p in live)
                pot = st.pot_start + sum(st.bets.values())
                allin = any(stack - st.inv_start[p] - st.bets[p] <= 1e-6 for p in live)
                last = [x["pos"] for x in acts if x["action"] in ("raise", "all-in")]
                aggr = order.index(last[-1]) if last and last[-1] in order else None
                out.append((prob, _line_str(acts), {"order": order, "ranges": [ranges[p] for p in order],
                                                    "pot": round(float(pot), 2), "stack": round(float(eff), 2),
                                                    "allin": allin, "aggr": aggr}, len(live)))
            return
        p = st.to_act
        dec[0] += prob
        strat = preflop_strategy(p, acts, positions, stack)
        r = preflop_ranges(acts, positions, stack)[p]
        tot = float(r.sum())
        for k, v in strat.items():
            q = float((r * v).sum()) / max(tot, 1e-12)
            if q * prob < min_prob:
                continue
            if k == "fold":
                a = {"pos": p, "action": "fold"}
            elif k in ("call", "check"):
                a = {"pos": p, "action": "check" if k == "check" or st.cur_bet <= st.bets[p] else "call"}
            else:
                to = round(_default_size(st, p, True), 2)
                a = {"pos": p, "action": "all-in"} if to >= stack - 1e-6 else {"pos": p, "action": "raise", "to": to}
            rec(acts + [a], prob * q)
    rec([], 1.0)
    return dec[0], out


def hu_entry_profile(e):
    """(decisions, stored decisions, P(the turn with chips behind), P(the turn on a stored line a turn file holds:
    TURN_LINE of the pairs or more)) a flop, from an entry's range frequencies."""
    nodes = {tuple(n["path"]): n for n in e["nodes"]}
    d = s = t = tl = 0.0

    def turn(r):
        nonlocal t, tl
        t += r
        tl += r if r >= fl.TURN_LINE else 0.0

    def visit(path, reach):
        nonlocal d, s, t
        d += reach
        n = nodes.get(path)
        if n is None:          # (a node the entry doesn't store: a re-raise line, about half of it called)
            t += reach * 0.5
            return
        s += reach
        for act, f in n["freqs"].items():
            if f <= 0 or act == "fold":
                continue
            if act == "call":
                if not (path and path[-1] == "all-in"):
                    turn(reach * f)
            elif act == "check":
                if path:
                    turn(reach * f)          # checked through
                else:
                    visit(path + (act,), reach * f)
            elif (act.startswith("raise") or act.startswith("all")) and path + (act,) not in nodes:
                d += reach * f * RERAISE   # facing the raise (and a re-raise now and then), live
                t += reach * f * 0.5
            else:
                visit(path + (act,), reach * f)
    visit((), 1.0)
    return d, s, t, tl


def mw_entry_profile(e):
    """(decisions, stored decisions, P(the turn), decisions again) a 3-way flop from a 3-way entry (stored: the
    checks to a bet; a bet is faced by the two others: two live decisions, the turn reached ~60% of the time; "again":
    those of a player who acted on the flop before the bet, the first to act facing the second's, both facing the
    third's, read at once off that player's earlier decision's live solve: advisor._study_mw_prev)."""
    nodes = {tuple(n["path"]): n for n in e["nodes"]}
    d = s = t = again = 0.0
    reach = 1.0
    for k, path in enumerate(mwl.STORED):
        n = nodes.get(tuple(path))
        if n is None:
            break
        d += reach
        s += reach
        bet = sum(f for a, f in n["freqs"].items() if a.startswith(AGGR))
        d += reach * bet * 2 * RERAISE
        again += reach * bet * k * RERAISE
        t += reach * bet * 0.6
        reach *= n["freqs"].get("check", 0.0)
    t += reach                                # checked through
    return d, s, t, again


def turn_profile(n_turns, seed=1):
    """Native Fast turn solves of the commonest heads-up spots, the flop's line drawn from the library entry:
    {'turn_decisions', 'p_river', 'river_decisions'} averaged over the turns (each walked by range frequencies)."""
    from gto import native
    from gto.solver import Decision, Chance
    rng = random.Random(seed)
    lib = fl.default_library()
    spots = {s["id"]: s for s in lib.index()["spots"]}
    picks = ["srp_btn_bb", "srp_co_bb", "srp_sb_bb", "3bp_bb_btn", "srp_hj_bb", "srp_btn_sb"]
    seats = {"srp_btn_bb": ("BTN", "BB", [("BTN", 2.5)]), "srp_co_bb": ("CO", "BB", [("CO", 2.5)]),
             "srp_sb_bb": ("SB", "BB", [("SB", 3.0)]), "srp_hj_bb": ("HJ", "BB", [("HJ", 2.5)]),
             "srp_btn_sb": ("BTN", "SB", [("BTN", 2.5)])}
    rows = []
    for i in range(n_turns):
        sid = picks[i % len(picks)]
        sp = spots.get(sid)
        if sp is None:
            continue
        flop = rng.choice(sorted(sp["flops"]))
        e = lib.entry(sp, flop)
        board = fl.parse_cards(flop)
        deck = [c for c in range(52) if c not in board]
        turn = rng.choice(deck)
        # the flop: check-check or check, bet 33%, call (whichever the entry takes more), the turn first to act
        nodes = {tuple(n["path"]): n for n in e["nodes"]}
        xb = nodes.get(("check",), {"freqs": {}})["freqs"]
        line = "xx" if xb.get("check", 0) >= xb.get("bet 33%", 0) else "xbc"
        line = rng.choice(["xx", "xbc"]) if rng.random() < 0.5 else line
        state = spot_state(sid, sp, flop + fl.cards_str([turn]), line)
        if state is None:
            continue
        # the hero's hand: a combo of its range off the board (any: the turn's whole tree is what's walked)
        from gto.cards import ALL_COMBOS
        rest = [c for c in deck if c != turn]
        state["hole"] = fl.cards_str(sorted(rng.sample(rest, 2), reverse=True))
        adv = native.advisor("fast")
        try:
            adv.advise(state)
        except Exception as ex:          # (a spot the replay doesn't take)
            print("skip", sid, flop, ex, file=sys.stderr)
            continue
        sols = [s for k, s in adv._cache.items() if len(k[2]) == 4]
        if not sols:
            continue
        sol = sols[-1]
        r = walk_turn(sol, Decision, Chance)
        r.update(spot=sid, board=state["board"], line=line, iters=sol.iters)
        print(json.dumps(r), file=sys.stderr)
        rows.append(r)
    keys = ("turn_decisions", "turn_first", "p_river", "river_decisions")
    return {**{k: float(np.mean([r[k] for r in rows])) for k in keys}, "n": len(rows), "rows": rows}


def spot_state(sid, sp, board, line):
    """The request state of the turn's first decision in spot sid after the flop line ('xx' or 'xbc')."""
    seats = {"srp_btn_bb": "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:f,BB:c", "srp_co_bb": "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c",
             "srp_sb_bb": "UTG:f,HJ:f,CO:f,BTN:f,SB:r3,BB:c", "srp_hj_bb": "UTG:f,HJ:r2.5,CO:f,BTN:f,SB:f,BB:c",
             "srp_btn_sb": "UTG:f,HJ:f,CO:f,BTN:r2.5,SB:c,BB:f", "3bp_bb_btn": sp.get("line", "")}
    pre = []
    for tok in (sp.get("line") or seats.get(sid, "")).split(","):
        pos, a = tok.split(":")
        if a == "f":
            pre.append({"pos": pos, "action": "fold"})
        elif a == "c":
            pre.append({"pos": pos, "action": "call"})
        elif a == "x":
            pre.append({"pos": pos, "action": "check"})
        else:
            pre.append({"pos": pos, "action": "raise", "to": float(a[1:])})
    oop, ip = sp["seats"]
    pot = float(sp["pot"])
    flop = [{"pos": oop, "action": "check"}]
    flop += [{"pos": ip, "action": "check"}] if line == "xx" else \
        [{"pos": ip, "action": "bet", "amount": round(pot * 0.33, 2)}, {"pos": oop, "action": "call"}]
    return {"players": 6, "stack": 100, "hero": oop, "hole": "", "board": board, "preflop": pre, "flop": flop,
            "turn": [], "river": [], "speed": "fast"}


def walk_turn(sol, Decision, Chance):
    """Turn and river decisions per turn reached (turn_first: the first and in position after a check, which a turn
    file holds) and P(the river), by range frequencies (river classes weighed alike, card removal ignored)."""
    r0, r1 = sol.r0.astype(np.float64), sol.r1.astype(np.float64)
    out = {"turn_decisions": 0.0, "turn_first": 0.0, "p_river": 0.0, "river_decisions": 0.0}

    def strat(n, b):
        return n.avg_strategy()[b].astype(np.float64)

    def visit(n, a0, a1, b, river, path=()):
        if isinstance(n, Chance):
            if river:
                return
            K = len(n.child.boards) if hasattr(n.child, "boards") else 1
            out["p_river"] += reach(a0, a1)
            for k in range(K):
                visit_river(n.child, a0, a1, k, 1.0 / K)
            return
        if not isinstance(n, Decision):
            return
        out["turn_decisions"] += reach(a0, a1)
        if path in ((), ("check",)):
            out["turn_first"] += reach(a0, a1)
        s = strat(n, b)
        for k, c in enumerate(n.children):
            if n.player == 0:
                visit(c, a0 * s[k], a1, b, river, path + (n.names[k],))
            else:
                visit(c, a0, a1 * s[k], b, river, path + (n.names[k],))

    def visit_river(n, a0, a1, b, w):
        if not isinstance(n, Decision):
            return
        out["river_decisions"] += w * reach(a0, a1)
        s = strat(n, b)
        for k, c in enumerate(n.children):
            if n.player == 0:
                visit_river(c, a0 * s[k], a1, b, w)
            else:
                visit_river(c, a0, a1 * s[k], b, w)

    def reach(a0, a1):
        return float(a0.sum() / r0.sum() * a1.sum() / r1.sum())
    visit(sol.root, r0, r1, 0, False)
    pr = max(out["p_river"], 1e-9)
    out["river_decisions"] /= pr
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--stack", type=float, default=100.0)
    ap.add_argument("--ante", type=float, default=0.0, help="each player's ante in bb (pf_formats: 0.125)")
    ap.add_argument("--straddle", action="store_true", help="UTG straddles 2bb")
    ap.add_argument("--min-prob", type=float, default=2e-5)
    ap.add_argument("--why", action="store_true", help="why the heads-up flops nothing answers at once aren't")
    ap.add_argument("--turns", type=int, default=0, help="native Fast turn solves to measure the turn and river with")
    ap.add_argument("--turn-json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".work",
                                                         "turn_profile.json"))
    ap.add_argument("--assume", default="", help="the baseline's builds (see above)")
    ap.add_argument("--candidate", nargs=2, action="append", default=[], metavar=("NAME", "ASSUME"),
                    help="a build to weigh against the baseline (repeatable)")
    a = ap.parse_args(argv)
    if a.turns:
        tp = turn_profile(a.turns)
        os.makedirs(os.path.dirname(a.turn_json), exist_ok=True)
        with open(a.turn_json, "w") as f:
            json.dump(tp, f, indent=1)
    elif os.path.exists(a.turn_json):
        with open(a.turn_json) as f:
            tp = json.load(f)
    else:
        tp = None
        print("(no turn profile: --turns N measures it; the turn and river use the flop's profile meanwhile)")

    pre_dec, lines = enumerate_preflop(a.players, a.stack, a.min_prob, a.ante, a.straddle)
    lib = fl.FlopLibrary.from_dir()
    mlib = mwl.MWLibrary.from_dir() if hasattr(mwl.MWLibrary, "from_dir") else mwl.default_library()
    weights = fl.all_flops()

    # per heads-up spot: (decisions, stored, P(turn), P(turn on a line a turn file holds)) averaged over its stored
    # flops by flop weight, the share of the flops it stores, and of those with turn files
    def share(names):
        return sum(weights[fl.canonical(fl.parse_cards(n))[0]] for n in names) / 22100.0
    hu_prof = {}
    near_memo = {}

    def near_share(names):
        """The share of the flops not in `names` (by weight) that some stored flop is near enough to answer
        approximately (gto/flop_approx.py: the same pattern within MAX_DIST)."""
        key = tuple(sorted(names))
        if key not in near_memo:
            from gto import flop_approx as fa
            tw = nw = 0.0
            for f, w in weights.items():
                if fl.cards_str(f) in names:
                    continue
                tw += w
                nb = fa.neighbours(key, list(f), 1)
                nw += w if nb and nb[0][2] <= fa.MAX_DIST else 0.0
            near_memo[key] = nw / tw if tw else 0.0
        return near_memo[key]
    for s in lib.index().get("spots", []):
        acc = np.zeros(5)
        for name in s.get("flops", {}):
            e = lib.entry(s, name)
            if e is None:
                continue
            w = weights[fl.canonical(fl.parse_cards(name))[0]]
            acc += w * np.array([*hu_entry_profile(e), 1.0])
        if acc[4] > 0:
            turns = s.get("turns")
            hu_prof[s["id"]] = (acc[:4] / acc[4], share(s["flops"]), share(s["flops"] if turns is True else turns or ()),
                                near_share(s["flops"]) if len(s["flops"]) < len(weights) else 0.0)
    kind_avg = {k: np.mean([v[0] for i, v in hu_prof.items() if i.startswith(k)], axis=0) for k in ("srp", "3bp")}
    # per heads-up spot that stores no turn files: the share of the flops (by weight) it stores whose turns the
    # nearest spot that does stores (advisor._near_turn_spots: instant_from_store reads them, approximately)
    from gto.advisor import _near_turn_spots
    near_turn = {}
    for s in lib.index().get("spots", []):
        if s["id"] in hu_prof and not s.get("turns"):
            src = set()
            for x in _near_turn_spots(lib, s["id"]):
                src |= set(x["flops"]) if x.get("turns") is True else set(x.get("turns") or ())
            near_turn[s["id"]] = share(set(s["flops"]) & src)
    mw_prof = {}
    for s in mlib.index().get("spots", []):
        acc = np.zeros(5)
        for name in s.get("flops", {}):
            e = mlib.entry(s, name) if hasattr(mlib, "entry") else None
            if e is None:
                continue
            w = weights[fl.canonical(fl.parse_cards(name))[0]]
            acc += w * np.array([*mw_entry_profile(e), 1.0])
        if acc[4] > 0:
            mw_prof[s["id"]] = (acc[:4] / acc[4], share(s["flops"]))
    mw_avg = np.mean([v[0] for v in mw_prof.values()], axis=0) if mw_prof else np.array([4.0, 1.5, 0.6, 0.4])
    td = tp["turn_decisions"] if tp else None
    tf = tp.get("turn_first") if tp else None
    if tf is None:
        print("(the turn profile has no turn_first: --turns N measures it; turn files count nothing meanwhile)")
        tf = 0.0

    near_memo2, exact_memo, why_memo = {}, {}, {}
    from gto import flop_approx as fa

    def near_spot(fp0, fp1, s):
        """The library spot flop_approx.near_spot reads a pot it holds no spot for off (its id), or None."""
        key = (fp0, fp1, round(s["pot"], 2), round(s["stack"], 2), s.get("aggr"))
        if key not in near_memo2:
            ns = fa.near_spot(lib, np.asarray(s["ranges"][0]), np.asarray(s["ranges"][1]), s["pot"], s["stack"],
                              aggr=s.get("aggr"), players=a.players)
            near_memo2[key] = ns["id"] if ns is not None and ns["id"] in hu_prof else None
        return near_memo2[key]

    def borrowed(fp0, fp1, s):
        """(the library spot read as this pot's own, flop_approx.near_exact: another table size's, or None; the share
        of the decisions it answers: the hands its ranges hold, the two ranges' cover averaged; the rest are solved)."""
        key = (fp0, fp1, round(s["pot"], 2), round(s["stack"], 2))
        if key not in exact_memo:
            ne = fa.near_exact(lib, np.asarray(s["ranges"][0]), np.asarray(s["ranges"][1]), s["pot"], s["stack"],
                               a.players)
            exact_memo[key] = (ne["spot"], sum(ne["cover"]) / 2) if ne is not None and ne["spot"]["id"] in hu_prof \
                else (None, 1.0)
        return exact_memo[key]

    def held_share(sid, s):
        """The share of a pot's decisions a near read of spot `sid` answers: of the same pot, the hands its ranges hold
        (the two ranges' cover averaged; the rest are solved live); of another pot, every hand (class_fill), but for
        tiers 3 and 4's (flop_approx.near_fill: the hands held, as for the same pot)."""
        sp = next(x for x in lib.index()["spots"] if x["id"] == sid)
        if round(float(sp["pot"]), 2) != round(s["pot"], 2) and \
                fa.near_fill(lib, np.asarray(s["ranges"][0]), np.asarray(s["ranges"][1]), s["pot"], s["stack"],
                             s.get("aggr"), a.players):
            return 1.0
        st = fa._spot_ranges(sp)
        return 1.0 if st is None else (fa.range_cover(np.asarray(s["ranges"][0]), st["full0"]) +
                                       fa.range_cover(np.asarray(s["ranges"][1]), st["full1"])) / 2

    def why_not(line, s):
        """Why nothing reads a heads-up pot: "limped" (nobody raised), else the gate its likeliest spot fails
        (flop_approx.near_candidates: the one with the least range distance among those that got that far)."""
        if s.get("aggr") is None and ":r" not in line:
            return "limped pot"
        key = (fl.range_fp(s["ranges"][0]), fl.range_fp(s["ranges"][1]), round(s["pot"], 2), round(s["stack"], 2),
               s.get("aggr"))
        if key not in why_memo:
            rows = fa.near_candidates(lib, np.asarray(s["ranges"][0]), np.asarray(s["ranges"][1]), s["pot"],
                                      s["stack"], aggr=s.get("aggr"), players=a.players)
            order = {"pot": 0, "spr": 1, "ranges": 2, "cover": 3, "dist": 4}
            best = max(rows, key=lambda r: (order.get(r["why"], -1), -(r["dist"] or 0.0)), default=None)
            why_memo[key] = {"pot": "another pot, no spot with the same preflop raiser",
                             "spr": "stack-to-pot ratio out of range", "ranges": "charts changed",
                             "cover": "range cover too low", "dist": "ranges too far apart"}.get(
                best and best["why"], "no spot")
        return why_memo[key]

    def evaluate(layers):
        """Decisions per hand by street, and those answered at once, with the assumptions `layers` (in order)."""
        tot = defaultdict(float)     # street -> decisions per hand; (street, 'inst') -> answered instantly
        p_flop = defaultdict(float)
        for prob, line, s, n_live in lines:
            if s["allin"]:
                continue                                  # all-in preflop: no decision after it
            if n_live == 2:
                fp0, fp1 = fl.range_fp(s["ranges"][0]), fl.range_fp(s["ranges"][1])
                sp = lib.spot(fp0, fp1, s["pot"], s["stack"])
                lent, held = (None, 1.0) if sp else borrowed(fp0, fp1, s)
                if lent is not None:
                    sp = lent    # (another table size's pot read as a library spot's own: instant, marked borrowed)
                kind = "srp" if sum(":r" in x for x in line.split(",")) <= 1 else "3bp"
                prof, fs, ft, fn = hu_prof.get(sp["id"], (kind_avg[kind], 0.0, 0.0, 0.0)) if sp else (kind_avg[kind], 0.0, 0.0, 0.0)
                fs, ft = _apply(layers, ("hu", fp0, fp1, round(s["pot"], 2), round(s["stack"], 2)), fs, ft)
                d, st, t, tl = prof
                p_flop["hu"] += prob
                tot["flop hu"] += prob * d
                tot["flop hu", "inst"] += prob * st * fs * held
                if lent is not None:
                    tot["flop hu", "borrowed"] += prob * st * fs * held
                if sp:   # (a library spot's flop not stored: an approximate answer at once, gto/flop_approx.py)
                    tot["flop hu", "approx"] += prob * st * (1.0 - fs) * fn * held
                else:    # (a pot the library holds no spot for: the nearest spot's flops, flop_approx.near_spot)
                    ns = near_spot(fp0, fp1, s)
                    if ns is not None:
                        _, nfs, _, nfn = hu_prof[ns]
                        h = held_share(ns, s)
                        tot["flop hu", "approx"] += prob * st * (nfs + (1.0 - nfs) * nfn) * h
                        tot["flop hu", "other"] += prob * st * (nfs + (1.0 - nfs) * nfn) * h
                    elif a.why:
                        tot["why", why_not(line, s)] += prob * st
                td_ = td if tp else d
                entry = fs * held     # (the flop read off a library entry, its own or borrowed: the hands it holds)
                nts = min(near_turn.get(sp["id"], 0.0), max(fs - ft, 0.0)) if sp else 0.0
                tot["p_turn"] += prob * t
                tot["turn"] += prob * t * td_
                tot["turn", "inst"] += prob * tl * ft * tf * held
                # a library flop with no turn file: the nearest spot's stored turn, for the combos it stores
                tot["turn", "approx"] += prob * tl * nts * tf * held * NEAR_TURN_STORED
                tot["turn", "near"] += prob * tl * nts * tf * held * NEAR_TURN_STORED
                # every other flop is solved live, and that solve answers the turn approximately at once (advisor.
                # instant_from_prev: the flop solve's own turn, by card class and hand buckets, one bet size)
                tot["turn", "approx"] += prob * t * (1.0 - entry) * td_
                tot["turn", "prev"] += prob * t * (1.0 - entry) * td_
                if a.why:
                    tot["twhy", "after a library flop with no turn file (its own or a near spot's)"] += \
                        prob * t * (fs - ft - nts) * held * td_
                    tot["twhy", "after a library flop with a turn file (its own or a near spot's), past its decisions "
                        "(the turn's own solve, from the first decision on)"] += \
                        prob * (t * (ft + nts) * td_ - tl * (ft + nts * NEAR_TURN_STORED) * tf) * held
                rv = prob * t * (tp["p_river"] * tp["river_decisions"] if tp else t * d)
                tot["river"] += rv
                tot["river", "approx"] += rv   # (every river: off the turn's live solve, instant_from_prev)
            elif n_live == 3:
                fps = tuple(fl.range_fp(r) for r in s["ranges"])
                sp = mlib.spot(list(fps), s["pot"], s["stack"]) if hasattr(mlib, "spot") else None
                lent = None
                if sp is None and hasattr(mlib, "borrow"):
                    # (another table size's pot read as a 3-way spot's: MWLibrary.borrow, marked borrowed)
                    lent = mlib.borrow([np.asarray(r) for r in s["ranges"]], s["pot"], s["stack"], a.players)
                    sp = lent and lent["spot"]
                stand = None
                if sp is None and hasattr(mlib, "stand_in"):
                    # (a pot neither held nor borrowed read approximately off the nearest spot: MWLibrary.stand_in)
                    stand = mlib.stand_in([np.asarray(r) for r in s["ranges"]], s["pot"], s["stack"])
                    stand = stand and stand["spot"]
                prof, fs = mw_prof.get(sp["id"], (mw_avg, 0.0)) if sp else (mw_avg, 0.0)
                fs, _ = _apply(layers, ("mw", fps, round(s["pot"], 2), round(s["stack"], 2)), fs, 0.0)
                held_v = sp is not None or fs > 0      # (held, borrowed, or a build assumed)
                if stand and not held_v:
                    prof, fs_stand = mw_prof.get(stand["id"], (mw_avg, 0.0))
                else:
                    stand = None
                d, st, t, again = prof
                p_flop["3-way"] += prob
                tot["flop 3-way"] += prob * d
                tot["flop 3-way", "inst"] += prob * st * fs
                if lent:
                    tot["flop 3-way", "borrowed"] += prob * st * fs
                # at once, approximately (advisor._study_mw_near, _study_mw_prev, for your hand: Advisor.instant): a
                # 3-way library spot's flop it doesn't store at the decisions it stores, off the stored flops most like
                # it; on a flop solved live (any but a stored one) a decision again after a bet, off that player's
                # earlier decision's solve, and the turn's first decision off the flop's
                live = 1.0 - fs if held_v else 1.0
                near = prob * st * (1.0 - fs) * MW_NEAR_HELD if held_v else 0.0
                if stand:    # (its stored flops' mixes, the others off its stored flops most like them)
                    near = prob * st * (fs_stand + (1.0 - fs_stand) * MW_NEAR_HELD)
                    tot["flop 3-way", "stand"] += near
                tot["flop 3-way", "approx"] += near + prob * again * live
                tot["flop 3-way", "near"] += near
                # (a 3-way turn: a heads-up turn's decisions times 1.5, the same share reaching the river; its decisions
                # again after a bet as the flop's share of them, off the turn's earlier decision's solve)
                tdec = prob * t * 1.5 * (td if tp else 2.0)
                tmw = prob * t * live + (tdec - prob * t) * again / max(d, 1e-9)
                tot["turn"] += tdec
                tot["turn", "approx"] += tmw
                tot["turn", "mw"] += tmw
                tot["twhy", "3-way or more (solved live)"] += tdec - tmw
                tot["river"] += prob * t * 1.5 * (tp["p_river"] * tp["river_decisions"] if tp else 1.2)
            else:
                p_flop["4+"] += prob
                d = 4.0 * (n_live / 4)
                tot["flop 4+"] += prob * d
                tot["turn"] += prob * 0.6 * d
                tot["twhy", "3-way or more (solved live)"] += prob * 0.6 * d
                tot["river"] += prob * 0.4 * d
        tot["preflop"] = pre_dec
        tot["preflop", "inst"] = pre_dec
        return tot, p_flop

    resolve = _resolver(a.players, a.stack, a.ante, a.straddle)
    base = [parse_assume(a.assume, resolve)] if a.assume else []
    tot, p_flop = evaluate(base)
    game = f"{a.players}-max {a.stack:g}bb" + (f", {a.ante:g}bb antes" if a.ante else "") + \
        (", UTG straddling" if a.straddle else "")
    print(f"{game}, every seat by the charts: a flop seen in "
          f"{sum(p_flop.values()) * 100:.1f}% of hands (heads-up {p_flop['hu'] * 100:.1f}%, 3-way "
          f"{p_flop['3-way'] * 100:.1f}%, 4+ {p_flop['4+'] * 100:.1f}%)" + (f"; assumed: {a.assume}" if a.assume else ""))
    if tp:
        print(f"turn profile ({tp['n']} native Fast turns): {tp['turn_decisions']:.2f} decisions a turn"
              + (f" ({tp['turn_first']:.2f} the first and in position after a check)" if "turn_first" in tp else "")
              + f", the river reached {tp['p_river'] * 100:.0f}% of the time, {tp['river_decisions']:.2f} decisions "
              "a river")
    streets = ["preflop", "flop hu", "flop 3-way", "flop 4+", "turn", "river"]
    all_d = sum(tot[k] for k in streets)
    all_i = sum(tot[k, "inst"] for k in streets)
    print(f"{'street':<11} {'per 100 hands':>13} {'share':>6} {'instant':>8} {'+approx':>8}")
    for k in streets:
        print(f"{k:<11} {tot[k] * 100:13.1f} {tot[k] / all_d * 100:5.1f}% {tot[k, 'inst'] / max(tot[k], 1e-12) * 100:7.1f}%"
              + (f" {(tot[k, 'inst'] + tot[k, 'approx']) / max(tot[k], 1e-12) * 100:7.1f}%" if tot[k, "approx"] else ""))
    post = [k for k in streets if k != "preflop"]
    pd, pi = sum(tot[k] for k in post), sum(tot[k, "inst"] for k in post)
    pa = sum(tot[k, "approx"] for k in post)
    print(f"{'postflop':<11} {pd * 100:13.1f} {pd / all_d * 100:5.1f}% {pi / pd * 100:7.1f}% {(pi + pa) / pd * 100:7.1f}%")
    print(f"{'all':<11} {all_d * 100:13.1f} {100:5.1f}% {all_i / all_d * 100:7.1f}% {(all_i + pa) / all_d * 100:7.1f}%")
    hu = tot["flop hu"]
    print(f"heads-up flop decisions answered by the library: {tot['flop hu', 'inst'] / hu * 100:.1f}% "
          f"(scripts/flop_library_coverage.py's measure); the turn reached from a heads-up flop "
          f"{tot['p_turn'] / p_flop['hu'] * 100:.0f}% of the time")
    print(f"...and approximately at once (a library spot's flop not stored, read off its stored flops most like it: "
          f"gto/flop_approx.py): {tot['flop hu', 'approx'] / hu * 100:.1f}% more, "
          f"{(tot['flop hu', 'inst'] + tot['flop hu', 'approx']) / hu * 100:.1f}% in all; "
          f"{tot['flop hu', 'approx'] * 100:.1f} decisions per 100 hands; {tot['flop hu', 'other'] / hu * 100:.1f} "
          f"points ({tot['flop hu', 'other'] * 100:.1f} decisions) in pots the library holds no spot for, read off "
          "the nearest spot's (flop_approx.near_spot)")
    if tot["flop hu", "borrowed"]:
        print(f"...of the library's: {tot['flop hu', 'borrowed'] / hu * 100:.1f} points in pots of this table size "
              "read as a library spot's own (flop_approx.near_exact: the same pot and stack, the ranges a few combos "
              "apart; marked borrowed)")
    if a.why:
        miss = {k[1]: v for k, v in tot.items() if isinstance(k, tuple) and k[0] == "why"}
        print("heads-up flop decisions nothing answers at once, by why: " +
              "; ".join(f"{k} {v / hu * 100:.1f}%" for k, v in sorted(miss.items(), key=lambda x: -x[1])))
        tmiss = {k[1]: v for k, v in tot.items() if isinstance(k, tuple) and k[0] == "twhy"}
        mw = tot["flop 3-way"]
        print(f"3-way flop decisions: {tot['flop 3-way', 'inst'] / max(mw, 1e-12) * 100:.1f}% the 3-way library's "
              f"({tot['flop 3-way', 'borrowed'] / max(mw, 1e-12) * 100:.1f} borrowed: MWLibrary.borrow), "
              f"{tot['flop 3-way', 'near'] / max(mw, 1e-12) * 100:.1f}% read off its stored flops most like it "
              f"({tot['flop 3-way', 'stand'] / max(mw, 1e-12) * 100:.1f} in pots read off another: "
              f"MWLibrary.stand_in), "
              f"{(tot['flop 3-way', 'approx'] - tot['flop 3-way', 'near']) / max(mw, 1e-12) * 100:.1f}% again after a "
              f"bet off the player's earlier decision's solve; the 3-way turns' {tot['turn', 'mw'] * 100:.1f} a 100 "
              "hands at once (the first decision off the flop's solve, again after a bet off the turn's)")
        print(f"turn decisions: {tot['turn', 'inst'] / max(tot['turn'], 1e-12) * 100:.1f}% a stored turn's, "
              f"{tot['turn', 'near'] / max(tot['turn'], 1e-12) * 100:.1f}% a near spot's stored turn, "
              f"{tot['turn', 'prev'] / max(tot['turn'], 1e-12) * 100:.1f}% read off the flop's live solve at once, "
              f"{tot['turn', 'mw'] / max(tot['turn'], 1e-12) * 100:.1f}% 3-way off the hand's live solves; "
              "not at once: " + "; ".join(f"{k} {v / max(tot['turn'], 1e-12) * 100:.1f}% ({v * 100:.1f} a 100 hands)"
                                          for k, v in sorted(tmiss.items(), key=lambda x: -x[1])))
    if a.candidate:
        print(f"\ninstant decisions per 100 hands each candidate adds (all {all_i * 100:.1f} of {all_d * 100:.1f} now):")
        print(f"{'candidate':<24} {'all':>6} {'flop':>6} {'turn':>6}")
        for name, text in a.candidate:
            ct, _ = evaluate(base + [parse_assume(text, resolve)])
            gain = {k: (ct[k, "inst"] - tot[k, "inst"]) * 100 for k in streets}
            print(f"{name:<24} {sum(gain.values()):6.2f} {gain['flop hu'] + gain['flop 3-way']:6.2f} "
                  f"{gain['turn']:6.2f}")


_REP = {}


def _apply(layers, key, fs, ft):
    """(flop share, turn-file share) of a spot after the assumptions: each layer's {key: (n flops or None, turns)},
    turns 'new' (on the flops the layer adds), 'all' (every flop it stores) or None."""
    for layer in layers:
        x = layer.get(key)
        if x is None:
            continue
        n, turns = x
        if n and n not in _REP:
            _REP[n] = rep_share(n)
        f2 = max(fs, _REP[n]) if n else fs
        ft = ft + (f2 - fs) if turns == "new" else f2 if turns == "all" else ft
        fs = f2
    return fs, ft


def _resolver(players, stack, ante=0.0, straddle=False):
    """name -> the key a game's enumerated lines carry: a spot id (flop_library.all_spots(), in its game, or
    mw_library.SPOTS) or a preflop line, heads-up or 3-way to the flop."""
    hu_ids = {sid: (line, fmt) for sid, _, line, fmt in fl.all_spots()}
    mw_ids = {sid: line for sid, _, line in mwl.SPOTS}

    def resolve(name):
        if name in mw_ids:
            m = mwl.spot_setup(mw_ids[name], players, stack)
            return ("mw", tuple(fl.range_fp(r) for r in m["ranges"]), round(m["pot"], 2), round(m["stack"], 2))
        line, fmt = hu_ids.get(name, (name, None))
        try:
            if fmt is not None:
                s = fl.spot_setup(line, fmt.players, fmt.stack, fmt.ante, fmt.straddle)
            else:
                s = fl.spot_setup(line, players, stack, ante, straddle)
        except Exception:
            m = mwl.spot_setup(line, players, stack, ante, straddle)
            return ("mw", tuple(fl.range_fp(r) for r in m["ranges"]), round(m["pot"], 2), round(m["stack"], 2))
        return ("hu", fl.range_fp(s["full0"]), fl.range_fp(s["full1"]), round(s["pot"], 2), round(s["stack"], 2))
    return resolve


def parse_assume(text, resolve):
    """--assume's ';' separated name=n, name=n,turns or name=turns -> {key: (n or None, 'new' / 'all' / None)}."""
    out = {}
    for x in filter(None, (y.strip() for y in text.split(";"))):
        name, v = x.rsplit("=", 1)
        parts = v.split(",")
        n = int(parts[0]) if parts[0].isdigit() else None
        turns = ("new" if n else "all") if "turns" in parts else None
        out[resolve(name.strip())] = (n, turns)
    return out

if __name__ == "__main__":
    main()
