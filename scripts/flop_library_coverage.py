"""How often a hand's heads-up flop decisions are answered by the precomputed flop library (gto/flop_library.py).

The preflop game is enumerated as the advisor plays it (6-max by default, every seat at the same stack, the page's
default sizes): each seat acts by its chart frequencies (advisor.preflop_strategy, weighted over its range so far:
the probability a line happens, card removal between players ignored), down to lines under --min-prob. A line
that ends heads-up to the flop is a spot (its two ranges, pot and stack, as the library keys them). On the flop:

* flop coverage: the share of the 22100 flops whose canonical flop the spot stores;
* node coverage: the share of the spot's flop decisions (by how often each is reached, players' strategies from
  the stored entries; a spot without entries borrows the averages of its kind, single raised or 3bet) that are
  stored nodes: the root, in position after a check, out of position facing a bet after checking, in position
  facing a lead, and (in entries built since they were stored) facing a raise of one of those bets. A re-raise and
  anything after it is a live solve.

A heads-up flop decision is answered when its spot, flop and node are all stored. Sizes off the tree (more than
15% from 33/75%) also fall back to a live solve; this assumes the tree's sizes.

    python scripts/flop_library_coverage.py                    # the shipped library, 6-max 100bb
    python scripts/flop_library_coverage.py --assume "srp_btn_bb=552;UTG:f,HJ:r2.5,CO:f,BTN:c,SB:f,BB:f=184"
    python scripts/flop_library_coverage.py --fixtures         # and the page's random hands, through the advisor
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import preflop as pf  # noqa: E402
from gto.advisor import preflop_strategy, preflop_ranges, replay_hand, _default_size  # noqa: E402


def _line_str(acts):
    out = []
    for a in acts:
        c = {"fold": "f", "call": "c", "check": "x", "raise": "r", "all-in": "r"}[a["action"]]
        to = a.get("to")
        out.append(f"{a['pos']}:{c}" + (f"{to:g}" if to is not None else ""))
    return ",".join(out)


def preflop_spots(players=6, stack=100.0, min_prob=2e-5):
    """{'hu': [(prob, line, setup)], 'multi': prob, 'none': prob} over the preflop lines (setup: fl.spot_setup's)."""
    positions = pf.at_depth(pf.Table(pf.SEATS[players], 0.0, False, None), stack)
    out = {"hu": [], "multi": 0.0, "none": 0.0}

    def rec(acts, prob):
        st = replay_hand(positions, stack, {"preflop": acts}, 1)[0]
        if st.closed or st.to_act is None:
            live = [p for p in st.order if p in st.active]
            if len(live) < 2:
                out["none"] += prob
            elif len(live) > 2:
                out["multi"] += prob
            else:
                ranges = preflop_ranges(acts, positions, stack)
                oop, ip = [p for p in pf.postflop_order(positions) if p in live]
                eff = min(stack - st.inv_start[oop] - st.bets[oop], stack - st.inv_start[ip] - st.bets[ip])
                pot = st.pot_start + sum(st.bets.values())
                setup = {"oop": oop, "ip": ip, "full0": ranges[oop], "full1": ranges[ip],
                         "pot": round(float(pot), 2), "stack": round(float(eff), 2)}
                out["hu"].append((prob, _line_str(acts), setup))
            return
        p = st.to_act
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
    return out


def node_profile(lib, spot):
    """(decisions per hand, stored decisions per hand) over a spot's stored flops (weighted by flop weight), or
    None without entries. Decisions: the root; in position after a check; facing each bet or lead; facing a
    raise (and a re-raise, roughly); stored ones are those the entry holds."""
    weights = fl.all_flops()
    tot_w = tot_d = tot_s = 0.0
    for name in spot.get("flops", {}):
        e = lib.entry(spot, name)
        if e is None:
            continue
        nodes = {tuple(n["path"]): n for n in e["nodes"]}
        w = weights[fl.canonical(fl.parse_cards(name))[0]]
        d = s = 0.0

        def visit(path, reach):
            nonlocal d, s
            d += reach
            n = nodes.get(path)
            if n is None:
                return
            s += reach
            for act, f in n["freqs"].items():
                if f <= 0 or act in ("fold", "call") or (act == "check" and path):
                    continue
                if (act.startswith("raise") or act.startswith("all")) and path + (act,) not in nodes:
                    d += reach * f * RERAISE   # facing the raise, and a re-raise behind it now and then
                else:
                    visit(path + (act,), reach * f)
        visit((), 1.0)
        tot_w += w
        tot_d += w * d
        tot_s += w * s
    if tot_w == 0:
        return None
    return tot_d / tot_w, tot_s / tot_w


RERAISE = 1.15


def flop_share(spot):
    weights = fl.all_flops()
    return sum(weights[fl.canonical(fl.parse_cards(n))[0]] for n in spot.get("flops", {})) / 22100.0


def rep_share(n):
    w = fl.all_flops()
    return sum(w[k] for k in fl.representative_flops(n)) / 22100.0


def fixtures():
    """(heads-up flop decisions, answered from the library) over the page's random hands, asking the advisor."""
    import json
    from gto.advisor import Advisor, _state_stacks, _norm_actions, STREETS
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data", "page_states.json")) as f:
        spots = json.load(f)
    adv = Advisor(fl.library_config(), time_budget=0.2, prior_budget=0.1)
    adv.library = fl.default_library()
    n = hit = 0
    for sp in spots:
        st, page = sp["state"], sp["page"]
        if page["street"] != "flop" or page["to_act"] != st["hero"]:
            continue
        n_table = st["players"]
        stack, stacks = _state_stacks(st["stack"], st.get("stacks"), pf.SEATS[n_table], 1.0)
        table = pf.Table(pf.SEATS[n_table], st.get("ante", 0.0), st.get("straddle", False), stacks=stacks)
        streets = {k: _norm_actions(st[k], 1.0) for k in STREETS}
        flop = replay_hand(table, stack, streets, 2)[-1]
        if len(flop.players_start) != 2:
            continue
        n += 1
        try:
            r = adv.advise(st)
        except Exception:
            continue
        hit += "precomputed" in r
    return n, hit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--stack", type=float, default=100.0)
    ap.add_argument("--min-prob", type=float, default=2e-5)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--assume", default="", help="project coverage as if spots had n representative flops: "
                    "';' separated id=n (a spot in SPOTS) or line=n (any preflop line, as printed)")
    ap.add_argument("--fixtures", action="store_true", help="also ask the advisor at every heads-up flop decision "
                    "of the page's random hands (tests/data/page_states.json)")
    a = ap.parse_args(argv)
    assume = {}
    for x in filter(None, a.assume.split(";")):
        k, n = x.rsplit("=", 1)
        assume[k] = int(n)
    ids = {sid: line for sid, _, line in fl.SPOTS}
    assume = {ids.get(k, k): n for k, n in assume.items()}
    res = preflop_spots(a.players, a.stack, a.min_prob)
    hu = sorted(res["hu"], key=lambda x: -x[0])
    p_hu = sum(x[0] for x in hu)
    print(f"{a.players}-max {a.stack:g}bb: heads-up to the flop {p_hu * 100:.1f}% of hands, multiway "
          f"{res['multi'] * 100:.1f}%, over preflop {res['none'] * 100:.1f}% ({len(hu)} heads-up lines)")
    lib = fl.FlopLibrary.from_dir()
    prof = {}
    for s in lib.index().get("spots", []):
        pr = node_profile(lib, s)
        if pr:
            prof[s["id"]] = pr
    kind_avg = {}
    for kind in ("srp", "3bp"):
        v = [prof[k] for k in prof if k.startswith(kind)]
        if v:
            kind_avg[kind] = (np.mean([x[0] for x in v]), np.mean([x[1] for x in v]))
    # group lines by spot key
    groups = defaultdict(lambda: {"prob": 0.0, "lines": []})
    for prob, line, s in hu:
        k = (fl.range_fp(s["full0"]), fl.range_fp(s["full1"]), s["pot"], s["stack"])
        g = groups[k]
        g["prob"] += prob
        g["lines"].append((prob, line))
        g["setup"] = s
    rows = []
    tot_dec = tot_ans = 0.0
    for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["prob"]):
        sp = lib.spot(*k)
        n_raises = sum(1 for x in g["lines"][0][1].split(",") if ":r" in x)
        kind = "srp" if n_raises <= 1 else "3bp"
        d, s = prof.get(sp["id"]) if sp and sp["id"] in prof else kind_avg.get(kind, (2.0, 1.8))
        fs = flop_share(sp) if sp else 0.0
        n_assumed = next((assume[l] for _, l in g["lines"] if l in assume), None)
        if n_assumed:
            fs = max(fs, rep_share(n_assumed))
        dec = g["prob"] * d
        ans = g["prob"] * s * fs
        tot_dec += dec
        tot_ans += ans
        rows.append((g, sp, d, s, fs, dec, ans, sp is not None or bool(n_assumed)))   # (stored, or --assume'd)
    print(f"heads-up flop decisions answered by the library: {tot_ans / tot_dec * 100:.1f}%")
    print(f"{'share':>6} {'cum':>6}  {'pot':>5} {'stack':>5} {'lib':<11} {'flops':>5} {'nodes':>5}  line")
    cum = 0.0
    for g, sp, d, s, fs, dec, ans, have in rows[:a.top]:
        cum += dec / tot_dec
        best = max(g["lines"])[1]
        print(f"{dec / tot_dec * 100:5.1f}% {cum * 100:5.1f}%  {g['setup']['pot']:5.1f} {g['setup']['stack']:5.1f} "
              f"{(sp['id'] if sp else 'assumed' if have else '-'):<11} {fs * 100:4.1f}% {s / d * 100:4.0f}%  {best}"
              + (f" (+{len(g['lines']) - 1} lines)" if len(g["lines"]) > 1 else ""))
    # what's missing, by kind
    miss_spot = sum(dec for g, sp, d, s, fs, dec, ans, have in rows if not have) / tot_dec
    miss_flop = sum(dec * s / d * (1 - fs) for g, sp, d, s, fs, dec, ans, have in rows if have) / tot_dec
    miss_node = sum(dec * (1 - s / d) for g, sp, d, s, fs, dec, ans, have in rows if have) / tot_dec
    if a.fixtures:
        n, hit = fixtures()
        print(f"page's random hands: {hit} of {n} heads-up flop decisions from the library")
    print(f"missed: spot not in the library {miss_spot * 100:.1f}%, flop not stored {miss_flop * 100:.1f}%, "
          f"node not stored (raises) {miss_node * 100:.1f}%")


if __name__ == "__main__":
    main()
