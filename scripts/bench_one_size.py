"""The one-size view (state "one_size"; Advisor._one_size_view, README "One bet size") measured on standard spots: at
each decision, the street solved from there with every candidate size alone and with all of them to mix freely (the
full menu), in the page's Fast tree with threads (gto/static/worker.js: the turn and river inside at 33/75/125%, the
flop's bucketed model), natively. Each solve stops once settled (--settled, % of the pot, by street) or at --budget.
One JSON line a spot to --out (a spot already there is skipped: resumable), then a table: each size's EV given up
against the full menu (% of the pot), the best size, the full menu's own bet mix.

  GTO_THREADS=1 python scripts/bench_one_size.py --out reports/one_size.jsonl [--spots t-k72-3,r-k72-3-5]
      [--settled turn=0.05,river=0.02,flop=0.1] [--budget 120] [--no-cut]

The spots are scripts/turn_river_reference.py's (BTN vs BB single-raised, CO vs BTN 3bet pots) plus two flops; the
player deciding is the one who can bet: the in-position bettor after a check, or the BB first to act after the street
before was checked through. The raise spots ("fr-", "tr-", "rr-": a flop check-raise, turn and river raises) are the
player facing a bet, whose sizes are raises (2.5x to 5x the bet).
"""
import argparse, json, os, sys, time
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gto.advisor import Advisor          # noqa: E402
from gto.solver import Config            # noqa: E402
from gto.flop_library import default_library   # noqa: E402
from gto import core                     # noqa: E402

F = lambda p: {"pos": p, "action": "fold"}
C = lambda p: {"pos": p, "action": "call"}
X = lambda p: {"pos": p, "action": "check"}
R = lambda p, to: {"pos": p, "action": "raise", "to": to}
B = lambda p, amt: {"pos": p, "action": "bet", "amount": amt}
SRP = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), C("BB")]                     # pot 5.5
THREEBET = [F("UTG"), F("HJ"), R("CO", 2.5), R("BTN", 7.5), F("SB"), F("BB"), C("CO")]  # pot 16.5, CO out of position
BC = [X("BB"), B("BTN", 1.8), C("BB")]           # a third-pot c-bet called (turn pot 9.1)
BC3 = [X("CO"), B("BTN", 5.5), C("CO")]          # 3bet pot (turn pot 27.5)
XX = [X("BB"), X("BTN")]
TBC = [X("BB"), B("BTN", 6.8), C("BB")]          # three quarters called (river pot 22.7)
TXX = [X("BB"), X("BTN")]


def st(pre, board, flop=(), turn=(), river=(), what=""):
    return {"players": 6, "stack": 100, "preflop": pre, "board": board, "flop": list(flop), "turn": list(turn),
            "river": list(river), "what": what}


# name: the request, asked as Study asks it (no hero or cards: the range's view for whoever acts next)
SPOTS = {
    "f-k72": st(SRP, "Ks7d2c", [X("BB")], what="flop, BTN after the BB checks (single-raised)"),
    "f-t94-3b": st(THREEBET, "Th9h4c", [X("CO")], what="flop, BTN after the CO checks (3bet pot, two-tone)"),
    "t-k72-3": st(SRP, "Ks7d2c3h", BC, [X("BB")], what="turn, c-bet called, brick: BTN after a check"),
    "t-k72s-4s": st(SRP, "Ks7s2d4s", BC, [X("BB")], what="turn, c-bet called, the flush completes"),
    "t-q83-a": st(SRP, "Qs8d3cAh", BC, [X("BB")], what="turn, c-bet called, an overcard"),
    "t-q83x-5": st(SRP, "Qs8d3c5h", XX, what="turn, flop checked through: the BB first to act"),
    "t-3b-k72-4": st(THREEBET, "Kd7c2h4s", BC3, [X("CO")], what="turn, 3bet pot, c-bet called, brick"),
    "t-3b-a85-8": st(THREEBET, "As8d5c8h", BC3, [X("CO")], what="turn, 3bet pot, the board pairs"),
    "r-k72-3-5": st(SRP, "Ks7d2c3h5c", BC, TBC, [X("BB")], what="river, bet bet: polar BTN after a check"),
    "r-k72s-4-9s": st(SRP, "Ks7s2d4h9s", BC, TBC, [X("BB")], what="river, bet bet, the flush completes"),
    "r-k72-3x-9": st(SRP, "Ks7d2c3h9d", BC, TXX, what="river, turn checked through: the BB first to act"),
    "r-q83x-5-j": st(SRP, "Qs8d3c5hJc", XX, TXX, what="river, checked to the river: the BB first to act"),
    # facing a bet: raise sizes
    "fr-k72": st(SRP, "Ks7d2c", [X("BB"), B("BTN", 1.8)], what="flop, the BB facing a third-pot c-bet: check-raise"),
    "fr-t94-3b": st(THREEBET, "Th9h4c", [X("CO"), B("BTN", 5.5)], what="flop, 3bet pot, the CO facing a third-pot c-bet"),
    "tr-k72-3": st(SRP, "Ks7d2c3h", BC, [X("BB"), B("BTN", 6.8)], what="turn, c-bet called, brick: the BB facing 75%"),
    "tr-q83-a": st(SRP, "Qs8d3cAh", BC, [X("BB"), B("BTN", 6.8)], what="turn, c-bet called, an overcard: the BB facing 75%"),
    "rr-k72-3-5": st(SRP, "Ks7d2c3h5c", BC, TBC, [X("BB"), B("BTN", 17.0)],
                     what="river, bet bet bet: the BB facing 75%"),
    "rr-q83x-5-j": st(SRP, "Qs8d3c5hJc", XX, TXX, [B("BB", 4.1)],
                      what="river, checked to the river: BTN facing the BB's 75% lead"),
}


def hero_of(s):
    street = {6: "flop", 8: "turn", 10: "river"}[len(s["board"])]
    line = s[street]
    pair = ("CO", "BTN") if s["preflop"] is THREEBET else ("BB", "BTN")
    return pair[0] if not line else next(p for p in pair if p != line[-1]["pos"])


def page_advisor() -> Advisor:
    """The page's Fast advisor with threads (gto/static/worker.js, THREADED)."""
    adv = Advisor(Config(n_turns=3, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4, river_buckets=32,
                         turn_buckets=True, bucket_river_step=3, river_classes=True, river_strength=32,
                         bet_sizes={"flop": [0.33, 0.75], "turn": [0.33, 0.75, 1.25], "river": [0.33, 0.75, 1.25]},
                         raise_sizes={"flop": [0.5], "turn": [0.75], "river": [0.75]}))
    adv.root_sizes = {"river": {"bet": [0.33, 0.75, 1.25, 2.5]}}
    adv.spare_prior = True
    adv.hu_settled = adv.expl_target = 0.2
    adv.library = default_library()
    return adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spots", default=",".join(SPOTS))
    ap.add_argument("--settled", default="flop=0.1,turn=0.05,river=0.02")
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--table", action="store_true", help="only print the table of --out")
    ap.add_argument("--no-cut", action="store_true", help="every size solved until settled (Advisor.one_size_cut off: "
                                                          "no size stops early once clearly behind the best)")
    a = ap.parse_args()
    done = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            d = json.loads(line)
            done[d["spot"]] = d
    if not a.table:
        adv = page_advisor()
        adv.one_size_cut = not a.no_cut
        adv.one_size_settled.update({k: float(v) for k, v in (x.split("=") for x in a.settled.split(","))})
        for name in a.spots.split(","):
            if name in done:
                continue
            s = {k: v for k, v in SPOTS[name].items() if k != "what"}
            s.update(study=True, budget=a.budget, prior_budget=12)   # (Study's request: the actor's whole range)
            t0 = time.time()
            r = adv.advise(dict(s, one_size="all"))
            o = r.get("one_size")
            if not o:
                print(name, r, flush=True)
                continue
            d = {"spot": name, "what": SPOTS[name]["what"], "hero": o["actor"], "threads": core.threads(),
                 "kind": o.get("kind", "bet"), "raises": o.get("raises"),
                 "seconds": round(time.time() - t0, 1), "pot_bb": o["pot_bb"], "stack_bb": o["stack_bb"],
                 "best": o["best"], "tied": o["tied"], "target_pct": o["target_pct"],
                 "full": {k: o["full"][k] for k in ("actions", "freq", "ev_bb", "iterations", "exploitability_pct",
                                                    "seconds")},
                 "rows": [{k: x[k] for k in ("size", "actions", "freq", "ev_bb", "loss_pct", "iterations",
                                             "exploitability_pct", "seconds")} | ({"cut": True} if x.get("cut") else {})
                          for x in o["rows"]]}
            with open(a.out, "a") as f:
                f.write(json.dumps(d) + "\n")
            done[name] = d
            print(name, d["seconds"], "s, best", d["best"], flush=True)
    for name, d in done.items():
        full = d["full"]
        mix = ", ".join(f"{n} {f * 100:.0f}" for n, f in zip(full["actions"], full["freq"]) if f >= 0.005)
        print(f"\n{name}: {d['what']} ({d['hero']}, pot {d['pot_bb']} bb, {d['stack_bb']} behind)")
        print(f"  full menu: {mix}  [{full['iterations']} it, {full['exploitability_pct']}%, {full['seconds']} s]")
        for x in d["rows"]:
            bet = sum(f for n, f in zip(x["actions"], x["freq"]) if n not in ("check", "fold", "call"))
            mark = " best" if x["size"] == d["best"] else " tied" if x["size"] in d["tied"] else ""
            to = f" (to {d['raises'][x['size']]['to']:g})" if d.get("raises") and x["size"] in d["raises"] else ""
            print(f"  {x['size']:>7}{to}: gives up {x['loss_pct']:6.3f}% of the pot, "
                  f"{'raises' if d.get('kind') == 'raise' else 'bets'} {bet * 100:3.0f}%  "
                  f"[{x['iterations']} it, {x['exploitability_pct']}%, {x['seconds']} s]{mark}")


if __name__ == "__main__":
    main()
