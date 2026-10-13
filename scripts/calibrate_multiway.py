"""Calibrate the multiway heuristic's rule constants against the heads-up solver.

The heuristic (Advisor._multiway) can play a heads-up spot too, where the solver gives the right
answer. For heads-up flop spots (CO opens, BB calls; a few boards and lines; hands across hero's
range) we compare the heuristic's EVs and pick with the solver's, both using the same sizes (bet
66%, raise 75% of the pot after calling; with --small also bet 33%, else the heuristic's 33% bet
isn't scored). FOLD_SHADE is set to 1 here: the solver's opponent
defends in balance, and the 0.75 is a deliberate adjustment for real players, not a fit.

Score: the solver-EV the heuristic's pick gives up against the solver's best action (bb), plus
0.2 x the mean absolute EV error across actions.

  python scripts/calibrate_multiway.py           # score the current constants
  python scripts/calibrate_multiway.py --tune    # coordinate search over the constants

  python scripts/calibrate_multiway.py --small   # the solver also bets 33% pot, as the heuristic can

Solver results are cached (--cache), so only the first run pays for the solves (one at a time,
--budget seconds each).
"""
import argparse
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gto.advisor as A
from gto import preflop as pf
from gto.cards import cards
from gto.solver import Config

PRE = [{"pos": "UTG", "action": "fold"}, {"pos": "HJ", "action": "fold"}, {"pos": "CO", "action": "raise", "to": 2.5},
       {"pos": "BTN", "action": "fold"}, {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}]
BOARDS = ["Kc7h2s", "Jh9h4c", "8s8d3c", "Qd6c5s"]
LINES = {   # name: (hero, flop actions so far)
    "BB first": ("BB", []),
    "BB vs bet": ("BB", [{"pos": "BB", "action": "check"}, {"pos": "CO", "action": "bet", "amount": 3.63}]),
    "CO checked to": ("CO", [{"pos": "BB", "action": "check"}]),
}
HANDS = ["AsAd", "KsKd", "QsQh", "JsJc", "TsTd", "9s9d", "7s7d", "5s5d", "3s3h", "AsKs", "AhQd", "AsJs", "KsQd",
         "KsJs", "QsJs", "JsTs", "Ts9s", "9d8d", "8h7h", "7c6c", "6d5d", "As5s", "Ks9d", "Qs8s", "Jd7d", "AdTc"]
PARAMS = {"VALUE_SHARE": [0.35, 0.5, 0.65], "RAISE_SHARE": [0.2, 1 / 3, 0.5], "LEAD_EQ": [0.45, 0.5, 0.6],
          "RAISE_EQ": [0.5, 0.55, 0.65], "NARROW_SOFT": [0.05, 0.15, 0.3],
          "OFF_RULE_MAX": [0.15, 0.25, 0.4, 1.0]}
def sizes(small):
    """The heuristic's sizes: bet 66% (and 33% with --small), raise 75% of the pot after calling."""
    return dict(bet_sizes={s: [0.33, 0.66] if small else [0.66] for s in ("flop", "turn", "river")},
                raise_sizes={s: [0.75] for s in ("flop", "turn", "river")})


def kind(name):
    """Action names matched across the two: bets by size (small under half pot, big above), the
    rest by their first word (the heuristic's raise sizes are rounded)."""
    if name.startswith("bet"):
        return "bet small" if float(name.split()[1].rstrip("%")) < 50 else "bet big"
    return name.split()[0]


def spots():
    for board in BOARDS:
        bset = set(cards(board))
        for line, (hero, flop) in LINES.items():
            for h in HANDS:
                if not set(cards(h)) & bset:
                    yield {"players": 6, "hero": hero, "hole": h, "board": board, "stack": 100,
                           "preflop": PRE, "flop": flop}, line


def solve_all(cache_path, budget, small):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    adv = A.Advisor(Config(**sizes(small)), time_budget=budget, prior_budget=1)
    adv._with_hero = lambda r0, r1, *a: (r0, r1)   # skip hands outside the range instead of re-solving for them
    for st, line in spots():
        key = f"{st['board']}|{line}|{st['hole']}"
        if key in cache:
            continue
        r = adv.advise(st)
        off = any("not solved" in n for n in r.get("notes", []))
        cache[key] = None if r.get("error") or off else {kind(a["action"]): a["ev_bb"] for a in r["actions"]}
        json.dump(cache, open(cache_path, "w"))
        print(f"solved {key}: {cache[key]}", flush=True)
    return cache


def heuristic(args):
    st, params = args
    for k, v in params.items():
        setattr(A, k, v)
    A.FOLD_SHADE = 1.0
    hero, hole, board = st["hero"], tuple(cards(st["hole"])), cards(st["board"])
    positions = pf.SEATS[6]
    rep = A.replay_hand(positions, 100.0, {"preflop": PRE, "flop": st["flop"]}, 2)
    cur = rep[-1]
    ranges = pf.replay_ranges(PRE, positions)
    A._narrow_street(ranges, board, cur, 100.0, hero, hole)
    remaining = [p for p in cur.order if p in cur.active]
    adv = A.Advisor(time_budget=8)
    r = adv._multiway(hero, hole, board, ranges, remaining, cur, 100.0, time.time(), [])
    return {kind(a["action"]): a["ev_bb"] for a in r["actions"]}


def score(cache, params, pool):
    items = [(st, line) for st, line in spots() if cache.get(f"{st['board']}|{line}|{st['hole']}")]
    outs = pool.map(heuristic, [(st, params) for st, _ in items])
    loss = err = n_err = 0.0
    by_line = {}
    for (st, line), h in zip(items, outs):
        s = cache[f"{st['board']}|{line}|{st['hole']}"]
        common = [k for k in h if k in s]
        pick = max(common, key=lambda k: h[k])
        l = max(s.values()) - s[pick]
        loss += l
        by_line[line] = by_line.get(line, 0.0) + l / len([1 for _, ln in items if ln == line])
        for k in common:
            err += abs(h[k] - s[k]); n_err += 1
    loss /= len(items); err /= max(n_err, 1)
    return loss + 0.2 * err, loss, err, by_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--budget", type=float, default=25.0, help="seconds per heads-up solve")
    ap.add_argument("--small", action="store_true", help="the solver also bets 33%% (scores the heuristic's size choice)")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    a.cache = a.cache or os.path.join(tempfile.gettempdir(), "gto_calibration%s.json" % ("_small" if a.small else ""))
    cache = solve_all(a.cache, a.budget, a.small)
    params = {k: getattr(A, k) for k in PARAMS}
    with Pool() as pool:
        best = score(cache, params, pool)
        print(f"current {params}: score {best[0]:.3f} (pick loss {best[1]:.3f} bb, EV error {best[2]:.3f} bb) "
              f"by line {{{', '.join(f'{k}: {v:.2f}' for k, v in best[3].items())}}}", flush=True)
        if not a.tune:
            return
        for sweep in range(2):
            for name, values in PARAMS.items():
                for v in values:
                    if v == params[name]:
                        continue
                    trial = dict(params, **{name: v})
                    sc = score(cache, trial, pool)
                    print(f"  {name}={v:.3f}: score {sc[0]:.3f} (loss {sc[1]:.3f}, err {sc[2]:.3f})", flush=True)
                    if sc[0] < best[0] - 0.01:
                        params, best = trial, sc
                print(f"sweep {sweep + 1}, {name}: keep {params[name]:.3f} (score {best[0]:.3f})", flush=True)
        print(f"best {params}: score {best[0]:.3f} (pick loss {best[1]:.3f} bb, EV error {best[2]:.3f} bb) "
              f"by line {{{', '.join(f'{k}: {v:.2f}' for k, v in best[3].items())}}}")


if __name__ == "__main__":
    main()
