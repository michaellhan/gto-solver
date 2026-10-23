"""The 3-way flop library (gto/mw_library.py) against a live Accurate solve of the same decision: for each flop and
stored node (first to act, checked to once, twice) the native Accurate advisor (library off, its 30 s) solves the
node as it would for a hand there (from the node, the earlier checks narrowing the ranges by the betting rules),
and the acting player's range is compared with the library's: range frequency per action (points), per-hand mix
(half the L1 distance, averaged by reach), how often the two pick the same most frequent action, and per-hand EVs
(mean absolute difference over the actions, % of the pot), plus the live solve's iterations and exploitability.

    python scripts/compare_mw_library.py --spot mw_co_btn_bb --boards Kh7d2c,9s8s5d,Ad6c6h --budget 30
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gto.native as native  # noqa: E402  (before numpy)
import numpy as np  # noqa: E402
from gto import mw_library as ml  # noqa: E402
from gto.cards import cards, cards_str, ALL_COMBOS  # noqa: E402
from gto.cli import parse_actions  # noqa: E402


def compare(spot, board: str, budget: float) -> list:
    sid, label, line = spot
    s = ml.spot_setup(line)
    lib = ml.default_library().solution(s["ranges"], cards(board), s["pot"], s["stack"])
    if lib is None:
        raise SystemExit(f"{sid} {board}: not in the library")
    pre = parse_actions(line)
    out = []
    for path in ml.STORED:
        p = len(path)
        seat = s["seats"][p]
        node = lib.nodes[path]
        R = lib.reach_at(path)
        bset = set(cards(board))
        hole = next(cards_str(ALL_COMBOS[i]) for i in np.argsort(-s["ranges"][p]) if not set(ALL_COMBOS[i]) & bset)
        adv = native.advisor("accurate", library=False, budget=budget)
        r = adv.advise({"hero": seat, "hole": hole, "board": board, "preflop": pre, "stack": 100,
                        "flop": [{"pos": s["seats"][k], "action": "check"} for k in range(p)]})
        live = next(iter(adv._mw_cache.values()))
        assert live.root.player == p and list(live.root.names) == list(node.names), (live.root.names, node.names)
        pos = {c: i for i, c in enumerate(lib.hands.active)}
        li = np.nonzero(live.reach0[p, 0] > 0)[0]
        bi = np.array([pos[c] for c in live.hands.active[li]])
        w = R[p, 0][bi]
        w = w / w.sum()
        sl = live.root.avg_strategy()[0][:, li]
        sb = node.avg_strategy()[0][:, bi]
        el = live.action_evs()[:, li]
        eb = lib.action_evs(node)[:, bi]
        pot = node.pot_now
        freq_l, freq_b = sl @ w, sb @ w
        out.append({"spot": sid, "board": board, "node": "|".join(path) or "root", "seat": seat,
                    "live_iters": live.iters, "live_expl": r["solver"]["exploitability_pct"],
                    "lib_iters": lib.iters, "lib_expl": round(lib.exploitability(), 3),
                    "freq_live": dict(zip(node.names, np.round(freq_l * 100, 1).tolist())),
                    "freq_lib": dict(zip(node.names, np.round(freq_b * 100, 1).tolist())),
                    "freq_pts": round(float(np.abs(freq_l - freq_b).max() * 100), 1),
                    "mix_pts": round(float((np.abs(sl - sb).sum(axis=0) / 2 * w).sum() * 100), 1),
                    "same_top": round(float(((sl.argmax(axis=0) == sb.argmax(axis=0)) * w).sum() * 100), 1),
                    "ev_pct": round(float((np.abs(el - eb).mean(axis=0) * w).sum() / pot * 100), 2)})
        print(json.dumps(out[-1]), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spot", default=ml.SPOTS[0][0])
    ap.add_argument("--boards", default="Kh7d2c")
    ap.add_argument("--budget", type=float, default=30.0)
    a = ap.parse_args(argv)
    spot = next(s for s in ml.SPOTS if s[0] == a.spot)
    rows = [x for b in a.boards.split(",") for x in compare(spot, b, a.budget)]
    for k in ("freq_pts", "mix_pts", "same_top", "ev_pct"):
        print(k, "mean", round(float(np.mean([x[k] for x in rows])), 2), "max", max(x[k] for x in rows))


if __name__ == "__main__":
    main()
