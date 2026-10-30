"""Manual-entry CLI.

Action strings: comma separated `POS:act`, act in f | x | c | r<to> | b<amount> | ai
  python -m gto.cli --hero BTN --hole AhKd --board Ks7d2c \
      --preflop "UTG:f,HJ:f,CO:r2.5,BTN:c,SB:f,BB:f" --flop "CO:b2"
"""
from __future__ import annotations
import argparse
import json
import re
from .native import advisor   # (first: it holds BLAS to one thread before numpy loads)

_ACT = re.compile(r"^(f|x|c|ai|r([\d.]+)?|b([\d.]+)?)$", re.I)


def parse_actions(s: str):
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        pos, act = tok.split(":")
        m = _ACT.match(act.strip())
        if not m:
            raise ValueError(f"bad action {tok}")
        a = act.lower()
        if a == "f":
            out.append({"pos": pos, "action": "fold"})
        elif a == "x":
            out.append({"pos": pos, "action": "check"})
        elif a == "c":
            out.append({"pos": pos, "action": "call"})
        elif a == "ai":
            out.append({"pos": pos, "action": "all-in"})
        elif a.startswith("r"):
            out.append({"pos": pos, "action": "raise", "to": float(a[1:]) if len(a) > 1 else None})
        else:
            out.append({"pos": pos, "action": "bet", "amount": float(a[1:]) if len(a) > 1 else None})
    return out


def parse_bounties(text: str, positions) -> dict:
    """--bounties: "pko 10", "ko 10 SB=30 field=12", "mystery 25" or "mystery pool=900 left=30" -> the state's
    "bounties" (the advisor checks the values)."""
    words = text.replace(",", " ").split()
    if not words:
        return {}
    out: dict = {"format": words[0].lower()}
    seats = {}
    for w in words[1:]:
        k, _, v = w.partition("=")
        if not v:
            out["envelope" if out["format"] in ("mystery", "mb") else "seats"] = k
        elif k.lower() in ("field", "envelope", "pool", "left"):
            out[k.lower()] = v
        else:
            seats[k.upper()] = v
    if seats:
        every = out.pop("seats", 0)
        out["seats"] = {**{p: every for p in positions}, **seats}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="GTO advisor (sandbox / manual entry)")
    ap.add_argument("--hero", required=True)
    ap.add_argument("--hole", required=True)
    ap.add_argument("--board", default="")
    ap.add_argument("--preflop", default="")
    ap.add_argument("--flop", default="")
    ap.add_argument("--turn", default="")
    ap.add_argument("--river", default="")
    ap.add_argument("--stack", type=float, default=100.0, help="starting stacks in bb")
    ap.add_argument("--ante", type=float, default=0.0, help="ante per player in bb")
    ap.add_argument("--straddle", action="store_true", help="the first seat straddles 2bb")
    ap.add_argument("--rake", type=float, default=0.0, help="rake, share of the pot (0.05 = 5%%); 0 = none")
    ap.add_argument("--rake-cap", type=float, default=None, help="most rake per pot in bb (default: no cap)")
    ap.add_argument("--players", type=int, default=6, help="players dealt in, 2-9")
    ap.add_argument("--stacks", default="", help='seats\' own stacks in bb, "CO=40,BTN=25,SB=8,BB=25"')
    ap.add_argument("--payouts", default="", help='a tournament\'s prizes still to be paid, "50 30 20" (ICM)')
    ap.add_argument("--bounties", default="", help='a bounty tournament: "pko 10" (10 on every head), "ko 10 SB=30 '
                    'field=12" (seats\' own, the other tables\' average), "mystery 25" (the average envelope)')
    ap.add_argument("--speed", choices=("fast", "accurate"), default="fast",
                    help="tree and budgets: fast 2.5 s a street, accurate 30 s (12 s per earlier street)")
    ap.add_argument("--budget", type=float, default=None, help="seconds for the current street solve")
    ap.add_argument("--prior-budget", type=float, default=None, help="seconds per earlier street (range narrowing)")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    state = {
        "players": args.players, "hero": args.hero, "hole": args.hole, "board": args.board, "stack": args.stack,
        "ante": args.ante, "straddle": args.straddle, "rake": args.rake, "rake_cap": args.rake_cap,
        "preflop": parse_actions(args.preflop), "flop": parse_actions(args.flop),
        "turn": parse_actions(args.turn), "river": parse_actions(args.river),
    }
    if args.stacks:
        state["stacks"] = {k.strip(): float(v) for k, v in (x.split("=") for x in args.stacks.split(",") if x.strip())}
    if args.payouts:
        state["payouts"] = args.payouts
    if args.bounties:
        from .preflop import SEATS
        state["bounties"] = parse_bounties(args.bounties, SEATS.get(args.players, SEATS[6]))
    adv = advisor(args.speed, args.depth, args.budget, args.prior_budget)
    res = adv.advise(state)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    print_result(res)


def print_result(res: dict):
    if "error" in res:
        print("error:", res["error"]); return
    print(f"[{res['street']}] hero {res['hero']} {res.get('hand','')} "
          f"{res.get('position','')} board {res.get('board','-')} pot {res.get('pot_bb','-')}bb")
    for a in res["actions"]:
        amt = f" ({a['amount_bb']}bb)" if a.get("amount_bb") is not None else ""
        ev = f"  EV {a['ev_bb']:+.2f}bb" if "ev_bb" in a else ""
        bar = "#" * int(round(a["freq"] * 30))
        print(f"  {a['action']:<14}{amt:<10} {a['freq']*100:5.1f}% {bar:<30}{ev}")
    print(f"  => recommended: {res['recommended']}   (sampled from mix: {res['sampled']})")
    if "range_action_freq" in res:
        print("  range-wide:", res["range_action_freq"])
        print("  villain range:", res["villain_range"][:160])
    if res.get("solver"):
        s = res["solver"]
        print(f"  solver: {s['iterations']} iters, {s['solve_seconds']}s, {s['nodes']} nodes, {s['active_combos']} combos")
    for n in res.get("notes", []):
        print("  note:", n)
    print(f"  total {res['seconds']}s")


if __name__ == "__main__":
    main()
