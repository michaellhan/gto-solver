"""5-way flop and turn bucket games against a richer one (no per-combo reference fits a 5-way flop: the README's N-player
pots): the limped 5-way pot of scripts/score_multiway4.py (UTG, HJ, BTN and SB limp, the BB checks; 6-max 100bb), a
line's ranges as the betting rules narrow them, solved in each game for a number of iterations and compared with the
first game: the advised player's range frequencies, its mix played against the first game's EVs (% of the pot given up,
weighted by its range), each game's own exploitability (mean best-response gain, % of the pot), and its CPU (the tables,
then ms an iteration, one thread unless GTO_THREADS says otherwise).

  python scripts/bench_mw5.py Kh7d2c first 8:3:0.5:300,4:6:0.5:200,3:6:0.5:200,4:6:0:200
  python scripts/bench_mw5.py Kh7d2c9s "vs bet" 10:1:0.5:200,6:1:0.5:200,4:1:0.5:200
  python scripts/bench_mw5.py Kh7d2c first 6:3:0.5:300,4:6:0.5:200 --rules 80   # the betting rules too, on 80 hands

--cache DIR keeps the first game's strategy and EVs, and the rules' row, in DIR (per preflop line, board, line, game and
--rules), so later games on the same spot skip it.

--rules N plays the betting rules (Advisor._multiway, 400 trials) on N hands drawn from the advised player's range by
weight and scores their mix against the first game's EVs the same way (the hands' mean, the games' on the same hands).

A game is K:step:eq:iterations[:nr][:j] (buckets a player below the root street, the turn's tables over every step-th
river from a flop, MWConfig.bucket_eq, iterations; nr: a flop's river not bet on, MWConfig.bucket_river False; j: no
all-in over a raise, MWConfig.jam_is_raise). --pres
limp6 (or iso6, limp7) plays score_multiway4.py's live 9-handed 6-way (7-way) pots instead:

  python scripts/bench_mw5.py Kh7d2c first 4:6:0.75:200,3:12:0.75:60:nr --pres limp6 --rules 60 Lines: first, last checked to, vs bet, vs bet+call (score_multiway4.py's).
"""
import sys, os, time, resource
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import score_multiway4 as S
import compare_multiway as CM
import gto.advisor as A
from gto.multiway_solver import MWConfig, MultiwaySolver


PRE = "limp5"     # (--pres: score_multiway4.py's preflop lines, limp6 / iso6 / limp7 for 6- and 7-way)
POT = {"limp5": 5.0, "limp6": 6.0, "limp7": 7.0, "iso6": 30.5}


def line_acts(board, line):
    hero, acts = S.lines_for(PRE)[line]
    if len(board) == 6:      # (a flop: the line's bet is a third of the pot, as the solve's)
        acts = [dict(a, amount=round(POT[PRE] * 0.33, 1)) if a["action"] == "bet" else a for a in acts]
    return hero, acts


def spot(board, line):
    hero, acts = line_acts(board, line)
    _, _, b, ranges, remaining, cur, stk = CM._spot_args(S.rules_adv(10), S.mk_state(PRE)(board, hero, "2s2h", acts))
    players = [p for p in cur.order if p in remaining]
    hp = players.index(hero)
    pending = {i for i, p in enumerate(players) if p not in cur.acted} | {hp}
    St = min(stk[p] - cur.inv_start[p] for p in players)
    return b, [ranges[p] for p in players], cur.pot_start, St, dict(bets=[cur.bets[p] for p in players], to_act=hp,
                                                                     n_raises=cur.n_raises, pending=pending), hp


def main():
    args = sys.argv[1:]
    n_rules = 0
    if "--rules" in args:
        i = args.index("--rules"); n_rules = int(args[i + 1]); del args[i:i + 2]
    cache = None
    if "--cache" in args:
        i = args.index("--cache"); cache = args[i + 1]; del args[i:i + 2]
    if "--pres" in args:
        global PRE
        i = args.index("--pres"); PRE = args[i + 1]; del args[i:i + 2]
    board, line, games = args[0], args[1], [g.split(":") for g in args[2].split(",")]
    b, rs, pot, St, kw, hp = spot(board, line)
    ref = None
    drawn = None
    fn = None
    if cache:
        os.makedirs(cache, exist_ok=True)
        fn = os.path.join(cache, f"{PRE}_{board}_{line.replace(' ', '_')}_{'-'.join(games[0])}_{n_rules}.npz")
        if os.path.exists(fn):
            z = np.load(fn, allow_pickle=True)
            ref, drawn = (z["rsig"], z["rev"], float(z["rp"])), (z["drawn"] if n_rules else None)
            print(str(z["row"]) + "  (cached)", flush=True)
            if n_rules:
                print(str(z["rules"]), flush=True)
            games = games[1:]
    for g in games:
        K, step, eq, it = g[:4]
        cfg = MWConfig(bet=A.MW4_BETS, max_raises=1, root_raise=True, later="buckets", buckets=int(K),
                       bucket_river_step=int(step), bucket_eq=float(eq), bucket_river="nr" not in g[4:],
                       jam_is_raise="j" in g[4:])
        t0 = time.process_time()
        s = MultiwaySolver(b, rs, pot, St, cfg=cfg, **kw)
        tb = time.process_time() - t0
        t0 = time.process_time()
        s.run(iters=int(it))
        ti = (time.process_time() - t0) / int(it) * 1000
        gn = s.br_gains() / s.root_pot() * 100
        sig = s.root.avg_strategy()[0]
        ev = s.action_evs()
        w = s.reach0[hp, 0] / s.reach0[hp, 0].sum()
        freqs = {n: float((w * sig[i]).sum()) for i, n in enumerate(s.root.names)}
        row = f"{':'.join(g)}  tables {tb:5.1f}s  {ti:6.1f} ms/it  expl {gn.mean():.3f}%  " + \
              " ".join(f"{n} {f * 100:4.1f}" for n, f in freqs.items())
        if ref is None:
            ref = (sig, ev, s.root_pot())
            rules = ""
            if n_rules:
                rng = np.random.default_rng(7)
                live = np.flatnonzero(w > 0)
                drawn = rng.choice(live, size=min(n_rules, len(live)), replace=False, p=w[live] / w[live].sum())
                rules = rules_row(s, drawn, ref, board, line, hp)
                print(rules, flush=True)
            if fn:
                np.savez(fn, rsig=sig, rev=ev, rp=s.root_pot(), drawn=drawn if drawn is not None else np.zeros(0, int),
                         row=row, rules=rules)
        else:
            rsig, rev, rp = ref
            loss = ((rsig * rev).sum(axis=0) - (sig * rev).sum(axis=0)) / rp * 100
            tv = 0.5 * np.abs(sig - rsig).sum(axis=0)
            row += f"  | vs first: mix loss {float((w * loss).sum()):.3f}% pot, TV {float((w * tv).sum()):.3f}"
            if drawn is not None:
                row += f", on the rules' hands {float(loss[drawn].mean()):.3f}%"
        print(row + f"  maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB", flush=True)


def rules_row(s, drawn, ref, board, line, hp):
    """The betting rules' mix on the drawn combos against the first game's EVs (% of the pot given up, their mean)."""
    from gto.cards import ALL_COMBOS, cards_str
    rsig, rev, rp = ref
    hero, acts = line_acts(board, line)
    adv = S.rules_adv(400)
    names = list(s.root.names)
    amts = [x or 0 for x in s.root.amounts]
    out, bet = [], 0.0
    for ci in drawn:
        h = cards_str(ALL_COMBOS[int(s.hands.active[ci])])
        r = adv.advise(S.mk_state(PRE)(board, hero, h, acts))
        v = 0.0
        for x in r["actions"]:
            same = [i for i, n in enumerate(names) if S.kind(n) == S.kind(x["action"])]
            if same:
                i = min(same, key=lambda i: abs(amts[i] - (x.get("amount_bb") or 0)))
                v += x["freq"] * rev[i, ci]
                bet += x["freq"] * (S.kind(x["action"]) in ("bet", "raise")) / len(drawn)
        out.append(((rsig[:, ci] * rev[:, ci]).sum() - v) / rp * 100)
    return f"rules on {len(drawn)} hands: mix loss {np.mean(out):.3f}% pot (bets/raises {bet * 100:.0f}%)"


if __name__ == "__main__":
    main()
