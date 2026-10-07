"""Sweep every standard preflop line (opens, vs an open, vs a 3bet, cold vs a 3bet, vs a 4bet) for 2..9
players through the charts and their adjustments, and flag errors, frequencies that don't sum to 1,
AA/KK folding, and 72o continuing. -v prints each line's continue rate.

    python scripts/preflop_sweep.py [-v]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, collections
from gto import preflop as pf
from gto.advisor import preflop_strategy, preflop_ranges, replay_hand
from gto.cards import COMBO_INDEX
from gto.advisor import parse_cards
idx = lambda h: COMBO_INDEX[tuple(sorted(parse_cards(h)))]
AA, KK, T72, QQ = idx("AhAd"), idx("KhKd"), idx("7h2d"), idx("QhQd")
R = lambda p, a, to=None: dict({"pos": p, "action": a}, **({"to": to} if to else {}))
issues, table = [], []
def check(n, label, line, hero, stack=100):
    T = pf.Table(pf.SEATS[n])
    st = replay_hand(T, stack, {"preflop": line}, 1)[0]
    if st.closed or st.to_act != hero:
        issues.append(f"{n}h {label}: not hero's turn ({st.to_act})"); return
    try:
        s = preflop_strategy(hero, line, T, stack)
    except Exception as e:
        issues.append(f"{n}h {label}: ERROR {e!r}"); return
    arr = preflop_ranges(line, T, stack)[hero]
    fold = s.get("fold", np.zeros(1326))
    tot = sum(v for k, v in s.items())
    if np.abs(tot - 1).max() > 1e-6: issues.append(f"{n}h {label}: freqs sum {tot.min():.3f}-{tot.max():.3f}")
    if (min(v.min() for v in s.values()) < -1e-9): issues.append(f"{n}h {label}: negative freq")
    if fold[AA] > 0.01: issues.append(f"{n}h {label}: AA folds {fold[AA]:.2f}")
    if fold[KK] > 0.05 and "5bet" not in label and "4bet" not in label: issues.append(f"{n}h {label}: KK folds {fold[KK]:.2f}")
    if "open" not in label and "check" not in s and 1 - fold[T72] > 0.05 and not label.startswith("BB") and n > 2:
        issues.append(f"{n}h {label}: 72o continues {1-fold[T72]:.2f}")
    w = arr / max(arr.sum(), 1e-9)
    cont = float((w * (1 - fold)).sum()) if "check" not in s else float((w * s["raise"]).sum())
    table.append((n, label, round(arr.sum()), round(cont, 2)))
for n in range(2, 10):
    seats = pf.SEATS[n]
    for i, h in enumerate(seats):
        if h != "BB":
            check(n, f"{h} open", [R(p, "fold") for p in seats[:i]], h)
        for j, o in enumerate(seats[:i]):
            pre = [R(p, "fold") for p in seats[:j]] + [R(o, "raise", 2.5)] + [R(p, "fold") for p in seats[j+1:i]]
            check(n, f"{h} vs {o} open", pre, h)
            # opener faces h's 3bet
            size = 10 if h in ("SB", "BB") else 7.5
            pre3 = pre + [R(h, "raise", size)] + [R(p, "fold") for p in seats[i+1:]]
            check(n, f"{o} open vs {h} 3bet", pre3, o)
            # cold: a later seat k faces o open + h 3bet
            for k in range(i + 1, len(seats)):
                c = seats[k]
                prec = pre + [R(h, "raise", size)] + [R(p, "fold") for p in seats[i+1:k]]
                check(n, f"{c} cold vs {o} open {h} 3bet", prec, c)
            # 3bettor faces 4bet
            pre4 = pre3 + [R(o, "raise", 22)] + [R(p, "fold") for p in seats[j+1:i] if False]
            check(n, f"{h} 3bet vs {o} 4bet", pre4, h)
print("issues:", len(issues)); print("\n".join(issues))
print("lines checked:", len(table))
if "-v" in sys.argv:
    for n, label, combos, cont in table:
        print(f"{n}-handed  {label:36s} {combos:5d} combos arrive, {cont:.0%} continue")
