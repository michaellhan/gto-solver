"""Train flop_approx's learned correction of the averaged mixes and write gto/flop_approx_model.py.

The data: each library spot's stored flops read off its other stored flops as flop_approx reads a flop it doesn't
store (the K nearest, the hand class test; the vs-BB spots, which store every flop, off the 552 representative flops
alone), N flops a spot. For each stored decision and combo some source holds an image of (the advisor leaves the
others unanswered): the averaged mix, the entry's own mix and EVs per action, the combo's reach, and
flop_approx.correction_features. The model (flop_approx.correction_z: a logistic correction of
log(mix + CORR_EPS) per node kind and action kind) is fit by Adam, full batch, to minimize the EV given up (each combo's
mix priced by the entry's EVs per action, its reach-weighted average, % of the pot, per flop over its decisions by how
often each comes, flops averaged) plus MU x the squared distance of the mixes from the entry's (reach-weighted), and an
L2 penalty LAM. --folds 2 holds out spots: each half's flops are corrected by the model of the other half's.

    python scripts/train_flop_correction.py --n 80               # fit on every spot, write the model
    python scripts/train_flop_correction.py --n 80 --folds 2     # held out by spot: the numbers, nothing written

Numpy only, one thread; about 10 minutes and under 1 GB for 18 spots x 80 flops.
"""
from __future__ import annotations
import argparse
import base64
import collections
import os
import random
import sys
import time

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from gto import flop_library as fl  # noqa: E402
from gto import flop_approx as fa  # noqa: E402
from gto.breakdown import MADE, DRAWS  # noqa: E402
from gto.cards import cards as parse_cards, cards_str  # noqa: E402
import bench_flop_approx as BA  # noqa: E402

NK, T, NM, ND = fa.N_KINDS, fa.N_TYPES, len(MADE), len(DRAWS)
SHAPES = [("b", (T,)), ("m", (T, NM)), ("d", (T, ND)), ("r", (T, 3)), ("a", (T,)), ("mr", (T, NM, 3)),
          ("g", (T,)), ("md", (T, NM)), ("dd", (T, ND))]
# the first decisions' kinds of node (the root, after a check): left as averaged unless --first (corrected, they gave up
# 8% more EV and mixed a little further, on flops the model wasn't fit on)
FIRST = sorted({fa.node_kind((), p) for p in (0, 1)} | {fa.node_kind(("check",), p) for p in (0, 1)})


def dataset(lib, sids, n, seed):
    """{spot: [node record]}: the averaged mixes and the entry's, per stored decision of N flops a spot."""
    spots = {s["id"]: s for s in lib.index()["spots"]}
    rep = set(cards_str(f) for f in fl.representative_flops(552))
    out = {}
    for sid in sids:
        t0 = time.time()
        spot = spots[sid]
        s = BA.setup(sid)
        stored = sorted(spot["flops"])
        train = [x for x in stored if x in rep] if len(stored) > 600 else stored
        test = [x for x in stored if x not in rep] if len(stored) > 600 else list(stored)
        random.Random(seed).shuffle(test)
        recs = []
        for tn in test[:n]:
            e = lib.entry(spot, tn)
            board = parse_cards(tn)
            nb = [x for x in fa.neighbours([x for x in train if x != tn], board, fa.K) if x[2] <= fa.MAX_DIST]
            if e is None or not nb:
                continue
            srcs = [(lib.entry(spot, nm), m, 1.0 / (1 + d)) for nm, m, d in nb]
            pred = fa.predict(srcs, board=list(board))
            reaches = BA.node_reaches(e, s["full0"], s["full1"])
            for nd in e["nodes"]:
                path = tuple(nd["path"])
                r, share = reaches[path]
                p = nd["player"]
                if not share or path not in pred:
                    continue
                idx = np.nonzero(e["masks"][p])[0]
                P = pred[path]["strat"][:, idx]
                seen = ~np.isnan(P[0])      # (a combo no source holds an image of: unanswered, as in the advisor)
                idx, P = idx[seen], P[:, seen]
                w = r[idx]
                if w.sum() <= 1e-12:
                    continue
                f = fa.correction_features(board, srcs, idx)
                recs.append({"flop": tn, "share": share, "pot": float(spot["pot"]), "w": w / w.sum(),
                             "S": nd["strat"][:, seen], "V": nd["evs"][:, seen],
                             "P": P / np.maximum(P.sum(0, keepdims=True), 1e-12),
                             "types": np.array([fa.action_type(x) for x in nd["names"]]),
                             "kind": fa.node_kind(path, p), "made": f["made"], "draw": f["draw"],
                             "agree": f["agree"], "dr": f["dr"]})
        out[sid] = recs
        print(f"{sid}: {len(recs)} decisions, {time.time() - t0:.0f} s", flush=True)
    return out


def flat(data, sids, first=False):
    """Every (decision, combo) row of these spots, actions padded; each row's weight averages flops; 'on' the rows
    corrected (without first, not the first decisions')."""
    nodes = [(s, x) for s in sids for x in data[s]]
    flops = collections.defaultdict(float)
    for s, x in nodes:
        flops[(s, x["flop"])] += x["share"]
    nf = len(flops)
    A = max(len(x["types"]) for _, x in nodes)
    R = sum(len(x["w"]) for _, x in nodes)
    D = {"L": np.full((R, A), -1e9, np.float32), "V": np.zeros((R, A), np.float32), "S": np.zeros((R, A), np.float32),
         "SV": np.zeros(R, np.float32), "t": np.zeros((R, A), np.int64), "valid": np.zeros((R, A), bool),
         "k": np.zeros(R, np.int64), "made": np.zeros(R, np.int64), "draw": np.zeros(R, np.int64),
         "dr": np.zeros((R, 3), np.float32), "agree": np.zeros(R, np.float32), "wt": np.zeros(R, np.float32),
         "wr": np.zeros(R, np.float32), "node": np.zeros(R, np.int64)}
    flop_of = []
    i = 0
    for j, (s, x) in enumerate(nodes):
        m, a = len(x["w"]), len(x["types"])
        sl = slice(i, i + m)
        D["L"][sl, :a] = np.log(x["P"].T + fa.CORR_EPS)
        D["V"][sl, :a] = x["V"].T
        D["S"][sl, :a] = x["S"].T
        D["SV"][sl] = (x["S"] * x["V"]).sum(0)
        D["t"][sl, :a] = x["types"][None]
        D["valid"][sl, :a] = True
        D["k"][sl] = x["kind"]
        D["made"][sl], D["draw"][sl] = x["made"], x["draw"]
        D["dr"][sl] = x["dr"][None]
        D["agree"][sl] = x["agree"]
        frac = x["share"] / flops[(s, x["flop"])] / nf
        D["wt"][sl] = x["w"] * frac / x["pot"] * 100   # (EV given up, % of the pot)
        D["wr"][sl] = x["w"] * frac                     # (mix distance)
        D["node"][sl] = j
        flop_of.append((s, x["flop"]))
        i += m
    D["flop_of"] = flop_of
    D["nf"] = nf
    D["on"] = np.ones(R, bool) if first else ~np.isin(D["k"], FIRST)
    return D


def probs(th, D):
    th = {g: v.astype(np.float32) for g, v in th.items()}
    k, t = D["k"][:, None], D["t"]
    Lv = np.where(D["valid"], D["L"], 0)
    dis = (1 - D["agree"])[:, None]
    z = th["b"][k, t] + th["m"][k, t, D["made"][:, None]] + th["d"][k, t, D["draw"][:, None]]
    z = z + (th["r"][k, t] * D["dr"][:, None, :]).sum(-1) + th["a"][k, t] * D["agree"][:, None]
    z = z + (th["mr"][k, t, D["made"][:, None]] * D["dr"][:, None, :]).sum(-1)
    z = z + th["g"][k, t] * Lv + (th["md"][k, t, D["made"][:, None]] + th["dd"][k, t, D["draw"][:, None]]) * dis
    z = np.where(D["valid"], D["L"] + z, -1e9)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z) * D["valid"]
    return np.where(D["on"][:, None], e / e.sum(1, keepdims=True), base_probs(D))


def loss_grad(th, D, mu, lam):
    P = probs(th, D)
    ev = (P * D["V"]).sum(1)
    L = float((D["wt"] * (D["SV"] - ev)).sum())
    g = -D["wt"][:, None] * P * (D["V"] - ev[:, None])
    d = (P - D["S"]) * D["valid"]
    L += mu * float((D["wr"][:, None] * d ** 2).sum())
    gp = 2 * mu * D["wr"][:, None] * d
    g = (g + P * (gp - (P * gp).sum(1, keepdims=True))) * D["valid"] * D["on"][:, None]   # d loss / d z
    k = np.broadcast_to(D["k"][:, None], g.shape)
    base = (k * T + D["t"]).ravel()
    gv = g.ravel().astype(np.float64)
    mm = np.broadcast_to(D["made"][:, None], g.shape).ravel()
    dd = np.broadcast_to(D["draw"][:, None], g.shape).ravel()
    Lv = np.where(D["valid"], D["L"], 0)
    gd = (g * (1 - D["agree"])[:, None]).ravel().astype(np.float64)
    G = {"b": np.bincount(base, gv, NK * T).reshape(NK, T),
         "m": np.bincount(base * NM + mm, gv, NK * T * NM).reshape(NK, T, NM),
         "d": np.bincount(base * ND + dd, gv, NK * T * ND).reshape(NK, T, ND),
         "r": np.stack([np.bincount(base, (g * D["dr"][:, j:j + 1]).ravel().astype(np.float64), NK * T)
                        for j in range(3)], -1).reshape(NK, T, 3),
         "a": np.bincount(base, (g * D["agree"][:, None]).ravel().astype(np.float64), NK * T).reshape(NK, T),
         "mr": np.stack([np.bincount(base * NM + mm, (g * D["dr"][:, j:j + 1]).ravel().astype(np.float64), NK * T * NM)
                         for j in range(3)], -1).reshape(NK, T, NM, 3),
         "g": np.bincount(base, (g * Lv).ravel().astype(np.float64), NK * T).reshape(NK, T),
         "md": np.bincount(base * NM + mm, gd, NK * T * NM).reshape(NK, T, NM),
         "dd": np.bincount(base * ND + dd, gd, NK * T * ND).reshape(NK, T, ND)}
    for key in G:
        L += lam * float((th[key] ** 2).sum())
        G[key] = G[key] + 2 * lam * th[key]
    return L, G


def fit(D, mu, lam, iters, lr=0.05):
    th = {g: np.zeros((NK,) + shape) for g, shape in SHAPES}
    m1 = {g: np.zeros_like(v) for g, v in th.items()}
    m2 = {g: np.zeros_like(v) for g, v in th.items()}
    for it in range(1, iters + 1):
        L, G = loss_grad(th, D, mu, lam)
        for g in th:
            m1[g] = 0.9 * m1[g] + 0.1 * G[g]
            m2[g] = 0.999 * m2[g] + 0.001 * G[g] ** 2
            th[g] -= lr * (m1[g] / (1 - 0.9 ** it)) / (np.sqrt(m2[g] / (1 - 0.999 ** it)) + 1e-8)
        if it == 1 or it % 100 == 0:
            print(f"  iteration {it}: loss {L:.4f}", flush=True)
    return th


def measure(D, P):
    """Per flop: (EV given up, % of the pot; mix distance, total variation, reach-weighted); and the share of rows'
    reach whose true mix is mixed with the TV there and where pure."""
    ev = (P * D["V"]).sum(1)
    tv = 0.5 * np.abs(P - D["S"]).sum(1)
    out = []
    for x in (D["wt"] * (D["SV"] - ev), D["wr"] * tv):
        per_node = np.bincount(D["node"], x.astype(np.float64))
        acc = collections.defaultdict(float)
        for j, key in enumerate(D["flop_of"]):
            acc[key] += per_node[j] * D["nf"]
        out.append(np.array([acc[k] for k in sorted(acc)]))
    mixed = D["S"].max(1) < 0.9
    w = D["wr"]
    extra = {"mixedness": float((w * (1 - P.max(1))).sum() / w.sum()),
             "tv_mixed": float((w * tv * mixed).sum() / (w * mixed).sum()),
             "tv_pure": float((w * tv * ~mixed).sum() / (w * ~mixed).sum())}
    return out[0], out[1], extra


def base_probs(D):
    P = np.where(D["valid"], np.exp(D["L"].astype(np.float64)) - fa.CORR_EPS, 0)
    P = np.maximum(P, 0)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-12)


def write(th, path, note):
    used = sorted({k for k in range(NK) if any(np.abs(th[g][k]).max() > 0 for g, _ in SHAPES)})
    flat_ = np.concatenate([th[g][used].ravel() for g, _ in SHAPES]).astype(np.float16)
    data = base64.b64encode(flat_.tobytes()).decode()
    lines = [data[i:i + 116] for i in range(0, len(data), 116)]
    with open(path, "w") as f:
        f.write('"""flop_approx\'s learned correction (flop_approx.correction_z), written by '
                'scripts/train_flop_correction.py:\n' + note + '\n"""\n')
        f.write(f"KINDS = {used}\n")
        f.write(f"SHAPES = {[(g, list(s)) for g, s in SHAPES]}\n")
        f.write("DATA = (\n" + "\n".join(f'    "{x}"' for x in lines) + "\n)\n")
    print(f"wrote {path}: {len(flat_)} weights for kinds {used}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", default=None, help="spot ids (default every library spot)")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--mu", type=float, default=3.0)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--first", action="store_true", help="correct the first decisions too (the root, after a check)")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gto",
                                                  "flop_approx_model.py"))
    a = ap.parse_args(argv)
    lib = fl.FlopLibrary.from_dir()
    sids = a.spots.split(",") if a.spots else sorted(s["id"] for s in lib.index()["spots"])
    data = dataset(lib, sids, a.n, a.seed)
    if a.folds:
        res = []
        for fi in range(a.folds):
            test = sids[fi::a.folds]
            train = [s for s in sids if s not in test]
            th = fit(flat(data, train, a.first), a.mu, a.lam, a.iters)
            D = flat(data, test, a.first)
            b, c = measure(D, base_probs(D)), measure(D, probs(th, D))
            res.append((b, c))
            print(f"fold {fi} (held out: {', '.join(test)}): EV given up {b[0].mean():.4f} -> {c[0].mean():.4f}, "
                  f"mix distance {b[1].mean():.4f} -> {c[1].mean():.4f}", flush=True)
        for nm, j in (("EV given up", 0), ("mix distance (TV)", 1)):
            bb = np.concatenate([r[0][j] for r in res])
            cc = np.concatenate([r[1][j] for r in res])
            print(f"{nm}, per flop ({len(bb)} flops): mean {bb.mean():.4f} -> {cc.mean():.4f}, median {np.median(bb):.4f}"
                  f" -> {np.median(cc):.4f}, 90th percentile {np.percentile(bb, 90):.4f} -> {np.percentile(cc, 90):.4f};"
                  f" lower by over 0.01 on {np.mean(cc < bb - 0.01):.0%} of flops, higher on {np.mean(cc > bb + 0.01):.0%}")
        for key in ("mixedness", "tv_mixed", "tv_pure"):
            print(key, " / ".join(f"{r[0][2][key]:.4f} -> {r[1][2][key]:.4f}" for r in res))
        return
    th = fit(flat(data, sids, a.first), a.mu, a.lam, a.iters)
    write(th, a.out, f"{len(sids)} spots, {a.n} stored flops each (seed {a.seed}), mu {a.mu}, lam {a.lam}, "
                     f"{a.iters} iterations.")


if __name__ == "__main__":
    main()
