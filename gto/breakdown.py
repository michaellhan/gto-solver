"""GTO Wizard-style "Breakdown" and "Equity" views of a postflop spot, computed only when a request asks for them
("breakdown": true in the state; the page sends it when its Breakdown view opens, on the same state as the advice,
so the solve comes from the advisor's cache).

* made_category / draw_category: what a combo holds on the board, by what its hole cards add (a pair the board
  makes for everyone isn't one; a river straight or flush the board makes alone "plays the board").
* range_equities: every combo's equity against the other range, both ways, over the runouts (all of them on the
  turn, an even sample on the flop), with card removal between the two hands.
* breakdown: the payload: per category its share of the range, the acting player's strategy mix and EV, the
  category's equity; each range's sorted equity curve, average equity and EV, and its share above NUT_EQ.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Optional, Sequence
import numpy as np
from .cards import ALL_COMBOS

MADE = [("quads", "Quads or better"), ("full_house", "Full house"), ("flush", "Flush"), ("straight", "Straight"),
        ("set", "Set"), ("trips", "Trips"), ("two_pair", "Two pair"), ("overpair", "Overpair"),
        ("top_tk", "Top pair, top kicker"), ("top_gk", "Top pair, good kicker"), ("top_wk", "Top pair, weak kicker"),
        ("middle", "Middle pair"), ("weak", "Weak pair / underpair"), ("ace_high", "Ace high"),
        ("nothing", "No made hand"), ("board", "Plays the board")]
DRAWS = [("combo", "Combo draw"), ("fd", "Flush draw"), ("oesd", "Open-ended"), ("gutshot", "Gutshot"),
         ("backdoor", "Backdoor draw"), ("none", "No draw")]
NUT_EQ = 0.8          # "nut advantage": a range's share with this much equity or more
CURVE_POINTS = 51     # the equity curve's points: every 2% of the range
PAIR_BOARD_OPS = 4e7  # flop equity: runouts sampled so (pairs of combos) x (runouts) stays near this

_STRAIGHTS = [0b11111 << i for i in range(10)]   # over 14 bits: bit 0 the ace played low, bit r + 1 rank r


def _bits(ranks) -> int:
    m = 0
    for r in ranks:
        m |= 1 << (r + 1)
        if r == 12:
            m |= 1
    return m


def _straight(m: int) -> bool:
    return any(m & s == s for s in _STRAIGHTS)


def _top(m: int) -> int:
    """The highest straight in the rank bits (its index in _STRAIGHTS), -1 for none."""
    return max((i for i, s in enumerate(_STRAIGHTS) if m & s == s), default=-1)


def _made_class(cards: Sequence[int]) -> int:
    """8 straight flush, 7 quads, 6 full house, 5 flush, 4 straight, 3 trips, 2 two pair, 1 pair, 0 high card."""
    ranks = [c // 4 for c in cards]
    cnt = sorted((ranks.count(r) for r in set(ranks)), reverse=True)
    suits = [c % 4 for c in cards]
    fl = [s for s in range(4) if suits.count(s) >= 5]
    if fl and _straight(_bits(c // 4 for c in cards if c % 4 == fl[0])):
        return 8
    if cnt[0] >= 4:
        return 7
    if cnt[0] == 3 and len(cnt) > 1 and cnt[1] >= 2:
        return 6
    if fl:
        return 5
    if _straight(_bits(ranks)):
        return 4
    return 3 if cnt[0] == 3 else 2 if cnt[0] == 2 and len(cnt) > 1 and cnt[1] == 2 else 1 if cnt[0] == 2 else 0


def made_category(hole: Sequence[int], board: Sequence[int], plays_board: bool = False) -> str:
    """The made-hand category (an id from MADE) of two hole cards on a 3-5 card board. Straights and better count
    when the hole cards play (plays_board: on the river, the hand is only as strong as the board alone: "board");
    below that, by what the hole cards pair: a pocket pair against the board's ranks, one hole card by the board
    rank it pairs (top pair by its kicker), two hole cards pairing two board ranks two pair."""
    cls = _made_class(list(hole) + list(board))
    bcls = _made_class(board) if len(board) >= 4 else 0
    if cls >= 4:
        if plays_board:
            return "board"
        if not (cls == 7 and bcls == 7):          # the board's own quads are no one's
            return {8: "quads", 7: "quads", 6: "full_house", 5: "flush", 4: "straight"}[cls]
    br = [c // 4 for c in board]
    distinct = sorted(set(br), reverse=True)
    h1, h2 = sorted((c // 4 for c in hole), reverse=True)
    if h1 == h2:
        if br.count(h1):
            return "set"
        return "overpair" if h1 > distinct[0] else "middle" if len(distinct) > 1 and h1 > distinct[1] else "weak"
    hit = [r for r in (h1, h2) if r in br]
    if any(br.count(r) >= 2 for r in hit):
        return "trips"
    if len(hit) == 2:
        return "two_pair"
    if hit:
        i = distinct.index(hit[0])
        if i == 0:
            kick = h2 if hit[0] == h1 else h1
            better = sum(1 for r in range(kick + 1, 13) if r not in distinct)   # better kickers left
            return "top_tk" if better == 0 else "top_gk" if better <= 2 else "top_wk"
        return "middle" if i == 1 else "weak"
    return "ace_high" if h1 == 12 else "nothing"


def draw_category(hole: Sequence[int], board: Sequence[int]) -> str:
    """The draw (an id from DRAWS) of two hole cards on a flop or turn ("none" on the river): a flush draw is four
    to a suit with a hole card in it; a straight draw counts the ranks that would make a straight higher than the
    board's own with that card (two or more: open-ended, double gutters included; one: gutshot); combo both; backdoor (flop) three
    to a suit or to a straight with a hole card, and nothing better."""
    if len(board) >= 5:
        return "none"
    cards = list(hole) + list(board)
    cls = _made_class(cards)
    fd = cls < 5 and any(sum(1 for c in cards if c % 4 == s) == 4 for s in {c % 4 for c in hole})
    m, bm = _bits(c // 4 for c in cards), _bits(c // 4 for c in board)
    outs = 0 if _straight(m) else sum(1 for r in range(13) if _top(m | _bits([r])) > _top(bm | _bits([r])))
    sd = "oesd" if outs >= 2 else "gutshot" if outs == 1 else None
    if fd and sd:
        return "combo"
    if fd or sd:
        return "fd" if fd else sd
    if len(board) == 3:
        bdfd = cls < 5 and any(sum(1 for c in cards if c % 4 == s) == 3 for s in {c % 4 for c in hole})
        hm = _bits(c // 4 for c in hole)
        bdsd = not _straight(m) and any(bin(m & w).count("1") >= 3 and m & w & hm and bin(bm & w).count("1") < 3
                                        for w in _STRAIGHTS)
        if bdfd or bdsd:
            return "backdoor"
    return "none"


def categorize(board: Sequence[int], idx: np.ndarray):
    """(made ids, draw ids) for the combos idx (indices into ALL_COMBOS) on the board."""
    board = list(board)
    plays = np.zeros(len(idx), dtype=bool)
    if len(board) == 5:
        from treys import Card, Evaluator
        from .cards import card_str
        from .evaluator import eval_board
        alone = 7463 - Evaluator().evaluate([], [Card.new(card_str(c)) for c in board])
        plays = eval_board(board, idx) == alone   # (as _plays_board, for many combos)
    made, draws = [], []
    for k, i in enumerate(idx):
        hole = ALL_COMBOS[int(i)]
        made.append(made_category(hole, board, bool(plays[k])))
        draws.append(draw_category(hole, board))
    return made, draws


def _plays_board(hole, board) -> bool:
    from treys import Card, Evaluator
    from .cards import card_str
    ev = Evaluator()
    t = lambda cs: [Card.new(card_str(c)) for c in cs]
    return ev.evaluate(t(hole), t(board)) == ev.evaluate([], t(board))


def _runouts(board: Sequence[int], n_pairs: int) -> List[tuple]:
    rest = [c for c in range(52) if c not in set(board)]
    k = 5 - len(board)
    runs = list(itertools.combinations(rest, k)) if k else [()]
    if len(board) == 3:
        step = max(1, int(np.ceil(len(runs) * n_pairs / PAIR_BOARD_OPS)))
        runs = runs[::step]
    return runs


def range_equities(board: Sequence[int], w0: np.ndarray, w1: np.ndarray):
    """Every combo's equity (1326 arrays, NaN outside the range) of range w0 against w1 and of w1 against w0, ties
    half, over the runouts (see _runouts), counting each pair of hands on the boards neither blocks."""
    from .evaluator import eval_boards
    from .solver import MEMBER_FULL
    i0, i1 = np.nonzero(w0 > 0)[0], np.nonzero(w1 > 0)[0]
    e0, e1 = np.full(len(w0), np.nan), np.full(len(w1), np.nan)
    if not len(i0) or not len(i1):
        return e0, e1
    share = MEMBER_FULL[:, i0].T @ MEMBER_FULL[:, i1]
    ok = share == 0
    N = np.zeros((len(i0), len(i1)), dtype=np.float32)    # wins (ties half) of i0's combo, summed over boards
    D = np.zeros((len(i0), len(i1)), dtype=np.float32)    # boards counted
    runs = _runouts(board, len(i0) * len(i1))
    both = np.concatenate([i0, i1])
    for b0 in range(0, len(runs), 64):
        for s in eval_boards([list(board) + list(r) for r in runs[b0:b0 + 64]], both):
            a, b = s[:len(i0)], s[len(i0):]
            valid = (a >= 0)[:, None] & (b >= 0)[None, :]
            D += valid
            N += valid * ((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :]))
    N, D = N * ok, D * ok
    v0, v1 = w0[i0][:, None], w1[i1][None, :]
    d0, d1 = (D * v1).sum(1), (D * v0).sum(0)
    e0[i0] = np.where(d0 > 0, (N * v1).sum(1) / np.maximum(d0, 1e-12), np.nan)
    e1[i1] = np.where(d1 > 0, ((D - N) * v0).sum(0) / np.maximum(d1, 1e-12), np.nan)
    return e0, e1


def _curve(eq: np.ndarray, w: np.ndarray) -> List[float]:
    """The range's equities sorted high to low, read at CURVE_POINTS even steps of its weight."""
    m = (w > 0) & ~np.isnan(eq)
    if not m.any():
        return []
    o = np.argsort(-eq[m], kind="stable")
    e, cw = eq[m][o], np.cumsum(w[m][o])
    cw /= cw[-1]
    q = np.linspace(0, 1, CURVE_POINTS)
    return [round(float(e[min(int(np.searchsorted(cw, x, "left")), len(e) - 1)]), 3) for x in q]


def _avg(x: np.ndarray, w: np.ndarray) -> Optional[float]:
    m = (w > 0) & ~np.isnan(x)
    t = float(w[m].sum())
    return round(float((x[m] * w[m]).sum() / t), 4) if t > 1e-12 else None


def _groups(order, ids: List[str], idx: np.ndarray, w: np.ndarray, eq, strat, ev) -> List[dict]:
    out = []
    for cid, label in order:
        sel = idx[[k for k, x in enumerate(ids) if x == cid]]
        c = float(w[sel].sum()) if len(sel) else 0.0
        if c < 0.005:
            continue
        row = {"id": cid, "label": label, "c": round(c, 2)}
        if eq is not None:
            row["eq"] = _avg(eq[sel], w[sel])
        if strat is not None:
            row["f"] = [round(float((s[sel] * w[sel]).sum() / c), 3) for s in strat]
        if ev is not None:
            a = _avg(ev[sel], w[sel])
            row["ev"] = None if a is None else round(a, 2)
        out.append(row)
    return out


def side(board, w: np.ndarray, who: str, eq=None, strat=None, ev=None, ev_avg=None) -> dict:
    """One player's breakdown: their range's categories (with the strategy `strat` (actions, 1326) and per-combo `ev`
    where given), and with equities `eq` (1326) its equity, curve and share at NUT_EQ or more."""
    idx = np.nonzero(w > 1e-9)[0]
    made, draws = categorize(board, idx)
    tot = float(w[idx].sum())
    out = {"who": who, "combos": round(tot, 1),
           "made": _groups(MADE, made, idx, w, eq, strat, ev),
           # (no draws on the river)
           "draws": _groups(DRAWS, draws, idx, w, eq, strat, ev) if len(board) < 5 else []}
    if eq is not None:
        m = ~np.isnan(eq[idx])
        out["eq"] = _avg(eq, w)
        out["nuts"] = round(float(w[idx][m & (eq[idx] >= NUT_EQ)].sum() / max(w[idx][m].sum(), 1e-12)), 3)
        out["curve"] = _curve(eq, w)
    if ev_avg is not None:
        out["ev"] = round(float(ev_avg), 2)
    elif ev is not None:
        a = _avg(ev, w)
        out["ev"] = None if a is None else round(a, 2)
    return out


def breakdown(board, actions: List[str], hero: str, hero_w: np.ndarray, strat: np.ndarray, ev: np.ndarray,
              villains: Dict[str, np.ndarray], pot: float, equity: bool = True, villain_ev: bool = True,
              hole: Optional[Sequence[int]] = None) -> dict:
    """The payload: hero's categories with its strategy mix (strat: actions x 1326) and EV (ev: 1326, bb), the
    villains' composition; heads-up with equity, each side's equity against the other and the villain's EV as
    the pot less hero's (no rake: villain_ev); "mine": the categories of hero's own hand (hole)."""
    board = list(board)
    one = len(villains) == 1 and equity
    e0 = e1 = None
    if one:
        (vname, vw), = villains.items()
        e0, e1 = range_equities(board, hero_w, vw)
    h = side(board, hero_w, hero, e0, strat, ev)
    if hole is not None:
        h["mine"] = [made_category(hole, board, len(board) == 5 and _plays_board(hole, board)),
                     draw_category(hole, board)]
    vs = []
    for name, vw in villains.items():
        vev = pot - h["ev"] if one and villain_ev and h.get("ev") is not None else None
        vs.append(side(board, vw, name, e1 if one else None, ev_avg=vev))
    return {"actions": list(actions), "pot": round(float(pot), 2), "hero": h, "villains": vs, "nut_eq": NUT_EQ}
