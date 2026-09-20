# GTO Solver

A No-Limit Hold'em hand-review solver. Enter a hand seat by seat and it returns
a GTO-style mixed strategy with EVs and sizes, preflop through river, in a few
seconds. 2 to 9 players.

Live at **https://www.michael-han.com/gto-solver/** (the solver runs in your
browser via Pyodide, nothing is uploaded). The **Download** button there gives
you the native version, about 30x faster: unzip, double-click `run.command`
(Mac) or `run.bat` (Windows), and the same page opens against a local solver.

It is a review tool for hands you have already played. There is deliberately no
screen capture, table reading or live-play hook; using anything like that on a
real-money site is against every site's terms.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m unittest -v
```

## Review UI

```bash
python -m gto.server --port 18877
```

Open http://127.0.0.1:18877/ . (The same page is what is deployed to the
website; there it detects that no server is present and runs the identical
Python code in a web worker with Pyodide. Pyodide's numpy is much slower than
native, so the browser build uses a lighter tree, fewer sampled runouts, and an
8 s budget; the local server gives better postflop accuracy.) Pick the number of players (2 to 9), your seat
and stack, click cards into your
hand and the board (click a placed card to remove it), then enter the action
seat by seat with the buttons: fold, check or call, one or a few preset sizes,
a custom size, all-in. A seat strip shows who has folded and who is to act;
clicking a seat folds everyone before it, and a single context shortcut
("Fold to me", "Everyone checks", "Everyone else folds") skips the boring bits.

Whenever it becomes your turn the advice panel solves the spot (about 4 s
postflop) and shows the recommended action with its size (preflop: 2.5bb
opens, 3bb from the small blind, 3x or 4x 3-bets, 2.3x 4-bets, then all-in;
postflop: the solver's bet and raise sizes in big blinds), the mixed strategy for your exact
hand with EVs, and a 13x13 grid of your whole range at that spot with your hand
boxed. Click an advice row to record it as what you did and keep going.
Solver stats and the villain's inferred range sit behind a "Details" toggle.
**Copy hand** puts a plain-text summary on the clipboard: the full action, the
advice with frequencies, EVs and sizes, the villain range, solver stats, a link
that reopens the exact spot, and the raw state JSON for the API or CLI.
The hand is kept in the browser between reloads; **New hand** clears it.

## Use from the shell

Manual entry:

```bash
python -m gto.cli --hero BB --hole 9h8h --board Ks7d2cTh \
  --preflop "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c" --flop "BB:x,CO:b2,BB:c"
```

Action tokens are `POS:act` with `f` fold, `x` check, `c` call, `r<to>` raise to,
`b<amount>` bet, `ai` all-in. Amounts are in big blinds.

Server for your sandbox table:

```bash
python -m gto.server --port 8765
curl -s -X POST localhost:8765/advise -d @examples/flop.json | python -m json.tool
```

Request body (see `examples/`):

```json
{
  "hero": "BTN", "hole": "AhQd", "board": "Qs8d3c",
  "bb": 1.0,
  "stacks": {"BTN": 100, "BB": 100},
  "preflop": [{"pos": "CO", "action": "fold"}, {"pos": "BTN", "action": "raise", "to": 2.5},
              {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}],
  "flop": [{"pos": "BB", "action": "check"}], "turn": [], "river": []
}
```

Set `bb` to your blind size and pass amounts in chips, or leave it at 1 and pass
big blinds. Positions are the 6-max names `UTG HJ CO BTN SB BB`.

Response: per-action `freq` (the mixed strategy for hero's exact hand), `ev_bb`
(relative to folding now), `amount_bb`, a `recommended` action (highest
frequency), a `sampled` action (drawn from the mix, use this if you want to
actually play unexploitably), the range-wide frequencies, the villain's inferred
range, and solver stats.

## How it works

| Layer | File | Method |
|---|---|---|
| Cards, ranges | `gto/cards.py`, `gto/ranges.py` | 1326-combo weight vectors, PIO-style range syntax |
| Hand strength | `gto/evaluator.py` | `treys` 7-card evaluator, cached per board |
| Preflop | `gto/preflop.py` | Hand-written 100bb charts: RFI, vs RFI, vs 3-bet, vs 4-bet, cold vs 3-bet, mixed frequencies. Seats for 2 to 9 players map onto the charts by how many players are left to act (9-max LJ plays the 6-max UTG chart, 7+ handed UTG plays a tighter EP chart, heads-up has its own button and big-blind charts). Replaying the preflop action through the charts gives every player's range going to the flop. |
| Postflop | `gto/solver.py` | Heads-up range-vs-range **discounted CFR** over a bet tree (2 bet sizes per street, raises to half pot and full pot after the call, all-in when the stack is short or a raise size would exceed it, max 2 raises). Vectors are compressed to combos in either range. A showdown is one matvec with a precomputed antisymmetric matrix. |
| Hand replay | `gto/advisor.py` | The whole hand is replayed for any number of seats: blinds, folds, dead money from players who folded, all-ins, acting order (heads-up the button acts first preflop and last postflop). Whenever exactly two players remain the spot is solved heads-up: from the start of the street if it began heads-up, or from the current node if a third player just folded. With three or more live players it falls back to Monte Carlo equity against every live range plus pot-odds logic, and says so. |
| Multi-street | `gto/advisor.py` | Earlier streets are solved with a small budget and each player's range is narrowed by the frequency of the action they actually took, mixed with a 15% uniform floor so an approximate solve never prunes a line to nothing. The current street is solved from the start of the street and the tree is walked to hero's node with exact frequencies. Solutions are cached per hand so the flop solve is reused on the turn. |
| Front ends | `gto/cli.py`, `gto/server.py`, `gto/static/index.html` | Manual CLI; local HTTP endpoint; click-driven review page served at `/`. |

### Accuracy vs speed

* **River**: exact for the bet-size abstraction; converges in well under a second.
* **Turn**: turn decisions are full; river subtrees are full for `n_rivers` sampled
  rivers (default 12). About 65 iterations in the 4 s budget on wide ranges.
* **Flop**: flop decisions are full, turn decisions are full on `n_turns` sampled
  turns (default 6), and the river is an **equity leaf**: the average showdown matrix
  over 16 sampled rivers. So flop play can't see river bluffs or thin value bets.
  About 80 iterations in 4 s on wide ranges.
* **Earlier streets** used for range narrowing get 2 s each, and a solve that is
  reused later in the same hand keeps iterating from its stored regrets instead of
  restarting. When a real bet or raise size differs from the abstraction by more
  than 15%, that street's tree uses the real size so narrowing is exact.
* Multiway pots (3+ players live at your decision) are not solved; an
  equity-vs-field pot-odds heuristic is returned with a note. Once the pot is
  heads-up it is solved, but streets that were multiway do not narrow ranges.

### Deploy to the website

```bash
scripts/deploy.sh
```

copies the page, the worker and the `gto` sources into the `gto-solver/` folder
of the michaellhan.github.io repo and pushes.

Tune with `Config` in `gto/solver.py` (`depth`, sizes, sample counts) or the
`--budget` / `--depth` flags. `depth=3` from the flop is the full tree; it is
much slower in pure Python. The real fix for a production version is moving
the tree traversal to C or Rust or a GPU, which makes full flop trees feasible.

### Known limits

* Preflop charts are approximations written by hand, not solver output. Replace
  them with real chart exports if you have them; the format is a range string.
* No limps, no straddles, no antes, one effective stack for the table.
* The bet-size abstraction is coarse; real actions are mapped to the nearest size.
