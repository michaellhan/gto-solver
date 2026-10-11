# GTO Solver

A No-Limit Hold'em hand-review solver. Enter a hand seat by seat and it returns
a GTO-style mixed strategy with EVs and sizes, preflop through river, in a few
seconds. 2 to 9 players.

Live at **https://www.michael-han.com/gto-solver/** (the solver runs in your
browser via Pyodide, nothing is uploaded).

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
Python code in a web worker with Pyodide. Pyodide's numpy is 10 to 30 times slower
per iteration than native, so the browser solves a smaller game: two flop bet sizes and one
turn and river bet and raise size. Fast uses 8 fixed rivers under each turn; Accurate uses
every river. The local server keeps the full size menu and gives better postflop accuracy.) Pick the number of players (2 to 9), your seat
and stack, click cards into your
hand and the board (click a placed card in its slot to remove it; cards already
on the table can't be picked again), or type them as rank then suit (`Ah`, `Tc`).
The setup folds into one line once the hand starts, and the deck only shows
while a card is needed ("Show deck" opens it otherwise). Enter the action seat
by seat with the buttons or keys: F fold, C check or call, 1 to 3 preset sizes,
R custom size, A all-in, S the context shortcut, U undo. A table view shows who
has folded, who is to act, the board and the pot; clicking a seat folds
everyone before it, and a single context shortcut ("Fold to me", "Everyone
checks", "Everyone else folds") skips the boring bits.

Whenever it becomes your turn the advice panel solves the spot. The **Fast /
Accurate** toggle in the header (remembered between visits) picks the solve
time: Fast is 8 s in the browser and 2.5 s against the local server; Accurate
is about 60 s in the browser and 30 s locally, with more time for the earlier
streets used to narrow ranges. River spots are solid either way; turn spots need
Accurate to get close to a full solver (see Accuracy vs speed below).
The panel shows the recommended action with its size (preflop: 2.5bb
opens, 3bb from the small blind, 3x or 4x 3-bets, 2.3x 4-bets, then all-in;
postflop: the solver's bet and raise sizes in big blinds), the mixed strategy for your exact
hand with EVs, your whole range's totals at that spot (the percentage and number of
combos that fold, call, check, bet or raise each size, with a bar), and a 13x13 grid of
that range by hand with your hand boxed. Click an advice row to record it as what you did and keep going. Once the advice
is in, your action buttons switch to the solver's own sizes with its frequency on each,
so what you record is exactly what was solved. Postflop, a badge shows how well the solve
converged (exploitability in % of the pot: converged under 3%, fair under 10%, rough
above) with a one-click re-solve on Accurate; a rough solve never grades a line ✗.
When you're facing a bet the prompt shows the pot odds a call needs.
Solver stats and the villain's inferred range sit behind a "Details" toggle.
Each decision you record is marked against the advice that was showing (✓ in
the mix, ~ rarely played, ✗ outside it, with the EV given up), and when the
hand ends the advice panel lists them as a hand review with the EV given up per street.
Click any of your decisions (in the review or the timeline) to replay the hand from that
point; later actions and board cards are cleared, with Undo. While others act, the
last advice stays visible but dimmed and labelled. **4-colour** switches
diamonds to blue and clubs to green.
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
`b<amount>` bet, `ai` all-in. Amounts are in big blinds. `--ante 0.2` adds an ante
per player, `--straddle` has the first seat straddle 2bb.

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
big blinds. Optional `"ante"` (per player, same units) and `"straddle": true` (the
first seat posts 2bb and acts last preflop). Positions are the 6-max names `UTG HJ CO BTN SB BB`.

Response: per-action `freq` (the mixed strategy for hero's exact hand), `ev_bb`
(relative to folding now), `amount_bb`, a `recommended` action (highest
frequency, or highest EV when the solve is 3%+ of the pot from equilibrium and that
EV beats the top of the mix by over 1% of the pot), a `sampled` action (drawn from the mix, use this if you want to
actually play unexploitably), the range-wide frequencies, the villain's inferred
range, and solver stats.

`"opponents": "pool"` (the page's Opponents: Pool toggle) is exploit mode: opponents play like a
typical player pool instead of GTO. Preflop, hero's calls and raises are re-drawn against the pool's
ranges and responses; after the flop the advice is the best response (each hand's highest-EV action,
`freq` 0 or 1, with `gto_freq` alongside). See Exploit mode below.

## How it works

| Layer | File | Method |
|---|---|---|
| Cards, ranges | `gto/cards.py`, `gto/ranges.py` | 1326-combo weight vectors, PIO-style range syntax |
| Hand strength | `gto/evaluator.py` | numpy-vectorised 7-card evaluator built on treys' lookup tables: all 21 five-card subsets of every combo on a board in one pass, about 1 ms per board |
| Preflop | `gto/preflop.py` | Hand-written 100bb charts: RFI, vs RFI, vs 3-bet, vs 4-bet, cold vs 3-bet, mixed frequencies. Seats for 2 to 9 players map onto the charts by how many players are left to act (9-max LJ plays the 6-max UTG chart, 7+ handed UTG plays a tighter EP chart, heads-up has its own button and big-blind charts). Replaying the preflop action through the charts gives every player's range going to the flop. The charts assume a standard-sized raise and nobody else in the pot; when the price or the players in differ (a min-3bet, callers ahead), calls are re-drawn by preflop equity against everyone in versus the pot odds, held to the equity bar the chart's own continue range sets in the standard line. With callers in, raises are re-drawn too: a value raise stays while its equity against what continues (each player's chart facing the raise) clears break-even by as much as the chart's weakest value raise heads-up does (callers' dead money makes thinner value raises pay), bluffs shrink by how much less often everyone folds, and the default raise size adds the callers' chips (a squeeze over two callers of a 2.5bb open is 15bb from the BB). Off-chart games and depths: limpers get iso-raise / over-limp / big-blind-raise charts (and an open-limp range); antes play the open chart of a later seat and re-price calls; a straddle makes the straddler the big blind and the rest play by seats behind; stacks other than 100bb re-price calls with implied odds for small pairs and suited hands; a raise that commits the stack (40% of it, or under 1.5 pots behind once called) goes all-in, with the push/fold equilibrium as the range when shallower than the charts (and every open at 12bb or less). The opener calls a blind's 3bet wider (in position), the heads-up button has its own vs-3bet chart, and a 5bet that leaves chips behind can be shoved over. Every player's replayed preflop range goes through the same adjustments as hero's advice (so a cold call of a cheap 3bet or a multiway big-blind defence is wide), a limper facing a raise, an iso-raiser facing a limp-reraise, the heads-up big blind facing a 4bet and every seat (blinds included) facing a 7+ handed early-position open have their own charts; facing a 3bet cold depends on how wide the 3bet is (over an early or a late open) and on position against the 3bettor (out of position, half the cold calls go: premiums 4bet, the rest fold), and facing an all-in that someone already called, equity counts the callers. A player who called a raise and faces the next one (a flat facing a squeeze, a cold call of a 3bet facing the 4bet, a call of a 4bet facing the jam) continues with at least 30% / 35% / 55% of the range they called with: the charts for those spots are written for someone who hasn't put chips in and continue mostly with QQ+/AK, which a flat almost never holds, so they left a flatter's range nearly empty (2% of BTN's flats called a BB squeeze). The extra calls are its strongest folds, ranked by how readily the button continues against a 3bet (TT-77 and suited broadways before 22-55), then equity against the raiser. Once another raise is in after the call (sandwiched) the charts' folding stands. |
| Postflop | `gto/solver.py` | Heads-up range-vs-range **discounted CFR** over a bet tree (2 bet sizes per street, raises to half pot and full pot after the call, all-in when the stack is short or a raise size would exceed it, max 2 raises). Vectors are compressed to combos in either range. A river showdown is O(N) per board: combos sorted by strength, cumulative reach sums, and a per-card correction for card removal. Equity leaves are one matvec with a precomputed antisymmetric matrix. |
| Hand replay | `gto/advisor.py` | The whole hand is replayed for any number of seats: blinds, folds, dead money from players who folded, all-ins, acting order (heads-up the button acts first preflop and last postflop). Whenever exactly two players remain the spot is solved heads-up: from the start of the street if it began heads-up, or from the current node if a third player just folded. With three or more live players it falls back to a Monte Carlo heuristic against every live range (see Limits), and says so. |
| Multi-street | `gto/advisor.py` | Earlier streets are solved with a small budget and each player's range is narrowed by the frequency of the action they actually took, mixed with a 15% uniform floor so an approximate solve never prunes a line to nothing. The current street is solved from the start of the street and the tree is walked to hero's node with exact frequencies. Solutions are cached per hand so the flop solve is reused on the turn. |
| Exploit mode | `gto/exploit.py` | The pool is the equilibrium plus common leaks, frozen (it doesn't adapt to you): facing a first bet it calls small bets too often and folds too often to big turn and river bets (an extra 4-15% of the range, its weakest calls first; the extra calls are its strongest folds), it raises less (weak hands keep 30% of their raises, the rest call), and it bluffs less (bets from its weakest half are cut to 80/60/40% on flop/turn/river). Strength is absolute: the hand's percentile on the board, averaged over runouts before the river. Preflop it flats more and 3bets less (non-premium 3bets keep 35%, the rest flat, plus half again as many flats from its strongest folds, the big blind excepted) and 4bets only QQ+/AK. Each street is still solved to equilibrium; the pool's actions narrow its range through the leaky strategy, and hero's EVs come from a best response to it over the rest of the tree. Preflop, hero facing a raise goes through the off-chart re-pricing (see Preflop) with the pool's ranges and responses: calls by equity against what the pool actually raises with (a pool 4bet is QQ+/AK), value raises by equity against what the pool continues with, bluff raises scaled (up to 1.5x) by how often the pool folds compared with the chart. Facing a jam over a raise, the jam range is the pool's. Opens follow the chart (the pool 3bets less but flats more), hero's replayed range stays the chart's, and multiway postflop spots keep the betting rules. |
| Front ends | `gto/cli.py`, `gto/server.py`, `gto/static/index.html` | Manual CLI; local HTTP endpoint; click-driven review page served at `/`. |

### Accuracy vs speed

Checked against exact enumeration, toy games with known solutions, and the
open-source [postflop-solver](https://github.com/b-inary/postflop-solver) on
identical trees and ranges (BTN open vs BB call, K72r runouts):

* **Evaluator**: identical to treys on 18,000 random hands.
* **River**: matches the reference to about 1% in every node's aggregate
  frequencies and converges in well under a second. The polarized river game
  (nuts or air vs a bluff-catcher) reproduces the theory: bluffs are b/(1+2b) of
  the bets and the catcher calls 1/(1+b).
* **Turn**: every river is in the tree (`n_rivers=None`), so the turn is solved
  exactly for the bet-size abstraction. Run long enough it matches the
  reference, but it takes time: about 20 s to reach ~3% of the pot
  exploitability on wide ranges and about 60 s to reach ~0.6%. At 2.5 to 8 s the
  answer is only roughly right. `river_sample=k` traverses k fresh random rivers
  per iteration instead (public chance sampling). That makes each iteration
  cheaper and stays unbiased, but it is noisy: in the ~50 iterations the browser
  gets in 8 s, BB donked 20-37% on K72r-T and J72-5 where the answer is 0-2%.
  A small fixed river set (`n_rivers=k`) is biased once converged (the solver plays
  a game where only those k rivers can come; on K72r-T it donk-bet 32% where the
  real answer is 6%), but under a tight time budget it was the least wrong option
  (10-15%), so only the browser's Fast mode uses it (8 rivers). The browser's
  Accurate mode enumerates every river with one turn/river size (donk 1-4%).
* **Flop**: flop decisions are full, turn decisions are full on `n_turns` sampled
  turns (default 6), and the river is an **equity leaf**: both hands check it
  down. This is the main remaining error. Against the full-tree reference on K72r
  (one 50% size), BB's check frequency is right (~100%), but BTN c-bets 57-65%
  where the full game c-bets ~100%. Turn sampling is not the cause: all 49 turns
  give the same result. The check-down leaf undervalues betting and position.
  Treat flop output as directional.
* `Solution.exploitability()` reports how far a solve is from equilibrium in
  % of the pot (it plays a best response for each side against the other).
* **Earlier streets** used for range narrowing get 2 s each, and a solve that is
  reused later in the same hand keeps iterating from its stored regrets instead of
  restarting. When a real bet or raise size differs from the abstraction by more
  than 15%, that street's tree uses the real size so narrowing is exact.
* Multiway pots (3+ players live at your decision) are not solved. Instead a Monte
  Carlo heuristic picks the highest-EV action among fold / call / raise to 75% pot
  (or check / bet 66% pot), with opponents following simple range rules:
  facing a bet they fold together a quarter less often than a balanced defence
  (people call too much), with everyone keeping the hands above one strength bar, so a
  weak range (a lead after checking twice) folds more of itself than a strong one (a
  call of that lead), the strongest part of what continues raises, and when
  they bet it's value plus balanced bluffs. Players still to act behind you can
  call or raise after you call, a raise can be re-raised, and every line plays to
  the river: on each later street you lead for value when you're ahead of what
  would call, otherwise an opponent may bet and you raise, call or fold by your
  equity against their range (on the turn, also counting the river bet that may
  follow). Trials continue until the gap between the two best actions is clear of
  the sampling noise (or a time cap), and the notes give that gap with its error.
* Streets that started multiway narrow every player's range by the same rules
  (a check drops most hands that would have bet, a bet keeps its value + bluff
  range, a call drops what would have folded or raised); hands a rule rules out
  keep 30% of their weight. That covers multiway spots and heads-up pots that
  were multiway earlier, including a pot that became heads-up on this street.
* In heads-up spots the villain's actions this street narrow their range exactly,
  unless the solve takes that line with under 10% of their range; then it is
  smoothed like earlier streets and the advice says so (EVs are then against the
  wider range, so actions the solver mixes can show different EVs).
* The rule constants (value share of a bet, share of continuing hands that raise,
  lead threshold, narrowing softness) are fitted to the heads-up solver with
  `scripts/calibrate_multiway.py`: on ~200 heads-up flop spots the heuristic's pick
  gives up 0.10 bb on average against the solver's best action, and its EVs are
  off by 1.0 bb on average (1.35 bb before fitting). The "players call too much"
  shade is deliberate and not fitted, and nothing here checks the multiway
  interactions themselves, which the solver can't play.
* If your hand isn't in your range for the line you took (a preflop call the charts
  never make, say), it is added to the solve as a near-zero-weight combo so it
  still gets a strategy and EVs without changing anyone else's play.
* If a preflop line still leaves a player with no range at all (a flat the charts
  never 4bet from, then a 4bet; a call sandwiched between a squeeze and a 4bet), that
  range is rebuilt from the hands that take the last such action. Otherwise the solve plays against a face-up hand: BB would fold
  everything to a jam with KK and pay off nothing smaller.

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

* Preflop charts are approximations written by hand, not solver output, and the
  off-chart adjustments are rules on top of them. `scripts/preflop_sweep.py` runs
  every standard line for 2-9 players and flags broken frequencies and any hand continuing less often than one it dominates (pairs, kickers, suited over offsuit; the strategy is also kept that way at run time); a test keeps
  key 6-max continue rates near typical solver output. Replace them with real chart
  exports if you have them; the format is a range string.
* Limps, antes, straddles and stack depth are handled by adjusting the 100bb
  no-ante charts (see Preflop above), not by charts built for those games.
* Exploit mode's pool is a judgment call about common player pools, not fitted to hand
  histories, and on a rough (Fast, flop or turn) solve the best response also picks up the
  solve's own noise.
* One effective stack for the table.
* The bet-size abstraction is coarse; real actions are mapped to the nearest size.
