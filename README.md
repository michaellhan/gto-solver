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
node --test 'tests/js/*.test.js'
```

The Node tests (Node 22+, no npm packages) cover the page's JavaScript: `tests/js/page.js` loads the
inline script of `gto/static/index.html` into a `vm` context with a minimal fake DOM and drives it
like a user (buttons, keys, the setup inputs): the hand engine (acting order for 2 to 9 players,
blinds, antes, straddle, pot and bet sizes, validation, undo and rewind, per-seat stacks with side pots and
incomplete raises), share links (`#h=`) and the
copied hand text, and the state sent to the solver (its keys are checked against what
`Advisor._advise` reads). They also write `tests/data/page_states.json`, random hands played through
the page (some with per-seat stacks); `tests/test_page_states.py` replays each spot with the advisor's own `replay_hand` (same
player to act, pot, bet and players in) and asks the advisor about a sample of them. After a change to
the page's engine, regenerate it with `UPDATE_FIXTURES=1 node --test 'tests/js/*.test.js'`.

## Review UI

```bash
python -m gto.server --port 18877
```

Open http://127.0.0.1:18877/ . (The same page is what is deployed to the
website; there it detects that no server is present and runs the identical
Python code in a web worker with Pyodide. Pyodide's numpy is 10 to 30 times slower
per iteration than native, so the browser solves a smaller game: two flop bet sizes and one
turn and river bet and raise size. Fast uses 8 fixed rivers under each turn; Accurate uses
every river. The local server (`gto/native.py`) runs Fast with the same sizes and every river, and
Accurate with two turn and river bet sizes.) Pick the number of players (2 to 9), your seat
and the stacks (one for the table; **per seat…** gives any seat its own, shown beside its name on
the table), click cards into your
hand and the board (click a placed card in its slot to remove it; cards already
on the table can't be picked again), or type them as rank then suit (`Ah`, `Tc`).
The setup folds into one line once the hand starts, and the deck only shows
while a card is needed ("Show deck" opens it otherwise). Enter the action seat
by seat with the buttons or keys: F fold, C check or call, 1 to 3 preset sizes
(preflop the advisor's default raise, limpers and callers included),
R custom size, A all-in, S the context shortcut, U undo. A table view shows who
has folded, who is to act, the board and the pot; clicking a seat folds
everyone before it (checking where there's nothing to call, and stopping at your own turn),
and a single context shortcut ("Fold to me", "Everyone checks", "Everyone else folds",
which checks a player who has nothing to call) skips the boring bits.
The engine plays no-limit rules with each seat's chips: a call or all-in for less puts in what the
player has, a raise is at least the last full raise, an all-in raise for less than that doesn't reopen
the betting to players who already acted, a bet nobody can match in full comes back down to the most
anyone else put in, the action ends once at most one player with chips is left, and the table shows
the main pot and side pots.

Whenever it becomes your turn the advice panel solves the spot. The **Fast /
Accurate** toggle in the header (remembered between visits) picks the solve
time: Fast is 8 s in the browser and 2.5 s against the local server (2.5 s per earlier street too); Accurate
is about 60 s in the browser and 30 s locally (12 s per earlier street), with more time for the earlier
streets used to narrow ranges. River spots are solid either way; turn spots need
Accurate to get close to a full solver (see Accuracy vs speed below).
The panel shows the recommended action with its size (preflop: 2.5bb
opens, 3bb from the small blind, 3x or 4x 3-bets, 2.3x 4-bets, then all-in; over limpers
3x (4x from the blinds) plus 1bb a limper, and a squeeze adds one bet per caller;
postflop: the solver's bet and raise sizes in big blinds), the mixed strategy for your exact
hand with EVs, your whole range's totals at that spot (the percentage and number of
combos that fold, call, check, bet or raise each size, with a bar), and a 13x13 grid of
that range by hand with your hand boxed. Click an advice row to record it as what you did and keep going. Once the advice
is in, your action buttons switch to the solver's own sizes with its frequency on each,
so what you record is exactly what was solved. Postflop, a badge shows how well the solve
converged (exploitability in % of the pot: converged under 3%, fair under 10%, rough
above) with a one-click re-solve on Accurate; a rough solve never grades a line ✗.
When you're facing a bet the prompt shows the pot odds a call needs.
Solver stats and the villain's inferred range (as text and as a 13x13 grid shaded by weight) sit behind a "Details" toggle.
Each decision you record is marked against the advice that was showing (✓ in
the mix, ~ rarely played, ✗ outside it, with the EV given up), and when the
hand ends the advice panel lists them as a hand review with the EV given up per street.
Click any of your decisions (in the review or the timeline) to replay the hand from that
point; later actions and board cards are cleared, with Undo. While others act, the
last advice stays visible but dimmed and labelled. The **⋯** menu has a four-colour
deck (diamonds blue, clubs green), a switch that puts the advice on the left, and Reset layout.

**Import hand** (in the ⋯ menu) takes a pasted hand history: PokerStars and GGPoker/Natural8
(other sites in the same "Seat N: name (stack)" / "name: raises 2 to 3" shape mostly parse too),
2 to 9 players, cash or tournament. Seats map onto the page's positions from the big blind, amounts
become big blinds, antes are averaged per player, a 2bb straddle from the first seat is kept, sitting-out
seats are dropped and returned bets are ignored; the history needs your hole cards. Every seat keeps its
own stack (after the antes, as the history has it), so a short all-in, its side pot and an uncalled bet play
out as they did. The parser is `gto/static/hh.js`,
loaded only when the dialog opens. **Review whole hand** (offered after an import, and at the end of any
hand with ungraded decisions) solves each of your decisions in order at the current Fast/Accurate setting,
grades it as if it had been entered live and ends on the hand review; it shows "decision 2 of 5" while it
runs and can be cancelled, keeping the grades so far.

The layout fits the window at any zoom. On a wide window each column scrolls on its own and
the action buttons stay pinned to the bottom of theirs; drag the divider between the columns
to resize them, drag the bar under the table to resize it, and click the Table or Hand title
to fold that panel (all remembered in the browser). On a narrow window (a phone, or a zoomed-in
page) it's one column with the action buttons pinned to the bottom of the screen. While a card
is needed the deck moves up under the setup; otherwise it's hidden behind **Edit cards**.
**Copy hand** puts a plain-text summary on the clipboard: the full action, the
advice with frequencies, EVs and sizes, the villain range, solver stats, a link
that reopens the exact spot, and the raw state JSON for the API or CLI.
The hand is kept in the browser between reloads; **New hand** clears it.

**Practice** in the header plays random hands against GTO opponents, with the player count,
stacks (each seat's own, if set), antes, straddle and rake of the setup; a showdown pays each side pot
to the best hand among the players in it. Your seat and cards are random; every other seat's
hand and the board are dealt up front, and the other seats move by themselves (a short "thinking"
status while they do): preflop from the chart for their actual hand, after the flop drawn from the
solve's mix for it (heads-up, the same solve that answers your own spot, so a reply to your move is
nearly instant; multiway the advisor's heuristic from their seat). On your turn the advice stays
hidden (the spot is solved in the background so your move can be graded) and shows once you've
acted; each decision is graded as in a review. At the end the opponents' cards are shown at
showdown with your result, the hand review lists the EV given up, and **Next hand** (N) deals again.
The running score (hands, average EV given up per decision, how often you took the solver's top
action, chips won) is kept in the browser. Folding ends the hand for you; Undo and replaying a
decision are off while practising. The page loads `practice.js` the first time Practice is turned on.

## Use from the shell

Manual entry:

```bash
python -m gto.cli --hero BB --hole 9h8h --board Ks7d2cTh \
  --preflop "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c" --flop "BB:x,CO:b2,BB:c"
```

Action tokens are `POS:act` with `f` fold, `x` check, `c` call, `r<to>` raise to,
`b<amount>` bet, `ai` all-in. Amounts are in big blinds. `--ante 0.2` adds an ante
per player, `--straddle` has the first seat straddle 2bb, `--rake 0.05 --rake-cap 3` sets a 5% rake capped at 3bb (see Rake).
`--speed accurate` uses the server's Accurate tree and budgets (default `fast`; `--budget` / `--prior-budget` override).

Server for your sandbox table:

```bash
python -m gto.server --port 8765
curl -s -X POST localhost:8765/advise -d @examples/flop.json | python -m json.tool
```

`POST /play` with `{"state": <request body>, "seat": "BB", "hole": "8c8d", "seed": 7}` answers a GTO
move for that seat holding those cards (practice opponents: `Advisor.play`), drawn from its mix, as
`{"pos", "action", "to" (bets and raises), "name", "freq", "actions"}`; `seed` is optional.

Request body (see `examples/`):

```json
{
  "hero": "BTN", "hole": "AhQd", "board": "Qs8d3c",
  "bb": 1.0,
  "stack": 100, "stacks": {"BB": 40},
  "preflop": [{"pos": "CO", "action": "fold"}, {"pos": "BTN", "action": "raise", "to": 2.5},
              {"pos": "SB", "action": "fold"}, {"pos": "BB", "action": "call"}],
  "flop": [{"pos": "BB", "action": "check"}], "turn": [], "river": []
}
```

Set `bb` to your blind size and pass amounts in chips, or leave it at 1 and pass
big blinds. `stack` is everyone's stack; the optional `stacks` gives seats their own (seats left out
have `stack`, or without one the smallest of `stacks`). Each player's chips cap what they can put in
(all-ins for less, side pots, incomplete raises, as on the page); hero plays preflop at the effective
stack against the deepest opponent who hasn't folded (the push/fold and chart depth, implied odds and
commitment rules), each opponent's replayed range at theirs, and heads-up postflop at the smaller of
the two stacks. Optional `"ante"` (per player, same units) and `"straddle": true` (the
first seat posts 2bb and acts last preflop). Optional `"rake"` (a share of the pot, `0.05` for 5%)
and `"rake_cap"` (the most rake per pot, same units as the amounts; leave it out for no cap): off
unless given (see Rake below). Positions are the 6-max names `UTG HJ CO BTN SB BB`.
Optional `"sizes"`, `"ranges"`, `"optimize_sizes"` and `"locks"` change the postflop game (see Custom sizes and ranges below).

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

### Rake

Off by default (the page's Rake select: off, 5% capped at 3bb, or a custom percent and cap; the
setting is saved with the hand and carried by share links). Rake is taken "no flop, no drop": a pot
that ends preflop isn't raked, so opens, steals and folds to them are unchanged.

* Postflop, the solver's terminal payoffs are raked: the winner pays rake = min(rate x pot, cap) out
  of the pot it collects, at showdowns, equity leaves and folds (the pot without an uncalled bet,
  which goes back to the bettor; a split pot costs each player half). An equity leaf where the
  stacks are in is raked on its final pot. One where the tree stops early (a street it doesn't play
  out) is raked on the pot as it stands: the leaf pays equity times that pot, leaving out the bets
  later streets would add, so raking the same pot keeps the rake the right share of what it pays
  (exact without a cap; with one, a bigger final pot would pay a smaller share, so it's a little
  high there, while in bb it's a little low). Raking a guessed final pot would charge rake on chips
  the leaf never pays out. The exploitability shown counts what each best response
  gains over its own raked payoff (a raked game isn't zero-sum). With no rake the numbers are
  bit-identical to the unraked solver.
* Preflop, the charts are fixed and calls that see a flop are re-priced through the off-chart
  re-pricing (see Preflop): the pot a call would win is short the rake it goes on to pay, taken on
  2.5x the pot the call takes to the flop (later streets grow it; no more than the stacks allow), to
  the cap. 3bets and 4bets aren't re-priced (they often win the pot preflop, where nothing is raked).
  The big blind's defence of an open and the opener's calls of a blind's 3bet in position (UTG, HJ, CO
  and the button) are charts written without rake and pay all of it. The other calls (the opener out
  of position facing a 3bet, cold calls of an open or a 3bet, calls of a 4bet) were fit to published
  raked solves, so they pay only the rake beyond gtopreflop.com's 5% capped at 4bb (they change with
  a higher rate or no cap; re-pricing them at that rake would count it twice: CO against a button 3bet
  already folds/calls/4bets 63/18/19 against gtopreflop.com's raked 64.0/16.9/19.2). The 2.5x is fit
  to gtopreflop.com's raked solve (8-max 100bb, 2.5bb opens, 12bb blind 3bets, 5% capped at 4bb;
  https://gtopreflop.com/posts/4-bet-ranges-guide and /posts/big-blind-defense), as calls (share of
  the opening range, or of hands for the big blind), unraked chart / 5% capped at 3bb or 4bb (the
  same here: the cap doesn't bind) / gtopreflop.com: CO against a BB 3bet 40.7 / 23.8 / 22.4, the
  button against a BB 3bet 34.4 / 20.3 / 20.0, against an SB 3bet 28.9 / 17.1 / 13.9, UTG against an SB
  3bet 35.5 / 24.5 / 14.1 (an 8-max UTG); the big blind against a CO open 37.0 / 22.9 / 23.8, the
  button 53.1 / 36.3 / 29.8 (defending 47.8% instead of 64.6%), a 3bb small-blind open 55.6 / 35.8 /
  33.8 (to 10bb there), UTG 28.2 / 18.2 / 14.6 (against the LJ). The old flop-pot rake had the big
  blind calling 48.4% against the button. Lower rake does less: 3% capped at 1bb has the big blind
  calling 44.3% against the button and CO calling 30.4% against a BB 3bet; 7% uncapped 28.4% and
  21.7%. More rake (a higher rate or cap) never adds a call; with no rake the advice is bit-identical.
  A called preflop all-in sees the flop, so its price counts the rake on its pot. The replayed ranges
  going to the flop use the same calls.
* Push/fold: the equilibrium behind short-stacked jams (hero's own shove, and the jam range hero
  faces) rakes a called jam's pot, since it runs out a board and so sees the flop, as the call of a
  preflop all-in is priced; a jam that takes the blinds uncalled isn't raked. Rake costs the caller
  (the pot it wins is short the rake) while the jam's fold equity is untouched, so callers tighten
  and the jam range widens: 10bb deep, 5% capped at 3bb, the small blind jams 60% of hands instead of
  58% and the big blind's K5s calls 74% instead of always.
* Multiway (three or more players see the flop): the Monte Carlo heuristic pays every pot hero wins
  less its rake (the pot without an uncalled bet; a split costs hero its share), so the EVs it ranks
  actions by are raked. The opponents' betting rules and hero's own call/lead/raise rules within each
  trial stay unraked.

### Custom sizes and ranges

The page's **Advanced** disclosure in the setup (saved with the hand and carried by share links, like rake;
sent from the flop on, so preflop advice is unchanged). All are off unless given.

* `"sizes": {"flop": [33, 75], "turn": [...], "river": [...], "raise": [60]}`: the postflop bet sizes per
  street in % of the pot, and the raise sizes (all streets) in % of the pot after calling, replacing the
  solver's menu for that request. At most 3 a street (each one multiplies the game tree; the browser's
  default is 2 flop sizes and 1 turn and river size), held to 5-500%. A size actually taken still replaces the
  nearest one in the tree (more than 15% off), so a villain's off-menu bet is solved at its real size.
* `"ranges": {"BB": "22+, A2s+, KTo+, QJo:0.5"}`: a player's range entering the flop (PIO style, as
  `gto/ranges.py` reads it), instead of replaying the preflop action through the charts; heads-up (later
  streets narrow it by the solve as usual) and multiway (the betting rules narrow it). A range that can't be
  read, an unknown seat, or one with no combos left on the board is an error; an empty one is the chart's.
* `"optimize_sizes": true`: where hero can bet (nothing to call) on a street that began heads-up, a first
  solve of the street with a wide menu (25, 33, 50, 75, 100, 150, 200% and all-in where the stacks allow;
  one street deep and one raise, so the size choice converges fast) keeps the sizes either player's first bet
  uses with 3%+ of their range, the 3 most used, and the advice is solved with those. The two solves split
  the time budget (35% / 65%), so a spot takes as long as without it. The response adds `size_usage`
  (hero's mix at that node over the wide menu, `{"check": 0.13, "25%": 0.47, ...}`, summing to 1) and
  `sizes_kept`. A kept overbet makes the second tree bigger, so it settles slower: use Accurate.

* `"locks": {"BB": {"fold": 15, "bluff": -50, "next_fold": 60}}`: nodelocking, your read of an opponent
  (the page: a profile per opponent, GTO, Calling station, Nit, Maniac, Overfolds to bets or Custom, whose
  numbers can be edited). Heads-up after the flop the advice is then the best response to their locked
  strategy, as in exploit mode (`freq` 0 or 1 with `gto_freq` alongside, ties within the solve's noise
  mixed), with `"locked": {"BB": {...}}` and a note; on top of GTO, or of the pool with `"opponents": "pool"`.
  Every key is optional (`gto/exploit.py` LOCK_KEYS, TendencyLock):
  * `fold` (-100..100, also `fold_flop` / `fold_turn` / `fold_river`, added): facing a first bet (not a raise
    of their own bet), this many points more of their range folds, their weakest calls first (at most every
    call; raises stay); negative, their strongest folds call.
  * `bluff` (-100..100): their weak hands' bets (below the 50th percentile of strength on the board, tapering
    to the 65th, as the pool's bluffs): -100 never bluff (value kept), +100 every weak hand bets.
  * `bet`, `raise` (-100..100): every hand's betting when they can bet, and raising facing a bet: -50 halves
    it (into checks and calls), +50 moves half of their checks or calls into it, split over the sizes as they
    use them now.
  * `next_fold`, `next_raise` (0..100): their fold and raise % at their next decision after each of your bets
    or raises this street (folds by strength, raises scaled); `next_bet` (0..100): their bet % when you check.
    The classic nodelock: "they fold 60% to my bet here". Notes give each, locked and at equilibrium.

  The profile keys apply wherever they act, on later streets too, and narrow their range through what they did
  earlier in the hand. Preflop and multiway pots (the betting rules) ignore them, with a note; an unknown seat
  or key, your own seat, a non-number or a value out of bounds is an error.

Errors come back as `{"error": "..."}` (e.g. `sizes: at most 3 flop sizes ...`, `ranges: can't read BB's
range ...`). Heads-up responses also carry `villain_grid` (the villain's range by hand class: `cells` with
`w` the average combo weight), which the page draws under Details.

## How it works

| Layer | File | Method |
|---|---|---|
| Cards, ranges | `gto/cards.py`, `gto/ranges.py` | 1326-combo weight vectors, PIO-style range syntax |
| Hand strength | `gto/evaluator.py` | numpy-vectorised 7-card evaluator built on treys' lookup tables: all 21 five-card subsets of every combo on a board in one pass, about 1 ms per board |
| Preflop | `gto/preflop.py` | Hand-written 100bb charts: RFI, vs RFI, vs 3-bet, vs 4-bet, cold vs 3-bet, mixed frequencies. Where published solves exist the charts are matched to them, checked by `scripts/preflop_reference.py` (tests/preflop_refs.py: 22 spots, each played at the rake its solve used, and 14 hand-level facts): opens of 18.7 / 22.9 / 30.0 / 45.1 / 44.7% from UTG to SB (raked 2.5bb solves from pokalab, Poker Academy, GTO Ranges+ and gtopreflop.com agree on 17.6-18.7 / 22.3-23.3 / 29.1-30.6 / 43.3-46.5 / 43.5-45.7; were 16.7 / 19.4 / 27.2 / 45.6 / 50.2, with 22-44 half-opened from UTG and A2o from the BTN), the big blind's defence against the button (58.2/30.2/11.6 fold/call/3bet at gtopreflop.com's 5% capped at 4bb against its 58.7/29.8/11.5 and Poker Academy's 59.5/29.0/11.5; was 52.2/36.4/11.5; unraked it defends 55% of hands, was 64.6%; its 3bets polarised with offsuit wheel aces and suited kings, A4s-A2s calling), the small blind against CO and BTN opens (88/4/8 and 85/4/11 against 8.2% 3bets and 83.7/4.7/11.5; was 80/6/13 against both), UTG against a BTN 3bet (56/25/19 of its opens against 57.0/24.4/18.7, QQ mostly calling) and the small blind against the big blind's 3bet (64/20/16 against 63.6/21.4/15.0 at 10bb, now the default size; was 39/45/16, the chart written for 12bb). Average error over the spots 1.0 points (was 2.9, 10 of 22 outside tolerance). Not found published: 4bets onward, squeezes, cold calls, heads-up. Seats for 2 to 9 players map onto the charts by how many players are left to act (9-max LJ plays the 6-max UTG chart, 7+ handed UTG plays a tighter EP chart, heads-up has its own button and big-blind charts). Replaying the preflop action through the charts gives every player's range going to the flop. The charts assume a standard-sized raise and nobody else in the pot; when the price or the players in differ (a min-3bet, callers ahead), calls are re-drawn by preflop equity against everyone in versus the pot odds, held to the equity bar the chart's own continue range sets in the standard line. With callers in, raises are re-drawn too: a value raise stays while its equity against what continues (each player's chart facing the raise) clears break-even by as much as the chart's weakest value raise heads-up does (callers' dead money makes thinner value raises pay), bluffs shrink by how much less often everyone folds against how often a bluff of that size needs them to (the callers' dead money lowers that bar), and the default raise size adds the callers' chips (a squeeze over two callers of a 2.5bb open is 15bb from the BB). Off-chart games and depths: limpers get iso-raise / over-limp / big-blind-raise charts (and an open-limp range); the iso-raises over one limper are a little tighter than the seat's open, at published widths (HJ 18%, CO 24%, BTN 36%, SB 19% of hands, the big blind 15%, against Preflop Wizard's and Grindlab's ~15-22%, 22-35%, 30-45%, 18-25% and ~15%; the old charts raised 10/13/16/10/8%; no solver frequencies for cash iso-raises were found: GTO Wizard's multiway preflop solver offers only iso-raise or fold facing a limp outside the blinds, and solved 6-max packs are mostly no-limp trees), about 0.6x that over two or more limpers (value-heavy, the dropped pairs and suited hands over-limping); antes play the open chart of a later seat and re-price calls; a straddle makes the straddler the big blind and the rest play by seats behind; stacks other than 100bb re-price calls with implied odds for small pairs and suited hands (from 70bb down, the small blind facing a CO or BTN open plays a 40bb chart instead, re-priced from 40bb: shallow, the hands under its 3bet value core flat, as in Poker Academy's 40bb solves where JTs, KJs, KTs, 88 and AQo call or jam; the 100bb chart thinned by implied odds left a 40bb flat of ~45 combos of 33-77 and A6s-A9s, now ~84 of suited broadways, 44-99, AQo/AJo/KQo and some suited aces, at 81/6/13 fold/call/3bet against the 100bb chart's 80/6/13); a raise that commits the stack (40% of it, or under 1.5 pots behind once called; 1.25 pots at the charts' own depth, where a 4bet to 2.3x leaves 1.45 and is called, as in 100bb solves) goes all-in, with the push/fold equilibrium as the range when shallower than the charts (and every open at 12bb or less); a jam over an open or limpers competes with flatting (equity against the raiser realized at the rate that makes the chart's weakest flats break even, plus implied odds) or the big blind's free check, not just with folding (the big blind rejammed a button open with 30% of hands at 25bb, now 22.5%). Preflop equity against several players is pooled per hand class over 400 boards weighted by how likely the players' ranges leave them (within 1-2 points of a dealt-out simulation five-way), and off-chart calls expect implied odds from the extra players by the square root of how many. Out of position facing a 3bet (a later seat's, or the big blind's against the small blind) the opener plays charts matched to published solver output: CO against a BTN 3bet to 7.5bb folds 63.7%, calls 17.2% and 4bets 19.1% of its opening range (Poker Academy, 100bb cash), openers respond in similar proportions from every seat (GTO Wizard), and the small blind against the big blind's 3bet to 10bb folds 63.6%, calls 21.4% and 4bets 15.0% (gtopreflop.com); the earlier charts called about twice that and 4bet half as often (CO 58/33/9). Also matched to published solver frequencies (100bb cash, 2.5bb opens): HJ and CO against a UTG open mostly 3bet or fold (90/2.5/7 fold/call/3bet of hands, against HJ 91.1/1.5/7.4 from Poker Academy and CO calling 1.4-3.2% and 3betting 8-9% from GTO Wizard; was 88/8/3), the button against a CO open (85/3/12, against 84.5/3.0/12.5 from Poker Academy; was 82/9/9), the big blind's 3bets against the button and the small blind (11.6% and 16.4% of hands, as Poker Academy and gtopreflop.com have them; were 8.5% and 11.7%), and the button facing the big blind's 3bet (54/34/12 fold/call/4bet of its opening range, against 54.6/33.9/11.5 from 888poker; was 59/35/6, a 4bet range of nearly all value) and the small blind's (60/29/11.5: gtopreflop.com's raked solve has the button 4betting as often against either blind, 10.4% and 10.3%, and calling 6 points less against the SB, with QQ, AKo, A9s and KTs mostly 4betting and AQs/KQs calling; was 58/36/6). The big blind's 3bets against UTG, HJ and CO are 4.8%, 6.2% and 8.0% of hands (gtopreflop.com 4.5/5.9/7.7, Poker Academy 4.5% against the LJ, pokalab 5.1/6.7, BeyondGTO 6/8/10; were 3.6/3.6/5.6, the HJ spot reusing the UTG chart), taken from calls so the defence totals stay; its squeezes follow (over CO + BTN 7.3%, was 4.9%; over UTG + a caller 4.5%, was 3.4%), about 0.9x the heads-up 3bet where GTO Wizard's NL500 cash solve at 40bb has 0.95x. Cold players facing an open and a 3bet fold 96-97% (GTO Wizard: over 90%, up to 97%) and the big blind's squeeze over a button open and a small-blind call is 10% of hands (Poker Academy 12.7%): both left. No cash solves of the small blind's or button's squeezes were found (gtopreflop.com's database has no node facing two players in the pot; the tables that turn up are labelled approximations). The opener calls a blind's 3bet wider (in position), the heads-up button has its own vs-3bet chart, and a 5bet that leaves chips behind can be shoved over. Every player's replayed preflop range goes through the same adjustments as hero's advice (so a cold call of a cheap 3bet or a multiway big-blind defence is wide), a limper facing a raise, an iso-raiser facing a limp-reraise, the heads-up big blind facing a 4bet and every seat (blinds included) facing a 7+ handed early-position open have their own charts; facing a 3bet cold depends on how wide the 3bet is (over an early or a late open) and on position against the 3bettor (out of position, half the cold calls go: premiums 4bet, the rest fold), and facing an all-in that someone already called, equity counts the callers. A player who called a raise and faces the next one (a flat facing a squeeze, a cold call of a 3bet facing the 4bet, a call of a 4bet facing the jam) continues with at least 30% / 35% / 55% of the range they called with: the charts for those spots are written for someone who hasn't put chips in and continue mostly with QQ+/AK, which a flat almost never holds, so they left a flatter's range nearly empty (2% of BTN's flats called a BB squeeze). Those shares are floors, set against the tightest raises the charts make (the big blind's old 3.6% 3bet over UTG, UTG's 4bet, the 3bettor's jam); a wider, more bluff-heavy raise is defended more: each point of equity the caller's range gains against the actual raise over that reference adds 2.5 points (BTN's flat facing the big blind's published 8% squeeze over CO continues 44% and calls with 77; with a fixed 30% the wider squeeze had pushed 77 out). The extra calls are its strongest folds, ranked by how readily the button continues against a 3bet (TT-77 and suited broadways before 22-55), then equity against the raiser. Once another raise is in after the call (sandwiched) the charts' folding stands. The small blind flats an early open rarely (about 3% of hands) but not only middling hands: a share of JJ, TT, AQs, AJs and KQs flat as traps (published summaries have it sometimes calling JJ and TT there), where the old chart 3bet them all, so its flat-then-call of a big-blind squeeze was 13 combos of 77-99 and the squeezer's QQ won every A-K-x flop; now about 12 combos of 99-JJ, some 77-88 and those suited broadways. The 3bettor facing a 4bet uses a chart written against the opener's 4bet, which carries bluffs; against a cold 4bet or one from a player who flatted first (nearly all value) each hand is re-drawn by its equity against the actual 4bet range versus the opener's: value hands keep continuing while their equity clears the price of calling over the 80% of it a call realizes (one bar for every spot, so JJ calls 53% against a cold 4bet over UTG, where it has 33%, and always over the button, where it has 36%), and marginal and blocker hands shrink by how far they drop (AQs calls 5%). Bluff jams, and the value jams the chart mixes with calls, scale by how often that range folds to a jam compared with the opener's: against its value a jam and a call both get the stacks in, so a jam's edge is the equity it denies the bluffs, while a call can still fold bad flops. The 4bettor folds QQ and AKo to the charts' tight jam ranges whoever it is, so against a cold 4bet the jams hardly change (QQ 56%, AK 47%); against a flat-then-4bet over an early open, which folds far less, QQ jams 10% and AK 7%, calling the rest; KK+ and AKs keep jamming. A 4bettor facing a 5bet jam calls by equity against the jammer's replayed range versus the pot odds, like hero's advice (the fixed chart called QQ and AK 70% whoever jammed, where they have 36% against a standard jam and need 41%). |
| Postflop | `gto/solver.py` | Heads-up range-vs-range **discounted CFR** over a bet tree (2 bet sizes per street, raises to half pot and full pot after the call, all-in when the stack is short or a raise size would exceed it, max 2 raises). Vectors are compressed to combos in either range. A river showdown is O(N) per board: combos sorted by strength, cumulative reach sums, and a per-card correction for card removal. Equity leaves are one matvec with a precomputed antisymmetric matrix; natively a pass first collects every leaf's opponent reach and multiplies each leaf matrix stack by all of them in one matrix-matrix product (one product per leaf was memory-bound). Subtrees below the root street with the same shape (the turn trees after most flop lines differ only in pot size) are merged into one batch and traversed once, so a flop solve walks ~30 decision nodes a pass instead of ~90, with the same numbers (~1.6x natively). In the browser, whose numpy has no BLAS, the leaf matvecs are most of the time: they run through einsum there, on leaf matrices trimmed to each player's range, and per-card sums replace the small matrix products. DCFR with gamma 4 (average weighted to recent iterations). From the flop the river beyond `depth` keeps its betting, played on the turn's boards, and the turns are one card per rank class (see Accuracy vs speed). `scripts/bench_solver.py --curve` plots exploitability against time, natively or in Pyodide (`scripts/pyodide_run.mjs`). |
| Hand replay | `gto/advisor.py` | The whole hand is replayed for any number of seats: blinds, folds, dead money from players who folded, all-ins, acting order (heads-up the button acts first preflop and last postflop). Whenever exactly two players remain the spot is solved heads-up: from the start of the street if it began heads-up, or from the current node if a third player just folded. With three or more live players it falls back to a Monte Carlo heuristic against every live range (see Limits), and says so. |
| Multi-street | `gto/advisor.py` | Earlier streets are solved with a small budget and each player's range is narrowed by the frequency of the action they actually took, mixed with a 15% uniform floor so an approximate solve never prunes a line to nothing, and followed only as far as the solve has settled (the rest by the betting rules' range for the action, see Accuracy vs speed). A range entering a later street with fewer than 8 live combos (preflop.MIN_LINE_COMBOS, counting the cards dealt since) is filled out with the rest of the range the player started the street with, weighted as preflop (1 - combos / 8): a 4bet pot's ~12-combo range checking back the flop left 2 for the turn, so the turn solve played a face-up hand. The current street is solved from the start of the street and the tree is walked to hero's node with exact frequencies. Solutions are cached per hand so the flop solve is reused on the turn. |
| Preflop subgame (experimental) | `gto/pf_solve.py` | Not used by the advice yet. Solves the 3bet pot heads-up as a game: the opener facing the 3bet folds, calls or 4bets; the 3bettor facing the 4bet folds, calls or jams; the opener facing the jam folds or calls. Hands are the 169 classes with class-level card removal; CFR+ with linear averaging converges to under 0.01% of the pot from equilibrium in 1000 iterations (~0.2s). An all-in pays equity; a call that sees a flop pays realized equity: a base by position and pot type (in position 1.0, out of position 0.8 in a 3bet pot and 0.9 in a 4bet pot), less for a hand behind (1.2 per unit of equity below 50% in a 3bet pot, 3.5 in a 4bet pot, where a dominated hand pays off or folds to a flop shove), a little more for one ahead in a 3bet pot (0.3), and a playability bonus for suited and connected hands; the two shares are normalised to add to 1, so the game is constant-sum. The constants are fitted to the charts' continue rates with `scripts/calibrate_pf_solve.py`: they landed within a few points of the old charts, but since the charts were matched to published solves (openers fold far more against a later seat's 3bet) they don't: openers call 37-55% of their range against the charts' 17-41% (calibration score 0.58, the sum of squared share differences over the six spots). The solve has 4bet bluffs (KJs, KQs, QJs) and jam bluffs (A4s, A3s, A5s), and QQ near indifference facing the jam. A strong hand facing a 4bet often calls rather than jams (so jam ranges are AA plus bluffs and KK sometimes folds to them), which comes from the realized-equity model of a call: it is the main limitation. `scripts/build_realization.py` measures it with the postflop solver instead, once, on the charts' ranges of six standard 3bet pots (openers out of position against a later seat's 3bet, in position against a blind's; 96 flops each, in the 3bet pot and the 4bet pot) into `gto/pf_realize.py`: per hand class against each of 9 opponent strength buckets (AA, KK, QQ-JJ, ...), how far its pot share after the call (EV over the pot, later bets included) runs above its all-in equity; the two sides' excesses combine so the shares add to 1, blended with the parametric model where measurements are thin. Per bucket matters (one number per class let KK's postflop edge count against AA, so KK folded to AA-heavy jams; now KK against AA is about -1 pot and AA against KK +1.8), and measuring on fixed ranges matters (re-measuring on the subgame's own ranges to a fixed point didn't settle). The flops are stratified by texture (suit pattern, pairing, high card, connectedness; 34 strata, at least two flops each) and weighted by the share of all flops each stands for: with random flops the two halves put the opener's calls against a 3bet up to 22 points apart, with weighted flops at most 10 (tests TestRealizationTable; the halves are in tests/data). The table is keyed by role as well as position (the caller, or the aggressor whose bet was called): position alone mixed an in-position 3bettor (realizing 1.15 of its equity in a 3bet pot) with an in-position caller (0.97), and with that mix the 3bettor trapped KK and the opener folded KK to the AA-heavy jams; by role KK calls them everywhere. `scripts/fit_realization.py` fits a smooth model to it (`gto/pf_realize_model.py`): per pot type, role and position, eight coefficients on the hand's equity against the bucket (x = e - 0.5: x, x·abs(x)) and its shape (pair, suited, connected, pair*x, suited*x). It stays off (pf_solve.MODE = "param"; "table" and "model" switch to the measurements; tests TestRealizationModel) because the measured caller-aggressor gap looks too large: callers realize 0.79 (out of position) and 0.97 (in position) in 3bet pots against the aggressor's 1.02 and 1.15, and about 0.6 in 4bet pots against 1.25, so in-position openers called 3bets less than the charts (BTN vs BB 28% against 34%, CO vs BB 25% against 41%) and out-of-position ones more (CO vs BTN 47% against 18%). Neither the postflop solver's depth nor its bet sizes explain the gap: playing the river instead of scoring it by equity moves these about a point, and four bet sizes a street (25-125% on the flop, 33-150% later) instead of two move them under one (40 stratified flops, BTN vs BB and CO vs BTN 3bet pots and the CO vs BTN 4bet pot). Nor does the 3bettor's range: widening BB's 3bet against a BTN open from the chart's 8.5% of hands to the published 11.5% leaves the BTN caller at 0.99 (40 flops). In aggregate an in-position caller realizes about its equity; the call rate falls short of published solves because marginal hands (KQo, A5s) measure a few points below their equity while strong hands run above, and checking that needs per-hand EVs from a commercial solver. So either the gap is real for the charts' ranges (a capped calling range against a strong, uncapped one: in the CO vs BTN 4bet pot the caller holds JJ, TT, AQs against KK+, AK) and the charts call too much, or the subgame's calls lack something the charts encode; settling it needs outside solver data. Public equity-realization figures cover only some of the roles. The GTO Wizard aggregate reports quoted by Poker Academy give 1.14-1.17 for the in-position raiser of a single-raised pot and 0.78-0.87 for the out-of-position caller (BTN vs BB, LJ vs BB, limped pots, 25-184 flops). A GTO Wizard blog example (BTN 3bets CO, 552r, 100bb) splits equity about evenly with EVs of 10.78bb against 7.72bb, which is 1.17 against 0.83. The measured aggressor in position (1.15) and caller out of position (0.80) match these. Nothing published covers an in-position caller of a 3bet. Rules of thumb (deepfold: 1.08-1.15 for late-position flats; Tournament Poker Edge: 1.25-1.5 for an in-position flat of a 3bet) put it above the measured 0.97, and per-class figures put offsuit broadways and offsuit aces below their equity and pairs and suited connectors above it, as measured (in-position caller: Axo 0.80, offsuit broadways 0.86, suited connectors 1.07, 22-66 1.11). The model mode now adds 0.18 to the in-position caller's realization in 3bet pots (pf_solve.MODEL_SHIFT, 0.97 to 1.15), which brings the in-position openers' calls to the charts (BTN vs BB 41% against 34%, CO vs BB 31% against 41%, BTN vs SB 38% against 36%; calibration score 0.30 against the parametric model's 0.58 and the unshifted model's 0.32). The out-of-position openers still call 2-2.5 times as often as published solves (CO vs BTN 47% against 18%, UTG vs CO 39% against 18%), even though the measured 0.80 matches public figures for that role. Fitting the charts would take it to about 0.64, against the 0.78-0.87 published for out-of-position callers, so the shift isn't applied there. The 3bettor also jams too often over a 4bet (28-39% against the charts' 15-39%). So the default stays "param", and the solver stays blocked: what's missing is per-hand EVs for an out-of-position caller of a 3bet. |
| Exploit mode | `gto/exploit.py` | The pool is the equilibrium plus common leaks, frozen (it doesn't adapt to you): facing a first bet it folds too often at every size on every street, most of all to small bets: on the river +20% of the range at a third pot, +13% at three quarters, +7.5% at an overbet, from mass data analysis of a real pool (GTOkiller, GG Poker NL100-200, BTN vs BB third barrels: 54.7% folds to half pot, 57.8% to pot, 70.6% to 175%, against ~33/50/64% at equilibrium); on the flop +8.5% at a third pot, tapering to +5% at an overbet, from the same source's c-bet analysis (47% folds on K74r against ~40% at equilibrium, ~45% on 652ss against ~35%, barely changing between a third and half pot; bigger flop bets only corroborated: whole-sample aggregates of CoinPoker NL10-NL100 and Ignition NL10, 2025, every pot type pooled, have folds rising from 39% against bets up to half pot to 54% against half to pot and 64% against overbets, about as fast as equilibrium's folds rise with size, so the taper stays), where the old guess had the pool calling small flop bets too much; on the turn +10.5% at a third pot, +7% at three quarters, +6.5% at an overbet, from Upswing's LeakBuster data of 500nl+ regulars' 2025 games (over-folding a delayed turn c-bet by 11/6/6 points at 30%/66-75%/120-150% pot and a turn probe by 10/8/7 at block/medium/overbet sizes; it replaces +14/+10/+6 interpolated between flop and river; one GG spot on Maxploit, a 150% turn overbet folded 72% against 58%, suggests lower stakes over-fold turn overbets more; the same CoinPoker and Ignition aggregates give the same shape at NL10-NL100: 41/52/63% folds by size band); the extra folds are its weakest calls, at most half of its equilibrium calls and never its top 10% of hands on the board (as a flat share of the range, a turn jam that equilibrium calls with a few percent lost every call, queens full included), the extra calls its strongest folds; it raises less (weak hands keep 30% of their raises, the rest call; a judgment call that one measured spot since bears out: GTOkiller has GG NL10-50 regulars raising a third-pot SB c-bet on K84r 9% of the time against GTO's 18%, and here the pool keeps 0.46-0.48 of equilibrium's raises in that spot; other figures since agree, none precisely: the same regulars re-raise a raise 16% against 25%, a GG river check-raise sits 10 points under GTO and is 95% two pair or better), and it bluffs less (bets from its weakest half are cut to 80/70/57% on flop/turn/river; the river from Upswing's same data: 500nl+ regulars' heads-up river bets in small pots are 12-25% low pairs or air by size against GTO's 18-37%, the bluffs kept at 0.55-0.62 of equilibrium's, 0.57 on average, where the old guess was 0.4; river showdowns are a fair sample, since a river call doesn't depend on the bettor's hand; the flop and turn are judgment calls, the turn between flop and river, since the only turn figure found comes from shown hands, which under-show turn bluffs). Strength is absolute: the hand's percentile on the board, averaged over runouts before the river. Preflop it is fitted to population HUD stats (Hand2Note mass data of GGPoker 6-max, all players, NL50 to NL500, published by Bluffaces: VPIP 28.75, PFR 18.7, 3bet 8.9%, fold to 3bet 48.25%, 4bet 12.75%, against the charts' 24.0, 17.7, 7.8%, 59.7% and 14.1% over every line weighted by how often it happens; scripts/calibrate_pool.py): facing an open it 3bets 1.23x the chart's share (the extra from its strongest calls); its extra hands are spread evenly over every cheap way in, since no data found splits this pool's VPIP by position or line: first in it enters, and facing one raise it flats, with 10.8% of what the chart folds (strongest first; fitted to VPIP; first in, the strongest 65% of those hands open-raise and the rest open-limp, fitted to PFR); first in from the small blind it also limps 7.6% of its range out of the chart's weakest opens, for 9.7% limps and 41.0% opens (chart 44.7%), since whole-sample aggregates of two other pools by seat (CoinPoker NL10-NL100 2025, 3.18M hands: small blind 7.1-9.7% limps, other seats 1.5-2.8%; Ignition NL10 2025: 15.0% against 8.6-9.9%) have the small blind limping 3-6x as often as any other seat and opening 7-12 points under the chart (their early seats open 1-3 points wider than the charts and their buttons 1-8 narrower, where this pool opens every other seat wider; left, since its PFR is above both of theirs), the big blind excepted (the chart's unraked defence is already wider than raked solves, and the one, unsourced, population figure has big blinds folding more to a button open than the chart, not less); before, all of it went to cold flats (non-BB calls facing one raise: 20.3% of those hands, now 13.8% with limpers calling an iso-raise included, against the chart's 5.5%); as the opener facing a 3bet it calls with 17.5% more of its range (its strongest folds; its wider opens are weak and fold to 3bets); its 4bets are the chart's. The pool's tree then gives VPIP 28.8, PFR 18.8, 3bet 8.9%, fold to 3bet 48.0%, 4bet 12.2%, open-limping 3.2% of its chances, between the other pools' 2.1-3.3% and 10.1% (with every extra first-in hand limped and the old, tight iso charts it gave PFR 16.4 and 4bet 14.8%). The old guess (3bets cut to 35%, 4bets only QQ+/AK) had the pool 3betting 4.1% and folding 57.5% to 3bets. Each street is still solved to equilibrium; the pool's actions narrow its range through the leaky strategy, and hero's EVs come from a best response to it over the rest of the tree. Actions whose best-response EVs are within the solve's noise of the best (its distance from equilibrium as a share of the pot, at least 0.5%) count as tied, and a tied hand plays the equilibrium mix among them: a short browser solve otherwise picked one at 100% on noise (AQo checking at 6.45 over bets at 6.40 in a solve 3.1% of the pot from settled). Preflop, hero facing a raise goes through the off-chart re-pricing (see Preflop) with the pool's ranges and responses: calls by equity against what the pool actually raises with, value raises by equity against what the pool continues with, bluff raises scaled (up to 1.5x) by how often the pool folds compared with the chart. Facing a jam over a raise, the jam range is the pool's. Opens follow the chart (the pool 3bets more, flats more and folds less to 3bets); a pool limper's range is its limps, hero's replayed range stays the chart's, and multiway postflop spots keep the betting rules. |
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
* **Flop**: flop and turn decisions are full; the river is beyond `depth`. It used to be an
  equity leaf both hands checked down, on 6 random turns, and the flop c-bet too little: BTN vs BB
  (100bb, a third and three quarters pot), QQ6 56% and K72 59% where published solves bet nearly
  their whole range. Two causes, measured against a solve with every turn and river betting:
  * A checked-down river gives position and betting no value there. The river now keeps its
    betting (`river_play`), played on the turn's boards without seeing the card (its showdowns are
    the turn's equity matrix): at 3 turns QQ6's c-bet went 50% (checked down) -> 75% (river played
    blind) -> 85% (every river card in the tree, ~60x the cost). It costs ~3x a checked-down leaf.
  * A few fixed turns let the flop's ranges fit the cards that come (BB's calls favour the hands
    that pair them), which with river betting cost c-bets: K72 c-bet 68-94% by seed with 8 random
    turns, 92% with 16, 97% with all 49. The turns are now one card per rank (`turn_classes`; a suit
    with two flop cards gets its own card per rank), each weighted by the turns it stands for, its
    pairs' weights symmetric so the game stays zero-sum: 13 turns on a rainbow flop, 23-24 on a
    two-tone or monotone one.
  Against every turn played (300 iterations, 12 flops: 9 BTN vs BB, 3 CO vs BTN 3bet pots) the
  default is within 1.4 points of c-bet frequency on average (worst 3.8), 3.0 on the split between
  sizes, 0.9 on BB's check and 2.0 on its fold/call/raise against the small bet; 6 random turns
  with the river checked down were 12.4, 27, 5.3 and 5.8 off. Inside a flop solve the turn and
  river have one bet and raise size (`later_sizes`): with two, K72 c-bets 78.8% for 77.8. Against
  published solves (a 25%/80% flop tree like ThinkGTO's): QQ6 96% (published 98.8%, was 64%), K72
  97% (BeyondGTO 98%, was 64%), A74 74% (67%, was 70%), 883 65% (68%, was 52%); BB vs a 25% bet on
  A74 folds 36 / calls 45 / raises 18 (34.6 / 50.0 / 15.4). An iteration costs ~7x the old tree's
  natively (~15 it/s on a rainbow flop, half that on a two-tone or monotone one) and ~15x in
  Pyodide, and a solve needs ~100: 30 iterations were 12 points of c-bet and 15 of BB's check from
  the reference, 60 were 8 and 8, 100 were 5 and 4. So the server's Accurate settles, and the
  browser's Accurate gets 60-75 iterations in 60 s, about 8 points of c-bet from the reference (12
  before) with BB's check a little noisier (7.5 points, 6 before).
  * Fast (2.5 s natively, 8 s in the browser) plays the same flop model with the turn and river by
    hand buckets (`river_buckets=32, turn_buckets=True`): each player's range on each turn card is split
    into 32 equal shares ordered by equity against the other's flop range, and the turn and river below
    a flop solve are played per bucket against a 32 x 32 bucket-pair equity matrix (the turn's every-river
    equity averaged over the pairs with the flop ranges' weights, card removal as each pair's compatible
    share; `bucket_river_step=3` averages every third river, no worse). The flop stays per combo. The two
    players' tables are each other's negated transpose, so the game stays zero-sum. Per combo the river
    beyond depth was ~95% of the arithmetic (N x N leaf products); by buckets an iteration costs what the
    old Fast tree's did natively (12-18 ms on a single-raised pot) and 2.4x less in Pyodide (30-42 ms
    against 74), so Fast gets 140-210 iterations natively and 190-260 in the browser, where it had ~200 and
    ~100 (setup, before the budget: 0.6-1.0 s natively, 1.1-1.6 s in Pyodide, against 0.6 and 0.7; the turn
    leaves' showdown sums now take one comparison and a byte add per board, and every board's hand
    strengths one batched evaluator call). Against every turn played (12 flops above), at the iterations each gets: c-bet 3.6 points off
    on average (native Fast was 12.2, the browser's 14.5), size split 7.6 (32.6, 35.5), BB's check 2.1
    (6.1, 13.0), its response to the small bet 1.6 (5.0, 6.2). Settled (400 iterations), 32 buckets are 3.0
    off in c-bet where per combo is ~1.4; QQ6 stays near 80% (88%), A74 near 61% (51%). Exploitability in
    the bucket game is 0.2-0.4% of the pot at 200 iterations (the old trees, not zero-sum with their
    sampled turns, read -0.35 to 1.15%). Exploit mode's pool lock reads a bucket's percentile as its place in
    the range.
  Tests TestFlopStrategy hold the published table, for both models.
  A flop all-in pays its equity over every turn card with 4 evenly spread rivers each
  (within 4% of exact; 4 sampled turns had flush draws up to 24% off).
* `Solution.exploitability()` reports how far a solve is from equilibrium in
  % of the pot (it plays a best response for each side against the other).
* **Solver speed** (`scripts/bench_solver.py --curve`, BTN vs BB single-raised pots, one thread):
  in Pyodide a flop solve runs ~1.5x the iterations per second it did (Fast 15 -> 23, Accurate
  11.5 -> 17.5) and needs ~30% fewer of them, so after 4 s it is 0.3-1.1% of the pot from
  equilibrium where it was 1.2-2.7%; the Accurate turn
  (Ks7d2cTh, every river) runs 6.3 -> 11.6 it/s, 1.76% -> 0.31% of the pot after 8 s.
  Natively Fast and Accurate flops run 150 -> 210 and 105 -> 170 it/s, and the server's
  default config 8 -> 35 it/s (Qs8d3c after 4 s: 8% -> 0.6%).
* **Native trees** (`gto/native.py`, server and CLI): `Config()`'s full menu (two bet and two raise
  sizes a street) ran 10 turn iterations in the server's 2.5 s, 60-70% of the pot from equilibrium,
  and narrowing on it left BB 7 combos checking a K72T5 river where a long solve keeps 81 (69.5 since
  the BB's flats follow published raked solves). Fast now
  plays the browser's sizes with every river: the turn is 1.5-2.5% of the pot from equilibrium after
  2.5 s (40-45 iterations unloaded) and that range kept 79 (fixed 8-river sets settle sooner but narrow to 63 even converged).
  Accurate adds a second turn and river bet size: 0.2-0.7% after 30 s, 1.7-2% after an earlier
  street's 12 s, 89 combos. BLAS is held to one thread there.
* **Earlier streets** used for range narrowing get their own budget each, and a solve that is
  reused later in the same hand keeps iterating from its stored regrets instead of
  restarting. When a real bet or raise size differs from the abstraction by more
  than 15%, that street's tree uses the real size so narrowing is exact. Narrowing follows an
  earlier street's solve by how settled it is (`advisor.PRIOR_TRUST`: fully at 0.5% of the pot
  from equilibrium on the flop and 2% on the turn, not at all at 5% and 10%, linearly between),
  the rest of each action's weights coming from the multiway betting rules' range for it (a check
  keeps the rest of its range whole: the big blind checks nearly every flop to the raiser). A loaded
  machine's (or the browser's) short solves otherwise collapsed ranges: K72 c-bet called, T turn bet
  called, BB's river check range is 69.5 combos after long solves, and a 20-iteration flop (9% of
  the pot; BB checking 61% and calling the c-bet 21% where it settles at 95% and 47%) with a
  26-iteration turn (6%) left 18-24 (with checks narrowed by the rules too, 89 there, 55-65 at
  40-iteration flops, were 41-43). With Fast's bucketed flop (long solves keep 87.7;
  20 iterations are 27% of the pot from equilibrium, 40 7.7%, 80 1.8%, 150 0.5%): 29 -> 112 at a
  20-iteration flop and 26-iteration turn, 50 -> 112 at 40/26, 52 -> 87 at 40/45, 70-82 -> 78-81 at
  80-150/45 (tests.test_native judges by iterations, not time).
* Multiway pots (3+ players live at your decision) are not solved. Instead a Monte
  Carlo heuristic picks the highest-EV action among fold / call / raise to 75% pot
  (the page's default raise size), or check / bet 33% / bet 66% pot, with opponents
  following simple range rules: facing a bet they fold together a quarter less often
  than a balanced defence (people call too much), with everyone keeping the hands above
  one strength bar, so a weak range (a lead after checking twice) folds more of itself
  than a strong one (a call of that lead), and the strongest part of what continues
  raises. Raises inside the trials go to 3x the bet plus the calls in front of them
  (a standard raise: 22 bb over a 5.5 bet and one call). When they bet it's mostly a
  third of the pot (SMALL_BET) with a merged range: value from the top 70% of what one
  player would call it with, compounding with more players (1 - 0.3^n of what the field
  calls), plus balanced bluffs; 66% takes the top of that value (what beats most of
  what would call it) in proportion to the bettor's nut advantage, its share of the
  top 10% of everyone's ranges pooled: from 40% of the nuts it starts betting big, from
  60% everything that would. Fitted to GTO Wizard's 3-way flop study (100bb, LJ opens
  and both blinds call: the opener checks 34%, bets a third of the pot 56%, two thirds
  8% and pot 1%, against 23/38/21/18 heads-up with the big blind; bigger multiway bets
  come where the bettor holds the nut advantage): over 200 random flops the 6-max UTG
  opener into both blinds now checks 43%, bets 33% 51% and 66% 6% (heads-up with the
  big blind 24/36/40), where the old rules (66% the top of the range, 33% the thin
  layer below) checked 67% and bet 66% 26%, 33% 7%. Hero's size is chosen by EV. The
  value is set against what the players it's bet into would call with, from their
  actual ranges (an opener betting into two flatters values thinner than a flatter
  betting into the opener), and the bluffs are the hands without showdown value that
  most often end up strong (draws and overcards first; on the river the weakest hands).
  Bluffs from the bottom of the range and value as if everyone held the bettor's range
  made a 66% bet nearly all sets and overpairs plus air, which a draw out-continued top
  pair against. A check rules out the big bet, the small bet's bluffs and its strong
  value; the merged middle (thin value, protection) checks as well, as in the solves,
  whose checks keep medium hands and traps (ruling out every hand that would bet small
  left the checkers so weak that top pair raised a river lead for +47 bb). Narrowing a
  small bet keeps the small and big ranges both (players mix). Players still to act
  behind you can call or raise after you call; they faced the bet with you, so each folds
  as often as when everyone it was made into faces it together (39% of the range continues
  against a pot-sized bet with two of you facing it; pricing it as a fresh bet into
  the pot your call had grown kept 81%, junk included, and that dead money paid for
  your calls: K5s on K72 called a pot bet at +19 bb, about +16 now, the checking big blind behind being weaker). Each bet size gets
  its own continuing ranges, so a third of the pot keeps more of the field in (3-way,
  each keeps about 57% of their range against 33% and 45% against 66%) and gets
  raised more often. When you check, the first player behind you whose betting range
  at either size holds their hand bets it (a player behind you who checked first holds
  the check range: with the full range, its folds to your check-raise came out a third of
  the time where its actual checking hands folded twice as often), and the others face that bet the same way as
  players behind a call (call, raise with the strongest part, fold the rest): those
  between the bettor and you act first, and you fold, call or check-raise knowing what
  they did (per bettor, size and what the players between did, whichever pays over the
  trials; a call your equity against the bettor and the callers already pays for is made
  even where those noisy splits came out slightly negative; a check-raise only when it
  beats calling or folding by two standard errors of the paired difference over at least
  30 trials of that situation: on the bare means middle pair 7s6s on K72 check-raised
  about 37% of the bets and the gutshot T9s on 863 15-64% depending on when the trials
  stopped, now about 0% each, while a set of sevens check-raises about 93% and two pair
  K7 35-87%), those after you act once
  you've called. A check-raise is faced by everyone still in (the bettor, the callers
  between, the players behind) by the same joint fold rule, the strongest part of what
  continues re-raises 3x and you call that by equity. The notes and `check_line` say how
  often someone bets after your check and how often you'd call, check-raise or fold. On
  8d6h3c checked to the big blind 3-way (CO opens, BTN and BB call), top pair Ad8s used
  to continue against 46% of those bets while the open-enders 7c5c and 5c4c continued
  against all of them and the gutshot T9s 44%; now everyone continues against the small
  bets (top pair check-raises 43% of all bets) and against the big, nut-advantage bet
  top pair folds where the draws' price keeps them in (top pair 78% of unraised bets,
  the draws 100%); a set of sevens on K72 check-raises most
  bets and air never does. A raise can be re-raised, and every line plays to the river
  by the same rules: on each later street players act in seat order, an opponent bets
  66% or 33% by the betting ranges above, you lead 66% when you're ahead of every range
  that would call it, else 33% when ahead of what calls that, and everyone still in faces
  a bet (opponents call, raise or fold by the joint rule; against your bet they call or
  fold: a range capped by checks raising its best weak hands stacked off 77 against top
  pair), you raise when ahead of each range that would continue, otherwise call or fold
  by your equity against the bettor and the callers (on the turn, also counting the river
  bet that may follow). One raise per later street, which everyone still in then calls or
  folds. With rake, every pot you win is paid less its rake, and your in-trial calls,
  leads and raises count it too: a call pays when your equity times the raked pot beats
  its price, and the lead and raise equity bars rise by what the rake takes of the chips
  they add (zero rake leaves every decision exactly as before). Trials continue until the
  gap between the two best actions is clear of the sampling noise or under 2% of the pot
  (or a time cap), and the notes give that gap with its error. The trials run about 1.4x
  as many per CPU-second as before, with the busier betting above (1.75x with the old
  betting rules): the hand evaluator works per hole-rank pair instead of per combo (9x
  faster, identical ranks, which also speeds the heads-up solves), equities are matrix
  products over the runouts, and later-street decisions are cached per runout. A warm
  3-way flop spot (a new hand on a board already seen) took about 2 s and hit the time
  cap; now 0.4-1.3 s, mostly stopping on the noise rule.
* Streets that started multiway narrow every player's range by the same rules
  (a check drops most hands that would have bet, a bet keeps its value + bluff
  range, a call drops what would have folded or raised, with a first call and an
  overcall of the same bet keeping the same share); hands a rule rules out
  keep 30% of their weight. That covers multiway spots and heads-up pots that
  were multiway earlier, including a pot that became heads-up on this street.
* In heads-up spots the villain's actions this street narrow their range exactly,
  unless the solve takes that line with under 10% of their range; then it is
  smoothed like earlier streets and the advice says so (EVs are then against the
  wider range, so actions the solver mixes can show different EVs).
* The rule constants (value share of a bet, share of continuing hands that raise,
  lead threshold, narrowing softness) are fitted to the heads-up solver with
  `scripts/calibrate_multiway.py`: on ~240 heads-up flop spots the heuristic's pick
  gives up 0.14 bb on average against the solver's best action, and its EVs are
  off by 1.38 bb on average. (Fitted at 0.10 bb and 1.0 bb; the later rules, played
  to the river with both sizes, check-raises and the old betting ranges, had drifted
  to 0.21 bb and 1.40 bb. Refit after the betting ranges above: OFF_RULE_MAX 0.25 to
  0.4, the rest unchanged; score 0.488, then 0.437 with the new betting and check-raise
  rules, 0.415 refit. The script now plays a fixed 1200 trials per spot, so its score
  doesn't depend on machine load.) The betting mix (SMALL_SHARE, NUT_TOP, ADV0,
  ADV_SPAN) is fitted to GTO Wizard's frequencies instead. The "players call too much"
  shade is deliberate and not fitted. The multiway interactions themselves, which
  the solver can't play, were sanity-checked on ~190 3-way spots (single-raised and
  squeezed pots, flop first to act, facing a bet, facing a bet and a call, turn and
  river after checks, hands from sets to air): no errors, no strong hand folding
  where a weaker one continues, its equity against the field within MC noise of an
  exact count, and against two players it bluffs far less than the heads-up solve
  (air checks where heads-up bets 80-100%). The small bet and the players behind a
  bettor were checked on ~570 3-way spots (open, cold call and big blind; open and two
  calls; a squeeze called twice; flop, turn and river; sets to air): a third of the
  pot is picked in 15% of them, mostly with hands between a fifth and four fifths
  of the field's equity (top pair weak kicker, second pair, draws: 79 of 296), never
  with air (0 of 141) and rarely with strong hands (9 of 130, sets among them,
  in squeezed pots, where the stacks are short of the pot and top pair also bets
  small: J8s on J94 bets 33% for 30.3 bb against 24.6 at 66%). Checking is worth
  about the same with weak hands, less with middling ones in big pots (players
  between call or raise the bet you'd call: QJs on 863QK in a squeezed pot 26.6 to
  17.6 bb) and more with strong ones (their calls add money: a set of nines on J94,
  76.7 to 83.0). Strong hands still call everything except a raise from a player
  between when the raiser's range is mostly better (a set of nines against a flatter
  whose top is jacks); the nuts call every time. On the heads-up calibration spots
  nothing changes with one size (score 0.351 before, 0.348 after: MC noise); with the
  solver also betting 33%, the pick loss stays 0.18 bb and the EV error is 1.12 bb
  against 1.08 (the solver is nearly indifferent between its sizes there, so picking
  among more actions barely moves the loss). The trials take about 1.3x as long
  (more actions to separate; the time caps are unchanged).
* If your hand isn't in your range for the line you took (a preflop call the charts
  never make, say), it is added to the solve as a near-zero-weight combo so it
  still gets a strategy and EVs without changing anyone else's play.
* If a preflop line leaves a player with fewer than 8 combos (a flat the charts never
  4bet from, then a 4bet; a cold call of a 3bet, then a call of the 4bet), the hands
  that take the last such action from scratch are mixed in, weighing 1 - combos / 8
  (all of them for an empty line), keeping the hands that got there (a flat then a 4bet
  is QQ+/AK with a few A5s/A4s over a late open, KK+/AKs over an early one), and a jam
  from such a line is faced with that range. Otherwise the solve plays against a face-up hand: BB would fold
  everything to a jam with KK and pay off nothing smaller. Where the action's own chart is too narrow
  to reach 8 combos even from scratch (an opener calling a cold 4bet calls 6.8), the strongest hands
  of the range before the action top it up to 8, by equity against a random hand.

### Deploy to the website

```bash
scripts/deploy.sh
```

copies the page, the worker, `practice.js` and the `gto` sources into the `gto-solver/` folder
of the michaellhan.github.io repo and pushes. First it runs the checks and aborts
the deploy if either fails: `python -m unittest`, then the quick sanity sweep
(`scripts/sanity_sweep.py --quick`, about 3 minutes on 10 cores: a fixed, seeded set
of ~80 preflop nodes and ~110 postflop spots covering every setup, every heads-up
spot kind on flop, turn and river, 3-way pots and exploit mode, plus ~30 hand-picked
rare lines over the setups that the quick preflop trees leave out (`RARE_LINES`:
3bet and 4bet pots, cold 4bets, flat then 4bet, a squeeze called twice or 4bet, 4bet
and 5bet jams faced, limp-reraises, straddled 3bet pots), with their decision nodes,
the ranges they reach the flop with and flop and turn spots in their pots; postflop
flags are re-solved at a longer budget and it exits non-zero when any flag but the
info ones, `filled_range` and `unsettled`, reproduces). The full sweep (~40 min) is
not part of the deploy; `scripts/sanity_sweep.py --full-report` runs it with the rare
lines and writes `reports/sanity_YYYY-MM-DD.md` (nothing schedules it). It uses `.venv/bin/python` if present,
else `python3` (`PYTHON=...` overrides). `DRY_RUN=1 scripts/deploy.sh` runs the
checks and stops before cloning or pushing; `SKIP_CHECKS=1 scripts/deploy.sh`
skips them, for emergencies only.

Tune with `Config` in `gto/solver.py` (`depth`, sizes, sample counts) or the
`--budget` / `--depth` flags. `depth=3` from the flop is the full tree; it is
much slower in pure Python. The real fix for a production version is moving
the tree traversal to C or Rust or a GPU, which makes full flop trees feasible.

### Known limits

* Preflop charts are written by hand, not solver output, and the off-chart adjustments are rules on
  top of them. They are matched to published solves where any exist: `scripts/preflop_reference.py`
  measures 22 spots (opens, BB and SB defence, openers facing a 3bet; the raked ones played at that
  rake) and 14 hand-level facts against them (tests/preflop_refs.py, average error 1.0 points, was
  2.9 with 10 spots and 8 facts out). Spots with no published solve (4bets onward, squeezes, cold
  calls, heads-up, limped pots) are judgment. `scripts/preflop_sweep.py` runs
  every standard line for 2-9 players and flags broken frequencies and any hand continuing less often than one it dominates (pairs, kickers, suited over offsuit; the strategy is also kept that way at run time). Replace them with real chart
  exports if you have them; the format is a range string.
* Limps, antes, straddles and stack depth are handled by adjusting the 100bb
  no-ante charts (see Preflop above), not by charts built for those games.
* Rake (optional) re-prices preflop calls (and push/fold) by one expected-rake rule (2.5x the flop
  pot, fit to one published raked solve), the charts fit to raked solves only past its 5%/4bb; 3bet
  and 4bet ranges aren't re-priced; equity leaves where the tree stops early are raked on the pot as it stands, and the multiway
  heuristic rakes its payouts but not the betting rules inside its trials (see Rake above).
  The re-pricing keeps the big blind's calls by all-in equity, so its totals match raked solves but
  not their shape: against a button open at 5% capped at 0.6bb it folds 76s, 54s, T8o and 98o and
  keeps every offsuit king K7o+ (raked solves keep the suited connectors), and BTN vs BB on K72
  then c-bets 47% (QQ6 63%, the BB checking 83%) against 69% unraked and published 90%+.
* Exploit mode's pool is fitted to public population data where any was found (flop and river
  folds, turn folds and river bluffs from 500nl+ regulars, preflop frequencies, one flop spot's raise
  frequency, and the small blind's open-limps); folds to big flop bets and the turn's shape at small
  stakes are only corroborated by whole-sample aggregates that pool every pot type, and how much less the
  pool bluffs on the flop and turn, and how its extra preflop hands split between opens, open-limps, flats
  and the big blind outside the small blind, are still judgment calls. On a rough (Fast, flop or turn)
  solve the best response also picks up the solve's own noise.
* Multiway with an all-in player for less, the heuristic pays each side pot to the best hand among the
  players in it, but the betting rules inside its trials don't play for the side pot (an opponent bets
  and calls as if the whole pot were at stake).
* Practice opponents heads-up after the flop read their move off your own solve, whose ranges leave out
  your cards (as your advice does): an edge the size of a blocker. A line of yours the solve rarely
  takes, or a pot that became heads-up mid-street, gets a solve from their seat instead (a few seconds,
  more in the browser).
* The bet-size abstraction is coarse; real actions are mapped to the nearest size.
* Flop solves play the river without seeing the river card (a draw is a middling hand there, so
  river betting is worth a little less than with every river: QQ6's c-bet ~10 points under at 3
  turns) and one turn card per rank class; the fast configs (server Fast, browser Fast, and the
  browser Accurate's earlier streets for narrowing) still check the river down on a few random
  turns, so their flop c-bets run low (see Accuracy vs speed). 3bet pots match every-turn solves
  less well than single-raised ones (narrow ranges hold the turn cards played more often).
