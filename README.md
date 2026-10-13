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
blinds, antes, straddle, pot and bet sizes, validation, undo and rewind), share links (`#h=`) and the
copied hand text, and the state sent to the solver (its keys are checked against what
`Advisor._advise` reads). They also write `tests/data/page_states.json`, random hands played through
the page; `tests/test_page_states.py` replays each spot with the advisor's own `replay_hand` (same
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
every river. The local server keeps the full size menu and gives better postflop accuracy.) Pick the number of players (2 to 9), your seat
and stack, click cards into your
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

Whenever it becomes your turn the advice panel solves the spot. The **Fast /
Accurate** toggle in the header (remembered between visits) picks the solve
time: Fast is 8 s in the browser and 2.5 s against the local server; Accurate
is about 60 s in the browser and 30 s locally, with more time for the earlier
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
per player, `--straddle` has the first seat straddle 2bb, `--rake 0.05 --rake-cap 3` sets a 5% rake capped at 3bb (see Rake).

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
first seat posts 2bb and acts last preflop). Optional `"rake"` (a share of the pot, `0.05` for 5%)
and `"rake_cap"` (the most rake per pot, same units as the amounts; leave it out for no cap): off
unless given (see Rake below). Positions are the 6-max names `UTG HJ CO BTN SB BB`.

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

## How it works

| Layer | File | Method |
|---|---|---|
| Cards, ranges | `gto/cards.py`, `gto/ranges.py` | 1326-combo weight vectors, PIO-style range syntax |
| Hand strength | `gto/evaluator.py` | numpy-vectorised 7-card evaluator built on treys' lookup tables: all 21 five-card subsets of every combo on a board in one pass, about 1 ms per board |
| Preflop | `gto/preflop.py` | Hand-written 100bb charts: RFI, vs RFI, vs 3-bet, vs 4-bet, cold vs 3-bet, mixed frequencies. Seats for 2 to 9 players map onto the charts by how many players are left to act (9-max LJ plays the 6-max UTG chart, 7+ handed UTG plays a tighter EP chart, heads-up has its own button and big-blind charts). Replaying the preflop action through the charts gives every player's range going to the flop. The charts assume a standard-sized raise and nobody else in the pot; when the price or the players in differ (a min-3bet, callers ahead), calls are re-drawn by preflop equity against everyone in versus the pot odds, held to the equity bar the chart's own continue range sets in the standard line. With callers in, raises are re-drawn too: a value raise stays while its equity against what continues (each player's chart facing the raise) clears break-even by as much as the chart's weakest value raise heads-up does (callers' dead money makes thinner value raises pay), bluffs shrink by how much less often everyone folds against how often a bluff of that size needs them to (the callers' dead money lowers that bar), and the default raise size adds the callers' chips (a squeeze over two callers of a 2.5bb open is 15bb from the BB). Off-chart games and depths: limpers get iso-raise / over-limp / big-blind-raise charts (and an open-limp range); the iso-raises over one limper are a little tighter than the seat's open, at published widths (HJ 18%, CO 24%, BTN 36%, SB 19% of hands, the big blind 15%, against Preflop Wizard's and Grindlab's ~15-22%, 22-35%, 30-45%, 18-25% and ~15%; the old charts raised 10/13/16/10/8%; no solver frequencies for cash iso-raises were found: GTO Wizard's multiway preflop solver offers only iso-raise or fold facing a limp outside the blinds, and solved 6-max packs are mostly no-limp trees), about 0.6x that over two or more limpers (value-heavy, the dropped pairs and suited hands over-limping); antes play the open chart of a later seat and re-price calls; a straddle makes the straddler the big blind and the rest play by seats behind; stacks other than 100bb re-price calls with implied odds for small pairs and suited hands; a raise that commits the stack (40% of it, or under 1.5 pots behind once called) goes all-in, with the push/fold equilibrium as the range when shallower than the charts (and every open at 12bb or less). Out of position facing a 3bet (a later seat's, or the big blind's against the small blind) the opener plays charts matched to published solver output: CO against a BTN 3bet to 7.5bb folds 63.7%, calls 17.2% and 4bets 19.1% of its opening range (Poker Academy, 100bb cash), openers respond in similar proportions from every seat (GTO Wizard), and the small blind against the big blind's 3bet folds about half and 4bets a third of its continues (888poker); the earlier charts called about twice that and 4bet half as often (CO 58/33/9). Also matched to published solver frequencies (100bb cash, 2.5bb opens): HJ and CO against a UTG open mostly 3bet or fold (90/2.5/7 fold/call/3bet of hands, against HJ 91.1/1.5/7.4 from Poker Academy and CO calling 1.4-3.2% and 3betting 8-9% from GTO Wizard; was 88/8/3), the button against a CO open (85/3/12, against 84.5/3.0/12.5 from Poker Academy; was 82/9/9), the big blind's 3bets against the button and the small blind (11.5% and 16.4% of hands, as Poker Academy and gtopreflop.com have them; were 8.5% and 11.7%, the new 3bets taken from calls: those solves are raked and defend less in all, but the big blind's flats are where rake bites hardest and the solver here has none, so the defence totals stay), and the button facing the big blind's 3bet (54/34/12 fold/call/4bet of its opening range, against 54.6/33.9/11.5 from 888poker; was 59/35/6, a 4bet range of nearly all value) and the small blind's (60/29/11.5: gtopreflop.com's raked solve has the button 4betting as often against either blind, 10.4% and 10.3%, and calling 6 points less against the SB, with QQ, AKo, A9s and KTs mostly 4betting and AQs/KQs calling; was 58/36/6). The big blind's 3bets against UTG, HJ and CO are 4.8%, 6.2% and 8.0% of hands (gtopreflop.com 4.5/5.9/7.7, Poker Academy 4.5% against the LJ, pokalab 5.1/6.7, BeyondGTO 6/8/10; were 3.6/3.6/5.6, the HJ spot reusing the UTG chart), taken from calls so the defence totals stay; its squeezes follow (over CO + BTN 7.3%, was 4.9%; over UTG + a caller 4.5%, was 3.4%), about 0.9x the heads-up 3bet where GTO Wizard's NL500 cash solve at 40bb has 0.95x. Cold players facing an open and a 3bet fold 96-97% (GTO Wizard: over 90%, up to 97%) and the big blind's squeeze over a button open and a small-blind call is 10% of hands (Poker Academy 12.7%): both left. No cash solves of the small blind's or button's squeezes were found (gtopreflop.com's database has no node facing two players in the pot; the tables that turn up are labelled approximations). The opener calls a blind's 3bet wider (in position), the heads-up button has its own vs-3bet chart, and a 5bet that leaves chips behind can be shoved over. Every player's replayed preflop range goes through the same adjustments as hero's advice (so a cold call of a cheap 3bet or a multiway big-blind defence is wide), a limper facing a raise, an iso-raiser facing a limp-reraise, the heads-up big blind facing a 4bet and every seat (blinds included) facing a 7+ handed early-position open have their own charts; facing a 3bet cold depends on how wide the 3bet is (over an early or a late open) and on position against the 3bettor (out of position, half the cold calls go: premiums 4bet, the rest fold), and facing an all-in that someone already called, equity counts the callers. A player who called a raise and faces the next one (a flat facing a squeeze, a cold call of a 3bet facing the 4bet, a call of a 4bet facing the jam) continues with at least 30% / 35% / 55% of the range they called with: the charts for those spots are written for someone who hasn't put chips in and continue mostly with QQ+/AK, which a flat almost never holds, so they left a flatter's range nearly empty (2% of BTN's flats called a BB squeeze). Those shares are floors, set against the tightest raises the charts make (the big blind's old 3.6% 3bet over UTG, UTG's 4bet, the 3bettor's jam); a wider, more bluff-heavy raise is defended more: each point of equity the caller's range gains against the actual raise over that reference adds 2.5 points (BTN's flat facing the big blind's published 8% squeeze over CO continues 44% and calls with 77; with a fixed 30% the wider squeeze had pushed 77 out). The extra calls are its strongest folds, ranked by how readily the button continues against a 3bet (TT-77 and suited broadways before 22-55), then equity against the raiser. Once another raise is in after the call (sandwiched) the charts' folding stands. The small blind flats an early open rarely (about 3% of hands) but not only middling hands: a share of JJ, TT, AQs, AJs and KQs flat as traps (published summaries have it sometimes calling JJ and TT there), where the old chart 3bet them all, so its flat-then-call of a big-blind squeeze was 13 combos of 77-99 and the squeezer's QQ won every A-K-x flop; now about 12 combos of 99-JJ, some 77-88 and those suited broadways. The 3bettor facing a 4bet uses a chart written against the opener's 4bet, which carries bluffs; against a cold 4bet or one from a player who flatted first (nearly all value) each hand is re-drawn by its equity against the actual 4bet range versus the opener's: value hands keep continuing while their equity clears the price of calling over the 80% of it a call realizes (one bar for every spot, so JJ calls 53% against a cold 4bet over UTG, where it has 33%, and always over the button, where it has 36%), and marginal and blocker hands shrink by how far they drop (AQs calls 5%). Bluff jams, and the value jams the chart mixes with calls, scale by how often that range folds to a jam compared with the opener's: against its value a jam and a call both get the stacks in, so a jam's edge is the equity it denies the bluffs, while a call can still fold bad flops. The 4bettor folds QQ and AKo to the charts' tight jam ranges whoever it is, so against a cold 4bet the jams hardly change (QQ 56%, AK 47%); against a flat-then-4bet over an early open, which folds far less, QQ jams 10% and AK 7%, calling the rest; KK+ and AKs keep jamming. A 4bettor facing a 5bet jam calls by equity against the jammer's replayed range versus the pot odds, like hero's advice (the fixed chart called QQ and AK 70% whoever jammed, where they have 36% against a standard jam and need 41%). |
| Postflop | `gto/solver.py` | Heads-up range-vs-range **discounted CFR** over a bet tree (2 bet sizes per street, raises to half pot and full pot after the call, all-in when the stack is short or a raise size would exceed it, max 2 raises). Vectors are compressed to combos in either range. A river showdown is O(N) per board: combos sorted by strength, cumulative reach sums, and a per-card correction for card removal. Equity leaves are one matvec with a precomputed antisymmetric matrix. Subtrees below the root street with the same shape (the turn trees after most flop lines differ only in pot size) are merged into one batch and traversed once, so a flop solve walks ~30 decision nodes a pass instead of ~90, with the same numbers (~1.6x natively). In the browser, whose numpy has no BLAS, the leaf matvecs are nearly all the time: they run through einsum there, on leaf matrices trimmed to each player's range (~2.6x in Pyodide; `scripts/bench_solver.py` benchmarks the native solver). |
| Hand replay | `gto/advisor.py` | The whole hand is replayed for any number of seats: blinds, folds, dead money from players who folded, all-ins, acting order (heads-up the button acts first preflop and last postflop). Whenever exactly two players remain the spot is solved heads-up: from the start of the street if it began heads-up, or from the current node if a third player just folded. With three or more live players it falls back to a Monte Carlo heuristic against every live range (see Limits), and says so. |
| Multi-street | `gto/advisor.py` | Earlier streets are solved with a small budget and each player's range is narrowed by the frequency of the action they actually took, mixed with a 15% uniform floor so an approximate solve never prunes a line to nothing. A range entering a later street with fewer than 8 live combos (preflop.MIN_LINE_COMBOS, counting the cards dealt since) is filled out with the rest of the range the player started the street with, weighted as preflop (1 - combos / 8): a 4bet pot's ~12-combo range checking back the flop left 2 for the turn, so the turn solve played a face-up hand. The current street is solved from the start of the street and the tree is walked to hero's node with exact frequencies. Solutions are cached per hand so the flop solve is reused on the turn. |
| Preflop subgame (experimental) | `gto/pf_solve.py` | Not used by the advice yet. Solves the 3bet pot heads-up as a game: the opener facing the 3bet folds, calls or 4bets; the 3bettor facing the 4bet folds, calls or jams; the opener facing the jam folds or calls. Hands are the 169 classes with class-level card removal; CFR+ with linear averaging converges to under 0.01% of the pot from equilibrium in 1000 iterations (~0.2s). An all-in pays equity; a call that sees a flop pays realized equity: a base by position and pot type (in position 1.0, out of position 0.8 in a 3bet pot and 0.9 in a 4bet pot), less for a hand behind (1.2 per unit of equity below 50% in a 3bet pot, 3.5 in a 4bet pot, where a dominated hand pays off or folds to a flop shove), a little more for one ahead in a 3bet pot (0.3), and a playability bonus for suited and connected hands; the two shares are normalised to add to 1, so the game is constant-sum. The constants are fitted to the charts' continue rates with `scripts/calibrate_pf_solve.py`: they landed within a few points of the old charts, but since the charts were matched to published solves (openers fold far more against a later seat's 3bet) they don't: openers call 37-55% of their range against the charts' 17-41% (calibration score 0.58, the sum of squared share differences over the six spots). The solve has 4bet bluffs (KJs, KQs, QJs) and jam bluffs (A4s, A3s, A5s), and QQ near indifference facing the jam. A strong hand facing a 4bet often calls rather than jams (so jam ranges are AA plus bluffs and KK sometimes folds to them), which comes from the realized-equity model of a call: it is the main limitation. `scripts/build_realization.py` measures it with the postflop solver instead, once, on the charts' ranges of six standard 3bet pots (openers out of position against a later seat's 3bet, in position against a blind's; 96 flops each, in the 3bet pot and the 4bet pot) into `gto/pf_realize.py`: per hand class against each of 9 opponent strength buckets (AA, KK, QQ-JJ, ...), how far its pot share after the call (EV over the pot, later bets included) runs above its all-in equity; the two sides' excesses combine so the shares add to 1, blended with the parametric model where measurements are thin. Per bucket matters (one number per class let KK's postflop edge count against AA, so KK folded to AA-heavy jams; now KK against AA is about -1 pot and AA against KK +1.8), and measuring on fixed ranges matters (re-measuring on the subgame's own ranges to a fixed point didn't settle). The flops are stratified by texture (suit pattern, pairing, high card, connectedness; 34 strata, at least two flops each) and weighted by the share of all flops each stands for: with random flops the two halves put the opener's calls against a 3bet up to 22 points apart, with weighted flops at most 10 (tests TestRealizationTable; the halves are in tests/data). The table is keyed by role as well as position (the caller, or the aggressor whose bet was called): position alone mixed an in-position 3bettor (realizing 1.15 of its equity in a 3bet pot) with an in-position caller (0.97), and with that mix the 3bettor trapped KK and the opener folded KK to the AA-heavy jams; by role KK calls them everywhere. `scripts/fit_realization.py` fits a smooth model to it (`gto/pf_realize_model.py`): per pot type, role and position, eight coefficients on the hand's equity against the bucket (x = e - 0.5: x, x·abs(x)) and its shape (pair, suited, connected, pair*x, suited*x). It stays off (pf_solve.MODE = "param"; "table" and "model" switch to the measurements; tests TestRealizationModel) because the measured caller-aggressor gap looks too large: callers realize 0.79 (out of position) and 0.97 (in position) in 3bet pots against the aggressor's 1.02 and 1.15, and about 0.6 in 4bet pots against 1.25, so in-position openers called 3bets less than the charts (BTN vs BB 28% against 34%, CO vs BB 25% against 41%) and out-of-position ones more (CO vs BTN 47% against 18%). Neither the postflop solver's depth nor its bet sizes explain the gap: playing the river instead of scoring it by equity moves these about a point, and four bet sizes a street (25-125% on the flop, 33-150% later) instead of two move them under one (40 stratified flops, BTN vs BB and CO vs BTN 3bet pots and the CO vs BTN 4bet pot). Nor does the 3bettor's range: widening BB's 3bet against a BTN open from the chart's 8.5% of hands to the published 11.5% leaves the BTN caller at 0.99 (40 flops). In aggregate an in-position caller realizes about its equity; the call rate falls short of published solves because marginal hands (KQo, A5s) measure a few points below their equity while strong hands run above, and checking that needs per-hand EVs from a commercial solver. So either the gap is real for the charts' ranges (a capped calling range against a strong, uncapped one: in the CO vs BTN 4bet pot the caller holds JJ, TT, AQs against KK+, AK) and the charts call too much, or the subgame's calls lack something the charts encode; settling it needs outside solver data. Public equity-realization figures cover only some of the roles. The GTO Wizard aggregate reports quoted by Poker Academy give 1.14-1.17 for the in-position raiser of a single-raised pot and 0.78-0.87 for the out-of-position caller (BTN vs BB, LJ vs BB, limped pots, 25-184 flops). A GTO Wizard blog example (BTN 3bets CO, 552r, 100bb) splits equity about evenly with EVs of 10.78bb against 7.72bb, which is 1.17 against 0.83. The measured aggressor in position (1.15) and caller out of position (0.80) match these. Nothing published covers an in-position caller of a 3bet. Rules of thumb (deepfold: 1.08-1.15 for late-position flats; Tournament Poker Edge: 1.25-1.5 for an in-position flat of a 3bet) put it above the measured 0.97, and per-class figures put offsuit broadways and offsuit aces below their equity and pairs and suited connectors above it, as measured (in-position caller: Axo 0.80, offsuit broadways 0.86, suited connectors 1.07, 22-66 1.11). The model mode now adds 0.18 to the in-position caller's realization in 3bet pots (pf_solve.MODEL_SHIFT, 0.97 to 1.15), which brings the in-position openers' calls to the charts (BTN vs BB 41% against 34%, CO vs BB 31% against 41%, BTN vs SB 38% against 36%; calibration score 0.30 against the parametric model's 0.58 and the unshifted model's 0.32). The out-of-position openers still call 2-2.5 times as often as published solves (CO vs BTN 47% against 18%, UTG vs CO 39% against 18%), even though the measured 0.80 matches public figures for that role. Fitting the charts would take it to about 0.64, against the 0.78-0.87 published for out-of-position callers, so the shift isn't applied there. The 3bettor also jams too often over a 4bet (28-39% against the charts' 15-39%). So the default stays "param", and the solver stays blocked: what's missing is per-hand EVs for an out-of-position caller of a 3bet. |
| Exploit mode | `gto/exploit.py` | The pool is the equilibrium plus common leaks, frozen (it doesn't adapt to you): facing a first bet it folds too often at every size on every street, most of all to small bets: on the river +20% of the range at a third pot, +13% at three quarters, +7.5% at an overbet, from mass data analysis of a real pool (GTOkiller, GG Poker NL100-200, BTN vs BB third barrels: 54.7% folds to half pot, 57.8% to pot, 70.6% to 175%, against ~33/50/64% at equilibrium); on the flop +8.5% at a third pot, tapering to +5% at an overbet, from the same source's c-bet analysis (47% folds on K74r against ~40% at equilibrium, ~45% on 652ss against ~35%, barely changing between a third and half pot; bigger flop bets unmeasured, found split by size nowhere above half pot), where the old guess had the pool calling small flop bets too much; on the turn +10.5% at a third pot, +7% at three quarters, +6.5% at an overbet, from Upswing's LeakBuster data of 500nl+ regulars' 2025 games (over-folding a delayed turn c-bet by 11/6/6 points at 30%/66-75%/120-150% pot and a turn probe by 10/8/7 at block/medium/overbet sizes; it replaces +14/+10/+6 interpolated between flop and river; one GG spot on Maxploit, a 150% turn overbet folded 72% against 58%, suggests lower stakes over-fold turn overbets more); the extra folds are its weakest calls, at most half of its equilibrium calls and never its top 10% of hands on the board (as a flat share of the range, a turn jam that equilibrium calls with a few percent lost every call, queens full included), the extra calls its strongest folds; it raises less (weak hands keep 30% of their raises, the rest call; a judgment call that one measured spot since bears out: GTOkiller has GG NL10-50 regulars raising a third-pot SB c-bet on K84r 9% of the time against GTO's 18%, and here the pool keeps 0.46-0.48 of equilibrium's raises in that spot; other figures since agree, none precisely: the same regulars re-raise a raise 16% against 25%, a GG river check-raise sits 10 points under GTO and is 95% two pair or better), and it bluffs less (bets from its weakest half are cut to 80/70/57% on flop/turn/river; the river from Upswing's same data: 500nl+ regulars' heads-up river bets in small pots are 12-25% low pairs or air by size against GTO's 18-37%, the bluffs kept at 0.55-0.62 of equilibrium's, 0.57 on average, where the old guess was 0.4; river showdowns are a fair sample, since a river call doesn't depend on the bettor's hand; the flop and turn are judgment calls, the turn between flop and river, since the only turn figure found comes from shown hands, which under-show turn bluffs). Strength is absolute: the hand's percentile on the board, averaged over runouts before the river. Preflop it is fitted to population HUD stats (Hand2Note mass data of GGPoker 6-max, all players, NL50 to NL500, published by Bluffaces: VPIP 28.75, PFR 18.7, 3bet 8.9%, fold to 3bet 48.25%, 4bet 12.75%, against the charts' 24.7, 17.9, 8.7%, 54.2% and 14.5% over every line weighted by how often it happens; scripts/calibrate_pool.py): facing an open it 3bets 1.12x the chart's share (the extra from its strongest calls); its extra hands are spread evenly over every cheap way in, since no data found splits VPIP by position or line: first in it enters, and facing one raise it flats, with 9.4% of what the chart folds (strongest first; fitted to VPIP; first in, the strongest 60% of those hands open-raise and the rest open-limp, fitted to PFR), the big blind excepted (the chart's unraked defence is already wider than raked solves, and the one, unsourced, population figure has big blinds folding more to a button open than the chart, not less); before, all of it went to cold flats (non-BB calls facing one raise: 20.3% of those hands, now 14.0% with limpers calling an iso-raise included, against the chart's 5.5%); as the opener facing a 3bet it calls with 12.6% more of its range (its strongest folds; its wider opens are weak and fold to 3bets); its 4bets are the chart's. The pool's tree then gives VPIP 28.8, PFR 18.7, 3bet 8.9%, fold to 3bet 48.2%, 4bet 12.8%, open-limping 2.8% of its chances (with every extra first-in hand limped and the old, tight iso charts it gave PFR 16.4 and 4bet 14.8%). The old guess (3bets cut to 35%, 4bets only QQ+/AK) had the pool 3betting 4.1% and folding 57.5% to 3bets. Each street is still solved to equilibrium; the pool's actions narrow its range through the leaky strategy, and hero's EVs come from a best response to it over the rest of the tree. Actions whose best-response EVs are within the solve's noise of the best (its distance from equilibrium as a share of the pot, at least 0.5%) count as tied, and a tied hand plays the equilibrium mix among them: a short browser solve otherwise picked one at 100% on noise (AQo checking at 6.45 over bets at 6.40 in a solve 3.1% of the pot from settled). Preflop, hero facing a raise goes through the off-chart re-pricing (see Preflop) with the pool's ranges and responses: calls by equity against what the pool actually raises with, value raises by equity against what the pool continues with, bluff raises scaled (up to 1.5x) by how often the pool folds compared with the chart. Facing a jam over a raise, the jam range is the pool's. Opens follow the chart (the pool 3bets more, flats more and folds less to 3bets); a pool limper's range is its limps, hero's replayed range stays the chart's, and multiway postflop spots keep the betting rules. |
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
  (the page's default raise size), or check / bet 33% / bet 66% pot, with opponents
  following simple range rules: facing a bet they fold together a quarter less often
  than a balanced defence (people call too much), with everyone keeping the hands above
  one strength bar, so a weak range (a lead after checking twice) folds more of itself
  than a strong one (a call of that lead), and the strongest part of what continues
  raises. Raises inside the trials go to 3x the bet plus the calls in front of them
  (a standard raise: 22 bb over a 5.5 bet and one call). When they bet it's 66% with value
  from the top of their range plus balanced bluffs, or 33% with the next, thinner value
  and its own balanced bluffs (SMALL_BET: GTO Wizard's 3-way flop study, 100bb, LJ
  opens and both blinds call, has the opener checking 34%, betting a third of the pot
  56%, two thirds 8% and pot 1%, against 23/38/21/18 heads-up with the big blind, so 85%
  of its 3-way bets are the small size; bigger multiway bets come where the bettor holds
  the nut advantage. These rules don't reproduce that mix: the 6-max UTG opener into
  both blinds checks 67%, bets 66% 26% and 33% 7% over 60 random flops, as the big bet
  takes the top of the range and the small one the thin layer below; hero's size is
  chosen by EV). The value is what beats most of what
  the players it's bet into would call with, from their actual ranges (an opener betting
  into two flatters values thinner than a flatter betting into the opener), and the
  bluffs are the hands without showdown value that most often end up strong (draws and
  overcards first; on the river the weakest hands). Bluffs from the bottom of the range
  and value as if everyone held the bettor's range made a 66% bet nearly all sets and
  overpairs plus air, which a draw out-continued top pair against. Players still to act
  behind you can call or raise after you call; they faced the bet with you, so each folds
  as often as when everyone it was made into faces it together (39% of the range continues
  against a pot-sized bet with two of you facing it; pricing it as a fresh bet into
  the pot your call had grown kept 81%, junk included, and that dead money paid for
  your calls: K5s on K72 called a pot bet at +19 bb, about +10 now). Each bet size gets
  its own continuing ranges, so a third of the pot keeps more of the field in (3-way,
  each keeps about 57% of their range against 33% and 45% against 66%) and gets
  raised more often. When you check, the first player behind you whose betting range
  at either size holds their hand bets it, and the others face that bet the same way as
  players behind a call (call, raise with the strongest part, fold the rest): those
  between the bettor and you act first, and you fold, call or check-raise knowing what
  they did (per bettor, size and what the players between did, whichever pays over the
  trials; a call your equity against the bettor and the callers already pays for is made
  even where those noisy splits came out slightly negative), those after you act once
  you've called. A check-raise is faced by everyone still in (the bettor, the callers
  between, the players behind) by the same joint fold rule, the strongest part of what
  continues re-raises 3x and you call that by equity. The notes and `check_line` say how
  often someone bets after your check and how often you'd call, check-raise or fold. On
  8d6h3c checked to the big blind 3-way (CO opens, BTN and BB call), top pair Ad8s used
  to continue against 46% of those bets while the open-enders 7c5c and 5c4c continued
  against all of them and the gutshot T9s 44%; now top pair continues about 96%, the
  open-enders 96-100%, the gutshot about 30%; a set of sevens on K72 check-raises most
  bets and air never does. A raise can be re-raised, and every line plays to the river
  by the same rules: on each later street players act in seat order, an opponent bets
  66% or 33% by the betting ranges above, you lead 66% when you're ahead of every range
  that would call it, else 33% when ahead of what calls that, and everyone still in faces
  a bet (opponents call, raise or fold by the joint rule; against your bet they call or
  fold: a range capped by checks raising its best weak hands stacked off 77 against top
  pair), you raise when ahead of each range that would continue, otherwise call or fold
  by your equity against the bettor and the callers (on the turn, also counting the river
  bet that may follow). One raise per later street, which everyone still in then calls or
  folds. Trials continue until the gap between the two best actions is clear of the
  sampling noise or under 2% of the pot (or a time cap), and the notes give that gap with
  its error.
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
  `scripts/calibrate_multiway.py`: on ~200 heads-up flop spots the heuristic's pick
  gives up 0.10 bb on average against the solver's best action, and its EVs are
  off by 1.0 bb on average (1.35 bb before fitting). The "players call too much"
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

copies the page, the worker and the `gto` sources into the `gto-solver/` folder
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

* Preflop charts are approximations written by hand, not solver output, and the
  off-chart adjustments are rules on top of them. `scripts/preflop_sweep.py` runs
  every standard line for 2-9 players and flags broken frequencies and any hand continuing less often than one it dominates (pairs, kickers, suited over offsuit; the strategy is also kept that way at run time); a test keeps
  key 6-max continue rates near typical solver output. Replace them with real chart
  exports if you have them; the format is a range string.
* Limps, antes, straddles and stack depth are handled by adjusting the 100bb
  no-ante charts (see Preflop above), not by charts built for those games.
* Rake (optional) re-prices preflop calls (and push/fold) by one expected-rake rule (2.5x the flop
  pot, fit to one published raked solve), the charts fit to raked solves only past its 5%/4bb; 3bet
  and 4bet ranges aren't re-priced; equity leaves where the tree stops early are raked on the pot as it stands, and the multiway
  heuristic rakes its payouts but not the betting rules inside its trials (see Rake above).
* Exploit mode's pool is fitted to public population data where any was found (flop and river
  folds, turn folds and river bluffs from 500nl+ regulars, preflop frequencies, and one flop spot's raise
  frequency); folds to big flop bets, how much less the pool bluffs on the flop and turn, and how its extra preflop hands split
  between opens, open-limps, flats and the big blind are still judgment calls. On a rough (Fast, flop or turn)
  solve the best response also picks up the solve's own noise.
* One effective stack for the table.
* The bet-size abstraction is coarse; real actions are mapped to the nearest size.
