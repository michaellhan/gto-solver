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

Open http://127.0.0.1:18877/ . Each view has its own URL, on the website and the local server alike:

| URL | View |
| --- | --- |
| `/gto-solver/` | the solver (advice); a share link `/gto-solver/#h=…` opens its spot here |
| `/gto-solver/practice/` | Practice |
| `/gto-solver/drills/`, `/gto-solver/drills/<spot>/` | Drills (on a spot: `srp-btn-bb`, `3bp-bb-btn`…, the flop library's ids with dashes) |
| `/gto-solver/reports/`, `/gto-solver/reports/<spot>/` | Flop reports |
| `/gto-solver/session/` | Analyze session |
| `/gto-solver/import/` | Import hand |

(locally without the `/gto-solver` prefix: http://127.0.0.1:18877/practice/). Opening a view pushes its URL
and closing it goes back to the solver's (or Practice's, when the view was opened over Practice); the browser's
back and forward buttons switch views, the tab's title names the view ("Practice · GTO Solver"), and picking another
spot in the reports or drills updates the URL. A URL opened directly opens its view: Practice is on at
`/practice/` and off at the solver's URL (whatever it was on the last visit). The list is `ROUTES` in
`gto/static/index.html`; `gto/routes.py` reads it, and since GitHub Pages has no rewrites `scripts/deploy.sh`
writes each URL's `index.html`, a copy of the page with a `<base href="../">` that points its scripts, worker and
library fetches at `gto-solver/` (`gto/server.py` serves the same paths, and redirects `/practice` to `/practice/`
as GitHub Pages does). (The same page is what is deployed to the
website; there it detects that no server is present and runs the identical
Python code in a web worker with Pyodide. Pyodide's numpy is 10 to 30 times slower
per iteration than native, so the browser solves a smaller game: below a flop solve the turn and
river are played by hand buckets (32 in Fast, 64 in Accurate), and from the turn the river is one
card per class. The local server (`gto/native.py`) plays the same turn and river menus and Accurate's
flop per combo; see Accuracy vs speed, Browser timings, for what each request takes in the browser.) Pick the number of players (2 to 9), your seat
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

Whenever it becomes your turn the advice panel solves the spot. The **Quick / Fast /
Accurate** toggle in the header (remembered between visits) picks the solve
time: Quick is about 2 s either way (the requests Practice makes at Fast, below: `QUICK_BUDGETS`, `"quick": true`;
rough on the turn, a 3-way turn or a 3-way flop the [3-way library](#precomputed-3-way-flop-library) doesn't hold left to the betting rules); Fast is 8 s in the browser and 2.5 s against the local server (2.5 s per earlier street too); Accurate
is about 60 s in the browser and 30 s locally (12 s per earlier street), with more time for the earlier
streets used to narrow ranges. River spots are solid either way; turn spots need
Accurate to get close to a full solver (see Accuracy vs speed below).
The panel shows the recommended action with its size (preflop: 2.5bb
opens, 3bb from the small blind, 3x or 4x 3-bets, 2.3x 4-bets, then all-in; over limpers
3x (4x from the blinds) plus 1bb a limper, and a squeeze adds one bet per caller;
postflop: the solver's bet and raise sizes in big blinds), the mixed strategy for your exact
hand with EVs, your whole range's totals at that spot (the percentage and number of
combos that fold, call, check, bet or raise each size, with a bar), and a 13x13 grid of
that range by hand with your hand boxed. Postflop, tabs beside the grid open GTO Wizard-style
**Breakdown** and **Equity** views of the spot: your range by hand category (made hands: quads or better down to
top pair by kicker, middle and weak pairs, ace high, nothing; draws: combo, flush draw, open-ended, gutshot,
backdoor), each with its share of the range, a bar split by the solver's mix, its equity and EV, and the villain's
range the same way without a mix; and each range's equity distribution (its hands' equities against the other
range sorted high to low), with average equity, EV and the share of each range at 80%+ equity (the nut advantage).
3-way solves get the Breakdown only. Opening a tab asks again with `"breakdown": true` on the same state, answered
from the solve the advice came from (no new solve; ~0.3 s native, ~0.5 s in the browser on a flop: the equities
every turn and river on the turn, an even sample of flop runouts, `gto/breakdown.py`); the views are
`gto/static/breakdown.js`, loaded when first opened, and the tab picked stays for later advice. Click an advice row to record it as what you did and keep going. Once the advice
is in, your action buttons switch to the solver's own sizes with its frequency on each,
so what you record is exactly what was solved. Postflop, a badge shows how well the solve
converged (exploitability in % of the pot: converged under 3%, fair under 10%, rough
above) with a one-click re-solve on Accurate; a rough solve never grades a line ✗. A jam the solve's tree has no all-in
for (all-in is offered at 2.5 pots behind or less: a 15x-pot flop jam isn't in it) is solved for its grade: the spot is
asked again with `"jam": true`, which offers all-in at every decision of the street advised on (`Config.root_allin`;
not the earlier streets' narrowing solves, nor the flop libraries), and the grade, its EV and the coach's card fill in
when that answer is in (A2s on 776r behind a HJ check, 100bb: the jam 0% at -2.28 bb, the 75% bet +2.53). Always in
the tree it cost too much: a K72T turn 60 iterations in was 4.0% of the pot from equilibrium against 1.6%.
When you're facing a bet the prompt shows the pot odds a call needs.
Solver stats and the villain's inferred range (as text and as a 13x13 grid shaded by weight) sit behind a "Details" toggle.
Each decision you record is marked against the advice that was showing (✓ in
the mix, ~ rarely played, ✗ outside it, with the EV given up), and when the
hand ends the advice panel lists them as a hand review with the EV given up per street.
Click any of your decisions (in the review or the timeline) to replay the hand from that
point; later actions and board cards are cleared, with Undo. While others act, the
last advice stays visible but dimmed and labelled. The **⋯** menu has a four-colour
deck (diamonds blue, clubs green), Sounds on/off, a switch that puts the advice on the left, and Reset layout.
Sounds (on by default, kept in the browser) are short effects synthesized with Web Audio, no files: in Practice,
the deal and each board card, a knock for a check, chips for a call, bet or all-in, a swish for a fold, a chime
when your move is in the solver's mix and a low note for a mistake, and a pot won or lost; in Drills, each answer's
verdict.

**Import hand** (in the ⋯ menu) takes a pasted hand history: PokerStars, GGPoker/Natural8, 888poker,
partypoker, Winamax, WPN/ACR, Ignition/Bovada and iPoker (its XML: Bet365, William Hill…), the site detected,
2 to 9 players, cash or tournament. Seats map onto the page's positions from the big blind, amounts
become big blinds, antes are averaged per player, a 2bb straddle from the first seat is kept, sitting-out
seats are dropped and returned bets are ignored; the history needs your hole cards. Every seat keeps its
own stack (after the antes, as the history has it), so a short all-in, its side pot and an uncalled bet play
out as they did. The parser is `gto/static/hh.js`,
loaded only when the dialog opens. **Review whole hand** (offered after an import, and at the end of any
hand with ungraded decisions) solves each of your decisions in order at the current Fast/Accurate setting,
grades it as if it had been entered live and ends on the hand review; it shows "decision 2 of 5" while it
runs and can be cancelled, keeping the grades so far.

**Analyze session** (in the ⋯ menu) reviews a whole session, like GTO Wizard's analyzer: drop, choose or paste a
hand-history file of any number of hands (any site above, mixed sites too). `hh.js`'s `splitHands` cuts it at each
hand's header (iPoker: a `<session>` per `<game>`), each hand is imported as above, and hands that aren't Hold'em,
lack your hole cards, have no decision of yours (a walk), repeat one already read, or can't be read are counted and
skipped. Pick a scope: **Preflop only** (the charts: instant), **Flop-library spots** (preflop plus the flops the
[precomputed library](#precomputed-flop-library) holds, asked with `"library_only": true` so any other flop is
skipped rather than solved: instant) or **All streets** (every flop, turn and river solved at Fast: with the local server an 80-hand session averaged 0.7 s a flop decision (library ones
included), 3.7 s a turn and 2.6 s a river; in the browser about 8 s plus 3 s per earlier street); each shows its decision count and estimated time. A queue
then asks each decision in order, exactly as Review whole hand would (the page's request at Fast against GTO,
graded the same way), with progress, an ETA from the measured time per street, Pause (after the decision in flight),
Stop, and the session kept in IndexedDB so a reload resumes it; a finished narrower scope can be widened. The
report: hands, decisions graded, EV lost (bb and bb/100; postflop only, as preflop chart answers carry no EVs and
are graded by frequency), share in the solver's mix; tables by street, position, preflop situation (RFI, vs limp,
vs open, squeeze spots, SB vs open, BB defence, vs 3bet, cold vs 3bet, vs 4bet+) and postflop spot (pot type, IP/OOP,
facing a bet / first to act / checked to); your frequencies against the solver's for the same hands at the same
decisions (opens, BB defence, folds and 4bets vs 3bets, flop c-bets, folds and raises vs c-bets, turn barrels,
river bets, folds to turn and river bets); the 20 biggest mistakes by EV (a click opens that hand in the page
rewound to that decision, Undo brings the rest back) and the clearest off-chart preflop plays. **Export CSV / JSON**
downloads one row per decision (hand, site and number, seat, cards, board, situation, your action and its
frequency, the solver's best, EVs, EV lost, grade, source). The panel is `gto/static/session.js`, loaded with `hh.js`
when first opened.

Each analyzed session's summary (not its hands) is also kept in the browser (IndexedDB, up to 200 sessions; a file
analyzed again, in any scope, replaces its earlier entry), and **Across your sessions** (or **Your sessions so far**
before a new one) sums them: sessions, hands, bb/100 lost, share in the solver's mix; the **leaks** that show over many
sessions, ranked by how far off times the square root of the decisions (a frequency of yours 10+ points from the
solver's on the same hands over 15+ decisions, a preflop situation with under 80% of your actions in the chart's mix
over 15+, a postflop spot losing 0.5bb+ a decision over 10+); each session's date, file, hands, share in the mix and
bb/100 with a line of bb/100 over time (Forget removes one, Clear history all); and the street, preflop-situation,
postflop-spot and frequency tables summed over every session.

**Flop reports** (in the ⋯ menu) are aggregate reports of the [precomputed flop library](#precomputed-flop-library):
pick a spot (BTN vs BB, …) and a node (out of position first; in position after a check; out of position facing
in position's most common bet after checking) for a table of its 184 flops: each flop's range frequencies as a
stacked bar and numbers, both players' EV (bb, at the start of the flop) and range equity. Click a column to sort;
filter by high card, suits (rainbow, two-tone, monotone), pairing and connectedness, or group by one of them for the
groups' averages (each flop weighted by how many of the 22100 flops it stands for; a group row filters to it).
Clicking a flop loads that hand: the spot's preflop action, the flop, the action up to the node, and you as the player
to act with the hand that mixes most there, so the precomputed advice appears at once. The data is one small
`flop_library/<spot>/report.json` per spot (~45 KB, ~12 KB gzipped; ~135 KB on 552 flops, 430 KB / 80 KB gzipped for BTN vs BB's 1755), written with the library's index
(`flop_library.build_report`, see Building below); the panel is `gto/static/reports.js`, loaded when first opened.

**Drills** (in the ⋯ menu) drill one flop decision of the precomputed library, like GTO Wizard's trainer: pick a
spot and a node (out of position first; in position after a check; out of position facing a bet after checking)
and optionally a texture (the reports' high card, suits, pairing, connectedness). Each hand is a flop drawn by how
often it comes (any stored bet size by how often in position makes it), a hand drawn from the acting player's range
there (its weight entering the flop, off the board, times its own frequency of the line so far) in random suits.
Pick an action (keys 1-4) and the hand's GTO mix and per-action EVs show at once with the grade: **Best** (the
highest EV, to the library's step of 1% of the pot), **Correct** (an action the hand plays 10% or more: "mixed"), or by
the EV given up against the best, Inaccuracy (under 2% of the pot), Mistake (under 8%), Blunder. Space or Enter
deals the next. The session's accuracy, average EV loss (bb and % of the pot) and streak show under the hand, and
each drill's history (hands, accuracy, last 100, best streak) is kept in the browser; **Open in the solver** loads
the hand into the page. `gto/static/drills.js` decodes the library's entries itself (no solver needed: answers are
instant, also before the in-browser solver has loaded) and reads the hand weights from `flop_library/<spot>/ranges.json`
(both ranges entering the flop, written with the index, fingerprinted as the index's).

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

**Practice** in the header plays hands against GTO opponents (**Exit practice** in the same place goes back
to the solver), with the player count, stacks (each seat's own, if set), antes, straddle and rake of the setup. The
row of seats under **Your seat** at the top of the coach panel picks how you sit (this hand's seat is outlined; arrow
keys move along the row, Enter or Space picks). **Table**, the default, is a ring game: you keep one seat and the
dealer button (the D chip on the felt) moves one seat clockwise after each hand played out, so your position goes
BTN, CO, HJ … UTG, BB, SB and round again, the seats relabelled each hand; every seat's chips carry over from hand
to hand (shown beside each seat; the solver and the charts get each seat's own stack, as with per-seat stacks in
the setup), a seat under a quarter of the setup's stack rebuys to it at the next deal ("UTG rebought to 100"), a
stack typed in the setup's seat stacks is taken over, a new stack or player count in the setup starts a new table,
and **Reset stacks** (beside your stack) puts every seat back to the setup's stack from a new hand. The table, the
button and the hand in play are kept in the browser over a reload. **Random** deals a random seat each hand, and a
position plays every hand from it (setup stacks, not carried over). Every seat's two cards come from the same
shuffled deck as yours and the board's. At a showdown everyone still in shows their cards on the table, the best
hand wins ("CO's Q-high flush beats your two pair, kings and sevens"; "Split pot: you and SB tie with …", the
deciding kicker named when two hands share a name), each side pot goes to the best hand among the players in it
(main and side pots listed, a tie split evenly, rake taken in proportion), and the seats that collect are lit with
what they won and their chips after the hand. When you fold, the hand ends for you: **Run it out** (beside Next
hand, and in the coach) plays it on without you, every opponent by its strategy for its own cards, to its real end,
with everyone's cards shown and its showdown named, and the table's chips go as it went; with one opponent left it's
**Show hands** (everyone's cards and the rest of the board). Neither changes your result or the score. Without a
run-out the others' chips go as if they checked it down from there, each winning at most what it put in from each
player. The row under it picks the spots: **Any
hand**, or a drill, **Raised pot**, **3-bet pot** or **Blind vs blind** (from a blind; a drill deals a random seat, or
the position picked, not the table), where your preflop is played
for you too (from the chart for your hand, not graded) and the hand dealt again ("Dealing a 3-bet pot…", the table
dimmed) until it reaches the flop heads-up with you in it as that kind of pot, so you play it from the flop. The first
time, a two-line hint in the coach (and the phone's dock line) says which seat is yours and how to act, until **Got
it** or the end of the first hand. From the flop on, the coach names under the board the best hand your
cards make with it ("Your hand: two pair, kings and sevens"). The table marks your seat "you" and the
seat to act (an opponent's glows while it thinks); every other seat's hand and the board are dealt up front, and the
other seats move by themselves ("BTN is thinking…" while they do): preflop from the chart for their actual hand,
after the flop drawn from the solve's mix for it (heads-up, the same solve that answers your own spot, so a reply
to your move is nearly instant; in a 3-way pot your own solve too once you've acted on the street, read off its tree
for their hand; before that the advisor from their seat: a 3-way solve where it applies, else the heuristic's pure
pick). They play GTO whatever the Opponents setting (hidden while practising, with the other solver-only controls).
On your turn the spot is solved in the background so your move can be graded: until it is, the coach and the action
dock say "Solving your spot…" and the (large) action buttons are disabled; each shows its key (F fold, C or X check
or call, 1-3 the sizes, R a size of your own, A all-in). Once it is solved the page looks ahead while you decide: the
lines you're likeliest to take (below) are played on (the opponents' replies drawn as they will be, the next street
dealt) up to the next request that takes a solve, your own next spot or an opponent's move on a new street, and the
answer is used if you take that line, so the wait for it goes into your thinking time. A local server stops a
look-ahead your move made useless; the browser's solver can't stop one, so there it looks ahead only at Fast, where a
wrong guess costs at most one quick answer's ~2 s. Until your first decision a dashed card says where
the verdict will be; right after each decision the coach shows
its verdict (Best move, Good, Inaccuracy, Mistake) in colour on a card tinted to match, the EV it gave up and the
solver's mix as a bar (each action of 20% or more named in it) with its frequencies (your move among them, at 0%
when the solve hasn't it); the card is headed "Last decision · flop" (at the end "Your last decision"), and once
it's your turn again it fades, so it never reads as a verdict on the spot above it (the phone's dock line then
starts "Last:"). The EV a move in the solver's strategy gave up is shown muted, not red (its mixed actions are worth
the same in theory); at Fast, Practice asks for quick answers, about 2 s each (`"quick": true`, budgets
`QUICK_BUDGETS` in index.html: 1.5 s on the flop and turn, 0.8 s on the river, 0.3 s an earlier street; an earlier
heads-up street that isn't in the flop library or already solved (your advice on it) is narrowed by the betting rules,
its checks keeping the range whole, rather than solved (a solve's 10 least iterations took 2-4 s natively and were
trusted 0%: 88% of flop narrowing solves at half a second were 5%+ of the pot from equilibrium; over 24 fuzzed
heads-up turns and rivers quick answers took 1.3 s natively where they took 3.6 s, as close to Fast's as before); no second
budget for an unsettled street; 3-way pots solved on the river only, in 1.8 s (`advisor.MW_QUICK_BUDGET`), a 3-way
flop's first decision in the commonest 3-way pots read from the [3-way library](#precomputed-3-way-flop-library) at
once, other 3-way flops and turns answered by the betting rules; measured in Pyodide, `BENCH_STATE='{"quick":true}'
BENCH_BUDGETS='{"fast":[1.5,0.3]}' scripts/bench_browser.mjs`, medians of 3 on an idle machine: a turn 1.9 s, 26
iterations, 6.4-6.9% of the pot from equilibrium (was 22-23, 8.1-8.9%) where Fast's 8 s settle it to 0.12%; a river
(budgets `[0.8,0.3]`) 1.4 s, 412 iterations, 0.04%, its turn narrowed by the rules (was 1.8 s, 244 iterations, 0.08%,
with a 1 s turn solve trusted 0%, and a 272 MB wasm heap, now 76 MB); a 3-way river 1.8 s,
14 iterations, 2.6%; a flop from the library at once). While you decide, the lines you're likeliest to take are solved ahead: your
hand's mix blended half and half with what you usually do (fold, check or call, bet or raise; facing a bet or not;
kept in this browser as `gto-practice-habits`), up to two lines at least 20% likely, a bet at the solver's size, never
a fold; the solver's own Fast (8 s) solves 3-way flops and turns in a smaller game (`advisor.MW_FAST_STREETS`), and a multiway spot left to the betting rules (4+ players, someone all-in, a 3-way flop or turn in a quick answer) is
graded by the EV your move gives up against the rules' best, as a share of the pot (their frequencies are all or
nothing: within 3% Good, within 10% an Inaccuracy, more a Mistake; `HEUR_OK`, `HEUR_MEH`), the card listing the rules'
EV for each action and saying it's a heuristic, not a solve; **Details** opens why and each action's EV (the choice is remembered), and from there **See how the
solver plays your whole range** opens your range at your last decision under the coach (never on your turn), headed
"Your range · flop · BTN vs BB": the advice panel's Grid (every hand class's mix, yours outlined; hover or tap a hand
for its mix), Breakdown and Equity, without the rows the verdict card already gives (a decision whose solve the page
has moved on from, or one from the lists below, has none). At the end the
opponents' cards are shown at showdown with your result, the verdict on your last decision and the hand's
decisions with the EV each gave up, and **Next hand** (N) takes the action buttons' place (N does nothing mid-hand). The score
(hands, how often your action was in the solver's strategy, average EV given up per decision, chips won; **Reset**)
is kept in the browser, with your last 60 decisions; on a wide screen the eight most recent are listed under it,
newest first, each opening its spot like a mistake below. **Review mistakes** under the score (with how many) lists the six of them that gave up
the most EV (then the rarest in the solver's mix: preflop has no EVs; never the solver's top move), each with its hand number, cards and street,
and a click (or Enter) opens that spot's board, the action before you and its verdict. On a wide screen the Hand
panel scrolls on its own while practising, its newest line in view, so the table and the action buttons stay on
screen; the action buttons wrap to a second row rather than cut a label short, and the one for a size of your own
reads "Bet" or "Raise to" over "your size…" (R opens it). Each new hand folds the setup back to its one-line
summary; a change to the table there (players, stacks, ante, straddle, rake) deals a new hand, the setup left open,
and while it's open on a wide screen the table is smaller and the left column scrolls as one, the action dock at its
end, rather than squeeze the Hand panel behind the dock. On a phone the coach's verdict comes
straight under the table (the table and the dock already say whose turn it is), and the action dock carries it in
one line (the verdict, the EV it gave up, the solver's top choice; the result once the hand is over) that a tap
scrolls to the full card; the line is always there, so the dock never changes height. Folding ends the hand for you; Undo and replaying a decision are off while
practising. Under each verdict's **Details** (the coach's card and an opened mistake or recent decision) two buttons explain
a decision. **Copy for AI** puts it on the clipboard as a prompt for any AI chat ("Copied"): the game (players, stacks,
blinds, ante, straddle, rake), your seat and cards, the board and every action by street with its size in bb and the
pot, the decision (pot, what's to call, the stacks behind), Hero's hand read from the cards (made hand, each draw with
the cards that complete it, the board's suits and pairing, position after the flop: worked out by the page, since a
model left to it gets draws wrong), what you did, the solver's mix with each action's frequency and EV, the EV lost
and the verdict, and a request for a short answer: a one-sentence verdict, three reasons, one takeaway. **Explain
with AI** streams Claude's explanation (Claude Sonnet 5.5, extended thinking off, at most about 70 words: a bold
Verdict, three one-line reasons shown as a list, cut at three if it writes more, and a bold Takeaway; the solver's
numbers and the hand reading taken as given) into the card. Through the proxy (`EXPLAIN_PROXY` in `practice.js`, see
[Explain with AI proxy](#explain-with-ai-proxy)) it works for every visitor with no key, a few a day each; when they're
used up the card says "Daily free explanations used up — add your own key" and offers the key form, and a small
**Use my own key** link switches to your own key (and **Use free explanations** back). Without the proxy, or with
your own key chosen, it goes straight from the browser to the Claude Messages API with your own Anthropic API key: the
first time it asks for one (a link to console.anthropic.com to get one); the key is kept only in this browser's
localStorage and sent only to api.anthropic.com (the site is static, nothing of ours in between), and **Forget key**
removes it. A refused key (401), a rate limit (429) or no network say so in the card with **Try again**; each
explanation is kept in memory for its decision, so opening it again costs nothing. The page loads `practice.js` the first time Practice is turned on.

## Explain with AI proxy

`cloudflare/explain-proxy/` is a small Cloudflare Worker that lets Practice mode's **Explain with AI** work for
visitors without their own API key. The page POSTs `{prompt}` (the Copy for AI text, at most 8000 characters) to
`/explain`; the worker builds the Claude Messages API request itself (Claude Sonnet 5.5, the coach system prompt,
700 tokens at most, streamed: the page can't choose any of these) with the key kept as a Worker secret, and streams
the answer back. Only `https://www.michael-han.com`, `https://michael-han.com` and `http://localhost:*` /
`http://127.0.0.1:*` may call it (`ALLOWED_ORIGINS` in `wrangler.toml`; others get 403). Each visitor's IP gets 5
a minute (the Workers Rate Limiting binding) and 20 a day, and everyone together 1000 a day (`DAILY_PER_IP`,
`DAILY_GLOBAL`; daily counters in a KV namespace, by a hash of the IP), each answered 429 `{error, retryAfter}`;
Anthropic's errors come back as short JSON errors (a refused key as "proxy misconfigured", 429/529 as "busy, try
again"), never with the key or Anthropic's headers. Its tests: `cd cloudflare/explain-proxy && npm test`. The
worker isn't part of the site: `scripts/deploy.sh` copies only the page's own files.

It's deployed at `https://gto-explain.combatrobotics.workers.dev` (`EXPLAIN_PROXY` is its `/explain`; the KV id is
in `wrangler.toml`), so redeploying after a change is `npx wrangler deploy` in its folder. To deploy it afresh (a free
Cloudflare account is enough):

1. Get an Anthropic API key from the Console's API Keys page, <https://console.anthropic.com/settings/keys>: it starts
   with `sk-ant-api03-`. (Set a monthly spend limit in the Console too; with the default caps it's at most 1000
   short explanations a day.)
2. In the folder: `cd cloudflare/explain-proxy && npm install`
3. `npx wrangler login` (opens the browser to sign in to Cloudflare)
4. `npx wrangler kv namespace create COUNTS` and paste the `id` it prints into `wrangler.toml` (the
   `[[kv_namespaces]]` entry's `id`)
5. `npx wrangler secret put ANTHROPIC_API_KEY` and paste the key when it asks (it's stored encrypted by Cloudflare,
   never in the repo)
6. `npx wrangler deploy`: it prints the worker's URL, like `https://gto-explain.<your-subdomain>.workers.dev`
7. Set `EXPLAIN_PROXY` in `gto/static/practice.js` to that URL plus `/explain` (or tell Claude the URL), then
   deploy the site.

To check it: `curl -i -X POST https://gto-explain.<your-subdomain>.workers.dev/explain -H 'Origin: https://michael-han.com' -H 'content-type: application/json' -d '{"prompt":"Is folding 72o UTG right?"}'`
streams an answer. `npx wrangler tail` shows its requests live; the limits are vars in `wrangler.toml` (redeploy
after changing them). To stop it: `npx wrangler delete`, or roll the key in the Console.

## Use from the shell

Manual entry:

```bash
python -m gto.cli --hero BB --hole 9h8h --board Ks7d2cTh \
  --preflop "UTG:f,HJ:f,CO:r2.5,BTN:f,SB:f,BB:c" --flop "BB:x,CO:b2,BB:c"
```

Action tokens are `POS:act` with `f` fold, `x` check, `c` call, `r<to>` raise to,
`b<amount>` bet, `ai` all-in. Amounts are in big blinds. `--ante 0.2` adds an ante
per player, `--straddle` has the first seat straddle 2bb, `--rake 0.05 --rake-cap 3` sets a 5% rake capped at 3bb (see Rake).
`--players 4` deals 2-9 players (default 6), `--stacks "CO=40,BTN=25,SB=8,BB=25"` gives seats their own stacks and `--payouts "50 30 20"` makes it a tournament
(see Tournaments).
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
unless given (see Rake below). Optional `"payouts"` (the prizes still to be paid, first place first, a list
or a string like `"50 30 20"`) makes it a tournament, the advice in prize equity (ICM), with two prizes or
more; `"field"` adds the players at other tables (a list of their stacks, or `{"players": n, "chips": total}`;
see Tournaments below). Positions are the 6-max names `UTG HJ CO BTN SB BB`.
Optional `"sizes"`, `"ranges"` and `"optimize_sizes"` change the postflop game, `"locks"` the preflop one too (see Custom sizes and ranges below).

Response: per-action `freq` (the mixed strategy for hero's exact hand), `ev_bb`
(relative to folding now), `amount_bb`, a `recommended` action (highest
frequency, or highest EV when the solve is 3%+ of the pot from equilibrium and that
EV beats the top of the mix by over 1% of the pot), a `sampled` action (drawn from the mix, use this if you want to
actually play unexploitably), the range-wide frequencies, the villain's inferred
range, and solver stats. With `"breakdown": true` a postflop answer also carries `breakdown` (`gto/breakdown.py`):
`actions`, `pot`, `hero` and `villains`, each with `made` and `draws` categories (`c` combos, `eq` equity, the
hero's `f` per action and `ev`), the range's `eq`, `ev`, `nuts` (share at `nut_eq`, 0.8, or more) and `curve`
(equities sorted high to low at every 2% of the range; heads-up only), and the hero's own categories (`mine`);
`null` where the betting rules answered. `"library_only": true` (the session analyzer's instant scope) answers a
flop only from the precomputed library and returns `{"skipped": …}` for any other postflop spot instead of solving it.

`"opponents": "pool"` (the page's Opponents: Pool toggle) is exploit mode: opponents play like a
typical player pool instead of GTO. Preflop, hero's calls and raises are re-drawn against the pool's
ranges and responses; after the flop the advice is the best response (each hand's highest-EV action,
`freq` 0 or 1, with `gto_freq` alongside). See Exploit mode below. `gto_freq` is the chart's mix preflop and
after it the equilibrium of the same solve, which starts from the pool's ranges into the flop (and is never the
precomputed library's), so it can differ from the GTO setting's answer for the same hand; the page says so.

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
  the cap. Which calls a worse price removes is set by realized equity, not all-in equity (all-in equity
  times the share the hand realizes after the call, gto/pf_eqr.py, with the river correction of
  advisor._river_fix), the number of them by the price: against a button open at 5% capped at 4bb the big
  blind folds A2o, A6o, K7o and K8o, defends Q9o 33% and K9o/KTo 73/69%, and keeps 76s, 54s, T8o, 98o, 87o and
  every suited king (with all-in equity it did the reverse; gtopreflop.com folds A2o, defends Q9o 27%, every
  suited king, and calls A2s-A4s). 3bets and 4bets aren't re-priced:
  the big blind's 3bets already match the raked solves' (4.8 / 6.2 / 8.0 / 11.6 / 16.4% against UTG, HJ, CO,
  the button and the small blind, published 4.5 / 5.9 / 7.7 / 11.5 / 16.2), so rake adding any would only
  move them further off, and the other charts were fit to raked solves.
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
* 3-way solves rake every pot at their terminals, as heads-up.
* Multiway (three or more players see the flop): the Monte Carlo heuristic pays every pot hero wins
  less its rake (the pot without an uncalled bet; a split costs hero its share), so the EVs it ranks
  actions by are raked. The opponents' betting rules and hero's own call/lead/raise rules within each
  trial stay unraked.

### Tournaments (ICM)

The page's **Payouts** row under Rake (saved with the hand, kept across new hands and carried by share links)
takes the prizes still to be paid, first place first ("50 30 20", in any unit), and, once there are two or more,
the players left at other tables and their average stack. With two prizes or more every chip is priced by what it
is worth in prizes (gto/icm.py): Malmuth-Harville ICM, each player's chance of each place from the stacks, by
dynamic programming over who is left (exact, checked against every finishing order). One prize is winner-take-all,
the same as chips, so nothing changes. A big field is modelled by at most 12 players: the table's seats plus
stand-ins for the rest (equal groups of the other stacks sorted, each at its group's average), the payouts
compressed so each model place pays the average of the real places it covers; every player keeps their equity
share (27 players with 12 paid: the button's 1/27 of the prizes exactly). Answers carry `tournament`
(`equity_share`, what hero's stack is worth of what's left to pay, and `bubble_factors`: what losing a chip costs
over what winning one gains, against each opponent) and a note naming the model.

* Push/fold (hero's shove and the jam range hero faces) runs fictitious play on the prize values of each outcome
  (steal, called and win, called and lose), with each caller's own range and payoffs in turn: a 5bb big blind
  calls far wider than a 15bb small blind on a bubble, which one shared caller averaged away. Calling an all-in
  needs the equity that pays in prizes, said in the notes against the chip price: 4 left, 3 paid (50/30/20),
  the small blind with 15bb facing a 20bb button jam (CO 30, BB 5) folds AJo and 99, which call in chips; AA
  still calls.
* Other preflop decisions take a risk premium on top of the chart's price: the extra equity getting the
  effective stack in needs in prizes over chips (against the last raiser, or the players behind for an open),
  times min(1, ICM_COMMIT / (1 + the stack-to-pot ratio after the call)), so a call that commits the stack pays
  all of it and a cheap call little. Calls are re-priced by realized equity as rake is, value raises are kept
  to hands that clear break-even plus the premium, bluff raises scaled by the fold rate they need with the
  premium added, and opens (advisor._icm_open) shrink by fold need / (fold need + premium), the weakest by
  equity against what continues dropped first. 4 left, 3 paid, stacks CO 40 / BTN 25 / SB 8 / BB 25 (chips ->
  prizes): the big blind continues 60.5% against a button min-raise instead of 93%, the button 3bets a CO open
  7.0% instead of 12.2%, CO continues 29% against that 3bet instead of 47%, and the button opens 43.6% instead
  of 45% (CO 29.3% / 30%): published ICM work has the same shape (opens change little, calls and 3bets tighten
  a lot).
* Heads-up after the flop the solver plays prize equity: every terminal (fold, showdown, equity leaf) pays each
  player the chip-equivalent of the prizes they end with (solver.Config.icm, icm.HUPayoffs: the change in prize
  equity times total chips over the prize pool, so winner-take-all is exactly chips, checked by the solve's
  action EVs). The game isn't zero-sum then, so exploitability counts each best response's gain over its own
  payoff, as with rake. Bubble pressure makes the bigger bets and all-ins rarer (the river all-in of a bubble
  solve below the chip solve's). The precomputed flop libraries (solved in chips) are skipped, so a tournament
  flop solves live, and solves are cached apart from the chip game's.
* Multiway pots after the flop (3-way solves and the rules) stay in chips; the notes say so.

### Custom sizes and ranges

The page's **Advanced** disclosure in the setup (saved with the hand and carried by share links, like rake;
sizes, ranges and size optimization are sent from the flop on, so preflop advice is unchanged; locks from the
first decision, since they act preflop too). All are off unless given.

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
  `sizes_kept`. A kept overbet makes the second tree bigger, so it settles slower: use Accurate. Custom
  `"sizes"` for that street join the wide menu (kept only if the first solve uses them, like any other; the
  response lists them as `sizes_custom` and the page's advice says whether they were kept).

* `"locks": {"BB": {"fold": 15, "bluff": -50, "next_fold": 60}}`: nodelocking, your read of an opponent
  (the page: a profile per opponent, GTO, Calling station, Nit, Maniac, Overfolds to bets or Custom, whose
  numbers can be edited). Preflop their decisions follow it and your line is re-drawn against them (below);
  heads-up after the flop the advice is the best response to their locked strategy, as in exploit mode (`freq`
  0 or 1 with `gto_freq` alongside, ties within the solve's noise mixed), with `"locked": {"BB": {...}}` and a
  note; on top of GTO, or of the pool with `"opponents": "pool"`.
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
  earlier in the hand. Multiway pots after the flop (the 3-way solve, the betting rules) play without them, with a
  note; an unknown seat or key, your own seat, a non-number or a value out of bounds is an error.

  Preflop (`exploit.lock_preflop`, `Advisor._locked`) `fold`, `bluff`, `bet` and `raise` act on the locked
  player's chart strategy (the pool's with `"opponents": "pool"`) at each of their decisions; the street keys and
  `next_*` are for after the flop. `bet` scales their first raise (opens, iso-raises, the big blind's raise over
  limps) by 1 + bet/100, the weakest opens folding or the strongest folds opening; `raise` scales their raises
  facing a raise (3bets, 4bets) the same way, the weakest raises calling (the strongest calls, then folds, raising
  for more), so a station's -50 3bets half as often; `bluff` scales their bluff raises facing a raise (under 45%
  equity when called, tapering to value at 50%), the dropped ones folding and the extra coming from their strongest
  folds; `fold` folds this many points more of their range facing a raise, weakest calls first, then weakest raises
  (negative: their strongest folds call), in points of a range at least half of which continues (the big blind
  facing an open, the opener facing a 3bet) and that much less where less does (fold +15 on a cold seat continuing
  7% of hands folds 30% of what continues, not all of it). The hands that move are first those the chart already
  mixes (indifferent at equilibrium, they lose least by switching: the nit's JJ flats before its AK stops 3betting),
  then the rest by equity: against the last raiser's range between calling and folding, against what the raiser
  continues with facing a raise between raising and not (equity when called: all-in equity against the opener's
  whole range ranks TT over AKo). The locked seat's preflop decisions then shape every range replayed after them
  (`preflop_ranges`), so the flop starts from the nit's tighter 3bet range, and a locked jammer's 4bet jam is their
  raising range as locked. Hero's own decision is re-drawn against the advice without the lock (the chart re-priced
  for the spot, against the pool in exploit mode), at the bars that advice sets, so only what the lock changes moves:
  with a locked player in the pot, each hand's equity against the locked ranges instead of the unlocked ones moves
  it against the price, a continuing hand going once the loss eats what it had over the bar (the equity at the
  continuing share; a hand continuing below it, for implied odds, goes as soon as it loses equity), a folding one
  starting once the gain covers its shortfall (all-in equity: the measured realizations are against the charts'
  ranges, where KQs isn't dominated); with a locked player to answer a raise (in the pot or behind, so an open with
  a locked blind counts), the raises split into value and bluffs as the multiway re-pricing does: value stays while
  its equity when called against what the locked players continue with clears the unlocked bar, a hand under it that
  now clears it raises, a value raise that no longer does calls where the lock costs the raise more than the call
  (first in it is a steal), and bluffs scale with how often everyone folds, locked over unlocked (at most 1.5x, the
  extra first from the hands mixing raises with folds). `gto_freq` carries the chart's mix and `"locked"` the locks
  on players still in, with notes giving the locked range's size, the fold rates and the share hero continues or
  raises, locked and not. Measured (6-max 100bb): UTG's TT facing a nit small blind's 3bet to 10bb (the nit preset:
  its 3bet range QQ+, AK, most of JJ, some AQo and TT, 45 combos against the chart's 74) folds 34% where the chart
  always calls, and UTG continues with 21% of its opening range instead of 44% (AQs, KQs and 77-99 fold, JJ+ and AK
  stay); against a 12bb 3bet TT folds 100% (the chart still calls). Against a station on the button the big blind
  3bets A8o for value (the chart calls it; small pairs and suited non-broadways, which play for implied odds a
  station pays more, never turn into new value raises: by all-in equity alone 44 and 55 3bet 33-46%) and bluff-3bets T9s and JTs 62% instead of 90%, everyone folding
  37% to its 3bet against 53%; with a nit or an overfolder in the big blind the button opens 50% or 51% of hands
  instead of 45%, with a station 50% but its suited connectors 56% instead of always (fewer steals, more thin value
  against a 75% defence); a maniac's 3bet gets TT to 4bet. The equity when called lumps re-raises in with calls, as
  the multiway re-pricing does, so a maniac's extra 3bets read as calls when hero opens into it. Without locks every
  preflop answer is bit-identical (the preflop caches key on the locks).

Errors come back as `{"error": "..."}` (e.g. `sizes: at most 3 flop sizes ...`, `ranges: can't read BB's
range ...`). Heads-up responses also carry `villain_grid` (the villain's range by hand class: `cells` with
`w` the average combo weight), which the page draws under Details.

## How it works

| Layer | File | Method |
|---|---|---|
| Cards, ranges | `gto/cards.py`, `gto/ranges.py` | 1326-combo weight vectors, PIO-style range syntax |
| Hand strength | `gto/evaluator.py` | numpy-vectorised 7-card evaluator built on treys' lookup tables: all 21 five-card subsets of every combo on a board in one pass, about 1 ms per board |
| Preflop | `gto/preflop.py` | Hand-written 100bb charts: RFI, vs RFI, vs 3-bet, vs 4-bet, cold vs 3-bet, mixed frequencies. Where published solves exist the charts are matched to them, checked by `scripts/preflop_reference.py` (tests/preflop_refs.py: 26 spots, each played at the rake its solve used, 15 hand-level facts and 11 more played through the advice at the solve's rake): opens of 18.7 / 22.9 / 30.0 / 45.1 / 44.7% from UTG to SB (raked 2.5bb solves from pokalab, Poker Academy, GTO Ranges+ and gtopreflop.com agree on 17.6-18.7 / 22.3-23.3 / 29.1-30.6 / 43.3-46.5 / 43.5-45.7; were 16.7 / 19.4 / 27.2 / 45.6 / 50.2, with 22-44 half-opened from UTG and A2o from the BTN), the big blind's defence against the button (58.2/30.2/11.6 fold/call/3bet at gtopreflop.com's 5% capped at 4bb against its 58.7/29.8/11.5 and Poker Academy's 59.5/29.0/11.5; was 52.2/36.4/11.5; unraked it defends 55% of hands, was 64.6%; its 3bets polarised with offsuit wheel aces and suited kings, A4s-A2s calling), the small blind against CO and BTN opens (88/4/8 and 85/4/11 against 8.2% 3bets and 83.7/4.7/11.5; was 80/6/13 against both), UTG against a BTN 3bet (56/25/19 of its opens against 57.0/24.4/18.7, QQ mostly calling) and the small blind against the big blind's 3bet (64/20/16 against 63.6/21.4/15.0 at 10bb, now the default size; was 39/45/16, the chart written for 12bb). Average error over the spots 1.0 points (was 2.9, 10 of 22 outside tolerance). Published but thin: cold calls of a UTG open (gtopreflop.com: 2.1-4.6% of hands), one squeeze (Poker Academy) and one response to a 4bet (a GTO Gecko library export); not found: 5bets, heads-up. Seats for 2 to 9 players map onto the charts by how many players are left to act (9-max LJ plays the 6-max UTG chart, 7+ handed UTG plays a tighter EP chart, heads-up has its own button and big-blind charts). Replaying the preflop action through the charts gives every player's range going to the flop. The charts assume a standard-sized raise and nobody else in the pot; when the price or the players in differ (a min-3bet, callers ahead), calls are re-drawn by preflop equity against everyone in versus the pot odds, held to the equity bar the chart's own continue range sets in the standard line (with callers in, every call is re-drawn that way, not only added to the chart's: equity shared three ways is bunched, so the big blind over a button open and a small-blind call calls 15.0% of hands at 5% capped at 4bb, Poker Academy 16.1%, where adding to the chart's calls had 44.5%). With callers in, raises are re-drawn too: a value raise stays while its equity against what continues (each player's chart facing the raise) clears break-even by as much as the chart's weakest value raise heads-up does (callers' dead money makes thinner value raises pay), bluffs shrink by how much less often everyone folds against how often a bluff of that size needs them to (the callers' dead money lowers that bar), and the default raise size adds the callers' chips (a squeeze over two callers of a 2.5bb open is 15bb from the BB). Off-chart games and depths: limpers get iso-raise / over-limp / big-blind-raise charts (and an open-limp range); the iso-raises over one limper are a little tighter than the seat's open, at published widths (HJ 18%, CO 24%, BTN 36%, SB 19% of hands, the big blind 15%, against Preflop Wizard's and Grindlab's ~15-22%, 22-35%, 30-45%, 18-25% and ~15%; the old charts raised 10/13/16/10/8%; no solver frequencies for cash iso-raises were found: GTO Wizard's multiway preflop solver offers only iso-raise or fold facing a limp outside the blinds, and solved 6-max packs are mostly no-limp trees), about 0.6x that over two or more limpers (value-heavy, the dropped pairs and suited hands over-limping); antes play the open chart of a later seat and re-price calls; a straddle makes the straddler the big blind and the rest play by seats behind; stacks other than 100bb re-price calls with implied odds for small pairs and suited hands (from 70bb down, the small blind facing a CO or BTN open plays a 40bb chart instead, re-priced from 40bb: shallow, the hands under its 3bet value core flat, as in Poker Academy's 40bb solves where JTs, KJs, KTs, 88 and AQo call or jam; the 100bb chart thinned by implied odds left a 40bb flat of ~45 combos of 33-77 and A6s-A9s, now ~84 of suited broadways, 44-99, AQo/AJo/KQo and some suited aces, at 81/6/13 fold/call/3bet against the 100bb chart's 80/6/13); a raise that commits the stack (40% of it, or under 1.5 pots behind once called; 1.25 pots at the charts' own depth, where a 4bet to 2.3x leaves 1.45 and is called, as in 100bb solves) goes all-in, with the push/fold equilibrium as the range when shallower than the charts (and every open at 12bb or less); a jam over an open or limpers competes with flatting (equity against the raiser realized at the rate that makes the chart's weakest flats break even, plus implied odds) or the big blind's free check, not just with folding (the big blind rejammed a button open with 30% of hands at 25bb, now 22.5%). Preflop equity against several players is pooled per hand class over 400 boards weighted by how likely the players' ranges leave them (within 1-2 points of a dealt-out simulation five-way), and off-chart calls expect implied odds from the extra players by the square root of how many. Out of position facing a 3bet (a later seat's, or the big blind's against the small blind) the opener plays charts matched to published solver output: CO against a BTN 3bet to 7.5bb folds 63.7%, calls 17.2% and 4bets 19.1% of its opening range (Poker Academy, 100bb cash), openers respond in similar proportions from every seat (GTO Wizard), and the small blind against the big blind's 3bet to 10bb folds 63.6%, calls 21.4% and 4bets 15.0% (gtopreflop.com); the earlier charts called about twice that and 4bet half as often (CO 58/33/9). Also matched to published solver frequencies (100bb cash, 2.5bb opens): HJ and CO against a UTG open mostly 3bet or fold (90/2.5/7 fold/call/3bet of hands, against HJ 91.1/1.5/7.4 from Poker Academy and CO calling 1.4-3.2% and 3betting 8-9% from GTO Wizard; was 88/8/3), the button against a CO open (85/3/12, against 84.5/3.0/12.5 from Poker Academy; was 82/9/9), the big blind's 3bets against the button and the small blind (11.6% and 16.4% of hands, as Poker Academy and gtopreflop.com have them; were 8.5% and 11.7%), and the button facing the big blind's 3bet (54/34/12 fold/call/4bet of its opening range, against 54.6/33.9/11.5 from 888poker; was 59/35/6, a 4bet range of nearly all value) and the small blind's (60/29/11.5: gtopreflop.com's raked solve has the button 4betting as often against either blind, 10.4% and 10.3%, and calling 6 points less against the SB, with QQ, AKo, A9s and KTs mostly 4betting and AQs/KQs calling; was 58/36/6). The big blind's 3bets against UTG, HJ and CO are 4.8%, 6.2% and 8.0% of hands (gtopreflop.com 4.5/5.9/7.7, Poker Academy 4.5% against the LJ, pokalab 5.1/6.7, BeyondGTO 6/8/10; were 3.6/3.6/5.6, the HJ spot reusing the UTG chart), taken from calls so the defence totals stay; its squeezes follow (over CO + BTN 7.3%, was 4.9%; over UTG + a caller 4.5%, was 3.4%), about 0.9x the heads-up 3bet where GTO Wizard's NL500 cash solve at 40bb has 0.95x. Cold players facing an open and a 3bet fold 96-97% (GTO Wizard: over 90%, up to 97%) and the big blind's squeeze over a button open and a small-blind call is 9% of hands (Poker Academy 12.7%): both left. The button cold-calls a UTG open 4.5% of hands and 3bets 6.4% (gtopreflop.com's raked solve: 4.6%, 3betting more often than calling; the chart had the NL500 solves' 9.1% flats and 5.2% 3bets). No cash solves of the small blind's or button's squeezes were found (gtopreflop.com's database has no node facing two players in the pot; the tables that turn up are labelled approximations). The opener calls a blind's 3bet wider (in position), the heads-up button has its own vs-3bet chart, and a 5bet that leaves chips behind can be shoved over. Every player's replayed preflop range goes through the same adjustments as hero's advice (so a cold call of a cheap 3bet or a multiway big-blind defence is wide), a limper facing a raise, an iso-raiser facing a limp-reraise, the heads-up big blind facing a 4bet and every seat (blinds included) facing a 7+ handed early-position open have their own charts; facing a 3bet cold depends on how wide the 3bet is (over an early or a late open) and on position against the 3bettor (out of position, half the cold calls go: premiums 4bet, the rest fold), and facing an all-in that someone already called, equity counts the callers. A player who called a raise and faces the next one (a flat facing a squeeze, a cold call of a 3bet facing the 4bet, a call of a 4bet facing the jam) continues with at least 30% / 35% / 55% of the range they called with: the charts for those spots are written for someone who hasn't put chips in and continue mostly with QQ+/AK, which a flat almost never holds, so they left a flatter's range nearly empty (2% of BTN's flats called a BB squeeze). Those shares are floors, set against the tightest raises the charts make (the big blind's old 3.6% 3bet over UTG, UTG's 4bet, the 3bettor's jam); a wider, more bluff-heavy raise is defended more: each point of equity the caller's range gains against the actual raise over that reference adds 2.5 points (BTN's flat facing the big blind's published 8% squeeze over CO continues 44% and calls with 77; with a fixed 30% the wider squeeze had pushed 77 out). The extra calls are its strongest folds, ranked by how readily the button continues against a 3bet (TT-77 and suited broadways before 22-55), then equity against the raiser. Once another raise is in after the call (sandwiched) the charts' folding stands. The small blind flats an early open rarely (about 3% of hands) but not only middling hands: a share of JJ, TT, AQs, AJs and KQs flat as traps (published summaries have it sometimes calling JJ and TT there), where the old chart 3bet them all, so its flat-then-call of a big-blind squeeze was 13 combos of 77-99 and the squeezer's QQ won every A-K-x flop; now about 12 combos of 99-JJ, some 77-88 and those suited broadways. The 3bettor facing a 4bet uses a chart written against the opener's 4bet, which carries bluffs; against a cold 4bet or one from a player who flatted first (nearly all value) each hand is re-drawn by its equity against the actual 4bet range versus the opener's: value hands keep continuing while their equity clears the price of calling over the 80% of it a call realizes (one bar for every spot, so JJ calls 53% against a cold 4bet over UTG, where it has 33%, and always over the button, where it has 36%), and marginal and blocker hands shrink by how far they drop (AQs calls 5%). Bluff jams, and the value jams the chart mixes with calls, scale by how often that range folds to a jam compared with the opener's: against its value a jam and a call both get the stacks in, so a jam's edge is the equity it denies the bluffs, while a call can still fold bad flops. Against a cold 4bet over UTG QQ and AKo jam 30% (the chart 80%), and against a flat-then-4bet over an early open, which folds far less, they call instead; KK+ and AKs keep jamming. Against the opener's own 4bet the chart jams KK+, AKs, most of QQ and AKo and some JJ and A5s (fit to the one published response found: GTO Gecko's library has the big blind's 3bet range folding 46.9%, calling 29.3% and jamming 23.8% against the button's 24bb 4bet, here 45.6 / 34.0 / 20.4 with the same cards removed; the chart jammed 15.9%). A 4bettor facing a 5bet jam calls by equity against the jammer's replayed range versus the pot odds, like hero's advice (the fixed chart called QQ and AK 70% whoever jammed, where they have 36% against a standard jam and need 41%). |
| Postflop | `gto/solver.py` | Heads-up range-vs-range **discounted CFR** over a bet tree (2 bet sizes per street, raises to half pot and full pot after the call, all-in when the stack is short or a raise size would exceed it, max 2 raises). Vectors are compressed to combos in either range. A river showdown is O(N) per board: combos sorted by strength, cumulative reach sums, and a per-card correction for card removal. Equity leaves are one matvec with a precomputed antisymmetric matrix; natively a pass first collects every leaf's opponent reach and multiplies each leaf matrix stack by all of them in one matrix-matrix product (one product per leaf was memory-bound). Subtrees below the root street with the same shape (the turn trees after most flop lines differ only in pot size) are merged into one batch and traversed once, so a flop solve walks ~30 decision nodes a pass instead of ~90, with the same numbers (~1.6x natively). In the browser, whose numpy has no BLAS, the leaf matvecs are most of the time: they run through einsum there, on leaf matrices trimmed to each player's range, and per-card sums replace the small matrix products. DCFR (alpha 1.5; beta 0, negative regrets halved every iteration, for the first 100 iterations, then 0.5) with the average weighted to recent iterations, gamma 3 rising to 8 by the 50th iteration; other regret minimizers (CFR+, PCFR+, PDCFR+, hyperparameter schedules) are `Config.cfr` options, compared in `docs/cfr_research.md`. From the flop the river beyond `depth` keeps its betting, played on the turn's boards, and the turns are one card per rank class (see Accuracy vs speed). `scripts/bench_solver.py --curve` plots exploitability against time, natively or in Pyodide (`scripts/pyodide_run.mjs`). |
| N-player pots | `gto/multiway_solver.py` | Four or more players in (up to 6; 5 at the browser's Quick and Fast), or three with two all-in: the same CFR (alternating updates) with pairwise card removal: a hand's share of a pot against the players contesting it is the product of its heads-up shares against each times the others' masses, each pair of opponents' overlap corrected (their exact joint mass on the root's board) and every board scaled to stay constant-sum (`_term_pw`, the factor held to [0.5, 2]: on lines nobody takes, ratios of products of near-zero reaches overflowed the regrets). Pots are layered by the all-in players' stakes, each to the best hand among the live players whose chips reach it (`MWConfig.allin`, `inv0`, `dead`). Against the exact 3-way kernels its EV per deal is within 0.3-0.7% of the pot on average (2.2% the worst hand). Four or more betting play one bet a street (half the pot) and no raise but the player advised's (`root_raise`): a raise for everyone made the 4-way tree 1,053 nodes that didn't settle; without, 65 nodes and 0.02% of the pot in 150 iterations (5-way 161, 0.01%). Only the river is solved (and any street where you are the only one left with chips: dealt out, exact): with the later streets dealt out (equity leaves) a 3-way turn solve gave up 6.5% of the pot against the exact (bucketed) solve, the rules 1.5%. On 3-way rivers, scored by the exact solve's EVs: facing a bet and a call the rules gave up 16.5% of the pot on average (35% of hands over 2%), the pairwise solve 0.14% (3%); first to act 0.09% and 0.00%. Natively a 4-way river takes 2.2 s, 5-way 4.9 s (Fast); in the browser 4-way 2.5 s (Quick 1.8 s), 5-way 7.6 s. |
| Hand replay | `gto/advisor.py` | The whole hand is replayed for any number of seats: blinds, folds, dead money from players who folded, all-ins, acting order (heads-up the button acts first preflop and last postflop). Whenever exactly two players remain the spot is solved heads-up: from the start of the street if it began heads-up, or from the current node if a third player just folded. With three live players the 3-way solver takes the spot where it applies (below); four or more (or three with two all-in) the N-player solve takes their river, and any street where you are the only one left with chips (below, N-player pots); otherwise (4+ players on the flop or turn, a street it doesn't take) a Monte Carlo heuristic plays against every live range (see Limits), and says so. On the river, facing a bet in a heads-up or 3-way solve, a hand that beats at most 5% of the bettor's range (combos the board or your cards block left out) gets a note naming that share, the share it ties and the range's five weakest hand classes, so a fold to a tiny bet reads as what it is: "A8s beats 0% of CO's range and ties …" on 7d9hQcKsJh, where every Ax ties and nothing in the range is worse. |
| 3-way postflop | `gto/multiway_solver.py` | Three-player range-vs-range CFR (alternating updates, DCFR discounting: alpha 1.5, the average weighted by (t / (t + 1))^3, which at equal time is 15-30% closer to equilibrium than ^2 on 3-way rivers and turns; simultaneous updates were 5-10x further) from the node in front of hero: bets of 33% and 75% on the flop, 50% on the turn and river (one size inside a flop solve too), one raise, all-in at SPR 1.5 or less, everyone at the shortest stack behind. A pot with one of the three all-in (a shove two players called) is solved too, where the betting rules answered it: the all-in player never acts, and a showdown pays the main pot (everyone's chips up to the all-in's) to the best of the three hands and the side pot to the better of the two who bet, by the same exact kernels (`MWConfig.allin`, `main`); a fold gives up both. In the browser a flop or turn of it takes 1.4-1.6 s (100 iterations, 0.05% of the pot from equilibrium), so Quick solves it on every street. Over 600 fuzzed hands with random shoves, the 3-way solves went from 5 to 22 of 272 postflop spots (the solved share 14% to 20%), none failing; two or more all-in, or four players still betting, stay with the rules. Card removal between the three hands is exact on the river: a terminal's value per hand sums the two opponents' hands that share no card with it or with each other, by inclusion-exclusion over the 52 cards on running sums (O(52 n) a terminal, not O(n^3)). Turn and flop solves bet the later streets too, by hand buckets (16 a player natively, 12 in the browser) on one card per rank class, the river from a flop played on the turn's boards; showdowns are constant-sum tensors over bucket triples, and subtrees of one shape after different lines are batched. Research, design and the error accepted: `docs/multiway_research.md`. |
| Multi-street | `gto/advisor.py` | Earlier streets are solved with a small budget and each player's range is narrowed by the frequency of the action they actually took, mixed with a 15% uniform floor so an approximate solve never prunes a line to nothing, and followed only as far as the solve has settled (the rest by the betting rules' range for the action, see Accuracy vs speed). A range entering a later street with fewer than 8 live combos (preflop.MIN_LINE_COMBOS, counting the cards dealt since) is filled out with the rest of the range the player started the street with, weighted as preflop (1 - combos / 8): a 4bet pot's ~12-combo range checking back the flop left 2 for the turn, so the turn solve played a face-up hand. The current street is solved from the start of the street and the tree is walked to hero's node with exact frequencies. Solutions are cached per hand so the flop solve is reused on the turn. |
| Preflop subgame (experimental) | `gto/pf_solve.py` | Not used by the advice yet. Solves the 3bet pot heads-up as a game: the opener facing the 3bet folds, calls or 4bets; the 3bettor facing the 4bet folds, calls or jams; the opener facing the jam folds or calls. Hands are the 169 classes with class-level card removal; CFR+ with linear averaging converges to under 0.01% of the pot from equilibrium in 1000 iterations (~0.2s). An all-in pays equity; a call that sees a flop pays realized equity: a base by position and pot type (in position 1.0, out of position 0.8 in a 3bet pot and 0.9 in a 4bet pot), less for a hand behind (1.2 per unit of equity below 50% in a 3bet pot, 3.5 in a 4bet pot, where a dominated hand pays off or folds to a flop shove), a little more for one ahead in a 3bet pot (0.3), and a playability bonus for suited and connected hands; the two shares are normalised to add to 1, so the game is constant-sum. The constants are fitted to the charts' continue rates with `scripts/calibrate_pf_solve.py`: they landed within a few points of the old charts, but since the charts were matched to published solves (openers fold far more against a later seat's 3bet) they don't: openers call 37-55% of their range against the charts' 17-41% (calibration score 0.58, the sum of squared share differences over the six spots). The solve has 4bet bluffs (KJs, KQs, QJs) and jam bluffs (A4s, A3s, A5s), and QQ near indifference facing the jam. A strong hand facing a 4bet often calls rather than jams (so jam ranges are AA plus bluffs and KK sometimes folds to them), which comes from the realized-equity model of a call: it is the main limitation. `scripts/build_realization.py` measures it with the postflop solver instead, once, on the charts' ranges of six standard 3bet pots (openers out of position against a later seat's 3bet, in position against a blind's; 96 flops each, in the 3bet pot and the 4bet pot) into `gto/pf_realize.py`: per hand class against each of 9 opponent strength buckets (AA, KK, QQ-JJ, ...), how far its pot share after the call (EV over the pot, later bets included) runs above its all-in equity; the two sides' excesses combine so the shares add to 1, blended with the parametric model where measurements are thin. Per bucket matters (one number per class let KK's postflop edge count against AA, so KK folded to AA-heavy jams; now KK against AA is about -1 pot and AA against KK +1.8), and measuring on fixed ranges matters (re-measuring on the subgame's own ranges to a fixed point didn't settle). The flops are stratified by texture (suit pattern, pairing, high card, connectedness; 34 strata, at least two flops each) and weighted by the share of all flops each stands for: with random flops the two halves put the opener's calls against a 3bet up to 22 points apart, with weighted flops at most 10 (tests TestRealizationTable; the halves are in tests/data). The table is keyed by role as well as position (the caller, or the aggressor whose bet was called): position alone mixed an in-position 3bettor (realizing 1.15 of its equity in a 3bet pot) with an in-position caller (0.97), and with that mix the 3bettor trapped KK and the opener folded KK to the AA-heavy jams; by role KK calls them everywhere. `scripts/fit_realization.py` fits a smooth model to it (`gto/pf_realize_model.py`): per pot type, role and position, eight coefficients on the hand's equity against the bucket (x = e - 0.5: x, x·abs(x)) and its shape (pair, suited, connected, pair*x, suited*x). It stays off (pf_solve.MODE = "param"; "table" and "model" switch to the measurements; tests TestRealizationModel) because the measured caller-aggressor gap looks too large: callers realize 0.79 (out of position) and 0.97 (in position) in 3bet pots against the aggressor's 1.02 and 1.15, and about 0.6 in 4bet pots against 1.25, so in-position openers called 3bets less than the charts (BTN vs BB 28% against 34%, CO vs BB 25% against 41%) and out-of-position ones more (CO vs BTN 47% against 18%). Neither the postflop solver's depth nor its bet sizes explain the gap: playing the river instead of scoring it by equity moves these about a point, and four bet sizes a street (25-125% on the flop, 33-150% later) instead of two move them under one (40 stratified flops, BTN vs BB and CO vs BTN 3bet pots and the CO vs BTN 4bet pot). Nor does the 3bettor's range: widening BB's 3bet against a BTN open from the chart's 8.5% of hands to the published 11.5% leaves the BTN caller at 0.99 (40 flops). In aggregate an in-position caller realizes about its equity; the call rate falls short of published solves because marginal hands (KQo, A5s) measure a few points below their equity while strong hands run above, and checking that needs per-hand EVs from a commercial solver. So either the gap is real for the charts' ranges (a capped calling range against a strong, uncapped one: in the CO vs BTN 4bet pot the caller holds JJ, TT, AQs against KK+, AK) and the charts call too much, or the subgame's calls lack something the charts encode; settling it needs outside solver data. Public equity-realization figures cover only some of the roles. The GTO Wizard aggregate reports quoted by Poker Academy give 1.14-1.17 for the in-position raiser of a single-raised pot and 0.78-0.87 for the out-of-position caller (BTN vs BB, LJ vs BB, limped pots, 25-184 flops). A GTO Wizard blog example (BTN 3bets CO, 552r, 100bb) splits equity about evenly with EVs of 10.78bb against 7.72bb, which is 1.17 against 0.83. The measured aggressor in position (1.15) and caller out of position (0.80) match these. Nothing published covers an in-position caller of a 3bet. Rules of thumb (deepfold: 1.08-1.15 for late-position flats; Tournament Poker Edge: 1.25-1.5 for an in-position flat of a 3bet) put it above the measured 0.97, and per-class figures put offsuit broadways and offsuit aces below their equity and pairs and suited connectors above it, as measured (in-position caller: Axo 0.80, offsuit broadways 0.86, suited connectors 1.07, 22-66 1.11). The model mode now adds 0.18 to the in-position caller's realization in 3bet pots (pf_solve.MODEL_SHIFT, 0.97 to 1.15), which brings the in-position openers' calls to the charts (BTN vs BB 41% against 34%, CO vs BB 31% against 41%, BTN vs SB 38% against 36%; calibration score 0.30 against the parametric model's 0.58 and the unshifted model's 0.32). The out-of-position openers still call 2-2.5 times as often as published solves (CO vs BTN 47% against 18%, UTG vs CO 39% against 18%), even though the measured 0.80 matches public figures for that role. Fitting the charts would take it to about 0.64, against the 0.78-0.87 published for out-of-position callers, so the shift isn't applied there. The 3bettor also jams too often over a 4bet (28-39% against the charts' 15-39%). So the default stays "param", and the solver stays blocked: what's missing is per-hand EVs for an out-of-position caller of a 3bet. |
| Preflop equilibrium (derived) | `gto/pf_game.py`, `gto/pf_eqr.py` | The full 6-max 100bb preflop game (2.5bb opens, SB 3bb; 3bets 3x in position, 12bb from a blind, 10bb BB over SB, one more open per cold caller; 4bets 2.3x; 5bet all-in; at most three players see a single-raised flop, two a raised one: 8,847 nodes) over the 169 classes, solved by discounted CFR level by level in numpy (1,500 iterations, ~0.002 bb/hand from equilibrium, ~3 minutes). A flop pays each player its realized pot share, measured with the postflop solver rather than assumed (`scripts/pf_equilibrium.py`): 17 heads-up flop spots (single-raised, 3bet and 4bet pots, aggressor in and out of position), 68-102 texture-stratified flops each, every class of the pre-action range added at weight 1e-3 so hands outside the range get their counterfactual value; per pot type, aggressor position and role a model of the excess share over all-in equity against 9 opponent strength buckets (hand features plus shrunk class residuals, `gto/pf_eqr.py`), iterated with the solve to a fixed point. It measures BTN vs BB single-raised pots at 1.13 / 0.85 (published solver reports 1.14-1.18 / 0.79-0.87) and a BTN-vs-CO 3bet pot at 1.13 / 0.87 (published example 1.17 / 0.83); three-way pots use the product of heads-up equities corrected per class (`scripts/build_three_way.py`) and a realization factor by seat and the opener's place, measured with the 3-player solver (`scripts/pf_three_way.py`). Against the published solves (tests/preflop_refs.py) the solve is 2.5-3.0 points off on average with 12-13 of 22 spots and 8-10 of 14 hand facts outside (the charts: 1.0, 0, 0): right on the big blind's defence against the button (58.1/29.3/12.6 against 58.7/29.8/11.5) and openers facing a 3bet (CO vs BTN 63.5/10.3/26.2 against 64.0/16.9/19.2), but its opens are tight (BTN 37.6%) and prefer offsuit broadways to small pairs and suited connectors. So the charts stay; `scripts/pf_compare.py` lists what each chart costs inside the solved game and `scripts/pf_subgames.py` solves the reference spots entered with the charts' ranges (docs/preflop_research.md section 6 has the details and why no spot without a reference was replaced). A second pass (section 7) tested each suspected structural cause by experiment: rake (moves calls, not opens), three-way flops (nothing), SB limps (`sb_limp`: limps 8.6%), published 4bet sizes (`Sizes.four_x_ip/oop`), the flop model's fidelity (`scripts/pf_fidelity.py`: with the river dealt and played, suited connectors and small pairs realize 0.1-0.3 more and the ranges within 0.03), per-spot fits (`--by-spot`) and nine realization shifts fitted to the references; the best without fitting is 2.42 points, 10 spots and 6 facts out, fitted 2.17, and a 0.03 change in one pot type's realization moves it by up to 0.5. So no charts were generated from it. The advice uses the realization model at run time in one place: re-priced calls (rake, another size) are re-ranked by realized equity (Advisor._priced, below), with the fast flop model's river correction (connectors and gappers +0.15, 22-66 +0.10, other suited hands +0.10, other offsuit hands -0.05 in single-raised pots, half that in 3bet pots; advisor._river_fix). |
| Exploit mode | `gto/exploit.py` | The pool is the equilibrium plus common leaks, frozen (it doesn't adapt to you): facing a first bet it folds too often at every size on every street, most of all to small bets: on the river +20% of the range at a third pot, +13% at three quarters, +7.5% at an overbet, from mass data analysis of a real pool (GTOkiller, GG Poker NL100-200, BTN vs BB third barrels: 54.7% folds to half pot, 57.8% to pot, 70.6% to 175%, against ~33/50/64% at equilibrium); on the flop +8.5% at a third pot, tapering to +5% at an overbet, from the same source's c-bet analysis (47% folds on K74r against ~40% at equilibrium, ~45% on 652ss against ~35%, barely changing between a third and half pot; bigger flop bets only corroborated: whole-sample aggregates of CoinPoker NL10-NL100 and Ignition NL10, 2025, every pot type pooled, have folds rising from 39% against bets up to half pot to 54% against half to pot and 64% against overbets, about as fast as equilibrium's folds rise with size, so the taper stays), where the old guess had the pool calling small flop bets too much; on the turn +10.5% at a third pot, +7% at three quarters, +6.5% at an overbet, from Upswing's LeakBuster data of 500nl+ regulars' 2025 games (over-folding a delayed turn c-bet by 11/6/6 points at 30%/66-75%/120-150% pot and a turn probe by 10/8/7 at block/medium/overbet sizes; it replaces +14/+10/+6 interpolated between flop and river; one GG spot on Maxploit, a 150% turn overbet folded 72% against 58%, suggests lower stakes over-fold turn overbets more; the same CoinPoker and Ignition aggregates give the same shape at NL10-NL100: 41/52/63% folds by size band); the extra folds are its weakest calls, at most half of its equilibrium calls and never its top 10% of hands on the board (as a flat share of the range, a turn jam that equilibrium calls with a few percent lost every call, queens full included), the extra calls its strongest folds; it raises less (weak hands keep 30% of their raises, the rest call; a judgment call that one measured spot since bears out: GTOkiller has GG NL10-50 regulars raising a third-pot SB c-bet on K84r 9% of the time against GTO's 18%, and here the pool keeps 0.46-0.48 of equilibrium's raises in that spot; other figures since agree, none precisely: the same regulars re-raise a raise 16% against 25%, a GG river check-raise sits 10 points under GTO and is 95% two pair or better), and it bluffs less (bets from its weakest half are cut to 80/70/57% on flop/turn/river; the river from Upswing's same data: 500nl+ regulars' heads-up river bets in small pots are 12-25% low pairs or air by size against GTO's 18-37%, the bluffs kept at 0.55-0.62 of equilibrium's, 0.57 on average, where the old guess was 0.4; river showdowns are a fair sample, since a river call doesn't depend on the bettor's hand; the flop and turn are judgment calls, the turn between flop and river, since the only turn figure found comes from shown hands, which under-show turn bluffs). Strength is absolute: the hand's percentile on the board, averaged over runouts before the river. Preflop it is fitted to population HUD stats (Hand2Note mass data of GGPoker 6-max, all players, NL50 to NL500, published by Bluffaces: VPIP 28.75, PFR 18.7, 3bet 8.9%, fold to 3bet 48.25%, 4bet 12.75%, against the charts' 24.0, 17.7, 7.8%, 59.7% and 14.1% over every line weighted by how often it happens; scripts/calibrate_pool.py): facing an open it 3bets 1.23x the chart's share (the extra from its strongest calls); its extra hands are spread evenly over every cheap way in, since no data found splits this pool's VPIP by position or line: first in it enters, and facing one raise it flats, with 10.8% of what the chart folds (strongest first; fitted to VPIP; first in, the strongest 65% of those hands open-raise and the rest open-limp, fitted to PFR); first in from the small blind it also limps 7.6% of its range out of the chart's weakest opens, for 9.7% limps and 41.0% opens (chart 44.7%), since whole-sample aggregates of two other pools by seat (CoinPoker NL10-NL100 2025, 3.18M hands: small blind 7.1-9.7% limps, other seats 1.5-2.8%; Ignition NL10 2025: 15.0% against 8.6-9.9%) have the small blind limping 3-6x as often as any other seat and opening 7-12 points under the chart (their early seats open 1-3 points wider than the charts and their buttons 1-8 narrower, where this pool opens every other seat wider; left, since its PFR is above both of theirs), the big blind excepted (the chart's unraked defence is already wider than raked solves, and the one, unsourced, population figure has big blinds folding more to a button open than the chart, not less); before, all of it went to cold flats (non-BB calls facing one raise: 20.3% of those hands, now 13.8% with limpers calling an iso-raise included, against the chart's 5.5%); as the opener facing a 3bet it calls with 17.5% more of its range (its strongest folds; its wider opens are weak and fold to 3bets); its 4bets are the chart's. The pool's tree then gives VPIP 28.8, PFR 18.8, 3bet 8.9%, fold to 3bet 48.0%, 4bet 12.2%, open-limping 3.2% of its chances, between the other pools' 2.1-3.3% and 10.1% (with every extra first-in hand limped and the old, tight iso charts it gave PFR 16.4 and 4bet 14.8%). The old guess (3bets cut to 35%, 4bets only QQ+/AK) had the pool 3betting 4.1% and folding 57.5% to 3bets. Each street is still solved to equilibrium; the pool's actions narrow its range through the leaky strategy, and hero's EVs come from a best response to it over the rest of the tree. Actions whose best-response EVs are within the solve's noise of the best (its distance from equilibrium as a share of the pot, at least 0.5%) count as tied, and a tied hand plays the equilibrium mix among them: a short browser solve otherwise picked one at 100% on noise (AQo checking at 6.45 over bets at 6.40 in a solve 3.1% of the pot from settled). Preflop, hero facing a raise goes through the off-chart re-pricing (see Preflop) with the pool's ranges and responses: calls by equity against what the pool actually raises with, value raises by equity against what the pool continues with, bluff raises scaled (up to 1.5x) by how often the pool folds compared with the chart. Facing a jam over a raise, the jam range is the pool's. Opens follow the chart (the pool 3bets more, flats more and folds less to 3bets); a pool limper's range is its limps, hero's replayed range stays the chart's, and multiway postflop spots are played at equilibrium (the 3-way solve or the betting rules) on the pool's preflop ranges, with a note. |
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
  real answer is 6%). The browser's Fast mode used 8 such rivers; against every river its
  per-combo EVs were 4-10% of the pot off. Every config now plays the river by
  `river_classes`: one card per class (a rank's cards, a suit with two or more cards on the
  turn apart: 13 rivers on a rainbow turn, 22-24 with a flush suit), the class's subtree
  standing for each of its cards through the swap of the played card and that card. No
  flush is possible in the swapped suits, so the swap keeps every hand's strength: a
  class's reach is the mean of its cards' swapped reaches and a combo's utility sums its
  swapped image's over every river, so it is exact for suit-symmetric ranges and zero-sum
  (the combo space takes the swapped images, with zero reach). An iteration costs 0.46-0.48x
  every river's with one size on a rainbow turn, 0.63 on a two-tone one, 0.3-0.4x with
  two or three sizes; at 100 iterations its strategy, lifted into the every-river tree
  (`solver.lift_classes`), is 0.35-0.55% of the pot from equilibrium there where every
  river's own is 0.28-0.38%, frequencies within a point.
* **Turn and river against reference solves** (`scripts/turn_river_reference.py`,
  tests/test_turn_river.py): six turn spots (BTN vs BB after a third-pot c-bet is called, on a
  brick, a flush-completing, a board-pairing and an overcard turn; the flop checked through; a
  CO vs BTN 3bet pot) and four river spots (bet-bet polar, bet then checked, a flush completing
  on the river, checked to the river), the ranges narrowed by long flop and turn solves, each
  solved to convergence with every river card and a rich menu (33/75/125% turn and river bets;
  33/50/75/125/200/250% on a river root; 500 turn and 3000 river iterations, 0.03-0.13% of the
  pot from equilibrium). The size menu was the biggest error: the rich trees overbet 29% of the
  range on K72r-3 and 36% on Q83r-A (BTN's turn barrel after a called c-bet), lead 39% at a third
  pot on K72-7 (the one 75% size gave 7%), and on a polar river bet 79% with half the range at
  200-250% pot. The configs at the iterations their budgets give (each config's cost against
  one size with every river on the same spot, that tree's idle 16.8 it/s natively and Pyodide
  at 1.45x native on the turn), range-weighted mean per-combo EV error against the rich reference
  in % of the pot, and the mean frequency error in points (each player's bets, overbets and
  fold / call / raise against bets, on the line's path):

  | config | turn EV err, before -> after | turn freq err | river root EV err | tree now |
  |---|---|---|---|---|
  | browser Fast (8 s) | 7.1 (3.8-9.5) -> 1.0 (0.5-1.5) | 7.9 -> 3.1 | 3.1 -> 0.3 | turn 33/125%, river 75%, classes |
  | browser Accurate (60 s) | 2.3 (0.6-3.8) -> 0.18 (0.08-0.33) | 5.7 -> 0.7 | 3.1 -> 0.3 | 33/75/125%, classes |
  | native Fast (2.5 s; turn 5 s with the flop free) | 1.9 (1.4-2.9) -> 1.0 (0.5-1.8) | 5.8 -> 3.4 | 3.1 -> 0.3 | turn 33/125%, river 75%, classes |
  | native Accurate (30 s) | 1.5 (0.7-2.9) -> 0.23 (0.11-0.32) | 5.2 -> 1.0 | 1.9 -> 0.3 | 33/75/125%, classes |

  Every river with one size converged was itself 0.6-3.8% of the pot from the rich tree, so
  the old Accurate configs' error was the menu, not convergence; Accurate now plays three sizes
  at about its old cost (river classes pay for the third size). Native Fast was held by its 2.5 s
  (75/125%: 35-62 iterations, 1.65 on average, 0.6-2.6, its missing small lead 2.6 on K72-7). Its turn
  now also gets the 2.5 s the narrowing flop didn't spend (`Advisor.spare_prior`): the flop comes from
  the library or from the hand's own flop advice, so a turn request takes 5 s, as a cold one always
  did, and that buys the browser Fast's 33/125% menu: 80-150 iterations, 1.0 (0.5-1.8; 75/125% at
  5 s: 1.5, 33/75/125%: 1.05). The time was the limit, not a warm start: the library stores only
  the flop's decisions and the Fast flop's turn is played per bucket with one size. Cold (the flop
  solved in the same request) the turn keeps 2.5 s: 1.9 (0.7-4.3). In a 3bet pot (four CO vs BTN turns,
  `t-3b-*`; the turn 3.2x the pot deep) the references barrel at three quarters pot where single-raised ones overbet,
  so Fast's turn there bets 33/75% (`Advisor.spr_sizes`, at most 4.5x the pot behind): per-combo EVs 1.95% of the
  pot off on average -> 0.81 natively at 5 s (0.55-4.8 -> 0.42-1.5; browser 1.96 -> 0.80), frequencies 6.8 points ->
  2.0 (33/75/125%: 1.07 and 2.4, fewer iterations; cold at 2.5 s 1.82 -> 1.38). In single-raised pots 33/75% was
  1.6-3.7 off (it has no overbet) and 33/75/125% 0.75-2.65 against 33/125%'s 0.54-1.42. A river root (Advisor.root_sizes,
  every config) bets 33/75/125/250% pot: per-combo EVs within 0.11-0.60% of the pot of the six-size
  menu where 75% alone was 1.2-7.2%; a river solve is milliseconds an iteration, ~1000 in 2.5 s.
  Published figures (TURN_REFERENCES / RIVER_REFERENCES; GTO Wizard's turn aggregates are behind an
  account and its blog gives single spots, so the bands are wide): no BB lead on a brick turn and
  some on a low pairing one (ours: K72-7 39% at a third pot, 2-3% over every K72 turn, nearly all on
  the pairing 7 and 2; GTO Wizard's donk article: BTN vs BB J53-3 ~40% at 40bb, its "~11% on K75r" is
  UTG vs BB averaged over every turn; the lead is EV-flat, 23-41% across near-identical solves with BB's
  EV 57.8% of the pot each time), BTN's barrel 41% on K72r-3 with 29% overbets (ThinkGTO's
  JJ4: 53.5% on a brick, overbets dominant on an A), BB probing 37% and BTN delaying 38% after a
  checked flop, the 3bettor barrelling 62% (GTO Wizard 47-63%), and on the polar river 79% bets,
  50% overbets and BB folding 60%. River theory, on converged real-range solves (TestRiverTheory):
  a polar bettor's bluffs (bets losing to the calling range) are within 6.5 points of b/(1+2b) and the
  capped defender within 3 of MDF 1/(1+b) at half, 125% and twice pot (25.4 / 30.9 / 34.5% bluffs
  against 25.3 / 35.8 / 40.0; 65.0 / 43.5 / 31.9% defence against 66.2 / 44.3 / 33.3); checked to the
  river the defender folds far past MDF (defends 37 / 30 / 25% against 65 / 44 / 33: the bettor's
  bluffs give up showdown value, as GTO Wizard's MDF article finds for out-of-position defenders);
  and every hand mixing call and fold, or bet and check with a losing hand, is indifferent to 0.5%
  of the pot.
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
  published solves (a 25%/80% flop tree like ThinkGTO's), measured with the preflop charts of the time: QQ6 96%
  (published 98.8%, was 64%), K72 97% (BeyondGTO 98%, was 64%; 76% with today's charts, whose BB defends like a
  raked solve, see Known limits), A74 74% (67%, was 70%), 883 65% (68%, was 52%); BB vs a 25% bet on
  A74 folds 36 / calls 45 / raises 18 (34.6 / 50.0 / 15.4). An iteration costs ~7x the old tree's
  natively (~15 it/s on a rainbow flop, half that on a two-tone or monotone one) and ~15x in
  Pyodide, and a solve needs ~100: 30 iterations were 12 points of c-bet and 15 of BB's check from
  the reference, 60 were 8 and 8, 100 were 5 and 4. So the server's Accurate settles. The browser's
  Accurate, measured in Pyodide (`scripts/bench_browser.mjs`), got 54 iterations in 60 s on K72 and
  ~25-30 on a two-tone flop, which still over 10% of the pot from equilibrium took a second budget (132 s,
  a 590 MB heap); against every turn played (12 flops, `bench_solver.py --flop --flop-iters`) 55 iterations
  are 7.5 points of c-bet off on average (worst 20), 12.7 on the size split, 8.5 on BB's check and 6.8 on
  its response to the small bet, 25 iterations 14.4 / 18.4 / 18.4 / 14.8. It now plays Fast's bucketed
  flop model (below) with 64 buckets over every river (`river_buckets=64, bucket_river_step=1`): 520-870
  iterations in 60 s (0.03-0.05% of the pot from equilibrium in its game, a 115-165 MB heap), and at 400
  iterations 1.8 points of c-bet off (worst 4.8), 2.3 on the split, 1.3 on the check, 1.4 on the response
  (200 iterations: 2.2 / 3.0 / 1.9 / 1.5; 32 buckets over every third river at 400: 2.5 / 4.1 / 1.6 / 1.8;
  96 over every river 1.9 / 1.7 / 0.9 / 1.4 but 270-470 iterations). Its narrowing flops (12 s) play the
  same model, so a hand's flop advice narrows its turn without another solve.
  * Fast (2.5 s natively, 8 s in the browser) plays the same flop model with the turn and river by
    hand buckets (`river_buckets=32, turn_buckets=True`): each player's range on each turn card is split
    into 32 equal shares ordered by equity against the other's flop range, and the turn and river below
    a flop solve are played per bucket against a 32 x 32 bucket-pair equity matrix (the turn's every-river
    equity averaged over the pairs with the flop ranges' weights, card removal as each pair's compatible
    share; `bucket_river_step=3` averages every third river, no worse). The flop stays per combo. The two
    players' tables are each other's negated transpose, so the game stays zero-sum. Per combo the river
    beyond depth was ~95% of the arithmetic (N x N leaf products); by buckets an iteration costs what the
    old Fast tree's did natively (12-18 ms on a single-raised pot) and 2.4x less in Pyodide (30-42 ms
    against 74), so Fast gets 140-210 iterations natively and 130-420 in the browser (measured: 217-421 on
    K72 by machine load, 133-142 on 986 two-tone), where it had ~200 and
    ~100 (setup, before the budget: 0.6-1.0 s natively, 1.1-1.6 s in Pyodide, against 0.6 and 0.7; the turn
    leaves' showdown sums now take one comparison and a byte add per board, and every board's hand
    strengths one batched evaluator call). Against every turn played (12 flops above), at the iterations each gets: c-bet 3.6 points off
    on average (native Fast was 12.2, the browser's 14.5), size split 7.6 (32.6, 35.5), BB's check 2.1
    (6.1, 13.0), its response to the small bet 1.6 (5.0, 6.2). Settled (400 iterations), 32 buckets are 3.0
    off in c-bet where per combo is ~1.4; QQ6 stays near 80% (88%), A74 near 61% (51%). Exploitability in
    the bucket game is 0.2-0.4% of the pot at 200 iterations (the old trees, not zero-sum with their
    sampled turns, read -0.35 to 1.15%). Exploit mode's pool lock and your locks read a bucket's percentile as its place in
    the range.
  * The river by river strength (`river_strength=32`, every config's flop solve in single-raised pots): played
    on the turn's boards, a draw was a middling hand there (its turn equity), so it could neither value bet the
    rivers that hit it nor bluff the ones that miss, and the flop c-bet ran low. Now on each river card each
    player's range splits into 32 equal shares by equity there against the other's flop range, and a hand (a
    combo, or a turn bucket in Fast) enters the river in each share with the fraction of river cards that put it
    there: a flush draw is in the top two shares on its 9 of 46 rivers and near the bottom on the rest. The
    players see their share, not the card; the showdowns are each pair of shares' mean over the rivers, per pair
    of hands on the same card (zero-sum, `Ctx.river_table`). Measured against the river with its card (each
    model on the same 3 turns of 5 BTN vs BB flops, QQ6, K72, A74, 986 and 765; the reference per combo with every
    river under each turn, 200 iterations, 4-6 minutes a flop; the models 300; `bench_solver.py --flop --flop-iters
    300 --configs river-cards-3t,accurate-blind-3t,accurate-3t,fast-blind-3t,fast-3t`): per combo the c-bet was
    2-11 points under on every flop (6.0 on average; QQ6 60.2% against 67.4%) and is now 1-6 over (3.2; QQ6
    69.4%), the size split 5.0 -> 3.1 off, BB's check 3.7 -> 3.0, the range-weighted error in per-combo node EV
    2.7 -> 3.0% of the pot at the root (BB) and 3.9 -> 3.0 for BTN after a check; Fast's buckets 6.9 -> 4.2 in
    c-bet (QQ6 62.5 -> 70.4), split 6.9 -> 5.0, check 1.2 -> 4.9 (986 and 765, where BB leads more), EV 5.4 ->
    5.3 and 6.1 -> 5.7. 64 shares are no closer than 32. In 3bet pots (K72, A83, 765) it had the 3-bettor bet
    three quarters pot with 15-31% of its range where the reference bets 0-9% and the turn's boards 0%, so a flop
    whose stack is under 8 times the pot keeps the river on the turn's boards (`river_strength_spr`). With the
    configs' own trees (one turn per rank class) the c-bet goes up 2-5.5 points: native Fast at 200 iterations
    QQ6 76.1 -> 78.7%, K72 66.5 -> 70.8, A74 50.4 -> 52.6, 986 41.0 -> 45.0, 765 38.6 -> 43.0; native Accurate
    at 300 79.3 -> 82.4, 66.5 -> 72.0, 46.1 -> 49.1, 41.9 -> 43.2, 39.4 -> 42.1. Cost (a machine shared with
    other jobs, CPU time): per combo the river's showdowns are 32 x 32 instead of N x N, so a native Accurate
    iteration is 3x cheaper (K72 62 -> 20 ms, 986 100 -> 33 ms; its 30 s run ~3x the iterations), its tables up to
    0.5 s; native Fast's tables add 0.15-0.3 s to its 2.5 s (K72 ~175 -> 155 iterations, 986 ~175 -> 145).
    In the browser (`bench_browser.mjs`, the library off so the flop is solved live, medians of 3-7 runs on a
    machine shared with other jobs) Fast gets ~17% fewer iterations in its 8 s: K72 350 -> 291 (0.09-0.12% of
    the pot from equilibrium either way), 986 two-tone 225 -> 185 (0.15 -> 0.2%); Accurate reaches its 1000
    iteration cap in 27 s on K72 and 45 s on 986 where it ran 664 and 854 in 60 s (0.04 / 0.02% -> 0.01%).
    16 shares for Fast would run more iterations than before (K72 ~445, 986 ~275) but were further from the
    river with its card (c-bet 5.9 off, size split 9.0). The flop library is still Accurate's river on
    the turn's boards (an entry stands in with a note).
  Tests TestFlopStrategy hold the published table, for both models; TestRiverStrength the tables and the game.
  A flop all-in pays its equity over every turn card with 4 evenly spread rivers each
  (within 4% of exact; 4 sampled turns had flush draws up to 24% off).
* `Solution.exploitability()` reports how far a solve is from equilibrium in
  % of the pot (it plays a best response for each side against the other).
* **Against postflop-solver on the Advisor's own spots** (`scripts/bench_postflop_solver.py`, records in
  `reports/postflop_solver/`). [postflop-solver](https://github.com/b-inary/postflop-solver) is b-inary's open-source
  Rust solver (DCFR, every turn and river card dealt, multithreaded; the engine behind WASM Postflop and Desktop
  Postflop). `scripts/pfbench` is a small harness around its crate (AGPL-3.0, built with cargo as its Cargo.toml says,
  not part of the site) that takes a spot as JSON: our betting tree line by line, the same ranges, pot and stack.
  Rivers and turns are the ones the Advisor solves in a hand: BTN vs BB, CO vs BB and BB's 3bet against BTN, three
  random library flops each, the lines the library's strategy plays often (check-check, the commonest c-bet called),
  a random turn (and river, the turn checked through), hero out of position. The Advisor answers at Fast and at
  Accurate and its solve of the street is taken from its cache. Theirs then solves the same game exactly (every
  river card where our turn plays river classes), solves it again on one core to the exploitability ours reached,
  and once per player solves it with every decision of that player on the street locked to ours, the rest (the river
  after a turn) at equilibrium: the EV that player gives up by our strategy, in % of the pot. Medians (range), wall
  seconds on a 10-core laptop (4 performance cores):

  | street, speed | n | ours: iterations, s | ours: % of the pot from equilibrium | EV given up, OOP / IP | root mix diff, points | theirs, all cores: s to 0.005% (river), 0.05% (turn) | theirs, 1 core: s to ours |
  |---|---|---|---|---|---|---|---|
  | river, Fast | 16 | 1000, 2.0 | 0.014 (0.005-0.024) | 0.008 / 0.008 (max 0.018) | 2.5 (0.2-5.7) | 0.30 (0.06-1.27) | 0.15 (0.05-0.60) |
  | river, Accurate | 16 | 1000, 2.1 | 0.012 (0.005-0.024) | 0.006 / 0.007 (max 0.023) | 2.2 (0.6-4.1) | 0.28 (0.05-0.61) | 0.15 (0.04-0.31) |
  | turn, Fast, single-raised | 12 | 96 (82-110), 4.9 | 0.46 (0.32-0.88) | 0.093 / 0.075 (max 0.34) | 4.3 (0.0-23.5) | 1.6 (0.9-2.1) | 1.9 (1.3-2.4) |
  | turn, Accurate, single-raised | 12 | 182 (143-213), 29.7 | 0.40 (0.18-0.60) | 0.044 / 0.044 (max 0.13) | 2.4 (0.0-18.8) | 9.4 (5.2-21.4) | 13.8 (8.7-21.9) |
  | turn, Fast, 3bet pot | 3 | 327 (322-355), 5.0 | 0.038 (0.034-0.045) | under 0.01 | 1.5 (1.1-2.5) | 0.4 (0.3-0.5) | 1.7 (1.5-2.3) |
  | turn, Accurate, 3bet pot | 3 | 802 (791-867), 29.9 | 0.014 (0.010-0.015) | under 0.01 | 2.2 (1.7-2.9) | 1.3 (1.1-1.4) | 6.6 (5.4-6.7) |

  Rivers are exact: their best responses to our whole river strategy measure it at our own number to 4 decimals (the
  two codes' exploitability agree), each player gives up under 0.025% of the pot, the root's range frequencies are
  within 0.7 points. On the turn our strategy gives up 0.04-0.09% of the pot (0.34 at worst, a Fast turn checked
  through on a wide range); its exploitability in our game overstates what it costs in play by 5-10x. The two turns
  where beta 0 swung late (Regret minimizer, below) answered at 0.99 and 1.48% of the pot and gave up 0.18-0.34; with
  the beta step 0.23 and 0.23, giving up 0.01-0.05. Their code is faster: on one core it needs as many river
  iterations as ours (~900) in 0.15 s where ours takes 2 s (ours is NumPy, bound by per-call overhead on small
  arrays), and reaches our turn accuracy in ~2.5x less time on single-raised turns (as many iterations, each ~2.5x
  cheaper) and 3-4.5x less on 3bet pots (half the iterations at Accurate's 0.014%); on all cores it solves a turn to
  0.05% in 0.3-21 s.

  Flops: a library entry's stored decisions (the root, IP after a check, OOP facing a c-bet after checking) locked
  into their full game of the library's tree (every turn and river card, where the entry plays turn classes and the
  river blind on the turn's boards), the rest solved to 0.15% of the pot (their game takes 0.5-5 GB and 40-430 s on
  all cores) and, for the 3bet pots, to 0.03%:

  | spot, flop | entry, % of the pot in its own game | EV given up, OOP / IP | root, ours / theirs |
  |---|---|---|---|
  | BTN vs BB, Kc7d2h | 0.111 | 0.40 / 0.48 | check 100 / 100 |
  | BTN vs BB, JcTd8d | 0.089 | 0.29 / 0.17 | check 97 / 100 |
  | CO vs BB, KcTd6h | 0.120 | 0.54 / 0.27 | check 100 / 100 |
  | SB vs BB, 8c6d4c | 0.069 | 0.30 / 0.28 | check 100 / 100 |
  | BB 3bet vs BTN, AcJd3h | 0.064 | 0.30 / 0.26; at 0.03%: 0.31 / 0.26 | bet 33% 89 / 60, 75% 11 / 40 |
  | BB 3bet vs BTN, Qc5h5d | 0.129 | 0.48 / 0.32; at 0.03%: 0.49 / 0.33 | bet 33% 99 / 44, 75% 1 / 56 |

  The library's flop strategies give up 0.2-0.5% of the pot against every card dealt, the cost of their abstraction
  (each entry is within 0.15% of the pot of its own game's equilibrium); solved five times tighter the 3bet pots
  give the same numbers, so it isn't their solve's noise. The single-raised roots check as theirs do; the 3bettor's
  c-bet splits between its two sizes differently (theirs bets 75% pot 40-56% of the time, ours 1-11%).
* **Solver speed** (`scripts/bench_solver.py --curve`, BTN vs BB single-raised pots, one thread):
  in Pyodide a flop solve runs ~1.5x the iterations per second it did (Fast 15 -> 23, Accurate
  11.5 -> 17.5) and needs ~30% fewer of them, so after 4 s it is 0.3-1.1% of the pot from
  equilibrium where it was 1.2-2.7%; the Accurate turn
  (Ks7d2cTh, every river) runs 6.3 -> 11.6 it/s, 1.76% -> 0.31% of the pot after 8 s.
  Natively Fast and Accurate flops run 150 -> 210 and 105 -> 170 it/s, and the server's
  default config 8 -> 35 it/s (Qs8d3c after 4 s: 8% -> 0.6%).
* **Browser timings** (`scripts/bench_browser.mjs`: `gto/static/worker.js` itself, its Python setup and request
  handler, run in Pyodide 0.26.4 under Node, a fresh worker a request, wall clock on a 10-core machine shared with
  other jobs, so +-30% in iterations by load). "Bar" is the page's estimate, what its progress bar counts down: the
  street's budget plus each earlier postflop street's narrowing budget (Fast 8 + 3 s, Accurate 60 + 12 s). Before,
  the solve loops had the budget and the tree's build, the iteration in flight at the deadline and the exploitability
  read after came on top (Accurate flop +4.5 s, a two-tone one +9 s); a heads-up solve's budget now covers its build
  and keeps one iteration's time back (`solve(reserve=)`, `advisor.HU_RESERVE`), as the 3-way solver's did. 3-way
  solves stop once settled (`advisor.MW_SETTLED`, 0.1% of the pot, checked at 25, 50, 100, ... iterations): Accurate
  ran 3-way turns and rivers its whole 60 s. Fast's turn also gets the 3 s its narrowing flop didn't spend
  (`spare_prior`, as natively). Each cell: seconds, iterations, % of the pot from equilibrium. To run it:
  `npm install pyodide@0.26.4` in a directory of its own, then `node scripts/bench_browser.mjs --pyodide <that dir>
  [--spots flop,turn] [--speeds fast,accurate] [--worker edited-copy.js] [--json out.json]` (`--list` shows the spots; it builds `solver.zip` as the deploy does, with the repo's `.venv` Python or `PYTHON=`).

  | request (6-max 100bb) | bar | before | after |
  |---|---|---|---|
  | flop in the library (K74r), either speed | 8 / 60 | 0.02 | 0.02 |
  | flop K72r, Fast | 8 | 8.1, 421, 0.07 | 8.1, 217, 0.22 |
  | flop K72r, Accurate | 60 | **64.6**, 54, 2.9 | 60.2, 870, 0.03 (64 buckets) |
  | flop 986 two-tone, Fast | 8 | 8.4, 133, 0.42 | 8.1, 142, 0.37 |
  | flop 986 two-tone, Accurate | 60 | **131.6**, 58 (two budgets), 2.1, 588 MB | 60.3, 518, 0.05, 164 MB |
  | turn, flop from the library, Fast | 11 | 8.5, 76, 0.57 | 11.2, 95, 0.40 |
  | turn, flop from the library, Accurate | 72 | 61.7, 203, 0.28 | 60.5, 186, 0.33 |
  | turn, flop solved to narrow, Fast | 11 | **12.1**, 51, 1.24 | 11.3, 66, 0.74 |
  | turn, flop solved to narrow, Accurate | 72 | 73.9, 141, 0.45 | 72.5, 188, 0.31 |
  | turn in exploit mode, Fast / Accurate | 11 / 72 | 11.9 / 73.7 | 11.3 / 73.5 |
  | river, Fast / Accurate | 14 / 84 | 13.7 / 31.8, 1000 | 11.5 / 29.4, 1000 |
  | raked flop (K74r, 5% capped at 3), Fast | 8 | 8.5, 172, 0.33 | 8.5, 88, 1.13 |
  | raked flop, Accurate | 60 | **65.8**, 66, 1.9 | 60.4, 413, 0.07 |
  | 3-way flop, first to act, Accurate | 60 | 59.8, 56, 6.3 | 59.7-60.0, 63-77, 3.3-5.0 |
  | 3-way flop, checked to, Accurate | 60 | | 60.1, 161, 0.27 |
  | 3-way turn, Accurate | 72 | 59.8, 330, 0.03 | 34.0, 200, 0.07 |
  | 3-way river, Fast / Accurate | 14 / 84 | 8.0, 41, 0.43 / 60.0, 386, 0.01 | 7.9, 48, 0.33 / 15.1, 100, 0.07 |
  | preflop | | 0.01 | 0.01 |

  Later (the heads-up solver's chance reaches computed once a pass and once a run, `Solver._child_reach`; medians of 3
  on an idle machine): the Fast turn after a library flop 172-177 -> 186-194 iterations, 0.14-0.16 -> 0.12-0.15% of the
  pot from equilibrium, in the same 11.1 s; the river unchanged (1000 iterations in 6.2 s).

  Every request now ends within 6% of the bar (a raked Fast flop: the rake's exploitability costs two more passes).
  Fast's 3-way flops and turns are answered by the rules in 2.4-3.0 s. The worker's setup (Pyodide, numpy, treys,
  the solver's files) took 3.3-9.9 s before the first request, outside every bar (now 1.2-1.4 s: Worker startup). The wasm heap peaks at 490 MB (an
  Accurate turn with its narrowing flop), was 588 MB (the two-tone Accurate flop). Those are one answer's; over a
  session the solves kept for reuse added up (each street's, every hand's, never dropped: a Fast turn holds ~195 MB, its
  river ~185, an Accurate turn ~240), and 12 answers over 5 hands (Fast and Accurate turns and rivers) took the heap to
  its 2 GB, after which every solve failed for want of memory until the page was reloaded. Now the solves both speeds
  keep share one budget, 700 MB in the browser (`Advisor.cache_mb`; 1000 MB natively), the least recently used going
  first (a later street then solves an earlier one again to narrow through it): the same 12 answers peak at 1065 MB,
  the solves kept 495-700 MB, every answer given. The deployed page in Chrome (the
  desktop app's browser, the code before these changes, the worker's own messages timed in the page) agreed: K72
  Accurate 63.4 s and 56 iterations, the two-tone flop 91.4 s and 36, Fast flop / turn / river 8.2 / 11.5 / 6.7 s.
* **Worker startup** (time to the first answer: the worker's setup, then a preflop request or a library flop, each
  ~0.02 s). The setup was the runtime, numpy and micropip, treys installed from PyPI, 16 `gto/*.py` fetched one after
  another (`no-store`) and imported. Most of the import was `gto.preflop`'s charts: `ranges._class_combos` scanned all
  1326 combos for each of ~3600 class lookups (1.4 s in Pyodide, 1.1 s natively; now a table: 0.06 s). Now the worker
  fetches `solver.zip` (every `gto/*.py` and treys, 290 KB, `scripts/build_bundle.py`; no micropip or PyPI) and numpy's
  wheel (12 MB, the largest download) at once, alongside the runtime; keeps the zip and the startup modules' bytecode
  in the Cache API under the worker's version (a repeat visit makes no request for them and compiles nothing); and
  imports only what a preflop or library answer needs (`exploit`, `breakdown`, `multiway_solver`, `pf_game` on first
  use). Its status says which of 3 stages it is in, and "Solving…" carries each stage's seconds. On the static site
  (https) the worker starts with the page instead of after the backend probe. Seconds, medians (Node: 4 runs a spot,
  preflop / library flop; Chrome: 6 runs, both spots), a quiet 10-core machine:

  | | runtime | numpy | treys | code | first answer |
  |---|---|---|---|---|---|
  | before, Node (`scripts/bench_browser.mjs`'s sandbox) | 0.76 | 0.13 | 0.91 | 1.90 | 3.68 / 3.78 |
  | after, first visit | 0.73 | 0.13 | | 0.47 | 1.32 / 1.38 |
  | after, repeat visit (bytecode cached) | 0.73 | 0.13 | | 0.40 | 1.24 / 1.31 |
  | before, Chrome (the deploy layout served locally, HTTP cache warm) | 0.65 | 0.15 | 0.23 | 1.85 | 2.93 (2.78-3.44) |
  | after, first visit (Cache API empty) | 0.69 | 0.15 | | 0.48 | 1.35 (1.27-1.79) |
  | after, repeat visit | 0.69 | 0.14 | | 0.40 | 1.24 (1.15-1.41) |

  On a loaded machine the gap grows: 8.0 s before, 4.2 s after; the deployed page took 10.9-12.3 s then, 7.4-8.3 s of
  it loading the code (its 16 fetches alone 0.66 s in a row, 0.05 s in parallel). A first visit also downloads
  ~25 MB (Pyodide 13.7, numpy 12.0; micropip and packaging's 0.3 MB are gone), which a slow connection pays once.
* **Regret minimizer** (`docs/cfr_research.md`, `scripts/bench_solver.py --algos`, exploitability per
  iteration): predictive CFR+ and its discounted form (PCFR+, PDCFR+), the state of the art on many
  non-poker games, are 2-7x more exploitable than DCFR here at 100-200 iterations on every spot (the river,
  turns, Fast's bucketed flops and Accurate's per-combo ones), as the papers themselves find on poker; CFR+
  3-5x; the published hyperparameter schedules (HS-DCFR, HS-PCFR+) need thousands of iterations to pay.
  Within DCFR only the averaging mattered: gamma 3 rising to 8 (the default now) is 10-20% less exploitable
  than gamma 4 from 50 iterations on (Fast flops at 100 iterations 0.75-1.35% of the pot where gamma 4 was
  0.98-1.55%) and as good before, and the flop's aggregate frequencies are 8-17% closer to a long solve at
  60-200 iterations; an iteration costs the same. Beta 0 throughout swung late on two of 15 Accurate turns
  (found by the postflop-solver benchmark above: from 0.15-0.33% of the pot to 2-2.8% around the 175th-300th
  iteration, a line one player had stopped taking losing its negative regrets to the halving), so beta rises to 0.5
  at the 100th iteration: no swings on those turns (0.06-0.08% at 400 iterations), 5-13% more exploitable at
  200-300 iterations on the standard spots, the same through 100 (Fast turns). `tests/test_cfr.py` checks the solver against AKQ,
  clairvoyance and [0, 1] games solved in closed form, and its exploitability against best responses
  computed independently.
* **Native trees** (`gto/native.py`, server and CLI): `Config()`'s full menu (two bet and two raise
  sizes a street) ran 10 turn iterations in the server's 2.5 s, 60-70% of the pot from equilibrium,
  and narrowing on it left BB 7 combos checking a K72T5 river where a long solve keeps 81 (69.5 since
  the BB's flats follow published raked solves). Fast played the browser's one size with every river
  (1.5-2.5% of the pot from equilibrium after 2.5 s, 40-45 iterations unloaded; that range kept 79);
  it now bets 33% or 125% on the turn with the river at 75% and river classes, with the narrowing
  flop's unspent 2.5 s (0.03-1.1% after 5 s, 80-150 iterations, 380 in a 3bet pot), Accurate
  33/75/125% on the turn and river with river classes (0.15-0.55% after 30 s;
  it had two sizes with every river). A river root bets 33/75/125/250% in both (Turn and river
  against reference solves above). BLAS is held to one thread there.
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
  80-150/45 (tests.test_native judges by iterations, not time). A call read by the rules keeps the hands they
  have raising instead at `SLOWPLAY` (0.5; 0.85 on a later street's call), as multiway, where heads-up they were
  cut to the off-rule weight, and facing an all-in (nothing to raise) it keeps them whole: HJ vs BB on
  Q♥3♠2♠ (c-bet called), J♠ (126% overbet called), 3♣, at 30 flop and 15 turn iterations (9% and 18% of the pot),
  BB's flop and turn calls kept 18% of its sets and 27% of its flushes where they kept half its air, its river
  range was 6.1 flushes in 131 combos (now 11.6 in 144), and A4c jammed 100% at +4.4 bb over a check (now +2.6).
  A turn solved to 0.1% keeps 57% of the flushes calling that overbet; the flop's check-raises take most sets.
* **3-way pots are solved** (`gto/multiway_solver.py`, `docs/multiway_research.md`) where three players are live,
  nobody is all-in and the street is one the speed solves: the river in Fast, every street in Accurate. A solve
  over 10% of the pot from equilibrium after its budget hands over to the rules. The result has the solver's
  mix, EVs (relative to folding now), grid and each player's best-response gain, and "exploitability" is their
  mean. A river iteration costs 30-90 ms natively and 0.13-0.16 s in Pyodide (measured, `scripts/bench_browser.mjs`),
  so Fast's 2.5 s / 8 s run 25-80 (the browser's: 41-48, 0.3-0.4% of the pot from equilibrium). Turn and flop solves
  bet the later streets by hand buckets (Known limits; `docs/multiway_research.md` has the model): a turn iteration
  costs 0.1-0.2 s / 0.18 s, a flop one (bets of 33 and 75%) 0.1-0.4 s / 0.34 s checked to and 0.7-0.95 s first to
  act, after 0.6-1.7 s of tables natively, so both wait for Accurate (30 s / 60 s; unloaded natively flops checked to
  run 200-290 iterations, 0.15-0.25% of the pot from equilibrium, first to act, a bigger tree, 100 and 0.8%; in the
  browser 161 and 0.27%, 56-77 and 3.3-6.3%; 8 buckets, 112 iterations in the same 60 s, was no closer to the deeper
  solve below on its 8 spots: best action 76% against 72%, 0.020 bb given up against 0.029, but EVs 0.79 bb off
  against 0.47 and range bets 6.1 points against 3.2). The budget covers the tables and the gains and EVs read off after the
  iterations: a 15 s flop answer took 16.4-16.9 s and now 14.8-14.95 (the browser's 59.7-60.0 s of its 60). A solve
  stops once settled (`advisor.MW_SETTLED`: the players' mean best-response gain under 0.1% of the pot, checked at
  25, 50, 100, 200, 400 and 700 iterations, about two iterations' time each): the browser's Accurate ran a 3-way river
  its whole 60 s to 386 iterations (0.01%) and a turn to 330 (0.03%), now 100 in 15 s and 200 in 34 s (0.07%). Against deeper solves (`scripts/compare_multiway.py
  --flop-reference`, `--turn-reference`, 88 and 63 hands) the bucketed flop picks the best action 90% of the time
  on the one-size tree, 75% on the two-size one (choosing between near-tied sizes; with the turn and river dealt out
  55%, the rules 56%), gives up 0.003 / 0.011 bb (dealt 0.055, the rules 0.076) and its range's bets are
  1.6 / 0.6 points off (dealt 18.3 / 14.6); the turn 89% (87%, 75%), 0.03 bb (0.06, 0.24). The best-response gains (CO opens, BTN
  and BB call, river) are 4.7/3.5/4.1% of the pot after 10 iterations, 0.8/0.8/0.5 after 30 and 0.1 after 100 (`scripts/bench_multiway.py`). With three players CFR has no equilibrium guarantee;
  these gains shrinking is the evidence. On four river boards (`scripts/compare_multiway.py --facts`):
  - The BB leads 4-14% of its range into two players, against 13-21% into the CO alone.
  - Last to act, a player bets 14-32% when checked to, against 23-44% heads-up, and the top 10% of the range
    always bets.
  - Facing the CO's half-pot bet, the BTN folds 68-73% and the BB 49-53% after it. Both fold 34-37% of the
    time, about MDF's 33%: the defence is shared, not each player's own.
  - Played two-handed, the solver matches the heads-up solver on the same tree (bet 16.6% against 16.5%,
    EVs within 0.03-0.06 bb).

  Against the betting rules (`--spots`): 580 hand-spots on 4 rivers and 2 turns, with BB first, BTN checked
  to, BTN facing a bet, and BB facing a bet and a call. Overall the rules pick the solve's most frequent action
  74% of the time and give up 0.15 bb (1-3% of the pot) by the solve's EVs. The sample hands lean strong, which
  shows where the two differ:
  - First to act on the river, the rules bet 1% of these hands and the solve 25%. On the turn the rules bet 26%
    and the solve 3%.
  - Checked to, the rules bet 54-55% and the solve 79-83%.
  - Facing a bet with a caller between, they agree 84-95% of the time.
  - The rules' EVs are 2.4 bb off first to act and 0.1-1.9 bb off elsewhere.
* Multiway pots neither solver takes (4+ players live on the flop or turn, 6+ on a river in the browser's Quick
  and Fast, or one of the cases above) are not solved. Instead a Monte
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
* 4+ way flops and turns stay with these rules (GTO Wizard doesn't solve 4+ way postflop either). Scored against
  4-way turn solves that play the river (`scripts/score_multiway4.py`: HJ opens, CO, BTN and BB call, the flop
  checked through; six turns, each of four lines, 40 hands of hero's range drawn by weight; one bet size a street,
  and only hero may raise, so if anything they overvalue hero's raises), the rules give up 0.70% of the pot to the
  solve's mix and pick its best action 58% of the time (were 0.91% and 54%): first to act 0.40%, checked to last 0.35%
  (was 0.77%: the players who checked raised a bet of ours with RAISE_SHARE of what continues, a fifth, where a
  check keeps few raises; CHECKED_RAISE_SHARE, 0.05, has the rules bet 53% of hands there as the solve does, not
  36%), facing a bet 0.57%, facing a bet and a call 1.49% (was 1.89%: the trials' later streets pay a raise off too
  often, so top pair with a weak kicker raised AcKd5c5s for +3.9bb over calling where the solve calls for +2.8bb;
  there a raise needs ROOT_RAISE_EQ, 65%, equity against what each player continues with). What's left facing a
  bet and a call is mostly semi-bluffs: the solves raise strong draws (KhTh on Jh9h4c8s, QJs on Kh7d2c9s) that the
  rules call or fold, but those solves never face a re-raise, so how much of that holds is unclear. The same solve
  with the river dealt out gives up 0.29% of the hand-weighted pot (natively 3.4-5.5 s for 30 iterations, settled to
  0.13% in its own game; the rules take 0.1-0.2 s), but it plays the references' own game (their one size and
  hero-only raises, the river aside), which flatters it: on 3-way turns, against a richer solve, dealt-out gave up
  6.5% of the pot where the rules gave up 1.5% (see N-player pots), so the rules stay.
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

### Precomputed flop library

The commonest flop decisions come from solves made offline (`gto/flop_library.py`, built by
`scripts/build_flop_library.py`), so they are instant and settled at either speed, in the browser too.

* **Spots** (`flop_library.SPOTS`, most common first): 6-max, 100bb, no ante or rake, the page's default sizes:
  BTN, CO, HJ, UTG and SB vs BB single-raised pots, BB vs BTN and SB vs BTN 3bet pots, and BTN 3bets CO. A spot
  is matched by the two ranges entering the flop (as the charts replay them: other table sizes, stacks or charts
  simply don't match), the pot and the stack; heads-up from the flop only, not in exploit mode, with locks on the
  villain, custom ranges or Optimize sizes.
* **Flops**: canonical flops matched up to suits (the ranges are suit-symmetric, so a stored flop's solve is any
  suit permutation's, its combos permuted); other flops are solved live. Each spot stores a representative set
  (`representative_flops(n)`: the 1755 suit-isomorphic flops sorted by texture and sampled by weight, so each suit
  pattern, pairing and height gets its share): 184 flops (12.7% of all flops) for the 3bet pots; 552 (38%; it holds
  the 184) for HJ, UTG and SB vs BB; every flop for BTN and CO vs BB (`SPOT_FLOPS`).
* **Coverage** (`scripts/flop_library_coverage.py`: the 6-max 100bb preflop game played by the charts, the flop by
  the stored strategies): 36% of hands go heads-up to the flop, and the library answers 49.2% of their flop
  decisions (36.1% before CO vs BB on every flop and SB vs BB on 552, 21.3% before format 2 made room, 9.4% with
  184 flops everywhere). Of the rest, a flop not stored is 26 points, a spot not in the library 22 (HJ / UTG
  flatted by the BTN 6.3, CO vs the SB's 3bet 1.2, each other 3bet pot ~1, 4bet pots 1.5 together), a raise 2.7.
  A flop's entry is worth its spot's share times its suit count, so more flops of the commonest spots beat new
  spots; by points per MB (`--assume`): BTN vs BB on every flop +12.4 for 4.5 MB, CO vs BB on 1656 (90%) +7.6 for
  2.8, UTG vs BB on 552 +2.4 for 1.05, SB vs BB on 552 +3.5 for 2.1 (its wide ranges' entries are the biggest), all
  built; next, on every flop: SB vs BB +8.6 for ~6.9 MB, HJ vs BB +7.8 for ~3.9, UTG vs BB +6.1 for ~3.4 (HJ vs
  the BTN's flat on 184: +0.3).
* **What is stored**: the flop's first decisions (the root; the in-position player after a check or facing a
  lead; the out-of-position player facing a bet after checking), only where the other player gets there with 5%
  or more of their range (a lead the big blind never makes is the solve's noise, and the advisor solves live
  behind a line under 10% anyway). Per combo: the average strategy in 1% steps and every action's EV (the best
  to 1% of the pot, the others as a gap below it, 1% steps then 4%), plus range totals.
* **Format 2** (`flop_library.pack`, `FORMAT`; `drills.js` reads it too, and every reader ignores another format's
  index): one value per suit class, the combos the flop's own suit permutations map to each other (AhKh and AsKs on
  K♣7♣2♦; two-tone flops store 832 of 1326, monotone 416, rainbow all), their strategies averaged by reach and
  their EVs plainly; per node in hand order (ranks high to low, suited first), zlib. The solve tells those combos
  apart only by its sampled turn and river cards, by 4.7 points of mix per combo on average: averaged, 5 BTN vs BB
  two-tone and monotone flops came 15-35% closer per combo (mix 8.5 -> 6.8 points, EV 3.0 -> 2.6% of the pot) to a
  solve playing every turn card. EVs in 1% of the pot (was 0.02 bb) add 0.2-0.35% of the pot of error per combo,
  against 2-5% between the library's model and every turn played (frequencies: 0.2 points). Recoded from format 1
  without solving (`--recode`): 14.06 -> 8.77 MB (rainbow flops -22%, the rest ~-50%), 1.5-6 KB a flop. Pages
  gzips `.bin` too (`application/octet-stream`, `content-encoding: gzip`), but zlib'd files gain nothing (+0.7%):
  what's stored is what's downloaded, so the 25 MB cap is on both.
* **Tree**: Accurate's flop model (one turn per rank class, the river bet on; the browser's Accurate config,
  which plays the same flop game as the native one), two flop sizes, 300-800 iterations, to at most 0.15% of the
  pot from equilibrium (mean 0.107%; paired and trips boards settle slowest, and 17 that the old cap of 500
  iterations left at 0.15-0.54% were solved again) (the live Accurate solve gets ~0.3-0.4% in the server's 30 s and 60-75 iterations in the
  browser's 60 s). The answer's `precomputed` field (a badge on the page) says which spot and how many iterations;
  `same_tree` is false in Fast mode, whose own flop model plays the turn and river by hand buckets: converged
  (300 iterations) on 20 of the library's BTN vs BB flops its c-bet is 2.8 points from the library's on average (up
  to 7, betting more), BB's response to the small c-bet 2.9; at its 2.5 s and 8 s budgets it's further. (The old
  Fast model, the river checked down on random turns, was 5.6 and 8.6 off, up to 15 on dry boards.) Natively the
  answer takes ~0.01 s instead of 2.5 / 30 s; in the browser ~0.03 s instead of 8 / 60 s, and a turn spot saves
  the earlier flop solve too (BTN vs BB, J75 c-bet called: 8.2 s for the turn in Fast mode, 11.5 s before; measured on the old Fast model).
* **Lines**: the advisor walks the actual flop action through the stored nodes (sizes taken more than 15% off the
  tree's, more raises than it allows, or a line through a raise fall back to the live solve). Turn and river spots
  narrow the flop's ranges through it as well, instead of the earlier street's short solve. The entry ignores the
  hero's cards in the villain's range (the live solve drops them): an edge the size of a blocker.
* **Flops not stored** are solved live from scratch, never answered from another flop. A warm start from the
  spot's most similar stored flop (same pairing and suit pattern, nearest ranks; strategies carried over by card
  substitution and seeded as regrets) was measured and left out (`scripts/bench_library_warm.py`, 12 unstored BTN vs
  BB flops, geometric mean % of the pot, cold -> seeded): it helps early (Accurate at 10 / 40 / 80 iterations: 81 ->
  53, 5.3 -> 3.9, 1.10 -> 0.98, the same with the browser's NO_BLAS kernels: 60 iterations 2.08 -> 1.60), is even by
  100 (0.68 / 0.68) and then worse, its seeded regrets outlasting DCFR's discount: 200 iterations 0.19 -> 0.24, 400
  0.07 -> 0.09, 10-12 of the 12 flops worse (the browser's bucketed Accurate the same, Fast at 120 0.50 -> 0.58).
  Every budget runs past 100 (Fast 140-420, Accurate 250-450 natively, 520-870 in the browser).
* **Invalidation**: `index.json` holds each spot's range fingerprints and the tree; an entry whose spot, tree or
  `MODEL_VERSION` (bump it when the flop game changes in a way the tree key can't see) no longer matches the code
  is ignored, and a rebuilt index leaves stale entries out (a test checks the shipped index against the charts).
* **Browser**: the worker fetches `flop_library/index.json` the first time a flop comes up and one entry
  (`<spot>/<flop>.bin`, cache-busted by its hash) when its spot does; `scripts/deploy.sh` copies the folder.

Building (resumable: entries on disk for the current charts and tree are kept, so re-run the same command after
a stop; progress lines go to stdout, the index is rewritten every 20 entries, and with it each spot's
`report.json` for the flop reports: per flop the stored nodes' range frequencies, the root's EVs, range equity over
every 8th runout (within ~0.3 points of exact) and a hand to load; only changed entries are recomputed, ~0.15 s
a flop, all eight spots from scratch ~3.5 min):

```bash
nohup nice python scripts/build_flop_library.py --jobs 5 > reports/flop_library_build.log &   # every spot, its SPOT_FLOPS
python scripts/build_flop_library.py --spots srp_btn_bb --flops all                     # all 1755 flops of a spot
python scripts/build_flop_library.py --spots srp_btn_bb --boards Ks7d2c,Qs8s3d          # given flops
python scripts/build_flop_library.py --index-only
python scripts/build_flop_library.py --recode        # format-1 entries to the current format, no solving (~5 min)
```

Each entry takes ~40-60 s natively on a rainbow flop and up to twice that two-tone, on an idle core (several
times that on a loaded machine; 3bet pots ~10-25 s): the whole library (5718 entries) ~16 hours on 5 cores (BTN vs
BB's other 1203 flops and UTG's 368 took 4 hours; CO vs BB's other 1203 and SB vs BB's 368 3.5 hours on 7). After
changing the charts or the flop model, run it again (stale entries are re-solved). A spot on every flop is 5.5-7 MB
(SB vs BB's wide ranges ~10): the test holds the library (20.06 MB) under 25 MB.

### Precomputed 3-way flop library

A 3-way flop needs ~8 s of live solving even in Fast's smaller game and Accurate's 30-60 s for its own, so Quick and
Practice's quick answers left it to the betting rules. The commonest 3-way pots' flops are solved offline instead
(`gto/mw_library.py`, built by `scripts/build_mw_library.py`), so the flop's first decision is a real 3-way solve,
answered at once at every speed (Quick and Practice's `"quick": true` too), in the browser as natively.

* **Spots** (`mw_library.SPOTS`; 6-max, 100bb, no ante or rake, 2.5bb opens), with how often the charts deal each
  line: BTN opens, SB and BB call (0.25% of hands); HJ opens, CO or BTN and BB call (0.22% each: the CO's cold call
  is the BTN's, so one game serves both); UTG opens, BTN and BB call (0.18%); CO opens, BTN and BB call (0.14%).
  `mw_index.json` lists the spots built (a spot in `SPOTS` not built yet just isn't answered from it).
  A spot is matched by the three ranges entering the flop (as the charts replay them), the pot and the stack; not
  with rake, custom sizes or ranges, exploit mode, locks or Optimize sizes.
* **Flops**: the heads-up library's 184 representative canonical flops (`flop_library.representative_flops`), plus
  K72 rainbow (`EXTRA_FLOPS`), matched up to suits; other flops are solved live (or, quick, left to the rules).
* **What is stored**: one 3-player CFR solve from the flop's start on native Accurate's 3-way game
  (`advisor.MW_LATER`, `MW_FLOP_BETS`: bets of 33 and 75% pot, one raise and all-in, the turn and river bet on by
  16 hand buckets a player), run to under 0.1% of the pot mean best-response gain (300-500 iterations, ~3-4 min a
  flop on a loaded core), stores the flop's first decisions: first to act, checked to once, checked to twice. The
  solve takes every hand the flop doesn't block (`include`, ~7% more time), so a hand off its player's range (a
  cold call the charts 3bet or fold) gets its current strategy, as a live solve gives it. Per hand: the mix in 1%
  steps and each action's EV, in the heads-up library's entry format (`flop_library.pack`, one mask per player).
  An entry is 3.8-7.6 KB; a spot's 185 flops ~1.1 MB, the four spots' 740 flops 4.53 MB
  (built in ~2.5 h a spot on 8 loaded cores).
* **Lines**: a line through a bet isn't stored; facing a bet the flop is solved live (or, quick, by the rules) as
  before. Practice opponents after a library answer read their move off it (`_mw_actor_move`), and a turn after the
  flop checks through narrows the flop's ranges through it (`_mw_walk`), where the rules did before.
* **Accuracy** (`scripts/compare_mw_library.py`, CO / BTN / BB on K72r, KJ2 two-tone, K82 monotone, K55, against
  the native Accurate advisor's live solve of each node in its 30 s, library off): the library's solves are 0.06-0.08%
  of the pot from equilibrium (300-600 iterations); the live ones 0.85-1.9% at the root (80-130 iterations) and
  0.1-0.8% checked to, from ranges the betting rules narrowed through the checks (the library's are the solve's own).
  Their acting ranges' action frequencies are 8 points apart on average (2-18; the BB's root leads 2-18% live,
  0-0.5% in the library), the same most frequent action for 85% of hands, per-hand mixes 18 points apart and EVs
  3.9% of the pot (1-12.5).
* **Browser** (`scripts/bench_browser.mjs`, `mw-flop` first to act, `mw-flop-x` checked to twice): answered from the
  library in 0.49-0.57 s at Quick (`BENCH_STATE='{"quick":true}'`), Fast and Accurate alike, most of it the first
  request importing the 3-way solver (Accurate's live solve took its 60 s, Quick's answer was the rules');
  `mw-turn` narrows the checked-through flop through it (followed 100%). The worker fetches `mw_index.json` the first
  time a 3-way flop comes up and one entry when its spot does.

```bash
nohup nice python scripts/build_mw_library.py --jobs 8 > reports/mw_library_build.log &   # every spot (resumable)
python scripts/build_mw_library.py --spots mw_co_btn_bb --boards Kh7d2c                  # given flops
python scripts/compare_mw_library.py --spot mw_co_btn_bb --boards Kh7d2c,9s8s5d          # against live Accurate
```

### Deploy to the website

```bash
scripts/deploy.sh
```

copies the page, the worker, `practice.js`, `reports.js`, `drills.js`, `breakdown.js`, `session.js`, `solver.zip` (the worker's Python code: every `gto/*.py` and treys, `scripts/build_bundle.py`), the `gto` sources and the flop library into the `gto-solver/` folder
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
skips them, for emergencies only. `BUILD_ONLY=<dir> scripts/deploy.sh` builds the folder as it would be published
(versions stamped, each view's URL written) into `<dir>/site/gto-solver` without the checks, cloning or pushing, to
try the static site: `python3 -m http.server -d <dir>/site 8000`, then http://127.0.0.1:8000/gto-solver/practice/.

Tune with `Config` in `gto/solver.py` (`depth`, sizes, sample counts) or the
`--budget` / `--depth` flags. `depth=3` from the flop is the full tree; it is
much slower in pure Python. The real fix for a production version is moving
the tree traversal to C or Rust or a GPU, which makes full flop trees feasible.

### Known limits

* Preflop charts are written by hand, not solver output, and the off-chart adjustments are rules on
  top of them. They are matched to published solves where any exist: `scripts/preflop_reference.py`
  measures 26 spots (opens, BB and SB defence, cold calls, openers facing a 3bet; the raked ones played
  at that rake), 15 hand-level facts and 11 raked ones against them (tests/preflop_refs.py, average error
  0.96 points over the 26, 0.99 over the first 22 where it was 1.03, and 2.9 before that with 10 spots and 8 facts out). Squeezes, the
  response to a 4bet, and cold players facing a 3bet have one published figure each (reported, not enforced:
  the big blind's squeeze over BTN + SB 75.9/15.0/9.1 against Poker Academy's 71.2/16.1/12.7, its response to
  a 24bb 4bet 44.9/33.4/21.7 against GTO Gecko's 46.9/29.3/23.8); 5bets, heads-up and limped pots have none
  and are judgment. A derived equilibrium (`gto/pf_game.py`, realization measured
  with the postflop solver) fits the published spots worse (2.4-3.0 points, 10-14 spots out; each structural cause tested in
  docs/preflop_research.md section 7) and was not used to replace them; it says the 3bettor folds too often to the charts' 4bets (it continues 44-92% where
  the charts continue 28-56%), but that is a best response to the charts' bluff-heavy 4bet ranges; the one
  published response found (the big blind against the button's 24bb 4bet) continues 53%, where the charts
  continue 54% with the same cards removed (docs/preflop_research.md sections 6 and 8). The charts' continue against a 4bet swings with its
  size more than a solve's would: the re-pricing adds calls by equity against the 4bet range, which a 3bet
  range has bunched just under the chart's bar (the big blind against the button's 4bet folds 64% at the
  default 27.6bb, 45% at 24bb, 5% at 20bb). `scripts/preflop_sweep.py` runs
  every standard line for 2-9 players and flags broken frequencies and any hand continuing less often than one it dominates (pairs, kickers, suited over offsuit; the strategy is also kept that way at run time). Replace them with real chart
  exports if you have them; the format is a range string.
* Limps, antes, straddles and stack depth are handled by adjusting the 100bb
  no-ante charts (see Preflop above), not by charts built for those games.
* Rake (optional) re-prices preflop calls (and push/fold) by one expected-rake rule (2.5x the flop
  pot, fit to one published raked solve), the charts fit to raked solves only past its 5%/4bb; how many
  calls a worse price removes is set by all-in equity, which hands by realized equity (gto/pf_eqr.py and
  the river correction, see Rake above: the big blind keeps 76s, 54s, T8o and 98o and folds A2o and K5o, as
  raked solves do, where all-in equity did the reverse). Its hand-level shape is checked against one source
  only (gtopreflop.com, the 5 big-blind facts and 6 of the button against a small-blind 3bet in
  tests/preflop_refs.py RAKED_HANDS, all in range: Q9o 33% against 27%); offsuit kings K9o/KTo 73/69% and
  K7o/K8o folded have no published hand frequencies, and the button folds A5s to a small-blind 3bet 80%
  (gtopreflop.com calls it 72%). 3bet and 4bet ranges aren't re-priced (the big blind's already match the raked
  solves'); equity leaves where the tree stops early are raked on the pot as it stands, and the multiway
  heuristic rakes its payouts but not the betting rules inside its trials (see Rake above).
  BTN vs BB on K72 at 5% capped at 0.6bb (both ranges re-priced, the flop raked) still c-bets 49% (QQ6 81%,
  the BB checking 95%) against 69% unraked and published 90%+: the hands the re-pricing swaps don't move it.
* Tournaments (ICM): Malmuth-Harville equity (no future-game model: a big stack's edge from covering others,
  or the blinds' next orbit, isn't counted), at most 12 players modelled (stand-ins for a bigger field). Preflop
  the charts are chip charts re-priced by a risk premium and the commitment factor ICM_COMMIT (1.0, judgment
  fit to the published shape: 2.5 cut the big blind's defence of a button min-raise to 31%), and push/fold
  equilibria run in prize equity. After the flop only heads-up pots are played in prizes; multiway pots stay
  in chips.
* Exploit mode's pool is fitted to public population data where any was found (flop and river
  folds, turn folds and river bluffs from 500nl+ regulars, preflop frequencies, one flop spot's raise
  frequency, and the small blind's open-limps); folds to big flop bets and the turn's shape at small
  stakes are only corroborated by whole-sample aggregates that pool every pot type, and how much less the
  pool bluffs on the flop and turn, and how its extra preflop hands split between opens, open-limps, flats
  and the big blind outside the small blind, are still judgment calls. On a rough (Fast, flop or turn)
  solve the best response also picks up the solve's own noise.
* 3-way solves play two flop bet sizes (33 and 75%), one turn and river size and one raise; they start from the node
  in front of you. This street's earlier actions and earlier 3-way streets narrow the ranges through the solve your
  advice there came from, followed as far as it had settled (advisor.trust_of), else by the betting rules: against
  the turn ranges a 400-iteration flop solve leaves (`--narrowing`, four flops, the blinds check, UTG bets a third,
  both call) the rules are 0.44 off (total variation; were 0.52, the BB overcalling with 140-170 combos for the
  solve's 30-50, now ~55: an overcall keeps advisor.OVERCALL, a quarter, of the first caller's share; 0.52 -> 0.46 on
  four other flops), the 60-iteration solve's walk 0.43 and the 150-iteration one 0.22. The blinds' flop checks in
  front of the opener narrow by the rules too, now keeping CHECK_SOFT (0.7, was 0.3) of what the rules have betting:
  against a solve from the SB's action (ten flops) their check ranges were 0.21 / 0.14 off (SB / BB), now 0.15 /
  0.05. Without such a solve an earlier street stays with the rules: solving it from its start
  (Advisor.mw_prior_solve) gets 40-65 iterations in native Accurate's 12 s, 4-6% of the pot from equilibrium,
  followed 0-20%, so it is off. Everyone plays the shortest stack behind (a deeper player's all-in there is shown
  and played as a bet to that stack). Below the street solved hands are bucketed (all of a bucket plays alike there)
  and the river under a flop is played without its card, as heads-up Fast. The opener's flop c-bet, UTG open called
  by both blinds (`--flop-study`, 33/75%): what the advice plays, the blinds' checks narrowed by the rules
  (`--rules-checks`, five flops), bets 51.5% into two (72.5% with the old check rule), where the street solved from
  the SB's action bets 49.0% at that node (`--full-street`; the blinds check through 21-98% of the time, 65% on
  average). With the blinds' full ranges (as if they never lead; ten flops) it bets 40% into two and 54%
  heads-up. GTO Wizard's 3-way study has 66% and 77%, with its own ranges, 20 weighted flops, the SB leading 9.8%
  and a 33%-pot raise: with that raise (`--raise-frac 33`) heads-up goes 54 -> 64%, 3-way stays at 40%. The flop
  sizes don't close the rest (as heads-up, a second size takes bets from the small one); playing the river with its
  card (heads-up, per combo, seven flops) bets 3 points more (64% against 60%); the ranges are worth 6 points 3-way;
  rake (5%, 3bb cap) and a 75% turn and river size moved a flop by 0-4 points.
* Heads-up, the betting rules that stand in for an unsettled earlier street's solve price a call by the bet's
  size alone (`_keep`: a quarter less folding than a balanced defence), not by the bettor's range, and keep some of
  every hand they rule out (`NARROW_SOFT`, up to `OFF_RULE_MAX` of the range), so the caller's range errs wide, and
  wide with weak hands, which overstates fold equity on later streets. The river solve on top of it is exact (an
  independent CFR+ with a brute-force showdown matrix agrees to 0.5 points and 0.01 bb), and the convergence badge
  reads only that solve. The Q♥3♠2♠-J♠-3♣ line above: Fast's ranges (flop 9%, turn 18% of the pot from equilibrium)
  give BB 144 river combos, 28% of them one pair, and A4c jams 100%; at 60/30 iterations (2.0% and 3.9%) 41 combos
  and A4c checks 100%; at 150/45 and beyond (0.3% and 1.2% or better) 28-30 combos, 31-33% flushes, and A4c mixes jam
  and check within 0.04 bb (about +0.75 bb each at 89bb stacks). Shading the rules to a balanced defence
  heads-up moved Fast's range only from 144 to 126 combos, so it is left as is; the earlier streets' budget is what
  settles it.
* Multiway with an all-in player for less, the heuristic pays each side pot to the best hand among the
  players in it, but the betting rules inside its trials don't play for the side pot (an opponent bets
  and calls as if the whole pot were at stake).
* Practice opponents heads-up after the flop read their move off your own solve, whose ranges leave out
  your cards (as your advice does): an edge the size of a blocker. A line of yours the solve rarely
  takes, or a pot that became heads-up mid-street, gets a solve from their seat instead (a few seconds,
  more in the browser).
* The bet-size abstraction is coarse. A real bet or raise more than 15% off every tree size is put
  into that street's tree (closer ones are played as the nearest size). The Fast turn bets 33/125% in
  single-raised pots and 33/75% where the turn starts at most 4.5x the pot deep (a 3bet pot's 3.2;
  `Advisor.spr_sizes`), so a pot in between barrels three quarters pot from 33/125%. Raises stay one size (75%)
  after the flop: a second one (40% of the pot after the call, about 2.5x) moved four 3bet turns' aggregates
  1-5 points at up to 2.6x the solve's cost, so it was left out.
* Turn and river references are this solver's own richest trees plus a few published spots; few
  published turn and river aggregates exist (GTO Wizard's are behind an account, and its blog's are
  UTG or 40bb MTT spots with wider calling ranges).
* Flop solves play the river without seeing the river card and one turn card per rank class. In
  single-raised pots the river is played by river strength (each hand knows where the river card puts
  it, not the card): against the river with its card on 3 fixed turns the c-bet is now 1-6 points
  over (it was 2-11 under, QQ6 7 under); 3bet pots keep the river on the turn's boards, where a draw is
  a middling hand (river strength overused the big bet there), so their c-bets still run a few points
  off (see Accuracy vs speed). The flop library was solved with the river on the turn's boards. 3bet
  pots match every-turn solves less well than single-raised ones (narrow ranges hold the turn cards
  played more often).
