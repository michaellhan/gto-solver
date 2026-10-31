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

The solver's compiled core (`core/gtocore.c`, see How it works) builds itself on first use with the system's C
compiler (`cc`, into `core/build/`); without one the solver runs in numpy. After a change to the C, rebuild the
browser's modules with `python scripts/build_core.py` (Homebrew's `llvm` and `lld`), which `tests/test_core.py` checks.
It runs on one thread natively unless `GTO_THREADS` says more (`gto/server.py` takes up to 8: `--threads`).

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
| `/gto-solver/study/` | Study |
| `/gto-solver/preflop/` | Preflop solve (Study on a game set in a form) |
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
per iteration than native, so the browser solves a smaller game (turn, river and flop solves run their iterations
in the compiled core instead, as WebAssembly: 5.5-6.7x numpy's on the turn, 2-16x on a flop): below a flop solve the
turn and river are played by hand buckets (32 in Fast, 64 in Accurate), and from the turn the river is one
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
rough on the turn, whose game is cut to fewer sizes when the whole one doesn't fit those 2 s, which the answer says; 3-way flops and turns solved in a smaller game, a 4-way flop left to the betting rules in the browser, and one whose tables can't be built in time anywhere); Fast is 8 s in the browser and 2.5 s against the local server (2.5 s per earlier street too); Accurate
is about 60 s in the browser and 30 s locally (12 s per earlier street), with more time for the earlier
streets used to narrow ranges. River spots are solid either way; turn spots need
Accurate to get close to a full solver (see Accuracy vs speed below).
While a heads-up turn or river solves in the browser, an approximate answer read off the solve before it (a live
flop's for the turn, the turn's for the river, or the street's own from your earlier decision; after a library flop,
its precomputed solve's own turn, where the library has stored it) shows at once under the progress bar, marked
approximate, until the answer replaces it. In position, the street is solved while your
opponent is to act, so the time you take entering their move counts toward the answer (a limped pot's turn answers
in 3.2-3.4 s with the move entered 4 s in, against 7.0-7.3 s otherwise).
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

**Simple** (the tab after Equity, where you can bet or raise: heads-up after the flop) is a one-size view,
like GTO Wizard's single-size solutions: your decision's street solved again with one bet size for each of 25, 33,
50, 75, 100, 125 and 150% of the pot (the river 200% too; all-in where the tree offers it) and once with all of them
to mix. It shows the **best single size** (the one that keeps your range the most EV, the opponent adapting to it), a
table of every size with its bet share, EV and the **EV it gives up** against mixing every size (% of the pot; sizes
within the solves' precision of the best marked tied), and the picked size's strategy: the range's check and bet
frequencies, your hand's mix and EVs, and the 13x13 grid (click a row for another size's). Facing a bet it is the
same for raise sizes: a raise to 2.5, 3, 4 and 5 times the bet (all-in where the tree offers it), each row with what
it raises to, the range's raise share, and the fold, call and raise frequencies. Always at Fast. On the
turn and river it starts when the tab opens, one size at a time with progress ("2 of 8 done · about 6 s left") and a
Stop button (Continue picks up from there): a turn takes 1-22 s for every size in the browser, a river under a second.
On the flop it waits for **Find the best size**, with its estimate (live flops take 1-3 s in all): the library's
flops answer the advice at once, and solving every flop you look at would only load the machine. Answers are kept
(by the page and the solver), so opening the tab again or coming back to the spot is instant; new advice stops its
solves. Not against the pool or locks (the tab is hidden there); in a tournament (payouts or bounties) every solve pays
prize equity as the advice's do, and the tab shows the tournament's notes. See [One bet size](#one-bet-size).

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
`flop_library/<spot>/report.json` per spot (~45 KB, ~12 KB gzipped; ~135 KB on 552 flops, 430 KB / 80 KB gzipped on all 1755, SB vs BB's 530 / 113 KB), written with the library's index
(`flop_library.build_report`, see Building below); the panel is `gto/static/reports.js`, loaded when first opened.
Each flop's **Turns** button opens its turn report in Study (below) on a line the flop library's stored turns hold:
the most reached flop line through the view's decision (any line for out of position's first decision; one that
starts with a check for in position's; check, a bet and a response facing a bet, the view's own bet size first),
named in the report's header with how often the flop solve takes it (3bet CO vs HJ on 9c6d6c, where HJ always checks:
"check, bet 12.5, call", 30% of the time, for every view; SB vs BB on Kc7d2h: "check, bet 6.6, call", 44%). Where
no held line goes through the view (or the flop has no turn file: the single-raised spots), Study stays on the
view's own line (checked through for out of position, else in position's most common bet after a check, called)
and offers **Solve the flop** first, saying why.

**Turn and river reports** (Study, at a closed flop or turn: **Turn report** / **River report** beside the next card's
deal, or a flop's **Turns** in the flop reports) are the next street's first decision on every card, like GTO
Wizard's: a table of the 49 turns (48 rivers) with the acting range's mix as a stacked bar and numbers, both
players' EV (bb, range against range: the two add up to the pot) and range equity (every river dealt under a turn),
for the first player or the second after a check. Each card is classed by what it does to the board (completes a
flush; completes a straight: three board ranks within a straight's five; pairs the board; overcard; undercard;
brick, under every board card); sort by any column, filter to a class, or group by class, rank or suit (the mix
averaged by the acting range on each card, EV and equity by the cards' chances given both ranges; a group row
filters to it). **Cards** shows the same as a 13x4 grid. Click a card to deal it. The report comes from what's in
hand first, at once: a library flop's stored turns (the 3bet pots with turn files: 0.1 s; the files shipped so far
hold strategies only, so the EV columns are "—"; a file built with the flop solve's EVs, format 2, fills them with
each turn class's range EVs and gives each action's EV for the hands that take it in its bar's tip; a line they don't
hold offers the lines they do), or the solve Study's own answer made (a
river report after Study answered the turn: 0.02 s natively; a flop solved live). Otherwise **Solve the flop** /
**Solve the turn** solves that street at the speed picked, with a progress bar against the time it should take: on
the local server the speed's budget (Fast: 2.5 s a street, and a river report's flop too where it isn't in hand:
K72r's turn report took 1.2 s, a Q84 river report 6.8 s); in the browser, where the solves stop well inside their
budgets (Fast 8 s, a river report 11; Accurate 60 and 72) once they reach the exploitability target, the time they
took (`scripts/bench_browser.mjs`'s ro-* spots, each solved cold in a fresh worker, 8 threads as the site runs / 1
where the page isn't cross-origin isolated): Fast turn reports 0.3-1.2 s / 0.5-2.3 s (single-raised, 3bet, limped
and two-tone flops), river reports 0.7-3.0 s / 1.3-7.1 s (after a bet, checked through, a 3bet pot); Accurate a turn
report 1.3 s / 3.5 s, a river report 7.2 s / 30.5 s. The button and the bar say the slowest of those, rounded up (~2
s and ~3 s at Fast on 8 threads, ~31 s for an unthreaded Accurate river report), in between for 2-7 threads. Like
Study, it reads
the equilibrium: tournament payouts, locks and exploit settings are left out. The pure parts (`gto/runouts.py`'s
equities, `gto/static/runouts.js`, loaded when one first opens) are tested from Python and Node.

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

**Study** (in the ⋯ menu) is a solution browser: walk a hand node by node with no cards of your own, and at each
decision see the solver's answer for every hand of the player to act. Start from a precomputed spot's preflop line
(BTN vs BB, …: instant on the library's flops), a new 6-max 100bb hand, or the hand on the page's table (its seats,
stacks, antes, straddle, rake, tournament payouts and bounties, and action so far, without your cards, locks or
exploit settings). In a tournament the answers are in prize equity, as the advice's are: preflop the charts' mix as
the advice re-prices it, for the whole range (ICM's risk premium, a bounty's discount: as quick as in chips, ~0.01 s;
six 20bb stacks, 50/30/20 to pay: the big blind facing the button's 2.2bb open jams 16% of hands and calls 41% where
chips have 18% and 63%, the small blind calls the button's jam with 6% instead of 11%, and with 10 PKO on every head
21% and 68%, 15%), after the flop solves that pay prizes, with the tournament's notes for the player to act (prize equity, bubble
factors, who covers whom). The 13x13 grid shows
each hand class's mix (each class's bar as tall as its share of the class's combos in the range), or by **EV** (bb
from here, relative to folding now), **Equity** (against the opponent's range, the rest of the board dealt) or
**EQR** (EV over equity times the pot), shaded red to green; **Breakdown** is the range by made hands and draws (as
the advice's Breakdown tab, both players); **Simple** is the advice's one-size view for the player to act, with their
whole range (heads-up, where they can bet or raise). The toggle beside the views shows the opponent's range instead (its
weights and equity). Hover over an action for the hands that take it; hover over a class (or click to keep it) for
its combos, best first, each with its own mix, EVs and equity. The actions table gives the range's frequency,
combos and the EV of the hands that take each action; click one to play it and the next player's decision appears.
When the betting closes, type the next card(s) or deal random ones; the line above the grid steps back to any earlier
decision (Backspace steps back one). Fast or Accurate picks the solver's speed for live solves (turns and rivers, or
flops the library doesn't hold: a Fast turn ~10 s in the browser), and answers already seen come back at once.
On a library flop's turn where the spot's entry stores one (the 3bet pots' turn files: every CO vs HJ and BB vs SB
flop, 495 of BB vs CO's) and the flop line is one it holds, the turn's first decision and the in-position player's
after a check show at once the flop solve's own turn strategy, marked approximate, while the turn's own solve runs:
the grid by strategy (each combo's stored mix, the card dealt read as its class's played card, the one bet size 75%
of the pot), the range's frequency and combos per action, the opponent's range, each class's combos with their mix,
and the range's EVs where the turn file holds them (format 2; strategies only from a format-1 file, as the site's
are). The turn's solve replaces it when it lands, with every hand's EVs and equities, its own sizes, the Breakdown and
Simple, as before. An action played on the approximate answer stops that solve and goes on at once (the next node's
answer at once first, where the file has it). At Accurate a flop the library doesn't store (a library spot's other
flops, a heads-up pot it holds no spot for, another stack depth: the advice's approximate flops) shows at once the
same way, read off the stored flops most like it for every combo, with the flops, or the spot and depth it borrowed,
named (KhJd6c, HJ vs CO: "the precomputed solves of the most similar flops (KhJd5c, KhJd7c, KhQd6c), by card
substitution"), while the flop's own solve runs (0.45 s in the browser on one thread, where the solve takes 2-4 s);
no EVs or equities until it lands. Fast's flop solve is about as quick as that, so Fast waits for it as before. At
Accurate Study fetches the library files a flop reads as it asks for the flop's first answer, all at once (its entry
and turn files, or the stored flops nearest it); at Fast a library flop's turn files once its answer is in. A turn
after a flop Study solved live (a pot or a flop the library doesn't hold), and a river after a turn, show at once the
street before's solve's own play of it for every combo (the flop solve's turn by card class and hand buckets, one bet
size; the turn solve's river at its river sizes), marked approximate with that solve's iterations and distance from
equilibrium, while the street's own solve runs (0.02-0.07 s in the browser on one thread, where the turn's solve takes 1.6-27 s); no EVs or equities until it lands.
Preflop it shows the charts' mix (no EVs or equities), or with **Solver** a solve of the game's preflop with every
hand's EVs (below). After the flop it browses heads-up, 3-way and 4-way pots (a pot
where 5+ players saw the flop is refused; a 4-way one by the 4-player solve, N-player pots in How it works). In a 3-way pot the toggle has each opponent (their ranges and combos), and the
answer is the 3-player solve's: on the [3-way library](#precomputed-3-way-flop-library)'s flops their checked-to
first decisions come straight from it (CO open, BTN and BB call on Kh7d2c: 1.1 s in the browser on one thread for the
first, the library's file fetched, 0.5 s for the next, most of it the equities), and every other node is solved live
from there, the ranges narrowed through the solves Study answered from (on one thread a flop facing a bet 1.9 s, a
Fast 3-way turn 7 s at ~400 iterations and 0.04% of the pot, a river 0.4 s). A 3-way flop the library doesn't hold is solved live too, for at least 15 s
(`MW_STUDY_FLOP_BUDGET`, stopping once settled; in numpy the advice's Fast 5 s rarely settled one: natively on a
loaded machine 12-23 iterations, 17-40% of the pot, where 15 s ran 51-77, 1.1-1.9%; with the compiled multiway core
5 s run 111-125, 0.7-1.05%). While it solves, Study shows at once the decisions the library would store (first to
act, checked to once or twice) read off the spot's stored flops most like it, as a heads-up flop the library
doesn't store (`MWLibrary.near`, `Advisor._study_mw_near`: the three nearest of the same pairing and suit pattern,
each carried over by card substitution, a combo's mix its images' of the same hand class averaged, not the heads-up
learned correction), the flops it read named, the mix alone (no EVs or equities until the solve lands); asked for at
Fast too (a heads-up flop's Fast solve is as quick as the reading). Leaving each of the 740 stored flops out in turn
and reading it off the others (`scripts/bench_mw_approx.py`), those decisions give up 0.14% of the pot on average
by the flop's own EVs (median 0.10, 90th percentile 0.32; first to act 0.06, checked to once 0.15, twice 0.22; the
worst low paired and connected boards 1.3-1.9), their range frequencies 7 points off (total variation); 16 of them (2%)
have no stored flop near enough (6 rank steps), and nothing shows until the solve lands. In the browser on one
thread (`scripts/bench_browser.mjs` study-mw-near, Kh8d3c) it shows in 0.7 s, the solve in 13 s. Equity (and EQR)
on every street, as a share of the pot at showdown against the two
other ranges with the rest of the board dealt (`multiway_solver.showdown_equities`): on the turn every river with
the exact three-way card removal (as the river's showdowns, ~0.1 s natively); on the flop all 1176 turn-and-river
runouts with the solver's pairwise correction for the opponents' shared cards, scaled so each runout's shares add
to the pot (~0.25 s natively; against the exact count over every runout of Kd7c2h and Jh8h4c, 0.2-0.3 points off on
average and 1.5 at most, where sampling every 168th runout exactly was 1.6-1.9 and 7-9). The Breakdown's equities
are the same numbers. A 4-way pot has equity on the river only (the pairwise count was 2.5-4 points low on strong
hands); before it the views are Strategy, EV and Breakdown. Before the river the 3-way answer says what is per hand:
each hand's mix and EVs on the street are its own, while the solve plays the streets after it by hand buckets (16 a
player on each card in the library, on the turn and at Accurate; 12 in a Fast flop solve), so a hand's EV counts its later play as its
bucket's. Lines the range takes under 1% of the time are marked rough, as heads-up. A 3-way spot the solver can't take (a bet
bigger than the shortest stack can call, a solve still over 10% of the pot from equilibrium) is an error, not the
betting rules. The panel is `gto/static/study.js`, loaded when first opened; it asks the solver with `"study": true`
(see Use from the shell).

**Preflop: Charts or Solver.** The **Charts | Solver** toggle over the grid at a preflop decision picks what Study
shows there (remembered). **Charts**, the default, is the hand-written charts' mix the advice plays, without EVs.
**Solver** is a solve of the game's whole preflop (`gto/pf_study.py`: `pf_game`'s tree for the table, its flops valued
by the flop library as the other games' charts are; heads-up the small blind may limp, 7-handed at most two see a
flop): at every preflop decision each hand class's mix with each action's EV (bb from here, relative to folding now)
and the range's, so the EV view, a class's EVs, the actions table's EV column and Build's EV given up work preflop
too. Its EVs are the solve's own, at its own equilibrium, and go with its mix, which is not the charts': the solve
opens tighter than the published solves the charts match (6-max 100bb unraked, UTG to SB 14.8 / 18.3 / 24.8 / 37.0 /
52.1% against the charts' 18.7 / 22.9 / 30.0 / 45.1 / 44.7; the big blind against a button open 40.8 / 47.7 / 11.5
fold / call / 3bet unraked and 56.2 / 31.9 / 12.0 at 5% capped at 3bb, the charts 58.2 / 30.2 / 11.6 at 5% capped at
4), so the advice keeps the charts (Preflop equilibrium in How it works), and the line under the grid says which
it is. The charts' line re-valued wasn't offered instead: valued inside the same game the charts' mix isn't an
equilibrium (each seat gains 0.15 bb/hand on average by best-responding to the others' charts, 0.87 summed over the
six), so EVs read off them contradict their own mix (a hand mixing two actions has them 0.86 bb apart in median, and
junk 4bets show +3.5 bb). The range's and each action's EV weigh a class by its combos times its chance against the
others' ranges (card removal: 0.555 bb for the big blind's range against a button open, 0.544 by combos alone);
frequencies and combos are the mix's alone.

The standard games answer at once: 6-max 100bb unraked and 5% capped at 3bb (the page's "5-3" preset) are stored
(`gto/static/pf/`, 84 and 89 KB: 1,500 iterations, 1.07 and 0.52 mbb/hand a seat from equilibrium;
`scripts/build_pf_study.py`), in the browser loaded in 0.4 s, a game's first node 0.4 s and each one after ~1 ms.
Any other game the table sets up (2-6 players in the browser, 7 with the local server; equal stacks of 10-250bb, an
ante, a straddle, rake and its cap; not tournaments, refused with the reason) offers **Solve** with its time: the
worker solves it in 4-second chunks, each going on from the last, and asks the node in view again after each, so its
mix and EVs converge in view, with a bar, the iterations and the distance from equilibrium, until it is within 1.5
mbb/hand a seat (a 2-4 handed tree to 2,000 iterations, seconds in all) or **Stop**; its nodes then answer as the
stored ones do (three games held). A finished solve is also kept in the browser (IndexedDB: the six latest, from 8 KB
3-handed to ~0.4 MB at 6-max, until the solver's code changes) and handed back to the worker when its game comes up
again after a reload, so a game solved once answers at once ("solved earlier in this browser"; in Pyodide packing a
6-max solve takes 0.3 s and taking it back 0.5 s with its first node).

**Preflop solve…** in the ⋯ menu (`/gto-solver/preflop/`) is Study with the game set in a form over the line instead
of a spot: players (2-6, 7 with the local server), stack, ante, straddle, rake and its cap, and the tree's raise
sizes (the open and the small blind's in bb, 3bets in position and from a blind as multiples of the open, the 4bet
of the 3bet; `pf_study.SIZE_LIMITS`), starting from the table's game; any change goes to that game's first decision
with the Solver source (the standard games' answers at once, any other offered to solve; a solve running for another
game stops). "Preflop solve: a game you set" in Study's start menu is the same. Measured in Pyodide on one thread (`pf_game` 2.3x quicker there than before:
`Game._masses` takes each seat's card-removal product once a node instead of once a terminal, and einsum stands in
for BLAS-less matmul): heads-up ~1 s, 3-handed 7.4 s (2,000 iterations, 0.69 mbb/hand a seat), 4-handed 42 s (1.23),
6-max 0.34 s an iteration and ~5 minutes to 1.5 mbb (~800 iterations; natively 0.12 s, under 2 minutes), an
exploitability check every 100 iterations 3.6 s (run as the worker does, in Pyodide under Node on one thread, the 6-max
game raked 5% capped at 4bb stopped at 800 iterations, 1.21 mbb, in 4.4 minutes; natively the stored games' settings
from scratch at 800 and 700 iterations in 114 and 95 s). The average strategy weighs iteration t by t³ (`pf_study.AVG_POWER`;
it was linear): the 6-max game from scratch is 7.4 / 2.3 / 1.6 / 1.3 mbb/hand a seat at 250 / 500 / 750 / 1,000
iterations, against 18.7 / 5.8 / 3.2 / 2.2 averaged linearly (5% capped at 3bb 10.6 / 2.8 / 1.3 / 0.85 against 25.0
/ 6.9 / 3.4 / 2.1; 40bb deep 7.0 / 2.5 / 1.8 / 1.5 against 13.8 / 5.2 / 3.1 / 2.3); t⁵ was a little quicker still,
DCFR's own discount of negative regrets (β 0.5) slower (`scripts/pf_study_convergence.py`). Against the same solve run
on to 2,000 iterations (1.0 mbb), at 750 a class's EV of an action it plays (5% or more of the class) is off by
0.001-0.008 bb on average at the first decisions and facing an open, 0.03-0.04 facing a 3bet (the worst class 0.45
bb), the range's EV by 0.015 bb at most and the actions' frequencies by under a point; at 500 iterations (2.3 mbb)
0.04-0.06 facing a 3bet, so the target went from 2 mbb to 1.5. The stored games, solved again this way, are nearer
equilibrium (1.07 and 0.52 mbb at 1,500 iterations, were 1.34 and 0.94) and a quarter the size (316 KB before: the
barely reached lines' late mixes are mostly pure and pack smaller); the two averagings' stored games agree at the
six nodes above within 0.3 points of each action's frequency and 0.001-0.013 bb in a class's EV on average. Starting a game from the stored 100bb solve didn't
help (linear averaging, the raked game from the unraked solve: 13.8 mbb at the start, then 4.0-12.0 after 700
iterations against 3.4 from scratch at 750, over four ways to seed the regrets and averages), so each starts from
scratch. A line the solve rarely or
never plays is flagged under the grid (an opponent's range under a combo, a range it never brings there, or a mix
giving up 0.05 bb a hand or more to its best actions at that node, `regret_bb`): the average strategy is barely
solved there (the stored 6-max's nodes reached by 1% of hands or more give up at most 0.003 bb a hand, median 0;
those reached by 0.0001-0.01% of hands 0.01 in median, 0.06 averaged linearly; lines almost never played far more). A player brought there by a 0% move is taken to make it with
every hand it held, so its range is as before the move. Checked (`tests/test_pf_study.py`): wherever two players are
left (every node heads-up, 829 of the stored 6-max's) each class's EVs equal a direct computation over pure class
pairs with card-removal weights from the 1326 combos enumerated there (no reaches, masses or counterfactual values)
to the answer's rounding, 0.0005 bb; and wherever three are left (every such node of a 3-handed solve, and all 1,079
of the stored 6-max's that each seat reaches with 0.01% of its hands) a direct sum over class triples at every
terminal below gives the same EVs to 0.0005 bb: each player's action probabilities on the way, a fold's or a heads-up
showdown's payoff, and a three-way showdown's share of the pot (corr x equity x equity, normalised over the three
ranges reaching that terminal, times the seat's realization), card removal from the enumerated combos. Folding is 0;
at the first decision the range's EV is the game's own value (`Game.values`, over the chance the others' cards don't
collide). Not checked independently: nodes four or more players are left at (a sum over four to six classes' tuples is
too big to run; the arithmetic is the same), and the game's model itself. Its card removal is pairwise, each
opponent's combos off the player's cards but not off each other's: counted exactly over combo triples under the same
strategies, the EVs at the stored 6-max's 25 most played 3-way nodes (BTN, CO, HJ or UTG first in with the blinds
behind, an open and a call facing the big blind, a squeeze) move by 0.001-0.09 bb on average (median 0.02, the worst
class 0.9 bb), and by more where the ranges are narrow (after a jam and a call, lines played well under 0.01% of the
time: the pairwise count weighs AA against two KK hands as 216 combo triples where there are 36). Three-way showdowns'
shares are a measured correction to pairwise equities (`MW_CORR`, `MW_REALIZE`), not an equity count.

The **All hands** menu beside the views filters the range to one made hand or draw, like GTO Wizard's filters (the
Breakdown's categories: set, top pair by kicker, middle pair, no made hand, flush draw, open-ended, …, each with its
share of the range): the grid then shows only those hands (a class as tall as its combos in the category), the
actions table and the range's line give their frequencies, combos, EV and equity, and a hand's detail lists only
its combos in the category. The filter stays on as you play through the hand (a draw's filter shows every hand on
the river, where there are none); **✕ All hands** clears it. In the Breakdown, a row of the player to act offers
**Show these hands in the grid**. Each combo of the answer carries its category (`"categories"`: the indices are into
its made-hand and draw lists).

When the flop's or turn's betting closes, **Turn report** / **River report** beside the next card's deal shows the
next street's strategy on every card at once (see **Turn and river reports** above): a table, or **Cards**, a
13x4 grid of the cards (suits down, ranks across; the board's blank), each card filled by the first player's mix
there with the share that bets in its corner, and a tab for the second player's after a check. It is read off the
stored turns of a library flop that has them, else one solve of the street that closed at the speed picked (in hand
already where Study's own answer solved it: the river's at once from the turn solve Study answered the turn's last
decision with; else a button solves it, Fast ~10 s in the browser): on the turn each card plays the flop solve's turn
tree, one bet size (75%) and a rank's cards alike apart from a flush suit's (Fast's turn by hand buckets); on the
river the turn solve's river sizes (33/75/125%; Fast on one thread 33/125%). Click a card to deal it; that street then
solves on its own, with its full menu of sizes. A solve cut short (3% of the pot or more from equilibrium: a busy
machine runs fewer iterations) says so above the cards, as a live answer does in the line under its grid.

**✎ Build** (or the B key) in Study turns the decision into a range-building drill, like GTO Wizard's range builder:
the grid shows the acting player's range with every hand at the passive action (fold, else check or call), and you
paint the strategy yourself. Pick an action (keys 1-9) and a share (100/75/50/25%: the rest of the hand's mix keeps
its proportions, so painting twice makes a mix), then click or drag over hands; **Fill all** paints the whole range,
and the range's frequencies update as you paint. **Check my range** scores it against the solver: accuracy (one minus
the share of the range, by combos, that you played differently), the EV given up (bb a combo and % of the pot, at
each action's EV against the solver's opponent; not preflop, where the charts have no EVs), each action's share
yours and the solver's, and the biggest differences. Each hand then shows the solver's mix above yours, outlined
where they differ by 20% or more (red at 50%); a hand's detail compares the two mixes with each action's EV. **Try
again** keeps your paint, **New flop** deals the same spot on a random flop, **Show the solution** leaves the drill.
Any decision Study reaches can be built, preflop included (the charts' mix).

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
answer is used if you take that line, so the wait for it goes into your thinking time. A look-ahead your move made
useless is stopped: by a local server, and in the browser where the page and its solver share a stop flag (a
cross-origin isolated page, as the site is under its service worker), within ~0.1 s, its solve kept to go on from
there if asked again. A browser that can't stop one looks ahead only at Fast, where a wrong guess costs at most one
quick answer's ~2 s. Until your first decision a dashed card says where
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
budget for an unsettled street; 3-way pots solved on every street in 1.8 s (`advisor.MW_QUICK_BUDGET`; flops and
turns in a smaller game, `MW_QUICK_FLOP`: 6 hand buckets, the flop betting half the pot; `MW_QUICK_TURN`: 8), a 3-way
flop's first decision in the commonest 3-way pots read from the [3-way library](#precomputed-3-way-flop-library) at
once; 4-way turns solved, a 4-way flop left to the betting rules in the browser (`Advisor.mw4_quick_flop`: its tables
alone took 1.6 s, then 15 iterations, 9.4% of the pot from equilibrium; natively 41, 1.07%); measured in Pyodide, `BENCH_STATE='{"quick":true}'
BENCH_BUDGETS='{"fast":[1.5,0.3]}' scripts/bench_browser.mjs`, medians of 3 on an idle machine: a turn 1.9 s, 26
iterations, 6.4-6.9% of the pot from equilibrium (was 22-23, 8.1-8.9%) where Fast's 8 s settle it to 0.12%; a river
(budgets `[0.8,0.3]`) 1.4 s, 412 iterations, 0.04%, its turn narrowed by the rules (was 1.8 s, 244 iterations, 0.08%,
with a 1 s turn solve trusted 0%, and a 272 MB wasm heap, now 76 MB); a 3-way river 1.8 s,
14 iterations, 2.6%; a flop from the library at once; with the compiled multiway core, Kh8d3c, CO opens, BTN and BB
call: a 3-way flop first to act 1.8 s, 114 iterations, 0.43%, checked to twice 1.4 s, 200, 0.07%; a 3-way turn 1.8 s,
87, 0.58%; a 4-way turn 1.3 s, 100, 0.04%). While you decide, the lines you're likeliest to take are solved ahead: your
hand's mix blended half and half with what you usually do (fold, check or call, bet or raise; facing a bet or not;
kept in this browser as `gto-practice-habits`), up to two lines at least 20% likely, a bet at the solver's size, never
a fold; the solver's own Fast (8 s) solves 3-way flops and turns in a smaller game (`advisor.MW_FAST_STREETS`), and a multiway spot left to the betting rules (5+ players before the river, a quick 4-way flop in the browser, a solve still over 10% of the pot from equilibrium) is
graded by the EV your move gives up against the rules' best, as a share of the pot (their frequencies are all or
nothing: within 3% Good, within 10% an Inaccuracy, more a Mistake; `HEUR_OK`, `HEUR_MEH`), the card listing the rules'
EV for each action and saying it's a heuristic, not a solve; **Details** opens why and each action's EV (the choice is remembered), and from there **See how the
solver plays your whole range** opens your range at your last decision under the coach (never on your turn), headed
"Your range · flop · BTN vs BB": the advice panel's Grid (every hand class's mix, yours outlined; hover or tap a hand
for its mix), Breakdown and Equity, without the rows the verdict card already gives (a decision whose solve the page
has moved on from, or one from the lists below, has none). At the end the
opponents' cards are shown at showdown with your result, the verdict on your last decision and the hand's
decisions with the EV each gave up, and **Next hand** (N) takes the action buttons' place (N does nothing mid-hand). The score
(hands, how often your action was in the solver's strategy, average EV given up per decision, chips won)
is kept in the browser, with your last 60 decisions; on a wide screen the eight most recent are listed under it,
newest first, each opening its spot like a mistake below. **Start over** under the score (or **Reset practice…** in
the ⋯ menu) clears the score, those decisions and the table's chips and deals a new hand at Hand 1, once you confirm;
your seat, the drill and the setup stay. **Review mistakes** under the score (with how many) lists the six of them that gave up
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

**Tables** (1-4, under the spots) plays several tables at once, as on a multi-tabling site; 1 is the table above,
unchanged. With 2-4 a view of their own takes the place of the page's table and coach (`multitable.js`, loaded the
first time; the page's own hand waits as it was): the tables in a grid (two columns, three for 3 tables on a screen
1300px or wider), each a compact felt with its own hand, seats, stacks, board, pot and your cards, your made hand and
the verdict on your last decision there, and what it's doing in its header ("Your turn", "BTN is thinking…",
"Solving your spot…", "Dealing an instant flop…", "Next hand…"). The table that needs you is outlined in front and
gets the action dock under the grid (its buttons and keys as at one table: F, C or X, 1-9 the sizes, R a size of
your own, A all-in); tables waiting for you queue in the order they started waiting, each header showing its place.
After each move the table that has waited longest comes forward (the one you acted at, if it's waiting again at once,
goes behind the others); for 0.35 s after the table in front changes by itself, keys and buttons are ignored, so a
double press doesn't land at the next table. **S** sends the table in front to the back of the queue, a click on a
table (or its tab) brings it forward until you act there, and **N** deals the finished tables at once (otherwise
1.6 s after a hand ends). **Spots**: **Instant flops**, the default, plays preflop for you and deals each table
until a heads-up flop the [flop library](#precomputed-flop-library) holds for its preflop line, with you in it, so
preflop and the flop answer at once and only turns (~2 s at Fast) and rivers (~1 s) are solved. The preflop it plays
for you (and for the drills) is drawn in the page, every seat's move from the charts' mix at its node as the solver
draws it: the page asks for a node's chart once a session (Study's preflop answer: each hand class's mix) instead of
asking for each seat's move every hand (a node it can't answer, or a hand its chart leaves out, is asked seat by seat
as before); it needs the
standard game (6 players, 100bb, no ante, straddle, rake, tournament or advanced settings) and is Any hand otherwise,
which the note under the pickers says. **Any hand**, **Raised pot**, **3-bet pot** and **Blind vs blind** are as at one
table; **Seat** deals a random seat each hand or one position. The setup (players, stacks, ante, rake, payouts) is
the page's, changed at 1 table. Going from one table to several first stops what the one table had asked and no
longer needs (your spot's advice, an opponent's move, a look-ahead: the local server's solve, the browser's requests
where the page shares a stop flag with its worker), so the tables' first requests don't wait behind it. Solves: the
solver answers one request at a time (the browser's worker, or the local server's lock), so one is in flight and
the next is picked by what it unblocks: instant answers first (a preflop
chart, a library flop, a street already solved, such as an opponent's reply heads-up), then the other tables'
short solves (your spot, an opponent's move: ~2 s at Fast), then the look-ahead at the table in front (your likeliest
moves played on to the next request, as at one table, the quickest first), then long solves (Accurate's), then the
other waiting tables' look-aheads; within a group your spot before an opponent's move, then the longest waiting. A
table waiting on a solve shows a spinner and never holds up the others. A look-ahead or a long solve gives way to an
instant or short request where it can be stopped (the local server; the browser where the page shares a stop flag
with its worker), its solve kept to go on from there; one that no table needs any more (its hand ended, you took
another line, fewer tables, Start over) is stopped the same way. At Accurate (up to a minute a solve in the
browser, 30 s on the local server) the note says tables wait longer and Fast suits several. Every decision is
graded as at one table, into one session score for all tables (hands, accuracy, bb lost per decision, won), a row
per table (its hands, accuracy, bb lost per decision, won and last hand), the verdict card on your last decision
("Table 3 · last decision", with Details and the AI buttons), the recent decisions of every table ("T2 #5 turn") and
**Review mistakes** across all tables (from the last 60 decisions); **Start over** clears the session and deals every
table again. Each table's hand and the session are kept in this browser over a reload (`gto-mt`, `gto-mt-score`; the
spots, seat and table count as `gto-mt-drill`, `gto-mt-seat`, `gto-practice-tables`). On a phone one table shows at
a time under a strip of tabs, one per table (lit with its place in the queue when it waits for you, a spinner while
it's solved), a tap switching table, and the dock stays at the bottom of the screen. Measured on the local server
(one thread, Fast) with a player acting every 2 s on 4 tables of Instant flops: 22 hands and 58 decisions in 126 s,
61 of the 62 spots answered before their table came up (the other in 0.07 s), never a moment with no table to act
at; acting every 0.6 s on 4 tables of Any hand: 114 hands and 144 decisions in 112 s, 140 of 147 spots answered
ahead, six waits of 0.2-1.6 s (live flops) and a turn of 6.3 s (queued behind the other tables' solves), no table to
act at 5% of the time. Drawing preflop from the charts cut the requests: acting every 2 s on 4 tables of Instant
flops (from one table switched to four), the page asked the local server 176 times in 120 s, 31 of them preflop (a
chart for each node met; 570 seat moves drawn from them), where it had asked 755 times, 564 of them a preflop seat's
move; 61 of 64 spots answered before their table came up (58 before), the longest wait 0.62 s (1.91). In the browser the quick answers take about as long (they're budgeted in time: a turn 1.9 s
in Pyodide, above). `MultiTable.metrics()` in the console gives a session's numbers.

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
(see Tournaments); `--bounties "pko 10"` (`"ko 10 SB=30 field=12"`, `"mystery 25"`) a bounty tournament (see Bounty
tournaments).
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
see Tournaments below); `"bounties"` makes it a bounty tournament (KO, PKO or mystery: see Bounty tournaments
below; with no payouts, chip EV). Positions are the 6-max names `UTG HJ CO BTN SB BB`.
Optional `"sizes"`, `"ranges"` and `"optimize_sizes"` change the postflop game, `"locks"` the preflop one too (see Custom sizes and ranges below).
Optional `"one_size"` (`"all"`, `"full"`, a size or `"list"`) answers with the one-size view instead of the advice (see One bet size below).

Response: per-action `freq` (the mixed strategy for hero's exact hand), `ev_bb`
(relative to folding now), `amount_bb`, `"rough_ev": true` heads-up where hero's range takes the action under 1% of
the time (`advisor.RARE_BRANCH`: the opponent's answer to it is barely solved, so its EV is rough and usually lower
than exact; the page shows it, and Practice's EV given up for such a move, with a "≈"; docs/cfr_research.md, EV gaps
on live turns), a `recommended` action (highest
frequency, or highest EV when the solve is 3%+ of the pot from equilibrium and that
EV beats the top of the mix by over 1% of the pot), a `sampled` action (drawn from the mix, use this if you want to
actually play unexploitably), the range-wide frequencies, the villain's inferred
range, and solver stats. With `"breakdown": true` a postflop answer also carries `breakdown` (`gto/breakdown.py`):
`actions`, `pot`, `hero` and `villains`, each with `made` and `draws` categories (`c` combos, `eq` equity, the
hero's `f` per action and `ev`), the range's `eq`, `ev`, `nuts` (share at `nut_eq`, 0.8, or more) and `curve`
(equities sorted high to low at every 2% of the range; heads-up only), and the hero's own categories (`mine`);
`null` where the betting rules answered. `"library_only": true` (the session analyzer's instant scope) answers a
flop only from the precomputed library and returns `{"skipped": …}` for any other postflop spot instead of solving it.

`"study": true` (the page's Study mode, `Advisor.study`) answers for whoever acts next on the hand as recorded, with
their whole range and no hero, at equilibrium (`hero`, `hole`, `opponents`, `locks`, `jam` and `optimize_sizes` are
ignored; a tournament's `payouts`, `field` and `bounties` count, as in the advice, with `tournament` and its notes
for the player to act):
`{"study": "node", "actor", "opponent", "street", "pot_bb", "to_call_bb", "stack_bb", "actions", "grid", "combos",
…}` with per action the range's `freq`, `combos` and `ev_bb` (of the hands that take it; `"rough_ev": true` heads-up
where the range takes it under `advisor.RARE_BRANCH`, as the advice marks it, and Study shows that action's EVs with
a "≈" and the tip: the range's, a class's mix, its combos' and the EV view's cells on it), per hand class in `grid`
its weight `w`, mix `f`, per-action EVs `ev`, EV `evm` and equity `eq`, per combo in `combos` the same as
`[cards, w, f, ev, eq, made, draw]` (the last two indices into `categories`' `made` and `draws` lists, `[id, label]`
each; no draws on the river), the opponent's range in `opponent_grid`, and the solver's stats (`breakdown` too with
`"breakdown": true`). Where nobody acts it says
what the hand needs: `{"study": "deal", "street": "turn", "allin": false}` (the next street's cards),
`{"study": "over", "street", "winner"}` (everyone else folded) or `{"study": "over", "street": "river", "showdown": true}`.
Preflop it is the charts' mix (no EVs), or with `"preflop_solve": true` the preflop solve of the request's game
(`gto/pf_study.py`, Study's Solver source): `{"study": "node", "street": "preflop", "source": "solver", …}` with each
class's `ev` and each action's `ev_bb`, `"opponents"` (the players in who have acted, their ranges), `"pf_solve"`
(`key`, `label`, `iterations`, `done`, `stored`, `mbb`, `target_mbb`), `"regret_bb"` and `"rough"`, where a solve
is in hand (stored, or run here); else `{"study": "pf_unsolved", "game", "key", "estimate_s"}`, or `{"study":
"pf_none", "reason"}` for a game it doesn't take. `"preflop_solve": {"run": seconds}` runs that game's solve 0.5-30 s
more from where it was and answers `{"study": "pf_progress", "pf_solve"}`. `"pf_sizes"` (`open`, `sb_open`,
`ip_3bet`, `blind_3bet`, `four_x`) sets the tree's raise sizes (the Preflop solve's form sends them).
`"preflop_solve": {"export": true}` packs a finished solve run here (`{"study": "pf_export", "key", "iterations",
"mbb", "blob"}`, base64; `"blob"` null for none, a stored game or one not finished), and `{"load": blob,
"iterations": n}` takes it back before answering the node (where none is held; `"pf_solve"` then has `"cached":
true`, and a blob of another tree is ignored). Postflop it browses heads-up, 3-way and 4-way pots. A multiway node has
`"multiway": true`, `"opponent"` naming the others ("BB and BTN"), `"opponents": [{"seat", "combos", "eq", "grid"}]` in acting order in
place of `opponent_grid`, and `eq` (`range_eq`) a share of the pot at showdown (3-way on every street, the rest of
the board dealt; 4-way on the river, else null); `"rough_ev": true` on an action the range takes under 1% of the time. With `"runouts": true` on a heads-up hand whose flop
or turn betting closed (the next card not dealt), it returns the next street's first decision on every card, from a
library flop's stored turns where they hold the line, else solving that street live: `{"study": "runouts", "street":
"turn" | "river", "first": {"actor", "actions", "avg", "cards": [{"card", "cards", "f", "w"}]}, "after_check": …,
"seats": [oop, ip], "pot_bb", "cards": [{"card", "p", "eq": [oop, ip], "ev": [oop, ip] | null}], "source": "store" |
"solve", "solver", "notes", "seconds"}` (`cards` in a part: the cards that share a class's mix `f`, `w` its player's
range there, `avg` the range's mix over every card; `after_check` the same for the second player after a check, or
null; the top-level `cards`: each card's chance given both ranges, range equities and EVs, `ev` null from the stored
turns; from the stored turns also `"lines"`, the flop lines they hold as Study's actions with how often the flop
solve takes each, and `"line"`, this one's). `"runouts": "held"` answers from what's in hand only (the stored turns,
a solve cached already) and else `{"study": "runouts", "skipped": "not solved yet", "lines"?}`, solving nothing.
`Advisor.instant` with `"study": true` (the browser's worker posts it before Study's own answer) reads a library
flop's turn off its stored turn file where it holds the line, solving nothing: the same node answer with
`"approximate": {"source": "store", "label", "iterations"}`, the stored tree's actions (check, bet 75%, all-in), `ev_bb`
and `range_ev_bb` from a format-2 file (else null), per hand class and combo the mix only (`ev`, `evm`, `eq` and each
combo's EVs and equity null, `range_eq` too) and `"solver": {"by": "the flop library", "iterations",
"exploitability_pct"}` (the flop solve's); on a library spot's flop the library doesn't store (or a heads-up pot
read off the spot nearest it, `instant_from_near`'s), the flop's node read off the stored flops most like it, with
`"approximate": {"source": "near", "label", "iterations": null}` (the label names the flops, or the spot and the depth
borrowed), the tree's sizes, no EVs and `"solver": {"by": "the flop library", "iterations": null,
"exploitability_pct": null}`; on a turn after a flop solved live, or a river after a turn (`instant_from_prev`'s read
of that solve), `"approximate": {"source": "prev", "label", "iterations"}` (the label as the advice's: "the flop
solve's view of the turn (by card class and hand buckets, one bet size)"), that solve's actions and sizes at the
hand's pot, no EVs and `"solver": {"by": "the flop's solve" | "the turn's solve", "iterations",
"exploitability_pct"}`; `null` anywhere else (facing a turn bet on a library flop, a flop without a turn file, a line
the stored flops or the solve take less than 10% of the time, a street before not solved yet, a tournament) and once
the street's own solve is in hand (settled).

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
* 3- and 4-way pots after the flop are solved in prize equity too: every outcome, side pots included, pays each
  player the chip equivalent of the prizes they end with (multiway_solver MWConfig.icm, icm.MWPayoffs; in the
  compiled core as well). A pot of three or four has more outcomes than a share of it can weigh (who wins when a
  player loses, who takes a side pot), so each terminal's value is kept linear in the solver's shares by pricing
  every order of the live hands and averaging: what that leaves out is the bystander's part of a loss, taken as
  even among the others. A player who folds on a later street is paid the pot as it stood. Winner-take-all is
  exactly chips (tested). Against the exact prize game (scripts/bench_mw_icm.py: a 3-way river, Kh7d2c9s4h, 60
  combos a player, every deal priced on its own: who wins, every seat's stack after, Malmuth-Harville over the
  table; 600 iterations, 23 in the pot): on the bubble (4 left, 50/30/20, the small blind's 8bb folded, three
  25bb stacks) the chip solve's strategies give up 3.3-5.5% of the pot to best responses in prizes and the prize
  solve's 0.002-0.015%, its values within 0.02% of the pot of the exact game's; with the big blind the 12bb short
  stack, 0.8-2.2% against 0.004-0.33% (the short stack shoves 13% of the time instead of 21%; values within 0.6%
  of the pot, where the bystander's part counts); 6 left at a final table (40/25/15/10/6/4) 1.3-2.1% against
  0.009-0.025%. The 3-way library (solved in chips) is skipped, so a tournament's 3-way flop solves live. Pots
  the solves don't play (five or more players, a solve that didn't settle) get the betting rules, in chips; the
  notes say which.

### Bounty tournaments (KO, PKO, mystery)

The **Bounties** row under Payouts (saved with the hand, kept across new hands and players, carried by share links
and the copied hand) picks the format and the bounties, in the payouts' units: **KO** (the whole bounty goes to
whoever busts the player), **PKO** (progressive knockout: half is paid at once, half is added to the eliminator's
own bounty) or **mystery** (each bust draws an envelope: the average envelope left, or the money left in
envelopes and how many). KO and PKO take one bounty for every seat ("each"), seats with their own ("per seat…",
one box a seat), and the other tables' average bounty next to the field (the table's average if left empty). The
request's `"bounties"` is `{"format": "ko" | "pko", "seats": {seat: bounty} (or one number for all, or a list in
seat order), "field": average}` or `{"format": "mystery", "envelope": average}` (or `"pool"` and `"left"`);
`"bounties"` with nothing at stake (every bounty 0) answers exactly as without (tested: the JSON is identical on
calls, shoves, opens, defence and a river solve).

The rules: a bounty pays only when the winner covers the busted player (they had at least as many chips at the
start of the hand), so a short stack that wins gets chips only. A split pot busts nobody. With side pots won by
different players, each bust's bounty goes to whoever wins the pot that takes the player's last chips: the top
layer their chips reach, whose winner put in at least as much and so covered them.

**The model** (gto/icm.py `Bounties`, `Model._bounty_setup`, `Model.value`) is the proportional model: every player
owns their chip share of the bounty money in play, so a chip is worth its prize equity plus the bounty pool over
the chips in play, and a bust moves the cash it pays out of that pool into the winner's pocket. It is GTO
Wizard's "bounty power" (chips over prize pool plus bounty pool,
[The Theory of Progressive Knockout Tournaments](https://blog.gtowizard.com/the-theory-of-progressive-knockout-tournaments/))
and ICMIZER's "Proportional" model (a player's bounty equity is their stack over the chips in play times the
bounty money left,
[ICMIZER 3.15: calculation model selection for KO and PKO](https://blog.icmizer.com/icmizer-3-15-0-calculation-model-selection-for-ko-and-pko-tournaments-in-icmizer-and-sng-coach/)).
Each outcome is worth, in chip equivalents,

    k × [ΔICM + (end × (P − C) − start × P) / T + cash won],    k = 1 / (prizes / chips + P / T)

with P the bounty pool, C the cash the hand's busts pay out of it, T the chips in play (the table's and every
other player's), start and end the stacks, and k the chips a unit of money is worth (shown as "1 is worth X bb").
A bust's own part, to the winner, comes to k × (1 − s') × cash, s' the winner's chip share after the hand: it owned
that share of the busted player's bounty already. Cash is the whole bounty in a KO, the envelope in a mystery
bounty, and half the bounty in a PKO. **The PKO's added half** stays in the pool on the winner's head, worth the
winner's chip share of it (as every chip's share of the pool): B/2 at once plus s' × B/2 for the added half, less
the s' × B it owned of the busted bounty, = (1 − s') × B/2 (the proportional model GTO Wizard and ICMIZER both
use; tested against the hand formula). In a winner-take-all every outcome is its chips plus that bonus exactly.
**With no payouts** chips are converted to money by taking the prizes still to be paid as equal to the bounties in
play (a buy-in split evenly between the prize pool and the bounties, the usual 50/50 KO and PKO), chip EV
otherwise; the answer's note says so. With payouts the prizes are ICM, as above.

It reaches everywhere ICM does: push/fold (the shove's and each caller's values of every outcome, so a covering
stack calls and shoves wider, and a short stack's own big bounty makes the stacks that cover it call wider),
calling an all-in (the equity it needs, against the chip price: "a bounty discount"), the preflop risk premium
(negative with a bounty: calls and 3bets re-priced, opens widen by at most 1.5x the chart's, the best against
what continues added), heads-up solves after the flop (icm.HUPayoffs: the loser all in busts, a split doesn't)
and 3- and 4-way solves (icm.MWPayoffs: each order of the live hands busts whoever ends with nothing, its bounty to
that layer's winner; the compiled core reads the same weights, so it needed no change). Answers carry
`tournament.bounties`: `format`, `model` ("proportional"), `bb_per_unit` (k), `pool`, every seat's `bounties`,
`you_bust` (each opponent hero covers: what busting them is worth to hero in `bb` and in money (`value`), the `cash`
it pays, and `in_hand`: whether hero's call or shove here can win it), `bust_you` (each opponent who covers hero:
what hero's bust is worth to them, bb) and a `note` naming the model.

Worked numbers (6-max, no payouts, UTG-BTN 30bb, the small blind 20bb jams into the big blind's 40bb, who covers):

| bounties | 1 unit | busting SB is worth | BB's call needs | BB calls | SB jams | A2o |
|---|---|---|---|---|---|---|
| none | | | 47.5% (chips) | 21% of hands | 39% | folds |
| PKO 10 on every head (pool 60) | 1.5bb | 5 cash, 5.0bb | 42.2% = 19 / (40 + 5) | 26% | 31% | calls 56% (42.5% equity) |
| KO 10 on every head | 1.5bb | 10, 10.0bb | 38.0% | 35% | 30% | calls |
| mystery, envelope 30 (pool 150) | 0.6bb | 30, 12.0bb | 36.5% | 46% | 33% | calls |
| KO 30 on SB, 10 on the others | 1.125bb | 30, 22.5bb | 30.4% = 19 / 62.5 | 97% | 53% | calls |

(The small blind's jam range tightens as the big blind calls wider, and widens again once the big blind calls
nearly everything: then a jam needs only its equity against any two.) The small blind with 40bb covering a 12bb big blind jams 53% of hands in chips, 75% with 10 PKO on every head, 96%
with 10 KO, and every hand with 30 KO on the big blind. KO winner-take-all (tests/test_bounty.py): the button's 10bb
jam, the big blind (30bb) calling 9 more, 300 in prizes and 50 on each of three heads: a unit is worth 0.2
chips, busting the button adds 0.2 × (1 − 40.5/90) × 50 = 5.5 to the chips won, so the call needs 9 / (20.5 + 5.5)
= 34.6% against 43.9% in chips (PKO 9 / 23.25 = 38.7%). On a river (Ks7d2c9h4s) where the button jams 22bb all in
into a big blind who covers, with 40 KO on the button the big blind's queens go from a fold (−2.94bb) to a call
(+1.27bb: their 37.7% equity times the 11.2bb bust) and the big blind calls 36% of its range instead of 25%. 3-way
(4 left, 50/30/20 paid; BTN 12bb jams the river, SB 30bb with queens, BB 30bb behind): with 80 KO on the button
the small blind raises all in 52% of the time to push the big blind out of the side pot and keep the bounty for
itself (35% with PKO), where it only calls without bounties.

Limits:

* The proportional model is the simple standard: it overvalues a big stack's share of the bounties (it can't
  collect them all soon) and undervalues a short stack's, and it ignores where a bounty sits (a big bounty at your
  table is easier to win than one across the field) and the future game (ICMIZER's TrueBounty and GTO Wizard's
  simulations model those). A PKO's added half is valued as chips are, not by when it can be lost.
* Mystery bounties are valued at the average envelope (no risk preference about the draw).
* Push/fold counts a split as half a win and half a loss (a split busts nobody, so its bounty part is left out).
* Opens widen by at most OPEN_WIDEN (1.5x the chart's range); preflop spots the push/fold model doesn't play take
  the bounty as a discount on the risk premium.
* The 3-way solves keep their bystander approximation (who wins when a player loses is taken as even), so which
  opponent collects a bust's bounty when hero loses is averaged too; pots of five or more players get the betting
  rules in chips.
* The flop libraries, instant answers and presolves (all in chips) leave bounties out: a bounty tournament's flop
  solves live (Study's and the one-size view's solves pay bounties and prizes).

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

### One bet size

`"one_size"` (the page's **Simple** tab and Study's **Simple** view, `gto/static/simple.js`; `Advisor._one_size_view`)
answers how much a single bet size gives up, in place of the advice, at a heads-up decision where the player to act
can bet, or facing a bet, raise (see Raise sizes below). The street is solved again from that decision with the
ranges the advice reached it with: once with one size there for each candidate, and once with every candidate to mix
freely (the full menu). The rest of the tree (the opponent's bets and raises, the later streets) is the advice's, the
same in every solve (`Config.root_bets` sets the root decision's bets and nothing below it, `Config.root_raises` its
raises; a tree too big for the speed is cut alike for every size).

* **Candidates** (`ONE_SIZES`): 25, 33, 50, 75, 100, 125 and 150% of the pot, on the river 200% too, and all-in where
  the tree offers it (2.5 pots behind or less: a 3bet pot's turn and river). A quarter pot to 1.5 pots spans what
  single-size strategies use (a range bet small, a polar range big), closer together at the small end where a few
  points of pot move the EV most; a polar river's best single size is often the biggest (the advice's river already
  bets up to 250%). Bets round to half a big blind as the tree builds them: a size that reaches the stack is the
  all-in, and sizes that come out the same bet are one (`one_size_candidates`).
* **Raise sizes** (`ONE_RAISES`, `one_size_raises`): facing a bet, a raise to 2.5, 3, 4 and 5 times the bet faced
  (the opponent's total on the street: "3x" over a bet of 6.8 is a raise to 20.3), labelled by the multiple, each
  with what it raises to and that as % of the pot after calling (the answer's `raises`: `{"3x": {"to", "pct"}}`); a
  check-raise of a third-pot bet to 3x is 40% of the pot after calling, of a 75% bet 60%. All-in where the tree offers
  it (at most 2.5 pots behind once called) or a multiple reaches the stack. Raises round as the tree builds them (the
  raise over the bet to half a big blind), the root's raise count isn't capped (a raise over a check-raise is offered
  as the tree's sizes would be below the cap), and below the root the tree is the advice's (the opponent's re-raises at
  its sizes, all-in past the cap). Rows hold fold, call and the one raise; the definitions, cache, order and early
  stop are the bets'.
* **Best** = the size whose one-size game keeps the most EV for the player betting (or raising): the range's EV with that size alone
  (the game's value, weighted by hand pairs; bb relative to folding now), the opponent playing its best against it.
  **EV given up** = the full menu's EV minus that size's, in % of the pot. It is what the bettor can be sure of with the
  one size (no opponent does better against its equilibrium); scoring a size against the full menu's opponent
  instead would credit it with folds and calls that opponent makes only because other sizes are possible. The page
  states the definition under the table.
* **Precision**: each solve stops once settled (`ONE_SIZE_SETTLED`: 0.1% of the pot from equilibrium on the flop and
  turn, 0.05% on the river; half of Fast's 0.2, at which most of a turn's sizes tied) or at Fast's time budget, and a
  size whose EV is within the larger exploitability of its solve and the best's is reported **tied** with the best.
  The page always asks at Fast, whatever speed is picked.
* **Speed**: the full menu's solve is read every 50 iterations to 500, then every 100 (`ONE_SIZE_CHECKS`: a read
  costs about an iteration, and the advice's `HU_CHECKS`, 400, 600, 800, ran a solve settled at 410 iterations to
  600); the sizes' at `HU_CHECKS`, since a size's exploitability is its tie's tolerance and read more finely its solve
  stops nearer the target (one more size came out tied on two spots of the twelve below). After the full menu the
  sizes go in the order it suggests (the answer's `"order"`, `one_size_order`: the sizes it bets most first, then the
  rest nearest them), and a size whose solve shows it can't be the best nor tied with it stops early: in a zero-sum
  game the two best responses bound the game's value (the bettor gets at most its best response's value against the
  opponent's average strategy, at least minus the opponent's), and once a size's upper bound is `ONE_SIZE_CUT` (2) x
  the target under the best size's lower bound, and more than the larger exploitability under the best's EV, the
  verdict can't count it best or tied. Its row says `"cut": true` and the page shows its EV given up as approximate
  (≈, to 0.1%). Starting a size's solve from the full menu's regrets, or the full menu's from the advice's, saved
  nothing (a turn's 150% solve 0.092% of the pot from equilibrium at 300 iterations cold, 0.102% so started) and is
  not done.
* **Flops** are opt-in: on the page **Find the best size** starts them, with the estimate. The library's flop entries
  store two sizes, which can't say which of seven is best, so the flop is solved live (the advice's Fast flop model:
  1-3 s in all in the browser); doing it for every flop browsed would load the machine for answers rarely read.
* **Request**: `"one_size": "all"` solves the full menu and every candidate, `"full"` the full menu, `"50%"` (or `50`,
  or `"all-in"`) one size (any 5-500%, a candidate or not), `"list"` none; facing a bet `"3x"` (or `3`, 2-10x the
  bet) or `"all-in"`, and the answer adds `"kind": "raise"`, `"to_call_bb"` and `"raises"`. The page asks `"full"`, then each candidate,
  a request each, so it shows progress and stops between sizes or mid-solve (the server's `POST /cancel`, the worker's
  stop flag) without losing what's done. The answer: `{"one_size": {"street", "actor", "opponent", "position",
  "board", "pot_bb", "stack_bb", "candidates", "asked", "definition", "target_pct", "full", "rows", "best", "tied",
  "order", "hand", "hole"}, "notes", "seconds"}`; `full` and each row `{"size", "actions", "freq" (the range's),
  "amounts", "ev_bb", "grid", "iterations", "exploitability_pct", "seconds", "cut", "hand": {"freq", "ev"}}`, the rows
  with `loss_pct` (the EV given up) once the full menu is in, and `order` (the sizes in the order to solve them) with
  it. The page asks the sizes in that order. With `"study": true` it is for whoever acts, their whole range (no
  `hand`). An error preflop, facing a bet with no raise left (calling puts the stacks in), all-in, multiway, against
  the pool or locks, or for a size out of range.
* **Tournaments**: with payouts or bounties every solve pays prize equity (`Config.icm` from the decision, the bets
  already in counted, as the advice's own solves from mid-street), so the EVs and what a size gives up are chips worth
  as much prize equity (the definition says so), and the answer carries the tournament's notes (`"tournament"`, which
  the page shows under the table). Such a game isn't zero-sum (the players outside the pot gain when either busts), so
  the early stop's bounds don't hold and every size runs until settled. Six 30bb stacks, 50/30/20 to pay, BTN vs BB
  single-raised, flop and turn checked through, the BB first on the river (Ks7d2cTh5s): ICM checks 61% of the range where chips check 51%, and 33% gives up 0.13% of the
  pot (25-75% tied with it) where in chips it gives up 0.64% (no size tied); with 10 PKO on every head 33% gives up
  1.22%. Facing the BB's 3bb lead there the button's best raise is 4x in ICM (3x and 5x tied), 5x in chips.
  On the same table's turn after a called c-bet (Ks7d2c 3h, the button's 1.6bb bet called, checked to it) chips' best is 100% (0.11%, 125%
  tied), ICM's 150% (125% tied), PKO's 100% (0.16%); there ICM's single sizes of 100-150% come out 0.03-0.14% of
  the pot ahead of the full menu (a game that isn't zero-sum can leave a player less with more options, and each
  solve is within 0.05% of equilibrium), shown as giving up nothing. The view's solves took 4.3 s in chips (three sizes
  stopped early), 5.2 s with ICM and 5.9 s with PKO (one thread); Study's turn node 5.1 s in each.
* **Cache**: the solver keeps every row it solved (`ONE_SIZE_CACHE`, the 256 used last; a few KB each), by spot (the
  ranges, pot, stacks, tree and target) and size, and answers with all it has for the spot; the page keeps its runs
  by request, so reopening the tab is instant and a stopped run continues where it stopped.
* Without `"one_size"` the advice is unchanged: `root_bets` unset builds the same tree with the same cache keys
  (default answers' JSON checked identical on library flops, turns, rivers, preflop and Study).

Measured (`scripts/bench_one_size.py`, Study's request at each decision: the page's Fast trees natively on one thread,
each solve settled to 0.1% of the pot on the flop, 0.05% on the turn, 0.02% on the river; single-raised BTN vs BB,
3bet CO vs BTN, 100bb; the EVs given up with every size settled, Seconds the view's solves with the speed-ups and,
in brackets, without: the same best size and ties on every spot, the sizes stopped early within 0.1% of the pot of
their settled EV given up):

| Spot (player betting) | Best single size: gives up | Next best | Worst | Full menu's mix | Seconds (before) |
|---|---|---|---|---|---|
| K72r flop, BTN after a check | 25%: 0.07% (33% tied) | 33%: 0.14 | 150%: 1.22 | check 24, 25% 47, 33% 16, 75% 12 | 3.3 (3.8) |
| T94 two-tone flop, 3bet pot, BTN | 33%: 0.00 | 25%: 0.19 | 150%: 1.88 | check 46, 25% 8, 33% 42, 50% 5 | 1.6 (1.9) |
| K72r 3 turn, c-bet called, BTN | 150%: 0.15 | 125%: 0.45 | 25%: 2.06 | check 60, 50% 15, 150% 25 | 25 (40) |
| K72 two-spade 4s turn (flush in), BTN | 75%: 0.04 | 100%: 0.09 | 25%: 0.83 | check 61, 50% 12, 75% 16, 100% 6, 150% 6 | 49 (61) |
| Q83 A turn (overcard), BTN | 150%: 0.19 | 100%: 0.40 | 25%: 2.46 | check 46, 50% 8, 75% 17, 150% 29 | 29 (39) |
| Q83 5 turn, flop checked, BB first | 50%: 0.005 | 75%: 0.08 | 150%: 0.45 | check 67, 33% 6, 50% 21, 75% 3, 150% 3 | 99 (127) |
| K72 4 turn, 3bet pot, BTN | 75%: 0.03 | 100%: 0.21 | 25%: 2.93 | check 35, 50% 3, 75% 42, 100% 21 | 3.8 (6.5) |
| A85 8 turn (pairs), 3bet pot, BTN | 75%: 0.45 (33%, 50% tied) | 33%: 0.45 | 150%: 1.14 | check 32, 25% 18, 33% 25, 50-100% 25 | 3.6 (4.7) |
| K72 3 5 river, bet-bet, BTN | 200%: 1.04 | 150%: 1.16 | 25%: 8.73 | check 24, 50-100% 27, 150% 8, 200% 40 | 0.10 (0.15) |
| K72 4 9s river (flush in), BTN | 125%: 0.32 | 150%: 0.35 | 25%: 2.81 | check 52, 33-75% 24, 100-200% 24 | 0.11 (0.14) |
| K72 3 9 river, turn checked, BB first | 100%: 0.23 | 125%: 0.29 | 25%: 0.95 | check 65, 25-75% 18, 100-150% 18 | 0.4 (0.6) |
| Q83 5 J river, checked to it, BB first | 50%: 0.38 | 75%: 0.40 | 200%: 1.18 | check 56, 25-50% 26, 75-200% 17 | 0.4 (0.6) |

The best single size gives up a median 0.17% of the pot on these spots: 0-0.07% on the flops, 0.005-0.45% on the
turns (median 0.10%), 0.23-1.04% on the rivers, the most where the full menu splits a polar range over several sizes
(the bet-bet river overbets 200% with 40% of the range and bets 50-100% with 27%). Which size matters far more than
having a menu: a quarter pot gives up 2-3% on the bet-called turns and 8.7% on that river, and the best size runs
from 25% (a dry flop) to 200% (a polar river). In the browser (`scripts/bench_browser.mjs --one-size`, the page's
Fast with 8 threads and the targets above: 0.1% on the turn, 0.05% on the river) the whole view takes 1.1-22 s on four
turns (without the speed-ups 1.7-24 s; the same best size and ties on all eight spots): a single-raised turn in
position 4.9 s (was 7.2: the full menu 3.0 s, then 150% and 125% 0.4 s each and five sizes stopped after 0.2-0.3 s),
a 3bet pot's 1.1 (1.7), a 40bb turn 7.9 (9.2), 0.5-2.4 s on three flops (0.6-2.7), 0.4 s on a river. At those
targets more sizes come out tied (a turn where the BB bets first after the flop checked through: all seven within 0.1%
of the pot), which is the answer: there, any of them will do. A tie can't stop early (its tolerance is the solves'
precision), so that turn still takes 22 s (was 24).

Raise sizes (`scripts/bench_one_size.py`'s facing-bet spots: the same trees, targets and thread; the player facing the
bet raises to a multiple of it, in brackets as % of the pot after calling; Seconds the view's solves, 7 of the 25 sizes
stopped early, the same best and ties as with every size settled, whose EVs these are):

| Spot (player raising) | Best single raise: gives up, raises | Next best | Worst | Full menu's mix | Seconds |
|---|---|---|---|---|---|
| K72r flop, BB facing a third-pot c-bet | 5x (to 10, 84%): 0.08%, 11% (4x tied) | 4x: 0.12 | 2.5x: 0.21 | fold 39, call 39, 2.5x 12, 3x 3, 4x 2, 5x 6 | 1.0 |
| T94 two-tone flop, 3bet pot, CO facing a third-pot c-bet | 2.5x (to 13.5, 29%): 0.00, 41% | 3x: 0.14 | 5x: 0.57 | fold 29, call 31, 2.5x 31, 3x 8 | 0.6 |
| K72r 3 turn, BB facing 75% | 4x: 0.00, never (all four tied) | 5x: 0.00 | 2.5x: 0.00 | fold 42, call 58 | 10.1 |
| Q83 A turn (overcard), BB facing 75% | 5x (to 35, 121%): 0.00, 7% (4x tied) | 4x: 0.03 | 2.5x: 0.26 | fold 46, call 47, 5x 7 | 6.9 |
| K72 3 5 river, bet-bet, BB facing 75% | 5x (to 85, 120%): 0.00, 11% (4x, all-in tied) | 4x: 0.00 | 2.5x: 0.20 | fold 40, call 48, 4x 5, 5x 4, all-in 3 | 0.03 |
| Q83 5 J river, BTN facing the BB's 75% lead | 2.5x (to 10, 44%): 0.13, 21% | 5x: 0.28 | 4x: 0.37 | fold 34, call 48, 2.5x 17, 5x 1 | 0.02 |

One raise size costs less than one bet size: the best gives up at most 0.13% of the pot here (four of six spots
within the solves' precision of the full menu), the worst 0.2-0.6% where the worst bet size gave up 0.45-8.7%, since a
range facing a bet raises 0-39% of the time and the full menu's raises mostly take one size already. The size still
follows the range: big where the raise is polar (the overcard turn and the river after three bets, 5x: 1.2 pots),
small where it is wide (the 3bet pot's flop, 41% of the range at 2.5x; the river lead's 2.5x).

## How it works

| Layer | File | Method |
|---|---|---|
| Cards, ranges | `gto/cards.py`, `gto/ranges.py` | 1326-combo weight vectors, PIO-style range syntax |
| Hand strength | `gto/evaluator.py` | numpy-vectorised 7-card evaluator built on treys' lookup tables: all 21 five-card subsets of every combo on a board in one pass, about 1 ms per board |
| Preflop | `gto/preflop.py` | Hand-written 100bb charts: RFI, vs RFI, vs 3-bet, vs 4-bet, cold vs 3-bet, mixed frequencies. Where published solves exist the charts are matched to them, checked by `scripts/preflop_reference.py` (tests/preflop_refs.py: 26 spots, each played at the rake its solve used, 15 hand-level facts and 11 more played through the advice at the solve's rake): opens of 18.7 / 22.9 / 30.0 / 45.1 / 44.7% from UTG to SB (raked 2.5bb solves from pokalab, Poker Academy, GTO Ranges+ and gtopreflop.com agree on 17.6-18.7 / 22.3-23.3 / 29.1-30.6 / 43.3-46.5 / 43.5-45.7; were 16.7 / 19.4 / 27.2 / 45.6 / 50.2, with 22-44 half-opened from UTG and A2o from the BTN), the big blind's defence against the button (58.2/30.2/11.6 fold/call/3bet at gtopreflop.com's 5% capped at 4bb against its 58.7/29.8/11.5 and Poker Academy's 59.5/29.0/11.5; was 52.2/36.4/11.5; unraked it defends 55% of hands, was 64.6%; its 3bets polarised with offsuit wheel aces and suited kings, A4s-A2s calling), the small blind against CO and BTN opens (88/4/8 and 85/4/11 against 8.2% 3bets and 83.7/4.7/11.5; was 80/6/13 against both), UTG against a BTN 3bet (56/25/19 of its opens against 57.0/24.4/18.7, QQ mostly calling) and the small blind against the big blind's 3bet (64/20/16 against 63.6/21.4/15.0 at 10bb, now the default size; was 39/45/16, the chart written for 12bb). Average error over the spots 1.0 points (was 2.9, 10 of 22 outside tolerance). Published but thin: cold calls of a UTG open (gtopreflop.com: 2.1-4.6% of hands), one squeeze (Poker Academy) and one response to a 4bet (a GTO Gecko library export); not found: 5bets, heads-up. Seats for 2 to 9 players map onto the charts by how many players are left to act (9-max LJ plays the 6-max UTG chart, 7+ handed UTG plays a tighter EP chart, heads-up has its own button and big-blind charts). Replaying the preflop action through the charts gives every player's range going to the flop. The charts assume a standard-sized raise and nobody else in the pot; when the price or the players in differ (a min-3bet, callers ahead), calls are re-drawn by preflop equity against everyone in versus the pot odds, held to the equity bar the chart's own continue range sets in the standard line (with callers in, every call is re-drawn that way, not only added to the chart's: equity shared three ways is bunched, so the big blind over a button open and a small-blind call calls 15.0% of hands at 5% capped at 4bb, Poker Academy 16.1%, where adding to the chart's calls had 44.5%). With callers in, raises are re-drawn too: a value raise stays while its equity against what continues (each player's chart facing the raise) clears break-even by as much as the chart's weakest value raise heads-up does (callers' dead money makes thinner value raises pay), bluffs shrink by how much less often everyone folds against how often a bluff of that size needs them to (the callers' dead money lowers that bar), and the default raise size adds the callers' chips (a squeeze over two callers of a 2.5bb open is 15bb from the BB). Off-chart games and depths: limpers get iso-raise / over-limp / big-blind-raise charts (and an open-limp range); the iso-raises over one limper are a little tighter than the seat's open, at published widths (HJ 18%, CO 24%, BTN 36%, SB 19% of hands, the big blind 15%, against Preflop Wizard's and Grindlab's ~15-22%, 22-35%, 30-45%, 18-25% and ~15%; the old charts raised 10/13/16/10/8%; no solver frequencies for cash iso-raises were found: GTO Wizard's multiway preflop solver offers only iso-raise or fold facing a limp outside the blinds, and solved 6-max packs are mostly no-limp trees), about 0.6x that over two or more limpers (value-heavy, the dropped pairs and suited hands over-limping); antes play the open chart of a later seat and re-price calls, a straddle makes the straddler the big blind and the rest play by seats behind (both where no game's own chart covers the decision, below); stacks other than 100bb re-price calls with implied odds for small pairs and suited hands (from 70bb down, the small blind facing a CO or BTN open plays a 40bb chart instead, re-priced from 40bb: shallow, the hands under its 3bet value core flat, as in Poker Academy's 40bb solves where JTs, KJs, KTs, 88 and AQo call or jam; the 100bb chart thinned by implied odds left a 40bb flat of ~45 combos of 33-77 and A6s-A9s, now ~84 of suited broadways, 44-99, AQo/AJo/KQo and some suited aces, at 81/6/13 fold/call/3bet against the 100bb chart's 80/6/13); a raise that commits the stack (40% of it, or under 1.5 pots behind once called; 1.25 pots at the charts' own depth, where a 4bet to 2.3x leaves 1.45 and is called, as in 100bb solves) goes all-in, with the push/fold equilibrium as the range when shallower than the charts (and every open at 12bb or less); a jam over an open or limpers competes with flatting (equity against the raiser realized at the rate that makes the chart's weakest flats break even, plus implied odds) or the big blind's free check, not just with folding (the big blind rejammed a button open with 30% of hands at 25bb, now 22.5%). Preflop equity against several players is pooled per hand class over 400 boards weighted by how likely the players' ranges leave them (within 1-2 points of a dealt-out simulation five-way), and off-chart calls expect implied odds from the extra players by the square root of how many. Out of position facing a 3bet (a later seat's, or the big blind's against the small blind) the opener plays charts matched to published solver output: CO against a BTN 3bet to 7.5bb folds 63.7%, calls 17.2% and 4bets 19.1% of its opening range (Poker Academy, 100bb cash), openers respond in similar proportions from every seat (GTO Wizard), and the small blind against the big blind's 3bet to 10bb folds 63.6%, calls 21.4% and 4bets 15.0% (gtopreflop.com); the earlier charts called about twice that and 4bet half as often (CO 58/33/9). Also matched to published solver frequencies (100bb cash, 2.5bb opens): HJ and CO against a UTG open mostly 3bet or fold (90/2.5/7 fold/call/3bet of hands, against HJ 91.1/1.5/7.4 from Poker Academy and CO calling 1.4-3.2% and 3betting 8-9% from GTO Wizard; was 88/8/3), the button against a CO open (85/3/12, against 84.5/3.0/12.5 from Poker Academy; was 82/9/9), the big blind's 3bets against the button and the small blind (11.6% and 16.4% of hands, as Poker Academy and gtopreflop.com have them; were 8.5% and 11.7%), and the button facing the big blind's 3bet (54/34/12 fold/call/4bet of its opening range, against 54.6/33.9/11.5 from 888poker; was 59/35/6, a 4bet range of nearly all value) and the small blind's (60/29/11.5: gtopreflop.com's raked solve has the button 4betting as often against either blind, 10.4% and 10.3%, and calling 6 points less against the SB, with QQ, AKo, A9s and KTs mostly 4betting and AQs/KQs calling; was 58/36/6). The big blind's 3bets against UTG, HJ and CO are 4.8%, 6.2% and 8.0% of hands (gtopreflop.com 4.5/5.9/7.7, Poker Academy 4.5% against the LJ, pokalab 5.1/6.7, BeyondGTO 6/8/10; were 3.6/3.6/5.6, the HJ spot reusing the UTG chart), taken from calls so the defence totals stay; its squeezes follow (over CO + BTN 7.3%, was 4.9%; over UTG + a caller 4.5%, was 3.4%), about 0.9x the heads-up 3bet where GTO Wizard's NL500 cash solve at 40bb has 0.95x. Cold players facing an open and a 3bet fold 96-97% (GTO Wizard: over 90%, up to 97%) and the big blind's squeeze over a button open and a small-blind call is 9% of hands (Poker Academy 12.7%): both left. The button cold-calls a UTG open 4.5% of hands and 3bets 6.4% (gtopreflop.com's raked solve: 4.6%, 3betting more often than calling; the chart had the NL500 solves' 9.1% flats and 5.2% 3bets). No cash solves of the small blind's or button's squeezes were found (gtopreflop.com's database has no node facing two players in the pot; the tables that turn up are labelled approximations). The opener calls a blind's 3bet wider (in position), the heads-up button has its own vs-3bet chart, and a 5bet that leaves chips behind can be shoved over. Every player's replayed preflop range goes through the same adjustments as hero's advice (so a cold call of a cheap 3bet or a multiway big-blind defence is wide), a limper facing a raise, an iso-raiser facing a limp-reraise, the heads-up big blind facing a 4bet and every seat (blinds included) facing a 7+ handed early-position open have their own charts; facing a 3bet cold depends on how wide the 3bet is (over an early or a late open) and on position against the 3bettor (out of position, half the cold calls go: premiums 4bet, the rest fold), and facing an all-in that someone already called, equity counts the callers. A player who called a raise and faces the next one (a flat facing a squeeze, a cold call of a 3bet facing the 4bet, a call of a 4bet facing the jam) continues with at least 30% / 35% / 55% of the range they called with: the charts for those spots are written for someone who hasn't put chips in and continue mostly with QQ+/AK, which a flat almost never holds, so they left a flatter's range nearly empty (2% of BTN's flats called a BB squeeze). Those shares are floors, set against the tightest raises the charts make (the big blind's old 3.6% 3bet over UTG, UTG's 4bet, the 3bettor's jam); a wider, more bluff-heavy raise is defended more: each point of equity the caller's range gains against the actual raise over that reference adds 2.5 points (BTN's flat facing the big blind's published 8% squeeze over CO continues 44% and calls with 77; with a fixed 30% the wider squeeze had pushed 77 out). The extra calls are its strongest folds, ranked by how readily the button continues against a 3bet (TT-77 and suited broadways before 22-55), then equity against the raiser. Once another raise is in after the call (sandwiched) the charts' folding stands. The small blind flats an early open rarely (about 3% of hands) but not only middling hands: a share of JJ, TT, AQs, AJs and KQs flat as traps (published summaries have it sometimes calling JJ and TT there), where the old chart 3bet them all, so its flat-then-call of a big-blind squeeze was 13 combos of 77-99 and the squeezer's QQ won every A-K-x flop; now about 12 combos of 99-JJ, some 77-88 and those suited broadways. The 3bettor facing a 4bet uses a chart written against the opener's 4bet, which carries bluffs; against a cold 4bet or one from a player who flatted first (nearly all value) each hand is re-drawn by its equity against the actual 4bet range versus the opener's: value hands keep continuing while their equity clears the price of calling over the 80% of it a call realizes (one bar for every spot, so JJ calls 53% against a cold 4bet over UTG, where it has 33%, and always over the button, where it has 36%), and marginal and blocker hands shrink by how far they drop (AQs calls 5%). Bluff jams, and the value jams the chart mixes with calls, scale by how often that range folds to a jam compared with the opener's: against its value a jam and a call both get the stacks in, so a jam's edge is the equity it denies the bluffs, while a call can still fold bad flops. Against a cold 4bet over UTG QQ and AKo jam 30% (the chart 80%), and against a flat-then-4bet over an early open, which folds far less, they call instead; KK+ and AKs keep jamming. Against the opener's own 4bet the chart jams KK+, AKs, most of QQ and AKo and some JJ and A5s (fit to the one published response found: GTO Gecko's library has the big blind's 3bet range folding 46.9%, calling 29.3% and jamming 23.8% against the button's 24bb 4bet, here 45.6 / 34.0 / 20.4 with the same cards removed; the chart jammed 15.9%). A 4bettor facing a 5bet jam calls by equity against the jammer's replayed range versus the pot odds, like hero's advice (the fixed chart called QQ and AK 70% whoever jammed, where they have 36% against a standard jam and need 41%). **Other games' charts** (`gto/pf_formats.py`, `preflop.format_chart`): a decision on one of the charts' heads-up lines (raise first in, facing an open, the opener facing a 3bet, the 3bettor facing a 4bet, the opener facing a 5bet) in a game the 100bb 6-max charts weren't written for plays that game's own chart: stacks of 20-100bb (6-max solves at 20, 40 and 60bb, linear between), a 12.5%-of-a-big-blind ante at 20, 40 and 100bb, 7-9 handed tables, a 2bb straddle, and heads-up at 20, 40 and 100bb, the small blind's limps included. Deeper than 100bb plays 100bb's charts (with that game's moves), re-priced by the rules' implied odds: the 150 and 200bb games were solved too but not adopted, since one depth factor per pot type can't pay small pairs and suited hands the implied odds a deep stack does (moved by the 200bb solve the button would call a big blind's 3bet with 22 37% of the time and 77 71%, where the 100bb chart calls 75% and 100%, and the rules at 200bb call both always). Each game is solved whole (`pf_game` for that table, its flops valued by the flop library, raked 5% capped at 4bb like the charts' references, 1,500 iterations, 0.0002-0.0125 bb/hand from equilibrium; `scripts/build_pf_formats.py`). Its chart is the 100bb chart plus how far that solve moves each hand class from the 100bb solve of the same table at the same decision, so the solve's own errors at 100bb (tight opens, few calls of 3bets), which are in both, cancel. 7-9 handed and straddled seats move the 6-max chart of the seats they play (9-handed seats as the 8-handed seat with as many players behind, at most 7). Depths and the ante move the table's own chart (the ante scaled by the dead money, players x ante over 6 x 0.125, at most 2x), and heads-up the heads-up chart. At another depth the library's 100bb pot shares are scaled by how much a flop's realization changes there (`build_pf_formats.DEPTH_SCALE`, from live solves of BTN vs BB, SB vs BB and BB vs BTN's 3bet pot on 12 flops at 20, 40 and 200bb against the library's at 100bb, `scripts/pf_depth_values.py`): single-raised pots barely move (0.89 / 0.98 / 0.96 of 100bb's excess over equity), 3bet pots realize less as the stack shrinks to the pot (0.48 / 0.76 / 1.03). A move counts only where both solves bring the hand to the decision (an unreached hand's strategy is arbitrary: unweighted, the 3bettor's 72o continued against a 4bet in 39 lines with the ante). A move that takes out hands the chart keeps below a weaker one it keeps is pooled along the dominance order rather than having them added back (the 8-handed UTG's solve drops 33 and 44 but not 22: pooled it opens 15.8%, added back 17.2). Calling an all-in may make a three-way showdown in the tree (before, a 3bettor facing a cold jam and a call could only fold, and at 40bb with the ante the solve 3bet nothing over a button open). Measured: 8-max opens are 15.8 / 16.7 / 17.9 / 22.8 / 29.7 / 45.1% from UTG to the button against gtopreflop.com's 13.9 / 15.8 / 18.1 / 22.2 / 29.2 / 43.7, 0.92 points apart on average and none outside 2 (the rules' EP chart: 11.5% from UTG and UTG1, 1.70 points; `scripts/preflop_format_reference.py`). Against the shallow references, which are looser matches (9-handed with a big-blind ante, min-opens, heads-up with an ante), the moved charts are 3.4 points off per reference where the rules were 11.4: Poker Academy's 40bb small blind against a CO open 78.1 / 11.1 / 10.8 fold / call / 3bet against 72.6 / 14.4 / 13.1 (rules 42.8 / 44.5 / 12.7), against the button 75.2 / 10.9 / 13.9 against 71.4 / 11.6 / 17.1; GTO Wizard's 40bb big blind 3bets a CO min-open 9.7% (10.3) and squeezes 9.3% (9.8); its heads-up 50bb small blind folds 0, limps 36.9 and raises 63.1% (4 / 32 / 64; the rules raised or folded). 6-max opens from UTG to the small blind: 19.1 / 23.9 / 28.1 / 36.6 / 43.4% at 20bb, 19.0 / 24.4 / 32.2 / 45.1 / 51.4 at 40bb, 18.7 / 22.9 / 30.0 / 45.1 / 44.7 at 100bb (the charts); the big blind against a button open 47 / 38 / 16 fold / call / 3bet at 20bb, 44 / 43 / 14 at 40 and 45 / 43 / 12 at 100. With the ante at 100bb the opens are 19.4 / 24.2 / 33.6 / 48.2 / 51.8% and the big blind defends the button 71.0% (29.0 / 57.3 / 13.6), at 40bb 22.2 / 26.3 / 33.7 / 46.7 / 58.8 and 73% (27 / 59 / 14). Straddled, the first seat after the straddle opens 20.5% and the button 33.6. Heads-up the small blind first in folds, limps and raises 6.6 / 66.7 / 26.7% at 20bb, 14.6 / 21.4 / 64.0 at 40 and 11.7 / 28.9 / 59.4 at 100. The advice names the game's chart in a note and re-prices from that game (Advisor._priced). Lines with limps (the big blind against a limp included) or callers, and games no solve covers, keep the rules above. |
| Postflop | `gto/solver.py` | Heads-up range-vs-range **discounted CFR** over a bet tree (2 bet sizes per street, raises to half pot and full pot after the call, all-in when the stack is short or a raise size would exceed it, max 2 raises). Vectors are compressed to combos in either range. A river showdown is O(N) per board: combos sorted by strength, cumulative reach sums, and a per-card correction for card removal. Equity leaves are one matvec with a precomputed antisymmetric matrix; natively a pass first collects every leaf's opponent reach and multiplies each leaf matrix stack by all of them in one matrix-matrix product (one product per leaf was memory-bound). Subtrees below the root street with the same shape (the turn trees after most flop lines differ only in pot size) are merged into one batch and traversed once, so a flop solve walks ~30 decision nodes a pass instead of ~90, with the same numbers (~1.6x natively). In the browser, whose numpy has no BLAS, the leaf matvecs are most of the time: they run through einsum there, on leaf matrices trimmed to each player's range, and per-card sums replace the small matrix products. DCFR (alpha 1.5; beta 0, negative regrets halved every iteration, for the first 100 iterations, then 0.5) with the average weighted to recent iterations, gamma 3 rising to 8 by the 50th iteration; other regret minimizers (CFR+, PCFR+, PDCFR+, hyperparameter schedules) are `Config.cfr` options, compared in `docs/cfr_research.md`. From the flop the river beyond `depth` keeps its betting, played on the turn's boards, and the turns are one card per rank class (see Accuracy vs speed). `scripts/bench_solver.py --curve` plots exploitability against time, natively or in Pyodide (`scripts/pyodide_run.mjs`). |
| Compiled core | `core/gtocore.c`, `gto/core.py` | `Solver.run`'s iterations on turn and river trees, and on flop trees whose turn and river are played by hand buckets, in C without libc: in the browser as `gto/core.wasm` (53 KB, wasm SIMD, shipped in `solver.zip`; `scripts/build_core.py`), natively as a shared library compiled from the same source on first use. Python builds the tree as before and lays it out as a program of int32 records (decisions, folds, river showdowns with their Ranking's sorted order and card-removal sums, all-in equity leaves with their matrices, the river dealt card by card or by class, a flop's turn by hand bucket and river by strength with their tables: each combo's bucket, the shares of rivers that put a bucket in each river bucket, the bucket pairs' equity and compatible-share matrices) in one arena with the regrets, strategy sums and reaches; the core runs iterations in chunks of ~0.1 s, with the time budget, `reserve` and cancelling checked between them, and the regrets and sums come back. It is `Solver._cfr` in float32 with the same order of sums, the discount factors from Python (`Solver._factors`), so the first iterations agree to rounding; later a hand indifferent between actions goes one way or the other by rounding and the two part at that node, as numpy's own native and browser paths do (the same nodes, by the same amounts), equally far from equilibrium. It also runs the passes after a solve, the EVs on the average strategies and the best responses (`node_action_evs`, `node_action_br_evs` without locks, `exploitability`), within 2e-5 chips of numpy's. The last solver's program stays in the arena for those passes and for `extend`, until another solver's replaces it. A tree of many equity leaves (over `core.LEAF_ROWS_BLAS`, 256 rows: the library's flop, every turn line ending in one, 7228 rows of 854 x 854 on the SB vs BB pot's Tc7d2d) multiplies them in a batch natively, as numpy does: a pass first walks down with the opponent's reach alone (their strategies, the chance nodes' masks) and writes it at every leaf row, each matrix then multiplies all of its rows at once (Accelerate's `cblas_sgemm` on macOS, the core's own register-blocked loops elsewhere), and the pass reads each leaf's utilities from the product. A training pass multiplies only the combos in the two ranges (the matrix's rows and columns of the root's live combos; a combo outside a player's range gets 0 there, as numpy's `trim_leaves`), the passes after a solve the whole matrix; in the browser (no BLAS) they stay one leaf at a time. A dealt river's bucket games (a 3bet pot's library flop: 64 buckets on each card under each turn line) are deferred natively (`core.DEFER_GAMES`): a training pass walks down storing each game's bucket reaches, plays every turn line's game on a card together, a card at a time (the card's tables, 32 KB a player, multiply every reach that reads them at once, over its nonzero entries), and reads the games' utilities in the pass proper: the same floats in about a third of the time. Sampled rivers, predictive regret schemes, exploit-mode locks, and a solve while another holds the core run in numpy, and so does a program the arena can't hold (over `core.ARENA_MAX`, 1 GB, or the browser's memory refusing to grow); `GTO_CORE=0` turns it off. Threads: every node runs on a range of its rows (its boards), and rows never mix below a chance node, so the first chance node on the way down (a turn's river classes, a flop's turn classes) hands its child's rows out in chunks to the thread and its helpers, each with scratch of its own (a lane, sized by one serial iteration a program that measures a chunk's; a deferred pass's chunks, a row at a time, apart from an evaluation's, which measures its own the first time): every row gets the same arithmetic on any thread, so a solve's numbers are the same bit for bit on any number of them (`core.threads`). Natively the helpers are pthreads; in the browser `gto/core-mt.wasm` (the same source with wasm atomics, its memory shared) runs in the solver's worker and up to 7 helper workers, which needs the page cross-origin isolated: GitHub Pages can't send those headers, so the page's service worker (`gto/static/sw.js`) adds them to its pages and worker scripts, and the first visit loads the page once more under it; without isolation `core.wasm` runs alone. `tests/test_core.py`, `tests/js/core.test.js`. |
| N-player pots | `gto/multiway_solver.py` | Four or more players in (up to 6; 5 at the browser's Quick and Fast), or three with two all-in: the same CFR (alternating updates) with pairwise card removal: a hand's share of a pot against the players contesting it is the product of its heads-up shares against each times the others' masses, each pair of opponents' overlap corrected (their exact joint mass on the root's board) and every board scaled to stay constant-sum (`_term_pw`, the factor held to [0.5, 2]: on lines nobody takes, ratios of products of near-zero reaches overflowed the regrets). Pots are layered by the all-in players' stakes, each to the best hand among the live players whose chips reach it (`MWConfig.allin`, `inv0`, `dead`). Against the exact 3-way kernels its EV per deal is within 0.3-0.7% of the pot on average (2.2% the worst hand). Four or more betting play one bet a street (half the pot) and no raise but the player advised's (`root_raise`): a raise for everyone made the 4-way tree 1,053 nodes that didn't settle; without, 65 nodes and 0.02% of the pot in 150 iterations (5-way 161, 0.01%). Four players on the flop or turn (at most one all-in: the others bet for the side pot) are solved too, the later streets bet on by hand buckets as 3-way (`advisor.MW4_STREETS`: under a flop the turn by 8 buckets, its tables over every third river, under a turn the river by 16; Fast `MW4_FAST_STREETS`, 6 on every fourth river and 8; Quick's flop 6 on every sixth; `MW4_BETS`, a third of the pot on the flop and half after): this street per combo by the pairwise terminals, the later ones by tensors over bucket quadruples (`BucketTables._class_n`), in the compiled core too (`con4_row`). Natively (HJ opens, CO, BTN and BB call, Kh7d2c) a flop first to act settles to 0.14% of the pot in 100 iterations of Fast's game (20-25 ms an iteration after 0.5-2.4 s of tables) and 0.21% in 100 of Accurate's; a turn (Kh7d2c9s) to 0.04% in 100 (31 ms an iteration). Against a per-combo turn solve that deals and bets every river (`scripts/score_multiway4.py`'s reference, Kh7d2c9s, 150 iterations each: 1010 s of CPU against 12 s), the first decision played by the 16-bucket game's mix gives up 0.015% of the pot by the reference's EVs (the range checks 86.6% against 85.9%; the reference itself 0.04-0.05% from equilibrium, the buckets 0.01-0.03%), and facing a bet and a call -0.02% (fold / call / raise 34 / 24 / 43% against 38 / 18 / 44%; 886 s against 3.6 s, the reference still up to 0.78% from equilibrium there, the buckets 0.06%). Everyone plays the shortest stack behind of those who bet. Five or more are solved on the river only (and any street where you are the only one left with chips: dealt out, exact): with the later streets dealt out (equity leaves) a 3-way turn solve gave up 6.5% of the pot against the exact (bucketed) solve, the rules 1.5%; in 6-max 100bb a 4-way flop is 0.10% of the flops seen (2-way 95.7%, 3-way 4.2%: the charts' preflop lines), so no 4-way flop library is built. On 3-way rivers, scored by the exact solve's EVs: facing a bet and a call the rules gave up 16.5% of the pot on average (35% of hands over 2%), the pairwise solve 0.14% (3%); first to act 0.09% and 0.00%. Natively a 4-way river takes 2.2 s, 5-way 4.9 s (Fast); in the browser 4-way 2.5 s (Quick 1.8 s), 5-way 7.6 s. |
| Hand replay | `gto/advisor.py` | The whole hand is replayed for any number of seats: blinds, folds, dead money from players who folded, all-ins, acting order (heads-up the button acts first preflop and last postflop). Whenever exactly two players remain the spot is solved heads-up: from the start of the street if it began heads-up, or from the current node if a third player just folded. With three live players the 3-way solver takes the spot where it applies (below); four or more (or three with two all-in) the N-player solve takes their river, and any street where you are the only one left with chips, and four players (at most one all-in) their flop and turn too (below, N-player pots); otherwise (5+ players on the flop or turn, two or more all-in among four, a street it doesn't take) a Monte Carlo heuristic plays against every live range (see Limits), and says so. On the river, facing a bet in a heads-up or 3-way solve, a hand that beats at most 5% of the bettor's range (combos the board or your cards block left out) gets a note naming that share, the share it ties and the range's five weakest hand classes, so a fold to a tiny bet reads as what it is: "A8s beats 0% of CO's range and ties …" on 7d9hQcKsJh, where every Ax ties and nothing in the range is worse. |
| 3-way postflop | `gto/multiway_solver.py` | Three-player range-vs-range CFR (alternating updates, DCFR discounting: alpha 1.5, the average weighted by (t / (t + 1))^3, which at equal time is 15-30% closer to equilibrium than ^2 on 3-way rivers and turns; simultaneous updates were 5-10x further) from the node in front of hero: bets of 33% and 75% on the flop, 50% on the turn and river (one size inside a flop solve too), one raise, all-in at SPR 1.5 or less, each player at its own stack as far as another can match it (`MultiwaySolver`'s `stack` per player; was everyone at the shortest, so a deep player's all-in was a bet to the short stack's cap and a bet over the short stack went to the rules). Side pots: a player all-in for less, at the root (a shove two players called, `MWConfig.allin`) or by a call or shove below it, never acts again, a bet over its call comes back to the bettor where nobody else matches it, and a showdown pays the pot in layers by every seat's chips in (`MultiwaySolver._layers`, from `MWConfig.inv0` and the folded seats' `dead`): the main pot to the best of the hands whose chips reach it, the side pot to the best among the players with more in; a fold gives up both. The exact kernels take each layer with its players; a bucketed terminal carries each row's main-pot share and the side pot's players (`BTerminal.fm`, `side`), as the compiled core does. In the browser a flop or turn with a player all-in at its root takes 1.4-1.6 s (100 iterations, 0.05% of the pot from equilibrium), so Quick solves it on every street. Over 600 fuzzed hands with random shoves, the 3-way solves went from 5 to 22 of 272 postflop spots (the solved share 14% to 20%), none failing; with each player's own stack and 4-way flops and turns, the 49 multiway postflop decisions of the page's random hands (tests/data/page_states.json, Fast) are solved 38 times against 27 (3-way 28 of 29, was 26; 4-way 9 of 10, was 0), none failing. Two or more all-in, or five players still betting before the river, stay with the rules. Card removal between the three hands is exact on the river: a terminal's value per hand sums the two opponents' hands that share no card with it or with each other, by inclusion-exclusion over the 52 cards on running sums (O(52 n) a terminal, not O(n^3)). Each board's bounds into those sums (a combo's run of equal strengths, and of equal (card, strength) keys) come from one scan of the sorted rows (`multiway_solver._tie_bounds`; was a binary search per combo, the same numbers): in the browser on one thread a 3-way Fast turn's solve took 6.4 s where it took 7.5, a 4-way flop's 3.85 where 4.7. Turn and flop solves bet the later streets too, by hand buckets (16 a player; Fast's flops 12, their turn's tables over every third river; in the browser without the compiled multiway core 12 and 6) on one card per rank class, the river from a flop played on the turn's boards; showdowns are constant-sum tensors over bucket triples, and subtrees of one shape after different lines are batched. The iterations of flop, turn and river solves run in a compiled core (`core/mwcore.c`, `gto/mwcore.py`: natively a shared library built on first use, in the browser `gto/mwcore.wasm`, ~125 KB, and `gto/mwcore-mt.wasm` on the page's helper threads: The multiway core on threads, in Accuracy vs speed): the tree as a program of records, each decision's regrets and sums in an arena, the showdown tensors contracted a row of K at a time and skipped where a reach is 0, numpy's arithmetic in float32 (values within ~1e-5 of the pot of numpy's; numpy stays the fallback and the reference). A river's showdowns are leaves of the program too (`RSHOW`: the exact kernel above, its card sums swept once by strength in float32: 3-way rivers in the compiled core, below). Natively a first-to-act flop iteration takes 50-90 ms where numpy took 0.6-1.15 s (12-13x: Kh7d2c in Fast's game, Kh8d3c in Accurate's, CO opens, BTN and BB call); in the browser 26 ms in Fast's game and 33 ms in Accurate's (Kh8d3c, an idle machine), converging the same (Fast's game on Kh7d2c: 100 iterations 1.32% of the pot from equilibrium, numpy 1.41). Research, design and the error accepted: `docs/multiway_research.md`. |
| Multi-street | `gto/advisor.py` | Earlier streets are solved with a small budget and each player's range is narrowed by the frequency of the action they actually took, mixed with a 15% uniform floor so an approximate solve never prunes a line to nothing, and followed only as far as the solve has settled (the rest by the betting rules' range for the action, see Accuracy vs speed). A range entering a later street with fewer than 8 live combos (preflop.MIN_LINE_COMBOS, counting the cards dealt since) is filled out with the rest of the range the player started the street with, weighted as preflop (1 - combos / 8): a 4bet pot's ~12-combo range checking back the flop left 2 for the turn, so the turn solve played a face-up hand. The current street is solved from the start of the street and the tree is walked to hero's node with exact frequencies. Solutions are cached per hand so the flop solve is reused on the turn. |
| Preflop subgame (experimental) | `gto/pf_solve.py` | Not used by the advice yet. Solves the 3bet pot heads-up as a game: the opener facing the 3bet folds, calls or 4bets; the 3bettor facing the 4bet folds, calls or jams; the opener facing the jam folds or calls. Hands are the 169 classes with class-level card removal; CFR+ with linear averaging converges to under 0.01% of the pot from equilibrium in 1000 iterations (~0.2s). An all-in pays equity; a call that sees a flop pays realized equity: a base by position and pot type (in position 1.0, out of position 0.8 in a 3bet pot and 0.9 in a 4bet pot), less for a hand behind (1.2 per unit of equity below 50% in a 3bet pot, 3.5 in a 4bet pot, where a dominated hand pays off or folds to a flop shove), a little more for one ahead in a 3bet pot (0.3), and a playability bonus for suited and connected hands; the two shares are normalised to add to 1, so the game is constant-sum. The constants are fitted to the charts' continue rates with `scripts/calibrate_pf_solve.py`: they landed within a few points of the old charts, but since the charts were matched to published solves (openers fold far more against a later seat's 3bet) they don't: openers call 37-55% of their range against the charts' 17-41% (calibration score 0.58, the sum of squared share differences over the six spots). The solve has 4bet bluffs (KJs, KQs, QJs) and jam bluffs (A4s, A3s, A5s), and QQ near indifference facing the jam. A strong hand facing a 4bet often calls rather than jams (so jam ranges are AA plus bluffs and KK sometimes folds to them), which comes from the realized-equity model of a call: it is the main limitation. `scripts/build_realization.py` measures it with the postflop solver instead, once, on the charts' ranges of six standard 3bet pots (openers out of position against a later seat's 3bet, in position against a blind's; 96 flops each, in the 3bet pot and the 4bet pot) into `gto/pf_realize.py`: per hand class against each of 9 opponent strength buckets (AA, KK, QQ-JJ, ...), how far its pot share after the call (EV over the pot, later bets included) runs above its all-in equity; the two sides' excesses combine so the shares add to 1, blended with the parametric model where measurements are thin. Per bucket matters (one number per class let KK's postflop edge count against AA, so KK folded to AA-heavy jams; now KK against AA is about -1 pot and AA against KK +1.8), and measuring on fixed ranges matters (re-measuring on the subgame's own ranges to a fixed point didn't settle). The flops are stratified by texture (suit pattern, pairing, high card, connectedness; 34 strata, at least two flops each) and weighted by the share of all flops each stands for: with random flops the two halves put the opener's calls against a 3bet up to 22 points apart, with weighted flops at most 10 (tests TestRealizationTable; the halves are in tests/data). The table is keyed by role as well as position (the caller, or the aggressor whose bet was called): position alone mixed an in-position 3bettor (realizing 1.15 of its equity in a 3bet pot) with an in-position caller (0.97), and with that mix the 3bettor trapped KK and the opener folded KK to the AA-heavy jams; by role KK calls them everywhere. `scripts/fit_realization.py` fits a smooth model to it (`gto/pf_realize_model.py`): per pot type, role and position, eight coefficients on the hand's equity against the bucket (x = e - 0.5: x, x·abs(x)) and its shape (pair, suited, connected, pair*x, suited*x). It stays off (pf_solve.MODE = "param"; "table" and "model" switch to the measurements; tests TestRealizationModel) because the measured caller-aggressor gap looks too large: callers realize 0.79 (out of position) and 0.97 (in position) in 3bet pots against the aggressor's 1.02 and 1.15, and about 0.6 in 4bet pots against 1.25, so in-position openers called 3bets less than the charts (BTN vs BB 28% against 34%, CO vs BB 25% against 41%) and out-of-position ones more (CO vs BTN 47% against 18%). Neither the postflop solver's depth nor its bet sizes explain the gap: playing the river instead of scoring it by equity moves these about a point, and four bet sizes a street (25-125% on the flop, 33-150% later) instead of two move them under one (40 stratified flops, BTN vs BB and CO vs BTN 3bet pots and the CO vs BTN 4bet pot). Nor does the 3bettor's range: widening BB's 3bet against a BTN open from the chart's 8.5% of hands to the published 11.5% leaves the BTN caller at 0.99 (40 flops). In aggregate an in-position caller realizes about its equity; the call rate falls short of published solves because marginal hands (KQo, A5s) measure a few points below their equity while strong hands run above, and checking that needs per-hand EVs from a commercial solver. So either the gap is real for the charts' ranges (a capped calling range against a strong, uncapped one: in the CO vs BTN 4bet pot the caller holds JJ, TT, AQs against KK+, AK) and the charts call too much, or the subgame's calls lack something the charts encode; settling it needs outside solver data. Public equity-realization figures cover only some of the roles. The GTO Wizard aggregate reports quoted by Poker Academy give 1.14-1.17 for the in-position raiser of a single-raised pot and 0.78-0.87 for the out-of-position caller (BTN vs BB, LJ vs BB, limped pots, 25-184 flops). A GTO Wizard blog example (BTN 3bets CO, 552r, 100bb) splits equity about evenly with EVs of 10.78bb against 7.72bb, which is 1.17 against 0.83. The measured aggressor in position (1.15) and caller out of position (0.80) match these. Nothing published covers an in-position caller of a 3bet. Rules of thumb (deepfold: 1.08-1.15 for late-position flats; Tournament Poker Edge: 1.25-1.5 for an in-position flat of a 3bet) put it above the measured 0.97, and per-class figures put offsuit broadways and offsuit aces below their equity and pairs and suited connectors above it, as measured (in-position caller: Axo 0.80, offsuit broadways 0.86, suited connectors 1.07, 22-66 1.11). The model mode now adds 0.18 to the in-position caller's realization in 3bet pots (pf_solve.MODEL_SHIFT, 0.97 to 1.15), which brings the in-position openers' calls to the charts (BTN vs BB 41% against 34%, CO vs BB 31% against 41%, BTN vs SB 38% against 36%; calibration score 0.30 against the parametric model's 0.58 and the unshifted model's 0.32). The out-of-position openers still call 2-2.5 times as often as published solves (CO vs BTN 47% against 18%, UTG vs CO 39% against 18%), even though the measured 0.80 matches public figures for that role. Fitting the charts would take it to about 0.64, against the 0.78-0.87 published for out-of-position callers, so the shift isn't applied there. The 3bettor also jams too often over a 4bet (28-39% against the charts' 15-39%). So the default stays "param", and the solver stays blocked: what's missing is per-hand EVs for an out-of-position caller of a 3bet. |
| Preflop equilibrium (derived) | `gto/pf_game.py`, `gto/pf_eqr.py` | The full 6-max 100bb preflop game (2.5bb opens, SB 3bb; 3bets 3x in position, 12bb from a blind, 10bb BB over SB, one more open per cold caller; 4bets 2.3x; 5bet all-in; at most three players see a single-raised flop, two a raised one, three an all-in: 11,010 nodes, 5,470 decisions; 8,847 before a third player could call an all-in) over the 169 classes, solved by discounted CFR level by level in numpy (1,500 iterations, ~0.002 bb/hand from equilibrium, ~3 minutes). A flop pays each player its realized pot share, measured with the postflop solver rather than assumed (`scripts/pf_equilibrium.py`): 17 heads-up flop spots (single-raised, 3bet and 4bet pots, aggressor in and out of position), 68-102 texture-stratified flops each, every class of the pre-action range added at weight 1e-3 so hands outside the range get their counterfactual value; per pot type, aggressor position and role a model of the excess share over all-in equity against 9 opponent strength buckets (hand features plus shrunk class residuals, `gto/pf_eqr.py`), iterated with the solve to a fixed point. It measures BTN vs BB single-raised pots at 1.13 / 0.85 (published solver reports 1.14-1.18 / 0.79-0.87) and a BTN-vs-CO 3bet pot at 1.13 / 0.87 (published example 1.17 / 0.83); three-way pots use the product of heads-up equities corrected per class (`scripts/build_three_way.py`) and a realization factor by seat and the opener's place, measured with the 3-player solver (`scripts/pf_three_way.py`). Against the published solves (tests/preflop_refs.py) the solve is 2.5-3.0 points off on average with 12-13 of 22 spots and 8-10 of 14 hand facts outside (the charts: 1.0, 0, 0): right on the big blind's defence against the button (58.1/29.3/12.6 against 58.7/29.8/11.5) and openers facing a 3bet (CO vs BTN 63.5/10.3/26.2 against 64.0/16.9/19.2), but its opens are tight (BTN 37.6%) and prefer offsuit broadways to small pairs and suited connectors. So the charts stay; `scripts/pf_compare.py` lists what each chart costs inside the solved game and `scripts/pf_subgames.py` solves the reference spots entered with the charts' ranges (docs/preflop_research.md section 6 has the details and why no spot without a reference was replaced). A second pass (section 7) tested each suspected structural cause by experiment: rake (moves calls, not opens), three-way flops (nothing), SB limps (`sb_limp`: limps 8.6%), published 4bet sizes (`Sizes.four_x_ip/oop`), the flop model's fidelity (`scripts/pf_fidelity.py`: with the river dealt and played, suited connectors and small pairs realize 0.1-0.3 more and the ranges within 0.03), per-spot fits (`--by-spot`) and nine realization shifts fitted to the references; the best without fitting is 2.42 points, 10 spots and 6 facts out, fitted 2.17, and a 0.03 change in one pot type's realization moves it by up to 0.5. A fourth pass (section 9) valued the flops by the flop library itself (`pf_game.LibraryModel`, `gto/pf_libvals.py` from `scripts/pf_library_values.py`: each class's mean pot share at the first decision over a stored spot's 552-1755 flops, for both players, as pairs `eq + a[i] - c[j]`; `scripts/pf_solve_library.py`): 3.14 points with the river correction (3.81 without), 15 spots and 12 facts out, as close as the charts or closer only at the big blind against a HJ open and the button against a CO open or a small-blind 3bet, and its hand order worse (the button opens A2o and folds A4o: the classes just outside a stored range are valued by the measured model). So no 100bb charts were generated from it. Solved for other games (20-100bb, antes, 7-9 handed tables, a straddle, heads-up) it gives those games' charts, as moves from its own 100bb solve (Preflop, above). The advice uses the realization model at run time in one place: re-priced calls (rake, another size) are re-ranked by realized equity (Advisor._priced, below), with the fast flop model's river correction (connectors and gappers +0.15, 22-66 +0.10, other suited hands +0.10, other offsuit hands -0.05 in single-raised pots, half that in 3bet pots; advisor._river_fix). |
| Preflop solves for Study | `gto/pf_study.py`, `scripts/build_pf_study.py` | The page's game (players, stack, ante, straddle, rake; raise sizes from the Preflop solve's form, `pf_sizes`) as `pf_game`'s tree, solved in chunks in the browser's worker to 1.5 mbb/hand a seat (2-4 handed to 2,000 iterations; DCFR, the average weighing iteration t by t³), the standard 6-max 100bb games stored (`gto/static/pf/`); each node's answer from the average strategy and its counterfactual values: per class the mix and each action's EV relative to folding (`V[child] / ` the others' card-removal masses `+` chips in), the range's and actions' EVs weighted by card removal, the node's own loss to best actions (`regret_bb`). Study's Solver source (Study, above). |
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
  | browser Fast (8 s) | 7.1 (3.8-9.5) -> 1.0 (0.5-1.5) -> 0.47 (0.34-0.55) | 7.9 -> 3.1 -> 1.8 | 3.1 -> 0.3 | turn 33/75/125%, river 33/125%, classes |
  | browser Accurate (60 s) | 2.3 (0.6-3.8) -> 0.18 (0.08-0.33) | 5.7 -> 0.7 | 3.1 -> 0.3 | 33/75/125%, classes |
  | native Fast (2.5 s; turn 5 s with the flop free) | 1.9 (1.4-2.9) -> 1.0 (0.5-1.8) -> ~0.47 | 5.8 -> 3.4 -> ~1.8 | 3.1 -> 0.3 | turn 33/75/125%, river 33/125%, classes |
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
  1.6-3.7 off (it has no overbet) and 33/75/125% 0.75-2.65 against 33/125%'s 0.54-1.42.
  The compiled core (see Compiled core) then made an iteration 2.8-3.6x cheaper natively and 5.5-6.7x in the
  browser, and the error left was the menu: 33/125% with the river at 75%, converged (~280 iterations at the browser's
  8 s on a loaded machine, 0.05-0.14% of the pot from equilibrium), was 0.58-1.48% of the pot off per combo on the
  five single-raised turns, 1.02 on average, 3.2 points of frequencies. Fast's single-raised turn now bets 33/75/125%
  with the river inside at 33/125% (2.6x the cost an iteration: ~105 iterations in the browser on a loaded machine, ~150
  natively in 5 s, 0.3-0.8% of the pot from equilibrium): 0.34-0.55, 0.47 on average, 1.8 points (33/125% on the turn
  with 33/75/125% on the river: 0.48 and 2.9; 33/125% on both: 0.62 and 2.5; the turn's three sizes with the river at
  75%: 0.97). A 3bet pot's shallow turn keeps 33/75% with the river at 75% (0.61; with the river at 33/125% 0.57, the
  single-raised menu 0.68, one board 1.1-1.8 either way). Then threads (see Threads) gave 2.5-3x the iterations,
  and with the core on 4 or more (`native.THREADED_FAST`: the server's `--threads`, the browser's helpers) Fast's turn
  plays Accurate's tree, the river inside at 33/75/125% and shallow turns too. In the core, per-combo EVs against the
  rich reference on the five single-raised turns (mean, % of the pot) at 150 / 200 / 300 / 450 iterations: Accurate's
  tree 0.31 / 0.21 / 0.12 / 0.07, Fast's own 0.38 / 0.35 / 0.34 / 0.34 (it settles there: its river has no 75%); the
  four 3bet turns 0.82 / 0.45 / 0.16 / 0.13 where the shallow menu stays at 0.69 (200) and 0.59 (450). Accurate's
  tree costs 1.4x an iteration in single-raised pots and 2.2x in 3bet pots', so it pays from ~200 of Fast's
  iterations; on one thread Fast keeps its own. A river root (Advisor.root_sizes,
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
  iterations in 60 s (0.03-0.05% of the pot from equilibrium in its game, a 115-165 MB heap; in the compiled core
  its 1000-iteration cap in 16-31 s, 0.01%), and at 400
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
    against 74), so Fast got 140-210 iterations natively and 130-420 in the browser (measured: 217-421 on
    K72 by machine load, 133-142 on 986 two-tone), where it had ~200 and
    ~100 (setup, before the budget: 0.6-1.0 s natively, 1.1-1.6 s in Pyodide, against 0.6 and 0.7; the turn
    leaves' showdown sums now take one comparison and a byte add per board, and every board's hand
    strengths one batched evaluator call). The compiled core now runs it (Compiled core, below): an iteration
    is 7-20 ms in Pyodide and 6-24 ms natively, 2-16x numpy's iterations in the browser's 8 s, and it builds the
    bucket and river tables too (the setup 0.2-1.0 s in Pyodide, where numpy's tables made it 0.6-2.6). Against every
    turn played (12 flops above), at the iterations each gets: c-bet 3.6 points off
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
* Per-combo EV errors against a reference count the actions each combo plays (its mean frequency over 5%) or its EV
  at its own mix, not every action: equilibrium doesn't pin the EV of an action a combo never takes. Two solves of the
  same turn run to 1500 iterations from different starts agree to 0.01-0.04% of the pot on those, and differ by up to
  1.0% on the unplayed actions, so a maximum over every action favors whichever start the reference's own run shares.
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
  seconds on a 10-core laptop (4 performance cores). Measured again with the compiled core (2026-10-29, two flops a
  spot, the library build paused, other jobs running; the table before it, NumPy's, in git history):

  | street, speed | n | ours: iterations, s | ours: % of the pot from equilibrium | EV given up, OOP / IP | root mix diff, points | theirs, 6 threads: s to 0.005% (river), 0.05% (turn) | theirs, 1 core: s to ours |
  |---|---|---|---|---|---|---|---|
  | river, Fast | 10 | 1000, 0.17 (0.06-0.34) | 0.014 (0.003-0.018) | 0.008 / 0.009 (max 0.016) | 2.8 (1.0-4.3) | 0.39 (0.05-0.89) | 0.18 (0.05-0.48) |
  | river, Accurate | 10 | 1000, 0.21 (0.06-0.28) | 0.013 (0.004-0.031) | 0.006 / 0.010 (max 0.028) | 3.2 (0.9-6.4) | 0.55 (0.06-0.88) | 0.23 (0.06-0.36) |
  | turn, Fast, single-raised | 8 | 81 (54-105), 4.8 | 0.93 (0.61-2.72) | 0.19 / 0.25 (max 0.50) | 9.6 (0.0-23.3) | 13.9 (11.0-27.2) | 9.1 (4.5-11.4) |
  | turn, Accurate, single-raised | 8 | 334 (205-435), 29.8 | 0.14 (0.06-0.41) | 0.010 / 0.010 (max 0.076) | 1.0 (0.0-13.0) | 25.6 (12.9-33.8) | 40.7 (24.5-61.1) |
  | turn, Fast, 3bet pot | 3 | 863 (579-1000), 4.9 | 0.007 (0.004-0.029) | under 0.005 | 1.3 (0.1-7.3) | 0.7 (0.2-1.8) | 2.2 (0.9-6.8) |
  | turn, Accurate, 3bet pot | 3 | 1000 (992-1000), 14.8 | 0.008 (0.005-0.019) | under 0.005 | 2.5 (0.0-6.4) | 3.3 (0.7-7.7) | 10.2 (2.0-22.3) |

  Rivers are exact: their best responses to our whole river strategy measure it at our own number to 4 decimals (the
  two codes' exploitability agree), each player gives up under 0.03% of the pot, the root's range frequencies are
  within a few points. On one core their code now needs as long as ours for our river accuracy (0.18-0.23 s against
  our 0.17-0.21 s for 1000 iterations; NumPy took 2 s, 13x theirs), 1.9x as long on single-raised turns at Fast (9.1 s
  against 4.8) and 1.4x at Accurate (40.7 against 29.8): about as many iterations, each 1.5-2x cheaper here, as our
  river classes play 13-24 cards where theirs deals 44 (per card theirs is still 1.5-2x cheaper). On 3bet turns theirs
  gets there in 2.2-2.9x less time, but ours runs on to its budget or 1000 iterations at 0.004-0.03% of the pot (150-570
  of their iterations, 579-1000 of ours), where neither player gives up anything measurable. Fast single-raised turns
  are the weak cell: Fast's menu (33/75/125%, the river inside at 33/125%) costs 2.6x the old one-size tree an
  iteration, so on this loaded machine 5 s ran 54-105 iterations (the old tree in NumPy 82-110), 0.6-2.7% of the pot
  in our game, and gives up 0.19 / 0.25% of the pot (max 0.50) where the one-size tree gave up 0.09 / 0.08 against
  the same every-card game (the menu's own gain, 0.6-1.5 -> 0.3-0.6% of the pot per combo against the rich
  reference, is measured there: gto/native.py). The two turns where beta 0 swung late (Regret minimizer, below)
  answered at 0.99 and 1.48% of the pot and gave up 0.18-0.34; with the beta step 0.23 and 0.23, giving up 0.01-0.05.

  Per iteration on one core (`periter`: the Advisor's Fast tree and ranges on Ks7d2c, BTN vs BB and BB's 3bet, 300
  iterations each after 20, no exploitability read between them; 2026-10-30), ms, ours / theirs:

  | spot | ours | theirs | exploitability after 300, % of the pot (each in its own game) |
  |---|---|---|---|
  | turn Th after a c-bet called | 14.5 | 54.7 | 0.117 / 0.124 |
  | turn Th, flop checked through | 17.4 | 62.4 | 0.169 / 0.158 |
  | turn Th, 3bet pot | 2.30 | 6.23 | 0.059 / 0.049 |
  | river 5s after turn bets called | 0.131 | 0.212 | 0.068 / 0.064 |
  | river 5s, checked through | 0.112 | 0.178 | 0.099 / 0.111 |
  | river 5s, 3bet pot | 0.020 | 0.032 | 0.010 / 0.010 |

  On a turn theirs deals all 48 river cards where ours plays 13 river classes, so per card dealt the two cost the same
  (1.12 ms a class, 1.14 a card); a river is the same game in both, 1.6x cheaper an iteration here. (The 2.5x and 13x
  theirs per core measured when this benchmark was first run were NumPy's, before the compiled core.)

  Flops: a library entry's stored decisions (the root, IP after a check, OOP facing a c-bet after checking) locked
  into their full game of the library's tree (every turn and river card, where the entry plays turn classes and the
  river blind on the turn's boards), the rest solved to 0.15% of the pot (their game takes 0.5-5 GB and 40-430 s on
  all cores) and, for the 3bet pots, to 0.03%:

  | spot, flop | entry, % of the pot in its own game | EV given up, OOP / IP | root, ours / theirs |
  |---|---|---|---|
  | BTN vs BB, Kc7d2h | 0.111 | 0.40 / 0.48; the river dealt (0.050): 0.18 / 0.25 | check 100 / 100 |
  | BTN vs BB, JcTd8d | 0.089 | 0.29 / 0.17; the river dealt (0.045): 0.04 / 0.07 | check 97 / 100 |
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
* **Compiled core** (`core/gtocore.c`, How it works; `scripts/bench_browser.mjs --spots river,turn-lib,turn`, the
  worker with and without the core, one after the other on a 10-core machine busy with a 5-process library build).
  Each cell: seconds to the answer, iterations, % of the pot from equilibrium; the river's solve alone in brackets
  (the rest is its turn, solved to narrow the ranges). The turn gets 5.5-6.7x the iterations in the same budget, and
  the rest of a turn's answer (EVs, exploitability, the read-off) takes 0.3-0.8 s where it took 1.2-3.8. The flops
  (`--spots flop,flop-2tone`, both in the library now, so with a worker whose library fetches fail; the machine busy
  with an 8-process build) are solved live by hand buckets (below): Fast gets 6.5-10x the iterations, and Accurate
  reaches its 1000-iteration cap in 16-31 s of its 60:

  | request (6-max 100bb) | numpy | core |
  |---|---|---|
  | turn, flop from the library, Fast | 12.1, 42, 2.19 | 11.2, 282, 0.07 |
  | turn, flop from the library, Accurate | 61.6, 91, 0.93 | 60.5, 486, 0.07 |
  | turn, flop solved to narrow, Fast | 11.9, 51, 1.49 | 11.1, 278, 0.08 |
  | turn, flop solved to narrow, Accurate | 62.8, 87, 1.01 | 60.4, 448, 0.07 |
  | river, Fast | 14.1 (10.4), 838, 0.01 | 3.7 (0.51), 1000, 0.01 |
  | river, Accurate | 25.8 (13.3), 1000, 0.01 | 12.7 (0.52), 1000, 0.01 |
  | flop K72r, Fast | 8.3, 62, 2.44 | 8.2, 646, 0.03 |
  | flop K72r, Accurate | 60.4, 587, 0.03 | 16.4, 1000, 0.01 |
  | flop 986 two-tone, Fast | 8.2, 35, 7.11 | 8.2, 229, 0.16 |
  | flop 986 two-tone, Accurate | 60.5, 430, 0.05 | 31.3, 1000, 0.01 |

  The site built locally (`BUILD_ONLY`) in Chrome, the same machine: the turn after a library flop 628 iterations in
  10.9 s, 0.02% of the pot from equilibrium, the river's solve 0.3 s (3.4 s with its narrowing turn).
* **Threads** (`core.threads`; the browser's helpers need the page cross-origin isolated, `gto/static/sw.js`). The core's
  iterations alone on a Fast turn (Qs8d3c4h, wide ranges, 13 river classes), natively, the library builds paused:
  17.3 ms an iteration on 1 thread, 11.0 on 2, 6.9 on 4, 5.9 on 8 (Accurate 20.3 -> 7.1); a split has as many rows
  as river classes (13 on a rainbow turn), so 4 threads can't beat 13/4 rows a round. In Chrome (the site built
  locally, 8 threads), the turn after a library flop at Accurate with a 15 s budget: 211 iterations, 0.28% of the
  pot from equilibrium, with the library build paused; 191, 0.32% beside an 8-process library build, where the page
  without isolation (1 thread) ran 80, 1.2%. With every core already busy (the build, two test runs) threads gain
  nothing: 153 iterations against 148-152 on one. The machine has 4 performance and 6 efficiency cores, which caps 8
  threads near 3x. Running a whole pass as one job (the chance nodes and the terminals above them collected as
  tasks, run, then their results taken on a second walk down) instead of a job per chance node was no faster: 193-202
  iterations against 211 in Chrome, the same natively. (A task there was a whole chance node; one job of every first
  chance node's rows is faster: The first chance nodes in one job, below.)
  Each thread then got ~1.3x faster (1 thread, side by side on a loaded machine: Fast 19.0 -> 14.1-14.5 ms an
  iteration, Accurate 24.9 -> 18.4-19.0): a showdown's running sums run as four chains at once (the strength order's
  quarters, each one's offset added after), each card's card-removal term is taken once a board instead of for each
  combo holding it, the opponent's total reach is summed in four lanes, and regret matching multiplies by one
  reciprocal a combo. numpy sums and divides the same way (`solver.quarter_cumsum`), so the two still round alike;
  that costs numpy ~12% of a turn iteration (four running sums where it took one: 1.3-7x a plain cumsum by row
  length), in the solves the core doesn't run (locks, sampled rivers). Without the quarters the rest gained ~3%.
  Natively (`cc -O3`, against the BLAS solver on the same loaded machine) a river runs 9.8x numpy's iterations a
  second and turns 2.8-3.6x; the passes after a solve 3-10x. Where the time goes in the core (a Fast turn, before the
  per-thread work): river showdowns ~38% (the sorted strength order's running sums, then each combo's card-removal
  corrections), regret matching ~20%, decisions' sums ~13%, the opponent's card-removal sums at folds and leaves
  ~10%, the discount ~5%.
  On the flop it runs Fast's model (the browser's Accurate's too: the turn by hand bucket, the river by strength in
  single-raised pots, in the turn's buckets in 3bet pots), whose turn and river are products of each turn board's
  bucket tables with the opponent's bucket reaches (32 x 32 in Fast; summed column by column, as numpy's matmul does
  without BLAS) and a turn bucket's reach spread over the river buckets and gathered back (`Ctx.river_table`'s X and
  its transpose). `scripts/bench_flop_core.py` (natively, or in Pyodide through `scripts/pyodide_run.mjs`, which loads
  `gto/core.wasm` as the worker does; `GTO_CORE=0` leaves it out): 8 flops of the HJ vs BB single-raised pot the
  library doesn't store (three rainbow, three two-tone, a monotone and a paired one, a seeded draw), numpy then the
  core back to back on the same loaded machine (10 cores, 8 of them building the library). In Pyodide, in the
  browser's 8 s, the core runs 2.0-15.6x numpy's iterations (geometric mean 6.6x): 9c8d5h 110 -> 986 (0.60 -> 0.015%
  of the pot from equilibrium), QcTd6c 65 -> 469 (1.76 -> 0.032%), QcTc6d 68 -> 475 (1.52 -> 0.035%), Jc5c2c 80 -> 511
  (1.47 -> 0.037%), Tc6d3h 29 -> 273 (11.5 -> 0.10%), Ac9d4h 20 -> 311 (23 -> 0.078%), 3c2d2c 59 -> 117 (2.7 ->
  0.56%), Ac9d5c 10 -> 45 (94 -> 3.8%). An iteration takes 7-20 ms there; what limited the last two was the setup
  before it, the bucket and river tables (`Ctx.bucket_table`, `river_table`, the flop all-in's `_flop_leaf`) in numpy,
  2.0-4.4 s of the 8 (since built in the core: below). Natively (2.5 s, against the BLAS solver, the same load) 32-215
  iterations where numpy ran 10-52 (3.1-83% -> 0.15-10.7%; 3.7x), the setup most of the budget there too. The same
  answers: at 50, 100, 200 and 400 iterations (`--checkpoints`, natively) the core is as far from equilibrium as numpy
  at each (at 400 0.044-0.067% of the pot, numpy 0.043-0.068%), 2.1-4.7x sooner (3.5x), and at 300 the c-bet's
  frequencies (the check and each size, every flop) are numpy's to 0.2 points. With threads (above) a flop's turn
  classes are the rows handed out, the numbers the same bit for bit on any number of them.
  The core builds those tables too (`gto/core.py` `Tables`, before it lays the tree out: the showdown counts in 16-bit
  lanes, eight columns at a time; each river's strength order swept once a player for its buckets and once more with a
  column per opponent bucket for their sums), 7-10x numpy natively, and they are numpy's: every combo in the same
  buckets (equities to 2^-20, `solver.eq_key`, so that ties, such as every hand at -1, go in combo order however the
  sums were rounded), the sums to float32 rounding. In Pyodide the setup to the first iteration is 0.17-1.0 s where it
  was 0.64-2.6 (3.6-5x less; Ac9d5c 0.34 s: the river boards' strengths 0.14, the tables 0.06; at a load of ~20
  2.4-9.2 s -> 0.6-1.8). In the browser's 8 s that is 1.3-7.5x the iterations at a load of ~18 (2.4x; Jc5c2c 37 ->
  279, 7.9 -> 0.13% of the pot; QcTc6d 98 -> 442, 0.69 -> 0.045%) and 1.05-1.33x idle (1.16x, at 1300-3300
  iterations); natively, loaded, 2.5 s run 1.1-2.2x (1.6x).
  The library's flop (`flop_library.library_config`, which `scripts/build_flop_library.py` solves: played per combo to
  the turn's equity leaves) runs in the core natively, its leaves batched (the compiled core, above). A library entry
  (300 iterations, the exploitability checks, the encoding), one process at a time, the core and numpy (one BLAS
  thread, as the build runs it) alternating: idle 2.1-2.6x faster (2.3x; srp_sb_bb Tc7d2d 40.3 -> 17.9 s, Kc8d6c
  44.5 -> 17.0, srp_hj_bb Kd7c2h 16.1 -> 6.5, Qh6s3s 54.0 -> 26.0, srp_utg_bb 9c8d5h 12.7 -> 5.8, As5d5c 36.0 -> 14.1,
  the last two as an 8-process build resumed), beside the build 2.3-2.9x (2.5x; Tc7d2d 135.5 -> 59.2 s, Kd7c2h 51.5
  -> 18.8, 9c8d5h 40.6 -> 14.2). The same entries: the exploitability numpy's to 0.006% of the pot (0.078-0.142%), the
  stored range frequencies within 0.24 points, per combo 0.2-0.3 on average (an indifferent hand up to 7). An
  iteration of Kd7c2h takes 59 ms where numpy's takes 168-190; on the core's own loops instead of Accelerate's sgemm
  150. Where its time goes: regret matching ~27% (a node's strategy is taken three times an iteration: both passes and
  the walk down), the card-removal sums at folds and leaves ~24%, the decisions' sums and updates ~18%, the rest of
  the folds and leaves ~10%, the products ~10%, the walk down ~7%, the discount ~6%. Regret matching's last loop
  hadn't vectorized (a load under a condition; it now takes the product either way, the same floats): 16% of this
  iteration (67 -> 56 ms side by side). Five of the single-raised pots' 2688 entries (srp_utg_bb AcQdTh among them)
  ran there to 23-73% of the pot where numpy's solves reach 0.1: rounding (numpy's own two leaf paths part as much)
  left the sum of a hand's positive regrets at an all-in node its opponent all but never reaches at 5e-41 by
  iteration 87, under float32's smallest normal, and its reciprocal at inf, a zero regret times it NaN. Regret
  matching, the core's and numpy's, now takes a sum under 2^-126 for none (uniform), the same floats otherwise; the
  five are 0.093-0.139% of the pot at 300 iterations, numpy's 0.094-0.140.
  A single-raised flop with the river dealt (`Config.river_dealt` 32,
  srp_btn_bb JcTd8d, 32M regrets) runs there too: 1.4 s an iteration where numpy takes 3.8 (side by side, loaded),
  in 1.1 GB where numpy took 1.8. A 3bet pot's entry, its river dealt in 64 buckets, gained less (3bp_bb_btn Kd7c2h,
  200 iterations, idle, 34.5 -> 26.5 s): ~2/3 of the core's time went to the river's bucket games, every card's under
  every turn line, each terminal reading its card's tables (E and kap, 32 KB a player) again, out of cache. Those
  games are now deferred (the compiled core, above), the regrets and strategy sums the same bit for bit: an entry
  beside the 7-process build, side by side with the core before, 2.2-2.9x (4c3d2h, 300 iterations, 167.5 -> 58.4 s;
  Kd7c2h, 200, 156.7 -> 71.0 s, 120.6 -> 42.3 s of CPU; 6c5d4h, 200, 156.7 -> 59.3 s, at a lighter load 102.0 ->
  40.0). Where its time goes now: the cards' bucket sums and reads ~25%, the products ~23%, regret matching ~17%, the
  games' updates ~13%, the discount ~7%.
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

  Later (3- and 4-way flop and turn solves' iterations in the compiled multiway core, `gto/mwcore.wasm`; CO opens, BTN
  and BB call, Kh8d3c; an idle machine): a 3-way flop first to act at Accurate 16.8 s, 400 iterations, 0.05% (was its
  60 s, 63-77, 3.3-5.0), at Fast 8.1 s, 203, 0.25% (was the rules'), at Quick 1.8 s, 114, 0.43% (the rules'); checked to
  twice Fast 5.0 s, 400, 0.03%, Accurate 7.4 s, 400, 0.04% (60.1, 161, 0.27); a 3-way turn Fast 7.1 s, 400, 0.04% (the
  rules'), Accurate 7.2 s, 400, 0.04% (34.0, 200, 0.07), Quick 1.8 s, 87, 0.58%; a 4-way flop (HJ opens, CO, BTN and BB
  call, Kh7d2c) Fast 5.8 s, 200, 0.03%, Accurate 8.6 s, 200, 0.05% (the rules'); a 4-way turn Fast 1.3 s, Accurate 4.4
  s, 100, 0.04%. The heap peaks at 566 MB (the Accurate 4-way turn: 16 buckets a player over four).

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
* **Precomputed answers, instant in the browser** (`scripts/instant_coverage.py`: of a 6-max 100bb session's
  decisions, every seat's, 83.6% are answered from precomputed data: preflop all of them, heads-up flops 74.7% from
  the flop library, 3-way flops 5.2%, turns and rivers none yet; the turn and river are 13% of the decisions, the
  flop 12%; `--assume` and `--candidate` weigh a build before it runs, a flop with turn files answering the turn's
  first decision and the in-position player's after a check, 69% of a turn's decisions in 12 native Fast turns).
  The answer itself takes 10-40 ms in Pyodide; what it waited for was the library's files, read
  synchronously when the request came (the index on a session's first flop, 0.8-4.9 s on a cold CDN; each entry
  0.13-0.67 s from here), and on the first multiway preflop spot (a big blind facing an open and a call, or a 3-way
  flop's ranges) the preflop equity tables, 0.5 s. Now the worker fetches both indexes as it starts, beside the
  runtime; the page tells it the hand's flop as soon as it is dealt, whoever is to act (a `prefetch` message), and
  `Advisor.prefetch` runs the flop's lookups dry, so the worker fetches the entries they name ahead and a request
  waits for any still in flight instead of asking again; and while idle the worker builds the equity tables 50 ms at
  a time (`advisor.warm_step`, then the modules answers import on first use). `scripts/bench_browser.mjs --session
  --idle 3 --prefetch 1` (one worker, the first request 3 s after its setup, the flop told 1 s before each
  request with one, `BENCH_NET_MS=300` per library file), first answers in a session, medians of 3, seconds:

  | | preflop | BB vs open and call | BTN vs BB flop | CO vs BB flop | 3bet pot flop | 3-way flop |
  |---|---|---|---|---|---|---|
  | before (synchronous fetches) | 0.01 | 0.51 | 0.62 (2 fetches) | 0.31 (1) | 0.31 (1) | 0.64 (2) |
  | after | 0.01 | 0.10 | 0.01 | 0.01 | 0.01 | 0.01 |

  The same answers (iterations and exploitability are the library's: 0.07-0.13% of the pot). The flop's entries
  arrive ahead when anyone acts before you; first to act on the flop, the request waits for the entry's one fetch
  (the index is in hand). The tables cost the same 150 MB of heap as before, now in every session.
* **Fast stops once settled; the river reads the hand's turn solve** (`Advisor.hu_settled`, `advisor.HU_CHECKS`,
  `HU_SETTLED_FROM`). The browser's Fast ran each heads-up street's solve for its whole budget however settled it
  was: a 3bet pot's turn on to 1000 iterations, 0.01% of the pot from equilibrium, a river to 0.01-0.02%. Now it reads
  the exploitability at 50, 100, 150, 200, 300, 400, 600 and 800 iterations and stops at the first under its target,
  0.3% of the pot on one thread and 0.2 with threads (Accurate's turn tree there: ~300 iterations, 0.12% of the pot
  off per combo against the rich reference on the five single-raised turns, 0.07 at 450). Not before the 300th:
  per-combo EVs settle after the exploitability does. Against the rich reference (`scripts/turn_river_reference.py`,
  natively, % of the pot off per combo), the four rivers at 100 iterations (0.15-0.45% from equilibrium) are 0.21-0.99
  off where the 1000th is 0.11-0.60, and a 3bet pot's turn at 150 (Th9h4c-2s, 0.27%) 1.60 against 0.68; at 300 the
  rivers are 0.16-0.61 and the four 3bet turns 0.45-0.69 in one thread's shallow menu (0.47-0.68 at 1000) and
  0.11-0.28 in Accurate's tree (0.12-0.28), all within 0.1 of the 1000th's. A deeper turn on one thread may stop
  from the 200th: Fast's own menu leaves it 0.34 off however long it runs (0.35 at 200). A settled solve asked again
  isn't run on; Accurate and the local server run their budgets as before. The river's ranges came through a second
  solve of the turn (3 s of the river's budget, 14-36 iterations: 6-37% of the pot from equilibrium, so mostly the
  betting rules), its ranges with the river card dead, so the turn advice's solve wasn't found; now they come
  through the hand's turn solve. `scripts/bench_browser.mjs` (one worker, one thread, Fast, before and after taken
  in turn on a loaded machine), seconds, iterations, % of the pot from equilibrium:

  | | 3bet pot turn | river after the hand's turn | river asked cold |
  |---|---|---|---|
  | before | 11.3 (723, 0.01) | 14.1 (607, 0.02) | 14.1 (673, 0.02) |
  | after | 5.0 (300, 0.06) | 0.52 (300, 0.06) | 3.5 (300, 0.07) |

  With the browser's threaded settings (Accurate's turn tree; the bench's core on one thread) the river after the
  hand's turn went 4.2 -> 0.29 s. A single-raised turn on one thread stops at 200 iterations, 0.23-0.25%, in 7.9-8.1
  s where it ran 11.1 s to 288-311 (0.11-0.13%) on a quieter machine; loaded, it gets 124-140 iterations in its 11 s
  and runs them all, as before. The page can also stop the worker's solve now: where it is cross-origin isolated
  (the site, under its service worker) the page and the worker share a stop flag (`stopRequests`, a
  SharedArrayBuffer the core's chunks read every ~0.1 s), so Stop, the hand, speed or opponents changed mid-solve,
  or a Practice look-ahead your move made useless stops it within ~0.1 s and the worker keeps its runtime, solves, library entries
  and tables (before, the page restarted the worker: its 2-4 s setup again, and everything it held gone). A solve
  stopped part way stays cached and goes on from there if asked again, and Practice looks ahead at Accurate too.
  The river's narrowing still solved the turn again when their turn bet was off the tree the turn's advice solved (a
  limped pot's BB betting 60% where it has 33/75/125%): the sizes the hand took go into the narrowing's tree, so its
  key missed the advice's solve, and the second solve took 3.1 s of the river's 4.75 in Pyodide (`--session` with the
  limped pot's flop, turn and river; natively 3 s of 3.2, 38 iterations where the advice's had settled at 200). Now `Advisor._hand_cfg` finds the hand's
  own solve of the street from an earlier decision (each earlier point of the street's action, the latest first), and
  the narrowing walks it at the size nearest the bet ("turn: mapped 'bet 49%'"): that river 0.11 s, natively 0.13.
* **Quick keeps to its time and memory** (`advisor.MAX_CELLS`, `CELL_S`, `CAP_ITERS`, `TRIMS`, `solver.tree_cells`,
  `TooSlow`, `advisor.MW_QUICK_BUILD`, `multiway_solver.BuildTooSlow`). The coverage fuzz found Quick answers at 2-8x
  their estimate. Heads-up turns with wide ranges took 11-14 s against 1.8 s and up to 2.1 GB: a limped 200bb tournament
  pot with three raise sizes, a plain HJ vs BB single-raised turn after the flop checked through, and limped 55bb pots
  with custom sizes. A 3-way flop with a player all-in and a 4-way limped flop took 6.5-8.8 s against 1.5, building
  tables they then didn't iterate. Solver.run's 10-iteration floor ran whatever the tree was, and every turn solve kept
  48 float showdown matrices (one a river) for its all-in leaves. Now:
  - The turn's equity leaf over every river is `_avg_showdown` on the rivers' hand strengths, in the compiled core
    while it lays the program out: the same floats (tested bitwise against the sum of the 48 matrices), 0.6-0.9 s ->
    0.07-0.14 s natively, and ~250 MB less on a limped pot's 1,128 combos.
  - A new heads-up solve's tree is counted first, without its arrays (cells: its decisions' boards x actions x
    combos), and cut until it fits under 24M cells for memory (20-27 bytes a cell: ~0.6 GB; Accurate's limped turns
    are ~20M, so Fast's and Accurate's usual trees aren't cut) and, on the turn and river, under what runs 30
    iterations in the budget at 8 ns a cell (a big tree's rate: a limped turn's 13.75M cells take 72 ms an iteration in
    Pyodide; at the floor's 10 that tree ran 19 iterations in Quick's budget, 38% of the pot from equilibrium). The cuts,
    in order: every street's raises to one size; the later streets' bets too; this street's; one raise a street. The
    sizes the hand has used are kept, and the answer says its game is smaller and offers fewer sizes. A first iteration
    showing the floor would take twice the budget sets the device's speed from it and cuts again.
  - A quick answer's 3- or 4-way tree and tables not built by 75% of its budget (or on course not to be, a quarter of
    the way through) go to the betting rules at once, and the rules' trials take what's left of the budget (at least
    0.4 s), where they took up to 1.5 s after the solve.

  `scripts/bench_browser.mjs` in Pyodide (one thread, Quick's 1.5 s, a fresh worker's first request / a worker idle
  6 s; seconds, iterations, % of the pot from equilibrium in the game solved, heap):

  | turn | before | after |
  |---|---|---|
  | limped 200bb tournament pot, raises 50/66/150% (cut to one raise size, one bet size later) | 8-12 s at 2.1 GB (the fuzz), 62% | 2.65 / 2.0 s, 25-31, 8-11%, 328 MB |
  | 6-max limped pot (the same cut) | 1.96 s, 10-19, 42-85%, 566 MB | 1.89 / 1.89 s, 33-36, 12-13%, 227-328 MB |
  | BTN vs BB single-raised (not cut) | 2.2-3.2 s, 10, 55%, 328-393 MB | 1.94 / 1.89 s, 42-64, 1.7-4.2%, 190-273 MB |
  | HJ vs BB single-raised, flop checked through (cut to one raise size, one bet size later) | | 1.91 / 1.88 s, 55-60, 1.4-1.6%, 190-273 MB |
  | the same facing a raise war's all-in (its turn solved first to narrow the ranges) | 11.8 s at 1.95 GB (the fuzz) | 2.0 / 1.99 s, 472-566 MB |

  The fuzz's slow records asked again natively in the worker's configuration (and the server's): the HJ vs BB all-in
  3.79 s, 1074 MB -> 1.78-1.84 s, 555-658 MB; the limped 55bb pots 3.10-3.94 s, 843-1108 MB, 42-46% -> 1.82 s,
  171-175 MB, 47-156 iterations, 0.14-1.44%; the 4-way limped flop 3.7-6.3 s (7 iterations, 22%, then the rules) ->
  1.68-2.28 s by the rules at once. The 3-way flop with a player all-in (8.8 s in the fuzz) had solved natively in
  1.81 s, 71 iterations; its tables pass the 75% mark natively too now, so the rules answer it in 1.48-1.60 s. Of 52
  Quick heads-up turns and rivers asked again (the fuzz's slowest 36 and 16 at random; 20 of them over twice their
  estimate, up to 13.6 s), 17 were cut; none is flagged (the fuzz's checks: a move the page can't play, an error, the
  catch-all) and none is over twice its estimate, the slowest 3.45 s against 1.8 (a rare line solved again from its
  node: two budgets).
* **The browser keeps its compiled core all session** (`core._clock`, `core.PER_MIN`, `core.CoreError`). `core.run`
  timed its chunks of iterations with `time.time()`, which in Pyodide is `Date.now()`: whole milliseconds. A small
  river's first iteration (0.2-0.3 ms: a limped pot's or a single-raised pot's narrowed river, 0.1M cells) read as 0 s
  whenever it didn't cross a millisecond tick, the chunk size was divided by it, and the ZeroDivisionError, caught as
  if the core had failed, switched the core off for the worker: every later solve of the session (the next hands,
  Practice's look-aheads, Study) ran in numpy, 5-7x fewer iterations a second. `scripts/bench_browser.mjs --session`
  (Pyodide, one worker, Fast: a limped pot's flop, turn and river, then a BTN vs BB turn) lost it in 7 of 14 sessions:
  the river's own solve 1.66 s where the core takes 0.1, and the turn after it 11.3-11.4 s, 82-84 iterations, 1.25-1.3%
  of the pot from equilibrium, where it takes 4.45 s, 200 iterations, 0.23%. With the fix 7 of 7 sessions kept it.
  Now the chunks are timed with `time.perf_counter` (`performance.now()` there: microseconds), an iteration is never
  taken for less than 0.1 us, and only the core's own failures turn it off: a trap in the browser's module (its memory
  refusing to grow, an access out of bounds) or a run or table build that didn't finish (`CoreError`). An error in the
  Python around it runs that one solve in numpy, from the iterations it had, and leaves the core on. The 3- and 4-way
  solver's chunk clock gets the same floor. Natively the clock is microseconds, so the native numbers (the coverage
  fuzz's browser-advisor mode included) are unaffected, and so are single requests on a fresh or idle worker, whose
  solves before the turn's take over a millisecond an iteration: Quick's table above didn't move (main and the fix
  taken in turn: the same iterations within 2 and seconds within 0.04; more iterations than the table's, the machine
  quieter now: BTN vs BB 1.86 s, 75-77, 1.0-1.1%; the 6-max limped pot 1.87-1.89 s, 52-53, 6.0-6.2%). The settled-stop
  table's "river after the hand's turn" (0.52 s, 300 iterations) had run its river in numpy: 0.10 s in the core.
* **An approximate answer at once while the solve runs** (`Advisor.instant`, `instant_sources`: `instant_from_own`,
  `instant_from_prev`; the worker's `early` request, the page's `renderEarly`). A heads-up turn after a live flop
  solve (a limped pot, a 3bet pot's flop not in the library) waits for its own solve, Fast's 6.7 s on a limped
  pot's turn in the browser, with only the progress bar to look at, while the worker already holds a solve that
  plays the street. Now the worker first reads an answer off what it holds and posts it before the request's solve:
  this street's own solve, stopped part way or made for your earlier decision on it (facing a bet off the tree your
  advice solved: the sizes since read at their nearest), else the previous street's live solve, which plays this street
  under it (the flop's by card class and hand buckets at one bet size; the turn's for the river by card class at its
  river sizes). The page shows it under the progress bar, the verdict and the mix with "approximate" and where it
  came from (and a cut solve's "smaller game" note), until the answer replaces it; nothing is solved for it, it has
  no EVs, and it isn't graded, reviewed or recorded. `scripts/bench_browser.mjs --session --early` (Pyodide, one
  thread, a fresh worker and one idle 6 s, Fast and Quick; seconds to the approximate answer / to the answer):

  | spot | Fast | Quick |
  |---|---|---|
  | limped pot's turn, after the flop's solve | 0.002 / 6.67-6.69 (the same verdict) | 0.002 / 1.85-1.86 (the same) |
  | facing a 60% bet off the tree your turn advice solved | 0.002 / 6.65-6.66 (raise to 11.89 / raise to 11) | 0.002 / 1.87 (call / raise) |
  | limped pot's river, after the turn's solve | 0.002 / 0.11-0.12 (check / bet 33%) | 0.002 / 0.11 (the same) |
  | 3bet pot's turn, after the flop's solve | 0.052-0.055 / 1.01 (the same) | 0.053-0.056 / 1.01-1.02 (the same) |

  EV given up playing the approximate mix instead of the turn's own solve (10 s, natively in the worker's Fast
  configuration; four limped pots' flops, Ks7d2c, Jh9h4c, 8c6d5s and Ad5d3c, each with a 2h and a Td turn; the small
  blind's whole range, each combo's approximate mix priced by the solve's EVs per action against its own mix; % of
  the pot):

  | the flop's solve | first to act: mean (worst) | facing 75% of the pot: mean (worst) |
  |---|---|---|
  | Fast's (300-400 iterations) | 0.06 (0.43) | 1.43 (2.62) |
  | Quick's (39-135) | 0.43 (1.60) | 1.56 (2.68) |

  First to act it is close; facing a bet the flop solve's one bet size and hand buckets cost 1.4-1.6% of the pot,
  so it stays marked approximate and the answer replaces it. Not
  offered where the answer is quick anyway or needs more than the equilibrium: preflop, library flops (and the turns
  after them, except from a stored turn: below), a street solved already, multiway pots, Practice (it grades
  your move against the answer), exploit mode, locks, custom sizes or ranges, tournaments, Study (but for a library
  flop's stored turn: below), the breakdown, a jam request.
* **The street solved while your opponent is to act** (`Advisor.presolve`; the worker's `presolve` request; the
  page's `presolveFor`, `stopPresolve`). In position, your answer can't start until you enter the opponent's move,
  and the worker sat idle meanwhile. Now, while your opponent is to act heads-up after the flop, the page asks the
  worker for the solve your next decision on the street reads: the street's solve with your cards, as your advice
  makes it, within the setting's own budget. Entering their move stops it through the shared stop flag (0.02-0.11 s
  later, or once the tree is built: a tree's build isn't interrupted, 0.3-0.4 s on these), and the advice goes on from
  where it stopped. A solve stopped part way stays cached with its iterations and their time, also one stopped
  before its first check (that was lost before); the advice gets what's left of the budget, so the work is counted
  once and the answer is the same solve: every pair below has the same iterations, exploitability and verdict at
  Fast. Its answer is never shown. A move off its tree (a size it doesn't hold) isn't read from it, not even for
  the approximate answer; the advice solves that tree as before. One at a time; never while another request is
  asked or running (your own advice still solving when you act runs on instead, and the presolve starts once it's
  answered); only while the page is visible (stopped when the tab is hidden or the page left, started again when
  it's shown); only where the page can stop it (cross-origin isolated, as the site is under its service worker);
  not on a flop the library holds (its answers are read off the entry), nor wherever the approximate answer isn't
  offered (Practice, which looks ahead on its own; Study; multiway pots; exploit mode, locks, custom sizes or ranges;
  tournaments). `scripts/bench_browser.mjs --session --gap G --presolve` (Pyodide, one thread, a fresh worker per
  hand and one idle 6 s; you in position, the opponent's move entered G s after it became theirs; seconds from it
  to the answer, today -> with the presolve, two runs each, Fast):

  | hand (each street after the one before, in one worker) | G = 0 | 2 s | 4 s |
  |---|---|---|---|
  | limped pot's flop (solved live) | 1.62-1.81 -> 1.66-1.76 | 1.60-1.77 -> 0.01 | 1.59-1.60 -> 0.01 |
  | its turn | 7.17-8.16 -> 7.59-7.60 | 7.15-7.54 -> 5.09-5.18 | 6.98-7.30 -> 3.16-3.35 |
  | its river | 0.11-0.13 -> 0.11-0.12 | 0.12-0.14 -> 0.01 | 0.11-0.13 -> 0.01 |
  | single-raised pot's turn after a library flop | 4.15-4.22 -> 4.15-4.23 | 4.19-4.55 -> 2.14-2.46 | 4.17-4.50 -> 0.20-0.55 |
  | 3bet pot's flop (solved live) | 0.77-0.79 -> 0.77 | 0.76-0.82 -> 0.35 | 0.76-0.83 -> 0.35 |
  | its turn | 0.92-0.94 -> 0.92-0.94 | 0.95-0.98 -> 0.01-0.02 | 0.94-1.04 -> 0.01-0.02 |
  | limped pot's turn facing a 60% lead (off the presolve's tree) | 8.54-9.15 -> 8.63-8.87 | 8.30-9.30 -> 8.38-8.62 (three runs) | |

  At Quick the limped hand's flop goes from 1.52-1.53 s to 0.01-0.02 with the move entered 2 or 4 s in, its turn
  from 1.85-1.87 to 0.03-0.06, its river from 0.11-0.14 to 0.01-0.03. With the move entered at once (G = 0) it's as
  fast as today everywhere; also off the tree with the move entered 0.25 s in, the presolve stopped once its tree
  was built (8.49-9.29 s, against 8.39-9.53). The 3bet pot's flop keeps 0.35 s: the opponent's check is rare there,
  so the advice solves again from it against their rule-based range, which the presolve can't know ahead.
* **The turn after a library flop** (`instant_from_store`, `STORE_UNTIL`; the library's turn files:
  `FlopLibrary.turns`, `turn_node`; `Advisor.prefetch`). A turn after a flop the library answered had nothing to
  show until its own solve answered (5.4-6.2 s at Fast in the browser here): the approximate answer reads the
  previous street's live solve, and a library flop has none. Where the spot's entry has a turn file (the index's
  `"turns"`), its flop solve's own turn strategy now answers at once, marked approximate: the turn's first decision
  and the in-position player's after a check, on the flop line the hand took (its sizes at their nearest, as the
  library's flop answer reads them), the card dealt read as its class's played card, the bet 75% of the hand's own
  pot (the flop solve's one turn size). The turn's own solve still answers the request and replaces it, with B3's
  exclusions. The worker fetches the turn file with the flop's entry once the flop is known (none fetched where the
  index lists none). It is read before the turn's own solve in hand (a presolve's, or a request's stopped part way)
  until that has run 100 iterations, then the solve is. `scripts/bench_browser.mjs --early` (`BENCH_LIB`, new: a
  library folder served first; here BTN vs BB's Kc7d2h, JcTd8d, 8c6d4c and Ac9d4d built with turn files; Pyodide,
  one thread, a fresh worker per spot, idle 6 s, the flop told 3 s ahead, 300 ms a library file; two runs; seconds to
  the approximate answer / the answer):

  | spot | today | with the stored turn |
  |---|---|---|
  | BB first on the turn after a third-pot c-bet called (QhJh, Ks7d2cTh) | none / 5.42-6.24 | 0.015 (check) / 5.73-5.79 (check) |
  | BTN after BB's check (AsQs) | none / 5.49-5.65 | 0.015-0.017 (bet 75%) / 5.52-5.60 (bet 125%) |
  | the same, BB's check entered 2 s in (the presolve, B2) | | 0.007 (bet 75%, the stored turn) / 3.61 |
  | ... entered 4 s in | | 0.003 (bet 125%, the turn's solve at 147 iterations) / 1.55 |

  Without the flop told ahead the stored turn waits for both files (0.63 s at 300 ms each). EV given up, the
  approximate answer's measure above (each combo's mix priced by the turn's own solve's EVs per action; here the
  browser's Fast tree solved to 600 iterations natively, 0.05% of the pot from equilibrium; 29 turns of the four
  flops after a check-through and after a third-pot c-bet called, five cards each where the library's flop answer
  holds the line; the acting player's whole range; % of the pot, mean (worst)):

  | | BB first | BTN after a check |
  |---|---|---|
  | the stored turn | 0.04 (0.17) | 0.08 (0.27) |
  | the turn's own solve at 50 iterations | 0.09 (0.19) | 0.06 (0.12) |
  | ... at 100 | 0.02 (0.06) | 0.02 (0.05) |
  | ... at 200 | 0.006 (0.03) | 0.007 (0.03) |

  The stored turn beats the solve's 50th iteration first to act (28 of 29 turns) and is about even with it in
  position (16 of 29), behind its 100th (13 and 5 of 29): `STORE_UNTIL`. On the classes' played cards 0.03 / 0.06,
  on the other cards 0.04 / 0.09. With the opponent free to adapt (flop-core's measure: Precomputed flop library,
  Turns) it gives up 0.5-0.8% of the pot, so it stays marked approximate. The first turn files ship with the 3bet
  pots on 552 (1,599, 15.2 MB: every BB vs SB and CO vs HJ flop and 495 of BB vs CO's, the entries solved since the
  builds write them); the single-raised spots' come with their re-solves, and `scripts/instant_coverage.py` counts 28.3 more instant decisions per 100 hands with them on
  the 13 spots the library holds (680.1 -> 708.4 of 813.5, 83.6% -> 87.1%; 46% of the turn's decisions).
* **Study's turn after a library flop** (`Advisor.instant` with `"study": true`, `Advisor._study_store`; study.js's
  answer at once, the page's `studyAsk` and `studyPrefetch`). Study solved a library flop's turn live with only the
  progress bar to look at. Where the entry's turn file holds the flop line it now shows the stored turn for the whole
  range at once (Study, above), and the turn's own solve replaces it. `scripts/bench_browser.mjs --session --early
  --prefetch 0.5` (Pyodide, one thread, a fresh worker per pair: Study's flop answer, then the turn's; 300 ms a
  library file; seconds to the first answer, today -> with the stored turn / to the turn's own solve):

  | Study's turn (one run each) | Fast | Accurate |
  |---|---|---|
  | CO's 3bet vs HJ, 8c5d3h check, bet 33%, call, As: HJ first | 0.84 -> 0.013 / 0.55 | 3.54 -> 0.014 / 3.40 |
  | ... CO after HJ's check | 0.85 -> 0.019 / 0.57 | 3.71 -> 0.019 / 3.38 |
  | BB's 3bet vs SB, Kc3d2h check, bet, call, Qs: SB first | 1.14 -> 0.017 / 0.69 | 4.96 -> 0.018 / 4.74 |
  | ... BB after SB's check | 0.94 -> 0.026 / 0.72 | 4.65 -> 0.026 / 4.72 |

  Today the first answer is the turn's solve, which also fetched the turn file (300 ms here: the solve seeds its
  regrets from it); fetched ahead from the flop's answer, the solve is that much quicker too. Without the fetch
  ahead the answer at once waits for it (0.32 s, Fast, the first row). In a desktop browser (the site built locally,
  the worker's own threads) the first row's stored turn showed 0.010 s after the deal and its solve's answer at 0.31 s
  (Fast) or 0.78 s (Accurate); a check played on the stored answer 0.06 s in had CO's stored answer up 0.02 s later
  and its solve's at 0.77 s, and stepping back to HJ's decision answered at once from that solve.

  Should the stored answer stand on its own (skip the turn's solve, or make it opt-in)? EV given up, each combo's mix
  priced by the turn's live Accurate solve's EVs per action (natively, one thread, 0.01% of the pot from
  equilibrium; the opponent not adapting), % of the pot: 108 nodes on CO vs HJ's and BB vs SB's 3bet pots (six flops
  each, the flop lines checked through and check, bet, call at each size the flop solve takes 5% or more, two turn
  cards each; the turn's first decision and the in-position player's after a check), against Study's live Fast solve:

  | flop line, node | the stored turn: mean (worst) | the live Fast solve |
  |---|---|---|
  | checked through, first | 0.05 (0.14) | 0.001 (0.01) |
  | ... after a check | 0.09 (0.49) | 0.003 (0.02) |
  | check, bet 33%, call, first | 0.03 (0.35) | 0.011 (0.25) |
  | ... after a check | 0.06 (0.50) | 0.014 (0.10) |
  | check, bet 75%, call, first | 0.22 (1.19) | 0.000 (0.00) |
  | ... after a check | 0.31 (0.86) | 0.001 (0.01) |

  Mean 0.10% of the pot (median 0.03), but 17 of the 108 above the library's 0.15% and the worst 1.19%: the stored
  tree has one turn size, and where the turn's solve bets a third of the pot 30% of the time or more (21 nodes) the
  stored mix gives up 0.31% on average (the other 87: 0.05%); after a check its range is the stored first decision's,
  so it holds 63-100% of the live range's weight there (median 97%; first to act 87-100%, median 95%). It has no
  hand's own EVs or equities either, and with the opponent free to adapt it gives up 0.5-0.8% (above). Study's live
  Fast solve gives up 0.006% (natively; 0.55-0.72 s in the browser above, Accurate 3.4-4.7 s). So the solve stays
  automatic and the stored answer is what Study shows while it runs; an action played on it stops the solve, and the
  next node's request resumes it where it stopped (after a check the in-position player's stored answer first).
* **Study's flop where the library doesn't store it** (`Advisor._study_near`, `_near_label`; flop_approx's kept
  prediction, `PRED_KEEP`; study.js asks for it at Accurate). Study's flop on one of the 1203 flops a library spot
  doesn't store, a heads-up pot read off the nearest spot or another stack depth showed the progress bar until the
  flop's live solve answered. At Accurate the stored flops most like it now answer at once for the whole range, as the
  advice's approximate flop does for one hand (`instant_from_near`, below; the same mix for each combo), marked
  approximate with the flops or the spot and depth borrowed, and the live solve replaces it.
  `scripts/bench_browser.mjs --early --prefetch 0` (Pyodide, one thread, a fresh worker a row, 300 ms a library file;
  seconds to the first answer, before -> with the answer at once / to the live solve):

  | Study's flop (one run each) | Accurate | Fast (unchanged) | Fast had it asked |
  |---|---|---|---|
  | HJ vs CO, KhJd6c (KhJd5c, KhJd7c, KhQd6c): HJ first | 2.81 -> 0.45 / 3.00 | 0.93 | 0.44 / 1.31 |
  | ... 8c7d5h (8c7d4h, 8c7d6h, 8c6d5h): CO after HJ's check | 2.67 -> 0.45 / 3.08 | 0.93 | 0.45 / 1.32 |
  | BB 3bets HJ (BB 3bet vs CO's flops), KhJd6c: BB first | 2.20 -> 0.45 / 2.60 | 0.50 | 0.45 / 0.91 |
  | 80bb BTN vs BB (its 100bb entry of the flop), Ks7d2c: BB first | 3.55 -> 0.47 / 4.01 | 1.24 | 0.48 / 1.69 |

  The answer at once waits for the stored flops' entries (fetched together as the request is sent: 300 ms here,
  130-670 ms an entry from the CDN) and reads them in ~0.15 s, and the live solve starts that much later (+0.2-0.5 s).
  At Fast that is about what the live solve takes (0.5-1.2 s on one thread, less on the worker's threads), so Fast
  asks for none and fetches nothing ahead (a library flop's turn files once its answer is in, as before). Its
  quality is the advice's approximate flop's (below: 0.11-0.18% of the pot given up per flop on average, median
  0.05-0.07, facing a bet ~0.4), so the live solve stays automatic. The whole range made the hand class test (a
  combo's images that hold its hand class) categorize every combo on four flops, 0.17 s natively; made for the combos
  either range holds (271 and 295 of the 1326 on HJ vs CO's two flops above; the same mixes), 0.06 s, and the flop's
  next node (played on the answer at once, its solve stopped) reads that prediction again, 0.01-0.02 s. A library
  flop answers as quickly with its entry and turn files fetched together ahead (CO 3bet vs HJ 8c5d3h, BB 3bet vs SB
  Kc3d2h: 0.49-0.52 s either way). Advise, instant and Study JSON identical to main (16 cases: the stored turn's eight, the
  advice's approximate flops first to act and after a check, at a pot read off another spot and another depth, Study's
  instant where it answers nothing).
* **Study's turn and river after a live solve** (`Advisor._study_prev`; study.js asks for an answer at once on a
  river too). A turn after a flop Study solved live (a limped pot, a flop the library doesn't store, a pot it holds no
  spot for) showed the progress bar until the turn's own solve answered: 1.6-4.3 s at Fast, 9-27 s at Accurate on one
  thread in the browser. Now the flop's solve, which plays the turn under it, answers at once for the whole range:
  the advice's early panel's read (`instant_from_prev`: the card dealt as its turn class's played card, each combo
  through the swap of the two, its hand bucket's mix at the one bet size) for every combo, each player's range
  narrowed by that solve's mix along the turn's line, marked approximate with the solve's iterations and distance from
  equilibrium, and the turn's own solve replaces it. A river after a turn solved live reads that solve's river (by
  card class, at its river sizes) the same way. Each combo's mix is the one the early panel gives that hand off the
  same solve (every combo of three turn nodes and two river nodes, exactly). Nothing where the hand's line is one the
  solve takes under 10% of the time, on its own street or this one (the walk gives that range to the betting rules),
  or the street before isn't solved yet. `scripts/bench_browser.mjs --session --early` (Pyodide, one thread, a fresh
  worker a row: Study's flop answer, then the turn's, then the river's; seconds to the first answer, before -> at
  once / the live solve):

  | Study's turn or river (one run each) | Fast | Accurate |
  |---|---|---|
  | Limped pot, Ks7d2c check, bet 75%, call, Th: SB first | 4.28 -> 0.036 / 4.33 | 26.81 -> 0.038 / 25.41 |
  | ... BB after SB's check (a check played on the answer at once: the turn's solve from the start) | 0.068 / 4.15 | 0.067 / 23.94 |
  | ... the turn check, bet 75%, call, 5s: SB first | 0.10 -> 0.020 / 0.12 | 0.22 -> 0.020 / 0.22 |
  | ... BB after SB's check | 0.031 / 0.14 | 0.033 / 0.25 |
  | HJ vs CO, 8c7d5h (not stored) check, bet 33%, call, 2s: HJ first | 1.57 -> 0.026 / 1.75 | 8.82 -> 0.027 / 9.27 |
  | ... CO after HJ's check | 0.012 / 1.63 | 0.012 / 9.05 |

  The live answer comes no later beyond run-to-run noise (-1.4 to +0.45 s); a river's own solve is quick already
  (0.1-0.25 s here), and asking costs ~0.02 s. In a desktop browser (the site built locally, the worker's threads,
  Accurate) the limped pot's Ks7d2c-9h turn showed SB's range 0.036 s after the deal; a check played on it 0.05 s in
  stopped the turn's solve, BB's answer at once came 0.17 s later and the live one at 6.05 s; the river after the
  turn checked through showed at 0.021 s (its own solve's answer at 0.24 s). Quality, natively: the read off a Fast
  flop solve priced by the live Accurate turn's per-combo action EVs (the opponent not adapting), % of the pot, on 32
  nodes of six flops (limped pots and HJ vs CO's flops the library doesn't store; the flop checked through, or check,
  bet, call at each size taken 5% or more; two turn cards; the first decision and in position after a check; 24 more
  answered nothing, their lines under 10%): mean 0.21 (median 0.13, worst 0.97, 14 above the library's 0.15), first
  to act 0.16 (median 0.03), after a check 0.26 (0.18); Study's live Fast answer 0.06 (0.02). It covers 94-100% of
  the live range's weight (median 99.8%). So the turn's solve stays automatic and the read is what Study shows while
  it runs. Advise, instant and Study JSON identical to main (22 cases: the 16 above, Study's stored turns, and none
  facing a library turn's bet or on a river before its turn is solved).
* **A library spot's flop the library doesn't store** (`instant_from_near`, `NEAR_UNTIL`; `gto/flop_approx.py`;
  `Advisor.prefetch`). The flatted opens and the 3bet pots store 552 of the 1755 flops; on the other 1203 (10.2 points
  of the heads-up flop decisions: Precomputed flop library, Coverage) the page had nothing to show until the flop's
  live solve answered. Now the flop's decisions are read at once off the spot's stored flops most like it, marked
  approximate: the three nearest of the same pairing and suit pattern (`neighbours`; the ranks' summed gaps at most 6,
  `MAX_DIST`), each carried over by card substitution (`card_map`: the flop's cards onto the stored flop's, the other
  ranks in order, so KhJd6c read off KhJd5c plays a 6 as the 5 and 54 as 64), their mixes averaged per combo with
  weights 1 / (1 + distance) (`predict`) over the images that hold the combo's hand class there (the made hand and the
  draw, `breakdown.categorize`: 8h9h on KhJd6c, no pair and a backdoor straight draw, reads its images on KhJd5c and
  KhQd6c, not the gutshot it is on KhJd7c; all of them where none does, `CLASS_EPS`; the advisor tests only hero's
  combo, four lookups), facing a bet then corrected by a small learned model (below). It answers wherever an entry
  stores the decision (the first decisions, facing a bet, facing a raise of it) on the line the hand took (sizes at
  their nearest), until the flop's own solve in hand has run 50 iterations (`NEAR_UNTIL`, from when
  `instant_from_own` reads it); the page names the flops it read. Nothing is solved for it (6-16 ms natively), and the
  worker fetches the three entries with the flop's once it's known (the prefetch's dry run names them).
  `scripts/instant_coverage.py` counts these apart: heads-up flop decisions answered at once 77.2% -> 87.5% with them
  (9.4 decisions per 100 hands; postflop 35.2% -> 39.9%, all decisions 83.9% -> 85.1%). Every other answer is
  unchanged (advise JSON identical on stored and unstored flops and a limped pot; `instant` differs only where it now
  answers).


  EV given up, priced by the flop's own library entry (`scripts/bench_flop_approx.py`; that flatters it: against a
  reference the answer doesn't read, the answer as it reads now gives up 3-4x what its library entry says, below: a
  stored flop approximated from the spot's other stored flops, each combo's mix priced by the entry's own EVs per
  action, weighted by the acting player's reach, % of the pot; per flop over its stored decisions weighted by how
  often each comes; 18 of the 2310 flops have no stored flop within 6, mostly trips, and get no answer; `knn3c`, the
  plain average `knn3` in brackets, on the library as it is now, before the learned correction, below: a third less):

  | flops | per flop: mean / median / 90th pct / worst | first decisions: mean (90th) | facing a bet | facing a raise |
  |---|---|---|---|---|
  | the five flatted opens (600, each from the spot's other 551) | 0.18 / 0.06 / 0.32 / 5.3 (0.24 / 0.13 / 0.42 / 5.2) | 0.037 (0.088) | 0.41 (0.86); 0.60 | |
  | the eight 3bet pots (960) | 0.16 / 0.07 / 0.38 / 4.3 (0.24 / 0.16 / 0.50 / 4.3) | 0.029 (0.065) | 0.40 (0.91); 0.56 | 1.21 (2.78); 1.66 |
  | five vs-BB spots' flops outside the 552, from the 552 only (750: the case met) | 0.11 / 0.05 / 0.25 / 2.8 (0.18 / 0.12 / 0.36 / 2.8) | 0.015 (0.040) | 0.43 (1.17); 0.67 | |

  These are priced by the library's own entry of the flop, a solve in the same tree as the stored flops the answer
  reads, and that flatters them: priced by the flop solved live in the Accurate tree instead (600 iterations, a
  reference neither the answer nor the library reads; `scripts/bench_flop_seed.py --truth accurate:600 --seed 11
  --skip 80`: 80 stored flops of the flatted opens and the 3bet pots, over the combos the answer gives), the answer as
  it reads now (hand classes, facing a bet corrected) gives up 0.24% of the pot per flop on the flatted opens and 0.20
  on the 3bet pots (median 0.19 and 0.17) where their library entries price it at 0.065 and 0.059; uncorrected 0.27
  and 0.24. The flop's live Fast solve gives up as much by its 50th iteration (below). That reference leans the other
  way, toward the live solve: it is the browser's own model (hands bucketed on the turn and river), and the library's
  entries another (the river dealt by 64 strength buckets a card, not the browser's hand buckets), so even the flop's
  own library entry gives up 0.12-0.21 against it (`bench_flop_depth.py`'s rows at 100bb, below); the true error lies
  between the two measures.

  The hand class test is better on 78% of the held-out flops by more than 0.01% of the pot, worse on 1.5% (the vs-BB
  spots' 85% and 0.1%): an image that plays another hand there (a gutshot for a backdoor draw, a pair for no pair)
  carried that hand's mix over. The bar it clears: per flop a median under 0.2% of the pot and a 90th percentile under
  0.5%, first decisions under 0.05. Read off the one nearest flop (the plain average) it gives up a third to two
  thirds more (per flop 0.20-0.45 against 0.14-0.29 on the vs-BB spots, 0.29-0.41 against 0.19-0.29 on the flatted
  opens); off four or six a little more than off three. With the hand class test four or six give up a little less
  than three (0.167 and 0.165 against 0.177 on six spots' 720 flops; median 0.073, 0.080 and 0.073), not worth the
  entries more to fetch. The worst are connected broadway boards whose straights don't carry over (KcQcJd read off
  KcJcTd, KcTc9d and KcTc8d: their third-pot c-bet with hands that have no straight here, 5.3), trips aces read off
  trips kings (3.7-4.3) and ace-high read off king-high (AcKd2d, 2.8): the hand class test leaves them, no image
  holding the hand's class. The approximate turn answer from the flop's solve gives up 1.4-1.6% facing a bet (above).
  Against the flop's own live solve (measured before the hand class test, as is the lock below;
  `scripts/bench_flop_seed.py`: 104 of these stored flops solved in the browser's Fast tree natively, one thread, the
  same measure at the stored decisions): the approximation 0.41-0.42 (mean), the solve 0.63-0.70 at 20 iterations,
  0.32-0.35 at 30, 0.21 at 40, 0.12-0.13 at 60, 0.07-0.09 from 100 on (its tree's own distance from the library's);
  better than the approximation on 40-44% of the flops at 30 iterations, 50-59% at 40, 68-81% at 60: hence
  `NEAR_UNTIL` at 50. Measured again as it answers now (`--seed 11 --skip 80`: 80 of these stored flops, none the
  learned correction was fit on; each combo off its own hand class's images, facing a bet corrected; over the combos
  it answers, the solve's too): priced by the library's entries the approximation (0.059-0.065) beats the solve even
  converged (0.18-0.20 at 50 iterations, 0.09-0.11 at 100, 0.08-0.10 from 150 on), but it is read off that same
  library. Priced instead by the flop solved live in the Accurate tree (`--truth accurate:600`, a reference neither
  reads), the approximation gives up 0.20-0.24, the solve 0.51-0.52 at 30 iterations, 0.31-0.33 at 40, 0.21-0.24 at
  50, 0.15-0.19 at 60, 0.07-0.10 at 100, 0.03-0.05 at 300; the approximation better on 31-32 of 40 flops at 40
  iterations, 18-22 at 50, 13-14 at 60, 0-5 at 100: the crossover is the 50th iteration, and `NEAR_UNTIL` stays. With
  the opponent free to adapt (`scripts/bench_flop_approx_lock.py`: 24 stored flops of six spots solved in the
  browser's Accurate tree, numpy, 300 iterations, then again with one player's stored flop decisions locked and
  everything else free: that player's EV lost, % of the pot) the approximation loses 0.27 a player more than the
  flop's own entry locked the same way (median 0.20, 90th percentile 0.60; in Fast's tree 0.28, 0.19, 0.62), in line
  with the measure above; the entry itself loses 0.94 there (1.08 in Fast's), what a strategy from the library's tree
  costs played in another: the measure's floor. In the browser (`scripts/bench_browser.mjs --early`, one thread, the
  compiled core) the answer comes at 0.02 s and the flop's Fast solve at 0.4-1.1 s (flop-near, flop-near-ip,
  flop-near-3bp).

  Facing a bet or a raise a learned correction then adjusts the averaged mixes on the spot's own flops (`correct_mix`;
  `gto/flop_approx_model.py`, written by `scripts/train_flop_correction.py`; numpy). An average blurs: a combo whose
  images split between a call and a raise gets a mix none of them plays, and a hand read off another rank keeps that
  rank's play. To each combo's log(mix + 0.005) it adds, per node kind (the last action's kind and who acts) and
  action kind (check, a bet to half pot, a bigger bet, fold, call, raise, all-in): a bias, a term for the combo's made
  hand and one for its draw, the share of its images that hold its exact class (and the made hand and draw again where
  they don't), the flop's ranks against its sources' (also per made hand) and the log mix itself (how far to sharpen
  it), and renormalizes. It is fit on 80 stored flops of each of the 18 spots, each read off the spot's other stored
  flops as above (the vs-BB spots' off the 552), to the EV given up plus 3 x the squared distance of the mixes from
  the entry's: 300 iterations of Adam, about ten minutes on one core. On each spot's other flops
  (`bench_flop_approx.py --seen --seed 11 --skip 80`: 60 a spot, 1062 in all, the combos the advisor answers; priced
  by their library entries, as above) per flop 0.090 -> 0.061% of the pot (median 0.049 -> 0.032, 90th percentile 0.20
  -> 0.15), lower by over 0.01 on 61% of the flops and higher on 0.8%: facing a bet 0.24 -> 0.15 (the flatted opens
  0.21 -> 0.11, the 3bet pots 0.23 -> 0.15, the vs-BB flops outside the 552 0.28 -> 0.19), facing a raise 0.92 ->
  0.74. Held out by spot (`--folds 2`: each half's nine spots corrected by the model fit on the other nine; 1435
  flops) 0.100 -> 0.066 (median 0.059 -> 0.037, 90th percentile 0.23 -> 0.15; lower on 64% of the flops, higher on
  1%), the mixes a little closer to the entry's (total variation, reach-weighted, 0.1225 -> 0.1215; closer by over
  0.01 on 5% of the flops, further on 1%) and a little purer: where the entry mixes (no action over 90%) 3-4% further
  from it, where it plays one action 7-8% closer. The first decisions (the root, after a check) stay as averaged:
  corrected they gave up 8% more and mixed a little further (0.119 -> 0.122). What didn't work: fit to the EV given up
  alone it gave up less still, but by tilting each combo toward the action the entry's EVs price best, the mixes a
  quarter further from the entry's (a best response to the solve's indifferences, not the solve); fit to the mixes
  alone (cross entropy, squared distance) it brought them no closer and gave up more; finer class fallbacks (the made
  hand alone, then the draw) changed nothing. Priced by a reference it wasn't fit to, the flop solved live in the
  Accurate tree (`scripts/bench_flop_seed.py --truth accurate:600 --seed 11 --skip 80`: 80 stored flops of the flatted
  opens and the 3bet pots, none it was fit on, over the combos it answers), it still gives up less: 0.273 -> 0.243
  (flatted opens), 0.239 -> 0.201 (3bet pots), lower by over 0.01 on 65 of the 80 flops and higher on 1. The reads off
  another spot (below) take it too, none of them a read it was fit on: priced by the held-out spot's library entries,
  off that spot's stored flops most like the flop (ten spots, each read off the one `near_spot` picks for it with it
  left out; 40 flops each, on the spots of 552 none the model was fit on: `--protocol spot --seen --seed 11 --skip
  80`, `nearxc` against `nearxcl`) 0.103 -> 0.075 (median 0.068 -> 0.047, 90th percentile 0.22 -> 0.18), lower on 63%
  of the flops and higher on 1%; off its entry of the flop itself (`nearc` against `nearcl`) 0.035 -> 0.029 (median
  0.020 -> 0.016, 90th percentile 0.085 -> 0.074), lower on 18% and higher on 0.5%; every spot lower on average both
  ways. So do the reads at another stack depth off the spot's stored flops most like the flop, priced by the flop
  solved at the hand's depth in the Accurate tree, a reference the answer doesn't read (`bench_flop_depth.py --knn`,
  `knn100s` against `knn100cl`: eight spots, six flops each, solved at 40-150bb, 132 flops and depths inside the
  gates) 0.299 -> 0.256 (median 0.248 -> 0.221, 90th percentile 0.63 -> 0.48), lower on 77% and higher on none
  (single-raised pots 0.344 -> 0.281, 3bet pots 0.221 -> 0.212). The 100bb entry of the flop itself, read at another
  depth, stays as stored: corrected it moved little (0.367 -> 0.364, lower on 20%, higher on 5%). It costs under a
  millisecond natively (hero's combo 0.9 ms; the whole range's read 147 -> 148 ms).
* **A pot the library holds no spot for** (`flop_approx.near_spot`, `NEAR_COVER`, `NEAR_SPOT_DIST`). The HJ's and the
  UTG's opens flatted by the SB or 3bet by either blind, and the BTN's 3bet of the UTG, have no spot of their own
  (4.5% of the heads-up flop decisions): their flops were solved live with nothing shown meanwhile. They're now read
  off the library spot whose ranges are nearest, marked approximate and named: one with the same pot and stack (so the
  same tree) whose ranges each hold at least 90% of this pot's by weight (`NEAR_COVER`), the least summed distance
  between them (each normalized, L1, at most 2: `NEAR_SPOT_DIST`). The blinds' 3bets of the HJ and the UTG are read
  off their 3bets of the CO (`3bp_bb_co`, `3bp_sb_co`: cover 1.00, distance 0.92-1.23), the SB's flat of either off
  its flat of the CO (`srp_co_sb`: 0.92, 1.39-1.63), the BTN's 3bet of the UTG off the CO's of the HJ (`3bp_co_hj`:
  0.98, 0.29). The CO's open flatted by the BTN has none (the nearest, the HJ's flatted by the CO, covers 80%), nor do
  the 4bet pots. It reads that spot's entry of the flop where it stores it, else its three stored flops most like it
  (as above; facing a bet either corrected), until the flop's own solve has run 50 iterations;
  the prefetch names the entries, and the first such pot sets up the candidates' ranges once (~0.15 s natively). In
  the browser (flop-near-other) the answer comes at 0.02 s, the Fast solve at 0.45 s. `scripts/instant_coverage.py`:
  heads-up flop decisions answered at once 87.5% -> 91.9% (4.1 more decisions per 100 hands).


  EV given up, holding out whole spots (for each library spot, the spot `near_spot` picks with it left out of the
  index; 60 of its stored flops, the measure above priced by its own library entries (the library flatters it, as
  above: against the Accurate tree's solve, `scripts/bench_flop_seed.py --truth accurate:600` with ten of these pairs
  and six flops each, the read as answered gives up 0.14% of the pot per flop in single-raised pots and 0.27 in 3bet
  pots, which these entries price at 0.017 and 0.046; the live Fast solve gives up 0.18 and 0.22 at its 50th
  iteration); per flop mean / median / 90th percentile, first decisions in brackets; "met": the two by how often the
  picked spot stores the flop, 38% of the time for those on 552 flops; the three most like it with the hand class
  test, 0.085-0.44 per flop before it, met 0.009-0.40):

  | held out <- read off (cover, distance) | its entry of the flop | its three most like it | met |
  |---|---|---|---|
  | UTG vs BB <- HJ vs BB (1.00, 0.63) | 0.009 / 0.008 / 0.02 (0.003) | 0.034 / 0.025 / 0.08 (0.008) | 0.009 |
  | CO vs BB <- BTN vs BB (0.99, 1.04) | 0.085 / 0.074 / 0.19 (0.010) | 0.092 / 0.079 / 0.17 (0.015) | 0.085 |
  | HJ vs BB <- UTG vs BB (0.92, 0.63) | 0.269 / 0.235 / 0.46 (0.010) | 0.260 / 0.213 / 0.40 (0.014) | 0.269 |
  | UTG vs BTN <- HJ vs CO (1.00, 0.34) | 0.011 / 0.004 / 0.03 (0.008) | 0.207 / 0.068 / 0.35 (0.036) | 0.132 |
  | HJ vs CO <- UTG vs BTN (0.93, 0.34) | 0.117 / 0.118 / 0.24 (0.008) | 0.272 / 0.151 / 0.69 (0.038) | 0.213 |
  | UTG vs HJ <- UTG vs BTN (0.94, 0.49) | 0.326 / 0.158 / 0.66 (0.056) | 0.398 / 0.242 / 0.65 (0.091) | 0.368 |
  | CO vs SB <- BTN vs SB (0.91, 1.06) | 0.383 / 0.254 / 0.80 (0.036) | 0.330 / 0.244 / 0.69 (0.057) | 0.347 |
  | HJ 3bet vs UTG <- CO 3bet vs HJ (0.95, 0.73) | 0.053 / 0.026 / 0.11 (0.016) | 0.121 / 0.078 / 0.36 (0.035) | 0.095 |
  | SB 3bet vs CO <- SB 3bet vs BTN (0.94, 0.92) | 0.325 / 0.277 / 0.65 (0.020) | 0.243 / 0.218 / 0.44 (0.037) | 0.269 |
  | BB 3bet vs CO <- BB 3bet vs BTN (0.95, 1.16) | 0.334 / 0.228 / 0.69 (0.026) | 0.348 / 0.284 / 0.67 (0.050) | 0.342 |

  The bar it clears: per flop a mean under 0.45% of the pot and a 90th percentile under 0.8, first decisions under
  0.1, about the flop's live solve at 30 iterations (0.32-0.35, above); a library spot's own flops not stored are
  0.16-0.18 (above). The other eight spots have no pick (BTN vs BB, SB vs BB, BTN vs SB, the BB's 3bet of the BTN and of
  the SB, the SB's of the BTN, the BTN's of the CO, the CO's of the HJ), and the pairs under 90% cover measured gave
  up more: 1.77 (the SB's 3bet of the BTN read off its 3bet of the CO, cover 0.73), 1.47 (the BB's, 0.75), 1.20 (HJ
  vs CO off UTG vs HJ, 0.78), 1.04 (BTN vs SB off CO vs SB, 0.73), 0.66 (the CO's 3bet of the HJ off the HJ's of the
  UTG, 0.80), 0.36 (the BTN's of the CO off the CO's of the HJ, 0.77): hence `NEAR_COVER`. Ranking the candidates
  by cover before distance would read HJ vs BB off CO vs BB (0.095 against 0.269) and pick the same spot for every
  pot met; distance alone was kept.
* **Another stack depth** (`flop_approx.near_spot`, `NEAR_SPR`, `NEAR_MIN_SPR`). The library's spots are 100bb deep
  (the 40bb and 200bb formats aren't built), so a flop at 40, 80 or 200bb was solved live with nothing shown
  meanwhile. Now the same spot at 100bb answers it at once, marked approximate (or the spot whose ranges are
  nearest, as above): a spot of the same pot covering both ranges by 90% as above, whose stack-to-pot ratio is within
  0.35-2.05x this pot's (`NEAR_SPR`), where this pot's is at least 3 (`NEAR_MIN_SPR`); a spot of the hand's own
  depth, once one is built, comes first. Single-raised pots are read from about 37bb to 202bb deep (BTN vs BB), 3bet
  pots from 57-72bb (an SPR of 3) to about 185bb where the 100bb spot's ranges still cover the deeper ones (the BB's
  3bet of the BTN; the in-position 3bets' ranges move sooner: the BTN's of the CO to 118bb). The tree is the hand's
  own, built at its stack, so a decision the entry stores with other actions (an all-in where the 100bb entry bets)
  isn't read. The page names the depth: "the precomputed solve of this flop in the library's BTN vs BB single-raised
  pot, solved deeper (a stack-to-pot ratio of 17.7, this pot's 14.1)". `scripts/instant_coverage.py --stack`:
  heads-up flop decisions answered at once 0% -> 51.7% at 40bb, 53.5% at 50bb, 75.8% at 60bb, 89.1% at 80bb, 80.6% at
  125bb, 75.9% at 150bb, 68.6% at 175bb, 43.1% at 200bb (all decisions 73.9 -> 79.9%, 74.4 -> 80.5, 74.8 -> 83.4, 75.0
  -> 85.0, 72.8 -> 82.3, 71.7 -> 80.8, 70.8 -> 79.0, 69.8 -> 74.9); 100bb unchanged (91.9%).

  EV given up (`scripts/bench_flop_depth.py`: ten spots, six stored flops each, every flop solved at 60, 80, 125 and
  150bb and at 100bb in the browser's Accurate tree, natively, 300 iterations, each solve made an entry; each combo's
  mix priced by the depth's entry's EVs as above, a reference the answer doesn't read (no library solve at that
  depth); per flop mean / median / 90th percentile; inside the gates):

  | | the 100bb solve of the flop (the depth's own cost) | the library's entry (what's shown) | the spot's three stored flops most like it (a flop it doesn't store) |
  |---|---|---|---|
  | single-raised pots (114 flops; the last column 42 of the three spots on 552) | 0.104 / 0.088 / 0.217 | 0.233 / 0.206 / 0.379 | 0.369 / 0.328 / 0.588 |
  | 3bet pots (48) | 0.097 / 0.088 / 0.172 | 0.222 / 0.202 / 0.424 | 0.290 / 0.284 / 0.504 |
  | the same flops at 100bb, single-raised / 3bet pots (the measure's floor: the library's tree against the browser's) | 0 | 0.17 / 0.14 | 0.31 / 0.25 |

  So the depth costs ~0.05-0.08% of the pot over the 100bb answer at the hand's depth, within the bar the other
  approximations clear (mean under 0.45, 90th percentile under 0.8). Toward the band's ends it costs more, still under
  the bar (`--depths 40,50`: nine single-raised spots, eight flops each; `--depths 175,200`: eight spots, six flops;
  inside the gates): 0.30 / 0.24 / 0.46 at 40bb (a ratio of 0.38), 0.32 / 0.22 / 0.47 at 50bb (0.49), 0.35 / 0.33 /
  0.58 at 175bb (1.77-1.83; single-raised and 3bet pots), 0.25 / 0.25 / 0.36 at 200bb (2.03; BTN vs BB and SB vs BB,
  the only spots there whose ranges the 100bb ones cover). Outside the gates: the pots under 90% cover 0.58-1.16
  (mean) at 40-50bb and 0.43-0.78 at 175-200bb, 3bet pots at 60bb (an SPR of 2.4) 1.3-1.9 with most of the entry's
  decisions not in the hand's tree, and 3bet pots at 125bb and deeper whose ranges the 100bb spot covers by 66-83%
  0.5-1.2. Facing a bet the read off the stored flops most like it (the last column) now takes the learned
  correction (above): a seventh less given up in a second sample of 132 flops and depths.
* **Another table size** (`flop_approx.near_exact`, `NEAR_EXACT_COVER`, `NEAR_EXACT_DIST`). The library's spots
  are 6-max's, so an 8- or 9-max flop was read approximately off the nearest spot (above: no EVs, until its own solve
  had run 50 iterations) and then solved live. Most of those tables' heads-up pots are a 6-max pot under other seat
  names: the same pot and stack, the ranges entering the flop a few combos apart (the BTN's, CO's, HJ's and SB's opens
  the big blind calls, the BTN's and the CO's flatted by the SB, the blinds' 3bets of the BTN and the CO, the BB's of
  the SB). Such a pot is now read as the library spot's own: at once, with the entry's EVs and stored turns, as the
  6-max hand gets it, badged "precomputed · borrowed" (its tip and the note say how far: "approximate: this table's
  ranges differ from that spot's by 0.4% of their weight at most", 8-max BTN vs BB) and named ("6-max BTN vs BB
  single-raised pot"). It's taken where the spot's ranges hold at least 97% of each of this pot's by weight
  (`NEAR_EXACT_COVER`) and the two ranges' distances (normalized L1, as above) sum to at most 0.15
  (`NEAR_EXACT_DIST`). The combos off the spot's ranges are left out of them; your own hand off them is solved live,
  as are ranges you typed in. Study reads the same spot. `scripts/instant_coverage.py --players 8`: heads-up flop
  decisions answered at once by a precomputed solve 0% -> 46.5% (with the approximate reads 84.9% -> 85.2%), all
  decisions 79.8% -> 83.8%; `--players 9` 0% -> 39.4% (82.1% -> 82.3%), 81.3% -> 84.3%; 6-max unchanged.

  EV given up (`scripts/bench_flop_formats.py`: 20 of each spot's stored flops, every flop solved at the 8-max pot's
  own ranges in the Accurate tree, natively, 300 iterations, 0.05-0.12% of the pot from equilibrium: the reference;
  each combo's mix priced by its per-action EVs there, % of the pot per flop, mean / median / worst flop, over the
  combos the spot's ranges hold; the floor: the 6-max pot's entry against its own solve the same way; the last
  column the 6-max pot's reference solve read for the 8-max pot, what the ranges' difference alone costs):

  | 8-max pot <- 6-max spot (cover, distance) | the borrowed entry | the 6-max pot's own entry | the 6-max solve read |
  |---|---|---|---|
  | BTN vs BB (1.00, 0.01; 9-max the same) | 0.120 / 0.127 / 0.16 | 0.119 / 0.119 / 0.17 | 0.002 (worst 0.005) |
  | SB vs BB (1.00, 0.01) | 0.212 / 0.208 / 0.32 | 0.210 / 0.204 / 0.32 | 0.004 (0.013) |
  | CO vs BB (1.00, 0.04) | 0.128 / 0.130 / 0.18 | 0.128 / 0.126 / 0.20 | 0.008 (0.019) |
  | HJ vs BB (0.99, 0.04) | 0.126 / 0.118 / 0.19 | 0.127 / 0.116 / 0.20 | 0.011 (0.023) |
  | BTN vs SB (0.98, 0.10) | 0.164 / 0.132 / 0.38 | 0.165 / 0.138 / 0.40 | 0.017 (0.043) |
  | CO vs SB (0.98, 0.14) | 0.178 / 0.143 / 0.52 | 0.162 / 0.135 / 0.38 | 0.025 (0.058) |
  | BB 3bet vs BTN (0.99, 0.06) | 0.207 / 0.199 / 0.35 | 0.213 / 0.210 / 0.36 | 0.005 (0.023) |
  | SB 3bet vs BTN (0.99, 0.07) | 0.251 / 0.246 / 0.51 | 0.243 / 0.259 / 0.47 | 0.009 (0.027) |
  | BB 3bet vs CO (0.99, 0.06) | 0.208 / 0.226 / 0.41 | 0.212 / 0.221 / 0.46 | 0.007 (0.029) |
  | SB 3bet vs CO (0.97, 0.12) | 0.236 / 0.234 / 0.47 | 0.248 / 0.256 / 0.48 | 0.010 (0.047) |
  | BB 3bet vs SB (1.00, 0.03) | 0.081 / 0.066 / 0.19 | 0.082 / 0.071 / 0.22 | 0.009 (0.025) |

  So a borrowed entry gives up what the library's own does at 6-max: the ranges' difference costs 0.002-0.025% of
  the pot, against the 0.08-0.25 the entries give up at all. Further out it grows slowly: the 6-max solve read for
  the 8-max pots 0.17-0.57 apart (HJ vs CO, LJ vs BTN, LJ vs CO, the HJ's 3bet of the LJ, UTG and UTG1 vs BB; 12
  flops each) gives up 0.009-0.039 (worst flop 0.072); they stay approximate reads.
* **Antes and a straddle** (`flop_approx.near_spot`'s `aggr`, `NEAR_POT_COVER`, `NEAR_POT_DIST`, `class_fill`).
  Antes and a straddle put dead money in, so none of those games' pots is a library pot, and nothing answered their
  flops at once. Now a spot of another pot is read for one where the same seat raised last preflop (out of position,
  or in position, in both: the straddled BTN's open called by UTG reads BTN vs BB), the stack-to-pot ratio within
  `NEAR_SPR` as for another depth, the spot's ranges holding 90% of each of this pot's by weight (`NEAR_POT_COVER`)
  and the distances summing to at most 1.0 (`NEAR_POT_DIST`; the same pot's limit is 2.0, but 6-max's squeezes and
  cold calls of a 3bet, 1.1-1.9 from the 3bet pots of other sizes, measured badly and stay unread). It's read as a
  pot with no spot is (approximate, no EVs, until the flop's own solve has run 50 iterations), with two
  differences: the trees' raises are matched by their size in chips ("raise to 15" in either pot), and the hands the
  spot's ranges don't hold, which the game's wider ranges have, play the average mix there of their hand class (made
  hand and draw, else the made hand alone, weighted by the range: `class_fill`), the panel saying so ("your hand
  isn't in that spot's range: the average mix there of its hand class (made hand and draw)"). The page names the pot:
  "another game's pot (5.5 bb, a stack-to-pot ratio of 17.7; this one 11.5 bb, 8.3)". `scripts/instant_coverage.py
  --ante 0.125`: heads-up flop decisions answered approximately at once 0% -> 24.4% (all decisions 71.0% -> 74.1%);
  `--straddle` 0% -> 38.0% (74.5% -> 78.9%). Heads-up (2 players) stays at 0: the raised pots' ranges are wider than
  any spot's (cover under 90%, 56.8% of its flop decisions), and no spot is a limped pot (38.7%).

  EV given up (as above, 20 flops each, 12 for the last three; the filled hands included; against the live Fast
  solve at its 50th iteration, where the read gives way to it):

  | pot <- spot (cover, distance) | the read | the live Fast solve at 50 iterations |
  |---|---|---|
  | ante CO vs BB <- BTN vs BB (0.91, 0.90) | 0.205 / 0.192 / 0.35 | 0.198 / 0.196 / 0.28 |
  | straddled BTN vs UTG <- BTN vs BB (0.95, 0.95) | 0.190 / 0.177 / 0.31 | 0.205 / 0.201 / 0.31 |
  | straddled CO vs UTG <- HJ vs BB (0.93, 0.33) | 0.172 / 0.168 / 0.31 | 0.194 / 0.186 / 0.28 |
  | ante UTG vs BB <- HJ vs BB (0.90, 0.76) | 0.186 / 0.159 / 0.28 | 0.188 / 0.192 / 0.23 |
  | straddled HJ vs UTG <- UTG vs BB (0.91, 0.37) | 0.177 / 0.180 / 0.29 | 0.194 / 0.187 / 0.25 |
  | straddled BB vs CO <- CO vs SB (0.95, 0.92) | 0.273 / 0.254 / 0.68 | 0.213 / 0.226 / 0.35 |

  Outside the gates they did worse (cover 0.55 and distance 1.4 tried): heads-up's 3bet pot off the BB's 3bet of the
  BTN (cover 0.51, distance 1.70) 0.579 / 0.622 / 0.89 against the live solve's 0.149; the ante's BTN vs SB (0.48)
  0.770; a straddled SB vs UTG (0.84, 1.41, a stack-to-pot ratio under half the spot's) 0.729; 6-max's BB cold-calling
  the BTN's 3bet of the CO off the CO's call (0.89, 0.85) 0.457, worst flop 2.37, and the SB's squeeze off its 3bet
  of the CO (1.00, 1.34) 0.463; the hands filled are the worst part (0.4-0.9 in the 3bet pots, 0.27-0.37 in
  single-raised ones).
* **Your hand out of your range** (`Advisor._as_advised`). A hand the line's range doesn't hold (a preflop call or
  check the charts never make: the big blind flatting aces against the button, which it 3bets, or checking KQs
  behind a limp, which it raises) is solved on each street as a near-zero-weight combo (`_with_hero`). The next
  street's narrowing looked the street's solve up by the ranges without it, didn't find the advice's, and solved the
  street again at the earlier streets' budget. It now reads the advice's own solve, your hand in, and takes the hand
  out again after the walk, so it is still out of its range on the next street, as before. Fast in the browser
  (`scripts/bench_browser.mjs --session --hole`, new: one hand street by street in one worker; Pyodide, one thread;
  seconds to the answer, today -> now, two runs; an in-range hand's in brackets):

  | spot (your hand) | turn | river |
  |---|---|---|
  | limped pot, the big blind checks behind (KhQh) | 10.70-11.09 -> 8.95-9.27 (9.23) | 3.23-3.25 -> 0.13-0.16 (0.15) |
  | BTN vs BB, a library flop, BB flats (AhAd; one run) | 5.57 -> 5.63 (5.87) | 3.13 -> 0.12 (0.12) |
  | 3bet pot, the flop solved live, BTN flats (AhKh) | 1.72-1.82 -> 1.23-1.34 (1.18) | 1.22-1.41 -> 0.05-0.06 (0.05) |
  | 40bb, BB flats (AhAd) | 3.97-4.13 -> 3.25-3.26 (3.30) | 3.08 -> 0.07 (0.05) |

  The approximate answer (B3) now comes for such a hand too, at 0.002-0.03 s on these turns and rivers: the earlier
  street's solve it reads holds the hand (before, nothing was cached for it to read). Nothing else solved a street
  twice: a library flop's narrowing reads the library (by the ranges without your hand), a multiway street's the
  hand's own solve by its line, and a hand that takes a line its strategy doesn't on a later street (a call of a
  c-bet it folds) keeps its smoothed weight, so it stays in its range (Jh9h on Ks7d2c: the river 0.13 s either way).
  Still slower than an in-range hand: a library flop's own decision (the entries don't hold the hand: solved live, 1.2
  s against 0.04 here), and the stored turn (its files hold the range's hands only). The hand's answer on the turn and
  river, natively at the browser's budgets over four flops each (Ks7d2c, 9h8h4c, Ad5d3c, Qc6s2h): priced by today's
  EVs per action (B3's measure), % of the pot, mean (worst):

  | | turn | river |
  |---|---|---|
  | limped | -0.004 (0.00) | 0.011 (0.05) |
  | BTN vs BB | 0.000 (0.00) | 0.008 (0.12) |
  | 3bet pot, 40bb | 0.00 (0.00) | 0.00 (0.00) |

  Against a reference (today's way with budgets of 20 s, settled at 0.1%): on BTN vs BB's rivers today's answer gave
  up 0.84% (3.25 worst: aces on Ks7d2cTh5s betting a third 66% where the reference checks, the button's turn range
  narrowed by a turn solved again in 3 s, 121 iterations) and the new one -0.03 (0.00); limped turns 0.027 (0.11)
  against 0.006 (0.03); the rest equal. The answer is the street's own solve, as for a hand in range, so it isn't
  marked approximate.
* **The showdowns' sums, packed** (`core/gtocore.c` `ranking`, `prefixes`, `compat`; `gto/core.py`
  `_ranking_tables`). A river showdown's value for each combo reads eight running sums (its rank's two ends in the
  strength order, each card's two in the card order) and two card terms. The core read them through eight int32
  index tables, one combo at a time, and built the two orders' running sums one after the other. Now each combo's
  eight indices are 16-bit slots in two 64-bit words, four combos are summed at once (each lane gathers its own
  terms), the two orders' sums advance in one loop, and they are laid out quarter by quarter, so the four quarters'
  sums the core already ran side by side are stored with one vector write. A fold's card-removal sums write four
  combos at a time too. Each combo adds the same terms in the same order, so a solve's regrets and strategy sums
  are the same bit for bit: natively, in `core.wasm` and in `core-mt.wasm` on two threads, on four turns (BTN vs BB
  after the library's flop, limped, a 3bet pot, a two-tone board) in Fast's tree and the threaded browser's, two of
  them raked, with the floor of DCFR+ too, at 30-200 iterations. Exploitability is unchanged.
  Natively, one thread, an iteration took 23.9 ms and now takes 19.5 (BTN vs BB), limped 36.4 -> 29.5, the 3bet
  pot 3.4 -> 2.8, two-tone 38.4 -> 29.5: 1.23-1.30x (1.15-1.27x in the threaded browser's tree). In Pyodide
  (`core.wasm`) it is 1.00-1.18x faster (the 3bet pot's small turn unchanged), as wasm's gathers cost more. In
  the browser (`scripts/bench_browser.mjs`, Pyodide, one thread, the library builds paused, two runs each, before
  -> after), Fast:

  | request | seconds to the answer | iterations, % of the pot |
  |---|---|---|
  | turn after a library flop (BTN vs BB) | 4.89-5.08 -> 4.40-4.41 | 200, 0.23 |
  | limped turn | 9.01-9.18 -> 8.12-8.19 | 200, 0.26 |
  | 3-way turn (the multiway core: unchanged) | 7.51-7.62 -> 7.49-7.52 | 400, 0.04 |

  The library's entries are unchanged, and so is their build time. Their river is dealt in buckets, so it has none
  of these showdowns, only the folds' sums: one process at the build's settings, BTN vs BB Kc7d2h took 72.5 s and
  now takes 70.5 (300 iterations, 0.110%), a 3bet pot's 17.8 and now 18.1 (200, 0.148%). Both entries decode to
  the same strategies, masks and reports.
* **The discount in the passes** (`core/gtocore.c` `core_run`, `decision`, `strategy`). DCFR discounts every
  regret and strategy sum once an iteration (`Solver._factors`). The core did it in a pass of its own after the
  iteration's two, one more trip through every regret and sum (40 MB on BTN vs BB's turn after the library's flop)
  that the passes had just left. Now each is discounted at its last touch in the iteration: player 0's regrets as
  the second pass reads them for player 0's strategy, player 1's regrets and every strategy sum as its owner's pass
  adds to them. They are the same products of the same floats, so a solve's regrets and sums are the same bit for
  bit: natively, in `core.wasm` and in `core-mt.wasm` on two threads, with DCFR's two factors and DCFR+'s floor, on
  turns in Fast's tree and the threaded browser's, a raked turn dealing every river, and a flop by hand buckets.
  Exploitability is unchanged. A program with deferred games (natively, a 3bet pot's library flop: its dealt
  rivers' bucket games) walks their decisions twice in a pass, so it keeps the pass after (`core_fuse(0)` keeps it
  everywhere, for the tests). Natively, one thread, the library builds paused, an iteration took 16.1 ms and now
  takes 15.1 (BTN vs BB), limped 24.8 -> 22.9, the 3bet pot 2.6 -> 2.3, two-tone 24.5 -> 23.2: 1.06-1.11x
  (1.02-1.07x in the threaded browser's tree). In the browser (`scripts/bench_browser.mjs`, Pyodide, one thread, the
  library builds paused, two runs each, before -> after), Fast:

  | request | seconds to the answer | iterations, % of the pot |
  |---|---|---|
  | turn after a library flop (BTN vs BB) | 4.45-4.47 -> 4.26-4.28 | 200, 0.23 |
  | limped turn | 8.35 -> 8.00-8.09 | 200, 0.26 |
  | 3-way turn (the multiway core: not this code) | 7.49-7.52, as before | 400, 0.04 |

  The library's build is unchanged: a single-raised flop's time is in its leaves' products and a 3bet pot's keeps
  the pass after (one process at the build's settings, BTN vs BB Kc7d2h 52.5-52.6 s -> 52.0-52.6, a 3bet pot's
  Kc7d2h 13.4 -> 13.4-13.5). Both entries decode to the same strategies, EVs, masks and reports.
* **Terminals nobody reaches** (`core/gtocore.c` `terminal`, `unreached`). As a solve settles, whole rows of a
  terminal go unreached: the opponent never takes the line on that river card, so its reach there is +0 on every
  combo (a reach is a product of nonnegative factors, never -0). By the 200th iteration that is 13% of the fold and
  showdown rows on BTN vs BB's turn and 20% limped (5-8% over the first 100). Their sums come to zeros: a fold's
  card-removal sums f x (+0), a showdown's running sums +0 times each combo's live flag times its factor (and the
  rake's card-removal term likewise), a leaf's product +0. The core now checks a row's reach first (the bits of 16
  floats at a time, ORed) and on a row that is all +0 writes those zeros without the sums. The same floats bit for
  bit: regrets and strategy sums natively, in `core.wasm` and in `core-mt.wasm` on two threads, on four turns in
  Fast's tree and the threaded browser's, two raked turns, DCFR+'s floor, at 60-200 iterations;
  and the passes after a solve (EVs, best responses, exploitability) the same. `core_skip(0)` sums every row (the
  tests). Natively, one thread, the library builds paused, 200 iterations: an iteration 15.0 -> 14.3 ms (BTN vs BB),
  limped 22.6 -> 21.2, the 3bet pot 2.29 -> 2.22, two-tone 23.0 -> 21.4 (1.03-1.07x). In the browser, Fast (five
  runs each, alternating, median and range, before -> after):

  | request | seconds to the answer | iterations, % of the pot |
  |---|---|---|
  | turn after a library flop (BTN vs BB) | 4.27 (4.22-4.36) -> 4.00 (3.96-4.43) | 200, 0.23 |
  | limped turn | 7.94 (7.88-8.06) -> 7.39 (7.37-8.01) | 200, 0.26 |

  The library's build is unchanged: its flop's lines end in batched leaves and its later streets' terminals are
  bucketed, so only the flop's and turn's folds can skip (one process at the build's settings, BTN vs BB Kc7d2h
  49.5-52.4 s -> 49.8-51.8, a 3bet pot's 13.1-13.4 -> 13.2-13.6; the entries decode to the same strategies, EVs,
  masks and reports).

  The helpers' lanes are sized by a serial pass over a chance node's chunks in the first iteration, and that pass now
  sums every row (`measuring`). It had skipped the rows nobody reached there too: the second player's pass meets the
  first's regret-matched zeros, and on a tree cut to one bet and one raise size a street (TRIMS' third level) the
  deepest showdowns went unreached on every row, so the lanes came 16 bytes short of a later chunk's. That chunk
  trapped in a helper: natively the process ended; in the browser the helper's worker died holding its rows and the
  solve spun on them, out of reach of the page's stop (read between chunks of iterations), so the worker answered
  nothing more until the page was reloaded. It hit Quick's limped turns, and Practice's at Fast, on 8 threads every
  time (their 1.5 s budget trims a limped turn's 10.9M cells), the Advisor's limped turn and river spots at a 1 s
  budget, and any device slow enough to trim them at Fast. Swept natively on the core before (686 trees of four turns,
  two rivers and a flop, four ranges, four trees and TRIMS' levels, 12 iterations each): 28 trapped on 8 threads, 28
  on 4 and 20 on 2, all turns at that level (28 of its 112); full trees didn't. Now none of the 686 on 2, 4 or 8
  threads, and none of 65 requests (13 of the bench's spots at budgets of 1, 3 and 8 s on 8 threads, and at Quick's on
  4 and 8). The lanes are the same with the skip on as off (`core_lanes`), and so are the floats.

  A helper that fails no longer hangs the page. In the browser a trap in a helper (a bug like that one) left the chunk
  of rows it held unfinished, and the solve waited on them inside the core for good, out of reach of the page's stop.
  Now worker.js's helper catches it and marks itself in the core's memory (`core_dead`), the wait on the job's rows
  traps on that, and the solve's `core_run` throws: `CoreError`, and numpy runs the solve and the session's later
  ones, as after any failure of the core (the passes after a solve too, `core.evaluate`, which let it through; now the
  core starts again first: The core starts again after a trap, below). In the browser bench (`BENCH_THREADS`, new: the
  worker's helpers as Node worker threads running worker.js's own helper script; `BENCH_FAIL`, new: `core_fail`, a
  test's, fails a helper holding rows), helper 0 failing on its first chunk on 2 threads and on 8: BTN vs BB's Fast
  turn answered in 12.2 / 11.9 s by numpy (36-39 iterations, 4.9-6.9% of the pot from equilibrium; 5.1 s and 300
  iterations, 0.15%, in the core), and the limped turn and a river after it answered too. And a stopped request that
  can't read the stop no longer holds the requests after it: the page pings the worker after a stop and restarts one
  that hasn't answered in 15 s (`STUCK_S`; a stop takes ~0.1 s), posting the requests still waiting to the new one.
* **The turn solve starts from the stored turn** (`Advisor._store_seed`, `STORE_SEED`, `solve(seed=...)`,
  `flop_library.turn_strategies`). After a library flop whose entry has a turn file, the turn's own solve (the
  request's, the presolve's and the river's narrowing) no longer starts from zero regrets: at the turn's first decision
  and in position after a check, each combo the file stores gets regrets of the pot x the opponent's reach compatible
  with it x its stored mix (each stored action at the live tree's nearest; in position the opponent's reach is the
  first player's times its stored check), so the first iteration plays the stored mix. Nothing else is seeded: the
  strategy sums start at zero and every other node and combo (your hand out of your range among them) starts as
  before. On a checked-through flop the threaded browser's settled stop (0.2% of the pot from the 300th) now fires at
  the 300th where it took the 400th. In the browser, Fast, 4 threads (`BENCH_THREADS`), BTN vs BB's turn files
  (`BENCH_LIB`), a fresh worker per spot, two runs each, alternating:

  | request | seconds to the answer | iterations, % of the pot from equilibrium |
  |---|---|---|
  | BB first on Ks7d2cTh, the flop checked through (`turn-x`, new) | 4.28 / 4.27 -> 3.25 / 3.29 | 400 -> 300, 0.14 -> 0.18 |
  | BTN after BB's check there (`turn-x-ip`, new) | 4.20 / 4.21 -> 3.21 / 3.21 | 400 -> 300, 0.13 -> 0.17 |
  | BB first after a third-pot c-bet called (`turn`) | 2.89 / 2.91 -> 2.86 / 2.85 | 300 either way, 0.14 -> 0.12 |

  Measured natively over 30 turns of four BTN vs BB flops (checked through and a third-pot c-bet called; the threaded
  tree, against each turn solved to 1000 iterations): the stop fires at the 300th on 28 of 30 turns where it did on 13
  (mean 357 -> 307 iterations), there 0.142 -> 0.160% of the pot from equilibrium on average (worst 0.190 -> 0.199),
  each combo's EVs over the actions it plays 0.269 -> 0.175% of the pot off the reference's (worst 0.628 -> 0.370),
  its EV at its own mix 0.218 -> 0.131, EV given up by the mixes (B3's measure) 0.0017 / 0.0015 -> 0.0012 / 0.0009,
  hands out of the ranges 0.293 -> 0.257. At the same 300 iterations: 0.206 -> 0.161% from equilibrium. On one thread
  (Fast's own tree, 0.3% from the 200th; 24 turns) the stop comes at 212 iterations on average where it came at 250,
  0.197 -> 0.228% from equilibrium, EVs over the played actions 0.420 -> 0.286 off. A stored turn is a 5% step mix
  of a 75% bet only, played on a class's card: a start, not an answer, so the stopping rules are unchanged. The
  turn file is read before the solve (the approximate answer reads it first, and the worker fetches it with the
  flop's entry).
* **The first chance nodes in one job** (`core/gtocore.c` `subs_job`, `subs_run`, `plan`, `core_subs`). On threads a
  training pass ran a job per chance node: on a turn, each line's river split into its classes' rows (13 on a rainbow
  turn), the helpers taking rows while the thread that set the job up spread the reaches, waited for the last rows and
  summed them, then the next line's. Profiled natively (the threaded browser's tree after a library flop), 19-25% of a
  turn iteration on 4 and 8 threads ran on that one thread (2.3-3.1 ms of 10-15): setting up 13 jobs a pass, each
  class's spread and sums (1.5-2.1 ms), and at 8 threads 2.5-2.7 ms waiting for each job's last rows. Now a training
  pass walks down to its first chance nodes (those with none above them) and keeps each one's reaches, runs every row of
  all of them as one job (169 rows on these turns: 13 lines, 13 classes each; the thread that finishes a node's last row
  sums it), and walks down again to take the sums where the nodes are. The serial share comes to 10-15% (0.85-1.3 ms:
  mostly the second walk), the wait to 0.1-0.5 ms. The same floats bit for bit: a row's arithmetic and a node's sums are
  the same functions' as a job a node, each row in a helper's lane of its own (sized by a serial pass over the rows in
  the second iteration, `core_lanes(2)`). The first iteration still runs a job a node: it measures a whole pass's lanes
  (`core_lanes(0)`), which the evaluations after a solve use (measured on the first evaluation alone, a later one's came
  16 bytes short and trapped on a limped turn at a 1 s budget). Checked against the core before on the lanes fix's 686
  trees (four turns, two rivers and a flop, four ranges, full trees and TRIMS' levels) on 2, 4 and 8 threads: regrets
  and strategy sums equal, and the passes after (EVs, best responses, exploitability); and 65 requests at budgets of 1,
  3 and 8 s and Quick's, none failing. One thread is unchanged (no helpers, no subs); `core_subs(0)` runs a job a node
  (the tests). `core-mt.wasm` 63 -> 76 KB (22 -> 27 KB gzipped); the heap unchanged.
  Natively, ms an iteration from the 100th (medians of 3, alternating, beside the library's 3bet pot build):

  | turn | 2 threads | 4 threads | 8 threads |
  |---|---|---|---|
  | after a third-pot c-bet called (`turn`) | 15.2 -> 11.8 | 9.4 -> 7.4 | 8.2 -> 6.9 |
  | the flop checked through (`turn-x`) | 17.4 -> 13.8 | 12.7 -> 8.1 | 11.1 -> 7.0 |
  | limped (SB vs BB, `turn-limp`) | 19.5 -> 16.0 | 14.6 -> 11.6 | 12.2 -> 9.2 |

  In the browser bench, Fast, BTN vs BB's turn files (`BENCH_LIB`), a fresh worker per spot, two runs each,
  alternating (seconds to the answer, before -> after; the same iterations and exploitability):

  | request | 8 threads (the live page's) | 4 threads | iterations, % of the pot |
  |---|---|---|---|
  | `turn` | 3.28 / 3.22 -> 2.27 / 2.34 | 3.67 / 3.66 -> 3.12 / 2.99 | 300, 0.12 |
  | `turn-x` | 3.72 / 3.91 -> 2.79 / 2.81 | 4.22 / 4.41 -> 3.60 / 3.79 | 300, 0.18 |
  | `turn-limp` | 5.81 / 6.06 -> 4.59 / 4.62 | 6.53 / 6.80 -> 5.76 / 5.88 | 300, 0.16 |

  The Accurate turn at a 15 s budget on 8 threads: 10.35 / 10.50 -> 7.74 / 7.67 s, 1000 iterations, 0.02%.
* **The turn's equity leaves in the job** (`core/gtocore.c` `keep`, `leaf_tasks`, `leaf_task`, `dot`, `core_leaves`).
  With the subs in one job, what was left of a threaded turn iteration's serial part (0.75-1.1 ms of 7.3-8.8 on 8
  threads, profiled) was nearly all the pass proper's equity leaves: an all-in called on the turn, its utility a
  product with an (n, n) matrix of equities over the rivers, 6 rows a pass on these turns, each a matvec on the one
  thread that walks the turn (0.6-0.9 ms an iteration; its fold and chance rows 0.05). Now the walk down keeps each
  leaf row's opponent reach beside the subs', the job's last rows multiply them (64 of a matrix's rows each, every
  reach on that matrix at once: a matrix row read once for all of them, from cache), and the pass proper takes the
  product of a row whose reach is the one kept bit for bit (else it multiplies, as before). Each entry is `dot()`'s sum,
  `matvec`'s, so the floats are the same: regrets and strategy sums equal with the leaves in the job or not, and as one
  thread, on 2, 4 and 8 threads over the 686 sweep trees (four turns, two rivers and a flop, four ranges, full and trimmed: 574 with leaves taken), and in the tests (four turns: full, trimmed, a
  25bb stack, the threaded page's tree with BTN vs BB's ranges; `core_leaves_taken` counts the products taken, so a
  test sees the path run). One thread is unchanged; `core_leaves(0)` multiplies them in the pass (the tests).
  `core-mt.wasm` 76 -> 83 KB. In the browser bench on 8 threads (Fast, BTN vs BB's turn files, a fresh worker per spot,
  two runs each, alternating, in a quiet hold; seconds to the answer, the same iterations and exploitability):

  | request | before -> after | the solve | iterations, % of the pot |
  |---|---|---|---|
  | `turn` | 2.22 / 2.28 -> 2.05 / 2.12 | 2.06-2.11 -> 1.90-1.96 | 300, 0.12 |
  | `turn-x` | 2.66 / 2.51 -> 2.45 / 2.49 | 2.33-2.49 -> 2.28-2.31 | 300, 0.18 |
  | `turn-limp` | 4.36 / 4.28 -> 3.91 / 3.93 | 2.86-2.96 -> 2.71-2.76 | 300, 0.16 |

  (`turn-limp`'s flop narrowing, a threaded flop solve with all-in leaves, 1.40 -> 1.18 s of it.) Natively, ms an
  iteration from the 100th on 8 threads (medians of 2, alternating, the same hold): `turn` 5.79 -> 5.13, `turn-limp`
  8.18 -> 7.48, `turn-x` 6.42 -> 6.43 (runs of 6.16-6.68 against 5.97-6.89: within the noise; outside the hold, the slot beside the build, 8.48 -> 7.90).
* **An evaluation's lanes, each kind its own** (`core/gtocore.c` `lane_size`, `lane_k`, `core_eval`, `job`, `SPARE`,
  `core_lanes_grown`). An evaluation from the root of a threaded tree (`Solver.exploitability`'s best responses and,
  raked, its averages) runs its chance nodes' rows on the helpers, each in a lane sized by a serial pass. A training
  pass sizes them in a solve's first iteration, and the evaluations after it use those. A program no training pass had
  sized (one built to evaluate a solve other than the arena's, `core.evaluate`; a solve stopped before its first
  iteration) sized them on its first evaluation alone (9d6a7f17), and one player's pass can take more scratch than the
  other's. On Ks7d2cTh (the tests' wide ranges, Fast's tree, 9.1bb pot) the first player's best response took 100,416
  bytes a lane and the second's 97,664, so the second's measure trapped the first's in a helper. On the tree trimmed to
  TRIMS' third level it went the other way: exploitability's own order (the first player, then the second) trapped.
  In the browser a trap restarts the core and runs the evaluation again, the same trap follows, and the core is off for
  the session: numpy then answers every solve, a Fast turn in ~12 s at ~5% of the pot instead of ~2 s at 0.15%.
  No page request was found that reaches it. The Advisor reads a solve's exploitability right after its iterations and
  keeps it per iteration count (`_expl`), so the program it evaluates is the arena's, sized by its own training. A
  probe over 13 test modules on 2 threads (advice, presolves, Study, the breakdown, settled stops, instant answers, the
  cache, page states) found a root evaluation in a program built only to evaluate in three tests alone. Each read an
  earlier solve's exploitability after a later solve, and those run on one thread. A script doing the same, the
  windows' accuracy sweep, found the trap. No flow runs the second player's best response before the first's; the
  trimmed tree trapped in that order anyway.
  Now, until a training pass has sized the whole pass's lane, each kind of evaluation (an average or a best response,
  either player) measures and keeps its own. A training pass's lane also covers every evaluation after it: on that
  turn 188,544 bytes. A lane that still comes short no longer traps. Each helper's lane is its measure plus half again,
  and a job in which a helper held more than the measure grows that lane to what it held for the next job
  (`core_lanes_grown` counts them). Scratch is scratch, so the floats are the same; only a helper past the spare (a bug)
  still traps. The lanes take 1.5x the bytes: on that turn on 8 threads, 1.3 -> 2.0 MB for the whole pass's and
  0.7 -> 1.1 MB for the subs' (one job at a time).
  Natively on 8 threads, ms an iteration from the 100th to the 200th of the Advisor's own Fast turns, old / new / new /
  old in a quiet hold: `turn` 5.99 / 6.23 -> 5.90 / 6.23, `turn-limp` 8.16 / 7.53 -> 8.18 / 7.43. An exploitability
  read took 10.7-13.2 ms before and 10.0-13.3 after, with the same exploitability to 5 decimals (0.23244 and 0.3078%
  of the pot).
  Tests: `test_evaluations_in_any_order` (a child process, as a trap ends it). It covers every kind of program the
  Advisor builds: evaluate-only, a presolve stopped before its first iteration, and a solve, a warm-started solve and a
  seeded solve, each after its next iteration. Each runs all four evaluations in all 24 orders on 2, 4 and 8 threads,
  on that turn full and trimmed. The floats match one thread's, and no lane grew. On the core before, it trapped on
  the first: full tree, 2 threads, evaluate-only. A lane set to 3/4 of its measure, the subs' in a training pass and
  then the whole pass's in an evaluation, grows to what its helpers held, with the same floats.
* **Each player's combos alone** (`Solver._support`, `_windows`, `WINDOW_MAX`; `gto/core.py` `Program` windows;
  `core/gtocore.c` `wa`/`wb`). A solve's combos are both players' (the tree's `ctx.active`), and every iteration worked
  out every one of them for each player, though a combo with no weight in a player's range never has reach there: its
  strategy changes nothing and its regrets matter to no answer. Now a solve works out each player's combos alone: those
  with weight in its range (anything above 0, so the Advisor's near-zero hand of ours outside our range, `_with_hero`,
  and locked and exploit-mode ranges stay in) and, under a turn's river classes, their images by the class's card swap
  (a class subtree's combo stands for its images on the class's other cards, so it has reach there). The compiled core
  lays the combos out as each player's window (`[only the first's | both | only the second's | neither]`, `Solver._perm`;
  the regrets and sums by window on upload and back on download, the cards, rankings, classes and buckets by window
  in the program) and runs a player's strategies, regrets, utilities, showdowns and folds over its own window, the
  opponent's sums over the opponent's (its reach +0 outside); numpy leaves out the same combos (its regret updates
  masked), so the fallback answers the same. Outside a player's window its regrets, sums and EVs are 0, and no answer
  shows one: every reader of a combo's mix or EV weighs it by its reach at the node (Study's combos and grid, the
  breakdown, the advice, drills and reports), which is 0 there; an evaluation with reach outside the windows runs in
  numpy. Where the windows leave out little, none: over 0.9 of the tree's combos on average (`WINDOW_MAX`; 0.8 at
  first). A turn's windows hold every river class's images of the ranges (0.95 / 0.81 of the combos on the Advisor's
  turn after a c-bet, ranges 0.70 / 0.56: a suited hand's swap is offsuit), and laying the buffers out by window costs
  each upload and download a little (`core_permute`, in place a row at a time: +4.6 and +8.6 ms natively on a limped
  turn's 31 MB of regrets and sums, the download's now +5.2). Measured first at the same ms an iteration as without,
  those turns (and two-tone ones, 0.93 / 0.82), rivers after a checked turn (0.94 / 0.76) and a 40bb flop (0.93 /
  0.75) ran as before; with the core's other work since, their windows save 5-10% an iteration, and they take them
  (A turn's windows, below). A limped pot's
  windows are its ranges (SB 0.42-0.56, BB 0.76-0.82), a 3bet pot's 0.75 / 0.66.
  Checked on the lanes fix's 686 sweep trees (four turns, two rivers and a flop, four ranges, full and TRIMS'
  levels), every one with its windows (`WINDOW_MAX` 1 for it): regrets and strategy sums the same bit for bit on 1,
  2, 4 and 8 threads. After an iteration from the same start the windows move the in-range regrets by at most
  4.4e-5 of a node's largest (median 2.2e-6) and the sums by 2.8e-7, inside the core's own distance from numpy
  (2.0e-4, 3.1e-7), and the root's EVs by at most 0.0002% of the pot (numpy's 0.0006%); numpy with its windows and
  without is the same bit for bit, and every EV outside a player's window is 0. Exploitability after 100 iterations:
  0.528% of the pot on average with the windows, 0.536 without, 0.541 without and one range nudged by an ulp; case by
  case on against off spreads as the nudge does (median 2.3% of the value, 90th percentile 14%; the nudge's 2.1%,
  13%): DCFR's rounding chaos this early, not a bias. On the Advisor's own spots, six ulp nudges of a range each, on / off (mean, % of the pot): `flop-limp` 0.1263
  / 0.1263 and `flop-limp-x` 0.1335 / 0.1335 (600 iterations), `turn-limp` 0.164 / 0.164 (300), `turn-3bp` 0.065 /
  0.065 (300) and 0.0208 / 0.0212 (600).
  In the browser bench on 8 threads (Fast, BTN vs BB's turn files, a fresh worker per spot, two runs each, alternating
  with the windows off, in a quiet hold; seconds to the answer, the same iterations and exploitability):

  | request | off -> on | the solve | iterations, % of the pot |
  |---|---|---|---|
  | `turn-limp` | 3.95 / 4.26 -> 3.20 / 3.48 | 2.72 / 3.02 -> 2.13 / 2.36 | 300, 0.16 |
  | `flop-limp` (solved live) | 1.03 / 1.03 -> 0.93 / 0.92 | 0.93 / 0.93 -> 0.83 / 0.82 | 600, 0.12 -> 0.13 |
  | `flop-limp-ip` | 1.03 / 1.08 -> 0.91 / 0.91 | 0.93 / 0.97 -> 0.81 / 0.82 | 600, 0.13 |
  | `turn-3bp` | 0.91 / 0.91 -> 0.85 / 0.86 | 0.52 / 0.51 -> 0.44 / 0.45 | 300, 0.06 -> 0.07 |
  | `turn` (no windows) | 2.13 / 2.14 -> 2.14 / 2.10 | 1.96 / 1.97 -> 1.97 / 1.94 | 300, 0.12 |

  Natively, ms an iteration from the 100th (two runs each, alternating, in a quiet hold): `turn-limp` 6.93 -> 6.30 on 8
  threads, `flop-limp` 0.88 -> 0.74 (3.17 -> 2.59 on 1), `flop-limp-x` 0.90 -> 0.73 (3.10 -> 2.44 on 1). Accurate
  warm-starts from Fast's solve: their windows are the same (the same ranges' zeros: `turn-limp`, `flop-limp`,
  `flop-limp-x`, `flop-3bp`), so nothing it copies is lost. Not done: a turn's windows per river class. Each class's
  subtree needs only the ranges plus the swaps of its own cards (0.68 / 0.55 of the combos on average on the
  Advisor's turn, against 0.95 / 0.81 for all of them at once), but one order of the combos can't make every class's
  window contiguous: it takes a layout per class (its own order, windows, cards, rankings and buckets; the class
  gather composed with each layout's order; numpy masked per row). Its bound, the windows set to the ranges alone (not
  a correct solve), natively on 8 threads: 1.2x on `turn` and `turn-limp`, 1.06x on two-tone.
  Tests: `tests/test_core.py` `TestWindows` (combos outside a range: their regrets, sums and EVs 0, the rest within
  rounding, core and numpy; off; none where they leave out little), `tests/test_study.py` (Study's combos and grid
  only in the range, our hand outside it). `core.wasm` 60 -> 65 KB, `core-mt.wasm` 83 -> 91 KB.
* **A turn's windows, and less around its iterations** (`WINDOW_MAX`; `core/gtocore.c` `strategy`; `Ctx.class_perm`;
  `gto/core.py` `Program._get`, `_laid`; `Advisor._street_info`). Profiled at one thread on the Advisor's own spots
  (each answer's solve captured natively at Fast and replayed: `turn`, `turn-x`, `turn-2tone`, `turn-3bp`,
  `turn-limp`, the turns' trees on threads, `river`, `river-x`, `river-3bp`, `river-limp`), a single-raised turn's
  iteration goes to the showdowns' sums (`ranking`, 32% of the samples), the passes' own arithmetic (`cfr`, 21%),
  regret matching (`strategy`, 19%), the folds' sums (`compat`, 16%), the other terminals (7%) and the river classes'
  spreads (4%). Per card dealt it costs what postflop-solver's does (above), so what changed is around those:
  - Turns take their windows (`WINDOW_MAX` 0.8 -> 0.9: a single-raised turn's 0.95 / 0.81 of the combos, two-tone
    0.93 / 0.82; and rivers after a checked turn, 0.94 / 0.76, and a 40bb flop, 0.93 / 0.75). Natively on one
    thread, seconds for 300 iterations (rivers 1000) and exploitability, without -> with:

    | spot | seconds | % of the pot from equilibrium | ulp nudges of a range, without |
    |---|---|---|---|
    | `turn` | 4.15 -> 3.90 | 0.1142 -> 0.1167 | 0.1207, 0.1181 |
    | `turn-x` | 5.31 -> 4.74 | 0.1683 -> 0.1686 | 0.1695, 0.1716 |
    | `turn-2tone` | 6.78 -> 6.27 | 0.1524 -> 0.1498 | 0.1512, 0.1494 |
    | `turn` (threads' tree) | 6.29 -> 5.68 | 0.1429 -> 0.1477 | |
    | `turn-x` (threads' tree) | 7.00 -> 6.47 | 0.2103 -> 0.2103 | 0.2131, 0.2043 |
    | `turn-2tone` (threads' tree) | 4.61 -> 4.31 | 0.0939 -> 0.0934 | 0.0906, 0.0905 |
    | `river` | 0.11 -> 0.10 | 0.0080 -> 0.0092 | 0.0082, 0.0094 |
    | `river-x` | 0.11 -> 0.10 | 0.0157 -> 0.0156 | 0.0156, 0.0155 |

    The root's mix moves by at most 0.02-0.05 a combo and 0.08 points a range frequency, where nudging one range by
    an ulp moves it 0.02-0.16 and 0.22: DCFR's rounding, not a bias. Natively on 8 threads (a quiet hold, two runs
    each, alternating), ms an iteration: `turn` 3.85 / 3.85 -> 3.68 / 3.62, the threads' tree 5.48 / 5.46 -> 5.24 /
    5.41. The spots with windows already (`turn-3bp`, `turn-limp`, `river-3bp`, `river-limp`) are the same bit for
    bit.
  - Regret matching in one loop an action (`strategy`): each action's regrets read once, the clipped value stored and
    added to the sum in the same loop, the reciprocal on the last action, and the discount (player 0's regrets as
    the second pass reads them) written back there too, where it took four loops. The same operations on each float
    in the same order, so the same floats. Natively `turn` 15.4 -> 15.2 ms an iteration, `turn-3bp` 2.00 -> 1.96; in
    wasm `turn` 17.4-18.5 -> 17.3-17.7.
  - A download leaves the arena's buffers in the solver's order (`Program._get`) instead of laying them out by
    window again: every call that reads them uploads them first, and an iteration or evaluation that doesn't lays
    out what it reads (`_laid`). A download of `turn-limp`'s 60 MB of regrets and sums 8.7 -> 5.2 ms natively.
  - The river classes' arrays once (`Ctx.class_perm`, cached by the classes): a turn's river nodes after a c-bet
    (13) share one set of classes, and their gathers were built for each. Natively the turn's tree 10.7 -> 3.8 ms,
    its program and first iteration 52 -> 36 ms; the same arrays.
  - The street's strengths in a few numpy calls (`Advisor._street_info`: each runout's percentile ranks from one
    `eval_boards` call, a bincount and a cumulative sum, where a loop ranked each runout): on a flop 38.7 -> 7.6 ms
    natively, a turn 18.6 -> 3.5; the same arrays.

  - In the browser (`scripts/bench_browser.mjs`, Fast, the previous tree and this one alternating; seconds to the
    answer, at the same iterations and exploitability):

    | spot | 1 thread (3 rounds) | 4 threads (2 rounds) | 8 threads (1 round) |
    |---|---|---|---|
    | `turn` | 3.46 / 3.64 / 4.26 -> 3.14 / 3.32 / 3.44 |  | 2.33 -> 2.09 |
    | `turn-x` | 4.12 / 4.24 / 5.37 -> 3.85 / 3.90 / 4.06 | 4.95 / 5.49 -> 4.76 / 5.37 | 3.47 -> 3.15 |
    | `turn-3bp` | 1.12 / 1.19 / 1.26 -> 1.11 / 1.11 / 1.14 | 1.11 / 1.13 -> 1.03 / 1.07 | 0.81 -> 0.75 |
    | `turn-limp` |  | 4.65 / 4.96 -> 4.74 / 4.89 | 3.37 -> 3.24 |
    | `ro-river` | 4.08 / 4.21 / 4.51 -> 3.91 / 4.22 / 3.98 |  | 3.00 -> 2.86 |
    | `ro-turn` |  |  | 0.67 -> 0.78 |
    | `flop-40` |  |  | 0.38 -> 0.41 |
    | `river-40` | 2.68 / 2.76 / 3.01 -> 2.48 / 2.68 / 2.51 |  | 1.94 -> 2.20 |

    On one thread the medians are 5-9% less. On the site's 8 threads (one round, a machine shared with other jobs)
    the turns are 4-10% less; the small spots moved both ways between rounds (`river-40` 2.11 -> 1.93 in a round
    under more load, `flop-40` 0.34 -> 0.37), and natively on 8 threads the 40bb flop's windows cost nothing (0.566
    -> 0.562 ms an iteration; on one thread 1.46 -> 1.37).

  Tests: `tests/test_core.py` `test_a_download_leaves_the_arena_in_the_solvers_order` (after a download the arena
  holds the solver's buffers in its order; laid out again, they are an upload's; an evaluation uploads only the
  sums; a solve run on after it is the same bit for bit). `core.wasm` 65 -> 69 KB, `core-mt.wasm` 91 -> 95 KB.
* **A live flop's tables on threads** (`core/gtocore.c` `core_tables`; `gto/core.py` `Tables.batch`,
  `Program._tables_ahead`; `Ctx.table_request`; `advisor._held_bytes`). Unstored flops, other depths and borrowed
  spots all end in a live flop solve. Profiled end to end at one thread (Fast, the browser's settled checks), seconds:

  | spot | answer | iterations | tables: core / Python around it | `_held_bytes` |
  |---|---|---|---|---|
  | `flop-40`, natively | 0.59 | 0.42 | 0.030 / 0.027 | 0.007 |
  | `flop-limp`, natively | 1.31 | 0.90 | 0.094 / 0.031 | 0.015-0.021 |
  | `flop-40`, Pyodide | 0.68 | 0.46 | 0.064 / 0.039 | 0.018 |
  | `flop-limp`, Pyodide | 1.58 | 1.07 | 0.183 / 0.057 | 0.037 |

  The rest is the tree (0.01-0.02 s), the answer's packing (0.02-0.03) and the Python of the solve. The tables (a
  flop's 13 turn bucket tables, a limped flop's 13 river-strength tables too, and the flop's equity leaf over 196
  runouts) were one core call each on one thread, while the iterations run on the site's 8: there the 40bb flop
  answers in 0.3-0.4 s, of which its 0.1 s of tables in Pyodide were a quarter or more. The Python around them is mostly the river boards' strengths, which
  `eval_boards` keeps by rank multiset (`_plain`: a session's first flop pays for them).
  - The tables in two batches (`Program._tables_ahead`, before the walk that lays the tree out): the turns' bucket
    tables and the equity leaves, then the river-strength tables, which read the turns' buckets. `core_tables` runs a
    batch as one job on the core's threads, a table a task, the costliest first (by the work each does, measured
    natively), the equity leaf's rows in chunks (its strengths laid out once); each helper's lane is sized from the
    tables' sizes (not measured: a table's scratch is known), and a batch holds at most 64 MB of inputs and outputs
    (`BATCH_BYTES`: a library flop's 49 turn leaves are 4 MB each). Each table is `core_table`'s code on its own
    inputs, so the floats are the same bit for bit: the answers hash the same at 1, 2 and 4 threads (`flop-40`,
    `flop-limp`, `turn-3bp`), and so do the library's flops.
  - `_held_bytes` (the cached solves' accounting, after every new solve) checks exact types first and reads a class's
    slots once: a limped flop's solve 36 -> 28 ms in Pyodide, the 40bb flop's 18 -> 13.5, the same bytes.
  - Measured and left: the threads' first iteration, which measures their lanes in serial chunks, costs about an
    iteration (the 40bb flop natively on 2 threads: 47 ms with the program built, then 1.8 ms against 1.2 steady).
  - Practice's re-solves, counted: 24 hands natively (Fast; BTN and CO against the big blind, the 3bet pot, the
    limped pot; both players drawing GTO moves, hands dealt from the ranges entering the flop, your advice asked at
    each of your decisions and each opponent move through `Advisor.play`, as `practice.js` does). 3 hands re-solved
    from a node for a rare line (0.07-0.56 s), 2 solved an earlier street again for an opponent's move (0.47 s, and
    a limped hand's river reply 3.5 s for its flop and turn): 5% of the 93 s the hands' solves took. That last kind is
    an opponent's move read from its own seat's view (`play`'s fallback when your line is rare in the solve, or its
    hand has no weight there), whose ranges can differ from your advice's by a near-zero-weight combo (a hand outside
    its range there: `_with_hero`, `_as_advised`), so it missed the solves your advice cached (since read: the
    next bullet).
  - Natively (one quiet hold, the previous tree and this one, the last two of three runs): on 8 threads the tables'
    core calls 26 -> 5.6 ms on `flop-40` and 81 -> 18 on `flop-limp` (on 4 threads 7.9 and 25), the whole step with
    its Python 30 -> 10 and 87 -> 25 ms, the answers 0.180 -> 0.160 s and 0.366 -> 0.310 (on 4 threads 0.191 -> 0.172
    and 0.410 -> 0.346); on one thread the same within 1-2%.
  - In the browser (`scripts/bench_browser.mjs`, Fast, the previous tree and this one alternating; seconds to the
    answer, the same iterations and exploitability):

    | spot | 4 threads (2 rounds) | the site's 8 threads (1 round) |
    |---|---|---|
    | `flop-40` | 0.39 / 0.42 -> 0.32 / 0.32 | 0.31 -> 0.29 |
    | `flop-limp` | 0.93 / 1.03 -> 0.81 / 0.66 | 0.70 -> 0.61 |
    | `flop-near-80bb` (a flop the library stores at 100bb, solved at 80bb) | | 0.39 -> 0.34 |
    | `ro-turn` (the turn report on a library flop, whose flop is solved for it) | | 0.58 -> 0.53 |
    | `turn-3bp` | 0.78 / 0.85 -> 0.79 / 0.65 | 0.59 -> 0.58 |

    The turn, a 40bb river, `flop-near` and a library flop within the noise (1.57 -> 1.56, 1.35 -> 1.29, 0.28 ->
    0.27, 0.02 either way).

    In Pyodide on one thread the tables take what they took (0.221 / 0.235 -> 0.237 / 0.226 s on `flop-limp`).

  Tests: `tests/test_core.py` `test_a_batch_builds_each_table_the_same_on_any_number_of_threads` (a flop's equity
  leaf, bucket and river-strength tables in batches on 1, 3 and 8 threads are the ones built one by one, bit for bit;
  the helpers' lanes never short; each table taken). `core.wasm` 69 -> 74 KB, `core-mt.wasm` 95 -> 96 KB.
* **Practice opponents' moves from their own seat read your advice's solves** (`Advisor.play`, `_seen_from`).
  Heads-up after the flop an opponent's move is read off your advice's solve of the street; when your line is rare
  in it (under `RARE_LINE` of your range) or the opponent's hand has no weight there, the move is that seat's own
  advice. That advice solved every earlier street again: its ranges took its cards out of yours where your
  advice takes yours out of its, and put its hand in at a near-zero weight where yours had been, so none of its
  solves were the cached ones. Now, heads-up from the street's start, it solves with your advice's ranges (your
  cards out of its range, your hand in where your range doesn't hold it, its own added on the street it acts), so
  the earlier streets are the solves your advice made, and so is the street itself where its hand has weight
  there. As when the move is read off your solve, its own cards stay in your range: a blocker's edge.
  - The Practice simulation above (24 hands natively at Fast, both players drawing GTO moves; two seeds): the
    opponents' moves from their own seat (3 and 5 a seed) solved 5.3 and 11.9 s, now 0.09 and 0.02 (the
    earlier streets 3.8 and 11.8 s, now none; a limped river's reply 3.37 s -> 0.005); all the hands' solving
    88.5 -> 81.9 and 90.7 -> 79.1 s. Every one of the 278 moves was drawn the same, those 8 with the same mix.
  - At Practice's quick answers (Fast in the page: an earlier street not solved already is narrowed by the betting
    rules) those moves took no earlier street's solve, and now read your advice's (its flop narrowing as yours,
    not the rules'): 1-2 s less a seed (34.9 -> 32.9 s, a flop reply 1.26 -> 0.24; 26.9 -> 25.9 up to the hand
    that changed). One move of 186 changed: a turn reply to a 125% bet that its own 82-iteration solve didn't
    read as rare (call 89%) is now read off your advice's turn solve, finds the bet rare there and is solved
    again from the bet (300 iterations): fold.
  - Where the reply mixes (the button's hands outside its range on a turn checked to it, solved to 1000
    iterations a street): check and bet as before; the split between two bet sizes whose EVs are 0.00-0.05 bb
    apart moved, as a near-zero-weight hand's does, and the EVs by up to 0.3 bb (the narrowing's flop solve and
    the blockers). Keeping its own cards out of yours on the street it acts moved them as much.
  - Looked at and left: your hand outside your range on a library flop takes a live flop solve (1.24 s natively).
    The entry stores strategies for its ranges' combos only; a hand at a near-zero weight plays a best response
    through turns and rivers the entry doesn't hold, and the approximate flops read the same ranges off similar
    flops. Practice grades your move by that solve, and the opponents' moves are read off it, so they cost nothing
    more.

  Tests: `tests/test_gto.py` `test_fallback_reads_the_earlier_streets_the_advice_solved` (a turn move for a hand
  the big blind 3bets: one new solve, the turn, where it was two).
* **A board's ranks ranked once** (`gto/evaluator.py` `eval_boards`, `_plain`). A program's showdowns (and numpy's
  tables) rank every combo on every five-card board the tree deals: on a flop solved live, its runouts and each turn's
  rivers, about 1200 boards. `eval_boards` ranked each one the same way. For each of the 169 pairs of hole ranks it
  took the 21 five-card subsets as prime products and searched treys' table for them (`searchsorted`, 3549 a board),
  then applied the board's flushes and blockers. Without a flush, though, a hand's rank depends only on its two ranks
  and the multiset of the board's ranks, and a flop's runouts have 91 of those between them (6188 exist).
  Now each multiset's 169 ranks are worked out the first time it comes up and kept as a row (`_PLAIN`, int16, 2 MB
  when all 6188 are there; rows are only added, under a lock). Every board reads its row, and the flushes and blockers
  are applied as before. The numbers are the same. A turn solved after its flop, and the next hand's flop with the
  same ranks, read the rows that solve added.
  Browser bench on 8 threads (Fast, BTN vs BB's turn files, a fresh worker per spot, two runs each alternating with
  the code before, in a quiet hold), seconds to the answer:
  - `flop-limp` 0.79 / 0.81 -> 0.70 / 0.71 (the flop's `Program._tables` 0.274 / 0.287 -> 0.190 / 0.190)
  - `flop-limp-ip` 0.79 / 0.79 -> 0.71 / 0.71 (0.280 / 0.280 -> 0.200 / 0.194)
  - `turn-limp`'s flop tables 0.276 -> 0.190, and its turn's 0.018 -> 0.006 (the flop's rows)
  - `turn`'s tables 0.028 -> 0.018
  Iterations and exploitability are unchanged (600 at 0.13% of the pot, 300 at 0.16 and 0.12).
  Tests: `tests/test_gto.py`.
  - `test_eval_boards` checks against treys' own 7-card evaluation and `eval_board` on boards of every kind: flushes,
    straights, the wheel, and two pair through quads. It also covers one call, chunks and the reverse order.
  - `test_eval_boards_kept_across_solves` takes a paired flop's runouts, then another flop's with the same ranks in
    other suits, 64 boards a call as `Solver._strengths` asks. The second adds no row, and trips, full houses, quads
    and flushes come out as `eval_board`'s.
* **The root's EVs worked out once** (`Solution.fresh`, `evs_at`; `Advisor._hu_run`, `Advisor.advise`). A Fast turn's
  answer worked out its root's action EVs (`Solver.node_action_evs`, a pass over the tree on the average strategies)
  three times on the same solve: in `solve()`'s `_refresh`, again in `_hu_run`'s, and again for the answer. Now the
  `Solution` notes the iteration count its root's mix and EVs were worked out at (`evs_at`). `_hu_run` refreshes
  only if iterations ran since then, and the answer at a solve's root, with the solve's own ranges, reads them. The
  numbers are the same.
  `node_action_evs` calls per answer, natively: `turn` and `turn-x` 3 -> 1, `turn-limp` 5 -> 3, `turn-2tone` 5 -> 2,
  `flop-limp` 3 -> 2. The answers are the same, all but the sampled action, which is random.
  In the browser on 8 threads, an evaluation of a Fast turn's root takes 13 ms (`turn`; 14-16 on `turn-x`), so an
  answer saves 26-30 ms.
  Tests: `tests/test_settled.py` `test_the_roots_evs_worked_out_once`. A river's answer evaluates its root once, at
  its iteration count. Its EVs are a fresh evaluation's bit for bit, and so are the answer's. An iteration on makes
  them stale.
* **Evaluations below the root on the helpers** (`core/gtocore.c` `core_eval`). A node's action EVs evaluate each of
  its children (`node_action_evs`, `node_action_br_evs`, Study's every node, the exploit modes' best responses). Below
  the root that ran on one thread, because lanes measured there would leave out the rest of the tree.
  Once a kind's lanes are sized, by a training pass or by that kind's first evaluation at the root, an evaluation
  below the root hands its chance nodes' rows to the helpers too. A subtree's chance nodes are the root's, and a chunk
  of their rows needs the same scratch, so the lanes fit (a lane that came short grows as before). Every row gets the
  same arithmetic, so the floats are the same on any number of threads.
  In the browser on 8 threads, the Fast turn root's remaining evaluation took 13 -> 5.4 ms on `turn` and 14-16 ->
  5.9-6.0 on `turn-x`.
  With the root's EVs worked out once, in the same hold:
  - `node_action_evs` per answer: `turn` 39-40 ms (3 calls) -> 5.4 (1); `turn-x` 43-47 -> 5.9-6.0; `turn-limp`
    54.5-55.1 (5) -> 13.0-13.1 (3)
  - the answer's time outside the solve: `turn` 0.13 / 0.13 -> 0.10 / 0.10 s; `turn-x` 0.14 / 0.15 -> 0.11 / 0.11
  - whole answers with the cached ranks too: `turn` 1.69 / 1.77 -> 1.68 / 1.71; `turn-x` 1.97 / 2.06 -> 1.99 / 2.00
    (each within its iterations' own run-to-run spread, 0.08 s); `turn-limp` 2.69 / 2.69 -> 2.53 / 2.55
  The same iterations and exploitability.
  Tests:
  - `tests/test_core.py` `test_evaluations_below_the_root` evaluates a turn's root children, then the root, then the
    children again, full and trimmed, on 2, 4 and 8 threads. It does this for a program built to evaluate (no lanes
    sized yet) and for one after an iteration. The floats are one thread's. The helpers ran rows once lanes were
    sized and none before, and no lane grew. On the core before, the helpers ran none.
  - `tests/js/core.test.js`: a helper that traps holding rows of an evaluation below the root (`core_fail`, worker.js's
    own helper script, on 2 and 8 threads). The evaluation throws (`CoreError`) instead of waiting for good. On the
    core before, no evaluation below the root reached a helper.
* **The multiway core on threads** (`core/mwcore.c` `pass`, `rwalk`, `job`, `helper`, `mw_threads`;
  `gto/mwcore-mt.wasm`). The multiway core (3- and 4-way flop and turn solves, How it works) ran on one thread in the
  browser while the heads-up core had helpers. Now a pass runs its root-street leaves as one job: a walk down
  (`COLLECT`) keeps each leaf's reaches (a root Exit's bucket subtree below it, a fold's deal masses), every leaf runs
  as a task in some thread's own scratch (a lane, sized by `need` for the deepest subtree), and the pass proper (`READ`)
  takes their values where the leaves are and updates the root street's decisions after them, as the serial pass does.
  A leaf's subtree is its own (its decisions' regrets, sums, kept strategies and flags; the tensors only read, a fold's
  deal masses in the thread's own scratch), so the tasks write nothing in common and each one's arithmetic is the
  serial pass's: the same floats bit for bit on any number of threads (regrets, sums, best-response gains, values and
  action EVs on 1 and 4 threads: 3-way flop and turn, a player all-in, 4-way; natively and in the wasm). The browser
  runs `gto/mwcore-mt.wasm` on a shared memory of its own with worker.js's helpers (the same helper script, `mw_`
  exports), which wait in `mw_helper` (spinning a while, then `memory.atomic.wait32`); a helper that fails marks itself
  (`mw_dead`) and the solve's wait traps instead of hanging, as the heads-up core's does. The bundle grows by
  `mwcore-mt.wasm`, 127 KB (30 KB gzipped). 3-way rivers are solved per combo in numpy, not in the core: unchanged.
  Natively, ms an iteration of the Advisor's own solves (40 iterations after it answered, medians of 2,
  alternating):

  | solve (CO opens, BTN and BB call; 4-way: HJ opens, CO, BTN and BB call) | 1 thread | 2 | 4 | 8 |
  |---|---|---|---|---|
  | 3-way flop (Kh8d3c), Fast | 23.1 | 13.0 | 7.8 | 6.1 |
  | 3-way flop, Accurate | 29.7 | 16.5 | 10.4 | 8.0 |
  | 3-way turn, the flop checked through | 14.7 | 8.6 | 5.1 | 3.6 |
  | 3-way turn, a flop bet called twice | 18.7 | 10.9 | 6.2 | 4.5 |
  | 4-way flop, Fast | 14.6 | 8.8 | 5.6 | 4.4 |
  | 4-way flop, Accurate | 21.3 | 12.8 | 7.5 | 5.7 |
  | 4-way turn, Fast | 3.0 | 1.7 | 1.0 | 0.9 |
  | 4-way turn, Accurate | 18.0 | 8.8 | 5.3 | 4.6 |

  In the browser bench on 8 threads (the live page's), a fresh worker per spot, the one-thread module before (the same
  worker with `mwcore.wasm`), two runs each, alternating; seconds to the answer:

  | request | one thread -> 8 | iterations, % of the pot |
  |---|---|---|
  | `mw-flop` live (3-way library off), first to act, Fast | 8.08 / 8.08 -> 4.40 / 4.42 | 230-234, 0.17 -> 400, 0.05 |
  | `mw-flop` solved live, Accurate | 17.51 / 17.51 -> 7.12 / 6.93 | 400, 0.05 |
  | `mw-flop-x` solved live, checked to, Fast | 4.98 / 5.03 -> 2.38 / 2.38 | 400, 0.03 |
  | `mw-flop-x` solved live, Accurate | 7.80 / 7.82 -> 4.39 / 4.34 | 400, 0.03 |
  | `mw-turn` (the flop checked through), Fast | 7.69 / 7.69 -> 2.57 / 2.60 | 400, 0.04 |
  | `mw-turn`, Accurate | 7.64 / 7.63 -> 2.59 / 2.53 | 400, 0.04 |
  | `mw4-flop`, Fast | 5.66 / 5.63 -> 2.83 / 2.74 | 200, 0.03 |
  | `mw4-flop`, Accurate | 8.90 / 8.95 -> 4.12 / 4.03 | 200, 0.05 |
  | `mw4-turn`, Fast | 1.25 / 1.27 -> 0.99 / 0.95 | 100, 0.04 |
  | `mw4-turn`, Accurate | 4.37 / 4.16 -> 2.12 / 2.10 | 100, 0.04 |

  The iterations alone: `mw-turn` 6.6 -> 1.7 s, `mw4-flop` 3.6 -> 1.1 (Accurate 5.2 -> 1.5); the rest of a request
  (the tree and its tables built in numpy, the earlier street's narrowing, the passes after the solve) now takes as long
  or longer: 0.9 s of `mw-turn`'s 2.6, 1.7 of `mw4-flop`'s 2.8. A flop in the 3-way library answers at once either way
  (0.5 s), and the 3-way river, solved in numpy, takes as long as before (`mw-river` 6.6 s, 100 iterations, 0.09%).
* **The core starts again after a trap** (`core.restart`, `mwcore.restart`, `RESTARTS`; `gto/static/worker.js`
  `start`, `reload`; `Advisor.expl_target`, `advisor.OVER_TARGET`). A trap in the browser's compiled core (a helper
  that failed, the memory refusing to grow) turned it off for the worker: that solve and every later one ran in numpy,
  a Fast turn in 12 s at 4.9-6.9% of the pot from equilibrium where the core answers in 2.3 s at 0.15% (A helper that
  fails, above). Now the worker starts the module again (`reload`: a new instance on a fresh memory with new helpers,
  the old ones ended) and the run starts over in it, once, from the solver's arrays as they were before it, in what's
  left of the budget; the passes after a solve the same (`core.evaluate`; the multiway core's values, best responses
  and EVs). Nothing the failed call wrote reaches the retry: a failed run's arena is never read back
  (`Program.download` only after a run that finished or was stopped, `MultiwaySolver._run_core` the same), a pass's
  output is read only after it returned, and the program is laid out again in the new memory (`Program.valid`), so the
  answer is the same floats as a run that didn't fail (the tests: failing on the first chunk or a later one, in a run
  or a pass, heads-up and 3-way). Failing again in the new instance, or after `RESTARTS` (3) restarts a session, the
  core is off and numpy answers, as before. Natively a trap ends the process: nothing changes there. In the browser
  bench, helper 0 failing on its first chunk (`BENCH_FAIL`) or task (`BENCH_MWFAIL`, new), a worker's session of
  requests:

  | session, Fast (seconds, iterations, % of the pot) | before: the core off, numpy | now: the core started again |
  |---|---|---|
  | heads-up helper 0 fails, 8 threads: `turn` | 11.9 s, 36-39, 4.9-6.9% | 2.38 s, 300, 0.15% (2.33 s without) |
  | then `turn-limp`, `river` | in numpy | 4.10 s, 300, 0.16%; 0.09 s, 300, 0.05% |
  | heads-up helper 0 fails, 2 threads: `turn` | 12.2 s in numpy | 2.53 s, 200, 0.23% (a smaller tree on 2 threads) |
  | the multiway core's helper 0 fails, 8 threads: `mw-turn` | in numpy | 2.61 s, 400, 0.04% (2.57 s without) |

  (The bench prints each core's restarts with `BENCH_FAIL` or `BENCH_MWFAIL`: one each, and both still on.)

  An answer that ran short now says so. A heads-up answer after the flop more than `OVER_TARGET` (10) times its speed's
  `expl_target` from equilibrium (Fast's settled stop, 0.2% of the pot on threads and 0.3 without; Accurate 0.1) is
  marked approximate: the badge reads "approximate" where 4.9-6.9% read "fair" (and under 3% "converged"), its tip
  gives the speed's target, and a note says the solve ran short (a busy or slow device, a big tree, or the compiled
  solver failed and the slower one answered) and to take its mix and EVs as a rough guide. The threaded page's Fast
  flops answer at 0.03-0.4%, turns at 0.12-0.18%, so 2% is far past them; numpy's Fast flops were at 2.4-7.1%. Not
  multiway solves (a 3-way flop at Accurate 0.7-1.9%; one over 10% goes to the betting rules), nor Quick's answers.
* **3-way rivers in the compiled core** (`RSHOW`, `show()` in `core/mwcore.c`; `Program._river` in `gto/mwcore.py`;
  `MultiwaySolver._core`). The 3-way river was the one multiway solve still run per combo in numpy: `mw-river` took
  8.0 s on the live page (Fast ran out its budget at 93-95 iterations, Accurate 8.5-8.7 s). Now the core runs it, the
  river the program's root street and its showdowns the leaves: a terminal two or three players reach is an `RSHOW`
  record (its pots in layers, each with its players and payouts; a tournament's prize columns), which `show()` values
  by the exact 3-way kernel (card removal by inclusion-exclusion over the 52 cards) in one sweep of the hands sorted
  by strength: per tie group each card's sums of the weaker and the tied hands, a hand's sums corrected by its own two
  cards' rows in 4-wide SIMD, ties over the group's cards only, O(52 n) a terminal with no n x 52 tables. The header
  carries the board's strengths (words 22-25). On threads each showdown is a task, as the flop's and turn's leaves: the
  same floats on any number of threads (tested). Three players, exact, alternating, at most one all-in; else numpy as
  before. One iteration from the same arrays: regrets and sums within ~1e-6 (relative) of numpy's, best responses,
  values and EVs within 1e-7 of the pot (6 boards: all hands tied, side pots, rake, a tournament's prizes). At 100
  iterations on the tests' ranges the exploitability is the same to 4 decimals (5 spots, 0.09-0.23% of the pot),
  strategies within 5e-4. On the Advisor's ranges the two solves drift apart after ~30 iterations (regret matching
  amplifies rounding: within 6e-6 at 20 iterations, 2e-2 at 40), as numpy drifts from itself: nudging the reaches by
  one ulp (6 runs each) moves numpy's exploitability at 100 iterations over 0.051-0.138% of the pot (mean 0.079) on
  `mw-river-bc`, the core's over 0.052-0.103 (0.084); `mw-river` 0.207-0.232 (0.219) and 0.207-0.221 (0.214). Both
  evaluate a strategy the same to 5 decimals. Natively, ms an iteration of the Advisor's own solves:

  | solve (CO opens, BTN and BB call, Kh7d2c9s4h checked to the river) | numpy | core, 1 thread | 2 | 4 | 8 |
  |---|---|---|---|---|---|
  | `mw-river`, BB first to act | 37.1 | 3.03 | 1.75 | 0.99 | 0.83 |
  | `mw-river-bc`, BB facing a bet and a call | 6.01 | 0.41 | 0.24 | 0.21 | 0.22 |

  In the browser bench, two runs each, seconds to the answer:

  | request | before (numpy), 8 threads | core, 8 threads (the live page) | core, 1 thread |
  |---|---|---|---|
  | `mw-river`, Fast | 8.03 / 8.01 (93-95 iterations) | 0.87 / 0.88 | 1.15 / 1.19 |
  | `mw-river`, Accurate | 8.53 / 8.69 | 0.88 / 0.86 | 1.15 / 1.17 |

  100 iterations, 0.08% of the pot (0.07-0.08 before). The solve itself 6.8-7.5 s -> 0.13 s (0.40-0.43 on one
  thread); the rest of the request (the earlier streets' narrowing, the tree, the passes after) is 0.74 s of the 0.87.
  `mwcore.wasm` grows 121 -> 127 KB, `mwcore-mt.wasm` 127 -> 133 KB.
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
* **3-way pots are solved** (`gto/multiway_solver.py`, `docs/multiway_research.md`) where three players are live
  (one of them may be all-in for less: side pots), on every street at every speed (Fast's flops in a smaller game,
  `advisor.MW_FAST_STREETS`; Quick's flops and turns in smaller ones, `MW_QUICK_FLOP`, `MW_QUICK_TURN`), and four on
  the flop or turn by the 4-player solve (N-player pots in How it works). A solve over 10% of the pot from
  equilibrium after its budget hands over to the rules. The result has the solver's
  mix, EVs (relative to folding now), grid and each player's best-response gain, and "exploitability" is their
  mean. A river iteration costs 30-90 ms natively and 0.13-0.16 s in Pyodide (measured, `scripts/bench_browser.mjs`),
  so Fast's 2.5 s / 8 s run 25-80 (the browser's: 41-48, 0.3-0.4% of the pot from equilibrium). Turn and flop solves
  bet the later streets by hand buckets (Known limits; `docs/multiway_research.md` has the model): a turn iteration
  costs 0.1-0.2 s / 0.18 s, a flop one (bets of 33 and 75%) 0.1-0.4 s / 0.34 s checked to and 0.7-0.95 s first to
  act, after 0.6-1.7 s of tables natively, so in numpy both waited for Accurate (30 s / 60 s; unloaded natively flops checked to
  run 200-290 iterations, 0.15-0.25% of the pot from equilibrium, first to act, a bigger tree, 100 and 0.8%; in the
  browser 161 and 0.27%, 56-77 and 3.3-6.3%; 8 buckets, 112 iterations in the same 60 s, was no closer to the deeper
  solve below on its 8 spots: best action 76% against 72%, 0.020 bb given up against 0.029, but EVs 0.79 bb off
  against 0.47 and range bets 6.1 points against 3.2). With the compiled multiway core (How it works: 3-way
  postflop) Fast solves them too: a first-to-act flop (Kh8d3c, CO opens, BTN and BB call) runs 111-125 iterations in
  native Fast's 5 s (0.7-1.05% of the pot from equilibrium) and 203 in the browser's 8 s (0.25%), checked to twice
  400 (0.03%); Accurate settles it at 400 in 21-25 s natively and 13 s in the browser (0.05%). The budget covers the tables and the gains and EVs read off after the
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
* Multiway pots neither solver takes (5+ players live on the flop or turn, 4 with two or more all-in, 6+ on a river
  in the browser's Quick and Fast, or one of the cases above) are not solved. Instead a Monte
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
* 4-way flops and turns are solved now (N-player pots in How it works: the 4-player bucketed solve, which against
  the per-combo turn references below gives up 0.015% of the pot first to act and -0.02% facing a bet and a call, in 4-12 s of CPU against 15-17 min); 5+ way flops and turns stay with these rules (GTO Wizard
  doesn't solve 4+ way postflop at all). Scored against
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
  6.5% of the pot where the rules gave up 1.5% (see N-player pots), so the rules stayed until the river was bet on
  by buckets.
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
  still gets a strategy and EVs without changing anyone else's play; the next street
  narrows the ranges through that same solve (Accuracy vs speed, "Your hand out of your range").
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
  BTN, CO, HJ, UTG and SB vs BB single-raised pots; the HJ's open flatted by the CO, the BTN's by the SB, the
  UTG's by the HJ, the CO's by the SB and the UTG's by the BTN; eight 3bet pots: the BB's and the SB's 3bet of the
  BTN, the BTN's, the SB's and the BB's of the CO, the HJ's of the UTG, the CO's of the HJ and the BB's of the SB. A spot is matched by the two ranges entering the flop (as the charts replay them: other table sizes, stacks
  or charts simply don't match), the pot and the stack; heads-up from the flop only, not in exploit mode, with locks on the
  villain, custom ranges or Optimize sizes.
* **Other games** (`flop_library.Format`, `FORMAT_SPOTS`): a spot of another stack depth, ante, table size or
  straddle is one of the spots above played in that game (`format_line`: the same raises and calls by the same
  seats at the same sizes, everyone else folding in that table's order), set up as the advice plays it there
  (`spot_setup(line, players, stack, ante, straddle)`: that game's charts, see Preflop), so the advisor finds it by
  the same key (the two ranges, pot and stack) and the page fetches it by its id (`srp_btn_bb_40bb`, "BTN vs BB,
  40bb"). The index carries its game (`"game": {"players", "stack", "ante", "straddle"}`), and Study, the drills
  and the flop reports play its hands in that game. Listed: BTN, CO, HJ, UTG and SB vs BB and BB vs BTN's 3bet pot
  at 40bb and 200bb, built by `--format` (until built they are solved live; their entries deal the river, as every
  spot's do).
* **Flops**: canonical flops matched up to suits (the ranges are suit-symmetric, so a stored flop's solve is any
  suit permutation's, its combos permuted); other flops are solved live. Each spot stores a representative set
  (`representative_flops(n)`: the 1755 suit-isomorphic flops sorted by texture and sampled by weight, so each suit
  pattern, pairing and height gets its share): 552 flops (38% of all flops; it holds the 184, 12.7%, that the first
  three 3bet pots stored before) for the eight 3bet pots and the opens flatted behind or by the small blind (the HJ's
  by the CO, the BTN's by the SB, the UTG's by the HJ, the CO's by the SB, the UTG's by the BTN); every flop for the
  opens the big blind calls (`SPOT_FLOPS`).
* **Coverage** (`scripts/flop_library_coverage.py`: the 6-max 100bb preflop game played by the charts, the flop by
  the stored strategies): 36% of hands go heads-up to the flop, and the library answers 77.2% of their flop
  decisions (74.5% before the eight 3bet pots on 552, 71.6% before the five flatted opens on 552, 49.2% before HJ,
  UTG and SB vs BB on every flop, 36.1% before CO vs BB on every flop and SB vs BB on 552, 21.3% before format 2
  made room, 9.4% with 184 flops everywhere). Of the rest, a spot not in the library is 9.5 points (the other 3bet
  pots ~0.6 each: the BTN's and the SB's 3bet of the UTG, the BB's and the SB's of the HJ; 4bet pots; 4.5 of them
  answered approximately at once off the spot whose ranges are nearest, `flop_approx.near_spot`), a flop not
  stored 10.2 (the flatted opens' and the 3bet pots': answered approximately at once from the stored flops most like
  it, `scripts/instant_coverage.py` 77.2% -> 87.5% with them, 91.9% with both), a raise 3.2. At 40-200bb, where no
  spot is built, the 100bb spots answer 43-89% of the heads-up flop decisions approximately at once (`--stack`). At 8- and 9-max
  the pots that play as 6-max ones are answered as the library's own (46.5% and 39.4% of their heads-up flop
  decisions, `--players`), and with antes or a straddle the nearest spot of another pot answers 24.4% and 38.0%
  approximately (`--ante`, `--straddle`). A
  flop's entry is worth its spot's share times its suit count, so more flops of the commonest spots beat new spots; by
  points per MB (`--assume`): BTN vs BB on every flop +12.4 for 4.5 MB, CO vs BB on 1656 (90%) +7.6 for 2.8, UTG vs
  BB on 552 +2.4 for 1.05, SB vs BB on 552 +3.5 for 2.1, then on every flop SB vs BB +8.6 for 6.0 MB (its wide
  ranges' entries are the biggest), HJ vs BB +7.8 for 3.5, UTG vs BB +6.1 for 3.1, the five flatted opens on 552
  +2.9 for 6.6, the eight 3bet pots on 552 (five of them new: the SB's and the BB's 3bet of the CO, the HJ's of
  the UTG, the CO's of the HJ, the BB's of the SB) +2.7 for 8.8, all built; next: the flatted opens on every flop
  +4.3 for ~14 MB.
* **What is stored**: the flop's first decisions (the root; the in-position player after a check or facing a lead;
  the out-of-position player facing a bet after checking) and the decisions facing a raise of those bets (a
  check-raise, a lead raised: entries built since 2026-10-29; 1-2 more nodes a flop, entries 26-45% bigger), only
  where the other player gets there with 5% or more of their range, and a raise only of a stored bet (a lead the big
  blind never makes is the solve's noise, and the advisor solves live behind a line under 10% anyway). Per combo: the
  average strategy in 1% steps and every action's EV (the best to 1% of the pot, the others as a gap below it, 1%
  steps then 4%), plus range totals.
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
  what's stored is what's downloaded, so the test's 150 MB cap is on both (the page fetches only the entry a spot
  needs, 1.5-6 KB).
* **Tree**: Accurate's flop model (one turn per rank class, the river bet on; the browser's Accurate config,
  which plays the same flop game as the native one), two flop sizes, 300-800 iterations, to at most 0.15% of the
  pot from equilibrium (mean 0.109%; paired and trips boards settle slowest, and 17 that the old cap of 500
  iterations left at 0.15-0.54% were solved again) (the live Accurate solve gets ~0.3-0.4% in the server's 30 s and 60-75 iterations in the
  browser's 60 s). Every spot deals the river instead of betting it on the turn's boards (`library_config(pot,
  stack)`; the 3bet pots since 5c3fba00, then the single-raised pots too): every river card under each turn,
  played on each card by 64 buckets of strength there
  (`Config.river_dealt`: each player's range split into 64 equal shares by equity on that card against the
  other's). Played blind, the river cost their flop strategies 0.40% of the pot out of position and 0.29% in
  position against every card dealt (the 3-bettor c-bet three quarters pot 0-11% of the time where the full game
  bets it 16-55%); dealt, 0.10 and 0.15 at 400 iterations (Accuracy vs speed, against postflop-solver). The shipped
  entries (200-300 iterations, at most 0.15% of the pot from equilibrium) give up 0.11 out of position and 0.27 in
  position on the same four flops, their root mixes 4-17 points per combo from the full game's where the blind
  river's were 16-55 (in-position losses move by up to 2x between measurements of one strategy: Qc5h5d's 0.21 and
  0.42). In a single-raised pot the blind river cost 0.29 and 0.17 (BTN vs BB, JcTd8d), dealt 0.04 and 0.07; on
  Kc7d2h 0.40 and 0.48 blind, 0.18 and 0.25 dealt (their game there to 0.14% of the pot); an entry takes 6-7x as long (93-179 s on one core at 300 iterations, 14-28 blind; the same size, 4-9 KB).
  The entries solved blind before stay in the index until each is solved again: a spot lists them under its
  `old_tree` (`old_tree()`) and `old`, and the advisor reads each entry by its own tree. The answer's `precomputed`
  field (a badge on the page) says which spot and how many iterations;
  `same_tree` is false where the live solve's flop model differs: for an entry that deals the river for Accurate too
  (its live flop solve bets the river on the turn's boards), and in Fast mode, whose own flop model plays the
  turn and river by hand buckets: converged (300 iterations) on 20 of the library's BTN vs BB flops its c-bet is 2.8 points from the library's on average (up
  to 7, betting more), BB's response to the small c-bet 2.9; at its 2.5 s and 8 s budgets it's further. (The old
  Fast model, the river checked down on random turns, was 5.6 and 8.6 off, up to 15 on dry boards.) Natively the
  answer takes ~0.01 s instead of 2.5 / 30 s; in the browser ~0.03 s instead of 8 / 60 s, and a turn spot saves
  the earlier flop solve too (BTN vs BB, J75 c-bet called: 8.2 s for the turn in Fast mode, 11.5 s before; measured on the old Fast model).
* **Menu** (`scripts/bench_menu_loss.py`: a richer flop menu solved, then each player's flop decisions locked to
  the library's; % of the pot given up out of / in position, BTN vs BB on Kc7d2h, JcTd8d, KcTd6h and 8c6d4c, 1000
  iterations, the river blind): a third flop bet (33/75/125%) 0.02 at most; a second raise size of 30% 0.16 / 0.20
  on Kc7d2h and 0.02 at most on the other three, of 100% 0.10 / 0.02 on KcTd6h and ~0 elsewhere. Each costs 32-43%
  more nodes, up to 66% more time and a rebuild of every entry, so the library keeps bets of 33/75% and a 50% raise.
* **Turns** (`<spot>/<flop>.turn.bin` beside each entry, `encode_turns`, GTOTN1; written by the builds from here,
  none shipped yet): the flop solve's own turn strategies, for an answer at once on the turn after a library flop
  (`FlopLibrary.turns`, `turn_node`; the advisor's approximate answer there, `instant_from_store`: Accuracy vs
  speed, "The turn after a library flop"). Per flop line that 1% or more of the hand pairs take to the turn (BTN vs
  BB on Kc7d2h: five lines, 66% of the pairs), the turn's first decision and the in-position player's after a check;
  per turn class (one card a rank, a flush suit's apart: 13 on a rainbow flop, 26 two-tone, 23 monotone) the played
  card's strategies for the acting player's combos that reach the line (0.5% of its most reached): strategies only, in 5% steps, at the flop solve's turn sizes (75% bets and raises).
  Another card of a class reads the played card's: each combo the same combo's, and one holding the played card that
  of its image holding the turn card. ~10 KB a rainbow flop, ~18 KB on average (~360 MB for 20,000 entries,
  committed with the entries). Against a live Accurate turn (33/75/125%, 400 iterations; BTN vs BB on Kc7d2h, its
  two commonest lines; `scripts/bench_turn_store.py` on `scripts/turn_store_probe.py`'s dump) the stored turn gives
  up 0.50 / 0.67% of the pot out of / in position on the played cards and 0.81 / 0.63 on the others (1.6 at most),
  where the live solve is 0.08-0.18% from equilibrium; reading the suit-swapped combo instead gives up 0.81 / 0.93
  (2.0), and the 5% steps add 0.003 (0.02 at most). A richer turn menu in the flop solve (33/75/125%) was left out:
  2.4x the nodes' memory (a build worker ~4.5-5 GB) and CPU. `--no-turns` builds without them. **Format 2**
  (`--turn-evs`, `turn_evs`; header `"format": 2`, `"ev": 0.001`) adds the flop solve's EVs after every line's
  strategies, where a format-1 reader (which reads what the header lays out and stops) doesn't look, so it reads the
  file as before: per line and turn class both players' range EV at the turn's first decision (range against range,
  as a live turn report's `Advisor._row_evs`: the two add up to the pot), then per stored node and class the acting
  player's EV of each action for the hands that take it (none under 0.05 combos); int16 in 0.1% of the line's turn
  pot, or the first of 0.2, 0.5 and 1% that holds the file's largest EV (the header's `"ev"`: a deep stack's all-in,
  200bb behind a 5.5bb turn pot, is ~36 pots, past 0.1%'s 32.7; beyond 1%'s 328 pots it's clipped). The turn report then has EVs as a live one's (each card its class's) and each action's in its bar's tip; a
  format-1 file's report is as before (EVs "—", a note saying the file holds strategies only). The approximate turn
  answer (`instant_from_store`) shows them too, as the range's, never your hand's own: a Range EV column, the stored EV
  of the hands in your range that take each action (`"range_ev_bb"` per action, `null` where none does), and first to
  act both ranges' EVs (`"range_ev_bb"`, `"villain_range_ev_bb"`; at 200bb on Kc7d2h checked through, 8s turn: "range
  EV: yours +2.61 bb · theirs +2.89 bb of the 5.5 bb pot", check +2.20, bet 75% +3.89, where QsJs checks), scaled from
  the line's pot to the hand's. There's no `"ev_bb"`: your hand's EVs come with the turn's own solve. A format-1
  file's answer is as before, to the character. The page's early panel shows it, and Study's stored turn at once
  (each action's EV and the range's, Study's own columns: Study's turn after a library flop); Practice grades the
  turn's own solve. Against the library's own game solved live, river dealt, the stored
  EVs are 0.15-0.36% of the pot off on average (at most 0.73) for the ranges and 0.16-0.34 (at most 1.0) for the
  actions; a turn report read off a flop solved live (the river by strength buckets) differs by more (0.6-0.9% of the
  pot in single-raised pots, 1.6-4.5 in 3bet pots). On three 3bet-pot flops
  at 300 iterations it adds 8-12% (CO vs HJ 9c6d6c 8131 -> 9073 bytes, BB vs SB Kc7d2h 3861 -> 4225, BB vs CO Qc9c4d
  16585 -> 17854) and 0.1-0.5 s a flop to a 19-47 s solve; a hand-class block on top (each grid class's EV per action,
  1% of the pot) would add another 111-133%, so it isn't stored. The index lists a
  spot's turn files that name the entry beside them by its hash (`"turns"`: true for all its flops, or a list), and
  removes any other `.turn.bin` from the spot's folder, since the library's deploy publishes the folder as it is (a
  test fails on an entry or turn file the indexes don't name).
* **Lines**: the advisor walks the actual flop action through the stored nodes (sizes taken more than 15% off the
  tree's, more raises than it allows, or a line through a re-raise fall back to the live solve, as does a raise in an
  entry built before raises were stored). The spots' reports and drills keep to the first decisions. Turn and river
  spots narrow the flop's ranges through it as well, instead of the earlier street's short solve. The entry ignores
  the hero's cards in the villain's range (the live solve drops them): an edge the size of a blocker.
* **Flops not stored** are solved live from scratch; meanwhile the stored flops most like it answer approximately
  (Accuracy vs speed, A library spot's flop the library doesn't store). A warm start from the
  spot's most similar stored flop (same pairing and suit pattern, nearest ranks; strategies carried over by card
  substitution and seeded as regrets) was measured and left out (`scripts/bench_library_warm.py`, 12 unstored BTN vs
  BB flops, geometric mean % of the pot, cold -> seeded): it helps early (Accurate at 10 / 40 / 80 iterations: 81 ->
  53, 5.3 -> 3.9, 1.10 -> 0.98, the same with the browser's NO_BLAS kernels: 60 iterations 2.08 -> 1.60), is even by
  100 (0.68 / 0.68) and then worse, its seeded regrets outlasting DCFR's discount: 200 iterations 0.19 -> 0.24, 400
  0.07 -> 0.09, 10-12 of the 12 flops worse (the browser's bucketed Accurate the same, Fast at 120 0.50 -> 0.58).
  Every budget runs past 100 (Fast 140-420, Accurate 250-450 natively, 520-870 in the browser; in the compiled core
  the browser's Fast 229-986 on a loaded machine where its tables leave it the time, and Accurate its 1000 in 16-31
  s). So was seeding it from the approximate answer (the three stored flops most like it, above;
  `scripts/bench_flop_seed.py`: 104 stored flops of the flatted opens and the 3bet pots in Fast's tree and 78 in
  Accurate's, natively, one thread; geometric mean exploitability, % of the pot, cold -> its mixes seeded as regrets
  at the pot's size, `regret:1`): at 100 iterations 0.45 -> 0.41 (3bet pots, Fast), 0.76 -> 0.72 (single-raised),
  0.47 -> 0.37 and 0.73 -> 0.67 (Accurate); at 300 3-9% lower, at 600 even, and 14-23 of 30-64 flops worse than cold
  at 300. Fast stops at 300 once settled (0.4-0.8 s natively, 0.4-1.1 s in the browser's compiled core) and Accurate
  runs its budget, so it saves no time. Seeding the strategy sums instead changes nothing (the average restarts at
  the first iteration); three times the pot (`regret:3`) is about the same.
* **Invalidation**: `index.json` holds each spot's range fingerprints and tree (`tree_key` of its
  `library_config(pot, stack)`, which deals the river; the flops still on the blind river's tree listed apart, `old`
  under `old_tree`); an entry whose spot, tree or `MODEL_VERSION` (bump it when the flop game changes in a way the
  tree key can't see) no longer matches the code is ignored, and a rebuilt index leaves stale entries out (a test
  checks the shipped index against the charts).
* **Browser**: the worker fetches `flop_library/index.json` as it starts and a flop's entries (`<spot>/<flop>.bin`,
  cache-busted by its hash) once the page says the flop is dealt (Precomputed answers, instant in the browser);
  the folder is published to its own site, www.michael-han.com/gto-solver-library/ (`scripts/deploy_library.sh`, run
  by `scripts/deploy.sh`), and the page's `LIB_BASE` points there.

Building (resumable: entries on disk for the current charts and tree are kept, so re-run the same command after
a stop; progress lines go to stdout, the index is rewritten every 20 entries, and with it each spot's
`report.json` for the flop reports: per flop the stored nodes' range frequencies, the root's EVs, range equity over
every 8th runout (within ~0.3 points of exact) and a hand to load; only changed entries are recomputed, ~0.15 s
a flop, all thirteen spots from scratch ~5 min):

```bash
nohup nice python scripts/build_flop_library.py --jobs 5 > reports/flop_library_build.log &   # every spot, its SPOT_FLOPS
python scripts/build_flop_library.py --spots srp_btn_bb --flops all                     # all 1755 flops of a spot
python scripts/build_flop_library.py --spots srp_btn_bb --flops all --keep-old          # the flops not on disk first
python scripts/build_flop_library.py --spots srp_btn_bb --boards Ks7d2c,Qs8s3d          # given flops
python scripts/build_flop_library.py --index-only
python scripts/build_flop_library.py --recode        # format-1 entries to the current format, no solving (~5 min)
python scripts/build_flop_library.py --format 40bb --flops 184 --jobs 5                 # another game's spots
```

Each entry takes ~40-60 s in numpy on a rainbow flop and up to twice that two-tone, on an idle core, and ~2.3x
less on the compiled core (`gto/core.py`; several times that on a loaded machine; 3bet pots ~10-25 s): the whole
library (9327 entries) ~8 hours on 8 processes on the core (SB vs BB's last 282 flops, HJ's and UTG's other 1203
took 133 minutes; in numpy BTN vs BB's other 1203 flops and UTG's 368 took 4 hours on 5, CO vs BB's other 1203 and
SB vs BB's 368 3.5 hours on 7). Five of those 2688 (srp_sb_bb KdKcQc and KcQcTc, srp_utg_bb AcQdTh,
KcTd9d and AcKdQd) ran to 23-73% of the pot after 1000 iterations, a regret sum under float32's smallest normal
(fixed since) gone NaN, and were solved again with `GTO_CORE=0`: an entry the build leaves over `--target` at
`--max-iters` is worth a second look.
After changing the charts or the flop model, run it again (stale entries are re-solved). A spot on every flop is
4.6-6.8 MB (SB vs BB's wide ranges 9.2): the test holds the library's entries (48.31 MB) under 150 MB. The five flatted opens on 552 (2760 entries, their callers'
ranges narrow) took 165 minutes on 7 processes, the BTN's flatted by the SB the slowest (its opens' wide range).

### Precomputed 3-way flop library

A live 3-way flop solve takes Fast's 5-8 s to settle in its smaller game (Quick's 1.8 s in a smaller one still) and
Accurate's 13-25 s in its own; before the compiled multiway core Quick and Practice's quick answers left it to the
betting rules. The commonest 3-way pots' flops are solved offline (`gto/mw_library.py`, built by
`scripts/build_mw_library.py`), so the flop's first decision is answered at once in Accurate's game at every speed
(Quick and Practice's `"quick": true` too), in the browser as natively.

* **Spots** (`mw_library.SPOTS`; 6-max, 100bb, no ante or rake, 2.5bb opens), with how often the charts deal each
  line: BTN opens, SB and BB call (0.25% of hands); HJ opens, CO or BTN and BB call (0.22% each: the CO's cold call
  is the BTN's, so one game serves both); UTG opens, BTN and BB call (0.18%); CO opens, BTN and BB call (0.14%).
  `mw_index.json` lists the spots built (a spot in `SPOTS` not built yet just isn't answered from it). Of the flops
  seen 3-way (6-max 100bb, the charts' lines) these four are 63.8%; listed to build next: UTG opens, HJ or CO and BB
  call (10.2%: one game), CO opens, SB and BB call (8.3%), HJ opens, SB and BB call (4.3%), UTG opens, SB and BB call
  (3.9%), 90.5% with them. With the stored flops 12.8% of flops by weight the library answers ~8.2% of 3-way flops'
  first decisions; every flop (`--flops 1755`) of the four built spots would answer 63.8%.
  A spot is matched by the three ranges entering the flop (as the charts replay them), the pot and the stack; not
  with rake, custom sizes or ranges, exploit mode, locks or Optimize sizes.
* **Flops**: the heads-up library's 184 representative canonical flops (`flop_library.representative_flops`), plus
  K72 rainbow (`EXTRA_FLOPS`), matched up to suits; other flops are solved live.
* **What is stored**: one 3-player CFR solve from the flop's start on native Accurate's 3-way game
  (`advisor.MW_LATER`, `MW_FLOP_BETS`: bets of 33 and 75% pot, one raise and all-in, the turn and river bet on by
  16 hand buckets a player), run to under 0.1% of the pot mean best-response gain (300-500 iterations, ~3-4 min a
  flop on a loaded core), stores the flop's first decisions: first to act, checked to once, checked to twice. The
  solve takes every hand the flop doesn't block (`include`, ~7% more time), so a hand off its player's range (a
  cold call the charts 3bet or fold) gets its current strategy, as a live solve gives it. Per hand: the mix in 1%
  steps and each action's EV, in the heads-up library's entry format (`flop_library.pack`, one mask per player).
  An entry is 3.8-7.6 KB; a spot's 185 flops ~1.1 MB, the four spots' 740 flops 4.53 MB
  (built in numpy in ~2.5 h a spot on 8 loaded cores; with the compiled multiway core an entry takes 15-45 s of CPU).
* **Lines**: a line through a bet isn't stored; facing a bet the flop is solved live as before. Practice opponents after a library answer read their move off it (`_mw_actor_move`), and a turn after the
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
  request importing the 3-way solver (Accurate's live solve took its 60 s then, Quick's answer was the rules');
  `mw-turn` narrows the checked-through flop through it (followed 100%). The worker fetches `mw_index.json` as it
  starts and a 3-way flop's entry once the flop is dealt (0.01 s an answer then, the preflop equity tables built
  while the worker was idle).

```bash
nohup nice python scripts/build_mw_library.py --jobs 8 > reports/mw_library_build.log &   # every spot (resumable)
python scripts/build_mw_library.py --spots mw_co_btn_bb --boards Kh7d2c                  # given flops
python scripts/compare_mw_library.py --spot mw_co_btn_bb --boards Kh7d2c,9s8s5d          # against live Accurate
```

### Deploy to the website

```bash
scripts/deploy.sh
```

copies the page, the worker, `practice.js`, `multitable.js`, `reports.js`, `drills.js`, `breakdown.js`, `session.js`, `study.js`, `solver.zip` (the worker's Python code: every `gto/*.py` and treys, `scripts/build_bundle.py`), the `gto` sources and Study's stored preflop solves (`gto/static/pf/*.bin` as `pf/`) into the `gto-solver/` folder
of the michaellhan.github.io repo and pushes. The precomputed libraries (`gto/static/flop_library`: the heads-up and
3-way indexes, entries and reports) go first to their own repo, michaellhan/gto-solver-library, served beside the page
at /gto-solver-library/ (`scripts/deploy_library.sh`): one commit with no history, force-pushed on each publish and
skipped when the files are the same, so rebuilt entries don't pile up in a repo's history (copied into the site repo
they had taken its history to 1.08 GB), and the page's `LIB_BASE` is stamped to point there. It publishes only from a
checkout at or ahead of origin/main (an older one would take back what was deployed since). First it runs the checks and aborts
the deploy if either fails: `python -m unittest`, then the quick sanity sweep
(`scripts/sanity_sweep.py --quick`, about 3 minutes on 10 cores: a fixed, seeded set
of ~80 preflop nodes and ~110 postflop spots covering every setup, every heads-up
spot kind on flop, turn and river, 3-way pots and exploit mode, plus ~30 hand-picked
rare lines over the setups that the quick preflop trees leave out (`RARE_LINES`:
3bet and 4bet pots, cold 4bets, flat then 4bet, a squeeze called twice or 4bet, 4bet
and 5bet jams faced, limp-reraises, straddled 3bet pots), with their decision nodes,
the ranges they reach the flop with and flop and turn spots in their pots, and ~20 whole page
requests from corners the setups don't reach (`EXOTIC`: spots `scripts/coverage_fuzz.py` found or drew, 7-9 handed
and 5-6 way pots, all-ins for less, tournaments, locks, custom sizes, and the flop library's own spots, flagged
when the answer isn't one the page can play, came from the catch-all, or missed the library); postflop
flags are re-solved at a longer budget and it exits non-zero when any flag but the
info ones, `filled_range` and `unsettled`, reproduces). The full sweep (~40 min) is
not part of the deploy; `scripts/sanity_sweep.py --full-report` runs it with the rare
lines and writes `reports/sanity_YYYY-MM-DD.md` (nothing schedules it). It uses `.venv/bin/python` if present,
else `python3` (`PYTHON=...` overrides). `DRY_RUN=1 scripts/deploy.sh` runs the
checks and stops before cloning or pushing; `SKIP_CHECKS=1 scripts/deploy.sh`
skips them, for emergencies only. `BUILD_ONLY=<dir> scripts/deploy.sh` builds the folder as it would be published
(versions stamped, each view's URL written) into `<dir>/site/gto-solver`, and the library into
`<dir>/site/gto-solver-library`, without the checks, cloning or pushing, to
try the static site: `python3 -m http.server -d <dir>/site 8000`, then http://127.0.0.1:8000/gto-solver/practice/.

Tune with `Config` in `gto/solver.py` (`depth`, sizes, sample counts) or the
`--budget` / `--depth` flags. `depth=3` from the flop is the full tree; it is
much slower in pure Python. Turn and river traversals run in C (the compiled core), and so do flop trees whose turn
and river are played by hand buckets, and natively the library's (played per combo to the turn's equity leaves,
their products batched: `core.LEAF_ROWS_BLAS`).

### Known limits

* Preflop charts are written by hand, not solver output, and the off-chart adjustments are rules on
  top of them. They are matched to published solves where any exist: `scripts/preflop_reference.py`
  measures 26 spots (opens, BB and SB defence, cold calls, openers facing a 3bet; the raked ones played
  at that rake), 15 hand-level facts and 11 raked ones against them (tests/preflop_refs.py, average error
  0.96 points over the 26, 0.99 over the first 22 where it was 1.03, and 2.9 before that with 10 spots and 8 facts out). Squeezes, the
  response to a 4bet, and cold players facing a 3bet have one published figure each (reported, not enforced:
  the big blind's squeeze over BTN + SB 75.9/15.0/9.1 against Poker Academy's 71.2/16.1/12.7, its response to
  a 24bb 4bet 44.9/33.4/21.7 against GTO Gecko's 46.9/29.3/23.8); 5bets and limped pots have none and are
  judgment, and heads-up has one (GTO Wizard's 50bb with an ante; see Other games' charts). A derived equilibrium (`gto/pf_game.py`, realization measured
  with the postflop solver, or by the flop library's own EVs) fits the published spots worse (2.4-3.0 points, 10-14 spots out, 3.1-3.8 and 15-16 by the library; each structural cause tested in
  docs/preflop_research.md section 7) and was not used to replace them; it says the 3bettor folds too often to the charts' 4bets (it continues 44-92% where
  the charts continue 28-56%), but that is a best response to the charts' bluff-heavy 4bet ranges; the one
  published response found (the big blind against the button's 24bb 4bet) continues 53%, where the charts
  continue 54% with the same cards removed (docs/preflop_research.md sections 6 and 8). The charts' continue against a 4bet swings with its
  size more than a solve's would: the re-pricing adds calls by equity against the 4bet range, which a 3bet
  range has bunched just under the chart's bar (the big blind against the button's 4bet folds 64% at the
  default 27.6bb, 45% at 24bb, 5% at 20bb). `scripts/preflop_sweep.py` runs
  every standard line for 2-9 players and flags broken frequencies and any hand continuing less often than one it dominates (pairs, kickers, suited over offsuit; the strategy is also kept that way at run time). Replace them with real chart
  exports if you have them; the format is a range string.
* Other games' charts (20-100bb, antes at 20, 40 and 100bb, 7-9 handed tables, a straddle, heads-up) are the
  100bb charts moved by solves of those games, on the charts' heads-up lines only: lines with limps or callers,
  the big blind against a heads-up limp (24% raises against GTO Wizard's 34% at 50bb; the solve raises 43-51%
  because a limped flop is valued as a single-raised pot with the limper as the aggressor), and the 6-max small
  blind's limps (not played: the published 100bb solves raise or fold) keep the rules. Under 20bb the advice
  is push/fold; deeper than 100bb plays the 100bb charts re-priced by the rules' implied odds (the 150 and 200bb
  solves, whose flops have one depth factor per pot type, call 3bets with fewer small pairs than at 100bb); the
  ante is solved at 20, 40 and 100bb only (interpolated at 60, 100bb's move deeper) and not heads-up; a 9-handed table's first two seats both play the 8-handed UTG's (no 9-handed solve), a
  straddle is solved only 6-handed, and the 8-handed moves are from the solves before a third player could call
  an all-in (re-solving it takes about 2.9 GB). The depth scale is measured on three spots and 12 flops, and the
  solves (1,500 iterations, a different equilibrium for each indifferent hand) leave about a point of noise in
  each decision's moves; the shallow and heads-up references are approximate matches (other antes, open sizes).
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
  equilibria run in prize equity. After the flop heads-up, 3- and 4-way solves play prize equity; in a 3- or
  4-way pot who wins when a player loses is taken as even among the others (0.002-0.33% of the pot given up
  against the exact prize game on a 3-way river), and pots left to the betting rules (five or more players)
  stay in chips.
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
  followed 0-20%, so it is off. Each player plays its own stack as far as another can match it, with side pots; the
  4-way solves play everyone at the shortest stack behind of those who bet (a deeper player's all-in there is shown
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
* 4-way flops and turns are solved live in a smaller game than 3-way ones: one bet a street (a third of the pot on
  the flop, half after), no raise but yours, everyone at the shortest stack behind, this street's card removal pair
  by pair. No 4-way flop library is built: 4-way flops are 0.10% of the flops seen in 6-max 100bb (3-way 4.2%,
  heads-up 95.7%), so a 4-way flop is never instant, and in the browser a quick answer leaves it to the betting
  rules. Five or more players on the flop or turn stay with the rules (their rivers are solved): 5-way flops are
  0.0016% of the flops seen (about 1 in 60,000; 6-way three lines of the charts' preflop tree, none measurable),
  and the same game for five doesn't fit. Its tree is 5,805 decisions on a flop where 4-way's is 820, and 1,625 on
  a turn against 196. The tables over bucket quintuples outgrow 1 GB on a flop even in Fast's smallest game (6
  buckets, every fourth river). A turn in 6 buckets takes 0.61 s an iteration in numpy natively and 675 MB, where
  100 iterations inside Fast's browser budget need about 60 ms an iteration. The compiled core has no 5-player
  kernel (a 4-way turn there takes 10 ms an iteration; the 3-way core gained 12-13x over numpy natively).
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
* Multiway with an all-in player for less where the heuristic answers (5+ players, or four with two all-in, before
  the river), it pays each side pot to the best hand among the players in it, but the betting rules inside its
  trials don't play for the side pot (an opponent bets and calls as if the whole pot were at stake). The 3- and
  4-way solves play side pots exactly (How it works).
* Every spot the page can set up gets an answer, not always a solve. Preflop: the charts (push/fold under 20bb). After
  the flop heads-up: the flop library where it stores the spot (its 6-max 100bb preflop lines at the default sizes, a
  stored flop, checks and bets within 15% of its sizes, your hand in the stored range; not in exploit mode, against a
  lock on the opponent, with the size optimizer, a jam request or ICM), else a live solve; 3-way the 3-way library,
  else a live solve; 4-way flops and turns a live solve in a smaller game (the betting rules at Quick in the browser);
  rivers of up to six players (five in the browser at Quick and Fast), and any street where only you have chips
  behind, the N-player solve. The betting rules take the rest: five or more on the flop or turn, six or more on a
  browser river, a 4+ way pot where you face a bet bigger than the shortest stack behind (the solves play everyone at
  that stack: your call would be priced at its size), and a 3+ way spot where the smallest raise you can make is more
  than the solve lets you put in (three: what another player can match). Behind all of it a catch-all
  (`Advisor.advise`, `Advisor._rules`): an answer that fails anyway (an exception other than a cancel, no actions, a
  recommendation that isn't one of them, a number that isn't finite) is answered from the chart preflop, else the
  betting rules, else your equity against random hands priced against the pot, marked `fallback` with the reason and
  a line on the page; the request's own mistakes (an unknown seat, clashing cards, bad sizes) stay errors, and Study
  answers preflop from the chart and shows nothing after the flop that it couldn't solve. `scripts/coverage_fuzz.py`
  checks it: random hands over the page's whole input space (2-9 players, 5-300bb and each seat's own stack, antes, a
  straddle, rake, tournaments, exploit mode, locks, custom sizes and ranges, limps, min-raises, overbets, all-ins for
  less; moves by the advice or at random, entered as the page enters them), every decision asked of the browser
  worker's advisors and the server's at Quick and Fast, Study and practice opponents now and then. It flags an error,
  a missing or non-finite answer, a recommended or mixed-in move the page can't play, a premium or a set folded, a
  slow answer, an answer from the catch-all, and on the flop library's own lines a decision it stores that came from
  elsewhere. Two runs of 300 hands (the second with the odd setups 2.5x as often and a quarter of its hands on the
  library's lines): 6,628 decisions, answered by the charts 63%, live solves 24%, the betting rules 12% and the flop
  library 1.4%; the catch-all answered none. The library answered all 92 of the decisions it stores (median 0.009 s,
  max 0.15 s). Seven holes turned up, now fixed with regression tests (tests/test_coverage.py, tests/js/stacks.test.js),
  all bets the page couldn't take or prices off: a 1bb small blind's all-in after the ante sized 1.0 of its 0.85; an
  advised raise after an all-in for less under the smallest raise (on the page too); the 4+ way solves pricing a deeper
  player's call at the shortest stack (71.5 as 5) and offering raises under the smallest (to 77.5 where it's 144, a
  0.2bb bet); the rules' raise held to what the deepest opponent can match shown under the smallest (55.88 where it's
  90, now shown at 90, the same move) and a raise of the whole stack not named all-in (1.55, which the page took as
  1.5); and a straddle's option after an all-in for less than the straddle answered as facing a jam (call, with
  nothing to call). The deploy's sanity sweep asks 22 of these spots again (`EXOTIC`). 6 of the 6,628 took over 2.5x
  their estimate plus 4 s: Quick heads-up turns with wide ranges (a limped pot, a single-raised pot checked through
  the flop) at 11-14 s against 1.8, and a 3-way flop with a player all-in or a 4-way flop at Quick natively whose
  solve spent the whole budget building its tables (0 iterations) before the rules answered (6.5-8.8 s against 1.5);
  they answer within twice their estimate now (Quick keeps to its time and memory). The same cap on a heads-up
  solve's tree holds the browser's Fast with three custom raise sizes, which could need ~3 GB on a turn, under
  `MAX_CELLS` (not fuzzed: the fuzz keeps each process under 1 GB). Left: the catch-all is a heuristic.
* Practice opponents heads-up after the flop read their move off your own solve, whose ranges leave out
  your cards (as your advice does): an edge the size of a blocker. A line of yours the solve rarely
  takes, a hand of theirs it gives no weight, or a pot that became heads-up mid-street, gets a solve from their
  seat instead (a few seconds, more in the browser), heads-up with your advice's ranges, so the earlier streets
  are your advice's solves and the same edge holds.
* The bet-size abstraction is coarse. A real bet or raise more than 15% off every tree size is put
  into that street's tree (closer ones are played as the nearest size). The Fast turn bets 33/75/125% in
  single-raised pots (the river inside 33/125%; 33/75/125% with the core on 4+ threads, as Accurate) and on one
  thread 33/75% where the turn starts at most 4.5x the pot deep (a 3bet pot's 3.2; the river inside 75%;
  `Advisor.spr_sizes`). Raises stay one size (75%)
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
