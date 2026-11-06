# Solving 3-way pots: what's done in practice, and what this repo does

## How multiway poker is solved

* **No equilibrium guarantee past two players.** CFR provably converges to a Nash equilibrium only in two-player
  zero-sum games. Run in a three-player game it can cycle, and even an exact equilibrium guarantees nothing:
  there can be several equilibria with different values, and they aren't interchangeable. What CFR does
  guarantee is that it avoids iteratively strictly dominated actions (Gibson, *Regret Minimization in
  Non-Zero-Sum Games with Applications to Building Champion Multiplayer Computer Poker Agents*, 2013,
  [arXiv:1305.0034](https://arxiv.org/abs/1305.0034)). Its long-run average approaches the set of coarse
  correlated equilibria, which is a weaker solution concept than Nash.
* **In practice CFR works anyway.** Risk & Szafron's CFR agents were competitive in three-player limit hold'em
  (AAMAS 2010, [abstract](https://webdocs.cs.ualberta.ca/~duane/publications/abstracts/2010aamas.shtml)).
  Gibson's CFR agents won the three-player events of the 2012 ACPC
  ([thesis](https://www.cs.ualberta.ca/~games/poker/publications/gibson.phd.pdf)). Pluribus (Brown & Sandholm,
  *Science* 2019, [Facebook AI](https://ai.facebook.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker))
  beat professionals at six-max no-limit. It used a Monte Carlo Linear CFR blueprint computed by self-play over an
  abstracted game, then refined it in real time with depth-limited search, again without a Nash guarantee. The usual
  measure is each player's best-response gain against the others' strategies. Their sum (NashConv) is 0 only at a
  Nash equilibrium. It shrinks in practice, though nothing forces it to. Ganzfried (2020,
  [arXiv:2001.11165](https://arxiv.org/abs/2001.11165)) reports that fictitious play beats CFR on that measure in
  some multiplayer games, which is a reminder that the measure is empirical, not proved.
* **Commercial 3-way solvers.** GTO Wizard added 3-way postflop solving in 2025, with custom ranges, sizes, mixed
  stacks and rake. It says a third player can raise the cost "by a factor of 1,000 or more"
  ([PokerNews](https://www.pokernews.com/news/2025/08/gto-wizard-becomes-even-more-powerful-3-way-postflop-spots-49408.htm)).
  Later releases added ICM 3-way solves
  ([Pokerfuse](https://pokerfuse.com/latest-news/2026/8/gtowizard-launches-multiway-icm-postflop-solving/)) and a
  multiway preflop solver
  ([PokerNews](https://www.pokernews.com/news/2026/02/gto-wizard-launches-multiway-preflop-solver-50550.htm)). Its
  method isn't public. MonkerSolver solves multiway trees by bucketing hands (card abstraction), and PioSolver is
  heads-up postflop. The published finding that keeps coming up is that multiway play is much tighter. GTO
  Wizard's 3-way flop study (already used to fit this repo's betting rules) has the opener checking 34% when
  checked to, against 23% heads-up. Bets get smaller (a third of the pot 56%, bigger 9%, against 38/39 heads-up),
  and summaries describe bluffing less and value-betting thinner rarely
  ([PokerStrategy](https://www.pokerstrategy.com/news/content/How-to-play-top-pair-top-kicker-multiway_136686/)).
* **Card abstraction and card removal.** Full-width range-vs-range solvers keep one value per hand, and a showdown
  against one opponent is O(n) per board once hands are sorted by strength. The cards two hands share come out
  by inclusion-exclusion over the 52 cards (Johanson et al., *Accelerating Best Response Calculation in Large
  Extensive Games*, IJCAI 2011; this repo's `solver.Ranking`). Three ranges add a hard part: the two
  opponents' hands must also be disjoint from each other. Abstraction-based solvers (Monker, Pluribus's buckets)
  sidestep it, since buckets are dealt from a joint distribution. A common range-solver shortcut is "pairwise"
  removal: each opponent's hand is kept disjoint from yours but the two are treated as independent.

## What's built here (`gto/multiway_solver.py`)

* **Algorithm.** Vectorised full-width CFR over the three ranges, with regrets per (board, action, combo) and
  alternating updates (one traverser per pass). Discounting is DCFR's (positive regrets × t^1.5/(t^1.5+1), negative
  × ½ through the 99th iteration and × √t/(√t+1) from the 100th, the heads-up solver's beta step, average strategy ×
  (t/(t+1))³). Halving the negatives throughout swung late on 4 of 16 3-way turns (one from 0.08% of the pot at the
  200th iteration to 0.93 at the 300th); the step removed it (README: The 3-way solves' negative regrets). Alternating converged about 10x faster than simultaneous updates per
  iteration (river, BB/CO/BTN: best-response gains 0.8% of the pot after 30 iterations against 5%, 0.1% after 100
  against 1%) at about the same cost, since each pass needs only the traverser's terminal values.
* **Tree.** Bets of ⅓ and ¾ of the pot on the flop (`MWConfig.bet` takes a list; the advisor's flop config), ½ on
  the turn and river (also inside a flop solve), one raise (to the bet plus 75% of the pot after calling), all-in
  where the stack behind is at most 1.5 pots, and at most a bet and one raise per street (then fold or call). The
  solve starts from the node in front of hero; how earlier actions narrow the ranges is under "Narrowing" below.
  Everyone plays the shortest stack behind.
* **Exact card removal for three ranges.** For player p's hand h, the value at a terminal is a sum over the other
  two players' disjoint hands. For kernels A and B (all hands, weaker than h, tied with h):

      S = (A·x_q)(h) (B·x_r)(h) − Σ_c P_q(h,c) P_r(h,c) + Σ_{h'} A B x_q x_r(h')

  Here P_q(h,c) is q's reach on hands holding card c that are compatible with h. A pair overlapping in one card is
  counted once in the middle sum, and a pair holding the same combo is counted twice and added back once. Every
  term is a running sum over hands sorted by strength (per card for P), which is O(52·n) per terminal against
  O(n³) for the triple loop. The joint mass (the weight of compatible pairs, needed by folds and every
  payoff) expands into 52×52 card-pair matrices and costs O(52³ + n). Showdown shares with ties come from
  W = weaker + tied/2 and E = tied: W_q·W_r + E_q·E_r/12 gives 1, ½ and ⅓ for beating both, tying one and tying
  both. The tests check every terminal kind against the brute-force triple loop to 1e-3 relative (float32
  tables; 1e-11 in float64).
* **Streets.** The river is solved per combo with the exact removal. The turn and flop are solved per combo on
  their own street, and the streets after it are bet on too, by hand buckets (`MWConfig.later = "buckets"`, the
  heads-up solver's Fast flop model for three players):
  * The next card is one card per class (a rank's cards; a suit with two or more board cards gets its own card
    per rank), weighted by the cards it stands for. A combo holding the card played plays as its image with that
    card swapped for another of the class. From a flop the river is then played on the turn's boards without its
    card (its showdowns are the turn's equity over every river), as the heads-up `river_play`.
  * On each class's card every player's root range is split into K equal shares (16 natively, 12 in the browser)
    ordered by equity against the other two ranges. Below the root street the strategies, reaches and values are
    per bucket.
  * Showdowns are tensors over bucket triples: J, the weight of the compatible deals, and each live set's pot
    shares, built from per-combo kernels on each river (pairwise removal, with the third pair's overlap share
    taken exactly per combo from the board above, as the old leaves did), then scaled per cell so the live
    players' shares add up to J. The bucket game is constant-sum, and the deals' total weight is scaled to the
    root's exact mass (it moved 1.4%).
  * The later streets' subtrees after root lines of the same shape are merged into one batch (a row per line and
    card class), so a flop solve walks ~650 nodes a pass instead of ~11,000. A terminal's value is a K³
    contraction per row. Below another player's fold the folded player's reach is fixed, so the tensors are summed
    over it once and each terminal costs K² a row; below the traverser's own fold its value is minus its chips in
    times the deals' weight, with no walk.

  The old model dealt the later streets out as equity leaves: every river from the turn, every turn with one
  sampled river from the flop. With no betting after the solved street a flop bet was worth too little, so the
  flop was left to the betting rules. Measured against deeper references (`scripts/compare_multiway.py`):
  * **Turn** (`--turn-reference`): two turns, BB first and BTN checked to, 63 hands, against a reference that
    bets the river per combo (16 sampled rivers, 100 iterations, 0.1-0.3% of the pot from equilibrium):

    | | picks the reference's best action | EV given up | EVs off | range bets off |
    |---|---|---|---|---|
    | river bet by buckets (16) | 89% | 0.032 bb | 0.75 bb | 3.8 points |
    | river dealt out | 87% | 0.056 bb | 0.98 bb | 4.5 points |
    | betting rules | 75% | 0.239 bb | 2.16 bb | |

    Checked to on Jh9h4c8s the buckets bet 61% for the reference's 49% (dealt out 49%). Elsewhere they are
    within 2 points. The old reference (12 rivers, 60 iterations) hadn't settled: it had BB leading Kh7d2c9s 9.4%
    where 100 iterations give 2.6%.
  * **Flop** (`--flop-reference`): UTG opens, SB and BB call (the advisor's ranges, the blinds' checks narrowed
    by the rules), UTG checked to and the SB first, against the same model with 24 buckets, every turn card and
    every river (200 iterations, 0.1-0.55% of the pot from equilibrium), three flops, 88 hands:

    | | picks the reference's best action | EV given up | EVs off | range bets off |
    |---|---|---|---|---|
    | 16 buckets, one turn per class, 150 iterations | 90% | 0.003 bb | 0.34 bb | 1.6 points |
    | 12 buckets (the browser) | 83% | 0.005 bb | 0.44 bb | 2.5 points |
    | turn and river dealt out | 67% | 0.033 bb | 1.56 bb | 18.3 points |
    | betting rules | 65% | 0.061 bb | 3.41 bb | |
    | *flop bets of 33 and 75% (`--flop-sizes 33,75`; the reference too):* | | | | |
    | 16 buckets | 75% | 0.011 bb | 0.33 bb | 0.6 points |
    | 12 buckets | 70% | 0.011 bb | 0.45 bb | 2.4 points |
    | turn and river dealt out | 55% | 0.055 bb | 1.45 bb | 14.6 points |
    | betting rules | 56% | 0.076 bb | 3.49 bb | |

    With two sizes the best action is more often a near tie between them (the EV given up stays at 0.01 bb).

    Checked to, the reference bets 84.9 / 40.1 / 99.6% on Kh7d2c / 8h7sJs / QdAd3s, the 16 buckets 82.5 / 39.4 /
    99.1% and dealing out 50.2 / 33.2 / 68.2%. First to act the SB leads 30.4 / 80.4 / 0.4%, the buckets 28.1 /
    77.2 / 0.7%, dealing out 41.9 / 58.4 / 3.8%. This reference is the same model made finer, so it checks the
    buckets, the card classes and convergence, not the bucketing itself; the two-player check below does that.
  * **Flop against the published study** (`--flop-study`, 150 iterations, ten random flops, the opener checked to
    with the blinds' full ranges): GTO Wizard's study (three sizes) has the opener betting 66% into two and 77%
    heads-up. With the chart ranges the solve bets 40.9% and 61.5%, where dealing the later streets out bet 33.2%
    and 48.2%. With ranges shaped like the study's (`--study-ranges`: its SB calls with low and middle pairs, suited
    connectors and suited Ax/Kx; its BB 3bets TT-JJ and suited broadways when the SB is in) it bets 47.1% and
    60.5%, where dealing out bet 39.3% and 48.3%. Its own flops are 20 weighted ones and its ranges aren't
    published, so this is a band, not a match. Played two-handed (`--hu-check`, UTG against the BB from the BB's
    first action, seven flops) the bucket model bets 4.0 points more than the heads-up solver's per-combo flop model
    on the same tree (7.3 absolute; the BB leads 5.2 more), which itself bets 58% there. What was tested for the
    rest (same ten flops, 150 iterations; heads-up solver: UTG against the BB checked to, seven flops):

    | | 3-way, charts | heads-up, charts | 3-way, study ranges | heads-up, study ranges |
    |---|---|---|---|---|
    | flop 33% | 40.9% | 61.6% | 47.1% | 60.5% |
    | flop 33 / 75% | 39.8% (33: 32.8, 75: 7.0) | 53.7% | 46.6% (41.8, 4.8) | 53.7% |
    | flop 33 / 66 / 100% (the study's) | 39.1% (29.4, 7.0, 2.7) | 53.2% | 45.9% (39.0, 4.9, 2.0) | 52.9% |
    | GTO Wizard's study (33 / 66 / 100%) | 65.9% (56, 8, 1) | 77.0% (38, 21, 18) | | |

    * More sizes don't raise the total: the bigger bets come out of the small one, and heads-up the total falls 8
      points. The heads-up per-combo solver does the same (60.4 / 57.1 / 52.6% on one size, 33/75% and Accurate's
      tree). The 3-way mix of sizes looks like the study's (mostly small, 7-8% big); the check share is what differs.
    * The river played without its card (here and in the heads-up flop model): with it dealt and played per combo
      (heads-up solver, depth 3, the one-size tree) the opener bets 63.6% against 60.4% (by flop -13 to +7 points).
    * Raises (`max_raises` 3, a re-raise allowed: K72 32 -> 34%), smaller raises (50%: 37%) and 32 buckets (28%)
      move one flop by a few points either way.
    * The ranges: the charts' SB flats 36 combos (66-TT, a few suited broadways) against the study's wide pairs and
      suited hands; the study-shaped ranges add 6 points 3-way.

    So the tree is not what is missing; the river's card is worth about 3 points, the ranges about 6, and the rest
    is the study's ranges and flops (not published).
  * **Which node.** The study's tree lets the blinds lead (its SB leads 9.8%), and its 66% is the opener's play after
    they checked. With the blinds' full ranges our number is that of blinds who never lead. Solved from the SB's
    action (`--full-street`, five of the ten flops, 100 iterations) the opener bets 49.0% at that node where the
    full ranges give 43.0% on the same flops; our blinds lead far more than the study's (they check through 21-98%,
    65% on average). The advice plays the blinds' checks narrowed by the rules (above): 51.5% (72.5% before).
  * **The raise size.** The study raises 33% of the pot; ours raises to the bet plus 75% of the pot after calling.
    With 33% (`--raise-frac 33`, ten flops, 150 iterations) heads-up bets 64.0% (was 53.7%; 51.4 / 78.7 / 60.7 /
    41.7 / 68.5 / 58.5 / 57.9 / 64.5 / 86.1 / 72.0%) and 3-way 40.3% (39.8%): a big raise punishes the small c-bet
    against one player, not two. The default stays (3-way is what the solver plays).
  * Rake (5%, 3bb cap: K72 32.1 -> 32.2%, Q75 29.2 -> 28.7%), turn and river bets of 75% inside the flop (K72 35.7%),
    raises to 50% (32.0%), a re-raise (31.5%), and 75% turn bets with 50% raises (37.6%) move a flop by 0-5 points
    (100 iterations, full ranges).

  The old equity leaves (still `later="deal"`, and the all-in runouts there) use the pairwise shortcut, with two
  corrections:
  1. The fraction of pairs that are mutually compatible is taken exactly from the board above the runouts and
     reused on every runout. A runout card moves that fraction very little.
  2. The live players' shares are scaled on each board so they sum to the pot, which keeps the game constant-sum.
     Without this, pairwise removal over-paid everyone by 2% of the pot.

* **Narrowing.** A 3-way street before the one advised on (and this street's actions before the solve's root) used to
  narrow the ranges by the betting rules alone. Now, where your advice on that street came from a 3-way solve
  (`Advisor._mw_lines`: the solve indexed by its place in the hand), each action after its root multiplies its
  player's range by the solve's average strategy for it, smoothed with 15% uniform as heads-up, and the result counts
  by how settled the solve is (`trust_of`, the flop's 0.5-5% of the pot band); the rest stays the rules'.
  `--narrowing` (UTG opens, SB and BB call; the blinds check, UTG bets a third and both call; four flops, a turn)
  measures the turn's ranges against the walk through a 400-iteration solve, followed fully (total variation
  between the normalized ranges, mean over the three players):

  | | mean distance | the BB's call (combos; reference 29-48) |
  |---|---|---|
  | betting rules (before) | 0.52 | 141-167 |
  | betting rules, an overcall at OVERCALL (0.25) of the first caller's share | 0.44 | 51-52 |
  | through a 60-iteration solve (1.4-3.4% of the pot, followed 35-80%) | 0.43 | 62-116 |
  | through a 150-iteration solve (0.3-0.9%, followed 91-100%) | 0.22 | 34-51 |

  The rules let the BB call a third-pot bet with nearly its whole check range after the SB called; the solve has it
  fold 70-80% (it overcalls 3.5-20% of its range, 13% on average over eight flops, where the SB's first call keeps
  53-78%). An overcall now keeps a quarter of the first caller's share (advisor.OVERCALL; fitted on these four flops,
  0.20-0.25 best; on four others, 7dQs5h 9s9d4c Ah6c2d JdTc8h, 0.52 -> 0.46). The BB's distance stays near 0.5: the
  solve raises its sets and top pairs and calls with the middle (KTo, JJ, TT, A7s), which a strength cut can't draw.
  UTG's third-pot bet is the other half: in the two-size tree the solve splits its betting range between the sizes
  (1-148 combos bet a third), which the rules don't model.
* **The blinds' checks.** In front of the opener's solve the blinds' flop checks are narrowed by the rules (with no
  solve of their actions). Against a solve of the street from the SB's action (`--full-street`, ten flops, 100
  iterations; its check ranges saved) they were 0.21 (SB) and 0.14 (BB) off: the rules kept 68-80% of the BB's range
  where the solve checks 73-91%. A flop check now keeps CHECK_SOFT = 0.7 (was NARROW_SOFT, 0.3) of the hands the rules
  have betting: 0.15 and 0.05 (0.5: 0.17 / 0.09, 1.0: 0.14 / 0.03). The opener's solve in front of them (`--flop-study
  --rules-checks`, five flops, 100 iterations) bet 72.5% (36.7 / 99.2 / 76.8 / 54.6 / 95.1%) where the full street's
  solve bets 49.0% at that node (30.6 / 81.0 / 41.6 / 30.0 / 61.9%); now 51.5% (21.3 / 91.8 / 39.1 / 32.9 / 72.7%),
  7 points off per flop where it was 23. The heads-up walk's fallback to the rules for a line the solve takes with under 10% of the range
  (RARE_LINE) made it worse here (0.46 at 60 iterations: the BB's call is such a line early on), so the 3-way walk
  relies on the smoothing alone. With no such solve (a hand entered on the turn) the rules narrow the street: a
  solve from its start (`Advisor.mw_prior_solve`) got 40-65 iterations in native Accurate's 12 s earlier-street
  budget, 4-6% of the pot from equilibrium, which the walk would follow 0-20%, so it is off.
* **Budget.** A solve's time budget covers its tree and bucket tables (`MultiwaySolver.build_seconds`) and the
  best-response gains and EVs read off after it (`MW_RESERVE`, 2.5 iterations' time; an iteration doesn't start when
  it would end past the rest). A 15 s 3-way flop answer took 16.4-16.9 s and takes 14.8-14.95 (the browser's
  Accurate ran ~10 s past 60 s). The tables' per-card sums are a matmul instead of an einsum: 16 buckets 0.9-1.7 s
  where they took 1.7-3.0, 12 buckets over every third river 0.3-0.5 s (0.5-0.9). A solve its budget ran out on is
  not run again by a later request for the spot, so the answer repeats.

## The error accepted (pairwise removal)

River, CO opens, BTN and BB call, 150 iterations each. These are the pairwise solve's strategies played in the exact
game, against the exact solve:

| board | exact solve's best-response gains (% pot) | pairwise strategies in the exact game | node mixes differ (mean) | values, exact / pairwise (bb) |
|---|---|---|---|---|
| Kh7d2c9s4h | 0.03 / 0.05 / 0.03 | 0.11 / 0.14 / 0.08 | 1.1 points | 1.949 2.611 3.440 / 1.959 2.607 3.433 |
| Ah8h3h5c2d | 0.04 / 0.04 / 0.04 | 0.12 / 0.11 / 0.21 | 2.6 points | 2.392 2.380 3.229 / 2.384 2.387 3.229 |
| 8s8d3c5h5d | 0.07 / 0.07 / 0.10 | 0.07 / 0.08 / 0.13 | 1.7 points | 2.190 2.335 3.475 / 2.193 2.343 3.464 |

Pairwise removal costs about 0.05-0.15% of the pot in exploitability and 0.3-0.5% of the pot in single action
EVs, and its values are within 0.01 bb. Before the per-board scaling it over-paid everyone: the three values
summed to 8.15 in an 8 bb pot. Reusing the board-above overlap share on the runouts kept turn and flop values
within 0.2% of the pot of summing to it. This is the error accepted on the turn's runouts. The river root is
always exact.

## Validation

See `tests/test_multiway_solver.py` and `scripts/compare_multiway.py` (numbers in the README):

* **Constant-sum.** The three players' values add up to the pot, exactly on the river and within 0.2% on turn and
  flop leaves (sampled runouts). With the later streets by buckets every live set's shares add up to the deals'
  weight in every bucket triple (any reaches pay out exactly the pot); the values at the root's reaches add up to
  the pot within 0.6% (the deals' weight is scaled at the root's reaches, and the bucket and exact masses part a
  little as the ranges narrow).
* **Convergence.** Each player's best-response gain shrinks: on the river 4.7/3.5/4.1% of the pot after 10
  iterations, 0.8/0.8/0.5 after 30, 0.09/0.10/0.06 after 100 and 0.01/0.02/0.01 after 300. A flop with the later
  streets by buckets (Kh7d2c, UTG checked to, 16 buckets; gains in the bucket game) 14.9/9.1/11.8 after 25,
  4.6/2.4/2.4 after 50, 1.2/0.6/0.7 after 100, 0.3/0.1/0.2 after 200 and 0.1/0.03/0.05 after 400, its bets
  44 / 36 / 32 / 32 / 33%.
* **Two players.** Played two-handed, the solver matches the heads-up solver on the same tree: root bet frequency
  16.6% against 16.5%, per-action EVs within 0.03-0.06 bb on average.
* **Qualitative facts.** First to act into two players, the weaker half of the range almost never bets. Checked
  to, the strongest 3% bets over 80%. On four river boards (100 iterations):
  * The BB leads 4-14% of its range into two players, against 13-21% into one.
  * Last to act, a player bets 14-32% against 23-44% heads-up. Its bluffs (the weakest 40% of the range) are
    about a quarter of its bets either way, so it bluffs less only because it bets less. The top 10% always
    bets.
  * Facing the CO's half-pot bet, the button folds 68-73% and the big blind 49-53% after it. Both fold 34-37%
    of the time, against MDF's 33%: the defence is shared between the two.
* **Against the betting rules.** `compare_multiway.py --spots` covers 580 hand-spots on 4 rivers and 2 turns.
  The rules pick the solve's most frequent action 74% of the time, and their pick gives up 0.15 bb (1-3% of the
  pot) by the solve's EVs. They disagree most when first to act (river: the rules bet 1% of the strong-leaning
  sample, the solve 25%; turn: 26% against 3%) and when checked to (54-55% against 79-83%). Facing a bet and a
  call they agree 84-95% of the time.

## Thin lines: how the floor should fill them

Rule narrowing can leave a line with very few combos. Under `MIN_LINE_COMBOS` (8) `_post_floor` adds lam x the street's
range, lam = 1 - combos / 8, so the fill grows with that range, not with the shortfall: a line of 5 combos from a range
of 139 becomes about 60. It is shared by heads-up and multiway pots. Filling to exactly 8 in the line's own shape looks
like the obvious repair. `scripts/bench_thin_lines.py` measures that, and it doesn't win.

**The reference.** A street-start solve can't be one: the lines the floor fills are the ones it hardly reaches (a 75%
turn bet from CO, 0.00 combos, whose shape changes with the iteration count and buckets). The benchmark solves the
game with the line forced instead: player p takes it with at least T combos of the street's range and everyone
best-responds, by fictitious play (6 steps of 100 iterations, each solving the line and the node without it). Nothing
knows how rare the line really is, so T runs at 8, at 5 and at the line's own combos.

Two earlier forms misled, each to a different winner. A best reply of exactly T combos took medium hands when more than
T preferred the line, and scale looked clearly better. One that counted combos outside the range (a bucket's EV reaches
them) took losing hands past T, and the current filler looked clearly better. The final form agrees with itself: from
the narrow and the wide seed, at 200 iterations and 10 steps on 20 lines, the median difference is 0.0 combos
(correlation 0.99).

**The measure.** The next decision by someone else: their mix from a solve with the candidate range, valued by the
reference solve's action EVs, as % of the pot. 227 lines from a census of 41,502, 12 spots, 73 spot-flops; heads-up
flop and turn, 3-way flop, turn and river; noise ~0.02.

| Mean % of the pot given up | T = 8 | T = 5 | T = line's own |
|---|---|---|---|
| Raw line | 0.72 | 0.66 | 0.68 |
| Current filler | 0.37 | 0.60 | 0.68 |
| Scaled in shape to 8 | 0.32 | 0.58 | 0.73 |
| Uniform to 8 | 0.35 | 0.62 | 0.76 |
| Whole street's range | 0.51 | 0.82 | 0.93 |
| Scaled less current (95%, by spot-flop) | -0.05 [-0.15, +0.03] | -0.03 [-0.14, +0.07] | +0.05 [-0.04, +0.14] |

The median difference is 0.00 at every T. By street, scaled less current at the three T: 3-way flop -0.07 / +0.02 /
+0.17, turn -0.01 / +0.06 / +0.06, river -0.04 / -0.07 / -0.02; heads-up turn -0.31 / -0.35 / -0.17 (intervals reach
zero), heads-up flop +0.50 / +0.49 / +0.47.

The tie hides large lines on both sides. Scaled saves 7.0% of the pot where the current filler holds 28 combos and the
reference 8 (Jh9h4c2c3d, a 3bet pot's check-call) and 4.6 at 36 against 12. It loses 3.0, 1.7 and 1.0 on called flop
raises in 3bet pots that the reference plays with 28-45 combos, which 8 can't hold. With the street's range under 40
combos T = 8 binds on 90% of lines; from 40 up on 20% (the reference's median is 17 combos). Picking between them by
that split would be fitted to this sample, and the signal that separates them is the reference itself, ~15 s a line.

So the filler stays, as does the shape it doesn't keep. The rule's shape isn't the thing to preserve: the raw line is
0.48 from the reference by total variation against 0.32-0.35 for the fills, and a line that stays raw only ties at
T = the line's own, the case where the floor needn't have fired.

**The hand that asked.** BTN ThTs facing CO's 34 into 45.5 (README, Root-street all-ins). The reference at T = 8 is JJ
36%, AQo 19, KQs 12, 77 8: Kx and 77 about 20%, not the rule's ~50, and from either seed. In the benchmark's own solve
(12 buckets, ranges not conditioned on hero's cards) TT folds against the reference, call -0.76 at T = 8 and -3.56 at T = 5, and calling
is within 0.1 bb of folding on the current and the scaled ranges alike. This study first reported that through the Advisor
"the same two ranges" read +1.40 and -6.87, unexplained. They weren't the same ranges: the Advisor took hero's cards out
of the opponents' ranges before the rules and the floor read them, and solved those. One change at a time on that turn
(16 buckets, 400 iterations, TT's call and shove):

| ranges | BB, CO, BTN combos | call | shove |
|---|---|---|---|
| Advisor as it was: lines read after hero's cards | 8.1, 11.3, 94.5 | +1.48 | +1.45 |
| whole ranges (the benchmark's) | 9.3, 12.6, 94.5 | 0.00 (15%) | -14.66 |
| whole ranges, then hero's cards out | 7.4, 9.7, 94.5 | 0.00 (4%) | -4.51 |

Buckets and iterations weren't it (12 or 16, 100 to 400 iterations: the call moves under 0.2 bb). The first and third rows
differ only in reading the line after hero's cards: filler. The second and third differ only in the game: with hero's cards
out, the opponents' strategy answers a BTN whose every hand plays as if it held ThTs's blockers, and the shove reads 10 bb
high (README, Your cards stay out of everyone's range). The Advisor now solves the second row, so it and this benchmark
agree; scaled in the same solve TT folds at 100%, call -0.18, shove -16.48, and the call stays within 0.2 bb of folding on
either fill.
