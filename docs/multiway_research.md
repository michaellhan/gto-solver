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
  × ½, average strategy × (t/(t+1))²). Alternating converged about 10x faster than simultaneous updates per
  iteration (river, BB/CO/BTN: best-response gains 0.8% of the pot after 30 iterations against 5%, 0.1% after 100
  against 1%) at about the same cost, since each pass needs only the traverser's terminal values.
* **Tree.** One bet size (flop ⅓ pot, turn and river ½), one raise (to the bet plus 75% of the pot after calling),
  all-in where the stack behind is at most 1.5 pots, and at most a bet and one raise per street (then fold or call).
  The solve starts from the node in front of hero: earlier actions this street narrow the ranges by the betting
  rules, as before. Everyone plays the shortest stack behind.
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
* **Streets.** The river is solved per combo with the exact removal. The turn and flop are solved for their own
  street (depth 1), and the remaining cards are dealt out as equity leaves: every river from the turn, and every
  turn with one sampled river from the flop. No betting is modelled after the solved street, a known bias that
  favours draws and the out-of-position player; the heads-up solver's flop model avoids it at a much higher cost.
  The leaves use the pairwise shortcut, with two corrections:
  1. The fraction of pairs that are mutually compatible is taken exactly from the board above the runouts and
     reused on every runout. A runout card moves that fraction very little.
  2. The live players' shares are scaled on each board so they sum to the pot, which keeps the game constant-sum.
     Without this, pairwise removal over-paid everyone by 2% of the pot.

## The error accepted (pairwise removal)

River, CO opens, BTN and BB call, 300 iterations each. These are the pairwise solve's strategies played in the exact
game, against the exact solve:

| board | exact solve's best-response gains (% pot) | pairwise strategies in the exact game | node strategies differ (mean) |
|---|---|---|---|
| Kh7d2c9s4h | 0.27 / 0.28 / 0.30 | 0.37 / 0.41 / 0.30 | 1.9% |
| Ah8h3h5c2d | 0.23 / 0.27 / 0.25 | 0.37 / 0.49 / 0.40 | 2.3% |

So the pairwise shortcut costs about 0.1-0.2% of the pot in exploitability and moves frequencies by about 2 points.
Without the renormalisation, its EVs ran up to 1.3% of the pot off on single actions. With it, the players' values
match the exact solve to 0.01 bb (1.95/2.61/3.44 against 1.947/2.617/3.436 on Kh7d2c9s4h). That is the error
accepted on the turn's and flop's runouts. The river root is always exact.

## Validation

See `tests/test_multiway_solver.py` and `scripts/compare_multiway.py` (numbers in the README):

* **Constant-sum.** The three players' values add up to the pot, exactly on the river and within 0.2% on turn and
  flop leaves (sampled runouts).
* **Convergence.** Each player's best-response gain shrinks: on the river 4.7/3.5/4.1% of the pot after 10
  iterations, 0.8/0.8/0.5 after 30, 0.09/0.10/0.06 after 100 and 0.01/0.02/0.01 after 300.
* **Two players.** Played two-handed, the solver matches the heads-up solver on the same tree: root bet frequency
  16.6% against 16.5%, per-action EVs within 0.03-0.06 bb on average.
* **Qualitative facts.** First to act into two players, the weaker half of the range almost never bets. Checked
  to, the strongest 3% bets over 80%. The BB bets less of its range into two players than into one.
