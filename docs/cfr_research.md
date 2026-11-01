# Regret minimization for the postflop solver

What the literature says about solving two-player zero-sum imperfect-information games, what this solver
does, and what was measured here (`Config.cfr`, `gto/solver.py` `CFR_SCHEMES`; `scripts/bench_solver.py --algos`;
toy games with closed-form solutions in `tests/test_cfr.py`).

## The methods

All of these are counterfactual regret minimization (CFR, Zinkevich et al., NeurIPS 2007): every
information set runs a regret minimizer on its counterfactual values, and the *average* strategy converges
to a Nash equilibrium at O(1/sqrt(T)) in theory. They differ in the local regret minimizer, in how
iterations are weighted, and in how the players' updates are ordered.

| Method | Regrets | Average strategy | Source |
|---|---|---|---|
| CFR | sum of instantaneous regrets, regret matching | uniform | Zinkevich et al. 2007 |
| CFR+ | regret matching+: cumulative regrets floored at 0 every iteration; alternating updates | linear (t); quadratic (t^2) is better in practice | Tammelin 2014 (arXiv 1407.5042); Tammelin, Burch, Johanson & Bowling, IJCAI 2015; Burch, Moravcik & Schmid, JAIR 2019 (alternation, corrected proof) |
| Linear CFR | iteration t's regrets weighted by t | weighted by t | Brown & Sandholm, AAAI 2019 |
| DCFR(alpha, beta, gamma) | after iteration t, positive regrets x t^a/(t^a+1), negative x t^b/(t^b+1) | x (t/(t+1))^g, i.e. iteration s weighted ~s^g | Brown & Sandholm, AAAI 2019 (arXiv 1809.04040); recommended (1.5, 0, 2); CFR+ is (inf, -inf, 2) |
| Predictive CFR+ (PCFR+) | predictive regret matching+: play the regret-matching strategy of R + m, where the prediction m is the last iteration's instantaneous regret | quadratic | Farina, Kroer & Sandholm, AAAI 2021 (arXiv 2007.14358) |
| PDCFR+ | PCFR+ with DCFR's positive-regret discount: R_t = [R_{t-1} d(t-1) + r_t]^+, strategy from [R_t d(t) + r_t]^+, d(t) = t^a/(t^a+1) | (t/(t+1))^g; alpha 2.3, gamma 5 | Xu, Li, Liu, Fu, Fu, Xing & Cheng, IJCAI 2024 (arXiv 2404.13891) |
| DDCFR | DCFR whose (alpha, beta, gamma) are chosen each few iterations by a policy trained with RL on small games | | Xu, Li, Fu, Fu, Xing & Cheng, ICLR 2024 (spotlight) |
| HS-DCFR / HS-PCFR+ | fixed hyperparameter *schedules* over a run of n iterations: alpha 1 -> 4, beta -1 -> -3, gamma 30 -> 25 (HS(30); HS(15): 15 -> 10); for PCFR+ only gamma | | Zhang, McAleer & Sandholm, AAAI 2026 (arXiv 2404.09097) |
| Stable / Smooth PRM+ | PRM+ restarted when the regret vector gets small, or projected onto {R >= 0, sum R >= R0}: bounded regret growth | | Farina, Grand-Clement, Kroer, Lee & Luo, NeurIPS 2023 (arXiv 2305.14709) |

Reported results, and how they bear on poker:

* **DCFR** beat CFR+ on every poker benchmark of Brown & Sandholm (HUNL subgames, Leduc), by a factor of
  2-3 in iterations, mainly because the floor of CFR+ and the slow decay of early mistakes in plain CFR both
  lose to discounting. It is the algorithm most open poker solvers use: b-inary's postflop-solver (the
  reference used in this repo's README) runs DCFR with alpha 1.5, a 0.5 factor on negative regrets (beta 0),
  gamma 3, and restarts the average at every power of 4 (its `solver.rs`, `DiscountParams`), with alternating
  updates.
* **PCFR+** is the state of the art on many *non-poker* games (Goofspiel, Liar's dice, Battleship: orders of
  magnitude), but in Farina et al.'s own experiments DCFR stayed better on poker (HUNL endgames, Leduc), and
  Xu et al. 2024 find the same for their HUNL subgame ("DCFR+ emerges as the fastest", PDCFR+ comparable to
  CFR+). The explanation offered is dominated actions: in poker many actions are terrible for many hands, their
  regrets grow very negative (CFR+) or, with the prediction, keep pulling the strategy back toward them.
* **HS schedules** report the largest gains (orders of magnitude over DDCFR / PCFR+ on 10 benchmarks), but
  the gains come at thousands of iterations and very low exploitability; for large poker games the authors
  recommend HS-DCFR(30) over HS-PCFR+. gamma = 30 averages essentially the last few percent of iterations.
* **DDCFR** needs a trained policy network and per-iteration features (extra cost); its schedules are what
  the HS paper reproduces without learning.
* **Alternating updates** (one player's regrets updated per pass, the second pass seeing the first player's
  new strategy) are about 2x better than simultaneous ones for CFR+ and DCFR (Burch et al. 2019; every paper
  above uses them in experiments, though the PCFR+ / PDCFR+ proofs assume simultaneous updates).
* **Pruning**: regret-based pruning (Brown & Sandholm, NeurIPS 2015) and best-response pruning (ICML 2017)
  skip actions whose regret is so negative they can't be played for many iterations. They pay off in
  sampled or scalar-per-hand traversals; in a range-vs-range vectorised solver every action is evaluated for
  all hands at once, and an action is only skippable when *no* hand plays it.
* **Warm starting** (Brown & Sandholm, AAAI 2016, strategy-based warm starting) carries regrets over from a
  related solve; this solver does it when the tree grows (`warm_start`), restarting the iteration count.
* **Commercial solvers** don't publish their algorithms. PioSolver, GTO+ and similar report exploitability
  in percent of the pot and are commonly run to 0.3-0.5% (study) or below; GTO Wizard is reported to
  precompute its library to about 0.2-0.3% "Nash distance" (third-party reviews; not verified). This solver
  reports the same quantity (`Solution.exploitability`), so the numbers compare directly.

## Correctness: toy games with derived solutions

`tests/test_cfr.py` plays three half-street river games through `solve()` on real cards and checks the
average strategies, the EVs and the exploitability against closed forms (derivations in the file):

* AKQ with card removal (AA > KK > QQ, disjoint cards): Y bets A, checks K, bluffs Q s/(1+s); X calls with
  K (1-s)/(1+s); Y's EVs P + cB/2, P/2, 0.
* Clairvoyance (nuts or air vs a bluff-catcher): every nut hand bets, bluffs s/(1+s) of the nut combos, the
  catcher calls 1/(1+s); X's value P x (air that checks) / (Y's hands).
* The [0, 1] game on 1081 river combos whose strengths are a random order: Y bets its top a = (1+s)/(s +
  2(1+s)^2) and bottom b = a s/(1+s), X calls with c = 2a of its hands.

The solver's exploitability is checked against best responses computed independently in the tests (the
six AKQ deals enumerated; dense win/lose and card-compatibility matrices for the [0, 1] game) for
strategies far from equilibrium, to 1e-4 / 1e-3 % of the pot. Every scheme converges on AKQ.

## Measurements

`scripts/bench_solver.py --algos <schemes> --spots ... --configs ... --iters N`: exploitability in % of the pot
at fixed iteration counts (deterministic: the same numbers on a loaded machine) and the per-iteration CPU cost
measured separately. BTN vs BB single-raised pots (Qs8d3c dry, 9h8h6c wet, Kh7h2h monotone, 8s8d3c paired), CO
vs BTN 3bet pot (Kd7c2h), the turn Ks7d2cTh and the river Ks7d2cTh5s. Fast is the browser's and server's quick
tree (the flop per combo, the turn and river by 32 hand buckets; its turn root has 8 fixed rivers, a tree whose
"exploitability" goes negative because the best response enumerates every river); Accurate plays the flop per
combo with the river bet on and one turn per rank class, and the turn with every river. Schedules (hs-*) were
given the run's length as their horizon, which a time-budgeted solve doesn't know.

**Fast, exploitability after 100 / 200 iterations (% of the pot)**

| scheme | srp-dry | srp-wet | srp-mono | srp-paired | 3bet-dry | river |
|---|---|---|---|---|---|---|
| dcfr (gamma 4, the old default) | 1.11 / 0.256 | 0.98 / 0.242 | 0.98 / 0.237 | 1.55 / 0.494 | 1.09 / 0.203 | 0.118 / 0.042 |
| **dcfr-ramp (gamma 3 -> 8, default)** | **0.89 / 0.220** | **0.83 / 0.197** | **0.83 / 0.194** | **1.35 / 0.438** | **0.75 / 0.151** | 0.113 / 0.040 |
| dcfr+ (negatives floored) | 1.21 / 0.256 | 1.13 / 0.279 | 1.06 / 0.246 | 2.14 / 0.539 | 1.38 / 0.248 | 0.113 / 0.039 |
| cfr+ (quadratic average) | 3.20 / 1.306 | 3.01 / 1.005 | 3.24 / 1.092 | 3.44 / 1.625 | 2.34 / 0.744 | 0.261 / 0.129 |
| pcfr+ | 5.10 / 1.866 | 4.72 / 1.695 | 5.57 / 2.086 | 5.73 / 2.213 | 4.15 / 2.168 | 0.257 / 0.135 |
| pdcfr+ | 3.32 / 1.097 | 3.32 / 0.978 | 3.35 / 1.010 | 4.32 / 1.308 | 3.52 / 1.374 | 0.245 / 0.100 |
| hs-dcfr (HS(30)) | 1.12 / 0.243 | 1.24 / 0.416 | 12.7 / 0.275 | 3.32 / 0.345 | 1.41 / 0.251 | 0.127 / 0.039 |
| hs-pcfr+ | 3.85 / 1.199 | 3.25 / 1.031 | 3.34 / 1.152 | 4.15 / 1.359 | 3.66 / 0.970 | 0.246 / 0.117 |

At 500 iterations the order holds (srp-dry: dcfr 0.054, dcfr-ramp 0.046, pdcfr+ 0.199, pcfr+ 0.491; dcfr+ spiked
to 1.15 on srp-wet). At 50 dcfr-ramp is 4-15% below dcfr (3bet-dry +1%), at 5-20 the same within noise.

**Accurate, exploitability after 50 / 100 / 200 iterations**

| scheme | turn (every river) | srp-dry | srp-mono | 3bet-dry |
|---|---|---|---|---|
| dcfr | 0.82 / 0.220 / 0.071 | 3.83 / 0.94 / 0.238 | 3.94 / 0.87 / 0.236 | 2.85 / 1.04 / 0.274 |
| **dcfr-ramp** | **0.78 / 0.217 / 0.068** | **3.59 / 0.84 / 0.223** | **3.55 / 0.78 / 0.222** | **2.78 / 0.95 / 0.215** |
| dcfr+ | 0.76 / 0.230 / 0.068 | 3.90 / 1.00 / 0.358 | 4.55 / 0.94 / 0.329 | 3.34 / 1.33 / 0.448 |
| pcfr+ | 2.75 / 0.955 / 0.321 | 11.9 / 4.39 / 1.644 | 11.8 / 4.77 / 1.656 | 6.72 / 3.60 / 1.875 |
| pdcfr+ | 1.63 / 0.522 / 0.213 | 10.4 / 2.90 / 0.928 | 10.4 / 3.03 / 0.908 | 6.47 / 3.23 / 1.320 |
| hs-dcfr | 1.10 / 0.204 / 0.064 | 4.36 / 0.77 / 0.225 | 5.14 / 0.83 / 0.318 | 4.78 / 1.74 / 0.299 |
| HS(15) DCFR (alpha 1 -> 4, beta -1 -> -3, gamma 15 -> 10) | 0.85 / 0.182 / 0.058 | 3.72 / 0.76 / 0.216 | 4.09 / 0.77 / 0.289 | 3.53 / 1.21 / 0.286 |

The server's default tree (`Config()`, two sizes everywhere): srp-dry at 50 / 100 iterations dcfr 4.58 / 1.09,
dcfr-ramp 4.15 / 0.96; the turn 2.48 / 0.69 and 2.37 / 0.67. NO_BLAS (the browser's kernels, natively), Fast at
100 / 200: srp-dry 1.12 / 0.268 -> 0.92 / 0.242, srp-wet 0.95 / 0.233 -> 0.81 / 0.194, srp-paired 1.53 / 0.455 ->
1.32 / 0.385, the river 0.117 / 0.044 -> 0.108 / 0.041 (the same per-iteration convergence as with BLAS: the
schemes only change scalars and, for the predictive ones, one array).

**Strategies, not just exploitability.** On six flops (K72, QQ6, A74, 765, 986, 883; Fast) the mean absolute error of
the flop's aggregate frequencies (BB's check, BTN's actions, BB's response to each bet) against a 1500-iteration
solve, in points: after 30 / 60 / 100 / 200 iterations dcfr 13.4 / 7.8 / 3.9 / 1.18, dcfr-ramp 13.1 / 7.2 / 3.3 / 0.98.

**Within DCFR** (Fast, 200 iterations, srp-dry / srp-paired / river): alpha 1 0.88 / 2.69 / 0.044, alpha 2 0.47 /
0.77 / 0.059 (alpha 1.5: 0.26 / 0.49 / 0.042); beta -1 (negatives x 1/(t+1)) 0.27 / 0.49 / 0.035, beta 0.5
0.29 / 0.49 / 0.052; gamma 2 0.50 / 0.81, 3 0.31 / 0.57, 4 0.26 / 0.49, 6 0.23 / 0.45, 8 0.22 / 0.44, 12 0.22 / 0.44;
postflop-solver's averaging (gamma 3, restarted at every power of 4) 0.23 / 0.45; DCFR with the PCFR+ prediction
0.76 / 1.07. A constant gamma of 8 is as good late but worse before 20 iterations (srp-dry at 10: 155 against 110),
which the ramp avoids (105). The average strategy beats the last iterate for every scheme (dcfr srp-dry at 200:
0.256 against 0.488; pcfr+ 1.87 against 4.04).

**Per-iteration cost** (CPU, min of 6 chunks of 10, interleaved): river 0.71 ms for dcfr, pcfr+ and dcfr+ alike;
Fast srp-dry 11.0 / 11.5 / 10.6 ms; the Accurate turn 73.6 / 75.5 / 72.4 ms. The leaf products dominate; a
predictive scheme adds one regret-sized array (memory) and one add per strategy; flooring negatives is one cheap
pass instead of a select.

**Late swings, and the beta step.** Benchmarked against postflop-solver on the Advisor's own turns
(`scripts/bench_postflop_solver.py`), two of 15 Accurate turns (BTN vs BB after check, bet, call: QcTc2c-7s and
Tc6d2c-Ks) answered 1.0 and 1.5% of the pot from equilibrium where the others were at 0.2-0.6%. Exploitability every
25 iterations with beta 0 throughout (`dcfr-ramp-b0`): QcTc2c7s 0.626 / 0.444 / 0.330 at 100-150, then 2.845 at
175, 1.487, 0.769, ... 0.124 at 400; Tc6d2cKs 0.157 at 275, then 1.975 at 300, ... 0.344 at 400. The mechanism: a
line one player has all but stopped taking (IP facing OOP's unused bet) gets almost no regret, halving the
negatives every iteration wipes what it has, its hands drift back toward their early strategy, the bet into it
turns profitable again and the iterates swing; the average, weighted to the last ~30 iterations by gamma 8, takes
the swing in. Averaging changes only moved it (gamma capped at 5: 0.157 -> 1.730 at 300; postflop-solver's gamma 3
restarted at powers of 4: 0.156 -> 2.295; postflop-solver itself, which also halves the negatives, has bumps of
0.3 -> 1.0 on Tc6d2cKs that recover), and keeping the regrets of rows the opponent didn't reach out of the discount
didn't stop it (0.190 -> 2.975 at 225). Beta 0.5 (negatives x sqrt(t) / (sqrt(t) + 1)) from the 60th, 100th or
130th iteration, or ramped from 0 to 0.5 over 100, removed it: with the step at 100, 0.076 and 0.063 at 400 and
never above the curve. Its cost on the standard spots, Fast / Accurate at 200 and 300 iterations (beta 0 in
brackets): the river 0.039 / 0.022 (0.040 / 0.021) and 0.067 / 0.044 (0.063 / 0.040); srp-wet 0.201 / 0.104 (0.191
/ 0.096) and 0.182 / 0.091 (0.166 / 0.083); the turn 0.146 / 0.068 (0.133 / 0.060) and 0.250 / 0.127 (0.342 / 0.134),
identical through 100 (the tables above are beta 0 throughout). Beta 0.5 from the start is 1.5-2x worse at 50-100 iterations (the turn 4.56 / 1.14 at 50 /
100 against 2.58 / 0.78).

**The same in the multiway solves.** `gto/multiway_solver.py` and its compiled core halved the negatives every
iteration too, and swung the same way: 16 3-way turns (CO opens, BTN and BB call; 10 turn boards, the flop checked
through or a 2.6bb bet called twice; the Advisor's own Fast solves, captured and solved cold in the core; the mean
best-response gain, % of the pot, every 100 iterations) had four that went back up after settling: Ts6c2d8h checked
through 0.084 at the 200th iteration, 0.931 at the 300th, 0.317 at the 400th, 0.148 at the 500th; 9h8h3c2s after the
bet 0.386 at the 100th, 0.569 at the 200th; 6s5s4dQc and Jc7c2hTd stalled at the 200th (0.265 -> 0.219, 0.338 ->
0.299) where the others were at 0.08-0.14. With the heads-up step (beta 0.5 from the 100th, `MWConfig.beta`,
`beta_from`) those four read 0.033, 0.084, 0.055 and 0.081 at the 200th/300th and fall steadily; the other twelve
are within 0.003 at every check (the same through the 100th). Mean over the 16 at the 300th 0.118 -> 0.050 (worst
0.931 -> 0.072), at the 400th 0.054 -> 0.031 (0.317 -> 0.045), at the 700th 0.016 -> 0.013 (0.049 -> 0.020).
Flops: two 3-way flops solved live (Kh7d2c, Jd9c3h, first to act) the same within 0.003 through the 700th; two 4-way
flops (HJ opens, three call) the same EV given up at the turn's two nodes, but a slower tail in the gain (Kh7d2c
0.011 -> 0.020 at the 400th, 0.005 -> 0.012 at the 700th; both stop at the 200th, 0.035 -> 0.040); two 4-way turns
within 0.002 (they stop at the 100th).

## EV gaps on live turns

How right are the per-combo EVs the page shows (`ev_bb`, `Solver.node_action_evs`) on a turn solved live? For 34
turns (18 single-raised, BTN or CO vs BB after a library flop's common line; 16 BB vs BTN 3bet pots; random turn
cards, picked as `scripts/bench_postflop_solver.py` picks them) the Advisor answered the turn at Fast and Accurate,
and postflop-solver then played the same tree with every turn decision of both players locked to our average strategy
and every river solved to 0.02% of the pot: each combo's exact EV of each action for the strategy we play. Ours
against theirs, per combo and action, in % of the turn's pot, at the turn's first decision (OOP), IP after a check
and IP facing OOP's commonest bet. (A variant of the benchmark and of its harness that dumps EVs below the root; not
checked in.) The figure is the median over nodes of the mean |gap| over the actions each combo plays (5% or more of
its mix), weighted by reach.

**It follows exploitability.** Gap ≈ 0.04 + 0.4 x exploitability (% of the pot). Single-raised turns: 0.358 at 100
iterations (Fast's turn; nodes 0.27-0.63, the 90th percentile pair 0.97), 0.125 at 300, 0.088 at Accurate's 400,
0.056 at 1000. 3bet pots (SPR 1-3, so the Fast answer reaches 114-690 iterations, median 322): Fast 0.090, Accurate
(925-1000) 0.071. At equal iterations the two are alike (300: 0.125 and 0.10); deep single-raised turns (SPR 7-18)
only converge slower.

**Where it sits.** By node (Fast / Accurate, single-raised): IP facing a bet 0.57 / 0.21, IP after a check 0.44 /
0.12, the root 0.31 / 0.07. By hand (Fast, single-raised, mean 0.26): strong made hands are higher and biased up
(full house 1.06, signed +0.97; flush 0.70, straight 0.52, trips 0.41, two pair 0.36), draws 0.29-0.33, air
0.22. Combos the line rarely reaches are only ~20% worse than the rest.

**The floor at 1000 iterations** (0.05-0.12 by node) is the river's hand classes (the live turn plays rivers by
class): solving every river instead takes single-raised 0.039 / 0.061 / 0.120 (root / after check / facing bet) to
0.032 / 0.042 / 0.107, 3bet pots 0.052 / 0.073 / 0.111 to 0.028 / 0.062 / 0.050; and the reference's own noise, 0.00
to 0.04 (its river target 0.02 -> 0.005 moves the 3bet facing-bet node 0.111 -> 0.068, single-raised 0.120 -> 0.114).

**Rare branches are the weak spot.** By how often the actor's range takes the action at the node (mean |gap| over the
range, reach-weighted, played or not; Fast 100 / 300 / Accurate 400 / 1000 iterations, single-raised): 0.01-1% of the
range 6.0 / 2.4 / 1.6 / 0.6 (ours lower: signed -4.5 / -1.5 / -0.8 / -0.3; 3bet pots 0.2-0.6); 1-5% 1.4 / 0.6 / 0.6 /
0.15; 5% or more 0.50 / 0.16 / 0.11 / 0.06. These are mostly the big bets and overbets at the root: KdKc5h-6s bet
125% (0.18% of the range), -9.1% of the pot; 9c7d2d-9d after check, bet, call, bet 125% (0.10%), -9.0. The cause: the
opponent's answer in a branch the actor rarely takes is barely trained, because its regrets move only while the
actor's current strategy still takes the branch (the reach weights them); against that half-trained answer the
branch's EV comes out wrong, usually low. Best-response EVs instead of ours don't help (the answer is the problem,
not the arithmetic), and at exactly 0% the reference's own answer is untrained too, so neither number can be judged
there. For Practice's "EV given up" (a move the combo plays under 5% that its range takes 1% or more, against the
combo's top action): 0.75% of the pot off at 100 iterations (90th percentile 1.8), 0.34 at 300, 0.29 at 400, 0.10 at
1000; 3bet pots 0.14-0.20.

**What was done:** a label. Heads-up answers mark an action hero's range takes under `advisor.RARE_BRANCH` (1%) with
`"rough_ev"`, and the page shows its EV (and Practice the EV given up) with a "≈" and a tip; grading is unchanged.
**Not done:** a reach floor in the opponent's regret updates (every solve would change, in the C core and
`solver.py`, to fix a display at the tails); re-solving each rare branch with the actor's turn strategy frozen (~2 s
per rare action at the browser's Fast, so 2-4 s more per turn, where speed matters most).

## Stopping a budgeted solve sooner

Measured on 32 turns after library flops (BTN, CO, HJ, UTG vs BB, SB vs BB, HJ vs CO, BB's 3bet pots against BTN and
CO; after a check-through and after a third-pot c-bet called, two random boards each; the threaded site's tree,
natively), each solved to 1000 iterations with the exploitability and both first decisions' mixes and EVs recorded
every 50th iteration, the 1000th the reference (0.007-0.04% of the pot from equilibrium). Errors per combo over the
acting range: EV over the actions it plays, % of the pot.

* **Finer checks** (every 50th to the 500th instead of 300, 400, 600): the stop fires nearer the target, so the
  solve stops sooner at the same exploitability target: 2.4% fewer iterations at the threaded site's 0.2% from the
  300th (5 of 32 turns 400 -> 350), 6.4% at one thread's 0.3% from the 200th. But it moves along the same curve: at
  equal mean iterations the two grids give the same EV error (300th floor: old grid at 0.2%, 328 iterations, 0.260;
  finer at 0.18%, 333, 0.252; one thread: old at 0.3%, 244, 0.427; finer at 0.25%, 247, 0.412), and at the shipped
  targets the finer grid's EVs were 7% (threaded) and 12% (one thread) further off. Not adopted: a speed-up bought
  with accuracy, which a target change would buy as well.
* **Stopping on settled EVs** (stop at the first check past the floor where the exploitability is under target and
  the node's per-combo EVs moved under a threshold since the last check, thresholds 0.02-0.08% of the pot): the same
  iterations-for-accuracy trade as the exploitability rule at every threshold; no gain.
* **Lower floors** trade accuracy the same way: a 3bet pot's turn stopped at the 200th instead of the 300th is
  0.289% of the pot off per combo instead of 0.118.
* **Seeding from the nearest stored turn** (CO, UTG vs BB's turns seeded from BTN vs BB's file, card-substituted):
  mixed on the first four turns measured. CO vs BB after a c-bet called 0.270% of the pot from equilibrium at the
  200th against 0.252 cold, the same by the 300th; in position after a check-through 0.169 at the 300th against 0.211
  (the stop at the 300th instead of the 400th); UTG vs BB the same as cold (0.222 / 0.228 at the 200th). BTN vs BB's
  own file (`Advisor._store_seed`) gives 0.192 against 0.270 on its turn. Measured again on 12 CO, HJ and UTG vs BB
  turns, and with a live flop's own turn on 14 more (limped, 40 and 60bb), against each turn's own 1000-iteration
  reference, it was adopted (`NEAR_SEED`, `PREV_SEED`; README): to the settled stop 325 -> 300 mean iterations on the
  12 and 450 -> 413 on the limped, each combo's EVs over the actions it plays 0.264 -> 0.166% of the pot off on the
  12. On 8 turns seeded from their spot's own file (BTN vs BB, BB vs CO's 3bet pot; the shipped `STORE_SEED`) the
  seed is flat: 325 -> 312 iterations, EVs 0.267 -> 0.271 off.
* **How much a seed can buy**: five turns (CO vs BB checked through, two limped, a 40bb, BTN vs BB checked through)
  seeded from their 1000-iteration reference's own strategy (an oracle) at the same two nodes are as far from
  equilibrium as with the shipped seeds: at the 300th 0.172 / 0.421 / 0.139 / 0.128 / 0.179% of the pot against
  0.170 / 0.489 / 0.147 / 0.123 / 0.179 (cold 0.194 / 0.557 / 0.178 / 0.144 / 0.226). Seeded at every turn decision
  before the river (20 nodes), the regrets three times the seed's, 0.148 / 0.332 / 0.155 / 0.112 / 0.168: at most one
  settled check sooner. The rivers below hold the rest, and a stored turn holds no river, so seeding stops there.
* **A seeded turn stopping from the 250th** (a check added there for seeded solves; the threaded site's 0.2%): the 26
  near-spot and live-flop turns 335 -> 308 mean iterations, each combo's EVs 0.169 -> 0.189% of the pot off (cold, as
  shipped, 0.222), mixes 3.90 -> 4.26 points per combo (cold 4.21); but the 8 seeded from their own file 312 -> 275 at
  0.271 -> 0.332 (cold 0.267; a 3bet pot's turn checked through 1.075 -> 1.331). Not adopted: iterations bought with
  the seed's accuracy, as the finer grid buys them with cold accuracy.

Besides the seeds, what was adopted is a first look rather than a sooner stop: the request's own solve stopped at its 100th
iteration (and Accurate's 300th), read and shown as approximate while it goes on to the same answer
(`Advisor.progress`, README).

## Conclusions

* The previous finding holds and is now measured carefully: PCFR+ (last instantaneous regret as the prediction,
  quadratic averaging, alternating updates, verified on AKQ and against a node-by-node traversal) is 3-7x more
  exploitable than DCFR at 100-200 iterations on every poker spot here, and PDCFR+ 2-5x. That matches what the
  papers report for poker; their wins are on games without poker's dominated actions.
* Hyperparameter schedules need the run's length and pay at thousands of iterations; at the 50-300 iterations a
  budgeted solve gets they are worse early and at best ~10-20% better at the end, no better than the ramp.
* The one change that helps everywhere here is weighting the average harder toward recent iterations once the
  first iterations are behind: `dcfr-ramp` (gamma 3 + t/10, capped at 8), now the default. 10-20% less exploitable
  at 50-200 iterations on flops, the same on the river and turn, no cost per iteration, horizon-free (budgets and
  `Solution.extend` don't need to know the iteration count).
* Beta 0 is fastest early but can swing late (above): `dcfr-ramp` now raises beta to 0.5 at the 100th iteration,
  5-13% more exploitable at 200-300 iterations on the standard spots and better on the wide Accurate turn, the
  same through 100, and no late swings on the turns that had them (`dcfr-ramp-b0` keeps beta 0).
* The EVs a live turn shows are as good as its convergence (gap ≈ 0.04 + 0.4 x exploitability, % of the pot), except
  for actions the range almost never takes, whose opponent answer is barely trained: 1-10% of the pot off at Fast,
  under 1% at Accurate. The page marks them (`rough_ev`, "≈") rather than paying for a fix in every solve.
* Not done: regret-based pruning (an action can only be skipped when no hand of the range plays it, rare in a
  vectorised range-vs-range solve), learned discounting (DDCFR: a trained policy and per-iteration features).
