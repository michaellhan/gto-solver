# Preflop solving, equity realization (EQR) and 6-max 100bb cash equilibrium: research note

All pages were accessed in 2026-10. Each number below comes from the cited page, either quoted or in a table I transcribed. Where a page's figures appear only in images, I say so. Reliability tags:

- **[vendor-solve]**: the vendor's own solver output, stated in text.
- **[solver-report]**: a solver aggregate report quoted by a third party.
- **[heuristic]**: an author's estimate with no solve cited.
- **[uncertain]**: I could not confirm the figure on a primary page.

---

## 1. How preflop solvers model postflop play

### 1.1 GTO Wizard

**Legacy precomputed preflop (the "General" and "Simple" libraries)**

Source: GTO Wizard blog, "Status and info about our solutions" (https://blog.gtowizard.com/status-and-info-about-our-solutions/).

- Preflop was solved in **MonkerSolver**: "Preflop was solved in MonkerSolver". The tree was "very complicated" and "allows for calling in all situations", which "adds massive amounts of RAM requirements".
- Card abstraction: buckets **"30,30,30"**, with "Perfect, Large, Large" (Monker's texture or abstraction settings for flop, turn and river; the page does not explain them).
  - This means preflop is solved jointly with a full postflop tree under card abstraction. There is no EQR shortcut.
  - The postflop spots in the library were then re-solved separately with an exact postflop solver.
- Rake: "Rake 5%", and "solved for PokerStars Zoom rake structure both preflop and postflop". Caps: NL500 0.6bb, NL50 4bb.
- "Best" open sizes: UTG and HJ 2x, CO 2.3x, BTN 2.5x, SB 3x. These belong to the 6-max "General Smaller 3bets" set.
- Postflop accuracy: 0.2% of pot (General), 0.25-0.4% (Complex), 0.4% (NL500 Basic). The page gives no preflop accuracy.

**Rake ladder used by GTO Wizard cash solutions**

Source: note.com, Hatano, "Types of Preflop Solutions and the Rake Problem" (https://note.com/ryu1poker_2nd/n/na39903901649?hl=en).

| Stake | Rake | Cap |
|---|---|---|
| NL25 | 4.5% | 8bb |
| NL50 | 5% | 4bb |
| NL100 | 5% | 2.5bb |
| NL200 | 5% | 1.375bb |
| NL500 | 5% | 0.6bb |
| NL1k | 4.5% | 0.3bb |
| NL2k | 4.5% | 0.15bb |
| NL5k | 4.5% | 0.06bb |
| NL10k | 4.5% | 0.05bb |

- GG R&C is 5% with a 3bb cap.
- The same author says "General" (complex postflop sizing) and "Simple" (simplified postflop) preflop solves give different preflop ranges, because postflop sizing changes preflop EV.
- He also says more postflop branches generally favour the IP player.

**Rake sensitivity**

A Two Plus Two user comparison (https://forumserver.twoplustwo.com/15/poker-theory-amp-gto/gto-wizard-rake-comparison-analysis-1790166/, via search snippet) said "the rake cap has almost no impact on opening or 3betting frequencies". **[uncertain: informal analysis]**

**GTO Wizard AI (newer, custom solving)**

- "GTO Wizard AI Explained" (https://blog.gtowizard.com/gto-wizard-ai-explained/):
  - It "thinks about one street at a time". Future streets are valued with "expected values learned from its neural networks", trained by self-play.
  - It handles up to 200bb, at about 3 s per street.
  - Benchmark: +19.4bb/100 over 150,000 hands vs Slumbot.
  - The article gives no exploitability numbers.
- "Introducing Multiway Preflop Solving" (https://blog.gtowizard.com/introducing-multiway-preflop-solving/):
  - It uses a "hybrid approach that combines traditional CFR solving with neural-network value estimation", with the network "trained through massive self-play over billions of hands".
  - **Fast mode** "uses a shorter lookahead and re-solves more often". **Classic mode** solves street by street. The page notes that "the AI evaluation assumes optimal play beyond the depth limit".
  - "Chip EV and Raked Solving" is supported. Example: live rake of 10% capped at 2bb.
  - Up to 9 players. The page gives no accuracy figures.
- Custom raked postflop solving was at one point "limited to two-player postflop solutions only" (https://blog.gtowizard.com/customizable-raked-solutions-with-gto-wizard-ai/, via search snippet).
- Summary: the GTO Wizard AI preflop engine is depth-limited CFR with neural-network leaf values that stand in for postflop play. It is not an EQR coefficient model.

### 1.2 MonkerSolver

- MonkerWare comparison page (https://monkerware.com/compare.html):
  - "MonkerSolver therefore comes with a number of abstraction settings which reduces the game size considerably."
  - Similar hands are merged into buckets "which must then be played with the same action frequencies".
  - It compares preflop abstraction settings (30/Large, 30/Small, 30/None) against Cepheus's full Limit Hold'em solution.
  - The page mentions no EQR or preflop-only mode. **[uncertain whether such a mode exists; I found no page describing one]**
- Two Plus Two MonkerSolver thread (https://forumserver.twoplustwo.com/167/poker-software/monkersolver-1651569/, pages 15/24/56-69, via search snippets):
  - The guide suggests "15-30 bucket strength per street".
  - A 6-max preflop solve "with all cold-calls" at very high precision needs "over 512 GB RAM". "If you are willing to sacrifice cold-calls" it fits "under 256 GB RAM".
- Summary: Monker plays the full game (preflop plus every postflop street) under imperfect-recall card abstraction (strength buckets times texture classes) and a small action abstraction.

### 1.3 Simple Preflop Holdem (SPH) and Simple Postflop

- GTOSims, "Comparison of Simple Preflop Holdem solutions calculated with different settings" (https://gtosims.com/blogs/news/comparison-of-simple-preflop-holdem-solutions-calculated-with-different-settings-examples-of-gtosims-solutions-formats):
  - For preflop runs, the postflop tree is calculated, but "the postflop strategy is not saved due to abstractions (bucketing)". SPH buckets similar hands from v2.0.
  - Two postflop sizings are enough to capture equity realization; more sizings "practically do not affect the strategy".
  - Bucket test (6-max 100bb, CO open, chipEV): 10,000/10,000/10,000 flop/turn/river buckets vs the maximum 10,000/100,000/100,000 gave "minimal differences", with "actions EV of mixed strategies differ by less than 0.001BB".
  - Tree test: a simplified tree vs a detailed tree (donks, many sizes), both at 5,000/5,000/5,000, gave "identical" results. The detailed tree needed "> 350 GB RAM" at larger bucketing.
- So SPH is **not** a static EQR-coefficient model. It is a full postflop tree with bucketed card abstraction, like HRC and Monker.
  - PokerNews (below) describes it loosely as factoring in "postflop betting and position". **[I found no SPH page describing explicit per-position EQR parameters]**
- PokerNews, "Big Blind Defense - Holdem Resources Calculator vs. Simple Preflop Holdem" (https://www.pokernews.com/strategy/bb-defense-holdem-resources-vs-simple-preflop-holdem-38183.htm):
  - Setting: 20bb, 9-handed, 0.13bb ante, BTN min-raise, pot 4.67bb, about 17.6% equity needed.
  - The old HRC (checked down after preflop) has the BB call 81.8% and jam 18.2%, playing 100% of hands.
  - It gives example EQR values (equity / EQR), apparently from the SPH-style model: **A9s 57.27% / 113.44%**, **76s 39.19% / 91.13%**, **K8o 44.79% / 67%**, **Q5o 38.11% / 56.44%**.
  - It says 72o would need above 60% EQR to call.
- Simple Postflop (https://simplepoker.com/en/Help): "Full calculation of GTO strategies without usage of any abstractions" (v1.0 notes). This applies to postflop solving, not preflop.

### 1.4 HRC (HoldemResources Calculator)

- "Tree Configuration in HRC" (https://www.holdemresources.net/docs/tree-config/):
  - **Postflop Off**: "the calculation will assume that pots are checked down after the flop with no further betting". Recommended only for push/fold.
  - **Simple**: enables the postflop betting model. "the postflop stage serves primarily to estimate equity realization accurately".
  - Abstraction defaults (HRC Pro): **1024 / 256 / 256** buckets for flop/turn/river. The page suggests 2048/512/512 or 4096/1024/1024. HRC Classic is fixed at 256 per street.
  - It advises a larger flop abstraction than turn or river, since the flop most affects preflop ranges.
  - Bet sizes are "Geo Size %" geometric sizes. For example, a 66% target at SPR 10 becomes 57% pot, getting stacks in over four bets. Optional fixed flop sizes, donk options, "Add all-in SPR" and a "Final Betting Round" setting (check down after a given street, for multiway) are available.
  - Warning: do not disable postflop for limp or flat-call lines, or "calling ranges will look unrealistically wide".
- "Introducing HRC v3" (https://www.holdemresources.net/blog/2023-hrc-v3-release/):
  - "This feature builds a postflop game tree, not just applying a static equity realization approximation."
  - It uses imperfect-recall abstractions that "consider not just hand equities but also how hands realize their equity on future streets". Options run from 64 to 16k buckets (Classic: 256). Classic trees are capped at 25,000 nodes.
- HRC has no per-hand-category EQR table. Its "postflop model" is a bucketed, abstracted postflop game.

### 1.5 PioSolver, GTO+, ICMIZER

- PioSolver Edge includes a heads-up preflop solver, per the product pages (https://piosolver.com/products/, https://piosolver.com/docs/product_comparison/, via search snippets).
- A 2+2 thread (https://forumserver.twoplustwo.com/15/poker-theory-amp-gto/piosolver-preflop-solving-1809016/, snippet) says it needs "at least 64GB of RAM".
- A 2019 guide (bitbcash.com, snippet) credits Pio with preflop solving "via the use of weighted flop subsets which closely approximate the full game". A forum user suggests about 100 flops. **[uncertain: secondary sources; official doc page returned 404]**
- GTO+ and ICMIZER: I found no primary documentation of a preflop-with-postflop model in this pass. **[not researched to confirmation]**

### 1.6 Academic

- **Libratus**: "does not use any card abstraction on the first (preflop) and second (flop) betting rounds". The turn has 55M hands in 2.5M buckets; the river has 2.4B in 1.25M buckets. Its action abstraction came from common actions of prior ACPC AIs. Source: IJCAI 2017 (https://www.ijcai.org/proceedings/2017/0772.pdf, via search snippet).
- **Pluribus**: the blueprint is used as-is only on the first round. "Pluribus uses information abstraction only to reason about situations on future betting rounds". Preflop needs no card abstraction (KDnuggets summaries, https://www.kdnuggets.com/2019/08/inside-pluribus-facebooks-new-ai-poker.html). **[blueprint bucket counts per later street not confirmed; a "200 buckets" figure is unverified]**
- **Modicum** (Depth-Limited Solving, arXiv 1805.08195, https://arxiv.org/pdf/1805.08195): the blueprint "treats every poker hand separately on the first betting round (where there are 169 strategically distinct hands)". Later rounds use "30,000 buckets".
- **DeepStack** (arXiv 1701.01724):
  - Flop and turn counterfactual-value networks, plus an "auxiliary network for values before any public cards are dealt" to speed up preflop re-solving.
  - "The turn network was trained by solving 10 million randomly generated poker turn games."
  - Lecture slides (cvut.cz) say 10M situations for the flop and preflop networks. **[flop count not confirmed in the paper text I could read]**
- **Will Tipton**: PokerStrategy write-up (https://www.pokerstrategy.com/?p=31669) on his approximation, EV(turn) ≈ R·Equity·Pot, later replaced by a "capture factor" mapping by board texture and equity. I found no preflop EQR table from *Expert HU NLHE*.
- **Equity-realization estimation**: I found no academic paper estimating EQR tables. In practice, solvers derive EQR from an abstracted postflop solve.

**Takeaway for implementers.** No serious commercial preflop solver uses a static per-hand EQR coefficient today. Every one found solves an abstracted postflop game:

- Monker 30-bucket strength plus texture.
- HRC 1024/256/256.
- SPH 10k/10k/10k with 2 sizings.
- GTO Wizard AI: depth-limit plus neural-network values.

SPH's own tests suggest postflop action abstraction can be very coarse (2 sizings) with negligible preflop effect.

---

## 2. Published equity realization (EQR) figures

Definition (GTO Wizard glossary, https://gtowizard.com/glossary/equity-realization-eqr/): EQR = pot share / equity. The worked example is 0.7 / 0.4 = 175%.

### 2.1 Range-level EQR by position (solver aggregate reports)

**Poker Academy, "Equity realization (EQR) in poker"** (https://poker.academy/blog/post/how-to-understand-equity-realisation-eqr) [solver-report]. Figures are for the first flop decision, weighted average over flops.

| Spot | Flops | IP EQR | OOP EQR |
|---|---|---|---|
| BTN vs BB, MTT 20bb | 25 | 1.15 (EV 3.77bb, eq 59.71%) | 0.80 (EV 1.79bb, eq 40.88%; report column 0.79) |
| LJ open, BB call, **6-max cash 100bb, no ante** | 25 | **1.14** | **0.87** (report 0.86) |
| SB limp, BB check (HU MTT 30bb) | 184 | 1.14 | 0.85 (see note: labels inconsistent) |
| BTN open, BB call, MTT 20bb | 184 | 1.17 | 0.78 |

- Summary ranges: IP 1.12-1.17, OOP 0.78-0.88.
- Note on the limped pot: in a limped HU pot the BB is IP postflop. But the page's by-high-card table gives the higher figures to the SB ("SB EQR" 1.21/1.12/1.14) and the lower ones to the BB ("BB EQR" 0.77/0.88/0.87). **[inconsistent on the page; it may be a BTN-open spot mislabelled. The page itself also flags 30bb vs a "20bb" title]**
- By flop high card (HU limped pot, 184 flops), BB/SB: A-high 0.77/1.21, K-T high 0.88/1.12, 9-high or lower 0.87/1.14.
- BB per-flop EQR (20bb, BTN vs BB): A53r 0.62, 975r 0.83, 742r 0.86, 888 0.48.
- By hand on 975r (BB):
  - Straights 86s 276%, 86o 275%.
  - Two pair 97s 217%, 75s 207%.
  - Offsuit broadways without a pair or draw: QJo 14%, QTo 16%, KJo 19%, KTo 19%.
  - Suited vs offsuit: KJs 38% vs KJo 19%, QTs 37% vs QTo 16%, QJs 34% vs QJo 14%.

**PokerCoaching, "Become A Poker Equity Expert"** (https://pokercoaching.com/blog/become-a-poker-equity-expert/) [solver-report]
- BTN open, BB call, PioSolver aggregate report, flop 9s3s2d: "the BB only realized 79.1% of his preflop equity where the IP player realized 118.1%".
- BB equity 46.5%, BB EV share 36.8%, IP 63.2%.

**GTO Wizard "The Importance of Board Coverage"** (https://blog.gtowizard.com/the-importance-of-board-coverage/) [vendor-solve]
- 40bb UTG vs BB SRP on 652r: UTG equity 55.1%, **EQR 107%**; BB **EQR 91%**.
- With an "Old School Live Player" UTG range of the same width: UTG 51.3% equity, EQR 88%; BB 113%.

### 2.2 Hand-level EQR examples

- **PokerNews (SPH/HRC article above), BB vs BTN min-raise, 20bb, 9-max, ante**: A9s 113.44%, 76s 91.13%, K8o 67%, Q5o 56.44%. These are preflop hand EQRs vs the BTN range.
- **Modern Poker Theory (Acevedo)**, via search snippets of the book text and a 2+2 thread (https://forumserver.twoplustwo.com/15/poker-theory-amp-gto/equity-realization-1774200/, which returned 403 on direct fetch):
  - The worked example is 9c5d calling a UTG open from the BB: "equity vs UTG range = 29.5%". The call needs about 61% EqR, and 9c5d's "EqR factor of 58%" is "3% lower than the minimum 61%".
  - **I could not find Acevedo's position or hand-class EQR tables online. Not reported here.**
- **PokerStrategy (GTO Wizard example), UTG vs BTN SRP, A92 two-diamond flop** (https://www.pokerstrategy.com/news/content/Which-hands-over-or-under-realise-equity_131327/):
  - 44 46-56%, 22 (set) 217%, KQo up to 47% (33% equity).
  - KdQd 117% (54% equity), 8h7h 72% (20% equity), KhJh 29% (49% equity).
- **GTO Wizard "Equity Realization"** (https://blog.gtowizard.com/equity-realization/), MTT 50bb, HJ open, BB call, JhTd9h checked to:
  - BB: 3h2h 87%, 98s "less than 25%", 9d8d "nearly 50%".
  - HJ A♦2♦ "almost 100%", while BB's A♦2♦ "less than 2%".
  - Also from the GTO Wizard glossary or blog: on J♠8♥5♥, A♠9♠ with 43.3% equity realizes under 1/3, while 6♥3♥ realizes over 90% (https://gtowizard.com/glossary/equity-realization-eqr/).

### 2.3 Heuristic EQR tables (no solve cited; use only as priors)

**Deepfold, "Equity Realization: Why Raw Equity Isn't Everything"** (https://deepfold.co/en/blog/equity-realization-guide) [heuristic: no sources cited]

| Position (vs single raiser) | EQR |
|---|---|
| BTN | ~115% |
| CO | ~108% |
| HJ | ~102% |
| MP/LJ | ~100% |
| UTG | ~98% |
| BB | ~75% |
| SB | ~65% |

| Hand class | EQR |
|---|---|
| Small pairs 22-66 | 105-120% |
| Suited connectors 54s-T9s | 105-115% |
| Suited broadways | 100-108% |
| Suited aces A5s-A9s | 95-105% |
| Offsuit broadways KJo/QJo | 75-90% |
| Offsuit aces A9o-A2o | 70-85% |
| Offsuit junk T8o/J7o | 60-80% |

Other figures from the same page:
- Modifiers: "+20% IP, -25% OOP".
- SPR multipliers: high-EQR hands 1.05-1.10x at SPR 1-3, 1.10-1.20x at 4-10, 1.20-1.40x above 10. Low-EQR hands 0.90-0.95x, 0.75-0.85x and 0.60-0.75x at the same SPRs.
- SB QJs: flat ~65%, 3bet ~85%.

A Deepfold 3-bet guide (https://deepfold.co/en/blog/three-bet-strategy-guide, via search snippet) gives 3bet-pot OOP ≈ 80% and IP ≈ 110% [heuristic].

**Other heuristic sources**
- deucescracked.com, via search snippet: IP 100-110%, OOP 75-90%. "In-position hands routinely realise 10-20% more equity than the identical holding out of position" [heuristic].
- Tournament Poker Edge (https://www.tournamentpokeredge.com/?p=21717):
  - OOP R "could be anywhere between 25% and 75% depending on how villain plays".
  - With a big edge, "100% or even slightly more".
  - IP flatting a 3-bet with about 30% equity, R of "125% or even 150%" [heuristic].
- 888poker (via snippet): analysts assume a blanket figure such as "65% realization out of position" [heuristic].

### 2.4 Synthesis: what the solver reports agree on

- **SRP, IP preflop raiser vs BB caller**: IP 1.12-1.18, BB 0.78-0.88. The 100bb cash LJ-vs-BB report gives 1.14/0.87. The Pio BTN-vs-BB single flop gives 1.18/0.79.
  - Note the product identity: IP_EQR·IP_eq + OOP_EQR·OOP_eq = 1 (pot shares sum to 1). With BB equity about 0.41-0.46, BB EQR about 0.8 implies IP about 1.15.
- **By hand class** (no clean preflop table found from a solver):
  - The only preflop per-hand numbers (PokerNews, 20bb) show suited Ax about 1.13, suited connectors about 0.91, offsuit kings/queens 0.56-0.67, all for the BB.
  - Flop-level data consistently shows suited hands about 2x the EQR of the same offsuit hand when unpaired (KJs 38% vs KJo 19%).
- **3bet and 4bet pots**: no solver-sourced range EQR found. The only figures are heuristic (OOP about 0.80, IP about 1.10).

---

## 3. The 6-max / 8-max 100bb cash equilibrium: published frequencies

Different sources use different rake settings, and rake matters.

### 3.1 Source A: GTO Preflop (gtopreflop.com) [vendor-solve]

- Format: 8-max, 100bb, **5% rake capped at 4bb**, opens 2.5bb (SB 3bb).
- Built from 503 solve files. The vendor claims 6-max runs land "within one point" of the 8-max figures for shared positions.
- The solver engine is not named.

**RFI** (https://gtopreflop.com/posts/multiway-preflop-ranges)

| Seat | UTG | UTG+1 | LJ | HJ | CO | BTN |
|---|---|---|---|---|---|---|
| Open % | 13.9 | 15.8 | 18.1 | 22.2 | 29.2 | 43.7 (fold 56.3, never jams) |

SB first-in (https://gtopreflop.com/posts/blind-vs-blind-strategy): raise to 3bb **43.4%**, complete **2.3%**, fold 54.3%, jam 0%.

**BB defense** (https://gtopreflop.com/posts/big-blind-defense). Fold is derived as 100 minus defend.

| Opener (size) | Call | 3bet | Defend | Fold |
|---|---|---|---|---|
| UTG 2.5 | 14.6 | 3.6 | 18.3 | 81.7 |
| UTG+1 2.5 | 15.9 | 4.1 | 20.0 | 80.0 |
| LJ 2.5 | 16.8 | 4.5 | 21.4 | 78.6 |
| HJ 2.5 | 19.6 | 5.9 | 25.6 | 74.4 |
| CO 2.5 | 23.8 | 7.7 | 31.4 | 68.6 |
| BTN 2.5 | 29.8 | 11.5 | 41.3 | 58.7 |
| SB 3.0 | 33.8 | 16.2 (to 10bb) | 50.0 | 50.0 |

- MTT comparison (ChipEV, 12.5% ante, no rake): BB vs BTN defends 82.4% at 100bb, vs 41.3% in raked cash.
- SB vs BB 3bet (BvB): fold 63.6%, call 21.4%, 4bet to 27bb 15.0%.
- Cold-calling vs an LJ open (multiway page): HJ 2.1%, CO 2.9%, BTN 4.6%, SB 4.1%, BB 16.8%. "The four seats with players still to act behind them cold-call between 2.1% and 4.6% of hands."

**4-bets and fold to 3-bet** (https://gtopreflop.com/posts/4-bet-ranges-guide)
- 4bet vs an IP 3bettor: about 18-19% of the opening range. UTG vs BTN 19.0%, CO vs BTN 19.2%, LJ vs BTN 18.7%, UTG vs UTG+1 18.0%.
- 4bet vs blind 3bets: about 9-15%. CO vs BB 9.6%, BTN vs SB 10.4%, BTN vs BB 10.3%, UTG vs SB 14.5%, SB vs BB 15.0%.
- All-in 4bets: 0.0% in every 100bb cash node.
- Sizes: about 2x the 3bet IP and about 2.7x OOP. 3bets are 7.5bb IP, 12bb from the blinds and 10bb SB vs BB.
- Fold to 3bet: 56-76% fold, 14-25% call. UTG vs BTN 56.1%, CO vs BTN 64.0%, UTG vs SB 71.4%, BTN vs SB 75.7%, BTN vs BB 69.7%.
- 4bet bluffs:
  - CO open vs BTN 3bet: A5s 84.8%, A4s 55.9%, KJs 100%.
  - BTN vs SB 12bb 3bet: A9s 77.5%, KTs 75.8%, A4s 42.8%, K9s 33.9%, KQo 32.6%, AJo 23.4%, A5s 21.9%.
  - QQ 4bets only 15% after a UTG open, TT 10%.
- Facing a 4bet (call vs 5bet jam): **not published** ("outside its scope").

### 3.2 Source B: Poker Academy, "Normal NL200 GG" 6-max 100bb, no ante, 2.5bb opens [vendor-solve]

The rake value is not stated. GG NL200 has a high rake profile.

Sources:
- https://poker.academy/blog/post/poker-hud-for-you-take-the-advantage
- https://poker.academy/blog/post/what-is-a-3-bet-in-poker-a-theoretical-and-strategic-guide

**RFI**

| Seat | Open % |
|---|---|
| LJ (=UTG 6-max) | 18.5 (245 combos) |
| HJ | 23.3 |
| CO | 30.6 |
| BTN | 46.5 (617 combos) |
| SB | 45.0 (raise 42.1 + limp 2.9) |

**Responses**
- BB vs BTN: call 29.0, 3bet (11bb) 11.5, fold 59.5.
- BB vs CO: call 21.7, 3bet 8.1, fold 70.2.
- BB vs LJ: call 16.4, 3bet 4.5, fold 79.1.
- BTN vs CO: 3bet (7.5bb) 12.5, call 3.0, fold 84.5. AJo 95%, KQo 84%, T9s 87%, 65s 85% and A8s-A3s 77-99% 3bet.

**Squeeze and 3bet pots**
- Squeeze, BB vs BTN open plus SB call: squeeze (12.2bb) 12.7%, call 16.1%, fold 71.2%.
- CO facing BTN 3bet: fold 63.7%, call 17.2%, 4bet (17.2bb) 19.1%. AA, KK and AKs 4bet 100%, QQ 98.6%.
- BTN facing BB 3bet (11bb): fold 68.7%. LJ facing CO 3bet: fold 61.2%.

### 3.3 Source C: GTO Wizard NL500, 6-max 100bb [vendor-solve, text figures only]

Rake: NL500 is 5% with a 0.6bb cap, per Section 1.1.

- BTN cold-calling (https://blog.gtowizard.com/playing-calls-from-the-button-in-cash-games/):
  - Vs UTG: call 8.2%, 3bet 7.5%, "nearly 16%" played. This drops to "under 10%" if limited to 3bet or fold.
  - Vs CO: 3bet ~12%, call 5%.
- BB squeeze range (https://blog.gtowizard.com/responding-to-bb-squeezes/): "7.9% of hands, with 100bb stacks" vs a UTG open, "vs 13% into a CO open". The caller seat is not given in the text.
- SB vs BTN 2.5x (https://blog.gtowizard.com/crush-3-bet-pots-oop-in-cash-games/): "the SB barely calls anything. Instead, they almost exclusively 3-bet or fold"; "The BB calls way more". The BB 3bets TT+, AJs+ and AQo+ purely. The exact frequencies appear only in images.
- Other GTO Wizard text: SB 100bb chipEV opening has "no pure raises and many hands that mix call and raise", and with very deep stacks "SB rarely open-raises, preferring to limp-reraise" (search snippets of GTO Wizard limping articles). This chipEV, no-rake result contrasts with raked cash, where the SB is raise-or-fold with about 2-3% limps.

### 3.4 Blind vs blind, other sources

- ThinkGTO "GTO Ranges+" (https://thinkgto.com/blog/blind-vs-blind-without-a-solver-heuristics), 100bb, rake and format unstated:
  - SB first-in: raise 3bb 33.1%, complete 20.5%, fold 46.4%.
  - BB vs 3bb raise: 3bet (9bb) 16.9%, call 38.7%, fold 44.5%.
  - **[uncertain: likely a chipEV or low-rake model, which explains the many limps]**
- PokerNews on GTO Wizard AI (https://www.pokernews.com/strategy/how-should-you-respond-to-a-2x-open-blind-on-blind-52371.htm), 100bb:
  - The SB GTO strategy opens 3.5bb at 9% and 2bb at 3.1%. The rest is presumably limp or fold. **[unclear]**
  - BB vs 3.5bb: call 56.8%.
  - BB vs the GTO 2bb open: fold 0.7%.
- GTO Preflop tournament BvB (ChipEV, 12.5% ante), SB at 100bb: fold 8.4%, limp 86.0%, raise 4bb 5.6%. BB raises about 41.6% of limps (40.4% "in cash").

### 3.5 Heads-up

- GTO Wizard "Heads up! Exploiting SB's Preflop Mistakes" (https://blog.gtowizard.com/heads-up-exploiting-sbs-preflop-mistakes/), HU **50bb**, chipEV, 0.12bb ante:
  - The SB "only folds 4% of hands", "open-raising at a high frequency with some limping". An open-raise share of 64% is cited, so limps are about 32%, derived.
  - BB vs limp raises 34%. Vs a 2.1bb open the BB 3bets to 8.4bb. The exact frequency is not given.
- **HU 100bb cash**: I found no solver-sourced frequencies.
  - Training blogs (preflopwizard.app, snippets) give SB limps "25-40%", BB defense "62-70%" and BB 3bet "15-20%" [heuristic, unreliable].

### 3.6 Population reference

The repository's own targets (VPIP 28.75 / PFR 18.7 / 3bet 8.9 / fold to 3bet 48.25 / 4bet 12.75) are population numbers, not equilibrium. For comparison:
- Equilibrium fold-to-3bet in raked cash is about 56-76% (GTO Preflop) and about 61-69% (Poker Academy).
- Equilibrium 4bet vs an IP 3bet is about 18-19% of opening range. Vs the blinds it is about 9-15%.

---

## 4. Multiway equity realization

- **I found no published solver EQR numbers for 3-way pots.**
- Qualitative and partial:
  - GTO Wizard "Playing In Position Against Two Callers" (https://blog.gtowizard.com/playing-in-position-against-two-callers/), 100bb symmetric 8-max chipEV, LJ open with SB and BB calls, 20-flop subset:
    - LJ checks "+11%" more than heads-up. The pot-sized c-bet drops from 18% (HU) to 1.3%.
    - The SB checks about 90% and donks 9.8%. SB average EV is 2.3bb.
    - The article says EV fell for both BB and LJ, with no figures given.
  - Upswing (https://upswingpoker.com/multiway-pot-preflop-squeezing-leaks/, snippet): in multiway pots "realizing equity is reduced because players see fewer board cards on average".
  - A search snippet attributed to a blog says OOP realization falls from about 85-90% heads-up to "75% or lower" multiway [heuristic, unattributed].
- GTO Preflop's database contains "0 decision points with two or more opponents already in the pot", so its solves exclude multiway preflop nodes beyond squeezes against a single caller as modeled.
- GTO Wizard AI now supports 3-way postflop solving and preflop up to 9 players, per its blog. Its multiway preflop is valued by its neural network, not by a published EQR table.

---

## 5. Gaps and caveats

- Acevedo's MPT position and hand-class EQR tables: not found online in legitimate form. Only the 9c5d example (29.5% equity, 58% EqR, 61% needed) was confirmed via snippet.
- GTO Wizard preflop frequencies are mostly in chart images. The text figures used here are BTN cold-call and BB squeeze only.
- I found no 100bb HU cash solver frequencies, and no 3bet or 4bet pot EQR from a solver.
- Rake strongly affects BB defense:
  - About 41% vs a BTN open at 5%/4bb (GTO Preflop) and about 40.5% at GG NL200 (Poker Academy).
  - About 82% in a no-rake ante MTT.
  - Rake-free 100bb cash BB defense was not found. Blogs claiming 52-65% cite no solve [heuristic].

---

## 6. What this repo does with it: a derived preflop equilibrium (2026-10)

Following section 1 (every serious preflop solver values a flop by solving an abstracted postflop game, not
by a fixed per-hand coefficient), the realization model here is *measured* with the repo's own postflop
solver and then used as the terminal value of a full preflop game:

- **Game** (`gto/pf_game.py`): 6-max, 100bb, 169 classes, class-level card removal. Opens 2.5bb (SB 3bb),
  3bets 3x in position / 12bb from a blind / 10bb BB over SB, one more open per cold caller, 4bets 2.3x,
  5bet all-in; at most three players see a single-raised flop and two a raised one (8,847 nodes). Discounted
  CFR level by level in numpy, 1,500 iterations (~0.002 bb/hand exploitability). Flops pay pot x realized
  share, raked like the advice's calls (5% to 4bb on 2.5x the flop pot); three-way pots use the product of
  heads-up equities corrected per class (`scripts/build_three_way.py`) and a guessed seat factor (0.88 / 0.97
  / 1.12; section 4: nothing published; measured since, section 7).
- **Realization** (`scripts/pf_equilibrium.py` -> `gto/pf_eqr.py`): 17 heads-up flop spots (single-raised,
  3bet and 4bet pots, the aggressor in and out of position, several seat pairs each), 68-102 texture-stratified
  flops each, the fast flop model (bucketed turn and river; on BTN vs BB its per-class realization is within
  ~0.05 of the per-combo model's, the range's within 0.003). Every class of the range a player had before
  its last action is added at weight 1e-3, so hands outside the range get their counterfactual value (what
  they would earn if added) and the preflop margins are measured, not extrapolated. Per (pot type, aggressor
  position, role): excess pot share over all-in equity against 9 opponent strength buckets, fitted on hand
  features and shrunk class residuals; pairs combine by a two-way additive decomposition that keeps every
  pair's shares summing to the pot.
- **Fixed point**: measure on the charts (round 0), solve, re-measure on the solve's ranges, damp 0.5.

**Realization it measures** (charts' ranges): BTN vs BB single-raised pot 1.13 / 0.85 (published 1.14-1.18 /
0.79-0.87, 2.1); 3bet pot, BTN 3bets CO: 1.13 / 0.87 (the one published example, 552r: 1.17 / 0.83); BB 3bets
BTN: 1.04 / 0.95. Per class the big blind's caller realizes 22 1.13, 76s 1.02, KQs 1.10, KJo 0.92, K7o 0.78,
A9o 0.73 (2.2's per-hand figures: A9s 1.13, 76s 0.91, K8o 0.67, Q5o 0.56 at 20bb).

**Against the published solves** (tests/preflop_refs.py, 22 spots / 14 hand facts; charts: 0.99 points,
0 outside tolerance, 0 hand facts off):

| | mean error | spots outside | hand facts off |
|---|---|---|---|
| round 0 (measured on the charts) | 2.50 | 12 | 8 |
| round 1 | 2.74 | 13 | 9 |
| round 2 (102 flops) | 3.01 | 13 | 10 |
| subgames entered with the charts' ranges (`scripts/pf_subgames.py`, 15 non-open spots) | 6.98 | 13 of 15 | 5 of 8 |

Where it lands: the big blind against a button open 58.1 / 29.3 / 12.6 (published 58.7 / 29.8 / 11.5),
CO against a BTN 3bet 63.5 / 10.3 / 26.2 (64.0 / 16.9 / 19.2), UTG against a BTN 3bet 59.8 / 20.6 / 19.6
(57.0 / 24.4 / 18.7), CO against a BB 3bet 65.7 / 22.4 / 11.9 (68.0 / 22.4 / 9.6). Where it misses: opens
are tight (BTN 37.6% against 43.7-44.5, CO 26.3 against 29.3) and favour offsuit broadways over small pairs
and suited connectors (UTG opens KTo and A9o, not 22-66 or T9s; published the reverse); the small blind
3bets the button 16% (11.5) and never flats. The fixed point did not move toward the references (rounds
0-2 above), and the per-combo flop model doesn't change the per-class realizations enough to explain the
hand order (above), so the misses look structural (the tree's sizes, no limps, the three-way model, the
realization model's form) rather than noise.

**Decision**: the hand-written charts fit the published solves far better (they were matched to them), so
they stay. The solve's other spots were checked for a clear improvement:

- The 3bettor facing the opener's 4bet: the full solve continues 44-68% of its 3bet range in the four seat
  pairs of `scripts/pf_compare.py` (the charts 28-56% of theirs); a subgame entered with the charts' own
  ranges continues 32-92% over the 15 seat pairs, jamming up to 75% (SB against BTN's
  bluff-heavy 4bet). Both say the charts over-fold against the charts' 4bet ranges, but the answer is a best
  response to those ranges (published 4bet ranges with A9s/KTs bluffs, 3bet ranges matched only in total),
  swings with them, and no published solve covers the spot: not adopted.
- Squeezes, cold calls, callers facing a squeeze: depend on the three-way model, which is a guess.
- Heads-up: the tree has no limp, and published 100bb heads-up play limps about a third of hands (3.5), so
  the solve (SB opens 63%, BB 3bets 31%) isn't comparable.
- Facing a 5bet jam: an all-in decision (equity against the jam range), which the advice already re-draws
  by equity at run time (preflop._vs_5bet_jam).

What is used: the realization model re-ranks re-priced calls (Advisor._priced, under rake or another size):
the number of hands a worse price removes is still set by all-in equity, but they are now the ones with the
least *realized* equity, so raked big-blind defence keeps 76s/54s/97s and folds K5o/Q7o/J5s (with all-in
equity it did the reverse, against raked solves). Since 2026-10 the realized equity there carries the river
correction above, smoothed on hand shape (advisor._river_fix: connectors and gappers up to J-high +0.15, 22-66
+0.10, offsuit aces -0.05; full in single-raised pots, half in 3bet pots; since extended to every suited hand +0.10 and every non-connected offsuit hand -0.05, section 8): at 5% capped at 4bb the big blind
now folds A2o (was 51% continue; gtopreflop.com: a pure fold) and keeps T8o/98o/87o (were 19/56/0%), the
totals unchanged (they are set by the price). Measured directly, rake's own per-hand incidence is too small to
reorder anything: 34 BTN-BB flops solved with and without 5% rake (fast model, chart ranges) cost the big
blind 7.7-9.1% of its pot share at a 0.6bb cap by hand group (offsuit aces 9.0, suited connectors 8.0, small
pairs 7.7) and 12.5-14.2% at 4bb, under 0.004 of the pot between groups.

## 7. The structural causes, one experiment each (2026-10, second pass)

Each row changes one thing in the derived game and re-solves it (5% rake capped at 4bb, 1,000-1,500
iterations); the columns are tests/preflop_refs.py's (mean error in points over the 22 certain spots, spots
outside tolerance, hand facts outside their band). The charts: **1.03 / 0 / 0**.

| change | mean | out | facts | what moved |
|---|---|---|---|---|
| baseline (gto/pf_eqr.py, round 2 of section 6) | 2.90 | 14 | 10 | BTN opens 37.6%; UTG folds 22-66, T9s and opens KTo, A9o |
| no rake | 5.46 | 16 | 10 | BB defends 61% against the button (published 41%); opens don't move (37.8%) |
| no three-way flops (`max_players=2`) | 2.80 | 13 | 9 | nearly nothing: the three-way model isn't the cause |
| SB may limp (`sb_limp`) | 3.26 | 15 | 10 | SB limps 8.6% (published 2.3-2.9%), BB raises 42% of limps (published ~40%); the rest unchanged |
| 4bets 2.0x in position, 2.7x out (`Sizes.four_x_ip/oop`, published) | 3.01 | 14 | 8 | 4bets stay 1.3-1.7x published |
| 4bet pots paid at all-in equity | 5.31 | 20 | 10 | 4bets fall, 3bets and folds rise, BTN opens 30%: a sensitivity test |
| suited and 22-99 +0.03 of the pot (and offsuit -0.015) | 2.93 (2.98) | 15 | 9 (8) | UTG still folds 44 |
| the flop model's river correction below, single-raised pots (all pots) | 3.01 (3.09) | 15 | 9 (10) | BTN A2o 98% -> 29%; the rest within noise |
| realization re-measured on the charts' ranges (round 0, 68 flops a spot) | 2.43 | 10 | 8 | |
| ... each measured spot's own fit (`--by-spot`, pf_game.SpotModel) | 2.70 | 16 | 9 | the pooled fit's range realization is within 0.03 of the direct measurement for 10 of 17 aggressors, 0.05-0.09 off for CO vs BTN and UTG vs CO 3bet |
| round 0 + measured three-way factors (below), 1,500 iterations | **2.42** | **10** | **6** | UTG opens 44 64%, KTo 2%; BB 3bets TT 65%, flats QQ vs a 3bet 89% |
| 9 global realization shifts fitted to the references (`+-0.03` by pot type and position, the river correction by pot type) | 2.17-2.27 | 8 | 7-8 | the best of 34 solves (coordinate search); no direction gains more than the noise between neighbours (about 0.3) |

**The flop model's fidelity.** The realization is measured with the fast flop model (turn by 32 equity buckets,
the river played without its card). `scripts/pf_fidelity.py` re-solves a few flops with the turn per combo and
the river dealt, bet and played (depth 3, 8 rivers a turn, ~100x the cost) on the charts' ranges:

| spot (flops) | range realization fast -> river dealt | per class (river dealt minus fast) |
|---|---|---|
| UTG vs BB single-raised (8) | UTG 1.084 -> 1.067, BB 0.880 -> 0.907 | UTG: 65s +0.25, J8s +0.20, 87s +0.17, KTs +0.14, 22 +0.08; AA -0.15, AJo -0.11, KK -0.10. BB: 65s +0.23, T8o +0.15, 54s +0.15, A5s +0.12, T9s +0.11; AJo -0.17 |
| CO vs BTN single-raised (6) | CO 1.095 -> 1.108, BTN 0.896 -> 0.885 | suited connectors and gappers +0.10-0.38, 22 +0.16-0.20, top pairs about 0 |
| BB 3bets BTN (6) | BB 1.002 -> 0.988, BTN 0.989 -> 1.009 | J8s, T8o, 54s +0.10-0.18; the rest within 0.1 |

So the fast model gets the ranges' realization within 0.03 but under-realizes the hands that make their
money when the river comes (suited connectors, gappers, small pairs) by 0.1-0.3 in single-raised pots, about
half that in 3bet pots, and over-realizes big pairs and offsuit aces a little. That is the direction the
published hand order needs (it opens 54s, folds A2o on the button), but applying the correction (smoothed on
hand features) to the game moves the hand order only partly (A2o) and the totals not at all.

**Three-way pots** (`scripts/pf_three_way.py`, the 3-player solver, charts' ranges, 6 flops, flop and turn bet):
first / middle / last to act realize 0.95 / 0.91 / 1.11 when the button opens and both blinds call, 0.88 / 1.05 /
1.05 and 0.87 / 0.97 / 1.14 when the opener is in the middle (CO open, BTN and BB call; UTG open, CO and BB call).
The guess was 0.88 / 0.97 / 1.12 by seat alone; pf_game.MW_REALIZE now has the measured factors by where the
opener sits. Either way three-way flops barely move the references (row 3).

**What the misses come from.** Two preflop terms decide most of the marginal opens, and both depend on the
other seats' raising: card removal (when CO opens, all three behind fold about 12% more often to K9o than
to 65s) and the 2.5bb lost when re-raised. A small pair or suited connector has to win that back postflop, and needs about
0.1-0.15 more realization in single-raised pots than the measured model gives; the river correction supplies
part of it. The equilibrium is sensitive enough that a 0.03 shift in one pot type's realization moves the mean
error by up to 0.5 points and up to 4 spots across their tolerance (last row), and nothing measured or fitted here
gets below 2.1. Matching 22 spots within their tolerances needs realization accurate to about 0.02 in every pot
type and per class, which a bucketed, two-size flop model can't give: the commercial solves play a full
postflop tree with the river dealt, and re-measuring every spot that way here is ~600 CPU-hours.

**Decision**: still not adopted, and no charts were generated from it: a fit to the references as good as the
charts' was the condition for trusting it where there are none, and the best here is 2.42 / 10 / 6 unfitted. Kept: the measured
three-way factors, the limp and 4bet-size options (off by default), per-spot fits (`--by-spot`), the fidelity
and three-way measurement scripts. The charts and the flop library are unchanged.

## 8. Third pass: raked hand order, cold calls, squeezes and 4bets (2026-10)

New published figures (searched: gtopreflop.com, GTO Wizard's blog, Poker Academy, GTO Gecko, thinkgto.com,
Preflop Wizard, Upswing-type guides; most guides give rules of thumb, not solver output):

- **Cold calls** (gtopreflop.com, multiway page, already in 3.1): against a LJ (6-max UTG) open HJ 2.1%, CO
  2.9%, BTN 4.6%, SB 4.1% of hands, "3-bet more often than they call". Now certain references.
- **BTN against a SB 3bet** (gtopreflop.com, 4-bet guide, fold / call / 4bet per hand): KK 0/0/100, QQ
  0/10.5/89.5, AKo 0/20/80, A9s 0/22/77.5, KTs 0/24/75.8, AKs 0/41/59.4, A4s 54/3/43, K9s 66/0/34, KQo 67/0/33,
  JJ 0/76/24, AJo 77/0/23, A5s 6/72/22, TT 0/81/19.5, AQs and KQs 0/100/0. Six are now raked hand facts.
- **CO against a BTN 3bet** (same): A5s 84.8%, A4s 55.9%, KJs 100% 4bet. UTG (8-max UTG, not the 6-max seat)
  TT 48/43/10, so not used.
- **GTO Gecko's library** (gtogecko.com/blog/wheel-ace-fourbet-study, "catalogued rake 5%, capped at 4bb",
  solver not named, exports "not validated optimal responses"): BTN vs a BB 12bb 3bet A3s 2/28/70 (fold /
  call / 4bet to 24bb), A4s 0/89/11, A5s and A9s 100% call; CO vs a BTN 7.5bb 3bet A2s pure fold, A4s 4/52/44,
  A5s 0/13/87; **the big blind's 3bet range against BTN's 24bb 4bet: fold 46.9%, call 29.3%, jam 23.8%**
  (127.7 of 146.7 combos, hero's As5s removed; As9s the same within 0.5). The only response to a 4bet found
  anywhere. Its A5s 4bet against a SB 3bet (6.4%) differs from gtopreflop.com's (21.9%): a different library.
  gtogecko.com/blog/suited-connectors-threebet-study: BTN folds 98s to a BB 3bet (99.9%), CO calls JTs (99.3%).
- **Squeeze** (Poker Academy, already in 3.2): BB over BTN open + SB call 71.2 / 16.1 / 12.7, calls at a 20%
  price against 29.0% at 27.3% heads-up. Reported (uncertain, like the other Poker Academy spots).
- **Facing an open and a 3bet** (GTO Wizard, should-you-ever-cold-call-a-3-bet): the button folds "more than
  90%" to "just under 97%". Reported.
- thinkgto.com's 3bet guide: "continue with approximately 30-40% of your 3-bet range" against a 4bet; no
  solve cited, not used.

What changed, each measured with `scripts/preflop_reference.py` (22 original spots: 1.03 -> 0.99 points, 0
outside; the 4 new cold-call spots 0 outside; hand facts 14/14 -> 15/15; raked hand facts 4/5 -> 11/11; 
`scripts/preflop_sweep.py` 0 issues unraked, at 5%/0.6bb and at 5%/4bb):

| change | before | after | published |
|---|---|---|---|
| river correction: other suited hands +0.10, every non-connected offsuit hand -0.05 (was: offsuit aces only; advisor._river_fix) | BB vs BTN at 5%/4bb: Q9o 100%, K2s-K9s 95% (K2s 60%), K9o/KTo 100%, K7o/K8o 75/63% | Q9o 33%, K2s-K9s 100%, K9o/KTo 73/69%, K7o/K8o 0% | Q9o 27%, K2s-K9s 100% |
| multiway calls re-drawn by equity against all players at the chart's bar (Advisor._priced), not only added to the chart's | BB over BTN + SB call, 5%/4bb: 46.4 / 44.5 / 9.1 | 75.9 / 15.0 / 9.1 (61.6 / 29.3 / 9.1 unraked) | 71.2 / 16.1 / 12.7 |
| BTN (and CO vs HJ) facing an early open: the NL500 flats cut, more 3bets | BTN vs UTG 85.7 / 9.1 / 5.2 | 89.1 / 4.5 / 6.4 | call 4.6, 3bets > calls |
| CO vs BTN 3bet: A5s 85%, A4s 55% 4bet, the rest folding, KQo 70%, K8s and Q9s 40% 4bet to keep the total (calls unchanged: the shipped flop library's BTN-3bet-vs-CO spot is built on them, so KJs keeps calling 60%) | A4s 100%, KJs 40% | A4s 55%, KJs 40% | 55.9%, KJs 100% |
| VS_4BET: QQ 0.8, AKo 0.8, JJ 0.3, A5s 0.4 jam (were 0.6 / 0.5 / 0 / 0.3) | BB vs BTN 24bb 4bet (As5s removed) 45.6 / 38.6 / 15.9 | 45.6 / 34.0 / 20.4 | 46.9 / 29.3 / 23.8 |

The suited correction follows section 7's river-dealt measurements (KTs +0.14, A5s +0.12, J8s +0.20 alongside
the connectors; AJo -0.11 to -0.17), smoothed to one value per shape; a grid (suited 0-0.12, offsuit 0 to
-0.08) left the 22 spots within 0.02 points of each other, and Q9o only came off 100% with both (offsuit
alone: 88-97%; suited alone: 93-100%). The multiway re-draw has no new parameter: the bar is the one the chart
already sets heads-up. It applies only with a raise in and the chart's own spot having an opponent (limped pots
keep the old rule: over-limps were the chart's calls). Squeezes stay at about 0.8x the heads-up 3bet over BTN
+ SB (9.1% against 11.6%; published 0.95-1.10x): flooring the bluffs' fold-equity scale at 0.85 moved it only
to 10.9%, so it was left.

**The 3bettor facing a 4bet.** The derived equilibrium (section 6) said the charts fold too often against
their 4bets. The one published response says the total is right where it can be compared: at 24bb the charts
continue 54% of the big blind's 3bets against the button with the same cards removed, the export 53%. What
the charts get wrong is the split (now closer) and the size response: the 4bet default is 2.3x (27.6bb over
12bb), the published ones 2x in position and 2.7x out (24bb, 20.25bb over 7.5bb), and the re-pricing adds
calls by equity at the chart's bar, under which much of a 3bet range sits bunched against a bluff-heavy 4bet
range: the big blind against the button folds 64% at 27.6bb, 45% at 24bb, 38% at 22bb and 5% at 20bb, against
the cutoff 47% at 27.6bb and 6% at 24bb. Moving the default 4bet to the published sizes makes 24bb the chart's
own line (64% folds, against the export's 47%) unless the chart widens, and changes every recommended 4bet
size; with one published spot it was left. Open: a 4bet-response reference per seat pair, which none of the
public libraries publishes.

## 9. Fourth pass: the flop library's values, and the other games (2026-10)

**The flop library as the realization.** The library (README: Precomputed flop library) holds, for 13 heads-up
spots, every combo's EV at the flop's first decision on 552-1755 flops each, from solves to 0.15% of the pot
(Accurate's tree; the 3bet pots with the river dealt). `scripts/pf_library_values.py` reads them into
`gto/pf_libvals.py`: per spot, each class's mean share of the pot over the flops it meets, weighted by how
often it meets each (for both players, 86 KB), the shares of each entry's two players adding to the pot within
0.0014-0.017 bb. `pf_game.LibraryModel` turns a spot into pairs S[i, j] = eq[i, j] + a[i] - c[j] (class i of the
aggressor against class j of the caller), a and c solved so each class's share against the other's range is
the library's; a class outside the spot's range (which the library can't value) takes the measured model's
excess (section 6), joined to the library's at the range's weakest 30%; a flop between seats with no spot of
its own takes the stored spot of the same pot type and position with the same caller, else the same
aggressor; 4bet pots the measured model. `scripts/pf_solve_library.py` solves the 6-max 100bb game with it
(5% rake capped at 4bb, 1,500 iterations, ~0.0004-0.004 bb/hand from equilibrium):

| model | mean error | spots outside | hand facts off | where it misses |
|---|---|---|---|---|
| the charts | **0.96** | **0** | **0** | |
| section 7's best (measured model, round 0, three-way factors) | 2.42 | 10 | 6 | |
| the library's values | 3.81 | 16 | 13 | BTN opens 39.3%; openers flat a 3bet 5-10% (published 17-24), 4bet 26-28% (18-19) |
| ... with the river correction (advisor._river_fix: full in single-raised pots, half in 3bet pots) | 3.14 | 15 | 12 | BTN opens 37.5%, BTN opens A2o 97% and not A4o; UTG folds 22-44; BB 3bets the SB 24% (16.2) |

Spot by spot it is as close as the charts or closer in three: the big blind against a HJ open (73.9 / 19.8 / 6.2
against 74.4 / 19.6 / 5.9; the charts 72.8 / 21.0 / 6.2), the button's 3bet of a CO open (10.4% against 11.2;
the charts 12.2) and the button against a small-blind 3bet (72.3 / 13.6 / 14.1 against 75.7 / 13.9 / 10.4; the
charts 71.9 / 16.5 / 11.7), each within the references' own spread (raked solves of one spot differ by about a
point). Its hand order is the trouble: a class the library holds is valued by what its combos earned on the
flops, one it doesn't by the measured model, and the two meet at the range's edge only on average, so the
classes just outside a range are mis-valued (the button opens A2o and folds A4o, the reverse of every published
chart; UTG folds 44). It also calls 3bets less than the published solves and 4bets more in every seat pair,
the direction section 6 found. Not adopted, there or anywhere: the three spots gain less than the
references' noise, their per-hand strategies fail the facts the charts pass, and the library's spots are
keyed by the ranges the charts replay (a big blind defending the HJ by the solve would orphan srp_hj_bb's 1,755
entries).
