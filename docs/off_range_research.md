# A hand outside its range: does the near-zero weight matter?

The Advisor solves a hand its range doesn't hold with that hand's row at a near-zero weight (`Advisor._with_hero`, 1e-3 of
a combo; multiway `MW_OFF_RANGE`, 1e-10). The hand that asked: 7-handed, UTG limps, LJ raises to 4, HJ, BB and UTG call, BB
bets As4s on 2s4d6c, LJ calls, the 6h turn. BB's chart has no call with A2s-A8s here, and the answer is bet 75% at 100%. Does
that come from the weight? And should the hand go back in its range (idea 8)?

**Short answer.** Near-zero weight is not a real source of error in what to do: the mix it gives costs what the noise floor
costs (0.12-0.23% of the pot against 0.11), and with ranges of 60 combos or more it can't be told from the floor. It is a
real source of error in how far apart the actions look: each EV 2.4-2.9% of the pot from a real weight's, usually high, and
the gaps about 1% of the pot wide. The large misses on flops (15-21% of the pot) weren't the weight at all: they were Fast's
hand buckets, which gave a hand far stronger than a narrow range no value of its own. That's fixed: a flop with a range under
200 combos keeps a bucket for such a hand, and 128 strong off-range hands of 3bet and 4bet pots went from a mean regret of 2.10%
of the pot to 0.08 against per combo, 49 misses over 1% to 2, for no time.

Nothing was put back in the range: every way of doing that either gave your cards to the opponents or moved more than it
mended. `scripts/bench_off_range.py`; raw reads and every table in `docs/off_range/` (`report.txt`, `buckets.txt`, `hand.txt`,
`flop_alone.txt`, `browser.txt`).

## What was measured

78 spots: 8 heads-up pots (BTN v BB from each side, UTG v BB, limped, a 3bet pot from each side, a 4bet pot, 40 bb), 4 3-way
pots and the hand's own 4-way line, 2 boards each, on flop, turn and river. At each, 5 hands the preflop range doesn't hold,
one per hand type (strong, top pair, lower pair, draw, gutshot, air) while they last: heads-up first to act and facing a
bet, multiway one of the two. 630 hand-decisions. Fast's tree and models, 1 thread, the library off (a real weight can't
read it), and the same solve for every arm: heads-up stopped at 0.5% of the pot, 300 iterations in practice.

| arm | what changes |
|---|---|
| near | the hand at near-zero: what the Advisor does |
| combo | the hand at 1 combo, on the street answered |
| class | its whole class at 0.25 a combo, on the street answered |
| history | the class and its neighbours at 0.25 a combo from before the flop, through every earlier street: idea 8 as written |
| control | the combo solve of another hand, this hand still near-zero: what a small change of range does to an answer that isn't about it, so the floor under everything else |
| tiny | the hand at 0.05 of a combo |
| wide | a band of near-bar hands at 0.25 a combo from before the flop, whatever you hold: 20% more combos |

*Regret* is the near-zero mix read at the other arm's EVs, % of the pot. *Back* is the other way round: the real weight's mix
read at near-zero's EVs. Neither arm is the truth: near-zero is the game where opponents don't expect the hand, the real
weight the game where they do, and a real player sits between. So both directions are given.

## Results

Means, from before the fix below: `report.txt`'s flops of 3bet and 4bet pots are the old buckets. The interval resamples spots. The median regret is 0.00 in every row. History ran on the 1st, 3rd and 5th hand of a
spot, so its 378 have no top pairs.

| against | top action changes | and costs > 0.5% | regret | [90%] | back | per-action EV off | p90 | rest of hero's range moves | next player moves |
|---|---|---|---|---|---|---|---|---|---|
| control | 6% | 2% | 0.11 | 0.06-0.18 | 0.10 | 1.02 | 4.4 | 4.1 | 4.0 |
| combo | 16% | 2% | 0.12 | 0.04-0.23 | 0.46 | 2.55 | 9.5 | 4.1 | 4.0 |
| class | 13% | 4% | 0.20 | 0.12-0.28 | 0.30 | 2.37 | 9.4 | 4.9 | 4.4 |
| history | 13% | 5% | 0.23 | 0.13-0.34 | 0.36 | 2.93 | 11.5 | 7.2 | 6.4 |

Regret by the size of hero's range at the decision:

| combos | reads | control | combo | class | history |
|---|---|---|---|---|---|
| 60 or more | 395 | 0.07 | 0.04 | 0.07 | 0.08 |
| 20-60 | 105 | 0.06 | 0.47 | 0.49 | 0.58 |
| under 20 | 130 | 0.31 | 0.10 | 0.35 | 0.43 |

By street (regret, combo / class): heads-up flop 0.32 / 0.28, turn 0.02 / 0.03, river 0.07 / 0.16; 3-way flop 0.03 / 0.05,
turn 0.09 / 0.54, river 0.11 / 0.19; 4-way (10 reads each) flop 0 / 0, turn 0.02 / 1.13, river 0.28 / 0.52. Against the
typical player (`"opponents": "pool"`, what the page sends for cash; 126 river and turn hand-decisions of 26 spots): control
0.11, combo 0.16, class 0.22, EVs 1.7-2.0 off against 0.9: the same shape.

**The decision is fine.** The top action changes two or three times as often as the floor, and nearly always between actions
that tie: only 2-5% of reads change it at a cost, 2% at the floor. That's what CFR says should happen: a hand's value doesn't
depend on its own weight, and the near-zero mix is a best response where the real weight's is a mix over ties. The answer's
"100%" is the artefact, not the action. Without the flop, near-zero's regret against 1 combo is 0.06.

**The EVs aren't.** Each action's EV is 2.4-2.9% of the pot from a real weight's, 2.5x the floor, 10% at the 90th percentile.
Near-zero reads high (by 1.4-1.6% on average; its best action's EV by 1.8-2.7%, the floor 0.09), since opponents stop paying
a hand once they expect it. The gaps between actions read wide: +1.35, +0.89, +0.57 of the pot against combo, class, history.
Of 2,135 graded actions on the page's lines (3% and 10% of the pot) 87% get the same grade, 10-12% a harsher one, 2-3% a kinder.

**A lead of over 3% holds up.** The Close line on `claude/close-decisions` calls an off-range hand's actions close within 3%
of the pot. When near-zero's top EV leads by more (258 reads) a real weight has the same top on 91-94%; what's left costs a mean
of 0.15, and within 3% (372 reads) 0.10-0.20. Nothing here says move the line.

## The tail: buckets, fixed, then real changes of game

Regret's mean sat in a few reads. Three reads of a 4bet pot were 90% of combo's flop regret and nearly all of the 20-60 row
(0.47; 0.03 without the flop). They're all trips or two pair against their own range of about 23 combos:

| 6 flops, Fast | near-zero, the range's buckets | near-zero, its own bucket | 1 combo, buckets | near-zero, per combo | 1 combo, per combo |
|---|---|---|---|---|---|
| 4bet, 8s8d3c, Ad8h | all-in; check 65.7, all-in 75.2 | bet 33%; check 83.2, bet 84.8 | check 29 bet 33% 71; 80.9 | bet 33%; check 83.0, bet 84.4 | check 22 bet 33% 78; 79.8 |
| 4bet, 8s8d3c, 8h3h | all-in; check 62.2, all-in 74.7 | bet 33%; check 84.3, bet 84.6 | check 49 bet 33% 51; 82.9 | bet 33%; check 84.5, bet 85.3 | check 34 bet 33% 66; 82.8 |
| 4bet, Ks7s2h, 7c2c, facing | all-in 48.0; call 47.2 | call 59.5 | call 58.0 | call 61.0 | call 59.7 |
| 3bet, Jh9h4c, Jd4d, facing | call 36.5 | call 40.0 | call 35.8 | call 40.1 | call 39.6 |
| BTN v BB, Ks7s2h, 7c2c | check 7.75 | (not in the rule) | check 7.56 | check 6.98 | check 6.95 |
| BTN v BB, 8s8d3c, Ah8h | tied, 13.3 | (not in the rule) | tied, 13.3 | tied, 12.3 | tied, 12.3 |

A flop solve at Fast plays the turn and river by 32 buckets, equal shares of each range *by weight* ordered by equity
(`Ctx.bucket_table`). A near-zero hand adds no weight, so it took the place its equity gave it and played that bucket's value:
the mean of the hands whose weight is there. Hold trips above all of them and every line with a later street was played as
those hands: check 65.7 where it is 83.0, 37% of the pot, and the shove looked best. At 1 combo of 24 the hand is a
bucket's worth of the range and the problem goes. So does near-zero's at the same weight, per combo: it has the top action a real
weight has on all six, and the gap to the real EV is the 0-5 bb a near-zero hand reads high by anywhere. It isn't a weight
problem and adding weight isn't the way to fix it. In ranges as wide as a single-raised pot's a bucket is many hands wide and
near-zero's place in it does no harm.

**Per combo by range size, tried and dropped.** Solving the flop per combo when a range is narrow keeps your cards out of it, as
it reads only sizes, and it costs the 3bet pots too much. Under 40 combos leaves 3bet pots missing by up to 12% of the pot. Under
200 is over budget in the browser: flop-3bp-ip 0.65 to 10.29 s of ~8 at a heap of 125 MB, 50 bb 0.3 to 4.33, 200 bb 0.71 to 2.81,
a turn with its flop 1.07 to 3.82 (`browser.txt`).

**A bucket of its own, what went in** (`Config.near_bucket`, `Ctx._near_row_bucket`). On a heads-up flop with a range under 200
combos (`Advisor._flop_model`), which is every 3bet pot and tighter and a single-raised pot with a cold caller (32-59 combos; a
raise called in the big blind is 248 or more on both sides), the last of the 32 buckets is kept for the rows under 0.01 of a combo: a hand outside its range. The range shares the other 31.
The kept bucket's row of the equity table is that hand's own showdown against the other's buckets, with the same rivers, so it
plays its own value and costs one more bucket's arithmetic.

It keeps the guarantee. The rule counts only rows of 0.01 or more, so holding such a hand can't change which model runs; the
range's 31 buckets are built without the row, so they're the same with it or without; and the table is the same built in C or
numpy (`TestANarrowFlopKeepsABucket`). Preflop's ranges have no row under 0.01 in any of the 11 pots, at equilibrium or against
the typical player (the smallest is 0.06), so the only such row is your hand. Several would share the bucket and play as their
mean, which is what `run` did with all of a spot's hands at near-zero until `--alone`.

Against per combo, Fast's tree, 300 iterations, native, off-range hands and hands in range on 6 boards (`narrow.jsonl.gz`,
`narrow_cold.jsonl.gz`; regret and EV distance % of the pot). Off range means the probes' strong type, overpair or better, in
the 3bet and 4bet pots, and what the probes found in the cold-called pots, where the caller's range leaves few strong hands out:

| | hands | regret, before | after | max, before | after | over 1%, before | after | |EV|, before | after |
|---|---|---|---|---|---|---|---|---|---|
| 11 3bet, 4bet pots, off range | 128 | 2.10 | 0.08 | 18.3 | 1.6 | 49 | 2 | 14.90 | 3.05 |
| ... in range | 132 | 0.07 | 0.09 | 1.4 | 2.7 | 3 | 3 | 3.55 | 3.39 |
| 4 cold-called pots, off range | 45 | 0.25 | 0.16 | 2.5 | 1.2 | 4 | 3 | 12.86 | 3.43 |
| ... in range | 48 | 0.02 | 0.02 | 0.35 | 0.32 | 0 | 0 | 2.12 | 2.29 |

Solve time 0.37 s before and after in the first two rows (per combo 1.14), 1.03 and 1.01 in the others (per combo 2.26),
exploitability unchanged. The cold-called pots' misses are mixes over near-ties (JTo on AhKdQc, 0.12 bb apart), and the buckets'
EVs were as far off there as in a 3bet pot. The worst pot for regret in range went from
0.30 to 0.53 (4bet CO v BTN), mainly AJo on Td8s6s, tied mixes per combo. A range with 31 buckets for 32 is a little coarser, and
that's the price.

In the browser, Fast, an off-range hand, nothing moves past the machine's noise: 0.51 to 0.51 s on flop-3bp, 0.76 to 0.77, 0.53 to
0.54, 50 bb 0.40 to 0.41, 200 bb 0.85 to 0.87, a turn with its flop 1.22 to 1.28 and 1.23 to 1.27, heap 72 MB, 300 iterations,
no errors (`browser.txt`).

The measurement's 60 hand-decisions on flops of 3bet and 4bet pots, one hand near-zero in each solve (`flop_alone.txt`):

| against | regret, before | after | max, before | after | over 1%, before | after |
|---|---|---|---|---|---|---|
| combo | 0.81 | 0.05 | 16.45 | 0.94 | 4 | 0 |
| class | 0.70 | 0.43 | 21.05 | 18.69 | 3 | 4 |

Class's 18.69 is the same miss on the class arm's side, not the answer's: 83s on 8s8d3c at 0.25 a combo is a row in the range's
buckets, so that arm shoves at 73.8 and checks at 69.6, where 1 combo reads check 82.9 and near-zero now bets a third at 84.6.
Without it class's regret is 0.12. Over all 630 hand-decisions that takes combo's regret from 0.12 to 0.05, and class's from 0.20
to 0.17 (0.14 without that read). On these flops an off-range answer also reads higher than it did, 6.6% of the pot over a
real weight at its best action where it was 4.0: its value was being held down by the bucket.

What the fix doesn't reach:

* A hand in its range at a small weight, on top of a narrow range. The class arm's read above: such a row is in the range's
  buckets. In-range hands of 0.2 a combo or more were measured (the table's 0.09); less than that wasn't.
* Multiway flops, whose later streets are buckets too. Their flop regrets were 0.03-0.05 and 0, with nothing there like a 4bet
  pot's 23 combos, so they were left alone.
* Heads-up flops with an SPR of 8 or more that have their ranges under 200 combos (a 3bet pot at 200 bb). Their river stays on
  river strength, which isn't kept per row, so the bucket carries the turn only.
* Pots with both ranges wide, by the rule. A kept bucket would read a little better there too (7.34 against 7.75, per combo
  6.98, on Ks7s2h above), and regret there is already under 0.1% of the pot.
* The flop library. It's for hands in range and doesn't change. An off-range hand gets a live solve, as it did.

The rest of the tail is a real change of game, on rivers and turns: a bluff raise
that works against a range that doesn't expect it. 97o on the river of 8s8d3c raises 100% at near-zero (+1.31) and folds at
class weight, 30 combos with the raise at -2.04 (9.95% of the pot; at 1 combo it still raises, +1.04); Q2o and 95o the same in
narrow multiway turns. Under 20 combos everything's fragile: the control has a 21% read of its own, a 1-combo range on a 3-way
river, which has nothing to do with the hand's weight.

## The hand

Turn, Fast, 48 combos against 89, 300 iterations, pot 29.5:

| | check | bet 33% | bet 75% | mix (check / 33 / 75) |
|---|---|---|---|---|
| near-zero | 6.12 | 6.51 | 6.77 | 0 / 0 / 100 |
| A4s at 1 combo | 5.28 | 5.29 | 5.27 | 83 / 7 / 11 |
| class at 0.25 | 5.76 | 5.74 | 5.74 | 83 / 4 / 14 |
| A2s-A6s at 0.25 from preflop | 6.19 | 6.27 | 6.28 | 26 / 42 / 32 |
| ... at 0.5 | 6.30 | 6.28 | 6.32 | 45 / 18 / 37 |
| ... at 1 | 6.34 | 6.27 | 6.34 | 89 / 0 / 11 |

The near-zero answer costs nothing at any real weight: the actions tie to 0.1 bb. What it gets wrong is calling a tie a 0.65 bb
decision (2.2% of the pot) at 100%. However the range is repaired the honest answer is "these tie", and the mix then depends on a
weight nothing pins down. Saying Close says that without choosing one.

## Idea 8: what was tried and why none of it went in

**Your hand and its neighbours, added when you hold it** (`history`. Neighbours: pairs a rank either way; else the same top
card, kicker within two. 0.25 a combo, added to `preflop_ranges`' result, so ahead of the flop). It can't keep the guarantee, by
class or by combo: "BB holds A4s, so BB's range holds A2s-A6s" is a range that depends on your cards, and everyone's strategies
and next-street ranges are read off it. `tests/test_your_cards.py` wouldn't have caught it: its hands were all in range, so the
rule passed untested where it would break. There's a test for that case now, and the guarantee stays as it was. Nor is the gain
there: near-zero's answer is at the floor against it, the rest of the range moves 7%, it misses the flop library (which matches
on ranges), and it trades one arbitrary answer for a weight's.

**A general softening of near-bar hands, whatever you hold** (`wide`). This one keeps the guarantee, and is the wrong trade. A
band worth 20% of the range moves the rest of hero's range 11.6% and the next player 10.9%; hands *in range*, whose answers were
fine, lose 0.2-0.4% of the pot in each type with enough reads (10-19% flip, 6-17% at a cost), more than the error it would mend;
the solve carries 1.29x the combos (1.52x at p90); and it misses the flop library, so an instant answer becomes a solve.

**A bigger near-zero** (`tiny`, 0.05 of a combo; river and turn). The closest call: against 1 combo, EV error 2.40 to 2.04, gap
error 1.92 to 1.44, regret 0.06 to 0.03; others move about 1%. It trims numbers that are at the floor, and it makes the
opponents' play depend on your exact hand by a visible amount.

**What went in.**

* A bucket kept for a hand outside its range on a narrow flop (`Config.near_bucket`, `Advisor._flop_model`, `NARROW_FLOP`), which
  was the one large error here, with `TestANarrowFlopKeepsABucket`.
* The note says what the answer is: the best play against players who don't expect the hand, EVs usually high, the actions about
  1% of the pot further apart than if the range held it.
* `TestAHandOffItsRangeMovesNoRange` holds the guarantee where the tests didn't look.
* The measurement.

The rest of idea 8's job is done better beside the answer than inside it: `claude/range-robustness` asks the same spot under a
wider range as a follow-up read, so "and if your range did hold this?" gets a real answer without the main solve's ranges moving,
and `claude/close-decisions` reads an off-range hand by its EVs and calls the ties.

## Something found on the way

Heads-up, an off-range hand's earlier-street solve isn't an in-range hand's: `_as_advised` puts the row in so the next street can
read that solve, and any added row runs it a little differently. At the same iterations (BTN v BB, Ks8d3c, BB check-calls a
third of the pot, the 2h): the BTN's turn range is 4-5 combos of 242 apart by L1 (2%) and BB's about 1 of 212, at a weight of
1e-3, 1e-6 or 2e-9 alike, for AA and 72o alike. It doesn't scale with the weight and doesn't depend on the hand: a different seed,
not your cards reaching the range. It was there before this work. Multiway there's no such path: the two ranges are the same
but for your row. The test holds the heads-up shift under 3%. Removing it would take the second solve `_as_advised` was written
to avoid (3.1 s of a limped river's 3.25 at Fast in the browser).
