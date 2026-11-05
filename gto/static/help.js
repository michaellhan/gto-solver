// The help pages (index.html's ⋯ menu, Help; /start/, /about/, /glossary/, /changelog/ and /fairplay/): Getting
// started, About & accuracy, the Glossary, the Changelog and Fair play, one overlay with a tab per page. Loaded the
// first time one opens. Every number here comes from the README's measurements or the commit messages (git log);
// change them there first. The page provides window.viewSwitch(page) to move the URL when a tab is picked.
(function () {
  "use strict";
  const PAGES = [["start", "Getting started"], ["about", "About & accuracy"], ["glossary", "Glossary"], ["changelog", "Changelog"], ["fairplay", "Fair play"]];
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const to = (page, label) => `<button class="link" data-page="${page}">${label}</button>`;
  const term = (id, label) => `<button class="link" data-term="${id}">${label}</button>`;

  // ---- Getting started: what to do first, by what you came for
  function startHtml() {
    return `<p class="hp-lead">GTO Solver reviews no-limit hold'em hands you've already played: 2 to 9 players, preflop to
      river, cash games and tournaments. Enter a hand you played, finish it, and the solver grades each decision you made.
      It runs in your browser, and nothing you enter is uploaded. It is a tool for studying and for reviewing hands after
      you've played them, not for use during a hand (${to("fairplay", "Fair play")}).</p>
    <h3>Review a hand you played</h3>
    <ol>
      <li>Set the number of players and the stacks in the setup panel, then pick your seat and your two cards (click a
        card slot, or on a keyboard type its rank then its suit, e.g. A then s for the ace of spades).</li>
      <li>Enter the action seat by seat with the buttons under the table. Deal the flop, turn and river the same way.</li>
      <li>Your decisions are recorded as you enter them. Nothing is solved and no advice shows while a hand is being
        entered, whoever is to act.</li>
      <li>The hand is finished when it ends, at a fold or a showdown, or when you press <b>Finish hand</b> (a hand you
        don't know the end of). A decision you didn't record isn't reviewed.</li>
      <li><b>Review whole hand</b> solves and grades each decision you made. Click a decision to see the solver's
        ${term("mix", "mix")} for your hand, each action's ${term("ev", "EV")}, and your whole range in a grid. Once you've
        seen a review the hand stays as entered.</li>
    </ol>
    <h3>Settings worth knowing</h3>
    <ul>
      <li><b>Solve: Fast / Accurate</b> trades time for accuracy in a review (${to("about", "how accurate each one is")}).
        Fast is the default.</li>
      <li><b>Opponents: GTO / Pool</b>, on a finished hand. GTO assumes opponents who play perfectly; Pool assumes a typical
        online player pool's leaks and gives the most profitable play against them.</li>
      <li>Words in a review, like <i>converged</i>, <i>precomputed</i> or <i>EQR</i>, are explained in the
        ${to("glossary", "Glossary")}. Click a badge on an answer to jump to its entry.</li>
    </ul>`;
  }

  // ---- About & accuracy: the README's benchmarks in plain words
  function aboutHtml() {
    return `<p class="hp-lead">How the answers are made, how close they are to a full solver's, and what the tool doesn't do.
      Every figure here is measured, not estimated.</p>
    <h3>How close is "close"?</h3>
    <p>A solve is measured by its ${term("exploitability", "exploitability")}: how much a perfect opponent, who knew your
      whole strategy, could win against it, in percent of the pot. A solve run to the end has 0%. The badge on each
      answer shows it: <span class="hp-b ok">converged</span> under 3%, <span class="hp-b meh">fair</span> under 10%,
      <span class="hp-b bad">rough</span> above.</p>
    <p>A second measure is the EV a strategy <i>gives up</i>: we hand our strategy to another solver, let it find the best
      counter in the full game, and see how much worse our player does than at equilibrium.</p>
    <h3>Checked against an open-source solver</h3>
    <p>The answers were benchmarked against b-inary's <a href="https://github.com/b-inary/postflop-solver" target="_blank" rel="noopener">postflop-solver</a>
      (the open-source engine behind WASM Postflop and Desktop Postflop) on the same spots, ranges, pot, stacks and
      bet sizes: button vs big blind, cutoff vs big blind and the big blind's 3bet against the button, with that solver
      solving each game exactly (every turn and river card dealt) and then measuring ours. Medians, measured natively
      on a 10-core laptop (2026-10-29):</p>
    <div class="hp-scroll"><table class="hp-t">
      <tr><th>Spot, speed</th><th>From equilibrium</th><th>EV given up</th></tr>
      <tr><td>River, Fast or Accurate</td><td>0.01% of the pot</td><td>under 0.03% of the pot</td></tr>
      <tr><td>Single-raised turn, Fast (about 5 s)</td><td>0.93% (0.61-2.72)</td><td>0.19-0.25% (at most 0.50)</td></tr>
      <tr><td>Single-raised turn, Accurate (about 30 s)</td><td>0.14% (0.06-0.41)</td><td>0.01% (at most 0.08)</td></tr>
      <tr><td>3bet-pot turn, Fast or Accurate</td><td>0.01%</td><td>under 0.005%</td></tr>
      <tr><td>Precomputed flop</td><td>within 0.15% in its own game</td><td>0.2-0.5% against every card dealt</td></tr>
    </table></div>
    <p>On rivers the other solver measures our strategy exactly as we do, to four decimals. A precomputed flop gives up
      a little more because it plays the turn and river in a simplified game (fewer bet sizes, turn cards grouped into
      classes); solving two of them five times tighter gave the same numbers, so that is the cost of the
      simplification, not of stopping early. Speed per
      core: postflop-solver needs about as long as ours on a river, 1.4-1.9x as long on a single-raised turn, and gets
      there 2.2-2.9x sooner on 3bet-pot turns.</p>
    <h3>Precomputed or solved live</h3>
    <ul>
      <li><b>Preflop</b> comes from charts, written by hand and matched to published solves on 26 spots (0.96 percentage
        points apart on average).</li>
      <li><b>Flops</b> in common heads-up pots of 6-max 100bb cash (the button, cutoff, hijack, UTG and small blind
        against the big blind, five flatted opens and eight 3bet pots) come from a ${term("precomputed", "precomputed library")}
        and answer at once: 83.6% of heads-up flop decisions in that game. A 3-way library holds the first decisions of
        the commonest 3-way pots.</li>
      <li><b>Turns and rivers</b>, and flops the library doesn't hold, are solved live in your browser at the speed you
        pick. Where a near precomputed answer exists, an ${term("approximate", "approximate")} answer shows at once while
        the solve runs. Of all the decisions in a 6-max 100bb session, about 85% are answered from precomputed data and
        about 95% at once, counting the approximate ones.</li>
    </ul>
    <h3>Fast and Accurate</h3>
    <div class="hp-scroll"><table class="hp-t">
      <tr><th></th><th>Time per spot</th><th>What you get</th></tr>
      <tr><td><b>Fast</b> (default)</td><td>about 8 s in the browser</td><td>Rivers exact; single-raised turns give up about 0.2% of the pot; 3bet-pot turns near exact.</td></tr>
      <tr><td><b>Accurate</b></td><td>about 60 s in the browser</td><td>Turns within a fraction of a percent of a full solver (0.01% of the pot given up).</td></tr>
    </table></div>
    <p>The speeds are time budgets, not iteration counts: a slower or busier device runs fewer iterations in the same
      time and gets a rougher answer, and the badge says so. The timings above were measured on desktop computers.</p>
    <h3>Known limits</h3>
    <ul>
      <li><b>Bet sizes are a menu.</b> Each street is solved with a few sizes (the library's flops bet a third and three
        quarters of the pot). A precomputed flop facing a size it didn't store reads it off the nearest stored sizes,
        adjusted for the price.</li>
      <li><b>Multiway pots are simplified.</b> 3-way pots are solved with two flop sizes and one turn and river size,
        4-way pots in a smaller game, and flops and turns with 5 or more players use simpler betting rules, marked
        ${term("heuristic", "heuristic, not a solve")}.</li>
      <li><b>Other games are adjustments.</b> Preflop in other games (20-100bb stacks, antes, 7-9 handed tables, a
        straddle, heads-up) is the 100bb charts moved by solves of those games, on their main lines; under 20bb it is
        push/fold, and deeper than 100bb the 100bb charts re-priced. A straddle is solved only 6-handed. Rake is paid in
        the postflop solves, but preflop it re-prices calls by one rule fitted to a published raked solve.</li>
      <li><b>Tournaments</b> use ICM (Malmuth-Harville) with at most 12 players in the model, and no model of future
        hands.</li>
      <li><b>The pool</b> (Opponents: Pool) is fitted to published population data from online cash games, with some
        judgment calls where no data was found. It is not a read on any particular player.</li>
      <li><b>Every spot gets an answer, but not always a solve.</b> When the solver can't answer, the advice says it
        fell back on the chart, the betting rules or an equity estimate. When earlier play in the hand left the solved
        lines, the answer is marked ${term("offsolved", "off the solved line")}.</li>
    </ul>
    <h3>Your data</h3>
    <p>The solver runs in your browser (Pyodide, Python compiled to WebAssembly, loaded from jsDelivr), so your hands
      and settings stay on your device, in its local storage. <b>Delete everything this site keeps</b> in
      My hands deletes them and nothing else on this domain; <b>Back up everything</b> saves them to a file first (without
      your Anthropic key), and Restore reads it back, in this browser or another. The page is hosted on GitHub Pages.</p>`;
  }

  // ---- the Glossary: [id, term, definition]; index.html's badges open it at their entry (data-term)
  const TERMS = [
    ["gto", "GTO, equilibrium", "Game-theory optimal: a strategy for every player such that no one can win more by changing theirs alone. Playing it can't be exploited, but it isn't the most profitable play against an opponent who makes mistakes; that's what Opponents: Pool and the opponent tendencies (nodelocking) are for."],
    ["mix", "Mix, mixed strategy, frequency", "How often the solver takes each action with this hand, e.g. bet 60%, check 40%. A mix means the actions are worth about the same, so any action in the mix is fine. The grading gives ✓ to an action the solver takes at least 15% of the time, ~ at 3-15%, ✗ under 3%."],
    ["ev", "EV (expected value)", "The average a decision wins, in big blinds, over all the cards still to come and all the hands the opponents could hold. On the advice panel, EV is counted from now, relative to folding: fold is always 0, because the chips already in the pot are gone either way. A ≈ before an EV marks an action your range almost never takes, whose EV is rough (usually lower than exact)."],
    ["evlost", "EV lost", "How much less your action is worth than the solver's best one, in big blinds. Hand reviews add it up to show where your mistakes cost the most."],
    ["equity", "Equity", "Your share of the pot if the hand went to showdown with no more betting: how often you win, ties counted as half, against the opponent's range with the rest of the board dealt."],
    ["eqr", "EQR (equity realization)", "How much of your equity you actually turn into winnings: EV divided by equity times the pot. Above 1 your hand wins more than its share (it's easy to play, or in position); below 1 less (it often has to fold before showdown)."],
    ["range", "Range", "All the hands a player could have in this spot, given how they've played so far, each with its weight. The solver plays against a range, never against one known hand. The grid shows a range by hand class (pairs on the diagonal, suited hands above it, offsuit below)."],
    ["combo", "Combo", "One exact two-card holding, e.g. A♠K♥. A hand class like AKo has 12 combos, AKs 4 and a pair 6; cards on the board remove combos."],
    ["exploitability", "Exploitability, % of the pot, converged / fair / rough", "How far a solve is from equilibrium: what a perfect opponent who knew this strategy could win against it, in percent of the pot. A solve run to the end has 0%. The badge on an answer reads converged under 3%, fair under 10% and rough above; with a rough answer, Solve again on Accurate usually helps. Most answers are far under 3% (see About & accuracy)."],
    ["approximate", "Approximate", "An answer shown at once, read off a nearby solve (a precomputed spot, or the solve of the street before), while the solver works on this exact spot. The real solve replaces it when it finishes. A solve that ran far short of its speed's usual accuracy (on a busy device) is also marked approximate."],
    ["precomputed", "Precomputed", "Solved ahead of time, offline, and stored in the flop library, so the answer appears at once at any speed. Each stored flop is within 0.15% of the pot of its own game's equilibrium."],
    ["borrowed", "Precomputed · borrowed", "The library has no spot for exactly this pot (a different table size, ante or stack), so the answer is read off the stored spot whose ranges are closest. Close, but approximate: the tooltip says how far apart the ranges are."],
    ["offsolved", "Off the solved line", "Earlier in this hand someone took a line the solve almost never takes (or the line left very few hands), so their range from there comes from simpler betting rules rather than the solve. The answer is still the best play against those ranges, but not the whole hand's equilibrium. The line under the verdict says what left the solve."],
    ["heuristic", "Heuristic, not a solve", "With 5 or more players still in on the flop or turn (or 4 on some flops), the spot is too big to solve in the browser, so the answer comes from equity against the field and betting rules fitted to the solver's play. Grading there is by EV: within 3% of the pot of the best is ✓, within 10% is ~."],
    ["pool", "Pool, exploit", "Opponents: Pool assumes opponents with a typical online player pool's leaks (folding too often to bets, bluffing and raising less, playing more hands preflop), and the advice is then the most profitable play against them, not GTO. GTO's own frequency still shows under each action."],
    ["nodelock", "Nodelocking, opponent tendencies", "Setting how an opponent plays (folds more, bluffs less, and so on) under Advanced. The solver then plays the best counter to that player, with GTO's frequencies alongside."],
    ["icm", "ICM", "The Independent Chip Model: in a tournament it values a stack by its share of the prize money, not its chips, so losing a chip costs more than winning one gains. With payouts set, the advice is in prize equity."],
    ["speed", "Fast, Accurate", "How long the solver works on each spot in a review: about 8 s and 60 s in the browser. Longer means closer to equilibrium on the turn and flop; rivers are close to exact at both speeds. See About & accuracy for the measurements."],
    ["grades", "Grades", "Reviews mark each decision ✓ (the solver's mix takes it at least 15% of the time), ~ (3-15%) or ✗ (under 3%); a rough solve, or a line that gave up no EV, never grades ✗."],
    ["spr", "SPR (stack-to-pot ratio)", "The smaller stack behind divided by the pot. A low SPR (under about 4) means stacks go in easily; with a high one, hands play deeper and the turn and river matter more."],
    ["cbet", "C-bet, 3bet, 4bet, squeeze", "A c-bet (continuation bet) is a flop bet by the player who raised preflop. A 3bet is the first re-raise preflop (the open is the second bet), a 4bet the next one. A squeeze is a 3bet after an open and a call."],
  ];
  function glossaryHtml() {
    return `<p class="hp-lead">The words you meet in reviews.</p><dl class="hp-dl">`
      + TERMS.map(([id, t, d]) => `<dt id="hp-${id}">${esc(t)}</dt><dd>${esc(d)}</dd>`).join("") + `</dl>`;
  }

  // ---- the Changelog: what changed for users, newest first, from the commit history
  const CHANGES = [
    ["November 3", [
      "Explain with AI in a hand's review: under a decision's Details, Claude explains that decision from the solver's numbers, as in Practice. Nothing is sent until you click it.",
      "Review first: a hand you enter is recorded, and the solver reviews it once it's finished. Advice no longer shows while a hand is being entered, nothing solves in the background, and Quick, Randomise and Pool leave the entry screen (Pool and locks are there on a finished hand). Finish hand marks a hand finished if you don't know the end of it. Practice, Study, the drills and reports work as before; a report's flop or a drill's spot now opens in Study.",
      "A reviewed hand you were still in when the betting closed shows your equity against the ranges that took the line: your share of the pot at showdown, win and tie, and each opponent's range by hand.",
    ]],
    ["November 1 - 2", [
      "My hands: every hand you enter, review or import is kept in this browser (New hand no longer replaces it), with search, tags, notes, export and import, and a backup file of everything the site keeps.",
      "Help pages: Getting started, About & accuracy, this Glossary, the Changelog and Fair play, in the ⋯ menu; a first-visit chooser; plainer tooltips on the speeds, Pool, Copy hand and the answer badges, which now open their Glossary entry.",
      "The eight 3bet pots are precomputed on every flop, with their turns: 83.6% of heads-up flop decisions in 6-max 100bb now answer at once (77.2% before).",
      "Answers whose ranges left the solves earlier in the hand say so: the \"off the solved line\" badge, and a line saying why.",
      "A flop or turn solved live shows its solve so far while it works.",
      "More spots answer at once: 3-way pots in more formats, single-raised pots at other stack depths, and the turn after more library flops.",
    ]],
    ["October 26 - 31", [
      "More answers at once off the nearest precomputed spot: heads-up games, 8- and 9-max tables, antes and straddles.",
      "Study mode: browse the solver's strategy for a whole range, node by node, with every hand's mix, EV, equity and EQR; turn and river reports; filters by hand category; 3-way pots.",
      "Preflop solve: set the players, stacks, ante, straddle, rake and raise sizes, then browse every hand's mix and EVs at every preflop decision.",
      "Range builder: paint a range's strategy yourself and score it against the solver's.",
      "Tournaments: ICM from the payouts, bounty tournaments, and leak reports across sessions in the analyzer.",
      "Multi-table Practice: two to four tables at once.",
      "One bet size: the advice's Simple tab answers with a single size.",
      "A compiled solver core (WebAssembly) running on several threads: much faster solves in the browser, so Fast plays a bigger game.",
      "Multiway pots solved at every speed, 4-way flops and turns and side pots included.",
      "An approximate answer at once while the browser solves, and precomputed answers in about 0.01 s.",
      "The flop library grew: every flop for the small blind, hijack, UTG and cutoff against the big blind, five flatted opens, and the 3bet pots with their turns.",
      "Benchmarked against b-inary's open-source postflop-solver (see About & accuracy).",
    ]],
    ["October 19 - 25", [
      "Practice mode: random hands against GTO opponents, each decision graded, with a coach panel, a score and Explain with AI.",
      "Hand-history import from PokerStars, GGPoker, 888poker, partypoker, Winamax, WPN/ACR, Ignition/Bovada and iPoker, and Review whole hand.",
      "The session analyzer: a whole session's hands graded decision by decision, the EV lost by street, position and spot.",
      "The precomputed flop library, Drills and Flop reports.",
      "The Breakdown and Equity views of your range.",
      "3-way pots solved, a 3-way flop library, then 4-way and bigger rivers.",
      "Quick speed, per-seat stacks with side pots, custom bet sizes and ranges, opponent tendencies (nodelocking) preflop and after the flop.",
      "Each tool has its own URL; sound effects.",
    ]],
    ["October 12 - 18", [
      "Preflop charts matched to published solver frequencies.",
      "Optional rake, and rake priced into preflop calls.",
      "Opponents: Pool fitted to population data from online games.",
      "Phone layout: the action buttons stay at the bottom of the screen and rise above the keyboard.",
    ]],
    ["October 4 - 11", [
      "Multiway advice: bets, calls and checks priced against every player in the pot.",
      "Off-chart preflop lines (limps, squeezes, cold 3bets) and short-stack jams by push/fold equilibrium.",
      "Opponents: Pool, the most profitable play against a typical player pool.",
      "Switching speed reuses the work: Accurate continues from a Fast solve of the same spot.",
    ]],
    ["September 20 - 28", [
      "First release: a no-limit hold'em hand-review solver that runs in the browser.",
      "Fast and Accurate speeds; the convergence badge; action buttons sized and labelled with the solver's frequencies.",
      "Review: your EV against the solver's best for each decision, and the total EV lost.",
      "Copy hand, and the N shortcut for a new hand.",
    ]],
  ];
  function changelogHtml() {
    return `<p class="hp-lead">What's new, newest first.</p>`
      + CHANGES.map(([when, items]) => `<h3>${esc(when)}</h3><ul>${items.map(i => `<li>${esc(i)}</li>`).join("")}</ul>`).join("");
  }

  // ---- Fair play
  function fairplayHtml() {
    return `<p class="hp-lead">GTO Solver is a study and review tool. Use it away from the table: before a session to
      study, after it to review. Do not use it during a hand you are playing, online or live.</p>
    <ul>
      <li><b>Online:</b> almost every poker site bans real-time assistance (solvers, charts or advice tools consulted
        while you are in a hand) and can close your account and keep your balance for it. Their terms apply to you, not
        to this site.</li>
      <li><b>Live:</b> casinos and card rooms treat outside help during a hand as cheating too.</li>
      <li><b>By design</b> the tool reads only what you type or paste. There is no screen capture, table reading or
        connection to any poker client, and there won't be.</li>
      <li><b>A hand you enter gets no advice until it's finished.</b> The page records your decisions and reviews the
        ones you made after the hand is over. That is a limit on what the page offers, not a promise it can tell when or
        where you're playing, so the rule above is still yours to keep.</li>
    </ul>
    <p>If you are unsure what a site allows, read its terms on third-party tools; when in doubt, close the solver
      while you play.</p>`;
  }

  const BUILD = { start: startHtml, about: aboutHtml, glossary: glossaryHtml, changelog: changelogHtml, fairplay: fairplayHtml };
  const isPage = (p) => Object.prototype.hasOwnProperty.call(BUILD, p);
  function pageHtml(page) { return BUILD[isPage(page) ? page : "start"](); }
  function tabsHtml(page) {
    return `<div class="hp-tabs" role="tablist">` + PAGES.map(([p, t]) =>
      `<button role="tab" data-page="${p}" aria-selected="${p === page}" class="${p === page ? "active" : ""}">${esc(t)}</button>`).join("") + `</div>`;
  }

  const CSS = `
  .hp { position: fixed; inset: 0; z-index: 90; background: rgba(0,0,0,.55); display: flex; justify-content: center; align-items: flex-start; padding: 28px 16px 16px; }
  .hp-card { background: var(--panel2); color: var(--text); border: 1px solid var(--line2); border-radius: var(--r); width: min(820px, 100%); max-height: calc(100dvh - 44px); display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,.5); }
  .hp-card:focus { outline: none; }
  .hp-head { display: flex; align-items: center; gap: 10px; padding: 10px 12px 6px; }
  .hp-x { margin-left: auto; padding: 2px 8px; }
  .hp-tabs { display: flex; flex-wrap: wrap; gap: 4px; padding: 0 12px 8px; border-bottom: 1px solid var(--line); }
  .hp-tabs button { background: none; border: 1px solid transparent; padding: 5px 10px; font-size: 13px; color: var(--muted); }
  .hp-tabs button.active { color: var(--text); border-color: var(--line2); background: var(--panel); }
  .hp-body { padding: 4px 16px 18px; overflow: auto; line-height: 1.55; font-size: 14px; }
  .hp-body h3 { font-size: 14px; margin: 18px 0 6px; color: var(--accent); font-weight: 600; }
  .hp-body p, .hp-body li, .hp-body dd { color: #c9cdd3; }
  .hp-body ul, .hp-body ol { padding-left: 20px; margin: 6px 0; }
  .hp-body li { margin: 4px 0; }
  .hp-body a { color: var(--accent); }
  .hp-lead { color: var(--text) !important; margin-top: 12px; }
  .hp-body button.link { color: var(--accent); font-size: inherit; }
  .hp-scroll { overflow-x: auto; }
  .hp-t { border-collapse: collapse; font-size: 13px; margin: 6px 0; min-width: 480px; width: 100%; }
  .hp-t th, .hp-t td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
  .hp-t th { color: var(--muted); font-weight: 500; }
  .hp-b { font-size: 11px; padding: 1px 6px; border-radius: 999px; border: 1px solid currentColor; white-space: nowrap; }
  .hp-b.ok { color: var(--good); } .hp-b.meh { color: var(--meh); } .hp-b.bad { color: var(--bad); }
  .hp-dl dt { font-weight: 600; margin-top: 14px; scroll-margin-top: 8px; }
  .hp-dl dt.hp-hit { color: var(--accent); }
  .hp-dl dd { margin: 3px 0 0; }
  @media (max-width: 760px) {
    .hp { padding: 0; }
    .hp-card { max-height: 100dvh; height: 100dvh; border-radius: 0; border: 0; width: 100%; }
    .hp-tabs { flex-wrap: nowrap; overflow-x: auto; }
    .hp-tabs button { white-space: nowrap; }
  }`;

  let box = null, shown = null;
  function render(page) {
    shown = isPage(page) ? page : "start";
    box.querySelector(".hp-nav").innerHTML = tabsHtml(shown);
    const body = box.querySelector(".hp-body");
    body.innerHTML = pageHtml(shown); body.scrollTop = 0;
  }
  // the Glossary scrolled to a term (a badge clicked on the advice panel: globalThis.HELP_TERM, or a link here)
  function showTerm(id) {
    const dt = box && box.querySelector("#hp-" + id);
    if (!dt) return;
    for (const d of box.querySelectorAll(".hp-hit")) d.classList.remove("hp-hit");
    dt.classList.add("hp-hit");
    if (dt.scrollIntoView) dt.scrollIntoView({ block: "start" });
  }
  // a tab picked: its page, and the URL moved without closing (index.html's viewSwitch)
  function show(page, termId) {
    if (!box || box.hidden) return open(page, termId);
    if (page !== shown) { render(page); if (globalThis.viewSwitch) globalThis.viewSwitch(shown); }
    if (termId) showTerm(termId);
  }
  function open(page, termId) {
    if (!document.getElementById("hpStyle")) {
      const s = document.createElement("style"); s.id = "hpStyle"; s.textContent = CSS; document.head.appendChild(s);
    }
    if (!box) {
      box = document.createElement("div"); box.id = "helpBox"; box.className = "hp"; box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true"); box.setAttribute("aria-label", "Help");
      box.innerHTML = `<div class="hp-card" tabindex="-1"><div class="hp-head"><b>Help</b><button class="ghost hp-x" data-close aria-label="Close">✕</button></div><div class="hp-nav"></div><div class="hp-body"></div></div>`;
      document.body.appendChild(box);
      box.addEventListener("click", (e) => {
        if (e.target === box) return close();
        const b = e.target.closest("[data-close],[data-page],[data-term]");
        if (!b) return;
        if ("close" in b.dataset) return close();
        if (b.dataset.term) return show("glossary", b.dataset.term);
        show(b.dataset.page);
      });
      // the page's own shortcuts (N, card keys, action keys) don't see keys pressed in here
      box.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Escape") close();
        else if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && e.target.closest(".hp-tabs")) {
          const i = PAGES.findIndex(([p]) => p === shown), n = PAGES.length;
          show(PAGES[(i + (e.key === "ArrowRight" ? 1 : n - 1)) % n][0]);
          const t = box.querySelector(".hp-tabs .active"); if (t) t.focus();
        }
      });
    }
    box.hidden = false;
    render(page);
    const t = termId || globalThis.HELP_TERM; globalThis.HELP_TERM = null;
    if (t) showTerm(t);
    const f = box.querySelector(".hp-tabs .active"); if (f && !t) f.focus({ preventScroll: true });
  }
  function close() {
    if (!box || box.hidden) return;
    box.hidden = true;
    const m = document.getElementById("menuBtn"); if (m) m.focus();
    if (globalThis.viewClosed) globalThis.viewClosed(shown);
  }

  const api = { PAGES, TERMS, CHANGES, pageHtml, tabsHtml, startHtml, aboutHtml, glossaryHtml, changelogHtml, fairplayHtml,
                open, show, close, get shown() { return box && !box.hidden ? shown : null; } };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Help = api;
})();
