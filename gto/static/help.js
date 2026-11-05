// About (the header's About; /about/): an overlay, loaded the first time it opens. Every number here comes from the
// README's measurements or the commit messages (git log); change them there first.
(function () {
  "use strict";
  // what it is, how accurate, what stays on your device, where the rest went
  function aboutHtml() {
    return `<p class="hp-lead">GTO Solver reviews no-limit hold'em hands you've already played, 2 to 9 players, preflop
      to river. Enter a hand, finish it, and the solver shows what game theory does at each decision you made and what your
      play cost. Each of your turns is solved as you reach it while you enter the hand. Don't use it during a hand at a
      real-money table: every site's terms forbid it.</p>
    <h3>How accurate it is</h3>
    <p>A solve is measured by its exploitability: how much a perfect opponent who knew the whole strategy could win
      against it, in percent of the pot. Each answer's source line gives it: under 3% is close to a full solver, and above that
      the line says the solve hasn't settled. Answers were benchmarked against b-inary's open-source
      <a href="https://github.com/b-inary/postflop-solver" target="_blank" rel="noopener">postflop-solver</a>, which solved
      each game exactly and then measured ours (medians, on a 10-core laptop, 2026-10-29):</p>
    <div class="hp-scroll"><table class="hp-t">
      <tr><th>Spot</th><th>From equilibrium</th><th>EV given up</th></tr>
      <tr><td>River</td><td>0.01% of the pot</td><td>under 0.03% of the pot</td></tr>
      <tr><td>Single-raised turn</td><td>0.93% (0.61-2.72)</td><td>0.19-0.25% (at most 0.50)</td></tr>
      <tr><td>3bet-pot turn</td><td>0.01%</td><td>under 0.005%</td></tr>
      <tr><td>Precomputed flop</td><td>within 0.15% in its own game</td><td>0.2-0.5% against every card dealt</td></tr>
    </table></div>
    <p>In the browser a live flop takes about 8 s a decision, a turn 11 s and a river 14 s. That's a time budget, so a slower
      device gets a rougher answer, and its source line says so.</p>
    <ul>
      <li><b>Preflop</b> comes from charts, matched to published solves on 26 spots (0.96 percentage points apart on
        average), so it has no EV.</li>
      <li><b>Flops</b> in common heads-up pots of 6-max 100bb cash are precomputed and answer at once. Other flops,
        turns and rivers are solved live in your browser.</li>
      <li><b>Multiway pots are simplified.</b> 3-way and 4-way pots are solved in smaller games, and flops and turns
        with 5 or more players use simple rules, marked Betting rules, not a solve.</li>
      <li><b>Other games are adjustments.</b> Preflop at other stacks, with antes or a straddle or at 7-9 handed tables is
        the 100bb charts moved by solves of those games. Rake is paid in the postflop solves.</li>
      <li><b>Every spot gets an answer, but not always a solve.</b> When the solver can't answer, the advice says what it
        fell back on, and when earlier play left the solved lines it says that too.</li>
    </ul>
    <p>A review marks each decision ✓ (the solver's mix takes it at least 15% of the time), ~ (3-15%) or ✗ (under 3%); a rough solve, or a line that gave up no EV, never grades ✗.
      Without a solve it goes by EV: within 3% of the pot of the best is ✓, within 10% is ~.</p>
    <h3>What stays on your device</h3>
    <p>The solver runs in your browser (Pyodide, Python compiled to WebAssembly, loaded from jsDelivr), so your hands and
      settings stay on your device, in its local storage, and nothing you enter is uploaded. A Share link carries the
      hand inside the link, so whoever you send it to can see it. The page is hosted on GitHub Pages.</p>
    <p><b>Back up everything</b> in Hands saves what this site keeps to a file (without the Anthropic key, if you gave the
      classic copy one), and Restore reads it back, in this browser or another. <b>Delete everything this site keeps</b>
      deletes it and nothing else on this domain. The classic copy shares this storage.</p>
    <h3>Everything else</h3>
    <p>This page used to do much more. The last full version is frozen at the
      <a href="/gto-solver-classic/">classic copy</a> and no longer changes. Practice, study, reports and the rest carry
      on in <a href="/lizard/">Lizard</a>.</p>`;
  }

  const CSS = `
  .hp { position: fixed; inset: 0; z-index: 90; background: rgba(0,0,0,.55); display: flex; justify-content: center; align-items: flex-start; padding: 28px 16px 16px; }
  .hp-card { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: var(--r); width: min(820px, 100%); max-height: calc(100dvh - 44px); display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,.5); }
  .hp-card:focus { outline: none; }
  .hp-head { display: flex; align-items: center; gap: 10px; padding: 8px 8px 6px 16px; }
  .hp-x { margin-left: auto; padding: 2px 8px; }
  .hp-body { padding: 4px 16px 18px; overflow: auto; line-height: 1.55; font-size: 14px; }
  .hp-title { font-weight: 600; }
  .hp-body h3 { font-size: 14px; margin: 18px 0 6px; color: var(--text); font-weight: 600; }
  .hp-body p, .hp-body li { color: var(--text); }
  .hp-body ul { padding-left: 20px; margin: 6px 0; }
  .hp-body li { margin: 4px 0; }
  .hp-body a { color: var(--text); }
  .hp-lead { color: var(--text) !important; margin-top: 12px; }
  .hp-scroll { overflow-x: auto; }
  .hp-t { border-collapse: collapse; font-size: 13px; margin: 6px 0; width: 100%; }
  .hp-t th, .hp-t td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
  .hp-t th { color: var(--muted); font-weight: 400; font-size: 12px; }
  @media (max-width: 760px) {
    .hp { padding: 0; }
    .hp-card { max-height: 100dvh; height: 100dvh; border-radius: 0; border: 0; width: 100%; }
  }`;

  let box = null;
  function open() {
    if (!document.getElementById("hpStyle")) {
      const s = document.createElement("style"); s.id = "hpStyle"; s.textContent = CSS; document.head.appendChild(s);
    }
    if (!box) {
      box = document.createElement("div"); box.id = "helpBox"; box.className = "hp"; box.setAttribute("role", "dialog");
      box.setAttribute("aria-modal", "true");
      box.innerHTML = `<div class="hp-card" tabindex="-1"><div class="hp-head"><b class="hp-title"></b><button class="ghost hp-x" data-close aria-label="Close">✕</button></div><div class="hp-body"></div></div>`;
      document.body.appendChild(box);
      box.addEventListener("click", (e) => { if (e.target === box || e.target.closest("[data-close]")) close(); });
      // the page's own shortcuts (N, card keys, action keys) don't see keys pressed in here
      box.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Escape") close();
      });
    }
    box.hidden = false;
    box.querySelector(".hp-title").textContent = "About";
    box.setAttribute("aria-label", "About");
    const body = box.querySelector(".hp-body");
    body.innerHTML = aboutHtml(); body.scrollTop = 0;
    box.querySelector(".hp-card").focus({ preventScroll: true });
  }
  function close() {
    if (!box || box.hidden) return;
    box.hidden = true;
    const a = document.getElementById("helpAbout"); if (a) a.focus();
    if (globalThis.viewClosed) globalThis.viewClosed("about");
  }

  const api = { aboutHtml, open, close };
  if (typeof module === "object" && module.exports) module.exports = api; else globalThis.Help = api;
})();
