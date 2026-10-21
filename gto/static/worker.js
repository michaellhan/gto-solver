// Runs the Python solver in the browser with Pyodide. Used when no local server is present.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

// no __init__ here: GitHub Pages (Jekyll) will not serve files starting with an underscore
const FILES = ["cards", "ranges", "evaluator", "preflop", "pf_equity", "pf_classes", "pf_solve", "pf_game", "pf_game_data", "pf_eqr", "solver", "multiway_solver", "exploit", "advisor", "flop_library"];

// The precomputed flop library (gto/flop_library.py): the Python side asks for its index once and for an entry
// (<spot>/<flop>.bin, ~10 KB) only when a flop spot it holds comes up. A synchronous request (allowed in a
// worker) because the solver runs synchronously; a missing or unreadable file is just no library.
self.libFetch = (path) => {
  try {
    const x = new XMLHttpRequest();
    x.open("GET", `flop_library/${path}`, false);
    x.responseType = "arraybuffer";
    x.send();
    return x.status === 200 ? new Uint8Array(x.response) : null;
  } catch (e) {
    return null;
  }
};

const ready = (async () => {
  postMessage({ status: "Loading the solver runtime (about 15 MB, cached after the first visit)…" });
  const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });
  postMessage({ status: "Loading numpy…" });
  await py.loadPackage(["numpy", "micropip"]);
  postMessage({ status: "Installing the hand evaluator…" });
  await py.pyimport("micropip").install("treys");
  postMessage({ status: "Loading solver code…" });
  py.FS.mkdir("/app"); py.FS.mkdir("/app/gto");
  py.FS.writeFile("/app/gto/__init__.py", "");
  for (const f of FILES) {
    const r = await fetch(`gto/${f}.py`, { cache: "no-store" });
    if (!r.ok) throw new Error(`could not load gto/${f}.py (${r.status})`);
    py.FS.writeFile(`/app/gto/${f}.py`, await r.text());
  }
  // Pyodide numpy is 10-30x slower per iteration than native on the flop (its leaf products have no BLAS), ~1.5x on the
  // turn (vectorised river showdowns), so the browser plays a smaller game. Two flop bet sizes; inside a flop solve one
  // turn and river size (Config.later_sizes).
  // From the turn (scripts/turn_river_reference.py: 6 turn spots against every-river solves run to convergence with a
  // 33/75/125% menu; README Accuracy vs speed), the river is one card per class (river_classes: 13 cards on a rainbow
  // turn, 22-24 with a flush suit, each standing for its cards through suit swaps; within ~0.1% of the pot of every
  // river at the same iterations, 2-3x cheaper). The old Fast's 8 fixed rivers settled (~450 iterations) into a game
  // where only those rivers come: per-combo EVs 5-10% of the pot from its own tree with every river.
  // Fast (8 s): a third and 125% pot on the turn, the river inside at 75%, ~90-150 iterations: per-combo EVs 0.5-1.5%
  //   of the pot from the rich reference (the old 8 rivers: 3.8-9.5; 75% alone, converged: 0.6-3.8).
  // Accurate (60 s): a third, three quarters and 125% on the turn and river, ~180-360 iterations: 0.08-0.33% (one size
  //   with every river: 0.6-3.8).
  // A river root bets 33/75/125/250% pot (Advisor.root_sizes): 0.1-0.6% of the pot per combo from a six-size menu,
  //   where 75% alone was 1.2-7.2 (polar rivers overbet half the range).
  // From the flop both play one turn per rank class with the river bet on, the turn and river by hand buckets
  //   (river_buckets, turn_buckets; README, Accuracy vs speed). Fast: 32 buckets over every third river, 130-420
  //   iterations in 8 s (scripts/bench_browser.mjs). Accurate: 64 over every river, 520-870 iterations in 60 s
  //   (0.03-0.05% of the pot from equilibrium in its game). Against a solve over every turn (12 flops, scripts/bench_solver.py --flop) at 400 iterations
  //   the c-bet is 2.5 / 1.8 points off on average, the size split 4.1 / 2.3, BB's check 1.6 / 1.3. Accurate played
  //   the turn and river per combo until measured here: 54 iterations in 60 s on a rainbow flop (7.5 points of
  //   c-bet, 12.7 of split, 8.5 of check), ~25-30 on a two-tone one (at 25: 14.4, 18.4, 18.4) where, still 10% of
  //   the pot from equilibrium, it took a second budget (132 s for the page's 60, a 590 MB heap). Its earlier-street solves
  //   (12 s, for range narrowing) play its own flop model, so a hand's flop advice narrows its turn without a solve.
  const RS = "[setattr(a, 'root_sizes', {'river': {'bet': [0.33, 0.75, 1.25, 2.5]}}) for a in advs.values()]; ";
  const RAISES = "raise_sizes={'flop': [0.5], 'turn': [0.75], 'river': [0.75]}";
  py.runPython("import sys, json; sys.path.insert(0, '/app'); from gto.advisor import Advisor; from gto.solver import Config; "
             + `advs = {'fast': Advisor(Config(n_turns=3, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4, `
             + `river_buckets=32, turn_buckets=True, bucket_river_step=3, river_classes=True, `
             + `bet_sizes={'flop': [0.33, 0.75], 'turn': [0.33, 1.25], 'river': [0.75]}, ${RAISES})), `
             + `'accurate': Advisor(Config(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8, river_classes=True, `
             + `river_buckets=64, turn_buckets=True, bucket_river_step=1, `
             + `bet_sizes={'flop': [0.33, 0.75], 'turn': [0.33, 0.75, 1.25], 'river': [0.33, 0.75, 1.25]}, ${RAISES}))}; `
             + RS
             // Accurate picks up from a Fast solve of the same spot (not the reverse: Fast's few seconds can't re-fit an
             // Accurate solve to its own tree; the page shows Accurate's answer instead)
             + "advs['accurate'].warm_from = advs['fast']; "
             // Fast's turn also gets the narrowing flop's 3 s when that flop costs nothing (from the library or the hand's
             // flop advice; Advisor.spare_prior, as natively): the page's bar counts them anyway
             + "advs['fast'].spare_prior = True; "
             // both speeds answer a flop spot the library holds from it (Accurate's flop model, solved offline)
             + "from gto.flop_library import FlopLibrary; from js import libFetch; "
             + "lib = FlopLibrary(lambda p: (lambda r: None if r is None else r.to_bytes())(libFetch(p))); "
             + "advs['fast'].library = advs['accurate'].library = lib; "
             // 3-way pots (measured, scripts/bench_browser.mjs): the 3-player solver on the river in Fast (0.13-0.16 s an
             // iteration: 41-48 in 8 s, 0.3-0.4% of the pot from equilibrium); turns and flops only in Accurate's 60 s, the
             // later streets bet by 12 hand buckets with the turn's tables over every third river, the flop betting a third or
             // three quarters of the pot (numpy has no BLAS here: a flop iteration 0.34 s checked to, 161 in the 60 s, 0.27%;
             // 0.7-0.95 s first to act, 56-77, 3-6%; the 60 s cover the tables and the gains and EVs read off after). Turns
             // (0.18 s) and rivers stop once settled (advisor.MW_SETTLED): 200 iterations in 34 s, 100 in 15 s, 0.07%.
             + "advs['fast'].mw_solve_streets = ('river',); advs['accurate'].mw_solve_streets = ('flop', 'turn', 'river'); "
             + "advs['accurate'].mw_street_config = {s: {'later': 'buckets', 'buckets': 12, 'bucket_river_step': 3} for s in ('flop', 'turn')}; "
             + "advs['accurate'].mw_street_config['flop']['bet'] = {'flop': [0.33, 0.75], 'turn': 0.5, 'river': 0.5}");
  postMessage({ status: "Solving…" });
  return py;
})();

// A solve can only be cancelled before it starts (runPython blocks this thread); the page terminates
// the worker to stop one that is running.
const cancelled = new Set();
onmessage = async (e) => {
  if (e.data.cancel != null) { cancelled.add(e.data.cancel); return; }
  const { id, state, play } = e.data;   // play: {seat, hole}, a practice opponent's move (Advisor.play)
  try {
    const py = await ready;
    if (cancelled.delete(id)) return postMessage({ id, result: { cancelled: true } });
    py.globals.set("state_json", JSON.stringify(state));
    py.globals.set("play_json", JSON.stringify(play || null));
    const out = py.runPython("st = json.loads(state_json); pl = json.loads(play_json); adv = advs.get(st.get('speed'), advs['fast']); "
                           + "json.dumps(adv.play(st, pl['seat'], pl['hole'], pl.get('seed')) if pl else adv.advise(st))");
    postMessage({ id, result: JSON.parse(out) });
  } catch (err) {
    postMessage({ id, error: String(err).split("\n").slice(-3).join(" ") });
  }
};
