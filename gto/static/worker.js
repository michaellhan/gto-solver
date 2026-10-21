// Runs the Python solver in the browser with Pyodide. Used when no local server is present.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

// no __init__ here: GitHub Pages (Jekyll) will not serve files starting with an underscore
const FILES = ["cards", "ranges", "evaluator", "preflop", "pf_equity", "pf_classes", "solver", "multiway_solver", "exploit", "advisor"];

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
  // Pyodide numpy is 10-30x slower per iteration than native, so the browser plays a smaller game:
  // two flop bet sizes and one turn/river bet and raise size. Measured against a long native solve
  // with the full size menu (BTN vs BB, 2c Jd 7s and Kh 7c 2d Ts runouts):
  // Fast (8 s): 8 fixed rivers under each turn. Sampling fresh rivers each iteration is unbiased but
  //   too noisy in the ~50 iterations 8 s affords (BB donked 20-37% where the answer is 0-2%); the
  //   fixed set was closest (10-15%). Its bias only dominates once a solve runs to convergence.
  //   From the flop: Accurate's flop model (the river bet on, one turn per rank class) with the turn and river
  //   by 32 hand buckets (river_buckets, turn_buckets): 190-260 iterations (30-42 ms each, 2.4x faster than 3
  //   random turns with the river checked down, which ran ~100 and bet 10-30 points too little on dry and paired
  //   flops); c-bets ~3.5 points from a solve over every turn (README, Accuracy vs speed).
  // Accurate (60 s): every river, so the turn is unbiased (donk 1-4%). From the flop, one turn per rank class
  //   with the river still bet on (Config's defaults): 60-75 iterations on a rainbow flop, ~30 on a two-tone
  //   or monotone one; c-bets within ~8 points of a solve over every turn there, against 12 for 4 random
  //   turns with the river checked down. A Fast flop solve plays another game there (buckets), so it seeds
  //   only turn and river solves. Its earlier-street solves (12 s, for range narrowing) play Fast's flop
  //   model, which settles there (~300 iterations; ~10 per combo).
  const LS = "bet_sizes={'flop': [0.33, 0.75], 'turn': [0.75], 'river': [0.75]}, raise_sizes={'flop': [0.5], 'turn': [0.75], 'river': [0.75]}";
  py.runPython("import sys, json; sys.path.insert(0, '/app'); from gto.advisor import Advisor; from gto.solver import Config; "
             + `advs = {'fast': Advisor(Config(n_turns=3, n_rivers=8, leaf_turns=4, leaf_rivers_flop=4, n_rivers_from_flop=4, `
             + `river_buckets=32, turn_buckets=True, bucket_river_step=3, ${LS})), `
             + `'accurate': Advisor(Config(n_turns=4, leaf_turns=4, leaf_rivers_flop=8, n_rivers_from_flop=8, ${LS}))}; `
             // Accurate picks up from a Fast solve of the same spot (not the reverse: Fast's few seconds can't re-fit an
             // Accurate solve to its own tree; the page shows Accurate's answer instead)
             + "advs['accurate'].warm_from = advs['fast']; advs['accurate'].prior_model = {'river_buckets': 32, 'turn_buckets': True, 'bucket_river_step': 3}; "
             // 3-way pots: the 3-player solver on the river in Fast (~0.1 s an iteration here: 8 s is ~60-80, under 1% of the pot
             // from equilibrium); turns and flops (every runout dealt out, ~0.5 s an iteration) only in Accurate's 60 s
             + "advs['fast'].mw_solve_streets = ('river',)");
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
