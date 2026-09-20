// Runs the Python solver in the browser with Pyodide. Used when no local server is present.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const FILES = ["__init__", "cards", "ranges", "evaluator", "preflop", "solver", "advisor"];

const ready = (async () => {
  postMessage({ status: "Loading the solver runtime (about 15 MB, cached after the first visit)…" });
  const py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });
  postMessage({ status: "Loading numpy…" });
  await py.loadPackage(["numpy", "micropip"]);
  postMessage({ status: "Installing the hand evaluator…" });
  await py.pyimport("micropip").install("treys");
  postMessage({ status: "Loading solver code…" });
  py.FS.mkdir("/app"); py.FS.mkdir("/app/gto");
  for (const f of FILES) {
    const src = await (await fetch(`gto/${f}.py`, { cache: "no-store" })).text();
    py.FS.writeFile(`/app/gto/${f}.py`, src);
  }
  // lighter tree than the native server: Pyodide numpy is far slower, so trade abstraction for iterations
  py.runPython("import sys, json; sys.path.insert(0, '/app'); from gto.advisor import Advisor; from gto.solver import Config; "
             + "adv = Advisor(Config(n_turns=2, n_rivers=3, leaf_turns=4, leaf_rivers_flop=8), time_budget=8.0, prior_budget=2.5)");
  postMessage({ status: "Solving…" });
  return py;
})();

onmessage = async (e) => {
  const { id, state } = e.data;
  try {
    const py = await ready;
    py.globals.set("state_json", JSON.stringify(state));
    const out = py.runPython("json.dumps(adv.advise(json.loads(state_json)))");
    postMessage({ id, result: JSON.parse(out) });
  } catch (err) {
    postMessage({ id, error: String(err).split("\n").slice(-3).join(" ") });
  }
};
