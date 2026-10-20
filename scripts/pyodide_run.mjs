// Runs a repo Python script under Pyodide in Node, as the browser's worker does (numpy without BLAS, one
// thread), for benchmarks such as scripts/bench_solver.py --curve. Pyodide is resolved from the current
// directory and fetches numpy and treys on first use:
//
//   mkdir -p /tmp/pyo && cd /tmp/pyo && npm install pyodide@0.26.4
//   node <repo>/scripts/pyodide_run.mjs scripts/bench_solver.py --curve --configs fast,accurate --budget 16
//
// The repo is mounted at /repo (the working directory); the script path is relative to it.
import { createRequire } from "module";
import path from "path";
import { fileURLToPath } from "url";

const require = createRequire(path.join(process.cwd(), "noop.js"));
const { loadPyodide } = require("pyodide");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [script, ...args] = process.argv.slice(2);
const py = await loadPyodide();
await py.loadPackage(["numpy", "micropip"]);
await py.pyimport("micropip").install("treys");
py.FS.mkdir("/repo");
py.FS.mount(py.FS.filesystems.NODEFS, { root }, "/repo");
py.globals.set("ARGV", py.toPy(["/repo/" + script, ...args]));
py.setStdout({ batched: (s) => console.log(s) });
py.setStderr({ batched: (s) => console.error(s) });
await py.runPythonAsync("import sys, os, runpy; sys.argv = list(ARGV); os.chdir('/repo'); runpy.run_path(sys.argv[0], run_name='__main__')");
