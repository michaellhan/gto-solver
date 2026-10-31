// The browser's solver, timed: runs gto/static/worker.js itself (its Python setup, configs and request handler,
// so the benchmark can't drift from the page) under Pyodide in Node, one request at a time, and reports each
// request's wall time against the page's budget (index.html BUDGETS: the street's budget plus each earlier postflop
// street's narrowing budget, what the page's progress bar counts down), the iterations and exploitability the
// answer reports, the time outside the solve loops, and the wasm heap (peak; Pyodide's memory never shrinks).
//
// Pyodide 0.26.4 (the worker's) is installed in a directory of its own and named with --pyodide:
//
//   mkdir -p ~/pyo && (cd ~/pyo && npm install --ignore-scripts pyodide@0.26.4)
//   node scripts/bench_browser.mjs --pyodide ~/pyo [--spots flop-lib,turn] [--speeds fast,accurate] [--json out.json]
//   node scripts/bench_browser.mjs --list
//
// --worker FILE runs an edited copy of worker.js instead (a config to try before changing the page's).
// --session runs every spot in one worker, one after another, as a page asks them over a session. --idle S waits S
// seconds after the setup before the first request (the worker warms up while idle); --prefetch S sends the page's
// prefetch (the flop's library entries fetched ahead) S seconds before each request with a flop; BENCH_NET_MS gives
// each library file that latency, and each line counts the requests' own synchronous fetches (sync-fetches).
// Each spot runs in a fresh worker (a child process: no solve cache carried between spots, like a page load);
// --warm runs Fast then Accurate in the same worker, as when the page is switched to Accurate after a Fast answer
// (Accurate starts from Fast's solve; the page's bar then counts the Accurate budget less Fast's seconds).
// numpy is read from the pyodide package's directory; the solver's code from solver.zip, built here as deploy.sh
// builds it (scripts/build_bundle.py, with the repo's .venv Python or $PYTHON). Each spot's first line gives the
// worker's setup by stage, the worker's own timings (runtime, numpy, imports, advisors).
// --hole CARDS gives hero those cards in every spot (one out of your range for the line: the spots' comments).
// --one-size: after a spot's Fast advice, the Simple view's requests as the page sends them (simple.js: "one_size":
// "full", then each candidate size, one solve a request at Fast's budget), each timed, and the best size.
// --early asks as the page does for its own advice (early: an approximate answer first where one is in hand at once,
// Advisor.instant) and reports when it came (early_s) and what it said.
// --gap G: the opponent's move (the spot's last action, theirs) is entered G seconds after it became their turn, and the
// time to the answer is counted from it; --presolve runs the page's presolve meanwhile (Advisor.presolve: the
// worker solves the street from where the opponent is to act, stopped through the shared stop flag at G s, as the
// page does when the move is entered), else the worker idles through the gap as it does today.
// The ro-* spots are Study's turn and river reports (study.js runouts(): {"study": true, "runouts": true}, no hero, at
// the speed's budget) solved cold in a fresh worker, as Study's Solve the flop / Solve the turn does, timed against
// the speed's budget (the street's, a river report's plus the flop's); the solves stop well inside it, and Study's
// estimate (study.js RO_SECONDS) is the slowest of these, at 1 and 8 threads.
// Times are wall clock on this machine: a browser tab is single-threaded too, but other load moves them.
import { spawnSync, execFileSync } from "child_process";
import { Worker as Thread } from "worker_threads";
import os from "os";
import fs from "fs";
import path from "path";
import vm from "vm";
import { createRequire } from "module";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = path.join(ROOT, "gto", "static");

// the page's budgets in the browser (index.html BUDGETS.wasm): [street, each earlier postflop street]
// (BENCH_BUDGETS='{"fast":[1.5,0.25]}' tries others, BENCH_STATE='{"quick":true}' adds request fields, e.g. Practice's;
// children inherit both)
// BENCH_NET_MS: each library file's request takes this long (the CDN measured 130-670 ms an entry, 0.8-4.9 s the index)
const NET_MS = +(process.env.BENCH_NET_MS || 0);
// BENCH_LIB: a flop library folder whose files are served before gto/static/flop_library's (an index with entries and
// turn files built locally, scripts/build_flop_library.py --out)
const LIB_DIR = process.env.BENCH_LIB || "";
function libFile(url) {
  const rel = url.split("?")[0];
  const f = LIB_DIR && path.join(LIB_DIR, rel.replace(/^flop_library\//, ""));
  return f && fs.existsSync(f) ? f : path.join(STATIC, rel);
}
const BUDGETS = { fast: [8, 3], accurate: [60, 12], ...JSON.parse(process.env.BENCH_BUDGETS || "{}") };

const F = (pos) => ({ pos, action: "fold" }), C = (pos) => ({ pos, action: "call" }), X = (pos) => ({ pos, action: "check" });
const R = (pos, to) => ({ pos, action: "raise", to }), B = (pos, amount) => ({ pos, action: "bet", amount });
const SRP = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), C("BB")];
const MW = [F("UTG"), F("HJ"), R("CO", 2.5), C("BTN"), F("SB"), C("BB")];
const MW4 = [F("UTG"), R("HJ", 2.5), C("CO"), C("BTN"), F("SB"), C("BB")];
const CO_BB = [F("UTG"), F("HJ"), R("CO", 2.5), F("BTN"), F("SB"), C("BB")];
const LIMP = [F("UTG"), F("HJ"), F("CO"), F("BTN"), C("SB"), X("BB")];   // (a limped pot: not in the flop library, solved live)
const BB_3BET = [F("UTG"), F("HJ"), F("CO"), R("BTN", 2.5), F("SB"), R("BB", 10), C("BTN")];
const HJ_CO = [F("UTG"), R("HJ", 2.5), C("CO"), F("BTN"), F("SB"), F("BB")];   // (the flop library's on 552 flops)
const BB_3BET_HJ = [F("UTG"), R("HJ", 2.5), F("CO"), F("BTN"), F("SB"), R("BB", 10), C("HJ")];   // (no spot: 3bp_bb_co's)
const CO_3BET_HJ = [F("UTG"), R("HJ", 2.5), R("CO", 7.5), F("BTN"), F("SB"), F("BB"), C("HJ")];   // (3bp_co_hj: stored turns)
const BB_3BET_SB = [F("UTG"), F("HJ"), F("CO"), F("BTN"), R("SB", 3), R("BB", 10), C("SB")];   // (3bp_bb_sb: stored turns)
const base = (o) => ({ players: 6, stack: 100, preflop: SRP, flop: [], turn: [], river: [], ...o });
const SRP_FLOP = [X("BB"), B("BTN", 1.8), C("BB")];                 // a third-pot c-bet called
const SRP_TURN = [X("BB"), B("BTN", 6), C("BB")];                   // two thirds called
// name: request state (the page's payload without budget / prior_budget / speed)
const SPOTS = {
  "preflop": base({ hero: "CO", hole: "AhJd", board: "", preflop: [F("UTG"), F("HJ")] }),
  "preflop-mw": base({ hero: "BB", hole: "Th9h", board: "", preflop: [F("UTG"), F("HJ"), R("CO", 2.5), C("BTN"), F("SB")] }),  // priced 3-way
  "flop-lib": base({ hero: "BTN", hole: "AsQs", board: "Kc7d4h", flop: [X("BB")] }),          // in the flop library
  "flop-lib-co": base({ hero: "BB", hole: "AhQd", board: "Qh9c4d", preflop: CO_BB }),            // in it: CO vs BB, the root
  "flop-lib-3bp": base({ hero: "BB", hole: "AhQd", board: "Kc7d4h", preflop: BB_3BET }),        // a 3bet pot's root (the
                                                    // BB's check there is rare: the BTN's answer to it is solved live)
  "flop-3bp": base({ hero: "BB", hole: "AhQd", board: "Ks7d2c", preflop: BB_3BET }),           // not in it: solved
  "flop": base({ hero: "BTN", hole: "AsQs", board: "Ks7d2c", flop: [X("BB")] }),              // (in it since BTN vs BB has every flop)
  // a library spot's flop it doesn't store: solved live, --early read off the stored flops most like it (instant_from_near)
  "flop-near": base({ hero: "HJ", hole: "AhQd", board: "KhJd6c", preflop: HJ_CO }),
  "flop-near-ip": base({ hero: "CO", hole: "JhTh", board: "KhJd6c", preflop: HJ_CO, flop: [X("HJ")] }),
  // (facing a bet: the averaged mix takes flop_approx's learned correction, here fold 67% -> 87%)
  "flop-near-bet": base({ hero: "HJ", hole: "Ah5h", board: "KhJd6c", preflop: HJ_CO, flop: [X("HJ"), B("CO", 4.88)] }),
  "flop-near-3bp": base({ hero: "BB", hole: "AhKd", board: "Qh8d5c", preflop: BB_3BET }),
  "flop-near-other": base({ hero: "BB", hole: "AhKd", board: "KhJd5c", preflop: BB_3BET_HJ }),   // (read off 3bp_bb_co's KcJd5h)
  "flop-near-80bb": base({ stack: 80, hero: "BB", hole: "Qh9h", board: "KhJd6c" }),   // (another depth: srp_btn_bb's KcJd6h)
  // (another depth facing a bet: srp_hj_co's three stored flops most like it at 100bb, corrected)
  "flop-near-80bb-bet": base({ stack: 80, hero: "HJ", hole: "Ah5h", board: "KhJd6c", preflop: HJ_CO, flop: [X("HJ"), B("CO", 4.88)] }),
  "flop-near-80bb-3bp": base({ stack: 80, hero: "BB", hole: "AhKd", board: "KhJd6c", preflop: BB_3BET }),   // (3bp_bb_btn's
                                                    // three stored flops most like it)
  "flop-2tone": base({ hero: "BB", hole: "JcTc", board: "9h8h6c" }),                          // not in it, two-tone
  "turn-lib": base({ hero: "BB", hole: "QhJh", board: "Kc7d4h9s", flop: SRP_FLOP }),         // the flop narrowed by the library
  "turn": base({ hero: "BB", hole: "QhJh", board: "Ks7d2cTh", flop: SRP_FLOP }),             // (its flop in the library now)
  "turn-3bp": base({ hero: "BB", hole: "AhQd", board: "Ks7d2cTh", preflop: BB_3BET, flop: [X("BB"), B("BTN", 6.8), C("BB")] }),  // the flop solved to narrow
  // a limped pot asked street by street (--session flop-limp,turn-limp,river-limp: each street's narrowing reads the
  // solve the one before made, and its approximate answer, --early, is read off it)
  "flop-limp": base({ hero: "SB", hole: "QhJh", board: "Ks7d2c", preflop: LIMP }),
  "turn-limp": base({ hero: "SB", hole: "QhJh", board: "Ks7d2cTh", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")] }),
  // you in position, the opponent's move last (--gap / --presolve: the worker presolves the street while it's theirs);
  // --session flop-limp-ip,turn-limp-ip,river-limp-ip plays one hand street by street, flop,turn-ip one after a library flop
  "flop-limp-ip": base({ hero: "BB", hole: "Kc6c", board: "Ks7d2c", preflop: LIMP, flop: [X("SB")] }),
  "turn-limp-ip": base({ hero: "BB", hole: "Kc6c", board: "Ks7d2cTh", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")],
                         turn: [X("SB")] }),
  "turn-limp-ip-f": base({ hero: "BB", hole: "Kc6c", board: "Ks7d2cTh", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")],
                           turn: [B("SB", 3)] }),   // a 60% lead: off the presolve's tree (33/75/125%)
  "river-limp-ip": base({ hero: "BB", hole: "Kc6c", board: "Ks7d2cTh5s", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")],
                          turn: [X("SB"), B("BB", 3.75), C("SB")], river: [X("SB")] }),
  "turn-ip": base({ hero: "BTN", hole: "AsQs", board: "Ks7d2cTh", flop: SRP_FLOP, turn: [X("BB")] }),    // after a library flop
  // the flop checked through (a library flop's commonest line: its stored turn starts the turn's solve, the turn file
  // served with BENCH_LIB)
  "turn-x": base({ hero: "BB", hole: "QhJh", board: "Ks7d2cTh", flop: [X("BB"), X("BTN")] }),
  "turn-x-ip": base({ hero: "BTN", hole: "AsQs", board: "Ks7d2cTh", flop: [X("BB"), X("BTN")], turn: [X("BB")] }),
  // one hand street by street with your hand out of your range for its line (--hole: yours in every spot of the run):
  // --session flop-bb,turn,river --hole AhAd (the big blind 3bets aces rather than calls), flop-limp-ip,turn-limp-ip,
  // river-limp-ip --hole KhQh, flop-3bp-ip,turn-3bp-ip,river-3bp-ip --hole AhKh, flop-40,turn-40,river-40 --hole AhAd;
  // turn,river --hole Jh9h calls the library flop's c-bet it folds
  "flop-bb": base({ hero: "BB", hole: "QhJh", board: "Ks7d2c" }),                              // first to act, in the library
  "river-3bp-ip": base({ hero: "BTN", hole: "AsQs", board: "Ks7d2cTh5s", preflop: BB_3BET, flop: [X("BB"), B("BTN", 6.8), C("BB")],
                         turn: [X("BB"), B("BTN", 15), C("BB")], river: [X("BB")] }),
  "flop-40": base({ stack: 40, hero: "BB", hole: "QhJh", board: "Ks7d2c" }),                   // 40bb: solved live
  "turn-40": base({ stack: 40, hero: "BB", hole: "QhJh", board: "Ks7d2cTh", flop: SRP_FLOP }),
  "river-40": base({ stack: 40, hero: "BB", hole: "QhJh", board: "Ks7d2cTh5s", flop: SRP_FLOP, turn: SRP_TURN }),
  "flop-3bp-ip": base({ hero: "BTN", hole: "AsQs", board: "Ks7d2c", preflop: BB_3BET, flop: [X("BB")] }),
  "turn-3bp-ip": base({ hero: "BTN", hole: "AsQs", board: "Ks7d2cTh", preflop: BB_3BET, flop: [X("BB"), B("BTN", 6.8), C("BB")],
                        turn: [X("BB")] }),
  "turn-limp-f": base({ hero: "SB", hole: "QhJh", board: "Ks7d2cTh", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")],
                        turn: [X("SB"), B("BB", 3)] }),   // facing 60%, off the tree turn-limp's advice solved
  "river-limp": base({ hero: "SB", hole: "QhJh", board: "Ks7d2cTh5s", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")],
                       turn: [X("SB"), B("BB", 3), C("SB")] }),
  "river": base({ hero: "BB", hole: "QhJh", board: "Ks7d2cTh5s", flop: SRP_FLOP, turn: SRP_TURN }),
  "mw-flop": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c", preflop: MW }),                // first to act
  "mw-flop-x": base({ hero: "BTN", hole: "AhJd", board: "Kh7d2c", preflop: MW, flop: [X("BB"), X("CO")] }),
  "mw-turn": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c9s", preflop: MW, flop: [X("BB"), X("CO"), X("BTN")] }),
  "mw-river": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c9s4h", preflop: MW, flop: [X("BB"), X("CO"), X("BTN")],
                     turn: [X("BB"), X("CO"), X("BTN")] }),
  // Study in a 3-way pot (no hero): the 3-way library's flop, first to act and checked to twice (at once), facing a
  // bet (solved live from there), the turn and river after checks (--session study-mw-flop,study-mw-flop-x,...: one
  // walk, each node narrowed through the solves Study answered from before it)
  "study-mw-flop": base({ board: "Kh7d2c", preflop: MW, study: true }),
  "study-mw-flop-x": base({ board: "Kh7d2c", preflop: MW, flop: [X("BB"), X("CO")], study: true }),
  "study-mw-flop-bet": base({ board: "Kh7d2c", preflop: MW, flop: [X("BB"), X("CO"), B("BTN", 2.6)], study: true }),
  "study-mw-turn": base({ board: "Kh7d2c9s", preflop: MW, flop: [X("BB"), X("CO"), X("BTN")], study: true }),
  "study-mw-river": base({ board: "Kh7d2c9s4h", preflop: MW, flop: [X("BB"), X("CO"), X("BTN")],
                           turn: [X("BB"), X("CO"), X("BTN")], study: true }),
  "mw4-flop": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c", preflop: MW4 }),               // 4-way, first to act
  "mw4-turn": base({ hero: "BB", hole: "Qh9h", board: "Kh7d2c9s", preflop: MW4, flop: [X("BB"), X("HJ"), X("CO"), X("BTN")] }),
  "exploit": base({ hero: "BB", hole: "QhJh", board: "Ks7d2cTh", flop: SRP_FLOP, opponents: "pool" }),
  "lock": base({ hero: "BB", hole: "QhJh", board: "Ks7d2cTh", flop: SRP_FLOP, locks: { BTN: { fold: 15, bluff: 30 } } }),
  "raked": base({ hero: "BTN", hole: "AsQs", board: "Kc7d4h", flop: [X("BB")], rake: 0.05, rake_cap: 3 }),
  // Study's reports (no hero): the turn report on a library flop (its entry holds the flop only: the flop solved for
  // it), checked through, a 3bet pot, a limped pot (no library); the river report (the turn solved, the flop to narrow)
  "ro-turn": base({ board: "Ks7d2c", flop: SRP_FLOP, study: true, runouts: true }),
  "ro-turn-x": base({ board: "Ks7d2c", flop: [X("BB"), X("BTN")], study: true, runouts: true }),
  "ro-turn-3bp": base({ board: "Ks7d2c", preflop: BB_3BET, flop: [X("BB"), B("BTN", 6.8), C("BB")], study: true, runouts: true }),
  "ro-turn-limp": base({ board: "Ks7d2c", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")], study: true, runouts: true }),
  "ro-turn-2tone": base({ board: "9h8h6c", flop: [X("BB"), B("BTN", 1.8), C("BB")], study: true, runouts: true }),
  "ro-river": base({ board: "Ks7d2cTh", flop: SRP_FLOP, turn: SRP_TURN, study: true, runouts: true }),
  "ro-river-x": base({ board: "Ks7d2cTh", flop: [X("BB"), X("BTN")], turn: [X("BB"), X("BTN")], study: true, runouts: true }),
  "ro-river-3bp": base({ board: "Ks7d2cTh", preflop: BB_3BET, flop: [X("BB"), B("BTN", 6.8), C("BB")], turn: [X("BB"), B("BTN", 15), C("BB")],
                         study: true, runouts: true }),
  // Study's own nodes (no hero): a library flop's first decision, then its turn after a called c-bet, first and in
  // position after a check (--early: answered at once from the flop's stored turn, Advisor.instant; --session
  // study-flop-3bp,study-turn-3bp with --prefetch S: the turn file fetched ahead once the flop's answer is in, as
  // study.js's studyPrefetch does)
  "study-flop-3bp": base({ board: "8c5d3h", preflop: CO_3BET_HJ, study: true }),
  "study-turn-3bp": base({ board: "8c5d3hAs", preflop: CO_3BET_HJ, flop: [X("HJ"), B("CO", 5.45), C("HJ")], study: true }),
  "study-turn-3bp-ip": base({ board: "8c5d3hAs", preflop: CO_3BET_HJ, flop: [X("HJ"), B("CO", 5.45), C("HJ")], turn: [X("HJ")], study: true }),
  "study-flop-3bp-sb": base({ board: "Kc3d2h", preflop: BB_3BET_SB, study: true }),
  "study-turn-3bp-sb": base({ board: "Kc3d2hQs", preflop: BB_3BET_SB, flop: [X("SB"), B("BB", 6.5), C("SB")], study: true }),
  "study-turn-3bp-sb-ip": base({ board: "Kc3d2hQs", preflop: BB_3BET_SB, flop: [X("SB"), B("BB", 6.5), C("SB")], turn: [X("SB")], study: true }),
  // Study's flop where the library doesn't store it (--early: read off the stored flops most like it, Advisor.instant;
  // --prefetch 0: those fetched ahead as the request is sent, as study.js does before a flop's first answer): HJ vs CO
  // (552 flops stored) first to act, and in position after a check on 8c7d5h (the opener checks all of it there);
  // BB 3bets the HJ (no spot: BB 3bet vs CO's flops); 80bb BTN vs BB (its 100bb entry of the flop)
  "study-flop-near": base({ board: "KhJd6c", preflop: HJ_CO, study: true }),
  "study-flop-near-x": base({ board: "8c7d5h", preflop: HJ_CO, flop: [X("HJ")], study: true }),
  "study-flop-near-3bp": base({ board: "KhJd6c", preflop: BB_3BET_HJ, study: true }),
  // (3bp_bb_co's own entry of KhJd5c, the HJ facing a third-pot bet: corrected)
  "study-flop-near-3bp-bet": base({ board: "KhJd5c", preflop: BB_3BET_HJ, flop: [B("BB", 6.77)], study: true }),
  "study-flop-near-80": base({ stack: 80, board: "Ks7d2c", study: true }),
  "study-flop-near-80-x": base({ stack: 80, board: "Ks7d2c", flop: [X("BB")], study: true }),
  // Study's turn and river after the street before was solved live (--early: read off that solve, Advisor.instant;
  // --session study-flop-limp,study-turn-limp,study-river-limp or study-flop-near-x,study-turn-near: the flop's
  // answer first, as Study asks; the -ip spots, asked straight after the flop's, are the in-position player's after a
  // check played on the answer at once, its solve stopped)
  "study-flop-limp": base({ board: "Ks7d2c", preflop: LIMP, study: true }),
  "study-turn-limp": base({ board: "Ks7d2cTh", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")], study: true }),
  "study-turn-limp-ip": base({ board: "Ks7d2cTh", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")], turn: [X("SB")], study: true }),
  "study-river-limp": base({ board: "Ks7d2cTh5s", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")],
                             turn: [X("SB"), B("BB", 3.75), C("SB")], study: true }),
  "study-river-limp-ip": base({ board: "Ks7d2cTh5s", preflop: LIMP, flop: [X("SB"), B("BB", 1.5), C("SB")],
                                turn: [X("SB"), B("BB", 3.75), C("SB")], river: [X("SB")], study: true }),
  "study-turn-near": base({ board: "8c7d5h2s", preflop: HJ_CO, flop: [X("HJ"), B("CO", 2.15), C("HJ")], study: true }),
  "study-turn-near-ip": base({ board: "8c7d5h2s", preflop: HJ_CO, flop: [X("HJ"), B("CO", 2.15), C("HJ")], turn: [X("HJ")], study: true }),
};
const STREET_IDX = { 0: 0, 3: 1, 4: 2, 5: 3 };
const STREET_OF = { 6: "flop", 8: "turn", 10: "river" };

function arg(name, dflt) {
  const i = process.argv.indexOf("--" + name);
  return i < 0 ? dflt : process.argv[i + 1];
}
const flag = (name) => process.argv.includes("--" + name);

// a Study node's answer in brief: its range's frequency per action
const mixOf = (r) => (r.actions || []).map((a) => `${a.action} ${Math.round(a.freq * 100)}%`).join(" / ");
// the page's estimate (index.html analyze(): the bar's "of ~N s"; a report: study.js estimate())
function pageBudget(state, speed, fastSeconds) {
  if (state.runouts) return state.board.length === 8 ? BUDGETS[speed][0] + BUDGETS[speed][1] : BUDGETS[speed][0];
  const street = STREET_IDX[state.board.length / 2];
  if (street === 0) return null;
  const [b, p] = BUDGETS[speed];
  let est = b + p * Math.max(0, street - 1);
  if (speed === "accurate" && fastSeconds) est = Math.round(Math.max(est * 0.2, est - fastSeconds));
  return est;
}

// worker.js in a sandbox with the browser APIs it uses, served from the repo (deploy.sh puts gto/*.py and the flop
// library next to the page)
function buildBundle() {
  const venv = path.join(ROOT, ".venv", "bin", "python");
  const python = process.env.PYTHON || (fs.existsSync(venv) ? venv : "python3");
  const out = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "gto-bundle-")), "solver.zip");
  execFileSync(python, [path.join(ROOT, "scripts", "build_bundle.py"), out]);
  return out;
}

async function startWorker(pyodideDir, bundlePath) {
  const require = createRequire(path.join(path.resolve(pyodideDir), "noop.js"));
  const { loadPyodide } = require("pyodide");
  let py = null, onMsg = null;
  const ctx = {
    console, setTimeout, clearTimeout, URL, URLSearchParams, TextDecoder, TextEncoder, Uint8Array, ArrayBuffer, JSON, Promise,
    Error, performance, WebAssembly, importScripts: () => {},
    loadPyodide: async () => (py = await loadPyodide()),   // the npm package's own files, not the CDN's
    fetch: async (url) => {
      if (/^https?:/.test(url)) return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };   // the numpy prefetch
      if (url.startsWith("flop_library/")) {   // the library's files fetched ahead (BENCH_NET_MS: a CDN's latency)
        ctx.aheadFetches++;
        await new Promise((r) => setTimeout(r, NET_MS));
      }
      const f = url.startsWith("solver.zip") ? bundlePath : url.startsWith("flop_library/") ? libFile(url)
        : path.join(ROOT, url.split("?")[0]);
      if (!fs.existsSync(f)) return { ok: false, status: 404 };
      const buf = fs.readFileSync(f);
      return { ok: true, status: 200, text: async () => buf.toString("utf8"),
               arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length) };
    },
    XMLHttpRequest: class {
      open(method, url) { this.url = url; }
      send() {
        const t = performance.now();
        while (performance.now() - t < NET_MS);   // (synchronous: the worker waits)
        const f = libFile(this.url);
        if (fs.existsSync(f)) { this.status = 200; const b = fs.readFileSync(f); this.response = b.buffer.slice(b.byteOffset, b.byteOffset + b.length); ctx.libFetches++; }
        else this.status = 404;
      }
    },
    libFetches: 0, aheadFetches: 0,
    postMessage: (m) => onMsg && onMsg(m),
  };
  // BENCH_THREADS N: a cross-origin isolated page on an N-core machine (the live site: sw.js's COOP/COEP), the worker's
  // helpers worker threads that run worker.js's own helper script (the Blob it makes, called as its onmessage)
  const NT = +(process.env.BENCH_THREADS || 0);
  if (NT > 0) {
    const blobs = new Map();
    Object.assign(ctx, {
      crossOriginIsolated: true, navigator: { hardwareConcurrency: NT },
      Blob: class { constructor(parts) { this.text = parts.join(""); } },
      URL: class extends URL { static createObjectURL(b) { const u = `blob:bench/${blobs.size}`; blobs.set(u, b.text); return u; } },
      Worker: class {
        constructor(url) { this.src = blobs.get(String(url)); }
        postMessage(d) {
          const t = this.t = new Thread(`const { workerData } = require("worker_threads");\n${this.src}\nonmessage({ data: workerData });`,
            { eval: true, workerData: d });
          t.on("error", () => {});   // (a helper's error goes nowhere in the browser either)
          t.unref();
        }
        terminate() { if (this.t) this.t.terminate(); }
      },
    });
  }
  ctx.self = ctx;
  vm.createContext(ctx);
  const src = fs.readFileSync(arg("worker", path.join(STATIC, "worker.js")), "utf8");
  const t0 = performance.now();
  // Python's `from js import ...` reads Node's global, not the sandbox's: the library's fetch and the cores the setup
  // instantiates are read through to the sandbox's from the start (the setup asks for the multiway core, gto/mwcore.py
  // lib(), and keeps the answer: set only after it, the bench ran every multiway solve in numpy)
  for (const k of ["libFetch", "gtoCore", "mwCore"]) Object.defineProperty(globalThis, k, { get: () => ctx[k], configurable: true });
  vm.runInContext(src, ctx, { filename: "worker.js" });
  // the setup has run once the worker reports it is solving
  let stages = null;
  await new Promise((res, rej) => {
    onMsg = (m) => { if (m.status === "Solving…") { stages = m.setup || null; res(); } if (m.error) rej(new Error(m.error)); };
  });
  const setup = (performance.now() - t0) / 1000;
  // BENCH_FAIL "i,n": the core's helper i fails holding the rows of its nth chunk (core_fail, a test's: what a trap in
  // a helper does), from the first request on; BENCH_MWFAIL the multiway core's (mw_fail: the task of its nth). Only the
  // instance the worker started: one started again after a failure (reload) runs clean.
  if (process.env.BENCH_FAIL && ctx.gtoCore && ctx.gtoCore.core_fail) ctx.gtoCore.core_fail(...process.env.BENCH_FAIL.split(",").map(Number));
  if (process.env.BENCH_MWFAIL && ctx.mwCore && ctx.mwCore.mw_fail) ctx.mwCore.mw_fail(...process.env.BENCH_MWFAIL.split(",").map(Number));
  const heap = () => py._module.HEAPU8.buffer.byteLength / 2 ** 20;
  // (with BENCH_FAIL or BENCH_MWFAIL: the times each core was started again, and whether it is still on)
  const restarts = () => py.runPython("from gto import core, mwcore\n"
    + "f'core {core._restarts}{\"\" if core.lib() else \" off\"}, mwcore {mwcore._restarts}{\"\" if mwcore.lib() else \" off\"}'");
  let id = 0;
  const prefetch = (state) => ctx.onmessage({ data: { prefetch: state } });
  // the page's stop flag, shared with the worker, and a thread that sets it on time while the worker's Python blocks
  // this one: cmd [seq, request id], at [when, Date.now() ms]; a newer command replaces one still waiting
  let stopAfter = null;
  if (flag("presolve")) {
    const stop = new SharedArrayBuffer(8), cmd = new Int32Array(new SharedArrayBuffer(8)), at = new Float64Array(new SharedArrayBuffer(8));
    ctx.onmessage({ data: { stop } });
    const t = new Thread(`const { workerData: d } = require("worker_threads");
      const cmd = new Int32Array(d.cmd), at = new Float64Array(d.at), stop = new Int32Array(d.stop);
      for (let seen = 0;;) {
        Atomics.wait(cmd, 0, seen);
        for (let newer = true; newer;) {
          seen = Atomics.load(cmd, 0); newer = false;
          const id = Atomics.load(cmd, 1), when = at[0];
          for (let left = when - Date.now(); left > 0; left = when - Date.now())
            if (Atomics.wait(cmd, 0, seen, left) !== "timed-out") { newer = true; break; }
          if (!newer) Atomics.store(stop, 1, id);
        }
      }`, { eval: true, workerData: { stop, cmd: cmd.buffer, at: at.buffer } });
    t.unref();
    stopAfter = (ms, id) => { at[0] = Date.now() + ms; Atomics.store(cmd, 1, id); Atomics.add(cmd, 0, 1); Atomics.notify(cmd, 0); };
  }
  // the page's presolve while `seat` is to act, stopped `ms` after it's posted (the move entered)
  const presolve = (state, seat, ms) => new Promise((res) => {
    const my = ++id, t0 = performance.now();
    onMsg = (m) => { if (m.id === my) res({ ...m.result, s: +((performance.now() - t0) / 1000).toFixed(2) }); };
    stopAfter(ms, my);
    ctx.onmessage({ data: { id: my, state, presolve: seat } });
  });
  const request = (state, early, from) => new Promise((res) => {   // (from: early_s counted from then)
    const my = ++id, t0 = from ?? performance.now();
    let part = null;
    onMsg = (m) => {
      if (m.id !== my) return;
      if (m.partial) part = { early_s: +((performance.now() - t0) / 1000).toFixed(3), ...m.partial };
      else res({ ...m, part });
    };
    ctx.onmessage({ data: early ? { id: my, state, early: true } : { id: my, state } });
  });
  return { setup, stages, heap, request, prefetch, presolve, restarts, ctx };
}

// one spot's requests in one worker (--session: every spot's, one after another in the same worker, as a page
// asks them over a session: the library's index fetched once, the solves kept); prints JSON lines
async function runOne(spotList, speeds, pyodideDir, bundlePath) {
  const w = await startWorker(pyodideDir, bundlePath);
  console.log(JSON.stringify({ spot: spotList.join(","), setup: +w.setup.toFixed(2), stages: w.stages, heap_mb: +w.heap().toFixed(0) }));
  // --idle S: the first request S seconds after the setup (the time a person takes to act: the worker warms up then)
  if (arg("idle")) await new Promise((r) => setTimeout(r, 1000 * +arg("idle")));
  for (const spot of spotList) await runSpot(w, spot, speeds);
}
async function runSpot(w, spot, speeds) {
  let fastSeconds = null;
  for (const speed of speeds) {
    const [budget, prior_budget] = BUDGETS[speed];
    const state = { ...SPOTS[spot], ...(arg("hole") ? { hole: arg("hole") } : {}), budget, prior_budget, speed,
                    ...JSON.parse(process.env.BENCH_STATE || "{}") };
    // --prefetch S: the page's prefetch as the flop is known, S seconds before the request (the time the hand takes
    // to reach your decision); --prefetch 0: sent with the request, counted in its time (Study's, before a flop's
    // first answer)
    const tp = performance.now();
    if (arg("prefetch") && state.board.length >= 6) {
      w.prefetch({ ...state, board: state.board.slice(0, 6), flop: [], turn: [], river: [] });
      await new Promise((r) => setTimeout(r, 1000 * +arg("prefetch")));
    }
    const [x0, a0] = [w.ctx.libFetches, w.ctx.aheadFetches];
    // --gap G: the opponent's move entered G s after it became theirs (--presolve: the worker presolving meanwhile)
    let t0 = arg("prefetch") === "0" ? tp : performance.now(), pre = null;
    const gap = arg("gap") == null ? null : +arg("gap");
    if (gap != null) {
      const st = STREET_OF[state.board.length], line = state[st] || [], last = line.at(-1);
      if (!last || last.pos === state.hero) throw new Error(`${spot}: --gap needs the opponent's move last on the ${st}`);
      if (flag("presolve")) pre = await w.presolve({ ...state, [st]: line.slice(0, -1) }, last.pos, Math.round(gap * 1000));
      const left = t0 + gap * 1000 - performance.now();
      if (left > 0) await new Promise((r) => setTimeout(r, left));
      t0 += gap * 1000;   // (the time to the answer counts from the move)
    }
    const m = await w.request(state, flag("early"), t0);
    const wall = (performance.now() - t0) / 1000;
    const r = m.result || {}, s = r.solver || {};
    const est = pageBudget(state, speed, fastSeconds);
    console.log(JSON.stringify({
      spot, speed, gap, presolve: pre, wall: +wall.toFixed(2), page_budget: est, over_pct: est ? Math.round((wall / est - 1) * 100) : null,
      iterations: s.iterations ?? null, solve_s: s.solve_seconds ?? null, expl_pct: s.exploitability_pct ?? null, cells: s.cells ?? null,
      outside_s: s.solve_seconds != null ? +(wall - s.solve_seconds).toFixed(2) : null,
      precomputed: !!r.precomputed, heap_mb: +w.heap().toFixed(0), lib_fetches: w.ctx.libFetches - x0, lib_ahead: w.ctx.aheadFetches - a0,
      street: r.street, recommended: r.recommended ?? (r.study === "node" ? mixOf(r) : r.study === "runouts" ? `${r.street} report: ${(r.first || {}).actor} ${((r.first || {}).actions || []).join("/")}, ${(r.cards || []).length} cards` : undefined),
      error: m.error || r.error || (r.skipped ? "skipped: " + r.skipped : null),
      restarts: process.env.BENCH_FAIL || process.env.BENCH_MWFAIL ? w.restarts() : null,
      early: m.part ? { s: m.part.early_s, recommended: m.part.recommended ?? mixOf(m.part), mix: mixOf(m.part),
                        source: m.part.approximate.source, label: m.part.approximate.label } : null,
      notes: (r.notes || []).filter((n) => process.env.BENCH_ALL_NOTES || /equilibrium|solved|iterations|rules/.test(n)).map((n) => n.slice(0, 160)),
    }));
    if (speed === "fast") fastSeconds = r.seconds;
    if (speed === "fast" && flag("one-size")) await runOneSize(w, spot, state);
  }
}
// the Simple view's requests after the advice (--one-size): the full menu first (it names the candidates and the order
// to solve them in: "order", the likely best first), then each
async function runOneSize(w, spot, state) {
  const seq = [], t0 = performance.now();
  let o = null;
  for (let label = "full"; label;) {
    const t = performance.now();
    const m = await w.request({ ...state, one_size: label });
    const r = m.result || {};
    if (!r.one_size) { seq.push({ size: label, error: r.error || m.error || "no answer" }); break; }
    o = r.one_size;
    const row = label === "full" ? o.full : o.rows.find((x) => x.size === label);
    seq.push({ size: label, wall: +((performance.now() - t) / 1000).toFixed(2), it: row.iterations, expl: row.exploitability_pct, ...(row.cut ? { cut: true } : {}) });
    label = (o.order || o.candidates).find((c) => !seq.some((x) => x.size === c)) || null;
  }
  const loss = o && o.rows.find((x) => x.size === o.best);
  console.log(JSON.stringify({ spot, one_size: seq, total: +((performance.now() - t0) / 1000).toFixed(1), best: o && o.best,
                               tied: o && o.tied, loss_pct: loss ? loss.loss_pct : null, heap_mb: +w.heap().toFixed(0) }));
}

if (flag("child")) {
  await runOne(arg("spot").split(","), arg("speeds").split(","), arg("pyodide"), arg("bundle"));
  process.exit(0);
}
if (flag("list")) {
  for (const [k, v] of Object.entries(SPOTS)) console.log(k.padEnd(12), JSON.stringify(v));
  process.exit(0);
}
const pyodideDir = arg("pyodide", process.env.PYODIDE_DIR);
if (!pyodideDir) { console.error("--pyodide DIR (where pyodide@0.26.4 is installed) is required"); process.exit(2); }
const spots = arg("spots", Object.keys(SPOTS).join(",")).split(",");
const speeds = arg("speeds", "fast,accurate").split(",");
const bundlePath = buildBundle();
const rows = [];
for (const spot of spots) if (!SPOTS[spot]) { console.error(`unknown spot ${spot} (--list)`); process.exit(2); }
for (const spot of flag("session") ? [spots.join(",")] : spots) {
  // a fresh worker per spot (and per speed unless --warm)
  for (const group of flag("warm") ? [speeds] : speeds.map((s) => [s])) {
    const p = spawnSync(process.execPath, [fileURLToPath(import.meta.url), "--child", "--spot", spot, "--speeds", group.join(","),
                                           "--pyodide", pyodideDir, "--bundle", bundlePath, ...(arg("worker") ? ["--worker", arg("worker")] : []), ...["prefetch", "idle", "gap", "hole"].flatMap((k) => (arg(k) ? ["--" + k, arg(k)] : [])), ...["early", "presolve", "one-size"].flatMap((k) => (flag(k) ? ["--" + k] : []))], { encoding: "utf8", maxBuffer: 1 << 26 });
    for (const line of p.stdout.split("\n").filter((l) => l.startsWith("{"))) {
      const r = JSON.parse(line);
      if (r.speed || r.one_size) rows.push(r);
      if (r.one_size) {
        console.log(`${r.spot.padEnd(11)} simple   ${String(r.total).padStart(6)} s  best ${r.best}${r.tied && r.tied.length ? " (tied " + r.tied.join(", ") + ")" : ""}`
                    + ` gives up ${r.loss_pct}% of the pot  heap ${r.heap_mb} MB\n             `
                    + r.one_size.map((x) => x.error ? `${x.size}: ERROR ${x.error}` : `${x.size} ${x.wall} s (${x.it} it, ${x.expl}%${x.cut ? ", cut" : ""})`).join(", "));
        continue;
      }
      console.log(r.speed ? `${r.spot.padEnd(11)} ${r.speed.padEnd(8)} ${String(r.wall).padStart(6)} s of ~${r.page_budget ?? "-"} ` +
                            `(${r.over_pct == null ? "-" : (r.over_pct > 0 ? "+" : "") + r.over_pct + "%"})  it ${r.iterations ?? "-"}  ` +
                            `expl ${r.expl_pct ?? "-"}%  solve ${r.solve_s ?? "-"} s  outside ${r.outside_s ?? "-"} s  heap ${r.heap_mb} MB` +
                            `${r.cells ? "  cells " + (r.cells / 1e6).toFixed(1) + "M" : ""}` +
                            `${r.precomputed ? "  library" : ""}${r.lib_fetches ? "  sync-fetches " + r.lib_fetches : ""}${r.error ? "  ERROR " + r.error : ""}` +
                            `${r.restarts ? "  restarts: " + r.restarts : ""}` +
                            (r.early ? `\n             approximate at ${r.early.s} s: ${r.early.recommended} [${r.early.mix}] (${r.early.label}); the answer: ${r.recommended}` : "") +
                            (r.gap != null ? `\n             the move ${r.gap} s in; ${r.presolve ? `presolve ${r.presolve.presolved ? "ran its budget, " + r.presolve.iterations + " iterations" : r.presolve.cancelled ? "stopped" : "skipped"} (${r.presolve.s} s)` : "no presolve"}; answered ${r.wall} s after it: ${r.recommended}` : "") +
                            (r.notes.length ? "\n             " + r.notes.join("\n             ") : "")
                          : `${r.spot}: worker setup ${r.setup} s${r.stages ? " (" + Object.entries(r.stages).filter(([k]) => k !== "total").map(([k, v]) => `${k} ${v}`).join(", ") + ")" : ""}, heap ${r.heap_mb} MB`);
    }
    if (p.status !== 0) console.error(p.stderr.split("\n").slice(-15).join("\n"));
  }
}
const out = arg("json");
if (out) fs.writeFileSync(out, JSON.stringify(rows, null, 1));
