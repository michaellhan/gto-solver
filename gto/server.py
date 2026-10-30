"""Tiny HTTP endpoint for the sandbox table to push game state to.

    python -m gto.server --port 8765
    curl -X POST localhost:8765/advise -d @examples/flop.json

POST /advise  body: game state JSON (see advisor.py)  -> recommendation JSON; "speed": "fast" (default) or
              "accurate" picks the tree and default budgets (gto/native.py), "budget"/"prior_budget" override them
POST /play    body: {"state": game state, "seat": "BB", "hole": "QcQd", "seed": 7 (optional)}
              -> a GTO move for that seat holding those cards (practice opponents; see Advisor.play)
POST /cancel  stops the solve in progress (its /advise answers {"cancelled": true})
GET  /, /practice/, /reports/<spot>/ ... the page (gto/routes.py); /worker.js, /practice.js ... its scripts
GET  /health (also /health.json, which the page asks: the static site has a health.json saying {"ok": false})
"""
from __future__ import annotations
import argparse
import json
import os
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .native import advisors   # (first: it holds BLAS to one thread before numpy loads)
from .advisor import Advisor
from .solver import Cancelled
from . import core, routes


class Handler(BaseHTTPRequestHandler):
    advisors: dict = {}   # speed -> Advisor
    lock = threading.Lock()
    cancel = threading.Event()

    def _send(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, {"ok": True})
        p = self.path.split("?")[0]
        static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        lib = os.path.join(static, "flop_library")
        if p.startswith("/flop_library/") and p.endswith((".json", ".bin")):   # the library (reports.js, drills.js)
            path = os.path.realpath(os.path.join(lib, p[len("/flop_library/"):]))
            if not path.startswith(os.path.realpath(lib) + os.sep) or not os.path.isfile(path):
                return self._send(404, {"error": "not found"})
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if p.endswith(".json") else "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if p != "/" and routes.is_route_without_slash(p):   # "/practice" -> "/practice/", as the static host does
            q = self.path[len(p):]
            self.send_response(301)
            self.send_header("Location", p + "/" + q)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        page = routes.page_for(p) if p.endswith("/") or p.endswith("/index.html") else None
        if page is not None:   # the page, or a view's URL (/practice/, /reports/srp-btn-bb/: gto/routes.py)
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if p in ("/worker.js", "/hh.js", "/practice.js", "/reports.js", "/drills.js", "/breakdown.js", "/session.js",
                 "/study.js", "/runouts.js", "/sw.js"):
            name = p[1:]
            path = os.path.join(static, name)
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript" if name.endswith(".js") else "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/cancel"):
            self.cancel.set()
            return self._send(200, {"ok": True})
        play = self.path.startswith("/play")
        if not self.path.startswith("/advise") and not play:
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            state = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as e:
            return self._send(400, {"error": f"bad json: {e}"})
        try:
            with self.lock:   # one solve at a time; the solver is CPU bound
                self.cancel.clear()
                speed = (state.get("state") if play else state).get("speed")
                adv: Advisor = self.advisors.get(speed) or self.advisors["fast"]
                adv.should_stop = self.cancel.is_set
                res = (adv.play(state["state"], state["seat"], state["hole"], state.get("seed")) if play
                       else adv.advise(state))
        except Cancelled:
            return self._send(200, {"cancelled": True})
        except Exception as e:  # noqa: BLE001
            print("[server] advise failed for body:", json.dumps(state)[:2000])
            traceback.print_exc()
            return self._send(500, {"error": repr(e)})
        self._send(200, res)

    def log_message(self, fmt, *args):
        print("[server]", fmt % args)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--budget", type=float, default=None,
                    help="Fast's default seconds per solve (2.5; the page sends its own)")
    ap.add_argument("--prior-budget", type=float, default=None,
                    help="Fast's default seconds per earlier street used for range narrowing (2.5)")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--open", action="store_true", help="open the review page in the default browser")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1),
                    help="threads for the compiled core's solves (gto/core.py; the same answers on any number)")
    args = ap.parse_args(argv)
    core.threads(args.threads)
    Handler.advisors = advisors(args.depth, args.budget, args.prior_budget)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"review UI:  {url}")
    print(f"API:        POST {url}advise")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
