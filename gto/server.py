"""Tiny HTTP endpoint for the sandbox table to push game state to.

    python -m gto.server --port 8765
    curl -X POST localhost:8765/advise -d @examples/flop.json

POST /advise  body: game state JSON (see advisor.py)  -> recommendation JSON
GET  /health
"""
from __future__ import annotations
import argparse
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .advisor import Advisor
from .solver import Config


class Handler(BaseHTTPRequestHandler):
    advisor: Advisor = None
    lock = threading.Lock()

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
        if p in ("/", "/index.html", "/worker.js"):
            name = "worker.js" if p == "/worker.js" else "index.html"
            path = os.path.join(os.path.dirname(__file__), "static", name)
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
        if not self.path.startswith("/advise"):
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            state = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as e:
            return self._send(400, {"error": f"bad json: {e}"})
        try:
            with self.lock:   # one solve at a time; the solver is CPU bound
                res = self.advisor.advise(state)
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
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args(argv)
    Handler.advisor = Advisor(Config(depth=args.depth), time_budget=args.budget)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"review UI:  http://{args.host}:{args.port}/")
    print(f"API:        POST http://{args.host}:{args.port}/advise")
    srv.serve_forever()


if __name__ == "__main__":
    main()
