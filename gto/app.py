"""Desktop app: runs the solver server on a private local port and shows the review
page in a native window (pywebview). Entry point for the packaged builds.

    python -m gto.app            # open the window
    python -m gto.app --check    # start, verify the server answers, exit (used by builds)
"""
from __future__ import annotations
import json
import socket
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from .server import Handler
from .advisor import Advisor
from .solver import Config


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    port = _free_port()
    Handler.advisor = Advisor(Config(), time_budget=2.5, prior_budget=1.2)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    if "--check" in argv:
        with urllib.request.urlopen(url + "health", timeout=5) as r:
            ok = json.load(r).get("ok")
        body = {"hero": "BTN", "hole": "AhKd", "board": "", "preflop": [{"pos": p, "action": "fold"} for p in ("UTG", "HJ", "CO")]}
        req = urllib.request.Request(url + "advise", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            rec = json.load(r).get("recommended")
        print("health", ok, "| preflop advice", rec)
        return 0 if ok and rec else 1
    import webview
    webview.create_window("GTO Solver", url, width=1320, height=920, min_size=(900, 640))
    webview.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
