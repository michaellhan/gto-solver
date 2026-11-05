"""The page's URLs: one per view (/practice/, /hands/ and the help pages /start/, /about/,
/glossary/, /changelog/, /fairplay/; the solver is /), and /<route>/<spot>/ for a precomputed spot (its id with
dashes: srp_btn_bb -> srp-btn-bb) under a view in SPOT_ROUTES (none now).

The list lives in one place, gto/static/index.html's ROUTES and SPOT_ROUTES lines; this module reads it from
there. The page is one file at the site's root (the static site has no rewrites), so each route gets a copy
of it with a <base href="../"> (one ../ per level) that makes every relative URL in it (scripts, the worker,
flop_library/, health.json, advise) the root's:

    python -m gto.routes <site dir>   writes <site dir>/<route>/index.html from <site dir>/index.html (deploy.sh)

gto/server.py serves the same paths with page_for().

A route that leaves the page moves to RETIRED: its old address stays, as a small stub that sends it to the frozen copy at
/gto-solver-classic/ (scripts/publish_classic.sh), search and hash kept. Never to /lizard/: Lizard links here for what it
doesn't have, so a redirect back would loop.
"""
from __future__ import annotations
import json
import os
import re
import sys

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SLUG = re.compile(r"^[a-z0-9-]+$")
CLASSIC = "/gto-solver-classic/"
RETIRED: list[str] = ["drills", "reports", "study", "preflop", "session", "import"]   # routes that have left the page's ROUTES


def routes(html: str) -> tuple[list[str], list[str]]:
    """(the view routes, those with a /<spot>/ level) from the page's `const ROUTES = {...};` and
    `const SPOT_ROUTES = [...];` lines."""
    m = re.search(r"^const ROUTES = \{(.*)\};", html, re.M)
    s = re.search(r"^const SPOT_ROUTES = \[(.*)\];", html, re.M)
    if not m or not s:
        raise ValueError("index.html has no ROUTES / SPOT_ROUTES line")
    names = re.findall(r"(\w+):\s*\"", m.group(1))
    spot = re.findall(r"\"(\w+)\"", s.group(1))
    assert set(spot) <= set(names), spot
    return names, spot


def spot_slugs(lib_dir: str) -> list[str]:
    """The URL names of the flop library's spots that have a report."""
    try:
        with open(os.path.join(lib_dir, "index.json")) as f:
            index = json.load(f)
    except (OSError, ValueError):
        return []
    out = [s["id"].replace("_", "-") for s in index.get("spots", []) if s.get("report")]
    return [s for s in out if SLUG.match(s)]


def pages(html: str, lib_dir: str) -> dict[str, int]:
    """Each route's directory (relative to the site root) -> its depth."""
    names, spot = routes(html)
    out = {r: 1 for r in names}
    for r in spot:
        for s in spot_slugs(lib_dir):
            out[f"{r}/{s}"] = 2
    return out


def with_base(html: str, depth: int) -> str:
    """The page with <base href="../.."> at the top of its <head> (after its charset), before anything that loads
    a URL."""
    if depth == 0:
        return html
    tag = f'<base href="{"../" * depth}">'
    out, n = re.subn(r"(<head>\s*(?:<meta charset=[^>]*>\s*)?)", lambda m: m.group(1) + tag + "\n", html, count=1)
    if not n:
        raise ValueError("index.html has no <head>")
    return out


def stub(route: str) -> str:
    """A retired route's page: a script that keeps the address's search and hash, a meta refresh where scripts don't run
    (in <noscript>: beside the script, the refresh could win the race and lose them), and a plain link."""
    url = f"{CLASSIC}{route}/"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GTO Solver</title>
<meta name="robots" content="noindex">
<script>location.replace({json.dumps(url)} + location.search + location.hash);</script>
<noscript><meta http-equiv="refresh" content="0; url={url}"></noscript>
</head>
<body>
<p>This page is now in the <a href="{url}">classic GTO Solver</a>.</p>
</body>
</html>
"""


def _retired(names: list[str]) -> list[str]:
    both = set(RETIRED) & set(names)
    if both:
        raise ValueError(f"both a route and RETIRED: {sorted(both)}")
    return RETIRED


def page_for(path: str, static: str = STATIC) -> str | None:
    """The page served at a URL path ("/practice/", "/session/index.html"), a retired route's stub, or None
    if it's neither. The root ("/") is the page itself."""
    p = path.split("?")[0].split("#")[0]
    if p.endswith("/index.html"):
        p = p[: -len("index.html")]
    if not p.endswith("/"):
        return None
    with open(os.path.join(static, "index.html"), encoding="utf-8") as f:
        html = f.read()
    rel = p.strip("/")
    if rel == "":
        return html
    if rel in _retired(routes(html)[0]):
        return stub(rel)
    depth = pages(html, os.path.join(static, "flop_library")).get(rel)
    return with_base(html, depth) if depth else None


def is_route_without_slash(path: str, static: str = STATIC) -> bool:
    """"/practice" (a route's directory without its slash: the static host redirects it to "/practice/")."""
    p = path.split("?")[0]
    if p.endswith("/") or "." in p.rsplit("/", 1)[-1]:
        return False
    return page_for(p + "/", static) is not None and p != ""


def write(site: str) -> list[str]:
    """Writes each route's index.html under the site directory (from its index.html, already stamped) and each
    retired route's stub; returns the directories written."""
    with open(os.path.join(site, "index.html"), encoding="utf-8") as f:
        html = f.read()
    done = []
    for rel, depth in pages(html, os.path.join(site, "flop_library")).items():
        d = os.path.join(site, *rel.split("/"))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
            f.write(with_base(html, depth))
        done.append(rel)
    for rel in _retired(routes(html)[0]):
        d = os.path.join(site, rel)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
            f.write(stub(rel))
        done.append(rel)
    return done


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m gto.routes <site dir>")
    print("routes:", " ".join(f"{r}/" for r in write(sys.argv[1])))
