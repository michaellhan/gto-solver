"""The views' URLs (gto/routes.py): the route table read from the page, each route's copy of the page with its
<base>, the copies deploy.sh writes, and the local server serving the same paths."""
import http.client
import os
import re
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from gto import routes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "gto", "static", "index.html"), encoding="utf-8").read()
LIB = os.path.join(ROOT, "gto", "static", "flop_library")


class TestRoutes(unittest.TestCase):
    def test_table(self):
        names, spot = routes.routes(HTML)
        self.assertEqual(names, ["practice", "drills", "reports", "session", "study", "preflop", "import", "hands",
                                 "start", "about", "glossary", "changelog", "fairplay"])
        self.assertEqual(spot, ["drills", "reports"])

    def test_pages_and_base(self):
        pages = routes.pages(HTML, LIB)
        self.assertEqual(pages["practice"], 1)
        for s in routes.spot_slugs(LIB):
            self.assertRegex(s, r"^[a-z0-9-]+$")
            self.assertEqual(pages[f"reports/{s}"], 2)
            self.assertEqual(pages[f"drills/{s}"], 2)
        page = routes.with_base(HTML, 2)
        head = page[page.index("<head>"): page.index("</head>")]
        # the base comes after the charset and before anything that loads a URL
        self.assertTrue(re.match(r'<head>\s*<meta charset="utf-8">\s*<base href="\.\./\.\./">', head), head[:120])
        self.assertEqual(len(re.findall(r"^<base ", page, re.M)), 1)
        self.assertEqual(re.findall(r"^<base ", HTML, re.M), [])
        self.assertEqual(routes.with_base(HTML, 0), HTML)

    def test_page_for(self):
        self.assertEqual(routes.page_for("/"), HTML)
        self.assertIn('<base href="../">', routes.page_for("/practice/"))
        self.assertIn('<base href="../">', routes.page_for("/session/index.html?x=1"))
        self.assertIsNone(routes.page_for("/nope/"))
        self.assertIsNone(routes.page_for("/practice"))
        self.assertIsNone(routes.page_for("/reports/not-a-spot/"))
        self.assertTrue(routes.is_route_without_slash("/practice"))
        self.assertFalse(routes.is_route_without_slash("/practice.js"))
        self.assertFalse(routes.is_route_without_slash("/practice/"))
        slugs = routes.spot_slugs(LIB)
        if slugs:
            self.assertIn('<base href="../../">', routes.page_for(f"/reports/{slugs[0]}/"))

    def test_write(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
                f.write(HTML.replace('const WORKER_VERSION = "', 'const WORKER_VERSION = "stamped'))
            done = routes.write(d)
            self.assertIn("practice", done)
            page = open(os.path.join(d, "practice", "index.html"), encoding="utf-8").read()
            self.assertIn('<base href="../">', page)
            self.assertIn('WORKER_VERSION = "stamped', page)   # a copy of the stamped page
            self.assertEqual(len(done), 13)   # (no flop_library here: no spot pages)


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gto.server import Handler
        Handler.log_message = lambda *a: None
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.srv.server_address[1], timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r, body

    def test_routes(self):
        r, body = self.get("/")
        self.assertEqual(r.status, 200)
        self.assertEqual(body.decode(), HTML)
        r, body = self.get("/practice/")
        self.assertEqual(r.status, 200)
        self.assertIn(b'<base href="../">', body)
        self.assertEqual(r.getheader("Content-Type"), "text/html; charset=utf-8")
        r, _ = self.get("/practice?x=1")
        self.assertEqual(r.status, 301)
        self.assertEqual(r.getheader("Location"), "/practice/?x=1")
        r, _ = self.get("/nope/")
        self.assertEqual(r.status, 404)
        for js in ("/practice.js", "/store.js", "/hands.js"):
            r, body = self.get(js)
            self.assertEqual(r.status, 200, js)
            self.assertEqual(r.getheader("Content-Type"), "text/javascript")
        r, body = self.get("/hands/")
        self.assertEqual(r.status, 200)
        self.assertIn(b'<base href="../">', body)
        r, _ = self.get("/health.json")
        self.assertEqual(r.status, 200)


if __name__ == "__main__":
    unittest.main()
