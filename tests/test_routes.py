"""The views' URLs (gto/routes.py): the route table read from the page, each route's copy of the page with its
<base>, the copies deploy.sh writes, and the local server serving the same paths."""
import http.client
import os
import re
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

from gto import routes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "gto", "static", "index.html"), encoding="utf-8").read()
LIB = os.path.join(ROOT, "gto", "static", "flop_library")


class TestRoutes(unittest.TestCase):
    def test_table(self):
        names, spot = routes.routes(HTML)
        self.assertEqual(names, ["practice", "session", "import", "hands",
                                 "start", "about", "glossary", "changelog", "fairplay"])
        self.assertEqual(spot, [])

    def test_pages_and_base(self):
        pages = routes.pages(HTML, LIB)
        self.assertEqual(pages["practice"], 1)
        # (no view has a /<spot>/ level now: a page that lists one still gets them)
        spotted = routes.pages(HTML.replace("const SPOT_ROUTES = [];", 'const SPOT_ROUTES = ["session"];'), LIB)
        for s in routes.spot_slugs(LIB):
            self.assertRegex(s, r"^[a-z0-9-]+$")
            self.assertEqual(spotted[f"session/{s}"], 2)
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
        self.assertIsNone(routes.page_for("/session/not-a-spot/"))
        self.assertTrue(routes.is_route_without_slash("/practice"))
        self.assertFalse(routes.is_route_without_slash("/practice.js"))
        self.assertFalse(routes.is_route_without_slash("/practice/"))

    def test_write(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
                f.write(HTML.replace('const WORKER_VERSION = "', 'const WORKER_VERSION = "stamped'))
            done = routes.write(d)
            self.assertIn("practice", done)
            page = open(os.path.join(d, "practice", "index.html"), encoding="utf-8").read()
            self.assertIn('<base href="../">', page)
            self.assertIn('WORKER_VERSION = "stamped', page)   # a copy of the stamped page
            self.assertEqual(len(done), 13)   # (the routes and the retired ones' stubs)


class TestRetired(unittest.TestCase):
    """A route that leaves the page keeps its address, as a stub to the frozen copy. "oldview" here is a route the page
    doesn't have."""

    def test_nothing_is_both(self):
        names, _ = routes.routes(HTML)
        self.assertEqual(set(routes.RETIRED) & set(names), set())
        for r in routes.RETIRED:
            self.assertRegex(r, r"^[a-z0-9-]+$")
        self.assertEqual(routes.RETIRED, ["drills", "reports", "study", "preflop"])
        self.assertEqual(routes.page_for("/reports/"), routes.stub("reports"))

    def test_stub(self):
        page = routes.stub("oldview")
        url = "/gto-solver-classic/oldview/"
        self.assertIn(f'<meta http-equiv="refresh" content="0; url={url}">', page)
        self.assertIn(f'location.replace("{url}" + location.search + location.hash);', page)   # search and hash kept
        self.assertIn(f'<a href="{url}">', page)
        self.assertNotIn("/lizard/", page)   # (Lizard links here: it would loop)
        self.assertNotIn("<base", page)      # (the address is absolute, and nothing else loads)
        self.assertLess(len(page), 600)

    def test_written_and_served(self):
        with mock.patch.object(routes, "RETIRED", ["oldview"]):
            self.assertEqual(routes.page_for("/oldview/"), routes.stub("oldview"))
            self.assertEqual(routes.page_for("/oldview/index.html?x=1"), routes.stub("oldview"))
            self.assertTrue(routes.is_route_without_slash("/oldview"))
            self.assertIsNone(routes.page_for("/oldview/more/"))
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
                    f.write(HTML)
                done = routes.write(d)
                self.assertIn("oldview", done)
                self.assertEqual(open(os.path.join(d, "oldview", "index.html"), encoding="utf-8").read(), routes.stub("oldview"))
                self.assertIn("practice", done)   # (the routes still there get their page)
        self.assertIsNone(routes.page_for("/oldview/"))

    def test_a_route_cant_be_both(self):
        with mock.patch.object(routes, "RETIRED", ["practice"]):
            with self.assertRaisesRegex(ValueError, "practice"):
                routes.page_for("/practice/")
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
                    f.write(HTML)
                with self.assertRaisesRegex(ValueError, "practice"):
                    routes.write(d)


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
        for js in ("/practice.js", "/explain.js", "/store.js", "/hands.js"):
            r, body = self.get(js)
            self.assertEqual(r.status, 200, js)
            self.assertEqual(r.getheader("Content-Type"), "text/javascript")
        r, body = self.get("/hands/")
        self.assertEqual(r.status, 200)
        self.assertIn(b'<base href="../">', body)
        r, _ = self.get("/health.json")
        self.assertEqual(r.status, 200)

    def test_retired(self):
        with mock.patch.object(routes, "RETIRED", ["oldview"]):
            r, body = self.get("/oldview/")
            self.assertEqual(r.status, 200)
            self.assertEqual(body.decode(), routes.stub("oldview"))
            self.assertEqual(r.getheader("Content-Type"), "text/html; charset=utf-8")
            r, _ = self.get("/oldview?x=1")
            self.assertEqual(r.status, 301)
            self.assertEqual(r.getheader("Location"), "/oldview/?x=1")


if __name__ == "__main__":
    unittest.main()
