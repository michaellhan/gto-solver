"""scripts/publish_classic.sh, the frozen copy at /gto-solver-classic/: no network and no build, only what it's pinned
to, that each change it makes is checked, and that scripts/deploy.sh can't touch the folder."""
import os
import re
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "publish_classic.sh")
ISOLATE = 'if (sessionStorage.getItem("gto-isolate")) return false; sessionStorage.setItem("gto-isolate", "1");'
LIB = 'const LIB_BASE = "../gto-solver-library/";\n'


def text(*path):
    with open(os.path.join(ROOT, *path), encoding="utf-8") as f:
        return f.read()


def bash(snippet, *args):
    return subprocess.run(["bash", "-c", f'source "{SCRIPT}"; {snippet}', "bash", *args], capture_output=True, text=True, cwd=ROOT)


def write(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


class TestClassic(unittest.TestCase):
    def test_pinned(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        self.assertRegex(text("scripts", "publish_classic.sh"), r"(?m)^CLASSIC_COMMIT=1a14605e$")

    def test_commit_exists_where_it_can_be_resolved(self):
        r = subprocess.run(["git", "rev-parse", "-q", "--verify", "1a14605e^{commit}"], capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            self.skipTest("no history here")
        self.assertTrue(r.stdout.startswith("1a14605e"))

    def test_replace_once_needs_exactly_one(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "worker.js")
            for count, ok in ((0, False), (1, True), (2, False)):
                before = "a\n" + 'caches.open("gto-solver");\n' * count
                write(f, before)
                r = bash('replace_once "$1" "$2" "$3"', f, 'caches.open("gto-solver")', 'caches.open("gto-solver-classic")')
                self.assertEqual(r.returncode == 0, ok, (count, r.stderr))
                if ok:
                    self.assertEqual(text(f), 'a\ncaches.open("gto-solver-classic");\n')
                else:
                    self.assertEqual(text(f), before)     # (left as it was)
                    self.assertIn("expected exactly one", r.stderr)

    def test_replace_once_is_literal(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "x")
            write(f, "a.b$c\naxb$c\n")
            self.assertEqual(bash('replace_once "$1" "$2" "$3"', f, "a.b$c", "y$1").returncode, 0)
            self.assertEqual(text(f), "y$1\naxb$c\n")

    def build(self, d):
        write(os.path.join(d, "worker.js"), 'const c = await caches.open("gto-solver");\n')
        for page in ("index.html", "practice/index.html"):
            write(os.path.join(d, page), f"<script>\n{LIB}{ISOLATE}</script>\n")

    def test_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            self.build(d)
            r = bash('freeze "$1"', d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(text(d, "worker.js"), 'const c = await caches.open("gto-solver-classic");\n')
            for page in ("index.html", "practice/index.html"):
                self.assertIn('getItem("gto-isolate-classic")', text(d, page))
                self.assertIn('setItem("gto-isolate-classic", "1")', text(d, page))
                self.assertNotIn('"gto-isolate"', text(d, page))

    def test_freeze_refuses_what_it_cant_stand_behind(self):
        for name, change in (("no cache line", lambda d: write(os.path.join(d, "worker.js"), "x\n")),
                             ("two cache lines", lambda d: write(os.path.join(d, "worker.js"), 'caches.open("gto-solver") caches.open("gto-solver")\n')),
                             ("a route with no isolate line", lambda d: write(os.path.join(d, "study", "index.html"), f"<script>\n{LIB}</script>\n")),
                             ("the library in the folder", lambda d: write(os.path.join(d, "index.html"), f'<script>\nconst LIB_BASE = "library/";\n{ISOLATE}</script>\n')),
                             ("a path keyed to the folder", lambda d: write(os.path.join(d, "a.js"), 'fetch("/gto-solver/x.json")\n'))):
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                self.build(d)
                change(d)
                self.assertNotEqual(bash('freeze "$1"', d).returncode, 0)

    def test_the_source_keeps_its_names(self):
        # (the copy's changes are made to the build: Lizard's build requires this exactly once in the source)
        self.assertEqual(text("gto", "static", "worker.js").count('caches.open("gto-solver")'), 1)

    def test_the_live_deploy_leaves_the_folder_alone(self):
        s = text("scripts", "deploy.sh")
        wipes = re.findall(r"^\s*rm -rf .*$", s, re.M)
        self.assertTrue(any('"$WORK/site/gto-solver"' in w for w in wipes), wipes)
        for w in wipes:
            self.assertNotRegex(w, r'gto-solver\*|site/\*|"\$WORK/site"(\s|$)', w)
        self.assertNotIn("gto-solver-classic", s)

    def test_the_publish_touches_only_its_folder(self):
        s = text("scripts", "publish_classic.sh")
        self.assertIn('git -C "$work/site" add -- "$FOLDER"', s)
        self.assertNotIn("add -A", s)
        self.assertNotIn("--force", s.replace("worktree remove --force", ""))
        self.assertNotRegex(s, r"(?m)^\s*rm -rf .*site/gto-solver[\"'/]?\s*$")


if __name__ == "__main__":
    unittest.main()
