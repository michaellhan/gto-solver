"""The end of scripts/deploy.sh against a local bare repo: a depth-1 clone that loses the race to push goes on top of
what won, once, and is never forced."""
import os
import re
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
           GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True, text=True).stdout.strip()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class TestPublish(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = tmp.name
        self.bare = os.path.join(tmp.name, "site.git")
        git("init", "-q", "--bare", "-b", "main", self.bare)
        self.url = "file://" + self.bare
        other = self.rival = os.path.join(tmp.name, "rival")
        git("clone", "-q", self.url, other)
        git("checkout", "-q", "-b", "main", cwd=other)
        for i in range(3):       # some history for the depth-1 clone to leave out
            write(os.path.join(other, "lizard/index.html"), f"lizard {i}\n")
            write(os.path.join(other, "gto-solver/index.html"), f"page {i}\n")
            git("add", "-A", cwd=other)
            git("commit", "-q", "-m", f"site {i}", cwd=other)
        git("push", "-q", "origin", "main", cwd=other)
        self.site = os.path.join(tmp.name, "site")
        git("clone", "-q", "--depth", "1", self.url, self.site)    # as deploy.sh's gh repo clone
        with open(os.path.join(ROOT, "scripts", "deploy.sh")) as f:
            self.publish = f.read().split('cd "$WORK/site"\n')[-1]

    def rival_pushes(self, path, text):
        write(os.path.join(self.rival, path), text)
        git("add", "-A", cwd=self.rival)
        git("commit", "-q", "-m", "rival", cwd=self.rival)
        git("push", "-q", "origin", "main", cwd=self.rival)

    def deploy(self):
        write(os.path.join(self.site, "gto-solver/index.html"), "new page\n")
        script = f'set -euo pipefail\nHERE="{ROOT}"\ncd "{self.site}"\n{self.publish}'
        return subprocess.run(["bash", "-c", script], env=ENV, capture_output=True, text=True)

    def remote(self, path):
        return git("show", "main:" + path, cwd=self.bare)

    def test_no_force_anywhere(self):
        self.assertIsNone(re.search(r"git push[^\n]*(-f\b|--force)|\+main", self.publish))

    def test_an_uncontested_push(self):
        r = self.deploy()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("rejected;", r.stdout)
        self.assertEqual(self.remote("gto-solver/index.html"), "new page")

    def test_a_lost_race_goes_on_top_of_what_won(self):
        self.rival_pushes("lizard/index.html", "lizard new\n")
        won = git("rev-parse", "main", cwd=self.bare)
        r = self.deploy()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("rejected;", r.stdout)
        self.assertEqual(self.remote("gto-solver/index.html"), "new page")
        self.assertEqual(self.remote("lizard/index.html"), "lizard new")
        self.assertEqual(git("rev-parse", "main~1", cwd=self.bare), won)     # nothing of theirs is lost
        self.assertEqual(git("rev-list", "--count", "main", cwd=self.bare), "5")

    def test_a_conflict_stops_with_nothing_published(self):
        self.rival_pushes("gto-solver/index.html", "their page\n")
        won = git("rev-parse", "main", cwd=self.bare)
        r = self.deploy()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nothing was published", r.stderr)
        self.assertEqual(git("rev-parse", "main", cwd=self.bare), won)


if __name__ == "__main__":
    unittest.main()
