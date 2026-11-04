# Taking the live site back

The solver is the folder `gto-solver/` of the `michaellhan/michaellhan.github.io` repo, and each deploy is one commit
there, `gto-solver: deploy <solver commit>`. Going back is a new commit on that repo's `main`, never a force push:
other jobs push to it too, and it serves 23 folders.

## Which deploy is live

    gh api repos/michaellhan/michaellhan.github.io/commits/main \
      --jq '.sha[:10] + " " + (.commit.message | split("\n")[0])'

That's the repo's head, which is often another job's commit (`reels: 1 scheduled`). The last commits to the folder say
which deploy it is:

    gh api 'repos/michaellhan/michaellhan.github.io/commits?path=gto-solver&per_page=5' \
      --jq '.[] | .sha[:10] + " " + .commit.committer.date + " " + (.commit.message | split("\n")[0])'

The newest line is what's in the folder. Whether Pages has built the head yet:

    gh api repos/michaellhan/michaellhan.github.io/pages/builds/latest --jq '.status + " " + .commit[:10]'

## Going back

A partial clone (24 MB; the full history is over 1 GB, and a `--depth 1` clone can't revert):

    git clone --filter=blob:none https://github.com/michaellhan/michaellhan.github.io site && cd site

One deploy back, with `D` the newest deploy from the list above:

    git revert --no-edit D
    git push origin main

Further back, to the folder exactly as deploy `T` left it:

    git rm -rq gto-solver
    git checkout T -- gto-solver
    git commit -m "gto-solver: back to deploy of <solver commit>"
    git push origin main

If the push is refused, another job got there first: `git pull --rebase origin main` and push again. Pages takes about a
minute. Both touch only `gto-solver/` (rehearsed on 6273321be9: 18 files, all under it).

## What going back doesn't change

- **The next deploy undoes it.** `scripts/deploy.sh` publishes whatever the solver repo's checkout holds, so revert or fix
  that too before deploying again.
- **The frozen copy stays.** `deploy.sh` wipes only `gto-solver/` (`rm -rf "$WORK/site/gto-solver"`, in a fresh clone,
  then `git add -A`), so `gto-solver-classic/` is never in a deploy's commit, and a revert of one doesn't touch it. It goes
  only if `gto-solver-classic: frozen copy of 1a14605e` itself is reverted.
- **The library doesn't go back.** It's the separate repo `michaellhan/gto-solver-library`: one commit with no history,
  force-pushed by `scripts/deploy_library.sh`, so there's nothing in it to revert. Its commit message names the solver
  commit it came from:

      gh api repos/michaellhan/gto-solver-library/commits/main --jq '.commit.message'

  To put an older one back, publish it from that solver commit (the entries are in `gto/static/flop_library`):

      git worktree add --detach ../gto-old <solver commit>
      cd ../gto-old && scripts/deploy_library.sh "$(mktemp -d)"

  The frozen copy reads the same library, so that changes what it answers too.
- **A browser's saved data doesn't go back.** Hands, sessions and settings stay as they are (IndexedDB `gto-store`, whose
  version an older page opens the same).

## A failed deploy isn't a no-op

`deploy.sh` runs `deploy_library.sh`, which force-pushes the library, before it stamps the versions, writes the route
copies and pushes the page. If anything after that fails (a stamp's check, `gto.routes`, or the push when another job got
in first), the library is new and the page is old, and nothing says so. After a failed deploy, compare the library's
commit message with the live deploy. Deploying again finishes it; to go back instead, publish the library from the live
deploy's solver commit, as above.
