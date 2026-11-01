"""make audit [HISTORY=1]: the PLAN.md M4 repo audit over every tracked file (git ls-files), or
with --history over every blob in the git history (what a public push publishes). No match data.

FAIL on: a bundle file or derivative (BAR-ATH_* bundle names, *.mp4 *.npz *.parquet), a tracked
.mtah.local, a file over 5 MB, a likely secret (private keys, cloud or API tokens, password
assignments), a missing LICENSE or README command. WARN (not fail) on absolute home-directory
paths, which are fine locally but personal in a public repo: listed by file and count only.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 5 * 1024 * 1024
BUNDLE = re.compile(r"(BAR-ATH_(tactical|broadcast)[^/]*\.(mp4|xml|txt|csv))$|\.(mp4|npz|parquet)$")
SECRETS = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|ghp_[0-9A-Za-z]{36}"
                     r"|sk-[0-9A-Za-z]{32,}|xox[baprs]-[0-9A-Za-z-]{10,}"
                     r"|(?i:(password|passwd|secret|api_key|apikey|token)\s*[:=]\s*['\"][^'\"]{8,}['\"])")
HOME = re.compile(r"/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/")
README_COMMANDS = ("make setup", "make test", "make submission", "make airlock")


def history():
    """FAIL lines for every blob ever committed: bundle names, size, secrets."""
    objs = subprocess.run(["git", "rev-list", "--objects", "--all"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.splitlines()
    paths = {}
    for line in objs:
        sha, _, path = line.partition(" ")
        if path:
            paths.setdefault(sha, set()).add(path)
    check = subprocess.run(["git", "cat-file", "--batch-check"], cwd=ROOT, input="\n".join(paths),
                           capture_output=True, text=True, check=True).stdout.splitlines()
    fails = []
    for row in check:
        sha, kind, size = row.split()
        if kind != "blob":
            continue
        size = int(size)
        for path in sorted(paths[sha]):
            if BUNDLE.search(path) and not path.startswith(("evaluation/", "starter/")):
                fails.append("history: bundle file or derivative {} ({})".format(path, sha[:10]))
            if size > LIMIT:
                fails.append("history: over 5 MB {} ({:.1f} MB)".format(path, size / 2 ** 20))
        if size <= LIMIT and not any(p == "scripts/audit_repo.py" for p in paths[sha]):
            blob = subprocess.run(["git", "cat-file", "blob", sha], cwd=ROOT, capture_output=True,
                                  check=True).stdout
            if SECRETS.search(blob.decode("utf-8", "replace")):
                fails.append("history: possible secret in " + ", ".join(sorted(paths[sha])))
    return fails, len(paths)


def main():
    if "--history" in sys.argv:
        fails, n = history()
        print("blobs in history: {}".format(n))
        for line in fails:
            print("  [FAIL] " + line)
        print("  [{}] history audit".format("FAIL" if fails else "PASS"))
        sys.exit(1 if fails else 0)
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split("\n")
    files = [f for f in files if f]
    fails, warns = [], {}
    for f in files:
        path = os.path.join(ROOT, f)
        if BUNDLE.search(f) and not f.startswith(("evaluation/", "starter/")):
            fails.append("bundle file or derivative tracked: " + f)
        if os.path.basename(f) == ".mtah.local":
            fails.append(".mtah.local is tracked")
        if os.path.isfile(path) and os.path.getsize(path) > LIMIT:
            fails.append("over 5 MB: {} ({:.1f} MB)".format(f, os.path.getsize(path) / 2 ** 20))
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except (UnicodeDecodeError, OSError):
            continue
        if f != "scripts/audit_repo.py" and SECRETS.search(text):
            fails.append("possible secret in " + f)
        n = len(HOME.findall(text))
        if n and f != "scripts/audit_repo.py":
            warns[f] = n
    if not os.path.isfile(os.path.join(ROOT, "LICENSE")):
        fails.append("LICENSE missing")
    readme = open(os.path.join(ROOT, "README.md")).read()
    fails += ["README lacks `{}`".format(c) for c in README_COMMANDS if c not in readme]
    print("tracked files: {}".format(len(files)))
    for f, n in sorted(warns.items()):
        print("  [WARN] {} absolute home path(s) in {}".format(n, f))
    for line in fails:
        print("  [FAIL] " + line)
    print("  [{}] repo audit".format("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
