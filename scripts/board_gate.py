"""CI gate (AGENTS.md §4): a PR must link the board issue it works on.

`Closes Jus973/mtah-board#N` needs N claimed (status:in-progress) or in review (status:review);
`Refs Jus973/mtah-board#N` (follow-ups to finished tasks) only needs N to exist. The link check
always runs; the board-state check runs when BOARD_TOKEN (read access to the private board) is
set, and is skipped with a warning otherwise (e.g. fork PRs, or before the secret exists)."""

import json
import os
import re
import subprocess
import sys

BOARD = "Jus973/mtah-board"
LINK = re.compile(r"\b(closes|fixes|resolves|refs)\s+" + re.escape(BOARD) + r"#(\d+)\b", re.I)
LIVE = ("status:in-progress", "status:review")


def links(body):
    """[(kind, N)] for every board link in a PR body; HTML comments don't count."""
    body = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    return [("refs" if k.lower() == "refs" else "closes", int(n)) for k, n in LINK.findall(body)]


def verdict(kind, issue):
    """None if the linked issue satisfies the gate, else the reason it doesn't."""
    if issue is None:
        return "not found on the board"
    if "pull_request" in issue:
        return "is a pull request, not a board issue"
    if kind == "refs":
        return None
    labels = {lab["name"] for lab in issue.get("labels", [])}
    if issue.get("state") == "open" and labels & set(LIVE):
        return None
    return f"is {issue.get('state')} without {' or '.join(LIVE)} — claim it with scripts/board claim"


def fetch(n):
    r = subprocess.run(["gh", "api", f"repos/{BOARD}/issues/{n}"], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def main():
    found = links(os.environ.get("PR_BODY", ""))
    if not found:
        print(f"::error::PR body must link its board issue: 'Closes {BOARD}#<N>' "
              f"(or 'Refs {BOARD}#<N>' for a follow-up) — see AGENTS.md §4")
        return 1
    if not os.environ.get("GH_TOKEN"):
        print("::warning::BOARD_TOKEN not available; link present, board state not checked")
        return 0
    bad = [(k, n, v) for k, n in found if (v := verdict(k, fetch(n)))]
    for k, n, v in bad:
        print(f"::error::{BOARD}#{n} ({k}) {v}")
    if not bad:
        print("board gate ok: " + ", ".join(f"{k} #{n}" for k, n in found))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
