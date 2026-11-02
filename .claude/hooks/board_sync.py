#!/usr/bin/env python3
"""AgentMax board hooks for Claude Code sessions in this repo (AGENTS.md).

SessionStart: inject `scripts/board start` output — the sync step of the
contribution loop, so every Claude session sees the board without being asked.

PostToolUse: throttled background `scripts/board beat-all`, so a busy session
heartbeats its claims without having to remember to (AGENTS.md §3).

Fail-open everywhere: coordination must never break the work itself.
"""
import json
import os
import subprocess
import sys
import time

THROTTLE_S = 20 * 60


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    board = os.path.join(root, "scripts", "board")
    if not os.path.exists(board):
        return 0
    event = payload.get("hook_event_name", "")

    if event == "SessionStart":
        try:
            out = subprocess.run([sys.executable, board, "start"], capture_output=True,
                                 text=True, timeout=25, cwd=root)
            if out.returncode == 0 and out.stdout.strip():
                print(out.stdout.strip())
                print("(AgentMax sync — protocol: AGENTS.md)")
        except Exception:
            pass
        return 0

    if event == "PostToolUse":
        stamp = os.path.join(root, ".git", "agentmax-last-beat")
        try:
            if not os.path.isdir(os.path.dirname(stamp)):
                return 0
            last = os.path.getmtime(stamp) if os.path.exists(stamp) else 0
            if time.time() - last < THROTTLE_S:
                return 0
            with open(stamp, "w") as fh:
                fh.write(str(time.time()))
            with open(os.devnull, "wb") as devnull:
                subprocess.Popen([sys.executable, board, "beat-all"],
                                 stdout=devnull, stderr=devnull, cwd=root,
                                 start_new_session=True)
        except Exception:
            pass
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
