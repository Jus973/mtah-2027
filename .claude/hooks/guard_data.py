#!/usr/bin/env python3
"""PreToolUse hook: block agent shell commands that name the match bundle, its cache or the data env vars.

Agents reach real data only through `make airlock T=<target>` (scripts/airlock.py), which never
needs a path or a data variable on the command line. Reads the hook event JSON on stdin and
exits 2 (block, reason on stderr) on a hit. The bundle path comes from .mtah.local.
"""

import json
import os
import re
import sys

ENV = re.compile(r"\b(MTAH_DATA_DIR|MTAH_CACHE_DIR|MTAH_AIRLOCK_OUT)\s*=|\$\{?(MTAH_DATA_DIR|MTAH_CACHE_DIR)\b")


def protected(project):
    paths = []
    try:
        with open(os.path.join(project, ".mtah.local")) as handle:
            for line in handle:
                key, _, value = line.strip().partition("=")
                if key == "MTAH_DATA_DIR" and value:
                    value = os.path.abspath(os.path.expanduser(value)).rstrip(os.sep)
                    paths += [value, value + "_cache"]
    except OSError:
        pass
    home = os.path.expanduser("~")
    out = []
    for path in paths:
        out.append(path)
        if path.startswith(home + os.sep):
            out += ["~" + path[len(home):], "$HOME" + path[len(home):], "${HOME}" + path[len(home):]]
    return out


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def main():
    event = json.load(sys.stdin)
    project = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or os.getcwd()
    text = "\n".join(strings(event.get("tool_input", {})))
    if any(path in text for path in protected(project)) or ENV.search(text):
        print("Blocked by .claude/hooks/guard_data.py: the command names the match bundle, its "
              "cache, or a data variable. Run real-data targets with `make airlock T=<target>`.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
