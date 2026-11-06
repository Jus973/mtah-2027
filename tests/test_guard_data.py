"""Exercise the hook protocol with inert command strings and an empty project."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude/hooks/guard_data.py"
PUBLIC_CHECK = "make airlock T=public-check CSV=submissions/candidates/example.csv"


@pytest.mark.parametrize(("command", "expected"), [
    (PUBLIC_CHECK, 0),
    ('make airlock T=public-check CSV="submissions/final/Example Team.csv"', 0),
    ("make airlock T=anchors", 0),
    ("make airlock -f .mtah.local", 2),
    ("make airlock T=anchors CSV=submissions/candidates/example.csv", 2),
    (PUBLIC_CHECK + " extra-goal", 2),
    (PUBLIC_CHECK + " PY=echo", 2),
    (PUBLIC_CHECK + " && true", 2),
    (PUBLIC_CHECK + "\necho extra", 2),
    ("make airlock T=public-check CSV=submissions/final/../../other.csv", 2),
    ("echo submissions/final/example.csv", 2),
])
def test_protected_command_exception(tmp_path, monkeypatch, command, expected):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    event = {"tool_input": {"command": command, "description": "synthetic hook test"}}
    result = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(event),
                            text=True, capture_output=True)
    assert result.returncode == expected
