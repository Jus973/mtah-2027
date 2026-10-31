"""Data-hygiene and vendoring invariants (PLAN.md §3)."""

import hashlib
import json
import os

from mtah.registry import ROOT

sys_path = os.path.join(ROOT, "scripts")
import sys  # noqa: E402
sys.path.insert(0, sys_path)
from check_no_data import problems  # noqa: E402


def read(name):
    with open(os.path.join(ROOT, name)) as handle:
        return handle.read()


def test_ignores():
    for name in (".gitignore", ".cursorignore"):
        lines = read(name).split()
        for entry in ("data/", "cache/", "*.mp4", "*.npz", "*.parquet", "validation/raw/"):
            assert entry in lines, (name, entry)


def test_agent_rules_present():
    assert "NEVER read/print/decode/display" in read("CLAUDE.md")
    assert "make airlock" in read("CLAUDE.md")
    settings = json.loads(read(".claude/settings.json"))
    deny = settings["permissions"]["deny"]
    assert "Read(./data/**)" in deny and any(d.startswith("Read(/") for d in deny)
    hooks = [h["command"] for m in settings["hooks"]["PreToolUse"] for h in m["hooks"]]
    assert any("guard_data.py" in c for c in hooks)
    assert ".mtah.local" in read(".gitignore").split()


def test_precommit_guard():
    assert problems("data/tactical/x.csv")
    assert problems("foo/BAR-ATH_tactical_atd.txt")
    assert problems("clip.mp4") and problems("cache/a.npz")
    assert problems("fakedata/out/BAR-FAKE/gt.csv")
    assert problems("src/mtah/io/atd.py") is None
    assert problems("evaluation/gt/BAR-ATH_possession_gt_public.csv") is None


def test_vendored_files_unmodified():
    with open(os.path.join(ROOT, "third_party", "vendored.sha256")) as handle:
        for line in handle:
            digest, path = line.split()
            with open(os.path.join(ROOT, path), "rb") as f:
                assert hashlib.sha256(f.read()).hexdigest() == digest, path
