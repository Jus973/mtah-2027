"""The airlock releases validated aggregates only; crash messages and raw-looking text never get out."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from mtah import airlock, registry
from mtah.registry import ROOT

HOOK = os.path.join(ROOT, ".claude", "hooks", "guard_data.py")


def report(values, checks=()):
    return {"values": values, "checks": list(checks)}


def run_airlock(job, game, data_dir, cache):
    env = dict(os.environ, MTAH_CACHE_DIR=str(cache))
    env.pop("MTAH_DATA_DIR", None)
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "airlock.py"), job,
                           "--game", game, "--data-dir", data_dir],
                          capture_output=True, text=True, env=env, cwd=ROOT)


def test_policy_accepts_aggregates():
    ok = report({"half 1": {"frame": 11189, "method": "formation+ball", "frac": 0.91},
                 "direction": {"1": {"H": 1, "A": -1}}, "missing": None},
                [{"name": "kick-off within +/-25 of 11185", "ok": True, "detail": {"offset": 4}}])
    assert airlock.violations(ok) == []


@pytest.mark.parametrize("bad, rule", [
    (report({"frames": list(range(65))}), "list longer"),
    (report({"line": "12:0.512,0.331,4.2;0.1,0.2,0.3"}), "string not allowed"),
    (report({"xml": '<Player id="3" teamId="1"/>'}), "string not allowed"),
    (report({"two lines": "a\nb"}), "string not allowed"),
    (report({"long": "x" * 121}), "string not allowed"),
    (report({"nan": float("nan")}), "type not allowed"),
    (report({"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}), "nested deeper"),
    (report({str(i): i for i in range(1100)}), "more than"),
    (report({"0.51,0.33,4.2": 1}), "key not allowed"),
    ({"values": {}}, "not a report"),
])
def test_policy_rejects(bad, rule):
    assert any(rule in r for _, r in airlock.violations(bad)), airlock.violations(bad)


def test_policy_paths_never_echo_keys():
    found = airlock.violations(report({"0.51,0.33,4.2": 1}))
    assert all("0.51" not in path for path, _ in found)


def test_plain():
    assert airlock.plain(np.float64(0.123456789)) == 0.123457
    assert airlock.plain({1: (np.int64(3), np.bool_(True))}) == {"1": [3, True]}
    assert airlock.plain(float("inf")) is None
    with pytest.raises(TypeError):
        airlock.plain(np.zeros(3))


def test_crash_summary_withholds_message():
    log = ('noise\nTraceback (most recent call last):\n'
           '  File "{0}/src/mtah/io/atd.py", line 87, in _group\n'
           '    return float(fields[0]), float(fields[1]), float(fields[2])\n'
           '  File "/usr/lib/python3.11/genericpath.py", line 9, in <module>\n'
           "ValueError: could not convert string to float: '0.4471,0.2210,3.1'\n").format(ROOT)
    crash = airlock.crash_summary(log, ROOT)
    assert crash == {"type": "ValueError", "frames": [["src/mtah/io/atd.py", 87, "_group"],
                                                      ["lib/genericpath.py", 9, "module"]]}
    assert airlock.violations(report({"crash": crash})) == []
    assert airlock.crash_summary("no traceback here", ROOT) is None


def test_airlock_end_to_end(fake, tmp_path):
    proc = run_airlock("anchors", fake["game"], fake["data_dir"], tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[PASS] half 1 kick-off within" in proc.stdout
    assert "[aggregates only]" not in proc.stdout  # nothing the child printed gets through
    with open(tmp_path / "airlock" / "released.jsonl") as handle:
        entries = [json.loads(line) for line in handle]
    assert len(entries) == 1 and entries[0]["released"] == proc.stdout.rstrip("\n")


def test_airlock_withholds_crash_message(fake, tmp_path):
    data = tmp_path / "bundle"
    shutil.copytree(fake["data_dir"], data)
    raw = registry.asset_paths(fake["game"], str(data))["atd_raw"]
    with open(raw) as handle:
        lines = handle.readlines()
    head, groups, _ = lines[0].rstrip("\n").split(":")
    lines[0] = "{}:{}:0.5,LEAKMARKER,1.0\n".format(head, groups)
    with open(raw, "w") as handle:
        handle.writelines(lines)
    proc = run_airlock("cache", fake["game"], str(data), tmp_path / "cache")
    assert proc.returncode != 0
    assert "crash: ValueError" in proc.stdout and "src/mtah/io/atd.py" in proc.stdout
    assert "LEAKMARKER" not in proc.stdout + proc.stderr
    logs = [p for p in os.listdir(tmp_path / "cache" / "airlock") if p.endswith(".log")]
    with open(tmp_path / "cache" / "airlock" / logs[0]) as handle:
        assert "LEAKMARKER" in handle.read()  # it was there, and it was held back


def test_real_scripts_refuse_without_terminal(tmp_path):
    env = dict(os.environ, MTAH_DATA_DIR=str(tmp_path))
    env.pop(airlock.OUT_ENV, None)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_anchors.py")],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    assert proc.returncode != 0 and "make airlock" in proc.stderr


@pytest.mark.parametrize("command, blocked", [
    ("make airlock T=anchors", False),
    ("grep -rn MTAH_DATA_DIR src", False),
    ("ls /tmp/mtah-test/bundle/tactical", True),
    ("du -sh /tmp/mtah-test/bundle_cache", True),
    ("MTAH_DATA_DIR=/x make anchors", True),
    ("export MTAH_AIRLOCK_OUT=/tmp/r.json", True),
    ("python -c 'import os; print(os.listdir(\"${MTAH_DATA_DIR}\"))'", True),
])
def test_guard_hook(tmp_path, command, blocked):
    (tmp_path / ".mtah.local").write_text("MTAH_DATA_DIR=/tmp/mtah-test/bundle\n")
    event = {"tool_name": "Bash", "tool_input": {"command": command}}
    proc = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True,
                          text=True, env=dict(os.environ, CLAUDE_PROJECT_DIR=str(tmp_path)))
    assert proc.returncode == (2 if blocked else 0), proc.stderr
