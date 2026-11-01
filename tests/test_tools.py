"""Human-only debugging tools (autopsy, clip export) and make submission, on a synthetic match."""

import json
import os
import subprocess
import sys

from mtah import registry

SCRIPTS = os.path.join(registry.ROOT, "scripts")


def run(script, fake, *extra, env=None):
    cmd = [sys.executable, os.path.join(SCRIPTS, script), "--game", fake["game"],
           "--data-dir", fake["data_dir"], *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=registry.ROOT)


def test_human_tools_refuse_the_airlock(fake, tmp_path):
    env = dict(os.environ, MTAH_AIRLOCK_OUT=str(tmp_path / "report.json"))
    for script in ("run_autopsy.py", "clip_export.py"):
        r = run(script, fake, env=env)
        assert r.returncode != 0 and "humans" in r.stderr


def test_clip_export(fake, tmp_path):
    r = run("clip_export.py", fake, "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    data = json.load(open(tmp_path / "clips.json"))
    assert data["candidates"] == ["v1", "v2", "v3", "v5", "v6"] and data["clips"]
    frame = data["clips"][0]["frames"][0]
    assert set(frame["labels"]) == {"v1", "v2", "v3", "v5", "v6"}
    assert all(len(p) == 3 for p in frame["players"])


def test_autopsy_runs_on_a_synthetic_match(fake):
    r = run("run_autopsy.py", fake)
    assert r.returncode == 0, r.stderr
    assert "summary" in r.stdout and "turnovers" in r.stdout


def test_make_submission_on_a_synthetic_match(fake):
    r = run("make_submission.py", fake, "--name", "TEST")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[PASS] official validator on the exact file" in r.stdout
    os.remove(os.path.join(registry.ROOT, "submissions", "final", "TEST.csv"))


def test_make_submission_accepts_a_team_name_with_spaces(fake):
    r = run("make_submission.py", fake, "--name", "Test Team Name")
    out = os.path.join(registry.ROOT, "submissions", "final", "Test Team Name.csv")
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert os.path.isfile(out)
    finally:
        if os.path.exists(out):
            os.remove(out)
    assert run("make_submission.py", fake, "--name", "bad;name").returncode != 0
