"""make airlock T=<target>: run a real-data target for an agent; release validated aggregates only.

The target's stdout and stderr go to a log beside the match cache (outside the repo; the agent's
deny rules and hook cover it). What the agent sees: the target's report if
mtah.airlock.violations() finds nothing in it, the exit code, and for a crash the exception type
with file:line frames but never its message. Every release is appended to released.jsonl beside
the log, so a human can audit exactly what the agent received.

The bundle path comes from $MTAH_DATA_DIR or from .mtah.local (gitignored, one line
MTAH_DATA_DIR=/path), never from the command line: an agent never needs to type it.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

from mtah import airlock, registry

JOBS = {
    "cache": "run_cache.py",
    "baseline-check": "baseline_check.py",
    "anchors": "run_anchors.py",
    "anchors-diag": "diag_kickoff.py",
    "probe": "probe_streams.py",
    "atd-stats": "atd_stats.py",
    "public-check": "public_check.py",
    "touch-stats": "touch_stats.py",
    "dead-stats": "dead_stats.py",
    "candidate": "run_candidate.py",
    "submission": "make_submission.py",
    "stability": "stability.py",
    "disagreement": "disagreement.py",
    "audio": "audio_stats.py",
    "sync": "sync_stats.py",
    "whistle": "whistle_stats.py",
}
# Extra options a job may take: CSV=..., CFG=..., NAME=... (anything else is refused).
JOB_OPTIONS = {"public-check": {"csv"}, "candidate": {"config", "name"}, "dead-stats": {"config"},
               "submission": {"config", "name"}, "stability": {"config"}}
REQUIRED = {"public-check": {"csv"}, "candidate": {"name"}}
LOCAL = os.path.join(registry.ROOT, ".mtah.local")
BLOCKED = 3


def local_data_dir():
    if os.environ.get("MTAH_DATA_DIR"):
        return os.environ["MTAH_DATA_DIR"]
    try:
        with open(LOCAL) as handle:
            for line in handle:
                key, _, value = line.strip().partition("=")
                if key == "MTAH_DATA_DIR" and value:
                    return os.path.expanduser(value)
    except OSError:
        pass
    return None


def git_sha():
    def git(*cmd):
        return subprocess.run(("git",) + cmd, cwd=registry.ROOT, capture_output=True,
                              text=True).stdout.strip()
    try:
        return git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain") else "")
    except OSError:
        return None


def release(job, game, rc, report_path, log_path):
    """(text the agent may see, blocked?)."""
    lines = ["[airlock] {} on {}: exit {}".format(job, game, rc)]
    blocked = False
    report = None
    if os.path.isfile(report_path):
        try:
            with open(report_path) as handle:
                report = json.load(handle)
        except ValueError:
            lines.append("report unreadable; nothing from it released")
            blocked = True
    if report is not None:
        problems = airlock.violations(report)
        if problems:
            blocked = True
            lines.append("BLOCKED: the report broke the release policy; nothing from it released")
            lines += ["  {} {}".format(path or "/", rule) for path, rule in problems[:20]]
        elif report["values"] or report["checks"]:
            lines.append(airlock.render(report))
    elif not blocked:
        lines.append("no report written")
    if rc != 0:
        with open(log_path, errors="replace") as handle:
            crash = airlock.crash_summary(handle.read(), registry.ROOT)
        if crash and not airlock.violations({"values": {"crash": crash}, "checks": []}):
            lines.append("crash: {} (message withheld; the full traceback is in the log)".format(
                crash["type"]))
            lines += ["  {}:{} in {}".format(*frame) for frame in crash["frames"]]
        elif crash:
            lines.append("crash: summary withheld (failed the release policy)")
    lines.append("full log, for humans: <cache>/airlock/{}".format(os.path.basename(log_path)))
    return "\n".join(lines), blocked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=sorted(JOBS))
    parser.add_argument("--game", default="BAR-ATH")
    parser.add_argument("--data-dir", default=None, help="synthetic matches only")
    parser.add_argument("--csv", default=None, help="candidate CSV (public-check)")
    parser.add_argument("--config", default=None, help="configs/*.yaml (candidate, dead-stats)")
    parser.add_argument("--name", default=None, help="candidate name (candidate) or team name (submission)")
    args = parser.parse_args()
    given = {k for k in ("csv", "config", "name") if getattr(args, k) is not None}
    allowed, needed = JOB_OPTIONS.get(args.job, set()), REQUIRED.get(args.job, set())
    if given - allowed or needed - given:
        sys.exit("airlock: {} takes {} (requires {})".format(
            args.job, sorted(allowed) or "no options", sorted(needed) or "none"))
    if args.csv and not (args.csv.endswith(".csv") and os.path.isfile(args.csv)):
        sys.exit("airlock: CSV must be an existing .csv file")
    if args.config and not (os.path.dirname(os.path.abspath(args.config))
                            == os.path.join(registry.ROOT, "configs")
                            and args.config.endswith(".yaml") and os.path.isfile(args.config)):
        sys.exit("airlock: CFG must be an existing configs/*.yaml")
    if args.name and not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.name):
        sys.exit("airlock: NAME: letters, digits, _ and - only")
    fake = registry.is_fake(args.game, args.data_dir)
    if args.data_dir and not fake:
        sys.exit("airlock: --data-dir is for synthetic matches; the bundle path comes from "
                 "MTAH_DATA_DIR or .mtah.local")
    env = dict(os.environ)
    if not fake:
        data_dir = local_data_dir()
        if not data_dir or not os.path.isdir(data_dir):
            sys.exit("airlock: no bundle. A human writes MTAH_DATA_DIR=/path/to/bundle into "
                     ".mtah.local once.")
        os.environ["MTAH_DATA_DIR"] = env["MTAH_DATA_DIR"] = data_dir

    log_dir = os.path.join(registry.cache_dir(), "airlock")
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(log_dir, "{}_{}_{}".format(stamp, args.game, args.job))
    report_path, log_path = base + ".json", base + ".log"
    env[airlock.OUT_ENV] = report_path
    cmd = [sys.executable, os.path.join(registry.ROOT, "scripts", JOBS[args.job]), "--game", args.game]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    if args.csv:
        cmd += ["--csv", os.path.abspath(args.csv)]
    if args.config:
        cmd += ["--config", os.path.abspath(args.config)]
    if args.name:
        cmd += ["--name", args.name]
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=env, cwd=registry.ROOT).returncode

    text, blocked = release(args.job, args.game, rc, report_path, log_path)
    print(text)
    entry = {"time": stamp, "job": args.job, "game": args.game, "git": git_sha(), "exit": rc,
             "blocked": blocked, "sha256": hashlib.sha256(text.encode()).hexdigest(),
             "released": text}
    with open(os.path.join(log_dir, "released.jsonl"), "a") as handle:
        handle.write(json.dumps(entry) + "\n")
    sys.exit(rc if rc else (BLOCKED if blocked else 0))


if __name__ == "__main__":
    main()
