"""make candidate CFG=configs/rules.yaml NAME=v1 (agents: make airlock T=candidate CFG=... NAME=...).

Runs mtah.pipeline on the match and writes submissions/candidates/<NAME>.csv (gitignored, never
read by an agent). Prints aggregates only: the official validator, the structural bands
(PLAN.md §6.3, kick-offs checked against the published facts with the detector's tolerance),
Smart-Tagging consistency (§6.2a), the public-window bug check as a verdict (§6.4), the held-out
sim spellF1 of the same config (§6.1, if the sim is built), sha256, config hash and git SHA. Then
appends one row to submissions/LOG.md. Deterministic: same config + same commit = same bytes.
"""

import datetime
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args, put  # noqa: E402

from mtah import config, pipeline, registry  # noqa: E402
from mtah.eval.bands import bands  # noqa: E402
from mtah.eval.consistency import (phase_cover, set_piece_alignment, set_piece_checks,  # noqa: E402
                                   tag_consistency)
from mtah.eval.score import public_check, score  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402
from mtah.submit.write import sha256, validate, write_csv  # noqa: E402

LOG = os.path.join(registry.ROOT, "submissions", "LOG.md")


def add(parser):
    parser.add_argument("--config", default="configs/rules.yaml")
    parser.add_argument("--name", required=True)


def git_sha():
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=registry.ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
                            ":(exclude)submissions/LOG.md"],  # this script writes LOG.md itself
                           cwd=registry.ROOT, capture_output=True, text=True).stdout.strip()
    return out + ("+dirty" if dirty else "")


args = game_args(add=add)
if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.name):
    sys.exit("--name: letters, digits, _ and - only")
cfg = config.load(os.path.abspath(args.config))
sp = registry.spec(args.game, args.data_dir)
labels, info = pipeline.run(args.game, cfg, args.data_dir)
out = write_csv(os.path.join(registry.ROOT, "submissions", "candidates", args.name + ".csv"), labels)
digest = sha256(out)
put("candidate", {"name": args.name, "config": os.path.relpath(os.path.abspath(args.config),
                                                               registry.ROOT),
                  "config_hash": config.config_hash(cfg), "git": git_sha(), "sha256": digest[:16]})
graded = registry.graded_mask(sp)
put("pipeline", {"touches": info["touches"], "d_runs_from_evidence": info["d_runs_evidence"],
                 "d_share_graded": info["d_share_graded"],
                 "kickoffs_found": sorted(info["kickoffs"])})
if "postprocess" in info:
    put("postprocess (mtah/postprocess.py rules applied to the final labels)",
        dict(info["postprocess"], d_share_graded_final=info["d_share_graded_final"]))
if "gap_frames" in info:
    put("gap decoder", {"long_gap_share_graded": info["gap_frames"] / int(graded.sum()),
                        "alive_share_in_gaps": info["gap_alive_share"],
                        "d_share_graded_after": float(np.mean(labels[graded] == "D"))})

ok = check("official validator", validate(out, sp["n_frames"]) is None)
pub = cfg["published"]
real = not sp["fake"]
kicks = {int(k): v for k, v in pub["kickoff_frames"].items()} if real else None
team = {int(k): v for k, v in pub["kickoff_team"].items()} if real else None
b = bands(labels, sp, cfg["bands"], kicks, team, tol=cfg["kickoff"]["tolerance_frames"])
put("structure", {k: b[k] for k in ("n_spells", "frac_lt_2s", "frac_lt_6s", "mean_spell_s",
                                    "n_d_runs", "home_share_alive", "x_frac_graded",
                                    "d_share_graded")})
for name, passed in b["checks"].items():
    ok &= check("band " + name, passed)
for name, passed in b.get("advisory", {}).items():
    check("band {} (advisory this cycle, never gates the exit code)".format(name), passed)

tag_path = registry.asset_paths(args.game, args.data_dir)["smart_tagging"]
consistency = None
if os.path.isfile(tag_path):
    tags = load_tags(tag_path, sp["fps"])
    c = tag_consistency(labels, tags, cfg["tag_team_map"], registry.graded_mask(sp))
    consistency = c["agree"]
    put("smart-tagging consistency", {"agree": c["agree"], "frames": c["frames"]})
    put("smart-tagging consistency by code [agree, frames]", c["by_code"])
    tc = cfg["tags"]
    put("set-piece checks (PLAN.md M2 T2 b, c)",
        set_piece_checks(labels, tags, cfg["tag_team_map"], registry.graded_mask(sp),
                         tc["set_piece_restart_after_start_s"], tc["set_piece_window_s"],
                         tc["set_piece_match_s"], sp["fps"]))
    # Amended alignment (§6.2a): the raw anchor is the median of v1's OWN restart lags, so a
    # candidate that fixes the restart lag is judged at the config-corrected anchor (raw - k/fps,
    # k a config constant, never fitted), with the raw number and a candidate-matched chance
    # baseline reported beside it. Half-end pseudo-restarts cannot score here.
    raw = tc["set_piece_restart_after_start_s"]
    put("set-piece alignment, raw (v1-fitted) anchor",
        set_piece_alignment(labels, tags, cfg["tag_team_map"], graded, raw,
                            tc["set_piece_window_s"], sp["fps"]))
    ppc = cfg.get("postprocess") or {}
    k_shift = (ppc.get("restart_backshift_frames") or 0) if ppc.get("enabled") else 0
    if k_shift:
        put("set-piece alignment, config-corrected anchor (raw - {}/{} s)".format(k_shift,
                                                                                  sp["fps"]),
            set_piece_alignment(labels, tags, cfg["tag_team_map"], graded,
                                raw - k_shift / sp["fps"], tc["set_piece_window_s"], sp["fps"]))
    if "gap_mask" in info:
        gap = info["gap_mask"]
        put("smart-tagging consistency, gap frames decoded alive",
            {k: v for k, v in tag_consistency(labels, tags, cfg["tag_team_map"], gap).items()
             if k != "by_code"})
        cover = phase_cover(tags, cfg["tag_team_map"], len(labels))
        alive = np.isin(labels, ["H", "A"])
        masks = (("alive outside gaps", graded & alive & ~gap),
                 ("D outside gaps", graded & ~alive & ~gap),
                 ("alive in gaps", gap & alive), ("D in gaps", gap & ~alive))
        put("share inside a team-tagged phase, by predicted label and gap",
            {name: float(cover[m].mean()) if m.any() else None for name, m in masks})

verdict = None
if real:
    r = public_check(out, cfg["public_check"])
    verdict = r["verdict"]
    put("public bug check verdict", verdict)
    ok &= check("public bug check passes (verdict only)", verdict == "pass")
    ok &= check("team sign: H/A as written beats swapped", r["swap_scores_higher"] is False)

sim_scores = []
for mid in cfg["matches"]["holdout"]:
    game = "DFL-" + mid
    if os.path.isfile(os.path.join(registry.SIM_DIR, game, registry.FAKE_META)):
        sim_labels, _ = pipeline.run(game, cfg)
        sim_scores.append(score(sim_labels, "sim:" + game)["spell_f1_mean"])
sim_f1 = float(np.mean(sim_scores)) if sim_scores else None
put("held-out sim spellF1 (same config)", {"mean": sim_f1, "per_match": sim_scores})

if real:
    row = "| {} | {} | {} | {} | {} | {} | {} | {} | {} | no |\n".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), args.name, config.config_hash(cfg),
        git_sha(), digest[:16], verdict, "{:.4f}".format(sim_f1) if sim_f1 is not None else "-",
        "pass" if b["pass"] else "FAIL",
        "consistency {:.3f}".format(consistency) if consistency is not None else "")
    with open(LOG) as handle:
        text = handle.read()
    head, sep, tail = text.partition("\n## Work log")
    # Re-running a logged candidate is the reproducibility check: same name, same bytes.
    logged = [cells[4] for cells in (line.split(" | ") for line in head.splitlines())
              if len(cells) > 5 and cells[1] == args.name]  # cells: date, name, cfg, git, sha
    if logged:
        same = logged[-1] == digest[:16]
        put("regenerated {}: same sha256 as its latest LOG.md row".format(args.name), same)
    if not logged or logged[-1] != digest[:16]:
        with open(LOG, "w") as handle:
            handle.write(head.rstrip("\n") + "\n" + row + "\n" + sep + tail)
sys.exit(0 if ok else 1)
