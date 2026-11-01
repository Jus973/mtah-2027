"""make autopsy (HUMANS ONLY): per-transition diagnostics on the public window (PLAN.md M1).

For each GT transition in the public window (turnovers and D onsets/ends), derived scalars only:
whether the ball is seen at the flip, the ball gap around it, the losing team's nearest-player
distance, ball speed before and after, the first frame the new owner's team is within 1.0 / 1.5 /
2.5 m of the ball, the first contact by that team, the offset of the final candidate's nearest
matching transition, and how many Smart-Tagging phases overlap it. Then summary rates.

DEBUGGING ONLY. The table tells humans whether detectors fire and whether something is grossly
misaligned. None of its numbers may become a threshold, an offset or a rule (RULES.md: the public
window is not a tuning set). For that reason it refuses to run under the airlock: the agent that
tunes the pipeline never sees it. On a synthetic match it uses that match's GT over a public-sized
window, which is how it is tested.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

from mtah import config, pipeline, registry  # noqa: E402
from mtah.airlock import OUT_ENV  # noqa: E402
from mtah.decode.latency import transitions  # noqa: E402
from mtah.eval.score import PUBLIC_GT, pm  # noqa: E402
from mtah.features.ball import clean_ball, velocity  # noqa: E402
from mtah.features.players import team_distances  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402
from mtah.learn import takeover  # noqa: E402
from mtah.learn.stats import runs  # noqa: E402

if os.environ.get(OUT_ENV):
    sys.exit("make autopsy is for humans at a terminal: an agent must not see per-transition "
             "public-window diagnostics (RULES.md: the public window is not a tuning set)")

SEARCH = 50          # frames either side of a GT flip
RADII = (1.0, 1.5, 2.5)
PUBLIC_FRAMES = 7500

args = game_args()
cfg = config.load(os.path.join(registry.ROOT, "configs", "final.yaml"))
sp = registry.spec(args.game, args.data_dir)
fps = sp["fps"]
if sp["fake"]:
    gt = np.loadtxt(os.path.join(registry.data_dir_for(args.game, args.data_dir), "gt.csv"),
                    delimiter=",", skiprows=1, dtype=str)[:, 1]
    first = sp["half_blocks"][0][1]
    gt = np.where((np.arange(len(gt)) >= first) & (np.arange(len(gt)) < first + PUBLIC_FRAMES),
                  gt, "X")
else:
    gt = pm.read_label_csv(PUBLIC_GT)["label"].to_numpy(dtype=str)

atd, _ = load_game(args.game, args.data_dir)
labels, _ = pipeline.run(args.game, cfg, args.data_dir, atd=atd)
ball_raw = atd.ball_m()
ball, observed, _ = clean_ball(ball_raw, atd.pitch, cfg["ball"], fps)
players = atd.players_m()[..., :2].astype(float)
d, _ = team_distances(players, ball, atd.team_of_slot)
speed = np.linalg.norm(velocity(ball, 3, fps), axis=1)
contacts = takeover.contacts(ball, d, cfg["touch"], cfg["owner"]["contact_r_m"], fps)
c_start = np.array([e.start for e in contacts])
c_team = np.array([e.team for e in contacts])
missing = ~np.isfinite(ball_raw[:, 0])
gap_start, gap_len = runs(missing)
tag_path = registry.asset_paths(args.game, args.data_dir)["smart_tagging"]
tags = load_tags(tag_path, fps) if os.path.isfile(tag_path) else []
pred = transitions(labels)


def first_within(team, f, r):
    lo, hi = max(0, f - SEARCH), min(len(d), f + SEARCH + 1)
    hits = np.flatnonzero(d[lo:hi, team] <= r)
    return int(lo + hits[0] - f) if len(hits) else None


def gap_around(f):
    i = np.searchsorted(gap_start, f + 2, side="right") - 1
    if i >= 0 and gap_start[i] + gap_len[i] >= f - 2:
        return int(gap_len[i])
    return 0


rows = []
for f, kind, new in transitions(gt):
    team = "HA".index(new) if new in ("H", "A") else None
    row = {"frame": f, "kind": kind, "to": new, "ball_seen": bool(observed[f]),
           "gap_frames": gap_around(f),
           "v_before": round(float(speed[max(0, f - 5)]), 2), "v_after": round(float(speed[min(len(speed) - 1, f + 5)]), 2)}
    if team is not None:
        row["losing_team_d_m"] = round(float(d[f, 1 - team]), 2) if np.isfinite(d[f, 1 - team]) else None
        for r in RADII:
            row["owner_within_{}m".format(r)] = first_within(team, f, r)
        near = np.flatnonzero((c_team == team) & (np.abs(c_start - f) <= SEARCH))
        row["first_contact"] = int(c_start[near[0]] - f) if len(near) else None
    same = [p - f for p, k, n in pred if k == kind and n == new and abs(p - f) <= SEARCH]
    row["pred_offset"] = min(same, key=abs) if same else None
    row["tags_overlapping"] = sum(t.start <= f <= t.end for t in tags)
    rows.append(row)
rows = rows[:40]
put("public-window transitions (debugging only; never thresholds)", len(rows))
for row in rows:
    put("transition @{}".format(row["frame"]), {k: v for k, v in row.items() if k != "frame"})
turn = [r for r in rows if r["kind"] == "turnover"]
put("summary", {
    "turnovers": len(turn),
    "with a contact by the new owner within 2 s": sum(r.get("first_contact") is not None for r in turn),
    "matched by the final candidate within 2 s": sum(r["pred_offset"] is not None for r in rows),
    "median pred offset (frames)": float(np.median([r["pred_offset"] for r in rows
                                                     if r["pred_offset"] is not None]))
    if any(r["pred_offset"] is not None for r in rows) else None})
