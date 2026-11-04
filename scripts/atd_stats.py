"""make atd-stats: label-free degradation statistics of an ATD, the targets for learn/degrade.py.

On graded frames only, and without any label: when the ball goes missing (onset hazard by ball
speed, crowding and pitch edge), how long it stays missing, ball and player jitter, ball jumps,
visible players per team and slot tracklet lengths. Aggregates only (counts, histograms,
quantiles); configs/sim.yaml defines the bins.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

from mtah import config, registry  # noqa: E402
from mtah.io.atd import AWAY, HOME, load_game  # noqa: E402
from mtah.learn import stats  # noqa: E402

args = game_args()
sp = registry.spec(args.game, args.data_dir)
cfg = config.load("sim.yaml")["measure"]
atd, _ = load_game(args.game, args.data_dir)
graded = registry.graded_mask(sp)
s = stats.atd_stats(atd.players_m()[..., :2].astype(np.float64), atd.ball_m(), atd.team_of_slot,
                    graded, atd.pitch, cfg, sp["fps"])

put("graded frames", int(graded.sum()))
put("ball missing on graded frames pct", 100 * s["ball_missing_frac"])
put("ball gaps", {"n": s["n_gaps"], "median_frames": s["gap_median"], "mean_frames": s["gap_mean"]})
for edge, name in ((0, "inside"), (1, "edge")):
    put("gap onsets {} [seen, onsets] by speed then crowd 0/1/2/3+".format(name),
        {lab: [[int(n), int(k)] for n, k in s["hazard"][edge, i]]
         for i, lab in enumerate(s["speed_labels"])})
put("gap lengths by onset speed (length bins)", {lab: [int(c) for c in s["gap_hist"][i]]
                                                  for i, lab in enumerate(s["speed_labels"])})
put("ball jitter d2 m (median, p75, p90)", s["ball_d2"])
put("player jitter d2 m (median, p75, p90)", s["player_d2"])
put("ball jumps", {"n": s["n_jumps"], "per_seen_frame": s["jump_rate"],
                   "far_outside_frames": s["n_far_outside"]})
put("ball between seen frames: share not moving / slower than 0.1 m/s",
    [s["ball_frozen_frac"], s["ball_still_frac"]])
for team, name in ((HOME, "home"), (AWAY, "away")):
    put("visible {} players per graded frame (counts for 0..{})".format(name, len(s["visible"][team]) - 1),
        [int(c) for c in s["visible"][team]])
    put("{} slot tracklets".format(name), {"n": s["tracklets"][team]["n"],
                                           "mean_frames": s["tracklets"][team]["mean"],
                                           "hist": [int(c) for c in s["tracklets"][team]["hist"]]})
    for kind, label in (("low_runs", "<= {} visible".format(cfg["low_max_visible"])),
                        ("high_runs", ">= {} visible".format(cfg["high_min_visible"]))):
        r = s[kind][team]
        put("{} runs with {} (tracklet bins)".format(name, label),
            {"n": r["n"], "mean_frames": r["mean"], "hist": [int(c) for c in r["hist"]]})
put("collective player speed m/s (median, p75, p90)", s["collective_speed"])
c = s["carrier"]
put("carrier loss proxies: ball seen in pitch, no player within 1 m",
    {"n_frames": c["n_seen_inpitch"], "absent": c["absent"],
     "carry_speed_frames": c["n_carry"], "carry_absent": c["carry_absent"]})
put("carrier loss proxies: nearest player at carry speed, m (le0.5, 0.5-1, 1-1.5, 1.5-2, 2-3, gt3, none)",
    c["carry_dmin_hist"])
put("carrier loss proxies: kicks (dv gt 3 m/s) with no player within 1.5 m",
    {"n_kicks": c["n_kicks"], "kicks_per_1e3_seen": c["kicks_per_1e3_seen"], "kick_far": c["kick_far"]})
