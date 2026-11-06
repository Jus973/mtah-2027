"""make airlock T=player-kin: label-free shape of the player tracks (twin of ball-kin, board #85).

Curvature quantiles by lag, provider speed field vs position-derived speed, slots seen per frame,
and seen-run lengths, on graded frames. The same function runs on the sims to check whether the
degrader's player noise and dropout look like the real feed. Training never reads it. Aggregates
only, no labels.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

from mtah import registry  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402
from mtah.learn import stats  # noqa: E402

args = game_args()
sp = registry.spec(args.game, args.data_dir)
atd, _ = load_game(args.game, args.data_dir)
graded = registry.graded_mask(sp)
pm = atd.players_m()
k = stats.player_kinematics(pm[..., :2].astype(float), pm[..., 2].astype(float), graded, sp["fps"])
put("graded frames", k["n_frames"])
put("player curvature dv m/s quantiles p50 p75 p90 p95 p99 by lag in frames", k["curv_q"])
put("position-derived speed m/s quantiles p50 p75 p90 p95 p99", k["derived_speed_q"])
put("speed field m/s quantiles p50 p75 p90 p95 p99", k["field_speed_q"])
put("abs speed field minus derived m/s quantiles p50 p75 p90 p95 p99", k["abs_field_minus_derived_q"])
put("speed field vs derived correlation", k["field_derived_corr"])
put("speed field vs derived correlation by shift s (derived t-s)", k["field_vs_derived_corr_by_shift"])
put("share of graded frames by slots seen (18-24)", k["slots_seen_per_frame_share"])
put("share of graded frames with under 18 slots seen", k["slots_seen_lt18_share"])
put("seen runs", k["seen_runs"])
put("seen run length frames p10 p25 p50 p75 p90", k["seen_run_len_frames_q"])
