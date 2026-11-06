"""make airlock T=ball-kin: label-free shape of the ball track's filtering (board #84).

Quantiles of the ball's curvature (as m/s velocity change) at lags 1-50 frames on graded,
in-pitch, seen frames, plus speed quantiles and the kick rate. The same function runs on the
sims, where a filter family and width are fitted to these aggregates to build the b-fit
evaluation world. Training never reads it. Aggregates only, no labels.
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
ball = atd.ball_m()
k = stats.ball_kinematics(ball, graded, atd.pitch, sp["fps"])
put("frames used", k["n"])
put("curvature dv m/s quantiles p50 p75 p90 p95 p99 by lag in frames", k["curv_q"])
put("ball speed m/s quantiles p50 p75 p90 p95 p99", k["speed_q"])
put("kicks per 1e3 seen", stats.carrier_stats(atd.players_m()[..., :2].astype(float), ball, graded,
                                              atd.pitch, sp["fps"])["kicks_per_1e3_seen"])
lag = stats.ball_player_lag(atd.players_m()[..., :2].astype(float), ball, graded, atd.pitch)
put("lag frames used", lag["n"])
put("nearest player median m by ball shift s (ball t+s vs players t)", lag["median_dmin_by_shift"])
put("share nearest player within 1.5 m by ball shift s", lag["share_within_1p5m_by_shift"])
