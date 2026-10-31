"""make airlock T=touch-stats: label-free touch statistics, to check that sim-tuned touch settings
transfer to the real ATD.

On graded frames with a (cleaned) ball: the distance from the ball to its nearest player
(quantiles; share within 0.5-3 m), contact episodes and touches per minute of ball-seen time at
the configs/rules.yaml settings and at other radii, episode lengths, and how often the nearest
team switches. Run on the real match (through the airlock) and on sim matches (--game DFL-<id>)
and compare: if the real ball sits systematically farther from players, a radius tuned on the sim
will miss real touches. Aggregates only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

from mtah import config, registry  # noqa: E402
from mtah.events.touches import contact_episodes, touch_events  # noqa: E402
from mtah.features.ball import clean_ball  # noqa: E402
from mtah.features.players import team_distances  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402

RADII = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

args = game_args()
sp = registry.spec(args.game, args.data_dir)
cfg = config.load("rules.yaml")
atd, _ = load_game(args.game, args.data_dir)
graded = registry.graded_mask(sp)
ball, observed, filled = clean_ball(atd.ball_m(), atd.pitch, cfg["ball"], sp["fps"])
seen = graded & np.isfinite(ball[:, 0])
ball[~graded] = np.nan
d, _ = team_distances(atd.players_m()[..., :2].astype(float), ball, atd.team_of_slot)
near = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
ok = seen & np.isfinite(near)
minutes = ok.sum() / sp["fps"] / 60

put("ball-seen graded minutes", minutes)
put("ball kept by cleaning pct (of raw ball frames)",
    100 * observed[graded].sum() / max(1, np.isfinite(atd.ball_m()[graded, 0]).sum()))
put("nearest player distance m (p10, p25, p50, p75, p90)",
    [float(q) for q in np.percentile(near[ok], [10, 25, 50, 75, 90])])
put("share of ball-seen frames with a player within r (r: pct)",
    {str(r): 100 * float(np.mean(near[ok] <= r)) for r in RADII})
team = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1)
switch = ok[1:] & ok[:-1] & (team[1:] != team[:-1])
put("nearest team switches per minute", switch.sum() / minutes)
rows = {}
for r in RADII:
    tcfg = dict(cfg["touch"], r_touch_m=r)
    episodes, _ = contact_episodes(d, tcfg)
    lens = np.array([b - a + 1 for a, b, _ in episodes])
    touches = touch_events(ball, d, tcfg, sp["fps"])
    rows[str(r)] = {"episodes_per_min": len(episodes) / minutes,
                    "touches_per_min": len(touches) / minutes,
                    "episode_frames_p50": float(np.median(lens)) if len(lens) else None,
                    "episode_frames_p90": float(np.percentile(lens, 90)) if len(lens) else None}
put("contact episodes and touches by r_touch_m (other touch settings from rules.yaml)", rows)
