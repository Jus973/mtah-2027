"""make airlock T=gap-stats [CFG=...]: label-free gap-decoder input distributions on one match.

The gap decoder (decode/gaps.py) re-labels long ball gaps (gaps.min_frames) from the players
alone, and the real match decodes far fewer in-gap frames alive than the sims do. This reports,
per match, the decoder's INPUT world so the two can be compared distribution by distribution:
the long-gap count, lengths and share of graded; each gapstate feature's quantiles over in-gap
graded frames; and the committed gap models' p_dead / p_home quantiles and threshold shares on
those frames. No GT is read and no labels are decoded (label-free on any match): a shifted input
distribution points at the decoder under-calling alive out of distribution, matching inputs point
at real gaps simply being deader. Run on the real match through the airlock
(make airlock T=gap-stats) and on sim matches directly (make gap-stats GAME=DFL-...); compare.
Aggregates only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args, put  # noqa: E402

from mtah import config, registry  # noqa: E402
from mtah.features.ball import clean_ball  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402
from mtah.learn import gapstate  # noqa: E402
from mtah.learn.stats import runs  # noqa: E402
from mtah.pipeline import kickoffs, pre_kickoff_mask  # noqa: E402

QS = (10, 25, 50, 75, 90)


def add(parser):
    parser.add_argument("--config", default="configs/final.yaml")


def quantiles(values, digits=4):
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return [float("{{:.{}g}}".format(digits).format(np.percentile(values, p))) for p in QS]


args = game_args(add=add)
sp = registry.spec(args.game, args.data_dir)
cfg = config.load(os.path.abspath(args.config))
gc = cfg["gaps"]
fps = sp["fps"]
atd, _ = load_game(args.game, args.data_dir)
graded = registry.graded_mask(sp)
ball, observed, _ = clean_ball(atd.ball_m(), atd.pitch, cfg["ball"], fps)
players = atd.players_m()[..., :2].astype(float)

# home_direction exactly as the pipeline derives it: Smart Tagging kick-offs where the match has
# them (the real match), {} on sims (the half-start line-up fallback).
kicks = kickoffs(args.game, atd, cfg, args.data_dir)
F = gapstate.features(players, atd.team_of_slot,
                      gapstate.home_direction(atd, sp, kicks, cfg["direction"]["window_frames"]),
                      atd.pitch, gc, fps)
F_home = F[0]
# The decoder's own gap mask: frames before a derived kick-off stay D and are never gap-decoded
# (pipeline.run), so they are not part of its input world.
gap_raw = gapstate.long_gaps(observed, graded, gc["min_frames"])
pre_kick = pre_kickoff_mask(sp, kicks, cfg["kickoff"]["tolerance_frames"])
gap = gap_raw & ~pre_kick
idx = np.flatnonzero(gap)
_, lens = runs(gap)
lens = np.asarray(lens, dtype=float)

put("game", args.game)
put("config", os.path.basename(args.config))
put("graded frames", int(graded.sum()))
put("long gaps (>= {} frames)".format(gc["min_frames"]), {
    "n": int(len(lens)),
    "in-gap graded frames": int(len(idx)),
    "share of graded": float(len(idx) / graded.sum()) if graded.sum() else 0.0,
    "pre-kick-off in-gap frames excluded": int((gap_raw & pre_kick).sum()),
    "length quantiles (s) p10/25/50/75/90": quantiles(lens / fps),
})

# The unshifted block only (through "dspeed"): the time-shifted context copies share the same
# marginal distributions. Home perspective; the away array mirrors it.
names = gapstate.feature_names(gc)
names = names[:names.index("dspeed") + 1]
feat = {}
for j, name in enumerate(names):
    feat[name] = quantiles(F_home[idx, j]) if idx.size else None
put("gapstate feature quantiles p10/25/50/75/90, in-gap frames (home persp.)", feat)

if idx.size:
    p_dead, p_home = gapstate.predict(gapstate.load(cfg), F, idx)
    put("committed gap models over in-gap graded frames", {
        "p_dead quantiles p10/25/50/75/90": quantiles(p_dead),
        "share p_dead > 0.5": float(np.mean(p_dead > 0.5)),
        "share p_dead < 0.2 (confidently alive)": float(np.mean(p_dead < 0.2)),
        "p_home quantiles p10/25/50/75/90": quantiles(p_home),
        "share p_home in [0.35, 0.65] (owner uncertain)": float(
            np.mean((p_home >= 0.35) & (p_home <= 0.65))),
    })
if idx.size:
    put("share of in-gap frames with any non-finite feature (unshifted block)",
        float(np.mean(~np.isfinite(F_home[idx][:, :len(names)]).all(axis=1))))
check("long gaps found", len(lens) > 0, n=int(len(lens)))
