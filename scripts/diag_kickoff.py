"""make anchors-diag: why did kick-off detection fail? Per-half aggregate scalars only.

For each "Kick Off (Start)" tag window (+/- search margins) prints counts, distances and
fractions; no positions or per-frame rows. Offsets are relative to the tag start, so nothing
printed is a raw tracking value.
"""

import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)  # empty slices before the teams walk out

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

from mtah import config, registry  # noqa: E402
from mtah.events.restarts import _formation_direction, _own_half_fraction, kickoff_tags  # noqa: E402
from mtah.io.atd import AWAY, HOME, load_game  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


args = game_args()
sp = registry.spec(args.game, args.data_dir)
cfg = config.load()["kickoff"]
fps = sp["fps"]
atd, _ = load_game(args.game, args.data_dir)
tags = load_tags(registry.require(["smart_tagging"], args.game, args.data_dir)[0], fps)
length, width = atd.pitch
centre = np.array([length / 2.0, width / 2.0])
ball = atd.ball_m()

all_present = atd.ball_present()
with np.errstate(all="ignore"):
    put("whole match: ball range (normalised)", {
        "x_min": np.nanmin(atd.ball[:, 0]), "x_max": np.nanmax(atd.ball[:, 0]),
        "y_min": np.nanmin(atd.ball[:, 1]), "y_max": np.nanmax(atd.ball[:, 1])})
    x1, x99 = np.nanpercentile(atd.xy[..., 0], [1, 99])
    y1, y99 = np.nanpercentile(atd.xy[..., 1], [1, 99])
    put("whole match: player 1st/99th pct (normalised)", {"x": [x1, x99], "y": [y1, y99]})

for half, tag in sorted(kickoff_tags(tags).items()):
    lo = max(0, tag.start - int(cfg["search_before_s"] * fps))
    hi = min(atd.n_frames - 1, tag.end + int(cfg["search_after_s"] * fps))
    w = np.arange(lo, hi + 1)
    present = all_present[w]
    d = np.linalg.norm(ball[w] - centre, axis=1)
    step = np.r_[np.nan, np.linalg.norm(np.diff(ball[w], axis=0), axis=1)]
    row = {"window_frames": len(w), "search_before_s": cfg["search_before_s"],
           "search_after_s": cfg["search_after_s"], "tag_frames": tag.end - tag.start,
           "ball_present": present.sum(), "ball_present_pct": 100 * present.mean(),
           "longest_present_run": longest_run(present), "longest_gap": longest_run(~present)}
    if present.any():
        i = int(np.nanargmin(d))
        row.update({"ball_centre_min_m": d[i], "ball_centre_min_at_tag_start_plus": w[i] - tag.start,
                    "ball_centre_median_m": np.nanmedian(d)})
        row["within_m (frames, longest run)"] = {
            r: [np.sum(np.isfinite(d) & (d <= r)), longest_run(np.isfinite(d) & (d <= r))]
            for r in (1, 2, 3, 5, 10)}
        still = np.isfinite(step) & (step < 0.1)
        row.update({"still_frames (step < 0.1 m)": still.sum(), "still_longest_run": longest_run(still),
                    "median_step_m": np.nanmedian(step)})
    nvis_h = np.sum(np.isfinite(atd.xy[w][:, atd.team_of_slot == HOME, 0]), axis=1)
    nvis_a = np.sum(np.isfinite(atd.xy[w][:, atd.team_of_slot == AWAY, 0]), axis=1)
    row["visible players (median, min)"] = {"home": [np.median(nvis_h), nvis_h.min()],
                                            "away": [np.median(nvis_a), nvis_a.min()]}
    fr = np.array([_own_half_fraction(atd, t, _formation_direction(atd, t), cfg) for t in w])
    ok = fr >= cfg["own_half_frac"]
    row["own-half fraction"] = {"max": np.nanmax(fr), "threshold": cfg["own_half_frac"],
                                "frames_at_or_above": ok.sum(), "longest_run": longest_run(ok)}
    with np.errstate(all="ignore"):
        hx = np.nanmean(atd.xy[w][:, atd.team_of_slot == HOME, 0], axis=1)
        ax = np.nanmean(atd.xy[w][:, atd.team_of_slot == AWAY, 0], axis=1)
    row["team centroid x at tag start (normalised)"] = {"home": hx[tag.start - lo],
                                                        "away": ax[tag.start - lo]}
    row["centroids on opposite sides pct"] = 100 * np.nanmean((hx - 0.5) * (ax - 0.5) < 0)
    # Where in the window does the formation hold AND the ball sit near the centre?
    both = ok & np.isfinite(d) & (d <= 3)
    row["formation OK and ball within 3 m"] = {"frames": both.sum(), "longest_run": longest_run(both)}
    put("half {}".format(half), row)
