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
from _guard import game_args  # noqa: E402

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
print("whole match: ball x range [{:.3f}, {:.3f}], y range [{:.3f}, {:.3f}] (normalised, min/max)".format(
    np.nanmin(atd.ball[:, 0]), np.nanmax(atd.ball[:, 0]), np.nanmin(atd.ball[:, 1]),
    np.nanmax(atd.ball[:, 1])))
with np.errstate(all="ignore"):
    px = atd.xy[..., 0]
    print("whole match: player x 1st/99th pct {:.3f} / {:.3f}, y {:.3f} / {:.3f}".format(
        *np.nanpercentile(px, [1, 99]), *np.nanpercentile(atd.xy[..., 1], [1, 99])))

for half, tag in sorted(kickoff_tags(tags).items()):
    lo = max(0, tag.start - int(cfg["search_before_s"] * fps))
    hi = min(atd.n_frames - 1, tag.end + int(cfg["search_after_s"] * fps))
    w = np.arange(lo, hi + 1)
    present = all_present[w]
    d = np.linalg.norm(ball[w] - centre, axis=1)
    step = np.r_[np.nan, np.linalg.norm(np.diff(ball[w], axis=0), axis=1)]
    print("\nhalf {} | window {} frames ({}s before tag start .. {}s after tag end), tag length {} frames".format(
        half, len(w), cfg["search_before_s"], cfg["search_after_s"], tag.end - tag.start))
    print("  ball present: {} frames ({:.0f}%), longest present run {}, longest gap {}".format(
        int(present.sum()), 100 * present.mean(), longest_run(present), longest_run(~present)))
    if present.any():
        i = int(np.nanargmin(d))
        print("  ball-to-centre distance (m): min {:.2f} at tag_start{:+d}, median {:.2f}".format(
            d[i], int(w[i] - tag.start), np.nanmedian(d)))
        for r in (1, 2, 3, 5, 10):
            near = np.isfinite(d) & (d <= r)
            print("    within {:>2} m: {:>4} frames, longest run {:>4}".format(r, int(near.sum()),
                                                                       longest_run(near)))
        still = np.isfinite(step) & (step < 0.1)
        print("  ball per-frame step < 0.1 m: {} frames, longest run {}; median step {:.3f} m".format(
            int(still.sum()), longest_run(still), np.nanmedian(step)))
    nvis_h = np.sum(np.isfinite(atd.xy[w][:, atd.team_of_slot == HOME, 0]), axis=1)
    nvis_a = np.sum(np.isfinite(atd.xy[w][:, atd.team_of_slot == AWAY, 0]), axis=1)
    print("  visible players: home median {:.0f} (min {}), away median {:.0f} (min {})".format(
        np.median(nvis_h), nvis_h.min(), np.median(nvis_a), nvis_a.min()))
    fr = np.array([_own_half_fraction(atd, t, _formation_direction(atd, t), cfg) for t in w])
    print("  own-half fraction (min over teams): max {:.2f}, frames >= {}: {}, longest run {}".format(
        np.nanmax(fr), cfg["own_half_frac"], int(np.sum(fr >= cfg["own_half_frac"])),
        longest_run(fr >= cfg["own_half_frac"])))
    with np.errstate(all="ignore"):
        hx = np.nanmean(atd.xy[w][:, atd.team_of_slot == HOME, 0], axis=1)
        ax = np.nanmean(atd.xy[w][:, atd.team_of_slot == AWAY, 0], axis=1)
    print("  team centroid x (normalised) at tag start: home {:.3f}, away {:.3f}; centroids on "
          "opposite sides of 0.5 on {:.0f}% of window".format(
              hx[tag.start - lo], ax[tag.start - lo], 100 * np.nanmean((hx - 0.5) * (ax - 0.5) < 0)))
    # Where in the window does the formation hold AND the ball sit near the centre?
    both = (fr >= cfg["own_half_frac"]) & np.isfinite(d) & (d <= 3)
    print("  frames with formation OK and ball within 3 m: {} (longest run {})".format(
        int(both.sum()), longest_run(both)))
