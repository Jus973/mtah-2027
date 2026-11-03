"""Collective / match-context features. M0: attacking direction per half."""

import numpy as np

from mtah.io.atd import AWAY, HOME


def team_centroid_x(atd, frames, team):
    """Median over `frames` of the team's mean normalised x (NaN-aware)."""
    cols = atd.team_of_slot == team
    x = atd.xy[frames][:, cols, 0]
    with np.errstate(all="ignore"):
        per_frame = np.nanmean(x, axis=1)
    return float(np.nanmedian(per_frame))


def attacking_direction(atd, kickoffs, window_frames=125):
    """{half: {"H": +1|-1, "A": -1|+1}} from team centroids just before/after each kick-off.

    +1 means attacking towards x = 1. At a kick-off each team is in its own half, so the team whose
    centroid sits at lower x attacks +x. If a team has no visible player in the window, the
    window after the kick-off is widened (2x, 4x, 8x). Raises if the teams still can't be
    separated.
    """
    out = {}
    for half, kick in kickoffs.items():
        for scale in (1, 2, 4, 8):           # a whole team may be untracked at the kick-off
            frames = np.arange(max(0, kick - window_frames // 5), kick + scale * window_frames)
            hx = team_centroid_x(atd, frames, HOME)
            ax = team_centroid_x(atd, frames, AWAY)
            if np.isfinite(hx) and np.isfinite(ax):
                break
        if not (np.isfinite(hx) and np.isfinite(ax)) or abs(hx - ax) < 0.05:
            raise ValueError("half {}: cannot separate team centroids ({:.3f} vs {:.3f})".format(
                half, hx, ax))
        home = 1 if hx < ax else -1
        out[half] = {"H": home, "A": -home}
    return out
