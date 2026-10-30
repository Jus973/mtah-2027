"""Restart events. M0: kick-off detection from tags + formation + ball at the centre spot.

A kick-off frame is the frame the ball is struck: the last frame it rests on the spot before it
moves. Nothing here hard-codes the published frames (11,185 / 113,742); scripts assert them.
"""

import numpy as np

from mtah.io import tags as tagio
from mtah.io.atd import AWAY, HOME


def _own_half_fraction(atd, t, direction, cfg):
    """Min over teams of the fraction of visible players in their own half, at frame t."""
    length = atd.pitch[0]
    tol = cfg["own_half_tol_m"] / length
    fracs = []
    for team, side in ((HOME, "H"), (AWAY, "A")):
        x = atd.xy[t, atd.team_of_slot == team, 0]
        x = x[np.isfinite(x)]
        if not len(x):
            return 0.0
        own = x <= 0.5 + tol if direction[side] > 0 else x >= 0.5 - tol
        fracs.append(float(np.mean(own)))
    return min(fracs)


def _formation_direction(atd, t):
    """Direction guess from centroids at frame t: the lower-x team attacks +x."""
    with np.errstate(all="ignore"):
        hx = np.nanmean(atd.xy[t, atd.team_of_slot == HOME, 0])
        ax = np.nanmean(atd.xy[t, atd.team_of_slot == AWAY, 0])
    home = 1 if hx < ax else -1
    return {"H": home, "A": -home}


def kickoff_tags(tag_list):
    out = {}
    for tag in tag_list:
        if tag.is_code("SET PIECES") and tag.is_type("Kick Off (Start)"):
            half = tagio.half_number(tag)
            if half is not None and half not in out:
                out[half] = tag
    return out


def detect_kickoffs(atd, tag_list, cfg, fps=25):
    """{half: dict(frame, method, rest_frames, own_half_frac, tag_start, tag_end)}.

    Within the tag window (+/- search margins) find frames where the ball sits within
    centre_radius_m of the centre spot; take a run of >= min_rest_frames such frames whose last
    frame has both teams in their own halves; the kick-off is the last frame of that run before the
    ball moves more than ball_moved_m from its resting position.
    """
    length, width = atd.pitch
    centre = np.array([length / 2.0, width / 2.0])
    ball = atd.ball_m()
    result = {}
    for half, tag in sorted(kickoff_tags(tag_list).items()):
        lo = max(0, tag.start - int(cfg["search_before_s"] * fps))
        hi = min(atd.n_frames - 1, tag.end + int(cfg["search_after_s"] * fps))
        d = np.linalg.norm(ball[lo:hi + 1] - centre, axis=1)
        at_spot = np.isfinite(d) & (d <= cfg["centre_radius_m"])
        found = None
        i = 0
        n = len(at_spot)
        while i < n:
            if not at_spot[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and at_spot[j + 1]:
                j += 1
            run_lo, run_hi = lo + i, lo + j
            if run_hi - run_lo + 1 >= cfg["min_rest_frames"]:
                rest = np.nanmedian(ball[run_lo:run_hi + 1], axis=0)
                k = run_lo
                while k + 1 <= hi and np.isfinite(ball[k + 1, 0]) and \
                        np.linalg.norm(ball[k + 1] - rest) <= cfg["ball_moved_m"]:
                    k += 1
                direction = _formation_direction(atd, k)
                frac = _own_half_fraction(atd, k, direction, cfg)
                if frac >= cfg["own_half_frac"]:
                    found = {"frame": int(k), "method": "ball_rest_then_move",
                             "rest_frames": int(k - run_lo + 1), "own_half_frac": frac}
                    break
            i = j + 1
        if found is None:
            found = {"frame": None, "method": "not_found"}
        found.update({"tag_start": tag.start, "tag_end": tag.end})
        result[half] = found
    return result
