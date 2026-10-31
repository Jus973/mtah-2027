"""Restart events. M0: kick-off detection from tags + team formation + ball near the centre.

A kick-off frame is the frame the ball is struck. The tracked ball is too jittery to show a clean
rest on the spot (measured on the real ATD: never closer than ~1.4 m, ~0.3 m per-frame noise), so
the formation is the primary signal: both teams in their own halves, a state that breaks shortly
after the kick. The ball only refines the frame inside a short window around that break.
Nothing here hard-codes the published frames (11,185 / 113,742); scripts compare against them
with a tolerance and report the offset.
"""

import warnings

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a team with no visible player
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


def _runs(mask, max_gap=0):
    """[(start, stop_inclusive)] of True runs in mask, bridging False gaps of <= max_gap frames."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    runs = []
    a = b = idx[0]
    for i in idx[1:]:
        if i - b - 1 <= max_gap:
            b = i
        else:
            runs.append((a, b))
            a = b = i
    runs.append((a, b))
    return runs


def detect_kickoffs(atd, tag_list, cfg, fps=25):
    """{half: dict(frame, method, formation_start, formation_end, ...)} per Kick Off (Start) tag.

    1. Inside the tag window (+/- search margins) mark frames where, for both teams, at least
       own_half_frac of visible players are in their own half (gaps <= gap_fill_frames bridged).
       The fraction is below 1 because the two kick-takers stand on the halfway line.
    2. Each such run of >= min_formation_frames is a line-up candidate; its last frame is the
       formation break, which follows the kick. Open play can also leave both blocks in their own
       halves, so candidates are ranked by: ball seen near the centre around the break, then
       distance of the break from the tag interval, then run length.
    3. Within [break - ball_search_before_s, break + ball_search_after_s], take the run of frames
       with the ball within loose_radius_m of the centre that holds its closest approach; the kick
       is the last frame of that run within rest_slack_m of that minimum (the ball still at its
       resting spot). Without such ball frames, fall back to the formation break.
    """
    length, width = atd.pitch
    centre = np.array([length / 2.0, width / 2.0])
    ball = atd.ball_m()
    dist = np.linalg.norm(ball - centre, axis=1)
    result = {}
    for half, tag in sorted(kickoff_tags(tag_list).items()):
        lo = max(0, tag.start - int(cfg["search_before_s"] * fps))
        hi = min(atd.n_frames - 1, tag.end + int(cfg["search_after_s"] * fps))
        frac = np.array([_own_half_fraction(atd, t, _formation_direction(atd, t), cfg)
                         for t in range(lo, hi + 1)])
        runs = [(a, b) for a, b in _runs(frac >= cfg["own_half_frac"], cfg["gap_fill_frames"])
                if b - a + 1 >= cfg["min_formation_frames"]]
        row = {"tag_start": tag.start, "tag_end": tag.end}
        if not runs:
            row.update({"frame": None, "method": "not_found", "max_own_half_frac": float(frac.max())})
            result[half] = row
            continue
        def ball_window(f_end):
            s_lo = max(lo, f_end - int(cfg["ball_search_before_s"] * fps))
            s_hi = min(hi, f_end + int(cfg["ball_search_after_s"] * fps))
            d = dist[s_lo:s_hi + 1]
            return s_lo, d, np.isfinite(d) & (d <= cfg["loose_radius_m"])

        def rank(run):
            f_end = lo + run[1]
            _, _, near = ball_window(f_end)
            off_tag = max(0, tag.start - f_end, f_end - tag.end)
            return (not near.any(), off_tag, -(run[1] - run[0]))

        a, b = min(runs, key=rank)
        f_start, f_end = lo + a, lo + b
        s_lo, d, near = ball_window(f_end)
        frame, method, n_near = f_end, "formation_break", int(near.sum())
        if near.any():
            # The near-centre run (unseen frames bridged) holding the ball's closest approach to
            # the spot: play passing back through midfield later is farther from it.
            seen = np.isfinite(d)
            closest = int(np.nanargmin(np.where(near, d, np.nan)))
            first = last = closest
            while first - 1 >= 0 and (near[first - 1] or not seen[first - 1]):
                first -= 1
            while last + 1 < len(d) and (near[last + 1] or not seen[last + 1]):
                last += 1
            seg = np.arange(first, last + 1)
            seg = seg[near[seg]]
            d_min = d[seg].min()
            frame = s_lo + int(seg[d[seg] <= d_min + cfg["rest_slack_m"]][-1])
            method = "formation+ball"
        row.update({"frame": int(frame), "method": method, "formation_start": int(f_start),
                    "formation_end": int(f_end), "formation_frames": int(f_end - f_start + 1),
                    "ball_near_frames": n_near, "kick_minus_break": int(frame - f_end),
                    "n_candidates": len(runs)})
        result[half] = row
    return result
