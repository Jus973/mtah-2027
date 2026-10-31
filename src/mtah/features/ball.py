"""Ball cleaning and kinematics (PLAN.md §4). Metres, origin at a pitch corner, NaN = missing.

The real ATD does not clamp the ball to the pitch (it is seen up to ~33 m off it) and never jumps
more than 5 m between consecutive frames; the sim has neither artefact. Cleaning drops positions
far outside the pitch and implausible speeds, then fills short gaps linearly so that a touch is
not lost to a one-frame dropout. Long gaps stay missing: the gap policy belongs to the decoder.
"""

import numpy as np

from mtah.learn.stats import runs


def clean_ball(ball, pitch, cfg, fps=25):
    """(ball (T,2), observed (T,) bool, filled (T,) bool): observed = kept raw positions,
    filled = short gaps interpolated."""
    ball = ball.copy()
    m = cfg["outside_m"]
    with np.errstate(invalid="ignore"):
        far = ((ball[:, 0] < -m) | (ball[:, 0] > pitch[0] + m)
               | (ball[:, 1] < -m) | (ball[:, 1] > pitch[1] + m))
    ball[far] = np.nan
    step = np.full(len(ball), np.nan)
    step[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    ball[step > cfg["v_max_mps"]] = np.nan
    observed = np.isfinite(ball[:, 0])
    filled = np.zeros(len(ball), dtype=bool)
    for a, n in zip(*runs(~observed)):
        if a == 0 or a + n >= len(ball) or n > cfg["max_interp_gap"]:
            continue
        w = np.arange(1, n + 1) / (n + 1)
        ball[a:a + n] = ball[a - 1] + w[:, None] * (ball[a + n] - ball[a - 1])
        filled[a:a + n] = True
    return ball, observed, filled


def velocity(ball, k, fps=25):
    """(T, 2) m/s, centred difference over +/- k frames (NaN where either end is missing)."""
    v = np.full(ball.shape, np.nan)
    v[k:-k] = (ball[2 * k:] - ball[:-2 * k]) * fps / (2 * k)
    return v
