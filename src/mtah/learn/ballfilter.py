"""Ball-track filters for building evaluation worlds that look like a filtered real feed (board #85).

Each filter runs inside every contiguous run of seen frames (NaN = missing) and leaves runs too
short for it unchanged. Families: "box" (centred moving average, width = param frames), "sg2"
(Savitzky-Golay, order 2, window = param frames), "bw4" (4th-order Butterworth low-pass, cutoff =
param Hz at 25 fps, applied forwards and backwards so it has zero phase). Evaluation only: the
models never train on a filtered world.
"""

import numpy as np

from mtah.learn import stats


def _per_run(ball, fn, minlen):
    out = ball.copy()
    seen = np.isfinite(ball[:, 0])
    for a, n in zip(*stats.runs(seen)):
        if n >= minlen:
            out[a:a + n] = fn(ball[a:a + n])
    return out


def apply(ball, family, param, fps=25):
    if family == "box":
        from mtah.learn.proxowner import smooth_ball
        return smooth_ball(ball, int(param))
    if family == "sg2":
        from scipy.signal import savgol_filter
        w = int(param)
        return _per_run(ball, lambda x: savgol_filter(x, w, 2, axis=0), w)
    if family == "bw4":
        from scipy.signal import butter, filtfilt
        b, a = butter(4, float(param) / (fps / 2))
        return _per_run(ball, lambda x: filtfilt(b, a, x, axis=0), 28)
    raise ValueError("unknown ball filter family: {}".format(family))


def apply_players(xy, family, param, fps=25):
    """The same filters on every player slot (T, S, 2), each slot's seen runs separately."""
    out = xy.copy()
    for s in range(xy.shape[1]):
        out[:, s] = apply(xy[:, s], family, param, fps)
    return out
