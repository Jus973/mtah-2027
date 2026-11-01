"""Broadcast -> tactical sync from tracking (PLAN.md M2 T3 fallback when audio fails).

Per broadcast window, the offset to tactical is the lag at which the two feeds' ball tracks
agree best: the median distance between the balls over frames where both see it. A coarse search
over every lag at `step` frames is refined at full rate around the best coarse lag. The window
is synced only if enough frames overlap and the median distance is small (validate() in
audio_xcorr then applies the PLAN.md acceptance test).
"""

import numpy as np


def _median_dist(ref, q, lags):
    """(len(lags),) median distance between q (W,2) and ref[lag:lag+W] where both are finite;
    and the count of such frames."""
    W = len(q)
    idx = lags[:, None] + np.arange(W)[None, :]
    r = ref[idx]                                   # (L, W, 2)
    d = np.linalg.norm(r - q[None], axis=2)
    ok = np.isfinite(d)
    n = ok.sum(axis=1)
    d = np.where(ok, d, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(n[:, None] > 0, d, 0.0), axis=1)
    return np.where(n > 0, med, np.inf), n


def window_offsets(ball_t, ball_b, W, hop, step, min_frames, block=4000):
    """(starts, offsets, median distance, overlapping frames) for broadcast windows."""
    T = len(ball_t)
    coarse_t = ball_t[::step]
    out = [[], [], [], []]
    for s in range(0, len(ball_b) - W + 1, hop):
        q = ball_b[s:s + W]
        if np.isfinite(q[:, 0]).sum() < min_frames:
            continue
        qc = q[::step]
        lags = np.arange(0, len(coarse_t) - len(qc) + 1)
        best, best_d = None, np.inf
        for a in range(0, len(lags), block):
            med, n = _median_dist(coarse_t, qc, lags[a:a + block])
            med = np.where(n >= max(3, min_frames // step), med, np.inf)
            j = int(np.argmin(med))
            if med[j] < best_d:
                best, best_d = lags[a + j] * step, med[j]
        if best is None or not np.isfinite(best_d):
            continue
        fine = np.arange(max(0, best - step), min(T - W, best + step) + 1)
        med, n = _median_dist(ball_t, q, fine)
        med = np.where(n >= min_frames, med, np.inf)
        j = int(np.argmin(med))
        if not np.isfinite(med[j]):
            continue
        out[0].append(s)
        out[1].append(int(fine[j]) - s)
        out[2].append(float(med[j]))
        out[3].append(int(n[j]))
    return tuple(np.array(x) for x in out)
