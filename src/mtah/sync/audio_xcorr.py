"""Broadcast -> tactical timeline sync from audio (PLAN.md M2 T3).

Both videos carry stadium sound. Per broadcast window, the offset to tactical is the lag that
maximises the normalised cross-correlation of a high-passed audio feature (loudness or
whistle-band energy, one value per video frame). Offsets are piecewise constant (cuts and
replays break them), so each window gets its own; windows whose peak is weak or not unique are
left unsynced. A sync is only trusted where the two feeds' tracking agrees (validate()).
"""

import numpy as np


def highpass(x, w):
    """x minus its centred w-frame moving average; NaN -> 0 (no audio = no information)."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    k = np.ones(w) / w
    return x - np.convolve(x, k, mode="same")


def _xcorr_valid(ref, q):
    """Normalised correlation of q (length W, zero mean, unit norm) at every start of ref."""
    W = len(q)
    n = len(ref) + W
    size = 1 << int(np.ceil(np.log2(n)))
    num = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(q, size)), size)[:len(ref) - W + 1]
    c1 = np.concatenate([[0.0], np.cumsum(ref)])
    c2 = np.concatenate([[0.0], np.cumsum(ref ** 2)])
    s1 = c1[W:] - c1[:-W]
    s2 = c2[W:] - c2[:-W]
    var = np.maximum(s2 - s1 ** 2 / W, 1e-12)
    return num / np.sqrt(var)


def window_offsets(ref, query, W, hop, exclude):
    """For broadcast windows query[s:s+W] (s every hop): (starts, offsets, scores, ratios), where
    offset = best ref start - s, score = peak correlation and ratio = peak / best peak at least
    `exclude` frames away."""
    starts, offsets, scores, ratios = [], [], [], []
    for s in range(0, len(query) - W + 1, hop):
        q = query[s:s + W] - query[s:s + W].mean()
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            continue
        c = _xcorr_valid(ref, q / norm)
        best = int(np.argmax(c))
        masked = c.copy()
        masked[max(0, best - exclude):best + exclude + 1] = -np.inf
        second = float(np.max(masked)) if np.isfinite(masked).any() else 0.0
        starts.append(s)
        offsets.append(best - s)
        scores.append(float(c[best]))
        ratios.append(float(c[best] / second) if second > 0 else np.inf)
    return (np.array(starts), np.array(offsets), np.array(scores), np.array(ratios))


def validate(ball_t, ball_b, offset, b_frames, tol_m):
    """Share of broadcast frames b in b_frames (tactical frame b + offset in range) where both
    feeds see the ball and agree within tol_m metres, and the number of such frames."""
    t = b_frames + offset
    ok = (t >= 0) & (t < len(ball_t))
    b, t = b_frames[ok], t[ok]
    both = np.isfinite(ball_t[t, 0]) & np.isfinite(ball_b[b, 0])
    if not both.any():
        return None, 0
    d = np.linalg.norm(ball_t[t[both]] - ball_b[b[both]], axis=1)
    return float(np.mean(d <= tol_m)), int(both.sum())
