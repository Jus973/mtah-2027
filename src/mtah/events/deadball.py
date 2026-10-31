"""Dead-ball detection v1 (PLAN.md M1 events/deadball.py): D runs from tracking alone.

Evidence that the ball is dead, each a mask of frames:
  oob    the observed ball beyond a touch or goal line by more than oob_margin_m for at least
         oob_min_frames (the real ATD does not clamp the ball to the pitch);
  gap    the ball missing for at least gap_min_frames, the D onset moved gap_offset_frames from
         the gap start (the real ATD loses the ball mostly in stoppages: its players move half as
         fast while it is missing);
  still  ball slower than v_still_mps and players' mean speed below v_players_mps (both
         averaged over speed_window_frames) for at least still_min_frames. A goalkeeper holding
         the ball keeps the other players moving, which separates it from a stoppage.
Evidence runs closer than merge_frames merge. Each merged run is D from its first frame up to the
restart: the first touch that starts after the evidence ends (the half's end if none).
"""

import warnings

import numpy as np

from mtah.features.ball import velocity
from mtah.learn.stats import runs

KINDS = ("oob", "gap", "still")


def _rolling_mean(x, w):
    """Centred nan-aware mean over w frames."""
    ok = np.isfinite(x)
    k = np.ones(w)
    num = np.convolve(np.where(ok, x, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _long_runs(mask, min_len):
    out = np.zeros_like(mask)
    for a, n in zip(*runs(mask)):
        if n >= min_len:
            out[a:a + n] = True
    return out


def evidence(ball_raw, ball, observed, players, graded, pitch, cfg, fps=25):
    """{kind: (T,) bool} dead-ball evidence on graded frames. ball_raw / observed: before
    cleaning (an out-of-bounds ball is real evidence, not an outlier); ball: cleaned."""
    T = len(ball)
    m = cfg["oob_margin_m"]
    with np.errstate(invalid="ignore"):
        out = observed & ((ball_raw[:, 0] < -m) | (ball_raw[:, 0] > pitch[0] + m)
                          | (ball_raw[:, 1] < -m) | (ball_raw[:, 1] > pitch[1] + m))
    oob = _long_runs(out & graded, cfg["oob_min_frames"])

    gap = np.zeros(T, dtype=bool)
    off = cfg["gap_offset_frames"]
    for a, n in zip(*runs(~np.isfinite(ball[:, 0]) & graded)):
        if n >= cfg["gap_min_frames"]:
            gap[max(0, a + off):a + n] = True
    gap &= graded

    w = cfg["speed_window_frames"]
    ball_speed = _rolling_mean(np.linalg.norm(velocity(ball, 2, fps), axis=1), w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.full(players.shape[:2], np.nan)
        v[1:] = np.linalg.norm(np.diff(players, axis=0), axis=-1) * fps
        crowd_speed = _rolling_mean(np.nanmean(v, axis=1), w)
    with np.errstate(invalid="ignore"):
        calm = (ball_speed < cfg["v_still_mps"]) & (crowd_speed < cfg["v_players_mps"])
    still = _long_runs(calm & graded, cfg["still_min_frames"])
    return {"oob": oob, "gap": gap, "still": still}


def dead_runs(ev, touches, graded, cfg, kinds=KINDS):
    """[(start, end_inclusive, kinds)] D runs from the evidence kinds used, ended at the restart."""
    any_ev = np.zeros(len(graded), dtype=bool)
    for k in kinds:
        any_ev |= ev[k]
    starts = np.array([t.start for t in touches], dtype=np.int64)
    out = []
    for half_start, half_len in zip(*runs(graded)):
        half_end = half_start + half_len - 1
        sub = any_ev[half_start:half_end + 1]
        merged = []
        for a, n in zip(*runs(sub)):
            a, b = a + half_start, a + half_start + n - 1
            if merged and a - merged[-1][1] - 1 <= cfg["merge_frames"]:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            i = np.searchsorted(starts, b + 1)
            restart = starts[i] if i < len(starts) and starts[i] <= half_end else half_end + 1
            end = restart - 1
            if out and a <= out[-1][1]:          # previous run's restart lies past this start
                out[-1] = (out[-1][0], max(out[-1][1], end), out[-1][2])
                continue
            used = tuple(k for k in kinds if ev[k][a:b + 1].any())
            out.append((int(a), int(end), used))
    return out


def dead_mask(ev, touches, graded, cfg, kinds=KINDS):
    mask = np.zeros(len(graded), dtype=bool)
    for a, b, _ in dead_runs(ev, touches, graded, cfg, kinds):
        mask[a:b + 1] = True
    return mask & graded
