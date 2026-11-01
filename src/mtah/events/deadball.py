"""Dead-ball detection v1 (PLAN.md M1 events/deadball.py): D runs from tracking alone.

Evidence that the ball is dead, each a mask of frames:
  oob    the observed ball beyond a touch or goal line by more than oob_margin_m for at least
         oob_min_frames (the real ATD does not clamp the ball to the pitch);
  gap    the ball missing for at least gap_min_frames, the D onset moved gap_offset_frames from
         the gap start (the real ATD loses the ball mostly in stoppages: its players move half as
         fast while it is missing). Only the calm part counts: frames where players' mean speed,
         averaged over gap_speed_window_frames, is below gap_v_players_mps, in runs of at least
         gap_calm_min_frames. That drops gaps in open play and trims a gap that starts in play;
  still  ball slower than v_still_mps and players' mean speed below v_players_mps (both
         averaged over speed_window_frames) for at least still_min_frames. A goalkeeper holding
         the ball keeps the other players moving, which separates it from a stoppage.
Evidence runs closer than merge_frames merge. Each merged run is D from its first frame up to the
restart, the earliest of:
  - the first touch that starts after the evidence ends (the half's end if none);
  - a release: the ball leaves a player whose touch started inside the evidence and ended at most
    restart_window_frames before it ends (null = off). A throw-in or a corner is taken while the
    ball is still beyond the line, so the taker's touch starts inside the oob evidence; the
    restart is the frame after the taker's last contact, and the taker's team owns the ball;
  - a flight: the evidence ends in a long ball gap after which the ball reappears moving at
    >= restart_flight_mps (the restart happened while the tracker had lost the ball).
"""

import warnings

import numpy as np

from mtah.events.touches import Touch
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


def players_speed(players, fps=25):
    """(T,) players' mean one-frame speed, m/s (NaN where no player is tracked on both frames)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.full(players.shape[:2], np.nan)
        v[1:] = np.linalg.norm(np.diff(players, axis=0), axis=-1) * fps
        return np.nanmean(v, axis=1)


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
    flight = np.zeros(T, dtype=bool)     # every frame of a long gap the ball leaves in flight
    off = cfg["gap_offset_frames"]
    speed = np.linalg.norm(velocity(ball, 2, fps), axis=1)
    for a, n in zip(*runs(~np.isfinite(ball[:, 0]) & graded)):
        if n >= cfg["gap_min_frames"]:
            gap[max(0, a + off):a + n] = True
            r = a + n + 2                # velocity over +/- 2 frames from the first frame seen
            if r < T and speed[r] >= cfg["restart_flight_mps"]:
                flight[a:a + n] = True
    gap &= graded

    frame_speed = players_speed(players, fps)
    with np.errstate(invalid="ignore"):
        calm_gap = gap & (_rolling_mean(frame_speed, cfg["gap_speed_window_frames"])
                          < cfg["gap_v_players_mps"])
    gap = _long_runs(calm_gap, cfg["gap_calm_min_frames"])

    w = cfg["speed_window_frames"]
    ball_speed = _rolling_mean(np.linalg.norm(velocity(ball, 2, fps), axis=1), w)
    crowd_speed = _rolling_mean(frame_speed, w)
    with np.errstate(invalid="ignore"):
        calm = (ball_speed < cfg["v_still_mps"]) & (crowd_speed < cfg["v_players_mps"])
    still = _long_runs(calm & graded, cfg["still_min_frames"])
    return {"oob": oob, "gap": gap, "still": still, "flight": flight}


def dead_runs(ev, touches, graded, cfg, kinds=KINDS, out_restarts=None):
    """[(start, end_inclusive, kinds)] D runs from the evidence kinds used, ended at the restart.

    out_restarts: if a list, each release used as a restart is appended to it as a one-frame
    Touch by the taker's team, for the owner decoder."""
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
        W = cfg["restart_window_frames"]
        flight = ev.get("flight")
        for a, b in merged:
            i = np.searchsorted(starts, b + 1)
            restart = starts[i] if i < len(starts) and starts[i] <= half_end else half_end + 1
            if flight is not None and flight[b]:
                g = b
                while g + 1 <= half_end and flight[g + 1]:
                    g += 1
                restart = min(restart, g + 1)
            if W is not None:
                inside = [t for t in touches[np.searchsorted(starts, a):i] if t.end >= b - W]
                if inside and inside[-1].end + 1 < restart:
                    t = inside[-1]
                    restart = t.end + 1
                    if out_restarts is not None:
                        out_restarts.append(Touch(start=restart, end=restart, team=t.team,
                                                  n_frames=1, d_min=t.d_min,
                                                  speed_before=np.nan,
                                                  speed_after=t.speed_after, turn_deg=np.nan))
            end = restart - 1
            if out and a <= out[-1][1]:          # previous run's restart lies past this start
                out[-1] = (out[-1][0], max(out[-1][1], end), out[-1][2])
                continue
            used = tuple(k for k in kinds if ev[k][a:b + 1].any())
            out.append((int(a), int(end), used))
    return out


def dead_mask(ev, touches, graded, cfg, kinds=KINDS, out_restarts=None):
    mask = np.zeros(len(graded), dtype=bool)
    for a, b, _ in dead_runs(ev, touches, graded, cfg, kinds, out_restarts):
        mask[a:b + 1] = True
    return mask & graded
