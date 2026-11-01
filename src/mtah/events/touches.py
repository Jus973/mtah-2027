"""Touch events from ball and player positions (PLAN.md §4 events/touches.py).

A contact frame has a player within r_touch_m of the (cleaned) ball. A contact episode is a run
of contact frames whose nearest team stays the same (gaps of <= bridge_frames bridged); a ball
flying past a player also makes one, so an episode counts as a touch only if the ball stays
close for >= min_frames, turns by >= min_turn_deg, or changes speed by >= min_dv_mps, or is
launched: it leaves at >= min_launch_mps from rest (slower than rest_mps) or from unseen. Launch
catches restart kicks, which the other tests miss: a placed or reappearing ball has no velocity
before the kick to turn or change. Positions are 2-d, so an aerial ball over a player is
indistinguishable from a touch except by those kinematic tests.
"""

from dataclasses import dataclass

import numpy as np

from mtah.features.ball import velocity


@dataclass
class Touch:
    start: int            # first contact frame
    end: int              # last contact frame (inclusive)
    team: int             # 0 = H, 1 = A
    n_frames: int
    d_min: float          # closest approach, metres
    speed_before: float   # m/s, k frames before start (NaN if unseen)
    speed_after: float    # m/s, k frames after end
    turn_deg: float       # heading change across the episode (NaN if unseen)


def contact_episodes(d, cfg):
    """[(start, end, team)] runs of contact frames with a constant nearest team."""
    near = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    team = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    contact = np.isfinite(near) & (near <= cfg["r_touch_m"])
    idx = np.flatnonzero(contact)
    episodes = []
    for t in idx:
        if episodes and episodes[-1][2] == team[t] and t - episodes[-1][1] - 1 <= cfg["bridge_frames"]:
            episodes[-1][1] = t
        else:
            episodes.append([t, t, int(team[t])])
    return [tuple(e) for e in episodes], near


def touch_events(ball, d, cfg, fps=25):
    """Touches in time order from the cleaned ball (T,2) and per-team distances d (T,2)."""
    episodes, near = contact_episodes(d, cfg)
    k = cfg["kin_offset_frames"]
    v = velocity(ball, k, fps)
    speed = np.linalg.norm(v, axis=1)
    T = len(ball)
    out = []
    for a, b, team in episodes:
        i, j = max(0, a - k), min(T - 1, b + k)
        vb, va = v[i], v[j]
        turn = np.nan
        if np.isfinite(vb).all() and np.isfinite(va).all() and np.linalg.norm(vb) > 0.5 \
                and np.linalg.norm(va) > 0.5:
            cos = np.dot(vb, va) / (np.linalg.norm(vb) * np.linalg.norm(va))
            turn = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        n = b - a + 1
        sb, sa = float(speed[i]), float(speed[j])
        launch = (np.isfinite(sa) and sa >= cfg["min_launch_mps"]
                  and not (np.isfinite(sb) and sb >= cfg["rest_mps"]))
        keep = (n >= cfg["min_frames"] or (np.isfinite(turn) and turn >= cfg["min_turn_deg"])
                or (np.isfinite(sb) and np.isfinite(sa) and abs(sa - sb) >= cfg["min_dv_mps"])
                or launch)
        if keep:
            out.append(Touch(start=int(a), end=int(b), team=team, n_frames=int(n),
                             d_min=float(np.min(near[a:b + 1])), speed_before=sb, speed_after=sa,
                             turn_deg=turn))
    return out
