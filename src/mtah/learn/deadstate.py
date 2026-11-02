"""Learned per-frame dead ball (v8), trained on the sim only, where the dead-time ball behaves as
a real tracker reports it.

The DFL source data freezes the ball whenever play is dead: on the sim 20.5-22.0% of seen ball
frames do not move at all, against 0.08% on the real ATD. A model trained on that sim learns
"a ball that never moves is dead", which no real match shows. So the training matches are first
given a realistic dead-time ball (realistic_dead_ball): a frozen ball gets tracker noise
(deadstate.jitter_m), or is fetched by the first player within pickup_radius_m and carried, or is
lost, as the real tracker mostly does in stoppages. The model reads team shape (gapstate
features, both perspectives, averaged) and the ball's position and speed, never whether the ball
is seen. Frames are then decoded as a 2-state path (decode/dead.py). It replaces the evidence-based
D and the gap decoder's dead model. Single-threaded and seeded.
"""

import copy
import os
import warnings

import numpy as np

from mtah import registry
from mtah.features.ball import velocity
from mtah.learn.stats import runs

BALL = ("speed", "speed_-25", "speed_+25", "speed_-50", "speed_+50", "line_m", "line_m_1s",
        "centre_x", "touchline_m")


def _roll(a, w):
    """Centred NaN-aware rolling mean over w frames."""
    ok = np.isfinite(a)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, a, 0.0))])
    n = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(len(a))
    lo, hi = np.clip(i - w // 2, 0, len(a)), np.clip(i + w // 2 + 1, 0, len(a))
    cnt = n[hi] - n[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _shift(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    return out


def ball_features(ball_raw, ball, pitch, fps=25):
    """(T, len(BALL)): rolling ball speed now and 1 s / 2 s either side, distance to the nearest
    line (negative outside), its 1 s mean, distance from the halfway line, to the touchline."""
    L, W = pitch
    sp = _roll(np.linalg.norm(velocity(ball, 2, fps), axis=1), 25)
    x, y = ball_raw[:, 0].astype(float), ball_raw[:, 1].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        line = np.fmin(np.fmin(x, L - x), np.fmin(y, W - y))
        return np.column_stack([sp, _shift(sp, -25), _shift(sp, 25), _shift(sp, -50),
                                _shift(sp, 50), line, _roll(line, 25),
                                np.abs(x - L / 2) / (L / 2), np.minimum(y, W - y)])


def features(F, B):
    """Both perspectives' rows: team shape (gapstate) plus the ball."""
    return np.hstack([F[0], B]), np.hstack([F[1], B])


def feature_names(gap_names):
    return list(gap_names) + list(BALL)


def causal_ball(B):
    """B without the future-speed columns (speed_+25, speed_+50): the P3e hybrid never reads
    the ball's future - v8's likely wholesale-shift mechanism (handoff/REPLY.md)."""
    keep = [i for i, n in enumerate(BALL) if not n.startswith("speed_+")]
    return B[:, keep]


def causal_names(gap_names):
    return list(gap_names) + [n for n in BALL if not n.startswith("speed_+")]


def realistic_dead_ball(atd, kind, cfg, seed):
    """A copy of a sim atd whose frozen (dead-time) ball is made realistic. kind: "jitter" (tracker
    noise of cfg jitter_m), "pickup" (fetched by the first player within pickup_radius_m, then
    carried with carry_noise_m, then jitter on what is still frozen) or "lost" (not seen)."""
    b = atd.ball_m().astype(float)
    step = np.r_[np.nan, np.linalg.norm(np.diff(b, axis=0), axis=1)]
    frozen = (step == 0) & np.isfinite(b[:, 0])
    rng = np.random.default_rng(seed)
    out = copy.copy(atd)
    out.ball = atd.ball.copy()
    nb = b.copy()
    if kind == "lost":
        out.ball[frozen] = np.nan
        return out
    still = frozen.copy()
    if kind == "pickup":
        players = atd.players_m()[..., :2].astype(float)
        for a, n in zip(*runs(frozen)):
            a = max(a - 1, 0)
            d = np.linalg.norm(players[a:a + n + 1] - b[a][None, None, :], axis=-1)
            dd = np.where(np.isfinite(d), d, np.inf)
            close = dd.min(axis=1) <= cfg["pickup_radius_m"]
            if not close.any():
                continue
            t0 = int(np.argmax(close))
            j = int(np.argmin(dd[t0]))
            seg = slice(a + t0, a + n + 1)
            track = players[seg, j].copy()
            track[~np.isfinite(track[:, 0])] = b[a]
            nb[seg] = track + rng.normal(0, cfg["carry_noise_m"], size=track.shape)
            still[seg] = False
    idx = np.flatnonzero(still)
    nb[idx] += rng.normal(0, cfg["jitter_m"], size=(len(idx), 2))
    L, W = atd.pitch
    out.ball[:, 0], out.ball[:, 1] = nb[:, 0] / L, nb[:, 1] / W
    out.ball[~np.isfinite(b[:, 0])] = np.nan
    return out


def training_rows(FD, gt, graded, step):
    """(X, y) from every step-th graded frame, both perspectives: is the ball dead."""
    r = np.flatnonzero(graded)[::step]
    return np.vstack([FD[0][r], FD[1][r]]), np.tile(gt[r] == "D", 2).astype(float)


def train(X, y, params, names):
    import lightgbm as lgb
    p = {k: v for k, v in params.items() if k != "num_boost_round"}
    p.update(objective="binary", num_threads=1, deterministic=True, verbose=-1)
    return lgb.train(p, lgb.Dataset(X, y, feature_name=list(names)),
                     num_boost_round=params["num_boost_round"])


def model_path(cfg):
    return os.path.join(registry.ROOT, cfg["deadstate"]["model"])


def load(cfg):
    import lightgbm as lgb
    return lgb.Booster(model_file=model_path(cfg))


def predict(model, FD, idx):
    """P(dead) at frames idx, the mean of the two perspectives."""
    if len(idx) == 0:
        return np.zeros(0)
    return (model.predict(FD[0][idx], num_threads=1) + model.predict(FD[1][idx], num_threads=1)) / 2
