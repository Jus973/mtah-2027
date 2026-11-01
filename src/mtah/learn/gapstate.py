"""Ball-free state inside long ball gaps (PLAN.md M2 T4 per-frame emissions, restricted to the
gaps), trained on the sim only.

The real ATD loses the ball for long stretches (149 gaps, median 268 frames). v3 calls every gap
of at least gaps.min_frames dead, but on the sim a fifth of long-gap frames are alive, and the
owner changes inside them without a single contact. Two LightGBM models read the players alone:
P(dead) per gap frame, and P(H owns | alive). The features are team shape and motion in the
perspective team's attacking direction (centroid, length, width, deepest and highest player,
players in the opponent half, mean speed, forward drift), now and gaps.context_frames either side
(offline context). They use no gap timing and no contacts: on the sim those added 0.003 held-out
spellF1 (< 0.01) and would tie the model to where the sim places its gaps. Team-symmetric: each
frame is a row from both teams' perspectives (owner label flipped), and predictions average the
two. Single-threaded and seeded, so training and prediction are deterministic.
"""

import os
import warnings

import numpy as np

from mtah import registry
from mtah.features.context import attacking_direction
from mtah.learn.stats import runs

SERIES = ("cu", "su", "sy", "umin", "umax", "nhigh", "n", "speed", "vu")
SHIFTED = ("cu", "vu", "speed")


def feature_names(cfg):
    names = ["x_" + s for s in SERIES] + ["y_" + s for s in SERIES] + ["dcu", "dspeed"]
    for s in SHIFTED:
        for k in cfg["context_frames"]:
            for off in (-k, k):
                names += ["x_{}_{:+d}".format(s, off), "y_{}_{:+d}".format(s, off)]
    return names


def home_direction(atd, sp, kicks):
    """(T,) +1 where the home team attacks towards x = 1, -1 where it attacks x = 0, 0 outside
    the graded halves. Each half from its kick-off, or its first graded frame (sim)."""
    out = np.zeros(sp["n_frames"])
    with warnings.catch_warnings():          # frames where a team has no visible player
        warnings.simplefilter("ignore", RuntimeWarning)
        found = attacking_direction(atd, {h: kicks.get(h, first)
                                          for h, first, _ in sp["half_blocks"]})
    for h, first, last in sp["half_blocks"]:
        out[first:last + 1] = found[h]["H"]
    return out


def long_gaps(observed, graded, min_frames):
    """(T,) bool: graded frames inside a run of unobserved ball at least min_frames long."""
    gap = np.zeros(len(graded), dtype=bool)
    for a, n in zip(*runs(~observed & graded)):
        if n >= min_frames:
            gap[a:a + n] = True
    return gap


def _shift(a, k):
    """a[t + k] at t (NaN past the ends)."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _team(players, cols, sign, length, fps, v_frames):
    x, y = players[:, cols, 0], players[:, cols, 1]
    u = np.where(sign[:, None] > 0, x, length - x)
    s = {"cu": np.nanmean(u, axis=1), "su": np.nanstd(u, axis=1), "sy": np.nanstd(y, axis=1),
         "umin": np.nanmin(u, axis=1), "umax": np.nanmax(u, axis=1),
         "nhigh": np.nansum(u > length / 2, axis=1).astype(float),
         "n": np.isfinite(u).sum(axis=1).astype(float)}
    step = np.linalg.norm(players[2:, cols] - players[:-2, cols], axis=-1) * fps / 2
    speed = np.full(len(u), np.nan)
    speed[1:-1] = np.nanmean(step, axis=1)
    s["speed"] = speed
    s["vu"] = (_shift(s["cu"], v_frames) - _shift(s["cu"], -v_frames)) * fps / (2 * v_frames)
    return s


def features(players, team_of_slot, direction, pitch, cfg, fps=25):
    """(F_home, F_away): (T, n_features) from each team's perspective. players: (T, S, 2) metres;
    direction: home_direction(); cfg: the gaps section."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for persp in (0, 1):
            sign = direction if persp == 0 else -direction
            X = _team(players, team_of_slot == persp, sign, pitch[0], fps, cfg["velocity_frames"])
            Y = _team(players, team_of_slot == 1 - persp, sign, pitch[0], fps,
                      cfg["velocity_frames"])
            cols = [X[s] for s in SERIES] + [Y[s] for s in SERIES]
            cols += [X["cu"] - Y["cu"], X["speed"] - Y["speed"]]
            for s in SHIFTED:
                for k in cfg["context_frames"]:
                    for off in (-k, k):
                        cols += [_shift(X[s], off), _shift(Y[s], off)]
            out.append(np.column_stack(cols))
    return out[0], out[1]


def training_rows(F, gap, gt, step):
    """{"dead": (X, y), "owner": (X, y)} from every step-th gap frame, both perspectives. dead:
    GT is D; owner (alive frames only): the perspective team owns the ball."""
    idx = np.flatnonzero(gap)[::step]
    alive = np.flatnonzero(gap & np.isin(gt, ["H", "A"]))[::step]
    dead = (np.vstack([F[0][idx], F[1][idx]]), np.tile(gt[idx] == "D", 2).astype(float))
    owner = (np.vstack([F[0][alive], F[1][alive]]),
             np.concatenate([gt[alive] == "H", gt[alive] == "A"]).astype(float))
    return {"dead": dead, "owner": owner}


def train(X, y, params, names):
    """A LightGBM booster; params: the gap_model section of the config."""
    import lightgbm as lgb
    p = {k: v for k, v in params.items() if k != "num_boost_round"}
    p.update(objective="binary", num_threads=1, deterministic=True, verbose=-1)
    return lgb.train(p, lgb.Dataset(X, y, feature_name=list(names)),
                     num_boost_round=params["num_boost_round"])


def model_paths(cfg):
    g = cfg["gaps"]
    return {k: os.path.join(registry.ROOT, g[k + "_model"]) for k in ("dead", "owner")}


def load(cfg):
    import lightgbm as lgb
    return {k: lgb.Booster(model_file=p) for k, p in model_paths(cfg).items()}


def predict(models, F, idx):
    """(p_dead, p_home) at frames idx, each the mean of the two perspectives."""
    if len(idx) == 0:
        return np.zeros(0), np.zeros(0)
    a, b = F[0][idx], F[1][idx]
    p_dead = (models["dead"].predict(a, num_threads=1)
              + models["dead"].predict(b, num_threads=1)) / 2
    p_home = (models["owner"].predict(a, num_threads=1)
              + 1 - models["owner"].predict(b, num_threads=1)) / 2
    return p_dead, p_home
