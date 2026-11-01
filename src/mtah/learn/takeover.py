"""Takeover probability of a contact (PLAN.md M2 learned emissions), trained on the sim only.

A contact is any contact episode (events/touches.py with min_frames 1) at radius owner.contact_r_m.
For a contact by the team that does not own the ball, the model gives P(takeover): the toucher's
team owns the ball right after it. It is trained on the sim's fit matches (DFL possession, other
games), never on the real match. Features are relative to the toucher's team (team-symmetric):
the contact's kinematics and its rhythm (time to the previous and next contacts and whether they
are the same team's). Pressure (players of each team within 3 m / 8 m) was tried and dropped: it
left held-out AUC unchanged (0.913) and depends on track dropouts and duplicates. Gradient-boosted
trees (LightGBM), single-threaded and seeded, so training and prediction are deterministic.
"""

import os

import numpy as np

from mtah import registry
from mtah.events.touches import touch_events

FEATURES = ("log_frames", "d_min", "opp_d_min", "speed_before", "speed_before_missing",
            "speed_after", "speed_after_missing", "dv", "turn", "turn_missing",
            "log_dt_prev", "prev_same_team", "log_dt_next", "next_same_team", "next2_same_team")
CAP_S = 10.0


def contacts(ball, d, touch_cfg, r, fps=25):
    """Every contact episode at radius r, with its kinematics, in time order."""
    return touch_events(ball, d, dict(touch_cfg, r_touch_m=r, min_frames=1), fps)


def features(events, d, fps=25):
    """(n_events, len(FEATURES)) float array; d: (T, 2) per-team nearest-player distance."""
    n = len(events)
    X = np.zeros((n, len(FEATURES)))
    dt_cap = np.log1p(CAP_S)
    for i, e in enumerate(events):
        prv = events[i - 1] if i else None
        nxt = events[i + 1] if i + 1 < n else None
        nxt2 = events[i + 2] if i + 2 < n else None
        opp_d = d[e.start:e.end + 1, 1 - e.team]
        sb, sa = e.speed_before, e.speed_after
        X[i] = [np.log1p(e.n_frames), e.d_min,
                min(float(np.nanmin(opp_d)), CAP_S) if np.isfinite(opp_d).any() else CAP_S,
                np.nan_to_num(sb), float(np.isnan(sb)), np.nan_to_num(sa), float(np.isnan(sa)),
                np.nan_to_num(sa - sb), np.nan_to_num(e.turn_deg) / 180.0,
                float(np.isnan(e.turn_deg)),
                np.log1p(min((e.start - prv.end) / fps, CAP_S)) if prv else dt_cap,
                float(prv is not None and prv.team == e.team),
                np.log1p(min((nxt.start - e.end) / fps, CAP_S)) if nxt else dt_cap,
                float(nxt is not None and nxt.team == e.team),
                float(nxt2 is not None and nxt2.team == e.team)]
    return X


def training_rows(events, gt, graded, lag):
    """(indices, labels) of contacts by the team that does not own the ball in the GT just before
    it: label 1 if the GT owner lag frames after the contact ends is the toucher's team."""
    idx, y = [], []
    T = len(gt)
    for i, e in enumerate(events):
        if e.start == 0 or not graded[e.start] or gt[e.start] == "D":
            continue
        before, team = gt[e.start - 1], "HA"[e.team]
        if before not in ("H", "A") or before == team:
            continue
        idx.append(i)
        y.append(float(gt[min(e.end + lag, T - 1)] == team))
    return np.array(idx, dtype=np.int64), np.array(y)


def train(X, y, params):
    """A LightGBM booster; params: the takeover_model section of the config."""
    import lightgbm as lgb
    p = {k: v for k, v in params.items() if k != "num_boost_round"}
    p.update(objective="binary", num_threads=1, deterministic=True, verbose=-1)
    return lgb.train(p, lgb.Dataset(X, y, feature_name=list(FEATURES)),
                     num_boost_round=params["num_boost_round"])


def model_path(cfg):
    return os.path.join(registry.ROOT, cfg["owner"]["model"])


def load(cfg):
    import lightgbm as lgb
    return lgb.Booster(model_file=model_path(cfg))


def predict(booster, X):
    if len(X) == 0:
        return np.zeros(0)
    return booster.predict(X, num_threads=1)
