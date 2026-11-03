"""Takeover probability of a contact (PLAN.md M2 learned emissions), trained on the sim only.

A contact is any contact episode (events/touches.py with min_frames 1) at radius owner.contact_r_m.
For a contact by the team that does not own the ball, the model gives P(takeover): the toucher's
team owns the ball right after it. It is trained on the sim's fit matches (DFL possession, other
games), never on the real match. Features are relative to the toucher's team (team-symmetric):
the contact's kinematics and its rhythm (time to the previous and next contacts and whether they
are the same team's). With owner.shape_features (v6), ten team-shape signals from the toucher's
perspective are added (learn/gapstate.py features): both teams' widths, the centroid and speed
gaps, both teams' forward drift 1 s and 2 s after the contact, and both centroids' change from
2 s before to 2 s after it. A team that wins the ball spreads and moves forward. On the sim
they raised spellF1 by 0.014 with v5's gap decoder, on all 7 matches. Pressure (players of each
team within 3 m / 8 m) was tried and dropped: it left held-out AUC unchanged (0.913) and depends on
track dropouts and duplicates. Gradient-boosted trees (LightGBM), single-threaded and seeded, so
training and prediction are deterministic.
"""

import os

import numpy as np

from mtah import registry
from mtah.events.touches import touch_events

FEATURES = ("log_frames", "d_min", "opp_d_min", "speed_before", "speed_before_missing",
            "speed_after", "speed_after_missing", "dv", "turn", "turn_missing",
            "log_dt_prev", "prev_same_team", "log_dt_next", "next_same_team", "next2_same_team")
SHAPE = ("x_sy", "y_sy", "dcu", "dspeed", "x_vu_+25", "x_vu_+50", "y_vu_+25", "y_vu_+50")
SHAPE_FEATURES = SHAPE + ("x_cu_change", "y_cu_change")
CAP_S = 10.0


def feature_names(cfg):
    return FEATURES + (SHAPE_FEATURES if cfg["owner"].get("shape_features") else ())


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


def shape_features(events, F, names):
    """(n_events, len(SHAPE_FEATURES)): team shape from the toucher's perspective. F: the
    (F_home, F_away) arrays of gapstate.features; names: their column names."""
    col = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(events), len(SHAPE_FEATURES)))
    for i, e in enumerate(events):
        a, b = F[e.team][e.start], F[e.team][e.end]
        out[i] = [b[col[n]] for n in SHAPE] + [b[col["x_cu_+50"]] - a[col["x_cu_-50"]],
                                               b[col["y_cu_+50"]] - a[col["y_cu_-50"]]]
    return out


def training_rows(events, gt, graded, lag, policy="legacy"):
    """(indices, labels) of contacts by the team that does not own the ball in the GT just before
    it: label 1 if the GT owner lag frames after the contact ends is the toucher's team.

    owner.label_policy (P2 d', handoff/REPLY.md):
      legacy             the label frame is end + lag as is - when it lands on D or X (exactly
                         where restart decisions live) the row is a systematic false 0;
      alive_lag          the label reads the GT owner at the first ALIVE graded frame >= end +
                         lag within the half (no alive frame left: the row is dropped);
      alive_lag_restarts alive_lag, plus rows for stretch-opening contacts (the frame before
                         them is GT D), which the decoder scores at every stretch opening but
                         legacy training never saw. The prior-owner filter cannot apply there;
                         the label is simply whether the toucher's team owns at the label frame.
    """
    idx, y = [], []
    T = len(gt)
    alive_label = policy in ("alive_lag", "alive_lag_restarts")
    for i, e in enumerate(events):
        if e.start == 0 or not graded[e.start] or gt[e.start] == "D":
            continue
        before, team = gt[e.start - 1], "HA"[e.team]
        opening = before == "D" and policy == "alive_lag_restarts"
        if not opening and (before not in ("H", "A") or before == team):
            continue
        j = min(e.end + lag, T - 1)
        if alive_label:
            while j < T and graded[j] and gt[j] not in ("H", "A"):
                j += 1
            if j >= T or not graded[j] or gt[j] not in ("H", "A"):
                continue
        idx.append(i)
        y.append(float(gt[j] == team))
    return np.array(idx, dtype=np.int64), np.array(y)


def possession_weights(events, rows, gt, params, fps=25):
    """Per training row: ref_s / (length in s of the GT spell the contact interrupts), clipped to
    takeover_model.possession_weight_clip, so short contested spells weigh more (board #25).
    The interrupted spell is the GT run holding frame e.start - 1, so the weight never reads the
    row's label; rows opening a stretch (the frame before is D) keep weight 1."""
    ref = params.get("possession_weight_s", 0.0)
    w = np.ones(len(rows))
    if not ref or not len(rows):
        return w
    lo, hi = params.get("possession_weight_clip", (0.25, 4.0))
    change = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(gt)]))
    for k, i in enumerate(rows):
        f = events[i].start - 1
        if gt[f] not in ("H", "A"):
            continue
        r = np.searchsorted(starts, f, side="right") - 1
        w[k] = np.clip(ref * fps / (ends[r] - starts[r]), lo, hi)
    return w


NOT_LGB = ("num_boost_round", "possession_weight_s", "possession_weight_clip")


def train(X, y, params, names=FEATURES, weight=None):
    """A LightGBM booster; params: the takeover_model section of the config."""
    import lightgbm as lgb
    if params.get("possession_weight_s") and weight is None:
        raise ValueError("takeover_model.possession_weight_s is set but this caller passes no "
                         "row weights - it would silently train unweighted")
    p = {k: v for k, v in params.items() if k not in NOT_LGB}
    p.update(objective="binary", num_threads=1, deterministic=True, verbose=-1)
    return lgb.train(p, lgb.Dataset(X, y, weight=weight, feature_name=list(names)),
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


def with_restarts(events, p, restarts, prob=1.0):
    """Contacts plus release restarts (events/deadball.py). prob is the injected takeover
    probability (deadball.restart_takeover_p): 1.0 makes the taker's ownership certain, which
    with alpha-sharpening makes the flip mandatory - v4's corners (3/10 right) are the warning
    that forced attribution is sometimes wrong, so a release candidate tunes this out of fold."""
    merged = sorted([(e.start, 0, e, q) for e, q in zip(events, p)]
                    + [(r.start, 1, r, prob) for r in restarts], key=lambda x: (x[0], x[1]))
    return [m[2] for m in merged], np.array([m[3] for m in merged])
