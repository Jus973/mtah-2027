"""Label-free degradation statistics of tracking in the ATD layout: the targets of learn/degrade.py.

The same function runs on the real ATD (scripts/atd_stats.py, through the airlock) and on a
degraded sim match, so the two can be compared number for number. Inputs are metres with the
origin at a pitch corner (Atd.players_m / Atd.ball_m), NaN where missing.
"""

import warnings

import numpy as np


def runs(mask):
    """(starts, lengths) of the True runs of a 1-d bool array."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    return starts, np.flatnonzero(d == -1) - starts


def speed_labels(cfg):
    e = cfg["speed_edges_mps"]
    return (["<{}".format(e[0])] + ["{}-{}".format(a, b) for a, b in zip(e[:-1], e[1:])]
            + [">={}".format(e[-1]), "unknown"])


def covariates(players, ball, pitch, cfg, fps):
    """Per frame: speed bin (last bin = unknown), crowd bin and edge flag, from positions only."""
    k = cfg["speed_window_frames"]
    edges = cfg["speed_edges_mps"]
    speed = np.full(len(ball), np.nan)
    speed[k:] = np.linalg.norm(ball[k:] - ball[:-k], axis=1) * fps / k
    sbin = np.where(np.isfinite(speed), np.digitize(speed, edges), len(edges) + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.linalg.norm(players - ball[:, None, :], axis=-1)
    crowd = np.sum(d <= cfg["crowd_radius_m"], axis=1)
    cbin = np.digitize(crowd, cfg["crowd_edges"])
    m = cfg["edge_margin_m"]
    with np.errstate(invalid="ignore"):
        ebin = ((ball[:, 0] < m) | (ball[:, 0] > pitch[0] - m)
                | (ball[:, 1] < m) | (ball[:, 1] > pitch[1] - m)).astype(int)
    return sbin, cbin, ebin


def _d2(xy, valid):
    """|x[t+1] - 2 x[t] + x[t-1]| where valid[t] and all three positions exist."""
    d2 = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=-1)
    ok = np.isfinite(d2) & valid[1:-1, ...] if d2.ndim == valid.ndim else np.isfinite(d2)
    return d2[ok]


def _quantiles(x):
    return [float(q) for q in np.percentile(x, [50, 75, 90])] if len(x) else [None] * 3


def atd_stats(players, ball, team_of_slot, graded, pitch, cfg, fps=25):
    present = np.isfinite(ball[:, 0])
    seen = present & graded
    sbin, cbin, ebin = covariates(players, ball, pitch, cfg, fps)
    n_s, n_c = len(cfg["speed_edges_mps"]) + 2, len(cfg["crowd_edges"]) + 1

    onset = np.zeros_like(seen)
    onset[:-1] = seen[:-1] & ~present[1:] & graded[1:]
    hazard = np.zeros((2, n_s, n_c, 2), dtype=np.int64)
    np.add.at(hazard, (ebin[seen], sbin[seen], cbin[seen], 0), 1)
    np.add.at(hazard, (ebin[onset], sbin[onset], cbin[onset], 1), 1)

    starts, lengths = runs(graded & ~present)
    keep = starts > 0
    keep[keep] = seen[starts[keep] - 1]
    starts, lengths = starts[keep], lengths[keep]
    lbin = np.digitize(lengths, cfg["gap_length_edges"]) - 1
    gap_hist = np.zeros((n_s, len(cfg["gap_length_edges"])), dtype=np.int64)
    np.add.at(gap_hist, (sbin[starts - 1], lbin), 1)

    pair = np.zeros_like(seen)
    pair[1:] = present[1:] & present[:-1] & graded[1:]
    step = np.full(len(ball), np.nan)
    step[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1)
    n_jumps = int(np.sum(pair & (step > cfg["jump_m"])))
    # A ball that does not move at all between seen frames. The DFL source data freezes the ball
    # in dead time, so the sim has many such frames; a real optical tracker rarely does.
    frozen = float(np.mean(step[pair] == 0)) if pair.any() else None
    near_still = float(np.mean(step[pair] * fps < 0.1)) if pair.any() else None
    with np.errstate(invalid="ignore"):
        far = seen & ((ball[:, 0] < -5) | (ball[:, 0] > pitch[0] + 5)
                      | (ball[:, 1] < -5) | (ball[:, 1] > pitch[1] + 5))

    def run_summary(lens, edges):
        return {"n": int(len(lens)), "mean": float(lens.mean()) if len(lens) else None,
                "hist": np.bincount(np.digitize(lens, edges) - 1, minlength=len(edges))}

    visible, tracklets, low, high = {}, {}, {}, {}
    finite = np.isfinite(players[..., 0])
    edges = cfg["tracklet_edges"]
    for team in (0, 1):
        cols = np.flatnonzero(team_of_slot == team)
        n_vis = finite[:, cols].sum(axis=1)
        visible[team] = np.bincount(n_vis[graded], minlength=len(cols) + 1)
        tracklets[team] = run_summary(
            np.concatenate([runs(finite[:, c] & graded)[1] for c in cols]), edges)
        low[team] = run_summary(runs(graded & (n_vis <= cfg["low_max_visible"]))[1], edges)
        high[team] = run_summary(runs(graded & (n_vis >= cfg["high_min_visible"]))[1], edges)

    # Label-free dead-ball proxy: how fast players move while the ball is seen vs missing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.full(finite.shape, np.nan)
        v[1:] = np.linalg.norm(np.diff(players, axis=0), axis=-1) * fps
        frame_speed = np.nanmean(v, axis=1)
    ok = np.isfinite(frame_speed) & graded
    collective = {name: _quantiles(frame_speed[ok & mask]) for name, mask in
                  (("ball seen", present), ("ball missing", ~present))}

    return {
        "ball_missing_frac": float(1 - present[graded].mean()),
        "n_gaps": int(len(lengths)),
        "gap_median": float(np.median(lengths)) if len(lengths) else None,
        "gap_mean": float(lengths.mean()) if len(lengths) else None,
        "speed_labels": speed_labels(cfg),
        "hazard": hazard,
        "gap_hist": gap_hist,
        "ball_d2": _quantiles(_d2(ball, graded)),
        "player_d2": _quantiles(_d2(players, np.repeat(graded[:, None], players.shape[1], 1))),
        "n_jumps": n_jumps,
        "jump_rate": n_jumps / max(1, int(pair.sum())),
        "ball_frozen_frac": frozen,
        "ball_still_frac": near_still,
        "n_far_outside": int(far.sum()),
        "visible": visible,
        "tracklets": tracklets,
        "low_runs": low,
        "high_runs": high,
        "collective_speed": collective,
    }
