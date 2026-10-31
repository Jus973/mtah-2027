"""Per-team proximity of players to the ball (PLAN.md §4). Team-symmetric: H = 0, A = 1."""

import warnings

import numpy as np


def team_distances(players, ball, team_of_slot):
    """(d (T, 2), slot (T, 2)): per team, distance from the ball to its nearest visible player and
    that player's slot. NaN / -1 where the ball or the whole team is missing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = np.linalg.norm(players - ball[:, None, :], axis=-1)
    d = np.full((len(ball), 2), np.nan)
    slot = np.full((len(ball), 2), -1, dtype=np.int64)
    for team in (0, 1):
        cols = np.flatnonzero(team_of_slot == team)
        sub = np.where(np.isfinite(dist[:, cols]), dist[:, cols], np.inf)
        best = np.argmin(sub, axis=1)
        dmin = sub[np.arange(len(ball)), best]
        ok = np.isfinite(dmin)
        d[ok, team] = dmin[ok]
        slot[ok, team] = cols[best[ok]]
    return d, slot
