"""Vectorised re-implementation of starter/baseline.py step 1 (nearest player's team).

Exists to prove the loader: regenerating the published baseline CSVs byte-for-byte from our arrays
checks frame alignment, team mapping and parsing in one shot. Float operations mirror the starter
exactly (normalised * metres, then dx*dx + dy*dy in float64; first slot wins ties).
"""

import numpy as np

NOBODY = -1


def nearest_player_codes(atd):
    length, width = atd.pitch
    present = ~np.isnan(atd.ball[:, 0])
    codes = np.full(atd.n_frames, NOBODY, dtype=np.int8)
    bx = atd.ball[present, 0] * length
    by = atd.ball[present, 1] * width
    dx = atd.xy[present, :, 0] * length - bx[:, None]
    dy = atd.xy[present, :, 1] * width - by[:, None]
    dist = dx * dx + dy * dy
    seen = ~np.isnan(atd.xy[present, :, 0])
    dist = np.where(seen, dist, np.inf)
    best = np.argmin(dist, axis=1)
    any_seen = seen.any(axis=1)
    team = atd.team_of_slot[best].astype(np.int8)
    codes[np.flatnonzero(present)] = np.where(any_seen, team, NOBODY)
    return codes
