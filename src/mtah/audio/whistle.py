"""Referee whistle detector (PLAN.md M2 T3), a spectral heuristic on the per-frame audio features.

A whistle is a run of at least min_frames video frames in which the 2-4.5 kHz band is strongly
tonal (peak-to-mean power >= min_tonality) and loud for that band (band energy at least
min_band_db_over_median above the match's median). Thresholds are distributional, set once in
configs, never tied to frames of the match. Output: whistle onset frames.
"""

import numpy as np

from mtah.learn.stats import runs


def detect(feats, cfg):
    tonal = feats["tonality"] >= cfg["min_tonality"]
    band = feats["band_db"]
    loud = band >= np.nanmedian(band) + cfg["min_band_db_over_median"]
    with np.errstate(invalid="ignore"):
        hit = np.nan_to_num(tonal & loud).astype(bool)
    starts, lengths = runs(hit)
    keep = lengths >= cfg["min_frames"]
    return starts[keep]


def near_share(events, whistles, before, after):
    """Share of event frames with a whistle onset in [event - before, event + after]."""
    events, whistles = np.asarray(events), np.sort(np.asarray(whistles))
    if not len(events):
        return None
    lo = np.searchsorted(whistles, events - before, side="left")
    hi = np.searchsorted(whistles, events + after, side="right")
    return float(np.mean(hi > lo))


def chance(whistles, n_frames, mask, before, after):
    """Share of mask frames that have a whistle onset in [f - before, f + after]."""
    cover = np.zeros(n_frames + 1, dtype=np.int64)
    for w in whistles:
        cover[max(0, w - after)] += 1
        cover[min(n_frames, w + before + 1)] -= 1
    near = np.cumsum(cover)[:n_frames] > 0
    return float(near[mask].mean()) if mask.any() else None
