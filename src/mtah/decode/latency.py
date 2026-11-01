"""Transition timing (PLAN.md §2.3, §4 decode/latency.py): offsets by construction, residuals on sim.

Changes are placed by construction (a takeover on its contact's first frame, a stoppage on its
first evidence frame, a restart on the restart contact). This module measures what is left, on
the sim only: per transition type, the median offset of matched predicted transitions from the
GT's, and the two corrections PLAN.md names. The corrections are applied only if a config says so
(latency section), after make sim-latency shows they help on held-out:
  shift_onsets    move every alive -> D boundary earlier by k frames;
  gap_flips       a takeover whose contact comes right after a long ball gap is moved into the gap
                  (its midpoint): the ball changed hands while the tracker had lost it.
"""

import numpy as np

from mtah.learn.stats import runs

KINDS = ("turnover", "d_onset", "d_offset")


def transitions(labels):
    """[(frame, kind, new_label)] where labels change between H, A and D."""
    labels = np.asarray(labels)
    out = []
    for f in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        a, b = labels[f - 1], labels[f]
        if a in ("H", "A") and b in ("H", "A"):
            out.append((int(f), "turnover", b))
        elif a in ("H", "A") and b == "D":
            out.append((int(f), "d_onset", b))
        elif a == "D" and b in ("H", "A"):
            out.append((int(f), "d_offset", b))
    return out


def offsets(pred, gt, window):
    """{kind: array of (pred - GT) frames} over one-to-one matches within +/- window frames."""
    P, G = transitions(pred), transitions(gt)
    frames = np.array([f for f, _, _ in P], dtype=np.int64)
    used = np.zeros(len(P), dtype=bool)
    out = {k: [] for k in KINDS}
    for f, kind, new in G:
        near = np.flatnonzero(~used & (np.abs(frames - f) <= window)) if len(P) else []
        cand = [j for j in near if P[j][1] == kind and P[j][2] == new]
        if cand:
            j = min(cand, key=lambda j: abs(frames[j] - f))
            used[j] = True
            out[kind].append(frames[j] - f)
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


def shift_onsets(labels, k):
    """Every alive -> D boundary moved k frames earlier (never past the previous boundary)."""
    labels = np.asarray(labels)
    out = labels.copy()
    if k <= 0:
        return out
    prev = 0
    for f in np.flatnonzero(labels[1:] != labels[:-1]) + 1:   # every change, X included
        if labels[f] == "D" and labels[f - 1] in ("H", "A"):
            out[max(prev, f - k):f] = "D"
        prev = f
    return out


def gap_flips(labels, ball_missing, min_gap, max_after):
    """Takeovers within max_after frames of the end of a ball gap of >= min_gap frames move to the
    gap's midpoint (only across alive frames of the old owner)."""
    out = np.array(labels, copy=True)
    starts, lengths = runs(ball_missing)
    ends = starts + lengths                       # first frame seen again
    for f, kind, new in transitions(labels):
        if kind != "turnover":
            continue
        i = np.searchsorted(ends, f, side="right") - 1
        if i < 0 or lengths[i] < min_gap or f - ends[i] > max_after:
            continue
        mid = int(starts[i] + lengths[i] // 2)
        old = out[f - 1]
        if mid < f and np.all(out[mid:f] == old):
            out[mid:f] = new
    return out
