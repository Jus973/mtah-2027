"""Classify where two owner sequences disagree (board #85). Label-free: only the two predictions.

A stretch is a maximal run of frames where both call the ball alive (H/A) but name different
owners. Its type says which sequence changes owner at each end:
  cand_early    the candidate switches at the start, the final catches up at the end
  cand_late     the final switches at the start, the candidate catches up at the end
  cand_inserted the candidate has a short spell inside one of the final's (switches out and back)
  cand_removed  the final has a short spell the candidate lacks
  edge          a stretch bounded by D/X/graded edges, or both change at once
"""

import numpy as np

KINDS = ("cand_early", "cand_late", "cand_inserted", "cand_removed", "edge")
LEN_BINS = (0, 13, 25, 50, 100, 10 ** 9)          # frames: <0.5 s, 0.5-1, 1-2, 2-4, >4 s


def stretches(final, cand, mask):
    alive = np.isin(final, ["H", "A"]) & np.isin(cand, ["H", "A"]) & mask
    d = alive & (final != cand)
    edges = np.diff(np.concatenate([[0], d.astype(np.int8), [0]]))
    out = []
    T = len(final)
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if s == 0 or e >= T or not (alive[s - 1] and alive[e]):
            out.append((s, e, "edge"))
            continue
        fs, cs = final[s - 1] != final[s], cand[s - 1] != cand[s]
        fe, ce = final[e - 1] != final[e], cand[e - 1] != cand[e]
        if fs == cs or fe == ce:
            kind = "edge"
        elif cs and fe:
            kind = "cand_early"
        elif fs and ce:
            kind = "cand_late"
        elif cs and ce:
            kind = "cand_inserted"
        else:
            kind = "cand_removed"
        out.append((s, e, kind))
    return out


def summary(final, cand, mask, truth, truth_ok):
    """Per kind and per length bin: [stretches, frames, final right, cand right, judged frames],
    judged on frames where truth_ok (GT on sims, unambiguous Smart-Tagging side on the real match)."""
    res = {}
    for s, e, kind in stretches(final, cand, mask):
        n = e - s
        lb = int(np.searchsorted(LEN_BINS, n, side="right") - 1)
        for key in (kind, "len{}".format(lb), "all"):
            r = res.setdefault(key, [0, 0, 0, 0, 0])
            ok = truth_ok[s:e]
            r[0] += 1
            r[1] += n
            r[2] += int(np.sum(final[s:e][ok] == truth[s:e][ok]))
            r[3] += int(np.sum(cand[s:e][ok] == truth[s:e][ok]))
            r[4] += int(ok.sum())
    return res
