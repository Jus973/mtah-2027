"""Ball-free decode inside long ball gaps (learn/gapstate.py): with no ball there is no contact,
so the owner decoder (decode/viterbi.py) has nothing to place a change on, and v3 calls the whole
gap dead.

Inside each gap the state is one of H, A, D per step of step_frames, with emissions
log P(H) = log((1 - p_dead) p_home), log P(A) = log((1 - p_dead)(1 - p_home)), log P(D) =
log p_dead from the gap models. A Viterbi path pays lam_dead per change into or out of D and
lam_owner per change of owner, including at the two ends, where the gap meets the labels the
decoder gave the frames around it. Only the ratio of the penalties to the emissions matters, so
there is no separate sharpening. Labels outside the gaps are unchanged.
"""

import numpy as np

from mtah.learn.stats import runs

STATES = "HAD"


def _path(em, start, end, cost):
    n = len(em)
    score = em[0] - (cost[start] if start >= 0 else 0.0)
    back = np.zeros((n, 3), dtype=np.int64)
    cols = np.arange(3)
    for t in range(1, n):
        cand = score[:, None] - cost
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], cols] + em[t]
    if end >= 0:
        score = score - cost[:, end]
    s = int(np.argmax(score))
    out = np.empty(n, dtype=np.int64)
    for t in range(n - 1, -1, -1):
        out[t] = s
        s = back[t, s]
    return out


def decode_gaps(labels, gap, p_dead, p_home, step, lam_dead, lam_owner):
    """Labels with every gap run re-decoded. p_dead, p_home: (T,) arrays, read at gap frames."""
    out = labels.copy()
    cost = np.array([[0.0, lam_owner, lam_dead], [lam_owner, 0.0, lam_dead],
                     [lam_dead, lam_dead, 0.0]])
    T = len(labels)
    for a, n in zip(*runs(gap)):
        at = np.arange(a, a + n, step)
        pd = np.clip(p_dead[at], 1e-4, 1 - 1e-4)
        ph = np.clip(p_home[at], 1e-4, 1 - 1e-4)
        em = np.log(np.column_stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd]))
        start = STATES.find(labels[a - 1]) if a > 0 else -1
        end = STATES.find(labels[a + n]) if a + n < T else -1
        states = np.repeat(_path(em, start, end, cost), step)[:n]
        out[a:a + n] = np.array(list(STATES))[states]
    return out
