"""Dead-ball decodes from per-frame P(dead) (learn/deadstate.py).

decode_dead (v8): a free 2-state Viterbi per graded run, in steps of step frames, paying lam per
change between alive and dead. Rejected on the real match: free of any anchor, it moved D
boundaries wholesale (median 2.0 s against v6's, 56% of ends > 2 s away).

refine_runs (the P3e hybrid): v8's lesson inverted - the model may only REFINE a boundary the
evidence rules already placed, within +/- window frames, by the classic changepoint objective
(sum log P(dead) before the boundary plus sum log P(alive) after it, ends; mirrored for starts).
It cannot create, delete or relocate a run."""

import numpy as np

from mtah.learn.stats import runs


def decode_dead(p_dead, graded, step, lam):
    """(T,) bool dead mask on graded frames. Frames without a probability count as 0.5."""
    dead = np.zeros(len(p_dead), dtype=bool)
    for a, n in zip(*runs(graded)):
        at = np.arange(a, a + n, step)
        q = np.clip(np.nan_to_num(p_dead[at], nan=0.5), 1e-4, 1 - 1e-4)
        em = np.log(np.column_stack([1 - q, q]))
        score = em[0].copy()
        back = np.zeros((len(at), 2), dtype=np.int8)
        for t in range(1, len(at)):
            stay, move = score, score[::-1] - lam
            back[t] = (move > stay).astype(np.int8)
            score = np.maximum(stay, move) + em[t]
        s = int(np.argmax(score))
        states = np.empty(len(at), dtype=np.int8)
        for t in range(len(at) - 1, -1, -1):
            states[t] = s
            s = s ^ back[t, s]
        dead[a:a + n] = np.repeat(states, step)[:n].astype(bool)
    return dead & graded


def _changepoint(q, before_dead, prefer):
    """Best split index i in [0, len(q)]: positions < i take the 'before' state. q: P(dead);
    before_dead: the state before the split (True for an end boundary, False for a start);
    prefer: the original boundary's split index — ties (no evidence either way, e.g. an unseen
    ball) resolve to it, so a run without signal keeps its evidence boundary exactly."""
    log_d, log_a = np.log(q), np.log(1 - q)
    first, then = (log_d, log_a) if before_dead else (log_a, log_d)
    score = np.concatenate([[0.0], np.cumsum(first)]) \
        + np.concatenate([np.cumsum(then[::-1])[::-1], [0.0]])
    score -= 1e-9 * np.abs(np.arange(len(score)) - prefer)
    return int(np.argmax(score))


def refine_runs(runs_list, p_dead, graded, window):
    """[(start, end_inclusive)] with each boundary refined within +/- window frames.

    Windows never cross a neighbouring run or leave the graded block, a run keeps at least one
    frame, and frames without a probability count as 0.5 (no evidence either way)."""
    gr = np.flatnonzero(graded)
    lo_g, hi_g = (int(gr[0]), int(gr[-1])) if gr.size else (0, -1)
    out = []
    for i, (a, b) in enumerate(runs_list):
        prev_end = out[-1][1] if out else lo_g - 1
        next_start = runs_list[i + 1][0] if i + 1 < len(runs_list) else hi_g + 1
        lo = max(a - window, prev_end + 1, lo_g)
        hi = min(a + window, b)
        q = np.clip(np.nan_to_num(p_dead[lo:hi + 1], nan=0.5), 1e-4, 1 - 1e-4)
        new_a = lo + _changepoint(q, False, a - lo) if hi >= lo else a
        new_a = min(new_a, b)                     # at least one frame stays D
        lo = max(b - window, new_a)
        hi = min(b + window, next_start - 1, hi_g)
        q = np.clip(np.nan_to_num(p_dead[lo:hi + 1], nan=0.5), 1e-4, 1 - 1e-4)
        new_b = lo + _changepoint(q, True, b - lo + 1) - 1 if hi >= lo else b
        new_b = max(new_b, new_a)
        out.append((int(new_a), int(new_b)))
    return out
