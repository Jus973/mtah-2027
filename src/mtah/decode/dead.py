"""Dead-ball path from per-frame P(dead) (learn/deadstate.py, v8): a 2-state Viterbi per graded
run, in steps of step frames, paying lam per change between alive and dead."""

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
