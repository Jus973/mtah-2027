"""Sparse-transition decoder (PLAN.md §2, §4 decode/viterbi.py): the owner can change only at a
contact, so the best owner sequence is a Viterbi path over contacts, not over frames.

Inside each alive stretch (graded, not D), the state is the owner after each contact. A contact by
the owner's team keeps the owner at no cost. A contact by the other team either takes the ball
(log p) or does not (log(1 - p)), p being its takeover probability (learn/takeover.py) sharpened
by alpha on the log-odds scale; every change also pays flip_penalty. The stretch starts in
whichever state scores best, which is the first contact's team unless later contacts say
otherwise. A change lands on the first frame of the contact that causes it (latency by
construction, as the GT is defined); a stretch with no contact keeps the previous stretch's owner,
or takes the next contact's team at the start of a half. Frames outside the halves are X.
"""

import numpy as np

from mtah.learn.stats import runs


def _path(teams, log_take, log_keep, flip_penalty):
    """(start state, states after each contact) maximising the path score."""
    K = len(teams)
    score = np.zeros(2)
    back = np.zeros((K, 2), dtype=np.int64)
    for k in range(K):
        t = teams[k]
        o = 1 - t
        take = score[o] + log_take[k] - flip_penalty
        new_t, from_t = (score[t], t) if score[t] >= take else (take, o)
        new = np.empty(2)
        new[t], back[k, t] = new_t, from_t
        new[o], back[k, o] = score[o] + log_keep[k], o
        score = new
    s = int(np.argmax(score))
    states = np.empty(K, dtype=np.int64)
    for k in range(K - 1, -1, -1):
        states[k] = s
        s = back[k, s]
    return s, states


def decode_viterbi(events, p, graded, dead, alpha, flip_penalty):
    """(T,) labels in H/A/D/X from time-ordered contacts and their takeover probabilities."""
    T = len(graded)
    labels = np.full(T, "X", dtype="<U1")
    labels[graded & dead] = "D"
    starts = np.array([e.start for e in events], dtype=np.int64)
    teams = np.array([e.team for e in events], dtype=np.int64)
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = alpha * np.log(q / (1 - q))
    log_take, log_keep = -np.logaddexp(0, -z), -np.logaddexp(0, z)
    for half_start, half_len in zip(*runs(graded)):
        carry = -1
        alive = graded[half_start:half_start + half_len] & ~dead[half_start:half_start + half_len]
        for a, n in zip(*runs(alive)):
            a += half_start
            lo, hi = np.searchsorted(starts, [a, a + n])
            if hi == lo:
                after = teams[hi] if hi < len(teams) else 0
                labels[a:a + n] = "HA"[carry if carry >= 0 else after]
                continue
            owner, states = _path(teams[lo:hi], log_take[lo:hi], log_keep[lo:hi], flip_penalty)
            frame = a
            for k in range(hi - lo):
                if states[k] != owner:
                    labels[frame:starts[lo + k]] = "HA"[owner]
                    owner, frame = int(states[k]), int(starts[lo + k])
            labels[frame:a + n] = "HA"[owner]
            carry = owner
    return labels
