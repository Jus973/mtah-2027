"""Owner hypotheses H1-H3 (PLAN.md M1): who owns the ball between touches, given touches and D.

All three share how a flip is placed (on the first frame of the deciding touch: latency by
construction, as the GT is defined) and the restart rule (after a D stretch, the first touch
decides the owner, back-filled to the end of the stretch). They differ in which touches can take
the ball from the current owner:
  H1  any touch by the other team;
  H2  a controlled touch: the ball stays close for >= n_c frames, or is slower than v_c after the
      touch, or the next touch is by the same team within t_c seconds;
  H3  two consecutive touches by the other team (a completed pass or a dribble).
A ball gap holds the state: no touch seen, no change. Frames outside the halves are X; D comes
from the caller (the sim GT's D for an owner-only comparison, or the dead-ball detector).
"""

import numpy as np

from mtah.learn.stats import runs

HYPOTHESES = ("H1", "H2", "H3")


def _takes(hypothesis, touch, nxt, cfg, fps):
    if hypothesis == "H1":
        return True
    same_next = nxt is not None and nxt.team == touch.team
    if hypothesis == "H3":
        return same_next
    if hypothesis == "H2":
        return (touch.n_frames >= cfg["n_c"]
                or (np.isfinite(touch.speed_after) and touch.speed_after <= cfg["v_c_mps"])
                or (same_next and nxt.start - touch.end <= cfg["t_c_s"] * fps))
    raise ValueError("unknown hypothesis " + hypothesis)


def decode_owner(touches, graded, dead, hypothesis, cfg, fps=25):
    """(T,) labels in H/A/D/X from time-ordered touches."""
    T = len(graded)
    labels = np.full(T, "X", dtype="<U1")
    labels[graded & dead] = "D"
    starts = np.array([t.start for t in touches], dtype=np.int64)
    carry = -1
    for half_start, half_len in zip(*runs(graded)):
        carry = -1
        for a, n in zip(*runs(graded[half_start:half_start + half_len]
                              & ~dead[half_start:half_start + half_len])):
            a += half_start
            lo, hi = np.searchsorted(starts, [a, a + n])
            seg = touches[lo:hi]
            if not seg:
                after = touches[hi].team if hi < len(touches) else 0
                labels[a:a + n] = "HA"[carry if carry >= 0 else after]
                continue
            owner, flips = seg[0].team, [(a, seg[0].team)]
            for i in range(1, len(seg)):
                touch = seg[i]
                if touch.team == owner:
                    continue
                nxt = seg[i + 1] if i + 1 < len(seg) else None
                if _takes(hypothesis, touch, nxt, cfg, fps):
                    owner = touch.team
                    flips.append((touch.start, owner))
            for (f, team), (g, _) in zip(flips, flips[1:] + [(a + n, None)]):
                labels[f:g] = "HA"[team]
            carry = owner
    return labels
