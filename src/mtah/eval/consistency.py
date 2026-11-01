"""Label-free agreement with Smart Tagging (PLAN.md §6.2a): a second opinion, never a target.

Team-tagged phases say which team has the ball: the tag's team for attacking phases (BUILD UP,
PROGRESSION, BALL IN FINAL THIRD, ...), its opponent for defending ones (DEFENDING IN ...,
DEFENSIVE TRANSITION). Over predicted-alive graded frames inside such phases, the share whose
predicted owner is that team. SET PIECES and GOALS windows straddle stoppages and are left out.
The tagger has false positives and negatives and phases are not possessions, so the rate flags
regressions and compares variants coarsely; it is not maximised. No person decides any label.
"""

import numpy as np

from mtah.io.tags import team_side

DEFENDING = ("DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
             "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")
EXCLUDED = ("SET PIECES", "GOALS")


def implied_side(tag, team_map):
    side = team_side(tag, team_map)
    if side is None or any(tag.is_code(c) for c in EXCLUDED):
        return None
    if any(tag.is_code(c) for c in DEFENDING):
        return "A" if side == "H" else "H"
    return side


def tag_consistency(labels, tags, team_map, graded):
    """{"agree": share, "frames": n, "by_code": {code: [share, frames]}} over tagged alive frames."""
    labels = np.asarray(labels)
    agree = total = 0
    by_code = {}
    for tag in tags:
        side = implied_side(tag, team_map)
        if side is None:
            continue
        lo, hi = max(0, tag.start), min(len(labels), tag.end + 1)
        w = labels[lo:hi][graded[lo:hi]]
        alive = np.isin(w, ["H", "A"])
        n, k = int(alive.sum()), int(np.sum(w[alive] == side))
        agree += k
        total += n
        code = tag.code.strip().upper()
        a, t = by_code.get(code, (0, 0))
        by_code[code] = (a + k, t + n)
    return {"agree": agree / total if total else None, "frames": total,
            "by_code": {c: [a / t if t else None, t] for c, (a, t) in sorted(by_code.items())}}
