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


def phase_cover(tags, team_map, n_frames):
    """(T,) bool: frames inside a phase that implies a team in possession (implied_side)."""
    cover = np.zeros(n_frames, dtype=bool)
    for tag in tags:
        if implied_side(tag, team_map) is not None:
            cover[max(0, tag.start):min(n_frames, tag.end + 1)] = True
    return cover


SET_PIECE_TYPES = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)")


def _runs_of(labels, value):
    out, start = [], None
    for i, lab in enumerate(np.append(labels, "#")):
        if lab == value and start is None:
            start = i
        elif lab != value and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def set_piece_checks(labels, tags, team_map, graded, restart_after_s, window_s, match_s, fps=25):
    """PLAN.md M2 T2 (b) and (c), label-free. Each set-piece tag's restart is placed
    restart_after_s after its start (the tagger's fixed-window convention, tags.* in configs).
    (b) share of tags with a predicted D run ending within window_s of that restart, and, for
    team-tagged ones whose nearest predicted D run ends within match_s of it, share whose spell
    right after that D run is the tag's team (the taker's side: is the restart owner right?);
    (c) share of predicted D runs that overlap no set-piece tag window (the tagger misses most
    restarts, so this is a rate to compare candidates, not an error count)."""
    labels = np.asarray(labels)
    sp = [t for t in tags if t.is_code("SET PIECES") and any(t.is_type(s) for s in SET_PIECE_TYPES)
          and 0 <= t.start < len(labels) and graded[t.start]]
    d_runs = [(a, b) for a, b in _runs_of(labels, "D") if graded[a]]
    ends = np.array([b + 1 for _, b in d_runs], dtype=np.int64)
    near, team_ok, team_n = 0, 0, 0
    by_type = {}
    for t in sp:
        restart = t.start + int(round(restart_after_s * fps))
        if len(ends) and np.min(np.abs(ends - restart)) <= window_s * fps:
            near += 1
        side = team_side(t, team_map)
        if side is not None and len(ends):
            j = int(np.argmin(np.abs(ends - restart)))
            if abs(ends[j] - restart) <= match_s * fps and ends[j] < len(labels) \
                    and labels[ends[j]] in ("H", "A"):
                ok = bool(labels[ends[j]] == side)
                team_n += 1
                team_ok += ok
                kind = (t.get("Type") or "").strip()
                a, n = by_type.get(kind, (0, 0))
                by_type[kind] = (a + ok, n + 1)
    windows = np.zeros(len(labels), dtype=bool)
    for t in sp:
        windows[max(0, t.start):t.end + 1] = True
    unexplained = sum(not windows[a:b + 1].any() for a, b in d_runs)
    return {"tags": len(sp), "restart_within_window": near / len(sp) if sp else None,
            "next_spell_is_tag_team": team_ok / team_n if team_n else None,
            "team_tagged": team_n,
            "next_spell_is_tag_team_by_type": {k: [a / n, n] for k, (a, n) in sorted(by_type.items())},
            "d_runs": len(d_runs),
            "d_runs_without_set_piece_tag": unexplained / len(d_runs) if d_runs else None}
