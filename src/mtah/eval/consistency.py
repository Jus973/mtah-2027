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


def tag_consistency_pair(labels_a, labels_b, tags, team_map, graded):
    """Alive-share-free decomposition of tag_consistency for candidate A against reference B.

    tag_consistency's denominator is the candidate's own predicted-alive frames, so its pooled
    rate tracks the candidate's D share almost linearly (~0.005 per D-share point over the logged
    candidates) and cannot rank candidates with different alive shares. This splits the tagged
    frames by who calls them alive: "both" (same denominator for both candidates — but identical
    by construction for a D-only transform) and each side's EXCLUSIVE-alive bucket, where a
    candidate's new or removed alive frames are actually judged. Floors for the exclusive buckets
    are pre-registered from the back-catalog (LOG.md), never tuned."""
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    agree = {k: [0, 0] for k in ("both", "only_a", "only_b")}
    for tag in tags:
        side = implied_side(tag, team_map)
        if side is None:
            continue
        lo, hi = max(0, tag.start), min(len(a), tag.end + 1)
        g = graded[lo:hi]
        wa, wb = a[lo:hi][g], b[lo:hi][g]
        alive_a, alive_b = np.isin(wa, ["H", "A"]), np.isin(wb, ["H", "A"])
        for key, mask, w in (("both", alive_a & alive_b, wa), ("only_a", alive_a & ~alive_b, wa),
                             ("only_b", alive_b & ~alive_a, wb)):
            agree[key][0] += int(np.sum(w[mask] == side))
            agree[key][1] += int(mask.sum())
    return {k: {"agree": (hits / n if n else None), "frames": n}
            for k, (hits, n) in agree.items()}


def set_piece_alignment(labels, tags, team_map, graded, anchor_s, window_s, fps=25):
    """Set-piece alignment at an explicit anchor, with its own chance baseline (PLAN.md §6.2a
    amendment). The legacy set_piece_checks number is anchored at tags.set_piece_restart_after_
    start_s = 9.48 s — the median of v1's OWN restart lags — so any candidate that fixes the
    measured restart lag is punished at the raw anchor. Judge candidates at the config-derived
    corrected anchor (9.48 - k/fps for rules that move D ends k frames earlier: a known constant,
    never fitted to the candidate) and report the raw-anchor number beside it.

    Restarts are predicted D-run ends whose next graded label is alive (a half-end pseudo-restart
    cannot score a hit). Returns the share of tags with a restart within window_s of the anchor,
    signed offset quantiles (restart - anchor, s), and the candidate-matched chance of a hit
    (1 - exp(-2 * window_s * n_restarts / graded_s))."""
    labels = np.asarray(labels)
    sp = [t for t in tags if t.is_code("SET PIECES") and any(t.is_type(s) for s in SET_PIECE_TYPES)
          and 0 <= t.start < len(labels) and graded[t.start]]
    ends = np.array([b + 1 for _, b in _runs_of(labels, "D")
                     if graded[b] and b + 1 < len(labels) and labels[b + 1] in ("H", "A")],
                    dtype=np.int64)
    offsets = []
    for t in sp:
        anchor = t.start + int(round(anchor_s * fps))
        if len(ends):
            offsets.append((ends[int(np.argmin(np.abs(ends - anchor)))] - anchor) / fps)
    offsets = np.array(offsets)
    within = float(np.mean(np.abs(offsets) <= window_s)) if len(offsets) else None
    graded_s = float(graded.sum()) / fps
    chance = float(1.0 - np.exp(-2.0 * window_s * len(ends) / graded_s)) if graded_s else None
    q = ([float(v) for v in np.percentile(offsets, [25, 50, 75])] if len(offsets) else None)
    return {"anchor_s": float(anchor_s), "tags": len(sp), "restarts": int(len(ends)),
            "share_within_window": within, "chance_within_window": chance,
            "offset_s_p25_p50_p75": q,
            "offset_iqr_s": (q[2] - q[0]) if q else None}


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
