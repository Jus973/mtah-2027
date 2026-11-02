"""Metric-aligned label post-processing (the v9 candidate family): two rules applied to the
final H/A/D/X labels after the owner and gap decoders. Both read labels (plus the dead-run
bookkeeping) only — never ball kinematics — so the sim's frozen dead-time ball cannot leak in.

  dflick  a predicted D run shorter than d_flicker_max_frames whose two neighbouring spells are
          the SAME team becomes that team. A spurious mid-play stoppage is the scorer's most
          expensive single error: it splits a matched spell into a worse match plus a pure false
          positive. (On the cross-fitted sim, 78-100% of this population is spurious per match.)
  dend    a D run followed by an alive spell ends restart_backshift_frames (k) earlier, because
          the run is ended at the first confirmed touch STARTING after the evidence
          (events/deadball.py): a taker whose touch starts inside the evidence is skipped and D
          extends to the receiver's touch. With backshift_min_late_frames (m) > 0, only runs
          whose END sits more than m frames past the evidence end are shifted (the restart touch
          started more than m+1 frames after it), by at most their own lateness (never back into
          the evidence): runs that already end on time are left alone. With m null/0, every D-to-alive boundary shifts by k (uniform) — the
          variant measured in handoff/fable/exp_post*.py.

Rules never rewrite a protected frame (the pre-kick-off D block), so a derived kick-off's D
stands. Both thresholds live in configs (postprocess section) and are registered in
stability.params. Coupling note (do not stack frozen constants): a native fix of the same
restart lag — release restarts (deadball.restart_window_frames) or the gap-tail seam — changes
what dend and dflick have left to correct; re-tune k and d_flicker_max_frames inside any such
combined candidate.
"""

import numpy as np

ALIVE = ("H", "A")


def _segments(labels):
    """[(start, stop_exclusive, label)] run-length encoding of a label array."""
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, labels[start]))
            start = i
    return out


def _dflick(out, max_frames, protect, stats):
    segs = _segments(out)
    for j in range(1, len(segs) - 1):
        a, b, lab = segs[j]
        if lab == "D" and b - a < max_frames and segs[j - 1][2] == segs[j + 1][2] \
                and segs[j - 1][2] in ALIVE \
                and not (protect is not None and protect[a:b].any()):
            out[a:b] = segs[j - 1][2]
            stats["dflick_runs"] += 1
            stats["dflick_frames"] += b - a


def _dend_uniform(out, k, protect, stats):
    segs = _segments(out)
    for j in range(len(segs) - 1):
        a, b, lab = segs[j]
        nxt = segs[j + 1][2]
        if lab == "D" and nxt in ALIVE:
            lo = max(a, b - k)
            if protect is not None and protect[lo:b].any():
                continue
            out[lo:b] = nxt
            stats["dend_runs"] += 1
            stats["dend_frames"] += b - lo


def _dend_conditional(out, k, m, runs_meta, protect, stats):
    if runs_meta is None:
        raise ValueError("postprocess.backshift_min_late_frames needs the dead-run metadata "
                         "(events/deadball.py dead_runs out_meta); the caller did not pass it")
    T = len(out)
    for run in runs_meta:
        end, late = run["end"], run["end"] - run["evidence_end"]
        if late <= m or end + 1 >= T:
            continue
        nxt = out[end + 1]
        # The decoder (gap re-decode) may have ended or moved this run already: shift only a
        # boundary that is still D-to-alive at the recorded end.
        if nxt not in ALIVE or out[end] != "D":
            continue
        lo = max(0, end - min(k, late) + 1)      # never back into the evidence
        seg = out[lo:end + 1]
        bad = np.flatnonzero(seg != "D")
        if bad.size:                             # stay inside the containing D segment
            lo += int(bad[-1]) + 1
        if protect is not None and protect[lo:end + 1].any():
            continue
        out[lo:end + 1] = nxt
        stats["dend_runs"] += 1
        stats["dend_frames"] += end + 1 - lo


def apply(labels, cfg, runs_meta=None, protect=None, out_stats=None):
    """A post-processed copy of labels ((T,) '<U1' of H/A/D/X). cfg: the config's postprocess
    section (or None); runs_meta: dead_runs out_meta, required when backshift_min_late_frames > 0;
    protect: (T,) bool frames never rewritten (the pre-kick-off block); out_stats: if a dict,
    filled with dflick/dend run and frame counts. Disabled or missing section: an unchanged copy.

    dflick runs first, then dend, each over a fresh segmentation — the order and semantics the
    sim measurements used (handoff/fable/exp_post6.py)."""
    out = np.asarray(labels).astype("<U1").copy()
    stats = {"dflick_runs": 0, "dflick_frames": 0, "dend_runs": 0, "dend_frames": 0}
    if out_stats is not None:
        out_stats.update(stats)
        stats = out_stats
    if not cfg or not cfg.get("enabled"):
        return out
    L = cfg.get("d_flicker_max_frames") or 0
    k = cfg.get("restart_backshift_frames") or 0
    m = cfg.get("backshift_min_late_frames") or 0
    if L:
        _dflick(out, L, protect, stats)
    if k and m:
        _dend_conditional(out, k, m, runs_meta, protect, stats)
    elif k:
        _dend_uniform(out, k, protect, stats)
    return out
