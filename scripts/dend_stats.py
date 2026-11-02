"""make airlock T=dend-stats CFG=configs/<candidate>.yaml: anchor-free D-boundary aggregates of a
candidate's final labels, label-free (PLAN.md §6.2a amendments, handoff/REPLY.md P0.6).

Everything a dend/dflick decision needs from the real match without any tag anchor:
  - D-run length histogram and the dflick census: D runs < 2 s whose flanking spells are the same
    team, with lengths, evidence kinds (oob/gap/still/none = created by the gap decoder) and
    whether they overlap a set-piece tag window (the sim says 78-100% of this population is
    spurious; if the real one is tiny or mostly tag-overlapping, ship dend without dflick);
  - the anchor-free D-end profile: ball-seen fraction and median rolling ball speed in 1 s
    buckets from -5 s to +5 s around each D-to-alive end (a placed-ball restart shows the ball
    reappearing slow and accelerating; ends that are already at the kick show it fast at once);
  - for gap-ended runs: the ball's reappearance speed (share >= 4 m/s = restarted in flight,
    the case the sim cannot represent) and the wait from the run's end to the next touch;
  - set-piece alignment at the raw (v1-fitted) and config-corrected anchors with the offset
    histogram, candidate-matched chance beside it (consistency.set_piece_alignment).
Aggregates only; works on the real match (airlock) and on sim/fake matches.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

from mtah import config, pipeline, registry  # noqa: E402
from mtah.eval.consistency import SET_PIECE_TYPES, set_piece_alignment  # noqa: E402
from mtah.events.deadball import evidence  # noqa: E402
from mtah.events.touches import touch_events  # noqa: E402
from mtah.features.ball import clean_ball, velocity  # noqa: E402
from mtah.features.players import team_distances  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402

LEN_EDGES_S = (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40)
OFFSET_EDGES_S = tuple(range(-10, 16))      # set-piece offset histogram, 1 s bins
PROFILE_S = 5                               # D-end profile half-width
FLIGHT_MPS = 4.0                            # reappearance faster than this = restarted in flight


def d_runs_of(labels, graded):
    """[(start, end_inclusive, next_label)] of predicted D runs on graded frames."""
    out, start = [], None
    for i in range(len(labels) + 1):
        d = i < len(labels) and labels[i] == "D"
        if d and start is None:
            start = i
        elif not d and start is not None:
            out.append((start, i - 1, labels[i] if i < len(labels) else "X"))
            start = None
    return [(a, b, nxt) for a, b, nxt in out if graded[a]]


def hist(values, edges):
    counts, _ = np.histogram(values, bins=[-np.inf] + list(edges) + [np.inf])
    return {"edges": list(edges), "counts": [int(c) for c in counts]}


args = game_args(add=lambda p: p.add_argument("--config", default="configs/final.yaml"))
sp = registry.spec(args.game, args.data_dir)
cfg = config.load(os.path.abspath(args.config))
fps = sp["fps"]
graded = registry.graded_mask(sp)
atd, _ = load_game(args.game, args.data_dir)
labels, info = pipeline.run(args.game, cfg, args.data_dir, atd=atd)
put("config", {"path": os.path.relpath(os.path.abspath(args.config), registry.ROOT),
               "hash": config.config_hash(cfg)})
# The dflick composition rule and the dend transfer gate (LOG.md pre-registration) read the
# population the candidate DECIDES on - the labels before its own postprocess rewrites it - so
# with postprocess enabled a second, postprocess-off run provides the pre block. Final-label
# statistics (length histogram, alignment, fragmentation) stay on the candidate's real output.
pre_labels = labels
if cfg.get("postprocess", {}).get("enabled"):
    put("postprocess counts", info["postprocess"])
    pre_labels, _ = pipeline.run(args.game,
                                 dict(cfg, postprocess=dict(cfg["postprocess"], enabled=False)),
                                 args.data_dir, atd=atd)
    put("gate populations", "census and D-end profile below read the PRE-postprocess labels")

ball_raw = atd.ball_m()
ball, observed, _ = clean_ball(ball_raw, atd.pitch, cfg["ball"], fps)
players = atd.players_m()[..., :2].astype(float)
ev = evidence(ball_raw, ball, observed, players, graded, atd.pitch, cfg["deadball"], fps)
speed = np.linalg.norm(velocity(ball, 2, fps), axis=1)

runs = d_runs_of(np.asarray(labels), graded)
lengths = np.array([(b - a + 1) / fps for a, b, _ in runs])
put("D runs (final labels)", {"n": len(runs)})
put("D-run length histogram, s", hist(lengths, LEN_EDGES_S))

# dflick census: short D runs between same-team spells, on the PRE-postprocess labels.
pre = np.asarray(pre_labels)
psegs, start = [], 0
for i in range(1, len(pre) + 1):
    if i == len(pre) or pre[i] != pre[start]:
        psegs.append((start, i, pre[start]))
        start = i
lab = np.asarray(labels)
segs, start = [], 0
for i in range(1, len(lab) + 1):
    if i == len(lab) or lab[i] != lab[start]:
        segs.append((start, i, lab[start]))
        start = i
tag_path = registry.asset_paths(args.game, args.data_dir)["smart_tagging"]
tags = load_tags(tag_path, fps) if os.path.isfile(tag_path) else []
sp_window = np.zeros(len(lab), dtype=bool)
for t in tags:
    if t.is_code("SET PIECES"):
        sp_window[max(0, t.start):t.end + 1] = True
census = {"n": 0, "same_team_lt_2s": 0, "overlaps_set_piece_tag": 0, "by_evidence": {}}
flick_lengths = []
for j in range(1, len(psegs) - 1):
    a, b, s = psegs[j]
    if s != "D" or not graded[a] or b - a >= 2 * fps:
        continue
    census["n"] += 1
    if psegs[j - 1][2] == psegs[j + 1][2] and psegs[j - 1][2] in ("H", "A"):
        census["same_team_lt_2s"] += 1
        flick_lengths.append((b - a) / fps)
        census["overlaps_set_piece_tag"] += bool(sp_window[a:b].any())
        kinds = [k for k in ("oob", "gap", "still") if ev[k][a:b].any()] or ["none"]
        for k in kinds:
            census["by_evidence"][k] = census["by_evidence"].get(k, 0) + 1
put("short-D census, pre-postprocess (lt 2 s, same-team flanks = dflick-eligible)", census)
if flick_lengths:
    put("dflick-eligible run length histogram, s", hist(flick_lengths, (0.4, 0.8, 1.2, 1.6, 2.0)))

# Fragmentation tripwire (handoff/REPLY.md P4 prerequisite): over-fragmentation that the wide
# frac_lt_2s band cannot catch - short alive spells and flip-backs (an alive spell flanked by the
# SAME other team, the owner flipping and flipping straight back). Banded against v6 by the
# humans in LOG.md once re-baselined; report-only here.
alive_spells = n_lt1 = n_lt2 = flipback_lt2 = 0
flipback_gap_s = []
for j in range(len(segs)):
    a, b, s = segs[j]
    if s not in ("H", "A") or not graded[a]:
        continue
    alive_spells += 1
    n_lt1 += b - a < fps
    n_lt2 += b - a < 2 * fps
    if 0 < j < len(segs) - 1 and b - a < 2 * fps \
            and segs[j - 1][2] == segs[j + 1][2] and segs[j - 1][2] in ("H", "A") \
            and segs[j - 1][2] != s:
        flipback_lt2 += 1
        flipback_gap_s.append((b - a) / fps)
put("fragmentation tripwire: alive spells", {"n": alive_spells, "lt_1s": n_lt1, "lt_2s": n_lt2,
                                             "flip_backs_lt_2s": flipback_lt2,
                                             "flip_back_length_s_p50":
                                                 float(np.median(flipback_gap_s))
                                                 if flipback_gap_s else None})

# Anchor-free D-end profile: the ball around each PRE-postprocess D-to-alive end (the dend
# transfer gate's population - what the rule would shift).
pre_runs = d_runs_of(pre, graded)
pre_ends = np.array([b for a, b, nxt in pre_runs if nxt in ("H", "A")], dtype=np.int64)
ends = np.array([b for a, b, nxt in runs if nxt in ("H", "A")], dtype=np.int64)
put("D-to-alive ends (restarts, no half ends)", {"pre": int(len(pre_ends)),
                                                 "final": int(len(ends))})
profile_seen, profile_speed = {}, {}
for off in range(-PROFILE_S, PROFILE_S + 1):
    at = pre_ends + 1 + off * fps
    at = at[(at >= 0) & (at < len(lab))]
    profile_seen["{:+d}s".format(off)] = float(np.mean(observed[at])) if len(at) else None
    s = speed[at]
    s = s[np.isfinite(s)]
    profile_speed["{:+d}s".format(off)] = float(np.median(s)) if len(s) else None
put("D-end profile (pre): ball-seen fraction by offset from the end", profile_seen)
put("D-end profile (pre): median ball speed (m/s) by offset from the end", profile_speed)

# Gap-ended runs: reappearance speed and the wait to the next touch.
d, _ = team_distances(players, ball, atd.team_of_slot)
touches = touch_events(ball, d, cfg["touch"], fps)
touch_starts = np.array([t.start for t in touches], dtype=np.int64)
reap_speed, waits = [], []
for e in pre_ends:
    if observed[min(e, len(observed) - 1)]:
        continue                            # not a gap-ended run
    nxt = e + 1
    while nxt < len(observed) and not observed[nxt]:
        nxt += 1
    if nxt >= len(observed):
        continue
    v = speed[min(nxt + 2, len(speed) - 1)]
    if np.isfinite(v):
        reap_speed.append(float(v))
    i = np.searchsorted(touch_starts, nxt)
    if i < len(touch_starts):
        waits.append(float(touch_starts[i] - nxt) / fps)
put("gap-ended runs: ball reappearance speed m/s",
    {"n": len(reap_speed),
     "share_ge_{:.0f}mps (restarted in flight)".format(FLIGHT_MPS):
         float(np.mean(np.array(reap_speed) >= FLIGHT_MPS)) if reap_speed else None,
     "p25_p50_p75": [float(q) for q in np.percentile(reap_speed, [25, 50, 75])]
         if reap_speed else None})
put("gap-ended runs: gap end to next touch, s",
    {"n": len(waits), "p25_p50_p75": [float(q) for q in np.percentile(waits, [25, 50, 75])]
        if waits else None})

# Set-piece alignment at both anchors, with the offset histogram (tags only; sim has none).
if tags:
    tc = cfg["tags"]
    raw = tc["set_piece_restart_after_start_s"]
    ppc = cfg.get("postprocess") or {}
    k_shift = (ppc.get("restart_backshift_frames") or 0) if ppc.get("enabled") else 0
    for name, anchor in (("raw (v1-fitted)", raw),
                         ("config-corrected", raw - k_shift / fps)):
        if name.startswith("config") and not k_shift:
            continue
        put("set-piece alignment, {} anchor".format(name),
            set_piece_alignment(lab, tags, cfg["tag_team_map"], graded, anchor,
                                tc["set_piece_window_s"], fps))
    sp_tags = [t for t in tags if t.is_code("SET PIECES")
               and any(t.is_type(s) for s in SET_PIECE_TYPES)
               and 0 <= t.start < len(lab) and graded[t.start]]
    restarts = ends + 1
    offsets = []
    for t in sp_tags:
        anchor = t.start + int(round(raw * fps))
        if len(restarts):
            offsets.append((restarts[int(np.argmin(np.abs(restarts - anchor)))] - anchor) / fps)
    if offsets:
        put("nearest restart minus raw anchor: histogram, 1 s bins", hist(offsets, OFFSET_EDGES_S))
