"""make airlock T=sync: broadcast -> tactical sync from audio, validated by tracking (PLAN.md M2
T3), and the broadcast ball's recovery of tactical ball gaps (the T3-broadcast falsifier).

1. Window offsets (sync/audio_xcorr.py) for each audio feature in sync.features; a window is
   confident if its peak correlation and peak ratio clear sync.min_score / sync.min_ratio.
2. Offsets of confident windows cluster into a few values (live segments; cuts and replays
   break them). Each cluster is validated on the tracking: share of frames where both feeds see
   the ball and agree within sync.tol_m (PLAN.md: >= 90% to accept), also with the broadcast
   pitch mirrored in case its axes point the other way.
3. With the accepted clusters, the share of tactical-missing graded frames whose synced broadcast
   ball is present (PLAN.md falsifier: < 10% -> drop T3-broadcast), split by predicted D / alive.
Aggregates and offsets only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args, put  # noqa: E402

import games  # noqa: E402
from mtah import config, pipeline, registry  # noqa: E402
from mtah.audio import features  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402
from mtah.learn.stats import runs  # noqa: E402
from mtah.sync import pointset  # noqa: E402
from mtah.sync.audio_xcorr import highpass, validate, window_offsets  # noqa: E402

args = game_args()
cfg = config.load()["sync"]
sp = registry.spec(args.game, args.data_dir)
nb = games.SOURCES["broadcast"]["n_video_frames"]
ft = features.load(args.game, "tactical", sp["n_frames"], args.data_dir, sp["fps"])
fb = features.load(args.game, "broadcast", nb, args.data_dir, sp["fps"])
atd_t, _ = load_game(args.game, args.data_dir)
atd_b, _ = load_game(args.game, args.data_dir, source="broadcast")
ball_t, ball_b = atd_t.ball_m(), atd_b.ball_m()
pitch = np.array(atd_b.pitch, dtype=float)
put("broadcast ATD", {"frames": int(len(ball_b)), "ball_seen": float(np.isfinite(ball_b[:, 0]).mean())})

best = None
for feat in cfg["features"]:
    ref, query = highpass(ft[feat], cfg["highpass_frames"]), highpass(fb[feat], cfg["highpass_frames"])
    s, off, score, ratio = window_offsets(ref, query, cfg["window_frames"], cfg["hop_frames"],
                                          cfg["exclude_frames"])
    conf = (score >= cfg["min_score"]) & (ratio >= cfg["min_ratio"])
    clusters = []
    for o in off[conf]:
        for c in clusters:
            if abs(o - c["offset"]) <= cfg["cluster_tol_frames"]:
                c["n"] += 1
                break
        else:
            clusters.append({"offset": int(o), "n": 1})
    clusters = sorted([c for c in clusters if c["n"] >= cfg["min_cluster_windows"]],
                      key=lambda c: -c["n"])[:8]
    rows = []
    for c in clusters:
        member = conf & (np.abs(off - c["offset"]) <= cfg["cluster_tol_frames"])
        frames = np.concatenate([np.arange(a, a + cfg["window_frames"]) for a in s[member]])
        frames = np.unique(frames[frames < nb])
        offset = int(np.median(off[member]))
        agree, n = validate(ball_t, ball_b, offset, frames, cfg["tol_m"])
        agree_m, _ = validate(ball_t, pitch - ball_b, offset, frames, cfg["tol_m"])
        rows.append({"offset": offset, "windows": c["n"], "broadcast_frames": int(len(frames)),
                     "both_ball_frames": n, "agree": agree, "agree_mirrored": agree_m})
        c.update(offset=offset, frames=frames, agree=agree, agree_m=agree_m)
    put("sync on {}".format(feat), {"windows": int(len(s)), "confident": float(conf.mean()),
                                    "score_p50_p90": [float(np.percentile(score, q)) for q in (50, 90)]})
    put("offset clusters on {} (tactical - broadcast frame)".format(feat), rows)
    accepted = [c for c in clusters if max(c["agree"] or 0, c["agree_m"] or 0) >= cfg["accept"]]
    if best is None or sum(len(c["frames"]) for c in accepted) > best[1]:
        best = (feat, sum(len(c["frames"]) for c in accepted), accepted)

# Fallback (PLAN.md): sync on the tracking itself, the two feeds' ball tracks.
pc = cfg["pointset"]
s, off, med, n = pointset.window_offsets(ball_t, ball_b, cfg["window_frames"], cfg["hop_frames"],
                                         pc["step_frames"], pc["min_frames"])
good = med <= pc["max_median_m"]
clusters = []
for o in off[good]:
    for c in clusters:
        if abs(o - c["offset"]) <= cfg["cluster_tol_frames"]:
            c["n"] += 1
            break
    else:
        clusters.append({"offset": int(o), "n": 1})
clusters = sorted([c for c in clusters if c["n"] >= cfg["min_cluster_windows"]], key=lambda c: -c["n"])
rows = []
for c in clusters:
    member = good & (np.abs(off - c["offset"]) <= cfg["cluster_tol_frames"])
    frames = np.concatenate([np.arange(a, a + cfg["window_frames"]) for a in s[member]])
    frames = np.unique(frames[frames < nb])
    offset = int(np.median(off[member]))
    agree, nboth = validate(ball_t, ball_b, offset, frames, cfg["tol_m"])
    rows.append({"offset": offset, "windows": c["n"], "broadcast_frames": int(len(frames)),
                 "both_ball_frames": nboth, "agree": agree})
    c.update(offset=offset, frames=frames, agree=agree, agree_m=None)
put("sync on the ball tracks", {"windows": int(len(s)), "synced": float(good.mean()) if len(s) else 0.0,
                                "median_m_p10_p50": [float(np.percentile(med, q)) for q in (10, 50)]
                                if len(s) else None})
put("offset clusters on the ball tracks (tactical - broadcast frame)", rows[:8])
# PLAN.md accepts or discards per live broadcast segment: runs of frames with a valid broadcast
# homography (the rest are replays, close-ups and crowd shots), each at the better of the main
# offsets, accepted if its balls agree on >= accept of >= min_both frames where both are seen.
sys.path.insert(0, os.path.join(registry.ROOT, "starter"))
import load_homography  # noqa: E402  (vendored, unmodified)
_, _, valid = load_homography.load(args.game, registry.data_dir_for(args.game, args.data_dir),
                                   "broadcast")
main_offsets = [c["offset"] for c in clusters[:pc["main_offsets"]]]
seg_rows = {"segments": 0, "validated": 0, "accepted": 0, "accepted_frames": 0}
by_offset = {}
starts_v, lengths_v = runs(valid)
for a0, n0 in zip(starts_v, lengths_v):
    seg_rows["segments"] += 1
    frames = np.arange(a0, a0 + n0)
    scored = [(validate(ball_t, ball_b, o, frames, cfg["tol_m"]), o) for o in main_offsets]
    scored = [(ag, nb_, o) for (ag, nb_), o in scored if ag is not None and nb_ >= pc["min_both"]]
    if not scored:
        continue
    seg_rows["validated"] += 1
    ag, _, o = max(scored)
    if ag >= cfg["accept"]:
        seg_rows["accepted"] += 1
        seg_rows["accepted_frames"] += int(n0)
        by_offset.setdefault(o, []).append(frames)
put("live broadcast segments (valid homography) at the main offsets", seg_rows)
seg_accepted = [{"offset": o, "frames": np.concatenate(f)} for o, f in by_offset.items()]
if sum(len(c["frames"]) for c in seg_accepted) > best[1]:
    best = ("ball tracks, per live segment", sum(len(c["frames"]) for c in seg_accepted),
            seg_accepted)

feat, _, accepted = best
put("accepted (agreement >= {:.0%})".format(cfg["accept"]),
    {"feature": feat, "n": len(accepted),
     "broadcast_frames": int(sum(len(c["frames"]) for c in accepted))})
graded = registry.graded_mask(sp)
missing = graded & ~np.isfinite(ball_t[:, 0])
recovered = np.zeros(len(graded), dtype=bool)
for c in accepted:
    t = c["frames"] + c["offset"]
    ok = (t >= 0) & (t < len(graded))
    seen = np.isfinite(ball_b[c["frames"][ok], 0])
    recovered[t[ok][seen]] = True
recovered &= missing
labels, _ = pipeline.run(args.game, config.load(os.path.join(registry.ROOT, "configs", "final.yaml")),
                         args.data_dir, atd=atd_t)
alive = np.isin(labels, ["H", "A"])
share = float(recovered.sum() / max(1, missing.sum()))
put("tactical-missing graded frames recovered by the broadcast ball",
    {"all": share, "in predicted alive": float(recovered[alive].sum() / max(1, (missing & alive).sum())),
     "in predicted D": float(recovered[~alive].sum() / max(1, (missing & ~alive).sum()))})
check("T3-broadcast falsifier: >= 10% of tactical-missing graded frames recovered",
      share >= 0.10, share=share)

# Broadcast cut-aways as dead-ball evidence, step 1 (board idea 32; pre-registered there).
# Every run of VALID broadcast homography (a live shot) is mapped onto the tactical clock with the
# main offset whose ball agreement on that run is best (runs too short to score inherit the
# previous run's offset). Cut-aways = the runs of invalid homography between two consecutive
# mapped live runs, bracketed by their tactical ends. Label-free aggregates: coverage and bracket
# lengths, enrichment of the final pipeline's D inside cut-aways vs live runs, set-piece tag
# overlap, the lag of the bracket end past the D-run end (replays outlast restarts), the live-play
# tail and the long-gap interior population an interior-only veto could ever move.
from mtah.io.tags import load_tags  # noqa: E402
from mtah.learn import gapstate  # noqa: E402
from mtah.features.ball import clean_ball, velocity  # noqa: E402

fps = sp["fps"]
cfg_final = config.load(os.path.join(registry.ROOT, "configs", "final.yaml"))
T = len(graded)
live_runs, last_o, unscorable = [], None, 0
for a0, n0 in zip(starts_v, lengths_v):
    frames = np.arange(a0, a0 + n0)
    cand = []
    for o in main_offsets:
        ag, nb_ = validate(ball_t, ball_b, o, frames, cfg["tol_m"])
        if ag is not None and nb_ >= 10:
            cand.append((ag, o))
    if cand:
        last_o = max(cand)[1]
    else:
        unscorable += 1
    if last_o is None:
        continue
    live_runs.append((int(a0), int(a0 + n0 - 1), int(last_o)))
live_t = np.zeros(T, dtype=bool)
for a0, b0, o in live_runs:
    live_t[max(0, a0 + o):min(T, b0 + o + 1)] = True
cut_t = np.zeros(T, dtype=bool)
brackets = []
for (pa, pb, po), (na, nb2, no) in zip(live_runs[:-1], live_runs[1:]):
    t0, t1 = max(0, pb + po + 1), min(T - 1, na + no - 1)
    if t1 >= t0:
        brackets.append((t0, t1)); cut_t[t0:t1 + 1] = True
cut_t &= graded; live_t &= graded
blen = np.array([(b - a + 1) / fps for a, b in brackets])
put("cut-aways: coverage", {"live_runs": len(live_runs), "unscorable_runs_inheriting_offset": unscorable,
                            "brackets": len(brackets),
                            "bracket_length_s_p50_p90": [float(np.percentile(blen, q)) for q in (50, 90)] if len(blen) else None,
                            "share_of_graded_frames": float(cut_t.sum() / max(1, graded.sum())),
                            "live_share_of_graded_frames": float(live_t.sum() / max(1, graded.sum()))})
is_d = np.asarray(labels) == "D"
enrich = {}
for name, mask in (("pooled", graded),) + tuple(("half {}".format(h), np.isin(np.arange(T), np.arange(a, b + 1)))
                                                for h, a, b in sp["half_blocks"]):
    pc_ = float(is_d[cut_t & mask].mean()) if (cut_t & mask).any() else None
    pl_ = float(is_d[live_t & mask].mean()) if (live_t & mask).any() else None
    enrich[name] = {"p_D_given_cutaway": pc_, "p_D_given_live": pl_,
                    "ratio": (pc_ / pl_) if (pc_ is not None and pl_) else None}
put("cut-aways: enrichment of final-pipeline D (gate ratio ge 2)", enrich)
tag_path = registry.asset_paths(args.game, args.data_dir)["smart_tagging"]
tags = load_tags(tag_path, fps) if os.path.isfile(tag_path) else []
sp_tags = [t for t in tags if t.is_code("SET PIECES")]
hit = 0
for t in sp_tags:
    lo, hi = max(0, t.start - 10 * fps), min(T, t.end + 10 * fps + 1)
    hit += bool(cut_t[lo:hi].any())
put("cut-aways: set-piece tags overlapped within 10 s (gate ge 0.70)",
    {"tags": len(sp_tags), "overlapped": hit, "share": float(hit / len(sp_tags)) if sp_tags else None})
starts_d, lengths_d = runs(is_d & graded)
lags = []
for a0, n0 in zip(starts_d, lengths_d):
    d1 = a0 + n0 - 1
    over = [b for b in brackets if b[0] <= d1 and b[1] >= a0]
    if over:
        lags.append((max(b[1] for b in over) - d1) / fps)
lags = np.array(lags)
put("cut-aways: bracket end minus D-run end, s (p50 gt 0 = replay outlasts restart)",
    {"d_runs_overlapping": int(len(lags)),
     "p50_p90": [float(np.percentile(lags, q)) for q in (50, 90)] if len(lags) else None,
     "share_positive": float((lags > 0).mean()) if len(lags) else None})
ball_c, observed, _ = clean_ball(ball_t, atd_t.pitch, cfg_final["ball"], fps)
speed = np.linalg.norm(velocity(ball_c, 2, fps), axis=1)
alive_cut = cut_t & ~is_d
put("cut-aways: live-play tail (alive, ball seen moving ge 4 m/s)",
    {"share_of_cutaway_frames": float((alive_cut & (speed >= 4)).sum() / max(1, cut_t.sum()))})
gap = gapstate.long_gaps(observed, graded, cfg_final["gaps"]["min_frames"])
put("cut-aways: long-gap interior population (interior-only veto domain)",
    {"gap_frames_in_cutaways_share_of_graded": float((gap & cut_t).sum() / max(1, graded.sum())),
     "alive_share_there": float((~is_d)[gap & cut_t].mean()) if (gap & cut_t).any() else None,
     "alive_share_in_gaps_outside_cutaways": float((~is_d)[gap & ~cut_t].mean()) if (gap & ~cut_t).any() else None})
