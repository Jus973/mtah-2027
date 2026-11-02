"""make airlock T=disagreement: where the candidates disagree (PLAN.md §6.5, debugging only).

Reads submissions/candidates/<name>.csv for each name in disagreement.candidates (per-frame,
bundle-derived: read here, under the airlock, never by an agent) and reports on graded frames:
pairwise disagreement shares, the share where any two disagree (PLAN.md expects < 10%), and the
disagreement segments (runs of at least disagreement.min_frames). The segments, with each
candidate's labels, go to <cache>/disagreement/segments.csv for the humans' clip viewer
(tools/clipviewer/), which they watch to find bugs, never to write labels or frame-specific rules.
Aggregates only.
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args, put  # noqa: E402

from mtah import config, registry  # noqa: E402
from mtah.eval.consistency import tag_consistency_pair  # noqa: E402
from mtah.eval.score import pm  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402
from mtah.learn.stats import runs  # noqa: E402

args = game_args()
full_cfg = config.load()
cfg = full_cfg["disagreement"]
sp = registry.spec(args.game, args.data_dir)
graded = registry.graded_mask(sp)
folder = os.path.join(registry.ROOT, "submissions", "candidates")
labels = {}
for name in cfg["candidates"]:
    path = os.path.join(folder, name + ".csv")
    if os.path.isfile(path):
        labels[name] = pm.read_label_csv(path)["label"].to_numpy(dtype=str)
put("candidates found", sorted(labels))
if len(labels) < 2:
    sys.exit("need at least two candidates")

pairs = {}
for a, b in itertools.combinations(sorted(labels), 2):
    pairs["{} vs {}".format(a, b)] = float(np.mean(labels[a][graded] != labels[b][graded]))
put("pairwise disagreement, share of graded frames", pairs)

stack = np.stack([labels[n] for n in sorted(labels)])
any_diff = graded & np.any(stack != stack[0], axis=0)
share = float(any_diff[graded].mean())
starts, lengths = runs(any_diff)
keep = lengths >= cfg["min_frames"]
starts, lengths = starts[keep], lengths[keep]
put("any two disagree, share of graded frames (all candidates)", share)
put("disagreement segments", {"n": int(len(starts)),
                              "length_s_p50_p90": [float(np.percentile(lengths, q)) / sp["fps"]
                                                   for q in (50, 90)] if len(starts) else None})
active = [n for n in cfg.get("active", sorted(labels)) if n in labels]
if len(active) >= 2:
    a_stack = np.stack([labels[n] for n in active])
    a_share = float((graded & np.any(a_stack != a_stack[0], axis=0))[graded].mean())
    put("any two disagree, share of graded frames (active set)", {"candidates": active,
                                                                  "share": a_share})
else:
    a_share = share
check("active candidates disagree on < 10% of graded frames (PLAN.md 6.5)", a_share < 0.10,
      share=a_share)

# Alive-share-free second opinions against the reference candidate (PLAN.md §6.2a amendment,
# 1 Nov 2026): the pooled Smart-Tagging consistency denominator is each candidate's own alive
# frames, so it tracks D share (~0.005/point over this back-catalog) and cannot rank candidates.
# Judged instead: agreement where both are alive, and each side's EXCLUSIVE-alive bucket (floors
# pre-registered in LOG.md from these rows). Plus D-end displacement: how far each candidate's
# D-to-alive ends sit from the reference's (a uniform shift shows as a moved median).
ref = cfg.get("reference", "v6")
tag_path = registry.asset_paths(args.game, args.data_dir)["smart_tagging"]
if ref in labels and os.path.isfile(tag_path):
    tags = load_tags(tag_path, sp["fps"])
    # The same decomposition inside long ball gaps (the third signal in v8's rejection carries
    # the same own-denominator coupling). The gap mask follows the reference config's ball and
    # gap thresholds, as mtah.pipeline computes it.
    from mtah.features.ball import clean_ball
    from mtah.io.atd import load_game
    from mtah.learn import gapstate
    fcfg = config.load("final.yaml")
    atd, _ = load_game(args.game, args.data_dir)
    _, observed, _ = clean_ball(atd.ball_m(), atd.pitch, fcfg["ball"], sp["fps"])
    gap = gapstate.long_gaps(observed, graded, fcfg["gaps"]["min_frames"])
    for name in sorted(labels):
        if name == ref:
            continue
        put("tag consistency split, {} (a) vs {} (b) [agree, frames]".format(name, ref),
            tag_consistency_pair(labels[name], labels[ref], tags, full_cfg["tag_team_map"],
                                 graded))
        put("tag consistency split in long gaps, {} (a) vs {} (b)".format(name, ref),
            tag_consistency_pair(labels[name], labels[ref], tags, full_cfg["tag_team_map"],
                                 graded & gap))


def d_to_alive_ends(lab):
    lab = np.asarray(lab)
    d = (lab == "D") & graded
    starts, lengths = runs(d)
    ends = starts + lengths  # first frame after each run
    keep = (ends < len(lab)) & np.isin(lab[np.minimum(ends, len(lab) - 1)], ["H", "A"])
    return ends[keep]


if ref in labels:
    ref_ends = d_to_alive_ends(labels[ref])
    for name in sorted(labels):
        if name == ref or not len(ref_ends):
            continue
        ends = d_to_alive_ends(labels[name])
        if not len(ends):
            continue
        near = ref_ends[np.clip(np.searchsorted(ref_ends, ends), 0, len(ref_ends) - 1)]
        prev = ref_ends[np.clip(np.searchsorted(ref_ends, ends) - 1, 0, len(ref_ends) - 1)]
        disp = np.where(np.abs(near - ends) <= np.abs(prev - ends), near, prev) - ends
        disp_s = disp / sp["fps"]
        put("D-to-alive end displacement, {} -> nearest {} end, s".format(name, ref),
            {"n": int(len(ends)),
             "p25_p50_p75": [float(q) for q in np.percentile(disp_s, [25, 50, 75])],
             "share_gt_2s_away": float(np.mean(np.abs(disp_s) > 2.0))})

out_dir = os.path.join(registry.cache_dir(), "disagreement")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "segments.csv"), "w") as handle:
    handle.write("start_frame,end_frame," + ",".join(sorted(labels)) + "\n")
    for a, n in zip(starts, lengths):
        cells = ["".join(sorted(set(labels[k][a:a + n]))) for k in sorted(labels)]
        handle.write("{},{},{}\n".format(a, a + n - 1, ",".join(cells)))
put("segments for the clip viewer, in the cache dir", "disagreement/segments.csv")
