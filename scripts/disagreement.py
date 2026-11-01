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
from mtah.eval.score import pm  # noqa: E402
from mtah.learn.stats import runs  # noqa: E402

args = game_args()
cfg = config.load()["disagreement"]
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
put("any two disagree, share of graded frames", share)
put("disagreement segments", {"n": int(len(starts)),
                              "length_s_p50_p90": [float(np.percentile(lengths, q)) / sp["fps"]
                                                   for q in (50, 90)] if len(starts) else None})
check("candidates disagree on < 10% of graded frames (PLAN.md 6.5)", share < 0.10, share=share)

out_dir = os.path.join(registry.cache_dir(), "disagreement")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "segments.csv"), "w") as handle:
    handle.write("start_frame,end_frame," + ",".join(sorted(labels)) + "\n")
    for a, n in zip(starts, lengths):
        cells = ["".join(sorted(set(labels[k][a:a + n]))) for k in sorted(labels)]
        handle.write("{},{},{}\n".format(a, a + n - 1, ",".join(cells)))
put("segments for the clip viewer, in the cache dir", "disagreement/segments.csv")
