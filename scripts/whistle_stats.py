"""make airlock T=whistle: the whistle detector's label-free gate (PLAN.md M2 T3).

On the tactical audio: whistles per minute of graded play; the share of tracking-derived D onsets
with a whistle within +/- whistle.window_s and of restarts (D run ends) with one within the
window before, each next to its chance rate; and whether a whistle precedes each derived
kick-off. PLAN.md: feed whistles as a D-onset cue only if >= whistle.gate of the anchors have
one. Aggregates only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args, put  # noqa: E402

from mtah import config, pipeline, registry  # noqa: E402
from mtah.audio import features, whistle  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402

args = game_args()
wc = config.load()["whistle"]
sp = registry.spec(args.game, args.data_dir)
fps = sp["fps"]
graded = registry.graded_mask(sp)
feats = features.load(args.game, "tactical", sp["n_frames"], args.data_dir, fps)
w = whistle.detect(feats, wc)
w = w[graded[w]]
put("whistles", {"n": int(len(w)), "per_minute_graded": float(len(w) / (graded.sum() / fps / 60))})

cfg = config.load(os.path.join(registry.ROOT, "configs", "final.yaml"))
atd, _ = load_game(args.game, args.data_dir)
labels, info = pipeline.run(args.game, cfg, args.data_dir, atd=atd)
change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
onsets = [f for f in change if labels[f] == "D" and labels[f - 1] in ("H", "A") and graded[f]]
ends = [f for f in change if labels[f - 1] == "D" and labels[f] in ("H", "A") and graded[f]]
k = int(round(wc["window_s"] * fps))
alive = graded & np.isin(labels, ["H", "A"])
rows = {"D onsets (+/- window)": [whistle.near_share(onsets, w, k, k),
                                  whistle.chance(w, len(graded), alive, k, k), len(onsets)],
        "restarts (window before)": [whistle.near_share(ends, w, k, 0),
                                     whistle.chance(w, len(graded), graded & (labels == "D"), k, 0),
                                     len(ends)],
        "derived kick-offs (window before)": [whistle.near_share(sorted(info["kickoffs"].values()),
                                                                 w, k, 0), None,
                                              len(info["kickoffs"])]}
put("share with a whistle [share, chance, n]", rows)
gate = rows["D onsets (+/- window)"][0] or 0.0
check("whistle gate: >= {:.0%} of D onsets have a whistle".format(wc["gate"]), gate >= wc["gate"],
      share=gate)
