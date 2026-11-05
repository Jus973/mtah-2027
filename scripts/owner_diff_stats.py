"""make airlock T=owner-diff CFG=configs/rules_v12.yaml: label-free check of an owner-only change.

Runs the FINAL config and CFG on the match. On graded frames where both call the ball alive but
name different owners, reports how often each one's owner agrees with the team the
Smart-Tagging phase implies, next to that agreement on the frames where they agree. Board #67
gate 3: a regression flag for v12, never a target. Aggregates only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

from mtah import config, pipeline, registry  # noqa: E402
from mtah.eval.consistency import implied_side  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402

args = game_args(add=lambda p: p.add_argument("--config", required=True))
sp = registry.spec(args.game, args.data_dir)
graded = registry.graded_mask(sp)
base_cfg = config.load(os.path.join(registry.ROOT, "configs", "final.yaml"))
cand_cfg = config.load(os.path.abspath(args.config))
a, _ = pipeline.run(args.game, base_cfg, args.data_dir)
b, info = pipeline.run(args.game, cand_cfg, args.data_dir)
put("kick rate per 1e3 seen frames, and the setting it picks", {
    "kick_rate": info.get("kick_rate"), "setting": list(info.get("proxowner_setting") or [])})
tag_path = registry.asset_paths(args.game, args.data_dir)["smart_tagging"]
tags = load_tags(tag_path, sp["fps"]) if os.path.isfile(tag_path) else []
side = np.full(len(a), "", dtype="<U1")
for tag in tags:
    s = implied_side(tag, cand_cfg["tag_team_map"])
    if s is not None:
        side[max(0, tag.start):min(len(a), tag.end + 1)] = s
alive = np.isin(a, ["H", "A"]) & np.isin(b, ["H", "A"]) & graded
tagged = alive & (side != "")
differ, same = tagged & (a != b), tagged & (a == b)
put("alive frames, owner differs (all / tagged)", [int((alive & (a != b)).sum()), int(differ.sum())])
put("tag agreement where owners differ: final, candidate", {
    "final": float(np.mean(a[differ] == side[differ])) if differ.any() else None,
    "candidate": float(np.mean(b[differ] == side[differ])) if differ.any() else None})
put("tag agreement where owners agree", float(np.mean(a[same] == side[same])) if same.any() else None)
put("D or X changed frames (must be 0)", int(((a == "D") != (b == "D")).sum() + ((a == "X") != (b == "X")).sum()))
# The same split, tag-weighted exactly as eval.consistency.tag_consistency counts (every tag's
# frames, so frames under overlapping tags count once per tag), for both configs on their own
# alive frames: pooled, where the owners differ, and where they agree.
acc = {k: [0, 0, 0] for k in ("all", "differ", "agree")}        # final hits, candidate hits, n
for tag in tags:
    s = implied_side(tag, cand_cfg["tag_team_map"])
    if s is None:
        continue
    lo, hi = max(0, tag.start), min(len(a), tag.end + 1)
    m = alive[lo:hi]
    wa, wb = a[lo:hi][m], b[lo:hi][m]
    for key, sel in (("all", np.ones(len(wa), bool)), ("differ", wa != wb), ("agree", wa == wb)):
        acc[key][0] += int(np.sum(wa[sel] == s))
        acc[key][1] += int(np.sum(wb[sel] == s))
        acc[key][2] += int(sel.sum())
put("tag-weighted agreement [final, candidate, tag-frames]", {
    k: [v[0] / v[2] if v[2] else None, v[1] / v[2] if v[2] else None, v[2]] for k, v in acc.items()})
# Unambiguous frames only: every phase tag covering the frame implies the same team.
hH = np.zeros(len(a), dtype=bool)
hA = np.zeros(len(a), dtype=bool)
for tag in tags:
    s = implied_side(tag, cand_cfg["tag_team_map"])
    if s is not None:
        (hH if s == "H" else hA)[max(0, tag.start):min(len(a), tag.end + 1)] = True
clear = alive & (hH ^ hA)
side1 = np.where(hH, "H", "A")
out = {}
for key, sel in (("differ", clear & (a != b)), ("agree", clear & (a == b)), ("all", clear)):
    out[key] = [float(np.mean(a[sel] == side1[sel])) if sel.any() else None,
                float(np.mean(b[sel] == side1[sel])) if sel.any() else None, int(sel.sum())]
put("unambiguous-tag frames [final, candidate, frames]", out)
put("alive frames under conflicting tags (both teams implied)", int((alive & hH & hA).sum()))
