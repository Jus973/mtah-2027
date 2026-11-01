"""make clips (HUMANS ONLY): export disagreement clips for tools/clipviewer/ (PLAN.md §6.5).

Runs each candidate config in clipviewer.configs on the match, finds the graded runs of at least
clipviewer.min_frames where any two disagree, pads each by clipviewer.pad_s, and writes
<cache>/clipviewer/clips.json: per clip, per frame, every candidate's label and the players and
ball projected to tactical-video pixels through the tactical homography. The file is
bundle-derived: it stays in the cache, outside the repo, and only a human opens it.

Watchers file BUGS against the algorithm. They never write labels, corrections or rules tied to
specific frames (RULES.md). Refuses to run under the airlock; works on synthetic matches.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

from mtah import config, pipeline, registry  # noqa: E402
from mtah.airlock import OUT_ENV  # noqa: E402
from mtah.io import homography  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402
from mtah.learn.stats import runs  # noqa: E402

if os.environ.get(OUT_ENV):
    sys.exit("make clips is for humans: its output is per-frame match data for a local viewer")


def add(parser):
    parser.add_argument("--out", default=None, help="output folder (default: <cache>/clipviewer)")


args = game_args(add=add)
cv = config.load()["clipviewer"]
sp = registry.spec(args.game, args.data_dir)
fps = sp["fps"]
graded = registry.graded_mask(sp)
atd, _ = load_game(args.game, args.data_dir)
labels = {name: pipeline.run(args.game, config.load(path), args.data_dir, atd=atd)[0]
          for name, path in cv["configs"].items()}
names = sorted(labels)
stack = np.stack([labels[n] for n in names])
diff = graded & np.any(stack != stack[0], axis=0)
starts, lengths = runs(diff)
keep = lengths >= cv["min_frames"]
starts, lengths = starts[keep][:cv["max_clips"]], lengths[keep][:cv["max_clips"]]

H, _, valid = homography.load(args.game, args.data_dir)
pitch = np.array(atd.pitch, dtype=float)
players = atd.players_m()[..., :2] / pitch            # normalised 0-1, as the homography expects
ball = atd.ball_m() / pitch
pad = int(round(cv["pad_s"] * fps))


def pixels(f, xy):
    if not valid[f]:
        return []
    x, y = homography.project(H[f], xy[..., 0], xy[..., 1])
    return np.round(np.column_stack([x, y]), 1)


clips = []
for a, n in zip(starts, lengths):
    lo, hi = max(0, a - pad), min(sp["n_frames"], a + n + pad)
    frames = []
    for f in range(lo, hi):
        pl = pixels(f, players[f])
        ok = np.isfinite(players[f, :, 0]) if len(pl) else np.zeros(0, bool)
        bp = pixels(f, ball[f:f + 1])
        frames.append({"f": f, "labels": {k: str(labels[k][f]) for k in names},
                       "players": [[float(p[0]), float(p[1]), int(t)]
                                   for p, t, o in zip(pl, atd.team_of_slot, ok) if o],
                       "ball": [float(bp[0][0]), float(bp[0][1])]
                       if len(bp) and np.isfinite(bp[0]).all() else None})
    clips.append({"start": int(a), "end": int(a + n - 1), "frames": frames})

out_dir = args.out or os.path.join(registry.cache_dir(), "clipviewer")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "clips.json"), "w") as handle:
    json.dump({"game": args.game, "fps": fps, "candidates": names, "clips": clips}, handle)
put("clips", {"n": len(clips), "candidates": names, "folder": "clipviewer (in the cache dir)"})
