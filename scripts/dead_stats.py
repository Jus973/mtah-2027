"""make airlock T=dead-stats [CFG=...]: label-free checks of the dead-ball rules on a match.

With the config (default configs/rules.yaml): the share of graded frames predicted D; the number of D runs against the
PLAN.md §6.3 band (60-150 over the real match's 158,857 graded frames, scaled to this match);
run lengths; how many runs each evidence kind took part in; and, where the match has Smart
Tagging, how often a set-piece tag (throw-in, corner, goal kick, free kick, kick-off after a goal)
starts within +/- W seconds of a predicted restart (D run end), next to the chance rate for that
window. The calm-gap transfer check: players' mean speed over long ball gaps, and the share of
long-gap frames the calm filter keeps; run it on the real match and on a sim match and compare.
On a sim match the GT D share is printed for comparison. Aggregates only (real or sim).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args, put  # noqa: E402

from mtah import config, registry  # noqa: E402
from mtah.events.deadball import KINDS, dead_runs, evidence, players_speed  # noqa: E402
from mtah.events.touches import touch_events  # noqa: E402
from mtah.features.ball import clean_ball  # noqa: E402
from mtah.features.players import team_distances  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402
from mtah.learn.stats import runs  # noqa: E402

SET_PIECES = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)")
WINDOWS_S = (2, 5)
REAL_GRADED = 158857

def add(parser):
    parser.add_argument("--config", default="configs/rules.yaml")


args = game_args(add=add)
sp = registry.spec(args.game, args.data_dir)
cfg = config.load(os.path.abspath(args.config))
fps = sp["fps"]
atd, _ = load_game(args.game, args.data_dir)
graded = registry.graded_mask(sp)
ball_raw = atd.ball_m()
ball, observed, _ = clean_ball(ball_raw, atd.pitch, cfg["ball"], fps)
players = atd.players_m()[..., :2].astype(float)
d, _ = team_distances(players, ball, atd.team_of_slot)
touches = touch_events(ball, d, cfg["touch"], fps)
ev = evidence(ball_raw, ball, observed, players, graded, atd.pitch, cfg["deadball"], fps)
dr = dead_runs(ev, touches, graded, cfg["deadball"])
dead = np.zeros(len(graded), dtype=bool)
for a, b, _ in dr:
    dead[a:b + 1] = True

lengths = np.array([b - a + 1 for a, b, _ in dr])
scale = graded.sum() / REAL_GRADED
lo, hi = (v * scale for v in config.load()["bands"]["d_runs"])
put("D share of graded frames", float(dead[graded].mean()))
gt_path = os.path.join(registry.data_dir_for(args.game, args.data_dir), "gt.csv")
if sp["fake"] and os.path.isfile(gt_path):
    gt = np.loadtxt(gt_path, delimiter=",", skiprows=1, dtype=str)[:, 1]
    put("GT D share of graded frames (synthetic match)", float(np.mean(gt[graded] == "D")))
put("D runs", {"n": len(dr), "band_scaled": [round(lo, 1), round(hi, 1)],
               "length_s_p10_p50_p90": [float(q) / fps for q in np.percentile(lengths, [10, 50, 90])]
               if len(lengths) else None})
put("D runs with each evidence kind", {k: sum(k in kinds for _, _, kinds in dr) for k in KINDS})
check("D runs within the scaled band", lo <= len(dr) <= hi, n=len(dr))

dc = cfg["deadball"]
speed = players_speed(players, fps)
gaps = [(a, n) for a, n in zip(*runs(~np.isfinite(ball[:, 0]) & graded)) if n >= dc["gap_min_frames"]]
if gaps:
    gap_speed = [float(np.nanmean(speed[a:a + n])) for a, n in gaps]
    long_gap = np.zeros(len(graded), dtype=bool)
    for a, n in gaps:
        long_gap[a:a + n] = True
    put("players mean speed over long ball gaps, m/s (p10, p25, p50, p75, p90)",
        {"n_gaps": len(gaps), "quantiles": [float(q) for q in np.nanpercentile(gap_speed,
                                                                               [10, 25, 50, 75, 90])]})
    put("long-gap frames kept as evidence by the calm filter, %",
        100 * float(ev["gap"][long_gap].mean()))

paths = registry.asset_paths(args.game, args.data_dir)
if os.path.isfile(paths["smart_tagging"]):
    tags = [t for t in load_tags(paths["smart_tagging"], fps)
            if t.is_code("SET PIECES") and any(t.is_type(s) for s in SET_PIECES)
            and 0 <= t.start < len(graded) and graded[t.start]]
    ends = np.array([b + 1 for _, b, _ in dr])
    rows = {}
    for w in WINDOWS_S:
        hit = [bool(len(ends)) and np.min(np.abs(ends - t.start)) <= w * fps for t in tags]
        near = np.zeros(len(graded), dtype=bool)
        for e in ends:
            near[max(0, e - w * fps):e + w * fps + 1] = True
        rows["{} s".format(w)] = {"tags_near_restart_pct": 100 * float(np.mean(hit)) if tags else None,
                                  "chance_pct": 100 * float(near[graded].mean())}
    inside = [bool(dead[t.start]) for t in tags]
    put("set-piece tags", {"n": len(tags), "start_inside_D_pct": 100 * float(np.mean(inside))
                           if tags else None})
    put("set-piece tag start within W of a predicted restart", rows)
    # PLAN.md M1 tag alignment, label-free: where does the predicted restart (first D run end at
    # or after the tag start) fall relative to the tag's start and end?
    after_start, after_end = [], []
    for t in tags:
        later = ends[ends >= t.start]
        if len(later):
            after_start.append((later[0] - t.start) / fps)
            after_end.append((later[0] - t.end) / fps)
    if after_start:
        put("predicted restart minus tag start, s (p25, p50, p75)",
            [float(q) for q in np.percentile(after_start, [25, 50, 75])])
        put("predicted restart minus tag end, s (p25, p50, p75)",
            [float(q) for q in np.percentile(after_end, [25, 50, 75])])
        put("tag length s (p25, p50, p75)",
            [float(q) for q in np.percentile([(t.end - t.start) / fps for t in tags], [25, 50, 75])])
