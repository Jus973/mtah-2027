"""make cache: parse the ATD once into an npz cache (outside the repo) and check it.

Prints aggregates only: line/frame counts, ball presence, mean visible objects, team mapping.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args  # noqa: E402

from mtah import config, registry  # noqa: E402
from mtah.io import atd as atdio  # noqa: E402

args = game_args()
sp = registry.spec(args.game, args.data_dir)
cfg = config.load()
pub = cfg["published"]
atd, stats = atdio.load_game(args.game, args.data_dir, use_cache=False)
if not sp["fake"]:
    atdio.save_cache(atd, atdio.cache_path(args.game))
    print("cache written outside the repo:", not atdio.cache_path(args.game).startswith(registry.ROOT))

graded = registry.graded_mask(sp)
present = atd.ball_present()
n_obj = atd.n_objects()
print("\nraw file: {n_lines} lines, frame_id {min_frame_id}..{max_frame_id}, bad lines {n_bad_lines}, "
      "out of range {n_out_of_range}, slot-count mismatches {n_slot_mismatch}".format(**stats))
print("slots {}  (home {}, away {})".format(atd.n_slots, int(np.sum(atd.team_of_slot == 0)),
                                           int(np.sum(atd.team_of_slot == 1))))
print("ball present: {} / {} all ({:.1f}%), {} / {} graded ({:.1f}%)".format(
    int(present.sum()), atd.n_frames, 100 * present.mean(), int(present[graded].sum()),
    int(graded.sum()), 100 * present[graded].mean()))
print("mean visible player objects on graded frames: {:.2f}".format(n_obj[graded].mean()))
print("ball outside [0,1] on {} frames (x) / {} frames (y)".format(
    int(np.sum(present & ((atd.ball[:, 0] < 0) | (atd.ball[:, 0] > 1)))),
    int(np.sum(present & ((atd.ball[:, 1] < 0) | (atd.ball[:, 1] > 1))))))

print("\nchecks")
ok = check("contiguous frame_id 1..n", stats["min_frame_id"] == 1
           and stats["max_frame_id"] == sp["n_frames"] and stats["n_lines"] == sp["n_frames"])
if not sp["fake"]:
    ok &= check("ball present overall == {}".format(pub["ball_present_all"]),
                present.sum() == pub["ball_present_all"])
    ok &= check("ball present graded == {}".format(pub["ball_present_graded"]),
                present[graded].sum() == pub["ball_present_graded"])
    ok &= check("mean objects graded ~= {}".format(pub["mean_objects_graded"]),
                abs(n_obj[graded].mean() - pub["mean_objects_graded"]) < 0.15)
    ok &= check("first metadata Team (home) is {}".format(pub["home_tag_club"]),
                pub["home_tag_club"].lower() in (atd.team_names[0] or "").lower())
sys.exit(0 if ok else 1)
