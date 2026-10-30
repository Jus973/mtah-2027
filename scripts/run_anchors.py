"""make anchors: derive kick-offs and attacking directions; assert them against published facts.

Prints per-event scalars only (frames, rest lengths, fractions, +/-1 directions).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args  # noqa: E402

from mtah import config, registry  # noqa: E402
from mtah.events.restarts import detect_kickoffs  # noqa: E402
from mtah.features.context import attacking_direction  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402

args = game_args()
sp = registry.spec(args.game, args.data_dir)
cfg = config.load()
atd, _ = load_game(args.game, args.data_dir)
tags = load_tags(registry.require(["smart_tagging"], args.game, args.data_dir)[0], sp["fps"])
ko = detect_kickoffs(atd, tags, cfg["kickoff"], sp["fps"])
for half, row in sorted(ko.items()):
    print("half {}: {}".format(half, row))
frames = {h: r["frame"] for h, r in ko.items() if r["frame"] is not None}
direction = attacking_direction(atd, frames, cfg["direction"]["window_frames"]) if frames else {}
print("attacking direction (+1 = towards x=1):", direction)

if sp["fake"]:
    with open(os.path.join(registry.data_dir_for(args.game, args.data_dir), "fake_meta.json")) as f:
        meta = json.load(f)
    expect = {int(k): v for k, v in meta["kickoffs"].items()}
    expect_dir = {int(k): v for k, v in meta["home_attack_dir"].items()}
else:
    expect = {int(k): v for k, v in cfg["published"]["kickoff_frames"].items()}
    expect_dir = None

print("\nchecks")
ok = True
for half, frame in sorted(expect.items()):
    got = ko.get(half, {}).get("frame")
    ok &= check("half {} kick-off == {}".format(half, frame), got == frame,
                "got {} (offset {})".format(got, None if got is None else got - frame))
for half, (_, first, last) in zip((1, 2), sp["half_blocks"]):
    ok &= check("half {} kick-off inside graded block".format(half),
                frames.get(half) is not None and first <= frames[half] <= last)
if expect_dir:
    for half, d in expect_dir.items():
        ok &= check("half {} home direction == {}".format(half, d),
                    direction.get(half, {}).get("H") == d)
if len(direction) == 2:
    ok &= check("direction flips at half-time", direction[1]["H"] == -direction[2]["H"])
sys.exit(0 if ok else 1)
