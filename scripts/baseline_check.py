"""make baseline-check: regenerate the published baselines from OUR loader and diff bytes.

Byte-identical output proves frame alignment, team mapping and parsing in one shot (PLAN.md M0).
Then scores smooth50 / smooth125 on the public window (expected 0.3256 / 0.4242).
For a fake game it instead diffs against the official starter code run on the same files.
"""

import filecmp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args  # noqa: E402

from mtah import registry  # noqa: E402
from mtah.eval.score import score  # noqa: E402
from mtah.features.baseline import nearest_player_codes  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402
from mtah.submit.write import sha256, write_csv  # noqa: E402

sys.path.insert(0, os.path.join(registry.ROOT, "starter"))
import baseline as starter  # noqa: E402  (vendored, unmodified)

EXPECTED_PUBLIC = {"baseline_smooth50": 0.3256, "baseline_smooth125": 0.4242}

args = game_args()
sp = registry.spec(args.game, args.data_dir)
atd, _ = load_game(args.game, args.data_dir)
codes = nearest_player_codes(atd)
out_dir = os.path.join(registry.cache_dir(), "baseline_check", args.game)
ok = True

if sp["fake"]:
    meta, raw = registry.require(["atd_meta", "atd_raw"], args.game, args.data_dir)
    ref = starter.nearest_player_labels(raw, starter.team_of_each_track(meta), sp["n_frames"],
                                        sp["pitch"])
    ok &= check("vectorised codes == starter codes", (ref == codes).all(),
                frames_differ=(ref != codes).sum())

for name, window in starter.VARIANTS:
    path = write_csv(os.path.join(out_dir, name + ".csv"), starter.smooth(codes, window))
    if sp["fake"]:
        continue
    official = os.path.join(registry.ROOT, "evaluation", "results", name + ".csv")
    same = filecmp.cmp(path, official, shallow=False)
    ok &= check("{} byte-identical".format(name), same, sha256_ours=sha256(path)[:12],
                sha256_official=sha256(official)[:12])
    if name in EXPECTED_PUBLIC:
        s = score(path, "public", allow_public=True)  # published numbers, not a candidate
        ok &= check("{} public spellF1 {:.4f} (expect {})".format(
            name, s["spell_f1_mean"], EXPECTED_PUBLIC[name]),
            abs(s["spell_f1_mean"] - EXPECTED_PUBLIC[name]) < 5e-5,
            macro_f1=round(s["macro_f1"], 4))
sys.exit(0 if ok else 1)
