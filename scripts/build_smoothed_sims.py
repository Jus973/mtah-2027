"""Board #67 verification helper: copy sim bundles with the ball smoothed over W frames.

    python scripts/build_smoothed_sims.py --width 13 --out ~/opendata/sim_b13 [--src SIM_DIR]

Every DFL-* bundle under --src (default SIM_DIR) is rewritten under --out/<game> with its ball
x/y replaced by a centred W-frame moving average inside each contiguous run of seen frames
(learn/proxowner.smooth_ball); players, missingness, gt.csv, metadata and fake_meta.json are
copied unchanged and speeds re-derived by the bundle writer, as in every sim. Open data only.
Point MTAH_SIM_DIR at --out to evaluate a candidate in that world (scripts/paired.py); keep
MTAH_TRAIN_SIM_DIR at the raw SIM_DIR so the proximity owner model still trains on raw sims.
"""

import argparse
import glob
import json
import os
import shutil

from mtah import registry
from mtah.io import atd_write
from mtah.io.atd import load_game
from mtah.learn.proxowner import smooth_ball


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--src", default=registry.SIM_DIR)
    args = parser.parse_args()
    out_root = os.path.expanduser(args.out)
    for src in sorted(glob.glob(os.path.join(args.src, "DFL-*"))):
        game = os.path.basename(src)
        atd, _ = load_game(game, src)
        ball = smooth_ball(atd.ball.copy(), args.width)      # normalised, NaN where missing
        dst = os.path.join(out_root, game)
        os.makedirs(os.path.join(dst, "tactical"), exist_ok=True)
        for f in ("fake_meta.json", "gt.csv"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        xml = glob.glob(os.path.join(src, "tactical", "*_atd.xml"))[0]
        shutil.copy(xml, os.path.join(dst, "tactical", os.path.basename(xml)))
        fps = json.load(open(os.path.join(src, "fake_meta.json")))["fps"]
        atd_write.write_raw(os.path.join(dst, "tactical", os.path.basename(xml)[:-4] + ".txt"),
                            atd.xy, ball, atd.pitch, fps)
        print(game, "-> b{}".format(args.width), dst)


if __name__ == "__main__":
    main()
