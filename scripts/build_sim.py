"""make sim: degrade every DFL/IDSSE match (configs/sim.yaml) into a synthetic bundle under SIM_DIR.

Each match is written in the real bundle layout (tactical ATD + gt.csv + fake_meta.json) as
DFL-<match id>, read back through io/atd.py, and its label-free statistics are printed next to
the real ATD's (configs/sim.yaml targets). Open data only.
"""

import argparse

import numpy as np

from mtah import config, registry
from mtah.io.atd import load_game
from mtah.learn import degrade, idsse, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match", action="append", help="default: all fit + holdout matches")
    args = parser.parse_args()
    cfg = config.load("sim.yaml")
    dcfg, mcfg, targets = cfg["degrade"], cfg["measure"], cfg["targets"]
    ids = args.match or cfg["matches"]["fit"] + cfg["matches"]["holdout"]
    print("{:<12} {:>8} {:>10} {:>8} {:>6} {:>6} {:>7}".format(
        "game", "missing", "gaps/1e3f", "med gap", "p11 H", "p11 A", "vis H+A"))
    print("{:<12} {:>8.3f} {:>10.3f} {:>8.0f} {:>6.3f} {:>6.3f} {:>7}".format(
        "real", targets["ball_missing_frac"], 1e3 * targets["gaps_per_frame"],
        targets["gap_median_frames"], *targets["p11_given_7to11"], "20.49"))
    for mid in ids:
        match = idsse.load(mid)
        out, meta = degrade.write_sim(match, dcfg, mcfg, args.seed)
        atd, _ = load_game(meta["game"], out)
        graded = registry.graded_mask(registry.spec(meta["game"], out))
        s = stats.atd_stats(atd.players_m()[..., :2].astype(float), atd.ball_m(), atd.team_of_slot,
                            graded, atd.pitch, mcfg, meta["fps"])
        p11 = []
        for team in (0, 1):
            c = s["visible"][team]
            p11.append(c[11] / c[7:12].sum())
        print("{:<12} {:>8.3f} {:>10.3f} {:>8.0f} {:>6.3f} {:>6.3f} {:>7.2f}".format(
            meta["game"], s["ball_missing_frac"], 1e3 * s["n_gaps"] / graded.sum(), s["gap_median"],
            p11[0], p11[1], atd.n_objects()[graded].mean()))


if __name__ == "__main__":
    main()
