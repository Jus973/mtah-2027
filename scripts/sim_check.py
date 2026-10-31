"""make sim-check: the sim acceptance test (PLAN.md M1, §6.1).

Runs the official baseline logic (nearest player per frame, then the starter's majority vote) on
every degraded sim match and scores it against the sim GT with the official scorer. The sim is
trusted only if smooth50 lands at the full-match reference 0.24 +/- 0.06; the whole raw /
smooth25 / smooth50 / smooth125 curve is printed next to the official real-match numbers, since
its shape is a stronger realism check than one point. Also prints the sim GT's spell structure.
"""

import os
import sys

import numpy as np

from mtah import config, registry
from mtah.eval.score import pm, score
from mtah.features.baseline import nearest_player_codes
from mtah.io.atd import load_game

sys.path.insert(0, os.path.join(registry.ROOT, "starter"))
import baseline as starter  # noqa: E402  (vendored, unmodified)

REAL_FULL_MATCH = {0: 0.0486, 25: 0.1996, 50: 0.2433, 125: 0.2242}  # PLAN.md §1, official
ACCEPT = (0.18, 0.30)


def spells(labels, blocks):
    out = []
    for _, first, last in blocks:
        w = labels[first:last + 1]
        start = 0
        for i in range(1, len(w) + 1):
            if i == len(w) or w[i] != w[start]:
                if w[start] in "HA":
                    out.append(i - start)
                start = i
    return np.array(out)


def main():
    cfg = config.load("sim.yaml")
    rows = {}
    print("{:<12} {:>7} {:>7} {:>7} {:>7}   {:>8} {:>6} {:>6} {:>6}".format(
        "game", "raw", "s25", "s50", "s125", "spells/90", "<2s", "<6s", "D"))
    for group in ("fit", "holdout"):
        for mid in cfg["matches"][group]:
            game = "DFL-" + mid
            sp = registry.spec(game)
            atd, _ = load_game(game)
            codes = nearest_player_codes(atd)
            f1 = {w: score(starter.smooth(codes, w), "sim:" + game)["spell_f1_mean"]
                  for w in REAL_FULL_MATCH}
            gt = pm.read_label_csv(os.path.join(registry.data_dir_for(game), "gt.csv"))
            gt = gt["label"].to_numpy(dtype=str)
            graded = registry.graded_mask(sp)
            sl = spells(gt, sp["half_blocks"])
            minutes = graded.sum() / sp["fps"] / 60
            rows[game] = (group, f1)
            print("{:<12} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}   {:>8.0f} {:>6.2f} {:>6.2f} {:>6.2f}".format(
                game, f1[0], f1[25], f1[50], f1[125], len(sl) / minutes * 90,
                np.mean(sl < 2 * sp["fps"]), np.mean(sl < 6 * sp["fps"]), np.mean(gt[graded] == "D")))
    mean = {w: float(np.mean([r[1][w] for r in rows.values()])) for w in REAL_FULL_MATCH}
    print("{:<12} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}".format("sim mean", *(mean[w] for w in REAL_FULL_MATCH)))
    print("{:<12} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}".format("real (ofc.)", *REAL_FULL_MATCH.values()))
    ok = ACCEPT[0] <= mean[50] <= ACCEPT[1]
    print("\n  [{}] sim smooth50 {:.4f} within {}-{} (full-match reference 0.2433)".format(
        "PASS" if ok else "FAIL", mean[50], *ACCEPT))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
