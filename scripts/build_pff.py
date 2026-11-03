"""PFF WC2022 evaluation family (board #48): degrade PFF matches into synthetic bundles PFF-<id>.

io/pff.py builds each CleanMatch with mechanical labels from the embedded game_event stream; this
script degrades it with the SAME machinery and fitted parameters as the DFL sims
(learn/degrade.py, configs/sim.yaml `degrade.fitted`) under registry.SIM_DIR. Evaluation only:
the ids are never listed under matches.fit / holdout, so no training script can pick them up.
Prints aggregates only: degradation statistics and the official-baseline smoothing curve against
the mechanical GT, with the GT structure next to sim_check's DFL rows.
"""

import argparse
import os
import sys

import numpy as np

from mtah import config, registry
from mtah.eval.score import pm, score
from mtah.features.baseline import nearest_player_codes
from mtah.io import pff
from mtah.io.atd import load_game
from mtah.learn import degrade

sys.path.insert(0, os.path.join(registry.ROOT, "scripts"))
sys.path.insert(0, os.path.join(registry.ROOT, "starter"))
import baseline as starter  # noqa: E402  (vendored, unmodified)
import sim_check            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = config.load("sim.yaml")
    dcfg, mcfg = cfg["degrade"], cfg["measure"]
    ids = args.games or pff.games()
    print("{:<12} {:>7} {:>7} {:>7} {:>7}   {:>8} {:>6} {:>6} {:>6}".format(
        "game", "raw", "s25", "s50", "s125", "spells/90", "<2s", "<6s", "D"))
    s50 = []
    for g in ids:
        try:
            match = pff.load(g)
        except (ValueError, KeyError, OSError) as e:
            print("PFF-{:<8} skipped: {}".format(g, e))
            continue
        game = "PFF-{}".format(g)
        out, _ = degrade.write_sim(match, dcfg, mcfg, args.seed, game=game)
        atd, _ = load_game(game, out)
        sp = registry.spec(game, out)
        graded = registry.graded_mask(sp)
        codes = nearest_player_codes(atd)
        f1 = {w: score(starter.smooth(codes, w), "sim:" + game)["spell_f1_mean"]
              for w in sim_check.REAL_FULL_MATCH}
        gt = pm.read_label_csv(os.path.join(registry.data_dir_for(game), "gt.csv"))
        gt = gt["label"].to_numpy(dtype=str)
        sl = sim_check.spells(gt, sp["half_blocks"])
        minutes = graded.sum() / sp["fps"] / 60
        s50.append(f1[50])
        print("{:<12} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}   {:>8.0f} {:>6.2f} {:>6.2f} {:>6.2f}".format(
            game, f1[0], f1[25], f1[50], f1[125], len(sl) / minutes * 90,
            np.mean(sl < 2 * sp["fps"]), np.mean(sl < 6 * sp["fps"]), np.mean(gt[graded] == "D")),
            flush=True)
    print("{:<12} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}   (DFL sims: spells/90 ~330-480, "
          "<2s 0.25-0.37, D 0.41-0.42)".format("real (ofc.)", *sim_check.REAL_FULL_MATCH.values()))
    if s50:
        print("PFF smooth50 mean {:.4f} over {} matches (band {})".format(
            np.mean(s50), len(s50), cfg["metrica_accept"]))


if __name__ == "__main__":
    main()
