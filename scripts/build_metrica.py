"""make metrica-sim: degrade Metrica Sample Game 3 into a synthetic bundle + acceptance check.

FAMILY=csv builds Sample Games 1-2 instead (io/metrica_csv.py, configs/sim.yaml
`matches.metrica_csv`), same mechanical labels and same degradation; NATIVE=1 skips the simulated
ball loss and keeps the source tracker's own (the CSV release has the ball missing on most dead
frames and few alive ones), writing MET-G1N/MET-G2N. The acceptance band gates the default family
only; for csv it is reported, never the exit code (the matches are easier than the real ATD, so
read their scores as an optimistic, out-of-sample second opinion).

The second validation family (handoff/REPLY.md P5). io/metrica.py builds the CleanMatch with
MECHANICAL event-derived GT (owner at controlled touches, D between out-of-play and restart);
this script degrades it with the SAME machinery and the SAME fitted parameters as the DFL sims
(learn/degrade.py, configs/sim.yaml `degrade.fitted`) and writes it as MET-G3 under
registry.SIM_DIR. The match ids live under configs/sim.yaml `matches.metrica`, never fit or
holdout, so no training script can pick them up.

Everything printed is candidate-blind:
  - the degradation statistics next to the real-ATD targets (as make sim prints),
  - the acceptance test: official-baseline smoothing curve (nearest player + starter majority
    vote) scored by the official scorer against the mechanical GT, with smooth50 gated by the
    configs/sim.yaml `metrica_accept` band (the sim_check.py band),
  - the GT structure (spells/90, sub-2s/sub-6s shares, D share) next to sim_check's DFL rows,
  - the event->label derivation census (io/metrica.py docstring).
"""

import argparse
import os
import sys

import numpy as np

from mtah import config, registry
from mtah.eval.score import pm, score
from mtah.features.baseline import nearest_player_codes
from mtah.io import metrica, metrica_csv
from mtah.io.atd import load_game
from mtah.learn import degrade, stats

sys.path.insert(0, os.path.join(registry.ROOT, "scripts"))
sys.path.insert(0, os.path.join(registry.ROOT, "starter"))
import baseline as starter  # noqa: E402  (vendored, unmodified)
import sim_check            # noqa: E402  (spells + the official reference curve)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family", choices=("g3", "csv"), default="g3")
    parser.add_argument("--native", action="store_true",
                        help="csv family: keep the source ball loss, add no simulated loss")
    args = parser.parse_args()
    if args.native and args.family != "csv":
        parser.error("--native applies to --family csv only")
    cfg = config.load("sim.yaml")
    dcfg, mcfg, targets = cfg["degrade"], cfg["measure"], cfg["targets"]
    accept = tuple(cfg["metrica_accept"])

    print("== degradation (same fitted parameters as the DFL sims) ==")
    print("{:<12} {:>8} {:>10} {:>8} {:>6} {:>6} {:>7}".format(
        "game", "missing", "gaps/1e3f", "med gap", "p11 H", "p11 A", "vis H+A"))
    print("{:<12} {:>8.3f} {:>10.3f} {:>8.0f} {:>6.3f} {:>6.3f} {:>7}".format(
        "real", targets["ball_missing_frac"], 1e3 * targets["gaps_per_frame"],
        targets["gap_median_frames"], *targets["p11_given_7to11"], "20.49"))
    rows = []
    games = cfg["matches"]["metrica" if args.family == "g3" else "metrica_csv"]
    if args.native:
        games = [g + "N" for g in games]
        degrade.ball_missing = lambda match, *a, **k: ~match.graded()
    for game in games:
        if args.family == "g3":
            match = metrica.load(game[len("MET-"):])
        else:
            match = metrica_csv.load(int(game[len("MET-G"):].rstrip("N")))
        src_missing = float(np.isnan(match.ball[match.graded(), 0]).mean())
        out, meta = degrade.write_sim(match, dcfg, mcfg, args.seed, game=game)
        atd, _ = load_game(game, out)
        sp = registry.spec(game, out)
        graded = registry.graded_mask(sp)
        s = stats.atd_stats(atd.players_m()[..., :2].astype(float), atd.ball_m(), atd.team_of_slot,
                            graded, atd.pitch, mcfg, meta["fps"])
        p11 = [s["visible"][team][11] / s["visible"][team][7:12].sum() for team in (0, 1)]
        print("{:<12} {:>8.3f} {:>10.3f} {:>8.0f} {:>6.3f} {:>6.3f} {:>7.2f}".format(
            meta["game"], s["ball_missing_frac"], 1e3 * s["n_gaps"] / graded.sum(),
            s["gap_median"], p11[0], p11[1], atd.n_objects()[graded].mean()))
        print("  (source ball already missing on {:.3f} of graded frames pre-degradation; "
              "kickoff_team {})".format(src_missing, meta["kickoff_team"]))
        rows.append((game, sp, atd, graded))

    print("\n== acceptance: official baseline vs the mechanical GT (sim_check.py logic) ==")
    print("{:<12} {:>7} {:>7} {:>7} {:>7}   {:>8} {:>6} {:>6} {:>6}".format(
        "game", "raw", "s25", "s50", "s125", "spells/90", "<2s", "<6s", "D"))
    curves = {}
    for game, sp, atd, graded in rows:
        codes = nearest_player_codes(atd)
        f1 = {w: score(starter.smooth(codes, w), "sim:" + game)["spell_f1_mean"]
              for w in sim_check.REAL_FULL_MATCH}
        gt = pm.read_label_csv(os.path.join(registry.data_dir_for(game), "gt.csv"))
        gt = gt["label"].to_numpy(dtype=str)
        sl = sim_check.spells(gt, sp["half_blocks"])
        minutes = graded.sum() / sp["fps"] / 60
        curves[game] = f1
        print("{:<12} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}   {:>8.0f} {:>6.2f} {:>6.2f} {:>6.2f}".format(
            game, f1[0], f1[25], f1[50], f1[125], len(sl) / minutes * 90,
            np.mean(sl < 2 * sp["fps"]), np.mean(sl < 6 * sp["fps"]), np.mean(gt[graded] == "D")))
    print("{:<12} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}   (DFL sims: spells/90 ~330-480, "
          "<2s 0.25-0.37, D 0.41-0.42)".format(
              "real (ofc.)", *sim_check.REAL_FULL_MATCH.values()))
    mean50 = float(np.mean([c[50] for c in curves.values()]))
    order = all(c[50] > c[125] > c[25] for c in curves.values())
    ok = accept[0] <= mean50 <= accept[1]
    print("\n  [{}] metrica smooth50 {:.4f} within {}-{} (full-match reference 0.2433); "
          "curve order smooth50>125>25: {}".format("PASS" if ok else "FAIL", mean50,
                                                   accept[0], accept[1], order))

    if args.family == "csv":
        print("  (csv family: band reported, not gated)")
        sys.exit(0)
    print("\n== event->label derivation census (io/metrica.py docstring) ==")
    for k, v in sorted(metrica.census().items()):
        print("  {:<34} {}".format(k, v))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
