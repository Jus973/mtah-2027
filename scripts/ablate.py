"""make sim-ablate [CFG=configs/final.yaml]: PLAN.md M3 ablations on the held-out sim, and the
video ball-recovery go/no-go condition (b). Open data only.

Each row removes or swaps one component of the config and scores the held-out matches with the
official scorer (bootstrap 1000 per match). PLAN.md M3: keep only components with a positive
effect; drop anything whose removal changes held-out spellF1 by < 0.01 (simplicity).
Go/no-go (b): of the GT spells the config misses or splits (no predicted spell of the same team
at IoU >= 0.5), the share that coincide with ball gaps (ball missing on >= half their frames).
"""

import argparse
import copy
import os

import numpy as np

from mtah import config, pipeline, registry
from mtah.eval.score import pm, score

BIG = 10 ** 9


def variants(cfg):
    rows = [("final", cfg)]
    v1 = config.load("rules.yaml")
    rows.append(("rules only (v1: H1 over touches)", v1))
    for kind, key in (("out of bounds", "oob_min_frames"), ("long gaps", "gap_min_frames"),
                      ("stillness", "still_min_frames")):
        c = copy.deepcopy(cfg)
        c["deadball"][key] = BIG
        rows.append(("without {} evidence".format(kind), c))
    c = copy.deepcopy(cfg)
    c["owner"]["alpha"] = 1.0
    rows.append(("no log-odds sharpening (alpha 1)", c))
    if cfg.get("gaps", {}).get("enabled"):
        c = copy.deepcopy(cfg)
        c["gaps"]["enabled"] = False
        rows.append(("without the gap decoder", c))
    if cfg["owner"].get("shape_features"):
        c = copy.deepcopy(cfg)
        v5 = config.load("rules_v5.yaml")["owner"]["model"]
        c["owner"].update(shape_features=False, model=v5)
        rows.append(("without team shape at contacts (v5)", c))
    return rows


def spells(codes, team):
    return [(a, b) for a, b, c in pm.segments(codes, pm.valid_runs(codes)) if c == team]


def missed_in_gaps(pred, gt, missing):
    """Share of GT spells without a same-team predicted spell at IoU >= 0.5 that lie mostly in a
    ball gap, and the number of such spells."""
    g, p = pm.label_codes(gt), pm.label_codes(pred)
    missed, in_gap = 0, 0
    for team in (pm.H, pm.A):
        ps = spells(p, team)
        starts = np.array([a for a, _ in ps])
        for a, b in spells(g, team):
            i = np.searchsorted(starts, b)
            best = 0.0
            for pa, pb in ps[max(0, i - 3):i + 1]:
                inter = max(0, min(b, pb) - max(a, pa))
                union = max(b, pb) - min(a, pa)
                best = max(best, inter / union if union else 0.0)
            if best < 0.5:
                missed += 1
                in_gap += missing[a:b].mean() >= 0.5
    return (in_gap / missed if missed else None), missed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    cfg = config.load(os.path.abspath(parser.parse_args().config))
    held = ["DFL-" + g for g in cfg["matches"]["holdout"]]
    base = None
    print("held-out ({}), spellF1 [95% CI] per match, mean, change vs final:".format(
        ", ".join(cfg["matches"]["holdout"])))
    for name, c in variants(cfg):
        res = [score(pipeline.run(g, c)[0], "sim:" + g, bootstrap=1000) for g in held]
        mean = float(np.mean([r["spell_f1_mean"] for r in res]))
        base = mean if base is None else base
        cells = "  ".join("{:.4f} [{:.3f}, {:.3f}]".format(r["spell_f1_mean"], r["spell_bootstrap_low"],
                                                         r["spell_bootstrap_high"]) for r in res)
        print("  {:<36} {}  mean {:.4f}  {:+.4f}  macro {:.4f}".format(
            name, cells, mean, mean - base, np.mean([r["macro_f1"] for r in res])))
    print("\nvideo ball-recovery go/no-go (b): missed or split GT spells that lie in ball gaps")
    for g in held:
        from mtah.io.atd import load_game
        atd, _ = load_game(g)
        missing = ~np.isfinite(atd.ball_m()[:, 0])
        gt = pm.read_label_csv(os.path.join(registry.data_dir_for(g), "gt.csv"))["label"].to_numpy(str)
        share, n = missed_in_gaps(pipeline.run(g, cfg)[0], gt, missing)
        print("  {}: {:.1%} of {} missed/split spells (PLAN.md go needs >= 25%)".format(g, share, n))


if __name__ == "__main__":
    main()
