"""make train-hybrid [CFG=configs/rules_v11.yaml]: train the P3e hybrid dead model and write
deadhybrid.model (committed only on adoption). Open data only (degraded DFL/IDSSE sim, §6.1).

The model is per-frame P(dead) trained on the sim fit matches' realistic-dead-ball variants
(deadhybrid.train_conditions, the v8 protocol) with the CAUSAL feature set - no future ball
speed (learn/deadstate.py causal_ball). Only the decode differs from v8: the model refines each
evidence-rule boundary within +/- deadhybrid.window_frames (decode/dead.py refine_runs) and can
never create, delete or relocate a run. The window is picked nested-OOF by
handoff/fable/exp_hybrid.py; the §6.4 comparison is make paired B=configs/rules_v11.yaml.
Trained twice to check byte-identity, as every committed model is.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paired  # noqa: E402
from train_takeover import Prepared  # noqa: E402

from mtah import config, registry  # noqa: E402
from mtah.learn import deadstate, gapstate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rules_v11.yaml")
    cfg = config.load(os.path.abspath(ap.parse_args().config))
    Q = [Prepared("DFL-" + g, cfg) for g in cfg["matches"]["fit"]]
    names = deadstate.causal_names(gapstate.feature_names(cfg["gaps"]))
    rows, _ = paired.dead_training_rows(cfg, Q, len(Q), "deadhybrid", deadstate.causal_ball)
    key = tuple(range(len(Q)))
    print("training rows (fit, all conditions): {}".format(
        sum(len(rows[j][ci][1]) for j in range(len(Q)) for ci in range(len(rows[0])))))
    booster = paired.train_dead_on(cfg, rows, key, names)
    again = paired.train_dead_on(cfg, rows, key, names)
    path = os.path.join(registry.ROOT, cfg["deadhybrid"]["model"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    booster.save_model(path)
    print("model -> {} ({} KB); retrained byte-identical: {}".format(
        os.path.relpath(path, registry.ROOT), os.path.getsize(path) // 1024,
        booster.model_to_string() == again.model_to_string()))


if __name__ == "__main__":
    main()
