"""Harness for tuning and comparing rules on the degraded open-data sim (PLAN.md §6.1).

A SimMatch holds one DFL-<id> sim match with everything the rules need precomputed (cleaned ball,
per-team distances, GT) and caches touch detections per touch config, so coordinate descent over
many configs stays cheap. Scores come from the official scorer.
"""

import os

import numpy as np

from mtah import registry
from mtah.eval.score import pm
from mtah.events.touches import touch_events
from mtah.features.ball import clean_ball
from mtah.features.players import team_distances
from mtah.io.atd import load_game


class SimMatch:
    def __init__(self, game, cfg):
        self.game = game
        self.spec = registry.spec(game)
        self.fps = self.spec["fps"]
        self.atd, _ = load_game(game)
        gt = pm.read_label_csv(os.path.join(registry.data_dir_for(game), "gt.csv"))
        self.gt = gt["label"].to_numpy(dtype=str)
        self.gt_codes = pm.label_codes(self.gt)
        self.graded = registry.graded_mask(self.spec)
        self.dead_gt = self.gt == "D"
        self.ball_raw = self.atd.ball_m()
        self.players = self.atd.players_m()[..., :2].astype(float)
        self.ball, self.observed, _ = clean_ball(self.ball_raw, self.atd.pitch, cfg["ball"], self.fps)
        self.d, _ = team_distances(self.players, self.ball, self.atd.team_of_slot)
        self._touches = {}

    def touches(self, tcfg):
        key = tuple(sorted(tcfg.items()))
        if key not in self._touches:
            self._touches[key] = touch_events(self.ball, self.d, tcfg, self.fps)
        return self._touches[key]

    def evaluate(self, labels):
        r = pm.evaluate(self.gt_codes, pm.label_codes(labels), bootstrap_n=0)
        tr = r["transitions"][25]
        return {"spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
                "alive_dead_f1": r["alive_dead_f1"],
                "spells": r["n_pred_spells"] / r["n_gt_spells"], "trans_f1": tr["f1"],
                "latency": tr["median_latency"]}


def coordinate_descent(objective, params, grids, rounds=3):
    """Maximise objective(params) one parameter at a time over its grid. grids: {section:
    {name: values}}, params: {section: {name: value}}. Returns (params, best score)."""
    params = {k: dict(v) for k, v in params.items()}
    best = objective(params)
    for _ in range(rounds):
        changed = False
        for section, grid in grids.items():
            for name, values in grid.items():
                for v in values:
                    trial = {k: dict(p) for k, p in params.items()}
                    trial[section][name] = v
                    s = objective(trial)
                    if s > best + 1e-9:
                        best, params, changed = s, trial, True
        if not changed:
            break
    return params, best


def table_row(name, results):
    mean = {k: np.mean([r[k] for r in results if r[k] is not None]) for k in results[0]}
    per = " ".join("{:.4f}".format(r["spell_f1"]) for r in results)
    return ("{:<22} {}  | mean {:.4f}  macro {:.4f}  alive/dead F1 {:.3f}  spells x{:.2f}  "
            "trans F1 {:.3f}  med lat {:+.1f}").format(
        name, per, mean["spell_f1"], mean["macro_f1"], mean["alive_dead_f1"], mean["spells"],
        mean["trans_f1"], mean["latency"])
