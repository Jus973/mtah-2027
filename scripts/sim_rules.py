"""make sim-rules: tune owner hypotheses H1-H3 on the sim's fit matches, compare them on held-out.

D comes from the sim GT (an oracle), so this compares the owner rule alone; the dead-ball
detector is separate. Each hypothesis gets its own coordinate descent over the touch parameters
(and, for H2, its control parameters) maximising mean spellF1 on the fit matches; the tuned
versions and the official baseline are then scored on the held-out matches, which are used for
nothing else. Open data only: the real match is never touched.
"""

import os
import sys

import numpy as np

from mtah import config, registry
from mtah.decode.rules import HYPOTHESES, decode_owner
from mtah.eval.score import pm
from mtah.events.touches import touch_events
from mtah.features.ball import clean_ball
from mtah.features.baseline import nearest_player_codes
from mtah.features.players import team_distances
from mtah.io.atd import load_game

sys.path.insert(0, os.path.join(registry.ROOT, "starter"))
import baseline as starter  # noqa: E402  (vendored, unmodified)

# Wide enough that no tuned value sits on an edge (an edge value would mean "look further").
# 180 degrees / 50 m/s switch those touch criteria off.
TOUCH_GRID = {"r_touch_m": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
              "min_frames": [1, 2, 3, 5, 8, 12, 20],
              "min_turn_deg": [10, 20, 35, 60, 90, 135, 180],
              "min_dv_mps": [1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 50.0]}
OWNER_GRID = {"H2": {"n_c": [1, 2, 3, 5, 8, 12, 20], "v_c_mps": [0.0, 1.0, 2.0, 4.0, 6.0, 9.0, 13.0],
                     "t_c_s": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]}}


class Match:
    def __init__(self, game, cfg):
        sp = registry.spec(game)
        self.atd, _ = load_game(game)
        gt = pm.read_label_csv(os.path.join(registry.data_dir_for(game), "gt.csv"))
        self.gt = gt["label"].to_numpy(dtype=str)
        self.gt_codes = pm.label_codes(self.gt)
        self.graded = registry.graded_mask(sp)
        self.dead = self.gt == "D"
        self.ball, _, _ = clean_ball(self.atd.ball_m(), self.atd.pitch, cfg["ball"])
        self.d, _ = team_distances(self.atd.players_m()[..., :2].astype(float), self.ball,
                                   self.atd.team_of_slot)
        self._touches = {}

    def touches(self, tcfg):
        key = tuple(sorted(tcfg.items()))
        if key not in self._touches:
            self._touches[key] = touch_events(self.ball, self.d, tcfg)
        return self._touches[key]

    def evaluate(self, labels):
        r = pm.evaluate(self.gt_codes, pm.label_codes(labels), bootstrap_n=0)
        tr = r["transitions"][25]
        return {"spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
                "spells": r["n_pred_spells"] / r["n_gt_spells"], "trans_f1": tr["f1"],
                "latency": tr["median_latency"]}


def run(match, hyp, tcfg, ocfg):
    return decode_owner(match.touches(tcfg), match.graded, match.dead, hyp, ocfg)


def tune(matches, hyp, cfg):
    tcfg, ocfg = dict(cfg["touch"]), dict(cfg["owner"])

    def objective(t, o):
        return float(np.mean([m.evaluate(run(m, hyp, t, o))["spell_f1"] for m in matches]))

    best = objective(tcfg, ocfg)
    for _ in range(3):
        changed = False
        for grid, which in ((TOUCH_GRID, "t"), (OWNER_GRID.get(hyp, {}), "o")):
            for name, values in grid.items():
                for v in values:
                    t = dict(tcfg, **{name: v}) if which == "t" else tcfg
                    o = dict(ocfg, **{name: v}) if which == "o" else ocfg
                    s = objective(t, o)
                    if s > best + 1e-9:
                        best, tcfg, ocfg, changed = s, t, o, True
        if not changed:
            break
    return tcfg, ocfg, best


def row(name, results):
    mean = {k: np.mean([r[k] for r in results if r[k] is not None]) for k in results[0]}
    per = " ".join("{:.4f}".format(r["spell_f1"]) for r in results)
    return "{:<16} {}  | mean {:.4f}  macro {:.4f}  spells x{:.2f}  trans F1 {:.3f}  med lat {:+.1f}".format(
        name, per, mean["spell_f1"], mean["macro_f1"], mean["spells"], mean["trans_f1"],
        mean["latency"])


def main():
    cfg = config.load("rules.yaml")
    fit = [Match("DFL-" + m, cfg) for m in cfg["matches"]["fit"]]
    held = [Match("DFL-" + m, cfg) for m in cfg["matches"]["holdout"]]
    tuned = {}
    for hyp in HYPOTHESES:
        tcfg, ocfg, score = tune(fit, hyp, cfg)
        tuned[hyp] = (tcfg, ocfg)
        params = {k: tcfg[k] for k in TOUCH_GRID}
        params.update({k: ocfg[k] for k in OWNER_GRID.get(hyp, {})})
        print("{} tuned on fit (mean spellF1 {:.4f}): {}".format(hyp, score, params))

    print("\nheld-out ({}), oracle D for H1-H3:".format(", ".join(cfg["matches"]["holdout"])))
    for name, labels_of in [
            ("baseline s50", lambda m: starter.smooth(nearest_player_codes(m.atd), 50)),
            ("s50 + oracle D", lambda m: np.where(m.dead, "D",
                                                  starter.smooth(nearest_player_codes(m.atd), 50)))]:
        print(row(name, [m.evaluate(labels_of(m)) for m in held]))
    for hyp, (tcfg, ocfg) in tuned.items():
        print(row(hyp + " + oracle D", [m.evaluate(run(m, hyp, tcfg, ocfg)) for m in held]))


if __name__ == "__main__":
    main()
