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
from mtah.eval.simeval import SimMatch, coordinate_descent, table_row
from mtah.features.baseline import nearest_player_codes

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


def run(match, hyp, params):
    return decode_owner(match.touches(params["touch"]), match.graded, match.dead_gt, hyp,
                        params["owner"], match.fps)


def tune(matches, hyp, cfg):
    def objective(params):
        return float(np.mean([m.evaluate(run(m, hyp, params))["spell_f1"] for m in matches]))
    grids = {"touch": TOUCH_GRID, "owner": OWNER_GRID.get(hyp, {})}
    return coordinate_descent(objective, {"touch": cfg["touch"], "owner": cfg["owner"]}, grids)


def main():
    cfg = config.load("rules.yaml")
    fit = [SimMatch("DFL-" + m, cfg) for m in cfg["matches"]["fit"]]
    held = [SimMatch("DFL-" + m, cfg) for m in cfg["matches"]["holdout"]]
    tuned = {}
    for hyp in HYPOTHESES:
        params, score = tune(fit, hyp, cfg)
        tuned[hyp] = params
        shown = {k: params["touch"][k] for k in TOUCH_GRID}
        shown.update({k: params["owner"][k] for k in OWNER_GRID.get(hyp, {})})
        print("{} tuned on fit (mean spellF1 {:.4f}): {}".format(hyp, score, shown))

    print("\nheld-out ({}), oracle D for H1-H3:".format(", ".join(cfg["matches"]["holdout"])))
    base = {m.game: starter.smooth(nearest_player_codes(m.atd), 50) for m in held}
    print(table_row("baseline s50", [m.evaluate(base[m.game]) for m in held]))
    print(table_row("s50 + oracle D", [m.evaluate(np.where(m.dead_gt, "D", base[m.game]))
                                       for m in held]))
    for hyp, params in tuned.items():
        print(table_row(hyp + " + oracle D", [m.evaluate(run(m, hyp, params)) for m in held]))


if __name__ == "__main__":
    main()
