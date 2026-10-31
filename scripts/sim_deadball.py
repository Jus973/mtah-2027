"""make sim-dead: tune dead-ball v1 on the sim's fit matches; ablate and compare on held-out.

The objective is end-to-end: spellF1 of the configs/rules.yaml owner rule with detected D on the
fit matches (the touch settings stay as rules.yaml has them). Held-out matches then show, next to
the oracle-D result, each evidence kind's contribution and H1 vs H2 with detected D.

Circularity warning: the sim loses the ball in stoppages by construction (learn/degrade.py,
fitted q = 1), so the "gap" evidence is partly the degradation model scoring itself. The real
ATD supports it (players move half as fast while its ball is missing), and "oob+still" is shown
as the variant that does not lean on it. Open data only.
"""

import numpy as np

from mtah import config
from mtah.decode.rules import decode_owner
from mtah.eval.simeval import SimMatch, coordinate_descent, table_row
from mtah.events.deadball import KINDS, dead_mask, evidence

DEAD_GRID = {  # wide enough that no tuned value sits on an edge
    "oob_margin_m": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
    "oob_min_frames": [1, 2, 5, 10, 15, 25, 50],
    "gap_min_frames": [5, 10, 15, 25, 50, 75, 100, 150, 250],
    "gap_offset_frames": [-50, -25, -12, 0, 12, 25],
    "v_still_mps": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0],
    "v_players_mps": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "still_min_frames": [10, 25, 50, 75, 125, 250],
    "merge_frames": [0, 12, 25, 50, 100, 200],
}


def labels(m, params, kinds=KINDS, dead=None):
    touches = m.touches(params["touch"])
    if dead is None:
        ev = evidence(m.ball_raw, m.ball, m.observed, m.players, m.graded, m.atd.pitch,
                      params["deadball"], m.fps)
        dead = dead_mask(ev, touches, m.graded, params["deadball"], kinds)
    return decode_owner(touches, m.graded, dead, params["owner"]["hypothesis"], params["owner"],
                        m.fps)


def d_quality(m, params, kinds=KINDS):
    ev = evidence(m.ball_raw, m.ball, m.observed, m.players, m.graded, m.atd.pitch,
                  params["deadball"], m.fps)
    dm = dead_mask(ev, m.touches(params["touch"]), m.graded, params["deadball"], kinds)
    tp = (dm & m.dead_gt).sum()
    return dm[m.graded].mean(), tp / max(1, dm.sum()), tp / max(1, m.dead_gt.sum())


def main():
    cfg = config.load("rules.yaml")
    h2 = config.load("rules_h2.yaml")
    fit = [SimMatch("DFL-" + m, cfg) for m in cfg["matches"]["fit"]]
    held = [SimMatch("DFL-" + m, cfg) for m in cfg["matches"]["holdout"]]
    start = {"touch": cfg["touch"], "owner": cfg["owner"], "deadball": cfg["deadball"]}

    def objective(params):
        return float(np.mean([m.evaluate(labels(m, params))["spell_f1"] for m in fit]))

    params, score = coordinate_descent(objective, start, {"deadball": DEAD_GRID})
    print("deadball tuned on fit (mean spellF1 {:.4f}): {}".format(score, params["deadball"]))
    h2_params = {"touch": h2["touch"], "owner": h2["owner"], "deadball": params["deadball"]}

    print("\nheld-out ({}):".format(", ".join(cfg["matches"]["holdout"])))
    print(table_row("H1 + oracle D", [m.evaluate(labels(m, params, dead=m.dead_gt)) for m in held]))
    for kinds in (KINDS, ("oob", "still"), ("gap",), ("oob",), ("still",)):
        print(table_row("H1 + D " + "+".join(kinds), [m.evaluate(labels(m, params, kinds))
                                                      for m in held]))
    print(table_row("H2 + D oob+gap+still", [m.evaluate(labels(m, h2_params)) for m in held]))
    print("\nD quality on held-out (D share of graded / precision / recall; GT D share {:.3f}):".format(
        np.mean([m.dead_gt[m.graded].mean() for m in held])))
    for kinds in (KINDS, ("oob", "still")):
        q = np.mean([d_quality(m, params, kinds) for m in held], axis=0)
        print("  {:<14} {:.3f} / {:.3f} / {:.3f}".format("+".join(kinds), *q))


if __name__ == "__main__":
    main()
