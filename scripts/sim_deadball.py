"""make sim-dead [CFG=configs/rules.yaml]: tune the dead-ball rules on the sim's fit matches;
ablate and compare on held-out.

The objective is end-to-end: spellF1 of the config's owner rule with detected D on the fit
matches. Tuned: the dead-ball evidence (including the calm-gap filter) and the launch test that
lets a restart kick end a stoppage (touch.min_launch_mps, touch.rest_mps); the other touch
settings stay as the config has them. Held-out matches then show the starting config, the tuned
one, each change's contribution, each evidence kind's contribution and H1 vs H2 with detected D.

Circularity warning: the sim loses the ball in stoppages by construction (learn/degrade.py,
fitted q = 1), so the "gap" evidence is partly the degradation model scoring itself. The real
ATD supports it (players move half as fast while its ball is missing), and "oob+still" is shown
as the variant that does not lean on it. Open data only.
"""

import argparse
import os

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
    "gap_v_players_mps": [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 99.0],   # 99 = off
    "gap_speed_window_frames": [1, 25, 50, 100, 250],
    "gap_calm_min_frames": [1, 12, 25, 50, 100],
    "restart_window_frames": [None, 0, 6, 12, 25, 50, 100],             # None = off
    "restart_flight_mps": [2.0, 4.0, 6.0, 8.0, 12.0, 99.0],             # 99 = off
}
LAUNCH_GRID = {"min_launch_mps": [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 99.0],   # 99 = off
               "rest_mps": [0.25, 0.5, 1.0, 2.0, 4.0]}


def labels(m, params, kinds=None, dead=None):
    kinds = kinds or tuple(params["deadball"]["kinds"])
    touches = m.touches(params["touch"])
    if dead is None:
        ev = evidence(m.ball_raw, m.ball, m.observed, m.players, m.graded, m.atd.pitch,
                      params["deadball"], m.fps)
        restarts = []
        dead = dead_mask(ev, touches, m.graded, params["deadball"], kinds, restarts)
        touches = sorted(touches + restarts, key=lambda t: t.start)
    return decode_owner(touches, m.graded, dead, params["owner"]["hypothesis"], params["owner"],
                        m.fps)


def d_quality(m, params, kinds=None):
    kinds = kinds or tuple(params["deadball"]["kinds"])
    ev = evidence(m.ball_raw, m.ball, m.observed, m.players, m.graded, m.atd.pitch,
                  params["deadball"], m.fps)
    dm = dead_mask(ev, m.touches(params["touch"]), m.graded, params["deadball"], kinds)
    tp = (dm & m.dead_gt).sum()
    return dm[m.graded].mean(), tp / max(1, dm.sum()), tp / max(1, m.dead_gt.sum())


def with_(params, section, **values):
    out = {k: dict(v) for k, v in params.items()}
    out[section].update(values)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rules.yaml")
    cfg = config.load(os.path.abspath(parser.parse_args().config))
    h2 = config.load("rules_h2.yaml")
    fit = [SimMatch("DFL-" + m, cfg) for m in cfg["matches"]["fit"]]
    held = [SimMatch("DFL-" + m, cfg) for m in cfg["matches"]["holdout"]]
    start = {"touch": cfg["touch"], "owner": cfg["owner"], "deadball": cfg["deadball"]}

    def objective(params):
        return float(np.mean([m.evaluate(labels(m, params))["spell_f1"] for m in fit]))

    params, score = coordinate_descent(objective, start,
                                       {"deadball": DEAD_GRID, "touch": LAUNCH_GRID})
    print("tuned on fit (mean spellF1 {:.4f}, start {:.4f}):".format(score, objective(start)))
    print("  deadball:", params["deadball"])
    print("  touch:", {k: params["touch"][k] for k in LAUNCH_GRID})
    h2_params = {"touch": dict(h2["touch"], **{k: params["touch"][k] for k in LAUNCH_GRID}),
                 "owner": h2["owner"], "deadball": params["deadball"]}

    print("\nheld-out ({}):".format(", ".join(cfg["matches"]["holdout"])))
    v1 = config.load("rules.yaml")
    v1 = {"touch": v1["touch"], "owner": v1["owner"], "deadball": v1["deadball"]}
    print(table_row("rules.yaml (v1)", [m.evaluate(labels(m, v1)) for m in held]))
    print(table_row("start config", [m.evaluate(labels(m, start)) for m in held]))
    print(table_row("tuned", [m.evaluate(labels(m, params)) for m in held]))
    print(table_row("tuned, no release", [m.evaluate(labels(m, with_(params, "deadball",
                                                                        restart_window_frames=None)))
                                          for m in held]))
    print(table_row("tuned, no flight", [m.evaluate(labels(m, with_(params, "deadball",
                                                                       restart_flight_mps=99.0)))
                                         for m in held]))
    print(table_row("tuned, no launch", [m.evaluate(labels(m, with_(params, "touch",
                                                                       min_launch_mps=99.0)))
                                         for m in held]))
    print(table_row("tuned, no calm gap", [m.evaluate(labels(m, with_(params, "deadball",
                                                                         gap_v_players_mps=99.0)))
                                           for m in held]))
    print(table_row("H1 + oracle D", [m.evaluate(labels(m, params, dead=m.dead_gt)) for m in held]))
    for kinds in (KINDS, ("oob", "still"), ("gap",), ("oob",), ("still",)):
        print(table_row("H1 + D " + "+".join(kinds), [m.evaluate(labels(m, params, kinds))
                                                      for m in held]))
    print(table_row("H2 + D oob+gap+still", [m.evaluate(labels(m, h2_params)) for m in held]))
    print("\nD quality on held-out (D share of graded / precision / recall; GT D share {:.3f}):".format(
        np.mean([m.dead_gt[m.graded].mean() for m in held])))
    for name, p in (("rules.yaml (v1)", v1), ("tuned", params)):
        for kinds in (KINDS, ("oob", "still")):
            q = np.mean([d_quality(m, p, kinds) for m in held], axis=0)
            print("  {:<14} {:<14} {:.3f} / {:.3f} / {:.3f}".format(name, "+".join(kinds), *q))


if __name__ == "__main__":
    main()
