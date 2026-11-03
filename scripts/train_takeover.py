"""make train-takeover [CFG=configs/rules_v3.yaml]: train the takeover model on the sim, tune the
decoder out of fold, compare on held-out. Open data only (degraded DFL/IDSSE sim, PLAN.md §6.1).

1. Contacts and features on every sim match (learn/takeover.py); training rows from the sim GT.
2. Out of fold: for each fit match, a model trained on the other four predicts it. alpha and
   flip_penalty are tuned on those predictions, end to end (the config's detected D, spellF1).
3. The final model is trained on all five fit matches (twice, to check it is byte-identical) and
   written to owner.model. The held-out matches then compare it with the config's rules decoder
   (owner.hypothesis), with the GT's D and with detected D.
Set the printed alpha and flip_penalty in the config; the model file is committed.
"""

import argparse
import os

import numpy as np

from mtah import config
from mtah.decode.rules import decode_owner
from mtah.decode.viterbi import decode_viterbi
from mtah.eval.simeval import SimMatch, table_row
from mtah.events.deadball import dead_runs, evidence
from mtah.learn import gapstate, takeover

ALPHA = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
PENALTY = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


class Prepared:
    def __init__(self, game, cfg):
        m = self.m = SimMatch(game, cfg)
        self.cfg = cfg
        oc = cfg["owner"]
        self.events = takeover.contacts(m.ball, m.d, cfg["touch"], oc["contact_r_m"], m.fps)
        if oc.get("shape_features"):
            gc = cfg["gaps"]
            self.F = gapstate.features(m.players, m.atd.team_of_slot,
                                       gapstate.home_direction(m.atd, m.spec, {},
                                                               cfg["direction"]["window_frames"]),
                                       m.atd.pitch, gc, m.fps)
        F = getattr(self, "F", None)
        self.X = takeover.design(self.events, m.d, F,
                                 gapstate.feature_names(cfg["gaps"]) if F is not None else None,
                                 cfg, m.fps)
        self.rows, self.y = takeover.training_rows(self.events, m.gt, m.graded,
                                                   oc["label_lag_frames"],
                                                   oc.get("label_policy", "legacy"))
        ev = evidence(m.ball_raw, m.ball, m.observed, m.players, m.graded, m.atd.pitch,
                      cfg["deadball"], m.fps)
        # As mtah.pipeline computes them: D runs with their evidence ends (postprocess backshift)
        # and any release restarts (injected as certain takeovers, takeover.with_restarts).
        self.restarts, self.runs_meta = [], []
        self.dead_runs = dead_runs(ev, m.touches(cfg["touch"]), m.graded, cfg["deadball"],
                                   tuple(cfg["deadball"]["kinds"]), out_restarts=self.restarts,
                                   out_meta=self.runs_meta)
        self.dead = np.zeros(len(m.graded), dtype=bool)
        for a, b, _ in self.dead_runs:
            self.dead[a:b + 1] = True
        self.dead &= m.graded

    def labels(self, p, alpha, penalty, dead=None, inject_restarts=True):
        events = self.events
        if inject_restarts and self.restarts:
            events, p = takeover.with_restarts(events, p, self.restarts,
                                               self.cfg["deadball"].get("restart_takeover_p",
                                                                        1.0))
        return decode_viterbi(events, p, self.m.graded, self.dead if dead is None else dead,
                              alpha, penalty)


def fit_on(prepared, params):
    names = takeover.feature_names(prepared[0].cfg)
    return takeover.train(np.concatenate([q.X[q.rows] for q in prepared]),
                          np.concatenate([q.y for q in prepared]), params, names)


def predict(booster, q):
    return takeover.predict(booster, q.X)


def auc(p, y):
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    return float(np.cumsum(ys)[ys == 0].sum() / (ys.sum() * (1 - ys).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rules_v3.yaml")
    cfg = config.load(os.path.abspath(parser.parse_args().config))
    params = cfg["takeover_model"]
    fit = [Prepared("DFL-" + g, cfg) for g in cfg["matches"]["fit"]]
    held = [Prepared("DFL-" + g, cfg) for g in cfg["matches"]["holdout"]]
    print("training rows (fit): {}, takeover rate {:.3f}".format(
        sum(len(q.y) for q in fit), np.concatenate([q.y for q in fit]).mean()))

    oof = [predict(fit_on([r for r in fit if r is not q], params), q) for q in fit]
    best = None
    for a in ALPHA:
        for lam in PENALTY:
            s = float(np.mean([q.m.evaluate(q.labels(p, a, lam))["spell_f1"]
                               for q, p in zip(fit, oof)]))
            if best is None or s > best[0]:
                best = (s, a, lam)
    score, alpha, penalty = best
    print("out-of-fold fit spellF1 {:.4f} at alpha {} flip_penalty {} (grid edges: alpha {}-{}, "
          "penalty {}-{})".format(score, alpha, penalty, ALPHA[0], ALPHA[-1], PENALTY[0],
                                  PENALTY[-1]))

    booster = fit_on(fit, params)
    again = fit_on(fit, params)
    path = takeover.model_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    booster.save_model(path)
    print("model -> {} ({} KB); retrained byte-identical: {}".format(
        os.path.relpath(path), os.path.getsize(path) // 1024,
        booster.model_to_string() == again.model_to_string()))
    p_held = [predict(booster, q) for q in held]
    print("held-out contacts: AUC {:.3f}".format(
        auc(np.concatenate([p[q.rows] for p, q in zip(p_held, held)]),
            np.concatenate([q.y for q in held]))))

    oc = cfg["owner"]
    print("\nheld-out ({}):".format(", ".join(cfg["matches"]["holdout"])))
    for dname in ("GT D", "detected D"):
        def dead(q):
            return q.m.dead_gt if dname == "GT D" else q.dead
        print(table_row("{} {}".format(oc["hypothesis"], dname),
                        [q.m.evaluate(decode_owner(q.m.touches(cfg["touch"]), q.m.graded, dead(q),
                                                   oc["hypothesis"], oc, q.m.fps)) for q in held]))
        print(table_row("viterbi " + dname,
                        [q.m.evaluate(q.labels(p, alpha, penalty, dead(q)))
                         for q, p in zip(held, p_held)]))
    print("\nset owner.alpha: {} and owner.flip_penalty: {} in the config".format(alpha, penalty))


if __name__ == "__main__":
    main()
