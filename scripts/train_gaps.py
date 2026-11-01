"""make train-gaps [CFG=configs/rules_v5.yaml]: train the gap models on the sim, tune the gap
decoder out of fold, and compare with the same config without it (PLAN.md §6.4, paired). Open data
only (degraded DFL/IDSSE sim, PLAN.md §6.1).

1. The config's labels without the gap decoder on every sim match: the takeover model out of fold
   on the fit matches (as make train-takeover), trained on all five for the held-out ones.
2. Gap features (learn/gapstate.py) on every match; rows from the sim GT on long-gap frames.
3. Out of fold: for each fit match, gap models trained on the other four predict it. lam_dead and
   lam_owner are tuned on those predictions, end to end (spellF1).
4. The final gap models are trained on all five fit matches (twice, to check they are
   byte-identical) and written to gaps.dead_model and gaps.owner_model; the held-out matches use
   them.
5. Paired per-match comparison: held-out matches always count, fit matches out of fold. Mean
   difference, wins and the one-sided sign test.
Set the printed lam_dead and lam_owner in the config; the model files are committed.
"""

import argparse
import itertools
import os
from math import comb

import numpy as np

from mtah import config
from mtah.decode.gaps import decode_gaps
from mtah.learn import gapstate
from train_takeover import Prepared, fit_on, predict

LAM_DEAD = [20.0, 30.0, 40.0, 60.0, 80.0]
LAM_OWNER = [10.0, 15.0, 20.0, 30.0, 40.0]


class Gaps:
    def __init__(self, q, gc):
        m = q.m
        self.q = q
        self.gap = gapstate.long_gaps(m.observed, m.graded, gc["min_frames"])
        self.F = gapstate.features(m.players, m.atd.team_of_slot,
                                   gapstate.home_direction(m.atd, m.spec, {}), m.atd.pitch, gc,
                                   m.fps)
        self.rows = gapstate.training_rows(self.F, self.gap, m.gt, gc["step_frames"])
        self.idx = np.flatnonzero(self.gap)

    def probs(self, models):
        T = len(self.gap)
        p_dead, p_home = np.full(T, np.nan), np.full(T, np.nan)
        p_dead[self.idx], p_home[self.idx] = gapstate.predict(models, self.F, self.idx)
        return p_dead, p_home


def fit_gaps(gs, params, names):
    return {k: gapstate.train(np.vstack([g.rows[k][0] for g in gs]),
                              np.concatenate([g.rows[k][1] for g in gs]), params, names)
            for k in ("dead", "owner")}


def sign_test(wins, n):
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rules_v5.yaml")
    cfg = config.load(os.path.abspath(parser.parse_args().config))
    gc, oc, params = cfg["gaps"], cfg["owner"], cfg["gap_model"]
    names = gapstate.feature_names(gc)
    fit_ids, held_ids = cfg["matches"]["fit"], cfg["matches"]["holdout"]
    Q = [Prepared("DFL-" + g, cfg) for g in fit_ids + held_ids]
    fit_q = Q[:len(fit_ids)]
    p = [predict(fit_on([r for r in fit_q if r is not q], cfg["takeover_model"]), q)
         for q in fit_q]
    full = fit_on(fit_q, cfg["takeover_model"])
    p += [predict(full, q) for q in Q[len(fit_ids):]]
    base = [q.labels(pq, oc["alpha"], oc["flip_penalty"]) for q, pq in zip(Q, p)]
    G = [Gaps(q, gc) for q in Q]
    fit_g = G[:len(fit_ids)]
    print("long-gap frames: {:.3f} of graded, GT D inside {:.3f}; rows (fit): dead {}, owner {}"
          .format(sum(g.gap.sum() for g in G) / sum(q.m.graded.sum() for q in Q),
                  np.mean(np.concatenate([q.m.dead_gt[g.gap] for q, g in zip(Q, G)])),
                  sum(len(g.rows["dead"][1]) for g in fit_g),
                  sum(len(g.rows["owner"][1]) for g in fit_g)))

    probs = [g.probs(fit_gaps([r for r in fit_g if r is not g], params, names)) for g in fit_g]
    models = fit_gaps(fit_g, params, names)
    again = fit_gaps(fit_g, params, names)
    probs += [g.probs(models) for g in G[len(fit_ids):]]

    def labels(i, ld, lo):
        return decode_gaps(base[i], G[i].gap, *probs[i], gc["step_frames"], ld, lo)

    best = None
    for ld, lo in itertools.product(LAM_DEAD, LAM_OWNER):
        s = float(np.mean([Q[i].m.evaluate(labels(i, ld, lo))["spell_f1"]
                           for i in range(len(fit_ids))]))
        if best is None or s > best[0]:
            best = (s, ld, lo)
    score, lam_dead, lam_owner = best
    print("out-of-fold fit spellF1 {:.4f} at lam_dead {} lam_owner {} (grid edges: lam_dead {}-{},"
          " lam_owner {}-{})".format(score, lam_dead, lam_owner, LAM_DEAD[0], LAM_DEAD[-1],
                                     LAM_OWNER[0], LAM_OWNER[-1]))

    for k, path in gapstate.model_paths(cfg).items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        models[k].save_model(path)
        print("{} model -> {} ({} KB); retrained byte-identical: {}".format(
            k, os.path.relpath(path), os.path.getsize(path) // 1024,
            models[k].model_to_string() == again[k].model_to_string()))
    for i in range(len(fit_ids), len(Q)):
        gap, gt = G[i].gap, Q[i].m.gt
        alive = gap & np.isin(gt, ["H", "A"])
        print("held-out {} gap frames: dead accuracy {:.3f} (prior {:.3f}); alive owner accuracy "
              "{:.3f}".format(Q[i].m.game, ((probs[i][0][gap] > 0.5) == (gt[gap] == "D")).mean(),
                              max((gt[gap] == "D").mean(), (gt[gap] != "D").mean()),
                              ((probs[i][1][alive] > 0.5) == (gt[alive] == "H")).mean()))

    before = np.array([q.m.evaluate(b)["spell_f1"] for q, b in zip(Q, base)])
    after = np.array([Q[i].m.evaluate(labels(i, lam_dead, lam_owner))["spell_f1"]
                      for i in range(len(Q))])
    diff = after - before
    wins = int((diff > 0).sum())
    print("\n{:<12} {:>8} {:>8} {:>8}".format("match", "without", "with", "diff"))
    for q, b, a, d in zip(Q, before, after, diff):
        print("{:<12} {:>8.4f} {:>8.4f} {:>+8.4f}{}".format(
            q.m.game, b, a, d, "  (held-out)" if q in Q[len(fit_ids):] else "  (out of fold)"))
    print("paired (PLAN.md §6.4): mean {:+.4f}, wins {}/{}, one-sided sign test p = {:.4f}; "
          "held-out mean {:.4f} -> {:.4f}".format(diff.mean(), wins, len(Q),
                                                  sign_test(wins, len(Q)),
                                                  before[len(fit_ids):].mean(),
                                                  after[len(fit_ids):].mean()))
    print("\nset gaps.lam_dead: {} and gaps.lam_owner: {} in the config".format(lam_dead,
                                                                            lam_owner))


if __name__ == "__main__":
    main()
