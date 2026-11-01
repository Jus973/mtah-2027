"""make paired A=configs/final.yaml B=configs/rules_v6.yaml: the PLAN.md §6.4 paired per-match
comparison of two viterbi configs on the sim. Open data only (degraded DFL/IDSSE sim, §6.1).

Each config's learned parts are cross-fitted: on each fit match, the takeover model (and, with
gaps.enabled, the gap models) are trained on the other four fit matches; the held-out matches
use models trained on all five. Every match is then scored end to end with the official scorer,
both configs on the same matches and GT. B replaces A only if the mean paired difference is
positive and the one-sided sign test gives p <= 0.05 (with 7 matches: 7 wins).
"""

import argparse
import os
from math import comb

import numpy as np

from mtah import config
from mtah.decode.gaps import decode_gaps
from mtah.learn import gapstate
from train_gaps import Gaps, fit_gaps
from train_takeover import Prepared, fit_on, predict


def scores(path):
    """(match ids, per-match spellF1) for one config, every learned part cross-fitted."""
    cfg = config.load(os.path.abspath(path))
    oc, gc = cfg["owner"], cfg.get("gaps")
    if oc["decoder"] != "viterbi":
        raise SystemExit("{}: paired.py compares viterbi configs".format(path))
    fit, held = cfg["matches"]["fit"], cfg["matches"]["holdout"]
    Q = [Prepared("DFL-" + g, cfg) for g in fit + held]
    nf = len(fit)
    p = [predict(fit_on([Q[j] for j in range(nf) if j != i], cfg["takeover_model"]), Q[i])
         for i in range(nf)]
    full = fit_on(Q[:nf], cfg["takeover_model"])
    p += [predict(full, q) for q in Q[nf:]]
    labels = [q.labels(pq, oc["alpha"], oc["flip_penalty"]) for q, pq in zip(Q, p)]
    if gc and gc["enabled"]:
        G = [Gaps(q, gc) for q in Q]
        names = gapstate.feature_names(gc)
        probs = [G[i].probs(fit_gaps([G[j] for j in range(nf) if j != i], cfg["gap_model"],
                                     names)) for i in range(nf)]
        models = fit_gaps(G[:nf], cfg["gap_model"], names)
        probs += [g.probs(models) for g in G[nf:]]
        labels = [decode_gaps(lab, g.gap, *pr, gc["step_frames"], gc["lam_dead"],
                              gc["lam_owner"]) for lab, g, pr in zip(labels, G, probs)]
    out = np.array([q.m.evaluate(lab)["spell_f1"] for q, lab in zip(Q, labels)])
    return [q.m.game for q in Q], nf, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="configs/final.yaml")
    parser.add_argument("--b", required=True)
    args = parser.parse_args()
    games, nf, a = scores(args.a)
    _, _, b = scores(args.b)
    d = b - a
    wins = int((d > 0).sum())
    p = sum(comb(len(d), k) for k in range(wins, len(d) + 1)) / 2 ** len(d)
    print("{:<12} {:>8} {:>8} {:>8}".format("match", "A", "B", "B - A"))
    for i, g in enumerate(games):
        print("{:<12} {:>8.4f} {:>8.4f} {:>+8.4f}  ({})".format(
            g, a[i], b[i], d[i], "out of fold" if i < nf else "held-out"))
    print("A = {}, B = {}".format(args.a, args.b))
    print("paired (PLAN.md §6.4): mean {:+.4f}, wins {}/{}, one-sided sign test p = {:.4f}; "
          "held-out mean {:.4f} -> {:.4f}".format(d.mean(), wins, len(d), p, a[nf:].mean(),
                                                  b[nf:].mean()))
    print("verdict: {}".format("B replaces A" if d.mean() > 0 and p <= 0.05 else
                               "tie: keep A (or apply the tie-breaks)"))


if __name__ == "__main__":
    main()
