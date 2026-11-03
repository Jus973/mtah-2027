"""make train-dead [CFG=configs/rules_v8.yaml]: train the learned dead-ball model on the sim with a
realistic dead-time ball, tune its penalty out of fold, and compare with the same config without
it (PLAN.md §6.4, paired) under six dead-ball conditions. Open data only (§6.1).

1. Each sim match is loaded as the DFL data has it ("frozen": the dead-time ball never moves, which
   no real tracker shows) and with learn/deadstate.realistic_dead_ball variants. Training uses
   deadstate.train_conditions; frozen and the 1 cm / 5 cm noise levels are conditions the model
   never sees.
2. The takeover and gap models are the config's, cross-fitted on the frozen sim as make
   train-takeover and make train-gaps train them.
3. Out of fold: for each fit match, a dead model trained on the other four (under every training
   condition) predicts it; lam is tuned on those predictions, end to end.
4. The final model is trained on all five fit matches (twice, to check it is byte-identical) and
   written to deadstate.model.
5. Under each condition, per-match spellF1 with and without the learned dead ball.
"""

import argparse
import contextlib
import os
from math import comb

import numpy as np

import mtah.eval.simeval as simeval
from mtah import config
from mtah.decode.dead import decode_dead
from mtah.decode.gaps import decode_gaps
from mtah.decode.viterbi import decode_viterbi
from mtah.events.deadball import dead_mask, evidence
from mtah.io.atd import load_game
from mtah.learn import deadstate, gapstate, takeover

LAM = [1.0, 2.5, 5.0, 10.0, 20.0]
# Conditions: the training ones (deadstate.train_conditions, with the config's noise) and three
# the model never sees: the DFL ball as is, and 1 cm / 5 cm of tracker noise.
CONDITIONS = {"jitter": ("jitter", {}), "pickup": ("pickup", {}), "lost": ("lost", {}),
              "frozen": None, "noise 1 cm": ("jitter", {"jitter_m": 0.01}),
              "noise 5 cm": ("jitter", {"jitter_m": 0.05})}


@contextlib.contextmanager
def dead_ball(kind, dcfg, seed):
    """SimMatch loads its atd through simeval.load_game; give it the condition's ball."""
    orig = simeval.load_game
    if kind is not None:
        simeval.load_game = lambda g, data_dir=None: (
            deadstate.realistic_dead_ball(load_game(g)[0], kind, dcfg, seed), None)
    try:
        yield
    finally:
        simeval.load_game = orig


class Match:
    """One sim match under one dead-ball condition: contacts, evidence D, gaps, features."""

    def __init__(self, game, cfg, cond, seed, F):
        kind, over = cond if cond else (None, {})
        with dead_ball(kind, dict(cfg["deadstate"], **over), seed):
            m = self.m = simeval.SimMatch(game, cfg)
        oc, gc = cfg["owner"], cfg["gaps"]
        self.events = takeover.contacts(m.ball, m.d, cfg["touch"], oc["contact_r_m"], m.fps)
        self.X = takeover.design(self.events, m.d, F, gapstate.feature_names(gc),
                                 dict(cfg, owner=dict(oc, shape_features=True)), m.fps)
        self.rows, self.y = takeover.training_rows(self.events, m.gt, m.graded,
                                                   oc["label_lag_frames"])
        ev = evidence(m.ball_raw, m.ball, m.observed, m.players, m.graded, m.atd.pitch,
                      cfg["deadball"], m.fps)
        self.dead = dead_mask(ev, m.touches(cfg["touch"]), m.graded, cfg["deadball"],
                              tuple(cfg["deadball"]["kinds"]))
        self.gap = gapstate.long_gaps(m.observed, m.graded, gc["min_frames"])
        self.FD = deadstate.features(F, deadstate.ball_features(m.ball_raw, m.ball, m.atd.pitch,
                                                                m.fps))
        self.F = F


def sign_p(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rules_v8.yaml")
    cfg = config.load(os.path.abspath(parser.parse_args().config))
    ds, gc, oc = cfg["deadstate"], cfg["gaps"], cfg["owner"]
    ids = cfg["matches"]["fit"] + cfg["matches"]["holdout"]
    nf = len(cfg["matches"]["fit"])
    gnames = gapstate.feature_names(gc)
    names = deadstate.feature_names(gnames)
    train_conds = list(ds["train_conditions"])
    conds = CONDITIONS
    M = {c: [] for c in conds}
    for n, g in enumerate(ids):
        m0 = simeval.SimMatch("DFL-" + g, cfg)
        F = gapstate.features(m0.players, m0.atd.team_of_slot,
                              gapstate.home_direction(m0.atd, m0.spec, {},
                                                      cfg["direction"]["window_frames"]),
                              m0.atd.pitch, gc, m0.fps)
        del m0
        for c, cond in conds.items():
            M[c].append(Match("DFL-" + g, cfg, cond, n, F))

    def fold(i):
        return tuple(j for j in range(nf) if j != i) if i < nf else tuple(range(nf))

    folds = sorted({fold(i) for i in range(len(ids))})

    def fit_dead(key):
        data = [deadstate.training_rows(M[c][j].FD, M[c][j].m.gt, M[c][j].m.graded,
                                        ds["step_frames"]) for c in train_conds for j in key]
        return deadstate.train(np.vstack([x for x, _ in data]),
                               np.concatenate([y for _, y in data]), cfg["dead_model"], names)
    frozen = M["frozen"]
    tk, gm, dm = {}, {}, {}
    for key in folds:
        tk[key] = takeover.train(np.vstack([frozen[j].X[frozen[j].rows] for j in key]),
                                 np.concatenate([frozen[j].y for j in key]), cfg["takeover_model"],
                                 takeover.feature_names(cfg))
        rows = [gapstate.training_rows(frozen[j].F, frozen[j].gap, frozen[j].m.gt,
                                       gc["step_frames"]) for j in key]
        gm[key] = {k: gapstate.train(np.vstack([r[k][0] for r in rows]),
                                     np.concatenate([r[k][1] for r in rows]), cfg["gap_model"],
                                     gnames) for k in ("dead", "owner")}
        dm[key] = fit_dead(key)

    gp = []
    for i in range(len(ids)):
        gr = np.flatnonzero(frozen[i].m.graded)
        pdd, ph = np.full(len(frozen[i].m.gt), np.nan), np.full(len(frozen[i].m.gt), np.nan)
        pdd[gr], ph[gr] = gapstate.predict(gm[fold(i)], frozen[i].F, gr)
        gp.append((pdd, ph))

    def p_learned(x, i):
        gr = np.flatnonzero(x.m.graded)
        out = np.full(len(x.m.gt), np.nan)
        out[gr] = deadstate.predict(dm[fold(i)], x.FD, gr)
        return out

    P = {c: [p_learned(M[c][i], i) for i in range(len(ids))] for c in conds}

    def score(c, i, lam):
        x = M[c][i]
        p = takeover.predict(tk[fold(i)], x.X)
        if lam is None:
            dead, pdead = x.dead, gp[i][0]
        else:
            dead, pdead = decode_dead(P[c][i], x.m.graded, ds["step_frames"], lam), P[c][i]
        base = decode_viterbi(x.events, p, x.m.graded, dead, oc["alpha"], oc["flip_penalty"])
        lab = decode_gaps(base, x.gap, pdead, gp[i][1], gc["step_frames"], gc["lam_dead"],
                          gc["lam_owner"])
        return x.m.evaluate(lab)["spell_f1"]

    lam = max(LAM, key=lambda v: np.mean([score(c, i, v) for c in train_conds
                                           for i in range(nf)]))
    print("lam {} out of fold over the training conditions (grid edges {}-{})".format(
        lam, LAM[0], LAM[-1]))

    full = fold(len(ids) - 1)
    again = fit_dead(full)
    path = deadstate.model_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dm[full].save_model(path)
    print("model -> {} ({} KB); retrained byte-identical: {}".format(
        os.path.relpath(path), os.path.getsize(path) // 1024,
        dm[full].model_to_string() == again.model_to_string()))

    print("\nwith minus without the learned dead ball, per match ({}; last {} held-out):".format(
        ", ".join(ids), len(ids) - nf))
    for c in conds:
        a = np.array([score(c, i, None) for i in range(len(ids))])
        b = np.array([score(c, i, lam) for i in range(len(ids))])
        d = b - a
        w = int((d > 0).sum())
        gr = [M[c][i].m.graded for i in range(nf, len(ids))]
        acc = np.mean([((P[c][i][g] > 0.5) == (M[c][i].m.gt[g] == "D")).mean()
                       for i, g in zip(range(nf, len(ids)), gr)])
        print("{:<11} {} | mean {:+.4f}, wins {}/{}, p = {:.4f} | held-out {:.4f} -> {:.4f} | "
              "frame accuracy {:.3f}{}".format(
                  c, " ".join("{:+.4f}".format(v) for v in d), d.mean(), w, len(d),
                  sign_p(w, len(d)), a[nf:].mean(), b[nf:].mean(), acc,
                  "  (trained on)" if c in train_conds else "  (never seen)"))
    print("\nset deadstate.lam: {} in the config".format(lam))


if __name__ == "__main__":
    main()
