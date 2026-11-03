"""Board #49 (idea #30, variant A pre-check): a LightGBM step-level owner model vs v11's owner.

The step model is the gap owner model's recipe (learn/gapstate.py features, `gap_model` params,
both perspectives, every `gaps.step_frames`-th frame) trained on ALL GT-alive graded frames, not
only long-gap frames. Readout is per-frame owner accuracy, no temporal decoding, on frames that are
GT-alive and called alive by v11, split into: in a long gap / new-or-old-team contact within
+-1 s / other. DFL: cross-fitted exactly as scripts/paired.py. PFF (board #48 sims): the model fit
on the 5 DFL fit matches against the committed v11 pipeline; nothing is fitted on PFF.
Pre-registered kill: step model < v11 + 2 points on non-contact alive frames (DFL or PFF mean).

  python scripts/step_owner_precheck.py --dfl                # cross-fit table + saves fit-5 model
  python scripts/step_owner_precheck.py --pff --shard k/n    # per-match cache under MTAH_PFF_DIR
  python scripts/step_owner_precheck.py --pff --summary
"""

import argparse
import json
import os

import numpy as np

from mtah import config, registry
from mtah.decode.gaps import decode_gaps
from mtah.learn import gapstate, takeover
from paired import check_sections, deadhybrid_parts, fold_of
from train_gaps import Gaps, fit_gaps
from train_takeover import Prepared

CFG = os.path.join(registry.ROOT, "configs", "final.yaml")
PFF_DIR = os.path.expanduser(os.environ.get("MTAH_PFF_DIR", "~/opendata/pff"))
OUT = os.path.join(PFF_DIR, "eval_step_owner")
CELLS = ("gap", "contact", "other", "noncontact", "all")


def step_rows(g, step):
    gt = np.asarray(g.q.m.gt)
    idx = np.flatnonzero(g.q.m.graded & np.isin(gt, ("H", "A")))[::step]
    return (np.vstack([g.F[0][idx], g.F[1][idx]]),
            np.concatenate([gt[idx] == "H", gt[idx] == "A"]).astype(float))


def fit_step(gs, cfg):
    X, y = zip(*[step_rows(g, cfg["gaps"]["step_frames"]) for g in gs])
    return gapstate.train(np.vstack(X), np.concatenate(y), cfg["gap_model"],
                          gapstate.feature_names(cfg["gaps"]))


def p_home(model, g, idx):
    a, b = g.F[0][idx], g.F[1][idx]
    return (model.predict(a, num_threads=1) + 1 - model.predict(b, num_threads=1)) / 2


def cell_acc(g, lab, model):
    """{cell: (n, v11 correct, step correct)} over GT-alive frames that v11 calls alive."""
    m = g.q.m
    gt, lab = np.asarray(m.gt), np.asarray(lab)
    base = m.graded & np.isin(gt, ("H", "A")) & np.isin(lab, ("H", "A"))
    near = np.zeros(len(gt), bool)
    w = int(m.fps)
    for e in g.q.events:
        near[max(e.start - w, 0):e.start + w + 1] = True
    idx = np.flatnonzero(base)
    step = np.where(p_home(model, g, idx) >= 0.5, "H", "A")
    ok_b, ok_s = lab[idx] == gt[idx], step == gt[idx]
    gap, nr = g.gap[idx], near[idx]
    masks = {"gap": gap, "contact": nr & ~gap, "other": ~nr & ~gap, "noncontact": ~nr,
             "all": np.ones(len(idx), bool)}
    return {k: (int(v.sum()), int(ok_b[v].sum()), int(ok_s[v].sum())) for k, v in masks.items()}


def show(rows):
    print("{:<12}".format("match") + "".join("{:>22}".format(c + " v11/step") for c in CELLS))
    d = {c: [] for c in CELLS}
    for name, r in rows:
        cells = []
        for c in CELLS:
            n, b, s = r[c]
            cells.append("{:>8.3f}/{:.3f} n{:>5}".format(b / max(n, 1), s / max(n, 1), n))
            if n:
                d[c].append((s - b) / n)
        print("{:<12}".format(name) + "".join("{:>22}".format(x) for x in cells))
    for c in CELLS:
        v = np.array(d[c])
        print("{:<11} step - v11: mean {:+.4f} median {:+.4f}, step better in {}/{}".format(
            c, v.mean(), np.median(v), int((v > 0).sum()), len(v)))


def dfl(cfg):
    check_sections(cfg, CFG)
    oc, gc, tm = cfg["owner"], cfg["gaps"], cfg["takeover_model"]
    fit, held = cfg["matches"]["fit"], cfg["matches"]["holdout"]
    Q = [Prepared("DFL-" + g, cfg) for g in fit + held]
    nf = len(fit)
    names = takeover.feature_names(cfg)
    dh = cfg.get("deadhybrid") or {}
    dead = deadhybrid_parts(cfg, Q, nf) if dh.get("enabled") else [None] * len(Q)
    G = [Gaps(q, gc) for q in Q]
    gn = gapstate.feature_names(gc)
    rows = []
    for i, q in enumerate(Q):
        key = fold_of(i, nf)
        b = takeover.train(np.concatenate([Q[j].X[Q[j].rows] for j in key]),
                           np.concatenate([Q[j].y for j in key]), tm, names)
        lab = q.labels(takeover.predict(b, q.X), oc["alpha"], oc["flip_penalty"], dead=dead[i])
        gm = fit_gaps([G[j] for j in key], cfg["gap_model"], gn)
        lab = decode_gaps(lab, G[i].gap, *G[i].probs(gm), gc["step_frames"], gc["lam_dead"],
                          gc["lam_owner"])
        rows.append((q.m.game + ("*" if i >= nf else ""), cell_acc(G[i], lab,
                                                               fit_step([G[j] for j in key], cfg))))
        print("done", q.m.game, flush=True)
    show(rows)
    os.makedirs(OUT, exist_ok=True)
    fit_step(G[:nf], cfg).save_model(os.path.join(OUT, "step_owner_fit5.txt"))


def pff(cfg, shard, summary):
    import lightgbm as lgb
    from regret_matrix import V11, committed_labels
    games = sorted(g for g in os.listdir(registry.SIM_DIR) if g.startswith("PFF-"))
    k, n = (int(x) for x in shard.split("/"))
    if not summary:
        model = lgb.Booster(model_file=os.path.join(OUT, "step_owner_fit5.txt"))
        for g in games[k::n]:
            path = os.path.join(OUT, g + ".json")
            if os.path.isfile(path):
                continue
            try:
                q = Prepared(g, cfg)
                r = cell_acc(Gaps(q, cfg["gaps"]), committed_labels(g, V11), model)
            except ValueError as e:
                print("FAILED", g, e, flush=True)
                continue
            with open(path, "w") as fh:
                json.dump(r, fh)
            print("done", g, flush=True)
    rows = []
    for g in games:
        path = os.path.join(OUT, g + ".json")
        if os.path.isfile(path):
            with open(path) as fh:
                rows.append((g, json.load(fh)))
    show(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", action="store_true")
    ap.add_argument("--pff", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    cfg = config.load(CFG)
    if args.dfl:
        dfl(cfg)
    if args.pff:
        pff(cfg, args.shard, args.summary)


if __name__ == "__main__":
    main()
