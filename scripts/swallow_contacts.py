"""Board #36 follow-up (claude-fable-c): for every swallowed short GT spell on the open-data sims
(a GT possession of team T whose frames the frozen v11 base labels >= 80% as the other team),
(1) does a detected contact by T start inside the spell +-0.5 s, and (2) what did the
cross-fitted takeover model say at that contact? Separates "decode economics price out the
switch" (contact present, p high) from "discrimination" (contact present, p low) from "contact
recall" (no contact). Open data only; same base as scripts/spell_errors.py.
"""

import argparse
import os

import numpy as np

from mtah import config
from mtah.decode.gaps import decode_gaps
from mtah.eval.score import pm
from mtah.learn import gapstate, takeover
from paired import check_sections, deadhybrid_parts, fold_of
from train_gaps import Gaps, fit_gaps
from train_takeover import Prepared

PAD_S = 0.5
MAX_S = 3.0


def base_with_probs(cfg):
    oc, gc, tm = cfg["owner"], cfg["gaps"], cfg["takeover_model"]
    fit, held = cfg["matches"]["fit"], cfg["matches"]["holdout"]
    prefix = cfg["matches"].get("prefix", "DFL-")
    Q = [Prepared(prefix + g, cfg) for g in fit + held]
    nf = len(fit)
    names = takeover.feature_names(cfg)
    dh = cfg.get("deadhybrid") or {}
    dead = deadhybrid_parts(cfg, Q, nf) if dh.get("enabled") else [None] * len(Q)
    G = [Gaps(q, gc) for q in Q] if gc["enabled"] else None
    if G:
        gn = gapstate.feature_names(gc)
        probs = [G[i].probs(fit_gaps([G[j] for j in range(nf) if j != i], cfg["gap_model"], gn))
                 for i in range(nf)]
        gm = fit_gaps(G[:nf], cfg["gap_model"], gn)
        probs += [g.probs(gm) for g in G[nf:]]
    out, P = [], []
    for i, q in enumerate(Q):
        key = fold_of(i, nf)
        b = takeover.train(np.concatenate([Q[j].X[Q[j].rows] for j in key]),
                           np.concatenate([Q[j].y for j in key]), tm, names)
        p = takeover.predict(b, q.X)
        lab = q.labels(p, oc["alpha"], oc["flip_penalty"], dead=dead[i])
        if G:
            lab = decode_gaps(lab, G[i].gap, *probs[i], gc["step_frames"], gc["lam_dead"],
                              gc["lam_owner"])
        out.append(np.asarray(lab))
        P.append(np.asarray(p))
    return Q, nf, out, P


def analyse(q, lab, p, fps):
    gt = np.asarray(q.m.gt)
    codes = pm.label_codes(gt)
    runs = pm.valid_runs(codes)
    pad = int(round(PAD_S * fps))
    starts = np.array([e.start for e in q.events])
    teams = np.array([e.team for e in q.events])
    rows = []
    for a, b, k in pm.possession_spells(codes, runs):
        if (b - a) / fps >= MAX_S:
            continue
        team = "HA"[k]
        other = "AH"[k]
        if np.mean(lab[a:b] == other) < 0.8:
            continue
        inside = (starts >= a - pad) & (starts < b + pad)
        own = inside & (teams == k)
        ret = (starts >= b - pad) & (starts < b + 4 * pad) & (teams == 1 - k)   # the return
        rows.append({
            "len_s": (b - a) / fps,
            "contact": bool(own.any()),
            "p_max": float(p[own].max()) if own.any() else np.nan,
            "n_own": int(own.sum()),
            "any_contact": bool(inside.any()),
            "return_contact": bool(ret.any()),
            "p_ret_max": float(p[ret].max()) if ret.any() else np.nan,
        })
    return rows


def summarise(rows, title):
    n = len(rows)
    c = np.array([r["contact"] for r in rows])
    anyc = np.array([r["any_contact"] for r in rows])
    pm_ = np.array([r["p_max"] for r in rows])
    rc = np.array([r["return_contact"] for r in rows])
    pr = np.array([r["p_ret_max"] for r in rows])
    print("{}: swallowed <{:.0f}s GT spells n={}".format(title, MAX_S, n))
    if not n:
        return
    print("  own-team contact inside +-{:.1f}s: {:.1%}   any contact inside: {:.1%}".format(
        PAD_S, c.mean(), anyc.mean()))
    for lo, hi in ((0.5, 1.01), (0.3, 0.5), (0.1, 0.3), (0.0, 0.1)):
        sel = c & (pm_ >= lo) & (pm_ < hi)
        print("    ... with max p(takeover) in [{:.1f},{:.1f}): {:.1%} of all swallowed".format(
            lo, min(hi, 1.0), sel.mean()))
    print("  return contact (other team, [end-{:.1f}s, end+{:.1f}s)): {:.1%}; both present: "
          "{:.1%}; both present with both p>=0.5: {:.1%}".format(
              PAD_S, 4 * PAD_S, rc.mean(), (c & rc).mean(),
              (c & rc & (pm_ >= 0.5) & (pr >= 0.5)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    args = ap.parse_args()
    cfg = config.load(os.path.abspath(args.config))
    check_sections(cfg, args.config)
    Q, nf, base, P = base_with_probs(cfg)
    allrows = []
    for i, (q, lab, p) in enumerate(zip(Q, base, P)):
        rows = analyse(q, lab, p, q.m.fps)
        allrows += rows
        c = np.mean([r["contact"] for r in rows]) if rows else float("nan")
        hi = np.mean([r["contact"] and r["p_max"] >= 0.5 for r in rows]) if rows else float("nan")
        print("{:12s} {} n={:4d} contact {:.1%} contact&p>=.5 {:.1%}".format(
            q.m.game, "held" if i >= nf else "fit ", len(rows), c, hi))
    summarise(allrows, "\nALL 7")
    summarise([r for r in allrows if r["len_s"] < 1], "\n<1 s")
    summarise([r for r in allrows if 1 <= r["len_s"] < 3], "\n1-3 s")


if __name__ == "__main__":
    main()
