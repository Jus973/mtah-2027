"""Board #36 step 0: are v11's spell errors false spells (FP) or missed spells (FN)?

Open data only. Same cross-fitted base as scripts/oracle_owner.py (base arm). For every match,
the official scorer's greedy one-to-one spell matching is run at each IoU of SPELL_IOUS; unmatched
predicted spells are FP, unmatched GT spells are FN, binned by spell length. A FN GT spell is
"swallowed" when >= 80% of its frames carry the other team's label in the base output; a FP
predicted spell is "flicker" when >= 80% of its frames are the other team in GT. Pre-registered
kill rule (#36 critique 2/2): kill MBR's trim action if FN > 2 x FP over the < 3 s bins.
Never consults real-match data.
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

BINS = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf)


def base_labels(cfg):
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
    out = []
    for i, q in enumerate(Q):
        key = fold_of(i, nf)
        b = takeover.train(np.concatenate([Q[j].X[Q[j].rows] for j in key]),
                           np.concatenate([Q[j].y for j in key]), tm, names)
        lab = q.labels(takeover.predict(b, q.X), oc["alpha"], oc["flip_penalty"], dead=dead[i])
        if G:
            lab = decode_gaps(lab, G[i].gap, *probs[i], gc["step_frames"], gc["lam_dead"],
                              gc["lam_owner"])
        out.append(np.asarray(lab))
    return Q, nf, out


def spell_errors(gt, lab, fps):
    gc, pc = pm.label_codes(gt), pm.label_codes(lab)
    runs = pm.valid_runs(gc)
    pc = pm.fill_abstentions(pc, runs)
    gs, ps = pm.possession_spells(gc, runs), pm.possession_spells(pc, runs)
    ov = pm.spell_overlaps(gs, ps)
    nb = len(BINS) - 1
    res = {k: np.zeros(nb) for k in ("gt", "pred", "fn", "fp", "swallowed", "flicker")}
    glen = np.array([(b - a) / fps for a, b, _ in gs])
    plen = np.array([(b - a) / fps for a, b, _ in ps])
    gbin = np.digitize(glen, BINS) - 1
    pbin = np.digitize(plen, BINS) - 1
    np.add.at(res["gt"], gbin, 1)
    np.add.at(res["pred"], pbin, 1)
    for tau in pm.SPELL_IOUS:
        gflag = np.zeros(len(gs), bool)
        pflag = np.zeros(len(ps), bool)
        for iou, gi, pj in ov:
            if iou < tau:
                break
            if gflag[gi] or pflag[pj]:
                continue
            gflag[gi] = pflag[pj] = True
        for gi in np.flatnonzero(~gflag):
            a, b, k = gs[gi]
            res["fn"][gbin[gi]] += 1 / len(pm.SPELL_IOUS)
            if np.mean(lab[a:b] == OTHER_CODE(k, gt)) >= 0.8:
                res["swallowed"][gbin[gi]] += 1 / len(pm.SPELL_IOUS)
        for pj in np.flatnonzero(~pflag):
            a, b, k = ps[pj]
            res["fp"][pbin[pj]] += 1 / len(pm.SPELL_IOUS)
            if np.mean(gt[a:b] == OTHER_CODE(k, gt)) >= 0.8:
                res["flicker"][pbin[pj]] += 1 / len(pm.SPELL_IOUS)
    return res


def OTHER_CODE(code, _):
    return "A" if code == pm.H else "H"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    args = ap.parse_args()
    cfg = config.load(os.path.abspath(args.config))
    check_sections(cfg, args.config)
    Q, nf, base = base_labels(cfg)
    tot = None
    for i, (q, lab) in enumerate(zip(Q, base)):
        r = spell_errors(np.asarray(q.m.gt), lab, q.m.fps)
        tot = r if tot is None else {k: tot[k] + r[k] for k in tot}
        s3 = slice(0, 3)
        print("{:12s} {} FN<3s {:6.1f}  FP<3s {:6.1f}  ratio {:.2f}".format(
            q.m.game, "held" if i >= nf else "fit ", r["fn"][s3].sum(), r["fp"][s3].sum(),
            r["fn"][s3].sum() / max(r["fp"][s3].sum(), 1e-9)))
    hdr = ["{:.0f}-{:.0f}s".format(BINS[j], BINS[j + 1]) for j in range(len(BINS) - 1)]
    print("\nbin        " + " ".join("{:>8s}".format(h) for h in hdr))
    for k in ("gt", "pred", "fn", "swallowed", "fp", "flicker"):
        print("{:10s} ".format(k) + " ".join("{:8.1f}".format(v) for v in tot[k]))
    fn3, fp3 = tot["fn"][:3].sum(), tot["fp"][:3].sum()
    print("\n<3 s: FN {:.1f}  FP {:.1f}  FN/FP {:.2f}  -> {}".format(
        fn3, fp3, fn3 / fp3, "KILL (FN > 2xFP)" if fn3 > 2 * fp3 else "pass"))


if __name__ == "__main__":
    main()
