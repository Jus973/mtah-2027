"""Board #26: fixed gap-wide p_dead logit offset for long gaps whose ball onset was fast (>= 8 m/s,
last observed ball inside the lines). Open data only. Base = v11 built as in scripts/paired.py
(cross-fitted on the DFL sims, full-fit models on MET-G1N/G2N native-ball Metrica).

Arms: base; fixed (pre-registered offset -1.0 everywhere); lomo (MET only: the offset derived on the
other MET match from GT alive shares, logit(alive_fast) - logit(alive_slow), applied with minus
sign). DFL is a non-regression check only (onset speed is circular there: the degrader places gaps
by onset speed). Never consults real-match data.
"""

import argparse
import os

import numpy as np

from mtah import config
from mtah.decode.gaps import decode_gaps, runs
from mtah.learn import gapstate, takeover
from oracle_receiver import sign_p
from paired import check_sections, deadhybrid_parts, fold_of
from train_gaps import Gaps, fit_gaps
from train_takeover import Prepared

FAST, SLOW, LOOK, LAST = 8.0, 3.0, 12, 6
MET = ("MET-G1N", "MET-G2N")


def onset_bins(ball, observed, gap, pitch, fps):
    """Per long-gap run (start, length, bin): bin in {"oob", "slow", "mid", "fast", "none"}."""
    out = []
    size = np.asarray(pitch[:2], float)
    for a, n in zip(*runs(gap)):
        lo = max(a - LOOK, 0)
        idx = np.flatnonzero(observed[lo:a]) + lo
        if len(idx) < 2:
            out.append((a, n, "none"))
            continue
        if np.any((ball[idx[-1], :2] < 0) | (ball[idx[-1], :2] > size)):
            out.append((a, n, "oob"))
            continue
        idx = idx[-LAST:]
        step = np.diff(idx)
        v = np.linalg.norm(np.diff(ball[idx, :2], axis=0), axis=1) / step * fps
        s = float(np.median(v))
        out.append((a, n, "fast" if s >= FAST else "slow" if s < SLOW else "mid"))
    return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def shifted(p_dead, bins, offset):
    out = p_dead.copy()
    for a, n, b in bins:
        if b == "fast":
            out[a:a + n] = 1 / (1 + np.exp(-(logit(p_dead[a:a + n]) + offset)))
    return out


def alive_share(gt, graded, bins, which):
    m = np.zeros(len(gt), bool)
    for a, n, b in bins:
        if b == which:
            m[a:a + n] = True
    m &= graded
    return np.isin(gt[m], ("H", "A")).mean() if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--offset", type=float, default=-1.0)
    args = ap.parse_args()
    cfg = config.load(os.path.abspath(args.config))
    check_sections(cfg, args.config)
    oc, gc, tm = cfg["owner"], cfg["gaps"], cfg["takeover_model"]
    fit, held = cfg["matches"]["fit"], cfg["matches"]["holdout"]
    prefix = cfg["matches"].get("prefix", "DFL-")
    Q = [Prepared(prefix + g, cfg) for g in fit + held] + [Prepared(g, cfg) for g in MET]
    nf, nd = len(fit), len(fit) + len(held)
    names = takeover.feature_names(cfg)
    dh = cfg.get("deadhybrid") or {}
    dead = deadhybrid_parts(cfg, Q, nf) if dh.get("enabled") else [None] * len(Q)
    G = [Gaps(q, gc) for q in Q]
    gn = gapstate.feature_names(gc)
    probs = [G[i].probs(fit_gaps([G[j] for j in range(nf) if j != i], cfg["gap_model"], gn))
             for i in range(nf)]
    gm = fit_gaps(G[:nf], cfg["gap_model"], gn)
    probs += [g.probs(gm) for g in G[nf:]]
    owner = []
    for i, q in enumerate(Q):
        key = fold_of(i, nf)
        b = takeover.train(np.concatenate([Q[j].X[Q[j].rows] for j in key]),
                           np.concatenate([Q[j].y for j in key]), tm, names)
        owner.append(q.labels(takeover.predict(b, q.X), oc["alpha"], oc["flip_penalty"],
                              dead=dead[i]))
    B = [onset_bins(q.m.ball, q.m.observed, G[i].gap, q.m.atd.pitch, q.m.fps)
         for i, q in enumerate(Q)]

    def run(i, offset):
        pd, ph = probs[i]
        lab = decode_gaps(owner[i], G[i].gap, shifted(pd, B[i], offset), ph, gc["step_frames"],
                          gc["lam_dead"], gc["lam_owner"])
        return np.asarray(lab)

    gt = [np.asarray(q.m.gt) for q in Q]
    shares = {q.m.game: (alive_share(gt[i], q.m.graded, B[i], "fast"),
                         alive_share(gt[i], q.m.graded, B[i], "slow")) for i, q in enumerate(Q)}
    print("onset bins (gap count fast/mid/slow/oob/none) and GT alive share fast | slow:")
    for i, q in enumerate(Q):
        c = [sum(b == k for _, _, b in B[i]) for k in ("fast", "mid", "slow", "oob", "none")]
        print("  {:<12} {}  alive fast {:.3f} | slow {:.3f}".format(q.m.game, c, *shares[q.m.game]))
    base, fixed, lomo = [], [], {}
    for i, q in enumerate(Q):
        lb, lf = run(i, 0.0), run(i, args.offset)
        base.append(q.m.evaluate(lb)["spell_f1"])
        fixed.append(q.m.evaluate(lf)["spell_f1"])
        if i >= nd:
            other = MET[1 - (i - nd)]
            af, as_ = shares[other]
            off = -(logit(af) - logit(as_))
            ll = run(i, off)
            lomo[q.m.game] = (off, q.m.evaluate(ll)["spell_f1"])
            g = q.m.graded
            fl = (lb == "D") & (ll != "D") & g
            good = fl & np.isin(gt[i], ("H", "A"))
            per = sorted((int(good[a:a + n].sum()) for a, n, b in B[i] if b == "fast"),
                         reverse=True)
            print("  {} lomo: D->alive flips {} (GT alive {}), top-3 fast gaps' share of recovered "
                  "{:.2f}".format(q.m.game, int(fl.sum()), int(good.sum()),
                                  sum(per[:3]) / max(sum(per), 1)))
    base, fixed = np.array(base), np.array(fixed)
    print("{:<12}{:>10}{:>10}{:>16}".format("match", "base", "fixed", "lomo (offset)"))
    for i, q in enumerate(Q):
        lo = lomo.get(q.m.game)
        print("{:<12}{:>10.4f}{:>10.4f}{}".format(
            q.m.game, base[i], fixed[i],
            "{:>10.4f} ({:+.2f})".format(lo[1], lo[0]) if lo else ""))
    d = fixed[:nd] - base[:nd]
    w, p = sign_p(d)
    print("DFL fixed vs base: mean {:+.4f}, wins {}/{}, p = {:.4f}, min {:+.4f}".format(
        d.mean(), w, nd, p, d.min()))
    for k, (off, f) in lomo.items():
        i = nd + MET.index(k)
        print("{} lomo {:+.4f} (offset {:+.2f}); fixed {:+.4f}".format(k, f - base[i], off,
                                                                     fixed[i] - base[i]))


if __name__ == "__main__":
    main()
