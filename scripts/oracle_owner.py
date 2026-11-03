"""Board #34: where is v11's owner-on-alive headroom? Oracle decomposition on the degraded DFL sim.

Open data only. Same cross-fitting protocol as scripts/oracle_receiver.py (scripts/paired.py): the
config's takeover model is cross-fitted on the fit matches, every other learned part is shared.
Arms (no model changes; each replaces labels of the base output on a frame mask, then re-scores):

  base           the config as is
  owner_alive    true owner on graded frames where GT and base are both alive (owner errors only)
  owner_outside  the same, restricted to frames outside long ball gaps
  owner_inside   the same, restricted to frames inside long ball gaps
  alive_all      true owner on every GT-alive graded frame (also undoes base's false D)
  perfect        no takeover model: the training-label rule on every contact (oracle_receiver's
                 perfect arm, the ceiling of any contact feature with this decoder)

Plus a table of GT open-play turnovers by (contact by the new owner's team within +-1 s?) x
(inside a long gap?), with base recall (a base change to the same team within +-1 s). Never
consults real-match data.
"""

import argparse
import os

import numpy as np

from mtah import config
from mtah.decode.gaps import decode_gaps
from mtah.learn import gapstate, takeover
from oracle_receiver import oracle_label, sign_p
from paired import check_sections, deadhybrid_parts, fold_of
from train_gaps import Gaps, fit_gaps
from train_takeover import Prepared


def with_truth(lab, gt, mask):
    out = lab.copy()
    out[mask] = gt[mask]
    return out


def turnovers(gt, graded):
    """Frames t where the GT owner switches H<->A between alive graded frames t-1 and t."""
    alive = graded & np.isin(gt, ("H", "A"))
    t = np.flatnonzero(alive[1:] & alive[:-1] & (gt[1:] != gt[:-1])) + 1
    return t


def cell_table(q, lab, gap, fps):
    """One row per GT open-play turnover: (new-team contact within +-1 s, inside a long gap, hit).
    Hits match GT turnovers one-to-one to base switches into the same team within +-1 s (nearest
    unused switch, GT turnovers in time order), so one base switch never recalls two turnovers."""
    gt, graded = np.asarray(q.m.gt), q.m.graded
    w = int(fps)
    starts = np.array([e.start for e in q.events])
    teams = np.array([e.team for e in q.events])
    switches = {k: np.flatnonzero((lab[1:] == k) & (lab[:-1] != k)) + 1 for k in ("H", "A")}
    used = {k: np.zeros(len(v), bool) for k, v in switches.items()}
    rows = []
    for t in turnovers(gt, graded):
        k = "HA".index(gt[t])
        near = np.abs(starts - t) <= w
        contact = bool(np.any(near & (teams == k)))
        s, u = switches[gt[t]], used[gt[t]]
        cand = np.flatnonzero((np.abs(s - t) <= w) & ~u)
        hit = bool(len(cand))
        if hit:
            u[cand[np.argmin(np.abs(s[cand] - t))]] = True
        rows.append((contact, bool(gap[t]), hit))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    args = ap.parse_args()
    cfg = config.load(os.path.abspath(args.config))
    check_sections(cfg, args.config)
    if (cfg.get("deadstate") or {}).get("enabled") or (cfg.get("postprocess") or {}).get("enabled"):
        raise SystemExit("{}: deadstate / postprocess configs are not supported".format(args.config))
    oc, gc, tm = cfg["owner"], cfg["gaps"], cfg["takeover_model"]
    fit, held = cfg["matches"]["fit"], cfg["matches"]["holdout"]
    prefix = cfg["matches"].get("prefix", "DFL-")
    Q = [Prepared(prefix + g, cfg) for g in fit + held]
    nf = len(fit)
    lag = oc["label_lag_frames"]
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

    def decode(p_list):
        out = []
        for i, (q, p) in enumerate(zip(Q, p_list)):
            lab = q.labels(p, oc["alpha"], oc["flip_penalty"], dead=dead[i])
            if G:
                lab = decode_gaps(lab, G[i].gap, *probs[i], gc["step_frames"], gc["lam_dead"],
                                  gc["lam_owner"])
            out.append(np.asarray(lab))
        return out

    def f1(i, lab):
        return Q[i].m.evaluate(lab)["spell_f1"]

    p_base = []
    for i in range(len(Q)):
        key = fold_of(i, nf)
        b = takeover.train(np.concatenate([Q[j].X[Q[j].rows] for j in key]),
                           np.concatenate([Q[j].y for j in key]), tm, names)
        p_base.append(takeover.predict(b, Q[i].X))
    base = decode(p_base)
    policy = oc.get("label_policy", "legacy")
    perfect = decode([np.clip(oracle_label(q.events, q.m.gt, q.m.graded, lag, policy), 1e-3,
                              1 - 1e-3) for q in Q])

    arms = {k: [] for k in ("base", "owner_alive", "owner_outside", "owner_inside", "alive_all",
                            "perfect")}
    cells = []
    for i, q in enumerate(Q):
        gt, graded, lab = np.asarray(q.m.gt), q.m.graded, base[i]
        gap = G[i].gap if G else np.zeros(len(gt), bool)
        ga, la = graded & np.isin(gt, ("H", "A")), np.isin(lab, ("H", "A"))
        arms["base"].append(f1(i, lab))
        arms["owner_alive"].append(f1(i, with_truth(lab, gt, ga & la)))
        arms["owner_outside"].append(f1(i, with_truth(lab, gt, ga & la & ~gap)))
        arms["owner_inside"].append(f1(i, with_truth(lab, gt, ga & la & gap)))
        arms["alive_all"].append(f1(i, with_truth(lab, gt, ga)))
        arms["perfect"].append(f1(i, perfect[i]))
        cells += cell_table(q, lab, gap, q.m.fps)
    arms = {k: np.array(v) for k, v in arms.items()}

    print("{:<12}".format("match") + "".join("{:>14}".format(k) for k in arms))
    for i, q in enumerate(Q):
        print("{:<12}".format(q.m.game) + "".join("{:>14.4f}".format(v[i]) for v in arms.values())
              + ("  held-out" if i >= nf else ""))
    b = arms["base"]
    for k in list(arms)[1:]:
        d = arms[k] - b
        w, p = sign_p(d)
        print("{:<14} vs base: mean {:+.4f} (7-match), wins {}/{}; held-out {:+.4f}".format(
            k, d.mean(), w, len(d), arms[k][nf:].mean() - b[nf:].mean()))
    print("owner_alive - perfect: mean {:+.4f}".format((arms["owner_alive"] - arms["perfect"]).mean()))
    c = np.array(cells, dtype=bool)
    print("GT open-play turnovers (all 7 matches): {}".format(len(c)))
    for contact in (True, False):
        for ingap in (False, True):
            m = (c[:, 0] == contact) & (c[:, 1] == ingap)
            print("  contact by new team +-1s={!s:<5} in_gap={!s:<5} n={:>5} ({:5.1%})  base recall "
                  "{:5.1%}".format(contact, ingap, m.sum(), m.mean(), c[m, 2].mean() if m.any()
                                   else float("nan")))


if __name__ == "__main__":
    main()
