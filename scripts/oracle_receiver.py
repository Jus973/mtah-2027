"""Board #10 (improvement plan §3a): oracle ceiling of a pass-receiver feature at contacts.

Open data only (degraded DFL/IDSSE sim). Under one cross-fitting protocol (scripts/paired.py) the
config's takeover model is retrained with one extra ORACLE column per contact, relative to the
toucher's team, and every other learned part (dead hybrid, gap models) is shared by both arms:

  base      the config as is (must reproduce make paired's numbers for it)
  receiver  + oracle_recv: 1 if the GT owner just before the NEXT contact (the ball's true
            receiver after this contact's flight) is the toucher's team, else 0; 0.5 when the
            half ends or only D follows
  flight    + flight_features (tracking only, no GT): the real feature family
  corridor  + corridor_features (tracking only): who stands in the ball's path
  perfect   no model: p = 1 - 1e-3 / 1e-3 by the training-label rule itself on every contact - the
            ceiling of ANY takeover feature with this decoder, contacts and D

Both tracking-only families (flight, corridor) live here, not in the pipeline: on the sim they
did not beat the config (board #10 NO-GO), so nothing ships them. The receiver arm is the go/no-go number: only if it clears ~+0.01 held-out does a real flight-
after-contact feature family get built. Never consults real-match data.
"""

import argparse
import os
from math import comb

import numpy as np

from mtah import config
from mtah.decode.gaps import decode_gaps
from mtah.learn import gapstate, takeover
from paired import check_sections, deadhybrid_parts, fold_of
from train_gaps import Gaps, fit_gaps
from train_takeover import Prepared


FLIGHT_FEATURES = ("fl_log_s", "land_d_own", "land_d_opp", "land_margin", "land_missing",
                   "fl_min_own", "fl_min_opp", "fl_own_nearer", "m10_margin", "m25_margin")
CORRIDOR_FEATURES = ("rc_perp_own", "rc_perp_opp", "rc_margin", "rc_first", "rc_valid")
CAP_M = 10.0
FLIGHT_CAP_S = 4.0


def flight_features(events, d, fps=25):
    """(n_events, len(FLIGHT_FEATURES)): where the ball goes after the contact, from the toucher's
    perspective (offline - reads the future). The flight runs from the contact's end to the frame
    before the next contact (capped at FLIGHT_CAP_S): its length, each team's distance to the ball
    at its end (the landing), their minima over it, the share of it the toucher's team is nearer,
    and the distance margin (opponent - own) 10 and 25 frames after the contact (0 when that
    frame is past the flight). No flight at all (adjacent contacts) is encoded as a missing
    landing."""
    T, n = len(d), len(events)
    cap = int(FLIGHT_CAP_S * fps)
    out = np.zeros((n, len(FLIGHT_FEATURES)))

    def margin(j, t, stop):
        if j > stop or not np.isfinite(d[j]).all():
            return 0.0
        return float(np.clip(d[j, 1 - t] - d[j, t], -CAP_M, CAP_M))

    for i, e in enumerate(events):
        t = e.team
        stop = int(min(events[i + 1].start - 1 if i + 1 < n else T - 1, T - 1))
        j = int(min(stop, e.end + cap))
        if j <= e.end:  # the next contact follows at once: no flight, landing unobserved
            out[i] = [0.0, CAP_M, CAP_M, 0.0, 1.0, CAP_M, CAP_M, 0.5, 0.0, 0.0]
            continue
        seg = d[e.end + 1:j + 1]
        own, opp = seg[:, t], seg[:, 1 - t]
        ok = np.isfinite(own) & np.isfinite(opp)
        land = d[j]
        miss = not np.isfinite(land).all()
        lo = CAP_M if not np.isfinite(land[t]) else min(float(land[t]), CAP_M)
        lp = CAP_M if not np.isfinite(land[1 - t]) else min(float(land[1 - t]), CAP_M)
        out[i] = [np.log1p((j - e.end) / fps), lo, lp, lp - lo, float(miss),
                  min(float(np.nanmin(own)), CAP_M) if np.isfinite(own).any() else CAP_M,
                  min(float(np.nanmin(opp)), CAP_M) if np.isfinite(opp).any() else CAP_M,
                  float((own[ok] < opp[ok]).mean()) if ok.any() else 0.5,
                  margin(e.end + 10, t, stop), margin(e.end + 25, t, stop)]
    return out


def corridor_features(events, ball, players, team_of_slot, fps=25, look=5, v_min=3.0,
                      along=(2.0, 35.0), corridor_m=3.0):
    """(n_events, len(CORRIDOR_FEATURES)): who stands in the ball's path after the contact, from
    the toucher's perspective (offline). u is the ball's direction over the first look frames
    after the contact (valid only at >= v_min m/s); among players ahead of the ball (along the
    ray within `along`), each team's least perpendicular distance to the ray, the margin
    (opponent - own), and the team of the first player inside the corridor (+1 own, -1 opp)."""
    T, n = len(ball), len(events)
    out = np.zeros((n, len(CORRIDOR_FEATURES)))
    out[:, :2] = CAP_M
    for i, e in enumerate(events):
        a, b = e.end, min(e.end + look, T - 1)
        v = (ball[b] - ball[a]) * fps / max(b - a, 1)
        sp = float(np.hypot(*v)) if np.isfinite(v).all() else 0.0
        if sp < v_min:
            continue
        u = v / sp
        r = players[a] - ball[a]
        al = r @ u
        perp = np.abs(r[:, 0] * u[1] - r[:, 1] * u[0])
        ok = np.isfinite(al) & np.isfinite(perp) & (al > along[0]) & (al < along[1])
        best = []
        for t in (e.team, 1 - e.team):
            m = ok & (team_of_slot == t)
            best.append(min(float(perp[m].min()), CAP_M) if m.any() else CAP_M)
        inc = np.flatnonzero(ok & (perp < corridor_m) & (team_of_slot >= 0))
        first = 0.0
        if inc.size:
            first = 1.0 if team_of_slot[inc[np.argmin(al[inc])]] == e.team else -1.0
        out[i] = [best[0], best[1], best[1] - best[0], first, 1.0]
    return out


def oracle_receiver(events, gt, graded, lag):
    out = np.full(len(events), 0.5)
    T = len(gt)
    for i, e in enumerate(events):
        j = events[i + 1].start - 1 if i + 1 < len(events) else min(e.end + lag, T - 1)
        j = max(j, min(e.end + 1, T - 1))
        while j < T and graded[j] and gt[j] not in ("H", "A"):
            j += 1
        if j < T and graded[j] and gt[j] in ("H", "A"):
            out[i] = float(gt[j] == "HA"[e.team])
    return out


def oracle_label(events, gt, graded, lag, policy="legacy"):
    """The training-label rule (takeover.training_rows) on every contact: 1 if the toucher's team
    owns at the label frame. Under the alive_lag policies the label frame is the first alive
    graded frame >= end + lag (0.5 when none is left, where training drops the row)."""
    T = len(gt)
    out = np.empty(len(events))
    for i, e in enumerate(events):
        j = min(e.end + lag, T - 1)
        if policy in ("alive_lag", "alive_lag_restarts"):
            while j < T and graded[j] and gt[j] not in ("H", "A"):
                j += 1
            if j >= T or not graded[j] or gt[j] not in ("H", "A"):
                out[i] = 0.5
                continue
        out[i] = float(gt[j] == "HA"[e.team])
    return out


def sign_p(d):
    w = int((d > 0).sum())
    return w, sum(comb(len(d), k) for k in range(w, len(d) + 1)) / 2 ** len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    args = ap.parse_args()
    cfg = config.load(os.path.abspath(args.config))
    check_sections(cfg, args.config)
    if (cfg.get("deadstate") or {}).get("enabled") or (cfg.get("postprocess") or {}).get("enabled"):
        raise SystemExit("{}: this harness shares only the deadhybrid / evidence-D and gap stages "
                         "of paired.labels_for; deadstate and postprocess configs are refused so "
                         "base cannot silently differ from make paired".format(args.config))
    oc, gc, tm = cfg["owner"], cfg["gaps"], cfg["takeover_model"]
    fit, held = cfg["matches"]["fit"], cfg["matches"]["holdout"]
    prefix = cfg["matches"].get("prefix", "DFL-")
    Q = [Prepared(prefix + g, cfg) for g in fit + held]
    nf = len(fit)
    lag = oc["label_lag_frames"]
    names = takeover.feature_names(cfg)
    assert Q[0].X.shape[1] == len(names)
    dh = cfg.get("deadhybrid") or {}
    dead = deadhybrid_parts(cfg, Q, nf) if dh.get("enabled") else [None] * len(Q)
    G = [Gaps(q, gc) for q in Q] if gc["enabled"] else None
    if G:
        gn = gapstate.feature_names(gc)
        probs = [G[i].probs(fit_gaps([G[j] for j in range(nf) if j != i], cfg["gap_model"], gn))
                 for i in range(nf)]
        gm = fit_gaps(G[:nf], cfg["gap_model"], gn)
        probs += [g.probs(gm) for g in G[nf:]]

    def score(p_list):
        out = []
        for i, (q, p) in enumerate(zip(Q, p_list)):
            lab = q.labels(p, oc["alpha"], oc["flip_penalty"], dead=dead[i])
            if G:
                lab = decode_gaps(lab, G[i].gap, *probs[i], gc["step_frames"], gc["lam_dead"],
                                  gc["lam_owner"])
            out.append(q.m.evaluate(lab)["spell_f1"])
        return np.array(out)

    def crossfit(Xs, nm):
        p = []
        for i in range(len(Q)):
            key = fold_of(i, nf)
            b = takeover.train(np.concatenate([Xs[j][Q[j].rows] for j in key]),
                               np.concatenate([Q[j].y for j in key]), tm, nm)
            p.append(takeover.predict(b, Xs[i]))
        return p

    arms = {}
    arms["base"] = score(crossfit([q.X for q in Q], names))
    Xr = [np.hstack([q.X, oracle_receiver(q.events, q.m.gt, q.m.graded, lag)[:, None]])
          for q in Q]
    arms["receiver"] = score(crossfit(Xr, names + ("oracle_recv",)))
    Xf = [np.hstack([q.X, flight_features(q.events, q.m.d, q.m.fps)]) for q in Q]
    arms["flight"] = score(crossfit(Xf, names + FLIGHT_FEATURES))
    Xc = [np.hstack([q.X, corridor_features(q.events, q.m.ball, q.m.players,
                                                      q.m.atd.team_of_slot, q.m.fps)])
          for q in Q]
    arms["corridor"] = score(crossfit(Xc, names + CORRIDOR_FEATURES))
    policy = oc.get("label_policy", "legacy")
    arms["perfect"] = score([np.clip(oracle_label(q.events, q.m.gt, q.m.graded, lag, policy),
                                     1e-3, 1 - 1e-3) for q in Q])
    agree = np.concatenate([(oracle_receiver(q.events, q.m.gt, q.m.graded, lag)[q.rows] == q.y)
                            for q in Q[nf:]]).mean()
    print("held-out takeover rows where oracle_recv == training label: {:.3f}".format(agree))
    games = [q.m.game for q in Q]
    print("{:<12}".format("match") + "".join("{:>10}".format(k) for k in arms))
    for i, g in enumerate(games):
        print("{:<12}".format(g) + "".join("{:>10.4f}".format(v[i]) for v in arms.values())
              + ("  held-out" if i >= nf else ""))
    b = arms["base"]
    for k in ("receiver", "flight", "corridor", "perfect"):
        d = arms[k] - b
        w, p = sign_p(d)
        print("{:<9} vs base: mean {:+.4f}, wins {}/{}, p = {:.4f}; held-out {:.4f} -> {:.4f} "
              "({:+.4f})".format(k, d.mean(), w, len(d), p, b[nf:].mean(), arms[k][nf:].mean(),
                                 arms[k][nf:].mean() - b[nf:].mean()))


if __name__ == "__main__":
    main()
