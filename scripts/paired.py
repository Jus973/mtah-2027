"""make paired A=configs/final.yaml B=configs/rules_v6.yaml: the PLAN.md §6.4 paired per-match
comparison of two viterbi configs on the sim. Open data only (degraded DFL/IDSSE sim, §6.1).

Each config's learned parts are cross-fitted: on each fit match, the takeover model (and, with
gaps.enabled, the gap models; with deadstate.enabled, the dead model, trained as train_dead.py
trains it on the realistic-dead-ball variants) are trained on the other four fit matches; the
held-out matches use models trained on all five. Every match is then scored end to end with the
official scorer — release restarts injected, deadstate decoding and the postprocess label rules
applied, exactly as mtah.pipeline would — both configs on the same matches and GT. B replaces A
only if the mean paired difference is positive and the one-sided sign test gives p <= 0.05 (with
7 matches: 7 wins). A possession-level paired bootstrap (same GT resamples in both arms) is
reported beside the sign test, as a second opinion, never the gate.

A config section this harness does not apply is REFUSED, never ignored: scoring an unapplied
section silently falls back to the base pipeline and returns a guaranteed tie (the v8 failure
class). New sections go into APPLIED (with code) or INERT (with a reason they cannot change sim
labels).
"""

import argparse
import os
from math import comb

import numpy as np

from mtah import config, postprocess
from mtah.decode.gaps import decode_gaps
from mtah.eval.score import pm
from mtah.learn import gapstate
from train_gaps import Gaps, fit_gaps
from train_takeover import Prepared, fit_on, predict

# Sections scores() applies to the sim labels. deadball includes the release-restart window
# (restarts injected by Prepared.labels); dead_model holds the dead boosters' parameters.
APPLIED = {"ball", "touch", "deadball", "owner", "takeover_model", "gaps", "gap_model",
           "deadstate", "deadhybrid", "dead_model", "postprocess", "matches", "proxowner"}
# Sections that cannot change sim labels: reporting, guards and real-match-only machinery.
# kickoff/direction are label-relevant only through Smart Tagging, which the sim does not have
# (pipeline.kickoffs returns {} there, and Prepared uses the same empty-kickoff direction).
INERT = {"team", "tag_team_map", "kickoff", "direction", "bands", "public_check", "published",
         "stability", "disagreement", "sync", "whistle", "clipviewer", "measure", "degrade",
         "targets", "tags", "metrica_accept"}  # metrica_accept: P5 validation-match acceptance
                                               # thresholds, never read by the DFL-sim pipeline


def check_sections(cfg, path):
    unknown = sorted(set(cfg) - APPLIED - INERT)
    if unknown:
        raise SystemExit(
            "{}: paired.py does not apply config section(s) {} - scoring would silently fall "
            "back to the base pipeline and tie by construction. Teach scores() the section and "
            "list it in APPLIED, or list it in INERT with a reason it cannot change sim "
            "labels.".format(path, unknown))


def fold_of(i, nf):
    return tuple(j for j in range(nf) if j != i) if i < nf else tuple(range(nf))


def dead_training_rows(cfg, Q, nf, section, ball_fn):
    """(rows[fit match][condition], F list) — the train_dead.py training data, shared by the
    deadstate and deadhybrid branches and by scripts/stability.py's seed retrains: realistic-
    dead-ball variants (train_conditions, seed = fit-match index), condition-major. ball_fn maps
    the raw ball-feature block to the branch's view (identity, or the causal subset)."""
    from mtah.features.ball import clean_ball
    from mtah.io.atd import load_game
    from mtah.learn import deadstate
    sec, gc = cfg[section], cfg["gaps"]
    F = [q.F if hasattr(q, "F") else
         gapstate.features(q.m.players, q.m.atd.team_of_slot,
                           gapstate.home_direction(q.m.atd, q.m.spec, {},
                                                   cfg["direction"]["window_frames"]),
                           q.m.atd.pitch, gc, q.m.fps) for q in Q]
    conds = list(sec["train_conditions"])
    rows = [[None] * len(conds) for _ in range(nf)]
    for ci, cond in enumerate(conds):
        for j in range(nf):
            atd = deadstate.realistic_dead_ball(load_game(Q[j].m.game)[0], cond, sec, seed=j)
            ball_raw = atd.ball_m()
            ball, _, _ = clean_ball(ball_raw, atd.pitch, cfg["ball"], Q[j].m.fps)
            FD = deadstate.features(F[j], ball_fn(deadstate.ball_features(ball_raw, ball,
                                                                          atd.pitch,
                                                                          Q[j].m.fps)))
            rows[j][ci] = deadstate.training_rows(FD, Q[j].m.gt, Q[j].m.graded,
                                                  sec["step_frames"])
    return rows, F


def train_dead_on(cfg, rows, key, names):
    from mtah.learn import deadstate
    data = [rows[j][ci] for ci in range(len(rows[0])) for j in key]
    return deadstate.train(np.vstack([x for x, _ in data]),
                           np.concatenate([y for _, y in data]), cfg["dead_model"], names)


def dead_fold_models(cfg, Q, nf, section, names, ball_fn):
    """{fold: booster}, F list — cross-fitted dead models (the train_dead.py protocol)."""
    rows, F = dead_training_rows(cfg, Q, nf, section, ball_fn)
    dm = {key: train_dead_on(cfg, rows, key, names)
          for key in sorted({fold_of(i, nf) for i in range(len(Q))})}
    return dm, F


def deadstate_parts(cfg, Q, nf):
    """Cross-fitted learned D (deadstate.enabled, v8): per match (decoded dead mask, p_learned);
    the gap decoder's p_dead is then replaced by p_learned (mtah.pipeline does the same)."""
    from mtah.decode.dead import decode_dead
    from mtah.learn import deadstate
    ds = cfg["deadstate"]
    names = deadstate.feature_names(gapstate.feature_names(cfg["gaps"]))
    dm, F = dead_fold_models(cfg, Q, nf, "deadstate", names, lambda B: B)
    dead, p_all = [], []
    for i, q in enumerate(Q):
        FD = deadstate.features(F[i], deadstate.ball_features(q.m.ball_raw, q.m.ball,
                                                              q.m.atd.pitch, q.m.fps))
        gr = np.flatnonzero(q.m.graded)
        pl = np.full(len(q.m.gt), np.nan)
        pl[gr] = deadstate.predict(dm[fold_of(i, nf)], FD, gr)
        dead.append(decode_dead(pl, q.m.graded, ds["step_frames"], ds["lam"]) & q.m.graded)
        p_all.append(pl)
    return dead, p_all


def deadhybrid_parts(cfg, Q, nf):
    """Cross-fitted P3e hybrid (deadhybrid.enabled): per match the dead mask whose evidence-run
    boundaries were refined within +/- window_frames by the CAUSAL dead model (no future-speed
    features). The runs themselves are v6's; the gap decoder is untouched."""
    from mtah.decode.dead import refine_runs
    from mtah.learn import deadstate
    dh = cfg["deadhybrid"]
    names = deadstate.causal_names(gapstate.feature_names(cfg["gaps"]))
    dm, F = dead_fold_models(cfg, Q, nf, "deadhybrid", names, deadstate.causal_ball)
    W = dh["window_frames"]
    dead = []
    for i, q in enumerate(Q):
        FD = deadstate.features(F[i], deadstate.causal_ball(
            deadstate.ball_features(q.m.ball_raw, q.m.ball, q.m.atd.pitch, q.m.fps)))
        T = len(q.m.gt)
        idx = [np.arange(max(0, x - W), min(T, x + W + 1))
               for a, b, _ in q.dead_runs for x in (a, b)]
        idx = np.unique(np.concatenate(idx)) if idx else np.array([], dtype=np.int64)
        idx = idx[q.m.graded[idx]]
        pl = np.full(T, np.nan)
        if idx.size:
            pl[idx] = deadstate.predict(dm[fold_of(i, nf)], FD, idx)
        mask = np.zeros(T, dtype=bool)
        for a, b in refine_runs([(a, b) for a, b, _ in q.dead_runs], pl, q.m.graded, W):
            mask[a:b + 1] = True
        dead.append(mask & q.m.graded)
    return dead


def labels_for(path):
    """(match ids, n fit, per-match label arrays, SimMatch list) for one config, every learned
    part cross-fitted and every enabled section applied (or the config refused)."""
    cfg = config.load(os.path.abspath(path))
    check_sections(cfg, path)
    oc, gc = cfg["owner"], cfg.get("gaps")
    ds, ppc = cfg.get("deadstate"), cfg.get("postprocess")
    if oc["decoder"] != "viterbi":
        raise SystemExit("{}: paired.py compares viterbi configs".format(path))
    learned_d = bool(ds and ds["enabled"])
    if ppc and ppc.get("enabled") and ppc.get("backshift_min_late_frames") and learned_d:
        raise SystemExit("{}: postprocess.backshift_min_late_frames reads the evidence-rule "
                         "dead-run metadata and cannot run with deadstate.enabled".format(path))
    dh = cfg.get("deadhybrid")
    hybrid = bool(dh and dh.get("enabled"))
    if hybrid and (learned_d or (ppc and ppc.get("enabled"))
                   or cfg["deadball"].get("restart_window_frames")):
        raise SystemExit("{}: deadhybrid refines the evidence boundaries and never stacks with "
                         "deadstate, postprocess or release restarts".format(path))
    fit, held = cfg["matches"]["fit"], cfg["matches"]["holdout"]
    prefix = cfg["matches"].get("prefix", "DFL-")
    Q = [Prepared(prefix + g, cfg) for g in fit + held]
    nf = len(fit)
    p = [predict(fit_on([Q[j] for j in range(nf) if j != i], cfg["takeover_model"]), Q[i])
         for i in range(nf)]
    full = fit_on(Q[:nf], cfg["takeover_model"])
    p += [predict(full, q) for q in Q[nf:]]
    dead, p_learned = ([None] * len(Q), None)
    if learned_d:
        dead, p_learned = deadstate_parts(cfg, Q, nf)
    elif hybrid:
        dead = deadhybrid_parts(cfg, Q, nf)
    labels = [q.labels(pq, oc["alpha"], oc["flip_penalty"], dead=dd,
                       inject_restarts=not learned_d)
              for q, pq, dd in zip(Q, p, dead)]
    if gc and gc["enabled"]:
        G = [Gaps(q, gc) for q in Q]
        names = gapstate.feature_names(gc)
        probs = [G[i].probs(fit_gaps([G[j] for j in range(nf) if j != i], cfg["gap_model"],
                                     names)) for i in range(nf)]
        models = fit_gaps(G[:nf], cfg["gap_model"], names)
        probs += [g.probs(models) for g in G[nf:]]
        if p_learned is not None:
            for g, pr, pl in zip(G, probs, p_learned):
                pr[0][g.idx] = pl[g.idx]
        labels = [decode_gaps(lab, g.gap, *pr, gc["step_frames"], gc["lam_dead"],
                              gc["lam_owner"]) for lab, g, pr in zip(labels, G, probs)]
    if ppc and ppc.get("enabled"):
        labels = [postprocess.apply(lab, ppc, q.runs_meta) for q, lab in zip(Q, labels)]
    pc = cfg.get("proxowner")
    if pc and pc.get("enabled"):
        # v12: the proximity owner model cross-fitted like every other learned part (trained on
        # the RAW fit sims, see train_proxowner.py), its blend picked by each match's kick rate
        # in the world being evaluated.
        from mtah.learn import proxowner
        from train_proxowner import fit_on as fit_prox
        games = [q.m.game for q in Q]
        models = [fit_prox([games[j] for j in range(nf) if j != i], cfg) for i in range(nf)]
        models += [fit_prox(games[:nf], cfg)] * (len(Q) - nf)
        out = []
        for q, lab, mdl in zip(Q, labels, models):
            m = q.m
            views = proxowner.match_views(m.atd, m.spec, {}, cfg, ball_raw=m.ball_raw)
            rate = proxowner.kick_rate(m.players, m.ball_raw, m.graded, m.atd.pitch, m.fps)
            w, lam = proxowner.setting(cfg, rate)
            out.append(proxowner.blend_decode(lab, proxowner.p_home(mdl, views), w, lam))
            print("{}: kick rate {:.1f} -> proxowner setting w {} lam {}".format(m.game, rate, w, lam))
        labels = out
    return [q.m.game for q in Q], nf, labels, Q


def scores(path):
    """(match ids, n fit, per-match spellF1) for one config (the §6.4 numbers)."""
    games, nf, labels, Q = labels_for(path)
    return games, nf, np.array([q.m.evaluate(lab)["spell_f1"] for q, lab in zip(Q, labels)])


def paired_bootstrap(gts, a_labels, b_labels, n_draws=2000, seed=0):
    """One-sided p for "mean paired spellF1 delta <= 0" and the delta's 5-95% interval,
    resampling GT possessions per match with the SAME draws in both arms (the possession-level
    analogue of the scorer's bootstrap_spell_f1; reported beside the sign test, never the gate)."""
    per_match = []
    for gt, la, lb in zip(gts, a_labels, b_labels):
        gt_c = pm.label_codes(np.asarray(gt))
        runs = pm.valid_runs(gt_c)
        gt_spells = pm.possession_spells(gt_c, runs)
        arms = []
        for lab in (la, lb):
            pred = pm.fill_abstentions(pm.label_codes(np.asarray(lab)), runs)
            spells = pm.possession_spells(pred, runs)
            overlaps = pm.spell_overlaps(gt_spells, spells)
            flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), t)
                               for t in pm.SPELL_IOUS])
            arms.append((flags, len(spells)))
        per_match.append((len(gt_spells), arms))
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_draws)
    for d in range(n_draws):
        ds = []
        for n_gt, arms in per_match:
            picks = rng.randint(0, n_gt, n_gt)
            f1 = []
            for flags, n_pred in arms:
                vals = []
                for row in flags:
                    tp = int(row[picks].sum())
                    fp = max(0, n_pred - tp)
                    vals.append(2.0 * tp / (2 * tp + fp + (n_gt - tp)) if tp else 0.0)
                f1.append(np.mean(vals))
            ds.append(f1[1] - f1[0])
        deltas[d] = np.mean(ds)
    lo, hi = np.percentile(deltas, [5, 95])
    return float(np.mean(deltas <= 0)), float(lo), float(hi)


def smoke(a_path, b_path):
    """P0.1 smoke: the harness measures, never ties. Same config twice = identical labels (their
    hashes printed, pinning the base); differing configs = differing labels on >= 1 match."""
    import hashlib
    games, nf, la, QA = labels_for(a_path)
    _, _, la2, _ = labels_for(a_path)
    for g, x, y in zip(games, la, la2):
        assert np.array_equal(x, y), g + ": the harness is not deterministic"
        print("{}  {}  ({})".format(g, hashlib.sha256("".join(x).encode()).hexdigest()[:16],
                                    a_path))
    _, _, lb, _ = labels_for(b_path)
    differ = [g for g, x, y in zip(games, la, lb) if not np.array_equal(x, y)]
    print("labels differ on {}/{} matches ({} vs {}): {}".format(
        len(differ), len(games), a_path, b_path, " ".join(differ) or "NONE"))
    if not differ:
        raise SystemExit("smoke FAILED: differing configs produced identical labels on every "
                         "match - the guaranteed-tie failure class is back")
    print("smoke PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="configs/final.yaml")
    parser.add_argument("--b", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true",
                        help="P0.1 harness smoke instead of the comparison")
    args = parser.parse_args()
    if args.smoke:
        smoke(args.a, args.b)
        return
    games, nf, la, QA = labels_for(args.a)
    games_b, nf_b, lb, _ = labels_for(args.b)
    if games_b != games or nf_b != nf:
        raise SystemExit("A and B must list the same matches AND the same fit/holdout split; "
                         "§6.4 compares two configs under one cross-fitting protocol")
    a = np.array([q.m.evaluate(lab)["spell_f1"] for q, lab in zip(QA, la)])
    b = np.array([q.m.evaluate(lab)["spell_f1"] for q, lab in zip(QA, lb)])
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
    if args.bootstrap:
        pb, lo, hi = paired_bootstrap([q.m.gt for q in QA], la, lb, args.bootstrap)
        print("possession-level paired bootstrap ({} draws, second opinion): "
              "P(mean delta <= 0) = {:.4f}, delta 5-95% [{:+.4f}, {:+.4f}]".format(
                  args.bootstrap, pb, lo, hi))
    print("verdict: {}".format("B replaces A" if d.mean() > 0 and p <= 0.05 else
                               "tie: keep A (or apply the tie-breaks)"))


if __name__ == "__main__":
    main()
