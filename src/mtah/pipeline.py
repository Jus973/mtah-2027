"""The pipeline end to end: one function from a match to its per-frame H / A / D / X labels.

The same code runs on the real match, the sim and the fake match; only the config differs:
clean ball -> per-team distances -> touches -> dead-ball evidence -> D runs -> owner decoder.
owner.decoder is "rules" (owner.hypothesis over touches, v1) or "viterbi" (every contact, with a
learned takeover probability, decoded as a sparse-transition path, v3). With gaps.enabled (v5),
long ball gaps are then re-decoded from the players alone (decode/gaps.py), except the frames
before a kick-off, which stay D. With deadstate.enabled (v8), D comes from a learned per-frame
model (learn/deadstate.py, decode/dead.py) instead of the evidence rules, and the gap decoder
takes its P(dead) from that model. Before a half's kick-off
the ball is dead: where the half's graded block starts more than kickoff.tolerance_frames before
the derived kick-off (events/restarts.py: Kick Off (Start) tag + line-up break), the frames up to
the kick-off are D. No published frame number is used. Frames outside the graded halves are X;
graded frames never are.
"""

import os

import numpy as np

from mtah import postprocess, registry
from mtah.decode.dead import decode_dead, refine_runs
from mtah.decode.gaps import decode_gaps
from mtah.decode.rules import decode_owner
from mtah.decode.viterbi import decode_viterbi
from mtah.events.deadball import dead_runs, evidence
from mtah.events.restarts import detect_kickoffs
from mtah.events.touches import touch_events
from mtah.features.ball import clean_ball
from mtah.features.players import team_distances
from mtah.io.atd import load_game
from mtah.io.tags import load_tags
from mtah.learn import deadstate, gapstate, takeover


def kickoffs(game, atd, cfg, data_dir=None):
    """{half: frame} from the tags + line-up, or {} for a match without Smart Tagging (sim)."""
    sp = registry.spec(game, data_dir)
    path = registry.asset_paths(game, data_dir)["smart_tagging"]
    if not os.path.isfile(path):
        return {}
    found = detect_kickoffs(atd, load_tags(path, sp["fps"]), cfg["kickoff"], sp["fps"])
    return {h: r["frame"] for h, r in found.items() if r["frame"] is not None}


def pre_kickoff_mask(sp, kicks, tol):
    """(T,) bool: graded frames before a half's derived kick-off, where the half's graded block
    opens more than tol frames before it; the pipeline holds these D and never gap-decodes them."""
    pre_kick = np.zeros(sp["n_frames"], dtype=bool)
    for half, first, last in sp["half_blocks"]:
        k = kicks.get(half)
        if k is not None and k - first > tol:
            pre_kick[first:min(k, last + 1)] = True
    return pre_kick


def run(game, cfg, data_dir=None, atd=None):
    """(labels (T,) '<U1', info dict of counts) for one match under one config."""
    sp = registry.spec(game, data_dir)
    fps = sp["fps"]
    if atd is None:
        atd, _ = load_game(game, data_dir)
    graded = registry.graded_mask(sp)
    ball_raw = atd.ball_m()
    ball, observed, _ = clean_ball(ball_raw, atd.pitch, cfg["ball"], fps)
    players = atd.players_m()[..., :2].astype(float)
    d, _ = team_distances(players, ball, atd.team_of_slot)
    touches = touch_events(ball, d, cfg["touch"], fps)
    ev = evidence(ball_raw, ball, observed, players, graded, atd.pitch, cfg["deadball"], fps)
    dead = np.zeros(sp["n_frames"], dtype=bool)
    restarts = []                       # releases that ended a stoppage: the taker's touch
    runs_meta = []                      # per-run evidence end, for the postprocess backshift
    runs = dead_runs(ev, touches, graded, cfg["deadball"], tuple(cfg["deadball"]["kinds"]),
                     out_restarts=restarts, out_meta=runs_meta)
    for a, b, _ in runs:
        dead[a:b + 1] = True
    kicks = kickoffs(game, atd, cfg, data_dir)
    pre_kick = pre_kickoff_mask(sp, kicks, cfg["kickoff"]["tolerance_frames"])
    oc = cfg["owner"]
    gc = cfg.get("gaps")
    ds = cfg.get("deadstate")
    dh = cfg.get("deadhybrid")
    F = None
    if (gc and gc["enabled"]) or oc.get("shape_features") or (ds and ds["enabled"]) \
            or (dh and dh.get("enabled")):
        F = gapstate.features(players, atd.team_of_slot,
                              gapstate.home_direction(atd, sp, kicks,
                                                      cfg["direction"]["window_frames"]),
                              atd.pitch, gc, fps)
    p_learned = None
    if ds and ds["enabled"]:
        FD = deadstate.features(F, deadstate.ball_features(ball_raw, ball, atd.pitch, fps))
        gr = np.flatnonzero(graded)
        p_learned = np.full(sp["n_frames"], np.nan)
        p_learned[gr] = deadstate.predict(deadstate.load(cfg), FD, gr)
        dead = decode_dead(p_learned, graded, ds["step_frames"], ds["lam"])
        restarts = []
    if dh and dh.get("enabled"):
        # P3e hybrid: the CAUSAL dead model may refine each evidence boundary within +/- W
        # frames (decode/dead.py refine_runs); the runs, gap decoder and owner path stay v6's.
        if (ds and ds["enabled"]) or restarts or (cfg.get("postprocess") or {}).get("enabled"):
            raise ValueError("deadhybrid refines the evidence boundaries and never stacks with "
                             "deadstate, release restarts or postprocess")
        import lightgbm as lgb
        FD = deadstate.features(F, deadstate.causal_ball(
            deadstate.ball_features(ball_raw, ball, atd.pitch, fps)))
        W = dh["window_frames"]
        T = sp["n_frames"]
        idx = [np.arange(max(0, x - W), min(T, x + W + 1)) for a, b, _ in runs for x in (a, b)]
        idx = np.unique(np.concatenate(idx)) if idx else np.array([], dtype=np.int64)
        idx = idx[graded[idx]]
        p_dh = np.full(T, np.nan)
        if idx.size:
            booster = lgb.Booster(model_file=os.path.join(registry.ROOT, dh["model"]))
            p_dh[idx] = deadstate.predict(booster, FD, idx)
        dead = np.zeros(T, dtype=bool)
        for a, b in refine_runs([(a, b) for a, b, _ in runs], p_dh, graded, W):
            dead[a:b + 1] = True
    dead |= pre_kick
    dead &= graded
    if oc["decoder"] == "viterbi":
        events = takeover.contacts(ball, d, cfg["touch"], oc["contact_r_m"], fps)
        X = takeover.features(events, d, fps)
        if oc.get("shape_features"):
            X = np.hstack([X, takeover.shape_features(events, F, gapstate.feature_names(gc))])
        p = takeover.predict(takeover.load(cfg), X)
        if restarts:                     # a release restart is the taker's (restart_takeover_p)
            events, p = takeover.with_restarts(events, p, restarts,
                                               cfg["deadball"].get("restart_takeover_p", 1.0))
        labels = decode_viterbi(events, p, graded, dead, oc["alpha"], oc["flip_penalty"])
        n_contacts = len(events)
    else:
        owner_touches = sorted(touches + restarts, key=lambda t: t.start)
        labels = decode_owner(owner_touches, graded, dead, oc["hypothesis"], oc, fps)
        n_contacts = None
    info = {"touches": len(touches), "contacts": n_contacts, "release_restarts": len(restarts),
            "d_runs_evidence": len(runs), "dead_runs_meta": runs_meta,
            "d_share_graded": float(dead[graded].mean()), "kickoffs": kicks}
    if gc and gc["enabled"]:
        gap = gapstate.long_gaps(observed, graded, gc["min_frames"]) & ~pre_kick
        idx = np.flatnonzero(gap)
        p_dead, p_home = np.full(sp["n_frames"], np.nan), np.full(sp["n_frames"], np.nan)
        p_dead[idx], p_home[idx] = gapstate.predict(gapstate.load(cfg), F, idx)
        if p_learned is not None:
            p_dead[idx] = p_learned[idx]
        labels = decode_gaps(labels, gap, p_dead, p_home, gc["step_frames"], gc["lam_dead"],
                             gc["lam_owner"])
        info["gap_mask"] = gap
        info["gap_frames"] = int(gap.sum())
        alive = np.isin(labels[gap], ["H", "A"])
        info["gap_alive_share"] = float(alive.mean()) if idx.size else 0.0
    pp = cfg.get("postprocess")
    if pp and pp.get("enabled"):
        if ds and ds["enabled"] and pp.get("backshift_min_late_frames"):
            raise ValueError("postprocess.backshift_min_late_frames reads the evidence-rule "
                             "dead-run metadata and cannot run with deadstate.enabled")
        stats = {}
        labels = postprocess.apply(labels, pp, runs_meta, protect=pre_kick, out_stats=stats)
        info["postprocess"] = stats
        info["d_share_graded_final"] = float(np.mean(labels[graded] == "D"))
        if gc and gc["enabled"]:
            alive = np.isin(labels[info["gap_mask"]], ["H", "A"])
            info["gap_alive_share"] = float(alive.mean()) if info["gap_frames"] else 0.0
    po = cfg.get("proxowner")
    if po and po.get("enabled"):
        # v12 (board #67): blend the per-frame proximity owner into every alive run, with the
        # pre-registered setting the match's kick rate selects (learn/proxowner.py).
        from mtah.learn import proxowner
        views = proxowner.match_views(atd, sp, kicks, cfg, ball_raw=ball_raw)
        rate = proxowner.kick_rate(players, ball_raw, graded, atd.pitch, fps)
        w, lam = proxowner.setting(cfg, rate)
        labels = proxowner.blend_decode(labels, proxowner.p_home(proxowner.load(cfg), views), w, lam)
        info["kick_rate"], info["proxowner_setting"] = rate, (w, lam)
    return labels, info
