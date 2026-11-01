"""The pipeline end to end: one function from a match to its per-frame H / A / D / X labels.

The same code runs on the real match, the sim and the fake match; only the config differs:
clean ball -> per-team distances -> touches -> dead-ball evidence -> D runs -> owner decoder.
owner.decoder is "rules" (owner.hypothesis over touches, v1) or "viterbi" (every contact, with a
learned takeover probability, decoded as a sparse-transition path, v3). With gaps.enabled (v5),
long ball gaps are then re-decoded from the players alone (decode/gaps.py), except the frames
before a kick-off, which stay D. Before a half's kick-off
the ball is dead: where the half's graded block starts more than kickoff.tolerance_frames before
the derived kick-off (events/restarts.py: Kick Off (Start) tag + line-up break), the frames up to
the kick-off are D. No published frame number is used. Frames outside the graded halves are X;
graded frames never are.
"""

import os

import numpy as np

from mtah import registry
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
from mtah.learn import gapstate, takeover


def kickoffs(game, atd, cfg, data_dir=None):
    """{half: frame} from the tags + line-up, or {} for a match without Smart Tagging (sim)."""
    sp = registry.spec(game, data_dir)
    path = registry.asset_paths(game, data_dir)["smart_tagging"]
    if not os.path.isfile(path):
        return {}
    found = detect_kickoffs(atd, load_tags(path, sp["fps"]), cfg["kickoff"], sp["fps"])
    return {h: r["frame"] for h, r in found.items() if r["frame"] is not None}


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
    runs = dead_runs(ev, touches, graded, cfg["deadball"], tuple(cfg["deadball"]["kinds"]),
                     out_restarts=restarts)
    for a, b, _ in runs:
        dead[a:b + 1] = True
    kicks = kickoffs(game, atd, cfg, data_dir)
    tol = cfg["kickoff"]["tolerance_frames"]
    pre_kick = np.zeros(sp["n_frames"], dtype=bool)
    for half, first, last in sp["half_blocks"]:
        k = kicks.get(half)
        if k is not None and k - first > tol:
            pre_kick[first:min(k, last + 1)] = True
    dead |= pre_kick
    dead &= graded
    oc = cfg["owner"]
    gc = cfg.get("gaps")
    F = None
    if (gc and gc["enabled"]) or oc.get("shape_features"):
        F = gapstate.features(players, atd.team_of_slot, gapstate.home_direction(atd, sp, kicks),
                              atd.pitch, gc, fps)
    if oc["decoder"] == "viterbi":
        events = takeover.contacts(ball, d, cfg["touch"], oc["contact_r_m"], fps)
        X = takeover.features(events, d, fps)
        if oc.get("shape_features"):
            X = np.hstack([X, takeover.shape_features(events, F, gapstate.feature_names(gc))])
        p = takeover.predict(takeover.load(cfg), X)
        if restarts:                     # a release restart is the taker's: a certain takeover
            events, p = takeover.with_restarts(events, p, restarts)
        labels = decode_viterbi(events, p, graded, dead, oc["alpha"], oc["flip_penalty"])
        n_contacts = len(events)
    else:
        owner_touches = sorted(touches + restarts, key=lambda t: t.start)
        labels = decode_owner(owner_touches, graded, dead, oc["hypothesis"], oc, fps)
        n_contacts = None
    info = {"touches": len(touches), "contacts": n_contacts, "release_restarts": len(restarts),
            "d_runs_evidence": len(runs),
            "d_share_graded": float(dead[graded].mean()), "kickoffs": kicks}
    if gc and gc["enabled"]:
        gap = gapstate.long_gaps(observed, graded, gc["min_frames"]) & ~pre_kick
        idx = np.flatnonzero(gap)
        p_dead, p_home = np.full(sp["n_frames"], np.nan), np.full(sp["n_frames"], np.nan)
        p_dead[idx], p_home[idx] = gapstate.predict(gapstate.load(cfg), F, idx)
        labels = decode_gaps(labels, gap, p_dead, p_home, gc["step_frames"], gc["lam_dead"],
                             gc["lam_owner"])
        info["gap_mask"] = gap
        info["gap_frames"] = int(gap.sum())
        alive = np.isin(labels[gap], ["H", "A"])
        info["gap_alive_share"] = float(alive.mean()) if idx.size else 0.0
    return labels, info
