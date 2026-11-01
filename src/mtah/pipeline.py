"""Rules v1 end to end: one function from a match to its per-frame H / A / D / X labels.

The same code runs on the real match, the sim and the fake match; only the config differs:
clean ball -> per-team distances -> touches -> dead-ball evidence -> D runs -> owner decoder
(owner.hypothesis). Before a half's kick-off the ball is dead: where the half's graded block
starts more than kickoff.tolerance_frames before the derived kick-off (events/restarts.py: Kick Off
(Start) tag + line-up break), the frames up to the kick-off are D. No published frame number is
used. Frames outside the graded halves are X; graded frames never are.
"""

import os

import numpy as np

from mtah import registry
from mtah.decode.rules import decode_owner
from mtah.events.deadball import dead_runs, evidence
from mtah.events.restarts import detect_kickoffs
from mtah.events.touches import touch_events
from mtah.features.ball import clean_ball
from mtah.features.players import team_distances
from mtah.io.atd import load_game
from mtah.io.tags import load_tags


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
    runs = dead_runs(ev, touches, graded, cfg["deadball"])
    for a, b, _ in runs:
        dead[a:b + 1] = True
    kicks = kickoffs(game, atd, cfg, data_dir)
    tol = cfg["kickoff"]["tolerance_frames"]
    for half, first, last in sp["half_blocks"]:
        k = kicks.get(half)
        if k is not None and k - first > tol:
            dead[first:min(k, last + 1)] = True
    dead &= graded
    labels = decode_owner(touches, graded, dead, cfg["owner"]["hypothesis"], cfg["owner"], fps)
    info = {"touches": len(touches), "d_runs_evidence": len(runs),
            "d_share_graded": float(dead[graded].mean()), "kickoffs": kicks}
    return labels, info
