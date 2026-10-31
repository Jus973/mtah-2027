"""Dead-ball v1: each evidence kind, merging, and D running up to the restart touch."""

import numpy as np

from mtah import config
from mtah.events.deadball import dead_mask, dead_runs, evidence
from mtah.events.touches import Touch

CFG = dict(config.load("rules.yaml")["deadball"], oob_margin_m=0.5, oob_min_frames=2,
           gap_min_frames=50, gap_offset_frames=-10, speed_window_frames=25, v_still_mps=0.5,
           v_players_mps=1.0, still_min_frames=50, merge_frames=20)
PITCH = (105.0, 68.0)


def world(T=600):
    """Ball rolling at 5 m/s along the halfway line, 4 players jogging at 3 m/s."""
    t = np.arange(T)
    ball = np.column_stack([10 + 0.2 * (t % 400), np.full(T, 34.0)])
    players = np.stack([np.column_stack([20 + 0.12 * t * (i % 2 * 2 - 1) + 30, np.full(T, 10.0 + 10 * i)])
                        for i in range(4)], axis=1)
    return ball, players


def ev_of(ball_raw, players, graded=None):
    graded = np.ones(len(ball_raw), bool) if graded is None else graded
    observed = np.isfinite(ball_raw[:, 0])
    return evidence(ball_raw, ball_raw, observed, players, graded, PITCH, CFG)


def touch(start, team=0):
    return Touch(start=start, end=start + 4, team=team, n_frames=5, d_min=0.5,
                 speed_before=1.0, speed_after=8.0, turn_deg=30.0)


def test_out_of_bounds():
    ball, players = world()
    ball[100:105, 1] = -1.0          # 1 m past the touch line: out
    ball[200:205, 1] = -0.3          # within the margin: in
    ball[300:301, 1] = -2.0          # one frame only: noise
    ev = ev_of(ball, players)
    assert ev["oob"][100:105].all() and not ev["oob"][200:205].any() and not ev["oob"][300]


def test_long_gap_with_offset():
    ball, players = world()
    ball[100:200] = np.nan           # 100 frames: a stoppage
    ball[300:320] = np.nan           # 20 frames: an occlusion
    ev = ev_of(ball, players)
    assert ev["gap"][90:200].all() and not ev["gap"][89] and not ev["gap"][300:320].any()


def test_still_needs_ball_and_players_calm():
    ball, players = world()
    ball[100:250] = ball[100]                       # ball parked ...
    calm = players.copy()
    calm[100:250] = calm[100]                       # ... and players standing: a stoppage
    assert ev_of(ball, calm)["still"][130:220].all()
    assert not ev_of(ball, players)["still"].any()  # players still jogging: a keeper's ball


def test_dead_runs_end_at_the_restart_touch():
    T = 600
    graded = np.zeros(T, bool)
    graded[0:300] = graded[350:600] = True
    ev = {k: np.zeros(T, bool) for k in ("oob", "gap", "still")}
    ev["oob"][100:110] = True
    ev["gap"][120:150] = True                       # within merge_frames: same stoppage
    ev["still"][400:460] = True                     # no touch after it: D to the half's end
    touches = [touch(50), touch(130), touch(170, team=1)]
    runs = dead_runs(ev, touches, graded, CFG)
    assert runs[0][:2] == (100, 169) and set(runs[0][2]) == {"oob", "gap"}
    assert runs[1][:2] == (400, 599) and runs[1][2] == ("still",)
    mask = dead_mask(ev, touches, graded, CFG)
    assert mask[100:170].all() and not mask[170] and not mask[300:350].any()
    only_oob = dead_runs(ev, touches, graded, CFG, kinds=("oob",))
    assert only_oob[0][:2] == (100, 129)             # restart = first touch after the evidence
