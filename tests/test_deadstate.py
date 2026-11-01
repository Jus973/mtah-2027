"""Learned dead ball (v8): realistic dead-time ball for training, features, 2-state decode."""

import copy
import os

import numpy as np

from mtah import config, pipeline, registry
from mtah.decode.dead import decode_dead
from mtah.learn import deadstate

V8 = config.load("rules_v8.yaml")


class _Atd:
    def __init__(self, ball, players, pitch=(100.0, 50.0)):
        self.pitch = pitch
        self.ball = ball / np.array(pitch)
        self._players = players

    def ball_m(self):
        return self.ball * np.array(self.pitch)

    def players_m(self):
        return self._players


def _world():
    T = 60
    ball = np.column_stack([np.linspace(10, 40, T), np.full(T, 20.0)])
    ball[20:40] = ball[19]                         # dead: the ball is frozen for 20 frames
    players = np.full((T, 2, 3), np.nan)
    players[:, 0, :2] = [80.0, 10.0]
    players[:, 1, :2] = np.column_stack([np.linspace(30, 20, T), np.full(T, 25.0)])
    players[30:, 1, :2] = ball[19] + [0.5, 0.0]    # a player reaches the ball at frame 30
    return _Atd(ball, players)


def test_realistic_dead_ball():
    atd = _world()
    cfg = dict(V8["deadstate"])
    lost = deadstate.realistic_dead_ball(atd, "lost", cfg, 0)
    assert np.isnan(lost.ball[20:40]).all() and np.isfinite(lost.ball[:20]).all()
    jit = deadstate.realistic_dead_ball(atd, "jitter", cfg, 0).ball_m()
    step = np.linalg.norm(np.diff(jit[19:40], axis=0), axis=1)
    assert (step > 0).all() and step.max() < 0.2
    up = deadstate.realistic_dead_ball(atd, "pickup", cfg, 0).ball_m()
    carried = np.linalg.norm(up[30:40] - atd.players_m()[30:40, 1, :2], axis=1)
    assert carried.max() < 1.5                     # moves with the player once fetched
    np.testing.assert_array_equal(atd.ball_m()[25], _world().ball_m()[25])   # input untouched


def test_ball_features():
    T = 120
    ball = np.column_stack([np.linspace(10, 70, T), np.full(T, 20.0)])
    ball[30:90] = ball[29]                         # a 60-frame stoppage
    B = deadstate.ball_features(ball, ball, (100.0, 50.0))
    assert B.shape == (T, len(deadstate.BALL))
    assert B[60, 0] < 0.01 < 1.0 < B[8, 0]         # 1 s mean speed: still vs moving ball
    assert B[0, 5] == 10.0                         # nearest line: 10 m from the goal line


def test_decode_dead():
    graded = np.ones(100, bool)
    graded[90:] = False
    p = np.full(100, 0.1)
    p[30:60] = 0.9
    p[45] = 0.2                                    # one doubtful step inside the stoppage
    dead = decode_dead(p, graded, step=5, lam=2.0)
    assert np.flatnonzero(dead).tolist() == list(range(30, 60))
    assert not dead[90:].any()
    assert not decode_dead(p, graded, step=5, lam=100.0).any()


def test_v8_pipeline_on_the_fake_match(fake, fake_atd):
    assert os.path.isfile(deadstate.model_path(V8))   # committed, trained on the sim only
    atd, _ = fake_atd
    sp = registry.spec(fake["game"], fake["data_dir"])
    a, info = pipeline.run(fake["game"], V8, fake["data_dir"], atd=copy.copy(atd))
    b, _ = pipeline.run(fake["game"], V8, fake["data_dir"], atd=copy.copy(atd))
    np.testing.assert_array_equal(a, b)
    graded = registry.graded_mask(sp)
    assert set(a[graded]) <= {"H", "A", "D"} and set(a[~graded]) == {"X"}
    (_, f2, _), k2 = sp["half_blocks"][1], info["kickoffs"][2]
    assert (a[f2:k2] == "D").all()                 # before a kick-off the ball stays dead
