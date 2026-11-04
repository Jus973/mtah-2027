"""carrier_stats (board #55): synthetic ball + players, label-free."""
import numpy as np

from mtah.learn.stats import carrier_stats

PITCH = (105.0, 68.0)


def world(T=400):
    ball = np.zeros((T, 2)); ball[:, 0] = 20 + 2.0 * np.arange(T) / 25.0; ball[:, 1] = 30   # 2 m/s carry
    players = np.full((T, 4, 2), np.nan)
    players[:, 0] = ball + [0.3, 0.0]                      # carrier, always within 1 m
    players[:, 1] = [95.0, 30.0]; players[:, 2] = [80.0, 5.0]
    graded = np.ones(T, dtype=bool)
    return ball, players, graded


def test_tracked_carrier_is_never_absent():
    ball, players, graded = world()
    c = carrier_stats(players, ball, graded, PITCH)
    assert c["n_seen_inpitch"] == len(ball) and c["absent"] == 0.0 and c["carry_absent"] == 0.0
    assert c["n_kicks"] == 0 and c["kick_far"] is None


def test_hidden_carrier_shows_as_absent_and_far_kick():
    ball, players, graded = world()
    players[200:, 0] = np.nan                               # carrier lost for the second half
    ball[250:, 0] += 8.0                                    # a kick nobody tracked at frame 250
    c = carrier_stats(players, ball, graded, PITCH)
    assert abs(c["absent"] - 0.5) < 0.03
    assert c["carry_absent"] is not None and abs(c["carry_absent"] - 0.5) < 0.06
    assert c["n_kicks"] > 0 and c["kick_far"] == 1.0


def test_ball_outside_pitch_and_ungraded_frames_ignored():
    ball, players, graded = world()
    ball[:100, 0] = -3.0                                    # out of play
    graded[300:] = False
    c = carrier_stats(players, ball, graded, PITCH)
    assert c["n_seen_inpitch"] == 200
