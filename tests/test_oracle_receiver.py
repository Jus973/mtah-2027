import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from oracle_receiver import (CORRIDOR_FEATURES, FLIGHT_FEATURES,  # noqa: E402
                             corridor_features, flight_features)


def ev(team, start, end):
    return SimpleNamespace(team=team, start=start, end=end)


def col(X, name):
    return X[:, FLIGHT_FEATURES.index(name)]


def test_landing_is_the_frame_before_the_next_contact_and_toucher_relative():
    d = np.full((100, 2), 5.0)
    d[20:30] = [6.0, 0.5]  # the away team reaches the ball first
    X = flight_features([ev(0, 5, 8), ev(1, 25, 28)], d)
    assert col(X, "land_d_own")[0] == 6.0 and col(X, "land_d_opp")[0] == 0.5
    assert col(X, "land_margin")[0] == -5.5
    assert col(X, "fl_min_opp")[0] == 0.5 and col(X, "fl_own_nearer")[0] == 0.0
    np.testing.assert_allclose(col(X, "fl_log_s")[0], np.log1p(16 / 25))


def test_flight_is_capped_and_missing_ball_is_flagged():
    d = np.full((400, 2), np.nan)
    X = flight_features([ev(1, 0, 3)], d)
    np.testing.assert_allclose(col(X, "fl_log_s"), np.log1p(4.0))
    assert col(X, "land_missing")[0] == 1.0 and col(X, "land_margin")[0] == 0.0
    assert col(X, "fl_own_nearer")[0] == 0.5


def test_corridor_first_player_in_the_balls_path_and_slow_ball_is_invalid():
    ball = np.zeros((50, 2))
    ball[10:] = np.arange(40)[:, None] * np.array([0.4, 0.0])  # 10 m/s along +x from frame 10
    players = np.array([[[10.0, 1.0], [20.0, 0.5], [15.0, -6.0]]] * 50)
    tos = np.array([0, 1, 1])
    X = corridor_features([ev(0, 8, 10), ev(1, 0, 2)], ball, players, tos)
    row = dict(zip(CORRIDOR_FEATURES, X[0]))
    assert row["rc_first"] == 1.0 and row["rc_valid"] == 1.0
    assert row["rc_perp_own"] == 1.0 and row["rc_perp_opp"] == 0.5 and row["rc_margin"] == -0.5
    assert X[1, CORRIDOR_FEATURES.index("rc_valid")] == 0.0  # ball still before frame 10


def test_adjacent_contacts_have_no_flight_and_margins_stop_at_the_next_contact():
    d = np.full((100, 2), 5.0)
    d[9:] = [1.0, 9.0]  # the next contact's play must not leak into this one's flight
    X = flight_features([ev(0, 5, 8), ev(1, 9, 12), ev(0, 30, 31)], d)
    assert col(X, "fl_log_s")[0] == 0.0 and col(X, "land_missing")[0] == 1.0
    assert col(X, "m10_margin")[0] == 0.0 and col(X, "m25_margin")[0] == 0.0
    assert col(X, "m10_margin")[1] == -8.0 and col(X, "m25_margin")[1] == 0.0  # 37 > stop 29


def test_oracle_label_follows_the_label_policy():
    from oracle_receiver import oracle_label
    gt = np.array(list("A" * 100 + "D" * 10 + "H" * 20))
    graded = np.ones(len(gt), dtype=bool)
    e = [ev(0, 95, 100)]
    assert oracle_label(e, gt, graded, 5)[0] == 0.0
    assert oracle_label(e, gt, graded, 5, "alive_lag")[0] == 1.0
