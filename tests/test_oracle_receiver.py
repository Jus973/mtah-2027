import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from oracle_receiver import (CORRIDOR_FEATURES, FLIGHT_FEATURES,  # noqa: E402
                             KINEMATIC_FEATURES, corridor_features, flight_features,
                             kinematic_features)


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


def test_kinematic_a_player_running_at_the_path_beats_a_standing_one_at_the_same_spot():
    ball = np.zeros((60, 2))
    ball[10:] = np.arange(50)[:, None] * np.array([0.4, 0.0])  # 10 m/s along +x from frame 10
    players = np.array([[[20.0, 8.0], [20.0, 8.0]]] * 60)
    players[:11, 1] = np.linspace(11.0, 8.0, 11)[:, None] * np.array([0.0, 1.0]) + [20.0, 0.0]
    X = kinematic_features([ev(0, 8, 10)], ball, players, np.array([0, 1]))
    row = dict(zip(KINEMATIC_FEATURES, X[0]))
    assert row["kn_valid"] == 1.0 and row["kn_first"] == -1.0
    assert row["kn_t_opp"] < row["kn_t_own"] and row["kn_margin"] < 0


def test_kinematic_is_invalid_for_a_slow_ball_or_an_immediate_next_contact():
    ball = np.zeros((60, 2))
    ball[10:] = np.arange(50)[:, None] * np.array([0.4, 0.0])
    players = np.array([[[20.0, 1.0], [30.0, 1.0]]] * 60)
    tos = np.array([0, 1])
    X = kinematic_features([ev(0, 0, 6), ev(1, 7, 7), ev(0, 8, 10), ev(1, 11, 12)], ball,
                           players, tos)
    valid = X[:, KINEMATIC_FEATURES.index("kn_valid")]
    assert valid[0] == 0.0  # ball still at rest
    assert valid[2] == 0.0  # next contact a frame later: no flight to extrapolate
    assert (X[valid == 0, :2] == 3.5).all()


def test_kinematic_intercepts_stop_at_the_next_contact_and_missing_velocity_means_standing():
    ball = np.zeros((80, 2))
    ball[10:] = np.arange(70)[:, None] * np.array([0.4, 0.0])
    players = np.array([[[25.0, 0.5], [8.0, 0.5]]] * 80)  # own far along the path, opp near
    players[5, 1] = np.nan  # opp's track starts inside the look-back
    tos = np.array([0, 1])
    far = kinematic_features([ev(0, 8, 10)], ball, players, tos)
    row = dict(zip(KINEMATIC_FEATURES, far[0]))
    assert row["kn_t_own"] < 3.5 and row["kn_t_opp"] < 3.5  # the standing opp still intercepts
    soon = kinematic_features([ev(0, 8, 10), ev(1, 30, 31)], ball, players, tos)
    row = dict(zip(KINEMATIC_FEATURES, soon[0]))
    assert row["kn_t_own"] == 3.5  # own's intercept lies past the next contact (0.8 s)
    assert row["kn_t_opp"] < 0.8 and row["kn_first"] == -1.0
