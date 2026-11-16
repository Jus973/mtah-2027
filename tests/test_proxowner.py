import numpy as np

from mtah.learn import proxowner


def test_smooth_ball_only_inside_seen_runs():
    ball = np.array([[0.0, 0.0], [3.0, 3.0], [np.nan, np.nan], [6.0, 0.0], [6.0, 0.0], [9.0, 3.0]])
    out = proxowner.smooth_ball(ball, 3)
    assert np.isnan(out[2]).all()
    assert np.allclose(out[0], [1.5, 1.5]) and np.allclose(out[1], [1.5, 1.5])
    assert np.allclose(out[4], [7.0, 1.0])
    assert proxowner.smooth_ball(ball, 1) is ball


def test_blend_decode_keeps_dead_and_x_and_follows_strong_evidence():
    labels = np.array(list("XHHHHDDAAAAX"))
    p = np.array([0.5, 0.9, 0.9, 0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.5])
    out = proxowner.blend_decode(labels, p, 0.0, 0.0)
    assert out[0] == "X" and out[-1] == "X" and (out[5:7] == "D").all()
    assert out[1:5].tolist() == ["H", "H", "A", "A"] and (out[7:11] == "H").all()
    assert (proxowner.blend_decode(labels, p, 50.0, 0.0) == labels).all()   # v11 dominates
    assert set(proxowner.blend_decode(labels, p, 0.0, 1e6)[1:5]) <= {"H", "A"}


def test_setting_switches_on_kick_rate():
    cfg = {"proxowner": {"kick_threshold": 18.0, "kicky": [2.0, 5.0], "smooth": [0.5, 5.0]}}
    assert proxowner.setting(cfg, 25.0) == [2.0, 5.0]
    assert proxowner.setting(cfg, 2.6) == [0.5, 5.0]


def test_features_are_team_symmetric():
    rng = np.random.default_rng(0)
    T, S = 300, 4
    players = rng.uniform(0, 60, (T, S, 2))
    ball = rng.uniform(0, 60, (T, 2))
    team = np.array([0, 0, 1, 1])
    hdir = np.ones(T)
    a, b, names = proxowner.both_views(players, ball, team, hdir, (105.0, 68.0), 25)
    i, j = names.index("d1h"), names.index("d1a")
    assert np.allclose(a[:, i], b[:, j], equal_nan=True) and np.allclose(a[:, j], b[:, i], equal_nan=True)
    k = names.index("diff")
    assert np.allclose(a[:, k], -b[:, k], equal_nan=True)


def test_turn_features_flag_and_symmetry():
    """Board #84: turn features only appear with the flag (old models keep their columns), and a
    turn's attributed margin flips sign between the two team views."""
    rng = np.random.default_rng(1)
    T, S = 400, 4
    players = rng.uniform(0, 60, (T, S, 2))
    t = np.arange(T)[:, None]
    ball = np.where(t < 200, [10.0, 30.0] + 0.4 * t * np.array([1.0, 0.0]),
                    [90.0, 30.0] + 0.4 * (t - 200) * np.array([0.0, 1.0]))   # one sharp turn
    team = np.array([0, 0, 1, 1])
    hdir = np.ones(T)
    _, _, off = proxowner.both_views(players, ball, team, hdir, (105.0, 68.0), 25)
    a, b, on = proxowner.both_views(players, ball, team, hdir, (105.0, 68.0), 25, turns=True)
    assert not any("turn" in n for n in off)
    assert set(off) < set(on) and "lastturn25_4" in on
    k = on.index("lastturn25_4")
    late = np.isfinite(a[:, k])
    assert late[250:].all() and not late[:150].any()
    assert np.allclose(a[late, k], -b[late, k])
