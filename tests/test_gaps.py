"""Ball-free decode inside long ball gaps (decode/gaps.py), gap features and the v5 pipeline."""

import os

import numpy as np

from mtah import config, pipeline, registry
from mtah.decode.gaps import decode_gaps
from mtah.learn import gapstate

V5 = config.load("rules_v5.yaml")


def test_long_gaps():
    observed = np.ones(100, bool)
    observed[10:20] = observed[40:80] = False
    graded = np.ones(100, bool)
    graded[70:] = False
    gap = gapstate.long_gaps(observed, graded, 25)
    assert np.flatnonzero(gap).tolist() == list(range(40, 70))


def test_decode_gaps_follows_the_models_and_pays_for_changes():
    labels = np.array(list("H" * 10 + "D" * 100 + "A" * 10))
    gap = np.zeros(120, bool)
    gap[10:110] = True
    p_dead = np.full(120, 0.05)
    p_dead[60:110] = 0.95                     # alive, then dead
    p_home = np.full(120, 0.9)
    p_home[35:60] = 0.1                       # H, then A, inside the alive part
    out = decode_gaps(labels, gap, p_dead, p_home, step=5, lam_dead=2.0, lam_owner=2.0)
    assert "".join(out) == "H" * 35 + "A" * 25 + "D" * 50 + "A" * 10
    stiff = decode_gaps(labels, gap, p_dead, p_home, step=5, lam_dead=50.0, lam_owner=50.0)
    assert set(stiff[10:110]) == {"H"}        # changes too dear: H holds, pays once at the end
    np.testing.assert_array_equal(out[~gap], labels[~gap])


def test_gap_features_are_team_symmetric():
    rng = np.random.default_rng(0)
    T = 300
    players = rng.uniform([0, 0], [105, 68], size=(1, 8, 2)) + np.cumsum(
        rng.normal(0, 0.1, size=(T, 8, 2)), axis=0)
    players[50:60, 2] = np.nan
    team = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    direction = np.ones(T)
    gc = V5["gaps"]
    home, away = gapstate.features(players, team, direction, (105.0, 68.0), gc)
    s_home, s_away = gapstate.features(players, 1 - team, -direction, (105.0, 68.0), gc)
    assert home.shape == (T, len(gapstate.feature_names(gc)))
    np.testing.assert_allclose(s_home, away, equal_nan=True)
    np.testing.assert_allclose(s_away, home, equal_nan=True)


def test_training_rows():
    F = (np.arange(20.0)[:, None], -np.arange(20.0)[:, None])
    gt = np.array(list("HHHHHAAAAADDDDDHHHHH"))
    gap = np.zeros(20, bool)
    gap[2:14] = True
    rows = gapstate.training_rows(F, gap, gt, step=2)
    X, y = rows["dead"]
    assert X[:, 0].tolist() == [2, 4, 6, 8, 10, 12, -2, -4, -6, -8, -10, -12]
    assert y.tolist() == [0, 0, 0, 0, 1, 1] * 2
    X, y = rows["owner"]                      # alive gap frames 2..9, every 2nd
    assert X[:, 0].tolist() == [2, 4, 6, 8, -2, -4, -6, -8]
    assert y.tolist() == [1, 1, 0, 0, 0, 0, 1, 1]


def test_v5_pipeline_on_the_fake_match(fake, fake_atd):
    for path in gapstate.model_paths(V5).values():
        assert os.path.isfile(path)           # committed, trained on the sim only
    atd, _ = fake_atd
    sp = registry.spec(fake["game"], fake["data_dir"])
    a, info = pipeline.run(fake["game"], V5, fake["data_dir"], atd=atd)
    b, _ = pipeline.run(fake["game"], V5, fake["data_dir"], atd=atd)
    np.testing.assert_array_equal(a, b)
    graded = registry.graded_mask(sp)
    assert set(a[graded]) <= {"H", "A", "D"} and set(a[~graded]) == {"X"}
    assert info["gap_frames"] > 0
    (_, f2, _), k2 = sp["half_blocks"][1], info["kickoffs"][2]
    assert (a[f2:k2] == "D").all()            # before a kick-off the ball stays dead
