"""Sparse-transition decoder (decode/viterbi.py), takeover features and the v3 pipeline."""

import os

import numpy as np

from mtah import config, pipeline, registry
from mtah.decode.viterbi import decode_viterbi
from mtah.events.touches import Touch
from mtah.features.players import team_distances
from mtah.learn import takeover

V3 = config.load("rules_v3.yaml")


def contact(start, team, n=3):
    return Touch(start=start, end=start + n - 1, team=team, n_frames=n, d_min=0.5,
                 speed_before=5.0, speed_after=5.0, turn_deg=30.0)


def test_viterbi_takes_likely_contacts_and_skips_a_lone_deflection():
    graded = np.ones(300, bool)
    dead = np.zeros(300, bool)
    events = [contact(10, 0), contact(50, 1), contact(60, 1), contact(120, 0), contact(130, 0)]
    p = np.array([0.5, 0.9, 0.5, 0.1, 0.9])
    lab = decode_viterbi(events, p, graded, dead, alpha=1.0, flip_penalty=0.0)
    # A takes at 50 (p 0.9). H's unlikely touch at 120 is not a takeover, its next one at 130 is.
    assert set(lab[:50]) == {"H"} and set(lab[50:130]) == {"A"} and set(lab[130:]) == {"H"}
    never = decode_viterbi(events, p, graded, dead, alpha=1.0, flip_penalty=10.0)
    assert set(never) == {"H"}


def test_viterbi_dead_stretches_and_halves():
    graded = np.zeros(400, bool)
    graded[10:200] = graded[250:400] = True
    dead = np.zeros(400, bool)
    dead[100:120] = True
    events = [contact(20, 1), contact(130, 0), contact(300, 1)]
    lab = decode_viterbi(events, np.full(3, 0.2), graded, dead, alpha=1.0, flip_penalty=0.0)
    assert set(lab[:10]) == {"X"} and set(lab[200:250]) == {"X"}
    assert set(lab[10:100]) == {"A"} and set(lab[100:120]) == {"D"}
    assert set(lab[120:200]) == {"H"}          # after D, the first contact decides
    assert set(lab[250:]) == {"A"}             # a new half starts from its first contact


def _world(T=200):
    """Six drifting players; the ball swings around player 0 (team 0), then sits at player 3."""
    rng = np.random.default_rng(0)
    players = rng.uniform([30, 20], [70, 50], size=(1, 6, 2)) + 0.05 * np.arange(T)[:, None, None]
    ball = players[:, 0] + np.column_stack([np.sin(np.arange(T) / 10), np.zeros(T)])
    ball[60:100] = players[60:100, 3] + 0.3
    return players, ball


def test_features_are_team_symmetric():
    players, ball = _world()
    team = np.array([0, 0, 0, 1, 1, 1])
    d, _ = team_distances(players, ball, team)
    swapped = 1 - team
    d_s, _ = team_distances(players, ball, swapped)
    ev = takeover.contacts(ball, d, V3["touch"], 1.0)
    ev_s = takeover.contacts(ball, d_s, V3["touch"], 1.0)
    assert len(ev) > 1 and [e.team for e in ev] == [1 - e.team for e in ev_s]
    np.testing.assert_array_equal(takeover.features(ev, d), takeover.features(ev_s, d_s))


def test_training_rows():
    gt = np.array(list("HHHHHAAAAAAAAAAD"))
    graded = np.ones(len(gt), bool)
    events = [contact(1, 0), contact(5, 1), contact(9, 0)]   # H keeps, A takes, H does not
    idx, y = takeover.training_rows(events, gt, graded, lag=2)
    assert idx.tolist() == [1, 2] and y.tolist() == [1.0, 0.0]


def test_v3_pipeline_on_the_fake_match(fake, fake_atd):
    assert os.path.isfile(takeover.model_path(V3))           # committed, trained on the sim only
    atd, _ = fake_atd
    sp = registry.spec(fake["game"], fake["data_dir"])
    a, info = pipeline.run(fake["game"], V3, fake["data_dir"], atd=atd)
    b, _ = pipeline.run(fake["game"], V3, fake["data_dir"], atd=atd)
    np.testing.assert_array_equal(a, b)
    graded = registry.graded_mask(sp)
    assert set(a[graded]) <= {"H", "A", "D"} and set(a[~graded]) == {"X"}
    assert info["contacts"] > info["touches"] > 0


def test_shape_features_follow_the_toucher():
    from mtah.learn import gapstate
    gc = config.load("rules_v6.yaml")["gaps"]
    names = gapstate.feature_names(gc)
    rng = np.random.default_rng(1)
    F = (rng.normal(size=(200, len(names))), rng.normal(size=(200, len(names))))
    events = [contact(20, 0), contact(80, 1), contact(150, 0)]
    swapped = [contact(20, 1), contact(80, 0), contact(150, 1)]
    a = takeover.shape_features(events, F, names)
    b = takeover.shape_features(swapped, (F[1], F[0]), names)
    assert a.shape == (3, len(takeover.SHAPE_FEATURES))
    np.testing.assert_array_equal(a, b)
    col = {n: i for i, n in enumerate(names)}
    assert a[1, -2] == F[1][82, col["x_cu_+50"]] - F[1][80, col["x_cu_-50"]]


def test_v6_pipeline_on_the_fake_match(fake, fake_atd):
    v6 = config.load("rules_v6.yaml")
    assert os.path.isfile(takeover.model_path(v6))           # committed, trained on the sim only
    atd, _ = fake_atd
    sp = registry.spec(fake["game"], fake["data_dir"])
    a, info = pipeline.run(fake["game"], v6, fake["data_dir"], atd=atd)
    graded = registry.graded_mask(sp)
    assert set(a[graded]) <= {"H", "A", "D"} and set(a[~graded]) == {"X"}
    assert info["contacts"] > 0
