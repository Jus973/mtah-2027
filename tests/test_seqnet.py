import os

import numpy as np

from mtah import registry
from mtah.learn import proxowner, seqnet


def _naive_conv(x, w, b, d):
    C, T = x.shape
    K = w.shape[2]
    out = np.tile(b[:, None], (1, T)).astype(float)
    for t in range(T):
        for j in range(K):
            s = t + (j - (K - 1) // 2) * d
            if 0 <= s < T:
                out[:, t] += w[:, :, j] @ x[:, s]
    return out


def test_conv_matches_naive_same_padding():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 40))
    for K, d in ((1, 1), (3, 1), (3, 4), (3, 64)):
        w, b = rng.normal(size=(6, 5, K)), rng.normal(size=6)
        assert np.allclose(seqnet._conv(x, w, b, d), _naive_conv(x, w, b, d))


def test_committed_model_is_team_symmetric_and_bounded():
    path = os.path.join(registry.ROOT, "models", "seqnet_v1.json")
    net = seqnet.SeqNet(path)
    rng = np.random.default_rng(1)
    T, S = 400, 6
    players = rng.uniform(0, 60, (T, S, 2))
    ball = rng.uniform(0, 60, (T, 2))
    ball[50:80] = np.nan
    team = np.array([0, 0, 0, 1, 1, 1])
    hdir = np.ones(T)
    a, b, names = proxowner.both_views(players, ball, team, hdir, (105.0, 68.0), 25, True)
    assert seqnet.feature_names(path) == list(names)
    p = net.p_home((a, b, names))
    q = net.p_home((b, a, names))
    assert p.shape == (T,) and np.all((p > 0) & (p < 1))
    assert np.allclose(p, 1 - q)
