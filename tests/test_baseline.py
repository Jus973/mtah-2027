"""Our loader + vectorised nearest-player == the official starter on the same files."""

import numpy as np

import baseline as starter
from mtah import registry
from mtah.features.baseline import nearest_player_codes


def test_vectorised_matches_starter(fake, fake_atd):
    atd, _ = fake_atd
    sp = registry.spec(fake["game"], fake["data_dir"])
    meta, raw = registry.require(["atd_meta", "atd_raw"], fake["game"], fake["data_dir"])
    ref = starter.nearest_player_labels(raw, starter.team_of_each_track(meta), sp["n_frames"],
                                        sp["pitch"])
    ours = nearest_player_codes(atd)
    assert np.array_equal(ref, ours)
    for _, window in starter.VARIANTS:
        assert np.array_equal(starter.smooth(ref, window), starter.smooth(ours, window))
