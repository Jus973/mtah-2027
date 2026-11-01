"""T3 tools on synthetic signals: audio frame features, audio and tracking sync, whistles."""

import numpy as np

from mtah.audio import whistle
from mtah.audio.features import RATE, frame_features
from mtah.sync import pointset
from mtah.sync.audio_xcorr import highpass, validate, window_offsets


def test_frame_features_place_a_tone_on_the_right_frames():
    t = np.arange(4 * RATE) / RATE
    x = 0.05 * np.random.default_rng(0).standard_normal(len(t))
    x[2 * RATE:3 * RATE] += 0.5 * np.sin(2 * np.pi * 3500 * t[2 * RATE:3 * RATE])
    f = frame_features(x.astype(np.float32), -0.5, 100)   # audio starts 0.5 s before the video
    tonal = np.flatnonzero(f["tonality"] > 20)
    assert tonal[0] == 37 and tonal[-1] == 62            # audio 2-3 s = video 1.5-2.5 s
    assert np.isfinite(f["log_rms"]).sum() == 87


def test_audio_xcorr_finds_the_offset():
    rng = np.random.default_rng(1)
    ref = highpass(rng.standard_normal(20000).cumsum() * 0.01 + rng.standard_normal(20000), 125)
    query = ref[5000:11000] + 0.3 * rng.standard_normal(6000)
    s, off, score, ratio = window_offsets(ref, query, 1500, 750, 50)
    assert (off == 5000).all() and (score > 0.9).all() and (ratio > 1.5).all()


def test_pointset_sync_and_validation():
    rng = np.random.default_rng(0)
    track = np.cumsum(rng.normal(0, 0.3, (20000, 2)), axis=0) + 50
    other = track[3000:9000] + rng.normal(0, 0.2, (6000, 2))
    other[rng.random(6000) < 0.4] = np.nan
    s, off, med, n = pointset.window_offsets(track, other, 1500, 750, 25, 250)
    assert (off == 3000).all() and (med < 0.5).all()
    agree, both = validate(track, other, 3000, np.arange(6000), 2.0)
    assert agree == 1.0 and both > 3000


def test_whistle_detection_and_shares():
    n = 1000
    feats = {"tonality": np.full(n, 10.0), "band_db": np.full(n, -20.0)}
    feats["tonality"][100:104] = feats["tonality"][500:501] = 80.0
    feats["band_db"][100:104] = feats["band_db"][500:501] = 0.0
    cfg = {"min_tonality": 40.0, "min_band_db_over_median": 10.0, "min_frames": 3}
    w = whistle.detect(feats, cfg)
    assert w.tolist() == [100]                         # the 1-frame blip is not a whistle
    assert whistle.near_share([110, 300], w, 25, 0) == 0.5
    assert abs(whistle.chance(w, n, np.ones(n, bool), 25, 0) - 26 / n) < 1e-12
