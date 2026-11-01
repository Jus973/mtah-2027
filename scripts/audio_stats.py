"""make airlock T=audio: decode both videos' audio into per-frame features (cached outside the
repo) and report their aggregates (PLAN.md M2 T3). Decoding is local (PyAV); nothing leaves the
machine. Quantiles and shares only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

import games  # noqa: E402
from mtah import registry  # noqa: E402
from mtah.audio import features  # noqa: E402

args = game_args()
sp = registry.spec(args.game, args.data_dir)
for source in ("tactical", "broadcast"):
    n = sp["n_frames"] if source == "tactical" else games.SOURCES[source]["n_video_frames"]
    f = features.load(args.game, source, n, args.data_dir, sp["fps"])
    ok = np.isfinite(f["log_rms"])
    q = [5, 25, 50, 75, 95]
    put(source, {"audio_s": float(f["n_samples"]) / features.RATE,
                 "audio_minus_video_start_s": float(f["offset_s"]),
                 "frames_with_audio": float(ok.mean()),
                 "log_rms_q5_q95": [float(x) for x in np.percentile(f["log_rms"][ok], q)],
                 "band_db_q5_q95": [float(x) for x in np.percentile(f["band_db"][ok], q)],
                 "tonality_q5_q95": [float(x) for x in np.percentile(f["tonality"][ok], q)]})
