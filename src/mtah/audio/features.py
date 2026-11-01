"""Per-video-frame audio features (PLAN.md M2 T3), decoded locally with PyAV; nothing leaves the
machine.

For each video frame i (the 40 ms of audio from i / fps on the video's own clock): the log RMS
loudness, and the energy and tonality (peak-to-mean power) of the 2-4.5 kHz band where referee
whistles sit. The envelope drives broadcast <-> tactical sync (sync/audio_xcorr.py), the band
features the whistle detector (audio/whistle.py). Real-match features are bundle-derived and are
cached beside the ATD cache, outside the repo.
"""

import os

import numpy as np

from mtah import registry

RATE = 16000
CACHE_VERSION = 1


def decode_mono(path, rate=RATE):
    """(samples float32, start offset of audio relative to video, seconds)."""
    import av
    with av.open(path) as container:
        audio = container.streams.audio[0]
        video = container.streams.video[0] if container.streams.video else None

        def start(stream):
            if stream is None or stream.start_time is None:
                return 0.0
            return float(stream.start_time * stream.time_base)

        offset = start(audio) - start(video)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks = []
        for frame in container.decode(audio):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32), offset


def frame_features(samples, offset_s, n_frames, fps=25, rate=RATE, band=(2000.0, 4500.0),
                   block=20000):
    """{"log_rms", "band_db", "tonality"}: (n_frames,) float32 arrays (NaN where no audio)."""
    hop = rate // fps
    first = int(round(-offset_s * rate))       # sample index of video frame 0
    freqs = np.fft.rfftfreq(hop, 1.0 / rate)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    window = np.hanning(hop).astype(np.float32)
    out = {k: np.full(n_frames, np.nan, dtype=np.float32) for k in ("log_rms", "band_db", "tonality")}
    for f0 in range(0, n_frames, block):
        f1 = min(n_frames, f0 + block)
        lo, hi = first + f0 * hop, first + f1 * hop
        a, b = max(0, lo), min(len(samples), hi)
        if b <= a:
            continue
        seg = np.zeros(hi - lo, dtype=np.float32)
        seg[a - lo:b - lo] = samples[a:b]
        frames = seg.reshape(f1 - f0, hop)
        valid = np.zeros(f1 - f0, dtype=bool)
        valid[max(0, (a - lo) // hop):(b - lo) // hop] = True
        rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9
        power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
        bp = power[:, in_band]
        out["log_rms"][f0:f1] = np.where(valid, np.log10(rms), np.nan)
        out["band_db"][f0:f1] = np.where(valid, 10 * np.log10(bp.sum(axis=1) + 1e-12), np.nan)
        out["tonality"][f0:f1] = np.where(valid, bp.max(axis=1) / (bp.mean(axis=1) + 1e-12),
                                          np.nan)
    return out


def cache_path(game, source):
    return os.path.join(registry.cache_dir(), "{}_{}_audio_v{}.npz".format(game, source,
                                                                          CACHE_VERSION))


def load(game, source, n_frames, data_dir=None, fps=25):
    """Cached frame features for one source's video (decoded on first use)."""
    path = cache_path(game, source)
    if os.path.isfile(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    video = registry.require(["video"], game, data_dir, source)[0]
    samples, offset = decode_mono(video)
    feats = frame_features(samples, offset, n_frames, fps)
    feats["offset_s"] = np.array(offset)
    feats["n_samples"] = np.array(len(samples))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **feats)
    return feats
