"""Match registry: the vendored official games.py for real matches, plus synthetic ones.

games.py is the official file and stays unmodified. This module wraps it so that the whole pipeline
asks one place for "how long is this match, where are its halves, where are its files" and gets the
same answer shape for BAR-ATH and for a fakedata/ match.

A synthetic match is a folder written by fakedata/generate.py with the same layout as the real
bundle (tactical/<GAME>_tactical_atd.xml, ...) plus a fake_meta.json describing its timeline.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import games  # noqa: E402  (vendored, unmodified)

FAKE_META = "fake_meta.json"
SUFFIXES = games.SUFFIXES
# Degraded open-data matches (learn/degrade.py): large, so outside the repo.
SIM_DIR = os.path.expanduser(os.environ.get("MTAH_SIM_DIR", "~/opendata/sim"))


def data_dir_for(game, data_dir=None):
    """Bundle root. Real matches default to games.DATA_DIR ($MTAH_DATA_DIR or ./data); synthetic
    ones live in fakedata/out/<game> (fake) or SIM_DIR/<game> (degraded open data)."""
    if data_dir:
        return data_dir
    if game in games.GAMES:
        return games.DATA_DIR
    fake = os.path.join(ROOT, "fakedata", "out", game)
    sim = os.path.join(SIM_DIR, game)
    return sim if not os.path.isdir(fake) and os.path.isdir(sim) else fake


def is_fake(game, data_dir=None):
    return os.path.isfile(os.path.join(data_dir_for(game, data_dir), FAKE_META))


def spec(game, data_dir=None):
    """dict(n_frames, fps, pitch, half_blocks=[(half, first, last)], source_frames) for any match.

    half_blocks are inclusive video-frame ranges of the graded halves, exactly what
    games.half_video_blocks returns for the real match.
    """
    root = data_dir_for(game, data_dir)
    meta_path = os.path.join(root, FAKE_META)
    if os.path.isfile(meta_path):
        with open(meta_path) as handle:
            meta = json.load(handle)
        return {
            "game": game,
            "n_frames": int(meta["n_frames"]),
            "fps": int(meta["fps"]),
            "pitch": tuple(meta["pitch"]),
            "half_blocks": [tuple(b) for b in meta["half_blocks"]],
            "fake": True,
        }
    entry = games.GAMES[game]
    return {
        "game": game,
        "n_frames": int(entry["n_video_frames"]),
        "fps": int(entry["fps"]),
        "pitch": tuple(entry["pitch"]),
        "half_blocks": [tuple(b) for b in games.half_video_blocks(game)],
        "fake": False,
    }


def asset_paths(game, data_dir=None, source="tactical"):
    """Same keys as games.asset_paths, for real or synthetic matches. Not checked for existence."""
    root = data_dir_for(game, data_dir)
    if game in games.GAMES and not is_fake(game, data_dir):
        return games.asset_paths(game, root, source)
    assets = games.SOURCES[source]["assets"]
    folder = os.path.join(root, source)
    stem = "{}_{}".format(game, source)
    return {name: os.path.join(folder, stem + SUFFIXES[name]) for name in assets}


def gate(game, data_dir=None):
    """Refuse the real bundle unless a human is at a terminal or the airlock is collecting output.

    An agent's shell has no terminal, so its only way in is scripts/airlock.py, which validates
    what comes back (mtah.airlock). Synthetic matches are never gated.
    """
    from mtah.airlock import OUT_ENV
    if is_fake(game, data_dir) or os.environ.get(OUT_ENV) or sys.stdout.isatty():
        return
    raise PermissionError("the real match bundle opens only in a terminal (a human) or through "
                          "the airlock: make airlock T=<target>")


def require(names, game, data_dir=None, source="tactical"):
    gate(game, data_dir)
    paths = asset_paths(game, data_dir, source)
    missing = [paths[n] for n in names if not os.path.isfile(paths[n])]
    if missing:
        raise IOError("match data not found:\n  " + "\n  ".join(missing)
                      + "\nSet MTAH_DATA_DIR, or run `make fake` for a synthetic match.")
    return [paths[n] for n in names]


def graded_mask(sp):
    """(T,) bool, True on graded frames."""
    import numpy as np
    mask = np.zeros(sp["n_frames"], dtype=bool)
    for _, first, last in sp["half_blocks"]:
        mask[first:last + 1] = True
    return mask


def cache_dir():
    """Where derived real-data caches live: never inside the repo when MTAH_DATA_DIR is set."""
    explicit = os.environ.get("MTAH_CACHE_DIR")
    if explicit:
        return explicit
    data = os.environ.get("MTAH_DATA_DIR")
    if data:
        return os.path.join(os.path.dirname(os.path.abspath(data)),
                            os.path.basename(os.path.abspath(data)) + "_cache")
    return os.path.join(ROOT, "cache")
