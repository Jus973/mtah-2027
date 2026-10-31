"""Parse the FIFA/EPTS ATD (metadata XML + raw text) into frame-indexed numpy arrays.

Raw line: "<frame_id>:<g>;<g>;...;<g>:<ball>", each group "x,y,speed", NaN for missing.
frame_id counts from 1; row t of every array here is tactical video frame t = frame_id - 1.

Floats are parsed with Python float() into float64 and kept normalised (0-1), so that anything
computed from this loader is bit-identical to starter/baseline.py, which does the same. Metres are
a view (Atd.players_m) rather than the stored representation.

Nothing in this module prints data. Callers that run on the real bundle print aggregates only.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from mtah import registry

HOME, AWAY = 0, 1
CACHE_VERSION = 1


@dataclass
class Atd:
    xy: np.ndarray            # (T, S, 2) float64, normalised, NaN where missing
    speed: np.ndarray         # (T, S) float64, m/s
    ball: np.ndarray          # (T, 2) float64, normalised
    ball_aux: np.ndarray      # (T,) float64, third field of the ball group (provider-defined)
    team_of_slot: np.ndarray  # (S,) int8, HOME=0 / AWAY=1
    team_names: tuple         # (home, away) as written in the metadata
    pitch: tuple              # (length, width) metres
    frame0: int = 0           # video frame of row 0

    @property
    def n_frames(self):
        return self.xy.shape[0]

    @property
    def n_slots(self):
        return self.xy.shape[1]

    def players_m(self):
        """(T, S, 3) float32: x, y in metres and speed in m/s (the §4 interface)."""
        length, width = self.pitch
        out = np.empty(self.xy.shape[:2] + (3,), dtype=np.float32)
        out[..., 0] = self.xy[..., 0] * length
        out[..., 1] = self.xy[..., 1] * width
        out[..., 2] = self.speed
        return out

    def ball_m(self):
        length, width = self.pitch
        return np.stack([self.ball[:, 0] * length, self.ball[:, 1] * width], axis=1)

    def ball_present(self):
        return ~np.isnan(self.ball[:, 0])

    def n_objects(self):
        """(T,) number of tracked player slots with a position."""
        return np.sum(~np.isnan(self.xy[..., 0]), axis=1)


def read_meta(meta_path):
    """(team_of_slot, (home_name, away_name)). First Team in the file is home (EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    team_of_slot = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    names = tuple(team.findtext("Name") for team in teams)
    return team_of_slot, names


def _group(text):
    """'x,y,s' -> (x, y, s); a bare 'NaN' or short group -> NaNs."""
    fields = text.split(",")
    if len(fields) < 3:
        nan = float("nan")
        values = [float(v) for v in fields] + [nan] * (3 - len(fields))
        return values[0], values[1], values[2]
    return float(fields[0]), float(fields[1]), float(fields[2])


def parse_raw(raw_path, n_frames, n_slots):
    """Stream the raw file. Returns (xy, speed, ball, ball_aux, stats) with stats aggregates only."""
    xy = np.full((n_frames, n_slots, 2), np.nan)
    speed = np.full((n_frames, n_slots), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    ball_aux = np.full(n_frames, np.nan)
    stats = {"n_lines": 0, "n_bad_lines": 0, "n_out_of_range": 0, "n_slot_mismatch": 0,
             "min_frame_id": None, "max_frame_id": None}

    with open(raw_path) as handle:
        for line in handle:
            stats["n_lines"] += 1
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                stats["n_bad_lines"] += 1
                continue
            frame_id = int(parts[0])
            if stats["min_frame_id"] is None or frame_id < stats["min_frame_id"]:
                stats["min_frame_id"] = frame_id
            if stats["max_frame_id"] is None or frame_id > stats["max_frame_id"]:
                stats["max_frame_id"] = frame_id
            t = frame_id - 1
            if not 0 <= t < n_frames:
                stats["n_out_of_range"] += 1
                continue
            groups = parts[1].split(";")
            if len(groups) != n_slots:
                stats["n_slot_mismatch"] += 1
            for s, group in enumerate(groups[:n_slots]):
                if group[0] == "N":
                    continue
                x, y, v = _group(group)
                xy[t, s, 0] = x
                xy[t, s, 1] = y
                speed[t, s] = v
            if parts[2][0] != "N":
                bx, by, ba = _group(parts[2])
                ball[t, 0] = bx
                ball[t, 1] = by
                ball_aux[t] = ba
    return xy, speed, ball, ball_aux, stats


def load_atd(meta_xml, raw_txt, pitch=(105.3, 68.0), n_frames=None):
    """Parse an ATD pair. n_frames defaults to the number of lines in the raw file."""
    team_of_slot, names = read_meta(meta_xml)
    if n_frames is None:
        with open(raw_txt) as handle:
            n_frames = sum(1 for _ in handle)
    xy, speed, ball, ball_aux, stats = parse_raw(raw_txt, n_frames, len(team_of_slot))
    atd = Atd(xy=xy, speed=speed, ball=ball, ball_aux=ball_aux, team_of_slot=team_of_slot,
              team_names=names, pitch=tuple(pitch))
    return atd, stats


def cache_path(game, source="tactical"):
    return os.path.join(registry.cache_dir(), "{}_{}_atd_v{}.npz".format(game, source, CACHE_VERSION))


def save_cache(atd, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, xy=atd.xy, speed=atd.speed, ball=atd.ball, ball_aux=atd.ball_aux,
                        team_of_slot=atd.team_of_slot, team_names=np.array(atd.team_names),
                        pitch=np.array(atd.pitch))


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    return Atd(xy=z["xy"], speed=z["speed"], ball=z["ball"], ball_aux=z["ball_aux"],
               team_of_slot=z["team_of_slot"], team_names=tuple(str(n) for n in z["team_names"]),
               pitch=tuple(float(p) for p in z["pitch"]))


def load_game(game, data_dir=None, source="tactical", use_cache=True):
    """Atd for a registered match (real or fake). Real matches are cached outside the repo."""
    registry.gate(game, data_dir)
    sp = registry.spec(game, data_dir)
    fake = sp["fake"]
    path = cache_path(game, source)
    if use_cache and not fake and os.path.isfile(path):
        return load_cache(path), None
    meta, raw = registry.require(["atd_meta", "atd_raw"], game, data_dir, source)
    n = sp["n_frames"] if source == "tactical" else None
    atd, stats = load_atd(meta, raw, pitch=sp["pitch"], n_frames=n)
    if use_cache and not fake:
        save_cache(atd, path)
    return atd, stats
