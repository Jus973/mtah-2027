"""Metrica Sports open sample data, Sample Games 1 and 2 (CSV tracking + CSV events) -> CleanMatch.

The CSV counterpart of io/metrica.py (Sample Game 3, EPTS + JSON). Events are mapped onto the same
record shape ({type, subs, team, start, end, period}) and labelled by the same mechanical rules
(metrica.derive), so every label decision documented there applies unchanged. Differences of the
CSV release, all handled here:
  - subtypes are one hyphen-joined string ("ON TARGET-SAVED", "TACKLE-FAULT-WON"); split on "-"
    they give the JSON names ({"ON TARGET", "SAVED"}), which is all derive() reads;
  - there is no CARRY type: the owner flips only at SET PIECE / PASS / SHOT / RECOVERY;
  - tracking is two files (home, away), normalised (0,0) top-left, 25 Hz, ball in the last pair
    of columns, NaN where the source tracker has no position.
Files are fetched by scripts/fetch_metrica.py --games 1 2 into MTAH_METRICA_DIR (~/opendata/metrica).
Open data about other games; it never enters the repo.
"""

import os

import numpy as np
import pandas as pd

from mtah.io import metrica
from mtah.learn import clean

PITCH = (105.0, 68.0)


def paths(game, open_dir=metrica.OPEN_DIR):
    stem = os.path.join(open_dir, "Sample_Game_{}_".format(game))
    out = {"events": stem + "RawEventsData.csv", "home": stem + "RawTrackingData_Home_Team.csv",
           "away": stem + "RawTrackingData_Away_Team.csv"}
    missing = [p for p in out.values() if not os.path.isfile(p)]
    if missing:
        raise IOError("Metrica Sample Game {} not found:\n  {}\nRun `make metrica-data`."
                      .format(game, "\n  ".join(missing)))
    return out


def load_events(path):
    """Events in metrica.load_events' shape, sorted by (period, start)."""
    ev = pd.read_csv(path)
    out = []
    for r in ev.itertuples(index=False):
        sub = getattr(r, "Subtype")
        subs = frozenset() if not isinstance(sub, str) else frozenset(
            s.strip() for s in sub.split("-") if s.strip())
        out.append({"type": str(getattr(r, "Type")).strip(), "subs": subs,
                    "team": 0 if str(getattr(r, "Team")).strip() == "Home" else 1,
                    "start": int(getattr(r, "_4")), "end": int(getattr(r, "_6")),
                    "period": int(getattr(r, "Period"))})
    out.sort(key=lambda v: (v["period"], v["start"]))
    return out


def read_tracking(path):
    """(period (T,), frame (T,), players (T, P, 2) normalised, ball (T, 2) normalised)."""
    df = pd.read_csv(path, skiprows=2)
    cols = list(df.columns)
    bi = cols.index("Ball")
    xy = df.iloc[:, 3:bi].to_numpy(dtype=float)
    players = xy.reshape(len(df), -1, 2)
    ball = df.iloc[:, bi:bi + 2].to_numpy(dtype=float)
    return df["Period"].to_numpy(), df["Frame"].to_numpy(), players, ball


def build(game, open_dir=metrica.OPEN_DIR):
    p = paths(game, open_dir)
    events = load_events(p["events"])
    per_h, fr_h, home, ball = read_tracking(p["home"])
    per_a, fr_a, away, _ = read_tracking(p["away"])
    if not (np.array_equal(fr_h, fr_a) and np.array_equal(fr_h, np.arange(1, len(fr_h) + 1))):
        raise ValueError("game {}: tracking frames are not contiguous 1..T in both files".format(game))
    length, width = PITCH
    players = np.concatenate([home, away], axis=1) * np.array([length, width])
    ball = ball * np.array([length, width])
    team_of_player = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    half_end = {q: int(fr_h[per_h == q].max()) for q in (1, 2)}
    n_src = len(fr_h)
    alive_src, owner_src, _ = metrica.derive(events, half_end, n_src)
    kicks = metrica.kickoff_frames(events)
    if sorted(kicks) != [1, 2]:
        raise ValueError("kick-off events not found for both periods: {}".format(sorted(kicks)))
    PAD = metrica.PAD
    starts, lengths, t = {}, {}, PAD
    for q in (1, 2):
        lengths[q] = half_end[q] - kicks[q] + 1
        starts[q] = t
        t += lengths[q] + PAD
    out_players = np.full((t, players.shape[1], 2), np.nan)
    out_ball = np.full((t, 2), np.nan)
    alive = np.zeros(t, dtype=bool)
    owner = np.full(t, -1, dtype=np.int8)
    for q in (1, 2):
        a, b, s = kicks[q] - 1, half_end[q], starts[q]
        out_players[s:s + lengths[q]] = players[a:b]
        out_ball[s:s + lengths[q]] = ball[a:b]
        alive[s:s + lengths[q]] = alive_src[a:b]
        owner[s:s + lengths[q]] = owner_src[a:b]
        if not alive[s] or owner[s] < 0:
            raise ValueError("half {} does not start alive with a known owner".format(q))
    halves = np.array([[starts[q], starts[q] + lengths[q] - 1] for q in (1, 2)])
    return clean.CleanMatch(name="G{}".format(game), fps=metrica.FPS, pitch=PITCH,
                            players=out_players, team_of_player=team_of_player, ball=out_ball,
                            alive=alive, owner=owner, halves=halves, team_names=("Home", "Away"))


def load(game, open_dir=metrica.OPEN_DIR, use_cache=True):
    path = metrica.cache_path("G{}".format(game), open_dir)
    if use_cache and os.path.isfile(path):
        return clean.load(path)
    match = build(game, open_dir)
    clean.save(match, path)
    return match
