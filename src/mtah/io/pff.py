"""PFF FC World Cup 2022 broadcast tracking -> CleanMatch (evaluation-only validation family).

Source: PFF's public WC2022 release (Metadata/<id>.json, Rosters/<id>.json, Tracking Data/<id>.jsonl.bz2)
under MTAH_PFF_DIR (default ~/opendata/pff), never inside the repo. Mechanical labels from the
game_event stream embedded in the tracking frames, the kloppy PFF convention: OUT / END make the
ball dead; a kick-off or an on-the-ball (OTB) event makes it alive; the owner is the team of the
last event's home_ball flag. In-pitch stoppages (fouls, offsides) emit no OUT, so the ball stays
alive through them: this GT is fit for paired candidate signs only, never for calibrating dead-ball
detection or spell-length distributions. Halves 1-2 only, resampled from the source rate (29.97 fps) to 25 fps
by nearest earlier source frame on the period clock; coordinates moved to a pitch-corner origin.
"""

import bz2
import json
import os

import numpy as np

from mtah.learn import clean

OPEN_DIR = os.path.expanduser(os.environ.get("MTAH_PFF_DIR", "~/opendata/pff"))
FPS = 25
PAD = 125
DEAD_EVENTS = ("OUT", "END")
KICKOFF_PERIOD = {"FIRSTKICKOFF": 1, "SECONDKICKOFF": 2}
ALIVE_EVENTS = ("FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB")


def paths(game, open_dir=OPEN_DIR):
    return {"meta": os.path.join(open_dir, "Metadata", "{}.json".format(game)),
            "roster": os.path.join(open_dir, "Rosters", "{}.json".format(game)),
            "tracking": os.path.join(open_dir, "Tracking Data", "{}.jsonl.bz2".format(game))}


def games(open_dir=OPEN_DIR):
    d = os.path.join(open_dir, "Tracking Data")
    if not os.path.isdir(d):
        return []
    return sorted(int(f.split(".")[0]) for f in os.listdir(d) if f.endswith(".jsonl.bz2"))


def _xy(objs, index):
    out = np.full((len(index), 2), np.nan)
    for o in objs or ():
        k = index.get(str(o.get("jerseyNum")))
        if k is not None and o.get("x") is not None:
            out[k] = (o["x"], o["y"])
    return out


def parse(meta, roster, lines):
    """CleanMatch from parsed metadata, roster list and an iterable of tracking jsonl lines."""
    meta = meta[0] if isinstance(meta, list) else meta
    pitch = meta["stadium"]["pitches"][0]
    length, width = float(pitch["length"]), float(pitch["width"])
    home_id, away_id = str(meta["homeTeam"]["id"]), str(meta["awayTeam"]["id"])
    jerseys = {t: sorted({str(r["shirtNumber"]) for r in roster if str(r["team"]["id"]) == t},
                         key=lambda j: int(j)) for t in (home_id, away_id)}
    idx_h = {j: i for i, j in enumerate(jerseys[home_id])}
    idx_a = {j: i for i, j in enumerate(jerseys[away_id])}
    nh, na = len(idx_h), len(idx_a)
    rows = {1: [], 2: []}
    period = None
    # The alive/owner state is a running fold over events, so it must see them in match time,
    # not file order: frames are sorted by (period, clock); a frame without a clock keeps the
    # place of the timed frame before it. For a file already in time order this is the identity.
    frames, key = [], (1, -np.inf)
    for line in lines:
        f = json.loads(line) if isinstance(line, (str, bytes)) else line
        q = f.get("period")
        event_type = (f.get("game_event") or {}).get("game_event_type")
        if q is None:
            q = KICKOFF_PERIOD.get(event_type)
        if q is not None:
            clock = f.get("periodElapsedTime")
            if clock is not None:
                key = (q, float(clock))
            elif key[0] != q or event_type in KICKOFF_PERIOD:
                key = (q, -np.inf)
        frames.append((key, len(frames), f))
    frames.sort(key=lambda x: (x[0], x[1]))
    for (q, _), _, f in frames:
        if q not in rows:
            continue
        if period != q:
            period = q
            alive, owner = False, -1
        ge = f.get("game_event")
        if ge:
            kind = ge.get("game_event_type")
            if kind in DEAD_EVENTS:
                alive = False
            elif kind in ALIVE_EVENTS:
                alive = True
            if ge.get("home_ball") is not None:
                owner = 0 if ge["home_ball"] else 1
        q = f.get("period")
        if q not in rows or f.get("periodElapsedTime") is None:
            continue
        hp = f.get("homePlayersSmoothed") or f.get("homePlayers")
        ap = f.get("awayPlayersSmoothed") or f.get("awayPlayers")
        b = f.get("ballsSmoothed")
        if not b or b.get("x") is None:
            bl = f.get("balls") or []
            b = bl[0] if bl and bl[0].get("x") is not None else None
        ball = (b["x"], b["y"]) if b else (np.nan, np.nan)
        rows[q].append((float(f["periodElapsedTime"]), np.vstack([_xy(hp, idx_h), _xy(ap, idx_a)]),
                        ball, alive, owner))
    for q in (1, 2):
        if not rows[q]:
            raise ValueError("period {} has no tracking frames".format(q))
    halves_out, t = [], PAD
    parts = []
    for q in (1, 2):
        r = rows[q]
        src_t = np.array([x[0] for x in r])
        order = np.argsort(src_t, kind="stable")
        src_t = src_t[order]
        tt = np.arange(0.0, src_t[-1] + 1e-9, 1.0 / FPS) + src_t[0]
        pick = order[np.clip(np.searchsorted(src_t, tt, side="right") - 1, 0, len(r) - 1)]
        parts.append((t, pick, r))
        halves_out.append([t, t + len(pick) - 1])
        t += len(pick) + PAD
    T, P = t, nh + na
    players = np.full((T, P, 2), np.nan)
    ball = np.full((T, 2), np.nan)
    al = np.zeros(T, dtype=bool)
    ow = np.full(T, -1, dtype=np.int8)
    shift = np.array([length / 2, width / 2])
    for s, pick, r in parts:
        n = len(pick)
        players[s:s + n] = np.stack([r[i][1] for i in pick]) + shift
        ball[s:s + n] = np.array([r[i][2] for i in pick], dtype=float) + shift
        al[s:s + n] = [r[i][3] for i in pick]
        ow[s:s + n] = [r[i][4] for i in pick]
        al[s] = True                          # each half starts at its kick-off
        if ow[s] < 0:
            raise ValueError("half starting at {} has no known owner".format(s))
    team_of_player = np.array([0] * nh + [1] * na, dtype=np.int8)
    return clean.CleanMatch(name=str(meta["id"]), fps=FPS, pitch=(length, width), players=players,
                            team_of_player=team_of_player, ball=ball, alive=al, owner=ow,
                            halves=np.array(halves_out),
                            team_names=(meta["homeTeam"]["name"], meta["awayTeam"]["name"]))


def cache_path(game, open_dir=OPEN_DIR):
    return os.path.join(open_dir, "cache", "PFF-v2-{}.npz".format(game))


def load(game, open_dir=OPEN_DIR, use_cache=True):
    path = cache_path(game, open_dir)
    if use_cache and os.path.isfile(path):
        return clean.load(path)
    p = paths(game, open_dir)
    with open(p["meta"]) as fm, open(p["roster"]) as fr, bz2.open(p["tracking"], "rt") as ft:
        match = parse(json.load(fm), json.load(fr), ft)
    clean.save(match, path)
    return match
