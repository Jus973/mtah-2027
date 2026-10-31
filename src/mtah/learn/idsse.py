"""DFL/IDSSE open data (Bassek et al., Sci Data 12, 195, 2025; CC BY) -> CleanMatch.

Positions XML: one <FrameSet> per (GameSection, object) with one <Frame> per line at 25 Hz, metres
with the origin at the pitch centre and X along the length. The ball FrameSet (TeamId="BALL")
carries BallStatus (1 = in play) and BallPossession (1 = home, 2 = guest) on every frame; DFL's
possession is human-produced data about *other* games, which RULES.md allows. Section frame
numbers N start at 10000 (first half) and 100000 (second half); the first frame of each section is
the kick-off. Some matches also track the referees (TeamId="referee"); only the two clubs are
kept, as in the real ATD. Parsed line by line and cached as npz beside the data.
"""

import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from mtah.learn import clean

OPEN_DIR = os.path.expanduser(os.environ.get("MTAH_OPEN_DIR", "~/opendata/idsse"))
SECTIONS = ("firstHalf", "secondHalf")
PAD = 125  # frames of nothing before, between and after the halves (5 s)

_SET = re.compile(r'<FrameSet\b')
_ATTR = {k: re.compile(r'\b{}="([^"]*)"'.format(k)) for k in
         ("GameSection", "TeamId", "PersonId", "N", "X", "Y", "BallPossession", "BallStatus")}


def files(match_id, open_dir=OPEN_DIR):
    def one(kind):
        hits = glob.glob(os.path.join(open_dir, "DFL_*_{}_*{}.xml".format(kind, match_id)))
        if len(hits) != 1:
            raise IOError("{}: expected one {} file in {}, found {}".format(match_id, kind, open_dir,
                                                                             len(hits)))
        return hits[0]
    return one("matchinformation"), one("positions_raw_observed")


def cache_path(match_id, open_dir=OPEN_DIR):
    return os.path.join(open_dir.rstrip(os.sep) + "_cache", "{}_clean_v2.npz".format(match_id))


def read_info(path):
    general = next(ET.parse(path).getroot().iter("General"))
    env = next(ET.parse(path).getroot().iter("Environment"))
    return {"home_id": general.get("HomeTeamId"), "guest_id": general.get("GuestTeamId"),
            "names": (general.get("HomeTeamName"), general.get("GuestTeamName")),
            "pitch": (float(env.get("PitchX")), float(env.get("PitchY")))}


def parse_positions(path):
    """{(section, object_id): (team_id, N int array, X, Y[, possession, status])} from the XML."""
    sets, current, rows = {}, None, None

    def close():
        if current is not None:
            sets[current[:2]] = (current[2], np.array(rows, dtype=np.float64))

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if current is None:
                    continue
                n, x, y = (_ATTR[k].search(line).group(1) for k in ("N", "X", "Y"))
                if current[2] == "BALL":
                    rows.append((int(n), float(x), float(y),
                                 int(_ATTR["BallPossession"].search(line).group(1)),
                                 int(_ATTR["BallStatus"].search(line).group(1))))
                else:
                    rows.append((int(n), float(x), float(y)))
            elif _SET.match(line):
                close()
                section = _ATTR["GameSection"].search(line).group(1)
                current = ((section, _ATTR["PersonId"].search(line).group(1),
                            _ATTR["TeamId"].search(line).group(1)) if section in SECTIONS else None)
                rows = []
    close()
    return sets


def build(match_id, open_dir=OPEN_DIR):
    info_path, pos_path = files(match_id, open_dir)
    info = read_info(info_path)
    sets = parse_positions(pos_path)
    length, width = info["pitch"]
    ball_sets = {sec: rows for (sec, _), (team, rows) in sets.items() if team == "BALL"}
    starts, lengths = {}, {}
    t = PAD
    for sec in SECTIONS:
        n = ball_sets[sec][:, 0]
        if not np.array_equal(n, np.arange(n[0], n[0] + len(n))):
            raise ValueError("{} {}: ball frames are not contiguous".format(match_id, sec))
        starts[sec], lengths[sec] = t, len(n)
        t += len(n) + PAD
    n_frames = t
    clubs = (info["home_id"], info["guest_id"])
    people = sorted({(team, pid) for (_, pid), (team, _) in sets.items() if team in clubs},
                    key=lambda tp: (tp[0] != info["home_id"], tp[1]))
    column = {pid: i for i, (_, pid) in enumerate(people)}
    players = np.full((n_frames, len(people), 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    alive = np.zeros(n_frames, dtype=bool)
    owner = np.full(n_frames, -1, dtype=np.int8)
    for (sec, pid), (team, rows) in sets.items():
        if sec not in starts or not len(rows) or (team != "BALL" and team not in clubs):
            continue
        n0 = ball_sets[sec][0, 0]
        idx = (rows[:, 0] - n0).astype(int) + starts[sec]
        keep = (idx >= starts[sec]) & (idx < starts[sec] + lengths[sec])
        xy = np.stack([rows[keep, 1] + length / 2, rows[keep, 2] + width / 2], axis=1)
        if team == "BALL":
            ball[idx[keep]] = xy
            alive[idx[keep]] = rows[keep, 4] == 1
            owner[idx[keep]] = np.where(rows[keep, 3] == 1, 0, np.where(rows[keep, 3] == 2, 1, -1))
        else:
            players[idx[keep], column[pid]] = xy
    halves = np.array([[starts[s], starts[s] + lengths[s] - 1] for s in SECTIONS])
    return clean.CleanMatch(
        name=match_id, fps=25, pitch=(length, width), players=players,
        team_of_player=np.array([0 if team == info["home_id"] else 1 for team, _ in people],
                                dtype=np.int8),
        ball=ball, alive=alive, owner=owner, halves=halves, team_names=info["names"])


def load(match_id, open_dir=OPEN_DIR, use_cache=True):
    path = cache_path(match_id, open_dir)
    if use_cache and os.path.isfile(path):
        return clean.load(path)
    match = build(match_id, open_dir)
    clean.save(match, path)
    return match
