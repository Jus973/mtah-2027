"""Metrica Sports open sample data, Sample Game 3 (EPTS tracking + JSON events) -> CleanMatch.

Second validation family (handoff/REPLY.md P5): same vendor as the competition, but labels are
derived MECHANICALLY from the published events per the official GT definition (PLAN.md section 1:
possession changes only at controlled touches by the other team; touches, deflections and unwon
aerials are NOT changes; loose ball stays with the last owner; D between out-of-play and the
restart; no smoothing). Fetched by scripts/fetch_metrica.py (pinned commit, sha256-verified, no
formal LICENSE - README terms only). Tracking parses with the pinned kloppy 3.18.0 EPTS reader
(the raw format has 11 DataFormatSpecifications - the channel layout changes at substitutions -
so io/atd.py's fixed x,y,speed grammar cannot read it); events parse as plain JSON.

Facts of the source (verified 1 Nov 2026): 25 Hz, pitch 105 x 68 m, normalised coordinates with
(0,0) top-left; 143,761 frames (first half 1-69661, second 69662-143761 per the metadata provider
parameters); 35 players (17 "Team A" = FIFATMA, home by EPTS order; 18 "Team B" = FIFATMB); 3,620
events, every one with team, period and 1-based start/end frames on the tracking clock, sorted by
start. Positions become metres as (x * 105, y * 68): no y flip, "origin at a pitch corner" like
the ATD convention, and the real ATD is Metrica-produced so its y sense is the same.

Halves START at the event-derived kick-off (the first SET PIECE + KICK OFF of each period: frames
361 and 70090), not at the tracking section start: gapstate.home_direction reads team centroids
from the half's first frames and asserts the two halves attack opposite ways, and
learn/degrade.py write_sim takes the kick-off owner from owner[half_start] (an unknown owner
would silently become "H"). Halves END at the tracking section end (69661 / 143761); the frames
after the END HALF whistle stay graded as D, as in the DFL sims. Timeline layout is
[125-frame pad][half 1][pad][half 2][pad], mapping 1:1 onto the frames learn/degrade.py writes.

Event -> label mapping (one decision per line; counts are Game 3; see also
~/opendata/metrica (documentation/events-definitions.pdf in the source repo)):

Owner. The owner changes at the START frame of every CONTROLLED touch event of the other team,
and nowhere else (loose/contested ball stays with the last owner):
  - SET PIECE (90): the restart taker's team takes the ball in play. Controlled.
  - PASS (1121): a completed pass between teammates, goal kicks included. Controlled.
  - CARRY (1395): the player moves with the ball. Controlled. Metrica also writes 1-2 frame glue
    CARRYs after receptions; the only ones that flip the owner are after a parried save (counted
    by the census as carry_glue_flips, 12 frames in Game 3), which a mechanical reading can
    accept either way.
  - SHOT (20): a deliberate strike. Controlled.
  - RECOVERY (294) without SAVED/BLOCKED: "significantly interferes with the opponent's
    possession" or restarts a non-set-piece dead ball (goal kicks) - a takeover. Controlled.
  - RECOVERY with SAVED (6) or BLOCKED (3): a shot-stop deflection, not control ("a shot
    deflected off another player" ends possession and the ball is loose). NOT controlled; a
    caught ball flips one frame later at the keeper's CARRY.
  - BALL LOST (288) / BALL OUT (64): the event team is LOSING the ball; the gaining team's
    controlled touch (RECOVERY/PASS/CARRY) carries the flip. NOT controlled, never flips - this
    is exactly the "touches, deflections, unwon aerials are not changes" clause (e.g. BALL LOST
    + HEAD out of an aerial, or a CLEARANCE conceding a throw-in). The census counts the events
    whose team disagrees with the derived owner at their start frame and the frames affected
    (ball_lost_owner_mismatch; such frames are where Metrica's "touch = possession" notion and
    the mechanical reading genuinely differ).
  - CHALLENGE (335): a duel is never a change ("unwon aerials are not changes"); a won duel
    materialises as that team's RECOVERY (THEFT/INTERCEPTION) at the same moment. Never flips.
  - FAULT RECEIVED (12), CARD (1): not touches. Never flip.
  - RECOVERY + GOAL (1, frame 44157): no kick-off follows and play continues, so it is not a
    goal (the 2 real goals are SHOT + GOAL; 4 KICK OFFs = 2 half starts + 2 goals). Treated as a
    plain RECOVERY and reported.

Dead ball. D runs from the out-of-play/stoppage moment to the restart moment (exclusive); the
restart frame itself is alive and owned by the restarting team:
  - BALL OUT (64): dead from the event's END frame (the crossing; the start frame is the last
    touch). Restart: the next SET PIECE (throw-in/corner/free kick), or - goal kicks are NOT set
    pieces in this taxonomy - the next RECOVERY or GOAL KICK-subtyped event (both appear at the
    restart frame, e.g. RECOVERY + "PASS GOAL KICK" at 26690).
  - SHOT + OUT (6): dead from the shot's END frame, same restart rule (goal kick).
  - SHOT + GOAL (2): dead from the shot's END frame until the next SET PIECE (the KICK OFF).
    An own goal (Own-goal GOAL subtype on other types; none in Game 3) is handled the same way.
  - FAULT RECEIVED (12): dead from its frame ("the referee immediately awards a free kick")
    until the next SET PIECE.
  - FORCED (7) / VOLUNTARY (0) / HAND BALL (2) / OFFSIDE (4) subtypes on any touch event: the
    referee stops play; dead from the event's END frame until the next SET PIECE. OFFSIDE needs
    the SET-PIECE-only restart: Metrica keeps logging the offside team's touches (CARRY, SHOT,
    even a save) between the flag and the free kick, and those must not revive the ball.
  - END HALF (2) subtype: dead from the event's END frame to the end of the half.
  - REFEREE HIT (1): NOT a stoppage (2019 laws: play on; the data confirms an open-play
    RECOVERY follows). The ball stays alive and with the last owner.
  - CHALLENGE + FAULT (38) and CARD: not dead onsets; the whistle is carried by the possession
    team's FAULT RECEIVED / BALL LOST + FORCED at the same incident (verified: every free kick
    in Game 3 has one).
  - SET PIECE + RETAKEN (2): the first (disallowed) take. It restarts play like any SET PIECE;
    the following BALL LOST + FORCED kills the ball again until the clean retake. The census
    counts the briefly-alive frames (retaken_alive_frames).
  - Overlapping dead intervals (e.g. a foul during a dead ball) simply union.

X is written outside the halves by CleanMatch.gt_labels(); inside a half an alive frame with no
owner yet cannot occur (the kick-off is the half's first frame and sets the owner).

A dead onset whose restart is missing (no matching event before the half ends) runs to the half
end; the census requires this to happen only for END HALF (no_restart_found == 0).

25 Hz is asserted; a source at an integer multiple of 25 Hz would be decimated (positions and
event frames), anything else raises. Parsed once and cached as npz beside the data
(~/opendata/metrica_cache/), like io DFL/IDSSE.

CSV format (Sample Games 1-2, match ids "G1"/"G2"; added 2 Nov 2026 for the pre-registered
mechanism replication, submissions/LOG.md Exp B). Two per-team tracking CSVs (3 header rows:
team, jersey numbers, column names; then Period, Frame, Time [s], x,y pairs, ball x,y; normalised
coordinates, 25 Hz asserted, frames contiguous 1..N in both files with identical Period and ball
columns - all asserted) and one events CSV (Team "Home"/"Away", Type, Subtype, 1-based frames on
the tracking clock). The SAME mechanical derive() produces the GT. Format differences from the
Game 3 JSON, each verified on both games before any result was read:
  - Multi-part subtypes are "-"-joined ("HEAD-ON TARGET-GOAL"); split on "-" (no subtype name
    contains a hyphen; "KICK OFF", "GOAL KICK" etc. contain spaces, never hyphens).
  - No CARRY events: the owner flips only at SET PIECE / PASS / SHOT / controlled RECOVERY, so
    e.g. a keeper holding a parried shot flips at his next controlled event, not at the pick-up.
  - ADVANTAGE exists (G1: 5, G2: 11) but ONLY as a CHALLENGE subtype, and CHALLENGE neither
    flips nor stops play under the mechanical rules - the played-advantage reading is automatic,
    derive() is unchanged.
  - END HALF appears once per game (a BALL LOST subtype ending H1); H2 simply ends with the
    tracking, so no end-of-match D tail is derived there.
  - One event has End Frame 0 < Start Frame (G1's frame-1 KICK OFF): ends are normalised to
    max(start, end).
  - No metadata XML: the pitch is the provider's nominal 105 x 68 and the half ends come from
    the tracking Period column; team ids are "Home"/"Away" (home file first, as in the EPTS
    order). The source ball is NaN on ~39-41% of frames (G3: 31.9%), so the degradation's
    missing fraction overshoots the real target even more than G3's.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from mtah.learn import clean

OPEN_DIR = os.path.expanduser(os.environ.get("MTAH_METRICA_DIR", "~/opendata/metrica"))
PAD = 125  # frames of nothing before, between and after the halves (5 s), as in io/idsse.py
FPS = 25

FILES = {"meta": "Sample_Game_3_metadata.xml", "raw": "Sample_Game_3_tracking.txt",
         "events": "Sample_Game_3_events.json"}
CSV_STEMS = {"G1": "Sample_Game_1", "G2": "Sample_Game_2"}
CSV_PITCH = (105.0, 68.0)   # the provider's nominal pitch; the CSV games publish no metadata

CONTROLLED_TYPES = ("SET PIECE", "PASS", "CARRY", "SHOT", "RECOVERY")
UNCONTROLLED_RECOVERY = frozenset(("SAVED", "BLOCKED"))   # deflected shot-stops
WHISTLE_SUBTYPES = frozenset(("FORCED", "VOLUNTARY", "HAND BALL", "OFFSIDE"))
RESTART_SUBTYPES = frozenset(("GOAL KICK",))


def files(match_id="G3", open_dir=OPEN_DIR):
    if match_id in CSV_STEMS:
        stem = CSV_STEMS[match_id]
        paths = {"events": os.path.join(open_dir, stem + "_RawEventsData.csv"),
                 "home": os.path.join(open_dir, stem + "_RawTrackingData_Home_Team.csv"),
                 "away": os.path.join(open_dir, stem + "_RawTrackingData_Away_Team.csv")}
    else:
        paths = {k: os.path.join(open_dir, v) for k, v in FILES.items()}
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise IOError("Metrica Sample Game {} not found:\n  ".format(match_id)
                      + "\n  ".join(missing)
                      + "\nRun `make metrica-data` (scripts/fetch_metrica.py).")
    return paths


def cache_path(match_id="G3", open_dir=OPEN_DIR):
    return os.path.join(open_dir.rstrip(os.sep) + "_cache", "{}_clean_v1.npz".format(match_id))


def read_info(path):
    """(pitch, fps, half_end_frame per period, home/away team ids+names) from the metadata XML."""
    root = ET.parse(path).getroot()
    fps = int(next(root.iter("FrameRate")).text)
    width = float(next(root.iter("Width")).text)      # FieldSize Width = pitch length (105)
    height = float(next(root.iter("Height")).text)    # FieldSize Height = pitch width (68)
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")
              if (p.findtext("Name") or "").endswith(("_half_start", "_half_end"))}
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(path, len(teams)))
    return {
        "pitch": (width, height), "fps": fps,
        "half_end": {1: int(params["first_half_end"]), 2: int(params["second_half_end"])},
        "team_ids": (teams[0].get("id"), teams[1].get("id")),   # first team is home (EPTS)
        "names": tuple(t.findtext("Name") for t in teams),
    }


def load_events(path, team_ids):
    """Events as [{type, subs (frozenset), team (0/1), start, end, period}], 1-based frames,
    sorted by (period, start) as published."""
    with open(path) as handle:
        data = json.load(handle)
    out = []
    for e in data["data"]:
        ss = e.get("subtypes")
        subs = (frozenset() if ss is None
                else frozenset(s["name"] for s in ss) if isinstance(ss, list)
                else frozenset((ss["name"],)))
        out.append({
            "type": e["type"]["name"], "subs": subs,
            "team": team_ids.index(e["team"]["id"]),
            "start": int(e["start"]["frame"]), "end": int(e["end"]["frame"]),
            "period": int(e["period"]),
        })
    if out != sorted(out, key=lambda v: (v["period"], v["start"])):
        raise ValueError("events are not sorted by (period, start)")
    return out


def is_controlled(ev):
    if ev["type"] == "RECOVERY" and ev["subs"] & UNCONTROLLED_RECOVERY:
        return False
    return ev["type"] in CONTROLLED_TYPES


def dead_onset(ev):
    """(onset_frame, restart_kind) if the event stops play, else None.
    restart_kind: "half" (to the half end), "whistle" (next SET PIECE), "out" (next SET PIECE /
    RECOVERY / GOAL KICK event)."""
    if "END HALF" in ev["subs"]:
        return ev["end"], "half"
    if ev["type"] == "FAULT RECEIVED":
        return ev["start"], "whistle"
    if ev["subs"] & WHISTLE_SUBTYPES:
        return ev["end"], "whistle"
    if "GOAL" in ev["subs"] and ev["type"] != "RECOVERY":   # RECOVERY+GOAL: see docstring
        return ev["end"], "whistle"
    if ev["type"] == "BALL OUT" or (ev["type"] == "SHOT" and "OUT" in ev["subs"]):
        return ev["end"], "out"
    return None


def is_restart(ev, kind):
    if ev["type"] == "SET PIECE":
        return True
    if kind == "out":
        return ev["type"] == "RECOVERY" or bool(ev["subs"] & RESTART_SUBTYPES)
    return False


def kickoff_frames(events):
    """{period: frame} of the first SET PIECE + KICK OFF of each period."""
    out = {}
    for ev in events:
        if ev["type"] == "SET PIECE" and "KICK OFF" in ev["subs"] and ev["period"] not in out:
            out[ev["period"]] = ev["start"]
    return out


def derive(events, half_end, n_src):
    """(alive (n_src,) bool, owner (n_src,) int8, census dict) over source frames (index = id-1).

    Mechanical possession per the module docstring. Frames outside [kick-off, half end] carry
    alive=False / owner=-1 and are dropped by build()."""
    alive = np.zeros(n_src, dtype=bool)
    owner = np.full(n_src, -1, dtype=np.int8)
    kicks = kickoff_frames(events)
    census = {"flips": 0, "carry_flips": 0, "carry_glue_flips": 0, "carry_glue_flip_frames": 0,
              "ball_lost_owner_mismatch": 0, "ball_lost_owner_mismatch_frames": 0,
              "retaken_alive_frames": 0, "no_restart_found": 0, "dead_runs": 0,
              "uncontrolled_recovery": 0, "recovery_goal": 0}

    # Owner: a step function over the controlled events' start frames.
    flips = []  # (frame_id, team), in order
    cur = {}    # period -> owner so far, for the mismatch census
    mismatches, glue_flips = [], []
    for ev in events:
        prev = cur.get(ev["period"], -1)
        if is_controlled(ev):
            if ev["team"] != prev:
                census["flips"] += 1
                if ev["type"] == "CARRY":
                    census["carry_flips"] += 1
                    if ev["end"] - ev["start"] <= 2:
                        census["carry_glue_flips"] += 1
                        glue_flips.append(ev["start"])
            flips.append((ev["start"], ev["team"]))
            cur[ev["period"]] = ev["team"]
        else:
            if ev["type"] == "RECOVERY":
                census["uncontrolled_recovery"] += 1
            if ev["type"] in ("BALL LOST", "BALL OUT") and prev >= 0 and ev["team"] != prev:
                census["ball_lost_owner_mismatch"] += 1
                mismatches.append(ev["start"])
        if ev["type"] == "RECOVERY" and "GOAL" in ev["subs"]:
            census["recovery_goal"] += 1
    for f, team in flips:
        owner[f - 1:] = team

    # Alive: each period alive from its kick-off, minus [onset, restart) dead intervals.
    for period, kick in kicks.items():
        alive[kick - 1:half_end[period]] = True
    by_period = {p: [e for e in events if e["period"] == p] for p in kicks}
    for period, evs in by_period.items():
        for i, ev in enumerate(evs):
            hit = dead_onset(ev)
            if hit is None:
                continue
            onset, kind = hit
            restart = None
            if kind != "half":
                for nxt in evs[i + 1:]:
                    if nxt["start"] >= onset and is_restart(nxt, kind):
                        restart = nxt["start"]
                        break
                if restart is None:
                    census["no_restart_found"] += 1
            stop = restart if restart is not None else half_end[period] + 1
            if stop > onset:
                alive[onset - 1:stop - 1] = False
            census["dead_runs"] += 1

    # Census spans that need the finished arrays: how long each debatable reading would persist.
    # A touch-possession reading of a mismatch (or a non-controlled reading of a glue-carry flip)
    # re-converges with the mechanical one at the next controlled touch by either team, so the
    # divergence window runs from the event to the next flip frame, a dead frame, or an owner
    # change, whichever comes first.
    flip_frames = np.zeros(n_src, dtype=bool)
    for f, _ in flips:
        flip_frames[f - 1] = True

    def span_from(f, until_flip=True, same_owner=True):
        t, n = f - 1, 0
        while (t + n < n_src and alive[t + n]
               and not (until_flip and n > 0 and flip_frames[t + n])
               and (not same_owner or owner[t + n] == owner[t])):
            n += 1
        return n

    census["ball_lost_owner_mismatch_frames"] = sum(span_from(f) for f in mismatches)
    census["carry_glue_flip_frames"] = sum(span_from(f) for f in glue_flips)
    census["retaken_alive_frames"] = sum(
        span_from(ev["start"], until_flip=False, same_owner=False) for ev in events
        if ev["type"] == "SET PIECE" and "RETAKEN" in ev["subs"])
    return alive, owner, census


def parse_tracking(meta_path, raw_path, info):
    """(players (T,P,2) m, ball (T,2) m, team_of_player, player order) via kloppy 3.18 EPTS."""
    from kloppy import metrica as kloppy_metrica
    ds = kloppy_metrica.load_tracking_epts(meta_data=meta_path, raw_data=raw_path,
                                           coordinates="metrica")
    if ds.metadata.frame_rate != info["fps"]:
        raise ValueError("metadata FrameRate {} != kloppy frame_rate {}".format(
            info["fps"], ds.metadata.frame_rate))
    home_id, away_id = info["team_ids"]
    people = []
    for team in ds.metadata.teams:
        if team.team_id not in (home_id, away_id):
            raise ValueError("unexpected team id {}".format(team.team_id))
        for p in team.players:
            people.append((0 if team.team_id == home_id else 1, p.player_id))
    people.sort()
    column = {pid: i for i, (_, pid) in enumerate(people)}
    length, width = info["pitch"]
    n_src = max(f.frame_id for f in ds.frames)
    if len(ds.frames) != n_src or ds.frames[0].frame_id != 1:
        raise ValueError("tracking frames are not contiguous 1..{}".format(n_src))
    players = np.full((n_src, len(people), 2), np.nan)
    ball = np.full((n_src, 2), np.nan)
    for f in ds.frames:
        t = f.frame_id - 1
        b = f.ball_coordinates
        if b is not None and b.x is not None and not np.isnan(b.x):
            ball[t, 0], ball[t, 1] = b.x * length, b.y * width
        for player, pdata in f.players_data.items():
            c = pdata.coordinates
            if c is not None and c.x is not None and not np.isnan(c.x):
                col = column[player.player_id]
                players[t, col, 0], players[t, col, 1] = c.x * length, c.y * width
    team_of_player = np.array([team for team, _ in people], dtype=np.int8)
    return players, ball, team_of_player, n_src


def decimate(arrays, events, half_end, src_fps):
    """Integer-factor decimation to FPS (unused for Game 3: the source is 25 Hz already)."""
    if src_fps == FPS:
        return arrays, events, half_end
    if src_fps % FPS:
        raise ValueError("source at {} Hz: only integer multiples of {} are supported".format(
            src_fps, FPS))
    k = src_fps // FPS
    arrays = [a[::k] for a in arrays]
    events = [dict(e, start=(e["start"] - 1) // k + 1, end=(e["end"] - 1) // k + 1)
              for e in events]
    half_end = {p: (f - 1) // k + 1 for p, f in half_end.items()}
    return arrays, events, half_end


def load_events_csv(path):
    """Events in load_events' shape from the CSV format: Team "Home"/"Away", "-"-joined
    subtypes (no subtype name contains a hyphen), ends normalised to max(start, end)."""
    import pandas as pd
    table = pd.read_csv(path)
    teams = {"Home": 0, "Away": 1}
    team = [teams[v] for v in table["Team"]]
    subs = [frozenset(p for p in str(v).split("-") if p)
            for v in table["Subtype"].fillna("")]
    start = table["Start Frame"].astype(int).tolist()
    end = table["End Frame"].astype(int).tolist()
    period = table["Period"].astype(int).tolist()
    out = [{"type": str(t), "subs": s, "team": m, "start": a, "end": max(a, b), "period": p}
           for t, s, m, a, b, p in zip(table["Type"], subs, team, start, end, period)]
    if out != sorted(out, key=lambda v: (v["period"], v["start"])):
        raise ValueError("events are not sorted by (period, start)")
    return out


def _read_tracking_csv(path):
    """(period (T,) int, players (T,P,2) normalised, ball (T,2) normalised) from one per-team
    tracking CSV (3 header rows; Period, Frame, Time [s], x,y pairs, ball x,y)."""
    import pandas as pd
    data = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=float)
    with open(path) as handle:
        handle.readline()
        handle.readline()
        names = [c.strip() for c in handle.readline().rstrip("\n").split(",")][:data.shape[1]]
    if names[:3] != ["Period", "Frame", "Time [s]"] or names[-2] != "Ball":
        raise ValueError("{}: unexpected column layout {}..{}".format(path, names[:4], names[-2:]))
    frame = data[:, 1].astype(int)
    if not np.array_equal(frame, np.arange(1, len(frame) + 1)):
        raise ValueError("{}: frames are not contiguous 1..{}".format(path, len(frame)))
    dt = float(np.nanmedian(np.diff(data[:, 2])))
    if abs(dt - 1.0 / FPS) > 1e-6:
        raise ValueError("{}: {} s between frames; only 25 Hz CSVs are supported".format(path, dt))
    n_players = (data.shape[1] - 5) // 2
    players = data[:, 3:3 + 2 * n_players].reshape(len(frame), n_players, 2)
    return data[:, 0].astype(int), players, data[:, 3 + 2 * n_players:3 + 2 * n_players + 2]


def _build_csv(match_id, open_dir):
    paths = files(match_id, open_dir)
    events = load_events_csv(paths["events"])
    period_h, players_h, ball_h = _read_tracking_csv(paths["home"])
    period_a, players_a, ball_a = _read_tracking_csv(paths["away"])
    if not np.array_equal(period_h, period_a):
        raise ValueError("{}: home and away tracking disagree on frames/periods".format(match_id))
    both = ~np.isnan(ball_h[:, 0]) & ~np.isnan(ball_a[:, 0])
    if not (np.array_equal(np.isnan(ball_h), np.isnan(ball_a))
            and np.allclose(ball_h[both], ball_a[both], atol=1e-9)):
        raise ValueError("{}: home and away ball columns disagree".format(match_id))
    scale = np.asarray(CSV_PITCH)
    players = np.concatenate([players_h, players_a], axis=1) * scale
    team_of_player = np.array([0] * players_h.shape[1] + [1] * players_a.shape[1], dtype=np.int8)
    half_end = {int(p): int(np.flatnonzero(period_h == p)[-1] + 1)
                for p in np.unique(period_h)}
    if sorted(half_end) != [1, 2]:
        raise ValueError("{}: expected periods 1 and 2, found {}".format(match_id,
                                                                         sorted(half_end)))
    return _finish(match_id, players, ball_h * scale, team_of_player, events, half_end,
                   CSV_PITCH, ("Home", "Away"))


def build(match_id="G3", open_dir=OPEN_DIR):
    if match_id in CSV_STEMS:
        return _build_csv(match_id, open_dir)
    paths = files(match_id, open_dir)
    info = read_info(paths["meta"])
    events = load_events(paths["events"], info["team_ids"])
    players, ball, team_of_player, n_src = parse_tracking(paths["meta"], paths["raw"], info)
    (players, ball), events, half_end = decimate([players, ball], events, info["half_end"],
                                                 info["fps"])
    return _finish(match_id, players, ball, team_of_player, events, half_end, info["pitch"],
                   info["names"])


def _finish(match_id, players, ball, team_of_player, events, half_end, pitch, team_names):
    """Derive the mechanical GT, cut the halves at [kick-off, half end], pad and assemble."""
    n_src = players.shape[0]
    alive_src, owner_src, _ = derive(events, half_end, n_src)
    kicks = kickoff_frames(events)
    if sorted(kicks) != [1, 2]:
        raise ValueError("kick-off events not found for both periods: {}".format(sorted(kicks)))

    starts, lengths = {}, {}
    t = PAD
    for period in (1, 2):
        lengths[period] = half_end[period] - kicks[period] + 1
        starts[period] = t
        t += lengths[period] + PAD
    n_frames = t
    out_players = np.full((n_frames, players.shape[1], 2), np.nan)
    out_ball = np.full((n_frames, 2), np.nan)
    alive = np.zeros(n_frames, dtype=bool)
    owner = np.full(n_frames, -1, dtype=np.int8)
    for period in (1, 2):
        a, b = kicks[period] - 1, half_end[period]            # source slice [a, b)
        s = starts[period]
        out_players[s:s + lengths[period]] = players[a:b]
        out_ball[s:s + lengths[period]] = ball[a:b]
        alive[s:s + lengths[period]] = alive_src[a:b]
        owner[s:s + lengths[period]] = owner_src[a:b]
        if not alive[s] or owner[s] < 0:
            raise ValueError("half {} does not start alive with a known owner".format(period))
    halves = np.array([[starts[p], starts[p] + lengths[p] - 1] for p in (1, 2)])

    # The two halves must attack opposite ways (gapstate.home_direction asserts this later).
    sides = []
    for period in (1, 2):
        k = starts[period]
        hx = np.nanmean(out_players[k, team_of_player == 0, 0])
        ax = np.nanmean(out_players[k, team_of_player == 1, 0])
        if not (np.isfinite(hx) and np.isfinite(ax)):
            raise ValueError("half {}: no players at the kick-off frame".format(period))
        sides.append(hx < ax)
    if sides[0] == sides[1]:
        raise ValueError("both halves resolve to the home team attacking the same way")

    return clean.CleanMatch(
        name=match_id, fps=FPS, pitch=pitch, players=out_players,
        team_of_player=team_of_player, ball=out_ball, alive=alive, owner=owner, halves=halves,
        team_names=team_names)


def census(match_id="G3", open_dir=OPEN_DIR):
    """The event->label derivation census (ambiguity counters) without building the tracking."""
    paths = files(match_id, open_dir)
    if match_id in CSV_STEMS:
        import pandas as pd
        events = load_events_csv(paths["events"])
        period = pd.read_csv(paths["home"], skiprows=3, header=None,
                             usecols=[0])[0].to_numpy(dtype=int)
        half_end = {int(p): int(np.flatnonzero(period == p)[-1] + 1) for p in np.unique(period)}
        _, _, out = derive(events, half_end, len(period))
        return out
    info = read_info(paths["meta"])
    events = load_events(paths["events"], info["team_ids"])
    _, _, out = derive(events, info["half_end"], info["half_end"][2])
    return out


def load(match_id="G3", open_dir=OPEN_DIR, use_cache=True):
    path = cache_path(match_id, open_dir)
    if use_cache and os.path.isfile(path):
        return clean.load(path)
    match = build(match_id, open_dir)
    clean.save(match, path)
    return match
