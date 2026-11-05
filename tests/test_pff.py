import json

import numpy as np
import pytest

from mtah.io import pff

META = [{"id": 7, "stadium": {"pitches": [{"length": 105.0, "width": 68.0}]},
         "homeTeam": {"id": 1, "name": "H"}, "awayTeam": {"id": 2, "name": "A"}}]
ROSTER = [{"shirtNumber": "9", "team": {"id": "1"}}, {"shirtNumber": "4", "team": {"id": "2"}}]


def frame(q, t, ev=None, home_ball=None, ball=(0.0, 0.0)):
    f = {"period": q, "periodElapsedTime": t,
         "homePlayersSmoothed": [{"jerseyNum": "9", "x": 1.0, "y": 2.0}],
         "awayPlayersSmoothed": [{"jerseyNum": "4", "x": -1.0, "y": -2.0}],
         "ballsSmoothed": {"x": ball[0], "y": ball[1]}, "game_event": None}
    if ev:
        f["game_event"] = {"game_event_type": ev, "home_ball": home_ball}
    return json.dumps(f)


def test_parse_labels_resample_and_origin():
    dt = 1 / 29.97
    lines = [frame(1, 0.0, "FIRSTKICKOFF", True)]
    lines += [frame(1, k * dt) for k in range(1, 60)]
    lines += [frame(1, 60 * dt, "OUT", False)] + [frame(1, k * dt) for k in range(61, 90)]
    lines += [frame(1, 90 * dt, "OTB", False)] + [frame(1, k * dt) for k in range(91, 120)]
    lines += [frame(2, 0.0, "SECONDKICKOFF", False)] + [frame(2, k * dt) for k in range(1, 30)]
    m = pff.parse(META, ROSTER, lines)
    (a1, b1), (a2, b2) = m.halves
    assert a1 == pff.PAD and b1 - a1 + 1 == len(np.arange(0, 119 * dt + 1e-9, 0.04))
    lab = m.gt_labels()
    h1 = lab[a1:b1 + 1]
    t = np.arange(len(h1)) * 0.04
    assert set(h1[t < 59 * dt]) == {"H"}
    assert set(h1[(t > 61 * dt) & (t < 89 * dt)]) == {"D"}
    assert set(h1[t > 91 * dt]) == {"A"}
    assert set(lab[a2:b2 + 1]) == {"A"}
    assert np.allclose(m.players[a1, 0], (53.5, 36.0)) and np.allclose(m.ball[a1], (52.5, 34.0))
    assert m.team_of_player.tolist() == [0, 1]


def test_attacking_direction_widens_window_when_a_team_is_untracked():
    from types import SimpleNamespace

    from mtah.features.context import attacking_direction
    T = 2000
    xy = np.full((T, 2, 2), np.nan)
    xy[:, 0, 0], xy[:, 1, 0] = 0.3, 0.7
    xy[400:700, 1] = np.nan                 # away untracked across the default window
    atd = SimpleNamespace(xy=xy, team_of_slot=np.array([0, 1]))
    assert attacking_direction(atd, {1: 500}) == {1: {"H": 1, "A": -1}}
    xy[:, 1] = np.nan
    try:
        attacking_direction(atd, {1: 500})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when a team is never visible")


def test_parse_state_follows_match_time_not_file_order():
    dt = 1 / 29.97
    lines = [frame(1, 0.0, "FIRSTKICKOFF", True)] + [frame(1, k * dt) for k in range(1, 40)]
    lines += [frame(1, 40 * dt, "OUT", False)] + [frame(1, k * dt) for k in range(41, 80)]
    lines += [frame(2, 0.0, "SECONDKICKOFF", False)] + [frame(2, k * dt) for k in range(1, 30)]
    ordered = pff.parse(META, ROSTER, lines)
    shuffled = lines[:]
    np.random.default_rng(0).shuffle(shuffled)
    m = pff.parse(META, ROSTER, shuffled)
    assert (m.gt_labels() == ordered.gt_labels()).all()
    assert np.allclose(np.nan_to_num(m.players), np.nan_to_num(ordered.players))


def test_second_period_cannot_inherit_first_period_owner():
    lines = [frame(1, 0, "FIRSTKICKOFF", True), frame(2, 0), frame(2, 0.04, "OTB", False)]
    with pytest.raises(ValueError, match="has no known owner"):
        pff.parse(META, ROSTER, lines)


def test_untimed_second_kickoff_belongs_to_its_own_period():
    lines = [frame(1, 0, "FIRSTKICKOFF", True), frame(2, None, "SECONDKICKOFF", False),
             frame(1, 0.04), frame(2, 0)]
    match = pff.parse(META, ROSTER, lines)
    assert match.gt_labels()[match.halves[0, 1]] == "H"
    assert match.gt_labels()[match.halves[1, 0]] == "A"


def test_periodless_untimed_kickoffs_start_the_named_half():
    lines = [frame(None, None, "FIRSTKICKOFF", True), frame(1, 0),
             frame(None, None, "SECONDKICKOFF", False), frame(1, 0.04), frame(2, 0)]
    match = pff.parse(META, ROSTER, lines)
    assert list(match.gt_labels()[match.halves[:, 0]]) == ["H", "A"]
