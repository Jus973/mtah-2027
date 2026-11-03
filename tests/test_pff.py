import json

import numpy as np

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
