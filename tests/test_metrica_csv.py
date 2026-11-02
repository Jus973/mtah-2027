"""io/metrica_csv.py: Metrica Sample Games 1-2 CSV release -> CleanMatch (synthetic files)."""

import numpy as np
import pytest

from mtah.io import metrica_csv

HEADER = "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To," \
         "Start X,Start Y,End X,End Y\n"


def _tracking(path, side, n, ball):
    rows = [",,,{0},,{0},,,".format(side), ",,,1,,2,,,",
            "Period,Frame,Time [s],Player1,,Player2,,Ball,"]
    for f in range(1, n + 1):
        period = 1 if f <= n // 2 else 2
        x = 0.25 if side == "Home" else 0.75
        bx, by = ball(f)
        rows.append("{},{},{:.2f},{},0.5,{},0.4,{},{}".format(period, f, f / 25, x, x, bx, by))
    path.write_text("\n".join(rows) + "\n")


def _write(tmp_path, n=40):
    ev = HEADER + "\n".join([
        "Home,SET PIECE,KICK OFF,1,1,0.04,0,0,Player1,,NaN,NaN,NaN,NaN",
        "Home,PASS,,1,1,0.04,3,0.12,Player1,Player2,0.5,0.5,0.6,0.5",
        "Away,RECOVERY,INTERCEPTION,1,6,0.24,6,0.24,Player3,,0.6,0.5,NaN,NaN",
        "Away,BALL OUT,HEAD-OFF TARGET-OUT,1,9,0.36,12,0.48,Player3,,0.6,0.5,1.0,0.5",
        "Home,SET PIECE,THROW IN,1,15,0.6,15,0.6,Player1,,NaN,NaN,NaN,NaN",
        "Away,SET PIECE,KICK OFF,2,21,0.84,0,0,Player3,,NaN,NaN,NaN,NaN",
        "Away,PASS,,2,21,0.84,24,0.96,Player3,Player4,0.5,0.5,0.4,0.5",
    ]) + "\n"
    (tmp_path / "Sample_Game_9_RawEventsData.csv").write_text(ev)
    ball = lambda f: ("NaN", "NaN") if 12 <= f < 15 else (0.5, 0.5)
    _tracking(tmp_path / "Sample_Game_9_RawTrackingData_Home_Team.csv", "Home", n, ball)
    _tracking(tmp_path / "Sample_Game_9_RawTrackingData_Away_Team.csv", "Away", n, ball)


def test_subtypes_split_like_the_json_release(tmp_path):
    _write(tmp_path)
    ev = metrica_csv.load_events(str(tmp_path / "Sample_Game_9_RawEventsData.csv"))
    out = [e for e in ev if e["type"] == "BALL OUT"][0]
    assert out["subs"] == {"HEAD", "OFF TARGET", "OUT"}
    assert out["team"] == 1 and out["start"] == 9 and out["end"] == 12 and out["period"] == 1
    assert [e["type"] for e in ev if e["period"] == 2] == ["SET PIECE", "PASS"]


def test_build_maps_metres_halves_and_labels(tmp_path):
    _write(tmp_path)
    m = metrica_csv.build(9, open_dir=str(tmp_path))
    assert m.pitch == metrica_csv.PITCH and m.players.shape[1] == 4
    assert m.team_of_player.tolist() == [0, 0, 1, 1]
    (a1, e1), (a2, e2) = m.halves.tolist()
    assert e1 - a1 + 1 == 20 and e2 - a2 + 1 == 20          # kick-off frame -> half end
    assert np.allclose(m.ball[a1], [0.5 * 105, 0.5 * 68])
    assert np.isnan(m.ball[a1 + 11, 0])                     # source-tracker loss is kept
    labels = m.gt_labels()
    assert labels[a1] == "H" and labels[a2] == "A"
    assert "D" in labels[a1:e1 + 1] and "A" in labels[a1:e1 + 1]
    assert (labels[:a1] == "X").all() and (labels[e2 + 1:] == "X").all()


def test_missing_files_say_how_to_fetch(tmp_path):
    with pytest.raises(IOError, match="metrica-data"):
        metrica_csv.paths(1, open_dir=str(tmp_path))
