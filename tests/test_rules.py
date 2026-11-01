"""Ball cleaning, team distances, touch detection and the owner hypotheses H1-H3."""

import os

import numpy as np

from mtah import config, registry
from mtah.decode.rules import decode_owner
from mtah.eval.score import score
from mtah.events.touches import Touch, touch_events
from mtah.features.ball import clean_ball
from mtah.features.players import team_distances

CFG = config.load("rules.yaml")
PITCH = (105.0, 68.0)


def test_clean_ball():
    T = 100
    ball = np.column_stack([np.linspace(10, 60, T), np.full(T, 30.0)])
    ball[20] = [10.0, 200.0]          # far outside the pitch
    ball[40:45] = np.nan              # short gap: filled
    ball[60:80] = np.nan              # long gap: kept
    out, observed, filled = clean_ball(ball, PITCH, CFG["ball"])
    assert not observed[20] and filled[20] and abs(out[20, 1] - 30.0) < 1e-9
    assert filled[40:45].all() and np.allclose(out[42, 0], np.linspace(10, 60, T)[42])
    assert np.isnan(out[60:80]).all() and not filled[60:80].any()


def test_team_distances():
    players = np.full((2, 3, 2), np.nan)
    players[:, 0] = [0.0, 0.0]
    players[:, 1] = [3.0, 4.0]
    players[1, 2] = [1.0, 0.0]
    ball = np.array([[0.0, 1.0], [np.nan, np.nan]])
    d, slot = team_distances(players, ball, np.array([0, 1, 1]))
    assert np.allclose(d[0], [1.0, np.hypot(3, 3)]) and slot[0].tolist() == [0, 1]
    assert np.isnan(d[1]).all() and slot[1].tolist() == [-1, -1]


def _one_player_world(ball):
    T = len(ball)
    players = np.full((T, 2, 2), np.nan)
    players[:, 0] = [50.0, 34.0]
    players[:, 1] = [10.0, 10.0]
    return team_distances(players, ball, np.array([0, 1]))[0]


def test_touch_needs_contact_and_a_kinematic_change():
    t = np.arange(60)
    # 1. Ball rolls straight past the player at 20 m/s: 3 contact frames, no turn, no slowdown.
    fly = np.column_stack([30 + 0.8 * t, np.full(60, 34.5)])
    tcfg = dict(CFG["touch"], min_frames=5)
    assert touch_events(fly, _one_player_world(fly), tcfg) == []
    # 2. Ball arrives and stops at the player's feet: a long contact = a touch.
    stop = np.column_stack([np.minimum(30 + 0.8 * t, 49.5), np.full(60, 34.0)])
    touches = touch_events(stop, _one_player_world(stop), tcfg)
    assert len(touches) == 1 and touches[0].team == 0 and touches[0].n_frames >= 3
    # 3. Ball deflected 90 degrees in one frame of contact: a touch by the turn test.
    turn = np.column_stack([np.where(t < 25, 30 + 0.8 * t, 50.0),
                            np.where(t < 25, 34.0, 34.0 + 0.8 * (t - 25))])
    touches = touch_events(turn, _one_player_world(turn), tcfg)
    assert len(touches) == 1 and touches[0].turn_deg > 60


def touch(start, team, n=5, after=3.0):
    return Touch(start=start, end=start + n - 1, team=team, n_frames=n, d_min=0.5,
                 speed_before=5.0, speed_after=after, turn_deg=30.0)


def test_hypotheses():
    T = 400
    graded = np.zeros(T, bool)
    graded[50:350] = True
    dead = np.zeros(T, bool)
    dead[200:250] = True
    ocfg = dict(CFG["owner"], n_c=5, v_c_mps=4.0, t_c_s=2.0)
    deflection = touch(100, 1, n=1, after=15.0)      # A brushes it: fast, short, alone
    seq = [touch(60, 0), deflection, touch(120, 0),  # H keeps it
           touch(150, 1, n=1, after=15.0), touch(160, 1, n=1, after=15.0),  # A passes twice
           touch(260, 0), touch(300, 1)]             # after D: H restarts; A wins it (controlled)
    labels = {h: decode_owner(seq, graded, dead, h, ocfg) for h in ("H1", "H2", "H3")}
    for lab in labels.values():
        assert set(lab[~graded]) == {"X"} and set(lab[dead & graded]) == {"D"}
        assert "X" not in set(lab[graded])
        assert (lab[50:60] == "H").all()                         # back-filled to the half start
        assert (lab[250:260] == "H").all()                       # back-filled to the D end
    assert labels["H1"][100] == "A" and labels["H1"][119] == "A"  # H1 flips on the deflection
    assert labels["H2"][100] == "H" and labels["H3"][100] == "H"  # H2/H3 ignore it
    assert labels["H2"][150] == "A" and labels["H3"][150] == "A"  # pass completed: flip at 1st
    assert labels["H2"][300] == "A"                               # controlled: slow after
    assert labels["H3"][300] == "H"                               # single touch: H3 holds


def test_h2_on_the_fake_match(fake, fake_atd):
    atd, _ = fake_atd
    sp = registry.spec(fake["game"], fake["data_dir"])
    gt_path = os.path.join(fake["data_dir"], "gt.csv")
    gt = np.loadtxt(gt_path, delimiter=",", skiprows=1, dtype=str)[:, 1]
    ball, _, _ = clean_ball(atd.ball_m(), atd.pitch, CFG["ball"])
    d, _ = team_distances(atd.players_m()[..., :2].astype(float), ball, atd.team_of_slot)
    labels = decode_owner(touch_events(ball, d, CFG["touch"]), registry.graded_mask(sp),
                          gt == "D", "H2", CFG["owner"])
    assert score(labels, gt_path)["spell_f1_mean"] > 0.3


def test_launch_from_rest_or_unseen_is_a_touch():
    t = np.arange(60)
    # A placed ball at the player's feet is kicked away at 8 m/s on frame 30: 3 frames of contact
    # before it leaves, no velocity before it, so no turn and only an 8 m/s change.
    x = np.where(t < 30, 49.5, 49.5 + 0.32 * (t - 30))
    kick = np.column_stack([x, np.full(60, 34.0)])
    unseen = kick.copy()
    unseen[:29] = np.nan                                  # the ball reappears as it is kicked
    d_kick, d_unseen = _one_player_world(kick), _one_player_world(unseen)
    off = dict(CFG["touch"], min_frames=40, min_dv_mps=12.0, min_launch_mps=99.0)
    on = dict(off, min_launch_mps=5.0, rest_mps=1.0)
    assert touch_events(kick, d_kick, off) == [] and touch_events(unseen, d_unseen, off) == []
    assert len(touch_events(kick, d_kick, on)) == 1 and len(touch_events(unseen, d_unseen, on)) == 1
    # A ball already moving past the player is no launch, however fast it leaves.
    fly = np.column_stack([30 + 0.8 * t, np.full(60, 34.5)])
    assert touch_events(fly, _one_player_world(fly), dict(on, min_frames=5)) == []
