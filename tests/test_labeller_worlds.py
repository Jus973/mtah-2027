"""Heuristic-labeller GT worlds behind scripts/labeller_worlds.py (board #69)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from labeller_worlds import (_move_switches, candidates, control, describe,  # noqa: E402
                             direct_switches, hysteresis, world_l1, world_l2, world_l4,
                             world_l5)
from mtah.eval.score import pm  # noqa: E402

FPS = 25
NEAR, MID, FAR = 0.5, 1.2, 5.0


def arr(spec):
    """'H5 A3 D2 ...' -> label array."""
    out = []
    for tok in spec.split():
        out += [tok[0]] * int(tok[1:])
    return np.array(out, dtype=str)


def spec(labels):
    codes = pm.label_codes(labels)
    return " ".join("{}{}".format(pm.LABELS[c], b - a)
                    for a, b, c in pm.segments(codes, [(0, len(codes))]))


def dist(n, near=None):
    """(n, 2) per-team distances, FAR everywhere except near = [(team, a, b, value)]."""
    d = np.full((n, 2), FAR)
    for team, a, b, v in near or []:
        d[a:b, team] = v
    return d


# A hand-built 3-spell fixture: H 40, A 40, H 40, each team at the ball through its own spell.
GT3 = arr("H40 A40 H40")
D3 = dist(120, [(0, 0, 40, NEAR), (1, 40, 80, NEAR), (0, 80, 120, NEAR)])


def test_every_labeller_reproduces_a_clean_three_spell_match():
    for h in (1, 5, 12):
        assert spec(world_l1(GT3, D3, h)) == "H40 A40 H40"
    assert spec(world_l2(GT3, D3, FPS)) == "H40 A40 H40"
    assert spec(world_l5(GT3, D3)) == "H40 A40 H40"       # winner touches right at the switch
    assert spec(world_l4(GT3, world_l1(GT3, D3, 5))) == "H40 A40 H40"


def test_hysteresis_filters_short_blips_and_backdates_switches():
    d = D3.copy()
    d[10:13] = [FAR, NEAR]                                 # a 3-frame A blip inside the H spell
    assert spec(world_l1(GT3, d, 1)) == "H10 A3 H27 A40 H40"
    assert spec(world_l1(GT3, d, 5)) == "H40 A40 H40"
    stream = hysteresis(np.array([0, 0, 1, 1, 1, -1, 1], dtype=np.int8), 3)
    assert stream.tolist() == [0, 0, 1, 1, 1, 1, 1]        # switch backdated to frame 2


def test_no_candidate_stretch_keeps_source_and_first_candidate_backfills():
    assert spec(world_l1(GT3, dist(120), 5)) == "H40 A40 H40"
    d = dist(120, [(1, 10, 120, NEAR)])                    # A is the only team ever near
    assert spec(world_l1(GT3, d, 5)) == "A120"
    assert candidates(np.array([[np.nan, np.nan]]), np.inf).tolist() == [-1]
    assert candidates(np.array([[1.4, 0.3]]), 1.5).tolist() == [1]
    assert candidates(np.array([[1.6, 2.0]]), 1.5).tolist() == [-1]


def test_control_rule_long_hold_and_two_episodes():
    gt = arr("H200")
    d = dist(200, [(0, 0, 5, NEAR), (1, 30, 54, NEAR)])    # A holds 24 frames: not enough
    assert spec(world_l2(gt, d, FPS)) == "H200"
    d = dist(200, [(0, 0, 5, NEAR), (1, 30, 55, NEAR)])    # 25 frames: switch backdated to 30
    assert spec(world_l2(gt, d, FPS)) == "H30 A170"
    d = dist(200, [(0, 0, 5, NEAR), (1, 40, 43, NEAR), (1, 110, 112, NEAR)])
    assert spec(world_l2(gt, d, FPS)) == "H40 A160"        # two episodes 70 frames apart
    d = dist(200, [(0, 0, 5, NEAR), (1, 40, 43, NEAR), (1, 120, 122, NEAR)])
    assert spec(world_l2(gt, d, FPS)) == "H200"            # 80 frames apart: > 3 s
    assert control(dist(10), FPS).tolist() == [-1] * 10


def test_operator_snap_moves_only_to_a_same_team_switch_in_window():
    gt = arr("H50 A50")
    assert spec(world_l4(gt, arr("H60 A40"))) == "H60 A40"
    assert spec(world_l4(gt, arr("H30 A70"))) == "H30 A70"
    assert spec(world_l4(gt, arr("H80 A20"))) == "H50 A50"  # 30 frames away: outside +/- 25
    assert spec(world_l4(gt, arr("A60 H20 A20"))) == "H50 A50"  # nearest is a switch to H


def test_pass_release_moves_the_switch_back_to_the_losing_touch():
    gt = arr("H50 A50")
    d = dist(100, [(0, 30, 40, NEAR), (1, 50, 60, NEAR)])
    assert spec(world_l5(gt, d)) == "H40 A60"
    d = dist(100, [(0, 30, 40, NEAR), (1, 45, 46, NEAR), (1, 50, 60, NEAR)])
    assert spec(world_l5(gt, d)) == "H50 A50"              # winner touched in between
    gt = arr("H100 A100")
    d = dist(200, [(0, 0, 40, NEAR), (1, 100, 110, NEAR)])
    assert spec(world_l5(gt, d)) == "H100 A100"            # 60 frames of flight: > 50
    d = dist(200, [(0, 0, 50, NEAR), (1, 100, 110, NEAR)])
    assert spec(world_l5(gt, d)) == "H50 A150"             # exactly 50 frames of flight


def test_moves_never_cross_or_erase_a_spell():
    gt = arr("H30 A5 H30")
    out = _move_switches(gt, lambda s, lo, hi: s + 10 if s == 30 else s - 10)
    assert spec(out) == "H34 A1 H30"
    assert len(direct_switches(out)) == 2


def random_world(seed, n=800):
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        out += [rng.choice(["H", "A", "D", "X"], p=[0.4, 0.4, 0.15, 0.05])] * rng.randint(1, 60)
    gt = np.array(out[:n], dtype=str)
    d = rng.choice([NEAR, MID, FAR, np.nan], size=(n, 2), p=[0.2, 0.2, 0.5, 0.1])
    return gt, d


def test_worlds_never_touch_d_or_x_and_keep_every_alive_frame_labelled():
    for seed in range(20):
        gt, d = random_world(seed)
        for lab in (world_l1(gt, d, 1), world_l1(gt, d, 12), world_l2(gt, d, FPS),
                    world_l4(gt, world_l1(gt, d, 5)), world_l5(gt, d)):
            assert ((lab == "D") == (gt == "D")).all()
            assert ((lab == "X") == (gt == "X")).all()
            assert np.isin(lab[np.isin(gt, ["H", "A"])], ["H", "A"]).all()


def test_describe_counts_spells_short_share_switches_and_flips():
    lab = arr("H10 A60 D5 H60")
    nearest = np.array([0] * 10 + [1, 0] * 30 + [-1] * 5 + [0] * 60, dtype=np.int8)
    n, short, sw, flips = describe(lab, nearest, FPS)
    assert (n, sw) == (3, 1)
    assert abs(short - 1 / 3) < 1e-9
    assert flips == 60                                     # 1 at the H->A edge + 59 inside A
