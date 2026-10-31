"""Scorer wrapper, team-sign catastrophe, structural bands, submission contract."""

import os
import subprocess
import sys

import numpy as np
import pytest

import baseline as starter
from mtah import config, registry
from mtah.eval.bands import bands
from mtah.eval.score import PUBLIC_GT, public_check, score
from mtah.features.baseline import nearest_player_codes
from mtah.submit.write import validate, write_csv


def swap(labels):
    labels = np.asarray(labels).astype("<U1").copy()
    h, a = labels == "H", labels == "A"
    labels[h], labels[a] = "A", "H"
    return labels


def gt_labels(fake):
    return np.loadtxt(os.path.join(fake["data_dir"], "gt.csv"), delimiter=",", skiprows=1,
                      dtype=str)[:, 1]


def test_public_gt_self_score():
    s = score(PUBLIC_GT, "public", allow_public=True)
    assert s["spell_f1_mean"] == 1.0 and s["n_gt_spells"] == 17


def test_public_window_refuses_exact_scores():
    with pytest.raises(PermissionError, match="tuning set"):
        score(PUBLIC_GT, "public")


def test_public_check_verdicts(tmp_path):
    cfg = config.load()["public_check"]
    gt = np.loadtxt(PUBLIC_GT, delimiter=",", skiprows=1, dtype=str)[:, 1]
    assert public_check(PUBLIC_GT, cfg) == {"verdict": "pass", "reason": None,
                                            "swap_scores_higher": False}
    swapped = public_check(swap(gt), cfg)
    assert swapped["verdict"] == "broken" and swapped["swap_scores_higher"]
    assert public_check(gt, dict(cfg, floor=1.01))["verdict"] == "fail"
    short = write_csv(str(tmp_path / "short.csv"), gt[:100])
    assert public_check(short, cfg)["verdict"] == "invalid"
    assert set(public_check(PUBLIC_GT, cfg)) == {"verdict", "reason", "swap_scores_higher"}


def test_team_sign_is_catastrophic(fake, fake_atd):
    gt_path = os.path.join(fake["data_dir"], "gt.csv")
    gt = gt_labels(fake)
    assert score(swap(gt), gt_path)["spell_f1_mean"] == 0.0
    pred = starter.smooth(nearest_player_codes(fake_atd[0]), 50)
    right = score(pred, gt_path)["spell_f1_mean"]
    wrong = score(swap(pred), gt_path)["spell_f1_mean"]
    assert right > wrong + 0.05 and wrong < 0.05


def test_bands_on_fake_gt(fake):
    sp = registry.spec(fake["game"], fake["data_dir"])
    cfg = config.load()["bands"]
    kick = {int(k): v for k, v in fake["meta"]["kickoffs"].items()}
    team = {int(k): v for k, v in fake["meta"]["kickoff_team"].items()}
    gt = gt_labels(fake)
    b = bands(gt, sp, cfg, kick, team)
    assert b["checks"]["pre_kickoff_D_half2"] and b["checks"]["kickoff_team_half1"]
    assert b["checks"]["x_frac"] and b["checks"]["frac_lt_2s"]
    all_x = np.where(registry.graded_mask(sp), "X", "X")
    assert not bands(all_x, sp, cfg)["pass"]


def test_submission_contract(tmp_path, fake):
    sp = registry.spec(fake["game"], fake["data_dir"])
    labels = gt_labels(fake)
    path = write_csv(str(tmp_path / "ok.csv"), labels)
    assert validate(path, sp["n_frames"]) is None
    assert validate(path, sp["n_frames"] + 1) is not None
    bad = tmp_path / "bad.csv"
    bad.write_text("frame,label\n0,H\n1,\n")
    assert "empty label" in validate(str(bad), 2)


def test_no_validation_gt():
    with pytest.raises(ValueError, match="RULES.md"):
        score(PUBLIC_GT, "validation")


EXPECTED_COSTS = [  # PLAN.md §2 table, spellF1 then macroF1 (2 dp)
    ("all boundaries 3 frames", 1.00, 0.99), ("all boundaries 8 frames", 0.95, 0.98),
    ("all boundaries 12 frames", 0.91, 0.97), ("all boundaries 25 frames", 0.78, 0.93),
    ("all boundaries 50 frames", 0.59, 0.87), ("blips < 1.0s", 0.91, 1.00),
    ("blips < 2.0s", 0.83, 0.99), ("blips < 4.0s", 0.56, 0.97),
    ("every D predicted", 0.81, 0.59), ("spurious 1s D", 0.64, 0.97),
    ("spurious 0.5s opponent", 0.55, 0.99), ("turnover flips 12", 0.91, None),
    ("turnover flips 25", 0.82, None), ("turnover flips 50", 0.68, None),
]


def test_sim_costs_reproduce_plan_table():
    out = subprocess.run([sys.executable, os.path.join(registry.ROOT, "scripts", "sim_costs.py")],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    for prefix, spell, macro in EXPECTED_COSTS:
        line = next(l for l in out if l.startswith(prefix))
        s, m, _ = eval(line.split(":", 1)[1])  # tuple printed by the script
        assert round(s, 2) == pytest.approx(spell, abs=0.006), line
        if macro is not None:
            assert round(m, 2) == pytest.approx(macro, abs=0.006), line
