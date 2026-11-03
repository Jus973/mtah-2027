import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from oracle_owner import cell_table, turnovers, with_truth  # noqa: E402


def ev(team, start, end):
    return SimpleNamespace(team=team, start=start, end=end)


def test_turnovers_only_alive_to_alive_switches():
    gt = np.array(list("HHHAAADDDHHA"))
    graded = np.ones(len(gt), bool)
    assert list(turnovers(gt, graded)) == [3, 11]
    graded[11] = False
    assert list(turnovers(gt, graded)) == [3]


def test_with_truth_replaces_only_masked_frames():
    lab, gt = np.array(list("HHDA")), np.array(list("AADH"))
    out = with_truth(lab, gt, np.array([True, False, False, True]))
    assert "".join(out) == "AHDH" and "".join(lab) == "HHDA"


def test_cell_table_contact_gap_and_recall():
    gt = np.array(list("H" * 50 + "A" * 50))
    q = SimpleNamespace(m=SimpleNamespace(gt=gt, graded=np.ones(100, bool)),
                        events=[ev(1, 52, 54), ev(0, 10, 12)])
    gap = np.zeros(100, bool)
    lab = np.array(list("H" * 55 + "A" * 45))
    assert cell_table(q, lab, gap, 25) == [(True, False, True)]
    q.events = [ev(0, 52, 54)]
    gap[40:60] = True
    lab = np.array(list("H" * 100))
    assert cell_table(q, lab, gap, 25) == [(False, True, False)]


def test_cell_table_one_base_switch_recalls_one_turnover():
    gt = np.array(list("H" * 60 + "A" * 10 + "H" * 10 + "A" * 40))
    q = SimpleNamespace(m=SimpleNamespace(gt=gt, graded=np.ones(120, bool)), events=[])
    lab = np.array(list("H" * 75 + "A" * 45))
    hits = [r[2] for r in cell_table(q, lab, np.zeros(120, bool), 25)]
    assert hits.count(True) == 1 and len(hits) == 3


def test_cell_table_matching_is_maximum_for_overlapping_windows():
    gt = np.array(list("H" * 60 + "A" * 10 + "H" * 10 + "A" * 40))
    q = SimpleNamespace(m=SimpleNamespace(gt=gt, graded=np.ones(120, bool)), events=[])
    lab = np.array(list("H" * 40 + "A" * 10 + "H" * 15 + "A" * 55))
    hits = [r[2] for r in cell_table(q, lab, np.zeros(120, bool), 25)]
    assert hits == [True, True, True]
