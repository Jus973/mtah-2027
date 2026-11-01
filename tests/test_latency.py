"""Transition timing: offsets of matched transitions and the two latency corrections."""

import numpy as np

from mtah.decode.latency import gap_flips, offsets, shift_onsets, transitions


def test_transitions_and_offsets():
    gt = np.array(list("XHHHHAAAADDDHHX"))
    pred = np.array(list("XHHHHHAAADDDDHX"))      # turnover 1 late, D end 1 late
    assert [(f, k) for f, k, _ in transitions(gt)] == [(5, "turnover"), (9, "d_onset"),
                                                       (12, "d_offset")]
    o = offsets(pred, gt, window=3)
    assert o["turnover"].tolist() == [1] and o["d_onset"].tolist() == [0]
    assert o["d_offset"].tolist() == [1]


def test_shift_onsets_stops_at_the_previous_change():
    labels = np.array(list("XHHAADDDH"))
    assert "".join(shift_onsets(labels, 1)) == "XHHADDDDH"
    assert "".join(shift_onsets(labels, 5)) == "XHHDDDDDH"   # not past the H -> A change
    assert "".join(shift_onsets(labels, 0)) == "".join(labels)


def test_gap_flips_moves_a_takeover_into_the_gap():
    labels = np.array(list("HHHHHHHHHHAAAA"))
    missing = np.zeros(14, bool)
    missing[2:8] = True                            # gap 2-7, ball seen again at 8, flip at 10
    assert "".join(gap_flips(labels, missing, min_gap=4, max_after=2)) == "HHHHHAAAAAAAAA"
    assert "".join(gap_flips(labels, missing, min_gap=4, max_after=1)) == "".join(labels)
    assert "".join(gap_flips(labels, missing, min_gap=7, max_after=2)) == "".join(labels)
