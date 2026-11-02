"""decode/dead.py refine_runs (the P3e hybrid): evidence-anchored boundary refinement only."""

import numpy as np

from mtah.decode.dead import refine_runs


def test_no_signal_keeps_every_boundary():
    graded = np.ones(200, bool)
    p = np.full(200, np.nan)                      # no probability anywhere = no evidence
    runs = [(30, 60), (100, 140)]
    assert refine_runs(runs, p, graded, 25) == runs


def test_clear_signal_moves_an_end_and_a_start():
    graded = np.ones(300, bool)
    p = np.full(300, np.nan)
    p[30:50] = 0.95                               # dead until 49, alive after
    p[50:85] = 0.05
    out = refine_runs([(30, 60)], p, graded, 25)
    assert out == [(30, 49)]                      # the end comes forward to the changepoint
    p = np.full(300, np.nan)
    p[80:100] = 0.05                              # still alive at 80-99: the start moves back
    p[100:160] = 0.95
    out = refine_runs([(90, 160)], p, graded, 25)
    assert out == [(100, 160)]


def test_windows_never_cross_neighbours_or_the_graded_block():
    graded = np.zeros(200, bool)
    graded[20:180] = True
    p = np.full(200, 0.05)                        # "alive everywhere": maximal shrink pressure
    out = refine_runs([(25, 40), (50, 70)], p, graded, 100)
    (a1, b1), (a2, b2) = out
    assert a1 >= 20 and b2 <= 179                 # inside the graded block
    assert b1 < a2                                # runs stay ordered and disjoint
    assert b1 >= a1 and b2 >= a2                  # every run keeps at least one frame


def test_refinement_is_capped_at_the_window():
    graded = np.ones(400, bool)
    p = np.full(400, 0.95)                        # "dead everywhere": maximal extend pressure
    out = refine_runs([(100, 150)], p, graded, 25)
    assert out == [(75, 175)]                     # start - W, end + W, never further
