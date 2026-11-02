"""The dead_runs release path (deadball.restart_window_frames) after the B6 fixes: a release is
a touch that STARTED inside the evidence, stale releases never survive a run merge, and merged
runs keep both runs' evidence kinds. All dormant in shipped configs (W null) - these tests pin
the semantics before any release-restart candidate is measured."""

import numpy as np

from mtah.events.deadball import dead_runs
from mtah.events.touches import Touch


def touch(start, end, team=0, speed_after=10.0):
    return Touch(start=start, end=end, team=team, n_frames=end - start + 1, d_min=0.5,
                 speed_before=np.nan, speed_after=speed_after, turn_deg=np.nan)


def setup(T, windows):
    graded = np.ones(T, dtype=bool)
    oob = np.zeros(T, dtype=bool)
    for a, b in windows:
        oob[a:b + 1] = True
    ev = {"oob": oob, "gap": np.zeros(T, bool), "still": np.zeros(T, bool),
          "flight": np.zeros(T, bool)}
    return ev, graded


def cfg(W=None, v_go=0.0, merge=2):
    return {"merge_frames": merge, "restart_window_frames": W, "restart_min_speed_mps": v_go}


def test_slow_post_evidence_touch_is_not_a_release():
    # restart_min_speed skips a slow touch AFTER the evidence; it must not then qualify as a
    # release (it did before the fix: the slice ran to the advanced touch index).
    ev, graded = setup(200, [(50, 60)])
    touches = [touch(65, 70, speed_after=1.0), touch(90, 95, speed_after=10.0)]
    restarts = []
    out = dead_runs(ev, touches, graded, cfg(W=6, v_go=4.0), ("oob",), out_restarts=restarts)
    assert out == [(50, 89, ("oob",))]      # ends at the fast touch, not the slow one
    assert restarts == []                    # the slow touch started outside the evidence


def test_release_inside_evidence_still_fires():
    ev, graded = setup(200, [(50, 60)])
    touches = [touch(55, 58), touch(100, 105)]
    restarts = []
    out = dead_runs(ev, touches, graded, cfg(W=6), ("oob",), out_restarts=restarts)
    assert out == [(50, 58, ("oob",))]      # taker's touch started inside the oob evidence
    assert len(restarts) == 1 and restarts[0].start == 59 and restarts[0].n_frames == 1


def two_window_setup():
    T = 300
    graded = np.ones(T, dtype=bool)
    oob = np.zeros(T, bool)
    oob[50:61] = True
    gap = np.zeros(T, bool)
    gap[90:121] = True
    ev = {"oob": oob, "gap": gap, "still": np.zeros(T, bool), "flight": np.zeros(T, bool)}
    return ev, graded


def test_absorbed_runs_release_is_dropped_and_kinds_merge():
    # Run 1 (oob 50-60) ends at the first FAST touch (140): the slow taker at 115 is skipped by
    # restart_min_speed. Run 2 (gap 90-120) starts before that end, so it is absorbed; its taker
    # release (the slow touch inside the gap evidence, restart 118) lies INSIDE the merged run
    # and must not be reported (before the fix it was appended before the merge check). The
    # merged run keeps both evidence kinds.
    ev, graded = two_window_setup()
    touches = [touch(115, 117, speed_after=1.0), touch(140, 145, speed_after=10.0)]
    restarts, meta = [], []
    out = dead_runs(ev, touches, graded, cfg(W=6, v_go=4.0), ("oob", "gap"),
                    out_restarts=restarts, out_meta=meta)
    assert out == [(50, 139, ("oob", "gap"))]   # one merged run, both kinds kept
    assert restarts == []                        # the inside release never left the run
    assert meta == [{"start": 50, "end": 139, "evidence_end": 120}]


def test_absorbing_run_whose_release_extends_the_merge_keeps_it():
    # Same geometry without the speed gate: the touch at 115 is run 1's restart (end 114), run 2
    # absorbs and its release (ball leaves the taker at 119) EXTENDS the merged run - that
    # release is the merged run's real restart and is kept.
    ev, graded = two_window_setup()
    touches = [touch(115, 118), touch(130, 135)]
    restarts, meta = [], []
    out = dead_runs(ev, touches, graded, cfg(W=6), ("oob", "gap"), out_restarts=restarts,
                    out_meta=meta)
    assert out == [(50, 118, ("oob", "gap"))]
    assert len(restarts) == 1 and restarts[0].start == 119
    assert meta == [{"start": 50, "end": 118, "evidence_end": 120}]


def test_w_null_path_untouched():
    ev, graded = setup(200, [(50, 60)])
    touches = [touch(90, 95)]
    restarts = []
    out = dead_runs(ev, touches, graded, cfg(W=None), ("oob",), out_restarts=restarts)
    assert out == [(50, 89, ("oob",))] and restarts == []
