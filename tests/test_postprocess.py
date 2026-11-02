"""mtah.postprocess (the v9 label rules) and the paired-harness section whitelist."""

import os
import sys

import numpy as np
import pytest

from mtah import config, pipeline, postprocess, registry

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

CFG = config.load("rules.yaml")


def lab(s):
    return np.array(list(s), dtype="<U1")


def test_disabled_or_missing_is_an_unchanged_copy():
    x = lab("HHHDDAAA")
    for cfg in (None, {}, {"enabled": False, "d_flicker_max_frames": 99}):
        out = postprocess.apply(x, cfg)
        np.testing.assert_array_equal(out, x)
        out[0] = "A"                                     # a copy, never a view
        assert x[0] == "H"


def test_dflick_deletes_short_same_team_d_only():
    cfg = {"enabled": True, "d_flicker_max_frames": 5}
    assert "".join(postprocess.apply(lab("HHHDDHHH"), cfg)) == "HHHHHHHH"
    assert "".join(postprocess.apply(lab("HHHDDAAA"), cfg)) == "HHHDDAAA"   # different teams
    assert "".join(postprocess.apply(lab("HHDDDDDHH"), cfg)) == "HHDDDDDHH"  # >= 5 frames
    assert "".join(postprocess.apply(lab("XXXDDHHH"), cfg)) == "XXXDDHHH"   # X is not a spell
    protect = np.zeros(8, bool)
    protect[3] = True
    assert "".join(postprocess.apply(lab("HHHDDHHH"), cfg, protect=protect)) == "HHHDDHHH"


def test_dend_uniform_shifts_every_d_to_alive_boundary():
    cfg = {"enabled": True, "restart_backshift_frames": 2}
    assert "".join(postprocess.apply(lab("HHHDDDDDAAA"), cfg)) == "HHHDDDAAAAA"
    assert "".join(postprocess.apply(lab("HHHDDDDD"), cfg)) == "HHHDDDDD"   # no next spell
    assert "".join(postprocess.apply(lab("AADDDXHH"), cfg)) == "AADDDXHH"   # X follows the run
    big = {"enabled": True, "restart_backshift_frames": 99}
    assert "".join(postprocess.apply(lab("HHDDDAA"), big)) == "HHAAAAA"     # capped at the start


def test_dend_conditional_shifts_late_runs_only():
    cfg = {"enabled": True, "restart_backshift_frames": 4, "backshift_min_late_frames": 2}
    x = lab("HHHDDDDDDDAA")
    late = [{"start": 3, "end": 9, "evidence_end": 4}]          # 5 frames late
    assert "".join(postprocess.apply(x, cfg, late)) == "HHHDDDAAAAAA"
    on_time = [{"start": 3, "end": 9, "evidence_end": 8}]       # 1 frame late <= m
    assert "".join(postprocess.apply(x, cfg, on_time)) == "HHHDDDDDDDAA"
    capped = [{"start": 3, "end": 9, "evidence_end": 6}]        # 3 late < k: shift 3, not 4
    assert "".join(postprocess.apply(x, cfg, capped)) == "HHHDDDDAAAAA"
    # The decoder already ended or moved the run: nothing to shift.
    moved = lab("HHHDDDDAAAAA")
    assert "".join(postprocess.apply(moved, cfg, late)) == "HHHDDDDAAAAA"
    # Stays inside the containing D segment even when the metadata spans re-decoded frames.
    patchy = lab("HHHDDHADDDAA")
    meta = [{"start": 3, "end": 9, "evidence_end": 3}]
    assert "".join(postprocess.apply(patchy, cfg, meta)) == "HHHDDHAAAAAA"
    with pytest.raises(ValueError):
        postprocess.apply(x, cfg, None)


def test_dflick_runs_before_dend_and_stats_count():
    cfg = {"enabled": True, "d_flicker_max_frames": 4, "restart_backshift_frames": 2}
    stats = {}
    out = postprocess.apply(lab("HHHDDHHHDDDDDAAA"), cfg, out_stats=stats)
    assert "".join(out) == "HHHHHHHHDDDAAAAA"
    assert stats == {"dflick_runs": 1, "dflick_frames": 2, "dend_runs": 1, "dend_frames": 2}


def test_pipeline_off_identity_and_prekick_protection(fake, fake_atd):
    atd, _ = fake_atd
    base, _ = pipeline.run(fake["game"], CFG, fake["data_dir"], atd=atd)
    off = dict(CFG, postprocess={"enabled": False, "d_flicker_max_frames": 50,
                                 "restart_backshift_frames": 15})
    labels, _ = pipeline.run(fake["game"], off, fake["data_dir"], atd=atd)
    np.testing.assert_array_equal(labels, base)          # disabled = byte-identical
    on = dict(CFG, postprocess={"enabled": True, "d_flicker_max_frames": 50,
                                "restart_backshift_frames": 15})
    labels, info = pipeline.run(fake["game"], on, fake["data_dir"], atd=atd)
    sp = registry.spec(fake["game"], fake["data_dir"])
    graded = registry.graded_mask(sp)
    assert set(labels[~graded]) == {"X"} and "X" not in set(labels[graded])  # no-X invariant
    assert (labels[labels != base] != "D").all()         # rules only turn D into a team
    # The pre-kick-off block stays D to the kick-off: the derived kick-off is never rewritten.
    (_, f2, _), k2 = sp["half_blocks"][1], info["kickoffs"][2]
    assert (labels[f2:k2] == "D").all()
    assert "postprocess" in info and info["postprocess"]["dend_runs"] >= 1


def test_paired_refuses_unapplied_sections():
    import paired
    cfg = config.load("final.yaml")
    paired.check_sections(cfg, "final.yaml")             # every section applied or inert
    paired.check_sections(config.load("rules_v9.yaml"), "rules_v9.yaml")
    with pytest.raises(SystemExit, match="novel_section"):
        paired.check_sections(dict(cfg, novel_section={"enabled": True}), "x.yaml")
