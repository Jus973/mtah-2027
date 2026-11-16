"""GT-convention worlds behind scripts/regret_matrix.py (board #8 D-end axis, #46 axes)."""

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from regret_matrix import (AXES, check_world, merge_short, shift_dends,  # noqa: E402
                           shift_dstarts, shift_switches)
from mtah.eval.score import pm  # noqa: E402

FPS = 25


def arr(spec):
    """'H5 A3 D2 X4 ...' -> label array."""
    out = []
    for tok in spec.split():
        out += [tok[0]] * int(tok[1:])
    return np.array(out, dtype=str)


def spec(labels):
    """Inverse of arr(): run-length spec over the whole array, X gaps included."""
    codes = pm.label_codes(labels)
    runs = [(0, len(codes))] if len(codes) else []
    return " ".join("{}{}".format(pm.LABELS[c], b - a) for a, b, c in pm.segments(codes, runs))


def random_labels(seed, n=600):
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        out += [rng.choice(["H", "A", "D", "X"], p=[0.4, 0.4, 0.15, 0.05])] * rng.randint(1, 40)
    return np.array(out[:n], dtype=str)


def test_identities():
    gt = arr("H5 A3 D2 H4 X3 D2 A6 H1")
    for axis, (_, _, transform, grid, _) in AXES.items():
        zero = transform(gt, 0, FPS)
        assert zero is not gt and (zero == gt).all(), axis
        assert 0 in grid


def test_shift_switches_direction_and_cap():
    gt = arr("H5 A5")
    assert spec(shift_switches(gt, 2)) == "H7 A3"
    assert spec(shift_switches(gt, -2)) == "H3 A7"
    assert spec(shift_switches(arr("H5 A2"), 5)) == "H6 A1"     # the later spell keeps 1 frame
    assert spec(shift_switches(arr("H2 A5"), -5)) == "H1 A6"    # the earlier spell keeps 1 frame


def test_shift_switches_ignores_boundaries_through_d_and_runs():
    gt = arr("H5 D2 A5 X2 A3 H3")
    out = shift_switches(gt, 2)
    assert spec(out) == "H5 D2 A5 X2 A5 H1"                     # only the direct A->H moved
    assert (out[:12] == gt[:12]).all()


def test_shift_switches_chain_of_short_spells_preserves_counts():
    gt = arr("H50 A3 H50 A2 H50")
    out = shift_switches(gt, 5)
    assert spec(out) == "H52 A6 H46 A6 H45"                     # every move from the originals
    for k in (-10, -5, 5, 10, 25):
        check_world("switch", pm.label_codes(gt), pm.label_codes(shift_switches(gt, k)), "t")


def test_shift_dstarts_direction_and_cap():
    gt = arr("H5 D5 A3")
    assert spec(shift_dstarts(gt, 2)) == "H7 D3 A3"
    assert spec(shift_dstarts(gt, -2)) == "H3 D7 A3"
    assert spec(shift_dstarts(arr("H5 D10 A5"), 25)) == "H14 D1 A5"   # 10-frame D run keeps 1
    assert spec(shift_dstarts(arr("H2 D10 A5"), -25)) == "H1 D11 A5"  # the spell keeps 1
    assert spec(shift_dstarts(arr("D3 H3"), 10)) == "D3 H3"           # no preceding spell
    assert spec(shift_dstarts(arr("H3 X2 D3 H3"), -2)) == "H3 X2 D3 H3"   # X is not a spell


def test_shift_dstarts_leaves_d_ends_alone():
    gt = arr("H5 D5 A5")
    for k in (-3, 3):
        out = shift_dstarts(gt, k)
        assert (out[10:] == gt[10:]).all()
        assert (shift_dends(gt, 3)[:10] == gt[:10]).all()


def test_merge_short_chain_fixed_point():
    gt = arr("A50 H20 A10 H20 A50")
    assert spec(merge_short(gt, 2, FPS)) == "A50 H50 A50"       # shortest first, then stop
    assert spec(merge_short(gt, 1, FPS)) == "A50 H50 A50"       # A10 < 25 merges, H50 does not
    assert spec(merge_short(arr("A50 H5 A3 H5 A50"), 2, FPS)) == "A113"


def test_merge_short_needs_a_true_sandwich():
    assert spec(merge_short(arr("A50 D5 H10 A50"), 2, FPS)) == "A50 D5 H10 A50"   # D between
    assert spec(merge_short(arr("H10 A50 H50"), 2, FPS)) == "H10 A50 H50"          # run edge
    assert spec(merge_short(arr("A50 H10 X3 A50"), 2, FPS)) == "A50 H10 X3 A50"    # X between
    assert spec(merge_short(arr("A50 H50 A50"), 2, FPS)) == "A50 H50 A50"          # not short


def test_random_worlds_pass_the_structural_checks():
    for seed in range(20):
        gt = random_labels(seed)
        base = pm.label_codes(gt)
        for axis, (_, _, transform, grid, _) in AXES.items():
            for k in grid:
                out = transform(gt, k, FPS)
                if k:
                    check_world(axis, base, pm.label_codes(out), str(seed))
                else:
                    assert (out == gt).all()


def test_regret_cache_reuses_then_rebuilds_on_source_edits(tmp_path, monkeypatch):
    import regret_matrix as matrix

    game = "TEST-SIM"
    root = tmp_path / game
    root.mkdir()
    monkeypatch.setattr(matrix.registry, "SIM_DIR", str(tmp_path))
    monkeypatch.setattr(matrix, "CACHE", str(tmp_path / "labels.pkl"))
    monkeypatch.setattr(matrix, "PRIMARY_EXTRA", (game,))
    monkeypatch.setattr(matrix.config, "load", lambda path: {"matches": {"fit": [], "holdout": []}})
    (root / matrix.registry.FAKE_META).write_text("{}")
    paths = matrix.registry.asset_paths(game)
    raw, xml = Path(paths["atd_raw"]), Path(paths["atd_meta"])
    raw.parent.mkdir()
    gt = root / "gt.csv"
    for path in (raw, xml, gt):
        path.write_bytes(b"A")

    builds = []

    def build(extra, committed):
        builds.append(1)
        return {"build": len(builds)}

    monkeypatch.setattr(matrix, "build", build)

    def cached_build():
        return matrix.load_or_build([], False, False)["build"]

    assert cached_build() == cached_build() == 1
    for expected, path in enumerate((gt, raw, xml), start=2):
        path.write_bytes(b"B")
        assert cached_build() == cached_build() == expected

    tags = Path(paths["smart_tagging"])
    tags.write_bytes(b"A")
    assert cached_build() == 5
    tags.write_bytes(b"B")
    assert cached_build() == 6
    tags.unlink()
    assert cached_build() == 7


def test_regret_cache_rejects_extra_overwriting_cross_fitted_game(monkeypatch):
    import pytest
    import regret_matrix as matrix

    monkeypatch.setattr(matrix.config, "load", lambda path: {"matches": {"fit": ["X"], "holdout": []}})
    with pytest.raises(ValueError, match="cross-fitted"):
        matrix.fingerprint(["DFL-X"])
