"""The fake bundle is in the real formats, and our loader reads it exactly like kloppy does."""

import warnings

import numpy as np

from mtah import registry
from mtah.io.atd import _group, load_atd
from mtah.io.tags import load_tags


def test_raw_contract(fake, fake_atd):
    atd, stats = fake_atd
    sp = registry.spec(fake["game"], fake["data_dir"])
    assert stats["min_frame_id"] == 1 and stats["max_frame_id"] == sp["n_frames"]
    assert stats["n_bad_lines"] == 0 and stats["n_slot_mismatch"] == 0
    assert atd.n_slots == 46 and atd.n_frames == sp["n_frames"]
    assert list(atd.team_of_slot[:23]) == [0] * 23 and list(atd.team_of_slot[23:]) == [1] * 23


def test_kloppy_agrees(fake, fake_atd):
    from kloppy import metrica
    atd, _ = fake_atd
    meta, raw = registry.require(["atd_meta", "atd_raw"], fake["game"], fake["data_dir"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = metrica.load_tracking_epts(meta_data=meta, raw_data=raw)
    assert len(ds.records) == atd.n_frames
    assert str(ds.metadata.teams[0]) == atd.team_names[0]
    df = ds.to_df()
    assert np.allclose(df["ball_x"].values, atd.ball[:, 0], equal_nan=True)
    pid = ds.metadata.teams[0].players[3].player_id
    assert np.allclose(df["{}_x".format(pid)].values, atd.xy[:, 3, 0], equal_nan=True)


def test_parser_edge_cases(tmp_path):
    assert all(np.isnan(v) for v in _group("NaN"))
    meta = tmp_path / "m.xml"
    meta.write_text('<main><Teams><Team id="a"><Name>H</Name></Team><Team id="b"><Name>A</Name>'
                    '</Team></Teams><Players><Player teamId="a"/><Player teamId="b"/></Players></main>')
    raw = tmp_path / "r.txt"
    raw.write_text("1:0.5,0.5,1.0;NaN,NaN,NaN:0.5,0.5,0.0\n2:NaN;0.1,0.2,3.0:NaN,NaN,NaN\nbroken\n")
    atd, stats = load_atd(str(meta), str(raw), n_frames=2)
    assert stats["n_bad_lines"] == 1
    assert atd.ball_present().tolist() == [True, False]
    assert np.isnan(atd.xy[1, 0, 0]) and atd.xy[1, 1, 1] == 0.2


def test_tags_time_base(fake):
    path = registry.require(["smart_tagging"], fake["game"], fake["data_dir"])[0]
    tags = load_tags(path)
    kicks = [t for t in tags if t.is_code("set pieces") and t.is_type("kick off (start)")]
    assert len(kicks) == 2
    for t in kicks:
        assert t.start <= fake["meta"]["kickoffs"][str(1 if t.get("Half") == "1st Half" else 2)] <= t.end


def test_gt_contract(fake):
    from mtah.eval.score import codes_of, pm
    sp = registry.spec(fake["game"], fake["data_dir"])
    gt = codes_of(fake["data_dir"] + "/gt.csv")
    graded = registry.graded_mask(sp)
    assert len(gt) == sp["n_frames"]
    assert np.all(gt[~graded] == pm.X) and not np.any(gt[graded] == pm.X)
    runs = pm.valid_runs(gt)
    assert [(a, b - 1) for a, b in runs] == [(f, l) for _, f, l in sp["half_blocks"]]
