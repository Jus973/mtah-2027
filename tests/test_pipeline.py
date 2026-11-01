"""Rules v1 end to end (mtah.pipeline), Smart-Tagging consistency, kick-off band tolerance."""

import numpy as np

from mtah import config, pipeline, registry
from mtah.eval.bands import bands
from mtah.eval.consistency import implied_side, tag_consistency
from mtah.io.tags import Tag

CFG = config.load("rules.yaml")


def test_pipeline_on_the_fake_match(fake, fake_atd):
    atd, _ = fake_atd
    sp = registry.spec(fake["game"], fake["data_dir"])
    a, info = pipeline.run(fake["game"], CFG, fake["data_dir"], atd=atd)
    b, _ = pipeline.run(fake["game"], CFG, fake["data_dir"], atd=atd)
    np.testing.assert_array_equal(a, b)                       # deterministic
    graded = registry.graded_mask(sp)
    assert len(a) == sp["n_frames"] and set(a[~graded]) == {"X"}
    assert "X" not in set(a[graded]) and set(a[graded]) <= {"H", "A", "D"}
    assert sorted(info["kickoffs"]) == [1, 2]
    # The second-half block starts before its kick-off: those frames are dead.
    (_, f2, _), k2 = sp["half_blocks"][1], info["kickoffs"][2]
    assert k2 - f2 > CFG["kickoff"]["tolerance_frames"] and (a[f2:k2] == "D").all()


def tag(code, team, start, end):
    return Tag(code=code, start=start, end=end, tags={"Team": team})


def test_tag_consistency():
    team_map = CFG["tag_team_map"]
    assert implied_side(tag("BUILD UP", "Barcelona", 0, 1), team_map) == "H"
    assert implied_side(tag("DEFENDING IN MIDDLE THIRD", "Barcelona", 0, 1), team_map) == "A"
    assert implied_side(tag("SET PIECES", "Barcelona", 0, 1), team_map) is None
    assert implied_side(tag("BUILD UP", "N/A", 0, 1), team_map) is None
    labels = np.array(list("HHHHAAAADD"))
    graded = np.ones(10, bool)
    tags = [tag("BUILD UP", "Barcelona", 0, 5),                  # H H H H A A -> 4/6
            tag("DEFENSIVE TRANSITION", "Barcelona", 6, 9)]      # A A (D D) -> 2/2
    c = tag_consistency(labels, tags, team_map, graded)
    assert c["frames"] == 8 and abs(c["agree"] - 6 / 8) < 1e-12
    assert c["by_code"]["BUILD UP"] == [4 / 6, 6]


def test_bands_kickoff_tolerance(fake):
    sp = registry.spec(fake["game"], fake["data_dir"])
    gt = np.loadtxt(fake["data_dir"] + "/gt.csv", delimiter=",", skiprows=1, dtype=str)[:, 1]
    kick = {int(k): v for k, v in fake["meta"]["kickoffs"].items()}
    team = {int(k): v for k, v in fake["meta"]["kickoff_team"].items()}
    late = gt.copy()
    late[kick[2]:kick[2] + 4] = "D"                             # kick-off detected 4 frames late
    strict = bands(late, sp, CFG["bands"], kick, team)["checks"]
    loose = bands(late, sp, CFG["bands"], kick, team, tol=25)["checks"]
    assert not strict["kickoff_team_half2"] and loose["kickoff_team_half2"]
    assert loose["pre_kickoff_D_half2"]
