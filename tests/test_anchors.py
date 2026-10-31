"""Kick-off detection and attacking direction recover the synthetic truth."""

from mtah import config, registry
from mtah.events.restarts import detect_kickoffs
from mtah.features.context import attacking_direction
from mtah.io.tags import load_tags


def test_kickoffs_and_direction(fake, fake_atd):
    atd, _ = fake_atd
    cfg = config.load()
    tags = load_tags(registry.require(["smart_tagging"], fake["game"], fake["data_dir"])[0])
    ko = detect_kickoffs(atd, tags, cfg["kickoff"])
    truth = {int(k): v for k, v in fake["meta"]["kickoffs"].items()}
    tol = cfg["kickoff"]["tolerance_frames"]
    for half, frame in truth.items():
        got = ko[half]["frame"]
        assert ko[half]["method"] == "formation+ball"
        assert abs(got - frame) <= tol
        # On the synthetic match the first pass moves the ball < rest_slack_m in one frame, so
        # the detector can land one frame late; anything worse is a regression.
        assert 0 <= got - frame <= 1, (half, got, frame)
    d = attacking_direction(atd, {h: r["frame"] for h, r in ko.items()},
                            cfg["direction"]["window_frames"])
    for half, home in fake["meta"]["home_attack_dir"].items():
        assert d[int(half)]["H"] == home and d[int(half)]["A"] == -home
