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
    assert {h: r["frame"] for h, r in ko.items()} == truth
    d = attacking_direction(atd, truth, cfg["direction"]["window_frames"])
    for half, home in fake["meta"]["home_attack_dir"].items():
        assert d[int(half)]["H"] == home and d[int(half)]["A"] == -home
