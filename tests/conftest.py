import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fakedata"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

GAME = "BAR-FAKE"


@pytest.fixture(scope="session")
def fake(tmp_path_factory):
    """A small synthetic match (2 x 5 min) generated fresh for the test session."""
    import generate
    root = tmp_path_factory.mktemp("fake")
    out, meta = generate.generate(GAME, seed=7, half_s=300.0, out_root=str(root))
    return {"game": GAME, "data_dir": out, "meta": meta}


@pytest.fixture(scope="session")
def fake_atd(fake):
    from mtah.io.atd import load_game
    atd, stats = load_game(fake["game"], fake["data_dir"])
    return atd, stats
