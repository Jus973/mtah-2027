import numpy as np
import pytest

from mtah.learn import takeover


class E:
    def __init__(self, start):
        self.start = start


def test_weights_off_by_default():
    gt = np.array(list("HHHHAAAA"))
    assert np.array_equal(takeover.possession_weights([E(4)], np.array([0]), gt, {}), [1.0])


def test_weights_follow_interrupted_spell_and_clip():
    gt = np.array(["H"] * 50 + ["A"] * 400 + ["D"] * 10 + ["H"] * 10)
    events = [E(50), E(450), E(460)]
    w = takeover.possession_weights(events, np.array([0, 1, 2]), gt,
                                    {"possession_weight_s": 2.0,
                                     "possession_weight_clip": [0.25, 4.0]}, fps=25)
    # row 0 interrupts the 2 s H spell -> 1.0; row 1 the 16 s A spell -> 0.125, clipped to 0.25;
    # row 2 opens a stretch after D -> 1.0
    assert np.allclose(w, [1.0, 0.25, 1.0])


def test_train_refuses_flag_without_weights():
    params = {"num_boost_round": 2, "possession_weight_s": 8.0}
    with pytest.raises(ValueError):
        takeover.train(np.zeros((4, 1)), np.array([0.0, 1, 0, 1]), params, ("a",))
