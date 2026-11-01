"""Per-frame homography (normalised pitch 0-1 -> video pixels) for any registered match.

Same file format and semantics as the vendored starter/load_homography.py (which only knows the
official game): frame, h00..h22, reprojection_error, valid; matrices are NaN where valid is 0.
"""

import numpy as np
import pandas as pd

from mtah import registry

COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]


def load(game, data_dir=None, source="tactical"):
    """(matrices (n, 3, 3), reprojection_error (n,), valid (n,) bool)."""
    path = registry.require(["homography"], game, data_dir, source)[0]
    table = pd.read_csv(path)
    return (table[COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 3, 3),
            table["reprojection_error"].to_numpy(), table["valid"].to_numpy().astype(bool))


def project(matrix, x, y):
    """Normalised pitch coordinates -> pixels (NaN in, NaN out)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return ((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / w,
            (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / w)
