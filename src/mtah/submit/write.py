"""Write and validate submission CSVs. Format identical to starter/baseline.py's writer."""

import hashlib
import os

from mtah.eval.score import pm


def write_csv(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:   # byte-identical CSVs on every platform
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def validate(path, n_rows):
    """None if valid (official checks), else the one-line reason."""
    return pm.validate_submission(pm.read_label_csv(path), n_rows)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
