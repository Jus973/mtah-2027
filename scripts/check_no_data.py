"""Pre-commit guard: refuse to commit anything that looks like match data or a derivative."""

import fnmatch
import os
import sys

BLOCKED_DIRS = ("data/", "cache/", "validation/", "submissions/candidates/", "fakedata/out/")
BLOCKED_GLOBS = ("*.mp4", "*.npz", "*.parquet", "*.mkv", "*.wav", "*_atd.txt", "*_atd.xml",
                 "*_homography.csv", "*_smart_tagging.xml")
MAX_BYTES = 5 * 1024 * 1024
ALLOW_LARGE = ("submissions/final/*.csv",)
ALLOW_PATTERN = ("evaluation/*",)  # vendored official files (public GT + baselines)


def problems(path):
    norm = path.replace(os.sep, "/")
    if any(norm.startswith(d) for d in BLOCKED_DIRS):
        return "blocked directory"
    if any(fnmatch.fnmatch(norm, g) for g in ALLOW_PATTERN):
        return None
    if any(fnmatch.fnmatch(os.path.basename(norm), g) for g in BLOCKED_GLOBS):
        return "blocked file type"
    if os.path.isfile(path) and os.path.getsize(path) > MAX_BYTES and \
            not any(fnmatch.fnmatch(norm, g) for g in ALLOW_LARGE):
        return "file > 5 MB"
    return None


if __name__ == "__main__":
    bad = [(p, problems(p)) for p in sys.argv[1:] if problems(p)]
    for p, why in bad:
        print("refusing to commit {}: {}".format(p, why))
    sys.exit(1 if bad else 0)
