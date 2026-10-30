"""Shared guard for real-data scripts: aggregates only, and say so up front."""

import os
import sys

BANNER = ("[aggregates only] This script reads the match bundle and prints counts, scores and "
          "per-event scalars. Paste back only what it prints.")


def game_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="BAR-ATH", help="BAR-ATH (real) or a fakedata game")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    if args.game == "BAR-ATH" and not (args.data_dir or os.environ.get("MTAH_DATA_DIR")):
        sys.exit("MTAH_DATA_DIR is not set. export MTAH_DATA_DIR=/path/to/mtah2027")
    print(BANNER)
    return args


def check(name, ok, detail=""):
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", name, "  " + detail if detail else ""))
    return bool(ok)
