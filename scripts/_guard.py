"""Shared setup for real-data scripts: arguments, the real-data gate, and the report.

Scripts emit everything through put() / check(). In a terminal these print as before; under
scripts/airlock.py they are collected as JSON and validated before an agent sees them.
"""

import os
import sys

from mtah import registry
from mtah.airlock import OUT_ENV, Report

BANNER = ("[aggregates only] This script reads the match bundle and prints counts, scores and "
          "per-event scalars. Paste back only what it prints.")

REPORT = Report()
put, check = REPORT.put, REPORT.check


def game_args(argv=None, add=None):
    """--game / --data-dir (plus whatever `add(parser)` declares), then the real-data gate."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="BAR-ATH", help="BAR-ATH (real) or a fakedata game")
    parser.add_argument("--data-dir", default=None)
    if add:
        add(parser)
    args = parser.parse_args(argv)
    if not registry.is_fake(args.game, args.data_dir):
        if not (args.data_dir or os.environ.get("MTAH_DATA_DIR")):
            sys.exit("MTAH_DATA_DIR is not set. export MTAH_DATA_DIR=/path/to/mtah2027")
        try:
            registry.gate(args.game, args.data_dir)
        except PermissionError as err:
            sys.exit(str(err))
    if not os.environ.get(OUT_ENV):
        print(BANNER)
    return args
