"""make sim-latency [CFG=configs/final.yaml]: transition timing residuals and the two latency
corrections (decode/latency.py), calibrated on the sim's fit matches, judged on held-out.

Prints, per transition type, the median offset of matched transitions (pred - GT, within 2 s) on
fit and held-out matches; then held-out spellF1 for the config as is, with D onsets moved earlier
by the fit median, and with takeovers after long ball gaps moved into the gap (settings tuned on
fit). PLAN.md M3: a correction is kept only if it adds >= 0.01 on held-out and helps both
matches. Open data only.
"""

import argparse
import os

import numpy as np

from mtah import config, pipeline, registry
from mtah.decode.latency import KINDS, gap_flips, offsets, shift_onsets
from mtah.eval.simeval import SimMatch

WINDOW = 50
GAP_GRID = [(g, a) for g in (25, 50, 100) for a in (0, 3, 12)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    cfg = config.load(os.path.abspath(parser.parse_args().config))
    runs = {}
    for split in ("fit", "holdout"):
        for mid in cfg["matches"][split]:
            game = "DFL-" + mid
            m = SimMatch(game, cfg)
            labels, _ = pipeline.run(game, cfg)
            missing = ~np.isfinite(m.ball[:, 0]) & m.graded
            runs[mid] = (split, m, labels, missing)

    med = {}
    for split in ("fit", "holdout"):
        allo = {k: [] for k in KINDS}
        for mid, (s, m, labels, _) in runs.items():
            if s == split:
                for k, v in offsets(labels, m.gt, WINDOW).items():
                    allo[k].extend(v)
        for k in KINDS:
            v = np.array(allo[k])
            print("{:8} {:9} matched {:5d}  median {:+5.1f}  p25/p75 {:+.0f}/{:+.0f}".format(
                split, k, len(v), np.median(v), *np.percentile(v, [25, 75])))
            if split == "fit":
                med[k] = float(np.median(v))

    def mean_score(split, fn):
        return [m.evaluate(fn(m, labels, missing))["spell_f1"]
                for s, m, labels, missing in runs.values() if s == split]

    k = max(0, int(round(med["d_onset"])))
    best = max(GAP_GRID, key=lambda ga: np.mean(mean_score(
        "fit", lambda m, l, miss: gap_flips(l, miss, ga[0], ga[1]))))
    rows = [("as is", lambda m, l, miss: l),
            ("D onsets {} frames earlier".format(k), lambda m, l, miss: shift_onsets(l, k)),
            ("gap flips (min gap {}, within {})".format(*best),
             lambda m, l, miss: gap_flips(l, miss, best[0], best[1]))]
    base = None
    print("\nheld-out ({}):".format(", ".join(cfg["matches"]["holdout"])))
    for name, fn in rows:
        sc = mean_score("holdout", fn)
        base = sc if base is None else base
        print("  {:<36} {}  mean {:.4f}  vs as is {:+.4f}, both matches better: {}".format(
            name, " ".join("{:.4f}".format(x) for x in sc), np.mean(sc),
            np.mean(sc) - np.mean(base), all(x > y for x, y in zip(sc, base))))


if __name__ == "__main__":
    main()
