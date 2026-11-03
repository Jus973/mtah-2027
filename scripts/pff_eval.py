"""Board #48: frozen v6 vs v11 on the PFF WC2022 evaluation family (scripts/build_pff.py output).

Committed models only (trained on the DFL fit matches); nothing is fitted, tuned or selected on
PFF. Per-match spellF1 is cached under MTAH_PFF_DIR/eval (outside the repo) so `--shard k/n`
runs can split the work; every run then prints the per-match table and the paired summary
(mean, median, sign test) over whatever is cached. Aggregates only.
"""

import argparse
import json
import os
import sys

import numpy as np

from mtah import registry
from mtah.io import pff

sys.path.insert(0, os.path.join(registry.ROOT, "scripts"))
from oracle_receiver import sign_p                      # noqa: E402
from regret_matrix import V6, V11, committed_labels     # noqa: E402
from mtah.eval.score import score                       # noqa: E402

OUT = os.path.join(pff.OPEN_DIR, "eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="k/n: score every n-th match from k")
    ap.add_argument("--summary", action="store_true", help="only print what is cached")
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))
    os.makedirs(OUT, exist_ok=True)
    games = sorted(g for g in os.listdir(registry.SIM_DIR) if g.startswith("PFF-"))
    for g in ([] if args.summary else games[k::n]):
        path = os.path.join(OUT, g + ".json")
        if os.path.isfile(path):
            continue
        try:
            f = {os.path.basename(c): score(committed_labels(g, c), "sim:" + g)["spell_f1_mean"]
                 for c in (V6, V11)}
        except ValueError as e:
            print("FAILED", g, e, flush=True)
            continue
        with open(path, "w") as fh:
            json.dump(f, fh)
        print("scored", g, flush=True)
    rows = []
    print("{:<12}{:>10}{:>10}{:>10}".format("match", "v6", "v11", "d"))
    for g in games:
        path = os.path.join(OUT, g + ".json")
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            f = json.load(fh)
        r = (f[os.path.basename(V6)], f[os.path.basename(V11)])
        rows.append(r)
        print("{:<12}{:>10.4f}{:>10.4f}{:>+10.4f}".format(g, r[0], r[1], r[1] - r[0]))
    if not rows:
        raise SystemExit("nothing scored; run scripts/build_pff.py first")
    r = np.array(rows)
    d = r[:, 1] - r[:, 0]
    w, p = sign_p(d)
    print("n={} v6 {:.4f} v11 {:.4f}; d mean {:+.4f} median {:+.4f}; v11 wins {}/{} (sign p {:.4f})"
          .format(len(d), r[:, 0].mean(), r[:, 1].mean(), d.mean(), np.median(d), w, len(d), p))


if __name__ == "__main__":
    main()
