"""Board #48: frozen v6 vs v11 on the PFF WC2022 evaluation family (scripts/build_pff.py output).

Committed models only (trained on the DFL fit matches); nothing is fitted, tuned or selected on
PFF. Prints per-match spellF1 and the paired summary (mean, median, sign test). Aggregates only.
"""

import os
import sys

import numpy as np

from mtah import registry

sys.path.insert(0, os.path.join(registry.ROOT, "scripts"))
from oracle_receiver import sign_p                      # noqa: E402
from regret_matrix import V6, V11, committed_labels     # noqa: E402
from mtah.eval.score import score                       # noqa: E402


def main():
    games = sorted(g for g in os.listdir(registry.SIM_DIR) if g.startswith("PFF-"))
    rows = []
    print("{:<12}{:>10}{:>10}{:>10}".format("match", "v6", "v11", "d"))
    for g in games:
        f = [score(committed_labels(g, c), "sim:" + g)["spell_f1_mean"] for c in (V6, V11)]
        rows.append(f)
        print("{:<12}{:>10.4f}{:>10.4f}{:>+10.4f}".format(g, f[0], f[1], f[1] - f[0]), flush=True)
    if not rows:
        raise SystemExit("no PFF-* sims under {}; run scripts/build_pff.py".format(registry.SIM_DIR))
    r = np.array(rows)
    d = r[:, 1] - r[:, 0]
    w, p = sign_p(d)
    print("n={} v6 {:.4f} v11 {:.4f}; d mean {:+.4f} median {:+.4f}; v11 wins {}/{} (sign p {:.4f})"
          .format(len(d), r[:, 0].mean(), r[:, 1].mean(), d.mean(), np.median(d), w, len(d), p))


if __name__ == "__main__":
    main()
