"""GT-convention regret matrix (pre-registered in submissions/LOG.md, 2 Nov 2026, before the
first run; results recorded there the same day).

Scores the FROZEN v6 (configs/rules_v6_final.yaml) and v11 (configs/rules_v11.yaml) against
transformed GTs: every GT D-run end shifted later by k ∈ {0, 10, 20, 38} frames (annotator-late
restart worlds; k = 0 is the published GT world). The shift is capped so the alive segment after
each D run keeps at least one frame, so spell and D-run counts are preserved in every world and
k = 0 is the identity — asserted per match. Outputs: the §6.4 cross-fitted protocol
(scripts/paired.py labels_for, the exact make-paired code path) on the 7 DFL sims; the
committed-model pipeline (mtah.pipeline.run) on MET-G3 and any --extra matches (supplementary).
Pure re-scoring with the official scorer (wrapped, never modified); open data only; no config
change; the labels are cached under cache/ so re-scoring and --extra additions are cheap.

Decision rule (pre-registered): min over k of the 8-match primary mean of spellF1(v11) −
spellF1(v6) ≥ −0.010 → "v11 convention-robust; keep FINAL"; otherwise escalate to the human with
the hedged-v12 ladder ranked by max-min across worlds. This script only prints; it adopts nothing.

Usage: .venv/bin/python scripts/regret_matrix.py [--rebuild] [--committed]
                                                 [--extra MET-G1 MET-G2]
"""

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from mtah import config, registry            # noqa: E402
from mtah.eval.score import pm               # noqa: E402
import paired                                # noqa: E402

CACHE = os.path.join(ROOT, "cache", "regret_matrix_labels.pkl")
SHIFTS = (0, 10, 20, 38)
V6 = os.path.join(ROOT, "configs", "rules_v6_final.yaml")
V11 = os.path.join(ROOT, "configs", "rules_v11.yaml")
THRESHOLD = -0.010
PRIMARY_EXTRA = ("MET-G3",)                  # committed-model matches in the primary mean


def shift_dends(gt, k):
    """Copy of the GT label array with every D-run end moved k frames later, capped so the alive
    segment after the run keeps >= 1 frame. Segment boundaries come from the ORIGINAL array."""
    out = gt.copy()
    if k == 0:
        return out
    codes = pm.label_codes(gt)
    for run in pm.valid_runs(codes):
        segs = pm.segments(codes, [run])
        for (_, b, c), (a2, b2, c2) in zip(segs, segs[1:]):
            if c == pm.D and c2 in (pm.H, pm.A):
                out[b:b + min(k, (b2 - a2) - 1)] = "D"
    return out


def gt_of(game):
    path = os.path.join(registry.data_dir_for(game), "gt.csv")
    return pm.read_label_csv(path)["label"].to_numpy(dtype=str)


def committed_labels(game, cfg_path):
    from mtah.pipeline import run as pipeline_run
    labels, _ = pipeline_run(game, config.load(cfg_path))
    return labels


def build(extra, committed):
    """{"gt": {game: labels}, "v6"/"v11": {game: pred}, "v6_cm"/"v11_cm": {DFL game: pred},
    "dfl": [...], "nf": int} — cross-fitted on the DFL sims, committed-model elsewhere."""
    data = {"gt": {}, "v6": {}, "v11": {}, "v6_cm": {}, "v11_cm": {}}
    for name, path in (("v6", V6), ("v11", V11)):
        print("cross-fitting {} ({})...".format(name, os.path.relpath(path, ROOT)), flush=True)
        games, nf, labels, Q = paired.labels_for(path)
        data[name].update(dict(zip(games, labels)))
        data["dfl"], data["nf"] = games, nf
        for g, q in zip(games, Q):
            data["gt"][g] = q.m.gt
    for game in list(PRIMARY_EXTRA) + list(extra):
        print("committed-model pipeline on {}...".format(game), flush=True)
        for name, path in (("v6", V6), ("v11", V11)):
            data[name][game] = committed_labels(game, path)
        data["gt"][game] = gt_of(game)
    if committed:
        for game in data["dfl"]:
            print("committed-model pipeline on {}...".format(game), flush=True)
            for name, path in (("v6_cm", V6), ("v11_cm", V11)):
                data[name][game] = committed_labels(game, path)
    return data


def load_or_build(extra, committed, rebuild):
    data = None
    if not rebuild and os.path.isfile(CACHE):
        with open(CACHE, "rb") as handle:
            data = pickle.load(handle)
    if data is None:
        data = build(extra, committed)
    else:  # top up what the cache is missing
        for game in [g for g in extra if g not in data["gt"]]:
            print("committed-model pipeline on {}...".format(game), flush=True)
            for name, path in (("v6", V6), ("v11", V11)):
                data[name][game] = committed_labels(game, path)
            data["gt"][game] = gt_of(game)
        if committed and not data["v6_cm"]:
            for game in data["dfl"]:
                print("committed-model pipeline on {}...".format(game), flush=True)
                for name, path in (("v6_cm", V6), ("v11_cm", V11)):
                    data[name][game] = committed_labels(game, path)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as handle:
        pickle.dump(data, handle)
    return data


def score_table(games, gt_by_game, a_by_game, b_by_game, title):
    """Per-world per-match spellF1 for both candidates; returns {k: (a, b)} arrays over games."""
    out = {}
    for k in SHIFTS:
        a_row, b_row = [], []
        for g in games:
            gt_k = shift_dends(gt_by_game[g], k)
            codes = pm.label_codes(gt_k)
            if k:
                base = pm.label_codes(gt_by_game[g])
                runs = pm.valid_runs(base)
                assert len(pm.possession_spells(codes, runs)) == \
                    len(pm.possession_spells(base, runs)), g
            a_row.append(pm.evaluate(codes, pm.label_codes(a_by_game[g]))["spell_f1_mean"])
            b_row.append(pm.evaluate(codes, pm.label_codes(b_by_game[g]))["spell_f1_mean"])
        out[k] = (np.array(a_row), np.array(b_row))

    print("\n== {} ==".format(title))
    print("{:<12} {}".format("match", "  ".join(
        "k={:<2} v6     v11    Δ      ".format(k) for k in SHIFTS)))
    for i, g in enumerate(games):
        cells = ["{:>7.4f} {:>6.4f} {:>+7.4f}".format(out[k][0][i], out[k][1][i],
                                                      out[k][1][i] - out[k][0][i])
                 for k in SHIFTS]
        print("{:<12} {}".format(g, "  ".join(cells)))
    print("{:<12} {}".format("mean", "  ".join(
        "{:>7.4f} {:>6.4f} {:>+7.4f}".format(out[k][0].mean(), out[k][1].mean(),
                                             (out[k][1] - out[k][0]).mean()) for k in SHIFTS)))
    print("{:<12} {}".format("wins (Δ>0)", "  ".join(
        "{:>21}".format("{}/{}".format(int(((out[k][1] - out[k][0]) > 0).sum()), len(games)))
        for k in SHIFTS)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--committed", action="store_true",
                        help="also score committed-model outputs on the 7 DFL sims")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="additional committed-model matches (e.g. MET-G1 MET-G2)")
    args = parser.parse_args()
    data = load_or_build(args.extra, args.committed, args.rebuild)
    dfl, nf = data["dfl"], data["nf"]
    primary_games = dfl + list(PRIMARY_EXTRA)

    res = score_table(primary_games, data["gt"], data["v6"], data["v11"],
                      "PRIMARY: cross-fitted on {} DFL sims + committed-model on {} "
                      "(official scorer vs shifted GT)".format(len(dfl), ", ".join(PRIMARY_EXTRA)))

    print("\n-- splits (mean Δ per world) --")
    idx_fit = list(range(nf))
    idx_held = list(range(nf, len(dfl)))
    idx_g3 = [len(dfl) + i for i in range(len(PRIMARY_EXTRA))]
    for name, idx in (("fit (5, OOF)", idx_fit), ("held-out (2)", idx_held),
                      ("MET-G3", idx_g3)):
        cells = ["k={}: {:+.4f}".format(k, float((res[k][1] - res[k][0])[idx].mean()))
                 for k in SHIFTS]
        print("  {:<14} {}".format(name, "  ".join(cells)))

    deltas = {k: float((res[k][1] - res[k][0]).mean()) for k in SHIFTS}
    worst_k = min(deltas, key=deltas.get)
    print("\n== PRE-REGISTERED DECISION (submissions/LOG.md, 2 Nov 2026) ==")
    print("  primary mean Δ(v11 − v6) per world: {}".format(
        "  ".join("k={}: {:+.4f}".format(k, deltas[k]) for k in SHIFTS)))
    print("  min over k = {:+.4f} (at k = +{}); threshold {:+.3f}".format(
        deltas[worst_k], worst_k, THRESHOLD))
    if deltas[worst_k] >= THRESHOLD:
        print("  VERDICT: v11 convention-robust — keep FINAL. No further action fires.")
    else:
        print("  VERDICT: v11 loses more than {:+.3f} in the worst world — ESCALATE to the "
              "human with the hedged-v12 ladder (max-min across worlds). Nothing is adopted or "
              "reverted by this script.".format(THRESHOLD))

    if args.extra:
        extra_present = [g for g in args.extra if g in data["gt"]]
        if extra_present:
            score_table(extra_present, data["gt"], data["v6"], data["v11"],
                        "SUPPLEMENTARY: committed-model on {}".format(", ".join(extra_present)))

    if data["v6_cm"]:
        score_table(dfl, data["gt"], data["v6_cm"], data["v11_cm"],
                    "SUPPLEMENTARY: committed-model outputs on the 7 DFL sims (in-sample on "
                    "the 5 fit matches)")


if __name__ == "__main__":
    main()
