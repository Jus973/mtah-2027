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

Extension (board #46, pre-registered 3 Nov 2026 on the issue and in submissions/LOG.md): three
more GT-convention axes, each run on its own against the SAME cached frozen labels, never
composed with another axis —
  switch  direct H<->A boundaries moved by k in {-10, -5, 0, +5, +10, +25} frames (positive =
          later = the earlier spell grows); boundaries through D untouched; counts preserved;
  merge   sandwiched possession spells shorter than s in {0, 1, 2} seconds (both neighbours are
          spells of the other team, no D between) folded into that team, shortest first, to a
          fixed point; the GT spell count falls by design;
  dstart  D-run starts moved by k in {-10, 0, +10, +25} frames (positive = later = the preceding
          spell grows into the D run); capped so both sides keep >= 1 frame; counts preserved.
Decision rule for those axes (pre-registered): a world FAILS iff its 8-match primary mean
delta < -0.010 AND v11 loses on >= 5 of the 8 matches; any failing world escalates to the human
naming the axis; none failing records "convention-robust on all axes". The default run (no
--axes) is the original D-end matrix and its decision print, unchanged.

Usage: .venv/bin/python scripts/regret_matrix.py [--rebuild] [--committed]
                                                 [--extra MET-G1 MET-G2]
                                                 [--axes dend switch merge dstart]
"""

import argparse
import glob
import hashlib
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


def shift_switches(gt, k, fps=None):
    """Copy of the GT with every DIRECT H<->A boundary moved k frames (k > 0 later: the earlier
    spell grows into the later one; k < 0 earlier). Boundaries through D are untouched. Moves
    come from the ORIGINAL segments and are capped so the spell losing frames keeps >= 1 of its
    own, so spell and D-run counts are preserved and k = 0 is the identity."""
    out = gt.copy()
    if k == 0:
        return out
    codes = pm.label_codes(gt)
    for run in pm.valid_runs(codes):
        segs = pm.segments(codes, [run])
        for (a1, b1, c1), (a2, b2, c2) in zip(segs, segs[1:]):
            if {c1, c2} != {pm.H, pm.A}:
                continue
            if k > 0:
                n = min(k, (b2 - a2) - 1)
                out[a2:a2 + n] = pm.LABELS[c1]
            else:
                n = min(-k, (b1 - a1) - 1)
                out[b1 - n:b1] = pm.LABELS[c2]
    return out


def shift_dstarts(gt, k, fps=None):
    """Copy of the GT with every D run that follows an alive segment (same valid run) starting k
    frames later (k > 0: the preceding spell grows into the D run) or earlier (k < 0: the D run
    eats the spell's tail). Capped so the D run and the spell both keep >= 1 frame; counts
    preserved; k = 0 is the identity. Never composed with shift_dends."""
    out = gt.copy()
    if k == 0:
        return out
    codes = pm.label_codes(gt)
    for run in pm.valid_runs(codes):
        segs = pm.segments(codes, [run])
        for (a1, b1, c1), (a2, b2, c2) in zip(segs, segs[1:]):
            if c2 != pm.D or c1 not in (pm.H, pm.A):
                continue
            if k > 0:
                n = min(k, (b2 - a2) - 1)
                out[a2:a2 + n] = pm.LABELS[c1]
            else:
                n = min(-k, (b1 - a1) - 1)
                out[b1 - n:b1] = pm.LABELS[pm.D]
    return out


def merge_short(gt, s, fps):
    """Copy of the GT with every SANDWICHED possession spell shorter than s seconds folded into
    its neighbours' team: a spell is sandwiched when the segments immediately before and after it
    (same valid run) are both spells of the other team, with no D between. Shortest spell first
    (ties: earliest), repeated to a fixed point, so the result is deterministic and
    order-independent. The GT spell count can only fall; s = 0 is the identity."""
    out = gt.copy()
    if s == 0:
        return out
    thr = s * fps
    codes = pm.label_codes(gt)
    for run in pm.valid_runs(codes):
        segs = [list(x) for x in pm.segments(codes, [run])]
        while True:
            best = None
            for i in range(1, len(segs) - 1):
                a, b, c = segs[i]
                if c not in (pm.H, pm.A) or (b - a) >= thr:
                    continue
                cp, cn = segs[i - 1][2], segs[i + 1][2]
                if cp == cn and cp in (pm.H, pm.A) and cp != c:
                    if best is None or (b - a) < (segs[best][1] - segs[best][0]):
                        best = i
            if best is None:
                break
            segs[best - 1:best + 2] = [[segs[best - 1][0], segs[best + 1][1], segs[best - 1][2]]]
        for a, b, c in segs:
            out[a:b] = pm.LABELS[c]
    return out


# axis -> (title, symbol, transform, grid, counts preserved?)
AXES = {
    "dend": ("D-run ends shifted later", "k", lambda gt, k, fps=None: shift_dends(gt, k),
             SHIFTS, True),
    "switch": ("direct H<->A boundaries shifted (+ = later)", "k", shift_switches,
               (-10, -5, 0, 5, 10, 25), True),
    "merge": ("sandwiched short spells merged (< s seconds)", "s", merge_short, (0, 1, 2), False),
    "dstart": ("D-run starts shifted (+ = later)", "k", shift_dstarts, (-10, 0, 10, 25), True),
}
NEW_AXES = ("switch", "merge", "dstart")
MIN_LOSSES = 5                                # of the 8 primary matches (#46 rule)


def fps_of(game):
    import json
    with open(os.path.join(registry.data_dir_for(game), registry.FAKE_META)) as handle:
        return int(json.load(handle)["fps"])


def check_world(axis, base_codes, new_codes, game):
    """The per-match structural assertions behind every world: counts preserved (or, for the
    merge axis, spells non-increasing and D runs preserved)."""
    runs = pm.valid_runs(base_codes)
    assert pm.valid_runs(new_codes) == runs, game
    n0 = len(pm.possession_spells(base_codes, runs))
    n1 = len(pm.possession_spells(new_codes, runs))
    d0 = sum(1 for _, _, c in pm.segments(base_codes, runs) if c == pm.D)
    d1 = sum(1 for _, _, c in pm.segments(new_codes, runs) if c == pm.D)
    if AXES[axis][4]:
        assert n1 == n0 and d1 == d0, (axis, game, n0, n1, d0, d1)
    else:
        assert n1 <= n0 and d1 == d0, (axis, game, n0, n1, d0, d1)


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


def fingerprint():
    """What the cached labels depend on: both resolved configs, every committed model file and
    the sim bundles' metadata. A cache built under a different fingerprint is stale."""
    h = hashlib.sha256()
    for path in (V6, V11):
        h.update(config.config_hash(config.load(path)).encode())
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "*"))):
        with open(path, "rb") as handle:
            h.update(os.path.basename(path).encode() + hashlib.sha256(handle.read()).digest())
    for path in sorted(glob.glob(os.path.join(registry.SIM_DIR, "*", registry.FAKE_META))):
        with open(path, "rb") as handle:
            h.update(path.encode() + handle.read())
    return h.hexdigest()[:16]


def load_or_build(extra, committed, rebuild):
    data, fp = None, fingerprint()
    if not rebuild and os.path.isfile(CACHE):
        with open(CACHE, "rb") as handle:
            data = pickle.load(handle)
        if data.get("fingerprint") != fp:
            print("cache {} is stale (configs, models or sims changed): rebuilding".format(CACHE))
            data = None
    if data is None:
        data = build(extra, committed)
        data["fingerprint"] = fp
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


def score_table(games, gt_by_game, a_by_game, b_by_game, title, axis="dend"):
    """Per-world per-match spellF1 for both candidates; returns {k: (a, b)} arrays over games."""
    _, sym, transform, grid, _ = AXES[axis]
    out = {}
    for k in grid:
        a_row, b_row = [], []
        for g in games:
            gt_k = transform(gt_by_game[g], k, fps_of(g))
            codes = pm.label_codes(gt_k)
            base = pm.label_codes(gt_by_game[g])
            if k:
                check_world(axis, base, codes, g)
            else:
                assert (codes == base).all(), (axis, g)
            a_row.append(pm.evaluate(codes, pm.label_codes(a_by_game[g]))["spell_f1_mean"])
            b_row.append(pm.evaluate(codes, pm.label_codes(b_by_game[g]))["spell_f1_mean"])
        out[k] = (np.array(a_row), np.array(b_row))

    print("\n== {} ==".format(title))
    print("{:<12} {}".format("match", "  ".join(
        "{}={:<3} v6     v11    Δ     ".format(sym, k) for k in grid)))
    for i, g in enumerate(games):
        cells = ["{:>7.4f} {:>6.4f} {:>+7.4f}".format(out[k][0][i], out[k][1][i],
                                                      out[k][1][i] - out[k][0][i])
                 for k in grid]
        print("{:<12} {}".format(g, "  ".join(cells)))
    print("{:<12} {}".format("mean", "  ".join(
        "{:>7.4f} {:>6.4f} {:>+7.4f}".format(out[k][0].mean(), out[k][1].mean(),
                                             (out[k][1] - out[k][0]).mean()) for k in grid)))
    print("{:<12} {}".format("wins (Δ>0)", "  ".join(
        "{:>21}".format("{}/{}".format(int(((out[k][1] - out[k][0]) > 0).sum()), len(games)))
        for k in grid)))
    if not AXES[axis][4]:
        print("{:<12} {}".format("GT spells", "  ".join("{:>21}".format(sum(
            len(pm.possession_spells(pm.label_codes(transform(gt_by_game[g], k, fps_of(g))),
                                     pm.valid_runs(pm.label_codes(gt_by_game[g]))))
            for g in games)) for k in grid)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--committed", action="store_true",
                        help="also score committed-model outputs on the 7 DFL sims")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="additional committed-model matches (e.g. MET-G1 MET-G2)")
    parser.add_argument("--axes", nargs="*", default=["dend"], choices=sorted(AXES),
                        help="GT-convention axes to score (default: the #8 D-end matrix)")
    args = parser.parse_args()
    data = load_or_build(args.extra, args.committed, args.rebuild)
    dfl, nf = data["dfl"], data["nf"]
    primary_games = dfl + list(PRIMARY_EXTRA)

    if "dend" not in args.axes:
        return new_axes(args, data, primary_games)

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

    if [a for a in args.axes if a != "dend"]:
        new_axes(args, data, primary_games)


def new_axes(args, data, primary_games):
    """The #46 axes: one table per axis on the primary 8, the pre-registered fail rule per
    world, supplementary split means and MET-G1/G2 rows. Prints only; adopts nothing."""
    dfl, nf = data["dfl"], data["nf"]
    axes = [a for a in args.axes if a != "dend"]
    failing = []
    for axis in axes:
        title, sym, _, grid, _ = AXES[axis]
        res = score_table(primary_games, data["gt"], data["v6"], data["v11"],
                          "PRIMARY, axis '{}': {} — cross-fitted on {} DFL sims + "
                          "committed-model on {}".format(axis, title, len(dfl),
                                                         ", ".join(PRIMARY_EXTRA)), axis)
        print("\n-- splits (mean Δ per world) --")
        idx_fit = list(range(nf))
        idx_held = list(range(nf, len(dfl)))
        idx_g3 = [len(dfl) + i for i in range(len(PRIMARY_EXTRA))]
        for name, idx in (("fit (5, OOF)", idx_fit), ("held-out (2)", idx_held),
                          ("MET-G3", idx_g3)):
            cells = ["{}={}: {:+.4f}".format(sym, k, float((res[k][1] - res[k][0])[idx].mean()))
                     for k in grid]
            print("  {:<14} {}".format(name, "  ".join(cells)))
        print("-- pre-registered rule (#46): FAIL iff mean Δ < {:+.3f} AND v11 loses on >= {} "
              "of {} --".format(THRESHOLD, MIN_LOSSES, len(primary_games)))
        for k in grid:
            d = res[k][1] - res[k][0]
            losses = int((d < 0).sum())
            fail = d.mean() < THRESHOLD and losses >= MIN_LOSSES
            print("  {}={:<3} mean Δ {:+.4f}  losses {}/{}  -> {}".format(
                sym, k, d.mean(), losses, len(primary_games), "FAIL" if fail else "ok"))
            if fail:
                failing.append((axis, k))
        extra_present = [g for g in args.extra if g in data["gt"]]
        if extra_present:
            score_table(extra_present, data["gt"], data["v6"], data["v11"],
                        "SUPPLEMENTARY, axis '{}': committed-model on {}".format(
                            axis, ", ".join(extra_present)), axis)

    print("\n== PRE-REGISTERED DECISION (board #46 / submissions/LOG.md, 3 Nov 2026) ==")
    if failing:
        print("  FAILING worlds: {} — ESCALATE to the human naming the axis; the follow-up is a "
              "hedge candidate with its own pre-registration. Nothing is adopted or reverted "
              "by this script.".format(", ".join("{}:{}".format(a, k) for a, k in failing)))
    else:
        print("  No world fails on {} — v11 convention-robust on these axes; keep FINAL. "
              "Nothing fires.".format(", ".join(axes)))


if __name__ == "__main__":
    main()
