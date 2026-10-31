"""Wrap the vendored official scorer (evaluation/possession_metrics.py). Never modify that file."""

import os

import numpy as np

from mtah import registry  # noqa: F401  (puts evaluation/ on sys.path)
import possession_metrics as pm  # noqa: E402

PUBLIC_GT = os.path.join(registry.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
HEADLINE = ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells", "spell_bootstrap_low",
            "spell_bootstrap_high", "n_abstain_on_graded")


def gt_path(gt):
    if gt == "public":
        return PUBLIC_GT
    if gt == "validation":
        raise ValueError("no hand-labelled validation set: RULES.md (2026-10-23) forbids hand "
                         "labels on the provided game for any purpose, validation included")
    if gt.startswith("sim:"):
        return os.path.join(registry.data_dir_for(gt[4:]), "gt.csv")
    return gt


def codes_of(labels_or_path):
    if isinstance(labels_or_path, str):
        return pm.label_codes(pm.read_label_csv(labels_or_path))
    return pm.label_codes(np.asarray(labels_or_path))


def score(pred, gt="public", bootstrap=0):
    """Official metrics dict (headline keys only) for a CSV path or a label array."""
    gt_codes = codes_of(gt_path(gt))
    result = pm.evaluate(gt_codes, codes_of(pred), bootstrap_n=bootstrap)
    return {k: result[k] for k in HEADLINE}
