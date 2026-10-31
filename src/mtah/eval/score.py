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


def score(pred, gt, bootstrap=0, allow_public=False):
    """Official metrics dict (headline keys only) for a CSV path or a label array.

    The public window refuses unless allow_public: exact public scores are for reproducing the
    published baselines and for GT self-tests only. Candidates go through public_check().
    """
    if gt == "public" and not allow_public:
        raise PermissionError("the public window is a pass/fail bug check, never a score "
                              "(RULES.md: not a tuning set). Use public_check() / "
                              "make airlock T=public-check CSV=...")
    gt_codes = codes_of(gt_path(gt))
    result = pm.evaluate(gt_codes, codes_of(pred), bootstrap_n=bootstrap)
    return {k: result[k] for k in HEADLINE}


def swap_teams(codes):
    out = codes.copy()
    out[codes == pm.H], out[codes == pm.A] = pm.A, pm.H
    return out


def public_check(pred, cfg):
    """The only way to score a candidate on the public window: a verdict, never the number.

    RULES.md: the public 5 minutes check correctness (alignment, team sign, format) and are not
    a tuning set. Returns verdict "invalid" (official validator fails; reason given), "broken"
    (spellF1 < broken_below: team sign inverted or frames misaligned), "fail" (< floor: a bug or a
    wrong owner hypothesis, debug on the sim) or "pass", plus whether swapping H/A would score
    higher (a sign bug). The spellF1 itself never leaves this function.
    """
    gt_frame = pm.read_label_csv(PUBLIC_GT)
    if isinstance(pred, str):
        reason = pm.validate_submission(pm.read_label_csv(pred), len(gt_frame))
    else:
        reason = None if len(pred) == len(gt_frame) else "row count {} != {}".format(
            len(pred), len(gt_frame))
    if reason:
        return {"verdict": "invalid", "reason": reason, "swap_scores_higher": None}
    gt, codes = pm.label_codes(gt_frame), codes_of(pred)
    f1 = pm.evaluate(gt, codes, bootstrap_n=0)["spell_f1_mean"]
    swapped = pm.evaluate(gt, swap_teams(codes), bootstrap_n=0)["spell_f1_mean"]
    verdict = ("broken" if f1 < cfg["broken_below"] else "fail" if f1 < cfg["floor"] else "pass")
    return {"verdict": verdict, "reason": None, "swap_scores_higher": bool(swapped > f1)}
