"""Structural bands (PLAN.md §6.3): a catastrophic-error filter, evaluated without any GT.

Count bands (spells, D runs) are given for the full real match and scaled by graded duration so
the same config applies to shorter synthetic matches.
"""

import numpy as np

from mtah import registry

REAL_GRADED_FRAMES = 158857


def _runs(labels):
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, labels[start]))
            start = i
    return out


def bands(labels, sp, cfg, kickoffs=None, kickoff_team=None, tol=0):
    """dict(stats..., checks={name: bool}, pass=bool). labels: (T,) array of 'H'/'A'/'D'/'X'.

    Kick-off checks allow `tol` frames of slack for a derived kick-off: everything up to
    kickoff - tol is D, and the first alive label within kickoff + tol is the kick-off team.
    """
    labels = np.asarray(labels).astype("<U1")
    fps = sp["fps"]
    graded = registry.graded_mask(sp)
    scale = graded.sum() / float(REAL_GRADED_FRAMES)
    spells, d_runs = [], 0
    for _, first, last in sp["half_blocks"]:
        for a, b, lab in _runs(labels[first:last + 1]):
            if lab in ("H", "A"):
                spells.append(b - a)
            elif lab == "D":
                d_runs += 1
    spells = np.array(spells, dtype=float)
    alive = graded & np.isin(labels, ["H", "A"])
    stats = {
        "n_spells": int(len(spells)),
        "frac_lt_2s": float(np.mean(spells < 2 * fps)) if len(spells) else 0.0,
        "frac_lt_6s": float(np.mean(spells < 6 * fps)) if len(spells) else 0.0,
        "mean_spell_s": float(spells.mean() / fps) if len(spells) else 0.0,
        "n_d_runs": int(d_runs),
        "home_share_alive": float(np.mean(labels[alive] == "H")) if alive.any() else 0.0,
        "x_frac_graded": float(np.mean(labels[graded] == "X")),
        "d_share_graded": float(np.mean(labels[graded] == "D")),
    }
    b = cfg
    checks = {
        "spells": b["spells"][0] * scale <= stats["n_spells"] <= b["spells"][1] * scale,
        "frac_lt_2s": b["frac_lt_2s"][0] <= stats["frac_lt_2s"] <= b["frac_lt_2s"][1],
        "frac_lt_6s": b["frac_lt_6s"][0] <= stats["frac_lt_6s"] <= b["frac_lt_6s"][1],
        "mean_spell_s": b["mean_spell_s"][0] <= stats["mean_spell_s"] <= b["mean_spell_s"][1],
        "d_runs": b["d_runs"][0] * scale <= stats["n_d_runs"] <= b["d_runs"][1] * scale,
        "home_share_alive": b["home_share_alive"][0] <= stats["home_share_alive"]
        <= b["home_share_alive"][1],
        "x_frac": stats["x_frac_graded"] <= b["max_x_frac"],
    }
    if kickoffs:
        for half, first, _ in sp["half_blocks"]:
            k = kickoffs.get(half)
            if k is None:
                continue
            if k - tol > first:
                checks["pre_kickoff_D_half{}".format(half)] = bool(
                    np.all(labels[first:k - tol] == "D"))
            if kickoff_team:
                w = labels[max(first, k - tol):k + tol + 1]
                alive = w[np.isin(w, ["H", "A"])]
                checks["kickoff_team_half{}".format(half)] = bool(
                    len(alive) and alive[0] == kickoff_team[half])
    # Advisory bands flag but never fail: no band bounded the D share at all until 1 Nov 2026
    # (a candidate halving D would have passed everything), so the new bound runs one adoption
    # cycle as advice before it hardens. It never enters "pass".
    advisory = {}
    if "d_share_graded_advisory" in b:
        lo, hi = b["d_share_graded_advisory"]
        advisory["d_share_graded"] = bool(lo <= stats["d_share_graded"] <= hi)
    stats["checks"] = checks
    stats["advisory"] = advisory
    stats["pass"] = all(checks.values())
    return stats
