"""make sim-fit: fit the four free degradation parameters to the real ATD's label-free statistics.

Ball (q, f_mean, s): coordinate descent over small grids, matching ball-missing fraction, gaps per
frame, median gap length and the collective-speed ratio (players' median speed with the ball
missing / seen), averaged over the `fit` matches with fixed random numbers. Players (g_frames):
P(11 tracks | 7-11 tracks) per team. Loss: weighted squared relative errors (WEIGHT). Prints the `fitted:` line for configs/sim.yaml and the
achieved statistics next to the targets. Open data only: no match-bundle access.
"""

import argparse
import warnings

import numpy as np

from mtah import config
from mtah.learn import degrade, idsse, stats

GRID = {"q": [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
        "f_mean": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        "s": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0],
        # Stoppage reappearance lag (P3a, configs/sim_repaired.yaml): fitted only when the
        # config's fitted dict carries r_frames.
        "r_frames": [0, 12, 25, 50, 100, 150, 250]}
# Where the ball goes missing matters more for possession than the exact gap-length shape.
WEIGHT = {"ball_missing_frac": 2.0, "collective_speed_ratio": 2.0, "gaps_per_frame": 1.0,
          "gap_median_frames": 0.5}
G_GRID = [10, 25, 50, 75, 100, 150, 200, 300, 500]
SEED = 0


def frame_speed(match):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.full(match.players.shape[:2], np.nan)
        v[1:] = np.linalg.norm(np.diff(match.players, axis=0), axis=-1) * match.fps
        return np.nanmean(v, axis=1)


def ball_summary(match, missing, speed):
    graded = match.graded()
    starts, lengths = stats.runs(graded & missing)
    keep = starts > 0
    keep[keep] = graded[starts[keep] - 1] & ~missing[starts[keep] - 1]
    ok = np.isfinite(speed) & graded
    return {"ball_missing_frac": missing[graded].mean(),
            "gaps_per_frame": keep.sum() / graded.sum(),
            "gap_median_frames": np.median(lengths[keep]) if keep.any() else 0.0,
            "collective_speed_ratio": np.median(speed[ok & missing]) / np.median(speed[ok & ~missing])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sim.yaml")
    cfg = config.load(parser.parse_args().config)
    dcfg, mcfg, targets = cfg["degrade"], cfg["measure"], cfg["targets"]
    matches = [idsse.load(m) for m in cfg["matches"]["fit"]]
    prep = [(m, degrade.base_rates(m, dcfg, mcfg), frame_speed(m),
             stats.covariates(m.players, m.ball, m.pitch, mcfg, m.fps)[0]) for m in matches]
    keys = ["ball_missing_frac", "gaps_per_frame", "gap_median_frames", "collective_speed_ratio"]

    def summary(p):
        rows = [ball_summary(m, degrade.ball_missing(m, dcfg, mcfg, degrade.rng_for(SEED, m.name),
                                                      p["q"], p["f_mean"], p["s"], base,
                                                      r=p.get("r_frames"), sbin=sbin), speed)
                for m, base, speed, sbin in prep]
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    def loss(p):
        got = summary(p)
        return sum(WEIGHT[k] * ((got[k] - targets[k]) / targets[k]) ** 2 for k in keys)

    best = dict(dcfg["fitted"])
    for _ in range(3):
        for name, grid in GRID.items():
            if name not in best:
                continue
            scores = [(loss(dict(best, **{name: v})), v) for v in grid]
            best[name] = min(scores)[1]
    got = summary(best)

    def p11(g):
        out = []
        for m in matches:
            slots = degrade.player_slots(m, dcfg, mcfg, degrade.rng_for(SEED, m.name), g)
            graded = m.graded()
            for team, size in enumerate(dcfg["slots"]):
                cols = slice(0, size) if team == 0 else slice(dcfg["slots"][0], None)
                n = np.isfinite(slots[graded, cols, 0]).sum(axis=1)
                band = (n >= 7) & (n <= 11)
                out.append(np.mean(n[band] == 11))
        return float(np.mean(out))

    target_p11 = float(np.mean(targets["p11_given_7to11"]))
    g_scores = [(abs(p11(g) - target_p11), g) for g in G_GRID]
    best["g_frames"] = min(g_scores)[1]

    line = "fitted: {{q: {q}, f_mean: {f_mean}, s: {s}, g_frames: {g_frames}".format(**best)
    print(line + (", r_frames: {}}}".format(best["r_frames"]) if "r_frames" in best else "}"))
    for k in keys:
        print("  {:<24} sim {:.6g}  real {:.6g}".format(k, got[k], targets[k]))
    print("  {:<24} sim {:.4f}  real {:.4f}".format("p11_given_7to11", p11(best["g_frames"]), target_p11))


if __name__ == "__main__":
    main()
