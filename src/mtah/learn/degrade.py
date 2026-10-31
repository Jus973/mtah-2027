"""Degrade a CleanMatch to look like the real ATD, and write it as a synthetic bundle with its GT.

Ball, two mechanisms (the real ATD loses the ball mostly in stoppages: its players move half as
fast while the ball is missing):
  1. Stoppages: each dead stretch loses the ball with probability q, starting up to
     stoppage_lag_frames after the ball goes dead, for a fraction f ~ U around f_mean of the
     stretch (the ball reappears before the restart).
  2. Open play: gap onsets at the real ATD's hazard by ball speed, crowding and pitch edge (shrunk
     towards its margins, scaled by s), lengths from the real gap-length histogram.
Players: per player, tracklets with the real length histogram separated by short invisible gaps
(mean g frames); team-wide dropout episodes and extra-track (duplicate) episodes at the real rates
and lengths; every tracklet goes to a random free slot of its team (24 home / 22 away, as real).

q, f_mean, s and g are fitted (scripts/fit_sim.py) so that the sim's label-free statistics
(learn/stats.py) match the real ones in configs/sim.yaml `targets`; all other numbers are measured.
"""

import json
import os
import zlib

import numpy as np

from mtah import registry
from mtah.io.atd_write import write_gt, write_meta, write_raw
from mtah.learn import stats


def rng_for(seed, name):
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def sample_lengths(rng, hist, edges, n, cap):
    """n lengths from a histogram over [edges[i], edges[i+1]) bins, log-uniform inside a bin;
    the open last bin runs to cap."""
    hist = np.asarray(hist, dtype=float)
    hi_edges = list(edges[1:]) + [cap]
    b = rng.choice(len(hist), size=n, p=hist / hist.sum())
    lo = np.log(np.asarray(edges, dtype=float)[b])
    hi = np.log(np.asarray(hi_edges, dtype=float)[b])
    return np.maximum(1, np.floor(np.exp(rng.uniform(lo, hi))).astype(int))


def hazard_rates(cfg):
    """(edge, speed, crowd) per-frame gap-onset rates from the real counts, shrunk to the margins."""
    alpha = cfg["hazard_prior_frames"]
    counts = np.array([cfg["hazard"]["inside"], cfg["hazard"]["edge"]], dtype=float)  # (2,S,C,2)
    n, k = counts[..., 0], counts[..., 1]
    r_e = k.sum(axis=(1, 2)) / n.sum(axis=(1, 2))
    r_es = (k.sum(axis=2) + alpha * r_e[:, None]) / (n.sum(axis=2) + alpha)
    return (k + alpha * r_es[..., None]) / (n + alpha)


def base_rates(match, cfg, mcfg):
    """(T,) real-ATD gap-onset rate for each frame's ball speed, crowding and edge."""
    sbin, cbin, ebin = stats.covariates(match.players, match.ball, match.pitch, mcfg, match.fps)
    return hazard_rates(cfg)[ebin, sbin, cbin]


def ball_missing(match, cfg, mcfg, rng, q, f_mean, s, base=None):
    """(T,) bool: frames where the degraded ATD has no ball (always outside the halves)."""
    graded = match.graded()
    missing = ~graded.copy()
    lag = cfg["stoppage_lag_frames"]
    lo, hi = max(0.0, 2 * f_mean - 1), min(1.0, 2 * f_mean)
    starts, lengths = stats.runs(graded & ~match.alive)
    lose = rng.random(len(starts)) < q
    f = rng.uniform(lo, hi, len(starts))
    off = rng.integers(0, lag + 1, len(starts))
    for a, n, l_, fr, o in zip(starts, lengths, lose, f, off):
        if l_:
            missing[a + o:min(a + n, a + o + int(round(fr * n)))] = True

    p = s * (base_rates(match, cfg, mcfg) if base is None else base)
    u = rng.random(len(p))
    lengths = sample_lengths(rng, cfg["gap_length_hist"], mcfg["gap_length_edges"], len(p),
                             cfg["gap_length_cap"])
    n = len(p)
    candidates = np.flatnonzero(graded & match.alive & (u < p))
    for t in candidates:
        if missing[t]:
            continue
        stop = t + 1 + lengths[t]
        block_end = t + 1
        while block_end < n and graded[block_end] and block_end < stop:
            block_end += 1
        missing[t + 1:block_end] = True
    return missing


def _episodes(rng, graded, rate, hist, edges, cap):
    frames = np.flatnonzero(graded)
    n = rng.poisson(rate * len(frames))
    starts = np.sort(rng.choice(frames, size=n)) if n else np.array([], dtype=int)
    return starts, sample_lengths(rng, hist, edges, n, cap) if n else np.array([], dtype=int)


def player_slots(match, cfg, mcfg, rng, g):
    """(T, S, 2) metres: players cut into tracklets and dealt into slots, plus episodes."""
    graded = match.graded()
    T = match.n_frames
    edges, cap = mcfg["tracklet_edges"], cfg["tracklet_cap"]
    tracks = []  # (team, start, stop_exclusive, positions (stop-start, 2))
    for team in (0, 1):
        cols = np.flatnonzero(match.team_of_player == team)
        on = np.isfinite(match.players[:, cols, 0])
        visible = on.copy()
        # Tracklets and invisible gaps per player.
        for j in range(len(cols)):
            t = 0
            lens = sample_lengths(rng, cfg["tracklet_hist"], edges, 4096, cap)
            gaps = rng.geometric(1.0 / max(1.0, g), 4096)
            i = 0
            while t < T:
                t += lens[i % 4096]
                visible[t:t + gaps[i % 4096], j] = False
                t += gaps[i % 4096]
                i += 1
        # Team-wide dropout episodes: keep k <= low_max_visible random players.
        starts, lens = _episodes(rng, graded, cfg["low_runs"]["rate"], cfg["low_runs"]["hist"],
                                 edges, cap)
        for a, n in zip(starts, lens):
            k = rng.integers(0, mcfg["low_max_visible"] + 1)
            hide = rng.permutation(len(cols))[k:]
            visible[a:a + n][:, hide] = False
        for j, c in enumerate(cols):
            for a, n in zip(*stats.runs(visible[:, j])):
                tracks.append((team, a, a + n, match.players[a:a + n, c]))
        # Extra-track episodes: a duplicate of one player, offset by a constant.
        starts, lens = _episodes(rng, graded, cfg["high_runs"]["rate"], cfg["high_runs"]["hist"],
                                 edges, cap)
        for a, n in zip(starts, lens):
            j = rng.integers(len(cols))
            seg = match.players[a:a + n, cols[j]] + rng.normal(0, cfg["dup_offset_m"], 2)
            for b, m in zip(*stats.runs(np.isfinite(seg[:, 0]))):
                tracks.append((team, a + b, a + b + m, seg[b:b + m]))

    pools = [list(range(cfg["slots"][0])),
             list(range(cfg["slots"][0], cfg["slots"][0] + cfg["slots"][1]))]
    slots = np.full((T, sum(cfg["slots"]), 2), np.nan)
    busy_until = np.full(slots.shape[1], -1)
    for team, a, b, xy in sorted(tracks, key=lambda r: (r[1], r[0])):
        free = [s for s in pools[team] if busy_until[s] < a]
        if not free:
            continue
        s = free[rng.integers(len(free))]
        slots[a:b, s] = xy
        busy_until[s] = b - 1
    return slots


def degrade(match, cfg, mcfg, seed):
    """(slots (T,S,2) m, ball (T,2) m, team_of_slot (S,)) for one seed."""
    rng = rng_for(seed, match.name)
    p = cfg["fitted"]
    missing = ball_missing(match, cfg, mcfg, rng, p["q"], p["f_mean"], p["s"])
    ball = match.ball.copy()
    ball[missing] = np.nan
    slots = player_slots(match, cfg, mcfg, rng, p["g_frames"])
    team_of_slot = np.array([0] * cfg["slots"][0] + [1] * cfg["slots"][1], dtype=np.int8)
    return slots, ball, team_of_slot


def write_sim(match, cfg, mcfg, seed, out_root=None, game=None):
    """Write <out_root>/<game>/ in the bundle layout (ATD + gt.csv + fake_meta.json)."""
    game = game or "DFL-{}".format(match.name)
    out = os.path.join(out_root or registry.SIM_DIR, game)
    tac = os.path.join(out, "tactical")
    os.makedirs(tac, exist_ok=True)
    slots, ball, team_of_slot = degrade(match, cfg, mcfg, seed)
    pitch = np.asarray(match.pitch)
    (k1, e1), (k2, e2) = match.halves.tolist()
    home_dir1 = 1 if (np.nanmean(match.players[k1, match.team_of_player == 0, 0])
                      < np.nanmean(match.players[k1, match.team_of_player == 1, 0])) else -1
    params = [("first_half_start", k1 + 1), ("first_half_end", e1 + 1),
              ("second_half_start", k2 + 1), ("second_half_end", e2 + 1)]
    stem = os.path.join(tac, game + "_tactical")
    write_meta(stem + "_atd.xml", match.n_frames, match.fps, match.pitch, team_of_slot,
               match.team_names, params, home_dir1, provider="MTAH sim (DFL/IDSSE open data)")
    write_raw(stem + "_atd.txt", slots / pitch, ball / pitch, match.pitch, match.fps)
    write_gt(os.path.join(out, "gt.csv"), match.gt_labels())
    meta = {
        "game": game, "kind": "sim", "source": match.name, "seed": seed,
        "n_frames": match.n_frames, "fps": match.fps, "pitch": list(match.pitch),
        "half_blocks": [[1, k1, e1], [2, k2, e2]],
        "kickoffs": {"1": k1, "2": k2},
        "kickoff_team": {str(h): "HA"[max(0, int(match.owner[k]))] for h, k in ((1, k1), (2, k2))},
        "home_attack_dir": {"1": home_dir1, "2": -home_dir1},
        "degrade": cfg["fitted"],
    }
    with open(os.path.join(out, registry.FAKE_META), "w") as handle:
        json.dump(meta, handle, indent=2)
    return out, meta
