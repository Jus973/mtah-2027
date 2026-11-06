"""Heuristic-labeller GT worlds, owner axis (board #69; protocol and amendment pre-registered on
the issue, 6 Nov 2026, before any world was built or scored).

Every world relabels the OWNER on alive frames of an open-data ground truth with a round-valued
automatic rule run on the CLEAN (pre-degradation) tracking, the way a provider might label
corrected data. D stays the source flag, so only the owner axis moves (D conventions: #8, #46).

  L0       the source flag (identity; asserted equal to the bundle's gt.csv)
  L1-h{h}  nearest-team hysteresis: candidate = team of the nearest player if within 1.5 m; the
           owner switches when the other team has been the candidate for h consecutive frames
  L2       control rule: an other-team player within 1.0 m for >= 25 consecutive frames, or two
           other-team contact episodes (1.0 m) starting within 3 s
  L3       last touch (events/touches.py v1 rules at 1.0 m) — circular with v11, reported only
  L4       operator-snapped: each direct H<->A switch of the source moves to the nearest L1-h5
           switch to the same team within +/- 25 frames
  L5       pass-release: each direct H<->A switch moves back to the end of the losing team's last
           contact (1.0 m) if that ended <= 50 frames before and the winner had no contact between

Pinned by the amendment: L1/L2 switches are backdated to the start of the qualifying run;
between D and the first candidate the owner is the first candidate team; an alive stretch with no
candidate keeps the source flag; every move keeps each spell >= 1 frame.

Frozen methods, committed models only (the primary matches are out of fold for all of them):
FINAL (v12-hedgeA), v11, v6, the official smooth50 and prox-only (the committed proximity owner
model of the FINAL decoded with w = 0 inside v11's labels; an evaluation stand-in, not a
candidate). Feeds: raw and b13 (scripts/build_smoothed_sims.py --width 13).

Usage: .venv/bin/python scripts/labeller_worlds.py [--b13-dir DIR] [--rebuild]
Prints only; adopts nothing, changes no config.
"""

import argparse
import glob
import hashlib
import os
import pickle
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from mtah import config, registry                    # noqa: E402
from mtah.eval.score import pm                       # noqa: E402

PRIMARY = ("DFL-J03WQQ", "DFL-J03WR9", "MET-G3")
WORLDS = ("L0", "L1-h1", "L1-h5", "L1-h12", "L2", "L3", "L4", "L5")
COUNTED = ("L1-h1", "L1-h5", "L1-h12", "L2", "L4", "L5")
METHODS = ("FINAL", "v11", "v6", "smooth50", "prox-only")
CONFIGS = {"FINAL": "final.yaml", "v11": "rules_v11.yaml", "v6": "rules_v6_final.yaml"}
R_CAND, R_CTRL = 1.5, 1.0            # metres: L1 candidate radius; L2/L3/L5 contact radius
CTRL_RUN, CTRL_PAIR_S = 25, 3.0      # L2: frames held within 1 m; seconds between two episodes
SNAP, RELEASE = 25, 50               # L4 snap window; L5 max contact-to-switch frames
PLAUSIBLE = (0.5, 2.0)               # amendment: descriptives within [0.5x, 2x] of L0's
GAP_R, GAP_WIDEN, ROBUST_R = -0.03, 0.02, -0.010
REPRO = {"raw": (0.6444, 0.6828), "b13": (0.4966, 0.5013)}   # #67 v11 on J03WQQ / J03WR9
CACHE = os.path.join(ROOT, "cache", "labeller_worlds_labels.pkl")


# ---------------------------------------------------------------- labellers (pure, tested)

def alive_runs(gt):
    """[(a, b)] half-open maximal runs of H/A frames in a label array."""
    alive = np.isin(gt, ["H", "A"]).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], alive, [0]))))
    return list(zip(edges[::2], edges[1::2]))


def _fill(gt, a, owner):
    """Write a per-frame owner (-1 = undetermined) over the alive run starting at a: -1 frames
    before the first determined frame take the first owner; a run with none keeps the source."""
    out = gt[a:a + len(owner)].copy()
    det = np.flatnonzero(owner >= 0)
    if det.size == 0:
        return out
    owner = owner.copy()
    owner[:det[0]] = owner[det[0]]
    return np.where(owner == 0, "H", "A")


def hysteresis(cand, h):
    """Owner per frame from a candidate stream (0/1, -1 = none): the first candidate sets the
    owner; afterwards the owner switches when the other team has been the candidate for h
    consecutive frames, backdated to the first of them. A -1 frame carries the owner and breaks
    the other team's streak."""
    n = len(cand)
    out = np.full(n, -1, dtype=np.int8)
    owner, start, length = -1, 0, 0
    for t in range(n):
        c = cand[t]
        if c < 0 or c == owner:
            length = 0
        else:
            if length == 0:
                start = t
            length += 1
            if owner < 0 or length >= h:
                owner = c
                out[start:t] = c
                length = 0
        out[t] = owner
    return out


def candidates(d, r):
    """Team of the nearest player if within r metres, else -1. d: (T, 2) per-team distances."""
    dd = np.where(np.isfinite(d), d, np.inf)
    team = np.argmin(dd, axis=1).astype(np.int8)
    best = dd.min(axis=1)
    return np.where(np.isfinite(best) & (best <= r), team, -1).astype(np.int8)


def world_l1(gt, d, h):
    out = gt.copy()
    cand = candidates(d, R_CAND)
    for a, b in alive_runs(gt):
        out[a:b] = _fill(gt, a, hysteresis(cand[a:b], h))
    return out


def control(d, fps):
    """L2 owner stream for one alive run (d: (n, 2)). Contact = a team's nearest player within
    1.0 m. The first contact's team (nearest at that frame) sets the owner; the other team takes
    over after CTRL_RUN consecutive contact frames, or when a second contact episode of it starts
    within CTRL_PAIR_S of its first since the last switch — backdated to the qualifying start,
    never before the frame after the previous switch."""
    n = len(d)
    out = np.full(n, -1, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        near = np.where(np.isfinite(d), d, np.inf) <= R_CTRL
    pair = int(round(CTRL_PAIR_S * fps))
    owner, last_switch = -1, -1
    run_start = [None, None]
    episodes = [[], []]
    for t in range(n):
        for team in (0, 1):
            if near[t, team]:
                if run_start[team] is None:
                    run_start[team] = t
                    episodes[team].append(t)
            else:
                run_start[team] = None
        if owner < 0:
            if near[t].any():
                owner = int(np.argmin(np.where(np.isfinite(d[t]), d[t], np.inf)))
                last_switch = t
                episodes = [[], []]
                if run_start[owner] is not None:
                    run_start[owner] = t
            out[t] = owner
            continue
        o = 1 - owner
        start = None
        if run_start[o] is not None and t - run_start[o] + 1 >= CTRL_RUN:
            start = run_start[o]
        eps = [e for e in episodes[o] if e > last_switch]
        if len(eps) >= 2 and eps[-1] - eps[-2] <= pair:
            start = eps[-2] if start is None else min(start, eps[-2])
        if start is not None:
            start = max(start, last_switch + 1)
            out[start:t] = o
            owner, last_switch = o, t
            episodes = [[], []]
            run_start = [t if near[t, k] else None for k in (0, 1)]
        out[t] = owner
    return out


def world_l2(gt, d, fps):
    out = gt.copy()
    for a, b in alive_runs(gt):
        out[a:b] = _fill(gt, a, control(d[a:b], fps))
    return out


def world_l3(gt, ball, d, fps, touch_cfg):
    from mtah.events.touches import touch_events
    out = gt.copy()
    owner = np.full(len(gt), -1, dtype=np.int8)
    for tch in touch_events(ball, d, dict(touch_cfg, r_touch_m=R_CTRL), fps):
        owner[tch.start] = tch.team
    for a, b in alive_runs(gt):
        seg = owner[a:b].copy()
        last = -1
        for i in range(len(seg)):
            if seg[i] >= 0:
                last = seg[i]
            seg[i] = last
        out[a:b] = _fill(gt, a, seg)
    return out


def direct_switches(labels):
    """Frames s where labels[s-1] and labels[s] are H and A in either order."""
    prev, cur = labels[:-1], labels[1:]
    m = np.isin(prev, ["H", "A"]) & np.isin(cur, ["H", "A"]) & (prev != cur)
    return np.flatnonzero(m) + 1


def _move_switches(gt, target):
    """Move each direct switch s of gt to target(s, lo, hi), clamped to [lo, hi] so each
    neighbouring spell keeps >= 1 frame (spell bounds from the ORIGINAL segments). Switches are
    processed in time order and a spell's start is its preceding switch's NEW position, so two
    moves never cross. target returns None to leave the switch where it is."""
    out = gt.copy()
    moved = {}
    for s in direct_switches(gt):
        s = int(s)
        lo = s - 1
        while lo > 0 and gt[lo - 1] == gt[s - 1]:
            lo -= 1
        hi = s
        while hi + 1 < len(gt) and gt[hi + 1] == gt[s]:
            hi += 1
        floor = moved.get(lo, lo) + 1
        new = target(s, lo + 1, hi)
        if new is None:
            new = s
        new = int(min(max(new, floor), hi))
        moved[s] = new
        if new == s:
            continue
        if new < s:
            out[new:s] = gt[s]
        else:
            out[s:new] = gt[s - 1]
    return out


def world_l4(gt, l1h5):
    sw = direct_switches(l1h5)
    team_at = {int(s): l1h5[s] for s in sw}
    keys = np.array(sorted(team_at), dtype=np.int64)

    def target(s, lo, hi):
        if not keys.size:
            return None
        near = keys[np.abs(keys - s) <= SNAP]
        near = [k for k in near if team_at[int(k)] == gt[s]]
        if not near:
            return None
        return int(min(near, key=lambda k: (abs(k - s), k)))
    return _move_switches(gt, target)


def world_l5(gt, d):
    with np.errstate(invalid="ignore"):
        near = np.where(np.isfinite(d), d, np.inf) <= R_CTRL

    def target(s, lo, hi):
        winner = 0 if gt[s] == "H" else 1
        loser = 1 - winner
        idx = np.flatnonzero(near[lo - 1:s, loser])
        if not idx.size:
            return None
        t = lo - 1 + int(idx[-1])
        if s - 1 - t > RELEASE or near[t + 1:s, winner].any():
            return None
        return t + 1
    return _move_switches(gt, target)


def describe(labels, nearest, fps):
    """(spells, share of spells < 2 s, owner switches on alive frames, nearest-team flips)."""
    codes = pm.label_codes(labels)
    sp = pm.possession_spells(codes, pm.valid_runs(codes))
    lens = np.array([b - a for a, b, _ in sp])
    switches = len(direct_switches(labels))
    alive = np.isin(labels, ["H", "A"])
    ok = alive[1:] & alive[:-1] & (nearest[1:] >= 0) & (nearest[:-1] >= 0)
    flips = int(np.sum(ok & (nearest[1:] != nearest[:-1])))
    return len(sp), float(np.mean(lens < 2 * fps)) if lens.size else 0.0, switches, flips


# ---------------------------------------------------------------- open data and methods

def clean_of(game):
    if game.startswith("DFL-"):
        from mtah.learn import idsse
        return idsse.load(game[4:])
    from mtah.io import metrica
    return metrica.load(game.split("-", 1)[1])


def team_d(m):
    """(T, 2) distance from the clean ball to each team's nearest player, NaN if none/no ball."""
    dist = np.linalg.norm(m.players - m.ball[:, None, :], axis=-1)
    d = np.full((m.n_frames, 2), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for team in (0, 1):
            d[:, team] = np.nanmin(dist[:, m.team_of_player == team], axis=1)
    return d


def worlds_of(game, fps_check=None):
    m = clean_of(game)
    gt = m.gt_labels()
    bundle = pm.read_label_csv(os.path.join(registry.data_dir_for(game), "gt.csv"))
    assert (bundle["label"].to_numpy(dtype=str) == gt).all(), "L0 != bundle gt.csv: " + game
    d = team_d(m)
    touch_cfg = config.load("rules.yaml")["touch"]
    w = {"L0": gt}
    for h in (1, 5, 12):
        w["L1-h{}".format(h)] = world_l1(gt, d, h)
    w["L2"] = world_l2(gt, d, m.fps)
    w["L3"] = world_l3(gt, m.ball, d, m.fps, touch_cfg)
    w["L4"] = world_l4(gt, w["L1-h5"])
    w["L5"] = world_l5(gt, d)
    for name, lab in w.items():
        assert (np.isin(lab, ["D", "X"]) == np.isin(gt, ["D", "X"])).all(), (game, name)
        assert ((lab == "D") == (gt == "D")).all(), (game, name)
    nearest = candidates(d, np.inf)
    return w, nearest, m.fps


def method_labels(game, feed_dir):
    """{method: labels} for one match on one feed (committed models, frozen configs)."""
    from mtah.features.baseline import nearest_player_codes
    from mtah.io.atd import load_game
    from mtah.learn import proxowner
    from mtah.pipeline import kickoffs, run as pipeline_run
    import baseline as starter                       # vendored, unmodified

    data_dir = os.path.join(feed_dir, game)
    out = {}
    for name, path in CONFIGS.items():
        out[name], info = pipeline_run(game, config.load(path), data_dir)
        if name == "FINAL":
            setting = info["proxowner_setting"]
    cfg = config.load(CONFIGS["FINAL"])
    atd, _ = load_game(game, data_dir)
    sp = registry.spec(game, data_dir)
    views = proxowner.match_views(atd, sp, kickoffs(game, atd, cfg, data_dir), cfg,
                                  ball_raw=atd.ball_m())
    p = proxowner.p_home(proxowner.load(cfg), views)
    w, lam = setting
    again = proxowner.blend_decode(out["v11"], p, w, lam)
    assert (again == out["FINAL"]).all(), "FINAL != blend(v11): " + game
    out["prox-only"] = proxowner.blend_decode(out["v11"], p, 0.0, lam)
    out["smooth50"] = starter.smooth(nearest_player_codes(atd), 50)
    out["_setting"] = (w, lam)
    return out


def fingerprint(feeds):
    """Configs, committed models and the feeds' bundle digests: a stale cache rebuilds."""
    from mtah.io.atd import sim_tracking_digest
    h = hashlib.sha256()
    for path in CONFIGS.values():
        h.update(config.config_hash(config.load(path)).encode())
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "*"))):
        with open(path, "rb") as handle:
            h.update(os.path.basename(path).encode()
                     + hashlib.file_digest(handle, "sha256").digest())
    for feed, root in sorted(feeds.items()):
        for game in PRIMARY:
            h.update("{}:{}:".format(feed, game).encode()
                     + sim_tracking_digest(game, os.path.join(root, game)).encode())
    return h.hexdigest()


def load_or_build(feeds, rebuild):
    data, fp = None, fingerprint(feeds)
    if not rebuild and os.path.isfile(CACHE):
        with open(CACHE, "rb") as handle:
            data = pickle.load(handle)
        if data.get("fingerprint") != fp:
            print("label cache is stale (configs, models or feeds changed): rebuilding")
            data = None
    if data is None:
        data = {"fingerprint": fp, "labels": {}}
        for feed, root in feeds.items():
            for game in PRIMARY:
                print("methods on {} ({})...".format(game, feed), flush=True)
                data["labels"][(feed, game)] = method_labels(game, root)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(CACHE), delete=False) as handle:
            pickle.dump(data, handle)
        os.replace(handle.name, CACHE)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b13-dir", default=os.path.expanduser("~/opendata/sim_b13"))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    feeds = {"raw": registry.SIM_DIR, "b13": os.path.expanduser(args.b13_dir)}

    # 1. worlds and descriptives first: the plausibility filter reads the GT only
    worlds, nearest, fps = {}, {}, {}
    for game in PRIMARY:
        worlds[game], nearest[game], fps[game] = worlds_of(game)
    print("== worlds (GT only; pooled over {}) ==".format(", ".join(PRIMARY)))
    print("{:<8} {:>7} {:>9} {:>9} {:>11}  {}".format("world", "spells", "<2s share", "switches",
                                                     "sw/flip", "per-match spells"))
    desc = {}
    for wname in WORLDS:
        rows = [describe(worlds[g][wname], nearest[g], fps[g]) for g in PRIMARY]
        n = sum(r[0] for r in rows)
        short = sum(r[1] * r[0] for r in rows) / max(1, n)
        sw, fl = sum(r[2] for r in rows), sum(r[3] for r in rows)
        desc[wname] = (short, sw / max(1, fl))
        print("{:<8} {:>7} {:>9.3f} {:>9} {:>11.4f}  {}".format(
            wname, n, short, sw, sw / max(1, fl), " ".join(str(r[0]) for r in rows)))
    plausible = {}
    for wname in WORLDS:
        ratios = [desc[wname][i] / desc["L0"][i] for i in (0, 1)]
        plausible[wname] = all(PLAUSIBLE[0] <= r <= PLAUSIBLE[1] for r in ratios)
    counted = [w for w in COUNTED if plausible[w]]
    print("plausibility filter [{}x, {}x] of L0 -> counted: {} (filtered out: {})".format(
        PLAUSIBLE[0], PLAUSIBLE[1], ", ".join(counted) or "none",
        ", ".join(w for w in COUNTED if not plausible[w]) or "none"))

    # 2. methods (frozen, committed models) and scores
    data = load_or_build(feeds, args.rebuild)
    for (feed, game), lab in sorted(data["labels"].items()):
        print("{} {}: proxowner setting (w, lam) = {}".format(feed, game, lab["_setting"]))
    F1 = {}
    for feed in feeds:
        for wname in WORLDS:
            for game in PRIMARY:
                gt_c = pm.label_codes(worlds[game][wname])
                for meth in METHODS:
                    pred = data["labels"][(feed, game)][meth]
                    F1[(feed, wname, game, meth)] = pm.evaluate(
                        gt_c, pm.label_codes(pred))["spell_f1_mean"]

    v11_l0 = {f: tuple(F1[(f, "L0", g, "v11")] for g in PRIMARY[:2]) for f in feeds}
    print("\n== L0 reproduction (#67 v11 held-out J03WQQ / J03WR9) ==")
    repro_ok = True
    for f in feeds:
        diff = max(abs(a - b) for a, b in zip(v11_l0[f], REPRO[f]))
        repro_ok &= diff <= 0.002
        print("  {}: {:.4f} / {:.4f} vs {:.4f} / {:.4f}  max diff {:.4f} -> {}".format(
            f, *v11_l0[f], *REPRO[f], diff, "ok" if diff <= 0.002 else "MISMATCH"))
    if not repro_ok:
        print("  STOP (pre-registered): L0 does not reproduce #67; debug before reading worlds.")

    R = {}
    for feed in feeds:
        print("\n== feed {}: spellF1 per world, 3-match mean (per match) ==".format(feed))
        print("{:<8} ".format("world") + " ".join("{:>9}".format(m) for m in METHODS)
              + "   {:>8} {:>8}  per-match R_F".format("R_F", "R_11"))
        for wname in WORLDS:
            mean = {m: np.mean([F1[(feed, wname, g, m)] for g in PRIMARY]) for m in METHODS}
            per = {m: np.array([F1[(feed, wname, g, m)] for g in PRIMARY]) for m in METHODS}
            ref = np.maximum(per["smooth50"], per["prox-only"])
            rf, r11 = per["FINAL"] - ref, per["v11"] - ref
            R[(feed, wname)] = (float(rf.mean()), float(r11.mean()))
            tag = {"L0": " (identity)", "L3": " (circular, reported only)"}.get(
                wname, "" if wname in counted else " (filtered out)")
            print("{:<8} ".format(wname) + " ".join("{:>9.4f}".format(mean[m]) for m in METHODS)
                  + "   {:>+8.4f} {:>+8.4f}  {}{}".format(
                      rf.mean(), r11.mean(), " ".join("{:+.3f}".format(x) for x in rf), tag))

    print("\n== PRE-REGISTERED DECISION (board #69 + amendment; on R_F, FINAL v12-hedgeA) ==")
    for idx, who in ((0, "FINAL (decides)"), (1, "v11 (reported)")):
        l0 = R[("b13", "L0")][idx]
        widened = [w for w in counted if R[("b13", w)][idx] < GAP_R
                   and R[("b13", w)][idx] < l0 - GAP_WIDEN]
        robust = all(R[("raw", w)][idx] >= ROBUST_R for w in counted)
        if len(widened) >= 2:
            verdict = "GAP TO THE REFERENCE WIDENS (b13 worlds: {})".format(", ".join(widened))
        elif robust:
            verdict = "LEAD ROBUST to automatic-labeller conventions (owner axis)"
        else:
            verdict = "MIXED — post the table, change nothing"
        print("  {:<16} L0-b13 R {:+.4f}; widened b13 worlds {}; raw min R {:+.4f} -> {}".format(
            who, l0, widened or "none", min(R[("raw", w)][idx] for w in counted) if counted
            else float("nan"), verdict))
    if not repro_ok:
        print("  (verdict void: the L0 reproduction check failed)")


if __name__ == "__main__":
    main()
