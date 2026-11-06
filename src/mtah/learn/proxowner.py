"""Per-frame ball-proximity owner model and the kick-rate regime blend (v12, board #67).

v11 changes owner only at contacts, which the touch detector finds from the ball's speed and turn.
The real ATD's ball is velocity-filtered (kick rate 2.6 vs 21-43 per 1,000 seen frames on the
sims, board #55); on sims whose ball is smoothed like that, v11 loses 0.165 spellF1 held-out, all
of it in owner decoding, while a per-frame proximity model loses almost nothing (board #67).

This model reads, per frame and team-symmetrically, who is near the ball and how near (now,
seconds either side, and the time since / until each team is next near it), where the ball is
heading, how the nearest players move relative to it, and both teams' shape. One LightGBM gives
P(home owns | alive). It is trained on the fit sims only, each also rebuilt with its ball
smoothed over `train_widths` frames, so it does not depend on how crisp the ball track is.

At run time the match's kick rate (an input statistic of the tracking) picks one of two
pre-registered settings, both chosen on the fit sims out of fold: on a kicky feed v11's owner
dominates (weight w on its +/-1 decision), on a smooth one the proximity log-odds do. A 2-state
Viterbi with switch cost lam then re-decides H/A inside every alive run; D and X never change.
Single-threaded and seeded, so training and prediction are deterministic.
"""

import os
import warnings

import numpy as np

from mtah import registry
from mtah.features.ball import clean_ball, velocity
from mtah.learn import stats

SHIFTS = (-25, -12, -5, 5, 12, 25)


def smooth_ball(ball, width):
    """Centred moving average of the ball over `width` frames inside every seen run (metres or
    normalised, NaN = missing). width <= 1 returns the ball unchanged."""
    if width <= 1:
        return ball
    out = ball.copy()
    seen = np.isfinite(ball[:, 0])
    h = width // 2
    for a, n in zip(*stats.runs(seen)):
        c = np.cumsum(np.vstack([np.zeros((1, 2)), ball[a:a + n]]), axis=0)
        i = np.arange(n)
        lo, hi = np.maximum(i - h, 0), np.minimum(i + h + 1, n)
        out[a:a + n] = (c[hi] - c[lo]) / (hi - lo)[:, None]
    return out


def kick_rate(players, ball_raw, graded, pitch, fps):
    """Kicks (ball velocity change > 3 m/s over +/- 3 frames) per 1,000 seen graded frames."""
    return stats.carrier_stats(players, ball_raw, graded, pitch, fps)["kicks_per_1e3_seen"]


def _shift(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    else:
        out[-k:] = a[:k]
    return out


def _ffill(a):
    idx = np.where(np.isfinite(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[~np.isfinite(a[idx])] = np.nan
    return out


def _window_mean(a, lo_off, hi_off):
    """Mean of a over [t + lo_off, t + hi_off), NaN-aware, clipped to the series."""
    ok = np.isfinite(a)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, a, 0.0))])
    n = np.concatenate([[0], np.cumsum(ok)])
    t = np.arange(len(a))
    lo, hi = np.clip(t + lo_off, 0, len(a)), np.clip(t + hi_off, 0, len(a))
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / cnt, np.nan)


def _since(mask):
    t = np.arange(len(mask))
    last = np.where(mask, t, -1)
    np.maximum.accumulate(last, out=last)
    return np.where(last >= 0, t - last, 9999).astype(float)


TURN_SCALES = (12, 25, 50)
TURN_THRESH = (2.0, 4.0)


def _turn_features(ball, d1h, d1a, fps):
    """Ball turns that survive smoothing (board #85). At scale s frames: the change between the
    mean velocity over the s frames before t and the s frames after, in m/s. Turns above a
    threshold, at their local peak, are 'events'; each event is attributed to the team nearer the
    ball there (margin d1a - d1h, + = home nearer). Per frame: the last and next event's
    attributed margin and the frames since / until it, plus the raw turn strength."""
    cols = {}
    T = len(ball)
    for s in TURN_SCALES:
        before = (ball - _shift2(ball, -s)) * (fps / s)
        after = (_shift2(ball, s) - ball) * (fps / s)
        tau = np.linalg.norm(after - before, axis=1)
        cols["turn%d" % s] = tau
        margin = d1a - d1h
        for th in TURN_THRESH:
            t0 = np.where(np.isfinite(tau), tau, -1.0)
            prev, nxt = np.nan_to_num(_shift(t0, -1), nan=-1.0), np.nan_to_num(_shift(t0, 1), nan=-1.0)
            peak = (t0 > th) & (t0 >= prev) & (t0 > nxt)
            ev = np.where(peak & np.isfinite(margin), np.clip(margin, -10, 10), np.nan)
            key = "%d_%g" % (s, th)
            cols["lastturn" + key] = _ffill(ev)
            cols["nextturn" + key] = _ffill(ev[::-1])[::-1]
            cols["sinceturn" + key] = _since(np.isfinite(ev))
            cols["untilturn" + key] = _since(np.isfinite(ev)[::-1])[::-1]
    return cols


def _shift2(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def features(players, ball, team_of_slot, hdir, pitch, fps, turns=False):
    """(T, k) features and their names. players (T,S,2) m, ball (T,2) m (cleaned), team_of_slot
    (S,) 0/1, hdir (T,) +1 where home attacks towards x = pitch length."""
    team = np.asarray(team_of_slot)
    T = len(ball)
    t = np.arange(T)
    cols = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.linalg.norm(players - ball[:, None, :], axis=-1)
        v = velocity(ball, 2, fps)
        pv = np.full(players.shape, np.nan)
        pv[2:-2] = (players[4:] - players[:-4]) * (fps / 4)
        seen = np.isfinite(ball[:, 0])
        for k, name in ((0, "h"), (1, "a")):
            on = team == k
            dk = d[:, on]
            srt = np.sort(np.where(np.isfinite(dk), dk, np.inf), axis=1)
            d1 = np.where(np.isfinite(srt[:, 0]), srt[:, 0], np.nan)
            d2 = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan)
            cols["d1" + name], cols["d2" + name] = d1, d2
            cols["n2" + name] = np.sum(dk < 2, axis=1).astype(float)
            cols["n5" + name] = np.sum(dk < 5, axis=1).astype(float)
            cols["soft" + name] = np.nansum(np.exp(-dk / 1.5), axis=1)
            pk = players[:, on]
            cols["vis" + name] = np.sum(np.isfinite(pk[..., 0]), axis=1).astype(float)
            for lead in (12, 25):
                dd = np.linalg.norm(pk - (ball + v * (lead / fps))[:, None, :], axis=-1)
                m = np.nanmin(np.where(np.isfinite(dd), dd, np.inf), axis=1)
                cols["head%d%s" % (lead, name)] = np.where(np.isfinite(m), m, np.nan)
            cols["close" + name] = d1 - _shift(d1, -5)
            xa = np.where(hdir[:, None] > 0, pk[..., 0], pitch[0] - pk[..., 0])
            cols["cx" + name] = np.nanmean(xa, axis=1)
            cols["len" + name] = np.nanmax(xa, axis=1) - np.nanmin(xa, axis=1)
            cols["wid" + name] = np.nanstd(pk[..., 1], axis=1)
            cols["cdist" + name] = np.linalg.norm(np.nanmean(pk, axis=1) - ball, axis=1)
            for w in (25, 75):
                cols["n2r%d%s" % (w, name)] = _window_mean(cols["n2" + name], -w, w + 1)
                cols["d1r%d%s" % (w, name)] = _window_mean(d1, -w, w + 1)
            for r in (1.5, 3.0):
                near = d1 < r
                cols["since%.1f%s" % (r, name)] = _since(near)
                cols["until%.1f%s" % (r, name)] = _since(near[::-1])[::-1]
            j = np.argmin(np.where(np.isfinite(dk), dk, np.inf), axis=1)
            vm = np.linalg.norm(pv[:, on][t, j] - v, axis=1)
            cols["vmatch" + name] = np.where(np.isfinite(d1), vm, np.nan)
            for k2 in (-25, -12, 12, 25):
                cols["d1%s%+d" % (name, k2)] = _shift(d1, k2)
        diff = cols["d1h"] - cols["d1a"]
        cols["diff"] = diff
        cols["diff_ff"] = _ffill(diff)
        for k2 in SHIFTS:
            cols["diff%+d" % k2] = _shift(diff, k2)
        for w in (25, 75, 150):
            cols["diff_roll%d" % w] = _window_mean(diff, -w, w + 1)
        for w in (25, 75):
            cols["diff_past%d" % w] = _window_mean(diff, -w, 1)
            cols["diff_fut%d" % w] = _window_mean(diff, 0, w + 1)
        for r in (1.5, 3.0):
            cols["recent%.1f" % r] = cols["since%.1fa" % r] - cols["since%.1fh" % r]
            cols["next%.1f" % r] = cols["until%.1fa" % r] - cols["until%.1fh" % r]
        cols["cx_diff"] = cols["cxh"] - cols["cxa"]
        cols["since_seen"] = _since(seen)
        cols["seen"] = seen.astype(float)
        cols["speed"] = np.linalg.norm(v, axis=1)
        cols["ballx_home"] = np.where(hdir > 0, ball[:, 0], pitch[0] - ball[:, 0])
        cols["bally"] = ball[:, 1]
        if turns:
            cols.update(_turn_features(ball, cols["d1h"], cols["d1a"], fps))
    names = sorted(cols)
    return np.column_stack([cols[n] for n in names]).astype(np.float32), names


def both_views(players, ball, team_of_slot, hdir, pitch, fps, turns=False):
    """Features from the home team's view and from the away team's (teams and direction swapped)."""
    a, names = features(players, ball, team_of_slot, hdir, pitch, fps, turns)
    b, _ = features(players, ball, 1 - np.asarray(team_of_slot), -hdir, pitch, fps, turns)
    return a, b, names


def training_rows(views, gt, step):
    """Alive GT frames every `step`, from both views, labelled 'this view's home team owns'."""
    a, b, _ = views
    rows = np.flatnonzero(np.isin(gt, ["H", "A"]))[::step]
    X = np.vstack([a[rows], b[rows]])
    y = np.concatenate([(gt[rows] == "H"), (gt[rows] == "A")]).astype(int)
    return X, y


def train(X, y, params, names):
    import lightgbm as lgb
    p = {k: v for k, v in params.items() if k != "rounds"}
    p.update(objective="binary", num_threads=1, deterministic=True, verbose=-1)
    return lgb.train(p, lgb.Dataset(X, y, feature_name=list(names)), params["rounds"])


def load(cfg):
    import lightgbm as lgb
    return lgb.Booster(model_file=os.path.join(registry.ROOT, cfg["proxowner"]["model"]))


def p_home(booster, views):
    a, b, _ = views
    return (booster.predict(a, num_threads=1) + 1 - booster.predict(b, num_threads=1)) / 2


def setting(cfg, rate):
    """The pre-registered (w, lam) for a match with this kick rate."""
    pc = cfg["proxowner"]
    return pc["smooth"] if rate < pc["kick_threshold"] else pc["kicky"]


def turn_gate(ball_raw, fps, gate):
    """Frames within gate["radius"] of a ball turn (scale gate["scale"] frames, strength >
    gate["thresh"] m/s; see _turn_features), on the raw-world ball in metres."""
    s, r = int(gate["scale"]), int(gate["radius"])
    before = (ball_raw - _shift2(ball_raw, -s)) * (fps / s)
    after = (_shift2(ball_raw, s) - ball_raw) * (fps / s)
    hit = np.nan_to_num(np.linalg.norm(after - before, axis=1)) > float(gate["thresh"])
    c = np.concatenate([[0], np.cumsum(hit)])
    t = np.arange(len(hit))
    return (c[np.minimum(t + r + 1, len(hit))] - c[np.maximum(t - r, 0)]) > 0


def weights(cfg, rate, ball_raw, fps):
    """(w, lam) for blend_decode: the pre-registered setting for this kick rate, and on a smooth
    feed with proxowner.smooth_turn_gate set, w replaced by gate["w_near"] near ball turns (a
    per-frame array). Without the gate this is setting()."""
    w, lam = setting(cfg, rate)
    gate = cfg["proxowner"].get("smooth_turn_gate")
    if gate and rate < cfg["proxowner"]["kick_threshold"]:
        w = np.where(turn_gate(ball_raw, fps, gate), float(gate["w_near"]), float(w))
    return w, lam


def blend_decode(labels, p, w, lam):
    """Re-decide H/A inside every alive run of `labels`: emissions = proximity log-odds + w * (+1
    where labels say H, -1 where A); 2-state Viterbi, switch cost lam. D and X are untouched."""
    out = labels.copy()
    q = np.clip(p, 1e-4, 1 - 1e-4)
    lo = np.log(q) - np.log1p(-q) + w * np.where(labels == "H", 1.0, np.where(labels == "A", -1.0, 0.0))
    alive = np.isin(labels, ["H", "A"])
    edges = np.diff(np.concatenate([[0], alive.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        e = np.stack([lo[a:b] / 2, -lo[a:b] / 2], axis=1)
        n = b - a
        score, back = e[0].copy(), np.zeros((n, 2), dtype=np.int8)
        for i in range(1, n):
            stay, sw = score, score[::-1] - lam
            back[i] = np.where(stay >= sw, [0, 1], [1, 0])
            score = np.maximum(stay, sw) + e[i]
        s = int(np.argmax(score))
        path = np.empty(n, dtype=np.int8)
        for i in range(n - 1, -1, -1):
            path[i] = s
            s = back[i, s]
        out[a:b] = np.where(path == 0, "H", "A")
    return out


def match_views(atd, sp, kicks, cfg, ball_raw=None, width=0):
    """Both views for one match (optionally with its raw ball smoothed first: an int is a box width,
    a "family:param" string a ballfilter.apply filter, e.g. "bw4:0.70"), using the same
    ball cleaning and home direction as the pipeline."""
    from mtah.learn import gapstate
    fps = sp["fps"]
    raw = atd.ball_m() if ball_raw is None else ball_raw
    players = atd.players_m()[..., :2].astype(float)
    if isinstance(width, str):
        # "family:param" filters the ball; an optional "/p<Hz>" also runs the players through a
        # 4th-order zero-phase Butterworth at that cutoff (the real feed's tracks are filtered).
        from mtah.learn import ballfilter
        spec, _, pl = width.partition("/p")
        family, param = spec.split(":")
        if float(param) > 0:
            raw = ballfilter.apply(raw, family, float(param), fps)
        if pl:
            players = ballfilter.apply_players(players, "bw4", float(pl), fps)
    else:
        raw = smooth_ball(raw, width)
    ball, _, _ = clean_ball(raw, atd.pitch, cfg["ball"], fps)
    hdir = gapstate.home_direction(atd, sp, kicks, cfg["direction"]["window_frames"])
    return both_views(players, ball, atd.team_of_slot, hdir, atd.pitch, fps,
                      bool(cfg["proxowner"].get("turn_features", False)))
