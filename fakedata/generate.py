"""Synthetic match in the exact bundle formats, with its own ground truth.

    python fakedata/generate.py --game BAR-FAKE --seed 0 --half-s 900

Writes fakedata/out/<GAME>/ with the same layout as the real bundle:

    tactical/<GAME>_tactical_atd.xml            EPTS metadata (kloppy-loadable), home team first
    tactical/<GAME>_tactical_atd.txt            "<frame_id>:<x,y,s>;...(46);:<x,y,s>", frame_id from 1
    tactical/<GAME>_tactical_homography.csv     frame,h00..h22,reprojection_error,valid
    tactical/<GAME>_tactical_smart_tagging.xml  SportsCode XML, <start>/<end> in seconds
    gt.csv                                      frame,label over the full timeline, X outside halves
    fake_meta.json                              timeline, kick-offs, directions, generation stats

Purpose: end-to-end tests with no real data present. It is NOT training data and its statistics are
only roughly matched to PLAN.md (possession lengths: median ~6 s, ~20% < 2 s; ball missing on ~41%
of live graded frames, clustered on contested moments; slot churn; ~20.5 visible players).

The world model is deliberately simple: players drift around a formation that follows a slow "play
position"; the ball sits next to a holder or travels in a straight line between holders. GT flips
on the frame the new team first controls the ball (a received pass/interception), D starts when the
ball leaves the pitch or play is stopped, and ends on the restart touch.
"""

import argparse
import json
import os
import sys

import numpy as np

from mtah.io.atd_write import write_gt, write_meta, write_raw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FPS = 25
PITCH = (105.3, 68.0)
N_SLOTS = 46
HOME_SLOTS = 23
HOME_NAME, AWAY_NAME = "Fake Home FC", "Fake Away CF"
TAG_TEAM = {0: "Barcelona", 1: "Atletic Club"}   # same strings as the real Smart Tagging
H, A = 0, 1

# 4-3-3 in "own goal at x=0, attacking +x" normalised coordinates, compact enough for own half.
FORMATION = np.array([
    [0.04, 0.50], [0.18, 0.15], [0.16, 0.38], [0.16, 0.62], [0.18, 0.85],
    [0.30, 0.30], [0.28, 0.50], [0.30, 0.70], [0.42, 0.18], [0.45, 0.50], [0.42, 0.82]])
# In play the block is stretched: forwards push past halfway (x up to ~0.72), so open play rarely
# looks like a kick-off line-up, as in real matches.
PLAY_FORMATION = FORMATION * np.array([1.6, 1.0])
# Kick-off: everyone in own half; one striker on the spot, one beside it.
KICKOFF_FORMATION = FORMATION.copy()
KICKOFF_FORMATION[9] = [0.497, 0.50]
KICKOFF_FORMATION[8] = [0.47, 0.42]


class Timeline:
    def __init__(self, half_s, pre=750, halftime=2500, pre2=300, post=500):
        n_half = int(half_s * FPS)
        self.k1 = pre
        self.h1 = (pre, pre + n_half - 1)
        h2_start = self.h1[1] + 1 + halftime
        self.k2 = h2_start + pre2
        self.h2 = (h2_start, self.k2 + n_half - 1)
        self.n = self.h2[1] + 1 + post


def team_xy(base, direction):
    """Map own-goal-at-0 coordinates to pitch coordinates for a team attacking `direction`."""
    out = base.copy()
    if direction < 0:
        out[..., 0] = 1.0 - out[..., 0]
        out[..., 1] = 1.0 - out[..., 1]
    return out


def smooth_noise(rng, n, k, scale, corr=150):
    """(n, k) smooth random walk-ish noise via filtered white noise."""
    white = rng.normal(size=(n + corr, k))
    kernel = np.hanning(corr)
    kernel /= kernel.sum()
    out = np.stack([np.convolve(white[:, j], kernel, mode="valid")[:n] for j in range(k)], axis=1)
    return out * scale / (out.std() + 1e-9)


def spell_length(rng):
    """Frames of one possession: 22% short (0.3-2 s), else lognormal around 8 s."""
    if rng.random() < 0.22:
        return int(rng.uniform(0.3, 2.0) * FPS)
    return int(np.clip(rng.lognormal(np.log(8.0), 0.75), 2.0, 60.0) * FPS)


class Sim:
    def __init__(self, seed, tl):
        self.rng = np.random.default_rng(seed)
        self.tl = tl
        n = tl.n
        self.pos = np.full((n, 22, 2), np.nan)      # players 0-10 home, 11-21 away
        self.ball = np.full((n, 2), np.nan)
        self.label = np.full(n, "X", dtype="<U1")
        self.tags = []                              # (code, start_frame, end_frame, {group: text})
        self.turnovers = []                         # frames where possession changed team
        self.in_flight = np.zeros(n, dtype=bool)
        self.home_dir = {1: int(self.rng.choice([-1, 1]))}
        self.home_dir[2] = -self.home_dir[1]
        self.kick_team = {1: A, 2: H}               # matches the real match (A kicks off first)

    # -- players --------------------------------------------------------------------------------
    def players(self):
        tl, rng = self.tl, self.rng
        n = tl.n
        play = 0.5 + smooth_noise(rng, n, 1, 0.12, corr=400)[:, 0]        # play position, home frame
        jitter = smooth_noise(rng, n, 44, 0.025, corr=75).reshape(n, 22, 2)
        for half, (first, last), kick in ((1, tl.h1, tl.k1), (2, tl.h2, tl.k2)):
            lo = kick - 60 if half == 1 else first
            for team in (H, A):
                d = self.home_dir[half] * (1 if team == H else -1)
                idx = slice(0, 11) if team == H else slice(11, 22)
                t = np.arange(lo, last + 1)
                # Blend from kick-off formation to the moving block over 3 s after the kick.
                w = np.clip((t - kick) / (3.0 * FPS), 0.0, 1.0)[:, None, None]
                shift = (play[t] - 0.5) * (1 if team == H else -1) * self.home_dir[half]
                block = PLAY_FORMATION[None] + np.stack([shift * 0.9 + 0.05, np.zeros_like(shift)],
                                                   axis=1)[:, None, :]
                local = (1 - w) * KICKOFF_FORMATION[None] + w * (block + jitter[t][:, idx])
                self.pos[t, idx] = team_xy(local, d)
        np.clip(self.pos, 0.005, 0.995, out=self.pos)

    # -- ball and labels ------------------------------------------------------------------------
    def run_half(self, half):
        tl, rng = self.tl, self.rng
        first, last = tl.h1 if half == 1 else tl.h2
        kick = tl.k1 if half == 1 else tl.k2
        self.label[first:kick] = "D"
        team = self.kick_team[half]
        striker = 9 + (0 if team == H else 11)
        centre = np.array([0.5, 0.5])
        self.ball[kick - 50:kick + 1] = centre
        self.tags.append(("SET PIECES", kick - 2 * FPS, kick + 3 * FPS,
                          {"Team": TAG_TEAM[team], "Half": "1st Half" if half == 1 else "2nd Half",
                           "Type": "Kick Off (Start)"}))
        # The kick: ball rests on the spot up to and including `kick`, then is passed straight
        # to the teammate beside it, so "the frame the ball is struck" is exactly `kick`.
        mate = 8 + (0 if team == H else 11)
        first_spell = True
        t = kick + 1
        holder = striker
        while t <= last:
            spell_start = kick if first_spell else t
            spell_end = min(last + 1, spell_start + max(spell_length(rng), 3 * FPS if first_spell else 0))
            if first_spell:
                self._fly(kick + 1, kick + 12, striker, mate)
                holder, t, first_spell = mate, kick + 12, False
            self.label[spell_start:spell_end] = "H" if team == H else "A"
            # Passes among teammates until the spell's time is used up.
            mates = list(range(0, 11)) if team == H else list(range(11, 22))
            while True:
                dwell = int(rng.integers(3, 30))
                hold_end = min(spell_end, t + dwell)
                self._hold(t, hold_end, holder)
                t = hold_end
                if t >= spell_end:
                    break
                cand = [m for m in mates if m != holder]
                dist = [np.linalg.norm((self.pos[t, m] - self.pos[t, holder]) * PITCH) for m in cand]
                order = np.argsort(dist)[:5]
                nxt = cand[int(rng.choice(order))]
                travel = int(np.clip(min(dist) / 14.0 * FPS * rng.uniform(1.0, 2.0), 4, 40))
                arrive = t + travel
                if arrive >= spell_end:
                    self._hold(t, spell_end, holder)
                    break
                self._fly(t, arrive, holder, nxt)
                holder, t = nxt, arrive
            if t > last:
                break
            # End of spell: turnover (interception) or dead ball.
            t = spell_end
            if t > last:
                break
            if rng.random() < 0.22 and spell_end - spell_start > FPS:
                t, team, holder = self._dead_ball(t, last, team, holder, half)
            else:
                opp = list(range(11, 22)) if team == H else list(range(0, 11))
                d = [np.linalg.norm((self.pos[t, o] - self.ball[t - 1]) * PITCH) for o in opp]
                new = opp[int(np.argmin(d))]
                # Contested: the ball is loose for a few frames before the opponent controls it.
                loose = int(rng.integers(2, 10))
                start_fly = max(spell_start, t - loose)
                self._fly(start_fly, t, holder, new)
                team, holder = 1 - team, new
                self.turnovers.append(t)
        self.label[first:last + 1][self.label[first:last + 1] == "X"] = "D"

    def _hold(self, t0, t1, p):
        self.ball[t0:t1] = self.pos[t0:t1, p] + [0.004, 0.003]

    def _fly(self, t0, t1, a, b):
        p0 = self.ball[t0 - 1] if t0 > 0 and not np.isnan(self.ball[t0 - 1, 0]) else self.pos[t0, a]
        p1 = self.pos[t1, b] + [0.004, 0.003]
        for u in range(t0, t1):
            f = (u - t0 + 1) / float(t1 - t0 + 1)
            self.ball[u] = p0 + f * (p1 - p0)
        self.in_flight[t0:t1] = True

    def _dead_ball(self, t, last, team, holder, half):
        """Ball goes out (or a foul); D until the restart touch. Returns (restart_frame, team, holder)."""
        rng = self.rng
        kind = rng.choice(["throw", "goal_kick", "corner", "foul"], p=[0.5, 0.15, 0.12, 0.23])
        p0 = self.ball[t - 1].copy()
        if kind == "throw":
            spot = np.array([p0[0], -0.01 if p0[1] < 0.5 else 1.01])
        elif kind in ("goal_kick", "corner"):
            spot = np.array([-0.01 if p0[0] < 0.5 else 1.01, p0[1]])
        else:
            spot = p0
        out_frames = int(rng.integers(5, 15))
        for u in range(t, min(last + 1, t + out_frames)):
            f = (u - t + 1) / float(out_frames)
            self.ball[u] = p0 + f * (spot - p0)
        dur = int(rng.uniform(5.0, 25.0) * FPS)
        stop = min(last + 1, t + dur)
        self.label[t:stop] = "D"
        restart_team = (1 - team) if rng.random() < 0.7 else team
        # Ball unseen while being fetched, then still on the restart spot for the last 2-4 s.
        rest = int(rng.uniform(2.0, 4.0) * FPS)
        spot_in = np.clip(spot, 0.0, 1.0)
        self.ball[max(t + out_frames, stop - rest):stop] = spot_in
        if stop > last:
            return stop, restart_team, holder
        mates = list(range(0, 11)) if restart_team == H else list(range(11, 22))
        d = [np.linalg.norm((self.pos[stop, m] - spot_in) * PITCH) for m in mates]
        taker = mates[int(np.argmin(d))]
        typ = {"throw": "Throw-in", "goal_kick": "Goal Kick", "corner": "Corner Kick",
               "foul": "Free Kick"}[kind]
        keep = {"throw": 0.6, "goal_kick": 0.8, "corner": 0.9, "foul": 0.3}[kind]
        if rng.random() < keep:
            team_tag = "N/A" if kind == "throw" else TAG_TEAM[restart_team]
            j = int(rng.normal(0, 12))
            self.tags.append(("SET PIECES", stop - 3 * FPS + j, stop + 2 * FPS + j,
                              {"Team": team_tag, "Half": "1st Half" if half == 1 else "2nd Half",
                               "Type": typ}))
        return stop, restart_team, taker

    # -- phase tags (team-tagged, noisy) ------------------------------------------------------------
    def phase_tags(self):
        rng = self.rng
        lab = self.label
        for half, (first, last) in ((1, self.tl.h1), (2, self.tl.h2)):
            seg_start = first
            for t in range(first + 1, last + 2):
                if t > last or lab[t] != lab[seg_start]:
                    if lab[seg_start] in "HA" and t - seg_start > 6 * FPS and rng.random() < 0.6:
                        team = H if lab[seg_start] == "H" else A
                        if rng.random() < 0.1:
                            team = 1 - team
                        code = str(rng.choice(["BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD"]))
                        self.tags.append((code, seg_start + int(rng.normal(0, 15)),
                                          t + int(rng.normal(0, 15)),
                                          {"Team": TAG_TEAM[team],
                                           "Half": "1st Half" if half == 1 else "2nd Half"}))
                    seg_start = t

    # -- observation model ----------------------------------------------------------------------
    def observe(self, target_missing=0.41):
        """Ball gaps clustered on contested moments; slot churn and dropouts for players."""
        rng, tl, n = self.rng, self.tl, self.tl.n
        graded = np.zeros(n, dtype=bool)
        graded[tl.h1[0]:tl.h1[1] + 1] = True
        graded[tl.h2[0]:tl.h2[1] + 1] = True
        alive = graded & np.isin(self.label, ["H", "A"])
        weight = np.where(alive, 1.0, 0.0) + 1.5 * self.in_flight
        for f in self.turnovers:
            weight[max(0, f - 30):f + 30] += 5.0
        protect = np.zeros(n, dtype=bool)
        for k in (tl.k1, tl.k2):
            protect[k - 50:k + 15] = True
        weight[protect] = 0.0
        ball_obs = self.ball.copy()
        missing = np.zeros(n, dtype=bool)
        p = weight / weight.sum()
        while missing[alive].mean() < target_missing:
            starts = rng.choice(n, size=10, p=p)
            for s in starts:
                length = int(np.clip(rng.lognormal(np.log(18), 0.9), 3, 300))
                seg = slice(s, min(n, s + length))
                missing[seg] = True
            missing &= ~protect
        ball_obs[missing] = np.nan
        ball_obs[~graded & ~protect] = np.nan
        self.ball_missing_alive = float(np.isnan(ball_obs[alive, 0]).mean())

        # Players -> slots. Each player holds one slot of its team; after an occlusion it may come
        # back in a fresh slot (tracklet break), which is what makes slots not identities.
        slots = np.full((n, N_SLOTS, 2), np.nan)
        busy = np.zeros((n, N_SLOTS), dtype=bool)
        for p_idx in range(22):
            team = H if p_idx < 11 else A
            pool = list(range(0, HOME_SLOTS)) if team == H else list(range(HOME_SLOTS, N_SLOTS))
            slot = pool[p_idx % 11]
            visible = ~np.isnan(self.pos[:, p_idx, 0])
            t = 0
            while t < n:
                run = int(rng.exponential(60 * FPS * 0.6)) + FPS
                stop = min(n, t + run)
                if busy[t:stop, slot].any():
                    free = [s for s in pool if not busy[t:stop, s].any()]
                    if not free:
                        t = stop
                        continue
                    slot = int(rng.choice(free))
                seg = np.arange(t, stop)
                slots[seg[visible[seg]], slot] = self.pos[seg[visible[seg]], p_idx]
                busy[t:stop, slot] = True
                t = stop + int(rng.integers(5, 60))
                if rng.random() < 0.5:
                    free = [s for s in pool if not busy[t:t + FPS, s].any()]
                    slot = int(rng.choice(free)) if free else slot
        self.slots = slots
        self.ball_obs = ball_obs


def write_homography(path, n):
    """Constant tactical camera: normalised pitch -> 1920x1080 pixels. Valid everywhere."""
    h = [1800.0, 0.0, 60.0, 0.0, 950.0, 65.0, 0.0, 0.0, 1.0]
    row = ",".join("{:g}".format(v) for v in h)
    with open(path, "w") as handle:
        handle.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        for t in range(n):
            handle.write("{},{},0.25,1\n".format(t, row))


def write_tags(path, tags):
    from xml.sax.saxutils import escape
    rows = []
    for i, (code, start, end, groups) in enumerate(sorted(tags, key=lambda r: r[1])):
        labels = "".join("<label><group>{}</group><text>{}</text></label>".format(escape(g), escape(v))
                         for g, v in groups.items())
        rows.append("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>{}"
                    "</instance>".format(i + 1, start / FPS, end / FPS, escape(code), labels))
    with open(path, "w") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n<file><ALL_INSTANCES>\n')
        handle.write("\n".join(rows))
        handle.write("\n</ALL_INSTANCES></file>\n")


def spell_stats(labels, blocks):
    lengths = []
    for first, last in blocks:
        w = labels[first:last + 1]
        start = 0
        for i in range(1, len(w) + 1):
            if i == len(w) or w[i] != w[start]:
                if w[start] in "HA":
                    lengths.append(i - start)
                start = i
    lengths = np.array(lengths)
    return {"n_spells": int(len(lengths)),
            "median_s": float(np.median(lengths) / FPS),
            "frac_lt_2s": float(np.mean(lengths < 2 * FPS)),
            "frac_lt_6s": float(np.mean(lengths < 6 * FPS))}


def generate(game="BAR-FAKE", seed=0, half_s=900.0, out_root=None):
    out = os.path.join(out_root or os.path.join(HERE, "out"), game)
    tac = os.path.join(out, "tactical")
    os.makedirs(tac, exist_ok=True)
    tl = Timeline(half_s)
    sim = Sim(seed, tl)
    sim.players()
    sim.run_half(1)
    sim.run_half(2)
    sim.phase_tags()
    sim.observe()

    stem = os.path.join(tac, game + "_tactical")
    team_of_slot = np.array([H] * HOME_SLOTS + [A] * (N_SLOTS - HOME_SLOTS))
    params = [("first_half_start", tl.k1 + 1), ("first_half_end", tl.h1[1] + 1),
              ("second_half_start", tl.k2 + 1), ("second_half_end", tl.h2[1] + 1)]
    write_meta(stem + "_atd.xml", tl.n, FPS, PITCH, team_of_slot, (HOME_NAME, AWAY_NAME), params,
               sim.home_dir[1])
    write_raw(stem + "_atd.txt", sim.slots, sim.ball_obs, PITCH, FPS)
    write_homography(stem + "_homography.csv", tl.n)
    write_tags(stem + "_smart_tagging.xml", sim.tags)
    write_gt(os.path.join(out, "gt.csv"), sim.label)

    blocks = [tl.h1, tl.h2]
    meta = {
        "game": game, "seed": seed, "n_frames": tl.n, "fps": FPS, "pitch": list(PITCH),
        "half_blocks": [[1, tl.h1[0], tl.h1[1]], [2, tl.h2[0], tl.h2[1]]],
        "kickoffs": {"1": tl.k1, "2": tl.k2},
        "kickoff_team": {"1": "A", "2": "H"},
        "home_attack_dir": {"1": sim.home_dir[1], "2": sim.home_dir[2]},
        "stats": dict(spell_stats(sim.label, blocks),
                      ball_missing_alive=sim.ball_missing_alive,
                      n_tags=len(sim.tags)),
    }
    with open(os.path.join(out, "fake_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    return out, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="BAR-FAKE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--half-s", type=float, default=900.0)
    parser.add_argument("--out", default=None, help="output root (default fakedata/out)")
    args = parser.parse_args()
    path, meta = generate(args.game, args.seed, args.half_s, args.out)
    print("wrote", os.path.relpath(path, ROOT))
    print(json.dumps(meta["stats"], indent=2))
