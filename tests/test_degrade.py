"""Sim on degraded open data: DFL parsing, the degradation model, the bundle it writes, its stats."""

import os

import numpy as np
import pytest

from mtah import config, registry
from mtah.eval.score import score
from mtah.features.baseline import nearest_player_codes
from mtah.io.atd import load_game
from mtah.learn import clean, degrade, idsse, stats

CFG = config.load("sim.yaml")
DCFG, MCFG = CFG["degrade"], CFG["measure"]


def make_clean(seed=0, n_half=3000, pad=100):
    """11 v 11 random walkers; the ball sits with a player of the owning team; owner flips every
    ~100 frames; a 150-frame stoppage every 600 frames."""
    rng = np.random.default_rng(seed)
    T = 3 * pad + 2 * n_half
    halves = np.array([[pad, pad + n_half - 1], [2 * pad + n_half, 2 * pad + 2 * n_half - 1]])
    graded = np.zeros(T, bool)
    for a, b in halves:
        graded[a:b + 1] = True
    steps = rng.normal(0, 0.15, (T, 22, 2))
    players = np.clip(np.cumsum(steps, axis=0) + rng.uniform([10, 10], [95, 58], (22, 2)),
                      [1, 1], [104, 67])
    players[~graded] = np.nan
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    owner = np.full(T, -1, dtype=np.int8)
    alive = np.zeros(T, bool)
    ball = np.full((T, 2), np.nan)
    for a, b in halves:
        cur = 0
        for t in range(a, b + 1):
            if (t - a) % 100 == 0:
                cur = 1 - cur
            owner[t] = cur
            alive[t] = (t - a) % 600 >= 150
            ball[t] = players[t, 11 * cur + (t // 100) % 11] + 0.5
    return clean.CleanMatch(name="TEST{}".format(seed), fps=25, pitch=(105.0, 68.0),
                            players=players, team_of_player=team, ball=ball, alive=alive,
                            owner=owner, halves=halves, team_names=("Home", "Away"))


@pytest.fixture(scope="module")
def match():
    return make_clean()


def test_gt_labels(match):
    lab = match.gt_labels()
    g = match.graded()
    assert set(lab[~g]) == {"X"}
    assert np.all(lab[g & ~match.alive] == "D")
    alive = g & match.alive
    assert np.all(lab[alive] == np.where(match.owner[alive] == 0, "H", "A"))


def test_degrade_is_deterministic(match):
    a = degrade.degrade(match, DCFG, MCFG, seed=1)
    b = degrade.degrade(match, DCFG, MCFG, seed=1)
    c = degrade.degrade(match, DCFG, MCFG, seed=2)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[1], c[1], equal_nan=True)


def test_ball_mechanisms(match):
    rng = degrade.rng_for(0, match.name)
    g = match.graded()
    only_stoppages = degrade.ball_missing(match, DCFG, MCFG, rng, q=1.0, f_mean=1.0, s=0.0)
    assert only_stoppages[~g].all()                      # never a ball outside the halves
    lag = DCFG["stoppage_lag_frames"]
    assert not (only_stoppages & g & match.alive).any()  # stoppage gaps stay inside stoppages
    dead = g & ~match.alive
    assert only_stoppages[dead].mean() > 1 - (lag + 1) / 150
    none = degrade.ball_missing(match, DCFG, MCFG, rng, q=0.0, f_mean=0.5, s=0.0)
    assert not none[g].any()
    play = degrade.ball_missing(match, DCFG, MCFG, rng, q=0.0, f_mean=0.5, s=50.0)
    assert play[g & match.alive].any()


def test_player_slots(match):
    slots = degrade.player_slots(match, DCFG, MCFG, degrade.rng_for(0, match.name), 100)
    home, away = DCFG["slots"]
    assert slots.shape[1] == home + away
    g = match.graded()
    finite = np.isfinite(slots[..., 0])
    assert not finite[~g].any()
    # Every visible track is one of its team's players (or a duplicate within a few metres).
    for team, cols in ((0, slice(0, home)), (1, slice(home, None))):
        own = match.players[:, match.team_of_player == team]
        d = np.linalg.norm(slots[:, cols, None, :] - own[:, None, :, :], axis=-1)
        nearest = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=2)[finite[:, cols]]
        assert np.percentile(nearest, 99) < 5.0
    n_home = finite[g, :home].sum(axis=1)
    assert 9.0 < n_home.mean() <= 11.5


def test_write_sim_round_trip(match, tmp_path):
    out, meta = degrade.write_sim(match, DCFG, MCFG, seed=0, out_root=str(tmp_path), game="SIM-T")
    assert registry.is_fake("SIM-T", out)
    sp = registry.spec("SIM-T", out)
    assert sp["n_frames"] == match.n_frames
    assert [tuple(b[1:]) for b in sp["half_blocks"]] == [tuple(h) for h in match.halves.tolist()]
    atd, stats_ = load_game("SIM-T", out)
    assert stats_["n_lines"] == match.n_frames and stats_["n_bad_lines"] == 0
    assert list(np.bincount(atd.team_of_slot)) == DCFG["slots"]
    s = score(np.where(nearest_player_codes(atd) == 0, "H", "A"), os.path.join(out, "gt.csv"))
    assert 0.0 <= s["spell_f1_mean"] <= 1.0


def test_hazard_rates_shrink_to_margins():
    rates = degrade.hazard_rates(DCFG)
    assert rates.shape == (2, 6, 4) and np.isfinite(rates).all() and (rates > 0).all()
    empty = np.array(DCFG["hazard"]["edge"])[3, 3]      # [0, 0] seen/onsets
    assert empty.tolist() == [0, 0] and rates[1, 3, 3] > 0


def test_stats_counts_gaps():
    T = 400
    ball = np.tile([50.0, 30.0], (T, 1)) + np.arange(T)[:, None] * 0.1
    ball[100:110] = np.nan                               # one 10-frame gap
    ball[200:201] = np.nan                               # one 1-frame gap
    players = np.full((T, 2, 2), np.nan)
    players[:, 0] = ball + 1.0
    s = stats.atd_stats(players, ball, np.array([0, 1]), np.ones(T, bool), (105.0, 68.0), MCFG)
    assert s["n_gaps"] == 2 and s["hazard"][..., 1].sum() == 2
    assert s["gap_hist"].sum() == 2
    assert abs(s["ball_missing_frac"] - 11 / T) < 1e-12


def _dfl_xml(tmp_path, match_id="J0TEST"):
    info = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-{0}" HomeTeamName="Home FC" HomeTeamId="CLU-H" GuestTeamName="Away FC" GuestTeamId="CLU-A" />
<Environment PitchX="105.00" PitchY="68.00" />
</MatchInformation></PutDataRequest>""".format(match_id)
    rows = ['<?xml version="1.0" encoding="UTF-8"?>', "<PutDataRequest>", "<Positions>"]
    for section, n0 in (("firstHalf", 10000), ("secondHalf", 100000)):
        for team, pid, x in (("CLU-H", "P-H1", -10.0), ("CLU-A", "P-A1", 10.0),
                             ("referee", "REF", 0.0), ("BALL", "B", 0.0)):
            rows.append('<FrameSet GameSection="{}" MatchId="DFL-MAT-{}" TeamId="{}" PersonId="{}">'
                        .format(section, match_id, team, pid))
            for i in range(5):
                extra = (' BallPossession="{}" BallStatus="{}"'.format(1 if i < 3 else 2, 0 if i == 4 else 1)
                         if team == "BALL" else "")
                rows.append('<Frame N="{}" T="t" X="{:.2f}" Y="{:.2f}" S="0.00"{}/>'
                            .format(n0 + i, x + i, -1.0, extra))
            rows.append("</FrameSet>")
    rows += ["</Positions>", "</PutDataRequest>"]
    (tmp_path / "DFL_02_01_matchinformation_DFL-COM-X_DFL-MAT-{}.xml".format(match_id)).write_text(info)
    (tmp_path / "DFL_04_03_positions_raw_observed_DFL-COM-X_DFL-MAT-{}.xml".format(match_id)).write_text(
        "\n".join(rows) + "\n")
    return match_id


def test_idsse_parser(tmp_path):
    mid = _dfl_xml(tmp_path)
    m = idsse.load(mid, open_dir=str(tmp_path), use_cache=False)
    pad = idsse.PAD
    assert m.players.shape == (3 * pad + 10, 2, 2)                  # referee dropped
    assert m.team_of_player.tolist() == [0, 1] and m.team_names == ("Home FC", "Away FC")
    assert m.halves.tolist() == [[pad, pad + 4], [2 * pad + 5, 2 * pad + 9]]
    assert np.allclose(m.ball[pad], [52.5, 33.0])                   # centre origin -> corner
    assert np.allclose(m.players[pad + 1, 0], [42.5 + 1, 33.0])
    assert m.alive[pad:pad + 5].tolist() == [True, True, True, True, False]
    assert m.owner[pad:pad + 5].tolist() == [0, 0, 0, 1, 1]
    assert m.gt_labels()[pad:pad + 5].tolist() == ["H", "H", "H", "A", "D"]
    assert np.isnan(m.ball[:pad]).all() and np.isnan(m.players[pad + 5:2 * pad + 5]).all()
