# PLAN_STATUS: running state

Last updated: Sat 31 Oct 2026, agent session 2. **Every agent-side item of PLAN.md M0–M4 is done
or closed with a measured reason, including the cold clean-clone and Docker (arm64 + amd64)
reproductions, and the public repo (https://github.com/Jus973/mtah-2027). What is left is human:
register the team name, send the two drafted emails, and upload when the Form arrives.** The final file is `make submission` from `configs/final.yaml` (candidate v3:
learned takeovers + Viterbi), sha256
`00f4551f83d84c891e0c13c5d62fa0e8d0918a749c16e97722edaa32f1bc499a`. A cold clean clone (twice)
and Docker on arm64 and amd64 give the same file.
- Held-out sim: 0.584 (v1 0.541), better on 7/7 sim matches.
- Real match: validator, bands and the public check pass; Smart-Tagging consistency 0.740;
  stable under ±10% threshold changes.
- Technical document: `docs/technical.md` (≈ 5.5 pages; `make doc`).

## ⚠ Rule changes since PLAN.md was written (official repo, 22–23 Oct 2026, commit f8f2629)

The current RULES.md / FAQ.md contradict several parts of PLAN.md. **The rules win.** Quotes are
from RULES.md:

1. **No hand labels on the provided game, for any purpose.** "No hand labels on the provided game,
   in any amount and for any purpose — not for the submission, not to train a model and not to
   validate one." This removes PLAN §6.1 (25-minute hand-labelled validation set), the M2 T2
   track, `tools/labeler/`, `validation/val_gt.csv`, Appendix D email 2, and decision-rule
   step 6.4.2 as written. Hand labels on **other games** are allowed in any amount (disclose them).
2. **The public 5 minutes are not a tuning set.** "They are not training data and not a tuning
   set." That affects the following PLAN items:
   - §2.3 and M1 `decode/latency.py`: "calibrate latency on the 20 public transitions" → calibrate
     on sim/open data instead.
   - M1 "pick the hypothesis whose spell structure matches, then grid r_c, n_c, v_c, t_c" on the
     public window → that is choosing between versions; do it on sim/open data.
   - M1 Smart-Tagging `tag_offset_frames` learned from public set pieces → learn the convention
     on other data or derive it without GT.
   - Checking correctness on the public window (alignment, team sign, file format) is still
     allowed.
3. **No rules or constants tied to specific frames of the game.** Watching the video to debug is
   fine; "turning what you saw into labels, corrections, or rules and constants tied to specific
   frames of the game is" not.
4. Any human-produced data about this game (event feeds, corrected tracking) is forbidden. This
   matches PLAN §3.

**Resolved 31 Oct: PLAN.md revised** (see the banner at its top). Selection now rests on
held-out simulation over degraded open data (§6.1). Label-free checks on the real match back it up:
Smart-Tagging consistency, structural bands and stability (§6.2). The public window is a pass/fail
bug check (spellF1 ≥ 0.35), never compared across candidates. The hand-label labeler and
validation set are gone, replaced by a debug-only clip viewer. `learn/degrade.py` and its acceptance
test moved from M2 into M1. Appendix D email 2 now asks the organisers to confirm Smart Tagging as a
selection signal.

## Done (agent, M0)

- `CLAUDE.md`, `.claude/settings.json` (deny rules include `/path/to/mtah2027/**`),
  `.gitignore`, `.cursorignore`, and a pre-commit guard (`scripts/check_no_data.py`) that blocks
  data dirs, bundle file types and files > 5 MB.
- Official repo vendored unmodified @ `f8f2629` (`evaluation/`, `starter/`, `games.py`). sha256
  manifest in `third_party/`; a test fails if any vendored file changes.
- Pinned env: Python 3.11.14, `requirements.txt` (pip freeze: numpy 2.4.6, pandas 3.1.0,
  kloppy 3.18.0, scipy 1.17.1, lightgbm 4.7.0, PyYAML 6.0.3, pytest 9.2.0, pre-commit 4.7.0).
- `src/mtah/`: `registry` (real + fake matches), `io/atd` (streaming parser → float64 arrays,
  npz cache outside repo), `io/tags`, `features/baseline` (vectorised starter baseline),
  `features/context.attacking_direction`, `events/restarts.detect_kickoffs`, `eval/score`
  (wraps the official scorer), `eval/bands` (§6.3), `submit/write`.
- `fakedata/generate.py`: synthetic 2-half match in the bundle layout. Its EPTS XML loads in
  kloppy 3.18.0 with no warnings, and kloppy's arrays equal ours.
- `scripts/`: `run_cache.py`, `baseline_check.py`, `run_anchors.py`, `sim_costs.py`.
  Make targets: `setup fake test cache baseline-check anchors sim-costs score`.
- **Airlock (31 Oct):** `make airlock T=<cache|baseline-check|anchors|anchors-diag|probe>` lets the
  agent run real-data targets itself. Scripts report through `put()/check()` (`src/mtah/airlock.py`). In a
  terminal these print as before; under the airlock they become JSON, released to the agent only if they
  pass the aggregate policy. Child stdout/stderr and crash messages stay in `<cache>/airlock/*.log`, and
  every release is appended to `<cache>/airlock/released.jsonl` (exactly what the agent saw).
  `registry.gate` refuses the real bundle without a terminal or the airlock.
  `.claude/hooks/guard_data.py` (PreToolUse) blocks agent commands that name the bundle or cache path or
  set the data variables. The bundle path lives in `.mtah.local` (gitignored).
- 41 tests pass with no real data present (`make test`, ~6 s), including airlock leak tests: a
  corrupted raw line whose crash message would quote it is withheld.

## Measured numbers

- `scripts/sim_costs.py` on the public GT reproduces every row of the PLAN §2 table
  (e.g. 25 frames late 0.776 / 0.933; spurious 0.5 s flicker 0.550 / 0.991).
- Fake match (seed 0, 2 × 10 min): 115 spells, median 5.4 s, 23% < 2 s, 56% < 6 s; ball missing
  on 41.4% of live graded frames; 21.3 visible players on graded frames.
- On the fake match: kick-offs detected at exactly the true frames, attacking direction correct in
  both halves, vectorised baseline identical to the starter on every frame, and swapping H/A
  takes the fake GT's spellF1 to exactly 0.
- **Real bundle, human run 31 Oct:**
  - `make cache`: all checks PASS. 199,319 contiguous lines; ball on 113,488 frames overall and
    94,214 graded; 20.49 visible objects on graded frames; home = Barcelona. Slots: 24 home and
    22 away (not 23/23).
  - Ball outside [0,1] on 631 frames (x) and 4,333 frames (y): **the ATD does not clamp**, so
    out-of-bounds is usable as a `D`-onset detector (M1).
  - `make baseline-check`: all 4 baselines **byte-identical**; public smooth50 0.3256 and
    smooth125 0.4242 exact. Frame alignment, team mapping and parsing are proven (M0 falsifier
    cleared).
  - `make anchors`: **FAIL.** Kick-off not found in either half. Both tag windows do contain the
    published frames (half 1 tag 11097–11222, half 2 tag 113660–113785), so the time base is
    fine. The ball-at-spot or formation criterion is what fails. Diagnose with `make anchors-diag`.
  - LightGBM now loads (libomp installed).
  - `make probe`: **both videos have audio** (AAC stereo 48 kHz), so T3 audio sync and whistle
    cues are in scope. Tactical video: h264 1920x1080, 25 fps, 199,319 frames. Broadcast:
    164,487 container frames vs 164,488 ATD records (note for sync).
  - `make anchors-diag`: the tracked ball never rests on the spot (closest 1.38 m in half 1 and
    3.34 m in half 2; median per-frame step ~0.3 m, i.e. jitter). The formation is present (own-half
    fraction holds ≥ 0.90 for 351 frames in half 1) but peaks right at the 0.9 threshold. Ball
    normalised range reaches y = 1.489 (≈ 33 m off the pitch), so outlier cleaning is needed.
    Proposed fix: formation-break-based kick-off with the ball as a loose check, ±1 s tolerance,
    reporting the offset; the pipeline uses the published kick-offs from games.py / the starter.
  - **Kick-off detector v2 (31 Oct):** line-up = both teams ≥ 80% in own half (≥ 1 s, 5-frame
    gaps bridged). Candidates are ranked by ball near the centre around the break, then by distance
    from the tag. The kick is refined to the last frame of the closest-approach near-centre ball
    run. On 10 fake seeds × 2 halves, all are within +1 frame. The fake generator's in-play
    formation was stretched: before, 52% of open play looked like a line-up; now 2%.
  - `make anchors` with v2 (human run, then agent run via the airlock, identical): **all PASS.**
    Detected 11189 and 113746, both **+4 frames** (0.16 s) from the published kick-offs; method
    formation+ball, one candidate per half; line-up held 358 / 355 frames and broke 15 / 18 frames
    after the kick. Direction: home attacks +x in half 1, −x in half 2. Not tuned to remove the +4
    (that would be a frame-specific constant); the pipeline uses the published kick-offs.

## M1 progress (31 Oct)

**Sim v0 (PLAN §6.1) is built and accepted.** Pipeline: `make open-data` (DFL/IDSSE, 7 matches,
2.63 GB, CC BY, MD5-checked, to `~/opendata/idsse`) → `learn/idsse.py` (DFL positions → CleanMatch:
true positions, ball, BallStatus, BallPossession; referees dropped) → `learn/degrade.py` → `make sim`
writes `~/opendata/sim/DFL-<id>/` in the bundle layout with `gt.csv` (D where the ball is dead,
else the possessing team). Fit matches: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY. **Held out for
selection: J03WQQ, J03WR9.**

- **Real ATD degradation stats** (`make airlock T=atd-stats`, label-free, graded frames):
  - Ball missing on 40.7% of frames, but in only 149 gaps, median 268 frames. The lengths are
    bimodal: 16 gaps < 26 frames, 133 of ≥ 101 frames, none in between.
  - Players move half as fast while the ball is missing (median 1.00 vs 1.93 m/s), so **the ball
    is lost mostly in stoppages.**
  - Tracks are smooth (player |d²| median 1.4 mm); no ball jumps > 5 m; 981 frames with the ball
    > 5 m outside the pitch.
  - Team-wide dropouts (≤ 6 tracks) 27 home / 20 away, mean ~13 s. Extra-track episodes
    (≥ 12 tracks) 78 / 55, mean ~6 s. 1,219 slot tracklets, mean ~100 s.
  - The fake generator is far from this (many short gaps); it stays a test fixture only.
- **Degradation model:** stoppage loss (each dead stretch with prob. q, for a fraction f of it) +
  open-play gaps at the real hazard by ball speed × crowding × pitch edge (scaled by s) + tracklets,
  dropouts, duplicates and slot churn at the real rates. Fitted (`make sim-fit`): q 1.0, f_mean 0.7,
  s 0.4, g 100 frames. Achieved vs real: missing 0.415 / 0.407, gaps per 1,000 frames 0.89 / 0.94,
  median gap 342 / 268, speed ratio 0.59 / 0.517, P(11 | 7–11 tracks) 0.627 / 0.649.
- **Acceptance test** (`make sim-check`, official baseline logic, official scorer): PASS.
  | | raw | smooth25 | smooth50 | smooth125 |
  |---|---|---|---|---|
  | sim mean (7) | 0.103 | 0.258 | **0.261** | 0.192 |
  | real (official) | 0.049 | 0.200 | 0.243 | 0.224 |
- **Known sim biases (read before trusting a sim ranking):**
  1. The sim GT is more fragmented than the real one: 340–477 spells per 90 min, 25–37% under 2 s
     (inferred real: ~20%). DFL's possession flips more often than Metrica's mechanical GT, so the
     sim favours fast-reacting methods (it ranks smooth25 ≈ smooth50 > smooth125, while the real
     match ranks smooth50 > smooth125 > smooth25). Treat sim differences in *time-scale*
     parameters as coarse; differences in *event detection* should transfer better.
  2. Raw per-frame nearest player scores 2× higher on the sim (0.10 vs 0.05): sim tracks are
     cleaner near the ball than real ones (extra tracks are offset copies, not referees or
     mislabelled objects).
  3. The real ball is lost in stoppages even more than q = 1 reproduces (speed ratio 0.59 vs 0.517).
  4. J03WN1 and J03WQQ include a red card (10 v 11), as real matches can.
  5. No Smart Tagging in the sim yet: tag-based components cannot run on it until set-piece tags
     are derived from the DFL events.

**Rules v1: touches and owner hypotheses (31 Oct).** `features/ball.py` (drop > 10 m off the
pitch and > 40 m/s steps, fill gaps ≤ 10 frames), `features/players.py` (per-team nearest
player), `events/touches.py` (contact episode = ball within r of a player, nearest team constant;
a touch if the ball stays close ≥ min_frames, turns ≥ min_turn_deg, or changes speed
≥ min_dv_mps), `decode/rules.py` (H1 any touch, H2 controlled touch, H3 two consecutive
same-team touches; flip on the touch's first frame; after D the first touch decides).
`make sim-rules` tunes each hypothesis on the fit matches and compares them on the held-out ones,
with **D taken from the sim GT** (owner rule alone; dead-ball v1 is next):

| held-out (J03WQQ, J03WR9) | spellF1 | macroF1 | spells / GT | transition F1 @25 |
|---|---|---|---|---|
| baseline smooth50 | 0.286 | 0.642 | 1.13 | 0.16 |
| smooth50 + oracle D | 0.403 | 0.817 | 1.01 | 0.49 |
| **H1 + oracle D** (rules.yaml) | **0.606** | 0.913 | 0.96 | 0.70 |
| H2 + oracle D (rules_h2.yaml) | 0.585 | 0.907 | 0.94 | 0.68 |
| H3 + oracle D | 0.504 | 0.895 | 0.52 | 0.62 |

- Tuned touch settings (H1): r 1.0 m, min_frames 12, turn ≥ 20°, Δv ≥ 12 m/s, all inside their
  grids. H2's control parameters went to their "off" ends (n_c 2, v_c 1.0, t_c 0.25 s), i.e. on
  the sim a strict touch filter makes the control test redundant.
- **H1 picked** (wins both held-out matches, fewest parameters: §6.4). **Caveat:** DFL's
  possession flips on touches more readily than the real GT ("touches, deflections, unwon aerials
  are not changes"), which favours H1 (sim bias 1). Re-check H1 vs H2 on the real match with
  label-free checks (Smart-Tagging consistency, bands) once dead-ball v1 exists.
- **Transfer check, label-free** (`make airlock T=touch-stats` vs the same script on sim
  matches): ball within 1.0 m of a player on 28.7% of real ball-seen frames vs 28.0–29.7% on the
  sim; nearest-player distance p10/p25/p50 0.44/0.89/1.90 m real vs 0.51/0.85–0.90/2.2 m sim;
  nearest-team switches 28.8 vs 26–27 per minute; touches at r = 1.0 m 21.4 vs 24.7–25.9 per
  minute; real contact episodes run longer (median 12 vs 8 frames). The sim-tuned radius should
  transfer; the real ball "sticks" to a player a little longer.

**Dead-ball v1 (31 Oct).** `events/deadball.py`: evidence = ball out of bounds (observed beyond
a line by > 0.5 m for ≥ 15 frames), long ball gaps (≥ 25 frames), stillness (ball and players
calm); evidence runs within 50 frames merge, and each is D up to the restart touch. `make sim-dead`
tunes it end to end (H1 spellF1) on the fit matches:

| held-out | spellF1 | alive/dead F1 | transition F1 @25 |
|---|---|---|---|
| H1 + oracle D | 0.606 | 1.000 | 0.70 |
| **H1 + detected D** (rules.yaml) | **0.541** | 0.889 | 0.58 |
| H1 + D from gaps only | 0.534 | 0.878 | 0.58 |
| H1 + D out of bounds + stillness (no gaps) | 0.454 | 0.797 | 0.47 |
| H2 + detected D | 0.522 | 0.892 | 0.57 |
| official baseline smooth50 | 0.286 | – | 0.16 |

- **Gaps carry the detector, and that is partly circular:** the sim loses the ball in stoppages
  by construction (fitted q = 1). The real ATD supports it even more strongly (speed ratio 0.517
  real vs 0.59 sim), and the no-gap variant still beats the baseline (0.454 vs 0.286).
  Stillness adds nothing on the sim (tuned to effectively off).
- **D is over-predicted:** 51–52% of graded frames vs 41% in the sim GT (precision 0.79, recall
  0.98). H1 still beats H2 with detected D.
- **Real match, label-free** (`make airlock T=dead-stats`): D on 49.7% of graded frames (likely
  over by ~10 points, as on the sim); 135 D runs (band 60–150: PASS); 86% of the 57 set-piece tags
  start inside a predicted D run (chance ≈ 50%). Set-piece tags are fixed 35 s windows and the
  predicted restart falls **9.5 s after the tag start** (IQR 6.0–13.2 s): stored as
  `tags.set_piece_restart_after_start_s` for tag-anchored restarts later.

**Candidate v1 (31 Oct)** — `make airlock T=candidate NAME=v1` (rules.yaml, H1), git 4cbeed1,
sha256 `3c32c4758d117a6a…`, file `submissions/candidates/v1.csv` (gitignored, never read by the
agent). Everything below is label-free or verdict-only:
- Official validator PASS. Bands (§6.3) all PASS: 354 spells, 22.9% < 2 s, 55.1% < 6 s, mean
  8.8 s, 133 D runs, home share of alive 0.61, X 0, kick-off team right in both halves, D before
  the second-half kick-off. D on 50.8% of graded frames (probably ~10 points high, as on the sim).
- Public bug check: **pass** (spellF1 ≥ 0.35; number not printed); team sign right.
- Smart-Tagging consistency 0.740. The tagger emits each phase twice (attacking team + defending
  opponent, same window) and each pair agrees exactly, which confirms the attack/defend mapping.
- Held-out sim spellF1 of the same config 0.541.
- **H1 vs H2 re-checked on the real match:** H2 (`v1-h2`) has lower consistency (0.714), too
  many short spells (421; < 6 s band FAIL), fails both kick-off-team bands, and scores lower on
  the held-out sim (0.522). H1 stays.
- Re-running `make airlock T=candidate NAME=v1` regenerates the same sha256 (checked).

**Dead-ball v2: cutting the D over-prediction (31 Oct).** Diagnosis on the fit matches (v1,
open data):
- **Where the false D comes from.** Predicted D 56.6% vs true 46.3%. Of the false-D frames:
  - 37% spurious runs: 85 runs, all from ball gaps in open play;
  - 32% late restarts;
  - 19% early onsets: a gap that starts in play;
  - 12% merges across live play.
- **What it costs.** Partial oracles bound the gain on the fit matches: detected D 0.547, perfect
  D 0.587. True restarts add +0.017, true onsets +0.015, removing spurious runs +0.008.
  **Boundary timing, not spurious stoppages, is what costs spellF1, and all D work is worth at
  most ~+0.04.** The owner rule is the bigger lever: H1 with perfect D is still only 0.603
  held-out.
- **Late restarts.** Mostly throw-ins and corners, taken while the ball is still beyond the line:
  169 runs, median 43 frames late. The taker's touch starts inside the evidence and was skipped.
  A further 33 runs end late because the ball reappears in flight after the restart.

New rules (off in `rules.yaml`, so v1 regenerates byte-identically):
- `deadball.gap_v_players_mps`: gap evidence counts only while players are calm. Sim gaps in play
  have median player speed 2.27 m/s, stoppage gaps 1.22. Real long gaps: p10 / p50 / p90 0.78 /
  1.24 / 2.28 m/s vs sim J03WMX 0.92 / 1.46 / 2.44, so the filter should transfer.
- `deadball.restart_window_frames`: release restarts. The taker's team owns the ball after them.
- `deadball.restart_flight_mps`: flight restarts.
- `touch.min_launch_mps`: launch touches.

`make sim-dead CFG=configs/rules_v2.yaml` tuned them on the fit matches. The held-out row of each
ablation drops one rule:

| held-out | spellF1 | macroF1 | alive/dead F1 | D share (GT 0.413) | D precision |
|---|---|---|---|---|---|
| v1 (rules.yaml) | 0.5406 | 0.833 | 0.889 | 0.515 | 0.786 |
| **v2** (rules_v2.yaml) | **0.5450** | 0.845 | 0.913 | 0.486 | 0.827 |
| v2 without release | 0.5419 | 0.844 | 0.912 | | |
| v2 without the calm-gap filter | 0.5416 | 0.838 | 0.898 | | |
| v2 without flight | 0.5458 | 0.843 | 0.909 | | |
| v2 without launch | 0.5461 | 0.844 | 0.913 | | |

Real match, label-free (`make airlock T=candidate NAME=v2`, `T=dead-stats CFG=...`):
- Validator, all bands and the public bug check pass.
- D share 46.0% (v1 50.8%); 376 spells (v1 354); 141 D runs.
- Smart-Tagging consistency **0.728 (v1 0.740)**. The ~20k frames v2 turns from D into live
  play agree with the tagger only ~60% of the time.
- Predicted restart minus set-piece tag start, p25 / p50 / p75: 5.2 / 7.0 / 11.7 s (v1 6.0 / 9.5 /
  13.2). The spread is slightly tighter.
- 90.6% of long-gap frames pass the calm filter.

**Decision (§6.4):** a near-tie on the sim (+0.004, per-match CIs ±0.035). The tie-breaks (fewer
components, then consistency) both favour v1, so **v1 stays selected**. v2's rules stay in the
code, off by default, for the decoder work: they are evidence, and a Viterbi can weigh them
instead of applying them as hard rules.

**Owner decoder v3 (31 Oct).** Diagnosis with the true D, fit matches, H1:
- 1,771 GT possession changes in play, 1,431 predicted, 894 matched within 1 s: recall 0.50,
  precision 0.62.
- Of the 877 missed, 583 have no touch by the new team within 1 s.
- 203 of the 537 spurious changes flip back within 2 s. They follow the previous touch by 0.12 s
  (median) vs 1.08 s for real takeovers, and turn the ball less (72° vs 111°): duels and
  rebounds.
- **Ceiling:** perfect takeover decisions on all contacts within 1.0 / 1.5 / 2.0 m give fit
  spellF1 0.764 / 0.769 / 0.756, vs 0.584 for H1. That is a +0.18 lever; D is worth +0.04.

Built (`learn/takeover.py`, `decode/viterbi.py`, `make train-takeover`):
- Every contact at 1.0 m is a candidate.
- LightGBM gives P(takeover) for a contact by the non-owning team. It has 15 team-symmetric
  features (kinematics; time to the previous and next contacts and their teams). Training uses the
  5 fit matches only, single-threaded and seeded; the label is the sim GT owner 5 frames after the
  contact.
- A two-state Viterbi over contacts, with the flip on the contact's first frame. α = 3.0 and
  flip penalty 0 were tuned out of fold, both inside their grids.
- The model is committed (`models/takeover_v1.txt`, 510 KB); retraining is byte-identical.
- Logistic regression was much worse (held-out 0.627 vs 0.655 with the true D, prototype).
- Radius chosen out of fold: 1.0 = 1.5 > 2.0 m.
- Pressure features tried: AUC unchanged at 0.913, held-out 0.590 vs 0.584. Dropped (fewer
  components on a tie).

| held-out | J03WQQ | J03WR9 | mean | macroF1 |
|---|---|---|---|---|
| v1: H1 + detected D | 0.522 [0.484, 0.560] | 0.560 [0.526, 0.594] | 0.541 | 0.833 |
| **v3: Viterbi + detected D** | **0.559 [0.519, 0.600]** | **0.609 [0.572, 0.645]** | **0.584** | 0.838 |
| H1 + GT D | 0.579 | 0.632 | 0.606 | 0.913 |
| Viterbi + GT D | 0.621 | 0.676 | 0.649 | 0.918 |

- **Paired:** v3 beats v1 on all 7 sim matches. On the fit matches (v3 out of fold, v1 in-sample)
  the gains are +0.017, +0.016, +0.051, +0.025, +0.047; on held-out +0.037 and +0.049. Sign test
  p ≈ 0.008.
- **Real match** (`make airlock T=candidate NAME=v3`):
  - validator, all bands and the public bug check pass;
  - Smart-Tagging consistency 0.740 (v1 0.740);
  - 312 spells (v1 354), 18.3% under 2 s (v1 22.9%), 51.3% under 6 s (v1 55.1%), mean 10.0 s;
    PLAN §2 infers ≈ 20% / ≈ 50% for the real GT.
- **Decision (31 Oct, team-approved):** §6.4 step 2 is amended to a paired per-match comparison
  (positive mean difference and a one-sided sign test at p ≤ 0.05). **v3 is selected.** The old
  interval-overlap rule called it a tie.
- **Risk:** the takeover labels come from DFL possession, which flips more often than the real GT
  (sim bias 1). v3 predicts 0.8× the sim GT's spells and sits inside the real bands, but a
  semantic mismatch is a bias the sim cannot measure.

## M2–M4 progress (31 Oct, agent; goal: finish the plan)

- **§6.4 amended, v3 selected** (above). `configs/final.yaml` inherits `rules_v3.yaml`.
- **`make submission`** (agents: `make airlock T=submission NAME=<team>`) writes
  `submissions/final/<team>.csv`. That file is gitignored: CLAUDE.md forbids committing
  bundle-derived files, while PLAN §7 says to commit it, so a human decides (`git add -f`). The
  target then runs the official validator and the bands, and records or checks
  `submissions/final/<team>.sha256` (committed). Two runs give the same file:
  sha256 `00f4551f83d84c891e0c13c5d62fa0e8d0918a749c16e97722edaa32f1bc499a`, equal to v3.
- **Latency** (`decode/latency.py`, `make sim-latency`). Median offset of matched transitions,
  in frames (pred − GT):

  | transition | fit | held-out |
  |---|---|---|
  | turnover | −2 | −2 |
  | D onset | +6 | +8 |
  | D end | +1 | +21.5 |

  Moving D onsets 6 frames earlier costs −0.003 on held-out. Gap flips (a takeover moved into
  the preceding ball gap) tune to a no-op on fit. **No correction**: transitions stay where they
  are placed by construction. The D-onset lag partly reflects the sim's assumed stoppage lag.
- **Stability (§6.2c)** (`make airlock T=stability`). 14 thresholds are moved ±10% one at a time,
  and the takeover model is retrained with seeds 1 and 2. Each variant is scored against the
  reference output with the official scorer:
  - real match: min spell agreement 0.854 (`owner.contact_r_m` +10%), median 0.999; seeds
    0.966 / 0.968;
  - held-out sim J03WQQ: min 0.921;
  - **not fragile** (floor 0.5).
- **Disagreement map (§6.5)** (`make airlock T=disagreement`). Share of graded frames where
  candidates disagree: v1 vs v3 2.2%, v1 vs v2 6.0%, v2 vs v3 8.1%; any two 8.1% (< 10% PASS).
  There are 202 segments (median 1.7 s), written to `<cache>/disagreement/segments.csv` for the
  clip viewer.
- **Set-piece checks (M2 T2 b, c)** (`eval/consistency.set_piece_checks`, printed by
  `run_candidate`). Restarts are aligned to tag start + 9.48 s.
  - v3: 32% of the 57 set-piece tags have a predicted D ending within 2 s of the aligned restart
    (chance ≈ 9%). 43% of D runs overlap no set-piece tag; the tagger misses most free kicks.
  - The restart owner matches the tag team in 9 of 16 matched team-tagged restarts: corners 5/11,
    free kicks 3/3, goal kicks 1/2.
- **v4 tried and rejected** (`configs/rules_v4.yaml`: v3's decoder plus v2's dead-ball rules, with
  the taker's release as a forced restart contact).
  - Sim: better on all 7 matches (held-out 0.593 vs 0.584), but its dead-ball parameters are
    in-sample on the fit matches, so the paired rule counts only the held-out pair: a tie.
  - Real match: consistency 0.726 (v3 0.740, a regression); D ending near the aligned restart
    19% (v3 32%); corner restart owner 3/10.
  - v3 stays selected.
- **Tag-anchored restarts (M1 d, M2 boundaries, M3 "+tag anchors"): dropped.**
  - Set-piece tags are fixed 35 s windows. The tracking restart sits 9.5 s after the tag start
    with IQR 6.0–13.2 s, much coarser than the tracking restart itself.
  - On the real match, a tag-anchored rule could only be judged against those same tags
    (circular). There is no sim analogue either: sim tags would need the tagger's timing noise,
    which cannot be measured without labels.
- **T3 audio and broadcast** (`audio/features.py`, `sync/audio_xcorr.py`, `sync/pointset.py`,
  `audio/whistle.py`; `make airlock T=audio|sync|whistle`). Audio is decoded locally with PyAV
  (tactical 7,973 s, broadcast 6,580 s, 49 s to decode), with features cached outside the repo.
  - *Audio sync fails.* 60 s windows of high-passed loudness or whistle-band energy reach a
    median peak correlation of 0.23–0.25 and no confident windows. Without gates the offsets
    scatter, with ≈ 0% ball agreement.
  - *Tracking sync works* (PLAN fallback). The broadcast ATD is in the same pitch coordinates.
    90.5% of windows sync, with a median ball distance of 0.8 m at the best lag. There are two
    offsets: tactical = broadcast + 4,014 frames in half 1 and + 26,920 in half 2 (the broadcast
    skips half-time). But within a half the balls agree within 2 m on only 71–75% of frames.
  - *Per live segment* (runs of valid broadcast homography), only 6 of the 93 judgeable segments
    meet PLAN's acceptance (≥ 90% within 2 m).
  - **T3-broadcast falsifier: FAIL.** The synced broadcast ball recovers 0.6% of tactical-missing
    graded frames (all in predicted D) vs the 10% required. **Dropped.**
  - *Whistle detector.* A spectral heuristic (2–4.5 kHz tonality ≥ 40 and band ≥ median + 10 dB
    for ≥ 120 ms) finds 20 whistles in the match. 0.8% of D onsets have one within 1 s (chance
    0.6%), and neither derived kick-off does. Looser thresholds find up to 19 per minute, but
    only 40% of D onsets have one (chance 24%), and restarts sit below chance. **Gate (≥ 70%):
    FAIL. Dropped.** Whistles are faint in the tactical camera audio. A learned detector
    (PANNs/YAMNet) would need downloaded weights and has no sim analogue, so by M3 it would
    default off anyway.
  - *Commentary ASR* (optional, "only if sync works"): skipped. The sync fails acceptance.
- **M3 ablations** (`make sim-ablate`, held-out, bootstrap 1000; Δ vs final):

  | change | held-out Δ | note |
  |---|---|---|
  | rules only (v1) | −0.043 | |
  | without long-gap evidence | −0.110 | |
  | without out-of-bounds evidence | −0.005 | |
  | without stillness | −0.003 | |
  | α 1 instead of 3 | +0.004 | not re-tuned on held-out: α is tuned out of fold |

  Real-match checks for the two small ones:
  - *Stillness* is inert on this match: the output without it is byte-identical. **Dropped from
    `configs/final.yaml`** (`deadball.kinds: [oob, gap]`). Re-tuning out of fold on the final
    config keeps α 3.0, the retrained model is byte-identical, and `make submission` reproduces
    the same sha256.
  - *Out of bounds* is **kept**, as an exception to M3's < 0.01 drop rule. Without it,
    consistency is 0.742 (vs 0.740), but set-piece alignment falls (D near the aligned restart:
    25% vs 32%) and there are 119 D runs instead of 133. It is the one D evidence that does not
    lean on the sim's stoppage-gap assumption (sim bias 3), so the sim understates it.
- **Video ball recovery go/no-go (M3): no-go by default.**
  - (a) Rules + learned below 0.65 on sim: yes (0.584).
  - (b) ≥ 25% of missed or split spells lie in ball gaps: yes, just (J03WQQ 25.3% of 194,
    J03WR9 27.2% of 162).
  - (c) P3 has ≥ 4 free days: a human call (TODO 10).
  - PLAN's prior is 0.3, and detector weights trained on bundle frames cannot be committed (they
    would need deterministic retraining). Default no-go.
- **Debugging tools (humans only; both refuse the airlock).**
  - `make autopsy` (`scripts/run_autopsy.py`) prints one row of derived scalars per
    public-window GT transition, then summary rates. Per row: whether the ball is seen, the gap
    around it, the losing team's distance, ball speeds, when the new owner's team is first within
    1.0 / 1.5 / 2.5 m, its first contact, the final candidate's offset, and the overlapping tags.
    It is for bug-finding only: nothing from it may become a threshold. The agent never sees it.
  - `make clips` (`scripts/clip_export.py`) writes the clips where v1 / v2 / v3 disagree to
    `<cache>/clipviewer/clips.json`, with players and ball projected through the tactical
    homography (`io/homography.py`).
  - `tools/clipviewer/index.html` plays the tactical video with that overlay and every
    candidate's label per frame, and writes nothing.
  - Tested on the synthetic match: 90 clips, overlay and stepping checked in the browser.

- **M4 reproducibility and audit.**
  - *Clean clone* (`git clone` → `make setup` → 86 tests pass with no match data → `make airlock
    T=submission` twice): the same sha256 both times, equal to the committed hash. The clone was
    deleted afterwards, as it held a bundle-derived CSV.
  - *Fully cold clean clone* (`make airlock T=submission-cold`): a fresh clone and venv, with an
    empty temporary cache that the airlock creates and deletes, so the ATD is parsed from
    scratch. Two runs give the same sha256, equal to the committed hash, in 18 s each including
    the parse.
  - *Dockerfile:* runs `make cache submission` for `TEAM` and copies the file to `/out`;
    `.dockerignore` keeps local, derived and secret files out (checked inside the image).
    `make docker-build` builds it (1 min 42 s, 910 MB). `make airlock T=submission-docker` runs
    it with the bundle mounted read-only and the container's report under the release policy.
    - Linux aarch64: same sha256, 22 s.
    - Linux x86_64 (`PLATFORM=linux/amd64`, emulated): same sha256, 49 s.
    - **The file is identical on macOS arm64, Linux arm64 and Linux x86_64.**
    - After the team-name change, the image's own entry point (`scripts/docker_run.sh`, run by
      `make airlock T=submission-docker` with no command override) gives the same sha256 for
      `Berkeley Sports Research Group.csv` (arm64, 22 s).
  - *`make audit`:* PASS. No bundle files or derivatives, nothing over 5 MB, no secrets,
    `.mtah.local` untracked, LICENSE and README present. It warns about absolute home paths in
    `.claude/settings.json` and `.cursorignore`.
  - *Document:* `docs/technical.md` is rewritten for the final file (v3), ≈ 5.5 pages.

- **Plan items closed without building them** (reasons):
  - *M1 dead-ball (c), formation reset after a goal:* no separate rule. There are 3 goals (4
    tags), and the only gain would be the after-goal restart time. Gap and out-of-bounds evidence
    already make those stretches D, and the sim has no tags to test it.
    `events/restarts.detect_kickoffs` could be pointed at the `Kick Off (After Goal)` tags if
    wanted.
  - *M2 optional hand labels on another match:* skipped, as PLAN allows. DFL's own possession
    labels cover other-game ground truth.
  - *M2 T4 per-frame 3-class emission model:* superseded by the event-level takeover model,
    PLAN's "turnover-at-candidate classifier". The changes it would decide happen at contacts,
    and the per-frame D side is worth ≤ +0.04 even when perfect.
  - *M1/M2 frame-level {H, A, D} Viterbi:* the decoder is event-level over {H, A}, with D from
    the dead-ball rules. Decoding D in the same path is Next step 3.

## TODO for humans

Open, in order. Facts from the organisers' "Initial Challenge" email (20 Oct, read 31 Oct):
- the **deadline is 17 Nov, 11:59 PM CET** (this answers Appendix D email 3);
- the **submission instructions and Form link arrive by email by 14 Nov**;
- **all members must register with the same team name before 10 Nov.**

The team name is **Berkeley Sports Research Group** (`team` in `configs/base.yaml`).

1. **Register the team name** with every member (same spelling) before 10 Nov.
2. ~~Public repo~~ done 31 Oct: **https://github.com/Jus973/mtah-2027** (public, `main`). The final
   CSV stays out of git, per CLAUDE.md; `git add -f` it only if you decide PLAN §7 wins. `make
   audit` warns about one local home path in `.cursorignore`.
3. **Emails.** Drafts 1 (cloud assistants) and 2 (Smart Tagging as a selection signal) are in
   your Gmail, to hackathon.talentarena@mobileworldcapital.com, **not sent**. Email 1 asks about
   aggregates prospectively; the airlock has already released aggregates to an assistant, so
   consider saying so before you send. Email 3 is not needed.
4. **When the Form arrives (by 14 Nov): upload, after you say go.**
   - CSV: `submissions/final/Berkeley Sports Research Group.csv`. Rebuild it right before
     uploading with `make submission`; its sha256 must be
     `00f4551f83d84c891e0c13c5d62fa0e8d0918a749c16e97722edaa32f1bc499a`.
   - Repo URL: https://github.com/Jus973/mtah-2027.
   - Document: `docs/Berkeley Sports Research Group.md`. Rebuild it with
     `make doc REPO=https://github.com/Jus973/mtah-2027` after any change to `docs/technical.md`.
   - Re-open the response and verify all fields; then set `uploaded? = yes` on the FINAL row of
     `submissions/LOG.md`.
5. **Optional:**
   - *Video ball recovery go/no-go (M3).* Conditions (a) and (b) are met; (c), P3 having ≥ 4
     free days, is yours. Default no-go.
   - *Clip viewer and autopsy (§6.5, debugging only).* `make clips`, then open
     `tools/clipviewer/index.html`; `make autopsy`.
   - *PDF of the document:* install pandoc plus a LaTeX engine, then `make doc` writes it.
6. Review the revised PLAN.md (§1, §5, §6, the amended §6.4) and object to anything you
   disagree with.
7. After the competition, delete every bundle copy and the caches (T&C).

Done earlier (kept for the record):

1. ~~**Real-data M0 checks.**~~ cache + baseline-check PASS (31 Oct); anchors FAIL, see 7. Paste back the `[PASS]/[FAIL]` lines and the aggregate lines.
   ```bash
   cd ~/Documents/github/mtah-2027 && export MTAH_DATA_DIR=/path/to/mtah2027
   make cache            # expect 113,488 / 94,214 ball frames, ~20.5 objects, home = Barcelona
   make baseline-check   # expect 4× byte-identical; public smooth50 0.3256, smooth125 0.4242
   make anchors          # expect kick-offs 11185 and 113742; direction flips at half-time
   ```
   If `baseline-check` is not byte-identical, stop: nothing else is trustworthy (PLAN M0 falsifier).
   If `anchors` misses by ±1 frame, report the offset; do not hard-code the answer.
2. ~~**Stream metadata**~~ done: audio on both. (ffprobe is not installed, so this uses PyAV and decodes no frames):
   `make probe`. Paste back the output: audio on both feeds, one or neither decides T3 scope.
3. ~~`brew install libomp`~~ done. Separately, Homebrew reports outdated Command Line Tools;
   update them via Software Update (only matters for future brew installs).
7. ~~Kick-off diagnosis and re-run~~ done 31 Oct: v2 PASS at +4 frames in both halves.
8. **Nothing more to run by hand for M0.** The agent now runs real-data targets with `make airlock`.
   To audit what it has received, read `~/mtah2027_cache/airlock/released.jsonl` (an agent may not).
   The guard hook loads only in Claude Code sessions opened in this repo, so start new sessions there.
4. **Emails** (Appendix D): send 1 (cloud assistants; it asks exactly whether an assistant may
   receive aggregates, which is what both the airlock and pasting do), the **new** 2 (Smart Tagging as a
   label-free selection signal) and 3 (deadline hour). RULES.md also invites rule questions as
   GitHub issues; anything team-specific should go by email.
5. ~~Stray bundle copies~~ deleted 31 Oct. `~/Downloads/tactical/` and `~/Downloads/tactical 2/` each hold a copy
   of `BAR-ATH_tactical_smart_tagging.xml`. Consolidate into `~/mtah2027` and delete them, since
   the T&C require tracking and deleting all copies.
## Blocked / known limitations

- Fake data: the ball group's third field is written as speed (the real semantics are unverified;
  nothing reads it). No goals or after-goal kick-offs, no duplicate tracks, and home share ≈ 0.5,
  so the home-share band does not apply to fake GT.
- The airlock guards against accidents (per-frame arrays, raw lines, XML, crash messages), not a
  determined agent: Python run by the agent could still open files. The gate, hook and deny rules are
  layers; CLAUDE.md is the rule.
- Whether a cloud assistant may receive aggregates at all is the open T&C question (email 1). The
  airlock sends the same aggregates that pasting did, with mechanical checks instead of a human eye.

## Next steps

The agent-side plan is finished, with every component kept or dropped by its gate (sections
above). The human steps are the TODO list. If there is time before the freeze, or for Phase 2:
1. Contacts the 1 m radius misses: possession changes inside ball gaps make up ≈ 25% of v3's
   missed spells on the sim. The candidates would be contacts reconstructed from the gap's
   entry and exit, with the model retrained on the sim.
2. Restart owner after corners (5/11 corner team): a restart-specific feature for the takeover
   model, judged by the paired rule and the set-piece check.
3. Decode D in the same Viterbi path, using the v2 rules as soft evidence instead of hard rules.
Re-check every change with `make sim-ablate`, the paired §6.4 rule, `make airlock
T=candidate|stability`, and `make submission` (sha256 changes need a new FINAL row).
