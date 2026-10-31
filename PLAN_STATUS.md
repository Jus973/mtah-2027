# PLAN_STATUS: running state

Last updated: Sat 31 Oct 2026, agent session 2 (PLAN.md revised for the rule changes).

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

- `CLAUDE.md`, `.claude/settings.json` (deny rules include `/Users/justinzhang/mtah2027/**`),
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
- 17 tests pass with no real data present (`make test`, ~4 s).

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
    formation was stretched: before, 52% of open play looked like a line-up; now 2%. Real-data
    result pending a human run.

## TODO for humans (run locally; paste back only the printed lines)

1. ~~**Real-data M0 checks.**~~ cache + baseline-check PASS (31 Oct); anchors FAIL, see 7. Paste back the `[PASS]/[FAIL]` lines and the aggregate lines.
   ```bash
   cd ~/Documents/github/mtah-2027 && export MTAH_DATA_DIR=/Users/justinzhang/mtah2027
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
7. ~~Kick-off diagnosis~~ done 31 Oct. **Detector rebuilt** (formation break + ball near
   centre, ±25-frame tolerance). Re-run `make anchors` and paste back the output.
4. **Emails** (Appendix D): send 1 (cloud assistants), the **new** 2 (Smart Tagging as a
   label-free selection signal) and 3 (deadline hour). RULES.md also invites rule questions as
   GitHub issues; anything team-specific should go by email.
5. ~~Stray bundle copies~~ deleted 31 Oct. `~/Downloads/tactical/` and `~/Downloads/tactical 2/` each hold a copy
   of `BAR-ATH_tactical_smart_tagging.xml`. Consolidate into `~/mtah2027` and delete them, since
   the T&C require tracking and deleting all copies.
6. Review the revised PLAN.md (§1, §5 M1–M3, §6) and object to anything you disagree with.

## Blocked / known limitations

- `make autopsy candidate sync submission doc` are stubs (M1+).
- Fake data: the ball group's third field is written as speed (the real semantics are unverified;
  nothing reads it). No goals or after-goal kick-offs, no duplicate tracks, and home share ≈ 0.5,
  so the home-share band does not apply to fake GT.
- The kick-off detector assumes the ball is visible resting on the spot. If the real ATD loses
  the ball at kick-off it returns `not_found`, and a formation/halfway-crossing fallback is needed.

## Next steps

M1 (after the human checks pass): first `learn/degrade.py` on DFL/IDSSE and its acceptance test,
since it is the selection signal. Then the autopsy table (bug check only), touch detector, dead-ball
v1, sparse Viterbi decoder, candidate v1 and an early Form upload.
