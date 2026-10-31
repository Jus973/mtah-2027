# PLAN_STATUS: running state

Last updated: Sat 31 Oct 2026, agent session 2. M0 done. M1: sim v0 accepted (smooth50 0.261 vs
reference 0.243); public check is verdict-only; touches + owner hypotheses done, H1 picked
(held-out sim spellF1 0.606 with oracle D, vs 0.403 for smooth50 + oracle D).

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
6. Review the revised PLAN.md (§1, §5 M1–M3, §6) and object to anything you disagree with.

## Blocked / known limitations

- `make autopsy candidate sync submission doc` are stubs (M1+).
- Fake data: the ball group's third field is written as speed (the real semantics are unverified;
  nothing reads it). No goals or after-goal kick-offs, no duplicate tracks, and home share ≈ 0.5,
  so the home-share band does not apply to fake GT.
- The airlock guards against accidents (per-frame arrays, raw lines, XML, crash messages), not a
  determined agent: Python run by the agent could still open files. The gate, hook and deny rules are
  layers; CLAUDE.md is the rule.
- Whether a cloud assistant may receive aggregates at all is the open T&C question (email 1). The
  airlock sends the same aggregates that pasting did, with mechanical checks instead of a human eye.

## Next steps

M1, in order:
1. ~~Public bug check~~ done (verdict-only).
2. ~~Touches + owner hypotheses~~ done: H1 (rules.yaml), H2 kept (rules_h2.yaml).
3. Dead-ball v1 (`events/deadball.py`): out of bounds (the real ATD does not clamp), ball still +
   players static, long ball gaps (the real ATD loses the ball mostly in stoppages), tag restarts.
   Tune and measure on the sim against its D; replaces the oracle D above.
4. Gap policy and the sparse Viterbi decoder; latency residuals on the sim.
5. Candidate v1 on the real match through the airlock: bands, public bug check (verdict only),
   then the early Form upload (human).
6. Smart-Tagging-style set-piece tags for the sim from the DFL events (sim bias 5), needed to
   test tag-based restarts on the sim.
