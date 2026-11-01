<!--
Source of the technical document (PLAN.md §8). `make doc REPO=<url>` writes docs/<team>.md (and
.pdf when pandoc is installed), <team> from configs/base.yaml. HTML comments like this
one are notes for the team and do not render. Every number below comes from a make target;
real-match numbers are label-free aggregates or verdicts (make airlock T=...).
Version of 1 Nov 2026: final candidate v6 (configs/final.yaml). Re-check numbers at M4 freeze.
-->

# Ball possession from automatic tracking: <TEAM_NAME>

More Than A Hack 2027, Phase 1: FC Barcelona (H) vs Athletic Club (A). The submitted file is
produced by `make submission` from `configs/final.yaml` (candidate **v6**). Code: <REPO_URL>.

## 1. Approach

Possession changes only at **contacts** between the ball and a player, and stoppages come from
**dead-ball evidence** in the tracking. We don't smooth a per-frame guess. The pipeline has five
steps:
1. Every contact is a candidate change.
2. A gradient-boosted model, trained only on *other* matches (open DFL data degraded to look
   like this match's tracking), gives each contact by the team without the ball a probability of
   taking it, from the contact's kinematics and both teams' shape around it.
3. A two-state Viterbi over the contacts picks the most likely owner sequence.
4. A `D` stretch starts where the ball is beyond a line or lost by the tracker, and ends at the
   restart.
5. Inside long ball gaps, where there is no contact to see, two more models read the players
   alone (team shape and motion), and a three-state Viterbi re-decides `H`, `A` or `D`.

Every change lands on the first frame of the contact or evidence that causes it, as the ground
truth is defined. On this match we use only label-free checks and a pass/fail public bug check.

Why events: spellF1 matches predicted to true spells one-to-one, same team only, at IoU 0.3–0.7.
Perturbing a perfect file with the official scorer shows what each kind of error costs
(public-window GT, `make sim-costs`):

| Perturbation of a perfect file | spellF1 |
|---|---|
| one spurious 0.5 s opponent flicker inside each spell > 10 s | 0.55 |
| one spurious 1 s `D` inside each spell > 10 s | 0.64 |
| every `D` predicted as continuing possession | 0.81 |
| possessions < 2 s merged into the previous spell | 0.83 |
| all boundaries 25 frames (1 s) late | 0.78 |

A majority-vote window removes flicker only by merging short possessions too, so the official
smoothing baseline levels off near 0.24.

## 2. Assets

| Asset | Used | How |
|---|---|---|
| Tactical ATD | yes | Every component. Our parser regenerates the four official baselines byte-for-byte. |
| Smart Tagging | yes | The `Kick Off (Start)` windows bound each half's kick-off search. Label-free consistency and set-piece checks (§5.4). It sets no label. |
| `games.py` | yes | Graded half ranges (`X` only outside them), pitch size. |
| Tactical homography | tools only | Projects tracking onto the video in the human clip viewer. |
| Broadcast ATD, both videos' audio | tried, dropped | Measured reasons below. |
| DFL/IDSSE open data | yes | Simulation, training, tuning and selection (§4, §5). |

**Kick-offs.** Each half's kick-off is where both teams' line-up (≥ 80% in their own half for
≥ 1 s) breaks inside the tag window. It lands 4 frames after the published frame, within our 1 s
tolerance, uncorrected. Before the derived second-half kick-off, the graded block is `D`.

**Tried and dropped, with the measured reason** (held-out sim spellF1 unless stated; §5):
- *Other owner rules:* *H2* (controlled touch) 0.522; *H3* (two touches) 0.504 with the true `D`.
- *Logistic regression* for takeovers (0.63 vs 0.65 with the true `D`); *pressure features*
  (players within 3 / 8 m: held-out AUC unchanged).
- *Stillness as `D` evidence:* inert on this match; the output is byte-identical without it.
- *Four extra `D` rules* (calm-gap filter, release and in-flight restarts, kicks from rest):
  +0.004 / +0.009, ties under the paired rule (§5.3), with consistency down to 0.728 / 0.726.
- *Latency corrections:* earlier `D` onsets −0.003; takeovers moved into a gap: a no-op.
- *Tag-anchored restarts:* set-piece tags are fixed 35 s windows (restart IQR 7 s).
- *Broadcast ball:* audio gives no consistent offset; ball-track sync finds the half offsets, but
  only 6 of 93 live segments pass PLAN's acceptance, recovering 0.6% of missing frames (falsifier
  10%).
- *Whistles* (2–4.5 kHz tonal heuristic): 0.8% of `D` onsets have one within 1 s vs 0.6% by
  chance (gate 70%).
- *Gap timing and contacts as gap-model features:* +0.003, dropped. They would tie the model to
  where the sim places its gaps. A 5 s context: −0.002.
- *Learned per-frame `D`:* +0.034, but only by reading the sim's frozen dead-time ball (§5.5);
  without ball kinematics +0.018 (a tie), from players alone −0.030.
- *Restart must launch the ball* (≥ 4 m/s, v7): a tie on the sim, −0.004 when the dead ball is
  fetched and carried; consistency here 0.755.
- *Takeover variants:* ball direction +0.001, a 5-seed average −0.002, other capacities ≤ 0,
  labels 0.5 s / 1 s after the contact −0.011 / −0.038.
- *Video ball detection:* no-go. Only 18–22% of the missed held-out spells lie in ball gaps,
  under PLAN's 25% go condition (prior success 0.3).

**External data.** Bassek, Rein, Weber et al., *An integrated dataset of spatiotemporal and event
data in elite soccer*, Scientific Data 12, 195 (2025), CC BY 4.0: 25 Hz positions from seven
professional German league matches (DFL), with per-frame ball status and possession. Those
labels are human-produced data about *other* games, which the rules allow. We hand-labelled no
frames, of this or any other game.

## 3. Possession rules (`configs/final.yaml`)

**3.1 Ball cleaning.** The ATD does not clamp the ball to the pitch: it reaches 33 m beyond a
touchline. We drop positions more than 10 m outside the pitch and one-frame steps faster than
40 m/s, then fill gaps of ≤ 10 frames linearly. An observed ball just beyond a line is kept as
dead-ball evidence.

**3.2 Contacts and touches.** For each frame we take each team's nearest player to the ball. A
*contact* is a run of frames with the ball within 1.0 m of a player and the same nearest team
(gaps ≤ 2 frames bridged). Every contact is a candidate for the owner decoder (§3.4). A *touch*
is a contact the kinematics confirm: the ball stays close ≥ 12 frames, turns ≥ 20°, or changes
speed by ≥ 12 m/s. Touches are only used to find restarts.

**3.3 Dead ball.** Two kinds of evidence:
- the observed ball more than 0.5 m beyond a touch or goal line for ≥ 15 frames;
- the ball missing for ≥ 25 frames.

On this match, ball gaps are bimodal (16 under 26 frames, 133 of 101 or more), and players move
half as fast while the ball is missing (median 1.00 vs 1.93 m/s). So the tracker loses the ball
mostly in stoppages. Evidence runs within 50 frames merge. Each merged run is `D` from its first
frame up to the restart, the first touch after the evidence ends. Long gaps are then re-decoded
(§3.5).

**3.4 Owner decoder.** Inside each alive stretch, the state is the owner after each contact:
a contact by the owner's team keeps it at no cost; one by the other team takes the ball (score
log p) or not (log(1 − p)), with p from §4 sharpened by α = 3 on the log-odds scale. The Viterbi
path is the best owner sequence; each change lands on the contact's first frame, and a stretch
starts in whichever state scores best. A gap under 25 frames holds the owner.

**3.5 Long ball gaps.** Long gaps cover 40% of this match's graded frames. On the sim a fifth of
long-gap frames are alive, and the owner changes inside them with no contact. Per gap frame, two
models (§4) give P(dead) and P(`H` owns | alive), and a three-state Viterbi over `H`, `A`, `D` in
5-frame steps decodes each gap. A change into or out of `D` costs 30, an owner change 15,
including where the gap meets the labels around it. Before a kick-off, the ball stays dead.

**3.6 Timing.** Residual offsets of matched transitions on the sim, as median frames (predicted
minus true, fit / held-out): turnovers −2 / −2, `D` onsets +6 / +8, `D` ends +1 / +21.5. The
corrections we tested made things worse (§2), so none is applied.

## 4. Learned components

Three LightGBM models, single-threaded and seeded, all trained on the sim only. The first is the
**takeover probability** (`models/takeover_v2.txt`, 300 trees of 15 leaves, rate 0.03).
- *Training data:* the five fit sim matches only. A row is a contact by the team that does not
  own the ball in the sim GT just before it (2,237 contacts, 51% takeovers). Its label is whether
  the GT owner 5 frames after the contact is the toucher's team. Nothing is trained on this
  match.
- *Features:* 25, relative to the toucher's team (team-symmetric): the contact's duration and
  closest distances, the ball's speeds, speed change and turn (with missing flags), the rhythm
  of the previous and next contacts, and, since v6, ten team-shape signals: both teams' widths,
  the centroid and speed gaps, both teams' forward drift 1 s and 2 s after the contact, and
  both centroids' change across it. A team that wins the ball spreads and moves forward.
- *Tuning:* out of fold, each fit match scored by a model trained on the other four, end to end
  with the detected `D`: α = 3 and flip penalty 0, inside their grids. Retraining is
  byte-identical.
- *Held-out:* contact AUC 0.915 (0.913 without the shape signals).

The other two are the **gap models** (`models/gap_dead_v1.txt`, `models/gap_owner_v1.txt`: 300
trees of 31 leaves, rate 0.05).
- *Training data:* every 5th long-gap frame of the fit matches, from both teams' perspectives.
- *Features:* 44, from the players only, in each team's attacking direction: per team centroid,
  length, width, deepest and highest player, players in the opponent half, visible players, mean
  speed and forward drift, plus centroid, drift and speed 1 s and 2 s either side.
- *Tuning:* the two penalties, out of fold as above: 30 and 15, inside their grids.
- *Held-out:* P(dead) is right on 0.948 / 0.927 of gap frames (majority class 0.789 / 0.781);
  the owner on 0.813 / 0.872 of alive gap frames (last contact's team 0.72 / 0.69). Team widths
  and forward drift matter most.

Touch and dead-ball thresholds were fitted by coordinate descent on the fit matches only.

## 5. Validation protocol

**5.1 Rules we hold ourselves to.** No hand labels on this match, for any purpose. The public 5
minutes are a correctness check only: the tool prints *pass* (spellF1 ≥ 0.35, H/A not swapped) or
*fail*, never the number, and our scoring wrapper refuses to score a candidate on the public GT.
No rule or constant is tied to specific frames of this match.

**5.2 Simulation on open data (primary).** Each DFL match is rewritten in the bundle's ATD format.
Its ground truth is `D` where DFL marks the ball dead, else the team in possession. It is then
degraded to label-free statistics of the real ATD: stoppage ball loss (probability q, fraction
f), open-play gaps at the real onset rate by ball speed, crowding and pitch edge (scaled by s)
with the real gap lengths, and player tracklets, dropouts and duplicates at the real rates
(gap g). q, f, s and g were fitted on the fit matches:

| Statistic (graded frames) | sim | real ATD |
|---|---|---|
| ball missing | 41.5% | 40.7% |
| median ball gap | 342 frames | 268 frames |
| player speed, ball missing / seen | 0.59 | 0.52 |
| P(11 tracks \| 7–11 tracks) | 0.63 | 0.65 |

*Acceptance:* the official baselines score on the sim roughly as on the real match: smooth50
0.261 vs 0.243 (raw 0.103 vs 0.049; smooth125 0.192 vs 0.224).

**5.3 Selection.** Two sim matches (J03WQQ, J03WR9) are held out from all fitting. Intervals
are the official scorer's bootstrap (1,000 resamples):

| Held-out sim | J03WQQ [95% CI] | J03WR9 [95% CI] | mean | outcome |
|---|---|---|---|---|
| smooth50 baseline | 0.287 [0.259, 0.314] | 0.284 [0.251, 0.315] | 0.286 | |
| v1: hard rules | 0.522 [0.484, 0.560] | 0.560 [0.526, 0.594] | 0.541 | early upload |
| v2: v1 + extra `D` rules | 0.522 | 0.568 | 0.545 | tie, not selected |
| v3: contacts + takeover model + Viterbi | 0.559 [0.519, 0.600] | 0.609 [0.572, 0.645] | 0.584 | selected 31 Oct |
| v4: v3 + extra `D` rules | 0.563 | 0.623 | 0.593 | tie, consistency regression |
| v5: v3 + ball-free gap decoder | 0.585 [0.544, 0.623] | 0.633 [0.596, 0.671] | 0.609 | selected 1 Nov |
| **v6: v5 + team shape at contacts** | **0.610 [0.571, 0.649]** | **0.644 [0.606, 0.682]** | **0.627** | **selected 1 Nov** |

Decision rule (PLAN.md §6.4, amended 31 Oct before v3 was selected):
1. Discard any candidate that fails the validator, the bands or the public check.
2. A challenger replaces the current choice only if a **paired per-match comparison** favours
   it. Held-out matches always count; fit matches count where the challenger's learned parts
   are out of fold. It needs a positive mean difference and a one-sided sign test at p ≤ 0.05.
   Otherwise it is a tie, decided by fewer components, then consistency, then macroF1.

`make paired` runs this with every learned part cross-fitted. Each selection won all seven
matches (p ≈ 0.008): v3 over v1; v5 over the v3 final (v3 without stillness, 0.581) by +0.030;
v6 over v5 by +0.014 (+0.002, +0.004, +0.013, +0.004, +0.038 fit; +0.026, +0.011 held-out). The
earlier interval-overlap rule cannot resolve less than about 0.07 with two matches. v4's extra
`D` thresholds are in-sample, so only its held-out pair counts: a tie.

*Ablations of the final config* (held-out mean, change on removal): back to the hard rules
−0.086; long-gap evidence −0.055; the gap decoder −0.029; team shape at contacts −0.018;
out-of-bounds evidence −0.007; stillness 0 (dropped). α = 1 instead of the out-of-fold α = 3
would be +0.002 held-out; held-out matches never tune.
Out-of-bounds stays despite PLAN's 0.01 simplicity bar, because it is the one `D` evidence that
does not lean on the sim's built-in stoppage gaps. On this match, without it fewer `D` runs end
near set-piece restarts (25% vs 32%).

**5.4 Label-free checks on the real match.** *Structural bands* were set in PLAN.md before any
candidate existed. They filter out catastrophic errors and never rank.

| Band (graded frames) | allowed | v6 (final) | v3 | v1 |
|---|---|---|---|---|
| spells | 280–550 | 328 | 312 | 354 |
| spells < 2 s / < 6 s | 10–30% / 35–60% | 15.9% / 48.5% | 18.3% / 51.3% | 22.9% / 55.1% |
| mean spell | 7–16 s | 10.1 s | 10.0 s | 8.8 s |
| `D` runs | 60–150 | 135 | 133 | 133 |
| home share of alive frames | 0.55–0.72 | 0.61 | 0.62 | 0.61 |
| `X` on graded frames; kick-off team; `D` before the 2nd-half kick-off | 0; yes; yes | pass | pass | pass |

The short-spell shares sit close to the ≈ 20% / ≈ 50% inferred from the official lag figures.

- *Smart-Tagging consistency:* the share of alive frames inside team-tagged phases whose owner
  is the team the phase implies (attacking: the tag's team; defending: the other) is **0.742
  for v6** (v5 0.746, v3 and v1 0.740, v2 0.728, v4 0.726). It flags
  regressions and is not a target.
- *Gap frames:* v6 decodes 6.9% of long-gap frames alive (sim 17–20%), and there the tags agree
  with its owner on 0.814 of 12,759 frames. Team-tagged phases cover 51% of the gap frames it
  keeps `D`, like other `D` (59%), not alive frames (83%): real dead time, not missed play.
- *Set pieces:* 32% of the 57 set-piece tags have a predicted `D` ending within 2 s of the
  tag-aligned restart (chance ≈ 9%).
- *Stability:* 18 thresholds were moved ±10% one at a time, and each model was retrained with
  two other seeds. The worst variant keeps 0.867 spell agreement with the final output (median
  1.000; takeover seeds 0.965 / 0.967, gap-model seeds 0.999 / 0.996). Not fragile.
- *Disagreement:* v1 / v3 / v5 / v6 differ on 6.2% of graded frames (< 10%); humans watch those
  clips to find bugs, never to label.
- *Public window:* **pass**, team sign right.

**5.5 Known sim biases.**
1. DFL possession flips more often than the mechanical GT: 25–37% of sim spells are under 2 s,
   vs about 20% inferred here. v3 and v5 predict 0.81× and 0.85× the sim GT's spell count, so
   they do not chase that fragmentation.
2. Sim tracks are cleaner near the ball; the real ball is lost in stoppages even more; the sim
   has no Smart Tagging.
3. The sim's alive share inside long gaps comes from the fitted ball-loss model; this match's
   own appears lower (§5.4).
4. The DFL data freezes the ball in dead time: 21% of seen sim ball frames do not move, against
   0.08% here (`make airlock T=atd-stats`). No learned part that decides `D` reads ball
   kinematics.

## 6. Limitations and next steps

- **Changes inside ball gaps:** the gap decoder recovers part of them from the players alone.
  18–22% of v6's missed held-out spells still lie in gaps. With the true labels inside long gaps,
  the sim ceiling is +0.10 over v3, against the gap decoder's +0.03.
- **`D` is over-predicted:** 48.1% of graded frames here (v3 50.8%), and 42–45% vs 41% true on
  the held-out sim.
- **Restarts:** after tagged corners, the first spell is the corner team's 5 times in 11; fouls
  and offsides with the ball still tracked produce no evidence.
- **Semantics:** the labels are DFL's notion of possession; a systematic mismatch with this
  match's GT is a bias the sim cannot measure, checked only by the bands and consistency.
- **Phase 2:** the pipeline is match-agnostic (matches in `games.py`, thresholds in `configs/`).

## 7. Reproduction

```bash
make setup                          # Python 3.11.14 venv, pinned requirements.txt
make test                           # 96 tests, no match data needed
export MTAH_DATA_DIR=/path/to/bundle
make cache                          # parse the ATD into an npz cache outside the repo
make submission                    # -> submissions/final/<team>.csv, validator, bands, sha256
# optional, to retrain and re-check on the sim:
make open-data && make sim && make train-takeover CFG=configs/rules_v6.yaml && make train-gaps
make sim-ablate && make paired B=configs/rules_v6.yaml A=configs/rules_v5.yaml
```

- **Submitted file:** sha256
  `fe4b6878b3a22d6a57313f6574558e99867de680f0d6b53c4fe6fefab901c7cc`
  (`submissions/final/Berkeley Sports Research Group.sha256`; FINAL row of `submissions/LOG.md`).
- **Determinism:** CPU only, single-threaded LightGBM, `PYTHONHASHSEED=0`. A fresh clone with
  an empty cache (twice) and the Docker image (linux/arm64 and amd64, from a mounted bundle) all
  produce the committed sha256.
- **Runtime:** about 13 s for `make submission` with the parse cache warm (Apple M4); training
  on the sim takes about 3 min.
- **Tooling:** the code was written with an AI coding assistant. It never received match data,
  video, raw tracking lines or per-frame labels. Real-data runs released only aggregate
  statistics and pass/fail verdicts, filtered by a mechanical policy and logged
  (`src/mtah/airlock.py`). The gated bundle and every derived file stay out of the repository.

<!--
TEAM, before uploading:
- The repo URL is filled by `make doc REPO=https://github.com/Jus973/mtah-2027`.
- Measure the cold ATD parse time (`time make cache` on an empty cache) for §7 if you want it.
- §7 "Tooling": keep, edit or drop once the organisers answer email 1 (cloud assistants and
  aggregates).
-->
