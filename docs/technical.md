<!--
Source of the technical document (PLAN.md §8). `make doc REPO=<url>` writes docs/<team>.md (and
.pdf when pandoc is installed), <team> from configs/base.yaml. HTML comments like this
one are notes for the team and do not render. Every number below comes from a make target;
real-match numbers are label-free aggregates or verdicts (make airlock T=...).
Version of 31 Oct 2026: final candidate v3 (configs/final.yaml). Re-check numbers at M4 freeze.
-->

# Ball possession from automatic tracking: <TEAM_NAME>

More Than A Hack 2027, Phase 1: FC Barcelona (H) vs Athletic Club (A). The submitted file is
produced by `make submission` from `configs/final.yaml` (candidate **v3**). Code: <REPO_URL>.

## 1. Approach

Possession changes only at **contacts** between the ball and a player, and stoppages come from
**dead-ball evidence** in the tracking. We don't smooth a per-frame guess. The pipeline has four
steps:
1. Every contact is a candidate change.
2. A gradient-boosted model, trained only on *other* matches (open DFL data degraded to look
   like this match's tracking), gives each contact by the team without the ball a probability of
   taking it.
3. A two-state Viterbi over the contacts picks the most likely owner sequence.
4. A `D` stretch starts where the ball is beyond a line or lost by the tracker while players are
   calm, and ends at the restart.

Every change lands on the first frame of the contact or evidence that causes it, which is how
the ground truth is defined, so latency is zero by construction. On this match we use only
label-free checks and a pass/fail bug check on the public window.

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

A majority-vote window can remove flicker only by also merging short possessions, which is why
the official smoothing baseline levels off near 0.24 on the full match.

## 2. Assets

| Asset | Used | How |
|---|---|---|
| Tactical ATD | yes | Every component. Our parser regenerates the four official baselines byte-for-byte. |
| Smart Tagging | yes | The `Kick Off (Start)` windows bound each half's kick-off search. Label-free consistency and set-piece checks (§5.4). It sets no label. |
| `games.py` | yes | Graded half ranges (`X` only outside them), pitch size. |
| Tactical homography | tools only | Projects tracking onto the video in the human clip viewer. |
| Broadcast ATD, both videos' audio | tried, dropped | Measured reasons below. |
| DFL/IDSSE open data | yes | Simulation, training, tuning and selection (§4, §5). |

**Kick-offs.** Each half's kick-off is the frame where both teams' line-up (≥ 80% of each team in
its own half for ≥ 1 s) breaks, with the ball near the centre, inside the tag window. It lands 4
frames after the published frame in both halves, within our 1 s tolerance. We did not correct
it, since that would be a constant fitted to this match. Before the derived second-half
kick-off, the graded block is `D`.

**Tried and dropped, with the measured reason** (held-out sim spellF1 unless stated; §5):
- *Smoothing baseline:* 0.286.
- *Hard owner rules* (v1: any opponent touch flips): 0.541. *H2* (controlled touch) 0.522;
  *H3* (two touches) 0.504 with the true `D`.
- *Logistic regression* for takeovers: about 0.63 vs 0.65 for LightGBM in the prototype,
  with the true `D`.
- *Pressure features* (players within 3 / 8 m): held-out AUC unchanged.
- *Stillness as `D` evidence:* inert on this match; the output is byte-identical without it.
- *Four extra `D` rules* (calm-gap filter, restart at the taker's release or at the ball's
  reappearance in flight, kicks from rest as touches): +0.004 with the rules decoder and +0.009
  with ours. Both are ties under the paired rule (§5.3), and Smart-Tagging consistency fell to
  0.728 / 0.726.
- *Latency corrections:* moving `D` onsets earlier costs −0.003; moving takeovers into a
  preceding ball gap tunes to a no-op.
- *Tag-anchored restarts:* set-piece tags are fixed 35 s windows whose restart varies by an IQR of
  7 s, far coarser than the tracking.
- *Broadcast ball.* Audio cross-correlation finds no consistent offset. Syncing on the two
  feeds' ball tracks finds both half offsets (median distance 0.8 m), but only 6 of 93 live
  segments meet PLAN's acceptance (balls within 2 m on ≥ 90% of frames). The broadcast ball
  recovers 0.6% of tactical-missing graded frames, against the 10% falsifier.
- *Whistles* (2–4.5 kHz tonal heuristic): 0.8% of `D` onsets have one within 1 s vs 0.6% by
  chance, and looser settings never pass 40% vs 24%. The gate was 70%.
- *Video ball detection:* no-go by default (prior success 0.3; weights trained on bundle frames
  cannot be committed).
- *Kick-off as "ball at rest on the spot":* the tracked ball never rests there.

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
frame up to the restart, the first touch after the evidence ends.

**3.4 Owner decoder.** Inside each alive stretch, the state is the owner after each contact:
- a contact by the owner's team keeps it at no cost;
- a contact by the other team either takes the ball (score log p) or does not (log(1 − p)), with
  p from §4 sharpened by α = 3 on the log-odds scale;
- the Viterbi path is the best owner sequence, and each change lands on the contact's first
  frame;
- the stretch starts in whichever state scores best, which after a stoppage is normally the
  first contact's team;
- a ball the tracker has lost holds the owner (no contact, no change).

**3.5 Timing.** Residual offsets of matched transitions on the sim, as median frames (predicted
minus true, fit / held-out): turnovers −2 / −2, `D` onsets +6 / +8, `D` ends +1 / +21.5. The
corrections we tested made things worse (§2), so none is applied.

## 4. Learned components

One model, the **takeover probability** (`models/takeover_v1.txt`, LightGBM, 300 trees of 15
leaves, learning rate 0.03, single-threaded, seeded).
- *Training data:* the five fit sim matches only. A row is a contact by the team that does not
  own the ball in the sim GT just before it (2,237 contacts, 51% takeovers). Its label is whether
  the GT owner 5 frames after the contact is the toucher's team. Nothing is trained on this
  match.
- *Features:* 15, all relative to the toucher's team, so the model is team-symmetric:
  - the contact: log duration, closest distance, the other team's closest distance;
  - the ball: speeds before and after, speed change, turn, with flags where they are missing;
  - the rhythm: log time to the previous and next contacts, and whether those and the
    second-next are the same team's.
- *Tuning:* α and a flip penalty were tuned on out-of-fold predictions. Each fit match was
  scored by a model trained on the other four, end to end with the detected `D`, giving α = 3
  and penalty 0, both inside their grids. Retraining is byte-identical, and models retrained with
  other seeds change the real-match output little (§5.4).
- *Held-out:* contact AUC 0.913.

The touch and dead-ball thresholds were fitted by coordinate descent over fixed grids, also on
the fit matches only.

## 5. Validation protocol

**5.1 Rules we hold ourselves to.** No hand labels on this match, for any purpose. The public 5
minutes are a correctness check only: the tool prints *pass* (spellF1 ≥ 0.35, H/A not swapped) or
*fail*, never the number, and our scoring wrapper refuses to score a candidate on the public GT.
No rule or constant is tied to specific frames of this match.

**5.2 Simulation on open data (primary).** Each DFL match is rewritten in the bundle's ATD format.
Its ground truth is `D` where DFL marks the ball dead, else the team in possession. It is then
degraded to statistics measured on the real ATD without labels:
- stoppage ball loss (probability q, fraction f);
- open-play gaps at the real onset rate by ball speed × crowding × near-a-line (scaled by s),
  with the real gap-length histogram;
- player tracklets, dropouts and duplicates at the real rates (inter-tracklet gap g).

q, f, s and g were fitted on the fit matches:

| Statistic (graded frames) | sim | real ATD |
|---|---|---|
| ball missing | 41.5% | 40.7% |
| median ball gap | 342 frames | 268 frames |
| player speed, ball missing / seen | 0.59 | 0.52 |
| P(11 tracks \| 7–11 tracks) | 0.63 | 0.65 |

*Acceptance:* the official baseline logic must score on the sim roughly as it does on the real
match. It does: smooth50 0.261 on the sim vs 0.243 real (raw 0.103 vs 0.049; smooth125 0.192 vs
0.224).

**5.3 Selection.** Two sim matches (J03WQQ, J03WR9) are held out from all fitting. Intervals
are the official scorer's bootstrap (1,000 resamples):

| Held-out sim | J03WQQ [95% CI] | J03WR9 [95% CI] | mean | outcome |
|---|---|---|---|---|
| smooth50 baseline | 0.287 [0.259, 0.314] | 0.284 [0.251, 0.315] | 0.286 | |
| v1: hard rules | 0.522 [0.484, 0.560] | 0.560 [0.526, 0.594] | 0.541 | early upload |
| v2: v1 + extra `D` rules | 0.522 | 0.568 | 0.545 | tie, not selected |
| **v3: contacts + takeover model + Viterbi** | **0.559 [0.519, 0.600]** | **0.609 [0.572, 0.645]** | **0.584** | **selected** |
| v4: v3 + extra `D` rules | 0.563 | 0.623 | 0.593 | tie, consistency regression |

Decision rule (PLAN.md §6.4, amended 31 Oct before v3 was selected):
1. Discard any candidate that fails the validator, the bands or the public check.
2. A challenger replaces the current choice only if a **paired per-match comparison** favours
   it. Held-out matches always count; fit matches count where the challenger's learned parts
   are out of fold. It needs a positive mean difference and a one-sided sign test at p ≤ 0.05.
   Otherwise it is a tie, decided by fewer components, then consistency, then macroF1.

v3 beats v1 on all seven matches: +0.017, +0.016, +0.051, +0.025 and +0.047 on the fit matches
(v3 out of fold, v1 in-sample), and +0.037 and +0.049 held-out, so p ≈ 0.008. The earlier rule,
"overlapping intervals means a tie", compared unpaired intervals that cannot resolve less than
about 0.07 with two matches. v4's extra `D` thresholds are in-sample on the fit matches, so only
the held-out pair counts for it: a tie.

*Ablations of the final config* (held-out mean, change on removal): back to the hard rules
−0.043; long-gap evidence −0.110; out-of-bounds evidence −0.005; stillness −0.003 (dropped).
Out-of-bounds stays despite PLAN's 0.01 simplicity bar, because it is the one `D` evidence that
does not lean on the sim's built-in stoppage gaps. On this match, without it fewer `D` runs end
near set-piece restarts (25% vs 32%).

**5.4 Label-free checks on the real match.** *Structural bands* were set in PLAN.md before any
candidate existed. They filter out catastrophic errors and never rank.

| Band (graded frames) | allowed | v3 (final) | v1 |
|---|---|---|---|
| spells | 280–550 | 312 | 354 |
| spells < 2 s / < 6 s | 10–30% / 35–60% | 18.3% / 51.3% | 22.9% / 55.1% |
| mean spell | 7–16 s | 10.0 s | 8.8 s |
| `D` runs | 60–150 | 133 | 133 |
| home share of alive frames | 0.55–0.72 | 0.62 | 0.61 |
| `X` on graded frames; kick-off team; `D` before the 2nd-half kick-off | 0; yes; yes | pass | pass |

v3's short-spell shares sit close to the ≈ 20% / ≈ 50% inferred from the official lag figures.

- *Smart-Tagging consistency:* team-tagged phases imply who has the ball (the tagged team for
  attacking phases, its opponent for defending ones). The share of tagged alive frames that
  agree is **0.740 for v3**, the same as v1 (v2 0.728, v4 0.726). It flags regressions; it is not
  a target. The tagger emits each phase once per team over the same window, and our agreement is
  identical within every pair, which confirms the attack/defend mapping.
- *Set pieces:* 32% of the 57 set-piece tags have a predicted `D` ending within 2 s of the
  tag-aligned restart (chance ≈ 9%).
- *Stability:* 14 thresholds were moved ±10% one at a time, and the model was retrained with two
  other seeds. The worst variant keeps 0.854 spell agreement with the final output (median
  0.999; seeds 0.966 / 0.968). Not fragile.
- *Disagreement:* v1 / v2 / v3 differ on 8.1% of graded frames (PLAN expects < 10%). Humans
  inspect those clips in the clip viewer to find bugs, never to label.
- *Public window:* **pass**, team sign right.

**5.5 Known sim biases.**
1. DFL possession flips more often than the mechanical GT: 25–37% of sim spells are under 2 s,
   vs about 20% inferred here. v3 predicts 0.8× the sim GT's spell count, so it does not chase
   that fragmentation.
2. Sim tracks are cleaner near the ball.
3. The real ball is lost in stoppages even more than the sim reproduces.
4. The sim has no Smart Tagging.

Touch statistics transfer: the ball is within 1.0 m of a player on 28.7% of real ball-seen
frames vs 28.0–29.7% on the sim.

## 6. Limitations and next steps

- **Missed changes without a touch:** two-thirds of the possession changes the rules missed
  on the sim had no touch by the new owner within 1 s, mostly inside ball gaps. Ball-gap spells account for a
  quarter of v3's missed spells.
- **`D` is over-predicted:** 50.8% of graded frames here, and 51–52% vs 41% true on the sim. But
  perfect `D` is worth only +0.04 on the sim, mostly from boundary timing.
- **Restart owner:** after tagged corners, the first predicted spell is the corner team's only 5
  times in 11.
- **Stoppages with the ball still tracked and moving** (fouls, offside) produce no evidence, and
  whistles are not detectable in the tactical audio.
- **Semantics:** the takeover labels are DFL's notion of possession. A systematic mismatch with
  this match's GT is a bias the sim cannot measure. The bands and consistency are our only check
  on it.
- **Phase 2:** the pipeline is match-agnostic. Matches are entries in `games.py`, thresholds
  live in `configs/`, and the sim builder works for any open dataset with ball status and
  possession.

## 7. Reproduction

```bash
make setup                          # Python 3.11.14 venv, pinned requirements.txt
make test                           # 86 tests, no match data needed
export MTAH_DATA_DIR=/path/to/bundle
make cache                          # parse the ATD into an npz cache outside the repo
make submission                    # -> submissions/final/<team>.csv, validator, bands, sha256
# optional, to retrain and re-check on the sim:
make open-data && make sim && make train-takeover && make sim-ablate
```

- **Submitted file:** sha256
  `00f4551f83d84c891e0c13c5d62fa0e8d0918a749c16e97722edaa32f1bc499a`
  (`submissions/final/Berkeley Sports Research Group.sha256`; the FINAL row of `submissions/LOG.md`).
- **Determinism:** inference has no randomness. It runs on the CPU, with single-threaded
  LightGBM and `PYTHONHASHSEED=0`. Each of these produced the same file, equal to the committed
  sha256:
  - a fresh clone with an empty cache, run twice;
  - the Docker image, on linux/arm64 and on linux/amd64.

  The Dockerfile reproduces the file from a mounted bundle.
- **Runtime:** about 4 s for `make submission` with the parse cache warm, on an Apple M4 (10
  cores, 16 GB). Model training takes 33 s on the sim.
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
