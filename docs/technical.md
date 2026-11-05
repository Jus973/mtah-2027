<!--
Source of the technical document (PLAN.md §8). `make doc REPO=<url>` writes docs/<team>.md (and
.pdf when pandoc is installed), <team> from configs/base.yaml. HTML comments like this
one are notes for the team and do not render. Every number below comes from a make target;
real-match numbers are label-free aggregates or verdicts (make airlock T=...).
Version of 1 Nov 2026 (evening): final candidate v11, adopted with the one-way sign-off recorded
in submissions/LOG.md. Re-check numbers at M4 freeze.
-->

# Ball possession from automatic tracking: <TEAM_NAME>

More Than A Hack 2027, Phase 1: FC Barcelona (H) vs Athletic Club (A). The submitted file is
produced by `make submission` from `configs/final.yaml` (candidate **v11**). Code: <REPO_URL>.

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
6. A causal dead-ball model then re-places each `D` run's two boundaries within ± 3 s of the
   evidence by a per-boundary changepoint. It cannot create, delete or relocate a run (§3.6).

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
- *Gap timing and contacts as gap-model features:* +0.003, tied to where the sim places gaps.
- *Learned per-frame `D`:* +0.034, but only by reading the sim's frozen dead-time ball (§5.5).
  Trained on a realistic dead-time ball (noise, carried, lost; v8): +0.022 on all 7 matches, but
  on this match its `D` boundaries moved wholesale (median 2.0 s) and the tags disagreed with its
  new alive frames (0.564), so the free per-frame decode is not used. v11 (adopted 1 Nov) keeps
  that model out of decoding entirely: it only re-places each evidence boundary within ± 3 s,
  with causal features (§3.6) — the v8 failure mode is excluded by construction.
- *Restart must launch the ball* (≥ 4 m/s, v7): a tie on the sim, −0.004 with a carried ball.
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

**3.6 `D`-boundary refinement (v11, adopted 1 Nov).** A fourth model (§4) re-places each
evidence run's two boundaries. Within ± 75 frames (3 s) of a boundary — capped at the
neighbouring runs and the graded block — its per-frame P(dead) curve is split at the changepoint
that best separates dead from alive, and the evidence boundary is kept when the signal is flat.
By construction this cannot create, delete or relocate a run, only move its edges; the owner
decoder and the gap decoder are untouched.

**3.7 Timing.** Residual offsets of matched transitions on the sim, as median frames (predicted
minus true, fit / held-out): turnovers −2 / −2, `D` onsets +0 / +2 (v6: +6 / +8), `D` ends
+0 / +0 (v6: +1 / **+21.5** — the lateness §3.6 removes). The global corrections we tested made
things worse (§2), so none is applied.

## 4. Learned components

Four LightGBM models, single-threaded and seeded, all trained on the sim only. The first is the
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

The fourth is the **boundary dead model** (`models/dead_hybrid_v1.txt`: 300 trees of 31 leaves,
rate 0.05), used only by the §3.6 refinement.
- *Training data:* the five fit matches, each rebuilt three ways with a realistic dead-time ball
  (jittered at rest; picked up and carried; lost) instead of DFL's frozen one — 856,206 rows,
  every 5th graded frame; the label is the sim GT's dead/alive.
- *Features:* the gap models' player features plus **causal** ball features only (present and
  speed up to the current frame): the future-speed columns (+1 s, +2 s) are dropped, so a
  boundary cannot be placed by reading the ball after the restart — v8's likely failure
  mechanism.
- *Tuning:* the window (± 75 frames) was picked nested out of fold — chosen on four fit matches,
  scored on the fifth, all five folds agreeing — inside the registered grid {50, 62, 75}; 75 is
  the grid's edge, recorded as such and not widened post hoc. Retraining is byte-identical
  (`make train-hybrid`).

Touch and dead-ball thresholds were fitted by coordinate descent on the fit matches only.

## 5. Validation protocol

**5.1 Rules we hold ourselves to.** No hand labels on this match, for any purpose. The public 5
minutes are a correctness check only: the tool prints *pass* (spellF1 ≥ 0.35, H/A not swapped) or
*fail*, never the number, and our scoring wrapper refuses to score a candidate on the public GT.
No rule or constant is tied to specific frames of this match.

*Training and tuning data.* Every model, threshold and decoder cost in the final configuration
was fitted or selected on degraded external open data (DFL/IDSSE), which the organisers called
permissible on the challenge repository's issue #16. The one link to this match is the
degradation recipe (§5.2). Before the organisers' 3 Nov clarification, it was calibrated to
label-free aggregate statistics of the provided tracking file: ball-missing share, gap rates and
lengths, player track lengths. No labels, video or frame-level information went into it. To
remove that link on the ball side, we re-fitted it to external automatic tracking only (Metrica
Sports Sample Games 1–2, the vendor's own ball loss). The fitted parameters agree with the
original within one grid step (q 1.0, f 0.6, s 0.5 vs 1.0, 0.7, 0.4), and on sims built from it the
final still beats its predecessor on all 7 matches (+0.018, p = 0.008;
`configs/sim_external.yaml`). The player side (track dropouts, g) is still calibrated to the
provided file: no external feed we hold has comparable tracker dropout (the sample games have
none, broadcast tracking sees about half the players). The selection does not depend on it: with
g at a quarter or five times the fitted value, the final still wins on all 7 matches (+0.022 /
+0.019). Real-match
signals were otherwise used only as automatic guards and to choose between whole, externally
tuned versions. The organisers confirmed that is allowed (3 Nov).

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
| v6: v5 + team shape at contacts | 0.610 [0.571, 0.649] | 0.644 [0.606, 0.682] | 0.627 | selected 1 Nov |
| **v11: v6 + `D`-boundary refinement** | **0.644 [0.603, 0.685]** | **0.683 [0.645, 0.721]** | **0.664** | **selected 1 Nov (sign-off)** |

Decision rule (PLAN.md §6.4, amended 31 Oct before v3 was selected):
1. Discard any candidate that fails the validator, the bands or the public check.
2. A challenger replaces the current choice only if a **paired per-match comparison** favours
   it. Held-out matches always count; fit matches count where the challenger's learned parts
   are out of fold. It needs a positive mean difference and a one-sided sign test at p ≤ 0.05.
   Otherwise it is a tie, decided by fewer components, then consistency, then macroF1.

`make paired` runs this with every learned part cross-fitted. Each selection won all seven
matches (p ≈ 0.008): v3 over v1; v5 over the v3 final (v3 without stillness, 0.581) by +0.030;
v6 over v5 by +0.014 (+0.002, +0.004, +0.013, +0.004, +0.038 fit; +0.026, +0.011 held-out);
v11 over v6 by +0.029 (7/7, paired possession bootstrap P(Δ ≤ 0) < 0.001), robust under three
degradation stress arms (7/7 on each; +0.044 under the most adversarial). The
earlier interval-overlap rule cannot resolve less than about 0.07 with two matches. v4's extra
`D` thresholds are in-sample, so only its held-out pair counts: a tie. v11 also tripped two
label-free checks on the real match that older candidates pass; §5.4 reports both, the
mechanical evidence that they are anchored to an error of the older candidates, and the
pre-registered one-way rule under which it was adopted with a written sign-off.

*Ablations of the v6 config* (held-out mean, change on removal): back to the hard rules
−0.086; long-gap evidence −0.055; the gap decoder −0.029; team shape at contacts −0.018;
out-of-bounds evidence −0.007; stillness 0 (dropped). α = 1 instead of the out-of-fold α = 3
would be +0.002 held-out; held-out matches never tune.
Out-of-bounds stays despite PLAN's 0.01 simplicity bar, because it is the one `D` evidence that
does not lean on the sim's built-in stoppage gaps. On this match, without it fewer `D` runs end
near set-piece restarts (25% vs 32%). Removing v11's boundary refinement (back to v6) is −0.037
held-out.

**5.4 Label-free checks on the real match.** *Structural bands* were set in PLAN.md before any
candidate existed. They filter out catastrophic errors and never rank.

| Band (graded frames) | allowed | v6 | v3 | v1 |
|---|---|---|---|---|
| spells | 280–550 | 328 | 312 | 354 |
| spells < 2 s / < 6 s | 10–30% / 35–60% | 15.9% / 48.5% | 18.3% / 51.3% | 22.9% / 55.1% |
| mean spell | 7–16 s | 10.1 s | 10.0 s | 8.8 s |
| `D` runs | 60–150 | 135 | 133 | 133 |
| home share of alive frames | 0.55–0.72 | 0.61 | 0.62 | 0.61 |
| `X` on graded frames; kick-off team; `D` before the 2nd-half kick-off | 0; yes; yes | pass | pass | pass |

The short-spell shares sit close to the ≈ 20% / ≈ 50% inferred from the official lag figures.
**v11 (the final) passes every band** (candidate and submission runs, 1 Nov) with the same spell
structure as v6 — 325 spells, 8 under 1 s — because it only moves `D` edges; its `D` share of
graded frames is 0.445, inside the advisory `D`-share band [0.35, 0.57] added 1 Nov (v6: 0.481).

- *Smart-Tagging consistency:* the share of alive frames inside team-tagged phases whose owner
  is the team the phase implies (attacking: the tag's team; defending: the other) is 0.742
  for v6 (v5 0.746, v3 and v1 0.740, v2 0.728, v4 0.726) and 0.730 for v11. The pooled rate
  tracks alive volume (≈ 0.005 per share point), so since 1 Nov the judging statistic is the
  **exclusive-alive decomposition**: on the 18,930 frames v11 newly decodes alive, the tags agree
  with its owner 0.600 (the rejected v8: 0.564), 0.686 inside ball gaps (v8: 0.525), while the
  frames both candidates label alive stay at 0.741 — mid-play is untouched. It flags regressions
  and is not a target.
- *Gap frames:* v6 decodes 6.9% of long-gap frames alive (sim 17–20%), and there the tags agree
  with its owner on 0.814 of 12,759 frames. Team-tagged phases cover 51% of the gap frames it
  keeps `D`, like other `D` (59%), not alive frames (83%): real dead time, not missed play.
- *Set pieces:* 32% of the 57 set-piece tags have a v6 `D` ending within 2 s of the tag-aligned
  restart (chance ≈ 9%). **v11 scores 14% on this check and its `D` ends move median 1.72 s
  earlier than v6's** — the one real-match regression, and the reason adoption needed a sign-off:
  the anchor was fitted to v1's own `D`-end timing, and the mechanical cross-check (§5.6) finds
  v6's ends are themselves ~1.5 s late at the median, so ends that move onto the true restarts
  leave a ±2 s window centred on the old error. The pre-registered one-way rule (old guard fails,
  every repaired guard passes, mechanical explanation documented) was exercised with a written
  sign-off, recorded in `submissions/LOG.md`.
- *Stability:* 18 thresholds (19 for v11, including the refinement window) were moved ±10% one at
  a time, and each model was retrained with two other seeds — since 1 Nov including the learned
  dead models. v6's worst variant keeps 0.867 spell agreement with its output; v11's keeps 0.917
  (boundary-model seed retrains 0.999–1.0; window ±10%: 0.997–0.999). Not fragile.
- *Disagreement:* the active candidates (v1 / v3 / v5 / v6 / v9d / v11) differ on 11.3% of graded
  frames — above the 10% tripwire, driven by v11 vs v1 (10.2%), which spans every learned
  component added since the hard rules; humans watch those clips to find bugs, never to label.
- *Public window:* **pass** for v6 and for v11, team sign right.

**5.5 Known sim biases.**
1. DFL possession flips more often than the mechanical GT: 25–37% of sim spells are under 2 s,
   vs about 20% inferred here. v3 and v5 predict 0.81× and 0.85× the sim GT's spell count, so
   they do not chase that fragmentation.
2. Sim tracks are cleaner near the ball; the real ball is lost in stoppages even more; the sim
   has no Smart Tagging.
3. The sim's alive share inside long gaps comes from the fitted ball-loss model; this match's
   own appears lower (§5.4).
4. The DFL data freezes the ball in dead time: 21% of seen sim ball frames do not move, against
   0.08% here (`make airlock T=atd-stats`). The gap models that decide `D` inside gaps read no
   ball kinematics at all; the v11 boundary model reads causal ball features, so it is trained
   only on fit matches rebuilt with a realistic dead-time ball (§4), never on the frozen one.
5. No validation build contains a dead stretch with the ball visible: the Metrica sample
   feeds blank the ball at dead time just as the degrader does (0 of 286 Metrica `D` runs have
   the ball seen through the run), while this match's ATD sees the ball through whole dead
   stretches. The visible-ball stoppage case therefore rests on the boundary model's synthetic
   dead-time ball conditions (§4) and on the airlock `D` aggregates, not on a scored bench.

**5.6 Mechanical cross-check (Metrica Game 3).** A second validation family, independent of DFL
and of the sim recipe: Metrica Sports Sample Game 3 (pinned commit, checksummed), whose event
feed has a frame-accurate clock, degraded like the sim. Against that mechanical ground truth,
v6's `D` ends are **+26 / +38 / +78 frames late at p25 / p50 / p75** (clock calibration ±2
frames) — about 1.5 s at the median, matching the sim's +21.5-frame held-out residual (§3.7) and
the restart pattern this match's own label-free profile shows (`make airlock T=dend-stats`: the
ball is missing on ~50% of frames within ±1 s of restarts, so evidence ends trail the restart
touch). This is the documented mechanical basis for the §5.4 one-way adoption: v11's median
`D`-end move (−1.72 s) is the size of the lateness the mechanical GT measures for v6.

**5.7 Robustness to ground-truth conventions and feed shifts (3–4 Nov, open data, all
pre-registered).** Three families of stress test, none of which changed the final configuration:

- *Ground-truth conventions.* The DFL possession flag is one convention among several; a target
  labeller that places `D` starts, `D` ends or owner switches elsewhere, or that absorbs short
  spells, is a bias the sim cannot see directly. So the outputs on the 8 primary matches (the 7
  DFL sims, cross-fitted, plus Metrica Game 3) were re-scored against transformed ground truths,
  one axis at a time. With direct H↔A boundaries moved −10 to +25 frames, `D` starts moved −10 to
  +25 frames, or sandwiched spells shorter than 1 s or 2 s folded into their neighbours, the final
  beats the previous configuration on all 8 matches in each of the 10 worlds (margin +0.023 to
  +0.035; it grows when short spells are merged and shrinks as `D` starts move later), and stays
  ahead on Metrica Games 1–2. Moving `D` ends later is the one axis that erodes the lead: +0.029
  at 0 frames, +0.021 at +10, +0.012 at +20 and +0.001 at +38 (about v6's measured lateness,
  §5.6), where the final wins only 5 of 8 and loses on the Metrica builds (Game 3 −0.017, Games
  1–2 −0.030 / −0.008). That is still above the pre-registered −0.010 bar, so the final is kept,
  but a late, whistle-timed `D`-end convention is its main residual exposure. Among the shifts
  tested (`D` ends only later), the final's own absolute score is highest at the untransformed
  ground truth on all three boundary axes, so no post-hoc boundary shift is applied.
- *Degradation redraw floor.* Re-degrading the two held-out matches with three more random
  draws moves a configuration's absolute spellF1 by ±0.020 per match (SD) while the paired
  difference between two configurations moves by only ±0.004; a paired held-out gain below
  +0.008 is therefore treated as a tie, and training on several draws of the same matches
  (augmentation) was itself a tie under that rule.
- *Feed shifts, paired.* The committed degraded sims were resampled or edited so that the
  random gap draw stays identical and only the tested property changes, then scored with
  cross-fitted models (per-cell noise ≈ 0.006). A ±10% change of the match clock (every speed
  and frame-counted duration) costs ≤ 0.007; hiding one visible track per team inside every
  long ball gap costs 0.001; hiding 10% of all slot tracklets costs 0.044, and that loss is
  contact loss when the player at the ball is untracked. Two label-free comparisons with this
  match's feed (`make airlock T=atd-stats`) show where the sim departs from it. First, nobody
  is near the ball more often here: on visible in-pitch ball frames at carry speed (0.5–4 m/s)
  no player is within 1 m on 52.6% of frames, against 30.5–35.4% on the sims, and every
  distance bin beyond 1.5 m holds about twice the sims' share, so the carrier-loss sensitivity
  applies to this match and is not only a stress case. Second, the real ball track is
  velocity-filtered: 2.6 kicks (speed change > 3 m/s) per 1,000 seen frames, against 21–43 on
  the sims. On sims whose ball is smoothed to match (13 frames), the final loses 0.17 (0.09–0.23
  for 7–25 frames, 7/7 matches), and every fix tried (retraining on smoothed-ball rows, relaxed
  touch thresholds) recovers ≤ 0.003: the lost contacts are short dwell contacts that the
  smoothed feed no longer shows. This hits any tracking-only method alike and does not change
  the choice of configuration, but the absolute sim scores in this document overstate what to
  expect on this match.

## 6. Limitations and next steps

- **Changes inside ball gaps:** the gap decoder recovers part of them from the players alone
  (the 18–22% missed-in-gaps share was measured on v5; v6 changes the takeover model, not the
  gap handling). With the true labels inside long gaps, the sim ceiling is +0.10 over v3,
  against the gap decoder's +0.03. Caveat (1 Nov): the committed sim samples open-play gap
  lengths from a pooled histogram, which plants stoppage-length gaps into live play, so part of
  the sim's alive-inside-gaps mass is an artefact (`configs/sim_repaired.yaml` is the
  onset-speed-conditioned lens).
- **`D` is over-predicted:** 44.5% of graded frames for v11 (v6 48.1%, v3 50.8%), and for v6
  42–45% vs 41% true on the held-out sim.
- **The adoption risk we accepted (v11):** the two v1-era real-match checks it fails (§5.4) are
  anchored to the older candidates' own `D`-end timing, and the mechanical evidence (§5.6) says
  that timing was late — but the mechanical GT is a *different match*. If this match's tag
  timing is right where the Metrica transfer is wrong, v11 over-corrects restarts by ~1 s. The
  spell structure is v6's either way (325 vs 328 spells); the exposure is boundary placement
  only, which is also where the held-out gain (+0.037) comes from.
- **Restarts:** after tagged corners, the first spell is the corner team's 5 times in 11; fouls
  and offsides with the ball still tracked produce no evidence.
- **Semantics:** the labels are DFL's notion of possession; a systematic mismatch with this
  match's GT is bounded by the convention worlds of §5.7 (the final leads in every one, by
  only +0.001 when `D` ends are placed ~1.5 s late) but not removed, and is otherwise checked
  only by the bands and consistency.
- **Ball kinematics:** the sim matches this feed's gaps, visibility and tempo but not its ball
  kinematics; the real ball track is about ten times smoother, which costs the final ~0.17 on
  matched sims (§5.7). Expect the absolute score here to sit below the held-out sim estimate.
- **Phase 2:** the pipeline is match-agnostic (matches in `games.py`, thresholds in `configs/`).

## 7. Reproduction

```bash
make setup                          # Python 3.11.14 venv, pinned requirements.txt
make test                           # 116 tests, no match data needed
export MTAH_DATA_DIR=/path/to/bundle
make cache                          # parse the ATD into an npz cache outside the repo
make submission                    # -> submissions/final/<team>.csv, validator, bands, sha256
# optional, to retrain and re-check on the sim:
make open-data && make sim && make train-takeover CFG=configs/rules_v6.yaml && make train-gaps
make train-hybrid                   # the v11 boundary dead model (byte-identical retrain)
make sim-ablate && make paired B=configs/rules_v11.yaml A=configs/rules_v6_final.yaml
```

- **Submitted file:** sha256
  `7b83f3a89bb15fa880f75e568bf289cdfb9d2868b24bbb2aa6130b13e2164b83`
  (`submissions/final/Berkeley Sports Research Group.sha256`; FINAL row of `submissions/LOG.md`).
- **Determinism:** CPU only, single-threaded LightGBM, `PYTHONHASHSEED=0`. A fresh parse cache
  (twice) and the Docker image (linux/arm64 and amd64, from a mounted bundle) reproduce the
  committed sha256.
- **Runtime:** about 30 s for `make submission` with the parse cache warm (Apple M4); training
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
