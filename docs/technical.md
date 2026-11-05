<!--
Source of the technical document (PLAN.md §8). `make doc REPO=<url>` writes docs/<team>.md (and
.pdf when pandoc is installed), <team> from configs/base.yaml. HTML comments like this
one are notes for the team and do not render. Every number below comes from a make target;
real-match numbers are label-free aggregates or verdicts (make airlock T=...).
Version of 5 Nov 2026: final candidate v12-hedge, adopted after the independent verification
recorded in submissions/LOG.md (v11 before it, adopted 1 Nov with a one-way sign-off). Re-check
numbers at M4 freeze; §7 "Determinism" is updated after the cold-cache and Docker drill.
-->

# Ball possession from automatic tracking: <TEAM_NAME>

More Than A Hack 2027, Phase 1: FC Barcelona (H) vs Athletic Club (A). The submitted file is
produced by `make submission` from `configs/final.yaml` (candidate **v12-hedge**). Code: <REPO_URL>.

## 1. Approach

Possession changes only at **contacts** between the ball and a player, and stoppages come from
**dead-ball evidence** in the tracking. We don't smooth a per-frame guess. The pipeline has seven
steps:
1. Every contact is a candidate change.
2. A gradient-boosted model, trained only on *other* matches (open DFL data degraded to look
   like this match's tracking), gives each contact by the team without the ball a probability of
   taking it, from the contact's kinematics and both teams' shape around it.
3. A two-state Viterbi over the contacts picks the most likely owner sequence.
4. A `D` stretch starts where the ball is beyond a line or lost by the tracker, and ends at the
   restart.
5. Inside long ball gaps, where there is no contact to see, two more models read the players
   alone, and a three-state Viterbi re-decides `H`, `A` or `D`.
6. A causal dead-ball model then re-places each `D` run's two boundaries within ± 3 s of the
   evidence by a per-boundary changepoint. It cannot create, delete or relocate a run (§3.6).
7. This feed's ball track is velocity-smoothed (§5.7), hiding kicks the contact model reads, so
   a per-frame ball-proximity owner model (§4) is blended into the owner path inside every alive
   stretch, weighted by the match's own kick rate (§3.8). `D` never changes.

Every change lands on the first frame of the contact or evidence that causes it, as the ground
truth is defined; the §3.8 blend can move an owner switch to where the proximity evidence turns.
On this match we use only label-free checks and a pass/fail public bug check.

Why events: spellF1 matches predicted to true spells one-to-one, same team only, at IoU 0.3–0.7.
Perturbing a perfect file with the official scorer shows what each kind of error costs
(public-window GT, `make sim-costs`):

| Perturbation of a perfect file | spellF1 |
|---|---|
| one spurious 0.5 s opponent flicker inside each spell > 10 s | 0.55 |
| one spurious 1 s `D` inside each spell > 10 s | 0.64 |
| all boundaries 25 frames (1 s) late | 0.78 |

A majority-vote window removes flicker only by merging short possessions too (the official
smoothing baseline levels off near 0.24).

## 2. Assets

| Asset | Used | How |
|---|---|---|
| Tactical ATD | yes | Every component. Our parser regenerates the four official baselines byte-for-byte. |
| Smart Tagging | yes | The `Kick Off (Start)` windows bound each half's kick-off search. Label-free consistency and set-piece checks (§5.4). It sets no label. |
| `games.py` | yes | Graded half ranges (`X` only outside them), pitch size. |
| Broadcast ATD, both videos' audio | tried, dropped | Measured reasons below. |
| DFL/IDSSE open data | yes | Simulation, training, tuning and selection (§4, §5). |

**Kick-offs.** Each half's kick-off is where both teams' line-up (≥ 80% in their own half for
≥ 1 s) breaks inside the tag window: 4 frames after the published frame, within tolerance.

**Tried and dropped** (held-out sim spellF1 unless stated; §5): other owner rules (controlled
touch 0.522, two touches 0.504 with the true `D`); logistic takeovers (0.63 vs 0.65); pressure
features and stillness as `D` evidence (inert); four extra `D` rules (+0.004 / +0.009, ties under
§5.3, consistency down); latency corrections (−0.003 / no-op); broadcast ball (6 of 93 live
segments pass sync acceptance, 0.6% of missing frames recovered vs a 10% falsifier); whistles
(0.8% of `D` onsets vs 0.6% by chance); video ball detection (18–22% of missed spells lie in ball
gaps, under the 25% go bar). A *learned per-frame `D`* gained +0.022 on a realistic dead-time ball (v8) but moved this match's `D` boundaries wholesale (median 2.0 s)
against the tags (0.564), so it is used only inside the bounded refinement of §3.6.

**External data.** Bassek, Rein, Weber et al., *An integrated dataset of spatiotemporal and event
data in elite soccer*, Scientific Data 12, 195 (2025), CC BY 4.0: 25 Hz positions from seven
German league matches (DFL) with per-frame ball status and possession — human-produced labels
about *other* games, which the rules allow. We hand-labelled no frames of any game.

## 3. Possession rules (`configs/final.yaml`)

**3.1 Ball cleaning.** The ATD does not clamp the ball to the pitch (it reaches 33 m beyond a
touchline). We drop positions more than 10 m outside it and one-frame steps faster than 40 m/s,
then fill gaps of ≤ 10 frames linearly; a ball just beyond a line is kept as dead-ball evidence.

**3.2 Contacts and touches.** For each frame we take each team's nearest player to the ball. A
*contact* is a run of frames with the ball within 1.0 m of a player and the same nearest team
(gaps ≤ 2 frames bridged). Every contact is a candidate for the owner decoder (§3.4). A *touch*
is a contact the kinematics confirm (the ball stays close ≥ 12 frames, turns ≥ 20°, or changes
speed by ≥ 12 m/s); touches only find restarts.

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
5-frame steps decodes each gap: a change into or out of `D` costs 30, an owner change 15,
including at the gap's edges. Before a kick-off, the ball stays dead.

**3.6 `D`-boundary refinement (v11, adopted 1 Nov).** A fourth model (§4) re-places each
evidence run's two boundaries. Within ± 75 frames (3 s) of a boundary — capped at the
neighbouring runs and the graded block — its per-frame P(dead) curve is split at the changepoint
that best separates dead from alive, and the evidence boundary is kept when the signal is flat.
By construction this cannot create, delete or relocate a run, only move its edges.

**3.7 Timing.** Median residual offsets of matched transitions on the sim (predicted minus true,
fit / held-out, v11's components): turnovers −2 / −2 frames, `D` onsets +0 / +2 (v6: +6 / +8),
`D` ends +0 / +0 (v6: +1 / **+21.5**, the lateness §3.6 removes). No global correction is
applied (§2).

**3.8 Proximity owner blend (v12-hedge, adopted 5 Nov).** Per alive frame, the emission is the
proximity model's log-odds (§4) plus w × (+1 where §3.4–3.5 say `H`, −1 for `A`); a two-state
Viterbi with switch cost 5 re-decides `H`/`A` inside every alive stretch. `D` and `X` frames never
change. w depends on the match's kick rate, counted on the provided tracking without labels:
w = 2 when ball speed changes above 3 m/s number at least 18 per 1,000 seen frames (raw sims
21–43), w = 1 below (7-frame-smoothed sims 12–13). The first setting was chosen out of fold on
the raw fit sims; the second is a hedge above the out-of-fold choice on the smoothed fit sims
(w = 0.5), because this feed keeps raw-like contact counts (§5.3, §5.7). This match has 2.6 kicks
per 1,000 frames: the smooth setting, under which the owner differs from v11's on about 9% of
alive frames.

## 4. Learned components

Five LightGBM models, single-threaded and seeded, all trained on the sim only. The first is the
**takeover probability** (`models/takeover_v2.txt`, 300 trees of 15 leaves, rate 0.03).
- *Training data:* the five fit sim matches only. A row is a contact by the team that does not
  own the ball in the sim GT just before it (2,237 contacts, 51% takeovers); the label is whether
  the GT owner 5 frames later is the toucher's team. Nothing is trained on this match.
- *Features:* 25, relative to the toucher's team (team-symmetric): the contact's duration and
  closest distances, the ball's speeds, speed change and turn (with missing flags), the rhythm
  of the previous and next contacts, and, since v6, ten team-shape signals (widths, centroid and
  speed gaps, forward drift 1 s and 2 s after the contact).
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
- *Held-out:* P(dead) is right on 0.948 / 0.927 of gap frames (majority class 0.789 / 0.781),
  the owner on 0.813 / 0.872 of alive gap frames (last contact's team 0.72 / 0.69).

The fourth is the **boundary dead model** (`models/dead_hybrid_v1.txt`: 300 trees of 31 leaves,
rate 0.05), used only by the §3.6 refinement.
- *Training data:* the five fit matches, each rebuilt three ways with a realistic dead-time ball
  (jittered at rest; picked up and carried; lost) instead of DFL's frozen one — 856,206 rows,
  every 5th graded frame; the label is the sim GT's dead/alive.
- *Features:* the gap models' player features plus **causal** ball features only (present and
  speed up to the current frame): the future-speed columns (+1 s, +2 s) are dropped, so a
  boundary cannot be placed by reading the ball after the restart — v8's likely failure
  mechanism.
- *Tuning:* the window (± 75 frames) was picked nested out of fold (chosen on four fit matches,
  scored on the fifth, all five folds agreeing) inside the registered grid {50, 62, 75}; 75 is
  the grid's edge, recorded as such. Retraining is byte-identical (`make train-hybrid`).

The fifth is the **proximity owner model** (`models/proxowner_v1.txt`: 300 trees of 31 leaves,
rate 0.05, ≥ 50 rows per leaf), used only by the §3.8 blend.
- *Training data:* every 3rd alive frame of the five fit matches, from both teams' perspectives,
  each match in three versions: the raw sim ball and the ball smoothed over 7 and 25 frames, so
  the model does not depend on how crisp the ball track is. The label is the sim GT's owner.
- *Features:* 77, per team and team-symmetric: nearest and second-nearest distance to the ball
  now and up to 1 s either side, players within 2 m and 5 m, time since and until each team is
  next within 1.5 m / 3 m, distance to where the ball is heading, how well the nearest player's
  velocity matches the ball's, the home-minus-away distance gap over 1–6 s, and team shape.
- *Tuning:* the blend settings of §3.8, out of fold as above. Retraining is byte-identical
  (`make train-proxowner`).

Touch and dead-ball thresholds were fitted by coordinate descent on the fit matches only.

## 5. Validation protocol

**5.1 Rules we hold ourselves to.** No hand labels on this match, for any purpose. The public 5
minutes are a correctness check only: the tool prints *pass* (spellF1 ≥ 0.35, H/A not swapped) or
*fail*, never the number, and our scoring wrapper refuses to score a candidate on the public GT.
No rule or constant is tied to specific frames of this match.

*Training and tuning data.* Every model, threshold and decoder cost in the final configuration
was fitted or selected on degraded external open data (DFL/IDSSE), which the organisers called
permissible on the challenge repository's issue #16. The one link to this match is the
degradation recipe (§5.2): before the organisers' 3 Nov clarification it was calibrated to
label-free aggregate statistics of the provided tracking (ball-missing share, gap rates and
lengths, player track lengths); no labels, video or frame-level information went into it. To
remove that link on the ball side we re-fitted it to external automatic tracking only (Metrica
Sports Sample Games 1–2). The parameters agree within one grid step (q 1.0, f 0.6, s 0.5 vs 1.0,
0.7, 0.4), and on sims built from it v11 still beats v6 on all 7 matches (+0.018, p = 0.008;
`configs/sim_external.yaml`). The player side (track dropouts, g) is still calibrated to the
provided file: no external feed we hold has comparable tracker dropout. The selection does not
depend on it: with g at a quarter or five times the fitted value, v11 still wins on all 7 matches
(+0.022 / +0.019). Real-match signals were otherwise used only as automatic guards and to choose
between whole, externally tuned versions. The organisers confirmed that is allowed (3 Nov). The §3.8 blend adds one more
such link, disclosed here: its training smoothing widths (7, 25 frames) and the two kick-rate
regimes bracket the kick rate observed label-free on the provided tracking (2.6 per 1,000 seen
frames vs 21–43 on the raw sims). No parameter was fitted to this match, the 13-frame scoring
width appears nowhere in training or selection, and every learned part is trained on open data.
At run time the kick rate is an input statistic that picks one of two pre-registered settings.

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

*Acceptance:* the official baselines score on the sim roughly as here: smooth50 0.261 vs
0.243 (raw 0.103 vs 0.049; smooth125 0.192 vs 0.224).

**5.3 Selection.** Two sim matches (J03WQQ, J03WR9) are held out from all fitting. Intervals
are the official scorer's bootstrap (1,000 resamples):

| Held-out sim | J03WQQ [95% CI] | J03WR9 [95% CI] | mean | outcome |
|---|---|---|---|---|
| smooth50 baseline | 0.287 [0.259, 0.314] | 0.284 [0.251, 0.315] | 0.286 | |
| v1: hard rules | 0.522 [0.484, 0.560] | 0.560 [0.526, 0.594] | 0.541 | early upload |
| v3: contacts + takeover model + Viterbi | 0.559 [0.519, 0.600] | 0.609 [0.572, 0.645] | 0.584 | selected 31 Oct |
| v5: v3 + ball-free gap decoder | 0.585 [0.544, 0.623] | 0.633 [0.596, 0.671] | 0.609 | selected 1 Nov |
| v6: v5 + team shape at contacts | 0.610 [0.571, 0.649] | 0.644 [0.606, 0.682] | 0.627 | selected 1 Nov |
| v11: v6 + `D`-boundary refinement | 0.644 [0.603, 0.685] | 0.683 [0.645, 0.721] | 0.664 | selected 1 Nov (sign-off) |
| **v12-hedge: v11 + proximity owner blend (§3.8)** | **0.630** | **0.690** | **0.660** | **selected 5 Nov in the smoothed-ball worlds (§5.7)** |

Decision rule (PLAN.md §6.4, amended 31 Oct before v3 was selected): discard any candidate that
fails the validator, the bands or the public check; a challenger then replaces the current
choice only if a **paired per-match comparison** favours it (held-out matches always count, fit
matches where the challenger's learned parts are out of fold): a positive mean difference and a
one-sided sign test at p ≤ 0.05, else a tie, decided by fewer components, then consistency.

`make paired` runs this with every learned part cross-fitted. Each selection won all seven
matches (p ≈ 0.008): v3 over v1; v5 over v3 by +0.030; v6 over v5 by +0.014; v11 over v6 by
+0.029 (paired possession bootstrap P(Δ ≤ 0) < 0.001), 7/7 under each of three degradation stress
arms. v12-hedge ties v11 on the raw sims (+0.002, 5/7, p = 0.23; held-out 0.664 → 0.660) and was
selected in the worlds whose ball is smoothed like this feed's (§5.7): +0.036 over v11 on all
seven matches at 13 frames (p = 0.008; held-out 0.495 → 0.538), +0.012 on six at 7 frames. Its
worst case, the smooth setting forced on a raw-like feed, is −0.011 (1/7; −0.043 at w = 0.5,
which is why the hedge ships). The rule was applied in the smoothed worlds because this feed's
kick rate selects that regime; an independent agent reproduced every number before adoption.
v11 also tripped two label-free real-match checks that older candidates pass (§5.4, its one-way
sign-off).

*Ablations* (held-out mean, change on removal): the §3.8 blend −0.043 at 13-frame smoothing
(+0.004 raw); v11's boundary refinement −0.037; then from v6, back to the hard rules −0.086,
long-gap evidence −0.055, the gap decoder −0.029, team shape −0.018, out-of-bounds evidence
−0.007. Held-out matches never tune.

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

v11 passes every band (1 Nov) with the same spell structure as v6 — 325 spells, 8 under 1 s —
because it only moves `D` edges; its `D` share of graded frames is 0.445, inside the advisory
`D`-share band [0.35, 0.57] added 1 Nov (v6: 0.481). **v12-hedge (the final) passes every band**
(candidate and submission runs, 5 Nov): 352 spells, the same `D` share (no `D` or `X` frame
changes) and a different owner on 8,027 alive frames, about 9% (the smoothed sims: 7–8%).

- *Smart-Tagging consistency:* the share of alive frames in team-tagged phases whose owner is
  the team the phase implies is 0.742 for v6, 0.730 for v11 and 0.733 for v12-hedge (v1–v5
  0.726–0.746). The pooled rate tracks alive volume, so the judging statistic is the split: on
  the 18,930 frames v11 newly decodes alive the tags agree with its owner 0.600 (rejected v8:
  0.564). On the 3,641 frames where v12-hedge's owner differs from v11's under unambiguous tags,
  the tags side with v12-hedge 0.548 to 0.452: contested frames by construction, in runs, so a
  coin flip — no regression and no confirmation. It flags regressions and is not a target.
- *Set pieces:* 32% of the 57 set-piece tags have a v6 `D` ending within 2 s of the tag-aligned
  restart (chance ≈ 9%). **v11 scores 14% and its `D` ends move median 1.72 s earlier than v6's**
  — the one real-match regression, and why its adoption needed a sign-off: the anchor was fitted
  to v1's own `D`-end timing, and the mechanical cross-check (§5.6) finds v6's ends ~1.5 s late,
  so ends that move onto the true restarts leave a ±2 s window centred on the old error. The
  pre-registered one-way rule (old guard fails, repaired guards pass, mechanical explanation
  documented) was exercised with a written sign-off (`submissions/LOG.md`).
- *Public window:* **pass** for v6, v11 and v12-hedge, team sign right.

**5.5 Known sim biases.**
1. DFL possession flips more often than the mechanical GT: 25–37% of sim spells are under 2 s,
   vs about 20% inferred here; our candidates predict 0.81–0.85× the sim GT's spell count.
2. Sim tracks are cleaner near the ball, and the sim has no Smart Tagging.
3. The DFL data freezes the ball in dead time (21% of seen sim ball frames do not move, 0.08%
   here, `make airlock T=atd-stats`). The gap models read no ball kinematics; the boundary model
   reads causal ball features and is trained only on fit matches rebuilt with a realistic
   dead-time ball (§4).
4. No validation build contains a dead stretch with the ball visible (0 of 286 Metrica `D` runs
   see the ball through the run), while this match's ATD does. That case rests on the boundary
   model's synthetic dead-time ball (§4) and on the airlock `D` aggregates, not a scored bench.

**5.6 Mechanical cross-check (Metrica Game 3).** A second validation family, independent of DFL
and of the sim recipe: Metrica Sports Sample Game 3 (pinned, checksummed), whose event feed has a
frame-accurate clock, degraded like the sim. Against it v6's `D` ends are **+26 / +38 / +78
frames late at p25 / p50 / p75**, about 1.5 s at the median, matching the sim's held-out residual
(§3.7) and this match's label-free restart profile (`make airlock T=dend-stats`). v11's median
`D`-end move (−1.72 s) is the size of that lateness: the basis for the §5.4 one-way adoption.

**5.7 Robustness to ground-truth conventions and feed shifts (3–5 Nov, open data).** We tested
v11 under alternative labelling conventions and tracking conditions, then compared the final's
blend with it.

- *Ground-truth conventions.* A target labeller may place `D` starts, `D` ends or owner switches
  elsewhere than DFL does, or absorb short spells. So the outputs on the 8 primary matches (7 DFL
  sims, cross-fitted, plus Metrica Game 3) were re-scored against transformed ground truths, one
  axis at a time. With H↔A boundaries or `D` starts moved −10 to +25 frames, or spells under 1 s
  or 2 s folded into their neighbours, v11 beats v6 on all 8 matches in each of the 10 worlds
  (+0.023 to +0.035) and on Metrica Games 1–2. Moving `D` ends later erodes the lead: +0.029 at
  0 frames, +0.012 at +20, +0.001 at +38 (about v6's measured lateness, §5.6), where v11 wins 5 of
  8 and loses on the Metrica builds (−0.017; −0.030 / −0.008). That stays above the −0.010 bar,
  but a late, whistle-timed `D`-end convention is the main residual exposure. v11's own score is
  highest at the untransformed ground truth on every axis tested, so no boundary shift is applied.
- *Degradation redraw floor.* Three more random degradations of the held-out matches move a
  configuration's absolute spellF1 by ±0.020 (SD) but a paired difference by only ±0.004, so a
  paired gain below +0.008 is a tie.
- *Feed shifts, paired.* The committed sims were edited so the random gap draw stays fixed and
  only the tested property changes (cross-fitted, noise ≈ 0.006): a ±10% change of the match clock
  costs ≤ 0.007; hiding one track per team inside long gaps 0.001; hiding 10% of all tracklets
  0.044, through contacts lost when the player at the ball is untracked. Two label-free
  comparisons with this match's feed (`make airlock T=atd-stats`): nobody is near the ball more
  often here (no player within 1 m on 52.6% of visible
  carry-speed ball frames, vs 30.5–35.4% on the sims), so the carrier-loss sensitivity applies.
  And the real ball track is velocity-filtered: 2.6 kicks (speed change > 3 m/s) per 1,000 seen
  frames, vs 21–43 on the sims. On the two held-out matches, smoothing the ball over 13 frames
  cost v11 0.165 of mean spellF1, while a per-frame proximity and team-shape model scored almost
  the same before and after: smoothing hides the short contacts v11 reads the owner from.
- *Owner blend.* v11's contact-based owner decoding was the exposed part, so the final blends a
  per-frame proximity owner into it (§3.8), trained at widths 0, 7 and 25 only; 13 frames, the
  width that matches this feed, was reserved for scoring (results in §5.3). This feed is mixed:
  its ball kinematics are 13-frame-smooth, but the ball hugs the carrier and contacts survive
  (8.9 per 1,000 frames vs 10.4–11 raw and 8.4–8.6 at 13 frames), so the smoothed worlds likely
  overstate both v11's loss and the blend's gain; the expected change here lies between −0.011
  and +0.036. Under the four convention axes above, on the raw sims, the hedge's difference to
  v11 runs from −0.005 (late `D` ends) to +0.012 (short spells merged): no world fails the bar.

## 6. Limitations and next steps

- **Changes inside ball gaps:** the gap decoder recovers part of them from the players alone
  (+0.03 over v3, against a +0.10 ceiling with the true labels inside long gaps); part of the
  sim's alive-inside-gaps mass is an artefact of its pooled gap-length histogram.
- **`D` is over-predicted:** 44.5% of graded frames for the final and v11 (v6 48.1%, v3 50.8%);
  v6 has 42–45% vs 41% true on the held-out sim.
- **The adoption risks we accepted:** (v11) the two v1-era real-match checks it fails (§5.4)
  are anchored to the older candidates' own `D`-end timing, which the mechanical evidence (§5.6)
  says was late — on a *different match*; if this match's tags are right where the Metrica
  transfer is wrong, v11 over-corrects restarts by ~1 s. (v12-hedge) its gain is measured in
  worlds smoothed to this feed's kick rate while its contact counts are raw-like, where the
  blend is a tie and the smooth setting costs up to 0.011 (§5.3); the tags cannot separate the
  two (§5.4).
- **Restarts:** fouls and offsides with the ball still tracked produce no evidence.
- **Semantics:** the labels use DFL's definition of possession. Across the eight primary matches
  in §5.7, v11 leads v6 on average under every convention tested, but with `D` ends placed about
  1.5 s later the lead is only +0.001 (five of eight); the final's blend changes v11's result by
  −0.005 to +0.012 across those worlds. Here, the bands and consistency checks are only indirect.
- **Ball kinematics:** the sim matches this feed's gaps, visibility and tempo but not its ball
  kinematics; the real ball track is about ten times smoother. On sims smoothed like it the
  contact model alone loses 0.17 and the §3.8 blend recovers 0.04 of that (§5.7). Expect the
  absolute score here to sit below the held-out sim estimate.
- **Phase 2:** the pipeline is match-agnostic (matches in `games.py`, thresholds in `configs/`).

## 7. Reproduction

```text
make setup                          # Python 3.11.14 venv, pinned requirements.txt
make test                           # no match data needed
export MTAH_DATA_DIR=/path/to/bundle
make cache                          # parse the ATD into an npz cache outside the repo
make submission                     # validate, check bands, and record the CSV hash
# optional, to retrain and re-check on the sim:
make open-data && make sim
make train-takeover CFG=configs/rules_v6.yaml && make train-gaps
make train-hybrid                   # the v11 boundary dead model (byte-identical retrain)
make train-proxowner                # the v12 proximity owner model (byte-identical retrain)
make sim-ablate
make paired B=configs/rules_v12_hedge.yaml A=configs/rules_v11.yaml
python scripts/build_smoothed_sims.py --width 13 --out DIR   # §5.7 world; MTAH_SIM_DIR=DIR
```

- **Submitted file:** sha256
  `961e800ef78f189a0a7cb832837d06e602d460b03cd59b255368f73b0517d65d`
  (`submissions/final/Berkeley Sports Research Group.sha256`; FINAL row of `submissions/LOG.md`).
- **Determinism:** CPU only, single-threaded LightGBM, `PYTHONHASHSEED=0`. Two independent
  runs reproduce the committed sha256 (5 Nov); the cold-cache and Docker (linux/arm64 and amd64,
  from a mounted bundle) reproductions of 4 Nov were of v11 and are repeated on the final at the
  freeze.
- **Tooling:** the code was written with an AI coding assistant that never received match data,
  video, raw tracking lines or per-frame labels: real-data runs released only aggregate
  statistics and pass/fail verdicts, filtered by a mechanical policy and logged
  (`src/mtah/airlock.py`). The gated bundle and every derived file stay out of the repository.

<!--
TEAM, before uploading:
- The repo URL is filled by `make doc REPO=https://github.com/Jus973/mtah-2027`.
- Measure the cold ATD parse time (`time make cache` on an empty cache) for §7 if you want it.
- After the freeze drill (cold cache ×2 + Docker arm64/amd64 on v12-hedge), rewrite §7
  "Determinism" to say they reproduce the committed sha256.
- The §5.3 v12-hedge row has no bootstrap interval: add one if `make score` is run on the
  held-out sims before the upload.
- §7 "Tooling": keep, edit or drop once the organisers answer email 1 (cloud assistants and
  aggregates).
-->
