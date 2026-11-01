<!--
Source of the technical document (PLAN.md §8). `make doc TEAM=<TEAM_NAME>` writes
docs/<TEAM_NAME>.md (and docs/<TEAM_NAME>.pdf when pandoc is installed). HTML comments like this
one are notes for the team and do not render. Every number below comes from a make target;
real-match numbers are label-free aggregates or verdicts (make airlock T=...).
DRAFT for the early upload of candidate v1 (31 Oct 2026). Update at freeze (M4).
-->

# Ball possession from automatic tracking: <TEAM_NAME>

More Than A Hack 2027, Phase 1: FC Barcelona (H) vs Athletic Club (A). Draft of 31 Oct 2026,
submitted with candidate **v1**. Code: <REPO_URL>.

## 1. Approach

We label possession from **events**, not by smoothing a per-frame guess. A touch (the ball
comes within 1 m of a player, confirmed by its motion) can move possession to the toucher's team.
Dead-ball evidence (the ball beyond a line, or lost by the tracker for at least a second) starts
a `D` stretch, and the next touch ends it. Every possession change sits on the first frame of
the touch that causes it. That is how the ground truth is defined, so latency is zero by
construction rather than fitted. All thresholds were fitted on *other* matches: open DFL data
degraded to look like this match's tracking (§5). On this match we use only label-free checks
and a pass/fail bug check on the public window.

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
the official smoothing baseline levels off near 0.24 on the full match. Events produce both
short possessions and long clean spells without that trade-off.

## 2. Assets

| Asset | Used in v1 | How |
|---|---|---|
| Tactical ATD (`_atd.xml/.txt`) | yes | Every component. Our streaming parser was cross-checked: the four official baselines regenerate byte-for-byte, and arrays equal kloppy 3.18's on a synthetic match in the same format. |
| Smart Tagging | yes | (a) The two `Kick Off (Start)` windows bound the search for each half's kick-off. (b) A label-free consistency check (§5.4). It sets no label directly. |
| `games.py` metadata | yes | Graded half ranges (`X` is written only outside them) and pitch size. |
| Homography | no | The ATD is already in pitch coordinates. |
| Tactical / broadcast video, audio, broadcast ATD | no | Stream metadata probed only (both videos carry audio). Whistle cues are planned (§6). |
| DFL/IDSSE open data | yes | Simulation, tuning and selection (§5). |

**Kick-offs.** Each half's kick-off is the frame where both teams' line-up (≥ 80% of each team in
its own half for ≥ 1 s) breaks, with the ball near the centre, inside the tag window. It lands
**4 frames (0.16 s) after the published frame in both halves**, within our 1 s acceptance
tolerance. We did not correct the 4 frames, since that would be a constant fitted to this match.
The published frames serve only as a check. Before the derived second-half kick-off, the graded
block is `D`.

**Tried and dropped, with the measured reason:**
- Kick-off as "ball at rest on the centre spot": the tracked ball never rests there (closest
  approach 1.4 m and 3.3 m).
- Per-frame nearest-player + smoothing (official baseline): 0.29 held-out sim spellF1, vs 0.54
  for v1.
- Owner rules H2 and H3 (§3.3): worse on the sim and on the real-match checks.
- Stillness as dead-ball evidence (ball and players calm): no gain on the sim, so it is tuned to
  effectively off.

**External data.** Bassek, Rein, Weber et al., *An integrated dataset of spatiotemporal and event
data in elite soccer*, Scientific Data 12, 195 (2025), CC BY 4.0: positions at 25 Hz from seven
professional German league matches (DFL), with per-frame ball status and possession. Those
labels are human-produced data about *other* games, which the rules allow. We hand-labelled
no frames, of this or any other game.

## 3. Possession rules (`configs/rules.yaml`)

**3.1 Ball cleaning.** The ATD does not clamp the ball to the pitch: it reaches 33 m beyond a
touchline. We drop positions more than 10 m outside the pitch and one-frame steps faster than
40 m/s, then fill gaps of ≤ 10 frames linearly. An observed ball just beyond a line is kept as
dead-ball evidence.

**3.2 Touches.** For each frame we take each team's nearest player to the ball. A *contact
episode* is a run of frames with the ball within 1.0 m of a player and the same nearest team
(gaps ≤ 2 frames bridged). An episode counts as a **touch** if the ball stays close ≥ 12
frames, or turns ≥ 20°, or changes speed by ≥ 12 m/s (velocity measured 3 frames before and
after). That rejects balls that merely fly past a player. On this match v1 finds 1,069 touches.

**3.3 Owner rule (H1).** Any touch by the other team flips possession, on the touch's first
frame. A ball the tracker loses holds the current owner (no touch seen, no change), so a loose
ball stays with the last owner. We compared three rules, all tuned the same way:
- H1: any touch flips;
- H2: only a *controlled* touch flips (the ball stays close, slows down, or the same team
  touches it next);
- H3: two consecutive touches by the other team flip.

H1 won on the held-out sim (§5.3). H2 was meant to ignore deflections, as the ground truth does.
On the sim its tuned control tests went to their permissive ends, and on the real match it did
worse on every label-free check. H1 does count a deflection as a change, which is a known
mismatch (§6).

**3.4 Dead ball.** Two kinds of evidence:
- *out of bounds*: the observed ball more than 0.5 m beyond a touch or goal line for ≥ 15
  frames;
- *long gap*: the ball missing for ≥ 25 frames.

On this match, ball gaps are bimodal (16 under 26 frames, 133 of 101 frames or more, none in
between), and players move half as fast while the ball is missing (median 1.00 vs 1.93 m/s). So
the tracker loses the ball mostly in stoppages. Evidence runs within 50 frames merge. Each merged
run is `D` from its first frame up to the **restart**, the first touch after the evidence ends.
After a `D` stretch, the restart touch's team owns the ball.

**3.5 Not yet in v1.** A sparse-transition decoder (Viterbi with calibrated change
probabilities) instead of hard rules. Latency residuals measured on the sim. Restarts anchored
on Smart-Tagging set-piece windows.

## 4. Learned components

None in v1: no model is trained, on this match or any other. About a dozen thresholds (touch:
radius, duration, turn, speed change; dead ball: margins, durations, merge window) were fitted by
coordinate descent over fixed grids. The objective was mean spellF1 with the official scorer on
the five *fit* sim matches. Grids were widened until every tuned value sat inside them, except
where a criterion tuned itself off (stillness; H2's control tests).

**Candidate v3: learned takeovers and a sparse-transition decoder** (`configs/rules_v3.yaml`,
not yet the submitted file). With the true `D`, H1 finds only half of the sim's possession
changes, and 38% of the changes it predicts are wrong. Perfect takeover decisions on all
contacts would lift spellF1 from 0.58 to 0.76 on the fit matches. That is a much larger lever
than `D`, which is worth at most +0.04.

v3 keeps v1's touches and dead-ball rules and changes how possession moves between contacts:
- **Candidates.** Every contact (ball within 1.0 m of a player) is a candidate, not only the
  kinematically confirmed ones.
- **Takeover model.** LightGBM gives a contact by the non-owning team its probability of taking
  the ball. It is single-threaded, seeded and trained on the five fit sim matches only. Its 15
  team-symmetric features cover the contact's kinematics and its rhythm (time to the previous
  and next contacts, and whether they are by the same team). The training label is whether the
  sim GT owner 5 frames after the contact is the toucher's team.
- **Decoder.** A two-state Viterbi over contacts. A takeover scores log p, a contact that fails
  to take the ball scores log(1 − p), and p is sharpened by α = 3.0 on the log-odds scale. A
  change still lands on the contact's first frame.
- **Tuning.** α and the flip penalty (0) were tuned on out-of-fold predictions for the fit
  matches.

On held-out contacts the model has AUC 0.913. End to end with v1's detected `D`, held-out
spellF1 is 0.584 (J03WQQ 0.559 [0.519, 0.600]; J03WR9 0.609 [0.572, 0.645]) vs 0.541 for v1.
v3 beats v1 on all seven sim matches, by +0.016 to +0.051. On the five fit matches that holds
even with v3 out of fold and v1 in-sample.

On this match, v3 passes the validator, all bands and the public check. Smart-Tagging
consistency is unchanged (0.740). It predicts fewer, longer spells: 312, with 18% under 2 s and
51% under 6 s (v1: 354, 23%, 55%). That is closer to the ≈ 20% / ≈ 50% inferred from the
official lag figures. Pressure features (players within 3 / 8 m) were tried and dropped: they
left the held-out AUC unchanged and depend on track dropouts and duplicates.

## 5. Validation protocol

**5.1 Rules we hold ourselves to.** No hand labels on this match, for any purpose. The public
5 minutes are a correctness check only: the tool prints *pass* (spellF1 ≥ 0.35, H/A not swapped)
or *fail*, never the number, and our scoring wrapper refuses to score a candidate on the public
GT. No rule or constant is tied to specific frames of this match.

**5.2 Simulation on open data (primary).** Each DFL match is rewritten in the bundle's ATD
format. Its ground truth is `D` where DFL marks the ball dead, else the team in possession. It is
then degraded to match statistics we measured on the real ATD without labels:
- *stoppage loss*: the tracker loses the ball in each dead stretch with probability q, for a
  fraction f of it;
- *open-play gaps*: started at the real onset rate by ball speed × crowding × near-a-line,
  scaled by s, with lengths from the real gap-length histogram;
- *player tracks*: cut into tracklets with the real length distribution, separated by short
  invisible gaps (mean g frames) and dealt into 24 / 22 slots as in the real ATD, plus
  team-wide dropouts and duplicate tracks at the real rates.

q, f, s and g were fitted on the five fit matches (q = 1.0, mean f = 0.7, s = 0.4, g = 100):

| Statistic (graded frames) | sim (fit) | real ATD |
|---|---|---|
| ball missing | 41.5% | 40.7% |
| ball gaps per 1,000 frames | 0.89 | 0.94 |
| median gap | 342 frames | 268 frames |
| player speed, ball missing / seen | 0.59 | 0.52 |
| P(11 tracks \| 7–11 tracks) | 0.63 | 0.65 |

*Acceptance test* (official baseline logic, official scorer). The sim is trusted only if the
baselines score on it roughly as they do on the real match (smooth50 within 0.24 ± 0.06). It
passes:

| spellF1 | raw | smooth25 | smooth50 | smooth125 |
|---|---|---|---|---|
| sim, mean of 7 matches | 0.103 | 0.258 | 0.261 | 0.192 |
| real match (official figures) | 0.049 | 0.200 | 0.243 | 0.224 |

**5.3 Selection on the held-out sim.** Two matches (J03WQQ, J03WR9) were never used for fitting.
95% intervals come from the official scorer's bootstrap (1,000 resamples) per match.

| Held-out sim | J03WQQ spellF1 [95% CI] | J03WR9 spellF1 [95% CI] | mean | macroF1 |
|---|---|---|---|---|
| official baseline smooth50 | 0.287 [0.259, 0.314] | 0.284 [0.251, 0.315] | 0.286 | 0.642 |
| **v1** (H1 + detected `D`) | 0.522 [0.484, 0.560] | 0.560 [0.526, 0.594] | **0.541** | 0.833 |
| H2 + detected `D` | – | – | 0.522 | – |

Owner rules alone, with the true `D` from the sim GT: H1 0.606, H2 0.585, H3 0.504 (macroF1
0.913 / 0.907 / 0.895), against 0.403 for smooth50 + true `D`. Dead-ball ablation (H1): gaps only
0.534; out of bounds + stillness, no gaps, 0.454; all evidence 0.541. Our decision rule:
1. discard any candidate that fails the validator, the structural bands or the public check;
2. rank by held-out sim spellF1;
3. break near-ties by fewer components, then Smart-Tagging consistency, then sim macroF1.

**5.4 Label-free checks on the real match.** *Structural bands* were set in PLAN.md before any
candidate existed, from published facts. They filter out catastrophic errors and never rank
passing candidates.

| Band (graded frames) | allowed | v1 | H2 variant |
|---|---|---|---|
| spells | 280–550 | 354 | 421 |
| spells < 2 s | 10–30% | 22.9% | in band |
| spells < 6 s | 35–60% | 55.1% | **out** |
| mean spell | 7–16 s | 8.8 s | in band |
| `D` runs | 60–150 | 133 | in band |
| home share of alive frames | 0.55–0.72 | 0.61 | in band |
| `X` on graded frames | ≤ 2% | 0 | 0 |
| kick-off team right, both halves | yes | yes | **no** |
| `D` before the 2nd-half kick-off | yes | yes | yes |

*Smart-Tagging consistency.* Team-tagged phases imply which team has the ball: the tagged team
for attacking phases, its opponent for defending ones. We report the share of tagged,
predicted-alive frames whose predicted owner agrees. Set pieces and goals are excluded. Phases
are not possessions and the tagger errs, so this flags regressions rather than serving as a
target. **v1: 0.740; H2: 0.714.** By phase, v1 agrees most in build-up (0.90) and progression
(0.85), and least in attacking transitions (0.64) and players-in-the-box (0.61). The tagger
emits each phase twice over the same window: an attacking phase for one team and the matching
defending phase for the other (e.g. build-up / defending in the attacking third). v1's agreement
is identical within every pair, which confirms our attack/defend mapping.

*Public window:* v1 **passes** the bug check, with the team sign right.

**5.5 Known sim biases.** (1) DFL possession flips more often than the mechanical GT: 25–37% of
sim spells are under 2 s vs ~20% inferred for this match, which favours fast-reacting rules such
as H1. Differences in time-scale parameters transfer coarsely; event detection should transfer
better. (2) Sim tracks are cleaner near the ball: raw nearest-player scores 0.10 on the sim vs
0.05 real. (3) The real ball is lost in stoppages even more than q = 1 reproduces. (4) The sim
has no Smart Tagging yet. Touch statistics do transfer: the ball is within 1.0 m of a player on
28.7% of real ball-seen frames vs 28.0–29.7% on the sim.

## 6. Limitations and next steps

- **`D` is over-predicted.** 50.8% of graded frames are `D` on this match, and on the sim the
  detector gives 51–52% vs 41% true (precision 0.79, recall 0.98). We tried four fixes, tuned on
  the fit matches:
  - gap evidence only while players are calm (in-play gaps: median 2.3 m/s, stoppage gaps 1.2);
  - a stoppage ends when the ball leaves the taker of a throw-in or corner;
  - a stoppage ends when the ball reappears in flight;
  - a kick from rest counts as a touch.

  Together they cut the sim `D` share to 48.6% and raised precision to 0.83, but held-out
  spellF1 moved only +0.004 (0.545), inside the noise. On this match `D` fell to 46.0%, while
  Smart-Tagging consistency fell slightly (0.728). Under our decision rule that is a tie, which
  goes to the simpler v1. A partial-oracle study on the fit matches bounds what any `D` work
  can gain: perfect `D` adds +0.04, mostly from boundary timing (onsets +0.015, restarts
  +0.017). Removing spurious stoppages adds only +0.008.
- **Deflections and unwon aerials** flip possession under H1. The ground truth does not count
  them, but every stricter rule we tried did worse.
- **Short possessions inside ball gaps** are invisible: a gap holds the owner.
- **Scrambles in the box and transitions** are where v1 and the tagger disagree most (§5.4).
- **Stoppages with the ball still tracked and moving** (fouls, offside) produce no evidence.
  Next: whistle cues from the tactical audio.
- **Decoder.** v3 decodes possession over contacts (§4). Next: let the same path decode `D`
  from the dead-ball evidence instead of taking it as fixed. Also add contacts the 1 m radius
  misses: two-thirds of the possession changes it misses have no contact nearby.
- **Sim semantics.** The takeover labels come from DFL possession, which changes more often
  than this match's mechanical GT. v3 predicts 20% fewer spells than the sim GT, and its
  real-match structure sits inside the bands. A semantic mismatch would show up as a systematic
  bias the sim cannot measure.
- **Phase 2.** The pipeline is match-agnostic: matches are entries in `games.py`, thresholds
  live in `configs/`, and the sim builder works for any open dataset with ball status and
  possession.

## 7. Reproduction

```bash
make setup                          # Python 3.11.14 venv, pinned requirements.txt
make test                           # 65 tests, no match data needed
export MTAH_DATA_DIR=/path/to/bundle
make cache                          # parse the ATD into an npz cache outside the repo
make candidate NAME=v1 CFG=configs/rules.yaml   # -> submissions/candidates/v1.csv
# optional, for the held-out sim score it prints:
make open-data && make sim          # DFL/IDSSE (2.63 GB, CC BY 4.0) -> ~/opendata/sim
make train-takeover                 # v3 only: retrain models/takeover_v1.txt (committed) from the sim
```

- **Determinism.** v1 has no randomness: it is pure rules on fixed inputs, run on the CPU. The
  sim's degradation is seeded per match. Re-running the candidate target on the same commit
  produces the same file, which the target itself checks against `submissions/LOG.md`.
- **Runtime.** About 4 s wall time for `make candidate` with the parse cache warm, on an Apple
  M4 (10 cores, 16 GB). The one-off ATD parse comes on top.
- **Submitted file.** `<TEAM_NAME>.csv` = candidate v1: commit `4cbeed1`, config hash
  `f96f2f64a285`, sha256 `3c32c4758d117a6a…`.
- **Tooling.** Code was written with an AI coding assistant. It never received match data,
  video, raw tracking lines or per-frame labels. Real-data runs released only aggregate
  statistics and pass/fail verdicts, filtered by a mechanical policy and logged
  (`src/mtah/airlock.py`). The gated bundle and every derived file stay out of the repository.

<!--
TEAM, before uploading:
- Replace <REPO_URL> (needs a public repo) and run `make doc TEAM=<TEAM_NAME>`.
- Paste the full sha256 from `shasum -a 256 <TEAM_NAME>.csv` into §7.
- Measure the cold ATD parse time (`time make cache` on an empty cache) for §7.
- §7 "Tooling": keep, edit or drop once the organisers answer email 1 (cloud assistants and
  aggregates).
-->
